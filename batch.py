"""Concurrent batch extraction over a directory of clinical notes."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from langfuse import get_client

from extractor import extract
from pipeline import CHAT_MODEL, EXTRACT_FIELD_NAMES
from schema import ExtractionOutput

load_dotenv()

DEFAULT_INPUT_DIR = Path("data/synthetic")
DEFAULT_OUT_DIR = Path("data/batch")

from cost import estimate_cost_usd


@dataclass
class BatchRow:
    example_id: str
    success: bool
    latency_ms: float
    needs_review: list[str]
    flagged_field_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


def load_notes(input_dir: Path, limit: Optional[int] = None) -> list[tuple[str, str]]:
    """Load (example_id, note text) from JSON or plain-text files."""
    pairs: list[tuple[str, str]] = []
    paths = sorted(input_dir.iterdir())
    for path in paths:
        if path.name == "manifest.json":
            continue
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            note = data.get("note") or data.get("text")
            if note:
                pairs.append((path.stem, note))
        elif path.suffix == ".txt":
            pairs.append((path.stem, path.read_text(encoding="utf-8")))
    if limit is not None:
        pairs = pairs[:limit]
    return pairs


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def process_one(
    example_id: str,
    note: str,
    *,
    review_threshold: Optional[float],
    extract_fn: Callable[..., ExtractionOutput],
) -> BatchRow:
    started = time.perf_counter()
    try:
        output = extract_fn(note, review_threshold=review_threshold)
        latency_ms = (time.perf_counter() - started) * 1000.0
        flagged = sum(1 for meta in output.fields.values() if meta.needs_review)
        cost = estimate_cost_usd(output.usage.prompt_tokens, output.usage.completion_tokens)
        return BatchRow(
            example_id=example_id,
            success=True,
            latency_ms=latency_ms,
            needs_review=list(output.needs_review),
            flagged_field_count=flagged,
            prompt_tokens=output.usage.prompt_tokens,
            completion_tokens=output.usage.completion_tokens,
            total_tokens=output.usage.total_tokens,
            estimated_cost_usd=cost,
            result=output.model_dump(mode="json"),
        )
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000.0
        return BatchRow(
            example_id=example_id,
            success=False,
            latency_ms=latency_ms,
            needs_review=[],
            flagged_field_count=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            error=str(exc),
        )


def build_summary(
    rows: list[BatchRow],
    *,
    input_dir: Path,
    output_jsonl: Path,
    summary_path: Path,
    workers: int,
    wall_time_sec: float,
    review_threshold: Optional[float],
    model: str,
) -> dict[str, Any]:
    successes = [row for row in rows if row.success]
    latencies = [row.latency_ms for row in successes]
    prompt_tokens = sum(row.prompt_tokens for row in successes)
    completion_tokens = sum(row.completion_tokens for row in successes)
    total_tokens = sum(row.total_tokens for row in successes)
    total_cost = sum(row.estimated_cost_usd for row in successes)
    notes_needing_review = sum(1 for row in successes if row.needs_review)
    flagged_fields = sum(row.flagged_field_count for row in successes)
    field_slots = 0
    for row in successes:
        if not row.result:
            continue
        extract_payload = row.result.get("extract", {})
        for name in EXTRACT_FIELD_NAMES:
            value = extract_payload.get(name)
            if value not in (None, "", []):
                field_slots += 1

    pct_notes_needs_review = round(100.0 * notes_needing_review / len(successes), 2) if successes else 0.0
    pct_fields_flagged = round(100.0 * flagged_fields / field_slots, 2) if field_slots else 0.0

    return {
        "run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "input_dir": str(input_dir),
        "output_jsonl": str(output_jsonl),
        "summary_path": str(summary_path),
        "model": model,
        "workers": workers,
        "review_threshold": review_threshold,
        "count": len(rows),
        "success_count": len(successes),
        "error_count": len(rows) - len(successes),
        "wall_time_sec": round(wall_time_sec, 3),
        "throughput_notes_per_sec": round(len(successes) / wall_time_sec, 3) if wall_time_sec > 0 else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
        },
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "estimated_cost_usd": round(total_cost, 4),
        "pct_notes_needs_review": pct_notes_needs_review,
        "pct_fields_flagged": pct_fields_flagged,
        "errors": [
            {"example_id": row.example_id, "error": row.error}
            for row in rows
            if not row.success
        ],
    }


def run_batch(
    input_dir: Path,
    out_dir: Path,
    *,
    workers: int = 4,
    limit: Optional[int] = None,
    review_threshold: Optional[float] = None,
    extract_fn: Callable[..., ExtractionOutput] = extract,
    model: str = CHAT_MODEL,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes = load_notes(input_dir, limit=limit)
    if not notes:
        raise FileNotFoundError(f"No notes found in {input_dir}")

    output_jsonl = out_dir / "results.jsonl"
    summary_path = out_dir / "run_summary.json"
    rows: list[BatchRow] = []
    wall_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=workers) as pool, output_jsonl.open("w", encoding="utf-8") as out_f:
        futures = {
            pool.submit(
                process_one,
                example_id,
                note,
                review_threshold=review_threshold,
                extract_fn=extract_fn,
            ): example_id
            for example_id, note in notes
        }
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            out_f.write(
                json.dumps(
                    {
                        "example_id": row.example_id,
                        "success": row.success,
                        "latency_ms": round(row.latency_ms, 2),
                        "needs_review": row.needs_review,
                        "flagged_field_count": row.flagged_field_count,
                        "usage": {
                            "prompt_tokens": row.prompt_tokens,
                            "completion_tokens": row.completion_tokens,
                            "total_tokens": row.total_tokens,
                        },
                        "estimated_cost_usd": row.estimated_cost_usd,
                        "result": row.result,
                        "error": row.error,
                    }
                )
                + "\n"
            )
            out_f.flush()

    wall_time_sec = time.perf_counter() - wall_start
    rows.sort(key=lambda row: row.example_id)
    summary = build_summary(
        rows,
        input_dir=input_dir,
        output_jsonl=output_jsonl,
        summary_path=summary_path,
        workers=workers,
        wall_time_sec=wall_time_sec,
        review_threshold=review_threshold,
        model=model,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    get_client().flush()
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(f"Batch run {summary['run_id']}")
    print(f"  notes: {summary['success_count']}/{summary['count']} succeeded")
    print(f"  throughput: {summary['throughput_notes_per_sec']} notes/sec")
    print(
        "  latency (ms): "
        f"p50={summary['latency_ms']['p50']}, "
        f"p95={summary['latency_ms']['p95']}, "
        f"mean={summary['latency_ms']['mean']}"
    )
    print(
        f"  tokens: {summary['tokens']['total']} "
        f"(prompt={summary['tokens']['prompt']}, completion={summary['tokens']['completion']})"
    )
    print(f"  estimated cost: ${summary['estimated_cost_usd']:.4f}")
    print(
        f"  needs_review: {summary['pct_notes_needs_review']}% of notes, "
        f"{summary['pct_fields_flagged']}% of populated fields flagged"
    )
    print(f"  results → {summary['output_jsonl']}")
    print(f"  summary → {summary['summary_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-extract oncology variables from a note directory.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=int(os.getenv("BATCH_WORKERS", "4")))
    parser.add_argument("--limit", type=int, default=None, help="process only the first N notes")
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=None,
        help="confidence cutoff for needs_review (defaults to REVIEW_CONFIDENCE_THRESHOLD env)",
    )
    args = parser.parse_args()

    summary = run_batch(
        args.input_dir,
        args.out_dir,
        workers=max(1, args.workers),
        limit=args.limit,
        review_threshold=args.review_threshold,
    )
    print_summary(summary)
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
