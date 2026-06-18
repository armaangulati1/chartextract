"""Run eval on synthetic + real gold sets and write combined results.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval import (
    EvalSummary,
    evaluate_dataset,
    format_dataset_eval_section,
    format_pct,
    load_pairs,
    macro_f1_score,
    metrics_table,
)
from extractor import extract_record

SYNTHETIC_DIR = Path("data/eval/ci_gold")
REAL_DIR = Path("data/real")
DEFAULT_OUT = Path("data/eval/results.md")
DATASET_SECTION_MARKER = "## Dataset evaluations"


def _field_f1_rows(rows: list[dict]) -> dict[str, float]:
    fields = (
        "primary_site", "histology", "stage", "ecog_performance_status",
        "line_of_therapy", "date_of_diagnosis", "biomarkers", "treatment_regimen",
    )
    return {r["field"]: r["f1"] for r in rows if r["field"] in fields}


def run_eval_on_dir(
    data_dir: Path,
    cache_name: str,
    *,
    use_cache: bool,
) -> tuple[EvalSummary, list[dict], int]:
    pairs = load_pairs(data_dir)
    if not pairs:
        raise FileNotFoundError(f"No labeled pairs in {data_dir}")
    cache = Path("data/eval") / cache_name
    summary = evaluate_dataset(pairs, extract_record, cache_path=cache, use_cache=use_cache)
    return summary, metrics_table(summary), summary.n_examples


def _preserve_experiment_section(out_path: Path) -> str:
    if not out_path.exists():
        return ""
    text = out_path.read_text(encoding="utf-8")
    for marker in (DATASET_SECTION_MARKER, "## Real data (MTSamples)"):
        if marker in text:
            return text.split(marker)[0].rstrip()
    return text.rstrip()


def append_dataset_evaluations(
    out_path: Path,
    synth_summary: EvalSummary,
    synth_rows: list[dict],
    real_summary: EvalSummary,
    real_rows: list[dict],
) -> None:
    synth_n = synth_summary.n_examples
    real_n = real_summary.n_examples
    synth_macro = macro_f1_score(synth_rows)
    real_macro = macro_f1_score(real_rows)
    gap_pp = (real_macro - synth_macro) * 100

    synth_f1 = _field_f1_rows(synth_rows)
    real_f1 = _field_f1_rows(real_rows)
    fields = list(synth_f1.keys())

    lines = [
        _preserve_experiment_section(out_path),
        "",
        DATASET_SECTION_MARKER,
        "",
        "Production extract path: `extract_record()` → pipeline with verifier (`gpt-4o-mini`).",
        "",
        *format_dataset_eval_section(
            f"Synthetic — CI gold ({synth_n} notes)",
            synth_summary,
            synth_rows,
        ),
        *format_dataset_eval_section(
            f"Real — MTSamples ({real_n} notes)",
            real_summary,
            real_rows,
        ),
        "### Synthetic vs real (summary)",
        "",
        "| dataset | notes | macro-F1 | Δ vs synthetic |",
        "|---|---:|---:|---:|",
        f"| synthetic (CI gold) | {synth_n} | {format_pct(synth_macro)} | — |",
        f"| real (MTSamples) | {real_n} | {format_pct(real_macro)} | {gap_pp:+.1f} pp |",
        "",
        "#### Per-field F1 gap (real − synthetic)",
        "",
        "| field | synthetic | real | gap (pp) |",
        "|---|---:|---:|---:|",
    ]
    for field in fields:
        s = synth_f1.get(field, 0.0)
        r = real_f1.get(field, 0.0)
        lines.append(f"| {field} | {format_pct(s)} | {format_pct(r)} | {(r - s) * 100:+.1f} |")

    lines.extend([
        "",
        "#### Takeaway",
        "",
        _real_takeaway(synth_macro, real_macro, synth_f1, real_f1, real_n),
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _real_takeaway(
    synth_macro: float,
    real_macro: float,
    synth_f1: dict[str, float],
    real_f1: dict[str, float],
    real_n: int,
) -> str:
    gap = (real_macro - synth_macro) * 100
    hardest = sorted(real_f1, key=real_f1.get)[:3]
    synth_better = [f for f in real_f1 if synth_f1.get(f, 0) - real_f1.get(f, 0) > 0.15]

    s1 = (
        f"On **{real_n}** real MTSamples oncology notes, macro-F1 is **{format_pct(real_macro)}** "
        f"vs **{format_pct(synth_macro)}** on synthetic CI gold (**{gap:+.1f} pp gap**)—"
        f"{'expected degradation on messy real text' if gap < 0 else 'surprisingly close to synthetic'}."
    )
    s2 = (
        f"Weakest real fields: **{', '.join(hardest)}** "
        f"(many notes lack explicit stage/line/biomarkers, so null-vs-extract mismatches dominate). "
        f"Largest synthetic advantage: **{', '.join(synth_better) or 'none >15pp'}**."
    )
    s3 = (
        "Real notes are hematology-heavy consults with sparse structured oncology variables; "
        "improve primary_site/histology recall before trusting stage/regimen metrics on production charts."
    )
    return f"{s1}\n\n{s2}\n\n{s3}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--real-dir", type=Path, default=REAL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="score from cached predictions (no live API calls)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="run live extraction (ignores cache)",
    )
    args = parser.parse_args()
    use_cache = args.use_cache or not args.live

    print("Evaluating synthetic CI gold...")
    synth_summary, synth_rows, _ = run_eval_on_dir(
        args.synthetic_dir, "synthetic_ci_predictions.jsonl", use_cache=use_cache
    )
    print(f"  macro-F1: {format_pct(macro_f1_score(synth_rows))}")

    print("Evaluating real MTSamples gold...")
    real_summary, real_rows, _ = run_eval_on_dir(
        args.real_dir, "real_mtsamples_predictions.jsonl", use_cache=use_cache
    )
    print(f"  macro-F1: {format_pct(macro_f1_score(real_rows))}")

    append_dataset_evaluations(args.out, synth_summary, synth_rows, real_summary, real_rows)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
