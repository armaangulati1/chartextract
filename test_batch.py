import json
from pathlib import Path

from schema import ExtractionOutput, FieldMeta, OncologyExtract, TokenUsage

from batch import (
    build_summary,
    load_notes,
    percentile,
    process_one,
    run_batch,
)
from cost import estimate_cost_usd


def _fake_extract(note: str, review_threshold=None) -> ExtractionOutput:
    flagged = "uncertain" in note.lower()
    fields = {
        "primary_site": FieldMeta(
            confidence=0.5 if flagged else 0.95,
            needs_review=flagged,
            source="test",
        )
    }
    return ExtractionOutput(
        extract=OncologyExtract(primary_site="lung" if flagged else "breast"),
        fields=fields,
        needs_review=["primary_site"] if flagged else [],
        review_threshold=review_threshold or 0.75,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
    )


def test_load_notes_from_synthetic(tmp_path: Path):
    note_path = tmp_path / "0001.json"
    note_path.write_text(json.dumps({"note": "Oncology follow-up note."}))
    pairs = load_notes(tmp_path)
    assert pairs == [("0001", "Oncology follow-up note.")]


def test_percentile():
    assert percentile([10.0, 20.0, 30.0, 40.0], 50) == 25.0
    assert percentile([], 50) == 0.0


def test_estimate_cost_usd():
    cost = estimate_cost_usd(1_000_000, 500_000, input_cost_per_1m=0.15, output_cost_per_1m=0.60)
    assert cost == 0.45


def test_process_one_success():
    row = process_one("ex1", "clear note", review_threshold=0.75, extract_fn=_fake_extract)
    assert row.success
    assert row.example_id == "ex1"
    assert row.total_tokens == 150
    assert row.needs_review == []


def test_process_one_flagged():
    row = process_one("ex2", "uncertain note", review_threshold=0.75, extract_fn=_fake_extract)
    assert row.success
    assert row.needs_review == ["primary_site"]
    assert row.flagged_field_count == 1


def test_run_batch_writes_outputs(tmp_path: Path):
    input_dir = tmp_path / "input"
    out_dir = tmp_path / "output"
    input_dir.mkdir()
    for idx in ("0000", "0001"):
        (input_dir / f"{idx}.json").write_text(
            json.dumps({"note": "uncertain note" if idx == "0001" else "clear note"})
        )

    summary = run_batch(
        input_dir,
        out_dir,
        workers=2,
        extract_fn=_fake_extract,
    )

    assert summary["count"] == 2
    assert summary["success_count"] == 2
    assert summary["pct_notes_needs_review"] == 50.0
    assert (out_dir / "results.jsonl").exists()
    assert (out_dir / "run_summary.json").exists()

    rows = [json.loads(line) for line in (out_dir / "results.jsonl").read_text().splitlines() if line]
    assert len(rows) == 2
    assert summary["latency_ms"]["p50"] >= 0


def test_build_summary_metrics():
    from batch import BatchRow

    rows = [
        BatchRow("a", True, 100.0, [], 0, 100, 50, 150, 0.01, result={"extract": {"primary_site": "lung"}, "fields": {}}),
        BatchRow("b", True, 200.0, ["stage"], 1, 200, 100, 300, 0.02, result={"extract": {"primary_site": "lung", "stage": "II"}, "fields": {}}),
        BatchRow("c", False, 50.0, [], 0, 0, 0, 0, 0.0, error="boom"),
    ]
    summary = build_summary(
        rows,
        input_dir=Path("data/synthetic"),
        output_jsonl=Path("out/results.jsonl"),
        summary_path=Path("out/run_summary.json"),
        workers=2,
        wall_time_sec=2.0,
        review_threshold=0.75,
        model="gpt-4o-mini",
    )
    assert summary["count"] == 3
    assert summary["success_count"] == 2
    assert summary["error_count"] == 1
    assert summary["throughput_notes_per_sec"] == 1.0
    assert summary["tokens"]["total"] == 450
    assert summary["pct_notes_needs_review"] == 50.0
