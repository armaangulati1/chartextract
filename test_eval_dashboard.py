from pathlib import Path

from eval_dashboard import format_metric_pct, load_eval_metrics, parse_results_md_table


def test_parse_results_md_table(tmp_path: Path):
    md = """# Extraction evaluation results

## Per-field metrics

| field | TP | FP | FN | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| primary_site | 6 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| macro_avg |  |  |  | 93.8% | 95.8% | 94.7% |
"""
    path = tmp_path / "results.md"
    path.write_text(md)
    rows = parse_results_md_table(path)
    assert rows[0]["field"] == "primary_site"
    assert rows[0]["f1"] == 1.0
    assert rows[-1]["field"] == "macro_avg"


def test_load_eval_metrics_fallback_to_ci_out():
    payload = load_eval_metrics(metrics_json=Path("data/eval/does-not-exist.json"))
    assert payload is not None
    assert payload["rows"]
    assert payload["macro_f1"] is not None


def test_format_metric_pct():
    assert format_metric_pct(0.947) == "94.7%"
