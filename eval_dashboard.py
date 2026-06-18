"""Load latest eval metrics for the Streamlit dashboard."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

DEFAULT_METRICS_JSON = Path("data/eval/latest_metrics.json")
FALLBACK_RESULTS_MD = Path("data/eval/ci_out/results.md")


def _parse_pct(value: str) -> float | str:
    value = value.strip()
    if not value:
        return ""
    if value.endswith("%"):
        return round(float(value.rstrip("%")) / 100.0, 4)
    return value


def parse_results_md_table(path: Path) -> list[dict[str, Any]]:
    """Parse the per-field metrics markdown table from eval results.md."""
    if not path.exists():
        return []

    text = path.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    in_table = False

    for line in text.splitlines():
        if line.startswith("| field |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            break
        if re.match(r"^\|[-:| ]+\|$", line):
            continue

        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) != 7:
            continue

        field, tp, fp, fn, precision, recall, f1 = parts
        row: dict[str, Any] = {
            "field": field,
            "tp": int(tp) if tp.isdigit() else tp,
            "fp": int(fp) if fp.isdigit() else fp,
            "fn": int(fn) if fn.isdigit() else fn,
            "precision": _parse_pct(precision),
            "recall": _parse_pct(recall),
            "f1": _parse_pct(f1),
        }
        rows.append(row)

    return rows


def load_eval_metrics(
    metrics_json: Path = DEFAULT_METRICS_JSON,
    fallback_md: Path = FALLBACK_RESULTS_MD,
) -> Optional[dict[str, Any]]:
    """Return the latest eval metrics payload for dashboard rendering."""
    if metrics_json.exists():
        payload = json.loads(metrics_json.read_text(encoding="utf-8"))
        if payload.get("rows"):
            return payload

    rows = parse_results_md_table(fallback_md)
    if not rows:
        return None

    macro = next((row for row in rows if row["field"] == "macro_avg"), None)
    return {
        "source": str(fallback_md),
        "n_examples": None,
        "rows": rows,
        "macro_f1": macro["f1"] if macro else None,
        "updated_at": None,
    }


def format_metric_pct(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{100 * value:.1f}%"
    return str(value)
