"""Distribution drift check: compares a current extraction batch to a stored baseline.

Field-level drift on the structured outputs of the extractor: categorical field
distributions and per-field presence rates via PSI (Population Stability Index), and
continuous signals (list lengths, and per-field confidence when a batch carries it) via
a two-sample Kolmogorov-Smirnov test. Both statistics are implemented in the standard
library on purpose: the repo does not depend on scipy, so adding it just to compute two
well-defined statistics would be a heavier dependency than the task warrants.

PSI bands (industry-standard): < 0.10 stable, 0.10-0.25 moderate shift, >= 0.25 significant.
KS: drift when the two-sample p-value < alpha (default 0.05).

Nothing here calls an LLM. It reads recorded/synthetic extraction records already on disk.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

DEFAULT_OUT = Path("data/mlops/drift_report.json")

CATEGORICAL_FIELDS = (
    "primary_site",
    "histology",
    "stage",
    "ecog_performance_status",
    "line_of_therapy",
)
PRESENCE_FIELDS = (
    "primary_site",
    "histology",
    "stage",
    "ecog_performance_status",
    "line_of_therapy",
    "date_of_diagnosis",
    "biomarkers",
    "treatment_regimen",
)

PSI_STABLE = 0.10
PSI_SIGNIFICANT = 0.25
KS_ALPHA = 0.05
_EPS = 1e-4


# --------------------------------------------------------------------------- #
# Statistics (stdlib implementations)
# --------------------------------------------------------------------------- #
def psi(expected_counts: dict, actual_counts: dict) -> float:
    """Population Stability Index over a shared set of categories.

    Zero-proportion bins are floored at a small epsilon so the log term stays finite,
    which is the standard treatment for empty bins in a PSI computation.
    """
    categories = set(expected_counts) | set(actual_counts)
    exp_total = sum(expected_counts.values()) or 1
    act_total = sum(actual_counts.values()) or 1

    score = 0.0
    for cat in categories:
        e = max(expected_counts.get(cat, 0) / exp_total, _EPS)
        a = max(actual_counts.get(cat, 0) / act_total, _EPS)
        score += (a - e) * math.log(a / e)
    return score


def psi_verdict(value: float) -> str:
    if value >= PSI_SIGNIFICANT:
        return "significant"
    if value >= PSI_STABLE:
        return "moderate"
    return "stable"


def ks_two_sample(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic D and asymptotic p-value.

    Returns (D, p). p uses the Kolmogorov distribution Q(t) = 2 sum (-1)^{k-1} e^{-2 k^2 t^2}
    with the effective sample size, the standard large-sample approximation.
    """
    a = sorted(x for x in sample_a if x is not None)
    b = sorted(x for x in sample_b if x is not None)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0.0, 1.0

    values = sorted(set(a) | set(b))
    d = 0.0
    for v in values:
        fa = sum(1 for x in a if x <= v) / n
        fb = sum(1 for x in b if x <= v) / m
        d = max(d, abs(fa - fb))

    en = math.sqrt(n * m / (n + m))
    t = (en + 0.12 + 0.11 / en) * d
    p = _kolmogorov_q(t)
    return d, max(0.0, min(1.0, p))


def _kolmogorov_q(t: float) -> float:
    if t <= 0:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = 2 * ((-1) ** (k - 1)) * math.exp(-2 * k * k * t * t)
        total += term
        if abs(term) < 1e-9:
            break
    return total


# --------------------------------------------------------------------------- #
# Feature extraction
# --------------------------------------------------------------------------- #
def _extract_dict(record: dict) -> dict:
    """Normalize a batch record to a plain OncologyExtract-shaped dict.

    Accepts prediction-cache rows ({"pred": {...}}), gold rows ({"gold": {...}}),
    ExtractionOutput dumps ({"extract": {...}, "fields": {...}}), or a raw extract dict.
    """
    if "pred" in record:
        return record["pred"]
    if "gold" in record:
        return record["gold"]
    if "extract" in record:
        return record["extract"]
    return record


def _confidences(record: dict) -> list[float]:
    """Per-field confidence values if the record carries an ExtractionOutput fields map."""
    fields = record.get("fields") if isinstance(record, dict) else None
    if not isinstance(fields, dict):
        return []
    out = []
    for meta in fields.values():
        if isinstance(meta, dict) and isinstance(meta.get("confidence"), (int, float)):
            out.append(float(meta["confidence"]))
    return out


def _is_present(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, list):
        return len(value) > 0
    return True


@dataclass
class BatchStats:
    """Feature summary of one batch, sufficient to compute drift later."""

    n: int
    categorical: dict  # field -> {category: count}
    presence: dict  # field -> present count
    n_biomarkers: list  # per-record biomarker counts
    n_regimen: list  # per-record regimen lengths
    confidences: list  # flattened per-field confidences (may be empty)


def summarize_batch(records: list[dict]) -> BatchStats:
    categorical: dict[str, Counter] = {f: Counter() for f in CATEGORICAL_FIELDS}
    presence: dict[str, int] = {f: 0 for f in PRESENCE_FIELDS}
    n_biomarkers: list[int] = []
    n_regimen: list[int] = []
    confidences: list[float] = []

    for raw in records:
        rec = _extract_dict(raw)
        for f in CATEGORICAL_FIELDS:
            val = rec.get(f)
            key = str(val).strip().lower() if _is_present(val) else "__absent__"
            categorical[f][key] += 1
        for f in PRESENCE_FIELDS:
            if _is_present(rec.get(f)):
                presence[f] += 1
        n_biomarkers.append(len(rec.get("biomarkers") or []))
        n_regimen.append(len(rec.get("treatment_regimen") or []))
        confidences.extend(_confidences(raw))

    return BatchStats(
        n=len(records),
        categorical={f: dict(c) for f, c in categorical.items()},
        presence=presence,
        n_biomarkers=n_biomarkers,
        n_regimen=n_regimen,
        confidences=confidences,
    )


# --------------------------------------------------------------------------- #
# Batch / baseline IO
# --------------------------------------------------------------------------- #
def load_batch(path: Path) -> list[dict]:
    """Load a batch from a .jsonl cache, a single .json array, or a gold directory."""
    path = Path(path)
    if path.is_dir():
        records = []
        for p in sorted(path.glob("[0-9]*.json")):
            records.append(json.loads(p.read_text()))
        return records
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else [payload]


def save_baseline(stats: BatchStats, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(stats), indent=2) + "\n")


def load_baseline(path: Path) -> BatchStats:
    payload = json.loads(Path(path).read_text())
    return BatchStats(**payload)


# --------------------------------------------------------------------------- #
# Drift report
# --------------------------------------------------------------------------- #
@dataclass
class DriftReport:
    created_at: Optional[str]
    baseline_n: int
    current_n: int
    fields: list = field(default_factory=list)
    verdict: str = "stable"
    n_significant: int = 0
    n_moderate: int = 0


def compute_drift(
    baseline: BatchStats,
    current: BatchStats,
    *,
    ks_alpha: float = KS_ALPHA,
    created_at: Optional[str] = None,
) -> dict:
    """Per-field drift report with PSI/KS scores and an overall threshold verdict."""
    field_rows: list[dict] = []

    for f in CATEGORICAL_FIELDS:
        value = psi(baseline.categorical.get(f, {}), current.categorical.get(f, {}))
        field_rows.append({
            "field": f,
            "signal": "categorical_distribution",
            "method": "psi",
            "score": round(value, 6),
            "verdict": psi_verdict(value),
        })

    for f in PRESENCE_FIELDS:
        exp = {"present": baseline.presence.get(f, 0),
               "absent": baseline.n - baseline.presence.get(f, 0)}
        act = {"present": current.presence.get(f, 0),
               "absent": current.n - current.presence.get(f, 0)}
        value = psi(exp, act)
        field_rows.append({
            "field": f + "_presence",
            "signal": "presence_rate",
            "method": "psi",
            "score": round(value, 6),
            "verdict": psi_verdict(value),
        })

    for f, base_vals, cur_vals in (
        ("n_biomarkers", baseline.n_biomarkers, current.n_biomarkers),
        ("n_regimen", baseline.n_regimen, current.n_regimen),
        ("confidence", baseline.confidences, current.confidences),
    ):
        if not base_vals or not cur_vals:
            field_rows.append({
                "field": f,
                "signal": "continuous",
                "method": "ks",
                "score": None,
                "p_value": None,
                "verdict": "n/a",
                "note": "no data in one or both batches",
            })
            continue
        d, p = ks_two_sample(base_vals, cur_vals)
        field_rows.append({
            "field": f,
            "signal": "continuous",
            "method": "ks",
            "score": round(d, 6),
            "p_value": round(p, 6),
            "verdict": "significant" if p < ks_alpha else "stable",
        })

    n_sig = sum(1 for r in field_rows if r["verdict"] == "significant")
    n_mod = sum(1 for r in field_rows if r["verdict"] == "moderate")
    overall = "significant" if n_sig else "moderate" if n_mod else "stable"

    report = DriftReport(
        created_at=created_at,
        baseline_n=baseline.n,
        current_n=current.n,
        fields=field_rows,
        verdict=overall,
        n_significant=n_sig,
        n_moderate=n_mod,
    )
    return asdict(report)


def print_report(report: dict) -> None:
    print(f"\nDrift report  baseline_n={report['baseline_n']}  current_n={report['current_n']}")
    print(f"{'field':<28} {'method':<6} {'score':>9} {'verdict':>12}")
    print("-" * 60)
    for row in report["fields"]:
        score = "n/a" if row["score"] is None else f"{row['score']:.4f}"
        print(f"{row['field']:<28} {row['method']:<6} {score:>9} {row['verdict']:>12}")
    print(f"\nOVERALL VERDICT: {report['verdict'].upper()} "
          f"({report['n_significant']} significant, {report['n_moderate']} moderate)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Distribution drift check for extraction batches.")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="baseline profile json (from --save-baseline) OR a batch source")
    parser.add_argument("--current", type=Path, required=True,
                        help="current batch: .jsonl cache, .json array, or gold dir")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--alpha", type=float, default=KS_ALPHA)
    parser.add_argument("--save-baseline", type=Path, default=None,
                        help="summarize --baseline as a batch and write a profile here, then exit")
    args = parser.parse_args()

    if args.save_baseline is not None:
        stats = summarize_batch(load_batch(args.baseline))
        save_baseline(stats, args.save_baseline)
        print(f"Wrote baseline profile ({stats.n} records) to {args.save_baseline}")
        return

    if args.baseline.suffix == ".json" and _looks_like_profile(args.baseline):
        baseline = load_baseline(args.baseline)
    else:
        baseline = summarize_batch(load_batch(args.baseline))
    current = summarize_batch(load_batch(args.current))

    from datetime import datetime

    report = compute_drift(
        baseline, current,
        ks_alpha=args.alpha,
        created_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print_report(report)
    print(f"\nWrote {args.out}")


def _looks_like_profile(path: Path) -> bool:
    try:
        payload = json.loads(Path(path).read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(payload, dict) and "categorical" in payload and "presence" in payload


if __name__ == "__main__":
    main()
