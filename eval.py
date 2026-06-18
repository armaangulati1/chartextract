"""Evaluation harness — per-field P/R/F1 and error taxonomy vs synthetic gold labels."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from schema import OncologyExtract

DEFAULT_DATA_DIR = Path("data/synthetic")
DEFAULT_OUT_DIR = Path("data/eval")
PREDICTIONS_CACHE = DEFAULT_OUT_DIR / "predictions.jsonl"

SCALAR_FIELDS = (
    "primary_site",
    "histology",
    "stage",
    "ecog_performance_status",
    "line_of_therapy",
    "date_of_diagnosis",
)
LIST_FIELDS = ("biomarkers", "treatment_regimen")

ERROR_TYPES = (
    "hallucinated",
    "wrong_value",
    "wrong_span",
    "normalization",
    "missed",
    "schema_violation",
)

# Canonical drug names -> synonyms (all lowercase).
DRUG_SYNONYMS: dict[str, set[str]] = {
    "pembrolizumab": {"pembrolizumab", "keytruda"},
    "nivolumab": {"nivolumab", "opdivo"},
    "atezolizumab": {"atezolizumab", "tecentriq"},
    "durvalumab": {"durvalumab", "imfinzi"},
    "ipilimumab": {"ipilimumab", "yervoy"},
    "osimertinib": {"osimertinib", "tagrisso"},
    "crizotinib": {"crizotinib", "xalkori"},
    "trametinib": {"trametinib", "mekinist"},
    "dabrafenib": {"dabrafenib", "tafinlar"},
    "bevacizumab": {"bevacizumab", "avastin"},
    "cetuximab": {"cetuximab", "erbitux"},
    "trastuzumab": {"trastuzumab", "herceptin"},
    "pertuzumab": {"pertuzumab", "perjeta"},
    "palbociclib": {"palbociclib", "ibrance"},
    "abiraterone": {"abiraterone", "zytiga"},
    "enzalutamide": {"enzalutamide", "xtandi"},
    "olaparib": {"olaparib", "lynparza"},
    "regorafenib": {"regorafenib", "stivarga"},
    "sunitinib": {"sunitinib", "sutent"},
    "cabozantinib": {"cabozantinib", "cabometyx"},
    "axitinib": {"axitinib", "inlyta"},
    "carboplatin": {"carboplatin"},
    "cisplatin": {"cisplatin"},
    "paclitaxel": {"paclitaxel", "taxol"},
    "docetaxel": {"docetaxel", "taxotere"},
    "pemetrexed": {"pemetrexed", "alimta"},
    "gemcitabine": {"gemcitabine", "gemzar"},
    "capecitabine": {"capecitabine", "xeloda"},
    "fluorouracil": {"fluorouracil", "5-fu", "5fu", "5 fu"},
    "leucovorin": {"leucovorin", "folinic acid"},
    "irinotecan": {"irinotecan", "camptosar"},
    "etoposide": {"etoposide", "vp-16"},
    "prednisone": {"prednisone"},
    "letrozole": {"letrozole", "femara"},
    "folfox": {"folfox"},
    "folfiri": {"folfiri"},
    "folfirinox": {"folfirinox"},
    "sacituzumab govitecan": {"sacituzumab govitecan", "trodelvy"},
    "mirvetuximab soravtansine": {"mirvetuximab soravtansine", "elahere"},
}

SITE_SYNONYMS: dict[str, set[str]] = {
    "lung": {"lung", "pulmonary", "pulmonary lung"},
    "breast": {"breast", "mammary"},
    "colorectal": {"colorectal", "colon", "rectal", "rectum", "large bowel"},
    "prostate": {"prostate", "prostatic"},
    "kidney": {"kidney", "renal", "renal cell"},
    "ovary": {"ovary", "ovarian"},
    "pancreas": {"pancreas", "pancreatic"},
    "melanoma": {"melanoma", "skin"},
    "head and neck": {"head and neck", "head/neck", "hnc", "oropharynx", "oral cavity"},
}

_DRUG_LOOKUP: dict[str, str] = {}
for canonical, aliases in DRUG_SYNONYMS.items():
    for alias in aliases:
        _DRUG_LOOKUP[alias] = canonical

_SITE_LOOKUP: dict[str, str] = {}
for canonical, aliases in SITE_SYNONYMS.items():
    for alias in aliases:
        _SITE_LOOKUP[alias] = canonical


@dataclass
class FieldCounts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: FieldCounts) -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn


@dataclass
class EvalError:
    example_id: str
    field: str
    error_type: str
    gold_value: str
    pred_value: str
    detail: str = ""


@dataclass
class EvalSummary:
    field_counts: dict[str, FieldCounts] = field(default_factory=lambda: defaultdict(FieldCounts))
    errors: list[EvalError] = field(default_factory=list)
    error_distribution: Counter = field(default_factory=Counter)
    n_examples: int = 0


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def normalize_stage(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _clean_text(str(value)).replace("stage ", "")
    text = text.upper().replace(" ", "")
    roman_map = {"1": "I", "2": "II", "3": "III", "4": "IV"}
    if text in roman_map:
        return roman_map[text]
    return text or None


def normalize_date(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y", "%d %B %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            continue
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return _clean_text(text) or None


def normalize_drug(value: str) -> str:
    text = _clean_text(value)
    text = text.replace("-", " ")
    return _DRUG_LOOKUP.get(text, text)


def normalize_site(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _clean_text(str(value))
    return _SITE_LOOKUP.get(text, text) or None


def normalize_histology(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _clean_text(str(value))
    text = text.replace("-", " ")
    return text or None


def normalize_ecog(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"\d", str(value))
    return int(match.group()) if match else None


def normalize_line(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = _clean_text(str(value))
    word_map = {"first": 1, "second": 2, "third": 3, "fourth": 4}
    for word, num in word_map.items():
        if word in text:
            return num
    match = re.search(r"\d+", text)
    return int(match.group()) if match else None


def normalize_biomarker_name(value: str) -> str:
    text = _clean_text(value)
    text = text.replace(" ", "")
    return text.upper()


def normalize_biomarker_status(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = _clean_text(str(value))
    aliases = {
        "pos": "positive",
        "+": "positive",
        "neg": "negative",
        "-": "negative",
        "wt": "negative",
        "wild-type": "negative",
        "wild type": "negative",
        "indeterminate": "equivocal",
        "pending": "unknown",
        "not tested": "unknown",
    }
    return aliases.get(text, text)


def normalize_biomarker_item(item: dict | Any) -> tuple[str, str]:
    if isinstance(item, dict):
        name = normalize_biomarker_name(item.get("name", ""))
        status = normalize_biomarker_status(item.get("status")) or "unknown"
    else:
        name = normalize_biomarker_name(item.name)
        status = normalize_biomarker_status(item.status) or "unknown"
    return name, status


def normalize_regimen(items: Iterable[Any]) -> set[str]:
    drugs: set[str] = set()
    for item in items:
        text = str(item)
        if "+" in text or "/" in text:
            parts = re.split(r"[+/]", text)
            drugs.update(normalize_drug(p.strip()) for p in parts if p.strip())
        else:
            drugs.add(normalize_drug(text))
    return {d for d in drugs if d}


def is_present_scalar(value: Any) -> bool:
    return value is not None and value != ""


def _raw_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def scalar_normalizer(field: str):
    return {
        "primary_site": normalize_site,
        "histology": normalize_histology,
        "stage": normalize_stage,
        "ecog_performance_status": normalize_ecog,
        "line_of_therapy": normalize_line,
        "date_of_diagnosis": normalize_date,
    }[field]


def _prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def compare_scalar(
    field: str,
    gold: Any,
    pred: Any,
    example_id: str,
) -> tuple[FieldCounts, list[EvalError]]:
    counts = FieldCounts()
    errors: list[EvalError] = []
    normalize = scalar_normalizer(field)
    gold_present = is_present_scalar(gold)
    pred_present = is_present_scalar(pred)

    gold_norm = normalize(gold) if gold_present else None
    pred_norm = normalize(pred) if pred_present else None
    gold_raw = _raw_scalar(gold)
    pred_raw = _raw_scalar(pred)

    if gold_present and pred_present:
        if gold_norm == pred_norm:
            if _clean_text(gold_raw) != _clean_text(pred_raw) and gold_raw != pred_raw:
                errors.append(
                    EvalError(example_id, field, "normalization", gold_raw, pred_raw,
                              "matched after normalization")
                )
            counts.tp += 1
        else:
            errors.append(
                EvalError(example_id, field, "wrong_value", gold_raw, pred_raw)
            )
            counts.fp += 1
            counts.fn += 1
    elif gold_present and not pred_present:
        errors.append(EvalError(example_id, field, "missed", gold_raw, ""))
        counts.fn += 1
    elif not gold_present and pred_present:
        errors.append(EvalError(example_id, field, "hallucinated", "", pred_raw))
        counts.fp += 1

    return counts, errors


def compare_biomarkers(
    gold_items: list,
    pred_items: list,
    example_id: str,
) -> tuple[FieldCounts, list[EvalError]]:
    counts = FieldCounts()
    errors: list[EvalError] = []
    gold_set = {normalize_biomarker_item(x) for x in gold_items}
    pred_set = {normalize_biomarker_item(x) for x in pred_items}

    matched = gold_set & pred_set
    counts.tp += len(matched)

    for item in pred_set - gold_set:
        name, status = item
        name_matches = [g for g in gold_set if g[0] == name]
        if name_matches:
            errors.append(
                EvalError(
                    example_id, "biomarkers", "wrong_value",
                    f"name={name_matches[0][0]}, status={name_matches[0][1]}",
                    f"name={name}, status={status}",
                    "biomarker name match, status mismatch",
                )
            )
        else:
            errors.append(
                EvalError(example_id, "biomarkers", "hallucinated", "", str(dict(name=name, status=status)))
            )
        counts.fp += 1

    for item in gold_set - pred_set:
        name, status = item
        partial = [p for p in pred_set if p[0] != name and (name in p[0] or p[0] in name)]
        if partial:
            errors.append(
                EvalError(
                    example_id, "biomarkers", "wrong_span",
                    str(dict(name=name, status=status)),
                    str(dict(name=partial[0][0], status=partial[0][1])),
                    "partial biomarker name overlap",
                )
            )
        else:
            errors.append(
                EvalError(example_id, "biomarkers", "missed", str(dict(name=name, status=status)), "")
            )
        counts.fn += 1

    return counts, errors


def compare_regimen(
    gold_items: list,
    pred_items: list,
    example_id: str,
) -> tuple[FieldCounts, list[EvalError]]:
    counts = FieldCounts()
    errors: list[EvalError] = []
    gold_set = normalize_regimen(gold_items)
    pred_set = normalize_regimen(pred_items)

    matched = gold_set & pred_set
    counts.tp += len(matched)

    for drug in pred_set - gold_set:
        overlap = [g for g in gold_set if drug in g or g in drug]
        if overlap:
            errors.append(
                EvalError(example_id, "treatment_regimen", "wrong_span", overlap[0], drug,
                          "partial drug-name overlap")
            )
        else:
            errors.append(
                EvalError(example_id, "treatment_regimen", "hallucinated", "", drug)
            )
        counts.fp += 1

    for drug in gold_set - pred_set:
        overlap = [p for p in pred_set if drug in p or p in drug]
        if overlap:
            errors.append(
                EvalError(example_id, "treatment_regimen", "wrong_span", drug, overlap[0],
                          "partial drug-name overlap")
            )
        else:
            errors.append(
                EvalError(example_id, "treatment_regimen", "missed", drug, "")
            )
        counts.fn += 1

    return counts, errors


def evaluate_record(example_id: str, gold: OncologyExtract, pred: OncologyExtract) -> EvalSummary:
    summary = EvalSummary(n_examples=1)
    gold_d = gold.model_dump(mode="json")
    pred_d = pred.model_dump(mode="json")

    for field_name in SCALAR_FIELDS:
        counts, errs = compare_scalar(field_name, gold_d[field_name], pred_d[field_name], example_id)
        summary.field_counts[field_name].add(counts)
        summary.errors.extend(errs)

    bio_counts, bio_errs = compare_biomarkers(
        gold_d["biomarkers"], pred_d["biomarkers"], example_id
    )
    summary.field_counts["biomarkers"].add(bio_counts)
    summary.errors.extend(bio_errs)

    reg_counts, reg_errs = compare_regimen(
        gold_d["treatment_regimen"], pred_d["treatment_regimen"], example_id
    )
    summary.field_counts["treatment_regimen"].add(reg_counts)
    summary.errors.extend(reg_errs)

    for err in summary.errors:
        summary.error_distribution[err.error_type] += 1

    return summary


def merge_summaries(summaries: list[EvalSummary]) -> EvalSummary:
    merged = EvalSummary(n_examples=len(summaries))
    for summary in summaries:
        for field_name, counts in summary.field_counts.items():
            merged.field_counts[field_name].add(counts)
        merged.errors.extend(summary.errors)
        merged.error_distribution.update(summary.error_distribution)
    return merged


def load_pairs(data_dir: Path, limit: Optional[int] = None) -> list[tuple[str, str, dict]]:
    files = sorted(data_dir.glob("[0-9]*.json"))
    if limit is not None:
        files = files[:limit]
    pairs = []
    for path in files:
        payload = json.loads(path.read_text())
        gold = payload.get("gold")
        if gold is None:
            continue
        pairs.append((path.stem, payload["note"], gold))
    return pairs


def load_predictions_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    cached = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cached[row["example_id"]] = row["pred"]
    return cached


def save_predictions_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def run_predictions(
    pairs: list[tuple[str, str, dict]],
    cache_path: Path,
    use_cache: bool,
    extract_fn,
) -> list[tuple[str, OncologyExtract, OncologyExtract]]:
    cached = load_predictions_cache(cache_path) if use_cache else {}
    results: list[tuple[str, OncologyExtract, OncologyExtract]] = []
    cache_rows: list[dict] = []

    for example_id, note, gold_dict in pairs:
        gold = OncologyExtract.model_validate(gold_dict)
        if use_cache and example_id in cached:
            pred = OncologyExtract.model_validate(cached[example_id])
        else:
            pred = extract_fn(note)
            cached[example_id] = pred.model_dump(mode="json")
        cache_rows.append({"example_id": example_id, "pred": cached[example_id]})
        results.append((example_id, gold, pred))

    if not use_cache:
        save_predictions_cache(cache_path, cache_rows)
    return results


def metrics_table(summary: EvalSummary) -> list[dict]:
    rows = []
    all_fields = list(SCALAR_FIELDS) + list(LIST_FIELDS)
    f1_values = []

    for field_name in all_fields:
        counts = summary.field_counts[field_name]
        p, r, f1 = _prf1(counts.tp, counts.fp, counts.fn)
        rows.append({
            "field": field_name,
            "tp": counts.tp,
            "fp": counts.fp,
            "fn": counts.fn,
            "precision": p,
            "recall": r,
            "f1": f1,
        })
        if counts.tp + counts.fn > 0:
            f1_values.append(f1)

    micro_tp = sum(c.tp for c in summary.field_counts.values())
    micro_fp = sum(c.fp for c in summary.field_counts.values())
    micro_fn = sum(c.fn for c in summary.field_counts.values())
    micro_p, micro_r, micro_f1 = _prf1(micro_tp, micro_fp, micro_fn)
    macro_p = sum(r["precision"] for r in rows) / len(rows)
    macro_r = sum(r["recall"] for r in rows) / len(rows)
    macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    rows.extend([
        {
            "field": "macro_avg",
            "tp": "",
            "fp": "",
            "fn": "",
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f1,
        },
        {
            "field": "micro_avg",
            "tp": micro_tp,
            "fp": micro_fp,
            "fn": micro_fn,
            "precision": micro_p,
            "recall": micro_r,
            "f1": micro_f1,
        },
    ])
    return rows


def macro_f1_score(rows: list[dict]) -> float:
    return next(r["f1"] for r in rows if r["field"] == "macro_avg")


def check_macro_f1_threshold(rows: list[dict], min_macro_f1: float) -> None:
    macro_f1 = macro_f1_score(rows)
    print(f"\nMacro-F1: {format_pct(macro_f1)} (threshold: {format_pct(min_macro_f1)})")
    if macro_f1 < min_macro_f1:
        raise SystemExit(
            f"Eval gate failed: macro-F1 {macro_f1:.3f} < threshold {min_macro_f1:.3f}"
        )
    print("Eval gate passed.")


def format_pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def save_latest_metrics(
    path: Path,
    summary: EvalSummary,
    rows: list[dict],
    *,
    source: str,
) -> None:
    macro = next((row for row in rows if row["field"] == "macro_avg"), None)
    payload = {
        "updated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "source": source,
        "n_examples": summary.n_examples,
        "macro_f1": macro["f1"] if macro else None,
        "rows": rows,
        "error_distribution": dict(summary.error_distribution),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_results_md(path: Path, summary: EvalSummary, rows: list[dict]) -> None:
    lines = [
        "# Extraction evaluation results",
        "",
        f"Examples evaluated: **{summary.n_examples}**",
        f"Total errors logged: **{len(summary.errors)}**",
        "",
        "## Per-field metrics",
        "",
        "| field | TP | FP | FN | precision | recall | F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['field']} | {row['tp']} | {row['fp']} | {row['fn']} | "
            f"{format_pct(row['precision'])} | {format_pct(row['recall'])} | {format_pct(row['f1'])} |"
        )

    lines.extend(["", "## Error taxonomy", ""])
    total_errors = sum(summary.error_distribution.values()) or 1
    lines.append("| error_type | count | share |")
    lines.append("|---|---:|---:|")
    for error_type in ERROR_TYPES:
        count = summary.error_distribution.get(error_type, 0)
        if count:
            lines.append(
                f"| {error_type} | {count} | {format_pct(count / total_errors)} |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_errors_csv(path: Path, errors: list[EvalError]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["example_id", "field", "error_type", "gold_value", "pred_value", "detail"],
        )
        writer.writeheader()
        for err in errors:
            writer.writerow({
                "example_id": err.example_id,
                "field": err.field,
                "error_type": err.error_type,
                "gold_value": err.gold_value,
                "pred_value": err.pred_value,
                "detail": err.detail,
            })


def print_report(summary: EvalSummary, rows: list[dict]) -> None:
    print(f"\nEvaluated {summary.n_examples} examples\n")
    print(f"{'field':<24} {'P':>7} {'R':>7} {'F1':>7}  {'TP':>4} {'FP':>4} {'FN':>4}")
    print("-" * 64)
    for row in rows:
        print(
            f"{row['field']:<24} "
            f"{format_pct(row['precision']):>7} "
            f"{format_pct(row['recall']):>7} "
            f"{format_pct(row['f1']):>7}  "
            f"{str(row['tp']):>4} "
            f"{str(row['fp']):>4} "
            f"{str(row['fn']):>4}"
        )

    print("\nError taxonomy:")
    total_errors = sum(summary.error_distribution.values()) or 1
    for error_type in ERROR_TYPES:
        count = summary.error_distribution.get(error_type, 0)
        if count:
            print(f"  {error_type:<18} {count:>5}  ({format_pct(count / total_errors)})")


def evaluate_dataset(
    pairs: list[tuple[str, str, dict]],
    extract_fn,
    cache_path: Path = PREDICTIONS_CACHE,
    use_cache: bool = False,
) -> EvalSummary:
    evaluated = run_predictions(pairs, cache_path, use_cache, extract_fn)
    summaries = [
        evaluate_record(example_id, gold, pred)
        for example_id, gold, pred in evaluated
    ]
    return merge_summaries(summaries)


def main():
    parser = argparse.ArgumentParser(description="Evaluate extraction against synthetic gold labels.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-cache", action="store_true",
                        help="reuse cached predictions from prior run")
    parser.add_argument("--cache", type=Path, default=PREDICTIONS_CACHE)
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=None,
        help="exit 1 if macro-averaged F1 falls below this threshold (0–1)",
    )
    args = parser.parse_args()

    from extractor import extract_record

    pairs = load_pairs(args.data_dir, limit=args.limit)
    if not pairs:
        raise SystemExit(f"No synthetic pairs found in {args.data_dir}")

    summary = evaluate_dataset(pairs, extract_record, cache_path=args.cache, use_cache=args.use_cache)
    rows = metrics_table(summary)

    write_results_md(args.out_dir / "results.md", summary, rows)
    write_errors_csv(args.out_dir / "errors.csv", summary.errors)
    save_latest_metrics(
        Path("data/eval/latest_metrics.json"),
        summary,
        rows,
        source=str(args.data_dir),
    )
    print_report(summary, rows)
    if args.min_macro_f1 is not None:
        check_macro_f1_threshold(rows, args.min_macro_f1)
    print(f"\nWrote {args.out_dir / 'results.md'} and {args.out_dir / 'errors.csv'}")


if __name__ == "__main__":
    main()
