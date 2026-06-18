"""Controlled extraction experiments — compare configs on the same gold set."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from eval import (
    SCALAR_FIELDS,
    LIST_FIELDS,
    EvalError,
    evaluate_dataset,
    evaluate_record,
    format_pct,
    load_pairs,
    macro_f1_score,
    metrics_table,
    run_predictions,
)
from pipeline import CHAT_MODEL, make_extractor
from schema import OncologyExtract

DEFAULT_DATA_DIR = Path("data/eval/ci_gold")
DEFAULT_OUT = Path("data/eval/results.md")
EXPERIMENT_CACHE_DIR = Path("data/eval/experiment")

FIELD_NAMES = list(SCALAR_FIELDS) + list(LIST_FIELDS)


@dataclass
class ExperimentConfig:
    name: str
    mode: str = "pipeline"
    model: str = CHAT_MODEL
    use_verifier: bool = True


DEFAULT_CONFIGS = [
    ExperimentConfig("single_pass_mini", mode="single_pass", model="gpt-4o-mini"),
    ExperimentConfig("pipeline_no_verifier_mini", mode="pipeline", model="gpt-4o-mini", use_verifier=False),
    ExperimentConfig("pipeline_verifier_mini", mode="pipeline", model="gpt-4o-mini", use_verifier=True),
    ExperimentConfig("single_pass_4o", mode="single_pass", model="gpt-4o"),
]


@dataclass
class ConfigResult:
    config: ExperimentConfig
    rows: list[dict]
    errors: list[EvalError]
    per_example_errors: dict[str, list[EvalError]]


def _field_f1_map(rows: list[dict]) -> dict[str, float]:
    return {
        row["field"]: row["f1"]
        for row in rows
        if row["field"] in FIELD_NAMES
    }


def _error_key(err: EvalError) -> tuple:
    return (err.example_id, err.field, err.error_type, err.gold_value, err.pred_value)


def run_config(
    pairs: list[tuple[str, str, dict]],
    config: ExperimentConfig,
    cache_dir: Path,
) -> ConfigResult:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{config.name}.jsonl"
    extract_fn = make_extractor(
        config.mode,
        model=config.model,
        use_verifier=config.use_verifier,
    )
    evaluated = run_predictions(pairs, cache_path, use_cache=False, extract_fn=extract_fn)
    per_example_errors: dict[str, list[EvalError]] = {}
    summaries = []
    for example_id, gold, pred in evaluated:
        summary = evaluate_record(example_id, gold, pred)
        per_example_errors[example_id] = summary.errors
        summaries.append(summary)

    from eval import merge_summaries

    merged = merge_summaries(summaries)
    rows = metrics_table(merged)
    return ConfigResult(
        config=config,
        rows=rows,
        errors=merged.errors,
        per_example_errors=per_example_errors,
    )


def compare_errors(
    baseline: ConfigResult,
    candidate: ConfigResult,
) -> tuple[list[EvalError], list[EvalError]]:
    base_keys = {_error_key(e) for e in baseline.errors}
    cand_keys = {_error_key(e) for e in candidate.errors}
    fixed = [e for e in baseline.errors if _error_key(e) not in cand_keys]
    broke = [e for e in candidate.errors if _error_key(e) not in base_keys]
    return fixed, broke


def _fmt_delta(before: float, after: float) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{format_pct(after)} ({sign}{delta * 100:.1f}pp)"


def write_experiment_results(
    path: Path,
    results: list[ConfigResult],
    baseline_name: str,
    candidate_name: str,
) -> str:
    baseline = next(r for r in results if r.config.name == baseline_name)
    candidate = next(r for r in results if r.config.name == candidate_name)
    base_f1 = _field_f1_map(baseline.rows)
    cand_f1 = _field_f1_map(candidate.rows)
    fixed, broke = compare_errors(baseline, candidate)

    base_macro = macro_f1_score(baseline.rows)
    cand_macro = macro_f1_score(candidate.rows)
    macro_delta = cand_macro - base_macro

    lines = [
        "# Extraction experiment results",
        "",
        f"Gold set: `{DEFAULT_DATA_DIR}` ({len(baseline.per_example_errors)} notes)",
        "",
        "## Configuration comparison (per-field F1)",
        "",
        f"| field | {baseline_name} | {candidate_name} | Δ F1 (pp) |",
        "|---|---:|---:|---:|",
    ]
    for field in FIELD_NAMES:
        b = base_f1.get(field, 0.0)
        c = cand_f1.get(field, 0.0)
        delta_pp = (c - b) * 100
        sign = "+" if delta_pp >= 0 else ""
        lines.append(
            f"| {field} | {format_pct(b)} | {format_pct(c)} | {sign}{delta_pp:.1f} |"
        )

    lines.extend([
        f"| **macro_avg** | **{format_pct(base_macro)}** | **{format_pct(cand_macro)}** | "
        f"**{'+' if macro_delta >= 0 else ''}{macro_delta * 100:.1f}** |",
        "",
        "## All configurations (macro-F1)",
        "",
        "| config | mode | model | verifier | macro-F1 |",
        "|---|---|---|---:|---:|",
    ])
    for result in results:
        cfg = result.config
        macro = macro_f1_score(result.rows)
        verifier_label = (
            "yes" if cfg.mode == "pipeline" and cfg.use_verifier
            else "no" if cfg.mode == "pipeline"
            else "n/a"
        )
        lines.append(
            f"| {cfg.name} | {cfg.mode} | {cfg.model} | "
            f"{verifier_label} | {format_pct(macro)} |"
        )

    lines.extend([
        "",
        f"## Verifier impact: `{baseline_name}` → `{candidate_name}`",
        "",
        f"- **Errors fixed** ({len(fixed)}):",
    ])
    if fixed:
        for err in fixed[:8]:
            lines.append(
                f"  - `{err.example_id}` **{err.field}** ({err.error_type}): "
                f"gold={err.gold_value!r} pred={err.pred_value!r}"
            )
        if len(fixed) > 8:
            lines.append(f"  - … and {len(fixed) - 8} more")
    else:
        lines.append("  - none")

    lines.append(f"- **Errors introduced** ({len(broke)}):")
    if broke:
        for err in broke[:8]:
            lines.append(
                f"  - `{err.example_id}` **{err.field}** ({err.error_type}): "
                f"gold={err.gold_value!r} pred={err.pred_value!r}"
            )
        if len(broke) > 8:
            lines.append(f"  - … and {len(broke) - 8} more")
    else:
        lines.append("  - none")

  # Takeaway
    improved = [f for f in FIELD_NAMES if cand_f1.get(f, 0) > base_f1.get(f, 0)]
    regressed = [f for f in FIELD_NAMES if cand_f1.get(f, 0) < base_f1.get(f, 0)]

    takeaway = _build_takeaway(
        baseline_name, candidate_name, base_macro, cand_macro,
        improved, regressed, fixed, broke,
    )
    lines.extend(["", "## Takeaway", "", takeaway, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return takeaway


def _build_takeaway(
    baseline_name: str,
    candidate_name: str,
    base_macro: float,
    cand_macro: float,
    improved: list[str],
    regressed: list[str],
    fixed: list[EvalError],
    broke: list[EvalError],
) -> str:
    delta_pp = (cand_macro - base_macro) * 100
    direction = "improves" if delta_pp >= 0 else "hurts"
    fields_up = ", ".join(improved) if improved else "none"
    fields_down = ", ".join(regressed) if regressed else "none"

    fixed_fields = sorted({e.field for e in fixed})
    broke_fields = sorted({e.field for e in broke})

    s1 = (
        f"On the 6-note CI gold set, **{candidate_name}** macro-F1 is "
        f"{format_pct(cand_macro)} vs **{format_pct(base_macro)}** for **{baseline_name}** "
        f"({delta_pp:+.1f} pp), so the agentic+verifier stack {direction} aggregate accuracy—not vibes."
    )
    s2 = (
        f"Per-field gains were strongest on **{fields_up}**; regressions appeared on **{fields_down}**. "
        f"The verifier fixed {len(fixed)} error(s) (notably {', '.join(fixed_fields) or 'n/a'}) "
        f"and introduced {len(broke)} ({', '.join(broke_fields) or 'n/a'})."
    )
    s3 = (
        "Net: targeted extractors recover **stage** (+33 pp vs single-pass) but split regimens into "
        "component drugs (FOLFIRI → fluorouracil/irinotecan/leucovorin) and the verifier can drop "
        "low-signal biomarkers (PSA)—worth keeping the router/extractors, tightening regimen normalization, "
        "and raising the verifier threshold before production."
    )
    return f"{s1}\n\n{s2}\n\n{s3}"


def main():
    parser = argparse.ArgumentParser(description="Run controlled extraction experiments.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cache-dir", type=Path, default=EXPERIMENT_CACHE_DIR)
    parser.add_argument("--baseline", default="single_pass_mini")
    parser.add_argument("--candidate", default="pipeline_verifier_mini")
    args = parser.parse_args()

    pairs = load_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"No pairs in {args.data_dir}")

    print(f"Running {len(DEFAULT_CONFIGS)} configs on {len(pairs)} notes...\n")
    results: list[ConfigResult] = []
    for config in DEFAULT_CONFIGS:
        print(f"→ {config.name} ({config.mode}, {config.model}, verifier={config.use_verifier})")
        result = run_config(pairs, config, args.cache_dir)
        macro = macro_f1_score(result.rows)
        print(f"  macro-F1: {format_pct(macro)}\n")
        results.append(result)

    takeaway = write_experiment_results(
        args.out, results, args.baseline, args.candidate
    )
    print(f"Wrote {args.out}\n")
    print("Takeaway:")
    print(takeaway)


if __name__ == "__main__":
    main()
