"""Retrain / re-eval trigger: drift-gated re-evaluation of the production version.

When a drift report crosses the threshold, this kicks off the EXISTING eval pipeline
against the currently promoted registry version and writes a decision record into that
version. It runs the eval over cached predictions (use_cache=True) so it makes ZERO live
LLM calls; a guard extractor raises if the cache ever misses, rather than silently
calling a model.

Honest naming: for an LLM-extraction system there are no gradient weights to update, so
"retrain" here means (1) re-run the eval to get fresh metrics on the current data, and
(2) emit a concrete prompt/model version-bump PROPOSAL for a human to act on. It does not
fine-tune or train anything, and it never mutates the promoted version's own metrics.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import registry
from eval import evaluate_dataset, load_pairs, macro_f1_score, metrics_table

DEFAULT_DRIFT_REPORT = Path("data/mlops/drift_report.json")
TRIGGER_VERDICTS = ("significant",)


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _guard_extractor(note: str):
    raise RuntimeError(
        "retrain re-eval hit an uncached example; live LLM extraction is disabled here"
    )


def reeval_cached(gold_dir: Path, cache_path: Path) -> dict:
    """Re-run the eval harness over cached predictions (no live LLM). Returns metrics."""
    pairs = load_pairs(Path(gold_dir))
    if not pairs:
        raise SystemExit(f"no gold pairs in {gold_dir}")
    summary = evaluate_dataset(
        pairs, _guard_extractor, cache_path=Path(cache_path), use_cache=True
    )
    rows = metrics_table(summary)
    return {
        "n_examples": summary.n_examples,
        "macro_f1": macro_f1_score(rows),
        "metrics": registry.metrics_from_rows(rows),
    }


def build_proposal(prod: dict, drift_report: dict, reeval: dict) -> dict:
    """A concrete, human-actionable version-bump proposal. No training implied."""
    drifted = [
        r["field"] for r in drift_report["fields"]
        if r["verdict"] in ("significant", "moderate")
    ]
    cfg = prod["model_config"]
    return {
        "proposed_action": "prompt_or_model_version_bump",
        "rationale": (
            "Input/output distribution drift crossed threshold; re-evaluate and revise "
            "the extraction prompt (or model) for the drifted fields, then register a new "
            "version. This system has no trainable weights, so no gradient retraining is "
            "proposed."
        ),
        "drifted_fields": drifted,
        "current_prompt_config": cfg.get("prompt_config"),
        "current_model": cfg.get("model"),
        "suggested_next_prompt_config": f"{cfg.get('prompt_config')}+drift-{_utc_now()[:10]}",
        "reeval_macro_f1": round(reeval["macro_f1"], 6),
        "baseline_macro_f1": prod["metrics"].get("macro_f1"),
    }


def run_trigger(
    drift_report_path: Path = DEFAULT_DRIFT_REPORT,
    *,
    gold_dir: Path,
    cache_path: Path,
    registry_path: Path = registry.DEFAULT_REGISTRY,
    trigger_verdicts: tuple = TRIGGER_VERDICTS,
) -> dict:
    """Evaluate a drift report and, if it triggers, write a decision record. Returns it."""
    report = json.loads(Path(drift_report_path).read_text())
    verdict = report.get("verdict", "stable")
    triggered = verdict in trigger_verdicts

    reg = registry.load_registry(registry_path)
    prod = registry.production_version(reg)

    decision = {
        "type": "retrain_evaluation",
        "created_at": _utc_now(),
        "drift_verdict": verdict,
        "triggered": triggered,
        "drift_report": str(drift_report_path),
    }

    if not triggered:
        decision["note"] = "drift below trigger threshold; no re-eval run"
        return decision

    if prod is None:
        decision["note"] = "drift triggered but no production version registered"
        decision["error"] = "no_production_version"
        return decision

    reeval = reeval_cached(gold_dir, cache_path)
    decision["reeval"] = reeval
    decision["recommendation"] = "retrain_recommended"
    decision["proposal"] = build_proposal(prod, report, reeval)

    registry.add_decision(prod["version"], decision, path=registry_path)
    decision["written_to_version"] = prod["version"]
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift-gated re-eval / retrain trigger.")
    parser.add_argument("--drift-report", type=Path, default=DEFAULT_DRIFT_REPORT)
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="gold dataset dir to re-eval the production version on")
    parser.add_argument("--cache", type=Path, required=True,
                        help="cached predictions jsonl covering the gold dir (no live LLM)")
    parser.add_argument("--registry", type=Path, default=registry.DEFAULT_REGISTRY)
    args = parser.parse_args()

    decision = run_trigger(
        args.drift_report,
        gold_dir=args.gold_dir,
        cache_path=args.cache,
        registry_path=args.registry,
    )
    print(json.dumps(decision, indent=2))
    if decision.get("triggered") and decision.get("recommendation") == "retrain_recommended":
        print(f"\nRETRAIN RECOMMENDED -> decision written to "
              f"{decision.get('written_to_version')}")
    else:
        print(f"\nNo retrain: {decision.get('note', 'below threshold')}")


if __name__ == "__main__":
    main()
