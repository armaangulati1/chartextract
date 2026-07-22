import json

import pytest

from registry import ModelConfig, get_version, load_registry, promote_version, register_version
from retrain import run_trigger


def _gold_and_cache(tmp_path):
    """A tiny self-contained gold dir + matching prediction cache (identical => perfect eval)."""
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    record = {
        "primary_site": "lung",
        "histology": "adenocarcinoma",
        "stage": "IIIA",
        "ecog_performance_status": 1,
        "line_of_therapy": 1,
        "date_of_diagnosis": "2023-01-01",
        "biomarkers": [{"name": "EGFR", "status": "positive"}],
        "treatment_regimen": ["pembrolizumab"],
    }
    (gold_dir / "0000.json").write_text(json.dumps({"note": "n", "gold": record}))
    cache = tmp_path / "cache.jsonl"
    cache.write_text(json.dumps({"example_id": "0000", "pred": record}) + "\n")
    return gold_dir, cache


def _registry_with_prod(tmp_path, gold_dir):
    reg_path = tmp_path / "registry.json"
    register_version(
        "prod_cfg", ModelConfig(model="gpt-4o-mini"),
        {"macro_f1": 0.90, "micro_f1": 0.90, "per_field": {"stage": 1.0}},
        dataset_dir=gold_dir, path=reg_path,
    )
    promote_version("v0001", path=reg_path)
    return reg_path


def _drift_report(tmp_path, verdict):
    path = tmp_path / "drift.json"
    path.write_text(json.dumps({
        "verdict": verdict,
        "baseline_n": 40,
        "current_n": 40,
        "n_significant": 1 if verdict == "significant" else 0,
        "n_moderate": 0,
        "fields": [
            {"field": "primary_site", "method": "psi", "score": 0.6, "verdict": verdict},
        ],
    }))
    return path


def test_trigger_fires_and_writes_decision(tmp_path):
    gold_dir, cache = _gold_and_cache(tmp_path)
    reg_path = _registry_with_prod(tmp_path, gold_dir)
    report = _drift_report(tmp_path, "significant")

    decision = run_trigger(
        report, gold_dir=gold_dir, cache_path=cache, registry_path=reg_path,
    )
    assert decision["triggered"] is True
    assert decision["recommendation"] == "retrain_recommended"
    assert decision["written_to_version"] == "v0001"
    assert decision["proposal"]["proposed_action"] == "prompt_or_model_version_bump"
    assert "primary_site" in decision["proposal"]["drifted_fields"]
    # decision is persisted on the production version
    reg = load_registry(reg_path)
    assert len(get_version(reg, "v0001")["decisions"]) == 1


def test_trigger_noop_below_threshold(tmp_path):
    gold_dir, cache = _gold_and_cache(tmp_path)
    reg_path = _registry_with_prod(tmp_path, gold_dir)
    report = _drift_report(tmp_path, "stable")

    decision = run_trigger(
        report, gold_dir=gold_dir, cache_path=cache, registry_path=reg_path,
    )
    assert decision["triggered"] is False
    assert "reeval" not in decision
    reg = load_registry(reg_path)
    assert get_version(reg, "v0001")["decisions"] == []


def test_reeval_runs_without_live_llm(tmp_path):
    gold_dir, cache = _gold_and_cache(tmp_path)
    reg_path = _registry_with_prod(tmp_path, gold_dir)
    report = _drift_report(tmp_path, "significant")

    decision = run_trigger(
        report, gold_dir=gold_dir, cache_path=cache, registry_path=reg_path,
    )
    # identical gold and pred => perfect macro-F1, proving the cached re-eval executed
    assert decision["reeval"]["macro_f1"] == 1.0
    assert decision["reeval"]["n_examples"] == 1


def test_trigger_handles_missing_production_version(tmp_path):
    gold_dir, cache = _gold_and_cache(tmp_path)
    reg_path = tmp_path / "empty_registry.json"
    report = _drift_report(tmp_path, "significant")

    decision = run_trigger(
        report, gold_dir=gold_dir, cache_path=cache, registry_path=reg_path,
    )
    assert decision["triggered"] is True
    assert decision["error"] == "no_production_version"


def test_reeval_guard_raises_on_cache_miss(tmp_path):
    from retrain import reeval_cached
    gold_dir, _ = _gold_and_cache(tmp_path)
    empty_cache = tmp_path / "empty.jsonl"
    empty_cache.write_text("")
    with pytest.raises(RuntimeError, match="live LLM extraction is disabled"):
        reeval_cached(gold_dir, empty_cache)
