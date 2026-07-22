import json

from registry import (
    ModelConfig,
    add_decision,
    compare_versions,
    dataset_hash,
    get_version,
    load_registry,
    metrics_from_rows,
    production_version,
    promote_version,
    register_version,
)


def _write_gold_dir(tmp_path, n=3):
    d = tmp_path / "gold"
    d.mkdir()
    for i in range(n):
        (d / f"{i:04d}.json").write_text(
            json.dumps({"note": f"note {i}", "gold": {"primary_site": "lung"}})
        )
    return d


def _metrics(macro=0.9):
    return {"macro_f1": macro, "micro_f1": macro, "per_field": {"stage": 1.0, "primary_site": 0.8}}


def test_dataset_hash_is_deterministic_and_content_sensitive(tmp_path):
    d = _write_gold_dir(tmp_path)
    h1 = dataset_hash(d)
    h2 = dataset_hash(d)
    assert h1 == h2 and len(h1) == 64
    (d / "0000.json").write_text(json.dumps({"note": "changed", "gold": {}}))
    assert dataset_hash(d) != h1


def test_register_and_list(tmp_path):
    reg_path = tmp_path / "registry.json"
    d = _write_gold_dir(tmp_path)
    rec = register_version(
        "cfg_a", ModelConfig(model="gpt-4o-mini"), _metrics(0.90),
        dataset_dir=d, path=reg_path,
    )
    assert rec["version"] == "v0001"
    assert rec["status"] == "registered"
    reg = load_registry(reg_path)
    assert len(reg["versions"]) == 1
    assert reg["versions"][0]["dataset"]["n_examples"] == 3


def test_compare_versions_delta(tmp_path):
    reg_path = tmp_path / "registry.json"
    d = _write_gold_dir(tmp_path)
    register_version("a", ModelConfig(model="m"), _metrics(0.80), dataset_dir=d, path=reg_path)
    register_version("b", ModelConfig(model="m"), _metrics(0.90), dataset_dir=d, path=reg_path)
    reg = load_registry(reg_path)
    result = compare_versions(reg, "v0001", "v0002")
    assert round(result["macro_f1_delta"], 4) == 0.10
    assert result["same_dataset"] is True


def test_promote_demotes_prior_production(tmp_path):
    reg_path = tmp_path / "registry.json"
    d = _write_gold_dir(tmp_path)
    register_version("a", ModelConfig(model="m"), _metrics(0.80), dataset_dir=d, path=reg_path)
    register_version("b", ModelConfig(model="m"), _metrics(0.90), dataset_dir=d, path=reg_path)
    promote_version("v0001", path=reg_path)
    promote_version("v0002", path=reg_path)
    reg = load_registry(reg_path)
    assert production_version(reg)["version"] == "v0002"
    assert get_version(reg, "v0001")["status"] == "archived"


def test_add_decision_is_append_only(tmp_path):
    reg_path = tmp_path / "registry.json"
    d = _write_gold_dir(tmp_path)
    register_version("a", ModelConfig(model="m"), _metrics(), dataset_dir=d, path=reg_path)
    add_decision("v0001", {"type": "retrain_evaluation", "triggered": True}, path=reg_path)
    add_decision("v0001", {"type": "retrain_evaluation", "triggered": False}, path=reg_path)
    reg = load_registry(reg_path)
    decisions = get_version(reg, "v0001")["decisions"]
    assert len(decisions) == 2
    assert all("created_at" in dc for dc in decisions)


def test_metrics_from_rows_extracts_macro_micro_and_fields():
    rows = [
        {"field": "stage", "f1": 1.0},
        {"field": "primary_site", "f1": 0.5},
        {"field": "macro_avg", "f1": 0.75},
        {"field": "micro_avg", "f1": 0.8},
    ]
    m = metrics_from_rows(rows)
    assert m["macro_f1"] == 0.75
    assert m["micro_f1"] == 0.8
    assert m["per_field"] == {"stage": 1.0, "primary_site": 0.5}
