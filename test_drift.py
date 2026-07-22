from drift import (
    compute_drift,
    ks_two_sample,
    load_batch,
    psi,
    psi_verdict,
    summarize_batch,
)


def _record(site="lung", stage="IIIA", biomarkers=None, regimen=None, ecog=1):
    return {
        "pred": {
            "primary_site": site,
            "histology": "adenocarcinoma",
            "stage": stage,
            "ecog_performance_status": ecog,
            "line_of_therapy": 1,
            "date_of_diagnosis": "2023-01-01",
            "biomarkers": biomarkers if biomarkers is not None else [{"name": "EGFR", "status": "positive"}],
            "treatment_regimen": regimen if regimen is not None else ["pembrolizumab"],
        }
    }


def test_psi_zero_for_identical_distributions():
    counts = {"lung": 40, "breast": 30, "colon": 30}
    assert psi(counts, counts) == 0.0


def test_psi_grows_with_shift_and_verdict_bands():
    base = {"lung": 50, "breast": 50}
    shifted = {"lung": 95, "breast": 5}
    value = psi(base, shifted)
    assert value >= 0.25
    assert psi_verdict(value) == "significant"
    assert psi_verdict(0.05) == "stable"
    assert psi_verdict(0.15) == "moderate"


def test_ks_detects_shifted_continuous():
    a = [0.9] * 40 + [0.85] * 40
    b = [0.4] * 40 + [0.45] * 40
    d, p = ks_two_sample(a, b)
    assert d > 0.9
    assert p < 0.05


def test_ks_stable_for_same_distribution():
    a = [0.8, 0.82, 0.79, 0.81, 0.83, 0.78] * 5
    d, p = ks_two_sample(a, a)
    assert d == 0.0
    assert p == 1.0


def test_load_batch_handles_pred_gold_and_extract(tmp_path):
    jl = tmp_path / "b.jsonl"
    jl.write_text(
        '{"example_id": "1", "pred": {"primary_site": "lung"}}\n'
        '{"gold": {"primary_site": "breast"}}\n'
    )
    records = load_batch(jl)
    stats = summarize_batch(records)
    assert stats.n == 2
    assert stats.categorical["primary_site"]["lung"] == 1
    assert stats.categorical["primary_site"]["breast"] == 1


def test_compute_drift_stable_when_batches_match():
    batch = [_record() for _ in range(30)]
    base = summarize_batch(batch)
    cur = summarize_batch(batch)
    report = compute_drift(base, cur)
    assert report["verdict"] == "stable"
    assert report["n_significant"] == 0


def test_compute_drift_flags_categorical_shift():
    base = summarize_batch([_record(site="lung") for _ in range(40)])
    cur = summarize_batch([_record(site="breast") for _ in range(40)])
    report = compute_drift(base, cur)
    site_row = next(r for r in report["fields"] if r["field"] == "primary_site")
    assert site_row["verdict"] == "significant"
    assert report["verdict"] == "significant"


def test_compute_drift_flags_presence_shift():
    base = summarize_batch([_record(ecog=1) for _ in range(40)])
    cur = summarize_batch([_record(ecog=None) for _ in range(40)])
    report = compute_drift(base, cur)
    pres_row = next(r for r in report["fields"] if r["field"] == "ecog_performance_status_presence")
    assert pres_row["verdict"] in ("moderate", "significant")


def test_confidence_ks_runs_when_present():
    base = summarize_batch([
        {"extract": {"primary_site": "lung"}, "fields": {"primary_site": {"confidence": 0.95}}}
        for _ in range(30)
    ])
    cur = summarize_batch([
        {"extract": {"primary_site": "lung"}, "fields": {"primary_site": {"confidence": 0.55}}}
        for _ in range(30)
    ])
    report = compute_drift(base, cur)
    conf_row = next(r for r in report["fields"] if r["field"] == "confidence")
    assert conf_row["verdict"] == "significant"
    assert conf_row["p_value"] is not None


def test_confidence_ks_na_when_absent():
    base = summarize_batch([_record() for _ in range(10)])
    cur = summarize_batch([_record() for _ in range(10)])
    report = compute_drift(base, cur)
    conf_row = next(r for r in report["fields"] if r["field"] == "confidence")
    assert conf_row["verdict"] == "n/a"
