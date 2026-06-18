from datetime import date

from eval import (
    check_macro_f1_threshold,
    compare_biomarkers,
    compare_regimen,
    compare_scalar,
    evaluate_record,
    format_dataset_eval_section,
    merge_summaries,
    metrics_table,
    normalize_date,
    normalize_drug,
    normalize_stage,
)
from schema import Biomarker, BiomarkerStatus, CancerStage, OncologyExtract


def test_normalize_stage():
    assert normalize_stage("stage iiia") == "IIIA"
    assert normalize_stage("3") == "III"


def test_normalize_date():
    assert normalize_date("2023-10-09") == "2023-10-09"
    assert normalize_date(date(2020, 8, 22)) == "2020-08-22"


def test_normalize_drug_synonym():
    assert normalize_drug("Keytruda") == "pembrolizumab"


def test_scalar_exact_match_counts_tp():
    counts, errors = compare_scalar("stage", "IIIA", "stage IIIA", "0001")
    assert counts.tp == 1
    assert not errors or errors[0].error_type == "normalization"


def test_scalar_missed_and_hallucinated():
    missed, _ = compare_scalar("line_of_therapy", 2, None, "0001")
    hallucinated, _ = compare_scalar("line_of_therapy", None, 2, "0001")
    assert missed.fn == 1
    assert hallucinated.fp == 1


def test_biomarker_set_match():
    gold = [Biomarker(name="EGFR", status=BiomarkerStatus.POSITIVE)]
    pred = [Biomarker(name="egfr", status=BiomarkerStatus.POSITIVE)]
    counts, errors = compare_biomarkers(gold, pred, "0001")
    assert counts.tp == 1
    assert errors == []


def test_regimen_set_partial_miss():
    counts, errors = compare_regimen(["pembrolizumab"], ["nivolumab"], "0001")
    assert counts.tp == 0
    assert counts.fp == 1
    assert counts.fn == 1
    assert errors


def test_evaluate_record_and_metrics():
    gold = OncologyExtract(
        primary_site="lung",
        histology="adenocarcinoma",
        stage=CancerStage.IIIA,
        line_of_therapy=1,
        treatment_regimen=["pembrolizumab"],
    )
    pred = OncologyExtract(
        primary_site="lung",
        histology="adenocarcinoma",
        stage=CancerStage.IIIA,
        line_of_therapy=1,
        treatment_regimen=["pembrolizumab"],
    )
    summary = merge_summaries([evaluate_record("0001", gold, pred)])
    rows = metrics_table(summary)
    assert summary.n_examples == 1
    assert any(r["field"] == "stage" and r["f1"] == 1.0 for r in rows)


def test_macro_f1_threshold_gate():
    rows = [{"field": "macro_avg", "f1": 0.90, "precision": 0.9, "recall": 0.9}]
    check_macro_f1_threshold(rows, 0.85)


def test_format_dataset_eval_section_includes_metrics_and_taxonomy():
    gold = OncologyExtract(primary_site="lung", line_of_therapy=1)
    pred = OncologyExtract(primary_site="breast", line_of_therapy=1)
    summary = evaluate_record("0001", gold, pred)
    rows = metrics_table(summary)
    section = "\n".join(format_dataset_eval_section("Test set (1 note)", summary, rows))
    assert "#### Per-field metrics" in section
    assert "#### Error taxonomy" in section
    assert "primary_site" in section
    assert "wrong_value" in section


def test_macro_f1_threshold_gate_fails():
    import pytest

    rows = [{"field": "macro_avg", "f1": 0.80, "precision": 0.8, "recall": 0.8}]
    with pytest.raises(SystemExit):
        check_macro_f1_threshold(rows, 0.85)
