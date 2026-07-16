"""Positive and negative unit tests for every pre-bill check + aggregation."""

from __future__ import annotations

from datetime import date

from prebill.checks import (
    check_code_documentation_support,
    check_diagnosis_procedure_consistency,
    check_documentation_level_mismatch,
    check_duplicate_or_overlapping_codes,
    check_field_conflicts,
    check_low_confidence_support,
    check_missing_attestation,
    check_missing_required_elements,
    run_checks,
)
from prebill.claim import ClaimStub
from prebill.findings import Severity
from prebill.report import HOLD, READY, REVIEW, build_report
from schema import ExtractionOutput


def _extraction(**extract_kwargs) -> ExtractionOutput:
    base = {
        "primary_site": "lung",
        "histology": "adenocarcinoma",
        "stage": "IIIA",
        "biomarkers": [{"name": "EGFR", "status": "positive"}],
        "ecog_performance_status": 1,
        "line_of_therapy": 1,
        "date_of_diagnosis": "2024-03-15",
        "treatment_regimen": ["osimertinib"],
    }
    base.update(extract_kwargs)
    payload = {"extract": base}
    for meta_key in ("fields", "needs_review", "review_threshold"):
        if meta_key in extract_kwargs:
            payload[meta_key] = extract_kwargs.pop(meta_key)
            base.pop(meta_key, None)
    return ExtractionOutput.model_validate(payload)


def _claim(**kwargs) -> ClaimStub:
    base = {
        "claim_id": "t",
        "procedure_codes": ["EVAL-4"],
        "diagnosis_codes": ["DX-MALIG-LUNG"],
        "date_of_service": date(2024, 4, 1),
        "rendering_provider": "Provider",
        "attestation_present": True,
    }
    base.update(kwargs)
    return ClaimStub.model_validate(base)


# --- PB001 -----------------------------------------------------------------
def test_pb001_fires_when_support_field_absent():
    ext = _extraction(biomarkers=[])
    claim = _claim(procedure_codes=["MOL-PANEL"])
    findings = check_code_documentation_support(ext, claim)
    assert [f.check_id for f in findings] == ["PB001"]
    assert findings[0].severity == Severity.HIGH


def test_pb001_silent_when_support_present():
    ext = _extraction()
    claim = _claim(procedure_codes=["MOL-PANEL"])
    assert check_code_documentation_support(ext, claim) == []


# --- PB002 -----------------------------------------------------------------
def test_pb002_fires_on_missing_required_elements():
    ext = _extraction(stage=None, date_of_diagnosis=None)
    claim = _claim(procedure_codes=["EVAL-4"])
    findings = check_missing_required_elements(ext, claim)
    refs = {r for f in findings for r in f.field_refs}
    assert {"stage", "date_of_diagnosis"} <= refs
    assert all(f.check_id == "PB002" for f in findings)


def test_pb002_silent_without_eval_code():
    ext = _extraction(stage=None, date_of_diagnosis=None)
    claim = _claim(procedure_codes=["PROC-BX"])
    assert check_missing_required_elements(ext, claim) == []


# --- PB003 -----------------------------------------------------------------
def test_pb003_service_before_diagnosis():
    ext = _extraction(date_of_diagnosis="2024-06-15")
    claim = _claim(procedure_codes=["PROC-BX"], date_of_service=date(2024, 6, 1))
    findings = check_field_conflicts(ext, claim)
    assert any(f.severity == Severity.HIGH for f in findings)
    assert all(f.check_id == "PB003" for f in findings)


def test_pb003_second_line_no_regimen():
    ext = _extraction(line_of_therapy=3, treatment_regimen=[])
    claim = _claim(procedure_codes=["PROC-BX"])
    findings = check_field_conflicts(ext, claim)
    assert any("line_of_therapy" in f.field_refs for f in findings)


def test_pb003_biomarker_contradiction():
    ext = _extraction(
        biomarkers=[
            {"name": "HER2", "status": "positive"},
            {"name": "HER2", "status": "negative"},
        ]
    )
    claim = _claim(procedure_codes=["PROC-BX"])
    findings = check_field_conflicts(ext, claim)
    assert any("biomarkers" in f.field_refs for f in findings)


def test_pb003_silent_when_consistent():
    ext = _extraction()
    claim = _claim(procedure_codes=["PROC-BX"])
    assert check_field_conflicts(ext, claim) == []


# --- PB004 -----------------------------------------------------------------
def test_pb004_fires_when_level_unsupported():
    ext = _extraction(
        histology=None,
        stage=None,
        biomarkers=[],
        ecog_performance_status=None,
        line_of_therapy=None,
        date_of_diagnosis=None,
        treatment_regimen=[],
    )
    claim = _claim(procedure_codes=["EVAL-5"])
    findings = check_documentation_level_mismatch(ext, claim)
    assert [f.check_id for f in findings] == ["PB004"]


def test_pb004_silent_when_richness_sufficient():
    ext = _extraction()
    claim = _claim(procedure_codes=["EVAL-3"])
    assert check_documentation_level_mismatch(ext, claim) == []


# --- PB005 -----------------------------------------------------------------
def test_pb005_fires_on_benign_dx_with_systemic():
    ext = _extraction()
    claim = _claim(procedure_codes=["PROC-INF"], diagnosis_codes=["DX-BENIGN-NODULE"])
    findings = check_diagnosis_procedure_consistency(ext, claim)
    assert [f.check_id for f in findings] == ["PB005"]
    assert findings[0].severity == Severity.HIGH


def test_pb005_silent_with_malignant_dx():
    ext = _extraction()
    claim = _claim(procedure_codes=["PROC-INF"], diagnosis_codes=["DX-MALIG-LUNG"])
    assert check_diagnosis_procedure_consistency(ext, claim) == []


# --- PB006 -----------------------------------------------------------------
def test_pb006_duplicate_and_exclusive():
    ext = _extraction()
    claim = _claim(procedure_codes=["EVAL-4", "EVAL-4", "EVAL-5"])
    findings = check_duplicate_or_overlapping_codes(ext, claim)
    assert len(findings) == 2
    assert all(f.check_id == "PB006" for f in findings)


def test_pb006_silent_on_distinct_codes():
    ext = _extraction()
    claim = _claim(procedure_codes=["EVAL-4", "PROC-BX"])
    assert check_duplicate_or_overlapping_codes(ext, claim) == []


# --- PB007 -----------------------------------------------------------------
def test_pb007_fires_on_missing_attestation_and_provider():
    ext = _extraction()
    claim = _claim(attestation_present=False, rendering_provider="")
    findings = check_missing_attestation(ext, claim)
    assert len(findings) == 2
    assert any(f.severity == Severity.HIGH for f in findings)


def test_pb007_silent_when_present():
    ext = _extraction()
    claim = _claim(attestation_present=True, rendering_provider="Dr. X")
    assert check_missing_attestation(ext, claim) == []


# --- PB008 -----------------------------------------------------------------
def test_pb008_fires_on_low_confidence_support():
    ext = _extraction(
        fields={"stage": {"confidence": 0.4, "needs_review": True}},
        needs_review=["stage"],
    )
    claim = _claim(procedure_codes=["EVAL-4"])
    findings = check_low_confidence_support(ext, claim)
    assert [f.check_id for f in findings] == ["PB008"]
    assert findings[0].severity == Severity.LOW


def test_pb008_silent_when_confidence_ok():
    ext = _extraction()
    claim = _claim(procedure_codes=["EVAL-4"])
    assert check_low_confidence_support(ext, claim) == []


# --- aggregation -----------------------------------------------------------
def test_report_ready_when_clean():
    ext = _extraction()
    claim = _claim(procedure_codes=["EVAL-4"])
    report = build_report(ext, claim)
    assert report.readiness == READY
    assert report.penalty == 0
    assert report.findings == []


def test_report_hold_on_high_severity():
    ext = _extraction()
    claim = _claim(procedure_codes=["PROC-INF"], diagnosis_codes=["DX-BENIGN-NODULE"])
    report = build_report(ext, claim)
    assert report.readiness == HOLD
    assert report.penalty >= int(Severity.HIGH)


def test_report_review_on_medium_only():
    ext = _extraction(
        biomarkers=[
            {"name": "HER2", "status": "positive"},
            {"name": "HER2", "status": "negative"},
        ]
    )
    claim = _claim(procedure_codes=["EVAL-3"])
    report = build_report(ext, claim)
    assert report.readiness == REVIEW


def test_penalty_is_sum_of_severity_weights():
    ext = _extraction()
    claim = _claim(attestation_present=False, rendering_provider="")
    report = build_report(ext, claim)
    assert report.penalty == sum(int(f.severity) for f in report.findings)


def test_run_checks_is_deterministic():
    ext = _extraction(biomarkers=[])
    claim = _claim(procedure_codes=["MOL-PANEL"])
    first = [(f.check_id, f.rationale) for f in run_checks(ext, claim)]
    second = [(f.check_id, f.rationale) for f in run_checks(ext, claim)]
    assert first == second
