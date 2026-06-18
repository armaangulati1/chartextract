from datetime import date

from pipeline import (
    PipelineState,
    RoutePlan,
    _build_extract,
    _coerce_stage,
    _dedupe_biomarkers,
    validator,
)
from schema import Biomarker, BiomarkerStatus, CancerStage, EcogPerformanceStatus, OncologyExtract
from pipeline import FieldCandidate


def test_build_extract_from_candidates():
    state = PipelineState(note="x")
    state.candidates = {
        "primary_site": FieldCandidate("lung", 0.9, source="tumor_extractor"),
        "histology": FieldCandidate("adenocarcinoma", 0.9, source="tumor_extractor"),
        "stage": FieldCandidate(CancerStage.IIIA, 0.9, source="tumor_extractor"),
        "line_of_therapy": FieldCandidate(1, 0.9, source="clinical_extractor"),
    }
    record = _build_extract(state)
    assert record.primary_site == "lung"
    assert record.stage == CancerStage.IIIA


def test_coerce_stage():
    assert _coerce_stage("stage iiia") == CancerStage.IIIA


def test_dedupe_biomarkers():
    items = [
        Biomarker(name="EGFR", status=BiomarkerStatus.POSITIVE),
        Biomarker(name="egfr", status=BiomarkerStatus.POSITIVE),
    ]
    deduped = _dedupe_biomarkers(items)
    assert len(deduped) == 1


def test_validator_produces_oncology_extract():
    state = PipelineState(note="synthetic")
    state.candidates = {
        "primary_site": FieldCandidate("breast", 0.9),
        "histology": FieldCandidate("ductal carcinoma", 0.9),
        "stage": FieldCandidate("II", 0.9),
        "ecog_performance_status": FieldCandidate(EcogPerformanceStatus.RESTRICTED_STRENUOUS, 0.9),
        "line_of_therapy": FieldCandidate(2, 0.9),
        "date_of_diagnosis": FieldCandidate(date(2021, 1, 15), 0.9),
        "biomarkers": FieldCandidate(
            [Biomarker(name="HER2", status=BiomarkerStatus.POSITIVE)], 0.9
        ),
        "treatment_regimen": FieldCandidate(["trastuzumab"], 0.9),
    }
    state = validator(state)
    assert isinstance(state.result, OncologyExtract)
    assert state.result.primary_site == "breast"
    assert state.result.line_of_therapy == 2


def test_default_route_plan_runs_all_groups():
    plan = RoutePlan()
    assert plan.run_tumor and plan.run_clinical and plan.run_molecular and plan.run_treatment
