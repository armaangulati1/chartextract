from datetime import date

from pipeline import (
    FieldCandidate,
    PipelineState,
    REVIEW_CONFIDENCE_THRESHOLD,
    _build_extract,
    build_extraction_output,
    validator,
)
from schema import Biomarker, BiomarkerStatus, CancerStage, EcogPerformanceStatus, ExtractionOutput


def test_build_extraction_output_flags_low_confidence():
    state = PipelineState(note="note")
    state.candidates = {
        "primary_site": FieldCandidate("lung", 0.9, source="tumor_extractor"),
        "stage": FieldCandidate(CancerStage.IIIA, 0.55, source="tumor_extractor"),
    }
    state.result = _build_extract(state)

    output = build_extraction_output(state, review_threshold=0.75)

    assert isinstance(output, ExtractionOutput)
    assert "stage" in output.needs_review
    assert "primary_site" not in output.needs_review
    assert output.fields["stage"].needs_review is True
    assert output.fields["stage"].confidence == 0.55
    assert output.fields["primary_site"].confidence == 0.9


def test_validator_penalty_routes_to_review():
    state = PipelineState(note="synthetic")
    state.candidates = {
        "primary_site": FieldCandidate("Lung", 0.85, source="tumor_extractor"),
        "histology": FieldCandidate("adenocarcinoma", 0.9, source="tumor_extractor"),
        "stage": FieldCandidate(CancerStage.II, 0.9, source="tumor_extractor"),
        "ecog_performance_status": FieldCandidate(EcogPerformanceStatus.FULLY_ACTIVE, 0.9),
        "line_of_therapy": FieldCandidate(1, 0.9),
        "date_of_diagnosis": FieldCandidate(date(2020, 5, 1), 0.9),
        "biomarkers": FieldCandidate(
            [Biomarker(name="EGFR", status=BiomarkerStatus.NEGATIVE)], 0.9
        ),
        "treatment_regimen": FieldCandidate(["pembrolizumab"], 0.9),
    }
    state = validator(state)

    output = build_extraction_output(state, review_threshold=0.75)
    assert "primary_site" in output.needs_review
    assert output.fields["primary_site"].confidence <= 0.6
    assert "normalization_drift" in output.fields["primary_site"].flags


def test_absent_fields_do_not_need_review():
    state = PipelineState(note="sparse note")
    state.result = _build_extract(state)
    output = build_extraction_output(state)

    assert output.needs_review == []
    for name, meta in output.fields.items():
        assert meta.needs_review is False
    assert output.review_threshold == REVIEW_CONFIDENCE_THRESHOLD
