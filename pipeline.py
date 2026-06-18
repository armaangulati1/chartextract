"""Agentic extraction pipeline: router → extractors → validator → verifier."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Optional

import instructor
from dotenv import load_dotenv
from langfuse import observe
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field

from schema import (
    Biomarker,
    BiomarkerStatus,
    CancerStage,
    EcogPerformanceStatus,
    ExtractionOutput,
    FieldMeta,
    OncologyExtract,
    TokenUsage,
)

load_dotenv()

CHAT_MODEL = "gpt-4o-mini"
VERIFIER_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_EXTRACTOR_CONFIDENCE = 0.85
REVIEW_CONFIDENCE_THRESHOLD = float(os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "0.75"))
FLAG_CONFIDENCE_PENALTY = 0.1

EXTRACT_FIELD_NAMES = (
    "primary_site",
    "histology",
    "stage",
    "ecog_performance_status",
    "line_of_therapy",
    "date_of_diagnosis",
    "biomarkers",
    "treatment_regimen",
)

SINGLE_PASS_PROMPT = (
    "Extract structured oncology variables from the clinical note into an OncologyExtract record. "
    "Only use information actually stated in the text; use null for absent scalar fields and "
    "empty lists for absent list fields. "
    "Use schema enums for stage (AJCC I–IV with substages), ECOG (0–4), and biomarker status. "
    "Do not invent fields or values."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = instructor.from_openai(OpenAI(timeout=90.0, max_retries=3))
    return _client


# --- Router ---


class RoutePlan(BaseModel):
    """Which extractor groups to run based on note content."""

    run_tumor: bool = Field(default=True, description="primary site, histology, stage")
    run_clinical: bool = Field(default=True, description="ECOG, line of therapy, diagnosis date")
    run_molecular: bool = Field(default=True, description="biomarker panel results")
    run_treatment: bool = Field(default=True, description="current treatment regimen / drugs")
    note_sections: list[str] = Field(
        default_factory=list,
        description="detected note sections, e.g. pathology, treatment, labs",
    )


# --- Extractor sub-schemas ---


class TumorProfile(BaseModel):
    primary_site: Optional[str] = Field(None, description="anatomic primary tumor site")
    histology: Optional[str] = Field(None, description="tumor histology / cell type")
    stage: Optional[CancerStage] = Field(None, description="AJCC stage when stated")


class ClinicalStatus(BaseModel):
    ecog_performance_status: Optional[EcogPerformanceStatus] = Field(
        None, description="ECOG 0–4 when documented"
    )
    line_of_therapy: Optional[int] = Field(None, ge=1, description="1=first-line, etc.")
    date_of_diagnosis: Optional[date] = Field(None, description="diagnosis date when stated")


class MolecularProfile(BaseModel):
    biomarkers: list[Biomarker] = Field(default_factory=list)


class TreatmentPlan(BaseModel):
    treatment_regimen: list[str] = Field(
        default_factory=list,
        description="drug names in current/documented regimen",
    )


# --- Pipeline state ---


@dataclass
class FieldCandidate:
    value: Any
    confidence: float
    evidence: str = ""
    source: str = ""


@dataclass
class PipelineState:
    note: str
    model: str = CHAT_MODEL
    route: RoutePlan | None = None
    candidates: dict[str, FieldCandidate] = field(default_factory=dict)
    flags: dict[str, list[str]] = field(default_factory=dict)
    result: OncologyExtract | None = None
    steps: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def log(self, step: str) -> None:
        self.steps.append(step)


# --- Verifier ---


class ScalarVerification(BaseModel):
    confirmed: bool = Field(description="value is supported by the note")
    value: Optional[str] = Field(None, description="corrected scalar as string, or null if absent")
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = Field(default="", description="short quote from the note")


class BiomarkerVerification(BaseModel):
    confirmed: bool
    biomarkers: list[Biomarker] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


class RegimenVerification(BaseModel):
    confirmed: bool
    treatment_regimen: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str = ""


def _accumulate_usage(state: PipelineState, completion: Any) -> None:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return
    state.prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
    state.completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
    state.total_tokens += int(getattr(usage, "total_tokens", 0) or 0)


def _llm_create(
    response_model,
    system: str,
    user: str,
    model: str | None = None,
    state: PipelineState | None = None,
):
    result, completion = _get_client().chat.completions.create_with_completion(
        model=model or CHAT_MODEL,
        response_model=response_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    if state is not None:
        _accumulate_usage(state, completion)
    return result


@observe()
def router(state: PipelineState) -> PipelineState:
    plan = _llm_create(
        RoutePlan,
        (
            "You route oncology note extraction. Inspect the note and decide which "
            "extractor groups are needed. Enable a group only if the note may contain "
            "that information. Always enable run_tumor for oncology notes."
        ),
        state.note,
        model=state.model,
        state=state,
    )
    state.route = plan
    state.log(f"router: tumor={plan.run_tumor} clinical={plan.run_clinical} "
              f"molecular={plan.run_molecular} treatment={plan.run_treatment}")
    return state


def _set_candidate(
    state: PipelineState,
    field_name: str,
    value: Any,
    source: str,
    confidence: float = DEFAULT_EXTRACTOR_CONFIDENCE,
) -> None:
    if value is None or value == "" or value == []:
        return
    state.candidates[field_name] = FieldCandidate(
        value=value,
        confidence=confidence,
        source=source,
    )


@observe()
def extractors(state: PipelineState) -> PipelineState:
    route = state.route or RoutePlan(
        run_tumor=True, run_clinical=True, run_molecular=True, run_treatment=True
    )
    note = state.note
    model = state.model

    if route.run_tumor:
        tumor = _llm_create(
            TumorProfile,
            (
                "Extract tumor profile fields from the oncology note. "
                "Only state what is explicitly documented; use null for absent fields."
            ),
            note,
            model=model,
            state=state,
        )
        _set_candidate(state, "primary_site", tumor.primary_site, "tumor_extractor")
        _set_candidate(state, "histology", tumor.histology, "tumor_extractor")
        _set_candidate(state, "stage", tumor.stage, "tumor_extractor")
        state.log("extractors: tumor profile")

    if route.run_clinical:
        clinical = _llm_create(
            ClinicalStatus,
            (
                "Extract clinical status fields from the oncology note. "
                "ECOG must be 0–4. Line of therapy is 1 for first-line, etc. "
                "Use null when not stated."
            ),
            note,
            model=model,
            state=state,
        )
        _set_candidate(state, "ecog_performance_status", clinical.ecog_performance_status, "clinical_extractor")
        _set_candidate(state, "line_of_therapy", clinical.line_of_therapy, "clinical_extractor")
        _set_candidate(state, "date_of_diagnosis", clinical.date_of_diagnosis, "clinical_extractor")
        state.log("extractors: clinical status")

    if route.run_molecular:
        molecular = _llm_create(
            MolecularProfile,
            (
                "Extract biomarker results from the oncology note. "
                "Use schema status values: positive, negative, equivocal, unknown. "
                "Return an empty list if none are stated."
            ),
            note,
            model=model,
            state=state,
        )
        if molecular.biomarkers:
            _set_candidate(state, "biomarkers", molecular.biomarkers, "molecular_extractor")
        state.log("extractors: molecular profile")

    if route.run_treatment:
        treatment = _llm_create(
            TreatmentPlan,
            (
                "Extract drug names in the current or documented treatment regimen. "
                "Use generic drug names. Return an empty list if none stated."
            ),
            note,
            model=model,
            state=state,
        )
        if treatment.treatment_regimen:
            _set_candidate(state, "treatment_regimen", treatment.treatment_regimen, "treatment_extractor")
        state.log("extractors: treatment plan")

    return state


def _coerce_stage(value: Any) -> Optional[CancerStage]:
    if value is None:
        return None
    if isinstance(value, CancerStage):
        return value
    try:
        return CancerStage(str(value).strip().upper().replace("STAGE ", ""))
    except ValueError:
        return None


def _coerce_ecog(value: Any) -> Optional[EcogPerformanceStatus]:
    if value is None:
        return None
    if isinstance(value, EcogPerformanceStatus):
        return value
    try:
        return EcogPerformanceStatus(int(value))
    except (ValueError, TypeError):
        return None


def _dedupe_biomarkers(items: list[Biomarker]) -> list[Biomarker]:
    seen: dict[str, Biomarker] = {}
    for item in items:
        key = item.name.strip().upper()
        if key in seen and seen[key].status != item.status:
            return items  # keep original; validator will flag conflict
        seen[key] = item
    return list(seen.values())


def _field_has_value(name: str, value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _compute_field_confidence(cand: FieldCandidate, flags: list[str]) -> float:
    penalty = FLAG_CONFIDENCE_PENALTY * len(flags)
    return max(0.0, min(1.0, cand.confidence - penalty))


def build_extraction_output(
    state: PipelineState,
    *,
    review_threshold: float | None = None,
) -> ExtractionOutput:
    """Attach per-field confidence and needs_review flags from pipeline state."""
    threshold = REVIEW_CONFIDENCE_THRESHOLD if review_threshold is None else review_threshold
    record = state.result or _build_extract(state)
    fields: dict[str, FieldMeta] = {}
    needs_review: list[str] = []

    for name in EXTRACT_FIELD_NAMES:
        cand = state.candidates.get(name)
        flags = state.flags.get(name, [])
        if cand is None or not _field_has_value(name, cand.value):
            fields[name] = FieldMeta(confidence=1.0, needs_review=False, flags=flags)
            continue

        confidence = round(_compute_field_confidence(cand, flags), 3)
        flagged = confidence < threshold
        fields[name] = FieldMeta(
            confidence=confidence,
            needs_review=flagged,
            source=cand.source,
            evidence=cand.evidence,
            flags=flags,
        )
        if flagged:
            needs_review.append(name)

    return ExtractionOutput(
        extract=record,
        fields=fields,
        needs_review=needs_review,
        review_threshold=threshold,
        usage=TokenUsage(
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            total_tokens=state.total_tokens,
        ),
    )


def _wrap_single_pass_output(
    record: OncologyExtract,
    *,
    review_threshold: float | None = None,
) -> ExtractionOutput:
    """Assign default confidence to single-pass extraction (no verifier scores)."""
    threshold = REVIEW_CONFIDENCE_THRESHOLD if review_threshold is None else review_threshold
    fields: dict[str, FieldMeta] = {}
    needs_review: list[str] = []
    payload = record.model_dump(mode="json")

    for name in EXTRACT_FIELD_NAMES:
        value = payload.get(name)
        if not _field_has_value(name, value):
            fields[name] = FieldMeta(confidence=1.0, needs_review=False, source="single_pass")
            continue

        confidence = DEFAULT_EXTRACTOR_CONFIDENCE
        flagged = confidence < threshold
        fields[name] = FieldMeta(
            confidence=confidence,
            needs_review=flagged,
            source="single_pass",
            flags=[],
        )
        if flagged:
            needs_review.append(name)

    return ExtractionOutput(
        extract=record,
        fields=fields,
        needs_review=needs_review,
        review_threshold=threshold,
    )


def _build_extract(state: PipelineState) -> OncologyExtract:
    values: dict[str, Any] = {}
    for name in EXTRACT_FIELD_NAMES:
        if name in state.candidates:
            values[name] = state.candidates[name].value

    biomarkers = values.get("biomarkers") or []
    if biomarkers:
        biomarkers = _dedupe_biomarkers(biomarkers)

    return OncologyExtract(
        primary_site=values.get("primary_site"),
        histology=values.get("histology"),
        stage=_coerce_stage(values.get("stage")),
        biomarkers=biomarkers,
        ecog_performance_status=_coerce_ecog(values.get("ecog_performance_status")),
        line_of_therapy=values.get("line_of_therapy"),
        date_of_diagnosis=values.get("date_of_diagnosis"),
        treatment_regimen=values.get("treatment_regimen") or [],
    )


@observe()
def validator(state: PipelineState) -> PipelineState:
    # Pydantic + value-set validation via OncologyExtract
    try:
        record = _build_extract(state)
        state.result = OncologyExtract.model_validate(record.model_dump(mode="json"))
    except Exception as exc:
        state.flags.setdefault("_schema", []).append(str(exc))
        state.result = OncologyExtract()
        state.log(f"validator: schema_violation ({exc})")
        return state

    # Normalization drift checks (reuse eval normalizers)
    from eval import scalar_normalizer

    for field_name in (
        "primary_site", "histology", "stage",
        "ecog_performance_status", "line_of_therapy", "date_of_diagnosis",
    ):
        if field_name not in state.candidates:
            continue
        cand = state.candidates[field_name]
        norm_fn = scalar_normalizer(field_name)
        raw = cand.value
        normalized = norm_fn(raw)
        if raw is not None and normalized is not None and str(raw) != str(normalized):
            state.flags.setdefault(field_name, []).append("normalization_drift")
            cand.confidence = min(cand.confidence, 0.6)

    biomarkers = state.result.biomarkers
    names = [b.name.strip().upper() for b in biomarkers]
    if len(names) != len(set(names)):
        state.flags.setdefault("biomarkers", []).append("duplicate_name_conflict")
        if "biomarkers" in state.candidates:
            state.candidates["biomarkers"].confidence = min(
                state.candidates["biomarkers"].confidence, 0.55
            )

    state.log("validator: passed" if not state.flags else f"validator: flags={state.flags}")
    return state


def _needs_verification(state: PipelineState, field_name: str) -> bool:
    if field_name in state.flags:
        return True
    cand = state.candidates.get(field_name)
    return cand is not None and cand.confidence < VERIFIER_CONFIDENCE_THRESHOLD


@observe()
def verifier(state: PipelineState) -> PipelineState:
    note = state.note
    model = state.model
    scalar_fields = (
        "primary_site", "histology", "stage",
        "ecog_performance_status", "line_of_therapy", "date_of_diagnosis",
    )

    for field_name in scalar_fields:
        if not _needs_verification(state, field_name):
            continue
        current = state.candidates[field_name].value
        outcome = _llm_create(
            ScalarVerification,
            (
                f"Verify the extracted {field_name} against the clinical note. "
                "Confirm if supported, or provide the corrected value. "
                "If absent from the note, set value to null and confirmed=false."
            ),
            f"Note:\n{note}\n\nExtracted {field_name}: {current!r}",
            model=model,
            state=state,
        )
        if outcome.confirmed and outcome.value is not None:
            state.candidates[field_name] = FieldCandidate(
                value=outcome.value,
                confidence=outcome.confidence,
                evidence=outcome.evidence,
                source="verifier",
            )
        elif not outcome.confirmed:
            state.candidates.pop(field_name, None)
        state.log(f"verifier: {field_name} confirmed={outcome.confirmed}")

    if _needs_verification(state, "biomarkers"):
        current = state.candidates["biomarkers"].value
        outcome = _llm_create(
            BiomarkerVerification,
            (
                "Verify biomarker name/status pairs against the note. "
                "Return only biomarkers explicitly supported. "
                "Use status: positive, negative, equivocal, unknown."
            ),
            f"Note:\n{note}\n\nExtracted biomarkers: {current!r}",
            model=model,
            state=state,
        )
        if outcome.confirmed and outcome.biomarkers:
            state.candidates["biomarkers"] = FieldCandidate(
                value=outcome.biomarkers,
                confidence=outcome.confidence,
                evidence=outcome.evidence,
                source="verifier",
            )
        elif not outcome.confirmed:
            state.candidates.pop("biomarkers", None)
        state.log(f"verifier: biomarkers confirmed={outcome.confirmed}")

    if _needs_verification(state, "treatment_regimen"):
        current = state.candidates["treatment_regimen"].value
        outcome = _llm_create(
            RegimenVerification,
            (
                "Verify treatment regimen drug names against the note. "
                "Return only drugs explicitly part of the cancer treatment regimen."
            ),
            f"Note:\n{note}\n\nExtracted regimen: {current!r}",
            model=model,
            state=state,
        )
        if outcome.confirmed and outcome.treatment_regimen:
            state.candidates["treatment_regimen"] = FieldCandidate(
                value=outcome.treatment_regimen,
                confidence=outcome.confidence,
                evidence=outcome.evidence,
                source="verifier",
            )
        elif not outcome.confirmed:
            state.candidates.pop("treatment_regimen", None)
        state.log(f"verifier: treatment_regimen confirmed={outcome.confirmed}")

    state.result = _build_extract(state)
    state.log("verifier: complete")
    return state


@observe()
def single_pass_extract(note: str, model: str = CHAT_MODEL) -> OncologyExtract:
    """One-shot extraction baseline for experiments."""
    return _get_client().chat.completions.create(
        model=model,
        response_model=OncologyExtract,
        messages=[
            {"role": "system", "content": SINGLE_PASS_PROMPT},
            {"role": "user", "content": note},
        ],
    )


@observe()
def run_pipeline(
    note: str,
    *,
    model: str = CHAT_MODEL,
    use_verifier: bool = True,
    review_threshold: float | None = None,
) -> ExtractionOutput:
    """Pipeline entry: router → extractors → validator → [verifier]."""
    state = PipelineState(note=note, model=model)
    state = router(state)
    state = extractors(state)
    state = validator(state)
    if use_verifier:
        state = verifier(state)
    else:
        state.result = _build_extract(state)
        state.log("verifier: skipped")
    return build_extraction_output(state, review_threshold=review_threshold)


def make_extractor(
    mode: str = "pipeline",
    *,
    model: str = CHAT_MODEL,
    use_verifier: bool = True,
):
    """Factory for eval/experiment configs."""
    if mode == "single_pass":
        def extract(note: str) -> OncologyExtract:
            return single_pass_extract(note, model=model)
        return extract

    def extract(note: str) -> OncologyExtract:
        return run_pipeline(
            note, model=model, use_verifier=use_verifier
        ).extract
    return extract
