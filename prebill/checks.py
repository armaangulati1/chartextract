"""Deterministic pre-bill review checks (self-authored demonstration heuristics).

Each check is a pure function of (ExtractionOutput, ClaimStub) and returns a list
of Finding objects. No LLM calls, no network, no external state: the same inputs
always produce the same findings.

None of these encode real payer rules, NCCI edits, or CMS documentation
guidelines. They are teaching-example heuristics over a self-authored demo code
system (see claim.py).
"""

from __future__ import annotations

from typing import Callable

from prebill.claim import (
    CODE_REQUIRES_EXTRACT_FIELDS,
    EM_LEVEL_CODES,
    EM_LEVEL_MIN_ELEMENTS,
    DX_MALIGNANT_PREFIX,
    MUTUALLY_EXCLUSIVE_CODE_SETS,
    PROC_REQUIRES_MALIGNANT_DX,
    REQUIRED_ONC_EVAL_ELEMENTS,
    ClaimStub,
)
from prebill.findings import Finding, Severity
from schema import ExtractionOutput

# Core oncology documentation elements considered "documented" when present.
_CORE_ELEMENTS = (
    "primary_site",
    "histology",
    "stage",
    "biomarkers",
    "ecog_performance_status",
    "line_of_therapy",
    "date_of_diagnosis",
    "treatment_regimen",
)


def _documented_elements(extraction: ExtractionOutput) -> set[str]:
    """Return the set of core elements that carry a non-empty documented value."""
    ex = extraction.extract
    documented: set[str] = set()
    for name in _CORE_ELEMENTS:
        value = getattr(ex, name)
        if value is None:
            continue
        if isinstance(value, list) and len(value) == 0:
            continue
        documented.add(name)
    return documented


def _billed_em_levels(claim: ClaimStub) -> list[str]:
    return [c for c in claim.procedure_codes if c in EM_LEVEL_CODES]


def _has_field(extraction: ExtractionOutput, field: str) -> bool:
    return field in _documented_elements(extraction)


# ---------------------------------------------------------------------------
# PB001 — code / documentation support
# ---------------------------------------------------------------------------
def check_code_documentation_support(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """A billed code that presupposes a documented field is flagged when that
    field is absent from the extraction."""
    findings: list[Finding] = []
    for code in claim.procedure_codes:
        required = CODE_REQUIRES_EXTRACT_FIELDS.get(code)
        if not required:
            continue
        for field in required:
            if not _has_field(extraction, field):
                findings.append(
                    Finding(
                        check_id="PB001",
                        severity=Severity.HIGH,
                        field_refs=[code, field],
                        rationale=(
                            f"Billed code {code} presupposes documented "
                            f"'{field}', but it is absent from the extraction."
                        ),
                    )
                )
    return findings


# ---------------------------------------------------------------------------
# PB002 — missing required documentation elements for the claim type
# ---------------------------------------------------------------------------
def check_missing_required_elements(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """When an evaluation claim is billed, required oncology documentation
    elements that are missing are each flagged."""
    findings: list[Finding] = []
    if not _billed_em_levels(claim):
        return findings
    for element in REQUIRED_ONC_EVAL_ELEMENTS:
        if not _has_field(extraction, element):
            findings.append(
                Finding(
                    check_id="PB002",
                    severity=Severity.MEDIUM,
                    field_refs=[element],
                    rationale=(
                        f"Evaluation claim requires documented '{element}' "
                        f"for this demo claim type, but it is missing."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# PB003 — internal field conflicts / contradictions
# ---------------------------------------------------------------------------
def check_field_conflicts(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """Contradictions within the extraction, or between the extraction and the
    claim's date of service."""
    findings: list[Finding] = []
    ex = extraction.extract

    # (a) Service billed before the documented diagnosis date.
    if claim.date_of_service and ex.date_of_diagnosis:
        if claim.date_of_service < ex.date_of_diagnosis:
            findings.append(
                Finding(
                    check_id="PB003",
                    severity=Severity.HIGH,
                    field_refs=["date_of_service", "date_of_diagnosis"],
                    rationale=(
                        "Claim date_of_service precedes the documented "
                        "date_of_diagnosis, a temporal contradiction."
                    ),
                )
            )

    # (b) Second-or-later line of therapy with no documented regimen.
    if ex.line_of_therapy is not None and ex.line_of_therapy >= 2:
        if not ex.treatment_regimen:
            findings.append(
                Finding(
                    check_id="PB003",
                    severity=Severity.MEDIUM,
                    field_refs=["line_of_therapy", "treatment_regimen"],
                    rationale=(
                        f"line_of_therapy is {ex.line_of_therapy} but no "
                        "treatment_regimen is documented."
                    ),
                )
            )

    # (c) Same biomarker documented with conflicting statuses.
    seen: dict[str, set[str]] = {}
    for bm in ex.biomarkers:
        seen.setdefault(bm.name.lower(), set()).add(bm.status.value)
    for name, statuses in seen.items():
        contradictory = {"positive", "negative"}
        if contradictory.issubset(statuses):
            findings.append(
                Finding(
                    check_id="PB003",
                    severity=Severity.MEDIUM,
                    field_refs=["biomarkers", name],
                    rationale=(
                        f"Biomarker '{name}' is documented as both positive "
                        "and negative, a contradictory result."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# PB004 — visit-level vs documentation-richness mismatch
# ---------------------------------------------------------------------------
def check_documentation_level_mismatch(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """The billed evaluation level is flagged when the documented element count
    does not meet the self-authored threshold for that level."""
    findings: list[Finding] = []
    documented_count = len(_documented_elements(extraction))
    for code in _billed_em_levels(claim):
        required = EM_LEVEL_MIN_ELEMENTS.get(code)
        if required is None:
            continue
        if documented_count < required:
            findings.append(
                Finding(
                    check_id="PB004",
                    severity=Severity.MEDIUM,
                    field_refs=[code],
                    rationale=(
                        f"Billed level {code} expects at least {required} "
                        f"documented elements (demo threshold); only "
                        f"{documented_count} are documented. Consider a lower "
                        "level."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# PB005 — diagnosis / procedure consistency via self-authored crosswalk
# ---------------------------------------------------------------------------
def check_diagnosis_procedure_consistency(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """A procedure that presupposes a malignant diagnosis is flagged when no
    malignant-family diagnosis code is on the claim."""
    findings: list[Finding] = []
    has_malignant_dx = any(
        code.startswith(DX_MALIGNANT_PREFIX) for code in claim.diagnosis_codes
    )
    if has_malignant_dx:
        return findings
    for code in claim.procedure_codes:
        if code in PROC_REQUIRES_MALIGNANT_DX:
            findings.append(
                Finding(
                    check_id="PB005",
                    severity=Severity.HIGH,
                    field_refs=[code] + list(claim.diagnosis_codes),
                    rationale=(
                        f"Procedure {code} presupposes a malignant diagnosis "
                        "in this demo crosswalk, but no malignant-family "
                        "diagnosis code is on the claim."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# PB006 — duplicate / mutually-exclusive codes on the claim
# ---------------------------------------------------------------------------
def check_duplicate_or_overlapping_codes(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """Exact duplicate procedure codes, or more than one code from a
    mutually-exclusive set, are flagged."""
    findings: list[Finding] = []
    codes = claim.procedure_codes

    seen: set[str] = set()
    duplicates: list[str] = []
    for code in codes:
        if code in seen and code not in duplicates:
            duplicates.append(code)
        seen.add(code)
    for code in duplicates:
        findings.append(
            Finding(
                check_id="PB006",
                severity=Severity.MEDIUM,
                field_refs=[code],
                rationale=f"Code {code} is billed more than once on the claim.",
            )
        )

    for excl_set in MUTUALLY_EXCLUSIVE_CODE_SETS:
        present = sorted(c for c in set(codes) if c in excl_set)
        if len(present) > 1:
            findings.append(
                Finding(
                    check_id="PB006",
                    severity=Severity.MEDIUM,
                    field_refs=present,
                    rationale=(
                        "Multiple mutually-exclusive codes billed on one "
                        f"encounter: {', '.join(present)}."
                    ),
                )
            )
    return findings


# ---------------------------------------------------------------------------
# PB007 — missing attestation / rendering provider
# ---------------------------------------------------------------------------
def check_missing_attestation(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """Absent provider attestation or rendering-provider field is flagged."""
    findings: list[Finding] = []
    if not claim.attestation_present:
        findings.append(
            Finding(
                check_id="PB007",
                severity=Severity.HIGH,
                field_refs=["attestation_present"],
                rationale="No provider attestation/signature is recorded on the claim.",
            )
        )
    if not claim.rendering_provider.strip():
        findings.append(
            Finding(
                check_id="PB007",
                severity=Severity.MEDIUM,
                field_refs=["rendering_provider"],
                rationale="No rendering provider is recorded on the claim.",
            )
        )
    return findings


# ---------------------------------------------------------------------------
# PB008 — claim rests on low-confidence extraction (consumes review routing)
# ---------------------------------------------------------------------------
def check_low_confidence_support(
    extraction: ExtractionOutput, claim: ClaimStub
) -> list[Finding]:
    """A billed code whose supporting field was routed to human review by the
    extractor is flagged, so the claim is not built on unreviewed low-confidence
    extraction."""
    findings: list[Finding] = []

    def _routed_to_review(field: str) -> bool:
        if field in extraction.needs_review:
            return True
        meta = extraction.fields.get(field)
        if meta is None:
            return False
        return meta.needs_review or meta.confidence < extraction.review_threshold

    support_fields: set[str] = set()
    for code in claim.procedure_codes:
        support_fields.update(CODE_REQUIRES_EXTRACT_FIELDS.get(code, ()))
        if code in EM_LEVEL_CODES:
            support_fields.update(REQUIRED_ONC_EVAL_ELEMENTS)

    for field in sorted(support_fields):
        if _has_field(extraction, field) and _routed_to_review(field):
            findings.append(
                Finding(
                    check_id="PB008",
                    severity=Severity.LOW,
                    field_refs=[field],
                    rationale=(
                        f"Supporting field '{field}' was routed to human "
                        "review (low confidence); resolve before submitting."
                    ),
                )
            )
    return findings


CheckFn = Callable[[ExtractionOutput, ClaimStub], "list[Finding]"]

# (check_id, human name, function). Order is the report display order.
ALL_CHECKS: list[tuple[str, str, CheckFn]] = [
    ("PB001", "code_documentation_support", check_code_documentation_support),
    ("PB002", "missing_required_elements", check_missing_required_elements),
    ("PB003", "field_conflicts", check_field_conflicts),
    ("PB004", "documentation_level_mismatch", check_documentation_level_mismatch),
    ("PB005", "diagnosis_procedure_consistency", check_diagnosis_procedure_consistency),
    ("PB006", "duplicate_or_overlapping_codes", check_duplicate_or_overlapping_codes),
    ("PB007", "missing_attestation", check_missing_attestation),
    ("PB008", "low_confidence_support", check_low_confidence_support),
]

CHECK_IDS: tuple[str, ...] = tuple(cid for cid, _, _ in ALL_CHECKS)


def run_checks(extraction: ExtractionOutput, claim: ClaimStub) -> list[Finding]:
    """Run every check and return the concatenated, ordered findings."""
    findings: list[Finding] = []
    for _cid, _name, fn in ALL_CHECKS:
        findings.extend(fn(extraction, claim))
    return findings


__all__ = ["ALL_CHECKS", "CHECK_IDS", "run_checks"] + [
    fn.__name__ for _c, _n, fn in ALL_CHECKS
]
