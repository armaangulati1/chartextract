"""Billing-claim stub schema and self-authored demo code system.

Everything in this file is a self-authored teaching example. The "codes" below
are invented placeholders (EVAL-*, PROC-*, MOL-*, IMG-*, DX-*) chosen precisely
so they do NOT collide with any real CPT, HCPCS, or ICD code. The crosswalks and
thresholds are demonstration heuristics, not real payer or coding rules.
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Self-authored demo code system (NOT real CPT/HCPCS/ICD).
# ---------------------------------------------------------------------------

# Evaluation / visit-complexity level analogs. Higher number = more complex
# visit that a payer would expect richer documentation to support.
EM_LEVEL_CODES = ("EVAL-2", "EVAL-3", "EVAL-4", "EVAL-5")

# Procedure analogs.
PROC_BIOPSY = "PROC-BX"          # tissue biopsy
PROC_SYSTEMIC = "PROC-INF"       # systemic-therapy administration
PROC_MOL_PANEL = "MOL-PANEL"     # molecular biomarker panel
PROC_IMAGING = "IMG-CT"          # cross-sectional imaging

# Diagnosis-code family prefixes (self-authored).
DX_MALIGNANT_PREFIX = "DX-MALIG"     # e.g. DX-MALIG-LUNG
DX_BENIGN_PREFIX = "DX-BENIGN"       # e.g. DX-BENIGN-NODULE
DX_SCREENING_PREFIX = "DX-SCREEN"    # e.g. DX-SCREEN

# ---------------------------------------------------------------------------
# Self-authored demonstration crosswalks / thresholds.
# ---------------------------------------------------------------------------

# A billed code that requires specific supporting extracted fields to be present.
# Demonstration heuristic only.
CODE_REQUIRES_EXTRACT_FIELDS: dict[str, tuple[str, ...]] = {
    PROC_MOL_PANEL: ("biomarkers",),
    PROC_SYSTEMIC: ("treatment_regimen",),
    PROC_BIOPSY: ("primary_site",),
}

# Minimum count of documented core elements a visit level is expected to carry.
# Self-authored thresholds; higher level demands more documented elements.
EM_LEVEL_MIN_ELEMENTS: dict[str, int] = {
    "EVAL-2": 1,
    "EVAL-3": 2,
    "EVAL-4": 4,
    "EVAL-5": 6,
}

# Documentation elements that an oncology evaluation claim is expected to carry.
REQUIRED_ONC_EVAL_ELEMENTS: tuple[str, ...] = (
    "primary_site",
    "stage",
    "date_of_diagnosis",
)

# Procedures that are only consistent with a malignant-diagnosis family in this
# demo crosswalk (systemic therapy / molecular panels presuppose a malignancy).
PROC_REQUIRES_MALIGNANT_DX: tuple[str, ...] = (PROC_SYSTEMIC, PROC_MOL_PANEL)

# Sets of codes that should not co-occur on a single encounter claim (only one
# evaluation level may be billed per encounter). Self-authored.
MUTUALLY_EXCLUSIVE_CODE_SETS: tuple[frozenset[str], ...] = (
    frozenset(EM_LEVEL_CODES),
)


class ClaimStub(BaseModel):
    """Minimal billing-claim stub paired with a chart extraction.

    A deliberately small, self-defined schema: enough claim context to run
    demonstration pre-bill checks, nothing resembling a real 837/claim form.
    """

    claim_id: str = Field(description="local identifier for the claim")
    procedure_codes: list[str] = Field(
        default_factory=list,
        description="self-authored demo procedure/visit codes billed on the claim",
    )
    diagnosis_codes: list[str] = Field(
        default_factory=list,
        description="self-authored demo diagnosis codes linked to the claim",
    )
    date_of_service: Optional[date] = Field(
        None, description="date the billed service was rendered"
    )
    rendering_provider: str = Field(
        default="", description="name/id of the rendering provider on the claim"
    )
    attestation_present: bool = Field(
        default=False,
        description="whether a provider attestation/signature is recorded",
    )


__all__ = [
    "ClaimStub",
    "EM_LEVEL_CODES",
    "PROC_BIOPSY",
    "PROC_SYSTEMIC",
    "PROC_MOL_PANEL",
    "PROC_IMAGING",
    "DX_MALIGNANT_PREFIX",
    "DX_BENIGN_PREFIX",
    "DX_SCREENING_PREFIX",
    "CODE_REQUIRES_EXTRACT_FIELDS",
    "EM_LEVEL_MIN_ELEMENTS",
    "REQUIRED_ONC_EVAL_ELEMENTS",
    "PROC_REQUIRES_MALIGNANT_DX",
    "MUTUALLY_EXCLUSIVE_CODE_SETS",
]
