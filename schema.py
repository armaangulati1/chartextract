"""Oncology extraction schema — the contract for structured chart variables."""

from __future__ import annotations

import json
from datetime import date
from enum import Enum, IntEnum
from typing import Optional

from pydantic import BaseModel, Field


class CancerStage(str, Enum):
    """AJCC-style stage value set (Roman numerals I–IV with common substages)."""

    I = "I"
    IA = "IA"
    IB = "IB"
    IC = "IC"
    II = "II"
    IIA = "IIA"
    IIB = "IIB"
    IIC = "IIC"
    III = "III"
    IIIA = "IIIA"
    IIIB = "IIIB"
    IIIC = "IIIC"
    IV = "IV"
    IVA = "IVA"
    IVB = "IVB"


class EcogPerformanceStatus(IntEnum):
    """ECOG performance status (0 = fully active … 4 = completely disabled)."""

    FULLY_ACTIVE = 0
    RESTRICTED_STRENUOUS = 1
    AMBULATORY = 2
    LIMITED_SELF_CARE = 3
    COMPLETELY_DISABLED = 4


class BiomarkerStatus(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EQUIVOCAL = "equivocal"
    UNKNOWN = "unknown"


class Biomarker(BaseModel):
    name: str = Field(description="biomarker name, e.g. EGFR, PD-L1, HER2")
    status: BiomarkerStatus = Field(description="test result for this biomarker")


class OncologyExtract(BaseModel):
    """Structured oncology variables extracted from clinical notes."""

    primary_site: Optional[str] = Field(
        None,
        description="anatomic primary tumor site, e.g. lung, breast, colon",
    )
    histology: Optional[str] = Field(
        None,
        description="tumor histology / cell type, e.g. adenocarcinoma",
    )
    stage: Optional[CancerStage] = Field(
        None,
        description="AJCC clinical or pathologic stage when stated",
    )
    biomarkers: list[Biomarker] = Field(
        default_factory=list,
        description="molecular biomarkers and their results",
    )
    ecog_performance_status: Optional[EcogPerformanceStatus] = Field(
        None,
        description="ECOG performance status 0–4 when documented",
    )
    line_of_therapy: Optional[int] = Field(
        None,
        ge=1,
        description="line of therapy: 1 = first-line, 2 = second-line, etc.",
    )
    date_of_diagnosis: Optional[date] = Field(
        None,
        description="date of cancer diagnosis when stated",
    )
    treatment_regimen: list[str] = Field(
        default_factory=list,
        description="drug names in the current or documented treatment regimen",
    )


def export_json_schema() -> dict:
    """Return the JSON Schema for OncologyExtract (for docs, eval harnesses, UI)."""
    return OncologyExtract.model_json_schema()


if __name__ == "__main__":
    print(json.dumps(export_json_schema(), indent=2))
