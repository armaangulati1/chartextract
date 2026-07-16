"""Finding and severity types for the pre-bill review demo."""

from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field


class Severity(IntEnum):
    """Ordered severity for a pre-bill finding (higher = more blocking).

    The integer values double as the transparent penalty weight used by the
    claim-readiness aggregation (see report.py).
    """

    INFO = 1
    LOW = 2
    MEDIUM = 4
    HIGH = 8


class Finding(BaseModel):
    """A single deterministic pre-bill flag."""

    check_id: str = Field(description="stable id of the check that produced this")
    severity: Severity = Field(description="ordered severity of the finding")
    field_refs: list[str] = Field(
        default_factory=list,
        description="extract fields and/or claim codes the finding points at",
    )
    rationale: str = Field(description="human-readable explanation of the flag")


__all__ = ["Severity", "Finding"]
