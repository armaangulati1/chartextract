"""Transparent claim-readiness aggregation for the pre-bill review demo.

The aggregation is intentionally simple and fully documented: no opaque model
score. The penalty is a plain sum of severity weights (Severity int values), and
the readiness label is a documented function of the highest-severity finding.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from prebill.checks import run_checks
from prebill.claim import ClaimStub
from prebill.findings import Finding, Severity
from schema import ExtractionOutput

# Readiness labels, from best to worst.
READY = "ready"          # no findings
REVIEW = "needs_review"  # at most MEDIUM findings
HOLD = "hold"            # at least one HIGH finding


class ClaimReadinessReport(BaseModel):
    """Aggregated, transparent view of all findings for one claim."""

    claim_id: str
    readiness: str = Field(description="ready | needs_review | hold")
    penalty: int = Field(
        description="sum of severity weights across findings (documented, not opaque)"
    )
    severity_counts: dict[str, int] = Field(
        default_factory=dict, description="count of findings by severity name"
    )
    findings: list[Finding] = Field(default_factory=list)

    def summary_line(self) -> str:
        return (
            f"[{self.claim_id}] {self.readiness.upper()} "
            f"(penalty={self.penalty}, findings={len(self.findings)})"
        )


def _readiness_label(findings: list[Finding]) -> str:
    if not findings:
        return READY
    if any(f.severity == Severity.HIGH for f in findings):
        return HOLD
    return REVIEW


def build_report(
    extraction: ExtractionOutput, claim: ClaimStub
) -> ClaimReadinessReport:
    """Run all checks and aggregate them into a transparent readiness report."""
    findings = run_checks(extraction, claim)
    penalty = sum(int(f.severity) for f in findings)
    counts: dict[str, int] = {}
    for f in findings:
        counts[f.severity.name] = counts.get(f.severity.name, 0) + 1
    return ClaimReadinessReport(
        claim_id=claim.claim_id,
        readiness=_readiness_label(findings),
        penalty=penalty,
        severity_counts=counts,
        findings=findings,
    )


__all__ = ["ClaimReadinessReport", "build_report", "READY", "REVIEW", "HOLD"]
