"""Pre-bill chart-review flag layer (demonstration).

A deterministic, self-authored set of pre-bill review heuristics that consume
ChartExtractor's structured extraction output plus a small billing-claim stub
and flag problems before a claim would be submitted.

This is a GENERIC demonstration analog of pre-bill review workflows. The codes,
crosswalks, thresholds, and rules here are all self-authored teaching examples.
They are NOT real payer rules, NCCI edits, CMS documentation guidelines, or any
proprietary rule content, and this module is not affiliated with any company.
"""

from __future__ import annotations

from prebill.checks import ALL_CHECKS, run_checks
from prebill.claim import ClaimStub
from prebill.findings import Finding, Severity
from prebill.report import ClaimReadinessReport, build_report

__all__ = [
    "ALL_CHECKS",
    "run_checks",
    "ClaimStub",
    "Finding",
    "Severity",
    "ClaimReadinessReport",
    "build_report",
]
