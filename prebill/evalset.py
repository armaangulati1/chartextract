"""Exact-match eval harness for the pre-bill review demo.

Loads the self-authored synthetic fixtures, runs every check, and computes
per-check precision / recall against the golden files. Evaluation granularity is
(fixture, check_id): for each fixture a check either fires (produces >= 1
finding) or it does not, and that boolean is compared to the golden set of
expected check ids. Fully offline and deterministic.

Reported numbers are always scoped to this fixture set (see README). They measure
agreement between the checks and their own self-authored goldens, not real-world
billing accuracy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import NamedTuple

from prebill.checks import CHECK_IDS, run_checks
from prebill.claim import ClaimStub
from schema import ExtractionOutput

FIXTURE_DIR = Path(__file__).parent / "fixtures"
GOLDEN_DIR = FIXTURE_DIR / "goldens"


class Fixture(NamedTuple):
    fixture_id: str
    extraction: ExtractionOutput
    claim: ClaimStub
    expected_check_ids: set[str]


class CheckScore(NamedTuple):
    check_id: str
    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return 1.0 if denom == 0 else self.tp / denom

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def load_fixtures(fixture_dir: Path = FIXTURE_DIR) -> list[Fixture]:
    """Load and validate every fixture + its golden file."""
    fixtures: list[Fixture] = []
    for path in sorted(fixture_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        fixture_id = raw["fixture_id"]
        golden_path = fixture_dir / "goldens" / f"{fixture_id}.json"
        golden = json.loads(golden_path.read_text())
        if golden["fixture_id"] != fixture_id:
            raise ValueError(
                f"golden id mismatch for {fixture_id}: {golden['fixture_id']}"
            )
        expected = set(golden["expected_check_ids"])
        unknown = expected - set(CHECK_IDS)
        if unknown:
            raise ValueError(f"{fixture_id} golden references unknown checks {unknown}")
        fixtures.append(
            Fixture(
                fixture_id=fixture_id,
                extraction=ExtractionOutput.model_validate(raw["extraction"]),
                claim=ClaimStub.model_validate(raw["claim"]),
                expected_check_ids=expected,
            )
        )
    return fixtures


def fired_check_ids(fixture: Fixture) -> set[str]:
    """Set of check ids that produced at least one finding on this fixture."""
    return {f.check_id for f in run_checks(fixture.extraction, fixture.claim)}


def score_checks(fixtures: list[Fixture]) -> list[CheckScore]:
    """Per-check TP/FP/FN across all fixtures."""
    scores: list[CheckScore] = []
    for check_id in CHECK_IDS:
        tp = fp = fn = 0
        for fx in fixtures:
            fired = check_id in fired_check_ids(fx)
            expected = check_id in fx.expected_check_ids
            if fired and expected:
                tp += 1
            elif fired and not expected:
                fp += 1
            elif not fired and expected:
                fn += 1
        scores.append(CheckScore(check_id, tp, fp, fn))
    return scores


def micro_totals(scores: list[CheckScore]) -> CheckScore:
    return CheckScore(
        "MICRO",
        sum(s.tp for s in scores),
        sum(s.fp for s in scores),
        sum(s.fn for s in scores),
    )


def macro_f1(scores: list[CheckScore]) -> float:
    return sum(s.f1 for s in scores) / len(scores) if scores else 0.0


def format_table(fixtures: list[Fixture], scores: list[CheckScore]) -> str:
    lines: list[str] = []
    n = len(fixtures)
    lines.append(f"Pre-bill review eval  |  {n} self-authored synthetic fixtures")
    lines.append("-" * 60)
    lines.append(f"{'check':<8}{'TP':>4}{'FP':>4}{'FN':>4}{'prec':>8}{'rec':>8}{'f1':>8}")
    for s in scores:
        lines.append(
            f"{s.check_id:<8}{s.tp:>4}{s.fp:>4}{s.fn:>4}"
            f"{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>8.3f}"
        )
    lines.append("-" * 60)
    micro = micro_totals(scores)
    lines.append(
        f"{'MICRO':<8}{micro.tp:>4}{micro.fp:>4}{micro.fn:>4}"
        f"{micro.precision:>8.3f}{micro.recall:>8.3f}{micro.f1:>8.3f}"
    )
    lines.append(f"{'MACRO-F1':<8}{macro_f1(scores):>36.3f}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pre-bill review eval.")
    parser.add_argument(
        "--min-macro-f1",
        type=float,
        default=None,
        help="fail (exit 1) if macro-F1 falls below this threshold",
    )
    args = parser.parse_args()

    fixtures = load_fixtures()
    scores = score_checks(fixtures)
    print(format_table(fixtures, scores))

    if args.min_macro_f1 is not None:
        achieved = macro_f1(scores)
        if achieved < args.min_macro_f1:
            print(f"FAIL: macro-F1 {achieved:.3f} < {args.min_macro_f1:.3f}")
            return 1
        print(f"PASS: macro-F1 {achieved:.3f} >= {args.min_macro_f1:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
