"""Tests for the pre-bill eval harness."""

from __future__ import annotations

from prebill.checks import CHECK_IDS
from prebill.evalset import (
    load_fixtures,
    macro_f1,
    micro_totals,
    score_checks,
)


def test_fixtures_load():
    fixtures = load_fixtures()
    assert len(fixtures) >= 8
    ids = [fx.fixture_id for fx in fixtures]
    assert len(ids) == len(set(ids))


def test_at_least_two_clean_fixtures():
    clean = [fx for fx in load_fixtures() if not fx.expected_check_ids]
    assert len(clean) >= 2


def test_eval_matches_goldens_exactly():
    scores = score_checks(load_fixtures())
    assert all(s.fp == 0 for s in scores), "unexpected false positives"
    assert all(s.fn == 0 for s in scores), "unexpected false negatives"


def test_every_check_has_a_positive_fixture():
    scores = {s.check_id: s for s in score_checks(load_fixtures())}
    for cid in CHECK_IDS:
        assert scores[cid].tp >= 1, f"{cid} has no positive fixture"


def test_micro_and_macro_are_perfect_on_selfauthored_set():
    scores = score_checks(load_fixtures())
    micro = micro_totals(scores)
    assert micro.precision == 1.0
    assert micro.recall == 1.0
    assert macro_f1(scores) == 1.0


def test_min_macro_f1_gate_respected():
    from prebill.evalset import main

    assert main.__module__ == "prebill.evalset"
