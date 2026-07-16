"""Schema validation for the pre-bill fixtures and their goldens."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from prebill.checks import CHECK_IDS
from prebill.claim import ClaimStub
from schema import ExtractionOutput

FIXTURE_DIR = Path(__file__).parent / "prebill" / "fixtures"
FIXTURE_FILES = sorted(FIXTURE_DIR.glob("*.json"))


def test_fixtures_exist():
    assert len(FIXTURE_FILES) >= 8


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_fixture_input_validates(path: Path):
    raw = json.loads(path.read_text())
    assert raw["fixture_id"] == path.stem
    assert raw.get("description", "").strip()
    ExtractionOutput.model_validate(raw["extraction"])
    claim = ClaimStub.model_validate(raw["claim"])
    assert claim.claim_id == raw["fixture_id"]


@pytest.mark.parametrize("path", FIXTURE_FILES, ids=lambda p: p.stem)
def test_golden_matches_and_is_valid(path: Path):
    fixture_id = path.stem
    golden_path = FIXTURE_DIR / "goldens" / f"{fixture_id}.json"
    assert golden_path.exists(), f"missing golden for {fixture_id}"
    golden = json.loads(golden_path.read_text())
    assert golden["fixture_id"] == fixture_id
    for cid in golden["expected_check_ids"]:
        assert cid in CHECK_IDS, f"{fixture_id} references unknown check {cid}"


def test_no_company_names_in_fixtures():
    banned = ("charta", "epic", "cerner", "athena", "optum", "waystar")
    for path in FIXTURE_FILES + sorted((FIXTURE_DIR / "goldens").glob("*.json")):
        text = path.read_text().lower()
        for token in banned:
            assert token not in text, f"{path.name} contains '{token}'"
