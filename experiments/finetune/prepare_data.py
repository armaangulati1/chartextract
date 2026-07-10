"""Prepare mlx-lm chat-format data for the LoRA fine-tuning experiment.

Clean split (documented in README.md):
  - TRAIN/VALID: synthetic gold notes that are NOT in the CI gold set
    (200 synthetic minus the 6 CI gold ids = 194 notes). Held out a small
    validation slice for the trainer.
  - TEST (held out, never trained on): the 50 real MTSamples notes and the
    6 CI gold notes. Those are scored separately by run_eval.py using the
    repo's own scoring path, so this file only writes a placeholder test.jsonl
    to satisfy the mlx-lm data-dir contract.

The instruction prompt is intentionally minimal and identical across every
example so the model learns the extraction task, not a prompt template.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SYNTH_DIR = REPO / "data" / "synthetic"
CI_GOLD_DIR = REPO / "data" / "eval" / "ci_gold"
OUT_DIR = Path(__file__).resolve().parent / "data"

# CI gold ids are excluded from training so the held-out CI eval is honest.
CI_GOLD_IDS = {"0000", "0006", "0028", "0063", "0114", "0150"}

INSTRUCTION = (
    "Extract structured oncology variables from the clinical note below. "
    "Return only a JSON object with these keys: primary_site, histology, "
    "stage, biomarkers (list of {name, status}), ecog_performance_status, "
    "line_of_therapy, date_of_diagnosis (YYYY-MM-DD), treatment_regimen "
    "(list of drug names). Use null for fields not stated in the note.\n\n"
    "Clinical note:\n"
)


def gold_to_json_str(gold: dict) -> str:
    """Canonical, compact-but-readable JSON target for the assistant turn."""
    return json.dumps(gold, ensure_ascii=False)


def make_example(note: str, gold: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": INSTRUCTION + note.strip()},
            {"role": "assistant", "content": gold_to_json_str(gold)},
        ]
    }


def main() -> None:
    random.seed(13)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_examples: list[dict] = []
    for path in sorted(SYNTH_DIR.glob("[0-9]*.json")):
        if path.stem in CI_GOLD_IDS:
            continue
        payload = json.loads(path.read_text())
        gold = payload.get("gold")
        if gold is None:
            continue
        train_examples.append(make_example(payload["note"], gold))

    random.shuffle(train_examples)
    n_valid = max(10, int(0.08 * len(train_examples)))
    valid = train_examples[:n_valid]
    train = train_examples[n_valid:]

    # Placeholder test set (real eval is run by run_eval.py against the repo
    # scoring path). Use a few CI gold notes so mlx-lm --test does not error.
    test = []
    for path in sorted(CI_GOLD_DIR.glob("[0-9]*.json")):
        payload = json.loads(path.read_text())
        gold = payload.get("gold")
        if gold is not None:
            test.append(make_example(payload["note"], gold))

    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        out = OUT_DIR / f"{name}.jsonl"
        with out.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {out} ({len(rows)} examples)")

    print(f"\nTRAIN excludes CI gold ids: {sorted(CI_GOLD_IDS)}")
    print(f"train={len(train)} valid={len(valid)} test={len(test)}")


if __name__ == "__main__":
    main()
