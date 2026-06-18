"""Hand-style gold annotation: strict human-equivalent labeling for real notes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import instructor
from dotenv import load_dotenv
from langfuse.openai import OpenAI

from schema import OncologyExtract

load_dotenv()

ANNOTATOR_PROMPT = (
    "You are an expert clinical annotator creating GOLD STANDARD labels for extraction evaluation. "
    "Read the oncology/hematology transcription and populate OncologyExtract. "
    "CRITICAL: only label values EXPLICITLY stated in the text. "
    "Use null for absent scalar fields and empty lists when not stated. "
    "Do not infer stage, line of therapy, or biomarkers unless clearly documented. "
    "Use schema enums for stage, ECOG (0-4), and biomarker status. "
    "For primary_site use the anatomic or disease site named (e.g. thyroid, breast, blood/bone marrow). "
    "For treatment_regimen list only active cancer-directed drugs mentioned (generic names)."
)

CHAT_MODEL = "gpt-4o-mini"


def annotate_note(note: str) -> OncologyExtract:
    client = instructor.from_openai(OpenAI(timeout=90.0, max_retries=3))
    return client.chat.completions.create(
        model=CHAT_MODEL,
        response_model=OncologyExtract,
        messages=[
            {"role": "system", "content": ANNOTATOR_PROMPT},
            {"role": "user", "content": note},
        ],
    )


def annotate_dir(data_dir: Path, limit: int | None = None) -> int:
    files = sorted(data_dir.glob("[0-9]*.json"))
    if limit:
        files = files[:limit]
    done = 0
    for path in files:
        payload = json.loads(path.read_text())
        if payload.get("gold") is not None:
            continue
        gold = annotate_note(payload["note"])
        payload["gold"] = gold.model_dump(mode="json")
        path.write_text(json.dumps(payload, indent=2) + "\n")
        done += 1
        print(f"labeled {path.name}")
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/real"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    n = annotate_dir(args.data_dir, limit=args.limit)
    print(f"Annotated {n} files in {args.data_dir}")


if __name__ == "__main__":
    main()
