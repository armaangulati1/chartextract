import json
import sys
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from langfuse import get_client, observe

from schema import OncologyExtract
from pipeline import run_pipeline

OPENFDA_URL = "https://api.fda.gov/drug/label.json"
# SPL label sections to pull as free-text (public FDA data, no PHI).
LABEL_SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "adverse_reactions",
    "drug_interactions",
    "warnings",
    "contraindications",
]

DEFAULT_SAMPLE_NOTE = Path("data/synthetic/0000.json")

load_dotenv()


@observe()
def extract(text: str) -> OncologyExtract:
    """Public extraction entrypoint (delegates to agentic pipeline)."""
    return run_pipeline(text)


def fetch_label_text(drug_name: str, max_chars: int = 12_000) -> tuple[str, dict]:
    """Pull free-text sections from openFDA drug labels (no API key, no PHI)."""
    resp = requests.get(
        OPENFDA_URL,
        params={
            "search": (
                f'openfda.generic_name:"{drug_name}" OR '
                f'openfda.brand_name:"{drug_name}"'
            ),
            "limit": 1,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if not data.get("results"):
        raise ValueError(f"No openFDA labels found for '{drug_name}'")

    label = data["results"][0]
    openfda = label.get("openfda", {})
    brand = (openfda.get("brand_name") or [drug_name])[0]
    generic = (openfda.get("generic_name") or ["unknown"])[0]

    parts = [f"Drug label: {brand} ({generic})"]
    for section in LABEL_SECTIONS:
        if section not in label:
            continue
        text = label[section]
        if isinstance(text, list):
            text = " ".join(text)
        parts.append(f"\n[{section}]\n{text.strip()}")

    full_text = "\n".join(parts)
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars] + "\n...[truncated]"

    meta = {
        "brand_name": brand,
        "generic_name": generic,
        "label_id": label.get("id"),
        "effective_time": label.get("effective_time"),
        "source": OPENFDA_URL,
    }
    return full_text, meta


def _default_sample_note() -> str:
    if DEFAULT_SAMPLE_NOTE.exists():
        return json.loads(DEFAULT_SAMPLE_NOTE.read_text())["note"]
    return (
        "Oncology follow-up: 67-year-old male with lung adenocarcinoma, stage IIIA, "
        "diagnosed 2023-10-09. ECOG 1. First-line pembrolizumab, carboplatin, pemetrexed. "
        "EGFR negative, PD-L1 positive."
    )


def run_file(in_path: str, out_path: str):
    text = open(in_path).read()
    result = extract(text)
    get_client().flush()
    with open(out_path, "w") as f:
        json.dump(result.model_dump(mode="json"), f, indent=2)
    print(f"Extracted oncology record → {out_path}")


def run_fda(drug_name: str, out_path: Optional[str] = None):
    print(f"Fetching openFDA label for '{drug_name}'...")
    text, meta = fetch_label_text(drug_name)
    print(f"Label: {meta['brand_name']} ({meta['generic_name']})")
    print(f"Text length: {len(text)} chars\n")

    result = extract(text)
    get_client().flush()
    payload = {"source": meta, "extract": result.model_dump(mode="json")}

    if out_path:
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Extracted oncology record → {out_path}")
    else:
        print(json.dumps(payload, indent=2))


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--fda":
        drug = sys.argv[2]
        out = sys.argv[3] if len(sys.argv) > 3 else None
        run_fda(drug, out)
        return

    if len(sys.argv) == 3:
        run_file(sys.argv[1], sys.argv[2])
        return

    result = extract(_default_sample_note())
    get_client().flush()
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
