"""Download MTSamples and select ~50 oncology transcriptions (public, no PHI)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

MTSAMPLES_URL = (
    "https://huggingface.co/datasets/harishnair04/mtsamples/resolve/main/mtsamples.csv"
)
DEFAULT_OUT = Path("data/real")


def fetch_csv(dest: Path) -> Path:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MTSAMPLES_URL, dest)
    return dest


def select_oncology_notes(csv_path: Path, count: int = 50) -> list[dict]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            spec = (row.get("medical_specialty") or "").strip()
            if "Oncology" not in spec:
                continue
            text = (row.get("transcription") or "").strip()
            if len(text) < 200:
                continue
            rows.append(row)

    seen: set[str] = set()
    selected: list[dict] = []
    for row in rows:
        key = (row.get("description") or "")[:100]
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= count:
            break
    return selected


def save_notes(notes: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, row in enumerate(notes):
        example_id = f"{i:04d}"
        payload = {
            "note": row["transcription"].strip(),
            "gold": None,
            "meta": {
                "source": "mtsamples",
                "specialty": (row.get("medical_specialty") or "").strip(),
                "description": (row.get("description") or "").strip(),
                "sample_name": (row.get("sample_name") or "").strip(),
            },
        }
        path = out_dir / f"{example_id}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        manifest.append(example_id)

    (out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "source": "MTSamples (Hematology-Oncology)",
                "license": "CC0 Public Domain",
                "count": len(manifest),
                "files": [f"{e}.json" for e in manifest],
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--csv", type=Path, default=Path("data/real/mtsamples.csv"))
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Downloading MTSamples → {args.csv}")
        fetch_csv(args.csv)

    notes = select_oncology_notes(args.csv, count=args.count)
    save_notes(notes, args.out_dir)
    print(f"Saved {len(notes)} notes to {args.out_dir}")


if __name__ == "__main__":
    main()
