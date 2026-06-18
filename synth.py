"""Synthetic oncology note generator — gold labels from schema sampling + LLM rendering."""

from __future__ import annotations

from __future__ import annotations

import argparse
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv
from langfuse.openai import OpenAI
from pydantic import BaseModel, Field

from schema import (
    Biomarker,
    BiomarkerStatus,
    CancerStage,
    EcogPerformanceStatus,
    OncologyExtract,
)

load_dotenv()

DEFAULT_OUT_DIR = Path("data/synthetic")
CHAT_MODEL = "gpt-4o-mini"
MAX_WORKERS = 6

# Site-specific profiles for correlated, realistic sampling.
SITE_PROFILES: dict[str, dict] = {
    "lung": {
        "weight": 0.22,
        "histologies": ["adenocarcinoma", "squamous cell carcinoma", "small cell carcinoma"],
        "biomarkers": ["EGFR", "ALK", "KRAS", "PD-L1", "ROS1", "BRAF"],
        "regimens": [
            ["pembrolizumab", "carboplatin", "pemetrexed"],
            ["osimertinib"],
            ["durvalumab", "etoposide", "carboplatin"],
            ["crizotinib"],
        ],
    },
    "breast": {
        "weight": 0.20,
        "histologies": ["invasive ductal carcinoma", "invasive lobular carcinoma", "HER2-positive carcinoma"],
        "biomarkers": ["ER", "PR", "HER2", "Ki-67"],
        "regimens": [
            ["trastuzumab", "pertuzumab", "docetaxel"],
            ["palbociclib", "letrozole"],
            ["capecitabine"],
            ["sacituzumab govitecan"],
        ],
    },
    "colorectal": {
        "weight": 0.14,
        "histologies": ["adenocarcinoma", "mucinous adenocarcinoma"],
        "biomarkers": ["MSI", "KRAS", "NRAS", "BRAF", "HER2"],
        "regimens": [
            ["FOLFOX"],
            ["FOLFIRI", "cetuximab"],
            ["bevacizumab", "capecitabine"],
            ["regorafenib"],
        ],
    },
    "prostate": {
        "weight": 0.10,
        "histologies": ["adenocarcinoma", "neuroendocrine carcinoma"],
        "biomarkers": ["PSA", "AR-V7"],
        "regimens": [
            ["enzalutamide"],
            ["abiraterone", "prednisone"],
            ["docetaxel"],
            ["olaparib"],
        ],
    },
    "melanoma": {
        "weight": 0.08,
        "histologies": ["cutaneous melanoma", "acral lentiginous melanoma"],
        "biomarkers": ["BRAF", "NRAS", "PD-L1"],
        "regimens": [
            ["nivolumab", "ipilimumab"],
            ["dabrafenib", "trametinib"],
            ["pembrolizumab"],
        ],
    },
    "pancreas": {
        "weight": 0.08,
        "histologies": ["pancreatic ductal adenocarcinoma"],
        "biomarkers": ["CA 19-9", "BRCA1", "BRCA2"],
        "regimens": [
            ["FOLFIRINOX"],
            ["gemcitabine", "nab-paclitaxel"],
            ["olaparib"],
        ],
    },
    "ovary": {
        "weight": 0.07,
        "histologies": ["high-grade serous carcinoma", "endometrioid carcinoma"],
        "biomarkers": ["BRCA1", "BRCA2", "HRD"],
        "regimens": [
            ["carboplatin", "paclitaxel"],
            ["bevacizumab", "olaparib"],
            ["mirvetuximab soravtansine"],
        ],
    },
    "kidney": {
        "weight": 0.06,
        "histologies": ["clear cell renal cell carcinoma", "papillary renal cell carcinoma"],
        "biomarkers": ["PD-L1", "VHL"],
        "regimens": [
            ["pembrolizumab", "axitinib"],
            ["nivolumab", "cabozantinib"],
            ["sunitinib"],
        ],
    },
    "head and neck": {
        "weight": 0.05,
        "histologies": ["squamous cell carcinoma"],
        "biomarkers": ["PD-L1", "HPV", "EBV"],
        "regimens": [
            ["pembrolizumab"],
            ["cetuximab", "carboplatin", "paclitaxel"],
            ["nivolumab"],
        ],
    },
}

STAGES_EARLY = [
    CancerStage.I,
    CancerStage.IA,
    CancerStage.IB,
    CancerStage.II,
    CancerStage.IIA,
    CancerStage.IIB,
    CancerStage.III,
    CancerStage.IIIA,
]
STAGES_LATE = [
    CancerStage.IIIB,
    CancerStage.IIIC,
    CancerStage.IV,
    CancerStage.IVA,
    CancerStage.IVB,
]

ECOG_WEIGHTS = [
    (EcogPerformanceStatus.FULLY_ACTIVE, 0.25),
    (EcogPerformanceStatus.RESTRICTED_STRENUOUS, 0.35),
    (EcogPerformanceStatus.AMBULATORY, 0.25),
    (EcogPerformanceStatus.LIMITED_SELF_CARE, 0.12),
    (EcogPerformanceStatus.COMPLETELY_DISABLED, 0.03),
]

LINE_WEIGHTS = [(1, 0.45), (2, 0.30), (3, 0.18), (4, 0.07)]

BIOMARKER_STATUS_WEIGHTS = [
    (BiomarkerStatus.POSITIVE, 0.42),
    (BiomarkerStatus.NEGATIVE, 0.38),
    (BiomarkerStatus.EQUIVOCAL, 0.12),
    (BiomarkerStatus.UNKNOWN, 0.08),
]

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(timeout=90.0, max_retries=3)
    return _client


def _weighted_choice(options: list[tuple], rng: random.Random) -> object:
    values, weights = zip(*options)
    return rng.choices(values, weights=weights, k=1)[0]


def _sample_stage(line_of_therapy: int, rng: random.Random) -> CancerStage:
    if line_of_therapy <= 1:
        pool = STAGES_EARLY * 3 + STAGES_LATE
    elif line_of_therapy == 2:
        pool = STAGES_EARLY + STAGES_LATE * 2
    else:
        pool = STAGES_EARLY + STAGES_LATE * 4
    return rng.choice(pool)


def _sample_diagnosis_date(rng: random.Random) -> date:
    days_ago = rng.randint(30, 8 * 365)
    return date.today() - timedelta(days=days_ago)


def _sample_biomarkers(site: str, rng: random.Random) -> list[Biomarker]:
    if rng.random() < 0.08:
        return []
    candidates = SITE_PROFILES[site]["biomarkers"]
    n = rng.randint(1, min(3, len(candidates)))
    names = rng.sample(candidates, k=n)
    return [
        Biomarker(name=name, status=_weighted_choice(BIOMARKER_STATUS_WEIGHTS, rng))
        for name in names
    ]


def sample_record(rng: random.Random | None = None) -> OncologyExtract:
    """Sample a valid OncologyExtract with site-correlated, realistic value distributions."""
    if rng is None:
        rng = random.Random()

    sites = list(SITE_PROFILES.keys())
    weights = [SITE_PROFILES[s]["weight"] for s in sites]
    site = rng.choices(sites, weights=weights, k=1)[0]
    profile = SITE_PROFILES[site]

    line = _weighted_choice(LINE_WEIGHTS, rng)
    stage = _sample_stage(line, rng)
    histology = rng.choice(profile["histologies"])
    regimen = list(rng.choice(profile["regimens"]))

    biomarkers = _sample_biomarkers(site, rng)
    ecog = _weighted_choice(ECOG_WEIGHTS, rng) if rng.random() > 0.06 else None
    dx_date = _sample_diagnosis_date(rng) if rng.random() > 0.05 else None

    return OncologyExtract(
        primary_site=site,
        histology=histology,
        stage=stage,
        biomarkers=biomarkers,
        ecog_performance_status=ecog,
        line_of_therapy=line,
        date_of_diagnosis=dx_date,
        treatment_regimen=regimen,
    )


class RenderedNote(BaseModel):
    note: str = Field(description="realistic oncology clinic or pathology note prose")


def render_note(record: OncologyExtract) -> str:
    """Call the LLM to write a note that embeds record values with noise and distractors."""
    gold_json = json.dumps(record.model_dump(mode="json"), indent=2)
    response = _get_client().chat.completions.parse(
        model=CHAT_MODEL,
        response_format=RenderedNote,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write de-identified synthetic oncology clinical and pathology notes for "
                    "NLP evaluation. Embed every value from the structured record naturally using "
                    "clinical phrasing, abbreviations, and synonyms (e.g. 'stage IIIA' vs 'T2N2M0', "
                    "'ECOG 1' vs 'restricted in strenuous activity'). "
                    "Add realistic distractors: unrelated comorbidities, benign labs, prior "
                    "surgeries, family history, or formatting quirks. "
                    "Do NOT list fields as key-value pairs. Do NOT mention this is synthetic. "
                    "Length: 180–420 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Write a note that truthfully reflects this gold record:\n\n"
                    f"{gold_json}"
                ),
            },
        ],
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise RuntimeError("LLM returned no parsed note")
    return parsed.note.strip()


def save_pair(out_dir: Path, index: int, note: str, gold: OncologyExtract) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{index:04d}.json"
    payload = {"note": note, "gold": gold.model_dump(mode="json")}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def load_pair(path: Path) -> dict:
    return json.loads(path.read_text())


def _generate_one(index: int, out_dir: Path, seed: int) -> Path:
    rng = random.Random(seed + index)
    record = sample_record(rng)
    note = render_note(record)
    return save_pair(out_dir, index, note, record)


def generate_dataset(
    count: int = 200,
    out_dir: Path = DEFAULT_OUT_DIR,
    seed: int = 42,
    workers: int = MAX_WORKERS,
    resume: bool = True,
) -> list[Path]:
    """Generate count (note, gold) pairs; skips existing files when resume=True."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        i
        for i in range(count)
        if not (resume and (out_dir / f"{i:04d}.json").exists())
    ]
    if not pending:
        return sorted(out_dir.glob("*.json"))

    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_generate_one, i, out_dir, seed): i for i in pending
        }
        for n, future in enumerate(as_completed(futures), start=1):
            idx = futures[future]
            path = future.result()
            paths.append(path)
            print(f"[{n}/{len(pending)}] wrote {path.name} (index {idx})")

    manifest = {
        "count": count,
        "seed": seed,
        "model": CHAT_MODEL,
        "files": [f"{i:04d}.json" for i in range(count)],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return sorted(out_dir.glob("[0-9]*.json"))


def _note_mentions(note: str, term: str) -> bool:
    return term.lower() in note.lower()


def _stage_patterns(stage: CancerStage) -> list[str]:
    roman = stage.value
    patterns = [roman, f"stage {roman}", f"Stage {roman}"]
    if roman.startswith("I") and not roman.startswith("II"):
        patterns.append(f"stage {roman.lower()}")
    return patterns


def spot_check_pair(pair: dict) -> list[str]:
    """Return list of gold fields that appear missing from the note (heuristic)."""
    note = pair["note"]
    gold = OncologyExtract.model_validate(pair["gold"])
    misses: list[str] = []

    if gold.primary_site and not _note_mentions(note, gold.primary_site):
        misses.append(f"primary_site={gold.primary_site}")

    if gold.histology:
        hist_tokens = [t for t in re.split(r"[\s\-]+", gold.histology.lower()) if len(t) > 4]
        if hist_tokens and not any(t in note.lower() for t in hist_tokens[:2]):
            misses.append(f"histology={gold.histology}")

    if gold.stage and not any(p.lower() in note.lower() for p in _stage_patterns(gold.stage)):
        misses.append(f"stage={gold.stage.value}")

    if gold.ecog_performance_status is not None:
        ecog = gold.ecog_performance_status.value
        ecog_hits = [
            f"ecog {ecog}",
            f"ecog-{ecog}",
            f"ecog of {ecog}",
            f"performance status {ecog}",
            f"karnofsky",
        ]
        if not any(h in note.lower() for h in ecog_hits):
            misses.append(f"ecog={ecog}")

    if gold.line_of_therapy is not None:
        line = gold.line_of_therapy
        line_hits = [
            f"{line}-line",
            f"{line} line",
            f"line {line}",
            "first-line" if line == 1 else None,
            "second-line" if line == 2 else None,
            "third-line" if line == 3 else None,
            "fourth-line" if line == 4 else None,
        ]
        if not any(h and h in note.lower() for h in line_hits):
            misses.append(f"line_of_therapy={line}")

    if gold.date_of_diagnosis:
        d = gold.date_of_diagnosis
        date_hits = [
            d.isoformat(),
            d.strftime("%m/%d/%Y"),
            d.strftime("%B %d, %Y"),
            d.strftime("%b %d, %Y"),
            str(d.year),
        ]
        if not any(h.lower() in note.lower() for h in date_hits):
            misses.append(f"date_of_diagnosis={d.isoformat()}")

    for drug in gold.treatment_regimen:
        if not _note_mentions(note, drug.split()[0]):
            misses.append(f"drug={drug}")

    for bm in gold.biomarkers:
        if not _note_mentions(note, bm.name):
            misses.append(f"biomarker={bm.name}")

    return misses


def spot_check_samples(out_dir: Path, n: int = 3, seed: int = 0) -> None:
    files = sorted(out_dir.glob("[0-9]*.json"))
    if not files:
        raise FileNotFoundError(f"No pairs in {out_dir}")
    rng = random.Random(seed)
    picks = rng.sample(files, k=min(n, len(files)))
    print(f"\n--- Spot-checking {len(picks)} pairs ---")
    for path in picks:
        pair = load_pair(path)
        misses = spot_check_pair(pair)
        gold = pair["gold"]
        print(f"\n{path.name}")
        print(f"  gold: site={gold.get('primary_site')} stage={gold.get('stage')} "
              f"line={gold.get('line_of_therapy')} ecog={gold.get('ecog_performance_status')}")
        print(f"  note excerpt: {pair['note'][:220].replace(chr(10), ' ')}…")
        if misses:
            print(f"  heuristic misses: {misses}")
        else:
            print("  heuristic: all key gold values appear present")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic oncology note/gold pairs.")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--spot-check", type=int, default=3)
    args = parser.parse_args()

    paths = generate_dataset(
        count=args.count,
        out_dir=args.out_dir,
        seed=args.seed,
        workers=args.workers,
        resume=not args.no_resume,
    )
    print(f"\nDone: {len(paths)} files in {args.out_dir}")
    spot_check_samples(args.out_dir, n=args.spot_check, seed=args.seed)


if __name__ == "__main__":
    main()
