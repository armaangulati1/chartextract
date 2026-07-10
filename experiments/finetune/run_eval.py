"""Evaluate an MLX model (base or LoRA-fine-tuned) on the held-out sets,
scoring with the REPO'S OWN scoring path (eval.evaluate_dataset + metrics_table).

Held-out sets (never trained on):
  - real: the 50 MTSamples notes (data/real)
  - ci_gold: the 6 CI gold notes (data/eval/ci_gold)

For each note we prompt the model with the SAME minimal instruction used in
training, parse the JSON reply into schema.OncologyExtract, and hand the
prediction to the repo scorer. Parse failures are counted honestly and scored
as an empty extract (so they hurt recall, exactly as a real pipeline failure
would).

Usage:
  run_eval.py --set real   --model <hf_or_path> [--adapter <adapter_dir>] --tag ft
  run_eval.py --set ci_gold --model <hf_or_path> [--adapter <adapter_dir>] --tag base
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eval import evaluate_dataset, load_pairs, metrics_table, format_pct  # noqa: E402
from schema import OncologyExtract  # noqa: E402

import mlx_lm  # noqa: E402

INSTRUCTION = (
    "Extract structured oncology variables from the clinical note below. "
    "Return only a JSON object with these keys: primary_site, histology, "
    "stage, biomarkers (list of {name, status}), ecog_performance_status, "
    "line_of_therapy, date_of_diagnosis (YYYY-MM-DD), treatment_regimen "
    "(list of drug names). Use null for fields not stated in the note.\n\n"
    "Clinical note:\n"
)

SET_DIRS = {
    "real": REPO / "data" / "real",
    "ci_gold": REPO / "data" / "eval" / "ci_gold",
}

# module-level counters filled during a run
_PARSE_FAILURES: list[str] = []
_TOTAL: list[str] = []


def extract_json(text: str) -> dict | None:
    """Pull the first balanced JSON object out of a model reply."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : i + 1]
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return None
    return None


def coerce_to_schema(obj: dict) -> OncologyExtract:
    """Best-effort mapping of a raw dict into OncologyExtract.

    Invalid enum/type values are dropped to null rather than raising, which is
    the honest behavior: a malformed field is a missed field, not a crash.
    """
    clean: dict = {}
    # scalars
    for k in ("primary_site", "histology"):
        v = obj.get(k)
        clean[k] = v if isinstance(v, str) and v.strip() else None
    # stage / ecog / line handled by pydantic validation below with fallback
    for k in ("stage", "ecog_performance_status", "line_of_therapy", "date_of_diagnosis"):
        clean[k] = obj.get(k)
    # lists
    bios = obj.get("biomarkers") or []
    clean_bios = []
    if isinstance(bios, list):
        for b in bios:
            if isinstance(b, dict) and b.get("name"):
                clean_bios.append({"name": str(b["name"]), "status": b.get("status", "unknown")})
    clean["biomarkers"] = clean_bios
    reg = obj.get("treatment_regimen") or []
    clean["treatment_regimen"] = [str(x) for x in reg] if isinstance(reg, list) else []

    try:
        return OncologyExtract.model_validate(clean)
    except Exception:
        # progressively null out fields pydantic rejects
        for k in ("stage", "ecog_performance_status", "line_of_therapy", "date_of_diagnosis"):
            try:
                return OncologyExtract.model_validate(clean)
            except Exception:
                clean[k] = None
        try:
            return OncologyExtract.model_validate(clean)
        except Exception:
            # last resort: biomarker status enum failures -> unknown
            for b in clean["biomarkers"]:
                b["status"] = "unknown"
            return OncologyExtract.model_validate(clean)


def build_extract_fn(model, tokenizer, max_tokens: int):
    def extract_fn(note: str) -> OncologyExtract:
        _TOTAL.append(1)
        messages = [{"role": "user", "content": INSTRUCTION + note.strip()}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        reply = mlx_lm.generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        obj = extract_json(reply)
        if obj is None:
            _PARSE_FAILURES.append(reply[:200])
            return OncologyExtract()  # empty -> counts as missed
        try:
            return coerce_to_schema(obj)
        except Exception:
            _PARSE_FAILURES.append(reply[:200])
            return OncologyExtract()

    return extract_fn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=list(SET_DIRS), required=True)
    parser.add_argument("--model", required=True, help="HF repo or local path")
    parser.add_argument("--adapter", default=None, help="LoRA adapter dir")
    parser.add_argument("--tag", default="run", help="label for output files")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    _PARSE_FAILURES.clear()
    _TOTAL.clear()

    print(f"Loading model={args.model} adapter={args.adapter} ...")
    load_kwargs = {}
    if args.adapter:
        load_kwargs["adapter_path"] = args.adapter
    model, tokenizer = mlx_lm.load(args.model, **load_kwargs)

    pairs = load_pairs(SET_DIRS[args.set], limit=args.limit)
    print(f"Evaluating {len(pairs)} notes from set={args.set}")

    extract_fn = build_extract_fn(model, tokenizer, args.max_tokens)
    t0 = time.time()
    summary = evaluate_dataset(pairs, extract_fn, use_cache=False,
                               cache_path=Path("/tmp") / f"ft_{args.tag}_{args.set}.jsonl")
    wall = time.time() - t0

    rows = metrics_table(summary)
    macro = next(r for r in rows if r["field"] == "macro_avg")
    micro = next(r for r in rows if r["field"] == "micro_avg")

    n_total = len(_TOTAL)
    n_fail = len(_PARSE_FAILURES)
    print("\n" + "=" * 60)
    print(f"SET={args.set}  TAG={args.tag}  N={summary.n_examples}")
    print(f"parse_failures = {n_fail}/{n_total} ({format_pct(n_fail / n_total if n_total else 0)})")
    print(f"macro-F1 = {format_pct(macro['f1'])}   micro-F1 = {format_pct(micro['f1'])}")
    print(f"wall = {wall:.0f}s ({wall / max(n_total,1):.1f}s/note)")
    print("=" * 60)
    print(f"\n{'field':<24}{'P':>8}{'R':>8}{'F1':>8}")
    for r in rows:
        print(f"{r['field']:<24}{format_pct(r['precision']):>8}{format_pct(r['recall']):>8}{format_pct(r['f1']):>8}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    payload = {
        "set": args.set,
        "tag": args.tag,
        "model": args.model,
        "adapter": args.adapter,
        "n_examples": summary.n_examples,
        "parse_failures": n_fail,
        "n_total": n_total,
        "macro_f1": macro["f1"],
        "micro_f1": micro["f1"],
        "wall_seconds": round(wall, 1),
        "rows": rows,
        "error_distribution": dict(summary.error_distribution),
        "parse_failure_samples": _PARSE_FAILURES[:5],
    }
    out_path = out_dir / f"{args.tag}_{args.set}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
