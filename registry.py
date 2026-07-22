"""Model/prompt registry: versioned eval metrics, dataset hashes, and promotion.

A deliberately small, file-backed registry (JSON, matching the repo's json-artifact
style) that records each model/prompt configuration version with the eval metrics it
scored, the hash of the gold dataset it was scored on, and a timestamp. It supports
register / list / compare / promote plus append-only decision records (used by the
retrain trigger). No database and no network: the registry is a single JSON file so it
diffs cleanly in git and is trivial to test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

DEFAULT_REGISTRY = Path("data/mlops/registry.json")
SCHEMA_VERSION = 1

STATUS_REGISTERED = "registered"
STATUS_PRODUCTION = "production"
STATUS_ARCHIVED = "archived"


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


@dataclass
class ModelConfig:
    """The knobs that define one extraction configuration version."""

    model: str
    mode: str = "pipeline"
    use_verifier: bool = True
    prompt_config: str = "default"


@dataclass
class DatasetInfo:
    dir: str
    hash: str
    n_examples: int


@dataclass
class VersionRecord:
    version: str
    label: str
    created_at: str
    model_config: dict
    dataset: dict
    metrics: dict
    status: str = STATUS_REGISTERED
    decisions: list = field(default_factory=list)


def dataset_hash(data_dir: Path) -> str:
    """Deterministic sha256 over the gold pairs in a directory.

    Hashes the sorted `[0-9]*.json` gold files by name + bytes so the same dataset
    always yields the same digest, and any label change is detectable across versions.
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("[0-9]*.json"))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def metrics_from_rows(rows: list[dict]) -> dict:
    """Reduce an eval `metrics_table` to the fields the registry stores."""
    per_field = {
        row["field"]: row["f1"]
        for row in rows
        if row["field"] not in ("macro_avg", "micro_avg")
    }
    macro = next((r["f1"] for r in rows if r["field"] == "macro_avg"), None)
    micro = next((r["f1"] for r in rows if r["field"] == "micro_avg"), None)
    return {"macro_f1": macro, "micro_f1": micro, "per_field": per_field}


def metrics_from_latest(path: Path) -> dict:
    """Load registry-shaped metrics from an eval `latest_metrics.json` artifact."""
    payload = json.loads(Path(path).read_text())
    if "rows" in payload:
        return metrics_from_rows(payload["rows"])
    # Already registry-shaped.
    return {
        "macro_f1": payload.get("macro_f1"),
        "micro_f1": payload.get("micro_f1"),
        "per_field": payload.get("per_field", {}),
    }


def load_registry(path: Path = DEFAULT_REGISTRY) -> dict:
    path = Path(path)
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "versions": [], "production": None}
    return json.loads(path.read_text())


def save_registry(reg: dict, path: Path = DEFAULT_REGISTRY) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reg, indent=2) + "\n")


def next_version_id(reg: dict) -> str:
    n = len(reg.get("versions", []))
    return f"v{n + 1:04d}"


def get_version(reg: dict, version: str) -> Optional[dict]:
    return next((v for v in reg.get("versions", []) if v["version"] == version), None)


def list_versions(reg: dict) -> list[dict]:
    return list(reg.get("versions", []))


def register_version(
    label: str,
    model_config: ModelConfig,
    metrics: dict,
    *,
    dataset_dir: Optional[Path] = None,
    dataset: Optional[DatasetInfo] = None,
    path: Path = DEFAULT_REGISTRY,
) -> dict:
    """Append a new immutable version record and return it."""
    reg = load_registry(path)

    if dataset is None:
        if dataset_dir is None:
            raise ValueError("register_version needs dataset or dataset_dir")
        data_dir = Path(dataset_dir)
        n = len(sorted(data_dir.glob("[0-9]*.json")))
        dataset = DatasetInfo(dir=str(data_dir), hash=dataset_hash(data_dir), n_examples=n)

    record = VersionRecord(
        version=next_version_id(reg),
        label=label,
        created_at=_utc_now(),
        model_config=asdict(model_config),
        dataset=asdict(dataset),
        metrics=metrics,
    )
    reg["versions"].append(asdict(record))
    save_registry(reg, path)
    return asdict(record)


def compare_versions(reg: dict, version_a: str, version_b: str) -> dict:
    """Metric deltas (b minus a), per field plus macro/micro. Positive favors b."""
    a = get_version(reg, version_a)
    b = get_version(reg, version_b)
    if a is None or b is None:
        raise KeyError(f"unknown version(s): {version_a}, {version_b}")

    a_fields = a["metrics"].get("per_field", {})
    b_fields = b["metrics"].get("per_field", {})
    field_deltas = {
        f: round((b_fields.get(f) or 0.0) - (a_fields.get(f) or 0.0), 6)
        for f in sorted(set(a_fields) | set(b_fields))
    }
    macro_delta = round(
        (b["metrics"].get("macro_f1") or 0.0) - (a["metrics"].get("macro_f1") or 0.0), 6
    )
    micro_delta = round(
        (b["metrics"].get("micro_f1") or 0.0) - (a["metrics"].get("micro_f1") or 0.0), 6
    )
    return {
        "version_a": version_a,
        "version_b": version_b,
        "macro_f1_delta": macro_delta,
        "micro_f1_delta": micro_delta,
        "per_field_delta": field_deltas,
        "same_dataset": a["dataset"]["hash"] == b["dataset"]["hash"],
    }


def promote_version(version: str, path: Path = DEFAULT_REGISTRY) -> dict:
    """Mark one version as production and demote the prior production to archived."""
    reg = load_registry(path)
    target = get_version(reg, version)
    if target is None:
        raise KeyError(f"unknown version: {version}")
    for v in reg["versions"]:
        if v["status"] == STATUS_PRODUCTION and v["version"] != version:
            v["status"] = STATUS_ARCHIVED
    target["status"] = STATUS_PRODUCTION
    reg["production"] = version
    save_registry(reg, path)
    return target


def production_version(reg: dict) -> Optional[dict]:
    prod = reg.get("production")
    return get_version(reg, prod) if prod else None


def add_decision(version: str, decision: dict, path: Path = DEFAULT_REGISTRY) -> dict:
    """Append an append-only decision record (e.g. retrain recommendation) to a version."""
    reg = load_registry(path)
    target = get_version(reg, version)
    if target is None:
        raise KeyError(f"unknown version: {version}")
    entry = dict(decision)
    entry.setdefault("created_at", _utc_now())
    target.setdefault("decisions", []).append(entry)
    save_registry(reg, path)
    return entry


def _fmt_metric(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def _cmd_register(args: argparse.Namespace) -> None:
    metrics = metrics_from_latest(args.metrics)
    config = ModelConfig(
        model=args.model,
        mode=args.mode,
        use_verifier=not args.no_verifier,
        prompt_config=args.prompt_config,
    )
    record = register_version(
        args.label, config, metrics, dataset_dir=args.dataset, path=args.registry
    )
    print(f"Registered {record['version']} ({record['label']})")
    print(f"  macro_f1={_fmt_metric(record['metrics'].get('macro_f1'))}"
          f"  dataset_hash={record['dataset']['hash'][:12]}"
          f"  n={record['dataset']['n_examples']}")


def _cmd_list(args: argparse.Namespace) -> None:
    reg = load_registry(args.registry)
    versions = list_versions(reg)
    if not versions:
        print("Registry is empty.")
        return
    print(f"{'version':<8} {'status':<12} {'macro_f1':>9}  {'label':<26} dataset")
    print("-" * 78)
    for v in versions:
        print(
            f"{v['version']:<8} {v['status']:<12} "
            f"{_fmt_metric(v['metrics'].get('macro_f1')):>9}  "
            f"{v['label'][:26]:<26} {v['dataset']['hash'][:12]}"
        )


def _cmd_compare(args: argparse.Namespace) -> None:
    reg = load_registry(args.registry)
    result = compare_versions(reg, args.version_a, args.version_b)
    print(f"Compare {args.version_a} -> {args.version_b} "
          f"(same dataset: {result['same_dataset']})")
    print(f"  macro_f1 delta: {result['macro_f1_delta']:+.4f}")
    print(f"  micro_f1 delta: {result['micro_f1_delta']:+.4f}")
    print("  per-field delta:")
    for f, d in result["per_field_delta"].items():
        print(f"    {f:<24} {d:+.4f}")


def _cmd_promote(args: argparse.Namespace) -> None:
    record = promote_version(args.version, path=args.registry)
    print(f"Promoted {record['version']} ({record['label']}) to production.")


def _cmd_show(args: argparse.Namespace) -> None:
    reg = load_registry(args.registry)
    record = get_version(reg, args.version)
    if record is None:
        raise SystemExit(f"unknown version: {args.version}")
    print(json.dumps(record, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model/prompt version registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="register a new version from eval metrics")
    p_reg.add_argument("--label", required=True)
    p_reg.add_argument("--model", required=True)
    p_reg.add_argument("--mode", default="pipeline")
    p_reg.add_argument("--no-verifier", action="store_true")
    p_reg.add_argument("--prompt-config", default="default")
    p_reg.add_argument("--metrics", type=Path, required=True,
                       help="path to an eval latest_metrics.json")
    p_reg.add_argument("--dataset", type=Path, required=True,
                       help="gold dataset dir the metrics were scored on")
    p_reg.set_defaults(func=_cmd_register)

    p_list = sub.add_parser("list", help="list registered versions")
    p_list.set_defaults(func=_cmd_list)

    p_cmp = sub.add_parser("compare", help="compare two versions' metrics")
    p_cmp.add_argument("version_a")
    p_cmp.add_argument("version_b")
    p_cmp.set_defaults(func=_cmd_compare)

    p_prom = sub.add_parser("promote", help="mark a version as production")
    p_prom.add_argument("version")
    p_prom.set_defaults(func=_cmd_promote)

    p_show = sub.add_parser("show", help="print a version record as JSON")
    p_show.add_argument("version")
    p_show.set_defaults(func=_cmd_show)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
