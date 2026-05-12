#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible release metadata manifest.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root directory.")
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=None,
        help="Optional manifests directory for hash/source-of-truth.",
    )
    parser.add_argument("--model-version", type=str, required=True)
    parser.add_argument("--dataset-version", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-json", type=Path, default=None, help="Optional JSON merged into output.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dir(root: Path) -> str:
    h = hashlib.sha256()
    for file in sorted([p for p in root.rglob("*") if p.is_file()]):
        rel = file.relative_to(root).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(sha256_file(file).encode("utf-8"))
    return h.hexdigest()


def count_split_classes(dataset_root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        split_dir = dataset_root / split / "images"
        counts: dict[str, int] = {}
        if split_dir.exists():
            for cls_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
                counts[cls_dir.name] = len([f for f in cls_dir.rglob("*") if f.is_file()])
        out[split] = counts
    return out


def main() -> None:
    args = parse_args()
    dataset_hash = hash_dir(args.dataset_root)
    manifests_hash = hash_dir(args.manifests_root) if args.manifests_root and args.manifests_root.exists() else None

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_version": args.model_version,
        "dataset_version": args.dataset_version,
        "dataset_root": args.dataset_root.as_posix(),
        "dataset_sha256": dataset_hash,
        "manifests_root": args.manifests_root.as_posix() if args.manifests_root else None,
        "manifests_sha256": manifests_hash,
        "split_class_counts": count_split_classes(args.dataset_root),
    }

    inventory_summary_path = None
    if args.manifests_root:
        inventory_summary_path = args.manifests_root / "inventory_summary.json"
    if inventory_summary_path and inventory_summary_path.exists():
        try:
            inventory_summary = json.loads(inventory_summary_path.read_text(encoding="utf-8"))
            cfg = inventory_summary.get("config", {})
            if isinstance(cfg, dict):
                payload["dataset_policy"] = {
                    "dataset_policy_version": cfg.get("dataset_policy_version", "v1"),
                    "annotation_policy": cfg.get("annotation_policy", {}),
                    "qc": cfg.get("qc", {}),
                }
        except Exception:
            pass

    if args.extra_json and args.extra_json.exists():
        extra = json.loads(args.extra_json.read_text(encoding="utf-8"))
        payload["extra"] = extra

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output.as_posix()}")


if __name__ == "__main__":
    main()
