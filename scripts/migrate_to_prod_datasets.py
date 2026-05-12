#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from dataset_layout import CANONICAL_SPLITS, detection_images_dir, detection_labels_dir, write_detection_data_yaml


SPLIT_ALIASES = {"train": "train", "val": "valid", "valid": "valid", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate local generated datasets into prod_datasets and pivot layout.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--force", action="store_true", help="Overwrite destination if it already exists.")
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def move_root(src: Path, dst: Path, force: bool) -> None:
    if not src.exists():
        return
    if dst.exists():
        if not force:
            raise SystemExit(f"Destination exists: {dst.as_posix()} (use --force)")
        shutil.rmtree(dst)
    ensure_parent(dst)
    shutil.move(src.as_posix(), dst.as_posix())


def _rewrite_seed_csv(csv_path: Path) -> None:
    if not csv_path.exists():
        return
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
        fieldnames = list(rows[0].keys()) if rows else []

    for row in rows:
        if row.get("split") == "val":
            row["split"] = "valid"
        if row.get("target_image"):
            row["target_image"] = row["target_image"].replace("images/val/", "valid/images/")
            row["target_image"] = row["target_image"].replace("images/train/", "train/images/")
            row["target_image"] = row["target_image"].replace("images/test/", "test/images/")
        if row.get("target_label"):
            row["target_label"] = row["target_label"].replace("labels/val/", "valid/labels/")
            row["target_label"] = row["target_label"].replace("labels/train/", "train/labels/")
            row["target_label"] = row["target_label"].replace("labels/test/", "test/labels/")

    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _pivot_detection(root: Path) -> None:
    if not root.exists():
        return
    legacy_images = root / "images"
    legacy_labels = root / "labels"

    for split in CANONICAL_SPLITS:
        detection_images_dir(root, split).mkdir(parents=True, exist_ok=True)
        detection_labels_dir(root, split).mkdir(parents=True, exist_ok=True)

    if legacy_images.exists():
        for split_dir in [p for p in legacy_images.iterdir() if p.is_dir()]:
            dst_split = SPLIT_ALIASES.get(split_dir.name)
            if not dst_split:
                continue
            for file in split_dir.rglob("*"):
                if not file.is_file():
                    continue
                rel = file.relative_to(split_dir)
                target = detection_images_dir(root, dst_split) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(file.as_posix(), target.as_posix())
        shutil.rmtree(legacy_images)

    if legacy_labels.exists():
        for split_dir in [p for p in legacy_labels.iterdir() if p.is_dir()]:
            dst_split = SPLIT_ALIASES.get(split_dir.name)
            if not dst_split:
                continue
            for file in split_dir.rglob("*"):
                if not file.is_file():
                    continue
                rel = file.relative_to(split_dir)
                target = detection_labels_dir(root, dst_split) / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(file.as_posix(), target.as_posix())
        shutil.rmtree(legacy_labels)

    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        names_map: dict[int, str] = {}
        in_names = False
        for line in data_yaml.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s == "names:":
                in_names = True
                continue
            if not in_names or ":" not in s:
                continue
            left, right = s.split(":", 1)
            left = left.strip()
            right = right.strip().strip("'\"")
            if left.isdigit():
                names_map[int(left)] = right
        if names_map:
            names = [names_map[i] for i in sorted(names_map)]
            write_detection_data_yaml(data_yaml, root, names)

    for seed_name in ("annotation_seed.csv", "annotation_seed_incremental.csv", "annotation_seed_full_backup.csv"):
        _rewrite_seed_csv(root / seed_name)


def _pivot_unified(root: Path) -> None:
    if not root.exists():
        return
    legacy_images = root / "images"
    if legacy_images.exists():
        for split_dir in [p for p in legacy_images.iterdir() if p.is_dir()]:
            dst_split = SPLIT_ALIASES.get(split_dir.name)
            if not dst_split:
                continue
            for class_dir in [p for p in split_dir.iterdir() if p.is_dir()]:
                target_dir = root / dst_split / "images" / class_dir.name
                target_dir.mkdir(parents=True, exist_ok=True)
                for file in class_dir.rglob("*"):
                    if not file.is_file():
                        continue
                    rel = file.relative_to(class_dir)
                    target = target_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(file.as_posix(), target.as_posix())
        shutil.rmtree(legacy_images)

    manifests = root / "manifests"
    if manifests.exists():
        val_csv = manifests / "val.csv"
        valid_csv = manifests / "valid.csv"
        if val_csv.exists() and not valid_csv.exists():
            shutil.move(val_csv.as_posix(), valid_csv.as_posix())
        if valid_csv.exists():
            with valid_csv.open("r", encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
                fieldnames = list(rows[0].keys()) if rows else []
            for row in rows:
                if row.get("split") == "val":
                    row["split"] = "valid"
            if rows:
                with valid_csv.open("w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)

        inv = manifests / "inventory_summary.json"
        if inv.exists():
            payload = json.loads(inv.read_text(encoding="utf-8"))
            split_ratio = payload.get("config", {}).get("split_ratio")
            if isinstance(split_ratio, dict) and "val" in split_ratio and "valid" not in split_ratio:
                split_ratio["valid"] = split_ratio.pop("val")
            per_split = payload.get("export_stats", {}).get("per_split_counts")
            if isinstance(per_split, dict) and "val" in per_split and "valid" not in per_split:
                per_split["valid"] = per_split.pop("val")
            inv.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.workspace_root.resolve()

    targets = {
        root / "dataset_unified_v1": root / "prod_datasets" / "unified" / "v1",
        root / "dataset_unified_v1_dryrun": root / "prod_datasets" / "unified" / "v1_dryrun",
        root / "dataset_detection_v1": root / "prod_datasets" / "detection" / "v1",
        root / "dataset_detection_v1_1": root / "prod_datasets" / "detection" / "v1_1",
        root / "dataset_detection_v1_1_merged": root / "prod_datasets" / "detection" / "v1_1_merged",
    }

    for src, dst in targets.items():
        move_root(src, dst, args.force)

    _pivot_unified(root / "prod_datasets" / "unified" / "v1")
    _pivot_unified(root / "prod_datasets" / "unified" / "v1_dryrun")
    _pivot_detection(root / "prod_datasets" / "detection" / "v1")
    _pivot_detection(root / "prod_datasets" / "detection" / "v1_1")
    _pivot_detection(root / "prod_datasets" / "detection" / "v1_1_merged")

    print("Migration completed.")


if __name__ == "__main__":
    main()
