#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from dataset_layout import (
    CANONICAL_SPLITS,
    detection_images_dir,
    detection_labels_dir,
    write_detection_data_yaml,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_APPEND_CLASSES = ["player", "impressora"]
DEFAULT_CAPS = {
    "train": {"player": 180, "impressora": 180},
    "valid": {"player": 40, "impressora": 40},
    "test": {"player": 24, "impressora": 24},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create append-only detection dataset v1.1 by copying base dataset and appending missing classes."
    )
    parser.add_argument("--base-root", type=Path, default=Path("prod_datasets/detection/v1"))
    parser.add_argument("--unified-images-root", type=Path, default=Path("prod_datasets/unified/v1"))
    parser.add_argument("--output-root", type=Path, default=Path("prod_datasets/detection/v1_1"))
    parser.add_argument(
        "--append-classes",
        type=str,
        default="player,impressora",
        help="Comma-separated classes to append after existing class order.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Overwrite output root if it already exists.")
    parser.add_argument(
        "--incremental-only",
        action="store_true",
        default=True,
        help="Keep only incremental rows/files in output dataset (default true).",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def parse_data_yaml_names(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_names = False
    names_map: dict[int, str] = {}
    for line in lines:
        raw = line.rstrip()
        if not raw.strip():
            continue
        if raw.strip() == "names:":
            in_names = True
            continue
        if in_names:
            if ":" not in raw:
                continue
            left, right = raw.split(":", 1)
            left = left.strip()
            right = right.strip()
            if left.isdigit():
                names_map[int(left)] = right
    if not names_map:
        raise ValueError(f"Could not parse names from {path.as_posix()}")
    return [names_map[idx] for idx in sorted(names_map)]


def write_data_yaml(path: Path, root: Path, names: list[str]) -> None:
    write_detection_data_yaml(path, root, names)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_seed_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_seed_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "target_class", "source_path", "target_image", "target_label", "annotation_status"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def count_split_files(root: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in CANONICAL_SPLITS:
        split_dir = root / split
        out[split] = len([p for p in split_dir.rglob("*") if p.is_file()]) if split_dir.exists() else 0
    return out


def next_index_for_class(existing_rows: list[dict[str, str]], split: str, cls: str) -> int:
    prefix = f"{split}__{cls}__"
    max_idx = 0
    for row in existing_rows:
        target_image = row.get("target_image", "")
        stem = Path(target_image).stem
        if stem.startswith(prefix):
            parts = stem.split("__")
            if len(parts) >= 3 and parts[2].isdigit():
                max_idx = max(max_idx, int(parts[2]))
    return max_idx + 1


def ensure_unique_target(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    base_root = args.base_root
    out_root = args.output_root
    unified_images_root = args.unified_images_root
    append_classes = [c.strip() for c in args.append_classes.split(",") if c.strip()]
    if not append_classes:
        append_classes = DEFAULT_APPEND_CLASSES

    if not base_root.exists():
        raise SystemExit(f"Base dataset does not exist: {base_root.as_posix()}")
    if not (base_root / "data.yaml").exists():
        raise SystemExit(f"Missing base data.yaml: {(base_root / 'data.yaml').as_posix()}")

    if out_root.exists():
        if not args.force:
            raise SystemExit(f"Output already exists: {out_root.as_posix()} (use --force to overwrite)")
        shutil.rmtree(out_root)

    shutil.copytree(base_root, out_root)
    base_labels = sorted((base_root).rglob("*/labels/*.txt"))
    pre_prune_label_hash_mismatches = 0
    for src in base_labels:
        rel = src.relative_to(base_root)
        dst = out_root / rel
        if not dst.exists() or sha1_file(src) != sha1_file(dst):
            pre_prune_label_hash_mismatches += 1

    base_names = parse_data_yaml_names(base_root / "data.yaml")
    final_names = list(base_names)
    appended_effective: list[str] = []
    for cls in append_classes:
        if cls not in final_names:
            final_names.append(cls)
            appended_effective.append(cls)
    write_data_yaml(out_root / "data.yaml", out_root, final_names)

    seed_rows = load_seed_rows(out_root / "annotation_seed.csv")
    write_seed_rows(out_root / "annotation_seed_full_backup.csv", seed_rows)
    existing_source_paths = {row.get("source_path", "") for row in seed_rows}

    incremental_rows: list[dict[str, str]] = []
    added_counts: dict[str, dict[str, int]] = {split: {} for split in CANONICAL_SPLITS}
    for split in CANONICAL_SPLITS:
        for cls in appended_effective:
            added_counts[split][cls] = 0
            cls_dir = unified_images_root / split / "images" / cls
            if not cls_dir.exists():
                continue
            candidates = [p for p in cls_dir.rglob("*") if is_image(p)]
            random.shuffle(candidates)
            cap = int(DEFAULT_CAPS.get(split, {}).get(cls, len(candidates)))
            take = []
            for src in candidates:
                src_rel = src.as_posix()
                if src_rel in existing_source_paths:
                    continue
                take.append(src)
                if len(take) >= cap:
                    break

            idx = next_index_for_class(seed_rows + incremental_rows, split, cls)
            for src in take:
                target_stem = f"{split}__{cls}__{idx:05d}__{src.stem}"
                target_img = detection_images_dir(out_root, split) / f"{target_stem}{src.suffix.lower()}"
                target_img = ensure_unique_target(target_img)
                target_lbl = detection_labels_dir(out_root, split) / f"{target_img.stem}.txt"
                target_lbl = ensure_unique_target(target_lbl)

                target_img.parent.mkdir(parents=True, exist_ok=True)
                target_lbl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target_img)
                target_lbl.write_text("", encoding="utf-8")

                row = {
                    "split": split,
                    "target_class": cls,
                    "source_path": src.as_posix(),
                    "target_image": target_img.relative_to(out_root).as_posix(),
                    "target_label": target_lbl.relative_to(out_root).as_posix(),
                    "annotation_status": "pending_new_class",
                }
                incremental_rows.append(row)
                added_counts[split][cls] += 1
                idx += 1

    all_rows = seed_rows + incremental_rows
    write_seed_rows(out_root / "annotation_seed_incremental.csv", incremental_rows)

    before_prune = {
        "images": {split: len([p for p in detection_images_dir(out_root, split).rglob("*") if p.is_file()]) for split in CANONICAL_SPLITS},
        "labels": {split: len([p for p in detection_labels_dir(out_root, split).rglob("*.txt") if p.is_file()]) for split in CANONICAL_SPLITS},
    }

    mode = "full_plus_incremental"
    if args.incremental_only:
        mode = "incremental_only"
        keep_images = {
            row["target_image"].replace("/", "\\").lower() for row in incremental_rows if row.get("target_image")
        }
        keep_labels = {
            row["target_label"].replace("/", "\\").lower() for row in incremental_rows if row.get("target_label")
        }

        for split in CANONICAL_SPLITS:
            img_dir = detection_images_dir(out_root, split)
            if img_dir.exists():
                for img in img_dir.rglob("*"):
                    if not img.is_file():
                        continue
                    rel = img.relative_to(out_root).as_posix().replace("/", "\\").lower()
                    if rel not in keep_images:
                        img.unlink()
            lbl_dir = detection_labels_dir(out_root, split)
            if lbl_dir.exists():
                for lbl in lbl_dir.rglob("*.txt"):
                    rel = lbl.relative_to(out_root).as_posix().replace("/", "\\").lower()
                    if rel not in keep_labels:
                        lbl.unlink()
        write_seed_rows(out_root / "annotation_seed.csv", incremental_rows)
    else:
        write_seed_rows(out_root / "annotation_seed.csv", all_rows)

    # No-rework integrity check: base labels after prune mode (informational in incremental-only mode).
    post_prune_label_hash_mismatches = 0
    for src in base_labels:
        rel = src.relative_to(base_root)
        dst = out_root / rel
        if not dst.exists() or sha1_file(src) != sha1_file(dst):
            post_prune_label_hash_mismatches += 1

    # Integrity checks in output after pruning mode selection.
    incremental_seed = load_seed_rows(out_root / "annotation_seed.csv")
    incremental_targets = {(r.get("target_image", ""), r.get("target_label", "")) for r in incremental_seed}
    orphan_images = 0
    orphan_labels = 0
    class_counter = Counter()
    for split in CANONICAL_SPLITS:
        img_dir = detection_images_dir(out_root, split)
        lbl_dir = detection_labels_dir(out_root, split)
        if img_dir.exists():
            for img in img_dir.rglob("*"):
                if not img.is_file():
                    continue
                rel_img = img.relative_to(out_root).as_posix()
                rel_lbl = (detection_labels_dir(out_root, split) / img.name).with_suffix(".txt").relative_to(out_root).as_posix()
                if (rel_img, rel_lbl) not in incremental_targets:
                    orphan_images += 1
        if lbl_dir.exists():
            for lbl in lbl_dir.rglob("*.txt"):
                rel_lbl = lbl.relative_to(out_root).as_posix()
                rel_img = (detection_images_dir(out_root, split) / lbl.name).with_suffix(".jpg").relative_to(out_root).as_posix()
                if not any(tlbl == rel_lbl for _, tlbl in incremental_targets):
                    orphan_labels += 1
    for r in incremental_seed:
        cls = r.get("target_class", "")
        split = r.get("split", "")
        class_counter[f"{split}::{cls}"] += 1

    after_prune = {
        "images": {split: len([p for p in detection_images_dir(out_root, split).rglob("*") if p.is_file()]) for split in CANONICAL_SPLITS},
        "labels": {split: len([p for p in detection_labels_dir(out_root, split).rglob("*.txt") if p.is_file()]) for split in CANONICAL_SPLITS},
    }

    report = {
        "mode": mode,
        "base_root": base_root.as_posix(),
        "output_root": out_root.as_posix(),
        "class_order_before": base_names,
        "class_order_after": final_names,
        "appended_requested": append_classes,
        "appended_effective": appended_effective,
        "added_counts": added_counts,
        "incremental_rows": len(incremental_rows),
        "base_label_files_checked": len(base_labels),
        "pre_prune_base_label_hash_mismatches": pre_prune_label_hash_mismatches,
        "post_prune_base_label_hash_mismatches": post_prune_label_hash_mismatches,
        "file_counts_before_prune": before_prune,
        "file_counts_after_prune": after_prune,
        "incremental_split_class_counts": dict(sorted(class_counter.items())),
        "orphan_images": orphan_images,
        "orphan_labels": orphan_labels,
        "required_split_presence": {
            "valid": {cls: added_counts["valid"].get(cls, 0) > 0 for cls in appended_effective},
            "test": {cls: added_counts["test"].get(cls, 0) > 0 for cls in appended_effective},
        },
    }
    (out_root / "migration_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "README.md").write_text(
        "\n".join(
            [
                "# prod_datasets/detection/v1_1 (incremental-only)",
                "",
                "This dataset contains only incremental uploads for newly appended classes.",
                "",
                "Current incremental classes:",
                "- player",
                "- impressora",
                "",
                "Workflow:",
                "1. Upload/annotate this dataset in Roboflow.",
                "2. Export YOLO labels back into this folder.",
                "3. Merge with base dataset before training:",
                "python model_pipeline/scripts/merge_detection_datasets_v1_1.py --base-root prod_datasets/detection/v1 --incremental-root prod_datasets/detection/v1_1 --output-root prod_datasets/detection/v1_1_merged --force",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_root.as_posix()}")
    print(f"Incremental rows: {len(incremental_rows)}")
    print(f"Pre-prune base label hash mismatches: {pre_prune_label_hash_mismatches}")
    print(f"Post-prune base label hash mismatches: {post_prune_label_hash_mismatches}")


if __name__ == "__main__":
    main()
