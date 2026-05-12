#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from dataset_layout import detection_images_dir, detection_labels_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_PRIORITY = {"test": 0, "valid": 1, "train": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove exact duplicate images from a YOLO detection dataset and drop paired labels."
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to dataset root.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated splits to scan.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Output report path. Default: <dataset-root>/dedupe_report.json",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only report; do not delete files.")
    return parser.parse_args()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def list_images(images_split_dir: Path) -> list[Path]:
    if not images_split_dir.exists():
        return []
    return sorted([p for p in images_split_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def split_of(rel_path: str) -> str:
    parts = rel_path.split("/")
    if len(parts) >= 1:
        return parts[0]
    return "unknown"


def pick_keeper(rel_paths: list[str]) -> str:
    def key_fn(rel: str) -> tuple[int, str]:
        split = split_of(rel)
        return (SPLIT_PRIORITY.get(split, 99), rel)

    return sorted(rel_paths, key=key_fn)[0]


def label_path_for_image(dataset_root: Path, rel_image: str) -> Path:
    parts = rel_image.split("/")
    # <split>/images/<file>
    split = parts[0]
    stem = Path(parts[-1]).stem
    return detection_labels_dir(dataset_root, split) / f"{stem}.txt"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    report_json = args.report_json or (dataset_root / "dedupe_report.json")

    hash_to_rels: dict[str, list[str]] = defaultdict(list)
    scanned = 0
    for split in splits:
        split_dir = detection_images_dir(dataset_root, split)
        for img in list_images(split_dir):
            rel = img.relative_to(dataset_root).as_posix()
            hash_to_rels[sha1_file(img)].append(rel)
            scanned += 1

    duplicate_groups: list[dict[str, object]] = []
    remove_images: list[str] = []

    for h, rels in sorted(hash_to_rels.items()):
        if len(rels) <= 1:
            continue
        keeper = pick_keeper(rels)
        to_remove = sorted([r for r in rels if r != keeper])
        remove_images.extend(to_remove)
        duplicate_groups.append(
            {
                "sha1": h,
                "count": len(rels),
                "keeper": keeper,
                "removed": to_remove,
            }
        )

    removed_labels: list[str] = []
    missing_labels_for_removed: list[str] = []
    removed_by_split: dict[str, int] = defaultdict(int)

    for rel in remove_images:
        split = split_of(rel)
        removed_by_split[split] += 1

        img_path = dataset_root / rel
        lbl_path = label_path_for_image(dataset_root, rel)

        if not args.dry_run and img_path.exists():
            img_path.unlink()
        if lbl_path.exists():
            if not args.dry_run:
                lbl_path.unlink()
            removed_labels.append(lbl_path.relative_to(dataset_root).as_posix())
        else:
            missing_labels_for_removed.append(lbl_path.relative_to(dataset_root).as_posix())

    remaining_counts: dict[str, int] = {}
    for split in splits:
        remaining_counts[split] = len(list_images(detection_images_dir(dataset_root, split)))

    report = {
        "dataset_root": dataset_root.as_posix(),
        "dry_run": args.dry_run,
        "files_scanned": scanned,
        "duplicate_groups": len(duplicate_groups),
        "duplicates_removed": len(remove_images),
        "removed_by_split": dict(sorted(removed_by_split.items())),
        "remaining_image_counts": remaining_counts,
        "missing_labels_for_removed": missing_labels_for_removed,
        "groups": duplicate_groups,
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {report_json.as_posix()}")
    print(f"Files scanned: {scanned}")
    print(f"Duplicate groups: {len(duplicate_groups)}")
    print(f"Duplicates removed: {len(remove_images)}")


if __name__ == "__main__":
    main()
