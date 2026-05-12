#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from dataset_layout import CANONICAL_SPLITS, detection_images_dir, detection_labels_dir, write_detection_data_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge base + incremental detection datasets into train-ready dataset.")
    parser.add_argument("--base-root", type=Path, default=Path("prod_datasets/detection/v1"))
    parser.add_argument("--incremental-root", type=Path, default=Path("prod_datasets/detection/v1_1"))
    parser.add_argument("--output-root", type=Path, default=Path("prod_datasets/detection/v1_1_merged"))
    parser.add_argument("--force", action="store_true", help="Overwrite output root if it exists.")
    return parser.parse_args()


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_data_yaml_names(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names: dict[int, str] = {}
    in_names = False
    for line in lines:
        raw = line.strip()
        if not raw:
            continue
        if raw == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if ":" not in raw:
            continue
        left, right = raw.split(":", 1)
        left = left.strip()
        right = right.strip()
        if left.isdigit():
            names[int(left)] = right
    if not names:
        raise ValueError(f"Could not parse names from {path.as_posix()}")
    return [names[i] for i in sorted(names)]


def write_data_yaml(path: Path, root: Path, names: list[str]) -> None:
    write_detection_data_yaml(path, root, names)


def load_seed(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_seed(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["split", "target_class", "source_path", "target_image", "target_label", "annotation_status"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ensure_unique(path: Path) -> Path:
    if not path.exists():
        return path
    i = 2
    while True:
        candidate = path.with_name(f"{path.stem}__{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def main() -> None:
    args = parse_args()
    base_root = args.base_root
    inc_root = args.incremental_root
    out_root = args.output_root

    if not base_root.exists():
        raise SystemExit(f"Missing base dataset: {base_root.as_posix()}")
    if not inc_root.exists():
        raise SystemExit(f"Missing incremental dataset: {inc_root.as_posix()}")

    if out_root.exists():
        if not args.force:
            raise SystemExit(f"Output exists: {out_root.as_posix()} (use --force)")
        shutil.rmtree(out_root)

    shutil.copytree(base_root, out_root)

    base_names = parse_data_yaml_names(base_root / "data.yaml")
    inc_names = parse_data_yaml_names(inc_root / "data.yaml")
    if inc_names[: len(base_names)] != base_names:
        raise SystemExit("Incremental class order does not preserve base prefix; refusing merge.")
    final_names = inc_names
    write_data_yaml(out_root / "data.yaml", out_root, final_names)

    # Build hash index from current output images to avoid exact duplicates.
    existing_hash_to_rel: dict[str, str] = {}
    for split in CANONICAL_SPLITS:
        for img in list_images(detection_images_dir(out_root, split)):
            existing_hash_to_rel[sha1_file(img)] = img.relative_to(out_root).as_posix()

    added = 0
    skipped_exact_duplicates = 0
    rename_collisions = 0
    for split in CANONICAL_SPLITS:
        inc_img_dir = detection_images_dir(inc_root, split)
        if not inc_img_dir.exists():
            continue
        for img in list_images(inc_img_dir):
            img_hash = sha1_file(img)
            if img_hash in existing_hash_to_rel:
                skipped_exact_duplicates += 1
                continue

            rel_img = img.relative_to(detection_images_dir(inc_root, split))
            dst_img = detection_images_dir(out_root, split) / rel_img.name
            if dst_img.exists():
                rename_collisions += 1
                dst_img = ensure_unique(dst_img)
            dst_lbl = detection_labels_dir(out_root, split) / f"{dst_img.stem}.txt"

            src_lbl = detection_labels_dir(inc_root, split) / f"{img.stem}.txt"
            if not src_lbl.exists():
                raise SystemExit(f"Missing incremental label for {img.as_posix()}")

            dst_img.parent.mkdir(parents=True, exist_ok=True)
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst_img)
            shutil.copy2(src_lbl, dst_lbl)
            existing_hash_to_rel[img_hash] = dst_img.relative_to(out_root).as_posix()
            added += 1

    # Merge seeds for traceability.
    base_seed = load_seed(base_root / "annotation_seed.csv")
    inc_seed = load_seed(inc_root / "annotation_seed.csv")
    merged_seed = base_seed + inc_seed
    write_seed(out_root / "annotation_seed.csv", merged_seed)

    # No-rework audit on base labels.
    base_label_hash_mismatches = 0
    base_labels_checked = 0
    for src in base_root.rglob("*/labels/*.txt"):
        rel = src.relative_to(base_root)
        dst = out_root / rel
        base_labels_checked += 1
        if not dst.exists() or sha1_file(src) != sha1_file(dst):
            base_label_hash_mismatches += 1

    # Basic merged counts.
    split_counts: dict[str, int] = {}
    for split in CANONICAL_SPLITS:
        split_counts[split] = len(list_images(detection_images_dir(out_root, split)))

    report = {
        "base_root": base_root.as_posix(),
        "incremental_root": inc_root.as_posix(),
        "output_root": out_root.as_posix(),
        "class_order_base": base_names,
        "class_order_incremental": inc_names,
        "class_order_merged": final_names,
        "added_images_from_incremental": added,
        "skipped_exact_duplicates": skipped_exact_duplicates,
        "rename_collisions": rename_collisions,
        "merged_image_counts": split_counts,
        "base_labels_checked": base_labels_checked,
        "base_label_hash_mismatches": base_label_hash_mismatches,
    }
    (out_root / "merge_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {out_root.as_posix()}")
    print(f"Added incremental images: {added}")
    print(f"Skipped exact duplicates: {skipped_exact_duplicates}")


if __name__ == "__main__":
    main()
