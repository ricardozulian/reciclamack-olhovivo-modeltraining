#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MD5_PREFIX = re.compile(r"^([0-9a-f]{32})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a reviewed Roboflow single-class roundtrip batch.")
    parser.add_argument("--reviewed-root", type=Path, required=True)
    parser.add_argument("--original-manifest", type=Path, required=True)
    parser.add_argument("--canonical-class", required=True)
    parser.add_argument("--fix-train-only-yaml", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def read_original_md5s(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["md5"] for row in csv.DictReader(fh)}


def md5_from_name(path: Path) -> str:
    match = MD5_PREFIX.match(path.name)
    return match.group(1) if match else ""


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def write_train_only_yaml(reviewed_root: Path, canonical_class: str) -> None:
    text = "\n".join(
        [
            f"path: {reviewed_root.resolve().as_posix()}",
            "train: train/images",
            "val: train/images",
            "",
            "nc: 1",
            f"names: ['{canonical_class}']",
            "",
        ]
    )
    (reviewed_root / "data.yaml").write_text(text, encoding="utf-8")


def audit(reviewed_root: Path, original_md5s: set[str], canonical_class: str) -> dict[str, Any]:
    images_dir = reviewed_root / "train" / "images"
    labels_dir = reviewed_root / "train" / "labels"
    images = sorted(path for path in images_dir.rglob("*") if path.is_file() and is_image(path)) if images_dir.exists() else []
    labels = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    image_md5s = {md5_from_name(path) for path in images if md5_from_name(path)}
    label_md5s = {md5_from_name(path) for path in labels if md5_from_name(path)}

    box_dist: Counter[int] = Counter()
    class_ids: Counter[int] = Counter()
    nulled_label_files: list[str] = []
    full_seed_files: list[str] = []
    malformed_rows: list[dict[str, str]] = []
    out_of_range_rows: list[dict[str, str]] = []
    nonzero_class_rows: list[dict[str, str]] = []
    overlap_warnings: list[dict[str, Any]] = []

    full_seed_variants = {
        "0 0.5 0.5 1 1",
        "0 0.5 0.5 1.0 1.0",
        "0 0.5 0.5 0.95 0.95",
    }
    for label in labels:
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        boxes: list[dict[str, Any]] = []
        box_dist[len(lines)] += 1
        if not lines:
            nulled_label_files.append(label.name)
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                malformed_rows.append({"label": label.name, "line": line})
                continue
            try:
                cid = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                malformed_rows.append({"label": label.name, "line": line})
                continue
            class_ids[cid] += 1
            if cid != 0:
                nonzero_class_rows.append({"label": label.name, "line": line})
            if any(value < 0 or value > 1 for value in coords):
                out_of_range_rows.append({"label": label.name, "line": line})
            if line in full_seed_variants:
                full_seed_files.append(label.name)
            x_center, y_center, width, height = coords
            boxes.append(
                {
                    "line": line,
                    "xmin": x_center - width / 2,
                    "ymin": y_center - height / 2,
                    "xmax": x_center + width / 2,
                    "ymax": y_center + height / 2,
                    "area": width * height,
                }
            )
        overlap_warnings.extend(find_symmetric_overlap_warnings(label.name, boxes))

    errors: list[str] = []
    if image_stems - label_stems:
        errors.append(f"images_without_labels:{len(image_stems - label_stems)}")
    if label_stems - image_stems:
        errors.append(f"labels_without_images:{len(label_stems - image_stems)}")
    if original_md5s - image_md5s:
        errors.append(f"original_md5s_missing_from_images:{len(original_md5s - image_md5s)}")
    if original_md5s - label_md5s:
        errors.append(f"original_md5s_missing_from_labels:{len(original_md5s - label_md5s)}")
    if image_md5s - original_md5s:
        errors.append(f"unexpected_image_md5s:{len(image_md5s - original_md5s)}")
    if label_md5s - original_md5s:
        errors.append(f"unexpected_label_md5s:{len(label_md5s - original_md5s)}")
    if malformed_rows:
        errors.append(f"malformed_rows:{len(malformed_rows)}")
    if out_of_range_rows:
        errors.append(f"out_of_range_rows:{len(out_of_range_rows)}")
    if nonzero_class_rows:
        errors.append(f"nonzero_class_rows:{len(nonzero_class_rows)}")

    return {
        "canonical_class": canonical_class,
        "dataset_root": reviewed_root.as_posix(),
        "image_count": len(images),
        "label_file_count": len(labels),
        "original_manifest_count": len(original_md5s),
        "box_count_distribution": {str(k): v for k, v in sorted(box_dist.items())},
        "class_id_counts": {str(k): v for k, v in sorted(class_ids.items())},
        "nulled_label_files": nulled_label_files,
        "empty_label_files": nulled_label_files,
        "nulled_policy": "leave_behind_exclude_from_downstream_export_never_negative_in_any_workflow_may_belong_to_other_class",
        "full_seed_or_full_image_label_files": full_seed_files,
        "malformed_rows": malformed_rows[:50],
        "out_of_range_rows": out_of_range_rows[:50],
        "nonzero_class_rows": nonzero_class_rows[:50],
        "overlap_warnings": overlap_warnings[:100],
        "overlap_warning_count": len(overlap_warnings),
        "status": "valid" if not errors else "needs_attention",
        "errors": errors,
    }


def intersection_area(a: dict[str, Any], b: dict[str, Any]) -> float:
    x_overlap = max(0.0, min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]))
    y_overlap = max(0.0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))
    return x_overlap * y_overlap


def find_symmetric_overlap_warnings(label_name: str, boxes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for left_idx in range(len(boxes)):
        for right_idx in range(left_idx + 1, len(boxes)):
            left = boxes[left_idx]
            right = boxes[right_idx]
            intersection = intersection_area(left, right)
            if intersection <= 0:
                continue
            left_area = max(float(left["area"]), 1e-12)
            right_area = max(float(right["area"]), 1e-12)
            left_covered_by_right = intersection / left_area
            right_covered_by_left = intersection / right_area
            smaller_coverage = intersection / min(left_area, right_area)
            if max(left_covered_by_right, right_covered_by_left) >= 0.90 or smaller_coverage >= 0.90:
                warnings.append(
                    {
                        "label": label_name,
                        "box_a_index": left_idx,
                        "box_b_index": right_idx,
                        "a_covered_by_b": round(left_covered_by_right, 6),
                        "b_covered_by_a": round(right_covered_by_left, 6),
                        "smaller_box_coverage": round(smaller_coverage, 6),
                        "reason": "symmetric_overlap_or_containment",
                    }
                )
    return warnings


def main() -> None:
    args = parse_args()
    if args.fix_train_only_yaml:
        write_train_only_yaml(args.reviewed_root, args.canonical_class)
    report = audit(args.reviewed_root, read_original_md5s(args.original_manifest), args.canonical_class)
    output = args.output_json or (args.reviewed_root / "roundtrip_audit_report.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Status: {report['status']}")
    print(f"Images: {report['image_count']}")
    print(f"Labels: {report['label_file_count']}")
    print(f"Nulled labels: {len(report['nulled_label_files'])}")
    print(f"Full-image labels: {len(report['full_seed_or_full_image_label_files'])}")
    print(f"Wrote: {output.as_posix()}")
    if report["errors"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
