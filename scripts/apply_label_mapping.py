#!/usr/bin/env python3
"""
Apply ReciclaMack label normalization to a COCO annotations file.

Example:
  python model_pipeline/scripts/apply_label_mapping.py \
    --input-coco data/annotations.json \
    --mapping-json model_pipeline/roboflow_label_mapping.json \
    --output-coco data/annotations_mapped.json \
    --report-json data/mapping_report.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply label mapping to COCO annotations.")
    parser.add_argument("--input-coco", required=True, type=Path, help="Input COCO JSON path.")
    parser.add_argument("--mapping-json", required=True, type=Path, help="Mapping policy JSON path.")
    parser.add_argument("--output-coco", required=True, type=Path, help="Output COCO JSON path.")
    parser.add_argument(
        "--report-json",
        required=True,
        type=Path,
        help="Output report JSON path (stats and review labels).",
    )
    parser.add_argument(
        "--unknown-action",
        choices=["review", "ignore"],
        default=None,
        help="Override unknown label action from mapping rules.",
    )
    return parser.parse_args()


def normalize_label(label: str, rules: dict[str, Any]) -> str:
    value = label
    if rules.get("trim_before_match", True):
        value = value.strip()
    if rules.get("lowercase_before_match", True):
        value = value.lower()
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_target_categories(target_classes: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    categories: list[dict[str, Any]] = []
    name_to_id: dict[str, int] = {}
    for idx, name in enumerate(target_classes, start=1):
        categories.append({"id": idx, "name": name, "supercategory": "e_waste_gadget"})
        name_to_id[name] = idx
    return categories, name_to_id


def main() -> None:
    args = parse_args()
    coco = load_json(args.input_coco)
    mapping = load_json(args.mapping_json)

    target_classes: list[str] = mapping["target_classes"]
    map_policy: dict[str, str] = mapping.get("map", {})
    ignore_policy = set(mapping.get("ignore", []))
    review_policy = set(mapping.get("review", []))
    rules = mapping.get("rules", {})

    unknown_action = args.unknown_action or rules.get("unknown_label_action", "review")
    if unknown_action not in {"review", "ignore"}:
        raise ValueError("unknown_action must be 'review' or 'ignore'")

    input_categories = coco.get("categories", [])
    category_id_to_name: dict[int, str] = {int(cat["id"]): str(cat["name"]) for cat in input_categories}

    new_categories, target_name_to_id = build_target_categories(target_classes)

    mapped_annotations: list[dict[str, Any]] = []
    mapped_images_ids: set[int] = set()
    source_label_counts: Counter[str] = Counter()
    target_label_counts: Counter[str] = Counter()
    ignored_label_counts: Counter[str] = Counter()
    review_label_counts: Counter[str] = Counter()

    for ann in coco.get("annotations", []):
        source_name = category_id_to_name.get(int(ann["category_id"]), "")
        normalized = normalize_label(source_name, rules)
        source_label_counts[normalized] += 1

        target_name: str | None = None
        action = "map"

        if normalized in map_policy:
            target_name = map_policy[normalized]
        elif normalized in ignore_policy:
            action = "ignore"
        elif normalized in review_policy:
            action = "review"
        else:
            action = unknown_action

        if action == "map":
            if target_name not in target_name_to_id:
                review_label_counts[normalized] += 1
                continue
            new_ann = dict(ann)
            new_ann["category_id"] = target_name_to_id[target_name]
            mapped_annotations.append(new_ann)
            mapped_images_ids.add(int(ann["image_id"]))
            target_label_counts[target_name] += 1
        elif action == "ignore":
            ignored_label_counts[normalized] += 1
        else:
            review_label_counts[normalized] += 1

    mapped_images = [img for img in coco.get("images", []) if int(img["id"]) in mapped_images_ids]

    output = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": mapped_images,
        "annotations": mapped_annotations,
        "categories": new_categories,
    }

    args.output_coco.parent.mkdir(parents=True, exist_ok=True)
    args.output_coco.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "input_file": str(args.input_coco),
        "output_file": str(args.output_coco),
        "mapping_file": str(args.mapping_json),
        "unknown_action": unknown_action,
        "input_images": len(coco.get("images", [])),
        "input_annotations": len(coco.get("annotations", [])),
        "output_images": len(mapped_images),
        "output_annotations": len(mapped_annotations),
        "source_label_counts": dict(sorted(source_label_counts.items())),
        "target_label_counts": dict(sorted(target_label_counts.items())),
        "ignored_label_counts": dict(sorted(ignored_label_counts.items())),
        "review_label_counts": dict(sorted(review_label_counts.items())),
    }

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Mapping complete.")
    print(f"Input annotations: {report['input_annotations']}")
    print(f"Output annotations: {report['output_annotations']}")
    print(f"Review labels: {len(review_label_counts)}")


if __name__ == "__main__":
    main()

