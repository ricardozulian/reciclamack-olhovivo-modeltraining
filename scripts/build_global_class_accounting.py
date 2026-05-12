#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from plan_scraped_staging import IMAGE_EXTS, ACTIVE, is_ignored_folder, md5_file, parse_sources, read_csv, write_csv


CANONICAL_SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global per-class image accounting reports.")
    parser.add_argument(
        "--sources-json",
        type=Path,
        default=Path("model_pipeline/config/dataset_v2_sources.json"),
    )
    parser.add_argument(
        "--class-map-csv",
        type=Path,
        default=Path("model_pipeline/config/dataset_v2_class_map.csv"),
    )
    parser.add_argument(
        "--scraped-inventory-csv",
        type=Path,
        default=Path("dataset_staging/manifests/scraped_inventory.csv"),
    )
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path("dataset_staging/staging_targets.example.csv"),
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("dataset_staging"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(workspace_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def iter_images(root: Path) -> list[Path]:
    images: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_image(path):
            continue
        if any(is_ignored_folder(part) for part in path.parts):
            continue
        images.append(path)
    return sorted(images)


def split_from_path(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "val" in lowered:
        return "valid"
    if "test" in lowered:
        return "test"
    return "train"


def yolo_image_to_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def active_class_map(rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        source = (row.get("source_folder") or "").strip()
        canonical = (row.get("canonical_class") or "").strip()
        status = (row.get("status") or "").strip()
        if source and canonical and status == ACTIVE:
            out[source] = canonical
    return out


def target_rows_by_class(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        cls = (row.get("canonical_class") or "").strip()
        if cls:
            out[cls] = row
    return out


def count_scraped_inventory(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scraped_raw_count": 0,
            "scraped_unique_available": 0,
            "scraped_duplicate_count": 0,
            "scraped_unused_available": 0,
            "scraped_planned_count": 0,
            "scraped_md5s": set(),
        }
    )
    for row in rows:
        cls = (row.get("canonical_class") or "").strip()
        if not cls:
            continue
        bucket = out[cls]
        bucket["scraped_raw_count"] += 1
        status = (row.get("status") or "").strip()
        duplicate_role = (row.get("duplicate_role") or "").strip()
        md5 = (row.get("md5") or "").strip()
        if duplicate_role == "canonical":
            bucket["scraped_unique_available"] += 1
            bucket["scraped_md5s"].add(md5)
        else:
            bucket["scraped_duplicate_count"] += 1
        if status == "scraped_unused":
            bucket["scraped_unused_available"] += 1
        if "planned" in status or "staged" in status:
            bucket["scraped_planned_count"] += 1
    return out


def count_yolo_sources(
    workspace_root: Path,
    sources_json: dict[str, Any],
    source_class_map: dict[str, str],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    sources = parse_sources(sources_json)
    counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "open_images_train_count": 0,
            "open_images_valid_count": 0,
            "open_images_test_count": 0,
            "open_images_labeled_count": 0,
            "open_images_missing_label_count": 0,
            "open_images_md5s": set(),
        }
    )
    source_rows: list[dict[str, Any]] = []
    for source in sources:
        if source.type != "direct_yolo" or source.action != ACTIVE:
            continue
        canonical = (source_class_map.get(source.id) or source.canonical_class or "").strip()
        root = workspace_root / source.root
        if not canonical:
            source_rows.append(
                {
                    "source_id": source.id,
                    "source_root": source.root,
                    "canonical_class": "",
                    "status": "unmapped_class",
                    "image_count": 0,
                    "labeled_count": 0,
                    "missing_label_count": 0,
                }
            )
            continue
        if not root.exists():
            source_rows.append(
                {
                    "source_id": source.id,
                    "source_root": source.root,
                    "canonical_class": canonical,
                    "status": "missing",
                    "image_count": 0,
                    "labeled_count": 0,
                    "missing_label_count": 0,
                }
            )
            continue
        images = iter_images(root)
        labeled_count = 0
        missing_label_count = 0
        for image in images:
            label = yolo_image_to_label(image)
            if not label.exists() or not label.read_text(encoding="utf-8").strip():
                missing_label_count += 1
                continue
            split = split_from_path(image)
            if split not in CANONICAL_SPLITS:
                split = "train"
            digest = md5_file(image)
            bucket = counts[canonical]
            bucket[f"open_images_{split}_count"] += 1
            bucket["open_images_labeled_count"] += 1
            bucket["open_images_md5s"].add(digest)
            labeled_count += 1
        source_rows.append(
            {
                "source_id": source.id,
                "source_root": source.root,
                "canonical_class": canonical,
                "status": ACTIVE,
                "image_count": len(images),
                "labeled_count": labeled_count,
                "missing_label_count": missing_label_count,
            }
        )
        counts[canonical]["open_images_missing_label_count"] += missing_label_count
    return counts, source_rows


def build_accounting_rows(
    taxonomy: list[str],
    scraped_counts: dict[str, dict[str, Any]],
    yolo_counts: dict[str, dict[str, Any]],
    targets: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    classes = sorted(set(taxonomy) | set(scraped_counts) | set(yolo_counts) | set(targets))
    rows: list[dict[str, Any]] = []
    for cls in classes:
        scraped = scraped_counts.get(cls, {})
        yolo = yolo_counts.get(cls, {})
        target = targets.get(cls, {})
        target_v2 = (target.get("target_v2") or target.get("target_total") or "").strip()
        batch_size = (target.get("batch_size") or "").strip()
        priority = (target.get("priority") or "").strip()
        notes = (target.get("notes") or "").strip()
        scraped_md5s = set(scraped.get("scraped_md5s", set()))
        yolo_md5s = set(yolo.get("open_images_md5s", set()))
        total_unique_available = len(scraped_md5s | yolo_md5s)
        current_labeled_yolo_count = int(yolo.get("open_images_labeled_count", 0))
        scraped_unused = int(scraped.get("scraped_unused_available", 0))
        suggested_target = current_labeled_yolo_count + scraped_unused
        gap_to_target = ""
        if target_v2:
            try:
                gap_to_target = max(0, int(target_v2) - current_labeled_yolo_count)
            except ValueError:
                gap_to_target = ""
        rows.append(
            {
                "canonical_class": cls,
                "target_v2": target_v2,
                "batch_size": batch_size,
                "priority": priority,
                "current_labeled_yolo_count": current_labeled_yolo_count,
                "open_images_train_count": int(yolo.get("open_images_train_count", 0)),
                "open_images_valid_count": int(yolo.get("open_images_valid_count", 0)),
                "open_images_test_count": int(yolo.get("open_images_test_count", 0)),
                "open_images_missing_label_count": int(yolo.get("open_images_missing_label_count", 0)),
                "scraped_raw_count": int(scraped.get("scraped_raw_count", 0)),
                "scraped_unique_available": int(scraped.get("scraped_unique_available", 0)),
                "scraped_duplicate_count": int(scraped.get("scraped_duplicate_count", 0)),
                "scraped_unused_available": scraped_unused,
                "scraped_planned_count": int(scraped.get("scraped_planned_count", 0)),
                "total_available_unique": total_unique_available,
                "suggested_target_v2": suggested_target,
                "gap_to_target": gap_to_target,
                "notes": notes,
            }
        )
    return rows


def write_target_planning(path: Path, accounting_rows: list[dict[str, Any]]) -> None:
    if path.exists():
        return
    rows = [
        {
            "canonical_class": row["canonical_class"],
            "current_labeled_yolo_count": row["current_labeled_yolo_count"],
            "scraped_unique_available": row["scraped_unique_available"],
            "scraped_unused_available": row["scraped_unused_available"],
            "total_available_unique": row["total_available_unique"],
            "suggested_target_v2": row["suggested_target_v2"],
            "target_v2": "",
            "batch_size": "",
            "priority": "",
            "notes": "Fill target_v2 and batch_size after reviewing global_class_accounting.csv",
        }
        for row in accounting_rows
    ]
    write_csv(
        path,
        rows,
        [
            "canonical_class",
            "current_labeled_yolo_count",
            "scraped_unique_available",
            "scraped_unused_available",
            "total_available_unique",
            "suggested_target_v2",
            "target_v2",
            "batch_size",
            "priority",
            "notes",
        ],
    )


def run(
    sources_json: Path,
    class_map_csv: Path,
    scraped_inventory_csv: Path,
    targets_csv: Path,
    workspace_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    sources_config = load_json(resolve_path(workspace_root, sources_json))
    class_rows = read_csv(resolve_path(workspace_root, class_map_csv))
    scraped_rows = read_csv(resolve_path(workspace_root, scraped_inventory_csv))
    target_rows = read_csv(resolve_path(workspace_root, targets_csv))
    source_class_map = active_class_map(class_rows)
    targets = target_rows_by_class(target_rows)
    taxonomy = [str(item) for item in sources_config.get("taxonomy", [])]

    scraped_counts = count_scraped_inventory(scraped_rows)
    yolo_counts, yolo_source_rows = count_yolo_sources(workspace_root, sources_config, source_class_map)
    accounting_rows = build_accounting_rows(taxonomy, scraped_counts, yolo_counts, targets)

    output_abs = resolve_path(workspace_root, output_root)
    manifests = output_abs / "manifests"
    write_csv(
        manifests / "global_class_accounting.csv",
        accounting_rows,
        [
            "canonical_class",
            "target_v2",
            "batch_size",
            "priority",
            "current_labeled_yolo_count",
            "open_images_train_count",
            "open_images_valid_count",
            "open_images_test_count",
            "open_images_missing_label_count",
            "scraped_raw_count",
            "scraped_unique_available",
            "scraped_duplicate_count",
            "scraped_unused_available",
            "scraped_planned_count",
            "total_available_unique",
            "suggested_target_v2",
            "gap_to_target",
            "notes",
        ],
    )
    write_csv(
        manifests / "open_images_source_accounting.csv",
        yolo_source_rows,
        [
            "source_id",
            "source_root",
            "canonical_class",
            "status",
            "image_count",
            "labeled_count",
            "missing_label_count",
        ],
    )
    write_target_planning(output_abs / "v2_target_planning.csv", accounting_rows)

    summary = {
        "class_count": len(accounting_rows),
        "scraped_unique_available": sum(int(row["scraped_unique_available"]) for row in accounting_rows),
        "open_images_labeled_count": sum(int(row["current_labeled_yolo_count"]) for row in accounting_rows),
        "total_available_unique": sum(int(row["total_available_unique"]) for row in accounting_rows),
        "output_root": output_root.as_posix(),
    }
    (manifests / "global_class_accounting_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        sources_json=args.sources_json,
        class_map_csv=args.class_map_csv,
        scraped_inventory_csv=args.scraped_inventory_csv,
        targets_csv=args.targets_csv,
        workspace_root=args.workspace_root,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
