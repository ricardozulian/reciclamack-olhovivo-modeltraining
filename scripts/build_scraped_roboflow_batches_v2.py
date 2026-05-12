#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_CLASSES = {"capacitor", "home_appliance", "home_theater"}
FULL_IMAGE_LABEL = "0 0.5 0.5 0.95 0.95"
SOURCE_MANIFEST = "dataset_staging/manifests/scraped_inventory.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build scraped-only single-class YOLO batches for Roboflow box inspection."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--scraped-inventory-csv",
        type=Path,
        default=Path(SOURCE_MANIFEST),
    )
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path("dataset_staging/v2_first_run_class_targets.csv"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("dataset_staging/class_batches"))
    parser.add_argument("--batch-id", default="batch_001")
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove existing per-class batch directories before copying.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def resolve(workspace_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def rel_to_workspace(workspace_root: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_repo_relative(value: str) -> bool:
    return bool(value) and not Path(value).is_absolute() and "://" not in value.replace("\\", "/")


def load_scraped_targets(rows: list[dict[str, str]]) -> dict[str, int]:
    targets: dict[str, int] = {}
    for row in rows:
        cls = (row.get("canonical_class") or "").strip()
        if not cls or cls in EXCLUDED_CLASSES:
            continue
        scraped_target = int(row.get("scraped_target") or 0)
        if scraped_target > 0:
            targets[cls] = scraped_target
    return targets


def collect_inventory_by_class(
    workspace_root: Path,
    rows: list[dict[str, str]],
    targets: dict[str, int],
) -> dict[str, list[dict[str, str]]]:
    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        cls = (row.get("canonical_class") or "").strip()
        if cls not in targets:
            continue
        if (row.get("duplicate_role") or "").strip() != "canonical":
            continue
        source_rel = (row.get("source_path") or "").strip()
        if not is_repo_relative(source_rel):
            continue
        source_path = resolve(workspace_root, Path(source_rel))
        if not source_path.exists() or not source_path.is_file() or not is_image(source_path):
            continue
        by_class[cls].append(row)

    for cls in list(by_class):
        by_class[cls] = sorted(
            by_class[cls],
            key=lambda item: ((item.get("md5") or "").strip(), (item.get("source_path") or "").strip()),
        )
    return by_class


def write_single_class_data_yaml(path: Path, dataset_root: Path, canonical_class: str) -> None:
    lines = [
        f"path: {dataset_root.resolve().as_posix()}",
        "train: train/images",
        "val: train/images",
        "",
        "nc: 1",
        f"names: ['{canonical_class}']",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_class_labelmaps(batch_root: Path, canonical_class: str) -> None:
    # Roboflow import paths vary by format; keep the one-class label map available
    # under the common filenames while YOLO txt labels continue to use class id 0.
    one_name = f"{canonical_class}\n"
    (batch_root / "classes.txt").write_text(one_name, encoding="utf-8")
    (batch_root / "obj.names").write_text(one_name, encoding="utf-8")
    (batch_root / "labelmap.txt").write_text(one_name, encoding="utf-8")
    (batch_root / "labelmap.pbtxt").write_text(
        "\n".join(
            [
                "item {",
                "  id: 0",
                f"  name: '{canonical_class}'",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def clean_batch_dir(workspace_root: Path, batch_root: Path, output_root: Path) -> None:
    if not batch_root.exists():
        return
    resolved_batch = batch_root.resolve()
    resolved_output = output_root.resolve()
    resolved_workspace = workspace_root.resolve()
    if not str(resolved_batch).startswith(str(resolved_output)):
        raise ValueError(f"Refusing to clean outside output root: {resolved_batch}")
    if not str(resolved_batch).startswith(str(resolved_workspace)):
        raise ValueError(f"Refusing to clean outside workspace: {resolved_batch}")
    shutil.rmtree(batch_root)


def export_class_batch(
    workspace_root: Path,
    output_root: Path,
    batch_id: str,
    canonical_class: str,
    selected_rows: list[dict[str, str]],
    clean: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    batch_root = output_root / canonical_class / batch_id
    if clean:
        clean_batch_dir(workspace_root, batch_root, output_root)

    images_dir = batch_root / "train" / "images"
    labels_dir = batch_root / "train" / "labels"
    manifests_dir = batch_root / "manifests"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    copy_rows: list[dict[str, Any]] = []
    for row in selected_rows:
        source_path = resolve(workspace_root, Path((row.get("source_path") or "").strip()))
        md5 = (row.get("md5") or "").strip()
        ext = (row.get("extension") or source_path.suffix).strip().lower()
        output_image = images_dir / f"{md5}{ext}"
        output_label = labels_dir / f"{md5}.txt"
        shutil.copy2(source_path, output_image)
        output_label.write_text(FULL_IMAGE_LABEL + "\n", encoding="utf-8")
        copy_rows.append(
            {
                "md5": md5,
                "canonical_class": canonical_class,
                "source_path": rel_to_workspace(workspace_root, source_path),
                "source_id": (row.get("source_id") or "").strip(),
                "source_family": (row.get("source_family") or "").strip(),
                "source_trust": (row.get("source_trust") or "").strip(),
                "original_filename": (row.get("original_filename") or source_path.name).strip(),
                "source_manifest": SOURCE_MANIFEST,
                "output_image": output_image.relative_to(batch_root).as_posix(),
                "output_label": output_label.relative_to(batch_root).as_posix(),
                "label_policy": "scraped_full_image_seed",
                "local_class_id": 0,
            }
        )

    write_single_class_data_yaml(batch_root / "data.yaml", batch_root, canonical_class)
    write_single_class_data_yaml(batch_root / "dataset.yaml", batch_root, canonical_class)
    write_single_class_labelmaps(batch_root, canonical_class)
    copy_fields = [
        "md5",
        "canonical_class",
        "source_path",
        "source_id",
        "source_family",
        "source_trust",
        "original_filename",
        "source_manifest",
        "output_image",
        "output_label",
        "label_policy",
        "local_class_id",
    ]
    write_csv(manifests_dir / "copy_manifest.csv", copy_rows, copy_fields)

    validation = validate_class_batch(batch_root, canonical_class, len(selected_rows), copy_rows)
    (manifests_dir / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return copy_rows, validation


def validate_class_batch(
    batch_root: Path,
    canonical_class: str,
    expected_count: int,
    copy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    images_dir = batch_root / "train" / "images"
    labels_dir = batch_root / "train" / "labels"
    images = sorted(path for path in images_dir.rglob("*") if path.is_file() and is_image(path))
    labels = sorted(labels_dir.rglob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    md5_counts = Counter(row["md5"] for row in copy_rows)

    if canonical_class in EXCLUDED_CLASSES:
        errors.append(f"excluded_class:{canonical_class}")
    if len(images) != expected_count:
        errors.append(f"image_count:{len(images)} != expected:{expected_count}")
    if len(labels) != expected_count:
        errors.append(f"label_count:{len(labels)} != expected:{expected_count}")
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels:
        errors.append(f"images_without_labels:{len(missing_labels)}")
    if missing_images:
        errors.append(f"labels_without_images:{len(missing_images)}")
    duplicate_md5s = sorted(md5 for md5, count in md5_counts.items() if count > 1)
    if duplicate_md5s:
        errors.append(f"duplicate_md5s:{len(duplicate_md5s)}")

    for label in labels:
        lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines != [FULL_IMAGE_LABEL]:
            errors.append(f"unexpected_label:{label.relative_to(batch_root).as_posix()}")
    for row in copy_rows:
        for field in ("source_path", "source_manifest", "output_image", "output_label"):
            if not is_repo_relative(str(row.get(field, ""))):
                errors.append(f"non_relative_manifest_path:{field}:{row.get(field)}")

    return {
        "canonical_class": canonical_class,
        "dataset_root": batch_root.as_posix(),
        "expected_count": expected_count,
        "image_count": len(images),
        "label_count": len(labels),
        "manifest_rows": len(copy_rows),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    output_root = resolve(workspace_root, args.output_root)
    targets = load_scraped_targets(read_csv(resolve(workspace_root, args.targets_csv)))
    inventory = collect_inventory_by_class(
        workspace_root,
        read_csv(resolve(workspace_root, args.scraped_inventory_csv)),
        targets,
    )

    summary_rows: list[dict[str, Any]] = []
    all_errors: list[str] = []
    for canonical_class, target_count in sorted(targets.items()):
        available = inventory.get(canonical_class, [])
        if len(available) < target_count:
            all_errors.append(f"{canonical_class}:available:{len(available)} < target:{target_count}")
            selected = available
        else:
            selected = available[:target_count]
        copy_rows, validation = export_class_batch(
            workspace_root=workspace_root,
            output_root=output_root,
            batch_id=args.batch_id,
            canonical_class=canonical_class,
            selected_rows=selected,
            clean=not args.no_clean,
        )
        source_family_counts = dict(sorted(Counter(row["source_family"] for row in copy_rows).items()))
        summary_rows.append(
            {
                "canonical_class": canonical_class,
                "scraped_target": target_count,
                "copied_count": len(copy_rows),
                "source_family_counts": json.dumps(source_family_counts, sort_keys=True),
                "output_root": (output_root / canonical_class / args.batch_id).relative_to(workspace_root).as_posix(),
                "validation_status": validation["status"],
            }
        )
        if validation["errors"]:
            all_errors.extend(f"{canonical_class}:{error}" for error in validation["errors"])

    write_csv(
        output_root / f"{args.batch_id}_summary.csv",
        summary_rows,
        [
            "canonical_class",
            "scraped_target",
            "copied_count",
            "source_family_counts",
            "output_root",
            "validation_status",
        ],
    )
    summary = {
        "batch_id": args.batch_id,
        "class_count": len(summary_rows),
        "total_copied": sum(int(row["copied_count"]) for row in summary_rows),
        "invalid_classes": [row["canonical_class"] for row in summary_rows if row["validation_status"] != "valid"],
        "errors": all_errors,
    }
    (output_root / f"{args.batch_id}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Output root: {output_root.as_posix()}")
    print(f"Classes: {summary['class_count']}")
    print(f"Copied images: {summary['total_copied']}")
    print(f"Invalid classes: {len(summary['invalid_classes'])}")
    if all_errors:
        raise SystemExit(2)
    return summary


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
