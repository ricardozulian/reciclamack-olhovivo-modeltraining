#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FULL_IMAGE_LABEL = "0 0.5 0.5 0.95 0.95"
DEFAULT_CLASSES = ("home_appliance", "home_theater")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build single-class YOLO staging batches from collaborator image folders."
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-root", type=Path, default=Path("collaborators"))
    parser.add_argument("--output-root", type=Path, default=Path("dataset_staging/class_batches"))
    parser.add_argument("--batch-id", default="batch_001")
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove existing per-class collaborator batch directories before copying.",
    )
    return parser.parse_args()


def resolve(workspace_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def rel_to_workspace(workspace_root: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_repo_relative(value: str) -> bool:
    return bool(value) and not Path(value).is_absolute() and "://" not in value.replace("\\", "/")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    one_name = f"{canonical_class}\n"
    (batch_root / "classes.txt").write_text(one_name, encoding="utf-8")
    (batch_root / "obj.names").write_text(one_name, encoding="utf-8")
    (batch_root / "labelmap.txt").write_text(one_name, encoding="utf-8")
    (batch_root / "labelmap.pbtxt").write_text(
        "\n".join(["item {", "  id: 0", f"  name: '{canonical_class}'", "}", ""]),
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


def source_bucket(canonical_class: str, source_path: Path) -> str:
    stem = source_path.stem
    prefix = f"{canonical_class}__"
    if stem.startswith(prefix):
        rest = stem[len(prefix) :]
        if "__" in rest:
            return rest.rsplit("__", 1)[0]
    return source_path.parent.name


def collect_images(class_root: Path) -> list[Path]:
    return sorted(
        (path for path in class_root.rglob("*") if path.is_file() and is_image(path)),
        key=lambda path: path.as_posix().lower(),
    )


def copy_or_convert_image(source_path: Path, output_image: Path) -> str:
    if source_path.suffix.lower() != ".webp":
        shutil.copy2(source_path, output_image)
        return "copy"

    with Image.open(source_path) as image:
        rgb_image = image.convert("RGB")
        rgb_image.save(output_image, format="JPEG", quality=95)
    return "webp_to_jpg"


def export_class_batch(
    workspace_root: Path,
    source_root: Path,
    output_root: Path,
    batch_id: str,
    canonical_class: str,
    clean: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    class_root = source_root / canonical_class
    if not class_root.exists():
        raise FileNotFoundError(f"Missing collaborator source folder: {class_root}")

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
    seen_md5: Counter[str] = Counter()
    for source_path in collect_images(class_root):
        md5 = md5_file(source_path)
        seen_md5[md5] += 1
        ext = ".jpg" if source_path.suffix.lower() == ".webp" else source_path.suffix.lower()
        output_image = images_dir / f"{md5}{ext}"
        output_label = labels_dir / f"{md5}.txt"
        image_export_policy = copy_or_convert_image(source_path, output_image)
        output_label.write_text(FULL_IMAGE_LABEL + "\n", encoding="utf-8")
        bucket = source_bucket(canonical_class, source_path)
        copy_rows.append(
            {
                "md5": md5,
                "canonical_class": canonical_class,
                "source_path": rel_to_workspace(workspace_root, source_path),
                "source_id": bucket,
                "source_family": "collaborator_manual",
                "source_trust": "manual_unverified",
                "original_filename": source_path.name,
                "source_manifest": rel_to_workspace(workspace_root, class_root),
                "output_image": output_image.relative_to(batch_root).as_posix(),
                "output_label": output_label.relative_to(batch_root).as_posix(),
                "label_policy": "collaborator_full_image_seed",
                "image_export_policy": image_export_policy,
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
        "image_export_policy",
        "local_class_id",
    ]
    write_csv(manifests_dir / "copy_manifest.csv", copy_rows, copy_fields)

    duplicates = sorted(md5 for md5, count in seen_md5.items() if count > 1)
    duplicate_rows = [
        {"md5": md5, "count": seen_md5[md5]}
        for md5 in duplicates
    ]
    write_csv(manifests_dir / "md5_duplicate_report.csv", duplicate_rows, ["md5", "count"])

    validation = validate_class_batch(batch_root, canonical_class, len(copy_rows), copy_rows)
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

    if len(images) != expected_count:
        errors.append(f"image_count:{len(images)} != expected:{expected_count}")
    if len(labels) != len(image_stems):
        errors.append(f"label_count:{len(labels)} != unique_image_count:{len(image_stems)}")
    missing_labels = sorted(image_stems - label_stems)
    missing_images = sorted(label_stems - image_stems)
    if missing_labels:
        errors.append(f"images_without_labels:{len(missing_labels)}")
    if missing_images:
        errors.append(f"labels_without_images:{len(missing_images)}")

    duplicate_md5s = sorted(md5 for md5, count in md5_counts.items() if count > 1)
    if duplicate_md5s:
        warnings.append(f"duplicate_source_md5s_collapsed_in_output:{len(duplicate_md5s)}")

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
        "expected_source_images": expected_count,
        "image_count": len(images),
        "label_count": len(labels),
        "manifest_rows": len(copy_rows),
        "unique_md5_count": len(md5_counts),
        "duplicate_md5_count": len(duplicate_md5s),
        "status": "valid" if not errors else "invalid",
        "errors": errors,
        "warnings": warnings,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    source_root = resolve(workspace_root, args.source_root)
    output_root = resolve(workspace_root, args.output_root)
    summary_rows: list[dict[str, Any]] = []

    for canonical_class in args.classes:
        copy_rows, validation = export_class_batch(
            workspace_root=workspace_root,
            source_root=source_root,
            output_root=output_root,
            batch_id=args.batch_id,
            canonical_class=canonical_class,
            clean=not args.no_clean,
        )
        source_id_counts = dict(sorted(Counter(row["source_id"] for row in copy_rows).items()))
        summary_rows.append(
            {
                "canonical_class": canonical_class,
                "source_image_count": len(copy_rows),
                "unique_md5_count": validation["unique_md5_count"],
                "copied_image_count": validation["image_count"],
                "source_id_counts": json.dumps(source_id_counts, sort_keys=True),
                "output_root": (output_root / canonical_class / args.batch_id).relative_to(workspace_root).as_posix(),
                "validation_status": validation["status"],
                "warnings": ";".join(validation["warnings"]),
            }
        )

    fields = [
        "canonical_class",
        "source_image_count",
        "unique_md5_count",
        "copied_image_count",
        "source_id_counts",
        "output_root",
        "validation_status",
        "warnings",
    ]
    write_csv(output_root / f"{args.batch_id}_collaborator_summary.csv", summary_rows, fields)
    summary = {
        "batch_id": args.batch_id,
        "class_count": len(summary_rows),
        "total_source_images": sum(int(row["source_image_count"]) for row in summary_rows),
        "total_unique_md5": sum(int(row["unique_md5_count"]) for row in summary_rows),
        "total_copied_images": sum(int(row["copied_image_count"]) for row in summary_rows),
        "invalid_classes": [row["canonical_class"] for row in summary_rows if row["validation_status"] != "valid"],
    }
    (output_root / f"{args.batch_id}_collaborator_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Output root: {output_root.as_posix()}")
    print(f"Classes: {summary['class_count']}")
    print(f"Source images: {summary['total_source_images']}")
    print(f"Copied images: {summary['total_copied_images']}")
    print(f"Invalid classes: {len(summary['invalid_classes'])}")
    if summary["invalid_classes"]:
        raise SystemExit(2)
    return summary


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
