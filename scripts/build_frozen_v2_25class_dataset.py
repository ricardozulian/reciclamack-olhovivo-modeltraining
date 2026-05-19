#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXCLUDED_CLASSES = {"capacitor"}
OLD_MANUAL_TO_CANONICAL = {
    "5": "flat_monitor",
    "12": "crt_monitor",
}
YOLO_SPLIT_MAP = {
    "train": "train",
    "valid": "valid",
    "val": "valid",
    "test": "test",
}
SCRAPED_SPLIT_RATIOS = (0.70, 0.20, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble frozen 25-class v2 YOLO dataset.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--class-order-csv",
        type=Path,
        default=Path("dataset_staging/v2_frozen_25_class_order.csv"),
    )
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=Path("dataset_staging/v2_first_run_class_targets.csv"),
    )
    parser.add_argument(
        "--class-map-csv",
        type=Path,
        default=Path("model_pipeline/config/dataset_v2_class_map.csv"),
    )
    parser.add_argument(
        "--manual-review-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
    )
    parser.add_argument(
        "--class-batches-root",
        type=Path,
        default=Path("dataset_staging/class_batches"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prod_datasets/detection/v2_25class"),
    )
    parser.add_argument("--clean", action="store_true", help="Remove output root before assembly.")
    return parser.parse_args()


def resolve(workspace_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def rel_to_workspace(workspace_root: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reviewed_md5_from_stem(stem: str) -> str:
    return stem.split("_", 1)[0]


def stable_split(key: str) -> str:
    value = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    train_cut = SCRAPED_SPLIT_RATIOS[0]
    valid_cut = SCRAPED_SPLIT_RATIOS[0] + SCRAPED_SPLIT_RATIOS[1]
    if value < train_cut:
        return "train"
    if value < valid_cut:
        return "valid"
    return "test"


def clean_output(workspace_root: Path, output_root: Path) -> None:
    if not output_root.exists():
        return
    resolved_output = output_root.resolve()
    resolved_workspace = workspace_root.resolve()
    expected_parent = (workspace_root / "prod_datasets" / "detection").resolve()
    if not str(resolved_output).startswith(str(expected_parent)):
        raise ValueError(f"Refusing to clean outside prod_datasets/detection: {resolved_output}")
    if not str(resolved_output).startswith(str(resolved_workspace)):
        raise ValueError(f"Refusing to clean outside workspace: {resolved_output}")
    shutil.rmtree(output_root)


def ensure_output_dirs(output_root: Path) -> None:
    for split in ("train", "valid", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)


def load_class_order(path: Path) -> tuple[dict[str, int], list[str]]:
    rows = read_csv(path)
    mapping = {row["canonical_class"]: int(row["class_id"]) for row in rows}
    names = [None] * len(mapping)
    for row in rows:
        names[int(row["class_id"])] = row["canonical_class"]
    if any(name is None for name in names):
        raise ValueError("Class IDs are not contiguous")
    return mapping, [str(name) for name in names]


def load_targets(path: Path) -> dict[str, dict[str, int]]:
    targets: dict[str, dict[str, int]] = {}
    for row in read_csv(path):
        cls = row["canonical_class"]
        targets[cls] = {
            "yolo_target": int(row.get("yolo_target") or 0),
            "manual_review_target": int(row.get("manual_review_target") or 0),
        }
    return targets


def load_yolo_source_map(path: Path, frozen_classes: set[str]) -> dict[str, str]:
    source_map: dict[str, str] = {}
    for row in read_csv(path):
        source = row["source_folder"]
        canonical = row["canonical_class"]
        status = row["status"]
        if not source.startswith("yolo_data_"):
            continue
        if status != "active":
            continue
        if canonical in frozen_classes and canonical not in EXCLUDED_CLASSES:
            source_map[source] = canonical
    return source_map


def remap_label_lines(lines: list[str], class_id: int, source_old_id: str | None = None) -> list[str]:
    out: list[str] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Malformed YOLO row: {line}")
        if source_old_id is not None and parts[0] != source_old_id:
            raise ValueError(f"Unexpected source class id {parts[0]} != {source_old_id}: {line}")
        vals = [float(x) for x in parts[1:]]
        if not all(0.0 <= x <= 1.0 for x in vals):
            raise ValueError(f"Out-of-range YOLO row: {line}")
        out.append(f"{class_id} {vals[0]:.12g} {vals[1]:.12g} {vals[2]:.12g} {vals[3]:.12g}")
    return out


def unique_output_paths(output_root: Path, split: str, md5: str, ext: str, used_stems: Counter[str]) -> tuple[Path, Path, str]:
    stem = md5
    used_stems[stem] += 1
    return (
        output_root / split / "images" / f"{stem}{ext.lower()}",
        output_root / split / "labels" / f"{stem}.txt",
        stem,
    )


def add_record(
    *,
    workspace_root: Path,
    output_root: Path,
    used_stems: Counter[str],
    source_image: Path,
    label_lines: list[str],
    canonical_class: str,
    class_id: int,
    split: str,
    source_type: str,
    source_family: str,
    source_manifest: str,
    original_md5: str | None,
    lineage_rows: list[dict[str, Any]],
    exported_md5s: set[str],
    warnings: list[str],
) -> None:
    md5 = original_md5 or md5_file(source_image)
    if md5 in exported_md5s:
        warnings.append(f"duplicate_md5_skipped:{md5}:{rel_to_workspace(workspace_root, source_image)}")
        return
    out_img, out_label, output_stem = unique_output_paths(
        output_root, split, md5, source_image.suffix, used_stems
    )
    shutil.copy2(source_image, out_img)
    out_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    exported_md5s.add(md5)
    lineage_rows.append(
        {
            "md5": md5,
            "output_stem": output_stem,
            "canonical_class": canonical_class,
            "class_id": class_id,
            "split": split,
            "source_type": source_type,
            "source_family": source_family,
            "source_path": rel_to_workspace(workspace_root, source_image),
            "source_manifest": source_manifest,
            "output_image": rel_to_workspace(workspace_root, out_img),
            "output_label": rel_to_workspace(workspace_root, out_label),
            "box_count": len(label_lines),
        }
    )


def find_image_for_label(images_dir: Path, label_stem: str) -> Path | None:
    for ext in IMAGE_EXTS:
        candidate = images_dir / f"{label_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def add_reviewed_batches(
    *,
    workspace_root: Path,
    output_root: Path,
    class_batches_root: Path,
    class_to_id: dict[str, int],
    used_stems: Counter[str],
    lineage_rows: list[dict[str, Any]],
    exported_md5s: set[str],
    warnings: list[str],
) -> None:
    for canonical_class, class_id in sorted(class_to_id.items(), key=lambda item: item[1]):
        if canonical_class in {"flat_monitor", "crt_monitor"}:
            continue
        batch_root = class_batches_root / canonical_class / "batch_001_reviewed"
        source_type = "reviewed_roundtrip"
        if not batch_root.exists():
            staged = class_batches_root / canonical_class / "batch_001"
            if staged.exists() and canonical_class == "usb_stick":
                batch_root = staged
                source_type = "staged_seed_unreviewed"
                warnings.append("usb_stick uses staged seed labels because no reviewed export was found")
            else:
                warnings.append(f"missing_reviewed_batch:{canonical_class}")
                continue
        labels_dir = batch_root / "train" / "labels"
        images_dir = batch_root / "train" / "images"
        if not labels_dir.exists() or not images_dir.exists():
            warnings.append(f"missing_train_dirs:{canonical_class}")
            continue
        for label in sorted(labels_dir.glob("*.txt")):
            raw_lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not raw_lines:
                continue
            image = find_image_for_label(images_dir, label.stem)
            if image is None:
                warnings.append(f"reviewed_label_without_image:{label.as_posix()}")
                continue
            try:
                label_lines = remap_label_lines(raw_lines, class_id, source_old_id="0")
            except ValueError as exc:
                warnings.append(f"bad_reviewed_label:{label.as_posix()}:{exc}")
                continue
            md5 = reviewed_md5_from_stem(label.stem)
            split = stable_split(f"{canonical_class}:{md5}")
            add_record(
                workspace_root=workspace_root,
                output_root=output_root,
                used_stems=used_stems,
                source_image=image,
                label_lines=label_lines,
                canonical_class=canonical_class,
                class_id=class_id,
                split=split,
                source_type=source_type,
                source_family="roboflow_reviewed_single_class",
                source_manifest=rel_to_workspace(workspace_root, batch_root / "roundtrip_audit_report.json")
                if (batch_root / "roundtrip_audit_report.json").exists()
                else rel_to_workspace(workspace_root, batch_root / "manifests" / "copy_manifest.csv"),
                original_md5=md5,
                lineage_rows=lineage_rows,
                exported_md5s=exported_md5s,
                warnings=warnings,
            )


def add_yolo_boosts(
    *,
    workspace_root: Path,
    output_root: Path,
    yolo_source_map: dict[str, str],
    targets: dict[str, dict[str, int]],
    class_to_id: dict[str, int],
    used_stems: Counter[str],
    lineage_rows: list[dict[str, Any]],
    exported_md5s: set[str],
    warnings: list[str],
) -> None:
    candidates_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_folder, canonical_class in sorted(yolo_source_map.items()):
        source_root = workspace_root / "notebooks" / source_folder
        if not source_root.exists():
            warnings.append(f"missing_yolo_source:{source_folder}")
            continue
        for source_split, output_split in YOLO_SPLIT_MAP.items():
            img_dir = source_root / "images" / source_split
            lab_dir = source_root / "labels" / source_split
            if not img_dir.exists() or not lab_dir.exists():
                continue
            for image in sorted(path for path in img_dir.iterdir() if path.is_file() and is_image(path)):
                label = lab_dir / f"{image.stem}.txt"
                if not label.exists():
                    continue
                raw_lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not raw_lines:
                    continue
                candidates_by_class[canonical_class].append(
                    {
                        "source_folder": source_folder,
                        "image": image,
                        "label": label,
                        "split": output_split,
                        "raw_lines": raw_lines,
                    }
                )

    for canonical_class, candidates in sorted(candidates_by_class.items()):
        target = targets.get(canonical_class, {}).get("yolo_target", 0)
        if target <= 0:
            continue
        selected = sorted(
            candidates,
            key=lambda item: (item["split"], item["source_folder"], item["image"].name),
        )[:target]
        if len(selected) < target:
            warnings.append(f"yolo_target_shortfall:{canonical_class}:{len(selected)}<{target}")
        class_id = class_to_id[canonical_class]
        for item in selected:
            try:
                label_lines = remap_label_lines(item["raw_lines"], class_id, source_old_id="0")
            except ValueError as exc:
                warnings.append(f"bad_yolo_label:{item['label'].as_posix()}:{exc}")
                continue
            add_record(
                workspace_root=workspace_root,
                output_root=output_root,
                used_stems=used_stems,
                source_image=item["image"],
                label_lines=label_lines,
                canonical_class=canonical_class,
                class_id=class_id,
                split=item["split"],
                source_type="yolo_boost",
                source_family=item["source_folder"],
                source_manifest=rel_to_workspace(workspace_root, item["label"]),
                original_md5=None,
                lineage_rows=lineage_rows,
                exported_md5s=exported_md5s,
                warnings=warnings,
            )


def add_manual_monitor_review(
    *,
    workspace_root: Path,
    output_root: Path,
    manual_review_csv: Path,
    targets: dict[str, dict[str, int]],
    class_to_id: dict[str, int],
    used_stems: Counter[str],
    lineage_rows: list[dict[str, Any]],
    exported_md5s: set[str],
    warnings: list[str],
) -> None:
    rows_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    row_by_md5: dict[str, dict[str, str]] = {}
    for row in read_csv(manual_review_csv):
        raw_lines = [line.strip() for line in row["preview_label_lines"].split("\\n") if line.strip()]
        if not raw_lines:
            continue
        row_by_md5[row["image_md5"]] = row
        classes: set[str] = set()
        for line in raw_lines:
            old_id = line.split()[0]
            canonical_class = OLD_MANUAL_TO_CANONICAL.get(old_id)
            if canonical_class is None:
                warnings.append(f"manual_unknown_class_id:{old_id}:{row['image_md5']}")
                continue
            classes.add(canonical_class)
        for canonical_class in classes:
            rows_by_class[canonical_class].append(row)

    selected_md5s: set[str] = set()
    for canonical_class, rows in sorted(rows_by_class.items()):
        target = targets.get(canonical_class, {}).get("manual_review_target", 0)
        if target <= 0:
            continue
        selected = sorted(rows, key=lambda row: (row["image_md5"], row["source_path"]))[:target]
        if len(selected) < target:
            warnings.append(f"manual_target_shortfall:{canonical_class}:{len(selected)}<{target}")
        selected_md5s.update(row["image_md5"] for row in selected)

    for image_md5 in sorted(selected_md5s):
        row = row_by_md5[image_md5]
        source_image = resolve(workspace_root, Path(row["source_path"]))
        if not source_image.exists():
            warnings.append(f"manual_missing_image:{row['source_path']}")
            continue
        raw_lines = [line.strip() for line in row["preview_label_lines"].split("\\n") if line.strip()]
        label_lines: list[str] = []
        line_classes: list[str] = []
        for line in raw_lines:
            old_id = line.split()[0]
            canonical_class = OLD_MANUAL_TO_CANONICAL.get(old_id)
            if canonical_class is None:
                continue
            try:
                label_lines.extend(
                    remap_label_lines([line], class_to_id[canonical_class], source_old_id=old_id)
                )
                line_classes.append(canonical_class)
            except ValueError as exc:
                warnings.append(f"bad_manual_label:{row['image_md5']}:{exc}")
        if not label_lines:
            continue
        primary_class = sorted(set(line_classes), key=lambda cls: class_to_id[cls])[0]
        split = stable_split(f"manual_monitor:{row['image_md5']}")
        add_record(
            workspace_root=workspace_root,
            output_root=output_root,
            used_stems=used_stems,
            source_image=source_image,
            label_lines=label_lines,
            canonical_class=primary_class,
            class_id=class_to_id[primary_class],
            split=split,
            source_type="manual_monitor_review",
            source_family="manual_review",
            source_manifest=rel_to_workspace(workspace_root, manual_review_csv),
            original_md5=row["image_md5"],
            lineage_rows=lineage_rows,
            exported_md5s=exported_md5s,
            warnings=warnings,
        )


def write_data_yaml(output_root: Path, class_names: list[str]) -> None:
    lines = [
        "path: .",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]
    for idx, name in enumerate(class_names):
        lines.append(f"  {idx}: {name}")
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_dataset(output_root: Path, class_names: list[str], lineage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    image_by_stem: dict[str, Path] = {}
    label_by_stem: dict[str, Path] = {}
    label_class_counts: Counter[int] = Counter()
    label_box_counts: Counter[int] = Counter()

    for split in ("train", "valid", "test"):
        img_dir = output_root / split / "images"
        lab_dir = output_root / split / "labels"
        for image in img_dir.rglob("*"):
            if image.is_file() and is_image(image):
                image_by_stem[f"{split}/{image.stem}"] = image
        for label in lab_dir.rglob("*.txt"):
            label_by_stem[f"{split}/{label.stem}"] = label
            for line in label.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"malformed_label:{label.as_posix()}:{line}")
                    continue
                try:
                    cls = int(parts[0])
                    vals = [float(x) for x in parts[1:]]
                except ValueError:
                    errors.append(f"non_numeric_label:{label.as_posix()}:{line}")
                    continue
                if cls < 0 or cls >= len(class_names):
                    errors.append(f"invalid_class_id:{label.as_posix()}:{cls}")
                if any(v < 0.0 or v > 1.0 for v in vals):
                    errors.append(f"out_of_range_label:{label.as_posix()}:{line}")
                label_class_counts[cls] += 1
                label_box_counts[cls] += 1

    missing_labels = sorted(set(image_by_stem) - set(label_by_stem))
    missing_images = sorted(set(label_by_stem) - set(image_by_stem))
    if missing_labels:
        errors.append(f"images_without_labels:{len(missing_labels)}")
    if missing_images:
        errors.append(f"labels_without_images:{len(missing_images)}")

    md5_counts = Counter(row["md5"] for row in lineage_rows)
    duplicate_md5s = sorted(md5 for md5, count in md5_counts.items() if count > 1)
    if duplicate_md5s:
        errors.append(f"duplicate_md5s:{len(duplicate_md5s)}")

    class_image_counts = Counter(row["canonical_class"] for row in lineage_rows)
    split_counts = Counter(row["split"] for row in lineage_rows)
    source_type_counts = Counter(row["source_type"] for row in lineage_rows)
    class_split_counts = Counter((row["canonical_class"], row["split"]) for row in lineage_rows)

    return {
        "status": "valid" if not errors else "invalid",
        "image_count": len(image_by_stem),
        "label_file_count": len(label_by_stem),
        "box_count": sum(label_box_counts.values()),
        "class_count": len(class_names),
        "class_image_counts": dict(sorted(class_image_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "class_split_counts": {
            f"{cls}/{split}": count for (cls, split), count in sorted(class_split_counts.items())
        },
        "label_class_counts": {
            class_names[class_id]: count for class_id, count in sorted(label_class_counts.items())
        },
        "duplicate_md5_count": len(duplicate_md5s),
        "errors": errors,
        "warnings": warnings,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    workspace_root = args.workspace_root.resolve()
    output_root = resolve(workspace_root, args.output_root)
    class_order_csv = resolve(workspace_root, args.class_order_csv)
    targets_csv = resolve(workspace_root, args.targets_csv)
    class_map_csv = resolve(workspace_root, args.class_map_csv)
    manual_review_csv = resolve(workspace_root, args.manual_review_csv)
    class_batches_root = resolve(workspace_root, args.class_batches_root)

    if args.clean:
        clean_output(workspace_root, output_root)
    ensure_output_dirs(output_root)

    class_to_id, class_names = load_class_order(class_order_csv)
    targets = load_targets(targets_csv)
    # Add collaborator targets for reporting; selection comes from reviewed batches.
    targets.setdefault("home_appliance", {"yolo_target": 0, "manual_review_target": 0})
    targets.setdefault("home_theater", {"yolo_target": 0, "manual_review_target": 0})
    yolo_source_map = load_yolo_source_map(class_map_csv, set(class_to_id))

    lineage_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    used_stems: Counter[str] = Counter()
    exported_md5s: set[str] = set()

    add_reviewed_batches(
        workspace_root=workspace_root,
        output_root=output_root,
        class_batches_root=class_batches_root,
        class_to_id=class_to_id,
        used_stems=used_stems,
        lineage_rows=lineage_rows,
        exported_md5s=exported_md5s,
        warnings=warnings,
    )
    add_yolo_boosts(
        workspace_root=workspace_root,
        output_root=output_root,
        yolo_source_map=yolo_source_map,
        targets=targets,
        class_to_id=class_to_id,
        used_stems=used_stems,
        lineage_rows=lineage_rows,
        exported_md5s=exported_md5s,
        warnings=warnings,
    )
    add_manual_monitor_review(
        workspace_root=workspace_root,
        output_root=output_root,
        manual_review_csv=manual_review_csv,
        targets=targets,
        class_to_id=class_to_id,
        used_stems=used_stems,
        lineage_rows=lineage_rows,
        exported_md5s=exported_md5s,
        warnings=warnings,
    )

    write_data_yaml(output_root, class_names)
    manifest_fields = [
        "md5",
        "output_stem",
        "canonical_class",
        "class_id",
        "split",
        "source_type",
        "source_family",
        "source_path",
        "source_manifest",
        "output_image",
        "output_label",
        "box_count",
    ]
    write_csv(output_root / "manifests" / "image_lineage_manifest.csv", lineage_rows, manifest_fields)
    write_csv(
        output_root / "manifests" / "class_id_map.csv",
        [{"class_id": idx, "canonical_class": name} for idx, name in enumerate(class_names)],
        ["class_id", "canonical_class"],
    )
    validation = validate_dataset(output_root, class_names, lineage_rows)
    validation["assembly_warnings"] = warnings
    (output_root / "manifests" / "validation_report.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    class_split_rows = [
        {"canonical_class": cls, "split": split, "image_count": count}
        for key, count in sorted(validation["class_split_counts"].items())
        for cls, split in [key.split("/", 1)]
    ]
    write_csv(
        output_root / "manifests" / "class_split_counts.csv",
        class_split_rows,
        ["canonical_class", "split", "image_count"],
    )
    source_rows = [
        {"source_type": source_type, "image_count": count}
        for source_type, count in sorted(validation["source_type_counts"].items())
    ]
    write_csv(
        output_root / "manifests" / "source_contribution_counts.csv",
        source_rows,
        ["source_type", "image_count"],
    )

    print(f"Output root: {output_root.as_posix()}")
    print(f"Status: {validation['status']}")
    print(f"Images: {validation['image_count']}")
    print(f"Labels: {validation['label_file_count']}")
    print(f"Boxes: {validation['box_count']}")
    print(f"Classes: {validation['class_count']}")
    print(f"Assembly warnings: {len(warnings)}")
    if validation["errors"]:
        raise SystemExit(2)
    return validation


def main() -> None:
    build(parse_args())


if __name__ == "__main__":
    main()
