#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRAPED_SOURCE_TYPE = "scraped_full_image_seed"
ACTIVE = "active"
UPLOAD_SPLIT = "train"
FULL_IMAGE_LABEL = "0 0.5 0.5 0.95 0.95"


@dataclass
class Source:
    id: str
    type: str
    root: str
    canonical_class: str
    action: str
    source_family: str
    source_trust: str


@dataclass
class InventoryItem:
    image_id: str
    md5: str
    source_id: str
    source_family: str
    source_trust: str
    source_root: str
    source_path: str
    original_filename: str
    extension: str
    file_size: int
    width: int | None
    height: int | None
    canonical_class: str
    duplicate_rank: int
    duplicate_role: str
    status: str
    future_staged_image: str
    future_staged_label: str
    full_image_label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan scraped-image staging batches without moving, copying, or linking images."
    )
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
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("dataset_staging"))
    parser.add_argument(
        "--targets-csv",
        type=Path,
        default=None,
        help="Optional targets CSV. Defaults to <output-root>/staging_targets.example.csv.",
    )
    parser.add_argument("--batch-id", default="batch_001")
    parser.add_argument(
        "--max-preview-per-class",
        type=int,
        default=None,
        help="Optional cap for preview rows when no per-class batch_size is configured.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_sources(config: dict[str, Any]) -> list[Source]:
    sources: list[Source] = []
    for raw in config.get("sources", []):
        sources.append(
            Source(
                id=str(raw["id"]),
                type=str(raw["type"]),
                root=str(raw["root"]),
                canonical_class=str(raw.get("canonical_class", "")),
                action=str(raw.get("action", ACTIVE)),
                source_family=str(raw.get("source_family", "")),
                source_trust=str(raw.get("source_trust", "")),
            )
        )
    return sources


def active_class_map(rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        source = (row.get("source_folder") or "").strip()
        canonical = (row.get("canonical_class") or "").strip()
        status = (row.get("status") or "").strip()
        if source and canonical and status == ACTIVE:
            out[source] = canonical
    return out


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


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


def is_ignored_folder(name: str) -> bool:
    lowered = name.lower()
    return lowered == "__pycache__" or "dismiss" in lowered or "dismis" in lowered


def rel_path(path: Path, workspace_root: Path) -> str:
    rel = path.resolve().relative_to(workspace_root.resolve())
    return rel.as_posix()


def planned_image_path(output_root: Path, canonical_class: str, batch_id: str, md5: str, ext: str) -> str:
    return (
        output_root
        / "class_batches"
        / canonical_class
        / batch_id
        / UPLOAD_SPLIT
        / "images"
        / f"{md5}{ext}"
    ).as_posix()


def planned_label_path(output_root: Path, canonical_class: str, batch_id: str, md5: str) -> str:
    return (
        output_root
        / "class_batches"
        / canonical_class
        / batch_id
        / UPLOAD_SPLIT
        / "labels"
        / f"{md5}.txt"
    ).as_posix()


def scan_scraped_sources(
    workspace_root: Path,
    output_root: Path,
    sources: list[Source],
    source_class_map: dict[str, str],
    batch_id: str,
) -> tuple[list[InventoryItem], list[dict[str, Any]]]:
    raw_items: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for source in sources:
        if source.type != SCRAPED_SOURCE_TYPE:
            continue

        mapped_class = source_class_map.get(source.id, source.canonical_class).strip()
        root = workspace_root / source.root
        source_warning = ""
        if source.action != ACTIVE:
            source_rows.append(
                {
                    "source_id": source.id,
                    "source_root": source.root,
                    "source_family": source.source_family,
                    "action": source.action,
                    "canonical_class": mapped_class,
                    "status": source.action,
                    "image_count": 0,
                    "candidate_count": 0,
                    "warning": "",
                }
            )
            continue
        if not mapped_class:
            source_rows.append(
                {
                    "source_id": source.id,
                    "source_root": source.root,
                    "source_family": source.source_family,
                    "action": source.action,
                    "canonical_class": "",
                    "status": "unmapped_class",
                    "image_count": 0,
                    "candidate_count": 0,
                    "warning": f"unmapped_source_class:{source.id}",
                }
            )
            continue
        if not root.exists():
            source_rows.append(
                {
                    "source_id": source.id,
                    "source_root": source.root,
                    "source_family": source.source_family,
                    "action": source.action,
                    "canonical_class": mapped_class,
                    "status": "missing",
                    "image_count": 0,
                    "candidate_count": 0,
                    "warning": f"missing_source:{source.id}",
                }
            )
            continue

        images = iter_images(root)
        for image_path in images:
            digest = md5_file(image_path)
            width, height = image_dimensions(image_path)
            ext = image_path.suffix.lower()
            raw_items.append(
                {
                    "image_id": digest,
                    "md5": digest,
                    "source_id": source.id,
                    "source_family": source.source_family,
                    "source_trust": source.source_trust,
                    "source_root": source.root,
                    "source_path": rel_path(image_path, workspace_root),
                    "original_filename": image_path.name,
                    "extension": ext,
                    "file_size": image_path.stat().st_size,
                    "width": width,
                    "height": height,
                    "canonical_class": mapped_class,
                    "future_staged_image": planned_image_path(output_root, mapped_class, batch_id, digest, ext),
                    "future_staged_label": planned_label_path(output_root, mapped_class, batch_id, digest),
                    "full_image_label": FULL_IMAGE_LABEL,
                }
            )

        if source.id not in source_class_map:
            source_warning = f"source_not_in_class_map:{source.id}"
        source_rows.append(
            {
                "source_id": source.id,
                "source_root": source.root,
                "source_family": source.source_family,
                "action": source.action,
                "canonical_class": mapped_class,
                "status": ACTIVE,
                "image_count": len(images),
                "candidate_count": len(images),
                "warning": source_warning,
            }
        )

    by_md5: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_items:
        by_md5[item["md5"]].append(item)

    inventory: list[InventoryItem] = []
    for digest, group in sorted(by_md5.items()):
        group_sorted = sorted(group, key=lambda row: (str(row["source_path"]), str(row["source_id"])))
        for idx, item in enumerate(group_sorted, start=1):
            duplicate_role = "canonical" if idx == 1 else "duplicate"
            status = "scraped_unused" if idx == 1 else "duplicate_exact_content"
            inventory.append(
                InventoryItem(
                    image_id=item["image_id"],
                    md5=item["md5"],
                    source_id=item["source_id"],
                    source_family=item["source_family"],
                    source_trust=item["source_trust"],
                    source_root=item["source_root"],
                    source_path=item["source_path"],
                    original_filename=item["original_filename"],
                    extension=item["extension"],
                    file_size=int(item["file_size"]),
                    width=item["width"],
                    height=item["height"],
                    canonical_class=item["canonical_class"],
                    duplicate_rank=idx,
                    duplicate_role=duplicate_role,
                    status=status,
                    future_staged_image=item["future_staged_image"],
                    future_staged_label=item["future_staged_label"],
                    full_image_label=item["full_image_label"],
                )
            )

    return sorted(inventory, key=lambda item: (item.canonical_class, item.md5, item.source_path)), source_rows


def inventory_to_rows(items: list[InventoryItem]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in items:
        rows.append(
            {
                "image_id": item.image_id,
                "md5": item.md5,
                "source_id": item.source_id,
                "source_family": item.source_family,
                "source_trust": item.source_trust,
                "source_root": item.source_root,
                "source_path": item.source_path,
                "original_filename": item.original_filename,
                "extension": item.extension,
                "file_size": item.file_size,
                "width": item.width or "",
                "height": item.height or "",
                "canonical_class": item.canonical_class,
                "duplicate_rank": item.duplicate_rank,
                "duplicate_role": item.duplicate_role,
                "status": item.status,
                "future_staged_image": item.future_staged_image,
                "future_staged_label": item.future_staged_label,
                "full_image_label": item.full_image_label,
            }
        )
    return rows


def duplicate_rows(items: list[InventoryItem]) -> list[dict[str, Any]]:
    grouped: dict[str, list[InventoryItem]] = defaultdict(list)
    for item in items:
        grouped[item.md5].append(item)
    rows: list[dict[str, Any]] = []
    for digest, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        canonical = sorted(group, key=lambda item: item.duplicate_rank)[0]
        for item in sorted(group, key=lambda item: item.duplicate_rank)[1:]:
            rows.append(
                {
                    "md5": digest,
                    "kept_source_path": canonical.source_path,
                    "duplicate_source_path": item.source_path,
                    "kept_source_id": canonical.source_id,
                    "duplicate_source_id": item.source_id,
                }
            )
    return rows


def class_assessment_rows(items: list[InventoryItem], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    classes = sorted({item.canonical_class for item in items} | {str(row["canonical_class"]) for row in source_rows if row["canonical_class"]})
    rows: list[dict[str, Any]] = []
    for cls in classes:
        class_items = [item for item in items if item.canonical_class == cls]
        canonical_items = [item for item in class_items if item.duplicate_role == "canonical"]
        duplicate_count = len(class_items) - len(canonical_items)
        family_counts = Counter(item.source_family for item in class_items)
        ext_counts = Counter(item.extension for item in class_items)
        warnings = [
            str(row["warning"])
            for row in source_rows
            if row.get("canonical_class") == cls and row.get("warning")
        ]
        rows.append(
            {
                "canonical_class": cls,
                "source_count": len({item.source_id for item in class_items}),
                "available_image_count": len(class_items),
                "unique_image_count": len(canonical_items),
                "duplicate_image_count": duplicate_count,
                "already_planned_count": 0,
                "unused_image_count": len(canonical_items),
                "source_family_breakdown": json.dumps(dict(sorted(family_counts.items())), sort_keys=True),
                "extension_breakdown": json.dumps(dict(sorted(ext_counts.items())), sort_keys=True),
                "warnings": "|".join(warnings),
            }
        )
    return rows


def read_targets(path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    targets: dict[str, dict[str, str]] = {}
    for row in rows:
        cls = (row.get("canonical_class") or "").strip()
        if cls:
            targets[cls] = row
    return targets


def write_targets_example(path: Path, classes: list[str]) -> None:
    if path.exists():
        return
    rows = [
        {
            "canonical_class": cls,
            "target_total": "",
            "batch_size": "",
            "priority": "",
            "notes": "Fill after reviewing class_assessment.csv",
        }
        for cls in classes
    ]
    write_csv(path, rows, ["canonical_class", "target_total", "batch_size", "priority", "notes"])


def batch_preview_rows(
    items: list[InventoryItem],
    targets: dict[str, dict[str, str]],
    max_preview_per_class: int | None,
) -> list[dict[str, Any]]:
    selectable_by_class: dict[str, list[InventoryItem]] = defaultdict(list)
    for item in items:
        if item.status == "scraped_unused" and item.duplicate_role == "canonical":
            selectable_by_class[item.canonical_class].append(item)

    rows: list[dict[str, Any]] = []
    for cls, class_items in sorted(selectable_by_class.items()):
        target = targets.get(cls, {})
        raw_batch_size = (target.get("batch_size") or "").strip()
        if raw_batch_size:
            try:
                limit = max(0, int(raw_batch_size))
            except ValueError:
                limit = 0
        elif max_preview_per_class is not None:
            limit = max(0, int(max_preview_per_class))
        else:
            limit = 0
        if limit == 0:
            continue
        for order, item in enumerate(sorted(class_items, key=lambda row: (row.md5, row.source_path))[:limit], start=1):
            rows.append(
                {
                    "batch_order": order,
                    "image_id": item.image_id,
                    "md5": item.md5,
                    "canonical_class": item.canonical_class,
                    "source_path": item.source_path,
                    "future_staged_image": item.future_staged_image,
                    "future_staged_label": item.future_staged_label,
                    "full_image_label": item.full_image_label,
                    "roboflow_local_class_id": 0,
                    "upload_split": UPLOAD_SPLIT,
                    "status": "planned_for_future_staging",
                }
            )
    return rows


def assert_relative_manifest_paths(rows: list[dict[str, Any]], fields: list[str]) -> None:
    bad: list[str] = []
    for row in rows:
        for field in fields:
            value = str(row.get(field, "")).strip()
            if value and Path(value).is_absolute():
                bad.append(f"{field}={value}")
    if bad:
        raise ValueError(f"Manifest paths must be relative: {', '.join(bad[:5])}")


def run(
    sources_json: Path,
    class_map_csv: Path,
    workspace_root: Path,
    output_root: Path,
    targets_csv: Path | None = None,
    batch_id: str = "batch_001",
    max_preview_per_class: int | None = None,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    output_root = output_root
    targets_csv = targets_csv or output_root / "staging_targets.example.csv"

    config = load_json(workspace_root / sources_json if not sources_json.is_absolute() else sources_json)
    class_rows = read_csv(workspace_root / class_map_csv if not class_map_csv.is_absolute() else class_map_csv)
    sources = parse_sources(config)
    source_class_map = active_class_map(class_rows)

    inventory, source_rows = scan_scraped_sources(
        workspace_root=workspace_root,
        output_root=output_root,
        sources=sources,
        source_class_map=source_class_map,
        batch_id=batch_id,
    )
    inventory_rows = inventory_to_rows(inventory)
    dup_rows = duplicate_rows(inventory)
    assessment_rows = class_assessment_rows(inventory, source_rows)
    classes = sorted({row["canonical_class"] for row in assessment_rows})

    output_abs = workspace_root / output_root if not output_root.is_absolute() else output_root
    manifests = output_abs / "manifests"
    write_targets_example(workspace_root / targets_csv if not targets_csv.is_absolute() else targets_csv, classes)
    targets = read_targets(workspace_root / targets_csv if not targets_csv.is_absolute() else targets_csv)
    preview_rows = batch_preview_rows(inventory, targets, max_preview_per_class)

    assert_relative_manifest_paths(
        inventory_rows,
        ["source_root", "source_path", "future_staged_image", "future_staged_label"],
    )
    assert_relative_manifest_paths(preview_rows, ["source_path", "future_staged_image", "future_staged_label"])
    assert_relative_manifest_paths(source_rows, ["source_root"])

    write_csv(
        manifests / "scraped_inventory.csv",
        inventory_rows,
        [
            "image_id",
            "md5",
            "source_id",
            "source_family",
            "source_trust",
            "source_root",
            "source_path",
            "original_filename",
            "extension",
            "file_size",
            "width",
            "height",
            "canonical_class",
            "duplicate_rank",
            "duplicate_role",
            "status",
            "future_staged_image",
            "future_staged_label",
            "full_image_label",
        ],
    )
    write_csv(
        manifests / "source_assessment.csv",
        source_rows,
        [
            "source_id",
            "source_root",
            "source_family",
            "action",
            "canonical_class",
            "status",
            "image_count",
            "candidate_count",
            "warning",
        ],
    )
    write_csv(
        manifests / "class_assessment.csv",
        assessment_rows,
        [
            "canonical_class",
            "source_count",
            "available_image_count",
            "unique_image_count",
            "duplicate_image_count",
            "already_planned_count",
            "unused_image_count",
            "source_family_breakdown",
            "extension_breakdown",
            "warnings",
        ],
    )
    write_csv(
        manifests / "duplicate_images.csv",
        dup_rows,
        ["md5", "kept_source_path", "duplicate_source_path", "kept_source_id", "duplicate_source_id"],
    )
    write_csv(
        manifests / "batch_preview.csv",
        preview_rows,
        [
            "batch_order",
            "image_id",
            "md5",
            "canonical_class",
            "source_path",
            "future_staged_image",
            "future_staged_label",
            "full_image_label",
            "roboflow_local_class_id",
            "upload_split",
            "status",
        ],
    )

    summary = {
        "scraped_inventory_rows": len(inventory_rows),
        "unique_images": sum(1 for item in inventory if item.duplicate_role == "canonical"),
        "duplicate_images": len(dup_rows),
        "class_count": len(classes),
        "batch_preview_rows": len(preview_rows),
        "output_root": output_root.as_posix(),
    }
    (manifests / "staging_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        sources_json=args.sources_json,
        class_map_csv=args.class_map_csv,
        workspace_root=args.workspace_root,
        output_root=args.output_root,
        targets_csv=args.targets_csv,
        batch_id=args.batch_id,
        max_preview_per_class=args.max_preview_per_class,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
