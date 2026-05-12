#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dataset_layout import CANONICAL_SPLITS, write_detection_data_yaml

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SCRAPED_SEED_LABEL = "seed_full_image_needs_refinement"
YOLO_REVIEW_LABEL = "imported_yolo_label"
MANUAL_SEPARATION = "manual_separation_required"
EXCLUDE = "exclude_for_now"
ACTIVE = "active"


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
class Candidate:
    source_id: str
    source_family: str
    source_trust: str
    source_path: Path
    label_lines: list[str]
    canonical_class: str
    split: str
    md5: str
    width: int | None
    height: int | None
    file_size: int
    annotation_status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ReciclaMack detection dataset v2 staging.")
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
        "--output-root",
        type=Path,
        default=None,
        help="Default comes from sources JSON output_root.",
    )
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rename-only", action="store_true")
    parser.add_argument("--skip-folder-renames", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


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
    blocked_dirs = {"dismissed_dark", "dismissed_small", "__pycache__"}
    images: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_image(path):
            continue
        if any(part in blocked_dirs for part in path.parts):
            continue
        images.append(path)
    return sorted(images)


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


def class_map_by_source(rows: list[dict[str, str]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        source = (row.get("source_folder") or "").strip()
        target = (row.get("canonical_class") or "").strip()
        status = (row.get("status") or "").strip()
        if source and target and status != EXCLUDE:
            out[source] = target
    return out


def class_display_labels(config: dict[str, Any], rows: list[dict[str, str]]) -> dict[str, str]:
    labels = dict(config.get("portuguese_display_labels", {}))
    for row in rows:
        cls = (row.get("canonical_class") or "").strip()
        label = (row.get("portuguese_display_label") or "").strip()
        if cls and label and cls not in labels:
            labels[cls] = label
    return labels


def apply_folder_renames(workspace_root: Path, config: dict[str, Any], dry_run: bool) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in config.get("folder_renames", []):
        old_rel = str(item["old"])
        new_rel = str(item["new"])
        old_path = workspace_root / old_rel
        new_path = workspace_root / new_rel
        old_exists = old_path.exists()
        new_exists = new_path.exists()
        if old_exists and new_exists:
            status = "error_both_exist"
        elif old_exists:
            status = "would_rename" if dry_run else "renamed"
            if not dry_run:
                old_path.rename(new_path)
        elif new_exists:
            status = "already_correct"
        else:
            status = "missing"
        rows.append(
            {
                "old_path": old_rel,
                "new_path": new_rel,
                "status": status,
            }
        )
    errors = [row for row in rows if row["status"] == "error_both_exist"]
    if errors:
        joined = ", ".join(f"{row['old_path']} + {row['new_path']}" for row in errors)
        raise ValueError(f"Cannot rename typo folders because both old and corrected paths exist: {joined}")
    return rows


def parse_yolo_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        raw = line.strip()
        if raw.startswith("names:") and "[" in raw and "]" in raw:
            try:
                parsed = ast.literal_eval(raw.split(":", 1)[1].strip())
                if isinstance(parsed, list):
                    return {idx: str(name).strip() for idx, name in enumerate(parsed)}
            except Exception:
                pass

    in_names = False
    names: dict[int, str] = {}
    for line in lines:
        raw = line.rstrip()
        if raw.strip() == "names:":
            in_names = True
            continue
        if not in_names or ":" not in raw:
            continue
        left, right = raw.split(":", 1)
        left = left.strip()
        if left.isdigit():
            names[int(left)] = right.strip().strip("'\"")
    return names


def find_data_yaml(root: Path) -> Path | None:
    for name in ("data.yaml", "dataset.yaml"):
        path = root / name
        if path.exists():
            return path
    return None


def split_from_path(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "val" in lowered:
        return "valid"
    if "test" in lowered:
        return "test"
    return ""


def stable_split(md5: str, ratios: dict[str, float]) -> str:
    value = int(md5[:8], 16) / 0xFFFFFFFF
    train_cut = float(ratios.get("train", 0.7))
    valid_cut = train_cut + float(ratios.get("valid", 0.2))
    if value < train_cut:
        return "train"
    if value < valid_cut:
        return "valid"
    return "test"


def yolo_image_to_label(root: Path, image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [p.lower() for p in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    split = split_from_path(image_path)
    return root / split / "labels" / f"{image_path.stem}.txt"


def remap_yolo_label_lines(
    label_path: Path,
    source: Source,
    source_names: dict[int, str],
    source_class_map: dict[str, str],
    class_to_id: dict[str, int],
) -> tuple[list[str], list[str]]:
    out: list[str] = []
    warnings: list[str] = []
    if not label_path.exists():
        return out, [f"missing_label:{label_path.as_posix()}"]

    for raw in label_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            warnings.append(f"malformed_label:{label_path.as_posix()}:{line}")
            continue
        try:
            source_id = int(float(parts[0]))
        except ValueError:
            warnings.append(f"non_integer_class:{label_path.as_posix()}:{line}")
            continue

        canonical = source.canonical_class
        if not canonical:
            source_label = source_names.get(source_id, str(source_id)).strip()
            canonical = source_class_map.get(source_label) or source_class_map.get(source_label.lower())
        if not canonical or canonical not in class_to_id:
            warnings.append(f"unmapped_class:{label_path.as_posix()}:{parts[0]}")
            continue
        out.append(" ".join([str(class_to_id[canonical]), *parts[1:]]))
    return out, warnings


def collect_yolo_candidates(
    workspace_root: Path,
    source: Source,
    class_to_id: dict[str, int],
    source_class_map: dict[str, str],
) -> tuple[list[Candidate], dict[str, Any], list[str]]:
    root = workspace_root / source.root
    warnings: list[str] = []
    if not root.exists():
        return [], {"status": "missing"}, [f"missing_source:{source.id}"]

    data_yaml = find_data_yaml(root)
    source_names = parse_yolo_names(data_yaml) if data_yaml else {}
    images = iter_images(root)
    label_files = sorted((root / "labels").rglob("*.txt")) if (root / "labels").exists() else []
    stale_label_count = 0
    candidates: list[Candidate] = []

    image_stems = {img.stem for img in images}
    for label in label_files:
        if label.stem not in image_stems:
            stale_label_count += 1

    for image_path in images:
        split = split_from_path(image_path)
        if split not in CANONICAL_SPLITS:
            split = "train"
        label_path = yolo_image_to_label(root, image_path)
        label_lines, label_warnings = remap_yolo_label_lines(
            label_path, source, source_names, source_class_map, class_to_id
        )
        warnings.extend(label_warnings)
        if not label_lines:
            continue
        digest = md5_file(image_path)
        width, height = image_dimensions(image_path)
        canonical = source.canonical_class or "mixed"
        candidates.append(
            Candidate(
                source_id=source.id,
                source_family=source.source_family,
                source_trust=source.source_trust,
                source_path=image_path,
                label_lines=label_lines,
                canonical_class=canonical,
                split=split,
                md5=digest,
                width=width,
                height=height,
                file_size=image_path.stat().st_size,
                annotation_status=YOLO_REVIEW_LABEL,
            )
        )

    return candidates, {
        "status": "active",
        "image_count": len(images),
        "label_count": len(label_files),
        "stale_label_count": stale_label_count,
        "candidate_count": len(candidates),
    }, warnings


def collect_scraped_candidates(
    workspace_root: Path,
    source: Source,
    class_to_id: dict[str, int],
    split_ratio: dict[str, float],
) -> tuple[list[Candidate], dict[str, Any], list[str]]:
    root = workspace_root / source.root
    if not root.exists():
        return [], {"status": "missing"}, [f"missing_source:{source.id}"]
    if source.canonical_class not in class_to_id:
        return [], {"status": "unmapped_class"}, [f"unmapped_source_class:{source.id}:{source.canonical_class}"]

    images = iter_images(root)
    candidates: list[Candidate] = []
    cid = class_to_id[source.canonical_class]
    for image_path in images:
        digest = md5_file(image_path)
        width, height = image_dimensions(image_path)
        candidates.append(
            Candidate(
                source_id=source.id,
                source_family=source.source_family,
                source_trust=source.source_trust,
                source_path=image_path,
                label_lines=[f"{cid} 0.5 0.5 1.0 1.0"],
                canonical_class=source.canonical_class,
                split=stable_split(digest, split_ratio),
                md5=digest,
                width=width,
                height=height,
                file_size=image_path.stat().st_size,
                annotation_status=SCRAPED_SEED_LABEL,
            )
        )
    return candidates, {
        "status": "active",
        "image_count": len(images),
        "label_count": 0,
        "stale_label_count": 0,
        "candidate_count": len(candidates),
    }, []


def collect_manual_queue(workspace_root: Path, source: Source) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = workspace_root / source.root
    if not root.exists():
        return [], {"status": "missing"}
    images = iter_images(root)
    rows = [
        {
            "source_id": source.id,
            "source_family": source.source_family,
            "source_path": image.relative_to(workspace_root).as_posix(),
            "suggested_class": source.canonical_class,
            "reason": MANUAL_SEPARATION,
            "candidate_classes": "flat_monitor|crt_monitor|exclude",
        }
        for image in images
    ]
    return rows, {
        "status": MANUAL_SEPARATION,
        "image_count": len(images),
        "label_count": 0,
        "stale_label_count": 0,
        "candidate_count": 0,
    }


def candidate_quality(candidate: Candidate) -> tuple[int, int]:
    area = int(candidate.width or 0) * int(candidate.height or 0)
    return area, candidate.file_size


def merge_label_lines(candidates: list[Candidate]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for candidate in sorted(candidates, key=lambda item: (item.source_id, item.source_path.as_posix())):
        for line in candidate.label_lines:
            if line in seen:
                continue
            seen.add(line)
            merged.append(line)
    return merged


def dedupe_candidates(candidates: list[Candidate]) -> tuple[list[Candidate], dict[str, Any]]:
    grouped: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.md5].append(candidate)

    kept: list[Candidate] = []
    duplicate_rows: list[dict[str, Any]] = []
    for digest, group in sorted(grouped.items()):
        selected = sorted(group, key=candidate_quality, reverse=True)[0]
        merged_lines = merge_label_lines(group)
        merged_canonical = selected.canonical_class
        if len({candidate.canonical_class for candidate in group}) > 1:
            merged_canonical = "mixed"
        merged_status = selected.annotation_status
        if len({candidate.annotation_status for candidate in group}) > 1:
            merged_status = "|".join(sorted({candidate.annotation_status for candidate in group}))
        kept.append(
            Candidate(
                source_id=selected.source_id,
                source_family=selected.source_family,
                source_trust=selected.source_trust,
                source_path=selected.source_path,
                label_lines=merged_lines,
                canonical_class=merged_canonical,
                split=selected.split,
                md5=selected.md5,
                width=selected.width,
                height=selected.height,
                file_size=selected.file_size,
                annotation_status=merged_status,
            )
        )
        for candidate in group:
            if candidate is selected:
                continue
            duplicate_rows.append(
                {
                    "md5": digest,
                    "kept_source_path": selected.source_path.as_posix(),
                    "dropped_source_path": candidate.source_path.as_posix(),
                    "kept_source_id": selected.source_id,
                    "dropped_source_id": candidate.source_id,
                    "merged_label_count": len(merged_lines),
                }
            )
    return kept, {
        "input_candidates": len(candidates),
        "kept_candidates": len(kept),
        "dropped_exact_duplicates": len(duplicate_rows),
        "duplicates": duplicate_rows,
    }


def reset_output_dirs(output_root: Path, dry_run: bool) -> None:
    if dry_run:
        return
    for split in CANONICAL_SPLITS:
        split_dir = output_root / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
    (output_root / "manifests").mkdir(parents=True, exist_ok=True)


def export_candidates(
    workspace_root: Path,
    output_root: Path,
    candidates: list[Candidate],
    dry_run: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rel_source = candidate.source_path.relative_to(workspace_root).as_posix()
        out_name = f"{candidate.md5}{candidate.source_path.suffix.lower()}"
        out_img = output_root / candidate.split / "images" / out_name
        out_lbl = output_root / candidate.split / "labels" / f"{candidate.md5}.txt"
        if not dry_run:
            out_img.parent.mkdir(parents=True, exist_ok=True)
            out_lbl.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate.source_path, out_img)
            out_lbl.write_text("\n".join(candidate.label_lines) + "\n", encoding="utf-8")
        rows.append(
            {
                "md5": candidate.md5,
                "source_id": candidate.source_id,
                "source_family": candidate.source_family,
                "source_trust": candidate.source_trust,
                "source_path": rel_source,
                "output_image": out_img.relative_to(output_root).as_posix(),
                "output_label": out_lbl.relative_to(output_root).as_posix(),
                "split": candidate.split,
                "canonical_class": candidate.canonical_class,
                "annotation_status": candidate.annotation_status,
                "width": candidate.width or "",
                "height": candidate.height or "",
                "file_size": candidate.file_size,
            }
        )
    return rows


def write_class_id_map(output_root: Path, taxonomy: list[str], display_labels: dict[str, str]) -> None:
    rows = [
        {
            "class_id": idx,
            "canonical_class": name,
            "portuguese_display_label": display_labels.get(name, name),
        }
        for idx, name in enumerate(taxonomy)
    ]
    write_csv(output_root / "manifests" / "class_id_map.csv", rows, ["class_id", "canonical_class", "portuguese_display_label"])


def write_reports(
    output_root: Path,
    config: dict[str, Any],
    folder_rename_rows: list[dict[str, str]],
    inventory_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, Any]],
    rename_rows: list[dict[str, Any]],
    dedupe_report: dict[str, Any],
    warnings: list[str],
) -> None:
    manifests = output_root / "manifests"
    write_csv(manifests / "folder_rename_history.csv", folder_rename_rows, ["old_path", "new_path", "status"])
    write_csv(
        manifests / "source_inventory.csv",
        inventory_rows,
        [
            "source_id",
            "source_type",
            "source_family",
            "source_root",
            "action",
            "canonical_class",
            "status",
            "image_count",
            "label_count",
            "stale_label_count",
            "candidate_count",
        ],
    )
    write_csv(
        manifests / "manual_separation_queue.csv",
        manual_rows,
        ["source_id", "source_family", "source_path", "suggested_class", "reason", "candidate_classes"],
    )
    write_csv(
        manifests / "image_rename_history.csv",
        rename_rows,
        [
            "md5",
            "source_id",
            "source_family",
            "source_trust",
            "source_path",
            "output_image",
            "output_label",
            "split",
            "canonical_class",
            "annotation_status",
            "width",
            "height",
            "file_size",
        ],
    )

    class_counts = Counter(row["canonical_class"] for row in rename_rows)
    split_counts = Counter(row["split"] for row in rename_rows)
    quality = {
        "image_count": len(rename_rows),
        "class_counts": dict(sorted(class_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "warnings": warnings,
    }
    merge = {
        "dataset_version": config.get("dataset_version", "v2"),
        "dataset_policy_version": config.get("dataset_policy_version", "v2"),
        "output_root": output_root.as_posix(),
        "source_count": len(inventory_rows),
        "manual_separation_queue_size": len(manual_rows),
        "dedupe": {
            "input_candidates": dedupe_report["input_candidates"],
            "kept_candidates": dedupe_report["kept_candidates"],
            "dropped_exact_duplicates": dedupe_report["dropped_exact_duplicates"],
        },
        "annotation_policy": config.get("annotation_policy", {}),
        "qc": config.get("qc", {}),
    }
    (manifests / "dedupe_report.json").write_text(json.dumps(dedupe_report, ensure_ascii=False, indent=2), encoding="utf-8")
    (manifests / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (manifests / "merge_report.json").write_text(json.dumps(merge, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_no_typo_sources(sources: list[Source]) -> None:
    typo_tokens = ("batery", "tonner", "cartdrige")
    bad = [source.id for source in sources if any(token in source.id for token in typo_tokens)]
    bad += [source.root for source in sources if any(token in source.root for token in typo_tokens)]
    if bad:
        raise ValueError(f"Typo source names are not allowed in v2 registry: {bad}")


def run(
    sources_json: Path,
    class_map_csv: Path,
    output_root: Path | None = None,
    workspace_root: Path = Path.cwd(),
    dry_run: bool = False,
    rename_only: bool = False,
    skip_folder_renames: bool = False,
) -> None:
    workspace_root = workspace_root.resolve()
    config = load_json((workspace_root / sources_json).resolve())
    class_rows = read_csv((workspace_root / class_map_csv).resolve())
    sources = parse_sources(config)
    validate_no_typo_sources(sources)

    output_root_abs = (workspace_root / (output_root or Path(config["output_root"]))).resolve()
    manifests_root = output_root_abs / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)

    folder_rename_rows: list[dict[str, str]] = []
    if not skip_folder_renames:
        folder_rename_rows = apply_folder_renames(workspace_root, config, dry_run)
        write_csv(manifests_root / "folder_rename_history.csv", folder_rename_rows, ["old_path", "new_path", "status"])
    if rename_only:
        return

    taxonomy = [str(cls) for cls in config["taxonomy"]]
    class_to_id = {name: idx for idx, name in enumerate(taxonomy)}
    display_labels = class_display_labels(config, class_rows)
    source_class_map = class_map_by_source(class_rows)
    split_ratio = config.get("split_ratio", {"train": 0.7, "valid": 0.2, "test": 0.1})

    candidates: list[Candidate] = []
    manual_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for source in sources:
        root = workspace_root / source.root
        if source.action == EXCLUDE:
            image_count = len(iter_images(root)) if root.exists() else 0
            stats = {"status": EXCLUDE, "image_count": image_count, "label_count": 0, "stale_label_count": 0, "candidate_count": 0}
        elif source.action == MANUAL_SEPARATION:
            rows, stats = collect_manual_queue(workspace_root, source)
            manual_rows.extend(rows)
        elif source.type == "direct_yolo":
            source_candidates, stats, source_warnings = collect_yolo_candidates(
                workspace_root, source, class_to_id, source_class_map
            )
            candidates.extend(source_candidates)
            warnings.extend(source_warnings)
        elif source.type == "scraped_full_image_seed":
            source_candidates, stats, source_warnings = collect_scraped_candidates(
                workspace_root, source, class_to_id, split_ratio
            )
            candidates.extend(source_candidates)
            warnings.extend(source_warnings)
        else:
            stats = {"status": f"unsupported_type:{source.type}", "image_count": 0, "label_count": 0, "stale_label_count": 0, "candidate_count": 0}
            warnings.append(f"unsupported_source_type:{source.id}:{source.type}")

        inventory_rows.append(
            {
                "source_id": source.id,
                "source_type": source.type,
                "source_family": source.source_family,
                "source_root": source.root,
                "action": source.action,
                "canonical_class": source.canonical_class,
                "status": stats.get("status", ""),
                "image_count": stats.get("image_count", 0),
                "label_count": stats.get("label_count", 0),
                "stale_label_count": stats.get("stale_label_count", 0),
                "candidate_count": stats.get("candidate_count", 0),
            }
        )

    kept, dedupe_report = dedupe_candidates(candidates)
    reset_output_dirs(output_root_abs, dry_run)
    rename_rows = export_candidates(workspace_root, output_root_abs, kept, dry_run)

    write_class_id_map(output_root_abs, taxonomy, display_labels)
    if not dry_run:
        write_detection_data_yaml(output_root_abs / "data.yaml", output_root_abs, taxonomy)
    write_reports(
        output_root_abs,
        config,
        folder_rename_rows,
        inventory_rows,
        manual_rows,
        rename_rows,
        dedupe_report,
        warnings,
    )

    print(f"Output: {output_root_abs.as_posix()}")
    print(f"Candidates: {len(candidates)}")
    print(f"Kept: {len(kept)}")
    print(f"Manual separation queue: {len(manual_rows)}")
    print(f"Warnings: {len(warnings)}")


def main() -> None:
    args = parse_args()
    run(
        sources_json=args.sources_json,
        class_map_csv=args.class_map_csv,
        output_root=args.output_root,
        workspace_root=args.workspace_root,
        dry_run=args.dry_run,
        rename_only=args.rename_only,
        skip_folder_renames=args.skip_folder_renames,
    )


if __name__ == "__main__":
    main()
