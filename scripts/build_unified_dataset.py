#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NEGATIVE_LABEL = "negative"
REVIEW_LABEL = "review"
DEFAULT_ANNOTATION_POLICY_V2 = {
    "instance_mode": "multi_instance_required_when_visible",
    "unlabeled_object_policy": "warn_review_queue",
}
DEFAULT_QC_POLICY_V2 = {
    "missing_instance_policy": "warn",
}
LEGACY_SINGLE_OBJECT_REASONS = {
    "single_object_problem",
    "single_object_only",
    "one_box_only",
    "single_instance_only",
}


@dataclass
class Record:
    unique_id: str
    source_dataset: str
    source_relpath: str
    original_filename: str
    source_path: Path
    original_label: str
    mapped_label: str
    is_negative: bool
    review_status: str
    reason: str
    file_hash_sha1: str
    phash: str
    dedup_group_id: str
    file_size: int
    width: int | None
    height: int | None
    split: str = "pending"
    final_filename: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified traceable imagefolder dataset.")
    parser.add_argument(
        "--config",
        default="model_pipeline/config/unified_dataset_config.json",
        type=Path,
        help="Path to processing config JSON.",
    )
    parser.add_argument(
        "--output-root",
        default="prod_datasets/unified/v1",
        type=Path,
        help="Output dataset root folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute manifests and reports without copying images.",
    )
    parser.add_argument(
        "--review-decisions-dir",
        type=Path,
        default=None,
        help="Optional directory containing manual review_reduced.csv files (searched recursively).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_review_decisions(decisions_dir: Path | None) -> dict[str, str]:
    if decisions_dir is None:
        return {}
    if not decisions_dir.exists():
        return {}

    decisions: dict[str, str] = {}
    for csv_path in decisions_dir.rglob("review_reduced.csv"):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    uid = (row.get("unique_id") or "").strip()
                    decision = (row.get("decision") or "").strip()
                    if uid and decision:
                        decisions[uid] = decision
        except Exception:
            continue
    return decisions


def normalize_label(raw: str) -> str:
    cleaned = raw.strip().lower().replace("-", "_").replace(" ", "_")
    compact = []
    for ch in cleaned:
        compact.append(ch if ch.isalnum() or ch == "_" else "_")
    out = "".join(compact)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


def get_policy_config(config: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, str]]:
    dataset_policy_version = str(config.get("dataset_policy_version", "v1")).strip().lower()
    is_v2 = dataset_policy_version == "v2"

    annotation_raw = config.get("annotation_policy", {})
    if not isinstance(annotation_raw, dict):
        annotation_raw = {}
    qc_raw = config.get("qc", {})
    if not isinstance(qc_raw, dict):
        qc_raw = {}

    annotation_policy: dict[str, str] = {
        "instance_mode": str(
            annotation_raw.get(
                "instance_mode",
                DEFAULT_ANNOTATION_POLICY_V2["instance_mode"] if is_v2 else "legacy",
            )
        ),
        "unlabeled_object_policy": str(
            annotation_raw.get(
                "unlabeled_object_policy",
                DEFAULT_ANNOTATION_POLICY_V2["unlabeled_object_policy"] if is_v2 else "off",
            )
        ),
    }
    qc_policy: dict[str, str] = {
        "missing_instance_policy": str(
            qc_raw.get(
                "missing_instance_policy",
                DEFAULT_QC_POLICY_V2["missing_instance_policy"] if is_v2 else "off",
            )
        )
    }
    return dataset_policy_version, annotation_policy, qc_policy


def normalize_review_reason(reason: str, annotation_policy: dict[str, str]) -> str:
    if not reason:
        return reason
    if annotation_policy.get("instance_mode") != "multi_instance_required_when_visible":
        return reason

    normalized = normalize_label(reason)
    if normalized in LEGACY_SINGLE_OBJECT_REASONS:
        return "possible_missing_instances"
    if "occlusion" in normalized:
        return "ambiguous_occlusion"
    if any(token in normalized for token in ("crowd", "dense", "multiple", "multi_object")):
        return "crowded_scene_needs_review"
    if "single" in normalized and "object" in normalized:
        return "possible_missing_instances"
    return reason


def sha1_file(path: Path) -> str:
    hasher = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_phash(path: Path) -> str:
    if Image is None:
        return sha1_file(path)[:16]
    try:
        with Image.open(path) as img:
            img = img.convert("L").resize((8, 8))
            pixels = list(img.getdata())
            avg = sum(pixels) / len(pixels)
            bits = "".join("1" if px >= avg else "0" for px in pixels)
            return f"{int(bits, 2):016x}"
    except Exception:
        return sha1_file(path)[:16]


def image_dimensions(path: Path) -> tuple[int | None, int | None]:
    if Image is None:
        return None, None
    try:
        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        return None, None


def hamming_distance_hex(a: str, b: str) -> int:
    try:
        ax = int(a, 16)
        bx = int(b, 16)
    except Exception:
        return 64 if a != b else 0
    return bin(ax ^ bx).count("1")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def canonicalize_label(
    source_id: str,
    normalized_label: str,
    config: dict[str, Any],
    annotation_policy: dict[str, str],
) -> tuple[str, bool, str, str]:
    label_map = config.get("label_map", {})
    component_map = config.get("component_label_map", {})
    review_map = config.get("review_map", {})

    mapped = label_map.get(normalized_label, normalized_label)
    if source_id == "electronic_components":
        mapped = component_map.get(normalized_label, NEGATIVE_LABEL)

    review_status = "approved"
    reason = ""
    if mapped in config.get("negative_labels", [NEGATIVE_LABEL]) or mapped == NEGATIVE_LABEL:
        return NEGATIVE_LABEL, True, review_status, "mapped_negative"
    if normalized_label in review_map:
        review_status = "pending"
        reason = normalize_review_reason(str(review_map[normalized_label]), annotation_policy)
    return mapped, False, review_status, reason


def enumerate_folder_source(root: Path, source_id: str, config: dict[str, Any]) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for file in root.rglob("*"):
        if not file.is_file() or not is_image(file):
            continue
        if file.name.lower().endswith(".md"):
            continue
        label = normalize_label(file.parent.name)
        samples.append((file, label))
    return samples


def parse_garbage_yolo_source(root: Path, source_id: str, src_cfg: dict[str, Any]) -> list[tuple[Path, str]]:
    allowed = set(src_cfg.get("allowed_negative_ids", []))
    blocked = set(src_cfg.get("blocked_ids", []))
    samples: list[tuple[Path, str]] = []
    for split in ("train", "valid", "test"):
        labels_dir = root / split / "labels"
        images_dir = root / split / "images"
        if not labels_dir.exists() or not images_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            try:
                lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            except Exception:
                continue
            if not lines:
                continue
            class_ids = {int(ln.split()[0]) for ln in lines if ln.split()}
            if class_ids.intersection(blocked):
                continue
            if not class_ids.issubset(allowed):
                continue
            img_path = images_dir / f"{label_path.stem}.jpg"
            if not img_path.exists():
                alternatives = list(images_dir.glob(f"{label_path.stem}.*"))
                img_path = alternatives[0] if alternatives else img_path
            if img_path.exists() and is_image(img_path):
                samples.append((img_path, NEGATIVE_LABEL))
    return samples


def build_records(config: dict[str, Any], workspace_root: Path) -> list[Record]:
    records: list[Record] = []
    quality_cfg = config.get("quality", {})
    min_size = int(quality_cfg.get("min_file_size_bytes", 0))
    _, annotation_policy, _ = get_policy_config(config)

    for source in config.get("sources", []):
        source_id = source["id"]
        source_root = (workspace_root / source["root"]).resolve()
        if not source_root.exists():
            continue

        if source["type"] == "folder":
            raw_samples = enumerate_folder_source(source_root, source_id, config)
        elif source["type"] == "garbage_yolo_negative":
            raw_samples = parse_garbage_yolo_source(source_root, source_id, source)
        else:
            continue

        for path, raw_label in raw_samples:
            try:
                size = path.stat().st_size
            except Exception:
                continue
            if size < min_size:
                continue

            normalized = normalize_label(raw_label)
            mapped_label, is_negative, review_status, reason = canonicalize_label(
                source_id, normalized, config, annotation_policy
            )
            taxonomy = set(config.get("taxonomy", []))
            if not is_negative and mapped_label not in taxonomy:
                mapped_label = NEGATIVE_LABEL
                is_negative = True
                reason = "out_of_taxonomy_default_negative"

            sha1 = sha1_file(path)
            phash = compute_phash(path)
            width, height = image_dimensions(path)
            rel = path.relative_to(workspace_root).as_posix()
            uid = hashlib.sha1(f"{source_id}:{rel}".encode("utf-8")).hexdigest()[:16]
            records.append(
                Record(
                    unique_id=uid,
                    source_dataset=source_id,
                    source_relpath=rel,
                    original_filename=path.name,
                    source_path=path,
                    original_label=normalized,
                    mapped_label=mapped_label,
                    is_negative=is_negative,
                    review_status=review_status,
                    reason=reason,
                    file_hash_sha1=sha1,
                    phash=phash,
                    dedup_group_id="",
                    file_size=size,
                    width=width,
                    height=height,
                )
            )
    return records


def deduplicate(records: list[Record], dedup_cfg: dict[str, Any]) -> tuple[list[Record], dict[str, int]]:
    if not dedup_cfg.get("enabled", True):
        for rec in records:
            rec.dedup_group_id = rec.file_hash_sha1[:12]
        return records, {"input_records": len(records), "kept_records": len(records), "dropped_records": 0}

    by_exact: dict[str, list[Record]] = defaultdict(list)
    for rec in records:
        by_exact[rec.file_hash_sha1].append(rec)

    representatives: list[Record] = []
    dropped_exact = 0
    for sha, group in by_exact.items():
        group_sorted = sorted(
            group,
            key=lambda x: ((x.width or 0) * (x.height or 0), x.file_size),
            reverse=True,
        )
        keep = group_sorted[0]
        keep.dedup_group_id = f"exact_{sha[:12]}"
        representatives.append(keep)
        dropped_exact += len(group_sorted) - 1

    threshold = int(dedup_cfg.get("near_hamming_threshold", 5))
    bucket_len = int(dedup_cfg.get("bucket_prefix_len", 4))
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    grouped: dict[str, list[Record]] = defaultdict(list)

    for rec in sorted(representatives, key=lambda r: r.unique_id):
        pref = rec.phash[:bucket_len]
        assigned_group = None
        for known_phash, group_id in buckets[pref]:
            if hamming_distance_hex(rec.phash, known_phash) <= threshold:
                assigned_group = group_id
                break
        if assigned_group is None:
            assigned_group = f"near_{rec.phash[:12]}"
            buckets[pref].append((rec.phash, assigned_group))
        rec.dedup_group_id = assigned_group
        grouped[assigned_group].append(rec)

    final_records: list[Record] = []
    dropped_near = 0
    for group_id, group in grouped.items():
        group_sorted = sorted(
            group,
            key=lambda x: ((x.width or 0) * (x.height or 0), x.file_size),
            reverse=True,
        )
        keep = group_sorted[0]
        keep.dedup_group_id = group_id
        final_records.append(keep)
        dropped_near += len(group_sorted) - 1

    return final_records, {
        "input_records": len(records),
        "kept_records": len(final_records),
        "dropped_records": dropped_exact + dropped_near,
        "dropped_exact": dropped_exact,
        "dropped_near": dropped_near,
    }


def apply_manual_review_decisions(
    records: list[Record], decisions: dict[str, str], config: dict[str, Any]
) -> dict[str, int]:
    if not decisions:
        return {"manual_decisions_loaded": 0, "manual_decisions_applied": 0}

    taxonomy = set(config.get("taxonomy", []))
    applied = 0
    for rec in records:
        if rec.review_status == "approved":
            continue
        decision = decisions.get(rec.unique_id)
        if not decision:
            continue
        choice = normalize_label(decision)
        if choice in taxonomy:
            rec.mapped_label = choice
            rec.is_negative = False
            rec.review_status = "approved"
            rec.reason = "manual_review_decision"
            applied += 1
            continue
        if choice == NEGATIVE_LABEL:
            rec.mapped_label = NEGATIVE_LABEL
            rec.is_negative = True
            rec.review_status = "approved"
            rec.reason = "manual_review_decision_negative"
            applied += 1
            continue
        rec.mapped_label = NEGATIVE_LABEL
        rec.is_negative = True
        rec.review_status = "approved"
        rec.reason = "manual_review_out_of_taxonomy_as_negative"
        applied += 1

    return {"manual_decisions_loaded": len(decisions), "manual_decisions_applied": applied}


def split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    splits = ["train", "valid", "test"]
    counts = {k: int(total * ratios[k]) for k in splits}
    remainder = total - sum(counts.values())
    fracs = sorted(((k, total * ratios[k] - counts[k]) for k in splits), key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        counts[fracs[i % len(fracs)][0]] += 1
    return counts


def assign_splits(records: list[Record], config: dict[str, Any]) -> list[Record]:
    random.seed(int(config.get("seed", 42)))
    ratios = config.get("split_ratio", {"train": 0.7, "valid": 0.2, "test": 0.1})
    pos = [r for r in records if not r.is_negative and r.review_status == "approved"]
    neg = [r for r in records if r.is_negative and r.review_status == "approved"]

    by_class: dict[str, list[Record]] = defaultdict(list)
    for rec in pos:
        by_class[rec.mapped_label].append(rec)

    split_records: dict[str, list[Record]] = {"train": [], "valid": [], "test": []}
    for label, items in sorted(by_class.items()):
        random.shuffle(items)
        counts = split_counts(len(items), ratios)
        idx = 0
        for split in ("train", "valid", "test"):
            for rec in items[idx : idx + counts[split]]:
                rec.split = split
                split_records[split].append(rec)
            idx += counts[split]

    # Keep negative ratio close to 20% per split.
    neg_target_ratio = float(config.get("negative_ratio_target", 0.2))
    random.shuffle(neg)
    neg_idx = 0
    for split in ("train", "valid", "test"):
        p = len(split_records[split])
        target_neg = int(round((p * neg_target_ratio) / (1.0 - neg_target_ratio))) if p > 0 else 0
        take = min(target_neg, len(neg) - neg_idx)
        for rec in neg[neg_idx : neg_idx + take]:
            rec.split = split
            split_records[split].append(rec)
        neg_idx += take

    for split in ("train", "valid", "test"):
        random.shuffle(split_records[split])

    return [r for split in ("train", "valid", "test") for r in split_records[split]]


def apply_class_rebalance(records: list[Record], config: dict[str, Any]) -> tuple[list[Record], dict[str, Any]]:
    rebalance_cfg = config.get("rebalance", {})
    if not rebalance_cfg.get("enabled", False):
        return records, {
            "enabled": False,
            "target_class": None,
            "max_multiplier_of_next_class": None,
            "before_count": 0,
            "after_count": 0,
            "next_class": None,
            "next_class_count": 0,
            "cap_count": 0,
            "dropped_count": 0,
        }

    target_class = normalize_label(str(rebalance_cfg.get("target_class", "")).strip())
    multiplier = float(rebalance_cfg.get("max_multiplier_of_next_class", 1.0))
    if not target_class or multiplier <= 0:
        return records, {
            "enabled": True,
            "target_class": target_class or None,
            "max_multiplier_of_next_class": multiplier,
            "before_count": 0,
            "after_count": 0,
            "next_class": None,
            "next_class_count": 0,
            "cap_count": 0,
            "dropped_count": 0,
        }

    positives = [r for r in records if not r.is_negative and r.review_status == "approved"]
    counts = Counter(r.mapped_label for r in positives)
    target_count = counts.get(target_class, 0)
    if target_count == 0:
        return records, {
            "enabled": True,
            "target_class": target_class,
            "max_multiplier_of_next_class": multiplier,
            "before_count": 0,
            "after_count": 0,
            "next_class": None,
            "next_class_count": 0,
            "cap_count": 0,
            "dropped_count": 0,
        }

    non_target_counts = [(label, cnt) for label, cnt in counts.items() if label != target_class]
    if not non_target_counts:
        return records, {
            "enabled": True,
            "target_class": target_class,
            "max_multiplier_of_next_class": multiplier,
            "before_count": target_count,
            "after_count": target_count,
            "next_class": None,
            "next_class_count": 0,
            "cap_count": target_count,
            "dropped_count": 0,
        }

    next_class, next_count = sorted(non_target_counts, key=lambda x: x[1], reverse=True)[0]
    cap_count = max(1, int(next_count * multiplier))
    if target_count <= cap_count:
        return records, {
            "enabled": True,
            "target_class": target_class,
            "max_multiplier_of_next_class": multiplier,
            "before_count": target_count,
            "after_count": target_count,
            "next_class": next_class,
            "next_class_count": next_count,
            "cap_count": cap_count,
            "dropped_count": 0,
        }

    random.seed(int(config.get("seed", 42)))
    target_records = [r for r in positives if r.mapped_label == target_class]
    target_ids = {r.unique_id for r in target_records}
    non_target_records = [r for r in records if r.unique_id not in target_ids]
    random.shuffle(target_records)
    kept_target = target_records[:cap_count]
    dropped_count = len(target_records) - len(kept_target)

    rebalanced = non_target_records + kept_target
    return rebalanced, {
        "enabled": True,
        "target_class": target_class,
        "max_multiplier_of_next_class": multiplier,
        "before_count": target_count,
        "after_count": len(kept_target),
        "next_class": next_class,
        "next_class_count": next_count,
        "cap_count": cap_count,
        "dropped_count": dropped_count,
    }


def write_split_manifest(path: Path, rows: list[Record]) -> None:
    fields = [
        "unique_id",
        "split",
        "mapped_label",
        "is_negative",
        "review_status",
        "reason",
        "source_dataset",
        "source_relpath",
        "original_filename",
        "file_hash_sha1",
        "phash",
        "dedup_group_id",
        "file_size",
        "width",
        "height",
        "final_filename",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "unique_id": r.unique_id,
                    "split": r.split,
                    "mapped_label": r.mapped_label,
                    "is_negative": str(r.is_negative).lower(),
                    "review_status": r.review_status,
                    "reason": r.reason,
                    "source_dataset": r.source_dataset,
                    "source_relpath": r.source_relpath,
                    "original_filename": r.original_filename,
                    "file_hash_sha1": r.file_hash_sha1,
                    "phash": r.phash,
                    "dedup_group_id": r.dedup_group_id,
                    "file_size": r.file_size,
                    "width": r.width if r.width is not None else "",
                    "height": r.height if r.height is not None else "",
                    "final_filename": r.final_filename,
                }
            )


def export_dataset(
    all_records: list[Record], review_records: list[Record], output_root: Path, dry_run: bool
) -> dict[str, Any]:
    images_root = output_root / "images"
    manifests_root = output_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        for split_dir in (output_root / "train", output_root / "valid", output_root / "test"):
            if split_dir.exists():
                shutil.rmtree(split_dir)
        output_root.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[Record]] = defaultdict(list)
    for rec in all_records:
        by_split[rec.split].append(rec)

    for split in ("train", "valid", "test"):
        rows = by_split.get(split, [])
        for rec in rows:
            label_dir_name = NEGATIVE_LABEL if rec.is_negative else rec.mapped_label
            hash8 = rec.file_hash_sha1[:8]
            stem = Path(rec.original_filename).stem
            ext = rec.source_path.suffix.lower()
            rec.final_filename = f"{rec.source_dataset}__{label_dir_name}__{split}__{hash8}__{stem}{ext}"
            if not dry_run:
                out_dir = output_root / split / "images" / label_dir_name
                out_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(rec.source_path, out_dir / rec.final_filename)
        write_split_manifest(manifests_root / f"{split}.csv", rows)

    write_split_manifest(manifests_root / "review_queue.csv", review_records)

    class_counts = Counter(r.mapped_label for r in all_records if not r.is_negative)
    low_support_threshold = None
    low_support_classes: list[str] = []
    # populated by caller, patched in summary step.

    return {
        "total_exported": len(all_records),
        "per_split_counts": {k: len(by_split.get(k, [])) for k in ("train", "valid", "test")},
        "class_counts": dict(sorted(class_counts.items())),
        "low_support_threshold": low_support_threshold,
        "low_support_classes": low_support_classes,
    }


def summarize(
    records: list[Record],
    review_records: list[Record],
    dedup_stats: dict[str, int],
    export_stats: dict[str, Any],
    config: dict[str, Any],
    rebalance_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_policy_version, annotation_policy, qc_policy = get_policy_config(config)
    by_source = Counter(r.source_dataset for r in records)
    by_source_label = Counter((r.source_dataset, r.mapped_label) for r in records)
    neg_by_source_reason = Counter((r.source_dataset, r.reason) for r in records if r.is_negative)
    split_label_counts = Counter((r.split, NEGATIVE_LABEL if r.is_negative else r.mapped_label) for r in records)

    split_counts_total = Counter(r.split for r in records)
    split_neg_counts = Counter(r.split for r in records if r.is_negative)
    split_neg_ratio = {
        split: (split_neg_counts[split] / split_counts_total[split] if split_counts_total[split] else 0.0)
        for split in ("train", "valid", "test")
    }

    min_support = int(config.get("minimum_class_support", 0))
    class_counts = Counter(r.mapped_label for r in records if not r.is_negative)
    low_support = sorted([lbl for lbl, cnt in class_counts.items() if cnt < min_support])
    export_stats["low_support_threshold"] = min_support
    export_stats["low_support_classes"] = low_support

    return {
        "config": {
            "split_ratio": config.get("split_ratio"),
            "negative_ratio_target": config.get("negative_ratio_target"),
            "negative_ratio_tolerance": config.get("negative_ratio_tolerance"),
            "seed": config.get("seed"),
            "dataset_policy_version": dataset_policy_version,
            "annotation_policy": annotation_policy,
            "qc": qc_policy,
        },
        "rebalance_stats": rebalance_stats or {},
        "dedup_stats": dedup_stats,
        "export_stats": export_stats,
        "review_queue_size": len(review_records),
        "counts_by_source": dict(sorted(by_source.items())),
        "counts_by_source_and_label": {f"{s}::{l}": c for (s, l), c in sorted(by_source_label.items())},
        "negative_breakdown": {f"{s}::{r}": c for (s, r), c in sorted(neg_by_source_reason.items())},
        "split_label_counts": {f"{sp}::{lbl}": c for (sp, lbl), c in sorted(split_label_counts.items())},
        "split_negative_ratio": split_neg_ratio,
        "low_support_classes": low_support,
    }


def run(config_path: Path, output_root: Path, dry_run: bool) -> None:
    workspace_root = Path.cwd()
    config = load_json((workspace_root / config_path).resolve())

    records = build_records(config, workspace_root)
    decision_dir = None
    # pulled from CLI by main() via dynamic attribute fallback
    # to keep run() directly callable in tests.
    global _RUNTIME_REVIEW_DECISIONS_DIR  # type: ignore
    try:
        decision_dir = _RUNTIME_REVIEW_DECISIONS_DIR
    except Exception:
        decision_dir = None
    decisions = load_review_decisions(decision_dir)
    manual_stats = apply_manual_review_decisions(records, decisions, config)
    deduped, dedup_stats = deduplicate(records, config.get("dedup", {}))
    review_records = [r for r in deduped if r.review_status != "approved"]
    approved = [r for r in deduped if r.review_status == "approved"]
    approved, rebalance_stats = apply_class_rebalance(approved, config)

    split_records = assign_splits(approved, config)
    output_root_abs = (workspace_root / output_root).resolve()
    export_stats = export_dataset(split_records, review_records, output_root_abs, dry_run)
    summary = summarize(split_records, review_records, dedup_stats, export_stats, config, rebalance_stats)
    summary["manual_review_stats"] = manual_stats
    summary["output_root"] = output_root_abs.as_posix()

    manifests_root = output_root_abs / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    (manifests_root / "inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Processed records: {dedup_stats['input_records']}")
    print(f"Exported records: {export_stats['total_exported']}")
    print(f"Review queue: {len(review_records)}")


def main() -> None:
    args = parse_args()
    global _RUNTIME_REVIEW_DECISIONS_DIR  # type: ignore
    _RUNTIME_REVIEW_DECISIONS_DIR = args.review_decisions_dir
    run(args.config, args.output_root, args.dry_run)


if __name__ == "__main__":
    main()
