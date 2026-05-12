#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from dataset_layout import CANONICAL_SPLITS, detection_images_dir, detection_labels_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO detection dataset structure and counts.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to detection dataset root.")
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Optional data.yaml path to read class names/order from (recommended).",
    )
    parser.add_argument(
        "--expected-classes",
        type=str,
        default="celular,notebook,televisao,bateria,placa_eletronica,capacitor,cabo,router,usb_stick,player,impressora",
        help="Comma-separated expected class names in class-id order.",
    )
    parser.add_argument(
        "--required-valid-classes",
        type=str,
        default="",
        help="Comma-separated class names that must appear at least once in valid instances.",
    )
    parser.add_argument(
        "--required-test-classes",
        type=str,
        default="",
        help="Comma-separated class names that must appear at least once in test instances.",
    )
    parser.add_argument(
        "--skip-coverage-check",
        action="store_true",
        help="Skip generic valid/test all-class coverage warnings.",
    )
    parser.add_argument(
        "--missing-instance-policy",
        choices=["warn", "error", "off"],
        default="warn",
        help="How to handle suspicious missing-instance patterns (warn, error, or off).",
    )
    parser.add_argument(
        "--min-multi-box-ratio",
        type=float,
        default=0.01,
        help="Warn/error when split multi-box ratio falls below this threshold.",
    )
    parser.add_argument(
        "--max-empty-label-ratio",
        type=float,
        default=0.40,
        help="Warn/error when split empty-label ratio exceeds this threshold.",
    )
    parser.add_argument(
        "--min-images-for-qc",
        type=int,
        default=100,
        help="Minimum split image count before multi-instance ratio checks are applied.",
    )
    parser.add_argument(
        "--min-source-images-for-drop-check",
        type=int,
        default=30,
        help="Minimum source image count for source-level multi-box drop warnings.",
    )
    parser.add_argument(
        "--source-multi-ratio-drop-factor",
        type=float,
        default=0.35,
        help="Source-level warning threshold as a fraction of split global multi-box ratio.",
    )
    parser.add_argument(
        "--output-review-queue",
        type=Path,
        default=None,
        help="Optional CSV output path for QC review queue (possible missing instances/crowded scenes).",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output report path.")
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def parse_data_yaml_classes(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    # Roboflow often exports: names: ['a', 'b', ...]
    for line in lines:
        raw = line.strip()
        if raw.startswith("names:") and "[" in raw and "]" in raw:
            rhs = raw.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(rhs)
                if isinstance(parsed, list) and parsed:
                    return [str(x).strip() for x in parsed]
            except Exception:
                pass

    in_names = False
    names_map: dict[int, str] = {}
    for line in lines:
        raw = line.rstrip()
        if not raw.strip():
            continue
        if raw.strip() == "names:":
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
            names_map[int(left)] = right
    if not names_map:
        raise ValueError(f"Could not parse names from data.yaml: {path.as_posix()}")
    return [names_map[i] for i in sorted(names_map)]


def infer_source_dataset(stem: str) -> str:
    # Expected naming pattern in this repository:
    # <source>__<label>__<split>__<hash>__<orig_stem>
    parts = stem.split("__")
    if len(parts) >= 2 and parts[0]:
        return parts[0]
    return "unknown"


def add_qc_signal(
    policy: str,
    warnings: list[str],
    errors: list[str],
    message: str,
) -> None:
    if policy == "error":
        errors.append(message)
    elif policy == "warn":
        warnings.append(message)


def write_review_queue(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "split",
        "source_dataset",
        "image_path",
        "label_path",
        "reason",
        "box_count",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.dataset_root
    if args.data_yaml:
        classes = parse_data_yaml_classes(args.data_yaml)
    else:
        classes = [c.strip() for c in args.expected_classes.split(",") if c.strip()]
    expected_ids = set(range(len(classes)))
    required_valid_classes = [c.strip() for c in args.required_valid_classes.split(",") if c.strip()]
    required_test_classes = [c.strip() for c in args.required_test_classes.split(",") if c.strip()]

    report: dict[str, object] = {
        "dataset_root": root.as_posix(),
        "expected_classes": classes,
        "splits": {},
        "errors": [],
        "warnings": [],
    }

    errors: list[str] = []
    warnings: list[str] = []
    split_instance_counts: dict[str, Counter[int]] = {}
    split_image_counts: dict[str, int] = {}
    split_box_count_distribution: dict[str, Counter[int]] = {}
    review_queue_rows: list[dict[str, str]] = []
    source_stats: dict[str, dict[str, Counter[str] | int]] = {}

    for split in CANONICAL_SPLITS:
        images_dir = detection_images_dir(root, split)
        labels_dir = detection_labels_dir(root, split)
        if not images_dir.exists() or not labels_dir.exists():
            errors.append(f"Missing required split folders for '{split}'.")
            continue

        images = sorted([p for p in images_dir.rglob("*") if p.is_file() and is_image(p)])
        split_image_counts[split] = len(images)
        class_counter: Counter[int] = Counter()
        missing_label_files = 0
        box_count_dist: Counter[int] = Counter()
        source_total: Counter[str] = Counter()
        source_multi: Counter[str] = Counter()

        for img in images:
            rel = img.relative_to(images_dir)
            txt = (labels_dir / rel).with_suffix(".txt")
            source_dataset = infer_source_dataset(img.stem)
            source_total[source_dataset] += 1
            if not txt.exists():
                missing_label_files += 1
                box_count_dist[0] += 1
                if args.missing_instance_policy != "off":
                    review_queue_rows.append(
                        {
                            "split": split,
                            "source_dataset": source_dataset,
                            "image_path": img.as_posix(),
                            "label_path": txt.as_posix(),
                            "reason": "possible_missing_instances",
                            "box_count": "0",
                            "notes": "missing_label_file",
                        }
                    )
                continue
            lines = [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
            image_box_count = 0
            for ln in lines:
                parts = ln.split()
                if len(parts) != 5:
                    warnings.append(f"Malformed label row in {txt.as_posix()}: '{ln}'")
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    warnings.append(f"Non-integer class id in {txt.as_posix()}: '{ln}'")
                    continue
                if cid not in expected_ids:
                    errors.append(f"Out-of-range class id {cid} in {txt.as_posix()}")
                    continue
                class_counter[cid] += 1
                image_box_count += 1

            box_count_dist[image_box_count] += 1
            if image_box_count > 1:
                source_multi[source_dataset] += 1
            if args.missing_instance_policy != "off":
                if image_box_count == 0:
                    review_queue_rows.append(
                        {
                            "split": split,
                            "source_dataset": source_dataset,
                            "image_path": img.as_posix(),
                            "label_path": txt.as_posix(),
                            "reason": "possible_missing_instances",
                            "box_count": "0",
                            "notes": "empty_label_file",
                        }
                    )
                elif image_box_count >= 5:
                    review_queue_rows.append(
                        {
                            "split": split,
                            "source_dataset": source_dataset,
                            "image_path": img.as_posix(),
                            "label_path": txt.as_posix(),
                            "reason": "crowded_scene_needs_review",
                            "box_count": str(image_box_count),
                            "notes": "high_box_density",
                        }
                    )

        split_instance_counts[split] = class_counter
        split_box_count_distribution[split] = box_count_dist
        source_stats[split] = {
            "source_total": source_total,
            "source_multi": source_multi,
        }
        if missing_label_files > 0:
            warnings.append(f"{split}: {missing_label_files} images without matching label txt.")

    # Coverage checks for valid/test
    if not args.skip_coverage_check:
        for split in ("valid", "test"):
            if split not in split_instance_counts:
                continue
            present = set(split_instance_counts[split].keys())
            missing = sorted(expected_ids - present)
            if missing:
                warnings.append(
                    f"{split}: missing classes by id {missing} ({[classes[i] for i in missing]})."
                )

    # Required class checks for valid/test
    class_to_id = {name: idx for idx, name in enumerate(classes)}
    for required in required_valid_classes:
        if required not in class_to_id:
            errors.append(f"required-valid class '{required}' not found in expected class list.")
            continue
        cid = class_to_id[required]
        if split_instance_counts.get("valid", Counter()).get(cid, 0) <= 0:
            errors.append(f"valid split has no instances for required class '{required}'.")

    for required in required_test_classes:
        if required not in class_to_id:
            errors.append(f"required-test class '{required}' not found in expected class list.")
            continue
        cid = class_to_id[required]
        if split_instance_counts.get("test", Counter()).get(cid, 0) <= 0:
            errors.append(f"test split has no instances for required class '{required}'.")

    # Multi-instance QC checks (warn/error/off)
    if args.missing_instance_policy != "off":
        for split in CANONICAL_SPLITS:
            image_count = split_image_counts.get(split, 0)
            dist = split_box_count_distribution.get(split, Counter())
            zero_count = int(dist.get(0, 0))
            one_count = int(dist.get(1, 0))
            multi_count = int(sum(cnt for box_count, cnt in dist.items() if box_count > 1))
            labeled_count = one_count + multi_count
            multi_ratio = (multi_count / labeled_count) if labeled_count else 0.0
            empty_ratio = (zero_count / image_count) if image_count else 0.0

            if image_count >= args.min_images_for_qc and multi_ratio < args.min_multi_box_ratio:
                add_qc_signal(
                    args.missing_instance_policy,
                    warnings,
                    errors,
                    (
                        f"{split}: low multi-box ratio {multi_ratio:.4f} "
                        f"(threshold {args.min_multi_box_ratio:.4f})."
                    ),
                )

            if image_count >= args.min_images_for_qc and empty_ratio > args.max_empty_label_ratio:
                add_qc_signal(
                    args.missing_instance_policy,
                    warnings,
                    errors,
                    (
                        f"{split}: high empty-label ratio {empty_ratio:.4f} "
                        f"(threshold {args.max_empty_label_ratio:.4f})."
                    ),
                )

            split_sources = source_stats.get(split, {})
            source_total = split_sources.get("source_total", Counter())
            source_multi = split_sources.get("source_multi", Counter())
            if isinstance(source_total, Counter) and isinstance(source_multi, Counter):
                for source, total in source_total.items():
                    if total < args.min_source_images_for_drop_check:
                        continue
                    source_multi_ratio = source_multi.get(source, 0) / total
                    drop_threshold = multi_ratio * args.source_multi_ratio_drop_factor
                    if multi_ratio > 0 and source_multi_ratio < drop_threshold:
                        add_qc_signal(
                            args.missing_instance_policy,
                            warnings,
                            errors,
                            (
                                f"{split}: source '{source}' multi-box ratio drop "
                                f"{source_multi_ratio:.4f} vs global {multi_ratio:.4f}."
                            ),
                        )

    report["errors"] = errors
    report["warnings"] = warnings
    report["qc_policy"] = {
        "missing_instance_policy": args.missing_instance_policy,
        "min_multi_box_ratio": args.min_multi_box_ratio,
        "max_empty_label_ratio": args.max_empty_label_ratio,
        "min_images_for_qc": args.min_images_for_qc,
        "min_source_images_for_drop_check": args.min_source_images_for_drop_check,
        "source_multi_ratio_drop_factor": args.source_multi_ratio_drop_factor,
    }
    report["qc_review_queue_size"] = len(review_queue_rows)
    report["splits"] = {
        s: {
            "image_count": split_image_counts.get(s, 0),
            "instance_count_by_class": {
                classes[cid]: int(cnt) for cid, cnt in sorted(split_instance_counts.get(s, Counter()).items())
            },
            "box_count_distribution": {
                str(box_count): int(count)
                for box_count, count in sorted(split_box_count_distribution.get(s, Counter()).items())
            },
        }
        for s in CANONICAL_SPLITS
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
        print(f"Wrote: {args.output_json.as_posix()}")
    else:
        print(text)

    if args.output_review_queue:
        write_review_queue(args.output_review_queue, review_queue_rows)
        print(f"Wrote: {args.output_review_queue.as_posix()}")

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
