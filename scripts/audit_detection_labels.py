#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from dataset_layout import detection_labels_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit YOLO detection label rows (box vs polygon vs malformed).")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to detection dataset root.")
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated split names to scan.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output report path. Default: <dataset-root>/label_audit_report.json",
    )
    parser.add_argument(
        "--missing-instance-policy",
        choices=["warn", "error", "off"],
        default="warn",
        help="How to report suspicious missing-instance patterns (warn, error, or off).",
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
        "--min-files-for-qc",
        type=int,
        default=100,
        help="Minimum split file count before ratio checks are applied.",
    )
    return parser.parse_args()


def _is_finite_number(token: str) -> bool:
    try:
        return math.isfinite(float(token))
    except Exception:
        return False


def classify_row(line: str) -> str:
    parts = line.split()
    if len(parts) == 0:
        return "empty"
    if not all(_is_finite_number(p) for p in parts):
        return "malformed"
    if len(parts) == 5:
        return "box"
    if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
        return "polygon"
    return "malformed"


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    output_json = args.output_json or (dataset_root / "label_audit_report.json")

    report: dict[str, object] = {
        "dataset_root": dataset_root.as_posix(),
        "splits": {},
        "summary": {
            "files_scanned": 0,
            "files_with_polygons": 0,
            "box_rows": 0,
            "polygon_rows": 0,
            "malformed_rows": 0,
            "empty_rows": 0,
            "box_count_distribution": {},
        },
        "files_with_polygons": [],
        "warnings": [],
        "errors": [],
        "qc_policy": {
            "missing_instance_policy": args.missing_instance_policy,
            "min_multi_box_ratio": args.min_multi_box_ratio,
            "max_empty_label_ratio": args.max_empty_label_ratio,
            "min_files_for_qc": args.min_files_for_qc,
        },
    }

    summary = report["summary"]
    files_with_polygons: list[str] = []
    per_split: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    errors: list[str] = []
    global_box_count_dist: defaultdict[int, int] = defaultdict(int)

    for split in splits:
        labels_dir = detection_labels_dir(dataset_root, split)
        split_stats: dict[str, object] = {
            "files_scanned": 0,
            "files_with_polygons": 0,
            "box_rows": 0,
            "polygon_rows": 0,
            "malformed_rows": 0,
            "empty_rows": 0,
            "box_count_distribution": {},
            "files": [],
        }
        if not labels_dir.exists():
            per_split[split] = split_stats
            continue

        for label_file in sorted(labels_dir.rglob("*.txt")):
            split_stats["files_scanned"] = int(split_stats["files_scanned"]) + 1
            summary["files_scanned"] = int(summary["files_scanned"]) + 1

            file_counts = defaultdict(int)
            text = label_file.read_text(encoding="utf-8")
            lines = text.splitlines()
            for raw in lines:
                row_type = classify_row(raw.strip())
                file_counts[row_type] += 1

            global_box_count_dist[file_counts["box"]] += 1

            split_stats["box_rows"] = int(split_stats["box_rows"]) + file_counts["box"]
            split_stats["polygon_rows"] = int(split_stats["polygon_rows"]) + file_counts["polygon"]
            split_stats["malformed_rows"] = int(split_stats["malformed_rows"]) + file_counts["malformed"]
            split_stats["empty_rows"] = int(split_stats["empty_rows"]) + file_counts["empty"]
            split_box_dist = split_stats["box_count_distribution"]
            split_box_dist[file_counts["box"]] = int(split_box_dist.get(file_counts["box"], 0)) + 1

            summary["box_rows"] = int(summary["box_rows"]) + file_counts["box"]
            summary["polygon_rows"] = int(summary["polygon_rows"]) + file_counts["polygon"]
            summary["malformed_rows"] = int(summary["malformed_rows"]) + file_counts["malformed"]
            summary["empty_rows"] = int(summary["empty_rows"]) + file_counts["empty"]

            has_polygon = file_counts["polygon"] > 0
            if has_polygon:
                split_stats["files_with_polygons"] = int(split_stats["files_with_polygons"]) + 1
                summary["files_with_polygons"] = int(summary["files_with_polygons"]) + 1
                rel = label_file.relative_to(dataset_root).as_posix()
                files_with_polygons.append(rel)

            split_stats["files"].append(
                {
                    "path": label_file.relative_to(dataset_root).as_posix(),
                    "box_rows": file_counts["box"],
                    "polygon_rows": file_counts["polygon"],
                    "malformed_rows": file_counts["malformed"],
                    "empty_rows": file_counts["empty"],
                    "has_polygon": has_polygon,
                }
            )

        per_split[split] = split_stats

        if args.missing_instance_policy != "off":
            files_scanned = int(split_stats["files_scanned"])
            split_box_dist = split_stats["box_count_distribution"]
            zero_count = int(split_box_dist.get(0, 0))
            one_count = int(split_box_dist.get(1, 0))
            multi_count = int(sum(v for k, v in split_box_dist.items() if int(k) > 1))
            labeled_count = one_count + multi_count
            multi_ratio = (multi_count / labeled_count) if labeled_count else 0.0
            empty_ratio = (zero_count / files_scanned) if files_scanned else 0.0

            if files_scanned >= args.min_files_for_qc and multi_ratio < args.min_multi_box_ratio:
                msg = (
                    f"{split}: low multi-box ratio {multi_ratio:.4f} "
                    f"(threshold {args.min_multi_box_ratio:.4f})."
                )
                if args.missing_instance_policy == "error":
                    errors.append(msg)
                else:
                    warnings.append(msg)

            if files_scanned >= args.min_files_for_qc and empty_ratio > args.max_empty_label_ratio:
                msg = (
                    f"{split}: high empty-label ratio {empty_ratio:.4f} "
                    f"(threshold {args.max_empty_label_ratio:.4f})."
                )
                if args.missing_instance_policy == "error":
                    errors.append(msg)
                else:
                    warnings.append(msg)

    report["splits"] = per_split
    report["files_with_polygons"] = files_with_polygons
    summary["box_count_distribution"] = {
        str(box_count): int(count) for box_count, count in sorted(global_box_count_dist.items())
    }
    report["warnings"] = warnings
    report["errors"] = errors

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {output_json.as_posix()}")
    print(f"Files scanned: {summary['files_scanned']}")
    print(f"Files with polygons: {summary['files_with_polygons']}")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
