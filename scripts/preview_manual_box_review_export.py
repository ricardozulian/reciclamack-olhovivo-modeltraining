#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACCEPTED_CLASSES = ("flat_monitor", "crt_monitor")
GLOBAL_CLASS_IDS = {
    "flat_monitor": 5,
    "crt_monitor": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview YOLO labels from accepted manual box decisions.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--queue-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/box_review_queue.csv"),
    )
    parser.add_argument(
        "--decisions-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/box_review_decisions.csv"),
        help="Optional compact decisions CSV from the notebook. If present, it overrides queue decisions.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
    )
    parser.add_argument(
        "--dedupe-analysis-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/box_dedupe_analysis.csv"),
    )
    parser.add_argument(
        "--source-merge-analysis-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/source_merge_analysis.csv"),
    )
    parser.add_argument(
        "--source-duplicate-candidates-csv",
        type=Path,
        default=Path("dataset_staging/manual_review/source_duplicate_candidates.csv"),
    )
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    return parser.parse_args()


def resolve_path(workspace_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def valid_box(row: dict[str, str]) -> bool:
    try:
        vals = [float(row[field]) for field in ("x_center", "y_center", "width", "height")]
    except ValueError:
        return False
    return all(0.0 <= val <= 1.0 for val in vals)


def box_xyxy(row: dict[str, str]) -> tuple[float, float, float, float]:
    x_center = float(row["x_center"])
    y_center = float(row["y_center"])
    width = float(row["width"])
    height = float(row["height"])
    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )


def box_area_xyxy(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def overlap_metrics(a: dict[str, str], b: dict[str, str]) -> dict[str, float]:
    box_a = box_xyxy(a)
    box_b = box_xyxy(b)
    area_a = box_area_xyxy(box_a)
    area_b = box_area_xyxy(box_b)
    inter = intersection_area(box_a, box_b)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    candidate_area = area_b
    return {
        "iou": inter / union if union > 0 else 0.0,
        "intersection_over_smaller_box": inter / smaller if smaller > 0 else 0.0,
        "intersection_over_candidate_box": inter / candidate_area if candidate_area > 0 else 0.0,
        "a_covered_by_b": inter / area_a if area_a > 0 else 0.0,
        "b_covered_by_a": inter / area_b if area_b > 0 else 0.0,
        "area_ratio": max(area_a, area_b) / smaller if smaller > 0 else 0.0,
        "area_a": area_a,
        "area_b": area_b,
    }


def box_priority(row: dict[str, str]) -> tuple[int, int, float, str]:
    box_source = row.get("box_source", "")
    synthetic_penalty = 1 if box_source.startswith("synthetic") else 0
    missing_penalty = 1 if box_source == "synthetic_missing_label" else 0
    return (
        synthetic_penalty,
        missing_penalty,
        float(row["width"]) * float(row["height"]),
        row.get("review_box_id", ""),
    )


def pick_keeper(a: dict[str, str], b: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    return tuple(sorted([a, b], key=box_priority))  # type: ignore[return-value]


SOURCE_PRIORITY = {
    "kb_monitor_scraped": 0,
    "yolo_data_computer_monitor": 1,
    "yolo_data_television": 2,
}


def source_priority(row: dict[str, str]) -> tuple[int, str, str]:
    source_id = row.get("source_id", "")
    return (SOURCE_PRIORITY.get(source_id, 999), source_id, row.get("source_path", ""))


def choose_kept_source(rows: list[dict[str, str]]) -> tuple[str, str]:
    selected = sorted(rows, key=source_priority)[0]
    return selected.get("source_id", ""), selected.get("source_path", "")


def rows_overlap(a: dict[str, str], b: dict[str, str], iou_threshold: float, coverage_threshold: float) -> bool:
    metrics = overlap_metrics(a, b)
    return (
        metrics["iou"] >= iou_threshold
        or metrics["intersection_over_smaller_box"] >= coverage_threshold
    )


def merge_sources_by_md5(
    accepted: list[dict[str, str]],
    iou_threshold: float = 0.85,
    coverage_threshold: float = 0.90,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in accepted:
        grouped[row["image_md5"]].append(row)

    merged_rows: list[dict[str, str]] = []
    analysis_rows: list[dict[str, object]] = []
    for image_md5, group in sorted(grouped.items()):
        source_ids = sorted({row.get("source_id", "") for row in group})
        classes = sorted({row.get("decision_class", "") for row in group})
        kept_source_id, kept_source_path = choose_kept_source(group)
        review_box_ids = [row.get("review_box_id", "") for row in group]
        dropped_ids: list[str] = []
        status = "single_source" if len(source_ids) <= 1 else "merged_multiclass_labels"
        reason = ""

        if len(source_ids) > 1 and {"flat_monitor", "crt_monitor"}.issubset(set(classes)):
            status = "resolved_flat_vs_crt_to_crt"
            reason = "flat_monitor_and_crt_monitor_conflict_resolved_to_crt_monitor"
            dropped_ids = [
                row.get("review_box_id", "")
                for row in group
                if row.get("decision_class") == "flat_monitor"
            ]
        elif len(source_ids) > 1 and len(classes) == 1:
            status = "deduped_same_class_labels"
            reason = "same_class_cross_source_md5_labels_passed_to_box_dedupe"
        elif len(source_ids) > 1:
            reason = "approved_multiclass_cross_source_md5_label_merge"

        for row in group:
            if row.get("review_box_id", "") in dropped_ids:
                continue
            out = dict(row)
            out["_kept_source_id"] = kept_source_id
            out["_kept_source_path"] = kept_source_path
            merged_rows.append(out)

        analysis_rows.append(
            {
                "image_md5": image_md5,
                "kept_source_id": kept_source_id,
                "all_source_ids": "|".join(source_ids),
                "merged_review_box_ids": "|".join(row_id for row_id in review_box_ids if row_id not in dropped_ids),
                "dropped_review_box_ids": "|".join(dropped_ids),
                "accepted_classes": "|".join(classes),
                "status": status,
                "reason": reason,
            }
        )

    return merged_rows, analysis_rows


def dedupe_accepted_boxes(
    accepted: list[dict[str, str]],
    iou_threshold: float = 0.85,
    coverage_threshold: float = 0.90,
) -> tuple[list[dict[str, str]], list[dict[str, object]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in accepted:
        groups[(row["image_md5"], row["decision_class"])].append(row)

    kept_by_id: dict[str, dict[str, str]] = {row["review_box_id"]: row for row in accepted}
    analysis_rows: list[dict[str, object]] = []

    for (image_md5, decision_class), group in sorted(groups.items()):
        active = {row["review_box_id"]: row for row in group}
        changed = True
        while changed:
            changed = False
            rows = list(active.values())
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    a = rows[i]
                    b = rows[j]
                    metrics = overlap_metrics(a, b)
                    reason = ""
                    if metrics["iou"] >= iou_threshold:
                        reason = "iou_threshold"
                    elif metrics["intersection_over_smaller_box"] >= coverage_threshold:
                        reason = "coverage_threshold"
                    if not reason:
                        continue

                    kept, dropped = pick_keeper(a, b)
                    dropped_id = dropped["review_box_id"]
                    if dropped_id not in active:
                        continue
                    active.pop(dropped_id)
                    kept_by_id.pop(dropped_id, None)
                    analysis_rows.append(
                        {
                            "image_md5": image_md5,
                            "decision_class": decision_class,
                            "kept_review_box_id": kept["review_box_id"],
                            "dropped_review_box_id": dropped_id,
                            "iou": f"{metrics['iou']:.6f}",
                            "intersection_over_smaller": f"{metrics['intersection_over_smaller_box']:.6f}",
                            "intersection_over_candidate": f"{metrics['intersection_over_candidate_box']:.6f}",
                            "a_covered_by_b": f"{metrics['a_covered_by_b']:.6f}",
                            "b_covered_by_a": f"{metrics['b_covered_by_a']:.6f}",
                            "area_ratio": f"{metrics['area_ratio']:.6f}",
                            "kept_box_source": kept.get("box_source", ""),
                            "dropped_box_source": dropped.get("box_source", ""),
                            "reason": reason,
                        }
                    )
                    changed = True
                    break
                if changed:
                    break

    return list(kept_by_id.values()), analysis_rows


def merge_decisions(queue_rows: list[dict[str, str]], decision_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_id = {row.get("review_box_id", ""): row for row in decision_rows if row.get("review_box_id")}
    merged: list[dict[str, str]] = []
    for row in queue_rows:
        out = dict(row)
        decision = by_id.get(row.get("review_box_id", ""))
        if decision:
            out["decision_class"] = decision.get("decision_class", out.get("decision_class", ""))
            out["review_status"] = decision.get("review_status", out.get("review_status", ""))
            out["notes"] = decision.get("notes", out.get("notes", ""))
        merged.append(out)
    return merged


def build_source_duplicate_candidates(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("image_md5", "")].append(row)

    out: list[dict[str, object]] = []
    for image_md5, group in sorted(grouped.items()):
        if not image_md5:
            continue
        source_ids = sorted({row.get("source_id", "") for row in group if row.get("source_id")})
        if len(source_ids) <= 1:
            continue
        accepted = [
            row
            for row in group
            if row.get("decision_class") in ACCEPTED_CLASSES
            and row.get("review_status") in {"accepted", "reviewed", "done"}
            and valid_box(row)
        ]
        accepted_sources = sorted({row.get("source_id", "") for row in accepted if row.get("source_id")})
        if len(accepted_sources) > 1:
            export_relevance = "accepted_cross_source"
        elif len(accepted_sources) == 1:
            export_relevance = "one_source_accepted"
        else:
            export_relevance = "not_export_ready"

        out.append(
            {
                "image_md5": image_md5,
                "all_source_ids": "|".join(source_ids),
                "review_box_ids": "|".join(row.get("review_box_id", "") for row in group),
                "decision_classes": "|".join(row.get("decision_class", "") for row in group),
                "review_statuses": "|".join(row.get("review_status", "") for row in group),
                "accepted_source_ids": "|".join(accepted_sources),
                "source_paths": "|".join(row.get("source_path", "") for row in group),
                "export_relevance": export_relevance,
            }
        )
    return out


def run(
    workspace_root: Path,
    queue_csv: Path,
    output_csv: Path,
    decisions_csv: Path | None = None,
    dedupe_analysis_csv: Path = Path("dataset_staging/manual_review/box_dedupe_analysis.csv"),
    source_merge_analysis_csv: Path = Path("dataset_staging/manual_review/source_merge_analysis.csv"),
    source_duplicate_candidates_csv: Path = Path("dataset_staging/manual_review/source_duplicate_candidates.csv"),
    iou_threshold: float = 0.85,
    coverage_threshold: float = 0.90,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    rows = read_csv(resolve_path(workspace_root, queue_csv))
    if decisions_csv is not None:
        decisions_path = resolve_path(workspace_root, decisions_csv)
        if decisions_path.exists():
            rows = merge_decisions(rows, read_csv(decisions_path))
    source_duplicate_rows = build_source_duplicate_candidates(rows)
    accepted = [
        row
        for row in rows
        if row.get("decision_class") in ACCEPTED_CLASSES
        and row.get("review_status") in {"accepted", "reviewed", "done"}
        and valid_box(row)
    ]
    source_merged, source_analysis_rows = merge_sources_by_md5(
        accepted,
        iou_threshold=iou_threshold,
        coverage_threshold=coverage_threshold,
    )
    deduped_accepted, analysis_rows = dedupe_accepted_boxes(
        source_merged,
        iou_threshold=iou_threshold,
        coverage_threshold=coverage_threshold,
    )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in deduped_accepted:
        grouped[row["image_md5"]].append(row)

    preview_rows: list[dict[str, Any]] = []
    for image_md5, group in sorted(grouped.items()):
        source_path = sorted({row.get("_kept_source_path") or row["source_path"] for row in group})[0]
        yolo_lines: list[str] = []
        for row in sorted(group, key=lambda item: int(item.get("box_index") or 0)):
            cls = row["decision_class"]
            class_id = GLOBAL_CLASS_IDS[cls]
            yolo_lines.append(
                " ".join(
                    [
                        str(class_id),
                        row["x_center"],
                        row["y_center"],
                        row["width"],
                        row["height"],
                    ]
                )
            )
        preview_rows.append(
            {
                "image_md5": image_md5,
                "source_path": source_path,
                "accepted_box_count": len(yolo_lines),
                "preview_label_lines": "\\n".join(yolo_lines),
                "future_label_name": f"{image_md5}.txt",
            }
        )

    output_path = resolve_path(workspace_root, output_csv)
    analysis_path = resolve_path(workspace_root, dedupe_analysis_csv)
    source_analysis_path = resolve_path(workspace_root, source_merge_analysis_csv)
    source_duplicate_path = resolve_path(workspace_root, source_duplicate_candidates_csv)
    write_csv(
        output_path,
        preview_rows,
        ["image_md5", "source_path", "accepted_box_count", "preview_label_lines", "future_label_name"],
    )
    write_csv(
        analysis_path,
        analysis_rows,
        [
            "image_md5",
            "decision_class",
            "kept_review_box_id",
            "dropped_review_box_id",
            "iou",
            "intersection_over_smaller",
            "intersection_over_candidate",
            "a_covered_by_b",
            "b_covered_by_a",
            "area_ratio",
            "kept_box_source",
            "dropped_box_source",
            "reason",
        ],
    )
    write_csv(
        source_analysis_path,
        source_analysis_rows,
        [
            "image_md5",
            "kept_source_id",
            "all_source_ids",
            "merged_review_box_ids",
            "dropped_review_box_ids",
            "accepted_classes",
            "status",
            "reason",
        ],
    )
    write_csv(
        source_duplicate_path,
        source_duplicate_rows,
        [
            "image_md5",
            "all_source_ids",
            "review_box_ids",
            "decision_classes",
            "review_statuses",
            "accepted_source_ids",
            "source_paths",
            "export_relevance",
        ],
    )
    analysis_summary = {
        "input_accepted_box_rows": len(source_merged),
        "deduped_accepted_box_rows": len(deduped_accepted),
        "dropped_duplicate_box_rows": len(analysis_rows),
        "iou_threshold": iou_threshold,
        "coverage_threshold": coverage_threshold,
        "output_csv": dedupe_analysis_csv.as_posix(),
    }
    analysis_path.with_suffix(".summary.json").write_text(
        json.dumps(analysis_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_status_counts = dict(sorted(Counter(row["status"] for row in source_analysis_rows).items()))
    source_summary = {
        "input_accepted_box_rows": len(accepted),
        "source_merged_box_rows": len(source_merged),
        "dropped_source_conflict_box_rows": len(accepted) - len(source_merged),
        "image_md5_count": len(source_analysis_rows),
        "status_counts": source_status_counts,
        "output_csv": source_merge_analysis_csv.as_posix(),
    }
    source_analysis_path.with_suffix(".summary.json").write_text(
        json.dumps(source_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_duplicate_counts = dict(sorted(Counter(row["export_relevance"] for row in source_duplicate_rows).items()))
    source_duplicate_summary = {
        "cross_source_md5_count": len(source_duplicate_rows),
        "export_relevance_counts": source_duplicate_counts,
        "output_csv": source_duplicate_candidates_csv.as_posix(),
    }
    source_duplicate_path.with_suffix(".summary.json").write_text(
        json.dumps(source_duplicate_summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "accepted_box_rows": len(deduped_accepted),
        "input_accepted_box_rows": len(accepted),
        "source_merged_box_rows": len(source_merged),
        "dropped_source_conflict_box_rows": len(accepted) - len(source_merged),
        "dropped_duplicate_box_rows": len(analysis_rows),
        "cross_source_duplicate_candidate_md5s": len(source_duplicate_rows),
        "iou_threshold": iou_threshold,
        "coverage_threshold": coverage_threshold,
        "accepted_image_count": len(preview_rows),
        "output_csv": output_csv.as_posix(),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        workspace_root=args.workspace_root,
        queue_csv=args.queue_csv,
        output_csv=args.output_csv,
        decisions_csv=args.decisions_csv,
        dedupe_analysis_csv=args.dedupe_analysis_csv,
        source_merge_analysis_csv=args.source_merge_analysis_csv,
        source_duplicate_candidates_csv=args.source_duplicate_candidates_csv,
        iou_threshold=args.iou_threshold,
        coverage_threshold=args.coverage_threshold,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
