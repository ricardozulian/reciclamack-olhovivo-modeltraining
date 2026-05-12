#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from dataset_layout import detection_labels_dir

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Roboflow polygon labels into YOLO box labels.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Path to detection dataset root.")
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=None,
        help="Optional audit report path from audit_detection_labels.py (recommended).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,valid,test",
        help="Comma-separated split names to process when audit-json is not provided.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output conversion report path. Default: <dataset-root>/polygon_conversion_report.json",
    )
    return parser.parse_args()


def _safe_float(value: str) -> float | None:
    try:
        val = float(value)
        if not math.isfinite(val):
            return None
        return val
    except Exception:
        return None


def _safe_int(value: str) -> int | None:
    try:
        as_float = float(value)
        if not math.isfinite(as_float):
            return None
        rounded = int(as_float)
        if abs(as_float - rounded) > 1e-9:
            return None
        return rounded
    except Exception:
        return None


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _format_box(class_id: int, xc: float, yc: float, w: float, h: float) -> str:
    return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def _is_polygon_row(parts: list[str]) -> bool:
    return len(parts) >= 7 and (len(parts) - 1) % 2 == 0


def _is_box_row(parts: list[str]) -> bool:
    return len(parts) == 5


def _row_to_bbox(parts: list[str]) -> tuple[bool, str | None]:
    class_id = _safe_int(parts[0])
    if class_id is None:
        return False, None

    coords_raw = [_safe_float(x) for x in parts[1:]]
    if any(v is None for v in coords_raw):
        return False, None
    coords = [float(v) for v in coords_raw if v is not None]
    if len(coords) < 6 or len(coords) % 2 != 0:
        return False, None

    xs = coords[0::2]
    ys = coords[1::2]
    xmin = _clamp01(min(xs))
    xmax = _clamp01(max(xs))
    ymin = _clamp01(min(ys))
    ymax = _clamp01(max(ys))

    w = xmax - xmin
    h = ymax - ymin
    if w <= 0.0 or h <= 0.0:
        return False, None

    xc = (xmin + xmax) / 2.0
    yc = (ymin + ymax) / 2.0
    return True, _format_box(class_id, xc, yc, w, h)


def discover_files_from_audit(dataset_root: Path, audit_json: Path) -> list[Path]:
    payload = json.loads(audit_json.read_text(encoding="utf-8"))
    files = payload.get("files_with_polygons", [])
    out: list[Path] = []
    for rel in files:
        p = dataset_root / rel
        if p.exists():
            out.append(p)
    return sorted(out)


def discover_files_by_splits(dataset_root: Path, splits: list[str]) -> list[Path]:
    out: list[Path] = []
    for split in splits:
        labels_dir = detection_labels_dir(dataset_root, split)
        if labels_dir.exists():
            out.extend(sorted(labels_dir.rglob("*.txt")))
    return out


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root
    output_json = args.output_json or (dataset_root / "polygon_conversion_report.json")

    if args.audit_json and args.audit_json.exists():
        targets = discover_files_from_audit(dataset_root, args.audit_json)
        source_mode = "audit_json"
    else:
        splits = [s.strip() for s in args.splits.split(",") if s.strip()]
        targets = discover_files_by_splits(dataset_root, splits)
        source_mode = "split_scan"

    report: dict[str, object] = {
        "dataset_root": dataset_root.as_posix(),
        "source_mode": source_mode,
        "files_targeted": len(targets),
        "files_rewritten": 0,
        "files_with_polygons": 0,
        "box_rows_unchanged": 0,
        "polygon_rows_converted": 0,
        "malformed_rows_dropped": 0,
        "invalid_polygon_rows_dropped": 0,
        "splits": {},
        "files": [],
    }

    split_stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for label_file in targets:
        rel = label_file.relative_to(dataset_root).as_posix()
        parts = rel.split("/")
        split = parts[0] if len(parts) >= 3 else "unknown"

        original = label_file.read_text(encoding="utf-8")
        lines = [ln.strip() for ln in original.splitlines() if ln.strip()]
        rewritten_lines: list[str] = []

        file_box_unchanged = 0
        file_polygon_converted = 0
        file_malformed_dropped = 0
        file_invalid_polygon_dropped = 0

        for ln in lines:
            tokens = ln.split()

            # Preserve valid YOLO detection rows unchanged.
            if _is_box_row(tokens) and all(_safe_float(tok) is not None for tok in tokens):
                rewritten_lines.append(ln)
                file_box_unchanged += 1
                continue

            # Convert polygon rows to tight box rows (one output per segment row).
            if _is_polygon_row(tokens) and all(_safe_float(tok) is not None for tok in tokens):
                ok, boxed = _row_to_bbox(tokens)
                if ok and boxed is not None:
                    rewritten_lines.append(boxed)
                    file_polygon_converted += 1
                else:
                    file_invalid_polygon_dropped += 1
                continue

            file_malformed_dropped += 1

        had_polygon = file_polygon_converted > 0 or file_invalid_polygon_dropped > 0
        if had_polygon:
            report["files_with_polygons"] = int(report["files_with_polygons"]) + 1

        new_text = ("\n".join(rewritten_lines) + "\n") if rewritten_lines else ""
        changed = new_text != original
        if changed:
            label_file.write_text(new_text, encoding="utf-8")
            report["files_rewritten"] = int(report["files_rewritten"]) + 1

        report["box_rows_unchanged"] = int(report["box_rows_unchanged"]) + file_box_unchanged
        report["polygon_rows_converted"] = int(report["polygon_rows_converted"]) + file_polygon_converted
        report["malformed_rows_dropped"] = int(report["malformed_rows_dropped"]) + file_malformed_dropped
        report["invalid_polygon_rows_dropped"] = int(report["invalid_polygon_rows_dropped"]) + file_invalid_polygon_dropped

        split_stats[split]["files_targeted"] += 1
        split_stats[split]["files_rewritten"] += 1 if changed else 0
        split_stats[split]["box_rows_unchanged"] += file_box_unchanged
        split_stats[split]["polygon_rows_converted"] += file_polygon_converted
        split_stats[split]["malformed_rows_dropped"] += file_malformed_dropped
        split_stats[split]["invalid_polygon_rows_dropped"] += file_invalid_polygon_dropped

        report["files"].append(
            {
                "path": rel,
                "rewritten": changed,
                "box_rows_unchanged": file_box_unchanged,
                "polygon_rows_converted": file_polygon_converted,
                "malformed_rows_dropped": file_malformed_dropped,
                "invalid_polygon_rows_dropped": file_invalid_polygon_dropped,
            }
        )

    report["splits"] = {k: dict(v) for k, v in sorted(split_stats.items())}

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {output_json.as_posix()}")
    print(f"Files targeted: {report['files_targeted']}")
    print(f"Polygon rows converted: {report['polygon_rows_converted']}")


if __name__ == "__main__":
    main()
