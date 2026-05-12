#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare compact manual review batch package.")
    parser.add_argument("--input-csv", type=Path, required=True, help="Path to batch_XX.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for copied images and reduced CSV.",
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root used to resolve source_relpath.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized: dict[str, str] = {}
            for k, v in row.items():
                key = (k or "").lstrip("\ufeff")
                normalized[key] = v or ""
            rows.append(normalized)
        return rows


def safe_stem(text: str) -> str:
    out = []
    for ch in text:
        out.append(ch if ch.isalnum() or ch in {"_", "-"} else "_")
    return "".join(out)


def main() -> None:
    args = parse_args()
    rows = read_csv(args.input_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    reduced_rows: list[dict[str, Any]] = []
    for row in rows:
        rank = row.get("priority_rank", "")
        uid = row.get("unique_id", "")
        src = row.get("source_dataset", "")
        original = row.get("original_filename", "")
        source_relpath = row.get("source_relpath", "")

        source_path = (args.workspace_root / source_relpath).resolve()
        ext = Path(original).suffix.lower() or source_path.suffix.lower() or ".jpg"
        image_name = f"{rank.zfill(4)}__{safe_stem(src)}__{uid}{ext}"
        target_path = images_dir / image_name

        if source_path.exists():
            shutil.copy2(source_path, target_path)
            copy_status = "ok"
        else:
            copy_status = "missing_source"

        reduced_rows.append(
            {
                "priority_rank": rank,
                "batch_rank": row.get("batch_rank", ""),
                "decision": row.get("mapped_label", ""),
                "notes": "",
                "reason": row.get("reason", ""),
                "current_label": row.get("mapped_label", ""),
                "source_dataset": src,
                "original_filename": original,
                "source_relpath": source_relpath,
                "copied_image": f"images/{image_name}",
                "copy_status": copy_status,
                "unique_id": uid,
            }
        )

    reduced_csv = args.output_dir / "review_reduced.csv"
    fields = [
        "priority_rank",
        "batch_rank",
        "decision",
        "notes",
        "reason",
        "current_label",
        "source_dataset",
        "original_filename",
        "source_relpath",
        "copied_image",
        "copy_status",
        "unique_id",
    ]
    with reduced_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(reduced_rows)

    print(f"Rows: {len(reduced_rows)}")
    print(f"Output CSV: {reduced_csv.as_posix()}")
    print(f"Images dir: {images_dir.as_posix()}")


if __name__ == "__main__":
    main()
