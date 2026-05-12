#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MANUAL_SEPARATION = "manual_separation_required"
FULL_IMAGE_BOX = ("0.5", "0.5", "1.0", "1.0")
ALLOWED_DECISIONS = ("flat_monitor", "crt_monitor", "exclude", "uncertain")
DEFAULT_SOURCE_IDS = {
    "kb_monitor_scraped",
    "yolo_data_computer_monitor",
    "yolo_data_television",
}


@dataclass
class Source:
    id: str
    type: str
    root: str
    canonical_class: str
    action: str
    source_family: str
    source_trust: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build box-level manual review queue for mixed sources.")
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--sources-json", type=Path, default=Path("model_pipeline/config/dataset_v2_sources.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("dataset_staging/manual_review/box_review_queue.csv"))
    parser.add_argument(
        "--source-id",
        action="append",
        default=None,
        help="Limit to one source id. Can be repeated. Defaults to known CRT/flat/appliance manual sources.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_sources(config: dict[str, Any]) -> list[Source]:
    return [
        Source(
            id=str(raw["id"]),
            type=str(raw["type"]),
            root=str(raw["root"]),
            canonical_class=str(raw.get("canonical_class", "")),
            action=str(raw.get("action", "")),
            source_family=str(raw.get("source_family", "")),
            source_trust=str(raw.get("source_trust", "")),
        )
        for raw in config.get("sources", [])
    ]


def is_ignored_folder(name: str) -> bool:
    lowered = name.lower()
    return lowered == "__pycache__" or "dismiss" in lowered or "dismis" in lowered


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or not is_image(path):
            continue
        if any(is_ignored_folder(part) for part in path.relative_to(root).parts):
            continue
        out.append(path)
    return sorted(out)


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


def split_from_path(path: Path) -> str:
    lowered = [part.lower() for part in path.parts]
    if "train" in lowered:
        return "train"
    if "valid" in lowered or "val" in lowered:
        return "valid"
    if "test" in lowered:
        return "test"
    return "unsplit"


def rel_path(path: Path, workspace_root: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def resolve_path(workspace_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else workspace_root / path


def yolo_image_to_label(image_path: Path) -> Path:
    parts = list(image_path.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        idx = lowered.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def find_data_yaml(root: Path) -> Path | None:
    for name in ("dataset.yaml", "data.yaml"):
        path = root / name
        if path.exists():
            return path
    return None


def parse_yolo_names(path: Path | None) -> dict[int, str]:
    if path is None or not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        raw = line.strip()
        if raw.startswith("names:") and "[" in raw and "]" in raw:
            try:
                parsed = ast.literal_eval(raw.split(":", 1)[1].strip())
                if isinstance(parsed, list):
                    return {idx: str(name) for idx, name in enumerate(parsed)}
            except Exception:
                pass
    names: dict[int, str] = {}
    in_names = False
    for line in lines:
        if line.strip().startswith("names:"):
            in_names = True
            continue
        if in_names:
            if line and not line.startswith((" ", "\t")):
                break
            stripped = line.strip()
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            try:
                names[int(key.strip())] = value.strip().strip("'\"")
            except ValueError:
                continue
    return names


def valid_yolo_box(parts: list[str]) -> bool:
    if len(parts) != 5:
        return False
    try:
        int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return False
    return all(0.0 <= v <= 1.0 for v in vals)


def parse_label_rows(label_path: Path, source_names: dict[int, str]) -> tuple[list[dict[str, Any]], str]:
    if not label_path.exists():
        return [], "missing_label"
    lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return [], "empty_label"
    rows: list[dict[str, Any]] = []
    malformed = 0
    for raw in lines:
        parts = raw.split()
        if not valid_yolo_box(parts):
            malformed += 1
            continue
        source_class_id = int(float(parts[0]))
        rows.append(
            {
                "source_class_id": source_class_id,
                "source_class_name": source_names.get(source_class_id, str(source_class_id)),
                "x_center": parts[1],
                "y_center": parts[2],
                "width": parts[3],
                "height": parts[4],
                "box_source": "source_yolo",
            }
        )
    if rows and malformed:
        return rows, "warning_malformed_ignored"
    if rows:
        return rows, "ok"
    return [], "malformed_label"


def synthetic_box(source: Source, reason: str) -> dict[str, Any]:
    x, y, w, h = FULL_IMAGE_BOX
    return {
        "source_class_id": 0,
        "source_class_name": source.canonical_class or source.id,
        "x_center": x,
        "y_center": y,
        "width": w,
        "height": h,
        "box_source": reason,
    }


def default_decision(source: Source) -> str:
    if source.id == "kb_monitor_scraped":
        return "flat_monitor"
    return "uncertain"


def candidate_decisions(source: Source) -> str:
    return "flat_monitor|crt_monitor|exclude|uncertain"


def build_rows_for_source(workspace_root: Path, source: Source) -> list[dict[str, Any]]:
    root = workspace_root / source.root
    if not root.exists():
        return []
    source_names = parse_yolo_names(find_data_yaml(root)) if source.type == "direct_yolo" else {}
    rows: list[dict[str, Any]] = []
    for image_path in iter_images(root):
        digest = md5_file(image_path)
        width_px, height_px = image_dimensions(image_path)
        label_path = yolo_image_to_label(image_path) if source.type == "direct_yolo" else None
        if source.type == "direct_yolo" and label_path is not None:
            boxes, parse_status = parse_label_rows(label_path, source_names)
            if not boxes:
                boxes = [synthetic_box(source, "synthetic_missing_label")]
        else:
            parse_status = "synthetic_full_image"
            boxes = [synthetic_box(source, "synthetic_full_image")]

        for box_index, box in enumerate(boxes):
            review_box_id = f"{digest}__{source.id}__box{box_index:04d}"
            rows.append(
                {
                    "review_box_id": review_box_id,
                    "image_md5": digest,
                    "source_id": source.id,
                    "source_family": source.source_family,
                    "source_type": source.type,
                    "source_path": rel_path(image_path, workspace_root),
                    "label_path": rel_path(label_path, workspace_root) if label_path is not None else "",
                    "current_split": split_from_path(image_path),
                    "image_width": width_px or "",
                    "image_height": height_px or "",
                    "box_index": box_index,
                    "box_source": box["box_source"],
                    "label_parse_status": parse_status,
                    "source_class_id": box["source_class_id"],
                    "source_class_name": box["source_class_name"],
                    "x_center": box["x_center"],
                    "y_center": box["y_center"],
                    "width": box["width"],
                    "height": box["height"],
                    "candidate_decisions": candidate_decisions(source),
                    "decision_class": default_decision(source),
                    "review_status": "pending",
                    "notes": "",
                }
            )
    return rows


def assert_relative_paths(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        for field in ("source_path", "label_path"):
            value = str(row.get(field, ""))
            if value and Path(value).is_absolute():
                raise ValueError(f"{field} must be relative: {value}")


def run(
    workspace_root: Path,
    sources_json: Path,
    output_csv: Path,
    source_ids: set[str] | None = None,
) -> dict[str, Any]:
    workspace_root = workspace_root.resolve()
    config = load_json(resolve_path(workspace_root, sources_json))
    wanted = source_ids or DEFAULT_SOURCE_IDS
    rows: list[dict[str, Any]] = []
    for source in parse_sources(config):
        if source.id not in wanted:
            continue
        if source.action != MANUAL_SEPARATION:
            continue
        rows.extend(build_rows_for_source(workspace_root, source))
    rows = sorted(rows, key=lambda row: (row["source_id"], row["source_path"], int(row["box_index"])))
    assert_relative_paths(rows)
    output_path = resolve_path(workspace_root, output_csv)
    fields = [
        "review_box_id",
        "image_md5",
        "source_id",
        "source_family",
        "source_type",
        "source_path",
        "label_path",
        "current_split",
        "image_width",
        "image_height",
        "box_index",
        "box_source",
        "label_parse_status",
        "source_class_id",
        "source_class_name",
        "x_center",
        "y_center",
        "width",
        "height",
        "candidate_decisions",
        "decision_class",
        "review_status",
        "notes",
    ]
    write_csv(output_path, rows, fields)
    summary = {
        "queue_rows": len(rows),
        "image_count": len({row["image_md5"] for row in rows}),
        "source_count": len({row["source_id"] for row in rows}),
        "output_csv": output_csv.as_posix(),
    }
    output_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(
        workspace_root=args.workspace_root,
        sources_json=args.sources_json,
        output_csv=args.output_csv,
        source_ids=set(args.source_id) if args.source_id else None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
