#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml

from train_detection_local import parse_names_from_data_yaml, per_class_detection_metrics, safe_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detection model and write per-class metrics.")
    parser.add_argument("--model", type=Path, required=True, help="Path to best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Path to data.yaml.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Dataset root used to resolve data.yaml path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve_device(preferred: str) -> str:
    if preferred != "auto":
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_resolved_yaml(source: Path, dataset_root: Path, output_dir: Path) -> Path:
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["path"] = dataset_root.resolve().as_posix()
    resolved = output_dir / "eval_data_resolved.yaml"
    resolved.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return resolved


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["class_id", "class_name", "precision", "recall", "mAP50", "mAP50_95"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_data = write_resolved_yaml(args.data, args.dataset_root, args.output_dir)
    class_names = parse_names_from_data_yaml(args.data)
    device = resolve_device(args.device)

    from ultralytics import YOLO

    model = YOLO(str(args.model))
    results = model.val(
        data=str(resolved_data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.output_dir),
        name=f"{args.split}_eval",
        exist_ok=True,
    )
    rows = per_class_detection_metrics(results, class_names)
    summary = {
        "precision_B": safe_metric(getattr(results.box, "mp", None)),
        "recall_B": safe_metric(getattr(results.box, "mr", None)),
        "mAP50_B": safe_metric(getattr(results.box, "map50", None)),
        "mAP50_95_B": safe_metric(getattr(results.box, "map", None)),
    }

    write_csv(args.output_dir / f"class_metrics_{args.split}.csv", rows)
    (args.output_dir / f"class_metrics_{args.split}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / f"metrics_summary_{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
