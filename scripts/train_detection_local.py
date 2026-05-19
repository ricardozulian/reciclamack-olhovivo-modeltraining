#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local YOLO11n detection training with reproducible artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Path to local training YAML config.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume an interrupted Ultralytics run from last.pt.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_names_from_data_yaml(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        raw = line.strip()
        if raw.startswith("names:") and "[" in raw and "]" in raw:
            rhs = raw.split(":", 1)[1].strip()
            try:
                parsed = ast.literal_eval(rhs)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass

    in_names = False
    names_map: dict[int, str] = {}
    for line in lines:
        raw = line.strip()
        if raw == "names:":
            in_names = True
            continue
        if not in_names or ":" not in raw:
            continue
        left, right = raw.split(":", 1)
        left = left.strip()
        right = right.strip().strip("'\"")
        if left.isdigit():
            names_map[int(left)] = right
    return [names_map[i] for i in sorted(names_map)] if names_map else []


def write_resolved_data_yaml(source_yaml: Path, dataset_root: Path, run_dir: Path) -> Path:
    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Invalid data YAML: {source_yaml}")

    data["path"] = dataset_root.as_posix()
    data.setdefault("train", "train/images")
    data.setdefault("val", "valid/images")
    data.setdefault("test", "test/images")

    resolved = run_dir / "data_resolved.yaml"
    resolved.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return resolved


def resolve_device(preferred: str | None) -> str:
    if preferred and preferred != "auto":
        return preferred
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def safe_metric(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def metric_values(value: Any) -> list[Any]:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def per_class_detection_metrics(results: Any, class_names: list[str]) -> list[dict[str, float | int | str | None]]:
    box = getattr(results, "box", None)
    if box is None:
        return [{"class_id": idx, "class_name": name} for idx, name in enumerate(class_names)]

    class_indices = metric_values(getattr(box, "ap_class_index", range(len(class_names))))
    precision = metric_values(getattr(box, "p", None))
    recall = metric_values(getattr(box, "r", None))
    ap50 = metric_values(getattr(box, "ap50", None))
    maps = metric_values(getattr(box, "maps", None))

    rows: list[dict[str, float | int | str | None]] = []
    for metric_idx, class_id_raw in enumerate(class_indices):
        class_id = int(class_id_raw)
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                "precision": safe_metric(precision[metric_idx]) if metric_idx < len(precision) else None,
                "recall": safe_metric(recall[metric_idx]) if metric_idx < len(recall) else None,
                "mAP50": safe_metric(ap50[metric_idx]) if metric_idx < len(ap50) else None,
                "mAP50_95": safe_metric(maps[class_id]) if class_id < len(maps) else None,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    dataset_root = Path(cfg["dataset_root"]).resolve()
    data_yaml = Path(cfg["data_yaml"]).resolve()
    output_root = Path(cfg.get("output_root", "model_pipeline/artifacts/detection_training")).resolve()
    resume_from = args.resume_from.resolve() if args.resume_from else None
    model_name = str(resume_from if resume_from else cfg.get("model", "yolo11n.pt"))
    project_name = str(cfg.get("project_name", "reciclamack_local"))
    run_name = str(cfg.get("run_name", datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")))
    training = cfg.get("training", {})
    export = cfg.get("export", {})

    epochs = int(args.epochs if args.epochs is not None else training.get("epochs", 20))
    imgsz = int(args.imgsz if args.imgsz is not None else training.get("imgsz", 640))
    batch = int(args.batch if args.batch is not None else training.get("batch", 8))
    patience = int(training.get("patience", 20))
    workers = int(training.get("workers", 4))
    cache = bool(training.get("cache", True))
    device = resolve_device(args.device if args.device is not None else training.get("device", "auto"))

    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    train_data_yaml = write_resolved_data_yaml(data_yaml, dataset_root, run_dir)

    from ultralytics import YOLO

    model = YOLO(model_name)
    if resume_from:
        train_results = model.train(resume=True)
    else:
        train_results = model.train(
            data=str(train_data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=str(output_root),
            name=run_name,
            patience=patience,
            workers=workers,
            cache=cache,
        )

    best_pt = Path(train_results.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_pt))
    test_results = best_model.val(
        data=str(train_data_yaml),
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(output_root),
        name=f"{run_name}_test_eval",
    )

    onnx_result = best_model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=bool(export.get("dynamic", False)),
        opset=int(export.get("opset", 17)),
    )
    onnx_path = Path(str(onnx_result)).resolve()

    class_names = parse_names_from_data_yaml(data_yaml)
    class_metrics = per_class_detection_metrics(test_results, class_names)

    metrics_summary = {
        "precision_B": safe_metric(getattr(test_results.box, "mp", None)),
        "recall_B": safe_metric(getattr(test_results.box, "mr", None)),
        "mAP50_B": safe_metric(getattr(test_results.box, "map50", None)),
        "mAP50_95_B": safe_metric(getattr(test_results.box, "map", None)),
    }

    resolved_cfg = {
        "dataset_root": dataset_root.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "resolved_data_yaml": train_data_yaml.as_posix(),
        "model": model_name,
        "resume_from": resume_from.as_posix() if resume_from else None,
        "project_name": project_name,
        "run_name": run_name,
        "output_root": output_root.as_posix(),
        "device": device,
        "training": {
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "patience": patience,
            "workers": workers,
            "cache": cache,
        },
        "export": {
            "format": "onnx",
            "dynamic": bool(export.get("dynamic", False)),
            "opset": int(export.get("opset", 17)),
        },
    }

    export_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_root": dataset_root.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "resolved_data_yaml": train_data_yaml.as_posix(),
        "data_yaml_sha256": sha256_file(data_yaml),
        "best_pt": best_pt.resolve().as_posix(),
        "best_pt_sha256": sha256_file(best_pt),
        "onnx_path": onnx_path.as_posix(),
        "onnx_sha256": sha256_file(onnx_path),
    }

    (run_dir / "run_config_resolved.json").write_text(json.dumps(resolved_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics_summary.json").write_text(json.dumps(metrics_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "class_metrics.json").write_text(json.dumps(class_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "export_manifest.json").write_text(json.dumps(export_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Run dir: {run_dir.as_posix()}")
    print(f"Best PT: {best_pt.as_posix()}")
    print(f"ONNX: {onnx_path.as_posix()}")


if __name__ == "__main__":
    main()
