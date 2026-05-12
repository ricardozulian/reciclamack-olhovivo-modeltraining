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


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    dataset_root = Path(cfg["dataset_root"]).resolve()
    data_yaml = Path(cfg["data_yaml"]).resolve()
    output_root = Path(cfg.get("output_root", "model_pipeline/artifacts/detection_training")).resolve()
    model_name = str(cfg.get("model", "yolo11n.pt"))
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

    from ultralytics import YOLO

    model = YOLO(model_name)
    train_results = model.train(
        data=str(data_yaml),
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
        data=str(data_yaml),
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
    class_maps: list[float] = []
    try:
        class_maps = [float(x) for x in list(test_results.box.maps)]
    except Exception:
        class_maps = []
    class_metrics = [
        {"class_name": name, "mAP50_95": class_maps[idx] if idx < len(class_maps) else None}
        for idx, name in enumerate(class_names)
    ]

    metrics_summary = {
        "precision_B": safe_metric(getattr(test_results.box, "mp", None)),
        "recall_B": safe_metric(getattr(test_results.box, "mr", None)),
        "mAP50_B": safe_metric(getattr(test_results.box, "map50", None)),
        "mAP50_95_B": safe_metric(getattr(test_results.box, "map", None)),
    }

    resolved_cfg = {
        "dataset_root": dataset_root.as_posix(),
        "data_yaml": data_yaml.as_posix(),
        "model": model_name,
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

