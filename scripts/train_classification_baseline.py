#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dataset_layout import unified_images_dir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ReciclaMack classification baseline.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("prod_datasets/unified/v1"),
        help="Unified dataset root with train/valid/test split-first image folders.",
    )
    parser.add_argument(
        "--manifests-root",
        type=Path,
        default=Path("prod_datasets/unified/v1/manifests"),
        help="Path containing train.csv/valid.csv/test.csv for dataset hash/versioning.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("model_pipeline/artifacts/classification_baseline"),
        help="Artifact output root.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=48,
        help="Square image resize dimension used before flattening.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def list_classes(split_root: Path) -> list[str]:
    return sorted([p.name for p in split_root.iterdir() if p.is_dir()])


def load_split(split_root: Path, classes: list[str], image_size: int) -> tuple[np.ndarray, np.ndarray]:
    x: list[np.ndarray] = []
    y: list[int] = []
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for cls in classes:
        cls_dir = split_root / cls
        if not cls_dir.exists():
            continue
        for file in cls_dir.rglob("*"):
            if not file.is_file() or not is_image(file):
                continue
            try:
                with Image.open(file) as img:
                    img = img.convert("RGB").resize((image_size, image_size))
                    arr = np.asarray(img, dtype=np.float32) / 255.0
            except Exception:
                continue
            x.append(arr.reshape(-1))
            y.append(class_to_idx[cls])

    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_hash(manifests_root: Path) -> str:
    targets = [manifests_root / "train.csv", manifests_root / "valid.csv", manifests_root / "test.csv"]
    h = hashlib.sha256()
    for p in targets:
        if not p.exists():
            continue
        h.update(p.name.encode("utf-8"))
        h.update(file_sha1(p).encode("utf-8"))
    return h.hexdigest()


def top_confusions(cm: np.ndarray, classes: list[str], top_k: int = 12) -> list[dict[str, Any]]:
    pairs: list[tuple[int, int, int]] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            val = int(cm[i, j])
            if val > 0:
                pairs.append((i, j, val))
    pairs.sort(key=lambda x: x[2], reverse=True)
    out = []
    for i, j, count in pairs[:top_k]:
        out.append({"true_label": classes[i], "pred_label": classes[j], "count": count})
    return out


def write_confusion_csv(path: Path, cm: np.ndarray, classes: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["true\\pred"] + classes)
        for i, row in enumerate(cm):
            writer.writerow([classes[i]] + [int(v) for v in row.tolist()])


def train_and_evaluate(args: argparse.Namespace) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_root = unified_images_dir(args.dataset_root, "train")
    val_root = unified_images_dir(args.dataset_root, "valid")
    test_root = unified_images_dir(args.dataset_root, "test")
    classes = list_classes(train_root)

    x_train, y_train = load_split(train_root, classes, args.image_size)
    x_val, y_val = load_split(val_root, classes, args.image_size)
    x_test, y_test = load_split(test_root, classes, args.image_size)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    max_iter=2000,
                    random_state=args.seed,
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(classes))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(classes))),
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    top_conf = top_confusions(cm, classes)
    weak_classes = [c for c in ("notebook", "router", "usb_stick") if c in classes]

    model_path = run_dir / "model.joblib"
    joblib.dump(model, model_path)
    write_confusion_csv(run_dir / "confusion_matrix.csv", cm, classes)
    (run_dir / "confusion_matrix.json").write_text(
        json.dumps({"classes": classes, "matrix": cm.tolist()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "top_confusions.json").write_text(
        json.dumps(top_conf, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    version = f"classifier-baseline-{ts}"
    ds_hash = dataset_hash(args.manifests_root)
    metadata = {
        "model_version": version,
        "model_type": "classification_baseline_sgd_logloss",
        "dataset_version": "prod_datasets/unified/v1",
        "dataset_export_hash": ds_hash,
        "created_at_utc": ts,
        "image_size": args.image_size,
        "seed": args.seed,
        "classes": classes,
        "split_sizes": {
            "train": int(len(y_train)),
            "valid": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "weak_class_watchlist": weak_classes,
        "overall_metrics": {
            "accuracy": report.get("accuracy", 0.0),
            "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
            "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0.0),
        },
        "artifact_files": {
            "model": "model.joblib",
            "classification_report": "classification_report.json",
            "confusion_matrix_csv": "confusion_matrix.csv",
            "confusion_matrix_json": "confusion_matrix.json",
            "top_confusions": "top_confusions.json",
        },
    }
    (run_dir / "release_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Run dir: {run_dir.as_posix()}")
    print(f"Model version: {version}")
    print(f"Accuracy(test): {metadata['overall_metrics']['accuracy']:.4f}")
    print(f"Macro-F1(test): {metadata['overall_metrics']['macro_f1']:.4f}")
    return run_dir


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
