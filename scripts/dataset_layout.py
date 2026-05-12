#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


CANONICAL_SPLITS = ("train", "valid", "test")


def detection_images_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / "images"


def detection_labels_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / "labels"


def unified_images_dir(dataset_root: Path, split: str) -> Path:
    return dataset_root / split / "images"


def write_detection_data_yaml(path: Path, root: Path, names: list[str]) -> None:
    lines = [
        f"path: {root.resolve().as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "",
        "names:",
    ]
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

