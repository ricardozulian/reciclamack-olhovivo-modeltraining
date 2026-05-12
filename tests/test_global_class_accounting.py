from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_jpg(path: Path, variant: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        img = Image.new("RGB", (16, 16), color=(variant, 20, 40))
        img.save(path)
    except Exception:
        path.write_bytes(b"fake-jpg-" + bytes([variant % 251]))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_global_accounting_combines_scraped_and_yolo_counts(tmp_path: Path) -> None:
    planner = _load_script_module("plan_scraped_staging.py")
    accounting = _load_script_module("build_global_class_accounting.py")

    _tiny_jpg(tmp_path / "notebooks" / "scraped_camera" / "usable.jpg", 1)
    _tiny_jpg(tmp_path / "notebooks" / "scraped_camera" / "dismissed_manual" / "ignored.jpg", 2)

    _tiny_jpg(tmp_path / "notebooks" / "yolo_camera" / "images" / "train" / "train.jpg", 3)
    _tiny_jpg(tmp_path / "notebooks" / "yolo_camera" / "images" / "val" / "valid.jpg", 4)
    _tiny_jpg(tmp_path / "notebooks" / "yolo_camera" / "images" / "test" / "test.jpg", 5)
    _tiny_jpg(tmp_path / "notebooks" / "yolo_camera" / "images" / "dismissed_bad" / "ignored.jpg", 6)
    label_root = tmp_path / "notebooks" / "yolo_camera" / "labels"
    for split, name in [("train", "train"), ("val", "valid"), ("test", "test"), ("dismissed_bad", "ignored")]:
        label = label_root / split / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    config = {
        "taxonomy": ["camera"],
        "sources": [
            {
                "id": "scraped_camera",
                "type": "scraped_full_image_seed",
                "root": "notebooks/scraped_camera",
                "canonical_class": "camera",
                "action": "active",
                "source_family": "kabum",
                "source_trust": "studio_product",
            },
            {
                "id": "yolo_camera",
                "type": "direct_yolo",
                "root": "notebooks/yolo_camera",
                "canonical_class": "camera",
                "action": "active",
                "source_family": "open_images",
                "source_trust": "real_context",
            },
        ],
    }
    _write_json(tmp_path / "config" / "sources.json", config)
    _write_csv(
        tmp_path / "config" / "class_map.csv",
        [
            {
                "source_folder": "scraped_camera",
                "canonical_class": "camera",
                "portuguese_display_label": "camera",
                "status": "active",
                "notes": "",
            },
            {
                "source_folder": "yolo_camera",
                "canonical_class": "camera",
                "portuguese_display_label": "camera",
                "status": "active",
                "notes": "",
            },
        ],
        ["source_folder", "canonical_class", "portuguese_display_label", "status", "notes"],
    )

    planner.run(
        sources_json=Path("config/sources.json"),
        class_map_csv=Path("config/class_map.csv"),
        workspace_root=tmp_path,
        output_root=Path("dataset_staging"),
        targets_csv=Path("dataset_staging/staging_targets.example.csv"),
    )
    _write_csv(
        tmp_path / "dataset_staging" / "staging_targets.example.csv",
        [
            {
                "canonical_class": "camera",
                "target_total": "10",
                "batch_size": "4",
                "priority": "high",
                "notes": "target test",
            }
        ],
        ["canonical_class", "target_total", "batch_size", "priority", "notes"],
    )

    summary = accounting.run(
        sources_json=Path("config/sources.json"),
        class_map_csv=Path("config/class_map.csv"),
        scraped_inventory_csv=Path("dataset_staging/manifests/scraped_inventory.csv"),
        targets_csv=Path("dataset_staging/staging_targets.example.csv"),
        workspace_root=tmp_path,
        output_root=Path("dataset_staging"),
    )

    assert summary["scraped_unique_available"] == 1
    assert summary["open_images_labeled_count"] == 3

    rows = _read_csv(tmp_path / "dataset_staging" / "manifests" / "global_class_accounting.csv")
    camera = rows[0]
    assert camera["canonical_class"] == "camera"
    assert camera["target_v2"] == "10"
    assert camera["batch_size"] == "4"
    assert camera["current_labeled_yolo_count"] == "3"
    assert camera["open_images_train_count"] == "1"
    assert camera["open_images_valid_count"] == "1"
    assert camera["open_images_test_count"] == "1"
    assert camera["scraped_raw_count"] == "1"
    assert camera["scraped_unique_available"] == "1"
    assert camera["scraped_unused_available"] == "1"
    assert camera["gap_to_target"] == "7"

    planning = _read_csv(tmp_path / "dataset_staging" / "v2_target_planning.csv")
    assert planning[0]["canonical_class"] == "camera"
    assert planning[0]["suggested_target_v2"] == "4"
