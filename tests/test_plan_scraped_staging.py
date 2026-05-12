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


def _tiny_png_bytes(variant: int = 0) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10"
        b"\x08\x02\x00\x00\x00\x90\x91h6\x00\x00\x00\x19IDATx\x9ccddbf\xa0\x040Q\xa4"
        b"\x86Q\rCH\x03\x00'\xd4\x01\xb0\x95\xc6\x8e\x8f\x00\x00\x00\x00IEND\xaeB`\x82"
        + bytes([variant % 251])
    )


def _write_image(path: Path, variant: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_tiny_png_bytes(variant))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_class_map(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "source_folder,canonical_class,portuguese_display_label,status,notes",
                "scraped_camera,camera,camera,active,",
                "scraped_camera_alt,camera,camera,active,",
                "yolo_camera,camera,camera,active,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_targets(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "canonical_class,target_total,batch_size,priority,notes",
                "camera,10,2,high,test target",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_scraped_staging_plan_uses_relative_md5_names_and_moves_no_images(tmp_path: Path) -> None:
    module = _load_script_module("plan_scraped_staging.py")

    _write_image(tmp_path / "notebooks" / "scraped_camera" / "cam_a.jpg", variant=1)
    _write_image(tmp_path / "notebooks" / "scraped_camera" / "nested" / "cam_b.png", variant=2)
    duplicate_bytes = _tiny_png_bytes(3)
    dup_a = tmp_path / "notebooks" / "scraped_camera" / "dup_a.jpg"
    dup_b = tmp_path / "notebooks" / "scraped_camera_alt" / "dup_b.jpg"
    dup_a.parent.mkdir(parents=True, exist_ok=True)
    dup_b.parent.mkdir(parents=True, exist_ok=True)
    dup_a.write_bytes(duplicate_bytes)
    dup_b.write_bytes(duplicate_bytes)

    # Non-scraped sources must not be inventoried by this planning script.
    _write_image(tmp_path / "notebooks" / "yolo_camera" / "images" / "val" / "cam_yolo.jpg", variant=4)

    config = {
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
                "id": "scraped_camera_alt",
                "type": "scraped_full_image_seed",
                "root": "notebooks/scraped_camera_alt",
                "canonical_class": "camera",
                "action": "active",
                "source_family": "mercado_livre",
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
        ]
    }
    config_path = tmp_path / "config" / "sources.json"
    class_map_path = tmp_path / "config" / "class_map.csv"
    targets_path = tmp_path / "dataset_staging" / "staging_targets.example.csv"
    _write_json(config_path, config)
    _write_class_map(class_map_path)
    _write_targets(targets_path)

    summary = module.run(
        sources_json=Path("config/sources.json"),
        class_map_csv=Path("config/class_map.csv"),
        workspace_root=tmp_path,
        output_root=Path("dataset_staging"),
        targets_csv=Path("dataset_staging/staging_targets.example.csv"),
        batch_id="batch_001",
    )

    assert summary["scraped_inventory_rows"] == 4
    assert summary["unique_images"] == 3
    assert summary["duplicate_images"] == 1
    assert not (tmp_path / "dataset_staging" / "class_batches").exists()

    inventory = _read_csv(tmp_path / "dataset_staging" / "manifests" / "scraped_inventory.csv")
    assert all(row["image_id"] == row["md5"] for row in inventory)
    assert all(not Path(row["source_path"]).is_absolute() for row in inventory)
    assert all(not Path(row["future_staged_image"]).is_absolute() for row in inventory)
    assert all(not Path(row["future_staged_label"]).is_absolute() for row in inventory)
    assert {row["status"] for row in inventory} == {"scraped_unused", "duplicate_exact_content"}

    for row in inventory:
        assert Path(row["future_staged_image"]).stem == row["md5"]
        assert Path(row["future_staged_label"]).stem == row["md5"]
        assert row["full_image_label"] == "0 0.5 0.5 1.0 1.0"
        assert row["future_staged_image"].startswith("dataset_staging/class_batches/camera/batch_001/train/images/")
        assert row["future_staged_label"].startswith("dataset_staging/class_batches/camera/batch_001/train/labels/")

    duplicates = _read_csv(tmp_path / "dataset_staging" / "manifests" / "duplicate_images.csv")
    assert len(duplicates) == 1

    preview = _read_csv(tmp_path / "dataset_staging" / "manifests" / "batch_preview.csv")
    assert len(preview) == 2
    assert all(row["roboflow_local_class_id"] == "0" for row in preview)
    assert all(row["upload_split"] == "train" for row in preview)
    assert all(row["full_image_label"] == "0 0.5 0.5 1.0 1.0" for row in preview)


def test_direct_yolo_split_behavior_remains_unchanged() -> None:
    module = _load_script_module("build_detection_dataset_v2.py")

    assert module.split_from_path(Path("notebooks/yolo_data_camera/images/train/a.jpg")) == "train"
    assert module.split_from_path(Path("notebooks/yolo_data_camera/images/val/a.jpg")) == "valid"
    assert module.split_from_path(Path("notebooks/yolo_data_camera/images/valid/a.jpg")) == "valid"
    assert module.split_from_path(Path("notebooks/yolo_data_camera/images/test/a.jpg")) == "test"


def test_dismiss_typo_folders_are_ignored() -> None:
    module = _load_script_module("plan_scraped_staging.py")

    assert module.is_ignored_folder("dismissed_small")
    assert module.is_ignored_folder("dismiss_manual")
    assert module.is_ignored_folder("toner_dismis")
    assert not module.is_ignored_folder("kb_toner_cartridge_scraped")
