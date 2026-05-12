from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_unified_dataset.py"
    spec = importlib.util.spec_from_file_location("build_unified_dataset", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_png_bytes() -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10\x00\x00\x00\x10"
        b"\x08\x02\x00\x00\x00\x90\x91h6\x00\x00\x00\x19IDATx\x9ccddbf\xa0\x040Q\xa4"
        b"\x86Q\rCH\x03\x00'\xd4\x01\xb0\x95\xc6\x8e\x8f\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _write_image(path: Path, variant: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
    except Exception:
        path.write_bytes(_tiny_png_bytes() * 3 + bytes([variant % 251]))
        return

    img = Image.new("L", (16, 16), color=32 + (variant % 16))
    x = variant % 16
    y = (variant // 16) % 16
    img.putpixel((x, y), 255)
    img.save(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_pipeline_mapping_dedup_and_negative_policy(tmp_path: Path) -> None:
    module = _load_builder_module()

    # Folder source 1
    _write_image(tmp_path / "raw" / "src_a" / "Mobile" / "img_mobile_1.jpg", variant=1)
    _write_image(tmp_path / "raw" / "src_a" / "laptop" / "img_laptop_1.jpg", variant=2)
    # Exact duplicate (same bytes) in another place to force dedup
    dup_path = tmp_path / "raw" / "src_a" / "laptop" / "img_laptop_dup.jpg"
    dup_path.parent.mkdir(parents=True, exist_ok=True)
    dup_path.write_bytes((tmp_path / "raw" / "src_a" / "laptop" / "img_laptop_1.jpg").read_bytes())

    # Folder source 2
    _write_image(tmp_path / "raw" / "src_b" / "keyboard I" / "img_keyboard_1.jpg", variant=3)

    # Garbage YOLO source with one allowed negative (0) and one blocked (5)
    garbage_root = tmp_path / "raw" / "garbage"
    _write_image(garbage_root / "train" / "images" / "ok.jpg", variant=4)
    (garbage_root / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (garbage_root / "train" / "labels" / "ok.txt").write_text(
        "0 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )
    _write_image(garbage_root / "train" / "images" / "blocked.jpg", variant=5)
    (garbage_root / "train" / "labels" / "blocked.txt").write_text(
        "5 0.5 0.5 0.3 0.3\n", encoding="utf-8"
    )

    config = {
        "seed": 7,
        "split_ratio": {"train": 0.7, "valid": 0.2, "test": 0.1},
        "negative_ratio_target": 0.2,
        "negative_ratio_tolerance": 0.02,
        "dedup": {"enabled": True, "near_hamming_threshold": 5, "bucket_prefix_len": 4},
        "quality": {"min_file_size_bytes": 1},
        "minimum_class_support": 1,
        "taxonomy": ["celular", "notebook", "keyboard"],
        "label_map": {"mobile": "celular", "laptop": "notebook", "keyboard_i": "keyboard"},
        "review_map": {},
        "negative_labels": ["negative"],
        "component_label_map": {},
        "sources": [
            {"id": "src_a", "type": "folder", "root": "raw/src_a"},
            {"id": "src_b", "type": "folder", "root": "raw/src_b"},
            {
                "id": "garbage_yolo",
                "type": "garbage_yolo_negative",
                "root": "raw/garbage",
                "allowed_negative_ids": [0, 1, 2, 3, 4, 6],
                "blocked_ids": [5],
            },
        ],
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    cwd = Path.cwd()
    try:
        # Run from tmp root so relative paths in config resolve as expected.
        import os

        os.chdir(tmp_path)
        module.run(Path("config.json"), Path("out"), dry_run=False)
    finally:
        os.chdir(cwd)

    summary = json.loads((tmp_path / "out" / "manifests" / "inventory_summary.json").read_text(encoding="utf-8"))
    train_rows = _read_csv(tmp_path / "out" / "manifests" / "train.csv")
    valid_rows = _read_csv(tmp_path / "out" / "manifests" / "valid.csv")
    test_rows = _read_csv(tmp_path / "out" / "manifests" / "test.csv")
    all_rows = train_rows + valid_rows + test_rows

    # Mapping correctness: laptop should be normalized to notebook.
    mapped_labels = {row["mapped_label"] for row in all_rows if row["mapped_label"]}
    assert "laptop" not in mapped_labels
    assert "notebook" in mapped_labels
    assert "celular" in mapped_labels
    assert "keyboard" in mapped_labels

    # Garbage negative policy: blocked id 5 image is excluded.
    source_paths = {row["source_relpath"] for row in all_rows}
    assert "raw/garbage/train/images/blocked.jpg" not in source_paths
    assert "raw/garbage/train/images/ok.jpg" in source_paths

    # Dedup should drop the exact duplicate.
    assert summary["dedup_stats"]["dropped_records"] >= 1

    # Traceability fields should exist and filenames should be deterministic-like.
    sample = all_rows[0]
    assert sample["source_dataset"]
    assert sample["source_relpath"]
    assert sample["file_hash_sha1"]
    assert "__" in sample["final_filename"]


def test_v2_policy_defaults_and_review_reason_mapping(tmp_path: Path) -> None:
    module = _load_builder_module()

    version, annotation, qc = module.get_policy_config(
        {
            "dataset_policy_version": "v2",
            "annotation_policy": {
                "instance_mode": "multi_instance_required_when_visible",
                "unlabeled_object_policy": "warn_review_queue",
            },
            "qc": {"missing_instance_policy": "warn"},
        }
    )
    assert version == "v2"
    assert annotation["instance_mode"] == "multi_instance_required_when_visible"
    assert annotation["unlabeled_object_policy"] == "warn_review_queue"
    assert qc["missing_instance_policy"] == "warn"

    assert (
        module.normalize_review_reason("single_object_problem", annotation)
        == "possible_missing_instances"
    )
    assert (
        module.normalize_review_reason("dense_scene_manual_check", annotation)
        == "crowded_scene_needs_review"
    )
    assert (
        module.normalize_review_reason("heavy_occlusion_case", annotation)
        == "ambiguous_occlusion"
    )


def test_v1_policy_default_keeps_legacy_behavior() -> None:
    module = _load_builder_module()
    version, annotation, qc = module.get_policy_config({})
    assert version == "v1"
    assert annotation["instance_mode"] == "legacy"
    assert annotation["unlabeled_object_policy"] == "off"
    assert qc["missing_instance_policy"] == "off"
    assert module.normalize_review_reason("single_object_problem", annotation) == "single_object_problem"
