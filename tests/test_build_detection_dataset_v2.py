from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


def _load_v2_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_detection_dataset_v2.py"
    scripts_dir = str(script_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("build_detection_dataset_v2", script_path)
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
    img.putpixel((variant % 16, (variant // 16) % 16), 255)
    img.save(path)


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
                "kb_hd_ssd_scraped,computer_part,placa eletronica,active,section 3a",
                "ml_loudspeaker_scraped,portable_music_player,player de musica portatil,active,section 3a",
                "kb_ink_cartridge_scraped,ink_toner_cartridge,cartucho de tinta ou toner,active,merge",
                "kb_toner_cartridge_scraped,ink_toner_cartridge,cartucho de tinta ou toner,active,merge",
                "ml_ink_cartridge_scraped,ink_toner_cartridge,cartucho de tinta ou toner,active,merge",
                "scraped_camera,camera,camera,active,",
                "yolo_camera,camera,camera,active,",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _base_config(tmp_path: Path) -> dict:
    return {
        "dataset_version": "v2",
        "dataset_policy_version": "v2",
        "output_root": "out",
        "split_ratio": {"train": 0.7, "valid": 0.2, "test": 0.1},
        "folder_renames": [
            {"old": "notebooks/ml_batery_scraped", "new": "notebooks/ml_battery_scraped"}
        ],
        "taxonomy": ["computer_part", "portable_music_player", "ink_toner_cartridge", "camera"],
        "portuguese_display_labels": {
            "computer_part": "placa eletronica",
            "portable_music_player": "player de musica portatil",
            "ink_toner_cartridge": "cartucho de tinta ou toner",
            "camera": "camera",
        },
        "annotation_policy": {
            "instance_mode": "multi_instance_required_when_visible",
            "unlabeled_object_policy": "warn_review_queue",
        },
        "qc": {"missing_instance_policy": "warn"},
        "sources": [],
    }


def test_real_registry_includes_section_3a_and_no_typo_source_names() -> None:
    module = _load_v2_module()
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "config" / "dataset_v2_sources.json").read_text(encoding="utf-8"))
    sources = module.parse_sources(config)

    module.validate_no_typo_sources(sources)
    by_id = {source.id: source for source in sources}
    assert by_id["kb_hd_ssd_scraped"].canonical_class == "computer_part"
    assert by_id["ml_loudspeaker_scraped"].canonical_class == "portable_music_player"
    assert by_id["kb_ink_cartridge_scraped"].canonical_class == "ink_toner_cartridge"
    assert by_id["kb_toner_cartridge_scraped"].canonical_class == "ink_toner_cartridge"
    assert by_id["ml_ink_cartridge_scraped"].canonical_class == "ink_toner_cartridge"


def test_folder_rename_preflight_renames_typo_folder(tmp_path: Path) -> None:
    module = _load_v2_module()
    config = _base_config(tmp_path)
    old_path = tmp_path / "notebooks" / "ml_batery_scraped"
    old_path.mkdir(parents=True)

    rows = module.apply_folder_renames(tmp_path, config, dry_run=False)

    assert rows[0]["status"] == "renamed"
    assert not old_path.exists()
    assert (tmp_path / "notebooks" / "ml_battery_scraped").exists()


def test_typo_source_names_are_rejected() -> None:
    module = _load_v2_module()
    source = module.Source(
        id="ml_batery_scraped",
        type="scraped_full_image_seed",
        root="notebooks/ml_batery_scraped",
        canonical_class="battery",
        action="active",
        source_family="mercado_livre",
        source_trust="studio_product",
    )

    try:
        module.validate_no_typo_sources([source])
    except ValueError as exc:
        assert "Typo source names" in str(exc)
    else:
        raise AssertionError("Expected typo source validation to fail")


def test_v2_builder_combines_yolo_scraped_section3a_and_manual_sources(tmp_path: Path) -> None:
    module = _load_v2_module()

    yolo_root = tmp_path / "notebooks" / "yolo_camera"
    _write_image(yolo_root / "images" / "train" / "cam1.jpg", variant=1)
    (yolo_root / "labels" / "train").mkdir(parents=True)
    (yolo_root / "labels" / "train" / "cam1.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (yolo_root / "labels" / "train" / "orphan.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    (yolo_root / "dataset.yaml").write_text(
        "names:\n  0: Camera\ntrain: images/train\nval: images/valid\ntest: images/test\n",
        encoding="utf-8",
    )

    scraped = tmp_path / "notebooks" / "scraped_camera"
    _write_image(scraped / "camera_product.jpg", variant=2)

    section3a = tmp_path / "notebooks" / "kb_hd_ssd_scraped"
    _write_image(section3a / "ssd_product.jpg", variant=3)

    manual = tmp_path / "notebooks" / "manual_screen"
    _write_image(manual / "screen.jpg", variant=4)

    typo = tmp_path / "notebooks" / "ml_batery_scraped"
    typo.mkdir(parents=True)

    config = _base_config(tmp_path)
    config["sources"] = [
        {
            "id": "yolo_camera",
            "type": "direct_yolo",
            "root": "notebooks/yolo_camera",
            "canonical_class": "camera",
            "action": "active",
            "source_family": "open_images",
            "source_trust": "real_context",
        },
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
            "id": "kb_hd_ssd_scraped",
            "type": "scraped_full_image_seed",
            "root": "notebooks/kb_hd_ssd_scraped",
            "canonical_class": "computer_part",
            "action": "active",
            "source_family": "kabum",
            "source_trust": "studio_product",
        },
        {
            "id": "manual_screen",
            "type": "direct_yolo",
            "root": "notebooks/manual_screen",
            "canonical_class": "flat_monitor",
            "action": "manual_separation_required",
            "source_family": "open_images",
            "source_trust": "mixed_screen",
        },
    ]

    config_path = tmp_path / "config.json"
    class_map_path = tmp_path / "class_map.csv"
    _write_json(config_path, config)
    _write_class_map(class_map_path)

    module.run(
        sources_json=Path("config.json"),
        class_map_csv=Path("class_map.csv"),
        output_root=Path("out"),
        workspace_root=tmp_path,
    )

    out = tmp_path / "out"
    rename_rows = _read_csv(out / "manifests" / "image_rename_history.csv")
    classes = {row["canonical_class"] for row in rename_rows}
    statuses = {row["annotation_status"] for row in rename_rows}

    assert "camera" in classes
    assert "computer_part" in classes
    assert "seed_full_image_needs_refinement" in statuses
    assert (tmp_path / "notebooks" / "ml_battery_scraped").exists()

    inventory = _read_csv(out / "manifests" / "source_inventory.csv")
    yolo_row = next(row for row in inventory if row["source_id"] == "yolo_camera")
    assert yolo_row["stale_label_count"] == "1"

    manual_rows = _read_csv(out / "manifests" / "manual_separation_queue.csv")
    assert len(manual_rows) == 1

    label_files = list((out / "train" / "labels").glob("*.txt"))
    assert label_files
    assert any(" 0.5 0.5 1.0 1.0" in label.read_text(encoding="utf-8") for label in label_files)


def test_v2_builder_merges_labels_but_outputs_one_image_per_md5(tmp_path: Path) -> None:
    module = _load_v2_module()

    first_root = tmp_path / "notebooks" / "yolo_camera"
    second_root = tmp_path / "notebooks" / "yolo_computer_part"
    _write_image(first_root / "images" / "train" / "dup.jpg", variant=7)
    second_image = second_root / "images" / "train" / "dup_copy.jpg"
    second_image.parent.mkdir(parents=True)
    second_image.write_bytes((first_root / "images" / "train" / "dup.jpg").read_bytes())
    (first_root / "labels" / "train").mkdir(parents=True)
    (second_root / "labels" / "train").mkdir(parents=True)
    (first_root / "labels" / "train" / "dup.txt").write_text("0 0.25 0.5 0.2 0.2\n", encoding="utf-8")
    (second_root / "labels" / "train" / "dup_copy.txt").write_text("0 0.75 0.5 0.2 0.2\n", encoding="utf-8")
    (first_root / "dataset.yaml").write_text("names:\n  0: Camera\n", encoding="utf-8")
    (second_root / "dataset.yaml").write_text("names:\n  0: Computer part\n", encoding="utf-8")

    config = _base_config(tmp_path)
    config["folder_renames"] = []
    config["sources"] = [
        {
            "id": "yolo_camera",
            "type": "direct_yolo",
            "root": "notebooks/yolo_camera",
            "canonical_class": "camera",
            "action": "active",
            "source_family": "open_images",
            "source_trust": "real_context",
        },
        {
            "id": "yolo_computer_part",
            "type": "direct_yolo",
            "root": "notebooks/yolo_computer_part",
            "canonical_class": "computer_part",
            "action": "active",
            "source_family": "open_images",
            "source_trust": "real_context",
        },
    ]
    _write_json(tmp_path / "config.json", config)
    _write_class_map(tmp_path / "class_map.csv")

    module.run(
        sources_json=Path("config.json"),
        class_map_csv=Path("class_map.csv"),
        output_root=Path("out"),
        workspace_root=tmp_path,
    )

    out = tmp_path / "out"
    image_files = list((out / "train" / "images").glob("*"))
    label_files = list((out / "train" / "labels").glob("*.txt"))
    rename_rows = _read_csv(out / "manifests" / "image_rename_history.csv")
    dedupe_report = json.loads((out / "manifests" / "dedupe_report.json").read_text(encoding="utf-8"))

    assert len(image_files) == 1
    assert len(label_files) == 1
    assert len(rename_rows) == 1
    assert dedupe_report["dropped_exact_duplicates"] == 1
    label_text = label_files[0].read_text(encoding="utf-8")
    assert "3 0.25 0.5 0.2 0.2" in label_text
    assert "0 0.75 0.5 0.2 0.2" in label_text
