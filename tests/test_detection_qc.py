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


def _write_label(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def _run_main(module, argv: list[str]) -> None:
    old_argv = sys.argv
    try:
        sys.argv = argv
        module.main()
    finally:
        sys.argv = old_argv


def test_validate_detection_accepts_multi_box_and_writes_review_queue(tmp_path: Path) -> None:
    module = _load_script_module("validate_detection_dataset.py")
    root = tmp_path / "dset"

    # train: 0 boxes (possible missing instances)
    train_img = root / "train" / "images" / "srca__celular__train__h1__img0.jpg"
    _write_image(train_img, variant=1)
    _write_label(root / "train" / "labels" / "srca__celular__train__h1__img0.txt", [])

    # valid: 2 boxes
    valid_img = root / "valid" / "images" / "srca__celular__valid__h2__img1.jpg"
    _write_image(valid_img, variant=2)
    _write_label(
        root / "valid" / "labels" / "srca__celular__valid__h2__img1.txt",
        ["0 0.5 0.5 0.2 0.2", "0 0.3 0.3 0.1 0.1"],
    )

    # test: 5 boxes (crowded scene review)
    test_img = root / "test" / "images" / "srcb__celular__test__h3__img2.jpg"
    _write_image(test_img, variant=3)
    _write_label(
        root / "test" / "labels" / "srcb__celular__test__h3__img2.txt",
        [
            "0 0.1 0.1 0.1 0.1",
            "0 0.2 0.2 0.1 0.1",
            "0 0.3 0.3 0.1 0.1",
            "0 0.4 0.4 0.1 0.1",
            "0 0.5 0.5 0.1 0.1",
        ],
    )

    output_json = root / "validation_report.json"
    output_review_queue = root / "review_queue.csv"
    _run_main(
        module,
        [
            "validate_detection_dataset.py",
            "--dataset-root",
            str(root),
            "--expected-classes",
            "celular",
            "--skip-coverage-check",
            "--missing-instance-policy",
            "warn",
            "--min-images-for-qc",
            "1",
            "--output-json",
            str(output_json),
            "--output-review-queue",
            str(output_review_queue),
        ],
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["errors"] == []
    assert "2" in report["splits"]["valid"]["box_count_distribution"]
    assert "5" in report["splits"]["test"]["box_count_distribution"]
    assert report["qc_review_queue_size"] >= 2

    with output_review_queue.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    reasons = {r["reason"] for r in rows}
    assert "possible_missing_instances" in reasons
    assert "crowded_scene_needs_review" in reasons


def test_audit_detection_includes_box_distribution_and_qc_warning(tmp_path: Path) -> None:
    module = _load_script_module("audit_detection_labels.py")
    root = tmp_path / "dset"

    # Build a split with only one-box files to trigger low multi-box warning.
    for idx, split in enumerate(["train", "valid", "test"], start=1):
        _write_label(
            root / split / "labels" / f"img_{idx}.txt",
            ["0 0.5 0.5 0.2 0.2"],
        )

    output_json = root / "label_audit_report.json"
    _run_main(
        module,
        [
            "audit_detection_labels.py",
            "--dataset-root",
            str(root),
            "--missing-instance-policy",
            "warn",
            "--min-files-for-qc",
            "1",
            "--min-multi-box-ratio",
            "0.50",
            "--output-json",
            str(output_json),
        ],
    )

    report = json.loads(output_json.read_text(encoding="utf-8"))
    assert report["errors"] == []
    assert report["summary"]["box_count_distribution"]["1"] == 3
    assert any("low multi-box ratio" in msg for msg in report["warnings"])
