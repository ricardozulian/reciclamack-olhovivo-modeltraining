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


def _write_image(path: Path, variant: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        img = Image.new("RGB", (16, 16), color=(variant % 251, 20, 40))
        img.save(path)
    except Exception:
        path.write_bytes(b"fake-image-" + bytes([variant % 251]))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_manual_box_queue_yolo_multibox_and_synthetic_rows(tmp_path: Path) -> None:
    module = _load_script_module("build_manual_box_review_queue.py")

    yolo_root = tmp_path / "notebooks" / "yolo_data_computer_monitor"
    _write_image(yolo_root / "images" / "train" / "multi.jpg", 1)
    _write_image(yolo_root / "images" / "train" / "missing.jpg", 2)
    _write_image(yolo_root / "images" / "dismissed_bad" / "ignored.jpg", 3)
    (yolo_root / "labels" / "train").mkdir(parents=True)
    (yolo_root / "labels" / "train" / "multi.txt").write_text(
        "0 0.5 0.5 0.2 0.3\n1 0.2 0.2 0.1 0.1\nbad row\n",
        encoding="utf-8",
    )
    (yolo_root / "dataset.yaml").write_text(
        "names:\n  0: Computer monitor\n  1: Laptop\n",
        encoding="utf-8",
    )

    scraped_root = tmp_path / "notebooks" / "kb_monitor_scraped"
    _write_image(scraped_root / "monitor.jpg", 4)

    config = {
        "sources": [
            {
                "id": "yolo_data_computer_monitor",
                "type": "direct_yolo",
                "root": "notebooks/yolo_data_computer_monitor",
                "canonical_class": "",
                "action": "manual_separation_required",
                "source_family": "open_images",
                "source_trust": "mixed_screen",
            },
            {
                "id": "kb_monitor_scraped",
                "type": "scraped_full_image_seed",
                "root": "notebooks/kb_monitor_scraped",
                "canonical_class": "flat_monitor",
                "action": "manual_separation_required",
                "source_family": "kabum",
                "source_trust": "mixed_screen_product",
            },
        ]
    }
    _write_json(tmp_path / "config.json", config)

    summary = module.run(
        workspace_root=tmp_path,
        sources_json=Path("config.json"),
        output_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        source_ids=None,
    )

    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv")
    assert summary["queue_rows"] == 4
    assert len(rows) == 4
    assert all(not Path(row["source_path"]).is_absolute() for row in rows)
    assert not any("dismissed_bad" in row["source_path"] for row in rows)

    multi_rows = [row for row in rows if row["source_path"].endswith("multi.jpg")]
    assert len(multi_rows) == 2
    assert {row["source_class_name"] for row in multi_rows} == {"Computer monitor", "Laptop"}
    assert {row["label_parse_status"] for row in multi_rows} == {"warning_malformed_ignored"}

    missing_rows = [row for row in rows if row["source_path"].endswith("missing.jpg")]
    assert len(missing_rows) == 1
    assert missing_rows[0]["box_source"] == "synthetic_missing_label"
    assert missing_rows[0]["x_center"] == "0.5"
    assert missing_rows[0]["width"] == "1.0"

    scraped_rows = [row for row in rows if row["source_path"].endswith("monitor.jpg")]
    assert len(scraped_rows) == 1
    assert scraped_rows[0]["box_source"] == "synthetic_full_image"


def test_manual_box_export_preview_groups_accepted_boxes(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_path",
        "box_index",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "a__0",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "0",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.2",
                    "height": "0.3",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "a__1",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "1",
                    "x_center": "0.2",
                    "y_center": "0.2",
                    "width": "0.1",
                    "height": "0.1",
                    "decision_class": "exclude",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "b__0",
                    "image_md5": "def",
                    "source_path": "notebooks/x/b.jpg",
                    "box_index": "0",
                    "x_center": "0.4",
                    "y_center": "0.4",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "crt_monitor",
                    "review_status": "reviewed",
                },
            ]
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=None,
    )
    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    assert summary["accepted_box_rows"] == 2
    assert len(rows) == 2
    by_md5 = {row["image_md5"]: row for row in rows}
    assert by_md5["abc"]["preview_label_lines"] == "5 0.5 0.5 0.2 0.3"
    assert by_md5["def"]["preview_label_lines"] == "12 0.4 0.4 0.4 0.4"
    analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "box_dedupe_analysis.csv")
    assert analysis == []


def test_manual_box_export_preview_merges_compact_decisions(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    decisions = tmp_path / "dataset_staging" / "manual_review" / "box_review_decisions.csv"
    queue.parent.mkdir(parents=True)
    with queue.open("w", encoding="utf-8", newline="") as fh:
        fields = [
            "review_box_id",
            "image_md5",
            "source_path",
            "box_index",
            "x_center",
            "y_center",
            "width",
            "height",
            "decision_class",
            "review_status",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "review_box_id": "a__0",
                "image_md5": "abc",
                "source_path": "notebooks/x/a.jpg",
                "box_index": "0",
                "x_center": "0.5",
                "y_center": "0.5",
                "width": "0.2",
                "height": "0.3",
                "decision_class": "uncertain",
                "review_status": "pending",
            }
        )
    with decisions.open("w", encoding="utf-8", newline="") as fh:
        fields = ["review_box_id", "decision_class", "review_status", "notes"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "review_box_id": "a__0",
                "decision_class": "crt_monitor",
                "review_status": "accepted",
                "notes": "",
            }
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=Path("dataset_staging/manual_review/box_review_decisions.csv"),
    )
    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    assert summary["accepted_box_rows"] == 1
    assert rows[0]["preview_label_lines"] == "12 0.5 0.5 0.2 0.3"


def test_manual_box_export_dedupes_high_iou_same_class(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_path",
        "box_index",
        "box_source",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "box_a",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "box_b",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "1",
                    "box_source": "source_yolo",
                    "x_center": "0.51",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
            ]
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=None,
    )
    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "box_dedupe_analysis.csv")
    assert summary["input_accepted_box_rows"] == 2
    assert summary["accepted_box_rows"] == 1
    assert rows[0]["accepted_box_count"] == "1"
    assert len(analysis) == 1
    assert analysis[0]["reason"] == "iou_threshold"


def test_manual_box_export_drops_synthetic_full_image_when_tighter_box_exists(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_path",
        "box_index",
        "box_source",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "synthetic",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "0",
                    "box_source": "synthetic_full_image",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "1.0",
                    "height": "1.0",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "tight",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "1",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
            ]
        )

    module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=None,
    )
    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "box_dedupe_analysis.csv")
    assert rows[0]["preview_label_lines"] == "5 0.5 0.5 0.4 0.4"
    assert analysis[0]["kept_review_box_id"] == "tight"
    assert analysis[0]["dropped_review_box_id"] == "synthetic"
    assert analysis[0]["reason"] == "coverage_threshold"


def test_manual_box_export_keeps_nearby_same_class_and_overlapping_different_class(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_path",
        "box_index",
        "box_source",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "left",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.25",
                    "y_center": "0.5",
                    "width": "0.2",
                    "height": "0.2",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "right",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "1",
                    "box_source": "source_yolo",
                    "x_center": "0.55",
                    "y_center": "0.5",
                    "width": "0.2",
                    "height": "0.2",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "different_class_overlap",
                    "image_md5": "abc",
                    "source_path": "notebooks/x/a.jpg",
                    "box_index": "2",
                    "box_source": "source_yolo",
                    "x_center": "0.25",
                    "y_center": "0.5",
                    "width": "0.2",
                    "height": "0.2",
                    "decision_class": "crt_monitor",
                    "review_status": "accepted",
                },
            ]
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=None,
    )
    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "box_dedupe_analysis.csv")
    assert summary["accepted_box_rows"] == 3
    assert rows[0]["accepted_box_count"] == "3"
    assert analysis == []


def test_manual_box_export_resolves_cross_source_flat_vs_crt_to_crt(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_id",
        "source_path",
        "box_index",
        "box_source",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "monitor_box",
                    "image_md5": "same-md5",
                    "source_id": "yolo_data_computer_monitor",
                    "source_path": "notebooks/yolo_data_computer_monitor/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "tv_box",
                    "image_md5": "same-md5",
                    "source_id": "yolo_data_television",
                    "source_path": "notebooks/yolo_data_television/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "crt_monitor",
                    "review_status": "accepted",
                },
            ]
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=None,
    )

    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    source_analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "source_merge_analysis.csv")
    assert summary["accepted_box_rows"] == 1
    assert summary["dropped_source_conflict_box_rows"] == 1
    assert len(rows) == 1
    assert rows[0]["preview_label_lines"] == "12 0.5 0.5 0.4 0.4"
    assert source_analysis[0]["status"] == "resolved_flat_vs_crt_to_crt"
    assert source_analysis[0]["merged_review_box_ids"] == "tv_box"
    assert source_analysis[0]["dropped_review_box_ids"] == "monitor_box"
    duplicate_candidates = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "source_duplicate_candidates.csv")
    assert duplicate_candidates[0]["export_relevance"] == "accepted_cross_source"


def test_manual_box_export_dedupes_same_class_after_cross_source_merge(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_id",
        "source_path",
        "box_index",
        "box_source",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "monitor_box",
                    "image_md5": "same-md5",
                    "source_id": "yolo_data_computer_monitor",
                    "source_path": "notebooks/yolo_data_computer_monitor/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
                {
                    "review_box_id": "tv_box",
                    "image_md5": "same-md5",
                    "source_id": "yolo_data_television",
                    "source_path": "notebooks/yolo_data_television/a.jpg",
                    "box_index": "1",
                    "box_source": "source_yolo",
                    "x_center": "0.51",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "flat_monitor",
                    "review_status": "accepted",
                },
            ]
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=None,
    )

    rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    box_analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "box_dedupe_analysis.csv")
    source_analysis = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "source_merge_analysis.csv")
    assert summary["source_merged_box_rows"] == 2
    assert summary["accepted_box_rows"] == 1
    assert rows[0]["accepted_box_count"] == "1"
    assert box_analysis[0]["dropped_review_box_id"] == "tv_box"
    assert source_analysis[0]["status"] == "deduped_same_class_labels"


def test_manual_box_export_reports_pending_cross_source_duplicates(tmp_path: Path) -> None:
    module = _load_script_module("preview_manual_box_review_export.py")
    queue = tmp_path / "dataset_staging" / "manual_review" / "box_review_queue.csv"
    decisions = tmp_path / "dataset_staging" / "manual_review" / "box_review_decisions.csv"
    queue.parent.mkdir(parents=True)
    fields = [
        "review_box_id",
        "image_md5",
        "source_id",
        "source_path",
        "box_index",
        "box_source",
        "x_center",
        "y_center",
        "width",
        "height",
        "decision_class",
        "review_status",
    ]
    with queue.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "review_box_id": "monitor_box",
                    "image_md5": "same-md5",
                    "source_id": "yolo_data_computer_monitor",
                    "source_path": "notebooks/yolo_data_computer_monitor/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "uncertain",
                    "review_status": "pending",
                },
                {
                    "review_box_id": "tv_box",
                    "image_md5": "same-md5",
                    "source_id": "yolo_data_television",
                    "source_path": "notebooks/yolo_data_television/a.jpg",
                    "box_index": "0",
                    "box_source": "source_yolo",
                    "x_center": "0.5",
                    "y_center": "0.5",
                    "width": "0.4",
                    "height": "0.4",
                    "decision_class": "uncertain",
                    "review_status": "pending",
                },
            ]
        )
    with decisions.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["review_box_id", "decision_class", "review_status", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "review_box_id": "monitor_box",
                "decision_class": "exclude",
                "review_status": "reviewed",
                "notes": "duplicate source",
            }
        )

    summary = module.run(
        workspace_root=tmp_path,
        queue_csv=Path("dataset_staging/manual_review/box_review_queue.csv"),
        output_csv=Path("dataset_staging/manual_review/accepted_box_label_preview.csv"),
        decisions_csv=Path("dataset_staging/manual_review/box_review_decisions.csv"),
    )

    preview_rows = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "accepted_box_label_preview.csv")
    source_merge = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "source_merge_analysis.csv")
    duplicate_candidates = _read_csv(tmp_path / "dataset_staging" / "manual_review" / "source_duplicate_candidates.csv")
    assert summary["accepted_box_rows"] == 0
    assert source_merge == []
    assert preview_rows == []
    assert duplicate_candidates[0]["image_md5"] == "same-md5"
    assert duplicate_candidates[0]["export_relevance"] == "not_export_ready"
    assert duplicate_candidates[0]["decision_classes"] == "exclude|uncertain"
