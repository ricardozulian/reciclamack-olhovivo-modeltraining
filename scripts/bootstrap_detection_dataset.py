#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

from dataset_layout import CANONICAL_SPLITS, detection_images_dir, detection_labels_dir, write_detection_data_yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_ORDER = [
    "celular",
    "notebook",
    "televisao",
    "bateria",
    "placa_eletronica",
    "capacitor",
    "cabo",
    "router",
    "usb_stick",
]
PRIORITY = {"celular", "bateria", "placa_eletronica", "capacitor", "televisao", "cabo"}
SECONDARY = {"notebook", "router", "usb_stick"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap YOLO detection dataset from ImageFolder split dataset.")
    parser.add_argument("--source-root", type=Path, default=Path("prod_datasets/unified/v1"))
    parser.add_argument("--output-root", type=Path, default=Path("prod_datasets/detection/v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-all",
        action="store_true",
        help="Copy all available images for selected classes; otherwise use conservative caps.",
    )
    parser.add_argument(
        "--include-secondary",
        action="store_true",
        help="Include notebook/router/usb_stick in the bootstrap set.",
    )
    return parser.parse_args()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def ensure_layout(root: Path) -> None:
    for split in CANONICAL_SPLITS:
        detection_images_dir(root, split).mkdir(parents=True, exist_ok=True)
        detection_labels_dir(root, split).mkdir(parents=True, exist_ok=True)


def write_data_yaml(root: Path) -> None:
    write_detection_data_yaml(root / "data.yaml", root, CLASS_ORDER)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    selected = set(PRIORITY)
    if args.include_secondary:
        selected.update(SECONDARY)

    source_root = args.source_root
    out_root = args.output_root
    ensure_layout(out_root)

    # Conservative caps for first sprint if --use-all is not set.
    caps = {
        "train": {"priority": 220, "secondary": 80, "negative": 240},
        "valid": {"priority": 70, "secondary": 30, "negative": 80},
        "test": {"priority": 35, "secondary": 20, "negative": 40},
    }

    manifest_rows: list[dict[str, str]] = []
    class_counts = {(split, cls): 0 for split in CANONICAL_SPLITS for cls in CLASS_ORDER}
    neg_counts = {split: 0 for split in CANONICAL_SPLITS}

    for split in CANONICAL_SPLITS:
        split_src = source_root / split / "images"
        if not split_src.exists():
            continue

        for cls in sorted(selected):
            cls_src = split_src / cls
            if not cls_src.exists():
                continue
            files = [p for p in cls_src.rglob("*") if is_image(p)]
            random.shuffle(files)
            if args.use_all:
                chosen = files
            else:
                cap = caps[split]["priority"] if cls in PRIORITY else caps[split]["secondary"]
                chosen = files[:cap]

            for idx, src in enumerate(chosen, start=1):
                target_stem = f"{split}__{cls}__{idx:05d}__{src.stem}"
                target_img = detection_images_dir(out_root, split) / f"{target_stem}{src.suffix.lower()}"
                target_lbl = detection_labels_dir(out_root, split) / f"{target_stem}.txt"
                shutil.copy2(src, target_img)
                target_lbl.write_text("", encoding="utf-8")  # empty label => pending annotation
                class_counts[(split, cls)] += 1
                manifest_rows.append(
                    {
                        "split": split,
                        "target_class": cls,
                        "source_path": src.as_posix(),
                        "target_image": target_img.relative_to(out_root).as_posix(),
                        "target_label": target_lbl.relative_to(out_root).as_posix(),
                        "annotation_status": "pending",
                    }
                )

        # Add hard negatives for each split.
        neg_src = split_src / "negative"
        if neg_src.exists():
            files = [p for p in neg_src.rglob("*") if is_image(p)]
            random.shuffle(files)
            chosen = files if args.use_all else files[: caps[split]["negative"]]
            for idx, src in enumerate(chosen, start=1):
                target_stem = f"{split}__negative__{idx:05d}__{src.stem}"
                target_img = detection_images_dir(out_root, split) / f"{target_stem}{src.suffix.lower()}"
                target_lbl = detection_labels_dir(out_root, split) / f"{target_stem}.txt"
                shutil.copy2(src, target_img)
                target_lbl.write_text("", encoding="utf-8")
                neg_counts[split] += 1
                manifest_rows.append(
                    {
                        "split": split,
                        "target_class": "negative",
                        "source_path": src.as_posix(),
                        "target_image": target_img.relative_to(out_root).as_posix(),
                        "target_label": target_lbl.relative_to(out_root).as_posix(),
                        "annotation_status": "pending_negative",
                    }
                )

    write_data_yaml(out_root)

    with (out_root / "annotation_seed.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "split",
                "target_class",
                "source_path",
                "target_image",
                "target_label",
                "annotation_status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    summary_lines = [
        "# prod_datasets/detection/v1 bootstrap summary",
        "",
        f"- seed: {args.seed}",
        f"- use_all: {str(args.use_all).lower()}",
        f"- include_secondary: {str(args.include_secondary).lower()}",
        "",
        "## Class counts by split",
    ]
    for split in CANONICAL_SPLITS:
        summary_lines.append(f"- {split}:")
        for cls in CLASS_ORDER:
            summary_lines.append(f"  - {cls}: {class_counts[(split, cls)]}")
        summary_lines.append(f"  - negative: {neg_counts[split]}")
    (out_root / "bootstrap_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    readme = out_root / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# prod_datasets/detection/v1",
                "",
                "This is a bootstrap detection dataset prepared from `prod_datasets/unified/v1`.",
                "",
                "Important:",
                "- All label files are currently empty and must be annotated with YOLO boxes.",
                "- `negative` images should remain empty label files.",
                "- Use `annotation_seed.csv` to track annotation progress.",
                "",
                "After annotation, run:",
                "python model_pipeline/scripts/validate_detection_dataset.py --dataset-root prod_datasets/detection/v1 --output-json prod_datasets/detection/v1/validation_report.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote: {out_root.as_posix()}")
    print(f"Annotation rows: {len(manifest_rows)}")


if __name__ == "__main__":
    main()
