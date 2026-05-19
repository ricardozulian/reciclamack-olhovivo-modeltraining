#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = {
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot YOLO training metrics from one or more results.csv files.")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing results.csv.")
    parser.add_argument("--output", type=Path, default=Path("model_pipeline/artifacts/detection_training/v2_metric_comparison.png"))
    parser.add_argument(
        "--metrics",
        default="map50,map50_95",
        help=f"Comma-separated metrics to plot. Options: {', '.join(METRICS)}",
    )
    parser.add_argument("--max-epoch", type=float, default=None, help="Only plot epochs up to this value.")
    parser.add_argument("--latest-window", type=float, default=None, help="Plot the latest N epochs.")
    parser.add_argument(
        "--latest-window-mode",
        choices=["per-run", "shared-min", "last-run"],
        default="per-run",
        help="How to anchor --latest-window. per-run uses each run's own latest epoch; shared-min uses the smallest latest epoch across runs; last-run uses the last run argument.",
    )
    parser.add_argument("--markers", action="store_true", help="Draw point markers on every epoch.")
    return parser.parse_args()


def read_rows(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            parsed = {"epoch": float(row["epoch"])}
            for name, column in METRICS.items():
                parsed[name] = float(row[column])
            rows.append(parsed)
    return rows


def run_label(run_dir: Path) -> str:
    name = run_dir.name
    return name.replace("v2_25class_macos_", "")


def marker_for_run(run_dir: Path, enabled: bool) -> str | None:
    if not enabled:
        return None
    name = run_dir.name
    if "576" in name:
        return "x"
    if "512" in name:
        return "D" if "yolo11s" in name else "^"
    if "768" in name:
        return "s"
    return "o"


def line_style_for_run(run_dir: Path) -> str:
    name = run_dir.name
    if "576" in name:
        return "-"
    if "512" in name:
        return "--" if "yolo11s" in name else "-."
    if "768" in name:
        return ":"
    if "640" in name:
        return "--"
    return "-"


def marker_size_for_run(run_dir: Path) -> int:
    if "576" in run_dir.name:
        return 9
    if "512" in run_dir.name:
        return 8 if "yolo11s" in run_dir.name else 7
    return 4


def alpha_for_run(run_dir: Path) -> float:
    return 0.5 if "640" in run_dir.name else 1.0


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.metrics.split(",") if item.strip()]
    unknown = [item for item in selected if item not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(unknown)}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6.5))
    all_rows_by_run: list[tuple[Path, list[dict[str, float]]]] = []
    for run_dir in args.runs:
        rows = read_rows(run_dir)
        if args.max_epoch is not None:
            rows = [row for row in rows if row["epoch"] <= args.max_epoch]
        all_rows_by_run.append((run_dir, rows))

    if args.latest_window is not None and all_rows_by_run:
        latest_by_run = [max((row["epoch"] for row in rows), default=0.0) for _, rows in all_rows_by_run]
        if args.latest_window_mode == "shared-min":
            anchor_epoch = min(epoch for epoch in latest_by_run if epoch > 0)
            min_epoch = max(0.0, anchor_epoch - args.latest_window + 1)
            all_rows_by_run = [
                (run_dir, [row for row in rows if min_epoch <= row["epoch"] <= anchor_epoch])
                for run_dir, rows in all_rows_by_run
            ]
        elif args.latest_window_mode == "last-run":
            anchor_epoch = latest_by_run[-1]
            min_epoch = max(0.0, anchor_epoch - args.latest_window + 1)
            all_rows_by_run = [
                (run_dir, [row for row in rows if min_epoch <= row["epoch"] <= anchor_epoch])
                for run_dir, rows in all_rows_by_run
            ]
        else:
            filtered: list[tuple[Path, list[dict[str, float]]]] = []
            for run_dir, rows in all_rows_by_run:
                latest_epoch = max((row["epoch"] for row in rows), default=0.0)
                min_epoch = max(0.0, latest_epoch - args.latest_window + 1)
                filtered.append((run_dir, [row for row in rows if row["epoch"] >= min_epoch]))
            all_rows_by_run = filtered

    for run_dir, rows in all_rows_by_run:
        if not rows:
            continue
        label = run_label(run_dir)
        epochs = [row["epoch"] for row in rows]
        for metric in selected:
            values = [row[metric] for row in rows]
            best = max(rows, key=lambda row: row[metric])
            marker = marker_for_run(run_dir, args.markers)
            plt.plot(
                epochs,
                values,
                linewidth=2.2,
                linestyle=line_style_for_run(run_dir),
                marker=marker,
                markersize=marker_size_for_run(run_dir),
                markeredgewidth=1.7,
                alpha=alpha_for_run(run_dir),
                label=f"{label} {metric}",
            )
            plt.scatter([best["epoch"]], [best[metric]], s=42, zorder=5, alpha=alpha_for_run(run_dir))

    plt.title("ReciclaMack v2 YOLO11n training metrics")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
