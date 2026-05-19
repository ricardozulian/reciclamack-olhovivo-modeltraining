#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRICS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metric deltas between two YOLO results.csv files.")
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline run directory containing results.csv.")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate run directory containing results.csv.")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--max-epoch", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", default="precision,recall,map50,map50_95")
    return parser.parse_args()


def read_rows(run_dir: Path) -> dict[int, dict[str, float]]:
    path = run_dir / "results.csv"
    rows: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            epoch = int(float(row["epoch"]))
            rows[epoch] = {name: float(row[column]) for name, column in METRICS.items()}
    return rows


def main() -> None:
    args = parse_args()
    selected = [item.strip() for item in args.metrics.split(",") if item.strip()]
    unknown = [item for item in selected if item not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics: {', '.join(unknown)}")

    baseline = read_rows(args.baseline)
    candidate = read_rows(args.candidate)
    epochs = sorted(set(baseline).intersection(candidate))
    if args.max_epoch is not None:
        epochs = [epoch for epoch in epochs if epoch <= args.max_epoch]
    if not epochs:
        raise ValueError("No shared epochs to plot")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6.5))
    for metric in selected:
        deltas = [candidate[epoch][metric] - baseline[epoch][metric] for epoch in epochs]
        plt.plot(epochs, deltas, marker="o", markersize=4, linewidth=2.2, label=f"{metric} delta")

    plt.axhline(0, color="black", linewidth=1, alpha=0.6)
    plt.title(f"{args.candidate_label} minus {args.baseline_label} metric deltas")
    plt.xlabel("Epoch")
    plt.ylabel("Metric delta")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=160)
    print(args.output.as_posix())


if __name__ == "__main__":
    main()
