#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METRIC_COLUMNS = (
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare YOLO detection training results.csv files.")
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories containing results.csv.")
    return parser.parse_args()


def read_results(run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def to_float(row: dict[str, str], key: str) -> float:
    return float(row.get(key, "0") or 0)


def summarize(run_dir: Path) -> dict[str, str | float | int]:
    rows = read_results(run_dir)
    if not rows:
        raise ValueError(f"No rows in {run_dir / 'results.csv'}")

    best = max(rows, key=lambda row: to_float(row, "metrics/mAP50-95(B)"))
    latest = rows[-1]
    summary: dict[str, str | float | int] = {
        "run": run_dir.as_posix(),
        "latest_epoch": int(float(latest["epoch"])),
        "best_epoch": int(float(best["epoch"])),
    }
    for column in METRIC_COLUMNS:
        short = column.replace("metrics/", "").replace("(B)", "")
        summary[f"latest_{short}"] = to_float(latest, column)
        summary[f"best_{short}"] = to_float(best, column)
    return summary


def main() -> None:
    args = parse_args()
    summaries = [summarize(run) for run in args.runs]
    headers = [
        "run",
        "latest_epoch",
        "best_epoch",
        "latest_precision",
        "latest_recall",
        "latest_mAP50",
        "latest_mAP50-95",
        "best_precision",
        "best_recall",
        "best_mAP50",
        "best_mAP50-95",
    ]

    print(",".join(headers))
    for row in summaries:
        print(",".join(str(row.get(header, "")) for header in headers))


if __name__ == "__main__":
    main()
