#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PATTERNS = (
    "*.jpg",
    "*.png",
    "weights/last.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune nonessential YOLO training artifacts.")
    parser.add_argument("runs", nargs="+", type=Path, help="Training run directories to prune.")
    parser.add_argument("--apply", action="store_true", help="Delete files. Without this, only prints a dry run.")
    parser.add_argument(
        "--pattern",
        action="append",
        default=list(DEFAULT_PATTERNS),
        help="Glob pattern relative to each run directory. Can be repeated.",
    )
    return parser.parse_args()


def collect_files(run_dir: Path, patterns: list[str]) -> list[Path]:
    files: dict[Path, None] = {}
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file():
                files[path] = None
    return sorted(files)


def size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    args = parse_args()
    total = 0.0
    candidates: list[Path] = []

    for run_dir in args.runs:
        if not run_dir.exists():
            raise FileNotFoundError(run_dir)
        files = collect_files(run_dir, args.pattern)
        candidates.extend(files)

    for path in candidates:
        mb = size_mb(path)
        total += mb
        action = "DELETE" if args.apply else "DRY-RUN"
        print(f"{action} {mb:8.2f} MB {path.as_posix()}")

    print(f"{'Deleted' if args.apply else 'Would delete'} {len(candidates)} files, {total:.2f} MB")

    if args.apply:
        for path in candidates:
            path.unlink()


if __name__ == "__main__":
    main()
