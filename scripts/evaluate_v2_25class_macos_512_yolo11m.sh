#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_DIR="${1:-model_pipeline/artifacts/detection_training/v2_25class_macos_512_yolo11m}"
OUTPUT_DIR="${2:-model_pipeline/artifacts/detection_training/v2_25class_macos_512_yolo11m_full_test_eval}"

python model_pipeline/scripts/evaluate_detection_model.py \
  --model "$RUN_DIR/weights/best.pt" \
  --data prod_datasets/detection/v2_25class/data.yaml \
  --dataset-root prod_datasets/detection/v2_25class \
  --split test \
  --imgsz 512 \
  --batch 4 \
  --output-dir "$OUTPUT_DIR"

python model_pipeline/scripts/plot_detection_training_metrics.py \
  model_pipeline/artifacts/detection_training/v2_25class_macos_512_parallel-2 \
  model_pipeline/artifacts/detection_training/v2_25class_macos_512_yolo11s-3 \
  "$RUN_DIR" \
  --metrics precision,recall,map50,map50_95 \
  --markers \
  --output model_pipeline/artifacts/detection_training/v2_512_11n_vs_11s_vs_11m_four_metrics.png

python model_pipeline/scripts/compare_detection_training_runs.py \
  model_pipeline/artifacts/detection_training/v2_25class_macos_512_parallel-2 \
  model_pipeline/artifacts/detection_training/v2_25class_macos_512_yolo11s-3 \
  "$RUN_DIR"
