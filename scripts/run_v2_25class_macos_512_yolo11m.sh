#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

python model_pipeline/scripts/train_detection_local.py \
  --config model_pipeline/config/detection_local_train.v2_25class_macos_512_yolo11m.yaml
