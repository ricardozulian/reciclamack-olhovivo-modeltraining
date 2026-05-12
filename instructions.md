# Local Training Instructions (Mac/SMB Side)

This guide is for running detection training from your MacBook, including when the project is mounted over SMB.

## 1) Open project folder on Mac

Use the mounted path to this repository, then open a terminal in repo root.

## 2) Create a fresh conda environment

```bash
conda create -n reciclamack-train python=3.11 -y
conda activate reciclamack-train
python -m pip install --upgrade pip
pip install -r model_pipeline/requirements.txt
```

## 3) Confirm key packages

```bash
python -c "import torch, ultralytics, yaml; print('torch', torch.__version__); print('ultralytics', ultralytics.__version__)"
```

## 4) Run dataset preflight

```bash
python model_pipeline/scripts/audit_detection_labels.py \
  --dataset-root prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8 \
  --output-json prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8/label_audit_report.json

python model_pipeline/scripts/validate_detection_dataset.py \
  --dataset-root prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8 \
  --data-yaml prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8/data.yaml \
  --output-json prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8/validation_report.json \
  --output-review-queue prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8/review_queue.csv \
  --missing-instance-policy warn
```

Expected:
- no validation errors,
- no polygon rows.

## 4a) Interpret Step 4 output (current dataset)

For `prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8`, this is the expected interpretation:

- PASS:
  - `files_with_polygons = 0`
  - `malformed_rows = 0`
  - `validation_report.errors = []`
  - split pairing is consistent (images/labels aligned)
- WARNING (non-blocking for first tentative run):
  - `telephone` is missing in `valid` and `test`.

Meaning:
- You can proceed with the first tentative training round.
- Metrics for `telephone` will not be reliable until you add `valid/test` coverage for that class.

Quick summary command (after Step 4 files exist):

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path("prod_datasets/detection/reciclamack-detection-v1.v2-2026-04-21-first-fix.yolov8")
audit = json.loads((root / "label_audit_report.json").read_text(encoding="utf-8"))
val = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
s = audit.get("summary", {})
print("AUDIT:", s)
print("VALIDATION errors:", val.get("errors", []))
print("VALIDATION warnings:", val.get("warnings", []))
ok = (
    s.get("files_with_polygons", 1) == 0
    and s.get("malformed_rows", 1) == 0
    and len(val.get("errors", [])) == 0
)
print("STATUS:", "PASS" if ok else "FAIL")
PY
```

## 5) Run first local training round

```bash
python model_pipeline/scripts/train_detection_local.py \
  --config model_pipeline/config/detection_local_train.first_round.yaml
```

Notes:
- Device is `auto` in config, so Apple Silicon should use `mps` when available.
- You can override per run, for example:
  - `--epochs 5`
  - `--imgsz 640`
  - `--batch 8`
  - `--device mps`

## 6) Where outputs are written

Training artifacts are saved under:

`model_pipeline/artifacts/detection_training/<run_name>/`

Files include:
- `run_config_resolved.json`
- `metrics_summary.json`
- `class_metrics.json`
- `export_manifest.json`
- Ultralytics run outputs (checkpoints, logs)
- exported ONNX path recorded in `export_manifest.json`

## 7) Backend integration check

Copy exported ONNX to:

`backend/app/model/yolo11n_ewaste.onnx`

Then run backend smoke checks in your backend environment.

## 8) Common issues

- `ModuleNotFoundError: torch`:
  - install requirements in the activated conda env.
- Slow training over SMB:
  - optionally copy dataset to local Mac disk for the run, then copy artifacts back.
- `telephone` missing in `valid/test` warning:
  - warning is expected with current dataset; add coverage later if you keep this class.

## 9) Round 1 Real Training (Mac M4)

Use this profile for your first non-smoke run:

`model_pipeline/config/detection_local_train.round1_real.yaml`

```bash
python model_pipeline/scripts/train_detection_local.py \
  --config model_pipeline/config/detection_local_train.round1_real.yaml
```

Recommended expectations:
- Duration: significantly longer than smoke run.
- Outputs under:
  - `model_pipeline/artifacts/detection_training/round1_real/`
- Key files to review after completion:
  - `metrics_summary.json`
  - `class_metrics.json`
  - `export_manifest.json`

Quick post-run check:

```bash
python - <<'PY'
import json
from pathlib import Path
run = Path("model_pipeline/artifacts/detection_training/round1_real")
summary = json.loads((run / "metrics_summary.json").read_text(encoding="utf-8"))
print("SUMMARY:", summary)
cm = json.loads((run / "class_metrics.json").read_text(encoding="utf-8"))
cm = sorted(cm, key=lambda x: x.get("mAP50_95", 0))
print("WEAKEST 5:")
for row in cm[:5]:
    print("-", row["class_name"], row["mAP50_95"])
PY
```

## 10) Reset backend confidence after debugging

After low-threshold debugging, return to safer default (`0.40`) before normal API tests:

PowerShell:

```powershell
Remove-Item Env:MIN_CONFIDENCE -ErrorAction SilentlyContinue
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

CMD:

```cmd
set MIN_CONFIDENCE=
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 11) Round 2 Heavy Training (Mac M4)

Use this heavier profile after `round1_real` completes:

`model_pipeline/config/detection_local_train.round2_heavy.yaml`

```bash
python model_pipeline/scripts/train_detection_local.py \
  --config model_pipeline/config/detection_local_train.round2_heavy.yaml
```

Target profile values:
- `epochs: 120`
- `imgsz: 960`
- `batch: 8`
- `patience: 30`
- `workers: 4`
- `cache: true`

Monitoring guidance:
- Check RAM usage trend by epochs 5-10.
- If stable and below your limit, next run can try `batch: 10`.
- Compare this run against `round1_real` using:
  - overall `mAP50-95`
  - weakest 5 class metrics
  - false positives on negative images

## 12) Round 3 Recall Recovery (Mac M4)

Use this profile to recover recall and weak classes after a precision-heavy run.

`model_pipeline/config/detection_local_train.round3_recall_recovery.yaml`

```bash
python model_pipeline/scripts/train_detection_local.py \
  --config model_pipeline/config/detection_local_train.round3_recall_recovery.yaml
```

Target profile values:
- `epochs: 110`
- `imgsz: 768`
- `batch: 8`
- `patience: 25`
- `workers: 4`
- `cache: true`

Post-run comparison targets:
- improve `recall_B` vs `round2_heavy_960`
- recover weak classes (`mouse`, `capacitor`, `usb_stick`)
- keep overall `mAP50-95` close to or above `round1_real`
