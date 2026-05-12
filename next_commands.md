# Immediate Next Commands

## 1) Create dedicated conda env (optional but recommended)

```bash
conda env create -f model_pipeline/conda/reciclamack-classifier-baseline.yml
conda activate reciclamack-classifier-baseline
```

## 2) Re-run baseline classifier (if needed)

```bash
python model_pipeline/scripts/train_classification_baseline.py \
  --dataset-root prod_datasets/unified/v1 \
  --manifests-root prod_datasets/unified/v1/manifests \
  --output-root model_pipeline/artifacts/classification_baseline \
  --image-size 48 \
  --seed 42
```

## 3) Generate release manifest

```bash
python model_pipeline/scripts/create_release_manifest.py \
  --dataset-root prod_datasets/unified/v1 \
  --manifests-root prod_datasets/unified/v1/manifests \
  --model-version classifier-baseline-20260416T190042Z \
  --dataset-version prod_datasets/unified/v1 \
  --output model_pipeline/artifacts/releases/classification_release_manifest.json \
  --extra-json model_pipeline/artifacts/classification_baseline/run_20260416T190042Z/release_metadata.json
```

## 4) Detection sprint prep

1. Build `prod_datasets/detection/v1` with structure in `model_pipeline/detection_dataset_spec.md`.
2. Use class order from `model_pipeline/detection_data_yaml.template.yaml`.
3. Validate structure and IDs:

```bash
python model_pipeline/scripts/validate_detection_dataset.py \
  --dataset-root prod_datasets/detection/v1 \
  --output-json prod_datasets/detection/v1/validation_report.json
```

## 5) Local YOLO training (Mac M4)

Run:

```bash
python model_pipeline/scripts/train_detection_local.py \
  --config model_pipeline/config/detection_local_train.round3_recall_recovery.yaml
```

Primary outputs to review:
- `best.pt`
- exported ONNX
- eval metrics and confusion outputs
