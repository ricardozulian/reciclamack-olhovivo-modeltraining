# Model Pipeline (YOLO11n -> ONNX)

## Objective
Fine-tune YOLO11n for prioritized e-waste gadget/device classes plus high-frequency part classes, and export ONNX optimized for CPU inference.

Reference taxonomy and annotation policy: `model_pipeline/dataset_spec.md`.

## Steps
1. Fetch the curated detection dataset from Kaggle into a local, gitignored folder:
   - `kaggle datasets download <kaggle-user>/reciclamack-olhovivo-detection-v1 -p data/kaggle --unzip`
   - Ensure the final layout is `data/kaggle/reciclamack-detection-v1/data.yaml`.
   - Use `model_pipeline/config/detection_local_train.kaggle.yaml` for local smoke tests against the Kaggle dataset.
2. Curate merged dataset from Kaggle/Roboflow with unified labels:
   - Device-level labels such as `celular`, `notebook`, `tablet`, `televisao`, `carregador`, `impressora`.
   - Part-level labels such as `bateria`, `placa_eletronica`, `capacitor`, `cabo`.
3. For each gadget class, maintain a knowledge mapping with typical internal materials and disposal guidance.
4. Train locally on MacBook (MPS) with fixed train/valid/test split.
5. Export best checkpoint to ONNX (`opset` compatible with onnxruntime 1.19+).
6. Validate in backend inference smoke test.
7. Publish artifact metadata:
   - `model_version`, `dataset_version`, `export_hash`, date.
8. Normalize labels with:
   - `python model_pipeline/scripts/apply_label_mapping.py --help`
9. Build unified traceable dataset with dedup/splits:
   - `python model_pipeline/scripts/build_unified_dataset.py --config model_pipeline/config/unified_dataset_config.json --output-root prod_datasets/unified/v1`
   - For v2 multi-instance policy:
     - `python model_pipeline/scripts/build_unified_dataset.py --config model_pipeline/config/unified_dataset_config_v2.json --output-root prod_datasets/unified/v2`
10. Train classification baseline (immediate tactical baseline):
   - `python model_pipeline/scripts/train_classification_baseline.py --dataset-root prod_datasets/unified/v1 --manifests-root prod_datasets/unified/v1/manifests --output-root model_pipeline/artifacts/classification_baseline`
11. Create release/freeze metadata:
   - `python model_pipeline/scripts/create_release_manifest.py --dataset-root prod_datasets/unified/v1 --manifests-root prod_datasets/unified/v1/manifests --model-version classifier-baseline-latest --dataset-version prod_datasets/unified/v1 --output model_pipeline/artifacts/releases/classification_release_manifest.json`
12. (Optional) Dedicated conda env (do not reuse `dado`):
   - `conda env create -f model_pipeline/conda/reciclamack-classifier-baseline.yml`
   - `conda activate reciclamack-classifier-baseline`
13. Bootstrap detection sprint dataset (image copy + empty labels + annotation seed):
   - `python model_pipeline/scripts/bootstrap_detection_dataset.py --source-root prod_datasets/unified/v1 --output-root prod_datasets/detection/v1 --include-secondary --seed 42`
14. Validate detection dataset structure before training:
   - `python model_pipeline/scripts/validate_detection_dataset.py --dataset-root prod_datasets/detection/v1 --output-json prod_datasets/detection/v1/validation_report.json`
   - Optional v2 QC review queue:
     - `python model_pipeline/scripts/validate_detection_dataset.py --dataset-root prod_datasets/detection/v1 --output-json prod_datasets/detection/v1/validation_report.json --output-review-queue prod_datasets/detection/v1/review_queue.csv --missing-instance-policy warn`

## Acceptance gate
- Report per-class precision/recall and confusion matrix.
- Approve model only when prioritized classes trend toward 95% practical accuracy.
