# Detection Sprint Plan (prod_datasets/detection/v1)

## Objective
Produce a YOLO-ready detection dataset and train/evaluate YOLO11n for production inference.

## Priority classes (phase-first)
- `celular`
- `bateria`
- `placa_eletronica`
- `capacitor`
- `televisao`
- `cabo`

Then expand:
- `notebook`
- `router`
- `usb_stick`
- `mouse`
- `player`
- `monitor`
- `impressora`

## Detection dataset build steps
1. Curate image subset from `prod_datasets/unified/v1/{train,valid,test}/images/*` emphasizing priority classes.
2. Annotate bounding boxes in Roboflow/CVAT with canonical class names.
3. Include explicit hard negatives in valid/test.
4. For incremental updates, audit imported labels first and normalize leftover polygon rows to boxes if present.
5. Freeze upload set as `prod_datasets/detection/v1_1`, run an optional structural snapshot merge, then re-merge after label sync into final `prod_datasets/detection/v1_1_merged`.
6. Record class counts and split ratios in release metadata.

## Quality gates
- Bounding box tightness spot-check across each class.
- Per-class minimum sample checks before training.
- Ambiguous screen objects handled by policy (`televisao` vs negative if uncertain).

## Training/eval gates
- Track per-class precision/recall/mAP.
- Track confusion matrix and false positives on negatives.
- Approve only if reliability is acceptable on priority classes and false positives are controlled.
