# Label Mapping Usage (Roboflow / Local)

## Files
- `model_pipeline/roboflow_label_mapping.csv`
- `model_pipeline/roboflow_label_mapping.json`
- `model_pipeline/scripts/apply_label_mapping.py`

## How to apply
1. Normalize incoming labels to lowercase and trim spaces.
2. If label exists in `map`, replace with the mapped target class.
3. If label exists in `ignore`, drop annotation from training targets.
4. If label exists in `review` (or unknown), send to manual review queue.
5. Keep only final target classes:
   - `celular`, `notebook`, `tablet`, `televisao`, `carregador`, `impressora`
   - `bateria`, `placa_eletronica`, `capacitor`, `cabo`

## COCO automation script
Use this for Roboflow/COCO exports:

```bash
python model_pipeline/scripts/apply_label_mapping.py \
  --input-coco data/annotations.json \
  --mapping-json model_pipeline/roboflow_label_mapping.json \
  --output-coco data/annotations_mapped.json \
  --report-json data/mapping_report.json
```

Optional override for unknown labels:

```bash
python model_pipeline/scripts/apply_label_mapping.py \
  --input-coco data/annotations.json \
  --mapping-json model_pipeline/roboflow_label_mapping.json \
  --output-coco data/annotations_mapped.json \
  --report-json data/mapping_report.json \
  --unknown-action ignore
```

## Recommended QA checks
- Verify no part-level classes remain in final training labels.
- Generate class count report after mapping.
- Spot-check at least 50 reviewed samples before train split freeze.
