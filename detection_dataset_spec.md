# Detection Dataset v1 Spec (`prod_datasets/detection/v1`)

This is the implementation-ready structure for the YOLO detection sprint.

## 1) Canonical class list (fixed order)

Use exactly this class order for YOLO `names`:

0. `celular`
1. `notebook`
2. `televisao`
3. `bateria`
4. `placa_eletronica`
5. `capacitor`
6. `cabo`
7. `router`
8. `usb_stick`

## 2) Required folder layout

```text
prod_datasets/detection/v1/
  train/
    images/
    labels/
  valid/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml
  README.md
  release_metadata.json
```

Rules:
- Each image in `<split>/images/` must have one label txt in `<split>/labels/` with the same stem.
- Images with no objects are allowed and should have an empty `.txt` file.
- Label format: `class_id x_center y_center width height` (normalized 0-1).

## 3) Split policy

- Target split ratio: `70/20/10`.
- Keep near-duplicates and same capture burst in the same split.
- Ensure hard negatives exist in `valid` and `test`.

## 4) Class sampling priorities

Priority set:
- `celular`, `bateria`, `placa_eletronica`, `capacitor`, `televisao`, `cabo`

Secondary (weaker support now):
- `notebook`, `router`, `usb_stick`

## 5) Minimum acceptance targets (dataset-level)

- No class should be missing in `valid` or `test`.
- For weak classes (`notebook`, `router`, `usb_stick`), explicitly flag support level in `release_metadata.json`.
- Track class instance counts and split counts before training freeze.
