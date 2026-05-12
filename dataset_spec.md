# ReciclaMack Dataset Spec (Hybrid: Gadgets + Parts)

## 1) Scope and labeling objective
Train detection for **whole electronic gadgets and high-frequency electronic parts** in user photos.  
Primary goal: identify the object and return disposal guidance.

Use these classes:
- `celular`
- `notebook`
- `tablet`
- `televisao`
- `carregador`
- `impressora`
- `bateria`
- `placa_eletronica`
- `capacitor`
- `cabo`

## 2) Class definitions (inclusion / exclusion)

### `celular`
- Include: smartphones and feature phones, front/back views, broken screens if still identifiable as phone.
- Exclude: tablets, smartwatches, phone-only accessories (cases, cables, earbuds).

### `notebook`
- Include: laptops and ultrabooks (open or closed), broken chassis if still notebook-shaped.
- Exclude: desktop towers, separate keyboards, monitors, tablets with detachable keyboards unless clearly notebook form.

### `tablet`
- Include: slate-style tablets and e-readers with tablet-like body.
- Exclude: smartphones, notebook 2-in-1 when clearly in notebook configuration.

### `televisao`
- Include: TVs (flat or CRT) where object is clearly a television.
- Exclude: computer monitors when uncertain; prefer not labeling ambiguous screen-only objects.

### `carregador`
- Include: wall chargers/adapters, power bricks for notebooks, phone adapters.
- Exclude: loose cables without adapter head, power strips, extension cords.

### `impressora`
- Include: home/office printers, multifunction printer/scanner bodies.
- Exclude: standalone scanners if not clearly printer-capable.

### `bateria`
- Include: standalone batteries from electronics (Li-ion phone/laptop packs, small rechargeable units).
- Exclude: car batteries and non-electronic waste categories unless explicitly in project scope.

### `placa_eletronica`
- Include: visible PCBs/boards from electronics, with components or traces visible.
- Exclude: full devices where board is not visible as primary target.

### `capacitor`
- Include: standalone capacitor components clearly visible as object target.
- Exclude: capacitors too small/unclear in scene where class certainty is low.

### `cabo`
- Include: electronic cables and charging/data cables as primary object.
- Exclude: power strips, extension cords when not cable target.

## 3) Annotation rules
- Task type: object detection bounding boxes.
- Draw tight boxes around visible device extent.
- Minimum visible area for labeling: ~20% of object.
- If severe occlusion and class uncertain: do not label.
- Multiple objects in one image: label all valid targets.
- Partially out-of-frame objects: label only if class confidence is still clear.
- One object = one class label only.
- Multi-instance requirement: if several valid targets are visible, annotate all of them (do not keep only one dominant object).
- Skip-note rubric for omitted instances in review logs:
  - `tiny_unreliable` = target appears too small to label confidently.
  - `ambiguous_occlusion` = heavy occlusion blocks class certainty.
  - `class_uncertain` = visible object present but class cannot be assigned reliably.
  - `out_of_scope_object` = visible object intentionally out of taxonomy.

## 4) Quality gates for dataset curation
- Remove blurred/low-light images where class is ambiguous.
- Remove near-duplicate frames from the same burst.
- Keep class balance as even as possible (target max 1.5x difference between largest and smallest class).
- Include realistic contexts: desk, classroom, home, recycling point.
- Include device condition diversity: intact, used, slightly damaged.

## 5) Hard negatives and ambiguity policy
- Add background images with no target object (tables, rooms, bags) for robustness.
- Add confusing objects:
  - monitor vs `televisao`
  - power strip vs `carregador`
  - phone case vs `celular`
- Ambiguous screen object: skip label unless there is strong class evidence.

## 6) Split policy (train/valid/test)
- Recommended split: 70/20/10.
- Stratify by class and scene type.
- Keep similar photos from same burst/device in the same split only (avoid leakage).
- Maintain a small “real-world holdout” set not used for tuning.

## 7) Minimum dataset targets (MVP baseline)
- Aim for at least 500 labeled instances per class before first production candidate.
- Prefer 800-1500 per class for better robustness in uncontrolled photos.

## 8) Label normalization
- Normalize source labels into the target taxonomy above.
- Example mapping:
  - `smartphone`, `mobile phone` -> `celular`
  - `laptop` -> `notebook`
  - `tv`, `television`, `crt tv` -> `televisao`
  - `charger`, `adapter`, `power brick` -> `carregador`
  - `printer`, `all-in-one printer` -> `impressora`
  - `battery`, `li-ion battery` -> `bateria`
  - `pcb`, `circuit board` -> `placa_eletronica`
  - `capacitor` -> `capacitor`
  - `cable`, `wire` -> `cabo`

## 9) Dataset versioning requirements
For each dataset release, record:
- `dataset_version`
- source datasets and licenses
- class counts (images + instances)
- exclusion criteria applied
- known blind spots and confusion risks
- annotation policy block:
  - `dataset_policy_version`
  - `annotation_policy.instance_mode`
  - `annotation_policy.unlabeled_object_policy`
  - `qc.missing_instance_policy`

## 10) V2 acceptance gate (multi-instance behavior)
- Run detection QC with warn-level missing-instance policy and export review queue CSV.
- Manually audit a sampled set of product-shot/dense-scene images from review queue.
- Confirm sampled images respect: "all visible valid targets are boxed" unless a documented skip-note code is present.
