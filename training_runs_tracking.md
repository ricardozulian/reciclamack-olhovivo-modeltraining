# Training Run Tracking

Runtime note: `duration_min_est` is estimated from run artifact timestamps (`args.yaml` creation to `weights/best.pt` last write).
Dataset size note: image counts come from the dataset split folders used by the run (`train/images`, `valid/images`, `test/images`).

| run_name | started_at (local) | finished_at (local) | duration_min_est | train_images | valid_images | test_images | device | epochs | imgsz | batch | precision_B | recall_B | mAP50_B | mAP50_95_B |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `round1_real` | 2026-04-22 05:12:17 | 2026-04-22 09:05:13 | 232.9 | 1819 | 587 | 316 | `mps` | 80 | 640 | 8 | 0.6711 | 0.7388 | 0.7264 | 0.5041 |
| `round2_heavy_960` | 2026-04-22 09:52:21 | 2026-04-22 20:42:51 | 650.5 | 1819 | 587 | 316 | `mps` | 120 | 960 | 8 | 0.7189 | 0.6077 | 0.6689 | 0.4531 |
| `round3_recall_recovery_768` | 2026-04-23 03:38:07 | 2026-04-23 11:02:46 | 444.7 | 1819 | 587 | 316 | `mps` | 110 | 768 | 8 | 0.6368 | 0.7038 | 0.6732 | 0.4722 |

## Run-to-Run Deltas

Delta convention: `current - previous` for each metric.

| current_run | previous_run | delta_precision_B | delta_recall_B | delta_mAP50_B | delta_mAP50_95_B |
|---|---|---:|---:|---:|---:|
| `round2_heavy_960` | `round1_real` | +0.0477 | -0.1311 | -0.0575 | -0.0510 |
| `round3_recall_recovery_768` | `round2_heavy_960` | -0.0821 | +0.0961 | +0.0043 | +0.0191 |

## Weak-Class Watchlist (Latest Run)

Latest run: `round3_recall_recovery_768`  
Selection rule: 5 lowest `mAP50_95` classes.

| rank | class_name | mAP50_95 |
|---:|---|---:|
| 1 | `monitor` | 0.2203 |
| 2 | `capacitor` | 0.2207 |
| 3 | `mouse` | 0.2786 |
| 4 | `usb_stick` | 0.2981 |
| 5 | `player` | 0.3969 |
