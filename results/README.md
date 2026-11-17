# Results

Start with [`m4/summary/goals.csv`](m4/summary/goals.csv). It lists the Milestone 4 goals for the model we ship (A3): each goal's number, its mark, pass or fail, and the file the number comes from. [Notebook 25](../notebooks/25_m4_summary.ipynb) writes every table in `m4/summary/` from the files below and checks each number against its source, so it can be rerun on a laptop without a GPU.

## Where each headline number lives

| Question | Summary table | Read from |
|---|---|---|
| Did the shipped model meet goals G1 to G5 and the safety rule? | `m4/summary/goals.csv` | `m4/full/A3/`, `m4/gate/non_snake_test.json`, `m4/video/A3/video_lock.csv`, `m4/arms/A0_check/` (the M3 numbers) |
| How do the three full runs compare, and which one ships? | `m4/summary/full_runs.csv` | `m4/full/A1/`, `m4/full/H2/`, `m4/full/A3/` |
| How do we compare with B1 on its 62 species? | `m4/summary/b1.csv` | `m4/full/*/b1_62.json` |
| How many species reach each F1 level? | `m4/summary/classes.csv` | `m4/full/*/per_class_counts.json`, `m4/full/*/final_metrics.json` |
| Which background arm won on the subset? | `m4/summary/arms.csv` | `m4/arms/arms.csv`, `m4/arms/pick.json` |
| How often does the front end call a non-snake a snake? | `m4/summary/gates.csv`, `gate_categories.csv`, `gate_snake_groups.csv`, `gate_checks.csv` | `m4/gate/` |
| Did head crops help? | `m4/summary/heads.csv`, `head_pilot.csv` | `m4/heads/` |
| Does the video lock steady the answer? | `m4/summary/video.csv`, `video_clips.csv`, `video_tuning.csv`, `video_negatives.csv`, `front_end_shift.csv` | `m4/video/A3/`, `m4/video/M3/` |

## Folders

- `m4/arms/`: the background arms on the experiment subset, one folder per arm (`M4_A1` to `M4_A3`), the re-scored Milestone 3 model (`A0_check`), the pick (`arms.csv`, `pick.json`), the app's time per frame (`app_cost.json`), and each arm's run of notebook 18 in `executed/`.
- `m4/full/`: the three full-training runs (`A1`, `H2`, `A3`), each with its config, training curve, metrics and a `manifest.json` that lists the SHA-256 of everything kept outside the repo, and each run of notebook 21 in `executed/`.
- `m4/gate/`: the no-snake gate, before and after the detector and U-Net were retrained with non-snake photos, and the one scoring of the non-snake test split.
- `m4/heads/`: the head and tail pilot (`pilot.csv`, `pilot_retry.csv`) and the detail-crop arm (`M4_H2`).
- `m4/video/`: the clip list, the lock's tuning and results on the Milestone 3 model (`M3/`) and on the shipped model (`A3/`), and the Milestone 3 run of notebook 24 in `executed/`.
- `m3/`: only the Milestone 3 files a notebook reads: the run tables in `summary/`, the shipped decision file `final/H1_top1_s1/decision_metrics.json`, `q1/`, `x1_laptop/`, `k1/`, `h1/`, `demo/`, the final model's recipe and training curve (`retrain/H1_top1_s1/config.yaml`, `metrics.jsonl`), which every Milestone 4 arm starts from, and the scenery test of the Milestone 3 final model, `r1/H1_top1_s1/robustness_metrics.json`, which notebooks 18 and 21 compare against.

## Run codes

| Code | Meaning |
|---|---|
| G1 to G5 | The Milestone 4 goals: scenery (G1), species reaching F1 0.5 (G2), against B1 (G3), false snakes (G4), video (G5) |
| B1 | The published baseline: Bolon et al. 2022, a ViT trained on SnakeCLEF 2021 data, with its labels mapped to ours; it knows 62 of our 184 species |
| H1 | Milestone 3's tuning study. `H1_top1_s1` is the Milestone 3 final model (the best H1 setting, seed 1) |
| R1 | The scenery test: top-1 on photos as they are, with the background blanked, and with the background swapped for another photo's |
| A0 | The Milestone 3 final model, re-scored, not retrained: the arm every other arm must beat |
| A1 | Box crop, with the scenery swapped half the time in training (front end F1, swap masks from SAM) |
| A2 | Snake only: everything outside the mask filled with one colour (front end F3) |
| A3 | Crop to the mask's box, with the scenery swapped half the time (front end F1m, masks from our U-Net). This is the model we ship |
| H2 | A1 plus the detail crop, a second, closer look at the region the model picks itself. `M4_H2` is its subset arm; `full/H2` is its full run. Not related to H1 |
| F1, F1m, F3 | Front ends: the detector's box crop (F1), the crop to the predicted mask's box (F1m), the snake alone (F3) |
| M4_FULL | The `run_id` every full run carries; the folder name (`A1`, `H2`, `A3`) says which recipe it was |
| box, mask, both | No-snake gate rules: the detector's box confidence, the U-Net's mask area, or both together. The app uses "both" (`m4/gate/gate.json`) |

## Kept outside the repo

- Photos, video clips, model weights and saved model outputs. They are private Kaggle datasets and run outputs, because of their licences and size. Each full run's `manifest.json` lists their SHA-256, size and row counts.
- `class_mix.json`, the sampler's per-class draw counts for each run. It is kept in the runs' Kaggle outputs, and the full runs' manifests list its SHA-256.
- Milestone 3's full run records (432 files: every training, tuning, front-end, safety and app run, with their executed notebooks). They are in `m3-run-records.tar.gz` (2.8 MB, SHA-256 `e7a926ecb9b167ab4f8249e23dd9b5a06e7d7eae85ed90992ba1f807a2628003`), attached to the [`m3-run-records` release](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/releases/tag/m3-run-records). The Milestone 3 report's tables come from `m3/summary/`, which stays here.
