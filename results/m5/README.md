# Milestone 5 results

Milestone 5 froze the whole pipeline, scored it once on the locked test split, and then looked at where it fails.
The report is [`documentation/milestone 5/project-report.md`](../../documentation/milestone%205/project-report.md).
This folder holds the files behind every number in it.

Start with two files:

- [`test/FREEZE.md`](test/FREEZE.md) fixes the models, thresholds and goals. We wrote it before any test photo
  was opened.
- [`test/RESULTS.md`](test/RESULTS.md) is what the one scoring pass produced.

## Where each headline number lives

| Question | File |
|---|---|
| Did the frozen pipeline meet each goal on test? | `test/test_scores.json` (`subsets/all/goals`), `test/RESULTS.md` |
| How good is each species, and what does it get confused with? | `test/per_species_heldout.csv`, `test/confusions_heldout.csv`, `test/analysis_heldout.json` |
| Which non-snakes did the snake check call a snake? | `test/false_snakes_nonsnake_test.csv`, `test/predictions_nonsnake_test.csv` |
| Was the scorer right before it saw test? | `test/val_dry_run.json` (validation rerun), `test/FREEZE.md` section 8 |
| Does the video answer stay steady (G5)? | `test/g5/results/g5_summary.csv`, `test/g5/results/negatives.csv`, `test/g5/control_results/` |
| How did we get G3 (the species classifier against B1)? | `g3/README.md`, `g3/g3_candidates.json`, `g3/refit_new.json` |
| How did we get G4 (the snake check)? | `g4/README.md`, `g4/results/variants/<variant>/g4_eval.json` |
| How was the video lock fixed and tuned (G5)? | `g5/README.md`, `g5/results/tuning.csv`, `g5/results/chosen_settings.json` |
| What happens to blurred, dark or covered photos? | `robustness/README.md`, `robustness/results/robustness.json`, `robustness_table.csv` |
| Where does the model look (Grad-CAM)? | `robustness/results/gradcam.json` |
| Can the models be made smaller (fp16, int8)? | `robustness/results/compression.json`, `robustness/results/latency.json`, `robustness/fp16_front_end/fp16_front_end.json` (detector and U-Net) |
| Are the *Ahaetulla* labels the model disagrees with right? | `ahaetulla_check/ahaetulla_check.json`, `ahaetulla_check/ahaetulla_check.csv` |
| Leakage, naive baselines and the report's derived counts | `report/data/report_numbers.json`, `report/data/leakage_detail.json` |

## Folders

The goal folders are named after the Milestone 4 goal each one worked on. G1 and G2 needed no new work, so they
have no folder of their own.

| Folder | What it holds |
|---|---|
| `test/` | The locked test: the freeze record, the scoring code exactly as it ran (`kernel/`), the kernel's outputs (`kernel_out/`), the scores, per-species and per-photo files, and the frozen video run (`g5/`) |
| `g3/` | The species classifier. Fine-tuning A3 on the app's crops (`out_finetune/`), distilling from B1 (`out_kd/`, the model we froze), the distillation pilot (`out_pilot/`), the prior correction, the safety refit, and the Kaggle notebooks with their outputs (`out_*/executed/`) |
| `g4/` | The snake check: how the non-snake photos were collected, the training kernel, the result of each variant, parity and latency |
| `g5/` | The video lock: the M4 bugs checked one by one, the retune on dev clips, the new test clips with their credits, the panning test |
| `robustness/` | Stress tests, Grad-CAM and int8/fp16 compression, all on the frozen pipeline with nothing changed |
| `report/` | The two laptop scripts that compute the report's derived numbers and the confusion figure, and the TA's M4 feedback |

## Checking the records

Before reading any output, we hashed the test scoring code. Anyone can check those hashes:

```bash
cd results/m5/test && grep -v '^#' FREEZE_scripts.sha256 | shasum -a 256 -c
```

All seven files should print `OK`. The robustness kernels have one hash file per push
(`robustness/kernel/PUSHED_*.sha256`). The robustness README says which push produced which result.

[`check_numbers.py`](check_numbers.py) reads only files in this folder. It checks the report's headline numbers
against them and stops at the first mismatch:

```bash
python results/m5/check_numbers.py
```

## What the scripts are, and what they are not

The scripts here are a record of what ran, not a pipeline you can rerun from a fresh clone. They ran from the
team's working folder on a laptop and on Kaggle. Many read private Kaggle datasets and import helpers that lived
there and not in this repo:

- `check_full_run.py`, the M4 run checker;
- the M4 app reference pipeline, `reference_pipeline.py`;
- the M4 threshold-refit helpers.

The frozen scoring scripts in `test/kernel/` and `robustness/kernel/` are kept byte for byte, so their hashes
still match. They name that working folder `m5-results/`, which is `results/m5/` here. Every other script has its
paths written relative to the repo.

## Not in git

Weights, photos and video clips are not in git. Neither are files larger than about 1 MB: the per-photo `*.jsonl` and
`*.csv` score dumps of each run, and the frozen video tables `test/g5/tables_frozen.json`. The team keeps them, together with the frozen weights package and
its `SHA256SUMS`, and can share them on request. FREEZE gives the SHA-256 of every model file, so a copy can be
checked against it.

| Kind | Where it would sit | Examples |
|---|---|---|
| Model weights and ONNX files | `g3/out_*/<run>/`, `g4/runs/<variant>/` | `M5_KD/best.pt`, `last.pt`, `snake_check_fp16.onnx` |
| Saved model outputs | `g3/out_*/<run>/` | `val_outputs.npz`, `calib_outputs_flip.npz` |
| Per-photo prediction and score files | `test/`, `test/kernel_out/out/`, `g3/`, `g4/runs/`, `robustness/kernel_out/` | `predictions_heldout.csv` (18,533 rows), `snake_check_scores.csv`, `g3_app_KD.jsonl` |
| Photos and clips | Kaggle dataset `serpentguard-indian-snakes-clean`; `g5/new_clips/` | the 18,533 held-out photos, the 22 new test clips |

## Codes used in these files

| Code | Meaning |
|---|---|
| A3 | The classifier shipped in M4. MobileViT-S with an ArcFace head, trained on crops to the U-Net mask's box, with the scenery swapped half the time |
| M5_FT | A3 fine-tuned for 5 epochs on the M4 front end's crops. Not used |
| M5_KD | M5_FT's recipe plus distillation from B1 on B1's 62 species. Epoch 1 is the frozen classifier |
| flip | The classifier is run on the crop and on its mirror image, and the two outputs are averaged |
| prior | `tau x log(training count)` added to each species' logit, with tau = 0.5 fitted on calibration photos only |
| B1 | The published baseline, Bolon et al. (2022), a ViT-L that can name 62 of our 184 species |
| G3 slice | Held-out photos uploaded from 2022 onwards, of B1's 62 species (12,203 photos, 1,781 photographers) |
| `distil_oi` | The snake check we froze: EfficientNet-B0, three classes (snake, lookalike, other), feature distillation from BioCLIP, Open Images negatives added |
| M4 gate | The M4 detector and U-Net rule ("both"), which says "no snake" when the box or the mask is too weak |
| held-out | `test` (5,144 photos, India, uploaded from 2022) plus `test_extra` (13,389), 18,533 photos in all |
