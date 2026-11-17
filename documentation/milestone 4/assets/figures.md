# Milestone 4 figures

Drawn by [`notebooks/27_m4_figures.ipynb`](../../../notebooks/27_m4_figures.ipynb) on a laptop's CPU from saved files;
no model was run on a photo to make them. Every photo is a validation photo under CC0 or CC BY, credited under its
panel and in [`sample_predictions_credits.csv`](sample_predictions_credits.csv).

## `learning_curves.png`

Training loss per head and in total (top, log scale) and validation macro-F1, top-1 and top-3 (bottom) for the three
full runs: A1 (box crop with the scenery swap), H2 (A1 with the detail crop) and A3 (crop to the mask's box with the
scenery swap, the model we ship). Validation ran every second epoch, so each run has 15 points, and the training loss
is the mean over that epoch's batches, logged on those epochs only. Validation loss was not logged. The total is the
weighted sum the optimiser minimises: species + 0.3 (genus + family) + 0.5 WHO. The dashed line and black dots mark
the epoch each run kept, the one with the highest validation macro-F1: A1 epoch 28 (0.481), H2 epoch 24 (0.437) and
A3 epoch 24 (0.481).

Source: `results/m4/full/{A1,H2,A3}/metrics.jsonl`; the kept epoch from each run's `final_metrics.json`.

## `reliability_A3.png`

A3's 17,151 validation photos binned by confidence (the top species' probability) into 15 equal bins. Each bar is
the share of the bin's photos whose top species is right, each dot sits at the bin's mean confidence, and the dashed
line is perfect calibration. Left: the softmax of the raw logits, expected calibration error (ECE) 0.142. Right: the
logits divided by the temperature T = 1.806, fitted on the 7,999 calibration photos, ECE 0.040. Before scaling the
model is overconfident: the 10,308 photos above 0.933 confidence average 0.995 but are right on 90.3% of them.

Source: `M4_FULL/val_outputs.npz` in the Kaggle output `jayeshbansal497/sg-m4-full-a3`; the temperature and both ECE
values from `results/m4/full/A3/decision_metrics.json`, which the notebook checks it reproduces.

## `sample_predictions_A3.png`

Eight validation photos whose top species A3 got right and eight it got wrong, drawn with `metrics.round_robin`
(species in turn, seed 42) from the 2,600 validation photos under CC0 or CC BY on iNaturalist's photo host. The red box
is the M3 detector's box; A3's input was the crop to the box around the M3 U-Net's mask, which is not drawn. Under each
photo: the true species and whether it is WHO-listed, A3's top 3 with temperature-scaled probabilities, and the verdict
with the reason the decision rules give. The verdict counts recomputed over all validation photos equal the run's (6,729 dangerous, 7,923 not on the WHO list, 2,499 we don't
know). Panels 1 to 16 in the credits table.

Source: `M4_FULL/val_outputs.npz`, `M4_FULL/prototypes.npz` and `full/frontend_preds.csv` in the Kaggle output
`jayeshbansal497/sg-m4-full-a3`; `results/m4/full/A3/decision_metrics.json`; the clean dataset's `metadata.csv` for
licences, credits and urls.

## `false_snakes_gate.png`

Four of the 164 non-snake validation photos (of 600) that the app's gate calls a snake. The gate's rule is "both": the
M4 detector's box confidence at least 0.0050 and the M4 segmenter's mask area at least 0.000048 of the photo. Each
photo is the one with the highest box confidence in its iNaturalist lookalike category (caecilian, legless lizard,
earthworm, eel); the red box is the M4 detector's most confident box. The 66 Open Images photos among the 164 (people,
hands, ropes and other categories) are not shown. Panels 17 to 20 in the credits table.

Source: `scores/M4_nonsnake_val.csv` in the Kaggle output `puneetbajajiitm/sg-m4-gate`; `results/m4/gate/gate.json`;
`manifest.csv` in the Kaggle output `puneetbajajiitm/sg-m4-non-snake`.
