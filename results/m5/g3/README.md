# G3: matching the published model on its own species

**The goal.** G3 asks our classifier to tie or beat B1 (Bolon et al. 2022) on B1's 62 species, on both top-1 and
macro-F1. It is scored on the G3 slice: photos uploaded from 2022 onwards, which B1 cannot have trained on. A
difference counts as tied when its 95% interval (1,000 resamples of whole photographers) contains 0. In M4, A3 lost
on top-1 by 0.081.

**The answer, on validation.** One candidate passes: M5_KD with flip and the prior. It scores top-1
**-0.012 (-0.030 to +0.002)** and macro-F1 **+0.002 (-0.014 to +0.025)** against B1. Both intervals contain 0, so
both count as tied. The model is the same size as A3 (20.8 MB). Flip runs it twice per photo, and the prior is a
fixed bias of 184 numbers.

Three things have to be read with that pass:

- **It costs the rarer species.** Validation macro-F1 over all 158 measurable species is 0.450, against A3's 0.480.
  G2 is 81 of 158, against A3's 87 and a mark of 80. The prior moves answers toward common species, and that is
  what closes the gap on B1's 62.
- **The idea of a prior came after looking at validation.** Tau is fitted on calibration photos only. But we chose
  to try a prior after probing A3's validation G3 at a few tau values (see "The prior" below). The validation pass
  is therefore optimistic. The locked test later confirmed it as a tie (report, section 4.1).
- **Without the prior, nothing passes.** The best candidate without it is KD with flip, at top-1 -0.056.

Everything in this folder is on validation or calibration photos. No test photo was read here.

## What we trained

Both runs start from A3's `best.pt` and change one thing at a time.

- **M5_FT.** A3 trained on M3 crops, but the app feeds it crops from the M4 front end. On the app's crops A3's G3
  top-1 gap was -0.087, against -0.081 on its own crops. So we fine-tuned A3 for 5 epochs on M4 front-end crops: lr
  2e-4 (the encoder at 0.1x), 1 warm-up epoch then cosine, evaluation every epoch. Everything else is A3's
  `config.yaml`: scenery swap 0.5, weighted sampler, predicted masks, crop to the mask's box.
- **M5_KD.** The same run plus a distillation term (weight 1, temperature 2) on the 37,847 training photos of B1's
  62 species. The teacher is B1's mapped probabilities on those photos. B1 knows only 62 species, so its
  probability is 0 on our other 122; the loss therefore also pushes the student away from them on these photos.
  The code is `models.distillation_loss`, tested in `tests/test_distillation.py`.

Each kernel first ran the M4 front end from scratch on all 75,383 photos it reads: 49,782 training, 7,999
calibration, 17,151 validation and 451 unknown. 99.6% got a box and 86 to 91% a mask. It then ran a 300-photo smoke
pass of every step, then the full run. After training it saved calibration, validation and unknown-species outputs,
the same again with every photo mirrored, the decision fit, the G1 scenery test, the G2 counts, the B1 comparison,
and A3 itself on the same M4 crops as the "before".

### Training curves (validation loss is now logged)

M5_FT:

| Epoch | Val macro-F1 | Val top-1 | Train loss | Val loss | Train species loss | Val species loss | lr at end |
|---|---|---|---|---|---|---|---|
| 1 | 0.4769 | 0.6583 | 5.429 | 6.665 | 5.017 | 6.135 | 2.0e-4 |
| 2 | 0.4755 | 0.6591 | 5.430 | 6.641 | 5.016 | 6.115 | 1.7e-4 |
| 3 | 0.4744 | 0.6619 | 5.360 | 6.600 | 4.955 | 6.077 | 1.0e-4 |
| **4 (best)** | **0.4770** | **0.6649** | 5.258 | 6.558 | 4.863 | 6.043 | 2.9e-5 |
| 5 | 0.4715 | 0.6668 | 5.196 | 6.563 | 4.806 | 6.043 | 0 |

Validation loss falls up to epoch 4 and is flat at epoch 5, so we see no overfitting in 5 epochs. As a control we
shuffled the labels on the saved outputs: macro-F1 0.006, under the 0.02 it must stay below.

| On the same M4 crops (all 17,151 validation photos) | Top-1 | Macro-F1 (158 species) |
|---|---|---|
| A3 (before) | 0.6611 | 0.4783 |
| M5_FT | 0.6649 | 0.4770 |
| M5_FT + flip | 0.6706 | 0.4843 |

On its own the fine-tune is not clearly better than A3. It gains 0.4 points of top-1 and gives back 0.1 of macro-F1.
For reference, A3 on its own M3 crops scores top-1 0.6644 and macro-F1 0.4800.

M5_KD:

| Epoch | Val macro-F1 | Val top-1 | Train loss | Val loss | Train species / kd | Val species / kd | lr at end |
|---|---|---|---|---|---|---|---|
| **1 (best)** | **0.4784** | **0.6746** | 10.406 | 9.797 | 5.141 / 4.850 | 6.131 / 3.147 | 2.0e-4 |
| 2 | 0.4543 | 0.6960 | 9.104 | 9.236 | 5.451 / 3.230 | 6.102 / 2.624 | 1.7e-4 |
| 3 | 0.4387 | 0.6923 | 8.703 | 9.022 | 5.445 / 2.845 | 6.092 / 2.426 | 1.0e-4 |
| 4 | 0.4536 | 0.7020 | 8.459 | 8.945 | 5.357 / 2.702 | 6.047 / 2.397 | 2.9e-5 |
| 5 | 0.4495 | 0.7025 | 8.377 | 8.931 | 5.308 / 2.673 | 6.033 / 2.391 | 0 |

Distillation pulls the model toward B1's species. Top-1 climbs to 0.70 while macro-F1 over all species falls. The
run keeps the epoch with the best macro-F1, which is epoch 1, so it is the only epoch scored for G3. Epochs 2 to 5
survive only as `last.pt` (epoch 5), and their outputs were not saved. The validation kd term uses B1's saved
validation probabilities and is logged only.

## G3 for each candidate

`g3_candidates.py` scores every candidate with the M4 G3 code unchanged: the same 11,931-photo slice, paired
difference, shared macro-F1 and bootstrap (1,000 photographer resamples, seed 42). Two checks come first. The
instrument check reproduces A3's -0.0808 / -0.0249 exactly. The negative control shuffles FT's predictions across
photos and gives top-1 -0.758 (-0.776 to -0.740) and macro-F1 -0.704, so the test can fail.

| Candidate | Top-1, ours minus B1 | Macro-F1, ours minus B1 | G3 | tau | Val top-1 (all) | Val macro-F1 (158) |
|---|---|---|---|---|---|---|
| A3 shipped, M3 crops (instrument) | -0.081 (-0.100 to -0.064) | -0.025 (-0.043 to -0.001) | fail | | 0.664 | 0.480 |
| A3 on M4 crops (before) | -0.083 (-0.103 to -0.066) | -0.025 (-0.042 to +0.002) | fail | | 0.661 | 0.478 |
| A3 on M4 crops + prior | -0.029 (-0.047 to -0.013) | -0.015 (-0.034 to +0.009) | fail | 0.50 | 0.700 | 0.463 |
| M5_FT | -0.078 (-0.097 to -0.061) | -0.027 (-0.044 to -0.002) | fail | | 0.665 | 0.477 |
| M5_FT + flip | -0.073 (-0.092 to -0.056) | -0.018 (-0.035 to +0.009) | fail | | 0.671 | 0.484 |
| M5_FT + prior | -0.036 (-0.053 to -0.021) | -0.020 (-0.037 to +0.004) | fail | 0.15 | 0.696 | 0.469 |
| **M5_FT + flip + prior** | **-0.026 (-0.043 to -0.011)** | **-0.009 (-0.026 to +0.017)** | **fail (top-1)** | 0.25 | 0.705 | 0.476 |
| M5_KD | -0.061 (-0.078 to -0.045) | -0.018 (-0.036 to +0.005) | fail | | 0.675 | 0.478 |
| M5_KD + flip | -0.056 (-0.074 to -0.040) | -0.011 (-0.028 to +0.012) | fail | | 0.679 | 0.482 |
| M5_KD + prior | -0.017 (-0.035 to -0.003) | -0.001 (-0.018 to +0.022) | fail (top-1) | 0.40 | 0.705 | 0.460 |
| **M5_KD + flip + prior** | **-0.012 (-0.030 to +0.002)** | **+0.002 (-0.014 to +0.025)** | **pass (both tied)** | 0.50 | 0.709 | 0.450 |
| FT + KD average (two models) | -0.063 (-0.081 to -0.045) | -0.016 (-0.033 to +0.009) | fail | | 0.676 | 0.483 |
| FT + KD average + flip (two models) | -0.060 (-0.079 to -0.042) | -0.011 (-0.027 to +0.012) | fail | | 0.680 | 0.487 |

Distillation is what moved the number. On the same recipe it adds about 1.5 to 2 points of top-1 on the slice.
The plain fine-tune fails in every variant. Averaging two models does not pass and would double the phone's
classifier cost, so we report it only.

As a diagnostic, not a candidate (G3 does not allow restricting the answer), we also made each model answer only
among B1's 62 species. M5_FT then gives top-1 -0.028 and macro-F1 -0.020, M5_KD -0.021 and -0.011. Even with the
answer restricted to B1's species, both stay behind B1.

### The prior

The weighted sampler flattens species frequencies in training, but validation photos, and real use, follow
natural frequencies. The prior puts them back: `logit_c + tau x log(n_c)`, where `n_c` is species c's photo count
in the training list before the caps. Tau is fitted per model on **calibration photos only**. The grid runs from 0
to 3 in steps of 0.05, and we pick the tau that maximises the mean of calibration top-1 and calibration macro-F1
(`prior.py`). The model stays phone-sized.

We found this idea by probing A3's saved outputs (`prior_probe.py`). That probe looked at A3's validation G3 at a
few tau values before we fixed the calibration rule. Tau itself never touches validation, but the decision to try
a prior was made after seeing validation, and it should be read that way.

### On the app's own crops

`g3_app_new.py` exports each `best.pt` to the app's fp32 ONNX classifier (ONNX against PyTorch: 2e-5). Exporting A3
the same way reproduces the shipped `classifier.onnx` byte for byte. It then runs the app's own detector, U-Net,
gate and crop on 11,924 of the 11,931 photos (7 have since been removed from iNaturalist). The A3 rows reproduce
our earlier app-path probe of A3 (20 November) exactly.

| On the app's crops | Top-1, ours minus B1 | Macro-F1, ours minus B1 |
|---|---|---|
| A3 | -0.087 (-0.109 to -0.069) | -0.026 (-0.045 to -0.000) |
| A3 + prior (tau 0.5) | -0.028 (-0.048 to -0.013) | -0.015 (-0.035 to +0.009) |
| M5_FT | -0.082 (-0.102 to -0.064) | -0.029 (-0.048 to -0.003) |
| M5_FT + flip | -0.073 (-0.093 to -0.055) | -0.018 (-0.037 to +0.007) |
| M5_FT + prior | -0.036 (-0.054 to -0.021) | -0.021 (-0.039 to +0.005) |
| **M5_FT + flip + prior** | **-0.025 (-0.044 to -0.011)** | **-0.007 (-0.026 to +0.018)** |
| M5_KD | -0.062 (-0.080 to -0.045) | -0.019 (-0.038 to +0.003) |
| M5_KD + flip | -0.056 (-0.075 to -0.039) | -0.012 (-0.032 to +0.012) |
| M5_KD + prior (tau 0.4) | -0.018 (-0.035 to -0.003) | -0.001 (-0.020 to +0.023) |
| **M5_KD + flip + prior (tau 0.5)** | **-0.012 (-0.030 to +0.003)** | **+0.001 (-0.017 to +0.024)**, tied on both: passes |

This is the gap the fine-tune was meant to close. On the app's crops the M4-crop models lose almost nothing against
their training-path crops: M5_KD -0.062 against -0.061, where A3 loses -0.087 against -0.081.

## Safety thresholds for the new classifier

A new classifier needs new decision thresholds. `refit_new.py` fits them on calibration outputs and raises the
target in steps of 0.0025 until the WHO recall lower bound **behind the app's gate** reaches 0.95. "Behind the
gate" uses the app's own M4 gate call for each photo; a WHO-listed snake the gate drops counts as a miss. The
calibration outputs come from the kernel (M4 crops through PyTorch), not from the ONNX app path. Two checks: the
default fit reproduces the kernel's own thresholds exactly (difference 0), and the refit on shuffled logits gives a
bound of 0.31 to 0.32, so it can fail.

| Model | Bound behind gate, kernel fit | Refit target | **Bound behind gate, refit** | Val WHO recall behind gate | False alarms | "We don't know" |
|---|---|---|---|---|---|---|
| M5_FT | 0.942 | 0.9600 | **0.956** | 0.966 (0.955 to 0.976) | 24.3% | 2,345 |
| M5_FT + prior | 0.942 | 0.9575 | **0.956** | 0.967 (0.956 to 0.977) | 24.4% | 2,341 |
| M5_FT + flip + prior | 0.942 | 0.9575 | **0.954** | 0.969 (0.957 to 0.979) | 24.4% | 2,224 |
| M5_KD | 0.941 | 0.9550 | **0.955** | 0.961 (0.946 to 0.972) | 22.0% | 2,321 |
| M5_KD + prior | 0.942 | 0.9550 | **0.955** | 0.958 (0.944 to 0.969) | 22.0% | 2,046 |
| **M5_KD + flip + prior** (frozen) | 0.944 | 0.9550 | **0.954** | **0.963 (0.950 to 0.973)** | 21.9% | 1,898 |

For comparison, A3's app refit in M4 gave a bound of 0.954, validation 0.965 (0.953 to 0.977) and 24.9% false
alarms. Each new model's own kernel-fit thresholds fall short behind the gate (0.941 to 0.944), as A3's did. So any
of them needs the refitted thresholds in `pipeline.json`. In every refit the WHO-head threshold comes out at
infinity: the WHO head never fires alone. The "dangerous" calls come from WHO-listed mass, the top species and
venomous-family mass. For the frozen candidate, the lower end of the validation interval sits exactly on 0.950.

## G1 and G2

Both are measured on the plain model. Flip and the prior act after the crop and were not put through the scenery
test.

- **G1** (scenery gap on photos with a mask, mark 0.084). M5_FT: 0.692 as is, 0.640 swapped, gap **0.053** (A3
  0.052). M5_KD: 0.700 and 0.645, gap **0.055**. Both pass.
- **G2** (species at F1 0.5 or more, mark 80 of 158). M5_FT **85** (with flip 86, prior 85, flip + prior 85). M5_KD
  **88** (with flip 87, prior 86, **flip + prior 81**). All pass; the frozen candidate passes by one species. A3 had
  87.

## The distillation pilot

We chose the full KD run's settings (weight 1, temperature 2, lr 2e-4) without a pilot. When the team asked for
one, the full run was already about an hour in, and the Kaggle account allows only two GPU sessions. So the pilot
ran as soon as the fine-tune freed a slot, to justify or replace that choice.

The pilot used the 16% experiment subset: 13,611 training photos after the caps, on M4 crops. Each setting started
from A3's `best.pt` and ran 2 epochs at lr 2e-4 (1 warm-up epoch, then cosine). Every setting was evaluated on all
17,151 validation photos, and the best epoch was kept by macro-F1. B1 ran on the subset's 10,542 training photos of
its 62 species (instrument check: maximum difference 0.0). The G3 slice, metric and bootstrap are the same as
above. The shuffled control fails in every row (top-1 about -0.75).

| Setting | Best epoch | Val macro-F1 | Val top-1 | G3 top-1, ours minus B1 | G3 macro-F1, ours minus B1 |
|---|---|---|---|---|---|
| no kd (A3's losses) | 1 | 0.486 | 0.674 | -0.067 (-0.085 to -0.051) | -0.017 (-0.033 to +0.008) |
| **kd weight 1, T 2 (the full run)** | 1 | **0.490** | **0.706** | **-0.024 (-0.041 to -0.009)** | **-0.000 (-0.017 to +0.023)** |
| kd weight 3, T 4 | 1 | 0.398 | 0.687 | -0.028 (-0.044 to -0.015) | -0.019 (-0.036 to +0.002) |

Both criteria, the G3 score and validation macro-F1, pick the setting the full run already used. The heavier,
softer setting costs 9 points of macro-F1. We did not vary the learning rate (three settings at most fit the
budget). Validation losses are not comparable across rows, because they include the kd term. Every setting peaked
at epoch 1 of 2. Per-epoch rows are in `out_pilot/pilot/pilot.json`.

## Runs

All GPU work ran on Jayesh's Kaggle account, on private kernels and datasets.

| Kernel | What | Session | Result |
|---|---|---|---|
| `sg-m5-finetune` v1, `sg-m5-kd` v1 | notebook 28 | about 20 min each | stopped by the smoke pass: a data file (`data/m3/medical_importance.csv`) was missing from the code bundle; fixed |
| `sg-m5-finetune` v2 | notebook 28, `VARIANT = "finetune"` | 1.7 h; 5 epochs of 8.6 min + 1.6 min evaluation | complete |
| `sg-m5-kd` v2 | notebook 28, `VARIANT = "kd"`, with B1 run on all 37,847 teacher photos first | 2.8 h; 5 epochs of 9.3 min + 1.7 min evaluation | complete |
| `sg-m5-kd-pilot` v1 | notebook 29 | 1.05 h | complete |

Inputs were the shared datasets `serpentguard-indian-snakes-clean`, `-m3-subset` and `-sam-masks`, earlier private
copies of the B1 predictions, the M3 run and the M4 front-end masks, and one new private dataset,
`serpentguard-m5-inputs`. That dataset holds the M4 detector and U-Net and A3's `best.pt`, and each file is
SHA-256 checked in the kernel against `results/m4/artifacts.md`. Before any GPU time, a laptop dry run on 24 local
photos exercised every new step: the front end from scratch, training from A3 with kd, the validation loss,
`last.pt`, resume, and the mirrored outputs.

## Files

| Path | What it is |
|---|---|
| `out_kd/M5_KD/` | The frozen model's run record: `config.yaml`, `metrics.jsonl` (the curve above), `final_metrics.json`, `decision_metrics.json`, `class_mix.json`, `versions.json`, `subset_hash.txt` |
| `out_kd/full/` | Its G1 (`robustness_metrics.json`), G2 (`per_class_counts.json`), B1 comparison (`b1_62.json`), the teacher summary (`teacher.json`) and `summary.json` |
| `out_finetune/M5_FT/`, `out_finetune/full/` | The same for M5_FT |
| `out_pilot/pilot/` | The pilot's per-setting metrics and `pilot.json` |
| `out_*/executed/` | Notebooks 28 (FT and KD) and 29 as they ran on Kaggle, with outputs |
| `g3_candidates.py`, `g3_candidates.json` | The candidate table |
| `g3_app_new.py`, `g3_app_A3.json`, `g3_app_FT.json`, `g3_app_KD.json` | The app-path table |
| `prior.py`, `prior_probe.py` | The prior fit, and the probe that suggested it |
| `refit_new.py`, `refit_new.json` | The safety refit; the frozen thresholds are its "KD flip + prior" row |
| `kaggle/` | How the kernels were built: `make_runner.py` bundles the repo's code with a notebook, each subfolder holds the kernel's metadata and parameters, and `build_nb28.py` and `build_nb29.py` write notebooks 28 and 29 |

The code changes behind these runs are in the repo itself: `src/serpentguard/{data,models,train}.py` (distillation,
validation loss, `last.pt` with optimizer, scheduler, scaler and RNG states, `resume=True`, `init_from`),
`scripts/b1_teacher.py` (B1 on a photo list, MD5-checked; it first rescores 64 validation photos whose B1
probabilities are saved, with a maximum difference of 0.0, while a swapped-colour-channel control differs by 0.96), and
`tests/test_distillation.py`.

Weights are outside git. Their SHA-256 values are: `M5_KD/best.pt` (epoch 1)
`1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f`, `M5_KD/last.pt` (epoch 5)
`db873a70cfcb21b439e62e61ba25a45fdc7f8758ba087e60bb9b8aaf790fc0c4`, `M5_FT/best.pt` (epoch 4)
`996fffd3def60596286f80e4ede7d896f275139d27e7a0638aa4d25b45f4f8f6`, `M5_FT/last.pt` (epoch 5)
`f55d738208c728d068cfd69cad1e4f3646466e0b72ed699f74802709846af973`.

## Not checked

- The new thresholds through the ONNX app path on calibration photos. The refit used PyTorch M4 crops with the
  app's gate calls.
- On-device latency. The architecture and size are unchanged (20.8 MB). The fp16 classifier is checked in
  `robustness/`.
- A higher input resolution (not tried).
