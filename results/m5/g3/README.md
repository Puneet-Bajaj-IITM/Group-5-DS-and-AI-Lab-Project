# G3: fine-tuning A3 on the app's crops, and the try at G3

Night of 24 November 2026. Validation only: no test or test_extra photo was read. Calibration photos were used only
to fit thresholds and the prior-correction scalar (see below), never to train. No paid model API was called; all
GPU work ran on Jayesh's Kaggle account, and every dataset and kernel is private (anonymous `curl` gives 403).

**Result: G3 passes, narrowly, with one candidate, and only with a caveat that has to be read with it.** The
distilled model M5_KD, with flip TTA and a training-prior correction, scores top-1 **-0.012 (-0.030 to +0.002)**
and macro-F1 **+0.002 (-0.014 to +0.025)** against B1 on the G3 slice. Both intervals include 0, so both count as
tied and the rule is met. It is one phone-size model (same architecture and size as A3, 20.8 MB) plus a
184-number logit bias, and flip TTA runs the classifier twice per photo. The caveats:

- **It trades away macro-F1 on the other species.** Validation macro-F1 over all 158 measurable species is
  0.450, against A3's 0.480. G2 is 81 of 158 (goal 80; A3 87). The prior correction moves predictions toward
  common species, which is what closes the gap on B1's 62.
- **The prior correction was found on the night.** Its scalar tau is fitted on calibration only, but I chose to
  try it after probing A3's validation G3 (see "Prior" below). The pass is a validation result, it sits at the
  edge (top-1 upper end +0.002), and it should be confirmed on test by the coordinator before anyone relies on it.
- Without the prior correction no candidate passes. The best is KD + flip at top-1 -0.056.

The plain fine-tune on M4 crops (M5_FT) fails G3 in every variant (best: FT + flip + prior, top-1 -0.026,
-0.043 to -0.011). Distillation from B1 is what moved the number: on the same recipe it gains about 1.5 to 2
points of top-1 on the slice.

## What ran

| Kernel (Jayesh's account) | What | Time | Result |
|---|---|---|---|
| `jayeshbansal497/sg-m5-finetune` v1, `sg-m5-kd` v1 | notebook 28 | ~20 min each | the in-kernel smoke caught a file missing from the code bundle (`data/m3/medical_importance.csv`); fixed, rerun |
| `jayeshbansal497/sg-m5-finetune` v2 | notebook 28, `VARIANT = "finetune"` | 1.7 h session (02:31-04:14 IST); training 5 x 8.6 min + 1.6 min eval | complete |
| `jayeshbansal497/sg-m5-kd` v2 | notebook 28, `VARIANT = "kd"`; B1 on 37,847 training photos on GPU 1 (all done before the full pass) | 2.8 h session (02:31-06:20 IST); training 5 x 9.3 min + 1.7 min eval | complete |
| `jayeshbansal497/sg-m5-kd-pilot` v1 | notebook 29, the KD pilot | 1.05 h session (04:14-05:18 IST) | complete |

Inputs: the shared `serpentguard-indian-snakes-clean`, `-m3-subset`, `-sam-masks` (Puneet's, already shared with
Jayesh), Jayesh's existing private copies (`serpentguard-m3-b1-predictions`, `-m3-run-final`,
`-m4-frontend-masks`), and one new private dataset `jayeshbansal497/serpentguard-m5-inputs`: the M4 detector and
U-Net (from `puneetbajajiitm/sg-m4-gate`) and A3's `best.pt`; every file is SHA-256 checked in the kernel against
`results/m4/artifacts.md`.

Each kernel: inputs and checksums; the M4 front end with `front_end.extend_predictions` from an empty base
(header-only `frontend_preds.csv`, empty masks dir) over all 75,383 photos the run reads (49,782 training list,
7,999 calibration, 17,151 validation, 451 unknown; 99.6% boxed, 86-91% masked); A3's scenery bank rebuilt and its
fingerprint matched; a smoke pass (300 photos, 1 epoch, every step) before the full pass; then the full pass:
5 epochs from A3's `best.pt`, eval every epoch, lr 2e-4 (A3: 9.5e-4; encoder at 0.1x), 1 warm-up epoch then
cosine, everything else A3's `config.yaml` (scenery swap 0.5, weighted sampler, predicted masks, F1m crop).
After training: calibration / validation / unknown outputs, the same with every photo mirrored (flip TTA), the
decision fit, G1 scenery test, G2 counts, B1 comparison, and A3 itself on the same M4 crops (the "before").

## Code changed (uncommitted, in `m5-wt`)

- `src/serpentguard/train.py` (from before this night): `val_loss_*` every eval, `last.pt` with model, optimizer,
  scheduler, scaler, RNG; `train(..., resume=True)`; `init_from`. Plus tonight: `check_config` accepts `kd` when
  `teacher_probs` names an existing file; `target_names` passes the teacher targets to the loss in training and
  in the validation-loss pass.
- `src/serpentguard/models.py`: `distillation_loss` (KL from the teacher at temperature T, times T^2, only on
  photos with a teacher row; the teacher's probabilities are 0 outside its 62 species, so mass on our other 122
  species is penalised on those photos) and the `kd` term in `total_loss` (weight `kd_weight`).
- `src/serpentguard/data.py`: settings `teacher_probs`, `kd_weight`, `kd_temperature`; `read_teacher` /
  `run_teacher`; `SnakeDataset` returns `teacher`, `teacher_valid`; `eval_loader(..., with_teacher=)`.
- `scripts/b1_teacher.py`: B1 (ONNX, LINDAT, MD5 checked) on a photo list, fed and mapped exactly as notebook 11.
  It first re-scores 64 validation photos whose notebook-11 probabilities are saved (in the run: max difference
  0.0, top-1 agreement 1.0; the swapped-channel control differs by 0.96) and writes parts so a restart resumes.
- `tests/test_distillation.py` (8 tests; the temperature test failed on a wrong expectation first and was fixed).
  All 16 repo tests pass.
- `notebooks/28_m5_finetune.ipynb` (built by `results/m5/g3/build_nb28.py`) and
  `notebooks/29_m5_kd_pilot.ipynb` (`build_nb29.py`). Notebook 21 is untouched.
- Laptop dry run before any GPU time: `local_smoke/run.py` (M4 front end from an empty base, train from A3 with
  kd, val loss, `last.pt`, resume, mirrored outputs) on 24 local photos; `local_smoke/run.log`.

## The fine-tuned model (M5_FT, M4 crops)

| Epoch | Val macro-F1 | Val top-1 | Train loss | Val loss | Train species loss | Val species loss | lr at end |
|---|---|---|---|---|---|---|---|
| 1 | 0.4769 | 0.6583 | 5.429 | 6.665 | 5.017 | 6.135 | 2.0e-4 |
| 2 | 0.4755 | 0.6591 | 5.430 | 6.641 | 5.016 | 6.115 | 1.7e-4 |
| 3 | 0.4744 | 0.6619 | 5.360 | 6.600 | 4.955 | 6.077 | 1.0e-4 |
| **4 (best)** | **0.4770** | **0.6649** | 5.258 | 6.558 | 4.863 | 6.043 | 2.9e-5 |
| 5 | 0.4715 | 0.6668 | 5.196 | 6.563 | 4.806 | 6.043 | 0 |

Validation loss falls through epoch 4 and is flat at 5; no sign of overfitting in 5 epochs. Shuffled-label
control on the saved outputs: macro-F1 0.006 (must be < 0.02).

| On the same M4 crops (all 17,151 validation photos) | Top-1 | Macro-F1 (158 species) |
|---|---|---|
| A3 (before) | 0.6611 | 0.4783 |
| M5_FT | 0.6649 | 0.4770 |
| M5_FT + flip | 0.6706 | 0.4843 |

The fine-tune gains 0.4 points of top-1 on the app's crops and gives back 0.1 of macro-F1; it is not clearly
better than A3. For reference A3 on its own M3 crops: top-1 0.6644, macro-F1 0.4800.

## The distilled model (M5_KD, M4 crops)

The same run plus the kd term (weight 1, temperature 2) on the 37,847 training photos of B1's 62 species. The
teacher is B1's mapped probabilities, all photos done, with B1's instrument check passed in the kernel. The
validation loss's kd term uses B1's saved validation probabilities; it is logged only.

| Epoch | Val macro-F1 | Val top-1 | Train loss | Val loss | Train species / kd | Val species / kd | lr at end |
|---|---|---|---|---|---|---|---|
| **1 (best)** | **0.4784** | **0.6746** | 10.406 | 9.797 | 5.141 / 4.850 | 6.131 / 3.147 | 2.0e-4 |
| 2 | 0.4543 | 0.6960 | 9.104 | 9.236 | 5.451 / 3.230 | 6.102 / 2.624 | 1.7e-4 |
| 3 | 0.4387 | 0.6923 | 8.703 | 9.022 | 5.445 / 2.845 | 6.092 / 2.426 | 1.0e-4 |
| 4 | 0.4536 | 0.7020 | 8.459 | 8.945 | 5.357 / 2.702 | 6.047 / 2.397 | 2.9e-5 |
| 5 | 0.4495 | 0.7025 | 8.377 | 8.931 | 5.308 / 2.673 | 6.033 / 2.391 | 0 |

Distillation pulls the model toward B1's species. Top-1 climbs to 0.70 while macro-F1 over all species falls.
`best.pt` keeps epoch 1 because the run selects by macro-F1, so later epochs (better top-1, worse macro-F1) are
only in `last.pt` (epoch 5), and their outputs were not saved or scored. Only epoch 1 is a G3 candidate here.

## G3 for each candidate

Computed by `g3_candidates.py`, which imports `m5-targets/g3_ensemble.py` and `m4-checks/check_full_run.py`
and uses their slice, `cf.paired`, `cf.shared_macro_f1` and bootstrap unchanged (11,931 photos, 1,000 photographer
resamples, seed 42). **Instrument check:** it first reproduces A3's -0.0808 / -0.0249 (passed). **Negative control:**
FT's predictions shuffled across photos: top-1 -0.758 (-0.776 to -0.740), macro-F1 -0.704; it fails, as it must.

"Prior" = species logits + tau x log(n_c), n_c the training-list photo count of species c before the caps (the
capped, weighted sampler flattened the natural frequencies, which the validation photos follow). tau is fitted on
**calibration photos only** (grid 0 to 3, step 0.05; maximise the mean of calibration top-1 and calibration
macro-F1), per model: `prior.py`. It is a 184-number bias inside the same model, so the model stays phone size.
I found this lever on the night by probing A3's saved outputs (`prior_probe.py`), which included looking at A3's
validation G3 at a few tau values before I fixed the calibration criterion; tau itself is never fitted on
validation, but that probe is a researcher choice made after seeing validation, and should be read as such.

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
| FT + KD average (two models, fallback) | -0.063 (-0.081 to -0.045) | -0.016 (-0.033 to +0.009) | fail | | 0.676 | 0.483 |
| FT + KD average + flip (two models) | -0.060 (-0.079 to -0.042) | -0.011 (-0.027 to +0.012) | fail | | 0.680 | 0.487 |

Diagnostic only, not a candidate (the head may not be restricted for G3): M5_FT made to answer among B1's 62
species gives top-1 -0.028 and macro-F1 -0.020; M5_KD -0.021 and -0.011. Even closed-set both stay behind B1 on
these species.

**As the app runs it** (`g3_app_new.py`: best.pt exported to the app's fp32 ONNX classifier, ONNX-torch parity
2e-5; exporting A3 the same way reproduces the shipped `classifier.onnx` byte for byte; the app's own detector,
U-Net, gate and crop; 11,924 of the 11,931 photos, 7 are gone from iNaturalist). The A3 rows reproduce
`m5-targets/g3_app.json` exactly.

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
| **M5_KD + flip + prior (tau 0.5)** | **-0.012 (-0.030 to +0.003)** | **+0.001 (-0.017 to +0.024)**: G3 tied on both, passes |

On the app's own crops the M4-crop fine-tunes lose almost nothing against their training-path crops: M5_KD
-0.062 vs -0.061, against A3's -0.087 vs -0.081. That gap is the one this fine-tune was meant to close.

## Safety (`refit_new.py`, `refit_new.json`)

As `m5-targets/refit_search.py`: `decide.fit_thresholds` on the model's calibration outputs, target raised in
0.0025 steps until the WHO recall lower bound **behind the app's gate** (the app's own M4 "both" gate call per
photo, from `m5-targets/app_outputs`, joined by path; a dropped WHO-listed snake is a miss) reaches 0.95.
Calibration outputs are the kernel's (M4 crops through PyTorch), not the ONNX app path. Instrument check: the
default fit reproduces the kernel's `decision_report` thresholds exactly (difference 0). Negative control: the
refitted thresholds on calibration outputs with all logits shuffled across photos give a bound of 0.32.

| Model | Calibration bound behind gate, kernel fit | Refit target | **Bound behind gate, refit** | Val WHO recall behind gate, refit | False alarms | "We don't know" |
|---|---|---|---|---|---|---|
| M5_FT | 0.942 | 0.9600 | **0.956** | 0.966 (0.955 to 0.976) | 24.3% | 2,345 |
| M5_FT + prior | 0.942 | 0.9575 | **0.956** | 0.967 (0.956 to 0.977) | 24.4% | 2,341 |
| M5_FT + flip + prior | 0.942 | 0.9575 | **0.954** | 0.969 (0.957 to 0.979) | 24.4% | 2,224 |

(A3's app refit for comparison: bound 0.954, val 0.965 (0.953 to 0.977), false alarms 24.9%.) In every refit
the WHO-head threshold `theta_who` comes out at infinity (the head never fires alone, as in the kernel's own fit
for this model); the dangerous calls come from WHO-listed mass, top species and venomous-family mass.

| Model | Calibration bound behind gate, kernel fit | Refit target | **Bound behind gate, refit** | Val WHO recall behind gate, refit | False alarms | "We don't know" |
|---|---|---|---|---|---|---|
| M5_KD | 0.941 | 0.9550 | **0.955** | 0.961 (0.946 to 0.972) | 22.0% | 2,321 |
| M5_KD + prior | 0.942 | 0.9550 | **0.955** | 0.958 (0.944 to 0.969) | 22.0% | 2,046 |
| **M5_KD + flip + prior** (the G3 pass) | 0.944 | 0.9550 | **0.954** | **0.963 (0.950 to 0.973)** | 21.9% | 1,898 |

All refits meet the rule: a calibration lower bound of at least 0.95 behind the app's gate. The shuffled-logits
controls give 0.31 to 0.32. Before the refit, every new model's kernel-fit thresholds fall short behind the gate
(0.941 to 0.944), as A3's did, so shipping any of them needs the refitted thresholds in `pipeline.json`. For
the G3-passing candidate the validation interval's lower end is 0.950.

## G1 and G2 (M5_FT, from the kernel)

- **G1** (scenery swap gap, photos with a mask): 0.692 as is vs 0.640 swapped, gap **0.053**; goal <= 0.084
  (half of M3's 0.168): passed (A3: 0.052). Measured on the plain model; flip and the prior act after the crop and
  were not run through the scenery test.
- **G2**: **85 of 158** species at F1 >= 0.5 (goal 80): passed (A3: 87). FT + flip 86, FT + prior 85,
  FT + flip + prior 85 (same per-class F1 rule, from the saved outputs).

- **M5_KD**: G1 gap 0.700 - 0.645 = **0.055**: passed. G2 **88** of 158: passed. KD + flip 87, KD + prior 86,
  **KD + flip + prior 81**: passed, just (goal 80).

## KD pilot (coordinator's request)

**The full KD run's settings (weight 1.0, temperature 2, lr 2e-4) were chosen without a pilot.** When the
coordinator asked for one, the full KD run was already about an hour in, and Jayesh's account allows only 2 GPU
sessions, so the pilot ran as soon as the fine-tune freed a slot (`jayeshbansal497/sg-m5-kd-pilot`, notebook 29,
1.05 h session, 04:14-05:18 IST). It was run to justify or replace the full run's choice; replacing it with a new
full run would not have finished before the deadline in any case.

Pilot: the 16% experiment subset's 13,611 training photos (capped), M4 crops, each setting from A3's `best.pt`
for 2 epochs at lr 2e-4 (1 warm-up epoch, then cosine), eval on all 17,151 validation photos, best epoch by val
macro-F1. B1 ran on the subset's 10,542 training photos of its 62 species (instrument check passed again: max
difference 0.0). G3 slice score with the same slice, metric and bootstrap; the shuffled control fails in every
row (top-1 about -0.75).

| Setting | Best epoch | Val macro-F1 | Val top-1 | G3 top-1, ours minus B1 | G3 macro-F1, ours minus B1 |
|---|---|---|---|---|---|
| no kd (A3's losses) | 1 | 0.486 | 0.674 | -0.067 (-0.085 to -0.051) | -0.017 (-0.033 to +0.008) |
| **kd weight 1, T 2 (the full run)** | 1 | **0.490** | **0.706** | **-0.024 (-0.041 to -0.009)** | **-0.000 (-0.017 to +0.023)** |
| kd weight 3, T 4 | 1 | 0.398 | 0.687 | -0.028 (-0.044 to -0.015) | -0.019 (-0.036 to +0.002) |

By both criteria (G3 slice score and val macro-F1) the pilot picks the setting the full run already uses; the
heavier, softer setting costs 9 points of macro-F1. The learning rate was not varied in the pilot (3 settings at
most); it is the fine-tune's. Validation losses are not comparable across rows (they include the kd term).
Every setting peaked at epoch 1 of 2 on the subset. Per-epoch rows are in `out_pilot/pilot/pilot.json`.

## Files

- Fine-tuned model: `out_finetune/M5_FT/best.pt` (SHA-256 `996fffd3def60596286f80e4ede7d896f275139d27e7a0638aa4d25b45f4f8f6`,
  epoch 4) and `out_finetune/M5_FT/last.pt` (`f55d738208c728d068cfd69cad1e4f3646466e0b72ed699f74802709846af973`,
  epoch 5, optimizer, scheduler, scaler, RNG); outputs `*_outputs.npz`, `*_outputs_flip.npz`, `prototypes.npz`,
  `decision_metrics.json`, `metrics.jsonl`; the executed notebook `out_finetune/executed/28_m5_finetune.ipynb`;
  A3 on M4 crops `out_finetune/A3_m4crops/`; M4 front-end file `out_finetune/full/frontend_preds.csv`.
  Also in the Kaggle output of `jayeshbansal497/sg-m5-finetune`.
- App ONNX of M5_FT: `app_models_FT/classifier.onnx` (other files copied from the app bundle unchanged).
- Distilled model: `out_kd/M5_KD/best.pt` (`1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f`,
  epoch 1), `out_kd/M5_KD/last.pt` (`db873a70cfcb21b439e62e61ba25a45fdc7f8758ba087e60bb9b8aaf790fc0c4`, epoch 5),
  the same output files, `out_kd/b1_teacher.log`, `out_kd/full/teacher.json`, the executed notebook, and
  `out_kd/A3_m4crops/`. Also in the Kaggle output of `jayeshbansal497/sg-m5-kd`. App ONNX: `app_models_KD/`.
- KD pilot: `out_pilot/pilot/pilot.json`, `out_pilot/executed/29_m5_kd_pilot.ipynb`.
- Scripts and results: `g3_candidates.py/.json`, `g3_app_new.py`, `g3_app_FT.json`, `g3_app_A3.json`, `prior.py`,
  `prior_probe.py`, `refit_new.py/.json`, `build_nb28.py`, `build_nb29.py`, `make_runner.py`, `kernel_*/`,
  `kg.sh` (runs the CLI as Jayesh, reading only that token line).

## What failed, and what I could not check

- v1 of both kernels failed in the smoke pass (bundle missing a data file); the smoke stopped them after ~20 min.
- Jayesh's account allows 2 GPU sessions at once, so the KD pilot could only start when the fine-tune finished.
- The full KD run's settings were not piloted beforehand. The pilot ran afterwards and agrees with them (above).
- KD epochs 2 to 5 (top-1 about 0.70) were not scored for G3: only the best-macro-F1 epoch's outputs are saved.
  `last.pt` (epoch 5) can be scored later.
- The two-model averages do not pass and would double the phone's classifier cost. They are reported only.
- Not checked: the new thresholds through the ONNX app path on calibration (the refit used PyTorch M4 crops with
  the app's gate calls); the fp16 classifier; on-device latency (unchanged architecture and size, 20.8 MB);
  higher input resolution (not tried); G2 counts with the prior correction beyond macro-F1.
