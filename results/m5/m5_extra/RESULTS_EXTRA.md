# M5 extra evaluation: robustness, Grad-CAM, int8 (24-25 November 2026)

These are evaluation-only runs of the frozen pipeline (`results/m5/test/FREEZE.md`). No weight, threshold, tau or rule was
changed. The metrics come from `score_test.py`'s own functions, imported unchanged. GPU work ran on Kaggle only, on two
private kernels, `jayeshbansal497/sg-m5-extra` v2 (197 min) and `vikas25s/sg-m5-extra` (v2 155 min, v4 460 min). No
paid model API was called. Intervals use 1,000 photographer (or owner) resamples with seed 42.

## Instrument checks (before any corrupted number)

- **Locked clean run through this scorer** (`score_robustness.py`, run on `test/kernel_out`): top-1 0.7141, macro-F1
  0.4615, WHO recall 0.9442 (0.923 to 0.973), lower bound 0.9261, snakes kept 99.31%, non-snakes called a snake 5.00%.
  All six numbers equal `test_scores.json` / RESULTS.md (asserted to within 1e-12).
- **Clean pass through the corruption harness, in each kernel.** Each kernel decodes every photo, writes it back
  (lossless PNG), and runs the front end, classifier, snake check and gate on that copy. In-kernel check on 600 held-out
  photos against the locked run: top-1 agreement 1.0 (plain and mirrored), max logit difference 0.0, box and mask
  presence 1.0, gate calls 1.0, snake check P diff 0.0 with 0 flips. Over all 18,533 photos, both kernels' clean
  rows equal the locked run exactly (top-1 change 0.000, every metric identical).
- **Gate on GPU** (the reference pipeline's two ONNX models on the CUDA provider): it gives the same calls as the saved
  CPU gate on the 300 validation photos (agreement 1.0), and on the clean held-out set its calls equal the locked run's.

## 1. Robustness stress test (section 6)

The set is the held-out split (18,533 snake photos) plus the 600 non-snake test photos. Each corruption is applied to
the decoded photo before anything else runs, so the front end (detector, U-Net, crop), M5_KD flip + prior, the
decision rules, the M4 gate and the G4 snake check all see the corrupted photo. Definitions are in `corrupt.py`; an
example of each is in `figures/corruption_examples.png`. Noise uses a fixed seed per photo.

- Blur sigma is a share of the short side (0.5 / 1 / 2%, about 1.7 / 3.3 / 6.7 px on a 500 x 334 photo).
- Noise standard deviation is 10 / 25 / 50 grey levels.
- Brightness multiplies every pixel and clips. The pipeline's exposure fix still runs afterwards.
- Rotation keeps the same canvas with reflected edges.
- Occlusion is a grey (128) centred square covering 10% or 25% of the photo's area.
- Low resolution downscales by area and scales back up bilinearly.

Changes are against the locked clean run. The top-1 interval is a paired photographer bootstrap of corrupted minus
clean on the same photos. WHO recall is measured behind the M4 gate, as the frozen safety goal is. False alarms are
the share of not-WHO-listed, non-viper/elapid photos called "dangerous".

| Condition | Top-1 | Change in top-1 (95% CI) | Macro-F1 (165 spp.) | Change | WHO recall | Change | False alarms | Snakes kept (check) | Change | Non-snakes called snake | Change |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clean (locked run = RESULTS.md)** | 0.714 | | 0.462 | | 0.944 | | 21.6% | 99.31% | | 5.00% (30/600) | |
| Clean, harness rerun (both kernels) | 0.714 | 0.000 (0.000 to 0.000) | 0.462 | 0.000 | 0.944 | 0.000 | 21.6% | 99.31% | 0.00 | 5.00% | 0.00 |
| Blur s1 (0.5%) | 0.606 | -0.109 (-0.118 to -0.098) | 0.368 | -0.094 | 0.905 | -0.039 | 31.4% | 99.19% | -0.12 | 6.83% | +1.83 |
| Blur s2 (1%) | 0.392 | -0.322 (-0.339 to -0.305) | 0.202 | -0.259 | 0.842 | -0.102 | 48.4% | 97.05% | -2.26 | 11.17% | +6.17 |
| Blur s3 (2%) | 0.180 | -0.535 (-0.551 to -0.520) | 0.046 | -0.415 | 0.731 | -0.213 | 63.8% | 93.92% | -5.39 | 36.50% | +31.50 |
| Noise s1 (sd 10) | 0.657 | -0.057 (-0.064 to -0.051) | 0.420 | -0.041 | 0.946 | +0.002 | 29.0% | 98.33% | -0.98 | 8.17% | +3.17 |
| Noise s2 (sd 25) | 0.466 | -0.248 (-0.265 to -0.234) | 0.279 | -0.182 | 0.949 | +0.005 | 51.3% | 97.70% | -1.61 | 17.83% | +12.83 |
| Noise s3 (sd 50) | 0.192 | -0.522 (-0.546 to -0.500) | 0.085 | -0.376 | 0.927 | -0.017 | 78.7% | 95.60% | -3.71 | 42.33% | +37.33 |
| JPEG q30 | 0.653 | -0.061 (-0.067 to -0.055) | 0.389 | -0.072 | 0.939 | -0.005 | 28.9% | 97.24% | -2.07 | 6.50% | +1.50 |
| JPEG q10 | 0.476 | -0.238 (-0.249 to -0.229) | 0.243 | -0.219 | 0.909 | -0.036 | 43.9% | 99.07% | -0.24 | 27.33% | +22.33 |
| Darker x0.6 | 0.698 | -0.016 (-0.021 to -0.012) | 0.440 | -0.022 | 0.957 | +0.013 | 25.1% | 99.66% | +0.35 | 9.50% | +4.50 |
| Darker x0.3 | 0.660 | -0.054 (-0.062 to -0.046) | 0.384 | -0.077 | 0.959 | +0.014 | 29.8% | 99.87% | +0.56 | 19.50% | +14.50 |
| Brighter x1.5 | 0.684 | -0.030 (-0.037 to -0.024) | 0.435 | -0.026 | 0.917 | -0.027 | 23.0% | 98.84% | -0.47 | 4.50% | -0.50 |
| Brighter x2.0 | 0.610 | -0.104 (-0.113 to -0.094) | 0.375 | -0.086 | 0.861 | -0.083 | 27.3% | 98.04% | -1.27 | 7.00% | +2.00 |
| Rotate 15 deg | 0.709 | -0.005 (-0.009 to -0.001) | 0.457 | -0.004 | 0.946 | +0.002 | 21.8% | 99.38% | +0.08 | 6.33% | +1.33 |
| Rotate 30 deg | 0.706 | -0.008 (-0.013 to -0.002) | 0.459 | -0.002 | 0.945 | +0.001 | 22.1% | 99.39% | +0.08 | 7.83% | +2.83 |
| Centre occlusion 10% | 0.578 | -0.136 (-0.145 to -0.126) | 0.351 | -0.110 | 0.826 | -0.118 | 28.7% | 98.11% | -1.20 | 7.50% | +2.50 |
| Centre occlusion 25% | 0.451 | -0.263 (-0.275 to -0.250) | 0.240 | -0.222 | 0.599 | -0.346 | 27.2% | 96.24% | -3.07 | 9.50% | +4.50 |
| Low-res x0.5 | 0.677 | -0.037 (-0.042 to -0.032) | 0.429 | -0.033 | 0.927 | -0.017 | 25.6% | 99.36% | +0.05 | 5.67% | +0.67 |
| Low-res x0.25 | 0.544 | -0.171 (-0.180 to -0.160) | 0.308 | -0.153 | 0.901 | -0.044 | 38.2% | 98.96% | -0.35 | 7.50% | +2.50 |

"Change" for snakes kept and non-snakes is in percentage points. The WHO lower bounds are in
`results/robustness_table.csv` (clean 0.926). Six conditions sit above it: darker x0.3 0.948, darker x0.6 0.945,
noise s2 0.940, noise s1 0.930, rotate 30 0.933 and rotate 15 0.931. All others are below, and none reaches the
0.95 mark.

What the table shows:

- **Only rotation is harmless.** It costs under 1 point of top-1, and the interval excludes 0 by a small margin. Mild
  darkening (x0.6, -1.6 points) and halving the resolution (-3.7) come next.
- **Blur and noise are the worst.** The mildest blur (sigma about 1.7 px) costs 10.9 points of top-1. At the strongest
  blur and noise settings, top-1 falls to 0.18-0.19.
- **WHO recall can stay high for the wrong reason.** Under noise it barely moves (0.927 to 0.949) because the decision
  rules call more photos "dangerous". At noise s3, 15,163 of 18,533 photos are "dangerous" against 7,281 clean, and
  false alarms reach 78.7% (clean 21.6%). Darkening raises WHO recall the same way. Occlusion and blur cut WHO recall
  for real: 0.599 at 25% occlusion, where 6,134 photos fall to "no snake" against 351 clean and the gate passes 12,399
  against 18,182 clean.
- **The snake check is fairly stable on snakes, but not on non-snakes.** It keeps 93.9% or more of snakes in every
  condition. Its non-snake false positives rise sharply under strong blur (36.5%), strong noise (42.3%), JPEG q10
  (27.3%) and heavy darkening (19.5%). The G4 mark of 5% holds only for clean photos and brighter x1.5.

### OOD summary (from existing results, not recomputed)

From RESULTS.md, `analysis_heldout.json` and `test_scores.json`:

- **`test` vs `test_extra`.** `test` (India, uploaded 2022+, 5,144 photos) has frozen top-1 0.586 against 0.763 on
  `test_extra` (13,389). A3 scores 0.588 vs 0.696. On `test` alone, macro-F1 is 0.401 (held-out 0.462), WHO recall
  0.930 with bound 0.910, and G2 is 46 of 106.
- **Lookalike categories** (snake check, 600 non-snake test photos, 30 called a snake): caecilian 30% (6/20),
  earthworm 20% (4/20), eel 15% (6/40), legless lizard 10% (2/20), other animal 8.3% (5/60), rope/hose/belt 5%
  (3/60), hand/arm 3.3% (2/60), person 1.1% (2/180), cat, dog and T-shirt 0% (0/140). By source: iNaturalist
  lookalikes 18%, Open Images 2.4%.
- **Front-end gap.** Photos without a usable mask (1,796) have frozen top-1 0.484, against 0.739 for the 16,737 photos
  with one.
- **Water snakes.** Sea snakes and kraits: top-1 0.820, snake check keeps 98.2%. Other water snakes: top-1 0.520, kept
  98.5%.

## 2. Grad-CAM (section 6)

The maps are on M5_KD's plain pass, taken for the class of the frozen answer (flip + prior). They cover all 16,737
clean held-out photos with a usable U-Net mask. "Mass" is the share of the map inside the mask, as the classifier's
own crop carries the mask into the 320 x 320 input. "Area" is the share of the input the mask covers (mean 0.162;
0.279 if the letterbox padding is excluded). The negative control is the same architecture with every weight
re-initialised (seed 42); every parameter tensor changed. The mass-minus-area interval is a photographer bootstrap.

| Map | Mean mass in mask | Mean mass minus mean area (95% CI) | Photos with mass > area | Median mass / area | Mass on letterbox padding |
|---|---:|---|---:|---:|---:|
| Trained, final block (`final_conv`, the layer asked for) | 0.060 | -0.102 (-0.106 to -0.098) | 11.4% | 0.19 | 64.1% |
| Trained, last MobileViT block (`stages.4`) | 0.085 | -0.070 (-0.076 to -0.065) | 17.7% | 0.16 | 63.2% |
| Trained, stride-16 stage (`stages.3`) | 0.139 | -0.023 (-0.029 to -0.018) | 33.9% | 0.61 | 50.0% |
| **Trained, stride-8 stage (`stages.2`)** | **0.230** | **+0.068 (+0.065 to +0.071)** | **70.9%** | **1.35** | 33.9% |
| Random weights, final block (control) | 0.096 | -0.066 (-0.068 to -0.064) | 10.1% | 0.59 | 53.0% |
| Random weights, stride-8 stage (control) | 0.084 | -0.079 (-0.084 to -0.074) | 27.9% | 0.59 | 60.1% |

Plain facts from the table and the figures:

- **The final block fails the sanity check.** Its Grad-CAM mass sits mostly on the grey letterbox padding and the
  input's corners (64% on padding). It is below the random-weights control on the snake (0.060 vs 0.096). The cause is
  structural: the model global-average-pools the final block, so the gradient there is the same at every position, and
  the map is only the class-weighted activation, which peaks at the padded borders. A laptop look at 37 local
  validation photos showed this before the test run, so the earlier layers were added as layer checks then.
- **The stride-8 stage gives sensible maps.** Its mass in the mask exceeds the mask's area on 71% of photos (median
  ratio 1.35, mean +0.068 above area). The randomised control at the same layer stays below area (-0.079). By this
  measure the trained model looks at the snake more than chance, and the control rules out an edge or contrast
  artefact of the architecture. The effect is moderate (mass 0.23 against area 0.16), not a concentration.
- **Right and wrong answers look alike.** At stride 8, mass minus area is +0.068 when the frozen answer is right
  (12,366 photos) and +0.069 when it is wrong (4,371). The confusions are not explained by the model looking off the
  snake.
- **Figures:**
  - `figures/gradcam_grid_stride8.png`: 12 correct and 12 misclassified photos, stride-8 maps.
  - `figures/gradcam_grid_final.png`: the same 24 photos, final-block maps.
  - `figures/gradcam_control.png`: six photos with trained final, stride-16 and stride-8 maps beside the
    random-weights maps.
  - The misclassified photos come from the top confusions: A. oxyrhynca or A. borealis -> A. prasina, P. mucosa ->
    N. naja, N. naja -> P. mucosa, P. mucosa -> P. korros, C. gramineus -> A. prasina, D. chairecacos -> D. tristis,
    L. fasciolatus -> L. aulicus.
- **The other shortcut check (G1 scenery swap, RESULTS.md).** On the 16,737 masked held-out photos, M5_KD's top-1 is
  0.701 as is and 0.645 with the scenery replaced, a gap of 0.057 against the 0.084 mark (validation 0.055). The
  shuffled control gap is 0.675. The background carries some, but not most, of the signal.

## 3. int8 compression (section 9)

Calibration used 500 calibration-split crops (`lists/quant_calib.csv`) for the classifier and 500 calibration snakes
(`kernel/inputs/lists/snake_calib.csv`) for the snake check, never a test photo.

- **Classifier variants:** the app's fp32 ONNX export of M5_KD (max logit diff 1.3e-5 against `best.pt`); fp16
  (`convert_float_to_float16`, keep_io_types, as the app's fp16 file); dynamic int8 (`quantize_dynamic`, int8
  weights); and static int8 (QDQ, per-channel int8 weights, uint8 activations, MinMax).
- **Snake check variants:** the frozen fp32 and fp16 ONNX files, plus dynamic and static int8 built the same way.
- **Scoring:** onnxruntime 1.31.1, CPU provider, on the Kaggle kernel (Intel Xeon 2.0 GHz). The classifier inputs are
  the clean pass's training-path crops, plain and mirrored, then prior added. The snake check inputs go through the
  app's own input path.
- **G3** is `score_test.g3_block` on the 12,203-photo G3 slice. On that slice B1 has top-1 0.780 and macro-F1 0.713.
- **G4** uses the frozen cut of 0.2048.
- **Latency** is on the laptop (Apple Silicon, onnxruntime 1.31.1, CPU provider, 1 thread, batch 1): the median of 200
  runs after 20 warm-up runs.

| Classifier (M5_KD) | File | Laptop median latency | Top-1, G3 slice | Macro-F1 (62 spp.) | G3 top-1 vs B1 | G3 macro-F1 vs B1 | Same answer as PyTorch |
|---|---:|---:|---:|---:|---|---|---:|
| PyTorch (locked test run, RESULTS.md) | | | 0.774 | 0.719 | -0.006 (-0.020 to +0.008), tied | +0.006 (-0.017 to +0.028), higher | |
| ONNX fp32 | 20.8 MB | 47.9 ms | 0.774 | 0.719 | -0.006 (-0.020 to +0.007), tied | +0.006 (-0.017 to +0.029), higher | 99.8% |
| ONNX fp16 | 10.6 MB | 73.4 ms | 0.774 | 0.719 | -0.006 (-0.020 to +0.007), tied | +0.006 (-0.017 to +0.029), higher | 99.8% |
| ONNX int8 dynamic | 5.8 MB | 71.7 ms | 0.214 | 0.135 | -0.566, lower (fails) | -0.577, lower | 22.0% |
| ONNX int8 static (MinMax) | 6.2 MB | 66.7 ms | 0.098 | 0.046 | -0.682, lower (fails) | -0.667, lower | 9.9% |

| Snake check (distil_oi) | File | Laptop median latency | Snakes kept (18,533) | Non-snakes called a snake (600) |
|---|---:|---:|---:|---:|
| PyTorch (locked test run) | | | 99.31% (99.15 to 99.47) | 5.00% (30) |
| ONNX fp32 | 16.0 MB | 23.0 ms | 99.31% | 5.00% (30) |
| ONNX fp16 | 8.1 MB | 23.3 ms | 99.31% | 5.00% (30) |
| ONNX int8 dynamic | 4.3 MB | 35.8 ms | **16.76%** (15.6 to 17.8) | 9.83% (59) |
| ONNX int8 static (MinMax) | 4.8 MB | 13.4 ms | 98.27% (97.7 to 98.8) | **37.67%** (226) |

**int8 breaks accuracy.** Post-training int8 breaks the classifier outright: dynamic int8 drops top-1 on the slice
from 0.774 to 0.214, static from 0.774 to 0.098, and G3 fails by about 57 to 68 points. It also breaks the snake check.
Dynamic int8 keeps only 17% of snakes. Static int8 keeps 98.3% of snakes but calls 37.7% of non-snakes a snake.
Neither int8 file is usable without quantization-aware training or mixed precision, and neither was tried here.
On this laptop, int8 is also not faster for the classifier (67-72 ms against 48 ms fp32). Only the static snake check
is faster (13.4 ms against 23.0 ms). fp16 halves both files with no accuracy change, and fp32 ONNX reproduces the
PyTorch numbers.

**Platform parity** (64 validation photos run on the laptop against the kernel's x86 outputs): fp32 and fp16 agree
fully (top class 64/64). The int8 classifier differs between the two machines: dynamic int8 matches on 61/64 photos
(max logit diff 1.7) and static int8 on 38/64 (max logit diff 6.7). So int8 accuracy is also platform-dependent; the
accuracy rows above are from the x86 kernel. The int8 snake check agrees on the top class 64/64.

## Numbers not produced

- **Percentile-calibrated static int8** was declared as a second variant and dropped. onnxruntime's histogram
  calibrator raised "Too many bins" on the laptop smoke test. In kernel v3 the process died with no traceback while
  calibrating the classifier (likely out of memory: it keeps every activation of the 500 crops). Only MinMax static
  int8 is reported.
- **Corrupted-condition intervals** are given for the top-1 difference only, as asked. The other metrics carry point
  changes; WHO recall intervals for each condition are in `results/robustness.json`.
- **G5 (video)** was not run under corruption.
- **Latency** was measured on the laptop only, not on a phone.

## Runs and failures (facts)

- **Kernel v1, both accounts:** failed after 1 minute in the inputs step. The manifest listed a `__pycache__` .pyc
  that the uploaded dataset did not contain. Fix: `__pycache__` was excluded from staging and from the manifest check,
  then new dataset versions were uploaded.
- **Jayesh v2:** clean pass, Grad-CAM and 10 corruptions (blur x3, noise x3, JPEG x2, rotation x2). Complete.
- **Vikas v2:** clean pass and 8 corruptions (dark x2, bright x2, occlusion x2, low-res x2), all complete. The
  compression step failed because onnxruntime writes a helper file next to its input model, inside read-only
  `/kaggle/input`. These corruption outputs are in `kernel_out/vikas25s_v2`.
- **Vikas v3:** compression only (after copying the model to a writable folder). Died during Percentile calibration,
  as above.
- **Vikas v4:** compression with MinMax only. Complete, 7.3 h of CPU scoring.

## Files (all under `results/m5/m5_extra/`)

- **Tables:**
  - `results/robustness.json` (every condition: metrics, verdict counts, intervals)
  - `results/robustness_table.csv`
  - `results/gradcam.json` (all, right, wrong)
  - `results/compression.json`
  - `results/latency.json`
- **Figures:**
  - `figures/gradcam_grid_stride8.png`
  - `figures/gradcam_grid_final.png`
  - `figures/gradcam_control.png`
  - `figures/corruption_examples.png`
- **Code:**
  - `corrupt.py`
  - `select_photos.py` (lists, seed 42)
  - `kernel/` (`stage_inputs.py`, `build_kernel.py`, `extra_kernel.py`, `gradcam_step.py`, `compression_step.py`;
    pushed-script hashes in `kernel/PUSHED_*.sha256`)
  - `score_robustness.py`, `gradcam_report.py`, `score_compression.py`, `latency.py`, `corruption_examples.py`
- **Kernel outputs:**
  - `kernel_out/jayeshbansal497/out` (10 corruptions + clean, `gradcam/`)
  - `kernel_out/vikas25s_v2/out` (8 corruptions + clean)
  - `kernel_out/vikas25s/out/compression` (int8 models, per-photo logits, parity inputs)
- **Laptop code check:** `local_smoke/` (stand-in validation photos only).
