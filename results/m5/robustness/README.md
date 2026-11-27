# Robustness, Grad-CAM and compression

The locked test tells us how the frozen pipeline does on photos as they were uploaded. Phone photos are often worse
than that: blurred, dark, partly covered or small. This folder asks three questions about the same frozen pipeline
(`../test/FREEZE.md`), with no weight, threshold, tau or rule changed:

1. What does each kind of damage do to accuracy and to safety?
2. Is the classifier looking at the snake?
3. Can the models be made smaller without losing accuracy?

The metrics are `score_test.py`'s own functions, imported unchanged. Intervals use 1,000 resamples of whole
photographers (owners, for non-snakes), seed 42. The GPU work ran on two private Kaggle kernels, one on Jayesh's
account (197 min) and one on Vikas's (155 min for the stress tests, 460 min for compression).

## Checks before any damaged number

- **The clean run through this scorer.** On the locked test outputs `score_robustness.py` gives top-1 0.7141,
  macro-F1 0.4615, WHO recall 0.9442 (0.923 to 0.973), lower bound 0.9261, snakes kept 99.31% and non-snakes called a
  snake 5.00%. All six equal `test_scores.json` to within 1e-12.
- **A clean pass through the damage harness, in each kernel.** Each kernel decodes every photo, writes it back as a
  lossless PNG, and runs the front end, classifier, snake check and gate on that copy. On 600 held-out photos checked
  inside the kernel, top-1 agreement with the locked run is 1.0 (plain and mirrored), the largest logit difference is
  0.0, box and mask presence and gate calls agree on all, and the snake check's P(snake) differs by 0.0 with 0 flips.
  Over all 18,533 photos both kernels' clean rows equal the locked run exactly.
- **The gate on GPU.** The reference pipeline's two ONNX models on the CUDA provider give the same calls as the saved
  CPU gate on 300 validation photos (agreement 1.0), and on the clean held-out set its calls equal the locked run's.

## 1. Stress tests

The set is the 18,533 held-out snake photos plus the 600 non-snake test photos. Each kind of damage is applied to the
decoded photo before anything else runs, so the front end (detector, U-Net, crop), the classifier, the decision
rules, the M4 gate and the snake check all see the damaged photo. `corrupt.py` defines them:

- **Blur.** Sigma is a share of the short side: 0.5, 1 and 2%, about 1.7, 3.3 and 6.7 px on a 500 x 334 photo.
- **Noise.** Standard deviation 10, 25 and 50 grey levels, with a fixed seed per photo.
- **JPEG.** Quality 30 and 10.
- **Brightness.** Every pixel is multiplied and clipped; the pipeline's exposure fix still runs afterwards.
- **Rotation.** 15 and 30 degrees, same canvas, reflected edges.
- **Covering.** A grey (128) square in the centre covering 10% or 25% of the photo's area.
- **Low resolution.** Downscaled by area to half or a quarter, then scaled back up bilinearly.

Every change is against the locked clean run. The top-1 interval is a paired photographer bootstrap of damaged minus
clean on the same photos. WHO recall is behind the M4 gate, as the frozen safety goal is. False alarms are the share
of photos neither WHO-listed nor a viper or elapid that are called "dangerous". The changes in snakes kept and
non-snakes called a snake are in percentage points.

| Condition | Top-1 | Change in top-1 (95% CI) | Macro-F1 (165 spp.) | Change | WHO recall | Change | False alarms | Snakes kept | Change | Non-snakes called a snake | Change |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Clean (the locked run)** | 0.714 | | 0.462 | | 0.944 | | 21.6% | 99.31% | | 5.00% (30/600) | |
| Clean, harness rerun (both kernels) | 0.714 | 0.000 (0.000 to 0.000) | 0.462 | 0.000 | 0.944 | 0.000 | 21.6% | 99.31% | 0.00 | 5.00% | 0.00 |
| Blur 0.5% | 0.606 | -0.109 (-0.118 to -0.098) | 0.368 | -0.094 | 0.905 | -0.039 | 31.4% | 99.19% | -0.12 | 6.83% | +1.83 |
| Blur 1% | 0.392 | -0.322 (-0.339 to -0.305) | 0.202 | -0.259 | 0.842 | -0.102 | 48.4% | 97.05% | -2.26 | 11.17% | +6.17 |
| Blur 2% | 0.180 | -0.535 (-0.551 to -0.520) | 0.046 | -0.415 | 0.731 | -0.213 | 63.8% | 93.92% | -5.39 | 36.50% | +31.50 |
| Noise sd 10 | 0.657 | -0.057 (-0.064 to -0.051) | 0.420 | -0.041 | 0.946 | +0.002 | 29.0% | 98.33% | -0.98 | 8.17% | +3.17 |
| Noise sd 25 | 0.466 | -0.248 (-0.265 to -0.234) | 0.279 | -0.182 | 0.949 | +0.005 | 51.3% | 97.70% | -1.61 | 17.83% | +12.83 |
| Noise sd 50 | 0.192 | -0.522 (-0.546 to -0.500) | 0.085 | -0.376 | 0.927 | -0.017 | 78.7% | 95.60% | -3.71 | 42.33% | +37.33 |
| JPEG q30 | 0.653 | -0.061 (-0.067 to -0.055) | 0.389 | -0.072 | 0.939 | -0.005 | 28.9% | 97.24% | -2.07 | 6.50% | +1.50 |
| JPEG q10 | 0.476 | -0.238 (-0.249 to -0.229) | 0.243 | -0.219 | 0.909 | -0.036 | 43.9% | 99.07% | -0.24 | 27.33% | +22.33 |
| Darker x0.6 | 0.698 | -0.016 (-0.021 to -0.012) | 0.440 | -0.022 | 0.957 | +0.013 | 25.1% | 99.66% | +0.35 | 9.50% | +4.50 |
| Darker x0.3 | 0.660 | -0.054 (-0.062 to -0.046) | 0.384 | -0.077 | 0.959 | +0.014 | 29.8% | 99.87% | +0.56 | 19.50% | +14.50 |
| Brighter x1.5 | 0.684 | -0.030 (-0.037 to -0.024) | 0.435 | -0.026 | 0.917 | -0.027 | 23.0% | 98.84% | -0.47 | 4.50% | -0.50 |
| Brighter x2.0 | 0.610 | -0.104 (-0.113 to -0.094) | 0.375 | -0.086 | 0.861 | -0.083 | 27.3% | 98.04% | -1.27 | 7.00% | +2.00 |
| Rotate 15 deg | 0.709 | -0.005 (-0.009 to -0.001) | 0.457 | -0.004 | 0.946 | +0.002 | 21.8% | 99.38% | +0.08 | 6.33% | +1.33 |
| Rotate 30 deg | 0.706 | -0.008 (-0.013 to -0.002) | 0.459 | -0.002 | 0.945 | +0.001 | 22.1% | 99.39% | +0.08 | 7.83% | +2.83 |
| 10% covered | 0.578 | -0.136 (-0.145 to -0.126) | 0.351 | -0.110 | 0.826 | -0.118 | 28.7% | 98.11% | -1.20 | 7.50% | +2.50 |
| 25% covered | 0.451 | -0.263 (-0.275 to -0.250) | 0.240 | -0.222 | 0.599 | -0.346 | 27.2% | 96.24% | -3.07 | 9.50% | +4.50 |
| Half resolution | 0.677 | -0.037 (-0.042 to -0.032) | 0.429 | -0.033 | 0.927 | -0.017 | 25.6% | 99.36% | +0.05 | 5.67% | +0.67 |
| Quarter resolution | 0.544 | -0.171 (-0.180 to -0.160) | 0.308 | -0.153 | 0.901 | -0.044 | 38.2% | 98.96% | -0.35 | 7.50% | +2.50 |

The WHO lower bounds are in `results/robustness_table.csv` (clean 0.926). Six conditions sit above the clean bound:
darker x0.3 0.948, darker x0.6 0.945, noise sd 25 0.940, rotate 30 0.933, rotate 15 0.931 and noise sd 10 0.930.
None reaches the 0.95 mark.

What the table says:

- **Only rotation is harmless.** It costs under 1 point of top-1. Mild darkening (-1.6 points) and half resolution
  (-3.7) come next.
- **Blur is the most likely way a phone photo goes wrong.** The mildest blur, about 1.7 px, already costs 10.9 points
  of top-1. At the strongest blur and noise, top-1 falls to 0.18 to 0.19.
- **Under noise and darkness WHO recall holds for the wrong reason.** The model becomes unsure, and the rules then
  call more photos "dangerous". At noise sd 50, 15,163 of 18,533 photos are "dangerous" against 7,281 clean, and
  false alarms reach 78.7% (clean 21.6%). The app stays cautious, but it stops being useful.
- **Covering and blur cut WHO recall for real.** At 25% covered, WHO recall is 0.599. 6,134 photos fall to "no
  snake" against 351 clean, and the gate passes 12,399 photos against 18,182 clean. The square sits in the centre
  of the photo, which is often where the snake is (Figure 5 in the report).
- **The snake check almost never drops a snake, but it is fooled more easily on damaged non-snakes.** It keeps
  93.9% or more of snakes in every condition. Its false snakes rise sharply under strong blur (36.5%), strong noise
  (42.3%), JPEG q10 (27.3%) and heavy darkening (19.5%). The 5% mark holds only for clean photos and brighter x1.5.

The report also compares these with the out-of-distribution slices of the locked test (India against outside India,
`test` against `test_extra`, lookalikes, photos without a mask, water snakes). Those come from `test_scores.json` and
`analysis_heldout.json` and were not recomputed here.

## 2. Grad-CAM

The maps are on M5_KD's plain pass, taken for the class of the frozen answer (with flip and prior). They cover all
16,737 clean held-out photos with a usable U-Net mask. "Heat on the snake" is the share of the map inside the mask;
the classifier's own crop carries the mask into the 320 x 320 input. "Area" is the share of the input the mask
covers: 0.162 on average, 0.279 if the letterbox padding is left out. The negative control is the same architecture
with every weight re-initialised (seed 42; every parameter tensor changed). The heat-minus-area interval is a
photographer bootstrap.

| Map | Heat on the snake | Heat minus area (95% CI) | Photos with heat above area | Median heat / area | Heat on letterbox padding |
|---|---:|---|---:|---:|---:|
| Trained, final block (`final_conv`) | 0.060 | -0.102 (-0.106 to -0.098) | 11.4% | 0.19 | 64.1% |
| Trained, last MobileViT block (`stages.4`) | 0.085 | -0.070 (-0.076 to -0.065) | 17.7% | 0.16 | 63.2% |
| Trained, stride-16 stage (`stages.3`) | 0.139 | -0.023 (-0.029 to -0.018) | 33.9% | 0.61 | 50.0% |
| **Trained, stride-8 stage (`stages.2`)** | **0.230** | **+0.068 (+0.065 to +0.071)** | **70.9%** | **1.35** | 33.9% |
| Random weights, final block (control) | 0.096 | -0.066 (-0.068 to -0.064) | 10.1% | 0.59 | 53.0% |
| Random weights, stride-8 stage (control) | 0.084 | -0.079 (-0.084 to -0.074) | 27.9% | 0.59 | 60.1% |

- **The final block's map fails the sanity check.** Most of its heat sits on the grey letterbox padding and the
  corners (64%), and it puts less heat on the snake than the random-weights control does (0.060 against 0.096). The
  cause is structural. The model global-average-pools the final block, so the gradient is the same at every
  position, and the map is only the class-weighted activation, which peaks at the padded borders. We saw this on 37
  local validation photos before the test run, and added the earlier layers then.
- **At stride 8 the model looks at the snake.** Heat on the snake beats its area on 71% of photos (median ratio
  1.35), while the random control at the same layer stays below area (-0.079). So the effect is not an edge or
  contrast artefact of the architecture. It is moderate (heat 0.23 against area 0.16), not a sharp focus.
- **Right and wrong answers look alike.** At stride 8, heat minus area is +0.068 when the answer is right (12,366
  photos) and +0.069 when it is wrong (4,371). The confusions are not explained by the model looking away from the
  snake.

The scenery-swap check (G1) points the same way. On the same 16,737 photos top-1 is 0.701 as is and 0.645 with the
scenery replaced, a gap of 0.057 against the 0.084 mark; the shuffled control gap is 0.675. The background carries
some of the signal, but not most of it.

## 3. Compression

Calibration used 500 calibration-split crops for the classifier (`lists/quant_calib.csv`) and 500 calibration
snakes for the snake check, never a test photo.

- **Classifier variants.** The app's fp32 ONNX export of M5_KD (largest logit difference 1.3e-5 against `best.pt`),
  fp16 (`convert_float_to_float16` with fp32 input and output, as the app's fp16 file), dynamic int8 (int8 weights),
  and static int8 (QDQ, per-channel int8 weights, uint8 activations, MinMax calibration).
- **Snake check variants.** The frozen fp32 and fp16 ONNX files, plus dynamic and static int8 built the same way.
- **Scoring.** ONNX Runtime 1.31.1 on the CPU provider, in the Kaggle kernel (Intel Xeon 2.0 GHz). The classifier
  sees the clean pass's training-path crops, plain and mirrored, with the prior added. The snake check sees the
  app's own input path. G3 is `score_test.g3_block` on the 12,203-photo G3 slice, where B1 has top-1 0.780 and
  macro-F1 0.713. G4 uses the frozen cut of 0.2048.
- **Latency.** On the laptop (Apple Silicon, ONNX Runtime 1.31.1, CPU provider, 1 thread, batch 1): the median of 200
  runs after 20 warm-up runs.

| Classifier (M5_KD) | File | Laptop median latency | Top-1, G3 slice | Macro-F1 (62 spp.) | G3 top-1 against B1 | G3 macro-F1 against B1 | Same answer as PyTorch |
|---|---:|---:|---:|---:|---|---|---:|
| PyTorch (the locked test run) | | | 0.774 | 0.719 | -0.006 (-0.020 to +0.008), tied | +0.006 (-0.017 to +0.028), tied | |
| ONNX fp32 | 20.8 MB | 47.9 ms | 0.774 | 0.719 | -0.006 (-0.020 to +0.007), tied | +0.006 (-0.017 to +0.029), tied | 99.8% |
| ONNX fp16 | 10.6 MB | 73.4 ms | 0.774 | 0.719 | -0.006 (-0.020 to +0.007), tied | +0.006 (-0.017 to +0.029), tied | 99.8% |
| ONNX int8 dynamic | 5.8 MB | 71.7 ms | 0.214 | 0.135 | -0.566, fails | -0.577, fails | 22.0% |
| ONNX int8 static (MinMax) | 6.2 MB | 66.7 ms | 0.098 | 0.046 | -0.682, fails | -0.667, fails | 9.9% |

| Snake check (`distil_oi`) | File | Laptop median latency | Snakes kept (18,533) | Non-snakes called a snake (600) |
|---|---:|---:|---:|---:|
| PyTorch (the locked test run) | | | 99.31% (99.15 to 99.47) | 5.00% (30) |
| ONNX fp32 | 16.0 MB | 23.0 ms | 99.31% | 5.00% (30) |
| ONNX fp16 | 8.1 MB | 23.3 ms | 99.31% | 5.00% (30) |
| ONNX int8 dynamic | 4.3 MB | 35.8 ms | **16.76%** (15.6 to 17.8) | 9.83% (59) |
| ONNX int8 static (MinMax) | 4.8 MB | 13.4 ms | 98.27% (97.7 to 98.8) | **37.67%** (226) |

**fp16 halves both files with no change in accuracy, and int8 breaks both models.** Dynamic int8 drops the
classifier's top-1 on the slice from 0.774 to 0.214, and static int8 to 0.098. Dynamic int8 keeps only 17% of
snakes. Static int8 keeps 98.3% of snakes but calls 37.7% of non-snakes a snake. Neither int8 file is usable
without quantization-aware training or mixed precision, and we tried neither. On this laptop int8 is not even faster
for the classifier (67 to 72 ms against 48 ms in fp32). Only the static snake check is faster (13.4 ms against 23.0
ms).

**Platform parity.** We ran 64 validation photos on the laptop and compared them with the kernel's x86 outputs. fp32
and fp16 agree on the top class for all 64. The int8 classifier does not: dynamic int8 agrees on 61 of 64 (largest
logit difference 1.7) and static int8 on 38 of 64 (6.7). So int8 accuracy also depends on the machine, and the
accuracy rows above are from the x86 kernel. The int8 snake check agrees on the top class for all 64.

## Not produced

- **Percentile-calibrated static int8.** We declared it as a second variant and dropped it. ONNX Runtime's histogram
  calibrator raised "Too many bins" on the laptop smoke test. In kernel v3 the process died with no traceback while
  calibrating the classifier, most likely out of memory, since it keeps every activation of the 500 crops. Only
  MinMax static int8 is reported.
- **Intervals for every damaged metric.** The table gives intervals for the top-1 change only. The WHO recall
  intervals of each condition are in `results/robustness.json`.
- **Video under damage** (G5 was not run on damaged frames), and **latency on a phone**.

## Runs

| Kernel push | What it ran | Outcome |
|---|---|---|
| v1, both accounts | everything | failed after 1 minute: the inputs manifest listed a `__pycache__` file that the uploaded dataset did not hold; staging now skips `__pycache__` |
| v2, Jayesh | clean pass, Grad-CAM, 10 conditions (blur, noise, JPEG, rotation) | complete, `kernel_out/jayeshbansal497/` |
| v2, Vikas | clean pass, 8 conditions (dark, bright, covered, low resolution) | conditions complete, `kernel_out/vikas25s_v2/`. Compression failed: ONNX Runtime writes a helper file next to its input model, inside read-only `/kaggle/input` |
| v3, Vikas | compression only, the model copied to a writable folder first | died during Percentile calibration (above) |
| v4, Vikas | compression, MinMax only | complete after 7.3 h of CPU scoring, `kernel_out/vikas25s/` |

Each push's script hashes are in `kernel/`. `PUSHED_scripts.sha256` is the v2 push. `PUSHED_scripts_vikas_v3.sha256`
is v3, and `PUSHED_compression_v4.sha256` covers the one file v4 changed. The scripts in `kernel/` are the last
versions pushed, byte for byte. So v3's record matches every file except `compression_step.py`, which v4's record
matches. v2's record matches the scoring code (`extra_kernel.py`, `corrupt.py`, `gradcam_step.py`,
`stage_inputs.py`) but not `build_kernel.py` or `compression_step.py`, which changed for v3. Check from the repo root:

```bash
shasum -a 256 -c results/m5/robustness/kernel/PUSHED_scripts_vikas_v3.sha256
```

## Files

| Path | What it is |
|---|---|
| `results/robustness.json`, `results/robustness_table.csv` | Every condition: metrics, verdict counts, intervals |
| `results/gradcam.json` | Grad-CAM for all, right and wrong answers, by layer, with the control |
| `results/compression.json`, `results/latency.json` | The compression and latency tables |
| `corrupt.py` | The damage definitions |
| `select_photos.py`, `lists/` | The photo lists (seed 42): the G3 slice, the Grad-CAM photos, parity and calibration photos |
| `score_robustness.py`, `gradcam_report.py`, `score_compression.py`, `latency.py`, `corruption_examples.py` | Laptop scoring and the figures |
| `kernel/` | The kernel code as pushed, with the push hashes |
| `kernel_out/<account>/out/status.json` | Which steps each kernel ran and passed |

The figures (the damage examples and the Grad-CAM grid) are in `../report/figures/`, with a credit for every photo
in `photo_credits.csv`.
