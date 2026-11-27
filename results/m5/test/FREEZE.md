# FREEZE: the M5 pipeline, fixed before the locked test split is read

Written 24 November 2026, 20:35 IST, before any snake test or test_extra photo was opened and before the new
snake check saw any non-snake test photo. Approved by Puneet the same day. Nothing below changes after a test
number is seen. A crash may be fixed and rerun (plumbing only); a model, threshold, tau, rule or set definition may not.

## 1. What is scored

| Set | Photos | Definition |
|---|---:|---|
| **Held-out (primary)** | 18,533 | `metadata.csv` split `test` (5,144) + `test_extra` (13,389): the whole held-out fold of notebook 02. Milestone 2 says `test` alone is too thin per class and to use `test_extra` too |
| `test` only (secondary row) | 5,144 | India, uploaded from 2022-01-01 |
| G3 slice | from held-out | photos uploaded from 2022-01-01 (`created_at >= "2022-01-01"`) whose species is one of B1's 62, the same rule as validation's 11,931-photo slice |
| Non-snake test | 600 | notebook 19's non-snake `test` split, `kaggle/m4_runs/gate/non_snake_test_photos` (fetched in M4 by `scripts/fetch_non_snake_test.py`; scored once by today's gate in M4, never by the new snake check) |
| G5 clips | 22 new + 6 old real test clips, 8 no-snake clips | `results/m5/g5` |

Photos: snake photos are read inside the Kaggle kernel from `puneetbajajiitm/serpentguard-indian-snakes-clean`, the
same dataset the validation runs read. The 600 non-snake photos are uploaded from the laptop into a new private
dataset on Jayesh's account (they are already on the laptop; nothing is downloaded again).

## 2. The frozen models (SHA-256)

| Part | File | SHA-256 |
|---|---|---|
| Species classifier M5_KD (epoch 1) | `results/m5/g3/out_kd/M5_KD/best.pt` | `1ec2ee88b95af0d5a0e6e69feed71c9be4bbdbe3584790ffa389fa52bdb1019f` |
| its prototypes (decision rules) | `out_kd/M5_KD/prototypes.npz` (`plain`) | `a7fa7cba23eb791e41e12187d022e1c1d7503e3e241dbec84da75e17820cade5` |
| its calibration outputs (tau fit, threshold fit) | `out_kd/M5_KD/calib_outputs.npz`, `calib_outputs_flip.npz` | `cf5554a26e9030cdbc91e775e46be4bfdfaa04e93b9a0829fe56e8179f3c877a`, `830a59d915f541063ff48580c822943ea3e2c82ff14d86f2f9f349e2a6fd6493` |
| its saved validation outputs (instrument check only) | `out_kd/M5_KD/val_outputs.npz`, `val_outputs_flip.npz` | `ba3c24d1a0c9311006c043b6bdecf01ddb408b03af400e545a0dac5adcd36711`, `e1433db8d407bc20ab7b30ca110c4ec0bba983dc770fd690b008b8f50e7b05d7` |
| M4 front end: detector | `serpentguard-m5-inputs/m4_front_end/detector/M4_detector.pt` | `29a1ad1c1b9b561c35ef4ffa1916998837c3c78525c21b51668d9d0e95607b61` |
| M4 front end: U-Net | `serpentguard-m5-inputs/m4_front_end/segmenter/segmenter.pt` | `b881c338356bdf44aa0b45e0306f59d7b38252d04316da2455c40055aef656ec` |
| App gate "both" (ONNX, for the gate call per photo) | `m4-final/for-jayesh/models/detector.onnx`, `segmenter.onnx`, `pipeline.json` | `1d2da244575a39c23ebab6ee2f4f04b1c672794031a56fab9d49dc5176f4f9ba`, `da5d59baca34eda5993d837673449b99ccd178c98bbd6755a6b20bbea6f99269`, `b92bb5e27a49d10068aedea4631f2958c70f2a6b732f2d473c50bd7131798063` |
| App reference pipeline (gate code) | `m4-final/for-jayesh/reference_pipeline.py` | `a0e0f6e6e2e9246eb12c9487c217d93071ff884d511bd4420c44827351969452` |
| Snake check `distil_oi` (scored in PyTorch fp32, as validation) | `results/m5/g4/runs/distil_oi/model.pt` | `88fd518c550a4f1502c86e7da1d8a3a8bd9c332b5377a8cc5d7f65f8144b7cfd` |
| Snake check, phone files (parity only) | `runs/distil_oi/snake_check_fp16.onnx`, `snake_check.onnx` | `fb5b5bda5e0dbdb1895448ae1c70e56e93042ef322f9e5595cfcd6e0b8ec9fc3`, `8d11ea7c265fa397ced36fb2d4908bd41bad8ebf1f50efe054ad23aad626da74` |
| Snake check kernel code (scoring functions) | `runs/distil_oi/pushed_code/sg-m5-g4-distil-oi.py` | hashed in section 7 |
| Snake check calibration scores (the cut) | `runs/distil_oi/scores_kernel/scores_calib.csv` | `517b1e4693a5e2e67a127fcebcd170ef67f2c2cc5b9ecd2c897fe005c6c4be1a` |
| A3, the shipped M4 classifier (before/after) | `kaggle/m4_runs/full_a3/out_jayesh/M4_FULL/best.pt` | `b19f0a5ece4f98fc08911f8781048011e14aeb84b80fde9dc1f4b9263322a02b` |
| B1 (Bolon et al. 2022 ViT-L ONNX, LINDAT) | downloaded in the kernel by `scripts/b1_teacher.py` | MD5 `838ae0f8ed19ffeee89aba5fa8b50956` (checked by the script) |
| Training counts for the prior | `log_counts()` of `results/m5/g3/prior.py` from `metadata.csv` + `subset.csv` | vector SHA-256 `d6cb68339c75b59af12f18a6a3a1601c18cfd42bb2a1f12a3459a32a286f4425`; `metadata.csv` `a3f7c188a40e67550a8297187ca7e9fbe1b56e0b1f3888eed5af4565f1be95ba`, `subset.csv` `ba68215a4122273358ee284044fd2a4a55bbd9309603ae50d9ed149087403e4c` |
| Class table | `m5-wt/data/classes.csv` | `c827334d42107636d8ea2aa29a211564cd974121f4ec158e3d7f1ad2e336d52a` |

Not used: `distil_oi_v2`, M5_FT, any two-model average, M5_KD `last.pt`.

## 3. Every cut-off, threshold and setting

**Species answer** (the G3 candidate "KD + flip + prior"):
`logits = (species_logits(crop) + species_logits(mirrored crop)) / 2 + tau * log(n_c)`, **tau = 0.5**
(`prior.fit_tau` on the calibration outputs averaged with their mirrored pass; recomputed today: 0.5. The app-path
fit in `g3_app_KD.json` is also `tau_flip` 0.5). Probabilities = softmax at temperature 1 for G2/G3 (as
`g3_candidates.py`). Crop = M4 front end (`front_end.extend_predictions` from an empty base: detector box at
conf >= 0.05, U-Net mask, F1m crop), as in the g3 kernels.

**Safety thresholds** (`refit_new.json`, "KD flip + prior", chosen at target 0.955, bound behind gate 0.954),
applied by `decide.decide` to the flip-averaged, prior-corrected outputs (every head averaged with the mirrored pass,
embedding re-normalised, as `refit_new.averaged`):

| temperature | temperature_who | theta_d | theta_who | tau_unknown | theta_conf | theta_v | theta_genus |
|---|---|---|---|---|---|---|---|
| 1.5851841617545683 | 0.7729837681215675 | 0.09202129880019957 | inf | 0.6611798278209527 | 0.354303269142032 | 0.5 | 0.5 |

Behind the app's gate "both": `snake = box_conf >= 0.00499307457357645 and mask_area >= 4.8e-05 (and > 0)`, the
call from `reference_pipeline.Pipeline.gate` (ONNX, CPU) on each photo, as `m5-targets/app_outputs.py` made it for
validation; a WHO-listed snake the gate drops is a miss.

**Snake check** (`distil_oi`): snake when `probs[0] >= 0.20484977503483762` (P(snake), softmax of the 3 logits),
input = PIL decode, exposure fix, letterbox 256 on (124, 116, 104), ImageNet normalise (the kernel's `app_rgb` +
`letterbox`). The cut is re-derived from the calibration scores by `analyse.cutoff` and must equal this value.

**Video lock** (`results/m5/g5/results/chosen_settings.json`, SHA-256
`32c6323114648247e1cf254559db2fd5df87410c9a3f00fe354417945377cfc8`): window 0.5 s, margin 0.05, switch 3.0 s,
min frames 1, gate share 0, hold (reset) 2.0 s, "dangerous" latch kept; lock confidence = the decision rules'
theta_conf; speeds 0.1, 0.25, 0.4, 0.8 s a frame.

## 4. Goals and marks (each judged on the held-out set; `test`-only rows reported beside)

| Goal | Mark | Measured on | Code (unchanged metric functions) |
|---|---|---|---|
| G1 scenery | gap <= 0.084 (top-1 as is, photos with a usable mask, minus top-1 with the scenery swapped) | M5_KD plain (as on validation) | `robustness.robustness_metrics`, `conditions.as_is_masked.top1 - conditions.swapped.top1` |
| G2 species | >= 80 species with F1 >= 0.5, among species with >= 5 scored photos | KD + flip + prior | `check_full_run.per_class_f1` on the photos of measurable species (as `head_macro_f1`) |
| G3 vs B1 | top-1 AND macro-F1 on B1's 62 species, G3 slice: each "higher" (diff > 0) or "tied" (95% CI holds 0); "lower" fails | KD + flip + prior vs B1 on the same photos | `g3_candidates.score`: `cf.paired` with `cf.top1` and `cf.shared_macro_f1(62)`, 1,000 photographer resamples, seed 42 |
| Safety | WHO recall lower bound >= 0.95 behind the app's gate | KD + flip + prior, refit thresholds | `refit_new.gated` / `val_row`; bound = `decide.recall_lower_bound` (5th percentile, 1,000 photographer resamples, seed 42), plus the two-sided 95% interval of `refit_new.interval` |
| G4 snake check | <= 5% of non-snakes called a snake AND >= 99% of snakes kept (point values) | `distil_oi` | `results/m5/g4/scripts/evaluate.rule_result` with the calibration scores, held-out snakes and the 600 test non-snakes; intervals 1,000 owner resamples, seed 42 |
| G5 video | <= 1 change of the shown species after the first answer on every real test clip | lock above | `results/m5/g5/code/lock_eval.py score`, `summarize.py` |

## 5. Also reported (not goals)

- B1 on the same photos (its 62 species, any upload date, and the G3 slice).
- A3 on the same M4 crops (the "before"), with A3's app thresholds behind the same gate.
- Safety behind the new snake check instead of the gate (the app as it would ship with G4).
- Per-species and per-category breakdowns; per-photo predictions as csv/npz.
- A negative control per metric: predictions (or scores, or logits) shuffled across photos must fail.

**G5.** The 22 new clips were scored once with A3 and A3's app thresholds (theta_conf 0.287). The frozen pipeline's
theta_conf is 0.354 and its classifier is M5_KD + flip + prior, so the clips are also re-scored once with the frozen
classifier and thresholds (`clip_outputs.py` with M5_KD's ONNX classifier and a mirrored pass, prior added to the
saved logits, `lock_eval.py score` with a pipeline.json and tables.json holding the frozen thresholds and M5_KD's
prototypes). The lock settings are unchanged. The gate in video stays the M4 gate (as scored); the snake check in
video is not part of the frozen G5 run. Both results are reported: "as scored" (A3) and "frozen" (M5_KD).

## 6. Instrument checks before any test number (validation, same code)

1. `score_test.py --split val` on the saved validation outputs must give the README numbers exactly: G3 KD + flip +
   prior -0.012 (-0.030 to +0.002) / +0.002 (-0.014 to +0.025); A3 M3-crop -0.0808 / -0.0249; G2 81 (KD + flip +
   prior), 88 (KD plain); safety 0.963 (0.950 to 0.973) behind the gate, bound on calibration 0.954; G4 5.17% (31
   of 600), 99.20% (17,013 of 17,151), cut 0.20484977503483762; G1 0.055 (from `out_kd/full/robustness_metrics.json`).
2. In the kernel, before the test pass: 600 random validation photos (seed 42) through the same fresh front end,
   M5_KD (plain and mirrored) and A3 must agree with the saved validation outputs (top-1 agreement >= 0.98, logits
   reported); 500 validation snakes through the snake check must agree with `scores_torch.csv` (0 decision flips at
   the cut tolerated up to 1); 300 validation photos through the ONNX gate must agree with
   `m5-targets/app_outputs/val` (gate calls identical); B1's own 64-photo check in `b1_teacher.py`.
3. Negative controls: shuffled predictions / scores / logits fail every goal.

## 7. Scoring commands

```text
# kernel (Jayesh's account, 2 x T4), inputs dataset jayeshbansal497/serpentguard-m5-test-inputs
python results/m5/test/kernel/stage_inputs.py            # stage + hash the inputs folder
zsh results/m5/g3/kg.sh datasets create -p results/m5/test/kernel/inputs -r zip
python results/m5/test/kernel/build_kernel.py            # bundle src + test_kernel.py into runner.py
zsh results/m5/g3/kg.sh kernels push -p results/m5/test/kernel/push
zsh results/m5/g3/kg.sh kernels output jayeshbansal497/sg-m5-test -p results/m5/test/kernel_out
# laptop
m3-final/.venv/bin/python results/m5/test/score_test.py --split val    # instrument check (dry run)
m3-final/.venv/bin/python results/m5/test/score_test.py --split test   # the one test score
m3-final/.venv/bin/python results/m5/test/g5/rescore_g5.py             # G5 with the frozen classifier
```

Script hashes are recorded in `FREEZE_scripts.sha256` at the moment the kernel is pushed, before its output is read.

## 8. Addendum, 2026-11-24 20:49 IST, before any test output was read

The validation dry run (`score_test.py --split val`, `val_dry_run.json`) first gave G2 = 82, not the README's 81:
section 4 described G2 as per-class F1 on the photos of measurable species only, but the kernels' G2
(`per_class_counts.json`, from `species_report`) takes each measurable species' F1 over all photos. The scorer now
does that (`cf.per_class_f1` over all scored photos, counted on species with >= 5 photos); with it every README number
is reproduced exactly (G3, A3 instrument, G2 81 and 88, safety 0.9625 (0.9498 to 0.9732), G4 31 / 600 and 17,013 /
17,151 with the same intervals, G1 0.0548). Nothing else changed. The safety bound by `decide.recall_lower_bound` on
validation is 0.9519.
