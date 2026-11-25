# M5 locked test: the frozen pipeline, scored once (24 November 2026)

Pipeline and rules: `FREEZE.md` (written 20:35 IST, before any test photo was opened; addendum section 8 before any
test output was read). Held-out = `test` (5,144) + `test_extra` (13,389) = 18,533 snake photos, 600 non-snake test
photos, the G5 clips. Scored once by Kaggle kernel `jayeshbansal497/sg-m5-test` v1 (Jayesh's account, 2 x T4, 48.5
min, every step passed first time) and `score_test.py --split test` (one run). No paid model API. Intervals: 1,000
photographer / owner resamples, seed 42.

## Goals: validation vs test

| Goal (mark) | Validation | Test (held-out, primary) | Test pass | `test` only (5,144) |
|---|---|---|---|---|
| G1 scenery gap (<= 0.084) | 0.055 | **0.057** (0.701 as is vs 0.645 swapped, 16,737 photos with a mask) | **pass** | not computed separately |
| G2 species F1 >= 0.5 (>= 80) | 81 of 158 | **89 of 165** | **pass** | 46 of 106 (fail) |
| G3 top-1 vs B1, tied or higher | -0.012 (-0.030 to +0.002) | **-0.006 (-0.020 to +0.008)**, tied | **pass** | +0.027 (-0.003 to +0.042) |
| G3 macro-F1 vs B1, tied or higher | +0.002 (-0.014 to +0.025) | **+0.006 (-0.017 to +0.028)**, higher | **pass** | +0.017 (-0.028 to +0.047) |
| Safety: WHO recall lower bound behind the M4 gate (>= 0.95) | recall 0.962 (0.950 to 0.973), bound 0.952 | **recall 0.944 (0.923 to 0.973), bound 0.926** (2,633 WHO-listed) | **FAIL** | 0.930 (0.908 to 0.975), bound 0.910 |
| G4 non-snakes called a snake (<= 5%) | 5.17% (31 of 600) | **5.00% (30 of 600)**, 3.3% to 6.7% | **pass** (at the mark) | same 600 |
| G4 snakes kept (>= 99%) | 99.20% (17,013 of 17,151) | **99.31% (18,405 of 18,533)**, 99.15% to 99.47% | **pass** | 99.36% (99.01 to 99.57) |
| G5 <= 1 change per real test clip, frozen classifier (22 new clips) | dev: max 1 | **max 1 at 0.1 / 0.25 / 0.4 / 0.8 s a frame** | **pass** | 6 old clips: max 0, pass |
| G5 as first scored (A3, theta_conf 0.287) | | max 1 at 0.1-0.4 s; 2 on 1 clip at 0.8 s | fail at 0.8 s only | old 6: pass |

G3 slice: 12,203 photos, 1,781 photographers (B1 top-1 0.780, macro-F1 0.713; frozen 0.774 / 0.719). `test`-only
slice: 3,849 photos, 387 photographers. G1 was computed once on the whole held-out set. Everything passes on test
except safety. G4 passes exactly at the mark (30 of 600; one more false snake would fail it).

## Before / after (same test photos, same metric code)

| | A3 on M4 crops (shipped M4) | Frozen M5 (M5_KD + flip + prior) |
|---|---|---|
| G3 top-1 vs B1 | -0.080 (-0.106 to -0.057), lower | -0.006 (-0.020 to +0.008), tied |
| G3 macro-F1 vs B1 | -0.013 (-0.037 to +0.008), tied | +0.006 (-0.017 to +0.028) |
| Top-1, all 18,533 | 0.666 | 0.714 |
| Macro-F1, 165 measurable species | 0.481 | 0.462 |
| G2 | 91 of 165 | 89 of 165 |
| WHO recall behind M4 gate | 0.933 (0.906 to 0.969), bound 0.909 (A3 shipped thresholds) | 0.944 (0.923 to 0.973), bound 0.926 |
| False alarms (not WHO-listed, not viper/elapid, called dangerous) | 21.6% | 21.6% |
| Non-snakes called a snake | 28.3% (M4 gate, scored in M4) | 5.00% (snake check) |
| Snakes kept | 98.11% (gate drops 351 of 18,533) | 99.31% (check drops 128) |

Also: frozen classifier behind the snake check instead of the gate (the app as it would ship with G4): WHO recall
0.960 (0.944 to 0.982), bound 0.946, false alarms 22.3%; still under the 0.95 bound.

## Negative controls (each must fail; all did)

G3 shuffled predictions: top-1 -0.744, macro-F1 -0.697 (lower). G2 shuffled: 0 of 165. Safety with all logits
shuffled across photos: bound 0.378. G4 random score: 99.2% false snakes; scores shuffled across snakes and non-snakes:
97.2% false snakes. G1 with swapped predictions shuffled: gap 0.675. **G5 control did not fail**: with each new clip's
per-frame outputs shuffled across its frames, the lock still shows at most 1 change at every speed
(`g5/control_results`). The 3 s switch rule keeps the shown species stable whatever the frames say, so G5 as measured
cannot tell a good classifier from a scrambled one; it tests the lock only.

## Instrument checks (before any test number)

- `score_test.py --split val` reproduces every README number exactly (`val_dry_run.json`); one scorer fix first (G2
  per-class F1 over all photos, FREEZE section 8).
- In the kernel on validation photos with saved outputs: fresh M4 front end, 600 photos, box and mask agreement 1.0,
  box_conf diff 0.0; M5_KD top-1 agreement 1.000 (max logit diff 0.035), mirrored 1.000, A3 0.998; snake check 500
  photos, max P(snake) diff 1e-6, 0 flips; ONNX gate 300 photos, calls agree 1.0; B1 64 photos, diff 0.0 (swapped
  control 0.96). robustness_metrics' own as-is check on the held-out: agreement 1.0.
- G5: re-scoring the saved A3 outputs with `lock_eval.py` reproduces `g5/results/new_test_scores.csv` exactly; the
  frozen per-frame gate calls equal the A3 run's frame for frame (gate is classifier-independent); the frozen app-path
  wrapper agrees with the kernel's flip + prior answer on 40 of 40 validation photos (without the prior 35 of 40).

## Errors and limits (facts)

- **Safety fails on test.** 147 of 2,633 WHO-listed photos (gate passed or not) are not called dangerous: Naja naja 44,
  Craspedocephalus gramineus 39, Daboia russelii 15, Naja kaouthia 10, Echis carinatus 9, C. malabaricus 8; 87 of them
  get "unknown", 60 "not listed". 351 of 18,533 held-out snakes are dropped by the M4 gate. The validation interval's
  lower end (0.950) sat on the mark; test's point value (0.944) is under it.
- **`test` (India, 2022+) is harder than `test_extra`**: frozen top-1 0.586 vs 0.763 (A3 0.588 vs 0.696); the prior's
  gain is in `test_extra`. On `test` alone G2 is 46 of 106 and safety bound 0.910.
- **Worst species** (>= 5 photos): 33 of 165 measurable species have F1 0, e.g. Ahaetulla farnsworthi (62 photos, 54
  called A. prasina), Ophiophagus kaalinga (20, 11 called O. hannah), Hydrophis lapemoides (15, 10 called Laticauda
  colubrina), Sahyadriophis uttaraghati (13, 9 called S. beddomei), Dendrelaphis bifrenalis (13, 7 called Chrysopelea
  ornata). Full list: `per_species_heldout.csv`.
- **Top confusions** (`confusions_heldout.csv`): A. oxyrhynca -> A. prasina 123, A. borealis -> A. prasina 76,
  Ptyas mucosa -> Naja naja 75, P. mucosa -> P. korros 67, Naja naja -> P. mucosa 63, C. gramineus -> A. prasina 47,
  Dendrelaphis chairecacos -> D. tristis 46, Lycodon fasciolatus -> L. aulicus 43. The prior pulls rare congeners to
  the common one (A. prasina, D. tristis, L. aulicus).
- **Lookalike categories** (G4, 30 false snakes): caecilian 6 of 20 (30%), earthworm 4 of 20 (20%), eel 6 of 40 (15%),
  legless lizard 2 of 20 (10%), other animal 5 of 60 (8.3%), rope/hose/belt 3 of 60, hand/arm 2 of 60, person 2 of 180,
  cat/dog/T-shirt 0 of 140. iNaturalist lookalikes 18%, Open Images 2.4%. List: `false_snakes_nonsnake_test.csv`.
- **Water snakes**: snake check keeps sea snakes and kraits 98.2% (768 photos, 97.0 to 99.2) and other water snakes
  98.5% (342, 96.7 to 99.7), both under 99%; blind snakes 98.9%. Species top-1: sea snakes 0.820, other water snakes
  0.520 (A3 0.564). Most-missed by the check: Indotyphlops braminus 18 of 1,657, Laticauda colubrina 8 of 320.
- **Crop front-end gap**: 16,737 photos with a usable mask, frozen top-1 0.739; 1,796 without a mask (box or whole
  photo) 0.484 (A3 0.693 / 0.421).
- **Video**: every no-snake clip (8 of 8) still locks onto a species and ends "Dangerous" with the frozen classifier
  (mostly Naja naja). Safe-species clips ending "Dangerous": 12 of 13 at 0.4 s (latch). Final species right 12 of 19 at
  0.4 s. Median first answer 1.2 s at 0.4 s, 3.2 s at 0.8 s.

## What could not be checked

- G5 with the snake check as the video gate (frozen G5 keeps the M4 gate, as FREEZE states); G5 on a phone.
- G1 on `test` alone; G1 for flip/prior (measured on the plain model, as on validation).
- The frozen pipeline end to end through the app's ONNX path on the held-out photos (the classifier ran in the kernel's
  PyTorch M4-crop path, as validation's G3 pass; only the gate call per photo used the ONNX app path). On validation the
  two paths differed by < 0.001 in G3.
- The non-snake test photos had already been read once in M4 (by the old gate). The laptop code check of the kernel
  script also passed them through the snake check before the kernel ran; that output was deleted unread
  (`local_smoke/NOTE_nonsnake_deleted.txt`).
- Label quality of the test photos and the YouTube clip licences were not audited.

## Files (all under `results/m5/test/`)

`FREEZE.md`, `FREEZE_scripts.sha256`; `test_scores.json` (every number above), `val_dry_run.json`;
`predictions_heldout.csv` (18,533 rows: true species, frozen / A3 / B1 prediction, top-3, gate call, P(snake),
verdict), `predictions_heldout.npz` (frozen and A3 logits), `predictions_nonsnake_test.csv`,
`false_snakes_nonsnake_test.csv`, `per_species_heldout.csv`, `confusions_heldout.csv`, `analysis_heldout.json`;
`kernel/` (stage, build, kernel code), `kernel_out/` (raw kernel outputs and log); `g5/results/` (frozen G5 scores,
`g5_summary.csv`, negatives), `g5/control_results/`, `g5/instrument_a3/`.
