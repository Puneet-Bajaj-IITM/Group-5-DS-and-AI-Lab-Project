# M3 run configs

One YAML per run id, read by `python -m serpentguard.train --config configs/m3/<run_id>.yaml ...` (notebook 12 runs
one or two at a time, one per GPU). Every run starts from the plan's section 6.1 recipe and changes only the fields
its question tests. The controller writes each config once its parent's winner is known. E1's winner is MobileViT-S (`E1_mvits`), so N0, E2, E3 and E4 derive from it; E3 and E4 run beside E2 and are read as differences against `E1_mvits`.

## The section 6.1 recipe

`head: linear`, `losses: [ce]`, `sampler: weighted` (notebook 06's 1/√n × WHO boost), `front_end: F0`,
`pooling: gap`, `mask_head: false`, `hierarchy: true`, `input_size: 320`, `epochs: 20`, `lr: 3.0e-4` (encoder ×0.1),
`weight_decay: 0.05`, `batch_size: 64`, `seed: 42`, `eval_every: 2`, `warmup_epochs: 1`. The other `RunConfig` fields
keep their defaults (`embedding_dim: 256`, `danger_weight: 5.0`, `aug_strength: 1.0`, `mask_source: sam`,
`exclude_species: []`, ...).

## Run ids

| Run id | Parent | Fields changed against the parent | Question |
|---|---|---|---|
| `E1_mnv4` | recipe | `encoder: mobilenetv4_conv_small` | E1: which encoder? |
| `E1_effb0` | recipe | `encoder: efficientnet_b0` | E1 |
| `E1_mvits` | recipe | `encoder: mobilevit_s` | E1 |
| `E1_effb0_s1` | `E1_effb0` | `seed: 1` | E1 seed spread, run on the spare GPU during E1; it is `N0_s1` if EfficientNet-B0 wins |
| `N0_s1`, `N0_s2` | E1 winner | `seed: 1`, `seed: 2` | N0: early noise floor |
| `E2_ce` | E1 winner | none: the E1 winner's run, already scored with the head and with prototypes | E2: does the face-recognition head help? |
| `E2_arcface` | E1 winner | `head: arcface`, `losses: [arcface]` | E2 |
| `E2_ce_triplet` | E1 winner | `losses: [ce, triplet]`, `sampler: pk` | E2 |
| `E2_arcface_triplet` | E1 winner | `head: arcface`, `losses: [arcface, triplet]`, `sampler: pk` | E2 |
| `E2b` | E1 winner | `sampler: pk` | E2b: CE with the triplet runs' P×K sampler (§14.1) |
| `E2c` | E1 winner | `losses: [la_ce]` | E2c: logit-adjusted CE, the long-tail baseline (§14.2) |
| `E3` | E1 winner, run beside E2 (Ruling R66) | `hierarchy: false` | E3: do the genus and family heads help? |
| `E4` | E1 winner, run beside E2 (Ruling R66) | `danger_weight: 1.0` | E4: does the danger weight cut O3 errors? |
| `R_F0` | recipe R (E2 winner + E3 + E4) | `front_end: F0`: recipe R on the whole photo; it is FE's F0 and H1's base | FE: does a front end beat the whole photo? |
| `FE_F1` | E4 winner | `front_end: F1`, `mask_source: predicted`, `box_source: predicted` | FE: YOLO box crop |
| `FE_F1m` | E4 winner | `front_end: F1m`, `mask_source: predicted` | FE: crop to the predicted mask's box, no masking |
| `FE_F2` | E4 winner | `front_end: F2`, `mask_source: predicted` | FE: cropped soft mask |
| `FE_F3` | E4 winner | `front_end: F3`, `mask_source: predicted` | FE: cropped hard mask |
| `FE_F4` | E4 winner | `front_end: F4`, `mask_head: true` | FE: auxiliary mask loss, plain pooling (mask head dropped at export) |
| `FE_F5` | E4 winner | `front_end: F5`, `mask_head: true`, `pooling: mask` | FE: mask-guided pooling at inference |
| `K1` | E2 winner | `exclude_species`: the 10 below | K1: rare and new species |
| `N1_s42` (tuned winner) or `N1_s1`, `N1_s2` (untuned) | final configuration | tuned: `seed: 42`, alongside the retrain's `H1_topK_s1`/`_s2` (Ruling R78); untuned: `seed: 1`, `seed: 2` | N1: noise floor over seeds {42, 1, 2} |
| `D2` | final configuration | `losses` + `kd` | D2 (Should): distillation |

## What the runner enforces

- `triplet` needs `sampler: pk` (16 classes × 4 photos at batch 64); `arcface` loss needs `head: arcface`.
- F1, F1m, F2 and F3 train **and** validate on front-end predictions (`frontend_preds.csv` under `--preds-root`), so
  they must set `mask_source: predicted`. Their validation, calibration and unknown inputs never use SAM masks.
  `box_source` is only recorded: F1's boxes come from `frontend_preds.csv`.
- F4 trains its mask head on SAM masks (`mask_source: sam`) with `pooling: gap`; F5 sets `pooling: mask`. Both
  validate on the whole photo. `pooling: mask` is F5 only.
- A run that reads masks or boxes raises if the training frame has none, and F1–F3 raise if no validation photo has
  a prediction. `final_metrics.json` records each frame's prediction coverage and the loader's `num_workers` (the
  augmentation stream depends on it).
- Notebook 12 pins albumentations 2.0.8 and timm 1.0.31 (`requirements-m3.txt`) before it loads the package.
- `exclude_species` (K1 only, Ruling R18) drops those species from training, validation, calibration and
  prototypes after notebook 06's caps, so every other class keeps its parent's training photos. Their train and val
  photos are embedded into `heldout_outputs.npz` for enrolment.
- `kd` is rejected for now: D2 needs a teacher's probabilities, which the runner doesn't load yet. The F3 run
  trained on SAM masks (the FE "Should") would also need a runner change, since F3 must set `mask_source: predicted`.

## K1 species

`subset.heldout_species(classes, metadata)` with seed 42 (rules R17 and R18): *Amphiesma monticola*, *Argyrophis
diardii*, *Boiga thackerayi*, *Elaphe hodgsonii*, *Lycodon jara*, *Melanophidium wynaudense*, *Platyceps
ventromaculatus*, *Sibynophis subpunctatus*, *Uropeltis ellioti*, *Uropeltis macrolepis*.

## Stage A recipe (5 Nov, 05:15)

E2 → ArcFace head (Ruling R96); E3 → genus and family heads kept (R97); E4 → danger weight 1 (R98: the 5× weight did
not cut verdict-level O3 or false alarms, and weight 1 scores higher). Recipe R = `R_F0` (ArcFace, hierarchy, danger
weight 1, whole photo). The FE runs (`FE_F1`, `FE_F1m`, `FE_F2`, `FE_F3`, `FE_F4`, `FE_F5`), `K1` and the H1 search
all start from R; `R_F0` is FE's F0 and H1's base.
