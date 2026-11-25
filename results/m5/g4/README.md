# G4: a small snake check for the phone (23 to 24 November 2026)

**Summary:** best model `distil_oi` calls 5.17% of validation non-snakes a snake (one photo over the 5% mark) and
keeps 99.20% of snakes, in 8.1 MB. The approved retry with more lookalike negatives (`distil_oi_v2`) gave 5.67% and
99.34%: no better. G4 is not met on validation.

G4: at most 5% of non-snake photos called a snake, and at least 99% of real snakes kept. The coordinator scores the
600-photo non-snake **test** split once at the end; it was not read here. Everything below is on **validation**
(600 non-snake photos, 17,151 snakes), with the cut-off fitted on the 7,999 calibration snakes. No paid model API
was called. GPU runs: Vikas's Kaggle account only.

## Result: fails on validation by one photo

| Final variant `distil_oi` (fixed before it ran; see "Variants") | Validation | 95% interval (1,000 owner resamples, seed 42) | G4 mark |
|---|---:|---:|---:|
| Non-snakes called a snake | **5.17%** (31 of 600) | 3.5% to 7.0% | at most 5% |
| Snakes kept | **99.20%** (17,013 of 17,151) | 98.88% to 99.45% | at least 99% |
| Calibration snakes kept (cut-off fitted here) | 99.01% (7,920 of 7,999) | | |

**Snakes kept meets the mark; false snakes misses it by one photo** (30 would be 5.0%). The interval on false snakes
spans the mark, so validation cannot say which side of 5% the model is on; G4 is not met on validation. Against
today's gate on the same validation photos (27.3% false snakes, 98.3% kept) it is about five times fewer false
snakes with more snakes kept, and against the BioCLIP + A3 head (13.7%, 599 MB) less than half the false snakes in
8.1 MB. The 31 photos are one cut-off step from passing: the next non-snake scores sit at 0.189 and 0.177 against
a cut of 0.205, the lowest false snakes at 0.223 and 0.236. Moving the cut by hand to pass would be fitting it on
validation, so it was not done.

ONNX: `runs/distil_oi/snake_check_fp16.onnx` (8.1 MB) and `snake_check.onnx` (16.0 MB fp32); cut-off
**0.20484977503483762** on `probs[0]`.

### Per category (validation non-snakes called a snake, final variant)

| Category | Photos | Called a snake |
|---|---:|---:|
| Caecilian | 20 | 25.0% (5) |
| Eel | 40 | 17.5% (7) |
| Legless lizard | 20 | 15.0% (3) |
| Earthworm | 20 | 0% |
| Other animal | 60 | 13.3% (8) |
| Person | 180 | 3.3% (6) |
| Hand / arm | 60 | 1.7% (1) |
| Rope / hose / belt | 60 | 1.7% (1) |
| Cat / dog / T-shirt | 140 | 0% |
| **Open Images (500)** | | **3.2%** |
| **iNaturalist lookalikes (100)** | | **15.0%** |

### Snakes kept by group (validation, final variant)

| Group | Photos | Kept | 95% interval |
|---|---:|---:|---:|
| Sea snakes and sea kraits (*Hydrophis*, *Laticauda*) | 715 | 97.9% | 96.6% to 98.9% |
| Blind snakes (Typhlopidae; the only blind-snake family in validation) | 1,672 | 99.2% | 98.7% to 99.7% |
| Shieldtails (Uropeltidae) | 91 | 100% | |
| Other water snakes (Homalopsidae, Acrochordidae) | 318 | 94.0% | 87.4% to 98.4% |
| Vine snakes and bronzebacks | 1,959 | 99.7% | 99.4% to 99.9% |

Sea snakes and sea kraits: 97.9% against 96.9% behind today's gate (notebook 20) and 92.9% behind approach 3's
retrained gate. Water snakes are the weak point: *Gerarda prevostiana* 16 of 22 kept, *Acrochordus granulatus*
49 of 54, *Enhydris enhydris* 100 of 106. *Python bivittatus* is the most-missed common species (447 of 468,
95.5%). Most-missed species per variant are in `runs/<variant>/g4_eval.json`.

## The model

- **EfficientNet-B0** (timm `efficientnet_b0.ra_in1k`, ImageNet weights), fine-tuned end to end with a 3-way head:
  snake / lookalike (caecilian, earthworm, eel, legless lizard) / other non-snake. Score = P(snake) from the softmax.
- Input: the photo as the app already prepares it (PIL decode, exposure fix, letterbox on the grey pad, ImageNet
  normalise), at **256 x 256** instead of 320.
- ONNX: `snake_check.onnx` fp32 **16.0 MB**, `snake_check_fp16.onnx` (fp16 weights, fp32 input and output, the same
  conversion as the app's `classifier_fp16.onnx`) **8.1 MB**. Outputs `probs` [N, 3] and `features` [N, 1280].
- "distil" variants add a feature-distillation loss: a linear projection of the pooled features is pulled (cosine)
  towards BioCLIP's image embedding of the same photo, weight 1.0. The projection is dropped at export, so the phone
  model is the same size.
- Training: 12 epochs x 48,000 photos drawn with replacement (class shares snake 0.5 / lookalike 0.3 / other 0.2;
  0.45 / 0.3 / 0.25 for `distil_oi`), batch 96, AdamW lr 6e-4, cosine, weight decay 0.02, label smoothing 0.05,
  mixed precision on one T4. Augmentation: random crop keeping 60-100% of the photo, flip, rare 90-degree turn,
  photo placed smaller in the letterbox, brightness / contrast / saturation, blur.

## The rule and where it runs

A photo is a snake when `P(snake) >= cut`, with `cut` the highest value at which at least 99% of the 7,999
calibration snakes pass (`m5-g4-probe/analyse.cutoff`). The calibration photographers are in no training run.
The check **replaces** the gate's decision (the detector and U-Net stay, for the crop only): the gate drops 1.7% of
validation snakes, so 99% kept is impossible behind it. On the probe's random 1,999 validation snakes (scored
through today's gate in `m5-g4-probe`), the gate drops 24 and the new check keeps 22 of those; the check drops 14
that the gate keeps.

## Data (every training photo keeps owner and photographer separation)

| Set | Photos | Source |
|---|---:|---|
| Training snakes | 77,068 (6,576 photographers) | the dataset's training split minus every calibration and validation photographer (0 overlap, checked) |
| Notebook 19 non-snake training split | 1,799 | unchanged |
| Extra lookalikes (approach 3) | 1,317 | `m5-g4-probe/hard_negatives` |
| New iNaturalist negatives (`data/negatives_v1`) | 5,922 | caecilians 946, eels 1,398, legless lizards 900, earthworms 899, Indian reptiles other than snakes 898, Indian animals other than snakes 881 |
| New Open Images negatives (`data/openimages_v1`, `distil_oi` only) | 4,940 | people 1,685, hands / arms / legs / feet 634, everyday things 677, plants and ground 443, animals 524, ropes and belts 272, cats 240, dogs 233, T-shirts 232 |
| Calibration snakes (cut-off only) | 7,999 | |
| Validation snakes | 17,151 | |
| Validation non-snakes | 600 | 500 Open Images + 100 iNaturalist lookalikes |

- iNaturalist: queried **by taxon id** (Gymnophiona 27880, Anguinae 797527, Anguilliformes 47314,
  Synbranchiformes 53512, Crassiclitellata 152943, Squamata 26172 and Animalia 1 in India, place 6681, without
  Serpentes 85553); each id's record must carry the expected name. Research grade, one photo per observation;
  licences CC0 / BY / BY-SA / BY-NC / BY-NC-SA (training use only; 89% are BY-NC, as are many of the snake photos).
  Owners with any photo in notebook 19's validation or test split are dropped.
- Open Images: notebook 19's own source and filters (validation and test splits of V7, any snake label checked,
  machine-made or boxed excludes a photo, CC BY 2.0 only), minus every photo notebook 19 already has and every author
  in notebook 19's validation or test split (354 dropped).
- pHash: every new photo is more than 8 bits from every notebook 19 photo (all splits, read from the manifest's
  stored hashes; no test photo was opened) and from every other training photo; `stage_dataset.py` refuses to stage
  if any training owner is in the validation or test split or any training photo is within 8 bits of one (0 and 0).
- A sample sheet of the new iNaturalist photos: `negatives_v1_sample.jpg` (looked at: all show the stated animal,
  no snake).

## Kaggle

| Kernel (Vikas's account, private, T4) | Variant | Version | Run |
|---|---|---|---|
| `vikas25s/sg-m5-g4-plain` | plain | 1 | 84 min |
| `vikas25s/sg-m5-g4-distil` | BioCLIP feature distillation | 1 | 90 min |
| `vikas25s/sg-m5-g4-shuffled` | negative control: training labels shuffled, 3 epochs | 1 | 41 min |
| `vikas25s/sg-m5-g4-distil-oi` | distil + Open Images negatives, other share 0.25 | 2 | 84 min |
| `vikas25s/sg-m5-g4-distil-oi-seed43` | the same, seed 43 | 1 | 73 min |
| `vikas25s/sg-m5-g4-distil-oi-v2` | retry: distil_oi + round-two lookalikes (dataset version 3) | 1 | 72 min |

Dataset `vikas25s/sg-m5-g4-train` (private; anonymous request returns 403, checked after each version): version 1
for plain / distil / shuffled, version 2 (adds `openimages_v1` and the 600 validation non-snakes, **scored only**)
for `distil_oi`. It is on Vikas's account because a kernel there cannot read a private dataset of Puneet's that was
not shared with it; Vikas's account already held the M3 / M4 datasets. Puneet's account ran nothing.
Version 1 of `distil-oi` started before dataset version 2 was ready and could not be cancelled from the CLI; its
output was replaced by version 2's and was not used. The exact code each kernel ran is in `runs/<variant>/pushed_code`.

## G4 retry: `distil_oi_v2` (approved by Puneet, 24 November; one run)

**Recipe fixed before the run (18:57 IST):** exactly `distil_oi` (EfficientNet-B0, 3 classes, BioCLIP feature
distillation weight 1.0, 256 px, 12 x 48,000 photos, class shares 0.45 / 0.3 / 0.25, seed 42), plus the round-two
negatives below; cut-off fitted on the 7,999 calibration snakes at 99% kept, the same function. One run, no further
iterations. `distil_oi` stays the fallback and is unchanged in `runs/distil_oi/`.

**These negatives were chosen after seeing validation errors, for the second time** (caecilians, eels, legless
lizards, other animals among `distil_oi`'s 31 false snakes), so any validation number from this run is optimistic.

Round-two iNaturalist negatives (`data/negatives_v2`, `scripts/collect_negatives_v2.py`, sample sheet
`negatives_v2_sample.jpg`): 4,791 photos, taxon ids found by walking parent ids and checked by name.

| Taxon (id) | Category | Photos |
|---|---|---:|
| Muraenidae, morays (47313) | eel | 800 |
| Anguilliformes, other eels (47314) | eel | 595 |
| Synbranchiformes, swamp eels (53512) | eel | 600 |
| *Anguis*, slow worms (32036) | legless lizard | 499 |
| *Ophisaurus* (31978) | legless lizard | 400 |
| *Pseudopus* (31974) | legless lizard | 400 |
| Pygopodidae (36925) | legless lizard | 500 |
| Amphisbaenia, worm lizards (85550) | worm lizard (lookalike class) | 597 |
| Syngnathoidei, pipefish and seahorses (1303580) | other | 400 |
| Gymnophiona, caecilians (27880) | caecilian | **0**: round one already took every research-grade CC photo on iNaturalist (3 new candidates, all near-copies) |

Licences: open licences were taken first; CC BY 1,009, CC0 274, CC BY-SA 126 (29%), CC BY-NC 3,310 and
BY-NC-SA 72 (71%) fill the rest. Owners in notebook 19's validation or test split: 0. Staging check (dataset version
3, 18,769 non-snake training photos): 0 training owners in validation or test, 0 photos within 8 pHash bits of a
validation or test photo. No validation or test non-snake photo is in training. Kernel
`vikas25s/sg-m5-g4-distil-oi-v2` (Vikas's account, T4), dataset `vikas25s/sg-m5-g4-train` version 3 (anonymous
request 403). The kernel refuses to train if any photo is missing or no `negatives_v2` photo is present.

### Retry result: fails, and is no better than `distil_oi`

| Validation | `distil_oi_v2` (retry) | `distil_oi` (fallback, unchanged) | G4 mark |
|---|---:|---:|---:|
| Non-snakes called a snake | **5.67%** (34 of 600), 3.8% to 7.3% | 5.17% (31), 3.5% to 7.0% | at most 5% |
| Snakes kept | **99.34%** (17,037 of 17,151), 99.12% to 99.52% | 99.20%, 98.88% to 99.45% | at least 99% |
| Calibration snakes kept | 99.01% | 99.01% | |
| Cut-off on P(snake) | 0.107 | 0.205 | |
| AUROC, validation | 0.997 | 0.995 | |

Intervals: 1,000 resamples of whole owners / photographers, seed 42. Kernel run 72 min, 95,837 training photos,
0 missing.

| Non-snake category | Photos | Retry | `distil_oi` |
|---|---:|---:|---:|
| Caecilian | 20 | 25.0% | 25.0% |
| Eel | 40 | 15.0% | 17.5% |
| Legless lizard | 20 | 20.0% | 15.0% |
| Earthworm | 20 | 5.0% | 0% |
| Other animal | 60 | 13.3% | 13.3% |
| Person | 180 | 2.8% | 3.3% |
| Hand / arm | 60 | 5.0% | 1.7% |
| Rope / hose / belt | 60 | 3.3% | 1.7% |
| Cat / dog / T-shirt | 140 | 0% | 0% |
| Open Images (500) / iNaturalist lookalikes (100) | | 3.6% / 16% | 3.2% / 15% |

| Snakes kept | Photos | Retry | `distil_oi` |
|---|---:|---:|---:|
| Sea snakes and sea kraits | 715 | **99.3%** (98.4 to 99.9) | 97.9% |
| Other water snakes (Homalopsidae, Acrochordidae) | 318 | 94.3% (88.8 to 98.6) | 94.0% |
| Blind snakes (Typhlopidae) | 1,672 | 98.9% (98.3 to 99.5) | 99.2% |
| Shieldtails | 91 | 100% | 100% |
| Vine snakes and bronzebacks | 1,959 | 99.9% | 99.7% |

What happened: the extra lookalikes made the model separate snakes from non-snakes slightly better overall (AUROC
0.995 to 0.997) and kept more sea snakes, but they also made it less sure about the hardest real snakes, so the
cut-off that keeps 99% of calibration snakes fell from 0.205 to 0.107. At that cut the lookalikes still pass at the
same rate (caecilians could not be added: iNaturalist has no more research-grade CC caecilian photos). The two
models' false snakes overlap in 22 photos; 9 are new failures of `distil_oi` only and 12 of the retry only, which is
the same size of churn as between two seeds of `distil_oi` (31 vs 32). The retry is within noise of `distil_oi` and
does not meet G4. **`distil_oi` stays the recommended model** (fewer false snakes); the retry would be preferred only
if sea-snake recall is weighted above the false-snake rate.

ONNX (`runs/distil_oi_v2/`): fp32 16.0 MB, fp16 8.1 MB. Parity: fp32 vs PyTorch max 5e-6, fp16 vs PyTorch max
0.006 (40 photos, same top class); kernel PyTorch vs laptop fp16 ONNX through the app path: 1,000 snakes, max
0.017, 0 flips; 600 validation non-snakes, max 0.009, 1 flip (33 called a snake by the laptop fp16 model, 34 by
the kernel). The test split stays unopened.

## Variants, controls, parity, latency

Four trained models were compared on validation (plus one control). `plain` and `distil` were run first; their
false snakes on Open Images people, hands and ropes (7 to 8%) showed the model had too few everyday negatives, so
`distil_oi` added 4,940 Open Images photos (and gave "other" a larger share). It was declared the final variant
before it ran; nothing was chosen among variants after seeing its number. `distil_oi_seed43` is the same recipe with
another seed, run to see how much the result moves between identical trainings (see below).

| Variant | Non-snakes called a snake | Snakes kept | AUROC (val) | Non-snake scores from |
|---|---:|---:|---:|---|
| plain | 9.67% (58) [7.3, 11.8] | 99.24% [99.03, 99.43] | 0.991 | local fp16 ONNX, app path |
| distil (BioCLIP teacher) | 8.33% (50) [6.0, 10.5] | 99.37% [99.17, 99.55] | 0.993 | local fp16 ONNX, app path |
| **distil_oi (final)** | **5.17% (31) [3.5, 7.0]** | **99.20% [98.88, 99.45]** | **0.995** | kernel fp32 (local fp16: also 31) |
| distil_oi_v2 (retry, round-two lookalikes) | 5.67% (34) [3.8, 7.3] | 99.34% [99.12, 99.52] | 0.997 | kernel fp32 (local fp16: 33) |
| distil_oi_seed43 (same recipe, seed 43) | 5.33% (32) [3.5, 7.0] | 99.29% [99.06, 99.51] | 0.996 | kernel fp32 |
| Control: training labels shuffled | 98.5% (591) [97.5, 99.5] | 98.90% | 0.360 | local fp16 ONNX, app path |
| Control: random score | 98.8% (593) | 98.83% | | |

Each cut-off was fitted on the calibration snakes for that variant (99.01% of calibration kept in every row).
Distillation helped a little (58 to 50); the Open Images negatives helped most (Open Images false snakes 7.0% to
3.2%). Lookalikes stayed at 12 to 15%: caecilians, eels and legless lizards are the remaining wall.

**Seed variance.** The same recipe with seed 43 gives 32 false snakes (5.33%) and 99.29% kept, against 31 and
99.20% for seed 42: the result is stable at about 5.2 to 5.3% and misses the mark both times. Which photos fail
moves more than the total (seed 43: eels 5% instead of 17.5%, ropes 6.7% instead of 1.7%).

**Controls.** Both must fail and do: the model trained on shuffled labels calls 98.5% of non-snakes a snake at the
99% cut, and a random score 98.8%, so the evaluation can fail.

**Separation of the instrument.** Snakes (calibration and validation) are scored in the kernel with PyTorch fp32;
for `distil_oi` the kernel uses the app's input path (PIL decode, exposure fix, letterbox) for every scored photo.
The same photos through the exported fp16 ONNX on the laptop (`scripts/app_inputs.py`, built from
`reference_pipeline.py`'s own functions):

| Check (final variant) | Photos | Max abs. difference in P(snake) | Decisions that flip at the cut |
|---|---:|---:|---:|
| fp32 ONNX vs PyTorch, same input | 40 | 2e-6 | 0 |
| fp16 ONNX vs PyTorch, same input | 40 | 0.006 | 0 |
| Kernel PyTorch vs laptop fp16 ONNX, 500 calibration + 500 validation snakes | 1,000 | 0.007 | 0 |
| Kernel PyTorch vs laptop fp16 ONNX, all validation non-snakes | 600 | 0.007 | 0 |

For `plain` and `distil` the kernel used OpenCV decoding without the exposure fix; their kernel-vs-laptop snake
parity (distil, 1,000 photos) is max 0.096, 0 flips.

**Latency** (`results/latency.json`, laptop CPU, ONNX Runtime 1.31, batch 1, median of 100 runs; the laptop was
busy with other jobs, load average about 14, so all numbers are high and only the ratios mean much):

| Model | 1 thread | 2 threads |
|---|---:|---:|
| snake check fp16 (256 px) | 23 ms | 13 ms |
| snake check fp32 | 28 ms | 20 ms |
| A3 classifier fp32 (320 px) | 48 ms | 32 ms |
| U-Net segmenter fp32 | 34 ms | 23 ms |
| detector | 7 ms | 7 ms |

An earlier run of the same architecture at lower load: 17 ms fp32 one thread, against 37 ms for A3.

## How the app would run it (`app/lib/inference/pipeline.dart`)

The snake check **replaces** the gate's decision; the detector and U-Net stay, but only to find the crop. Behind
today's gate the target cannot be met: the gate alone drops 1.7% of validation snakes (2.0% of calibration snakes).

Order of steps in `SnakePipeline.run`:

1. Decode and exposure fix: unchanged (`fixExposure`).
2. **New: snake check.** `normalise(letterbox(rgb, 256, s.padRgb), s.mean, s.std, s.scale)` (the existing Dart
   functions, with the check's own input size), one run of `snake_check_fp16.onnx`, read output `probs` (shape
   [1, 3]: snake, lookalike, other). If `probs[0] < cutoff`, return `ModelAnswer.noSnake` straight away, as the
   gate does today. That also skips the detector, U-Net and classifier for most non-snake photos.
3. Detector and U-Net: unchanged, but their result no longer decides "no snake". `cleanMask` and `cropWindow` give
   the crop as today; when there is no mask (the gate would have said no snake), the classifier gets the whole
   photo, which `cropWindow` already returns as `null` -> `rgb`.
4. Classifier and decision rules: unchanged.

Changes, all small:

- `assets/models/snake_check_fp16.onnx` (8.1 MB) added to `modelFiles`; one more `OrtSession`.
- `assets/model/pipeline.json` gains a block, read by `PipelineSettings`:
  `"snake_check": {"input_size": 256, "output": "probs", "snake_index": 0, "cutoff": <fitted value>,
  "file": "snake_check_fp16.onnx"}`, and `"gate": {"rule": "snake_check"}` so the old rule's thresholds are kept
  for the record but not applied.
- `_run` takes the input size as an argument (today it reads `settings.inputSize`, 320, for every model).
- `GateResult` gains `snakeProbability` so the golden-photo check can compare it with the Python reference.
- Golden photos: `make_golden.py` adds the check's probability per photo; the Dart test compares it to 1e-3.

Size on the phone: 8.1 MB fp16 (16.0 MB fp32), against 54 MB for the three models today. Latency (laptop, see
`results/latency.json`) is under half of the A3 classifier's.

Not done in the app: no Dart was written or run; the phone latency, fp16 on Android's CPU provider, and the
Dart-vs-Python parity of the new step are untested.

## What could not be checked

- **G4 itself.** The non-snake test split was not read. On validation the result is a fail by one photo with an
  interval that spans the mark; the coordinator's one test score can land on either side.
- **Validation is not fresh.** The same 600 non-snakes were read for five trained variants (and, in the earlier
  probes, for the BioCLIP and A3 heads). Training data was chosen from their errors twice: the Open Images
  negatives after `plain`, the round-two lookalikes after `distil_oi`. Every number here is optimistic for that.
- **Snakes were not all scored on the laptop.** The machine was too loaded (load average above 60 at times) to put
  25,150 snakes through the laptop ONNX path in time; 1,000 were, with no decision changed.
- **Label quality of the validation non-snakes** was not audited. The 31 false snakes are on one sheet,
  `results/false_snakes_val_distil_oi.jpg`; one "other animal" photo (number 11 on the sheet) looks to us like a
  snake, and several are lizards with short legs. None were removed or relabelled.
- **The phone.** No Dart was written or run; phone latency, fp16 on Android's CPU provider and Dart-vs-Python
  parity of the new step are untested.
- **Licences.** 89% of the new iNaturalist negatives are CC BY-NC; fine for training a course model, to be
  checked before any commercial use (as for the NC snake photos).
- **Owner separation beyond notebook 19.** Non-snake training owners were checked against notebook 19's
  validation and test owners only. iNaturalist logins and Open Images Flickr authors are different namespaces, so a
  person with accounts in both is not caught.
- **Seed variance** is measured with one extra seed only (two runs: 31 and 32 false snakes).

## Files

- `scripts/collect_negatives.py`, `scripts/collect_openimages.py`: the new negatives (manifests and reports in
  `data/*/manifest.csv`, `report.json`).
- `scripts/stage_dataset.py`: builds the Kaggle dataset folder and runs the separation checks.
- `scripts/kernel/train_g4.py` (+ `make_kernels.py`): the training kernel; `runs/<variant>/pushed_code` is what ran.
- `scripts/export_onnx.py`, `scripts/app_inputs.py`, `scripts/score_onnx.py`, `scripts/kernel_scores.py`,
  `scripts/evaluate.py`, `scripts/parity.py`, `scripts/time_onnx.py`, `scripts/after_kernel.sh`: export, scoring,
  G4 evaluation, parity, latency.
- `runs/<variant>/`: `model.pt`, `snake_check.onnx`, `snake_check_fp16.onnx`, `train_config.json` (with the loss
  per epoch), `g4_eval.json`, `export.json`, `parity_torch_vs_onnx.json`, the kernel log.
