# G4: a small snake check for the phone

**The goal.** G4 asks the app to call at most 5% of non-snake photos a snake, and to keep at least 99% of real
snakes. In M4 the gate (detector plus U-Net) called 28.3% of the non-snake test photos a snake. A gate that stops
non-snakes must not drop real snakes, because a dropped venomous snake is a safety miss. So both halves are
marked.

**The answer.** We trained a separate snake check, `distil_oi`: an 8.1 MB EfficientNet-B0. On validation it called
**5.17%** of non-snakes a snake (31 of 600, one photo over the mark) and kept **99.20%** of snakes. The locked test
later gave **5.00%** (30 of 600, exactly at the mark) and **99.31%** (report, section 4.1). A retry with more
lookalike negatives (`distil_oi_v2`) was no better, at 5.67% on validation, and we did not use it.

Everything below is on validation: 600 non-snake photos and 17,151 snakes. The cut-off is fitted on the 7,999
calibration snakes. The non-snake test split was not read here.

## Result on validation

| `distil_oi` (fixed before it ran) | Validation | 95% interval (1,000 owner resamples, seed 42) | Mark |
|---|---:|---:|---:|
| Non-snakes called a snake | **5.17%** (31 of 600) | 3.5% to 7.0% | at most 5% |
| Snakes kept | **99.20%** (17,013 of 17,151) | 98.88% to 99.45% | at least 99% |
| Calibration snakes kept (the cut is fitted here) | 99.01% (7,920 of 7,999) | | |

Snakes kept meets the mark. False snakes miss it by one photo: 30 would have been 5.0%. The interval spans the
mark, so validation cannot say which side of 5% the model is on. Against the M4 gate on the same photos (27.3% false
snakes, 98.3% kept), the check makes about five times fewer false snakes and keeps more snakes. Against our
earlier BioCLIP + A3 head (13.7% false snakes, 599 MB), it makes less than half as many, in 8.1 MB.

The 31 photos sit one step of the cut from passing. The next two non-snake scores are 0.189 and 0.177, the cut is
0.205, and the two lowest false snakes score 0.223 and 0.236. Moving the cut by hand to pass would mean fitting it
on validation, so we did not do it.

### By kind of non-snake

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

### By kind of snake

| Group | Photos | Kept | 95% interval |
|---|---:|---:|---:|
| Sea snakes and sea kraits (*Hydrophis*, *Laticauda*) | 715 | 97.9% | 96.6% to 98.9% |
| Blind snakes (Typhlopidae, the only blind-snake family in validation) | 1,672 | 99.2% | 98.7% to 99.7% |
| Shieldtails (Uropeltidae) | 91 | 100% | |
| Other water snakes (Homalopsidae, Acrochordidae) | 318 | 94.0% | 87.4% to 98.4% |
| Vine snakes and bronzebacks | 1,959 | 99.7% | 99.4% to 99.9% |

Sea snakes do better than behind the M4 gate (97.9% against 96.9%) and much better than behind the gate we
retrained with extra lookalikes in our 19 November probe (92.9%). Other water snakes are the weak point: *Gerarda prevostiana* 16 of 22
kept, *Acrochordus granulatus* 49 of 54, *Enhydris enhydris* 100 of 106. *Python bivittatus* is the most-missed
common species (447 of 468, 95.5%). The most-missed species of each variant are in
`results/variants/<variant>/g4_eval.json`.

## The model

- **EfficientNet-B0** (timm `efficientnet_b0.ra_in1k`, ImageNet weights), fine-tuned end to end with a three-way
  head: snake, lookalike (caecilian, earthworm, eel, legless lizard), other non-snake. The score is P(snake) from
  the softmax.
- **Input.** The photo as the app already prepares it (decode, exposure fix, letterbox on the grey pad, ImageNet
  normalisation), at 256 x 256 instead of 320.
- **Feature distillation** (the `distil` variants). A linear projection of the pooled features is pulled, by cosine
  loss with weight 1.0, toward BioCLIP's embedding of the same photo. The projection is dropped at export, so the
  phone model does not grow.
- **Training.** 12 epochs of 48,000 photos drawn with replacement. Class shares are snake 0.5, lookalike 0.3 and
  other 0.2 (0.45 / 0.3 / 0.25 for `distil_oi`). Batch 96, AdamW at lr 6e-4 with cosine decay, weight decay 0.02,
  label smoothing 0.05, mixed precision on one T4. Augmentation: random crops keeping 60 to 100% of the photo,
  flips, a rare 90-degree turn, the photo placed smaller in the letterbox, colour jitter and blur.
- **Phone files.** `snake_check.onnx` (fp32, 16.0 MB) and `snake_check_fp16.onnx` (fp16 weights with fp32 input
  and output, the same conversion as the app's fp16 classifier, 8.1 MB). Outputs are `probs` [N, 3] and
  `features` [N, 1280].

**The rule.** A photo is a snake when `P(snake) >= cut`, where `cut` is the highest value at which at least 99% of
the 7,999 calibration snakes pass. No training run sees a calibration photographer. For `distil_oi` the cut is
**0.20484977503483762**.

**Why it replaces the gate's decision.** The M4 gate alone drops 1.7% of validation snakes (2.0% of calibration
snakes), so 99% kept is impossible behind it. The detector and U-Net stay, but only to find the crop. On a random
1,999 validation snakes scored through the M4 gate, the gate dropped 24, and the new check kept 22 of those. The
check dropped 14 that the gate kept.

The app now runs the check first, on this branch:
[`documentation/milestone 5/app-m5.md`](../../../documentation/milestone%205/app-m5.md).

## Training data

Every training photo keeps owners and photographers apart from validation and test.

| Set | Photos | Source |
|---|---:|---|
| Training snakes | 77,068 (6,576 photographers) | the training split minus every calibration and validation photographer (overlap checked: 0) |
| Notebook 19's non-snake training split | 1,799 | unchanged from M4 |
| Earlier lookalikes | 1,317 | collected for our 19 November probe that retrained the gate |
| New iNaturalist negatives (`negatives_v1`) | 5,922 | caecilians 946, eels 1,398, legless lizards 900, earthworms 899, Indian reptiles other than snakes 898, Indian animals other than snakes 881 |
| New Open Images negatives (`openimages_v1`, `distil_oi` on) | 4,940 | people 1,685, hands, arms, legs and feet 634, everyday things 677, plants and ground 443, animals 524, ropes and belts 272, cats 240, dogs 233, T-shirts 232 |
| Calibration snakes (cut-off only) | 7,999 | |
| Validation snakes | 17,151 | |
| Validation non-snakes | 600 | 500 Open Images + 100 iNaturalist lookalikes |

- **iNaturalist.** Queried by taxon id, never by name: Gymnophiona 27880, Anguinae 797527, Anguilliformes 47314,
  Synbranchiformes 53512, Crassiclitellata 152943, Squamata 26172 and Animalia 1, in India (place 6681), excluding
  Serpentes 85553. Each id's record had to carry the expected name. Research grade, one photo per observation, CC0,
  BY, BY-SA, BY-NC or BY-NC-SA. 89% are BY-NC, as are many of the snake photos. We dropped any owner with a photo in
  notebook 19's validation or test split.
- **Open Images.** Notebook 19's own source and filters: the validation and test splits of V7, any snake label
  checked, machine-made or boxed labels excluded, CC BY 2.0 only. We removed every photo notebook 19 already had and
  every author in its validation or test split (354 dropped).
- **Near copies.** Every new photo is more than 8 pHash bits from every notebook 19 photo, in all splits, read from
  the manifest's stored hashes (no test photo was opened). It is also more than 8 bits from every other training
  photo. `stage_dataset.py` refuses to stage if any training owner is in validation or test, or any training photo
  is within 8 bits of one (both counts: 0).

Each set's counts and licences are in `data/<set>/report.json`. The full per-photo manifests (licence, owner,
source URL) are outside git because of their size.

## Variants, controls and seed variance

We ran `plain` and `distil` first. Their false snakes on Open Images people, hands and ropes (7 to 8%) showed the
model had too few everyday negatives. So `distil_oi` added 4,940 Open Images photos and gave "other" a larger
share. We named it the final variant before it ran, and did not choose among variants after seeing its number.
`distil_oi_seed43` is the same recipe with another seed, to see how much the result moves between identical
trainings.

| Variant | Non-snakes called a snake | Snakes kept | AUROC (val) | Non-snake scores from |
|---|---:|---:|---:|---|
| plain | 9.67% (58) [7.3, 11.8] | 99.24% [99.03, 99.43] | 0.991 | laptop fp16 ONNX, app path |
| distil (BioCLIP teacher) | 8.33% (50) [6.0, 10.5] | 99.37% [99.17, 99.55] | 0.993 | laptop fp16 ONNX, app path |
| **distil_oi (final)** | **5.17% (31) [3.5, 7.0]** | **99.20% [98.88, 99.45]** | **0.995** | kernel fp32 (laptop fp16: also 31) |
| distil_oi_v2 (retry, round-two lookalikes) | 5.67% (34) [3.8, 7.3] | 99.34% [99.12, 99.52] | 0.997 | kernel fp32 (laptop fp16: 33) |
| distil_oi_seed43 (same recipe, seed 43) | 5.33% (32) [3.5, 7.0] | 99.29% [99.06, 99.51] | 0.996 | kernel fp32 |
| Control: training labels shuffled | 98.5% (591) [97.5, 99.5] | 98.90% | 0.360 | laptop fp16 ONNX, app path |
| Control: random score | 98.8% (593) | 98.83% | | |

Each cut was fitted on that variant's calibration snakes, and 99.01% of calibration snakes are kept in every row.
Distillation helped a little (58 false snakes to 50). The Open Images negatives helped most: Open Images false
snakes fell from 7.0% to 3.2%. Lookalikes stayed at 12 to 15%. Caecilians, eels and legless lizards are what is
left.

- **Seed variance.** Seed 43 gives 32 false snakes (5.33%) and 99.29% kept, against 31 and 99.20% for seed 42. The
  total is stable at about 5.2 to 5.3%, and it misses the mark both times. Which photos fail moves more than the
  total: with seed 43, eels fall to 5% from 17.5% and ropes rise to 6.7% from 1.7%.
- **Controls.** Both must fail, and both do. A model trained on shuffled labels calls 98.5% of non-snakes a snake
  at the 99% cut, and a random score calls 98.8%. So this evaluation can fail.

### The retry: `distil_oi_v2`

We fixed the recipe before the run: exactly `distil_oi` plus the round-two negatives below, with the cut fitted the
same way. One run, no further iterations, and `distil_oi` kept as the fallback. These negatives were chosen after
seeing `distil_oi`'s 31 false snakes (caecilians, eels, legless lizards, other animals). That was the second time
validation errors shaped the training data, so any validation number from this run is optimistic.

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
| Gymnophiona, caecilians (27880) | caecilian | **0**: round one already took every research-grade CC caecilian photo on iNaturalist |

That is 4,791 photos. 29% carry open licences (CC BY 1,009, CC0 274, CC BY-SA 126) and 71% non-commercial ones (CC
BY-NC 3,310, BY-NC-SA 72). No owner is in notebook 19's validation or test split. The staged dataset (18,769
non-snake training photos) has no training owner in validation or test and no photo within 8 pHash bits of one.

| Validation | `distil_oi_v2` (retry) | `distil_oi` (unchanged) | Mark |
|---|---:|---:|---:|
| Non-snakes called a snake | **5.67%** (34 of 600), 3.8% to 7.3% | 5.17% (31), 3.5% to 7.0% | at most 5% |
| Snakes kept | **99.34%** (17,037 of 17,151), 99.12% to 99.52% | 99.20%, 98.88% to 99.45% | at least 99% |
| Calibration snakes kept | 99.01% | 99.01% | |
| Cut on P(snake) | 0.107 | 0.205 | |
| AUROC, validation | 0.997 | 0.995 | |

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

Why it did not help: the extra lookalikes made the model separate snakes from non-snakes slightly better overall
(AUROC 0.995 to 0.997) and kept more sea snakes. But they also made it less sure about the hardest real snakes, so
the cut that keeps 99% of calibration snakes fell from 0.205 to 0.107. At that lower cut the lookalikes pass at the
same rate as before. The two models share 22 of their false snakes; 9 fail only for `distil_oi` and 12 only for the
retry. That is the same churn as between two seeds of `distil_oi` (31 against 32), so the retry is within noise.
`distil_oi` stays, because it makes fewer false snakes. The retry would win only if sea-snake recall were weighted
above the false-snake rate.

The retry ran for 72 minutes on 95,837 training photos, 0 missing. Its ONNX parity: fp32 against PyTorch at most
5e-6, fp16 at most 0.006 (40 photos, same top class). Kernel PyTorch against laptop fp16 ONNX through the app path:
1,000 snakes, at most 0.017, 0 flips; 600 non-snakes, at most 0.009, 1 flip (33 called a snake on the laptop, 34 in
the kernel).

## Parity and latency

For `distil_oi` the kernel scores every photo in PyTorch fp32 through the app's input path. The same photos then
went through the exported fp16 ONNX on the laptop (`scripts/app_inputs.py`, built from the app reference
pipeline's own functions):

| Check (`distil_oi`) | Photos | Largest difference in P(snake) | Decisions that flip at the cut |
|---|---:|---:|---:|
| fp32 ONNX against PyTorch, same input | 40 | 2e-6 | 0 |
| fp16 ONNX against PyTorch, same input | 40 | 0.006 | 0 |
| Kernel PyTorch against laptop fp16 ONNX, 500 calibration + 500 validation snakes | 1,000 | 0.007 | 0 |
| Kernel PyTorch against laptop fp16 ONNX, all validation non-snakes | 600 | 0.007 | 0 |

For `plain` and `distil` the kernel decoded with OpenCV and skipped the exposure fix. Their kernel-against-laptop
parity on snakes (distil, 1,000 photos) is at most 0.096, with 0 flips.

Latency (`results/latency.json`): laptop CPU, ONNX Runtime 1.31, batch 1, median of 100 runs. The laptop was busy
with other jobs (load average about 14), so every number is high and only the ratios mean much.

| Model | 1 thread | 2 threads |
|---|---:|---:|
| Snake check fp16 (256 px) | 23 ms | 13 ms |
| Snake check fp32 | 28 ms | 20 ms |
| A3 classifier fp32 (320 px) | 48 ms | 32 ms |
| U-Net segmenter fp32 | 34 ms | 23 ms |
| Detector | 7 ms | 7 ms |

An earlier run of the same architecture at lower load gave 17 ms (fp32, one thread), against 37 ms for A3.

## Runs

All six kernels ran on Vikas's Kaggle account, on private T4 kernels, reading the private dataset
`sg-m5-g4-train` (an anonymous request returns 403; we checked after each version). Version 1 of that dataset served plain, distil and shuffled. Version 2 added the Open Images
negatives and the 600 validation non-snakes (scored only, never trained on). Version 3 added the round-two
lookalikes.

| Kernel | Variant | Version | Run time |
|---|---|---|---|
| `sg-m5-g4-plain` | plain | 1 | 84 min |
| `sg-m5-g4-distil` | BioCLIP feature distillation | 1 | 90 min |
| `sg-m5-g4-shuffled` | control: training labels shuffled, 3 epochs | 1 | 41 min |
| `sg-m5-g4-distil-oi` | distil + Open Images negatives, other share 0.25 | 2 | 84 min |
| `sg-m5-g4-distil-oi-seed43` | the same, seed 43 | 1 | 73 min |
| `sg-m5-g4-distil-oi-v2` | the retry with round-two lookalikes | 1 | 72 min |

Version 1 of `distil-oi` started before dataset version 2 was ready, and it could not be cancelled from the command
line. Version 2 replaced its output, and we did not use it.

## Not checked

- **Validation is not fresh.** Five trained variants read the same 600 non-snakes, and so did our earlier BioCLIP
  and A3 probes. The training data was chosen from their errors twice: the Open Images negatives after `plain`, and
  the round-two lookalikes after `distil_oi`. Every validation number here is optimistic for that reason.
- **Not every snake went through the laptop ONNX path.** The laptop was too loaded to put all 25,150 snakes through
  it in time. 1,000 went through, with no decision changed.
- **Label quality of the validation non-snakes.** We looked at the 31 false snakes on one contact sheet. One "other
  animal" photo looks to us like a snake, and several are lizards with short legs. We removed or relabelled none.
- **Owner separation beyond notebook 19.** Non-snake owners were checked against notebook 19's validation and test
  owners only. iNaturalist logins and Open Images Flickr authors are different namespaces, so one person with
  accounts on both would not be caught.
- **Seed variance** rests on one extra seed (two runs: 31 and 32 false snakes).
- **The phone.** Phone latency and fp16 on Android's CPU provider are untested.
- **Licences.** 89% of the new iNaturalist negatives are CC BY-NC. That is fine for training a course model, and
  must be checked before any commercial use, as for the NC snake photos.

## Files

| Path | What it is |
|---|---|
| `results/variants/<variant>/` | Each variant's `train_config.json` (with the loss per epoch), `g4_eval.json` (every number above, by category and group), and, where exported, `export.json` and `parity_torch_vs_onnx.json` |
| `results/latency.json` | The latency table |
| `data/<set>/report.json` | Counts, owners and licences of each new negative set |
| `logs/g4_data_manifest_v3.json` | What the staged dataset version 3 holds, with the separation checks |
| `scripts/collect_negatives.py`, `collect_negatives_v2.py`, `collect_openimages.py` | Collect the new negatives |
| `scripts/stage_dataset.py` | Builds the Kaggle dataset and runs the separation checks |
| `scripts/kernel/train_g4.py`, `scripts/kernel/make_kernels.py` | The training kernel and the script that builds one per variant |
| `scripts/export_onnx.py`, `app_inputs.py`, `score_onnx.py`, `kernel_scores.py`, `evaluate.py`, `parity.py`, `time_onnx.py` | Export, scoring, the G4 rule, parity and latency |

The weights, ONNX files and per-photo score files sit in `runs/<variant>/` in the working copy and are not in git.
