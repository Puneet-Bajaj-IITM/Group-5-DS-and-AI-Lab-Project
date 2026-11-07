# Milestone 3: Model architecture and end-to-end pipeline

SerpentGuard AI · Group 5 · IIT Madras DS and AI Lab<br>
Team lead: Jayesh Bansal. Members: Vikas, Puneet Bajaj and Manas Rastogi.<br>
The signatures at the end are added at the team review.

## Summary

We chose a model for each stage of SerpentGuard, compared each one against baselines on a 16.4% training subset, and ran the whole chain from a photo or a clip to a verdict with WHO first-aid guidance. Every number comes from a saved run file; section 8 says how to reproduce it. Sections 1 to 6 follow the six Milestone 3 requirements: (1) architecture selection, (2) justification with measured comparisons, (3) baselines, subset, metrics and industry benchmarks, (4) tuning with plots, (5) the end-to-end setup and (6) the two diagrams. Sections 7 and 8 cover limitations and reproducibility.

**Results at a glance.**

| Item | Value |
|---|---|
| Candidate subset | 13,925 photos (16.4% of train), all 184 classes, 6,576 photographers |
| SnakeID | MobileViT-S (head macro-F1 0.256) with an ArcFace head (0.263, against 0.2546 for cross-entropy) |
| Front end | A crop to our fine-tuned YOLO26n's box lifts head macro-F1 from 0.257 to 0.311 and cuts false alarms from 0.410 to 0.279; segmenter masks add nothing |
| Tuning and final model | 24 random trials of 8 epochs; the best setting, retrained on the crop, reaches head macro-F1 0.3668, mean of 3 seeds that span 0.0014, against 0.311 for the default recipe on the same crop. We ship seed 1 |
| Against B1, scored the same way | Macro-F1 0.367 against 0.257, and 0.357 against 0.252 on photos uploaded from 2022, a lead that comes from coverage: B1 cannot name 122 of our 184 species. On the 62 species both know, B1 leads on top-1, 0.801 against 0.682 |
| Verdicts of the final model | WHO-listed snakes called dangerous 94.8% (95% interval 93.0 to 96.3%); called "not listed" 2.1%; false alarms 31.1% |
| Unknown species | Only 7.5% of unknown-pool photos flagged as unfamiliar (target 70%) |
| Response retrieval | BM25 shipped (Recall@3 0.47, against 0.40 for a fixed lookup); MiniLM-L6 (0.68) is the upgrade once the size budget allows |
| End to end | 100 validation photos and 5 clips through the final model, every node logged; 87.4 ms per photo on a laptop CPU, detector included |
| Robustness | With the snake blanked, 17.8% of the photos the final model gets right stay right, against 23.6% on the whole photo; Indian photos score 15.6 points lower on top-1 |
| Video | On 200 damaged synthetic clips, the 4-of-6 vote cuts verdict flips from 26.2 to 6.1 per clip; the clip verdict is right on 29.5%, against 58.5% on the clean photos |
| Size and speed | A 13.6 MB package with BM25, against 30 MB; 84.9 ms per photo (11.8 FPS) on one laptop CPU thread, against 10 FPS |

What we learned, good and bad:

- **Our lead over the best published model is coverage, not accuracy.** B1 cannot name 122 of our 184 species, and each counts as zero in its macro-F1; on the 62 species both models know, it leads by about 12 points of top-1. Ours is about 1% of its size and trained on 16% of our photos.
- **Cropping to the snake helps; segmentation does not**, and tuning added about as much again as the crop.
- **ArcFace helps a little and costs false alarms**: 42.2% against 36.4 to 37.9% for cross-entropy. Triplet loss, the 5× danger weight and logit-adjusted cross-entropy bought nothing.
- **The end-to-end check caught a train/serve skew.** After the fix it matches the training run's outputs on 98% of verdicts, up from 89%.
- **Voting steadies video verdicts, but damaged clips are hard**, and **full int8 breaks trained models**: on the final model it keeps the float32 top-1 species on only 8.0 to 9.3% of photos, so we ship fp16.
- **The final model sits just under the WHO recall target and just over the error target** (94.8% and 2.1%, against 0.95 and 2%), false alarms are about 3 in 10 and no decision rule we tried cuts them, and unknown-species detection does not work yet.

## 1. Model architecture selection

SerpentGuard runs offline on the phone and generates no text. A photo, or each frame of a clip, passes through five stages. SnakeID, a phone-sized network, turns each photo into a 256-number fingerprint and names the species by its closest known fingerprint, the way face recognition works. A decision layer turns its outputs into one of three verdicts, and the response step quotes fixed WHO passages with their page. Section 2 lists every alternative we tried.

| Stage | Input → output | Chosen (section) |
|---|---|---|
| Preprocess | photo → `(B, 3, 320, 320)`, notebook 06 unchanged | from Milestone 2 |
| Front end | photo → crop to the YOLO26n box plus 6%, or "no snake found" | YOLO26n box crop (2.2) |
| SnakeID | image → 256-d embedding; species (184), genus (69), family (13) and WHO (3) outputs | MobileViT-S, ArcFace head, tuned recipe (2.1, 4) |
| Decision | outputs → dangerous / not on the WHO list, still keep away / we don't know, with a reason | rules with thresholds fitted on a calibration set (2.3) |
| Response | verdict and top species → species card and WHO passages | BM25 retrieval plus the fixed first-aid lookup (2.4) |

**What counts as dangerous.** Our danger table (`data/m3/medical_importance.csv`) marks 50 of the 184 classes: 6 WHO Category 1, 10 WHO Category 2, 29 more vipers and elapids (every species in those families is venomous) and 5 *Rhabdophis* keelbacks, which the guideline names. Each row cites its guideline page. "Not listed" is always shown as "not on the WHO list, still keep away".

**One recipe, one rule.** Every run uses one recipe (AdamW at learning rate 3e-4, encoder at 0.1×, weight decay 0.05, cosine schedule after 1 warm-up epoch, 20 epochs, batch 64, 320 px input, notebook 06's augmentation) and changes only what it tests. We pick the highest validation macro-F1 among models whose raw WHO recall and species error on WHO-listed photos are no worse than cross-entropy's. Within noise, we pick the smaller, faster model with fewer parts.

## 2. Architecture justification

Each stage was chosen by that rule, on the 17,151 validation photos; the last column links the notebook with every result.

| Stage | Alternatives tried | Deciding score | Winner | Notebook |
|---|---|---|---|---|
| Encoder | MobileNetV4-S, EfficientNet-B0 (2 seeds), MobileViT-S | Head macro-F1 0.196; 0.231 and 0.237; 0.256 | MobileViT-S | [12](../../notebooks/12_training.ipynb) |
| Species head | Cross-entropy (3 seeds), ArcFace, cross-entropy + triplet, ArcFace + triplet, balanced sampler alone, logit-adjusted cross-entropy | Best macro-F1 0.2546, 0.263, 0.254, 0.257, 0.252, 0.250 | ArcFace | [12](../../notebooks/12_training.ipynb) |
| Genus and family heads | With and without | Head macro-F1 0.256 with, 0.255 without | Kept, for the fallback | [12](../../notebooks/12_training.ipynb) |
| Extra weight on missed WHO-listed snakes | 5× and 1× | Macro-F1 0.2546 at 5× (3 seeds), 0.263 at 1× | 1× | [12](../../notebooks/12_training.ipynb) |
| Segmenter | U-Net MobileNetV2, YOLO26n-seg, LR-ASPP MobileNetV3, mask head on SnakeID | IoU on human masks 0.837, 0.828, 0.787, 0.772 | U-Net, then dropped | [13](../../notebooks/13_cropping.ipynb) |
| Detector | YOLO26n fine-tuned, yolov8n-oiv7 as is | mAP@0.5 0.935, 0.514 | YOLO26n | [13](../../notebooks/13_cropping.ipynb) |
| What SnakeID sees | Whole photo, box crop, mask-box crop, background dimmed or removed, mask loss, mask pooling | Head macro-F1 0.257, 0.311, 0.312, 0.291, 0.259, 0.250, 0.259 | Box crop | [13](../../notebooks/13_cropping.ipynb) |
| Decision rule | Current rule, learned score, top 3 or top 5 species, two species and genus mixes, weak-signal variant | False alarms at the 0.95 target 0.311, 0.326, 0.394 and 0.397, 0.314 and 0.302, 0.162 (a variant at 0.842 recall) | Current rule | [15](../../notebooks/15_safety.ipynb) |
| Species ranking | Head, prototypes | Unknown-species AUROC on the final model 0.692, 0.616 | Head | [15](../../notebooks/15_safety.ipynb) |
| Retriever | Fixed lookup, BM25, MiniLM-L6, MiniLM-L3 | Recall@3 0.40, 0.47, 0.68, 0.53 | BM25, for size | [16](../../notebooks/16_app.ipynb) |
| Export precision | fp16, int8 weights, int8 arithmetic (static, dynamic) | Top-1 agreement with fp32 100.0%, 92.3%, 9.3% and 8.0% | fp16 | [16](../../notebooks/16_app.ipynb) |
| Hyperparameters | Default and the 3 best of 24 trials, 3 seeds each | Head macro-F1 on the crop 0.311; 0.3668, 0.3630, 0.3551 | Trial ranked 1 | [14](../../notebooks/14_tuning.ipynb) |

**Noise floor.** Three seeds of the encoder winner give head macro-F1 0.2560, 0.2541 and 0.2535: a spread of 0.0025. Twice that, 0.005, is the smallest difference we call real. The final recipe's seeds span 0.0014, but the second-best tuned setting spans 0.0137, so the floor depends on the recipe.

**Design changes.** SnakeID keeps MobileViT-S's ImageNet weights and swaps its classifier for an embedding with ArcFace, genus, family and WHO heads (5.2). YOLO26n is fine-tuned to one snake class, and its box is widened by 6% before the crop.

<img src="assets/m3_macro_f1_by_run.png" alt="Macro-F1 for each run, head and balanced prototypes, with the seed spread shaded" width="680">

### 2.1 SnakeID: encoder, head and losses

**Encoder.** MobileViT-S leads the mean of EfficientNet-B0's two seeds by 0.022, four times their own 0.0054 difference, and has the best raw WHO recall, 0.717. Its attention layers see the whole snake at once, which suits long thin bodies. The cost is size: at 4.94M parameters it is the largest of the three, and full int8 breaks it, so it ships in fp16 (2.4). Manna et al. found it the weakest of four models on venomous recall [5]; this run reaches 0.957 at the verdict. MobileNetV4-S peaks at epoch 16, while the others still gain a little at epoch 20.

**Species head.** ArcFace's best score, 0.263 with balanced prototypes, beats cross-entropy's best (mean of 3 seeds, 0.2546) by 0.0082 and raises raw WHO recall from 0.716 to 0.741. An embedding head also lets us add species from a few photos (5.3). The costs: its verdicts raise false alarms to 0.422, against 0.364 to 0.379, and its scores are overconfident until temperature scaling (ECE 0.193, then 0.036). Triplet loss added nothing (cross-entropy + triplet scores 0.254, ArcFace + triplet 0.257), nor did a balanced sampler, and logit-adjusted cross-entropy cut top-1 to 37.3%. The final model ranks species with the head, which scores 0.367 against 0.356 for prototypes and separates unknown species better (AUROC 0.696 against 0.622 across its seeds).

**Genus and family heads; the danger weight.** Without genus and family heads, head macro-F1 moves from 0.256 to 0.255, inside the noise, so they stay for the fallback, which shows a genus on 17.5% of "we don't know" photos and is right 67.8% of those times. Summing species probabilities predicts the genus better than the head (0.612 against 0.591), so Milestone 4 will use the sum. The 5× weight on "listed predicted as not listed" bought nothing: without it head macro-F1 is 0.263 and the verdicts are no worse, and the threshold fit switched the WHO head off in 7 of 11 runs because the species head alone reached the target.

### 2.2 Finding the snake first

<img src="assets/m3_frontends.png" alt="IoU of four segmenters and precision, recall and mAP of two detectors" width="720">

**Segmenters and detectors.** The U-Net wins by 0.009 IoU, outside the 0.005 tie band, though YOLO26n-seg is half its size. Our fine-tuned YOLO26n reaches mAP@0.5 0.935 against 0.514 off the shelf, and boxes 99.95% of validation photos above confidence 0.05; only 93.2% get a usable mask.

<img src="assets/m3_fe_ablation.png" alt="Macro-F1 of the seven front-end variants, head and balanced prototypes" width="680">

**The box crop wins.** Cropping to the detector's box raises head macro-F1 from 0.257 to 0.311 (+0.053, about ten times the floor) and top-1 from 53.1% to 58.3%. Cropping to the mask's box ties (0.002 apart) but needs the 6.6M-parameter U-Net, while the box needs only the 2.5M-parameter YOLO26n, which the pipeline runs anyway. Dimming the background costs 0.020 and removing it 0.052, so the scenery next to the snake seems to help. A mask loss or mask-guided pooling leaves the whole photo where it was (0.250 and 0.259 against 0.257). So the U-Net leaves the pipeline.

### 2.3 From outputs to a verdict

Thresholds are fitted on the calibration set (566 WHO-listed photos), never on validation. A photo is **dangerous** if the probability on WHO-listed species reaches a threshold, the top species is in the danger table, or vipers and elapids together reach 0.5; the threshold is the largest that keeps WHO recall at 0.95 or more. It is **we don't know** if the top species is unsure or far from every prototype, and never "not listed" for a genus with a WHO-listed species.

<img src="assets/m3_verdicts_by_run.png" alt="WHO-listed photos called dangerous and false alarms by run, with 95% intervals" width="720">

On the ArcFace model the rules turn a 25.9% species error on WHO-listed snakes into 2.7% wrong verdicts, at the price of about 4 in 10 false alarms. The box crop cuts false alarms from 0.410 to 0.279, the largest drop of any change, but its WHO recall on validation falls to 0.922, although the same thresholds reach 0.951 on calibration. So the final model's threshold is set where the lower end of the 95% bootstrap interval of calibration recall reaches 0.95, which also cuts its "not listed" rate from 0.028 to 0.021. The crop answers "we don't know" more often (12.0% against 6.7%), partly because 151 validation photos have no box above the no-snake threshold of 0.102.

**No other rule cuts false alarms.** On the final model, none of 6 other rules in the table beats the current one at the same 0.95 WHO recall target. The best, a weighted species and genus mix, gives 0.302 false alarms against 0.311, inside the current rule's interval. A variant that turns weak-signal calls into "we don't know (keep away)" cuts false alarms to 0.162, but calls only 0.842 of WHO-listed snakes dangerous and answers "we don't know" for 22.8% of photos, which changes what the verdict promises.

<img src="assets/m3_decision_rules_H1_top1_s1.png" alt="Validation false alarms against WHO recall for the current rule and its alternatives on the final model, each fitted at three recall targets" width="680">

Moving the current rule's target traces one curve: 0.908 recall comes with 0.226 false alarms (lower-bound fit at 0.90), 0.948 with 0.311 (lower-bound fit at 0.95) and 0.975 with 0.415 (lower-bound fit at 0.975). Every rule but the top-species ones lies near it. We read this as the classifier confusing lookalikes, which only a better model can fix.

### 2.4 Response, size and speed

**Retrieval.** On 60 draft questions over 168 WHO passages and 184 species cards, MiniLM-L6 beats BM25 on 20 questions and loses on 7 (sign test p = 0.019), all on paraphrases (11 to 0), of which BM25 finds only 0.05. MiniLM-L3 is not better than BM25 (p = 0.52). We ship BM25 because MiniLM-L6 takes the package to 36.24 MB and MiniLM-L3 to 30.99 MB, over the 30 MB budget. The fixed lookup always shows the first-aid passages, whatever retrieval returns.

**Export.** Full int8 breaks trained models: on 300 validation photos, the final model in int8 keeps the float32 top-1 species on only 9.3% (static) and 8.0% (dynamic). Trained weights have activation outliers (a SiLU output peaking near 100 where most values are under 1) that one int8 range per tensor rounds away; the untrained models of our first export tests hid this. In fp16 the final model loses nothing: its 10.67 MB file gets the same top-1 as float32, with an embedding cosine of 1.0000.

**The package fits.** fp16 SnakeID, the int8-weight detector and BM25 come to 13.6 MB, and the two networks take 84.9 ms per photo (11.8 FPS) on one laptop CPU thread, against the target of 10 FPS. On this CPU fp16 is slower than fp32 (1.5 times on one thread), but download size is the phone's constraint. The detector keeps int8 weights because its box matches float32's on 99.7% of the photos where float32 finds a snake.

### 2.5 Trade-offs

We traded size for accuracy (the largest encoder, in fp16), false alarms for WHO recall (ArcFace, the danger rules and the lower-bound fit), and retrieval quality for size (BM25 over MiniLM-L6).

## 3. Baselines, subset and metrics

### 3.1 Candidate subset

**Rule.** For each class we take 15% of its training photos, but at least 20 (all of them if the class has fewer). Photographers take turns, so one person can't fill a small class. The seed is 42. [Notebook 10](../../notebooks/10_subset.ipynb) builds it; every run records the subset's SHA-256, which starts `40196d19a68e`.

| Option | Photos | Share of train | Median class | Indian | WHO-listed | JS divergence of class mix |
|---|---:|---:|---:|---:|---:|---:|
| Full train | 85,067 | 100% | | 37.4% | 12.9% | |
| **15%, floor 20 (chosen)** | **13,925** | **16.4%** | **20** | **39.7%** | **12.1%** | **0.012** |
| 15%, floor 30 | 14,773 | 17.4% | 30 | | 11.7% | 0.021 |
| 15%, floor 40 | 15,615 | 18.4% | 40 | | 11.2% | 0.028 |
| 20%, floor 20 | 17,964 | 21.1% | 24 | | 12.4% | 0.007 |

**Why it is representative.** It sits inside the 15 to 30% the TA asked for and keeps every class and 6,576 of the 11,272 training photographers. It stays within 2.3 points of full train on the Indian share and the WHO mix. The floor of 20 lifts rare classes on purpose: tier 2 holds 19.9% of the subset against 12.7% of train. Higher floors pull the class mix further from train, and a 20% subset has 29% more photos for a closer mix.

**Evaluation sets.** After notebook 06's caps (50 photos per photographer per class, 1,000 per class), 13,611 photos remain for training. Thresholds are fitted on a calibration set of 7,999 training photos in 111 classes, outside the subset and from photographers not in it. Model choices use the 17,151 validation photos. `test` and `test_extra` stay locked, and a test enforces it.

### 3.2 Evaluation metrics

Our classes are long-tailed and our mistakes are not equal: calling a krait "not listed" is far worse than calling a rat snake dangerous.

- **Macro-F1**, the main score: the mean F1 over the 158 classes with at least 5 validation photos (the rest are not measurable, never zero), so rare species count as much as common ones; tier 1 and tier 2 cover classes with 400 or more photos / 20 to 399. We also report top-1, top-3 and raw WHO recall, the share of WHO-listed photos whose top species is WHO-listed.
- **Verdicts against the Milestone 1 targets:** WHO-listed called dangerous, target at least 0.95; WHO-listed called "not listed", target at most 2%; and false alarms, the share of snakes neither WHO-listed nor a viper or elapid that are called dangerous.
- **Confidence:** ECE over 15 equal-width bins, target at most 0.05; the error among the 80% most confident photos, target at most 5%; unknown-species AUROC and the share flagged, target 70%.
- **The rest:** IoU, mAP@0.5, Recall@3, and size and speed, with targets 25 to 30 MB and at least 10 FPS on a laptop CPU.

Verdict rates carry 95% intervals from 1,000 resamples of photographers.

### 3.3 Baselines before training on our photos

| ID | Baseline | Macro-F1 | Top-1 | WHO recall (raw) | Note |
|---|---|---:|---:|---:|---|
| B0 | Predict class frequency | 0.001 | 9.5% | 0.000 | The floor |
| B1 | Bolon et al. 2022 global ViT (SnakeCLEF 2021 data), labels mapped to ours | 0.257 | 68.3% | 0.778 | Knows 62 of 184 classes (85% of photos). 0.252 on photos uploaded from 2022, after its training data ends |
| B2 | BioCLIP zero-shot with species names | 0.187 | 35.4% | 0.437 | Upper bound: its training data may include our photos |
| B3 | Each encoder frozen (ImageNet), nearest prototype from the subset | 0.140 / 0.092 / 0.135 | 22.9% / 15.7% / 21.7% | 0.451 / 0.328 / 0.443 | EfficientNet-B0 / MobileNetV4-S / MobileViT-S |
| B4 | DINOv2 ViT-S/14 frozen, linear probe (prototypes) | 0.105 (0.078) | 18.0% | 0.353 | The large reference |
| B5 | yolov8n-oiv7 as is, boxes on Open Images | precision 0.82, recall 0.22, mAP@0.5 0.51 | | | Finds few snakes off the shelf |
| R0 | Response lookup by verdict | Recall@3 0.40 | | | The response floor |

<img src="assets/m3_baselines.png" alt="Macro-F1 of the baselines against our trained models" width="640">

Training on 16% of our photos takes MobileViT-S from 0.135 (frozen) to 0.256. B1 may have seen some of our validation photos in training, so its score is an upper bound. The Grounding DINO + SAM teacher moved to Milestone 4, since Milestone 2 already found about 84% of its masks good by eye. [Notebook 11](../../notebooks/11_baselines.ipynb) runs every baseline.

### 3.4 Our final model against B1

| On our validation set | Ours (final model) | B1 |
|---|---:|---:|
| Macro-F1 | **0.367** | 0.257 |
| Tier 1 (400+ photos) / tier 2 (20 to 399) macro-F1 | 0.576 / **0.252** | **0.604** / 0.066 |
| Top-1 | 62.2% | **68.3%** |
| Top-3 | 76.2% | **77.1%** |
| WHO recall (raw) | 0.768 | **0.778** |
| Macro-F1, photos uploaded from 2022 (13,947) | **0.357** | 0.252 |
| Size | **10.7 MB** in fp16 | about 1.13 GB |

Both models are scored with the function B1 was scored with ([notebook 11](../../notebooks/11_baselines.ipynb)); it reproduces our run's own macro-F1 exactly, and with shuffled labels our model scores 0.006. Ours leads on macro-F1 by 0.110, and by 0.105 on photos uploaded from 2022, which B1 cannot have trained on. B1 wins on top-1 by about 6 points, on top-3 by about 1 point and on raw WHO recall. The macro-F1 lead is a coverage effect: B1 cannot name 122 of our classes, which hold 15% of the photos, and each of them scores zero in its macro-F1.

**On the 62 species both models know, B1 is clearly better.** Its top-1 is 0.801 against our 0.682 on 14,610 photos, still 10 points ahead when ours may answer only with those 62 species (0.701), and its mean recall is 0.755 against 0.585. The same holds on photos uploaded from 2022 (top-1 0.788 against 0.679). Ours is about 1% of B1's size and trained on 16% of our photos.

### 3.5 Industry benchmarks (other datasets, never directly comparable)

| Source | Task and data | Reported result |
|---|---|---|
| Bolon et al. 2022 [2] | 772 species, 23,673 test photos | 96.0% top-1, 92.2% macro-F1, 99.0% genus top-1 |
| Durso et al. 2021 [3] | 45 North American species | 84% accuracy, F1 0.83; 72% and 0.73 on expert-labelled photos |
| Manna et al. 2026 [5] | Indian venomous species, four models | MobileViT-S had the lowest venomous recall, 85.98% |
| Najm et al. 2025 [6] | YOLOv8-Nano, 10 species | Precision 0.88, recall 0.91, mAP@0.5 0.892, 142 FPS |
| Naresh et al. 2024 [7] | YOLO snake detection | 87% precision |
| torchvision, Ultralytics [8, 9] | COCO, general objects | LR-ASPP MobileNetV3 mean IoU 57.9; DeepLabV3 MobileNetV3 60.3; YOLO26n-seg mask mAP 33.9 |
| SnakeCLEF 2021 winner, BME-TMIT [1] | 772 training species, scored per country | Country-averaged macro F1c 0.903; species macro-F1 0.855 |
| SnakeCLEF 2023 winner, word2vector [10] | 1,784 training species; private test of 182 species from 8 countries | Track 1 score 91.43% (91.31% in the same paper's discussion); macro-F1 53.58%, against 61.39% for the round's best; India subset Track 1 89.74% |
| SnakeCLEF 2024 winner, upupup [4] | 1,784 training species; private test of 199 species from 12 countries | Track 1 score 83.6%; India subset 81.16%; macro-F1 not reported |
| DINOv2, linear probe on ViT-g/14 [14] | iNaturalist 2018 and 2021, all species | Top-1 81.6% and 85.7% |

None of these uses our 184 classes, our split or our metric, and SnakeCLEF's Track 1 score weighs venomous mistakes. The nearest number to ours is the 2023 winner's 53.58% macro-F1, over a far larger label set. Values were read from the cited sources on 5 November 2026.

## 4. Hyperparameter tuning

**Search.** We tuned four parameters: learning rate (1e-4 to 1e-3, log scale), weight decay (1e-4 to 0.1, log scale), ArcFace margin (0.1 to 0.5) and augmentation strength (0.5 to 1.5 times notebook 06's). Random search ran 24 trials of 8 epochs, as 6 slices of 4 on three Kaggle accounts merged into one Optuna study: the slices run in parallel, and with pruning off every trial is a clean sample for the importance plot. The search ran on the whole photo, alongside the front-end comparison; we then retrained the top 3 on the box crop for 20 epochs, with seeds 1, 2 and 42 each ([notebook 14](../../notebooks/14_tuning.ipynb)). Trials scored 0.140 to 0.260 (median 0.192), all still improving at their last epoch, in 15.7 GPU-hours.

| Setting | Learning rate | Weight decay | Augmentation | ArcFace margin | Search, 8 epochs | Retrain on the crop, mean (seed range) | False alarms, mean (seed range) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default recipe | 3e-4 | 0.05 | 1.0 | 0.3 | 0.220 at epoch 8 of 20 | 0.311 (seed 42 only) | |
| Trial ranked 1 | 9.5e-4 | 1.6e-3 | 0.75 | 0.356 | 0.260 | **0.3668** (0.3660 to 0.3674) | 0.322 (0.311 to 0.334) |
| Trial ranked 2 | 9.4e-4 | 1.5e-2 | 0.91 | 0.346 | 0.251 | 0.3630 (0.3563 to 0.3701) | 0.314 (0.299 to 0.336) |
| Trial ranked 3 | 5.8e-4 | 4.6e-3 | 0.56 | 0.391 | 0.248 | 0.3551 (0.3528 to 0.3580) | 0.331 (0.319 to 0.348) |

<img src="assets/h1_history.png" alt="Best validation macro-F1 by trial in the tuning search" width="560">

<img src="assets/h1_parallel.png" alt="Parallel coordinates of the four tuned parameters and macro-F1, one line per trial" width="640">

<img src="assets/h1_importances.png" alt="Parameter importances from fANOVA over the 24 trials" width="560">

**Reading the search.** The learning rate matters most, and the two best trials sit at the top of its range. The default's 0.220 at epoch 8 is unfair to it, since its learning rate was still high on a 20-epoch schedule; the retrain decides. Trial 2 is only 0.0038 behind trial 1, inside the 0.005 floor. Neither false alarms (0.322 against 0.314, a gap of 0.008, smaller than either setting's own range across seeds) nor raw WHO recall (0.766 against 0.772; trial 2 is higher on all 3 seeds, but both call 0.949 of WHO-listed photos dangerous) separates them, so stability decides: trial 1's seeds span 0.0014 of head macro-F1, trial 2's 0.0137.

<img src="assets/m3_retrain_curves.png" alt="Validation head macro-F1 by epoch for the default and the tuned recipe on the box crop" width="600">

**The final model** is trial 1 on the box crop (learning rate 9.5e-4, weight decay 1.6e-3, augmentation strength 0.75 and ArcFace margin 0.356); we ship its median seed, seed 1. Against the default on the same crop, head macro-F1 rises by 0.056 on average over the 3 seeds, most on rare classes (tier 2 from 0.189 to 0.252), and leads at all 10 scored epochs. The default was still rising at epoch 20, so a longer schedule might close part of the gap. At the verdict, only 1 of the 3 seeds reaches the 0.95 target on validation.

## 5. End-to-end setup

### 5.1 Data flow and notebooks

A photo is read as RGB, its exposure fixed, letterboxed to 320 × 320 and normalised with ImageNet statistics (notebook 06). SnakeID sees the detector's box widened by 6%, or the app says "no snake found" if no box passes the no-snake threshold of 0.102. A clip is read at most 10 frames a second, newest frame first, and the screen shows a 4-of-6 vote over the frame verdicts. Each step is one notebook, saved with the outputs of its run:

| Notebook | What it does (section) | Runs on |
|---|---|---|
| [10_subset](../../notebooks/10_subset.ipynb) | the candidate subset and calibration set (3.1) | local CPU |
| [11_baselines](../../notebooks/11_baselines.ipynb) | baselines, few-shot enrolment and the comparison with B1 (3.3, 3.4, 5.3) | Kaggle T4 GPU |
| [12_training](../../notebooks/12_training.ipynb) | trains SnakeID; every training run in one table (2.1, 4) | Kaggle T4 GPU |
| [13_cropping](../../notebooks/13_cropping.ipynb) | segmenters, detectors, and crop against mask (2.2) | Kaggle T4 GPU ×2 |
| [14_tuning](../../notebooks/14_tuning.ipynb) | the search and the retrain of the best three (4) | Kaggle T4 GPU ×2 |
| [15_safety](../../notebooks/15_safety.ipynb) | verdicts, decision rules, the blanked-snake test, unknown species and video (2.3, 5.3) | Kaggle T4 GPU |
| [16_app](../../notebooks/16_app.ipynb) | the whole app, the video demo, retrieval, size and speed (2.4, 5.3, 5.4) | local CPU |

### 5.2 Model, training and validation

**Model.** SnakeID is MobileViT-S [12] without its classifier. Its averaged last feature map goes through a linear layer with batch normalisation to the 256-number embedding, which is L2-normalised. The ArcFace head [11] compares it with one learned direction per species, for all 184 classes, and scales the cosine by s = 30; the genus (69), family (13) and WHO (3) heads read the embedding before normalisation. Prototypes, each species' mean embedding, give a second way to name the species.

**Training and validation.** Notebook 12 trains each run on a Kaggle T4 GPU from a printed settings dict, on the subset after notebook 06's caps (13,611 photos). It follows the recipe of section 1 with gradient clipping at norm 5, scoring validation every 2 epochs and keeping the best epoch. The first runs' GPUs spent 76 to 89% of their time waiting for photos; reading every file once before training cut an epoch from 423 s to 194 s for MobileNetV4-S (189 s for MobileViT-S). Milestone 2's folds are grouped by photographer, so no photographer's photos sit on both sides of the split (3.1).

### 5.3 Evaluation

The final model's verdicts on validation, with species ranked by the head (shipped) or by prototypes:

| On validation (17,151 photos) | Head (shipped) | Prototypes | Milestone 1 target |
|---|---:|---:|---:|
| WHO-listed called dangerous | 0.948 (0.930 to 0.963) | 0.948 (0.930 to 0.964) | ≥ 0.95 |
| WHO-listed called "not listed", all photos | 0.021 | 0.015 | ≤ 2% |
| Raw top-1 species error on WHO-listed photos | 0.232 | 0.299 | |
| False alarms (neither WHO-listed nor a viper or elapid) | 0.311 (0.292 to 0.331) | 0.320 (0.301 to 0.340) | |
| "We don't know" | 9.5% | 10.9% | |
| ECE before / after temperature | 0.211 / 0.030 | 0.178 / 0.015 | ≤ 0.05 |
| Error at 80% coverage | 0.268 | 0.326 | ≤ 5% |
| Unknown-pool photos flagged (AUROC of the top softmax probability) | 7.5% (0.692) | 7.5% (0.616) | 70% |

The crop and the tuning cut false alarms from the ArcFace model's 0.422 to 0.311. WHO recall is 0.948 on validation, just under 0.95, although the fit's lower bound on calibration reaches 0.952. Ranking by prototypes meets the "not listed" target (0.015) but raises false alarms and separates unknown species worse, so the head stays. A photo quality check changes reason labels only: 202 "we don't know" photos now say "the photo is dark or blurry".

**End-to-end check** ([notebook 16](../../notebooks/16_app.ipynb)). One photo from each of 100 classes and 5 clips went through the full pipeline on a laptop CPU. It gives the same verdict as the training run's saved outputs on 98% of photos and the same top-1 species on 98%, while a control pairing each photo with the next photo's outputs agrees on 37% and 1%. All 12 WHO-listed photos were called dangerous; one other dangerous snake, a mountain pit viper, got "we don't know", and 21 of the 74 photos outside the danger table were false alarms. Of the 5 clips, 4 ended "dangerous", 3 of them false alarms, and 1 ended "we don't know" because the frames never agreed.

**A train/serve skew, found and fixed.** The first run matched the saved outputs on only 0.89 of verdicts. The app padded a lone photo to its smallest rectangle, while training batches were padded to a square, so the detector's boxes drifted (median IoU 0.944 with the training boxes, at worst 0.215). Every call now pads to the same square (median IoU 0.999, worst 0.963); the numbers above are from the rerun.

**Video** ([notebook 15](../../notebooks/15_safety.ipynb)). We made 200 synthetic clips from validation photos with the Milestone 1 recipe (10 s at 15 FPS, camera motion, blur, low light, noise, and leaves over 30 to 70% of the snake) and ran them on the laptop's schedule, which skips 21.3% of frames. The 4-of-6 vote cuts verdict flips from 26.2 to 6.1 per clip, and with the vote 3 of the 51 dangerous clips ever show "not listed", against 15 frame by frame, and with every method 1 dangerous clip ends on "not listed". Still, only 40.5% of clips meet the target of at most 1 flip, and the voted verdict is right on 29.5% of clips against 58.5% for the clean photos. The damage is deliberate, and real snake videos are still untested.

**Snake or scenery?** ([notebook 15](../../notebooks/15_safety.ipynb)) On the 15,977 validation photos with a predicted mask, we blanked the snake to grey, or moved it onto another class's scenery; untouched, both models reproduce their saved top-1 species on 99.9% of photos. Of the photos the final model gets right, 17.8% stay right with the snake blanked, against 23.6% for the whole-photo model, and with swapped scenery it keeps its own class on 46.0% of photos against 32.4%. The crop leans less on the scenery, partly by design, but not on nothing. With the snake blanked, fewer dangerous snakes turn "not listed" on the crop (336 against 516 photos). Indian photos are harder by 15.6 points of top-1, dark and blurry ones are not, and classes with at most 10 photographers gained nothing (22.7% top-1 for both models).

**Unknown species** ([notebook 15](../../notebooks/15_safety.ipynb)). On 451 photos of 49 species outside our classes, no signal separates them well: AUROC 0.557 to 0.654 on the ArcFace model, and only 7.3% flagged at a threshold that keeps 95% of known calibration photos, against the 70% target. The final model is no better. The danger rules still catch the two WHO-listed species in the pool: the final model calls *Bungarus sindanus* dangerous in 13 of 16 photos and *Naja sagittifera* in 3 of 3.

**New species from a few photos** ([notebook 11](../../notebooks/11_baselines.ipynb)). Holding 10 tier 2 species out of training did not hurt the rest (head macro-F1 0.260 against the whole-photo model's 0.257). Enrolling them as new prototypes needs no retraining and leaves known species almost untouched (top-1 0.434 before, 0.427 after). New-species top-1 grows from 0.039 with 1 photo to 0.262 with 10, nearly as well on other photographers' photos, but at 10 photos, 69.8% of their photos are still taken for a known species. Frozen DINOv2 prototypes, on a different class set, reach 0.301 from 10 photos.

### 5.4 What the app returns

For each photo the app shows a verdict with its reason, the top 3 species, the likeliest genus when unsure, the species card and the WHO first-aid passages with their page. Two examples from the end-to-end check:

| Photo of | Screen | Top 3 | Note |
|---|---|---|---|
| *Ovophis monticola* (mountain pit viper) | we don't know: unclear between species | *Boiga multomaculata* 0.35, *Ovophis monticola* 0.19, *Eryx conicus* 0.10 | **missed danger**: a cat snake on top |
| *Boiga siamensis* (cat snake) | dangerous: WHO Category 1 | *Daboia russelii* 0.33, *Naja naja* 0.32, *Boiga multomaculata* 0.12 | **false alarm**: a lookalike |

The category shown is the top species': a coral snake photo read "WHO Category 2" because a banded krait was on top. A venomous top species that is not WHO-listed reads "venomous family, not on WHO's priority list" (7 photos).

**Video demo.** [Notebook 16](../../notebooks/16_app.ipynb) also turns a clip into an annotated video and an insights file. With no real snake video yet, the demos film validation photos. A common krait clip shows "dangerous: WHO Category 1" from 0.4 s, top species in 100% of the 101 frames read; on a *Coelognathus helena* clip the vote lets through 7 of 20 frame-by-frame flips; on a *Dendrelaphis subocularis* clip the top species wanders, but the screen shows its genus at 0.63. Frames took 77.8 to 81.6 ms on the laptop. The videos are not committed, as they use other people's photos.

## 6. Architecture visualization

**The pipeline.** One photo, or each kept frame of a clip, runs from left to right; the stage table in section 1 names the part chosen at each stage.

<img src="assets/m3_pipeline.png" alt="Pipeline from a photo or clip through preprocessing, front end, SnakeID, decision layer and response" width="720">

**SnakeID.** The final model on the box crop, with the shape of each output.

<img src="assets/m3_snakeid.png" alt="SnakeID: the 320 px crop, MobileViT-S encoder, 256-number embedding, ArcFace species head with prototypes, genus, family and WHO heads, and the decision layer" width="720">

## 7. Limitations and Milestone 4

- **The verdicts sit at the edge of the targets**, partly because thresholds fitted on calibration photos lose recall on validation (2.3). Thresholds come from training photos (73 classes have none) and are frozen before any test run in Milestone 5.
- **Unknown detection fails** (7.5% flagged against 70% on the final model), and **false alarms are high**: about 3 in 10 for the final model. Milestone 4 tries cross-photographer positives and several prototypes per species.
- **Caveats.** The blanked-snake shortcut rate is an upper bound, since predicted masks can miss snake pixels; video stability counts a stable "we don't know" as stable; the segmenters had different augmentation; the video tests film photos; the laptop times are upper bounds and no phone is measured yet; and the fp16 package stores each weight in twice the bits of int8.
- **The search ran on the whole photo**, so a search on the crop might do better, and three seeds per recipe is a small sample.
- **Software.** Kaggle ran torch 2.11.0 and our laptop 2.15.0. onnxruntime and ultralytics attempt network calls on a desktop, and a switched-off signal is stored as `Infinity`; the Android build in Milestone 4 must handle both.
- **WHO text licence.** The guideline's copyright page reads "© World Health Organization 2016" and "All rights reserved". We keep only passage ids, page numbers and character offsets, extract the text from the PDF at build time and never commit it. Shipping it in the app needs WHO's permission, a team decision for Milestone 4.
- **Draft questions.** The 60 retrieval questions are drafts until the team checks them.

**Milestone 4:** train the final recipe on the full training set for longer, choose the retriever within the size budget, build the Android package, and run the 200-photo hand check promised in Milestone 2.

## 8. Reproducibility

The notebooks in 5.1 reproduce every experiment. Each run's outputs stay in its private Kaggle output, which this report quotes. We ran the training notebook once per configuration; notebooks 12 and 14 collect those runs in tables rather than rerun them. The code is in `src/serpentguard/`; `requirements-m3.txt` pins timm 1.0.31, albumentations 2.0.8, scikit-learn 1.9.1, segmentation-models-pytorch 0.6.0, ultralytics 8.4.202 and onnxruntime 1.31.1. Seeds are 42 for the subset and default runs, 1 and 2 for the noise floor, and 1, 2 and 42 for the retrain.

## References

1. [Overview of SnakeCLEF 2021](https://ceur-ws.org/Vol-2936/paper-125.pdf). CLEF 2021 Working Notes, CEUR-WS Vol-2936, paper 125. Challenge page: [AIcrowd](https://www.aicrowd.com/challenges/snakeclef2021-snake-species-identification-challenge).
2. Bolon I. et al. (2022). [An artificial intelligence model to identify snakes from across the world](https://journals.plos.org/plosntds/article?id=10.1371/journal.pntd.0010647). *PLOS Neglected Tropical Diseases* 16(8): e0010647.
3. Durso A. M. et al. (2021). [Supervised learning computer vision benchmark for snake species identification from photographs](https://pmc.ncbi.nlm.nih.gov/articles/PMC8093445/). *Frontiers in Artificial Intelligence*.
4. Picek L., Hrúz M., Durso A. M. (2024). [Overview of SnakeCLEF 2024](https://ceur-ws.org/Vol-3740/paper-188.pdf). CLEF 2024 Working Notes, CEUR-WS Vol-3740, paper 188.
5. Manna I. I. A. et al. (2026). [Indian venomous snake classification with explainable AI](https://doi.org/10.1371/journal.pntd.0014147). *PLOS Neglected Tropical Diseases* 20(6): e0014147.
6. Najm H. et al. (2025). [Real-time detection of multiple snake species using YOLOv8-Nano](https://doi.org/10.18280/isi.301025). *Ingénierie des Systèmes d'Information* 30(10).
7. Naresh E. et al. (2024). [A novel framework for detection of harmful snakes using YOLO](https://doi.org/10.1007/s42979-023-02366-z). *SN Computer Science*.
8. PyTorch, [torchvision pretrained models](https://docs.pytorch.org/vision/stable/models.html), checked 31 October 2026.
9. Ultralytics, [segmentation models](https://docs.ultralytics.com/tasks/segment/), checked 31 October 2026.
10. [Overview of SnakeCLEF 2023](https://ceur-ws.org/Vol-3497/paper-168.pdf). LifeCLEF 2023 Working Notes, CEUR-WS Vol-3497, paper 168.
11. Deng J. et al. (2019). [ArcFace: additive angular margin loss for deep face recognition](https://arxiv.org/abs/1801.07698). CVPR.
12. Mehta S., Rastegari M. (2022). [MobileViT](https://arxiv.org/abs/2110.02178). ICLR.
13. World Health Organization, Regional Office for South-East Asia (2016). [Guidelines for the Management of Snakebites, 2nd edition](https://cdn.who.int/media/docs/default-source/searo/india/health-topic-pdf/who-guidance-on-management-of-snakebites.pdf). Cited by page; the text is not reproduced here.
14. Oquab M. et al. (2023). [DINOv2: learning robust visual features without supervision](https://arxiv.org/abs/2304.07193). arXiv:2304.07193.

## Team contributions

| Member | Work this milestone |
|---|---|
| Jayesh Bansal | PENDING |
| Vikas | Presentation (PPT) |
| Puneet Bajaj | Ablations |
| Manas Rastogi | PENDING |

<div style="page-break-before: always"></div>

# Team review and acknowledgment

Each team member signs one row after the review. By signing, each member confirms that they have reviewed this document, understand its contents and agree with the submission.

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
| Jayesh Bansal | 24f100185 | ![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0) | 07/11/2026 |
| Vikas | 23f3001800 | Vikas | 05/11/2026 |
| Puneet Bajaj | 23f3002416 | Puneet Bajaj | 05/11/2026 |
| Manas Rastogi | 22f3001477 | PENDING | PENDING |
