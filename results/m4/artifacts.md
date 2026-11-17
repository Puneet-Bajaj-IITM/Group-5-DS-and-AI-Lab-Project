# Milestone 4 artifacts

Where each artifact behind the shipped model (A3) is kept, with its SHA-256 and size. Weights, saved outputs and the
front end's prediction file are too large or not licensed for the repo, so they stay in the Kaggle output of the run
that made them; the rest is in this repo.

## Model and front end

| Artifact | Stored in | Path | SHA-256 | Bytes | Hash from |
|---|---|---|---|---|---|
| A3 weights | Kaggle output `jayeshbansal497/sg-m4-full-a3` | `M4_FULL/best.pt` | `b19f0a5ece4f98fc08911f8781048011e14aeb84b80fde9dc1f4b9263322a02b` | 20,820,579 | `full/A3/manifest.json` |
| A3 prototypes | Kaggle output `jayeshbansal497/sg-m4-full-a3` | `M4_FULL/prototypes.npz` | `14f7475c4dd1026d435d2b5f8b1c4ca51f569897c89b666163d9001dc1b53234` | 348,630 | `full/A3/manifest.json` |
| A3 decision file | this repo (also `M4_FULL/` in the Kaggle output) | `results/m4/full/A3/decision_metrics.json` | `b2884e1a229903fd1fbb6f6221db321ec4b5e583d2c481df88d6456ae44a6f23` | 13,662 | `full/A3/manifest.json` |
| A3 config, with the seed | this repo | `results/m4/full/A3/config.yaml` | `4c5398eeb600e2c5210ec1d596355af28ee12e7f2ad257d1c05895c59981a3b4` | 824 | `full/A3/manifest.json` |
| A3 validation outputs | Kaggle output `jayeshbansal497/sg-m4-full-a3` | `M4_FULL/val_outputs.npz` | `ddf6ce23ceeb2c8a84a970a311db311d3f5a8a381a9c8d9ada4ba083c34749b2` | 26,611,248 | `full/A3/manifest.json` |
| A3 calibration outputs | Kaggle output `jayeshbansal497/sg-m4-full-a3` | `M4_FULL/calib_outputs.npz` | `ec8ac350d4df6cadb3c73ee5d3de9651109629d070615855c189bbfffbf2b285` | 12,449,246 | `full/A3/manifest.json` |
| M4 detector (the app's gate and crop) | Kaggle output `puneetbajajiitm/sg-m4-gate` | `m4_front_end/detector/M4_detector.pt` | `29a1ad1c1b9b561c35ef4ffa1916998837c3c78525c21b51668d9d0e95607b61` | 5,407,301 | `gate/non_snake_test.json`; size computed later |
| M4 segmenter (the app's gate and crop) | Kaggle output `puneetbajajiitm/sg-m4-gate` | `m4_front_end/M4_segmenter.pt` | `b881c338356bdf44aa0b45e0306f59d7b38252d04316da2455c40055aef656ec` | 26,796,107 | `gate/non_snake_test.json`; size computed later |
| No-snake gate rule and cut-offs | this repo | `results/m4/gate/gate.json` | `9893a41db69e89f7978160464e134aa487d8482114381230337182f73163b405` | 348 | computed later |
| M3 detector (the boxes and confidences in `frontend_preds.csv`; A3's no-snake rule reads the confidences) | Kaggle output `puneetbajajiitm/sg-m3-frontends` (teammates: dataset `<owner>/serpentguard-m3-frontend-weights`) | `D1/yolo26n-ft/detector.pt` | `099b36bd8271c328b60be8152161f74558a0083ebe4e6fec1f558e7c0951c5b0` | 5,408,133 | notebook 21 cell 3, checked at the run's start; size computed later |
| M3 U-Net (the masks that cut A3's training and validation crops) | Kaggle output `puneetbajajiitm/sg-m3-frontends` (teammates: as above) | `S1/unet/segmenter.pt` | `f7646444ec409614e992cd7a112a1c2c47fd81a1dc62afea54f50f83664abf3f` | 26,776,971 | notebook 21 cell 3, checked at the run's start; size computed later |
| Front-end predictions (`frontend_preds.csv`, 78,379 rows) | Kaggle output `jayeshbansal497/sg-m4-full-a3` | `full/frontend_preds.csv` | `84140c320fe9a368c2cb3c219d26b4b34af7009ea1be75fc32d69283a17e169e` | 11,730,336 | `full/A3/manifest.json` |
| Scenery bank (3,000 sceneries of 184 species) | not kept: notebook 21 cell 9 rebuilds it in each session's scratch space | `bank.csv` | `2eed027132bfacf956e85895b29131fa09f845c5b60bfcfa58272a687544fa5e` | not recorded | printed by notebook 21 cell 9 in all three full runs; `arms/M4_A3/arm_row.json` |

The two M4 front-end files are copied into a model folder as `detector.pt` and `segmenter.pt` for notebook 24 and
`scripts/verify_inference.py`; the gate's output also holds the segmenter as `m4_front_end/segmenter/segmenter.pt`,
with the same bytes. "Computed later" means the hash or size was taken from the team's local copy after the run, not
recorded by the run itself; each local copy's SHA-256 matches the recorded one where a record exists.

## Executed notebooks

SHA-256 of the copies in this repo.

| Notebook run | Path | SHA-256 | Bytes |
|---|---|---|---|
| Full training, A3 (shipped) | `results/m4/full/executed/21_full_training_A3.ipynb` | `f1fc32f14d0d7aa2c6aff9567214f6a76ed670914e98e7df82489ab9d05fa94d` | 179,361 |
| Full training, A1 | `results/m4/full/executed/21_full_training_A1.ipynb` | `4ece573c03eaaa1d8c280194a307eec0571fa9d4337c4f0680b4590f059bee1c` | 179,216 |
| Full training, H2 | `results/m4/full/executed/21_full_training_H2.ipynb` | `6d933aca86ebf76c9f49d6d7ae1ab3e0c464017d3f750c55d309f0512fe124fa` | 179,153 |
| Background arm A0 | `results/m4/arms/executed/18_background_arms_A0.ipynb` | `e2c37bc92f8bbfe970a771329091968258b22bb608b5943b737502fcc3f8db88` | 32,193 |
| Background arm A1 | `results/m4/arms/executed/18_background_arms_A1.ipynb` | `1a640c5f4b5c2a98d933a01be41c4030ff252410ca0527a56cfc696de626f1bf` | 33,411 |
| Background arm A2 | `results/m4/arms/executed/18_background_arms_A2.ipynb` | `e7491c81fa2f8652c3febdc8fe7c77edec3ca045ebfcedc391063de9bdd635a0` | 32,558 |
| Background arm A3 | `results/m4/arms/executed/18_background_arms_A3.ipynb` | `3c0bc44f8bd18a2c06f77997ea75a0abb5aaedd64aaf0b5bc629c6f880ed006c` | 33,559 |
| Non-snake set | `notebooks/19_non_snake_set.ipynb` | `fd5b4e5c6edf425e5fe547fb622ebd47ce3daa1d3e6b36584c2211d627dda3bf` | 55,538 |
| No-snake gate | `notebooks/20_no_snake_gate.ipynb` | `7020e40b3376efd8958e6abff3a5da06636dee88f5fee7586ffa959cad743e36` | 211,135 |
| Head pilot | `notebooks/22_head_pilot.ipynb` | `3f86c86a17a479e94289056781d3f24156eb8df4df02f4364b9f9c8a1474fc6d` | 21,295 |
| Detail crop | `notebooks/23_head_arms.ipynb` | `1be4ada6aed36837817df865705ddfc6ec1640c877d8602e879a8f54c7aa20a4` | 63,742 |
| Video lock, A3 | `notebooks/24_video_lock.ipynb` | `7cda9b5180494c1bdc092052e1a76f11da4efb4e766447f5cba1b67496a2beb1` | 177,495 |
| Video lock, Milestone 3 model | `results/m4/video/executed/24_video_lock_M3.ipynb` | `68e02e36147318e968ec5cbc6e96279976f1b5fc27ff4901a16b6169b57564d7` | 192,897 |
| Summary tables | `notebooks/25_m4_summary.ipynb` | `0fb09c24099ecf990ad6deb07d72426f9bc31a1b469d61c3bb699cbfa284702b` | 175,212 |
| Report figures | `notebooks/27_m4_figures.ipynb` | `d9bde3eb38a9504f2b105d6836ea444f840a882cd09821120590b36f515d43b1` | 341,262 |

## What the checkpoint holds

`best.pt` holds the model's weights, its config, the epoch it was saved at (24 of 30) and the class, genus and family
counts. The optimizer, learning-rate scheduler, gradient scaler and random-number-generator states were not saved, so
training cannot resume from it; a run can only be repeated from the start. The seed (1) is in `config.yaml`.

## How to download

The Kaggle outputs and datasets are private. The team adds a reviewer's Kaggle username on request. With access and
the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) set up, one command downloads a run's output, for example A3's:

```text
kaggle kernels output jayeshbansal497/sg-m4-full-a3 -p sg-m4-full-a3
shasum -a 256 sg-m4-full-a3/M4_FULL/best.pt
```

The second line checks the download against the table above. The gate's output is
`kaggle kernels output puneetbajajiitm/sg-m4-gate -p sg-m4-gate`.
