# Notebooks

Each notebook attaches its inputs by handle, does one job and saves its output as a private Kaggle dataset under the account that runs it, so any step can be rerun on its own.

| # | Notebook | Reads | Saves | Runs on | Time |
|---|---|---|---|---|---|
| 01 | [`01_build_raw_dataset`](01_build_raw_dataset.ipynb) | iNaturalist API | `serpentguard-indian-snakes` | CPU, internet on | about 2.5 h |
| 02 | [`02_clean_and_split`](02_clean_and_split.ipynb) | 01 | `serpentguard-m2-splits` | CPU, internet on | about 3 min |
| 03 | [`03_process_dataset`](03_process_dataset.ipynb) | 01, 02 | `serpentguard-indian-snakes-clean`, the main dataset | CPU, internet on | about 30 min |
| 04 | [`04_sam_masks`](04_sam_masks.ipynb) | 01, 02 | snake masks, one zip per shard | GPU T4 x2, internet on | about 3.8 h per shard |
| 05 | [`05_merge_sam_masks`](05_merge_sam_masks.ipynb) | 04 | `serpentguard-sam-masks` | CPU | a few minutes |
| 06 | [`06_segmentation_preprocessing`](06_segmentation_preprocessing.ipynb) | 03, 05 | loaders, manifest, class weights, previews | CPU | a few minutes |
| 07 | [`07_eda_findings`](07_eda_findings.ipynb) | 03, 05 | the main findings and the report's plots | CPU | about 4 min |
| 08 | [`08_extended_sources`](08_extended_sources.ipynb) | 02, other sources | `serpentguard-extended-sources` | CPU, internet on | about 6 min |
| 09 | [`09_combine_and_reclean`](09_combine_and_reclean.ipynb) | 03, 08 | `serpentguard-extended-combined` | CPU | about 5 min |
| | [`load_dataset_example`](load_dataset_example.ipynb) | 03 | nothing; the starting point for a new notebook | CPU | a minute |

The full exploratory analysis is [`Group 5 - 2026T3 - DSAI Lab.ipynb`](../Group%205%20-%202026T3%20-%20DSAI%20Lab.ipynb) in the repository root. It was run on the clean dataset on 31 October 2026.

The team's datasets are private and owned by `puneetbajajiitm`. Ask the team to add your Kaggle username.

Notes:

- Notebooks 03, 06 and 07 and the full analysis are saved with the outputs of their 31 October 2026 runs on the clean dataset. Notebooks 08 and 09 keep their 30 October outputs, and their photos aren't in the main dataset.
- Notebook 02 pins scikit-learn 1.9.1, because `StratifiedGroupKFold` assigns folds differently between releases.
- Scheduled Kaggle runs can only read attached datasets, so attach each one before you save and run.
- Always take class ids from `classes.csv`. `val/` and `test/` lack some species folders, so plain `ImageFolder` would mislabel photos.

## Milestone 3

The Milestone 3 notebooks find the team's datasets the same way. Several also read another notebook's Kaggle output, so declare that kernel as a source; each notebook's first cell says which. Each notebook is saved with the outputs of its run, and its result files stay in that run's Kaggle output. The repo keeps only the few files another notebook reads: the run tables in `results/m3/summary/`, `results/m3/q1/retrieval.csv`, the `results/m3/x1_laptop` CSVs and the shipped decision file `results/m3/final/H1_top1_s1/decision_metrics.json`.

| # | Notebook | Answers | Runs on |
|---|---|---|---|
| 10 | [`10_subset`](10_subset.ipynb) | Which 16% of the training photos we experiment on, and whether that subset is representative; also builds the calibration set | local CPU |
| 11 | [`11_baselines`](11_baselines.ipynb) | How far existing models get before we train anything, how well new species can be added from a few photos, and how our final model compares with the published SnakeCLEF model | GPU T4, internet on |
| 12 | [`12_training`](12_training.ipynb) | How SnakeID is built and trained: one full training run, then every training run (encoders, heads, losses, noise floor, front ends, retrains) in one table | GPU T4, internet on |
| 13 | [`13_cropping`](13_cropping.ipynb) | Whether finding the snake first helps: four segmenters and two detectors, and a crop against a mask | GPU T4 x2, internet on |
| 14 | [`14_tuning`](14_tuning.ipynb) | Which learning rate, weight decay, ArcFace margin and augmentation strength work best, and whether the gain holds when the best settings are retrained | GPU T4 x2, internet on |
| 15 | [`15_safety`](15_safety.ipynb) | How outputs become a verdict: thresholds, the decision rule study, whether the model looks at the snake or the scenery, unknown species and video stability | GPU T4, internet on |
| 16 | [`16_app`](16_app.ipynb) | The whole app from a photo or clip to a verdict and WHO passages, the video demo, passage retrieval, and package size and speed on a laptop | local CPU |

## Milestone 4

The Milestone 4 notebooks work the same way: the Kaggle notebooks find the team's datasets and other runs' outputs by handle, and each notebook's first cell says what it reads. The repo keeps their result files under `results/m4/` and the summary tables under `results/m4/summary/`.

| # | Notebook | Answers | Runs on |
|---|---|---|---|
| 17 | [`17_front_end_full`](17_front_end_full.ipynb) | A box and a mask for every training photo the cropping notebook did not cover, from the same detector and U-Net; not run on its own, since notebook 21 runs the same step | GPU, internet on |
| 18 | [`18_background_arms`](18_background_arms.ipynb) | Whether the model can lean less on the scenery: three ways of showing it the snake (the arms) against Milestone 3's final model, one rule to pick one | GPU T4, internet on |
| 19 | [`19_non_snake_set`](19_non_snake_set.ipynb) | A set of about 3,000 openly licensed photos with no snake in them, split by photographer | CPU, internet on |
| 20 | [`20_no_snake_gate`](20_no_snake_gate.ipynb) | How often the front end sees a snake where there is none, the detector and U-Net retrained with the non-snake photos, and which gate the app uses | GPU T4, internet on |
| 21 | [`21_full_training`](21_full_training.ipynb) | The chosen recipes trained on every training photo, and scored against the goals and against B1 | GPU T4, internet on |
| 22 | [`22_head_pilot`](22_head_pilot.ipynb) | Whether open-vocabulary detectors can find a snake's head and tail | GPU, internet on |
| 23 | [`23_head_arms`](23_head_arms.ipynb) | Whether a second, closer look the model picks itself (the detail crop) helps | GPU T4, internet on |
| 24 | [`24_video_lock`](24_video_lock.ipynb) | Whether pooling frames and locking the answer steadies the shown species on a clip, with the shipped model behind the app's front end | local CPU |
| 25 | [`25_m4_summary`](25_m4_summary.ipynb) | The goals and the report's tables, read from the saved result files | local CPU |

Where each run is saved with its outputs:

- Notebooks 19, 20, 22, 23 and 25 are saved here with the outputs of the run their results come from.
- Notebook 18 ran once per arm. The copy here holds no outputs; each run is in `results/m4/arms/executed/18_background_arms_<arm>.ipynb` (A0 to A3).
- Notebook 21 ran once per recipe. The copy here holds no outputs; each run is in `results/m4/full/executed/21_full_training_<run>.ipynb` (A1, H2 and A3).
- Notebook 24 is saved here with its run on the shipped model, A3; its earlier run on the Milestone 3 model is `results/m4/video/executed/24_video_lock_M3.ipynb`.
- Notebook 17 was not run on its own, so it has no outputs; its step ran inside each run of notebook 21.
- Three numbers were measured by scripts on the laptop's CPU rather than by a notebook: the non-snake test split (goal G4), the app's front-end cost and the detail crop's laptop timing. The scripts are in [`../scripts/`](../scripts/README.md).
