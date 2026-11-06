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
