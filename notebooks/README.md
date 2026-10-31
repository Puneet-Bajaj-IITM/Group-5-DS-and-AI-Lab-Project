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

The full exploratory analysis is [`Group 5 - 2026T3 - DSAI Lab.ipynb`](../Group%205%20-%202026T3%20-%20DSAI%20Lab.ipynb) in the repository root. It ran before the licence filter.

The team's datasets are private and owned by `puneetbajajiitm`. Ask the team to add your Kaggle username.

Notes:

- Notebooks 03, 06 and 07 are saved with the outputs of their 31 October 2026 runs on the clean dataset. Notebooks 08 and 09 keep their 30 October outputs, and their photos aren't in the main dataset.
- Notebook 02 pins scikit-learn 1.9.1, because `StratifiedGroupKFold` assigns folds differently between releases.
- Scheduled Kaggle runs can only read attached datasets, so attach each one before you save and run.
- Always take class ids from `classes.csv`. `val/` and `test/` lack some species folders, so plain `ImageFolder` would mislabel photos.
