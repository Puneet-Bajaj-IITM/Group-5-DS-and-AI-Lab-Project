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

The M3 notebooks find the team's datasets the same way. Several also read another notebook's Kaggle output, so declare that kernel as a source; each notebook's first cell says which. Each notebook is saved with the outputs of its run, and its result files stay in that run's Kaggle output. The repo keeps only the few files another notebook reads (`results/m3/q1/retrieval.csv`, `results/m3/x1_laptop/package.csv`, `results/m3/final/H1_top1_s1/decision_metrics.json`) and notebook 21's tables in `results/m3/summary/`.

| # | Notebook | Answers | Runs on |
|---|---|---|---|
| 10 | [`10_m3_subset`](10_m3_subset.ipynb) | the candidate subset and calibration set | CPU |
| 11 | [`11_m3_baselines`](11_m3_baselines.ipynb) | B0 to B5, C1, K1 enrolment | GPU T4, internet on |
| 12 | [`12_m3_train`](12_m3_train.ipynb) | E1 to E4, N0, FE, K1, the H1 retrains, N1 | GPU T4 x2, internet on |
| 13 | [`13_m3_frontends`](13_m3_frontends.ipynb) | S1, D1 | GPU T4 x2, internet on |
| 14 | [`14_m3_tuning`](14_m3_tuning.ipynb) | H1 | GPU T4 x2, internet on |
| 15 | [`15_m3_end_to_end`](15_m3_end_to_end.ipynb) | P1 and the video demo | local CPU |
| 16 | [`16_m3_response`](16_m3_response.ipynb) | Q1 | local CPU |
| 17 | [`17_m3_robustness`](17_m3_robustness.ipynb) | R1 | GPU T4, internet on |
| 18 | [`18_m3_video`](18_m3_video.ipynb) | V1 | GPU T4, internet on |
| 19 | [`19_m3_decision`](19_m3_decision.ipynb) | the verdicts, U1 and the decision-rule study | CPU, internet on |
| 20 | [`20_m3_x1`](20_m3_x1.ipynb) | X1 on the laptop CPU (or a Kaggle CPU) | local CPU |
| 21 | [`21_m3_summary`](21_m3_summary.ipynb) | every run's results in one table, with a plot | local CPU |
