# Scripts

Four Milestone 4 numbers were measured on the team laptop's CPU by scripts rather than notebooks. The code here is
the code that ran; after the runs, only comments, descriptive text strings and one hard-coded working-folder path
were changed (`time_detail_crop.py` now reads it from `SERPENTGUARD_WORK`). They read their inputs from the folder
layout they ran in, outside the repo (the photos, clips and weights may not be redistributed), so they do not run
from this folder unchanged.

| Script | What it produced | How it was run |
|---|---|---|
| [`fetch_non_snake_test.py`](fetch_non_snake_test.py) | the 600 non-snake test photos, re-made from `manifest.csv`'s source urls with notebook 19's `negatives.save_photo`, each checked against the manifest's size and pHash; kept outside the repo | once, on the laptop's CPU with internet on, no arguments |
| [`score_non_snake_test.py`](score_non_snake_test.py) | `results/m4/gate/non_snake_test.json` (goal G4) and the per-photo scores it cites; its `script_sha256` is this file's | once, on the laptop's CPU, no arguments; it refuses to run when its outputs exist |
| [`measure_app_cost.py`](measure_app_cost.py) | `results/m4/arms/app_cost.json`, the time per frame of the detector and U-Net the app runs; the file's `front_end_source` and `script` fields were written afterwards | `python measure.py <src folder> <out json>`, on the laptop's CPU (run under the name `measure.py`) |
| [`time_detail_crop.py`](time_detail_crop.py) | the `laptop_ms_per_frame` block of `results/m4/heads/M4_H2.json`: one look and the detail crop, timed on the same frames; its printed medians were copied into the file | `PYTHONPATH=src python laptop_h2.py`, on the laptop's CPU (run under the name `laptop_h2.py`) |

SHA-256 of each file as it ran and as it is here:

| Script | As it ran | As it is here | Cited by |
|---|---|---|---|
| `fetch_non_snake_test.py` | `66da3cc4046e76fa60bf6ae3a8f1bb9a67b6be1415a2a25a31bba50af3ac234a` | `ee8acaa351023f0f452551cded37136d3424a1ea68028d885b410c6f62b71bc4` | nothing; no SHA-256 was recorded at the run |
| `score_non_snake_test.py` | `2a69ad5c9e781c070d98a5b18d61bb1d9cd08f2467bdd816ed5401629a894469` | `97855afdc386974196e20166fe59be36c3c39bd589a95c1d2f79e3f1db8169f5` | `results/m4/gate/non_snake_test.json` (`script_sha256`), which matches the copy as it ran |
| `measure_app_cost.py` | `ca71c8efbe93e44ebb41c60cd61667dab130306377b766e21ea2e3e1e89cbd72` | `eb9db7cd75f537aa5be5616d8e9ef4f2fda50b006df976d86928da2798bc955e` | nothing; no SHA-256 was recorded at the run |
| `time_detail_crop.py` | `2878268a255e34119a2ff9e0f331860758e41dba4753f5323acc1219e4a66f3c` | `935936fd16082b380db6d431ec1553f73ce11102fecbff3fc753c15f9a698dd3` | nothing; no SHA-256 was recorded at the run |

The copies as they ran are kept outside the repo with the run records. Only `score_non_snake_test.py` can be matched
to its run by its hash. The other three are the copies found next to their outputs after the runs; no hash was taken
when they ran, so a later edit to them cannot be ruled out.
Some descriptive text fields in the three result files were reworded after the runs to name repo files instead of
folders outside the repo; no number changed.
