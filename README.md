# SerpentGuard AI: snake segmentation and classification

Group 5, DS and AI Lab Project (IIT Madras BS, DS & AI Lab course).

SerpentGuard AI finds a snake in a photo or camera feed and suggests its species from 184 classes of snakes recorded in India. It shows the WHO medical-importance category of each candidate and says "unknown" when it is not confident. As built, a detector and a segmenter find the snake and crop the photo to it, and a MobileViT-S classifier (a 20.8 MB checkpoint) names the species and also predicts its genus, family and WHO category; decision rules turn its outputs into the verdict. Milestone 3 compared candidate architectures; Milestone 4 trains the chosen pipeline on every training photo, adds a no-snake gate and a video lock. Three of the five goals were not met: G3 (against B1), G4 (false snakes) and G5 (video). Results are in [results/README.md](results/README.md).

> Safety: this is a student research and education project. It does not diagnose bites or give treatment advice, and "no snake detected" never means an area is safe. In an emergency, go to the nearest hospital.

## Team

| Member | Role |
|---|---|
| Jayesh Bansal | Team lead |
| Vikas | Member |
| Puneet Bajaj | Member |
| Manas Rastogi | Member |

Each member's weekly contributions are in [work-contribution-log.md](work-contribution-log.md).

## Milestones

| # | Milestone | Due (2026) | Status | Documents |
|---|---|---|---|---|
| 1 | Problem definition and literature review | 22 Oct | Submitted | [Report](documentation/milestone%201/project-report.md) · [Slides](documentation/milestone%201/assets/SerpentGuard_AI_Group5_Presentation.pdf) |
| 2 | Dataset preparation | 29 Oct | Submitted | [Report](documentation/milestone%202/project-report.md) · [Slides](documentation/milestone%202/SerpentGuard_AI_M2_Presentation.pdf) (the slides show the dataset before the licence filter; the report's numbers are the final ones) · [Notebooks](notebooks/README.md) |
| 3 | Model architecture and end-to-end pipeline | 5 Nov | Submitted | [Report](documentation/milestone%203/project-report.md) · [Slides](documentation/milestone%203/SerpentGuard_AI_M3_Presentation.pdf) · [Notebooks](notebooks/README.md) |
| 4 | Model training | 19 Nov | In review | [Results](results/README.md) · [Figures](documentation/milestone%204/assets/figures.md) · [Notebooks](notebooks/README.md#milestone-4) |
| 5 | Evaluation and analysis | 26 Nov | Planned | |
| 6 | Deployment and documentation | 10 Dec | Planned | |

Course requirements: [guidelines](https://bsc-iitm.github.io/dsai-lab-project/guidelines.html) · [milestones](https://bsc-iitm.github.io/dsai-lab-project/milestones.html).

## Repository layout

```text
documentation/          milestone reports and slides (one folder per milestone)
notebooks/              the notebook chain, from raw download to full training, the video lock and the summary
src/                    the serpentguard package the notebooks share
results/                Milestone 4's result files and summary tables, and the Milestone 3 files other notebooks read; start at results/README.md
scripts/                CPU scripts: four measurements, verify_inference.py and two bias-variance checks (see scripts/README.md)
data/                   class table and processing reports (images live on Kaggle, never in git)
work-contribution-log.md  weekly per-member contributions
```

Each notebook attaches its inputs by handle, does one job and saves its output as a private Kaggle dataset; see [notebooks/README.md](notebooks/README.md).

## Reproduce Milestone 4

1. Check the numbers without a GPU: run [`notebooks/25_m4_summary.ipynb`](notebooks/25_m4_summary.ipynb) on a laptop. It reads only the saved files in `results/` and stops if any table disagrees with its source.
2. Environment: the laptop notebooks (24, 25 and 27) and `scripts/` ran with Python 3.12 and the packages pinned in `requirements-m3.txt` (the file keeps its Milestone 3 name). The Kaggle notebooks use the Kaggle image's packages, and each run records them in its `versions.json` (the shipped run: Python 3.13, PyTorch 2.11 with CUDA 13.0, on one Tesla T4).
3. Data: the photos, masks and weights are private Kaggle datasets owned by `puneetbajajiitm`; ask the team to add your Kaggle username. Each notebook's first cell says which datasets and run outputs to attach.
4. Order: 19 (non-snake photos), then 20 (the no-snake gate) and 18 (background arms on the experiment subset), then 21 (full training, about 7 hours on one T4 per recipe, including its front-end step and a short smoke pass), 22 and 23 (head crops), 24 (the video lock, after 20 and 21), then 25, and 27 (the report's figures, on a laptop) last.
5. Don't rerun notebook 24 to check it: that would score the video test clips a second time. The non-snake test split was scored once, by `scripts/score_non_snake_test.py`, which refuses to run again once its outputs exist. The snake test split has not been read; it stays locked until Milestone 5.

## Data and licences

Images come from iNaturalist research-grade observations of snake species recorded in India, with photos from anywhere in the world ([Milestone 2 report](documentation/milestone%202/project-report.md)). The main dataset keeps only CC0, CC BY, CC BY-SA, CC BY-NC and CC BY-NC-SA photos, with each photographer's licence and credit recorded in `metadata.csv`. It is a private Kaggle dataset, images are never committed here apart from credited samples in the reports, and it is used only for this non-commercial course project.
