# SerpentGuard AI: snake segmentation and classification

Group 5, DS and AI Lab Project (IIT Madras BS, DS & AI Lab course).

SerpentGuard AI finds a snake in a photo or camera feed and suggests its species from 184 classes of snakes recorded in India. It shows the WHO medical-importance category of each candidate and says "unknown" when it is not confident. The planned model is a compact segmentation and classification network for Android; Milestone 3 compares candidate architectures.

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
| 2 | Dataset preparation | 29 Oct | Submitted | [Report](documentation/milestone%202/project-report.md) · [Slides](documentation/milestone%202/SerpentGuard_AI_Milestone2_Dataset_Presentation.pdf) · [Notebooks](notebooks/README.md) |
| 3 | Model architecture and end-to-end pipeline | 5 Nov | Planned | |
| 4 | Model training | 19 Nov | Planned | |
| 5 | Evaluation and analysis | 26 Nov | Planned | |
| 6 | Deployment and documentation | 10 Dec | Planned | |

Course requirements: [guidelines](https://bsc-iitm.github.io/dsai-lab-project/guidelines.html) · [milestones](https://bsc-iitm.github.io/dsai-lab-project/milestones.html).

## Repository layout

```text
documentation/          milestone reports and slides (one folder per milestone)
notebooks/              the Kaggle notebook chain, from raw download to data loaders
data/                   class table and processing reports (images live on Kaggle, never in git)
work-contribution-log.md  weekly per-member contributions
```

Each notebook attaches its inputs by handle, does one job and saves its output as a private Kaggle dataset; see [notebooks/README.md](notebooks/README.md).

## Data and licences

Images come from iNaturalist research-grade observations of snake species recorded in India, with photos from anywhere in the world ([Milestone 2 report](documentation/milestone%202/project-report.md)). The main dataset keeps only CC0, CC BY, CC BY-SA, CC BY-NC and CC BY-NC-SA photos, with each photographer's licence and credit recorded in `metadata.csv`. It is a private Kaggle dataset, images are never committed here apart from credited samples in the reports, and it is used only for this non-commercial course project.
