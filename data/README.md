# Data tables

The photos are private Kaggle datasets; this folder keeps only the tables that describe them.

| File | What it describes | Written by |
|---|---|---|
| `classes.csv` | The final dataset's 184 species (56 tier 1, 128 tier 2): class id, genus, family, WHO category and photo counts per split, 120,751 photos in all (85,067 train, 17,151 val) | notebook 03 |
| `processing_report.json` | The final dataset: 121,566 photos of 184 classes after the licence filter and the near-copy removal, from 181,503 input rows. Its 815 unknown-pool photos (species outside the 184 classes, kept to test unknown species) are counted here but not in `classes.csv` | notebook 03 |
| `split_report.json` | The split before the licence filter: 181,503 photos of 208 classes. It is kept because the final dataset is cut from this split | notebook 02 |
| `m3/` | The Milestone 3 tables the app reads: the danger table (`medical_importance.csv`), the species cards and WHO passage records (`kb/`, page numbers and offsets only, no WHO text) and the 60 draft retrieval questions (`response_questions.csv`) | read by notebook 16 |

Always take class ids from `classes.csv`.
