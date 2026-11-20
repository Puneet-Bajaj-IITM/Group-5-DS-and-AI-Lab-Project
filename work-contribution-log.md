# Work contribution log

Project: SerpentGuard AI, snake detection and classification<br>
Team: Group 5, DS and AI Lab Project. Team lead Jayesh Bansal; members Vikas, Puneet Bajaj and Manas Rastogi

The course asks for this log to be updated every week with each member's contributions. Add each new week above the previous one. Members check their own rows, and everyone also signs the acknowledgment page of each milestone report.

## Week 5: 15 to 21 November 2026 (Milestone 4, community voting and app integration)

### Contribution entries

| Work item | Contribution / evidence | Contributor(s) | Date or period | Reviewed / confirmed by |
|---|---|---|---|---|
| Community voting notebook execution | Ran [notebook 26](notebooks/26_community_voting.ipynb) with the project Python environment and saved its actual outputs and execution counts; checked eight scenarios, species/genus fallback, vote replacement, repeat submissions, invalid inputs and threshold boundaries. Added its [README row](notebooks/README.md#milestone-4) and a standard-library [runner](scripts/run_voting_notebook.py). | Vikas | 19 Nov 2026 | Pending team review |
| Community backend and training review | SQLAlchemy API/database for sightings, authenticated evidence, one active species/genus vote per reviewer, weighted Bayesian consensus, idempotent retries and curator approval/export tied to the current consensus revision. Added consent and approval-invalidation checks. [PR #19](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/19), [report section 8.1](documentation/milestone%204/project-report.md#81-community-voting-and-offline-evidence) | Vikas | Recorded 19 Nov 2026 | Pending team review |
| Offline app and live-camera integration | SQLite drafts/outbox, immutable evidence retries, secure remembered accounts, foreground/background sync leases, cached review/consensus, vote-edit conflict reconciliation and live-camera still capture. Tests include the real SQLAlchemy backend. [PR #24](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/24) | Vikas | Recorded 19 Nov 2026 | Pending team review |
| Supplied model-bundle verification | Added the [reproducible verifier](scripts/verify_mobile_bundle.py); verified five model/config hashes, 200/200 decision fixtures and 5/5 checksum-checked photos in the supplied FP32 Python reference. This is separate from Jayesh's Android checks and is not a rerun of the supplied 300-photo parity claim. | Vikas | 19 Nov 2026 | Pending team review |
| M3 presentation correction | Corrected the existing [M3 PDF](documentation/milestone%203/SerpentGuard_AI_M3_Presentation.pdf) against its report, including 184 of India's 300+ species, seeds 1/2/42, recorded values and a final acknowledgment slide. [PR #19](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/19) | Vikas | 19 Nov 2026 | Pending team review |
| M4 team presentation and documentation | Prepared the whole-team [presentation PDF](documentation/milestone%204/SerpentGuard_AI_M4_Presentation.pdf) and aligned the [Markdown report](documentation/milestone%204/project-report.md) with voting/backend evidence and remaining limits. The editable deck stays outside the repository. Updated contribution entries without adding members' signatures. | Vikas | 19 Nov 2026 | Pending team review |
| Training and evaluation record | Background arms, non-snake set and gate, capped full-training runs, head/detail-crop studies, video lock, saved summaries/figures and model bundle. Authorship follows the report and [PR #20](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/20); Kaggle account ownership is recorded separately. | Puneet Bajaj | Milestone 4; recorded 19 Nov 2026 | Pending team review |
| Real model in the app | Ported the model bundle's `reference_pipeline.py` to Dart and connected the fp32 A3 classifier, detector and segmenter with ONNX Runtime; replaced the placeholder decision rule with DR0 at A3's thresholds (200/200 [golden decisions](app/test/golden_decisions_test.dart)); image steps matched to cv2 with the report's documented INTER_AREA rounding exceptions ([test](app/test/image_ops_test.dart)); Android decoding fix; 5/5 [golden photos on the Android emulator](app/integration_test/golden_photos_test.dart). [Section](documentation/milestone%204/project-report.md#8-the-model-in-the-app), [PR #25](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/25) | Jayesh Bansal | 18 to 19 Nov 2026 | Pending team review |
| Video check and live camera | Video check screen (up to 10 frames a second, M4 video lock) and the live-camera adapter for PR #24's screen; demo on two Wikimedia clips: cobra "Dangerous, WHO Category 1" with 0 changes, red-tailed ratsnake "Not on the WHO list" ([section](documentation/milestone%204/project-report.md#8-the-model-in-the-app)) | Jayesh Bansal | 19 Nov 2026 | Pending team review |
| Regional names | Names in up to 7 Indian languages for 9 of the 16 WHO-listed species in `species.json`; draft until native speakers check them ([list](app/tool/regional_names.csv)) | Jayesh Bansal | 19 Nov 2026 | Pending team review |
| Flutter startup and branding | Built the animated splash screen with a fallback image, Skip, automatic navigation, and direct access to snakebite SOS. Refined the transition from the fallback image to video. Loaded the species catalog and local store in parallel and deferred building secondary tabs until first use. Added Android and iOS launcher icons, source artwork and video, and [three widget tests](app/test/widget_test.dart) for Skip, automatic navigation, and SOS access. [PR #22](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/22) | Manas Rastogi | 18 Nov 2026 | Manas Rastogi |
| Android CameraX build fix | Added the AndroidX concurrent-futures dependency to the CameraX plugin compile classpath so javac can resolve CallbackToFutureAdapter. [PR #28](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/28) | Manas Rastogi | 20 Nov 2026 | Manas Rastogi |

### Member contribution summary

| Member name | Work completed | Evidence / report sections | Member confirmation |
|---|---|---|---|
| Jayesh Bansal | Flutter app (Week 4); the model in the app, video check, live-camera adapter and regional names; report section 8. A3's full run is on his Kaggle account | [project-report.md, section 8](documentation/milestone%204/project-report.md#8-the-model-in-the-app), [PR #21](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/21), [PR #25](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/25) | Jayesh Bansal |
| Vikas | Community backend and curator export, executed notebook 26, offline SQLite review/sync, live-camera screen, supplied Python model-bundle verification, M3 slide corrections, M4 team presentation PDF and Markdown report updates. H2 ran on his Kaggle account; that fact alone is not training-code authorship. | [PR #19](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/19), [PR #24](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/24), [M4 report](documentation/milestone%204/project-report.md), [slides](documentation/milestone%204/SerpentGuard_AI_M4_Presentation.pdf) | Pending review |
| Puneet Bajaj | Background arms, non-snake set/gate, full training, head/detail-crop studies, video lock, summary tables/figures and model bundle, as recorded in the M4 report | [PR #20](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/20), [M4 report](documentation/milestone%204/project-report.md) | Pending review |
| Manas Rastogi | Built the animated Flutter splash screen with Skip and direct access to snakebite SOS. Improved startup and video playback, added launcher branding for Android and iOS, wrote three widget tests, and fixed the Android CameraX build. | [PR #22](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/22), [PR #28](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/28) | Manas Rastogi |

### Milestone 4 acknowledgment

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
| Jayesh Bansal | 24f100185 | ![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0) | 19/11/2026 |
| Vikas | 23f3001800 | | |
| Puneet Bajaj | 23f3002416 | | |
| Manas Rastogi | 22f3001477 | ![Manas Rastogi](https://drive.google.com/uc?export=download&id=1j9vC0yjI-eb3lGfETUdnK3GAXq77__lT) | 19/11/2026 |

## Week 4: 8 to 14 November 2026 (Milestone 4, model training and the phone app)

### Contribution entries

| Work item | Contribution / evidence | Contributor(s) | Date or period | Reviewed / confirmed by |
|---|---|---|---|---|
| Flutter phone app | First version of the app in `app/`: identify flow, SOS by verdict (112, 108, 102, location SMS), nearby hospitals from OpenStreetMap kept for offline use, cached map, contacts, opt-in Bluetooth snake alerts; 18 tests, run on an Android emulator. [PR #21](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/21) | Jayesh Bansal | 13 Nov 2026 | Pending team review |
| A3 training run | The full A3 training run was on Jayesh Bansal's Kaggle account (`jayeshbansal497/sg-m4-full-a3`); the account owner is named as the account, not the author | Kaggle account | 8 to 14 Nov 2026 | Pending team review |

## Week 3: 1 to 7 November 2026 (Milestone 3, model selection, baselines, tuning, end-to-end pipeline)

### Contribution entries

Evidence for each row is the notebook it names, saved with the outputs of its run. Runs listed as on another member's Kaggle account were run there; the account owner is named only as the account, not as the author.

| Work item | Contribution / evidence | Contributor(s) | Date or period | Reviewed / confirmed by |
|---|---|---|---|---|
| Candidate subset | 16.4% subset stratified by class and photographer (13,925 photos, all 184 classes), calibration set and hand-check sample; [notebook 10](notebooks/10_subset.ipynb) | Puneet Bajaj | 4 Nov 2026 | Pending |
| Package, model and training | `serpentguard` package: metrics, SnakeID (three mobile encoders with ArcFace, genus, family and WHO heads), losses and the training runner; [notebook 12](notebooks/12_training.ipynb) | Puneet Bajaj | 4 Nov 2026 | Pending |
| Baselines | B0 to B5 before any training, including the published SnakeCLEF 2021 model and BioCLIP zero-shot; [notebook 11](notebooks/11_baselines.ipynb) | Puneet Bajaj | 4 to 5 Nov 2026 | Pending |
| Encoder, head and loss comparison | E1 to E4 and the N0 noise floor; [notebook 12](notebooks/12_training.ipynb), with all runs in one table | Puneet Bajaj | 5 Nov 2026 | Pending |
| Front end | S1 segmenters, D1 detector and the FE crop and mask ablation; [notebook 13](notebooks/13_cropping.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Tuning and retrain | H1 random search over 24 trials, then the top three configurations retrained at seeds 1, 2 and 42 (N1); [notebook 14](notebooks/14_tuning.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Decision layer | Verdict rules with thresholds fitted to the lower bound of 95% WHO recall on the calibration set; the DR0 to DR4 rule study; [notebook 15](notebooks/15_safety.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Few-shot enrolment and retrieval | K1 enrolment of new species from a few photos; Q1 retrieval of WHO first-aid passages with 60 draft questions; [notebook 11](notebooks/11_baselines.ipynb), [notebook 16](notebooks/16_app.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Robustness and video | R1 blanked-snake and swapped-background checks; V1 on 200 synthetic clips with frame voting; [notebook 15](notebooks/15_safety.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Export and speed | X1 ONNX export, fp16 and int8 variants, laptop latency and package size; [notebook 16](notebooks/16_app.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| End-to-end pipeline and demo | P1 from a photo or clip to the verdict, species and WHO passages, the video demo, and the detector padding fix; [notebook 16](notebooks/16_app.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Comparison with the published model | B1 on all 184 classes and on the 62 species both models know; [notebook 11](notebooks/11_baselines.ipynb) | Puneet Bajaj | 5 Nov 2026 | Pending |
| Code clean-up and run outputs | `serpentguard` cut to the code the notebooks use; notebooks 10 to 16 saved with their run outputs; [notebook 12](notebooks/12_training.ipynb) puts every run in one table | Puneet Bajaj | 6 Nov 2026 | Pending |
| Kaggle runs | Training and evaluation kernels (E1 to E4, FE, H1 slices, retrain, baselines) run on Puneet Bajaj's, Jayesh Bansal's and Vikas's Kaggle accounts | Puneet Bajaj | 4 to 5 Nov 2026 | Pending |
| Report and deck | [Report](documentation/milestone%203/project-report.md) with figures, and the [presentation](documentation/milestone%203/SerpentGuard_AI_M3_Presentation.pdf) | Puneet Bajaj | 4 to 5 Nov 2026 | Pending |
| Jayesh Bansal's work this week | TO BE FILLED BY Jayesh Bansal | Jayesh Bansal | | Jayesh Bansal |
| Vikas's work this week | Milestone 3 presentation deck (PPT) | Vikas | 5 Nov 2026 | Vikas |
| Manas Rastogi's work this week | TO BE FILLED BY Manas Rastogi | Manas Rastogi | | Manas Rastogi |

### Member contribution summary

| Member name | Work completed | Evidence / report sections | Member confirmation |
|---|---|---|---|
| Jayesh Bansal | TO BE FILLED BY Jayesh Bansal. Kaggle runs on his account are listed above | | |
| Puneet Bajaj | Subset, model and training code, baselines, front end, tuning, decision layer, enrolment, retrieval, robustness, video, export, end-to-end pipeline and demo, report and deck | [project-report.md](documentation/milestone%203/project-report.md) | Puneet Bajaj |
| Vikas | Presentation (PPT) | [SerpentGuard_AI_M3_Presentation.pdf](documentation/milestone%203/SerpentGuard_AI_M3_Presentation.pdf) | Vikas |
| Manas Rastogi | TO BE FILLED BY Manas Rastogi | | |

### Milestone 3 acknowledgment

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
| Jayesh Bansal | 24f100185 | ![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0) | 07/11/2026 |
| Vikas | 23f3001800 | Vikas | 05/11/2026 |
| Puneet Bajaj | 23f3002416 | Puneet Bajaj | 05/11/2026 |
| Manas Rastogi | 22f3001477 | | |

## Week 2: 23 to 31 October 2026 (Milestone 2, dataset preparation)

### Contribution entries

| Work item | Contribution / evidence | Contributor(s) | Date or period | Reviewed / confirmed by |
|---|---|---|---|---|
| Dataset build | Collected every research-grade iNaturalist photo of the 289 snake species recorded in India (182,497 photos, 0 failed) into a private Kaggle dataset, and checked 30 random photos (SHA-1 and size) and 8 species counts against iNaturalist | Puneet Bajaj, Jayesh Bansal | 24 to 25 Oct 2026 | Jayesh Bansal |
| Cleaning and splits | Removed duplicates, conflicting labels and tiny files; added genus, family, WHO category and tiers; built photographer-grouped train, validation and test splits with leakage checks ([`data/split_report.json`](data/split_report.json)) | Puneet Bajaj | 29 Oct 2026 | Jayesh Bansal |
| Processed dataset | Kaggle processing run with near-duplicate removal, final folder layout, sample grids and the private processed dataset | Puneet Bajaj | 29 Oct 2026 | Jayesh Bansal |
| Reproducibility | Wrote the notebook chain in [`notebooks/`](notebooks/README.md) (01 raw build, 02 split, 03 process, plus a loader for EDA and training); checked that it rebuilds the processed dataset exactly, and pinned scikit-learn so the split reproduces | Puneet Bajaj | 29 Oct 2026 | Jayesh Bansal |
| Report | Milestone 2 report sections 1 to 4 and 6 to 8 (dataset documentation, splits, folder structure, reproducibility) | Puneet Bajaj, Jayesh Bansal | 29 Oct 2026 | Jayesh Bansal |
| Snake masks | Built [`notebooks/04_sam_masks.ipynb`](notebooks/04_sam_masks.ipynb) (Grounding DINO and SAM 2.1); ran a 500-photo pilot through four versions (hand removal, single-box rule, quality flags, two-GPU speed-up), then masked all 129,241 training photos in six shards on three team accounts and merged them into a private dataset | Puneet Bajaj | 29–30 Oct 2026 | Jayesh Bansal |
| Presentation | Prepared the Milestone 2 dataset presentation slides ([`SerpentGuard_AI_M2_Presentation.pdf`](documentation/milestone%202/SerpentGuard_AI_M2_Presentation.pdf)) | Vikas | 29 Oct 2026 | Vikas |
| Preprocessing pipeline | Built SAM mask-guided foreground extraction (soft-dimming and hard-masking), aspect-ratio-preserving letterbox resizing (320x320), tier-aware joint augmentation (Albumentations), and the PyTorch multi-task dataset and class-balanced DataLoader ([`notebooks/06_segmentation_preprocessing.ipynb`](notebooks/06_segmentation_preprocessing.ipynb)) | Vikas | 30 Oct 2026 | Vikas |
| EDA notebook | Wrote and published comprehensive 18-section EDA notebook covering species distribution, WHO categories, split balance, image quality, geographic distribution, temporal patterns, observer diversity, family taxonomy, excluded photos, lookalike pairs, and class imbalance analysis. Pushed to Kaggle notebook wreckermoon/group-5-2026t3-dsai-lab. | Manas Rastogi | 29 Oct 2026 | Manas Rastogi |
| Extended sources plan | Researched and documented all additional data sources beyond iNaturalist: SnakeCLEF 2021/2023/2024, Open Images V7, GBIF India, Roboflow Universe, India-specific Kaggle datasets. Created notebooks 08 and 09 for extended data pull and cross-source deduplication. | Manas Rastogi | 29 Oct 2026 | Manas Rastogi |
| Notebook fixes and runs | Made notebooks 06, 08 and 09 run on the real datasets (class ids from `classes.csv`, photo and mask paths, WHO ids, the whole photo as model input, real duplicate checks against all our photos) and ran all three on Kaggle; built the private datasets `serpentguard-extended-sources` (2,038 extra photos, 296 Open Images snake masks) and `serpentguard-extended-combined` (1,771 checked photos, not merged into the main dataset) | Puneet Bajaj | 30 Oct 2026 |Puneet Bajaj |
| Licence-clean dataset | After the review, removed all-rights-reserved and no-derivatives photos and near copies, and rebuilt the main dataset as the private `serpentguard-indian-snakes-clean` (121,566 photos, 184 classes) | Puneet Bajaj | 31 Oct 2026 |Puneet Bajaj |
| Preprocessing and EDA update | Added exposure correction, photographer and class caps, camera-style augmentation and a valid-mask flag to notebook 06; shortened the notebooks; added `07_eda_findings`; re-ran the full EDA on the clean dataset; rewrote the Milestone 2 report | Puneet Bajaj | 31 Oct 2026 |Puneet Bajaj |

### Member contribution summary

| Member name | Work completed | Evidence / report sections | Member confirmation |
|---|---|---|---|
| Jayesh Bansal | Team lead: coordination and distributing the work | | Jayesh Bansal |
| Puneet Bajaj | Data collection, cleaning, splits, licence filter, Kaggle datasets, mask pipeline, notebooks 01 to 05 and 07, fixes to 06, 08 and 09, report | [notebooks](notebooks/README.md); [report](documentation/milestone%202/project-report.md) | Puneet Bajaj |
| Vikas | Presentation preparation; SAM segmentation preprocessing pipeline, multi-task dataset and data loader | [SerpentGuard_AI_M2_Presentation.pdf](documentation/milestone%202/SerpentGuard_AI_M2_Presentation.pdf); [06_segmentation_preprocessing.ipynb](notebooks/06_segmentation_preprocessing.ipynb); report section 5.2 | Vikas |
| Manas Rastogi | EDA notebook (18 sections); extended sources discovery and notebooks 08+09 | wreckermoon/group-5-2026t3-dsai-lab on Kaggle; notebooks/08_extended_sources.ipynb; notebooks/09_combine_and_reclean.ipynb | Manas Rastogi |

### Milestone 2 acknowledgment

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
| Jayesh Bansal | 24f100185 | ![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0) | 31/10/2026 |
| Vikas | 23f3001800 | Vikas | 31/10/2026 |
| Puneet Bajaj | 23f3002416 | Puneet Bajaj | 31/10/2026 |
| Manas Rastogi | 22f3001477 | ![Manas Rastogi](https://drive.google.com/uc?export=download&id=1j9vC0yjI-eb3lGfETUdnK3GAXq77__lT) | 31/10/2026 |

## Week 1: 18 to 22 October 2026 (Milestone 1, problem definition and literature review)

### Contribution entries

| Work item | Contribution / evidence | Contributor(s) | Date or period | Reviewed / confirmed by |
|---|---|---|---|---|
| Repository setup | Created the GitHub repository | Puneet Bajaj | 18 Oct 2026 | Puneet Bajaj |
| Literature review | Summarized SnakeCLEF benchmarks and added a sourced comparison of Google Lens/Search, Gemini Live, ChatGPT camera/video, direct LLM answers, and prior mobile/offline/video snake systems. | Jayesh Bansal | 22 Oct 2026 | Jayesh Bansal |
| Report editing and verification | Updated the report with current product/research comparisons, cited implementation precedents, and marked unbuilt capabilities as proposals rather than results. | Jayesh Bansal | 22 Oct 2026 | Jayesh Bansal |
| Presentation | Prepared the Milestone 1 slide deck. | Vikas | 22 Oct 2026 | Vikas |
| Feasibility review and data check | Checked the plan against each milestone's requirements. Counted the available data with the iNaturalist API (27 Indian species with at least 200 CC-licensed research-grade observations each), located the released PLOS baseline model, the Open Images snake boxes and the WHO species categories, and found no public species-labelled snake video. | Puneet Bajaj | 22 Oct 2026 | Jayesh Bansal |
| Report and slide revision | Added scope, stakeholders and measurable targets for the M1 rubric, the species table, real data samples and two output mockups, the grouped-split and video-clip plan, the Kaggle and Colab compute plan, and the detector, tracker and classifier architecture. Corrected the PLOS first author, replaced slide figures and chart values that had no source, matched the slide claims to the report, and added a slide on scope, stakeholders and targets. | Puneet Bajaj | 22 Oct 2026 | Jayesh Bansal |
| Repository documentation | Wrote the README, extended `.gitignore` to keep data and model files out of git, and fixed the links in this log. | Puneet Bajaj | 22 Oct 2026 | Jayesh Bansal |
| Lookalike species research | Counted how often iNaturalist identifiers in India confuse our 27 classes (25 pairs that join a WHO-listed and a not-listed species, 107 confusions), checked distinguishing features against the WHO SEARO 2016 guidelines and published papers, and added report section 3.6 with four new references. | Manas Rastogi | 22 Oct 2026 | Manas Rastogi |

### Member contribution summary

| Member name | Work completed | Evidence / report sections | Member confirmation |
|---|---|---|---|
| Jayesh Bansal | Project report | [project-report.md](documentation/milestone%201/project-report.md) | Jayesh Bansal |
| Vikas | Presentation preparation | [SerpentGuard_AI_Group5_Presentation.pdf](documentation/milestone%201/assets/SerpentGuard_AI_Group5_Presentation.pdf) | Vikas |
| Puneet Bajaj | Repository setup; feasibility and data check; report sections 1, 3, 5, 6, 7 and 9 and the references; data samples and output mockups; slide revision; README | [project-report.md](documentation/milestone%201/project-report.md), [README.md](README.md) | Puneet Bajaj |
| Manas Rastogi | Lookalike species pairs: iNaturalist confusion counts, distinguishing features and how they feed the evaluation; report section 3.6 and references 29 to 32 | [project-report.md, section 3.6](documentation/milestone%201/project-report.md#36-lookalike-species-pairs) | Manas Rastogi |

All members must also sign the acknowledgment page at the end of the [Milestone 1 project report](documentation/milestone%201/project-report.md) and the last slide of the presentation.

## Team review and acknowledgment

By signing, each member confirms that they have reviewed this log, that their own entries are accurate, and that they agree with the submission.

| Team member name | Roll Number | Signature / acknowledgment | Date |
|---|---|---|---|
| Jayesh Bansal | 24f100185 |![Jayesh Bansal](https://drive.google.com/uc?export=download&id=1imepTm_t9vgJVL367tWU-LPYcLq9Zet0)|22/10/2026|
| Vikas | 23f3001800 | Vikas | 22/10/2026 |
| Puneet Bajaj | 23f3002416 | Puneet Bajaj | 22/10/2026 |
| Manas Rastogi | 22f3001477 | ![Manas Rastogi](https://drive.google.com/uc?export=download&id=1j9vC0yjI-eb3lGfETUdnK3GAXq77__lT) | 22/10/2026 |

