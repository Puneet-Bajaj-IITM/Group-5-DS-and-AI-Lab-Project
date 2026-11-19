# Milestone 4: the phone app and the shipped model

SerpentGuard AI · Group 5 · section for the Milestone 4 report (app and backend)<br>
Written by Jayesh Bansal. To be merged into the Milestone 4 report.

## Summary

The Flutter app now runs the model we ship, A3, behind the Milestone 4 front end, entirely on the phone. A photo, a live camera feed or a video file goes through the same path as the team's Python reference: exposure fix, detector and U-Net gate, crop to the cleaned mask, MobileViT-S classifier, and the DR0 decision rules with A3's thresholds. Videos and the live camera use the Milestone 4 video lock.

| Check | Result |
|---|---|
| Decision rules against `golden_decisions.json` (no model needed) | 200 of 200 cases: same verdict, reasons, message, genus fallback and top 3; probabilities within 1e-6 |
| Image steps against OpenCV (`cv2` 5.0, as pinned) on the 5 golden photos | Letterbox, detector input, resizes, mask, cleanup, LAB, CLAHE and exposure fix identical, except a one-level difference on under 0.1% of pixels in two INTER_AREA downscales |
| Whole path on Android (emulator, `integration_test`) against `golden_photos.json` | 5 of 5 photos: same gate, crop window, top 3, verdict and message (including "11% combined probability"); probabilities within 0.01 |
| Time per photo on the emulator (Apple M-series host, arm64 Android 15 image) | 0.5 to 1.0 s for the three models; about 0.4 to 0.7 s per video frame at 640 px |
| Video: *Naja naja* clip (Wikimedia Commons, CC BY 2.0) | **Dangerous, WHO Category 1, Spectacled cobra** from the first frame; locked, 0 changes over 200 frames (20 s) |
| Video: *Gonyosoma oxycephalum* clip (Wikimedia Commons, CC BY-SA 2.0) | **Not on the WHO list, *Gonyosoma oxycephalum***, first answer at 0.4 s, 0 changes over 67 frames |

Both clips are test clips from Milestone 4's video study, and the app gives the same answer as `results/m4/video/A3/video_lock.csv`. The screen recordings are kept out of git, because they show other people's videos; they are in `outputs/demo/` on Jayesh's machine.

## Who built what

| Part | Built by | Where |
|---|---|---|
| A3 training, M4 front end (detector, U-Net, gate), video lock study | Puneet Bajaj | [PR #20](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/20) |
| Model bundle for the app: ONNX export, `pipeline.json` and `tables.json`, `reference_pipeline.py`, golden decisions and photos, parity check against the repo (300/300 photos) | Puneet Bajaj | handed over as `serpentguard-app-model.zip` (not in git: 54 MB of models) |
| A3 full training run | Kaggle account `jayeshbansal497` (`sg-m4-full-a3`) | Kaggle |
| Flutter app: identify flow, SOS, hospitals, offline map, contacts, Bluetooth alerts | Jayesh Bansal | [PR #21](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/21) |
| Splash screen and app branding | GitHub user Moon-Wrecker | [PR #22](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/22) |
| Community voting backend | Vikas | [PR #19](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/19) |
| Offline community voting, live camera screen, model interfaces in the app | Vikas | [PR #24](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/24) |
| Real model in the app: Dart port of the pipeline and decision rules, ONNX Runtime, Android decoding, video check, live-camera adapter, golden tests, regional names | Jayesh Bansal | [PR #MODEL_PR](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/pull/MODEL_PR) |

## How the app runs the model

| Step | Code | Notes |
|---|---|---|
| Decode | `SnakePipeline.decode` | Android's `BitmapFactory`, upright by EXIF. Flutter's own decoder gave different pixels on 3 of the 5 golden photos on Android, enough to move a detector score from 0.54 to 0.32 |
| Exposure fix, letterbox, detector input, mask cleanup, crop | `lib/inference/image_ops.dart` | Plain Dart following OpenCV's algorithms, so `flutter test` needs no native build. `INTER_LINEAR` is OpenCV's bit-exact resize: the cv2 wheel itself rounds differently on ARM (carotene) and x86 (IPP), and on the golden photos the generic version moves no answer and probabilities by under 0.001 |
| Detector, segmenter, classifier | `lib/inference/pipeline.dart` | `flutter_onnxruntime` 1.9.0, fp32 models as the bundle asks |
| Decision | `lib/inference/decision.dart` | `decide`, `danger_note` and `verdict` ported line by line; it replaces the placeholder rule the app had |
| Video and live camera | `lib/inference/video_lock.dart`, `lib/services/video_analyzer.dart` | Pools the last 5 snake frames, locks at θ_conf with a 0.05 lead, switches only after 1 s; at most 10 frames a second, as in M4 |

The settings come only from `assets/model/pipeline.json` and `tables.json`, which are committed. The three `.onnx` files are not: they are copied into `app/assets/models/` before a build. Without them, the app still runs, but every result is clearly marked as a demo.

## How to run it

```bash
cd app
flutter test                                   # 45 tests, including the 200 golden decisions
SG_CV_FIXTURES=<folder> flutter test test/image_ops_test.dart   # pixel check against cv2 (see tool/make_cv_fixtures.py)
python3 tool/golden_defines.py > build/golden_defines.json
flutter test integration_test/golden_photos_test.dart -d <android device> --dart-define-from-file=build/golden_defines.json
flutter run                                    # Identify → Take a photo / Live camera / Check a video
```

## Regional names

`species.json` now carries names in Hindi, Marathi, Bengali, Tamil, Telugu, Kannada and Malayalam for 9 of the 16 WHO-listed species: the Big Four in up to all 7, plus both king cobras, the monocled cobra, the banded krait and the hump-nosed pit viper. The other 7 (two Western Ghats pit vipers, the red-tailed pit viper, the Levantine viper, the Oxus cobra, and the black and Wall's kraits) have no regional name we were sure of, so they are left out rather than guessed. **All names are a draft until a native speaker of each language checks them** (`app/tool/regional_names.csv`).

## Limits

- **No phone has been measured yet; only the emulator has.** Speed and memory on a mid-range Android phone are still to check, as is a real outdoor video filmed with the live camera.
- **The debug APK is 339 MB** (every CPU architecture plus 54 MB of models). A release build for one architecture is far smaller, but it has not been measured.
- **Full int8 is not used, and fp16 is not used either:** the bundle measured fp16 changing 1 verdict and 4 top species in 300 photos.
- **The dark golden photo is fragile.** One-level pixel differences in its exposure fix moved the top probability from 0.26 to 0.14 in a Python check. That is why the LAB and CLAHE steps were ported bit for bit.

## AI assistance (appendix row)

| Tool | Used by | Used for | How the output was checked |
|---|---|---|---|
| OpenAI Codex | Jayesh Bansal | Earlier drafts: the Milestone 1 dataset and licence notes, and a Milestone 3 report draft that the team's version replaced | Reviewed and rewritten by the team before submission |
| Claude Code (Anthropic) | Jayesh Bansal | The Flutter app (PR #21) and the model integration in this section: the Dart port, tests, video screen and this write-up | Every number above comes from a test or run in the PR: golden decisions, cv2 pixel checks, golden photos on Android and the two demo clips |
