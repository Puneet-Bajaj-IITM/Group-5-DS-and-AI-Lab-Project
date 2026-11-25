# SerpentGuard app with the M5 pipeline (demo copy)

This branch is cut from `main` (4f69f99). It carries the M5 training changes and
`m5-g5`'s uncommitted video-lock work (`app/lib/inference/video_lock.dart` and its tests, plus `src/serpentguard/video.py`
and `tests/test_video_lock.py`, which the Dart lock is ported from).

The models, cut-offs, tau and lock settings are the frozen M5 pipeline (`results/m5/test/FREEZE.md`). **Two changes
come after the test score and are not part of it** (see the last two sections). The official M5 test number stays 0.926.

## What changed in the app

| Step | Before (M4, A3) | Now |
|---|---|---|
| No-snake decision | M4 gate: box_conf >= 0.00499 and mask_area >= 4.8e-05 | **snake check** `snake_check_fp16.onnx` (distil_oi, 256 px letterbox): snake if `probs[0] >= 0.20484977503483762`. When it says no snake, the detector, U-Net and classifier are skipped |
| Crop | detector + U-Net, crop to the cleaned mask's box | same models, used for the crop only; no mask gives the whole photo |
| Classifier | A3, one pass | **M5_KD**, run on the crop and on its mirror image; every head averaged, embedding re-normalised, then `species_logits += 0.5 * log(training count)` |
| Prototypes (unfamiliar check) | A3's | M5_KD's `plain` prototypes (`tables.json`) |
| Decision thresholds | A3 refit | **stricter post-test refit** (bound 0.97); the frozen refit is stored beside it as `frozen_thresholds` |
| "Unknown" on screen | amber "We don't know" | **red "Could be dangerous"**: keep away, treat it as venomous, the SOS / first-aid button (post-test) |
| Video lock | M4 lock | `m5-g5` lock (window 0.5 s, margin 0.05, switch 3 s, hold 2 s); lock confidence = `theta_conf` 0.354 |

Code: `lib/inference/pipeline.dart` (snake check, mirror pass), `lib/inference/flip_prior.dart` (mirror + prior),
`lib/inference/bundle.dart` (settings), `lib/inference/image_ops.dart` (`RgbImage.mirrored`), `lib/models/verdict.dart`,
`lib/widgets/common.dart` and the result, video and map screens (unknown warning). Every number is read from
`assets/model/pipeline.json`; none is typed into Dart.

Tools in `app/tool/m5/` (Python, `m3-final/.venv`):

- `build_assets.py`: writes `pipeline.json` and `tables.json` from the committed A3 versions and copies the four ONNX
  files into `assets/models/`, checking each source against its FREEZE.md SHA-256 and the prior vector against its
  hash. The tables equal `results/m5/test/g5/tables_frozen.json`.
- `reference_m5.py`: the Python reference of the app path: `results/m5/test/g5/reference_frozen.py`'s classifier
  (mirror + prior) plus the snake check, with the app's own settings and model files.
- `make_golden_m5.py`: the golden files; `pick_nonsnakes.py`: picks the validation non-snakes.
- `refit_strict.py`: the stricter thresholds (below).

## Parity: Python reference vs Dart

| Check | n | Result |
|---|---|---|
| Decision goldens (`test/golden_decisions_test.dart`): M5_KD saved **validation** outputs, plain + mirrored, through Dart's mirror + prior + rules vs the repo's `decide.verdict` | 200 rows, 15 verdict/reason/fallback combinations | all match with the app thresholds and all match with the frozen ones (21 rows differ between the two sets); probabilities within 1e-6, 0 verdict flips |
| Controls on the same rows: no mirror pass; no prior; theta_conf + 0.05 | | each fails some rows, so the check can fail |
| Mutation check: drop the head averaging, or scale the prior by 0.9 | | both fail the decision goldens |
| Snake-check input built by Dart on the test host vs Python (`test/golden_photo_inputs_test.dart`, `GOLDEN_PHOTO_DIR` set) | 15 photos | mean and std within 4.2e-6 (tolerance 1e-4) |
| fp16 vs fp32 snake check, Python | 15 photos | max P(snake) difference 6.0e-4, 0 calls flip |
| Whole path on the device (`integration_test/golden_photos_test.dart`): P(snake) and top-3 probabilities within 1e-3, same snake call, verdict, frozen verdict, message, top-3 species and crop | 15 photos | **not run**, see "Not tested" |

Golden photos (`test/golden/golden_photos.json`, downloaded by URL and checked by SHA-1, not in git): the five demo
photos and ten non-snakes from notebook 19's **validation** split (one each of person, rope, cat, dog, hand/arm, eel,
legless lizard, caecilian, plus two that the snake check called a snake on validation). The test split was not read.
The non-snake URLs serve the full-size originals, not the 500 px copies validation scored, so their P(snake)
differs from the validation scores (hand/arm 0.117 then, 0.456 now; legless lizard 0.013 then, 0.792 now).

| Photo | Set | P(snake) | App verdict | Top species | Frozen thresholds' verdict |
|---|---|---:|---|---|---|
| 228115560 | demo | 0.943 | dangerous | *Naja naja* | dangerous |
| 557437621 | demo | 0.979 | not_listed | *Coelognathus helena* | not_listed |
| 462345804 | demo | 0.966 | dangerous | *Elaphe taeniura* | dangerous |
| 22679235 | demo | 0.980 | not_listed | *Ahaetulla prasina* | not_listed |
| 640148271 | demo | 0.963 | dangerous | *Fowlea piscator* | unknown |
| cdf060cec6f45e89 | person | 0.014 | no snake | - | no snake |
| d3bf97879410e33a | rope | 0.018 | no snake | - | no snake |
| 4b961ac47fbd8694 | cat | 0.011 | no snake | - | no snake |
| 27028f72bf18f900 | dog | 0.012 | no snake | - | no snake |
| 76367a8aae3925ea | hand/arm | 0.456 | dangerous | *Ptyas mucosa* | unknown |
| 58773227 | eel | 0.047 | no snake | - | no snake |
| 721342637 | legless lizard | 0.792 | not_listed | *Indotyphlops braminus* | not_listed |
| 53179163 | caecilian | 0.022 | no snake | - | no snake |
| 806be9da35aca2bc | other animal | 0.968 | dangerous | *Fowlea piscator* | dangerous |
| 22679134 | caecilian | 0.912 | unknown | *Elaphe taeniura* | unknown |

## Tests

`flutter test --no-pub` in `app/`: **81 passed, 3 skipped** (82 passed, 2 skipped with `GOLDEN_PHOTO_DIR` set);
before these changes 69 passed, 2 skipped. Python `tests/test_video_lock.py`: 10 passed. `flutter analyze`: no issues.
`pubspec.yaml` and `pubspec.lock` are unchanged: a plain `flutter test` re-resolves the lock on this machine's
Flutter SDK (it pins `test_api` 0.7.14 and adds `listen`), so the tests were run with `--no-pub` and the lock restored.

The video-lock parity fixture holds A3 outputs of dev clips, so `test/video_lock_parity_test.dart` now reads the A3
thresholds and prototypes it was made with (`test/golden/video_lock_parity_a3.json`) rather than the app's.

## Model asset sizes

| | Before (A3 app) | Now |
|---|---:|---:|
| classifier.onnx | 20,826,597 (A3) | 20,826,597 (M5_KD, same architecture) |
| detector.onnx | 9,680,263 | 9,680,263 |
| segmenter.onnx | 26,498,348 | 26,498,348 |
| snake_check_fp16.onnx | - | 8,064,356 |
| **Models total** | **57.0 MB** | **65.1 MB** (+8.1 MB) |
| pipeline.json + tables.json | 1,045,250 bytes | 1,050,436 bytes |

The fp16 segmenter (13.3 MB) would save 13.2 MB but is not what the frozen pipeline scored, so it is not used.

## Running the demo on the Android emulator

```sh
cd work-dirs/ds-lab-feasibility
python app/tool/m5/build_assets.py     # models into assets/models (not in git)
~/Library/Android/sdk/emulator/emulator -avd Medium_Phone_API_36.1 &
cd app
flutter run --no-pub -d emulator-5554
```

The golden-photo check on the emulator (needs network for the photo URLs):

```sh
python3 tool/golden_defines.py > build/golden_defines.json
flutter test --no-pub integration_test/golden_photos_test.dart -d emulator-5554 \
  --dart-define-from-file=build/golden_defines.json
```

To rebuild the goldens: `python app/tool/m5/make_golden_m5.py --photos <folder of <photo_id>.jpg>`
(the photos used are in `work-dirs/ds-lab-feasibility/m5-app-golden/photos`).

## Not tested

- **The integration test on the emulator.** The emulator booted (`Medium_Phone_API_36.1`), but the first Android build
  needs Gradle 9.3.1, which is not cached on this laptop, and its download from services.gradle.org timed out. So the
  whole path in Dart (ONNX Runtime on Android, BitmapFactory decode, mirror pass, snake check) is untested against
  the Python reference; only the decision rules, the mirror + prior step and the snake-check input are.
- **fp16 on Android.** The snake check has fp16 weights with fp32 input and output; it matches fp32 on the laptop
  (above) but has not run on ONNX Runtime for Android.
- **On-device latency.** The mirror pass doubles the classifier's time per photo and per video frame; the snake check
  adds one 256 px run but skips the other three models on most non-snake photos. Not measured.
- The video screens with the new lock and the frozen classifier, on a device.
- The video screen still says the answer "pools the last 5 frames"; the lock now pools 0.5 s. Not changed here.

## Post-test safety change: stricter thresholds

POST-TEST, app copy only. `tool/m5/refit_strict.py` repeats `results/m5/g3/refit_new.py` for "KD flip + prior": the
same app-gate setup (the M4 gate's call per photo from `m5-targets/app_outputs`), `decide.fit_thresholds` on the
7,999 **calibration** photos only, target raised in steps of 0.0025 until the WHO recall lower bound behind the gate
reaches 0.97. It stopped at target 0.9775 (calibration bound 0.9751; the step before gave 0.9645). Only `theta_d`
moves: 0.0920 to 0.0394. Checked on **validation** only (17,143 photos, 2,131 WHO-listed); the test split was not read.
A control with the calibration logits shuffled across photos gives a bound of 0.368, so the fit can fail.

| Validation, behind the M4 gate | Frozen refit (bound 0.95) | Stricter refit (bound 0.97) |
|---|---:|---:|
| WHO recall | 0.962 (95% interval 0.950 to 0.973) | **0.979** (0.969 to 0.988) |
| WHO recall lower bound (5th percentile) | 0.952 | **0.971** |
| False alarms (safe species called dangerous) | 21.9% | **30.1%** |
| Verdicts dangerous / unknown / not listed | 6,608 / 1,898 / 8,637 | 7,767 / 1,335 / 8,041 |
| Calibration: bound, false alarms | 0.954, 18.7% | 0.975, 25.5% |

The frozen numbers reproduce FREEZE.md (0.9625, 0.9498 to 0.9732, bound 0.9519). The app's `pipeline.json` holds the
stricter set as `decision.thresholds` and the frozen set as `decision.frozen_thresholds`. These numbers are behind
the M4 gate, as the refit is defined; the app decides "snake" with the snake check, and safety behind the check was
not recomputed for the stricter set.

## Post-test change: "unknown" is a warning

POST-TEST. When a snake is found and the verdict is "unknown", the app now shows **"Could be dangerous"** with
"Keep away and treat it as venomous. If anyone is bitten, go to hospital now.", in the same red style as
"Dangerous", with the SOS / first-aid button (it already escalated). The same red is used for unknown alerts on the
map and the nearby-alert banner. A photo with no snake is not a warning: it shows a neutral grey "No snake found"
with "This does not mean the area is safe" (the emergency button stays). The decision rules and their messages are
unchanged, so the golden files did not change for this; `test/unknown_warning_test.dart` covers the new screens.

Counting a shown "unknown" (snake found) as a warning, on validation behind the M4 gate:

| | Frozen refit | Stricter refit |
|---|---:|---:|
| WHO-listed snakes warned (dangerous or unknown) | 0.972 (bound 0.964) | 0.982 (bound 0.975) |
| Safe species warned | 33.6% | 37.8% |
