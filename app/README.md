# SerpentGuard app (Flutter)

The phone app for SerpentGuard AI. Android first; it also builds for iOS, with the limits noted below.

**Status:** the shipped model (A3 behind the M4 front end, fp32 ONNX) runs on the phone for photos, the live camera and video files. If the model files are not copied in before a build, the app falls back to a demo that is clearly marked on every result.

## What works

| Feature | Where | Notes |
|---|---|---|
| Identify from camera, gallery or video | `screens/identify_screen.dart`, `screens/result_screen.dart`, `screens/video_check_screen.dart` | Verdict (dangerous / not on the WHO list / we don't know), top 3 species with regional names, WHO category, genus fallback. Real model: `OnnxSnakeClassifier` |
| Live camera | `screens/live_review_screen.dart`, `inference/live_session.dart` | Continuous preview with a box on every snake in view and the verdict on the one it is about (two model lanes, below). Select 1–3 stills for one review encounter |
| Community | `screens/community_screen.dart`, `services/voting_queue.dart` | SQLite drafts/outbox, secure remembered accounts, Android background sync, cached species/genus voting, conflict reconciliation and consensus status |
| Verdict rules | `inference/decision.dart`, `models/verdict.dart` | DR0 with A3's thresholds, ported from `reference_pipeline.py`; 200/200 golden decisions |
| Location | `services/location_service.dart` | Foreground only. Tags sightings, fills the SOS message, finds hospitals |
| SOS | `screens/sos_screen.dart`, `services/sos_service.dart` | Calls 112 / 108 / 102, texts your location to emergency contacts, has an "I've been bitten" path. Always opens the phone's dialler or SMS app; never calls on its own |
| Nearby hospitals | `services/hospital_service.dart` | OpenStreetMap via Overpass, falling back to Nominatim. The last list is saved for offline use. Specialty-only hospitals (eye, dental…) are listed last. OSM does not record anti-snake venom stock |
| Map | `screens/map_screen.dart` | OSM tiles, cached as you browse (30 days, up to 300 MB). Shows you, hospitals, your sightings and alerts |
| Bluetooth alerts | `services/ble_alert_radio.dart`, `services/alert_book.dart`, `models/snake_alert.dart` | Opt-in. A 20-byte BLE advertisement carrying a random id, species, verdict, location rounded to ~100 m and the time. Relayed up to 2 hops, expires after 60 min, ignored beyond 5 km, and limited to 3 sends per hour |
| Contacts | `screens/contacts_screen.dart` | Emergency contacts, snake rescuers, forest offices, doctors. Stored on the phone only |

## Run it

```bash
cd app
flutter pub get
flutter test
flutter run            # with an Android phone or emulator connected
```

Building needs Android SDK platform 37 (`platforms;android-37.0`). `permission_handler` requires it, and `android/build.gradle.kts` maps its `android-37` request onto the installed `android-37.0`.

Testing Bluetooth alerts needs **two real Android phones**, because emulators can't advertise to each other. On a single phone, use **Alerts → Simulate an alert from another phone**.

## The model

The path from photo to verdict is a Dart port of the model bundle's `reference_pipeline.py` (`lib/inference/`). The full write-up is [section 7.10 of the Milestone 4 report](../documentation/milestone%204/project-report.md#710-the-model-in-the-app).

1. Copy `classifier.onnx`, `detector.onnx` and `segmenter.onnx` (fp32) from the bundle's `models/` into `assets/models/`, and check them against its `SHA256SUMS`. They are git-ignored (54 MB). `assets/model/pipeline.json` and `tables.json` are committed.
2. `flutter test` runs the 200 golden decisions (`test/golden_decisions_test.dart`). With `SG_CV_FIXTURES` set, it also runs the pixel checks against cv2 (`test/image_ops_test.dart`; make the fixtures with `tool/make_cv_fixtures.py` and opencv-python-headless 5.0.0.93).
3. The whole path on a device, against `golden_photos.json` (the test downloads the 5 photos and checks their SHA-1):
   ```bash
   python3 tool/golden_defines.py > build/golden_defines.json
   flutter test integration_test/golden_photos_test.dart -d <device> --dart-define-from-file=build/golden_defines.json
   ```
4. **Video:** Identify → Check a video. The app reads up to 10 frames a second for the first 20 seconds (`MediaMetadataRetriever`, in `android/.../VideoFrames.kt`) and applies the M4 video lock. The live camera uses the same lock.
5. **Live camera, two lanes** (`lib/inference/live_session.dart`, `android/.../LiveModels.kt`). The box lane runs the detector on as many camera frames as it can take and tracks every box (`lib/inference/detections.dart`; display only). The verdict lane, whenever it is free, runs the segmenter, gate, mask crop, classifier, DR0 and the lock on the newest frame the box lane finished, reusing its detector output. Both lanes have their own ONNX Runtime sessions on their own threads, because `flutter_onnxruntime` runs every call on one queue. A frame whose detector score is under the gate's box cutoff is "no snake" whatever the mask says, so it skips the verdict lane. Long-press the status bar to show each lane's rate and time. Where the lanes cannot open (not Android), the screen falls back to `OnnxSnakeClassifier.identifyFrame`. Check the lanes against `flutter_onnxruntime` and time them on a device:
   ```bash
   flutter test integration_test/live_lanes_test.dart -d <device>
   ```

On Android, photos are decoded with `BitmapFactory`, because Flutter's decoder there gives different pixels from PIL on some JPEGs. Regional names come from `tool/regional_names.csv`, a draft until native speakers have checked it.

## Offline community setup

Capture works without an account: confirmed evidence is saved as a local SQLite draft. Attaching drafts to an account requires confirmation. UUIDs, manifests and JPEG bytes stay unchanged on retries. Model predictions from the demo are excluded from community submissions.

Provision tokens and start the SQLAlchemy backend from the repository root:

```bash
PYTHONPATH=src .venv/bin/python -m serpentguard.voting_api --db outputs/community-voting.sqlite3 add-user citizen
PYTHONPATH=src .venv/bin/python -m serpentguard.voting_api --db outputs/community-voting.sqlite3 serve
```

Use `http://10.0.2.2:8080` on the emulator. For USB, run `adb reverse tcp:8080 tcp:8080` and use `http://127.0.0.1:8080`. Debug HTTP is limited to local hosts; remote servers require HTTPS. Review with a separate account because self-voting is rejected. The backend is supplied by PR #19.

Remembered tokens use secure storage; WorkManager receives only an account identifier. SQLite leases prevent simultaneous foreground/background drains. Session-only mode and manual Sync are available; disconnect clears saved credentials and cancels scheduled jobs. Feed pagination caches at most 100 encounters, and every still must be cached before reviewing. Vote-edit conflicts require `/my-vote` reconciliation. Cached consensus is labelled potentially stale. Labels need curator approval before training export; no automatic retraining occurs.

Run the real backend integration test with `SERPENTGUARD_PYTHON="$PWD/../.venv/bin/python" flutter test`. The Python environment needs `requirements-voting.txt`; without that environment variable, the backend integration test is skipped.

`assets/species.json` is generated from the repo's species cards by `python3 tool/build_species_asset.py`. Rerun it whenever the class list or the danger table changes.

## Known limits

- **Alerts work only while the app is open.** There's no background service yet.
- **iPhones can receive alerts but not send them.** iOS doesn't allow manufacturer data in an advertisement.
- **Alerts use the Bluetooth SIG's test company id, `0xFFFF`.** A released app needs its own id.
- **Hospital lists need downloaded regional coverage.** SQLite/R-tree searches preserve saved regions across restarts, but no complete regional bundle or offline road graph is included. Hospital tags do not establish current antivenom stock.
- **There's no bulk "download my district" for the map.** OSM's tile policy forbids it, so we'd need our own tile source.
- **Reviewed regional first-aid text is not bundled.** The card loader expects an approved English/Hindi/Tamil/Telugu pack at `assets/first_aid.json`. Native Android speech selects an installed offline voice for the actual text language; missing content/voices are reported.
