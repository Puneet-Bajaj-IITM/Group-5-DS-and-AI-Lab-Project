# SerpentGuard app (Flutter)

The phone app for SerpentGuard AI. Android first; it also builds for iOS, with the limits noted below.

**Status:** a working front end with the safety features. Species identification is a **demo** stand-in until the trained model is added, and every result says so on screen.

## What works

| Feature | Where | Notes |
|---|---|---|
| Identify from camera or gallery | `screens/identify_screen.dart`, `screens/result_screen.dart` | Verdict (dangerous / not on the WHO list / we don't know), top 3 species, WHO category. Uses `DemoClassifier` for now |
| Live camera | `screens/live_review_screen.dart` | Continuous preview, separate from photos; `LiveSnakeClassifier` is the frame adapter for verified model integration. Select 1–3 stills for one review encounter |
| Community | `screens/community_screen.dart`, `services/voting_queue.dart` | SQLite drafts/outbox, secure remembered accounts, Android background sync, cached species/genus voting, conflict reconciliation and consensus status |
| Verdict rules | `models/verdict.dart` | Same shape as `src/serpentguard/decide.py`; the thresholds are placeholders |
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

## Adding the trained model

1. Use the supplied **FP32 M4 bundle**: classifier, detector **and segmenter**, `pipeline.json`, `tables.json` and `reference_pipeline.py`. Keep all model assets under ignored `assets/models/` or `outputs/`, and keep the ZIP out of Git.
2. Implement `SnakeClassifier.identify` using the reference's exact exposure correction, letterbox/normalisation, both-network gate, mask cleanup/crop and A3 scoring. The M3 detector-only crop is not the M4 reference pipeline.
3. Replace the placeholder `DecisionRule` with the supplied A3 calibrated rules, including WHO/family/genus heads, prototype scoring, class ordering and messages. The supplied decision and photo goldens must match on Android before removing the demo state.
4. Implement `LiveSnakeClassifier` for continuous frames. The live screen supplies copied planes, row/pixel strides, format, dimensions, sensor/device orientation, lens direction and timestamps; it drops new frames while inference is busy. The adapter owns preprocessing and the actual temporal rules, reset by `resetLiveSession`. Late results after camera restart are ignored.
5. Swap the verified adapter into `main.dart`, then run a real video on a phone. Verify unknown/no-snake handling, orientation, pause/resume, memory and latency. Photo parity does not establish Android video parity.

The supplied Python reference passed 200/200 decision fixtures and 5/5 SHA-1-checked photo fixtures here; all five model/config hashes matched. The supplied `parity.json` reports an additional 300-photo PyTorch comparison, which was not rerun here. `scripts/verify_mobile_bundle.py` reproduces the smaller verification without committing weights or photos. Jayesh owns real Android model wiring and video acceptance.

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
