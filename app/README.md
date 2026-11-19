# SerpentGuard app (Flutter)

The phone app for SerpentGuard AI. Android first; it also builds for iOS, with the limits noted below.

**Status:** a working front end with the safety features. Species identification is a **demo** stand-in until the trained model is added, and every result says so on screen.

## What works

| Feature | Where | Notes |
|---|---|---|
| Identify from camera or gallery | `screens/identify_screen.dart`, `screens/result_screen.dart` | Verdict (dangerous / not on the WHO list / we don't know), top 3 species, WHO category. Uses `DemoClassifier` for now |
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

1. Export the detector and SnakeID to ONNX (notebook 16 already does this: fp16 SnakeID, int8-weight YOLO26n) and put them in `assets/models/`.
2. Add an ONNX runtime package and write an `OnnxClassifier implements SnakeClassifier` (`services/classifier.dart`). It should letterbox to 320×320, crop to the detector box +6%, run SnakeID, then return the species probabilities.
3. Load the shipped thresholds from `results/m3/final/H1_top1_s1/decision_metrics.json` into `DecisionRule`.
4. Swap `DemoClassifier` for it in `main.dart`. The demo badge disappears because `isDemo` becomes false.

`assets/species.json` is generated from the repo's species cards by `python3 tool/build_species_asset.py`. Rerun it whenever the class list or the danger table changes.

## Known limits

- **Alerts work only while the app is open.** There's no background service yet.
- **iPhones can receive alerts but not send them.** iOS doesn't allow manufacturer data in an advertisement.
- **Alerts use the Bluetooth SIG's test company id, `0xFFFF`.** A released app needs its own id.
- **Hospital lists depend on public OpenStreetMap servers, which are often busy.** The planned fix is a bundled per-state hospital file.
- **There's no bulk "download my district" for the map.** OSM's tile policy forbids it, so we'd need our own tile source.
- **The WHO first-aid text isn't included.** It's waiting on the licence decision (see the Milestone 3 report, section 7).
