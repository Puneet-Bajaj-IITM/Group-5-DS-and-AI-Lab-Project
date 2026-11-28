# Vikas: rescuer/vet search and Bluetooth evidence

Updated 27 November 2026. Scope follows the team's 26 November assignments.
Other members' training, photo audits, device speed/RAM and startup-video work
are not changed here.

## Rescuer and veterinarian directory

The Contacts screen searches a bundled pilot directory by organization, locality
and state, filters rescuers/vets, saves numbers on the phone and opens the dialler.
No network is needed for search or the first load. A reference cache uses the
existing Store/SharedPreferences mechanism, falling back to the reviewed bundle
when a cache is corrupt or changed. Personal SOS recipients remain separate.
Veterinary contacts are excluded from the human snakebite help list.

Ten public organization numbers were checked against these official pages on
**25 November 2026**, not by telephone:

- [Wildlife SOS](https://wildlifesos.org/contact-us/): Delhi, Agra, Vadodara, Srinagar.
- [RESQ](https://www.resqct.org/contact): Pune, Baramati, Roha, Nashik.
- [TANUVAS](https://www.tanuvas.ac.in/contacts.php): Madras Veterinary College
  Teaching Hospital and Veterinary University Peripheral Hospital, Chennai.

The asset stores source URL, source-check date and verification method. Records
older than 180 days show a recheck notice. Only RESQ Pune has a coordinate, from
the map embedded on its contact page. Distance sorting and radius filters use
straight-line distance to that centre. The other nine regional/facility records
have no invented coordinates; radius filters exclude them, with this explained
on screen. Locality search still works without GPS. This is not national coverage,
road routing or a guarantee of availability/capability. Source pages need internet
when opened. Vets treat animals, not people with snakebite.

Source: `app/tool/build_contact_directory.py`; generated asset:
`app/assets/contact_directory.json`. Tests cover search, radius/no-coordinate
handling, stale dates, cache repair, separate veterinary types, narrow-screen
save actions and the human SOS exclusion. The earlier 25 November app test
receipts apply to the older checkout; they do not certify this latest M5 merge.

Current-branch check, 27 November: [private Kaggle run](https://www.kaggle.com/code/vikas25s/sg-m5-vikas-assigned-checks)
version 1 completed with **six targeted contact/SOS tests passing** and **no
Flutter analysis issues**, including static analysis of the BLE probe. The
[receipt](../../results/m5/vikas_app/run.json) and source hashes identify the
checked files. This cloud harness substitutes artwork/video fixtures and omits
font registration; it is not a release build, physical BLE test or whole-app
test-suite result. The contact generator also reproduced its asset byte-for-byte,
and `results/m5/check_numbers.py` passed without changing the report's model results.

## Device availability and evidence status

The owner currently has **one phone**. ADB enumeration on 27 November found
**no connected devices**. No physical BLE result is claimed. One phone can check
local scan and advertiser startup. It cannot establish inter-phone delivery,
relay behaviour, loss rate or range. The app's existing "Simulate an alert"
button is UI simulation; it never establishes RF delivery.

### One-phone hardware probe

Connect the Android phone by USB, enable USB debugging and accept its RSA prompt.
On Windows, `adb devices -l` must show `device`, not `unauthorized`. Enable
Bluetooth and grant requested permissions. In a Flutter development environment:

```sh
cd app
flutter test integration_test/ble_radio_probe_test.dart -d DEVICE_ID \
  --dart-define=BLE_PROBE_ROLE=self --dart-define=BLE_PROBE_SESSION=27110001
```

The integration test starts a foreground scan and advertiser for thirty seconds,
checks plugin-reported state and always attempts cleanup. It emits a JSON
`BLE_PROBE_RECEIPT` line. Save the full command output with the phone model,
Android version, app commit and date. Success means **local startup only**.
No trained model is needed by the probe.

Probe packets use **ST** magic instead of production **SG**. Production decoding
rejects them; no fake snake warning, real location, saved sighting or relay queue
is created. A shared session ID isolates a run. Packet format/version otherwise
matches the twenty-byte advertisement layout. Manufacturer ID 0xFFFF is the
existing test identifier, not a production company allocation.

### Two-phone radio probe when a second phone is available

Run the test first on B with `BLE_PROBE_ROLE=receiver`; while it is listening,
run it on A with `BLE_PROBE_ROLE=sender`, using the same fresh session ID on both.
Both windows last thirty seconds; allow for build/install time and start them
close enough to overlap. The receiver requires a matching ST packet to pass.
Use `--no-pub` after dependencies are resolved if appropriate; do not confuse
build time with radio latency. Reverse A/B and repeat. Save receipts from both
phones, distance/environment and app commit. Matching scan reports can include
repeats; their count is not a unique-packet delivery ratio or a latency metric.

This isolated probe verifies radio transport only. To accept the **application**
path, also use two phones running the same real-model app build: disable Wi-Fi
and mobile data, keep Bluetooth/location and foreground app active, enable nearby
alerts on both, identify a real saved snake photo on A and send its alert. Record
the queued warning on A and received alert/RSSI/time on B, then reverse direction.
Check duplicate handling, relay-off behaviour and turning alerts off. Do not use
DemoClassifier results or the simulation button as transmitted evidence. Do not
stage a live snake encounter for this test. Two phones cannot prove a three-node
relay chain or justify a measured-range claim without a separate experiment.

| Check | Current result | Needed evidence |
|---|---|---|
| Directory source review | Website checked 25 Nov; not called | Call-check coverage, availability and capability |
| One-phone local BLE startup | Not run; no device connected | Hardware probe receipt and device details |
| A → B and B → A radio delivery | Not run; second phone unavailable | Paired ST probe receipts |
| Production app alert delivery | Not run | Screens/recording from both phones, matching alert and commit |
| Offline directory on physical phone | Not run | Airplane-mode search, denied GPS, saved contact after restart, dialler |

No scan/advertise receipt, delivery result or another member's signature is
filled in ahead of the actual test.

## Integration with current main — 28 November 2026

Merged `main` at `ee9c49c`, preserving AniDex and the team report updates.
Kaggle version 2 passed six targeted contact/SOS tests and clean Flutter analysis,
including the BLE probe. All 63 recorded app source/config hashes matched.
[New receipt](../../results/m5/vikas_app/main_merge/run.json); the earlier receipt
above remains the historical 27 November run. The same cloud fixture limitations
apply. Report-number checks also passed. No physical Bluetooth result is added.
