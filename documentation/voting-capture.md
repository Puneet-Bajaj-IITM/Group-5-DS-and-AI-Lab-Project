# Community review from photo and live video

One sighting represents one snake encounter and gets one community ballot. The M4 live-camera lock pools model outputs to steady the display; it is separate from community voting. A reviewed sighting can have multiple stills, and each reviewer submits one species or genus identification for the sighting. [M4 report, section 7](milestone%204/project-report.md#7-video-the-lock) · [Lean Multiclass Crowdsourcing](https://authors.library.caltech.edu/records/5qptj-5zy81)

| Input | Evidence to queue on Android |
|---|---|
| Photo | When a snake is detected and the identification is uncertain, offer to share a JPEG crop at useful resolution. Keep enough margin to show identifying features. |
| Live video | Keep a short, bounded buffer of candidate crops from analyzed frames of the same snake. When the live identification is uncertain, let the user confirm the best still and up to two nonredundant views. One usable frame is sufficient. Save the time offset of each frame from the encounter start. |

Rank live candidates by sharpness, visible snake detail and framing. Use different views when available. Species confidence can be low on an informative image, so it should not determine which crops to keep. When the encounter contains multiple snakes or the subject changes, the user should confirm one subject before sharing. Adjacent detections often repeat the same information and can vary with blur and occlusion. The one-to-three frame limit is a project starting point to test on real clips. [Han et al., Seq-NMS](https://arxiv.org/abs/1602.08465) · [Yao et al., Video Object Detection via Object-level Temporal Aggregation](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123590154.pdf)

Offer sharing only when a snake was detected. In video, a `no_majority` verdict can qualify if there is usable detected-snake evidence. Save after the user confirms the share. Keep the original safety guidance attached to the on-device verdict. Reviewers can skip a sighting when its evidence is inconclusive. The API keeps model guesses off review cards to reduce anchoring, and role accuracies still need calibration using real reviewer history. [Van Horn et al.](https://authors.library.caltech.edu/records/5qptj-5zy81) · [Wang et al.](https://arxiv.org/html/2406.15844v1#S2.SS4)

## Upload and retry

1. Write the approved JPEGs to app-private storage. Store the sighting UUID, capture metadata, JPEG paths and SHA-256 digests in the SQLite outbox.
2. Send `POST /sightings` with the same UUID and manifest on every retry. Existing metadata-only payloads use the legacy route behavior.
3. For each entry, send the exact saved JPEG bytes with `PUT /sightings/{id}/evidence/{slot}` and `Content-Type: image/jpeg`. Slots are zero-based. Each JPEG can be up to 1 MiB with a maximum side of 2048 pixels. The SHA-256 is calculated from the uploaded JPEG. The server applies orientation and stores a fresh JPEG containing the pixels.
4. When all slots are confirmed, the server opens the sighting for review. The Android outbox can mark it synced when a response reports `ready: true`. A byte-for-byte retry returns `200`. A mismatched digest or changed manifest returns `409`. Create a fresh sighting UUID if the user replaces the images before posting.
5. `GET /sightings/review-feed` returns authenticated evidence URLs. Reviewers use the existing vote endpoint with the sighting UUID. One ballot covers all the stills.

Queue each vote with a client-generated UUID, sighting UUID, rank, label, and the previous active vote UUID for an edit. Retry `POST /sightings/{id}/votes` with the same payload. If an edit conflicts, fetch `GET /sightings/{id}/my-vote` to reconcile against the authenticated reviewer's current vote, even after the sighting has left the public feed. Let the reviewer confirm any replacement after reconciling.

Example live manifest with one frame. For a photo use `"source": "photo"` and omit `offset_ms` from its single evidence entry. The optional model fields describe the primary still in slot 0, while `reason` describes why the app offered sharing.

```json
{
  "id": "3dee0ac8-8a95-4fa3-9779-1ae4950c7613",
  "source": "live_video",
  "trigger": "unknown",
  "reason": "no_majority",
  "snake_found": true,
  "captured_at": "2026-11-16T08:30:00Z",
  "coarse_location": "IN-TN",
  "training_consent": true,
  "model_version": "m4",
  "model_confidence": 0.41,
  "top_predictions": [{"label": "Naja naja", "confidence": 0.41}],
  "genus_prediction": {"label": "Naja", "confidence": 0.63},
  "family_prediction": {"label": "Elapidae", "confidence": 0.72},
  "evidence": [{"offset_ms": 640, "sha256": "<SHA-256 of the saved JPEG>"}]
}
```

## Training review and export

Record community sharing and training permission as separate choices. Send `training_consent: true` only after the reporter explicitly opts in to model improvement. Missing or false consent still allows community review.

1. Provision a curator account with `add-user expert --training-curator`. The grant is separate from the vote role.
2. Call `GET /training/review-feed` with the curator token. It lists consented sightings with a resolved consensus and shows the evidence and consensus revision.
3. After checking the evidence and label, call `POST /sightings/{id}/approve-for-training` with `{"revision": 3}`, replacing 3 with the revision reviewed. An identical approval returns the existing record. If votes change the revision, review it again before approval.
4. Export from the backend machine. The command prompts for the curator token and creates a new archive at the supplied path.

```bash
PYTHONPATH=src .venv/bin/python -m serpentguard.voting_api \
  --db outputs/community-voting.sqlite3 export-training \
  --output outputs/community-training.zip
```

The ZIP contains a versioned JSONL manifest and sanitized JPEGs. Each manifest row ties the images and checksums to a sighting, consent, the approved rank and label, consensus revision, rule version, and curator approval. A newer vote excludes the old revision from future exports. Keep all stills for a sighting together when deduplicating and splitting data. The archive is an input for a separate, manually reviewed experiment.

## Android integration and validation

The companion Flutter app adds a Community tab, local SQLite drafts/outbox, photo/gallery sharing and live-camera still capture. PR #25 connects the real A3 model and M4 video lock; the recorded model checks used an Android emulator. Community submissions require confirmation and exclude simulated predictions when the app falls back to its demo. Foreground sync, Android background sync and cached review are implemented. Physical-device acceptance and reviewer calibration remain pending; see [app setup and handoff](../app/README.md). Compare one image versus up to three on a labeled sample of real encounters. Record whether more views improve reviewer accuracy or merely add upload time. Use independent expert labels and real reviewer votes to measure species and genus accuracy and coverage before calibrating the consensus thresholds.
