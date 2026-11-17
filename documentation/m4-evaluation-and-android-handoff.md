# SerpentGuard Milestone 4 evaluation and Android handoff

Model baseline: [main at acd8fe6](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/tree/acd8fe6). This evidence covers the existing model and the retrieval and regional card work. The detector and classifier recipes were left as merged.

## Phase 3: regional cards and optional search

### Why regional cards use direct lookup

The existing app selects first-aid cards by verdict and stable card ID. For Hindi, Tamil, and Telugu support, keep the card ID and look up the same card in a reviewed language pack. Android can display the text and read it with an installed offline voice for that language. This path works without a retrieval model. [Existing app notebook](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/blob/acd8fe6/notebooks/16_app.ipynb) · [Android TTS voice API](https://developer.android.com/reference/android/speech/tts/Voice)

A reviewed pack provides the same IDs for English, Hindi, Tamil, and Telugu. Each entry carries `text`, an HTTPS `source_url`, and a `review_id` provided by the team. The Python pack validator checks completeness, and the locale resolver reports the language of the text it selected so that Android can pick the matching voice. The team still needs to supply reviewed, authorized translations. [WHO source publication](https://www.who.int/southeastasia/publications/i/item/9789290225300)

Validate a supplied pack against the six IDs used by the current fixed first-aid lookup with `PYTHONPATH=src .venv/bin/python -m serpentguard.localized_cards cards.json`.

### What hybrid retrieval would add

Hybrid retrieval serves free-form typed questions. BM25 matches shared words and dense retrieval can match paraphrases. RRF combines their rankings. The English experiment below tests that hypothesis. Any future regional-language search needs a multilingual encoder or a separately validated translation path, native-speaker queries, and per-language retrieval measurements. [Cormack et al., SIGIR 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf)

Free-text search is a separate experiment. RRF combines complete BM25 and dense passage rankings with `k=60`. The pinned MiniLM-L3 model was quantized to int4 weight blocks of 32 for 18 projection operators and three embedding lookup operators. The candidate's ONNX model is 10.94 MB, its tokenizer is 0.47 MB, and the float32 vectors are 0.54 MB before index metadata. These are component sizes. [Cormack et al., SIGIR 2009](https://cormack.uwaterloo.ca/cormack/cormacksigir09-rrf.pdf) · [ONNX Runtime quantization](https://onnxruntime.ai/docs/how-to/quantization.html) · [Pinned MiniLM-L3](https://huggingface.co/sentence-transformers/paraphrase-MiniLM-L3-v2/tree/4ca70771034acceecb2e72475f72050fcdde4ddc)

| English method | Recall@3 on 59 draft questions | Recall@3 on 20 paraphrases |
|---|---:|---:|
| BM25 | 28/59 (47.5%) | 1/20 (5%) |
| MiniLM-L3 fp32 | 29/59 (49.2%) | 9/20 (45%) |
| MiniLM-L3 int4 | 25/59 (42.4%) | 5/20 (25%) |
| BM25 + MiniLM-L3 fp32 RRF | 30/59 (50.8%) | 3/20 (15%) |
| BM25 + MiniLM-L3 int4 RRF | 30/59 (50.8%) | 3/20 (15%) |

The run used 351 passages and cards. One extracted WHO passage (`who-067-03`) failed the historical hash, so the run excluded it and its question `V15`. Do not compare this subset directly with the historical, full-corpus int8 MiniLM-L6 baseline. Corpus embedding cosine between fp32 and int4 was 0.958 at the median and 0.917 at the minimum. The int4 hybrid scored two more top-three hits overall than BM25, but dense fp32 alone did better on paraphrases. Keep RRF experimental. Typed search in Hindi, Tamil, or Telugu needs its own native-speaker query set and per-language evaluation.

### Reproduce the English experiment

Use the pinned upstream L3 ONNX graph and tokenizer, the team WHO PDF, and Poppler `pdftotext`. The model revision is `4ca70771034acceecb2e72475f72050fcdde4ddc`. The benchmark defaults to strict passage hash validation. Add `--exclude-unmatched-passages` to reproduce this subset.

```bash
.venv/bin/python -m pip install -r requirements-retrieval.txt
PYTHONPATH=src .venv/bin/python scripts/quantize_retriever.py model.onnx model-int4.onnx
PYTHONPATH=src .venv/bin/python scripts/evaluate_hybrid_retrieval.py \
  --who-pdf who.pdf --int4-model model-int4.onnx --fp32-model model.onnx \
  --tokenizer tokenizer.json --max-length 128 --exclude-unmatched-passages
```

## Phase 4: Android acceptance checks

| Area | Integration and device check |
|---|---|
| Camera and models | Wire the app’s camera analyzer to the matched detector, segmenter, and A3 classifier. Verify crop preprocessing and output parity before timing the live pipeline. [CameraX image analysis](https://developer.android.com/media/camera/camerax/analyze) · [ONNX Runtime Mobile](https://onnxruntime.ai/docs/tutorials/mobile/) |
| Regional cards | Bundle the reviewed pack and select the voice for the displayed text. Check `Voice.isNetworkConnectionRequired()` and installation state. Keep the card visible when an offline voice is unavailable. Test all four languages in airplane mode. [Android TTS voice API](https://developer.android.com/reference/android/speech/tts/Voice) |
| Community voting | Reuse the photo/video sighting upload contract. Queue evidence and votes in SQLite, then test capture, process restart, reconnect, and a conflicting vote edit on a physical phone. [Voting capture contract](voting-capture.md) |
| BLE, hospital lookup, and SOS | Integrate and test in the Android checkout. Validate BLE payload size, permissions, duplicate handling and TTL independently from inference. Use a reviewed facility dataset with source dates, and distinguish straight-line proximity from road routing. Exercise SOS with test destinations. |

The Android source and the selected model weights are external inputs to this checkout. The first device pass depends on both.

### Running the frontend

With Jayesh's Android source, open its Gradle project in Android Studio and run the debug build on an emulator or an attached phone. The project’s Gradle wrapper can also run `assembleDebug`. A debug APK can be installed with `adb install <path-to-debug.apk>`. For local voting tests over USB, `adb reverse tcp:8080 tcp:8080` forwards the phone’s port to a backend reachable at host port 8080. With Windows ADB and a WSL backend, verify the server first from Windows at `http://127.0.0.1:8080`. [Android build commands](https://developer.android.com/build/building-cmdline) · [Local server forwarding](https://developer.android.com/develop/ui/views/layout/webapps/access-local-server)

This checkout can run the voting backend in WSL and the Python inference prototype once the listed model artifacts are available. It contains no Android project or APK.

## Phase 5: validation evidence and open items

| Evidence | Status |
|---|---|
| Human and non-snake rejection | M4 false triggers were 170/600 (28.3%) versus 574/600 (95.7%) for M3. End-to-end WHO recall through the M4 front end remains unmeasured. |
| Larger training set | A3 trained on 49,782 eligible photos for 30 epochs. Validation top-1 was 66.4%. The proposed 85,067-photo experiment has no result. |
| Sub-Center ArcFace | The K=1 versus K=3 comparison awaits the K=3 run. |
| Hybrid retrieval | The 59-question English comparison above is reproducible. Regional search and the whole-app size target remain unverified. |
| Expert audit and demo | The 200-photo expert audit, outdoor app recording, and human-rejection phone recording remain pending. Use independent labels and retain per-sighting identifiers so photos from one encounter stay grouped. |

### Prepare the expert audit

The team validation metadata and images are needed to create the pack. The command uses one image per observation, stratifies by tier and medical-importance category, and samples reproducibly at seed 42. Send only the `review/` folder to an authorized reviewer. It contains numbered JPEGs and a blank form. Retain `private/answer-key.csv` to compare decisions with the existing labels after the reviewer has submitted the form. For `decision`, use `species`, `genus`, `no_snake`, or `unsure`. Record a taxon label when the reviewer can identify one.

```bash
mkdir -p outputs
.venv/bin/python scripts/prepare_expert_audit.py \
  --metadata /path/to/metadata.csv --dataset-root /path/to/dataset \
  --output outputs/m4-expert-audit
```

[Merged goal results](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/blob/acd8fe6/results/m4/summary/goals.csv) · [Merged training comparison](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/blob/acd8fe6/results/m4/summary/full_runs.csv) · [Weight inventory](https://github.com/Puneet-Bajaj-IITM/Group-5-DS-and-AI-Lab-Project/blob/acd8fe6/results/m4/artifacts.md)

## Repository changes

| File | Change |
|---|---|
| [`src/serpentguard/localized_cards.py`](../src/serpentguard/localized_cards.py) | Validates the four-language card pack and resolves a card with the language Android should speak. |
| [`src/serpentguard/retrieval.py`](../src/serpentguard/retrieval.py) | Implements RRF over ordered passage IDs with deterministic tie handling. |
| [`scripts/quantize_retriever.py`](../scripts/quantize_retriever.py) | Produces the experimental int4 ONNX model and verifies the quantized operators and CPU load. |
| [`scripts/evaluate_hybrid_retrieval.py`](../scripts/evaluate_hybrid_retrieval.py) | Checks corpus hashes, scores the three retrieval approaches, and reports accuracy by query style and component sizes. |
| [`scripts/prepare_expert_audit.py`](../scripts/prepare_expert_audit.py) | Prepares a reproducible, blinded sample and keeps the reference labels in a separate folder. |
| [`requirements-retrieval.txt`](../requirements-retrieval.txt) | Pins the dependencies for reproducing the search experiment. |
| [`tests/test_retrieval.py`](../tests/test_retrieval.py) | Tests rank fusion, card-pack validation, and locale selection. |
| [`tests/test_expert_audit.py`](../tests/test_expert_audit.py) | Tests unique validation observations, deterministic sampling, and the blinded package. |
| [`documentation/m4-evaluation-and-android-handoff.md`](m4-evaluation-and-android-handoff.md) | Records the research rationale, measured results, Android acceptance checks, and pending evidence. |

Run the retrieval, locale, and existing voting tests from the repository root with `PYTHONPATH=src .venv/bin/python -B -m unittest discover -s tests -v`.
