# G5: the video lock, fixed and retuned

G5: at most 1 change of the shown species after the first answer, on every real test clip.

**Result.** On 22 new real test clips that nothing was tuned on, scored once, the new lock meets G5 at
0.1, 0.25 and 0.4 s a frame (most changes 1, no clip over 1). At 0.8 s a frame (a slow live camera)
it fails on 1 of 22 clips (`indotyphlops_braminus_almeria.mp4`, 2 changes). As a second look, the 6
old test clips meet G5 at all four speeds; `naja_naja_ditmars_1915` now has 0 changes. Two things are
not fixed. Every no-snake clip, old and new, still locks onto a species and shows "Dangerous". The
"dangerous" safety latch also ends 12 of 13 safe-species new clips on "Dangerous".

Everything here ran on the laptop CPU with no paid calls. Code changes are uncommitted in
`m5-g5` (branch m5-g5):

- `src/serpentguard/video.py`: the lock (`lock_settings`, `pooled_lock`, `lock_verdicts`, new constants)
- `app/lib/inference/video_lock.dart`: the Dart port (`LockSettings`, `VideoLock`, `LockStep.event`)
- `tests/test_video_lock.py`
- `app/test/video_lock_test.dart`, `app/test/video_lock_parity_test.dart`, `app/test/camera_frame_test.dart`
- `app/test/golden/video_lock_parity.json`: 2.3 MB, generated

## 1. The review's problems, checked one by one

The M4 lock (git HEAD `video.py`) was rerun on notebook 24's saved A3 outputs. The output is
`bug_check_m4_lock.csv`, made by `code/verify_old_bugs.py`.

### Problem 1: the shown species did not hold

**Verified.** Once locked, the app showed DR0's verdict on the current pool, so the species on screen
was the pool's top species at that frame, not the locked one. Notebook 24 scored G5 on the locked
class. Changes after the first answer, scored vs displayed:

| Test clip | Changes, as notebook 24 scored (locked class) | Changes, as the app displayed |
|---|---|---|
| naja_naja_ditmars_1915 | 2 | 21 |
| gonyosoma_oxycephalum_mulu | 0 | 0 |
| ahaetulla_borealis_kerala | 0 | 25 |
| ptyas_mucosa_swimming | 0 | 9 |
| naja_cobra_2017 | 0 | 2 |
| naja_jaipur_charmer | 0 | 3 |

So the M4 G5 numbers described a screen the app never showed.

**Fix.** `lock_verdicts` (Python) and `VideoLock.add` (Dart) now always show the locked species.
- **Verdict while locked.** The verdict is DR0's on the pool at the frame the lock was set or switched.
- **Safety latch.** If DR0 on the current pool says "dangerous" while locked, the shown verdict turns
  "dangerous" and stays so until the lock changes.
- **While checking.** Only "dangerous" can show, with no species named (empty top 3).

### Problem 2: the window counted frames

**Verified.** It was 5 snake frames, tuned at 0.1 s a frame. **Fix:**
- The window is now `window_seconds`.
- Every other rule was already in seconds or is now: the switch time, and the new hold and window.
- The pool also never holds fewer than the last `min_frames` snake frames, so a slow phone still pools.
- The tuning and every result below were run at 0.1, 0.25, 0.4 and 0.8 s a frame, using `video.schedule` on every frame of each clip.

### Problem 3: the window never expired

**Verified.** Old frames stayed in the pool forever; a no-snake frame just repeated the last step.
**Fix:**
- Frames older than the window drop out of the pool.
- After `hold_seconds` (2 s) with no snake frame, the lock and the pool are dropped. The next snake then starts afresh.
- With no gap, panning to another snake switches after the rival has clearly led for `switch_seconds` (see section 5).

### Problem 4: no-snake clips locked

**Verified, and not fixed.** With the M4 lock, all 3 negatives lock and show "dangerous". The cause
is the front end, not the lock: the gate calls a snake on most frames of these clips.

| Clip | Frames the gate called a snake |
|---|---|
| chameleon | 294/301 |
| eels | 284/301 |
| slug | 217/217 |
| glass lizard | 146/146 |
| slowworm | 175/175 |
| monitor | 241/241 |
| eagle | 82/189 |
| river | 84/301 |

So a gate-share rule cannot separate them. It was tuned (`gate_share` 0, 0.5, 0.8) and made no
difference to locking on any dev or negative clip (it only delayed a few first answers), so it ships at 0.

A higher lock confidence was also tried (`results/tuning_lock_conf.csv`):

| Lock confidence | Fewest negative clip × speed pairs locked (of 6) | Dev clips never answered, at 0.4 s |
|---|---|---|
| theta_conf 0.287 | 4 | 15/50 |
| 0.4 | 3 | 22/50 |
| 0.5 | 2 | 27/50 |
| 0.6 | 0 | 27/50 |

Rejecting the negatives costs about half the real answers, so the lock confidence stays theta_conf.
Fixing this needs a better no-snake gate (a non-snake class or a stricter gate), not a lock rule.

### Problem 5: thresholds

**Done.**
- Every run uses the app's `app/assets/model/pipeline.json` (the refitted thresholds from
  `decision_app_front_end.json`): temperatures, DR0 thresholds, and lock confidence = theta_conf 0.2870.
- Prototypes come from the app's `tables.json`.
- The test fixture checks that its `theta_conf` equals the app's.

### Problem 6: video frames are resized differently

**Verified, not fixed (the file belongs to another team member).** The app's video path resizes
differently from notebook 24:
- `VideoAnalyzer` asks the Android side for frames at `maxSide: 640`.
- `VideoFrames.kt` scales them bilinear (line 128), so `OnnxSnakeClassifier._phoneSize`'s INTER_AREA
  `resizeArea` is then a no-op.
- The live camera path does use `resizeArea`.

**Measured effect.** The test clips were rerun with `cv2.INTER_LINEAR` frame scaling
(`outputs_linear/`, `results/bilinear_frames/`): on 28 test clips (22 new, 6 old):
- The gate decision agrees on 10,153 of 10,212 frames.
- The top-1 species agrees on 94.6% of frames where both pass (8,577 of 9,069).
- G5 on the new clips is unchanged: met at 0.1, 0.25 and 0.4 s, and 1 clip over 1 change at 0.8 s.

So the difference moves single frames but not the G5 result.

**Simple fix for Manas.** In `video_analyzer.dart`, pass a large `maxSide` (for example 4096, or the
native size) so Dart's `resizeArea` does the 640 px step, as in live and in notebook 24.

### Problem 7: no tests

**Done for VideoLock and cameraFrameToRgb.** VideoAnalyzer is not tested: it needs a mocked platform
channel and a model, and its file is being changed by Manas.

## 2. The lock now

Python: `video.pooled_lock(items, times, conf, **settings)` and `video.lock_verdicts(steps, ...)`.
Dart: `VideoLock(rules:, settings: LockSettings(...))`. The setting names are the same on both sides.

| Setting | Value | What it does |
|---|---|---|
| `window_seconds` | 0.5 | Pool the snake frames of the last 0.5 s. A lock also needs the sighting to have lasted 0.5 s |
| `min_frames` | 1 | Never pool fewer than the last N snake frames; a lock needs N |
| `margin` | 0.05 | The pooled top species must lead the runner-up by this to lock, and a rival must lead by more than this to switch |
| `switch_seconds` | 3.0 | A rival must lead on every snake frame for this long to replace the lock |
| `gate_share` | 0.0 | Share of the spanned frames the gate must pass to lock (no effect found) |
| `hold_seconds` | 2.0 | No snake for this long drops the lock (fixed, not tuned) |
| lock confidence | theta_conf = 0.2870 | The pooled top species must hold this to lock or to switch |

### API change for the app screens (Manas)

The public API is kept: `VideoLock(rules: ...)`, `add(parts, seconds)` returning `LockStep`, `reset()`,
`lockedClass`. Changes:

1. `VideoLock` takes an optional `settings: LockSettings` (defaults above). The old `window`, `margin`
   and `switchSeconds` constructor arguments are gone.
2. `LockStep` adds `event` (`'lock'` or `'switch'` on the frame the lock is set or changed, else null)
   and a `repeated` getter.
3. `LockStep.shown`, once locked, has the locked class as `top3[0]`, with its pooled probability, then
   the next two. While checking, `shown` is either null or a "dangerous" answer with **an empty top 3**:
   the screens should show "Dangerous" with no species name. `video_check_screen.dart` already falls
   back to the reason when `topSpecies` is null.
4. After 2 s with no snake, `add(null, t)` returns a step with `pooled == null`.
   `OnnxSnakeClassifier.identifyFrame` then shows "no snake found", as it does before the first snake.

`classifier.dart` needed no change.

**Notebook 24** calls `video.pooled_lock(..., window=5)`, which now raises `TypeError`, on purpose.
Its saved outputs are the M4 record. Rerunning it needs its lock calls moved to the new settings;
`code/lock_eval.py` does this run for M5. `video.POOL_FRAMES = 10` is kept because notebook 25 reads it.

## 3. Retune (dev clips only)

- **Clips.** 50 dev clips: notebook 24's 46 synthetic clips (same draw, same seeds) and its 4 real dev
  clips. Tuning eligibility used the 3 old negatives.
- **How each clip was run.** Every frame (at most 15 fps, 30 s) went through the app's path: the ONNX
  reference pipeline (`m4-final/for-jayesh`, fp32), with the app's refitted thresholds applied at
  scoring time.
- **Speeds simulated.** 0.1, 0.4 and 0.8 s a frame, with `video.schedule`.
- **Grid.** 360 settings: window 0.5, 1, 1.5, 2 or 3 s; margin 0.05, 0.1, 0.2 or 0.3; switch 1, 2 or
  3 s; gate share 0 or 0.5; min frames 1, 2 or 3. Full table: `results/tuning.csv`.
- **Rule**, applied in order:
  1. No dev clip over 1 change at any speed.
  2. Median first answer ≤ 3 s at every speed (never answered counts as never).
  3. Fewest negative clips locked.
  4. Most right final answers.
  5. Fastest.
  6. Then the shortest window, smallest margin, shortest switch, lowest gate share.

**The rule changed once, after seeing dev numbers (dev only, before any test clip was scored).**
- The first rule ranked fewest negatives locked first. It picked window 3 s and margin 0.3, which never
  answers on 32 of 50 dev clips.
- I moved negatives below G5 and speed. That pass is in `results/superseded/`.
- The pool rule also changed between passes. "At least `min_frames`" was added after I saw that a
  0.5 s window holds no frames to pool at 0.8 s a frame.

Excerpt from `results/tuning.csv` (rank 1 is chosen). Columns:
- over 1 = dev clip × speed pairs with more than 1 change.
- neg locked = old negative clip × speed pairs that lock, of 9.
- median first = median first answer in seconds, at each speed.

| rank | window s | margin | switch s | min frames | over 1 | neg locked | median first @0.1 | @0.4 | @0.8 | never @0.4 | final right @0.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **0.5** | **0.05** | **3** | **1** | 0 | 9 | 1.3 | 2.2 | 2.8 | 15/50 | 14/50 |
| 9 | 3 | 0.3 | 1 | 1 | 0 | 1 | never | never | never | 32/50 | 11/50 |
| 149 | 0.5 | 0.3 | 1 | 1 | 0 | 6 | 3.05 | 5.8 | 5.6 | 21/50 | 11/50 |
| 189 | 1.5 | 0.05 | 3 | 1 | 0 | 7 | 5.0 | 3.8 | 5.2 | 20/50 | 12/50 |
| 221 | 0.5 | 0.05 | 3 | 2 | 0 | 9 | 1.3 | 2.2 | 3.2 | 15/50 | 14/50 |
| 237 | 1 | 0.05 | 3 | 1 | 0 | 9 | 4.0 | 2.2 | 5.2 | 17/50 | 12/50 |

**Chosen** (`results/chosen_settings.json`): window 0.5 s, margin 0.05, switch 3 s, gate share 0,
min frames 1, hold 2 s.

On dev at 0.1, 0.25, 0.4 and 0.8 s a frame:
- Most changes: 1, with no clip over 1.
- Never answered: 14, 11, 15 and 14 of 50. Many synthetic clips never reach theta_conf.
- Right at the end: 14 to 15 of 50.

## 4. New test clips

27 new clips, all in `new_clips/`. They are untrusted downloads, not in the repo. Full credits are in
`new_clips.csv` and `new_negatives.csv`; every candidate looked at is in `search/candidates.csv`
(963 rows). Contact sheets are in `new_clips_sheets/`.

The set was fixed before any of it was scored and was scored once. I looked at 3 contact sheets myself
(Daboia, a Naja kaouthia clip and the slowworm); the rest are as the search recorded them.

**Licence caveat.**
- 2 snake clips and 2 negatives are Wikimedia Commons, with a stated licence and author.
- 20 snake clips and 3 negatives are YouTube uploads whose licence field reads "Creative Commons
  Attribution licence (reuse allowed)", i.e. CC BY with no version.
- The uploader is not always the author: `daboia_russelii_closeup` is from a channel named
  "Royalty Free Footage". Please review that subset before quoting it.
- Downloads are 360p.

| Clip | Species | How the species is known | Licence | Credit | Source |
|---|---|---|---|---|---|
| `ophiophagus_hannah_singapore_hunting` | *Ophiophagus hannah* | stated | CC BY-SA 4.0 | Jason Quinn | [Commons](https://commons.wikimedia.org/wiki/File:King_Cobra_(Ophiophagus_hannah)_hunting_in_the_afternoon_in_Singapore's_Sungei_Buloh_Wetland_Reserve_(May_27,_2023).ogg), from 12 s |
| `ptyas_mucosa_india_drinking` | *Ptyas mucosa* | inferred | CC BY 2.0 | Ganesh Subramaniam | [Commons](https://commons.wikimedia.org/wiki/File:Rat_snake_drinking_water.webm) |
| `bungarus_fasciatus_taiwan_night` | *Bungarus fasciatus* | stated | CC BY | Ryan Chan | [YouTube lV1z-1y2e6s](https://www.youtube.com/watch?v=lV1z-1y2e6s) |
| `bungarus_fasciatus_kerala_pond` | *Bungarus fasciatus* | stated | CC BY | The Island Stories | [YouTube QoGcWIgpsww](https://www.youtube.com/watch?v=QoGcWIgpsww) |
| `ahaetulla_prasina_handled` | *Ahaetulla prasina* | stated | CC BY | Алексей Солдатов | [YouTube IUanO6AcvS8](https://www.youtube.com/watch?v=IUanO6AcvS8) |
| `elaphe_taeniura_taiwan_tree` | *Elaphe taeniura* | stated | CC BY | 王士豪 | [YouTube _qpCyYlcs00](https://www.youtube.com/watch?v=_qpCyYlcs00) |
| `chrysopelea_ornata_thailand_gecko` | *Chrysopelea ornata* | stated | CC BY | Gonzalo Mucientes | [YouTube JHdFB4kzZB0](https://www.youtube.com/watch?v=JHdFB4kzZB0), from 8 s |
| `cerberus_rynchops_eating_fish` | *Cerberus rynchops* | stated | CC BY | praseodymi | [YouTube iuuZxF5pBpg](https://www.youtube.com/watch?v=iuuZxF5pBpg) |
| `oreocryptophis_porphyraceus_box` | *Oreocryptophis porphyraceus* | stated | CC BY | Илья Дроздов | [YouTube rHaBYRTxEoo](https://www.youtube.com/watch?v=rHaBYRTxEoo) |
| `indotyphlops_braminus_home` | *Indotyphlops braminus* | stated | CC BY | MeBlog | [YouTube bZ6ODBE53DI](https://www.youtube.com/watch?v=bZ6ODBE53DI), from 5 s |
| `indotyphlops_braminus_almeria` | *Indotyphlops braminus* | stated | CC BY | SERBAL | [YouTube sagWY6JJ3R0](https://www.youtube.com/watch?v=sagWY6JJ3R0) |
| `laticauda_colubrina_bali` | *Laticauda colubrina* | stated | CC BY | MrChopsta | [YouTube qHJYZ3MakxQ](https://www.youtube.com/watch?v=qHJYZ3MakxQ) |
| `daboia_russelii_closeup` | *Daboia russelii* | stated | CC BY | Royalty Free Footage (uploader) | [YouTube zwxn2JxntUw](https://www.youtube.com/watch?v=zwxn2JxntUw) |
| `craspedocephalus_malabaricus_branch` | *Craspedocephalus malabaricus* | stated (as *Trimeresurus malabaricus*) | CC BY | Thejasvi M | [YouTube nxZI3XC1XIo](https://www.youtube.com/watch?v=nxZI3XC1XIo) |
| `dendrelaphis_tristis_rock` | *Dendrelaphis tristis* | inferred | CC BY | Bhanu and Ashwin's Channel | [YouTube SACCN2dinV4](https://www.youtube.com/watch?v=SACCN2dinV4) |
| `coelognathus_helena_floor` | *Coelognathus helena* | stated | CC BY | PersonaLife Wildlife | [YouTube Yx2lzpY9gcY](https://www.youtube.com/watch?v=Yx2lzpY9gcY) |
| `python_molurus_galibore` | *Python molurus* | inferred | CC BY | Garima Bhatia | [YouTube 8uw4wpoA5J8](https://www.youtube.com/watch?v=8uw4wpoA5J8) |
| `naja_kaouthia_baby_container` | *Naja kaouthia* | stated | CC BY | denysza | [YouTube nVKT4Z3hoFA](https://www.youtube.com/watch?v=nVKT4Z3hoFA) |
| `naja_kaouthia_captive_feeding` | *Naja kaouthia* | stated | CC BY | Andy Maci | [YouTube 6_lBJvP764w](https://www.youtube.com/watch?v=6_lBJvP764w) |
| `naja_kaouthia_bangladesh_hood` | *Naja kaouthia* | stated | CC BY | GHORIAL BANGLA | [YouTube X0nZn8sFPFo](https://www.youtube.com/watch?v=X0nZn8sFPFo) |
| `fowlea_piscator_rock` | *Fowlea piscator* | stated | CC BY | Bird Watching - India | [YouTube VEEsXjK9Ni8](https://www.youtube.com/watch?v=VEEsXjK9Ni8) |
| `lycodon_aulicus_baby` | *Lycodon aulicus* | stated | CC BY | SDM Astronomy & Adventure | [YouTube aGA4ANO5d-w](https://www.youtube.com/watch?v=aGA4ANO5d-w) |

**Notes on individual clips.**
- 3 clips are `inferred` (the page does not name the binomial). They count for G5 but not for "final
  species right".
- `naja_kaouthia_bangladesh_hood`: its last frames show a coiled dark snake that may be another animal.

**Negatives (no snake).**

| Clip | What it shows | Licence | Credit | Source |
|---|---|---|---|---|
| `negative_eels_hawaii` | two eels | CC0 | John Veit | [Commons](https://commons.wikimedia.org/wiki/File:2_eels_-_Anguilliformes_in_a_tidepool_near_Kona,_Hawaii_get_into_a_fight.ogv) |
| `negative_slug_austria` | a slug | CC BY-SA 4.0 | Martinvie | [Commons](https://commons.wikimedia.org/wiki/File:Limax.webm) |
| `negative_glass_lizard` | a legless lizard | CC BY | Dieuw's wildlife | [YouTube vnTVtuqM8SQ](https://www.youtube.com/watch?v=vnTVtuqM8SQ) |
| `negative_slowworm` | a slowworm | CC BY | Gary Crowder | [YouTube EEBWWyjrZr4](https://www.youtube.com/watch?v=EEBWWyjrZr4) |
| `negative_bengal_monitor` | a monitor lizard | CC BY | Gonzalo Mucientes | [YouTube 5yoXZSawtXo](https://www.youtube.com/watch?v=5yoXZSawtXo) |

**Coverage.**
- Commons has very few videos of our species. The search ran 206 binomial and common-name searches,
  and Internet Archive gave nothing usable.
- No *Naja naja*, *Echis* or *Bungarus caeruleus* clip passed the checks. Hydrophis was not found.

## 5. G5 results

Measures:
- Changes = changes of the shown species after the first answer.
- First answer = when a species is first named.
- Final right = the species on screen at the end is the clip's species. Counted only on `stated` and
  `category` clips; "Checking..." or "no snake" at the end counts as wrong.

Files: `results/g5_summary.csv` and the per-clip `results/*_scores.csv`.

### New real test clips: 22 clips, scored once

| s per frame | Most changes | Clips over 1 | G5 | Median first answer | Never answered | Final species right |
|---|---|---|---|---|---|---|
| 0.1 | 1 | 0 | **met** | 0.6 s | 1 | 12/19 |
| 0.25 | 1 | 0 | **met** | 0.75 s | 0 | 12/19 |
| 0.4 | 1 | 0 | **met** | 1.2 s | 0 | 12/19 |
| 0.8 | 2 | 1 | **not met** | 1.6 s | 0 | 11/19 |

**Why the 0.8 s clip fails.** At 0.8 s, `indotyphlops_braminus_almeria` locks on *Indotyphlops
braminus*, switches to *Dendrelaphis tristis* at 16.8 s, then switches back at 27.2 s. With min frames
1, a slow phone pools a single frame, so 5 frames in a row favouring the wrong species pass the 3 s
switch rule. At 0.4 s the same clip has 0 changes.

The clips with 1 change are those where the first lock was wrong and the lock corrected itself. For
example, `indotyphlops_braminus_home` goes *Echis carinatus* → *Indotyphlops braminus*, and
`oreocryptophis_porphyraceus_box` goes *Naja naja* → *O. porphyraceus*.

**Verdicts on the new clips** (0.4 s a frame):
- Every dangerous-species clip showed "dangerous" at some point: 9/9.
- Safe species ending on "Dangerous": 12/13. Without the latch, DR0 on the final pool would say
  "dangerous" for 5/13.
- The latch therefore keeps a lot of false alarms on screen. That is a safety-vs-false-alarm choice
  for the team. The non-sticky option shows "dangerous" only while the current pool says so: the
  species still holds, the verdict can flip.
- The shown verdict changes at most 2 times on a clip (4 at 0.8 s), for example "dangerous" while
  checking, then "not listed" at the lock, then the latch.

### Old 6 test clips: second look

These clips already judged the old lock, so this is not a fresh test.

| s per frame | Most changes | G5 | Median first answer | Never answered | Final species right |
|---|---|---|---|---|---|
| 0.1 | 0 | met | 0.6 s | 1 | 1/4 |
| 0.25 | 0 | met | 0.75 s | 0 | 1/4 |
| 0.4 | 0 | met | 1.2 s | 1 | 1/4 |
| 0.8 | 1 | met | 1.6 s | 1 | 1/4 |

- `naja_naja_ditmars_1915`: 0 changes, but it locks on *Fowlea piscator* (still "dangerous" via the latch).
- `ahaetulla_borealis_kerala` mostly never answers. `ptyas_mucosa_swimming` ends with no snake on screen.
- The 2 `inferred` Naja clips are right.

### No-snake clips: what the screen shows

Every one locks onto a species and shows "Dangerous" at every speed. The species shown at 0.4 s:

| Clip | Frames the gate called a snake (0.4 s) | Species shown |
|---|---|---|
| chameleon (old) | 73/76 | *Echis carinatus* |
| eagle (old) | 19/48 | *Hydrophis platurus* |
| river (old) | 21/76 | *Python bivittatus* (2 locked frames) |
| eels (new) | 73/76 | *Macrovipera lebetinus* |
| slug (new) | 55/55 | *Naja naja* |
| glass lizard (new) | 38/38 | *Naja naja* |
| slowworm (new) | 45/45 | *Naja naja* |
| monitor (new) | 61/61 | *Naja naja* |

### Panning test (dev clips only)

Two dev clips of different species are put back to back, each one the lock gets right on its own.
File: `results/panning.csv`.

- **No gap (4 pairs × 4 speeds).** The answer switched to the second species in **15 of 16**, 3.0 to
  10 s into the second clip.
- **The miss.** *Daboia* → *Naja kaouthia* at 0.1 s. There the first clip had locked on the wrong
  species (*Boiga multomaculata*), and the second clip never made a rival lead it for 3 s.
- **3 s gap with no snake (2 pairs × 4 speeds).** The lock is dropped after 2 s, and the second species
  shows in **8 of 8**, 0.5 to 0.8 s into the second clip.

## 6. Parity and tests

- **Python tests** (`tests/test_video_lock.py`): 10 passed. They cover the window in seconds at three
  speeds, a short rival, one switch, hold expiry, gate share, no snake, unknown settings refused, and
  the slow phone still pooling.
- **Dart unit tests** (`app/test/video_lock_test.dart`, 12 tests): window timing at 10, 2.5 and 1.25
  fps, min frames at 0.8 s a frame, the species held through a short rival, one switch, hold expiry,
  a short gap keeping the lock, gate share, never locking when unsure, "dangerous" while checking
  with no species, the dangerous latch, and reset.
- **cameraFrameToRgb** (`app/test/camera_frame_test.dart`, 5 tests): BGRA with row stride, YUV 4:2:0
  grey and red, NV21 order, sensor rotation and the front-camera mirror, and an unknown format.
- **Parity** (`app/test/video_lock_parity_test.dart`): 7 real dev-clip sequences from
  `code/make_parity.py`, 627 steps in all. They cover lock, switch, hold reset, "dangerous" while
  checking, the latch, a not-listed lock, and gate share with a 1.5 s window. Every step matches
  Python on checking, locked class, event, shown verdict and shown species. Values are rounded to 7
  decimals before Python computes the expected steps.
- **Full suite.** `flutter test` in `m5-g5/app`: **69 passed, 2 skipped** (baseline was 44 passed, 2 skipped).
- **Mutation check.** Each of these was killed by the tests:
  - showing the pooled top species instead of the locked one: 7 failures;
  - removing the hold reset;
  - removing the dangerous latch.
- **The ONNX outputs agree with notebook 24's PyTorch cache** on a sample of 11 clips: gate decision
  100%, top-1 100% (`code/check_outputs.py`). A control with the frames misaligned agrees only 56%.
  The control is high because the cobra film is mostly one species.

## 7. What I could not check

- **On a phone or emulator.** The Dart lock is tested only in unit and parity tests, on real model
  outputs. Real phone latency was simulated with `video.schedule`, not measured.
- **The YouTube licences.** I did not verify them beyond the page field, and did not review all
  contact sheets by eye. Several species IDs rest on the uploader's title.
- **The new test set's balance.** It is small and lopsided: 3 *Naja kaouthia*, 2 *Bungarus
  fasciatus*, 2 *Indotyphlops*. G5 on 22 clips at the passing speeds is consistent with "at most 1
  change", but one clip failing at 0.8 s shows the margin is thin.
- **VideoAnalyzer tests** were not written (see problem 7).
- **Streaming video** in the app was dropped from scope (owned by Manas).
- **Notebook 24** was not rerun under the new lock.

## Files

| Path | What it is |
|---|---|
| `code/clip_outputs.py` | Every frame of every clip through the ONNX reference pipeline. `--models` is the models folder, so it is model-agnostic. Set `RESIZE=linear` for the bilinear check |
| `code/lock_eval.py` | Tuning, scoring and panning on the saved outputs. Takes `--pipeline` and `--tables` (the app's) |
| `code/make_parity.py` | The Dart parity fixture |
| `code/summarize.py` | Builds `results/g5_summary.csv` |
| `code/verify_old_bugs.py` | The M4 lock checks behind `bug_check_m4_lock.csv` |
| `code/check_outputs.py` | The ONNX vs PyTorch sample check |
| `outputs/` | Per-frame raw outputs, 86 clips, with `manifest_<model tag>.csv` |
| `results/` | `tuning.csv`, `chosen_settings.json`, `*_scores.csv`, `negatives.csv`, `panning.csv`, `g5_summary.csv` |
| `logs/` | Run logs |
| `new_clips/`, `raw/`, `raw_yt/` | Downloads (about 870 MB); delete `raw*` if space is needed |

**Rerunning with another model.** Pass its models folder to `clip_outputs.py`, then `lock_eval.py`
with its `pipeline.json` and `tables.json`. Outputs are keyed on the model files' hashes.

Rerun order:
1. `clip_outputs.py --models DIR --reference reference_pipeline.py --out DIR` (shards: `--shard i --shards n`), then `--manifest`.
2. `lock_eval.py tune`, then copy `results/tuning_chosen.json` to `results/chosen_settings.json`.
3. `lock_eval.py score --sets ...` and `lock_eval.py pan`.
