# G5: the video lock, fixed and retuned

**The goal.** G5 asks the video answer to change its shown species at most once after the first answer, on every
real test clip. In M4 one clip changed twice. The TA's review also found that no-snake clips locked onto a species.

**What we found first.** The M4 numbers described a screen the app never showed. Notebook 24 scored the locked
class, but the app displayed the current pool's top species, which moved far more (problem 1 below). We fixed the
lock in Python and Dart together, retuned it on dev clips only, and scored it once on 22 new real clips.

**The answer.** With A3 and A3's thresholds (how the clips were first scored), the new lock meets G5 at 0.1, 0.25
and 0.4 s a frame. At 0.8 s a frame one clip of 22 changes twice. With the frozen M5 classifier, the locked test
shows at most 1 change at all four speeds (`../test/g5/`; report, section 4.4). Two things are not fixed:

- every no-snake clip still locks onto a species and shows "Dangerous";
- the "dangerous" latch ends 12 of 13 safe-species clips on "Dangerous".

Everything here ran on the laptop CPU.

## 1. The review's problems, checked one by one

We reran the M4 lock (`code/old/video_m4.py`, a copy of M4's `video.py`) on notebook 24's saved A3 outputs.
`code/verify_old_bugs.py` writes the result to `bug_check_m4_lock.csv`.

**Problem 1: the shown species did not hold. Confirmed, fixed.** Once locked, the app showed the decision rules'
answer on the current pool. So the species on screen was the pool's top species at that frame, not the locked one.

| Test clip | Changes as notebook 24 scored them (locked class) | Changes as the app showed them |
|---|---|---|
| naja_naja_ditmars_1915 | 2 | 21 |
| gonyosoma_oxycephalum_mulu | 0 | 0 |
| ahaetulla_borealis_kerala | 0 | 25 |
| ptyas_mucosa_swimming | 0 | 9 |
| naja_cobra_2017 | 0 | 2 |
| naja_jaipur_charmer | 0 | 3 |

Now `lock_verdicts` (Python) and `VideoLock.add` (Dart) always show the locked species:

- **While locked**, the verdict is the decision rules' answer on the pool at the frame the lock was set or switched.
- **The safety latch.** If the rules call the current pool "dangerous" while locked, the shown verdict turns
  "dangerous" and stays so until the lock changes.
- **While checking**, only "dangerous" can show, with no species named.

**Problem 2: the window counted frames. Confirmed, fixed.** It was 5 snake frames, tuned at 0.1 s a frame. The
window is now in seconds (`window_seconds`), as are the switch time and the new hold. The pool also never holds
fewer than the last `min_frames` snake frames, so a slow phone still pools. Every tuning run and result below was
simulated at 0.1, 0.25, 0.4 and 0.8 s a frame with `video.schedule`, on every frame of each clip.

**Problem 3: the window never expired. Confirmed, fixed.** Old frames stayed in the pool for ever, and a no-snake
frame only repeated the last step. Now frames older than the window drop out. After `hold_seconds` (2 s) with no
snake frame, the lock and the pool are dropped, so the next snake starts afresh. With no gap, panning to another
snake switches once the rival has clearly led for `switch_seconds` (section 5).

**Problem 4: no-snake clips locked. Confirmed, not fixed.** With the M4 lock all 3 old negatives lock and show
"dangerous". The cause is the front end, not the lock: the gate calls a snake on most frames of these clips.

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

So a gate-share rule cannot separate them. We tuned one anyway (`gate_share` 0, 0.5 and 0.8). It changed locking on
no dev or negative clip and only delayed a few first answers, so it ships at 0. A higher lock confidence
(`results/tuning_lock_conf.csv`) does reject negatives, but at a cost:

| Lock confidence | Fewest negative clip-speed pairs locked (of 6) | Dev clips never answered, at 0.4 s |
|---|---|---|
| theta_conf 0.287 | 4 | 15/50 |
| 0.4 | 3 | 22/50 |
| 0.5 | 2 | 27/50 |
| 0.6 | 0 | 27/50 |

Rejecting the negatives this way costs about half the real answers, so the lock confidence stays at theta_conf.
The fix belongs in the no-snake gate, not in the lock. The snake check (G4) is that gate, but it is not yet used in
video.

**Problem 5: thresholds. Done.** Every run uses the app's `pipeline.json` with the thresholds refitted to the
app's front end: temperatures, decision thresholds, and lock confidence = theta_conf 0.2870. Prototypes come from
the app's `tables.json`, and a test fixture checks that its theta_conf equals the app's.

**Problem 6: video frames are resized differently. Confirmed, not fixed in this branch.** The app's video path asks
Android for frames at `maxSide: 640`, scaled bilinearly. The live camera path and notebook 24 use area resizing. We
reran the test clips with bilinear scaling (`outputs_linear/`, `results/bilinear_frames/`). On 28 clips (22 new, 6
old) the gate decision agrees on 10,153 of 10,212 frames, and the top-1 species on 94.6% of the frames where both
pass (8,577 of 9,069). G5 on the new clips is unchanged. The difference moves single frames, not the G5 result.
The simple fix is to pass a large `maxSide` in `video_analyzer.dart` so that Dart's area resize does the 640 px
step, as in live and in notebook 24.

**Problem 7: no tests. Done** for `VideoLock` and `cameraFrameToRgb` (section 6). `VideoAnalyzer` is not tested: it
needs a mocked platform channel and a model.

## 2. The lock now

Python: `video.pooled_lock(items, times, conf, **settings)` and `video.lock_verdicts(steps, ...)`. Dart:
`VideoLock(rules:, settings: LockSettings(...))`. The setting names are the same on both sides.

| Setting | Value | What it does |
|---|---|---|
| `window_seconds` | 0.5 | Pool the snake frames of the last 0.5 s; a lock also needs the sighting to have lasted 0.5 s |
| `min_frames` | 1 | Never pool fewer than the last N snake frames; a lock needs N |
| `margin` | 0.05 | The pooled top species must lead the runner-up by this to lock; a rival must lead by more to switch |
| `switch_seconds` | 3.0 | A rival must lead on every snake frame for this long to replace the lock |
| `gate_share` | 0.0 | Share of the spanned frames the gate must pass to lock (no effect found) |
| `hold_seconds` | 2.0 | No snake for this long drops the lock (fixed, not tuned) |
| lock confidence | theta_conf = 0.2870 | The pooled top species must hold this to lock or switch |

For the app screens the public API is kept (`add(parts, seconds)` returning `LockStep`, `reset()`, `lockedClass`),
with four changes:

1. `VideoLock` takes an optional `settings: LockSettings`. The old `window`, `margin` and `switchSeconds`
   arguments are gone.
2. `LockStep` adds `event` (`'lock'` or `'switch'` on the frame the lock is set or changed) and a `repeated`
   getter.
3. Once locked, `LockStep.shown` has the locked class as `top3[0]`. While checking, `shown` is null or a "dangerous"
   answer with an empty top 3, which the screens show as "Dangerous" with no species name.
4. After 2 s with no snake, `add(null, t)` returns a step with `pooled == null`, and the screen shows "no snake
   found".

Notebook 24 calls `video.pooled_lock(..., window=5)`. That now raises `TypeError` on purpose. Its saved outputs are
the M4 record, and `code/lock_eval.py` does the M5 run. `video.POOL_FRAMES = 10` stays because notebook 25 reads it.

## 3. Retune on dev clips only

- **Clips.** 50 dev clips: notebook 24's 46 synthetic clips (the same draw and seeds) and its 4 real dev clips. The
  3 old negatives decided eligibility.
- **How each clip was run.** Every frame (at most 15 fps, 30 s) went through the app's path: the fp32 ONNX reference
  pipeline, with the app's refitted thresholds applied at scoring time.
- **Speeds.** 0.1, 0.4 and 0.8 s a frame, simulated with `video.schedule`.
- **Grid.** 360 settings: window 0.5, 1, 1.5, 2 or 3 s; margin 0.05, 0.1, 0.2 or 0.3; switch 1, 2 or 3 s; gate share
  0 or 0.5; min frames 1, 2 or 3. The full table is `results/tuning.csv`.
- **Selection rule**, in order:
  1. no dev clip over 1 change at any speed;
  2. median first answer of 3 s or less at every speed (never answered counts as never);
  3. fewest negative clips locked;
  4. most right final answers;
  5. fastest;
  6. then the shortest window, smallest margin, shortest switch, lowest gate share.

**The rule changed once, after we saw dev numbers.** This was on dev clips only, before any test clip was scored.
The first rule ranked fewest negatives locked first. It picked a 3 s window with margin 0.3, which never answers on
32 of 50 dev clips. We moved negatives below G5 and speed; that first pass is in `results/superseded/`. The pool
rule also changed between passes: we added "at least `min_frames`" after seeing that a 0.5 s window holds no frame
to pool at 0.8 s a frame.

An excerpt of `results/tuning.csv`; rank 1 is the chosen setting. "Over 1" counts dev clip-speed pairs with more
than 1 change. "Neg locked" counts old negative clip-speed pairs that lock, of 9. "Median first" is the median first
answer in seconds at each speed.

| rank | window s | margin | switch s | min frames | over 1 | neg locked | median first @0.1 | @0.4 | @0.8 | never @0.4 | final right @0.4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **0.5** | **0.05** | **3** | **1** | 0 | 9 | 1.3 | 2.2 | 2.8 | 15/50 | 14/50 |
| 9 | 3 | 0.3 | 1 | 1 | 0 | 1 | never | never | never | 32/50 | 11/50 |
| 149 | 0.5 | 0.3 | 1 | 1 | 0 | 6 | 3.05 | 5.8 | 5.6 | 21/50 | 11/50 |
| 189 | 1.5 | 0.05 | 3 | 1 | 0 | 7 | 5.0 | 3.8 | 5.2 | 20/50 | 12/50 |
| 221 | 0.5 | 0.05 | 3 | 2 | 0 | 9 | 1.3 | 2.2 | 3.2 | 15/50 | 14/50 |
| 237 | 1 | 0.05 | 3 | 1 | 0 | 9 | 4.0 | 2.2 | 5.2 | 17/50 | 12/50 |

The chosen setting (`results/chosen_settings.json`) is window 0.5 s, margin 0.05, switch 3 s, gate share 0, min
frames 1 and hold 2 s. On dev at 0.1, 0.25, 0.4 and 0.8 s a frame it gives at most 1 change, with no clip over 1.
Never answered: 14, 11, 15 and 14 of 50 (many synthetic clips never reach theta_conf). Right at the end: 14 to 15 of
50.

## 4. The new test clips

We collected 27 new clips: 22 of snakes and 5 with no snake. Their credits are in `new_clips.csv` and
`new_negatives.csv`, and every candidate we looked at, with the reason it was kept or dropped, is in
`search/candidates.csv` (963 rows). The search covered 206 binomial and common-name queries on Wikimedia Commons and
YouTube. Internet Archive gave nothing usable. The clips themselves are downloads (360p) and are not in the repo. We
fixed the set before scoring any of it, and scored it once.

- **Licences.** 2 snake clips and 2 negatives come from Wikimedia Commons, with a stated licence and author. 20
  snake clips and 3 negatives are YouTube uploads whose licence field reads "Creative Commons Attribution licence
  (reuse allowed)", which is CC BY with no version. The uploader is not always the author:
  `daboia_russelii_closeup` comes from a channel named "Royalty Free Footage". This subset needs review before we
  quote or show it.
- **How we know the species.** For 19 clips the page names the species. For 3 we inferred it, so they count for G5
  but not for "final species right". In `naja_kaouthia_bangladesh_hood` the last frames show a coiled dark snake
  that may be a different animal.
- **Coverage.** Commons has very few videos of our species. No *Naja naja*, *Echis* or *Bungarus caeruleus* clip
  passed the checks, and we found no *Hydrophis*.

| Clip | Species | Species known | Licence | Credit | Source |
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

| Negative clip | What it shows | Licence | Credit | Source |
|---|---|---|---|---|
| `negative_eels_hawaii` | two eels | CC0 | John Veit | [Commons](https://commons.wikimedia.org/wiki/File:2_eels_-_Anguilliformes_in_a_tidepool_near_Kona,_Hawaii_get_into_a_fight.ogv) |
| `negative_slug_austria` | a slug | CC BY-SA 4.0 | Martinvie | [Commons](https://commons.wikimedia.org/wiki/File:Limax.webm) |
| `negative_glass_lizard` | a legless lizard | CC BY | Dieuw's wildlife | [YouTube vnTVtuqM8SQ](https://www.youtube.com/watch?v=vnTVtuqM8SQ) |
| `negative_slowworm` | a slowworm | CC BY | Gary Crowder | [YouTube EEBWWyjrZr4](https://www.youtube.com/watch?v=EEBWWyjrZr4) |
| `negative_bengal_monitor` | a monitor lizard | CC BY | Gonzalo Mucientes | [YouTube 5yoXZSawtXo](https://www.youtube.com/watch?v=5yoXZSawtXo) |

## 5. G5 results with A3 (as first scored)

"Changes" counts changes of the shown species after the first answer. "First answer" is when a species is first
named. "Final right" means the species on screen at the end is the clip's species. It is counted only on clips whose
species is stated, and "Checking..." or "no snake" at the end counts as wrong. Files: `results/g5_summary.csv` and
the per-clip `results/*_scores.csv`.

**The 22 new clips, scored once:**

| s per frame | Most changes | Clips over 1 | G5 | Median first answer | Never answered | Final species right |
|---|---|---|---|---|---|---|
| 0.1 | 1 | 0 | **met** | 0.6 s | 1 | 12/19 |
| 0.25 | 1 | 0 | **met** | 0.75 s | 0 | 12/19 |
| 0.4 | 1 | 0 | **met** | 1.2 s | 0 | 12/19 |
| 0.8 | 2 | 1 | **not met** | 1.6 s | 0 | 11/19 |

At 0.8 s, `indotyphlops_braminus_almeria` locks on *Indotyphlops braminus*, switches to *Dendrelaphis tristis* at
16.8 s, and switches back at 27.2 s. With min frames 1, a slow phone pools a single frame, so 5 frames in a row
that favour the wrong species pass the 3 s switch rule. At 0.4 s the same clip has 0 changes. The clips with 1
change are those where the first lock was wrong and the lock then corrected itself. For example
`indotyphlops_braminus_home` goes from *Echis carinatus* to *Indotyphlops braminus*, and
`oreocryptophis_porphyraceus_box` from *Naja naja* to *O. porphyraceus*.

**Verdicts on the new clips** at 0.4 s a frame: every dangerous-species clip showed "dangerous" at some point (9 of
9). But 12 of 13 safe-species clips end on "Dangerous". Without the latch, the rules on the final pool would say
"dangerous" for 5 of 13. So the latch keeps many false alarms on screen, and keeping it is a safety-against-false-
alarm choice for the team. The other option shows "dangerous" only while the current pool says so: the species
still holds, and the verdict can flip. The shown verdict changes at most 2 times on a clip (4 at 0.8 s), for
example "dangerous" while checking, then "not listed" at the lock, then the latch.

**The 6 old test clips, a second look.** These clips already judged the old lock, so this is not a fresh test.

| s per frame | Most changes | G5 | Median first answer | Never answered | Final species right |
|---|---|---|---|---|---|
| 0.1 | 0 | met | 0.6 s | 1 | 1/4 |
| 0.25 | 0 | met | 0.75 s | 0 | 1/4 |
| 0.4 | 0 | met | 1.2 s | 1 | 1/4 |
| 0.8 | 1 | met | 1.6 s | 1 | 1/4 |

`naja_naja_ditmars_1915` now has 0 changes, but it locks on *Fowlea piscator* (still "dangerous", through the
latch). `ahaetulla_borealis_kerala` mostly never answers, and `ptyas_mucosa_swimming` ends with no snake on screen.
The 2 inferred *Naja* clips are right.

**No-snake clips.** Every one locks onto a species and shows "Dangerous" at every speed. At 0.4 s:

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

**Panning (dev clips only, `results/panning.csv`).** We joined two dev clips of different species, each one the lock
gets right on its own. With no gap (4 pairs at 4 speeds), the answer switched to the second species in **15 of 16**
runs, 3.0 to 10 s into the second clip. The one miss was *Daboia* to *Naja kaouthia* at 0.1 s: the first clip had
locked on the wrong species (*Boiga multomaculata*), and in the second clip no rival led it for 3 s. With a 3 s gap
and no snake (2 pairs at 4 speeds), the lock dropped after 2 s and the second species showed in **8 of 8**, 0.5 to
0.8 s into the second clip.

## 6. Parity and tests

- **Python** (`tests/test_video_lock.py`, 10 tests): the window in seconds at three speeds, a short rival, one
  switch, hold expiry, gate share, no snake, unknown settings refused, a slow phone still pooling.
- **Dart** (`app/test/video_lock_test.dart`, 12 tests): window timing at 10, 2.5 and 1.25 fps, min frames at 0.8 s a
  frame, the species held through a short rival, one switch, hold expiry, a short gap keeping the lock, gate share,
  never locking when unsure, "dangerous" while checking with no species, the latch, reset.
- **cameraFrameToRgb** (`app/test/camera_frame_test.dart`, 5 tests): BGRA with row stride, YUV 4:2:0 grey and red,
  NV21 order, sensor rotation with the front-camera mirror, an unknown format.
- **Python against Dart** (`app/test/video_lock_parity_test.dart`): 7 real dev-clip sequences from
  `code/make_parity.py`, 627 steps in all. They cover a lock, a switch, a hold reset, "dangerous" while checking, the
  latch, a not-listed lock, and gate share with a 1.5 s window. Every step matches Python on checking, locked class,
  event, shown verdict and shown species. Values are rounded to 7 decimals before Python computes the expected
  steps.
- **Mutation check.** The tests catch each of these deliberate bugs: showing the pooled top species instead of the
  locked one (7 failures), removing the hold reset, and removing the latch.
- **ONNX against notebook 24's PyTorch outputs**, on 11 clips (`code/check_outputs.py`): gate decision 100%, top-1
  100%. A control with the frames misaligned agrees on only 56%. It is that high because the cobra film is mostly
  one species.

`flutter test` on the G5 branch gave 69 passed and 2 skipped, against 44 passed and 2 skipped before. The full M5
branch count is in the app doc.

## Not checked

- **A phone or emulator.** The Dart lock is tested only in unit and parity tests, on real model outputs. Phone speed
  was simulated with `video.schedule`, not measured.
- **The YouTube licences**, beyond the page field. We did not review every contact sheet by eye, and several species
  IDs rest on the uploader's title.
- **Balance of the new test set.** It is small and lopsided: 3 *Naja kaouthia*, 2 *Bungarus fasciatus*, 2
  *Indotyphlops*. G5 on 22 clips at the passing speeds is consistent with "at most 1 change", but the one failure at
  0.8 s shows the margin is thin.
- **`VideoAnalyzer` tests**, and streaming video in the app, which is outside this branch.
- **Notebook 24 under the new lock** (not rerun).

## Files

| Path | What it is |
|---|---|
| `code/clip_outputs.py` | Runs every frame of every clip through the ONNX reference pipeline. `--models` takes the models folder, so it works for any model. `RESIZE=linear` gives the bilinear check |
| `code/lock_eval.py` | Tuning, scoring and panning on the saved outputs. Takes the app's `--pipeline` and `--tables` |
| `code/make_parity.py` | The Dart parity fixture |
| `code/summarize.py` | Builds `results/g5_summary.csv` |
| `code/verify_old_bugs.py`, `code/old/video_m4.py` | The M4 lock checks behind `bug_check_m4_lock.csv` |
| `code/check_outputs.py` | The ONNX against PyTorch sample check |
| `results/` | `tuning.csv`, `chosen_settings.json`, the per-clip `*_scores.csv`, `negatives.csv`, `panning.csv`, `g5_summary.csv`, the bilinear check, and the first tuning pass (`superseded/`) |
| `outputs_linear/manifest_*.csv` | Which per-frame output file of the bilinear check belongs to which clip. The per-frame outputs themselves (`outputs/`, `outputs_linear/`) are outside git |
| `new_clips.csv`, `new_negatives.csv`, `search/candidates.csv` | The new clips, their credits, and every candidate considered |

To rerun with another model, pass its models folder to `clip_outputs.py`, then run `lock_eval.py` with its
`pipeline.json` and `tables.json`. The outputs are keyed on the hash of the model files. The order is:

1. `clip_outputs.py --models DIR --reference reference_pipeline.py --out DIR` (in shards with `--shard i --shards
   n`), then `--manifest`;
2. `lock_eval.py tune`, then copy `results/tuning_chosen.json` to `results/chosen_settings.json`;
3. `lock_eval.py score --sets ...` and `lock_eval.py pan`.
