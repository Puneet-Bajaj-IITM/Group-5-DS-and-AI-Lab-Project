# Milestone 4 presentation: speaker script

For `SerpentGuard_AI_M4_Presentation.pdf` (12 slides, updated 20 November 2026). About 12 to 14 minutes in all. Each slide has:

- **On the slide:** what the audience sees.
- **Say:** the script to speak, in plain words.
- **If asked:** short answers for the questions that slide tends to raise.

Numbers are from the Milestone 4 report; the section is given where useful. Suggested speakers follow the team contributions on slide 12, but one person can present everything.

---

## Slide 1: Title, "SerpentGuard AI: model training and offline application"

*About 40 seconds. Suggested speaker: Jayesh (team lead).*

**On the slide:** project name, Milestone 4, "Model training and offline application", 184 of India's 300+ snake species, Group 5, the four members, updated 20 November 2026.

**Say:**
"Good morning. We are Group 5, and this is SerpentGuard AI. It identifies snakes from a photo or video and gives a cautious safety answer, entirely offline on a phone.

The model covers 184 of the more than 300 snake species recorded in India. Those are the species with enough licensed photos to train on.

Milestone 4 had two jobs. The first was to train the model properly on all our data, and to fix the two problems Milestone 3 left us. The second was to get that trained model running inside the phone app.

I'll go through what we set out to do, what worked, what didn't, and what we check next."

---

## Slide 2: Milestone 4 goals

*About 1 minute 30 seconds. Suggested speaker: Puneet or Jayesh.*

**On the slide:** six goals with target, A3's result and status. G1 and G2 are met, G3, G4 and G5 are not met, and Safety is met. The red note says the app refit gives WHO recall 96.5% (95.3% to 97.7%) with false alarms of 24.9%, and that the original safety bound 0.958 uses M3 crops while the app's bound is 0.954.

**Say:**
"These are the goals we set at the end of Milestone 3. They are judged once, on the model we ship, which is called A3.

**G1 is about the background.** In Milestone 3 the model leaned on the scenery around the snake. If you swapped the background for another photo's, accuracy dropped by about 17 points. We call that drop the swap gap. Our target was to halve it, to 0.084 or less. A3's gap is 0.052, so G1 is met.

**G2 is about how many species the model knows well.** Our target was at least 80 of the 158 species we can measure reaching an F1 score of 0.5. A3 reaches 87, up from 56 in Milestone 3. Met.

**G3 compares us with B1, a published global model.** On the 62 species B1 knows, it is still better than us: 8 points of top-1 and 0.025 macro-F1. Not met.

**G4 is about false snakes,** photos with no snake that the app still calls a snake. The target was 5% or less. We are at 28.3%, while still keeping 98.3% of real snakes. Not met.

**G5 is about video stability.** We wanted the displayed species to change at most once per clip. The worst real clip changed twice. Not met.

**The safety rule is the most important one:** the lower bound of WHO recall must be at least 0.95. WHO recall is how often a WHO-listed venomous snake is called dangerous. The original bound was 0.958, but that was measured on Milestone 3's crops. The app crops photos slightly differently, so we refitted the thresholds for the app's own front end. There the bound is 0.954, and validation recall is 96.5%. The cost is false alarms of 24.9%, which we accept, because missing a dangerous snake is the worse mistake.

So three goals are met, three are not, and the safety rule holds in the app."

**If asked:**
- *Why accept 25% false alarms?* A false alarm means someone keeps away from a harmless snake. A miss means someone may treat a krait as harmless. The thresholds are set from the safety side.
- *Why "lower bound"?* We resample the calibration photographers 1,000 times. The 5th percentile of recall must reach 0.95, so the rule holds even with some bad luck in the sample.

---

## Slide 3: Background ablation on the subset

*About 1 minute 15 seconds. Suggested speaker: Puneet.*

**On the slide:** four arms on the 16% subset (A0 baseline, A1 box crop plus scenery swap, A2 snake only on grey, A3 mask-box crop plus scenery swap), with macro-F1, swap gap and whether each qualifies. A3 is selected.

**Say:**
"Before spending GPU time on the full data, we tested ideas on a 16% subset of the training photos. Each idea is an 'arm': a different way of showing the model the snake.

**A0 is Milestone 3's model,** a crop to the detector's box. Macro-F1 0.367, swap gap 0.168. This is what every other arm must beat.

**A1 keeps the box crop but swaps the background during training.** Half the time, the snake is pasted onto another species' scenery, so the background stops being a reliable clue. Accuracy stays the same and the gap almost halves, to 0.093.

**A2 is the extreme:** the snake alone, everything else grey. The gap nearly vanishes, 0.006, but accuracy falls to 0.333. The scenery does carry some honest information, like habitat, so removing it all costs too much. A2 does not qualify.

**A3 crops to the box around the predicted snake mask,** which is tighter than the detector's box, and also swaps the background. Accuracy is the same as A0's and the gap is lowest among the qualifying arms, 0.063.

The rule was fixed in advance: an arm qualifies if it keeps macro-F1 and WHO recall close to A0's, and the smallest gap wins. A3 won."

**If asked:**
- *What does "qualifies" mean exactly?* Macro-F1 within 0.005 and WHO recall within 0.006 of A0.
- *Why not use A2 if it removes the shortcut?* It loses 3.4 points of macro-F1. Swapping teaches the model to ignore misleading scenery while keeping useful context.

---

## Slide 4: Training on every photo

*About 1 minute 15 seconds. Suggested speaker: Puneet.*

**On the slide:** a bar chart of validation macro-F1 (M3 0.367, A1 0.481, H2 0.437, A3 0.481); 49,782 training photos, 30 epochs, 184 species. A3 is selected: 66.4% top-1, 82.2% top-3, 87 species at F1 ≥ 0.5, best epoch 24 of 30, AdamW, batch 64, seed 1, learning rate 0.000953, weight decay 0.00156.

**Say:**
"We then trained the best recipes on every eligible training photo: 49,782 photos, 184 species, 30 epochs each.

The jump from Milestone 3 is large. Macro-F1 goes from 0.367 to 0.481. Most of that gain comes from simply training on all the data instead of the subset.

**H2 adds a 'detail crop',** a second, closer look at a region the model chooses itself. It helped a little on the subset, but on the full data it is worse, 0.437. It also doubles the time per frame, so we dropped it.

**A1 and A3 tie at 0.481.** We ship A3 for three reasons: its swap gap is smaller (0.052 against 0.075), fewer of its species score zero, and its crop is the one the phone app's front end produces.

A3 reaches 66.4% top-1 and 82.2% top-3 on validation, with 87 species at F1 0.5 or above. Its best epoch was 24 of 30, chosen on validation macro-F1. The optimiser settings come from Milestone 3's tuning search: AdamW, batch 64, learning rate about 0.00095, weight decay about 0.0016."

**If asked:**
- *Why does training on all photos help so much?* Most species are rare. On the subset many had only 20 photos; with the full data the model sees far more examples of each.
- *Did you re-tune hyperparameters for the full data?* No. We kept Milestone 3's tuned recipe and changed only what was being tested.

---

## Slide 5: Calibration and published baseline

*About 1 minute 15 seconds. Suggested speaker: Puneet or Jayesh.*

**On the slide:** original calibration (ECE 0.142 before scaling, 0.040 after, temperature 1.806, 7,999 calibration photos). On B1's 62 species, 2022 photos: top-1 70.7% for A3 against 78.8% for B1, macro-F1 0.694 against 0.719. "B1 still leads on the species it knows. A3 covers 122 additional species." The red note says G3 remains unmet, this calibration uses the original M3 crops, and the app uses temperature 1.804 and refitted safety thresholds.

**Say:**
"Two things on this slide.

**First, calibration: whether the model's confidence can be trusted.** Out of the box A3 is overconfident. When it says 99%, it is right less often than that. The expected calibration error is 0.142. We fix this with temperature scaling: divide the model's scores by a single number, here 1.806, fitted on 7,999 calibration photos kept apart from training. The error drops to 0.040. This matters because our decision rules read these probabilities directly. In the app, refitted on the app's own crops, the temperature is 1.804, almost the same.

**Second, the published baseline B1,** a vision transformer trained on the SnakeCLEF 2021 data. B1 knows only 62 of our species. For a fair test we use validation photos uploaded from 2022 onwards, which B1 cannot have seen in training. On those species B1 still leads: 78.8% against 70.7% top-1, and 0.719 against 0.694 macro-F1. So G3 is not met.

But the picture is broader. A3 covers 122 more species that B1 cannot name at all. A3 is also about 1% of B1's size, small enough to run on a phone. The gap also shrank: in Milestone 3 we trailed by about 11 points of top-1 and 0.125 macro-F1."

**If asked:**
- *What is ECE?* Photos are grouped by confidence, and we compare the average confidence with the actual accuracy in each group. A lower ECE means the stated confidence is more honest.
- *Why compare on 2022 photos?* B1's training data ends before 2022, so these photos are new to it.

---

## Slide 6: The no-snake gate improves, but misses G4

*About 1 minute 15 seconds. Suggested speaker: Puneet.*

**On the slide:** non-snake test photos called a snake: M3 box 95.7%, M4 box 40.7%, M4 both 28.3%. Example false positives (caecilian, earthworm-like animals, a lizard in a hand). "Lookalikes still pass." 98.3% of validation snakes are kept. The red note says gate test false positives remain 28.3% (target 5%), and an exploratory BioCLIP check reached 13.7% on validation but is 599 MB and not shipped in the app.

**Say:**
"Before the classifier runs, a 'gate' decides whether there is a snake at all. In Milestone 3 that gate was nearly useless on non-snakes: it called 95.7% of photos without a snake a snake.

We collected about 3,000 openly licensed photos with no snake in them and retrained the detector and the U-Net segmenter with them. Using the detector alone, false snakes fall to 40.7%. Requiring both the detector box and the segmenter mask, which we call the 'both' gate, brings it to 28.3% on the test photos, while keeping 98.3% of real snakes.

That is a big improvement, but still far from the 5% target, so G4 is not met.

The pictures on the right show why. The photos that still pass are lookalikes: caecilians, earthworms, legless lizards. They have a snake's shape, and our detector and classifier were trained to find exactly that shape.

We also tried an exploratory check using BioCLIP, a large biology model. It halves the false snakes to 13.7% on validation, but it is 599 MB, far too big for a phone, so it is not in the app. It points to the next step: a small snake / not-snake classifier, trained with lookalikes as their own class."

**If asked:**
- *Why not just raise the gate thresholds?* Every step up drops more real snakes, especially sea snakes and sea kraits. Adding more lookalikes to the gate's training also made it worse.
- *What does the app do with a false snake?* It gives a cautious verdict. A false snake is an inconvenience, not a safety failure.

---

## Slide 7: Video lock steadies the displayed species

*About 1 minute 15 seconds. Suggested speaker: Puneet or Jayesh.*

**On the slide:** A3 on six real test clips, before the threshold refit. The frame vote answered 6 of 6 clips, with at most 17 changes, first answer at 0.4 s, final species right on 1 of 4. The video lock answered 5 of 6, with at most 2 changes, first answer at 0.5 s, final species right on 1 of 4. "Stability improves, but G5 allows at most one change." In red: "All three non-snake clips still lock onto a snake and show 'Dangerous'."

**Say:**
"On a live camera or a video, the model sees many frames, and its answer can jump from species to species. In Milestone 3 we showed a species when 4 of the last 6 frames agreed. With A3, that frame vote changed the displayed species up to 17 times on one real clip. That is confusing and unsafe.

**The video lock is a new rule.** It averages the model's outputs over the last 5 frames that contain a snake. It shows 'Checking…' until the averaged answer is confident and clearly ahead of the runner-up, then it locks on that species. Another species can take over only if it clearly leads for a full second. For safety, a 'Dangerous' warning can still show before the lock.

On six real test clips, the worst case drops from 17 changes to 2, and the first answer still arrives in about half a second. The lock answered 5 of the 6 clips; on the sixth the frames never agreed.

**Two honest limits.** G5 asked for at most one change, so it is not met. And all three clips with no snake (a chameleon, an eagle and a river) still lock onto a snake and show 'Dangerous'. That is the false-snake problem from the previous slide, carried into video."

**If asked:**
- *Why only 1 of 4 right on the final species?* These are hard real clips, and the lock fixes stability, not accuracy. The verdict can still be right when the exact species is not.
- *How were the window and margin chosen?* Tuned on development clips only: a window of 5 frames and a margin of 0.05. The test clips were scored once.

---

## Slide 8: Android inference pipeline and checks

*About 1 minute 30 seconds. Suggested speaker: Jayesh.*

**On the slide:** a six-step flow:

1. Photo or frame, upright decode and exposure fix
2. M4 YOLO26n detector and U-Net segmenter
3. "Both" gate: no snake, or the cleaned mask crop plus 6%
4. 320 × 320 crop, MobileViT-S and hierarchy heads
5. DR0 with thresholds refitted for the M4 front end
6. Video or live camera: pooled species lock

On the right: 200/200 refitted decision fixtures with Python/Dart agreement; 5/5 pre-refit emulator photos, to repeat at the new thresholds. The red note says device and video results predate the refit, and physical-phone acceptance remains pending.

**Say:**
"This is how the trained model runs inside the app, fully offline on the phone. The three models (detector, segmenter and classifier) are exported to ONNX and run with ONNX Runtime. Every step follows the team's Python reference pipeline exactly.

Step by step:
- The photo or video frame is read upright, and if it is very dark or very bright we fix its exposure.
- The YOLO26n detector and the U-Net segmenter look for a snake.
- The 'both' gate decides: either 'no snake found', or we clean up the mask and crop to its box plus 6% padding.
- That crop is resized to 320 by 320 and goes into MobileViT-S, which outputs species, genus, family and WHO-category scores.
- The DR0 decision rules, with the thresholds refitted for the app, turn those scores into one of three verdicts: dangerous, not on the WHO list, or we don't know.
- For video and the live camera, the lock from the previous slide steadies the answer.

**How do we know the app matches the Python results?**
- The decision rules agree on 200 out of 200 saved test cases. These are the refitted fixtures.
- Every image step was written in Dart to match OpenCV pixel for pixel.
- On the Android emulator, all 5 reference photos give the same detection, crop, top 3 species, verdict and message.

The slide says the 5 photos were checked before the refit. Since the slide was made, we repeated that check with the refitted thresholds, and all 5 still pass.

We also ran two real video clips through the app. A cobra clip gives 'Dangerous, WHO Category 1, Spectacled cobra' with no changes. A red-tailed ratsnake clip gives 'Not on the WHO list'.

What is still pending is testing on a physical phone: speed, memory and battery, and a real snake filmed live."

**If asked:**
- *How is the model connected to the app?* The ONNX files sit inside the app, ONNX Runtime runs them on the phone, and our Dart code does the preprocessing and the decision rules. There is no server.
- *How fast is it?* About 0.5 to 1 second per photo on the emulator. It has not been timed on a phone yet.
- *What was the hardest bug?* On Android, Flutter's own JPEG decoder gave slightly different pixels, enough to change a detector score from 0.54 to 0.32. Using Android's native decoder fixed it.
- *Why not fp16 or int8 to save space?* fp16 changed 1 verdict and 4 top species in 300 photos, so we kept fp32 (about 54 MB).

---

## Slide 9: Community review survives loss of signal

*About 1 minute. Suggested speaker: Vikas.*

**On the slide:** four steps:

1. Save 1–3 stills as a local encounter.
2. Upload the same evidence when signal returns.
3. Review at species or genus level.
4. Export only after consent and curator approval.

Role weights: citizen 1, verified rescuer 2, expert 3. "Species first, genus fallback. Consensus: posterior ≥ 0.85 and support weight ≥ 3. Each reviewer has one active vote." "Notebook scenarios validate behaviour. Real reviewer calibration remains pending."

**Say:**
"The model cannot know every snake, so we added community review, and it is designed for places with poor signal.

The user saves 1 to 3 still photos as an 'encounter' on the phone, in a local database, even with no network. When signal returns, the app uploads exactly the same evidence. Retries never create duplicates.

Other users then review it, at species level or, if they are unsure, at genus level. Votes are weighted by role: a citizen counts 1, a verified snake rescuer 2, an expert 3.

A species label is accepted only when its posterior confidence reaches 0.85 and the supporting weight reaches 3. So one expert can settle it, or three citizens who agree. If no species reaches consensus, the system falls back to the genus. Each reviewer has one active vote, and voting on your own sighting is not allowed.

Nothing goes into future training data without the user's consent and a curator's approval. If votes change later, that approval is cancelled.

We tested the behaviour in a notebook with eight scenarios. The role weights are still assumptions: they need calibrating with real reviewers."

**If asked:**
- *Why weights 1, 2, 3?* They are starting assumptions, to be calibrated once we have real reviewers' accuracy.
- *Does community data retrain the model automatically?* No. Labels need consent, consensus and curator approval, and retraining is a separate, deliberate step.

---

## Slide 10: Regional support and offline services

*About 1 minute 15 seconds. Suggested speaker: Jayesh and Vikas.*

**On the slide:**

| Feature | What exists | Next check |
|---|---|---|
| Regional names | Draft names for 9 of 16 WHO-listed species | Native-speaker review |
| Community review | SQLite queue and cached reviews | Reconnect to sync |
| Hospitals and map | Downloaded hospitals and map tiles | Complete regional coverage |
| BLE alerts | Opt-in foreground alerts and relay | Two-phone acceptance test |
| First aid and TTS | Card loader and offline voice support | Reviewed text and installed voices |
| English retrieval | RRF recall@3 50.8% on 59 questions | 15% on 20 paraphrases, experimental |

"Offline inference runs locally. Shared community updates require connectivity."

**Say:**
"Beyond identification, the app is built to help in a real encounter, mostly without internet.

- **Regional names.** Species can now show names in Hindi, Marathi, Bengali, Tamil, Telugu, Kannada and Malayalam. So far 9 of the 16 WHO-listed species have them, including the Big Four. They are drafts until native speakers check them. Where we weren't sure, we left the name out rather than guess.
- **Community review** works offline through a local queue and syncs when connected, as on the previous slide.
- **Hospitals and map.** The app downloads nearby hospitals from OpenStreetMap and caches map tiles, so both work later without signal. We still need complete regional coverage. OpenStreetMap also doesn't record anti-venom stock, so the app tells users to call ahead.
- **Bluetooth alerts.** Users can opt in to warn nearby phones with a small Bluetooth broadcast. It carries the species, the verdict and a location rounded to about 100 metres, and no personal details. It still needs a two-phone test.
- **First aid.** The app can show first-aid cards and read them aloud with an offline voice. The reviewed regional text is not bundled yet.
- **Retrieval.** We tested searching WHO first-aid passages for typed questions in English. Combining keyword search with a small language model reached 50.8% recall in the top 3 on 59 questions, but only 15% on reworded questions. It stays experimental and is not in the app.

The key point: identification always runs locally. Only shared community updates need a connection."

**If asked:**
- *What if someone is bitten and there's no internet?* The verdict, the SOS screen, saved hospitals and the dialler all work offline. Calls and SMS need mobile signal, but not data.
- *Why only 9 of 16 names?* The other seven are mostly north-eastern or Kashmir species without names we could confirm. We prefer no name to a wrong one.

---

## Slide 11: Generalization and next checks

*About 1 minute. Suggested speaker: Puneet.*

**On the slide:**

| Check | Train | Validation | Note |
|---|---|---|---|
| Top-1 on an even species sample | 72.4% | 47.1% | 1,499 / 1,500 photos |
| Rare-species macro-F1 | 0.888 | 0.272 | fewer than 30 training photos |
| Full-validation macro-F1, A3 vs A1+A3 average | 0.481 | 0.497 | experimental average, not shipped |

Next: repeat device and video goldens after the threshold refit; measure live camera, latency, memory and battery on a physical phone. In red: "G3, G4 and G5 remain unmet. The snake test split stays locked for M5."

**Say:**
"This slide asks whether the model generalises.

On a sample with the same number of photos per species, top-1 is 72.4% on training photos but 47.1% on validation photos. That gap tells us part of the problem is variance.

It is clearest for rare species, those with fewer than 30 training photos: macro-F1 is 0.888 on their training photos but only 0.272 on new ones. The model nearly memorises the few photos it has, and struggles with new ones.

As a quick test of that reading, we averaged the predictions of A1 and A3. Macro-F1 rises from 0.481 to 0.497. Averaging reduces variance, so that fits. But running two models doubles the cost on the phone, so it is an experiment, not shipped.

Next checks:
- Repeat the device and video tests at the new thresholds. The photo tests are already repeated and pass.
- Measure the live camera, speed, memory and battery on a real phone.

G3, G4 and G5 remain open. And importantly, the snake test split has never been touched. It stays locked for Milestone 5, so our final numbers will be honest."

**If asked:**
- *How will you fix rare species?* More regularisation (dropout, DropPath), more licensed photos where possible, and possibly a two-model average if it fits the size and time budget.
- *Why keep the test set locked?* Every choice so far used validation. Scoring the test set once at the end gives an unbiased number.

---

## Slide 12: Team contributions and sign-off

*About 40 seconds. Suggested speaker: Jayesh, or each member says their own line.*

**On the slide:**

| Member | Contribution | Roll number |
|---|---|---|
| Jayesh Bansal | Flutter app, real model, video and regional names | 24f100185 |
| Vikas | Voting, offline app, retrieval and team presentation | 23f3001800 |
| Puneet Bajaj | Training, gate, video lock, model bundle and safety refit | 23f3002416 |
| Manas Rastogi | Startup, branding, widget tests and CameraX build fix | 22f3001477 |

Each row has a typed-name sign-off. "Milestone 4 team submission."

**Say:**
"To close, here is who did what.

- **Puneet** led the model work: the training runs, the no-snake gate, the video lock, the model bundle for the app, and the safety refit.
- **I, Jayesh,** built the Flutter app, connected the real model to it, added video checking and the live-camera connection, and added the regional names.
- **Vikas** built community voting and the offline app features, ran the retrieval experiment, and prepared this presentation.
- **Manas** built the app's startup and branding, wrote widget tests, and fixed the Android camera build.

All four of us have reviewed and signed off this submission.

In short: the model is much better than in Milestone 3, it runs fully offline in the app, and the safety rule holds. Three goals remain open, and we've shown exactly where we fall short and what we test next. Thank you. We're happy to take questions."

---

## Quick reference: numbers worth remembering

| Number | Meaning |
|---|---|
| 184 | species the model covers |
| 49,782 | training photos |
| 0.367 → 0.481 | macro-F1, Milestone 3 to A3 |
| 66.4% / 82.2% | A3 top-1 / top-3 on validation |
| 56 → 87 | species at F1 ≥ 0.5 |
| 0.168 → 0.052 | swap gap (reliance on scenery) |
| 0.142 → 0.040 | calibration error before / after temperature 1.806 |
| 0.958 / 0.954 | WHO recall lower bound: original / app |
| 96.5% | WHO recall in the app after the refit (false alarms 24.9%) |
| 95.7% → 28.3% | non-snakes called a snake, M3 gate to M4 "both" gate |
| 98.3% | real snakes kept by the gate |
| 17 → 2 | most species changes on a real clip, frame vote to lock |
| 200/200, 5/5 | decision cases and golden photos matching in the app |
| 0.5 to 1 s | time per photo on the Android emulator |
| 54 MB | the three fp32 models |
