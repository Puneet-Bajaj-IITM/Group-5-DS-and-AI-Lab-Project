# Milestone 4 review feedback (TA Siddhant Bapna, posted to the group after the review)

Pasted by Puneet on 2026-11-25. Summary of the points, in the order given:

## Team
1. App readiness: live camera detection non-functional; uploaded video runs at about 1 frame/s behind a blocking
   progress bar. Stream uploaded and live frames, draw bounding boxes on images and video, let the video lock update
   when panning across several snakes.
2. All device checks ran on an emulator: test on a physical mid-range phone.
3. Replace raw coordinates with readable location names; replace the green startup video; submit the promised live
   camera demo.
4. G3, G4 (28.3% false positives; all three non-snake clips lock as "Dangerous") and G5 (up to 2 changes) unmet.
5. Align A3's training crops with the M4 front-end crops; rare-species overfitting (0.888 train vs 0.272 validation
   macro-F1); log validation loss and full checkpoint states; compress the 53.0 MB fp32 bundle toward 25-30 MB.
6. A ~3.5 GB Gemma model on the phone next to the vision pipeline is unrealistic for mobile RAM.
7. Open app gaps: local rescuer/vet contact search and routing; two-phone Bluetooth alert test; regional names for
   only 9 of 16 WHO-listed species.
8. M5: evaluate the frozen pipeline once on the locked test split, with thorough error and limitation analysis.
9. M6: bonus "Pokedex mode for Humans" (separate tab identifying common everyday and dangerous animals).

## Individual
- Jayesh: fix live camera, stream video inference with boxes, readable location names, report sign-off.
- Puneet: resolve G4/G5, log validation loss, complete the locked test evaluation.
- Manas: push his changes, complete the live camera pipeline, improve the startup screen, Appendix B entry and sign-off.
- Vikas: attend reviews, report sign-off, help with M5 evaluation and error analysis.
