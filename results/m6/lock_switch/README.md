# The video lock switches to a new snake after 2 s instead of 3 s

After the M5 review (28 November 2026) the lock was judged too slow to move to a new snake. Development clips
only; the M5 test clips were not used. The M5 frozen pipeline keeps 3 s; the app uses 2 s from M6 on.

| Switch time | Dev clips with more than 1 change (any speed) | Final species right (all speeds) | Two joined clips, no gap: switched | Seconds into the second clip until shown (median) |
|---|---:|---:|---|---:|
| 3 s (M5) | 0 | 43 | 15 of 16 | 4.0 |
| **2 s (M6)** | **0** | 42 | **16 of 16** | **2.8** |
| 1 s | 1 | 42 | not run (fails the change rule) | |

Dev-clip rows: `results/m5/g5/results/tuning.csv` (window 0.5 s, margin 0.05, min frames 1, gate share 0). Panning:
`code/lock_eval.py pan` from `results/m5/g5` on the saved clip outputs, with `chosen_settings.json` changed only in
`switch_seconds` (`switch_3.0/`, `switch_2.0/`). The 3 s run reproduces `results/m5/g5/results/panning.csv` (15 of
16), so the harness is the one M5 used. With a 3 s gap and no snake, both settings switch in 8 of 8 runs.
