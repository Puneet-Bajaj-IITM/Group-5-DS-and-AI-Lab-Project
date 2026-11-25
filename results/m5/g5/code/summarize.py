"""G5 summary per clip set and latency from the score CSVs."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
R = Path(__file__).resolve().parents[1] / "results"
rows = []
for name in ["new_test_scores", "old_test_second_look", "dev_scores"]:
    d = pd.read_csv(R / f"{name}.csv")
    for lat, g in d.groupby("latency_s"):
        scored = g[g.species_source.isin(["stated", "category"])]
        first = g.first_answer_s.where(g.answered, np.inf)
        rows.append({"set": name, "latency_s": lat, "clips": len(g), "max_changes": int(g.changes_after_first_answer.max()),
                     "clips_over_1_change": int((g.changes_after_first_answer > 1).sum()),
                     "G5_met": bool((g.changes_after_first_answer <= 1).all()),
                     "median_first_answer_s": float(np.median(first)),
                     "median_first_answer_answered_s": float(g.first_answer_s.median()),
                     "never_answered": int((~g.answered).sum()),
                     "final_right": f"{int(scored.final_right.sum())}/{len(scored)}",
                     "max_verdict_changes": int(g.verdict_changes.max()),
                     "final_verdict_as_expected": f"{int((g.final_verdict == g.expected_verdict).sum())}/{len(g)}",
                     "safe_species_ending_dangerous": f"{int(((g.expected_verdict == 'not_listed') & (g.final_verdict == 'dangerous')).sum())}/{int((g.expected_verdict == 'not_listed').sum())}",
                     "same_without_latch": f"{int(((g.expected_verdict == 'not_listed') & (g.final_pooled_verdict_no_latch == 'dangerous')).sum())}/{int((g.expected_verdict == 'not_listed').sum())}",
                     "dangerous_species_shown_dangerous": f"{int(((g.expected_verdict == 'dangerous') & g.ever_dangerous).sum())}/{int((g.expected_verdict == 'dangerous').sum())}"})
t = pd.DataFrame(rows); t.to_csv(R / "g5_summary.csv", index=False); print(t.to_string())
n = pd.read_csv(R / "negatives.csv")
print(n[["clip", "set", "latency_s", "snake_frames", "frames", "locked_frames", "species_shown", "final_verdict", "ever_dangerous"]].to_string())
