"""Sample check: ONNX per-frame outputs against notebook 24's PyTorch cache (same frames, same A3)."""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
WORK = Path(__file__).resolve().parents[3]
G = WORK / "results/m5/g5"
man = pd.read_csv(next((G / "outputs").glob("manifest_*.csv")))
rng = np.random.default_rng(7)
rows = []
for row in man[man.set.isin(["dev_synthetic", "dev_real", "old_test", "old_negative"])].sample(10, random_state=3).pipe(lambda f: pd.concat([f, man[man["clip"] == "bungarus_fasciatus_milking.webm"]])).itertuples():
    old = glob.glob(str(WORK / f"m4-video-clips/cache/A3_*__{row.clip}.npz"))
    with np.load(G / "outputs" / row.outputs) as new, np.load(old[0]) as o:
        idx = o["frame_index"]
        both = o["passed"] & new["passed"][idx]
        new_top = np.nanargmax(np.where(np.isnan(new["species_logits"][idx]), -np.inf, new["species_logits"][idx]), axis=1)
        old_top = o["probs"].argmax(axis=1)
        rows.append({"clip": row.clip, "frames": len(idx), "gate_agree": float(np.mean(o["passed"] == new["passed"][idx])),
                     "top1_agree_on_both_passed": float(np.mean(new_top[both] == old_top[both])) if both.any() else None,
                     "nan_in_passed_rows": int(np.isnan(new["species_logits"][new["passed"]]).sum()),
                     "non_nan_in_failed_rows": int((~np.isnan(new["species_logits"][~new["passed"]])).sum())})
print(pd.DataFrame(rows).to_string())
# negative control: shuffled frame alignment should agree far less
row = man[man["clip"] == "naja_naja_ditmars_1915.mp4"].iloc[0]
with np.load(G / "outputs" / row.outputs) as new, np.load(glob.glob(str(WORK / "m4-video-clips/cache/A3_*__naja_naja_ditmars_1915.mp4.npz"))[0]) as o:
    idx = rng.permutation(o["frame_index"]); top = np.nanargmax(new["species_logits"][idx], axis=1)
    print("control, misaligned frames, top-1 agreement:", float(np.mean(top == o["probs"].argmax(axis=1))))
