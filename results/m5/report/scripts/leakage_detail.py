"""Held-out vs other splits: shared photographers by split, and pHash near-copies (<= 8 bits) vs train/val."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPORT = Path(__file__).resolve().parents[1]
WORK = REPORT.parents[1]
MAX_BITS, CHUNK = 8, 256
HELD = ("test", "test_extra")

meta = pd.read_csv(WORK / "kaggle/clean_build/out/metadata.csv", low_memory=False,
                   usecols=["path", "split", "observer_id", "phash", "species"], dtype=str)
held = meta[meta.split.isin(HELD)].reset_index(drop=True)
out = {"shared_observers_by_split": {}, "heldout_with_phash": int(held.phash.notna().sum())}
for split in ("train", "val", "unknown_pool"):
    shared = set(held.observer_id) & set(meta[meta.split == split].observer_id)
    out["shared_observers_by_split"][split] = {
        "observers": len(shared), "heldout_photos_of_them": int(held.observer_id.isin(shared).sum())}


def as_int(series):
    return np.array([int(h, 16) for h in series], dtype=np.uint64)


h = as_int(held.phash)
for split in ("train", "val"):
    other = meta[meta.split == split].reset_index(drop=True)
    o = as_int(other.phash)
    pairs = []
    for i in range(0, len(h), CHUNK):
        bits = np.bitwise_count(h[i:i + CHUNK, None] ^ o[None, :])
        nearest, dist = bits.argmin(axis=1), bits.min(axis=1)
        pairs += [(i + k, int(nearest[k]), int(dist[k])) for k in np.flatnonzero(dist <= MAX_BITS)]
    same = [(held.path[a], other.path[b], d) for a, b, d in pairs if held.species[a] == other.species[b]]
    out[split] = {f"heldout_within_{MAX_BITS}_bits": len(pairs), "exact": sum(d == 0 for *_, d in pairs),
                  "same_species": len(same), "same_species_pairs": same}
(REPORT / "data/leakage_detail.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
