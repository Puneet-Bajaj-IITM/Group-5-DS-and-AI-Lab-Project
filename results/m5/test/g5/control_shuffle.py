"""G5 negative control: each new test clip's per-frame classifier outputs shuffled across its frames (gate calls kept),
then lock_eval scored as the frozen run. The lock must then show more than one change on some clip."""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "outputs", HERE / "control_outputs"
KEYS = ("embedding", "species_logits", "who_logits", "genus_logits", "family_logits")
DST.mkdir(exist_ok=True)
manifest = next(SRC.glob("manifest_*.csv"))
rows = pd.read_csv(manifest)
rows = rows[rows["set"].eq("new_test")]
rows.to_csv(DST / manifest.name, index=False)
rng = np.random.default_rng(0)
for name in rows["outputs"]:
    saved = dict(np.load(SRC / name))
    passed = np.flatnonzero(saved["passed"])
    order = rng.permutation(passed)
    for key in KEYS:
        saved[key][passed] = saved[key][order]
    np.savez(DST / name, **saved)
print(len(rows), "clips shuffled")
