"""Turn a kernel's scores_torch.csv (logits) into evaluate.py's per-set score files.

Snakes (calibration and validation) always come from the kernel. Validation non-snakes come from the kernel when
it scored them (variant distil_oi onwards), else from a local ONNX scoring folder given with --nonsnake.

Usage: python -I kernel_scores.py --run <runs/variant> [--nonsnake <folder with scores_nonsnake_val.csv>]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
METADATA = ROOT / "kaggle/clean_build/out/metadata.csv"
NB19 = ROOT / "kaggle/m4_runs/non_snake/out_full/manifest.csv"
CLASSES = ("snake", "lookalike", "other")


def main(args):
    run = Path(args.run)
    out = run / "scores_kernel"
    out.mkdir(exist_ok=True)
    rows = pd.read_csv(run / "scores_torch.csv")
    logits = rows[[f"logit_{c}" for c in CLASSES]].to_numpy()
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    for i, c in enumerate(CLASSES):
        rows[f"p_{c}"] = probs[:, i]
    meta = pd.read_csv(METADATA, usecols=["path", "genus", "family"])
    keep = ["path", "group", "category", "label", "genus", "family", "p_snake", "p_lookalike", "p_other"]
    sources = {}
    for name in ("calib", "val"):
        part = rows[rows["set"].eq(name)].merge(meta, on="path", how="left")
        part = part.assign(group=part["observer_id"].astype(int).astype(str), category=part["species"], label=1)
        part[keep].to_csv(out / f"scores_{name}.csv", index=False)
        sources[name] = "kernel"
    kernel_non = rows[rows["set"].eq("nonsnake_val")]
    if len(kernel_non):
        nb19 = pd.read_csv(NB19)
        part = kernel_non.assign(file=kernel_non["path"].str.replace("nonsnake_val/", "", regex=False))
        part = part.drop(columns=["category", "group"], errors="ignore").merge(
            nb19[["file", "owner", "category"]], on="file")
        part = part.assign(path=part["file"], group=part["owner"].astype(str), label=0, genus="", family="")
        part[keep].to_csv(out / "scores_nonsnake_val.csv", index=False)
        sources["nonsnake_val"] = "kernel"
    else:
        local = pd.read_csv(Path(args.nonsnake) / "scores_nonsnake_val.csv")
        local.to_csv(out / "scores_nonsnake_val.csv", index=False)
        sources["nonsnake_val"] = f"local ONNX fp16 ({args.nonsnake})"
    (out / "sources.json").write_text(json.dumps(sources, indent=1) + "\n")
    print(sources, {n: len(pd.read_csv(out / f"scores_{n}.csv")) for n in sources})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--nonsnake")
    main(parser.parse_args())
