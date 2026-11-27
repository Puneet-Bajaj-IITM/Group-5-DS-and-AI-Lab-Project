"""PyTorch on the Kaggle GPU (cv2 decode, fp32) against the local fp16 ONNX through the app's input path (PIL decode,
exposure fix), on every calibration and validation snake both scored.

Usage: python -I parity.py --run <runs/variant>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ("snake", "lookalike", "other")


def main(args):
    run = Path(args.run)
    torch_scores = pd.read_csv(run / "scores_torch.csv")
    logits = torch_scores[[f"logit_{c}" for c in CLASSES]].to_numpy()
    probs = np.exp(logits - logits.max(1, keepdims=True))
    torch_scores["torch_p_snake"] = probs[:, 0] / probs.sum(1)
    local = pd.concat([pd.read_csv(run / args.local / f"scores_{s}.csv") for s in ("calib", "val")])
    both = torch_scores.merge(local[["path", "p_snake"]], on="path")
    diff = np.abs(both["torch_p_snake"] - both["p_snake"])
    cut = json.loads((run / "g4_eval.json").read_text())["rules"][0]["cutoff"]
    flips = int(((both["torch_p_snake"] >= cut) != (both["p_snake"] >= cut)).sum())
    report = {"photos": len(both), "median_abs_diff": float(diff.median()), "p99_abs_diff": float(diff.quantile(0.99)),
              "max_abs_diff": float(diff.max()), "decision_flips_at_cutoff": flips, "cutoff": cut,
              "note": "differences include the decoder (cv2 vs PIL), the exposure fix (app path only) and fp16"}
    (run / "parity_torch_vs_onnx.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--local", default="scores_local", help="folder (in the run) of local ONNX scores")
    main(parser.parse_args())
