"""Section 6 Grad-CAM: the shortcut check numbers and the two figures, from the kernel's gradcam outputs.

Shortcut check per photo with a usable U-Net mask: mass = share of the Grad-CAM map inside the mask, area = share of
the classifier's input the mask covers. Reported as means and medians, mass minus area with a photographer bootstrap
(check_full_run.resamples, 1,000, seed 42), and the share of photos with mass > area; for the trained model at the
final block (primary), at the stride-8 map (layer check), and for the randomised-weights model (negative control).
"Photo region" drops the letterbox padding from both mass and area.

Usage: m3-final/.venv/bin/python results/m5/robustness/gradcam_report.py
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
sys.path.insert(0, str(WORK / "m4-checks"))
import check_full_run as cf  # noqa: E402

GRADCAM = HERE / "kernel_out/jayeshbansal497/out/gradcam"
PREDICTIONS = WORK / "results/m5/test/predictions_heldout.csv"
OUT, FIGURES = HERE / "results", HERE / "figures"
COLS, OVERLAY_ALPHA, CONTROL_TILES = 6, 0.45, 6
LAYER_NAMES = {"final": "final block (final_conv, asked for)", "stage4": "last MobileViT block (stages.4)",
               "stride16": "stride-16 stage (stages.3)", "stride8": "stride-8 stage (stages.2)"}
MAPS = {**{f"trained, {label}": (f"mass_in_mask_{k}", f"mass_on_pad_{k}") for k, label in LAYER_NAMES.items()},
        **{f"random weights, {label}": (f"control_mass_in_mask_{k}", f"control_mass_on_pad_{k}")
           for k, label in LAYER_NAMES.items()}}


def interval(values, groups):
    draws = [values[r].mean() for r in cf.resamples(groups)]
    return [float(v) for v in np.percentile(draws, (2.5, 97.5))]


def summary(table, groups):
    area = table["mask_share_of_input"].to_numpy()
    photo_area = area / (1 - table["pad_share_of_input"].to_numpy())
    out = {"photos": len(table), "mask_area_mean": float(area.mean()), "mask_area_median": float(np.median(area)),
           "mask_area_photo_region_mean": float(photo_area.mean())}
    for name, (mass_col, pad_col) in MAPS.items():
        if mass_col not in table:
            continue
        mass = table[mass_col].to_numpy()
        ok = np.isfinite(mass)
        diff = mass[ok] - area[ok]
        row = {"photos": int(ok.sum()), "mass_mean": float(mass[ok].mean()), "mass_median": float(np.median(mass[ok])),
               "mass_minus_area_mean": float(diff.mean()), "mass_minus_area_ci": interval(diff, groups[ok]),
               "share_with_mass_above_area": float((diff > 0).mean()),
               "median_mass_over_area": float(np.median(mass[ok] / area[ok]))}
        if pad_col in table:
            on_photo = 1 - table[pad_col].to_numpy()[ok]
            usable = on_photo > 0
            photo_mass = mass[ok][usable] / on_photo[usable]
            row["photo_region_mass_mean"] = float(photo_mass.mean())
            row["photo_region_mass_minus_area_mean"] = float((photo_mass - photo_area[ok][usable]).mean())
            row["mass_on_padding_mean"] = float(table[pad_col].to_numpy()[ok].mean())
        out[name] = row
    return out


def overlay(ax, canvas, cam, mask, title):
    cam = cam.astype(np.float32)
    cam = cam / cam.max() if cam.max() > 0 else cam
    ax.imshow(canvas)
    ax.imshow(cam, cmap="jet", alpha=OVERLAY_ALPHA, vmin=0, vmax=1)
    ax.contour(mask, levels=[0.5], colors="white", linewidths=0.8)
    ax.set_title(title, fontsize=7)
    ax.axis("off")


def short(name):
    genus, *rest = name.split()
    return f"{genus[0]}. {' '.join(rest)}"


def figures(chosen):
    for layer, label in (("final", "final block (final_conv)"), ("stride8", "stride-8 stage (stages.2)")):
        grid(chosen, layer, label)
    control_figure(chosen)


def grid(chosen, layer, label):
    saved = np.load(GRADCAM / "gradcam_selected.npz", allow_pickle=True)
    index = {p: i for i, p in enumerate(saved["paths"].astype(str))}
    chosen = chosen[chosen["path"].isin(index)].reset_index(drop=True)
    rows = int(np.ceil(len(chosen) / COLS))
    fig, axes = plt.subplots(rows, COLS, figsize=(COLS * 2.1, rows * 2.35))
    for ax in axes.ravel():
        ax.axis("off")
    for k, row in chosen.iterrows():
        i = index[row["path"]]
        mark = "correct" if row["kind"] == "correct" else "wrong"
        title = "\n".join(textwrap.wrap(f"{mark}: {short(row['species'])}" + (
            "" if row["kind"] == "correct" else f" -> {short(row['frozen_pred'])}"), 30))
        overlay(axes.ravel()[k], saved["canvas"][i], saved[f"cam_{layer}"][i], saved["mask"][i], title)
    fig.suptitle(f"Grad-CAM, M5_KD {label}, for the frozen answer's class (white line: U-Net mask)", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / f"gradcam_grid_{layer}.png", dpi=160)
    plt.close(fig)


def control_figure(chosen):
    saved = np.load(GRADCAM / "gradcam_selected.npz", allow_pickle=True)
    index = {p: i for i, p in enumerate(saved["paths"].astype(str))}
    chosen = chosen[chosen["path"].isin(index)].reset_index(drop=True)

    pick = pd.concat([chosen[chosen["kind"].eq("correct")].head(CONTROL_TILES // 2),
                      chosen[chosen["kind"].eq("misclassified")].head(CONTROL_TILES // 2)])
    columns = [("cam_final", "trained, final block"), ("cam_stride16", "trained, stride 16"),
               ("cam_stride8", "trained, stride 8"), ("control_cam_final", "random, final block"),
               ("control_cam_stride8", "random, stride 8")]
    fig, axes = plt.subplots(len(pick), len(columns) + 1, figsize=(2.1 * (len(columns) + 1), 2.1 * len(pick)))
    for r, (_, row) in enumerate(pick.iterrows()):
        i = index[row["path"]]
        axes[r, 0].imshow(saved["canvas"][i])
        axes[r, 0].set_title(short(row["species"]), fontsize=7)
        axes[r, 0].axis("off")
        for c, (key, label) in enumerate(columns, start=1):
            overlay(axes[r, c], saved["canvas"][i], saved[key][i], saved["mask"][i], label)
    fig.tight_layout()
    fig.savefig(FIGURES / "gradcam_control.png", dpi=160)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    table = pd.read_csv(GRADCAM / "gradcam_mass.csv")
    table = table[table["mask_share_of_input"] > 0].reset_index(drop=True)
    preds = pd.read_csv(PREDICTIONS, usecols=["path", "observer_id", "species", "frozen_pred"],
                        dtype={"observer_id": str}).set_index("path").loc[table["path"]]
    groups = cf.groups_of(preds["observer_id"].fillna("").to_numpy(), table["path"].to_numpy())
    correct = (preds["species"] == preds["frozen_pred"]).to_numpy()
    report = {"all": summary(table, groups), "frozen answer right": summary(table[correct].reset_index(drop=True),
                                                                            groups[correct]),
              "frozen answer wrong": summary(table[~correct].reset_index(drop=True), groups[~correct])}
    (OUT / "gradcam.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["all"], indent=1))
    figures(pd.read_csv(HERE / "lists/gradcam_photos.csv"))


if __name__ == "__main__":
    main()
