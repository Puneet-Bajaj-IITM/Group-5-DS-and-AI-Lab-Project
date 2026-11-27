"""The report's two data figures, from saved M5 outputs only (laptop CPU, no model run).

Figure 2: the snake check's two error rates on test as the cut-off moves, with the frozen cut marked.
Figure 4: what each kind of photo damage does to top-1 and to WHO recall.

Usage: python report_figures.py <predictions_heldout.csv>
The held-out predictions file (18,533 rows) is outside git; every other input is in results/m5.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

M5 = Path(__file__).resolve().parents[2]
FIGURES = M5 / "report/figures"
NONSNAKE_TEST = M5 / "test/predictions_nonsnake_test.csv"
ROBUSTNESS_TABLE = M5 / "robustness/results/robustness_table.csv"

FROZEN_CUT = 0.20484977503483762
FALSE_SNAKE_MARK, MISSED_SNAKE_MARK = 0.05, 0.01
SAFETY_MARK = 0.95
RECALL_AXIS = (0.55, 1.0)
CUT_RANGE = (0.01, 0.6)
DPI = 200

SURFACE, INK, INK_SOFT, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
CLEAN_ROW = "clean (harness rerun, jayeshbansal497)"
DAMAGE_LABELS = {
    "blur_s1": "Blur 0.5%", "blur_s2": "Blur 1%", "blur_s3": "Blur 2%",
    "noise_s1": "Noise sd 10", "noise_s2": "Noise sd 25", "noise_s3": "Noise sd 50",
    "jpeg_q30": "JPEG q30", "jpeg_q10": "JPEG q10",
    "dark_x0.6": "Darker x0.6", "dark_x0.3": "Darker x0.3",
    "bright_x1.5": "Brighter x1.5", "bright_x2.0": "Brighter x2.0",
    "rotate_15": "Rotate 15°", "rotate_30": "Rotate 30°",
    "occlude_10pct": "10% covered", "occlude_25pct": "25% covered",
    "lowres_x0.5": "Half resolution", "lowres_x0.25": "Quarter resolution",
}


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def snake_check_tradeoff(heldout_csv):
    snakes = pd.read_csv(heldout_csv, usecols=["p_snake"])["p_snake"].to_numpy()
    nonsnakes = pd.read_csv(NONSNAKE_TEST, usecols=["p_snake"])["p_snake"].to_numpy()
    cuts = np.linspace(*CUT_RANGE, 600)
    missed = [(snakes < c).mean() for c in cuts]
    passed = [(nonsnakes >= c).mean() for c in cuts]
    at_cut = (snakes < FROZEN_CUT).mean(), (nonsnakes >= FROZEN_CUT).mean()

    fig, ax = plt.subplots(figsize=(7.2, 3.6), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.plot(cuts, np.array(passed) * 100, color=ORANGE, linewidth=2,
            label="Non-snakes called a snake (600 photos)")
    ax.plot(cuts, np.array(missed) * 100, color=BLUE, linewidth=2,
            label="Snakes dropped (18,533 photos)")
    for mark, color in ((FALSE_SNAKE_MARK, ORANGE), (MISSED_SNAKE_MARK, BLUE)):
        ax.axhline(mark * 100, color=color, linewidth=1, linestyle=(0, (4, 3)))
    ax.axvline(FROZEN_CUT, color=INK_SOFT, linewidth=1)
    ax.text(FROZEN_CUT + 0.006, 11.5, f"frozen cut {FROZEN_CUT:.3f}\n(fitted on calibration)",
            fontsize=7.5, color=INK_SOFT, va="top")
    ax.text(CUT_RANGE[1], FALSE_SNAKE_MARK * 100 + 0.4, "G4 mark: at most 5%", fontsize=7.5,
            color=INK_SOFT, ha="right")
    ax.text(CUT_RANGE[1], MISSED_SNAKE_MARK * 100 - 0.75, "G4 mark: at most 1% dropped",
            fontsize=7.5, color=INK_SOFT, ha="right")
    ax.plot([FROZEN_CUT] * 2, [at_cut[1] * 100, at_cut[0] * 100], "o", markersize=6,
            color=INK, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    ax.annotate(f"{at_cut[1]:.1%}", (FROZEN_CUT, at_cut[1] * 100), xytext=(-34, 4),
                textcoords="offset points", fontsize=8, color=INK)
    ax.annotate(f"{at_cut[0]:.2%}", (FROZEN_CUT, at_cut[0] * 100), xytext=(-38, 4),
                textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlim(*CUT_RANGE)
    ax.set_ylim(0, 18)
    ax.set_xlabel("Cut-off on P(snake)", fontsize=8.5, color=INK_SOFT)
    ax.set_ylabel("Error rate on test (%)", fontsize=8.5, color=INK_SOFT)
    ax.legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "snake_check_tradeoff.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return at_cut


def stress_tests():
    table = pd.read_csv(ROBUSTNESS_TABLE)
    clean = table[table["condition"] == CLEAN_ROW].iloc[0]
    damaged = table[table["condition"].isin(DAMAGE_LABELS)].copy()
    damaged["label"] = damaged["condition"].map(DAMAGE_LABELS)
    damaged = damaged.sort_values("top1")
    rows = np.arange(len(damaged))

    fig, (left, right) = plt.subplots(1, 2, figsize=(8.0, 5.0), sharey=True, facecolor=SURFACE)
    for ax in (left, right):
        style(ax)
    left.barh(rows, damaged["top1"], height=0.62, color=BLUE)
    left.set_xlim(0, 1)
    left.set_title("Top-1 accuracy", fontsize=9, color=INK, loc="left")
    right.plot(damaged["who_recall"], rows, "o", markersize=7, color=BLUE,
               markeredgecolor=SURFACE, markeredgewidth=1.5)
    right.set_xlim(*RECALL_AXIS)
    right.set_title("WHO recall behind the M4 gate (axis from 0.55)", fontsize=9, color=INK,
                    loc="left")
    right.axvline(SAFETY_MARK, color=ORANGE, linewidth=1, linestyle=(0, (4, 3)))
    right.text(SAFETY_MARK + 0.004, -1.1, "0.95 mark", fontsize=7.5, color=INK_SOFT)
    for ax, reference in ((left, clean["top1"]), (right, clean["who_recall"])):
        ax.axvline(reference, color=INK, linewidth=1)
        ax.text(reference - 0.004, len(rows) - 0.2, f"clean {reference:.3f}", fontsize=7.5,
                color=INK, ha="right", va="bottom")
    left.set_yticks(rows, damaged["label"], fontsize=8, color=INK)
    left.set_ylim(-1.4, len(rows))
    fig.tight_layout()
    fig.savefig(FIGURES / "stress_tests.png", dpi=DPI, facecolor=SURFACE)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    dropped, passed = snake_check_tradeoff(Path(sys.argv[1]))
    print(f"at the frozen cut: {passed:.4f} of non-snakes pass, {dropped:.4f} of snakes dropped")
    stress_tests()


if __name__ == "__main__":
    main()
