"""Paper figures from the saved Stage 1-2 outputs. Nothing is fitted here.

  stage1_accuracy_map      acoustic model accuracy on the subject's anatomy
  stage1_validation        accuracy vs noise ceiling, by region, and by delay
  stage2_partition_map     unique acoustic and unique structural variance
  stage2_partition_rois    the same partition per region, with intervals
  stage2_analysis_b        signal available in the regions Analysis B needs
  stage3_survival          retention as voxels go in Braak-stage order (after src.ablation)

Needs stage2.npz (src.encoding) and stats.npz / regions.csv / delays.csv
(src.validate). regions.csv is the table behind every regional panel.

Usage:
  python -m src.figures --sub 001      -> figures/sub-001/*.png
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from . import features as F
from .validate import PARTS, paths

FIGURES = F.ROOT / "figures"

SURFACE, INK, SECONDARY, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # acoustic, structural: fixed wherever a band is shown
BAND_COLORS = {"unique_acoustic": BLUE, "unique_structural": ORANGE, "shared": MUTED}
BAND_NAMES = {"unique_acoustic": "Unique to acoustic", "unique_structural": "Unique to structural",
              "shared": "Shared"}
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
ORANGES = LinearSegmentedColormap.from_list("oranges", ["#fbe0d3", "#f5b49a", "#eb6834", "#b9471b", "#7a2f10"])

SHORT = {
    "Heschl's Gyrus (includes H1 and H2)": "Heschl's gyrus",
    "Planum Temporale": "Planum temporale",
    "Planum Polare": "Planum polare",
    "Superior Temporal Gyrus, posterior division": "STG, posterior",
    "Superior Temporal Gyrus, anterior division": "STG, anterior",
    "Temporal Pole": "Temporal pole",
    "Middle Temporal Gyrus, posterior division": "MTG, posterior",
    "Middle Temporal Gyrus, temporooccipital part": "MTG, temporo-occipital",
    "Supramarginal Gyrus, posterior division": "Supramarginal, posterior",
    "Parietal Opercular Cortex": "Parietal operculum",
    "Central Opercular Cortex": "Central operculum",
    "Precentral Gyrus": "Precentral gyrus",
    "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)": "Supplementary motor",
    "Insular Cortex": "Insula",
    "Cingulate Gyrus, anterior division": "Anterior cingulate",
    "Cingulate Gyrus, posterior division": "Posterior cingulate",
    "Precuneous Cortex": "Precuneus",
    "Frontal Medial Cortex": "Medial frontal",
    "Occipital Pole": "Occipital pole",
    "Frontal Pole": "Frontal pole",
}
AUDITORY = list(SHORT)[:5]  # core outward: the order the gradient is read in
PARTITION_ROWS = list(SHORT)[:12]
CONTROLS = ["Occipital Pole", "Frontal Pole", "Precuneous Cortex", "Hippocampus"]
ANALYSIS_B = list(SHORT)[12:18] + ["Hippocampus", "Amygdala", "Accumbens"]


def short(region):
    return SHORT.get(region, region)


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.xaxis.label.set_color(SECONDARY)
    ax.yaxis.label.set_color(SECONDARY)
    ax.grid(axis=grid, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def title(ax, text):
    ax.set_title(text, loc="left", fontsize=9.5, color=INK, pad=8)


def save(fig, name):
    folder = FIGURES / save.sub
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / f"{name}.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"figures/{save.sub}/{name}.png")


# ---------------------------------------------------------------- brain maps

class Anatomy:
    """The subject's T1 on the EPI grid, reoriented so arrays index as RAS."""

    def __init__(self, sub):
        import nibabel as nib
        d, _ = paths(sub)
        img = nib.load(d / "epi_t1.nii.gz")
        self.ornt = nib.orientations.io_orientation(img.affine)
        self.shape = img.shape
        self.ijk = pd.read_csv(d / "voxels.csv")[["i", "j", "k"]].values
        t1 = self.ras(img.get_fdata())
        self.t1 = np.where(t1 > 0, t1 / np.percentile(t1[t1 > 0], 99), np.nan)
        inside = np.argwhere(t1 > 0)
        self.box = [slice(max(lo - 2, 0), hi + 3) for lo, hi in zip(inside.min(0), inside.max(0))]

    def ras(self, vol):
        import nibabel as nib
        return nib.orientations.apply_orientation(vol, self.ornt)

    def volume(self, values, mask):
        vol = np.full(self.shape, np.nan, np.float32)
        vol[tuple(self.ijk[mask].T)] = values[mask]
        return self.ras(vol)

    def axial_cuts(self, mask, n=6):
        """Evenly spaced axial slices through the bulk of the masked voxels."""
        z = np.argwhere(~np.isnan(self.volume(np.ones(len(mask), np.float32), mask)))[:, 2]
        return np.unique(np.percentile(z, np.linspace(8, 92, n)).astype(int))


def brain_rows(anat, rows, cuts, heading):
    """One row of axial slices per (label, volume, colormap, vmax)."""
    fig, axes = plt.subplots(len(rows), len(cuts), figsize=(1.45 * len(cuts) + 0.9, 1.75 * len(rows) + 0.5),
                             squeeze=False, facecolor=SURFACE)
    x, y = anat.box[0], anat.box[1]
    for r, (label, vol, cmap, vmax) in enumerate(rows):
        for c, k in enumerate(cuts):
            ax = axes[r, c]
            ax.imshow(anat.t1[x, y, k].T, origin="lower", cmap="gray", vmin=-1.1, vmax=1.25,
                      interpolation="bilinear")
            im = ax.imshow(np.clip(vol[x, y, k], 0, None).T, origin="lower", cmap=cmap, vmin=0, vmax=vmax,
                           interpolation="nearest")
            ax.set_axis_off()
        axes[r, 0].text(-0.04, 0.5, "L", transform=axes[r, 0].transAxes, fontsize=7, color=MUTED,
                        va="center", ha="right")
        axes[r, -1].text(1.04, 0.5, "R", transform=axes[r, -1].transAxes, fontsize=7, color=MUTED,
                         va="center")
        bar = fig.colorbar(im, ax=axes[r, :], fraction=0.018, pad=0.015, aspect=14)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
        bar.set_ticks([0, vmax])
        bar.set_label(label, fontsize=8, color=SECONDARY)
    fig.suptitle(heading, x=0.02, ha="left", fontsize=9.5, color=INK)
    fig.text(0.02, 0.01, "Axial slices, inferior to superior. Native EPI grid on the subject's T1.",
             fontsize=7, color=MUTED)
    return fig


def accuracy_map(anat, res, stats):
    sig = stats["sig_acoustic"]
    rows = [("held-out r", anat.volume(res["r_acoustic"], sig), BLUES, 0.5)]
    fig = brain_rows(anat, rows, anat.axial_cuts(sig),
                     f"Acoustic model accuracy, {int(sig.sum()):,} voxels significant at FDR q < 0.05")
    save(fig, "stage1_accuracy_map")


def partition_map(anat, res, stats):
    sig = stats["sig_joint"]
    rows = [("unique acoustic R²", anat.volume(res["unique_acoustic"], sig), BLUES, 0.10),
            ("unique structural R²", anat.volume(res["unique_structural"], sig), ORANGES, 0.10)]
    fig = brain_rows(anat, rows, anat.axial_cuts(sig),
                     "Held-out variance each band explains alone, in voxels the joint model predicts")
    save(fig, "stage2_partition_map")


# -------------------------------------------------------------------- charts

def validation(res, regions, delays):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.3), facecolor=SURFACE,
                                  gridspec_kw=dict(width_ratios=[1.05, 1.0, 0.8], wspace=0.75))

    ok = np.isfinite(res["ceiling"])
    hb = a.hexbin(res["ceiling"][ok], res["r_acoustic"][ok], gridsize=55, bins="log", cmap=BLUES,
                  extent=(0, 1, -0.25, 0.75), linewidths=0)
    a.plot([0, 0.75], [0, 0.75], color=SECONDARY, linewidth=1)
    a.text(0.62, 0.68, "r = ceiling", fontsize=7.5, color=SECONDARY, ha="right")
    a.set(xlim=(0, 1), ylim=(-0.25, 0.75), xlabel="noise ceiling", ylabel="held-out r")
    style(a, grid="both")
    bar = fig.colorbar(hb, ax=a, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(labelsize=7, colors=MUTED, length=0)
    bar.set_label("voxels", fontsize=8, color=SECONDARY)
    title(a, f"Accuracy against the ceiling, {int(ok.sum()):,} voxels")

    rows = [r for r in AUDITORY + CONTROLS if r in regions.index][::-1]
    b.barh([short(r) for r in rows], regions.loc[rows, "r_acoustic"], height=0.55,
           color=[BLUE if r in AUDITORY else MUTED for r in rows])
    b.axvline(0, color=AXIS, linewidth=0.8)
    b.set(xlabel="mean held-out r, all voxels in region")
    style(b)
    title(b, "Auditory regions (blue) against controls (gray)")

    c.plot(delays.delay_s, delays.share, color=BLUE, linewidth=2, marker="o", markersize=5,
           markeredgecolor=SURFACE, markeredgewidth=1.5)
    c.axhline(1 / len(delays), color=AXIS, linewidth=0.8)
    c.text(delays.delay_s.iloc[0], 1 / len(delays) + 0.003, "uniform", fontsize=7.5, color=MUTED,
           va="bottom")
    c.set(xlabel="delay after stimulus (s)", ylabel="share of weight energy",
          xticks=delays.delay_s, ylim=(0.1, 0.3))
    style(c, grid="y")
    title(c, "Weights peak at 4.5 s")
    save(fig, "stage1_validation")


def partition_rois(regions):
    rows = [r for r in PARTITION_ROWS if r in regions.index and np.isfinite(regions.loc[r, PARTS[0]])][::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.3), facecolor=SURFACE)
    for offset, part in zip((0.24, 0.0, -0.24), ("unique_acoustic", "unique_structural", "shared")):
        y = np.arange(len(rows)) + offset
        v = regions.loc[rows, part]
        ax.hlines(y, regions.loc[rows, f"{part}_lo"], regions.loc[rows, f"{part}_hi"],
                  color=BAND_COLORS[part], linewidth=1.5, alpha=0.55)
        ax.plot(v, y, "o", color=BAND_COLORS[part], markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=BAND_NAMES[part])
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([f"{short(r)}  ({int(regions.loc[r, 'n_sig_joint'])})" for r in rows])
    ax.set(xlabel="held-out variance explained (R²), mean over significant voxels")
    style(ax)
    ax.tick_params(axis="y", labelcolor=SECONDARY)
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="lower right", handletextpad=0.2)
    title(ax, "Variance partition by region")
    fig.text(0.02, -0.02, "Significant voxels per region in brackets. Lines are 95% intervals from "
             "resampling the 60 test clips.", fontsize=7, color=MUTED)
    save(fig, "stage2_partition_rois")


def analysis_b(regions):
    rows = [r for r in ["Heschl's Gyrus (includes H1 and H2)", "Superior Temporal Gyrus, posterior division"]
            + ANALYSIS_B if r in regions.index][::-1]
    colors = [MUTED if r in AUDITORY else BLUE for r in rows]
    share = 100 * regions.loc[rows, "n_sig_joint"] / regions.loc[rows, "n"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.6, 0.36 * len(rows) + 1.2), sharey=True, facecolor=SURFACE,
                               gridspec_kw=dict(wspace=0.08))
    a.barh([short(r) for r in rows], regions.loc[rows, "ceiling"], height=0.55, color=colors)
    a.set(xlabel="mean noise ceiling")
    style(a)
    a.tick_params(axis="y", labelcolor=SECONDARY)
    title(a, "Repeatable signal")
    b.barh([short(r) for r in rows], share, height=0.55, color=colors)
    for y, (r, s) in enumerate(zip(rows, share)):
        b.text(s + 1.2, y, f"{int(regions.loc[r, 'n_sig_joint'])} of {int(regions.loc[r, 'n']):,}",
               fontsize=7.5, color=SECONDARY, va="center")
    b.set(xlabel="voxels the joint model predicts (%, FDR q < 0.05)", xlim=(0, max(share.max() * 1.25, 10)))
    style(b)
    title(b, "Modelled representation")
    fig.suptitle("Regions Analysis B depends on (blue) against auditory cortex (gray)",
                 x=0.02, y=1.02, ha="left", fontsize=9.5, color=INK)
    save(fig, "stage2_analysis_b")


def survival(sub):
    """Retention of each band as voxels are removed in disease order."""
    d = F.DATA / "derivatives" / "ablation" / f"sub-{sub}"
    curves, stages = pd.read_csv(d / "curves.csv"), pd.read_csv(d / "stages.csv")
    lines = [("r2_joint", INK, "All explained variance"), ("unique_acoustic", BLUE, BAND_NAMES["unique_acoustic"]),
             ("unique_structural", ORANGE, BAND_NAMES["unique_structural"])]
    roman = {"1": "I", "2": "II", "3": "III", "4": "IV", "5": "V", "6": "VI"}
    c, s = curves[curves.ordering == "alzheimers"], stages[stages.ordering == "alzheimers"]
    fig, ax = plt.subplots(figsize=(6.2, 3.9), facecolor=SURFACE)
    ax.plot([0, 1], [1, 0], color=AXIS, linewidth=1)
    start = 0.0
    for stage, end in zip(s.stage.astype(str), s.ablated_after):
        if stage != "unstaged":
            ax.axvline(end, color=GRID, linewidth=0.8)
            if end - start > 0.03:
                ax.text((start + end) / 2, 1.04, roman[stage], fontsize=7.5, color=MUTED, ha="center")
        start = end
    for measure, color, label in lines:
        ax.plot(c.fraction_ablated, c[measure], color=color, linewidth=2 if color != INK else 1.2, label=label)
    ax.set(xlim=(0, 1), ylim=(0, 1.1), xlabel="fraction of gray-matter voxels removed",
           ylabel="representation retained")
    style(ax, grid="y")
    title(ax, "Removal in Alzheimer's order (Braak stages)")
    ax.text(0.60, 0.44, "random voxel order", fontsize=7.5, color=MUTED, rotation=-33, ha="center")
    ax.legend(frameon=False, fontsize=8, labelcolor=SECONDARY, loc="lower left")
    fig.text(0.02, -0.06, "Braak stage boundaries are marked along the top; voxels after the last are not staged.\n"
             "Representation is held-out R² in voxels the joint model predicts.", fontsize=7, color=MUTED)
    save(fig, "stage3_survival")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    a = ap.parse_args()
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    save.sub = f"sub-{a.sub}"
    _, out = paths(a.sub)
    res, stats = np.load(out / "stage2.npz"), np.load(out / "stats.npz")
    regions = pd.read_csv(out / "regions.csv", index_col=0)
    delays = pd.read_csv(out / "delays.csv")
    anat = Anatomy(a.sub)
    accuracy_map(anat, res, stats)
    validation(res, regions, delays)
    partition_map(anat, res, stats)
    partition_rois(regions)
    analysis_b(regions)
    if (F.DATA / "derivatives" / "ablation" / f"sub-{a.sub}" / "curves.csv").exists():
        survival(a.sub)


if __name__ == "__main__":
    main()
