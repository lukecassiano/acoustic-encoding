"""Analysis B: remove voxels in the order a disease removes them and track how
much of the model's musical representation is left.

Nothing is refitted. Voxels are predicted independently, so ablation is an
accounting exercise: at each step, what share of each band's held-out explained
variance sits in voxels not yet removed? Representation is counted only in
voxels the joint model predicts significantly (src.validate).

Orderings are staged by region. Voxels within a stage go in random order, whose
expectation is a straight line across the stage, so curves are exact rather
than sampled.

Controls
  random voxels    retention falls as 1 - fraction ablated; area under curve 0.5
  shuffled regions the same regions removed in random sequence: is the disease
                   order special among spatially contiguous orders?

Writes data/derivatives/ablation/sub-XXX/: curves.csv, stages.csv, summary.csv

Usage:
  python -m src.ablation --sub 001
"""

import argparse

import numpy as np
import pandas as pd

from . import encoding as E
from .validate import PARTS, paths, region_labels

MEASURES = ["r2_joint"] + PARTS
FILL_MM = 6.0  # cortical voxels with no atlas label take the nearest label within this distance
VOXEL_MM = 2.0

# Braak stages as tau-PET regions of interest, Pascoal et al. 2020 (Brain 143:2818),
# mapped onto Harvard-Oxford labels. Calls the source does not settle:
#   - temporal pole with stage IV (as in the ADNI composites)
#   - Heschl's gyrus with stage VI, the other primary cortices; the planum
#     regions stay with superior temporal in stage V
#   - supplementary motor with superior frontal (V); opercular cortex with V
BRAAK = {
    1: ["Parahippocampal Gyrus, anterior division"],  # transentorhinal / entorhinal
    2: ["Hippocampus"],
    3: ["Amygdala", "Parahippocampal Gyrus, posterior division", "Temporal Fusiform Cortex, anterior division",
        "Temporal Fusiform Cortex, posterior division", "Temporal Occipital Fusiform Cortex",
        "Occipital Fusiform Gyrus", "Lingual Gyrus"],
    4: ["Insular Cortex", "Inferior Temporal Gyrus, anterior division", "Inferior Temporal Gyrus, posterior division",
        "Inferior Temporal Gyrus, temporooccipital part", "Middle Temporal Gyrus, anterior division",
        "Middle Temporal Gyrus, posterior division", "Middle Temporal Gyrus, temporooccipital part",
        "Cingulate Gyrus, posterior division", "Angular Gyrus", "Temporal Pole"],
    5: ["Frontal Orbital Cortex", "Superior Temporal Gyrus, anterior division",
        "Superior Temporal Gyrus, posterior division", "Planum Polare", "Planum Temporale",
        "Inferior Frontal Gyrus, pars triangularis", "Inferior Frontal Gyrus, pars opercularis",
        "Frontal Opercular Cortex", "Central Opercular Cortex", "Parietal Opercular Cortex", "Cuneal Cortex",
        "Cingulate Gyrus, anterior division", "Paracingulate Gyrus", "Subcallosal Cortex",
        "Supramarginal Gyrus, anterior division", "Supramarginal Gyrus, posterior division",
        "Lateral Occipital Cortex, superior division", "Lateral Occipital Cortex, inferior division",
        "Precuneous Cortex", "Superior Parietal Lobule", "Superior Frontal Gyrus",
        "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)", "Frontal Medial Cortex",
        "Frontal Pole", "Middle Frontal Gyrus"],
    6: ["Precentral Gyrus", "Postcentral Gyrus", "Intracalcarine Cortex", "Supracalcarine Cortex",
        "Occipital Pole", "Heschl's Gyrus (includes H1 and H2)"],
}

ORDERINGS = {"alzheimers": BRAAK}


def voxel_table(sub):
    """Voxels with a region (atlas gaps filled from nearby labels) and the
    representation each one carries."""
    from scipy.spatial import cKDTree
    d, out = paths(sub)
    vox = pd.read_csv(d / "voxels.csv")
    region = region_labels(vox).astype(object)
    xyz = vox[["i", "j", "k"]].values * VOXEL_MM
    known = region != "Background"
    dist, nearest = cKDTree(xyz[known]).query(xyz[~known], distance_upper_bound=FILL_MM)
    filled = np.full((~known).sum(), "Unlabelled", dtype=object)
    filled[np.isfinite(dist)] = region[known][nearest[np.isfinite(dist)]]
    region[~known] = filled
    vox["region"] = region

    res, sig = np.load(out / "stage2.npz"), np.load(out / "stats.npz")["sig_joint"]
    for m in MEASURES:
        vox[m] = np.where(sig, np.clip(res[m], 0, None), 0.0)
    return vox


def stage_of(vox, ordering):
    """Stage per voxel; regions the ordering does not name go last."""
    lookup = {region: stage for stage, regions in ordering.items() for region in regions}
    return vox.region.map(lookup).fillna(max(ordering) + 1).astype(int).values


def totals(groups, sequence, weights):
    """Voxel count and summed measures of each group in the sequence."""
    n = np.array([(groups == g).sum() for g in sequence], dtype=float)
    return n, np.stack([weights[groups == g].sum(0) for g in sequence])


def knots(n, w):
    """Cumulative (fraction of voxels removed, fraction of each measure retained)
    after each group, given the groups' totals in removal order."""
    x = np.concatenate([[0.0], np.cumsum(n) / n.sum()])
    y = np.vstack([np.zeros(w.shape[1]), np.cumsum(w, 0) / w.sum(0)])
    return x, 1 - y


def area(x, retained):
    """Area under the retention curve: 0.5 for random order, towards 1 when
    representation sits in what is removed last."""
    return ((retained[1:] + retained[:-1]) / 2 * np.diff(x)[:, None]).sum(0)


def half_loss(x, retained):
    """Fraction of the brain removed when half of each measure is gone."""
    return np.array([np.interp(0.5, 1 - retained[:, j], x) for j in range(retained.shape[1])])


def run(sub, n_null):
    vox = voxel_table(sub)
    weights = vox[MEASURES].values
    rng = np.random.default_rng(0)
    out = E.DATA / "derivatives" / "ablation" / f"sub-{sub}"
    out.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, 201)

    n_region, w_region = totals(vox.region.values, sorted(vox.region.unique()), weights)
    shuffles = [rng.permutation(len(n_region)) for _ in range(n_null)]
    null = np.stack([area(*knots(n_region[s], w_region[s])) for s in shuffles])

    curves, stages, summary = [], [], []
    for name, ordering in ORDERINGS.items():
        stage = stage_of(vox, ordering)
        sequence = np.arange(1, max(ordering) + 2)
        x, retained = knots(*totals(stage, sequence, weights))
        curve = pd.DataFrame({m: np.interp(grid, x, retained[:, j]) for j, m in enumerate(MEASURES)})
        curves.append(curve.assign(ordering=name, fraction_ablated=grid))
        for s, x0, x1, r0, r1 in zip(sequence, x[:-1], x[1:], retained[:-1], retained[1:]):
            stages.append({"ordering": name, "stage": "unstaged" if s > max(ordering) else s,
                           "voxels": int((stage == s).sum()), "ablated_after": x1,
                           **{m: r0[j] - r1[j] for j, m in enumerate(MEASURES)}})
        a, h = area(x, retained), half_loss(x, retained)
        for j, m in enumerate(MEASURES):
            summary.append({"ordering": name, "measure": m, "area": a[j], "half_loss_at": h[j],
                            "null_area_mean": null[:, j].mean(),
                            "p_later_than_shuffled": (1 + (null[:, j] >= a[j]).sum()) / (n_null + 1),
                            "p_earlier_than_shuffled": (1 + (null[:, j] <= a[j]).sum()) / (n_null + 1)})

    pd.concat(curves).to_csv(out / "curves.csv", index=False)
    stages, summary = pd.DataFrame(stages), pd.DataFrame(summary)
    stages.to_csv(out / "stages.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)

    n_sig = int((vox.r2_joint > 0).sum())
    print(f"{n_sig} voxels carry representation; {int((vox.region[vox.r2_joint > 0] == 'Unlabelled').sum())} "
          f"of them have no region within {FILL_MM:.0f} mm")
    pd.set_option("display.width", 200)
    for name in ORDERINGS:
        print(f"\n{name}: share of each measure removed at each stage")
        print(stages[stages.ordering == name].drop(columns="ordering").round(3).to_string(index=False))
    print("\nsummary (area under retention curve; random voxel order = 0.5)")
    print(summary.round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    ap.add_argument("--n-null", type=int, default=5000)
    a = ap.parse_args()
    run(a.sub, a.n_null)


if __name__ == "__main__":
    main()
