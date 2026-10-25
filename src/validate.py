"""Checks that the fitted model earns its predictions, for the Stage 2 outputs.

  significance  per-voxel p for held-out r, permuting which clip each predicted
                block belongs to; FDR across voxels. The permutation keeps the
                profile every clip shares, so it tests clip-specific prediction.
  partition     unique / shared variance per region with bootstrap intervals
                (test clips resampled), over voxels the joint model predicts
  delays        how the acoustic weights distribute over hemodynamic delays

Writes next to stage2.npz:
  stats.npz     p_<model>, sig_<model> for acoustic / structural / joint
  regions.csv   per region: ceiling, accuracy, significant voxels, partition
  delays.csv    share of weight energy at each delay

Usage:
  python -m src.validate --sub 001
"""

import argparse

import numpy as np
import pandas as pd

from . import encoding as E
from . import features as F

MODELS = {"acoustic": ("acoustic",), "structural": ("structural",), "joint": tuple(F.BANDS)}
PARTS = ["unique_acoustic", "unique_structural", "shared"]
CHUNK = 10000  # voxels per block of clip-by-clip cross products
MIN_VOXELS = 20  # significant voxels a region needs before its partition is reported


def paths(sub):
    return (E.DATA / "derivatives" / "preproc" / f"sub-{sub}",
            E.DATA / "derivatives" / "encoding" / f"sub-{sub}" / "preproc")


def predictions(sub):
    """Test-set predictions of each fitted model, rebuilt from the saved weights."""
    _, out = paths(sub)
    _, X_test, _, _ = E.build_design(sub, dict(np.load(F.CACHE / "clips.npz")))
    return {name: np.hstack([X_test[b] for b in bands]) @ np.load(out / f"weights_{name}.npy")
            for name, bands in MODELS.items()}


def region_labels(vox):
    """One label per voxel: Harvard-Oxford subcortical structure (hemispheres
    merged) where there is one, otherwise the cortical label."""
    sub = vox.ho_sub.str.replace("^(Left|Right) ", "", regex=True)
    return np.where(vox.subcortical, sub, vox.ho_cort)


# -------------------------------------------------------------- significance

def permutation_p(pred, Y, perms):
    """One-sided p per voxel for corr(pred, Y) under shuffled clip identity."""
    n_clip = perms.shape[1]
    cols = np.arange(n_clip)
    p = np.ones(Y.shape[1])
    for i in range(0, Y.shape[1], CHUNK):
        with np.errstate(invalid="ignore", divide="ignore"):
            zp, zy = (np.nan_to_num((x - x.mean(0)) / x.std(0)).reshape(n_clip, E.TRS_PER_CLIP, -1)
                      for x in (pred[:, i:i + CHUNK], Y[:, i:i + CHUNK]))
        cross = np.einsum("atv,btv->abv", zp, zy, optimize=True)  # predicted clip a vs measured clip b
        observed = cross[cols, cols].sum(0)
        count = np.zeros(cross.shape[2])
        for perm in perms:
            count += cross[perm, cols].sum(0) >= observed
        p[i:i + CHUNK] = (count + 1) / (len(perms) + 1)
    return p


def fdr(p, q=0.05):
    """Benjamini-Hochberg: boolean mask of discoveries at false discovery rate q."""
    order = np.argsort(p)
    passed = np.flatnonzero(p[order] <= q * np.arange(1, len(p) + 1) / len(p))
    sig = np.zeros(len(p), bool)
    if len(passed):
        sig[order[:passed.max() + 1]] = True
    return sig


# ----------------------------------------------------------------- partition

def bootstrap_partition(preds, Y, labels, n_boot, rng):
    """Region means of unique / shared R2 with 95% intervals, resampling test clips.

    R2 under a resample only needs per-clip sums (of y, y^2 and squared error),
    so every bootstrap draw is a 60-vector of clip counts times those sums.
    """
    n_clip = Y.shape[0] // E.TRS_PER_CLIP
    per_clip = lambda x: x.reshape(n_clip, E.TRS_PER_CLIP, -1).sum(1)
    sy, syy = per_clip(Y), per_clip(Y ** 2)
    sse = {m: per_clip((Y - preds[m]) ** 2) for m in MODELS}
    regions = sorted(set(labels))
    member = np.stack([(labels == r) / (labels == r).sum() for r in regions]).T  # (voxels, regions)

    def partition(w):  # w: (draws, clips) clip counts
        total = w @ syy - (w @ sy) ** 2 / Y.shape[0]
        r2 = {m: 1 - (w @ sse[m]) / total for m in MODELS}
        parts = [r2["joint"] - r2["structural"], r2["joint"] - r2["acoustic"],
                 r2["acoustic"] + r2["structural"] - r2["joint"]]
        return np.stack([x @ member for x in parts], axis=-1)  # (draws, regions, parts)

    point = partition(np.ones((1, n_clip)))[0]
    draws = np.stack([np.bincount(rng.integers(0, n_clip, n_clip), minlength=n_clip)
                      for _ in range(n_boot)]).astype(np.float64)
    lo, hi = np.percentile(partition(draws), [2.5, 97.5], axis=0)
    rows = {}
    for i, r in enumerate(regions):
        rows[r] = {f"{part}{suffix}": v[i, j] for j, part in enumerate(PARTS)
                   for suffix, v in (("", point), ("_lo", lo), ("_hi", hi))}
    return pd.DataFrame.from_dict(rows, orient="index")


# ----------------------------------------------------------------------- run

def run(sub, n_perm, n_boot):
    d, out = paths(sub)
    rng = np.random.default_rng(0)
    res = np.load(out / "stage2.npz")
    Y = np.load(d / "Resp_Test_Mean.npy").astype(np.float32)
    Y -= E.nuisance_design(sub)[1] @ np.load(out / "nuisance_beta.npy")  # as in the fit
    preds = predictions(sub)
    vox = pd.read_csv(d / "voxels.csv")
    vox["region"] = region_labels(vox)

    n_clip = Y.shape[0] // E.TRS_PER_CLIP
    perms = np.stack([rng.permutation(n_clip) for _ in range(n_perm)])
    stats = {}
    for m in MODELS:
        stats[f"p_{m}"] = permutation_p(preds[m], Y, perms)
        stats[f"sig_{m}"] = fdr(stats[f"p_{m}"])
        r = res[f"r_{m}"][stats[f"sig_{m}"]]
        print(f"{m:11s} {int(stats[f'sig_{m}'].sum()):6d} voxels significant (FDR q < 0.05)"
              + (f", smallest r among them {r.min():.3f}" if len(r) else ""), flush=True)
    np.savez(out / "stats.npz", **stats)

    vox["ceiling"] = np.nan_to_num(res["ceiling"])
    for m in MODELS:
        vox[f"r_{m}"], vox[f"sig_{m}"] = res[f"r_{m}"], stats[f"sig_{m}"]
    regions = vox.groupby("region").agg(
        n=("ceiling", "size"), ceiling=("ceiling", "mean"),
        **{f"r_{m}": (f"r_{m}", "mean") for m in MODELS},
        **{f"n_sig_{m}": (f"sig_{m}", "sum") for m in MODELS})

    sig = stats["sig_joint"]
    labels = vox.region.values[sig]
    keep = np.isin(labels, regions.index[regions.n_sig_joint >= MIN_VOXELS])
    cols = np.flatnonzero(sig)[keep]
    part = bootstrap_partition({m: preds[m][:, cols] for m in MODELS}, Y[:, cols],
                               labels[keep], n_boot, rng)
    regions.join(part).sort_values("n_sig_joint", ascending=False).to_csv(out / "regions.csv")

    total = sum(res[k][sig].sum() for k in PARTS)
    print("share of joint R2 in significant voxels: "
          + ", ".join(f"{k} {res[k][sig].sum() / total:.0%}" for k in PARTS))

    W = np.load(out / "weights_acoustic.npy", mmap_mode="r")
    w = np.asarray(W[:, np.flatnonzero(stats["sig_acoustic"])]).reshape(len(E.DELAYS), -1)
    energy = (w ** 2).sum(1)
    delays = pd.DataFrame({"delay_s": np.array(E.DELAYS) * F.TR, "share": energy / energy.sum()})
    delays.to_csv(out / "delays.csv", index=False)
    print("acoustic weight energy by delay: "
          + ", ".join(f"{t:.1f} s {s:.0%}" for t, s in zip(delays.delay_s, delays.share)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    ap.add_argument("--n-perm", type=int, default=5000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    run(a.sub, a.n_perm, a.n_boot)


if __name__ == "__main__":
    main()
