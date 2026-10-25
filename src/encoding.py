"""Voxelwise encoding model: stimulus features -> BOLD. Fit once, on the dataset.

Stages
  1  acoustic band alone (ridge)
  2  acoustic + structural banded ridge (one penalty per band), plus the two
     single-band fits needed to partition held-out variance:
         unique_structural = R2(joint) - R2(acoustic)
         unique_acoustic   = R2(joint) - R2(structural)
         shared            = R2(acoustic) + R2(structural) - R2(joint)

Design matrix, per run: 410-row feature timeline -> z-score with training
statistics -> stack delayed copies (1.5-7.5 s) -> drop the repeated first clip.
Delaying before dropping lets the first clip's features reach the TRs after it.
Test rows are averaged over the four presentations of each clip, matching the
repeat-averaged response they are scored against.

Nuisance: every clip boundary carries the same 2 s fade-out/fade-in and cut, so
it drives a generic response no feature distinguishes clips by. Delayed copies
of a clip-onset impulse train (plus an intercept) are regressed out of the
features and the responses with training-set betas -- an unpenalized nuisance
band, without a third penalty for the banded-ridge search.

Usage:
  python -m src.encoding --sub 001 --stage 2
  python -m src.encoding --sub 001 --stage 1 --source zenodo   # authors' cortical derivatives
"""

import argparse

import numpy as np
import pandas as pd

from . import features as F

DATA = F.DATA
TRS_PER_CLIP = 10
DELAYS = (1, 2, 3, 4, 5)  # TRs: 1.5-7.5 s, as in Nakai et al. 2021
ALPHAS = np.logspace(0, 8, 17)  # wide: most voxels are weakly driven and want heavy shrinkage
N_TARGETS_BATCH = 20000  # voxels solved at once; bounds memory


def run_numbers(sub, task):
    func = F.BIDS / f"sub-{sub}" / "func"
    return sorted(int(p.name.split("run-")[1][:2])
                  for p in func.glob(f"sub-{sub}_task-{task}_run-*_events.tsv"))


def test_blocks(sub):
    """Test clips in order of first appearance, and the row offset of each
    presentation in the stacked test runs (repeated first clips dropped)."""
    ids = np.concatenate([F.run_events(sub, "Test", r).clip_id.values[1:]
                          for r in run_numbers(sub, "Test")])
    order = list(dict.fromkeys(ids))
    return order, {c: np.flatnonzero(ids == c) * TRS_PER_CLIP for c in order}


# -------------------------------------------------------------------- design

def delayed(x, delays=DELAYS):
    """Stack copies of x shifted later by each delay, so row t holds x[t - d]."""
    n, f = x.shape
    out = np.zeros((n, f * len(delays)), dtype=x.dtype)
    for i, d in enumerate(delays):
        out[d:, i * f:(i + 1) * f] = x[:-d]
    return out


def nuisance_design(sub):
    """Clip-boundary regressors N_train (4800, 6) and N_test (600, 6): the onset
    impulse train at each delay, plus an intercept, laid out like the design."""
    def rows(task):
        n_runs = len(run_numbers(sub, task))
        x = np.zeros((len(F.run_events(sub, task, 1)) * TRS_PER_CLIP, 1), np.float32)
        x[::TRS_PER_CLIP] = 1
        return np.tile(delayed(x)[TRS_PER_CLIP:], (n_runs, 1))

    order, _ = test_blocks(sub)
    n_tr, n_te = rows("Training"), rows("Test")[:len(order) * TRS_PER_CLIP]  # identical per clip
    return tuple(np.column_stack([n, np.ones(len(n), np.float32)]) for n in (n_tr, n_te))


def residualize(A, N_train, N_test, A_test=None, chunk=10000):
    """Remove the nuisance fit on training rows: A -= N_train @ beta (in place),
    and the same beta's prediction from A_test if given. Returns beta."""
    pinv = np.linalg.pinv(N_train)
    beta = np.empty((N_train.shape[1], A.shape[1]), np.float32)
    for i in range(0, A.shape[1], chunk):
        beta[:, i:i + chunk] = pinv @ A[:, i:i + chunk]
        A[:, i:i + chunk] -= N_train @ beta[:, i:i + chunk]
        if A_test is not None:
            A_test[..., i:i + chunk] -= N_test @ beta[:, i:i + chunk]
    return beta


def build_design(sub, cache):
    """Delayed, training-standardized design matrices, clip-boundary nuisance removed.

    Returns X_train {band: (4800, F * 5)}, X_test {band: (600, F * 5)}, the
    training run of every row (CV folds) and per band the feature (mean, std,
    nuisance beta) -- everything needed to put new audio on the same footing.
    """
    timelines = {task: [F.run_features(sub, task, r, cache) for r in run_numbers(sub, task)]
                 for task in ("Training", "Test")}
    N_train, N_test = nuisance_design(sub)
    stats = {}
    for band in F.BANDS:
        x = np.concatenate([t[band] for t in timelines["Training"]])
        stats[band] = (x.mean(0), x.std(0))

    def rows(task, band):
        mu, sd = stats[band][:2]
        return np.concatenate([delayed(((t[band] - mu) / sd).astype(np.float32))[TRS_PER_CLIP:]
                               for t in timelines[task]])

    order, blocks = test_blocks(sub)
    X_train, X_test = {}, {}
    for band in F.BANDS:
        X_train[band] = rows("Training", band)
        x = rows("Test", band)
        X_test[band] = np.concatenate([np.mean([x[s:s + TRS_PER_CLIP] for s in blocks[c]], axis=0)
                                       for c in order])
        stats[band] += (residualize(X_train[band], N_train, N_test, X_test[band]),)
    n_rows = len(X_train[F.BANDS[0]]) // len(timelines["Training"])
    folds = np.repeat(np.arange(len(timelines["Training"])), n_rows)
    return X_train, X_test, folds, stats


# ----------------------------------------------------------------- responses

def load_responses(sub, source="preproc"):
    """Y_train (4800, V) and the test repeats (4, 600, V) in test_blocks order."""
    if source == "zenodo":
        z = DATA / "zenodo_8275363"
        train = np.load(z / f"sub-{sub}_Resp_Training.npy", mmap_mode="r").T
        reps = np.load(z / f"sub-{sub}_Resp_Test.npy", mmap_mode="r").transpose(1, 2, 0)
        return np.ascontiguousarray(train), np.ascontiguousarray(reps)
    d = DATA / "derivatives" / "preproc" / f"sub-{sub}"
    test = np.load(d / "Resp_Test.npy", mmap_mode="r")
    order, blocks = test_blocks(sub)
    n_rep = len(blocks[order[0]])
    reps = np.stack([np.concatenate([test[blocks[c][k]: blocks[c][k] + TRS_PER_CLIP] for c in order])
                     for k in range(n_rep)])
    return np.load(d / "Resp_Training.npy"), reps


def noise_ceiling(reps):
    """Highest r a model can reach against the repeat-averaged response
    (Schoppe et al. 2016).

    Signal power SP = (var(sum of repeats) - sum of repeat variances) / (N (N - 1));
    ceiling = sqrt(SP / var(mean of repeats)). NaN where SP <= 0, i.e. nothing
    in the voxel repeats across presentations.
    """
    n = reps.shape[0]
    sp = (reps.sum(0).var(0, ddof=1) - reps.var(1, ddof=1).sum(0)) / (n * (n - 1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.sqrt(np.where(sp > 0, sp, np.nan) / reps.mean(0).var(0, ddof=1))


# ----------------------------------------------------------------------- fit

def band_weights(n_bands):
    """Candidate splits of the penalty across bands, searched per voxel.

    Two bands leave one free ratio, so an even grid on its log-odds covers it;
    more bands fall back to himalaya's Dirichlet random search (n draws).
    """
    if n_bands != 2:
        return 30
    share = 1 / (1 + np.exp(-np.linspace(-4, 4, 9)))
    return np.column_stack([share, 1 - share]).astype(np.float32)


def fit(X_train, Y_train, X_test, folds, bands):
    """Ridge for one band, banded ridge for several; leave-one-run-out CV.

    Returns the fitted himalaya model and its predictions for X_test.
    """
    from himalaya.ridge import GroupRidgeCV, RidgeCV
    from sklearn.model_selection import PredefinedSplit

    cv = PredefinedSplit(folds)
    Xtr = np.hstack([X_train[b] for b in bands])
    Xte = np.hstack([X_test[b] for b in bands])
    if len(bands) == 1:
        model = RidgeCV(alphas=ALPHAS, cv=cv,
                        solver_params=dict(n_targets_batch=N_TARGETS_BATCH))
    else:
        groups = np.concatenate([np.full(X_train[b].shape[1], i) for i, b in enumerate(bands)])
        model = GroupRidgeCV(groups=groups, cv=cv, random_state=0,
                             solver_params=dict(n_iter=band_weights(len(bands)), alphas=ALPHAS,
                                                n_targets_batch=N_TARGETS_BATCH,
                                                progress_bar=False))
    model.fit(Xtr, Y_train)
    return model, np.asarray(model.predict(Xte))


def score(Y, pred):
    from himalaya.scoring import correlation_score, r2_score
    return np.asarray(correlation_score(Y, pred)), np.asarray(r2_score(Y, pred))


def clip_specific(x):
    """Remove the profile every clip shares (mean at each TR position), leaving
    what distinguishes one clip from another."""
    clips = x.reshape(-1, TRS_PER_CLIP, x.shape[1])
    return (clips - clips.mean(0)).reshape(x.shape)


def summarize(name, r, ceiling):
    ok = ceiling > 0.3
    print(f"{name:11s} r: median {np.nanmedian(r):.3f}, 99th pct {np.nanpercentile(r, 99):.3f}, "
          f"{int((r > 0.2).sum())} voxels > 0.2 | r/ceiling median "
          f"{np.nanmedian(r[ok] / ceiling[ok]):.2f} over {int(ok.sum())} voxels with ceiling > 0.3",
          flush=True)


def write_maps(sub, out, res):
    """Per-voxel results -> NIfTI volumes on the native EPI grid."""
    import nibabel as nib
    d = DATA / "derivatives" / "preproc" / f"sub-{sub}"
    ref = nib.load(d / "epi_ref.nii.gz")
    ijk = pd.read_csv(d / "voxels.csv")[["i", "j", "k"]].values
    for name, v in res.items():
        vol = np.zeros(ref.shape, dtype=np.float32)
        vol[tuple(ijk.T)] = np.nan_to_num(v)
        nib.save(nib.Nifti1Image(vol, ref.affine), out / f"{name}.nii.gz")


def run(sub, stage, source):
    cache = dict(np.load(F.CACHE / "clips.npz"))
    X_train, X_test, folds, stats = build_design(sub, cache)
    Y_train, reps = load_responses(sub, source)
    Y_train, reps = Y_train.astype(np.float32, copy=False), reps.astype(np.float32, copy=False)
    N_train, N_test = nuisance_design(sub)
    nuisance_beta = residualize(Y_train, N_train, N_test, reps)
    Y_test = reps.mean(0)
    res = {"ceiling": noise_ceiling(reps)}
    del reps
    print(f"{source}: {Y_train.shape[1]} voxels, "
          f"{int((res['ceiling'] > 0.3).sum())} with ceiling > 0.3", flush=True)

    out = DATA / "derivatives" / "encoding" / f"sub-{sub}" / source
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "nuisance_beta.npy", nuisance_beta)
    models = [("acoustic",)] if stage == 1 else [("acoustic",), ("structural",), tuple(F.BANDS)]
    for bands in models:
        name = "joint" if len(bands) > 1 else bands[0]
        model, pred = fit(X_train, Y_train, X_test, folds, bands)
        res[f"r_{name}"], res[f"r2_{name}"] = score(Y_test, pred)
        np.save(out / f"weights_{name}.npy", np.asarray(model.coef_, dtype=np.float32))
        summarize(name, res[f"r_{name}"], res["ceiling"])
        res[f"r_specific_{name}"], _ = score(clip_specific(Y_test), clip_specific(pred))
        print(f"            {int((res[f'r_specific_{name}'] > 0.2).sum())} voxels > 0.2 "
              f"with the clip-uniform profile removed", flush=True)
        if len(bands) == 1:
            res[f"alpha_{name}"] = np.asarray(model.best_alphas_)
            print(f"            {np.mean(res[f'alpha_{name}'] >= ALPHAS[-1] * 0.99):.0%} of voxels "
                  f"at the largest alpha", flush=True)
        del model, pred

    if stage == 2:
        res["unique_acoustic"] = res["r2_joint"] - res["r2_structural"]
        res["unique_structural"] = res["r2_joint"] - res["r2_acoustic"]
        res["shared"] = res["r2_acoustic"] + res["r2_structural"] - res["r2_joint"]
    np.savez(out / f"stage{stage}.npz", **res, delays=np.array(DELAYS),
             **{f"feat_{k}_{b}": v for b in F.BANDS for k, v in zip(("mean", "std", "nuisance"), stats[b])})
    if source == "preproc":
        write_maps(sub, out, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=2)
    ap.add_argument("--source", choices=["preproc", "zenodo"], default="preproc")
    a = ap.parse_args()
    run(a.sub, a.stage, a.source)


if __name__ == "__main__":
    main()
