"""Page readout export: the L1 + L2 ridge readout of each listener, as numbers the browser can apply.

Per listener (sub-001..005) writes data/web/sub-XXX/readout.json, readout_boot.f32 (+ .gz.b64.txt).

Page features, in this order (matching readout.json "groups"):
    x = [L1 (66, "ear") | L2 acoustic (R) | L2 structural (R)]      no L0 (spec 7.3, option c)
Split: readout.split(DEAM groups from levels.npz, SEED), then songs without a rating are dropped.

Coefficient-space convention (the page applies it exactly as written):
    xk = x[keep]                              keep = int indices of non-constant training columns
    z  = (xk - scaler_mean) / scaler_std      training-split statistics, kept columns only
    y_hat = z @ coef[rating] + intercept[rating]
so coef is the Ridge coefficient on standardized features (not in raw feature units).

readout_boot.f32: float32 little-endian, shape (n_boot, 2, n_kept + 1); axis 1 is
(valence, arousal) as in readout.json "ratings"; the last entry on the last axis is the intercept.
Each row is a refit (fit_readout, alpha re-chosen by CV) on a bootstrap resample of the TRAINING
songs drawn BY GROUP (groups with replacement, all their songs). The refit has its own scaler, so
it is re-expressed in the main scaler's space: with refit mean m_b, std s_b, coefficient c_b,
    coef_j = c_b,j * s_j / s_b,j ;  intercept = b_b + sum_j c_b,j * (m_j - m_b,j) / s_b,j
(columns constant in the resample get coef 0). Applying a row with the formula above reproduces
that refit's predictions exactly, so the page can recompute contributions per bootstrap (spec 7.3).
"""

import json

import numpy as np

from src import features as F
from src.deam import load_ratings
from src.export_web import save
from src.perception import levels, readout
from src.perception.levels import L1_NAMES, SUBS, l2, load_bundle

FREEZE = F.DATA.parent / "docs" / "superpowers" / "specs" / "2026-11-02-perception-freeze.md"
RATINGS = ("valence", "arousal")
N_BOOT = 200
CONFIG = dict(azimuth=0, distance=1, preset="studio", gain_db=0)


def page_features(L1, xbar, bundle):
    parts = l2(xbar, bundle)
    return np.hstack([L1, parts["acoustic"], parts["structural"]])


def _in_main_space(model_b, mean, std, keep):
    """Express a bootstrap refit in the main scaler/keep space. Returns (coef, intercept)."""
    sc, rd = model_b.named_steps["scale"], model_b.named_steps["ridge"]
    pos = {j: i for i, j in enumerate(np.flatnonzero(model_b.keep_))}
    coef = np.zeros(len(keep))
    icpt = float(rd.intercept_)
    for k, j in enumerate(keep):
        if j in pos:
            i = pos[j]
            coef[k] = rd.coef_[i] * std[k] / sc.scale_[i]
            icpt += rd.coef_[i] * (mean[k] - sc.mean_[i]) / sc.scale_[i]
    return coef, icpt


def bootstrap(X_tr, Y_tr, g_tr, mean, std, keep, n_boot, seed):
    """(n_boot, 2, n_kept + 1) refits on group-resampled training songs."""
    ug = np.unique(g_tr)
    rows_of = {g: np.flatnonzero(g_tr == g) for g in ug}
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(RATINGS), len(keep) + 1))
    for b in range(n_boot):
        drawn = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([rows_of[g] for g in drawn])
        for r in range(len(RATINGS)):
            m = readout.fit_readout(X_tr[idx], Y_tr[idx, r], g_tr[idx])
            c, i0 = _in_main_space(m, mean, std, keep)
            out[b, r, :-1], out[b, r, -1] = c, i0
    return out


def export_listener(sub, lv, ratings, bundle, out_dir, n_boot=N_BOOT):
    """lv: dict-like with song_id, group, L1, xbar. ratings: DataFrame(song_id, valence, arousal)."""
    sid = np.asarray(lv["song_id"])
    groups = np.asarray(lv["group"])
    tr, te = readout.split(groups, readout.SEED)
    X = page_features(np.asarray(lv["L1"], float), np.asarray(lv["xbar"], float), bundle)
    rt = ratings.drop_duplicates("song_id").set_index("song_id")
    has = np.isin(sid, rt.index.values)
    tr, te = tr[has[tr]], te[has[te]]
    Y = np.full((len(sid), 2), np.nan)
    Y[has] = rt.loc[sid[has], list(RATINGS)].to_numpy(float)
    models = [readout.fit_readout(X[tr], Y[tr, r], groups[tr]) for r in range(2)]
    keep = np.flatnonzero(models[0].keep_)  # variance mask depends on X_train only
    sc = models[0].named_steps["scale"]
    mean, std = sc.mean_, sc.scale_
    names = bundle["region_names"]
    held = {k: readout.heldout(m, X[te], Y[te, r]) for r, (k, m) in enumerate(zip(RATINGS, models))}
    doc = dict(
        groups=dict(ear=list(L1_NAMES), acoustic=list(names), structural=list(names)),
        ratings=list(RATINGS),
        scaler_mean=mean.tolist(), scaler_std=std.tolist(), keep=keep.tolist(),
        coef={k: m.named_steps["ridge"].coef_.tolist() for k, m in zip(RATINGS, models)},
        intercept={k: float(m.named_steps["ridge"].intercept_) for k, m in zip(RATINGS, models)},
        alpha={k: float(m.named_steps["ridge"].alpha) for k, m in zip(RATINGS, models)},
        heldout={k: dict(r2=h["r2"], r=h["r"]) for k, h in held.items()},
        n_train=int(len(tr)), n_test=int(len(te)),
        train_min=X[tr].min(0).tolist(), train_max=X[tr].max(0).tolist(),
        config=CONFIG, n_boot=int(n_boot))
    boot = bootstrap(X[tr], Y[tr], groups[tr], mean, std, keep, n_boot, [readout.SEED, int(sub)])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "readout.json").write_text(json.dumps(doc))
    save(out_dir, "readout_boot.f32", boot.astype("<f4"))
    return doc


def main(argv=None):
    z = np.load(levels.OUT / "levels.npz")
    lv = {k: z[k] for k in ("song_id", "group", "L1", "xbar")}
    ratings = load_ratings(FREEZE)
    for sub in SUBS:
        doc = export_listener(sub, lv, ratings, load_bundle(sub), levels.WEB / f"sub-{sub}")
        print(sub, "n_train", doc["n_train"], "n_test", doc["n_test"],
              {k: round(v["r2"], 3) for k, v in doc["heldout"].items()})


if __name__ == "__main__":
    main()
