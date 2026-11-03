"""Readout and statistics for the affect analysis: group-aware ridge, paired bootstrap,
permutation test, random-projection control, and the theory-driven baseline.

All randomness comes from numpy.random.default_rng(seed). Standardization uses training-split
statistics only (inside every CV fold as well).
"""

import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.perception.levels import L0_NAMES, L1_NAMES

SEED = 20261102

# frozen analysis settings
TEST_SIZE = 0.3
ALPHAS = np.logspace(-2, 5, 30)
N_FOLDS = 5
N_BOOT = 2000
N_PERM = 5000
N_PROJ = 1000


def split(groups, seed=SEED):
    """Group-disjoint train/test indices (GroupShuffleSplit, test_size 0.3)."""
    groups = np.asarray(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(groups)), groups=groups))
    return tr, te


def fit_readout(X_train, y_train, groups_train):
    """VarianceThreshold(0) -> StandardScaler -> Ridge, alpha chosen by 5-fold GroupKFold CV (R2).

    Implemented as GridSearchCV over the whole pipeline with precomputed grouped folds, so the
    scaler is refit on each fold's training part. Returns the refit Pipeline; `.keep_` is the
    boolean mask of non-constant columns of X_train.
    """
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float)
    folds = list(GroupKFold(N_FOLDS).split(X_train, y_train, np.asarray(groups_train)))
    pipe = Pipeline([("vt", VarianceThreshold(0.0)), ("scale", StandardScaler()),
                     ("ridge", Ridge())])
    gs = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=folds, scoring="r2", refit=True)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    model.keep_ = model.named_steps["vt"].get_support()
    return model


def _r2(y, p):
    ss = np.sum((y - y.mean()) ** 2)
    return 1.0 - np.sum((y - p) ** 2) / ss


def _corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a @ a) * (b @ b))
    return float(a @ b / d) if d > 0 else 0.0


def heldout(model, X_test, y_test):
    """Held-out R2, Pearson r and predictions."""
    y = np.asarray(y_test, float)
    pred = model.predict(np.asarray(X_test, float))
    return dict(r2=float(_r2(y, pred)), r=_corr(y, pred), pred=pred)


def boot_diff(y, pred_a, pred_b, n=N_BOOT, seed=SEED):
    """Paired bootstrap over test songs of R2_a - R2_b. diff is the full-set point estimate;
    lo/hi are the 2.5/97.5 percentiles of the resampled differences."""
    y, a, b = (np.asarray(v, float) for v in (y, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    m = len(y)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        d[i] = _r2(y[idx], a[idx]) - _r2(y[idx], b[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return dict(diff=float(_r2(y, a) - _r2(y, b)), lo=float(lo), hi=float(hi))


def perm_p(y, pred, n=N_PERM, seed=SEED):
    """One-sided permutation p on r: permute y against fixed predictions;
    p = (1 + #{r_perm >= r_obs}) / (1 + n)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    r_obs = _corr(y, pred)
    count = sum(_corr(rng.permutation(y), pred) >= r_obs for _ in range(n))
    return (1 + count) / (1 + n)


def _scaled_gaussian(W_block, rng):
    G = rng.standard_normal(W_block.shape)
    gn = np.linalg.norm(G, axis=0)
    return G * (np.linalg.norm(W_block, axis=0) / np.where(gn > 0, gn, 1.0))


def random_projection_r2(xbar_tr, xbar_te, y_tr, y_te, groups_tr, W_region, blocks,
                         n=N_PROJ, seed=SEED):
    """Held-out R2 of readouts on random projections (the null for 'any 340-d -> R map works').

    Per draw and per block, G = standard normal with the block's shape of W_region, each column
    rescaled to the norm of the matching W_region column; features are the concatenation over
    blocks of xbar[:, block] @ G_block. Fit on train (groups_tr gives the CV groups; it comes
    right after y_te), score on test. Returns an (n,) array of R2.
    """
    rng = np.random.default_rng(seed)
    W_region = np.asarray(W_region, float)
    out = np.empty(n)
    for i in range(n):
        Gs = [_scaled_gaussian(W_region[b], rng) for b in blocks]
        F_tr = np.concatenate([xbar_tr[:, b] @ G for b, G in zip(blocks, Gs)], axis=1)
        F_te = np.concatenate([xbar_te[:, b] @ G for b, G in zip(blocks, Gs)], axis=1)
        out[i] = heldout(fit_readout(F_tr, y_tr, groups_tr), F_te, y_te)["r2"]
    return out


def _z(tr, te):
    sd = tr.std()
    return (te - tr.mean()) / (sd if sd > 0 else 1.0), (tr - tr.mean()) / (sd if sd > 0 else 1.0)


def theory_baseline(L0_tr, L1_tr, L0_te, L1_te, y_tr, l0_names=L0_NAMES, l1_names=L1_NAMES):
    """Hand-built affect predictor with no fitted weights beyond a 1-D calibration.

    Signature: (L0_tr, L1_tr, L0_te, L1_te, y_tr) with y_tr a dict {"arousal": (n_tr,),
    "valence": (n_tr,)}. Columns are located by name (phon_mean in L1; onset_rate, tempo, mode,
    key_clarity in L0). Each feature is z-scored with TRAINING mean/sd only.
      arousal score = z(phon_mean) + z(onset_rate) + z(tempo)
      valence score = z(mode) + z(key_clarity)
    Each score is mapped to the rating scale by a least-squares line fit on train, then applied
    to test. Returns {"arousal": (n_te,), "valence": (n_te,)} test predictions.
    """
    def col(L0, L1, name):
        if name in l1_names:
            return np.asarray(L1)[:, l1_names.index(name)]
        return np.asarray(L0)[:, l0_names.index(name)]

    def score(names):
        s_tr = np.zeros(len(L0_tr))
        s_te = np.zeros(len(L0_te))
        for nm in names:
            z_te, z_tr = _z(col(L0_tr, L1_tr, nm).astype(float), col(L0_te, L1_te, nm).astype(float)) \
                if False else (None, None)
            tr = col(L0_tr, L1_tr, nm).astype(float)
            te = col(L0_te, L1_te, nm).astype(float)
            z_te, z_tr = _z(tr, te)
            s_tr += z_tr
            s_te += z_te
        return s_tr, s_te

    out = {}
    for key, names in (("arousal", ["phon_mean", "onset_rate", "tempo"]),
                       ("valence", ["mode", "key_clarity"])):
        s_tr, s_te = score(names)
        y = np.asarray(y_tr[key], float)
        sc = s_tr - s_tr.mean()
        denom = sc @ sc
        slope = (sc @ (y - y.mean()) / denom) if denom > 0 else 0.0
        out[key] = y.mean() + slope * (s_te - s_tr.mean())
    return out


def verdict(lo, hi):
    """Decision from a CI on a pre-registered difference."""
    if lo > 0:
        return "supported"
    if hi < 0:
        return "not supported"
    return "inconclusive"
