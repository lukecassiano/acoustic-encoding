import json

import numpy as np
import pandas as pd

from src.perception import export, readout
from src.perception.levels import L1_NAMES


def fixture(n=90, R=4, seed=0):
    rng = np.random.default_rng(seed)
    L1 = rng.normal(size=(n, len(L1_NAMES)))
    L1[:, 3] = 1.0  # a constant column, to exercise keep
    xbar = rng.normal(size=(n, 340))
    W = rng.normal(size=(340, R)) * 0.1
    bundle = dict(W_region=W, region_names=[f"r{i}" for i in range(R)],
                  acoustic_cols=slice(0, 300), structural_cols=slice(300, 340))
    X = export.page_features(L1, xbar, bundle)
    y = X[:, 0] - 0.5 * X[:, 70] + 0.3 * rng.normal(size=n)
    ratings = pd.DataFrame(dict(song_id=np.arange(n), valence=y, arousal=-y + rng.normal(size=n)))
    ratings = ratings[ratings.song_id != 7]  # one unrated song is dropped
    lv = dict(song_id=np.arange(n), group=np.array([f"a{i // 3}" for i in range(n)]), L1=L1, xbar=xbar)
    return lv, ratings, bundle, X


def apply(doc, x, rating):
    z = (x[doc["keep"]] - np.array(doc["scaler_mean"])) / np.array(doc["scaler_std"])
    return z @ np.array(doc["coef"][rating]) + doc["intercept"][rating]


def test_export_roundtrip(tmp_path):
    lv, ratings, bundle, X = fixture()
    n_boot = 5
    export.export_listener("001", lv, ratings, bundle, tmp_path, n_boot=n_boot)
    doc = json.loads((tmp_path / "readout.json").read_text())
    boot = np.fromfile(tmp_path / "readout_boot.f32", dtype="<f4")
    assert (tmp_path / "readout_boot.f32.gz.b64.txt").exists()
    k = len(doc["keep"])
    assert 3 not in doc["keep"] and k == X.shape[1] - 1
    boot = boot.reshape(n_boot, 2, k + 1)
    assert doc["n_boot"] == n_boot and doc["config"]["preset"] == "studio"
    assert doc["n_train"] + doc["n_test"] == len(ratings)
    assert len(doc["train_min"]) == X.shape[1]

    # refit independently, then recompute predictions from the exported numbers alone
    tr, te = readout.split(lv["group"], readout.SEED)
    rated = ratings.set_index("song_id")
    tr = np.array([i for i in tr if i in rated.index])
    for rating in export.RATINGS:
        y = rated.loc[tr, rating].to_numpy()
        model = readout.fit_readout(X[tr], y, lv["group"][tr])
        for i in tr[:5]:
            assert abs(apply(doc, X[i], rating) - model.predict(X[i:i + 1])[0]) < 1e-5
    zb = (X[0][doc["keep"]] - np.array(doc["scaler_mean"])) / np.array(doc["scaler_std"])
    assert np.isfinite(boot[:, 0, :-1] @ zb + boot[:, 0, -1]).all()


def test_bootstrap_row_reproduces_refit():
    lv, ratings, bundle, X = fixture(seed=1)
    tr, _ = readout.split(lv["group"], readout.SEED)
    tr = np.array([i for i in tr if i != 7])
    y = ratings.set_index("song_id").loc[tr, "valence"].to_numpy()
    m = readout.fit_readout(X[tr], y, lv["group"][tr])
    sc = m.named_steps["scale"]
    keep = np.flatnonzero(m.keep_)
    idx = np.random.default_rng(0).choice(len(tr), len(tr))
    mb = readout.fit_readout(X[tr][idx], y[idx], lv["group"][tr][idx])
    c, i0 = export._in_main_space(mb, sc.mean_, sc.scale_, keep)
    z = (X[:, keep] - sc.mean_) / sc.scale_
    assert np.allclose(z @ c + i0, mb.predict(X), atol=1e-8)
