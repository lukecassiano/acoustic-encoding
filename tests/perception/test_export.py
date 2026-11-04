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
    fp = {"w_region_sha256": "w", "bands_sha256": "b"}
    export.export_listener("001", lv, ratings, bundle, tmp_path, n_boot=n_boot, fingerprint=fp)
    doc = json.loads((tmp_path / "readout.json").read_text())
    assert doc["bundle_fingerprint"] == fp
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
    for rating in export.RATINGS:  # held-out rows too
        for i in te[np.isin(te, rated.index.values)][:5]:
            assert abs(apply(doc, X[i], rating) - readout.fit_readout(
                X[tr], rated.loc[tr, rating].to_numpy(), lv["group"][tr]).predict(X[i:i + 1])[0]) < 1e-5
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


def test_export_deterministic_and_train_groups_only(tmp_path, monkeypatch):
    lv, ratings, bundle, X = fixture(n=60, seed=2)
    seen = []
    real = readout.fit_readout

    def spy(Xa, y, g):
        seen.append(set(np.asarray(g)))
        return real(Xa, y, g)

    monkeypatch.setattr(readout, "fit_readout", spy)
    export.export_listener("002", lv, ratings, bundle, tmp_path / "a", n_boot=3)
    export.export_listener("002", lv, ratings, bundle, tmp_path / "b", n_boot=3)
    assert (tmp_path / "a" / "readout_boot.f32").read_bytes() == (tmp_path / "b" / "readout_boot.f32").read_bytes()
    tr, _ = readout.split(lv["group"], readout.SEED)
    rated = set(ratings.song_id)
    train_groups = {lv["group"][i] for i in tr if i in rated}
    assert seen and all(s <= train_groups for s in seen)


def test_zero_variance_in_resample_is_dropped():
    lv, ratings, bundle, X = fixture(n=150, seed=3)
    tr, _ = readout.split(lv["group"], readout.SEED)
    tr = np.array([i for i in tr if i != 7])
    g = lv["group"]
    ug = np.unique(g[tr])
    rare = set(ug[:2])  # a rare binary column confined to two training groups
    L1 = lv["L1"].copy()
    L1[:, 5] = np.isin(g, list(rare)).astype(float)
    X = export.page_features(L1, lv["xbar"], bundle)
    y = ratings.set_index("song_id").loc[tr, "valence"].to_numpy()
    m = readout.fit_readout(X[tr], y, g[tr])
    assert m.keep_[5]
    sc, keep = m.named_steps["scale"], np.flatnonzero(m.keep_)
    rows_of = {u: np.flatnonzero(g[tr] == u) for u in ug}
    for seed in range(200):
        drawn = np.random.default_rng(seed).choice(ug, size=len(ug), replace=True)
        if not rare & set(drawn):
            break
    else:
        raise AssertionError("no resample drops the rare groups")
    idx = np.concatenate([rows_of[u] for u in drawn])
    mb = readout.fit_readout(X[tr][idx], y[idx], g[tr][idx])
    assert not mb.keep_[5]
    c, i0 = export._in_main_space(mb, sc.mean_, sc.scale_, keep)
    assert c[list(keep).index(5)] == 0.0
    z = (X[:, keep] - sc.mean_) / sc.scale_
    assert np.allclose(z @ c + i0, mb.predict(X), atol=1e-8)


def test_export_checks_fingerprints_against_results(tmp_path):
    import pytest
    from src.perception.levels import file_sha256
    lv = tmp_path / "levels.npz"
    lv.write_bytes(b"levels")
    fps = {"001": {"w_region_sha256": "w1", "bands_sha256": "b"}, "002": {"w_region_sha256": "w2", "bands_sha256": "b"}}
    res = tmp_path / "results.json"
    export.check_against_results(fps, lv, res)  # no results.json yet: warn only
    res.write_text(json.dumps({"meta": {"seed": 0}}))
    with pytest.raises(RuntimeError, match="no bundle fingerprints"):
        export.check_against_results(fps, lv, res)
    meta = {"bundle_fingerprints": fps, "levels_sha256": file_sha256(lv)}
    res.write_text(json.dumps({"meta": meta}))
    export.check_against_results(fps, lv, res)  # matching: passes
    with pytest.raises(RuntimeError, match="sub-002"):
        export.check_against_results({**fps, "002": {"w_region_sha256": "other", "bands_sha256": "b"}}, lv, res)
    lv.write_bytes(b"rebuilt levels")
    with pytest.raises(RuntimeError, match="differs from the levels file"):
        export.check_against_results(fps, lv, res)
