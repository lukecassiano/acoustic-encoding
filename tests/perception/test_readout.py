import numpy as np

from src.perception.levels import L0_NAMES, L1_NAMES
from src.perception.readout import (SEED, boot_diff, fit_readout, heldout, perm_p,
                                    random_projection_r2, split, theory_baseline, verdict)

ALPHAS_SMALL = 3


def _data(n=240, p=12, n_groups=40, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), n // n_groups)
    X = rng.standard_normal((n, p))
    w = rng.standard_normal(p)
    y = X @ w + noise * rng.standard_normal(n)
    return X, y, groups


def _split_data(**kw):
    X, y, g = _data(**kw)
    tr, te = split(g)
    return X[tr], X[te], y[tr], y[te], g[tr], g[te]


def test_split_respects_groups():
    _, _, g = _data()
    tr, te = split(g)
    assert len(tr) and len(te)
    assert not set(g[tr]) & set(g[te])
    tr2, te2 = split(g)
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)
    assert SEED == 20261102


def test_readout_recovers_planted_signal():
    Xtr, Xte, ytr, yte, gtr, _ = _split_data()
    m = fit_readout(Xtr, ytr, gtr)
    out = heldout(m, Xte, yte)
    assert out["r2"] > 0.5 and out["r"] > 0.7 and out["pred"].shape == yte.shape


def test_constant_block_no_nan():
    Xtr, Xte, ytr, yte, gtr, _ = _split_data()
    m = fit_readout(np.c_[Xtr, np.ones(len(Xtr))], ytr, gtr)
    out = heldout(m, np.c_[Xte, np.ones(len(Xte))], yte)
    assert np.isfinite(out["r2"]) and np.all(np.isfinite(out["pred"]))
    assert m.keep_.dtype == bool and m.keep_.sum() == Xtr.shape[1] and not m.keep_[-1]


def test_perm_p_null_uniformish():
    rng = np.random.default_rng(3)
    y = rng.standard_normal(100)
    p = perm_p(y, rng.standard_normal(len(y)), n=500, seed=1)
    assert 0.2 < p < 1.0
    assert perm_p(y, y + 0.1 * rng.standard_normal(100), n=500, seed=1) < 0.01


def test_random_projection_shapes_and_norms(monkeypatch):
    rng = np.random.default_rng(5)
    n, d, R = 200, 30, 4
    W = rng.standard_normal((d, R)) * rng.uniform(0.5, 3, R)
    blocks = [slice(0, 20), slice(20, 30)]
    xbar = rng.standard_normal((n, d))
    y = xbar[:, :20] @ W[:20].sum(1) + rng.standard_normal(n)
    g = np.repeat(np.arange(40), 5)
    tr, te = split(g)
    r2 = random_projection_r2(xbar[tr], xbar[te], y[tr], y[te], g[tr], W, blocks, n=5, seed=2)
    assert r2.shape == (5,) and np.all(np.isfinite(r2))
    # column norms of the generated matrix equal W's, per block
    from src.perception import readout
    seen = []
    orig = readout._scaled_gaussian
    monkeypatch.setattr(readout, "_scaled_gaussian",
                        lambda Wb, rng_: seen.append(orig(Wb, rng_)) or seen[-1])
    random_projection_r2(xbar[tr], xbar[te], y[tr], y[te], g[tr], W, blocks, n=2, seed=2)
    assert len(seen) == 4
    for G, Wb in zip(seen, [W[b] for b in blocks] * 2):
        assert G.shape == Wb.shape
        assert np.allclose(np.linalg.norm(G, axis=0), np.linalg.norm(Wb, axis=0))


def test_verdict():
    assert verdict(0.01, 0.2) == "supported"
    assert verdict(-0.1, 0.1) == "inconclusive"
    assert verdict(-0.3, -0.01) == "not supported"


def test_boot_diff_planted_difference():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(150)
    good = y + 0.3 * rng.standard_normal(150)
    bad = rng.standard_normal(150)
    d = boot_diff(y, good, bad, n=300, seed=1)
    assert d["lo"] > 0 and d["lo"] < d["diff"] < d["hi"]
    d0 = boot_diff(y, good, good, n=100, seed=1)
    assert d0["diff"] == 0 and d0["lo"] == 0 and d0["hi"] == 0


def test_theory_baseline_recovers_planted():
    rng = np.random.default_rng(11)
    n = 300
    L0 = rng.standard_normal((n, len(L0_NAMES)))
    L1 = rng.standard_normal((n, len(L1_NAMES)))
    c0 = {k: L0_NAMES.index(k) for k in ("onset_rate", "tempo", "mode", "key_clarity")}
    ph = L1_NAMES.index("phon_mean")
    ar = L1[:, ph] + L0[:, c0["onset_rate"]] + L0[:, c0["tempo"]]
    va = L0[:, c0["mode"]] + L0[:, c0["key_clarity"]]
    y = {"arousal": 2 * ar + 5, "valence": -1.5 * va + 3}
    tr, te = np.arange(200), np.arange(200, n)
    out = theory_baseline(L0[tr], L1[tr], L0[te], L1[te], {k: v[tr] for k, v in y.items()})
    for k in ("arousal", "valence"):
        assert out[k].shape == (len(te),) and np.all(np.isfinite(out[k]))
        assert np.corrcoef(out[k], y[k][te])[0, 1] > 0.99
        assert np.allclose(out[k], y[k][te], atol=0.5)


def test_theory_baseline_no_test_leak():
    rng = np.random.default_rng(12)
    L0 = rng.standard_normal((100, len(L0_NAMES)))
    L1 = rng.standard_normal((100, len(L1_NAMES)))
    y = {"arousal": rng.standard_normal(60), "valence": rng.standard_normal(60)}
    a = theory_baseline(L0[:60], L1[:60], L0[60:70], L1[60:70], y)
    b = theory_baseline(L0[:60], L1[:60], L0[60:], L1[60:], y)
    assert np.allclose(a["arousal"], b["arousal"][:10])
