"""End-to-end test of the pre-registered run on synthetic data only.

Real ratings, real levels files and the real web bundles are never touched: the DEAM and PMEmo
roots are redirected to tmp_path, the freeze is a synthetic file, and load_bundle is replaced.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import deam, pmemo
from src.deam import FreezeError
from src.perception import run as R
from src.perception.levels import L0_NAMES, L1_NAMES, SUBS
from src.perception.readout import SEED, split

VERDICTS = {"supported", "not supported", "inconclusive"}
N_COLS = 340


def _levels(path, n, n_groups, seed, id0):
    rng = np.random.default_rng(seed)
    sid = np.arange(id0, id0 + n)
    group = np.array([f"artist {i % n_groups}" for i in range(n)])
    L0 = rng.standard_normal((n, len(L0_NAMES)))
    L1 = rng.standard_normal((n, len(L1_NAMES)))
    xbar = rng.standard_normal((n, N_COLS))
    np.savez(path, song_id=sid, group=group, L0=L0, L1=L1, xbar=xbar)
    return sid, L0, L1


def _planted(L0, L1, rng):
    arousal = 1.5 * L1[:, :4].sum(1) + 0.5 * rng.standard_normal(len(L1)) + 5
    valence = L0[:, L0_NAMES.index("mode")] + 0.8 * rng.standard_normal(len(L0)) + 5
    return valence, arousal


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    # DEAM: 120 songs in the levels file, 2 without a rating row, 1 with a NaN valence
    sid, L0, L1 = _levels(tmp_path / "levels.npz", 120, 40, 1, 2)
    v, a = _planted(L0, L1, rng)
    v[2] = np.nan
    d = tmp_path / "deam"
    f = d / deam.STATIC_REL[0]
    f.parent.mkdir(parents=True)
    pd.DataFrame({"song_id": sid[2:], " valence_mean": v[2:], " arousal_mean": a[2:]}).to_csv(f, index=False)
    monkeypatch.setattr(deam, "DEAM", d)
    # PMEmo: 50 songs, 1 without a rating, same planted mapping
    psid, P0, P1 = _levels(tmp_path / "pmemo_levels.npz", 50, 50, 2, 1000)
    pv, pa = _planted(P0, P1, rng)
    p = tmp_path / "pmemo"
    pf = p / pmemo.STATIC_REL
    pf.parent.mkdir(parents=True)
    pd.DataFrame({"musicId": psid[1:], "Arousal(mean)": pa[1:], "Valence(mean)": pv[1:]}).to_csv(pf, index=False)
    monkeypatch.setattr(pmemo, "PMEMO", p)
    # synthetic bundles, R differs per listener
    def bundle(sub):
        r = 2 + SUBS.index(sub) % 2
        W = np.random.default_rng(100 + SUBS.index(sub)).standard_normal((N_COLS, r))
        return dict(stats=None, W_region=W, region_names=[f"reg{i}" for i in range(r)],
                    acoustic_cols=slice(0, 300), structural_cols=slice(300, 340))
    monkeypatch.setattr(R, "load_bundle", bundle)
    monkeypatch.setattr(R, "bundle_fingerprint", lambda sub: {"w_region_sha256": f"w{sub}", "bands_sha256": "b"})
    freeze = tmp_path / "freeze.md"
    freeze.write_text("# synthetic freeze\n\nGo-ahead: Luke, 2026-11-03\n", encoding="utf-8")
    return tmp_path, freeze


def _run(tmp, freeze):
    return R.run(freeze, n_boot=200, n_perm=99, n_proj=3, workers=1,
                 levels_path=tmp / "levels.npz", pmemo_path=tmp / "pmemo_levels.npz", out_dir=tmp / "out")


def test_run_end_to_end_on_fixture(fixture):
    tmp, freeze = fixture
    res = _run(tmp, freeze)
    assert res["verdicts"]["H1"] == "supported"
    assert set(res["verdicts"]) == {"H1", "H2_L0", "H2_L2", "H3_valence", "H3_arousal", "H4_valence", "H4_arousal"}
    assert all(v in VERDICTS for v in res["verdicts"].values())
    # every verdict anywhere in the tree is one of the three strings
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "verdict":
                    assert v in VERDICTS
                walk(v)
    walk(res)
    # counts, split and shapes
    dd, pd_ = res["data"]["deam"], res["data"]["pmemo"]
    assert dd["dropped_no_rating_row"] == 2 and dd["dropped_nan_rating"] == 1 and dd["dropped_no_rating"] == 3
    assert dd["n_rated"] == 117 and dd["n_train"] + dd["n_test"] == 117
    assert not set(dd["train_song_ids"]) & set(dd["test_song_ids"])
    # same split order as Task 10 (export.py): split ALL songs in levels.npz, then drop unrated
    z = np.load(tmp / "levels.npz")
    tr, te = split(z["group"], SEED)
    rated = np.isin(z["song_id"], z["song_id"][3:])
    assert dd["train_song_ids"] == [int(v) for v in z["song_id"][tr[rated[tr]]]]
    assert dd["test_song_ids"] == [int(v) for v in z["song_id"][te[rated[te]]]]
    assert pd_["dropped_no_rating"] == 1 and pd_["dropped_no_rating_row"] == 1 and pd_["dropped_nan_rating"] == 0
    assert pd_["n_rated"] == 49
    assert res["models"]["L0_acoustic"]["valence"]["dims"] == 61
    assert res["models"]["L0_structural"]["valence"]["dims"] == 8
    for s in SUBS:
        r = res["meta"]["R"][s]
        assert res["models"]["page"][s]["arousal"]["dims"] == len(L1_NAMES) + 2 * r
        assert res["models"]["L2"][s]["arousal"]["dims"] == 2 * r
        assert len(res["hypotheses"]["H3"]["arousal"]["listeners"][s]["null"]) == 3
    # H4: planted L1 -> arousal transfers to the synthetic PMEmo set through the page readout
    assert res["verdicts"]["H4_arousal"] == "supported"
    assert set(res["pmemo_models"]) == {"L0", "L0+L1", "theory", "L2", "page"}
    _assert_consistent(res)
    # outputs on disk; strict json (no NaN / Infinity) round-trips
    def no_const(c):
        raise ValueError(f"non-finite constant {c} in results.json")
    on_disk = json.loads((tmp / "out" / "results.json").read_text(encoding="utf-8"), parse_constant=no_const)
    assert on_disk["verdicts"] == res["verdicts"]
    md = (tmp / "out" / "results.md").read_text(encoding="utf-8")
    assert "## Verdicts" in md and "| H1 | supported |" in md
    assert "R2(a) | R2(b) | alpha(a) | alpha(b)" in md and "DEAM R2 (page)" in md
    # provenance: freeze path as given plus its sha256, input file hashes, every listener's bundle
    from src.perception.levels import file_sha256
    meta = on_disk["meta"]
    assert meta["freeze"] == str(freeze) and meta["freeze_sha256"] == file_sha256(freeze)
    assert meta["levels_sha256"] == file_sha256(tmp / "levels.npz")
    assert meta["pmemo_levels_sha256"] == file_sha256(tmp / "pmemo_levels.npz")
    assert meta["bundle_fingerprints"] == {s: {"w_region_sha256": f"w{s}", "bands_sha256": "b"} for s in SUBS}
    assert meta["levels_provenance"] is None and meta["pmemo_levels_provenance"] is None  # no sidecars here
    assert "git_commit" in meta


def _assert_consistent(res):
    """Cross-checks between the hypothesis blocks and the model records they are built from."""
    M, H = res["models"], res["hypotheses"]
    eq = lambda a, b: a == pytest.approx(b, abs=1e-12)  # noqa: E731
    h1 = H["H1"]
    assert eq(h1["diff"], M["L0+L1"]["arousal"]["r2"] - M["L0"]["arousal"]["r2"])
    def h2(x, get):
        assert eq(x["valence"]["diff"], get("structural", "valence") - get("acoustic", "valence"))
        assert eq(x["arousal"]["diff"], get("acoustic", "arousal") - get("structural", "arousal"))
        assert x["verdict"] == R.h2_combine(x["valence"]["verdict"], x["arousal"]["verdict"])
    h2(H["H2"]["L0"], lambda p, k: M[f"L0_{p}"][k]["r2"])
    for s, x in H["H2"]["L2"]["listeners"].items():
        h2(x, lambda p, k, s=s: M[f"L2_{p}"][s][k]["r2"])
    for k, x in H["H3"].items():
        for s, v in x["listeners"].items():
            assert v["r2"] == M["L2"][s][k]["r2"]
            assert v["null_p95"] == float(np.percentile(v["null"], 95))
            assert v["pass"] == (v["r2"] > v["null_p95"])
        assert x["n_pass"] == sum(v["pass"] for v in x["listeners"].values())
    for k, x in H["H4"].items():
        for s, v in x["listeners"].items():
            assert v == res["pmemo_models"]["page"][s][k]
        assert x["n_lo_pos"] == sum(v["lo"] > 0 for v in x["listeners"].values())
        assert x["n_hi_neg"] == sum(v["hi"] < 0 for v in x["listeners"].values())


def test_pmemo_isolation(fixture):
    """Perturbing PMEmo levels and ratings changes nothing fitted or scored on DEAM."""
    tmp, freeze = fixture
    base = _run(tmp, freeze)
    rng = np.random.default_rng(9)
    z = dict(np.load(tmp / "pmemo_levels.npz"))
    for k in ("L0", "L1", "xbar"):
        z[k] = z[k] + rng.standard_normal(z[k].shape)
    np.savez(tmp / "pmemo_levels.npz", **z)
    pf = pmemo.PMEMO / pmemo.STATIC_REL
    df = pd.read_csv(pf)
    df[["Arousal(mean)", "Valence(mean)"]] = rng.standard_normal((len(df), 2))
    df.to_csv(pf, index=False)
    pert = _run(tmp, freeze)
    assert pert["pmemo_models"] != base["pmemo_models"]  # the perturbation took effect
    assert pert["models"] == base["models"]
    assert pert["data"]["deam"] == base["data"]["deam"]
    for h in ("H1", "H2", "H3"):
        assert pert["hypotheses"][h] == base["hypotheses"][h]


def test_inputs_checked_before_ratings(fixture):
    tmp, freeze = fixture
    freeze.write_text("# unsigned: a ratings read would raise FreezeError\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="--dataset pmemo"):
        R.run(freeze, levels_path=tmp / "levels.npz", pmemo_path=tmp / "missing.npz", out_dir=tmp / "out")


def test_merge_ratings_counts():
    ratings = pd.DataFrame({"song_id": [1, 2, 3], "valence": [5.0, np.nan, 4.0], "arousal": [3.0, 2.0, np.nan]})
    m = R.merge_ratings(np.array([1, 2, 3, 4]), ratings)
    assert m["rated"].tolist() == [True, False, False, False]
    assert (m["dropped_no_rating_row"], m["dropped_nan_rating"], m["dropped_no_rating"]) == (1, 2, 3)
    assert m["valence"][0] == 5.0 and np.isnan(m["valence"][3])


def test_export_shares_run_definitions():
    from src.perception import export
    assert export.page_features is R.page_features
    assert export.merge_ratings is R.merge_ratings and export.deam_split is R.deam_split
    from src.perception import figures, levels
    assert export.RATINGS is R.RATINGS and figures.RATINGS is R.RATINGS
    assert export.CONFIG is levels.CONFIG
    assert figures.alpha_at_edge is R.alpha_at_edge


def test_alpha_edge_both_ends():
    from src.perception.readout import ALPHAS
    assert R.alpha_at_edge(float(ALPHAS[0])) and R.alpha_at_edge(float(ALPHAS[-1]))
    assert not R.alpha_at_edge(float(ALPHAS[1])) and not R.alpha_at_edge(None)
    assert R._is_edge({"alpha": float(ALPHAS[0])}, ALPHAS) and not R._is_edge({"alpha": None}, ALPHAS)


def _old_boot_r2(y, pred, n, seed):  # the pre-refactor implementation, verbatim
    rng = np.random.default_rng(seed)
    m = len(y)
    v = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        v[i] = R._r2(y[idx], pred[idx])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def _old_boot_r(y, pred, n, seed):
    rng = np.random.default_rng(seed)
    m = len(y)
    v = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        v[i] = R._corr(y[idx], pred[idx])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return dict(r=R._corr(y, pred), lo=float(lo), hi=float(hi))


def test_boot_helpers_bit_identical():
    rng = np.random.default_rng(4)
    for m in (30, 101):
        y = rng.standard_normal(m)
        pred = 0.5 * y + rng.standard_normal(m)
        assert R.boot_r2(y, pred, n=300) == _old_boot_r2(y, pred, 300, SEED)
        assert R.boot_r(y, pred, n=300) == _old_boot_r(y, pred, 300, SEED)


def test_run_refuses_without_goahead(fixture):
    tmp, freeze = fixture
    freeze.write_text("# synthetic freeze, unsigned\n", encoding="utf-8")
    with pytest.raises(FreezeError):
        _run(tmp, freeze)


def test_page_feature_order():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((N_COLS, 3))
    b = dict(W_region=W, acoustic_cols=slice(0, 300), structural_cols=slice(300, 340))
    L1, xb = rng.standard_normal((4, len(L1_NAMES))), rng.standard_normal((4, N_COLS))
    X = R.page_features(L1, xb, b)
    assert X.shape == (4, len(L1_NAMES) + 6)
    assert np.allclose(X[:, :66], L1)
    assert np.allclose(X[:, 66:69], xb[:, :300] @ W[:300])
    assert np.allclose(X[:, 69:], xb[:, 300:] @ W[300:])
    assert np.allclose(R.l2_features(xb, b), X[:, 66:])


def test_aggregation_rules():
    assert R.h2_combine("supported", "supported") == "supported"
    assert R.h2_combine("supported", "inconclusive") == "inconclusive"
    assert R.h2_combine("not supported", "supported") == "not supported"
    assert R.majority(4, 1) == "supported" and R.majority(1, 4) == "not supported"
    assert R.majority(3, 2) == "inconclusive" and R.majority(3, 0) == "inconclusive"


def test_h3_pool_matches_serial():
    rng = np.random.default_rng(3)
    n = 60
    xb = rng.standard_normal((n, N_COLS))
    y = xb[:, :5].sum(1) + rng.standard_normal(n)
    g = np.repeat(np.arange(20), 3)
    tr, te = R.deam_split(g, np.ones(n, bool))
    W = rng.standard_normal((N_COLS, 2))
    jobs = [((s, "arousal"), xb[tr], xb[te], y[tr], y[te], g[tr], W, 2) for s in ("001", "002")]
    serial = dict(map(R._h3_job, jobs))
    from multiprocessing import Pool
    with Pool(2) as pool:
        par = dict(pool.map(R._h3_job, jobs))
    for k in serial:
        assert np.array_equal(serial[k], par[k])
