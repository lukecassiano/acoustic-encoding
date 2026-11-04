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
    # DEAM: 120 songs in the levels file, 2 of them without a rating
    sid, L0, L1 = _levels(tmp_path / "levels.npz", 120, 40, 1, 2)
    v, a = _planted(L0, L1, rng)
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
    assert dd["dropped_no_rating"] == 2 and dd["n_rated"] == 118 and dd["n_train"] + dd["n_test"] == 118
    assert not set(dd["train_song_ids"]) & set(dd["test_song_ids"])
    # same split order as Task 10 (export.py): split ALL songs in levels.npz, then drop unrated
    z = np.load(tmp / "levels.npz")
    tr, te = split(z["group"], SEED)
    rated = np.isin(z["song_id"], z["song_id"][2:])
    assert dd["train_song_ids"] == [int(v) for v in z["song_id"][tr[rated[tr]]]]
    assert dd["test_song_ids"] == [int(v) for v in z["song_id"][te[rated[te]]]]
    assert pd_["dropped_no_rating"] == 1 and pd_["n_rated"] == 49
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
    # outputs on disk; json round-trips
    on_disk = json.loads((tmp / "out" / "results.json").read_text(encoding="utf-8"))
    assert on_disk["verdicts"] == res["verdicts"]
    md = (tmp / "out" / "results.md").read_text(encoding="utf-8")
    assert "## Verdicts" in md and "| H1 | supported |" in md


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
