"""Figures from a schema-conformant synthetic results.json (no real data is read or written)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.perception import figures as G
from tests.perception.test_run import _run, fixture  # noqa: F401  (pytest fixture)

SUBS = ["001", "002", "003"]
NAMES = ("hierarchy", "dissociation", "projection_null", "psychophysics", "generalisation")


def _rec(rng, alpha=10.0):
    r2 = float(rng.uniform(-0.05, 0.4))
    return dict(r2=r2, r=0.5, r2_lo=r2 - 0.05, r2_hi=r2 + 0.05, perm_p=0.01, alpha=alpha, n_kept=60, dims=61)


def _prec(rng):
    r = float(rng.uniform(-0.1, 0.6))
    return dict(r=r, lo=r - 0.1, hi=r + 0.1)


def _d(rng, verdict):
    return dict(a="x", b="y", diff=0.05, lo=0.01, hi=0.09, verdict=verdict)


def fake_results(seed=0):
    rng = np.random.default_rng(seed)
    both = lambda f: {k: f() for k in G.RATINGS}
    per = lambda f: {s: both(f) for s in SUBS}
    models = {k: both(lambda: _rec(rng)) for k in ("L0", "L0+L1", "L0_acoustic", "L0_structural")}
    models["theory"] = both(lambda: {**_rec(rng), "alpha": None, "n_kept": None})
    models["L0_structural"]["arousal"]["alpha"] = 1e5  # exercises the max-penalty mark
    for k in ("L2", "L2_acoustic", "L2_structural", "page"):
        models[k] = per(lambda: _rec(rng))
    h3 = {k: {"listeners": {s: dict(r2=0.2, null_p5=-0.1, null_p50=0.0, null_p95=0.1, null_mean=0.0,
                                    **{"pass": True}, null=list(rng.normal(0, 0.05, 30))) for s in SUBS},
              "n_pass": 3, "n_fail": 0, "verdict": "supported"} for k in G.RATINGS}
    h4 = {k: {"model": "page", "listeners": {s: _prec(rng) for s in SUBS}, "n_lo_pos": 3, "n_hi_neg": 0,
              "verdict": "inconclusive"} for k in G.RATINGS}
    res = dict(
        meta=dict(listeners=SUBS),
        models=models,
        hypotheses=dict(
            H1=dict(rating="arousal", a="L0+L1", b="L0", diff=0.1, lo=0.0, hi=0.2, verdict="supported"),
            H2=dict(L0={**both(lambda: _d(rng, "supported")), "verdict": "supported"},
                    L2=dict(listeners={s: {**both(lambda: _d(rng, "supported")), "verdict": "not supported"} for s in SUBS},
                            n_support=0, n_fail=3, verdict="not supported")),
            H3=h3, H4=h4),
        pmemo_models={"L0": both(lambda: _prec(rng)), "L0+L1": both(lambda: _prec(rng)),
                      "theory": both(lambda: _prec(rng)), "L2": per(lambda: _prec(rng)),
                      "page": per(lambda: _prec(rng))},
        verdicts=dict(H1="supported", H2_L0="supported", H2_L2="not supported", H3_valence="supported",
                      H3_arousal="supported", H4_valence="inconclusive", H4_arousal="inconclusive"))
    return json.loads(json.dumps(res))


def _recs(res):
    """Records the figures may mark with a dagger, per figure."""
    m, subs = res["models"], res["meta"]["listeners"]
    per = lambda k: [m[k][s][r] for s in subs for r in G.RATINGS]
    top = lambda recs: sum(1 for x in recs if x.get("alpha") is not None and x["alpha"] >= 0.99e5)
    both = lambda k: [m[k][r] for r in G.RATINGS]
    return dict(
        hierarchy=top(both("L0") + both("L0+L1") + both("theory") + per("L2")),
        dissociation=top(both("L0_acoustic") + both("L0_structural") + per("L2_acoustic") + per("L2_structural")),
        generalisation=top(both("L0") + both("L0+L1") + per("L2") + per("page")),
        projection_null=0, psychophysics=0)


def _run_figs(res, tmp_path, monkeypatch):
    rp = tmp_path / "results_in.json"
    rp.write_text(json.dumps(res), encoding="utf-8")
    out = tmp_path / "figs"
    daggers, orig = {}, G._save

    def spy(fig, o, name):  # every axis must carry data artists; count dagger annotations
        for ax in fig.axes:
            if ax.get_visible() and (ax.get_xlabel() or ax.get_title(loc="left")):
                assert ax.lines or ax.patches or ax.collections, (name, ax.get_title(loc="left"))
        daggers[name] = sum(t.get_text() == G.TOP_MARK for ax in fig.axes for t in ax.texts)
        has_note = any(t.get_text() == G.FOOTNOTE for t in fig.texts)
        assert has_note == (daggers[name] > 0), name
        orig(fig, o, name)

    monkeypatch.setattr(G, "_save", spy)
    G.main(["--results", str(rp), "--out", str(out)])
    assert sorted(daggers) == sorted(NAMES)
    for n in NAMES:
        f = out / f"{n}.png"
        assert f.exists() and f.stat().st_size > 5000
    plt.close("all")
    return daggers


def test_all_five_figures_nonblank(tmp_path, monkeypatch):
    res = fake_results()
    for s in SUBS[:2]:  # exercise the max-penalty marks in every figure
        res["models"]["L2"][s]["valence"]["alpha"] = 1e5
        res["models"]["page"][s]["arousal"]["alpha"] = 1e5
        res["models"]["L2_acoustic"][s]["valence"]["alpha"] = 1e5
    res["models"]["L0+L1"]["valence"]["alpha"] = 1e5
    d = _run_figs(res, tmp_path, monkeypatch)
    assert d == _recs(res) and d["hierarchy"] > 0 and d["generalisation"] > 0


def test_figures_from_genuine_run_results(fixture, monkeypatch, tmp_path):
    tmp, freeze = fixture
    res = _run(tmp, freeze)  # real run.run on the synthetic fixture, never the real data
    d = _run_figs(res, tmp_path, monkeypatch)
    assert d == _recs(res)
