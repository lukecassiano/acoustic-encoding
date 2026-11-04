"""Pre-registered hypothesis run (Task 8): H1-H3 on DEAM, H4 on PMEmo, descriptive models.

Everything follows docs/superpowers/specs/2026-11-02-perception-freeze.md (rulings R16-R23).
One group-disjoint split is shared by every model and by Task 10 (export.py): readout.split on
the artist groups of ALL songs in levels.npz (SEED), then unrated songs are dropped from train/test. PMEmo is never used for fitting, alpha selection or standardization.

    python -m src.perception.run --freeze docs/superpowers/specs/2026-11-02-perception-freeze.md

Outputs data/perception/results.json (all numbers) and data/perception/results.md (tables only).

results.json schema
-------------------
Per-rating record ("rec"), DEAM held-out test set:
    {"r2", "r", "r2_lo", "r2_hi", "perm_p", "alpha", "n_kept", "dims"}
    r2 = 1 - SS_res/SS_tot about the test mean; r2_lo/hi = 2.5/97.5 percentiles of R2 over
    N_BOOT bootstrap resamples of test songs (descriptive); perm_p = readout.perm_p (descriptive);
    alpha = CV-chosen ridge alpha; n_kept = columns surviving VarianceThreshold(0); dims = input
    columns. The theory baseline has alpha = n_kept = null and dims = number of summed features.
PMEmo record ("prec"): {"r", "lo", "hi"} = Pearson r and 2.5/97.5 bootstrap percentiles.

{
  "meta": {"freeze", "seed", "test_size", "alphas", "n_folds", "n_boot", "n_perm", "n_proj",
           "listeners": ["001",..], "R": {sub: R}, "region_names": {sub: [..]},
           "l0_acoustic_cols": [..61 L0 indices], "l0_structural_cols": [..8 indices]},
  "data": {"deam":  {"n_levels", "n_rated", "dropped_no_rating" (= dropped_no_rating_row +
                     dropped_nan_rating), "dropped_no_rating_row", "dropped_nan_rating", "n_train", "n_test",
                     "n_groups" (rated songs), "n_groups_train", "n_groups_test",
                     "train_song_ids": [..], "test_song_ids": [..]},
           "pmemo": {"n_levels", "n_rated", "dropped_no_rating", "dropped_no_rating_row",
                     "dropped_nan_rating"}},
  "models": {                                         # DEAM held-out, descriptive
     "L0" | "L0+L1" | "L0_acoustic" | "L0_structural" | "theory": {"valence": rec, "arousal": rec},
     "L2" | "L2_acoustic" | "L2_structural" | "page":  {sub: {"valence": rec, "arousal": rec}}},
  "hypotheses": {
     "H1": {"rating": "arousal", "a": "L0+L1", "b": "L0", "diff", "lo", "hi", "verdict"},
     "H2": {"L0": {"valence": d, "arousal": d, "verdict"},
            "L2": {"listeners": {sub: {"valence": d, "arousal": d, "verdict"}},
                   "n_support", "n_fail", "verdict"}},
           d = {"a", "b", "diff", "lo", "hi", "verdict"} with diff = R2(a) - R2(b)
           (valence: a = structural, b = acoustic; arousal: a = acoustic, b = structural)
     "H3": {rating: {"listeners": {sub: {"r2", "null_p5", "null_p50", "null_p95", "null_mean",
                                         "pass", "null": [n_proj R2 values]}},
                     "n_pass", "n_fail", "verdict"}},
     "H4": {rating: {"model": "page", "listeners": {sub: prec}, "n_lo_pos", "n_hi_neg", "verdict"}}},
  "pmemo_models": {"L0" | "L0+L1" | "theory": {rating: prec},
                   "L2" | "page": {sub: {rating: prec}}},
  "verdicts": {"H1", "H2_L0", "H2_L2", "H3_valence", "H3_arousal", "H4_valence", "H4_arousal"}
}
Verdict strings are always one of "supported", "not supported", "inconclusive".
"""

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src import deam, pmemo
from src import features as F
from src.perception.levels import (ACOUSTIC_COLS, L0_NAMES, OUT, STRUCTURAL_COLS, SUBS, l2,
                                   load_bundle)
from src.perception.readout import (ALPHAS, N_BOOT, N_FOLDS, N_PERM, N_PROJ, SEED, TEST_SIZE,
                                    _corr, _r2, boot_diff, fit_readout, heldout, perm_p,
                                    random_projection_r2, split, theory_baseline, verdict)

RATINGS = ("valence", "arousal")
MIN_LISTENERS = 4  # ">= 4 of 5"

L0_ACOUSTIC = [L0_NAMES.index(n) for n in F.FEATURE_NAMES["acoustic"]] + [L0_NAMES.index("raw_log_rms")]
L0_STRUCTURAL = [L0_NAMES.index(n) for n in F.FEATURE_NAMES["structural"]]
assert len(L0_ACOUSTIC) == 61 and len(L0_STRUCTURAL) == 8


# ------------------------------------------------------------------ data and features

def merge_ratings(song_id, ratings):
    """Ratings aligned to song_id (the one definition shared with Task 10, export.py).

    Returns {"valence", "arousal": (n,) float, NaN where unrated; "rated": (n,) bool;
    "dropped_no_rating_row", "dropped_nan_rating", "dropped_no_rating" (their sum)}.
    """
    sid = np.asarray(song_id).astype(int)
    rt = ratings.drop_duplicates("song_id").set_index("song_id")
    has = np.isin(sid, rt.index.values)
    out = {}
    for k in RATINGS:
        out[k] = np.full(len(sid), np.nan)
        out[k][has] = rt.loc[sid[has], k].to_numpy(float)
    # FREEZE-AMBIGUITY: a rating row with a missing (NaN) valence or arousal is treated as
    # "without a rating" (dropped at merge, counted) rather than dropped per rating.
    out["rated"] = has & np.isfinite(out["valence"]) & np.isfinite(out["arousal"])
    out["dropped_no_rating_row"] = int((~has).sum())
    out["dropped_nan_rating"] = int((has & ~out["rated"]).sum())
    out["dropped_no_rating"] = out["dropped_no_rating_row"] + out["dropped_nan_rating"]
    return out


def load_levels(path, ratings):
    """All songs of a levels file (levels order kept) merged with ratings (merge_ratings); unrated
    songs carry NaN and are dropped by the caller."""
    z = np.load(path)
    out = dict(song_id=z["song_id"].astype(int), group=z["group"].astype(str), L0=z["L0"].astype(float),
               L1=z["L1"].astype(float), xbar=z["xbar"].astype(float), n_levels=int(len(z["song_id"])))
    out.update(merge_ratings(out["song_id"], ratings))
    return out


def rated_only(lv):
    """The rated songs of load_levels output (the `rated` mask is consumed)."""
    keys = ("song_id", "group", "L0", "L1", "xbar") + RATINGS
    m = lv["rated"]
    return {**{k: v for k, v in lv.items() if k != "rated"}, **{k: lv[k][m] for k in keys}}


def l2_features(xbar, bundle):
    """'L2 alone': np.hstack([L2_acoustic, L2_structural]) -> (n, 2R)."""
    p = l2(xbar, bundle)
    return np.hstack([p["acoustic"], p["structural"]])


def page_features(L1, xbar, bundle):
    """Page readout input: np.hstack([L1, L2_acoustic, L2_structural]) -> (n, 66 + 2R).
    Task 10 (export) must build its features with exactly this function / column order."""
    p = l2(xbar, bundle)
    return np.hstack([L1, p["acoustic"], p["structural"]])


def deam_split(groups, rated):
    """The one shared split, same order as Task 10 (src/perception/export.py): readout.split on the
    artist groups of ALL songs in levels.npz (SEED), then unrated songs are dropped from train/test.
    Returns indices into the full levels arrays."""
    tr, te = split(groups, seed=SEED)
    return tr[rated[tr]], te[rated[te]]


# ------------------------------------------------------------------ statistics helpers

def _boot_ci(stat, y, pred, n, seed):
    """2.5/97.5 percentiles of stat(y[idx], pred[idx]) over n resamples of songs, default_rng(seed)."""
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    rng = np.random.default_rng(seed)
    m = len(y)
    v = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        v[i] = stat(y[idx], pred[idx])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def boot_r2(y, pred, n=N_BOOT, seed=SEED):
    """Descriptive 95% interval of a single model's held-out R2: N_BOOT resamples of test songs."""
    # FREEZE-AMBIGUITY: the freeze bootstraps only R2 differences; per-model R2 intervals are
    # needed by the Task 9 hierarchy figure. Same resampling as boot_diff (default_rng(seed),
    # resample test songs with replacement, 2.5/97.5 percentiles). Descriptive; gates nothing.
    return _boot_ci(_r2, y, pred, n, seed)


def boot_r(y, pred, n=N_BOOT, seed=SEED):
    """H4 metric: Pearson r with a 95% CI from n bootstrap resamples of songs (default_rng(seed))."""
    # FREEZE-AMBIGUITY: "(SEED)" read as a fresh default_rng(SEED) for every (model, listener,
    # rating) interval, the same convention as boot_diff and random_projection_r2 (R17).
    lo, hi = _boot_ci(_corr, y, pred, n, seed)
    return dict(r=_corr(np.asarray(y, float), np.asarray(pred, float)), lo=lo, hi=hi)


def _record(y_te, pred, n_boot, n_perm, model=None, dims=None):
    lo, hi = boot_r2(y_te, pred, n=n_boot)
    rec = dict(r2=float(_r2(np.asarray(y_te, float), pred)), r=_corr(np.asarray(y_te, float), pred),
               r2_lo=lo, r2_hi=hi, perm_p=float(perm_p(y_te, pred, n=n_perm)),
               alpha=None, n_kept=None, dims=dims)
    if model is not None:
        rec.update(alpha=float(model.named_steps["ridge"].alpha), n_kept=int(model.keep_.sum()))
    return rec


def _fit(Xtr, Xte, ytr, yte, gtr, n_boot, n_perm):
    m = fit_readout(Xtr, ytr, gtr)
    pred = heldout(m, Xte, yte)["pred"]
    return m, pred, _record(yte, pred, n_boot, n_perm, model=m, dims=int(Xtr.shape[1]))


def _diff(y, pa, pb, a, b, n_boot):
    d = boot_diff(y, pa, pb, n=n_boot)
    return dict(a=a, b=b, **d, verdict=verdict(d["lo"], d["hi"]))


def h2_combine(v_valence, v_arousal):
    """R16: supports iff both 'supported'; fails iff either 'not supported'; else inconclusive."""
    if v_valence == "supported" and v_arousal == "supported":
        return "supported"
    if "not supported" in (v_valence, v_arousal):
        return "not supported"
    return "inconclusive"


def majority(n_pos, n_neg, k=MIN_LISTENERS):
    """Symmetric >= 4 of 5 rule (R16, R17, R23)."""
    if n_pos >= k:
        return "supported"
    if n_neg >= k:
        return "not supported"
    return "inconclusive"


def _h3_job(args):
    key, xtr, xte, ytr, yte, gtr, W, n_proj = args
    return key, random_projection_r2(xtr, xte, ytr, yte, gtr, W, [ACOUSTIC_COLS, STRUCTURAL_COLS],
                                     n=n_proj, seed=SEED)


# ------------------------------------------------------------------ run

def run(freeze_path, n_boot=N_BOOT, n_perm=N_PERM, n_proj=N_PROJ, workers=1,
        levels_path=None, pmemo_path=None, out_dir=None):
    """Run every pre-registered analysis; write results.json and results.md; return the dict.
    n_boot / n_perm / n_proj default to the frozen values and are lowered only by tests."""
    levels_path = Path(levels_path or OUT / "levels.npz")
    pmemo_path = Path(pmemo_path or OUT / "pmemo_levels.npz")
    out_dir = Path(out_dir or OUT)

    # inputs must exist before any rating is read
    if not levels_path.exists():
        raise FileNotFoundError(f"{levels_path} missing; run `python -m src.perception.levels` first")
    if not pmemo_path.exists():
        raise FileNotFoundError(f"{pmemo_path} missing; run `python -m src.perception.levels --dataset pmemo` "
                                "first (H4 is pre-registered; the run needs both datasets)")
    # ratings: both loaders refuse without the go-ahead line
    D = load_levels(levels_path, deam.load_ratings(freeze_path))
    P = rated_only(load_levels(pmemo_path, pmemo.load_ratings(freeze_path)))

    # tr / te index the full DEAM arrays (unrated songs are in neither)
    tr, te = deam_split(D["group"], D["rated"])
    gtr = D["group"][tr]
    Y = {k: (D[k][tr], D[k][te]) for k in RATINGS}
    bundles = {s: load_bundle(s) for s in SUBS}

    models, pmodels, preds = {}, {}, {}

    def flat(name, Xall, Xp):
        models[name], pmodels[name] = {}, {}
        for k in RATINGS:
            m, pred, rec = _fit(Xall[tr], Xall[te], *Y[k], gtr, n_boot, n_perm)
            models[name][k], preds[(name, None, k)] = rec, pred
            if Xp is not None:
                pmodels[name][k] = boot_r(P[k], m.predict(Xp), n=n_boot)
        if Xp is None:
            del pmodels[name]

    # --- DEAM-level models
    flat("L0", D["L0"], P["L0"])
    flat("L0+L1", np.hstack([D["L0"], D["L1"]]), np.hstack([P["L0"], P["L1"]]))
    flat("L0_acoustic", D["L0"][:, L0_ACOUSTIC], None)
    flat("L0_structural", D["L0"][:, L0_STRUCTURAL], None)

    th = theory_baseline(D["L0"][tr], D["L1"][tr], D["L0"][te], D["L1"][te], {k: Y[k][0] for k in RATINGS})
    th_p = theory_baseline(D["L0"][tr], D["L1"][tr], P["L0"], P["L1"], {k: Y[k][0] for k in RATINGS})
    models["theory"] = {k: _record(Y[k][1], th[k], n_boot, n_perm, dims={"arousal": 3, "valence": 2}[k])
                        for k in RATINGS}
    pmodels["theory"] = {k: boot_r(P[k], th_p[k], n=n_boot) for k in RATINGS}

    # --- per-listener models
    for name in ("L2", "L2_acoustic", "L2_structural", "page"):
        models[name] = {}
    pmodels["L2"], pmodels["page"] = {}, {}
    for s in SUBS:
        b = bundles[s]
        parts = l2(D["xbar"], b)
        feats = {"L2": (l2_features(D["xbar"], b), l2_features(P["xbar"], b)),
                 "L2_acoustic": (parts["acoustic"], None),
                 "L2_structural": (parts["structural"], None),
                 "page": (page_features(D["L1"], D["xbar"], b), page_features(P["L1"], P["xbar"], b))}
        for name, (X, Xp) in feats.items():
            models[name][s] = {}
            if Xp is not None:
                pmodels[name][s] = {}
            for k in RATINGS:
                m, pred, rec = _fit(X[tr], X[te], *Y[k], gtr, n_boot, n_perm)
                models[name][s][k], preds[(name, s, k)] = rec, pred
                if Xp is not None:
                    pmodels[name][s][k] = boot_r(P[k], m.predict(Xp), n=n_boot)

    # --- H1
    yA = Y["arousal"][1]
    h1 = dict(rating="arousal", **_diff(yA, preds[("L0+L1", None, "arousal")], preds[("L0", None, "arousal")],
                                        "L0+L1", "L0", n_boot))

    # --- H2
    def h2_pair(ac, st, sub):
        dv = _diff(Y["valence"][1], preds[(st, sub, "valence")], preds[(ac, sub, "valence")], "structural",
                   "acoustic", n_boot)
        da = _diff(Y["arousal"][1], preds[(ac, sub, "arousal")], preds[(st, sub, "arousal")], "acoustic",
                   "structural", n_boot)
        return dict(valence=dv, arousal=da, verdict=h2_combine(dv["verdict"], da["verdict"]))

    h2_l0 = h2_pair("L0_acoustic", "L0_structural", None)
    h2_l2 = {s: h2_pair("L2_acoustic", "L2_structural", s) for s in SUBS}
    n_sup = sum(v["verdict"] == "supported" for v in h2_l2.values())
    n_fail = sum(v["verdict"] == "not supported" for v in h2_l2.values())
    h2 = dict(L0=h2_l0, L2=dict(listeners=h2_l2, n_support=n_sup, n_fail=n_fail, verdict=majority(n_sup, n_fail)))

    # --- H3 (projection nulls; each call seeded default_rng(SEED), so worker count cannot change results)
    jobs = [((s, k), D["xbar"][tr], D["xbar"][te], Y[k][0], Y[k][1], gtr, bundles[s]["W_region"], n_proj)
            for s in SUBS for k in RATINGS]
    if workers > 1:
        with Pool(min(workers, len(jobs))) as pool:
            nulls = dict(pool.map(_h3_job, jobs))
    else:
        nulls = dict(map(_h3_job, jobs))
    h3 = {}
    for k in RATINGS:
        lst = {}
        for s in SUBS:
            null, r2 = nulls[(s, k)], models["L2"][s][k]["r2"]
            p95 = float(np.percentile(null, 95))
            lst[s] = dict(r2=r2, null_p5=float(np.percentile(null, 5)), null_p50=float(np.percentile(null, 50)),
                          null_p95=p95, null_mean=float(null.mean()), **{"pass": bool(r2 > p95)},
                          null=[float(v) for v in null])
        n_pass = sum(v["pass"] for v in lst.values())
        h3[k] = dict(listeners=lst, n_pass=n_pass, n_fail=len(SUBS) - n_pass,
                     verdict=majority(n_pass, len(SUBS) - n_pass))

    # --- H4 (primary: page readout per listener; DEAM-train models applied unchanged)
    h4 = {}
    for k in RATINGS:
        lst = {s: pmodels["page"][s][k] for s in SUBS}
        n_pos = sum(v["lo"] > 0 for v in lst.values())
        n_neg = sum(v["hi"] < 0 for v in lst.values())
        h4[k] = dict(model="page", listeners=lst, n_lo_pos=n_pos, n_hi_neg=n_neg, verdict=majority(n_pos, n_neg))

    res = dict(
        meta=dict(freeze=str(freeze_path), seed=SEED, test_size=TEST_SIZE, alphas=[float(a) for a in ALPHAS],
                  n_folds=N_FOLDS, n_boot=n_boot, n_perm=n_perm, n_proj=n_proj, listeners=list(SUBS),
                  R={s: int(bundles[s]["W_region"].shape[1]) for s in SUBS},
                  region_names={s: list(bundles[s]["region_names"]) for s in SUBS},
                  l0_acoustic_cols=L0_ACOUSTIC, l0_structural_cols=L0_STRUCTURAL),
        data=dict(deam=dict(n_levels=D["n_levels"], n_rated=int(D["rated"].sum()),
                            dropped_no_rating=D["dropped_no_rating"],
                            dropped_no_rating_row=D["dropped_no_rating_row"],
                            dropped_nan_rating=D["dropped_nan_rating"], n_train=int(len(tr)), n_test=int(len(te)),
                            n_groups=int(len(set(D["group"][D["rated"]]))), n_groups_train=int(len(set(gtr))),
                            n_groups_test=int(len(set(D["group"][te]))),
                            train_song_ids=[int(v) for v in D["song_id"][tr]],
                            test_song_ids=[int(v) for v in D["song_id"][te]]),
                  pmemo=dict(n_levels=P["n_levels"], n_rated=int(len(P["song_id"])),
                             dropped_no_rating=P["dropped_no_rating"],
                             dropped_no_rating_row=P["dropped_no_rating_row"],
                             dropped_nan_rating=P["dropped_nan_rating"])),
        models=models,
        hypotheses=dict(H1=h1, H2=h2, H3=h3, H4=h4),
        pmemo_models=pmodels,
        verdicts=dict(H1=h1["verdict"], H2_L0=h2_l0["verdict"], H2_L2=h2["L2"]["verdict"],
                      H3_valence=h3["valence"]["verdict"], H3_arousal=h3["arousal"]["verdict"],
                      H4_valence=h4["valence"]["verdict"], H4_arousal=h4["arousal"]["verdict"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, allow_nan=False), encoding="utf-8")
    (out_dir / "results.md").write_text(to_markdown(res), encoding="utf-8")
    return res


# ------------------------------------------------------------------ results.md

EDGE, WEAK = "†edge", "‡"
WEAK_DIFF = 0.01  # descriptive annotation threshold for "‡" (results.md only; gates nothing)
FOOTNOTES = [
    f"{EDGE}: the CV-chosen ridge alpha sits at an end of the frozen grid ALPHAS (first or last value).",
    f"{WEAK}: a \"supported\" outcome, or a listener counted toward one, where the winning model's held-out "
    f"R2 <= 0 or the |difference| < {WEAK_DIFF} (H3: difference = L2 R2 - null p95; H4: winning model = the "
    "listener's page readout, DEAM held-out R2).",
    "Markers are descriptive annotations; every verdict follows the freeze unchanged.",
]


def _f(v, d=3):
    return "n/a" if v is None else f"{v:.{d}f}"


def _ci(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def _rec(res, name, sub, k):
    m = res["models"][name]
    return m[k] if sub is None else m[sub][k]


def _is_edge(rec, alphas):
    a = rec.get("alpha")
    return a is not None and bool(np.isclose(a, alphas[0]) or np.isclose(a, alphas[-1]))


def _alpha(rec, alphas):
    a = rec.get("alpha")
    if a is None:
        return "n/a"
    return f"{a:.4g}" + (f" {EDGE}" if _is_edge(rec, alphas) else "")


def _weak(r2_win, diff):
    return r2_win <= 0 or abs(diff) < WEAK_DIFF


def to_markdown(res):
    alphas = res["meta"]["alphas"]
    H = res["hypotheses"]
    notes = {k: set() for k in res["verdicts"]}  # verdict key -> markers

    # --- H1
    h1 = H["H1"]
    ra, rb = _rec(res, h1["a"], None, "arousal"), _rec(res, h1["b"], None, "arousal")
    w = h1["verdict"] == "supported" and _weak(ra["r2"], h1["diff"])
    notes["H1"] |= ({WEAK} if w else set()) | ({EDGE} if _is_edge(ra, alphas) or _is_edge(rb, alphas) else set())
    S_h1 = ["## H1 (arousal): R2(L0+L1) - R2(L0)", "",
            "| a | b | R2(a) | R2(b) | alpha(a) | alpha(b) | diff | 95% CI | verdict |",
            "|---|---|---|---|---|---|---|---|---|",
            f"| {h1['a']} | {h1['b']} | {_f(ra['r2'])} | {_f(rb['r2'])} | {_alpha(ra, alphas)} | {_alpha(rb, alphas)} "
            f"| {_f(h1['diff'])} | {_ci(h1['lo'], h1['hi'])} | {h1['verdict']}{' ' + WEAK if w else ''} |", ""]

    # --- H2
    S_h2 = ["## H2: pathway double dissociation", "",
            "Valence: R2(structural) - R2(acoustic); arousal: R2(acoustic) - R2(structural).", "",
            "| Level | Listener | Rating | a | b | R2(a) | R2(b) | alpha(a) | alpha(b) | diff | 95% CI | verdict "
            "| level/listener verdict |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def h2rows(level, sub, x, key):
        out = []
        for k in RATINGS:
            d = x[k]
            ma = _rec(res, f"{level}_{d['a']}", sub, k)
            mb = _rec(res, f"{level}_{d['b']}", sub, k)
            w = d["verdict"] == "supported" and _weak(ma["r2"], d["diff"])
            if w and x["verdict"] == "supported":
                notes[key].add(WEAK)
            if _is_edge(ma, alphas) or _is_edge(mb, alphas):
                notes[key].add(EDGE)
            out.append(f"| {level} | {sub or '-'} | {k} | {d['a']} | {d['b']} | {_f(ma['r2'])} | {_f(mb['r2'])} "
                       f"| {_alpha(ma, alphas)} | {_alpha(mb, alphas)} | {_f(d['diff'])} | {_ci(d['lo'], d['hi'])} "
                       f"| {d['verdict']}{' ' + WEAK if w else ''} | {x['verdict']} |")
        return out

    S_h2 += h2rows("L0", None, H["H2"]["L0"], "H2_L0")
    for s, x in H["H2"]["L2"]["listeners"].items():
        S_h2 += h2rows("L2", s, x, "H2_L2")
    h2l2 = H["H2"]["L2"]
    S_h2 += ["", f"H2-L0: {H['H2']['L0']['verdict']}. H2-L2: {h2l2['n_support']} of 5 support, "
                 f"{h2l2['n_fail']} of 5 fail -> {h2l2['verdict']}.", ""]

    # --- H3
    S_h3 = ["## H3: L2 vs random block-wise projections", "",
            "| Rating | Listener | L2 R2 | alpha | null p50 | null p95 | pass |", "|---|---|---|---|---|---|---|"]
    for k, x in H["H3"].items():
        key = f"H3_{k}"
        for s, v in x["listeners"].items():
            m = _rec(res, "L2", s, k)
            w = v["pass"] and _weak(v["r2"], v["r2"] - v["null_p95"])
            if w and x["verdict"] == "supported":
                notes[key].add(WEAK)
            if _is_edge(m, alphas):
                notes[key].add(EDGE)
            S_h3.append(f"| {k} | {s} | {_f(v['r2'])} | {_alpha(m, alphas)} | {_f(v['null_p50'])} | {_f(v['null_p95'])} "
                        f"| {'yes' if v['pass'] else 'no'}{' ' + WEAK if w else ''} |")
    S_h3 += [""] + [f"H3-{k}: {x['n_pass']} of 5 pass -> {x['verdict']}." for k, x in H["H3"].items()] + [""]

    # --- H4
    S_h4 = ["## H4: DEAM-trained page readout on PMEmo (Pearson r)", "",
            "| Rating | Listener | PMEmo r | 95% CI | CI lo > 0 | DEAM R2 (page) | alpha |", "|---|---|---|---|---|---|---|"]
    for k, x in H["H4"].items():
        key = f"H4_{k}"
        for s, v in x["listeners"].items():
            m = _rec(res, "page", s, k)
            pos = v["lo"] > 0
            w = pos and m["r2"] <= 0
            if w and x["verdict"] == "supported":
                notes[key].add(WEAK)
            if _is_edge(m, alphas):
                notes[key].add(EDGE)
            S_h4.append(f"| {k} | {s} | {_f(v['r'])} | {_ci(v['lo'], v['hi'])} | {'yes' if pos else 'no'}"
                        f"{' ' + WEAK if w else ''} | {_f(m['r2'])} | {_alpha(m, alphas)} |")
    S_h4 += [""] + [f"H4-{k}: {x['n_lo_pos']} of 5 with CI lo > 0, {x['n_hi_neg']} of 5 with CI hi < 0 -> {x['verdict']}."
                    for k, x in H["H4"].items()] + [""]

    # --- header, data, verdicts
    L = ["# Perception layer: pre-registered results", "",
         f"Freeze: `{res['meta']['freeze']}`. Seed {res['meta']['seed']}; n_boot {res['meta']['n_boot']}, "
         f"n_perm {res['meta']['n_perm']}, n_proj {res['meta']['n_proj']}.", ""]
    dd, pd_ = res["data"]["deam"], res["data"]["pmemo"]
    L += ["## Data", "",
          "| Dataset | In levels file | Rated | Dropped: no rating row | Dropped: NaN rating | Train | Test "
          "| Groups (train/test) |",
          "|---|---|---|---|---|---|---|---|",
          f"| DEAM | {dd['n_levels']} | {dd['n_rated']} | {dd['dropped_no_rating_row']} | {dd['dropped_nan_rating']} "
          f"| {dd['n_train']} | {dd['n_test']} | {dd['n_groups']} ({dd['n_groups_train']}/{dd['n_groups_test']}) |",
          f"| PMEmo | {pd_['n_levels']} | {pd_['n_rated']} | {pd_['dropped_no_rating_row']} | {pd_['dropped_nan_rating']} "
          f"| 0 | {pd_['n_rated']} | n/a |", ""]
    L += ["## Verdicts", "", "| Hypothesis | Verdict | Markers |", "|---|---|---|"]
    order = [WEAK, EDGE]
    L += [f"| {k} | {v} | {' '.join(m for m in order if m in notes[k] and (m != WEAK or v == 'supported'))} |"
          for k, v in res["verdicts"].items()]
    L += [""] + [f"- {f}" for f in FOOTNOTES] + [""]
    L += S_h1 + S_h2 + S_h3 + S_h4

    # --- descriptive
    L += ["## Descriptive: DEAM held-out models", "",
          "Per-listener rows are reported per listener; differences between listeners are exploratory.", "",
          "| Model | Listener | Rating | dims | R2 | 95% CI | r | perm p | alpha |", "|---|---|---|---|---|---|---|---|---|"]
    for name, x in res["models"].items():
        per = x.items() if set(x) - set(RATINGS) else [("-", x)]
        for s, recs in per:
            for k in RATINGS:
                r = recs[k]
                L.append(f"| {name} | {s} | {k} | {r['dims']} | {_f(r['r2'])} | {_ci(r['r2_lo'], r['r2_hi'])} "
                         f"| {_f(r['r'])} | {_f(r['perm_p'], 4)} | {_alpha(r, alphas)} |")
    L += ["", "## Descriptive: DEAM-trained models on PMEmo (Pearson r)", "",
          "| Model | Listener | Rating | r | 95% CI |", "|---|---|---|---|---|"]
    for name, x in res["pmemo_models"].items():
        per = x.items() if set(x) - set(RATINGS) else [("-", x)]
        for s, recs in per:
            for k in RATINGS:
                r = recs[k]
                L.append(f"| {name} | {s} | {k} | {_f(r['r'])} | {_ci(r['lo'], r['hi'])} |")
    L += ["", "Markers:"] + [f"- {f}" for f in FOOTNOTES]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=10, help="processes for the H3 projection nulls")
    args = ap.parse_args(argv)
    res = run(args.freeze, workers=args.workers)
    for k, v in res["verdicts"].items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
