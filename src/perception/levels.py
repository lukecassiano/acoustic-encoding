"""Three representation levels per music excerpt, the regressors for the affect analysis.

L0  CAMP's own audio features (60 acoustic + 8 structural clip means, plus raw log RMS, which
    the prepare() RMS normalization would otherwise erase).
L1  a simulated ear's output: per-channel envelope means, per-channel modulation depth, loudness.
L2  CAMP's predicted cortical response, split into the acoustic and structural pathways. L2 is
    linear in the clip-mean design row xbar, so acoustic + structural = the full prediction.

L0 and L2 come from the raw excerpt (F.extract(F.prepare(y22))); only L1 sees the ear-rendered signal.
Clip means use TR rows with index >= TR_START (>= 7.5 s) for every level.
"""

import argparse
import json
from functools import lru_cache
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from src import features as F
from src.deam import excerpts, load_excerpt
from src.pmemo import excerpts as pmemo_excerpts
from src.ear import Source, listen
from src.ear.constants import FRAME_S, N_CH
from src.export_web import probe_design

TR_START = 5  # 5 * 1.5 s = 7.5 s
MIN_DURATION_S = 9.0
SILENT_RMS = 1e-6

# L1 summary constants (a property of this summary, not of the ear model)
MOD_LO_HZ, MOD_HI_HZ, MOD_ORDER = 0.5, 20.0, 2
L1_START_S = 7.5
MOD_FLOOR = 1e-12

WEB = F.DATA / "web"
OUT = F.DATA / "perception"
SUBS = [f"00{i}" for i in range(1, 6)]
ACOUSTIC_COLS, STRUCTURAL_COLS = slice(0, 300), slice(300, 340)

L0_NAMES = list(F.FEATURE_NAMES["acoustic"]) + list(F.FEATURE_NAMES["structural"]) + ["raw_log_rms"]
L1_NAMES = ([f"env_{i:02d}" for i in range(N_CH)] + [f"mod_{i:02d}" for i in range(N_CH)]
            + ["phon_mean", "phon_p95"])


# ------------------------------------------------------------------ bundles

def _read_meta(sub):
    return json.loads((WEB / f"sub-{sub}" / "meta.json").read_text())


def _stats(meta):
    return {b: (np.array(v["mean"]), np.array(v["std"]), np.array(v["nuisance"]))
            for b, v in meta["bands"].items()}


def load_bundle(sub):
    """Feature stats, region weights and names for one listener. Stats and nuisance betas must be
    identical across all five listeners (the design is subject-independent); fail loudly otherwise."""
    ref = _stats(_read_meta(SUBS[0]))
    for s in SUBS[1:]:
        other = _stats(_read_meta(s))
        assert set(other) == set(ref), f"band sets differ between sub-{SUBS[0]} and sub-{s}"
        for band in ref:
            for a, b, what in zip(ref[band], other[band], ("mean", "std", "nuisance")):
                assert a.shape == b.shape and np.allclose(a, b), \
                    f"{band} {what} differs between sub-{SUBS[0]} and sub-{s}"
    meta = _read_meta(sub)
    regions = sorted((r for r in meta["regions"] if r["column"] is not None), key=lambda r: r["column"])
    raw = np.fromfile(WEB / f"sub-{sub}" / "w_region.f32", dtype="<f4")
    W = raw.reshape(meta["n_cols"], len(regions)).astype(np.float64)
    return dict(stats=ref, W_region=W, region_names=[r["name"] for r in regions],
                acoustic_cols=ACOUSTIC_COLS, structural_cols=STRUCTURAL_COLS)


@lru_cache(maxsize=1)
def _stats_001():
    return load_bundle("001")["stats"]


def l2(xbar_rows, bundle):
    """Block products of the clip-mean design rows with the region weights."""
    X, W = np.atleast_2d(xbar_rows), bundle["W_region"]
    a, s = bundle["acoustic_cols"], bundle["structural_cols"]
    return dict(acoustic=X[:, a] @ W[a], structural=X[:, s] @ W[s])


# ------------------------------------------------------------------ levels

def l0(y22):
    feats = F.extract(F.prepare(y22))
    means = np.concatenate([feats["acoustic"][TR_START:].mean(0), feats["structural"][TR_START:].mean(0)])
    raw_log_rms = np.log(np.sqrt(np.mean(np.square(y22))) + 1e-8)
    return np.append(means, raw_log_rms)


def xbar(y22, stats):
    return probe_design(F.extract(F.prepare(y22)), stats)[TR_START:].mean(0)


def _modulation_depth(env):
    sos = butter(MOD_ORDER, [MOD_LO_HZ, MOD_HI_HZ], btype="band", fs=1 / FRAME_S, output="sos")
    band = sosfiltfilt(sos, env, axis=1)
    return np.mean(band ** 2, axis=1) / np.maximum(np.mean(env, axis=1) ** 2, MOD_FLOOR)


def l1(y44):
    out = listen([Source(y44, 0.0, 1.0, "studio", 0.0)])
    start = int(round(L1_START_S / FRAME_S))
    per_ear = []
    for ear in (out.left, out.right):
        env, phon = ear.env[:, start:], ear.loud["phon"][start:]
        per_ear.append(np.concatenate([env.mean(1), _modulation_depth(env),
                                       [phon.mean(), np.percentile(phon, 95)]]))
    return np.mean(per_ear, axis=0)


# ------------------------------------------------------------------ per song

def process_one(path):
    """("ok", {l0, l1, xbar}) or ("skip", reason)."""
    try:
        y22, y44 = load_excerpt(path)
    except Exception:
        return "skip", "unreadable"
    if len(y22) / F.SR < MIN_DURATION_S:
        return "skip", "shorter than 9 s"
    if np.sqrt(np.mean(np.square(y22))) < SILENT_RMS:
        return "skip", "silent"
    return "ok", dict(l0=l0(y22), l1=l1(y44), xbar=xbar(y22, _stats_001()))


def _work(args):
    sid, path, cache = args
    status, res = process_one(path)
    if status == "ok":
        tmp = cache / f"{sid}.tmp.npz"
        np.savez(tmp, **res)
        tmp.replace(cache / f"{sid}.npz")
    else:
        tmp = cache / f"{sid}.skip.tmp"
        tmp.write_text(res, encoding="utf-8")
        tmp.replace(cache / f"{sid}.skip")
    return sid, status, res if status == "skip" else None


def _record(result, skipped, i, n):
    sid, status, reason = result
    if status == "skip":
        skipped[sid] = reason
    if i % 50 == 0 or i == n:
        print(f"{i}/{n}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="only the first N excerpts (testing)")
    ap.add_argument("--dataset", choices=["deam", "pmemo"], default="deam")
    args = ap.parse_args(argv)
    pre = "" if args.dataset == "deam" else "pmemo_"
    df = (excerpts if args.dataset == "deam" else pmemo_excerpts)()
    if args.limit is not None:
        df = df.head(args.limit)
    cache = OUT / f"{pre}levels"
    cache.mkdir(parents=True, exist_ok=True)
    todo = [(int(r.song_id), r.path, cache) for r in df.itertuples() if not (cache / f"{int(r.song_id)}.npz").exists()
            and not (cache / f"{int(r.song_id)}.skip").exists()]
    skipped = {}
    if todo:
        if args.workers > 1:
            with Pool(args.workers) as pool:
                for i, r in enumerate(pool.imap_unordered(_work, todo), 1):
                    _record(r, skipped, i, len(todo))
        else:
            for i, r in enumerate(map(_work, todo), 1):
                _record(r, skipped, i, len(todo))
    rows = []
    for r in df.itertuples():
        sk = cache / f"{int(r.song_id)}.skip"
        if sk.exists():
            skipped[int(r.song_id)] = sk.read_text(encoding="utf-8")
        f = cache / f"{int(r.song_id)}.npz"
        if f.exists():
            z = np.load(f)
            rows.append((int(r.song_id), r.group, z["l0"], z["l1"], z["xbar"]))
    np.savez(OUT / f"{pre}levels.npz", song_id=np.array([r[0] for r in rows], dtype=int),
             group=np.array([r[1] for r in rows], dtype=str),
             L0=np.array([r[2] for r in rows]).reshape(len(rows), -1),
             L1=np.array([r[3] for r in rows]).reshape(len(rows), -1),
             xbar=np.array([r[4] for r in rows]).reshape(len(rows), -1))
    pd.DataFrame(sorted(skipped.items()), columns=["song_id", "reason"]).to_csv(OUT / f"{pre}skipped.csv", index=False)
    print(f"{len(rows)} excerpts, {len(skipped)} skipped: "
          f"{pd.Series(list(skipped.values())).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
