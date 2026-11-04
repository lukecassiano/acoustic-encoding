import json
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src import features as F
from src.export_web import probe_design
from src.perception import levels
from src.perception.levels import (L0_NAMES, L1_NAMES, TR_START, l0, l1, l2, load_bundle,
                                   process_one, xbar)

NEEDS_WEB = pytest.mark.skipif(not (levels.WEB / "sub-001" / "meta.json").exists(),
                               reason="export_web bundles (data/web) not built")


def synth(sr, sec=15.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * sec)) / sr
    y = np.zeros_like(t)
    for k, on in enumerate(np.arange(0.0, sec, 0.5)):
        f0 = 220 * 2 ** (rng.integers(0, 8) / 12)
        m = t >= on
        env = np.exp(-(t - on) * 4) * m
        y += env * sum(np.sin(2 * np.pi * f0 * h * t) / h for h in (1, 2, 3))
    return 0.2 * y / np.abs(y).max()


@pytest.fixture(scope="module")
def y22():
    return synth(22050)


@pytest.fixture(scope="module")
def y44():
    return synth(44100, sec=10.0)


def test_names():
    assert len(L0_NAMES) == 69 and L0_NAMES[-1] == "raw_log_rms"
    assert len(L1_NAMES) == 66 and L1_NAMES[-2:] == ["phon_mean", "phon_p95"]


@NEEDS_WEB
def test_l2_is_xbar_times_weights(y22):
    b = load_bundle("001")
    xb = xbar(y22, b["stats"])[None]
    X = probe_design(F.extract(F.prepare(y22)), b["stats"])[TR_START:].mean(0)
    full = X @ b["W_region"]
    parts = l2(xb, b)
    assert parts["acoustic"].shape == (1, b["W_region"].shape[1])
    assert len(b["region_names"]) == b["W_region"].shape[1]
    assert np.allclose(parts["acoustic"] + parts["structural"], full, atol=1e-5)


def test_raw_rms_survives_normalization(y22):
    a, h = l0(y22), l0(0.5 * y22)
    assert a.shape == (69,)
    assert h[-1] < a[-1] and np.allclose(h[:68], a[:68], atol=1e-6)


def test_l1_level_sensitive(y44):
    a = l1(y44)
    assert a.shape == (66,) and np.isfinite(a).all()
    assert l1(0.1 * y44)[64] < a[64]


@NEEDS_WEB
def test_bundle_stats_identical_across_listeners():
    s = [load_bundle(f"00{i}")["stats"] for i in range(1, 6)]
    assert all(set(x) == set(s[0]) for x in s)


@NEEDS_WEB
def test_bundle_mismatch_fails(monkeypatch):
    real = levels._read_meta
    def fake(sub):
        m = real(sub)
        if sub == "003":
            m["bands"]["acoustic"]["mean"][0] += 1.0
        return m
    monkeypatch.setattr(levels, "_read_meta", fake)
    with pytest.raises(AssertionError):
        load_bundle("001")


def test_short_or_silent_excerpt_skipped(tmp_path):
    sil, short = tmp_path / "sil.wav", tmp_path / "short.wav"
    sf.write(sil, np.zeros(22050 * 12), 22050)
    sf.write(short, synth(22050, sec=8.0), 22050)
    assert process_one(sil) == ("skip", "silent")
    assert process_one(short) == ("skip", "shorter than 9 s")
    assert process_one(tmp_path / "nope.wav") == ("skip", "unreadable")


@NEEDS_WEB
def test_cli_end_to_end(tmp_path, monkeypatch):
    paths = []
    for i in range(3):
        p = tmp_path / f"{i + 10}.wav"
        sf.write(p, synth(22050, sec=10.0, seed=i), 22050)
        paths.append(p)
    sil = tmp_path / "13.wav"
    sf.write(sil, np.zeros(22050 * 10), 22050)
    paths.append(sil)
    df = pd.DataFrame({"song_id": [10, 11, 12, 13], "path": paths,
                       "artist": ["a", "b", "c", "d"], "group": ["a", "b", "c", "d"]})
    monkeypatch.setattr(levels, "excerpts", lambda: df)
    out = tmp_path / "out"
    monkeypatch.setattr(levels, "OUT", out)
    levels.main(["--workers", "1"])
    z = np.load(out / "levels.npz", allow_pickle=False)
    assert list(z["song_id"]) == [10, 11, 12] and z["L0"].shape == (3, 69)
    assert z["L1"].shape == (3, 66) and z["xbar"].shape == (3, 340)
    assert not np.isnan(z["L0"]).any() and not np.isnan(z["L1"]).any()
    assert pd.read_csv(out / "skipped.csv").to_dict("records") == [{"song_id": 13, "reason": "silent"}]
    mt = (out / "levels" / "10.npz").stat().st_mtime_ns
    smt = (out / "levels" / "13.skip").stat().st_mtime_ns
    calls = []
    monkeypatch.setattr(levels, "process_one", lambda p: calls.append(p) or ("ok", {}))
    levels.main(["--workers", "1"])  # resumable: nothing reprocessed
    assert calls == []
    assert (out / "levels" / "10.npz").stat().st_mtime_ns == mt
    assert (out / "levels" / "13.skip").stat().st_mtime_ns == smt
    assert pd.read_csv(out / "skipped.csv").to_dict("records") == [{"song_id": 13, "reason": "silent"}]
    assert list(np.load(out / "levels.npz")["song_id"]) == [10, 11, 12]


# ------------------------------------------------------------------ provenance and --limit

FP = dict(w_region_sha256="a" * 64, bands_sha256="b" * 64)


def _fake_bundle(web, sub="001", w=b"\x00\x01\x02\x03", bands=None):
    d = web / f"sub-{sub}"
    d.mkdir(parents=True, exist_ok=True)
    bands = bands or {"acoustic": {"mean": [0.0, 1.0], "std": [1.0, 2.0], "nuisance": [[0.5]]},
                      "structural": {"mean": [3.0], "std": [4.0], "nuisance": [[1.5]]}}
    (d / "meta.json").write_text(json.dumps({"sub": sub, "bands": bands, "regions": []}))
    (d / "w_region.f32").write_bytes(w)
    return d


def test_bundle_fingerprint(tmp_path):
    import hashlib
    web = tmp_path / "web"
    _fake_bundle(web)
    fp = levels.bundle_fingerprint("001", web)
    assert fp["w_region_sha256"] == hashlib.sha256(b"\x00\x01\x02\x03").hexdigest()
    assert set(fp) == {"w_region_sha256", "bands_sha256"}
    # canonical JSON: key order and other meta fields do not matter, band values do
    d = web / "sub-001"
    meta = json.loads((d / "meta.json").read_text())
    reordered = {"structural": meta["bands"]["structural"], "acoustic": meta["bands"]["acoustic"]}
    (d / "meta.json").write_text(json.dumps({"bands": reordered, "other": 1}, indent=2))
    assert levels.bundle_fingerprint("001", web) == fp
    meta["bands"]["acoustic"]["std"][0] = 1.5
    (d / "meta.json").write_text(json.dumps(meta))
    assert levels.bundle_fingerprint("001", web)["bands_sha256"] != fp["bands_sha256"]
    _fake_bundle(web, w=b"\x00\x01\x02\x04")
    assert levels.bundle_fingerprint("001", web)["w_region_sha256"] != fp["w_region_sha256"]


def test_check_provenance_warns(tmp_path, capsys):
    web = tmp_path / "web"
    _fake_bundle(web)
    lv = tmp_path / "levels.npz"
    assert levels.provenance_path(lv) == tmp_path / "levels_provenance.json"
    assert levels.check_provenance(lv, web) is None
    assert "WARNING" in capsys.readouterr().err
    levels.write_provenance(lv, levels.bundle_fingerprint("001", web))
    assert levels.check_provenance(lv, web)["bundle_fingerprint"] == levels.bundle_fingerprint("001", web)
    assert capsys.readouterr().err == ""
    _fake_bundle(web, w=bytes(4) + b"x")  # the bundle changed after extraction
    levels.check_provenance(lv, web)
    assert "different sub-001 bundle" in capsys.readouterr().err


def _fake_cli(tmp_path, monkeypatch, n=4, fp=FP):
    df = pd.DataFrame({"song_id": list(range(10, 10 + n)), "path": [tmp_path / f"{i}.wav" for i in range(n)],
                       "artist": list("abcdefgh"[:n]), "group": list("abcdefgh"[:n])})
    monkeypatch.setattr(levels, "excerpts", lambda: df)
    monkeypatch.setattr(levels, "process_one",
                        lambda p: ("ok", dict(l0=np.zeros(69), l1=np.ones(66), xbar=np.zeros(340))))
    monkeypatch.setattr(levels, "bundle_fingerprint", lambda sub, web=None: dict(fp))
    out = tmp_path / "out"
    monkeypatch.setattr(levels, "OUT", out)
    return out


def test_main_writes_provenance_sidecar(tmp_path, monkeypatch):
    out = _fake_cli(tmp_path, monkeypatch)
    levels.main(["--workers", "1"])
    side = json.loads((out / "levels_provenance.json").read_text(encoding="utf-8"))
    assert side == dict(levels_file="levels.npz", bundle_sub="001", bundle_fingerprint=FP)
    # a later run against a changed bundle refuses to reuse the stale per-song cache
    monkeypatch.setattr(levels, "bundle_fingerprint", lambda sub, web=None: dict(FP, bands_sha256="c" * 64))
    with pytest.raises(RuntimeError, match="stale"):
        levels.main(["--workers", "1"])


def test_limit_never_overwrites_real_aggregate(tmp_path, monkeypatch):
    out = _fake_cli(tmp_path, monkeypatch)
    out.mkdir(parents=True)
    (out / "levels.npz").write_bytes(b"real aggregate")
    (out / "skipped.csv").write_text("real skipped")
    levels.main(["--workers", "1", "--limit", "2"])
    assert (out / "levels.npz").read_bytes() == b"real aggregate"
    assert (out / "skipped.csv").read_text() == "real skipped"
    assert not (out / "levels_provenance.json").exists()
    z = np.load(out / "levels_limit2.npz")
    assert list(z["song_id"]) == [10, 11]
    assert (out / "skipped_limit2.csv").exists()
    assert json.loads((out / "levels_limit2_provenance.json").read_text())["levels_file"] == "levels_limit2.npz"
