import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src import deam, pmemo
from src.perception import levels
from tests.perception.test_levels import synth

HEADER = "musicId,fileName,title,artist,album,duration,chorus_start_time,chorus_end_time\n"


def layout(tmp_path):
    root = tmp_path / "pmemo"
    (root / pmemo.AUDIO_REL).mkdir(parents=True)
    for sid in (1, 4, 5, 6):
        (root / pmemo.AUDIO_REL / f"{sid}.mp3").write_bytes(b"x")
    (root / pmemo.AUDIO_REL / "notes.mp3").write_bytes(b"x")
    (root / pmemo.METADATA_REL).write_text(
        HEADER
        + '1,1.mp3,Good Drank,2 Chainz,"Def Jam Presents: Direct Deposit, Vol. 2",32.1,02:35,03:05\n'
        + '4,4.mp3,"Song, With Comma","Tyler, The Creator",Album,28.09,03:00,03:26\n'
        + "5,5.mp3,A,Tyler,B,1,00:01,00:02\n"
        + "6,6.mp3,B,Tyler-Extra,B,1,00:01,00:02\n", encoding="utf-8")
    ann = root / pmemo.STATIC_REL
    ann.parent.mkdir(parents=True)
    pd.DataFrame({"musicId": [1, 4, 4], "Arousal(mean)": [0.6, 0.5, 0.5],
                  "Valence(mean)": [0.7, 0.4, 0.4]}).to_csv(ann, index=False)
    return root


def test_excerpts_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(pmemo, "PMEMO", layout(tmp_path))
    df = pmemo.excerpts()
    assert list(df.columns) == ["song_id", "path", "artist", "group"]
    assert list(df.song_id) == [1, 4, 5, 6]
    assert df.path.map(lambda p: p.is_absolute()).all()
    assert list(df.artist) == ["2 Chainz", "Tyler, The Creator", "Tyler", "Tyler-Extra"]
    assert list(df.group) == ["2 chainz", "tyler, the creator", "tyler", "tyler"]


def test_ratings_refused_without_goahead(tmp_path, monkeypatch):
    monkeypatch.setattr(pmemo, "PMEMO", layout(tmp_path))
    f = tmp_path / "freeze.md"; f.write_text("# freeze\nno go-ahead yet\n")
    with pytest.raises(deam.FreezeError):
        pmemo.load_ratings(f)
    with pytest.raises(deam.FreezeError):
        pmemo.load_ratings(tmp_path / "missing.md")


def test_ratings_load_with_goahead_on_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(pmemo, "PMEMO", layout(tmp_path))
    f = tmp_path / "freeze.md"; f.write_text("Go-ahead: Luke, 2026-11-10\n")
    r = pmemo.load_ratings(f)
    assert list(r.columns) == ["song_id", "valence", "arousal"]
    assert list(r.song_id) == [1, 4] and list(r.valence) == [0.7, 0.4] and list(r.arousal) == [0.6, 0.5]


def test_shared_gate():
    assert pmemo.require_goahead is deam.require_goahead
    assert not hasattr(pmemo, "GOAHEAD")


def test_levels_cli_pmemo_end_to_end(tmp_path, monkeypatch):
    paths = []
    for i in range(2):
        p = tmp_path / f"{i + 20}.wav"
        sf.write(p, synth(22050, sec=10.0, seed=i), 22050)
        paths.append(p)
    sil = tmp_path / "22.wav"
    sf.write(sil, np.zeros(22050 * 10), 22050)
    paths.append(sil)
    df = pd.DataFrame({"song_id": [20, 21, 22], "path": paths, "artist": list("abc"), "group": list("abc")})
    monkeypatch.setattr(levels, "pmemo_excerpts", lambda: df)
    monkeypatch.setattr(levels, "excerpts", lambda: pytest.fail("deam excerpts used"))
    out = tmp_path / "out"
    monkeypatch.setattr(levels, "OUT", out)
    levels.main(["--workers", "1", "--dataset", "pmemo"])
    z = np.load(out / "pmemo_levels.npz", allow_pickle=False)
    assert list(z["song_id"]) == [20, 21] and z["L0"].shape == (2, 69) and z["L1"].shape == (2, 66)
    assert pd.read_csv(out / "pmemo_skipped.csv").to_dict("records") == [{"song_id": 22, "reason": "silent"}]
    assert (out / "pmemo_levels" / "20.npz").exists() and (out / "pmemo_levels" / "22.skip").exists()
    assert not (out / "levels.npz").exists() and not (out / "levels").exists() and not (out / "skipped.csv").exists()
