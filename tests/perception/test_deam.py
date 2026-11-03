import wave

import numpy as np
import pandas as pd
import pytest

from src import deam
from src.deam import DEAM_AUDIO, FreezeError, excerpts, load_excerpt, load_ratings


def make_wav(tmp_path, sr=48000, channels=2, sampwidth=1, sec=1.0):
    p = tmp_path / "t.wav"
    t = np.arange(int(sr * sec)) / sr
    x = (127 + 60 * np.sin(2 * np.pi * 440 * t)).astype(np.uint8)
    data = np.repeat(x[:, None], channels, axis=1).tobytes()
    with wave.open(str(p), "wb") as w:
        w.setnchannels(channels); w.setsampwidth(sampwidth); w.setframerate(sr)
        w.writeframes(data)
    return p


def layout(tmp_path):
    root = tmp_path / "deam"
    (root / deam.AUDIO_REL).mkdir(parents=True)
    for sid in (3, 4, 5, 1999, 2000, 2001):
        (root / deam.AUDIO_REL / f"{sid}.mp3").write_bytes(b"x")
    meta = root / deam.METADATA_REL[0]
    meta.parent.mkdir(parents=True)
    pd.DataFrame({"song_id": [3, 4], "Artist": ["  The Band ", None]}).to_csv(meta, index=False)
    static = root / deam.STATIC_REL[0]
    static.parent.mkdir(parents=True)
    pd.DataFrame({"song_id": [3, 4, 2001], " valence_mean": [5.0, 4.0, 3.0],
                  " arousal_mean": [6.0, 5.0, 4.0], "valence_std": [1, 1, 1]}
                 ).to_csv(static, index=False)
    return root


@pytest.mark.skipif(not DEAM_AUDIO.exists(), reason="DEAM not downloaded")
def test_excerpts_schema_and_inclusion():
    df = excerpts()
    assert df.song_id.max() < 2000 and df.group.notna().all() and len(df) > 1500


def test_excerpts_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(deam, "DEAM", layout(tmp_path))
    df = excerpts()
    assert list(df.columns) == ["song_id", "path", "artist", "group"]
    assert list(df.song_id) == [3, 4, 5, 1999]
    assert df.path.map(lambda p: p.is_absolute()).all()
    assert list(df.group) == ["the band", "song-4", "song-5", "song-1999"]


def test_load_excerpt_mono_both_rates(tmp_path):
    y22, y44 = load_excerpt(make_wav(tmp_path, sr=48000, channels=2, sampwidth=1))
    assert y22.ndim == y44.ndim == 1 and abs(len(y44) / len(y22) - 2) < 0.01
    assert y22.dtype == y44.dtype == np.float64


def test_ratings_refused_without_goahead(tmp_path):
    f = tmp_path / "freeze.md"; f.write_text("# freeze\nno go-ahead yet\n")
    with pytest.raises(FreezeError):
        load_ratings(f)


def test_ratings_load_with_goahead_on_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(deam, "DEAM", layout(tmp_path))
    f = tmp_path / "freeze.md"; f.write_text("Go-ahead: Luke, 2026-11-10\n")
    r = load_ratings(f)
    assert list(r.columns) == ["song_id", "valence", "arousal"]
    assert list(r.song_id) == [3, 4]


def test_excerpts_raises_on_metadata_without_artist_column(tmp_path, monkeypatch):
    root = layout(tmp_path)
    pd.DataFrame({"song_id": [3], "title": ["x"]}).to_csv(root / deam.METADATA_REL[0], index=False)
    monkeypatch.setattr(deam, "DEAM", root)
    with pytest.raises(ValueError, match="artist"):
        excerpts()


def test_excerpts_raises_when_no_metadata_file(tmp_path, monkeypatch):
    root = layout(tmp_path)
    (root / deam.METADATA_REL[0]).unlink()
    monkeypatch.setattr(deam, "DEAM", root)
    with pytest.raises(FileNotFoundError):
        excerpts()
