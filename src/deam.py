"""DEAM (Aljanaki, Yang & Soleymani 2017) loader. Ratings are gated behind a freeze.

The directory layout below is the DOCUMENTED one, not yet verified against the real
archives; confirm it when the archives arrive (each path is a one-line change):

    data/deam/MEMD_audio/<song_id>.mp3
    data/deam/metadata/metadata_2013.csv, metadata_2014.csv      (song_id, Artist, ...)
    data/deam/annotations/annotations averaged per song/song_level/
        static_annotations_averaged_songs_1_2000.csv             (song_id, valence_mean, arousal_mean)

Inclusion: the 2013-2014 45 s excerpts only (song_id < 2000); the 2015 full songs are excluded.
Nothing here reads rating values unless load_ratings is given a freeze file with a go-ahead line.
"""

import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from src.features import DATA

DEAM = DATA / "deam"
AUDIO_REL = Path("MEMD_audio")
METADATA_REL = [Path("metadata") / "metadata_2013.csv", Path("metadata") / "metadata_2014.csv"]
STATIC_REL = [Path("annotations") / "annotations averaged per song" / "song_level"
              / "static_annotations_averaged_songs_1_2000.csv"]
DEAM_AUDIO = DEAM / AUDIO_REL

MAX_EXCERPT_ID = 2000  # exclusive
GOAHEAD = re.compile(r"^Go-ahead: Luke, \d{4}-\d{2}-\d{2}", re.MULTILINE)


class FreezeError(Exception):
    """Ratings requested without a signed go-ahead in the freeze file."""


def _artists() -> dict[int, str]:
    out: dict[int, str] = {}
    found = False
    for rel in METADATA_REL:
        p = DEAM / rel
        if not p.exists():
            continue
        found = True
        df = pd.read_csv(p, skipinitialspace=True)
        cols = {c.strip().lower(): c for c in df.columns}
        if "song_id" not in cols or "artist" not in cols:
            raise ValueError(f"{p} needs song_id and artist columns; found {list(df.columns)}")
        for sid, art in zip(df[cols["song_id"]], df[cols["artist"]]):
            if pd.notna(sid) and pd.notna(art) and str(art).strip():
                out[int(sid)] = str(art).strip()
    if not found:
        raise FileNotFoundError(f"no DEAM metadata file found under {DEAM}: {[str(r) for r in METADATA_REL]}")
    return out


def excerpts() -> pd.DataFrame:
    """Included songs whose mp3 exists: song_id, path (absolute), artist, group."""
    artists = _artists()
    rows = []
    for p in (DEAM / AUDIO_REL).glob("*.mp3"):
        if not p.stem.isdigit() or int(p.stem) >= MAX_EXCERPT_ID:
            continue
        sid = int(p.stem)
        art = artists.get(sid, "")
        rows.append((sid, p.resolve(), art, art.lower().strip() if art else f"song-{sid}"))
    df = pd.DataFrame(rows, columns=["song_id", "path", "artist", "group"])
    df = df.sort_values("song_id").reset_index(drop=True)
    df["song_id"] = df["song_id"].astype(int)
    return df


def load_excerpt(path) -> tuple[np.ndarray, np.ndarray]:
    """(y22, y44): mono float64 at 22,050 and 44,100 Hz; no fade, no normalization."""
    y, sr = librosa.load(str(path), sr=None, mono=True)
    y = y.astype(np.float64)
    return (librosa.resample(y, orig_sr=sr, target_sr=22050),
            librosa.resample(y, orig_sr=sr, target_sr=44100))


def load_ratings(freeze_path: Path) -> pd.DataFrame:
    """Static song-level valence/arousal. Refuses unless the freeze file has a go-ahead line."""
    text = Path(freeze_path).read_text(encoding="utf-8") if Path(freeze_path).exists() else ""
    if not GOAHEAD.search(text):
        raise FreezeError(f"no 'Go-ahead: Luke, YYYY-MM-DD' line in {freeze_path}; ratings stay sealed")
    frames = []
    for rel in STATIC_REL:
        df = pd.read_csv(DEAM / rel, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns={"valence_mean": "valence", "arousal_mean": "arousal"})
        frames.append(df[["song_id", "valence", "arousal"]])
    out = pd.concat(frames, ignore_index=True)
    out["song_id"] = out["song_id"].astype(int)
    out = out[out.song_id < MAX_EXCERPT_ID].drop_duplicates("song_id")
    return out.sort_values("song_id").reset_index(drop=True)
