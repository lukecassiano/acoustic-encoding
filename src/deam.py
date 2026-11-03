"""DEAM (Aljanaki, Yang & Soleymani 2017) loader. Ratings are gated behind a freeze.

Directory layout (verified against the real archives, 2026-11-02):

    data/deam/MEMD_audio/<song_id>.mp3
    data/deam/metadata/metadata_2013.csv, metadata_2014.csv      (2013: song_id,...,Artist; 2014: Id,Artist,... ragged)
    data/deam/annotations/annotations averaged per song/song_level/
        static_annotations_averaged_songs_1_2000.csv             (song_id, valence_mean, arousal_mean)

Inclusion: the 2013-2014 45 s excerpts only (song_id < 2000); the 2015 full songs are excluded.
Nothing here reads rating values unless load_ratings is given a freeze file with a go-ahead line.
"""

import csv
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


ID_ALIASES = ("song_id", "id")


def _artists() -> dict[int, str]:
    """song_id -> artist. csv-parsed so ragged rows (spilled last.fm tags) cannot misalign."""
    out: dict[int, str] = {}
    found = False
    for rel in METADATA_REL:
        p = DEAM / rel
        if not p.exists():
            continue
        found = True
        with open(p, newline="", encoding="utf-8-sig") as fh:
            rows = csv.reader(fh)
            header = [c.strip() for c in next(rows, [])]
            low = [c.lower() for c in header]
            i_id = next((low.index(a) for a in ID_ALIASES if a in low), None)
            i_art = low.index("artist") if "artist" in low else None
            if i_id is None or i_art is None:
                raise ValueError(f"{p} needs an id ({'/'.join(ID_ALIASES)}) and artist column; found {header}")
            for row in rows:
                if len(row) <= max(i_id, i_art):
                    continue
                sid, art = row[i_id].strip(), row[i_art].strip()
                if sid.isdigit() and art:
                    out[int(sid)] = art
    if not found:
        raise FileNotFoundError(f"no DEAM metadata file found under {DEAM}: {[str(r) for r in METADATA_REL]}")
    return out


def _merge_title(key: str, keys: set[str]) -> str:
    """Fold "artist-title" keys into an existing "artist" key; otherwise unchanged."""
    if "-" in key and not key.startswith("song-"):
        prefix = key.split("-", 1)[0].strip()
        if prefix in keys:
            return prefix
    return key


def excerpts() -> pd.DataFrame:
    """Included songs whose mp3 exists: song_id, path (absolute), artist, group."""
    artists = _artists()
    rows = []
    for p in (DEAM / AUDIO_REL).glob("*.mp3"):
        if not p.stem.isdigit() or int(p.stem) >= MAX_EXCERPT_ID:
            continue
        sid = int(p.stem)
        art = artists.get(sid, "")
        rows.append((sid, p.resolve(), art, " ".join(art.split()).lower() if art else f"song-{sid}"))
    df = pd.DataFrame(rows, columns=["song_id", "path", "artist", "group"])
    keys = set(df.group)
    df["group"] = [_merge_title(k, keys) for k in df.group]
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
