"""PMEmo 2019 (Zhang et al. 2018) loader for the H4 generalisation test. Ratings are gated behind the freeze.

Layout (data/ is gitignored; PMEmo audio is commercial and is never copied outside it):

    data/pmemo/chorus/<musicId>.mp3
    data/pmemo/metadata.csv                         (musicId,fileName,title,artist,album,duration,...; quoted commas)
    data/pmemo/annotations/static_annotations.csv   (musicId,Arousal(mean),Valence(mean))

All choruses with an mp3 are included. Nothing here reads rating values unless load_ratings is
given a freeze file with a go-ahead line (the same gate as src.deam).
"""

import csv
from pathlib import Path

import pandas as pd

from src.deam import _merge_title, require_goahead
from src.features import DATA

PMEMO = DATA / "pmemo"
AUDIO_REL = Path("chorus")
METADATA_REL = Path("metadata.csv")
STATIC_REL = Path("annotations") / "static_annotations.csv"


def _artists() -> dict[int, str]:
    p = PMEMO / METADATA_REL
    if not p.exists():
        raise FileNotFoundError(f"no PMEmo metadata file at {p}")
    out: dict[int, str] = {}
    with open(p, newline="", encoding="utf-8-sig") as fh:
        rows = csv.reader(fh)
        header = [c.strip().lower() for c in next(rows, [])]
        if "musicid" not in header or "artist" not in header:
            raise ValueError(f"{p} needs musicId and artist columns; found {header}")
        i_id, i_art = header.index("musicid"), header.index("artist")
        for row in rows:
            if len(row) <= max(i_id, i_art):
                continue
            sid, art = row[i_id].strip(), row[i_art].strip()
            if sid.isdigit() and art:
                out[int(sid)] = art
    return out


def excerpts() -> pd.DataFrame:
    """All choruses whose mp3 exists: song_id, path (absolute), artist, group (unused for fitting)."""
    artists = _artists()
    rows = []
    for p in (PMEMO / AUDIO_REL).glob("*.mp3"):
        if not p.stem.isdigit():
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


def load_ratings(freeze_path: Path) -> pd.DataFrame:
    """Static song-level valence/arousal (PMEmo's own scale). Refuses without a go-ahead line."""
    require_goahead(freeze_path)
    df = pd.read_csv(PMEMO / STATIC_REL, skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"musicId": "song_id", "Valence(mean)": "valence", "Arousal(mean)": "arousal"})
    out = df[["song_id", "valence", "arousal"]].copy()
    out["song_id"] = out["song_id"].astype(int)
    return out.drop_duplicates("song_id").sort_values("song_id").reset_index(drop=True)
