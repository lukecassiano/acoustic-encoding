"""Audio -> feature matrices on the fMRI TR grid. Never touches brain data.

One extractor for dataset clips and probe stimuli alike:

    y = load_audio("probe.wav")               # 22,050 Hz mono, faded, RMS-normalized
    feats = extract(y)                        # {"acoustic": (n_tr, 60), "structural": (n_tr, 8)}

Every feature is a plain spectral computation (STFT, mel/chroma filterbanks,
autocorrelation, template correlation, pairwise partial roughness) so it can be
ported to the browser and checked for parity against this implementation.

Bands
  acoustic    log-mel (32), MFCC 1-13, spectral centroid + bandwidth (log Hz),
              chroma (12), log RMS
  structural  log2 local tempo, pulse clarity, onset rate, beat-interval CV,
              key clarity, mode (major - minor), sensory roughness, harmonic change

Dataset usage
  python -m src.features --sub 001        # caches every clip, then per-run matrices
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BIDS = DATA / "ds003720"
GTZAN = DATA / "gtzan" / "genres"
CACHE = DATA / "features"

SR = 22050
N_FFT = 2048
HOP = 512
TR = 1.5
CLIP_SEC = 15.0
FADE_SEC = 2.0
TARGET_RMS = 0.13210295550519335  # Mean_RMS.mat: RMS of every presented stimulus

N_MELS = 32
N_MFCC = 13

# Krumhansl-Kessler key profiles (C major / C minor), rotated for the other 11 keys
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

PITCH = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
FEATURE_NAMES = {
    "acoustic": ([f"mel_{i:02d}" for i in range(N_MELS)]
                 + [f"mfcc_{i:02d}" for i in range(1, N_MFCC + 1)]
                 + ["centroid", "bandwidth"]
                 + [f"chroma_{p}" for p in PITCH]
                 + ["rms"]),
    "structural": ["tempo", "pulse_clarity", "onset_rate", "beat_cv",
                   "key_clarity", "mode", "roughness", "harmonic_change"],
}
BANDS = list(FEATURE_NAMES)


# ------------------------------------------------------------------ audio in

def load_audio(path, start=0.0, duration=None, fade=True, normalize=True):
    """Load mono 22,050 Hz audio, optionally cut, faded and RMS-normalized.

    Defaults reproduce the stimulus preparation in GTZAN_Preprocess.py
    (2 s sine fade in/out, RMS scaled to the stimulus-set mean), so probes
    enter the model on the same footing as the training clips.
    """
    import librosa
    y, _ = librosa.load(path, sr=SR, mono=True, offset=start, duration=duration)
    return prepare(y, fade=fade, normalize=normalize)


def prepare(y, fade=True, normalize=True):
    y = np.asarray(y, dtype=np.float64).copy()
    if fade:
        n = min(int(FADE_SEC * SR), len(y) // 2)
        ramp = np.sin(np.linspace(0, np.pi / 2, n))
        y[:n] *= ramp
        y[len(y) - n:] *= ramp[::-1]
    if normalize:
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 0:
            y *= TARGET_RMS / rms
    return y


# ------------------------------------------------------------- frame features

def _frames_to_trs(x, n_tr):
    """Mean-pool frame-rate features (frames, F) into TR bins (n_tr, F)."""
    times = np.arange(x.shape[0]) * HOP / SR  # librosa centers frame t at t * HOP
    idx = np.minimum((times // TR).astype(int), n_tr - 1)
    out = np.zeros((n_tr, x.shape[1]))
    for t in range(n_tr):
        sel = idx == t
        if sel.any():
            out[t] = x[sel].mean(0)
    return out


def _key_profiles():
    major = np.stack([np.roll(KK_MAJOR, k) for k in range(12)])
    minor = np.stack([np.roll(KK_MINOR, k) for k in range(12)])
    z = lambda p: (p - p.mean(1, keepdims=True)) / p.std(1, keepdims=True)
    return z(major), z(minor)


def key_features(chroma, n_tr, context_tr=1):
    """Krumhansl-Schmuckler key finding per TR on a (2*context+1)-TR window.

    key_clarity = best correlation with any of the 24 key profiles
    mode        = best major correlation - best minor correlation (>0 major-like)
    """
    major, minor = _key_profiles()
    pooled = _frames_to_trs(chroma.T, n_tr)
    out = np.zeros((n_tr, 2))
    for t in range(n_tr):
        c = pooled[max(0, t - context_tr): t + context_tr + 1].sum(0)
        if c.std() == 0:
            continue
        c = (c - c.mean()) / c.std()
        rmaj, rmin = major @ c / 12, minor @ c / 12
        out[t] = [max(rmaj.max(), rmin.max()), rmaj.max() - rmin.max()]
    return out


def roughness(S, freqs, n_peaks=20):
    """Sethares (1993) parameterization of Plomp-Levelt sensory dissonance.

    Per frame: take the strongest spectral peaks, sum pairwise dissonance
    a_i a_j (e^{-3.5 s df} - e^{-5.75 s df}) with s = 0.24 / (0.021 f_min + 19),
    and divide by total peak energy so the measure tracks interval structure
    rather than loudness.
    """
    from scipy.signal import find_peaks
    out = np.zeros(S.shape[1])
    for t in range(S.shape[1]):
        mag = S[:, t]
        pk, _ = find_peaks(mag)
        if len(pk) < 2:
            continue
        pk = pk[np.argsort(mag[pk])[-n_peaks:]]
        f, a = freqs[pk], mag[pk]
        fi, fj = np.meshgrid(f, f)
        ai, aj = np.meshgrid(a, a)
        s = 0.24 / (0.021 * np.minimum(fi, fj) + 19)
        df = np.abs(fi - fj)
        d = ai * aj * (np.exp(-3.5 * s * df) - np.exp(-5.75 * s * df))
        out[t] = np.triu(d, 1).sum() / (a ** 2).sum()
    return out


def extract(y, sr=SR):
    """Audio -> {band: (n_tr, n_features)} on the 1.5 s TR grid."""
    import librosa
    assert sr == SR, "resample to 22,050 Hz first (load_audio does this)"
    n_tr = int(round(len(y) / SR / TR))

    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    P = S ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)

    # acoustic
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=SR, n_mels=N_MELS))
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=N_MFCC + 1)[1:]
    centroid = librosa.feature.spectral_centroid(S=S, sr=SR)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=SR)
    chroma = librosa.feature.chroma_stft(S=P, sr=SR)
    rms = librosa.feature.rms(S=S, frame_length=N_FFT)
    acoustic = np.vstack([logmel, mfcc, np.log(centroid + 1), np.log(bandwidth + 1),
                          chroma, np.log(rms + 1e-8)]).T

    # structural: rhythm
    onset_env = librosa.onset.onset_strength(S=logmel, sr=SR, hop_length=HOP)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=SR, hop_length=HOP, aggregate=None)
    tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    lags = np.arange(tgram.shape[0]) * HOP / SR
    in_range = (lags >= 0.25) & (lags <= 2.0)  # 30-240 BPM
    pulse = tgram[in_range].max(0) / np.maximum(tgram[0], 1e-8)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    onset_train = np.zeros(onset_env.shape[0])
    onset_train[onsets] = SR / HOP  # pooled mean -> onsets per second
    _, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=SR, hop_length=HOP)
    ibi = np.diff(beats) * HOP / SR
    beat_cv = ibi.std() / ibi.mean() if len(ibi) > 2 else 0.0

    # structural: harmony
    keys = key_features(chroma, n_tr)
    rough = roughness(S, freqs)
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=SR)
    tonnetz = np.apply_along_axis(lambda v: np.convolve(v, np.ones(9) / 9, mode="same"), 1, tonnetz)
    hcdf = np.r_[0, np.linalg.norm(np.diff(tonnetz, axis=1), axis=0)]

    n = min(S.shape[1], len(tempo), len(pulse), len(onset_train))
    frame_struct = np.vstack([np.log2(tempo[:n]), pulse[:n], onset_train[:n],
                              rough[:n], hcdf[:n]]).T
    fs = _frames_to_trs(frame_struct, n_tr)
    structural = np.column_stack([fs[:, 0], fs[:, 1], fs[:, 2], np.full(n_tr, beat_cv),
                                  keys[:, 0], keys[:, 1], np.log(fs[:, 3] + 1e-6), fs[:, 4]])

    return {"acoustic": _frames_to_trs(acoustic, n_tr), "structural": structural}


# -------------------------------------------------------------- dataset glue

def clip_id(genre, track, start):
    return f"{genre}.{int(track):05d}@{round(float(start), 2)}"


def run_events(sub, task, run):
    f = BIDS / f"sub-{sub}" / "func" / f"sub-{sub}_task-{task}_run-{run:02d}_events.tsv"
    ev = pd.read_csv(f, sep="\t")
    ev["genre"] = ev["genre"].str.strip("'")
    ev["clip_id"] = [clip_id(g, t, s) for g, t, s in zip(ev.genre, ev.track, ev.start)]
    return ev


def all_events(sub):
    func = BIDS / f"sub-{sub}" / "func"
    return pd.concat([run_events(sub, task, int(p.name.split("run-")[1][:2]))
                      for task in ("Training", "Test")
                      for p in sorted(func.glob(f"sub-{sub}_task-{task}_run-*_events.tsv"))])


def clip_audio(genre, track, start):
    path = GTZAN / genre / f"{genre}.{int(track):05d}.wav"
    return load_audio(path, start=float(start), duration=CLIP_SEC)


def _extract_clip(row):
    genre, track, start = row
    return clip_id(genre, track, start), extract(clip_audio(genre, track, start))


def build_clip_cache(sub, workers=4):
    """Extract every unique stimulus once -> data/features/clips.npz."""
    from concurrent.futures import ProcessPoolExecutor
    CACHE.mkdir(parents=True, exist_ok=True)
    target = CACHE / "clips.npz"
    if target.exists():
        return dict(np.load(target))
    ev = all_events(sub).drop_duplicates("clip_id")
    rows = list(zip(ev.genre, ev.track, ev.start))
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (cid, feats) in enumerate(pool.map(_extract_clip, rows, chunksize=4)):
            for band, x in feats.items():
                out[f"{band}/{cid}"] = x.astype(np.float32)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(rows)} clips", flush=True)
    np.savez_compressed(target, **out)
    return out


def run_features(sub, task, run, cache):
    """Full-run feature timeline (41 clips x 10 TRs = 410 rows) per band.

    Includes the repeated first clip: the response matrices drop its 10 TRs,
    but hemodynamic delays mean the next clip's TRs still see its features.
    """
    ev = run_events(sub, task, run)
    return {band: np.concatenate([cache[f"{band}/{c}"] for c in ev.clip_id]) for band in BANDS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cache = build_clip_cache(a.sub, a.workers)
    n = len({k.split("/", 1)[1] for k in cache})
    print(f"{n} clips cached; bands: "
          + ", ".join(f"{b} {cache[next(k for k in cache if k.startswith(b))].shape}" for b in BANDS))


if __name__ == "__main__":
    main()
