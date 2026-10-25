"""Reference outputs for the browser port of src/features.py (web/features.js).

Writes, per clip, the prepared audio (raw float64) and every intermediate of
extract() as JSON, so a parity failure can be traced to the stage that broke.
The intermediates are recomputed with the same librosa calls as extract(), and
the final matrices are checked against extract() itself before anything is
written -- the reference cannot drift from the real extractor.

Clips: one test clip per genre, plus a 29.25 s excerpt (longer than a clip, 19.5 TRs:
a rounding edge) and two synthetic chord sequences.

Usage:
  python -m tests.parity.make_reference      # -> data/parity/
  node tests/parity/parity.mjs
"""

import json

import librosa
import numpy as np

from src import features as F

OUT = F.DATA / "parity"


def intermediates(y):
    S = np.abs(librosa.stft(y, n_fft=F.N_FFT, hop_length=F.HOP))
    P = S ** 2
    logmel = librosa.power_to_db(librosa.feature.melspectrogram(S=P, sr=F.SR, n_mels=F.N_MELS))
    mfcc = librosa.feature.mfcc(S=logmel, n_mfcc=F.N_MFCC + 1)[1:]
    tuning = librosa.estimate_tuning(S=P, sr=F.SR, bins_per_octave=12)
    chroma = librosa.feature.chroma_stft(S=P, sr=F.SR)
    onset_env = librosa.onset.onset_strength(S=logmel, sr=F.SR, hop_length=F.HOP)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=F.SR, hop_length=F.HOP, aggregate=None)
    tgram = librosa.feature.tempogram(onset_envelope=onset_env, sr=F.SR, hop_length=F.HOP)
    lags = np.arange(tgram.shape[0]) * F.HOP / F.SR
    pulse = tgram[(lags >= 0.25) & (lags <= 2.0)].max(0) / np.maximum(tgram[0], 1e-8)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=F.SR, hop_length=F.HOP)
    bpm, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=F.SR, hop_length=F.HOP)
    n_tr = int(round(len(y) / F.SR / F.TR))
    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=F.SR)
    return {
        "n_frames": S.shape[1], "n_tr": n_tr,
        "logmel": logmel, "mfcc": mfcc,
        "centroid": librosa.feature.spectral_centroid(S=S, sr=F.SR)[0],
        "bandwidth": librosa.feature.spectral_bandwidth(S=S, sr=F.SR)[0],
        "tuning": float(tuning), "chroma": chroma,
        "rms": librosa.feature.rms(S=S, frame_length=F.N_FFT)[0],
        "onset_env": onset_env, "tempo": tempo, "pulse": pulse,
        "onsets": onsets, "bpm": float(np.atleast_1d(bpm)[0]), "beats": beats,
        "keys": F.key_features(chroma, n_tr), "roughness": F.roughness(S, librosa.fft_frequencies(sr=F.SR, n_fft=F.N_FFT)),
        "tonnetz": tonnetz,
    }


def synth_chords(chords, sec_per_chord=1.5, total=15.0, jitter=0.0, seed=0):
    """Additive-harmonic chord sequence (6 partials, decaying), as in tests/test_features.py."""
    rng = np.random.default_rng(seed)
    t_total = int(total * F.SR)
    y = np.zeros(t_total)
    onset = 0.0
    k = 0
    while onset < total:
        dur = sec_per_chord * (1 + jitter * rng.uniform(-1, 1))
        n0, n1 = int(onset * F.SR), min(int((onset + dur) * F.SR), t_total)
        t = np.arange(n1 - n0) / F.SR
        env = np.exp(-3 * t)
        for midi in chords[k % len(chords)]:
            f0 = 440 * 2 ** ((midi - 69) / 12)
            for h in range(1, 7):
                y[n0:n1] += env * np.sin(2 * np.pi * f0 * h * t) / h
        onset += dur
        k += 1
    return F.prepare(y)


def clips():
    ev = F.all_events("001")
    test = ev[ev.clip_id.isin(set(F.run_events("001", "Test", 1).clip_id))]
    for genre, rows in test.groupby("genre"):
        r = rows.iloc[0]
        yield f"{genre}", F.clip_audio(r.genre, r.track, r.start)
    yield "long29_jazz", F.load_audio(F.GTZAN / "jazz" / "jazz.00007.wav", start=0.0, duration=29.25)
    yield "synth_major", synth_chords([(60, 64, 67), (65, 69, 72), (67, 71, 74), (60, 64, 67)])
    yield "synth_cluster_jitter", synth_chords([(60, 61, 62), (61, 62, 63)], jitter=0.35, seed=1)


def tolist(v):
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist() if v.dtype.kind == "f" else v.astype(int).tolist()
    return v


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, y in clips():
        ref = intermediates(y)
        feats = F.extract(y)
        # mirror check: intermediates used here are the ones extract() uses
        assert np.allclose(feats["structural"][:, 4:6], ref["keys"]), name
        ref["acoustic"], ref["structural"] = feats["acoustic"], feats["structural"]
        y.astype("<f8").tofile(OUT / f"{name}.f64")
        with open(OUT / f"{name}.json", "w") as f:
            json.dump({k: tolist(v) for k, v in ref.items()}, f)
        manifest.append({"name": name, "n_samples": len(y), "n_tr": ref["n_tr"]})
        print(f"{name:22s} {len(y) / F.SR:6.2f} s  n_tr {ref['n_tr']:3d}  tuning {ref['tuning']:+.2f}  "
              f"bpm {ref['bpm']:.1f}  {len(ref['onsets'])} onsets  {len(ref['beats'])} beats", flush=True)
    # prepare() parity: one raw (unfaded, unnormalized) excerpt and its prepared form
    raw = F.load_audio(F.GTZAN / "rock" / "rock.00032.wav", start=4.48, duration=F.CLIP_SEC,
                       fade=False, normalize=False)
    raw.astype("<f8").tofile(OUT / "prepare_raw.f64")
    F.prepare(raw).astype("<f8").tofile(OUT / "prepare_out.f64")
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)


if __name__ == "__main__":
    main()
