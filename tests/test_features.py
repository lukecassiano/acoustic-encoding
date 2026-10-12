"""Behavioural checks for the structural band of src/features.py.

Each test synthesizes a pair of 15 s additive-harmonic chord sequences that
differ along one musical dimension and asserts the matching feature moves the
right way. The same pairs are the reference stimuli for the browser port.

    python -m pytest tests/test_features.py
    python -m tests.test_features          # prints the feature values
"""

import time

import numpy as np

from src import features as F

N_TR = int(F.CLIP_SEC / F.TR)
NAMES = F.FEATURE_NAMES["structural"]


def tone(midi, dur, n_harmonics=6):
    """Harmonic complex with 1/k partials, 10 ms attack and exponential decay."""
    t = np.arange(int(dur * F.SR)) / F.SR
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, n_harmonics + 1))
    return y * np.minimum(t / 0.01, 1.0) * np.exp(-3.0 * t / dur)


def sequence(chords, onsets, dur=0.5):
    """Place chords (lists of MIDI notes) at onset times in a 15 s clip."""
    y = np.zeros(int(F.CLIP_SEC * F.SR))
    for notes, on in zip(chords, onsets):
        i = int(on * F.SR)
        seg = sum(tone(m, dur) for m in notes)[: len(y) - i]
        y[i: i + len(seg)] += seg
    return F.prepare(y)


def structural(chords, onsets):
    return dict(zip(NAMES, F.extract(sequence(chords, onsets))["structural"].T))


STEADY = np.arange(0, F.CLIP_SEC, 0.5)


def progression(triads):
    return [triads[i % len(triads)] for i in range(len(STEADY))]


def major_minor():
    # I-IV-V-I in C major against i-iv-v-i in C minor
    major = progression([[60, 64, 67], [65, 69, 72], [67, 71, 74], [60, 64, 67]])
    minor = progression([[60, 63, 67], [65, 68, 72], [67, 70, 74], [60, 63, 67]])
    return structural(major, STEADY), structural(minor, STEADY)


def consonant_cluster():
    open_fifth = progression([[60, 67, 72]])
    cluster = progression([[60, 61, 62]])
    return structural(open_fifth, STEADY), structural(cluster, STEADY)


def steady_jittered():
    rng = np.random.default_rng(0)
    jitter = STEADY + rng.uniform(-0.35, 0.35, len(STEADY)) * 0.5
    jitter = np.clip(jitter, 0, F.CLIP_SEC - 0.5)
    chords = progression([[60, 64, 67]])
    return structural(chords, STEADY), structural(chords, jitter)


def test_mode_higher_for_major():
    major, minor = major_minor()
    assert major["mode"].mean() > minor["mode"].mean()


def test_roughness_higher_for_cluster():
    consonant, cluster = consonant_cluster()
    assert cluster["roughness"].mean() > consonant["roughness"].mean()


def test_jitter_breaks_the_beat():
    steady, jittered = steady_jittered()
    assert jittered["beat_cv"].mean() > steady["beat_cv"].mean()
    assert jittered["pulse_clarity"].mean() < steady["pulse_clarity"].mean()


def test_real_clip_shapes():
    feats = F.extract(F.clip_audio("rock", 32, 4.48))
    assert feats["acoustic"].shape == (N_TR, len(F.FEATURE_NAMES["acoustic"]))
    assert feats["structural"].shape == (N_TR, len(NAMES))
    for x in feats.values():
        assert np.isfinite(x).all()


def main():
    pairs = [("major", "minor", major_minor()),
             ("open fifth", "cluster", consonant_cluster()),
             ("steady", "jittered", steady_jittered())]
    for a, b, (fa, fb) in pairs:
        print(f"\n{'':16s}{a:>12s}{b:>12s}")
        for name in NAMES:
            print(f"{name:16s}{fa[name].mean():12.4f}{fb[name].mean():12.4f}")

    t0 = time.perf_counter()
    feats = F.extract(F.clip_audio("rock", 32, 4.48))
    print(f"\nrock.00032@4.48: {time.perf_counter() - t0:.1f} s/clip")
    for band, x in feats.items():
        print(f"{band:11s} {x.shape}  finite={np.isfinite(x).all()}  "
              f"per-feature std min={x.std(0).min():.3g}")
    print("structural per TR:")
    print(np.array2string(feats["structural"], precision=3, suppress_small=True, max_line_width=140))


if __name__ == "__main__":
    main()
