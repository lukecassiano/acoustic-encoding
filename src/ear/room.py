"""Speaker -> air -> head: the physical front end of the reference ear model.

    L, R = render_binaural([Source(sig, azimuth_deg=40, distance_m=2.0, preset="laptop")])

Per source: speaker colouration (RBJ biquads), 1/r gain and r/c delay, then the Brown & Duda (1998)
spherical head: a one-pole/one-zero shadow filter per ear (bilinear transform, prewarped at
w0 = c/a) plus an ITD delay. Azimuth: 0 = straight ahead, +90 = listener's left. Only plain
numpy/scipy filters and linear-interpolated delays, so the same arithmetic ports to JS.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from .constants import (ALPHA_MIN, C_SOUND, DIRECTIVITY_FLOOR, EAR_SR, HEAD_RADIUS, MAX_DIST,
                        MIN_DIST, PRESETS, REF_DIST, THETA_MIN_DEG)


@dataclass
class Source:
    signal: np.ndarray
    azimuth_deg: float
    distance_m: float
    preset: str = "studio"
    gain_db: float = 0.0


def _rbj(kind, f0, q, db, sr):
    """RBJ Audio EQ Cookbook biquad -> (b, a) normalised by a0."""
    w0 = 2 * np.pi * f0 / sr
    cw, alpha = np.cos(w0), np.sin(w0) / (2 * q)
    if kind == "highpass":
        b, a = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2], [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "lowpass":
        b, a = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2], [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peaking":
        A = 10 ** (db / 40)
        b, a = [1 + alpha * A, -2 * cw, 1 - alpha * A], [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        raise ValueError(f"unknown biquad {kind}")
    return np.array(b) / a[0], np.array(a) / a[0]


def speaker_filter(x, preset, sr):
    """Apply the preset's biquad chain; an empty chain returns x untouched."""
    for kind, f0, q, db in PRESETS[preset]["biquads"]:
        b, a = _rbj(kind, f0, q, db, sr)
        x = lfilter(b, a, x)
    return x


def directivity(preset, off_axis_deg):
    """Speaker gain g(phi) = max(floor, (1 - beta) + beta cos phi); phi = 0 faces the listener."""
    beta = PRESETS[preset]["beta"]
    return float(max(DIRECTIVITY_FLOOR, (1 - beta) + beta * np.cos(np.radians(off_axis_deg))))


def _ear_theta(azimuth_deg):
    """Angle (rad, 0..pi) between the source direction and each ear axis (left +90, right -90)."""
    def wrap(d):
        d = abs((d + 180.0) % 360.0 - 180.0)
        return np.radians(d)
    return wrap(azimuth_deg - 90.0), wrap(azimuth_deg + 90.0)


def _ear_delay(theta):
    """Brown-Duda ear delay (s), offset by a/c so it is non-negative."""
    a_c = HEAD_RADIUS / C_SOUND
    t = -a_c * np.cos(theta) if theta < np.pi / 2 else a_c * (theta - np.pi / 2)
    return t + a_c


def head_itd_s(azimuth_deg):
    """Right-ear minus left-ear arrival time; positive when the source is on the left."""
    th_l, th_r = _ear_theta(azimuth_deg)
    return float(_ear_delay(th_r) - _ear_delay(th_l))


def _shadow(x, theta, sr):
    """H(w) = (1 + j alpha w/2w0)/(1 + j w/2w0), bilinear with prewarping at w0."""
    w0 = C_SOUND / HEAD_RADIUS
    alpha = (1 + ALPHA_MIN / 2) + (1 - ALPHA_MIN / 2) * np.cos(theta / np.radians(THETA_MIN_DEG) * np.pi)
    tau = 1 / (2 * w0)
    k = w0 / np.tan(w0 / (2 * sr))
    b = np.array([1 + alpha * tau * k, 1 - alpha * tau * k])
    a = np.array([1 + tau * k, 1 - tau * k])
    return lfilter(b / a[0], a / a[0], x)


def _delay(x, seconds, n_out, sr):
    """Fractional delay by linear interpolation, zero-padded/truncated to n_out samples."""
    d = seconds * sr
    i = int(np.floor(d))
    f = d - i
    out = np.zeros(n_out)
    for shift, w in ((i, 1 - f), (i + 1, f)):
        if w and shift < n_out:
            m = min(len(x), n_out - shift)
            out[shift:shift + m] += w * x[:m]
    return out


def render_binaural(sources, sr=EAR_SR):
    """Sum of all sources at the two ear-canal entrances -> (left, right).

    Output length = longest input + the largest delay a source can have (head ITD plus MAX_DIST
    of air), so separate renders superpose sample-for-sample. The bulk propagation delay is
    rounded to whole samples (linear interpolation low-passes noise by a distance-dependent
    amount, which would corrupt the 1/r law); only the head's ITD delay is fractional.
    """
    n_in = max(len(s.signal) for s in sources)
    max_delay = MAX_DIST / C_SOUND + HEAD_RADIUS / C_SOUND * (np.pi / 2 + 1)
    n_out = n_in + int(np.ceil(max_delay * sr)) + 1
    left, right = np.zeros(n_out), np.zeros(n_out)
    for s in sources:
        d = max(s.distance_m, MIN_DIST)
        sig = speaker_filter(np.asarray(s.signal, float), s.preset, sr)
        sig = sig * 10 ** (s.gain_db / 20) * REF_DIST / d
        for out, th in zip((left, right), _ear_theta(s.azimuth_deg)):
            out += _delay(_shadow(sig, th, sr), round(d / C_SOUND * sr) / sr + _ear_delay(th), n_out, sr)
    return left, right
