"""Outer/middle ear, cochlea, hair cells and loudness: the auditory periphery of the reference ear model.

    out = cochlea(ear_signal)   # CochleaOut(cfs, bm, hc, env, loud)

A 32-channel ERB-spaced gammatone bank (cascade of complex one-pole filters at baseband, so every
coefficient is explicit and ports to JS) gives the basilar-membrane output `bm`. Hair cells half-wave
rectify, compress (x^0.3) and low-pass at 1 kHz (phase locking fades): `hc` at full rate (for the
interaural cross-correlation), `env` as 10 ms frame means. Loudness: each channel's frame power is weighted
by the inverted 40-phon ISO 226 contour (outer + middle ear), compressed, summed over channels (sone), and
converted to phon. The compressive power law is the physics of Weber-Fechner / Stevens: perceived magnitude
grows as a power of intensity, so +10 dB doubles sone.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, lfilter

from .constants import (EAR_SR, F_HI, F_LO, FRAME_S, GT_B, GT_ORDER, HC_EXP, HC_LP_HZ,
                        ISO226_AF_FLOOR, ISO226_AF_OFFSET, ISO226_AF_SCALE, ISO226_ALPHA, ISO226_F,
                        ISO226_L_REF, ISO226_LU, ISO226_PHON_REF, ISO226_TF, LOUD_EXP, LOUD_K, N_CH,
                        PHON_LOW_EPS, PHON_LOW_EXP, PHON_LOW_SCALE, PHON_REF)


def _erb(f):
    """Glasberg & Moore equivalent rectangular bandwidth (Hz)."""
    return 24.7 * (4.37 * np.asarray(f, float) / 1000 + 1)


def erb_space(n=N_CH, lo=F_LO, hi=F_HI):
    """n centre frequencies (Hz) equally spaced on the ERB-number scale between lo and hi."""
    e = lambda f: 21.4 * np.log10(4.37 * f / 1000 + 1)
    return (10 ** (np.linspace(e(lo), e(hi), n) / 21.4) - 1) * 1000 / 4.37


def iso226_contour(phon, freqs):
    """Equal-loudness contour (dB SPL) of `phon` at `freqs`, ISO 226:2003 section 4.1.

    Table 1 parameters (alpha_f, L_U, T_f at the 29 standard frequencies, 20 Hz-12.5 kHz) as published in
    ISO 226:2003 (Acoustics - Normal equal-loudness-level contours), reproduced in Suzuki & Takeshima
    (2004) and the standard's Wikipedia summary; interpolated in log-frequency between the table rows.
    """
    lf, lt = np.log(np.asarray(freqs, float)), np.log(ISO226_F)
    a, lu, tf = (np.interp(lf, lt, p) for p in (ISO226_ALPHA, ISO226_LU, ISO226_TF))
    af = ISO226_AF_SCALE * (10 ** (0.025 * phon) - ISO226_AF_OFFSET) + (ISO226_AF_FLOOR * 10 ** ((tf + lu) / 10 - 9)) ** a
    return 10 / a * np.log10(af) - lu + ISO226_L_REF


def gammatone(x, cfs, sr):
    """Gammatone bank -> (len(cfs), len(x)), real, unit gain at each cf.

    Per channel: shift down by cf, GT_ORDER complex one-poles (pole exp(-2 pi GT_B ERB / sr), unit DC gain),
    shift back, take 2*Re (the factor 2 restores the half of the energy that sits at -cf).
    """
    x = np.asarray(x, float)
    n = np.arange(len(x))
    out = np.empty((len(cfs), len(x)))
    for i, f in enumerate(cfs):
        p = np.exp(-2 * np.pi * GT_B * _erb(f) / sr)
        z = x * np.exp(-2j * np.pi * f * n / sr)
        for _ in range(GT_ORDER):
            z = lfilter([1 - p], [1, -p], z)
        out[i] = 2 * (z * np.exp(2j * np.pi * f * n / sr)).real
    return out


def hair_cells(bm, sr):
    """Half-wave rectify, compress (HC_EXP), low-pass at HC_LP_HZ (2nd-order Butterworth)."""
    b, a = butter(2, HC_LP_HZ / (sr / 2))
    return lfilter(b, a, np.maximum(bm, 0) ** HC_EXP, axis=-1)


def _frame_len(sr):
    return int(round(FRAME_S * sr))


def frames(hc, sr):
    """10 ms frame means along the last axis -> (channels, n_frames)."""
    k = _frame_len(sr)
    n = hc.shape[-1] // k
    return hc[..., :n * k].reshape(*hc.shape[:-1], n, k).mean(-1)


def loudness(bm, sr):
    """dict(sone (n_frames,), phon (n_frames,), specific (N_CH, n_frames)) from basilar-membrane output."""
    cfs = erb_space(bm.shape[0])
    w = 10 ** (-(iso226_contour(ISO226_PHON_REF, cfs) - ISO226_PHON_REF) / 10)  # power weighting
    power = frames(bm ** 2, sr)
    specific = LOUD_K * (w[:, None] * power) ** LOUD_EXP
    sone = specific.sum(0)
    phon = np.where(sone >= 1, PHON_REF + 10 * np.log2(np.maximum(sone, 1)),
                    PHON_LOW_SCALE * (sone + PHON_LOW_EPS) ** PHON_LOW_EXP)
    return dict(sone=sone, phon=np.maximum(phon, 0.0), specific=specific)


@dataclass
class CochleaOut:
    cfs: np.ndarray    # (N_CH,) Hz
    bm: np.ndarray     # (N_CH, n) basilar-membrane output
    hc: np.ndarray     # (N_CH, n) hair-cell output, 1 kHz low-passed, full rate
    env: np.ndarray    # (N_CH, n_frames) 10 ms frame means of hc
    loud: dict         # sone, phon, specific


def cochlea(x, sr=EAR_SR):
    cfs = erb_space()
    bm = gammatone(x, cfs, sr)
    hc = hair_cells(bm, sr)
    return CochleaOut(cfs, bm, hc, frames(hc, sr), loudness(bm, sr))
