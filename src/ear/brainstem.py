"""Brainstem binaural cues (ITD, ILD, correlation prominence) and the top-level hear()/listen()."""
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .cochlea import CochleaOut, cochlea
from .constants import (CAL_DB_AT_RMS_0_1, CUE_ENERGY_FLOOR, CUE_HOP_S, CUE_WIN_S, EAR_SR, ITD_MAX_LAG_S,
                        RMS_REF, SPL_MIN_DB, SPL_RMS_FLOOR)
from .room import render_binaural


def _windows(a, win, hop):
    """(N_CH, n) -> (N_CH, n_win, win)."""
    return sliding_window_view(a, win, axis=1)[:, ::hop]


def cues(left: CochleaOut, right: CochleaOut, sr):
    """Per-channel, per-window ITD (s, positive when the left ear leads), ILD (dB, positive when left is
    louder) and cross-correlation prominence (peak - min) / peak.

    ITD/prominence come from hair-cell output (phase-locking-limited), using the normalized
    cross-correlation without mean removal, r(tau) = sum L(t) R(t+tau) / sqrt(sum L^2 sum R^2),
    over |tau| <= ITD_MAX_LAG_S with parabolic peak interpolation. ILD comes from basilar-membrane
    window energy (hair-cell compression would shrink it in dB).
    """
    win, hop = int(round(CUE_WIN_S * sr)), int(round(CUE_HOP_S * sr))
    m = int(round(ITD_MAX_LAG_S * sr))
    n_ch, n = left.hc.shape
    n_win = 1 + (n - win) // hop

    # ILD
    el = (_windows(left.bm, win, hop) ** 2).sum(-1)[:, :n_win]
    er = (_windows(right.bm, win, hop) ** 2).sum(-1)[:, :n_win]
    ild = 10 * np.log10(np.maximum(el, CUE_ENERGY_FLOOR) / np.maximum(er, CUE_ENERGY_FLOOR))

    # ITD: R padded by m zeros each side so every lag has a full window
    lw = _windows(left.hc, win, hop)[:, :n_win]
    rp = np.pad(right.hc, ((0, 0), (m, m)))
    rw = _windows(rp, win + 2 * m, hop)[:, :n_win]
    norm = np.sqrt((lw ** 2).sum(-1) * (rw[..., m:m + win] ** 2).sum(-1))
    norm = np.maximum(norm, CUE_ENERGY_FLOOR)
    r = np.stack([(lw * rw[..., m + k:m + k + win]).sum(-1) for k in range(-m, m + 1)], axis=-1) / norm[..., None]

    k = r.argmax(-1)
    peak = np.take_along_axis(r, k[..., None], -1)[..., 0]
    rm1 = np.take_along_axis(r, np.clip(k - 1, 0, 2 * m)[..., None], -1)[..., 0]
    rp1 = np.take_along_axis(r, np.clip(k + 1, 0, 2 * m)[..., None], -1)[..., 0]
    den = rm1 - 2 * peak + rp1
    interior = (k > 0) & (k < 2 * m) & (den < 0)
    frac = np.where(interior, 0.5 * (rm1 - rp1) / np.where(interior, den, 1.0), 0.0)
    itd = (k - m + frac) / sr

    pmax = r.max(-1)
    prominence = np.where(pmax > 0, (pmax - r.min(-1)) / np.where(pmax > 0, pmax, 1.0), 0.0)
    return {"itd_s": itd, "ild_db": ild, "prominence": prominence}


@dataclass
class EarOut:
    left: CochleaOut
    right: CochleaOut
    cues: dict
    spl_db: tuple


def _spl(sig):
    rms = np.sqrt(np.mean(np.square(sig)))
    return float(max(SPL_MIN_DB, CAL_DB_AT_RMS_0_1 + 20 * np.log10(max(rms, SPL_RMS_FLOOR) / RMS_REF)))


def hear(left, right, sr=EAR_SR):
    """Ear-canal signals -> cochlear output per ear, binaural cues, and the level at each ear (dB SPL)."""
    cl, cr = cochlea(left, sr), cochlea(right, sr)
    return EarOut(cl, cr, cues(cl, cr, sr), (_spl(left), _spl(right)))


def listen(sources):
    return hear(*render_binaural(sources))
