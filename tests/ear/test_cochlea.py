import numpy as np
import pytest

from src.ear.cochlea import cochlea, erb_space, gammatone, iso226_contour, loudness
from src.ear.constants import EAR_SR
from tests.ear.conftest import cf, freqs, impulse, noise, tone


def test_erb_spacing():
    cf = erb_space(); assert len(cf) == 32 and cf[0] == pytest.approx(50) and cf[-1] == pytest.approx(8000)


def test_iso226_shape():
    f = np.array([100., 1000., 3500.]); c = iso226_contour(40, f)
    assert c[1] == pytest.approx(40, abs=0.1) and c[0] > 55 and c[2] < 40


def test_gammatone_peak_and_bandwidth():  # impulse response per channel
    H = np.abs(np.fft.rfft(gammatone(impulse, cf, EAR_SR), axis=1))
    peak = freqs[H.argmax(1)]; assert np.all(np.abs(peak / cf - 1) < 0.05)
    # 3 dB bandwidth within 15% of 0.887 * ERB(cf) (4th-order gammatone)
    bw = (H >= H.max(1, keepdims=True) / np.sqrt(2)).sum(1) * (freqs[1] - freqs[0])
    erb = 24.7 * (4.37 * cf / 1000 + 1)
    assert np.all(np.abs(bw / (0.887 * erb) - 1) < 0.15)


def test_one_sone_at_40db_1khz():
    assert np.median(loudness(gammatone(tone(1000, 40), cf, EAR_SR), EAR_SR)["sone"]) == pytest.approx(1.0, rel=0.05)


def test_loudness_doubles_per_10db():
    s50, s60 = (np.median(loudness(gammatone(tone(1000, d), cf, EAR_SR), EAR_SR)["sone"]) for d in (50, 60))
    assert 1.85 < s60 / s50 < 2.15


def test_extreme_gain_finite():
    out = cochlea(np.clip(40 * noise, -1e3, 1e3)); assert np.isfinite(out.loud["phon"]).all()


def test_silence_finite():
    out = cochlea(np.zeros(EAR_SR)); assert np.isfinite(out.loud["phon"]).all() and (out.loud["sone"] == 0).all()
