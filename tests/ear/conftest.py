"""Shared signal helpers for the ear tests, imported as `from tests.ear.conftest import x, rms`."""
import numpy as np

from src.ear.constants import CAL_DB_AT_RMS_0_1, EAR_SR

rng = np.random.default_rng(0)


def rms(a):
    return float(np.sqrt(np.mean(np.square(a))))


def noise_rms(r):
    """1 s of white noise scaled to RMS r."""
    n = rng.standard_normal(EAR_SR)
    return n * r / rms(n)


def tone(f, db, dur=1.0):
    """Sine at level db (70 dB <-> RMS 0.1)."""
    t = np.arange(int(dur * EAR_SR)) / EAR_SR
    return np.sqrt(2) * 0.1 * 10 ** ((db - CAL_DB_AT_RMS_0_1) / 20) * np.sin(2 * np.pi * f * t)


x, y, noise = noise_rms(0.1), noise_rms(0.1), noise_rms(0.1)

# Cochlea fixtures: ERB-spaced centre frequencies and a unit impulse (long enough for sub-Hz bins at 50 Hz).
from src.ear.cochlea import erb_space  # noqa: E402

cf = erb_space()
impulse = np.zeros(2 ** 16)
impulse[0] = 1.0
freqs = np.fft.rfftfreq(len(impulse), 1 / EAR_SR)
