import numpy as np
import pytest
from scipy.signal import freqz

from src.ear.room import Source, _shadow_coeffs, head_itd_s
from src.ear.brainstem import listen
from tests.ear.conftest import EAR_SR, noise, tone


def _phase_delay(theta, f):
    b, a = _shadow_coeffs(theta, EAR_SR)
    _, h = freqz(b, a, worN=[2 * np.pi * f / EAR_SR])
    return float(-np.angle(h[0]) / (2 * np.pi * f))


def test_itd_low_frequency_matches_geometry():  # 500 Hz tone at 90 deg left
    e = listen([Source(tone(500, 70, 1.0), 90.0, 1.0)])
    ch = np.argmin(np.abs(e.left.cfs - 500))
    # the model's own expectation: Woodworth ITD plus the shadow filter's far-minus-near phase delay
    # (the known low-frequency ITD enlargement, Kuhn 1977)
    expected = head_itd_s(90.0) + _phase_delay(np.pi, e.left.cfs[ch]) - _phase_delay(0.0, e.left.cfs[ch])
    assert abs(np.median(e.cues["itd_s"][ch]) - expected) < 5e-5


def test_duplex_high_frequency_itd_ambiguous():  # 4 kHz pure tone: fine structure lost after 1 kHz low-pass
    e = listen([Source(tone(4000, 70, 1.0), 90.0, 1.0)]); ch = np.argmin(np.abs(e.left.cfs - 4000))
    assert np.median(e.cues["prominence"][ch]) < 0.1


def test_ild_grows_with_frequency():
    e = listen([Source(noise, 90.0, 1.0)]); ild = np.median(e.cues["ild_db"], axis=1)
    assert ild[e.left.cfs < 300].max() < 3 and ild[e.left.cfs > 6000].min() > 10


def test_cone_of_confusion():  # 30 deg and 150 deg are mirror images about the interaural axis
    a, b = listen([Source(noise, 30.0, 1.0)]), listen([Source(noise, 150.0, 1.0)])
    assert np.allclose(np.median(a.cues["itd_s"], 1), np.median(b.cues["itd_s"], 1), atol=2e-5)
    assert np.allclose(np.median(a.cues["ild_db"], 1), np.median(b.cues["ild_db"], 1), atol=0.2)


def test_spl_calibration():  # level at each ear; a tone, since the head shadows noise's highs even at 0 deg
    assert listen([Source(tone(250, 70), 0.0, 1.0)]).spl_db[0] == pytest.approx(70.0, abs=0.5)
