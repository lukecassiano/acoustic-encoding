import numpy as np

from src.ear.constants import C_SOUND, EAR_SR, HEAD_RADIUS
from src.ear.room import Source, head_itd_s, render_binaural, speaker_filter
from tests.ear.conftest import rms, rng, x, y


def test_studio_front_is_identity_up_to_delay_and_head():
    sig = rng.standard_normal(EAR_SR)  # 1 s noise
    assert np.array_equal(speaker_filter(sig, "studio", EAR_SR), sig)


def test_front_source_is_symmetric():
    L, R = render_binaural([Source(x, 0.0, 1.0)])
    assert np.allclose(L, R, atol=1e-12)


def test_max_itd_matches_woodworth():
    assert abs(head_itd_s(90.0) - HEAD_RADIUS / C_SOUND * (np.pi / 2 + 1)) < 1e-9
    assert 0.00065 <= head_itd_s(90.0) <= 0.0007


def test_inverse_distance():
    L1, _ = render_binaural([Source(x, 0.0, 1.0)]); L2, _ = render_binaural([Source(x, 0.0, 2.0)])
    assert abs(20 * np.log10(rms(L2) / rms(L1)) + 6.02) < 0.1


def test_distance_clamped_finite():
    L, R = render_binaural([Source(x, 0.0, 0.0)])
    assert np.isfinite(L).all() and np.isfinite(R).all()


def test_two_sources_superpose():
    a, b = Source(x, 40.0, 2.0), Source(y, -55.0, 1.5, "horn")
    La, Ra = render_binaural([a]); Lb, Rb = render_binaural([b]); L, R = render_binaural([a, b])
    assert np.allclose(L, La + Lb) and np.allclose(R, Ra + Rb)
