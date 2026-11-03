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


def test_shadow_filter_dc_and_nyquist_gain():
    from src.ear.constants import ALPHA_MIN, THETA_MIN_DEG
    from src.ear.room import _shadow_coeffs
    for theta in (0.0, np.pi):
        b, a = _shadow_coeffs(theta, EAR_SR)
        alpha = (1 + ALPHA_MIN / 2) + (1 - ALPHA_MIN / 2) * np.cos(theta / np.radians(THETA_MIN_DEG) * np.pi)
        assert abs(b.sum() / a.sum() - 1.0) < 1e-12                      # DC
        assert abs((b[0] - b[1]) / (a[0] - a[1]) - alpha) < 1e-9         # Nyquist -> alpha(theta)
    # alpha(0) = 2 boosts highs on the near side; alpha(pi) < 1 shadows the far side
    nyq = lambda th: abs(np.subtract(*_shadow_coeffs(th, EAR_SR)[0]) / np.subtract(*_shadow_coeffs(th, EAR_SR)[1]))
    assert nyq(0.0) > 1.9 and nyq(np.pi) < 0.3


def test_lateral_source_near_ear_leads_and_is_louder():
    from src.ear.constants import EAR_SR as sr
    def onset(a):
        return int(np.argmax(np.abs(a) > 0.5 * np.abs(a).max()))
    imp = np.zeros(256); imp[0] = 1.0
    L, R = render_binaural([Source(imp, 90.0, 1.0)])
    assert onset(L) < onset(R) and rms(L) > rms(R)
    assert abs((onset(R) - onset(L)) / sr - head_itd_s(90.0)) < 2 / sr
    L2, R2 = render_binaural([Source(imp, -90.0, 1.0)])
    assert onset(R2) < onset(L2) and rms(R2) > rms(L2)
    assert np.allclose(L2, R) and np.allclose(R2, L)


def test_itd_sign_is_antisymmetric():
    assert head_itd_s(90.0) > 0 > head_itd_s(-90.0)
    assert abs(head_itd_s(-90.0) + head_itd_s(90.0)) < 1e-15
    assert head_itd_s(0.0) == 0.0


def test_fractional_delay_linear_interpolation():
    from src.ear.room import _delay
    imp = np.zeros(8); imp[0] = 1.0
    out = _delay(imp, 2.25 / EAR_SR, 12, EAR_SR)
    assert np.allclose(out[2:4], [0.75, 0.25]) and np.count_nonzero(out) == 2
    assert abs((out * np.arange(12)).sum() - 2.25) < 1e-12              # centroid = requested shift
