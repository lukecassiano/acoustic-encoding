"""Every numeric constant of the reference ear model lives here (nothing numeric elsewhere in src/ear)."""

EAR_SR = 44100            # Hz, sample rate of the whole ear model
C_SOUND = 343.0           # m/s
HEAD_RADIUS = 0.0875      # m, Brown & Duda spherical head
MIN_DIST = 0.1            # m, source distance clamp
REF_DIST = 1.0            # m, distance at which propagation gain is 1
MAX_DIST = 20.0           # m, sizes the zero-padded tail so output length is independent of distance
ALPHA_MIN = 0.1           # head-shadow alpha_min
THETA_MIN_DEG = 150.0     # head-shadow theta_min
CAL_DB_AT_RMS_0_1 = 70.0  # dB SPL assigned to a signal of RMS 0.1
DIRECTIVITY_FLOOR = 0.05  # lower bound of speaker directivity gain

# Speaker presets: RBJ biquad chain (kind, f0 Hz, Q, gain dB) + directivity beta,
# g(phi) = max(floor, (1 - beta) + beta cos phi)
PRESETS = {
    "studio": {"biquads": [], "beta": 0.0},
    "laptop": {"biquads": [("highpass", 300.0, 0.707, 0.0), ("peaking", 3000.0, 1.0, 4.0)], "beta": 0.0},
    "subwoofer": {"biquads": [("lowpass", 120.0, 0.707, 0.0)], "beta": 0.0},
    "horn": {"biquads": [("highpass", 500.0, 0.707, 0.0), ("lowpass", 4000.0, 0.707, 0.0)], "beta": 0.63},
}
