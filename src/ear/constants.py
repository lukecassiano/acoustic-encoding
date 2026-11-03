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

# Cochlea (spec 6.7)
N_CH = 32                 # ERB-spaced gammatone channels
F_LO = 50.0               # Hz, lowest centre frequency
F_HI = 8000.0             # Hz, highest centre frequency
GT_ORDER = 4              # gammatone order (cascade of complex one-pole filters)
GT_B = 1.019              # bandwidth parameter in ERBs
HC_EXP = 0.3              # hair-cell compression exponent on the half-wave rectified output
HC_LP_HZ = 1000.0         # Hz, 2nd-order Butterworth low-pass modelling loss of phase locking
FRAME_S = 0.01            # s, loudness / envelope frame
# Deviation from spec 6.7 (which says ~0.23, Zwicker's exponent for excitation *with* spreading): without
# spreading, 0.3 on channel intensity reproduces the spec's own check of loudness doubling per 10 dB.
LOUD_EXP = 0.3
# Scales summed specific loudness so a 1 kHz tone at 40 dB SPL gives 1.0 sone. Derived once numerically:
# 1 / median over frames of sum_c (w_c P_c)^LOUD_EXP for tone(1000, 40) through the gammatone bank
# (test_one_sone_at_40db_1khz verifies it).
LOUD_K = 12.73523844895365
PHON_REF = 40.0           # phon of 1 sone
PHON_LOW_SCALE = 40.0     # ANSI S3.4-2007 approximation below 1 sone: 40 (sone + eps)^0.35
PHON_LOW_EPS = 0.0005
PHON_LOW_EXP = 0.35

# ISO 226:2003 Table 1 (f Hz, alpha_f, L_U dB, T_f dB) at the 29 standard frequencies
ISO226_F = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250,
            1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500]
ISO226_ALPHA = [0.532, 0.506, 0.480, 0.455, 0.432, 0.409, 0.387, 0.367, 0.349, 0.330, 0.315, 0.301, 0.288,
                0.276, 0.267, 0.259, 0.253, 0.250, 0.246, 0.244, 0.243, 0.243, 0.243, 0.242, 0.242, 0.245,
                0.254, 0.271, 0.301]
ISO226_LU = [-31.6, -27.2, -23.0, -19.1, -15.9, -13.0, -10.3, -8.1, -6.2, -4.5, -3.1, -2.0, -1.1, -0.4, 0.0,
             0.3, 0.5, 0.0, -2.7, -4.1, -1.0, 1.7, 2.5, 1.2, -2.1, -7.1, -11.2, -10.7, -3.1]
ISO226_TF = [78.5, 68.7, 59.5, 51.1, 44.0, 37.5, 31.5, 26.5, 22.1, 17.9, 14.4, 11.4, 8.6, 6.2, 4.4, 3.0, 2.2,
             2.4, 3.5, 1.7, -1.3, -4.2, -6.0, -5.4, -1.5, 6.0, 12.6, 13.9, 12.3]
ISO226_PHON_REF = 40.0    # contour used for the outer/middle-ear weighting
ISO226_AF_SCALE = 4.47e-3  # section 4.1: A_f = 4.47e-3 (10^(0.025 L_N) - 1.15) + [0.4 10^((T_f+L_U)/10 - 9)]^alpha
ISO226_AF_OFFSET = 1.15
ISO226_AF_FLOOR = 0.4
ISO226_L_REF = 94.0
