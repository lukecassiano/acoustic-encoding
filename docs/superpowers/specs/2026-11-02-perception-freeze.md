# CAMP perception layer: pre-registration freeze (DRAFT, counts pending)

Date: 2026-11-02 (frozen on go-ahead)
Spec: `docs/superpowers/specs/2026-11-02-perception-layer-design.md`, commit `70a13b5a411a4ba2c0e4deec38e3d23f10e19df1`
Branch: `perception-1`. No ratings have been read. `src.deam.load_ratings` refuses until the final line below carries a go-ahead.

## Inclusion and grouping (`src/deam.py`)
- Included: DEAM 2013-2014 45 s excerpts only, `song_id < MAX_EXCERPT_ID = 2000`, with an existing `data/deam/MEMD_audio/<id>.mp3`. The 2015 full songs are excluded.
- Skip rules (`levels.process_one`): unreadable; shorter than `MIN_DURATION_S = 9.0` s; silent (`SILENT_RMS = 1e-6`, RMS of the 22,050 Hz signal).
- Group = artist, lower-cased and stripped, from `metadata_2013.csv` / `metadata_2014.csv`; a song without an artist is its own group `song-<id>`.
- Ratings: static song-level `valence_mean`, `arousal_mean` (1-9). Continuous 2 Hz ratings are out of scope.

## Counts (filled in after extraction, before go-ahead; audio-only, no ratings involved)
| Item | Value |
|---|---|
| Excerpts found | TBD |
| Excerpts included | TBD |
| Skipped: unreadable | TBD |
| Skipped: shorter than 9 s | TBD |
| Skipped: silent | TBD |
| Skipped: no rating (dropped at merge, counted) | TBD |
| Groups (artists) | TBD |

## Decisions
1. L0 and L2 from the raw excerpt (`F.extract(F.prepare(y22))`); L1 from the ear-rendered signal: `Source(y44, 0.0, 1.0, "studio", 0.0)` = 0 deg, 1 m, studio preset, 0 dB.
2. TR rows >= `TR_START = 5` (>= 7.5 s) for every level; L1 uses `L1_START_S = 7.5` s of 10 ms frames.
3. L0 includes `raw_log_rms` (log RMS of the un-normalized signal); it counts toward the acoustic pathway for H2.
4. L2 from the published web bundle `data/web/sub-001..005`; acoustic cols `0:300`, structural `300:340`; stats and nuisance betas must be identical across listeners.
5. Random projections block-wise (acoustic block, structural block), column norms matched to the matching `W_region` block.
6. Loudness exponent `LOUD_EXP = 0.3` instead of the spec's ~0.23: without spreading, 0.3 on channel intensity reproduces the spec's own check of loudness doubling per 10 dB.
7. The low-frequency ITD test expectation includes the head-shadow phase (not used in the DEAM analysis).
8. The page readout L1+L2 is reported descriptively, not as a hypothesis.
9. Continuous (2 Hz) ratings are out of scope.

## Representations
| Level | Dim | Definition |
|---|---|---|
| L0 | 69 | `L0_NAMES` = `F.FEATURE_NAMES["acoustic"]` (60) + `["structural"]` (8: tempo, pulse_clarity, onset_rate, beat_cv, key_clarity, mode, roughness, harmonic_change) + `raw_log_rms`. Clip mean over TR rows >= 5; `raw_log_rms = log(sqrt(mean(y22^2)) + 1e-8)`. |
| L1 | 66 | `L1_NAMES` = `env_00..31`, `mod_00..31`, `phon_mean`, `phon_p95`. Per ear, on frames from `start = round(7.5 / FRAME_S)`: `env_c` = mean over frames of hair-cell envelope; `mod_c` = `mean(band^2) / max(mean(env)^2, MOD_FLOOR)` with `band = sosfiltfilt(butter(2, [0.5, 20.0] Hz, bandpass, fs = 1/FRAME_S), env)`; `phon_mean`, `phon_p95` = mean and 95th percentile of the loudness `phon` track. The two ears' 66-vectors are averaged. |
| L2 | 2R per listener | `acoustic = xbar[:, 0:300] @ W_region[0:300]`, `structural = xbar[:, 300:340] @ W_region[300:340]`; `xbar` = mean over TR rows >= 5 of `probe_design(F.extract(F.prepare(y22)), stats_sub-001)`; R = number of regions with a column in `meta.json` (sub-001..005 each). |

L1 summary constants: `MOD_LO_HZ = 0.5`, `MOD_HI_HZ = 20.0`, `MOD_ORDER = 2`, `L1_START_S = 7.5`, `MOD_FLOOR = 1e-12`.

## Ear constants (`src/ear/constants.py`, name = value)
- Global: `EAR_SR = 44100`, `C_SOUND = 343.0`, `HEAD_RADIUS = 0.0875`, `MIN_DIST = 0.1`, `REF_DIST = 1.0`, `MAX_DIST = 20.0`, `ALPHA_MIN = 0.1`, `THETA_MIN_DEG = 150.0`, `CAL_DB_AT_RMS_0_1 = 70.0`, `RMS_REF = 0.1`, `DIRECTIVITY_FLOOR = 0.05`.
- Preset used: `PRESETS["studio"] = {"biquads": [], "beta": 0.0}`. Others (not used): laptop `[("highpass", 300.0, 0.707, 0.0), ("peaking", 3000.0, 1.0, 4.0)]`, beta 0.0; subwoofer `[("lowpass", 120.0, 0.707, 0.0)]`, beta 0.0; horn `[("highpass", 500.0, 0.707, 0.0), ("lowpass", 4000.0, 0.707, 0.0)]`, beta 0.63.
- Cochlea: `N_CH = 32`, `F_LO = 50.0`, `F_HI = 8000.0`, `GT_ORDER = 4`, `GT_B = 1.019`, `HC_EXP = 0.3`, `HC_LP_HZ = 1000.0`, `FRAME_S = 0.01`.
- Loudness: `LOUD_EXP = 0.3`, `LOUD_K = 12.73523844895365`, `PHON_REF = 40.0`, `PHON_LOW_SCALE = 40.0`, `PHON_LOW_EPS = 0.0005`, `PHON_LOW_EXP = 0.35`.
- Outer/middle ear (ISO 226:2003): `ISO226_PHON_REF = 40.0`, `ISO226_AF_SCALE = 4.47e-3`, `ISO226_AF_OFFSET = 1.15`, `ISO226_AF_FLOOR = 0.4`, `ISO226_L_REF = 94.0`; tables `ISO226_F` (29 freqs, 20 to 12500 Hz), `ISO226_ALPHA`, `ISO226_LU`, `ISO226_TF` exactly as in `constants.py`.
- Cues: `CUE_WIN_S = 0.05`, `CUE_HOP_S = 0.025`, `ITD_MAX_LAG_S = 0.001`, `CUE_ENERGY_FLOOR = 1e-12`, `SPL_RMS_FLOOR = 1e-12`, `SPL_MIN_DB = 0.0`.

## Readout and statistics (`src/perception/readout.py`)
| Item | Frozen value |
|---|---|
| Seed | `SEED = 20261102` (all `numpy.random.default_rng` and split) |
| Split | `GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE=0.3, random_state=SEED)`, groups = artist; test set scored once |
| Model | `VarianceThreshold(0.0)` -> `StandardScaler` -> `Ridge`; scaler refit inside each fold |
| Alpha grid | `ALPHAS = np.logspace(-2, 5, 30)` |
| CV | `N_FOLDS = 5`, `GroupKFold`, scoring R2, `GridSearchCV(refit=True)` |
| Bootstrap | `N_BOOT = 2000` paired resamples over test songs of R2_a - R2_b; CI = 2.5 / 97.5 percentiles |
| Permutation | `N_PERM = 5000`, one-sided on r, permute y against fixed predictions, `p = (1 + #{r_perm >= r_obs}) / (1 + n)` |
| Random projections | `N_PROJ = 1000`; per draw per block `G` standard normal, columns rescaled to the norm of the matching `W_region` column; fit on train, R2 on test |
| Verdict rule (`verdict(lo, hi)`) | `lo > 0` supported; `hi < 0` not supported; otherwise inconclusive |

Theory baseline (`theory_baseline`, no fitted weights beyond a 1-D calibration; each feature z-scored with TRAINING mean/sd only):
- arousal score = z(`phon_mean`) + z(`onset_rate`) + z(`tempo`)
- valence score = z(`mode`) + z(`key_clarity`)
- each score mapped to the rating scale by a least-squares line fit on train, applied to test. Exploratory comparison only (spec 5.5).

## Hypotheses and verdict rules (spec 5.3 and 5.4, quoted)
> **H1, the ear front end earns its place.** For arousal, held-out R²(L0 + L1) > R²(L0). Supported if the bootstrap interval of the difference excludes 0. Expected mechanism: perceptual loudness and cochlear modulation energy track arousal better than RMS and the mel spectrum.
>
> **H2, a pathway double dissociation.** Within L0 (the feature bands themselves) *and* within L2 (through the cortical projection), the structural pathway predicts valence better than the acoustic pathway does, and the acoustic pathway predicts arousal better than the structural pathway does. Supported per level if both differences hold with intervals excluding 0. Evaluate L2 per listener; the verdict needs ≥ 4 of 5 listeners. This is the link to Analysis A: the acoustic/structural division in auditory cortex would have measurable perceptual consequences.
>
> **H3, the cortical projection preserves affect information beyond chance.** See §5.4 for why this is framed against random projections, not against L0.

> - CAMP's prediction is a **linear function of L0 features** (delayed copies, then weights). Averaged over a clip, L2 lies essentially inside the span of L0. A linear readout from L2 therefore cannot beat L0 except through its bottleneck acting as a prior. "Cortex beats raw features" is not a meaningful test, and is not run as a hypothesis.
>
> - H3 instead asks whether the **brain's** projection keeps more affect information than an **arbitrary** projection of the same dimensionality. Compute held-out R² for L2 (2R dimensions per listener), then repeat with 1,000 random linear projections of the delayed L0 features to 2R dimensions. Projections are built **block-wise**: acoustic columns → R dimensions and structural columns → R dimensions, each matched in column norms to that pathway's block of `w_region`, so the null preserves the two-pathway structure and tests only the brain's weighting within it **(Claude's call)**. Supported if L2's R² exceeds the 95th percentile of the random-projection distribution in ≥ 4 of 5 listeners.
>
> - H2 at L0 is partly built in by feature choice (the structural band contains `mode`). That is why H2 is tested at both L0 and L2, and why the paper must say that only the L2 version says anything about cortex.
>
> - Every L2 result is reported per listener, because the structural pathway ranges from absent (sub-003) to dominant (sub-005). Whether sub-005's cortex carries more valence information than sub-003's is **exploratory**, not a hypothesis.

Spec 5.6: each hypothesis gets a supported / not-supported / inconclusive verdict; a clean null is a result.

## Aggregation rules (rulings R16 to R20)
- **R2** = 1 - SS_res/SS_tot, SS_tot about the test-set mean of y. Songs without a rating are dropped and counted (table above).
- **H1 (arousal)**: `boot_diff(y, pred_L0+L1, pred_L0)` -> `verdict(lo, hi)`.
- **H2**, per level (L0; L2 per listener): each pathway is its own `fit_readout` with its own CV-chosen alpha (L0: acoustic 61 columns incl. `raw_log_rms` vs structural 8; L2: R vs R columns). Two paired `boot_diff`s, each passed to `verdict()`: valence R2(structural) - R2(acoustic); arousal R2(acoustic) - R2(structural). A level/listener supports H2 iff both are "supported"; fails iff either is "not supported"; otherwise inconclusive. H2-L0 verdict = that level's verdict. H2-L2 overall: "supported" if >= 4 of 5 listeners support, "not supported" if >= 4 of 5 fail, else "inconclusive".
- **H3**, separately for each rating and each listener: null uses that listener's `W_region` and R; input is the clip-mean design row `xbar` (340 columns), i.e. the spec's "delayed L0 features" after training z-scoring and clip-onset nuisance removal, averaged over TR rows >= 5. `random_projection_r2` is called with `numpy.random.default_rng(SEED)` for each (listener, rating) (`N_PROJ = 1000`). A listener passes if its L2 held-out R2 > `np.percentile(null, 95)`. Per rating: "supported" if >= 4 of 5 listeners pass, "not supported" if >= 4 of 5 fail, else "inconclusive".
- **Permutation p** (`N_PERM = 5000`, against fixed held-out predictions): descriptive, reported for every model; does not gate H1-H3.

## Environment (.venv)
python 3.13.14, numpy 2.3.5, scipy 1.15.3, scikit-learn 1.9.1, librosa 1.0.0.

Go-ahead: (Luke writes "Go-ahead: Luke, YYYY-MM-DD" here)
