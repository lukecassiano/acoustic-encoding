# CAMP Perception Layer — Plan 1: Python ear model and the affect-rating test

**Status (2026-11-03):** Tasks 1–11 complete on `perception-1`; real run 758b24e; final review done. Exception: Task 8 Step 6 committed code only; `results.md`/`results.json` are deliberately uncommitted (only if Luke asks).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python ear model (speaker → spherical head → cochlea → loudness → brainstem cues), the DEAM level representations L0/L1/L2, the pre-registered tests H1–H3 with figures for the Music 108 paper, and the page readout (L1 + L2) exported for the website.

**Architecture:** `src/ear/` is the reference ear model in plain numpy/scipy DSP, so Plan 2 can port it to JS with a parity test. `src/deam.py` loads DEAM audio and metadata; its ratings loader refuses to run until Luke's go-ahead line exists in the freeze file. `src/perception/` computes the representation levels per excerpt (cached and resumable), fits ridge readouts, and runs the hypotheses, figures and web export. L2 is computed from the **published web bundle** (`data/web/sub-XXX/meta.json` + `w_region.f32`), so the paper and the page use identical weights.

**Tech Stack:** Python 3.13 venv; numpy, scipy, librosa, pandas, matplotlib (existing); scikit-learn (installed, 1.9.1) and pytest (installed, 9.1.1), both added to `requirements.txt`.

**Spec:** `docs/superpowers/specs/2026-11-02-perception-layer-design.md` (approved by Luke 2026-11-02). This is plan 1 of 2. Plan 2 (JS ear port, room, Listen and Analysis tabs) follows. It needs Task 3's interfaces and Task 10's export, nothing else.

## Global Constraints

- Work on a new branch `perception-1` from `main`. `memory-model-1` is unrelated and unmerged; do not touch it.
- **Ratings are not opened before the freeze go-ahead.** Only `src.deam.load_ratings` reads rating files. It raises unless the freeze file contains a line matching `^Go-ahead: Luke, \d{4}-\d{2}-\d{2}`, which **Luke writes himself**. Tests use synthetic rating fixtures only.
- Spec §5 hypotheses, splits, metrics and verdict rules are frozen. Do not add, drop or reword hypotheses. Anything extra is labelled exploratory.
- L0 and L2 come from the **raw excerpt**: `F.extract(F.prepare(y22))`. Only L1 sees the ear-rendered signal (spec §5.1).
- Clip-level means use TR rows with index ≥ 5 (≥ 7.5 s) for every level. Ear frames map onto the same 1.5 s grid.
- DEAM rendering: one source at azimuth 0°, distance 1 m, preset `"studio"` (exact identity), gain 0 dB.
- Level calibration: digital RMS 0.1 at the ear ≙ 70 dB SPL (`CAL_DB_AT_RMS_0_1 = 70.0`).
- Ear sample rate 44,100 Hz. CAMP features stay at 22,050 Hz (`F.SR`); do not modify `src/features.py`, `src/encoding.py` or any fitted output.
- All randomness comes from `numpy.random.default_rng(seed)` with `SEED = 20261102` unless a test sets its own.
- Standardization uses training-split statistics only.
- Every ear constant lives in `src/ear/constants.py`. Plan 2 exports that module to JSON, so no ear numbers are written anywhere else.
- DEAM audio and ratings stay on this machine (`data/deam/`, gitignored with the rest of `data/`).
- Run tests with `.venv/Scripts/python -m pytest tests/ear tests/perception -q` from `acoustic-encoding/`.
- Claude does not write paper prose. `results.md` contains numbers, verdicts and tables only.

## Review Focus

1. **Speaker placed on top of the listener (distance → 0).** Distance clamps at 0.1 m and the output stays finite. Test in Task 1.
2. **Extreme gain (+40 dB) or a clipped source.** Loudness and ILD stay finite (no `log(0)`, no overflow). Test in Task 2.
3. **A DEAM file that is silent, shorter than 9 s, or unreadable.** The levels run skips it with a recorded reason in `skipped.csv` and never writes NaN rows. Test in Task 5.
4. **Stereo, 48 kHz or 8-bit input files.** The loader mixes to mono and resamples to both rates. Test in Task 4.
5. **A listener whose structural pathway is near-constant (sub-003).** The ridge readout must not return NaN, and H2 records `inconclusive` for that listener instead of crashing. Test in Task 7.

---

### Task 1: Ear model I — speaker, propagation, spherical head

**Files:**
- Create: `src/ear/__init__.py`, `src/ear/constants.py`, `src/ear/room.py`
- Create: `tests/ear/__init__.py`, `tests/ear/conftest.py`, `tests/ear/test_room.py`
- Modify: `requirements.txt` (add `scikit-learn`, `pytest`)

`tests/ear/conftest.py` holds the shared signal helpers used by Tasks 1–3, exposed as module-level functions and imported by the test files:
- `rng = default_rng(0)`
- `x`, `y`, `noise`: 1 s white noise at RMS 0.1
- `rms(a)`
- `noise_rms(r)`
- `tone(f, db, dur=1.0)`: a sine with RMS `0.1 * 10**((db - 70) / 20)`
- `impulse`
- `cf = erb_space()`
- `freqs` (rfft bin frequencies)

**Interfaces:**
- Produces:
  - `Source(signal: np.ndarray, azimuth_deg: float, distance_m: float, preset: str = "studio", gain_db: float = 0.0)` (a dataclass). Azimuth is measured from straight ahead, positive to the listener's left.
  - `speaker_filter(x, preset: str, sr: int) -> np.ndarray`
  - `directivity(preset: str, off_axis_deg: float) -> float`
  - `render_binaural(sources: list[Source], sr: int = EAR_SR) -> tuple[np.ndarray, np.ndarray]` returns (left, right) at the ear canal entrance.
  - `head_itd_s(azimuth_deg) -> float` (left lead positive).

Constants (`constants.py`, exact values):
- `EAR_SR = 44100`, `C_SOUND = 343.0`, `HEAD_RADIUS = 0.0875`, `MIN_DIST = 0.1`, `REF_DIST = 1.0`
- `ALPHA_MIN = 0.1`, `THETA_MIN_DEG = 150.0`, `CAL_DB_AT_RMS_0_1 = 70.0`
- `PRESETS` dict:
  - `"studio"`: biquads `[]`, β 0.0
  - `"laptop"`: highpass 300 Hz Q 0.707, peaking 3 kHz +4 dB Q 1.0, β 0.0
  - `"subwoofer"`: lowpass 120 Hz Q 0.707, β 0.0
  - `"horn"`: highpass 500 Hz Q 0.707, lowpass 4 kHz Q 0.707, β 0.63

  Directivity is g(φ) = max(0.05, (1 − β) + β cos φ); the speaker faces the listener, so φ = 0 for the listener. Biquads follow the RBJ Audio EQ Cookbook formulas so JS can reproduce them exactly.

- [x] **Step 1: Write the failing tests** in `tests/ear/test_room.py`

```python
def test_studio_front_is_identity_up_to_delay_and_head():
    x = rng.standard_normal(EAR_SR)  # 1 s noise
    assert np.array_equal(speaker_filter(x, "studio", EAR_SR), x)

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
```

- [x] **Step 2: Run** `.venv/Scripts/python -m pytest tests/ear/test_room.py -q`. Expected: FAIL (`ModuleNotFoundError: src.ear`).
- [x] **Step 3: Implement `src/ear/constants.py` and `src/ear/room.py`.**
  - Head model (Brown & Duda 1998): for each ear, θ is the angle between the source direction and that ear's axis (ears at ±90°).
  - Shadow filter H(ω) = (1 + jα(θ)ω/2ω₀)/(1 + jω/2ω₀), with ω₀ = C_SOUND/HEAD_RADIUS and α(θ) = (1 + ALPHA_MIN/2) + (1 − ALPHA_MIN/2)·cos(θ/THETA_MIN_DEG·180°). It is discretized by the bilinear transform with prewarping at ω₀.
  - Per-ear delay T(θ) = −(a/c)cos θ for θ < 90°, else (a/c)(θ − π/2). Add a constant a/c so delays are non-negative.
  - Propagation: gain REF_DIST/max(d, MIN_DIST) and delay max(d, MIN_DIST)/C_SOUND, where d is the source's distance. Fractional delays use linear interpolation.
  - Output length = input length + max delay in samples, zero-padded.
- [x] **Step 4: Run tests.** Expected: 6 passed.
- [x] **Step 5: Commit** `feat(ear): speaker presets, propagation, spherical head`.

### Task 2: Ear model II — outer/middle ear, cochlea, hair cells, loudness

**Files:**
- Create: `src/ear/cochlea.py`, `tests/ear/test_cochlea.py`
- Modify: `src/ear/constants.py`

**Interfaces:**
- Consumes: `EAR_SR`, `CAL_DB_AT_RMS_0_1`.
- Produces:
  - `erb_space(n=N_CH, lo=F_LO, hi=F_HI) -> np.ndarray` (centre frequencies, Hz)
  - `iso226_contour(phon: float, freqs: np.ndarray) -> np.ndarray` (dB SPL)
  - `gammatone(x, cfs, sr) -> np.ndarray` of shape (N_CH, n_samples), real
  - `hair_cells(bm: np.ndarray, sr) -> np.ndarray` (N_CH, n_samples), 1 kHz low-passed
  - `frames(hc, sr) -> np.ndarray` (N_CH, n_frames), 10 ms frame means (`FRAME_S = 0.01`)
  - `loudness(bm, sr) -> dict(sone=(n_frames,), phon=(n_frames,), specific=(N_CH, n_frames))`
  - `CochleaOut` (dataclass) with fields `cfs, bm, hc, env, loud`, and `cochlea(x, sr=EAR_SR) -> CochleaOut`

Constants:
- `N_CH = 32`, `F_LO = 50.0`, `F_HI = 8000.0`, `GT_ORDER = 4`, `GT_B = 1.019`
- `HC_EXP = 0.3`, `HC_LP_HZ = 1000.0` (2nd-order Butterworth)
- `LOUD_EXP = 0.3`
- **Deviation from spec §6.7, recorded:** the spec says about 0.23 (Zwicker's exponent applies to excitation *with* spread across bands). Without spreading, 0.3 on channel intensity is what reproduces the spec's own check of loudness doubling per 10 dB.

Outer and middle ear: each channel's power is weighted by `−(iso226_contour(40, cf) − 40)` dB. That is the inverted 40-phon equal-loudness contour, applied per channel. Implement ISO 226:2003 §4.1 with its Table 1 parameters (copy them from the standard; cite it in the docstring). Sone calibration: a 1 kHz tone at 40 dB SPL gives 1.0 sone. Phon = 40 + 10·log2(sone) for sone ≥ 1, and 40·(sone + 0.0005)^0.35 below 1 sone (ANSI S3.4-2007 approximation).

- [x] **Step 1: Write the failing tests** in `tests/ear/test_cochlea.py`

```python
def test_erb_spacing():
    cf = erb_space(); assert len(cf) == 32 and cf[0] == pytest.approx(50) and cf[-1] == pytest.approx(8000)

def test_iso226_shape():
    f = np.array([100., 1000., 3500.]); c = iso226_contour(40, f)
    assert c[1] == pytest.approx(40, abs=0.1) and c[0] > 55 and c[2] < 40

def test_gammatone_peak_and_bandwidth():  # impulse response per channel
    H = np.abs(np.fft.rfft(gammatone(impulse, cf, EAR_SR), axis=1))
    peak = freqs[H.argmax(1)]; assert np.all(np.abs(peak / cf - 1) < 0.05)
    # 3 dB bandwidth within 15% of 0.887 * 1.019 * ERB(cf) (4th-order gammatone)

def test_one_sone_at_40db_1khz():
    assert np.median(loudness(gammatone(tone(1000, 40), cf, EAR_SR), EAR_SR)["sone"]) == pytest.approx(1.0, rel=0.05)

def test_loudness_doubles_per_10db():
    s50, s60 = (np.median(loudness(gammatone(tone(1000, d), cf, EAR_SR), EAR_SR)["sone"]) for d in (50, 60))
    assert 1.85 < s60 / s50 < 2.15

def test_extreme_gain_finite():
    out = cochlea(np.clip(40 * noise, -1e3, 1e3)); assert np.isfinite(out.loud["phon"]).all()

def test_silence_finite():
    out = cochlea(np.zeros(EAR_SR)); assert np.isfinite(out.loud["phon"]).all() and (out.loud["sone"] == 0).all()
```

`tone` comes from `tests/ear/conftest.py`. Its level convention matches `CAL_DB_AT_RMS_0_1`.

- [x] **Step 2: Run.** Expected: FAIL.
- [x] **Step 3: Implement.**
  - Gammatone: a cascade of GT_ORDER complex one-pole filters at baseband (shift by e^{−j2πcf·t}, filter, shift back, take the real part), with bandwidth `GT_B·ERB(cf)`, where ERB(f) = 24.7(4.37f/1000 + 1). Normalize each channel to 0 dB gain at cf.
  - Silence: sone 0, phon floor 0.
- [x] **Step 4: Run.** Expected: 7 passed. If the 3-dB bandwidth test fails by a constant factor, fix the gain normalization, not the test.
- [x] **Step 5: Commit** `feat(ear): ISO 226 weighting, gammatone cochlea, hair cells, loudness`.

### Task 3: Ear model III — brainstem cues, `hear()`, psychophysics

**Files:**
- Create: `src/ear/brainstem.py`, `tests/ear/test_psychophysics.py`
- Modify: `src/ear/__init__.py` (export `Source`, `render_binaural`, `hear`, `EarOut`)

**Interfaces:**
- Consumes: Task 1 `render_binaural`, `Source`; Task 2 `cochlea`, `CochleaOut`.
- Produces:
  - `cues(hc_left, hc_right, sr) -> dict(itd_s=(N_CH, n_win), ild_db=(N_CH, n_win), prominence=(N_CH, n_win))`
    - windows: `CUE_WIN_S = 0.05`, hop `CUE_HOP_S = 0.025`
    - ITD from normalized cross-correlation over lags ±`ITD_MAX_LAG_S = 0.001` with parabolic peak interpolation
    - prominence = (peak − min)/peak of the normalized cross-correlation
    - ILD = 10·log10(E_L/E_R), each energy floored at 1e-12
  - `EarOut` (dataclass) with fields `left: CochleaOut, right: CochleaOut, cues: dict, spl_db: tuple[float, float]`
  - `hear(left, right, sr=EAR_SR) -> EarOut`
  - `listen(sources: list[Source]) -> EarOut` (= `hear(*render_binaural(sources))`)

- [x] **Step 1: Write the failing tests** (these are spec §6's psychophysical checks)

```python
def test_itd_low_frequency_matches_geometry():  # 500 Hz tone at 90 deg left
    e = listen([Source(tone(500, 70, 1.0), 90.0, 1.0)])
    ch = np.argmin(np.abs(e.left.cfs - 500))
    assert abs(np.median(e.cues["itd_s"][ch]) - head_itd_s(90.0)) < 1e-4

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

def test_spl_calibration():
    assert listen([Source(noise_rms(0.1), 0.0, 1.0)]).spl_db[0] == pytest.approx(70.0, abs=0.5)
```

- [x] **Step 2: Run.** Expected: FAIL.
- [x] **Step 3: Implement** `cues`, `hear`, `listen`. `spl_db` = CAL_DB_AT_RMS_0_1 + 20·log10(rms/0.1) per ear, floored at 0.
- [x] **Step 4: Run the whole ear suite.** Expected: 18 passed.
- [x] **Step 5: Commit** `feat(ear): brainstem ITD/ILD and psychophysical checks`.

### Task 4: DEAM access and loader (no ratings)

**Files:**
- Create: `src/deam.py`, `tests/perception/__init__.py`, `tests/perception/test_deam.py`
- Create: `data/deam/` (downloaded; gitignored)

**Interfaces:**
- Produces:
  - `DEAM = DATA / "deam"`
  - `excerpts() -> pd.DataFrame` with columns `song_id:int, path:Path, artist:str, group:str`
  - `load_excerpt(path) -> tuple[np.ndarray, np.ndarray]` returns (y22 at 22,050 Hz, y44 at 44,100 Hz), both mono float64, neither faded nor normalized
  - `load_ratings(freeze_path: Path) -> pd.DataFrame` with columns `song_id, valence, arousal`
  - `FreezeError(Exception)`

Inclusion (frozen here, restated in the freeze file): the 2013–2014 45 s excerpts only (song IDs below 2000). The 2015 full-length songs are excluded because their rating protocol and length differ. `group` = the artist name, lower-cased and stripped; if the artist is missing, `"song-<id>"`.

- [x] **Step 1: Get the data.** Download the DEAM audio and metadata archives from the official DEAM page (cvml.unige.ch/databases/DEAM, mirrored on Kaggle). Unzip to `data/deam/`.
  - Do **not** open the annotation CSVs. Unzip the annotations archive, but leave the files unread.
  - Record in `data/deam/SOURCE.md`: the URL, the date, the archive checksums, the licence text as published, and the file counts.
  - **If access fails or the licence forbids research use, stop and report to Luke** with the fallbacks from spec §4 (PMEmo, then the "1000 songs" set).
- [x] **Step 2: Write the failing tests**

```python
def test_excerpts_schema_and_inclusion(): df = excerpts(); assert df.song_id.max() < 2000 and df.group.notna().all() and len(df) > 1500
def test_load_excerpt_mono_both_rates(tmp_path):  # synthetic 48 kHz stereo 8-bit wav
    y22, y44 = load_excerpt(make_wav(tmp_path, sr=48000, channels=2, sampwidth=1))
    assert y22.ndim == y44.ndim == 1 and abs(len(y44) / len(y22) - 2) < 0.01
def test_ratings_refused_without_goahead(tmp_path):
    f = tmp_path / "freeze.md"; f.write_text("# freeze\nno go-ahead yet\n")
    with pytest.raises(FreezeError): load_ratings(f)
def test_ratings_load_with_goahead_on_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(deam, "DEAM", fixture_dir_with_synthetic_static_csv(tmp_path))
    f = tmp_path / "freeze.md"; f.write_text("Go-ahead: Luke, 2026-11-10\n")
    assert list(load_ratings(f).columns) == ["song_id", "valence", "arousal"]
```

- [x] **Step 3: Run.** Expected: FAIL.
- [x] **Step 4: Implement.** `load_ratings` checks the go-ahead line first, then reads the static averaged-song CSV(s) and renames `valence_mean`/`arousal_mean` (column names stripped of whitespace). Audio decoding goes through `librosa.load(..., sr=None, mono=True)`, then resampling to each rate.
- [x] **Step 5: Run.** Expected: 4 passed.
- [x] **Step 6: Commit** `feat(perception): DEAM loader with ratings gate` (code only; data stays gitignored).

### Task 5: Representation levels per excerpt (resumable)

**Files:**
- Create: `src/perception/__init__.py`, `src/perception/levels.py`, `tests/perception/test_levels.py`

**Interfaces:**
- Consumes:
  - `F.extract`, `F.prepare`, `F.BANDS`, `F.FEATURE_NAMES`
  - `export_web.probe_design(feats, stats)`
  - `src.ear.listen`, `Source`
  - `src.deam.excerpts`, `load_excerpt`
- Produces:
  - `TR_START = 5`
  - `l0(y22) -> np.ndarray (69,)`: acoustic 60 + structural 8 means over TR ≥ 5, plus raw log RMS of the un-normalized y22 (named `raw_log_rms`)
  - `l1(y44) -> np.ndarray (66,)`: per-channel mean of `env` (32), per-channel modulation depth (32), mean phon, 95th-percentile phon. Ears are averaged; frames before 7.5 s are dropped.
    - Modulation depth = power of the 10 ms envelope band-passed 0.5–20 Hz (2nd-order Butterworth, zero-phase) divided by the squared envelope mean.
    - **This L1 summary is the proposal Task 6 freezes.**
  - `xbar(y22, stats) -> np.ndarray (340,)`: mean over TR ≥ 5 of `probe_design(F.extract(F.prepare(y22)), stats)`
  - `load_bundle(sub) -> dict(stats, W_region (340, R), region_names, acoustic_cols=slice(0, 300), structural_cols=slice(300, 340))`, read from `data/web/sub-XXX/meta.json` + `w_region.f32`
  - `l2(xbar_rows (n, 340), bundle) -> dict(acoustic=(n, R), structural=(n, R))`: block products `xbar[:, cols] @ W_region[cols]`
  - `L0_NAMES`, `L1_NAMES`
  - CLI `python -m src.perception.levels --workers 4`: writes `data/perception/levels/<song_id>.npz` (skips existing ones), then `data/perception/levels.npz` (song_id, group, L0, L1, xbar) and `data/perception/skipped.csv` (song_id, reason)

`load_bundle` converts `meta["bands"][band]` into the tuple form `probe_design` expects: `stats[band] = (np.array(mean), np.array(std), np.array(nuisance))`, with nuisance of shape (6, F·5). `stats` for `xbar` comes from **sub-001's** bundle. Every listener's bundle has identical feature stats and nuisance betas, because the design is subject-independent. Assert this across sub-001…005 in `load_bundle`, and fail loudly if any differ.

- [x] **Step 1: Write the failing tests**

```python
def test_l2_is_xbar_times_weights():  # the span property spec §5.4 relies on
    b = load_bundle("001"); xb = xbar(y22_synth, b["stats"])[None]
    X = probe_design(F.extract(F.prepare(y22_synth)), b["stats"])[TR_START:].mean(0)
    full = X @ b["W_region"]; parts = l2(xb, b)
    assert np.allclose(parts["acoustic"] + parts["structural"], full, atol=1e-5)

def test_raw_rms_survives_normalization():
    assert l0(0.5 * y22_synth)[-1] < l0(y22_synth)[-1] and np.allclose(l0(0.5 * y22_synth)[:68], l0(y22_synth)[:68], atol=1e-6)

def test_l1_level_sensitive():
    assert l1(0.1 * y44_synth)[64] < l1(y44_synth)[64]  # mean phon

def test_bundle_stats_identical_across_listeners():
    s = [load_bundle(f"00{i}")["stats"] for i in range(1, 6)]  # asserts inside load_bundle; smoke here

def test_short_or_silent_excerpt_skipped(tmp_path):
    assert process_one(silent_path) == ("skip", "silent") and process_one(short_8s_path) == ("skip", "shorter than 9 s")
```

`process_one(path) -> tuple[str, str | dict]` is the per-song worker the CLI maps over. It returns `("ok", arrays)` or `("skip", reason)`.

- [x] **Step 2: Run.** Expected: FAIL.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run.** Expected: 5 passed.
- [x] **Step 5: Full extraction (no ratings involved).** Run `.venv/Scripts/python -m src.perception.levels --workers 4` with sleep disabled; it resumes if interrupted. Expected: `levels.npz` with > 1,500 rows and no NaN. Report the skip counts by reason and the wall time.
- [x] **Step 6: Commit** `feat(perception): L0/L1/L2 representations per excerpt`.

### Task 6: Freeze file — STOP for Luke's go-ahead

**Files:**
- Create: `docs/superpowers/specs/2026-11-XX-perception-freeze.md` (use that day's date)

This is a human gate: nothing after this task opens ratings until Luke adds the go-ahead line.

- [x] **Step 1: Write the freeze file** (one page). It must contain:
  - the spec commit hash
  - the inclusion rule and the counts from `skipped.csv`
  - L0/L1/L2 definitions with dimensions, including the exact L1 summary from Task 5 and the ear constants, quoted from `constants.py` by name and value
  - TR window, split seed (`SEED`), test fraction 0.3, the ridge alpha grid `np.logspace(-2, 5, 30)`, 5-fold `GroupKFold`
  - 2,000 bootstrap resamples, 5,000 test-set permutations, 1,000 random projections
  - the theory baseline (Task 7)
  - H1–H3 verdict rules verbatim from spec §5.3–5.4
  - a final line: `Go-ahead: (Luke writes "Go-ahead: Luke, YYYY-MM-DD" here)`
- [x] **Step 2: Commit** `docs(perception): pre-registration freeze`.
- [x] **Step 3: Stop and report to Luke** with a 5-line summary of the freeze. Continue with Task 7 (synthetic data only) while waiting. Task 8 must not start until `load_ratings` stops raising.

### Task 7: Readout and statistics library (synthetic data only)

**Files:**
- Create: `src/perception/readout.py`, `tests/perception/test_readout.py`

**Interfaces:**
- Produces:
  - `split(groups: np.ndarray, seed=SEED) -> tuple[np.ndarray, np.ndarray]` (train and test indices; `GroupShuffleSplit`, test_size 0.3)
  - `fit_readout(X_train, y_train, groups_train) -> Pipeline` (`StandardScaler` → `RidgeCV(alphas=np.logspace(-2, 5, 30), cv=GroupKFold(5))`). Constant columns are dropped before scaling, and the kept-column mask is stored on the pipeline as `.keep_`.
  - `heldout(model, X_test, y_test) -> dict(r2, r, pred)`
  - `boot_diff(y, pred_a, pred_b, n=2000, seed) -> dict(diff, lo, hi)` (paired bootstrap over test songs of R²_a − R²_b)
  - `perm_p(y, pred, n=5000, seed) -> float` (permute y against fixed held-out predictions; one-sided on r)
  - `random_projection_r2(xbar_tr, xbar_te, y_tr, y_te, W_region, blocks, n=1000, seed) -> np.ndarray (n,)`. Each block's Gaussian matrix has W_region's block shape, with each column rescaled to that W column's norm.
  - `theory_baseline(L0, L1, names) -> dict(arousal=(n,), valence=(n,))`:
    - arousal = z(mean_phon) + z(onset_rate) + z(tempo)
    - valence = z(mode) + z(key_clarity)
    - z uses training stats
    - each score is mapped to the rating scale by a 1-D least-squares fit on train
  - `verdict(lo, hi) -> str` ∈ {"supported", "not supported", "inconclusive"}:
    - "supported" if lo > 0
    - "not supported" if hi < 0
    - "inconclusive" otherwise

- [x] **Step 1: Write the failing tests**

```python
def test_split_respects_groups(): tr, te = split(groups); assert not set(groups[tr]) & set(groups[te])
def test_readout_recovers_planted_signal(): assert heldout(fit_readout(Xtr, ytr, gtr), Xte, yte)["r2"] > 0.5
def test_constant_block_no_nan():  # sub-003-like structural pathway
    m = fit_readout(np.c_[Xtr, np.ones(len(Xtr))], ytr, gtr); assert np.isfinite(heldout(m, np.c_[Xte, np.ones(len(Xte))], yte)["r2"])
def test_perm_p_null_uniformish(): assert 0.2 < perm_p(y, rng.standard_normal(len(y)), seed=1) < 1.0
def test_random_projection_shapes_and_norms(): ...  # n draws, finite; block column norms equal W's
def test_verdict(): assert verdict(0.01, 0.2) == "supported" and verdict(-0.1, 0.1) == "inconclusive"
```

- [x] **Step 2: Run.** Expected: FAIL.
- [x] **Step 3: Implement.**
- [x] **Step 4: Run.** Expected: 6 passed.
- [x] **Step 5: Commit** `feat(perception): ridge readout, bootstrap, permutation, random projections`.

### Task 8: Hypotheses run (after go-ahead only)

**Files:**
- Create: `src/perception/run.py`, `tests/perception/test_run.py`
- Output: `data/perception/results.json`, `data/perception/results.md`

**Interfaces:**
- Consumes: Tasks 4, 5, 7.
- Produces: `run(freeze_path) -> dict`, and CLI `python -m src.perception.run --freeze <path>`. One split is shared by every model.
  - **H1:** arousal, L0 vs L0 + L1 → `boot_diff`, `verdict`.
  - **H2:**
    - At L0, for each rating: acoustic block (60 + `raw_log_rms`) vs structural block (8).
    - At L2, per listener, for each rating: acoustic vs structural pathway.
    - Supported at a level if valence favours structural *and* arousal favours acoustic, each with an interval excluding 0. For L2 the overall verdict needs ≥ 4 of 5 listeners.
  - **H3:** per listener and rating: L2 (2R dims) R² vs `random_projection_r2`. Supported per listener if above the 95th percentile; overall supported if ≥ 4 of 5.
  - **Descriptive:** the page readout L1 + L2 held-out R² per listener; theory baseline R²; `perm_p` for every model.
  - **Exploratory (labelled):** per-listener differences. Continuous ratings are left out of this plan (spec §5.5 allows it).

- [x] **Step 1: Write the failing test.** `test_run_end_to_end_on_fixture` uses a 120-song synthetic levels file plus synthetic ratings with a planted L1→arousal signal, and asserts H1 = "supported" and that every verdict is one of the three strings.
- [x] **Step 2: Run.** Expected: FAIL.
- [x] **Step 3: Implement.** `results.md` holds tables and verdicts only, with no interpretation prose.
- [x] **Step 4: Run the test.** Expected: PASS.
- [x] **Step 5: Real run.** Confirm `load_ratings` passes (the go-ahead is present), then run the CLI. Report the verdict table to Luke exactly as produced.
- [x] **Step 6: Commit** `feat(perception): pre-registered hypothesis run` (code; commit `results.md` only if Luke asks). Done for code (758b24e); `results.md`/`results.json` deliberately NOT committed (Luke has not asked).

### Task 9: Paper figures

**Files:**
- Create: `src/perception/figures.py`
- Output: `figures/perception/{hierarchy,dissociation,projection_null,psychophysics}.png`

**Interfaces:**
- Consumes: `results.json`; Task 3 `listen`.
- Produces: `main()` → the four PNGs at 300 dpi, matching the existing `src/figures.py` style:
  - `hierarchy`: held-out R² with 95% CIs, for L0, L0 + L1, L2 (listener mean ± range) and theory, × valence/arousal
  - `dissociation`: acoustic vs structural R², per rating, per level/listener
  - `projection_null`: a histogram per listener with an L2 marker
  - `psychophysics`: ITD and ILD against azimuth for 4 channels from the ear model, with the Woodworth curve overlaid

- [x] **Step 1: Implement and run** `.venv/Scripts/python -m src.perception.figures`. Expected: 4 files, and none of them blank (each figure's axes contain data). Done: five figures exist (the four above plus `generalisation.png` from the H4 addendum).
- [x] **Step 2: Commit** `feat(perception): paper figures`.

### Task 10: Page readout and web export

**Files:**
- Create: `src/perception/export.py`, `tests/perception/test_export.py`
- Output per listener: `data/web/sub-XXX/readout.json`, `readout_boot.f32` (+ `.gz.b64.txt` via `export_web.save`)
- Modify: `CLAUDE.md` (Current status and Resume here: the perception layer's state and file locations)

**Interfaces:**
- Consumes: Tasks 5, 7, 8 (the shared split).
- Produces: `readout.json` with fields:
  - `groups`: {ear: L1_NAMES, acoustic: region names, structural: region names}
  - `scaler_mean`, `scaler_std`, `keep`
  - `coef` and `intercept` for each rating
  - `heldout`: {valence: {r2, r}, arousal: {r2, r}}
  - `n_train`, `n_test`
  - `train_min`, `train_max` per feature, for out-of-evidence greying
  - `config`: {azimuth 0, distance 1, preset "studio", gain_db 0}
  - `n_boot`: 200

  `readout_boot.f32` has shape (200, 2, n_kept + 1): coefficients plus intercept from 200 refits on bootstrap resamples of training songs (resampled by group). This is what the page's stability rule needs (spec §7.3).

- [x] **Step 1: Write the failing test.** `test_export_roundtrip` exports from a fixture, reloads the JSON and f32, recomputes predictions for 5 fixture songs from the exported numbers alone, and asserts they equal `model.predict` to 1e-5. It also checks the bootstrap array shape.
- [x] **Step 2: Run.** Expected: FAIL.
- [x] **Step 3: Implement** and run `python -m src.perception.export` for listeners 001–005.
- [x] **Step 4: Run.** Expected: PASS. Then update CLAUDE.md.
- [x] **Step 5: Commit** `feat(perception): page readout export`.

---

## Addendum (2026-11-02, Luke approved): PMEmo generalisation test (H4)

Luke questioned DEAM's ecological validity (all Creative Commons production music). H4 tests whether the DEAM-trained readouts transfer, unchanged, to commercial chart music. It is pre-registered in the freeze before any rating is read. PMEmo 2019 (Zhang et al. 2018) is on disk at `data/pmemo/` (chorus/<musicId>.mp3, metadata.csv, annotations/); the archive sha256 is acfcd0e5…, and Luke downloaded it from the official Google Drive link.

**H4 (frozen in the freeze file):**
- Representations: L0, L1 and L2 are computed for every PMEmo chorus exactly as for DEAM (0°, 1 m, "studio", 0 dB; TR rows ≥ 5; same skip rules).
- Readouts: the fitted DEAM-training-split models from Task 8 are applied unchanged. No PMEmo data is used for fitting, alpha selection or standardization.
- Metric: Pearson r between predictions and PMEmo static ratings, per rating, with a 95% CI from 2,000 bootstrap resamples of PMEmo songs (SEED). PMEmo's rating scale differs from DEAM's, so the metric is r, not R².
- Primary model: the page readout (L1 + L2), per listener. Per rating, H4 is supported if ≥ 4 of 5 listeners have CI lo > 0, not supported if ≥ 4 of 5 have CI hi < 0, and inconclusive otherwise.
- Descriptive: L0, L0 + L1, L2 alone and the theory baseline.
- **PMEmo audio is commercial and is never published or served by the website.**

### Task 11: PMEmo loader, levels and freeze H4

**Files:**
- Create: `src/pmemo.py`, `tests/perception/test_pmemo.py`
- Modify: `src/perception/levels.py` (add `--dataset {deam,pmemo}`; PMEmo cache at `data/perception/pmemo_levels/`, aggregate `data/perception/pmemo_levels.npz`, `pmemo_skipped.csv`), `docs/superpowers/specs/2026-11-02-perception-freeze.md` (H4 section plus a PMEmo counts placeholder table)

**Interfaces:**
- `src.pmemo.excerpts() -> DataFrame(song_id, path, artist, group)`. `group` follows the same normalization as DEAM but is unused for fitting.
- `src.pmemo.load_ratings(freeze_path) -> DataFrame(song_id, valence, arousal)`. It uses the same go-ahead gate as `src.deam` (reuse `src.deam`'s gate function; do not duplicate the regex).
- Audio decoding reuses `src.deam.load_excerpt`.

Task 8 gains H4 (run after H1–H3 on the same fitted models). Task 9 gains a `generalisation.png` panel. Task 10 is unchanged.
