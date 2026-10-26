# CAMP Memory Model — Plan 1: Python reference model and power check

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python implementation of the memory model (hippocampal index + cortical consolidation over CAMP song patterns, Braak degeneration, recall readouts) and a power-check harness on simulated listeners.

**Architecture:** `src/memory/` is a small package. Song content comes from CAMP's fitted joint model, split exactly into melody/rhythm/timbre components; a Hebbian index binds content to context; slow cortical weights learn by exposure and replay; degeneration zeroes units by Braak stage; readouts score recall per component, gist, episode and familiarity. The analysis module fits survival with and without encoding variables and tests the gain with a permutation null.

**Tech Stack:** Python 3.13, numpy, pandas, scipy (existing venv); pytest (added).

**Spec:** `docs/superpowers/specs/2026-10-25-memory-model-design.md`. This is plan 1 of 3; plan 2 (importers, inferred salience, song matching) and plan 3 (JS port, memory panel, what-if control, scenes) follow after this one ships.

## Global Constraints

- Hypotheses, readouts and analysis are fixed by the spec §3 and §5.8; do not add or drop readouts.
- Every free parameter lives in `Params` (Task 1) with a default and a sweep range; no magic numbers elsewhere.
- Precision mapping α and salience-inference weights are **swept, never fitted** (spec §5.6, §11).
- Deterministic given a seed: every function that draws randomness takes a `numpy.random.Generator`.
- Units are the listener's FDR-significant joint voxels; Braak stages per unit come from `src.ablation` (`voxel_table`, `stage_of`, `ORDERINGS["alzheimers"]`); stage 7 = not in the ordering, never removed.
- Feature groups (spec §5.1): melody/harmony = chroma ×12, key_clarity, mode, harmonic_change, roughness; rhythm/dynamics = tempo, pulse_clarity, onset_rate, beat_cv, rms; timbre = mel ×32, mfcc ×13, centroid, bandwidth.
- No real listening data in this plan; songs are GTZAN tracks; nothing leaves the machine.
- Run tests with `.venv/Scripts/python -m pytest tests/memory -q` from `acoustic-encoding/`.

## Review Focus

1. **Listener with very few lit voxels in a stage** (e.g. no stage-1–3 units): stage removal must be a no-op for empty stages, not a divide-by-zero. Test in Task 5.
2. **A song played once, decades ago** (one weak episode): recall must degrade toward foil level, never return NaN from a zero-norm pattern. Test in Task 6.
3. **Silent or near-constant audio in the song bank** (zero-variance component): normalization must not produce NaN. Test in Task 2.
4. **Everything removed** (stage 6 with all units staged ≤ 6 in a toy layout): readouts return foil-level scores, not errors. Test in Task 6.
5. **Survival regression with fewer songs than predictors**: the analysis must refuse with a clear error rather than report a perfect R². Test in Task 8.

---

### Task 1: Package, parameters, and exact component split

**Files:**
- Create: `src/memory/__init__.py`, `src/memory/params.py`, `src/memory/songs.py`
- Create: `tests/memory/__init__.py`, `tests/memory/test_songs.py`
- Modify: `requirements.txt` (add `pytest`)

**Interfaces:**
- Produces: `Params` (frozen dataclass) with fields and defaults: `n_index=2000, index_k=40, ctx_dim=512, n_segments=4, segment_scans=5, cue_scans=3, eta_cortex_exposure=0.002, eta_replay=0.01, replays_per_month=4, n0=5.0, decay_index=0.5, decay_cortex=0.15, kappa=1.0, alpha=(0.0, 0.0, 0.0), theory="tt", years_per_stage=2.0, gap_years=10.0, n_foils=50`; and `SWEEP: dict[str, list]` giving each field's sweep values (alpha grid = `[(0,0,0),(1,1,1),(2,0,0),(0,2,0),(0,0,2)]`, theory = `["sct","tt"]`, gap_years = `[0,10,30,50]`, others ×0.5/×1/×2 of default).
- Produces: `GROUPS: dict[str, list[str]]` (keys `"melody"`, `"rhythm"`, `"timbre"`, feature names from `src.features.FEATURE_NAMES`) and `component_patterns(X: np.ndarray, W: np.ndarray, delays: int = 5) -> dict[str, np.ndarray]` returning `(n_scans, n_units)` arrays whose sum equals `X @ W`, where `X` is a design matrix in `src.export_web.probe_design` column order (per band, delay-major) and `W` is `(n_cols, n_units)`.

- [ ] **Step 1: Add pytest and write the failing tests**

```python
# tests/memory/test_songs.py
import numpy as np
from src import features as F
from src.memory.songs import GROUPS, component_patterns

def test_groups_cover_every_feature_once():
    names = F.FEATURE_NAMES["acoustic"] + F.FEATURE_NAMES["structural"]
    grouped = sum(GROUPS.values(), [])
    assert sorted(grouped) == sorted(names) and len(grouped) == 68

def test_components_sum_to_full_prediction():
    rng = np.random.default_rng(0)
    X, W = rng.normal(size=(20, 340)), rng.normal(size=(340, 30))
    parts = component_patterns(X, W)
    assert set(parts) == {"melody", "rhythm", "timbre"}
    assert np.allclose(sum(parts.values()), X @ W)
```

- [ ] **Step 2:** Run `.venv/Scripts/python -m pip install pytest` then `.venv/Scripts/python -m pytest tests/memory/test_songs.py -q`. Expected: FAIL, `ModuleNotFoundError: src.memory`.
- [ ] **Step 3:** Implement `Params`, `SWEEP`, `GROUPS`, `component_patterns` (build a column mask per group from the probe_design layout: band offset + delay·F + feature index).
- [ ] **Step 4:** Run the tests. Expected: 2 passed.
- [ ] **Step 5:** Commit `feat(memory): params and exact component split`.

### Task 2: Song bank from GTZAN through CAMP

**Files:**
- Modify: `src/memory/songs.py`
- Test: `tests/memory/test_songs.py`

**Interfaces:**
- Consumes: `component_patterns`, `Params`; `src.export_web.probe_design`; `src.encoding.build_design` (for training stats); `src.validate.paths`; `src.ablation.voxel_table`, `stage_of`, `ORDERINGS`.
- Produces: `SongBank` dataclass: `ids: list[str]`, `content: np.ndarray (n_songs, 3, n_segments, n_units)` (component axis order melody, rhythm, timbre), `unit_stage: np.ndarray (n_units,) int`, `sub: str`. `segment(parts, p) -> np.ndarray (3, n_segments, n_units)`: averages scans into `p.n_segments` segments of `p.segment_scans`, then z-scores each unit across the bank and scales each segment to unit norm (zero-norm segments stay zero). `build_song_bank(sub: str, per_genre: int, p: Params) -> SongBank`, cached at `data/memory/songbank_sub-{sub}_{per_genre}.npz`; songs = first 30 s of GTZAN tracks 0..per_genre-1 per genre via `F.load_audio(path, duration=30)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_segment_shapes_and_unit_norm():
    rng = np.random.default_rng(1)
    parts = {g: rng.normal(size=(20, 30)) for g in ("melody", "rhythm", "timbre")}
    seg = segment(parts, Params())
    assert seg.shape == (3, 4, 30)
    assert np.allclose(np.linalg.norm(seg, axis=-1), 1.0)

def test_segment_zero_input_stays_finite():
    parts = {g: np.zeros((20, 30)) for g in ("melody", "rhythm", "timbre")}
    assert np.isfinite(segment(parts, Params())).all()
```

- [ ] **Step 2:** Run; expected FAIL (`segment` undefined).
- [ ] **Step 3:** Implement `SongBank`, `segment`, `build_song_bank` (bank-wide z-scoring happens in `build_song_bank`; `segment` does averaging and per-segment norm only).
- [ ] **Step 4:** Run tests (2 new pass). Then run `.venv/Scripts/python -c "from src.memory.songs import build_song_bank; from src.memory.params import Params; b=build_song_bank('001', 2, Params()); print(b.content.shape, sorted(set(b.unit_stage)))"`. Expected: `(20, 3, 4, 1786)` and stages within 1..7.
- [ ] **Step 5:** Commit `feat(memory): GTZAN song bank through CAMP`.

### Task 3: Context codes, episodes, simulated listeners

**Files:**
- Create: `src/memory/history.py`
- Test: `tests/memory/test_history.py`

**Interfaces:**
- Produces: `ContextSpace(dim: int, rng)` with `.code(attr: str, value: str) -> np.ndarray` (fixed sparse binary code per value, 4 blocks of `dim//4` for `scene`, `people`, `occasion`, `age`, 5% active, cached) and `.encode(scene, people, occasion, age_bin) -> np.ndarray (dim,)`.
- Produces: `Play(song: int, month: int, salience: float, context: dict[str,str])`; `Episode(song: int, month: int, n: int, salience: float, context: np.ndarray, strength: float)`; `episodes_from_plays(plays, ctx: ContextSpace, p: Params) -> list[Episode]` binning plays per (song, month): `n` = count, `salience` = max, `strength = (1 - exp(-n/p.n0)) * (1 + p.kappa * salience)`; context = encode of the bin's most frequent context.
- Produces: `simulate_listener(rng, n_songs: int, years: int, kappa: float, bank_size: int) -> list[Play]` — songs drawn without replacement from the bank; per song a first month uniform in the period, total plays lognormal (median 8), spread over a random active span; salience per song from Beta(2, 5) with occasional spikes; contexts drawn from small vocabularies (6 scenes, 5 people sets, 4 occasions).

- [ ] **Step 1: Write the failing tests**

```python
def test_binning_counts_and_saturation():
    ctx = ContextSpace(512, np.random.default_rng(0))
    plays = [Play(0, 3, 0.2, {"scene": "backyard"})] * 4 + [Play(0, 4, 0.9, {"scene": "beach"})]
    eps = episodes_from_plays(plays, ctx, Params(kappa=0.0))
    assert [(e.month, e.n) for e in eps] == [(3, 4), (4, 1)]
    assert eps[0].strength == pytest.approx(1 - np.exp(-4 / 5))

def test_same_value_same_code():
    ctx = ContextSpace(512, np.random.default_rng(0))
    assert np.array_equal(ctx.code("scene", "beach"), ctx.code("scene", "beach"))
    assert ctx.code("scene", "beach").sum() == round(0.05 * 128)

def test_simulated_listener_is_deterministic():
    a = simulate_listener(np.random.default_rng(3), 20, 10, 1.0, 100)
    b = simulate_listener(np.random.default_rng(3), 20, 10, 1.0, 100)
    assert a == b and len({pl.song for pl in a}) == 20
```

- [ ] **Step 2:** Run; expected FAIL (module missing).
- [ ] **Step 3:** Implement as specified (missing context attributes encode as the value `"none"`).
- [ ] **Step 4:** Run; 3 passed.
- [ ] **Step 5:** Commit `feat(memory): context codes, episodes, simulated listeners`.

### Task 4: The network — encoding, replay, forgetting, recall pathways

**Files:**
- Create: `src/memory/network.py`
- Test: `tests/memory/test_network.py`

**Interfaces:**
- Consumes: `Params`, `Episode`, `SongBank.content`.
- Produces: `MemoryModel(n_units: int, p: Params, rng)` holding `M_hc (n_index, n_units)`, `M_hk (n_index, ctx_dim)`, `L (n_index, n_index)` (index sequence links), `W_cc (n_units, n_units)`, `W_seq (n_units, n_units)`, `W_ck (n_units, ctx_dim)`, masks `alive_units`, `alive_index`, `entry_gain: float`.
  - `encode(ep: Episode, content: np.ndarray (3, S, U))`: precision gains `g_c = 1 + p.alpha[c] * ep.salience`; pattern per segment = unit-normed `Σ_c g_c content[c]`; one fresh sparse index code per segment (`index_k` active), Hebbian outer products scaled by `ep.strength` into `M_hc`, `M_hk` (context), `L` (segment s → s+1); exposure learning into `W_cc`, `W_seq` at `eta_cortex_exposure * ep.strength`. Stores the episode's codes for replay.
  - `step_month(rng)`: `replays_per_month` replays sampled ∝ current index strength; each reactivates its stored codes through `M_hc` (alive parts only) and trains `W_cc`, `W_seq` at `eta_replay`, and `W_ck` with the episode context (`theory="sct"`) or the song's running mean context (`theory="tt"`). Then decay: index matrices ×`(t/(t+1))**decay_index`, cortical ×`(t/(t+1))**decay_cortex` with t months since each trace's last reinforcement (track per episode for index, global month for cortex).
  - `recall(cue: np.ndarray (U,)) -> Recall` with fields `content (S-1, U)` (segments 2..S), `context (ctx_dim,)`: index path `a = kwta(entry_gain * M_hc[:, alive] @ cue)`, roll `L` for the rest, read `M_hcᵀ a`, `M_hkᵀ a`; cortex path `x = W_cc @ cue`, roll `W_seq`, context `W_ckᵀ x`; combine by summing the unit-normed outputs of both paths (zero-norm paths contribute zero).

- [ ] **Step 1: Write the failing tests** (toy bank: 5 random songs, 3 components, 4 segments, 40 units; `Params(n_index=300, index_k=10, ctx_dim=64)`)

```python
def test_fresh_episode_recalled_through_index():
    m, bank, eps = toy_model(theory="tt")      # helper encodes one episode per song
    r = m.recall(cue_of(bank, 0))
    assert corr(r.content, true_rest(bank, 0)) > 0.8

def test_without_replay_cortex_alone_is_weak():
    m, bank, eps = toy_model(replay_months=0)
    m.alive_index[:] = False
    assert corr(m.recall(cue_of(bank, 0)).content, true_rest(bank, 0)) < 0.3

def test_replay_consolidates_into_cortex():
    m, bank, eps = toy_model(replay_months=120)
    m.alive_index[:] = False
    assert corr(m.recall(cue_of(bank, 0)).content, true_rest(bank, 0)) > 0.5

def test_tt_keeps_only_gist_context_in_cortex():
    sct, tt = toy_model(theory="sct", replay_months=120, two_contexts=True)[0], toy_model(theory="tt", replay_months=120, two_contexts=True)[0]
    for m in (sct, tt): m.alive_index[:] = False
    # episode-specific context recoverable from cortex under SCT only
    assert episode_margin(sct) > episode_margin(tt) + 0.1
```

- [ ] **Step 2:** Run; expected FAIL.
- [ ] **Step 3:** Implement `MemoryModel` and test helpers (`toy_model`, `cue_of`, `true_rest`, `corr`, `episode_margin`) in `tests/memory/conftest.py`.
- [ ] **Step 4:** Run; 4 passed. If a threshold fails, tune defaults in `Params` (not in code) and record the change in the commit message.
- [ ] **Step 5:** Commit `feat(memory): index + cortex network with replay and forgetting`.

### Task 5: Degeneration by Braak stage

**Files:**
- Modify: `src/memory/network.py`
- Test: `tests/memory/test_network.py`

**Interfaces:**
- Produces: `MemoryModel.degenerate(stage: float, unit_stage: np.ndarray, rng)`: continuous stage in [0, 6]. Stage I (0→1): `entry_gain = 1 - frac`. Stage II (1→2): kill a fixed random order of index units, fraction `frac`. Stages III–VI (s−1→s): kill units with `unit_stage == s` in a fixed random order, fraction `frac`; units of earlier completed stages all dead; stage 7 never. Idempotent for the same stage; empty stages are no-ops.

- [ ] **Step 1: Write the failing tests**

```python
def test_stage_zero_is_noop():   ...  # alive masks all True, entry_gain == 1
def test_stage_two_kills_all_index(): ... # after degenerate(2.0): alive_index.sum() == 0, entry_gain == 0
def test_stage_six_spares_stage_seven(): ... # unit_stage with 7s: after degenerate(6.0) only stage-7 units alive
def test_empty_stage_is_noop(): ...  # unit_stage without any 3s: degenerate(2.5) vs (2.0) same alive_units
```

(Each body builds `toy_model()`, calls `degenerate` with a fixed `unit_stage`, and asserts the line in its comment.)

- [ ] **Step 2–4:** Run (FAIL), implement, run (4 passed).
- [ ] **Step 5:** Commit `feat(memory): Braak-order degeneration`.

### Task 6: Readouts

**Files:**
- Create: `src/memory/readouts.py`
- Test: `tests/memory/test_readouts.py`

**Interfaces:**
- Consumes: `Recall`, `SongBank`, `MemoryModel`, episodes.
- Produces: `score_song(m: MemoryModel, bank: SongBank, song: int, episodes: list[Episode], foils: np.ndarray, cue: str = "full") -> dict[str, float]` with keys `melody`, `rhythm`, `timbre`, `gist`, `episode`, `confidence`, `familiarity`. Component g: mean over segments 2..S of corr(recalled, true component g) minus the same against foil songs (mean over `foils`). `cue="melody"` uses only the melody component of the first `cue_scans` scans. Gist: corr(recalled context, the song's mean episode context) minus mean over other songs' mean contexts. Episode: corr with the best-matching own episode context; confidence: best minus second best (0 if one episode). Familiarity: `1 - PE(song)/mean PE(foils)`, PE = mean over segments of `‖c_{s+1} − W_seq c_s‖²` on alive units. All correlations return 0 for zero-variance inputs.

- [ ] **Step 1: Write the failing tests**

```python
def test_intact_learned_song_beats_foils():   # toy_model with replay; every component score > 0.3, familiarity > 0
def test_unlearned_song_is_foil_level():      # score of a never-encoded song: |component| < 0.15
def test_single_ancient_play_degrades_finite(): # one weak episode, 600 months of decay: all scores finite and < intact scores
def test_everything_removed_is_finite():      # all units stage <= 6, degenerate(6.0): scores finite, components ≈ 0 (|x| < 0.1)
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (4 passed).
- [ ] **Step 5:** Commit `feat(memory): recall readouts`.

### Task 7: Life simulation, survival curves, validation checks

**Files:**
- Create: `src/memory/simulate.py`
- Test: `tests/memory/test_simulate.py`

**Interfaces:**
- Consumes: Tasks 1–6.
- Produces: `STAGES = np.arange(0, 6.01, 0.5)`; `run_life(bank: SongBank, plays: list[Play], p: Params, rng) -> pd.DataFrame` with one row per (song, readout, cue, stage) and column `score`: encode episodes month by month with `step_month` in between, then `gap_years*12` months of `step_month` with no plays, then for each stage in `STAGES` call `degenerate` and `score_song` for every played song under both cues. `survival(df) -> pd.DataFrame` one row per (song, readout, cue) with `S` = trapezoid area under score-vs-stage, divided by the stage-0 score (NaN if stage-0 score ≤ 0, reported, not dropped silently).

- [ ] **Step 1: Write the failing tests** (small real bank: `build_song_bank("001", 2, Params())`, 12 songs, 8 years)

```python
def test_run_is_deterministic(): ...          # same seed -> identical DataFrame
def test_validation_v1_ribot(): ...           # songs with plays >= 30 and last play > 5 y before onset have higher mean S (melody) than songs with <= 3 plays
def test_validation_v2_gist_outlasts_episode(): ... # mean S(gist) > mean S(episode) under theory="tt"
def test_validation_v3_content_outlasts_context(): ... # mean S over melody/rhythm/timbre > mean S(episode)
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (4 passed). These are the spec's V1–V3; if one fails, report it rather than tuning until it passes, since a failure means the model does not reproduce the clinical baseline.
- [ ] **Step 5:** Commit `feat(memory): life simulation and survival curves`.

### Task 8: Primary test and power harness

**Files:**
- Create: `src/memory/analysis.py`, `src/memory/power.py`
- Test: `tests/memory/test_analysis.py`

**Interfaces:**
- Consumes: `survival`, `simulate_listener`, `run_life`.
- Produces: `incremental_r2(y: np.ndarray, base: np.ndarray, block: np.ndarray, n_perm: int, rng) -> dict` with `r2_base`, `r2_full`, `delta`, `p` (permute `block` rows across songs; `p = (k+1)/(n_perm+1)`); raises `ValueError("need more songs than predictors")` when `len(y) <= base.shape[1] + block.shape[1] + 1`. `design_matrices(surv: pd.DataFrame, bank, plays, p) -> (geo, enc)`: geo = per song × component, the share of the component's intact content energy in each Braak stage 1–7 (drop stage 7 as reference); enc = log plays, months active, months since last play, mean and max salience, precision gain `alpha[c] * mean salience`. `python -m src.memory.power --sub 001 --sims 100 --kappa 0 0.5 1 2` runs simulated listeners (60 songs, 15 years, `per_genre=10` bank) for each kappa and writes `data/memory/power_sub-001.csv` (kappa, readout, detection rate at p < 0.05, median delta) and prints it.

- [ ] **Step 1: Write the failing tests**

```python
def test_detects_planted_block():
    rng = np.random.default_rng(0); n = 80
    base, block = rng.normal(size=(n, 3)), rng.normal(size=(n, 2))
    y = base @ [1, 0.5, 0] + 0.8 * block[:, 0] + rng.normal(scale=0.5, size=n)
    assert incremental_r2(y, base, block, 500, rng)["p"] < 0.01

def test_null_block_false_positive_rate():
    rng = np.random.default_rng(1)
    ps = [incremental_r2(rng.normal(size=60), rng.normal(size=(60, 3)), rng.normal(size=(60, 2)), 200, rng)["p"] for _ in range(100)]
    assert np.mean(np.array(ps) < 0.05) <= 0.10

def test_too_few_songs_raises():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="more songs than predictors"):
        incremental_r2(rng.normal(size=5), rng.normal(size=(5, 3)), rng.normal(size=(5, 2)), 10, rng)
```

- [ ] **Step 2–4:** Run (FAIL), implement, run (3 passed).
- [ ] **Step 5:** Smoke-run `.venv/Scripts/python -m src.memory.power --sub 001 --sims 5 --kappa 0 1`. Expected: CSV written, detection rates in [0, 1]. Do not interpret 5-sim numbers.
- [ ] **Step 6:** Commit `feat(memory): incremental-R² test and power harness`.

---

**Not in this plan:** the full power run and parameter sweeps (a compute job after review), importers and salience inference (plan 2), website (plan 3).
