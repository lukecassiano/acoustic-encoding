# CAMP perception layer — design

Status: **draft, awaiting Luke's approval.** Nothing in here is built. When Luke approves, §5 (hypotheses and analysis plan) is frozen *before* any affect ratings are loaded.

**Freeze procedure (Claude's call):** the items §5 leaves to the builder (the L1 summary, ear-model constants used for DEAM) go into a dated one-page file, `docs/superpowers/specs/2026-11-XX-perception-freeze.md`, written before any rating file is opened. Luke gets a short summary and the go-ahead is his. The ratings are not loaded until he gives it. The file is kept beside this spec as the pre-registration record.
Source conversation: brainstorming session 2026-11-02. Decisions are marked **(Luke)**; choices Claude made are marked **(Claude's call)** so they can be checked.

## 1. Intent

CAMP today is an encoding model: audio → features → predicted BOLD on cortex, fitted to ds003720. Everything it outputs is *predicted sensory response*. Luke wants it to become more of a **perception model with its own information flow**: a simulated listener with ear-like structures, in a virtual space, whose variables can be manipulated to watch processing change from low-level sensory response up to a predicted affective response. **(Luke)**

It serves, in order:
1. **The Music 108 final paper** (due 2026-12-15). It is "a necessary inclusion" both for interpreting the model and for getting **actual perceptual results**, which the project does not have now. **(Luke)** Luke will fit the extra analysis into the word budget himself. **(Luke)**
2. **The CAMP website**, as the interpretive face of the same model. **(Luke)**

It is not part of the personal research paper (memory/degeneration) unless Luke later says so.

## 2. The information flow

Four levels. Each has a different evidential status, and the page and paper always say which one a claim comes from. **(Claude's proposal; Luke agreed)**

| Level | What it is | Status tag | Evidence behind it |
|---|---|---|---|
| Physics | Speaker → air → head: direction, distance, level, speaker type | **biology** (with physics) | Acoustics; not fitted |
| L1 Ear | Head filtering per ear → cochlear filterbank → hair-cell stage → loudness; brainstem interaural cues (ITD, ILD) | **biology** | Hearing science; checked against psychophysics (§6), and tested as a front end in §5 |
| L2 Cortex | CAMP's fitted banded ridge → predicted response, split into the **acoustic** and **structural** pathways | **data** | ds003720 fit (existing) |
| Readout | Linear readout from L1 + L2 → predicted **arousal** and **valence** (§7.3) | **data** (learned from human ratings, held-out tested) | Affect-rated music dataset (§4) |

A theory-driven mapping (arousal from loudness, onset rate and tempo; valence from mode and key clarity) is kept only as a **baseline the learned readout must beat**, tagged **theory**. **(Luke)**

Framing for the paper and the page **(Claude's proposal)**: the ear and brainstem are a fixed, physiologically specified front end; CAMP is the learned middle level; the readout asks where in the hierarchy affect becomes linearly readable. This is the logic of Kell et al. 2018 (hierarchy in auditory cortex), applied to perception instead of brain data.

### What each level can and cannot see

- ds003720 was played diotically over headphones with every clip RMS-normalized (`prepare()` → 0.1321). **CAMP's weights contain no information about direction or level.** Location and loudness reach cortex in this model only through the ear model's mechanistic layer, never through CAMP. The page says so. **(Claude's call, stated in conversation)**
- Speaker type changes the spectrum, so it **does** reach CAMP's acoustic pathway and the readout. **(Claude's observation)**
- **CAMP's input is the speaker-filtered source signal, *before* head filtering**, mixed to mono across speakers and passed through `prepare()`, exactly as for a probe. Rationale: ds003720 was delivered over headphones, which bypass the head's spatial filtering, so feeding CAMP a head-filtered signal would put it outside its training conditions. Consequences:
  - Location never reaches CAMP.
  - Level is normalized away by `prepare()`.
  - The **studio monitor preset is an exact identity filter**, so at the default configuration CAMP's input is the raw excerpt, and the page and the paper's analysis agree.
  - The page states all of this. **(Claude's call, resolving a builder-check question)**

## 3. Goals, non-goals, anti-goals

**Goals**
- A Python reference ear model plus a JS port, with a parity test (the same pattern as `features.py` / `features.js`). **(Claude's call, Luke agreed)**
- A pre-specified test of the hierarchy against human affect ratings (§5). Its results go into the Music 108 paper. **(Luke)**
- Two website tabs: **Listen** (3D room, live processing) and **Analysis** (after playback: readout, reasons, cortical maps in 3D and 2D). **(Luke)**
- A readout that is **technically fluent but plain**: it directly explains a perceptual prediction, with reasons rooted in theory, biology and data. It is simple, gives an obvious, interpretable and intriguing result, and avoids overcompensation. **(Luke)**

**Non-goals (v1)**
- Room acoustics: no reverberation, wall absorption or room size. Free field only. **(Claude's call)** (The first mockup showed room-size and absorption controls; they are dropped for simplicity.)
- Elevation: speakers sit on the floor plane at ear height. Azimuth and distance only. **(Claude's call)**
- Server-side inference, MERT or TRIBE.
- An LLM writing the reason sentences. They are generated from templates over exact linear contributions (§7.3). **(Claude's call)**

**Anti-goals: failure even if it works**
- A page that looks like it is selling rather than reporting: dashboards, decorative effects, confident numbers without their held-out accuracy beside them. **(Luke)**
- Presenting a mechanistic assumption (contralateral bias, tonotopy, per-source colouring) as if it came from data.
- Faking a fast brain: showing CAMP's 1.5 s, hemodynamically lagged prediction as if it were millisecond processing.
- An affect test whose positive result is built in by construction (see §5.4).
- Waves that are tacky or superfluous. **(Luke)**

## 4. Affect data

**Dataset: DEAM** (MediaEval Database for Emotional Analysis in Music, Aljanaki, Yang & Soleymani 2017). About 1,800 excerpts and songs with song-level ("static") valence and arousal ratings on 1–9 scales, plus continuous 2 Hz ratings. **(Luke chose to test against an affect-rated dataset; DEAM is Claude's pick.)**
- **The builder must verify**: access, licence for annotations and audio (the audio is from Free Music Archive, Jamendo and MedleyDB, under per-track Creative Commons licences), the exact song count, and whether artist IDs exist for grouped splits.
- Fallbacks if DEAM fails access or licence: PMEmo (Zhang et al. 2018, 794 choruses), then the Soleymani et al. 2013 "1000 songs" set. **(Claude's call)**
- **Use the static ratings** for the primary tests. The continuous ratings are exploratory only (§5.5). **(Claude's call)**
- Out-of-distribution note for the paper: CAMP was fitted on GTZAN clips; DEAM's genre mix differs.

## 5. Hypotheses and analysis plan (frozen on approval, before ratings are loaded)

### 5.1 Representations per excerpt

All levels are computed on each DEAM excerpt rendered from a single front speaker at 0° azimuth and 1 m, with studio-monitor (flat) type and default gain. The binaural stage is therefore uninformative for DEAM: **DEAM tests the ear model's monaural front end (cochlea, hair cells, loudness), not its localization.** Localization is checked only against psychophysics (§6). **(Claude's call)**

**Which signal feeds which level (Claude's call, resolving a builder-check question):** L0 and L2 are both computed from the **raw excerpt**, never the ear-rendered signal. This keeps §5.4's argument that L2 lies inside the span of L0 exact, and it matches CAMP's headphone training conditions (§2). Only L1 sees the ear-rendered signal.

**Excerpt handling (Claude's call):** each excerpt (DEAM's are about 45 s) is treated as one clip starting at t = 0, exactly as the probe path in `web/model.js` and `export_web.probe_design` do: `prepare()` (2 s fades, RMS normalization), then the clip-boundary nuisance. Clip-level means use TRs from index 5 onward (≥ 7.5 s), so all five delays are filled and the onset transient is excluded. The same TR window applies to L0, L1 and L2.

- **L0, raw features**: `features.py` `extract()` on the excerpt, mean over the TR window. 68 values (60 acoustic + 8 structural), plus the raw log RMS of the **un-normalized** signal. *Fairness rule:* L0 gets level information too, so any L1 advantage cannot come merely from L1 skipping normalization. **(Claude's call)** For H2 the raw log RMS counts toward the **acoustic** pathway, where CAMP's own log RMS feature lives.
- **L1, ear**: clip-level summaries of the ear model's output: per-channel mean and per-channel temporal-modulation energy of the hair-cell output, plus summary loudness (mean and 95th percentile, in sone or phon). The exact summary is the builder's choice but **must be fixed in the plan before ratings are loaded.**
- **L2, cortex**: CAMP's predicted response for each listener (sub-001…005), split by pathway. The predicted response is the sum of acoustic-column and structural-column contributions in `w_region`, averaged over the excerpt's TRs. Representation = region means, acoustic pathway (R values) and structural pathway (R values), per listener.

### 5.2 Readout

The readout is a ridge regression from a level's features to each rating (valence and arousal separately):

$$\hat y = \beta_0 + \sum_{\ell} \mathbf{w}_\ell^\top \mathbf{x}_\ell$$

- **Split**: 70/30 train/test by song, grouped by artist if artist IDs exist. Inside train, regularization is set by 5-fold CV. **The test set is scored once, at the end.** **(Claude's call)**
- **Metric**: held-out R² and r for each rating, with 95% bootstrap intervals over test songs.
- **Null**: ratings permuted across songs (5,000 permutations) to give a p value for each model.
- Standardization uses training statistics only (the same rule as CLAUDE.md "Encoding model").

### 5.3 Hypotheses

- **H1, the ear front end earns its place.** For arousal, held-out R²(L0 + L1) > R²(L0). Supported if the bootstrap interval of the difference excludes 0. Expected mechanism: perceptual loudness and cochlear modulation energy track arousal better than RMS and the mel spectrum.
- **H2, a pathway double dissociation.** Within L0 (the feature bands themselves) *and* within L2 (through the cortical projection), the structural pathway predicts valence better than the acoustic pathway does, and the acoustic pathway predicts arousal better than the structural pathway does. Supported per level if both differences hold with intervals excluding 0. Evaluate L2 per listener; the verdict needs ≥ 4 of 5 listeners. This is the link to Analysis A: the acoustic/structural division in auditory cortex would have measurable perceptual consequences.
- **H3, the cortical projection preserves affect information beyond chance.** See §5.4 for why this is framed against random projections, not against L0.

### 5.4 Non-circularity (state explicitly in the paper)

- CAMP's prediction is a **linear function of L0 features** (delayed copies, then weights). Averaged over a clip, L2 lies essentially inside the span of L0. A linear readout from L2 therefore cannot beat L0 except through its bottleneck acting as a prior. "Cortex beats raw features" is not a meaningful test, and is not run as a hypothesis.
- H3 instead asks whether the **brain's** projection keeps more affect information than an **arbitrary** projection of the same dimensionality. Compute held-out R² for L2 (2R dimensions per listener), then repeat with 1,000 random linear projections of the delayed L0 features to 2R dimensions. Projections are built **block-wise**: acoustic columns → R dimensions and structural columns → R dimensions, each matched in column norms to that pathway's block of `w_region`, so the null preserves the two-pathway structure and tests only the brain's weighting within it **(Claude's call)**. Supported if L2's R² exceeds the 95th percentile of the random-projection distribution in ≥ 4 of 5 listeners.
- H2 at L0 is partly built in by feature choice (the structural band contains `mode`). That is why H2 is tested at both L0 and L2, and why the paper must say that only the L2 version says anything about cortex.
- Every L2 result is reported per listener, because the structural pathway ranges from absent (sub-003) to dominant (sub-005). Whether sub-005's cortex carries more valence information than sub-003's is **exploratory**, not a hypothesis.

### 5.5 Exploratory (labelled as such)

- Continuous (2 Hz) ratings against TR-level representations.
- Per-listener differences in readout accuracy.
- The theory baseline's R² against the learned readout's.

### 5.6 What counts as an interpretable result

Each hypothesis has a supported, not-supported or inconclusive verdict. A clean null is a result. For example, "the ear model adds nothing over RMS for arousal" belongs in the paper. **(Consistent with Luke's stated anti-goal on the memory model: the only failure is an uninterpretable result.)**

## 6. Ear model specification

Python reference: `src/ear.py`. JS port: `web/ear.js`. Parity test: `tests/parity/` (pattern of `parity.mjs`). Everything should be plain DSP. **(Claude's call)**

1. **Source and speaker type.** Four presets, each a fixed biquad chain plus a directivity pattern. **(Luke asked for speaker type; presets are Claude's call.)**
   - laptop: high-pass at about 300 Hz with a presence bump; near-omnidirectional
   - studio monitor: flat 50 Hz – 20 kHz; mild forward directivity
   - subwoofer: low-pass at about 120 Hz; omnidirectional
   - horn: band-limited midrange; narrow beam (cardioid-to-supercardioid); emits a cone in the visualization

   The exact corner frequencies and patterns are the builder's call but are identical in Python and JS.
2. **Propagation.** Inverse-square level (−6 dB per doubling of distance) plus propagation delay. No air absorption.
3. **Head.** **Brown & Duda 1998 spherical-head model** (analytic, so parity is easy): it yields ITD and the frequency-dependent ILD. It has no pinna cues, so front/back confusions are faithful and the cone of confusion is visible. That is a feature for teaching, and the page says so. KEMAR HRTFs are an optional later upgrade. **(Claude's call: the browser's built-in HRTF panner cannot be reproduced in Python, which would break parity.)**
4. **Outer and middle ear.** A fixed transfer-function filter, derived from ISO 226 equal-loudness data or a standard middle-ear curve. The builder chooses and cites it.
5. **Cochlea.** A gammatone filterbank with about 32–64 ERB-spaced channels from 50 Hz to 8 kHz, per ear.
6. **Hair cells.** Half-wave rectification → power-law compression (exponent about 0.3) → low-pass (about 1 kHz) → downsample for display.
7. **Loudness.** A simplified time-varying loudness: specific loudness from ERB-channel excitation with compressive exponent about 0.23, summed to sone and converted to phon. It is stated as simplified, not ISO 532. The builder may substitute a citable standard if it is portable to JS.
8. **Brainstem.** ITD per channel by interaural cross-correlation (a Jeffress-style coincidence array), ILD per channel by level difference. With several speakers, cues are computed **from the mixture** (as a real brainstem would get them). Known source positions are overlaid, never fed in. **(Claude's call)**
9. **Level calibration.** A constant maps digital level to dB SPL. Default: a DEAM excerpt at its own level, default gain, 1 m → about 70 dB SPL. The constant is shown on the page. **(Claude's call)**

**Psychophysical checks** (the ear model's validation; in tests and as a paper figure if Luke wants one):
- the duplex theory crossover: ITD dominates below about 1.5 kHz, ILD above
- the maximum ITD of about 0.65–0.7 ms at 90°
- front/back ambiguity in the spherical head
- loudness growth of roughly doubling per +10 dB above about 40 phon
- equal-loudness shape at low frequencies

## 7. Website

The page builds on the existing `web/index.html` (three.js brain, fsaverage6, slice viewer, worker, CAMP cream / bold serif style). The new modules are `web/ear.js`, `web/room.js` and `web/readout.js`. Whether the Listen and Analysis tabs live in `index.html` or a second page is the builder's call. `index.html` is already about 1,600 lines.

### 7.1 Listen tab (live, sensory only)

Layout: **the room beside the brain, with one row of ear readouts underneath** (mockup "layout-v2", Tab 1). **(Luke asked to keep both the environment and the analysis; arrangement is Claude's proposal)**
- **3D room**: a floor plane, a listener head at the centre, and draggable speakers (up to 3; **Claude's call**). Each speaker has a source (upload, Apple-preview song search, or one of the existing synthetic probes), gain, and type preset. The camera orbits.
- **Waves (Luke)**: kept simple. Each speaker emits faint expanding rings; ring brightness follows the playing signal's envelope and fades with distance; horn speakers emit a cone. At the head, two small traces show the waveform arriving at each ear, so the near ear's earlier, stronger arrival is visible (ITD and ILD made concrete). Nothing else is animated.
- **Ear readouts row**: per-ear level, ITD and ILD, a compact cochleagram per ear, and loudness in phon.
- **Live brain, two speeds (Claude's call, stated in conversation):**
  - The **fast sensory layer** is driven by the ear model at display rate. Cochlear channel energy maps onto a **tonotopic gradient along Heschl's gyrus** (frequency to position along its long axis, using `vertex_region` Heschl's vertices on fsaverage6). A contralateral bias sets each hemisphere's gain. Both are mechanistic assumptions, tagged **biology / assumed** on the page. The builder sets the bias magnitude from the literature and cites it.
  - The **CAMP layer** fades in behind it at the TR rate (1.5 s) with its hemodynamic lag. On-page text says it is slower because it was fitted to blood-flow signal.
  - **CAMP is computed once per play (Claude's call, resolving a builder-check question).** When Play is pressed, the CAMP prediction is computed for the configuration at that moment, using the existing precompute-and-sync design and the unchanged, parity-tested `features.js`. Moving a speaker, changing gain or changing type mid-play updates the ear layer live. Only speaker type can change CAMP's input, so changing it marks the CAMP layer "stale: replay to update". The Analysis tab scores the configuration at play start and says so if the room changed during playback. A streaming rewrite of `features.js` is out of scope: it would break parity with `features.py`, which the paper depends on.
- **Audible output** uses the *same* rendering as the model (speaker filter → propagation → spherical head), via an AudioWorklet or ConvolverNodes, so what you hear is what the model processes. **(Claude's call)**

### 7.2 Analysis tab (after playback stops)

- **First thing seen**: one dot on an arousal × valence plane, with the readout's **held-out accuracy printed beside it** (R² for each axis), for the currently selected listener.
- **Under it, one or two plain sentences of reasons**, each tagged **biology**, **data** or **theory**. Example of the register: *"Predicted high arousal, mostly from loudness at the ear (68 phon) and dense onsets; predicted low valence, carried by the structural pathway: minor mode, unstable harmony."* **(Luke)**
- **One click deeper:**
  - contribution per group: ear, acoustic pathway, structural pathway (§7.3), with unstable ones marked "shared"
  - the theory baseline's prediction for comparison
  - the time-averaged cortical response in **3D** and **2D**

  The 2D view is a flattened cortical map if a flat patch for fsaverage6 (or a resampled fsaverage flat patch) is available and portable. Otherwise it is the existing slice viewer plus a 2D schematic of auditory regions (core, belt, parabelt, L and R). The builder verifies availability. **(Luke asked for "potentially both 3D and 2D")**
- **Out-of-evidence notices**: when an input falls outside what the readout was trained on, the readout is greyed and the page says the model has no evidence there. Examples: a speaker not at the front 0° / 1 m position, level outside DEAM's range, or a speaker type other than flat. **(Claude's call, Luke agreed in playback)**

### 7.3 Readout on the page

- **The page readout reads perception from the simulated listener's own internal states: L1 (ear) + L2 (cortex), with no L0. (Luke, option c, 2026-11-02.)** Raw features (L0) remain the paper's baseline and appear in §5's tests, but are not part of the page's readout.
  - Reason: L2 is a linear re-weighting of L0, so a readout containing both would divide credit between them according to regularization, not data (builder check).
  - This readout matches the intent in §1: cortical activity adding up to an emergent perception.
- Fit it with the same split and CV as §5.2, per listener. Its held-out R² is the number printed beside the dot. Report it in the paper alongside H1–H3 as a descriptive result, not a hypothesis. Readout weights for each listener are exported into the web bundle (an extension of `src/export_web.py`).
- **Contributions.** Contribution per group: $c_g = \mathbf{w}_g^\top(\mathbf{x}_g - \bar{\mathbf{x}}_g)$, with groups **ear** (L1), **acoustic pathway** and **structural pathway** (L2).
  - Within L2, the arithmetic is exact, but credit between the two pathways still depends on how correlated they are. A reason therefore appears only if it is **stable**: its group ranks in the same position (largest or second-largest |c_g|) and keeps the same sign in ≥ 90% of 200 bootstrap refits of the readout over training songs. **(Claude's call)**
  - Otherwise the page says the credit is shared, e.g. "carried jointly by the ear and both cortical pathways".
- **Reason sentences** are templates filled from the stable contributions and their top features, with no free text.

## 8. Constraints

- **Paper deadline 2026-12-15.** The paper-critical path is: DEAM access → `src/ear.py` → freeze the L1 summary → `src/perception.py` (levels, readout, H1–H3) → figures. The website runs in parallel; it never blocks the paper. **(Luke wants parallel, ambitious progress; Claude's ordering of the critical path)**
- **Proposal** due about two weeks after 2026-10-25 (CLAUDE.md). Luke decides whether the perception layer goes in it.
- **Writing rule**: Claude does not write the paper's prose (CLAUDE.md "AI use"). The design, code, figures and checks are fine.
- **Existing Analysis A work still pending** (structural permutation importance, per-feature structural analysis, group summaries) is not replaced by this.
- **Environment**: Windows, `.venv`, no pytest. Tests are run as modules, in the style of `python -m tests.test_features`. The local server is `python -m src.serve_web`. Long jobs run with sleep disabled.
- **The published artifact** cannot serve raw binary (`.gz.b64.txt` convention), and song search works only from the local server.

## 9. Left to the builder

- Ear model constants within the ranges in §6, and the outer/middle ear curve (cited).
- The exact L1 clip-level summary (fixed before ratings are loaded).
- Speaker preset filter values.
- Contralateral bias magnitude (cited).
- The 2D cortical view's implementation (flat patch vs. schematic).
- File layout of the tabs.
- Visual detail within CAMP's existing style and the "reporting, not selling" rule.
