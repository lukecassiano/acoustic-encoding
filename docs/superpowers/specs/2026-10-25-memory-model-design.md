# CAMP memory model — design

Status: **approved by Luke, 2026-10-25** ("Document holds up"). Next: implementation plan. Nothing in here is built yet.
Source conversation: brainstorming session 2026-10-25. Decisions are marked **(Luke)**; choices Claude made are marked **(Claude's call)** so they can be checked.

## 1. Intent

Luke is drawn to two things at once: what music does cognitively, and how the circumstances of hearing it shape that. His example, *Yeppers* (Ray Barbee and the Matson 2): a song tied to his dad, to summer evenings in the backyard and, later, to surfing together. When it plays, what arrives first is a multisensory scene (beach, sun, almost-heard gulls, backyard grass, warm Santa Ana wind), then, sometimes, a reconstructed specific episode (probably a Father's Day around age 7–8, family friends over, the song on his dad's new stereo through to the backyard). **(Luke)**

The model asks whether **the way a song is encoded** (its cortical representation, and the circumstances it was heard in) **predicts what survives Alzheimer's-type degeneration**, or whether **the disease's own spatial order** decides it regardless. **(Luke)** Both answers are results. The only failure is an uninterpretable one. **(Luke)**

It serves two outputs:
- a **CAMP website feature** (interactive, explorable), and
- a **personal research paper**, written only if the simulations give novel findings (CLAUDE.md, "Two papers"). **(Luke)**

It is **not** part of the Music 108 paper beyond, at most, a future-work paragraph. **(Luke)**

## 2. What a song memory is (agreed playback)

1. **Many listening episodes hung on one tune.** One cortical pattern per song (CAMP's predicted response), linked to many episodes across years, each with its own multisensory scene, people and occasion. (This is multiple-trace theory's picture.) **(Luke agreed)**
2. **Episodes are not equal.** Emotion, novelty and significance stamp some harder: each episode has a **salience**, not just a play count. **(Luke agreed)**
3. **Recall is layered and ordered.** First a blended sensory **gist** ("every summer at once"), then sometimes a **specific episode**, which is reconstructed with uncertainty, not replayed. The model reports episode **confidence** separately from gist strength. **(Luke agreed)**
4. **Degeneration strips specifics before atmosphere.** Episodes depend on the hippocampal index (Braak I–II, early); gist consolidated in cortex lasts longer; the tune itself, in auditory cortex (Braak V–VI), lasts longest. **(Luke agreed)**
5. **Circumstances act on the music's processing, not alongside it.** The same song heard on Father's Day with his dad is encoded differently from the same song heard in passing. **(Luke)** Mechanism, in predictive-processing terms: context sets **precision weights** on parts of the incoming music during encoding. **(Claude's proposal; Luke did not object)**

## 3. Hypotheses (fixed before any results)

- **H-encoding:** how a song was encoded (its CAMP representation × the precision of its encoding circumstances, plus its listening history) predicts which components survive degeneration, **beyond** where the disease strikes.
- **H-disease:** survival is explained by the disease's spatial order alone; encoding adds nothing once geography is known.

**Primary test.** For each song × component, survival `S` = area under its recall-vs-Braak-stage curve. Fit `S ~ geography` (the share of the component's cortical content in each Braak stage), then `S ~ geography + encoding` (component precision gains, salience, play count, spacing, age at encoding, years since). H-encoding is supported if the incremental R² of the encoding block exceeds the 95th percentile of a null in which encoding variables are permuted across songs, **and** this holds across the precision mappings and both consolidation theories (§5.6, §7). Otherwise report H-disease, with the minimum detectable effect from the power check (§7) so the null is interpretable.

**Validation, expected by construction (reported as validation, not findings):**
- V1 Ribot gradient: remote, well-rehearsed songs survive better than recent ones.
- V2 gist outlasts specific episodes.
- V3 musical content outlasts context (the clinical picture; Jacobsen et al. 2015).

**Predictions that are not built in (candidate findings):**
- P1 Which component (melody/harmony, rhythm, timbre) survives longest, given each listener's empirical CAMP feature→region map and Braak stages.
- P2 Interactions: does salience protect gist but not episodes? Does precision only help when the music is strongly represented?

**Baseline H-disease has to beat:** Analysis B already found that Alzheimer's-order removal spares CAMP's representation no better than random removal (AD area 0.48–0.52, all five listeners).

## 4. Non-goals and anti-goals

Non-goals: a model of any real patient; clinical claims; an affective feature band (affect enters only through salience/precision parameters); personal brain models (CAMP stays fitted to the five ds003720 listeners); hosting copyrighted audio; Claude writing paper prose.

Anti-goals (would make it a failure even if it runs):
- **An uninterpretable result** — no power check, so a null can't be told from a blind test. **(Luke)**
- **Assumptions presented as findings** — anything that only restates a built-in mechanism must be labelled validation.
- Context scenes appearing to change CAMP's predictions (CAMP is audio-only).

## 5. The model

Notation: one CAMP listener at a time (default sub-001; all five as replication). None of the five is Luke, so for his own run each of the five cortices stands in, and results are reported only where they agree across all five **(Claude's call)**. Cortical units = that listener's FDR-significant joint voxels (V ≈ 1.8–2.7k), each with a Harvard-Oxford label and Braak stage. Time step: songs are chunked into 1.5 s scans (CAMP's TR).

### 5.1 Content (data-grounded)

CAMP is linear, so a song's predicted cortical response splits exactly into feature-group components: `c_t = c_t^mel + c_t^rhy + c_t^tim`, each `c_t^g = X_t[g] W[g]` (delayed feature columns of group g times their joint-model weights). The three groups cover all 68 features, so nothing is left over. **Groups (Claude's call):**
- melody/harmony: chroma ×12, key clarity, mode, harmonic change, roughness
- rhythm/dynamics: tempo, pulse clarity, onset rate, beat CV, RMS
- timbre: log-mel ×32, MFCC ×13, centroid, bandwidth

### 5.2 Context (assumed, multisensory)

Each episode e has a context vector `κ_e` of fixed random sparse codes, one per attribute value, concatenated: **scene** (visual setting, ambient sound, touch/temperature — the scene is multisensory, per Luke), **people**, **occasion/significance**, **age at encoding**. Dimension ≈ 512 **(Claude's call)**.

### 5.3 Hippocampal index (theoretical)

`N_h` ≈ 2,000 index units **(Claude's call)**. Each episode gets a sparse random code `h_e` (k-winners, ~2% active). One-shot Hebbian binding, scaled by encoding strength: cortex→index `W_ch`, index→content `W_hc`, index→context `W_hk`. Retrieval: cue → `k-WTA(W_ch · cue)` → content `W_hc a` and context `W_hk a`. CAMP has no reliable hippocampal signal; this layer is assumed and labelled so.

### 5.4 Cortex (consolidation)

Slow weights within cortex: auto-associative `W_cc` (pattern completion) and sequential `W_seq` (scan t → t+1, so a 5 s cue can continue the melody). Learned by (a) small direct learning on every play (exposure) and (b) **replay**: stored episodes reactivate through the index and train cortex, at a rate proportional to index strength.

Two consolidation theories, switchable **(Claude's design from the CLAUDE.md spec)**:
- **Standard (SCT):** replay trains cortex on content + the episode's full context, so remote memories become hippocampus-independent.
- **Trace transformation (MTT/TTT):** replay trains cortex on content + **gist context** (the running average of that song's episode contexts). Specific episodes stay index-dependent.

### 5.5 Listening history → episodes

A history is a list of plays (song, timestamp, optional context and salience). Plays are binned per song per month **(Claude's call)**: a bin with n plays forms one episode whose encoding strength saturates with n (`1 − exp(−n/n₀)`) times salience σ. Forgetting: weights decay as a power law of time since last reinforcement, faster for the index than for cortex.

### 5.6 Precision weighting (circumstances act on processing)

During encoding of episode e, content is gain-modulated per component: `c_e = Σ_g (1 + α_g · π(σ_e)) · c^g`, where `π` maps salience to precision and `α = (α_mel, α_rhy, α_tim)` is the **precision mapping**. α is not fitted; it is **swept** over a grid (including all-equal, each component dominant, and zero), and conclusions are reported only where they hold across the grid. **(Claude's proposal, §2.5)**

### 5.7 Degeneration

Braak stages advance in simulated years after onset. Stage I removes the index's entorhinal entry (cortex→index pathway, progressively); stage II removes index units; stages III–VI remove cortical units by their voxel's Braak stage (as in `src/ablation.py`, `BRAAK`, Pascoal et al. 2020 mapping); labels the ordering doesn't name (stage 7) are never removed. Within a stage, removal is continuous.

### 5.7b Life timeline (Claude's call)

A simulated life runs in monthly steps: listening history first, then a gap with no new plays, then Braak progression. The gap (0–50 years) and the progression rate are swept parameters, not fixed. For Luke's own run the history ends at the present and the model ages forward; the paper reports results across the gap range.

### 5.8 Recall readouts (per song, at each degeneration point)

Cue = the first 5 s of the song (all components), and separately a **melody-only** cue. Both pathways run (index and cortex) and are combined.
- **Component fidelity** (melody, rhythm, timbre): correlation of the recovered pattern with the true component over the rest of the song, minus the mean correlation with foil songs.
- **Gist:** recovered context vs the song's mean context, against other songs.
- **Episode + confidence:** the best-matching stored episode, and the margin over the runner-up (reconstruction uncertainty).
- **Familiarity:** reduced prediction error — `1 − PE(song)/PE(novel)`, with PE the cortex's next-scan prediction error.

## 6. Data

- **Paper simulations:** songs = GTZAN tracks (on disk, 1,000 × 30 s) through CAMP; **simulated listeners** with generated histories and planted effects. No audio leaves the machine.
- **Luke's own run:** his Spotify extended streaming history (JSON) and/or Apple play activity (CSV). **Requested; about 30 days. (Luke)** Songs matched to Apple 30 s previews (iTunes Search, local server). Pre-streaming songs (e.g. *Yeppers*) entered by hand **after the exports arrive (Luke)**.

## 7. Power and sensitivity (before any real-data run)

- **Power check:** simulated listeners with planted encoding effects of known size. The primary test must detect them with ≥ 80% power; report the minimum detectable effect. **No real-data run before this passes.**
- **Sensitivity:** sweep every free parameter (index size and sparsity, learning rates, decay exponents, replay rate, n₀, π, α grid, consolidation theory); for each conclusion, report the fraction of parameter space where it holds.
- **Known risk:** CAMP's lit voxels are almost all superior temporal (Braak V–VI), so geography varies little; the power check decides whether the test can work at all.

## 8. Website (CAMP)

A **memory panel** beside the brain, clearly labelled *simulation*:
- **What-if control (Luke, 2026-10-25):** pick a song (current stimulus), set "Y plays over Z years", a scene, people and significance; see recall per component, gist, episode confidence and familiarity under the Braak dial. Stores nothing about the user.
- **Scene screen:** the user's own photos (kept local) or scenes drawn in the page (field, bar, city at night, home with family), with ambient sound played to the user only. The scene **fades and blurs as a simulation output** as index integrity falls; the UI says scenes do not affect CAMP's predictions.
- **History import** (later): Spotify / Apple files, processed locally.

## 9. How it gets made (Claude's calls unless marked)

- **Python reference** in `src/memory/` (numpy) for the paper's simulations, sweeps and power analysis; **JS port** in `web/memory.js` for the website; a **parity test** like CAMP's.
- Importers: Spotify extended history fields (`ts`, `ms_played`, `master_metadata_track_name`, `master_metadata_album_artist_name`, `spotify_track_uri`, `skipped`, …) and Apple's play-activity CSV, tolerant column mapping, tested on synthetic files until the real exports arrive. **The Apple column names must be checked against the real export.**
- Order: power-check harness and Python model → importers and song matching → JS port + memory panel + scenes.

## 10. Left to the builder

Exact parameter defaults and sweep ranges; k-WTA details; how the two pathways combine (sum vs max); foil-set size (default: 50 GTZAN tracks not in the history); how simulated histories are generated (default: power-law play counts, salience drawn from a skewed distribution, planted effects as specified by the power check); scene artwork. Each must be written into the implementation plan before coding.

## 11. Context and salience for streamed plays (Luke, 2026-10-25)

A Spotify or Apple export records *what* played and *when*, not the scene, people or significance.

- **Primary: inferred salience (Luke: "inference is likely key").** Salience σ for each monthly episode is inferred from listening behaviour. Candidate signals **(Claude's call, to verify against the real export's fields)**: deliberate starts (clicked or searched, vs autoplay or track-done), completion (`ms_played` against track length), skips, repeat density (the same song replayed within a short window), saved/offline plays. How the signals combine is an assumption, so its weights are **swept** like the precision mapping, and every conclusion is reported across that range.
- **Also supported (Luke):** **neutral** context (σ constant for all streamed plays; also the control condition showing what inference adds) and **annotated** context (Luke gives a scene, people and significance to chosen songs; annotations override inference).
- Hand-entered pre-streaming songs carry annotated context by definition.
