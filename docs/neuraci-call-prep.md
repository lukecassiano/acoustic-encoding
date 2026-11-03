# CAMP — technical briefing for the Neuraci Labs call

`~/projects/music-encoding/acoustic-encoding` · prepared 2026-11-02 · numbers from `data/derivatives/` as of today

---

## 0. The one-paragraph version (say this first)

CAMP is a **voxelwise encoding model**: it takes any piece of audio, turns it into 68 named, interpretable features every 1.5 s, and predicts the fMRI response in each 2 mm voxel of a specific listener's brain. It is fitted separately to each of five people who lay in a scanner listening to 540 fifteen-second music clips (OpenNeuro ds003720, Nakai et al.), and it is scored only on 60 clips it never saw during fitting. It reliably predicts auditory cortex (Heschl's gyrus, planum temporale/polare, superior temporal gyrus) in all five listeners, and nothing outside it survives a permutation test. Because it is linear, it runs entirely in the browser: upload a song, watch the predicted cortical response, and drag a dial that removes cortex in the order Alzheimer's tau spreads.

**The bridge to Neuraci in one line:** CAMP is the auditory version of what an olfactory AI needs — a map from stimulus structure to human neural/perceptual response, with a noise ceiling that says how much of the response is predictable at all — and audition and olfaction sit at *opposite ends* of the Braak degeneration sequence, which is exactly why smell is an early warning signal and music a late-preserved channel.

---

## 1. Neuraci Labs — what is public, and what you should not assume

What the public record supports (sources at the end):
- **Neuraci Labs Inc.** (neuracilabs.com), founder Aditi Bhatia, New York. Tagline: "Programming the Sense of Smell"; describes "a living system that learns how humans perceive the world through smell", combining "computational models" with "real human input". A homepage heading reads **"Human Smell Graph"** (no public description).
- A job listing describes a "foundational intelligence layer for human smell" feeding **therapeutics, diagnostics, consumer products and olfactory AI**. A Chief Scientific Officer was announced on LinkedIn (name not in what I could read).
- Their blog posts are field reviews: the olfactory pathway (receptors → bulb → piriform → amygdala/entorhinal/hippocampus), smell loss as an early marker for **Parkinson's and Alzheimer's** (with the honest caveat that specificity is too low for a smell test to diagnose alone), and no approved treatment for smell loss.
- A sibling consumer brand, **neuraci.com**, sells "functional fragrance" (sleep, focus) citing EEG/psychophysiology evidence; the Labs homepage links to it. Treat the relationship as likely but unconfirmed.
- **Not public:** the GNN architecture, what the "Human Smell Perception Graph" contains, data sources. That came from your founder call. Ask, don't assert.

**The closest public analogue they will certainly know:** Lee et al. 2023, *Science* — Google Research / Osmo's **Principal Odor Map (POM)**. A message-passing GNN reads each molecule as a graph (atoms = nodes with valence, degree, H-count, hybridization, charge, atomic number; bonds = edges), produces an embedding, and predicts ~138 odor descriptors. Its headline evaluation: on 400 novel odorants, the model's profile was closer to the panel mean than the **median panelist** was. OpenPOM is the open re-implementation. Graph-transformer follow-ups (e.g. GraphNOSE) report the known weakness: poor generalisation to molecules unlike the training set.

**How a "perception graph" plausibly differs from POM** (good questions to ask them): is the graph over *molecules* (chemistry), over *descriptors* (perceptual space, edges = co-rating), over *people* (individual differences, e.g. OR genotype, age, anosmia), or a heterogeneous graph linking all three? Where does the human data come from — trained panels, consumers at scale, clinical smell tests (UPSIT, Sniffin' Sticks)? Is any of it neural (EEG, fMRI, intranasal)?

---

## 2. How CAMP works, stage by stage

### 2.1 Data
- **ds003720** (Nakai, Koide-Majima & Nishimoto 2022, *Data in Brief*; methods in Nakai et al. 2021). Five listeners, 3 days each, whole-brain BOLD, **TR = 1.5 s**, 2 mm isotropic.
- 540 GTZAN clips, 15 s each, 10 genres, 2 s fade in/out, RMS-normalised. **480 training clips heard once; 60 test clips heard 4×.** Training and test never overlap.
- Per listener: 4,800 training time points (480 clips × 10 TRs) and 600 test time points.

### 2.2 Preprocessing (`src/preprocess.py`) — and why it is our own
The authors published cleaned responses (Zenodo), but they are **cortex-only and carry no voxel locations**, so you can't place a voxel in the brain, can't include hippocampus or amygdala, and can't assign Braak stages. So CAMP preprocesses the raw BOLD itself (ANTsPy + TemplateFlow + nilearn): rigid motion correction, regress out 6 motion parameters and their derivatives, detrend, **z-score each voxel within each run**, register the T1 to MNI152 (SyN) and pull the Harvard-Oxford atlas *into* each person's native space. The BOLD is never warped; analysis stays in native 2 mm voxels (~85–95k per person, ~10k subcortical). Not done: slice-timing (absorbed by the delays) and distortion correction (no fieldmaps → some orbitofrontal / anterior temporal dropout).

### 2.3 Features (`src/features.py`) — 68 per 1.5 s scan, all hand-built DSP
Audio at 22,050 Hz → STFT (2048 / 512) → frame features mean-pooled into TR bins.

| Band | Features | What they index |
|---|---|---|
| **Acoustic (60)** | log-mel spectrum ×32 | energy per frequency band — the cochlea's view; tonotopy |
| | MFCC 1–13 | spectral envelope shape — timbre |
| | spectral centroid, bandwidth | brightness, spread |
| | chroma ×12 | energy per pitch class — harmony content |
| | RMS | loudness |
| **Structural (8)** | tempo (log₂ BPM) | beat rate |
| | pulse clarity | how strongly one periodicity dominates the onset envelope (tempogram peak / lag 0) |
| | onset rate | note/event density per second |
| | beat-interval CV | beat irregularity (clip-level) |
| | key clarity | best correlation of chroma with 24 Krumhansl–Kessler key profiles |
| | mode | best major r − best minor r |
| | roughness | Plomp–Levelt / Sethares sensory dissonance over the 20 strongest spectral peaks |
| | harmonic change | flux of the tonal centroid (tonnetz) — how fast the harmony moves |

The split is the scientific point: acoustic = *what the ear receives*; structural = *how the music is organised* (rhythm, tonality, consonance). Every feature is plain enough that `web/features.js` reproduces it in the browser, verified by a parity test (all discrete stages exact, features within 1e-7 SD).

### 2.4 The model (`src/encoding.py`)
Per voxel *v*:

$$\hat y_v(t) = \sum_{d=1}^{5} z\big(x(t-d)\big)^\top w_{v,d}$$

- **Delays** d = 1…5 TRs (1.5–7.5 s): a finite-impulse-response HRF. Blood flow trails neural activity by seconds; instead of assuming a canonical HRF shape, each voxel learns its own lag profile. (Weight energy peaks at 3.0–4.5 s in all five listeners — physiologically right.)
- **z(·)**: features standardised with *training* statistics only; the same mean/SD is applied to test clips and to uploaded audio.
- **Nuisance**: every clip boundary has the same fade/cut, which drives a generic response no feature distinguishes. An onset impulse at each delay plus an intercept is regressed out of features and responses (training betas only).
- **Banded ridge** (himalaya `GroupRidgeCV`): minimise $\lVert y - X_a w_a - X_s w_s\rVert^2 + \lambda_a\lVert w_a\rVert^2 + \lambda_s\lVert w_s\rVert^2$. Each band gets its own penalty, so the 8 structural features aren't drowned by the 60 acoustic ones and collinearity between bands is handled explicitly. Penalties are chosen **per voxel** from 17 overall strengths (1 to 10⁸) × 9 band ratios.

**"How did you train it?"** There is no gradient descent. Ridge has a closed-form solution, $w = (X^\top X + \Lambda)^{-1}X^\top y$, solved for ~90k voxels at once. Hyper-parameters are picked by **leave-one-run-out cross-validation inside the training set** (12 runs → 12 folds), so test data never touch the fit. ~20 minutes per listener on a laptop CPU. Three fits per listener: acoustic-only, structural-only, joint.

**Why linear?** (1) 4,800 noisy samples per voxel is a small-data regime where linear models are hard to beat; (2) the nonlinearity lives in the feature extraction, so each weight means something ("this voxel responds to energy near 1 kHz, 4.5 s later"); (3) it's the standard in the field (Naselaris et al. 2011; Huth/Gallant lab; Dupré la Tour et al. 2025); (4) it runs in a browser. Deep features (MERT) are planned as a *benchmark* against it, not a component of it.

### 2.5 Evaluation (`src/validate.py`)
- **Accuracy**: Pearson r between predicted and measured response on the 60 test clips (averaged over their 4 repeats).
- **Noise ceiling** (Schoppe et al. 2016): from the 4 repeats, how much of each voxel's response repeats at all. It is the best r *any* model could reach. Accuracy is read against it. **This is the concept to bring to olfaction**: POM's "beats the median panelist" is a ceiling argument — inter-rater reliability bounds what a model can predict.
- **Significance**: 5,000 permutations of which test clip each prediction is paired with (keeps everything every clip shares, so it tests *clip-specific* prediction), Benjamini–Hochberg FDR q < 0.05 across voxels. Only voxels passing this are coloured in CAMP.
- **Variance partition**: unique acoustic = R²(joint) − R²(structural); unique structural = R²(joint) − R²(acoustic); shared = the rest. Cross-checked with himalaya's product-measure split.

### 2.6 Degeneration (`src/ablation.py`)
Braak tau stages from tau-PET (Pascoal et al. 2020, *Brain*), mapped by hand onto Harvard-Oxford regions (reviewed by you, 2026-10-12): I entorhinal/anterior parahippocampal → II hippocampus → III amygdala, fusiform, lingual → IV insula, inferior/middle temporal, PCC, temporal pole → V most association cortex incl. **STG and planum** → VI primary cortices incl. **Heschl's**. Remove voxels stage by stage; track what share of held-out explained variance remains. Control: the same regions removed in 5,000 random orders.

### 2.7 Web export (`src/export_web.py`, `web/`)
Prediction and surface projection are both linear, so they are folded together: vertex prediction = $x_t (W M^\top)$, where *M* averages the significant voxels under each fsaverage6 vertex (sampled at 5 depths through the cortical ribbon). Weights ship as float16, gzip+base64. A second parity test checks the browser's predictions against Python (error 2.9e-6 of RMS).

---

## 3. The interface, element by element (the questions from the founder call)

**What does the colour mean?** Predicted BOLD, signed, in **z units of that voxel's own response**. Pink: the model predicts this voxel above its *average level during music listening*; blue: below. Three things to say precisely:
1. **Zero is not silence or rest.** Every training sample was music, responses were z-scored within run and the intercept removed, so 0 = "what this voxel typically does while hearing music." Blue is not "deactivated," it's "less than usual for music."
2. **The scale is relative to the song.** The colour bar runs to the 99th percentile of |prediction| for the current audio, so colour intensity compares voxels and moments *within* a song. To compare songs, read the tick values (or use a contrast).
3. **Beige-tinted cortex** = voxels where the model beats chance but predicts ≈ 0 right now. **Untinted** = the model can't predict there; no claim is made.

**What does a BOLD level mean for perception?** BOLD is a slow (peaks ~4–6 s), indirect, vascular proxy for local neural population activity, driven mostly by synaptic input (Logothetis et al. 2001). A high prediction in Heschl's gyrus means *this sound has the properties this voxel's population is tuned to* — e.g. energy in its preferred frequency band (tonotopy) or fast harmonic movement. It is **encoding of the sound by sensory cortex, not a readout of experience**: the dataset has no ratings, so CAMP can say what the brain represents, not what the listener felt. The perceptual content is carried by *which features* drive a voxel (§4.3), not by the raw level.

**3D surface vs slices.** The surface is a standard brain (fsaverage6) with the listener's voxels projected onto it: good for orientation, blurs individual anatomy. Glow and spot spread are display aids. The **slice view is closer to the data**: the listener's own mean EPI or T1, native 2 mm voxels, each painted with its own prediction.

**Regions panel.** Predicted response averaged over each region's significant voxels at the playhead, with its Braak stage chip. Regions listed were chosen *a priori* for hypotheses (auditory hierarchy + memory/emotion), so several show nothing.

**Degeneration dial.** Removes vertices whose Braak stage ≤ the dial; "kept" = share of the song's total predicted response energy (Σ over time of ŷ²) in what remains. Stages I–III remove ~12% of cortex and ~0% of the music response; V removes most of it.

**Spread of the predicted response.** Effective number of voxels carrying the prediction, $2^H$ with $H = -\sum_k p_k \log_2 p_k$, $p_k = \hat y_k^2/\sum \hat y^2$. Low = one focused hotspot; high = evenly shared. Shannon entropy as a measure of how distributed the representation is at each moment.

**Rhythm on offer.** Spectrum of the sound's onset envelope (0.4–5 Hz, 8 s window) — the periodicities cortex can entrain to (frequency tagging, Nozaradan et al. 2011). It measures the *sound*, not a predicted EEG.

**Listener menu (001–005).** Five separate models. Same song, five brains. Differences between them are real individual differences (§4.2), not noise.

---

## 4. Results

### 4.1 Where CAMP works (all five listeners)

| | sub-001 | sub-002 | sub-003 | sub-004 | sub-005 |
|---|---|---|---|---|---|
| significant voxels, acoustic-only | 1,555 | 2,126 | 2,757 | 1,475 | 776 |
| significant voxels, structural-only | 876 | 409 | **0** | 995 | **1,866** |
| significant voxels, joint | 1,786 | 1,919 | 2,747 | 1,912 | 2,439 |
| structural share of joint R² (product measure) | 36% | 28% | 22% | 24% | 52% |
| acoustic weight peak (s) | 4.5 | 3.0 | 4.5 | 4.5 | 4.5 |
| Alzheimer's-order area (0.5 = random) | 0.518 | 0.506 | 0.509 | 0.523 | 0.481 |

- Mean held-out r in Heschl's / planum temporale ≈ 0.17–0.26; best voxel r = 0.50–0.67 per listener; in the 500 most reliable voxels, median r 0.27–0.44 against a median ceiling of 0.68–0.81. Across significant voxels the model reaches about **half to 60% of the ceiling** (median r/ceiling 0.48–0.60). Typical for 15 s clip fMRI.
- Only the five superior temporal regions are significant at region level. **Hippocampus, amygdala, medial frontal: essentially nothing** (0–6 significant voxels of ~500–1,200 each, every listener). Cingulate and precuneus show scattered voxels in two listeners (sub-002: 54 anterior cingulate, 22 precuneus; sub-005: 46 / 41 / 64 in anterior cingulate, posterior cingulate, precuneus) out of 1,300–2,500 voxels per region — too sparse to carry a region-level effect.

### 4.2 Finding 1 — sound vs. organisation, and big individual differences
Acoustic detail predicts auditory cortex in everyone. Musical structure adds explained variance in everyone too (22–52% of the joint R²), but how much it stands on its own varies enormously: in sub-003 no voxel is predicted by structure alone, yet structure still helps inside the joint model (it works only *alongside* acoustics); in sub-005 structure alone predicts more voxels than acoustics, and it reaches beyond auditory cortex into supramarginal, superior parietal, lateral occipital and precuneus. Same stimuli, same model, five different divisions of labour between "the sound" and "the music."

Honest caveat: per region, unique-structural bootstrap intervals often include zero with only 60 test clips, so the *gradient* across auditory regions is suggestive, not established. The split between "unique" and "shared" depends on the partition method (differencing vs product measure); the *total* structural contribution is the robust number.

### 4.3 Finding 2 (new today) — which musical features carry the weight
Method: the fitted joint model is linear, so its prediction splits exactly into one piece per feature group, $\hat y = \sum_g X_g W_g$. Each group's share of held-out R² comes from the product measure, $R^2_g = (2\,y\cdot\hat y_g - \hat y\cdot\hat y_g)/\lVert y\rVert^2$, and the shares sum to the joint R². Significance: 2,000 shuffles of which test clip the measured response belongs to. No refitting. (Shuffle-the-feature "permutation importance" was tried first and rejected: at held-out R² ≈ 0.07, shuffling any group that carries a large slice of prediction variance adds more error than the model ever explained, so the numbers exceed 100%.) Script: scratchpad `feature_split.py`; output `data/derivatives/importance/feature_split.csv`.

Share of each listener's held-out joint R² (all significant voxels; * = p < 0.05):

| feature group | 001 | 002 | 003 | 004 | 005 |
|---|---|---|---|---|---|
| spectrum (log-mel ×32) | 24.9* | 26.1* | 23.2* | 29.0* | 19.4* |
| timbre (MFCC ×13) | 9.9* | 22.2* | 23.6* | 9.4* | 8.8* |
| pitch class (chroma ×12) | 17.3* | 12.0* | 9.3* | 21.7* | 11.6* |
| brightness (centroid, bandwidth) | 10.7* | 11.6* | 20.9* | 14.3* | 6.8* |
| loudness (RMS) | 1.2* | 0.2 | 1.6 | 0.6* | 0.4* |
| **onset rate** | **18.0*** | 7.8* | 4.0* | 14.7* | 5.1* |
| **harmonic change** | 8.9* | **15.6*** | **15.2*** | −2.1 | **20.4*** |
| key clarity | 7.8* | 3.7* | −0.1 | 3.6* | 12.0* |
| pulse clarity | 1.6 | 3.0* | 4.4 | 6.7* | 10.1* |
| roughness | 3.1* | 2.1 | 4.6* | 3.7* | 6.1* |
| beat irregularity | 1.6* | 1.0* | 0.6 | 2.1* | 4.6* |
| **mode (major/minor)** | −1.3 | −0.2 | −0.3 | −1.1 | −0.6 |
| tempo | −3.5 | −5.0 | −7.0 | −2.6 | −4.6 |

(The structural rows sum to the band shares in §4.1, e.g. 36% for sub-001, which checks the decomposition.)

What it says:
1. **Within musical structure, two features do the work: how often events happen (onset rate) and how fast the harmony moves (harmonic change).** Together 13–27% of explained variance per listener (19–27% in four of five). Key clarity and pulse clarity add a few percent in most.
2. **Major vs minor contributes nothing in any listener, in any auditory region** (significant only outside auditory cortex in one listener). Mode is the feature most associated with musical emotion (happy/sad), and auditory cortex doesn't carry it measurably here. The most plausible reading is that mode's effect is evaluative and lives downstream (limbic/reward, insula, OFC), which this dataset can't see; a weaker reading is that a 4.5 s chroma-template estimate of mode is too noisy. Either way, "auditory cortex encodes the music's tonal *motion*, not its emotional *colour*."
3. **Tempo hurts held-out prediction in all five** (negative share, never significant). Beat-rate estimates on 15 s excerpts are unstable, so the weight the model gives tempo doesn't generalise. A methods note, and a reason to drop or smooth it.
4. **A gradient from Heschl's gyrus to anterior STG, same direction in 5 of 5 listeners:**

| share of R² | Heschl's (primary, Braak VI) | anterior STG (Braak V) | change, per listener |
|---|---|---|---|
| raw spectrum | 28.8% | 18.8% | falls in 5/5 (−1 to −21 pts) |
| harmonic change | 20.3% | −0.2% | falls in 5/5 (−10 to −28 pts) |
| pitch class (chroma) | 10.3% | 22.2% | **rises in 5/5 (+10 to +14 pts)** |
| brightness | 6.8% | 19.6% | rises in 5/5 (+4 to +24 pts) |

   Primary auditory cortex is driven by the raw spectrum and by moment-to-moment change; anterior STG is driven by *which pitch classes* are present and by spectral brightness, i.e. tonal content and timbre colour abstracted from raw spectral detail. That is the classic posterior-to-anterior "what" hierarchy (Patterson et al. 2002 on pitch; Norman-Haignere et al. 2015 on music-selective populations; Kell et al. 2018 on hierarchy in auditory cortex), recovered here by an interpretable model in each listener separately. Strength: a sign test across 5 listeners gives one-sided p = 1/32 ≈ 0.03 per gradient; between-region differences have not been tested with a clip bootstrap yet, so call it consistent, not established.
5. **Direction of the weights** (sum over delays = response to a sustained +1 SD change): harmonic change pushes BOLD **up** in 68–95% of significant voxels in every listener; pulse clarity pushes it **down** in 63–92% in every listener. Faster harmonic motion → more auditory cortex response; a clearer, more predictable beat → less. That is the signature predictive processing expects (surprise drives the response, predictability suppresses it), though here it is suggestive, not tested; the clean test is to add an explicit surprisal feature (IDyOM-style, Pearce 2005; Di Liberto et al. 2020).

**What it means for memory and degeneration (the part to say carefully).** Heschl's is Braak VI, the very last stage; anterior STG is V. Under the degeneration dial, the last representation standing is weighted toward raw spectral detail and harmonic motion, while the pitch-class/tonal-content representation in anterior STG goes one stage earlier. In the memory model's terms (melody/harmony vs rhythm vs timbre components), CAMP's empirical feature→region map already makes a non-circular prediction: *what's left of the music late in degeneration is its sound and its motion, more than its tonal identity.* That's a model prediction from the spatial order alone, with no compensation and no familiarity data. It is not a clinical claim.

**The neurodegeneration bridge in one sentence:** music's measurable cortical code sits at the end of the Braak sequence and smell's at the beginning, so one is a late-preserved channel and the other an early warning, and the same encoding-plus-ablation framework can quantify both.

### 4.4 Finding 3 — the Alzheimer's simulation is a clean null, and that is informative
Removing cortex in Braak order spares the music representation no better than removing the same regions in random order (area 0.48–0.52, p 0.42–0.56, all five; robust to the four ambiguous atlas calls). Through stage IV (25–27% of cortex removed), 82–92% of representation survives versus ~73–75% for random *voxels*, but that is just because the measurable representation sits in STG/Heschl's (stages V–VI); among *region* orders, Braak is unremarkable. What CAMP can say: **the part of music this dataset can measure lives in late-degenerating cortex.** What it can't say: anything about the memory/emotion regions that make a familiar song survive in patients (Jacobsen et al. 2015: caudal ACC, pre-SMA), because passive listening to unfamiliar clips leaves those regions silent.

### 4.5 Memory model (simulation, not data)
On branch `memory-model-1`: complementary-learning-systems model (hippocampal index + cortical consolidation, Teyler & DiScenna; McClelland et al. 1995), key–value attention retrieval (modern Hopfield), Braak-ordered degeneration, competing hypotheses *H-encoding* vs *H-disease*. The power run stopped at 18 of 100 simulated listeners (log last written 2026-10-26 01:12); interim false-positive rate for the interaction test is 2/18, too few to judge against the ≤ 5% bar. Present it as a designed, pre-registered simulation framework, not a result.

---

## 5. Why these brain regions

- **Heschl's gyrus** — primary auditory cortex (core, A1), tonotopic. Expect acoustic dominance. Braak VI.
- **Planum temporale** (posterior to HG) — complex spectrotemporal analysis, segregating sound sources; "computational hub" (Griffiths & Warren 2002). Braak V.
- **Planum polare** (anterior to HG) and **anterior STG** — ventral "what" stream; music-selective neural populations sit anterior and posterior to primary cortex (Norman-Haignere et al. 2015, *Neuron*); pitch and melody processing. Braak V.
- **Posterior STG** — belt/parabelt, spectrotemporal modulation, links to auditory–motor (dorsal) pathways for rhythm. Braak V.
- **The empty ones, chosen a priori:** hippocampus/MTL (episodic memory, Braak I–II), amygdala (emotion), caudal ACC and pre-SMA (preserved musical memory in AD, Jacobsen 2015), medial PFC (music-evoked autobiographical memory, Janata 2009), precuneus/PCC (episodic retrieval). They're in the panel *to show they are empty*: that gap is a property of the dataset (passive, unfamiliar music) and a pointer to the experiment that would fill it (familiar vs unfamiliar music).

---

## 6. Bridges to Neuraci (bring these up)

1. **Same problem, different sense.** Stimulus representation → human response. Their stimulus is a molecular graph (or a mixture); CAMP's is a waveform. Their readout is perceptual descriptors (and maybe neural/clinical measures); CAMP's is voxels. The methodology transfers directly: held-out stimuli, noise ceilings, permutation tests, feature-space partitioning.
2. **Interpretable vs learned features.** CAMP chose 68 named features so every weight is legible, and plans MERT (learned embedding) as a benchmark. A GNN embedding is the MERT side of that trade-off. Variance partitioning answers "what does the learned embedding capture beyond interpretable chemistry (functional groups, molecular weight, volatility)?" — the same question CAMP asks of acoustics vs structure.
3. **Noise ceilings.** Olfactory ratings are notoriously variable across and within people. A model can't beat the ceiling; reporting accuracy relative to it (as POM's median-panelist comparison does) is what makes claims trustworthy.
4. **Individual differences are signal.** CAMP's five listeners divide sound vs structure very differently. Smell perception varies even more (OR genotype polymorphisms, e.g. OR7D4 and androstenone). A per-person model, like CAMP's per-listener fits, is the natural unit for personalised consumer products and for detecting individual decline.
5. **The Braak link.** Olfactory structures are hit *first*: entorhinal/transentorhinal cortex is Braak I for tau in AD, and in Parkinson's, α-synuclein pathology begins in the olfactory bulb and anterior olfactory nucleus (Braak et al. 2003). Hyposmia precedes motor PD symptoms by years. CAMP's measurable music representation sits at stages V–VI. **The two senses bracket the disease**: smell is an early indicator, music is a late-preserved channel. An olfactory encoding model run through CAMP's ablation framework predicts early loss of representation — a testable contrast with a matched control.
6. **Prediction error as a common currency.** Familiarity as reduced prediction error (CAMP's memory-model readout) applies to smell recognition. A drift in a person's own perceptual predictions over time could be a sharper early marker than a single smell-test score, which is what their own blog flags as too non-specific on its own.

---

## 7. Questions to expect, and crisp answers

**"r = 0.2 seems low."** It's per 1.5 s time point on single voxels of noisy fMRI, held out. Compare to the noise ceiling: the best voxels reach r 0.5–0.67, and across significant voxels the model gets about half to 60% of the way to the ceiling. Every significant voxel passed a 5,000-permutation test with FDR correction.

**"How do you know it isn't overfitting?"** Hyper-parameters are chosen by cross-validation inside the training runs; the 60 test clips are scored once, at the end. The permutation test is on held-out data.

**"Is it one model or five?"** Five, one per listener, same features. There's no cross-subject model yet; a shared space (hyperalignment / surface) is future work.

**"What's novel?"** (a) An interpretable, browser-runnable whole-brain music encoding model with a live degeneration dial; (b) the finding that musical structure contributes 22–52% of explained variance in every listener but in very different ways; (c) the feature-level pattern in §4.3; (d) a well-controlled null: Braak order doesn't specially spare music representation once you control for region contiguity. The interface is the delivery mechanism, not the finding.

**"Why not a deep net / transformer?"** Data regime (4,800 samples/voxel), interpretability, and browser inference. MERT embeddings are the planned benchmark: if a learned representation predicts better, the partition shows what it captures beyond the 68 features.

**"How would you do this for smell?"** Odorant → representation (GNN embedding *and* interpretable descriptors as separate bands) → banded ridge to human ratings or neural data → held-out odorants, a ceiling from repeated presentations, a permutation null that shuffles odorant identity. Smell has no TR, so the "delays" become sniff-locked time windows (EEG/MEG) or block responses (fMRI, where piriform responses habituate fast).

**"Why GTZAN?"** It's what the dataset used. Limits: 15 s excerpts, Western genres, unfamiliar to listeners. Out-of-distribution uploads (speech, long songs, unusual sounds) are flagged in the UI.

**"Can it tell you what someone experiences?"** No. It predicts sensory-cortical encoding. Experience needs behavioural data (ratings, familiarity); this dataset has none.

---

## 8. Who's on the call: Swati Gupta, CSO

**What's public (LinkedIn wouldn't load; matched through search):** she joined Neuraci Labs as CSO after about seven years in Deanna Benson's lab (with George Huntley) at Mount Sinai's Friedman Brain Institute. Her work is on **LRRK2-G2019S**, the most common genetic risk mutation for Parkinson's: how it disrupts **AMPA-type glutamate receptor trafficking** at synapses in the **striatum**, specifically in D1 spiny projection neurons. First author, *PNAS* 2024: GluA1-containing AMPARs fail to increase at synapses after potentiation, a plasticity deficit. She is a cellular and molecular synapse neuroscientist working on Parkinson's, not an ML person.

**Bridges that speak to her work:**
- **BOLD is mostly synaptic.** It tracks synaptic input and local field potentials more than spiking (Logothetis 2001), much of it glutamatergic signalling coupled to blood flow through astrocytes. CAMP's signal sits on the machinery she studies.
- **Your ablation is the crude version.** CAMP deletes whole regions, which amounts to modelling cell death. Her work points to synaptic dysfunction well before cells die. The more realistic simulation weakens weights, or slows learning, instead of deleting voxels. Her plasticity deficit maps directly onto the consolidation rate in your memory model: a "synaptic" disease mode you could sweep against the Braak mode. **Ask her how she would model it.**
- **Rhythm, the striatum and PD.** Beat perception recruits the basal ganglia (Grahn & Brett 2007); people with PD have impaired beat-based timing; rhythmic auditory cueing improves PD gait (Thaut and colleagues). CAMP has a pulse-clarity feature. Music-evoked pleasure involves striatal dopamine (Salimpoor et al. 2011), yet CAMP finds nothing in nucleus accumbens: passive listening to unfamiliar clips again.
- **Smell meets the striatum.** The olfactory tubercle is part of the ventral striatum, gets direct olfactory-bulb input, and contains D1/D2 spiny projection neurons.
- **A homework question to ask her:** you've read that hyposmia is less common or milder in LRRK2-G2019S Parkinson's than in idiopathic PD (e.g. Saunders-Pullman et al. 2014). Is that her understanding, and does it matter for using smell as a marker across PD subtypes?

**How to pitch it:** translate the ML into her terms:
- Ridge regression = a linear fit with shrinkage, so it doesn't fit noise
- Noise ceiling = test-retest reliability across technical replicates
- Permutation test = shuffled-label control
- Held-out clips = an independent validation set
- Per-voxel model = one tuning curve per patch of cortex

Be precise and confident about CAMP, open about cellular biology, and ask real questions.

## Sources
- [Neuraci Labs](https://www.neuracilabs.com/) · [From Molecule to Meaning](https://www.neuracilabs.com/post/from-molecule-to-meaning) · [Neuraci (fragrance)](https://neuraci.com/) · [Neuraci Labs job listing](https://bebee.com/us/jobs/fractional-chief-marketing-officer-cmo-neuraci-labs-new-york--fj-2419118552)
- [Lee et al., A principal odor map unifies diverse tasks in olfactory perception, *Science*](https://www.science.org/doi/10.1126/science.ade4401) · [PMC version](https://pmc.ncbi.nlm.nih.gov/articles/PMC11898014/) · [GraphNOSE](https://arxiv.org/pdf/2609.05694)
