// CAMP feature extractor: browser / Node port of src/features.py.
//
// Mirrors librosa 1.0.0 defaults as called by extract(): centred STFT with zero
// padding and a periodic Hann window, Slaney mel filterbank, power_to_db with
// top_db = 80, orthonormal DCT-II MFCCs, chroma_stft with piptrack tuning
// estimation, onset_strength on the log-mel, tempogram autocorrelation, greedy
// onset peak picking and the dynamic-programming beat tracker. Rounding follows
// Python (half to even) wherever librosa or numba round.
//
// Checked against the Python implementation by tests/parity/parity.mjs.
//
//   const y = prepare(mono22050);      // 2 s sine fade, RMS -> stimulus mean
//   const { acoustic, structural } = extract(y);   // arrays of TR rows

export const SR = 22050;
export const N_FFT = 2048;
export const HOP = 512;
export const TR = 1.5;
export const FADE_SEC = 2.0;
export const TARGET_RMS = 0.13210295550519335;
export const N_MELS = 32;
export const N_MFCC = 13;

const N_BINS = N_FFT / 2 + 1;
const TINY = 2.2250738585072014e-308; // np.finfo(float64).tiny
const KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

// ------------------------------------------------------------------ helpers

export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

function pyMod(a, b) { // numpy float remainder: fmod, shifted to the divisor's sign
  let m = a % b;
  if (m && (b < 0) !== (m < 0)) m += b;
  return m;
}

// CPython float floor division (differs from Math.floor(a / b) at exact multiples)
function floorDiv(a, b) {
  const mod = a % b;
  let div = (a - mod) / b;
  if (mod && (b < 0) !== (mod < 0)) div -= 1;
  let fl = Math.floor(div);
  if (div - fl > 0.5) fl += 1;
  return fl;
}

function linspace(a, b, n, endpoint = true) {
  const div = endpoint ? n - 1 : n, step = (b - a) / div, out = new Float64Array(n);
  for (let i = 0; i < n; i++) out[i] = i * step + a;
  if (endpoint && n > 1) out[n - 1] = b;
  return out;
}

function median(v) {
  const s = Float64Array.from(v).sort(), n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

const mean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const std = (v, ddof = 0) => {
  const m = mean(v);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - ddof));
};

function hann(n) { // scipy.signal.get_window('hann', n, fftbins=True)
  const w = new Float64Array(n);
  for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n);
  return w;
}

// In-place iterative radix-2 complex FFT of fixed size.
function makeFFT(n) {
  const bits = Math.log2(n), rev = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0;
    for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
    rev[i] = r;
  }
  const cos = new Float64Array(n / 2), sin = new Float64Array(n / 2);
  for (let i = 0; i < n / 2; i++) {
    cos[i] = Math.cos((2 * Math.PI * i) / n);
    sin[i] = -Math.sin((2 * Math.PI * i) / n);
  }
  return (re, im) => {
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, step = n / size;
      for (let s = 0; s < n; s += size) {
        for (let k = 0; k < half; k++) {
          const c = cos[k * step], d = sin[k * step], a = s + k, b = a + half;
          const tr = re[b] * c - im[b] * d, ti = re[b] * d + im[b] * c;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  };
}

// ----------------------------------------------------------------- audio in

export function prepare(input, { fade = true, normalize = true } = {}) {
  const y = Float64Array.from(input);
  if (fade) {
    const n = Math.min(Math.trunc(FADE_SEC * SR), Math.floor(y.length / 2));
    const ramp = linspace(0, Math.PI / 2, n).map(Math.sin);
    for (let i = 0; i < n; i++) {
      y[i] *= ramp[i];
      y[y.length - n + i] *= ramp[n - 1 - i];
    }
  }
  if (normalize) {
    let ss = 0;
    for (const v of y) ss += v * v;
    const rms = Math.sqrt(ss / y.length);
    if (rms > 0) for (let i = 0; i < y.length; i++) y[i] *= TARGET_RMS / rms;
  }
  return y;
}

// -------------------------------------------------------------- spectrogram

// |STFT|: frames of N_BINS magnitudes, centred with zero padding of N_FFT / 2.
export function stftMagnitude(y) {
  const pad = N_FFT / 2, n = y.length, nFrames = Math.floor(n / HOP) + 1;
  const win = hann(N_FFT), fft = makeFFT(N_FFT);
  const re = new Float64Array(N_FFT), im = new Float64Array(N_FFT), S = [];
  for (let t = 0; t < nFrames; t++) {
    const start = t * HOP - pad;
    for (let i = 0; i < N_FFT; i++) {
      const j = start + i;
      re[i] = j >= 0 && j < n ? y[j] * win[i] : 0;
      im[i] = 0;
    }
    fft(re, im);
    const mag = new Float64Array(N_BINS);
    for (let k = 0; k < N_BINS; k++) mag[k] = Math.hypot(re[k], im[k]);
    S.push(mag);
  }
  return S;
}

export function fftFrequencies() { // np.fft.rfftfreq(N_FFT, 1 / SR)
  const val = 1.0 / (N_FFT * (1.0 / SR)), f = new Float64Array(N_BINS);
  for (let k = 0; k < N_BINS; k++) f[k] = k * val;
  return f;
}

function hzToMel(f) {
  const fsp = 200 / 3;
  return f >= 1000 ? 1000 / fsp + Math.log(f / 1000) / (Math.log(6.4) / 27) : f / fsp;
}

function melToHz(m) {
  const fsp = 200 / 3, minLogMel = 1000 / fsp;
  return m >= minLogMel ? 1000 * Math.exp((Math.log(6.4) / 27) * (m - minLogMel)) : fsp * m;
}

// librosa.filters.mel(sr, n_fft, n_mels), Slaney scale and norm, stored float32
function melFilterbank(nMels = N_MELS) {
  const fftf = fftFrequencies();
  const melF = linspace(hzToMel(0), hzToMel(SR / 2), nMels + 2).map(melToHz);
  const W = [];
  for (let i = 0; i < nMels; i++) {
    const w = new Float64Array(N_BINS);
    const d0 = melF[i + 1] - melF[i], d1 = melF[i + 2] - melF[i + 1];
    const enorm = 2.0 / (melF[i + 2] - melF[i]);
    for (let k = 0; k < N_BINS; k++) {
      const lower = -(melF[i] - fftf[k]) / d0, upper = (melF[i + 2] - fftf[k]) / d1;
      w[k] = Math.fround(Math.fround(Math.max(0, Math.min(lower, upper))) * enorm);
    }
    W.push(w);
  }
  return W;
}

// librosa.filters.chroma(sr, n_fft, tuning), stored float32
function chromaFilterbank(tuning) {
  const nC = 12, A440 = 440.0 * 2.0 ** (tuning / nC);
  const freqs = linspace(0, SR, N_FFT, false);
  const frq = new Float64Array(N_FFT);
  for (let i = 1; i < N_FFT; i++) frq[i] = nC * Math.log2(freqs[i] / (A440 / 16));
  frq[0] = frq[1] - 1.5 * nC;
  const bw = new Float64Array(N_FFT);
  for (let i = 0; i < N_FFT - 1; i++) bw[i] = Math.max(frq[i + 1] - frq[i], 1.0);
  bw[N_FFT - 1] = 1;
  const half = roundHalfEven(nC / 2), wts = [];
  for (let c = 0; c < nC; c++) wts.push(new Float64Array(N_FFT));
  for (let i = 0; i < N_FFT; i++) {
    let norm = 0;
    for (let c = 0; c < nC; c++) {
      const x = frq[i] - c + half + 10 * nC;
      const D = pyMod(x, nC) - half;
      wts[c][i] = Math.exp(-0.5 * ((2 * D) / bw[i]) ** 2);
      norm += wts[c][i] ** 2;
    }
    norm = Math.sqrt(norm);
    if (norm < TINY) norm = 1;
    const oct = Math.exp(-0.5 * ((frq[i] / nC - 5.0) / 2) ** 2);
    for (let c = 0; c < nC; c++) wts[c][i] = (wts[c][i] / norm) * oct;
  }
  // base_c: roll by -3 so row 0 is C; keep the non-negative frequencies
  return Array.from({ length: nC }, (_, c) =>
    Float64Array.from(wts[(c + 3) % nC].subarray(0, N_BINS), Math.fround));
}

// librosa.estimate_tuning(S=P): piptrack peaks above median magnitude -> histogram
export function estimateTuning(P) {
  const fftf = fftFrequencies(), pitches = [], mags = [];
  for (const s of P) {
    let mx = -Infinity;
    for (const v of s) if (v > mx) mx = v;
    const ref = 0.1 * mx, m = (k) => (s[k] > ref ? s[k] : 0);
    for (let k = 1; k < N_BINS; k++) {
      if (!(fftf[k] >= 150 && fftf[k] < 4000)) continue;
      const isMax = k === N_BINS - 1 ? m(k) > m(k - 1) : m(k) > m(k - 1) && m(k) >= m(k + 1);
      if (!isMax) continue;
      let shift = 0, avg;
      if (k < N_BINS - 1) {
        const a = s[k + 1] + s[k - 1] - 2 * s[k], b = (s[k + 1] - s[k - 1]) / 2;
        shift = Math.abs(b) >= Math.abs(a) ? 0 : -b / a;
        avg = (s[k + 1] - s[k - 1]) / 2;
      } else avg = s[k] - s[k - 1];
      const pitch = ((k + shift) * SR) / N_FFT;
      if (pitch > 0) { pitches.push(pitch); mags.push(s[k] + 0.5 * avg * shift); }
    }
  }
  if (!pitches.length) return 0.0;
  const thr = median(mags), edges = linspace(-0.5, 0.5, 101), counts = new Int32Array(100);
  let any = false;
  for (let i = 0; i < pitches.length; i++) {
    if (mags[i] < thr) continue;
    any = true;
    let r = pyMod(12 * Math.log2(pitches[i] / 27.5), 1.0);
    if (r >= 0.5) r -= 1.0;
    // np.histogram with explicit edges: bin i holds edges[i] <= r < edges[i + 1]
    let lo = 0, hi = 100;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (edges[mid] <= r) lo = mid; else hi = mid; }
    counts[lo] += 1;
  }
  if (!any) return 0.0;
  let best = 0;
  for (let i = 1; i < 100; i++) if (counts[i] > counts[best]) best = i;
  return edges[best];
}

// ------------------------------------------------------------- frame features

function powerToDb(M) { // ref 1, amin 1e-10, top_db 80 over the whole matrix
  let mx = -Infinity;
  const out = M.map((row) => row.map((v) => {
    const d = 10.0 * Math.log10(Math.max(1e-10, v));
    if (d > mx) mx = d;
    return d;
  }));
  return out.map((row) => row.map((v) => Math.max(v, mx - 80.0)));
}

function dctOrtho(x, nOut) { // scipy.fft.dct type 2, norm='ortho'
  const N = x.length, out = new Float64Array(nOut);
  for (let k = 0; k < nOut; k++) {
    let s = 0;
    for (let n = 0; n < N; n++) s += x[n] * Math.cos((Math.PI * k * (2 * n + 1)) / (2 * N));
    out[k] = s * 2 * (k === 0 ? Math.sqrt(1 / (4 * N)) : Math.sqrt(1 / (2 * N)));
  }
  return out;
}

// onset_strength(S=logmel): mean positive first difference, lagged 1 + n_fft / (2 hop)
function onsetStrength(logmel) { // logmel: frames x mels
  const T = logmel.length, env = new Float64Array(T), padW = 1 + Math.floor(N_FFT / (2 * HOP));
  for (let m = 0; m + 1 < T && m + padW < T; m++) {
    let s = 0;
    for (let f = 0; f < N_MELS; f++) s += Math.max(0, logmel[m + 1][f] - logmel[m][f]);
    env[m + padW] = s / N_MELS;
  }
  return env;
}

// tempogram: windowed autocorrelation of the onset envelope, inf-normalized per frame
function tempogram(env, winLength) {
  const n = env.length, h = Math.floor(winLength / 2), w = hann(winLength);
  const padded = new Float64Array(n + 2 * h);
  const ramp = (edge, j) => j * (edge / h); // np.pad linear_ramp to 0 (linspace, no endpoint)
  for (let j = 0; j < h; j++) {
    padded[j] = ramp(env[0], j);
    padded[n + h + j] = ramp(env[n - 1], h - 1 - j);
  }
  padded.set(env, h);
  const tg = [], x = new Float64Array(winLength);
  for (let t = 0; t < n; t++) {
    for (let i = 0; i < winLength; i++) x[i] = padded[t + i] * w[i];
    const ac = new Float64Array(winLength);
    for (let lag = 0; lag < winLength; lag++) {
      let s = 0;
      for (let i = 0; i + lag < winLength; i++) s += x[i] * x[i + lag];
      ac[lag] = s;
    }
    let mx = 0;
    for (const v of ac) if (Math.abs(v) > mx) mx = Math.abs(v);
    if (mx < TINY) mx = 1;
    for (let i = 0; i < winLength; i++) ac[i] /= mx;
    tg.push(ac);
  }
  return tg;
}

function tempoBpms(winLength) {
  const b = new Float64Array(winLength);
  b[0] = Infinity;
  for (let k = 1; k < winLength; k++) b[k] = (60.0 * SR) / (HOP * k);
  return b;
}

// librosa.feature.tempo: argmax of log1p(1e6 tg) + log-normal prior around 120 BPM
function tempoFromTg(tgCol, bpms) {
  let best = -1, bestV = -Infinity;
  for (let k = 0; k < bpms.length; k++) {
    if (!(bpms[k] < 320.0)) continue; // logprior -inf up to the first bpm < max_tempo
    const v = Math.log1p(1000000.0 * tgCol[k]) - 0.5 * (Math.log2(bpms[k]) - Math.log2(120)) ** 2;
    if (v > bestV) { bestV = v; best = k; }
  }
  return bpms[best];
}

const AC_FRAMES = Math.floor((8.0 * SR) / HOP); // time_to_frames(ac_size = 8 s) = 344

function localTempo(env) {
  const bpms = tempoBpms(AC_FRAMES);
  return tempogram(env, AC_FRAMES).map((col) => tempoFromTg(col, bpms));
}

function globalTempo(env) {
  const tg = tempogram(env, AC_FRAMES), avg = new Float64Array(AC_FRAMES);
  for (const col of tg) for (let k = 0; k < AC_FRAMES; k++) avg[k] += col[k];
  for (let k = 0; k < AC_FRAMES; k++) avg[k] /= tg.length;
  return tempoFromTg(avg, tempoBpms(AC_FRAMES));
}

// onset_detect: normalize to [0, 1], greedy peak pick (librosa defaults at 22,050 / 512)
export function onsetDetect(env) {
  let mn = Infinity, mx = -Infinity;
  for (const v of env) mn = Math.min(mn, v);
  const x = env.map((v) => v - mn);
  for (const v of x) mx = Math.max(mx, v);
  for (let i = 0; i < x.length; i++) x[i] /= mx + TINY;
  if (!x.some((v) => v !== 0) || !x.every(Number.isFinite)) return [];
  const N = x.length, preMax = 1, postMax = 1, preAvg = 4, postAvg = 5, wait = 1, delta = 0.07;
  const win = (a, b) => x.subarray(Math.max(0, a), Math.min(b, N));
  const peaks = [];
  let p0 = x[0] >= Math.max(...win(0, postMax)) && x[0] >= mean(win(0, postAvg)) + delta;
  if (p0) peaks.push(0);
  let n = p0 ? wait + 1 : 1;
  while (n < N) {
    if (x[n] !== Math.max(...win(n - preMax, n + postMax)) || x[n] < mean(win(n - preAvg, n + postAvg)) + delta) {
      n += 1;
      continue;
    }
    peaks.push(n);
    n += wait + 1;
  }
  return peaks;
}

function localmax(x) { // x[i] > x[i-1] and x[i] >= x[i+1]; last element: x[-1] > x[-2]
  const N = x.length, out = new Uint8Array(N);
  for (let i = 1; i < N - 1; i++) out[i] = x[i] > x[i - 1] && x[i] >= x[i + 1];
  if (N > 1) out[N - 1] = x[N - 1] > x[N - 2];
  return out;
}

// librosa.beat.beat_track(onset_envelope): Ellis (2007) dynamic programming
export function beatTrack(env) {
  if (!env.some((v) => v !== 0)) return { bpm: 0, beats: [] };
  const bpm = globalTempo(env), N = env.length, tightness = 100;
  const fpb = roundHalfEven(((SR / HOP) * 60.0) / bpm);
  const sd = std(env, 1) + TINY, oe = env.map((v) => v / sd);

  const K = 2 * fpb + 1, h = Math.floor(K / 2), window = new Float64Array(K);
  for (let k = 0; k < K; k++) window[k] = Math.exp(-0.5 * (((k - fpb) * 32.0) / fpb) ** 2);
  const local = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = Math.max(0, i + h - N + 1); k < Math.min(i + h, K); k++) s += window[k] * oe[i + h - k];
    local[i] = s;
  }

  let lmax = -Infinity;
  for (const v of local) lmax = Math.max(lmax, v);
  const thresh = 0.01 * lmax, back = new Int32Array(N), cum = new Float64Array(N);
  let first = true;
  for (let i = 0; i < N; i++) {
    let best = -Infinity, loc = -1;
    const stop = Math.trunc(i - 2 * fpb - 1);
    for (let l = i - roundHalfEven(fpb / 2); l > stop; l--) {
      if (l < 0) break;
      const sc = cum[l] - tightness * (Math.log(i - l) - Math.log(fpb)) ** 2;
      if (sc > best) { best = sc; loc = l; }
    }
    cum[i] = loc >= 0 ? local[i] + best : local[i];
    if (first && local[i] < thresh) back[i] = -1;
    else { back[i] = loc; first = false; }
  }

  const lm = localmax(cum), peaksAt = [];
  for (let i = 0; i < N; i++) if (lm[i]) peaksAt.push(cum[i]);
  const thr = 0.5 * median(peaksAt);
  let tail = N - 1;
  for (let n = N - 1; n >= 0; n--) if (lm[n] && cum[n] >= thr) { tail = n; break; }
  const isBeat = new Uint8Array(N);
  for (let n = tail; n >= 0; n = back[n]) isBeat[n] = 1;

  // trim weak leading / trailing beats (threshold from Hann-smoothed beat scores)
  const vals = [];
  for (let i = 0; i < N; i++) if (isBeat[i]) vals.push(local[i]);
  const w = [0, 0.5, 1, 0.5, 0], full = new Float64Array(vals.length + 4);
  for (let i = 0; i < vals.length; i++) for (let k = 0; k < 5; k++) full[i + k] += vals[i] * w[k];
  const smooth = full.subarray(2, Math.min(N + 2, full.length));
  const trimThr = 0.5 * Math.sqrt(mean(smooth.map((v) => v * v)));
  for (let n = 0; n < N && local[n] <= trimThr; n++) isBeat[n] = 0;
  for (let n = N - 1; n >= 0 && local[n] <= trimThr; n--) isBeat[n] = 0;
  const beats = [];
  for (let i = 0; i < N; i++) if (isBeat[i]) beats.push(i);
  return { bpm, beats };
}

function keyProfiles() {
  const z = (p) => { const m = mean(p), s = std(p); return p.map((v) => (v - m) / s); };
  const roll = (p, k) => p.map((_, i) => p[(i - k + 12) % 12]);
  return [KK_MAJOR, KK_MINOR].map((p) => Array.from({ length: 12 }, (_, k) => z(roll(p, k))));
}

// Krumhansl-Schmuckler key finding per TR on a 3-TR window: [key clarity, mode]
export function keyFeatures(chroma, nTr) {
  const pooled = framesToTrs(chroma, nTr), [major, minor] = keyProfiles();
  return pooled.map((_, t) => {
    const c = new Float64Array(12);
    for (let u = Math.max(0, t - 1); u < Math.min(nTr, t + 2); u++) for (let i = 0; i < 12; i++) c[i] += pooled[u][i];
    const s = std(c);
    if (s === 0) return [0, 0];
    const m = mean(c), z = c.map((v) => (v - m) / s);
    const best = (profiles) => Math.max(...profiles.map((p) => p.reduce((a, v, i) => a + v * z[i], 0) / 12));
    const rmaj = best(major), rmin = best(minor);
    return [Math.max(rmaj, rmin), rmaj - rmin];
  });
}

function findPeaks(x) { // scipy.signal.find_peaks without conditions (plateau midpoints)
  const peaks = [], iMax = x.length - 1;
  let i = 1;
  while (i < iMax) {
    if (x[i - 1] < x[i]) {
      let ahead = i + 1;
      while (ahead < iMax && x[ahead] === x[i]) ahead++;
      if (x[ahead] < x[i]) {
        peaks.push(Math.floor((i + ahead - 1) / 2));
        i = ahead;
      }
    }
    i++;
  }
  return peaks;
}

// Sethares (1993) Plomp-Levelt dissonance over the 20 strongest peaks, per frame
export function roughness(S, freqs, nPeaks = 20) {
  return S.map((mag) => {
    let pk = findPeaks(mag);
    if (pk.length < 2) return 0;
    pk = pk.sort((a, b) => mag[a] - mag[b]).slice(-nPeaks);
    let d = 0, e = 0;
    for (let i = 0; i < pk.length; i++) {
      const fi = freqs[pk[i]], ai = mag[pk[i]];
      e += ai * ai;
      for (let j = i + 1; j < pk.length; j++) {
        const fj = freqs[pk[j]], s = 0.24 / (0.021 * Math.min(fi, fj) + 19), df = Math.abs(fi - fj);
        d += ai * mag[pk[j]] * (Math.exp(-3.5 * s * df) - Math.exp(-5.75 * s * df));
      }
    }
    return d / e;
  });
}

// librosa.feature.tonnetz(chroma): six tonal-centroid coordinates per frame
function tonnetz(chroma) {
  const scale = [7 / 6, 7 / 6, 3 / 2, 3 / 2, 2 / 3, 2 / 3], R = [1, 1, 1, 1, 0.5, 0.5];
  const phi = scale.map((sc, p) => Array.from({ length: 12 }, (_, c) =>
    R[p] * Math.cos(Math.PI * (sc * c - (p % 2 === 0 ? 0.5 : 0)))));
  return chroma.map((col) => {
    let s = 0;
    for (const v of col) s += Math.abs(v);
    if (s < TINY) s = 1;
    return phi.map((row) => row.reduce((a, w, c) => a + w * (col[c] / s), 0));
  });
}

export function framesToTrs(x, nTr) { // mean-pool frame rows into TR bins
  const nF = x[0].length, out = Array.from({ length: nTr }, () => new Float64Array(nF));
  const count = new Int32Array(nTr);
  for (let t = 0; t < x.length; t++) {
    const idx = Math.min(floorDiv((t * HOP) / SR, TR), nTr - 1);
    count[idx]++;
    for (let f = 0; f < nF; f++) out[idx][f] += x[t][f];
  }
  for (let t = 0; t < nTr; t++) if (count[t]) for (let f = 0; f < nF; f++) out[t][f] /= count[t];
  return out;
}

// ------------------------------------------------------------------ extract

// Audio (22,050 Hz mono, prepared) -> { acoustic: nTr x 60, structural: nTr x 8 }.
// With { debug: true } also returns the intermediates the parity test checks.
export function extract(y, { debug = false } = {}) {
  const nTr = roundHalfEven(y.length / SR / TR);
  const S = stftMagnitude(y), P = S.map((s) => s.map((v) => v * v)), T = S.length;
  const freqs = fftFrequencies();

  // acoustic
  const melFb = melFilterbank();
  const logmel = powerToDb(P.map((p) => melFb.map((w) => w.reduce((a, wk, k) => a + wk * p[k], 0))));
  const mfcc = logmel.map((col) => dctOrtho(col, N_MFCC + 1).subarray(1));
  const centroid = new Float64Array(T), bandwidth = new Float64Array(T), rms = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    const s = S[t];
    let sum = 0, c = 0, bw = 0, pw = 0;
    for (let k = 0; k < N_BINS; k++) sum += s[k];
    if (sum < TINY) sum = 1;
    for (let k = 0; k < N_BINS; k++) c += freqs[k] * (s[k] / sum);
    for (let k = 0; k < N_BINS; k++) bw += (s[k] / sum) * Math.abs(c - freqs[k]) ** 2;
    for (let k = 0; k < N_BINS; k++) {
      const x = Math.fround(Math.fround(s[k]) ** 2); // rms(S=) squares in float32
      pw += k === 0 || k === N_BINS - 1 ? 0.5 * x : x;
    }
    centroid[t] = c; bandwidth[t] = Math.sqrt(bw); rms[t] = Math.sqrt((2 * pw) / N_FFT ** 2);
  }
  const tuning = estimateTuning(P), chromaFb = chromaFilterbank(tuning);
  const chroma = P.map((p) => {
    const raw = chromaFb.map((w) => w.reduce((a, wk, k) => a + wk * p[k], 0));
    let mx = 0;
    for (const v of raw) mx = Math.max(mx, Math.abs(v));
    if (mx < TINY) mx = 1;
    return raw.map((v) => v / mx);
  });
  const acousticFrames = logmel.map((lm, t) => [
    ...lm, ...mfcc[t], Math.log(centroid[t] + 1), Math.log(bandwidth[t] + 1), ...chroma[t], Math.log(rms[t] + 1e-8),
  ]);

  // structural: rhythm
  const env = onsetStrength(logmel);
  const tempo = localTempo(env);
  const tg = tempogram(env, 384), lagSec = (k) => (k * HOP) / SR;
  const pulse = tg.map((col) => {
    let mx = -Infinity;
    for (let k = 0; k < col.length; k++) if (lagSec(k) >= 0.25 && lagSec(k) <= 2.0) mx = Math.max(mx, col[k]);
    return mx / Math.max(col[0], 1e-8);
  });
  const onsets = onsetDetect(env), onsetTrain = new Float64Array(T);
  for (const o of onsets) onsetTrain[o] = SR / HOP;
  const { bpm, beats } = beatTrack(env);
  const ibi = beats.slice(1).map((b, i) => ((b - beats[i]) * HOP) / SR);
  const beatCv = ibi.length > 2 ? std(ibi) / mean(ibi) : 0.0;

  // structural: harmony
  const keys = keyFeatures(chroma, nTr);
  const rough = roughness(S, freqs);
  const ton = tonnetz(chroma);
  const smooth = ton.map((_, t) => ton[0].map((__, p) => {
    let s = 0;
    for (let u = t - 4; u <= t + 4; u++) if (u >= 0 && u < T) s += ton[u][p] * (1 / 9);
    return s;
  }));
  const hcdf = smooth.map((v, t) => (t === 0 ? 0 : Math.hypot(...v.map((x, p) => x - smooth[t - 1][p]))));

  const fs = framesToTrs(tempo.map((_, t) => [Math.log2(tempo[t]), pulse[t], onsetTrain[t], rough[t], hcdf[t]]), nTr);
  const structural = fs.map((r, t) => [r[0], r[1], r[2], beatCv, keys[t][0], keys[t][1], Math.log(r[3] + 1e-6), r[4]]);
  // rhythm: the frame-rate onset envelope and global tempo, for the page's entrainment spectrum
  const out = { acoustic: framesToTrs(acousticFrames, nTr).map((r) => Array.from(r)), structural,
                rhythm: { onsetEnv: env, bpm, frameRate: SR / HOP } };
  if (debug) {
    Object.assign(out, { nFrames: T, nTr, logmel, mfcc, centroid, bandwidth, tuning, chroma, rms,
                         onsetEnv: env, tempo, pulse, onsets, bpm, beats, keys, roughness: rough, tonnetz: ton });
  }
  return out;
}
