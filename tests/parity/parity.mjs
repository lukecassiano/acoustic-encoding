// Parity: web/features.js against src/features.py, stage by stage.
//
//   python -m tests.parity.make_reference   # once, writes data/parity/
//   node tests/parity/parity.mjs [--verbose]
//
// Continuous stages: worst |js - py| relative to the stage's own scale.
// Discrete stages (tuning bin, onset frames, beat frames): exact agreement.
// Final features: worst error per feature in units of that feature's spread
// across all reference TRs -- the scale the encoding model sees after z-scoring.

import { readFileSync } from "node:fs";
import { extract, prepare } from "../../web/features.js";

const DIR = new URL("../../data/parity/", import.meta.url);
const TOL = 1e-6; // relative / z units
const verbose = process.argv.includes("--verbose");

const T = (m) => m[0].map((_, j) => m.map((r) => r[j])); // python (F, T) -> (T, F)
const flat = (x) => (Array.isArray(x) || ArrayBuffer.isView(x) ? Array.from(x).flatMap((v) => flat(v)) : [x]);

function relErr(js, py) {
  const a = flat(js), b = flat(py);
  if (a.length !== b.length) return { err: Infinity, note: `length ${a.length} vs ${b.length}` };
  let worst = 0, scale = 0;
  for (const v of b) scale = Math.max(scale, Math.abs(v));
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return { err: worst / (scale || 1) };
}

const setDiff = (a, b) => {
  const A = new Set(a), B = new Set(b);
  return { extra: a.filter((v) => !B.has(v)), missing: b.filter((v) => !A.has(v)) };
};

const manifest = JSON.parse(readFileSync(new URL("manifest.json", DIR)));
const finals = { acoustic: [], structural: [] }; // [js, py] TR rows, pooled over clips
const stageWorst = {};
let discreteFail = 0;

for (const { name } of manifest) {
  const buf = readFileSync(new URL(`${name}.f64`, DIR));
  const y = new Float64Array(buf.buffer, buf.byteOffset, buf.byteLength / 8);
  const py = JSON.parse(readFileSync(new URL(`${name}.json`, DIR)));
  const t0 = performance.now();
  const js = extract(y, { debug: true });
  const ms = performance.now() - t0;

  const stages = {
    logmel: [js.logmel, T(py.logmel)], mfcc: [js.mfcc, T(py.mfcc)],
    centroid: [js.centroid, py.centroid], bandwidth: [js.bandwidth, py.bandwidth],
    chroma: [js.chroma, T(py.chroma)], rms: [js.rms, py.rms],
    onset_env: [js.onsetEnv, py.onset_env], tempo: [js.tempo, py.tempo], pulse: [js.pulse, py.pulse],
    keys: [js.keys, py.keys], roughness: [js.roughness, py.roughness], tonnetz: [js.tonnetz, T(py.tonnetz)],
  };
  const line = [];
  for (const [k, [a, b]] of Object.entries(stages)) {
    const { err, note } = relErr(a, b);
    stageWorst[k] = Math.max(stageWorst[k] ?? 0, err);
    if (err > TOL || verbose) line.push(`${k} ${note ?? err.toExponential(1)}`);
  }
  const disc = [];
  if (js.tuning !== py.tuning) disc.push(`tuning ${js.tuning} vs ${py.tuning}`);
  if (Math.abs(js.bpm - py.bpm) > 1e-9) disc.push(`bpm ${js.bpm} vs ${py.bpm}`);
  for (const k of ["onsets", "beats"]) {
    const { extra, missing } = setDiff(js[k], py[k]);
    if (extra.length || missing.length) disc.push(`${k} +[${extra}] -[${missing}]`);
  }
  if (js.nTr !== py.n_tr || js.nFrames !== py.n_frames) disc.push(`shape n_tr ${js.nTr}/${py.n_tr} frames ${js.nFrames}/${py.n_frames}`);
  discreteFail += disc.length > 0;
  for (const b of ["acoustic", "structural"]) finals[b].push(...js[b].map((r, t) => [r, py[b][t]]));
  console.log(`${name.padEnd(22)} ${ms.toFixed(0).padStart(5)} ms  ${disc.length ? "DISCRETE " + disc.join("; ") : "discrete ok"}`
              + (line.length ? `\n${" ".repeat(24)}${line.join(", ")}` : ""));
}

console.log("\nworst relative error per stage:");
console.log("  " + Object.entries(stageWorst).map(([k, v]) => `${k} ${v.toExponential(1)}`).join(", "));

let featFail = 0;
for (const band of ["acoustic", "structural"]) {
  const rows = finals[band], nF = rows[0][1].length, bad = [];
  let worst = 0;
  for (let f = 0; f < nF; f++) {
    const col = rows.map(([, p]) => p[f]), m = col.reduce((a, b) => a + b, 0) / col.length;
    const sd = Math.sqrt(col.reduce((a, b) => a + (b - m) ** 2, 0) / col.length) || 1;
    const e = Math.max(...rows.map(([j, p]) => Math.abs(j[f] - p[f]))) / sd;
    worst = Math.max(worst, e);
    if (e > TOL) bad.push(`#${f} ${e.toExponential(1)}`);
  }
  featFail += bad.length;
  console.log(`${band.padEnd(10)} worst ${worst.toExponential(1)} SD over ${rows.length} TRs`
              + (bad.length ? `  -- above tolerance: ${bad.join(", ")}` : ""));
}

const f64 = (f) => { const b = readFileSync(new URL(f, DIR)); return new Float64Array(b.buffer, b.byteOffset, b.byteLength / 8); };
const prepErr = relErr(prepare(f64("prepare_raw.f64")), f64("prepare_out.f64")).err;
console.log(`prepare    worst ${prepErr.toExponential(1)} relative`);
if (prepErr > TOL) featFail += 1;

const ok = !featFail && !discreteFail;
console.log(ok ? "\nPARITY OK" : `\nPARITY FAIL (${featFail} features, ${discreteFail} clips with discrete mismatches)`);
process.exit(ok ? 0 : 1);
