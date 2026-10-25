// Parity: web/model.js prediction against src/export_web.py's reference (check.json).
//
//   python -m src.export_web --sub 001
//   node tests/parity/predict.mjs 001
//
// Feeds the Python features of one test clip through the browser's design and
// projection code and compares with the Python predictions, in units of the
// spread of the predictions themselves.

import { readFileSync } from "node:fs";
import { loadBundle, predict } from "../../web/model.js";

const sub = process.argv[2] ?? "001";
const BASE = new URL("../../data/web/", import.meta.url);
const get = async (p) => { const b = readFileSync(new URL(p, BASE)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };

const bundle = await loadBundle(get, sub);
const check = JSON.parse(readFileSync(new URL(`sub-${sub}/check.json`, BASE)));
const { vertex, region, voxel } = predict(check.features, bundle);

let ok = true;
for (const [name, js, py] of [["vertex", vertex, check.vertex_pred.flat()], ["region", region, check.region_pred.flat()], ["voxel", voxel, check.voxel_pred.flat()]]) {
  const sd = Math.sqrt(py.reduce((a, v) => a + v * v, 0) / py.length);
  let worst = 0;
  for (let i = 0; i < py.length; i++) worst = Math.max(worst, Math.abs(js[i] - py[i]));
  const rel = worst / sd;
  ok &&= js.length === py.length && rel < 1e-4;
  console.log(`${name.padEnd(7)} ${js.length} values, worst error ${rel.toExponential(1)} of the prediction RMS`);
}
console.log(ok ? "PREDICTION PARITY OK" : "PREDICTION PARITY FAIL");
process.exit(ok ? 0 : 1);
