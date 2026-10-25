// CAMP model: features -> design rows -> predicted responses on the cortical
// surface, in named regions and in the participant's own voxels. Mirrors
// src/export_web.py (probe_design and the bundle layout it writes). Pure
// functions; runs in the browser, a worker or Node.

// float16 -> float32 (weights ship as half precision)
const HALF = (() => {
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    t[h] = e === 0 ? s * 2 ** -14 * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * 2 ** (e - 15) * (1 + m / 1024);
  }
  return t;
})();
export const fromHalf = (buf) => Float32Array.from(new Uint16Array(buf), (h) => HALF[h]);

// Bundle loader. `get(path)` returns a Promise<ArrayBuffer> for a published file.
export async function loadBundle(get, sub) {
  const p = (f) => `sub-${sub}/${f}`;
  const meta = JSON.parse(new TextDecoder().decode(await get(p("meta.json"))));
  const [wVertex, vertexIndex, vertexR, vertexRegion, vertexBraak, wRegion, wVoxel, voxelIndex, voxelR, labelVol] =
    await Promise.all([
      get(p("w_vertex.f16")).then(fromHalf),
      get(p("vertex_index.u32")).then((b) => new Uint32Array(b)),
      get(p("vertex_r.f32")).then((b) => new Float32Array(b)),
      get(p("vertex_region.u8")).then((b) => new Uint8Array(b)),
      get(p("vertex_braak.u8")).then((b) => new Uint8Array(b)),
      get(p("w_region.f32")).then((b) => new Float32Array(b)),
      get(p("w_voxel.f16")).then(fromHalf),
      get(p("voxel_index.u32")).then((b) => new Uint32Array(b)),
      get(p("voxel_r.f32")).then((b) => new Float32Array(b)),
      get(p("label_vol.u8")).then((b) => new Uint8Array(b)),
    ]);
  return {
    meta, wVertex, vertexIndex, vertexR, vertexRegion, vertexBraak, wRegion, wVoxel, voxelIndex, voxelR, labelVol,
    nRegionCols: wRegion.length / meta.n_cols,
  };
}

// One piece of audio, treated as one clip starting at t = 0: z-score with the
// training statistics, stack the delayed copies, remove the clip-onset nuisance
// with the training betas. Column order matches the fit: per band, delay-major.
export function designRows(feats, meta) {
  const nTr = feats.acoustic.length, delays = meta.delays, nD = delays.length;
  const X = Array.from({ length: nTr }, () => new Float64Array(meta.n_cols));
  let offset = 0;
  for (const band of ["acoustic", "structural"]) {
    const { mean, std, nuisance } = meta.bands[band], F = mean.length;
    const z = feats[band].map((row) => row.map((v, f) => (v - mean[f]) / std[f]));
    for (let t = 0; t < nTr; t++) {
      for (let i = 0; i < nD; i++) {
        const src = t - delays[i];
        for (let f = 0; f < F; f++) {
          const c = i * F + f;
          // nuisance row: delayed onset impulse (1 at row delays[j]) + intercept
          let nb = nuisance[nD][c];
          for (let j = 0; j < nD; j++) if (t === delays[j]) nb += nuisance[j][c];
          X[t][offset + c] = (src >= 0 ? z[src][f] : 0) - nb;
        }
      }
    }
    offset += F * nD;
  }
  return X;
}

// X (nTr x nCols) times W stored row-major (nCols x K) -> Float32Array (nTr * K)
export function project(X, W, K) {
  const nTr = X.length, nCols = X[0].length, out = new Float32Array(nTr * K);
  const acc = new Float64Array(K);
  for (let t = 0; t < nTr; t++) {
    acc.fill(0);
    const x = X[t];
    for (let c = 0; c < nCols; c++) {
      const xc = x[c];
      if (xc === 0) continue;
      const row = c * K;
      for (let k = 0; k < K; k++) acc[k] += xc * W[row + k];
    }
    out.set(acc, t * K);
  }
  return out;
}

export function predict(feats, bundle) {
  const X = designRows(feats, bundle.meta);
  return {
    nTr: X.length,
    vertex: project(X, bundle.wVertex, bundle.vertexIndex.length),
    region: project(X, bundle.wRegion, bundle.nRegionCols),
    voxel: project(X, bundle.wVoxel, bundle.voxelIndex.length),
  };
}
