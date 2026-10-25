"""Fitted joint model -> browser bundle for the CAMP interface (web/).

The model is linear: the predicted response at TR t is x_t W, with x_t the
delayed, training-standardized, nuisance-cleaned features. Projecting voxels
onto the cortical surface is linear too, so it is folded into the weights:
vertex predictions are x_t (W M^T), where M averages the model voxels under
each fsaverage6 vertex. Region bars work the same way, and the slice viewer
uses the voxel weights themselves.

Only voxels the joint model predicts significantly (FDR q < 0.05, stats.npz)
carry weights; every other vertex and voxel is drawn unlit.

Voxel -> MNI: three MNI-grid images holding their own RAS coordinates are
pulled into EPI space through the mni_to_epi chain (the one the atlases used,
checked against the subject's T1), so each voxel reads off its MNI position.
Interpolating a linear coordinate field is exact.

Probes: new audio is treated as one clip starting at t = 0, so its nuisance
regressors are the onset impulse at each delay plus the intercept.

Weights are stored as float16 (the parity reference uses the same rounded
weights). Hosting serves text, not binary, so every array is also written as
gzip + base64 (.gz.b64.txt); the page inflates it, Node tests read the raw file.

Writes data/web/sub-XXX/
  meta.json          feature stats + nuisance betas, delays, regions, volume layout
  w_vertex.f16       (n_cols, K) weights onto the K lit vertices
  vertex_index.u32   (K,) surface vertex ids, left hemisphere first
  vertex_r.f32       (K,) held-out joint r of the voxels under each lit vertex
  vertex_region.u8   per surface vertex: index into meta.regions, 255 = none
  vertex_braak.u8    per surface vertex: Braak stage 1-6, 7 unstaged, 0 no voxel
  w_region.f32       (n_cols, R) weights onto the regions with enough lit voxels
  w_voxel.f16        (n_cols, V) weights of the V lit voxels
  voxel_index.u32    (V,) flat index of each lit voxel in the RAS-ordered volume
  voxel_r.f32        (V,) held-out joint r
  label_vol.u8       atlas label of every voxel in the RAS volume: index into meta.labels, 255 outside the mask
  epi_mean.u8        mean functional image (training run 1, N4), RAS order, 0-255
  epi_t1.u8          the participant's T1 resampled to the same grid
  check.json         one test clip's features and predictions, for the JS parity test
and data/web/mesh/ (shared): pial / inflated coordinates, faces, sulcal depth, curvature.

Usage:
  python -m src.export_web --sub 001
"""

import argparse
import base64
import gzip
import json
import os

import numpy as np
import pandas as pd

from . import encoding as E
from . import features as F
from .ablation import BRAAK
from .validate import paths, region_labels

OUT = F.DATA / "web"
MESH = "fsaverage6"
SAMPLE_DEPTHS = np.linspace(0, 1, 5)  # white -> pial
MAX_MM = 3.0  # a depth sample takes its nearest voxel only if it is this close
MIN_REGION_VOXELS = 5

REGIONS = [  # (display name, Harvard-Oxford label after hemisphere merge)
    ("Heschl's gyrus", "Heschl's Gyrus (includes H1 and H2)"),
    ("Planum temporale", "Planum Temporale"),
    ("Planum polare", "Planum Polare"),
    ("Superior temporal, anterior", "Superior Temporal Gyrus, anterior division"),
    ("Superior temporal, posterior", "Superior Temporal Gyrus, posterior division"),
    ("Middle temporal, posterior", "Middle Temporal Gyrus, posterior division"),
    ("Insula", "Insular Cortex"),
    ("Supplementary motor", "Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)"),
    ("Anterior cingulate", "Cingulate Gyrus, anterior division"),
    ("Posterior cingulate", "Cingulate Gyrus, posterior division"),
    ("Precuneus", "Precuneous Cortex"),
    ("Medial frontal", "Frontal Medial Cortex"),
    ("Hippocampus", "Hippocampus"),
    ("Amygdala", "Amygdala"),
]
BRAAK_STAGE = {label: stage for stage, labels in BRAAK.items() for label in labels}


# ------------------------------------------------------------------ geometry

def voxel_mni(sub, ijk):
    """MNI (RAS mm) position of each EPI voxel, via the mni_to_epi chain."""
    os.environ.setdefault("TEMPLATEFLOW_HOME", str(F.DATA / "templateflow"))
    import ants
    import nibabel as nib
    from templateflow import api as tf
    d, _ = paths(sub)
    chain = json.loads((d / "xfm" / "chains.json").read_text())["mni_to_epi"]
    epi_ref = ants.image_read(str(d / "epi_ref.nii.gz"))
    tpl = nib.load(tf.get("MNI152NLin6Asym", resolution=1, suffix="T1w", desc=None))
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in tpl.shape[:3]], indexing="ij"), -1)
    ras = grid @ tpl.affine[:3, :3].T + tpl.affine[:3, 3]
    out = []
    for axis in range(3):
        img = ants.from_nibabel_nifti(nib.Nifti1Image(ras[..., axis].astype(np.float32), tpl.affine))
        warped = ants.apply_transforms(epi_ref, img, chain, interpolator="linear").numpy()
        out.append(warped[tuple(ijk.T)])
    return np.column_stack(out)


def load_mesh():
    from nilearn import datasets, surface
    fs = datasets.fetch_surf_fsaverage(MESH, data_dir=str(F.DATA / "nilearn"))
    mesh = {}
    for hemi in ("left", "right"):
        for kind in ("pial", "white", "infl"):
            coords, faces = surface.load_surf_mesh(fs[f"{kind}_{hemi}"])
            mesh[f"{kind}_{hemi}"] = (np.asarray(coords, np.float32), np.asarray(faces, np.uint32))
        for kind in ("sulc", "curv"):
            mesh[f"{kind}_{hemi}"] = np.asarray(surface.load_surf_data(fs[f"{kind}_{hemi}"]), np.float32)
    return mesh


def depth_samples(mesh):
    for hemi in ("left", "right"):
        white, pial = mesh[f"white_{hemi}"][0], mesh[f"pial_{hemi}"][0]
        yield hemi, white, pial


def vertex_map(mesh, vox_mni, lit):
    """Sparse (vertices x voxels) averaging matrix: each vertex takes the nearest
    mask voxel at five depths through the ribbon and keeps those that are lit."""
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree
    tree = cKDTree(vox_mni)
    rows, cols, offset = [], [], 0
    for _, white, pial in depth_samples(mesh):
        for f in SAMPLE_DEPTHS:
            dist, nearest = tree.query(white + f * (pial - white), distance_upper_bound=MAX_MM)
            ok = np.isfinite(dist)
            ok[ok] = lit[nearest[ok]]
            rows.append(np.flatnonzero(ok) + offset)
            cols.append(nearest[ok])
        offset += len(white)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    M = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(offset, len(vox_mni))).tocsr()
    M.sum_duplicates()
    counts = np.asarray(M.sum(1)).ravel()
    keep = np.flatnonzero(counts)
    return M[keep].multiply(1 / counts[keep][:, None]).tocsr(), keep


def vertex_labels(mesh, vox_mni, labels):
    """Atlas label under every vertex: the nearest mask voxel at mid-ribbon,
    falling back to the other depths; '' where no voxel is within MAX_MM."""
    from scipy.spatial import cKDTree
    tree, out = cKDTree(vox_mni), []
    for _, white, pial in depth_samples(mesh):
        lab = np.full(len(white), "", dtype=object)
        for f in (0.5, 0.25, 0.75, 0.0, 1.0):
            dist, nearest = tree.query(white + f * (pial - white), distance_upper_bound=MAX_MM)
            fill = (lab == "") & np.isfinite(dist)
            lab[fill] = labels[nearest[fill]]
        out.append(lab)
    return np.concatenate(out)


def ras_volume(sub):
    """EPI grid reordered to RAS axes: (index map voxel ijk -> flat RAS index,
    RAS shape, mean EPI and T1-in-EPI as 0-255 arrays)."""
    import nibabel as nib
    from nibabel.orientations import io_orientation
    d, _ = paths(sub)
    ref = nib.load(d / "epi_ref.nii.gz")
    ornt = io_orientation(ref.affine)  # per input axis: (output axis, flip)
    shape_in = np.array(ref.shape[:3])
    shape = tuple(int(shape_in[np.flatnonzero(ornt[:, 0] == a)[0]]) for a in range(3))

    def to_ras(ijk):
        out = np.zeros_like(ijk)
        for ax in range(3):
            a, flip = int(ornt[ax, 0]), ornt[ax, 1] < 0
            out[:, a] = shape_in[ax] - 1 - ijk[:, ax] if flip else ijk[:, ax]
        return np.ravel_multi_index(tuple(out.T), shape)

    grid = np.indices(ref.shape[:3]).reshape(3, -1).T
    flat = to_ras(grid)

    def u8(path, mask):
        x = np.asarray(nib.load(path).dataobj, np.float32).reshape(-1)
        lo, hi = np.percentile(x[mask], [1, 99.5])
        out = np.zeros(int(np.prod(shape)), np.uint8)
        out[flat] = np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        return out

    brain = np.asarray(nib.load(d / "epi_brainmask.nii.gz").dataobj).reshape(-1) > 0
    return to_ras, shape, u8(d / "epi_ref.nii.gz", brain), u8(d / "epi_t1.nii.gz", brain)


def probe_design(feats, stats):
    """Features of one piece of audio -> design rows, exactly as training rows were built."""
    N = np.zeros((len(feats["acoustic"]), 1), np.float32)
    N[0] = 1
    N = np.column_stack([E.delayed(N), np.ones(len(N), np.float32)])
    cols = []
    for band in F.BANDS:
        mu, sd, beta = stats[band]
        x = E.delayed(((feats[band] - mu) / sd).astype(np.float32))
        cols.append(x - N @ beta)
    return np.hstack(cols)


# -------------------------------------------------------------------- output

def save(d, name, arr):
    """Raw little-endian array plus its gzip + base64 text twin."""
    raw = np.ascontiguousarray(arr).tobytes()
    (d / name).write_bytes(raw)
    (d / f"{name}.gz.b64.txt").write_text(base64.b64encode(gzip.compress(raw, 9)).decode("ascii"))


def export_mesh(mesh):
    d = OUT / "mesh"
    d.mkdir(parents=True, exist_ok=True)
    for kind in ("pial", "infl"):
        save(d, f"{kind}.f32", np.concatenate([mesh[f"{kind}_left"][0], mesh[f"{kind}_right"][0]]).astype("<f4"))
    n_left = len(mesh["pial_left"][0])
    save(d, "faces.u32", np.concatenate([mesh["pial_left"][1], mesh["pial_right"][1] + n_left]).astype("<u4"))
    for kind in ("sulc", "curv"):
        save(d, f"{kind}.f32", np.concatenate([mesh[f"{kind}_left"], mesh[f"{kind}_right"]]).astype("<f4"))
    (d / "meta.json").write_text(json.dumps({"name": MESH, "n_vertices": 2 * n_left, "n_left": n_left,
                                             "n_faces": 2 * len(mesh["pial_left"][1])}))


def run(sub):
    d, out = paths(sub)
    res, stats_sig = np.load(out / "stage2.npz"), np.load(out / "stats.npz")
    lit = stats_sig["sig_joint"]
    vox = pd.read_csv(d / "voxels.csv")
    vox["region"] = region_labels(vox)
    _, X_test, _, stats = E.build_design(sub, dict(np.load(F.CACHE / "clips.npz")))
    for b in F.BANDS:  # the bundle's stats must be the fit's stats
        assert np.allclose(stats[b][0], res[f"feat_mean_{b}"]) and np.allclose(stats[b][1], res[f"feat_std_{b}"])

    W = np.load(out / "weights_joint.npy", mmap_mode="r")
    ijk = vox[["i", "j", "k"]].values
    vox_mni = voxel_mni(sub, ijk)
    hg = vox.region.values == REGIONS[0][1]
    print(f"Heschl's gyrus voxels at MNI x {np.median(np.abs(vox_mni[hg, 0])):.0f} (|x|), "
          f"y {np.median(vox_mni[hg, 1]):.0f}, z {np.median(vox_mni[hg, 2]):.0f} "
          f"(atlas centre about 45, -20, 8)", flush=True)

    mesh = load_mesh()
    export_mesh(mesh)
    M, verts = vertex_map(mesh, vox_mni, lit)
    half = lambda x: np.asarray(x, np.float16)
    W_vert = half((M @ np.asarray(W).T).T)  # (n_cols, K)
    vertex_r = (M @ np.nan_to_num(res["r_joint"])).astype(np.float32)

    lit_idx = np.flatnonzero(lit)
    W_vox = half(W[:, lit_idx])
    regions, region_cols = [], []
    for name, label in REGIONS:
        member = (vox.region.values == label)
        n_lit = int((member & lit).sum())
        entry = {"name": name, "label": label, "braak": BRAAK_STAGE.get(label, 6),
                 "n_voxels": int(member.sum()), "n_lit": n_lit, "column": None}
        if n_lit >= MIN_REGION_VOXELS:
            entry["column"] = len(region_cols)
            region_cols.append(np.asarray(W[:, np.flatnonzero(member & lit)]).mean(1))
        regions.append(entry)
    W_region = np.column_stack(region_cols).astype(np.float32)

    # per vertex: which listed region it sits in (255 = none) and its Braak stage --
    # 1-6, 7 for labels the ordering does not name (removed last, as in ablation.stage_of),
    # 0 where no voxel lies under the vertex
    vlab = vertex_labels(mesh, vox_mni, vox.region.values)
    code = {label: i for i, (_, label) in enumerate(REGIONS)}
    vertex_region = np.array([code.get(l, 255) for l in vlab], np.uint8)
    vertex_braak = np.array([0 if l == "" else BRAAK_STAGE.get(l, 7) for l in vlab], np.uint8)

    to_ras, shape, epi_mean, epi_t1 = ras_volume(sub)
    labels = sorted(set(vox.region.values))
    label_vol = np.full(int(np.prod(shape)), 255, np.uint8)
    label_vol[to_ras(ijk)] = [labels.index(l) for l in vox.region.values]

    sd = OUT / f"sub-{sub}"
    sd.mkdir(parents=True, exist_ok=True)
    for f in sd.glob("*"):
        f.unlink()
    save(sd, "w_vertex.f16", W_vert.astype("<f2"))
    save(sd, "vertex_index.u32", verts.astype("<u4"))
    save(sd, "vertex_r.f32", vertex_r.astype("<f4"))
    save(sd, "vertex_region.u8", vertex_region)
    save(sd, "vertex_braak.u8", vertex_braak)
    save(sd, "w_region.f32", W_region.astype("<f4"))
    save(sd, "w_voxel.f16", W_vox.astype("<f2"))
    save(sd, "voxel_index.u32", to_ras(ijk[lit_idx]).astype("<u4"))
    save(sd, "voxel_r.f32", np.nan_to_num(res["r_joint"][lit_idx]).astype("<f4"))
    save(sd, "label_vol.u8", label_vol)
    save(sd, "epi_mean.u8", epi_mean)
    save(sd, "epi_t1.u8", epi_t1)

    delays = pd.read_csv(out / "delays.csv")
    meta = {
        "sub": sub, "tr": F.TR, "delays": list(E.DELAYS), "n_cols": int(W.shape[0]),
        "peak_delay_s": float(delays.delay_s[delays.share.idxmax()]),
        "median_lit_r": float(np.median(res["r_joint"][lit])), "max_lit_r": float(np.max(res["r_joint"][lit])),
        "n_lit_vertices": int(len(verts)), "n_lit_voxels": int(lit.sum()), "mesh": MESH,
        "bands": {b: {"names": F.FEATURE_NAMES[b], "mean": stats[b][0].tolist(), "std": stats[b][1].tolist(),
                      "nuisance": stats[b][2].tolist()} for b in F.BANDS},
        "regions": regions, "labels": labels,
        "label_braak": [0 if l == "Background" else BRAAK_STAGE.get(l, 7) for l in labels],
        "volume": {"shape": list(shape), "voxel_mm": 2.0, "order": "RAS, C-order flat index (i * nj + j) * nk + k"},
        "units": "BOLD z-score (within-run), clip-boundary response removed",
        "lit": "joint-model voxels significant at FDR q < 0.05 (clip-permutation test)",
    }
    (sd / "meta.json").write_text(json.dumps(meta))

    # parity reference for the browser's prediction step: a real test clip, float16 weights as shipped
    ev = F.run_events(sub, "Test", 1).iloc[1]
    feats = F.extract(F.clip_audio(ev.genre, ev.track, ev.start))
    X = probe_design(feats, stats)
    (sd / "check.json").write_text(json.dumps({
        "clip": ev.clip_id, "features": {b: feats[b].tolist() for b in F.BANDS},
        "vertex_pred": (X @ W_vert.astype(np.float32)).tolist(), "region_pred": (X @ W_region).tolist(),
        "voxel_pred": (X @ W_vox.astype(np.float32)).tolist()}))

    subs = sorted(p.name[4:] for p in OUT.glob("sub-*") if (p / "meta.json").exists())
    (OUT / "index.json").write_text(json.dumps({"subjects": subs}))
    size = sum(p.stat().st_size for p in sd.glob("*.gz.b64.txt")) / 1e6
    mesh_size = sum(p.stat().st_size for p in (OUT / "mesh").glob("*.gz.b64.txt")) / 1e6
    print(f"{int(lit.sum())} lit voxels -> {len(verts)} of {2 * len(mesh['pial_left'][0])} {MESH} vertices; "
          f"median vertex r {np.median(vertex_r):.3f}; regions with bars: "
          f"{sum(r['column'] is not None for r in regions)} of {len(regions)}; volume {shape}; "
          f"published size: subject {size:.1f} MB + mesh {mesh_size:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    run(ap.parse_args().sub)


if __name__ == "__main__":
    main()
