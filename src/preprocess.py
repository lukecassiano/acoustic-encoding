"""Whole-brain preprocessing of ds003720 raw BOLD, including subcortex.

The authors' Zenodo derivatives are cortex-only with no voxel mask, which rules
out hippocampus / subcortical ROIs and any placement in MNI space. This module
rebuilds the response matrices from raw BOLD with ANTsPy, keeping everything in
the subject's native 2 mm EPI grid and bringing atlases *to* the data.

Stages (each cached under data/derivatives/preproc/sub-XXX/):
  anat      T1 N4 -> MNI head registration -> brain mask -> MNI brain SyN ->
            k-means GM segmentation -> EPI reference -> EPI<->T1 rigid ->
            Harvard-Oxford labels + analysis mask in EPI space
  runs      per run: motion-correct to the EPI reference, mask, regress motion,
            detrend, z-score within run, drop the repeated first clip (10 TRs)
  assemble  stack runs into Resp_Training (4800, V) / Resp_Test (2400, V),
            Resp_Test_Mean (600, V), plus clip tables and a per-voxel table

Usage:
  python -m src.preprocess --sub 001 --stage all --workers 3
"""

import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BIDS = DATA / "ds003720"
TR = 1.5
TRS_PER_CLIP = 10  # 15 s clips / 1.5 s TR
SUBCORTICAL = [  # Harvard-Oxford subcortical labels kept in the analysis mask
    "Left Thalamus", "Left Caudate", "Left Putamen", "Left Pallidum", "Brain-Stem",
    "Left Hippocampus", "Left Amygdala", "Left Accumbens",
    "Right Thalamus", "Right Caudate", "Right Putamen", "Right Pallidum",
    "Right Hippocampus", "Right Amygdala", "Right Accumbens",
]
GM_THRESHOLD = 0.3  # partial-volume tolerant at 2 mm


def outdir(sub):
    d = DATA / "derivatives" / "preproc" / f"sub-{sub}"
    (d / "xfm").mkdir(parents=True, exist_ok=True)
    (d / "runs").mkdir(parents=True, exist_ok=True)
    return d


def bold_runs(sub):
    func = BIDS / f"sub-{sub}" / "func"
    runs = [(task, int(p.name.split("run-")[1][:2]), p)
            for task in ("Training", "Test")
            for p in sorted(func.glob(f"sub-{sub}_task-{task}_run-*_bold.nii"))]
    return runs


def load_bold(path):
    import ants
    import nibabel as nib
    img = nib.load(path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    return ants.from_nibabel_nifti(nib.Nifti1Image(data, img.affine, img.header))


def keep_transforms(files, dest, prefix):
    """ANTs writes transforms to temp files; copy them somewhere durable."""
    kept = []
    for i, f in enumerate(files):
        target = dest / f"{prefix}_{i}{''.join(Path(f).suffixes)}"
        shutil.copy(f, target)
        kept.append(str(target))
    return kept


# ---------------------------------------------------------------- anat stage

def run_anat(sub):
    os.environ.setdefault("TEMPLATEFLOW_HOME", str(DATA / "templateflow"))  # read at import
    import ants
    from nilearn import datasets
    from templateflow import api as tf

    out = outdir(sub)

    # T1: bias-correct, then skull-strip by pulling the MNI brain mask through
    # a whole-head registration
    t1 = ants.n4_bias_field_correction(
        ants.image_read(str(BIDS / f"sub-{sub}" / "anat" / f"sub-{sub}_T1w.nii")))
    mni_head = ants.image_read(str(tf.get("MNI152NLin6Asym", resolution=1, suffix="T1w", desc=None)))
    mni_mask = ants.image_read(str(tf.get("MNI152NLin6Asym", resolution=1, desc="brain", suffix="mask")))
    head_reg = ants.registration(fixed=t1, moving=mni_head, type_of_transform="SyN")
    t1_mask = ants.apply_transforms(t1, mni_mask, head_reg["fwdtransforms"], interpolator="nearestNeighbor")
    t1_mask = ants.morphology(t1_mask, "close", 2)
    t1_brain = t1 * t1_mask

    # MNI -> T1, brain to brain. fwdtransforms map T1-space points into MNI.
    mni_reg = ants.registration(fixed=t1_brain, moving=mni_head * mni_mask, type_of_transform="SyN")
    mni_to_t1 = keep_transforms(mni_reg["fwdtransforms"], out / "xfm", "mni_to_t1")
    t1_to_mni = keep_transforms(mni_reg["invtransforms"], out / "xfm", "t1_to_mni")

    # tissue classes from the subject's own T1 (k-means: CSF < GM < WM)
    seg = ants.kmeans_segmentation(t1_brain, k=3, kmask=t1_mask)
    gm_t1 = seg["probabilityimages"][1]

    # EPI reference: mean of the first training run (motion is small; every run
    # is then rigidly corrected to this volume)
    first = next(p for task, run, p in bold_runs(sub) if task == "Training" and run == 1)
    epi_ref = ants.n4_bias_field_correction(ants.get_average_of_timeseries(load_bold(first)))

    # T1 -> EPI, rigid. fwdtransforms map EPI-space points into T1.
    epi_reg = ants.registration(fixed=epi_ref, moving=t1_brain, type_of_transform="Rigid")
    t1_to_epi = keep_transforms(epi_reg["fwdtransforms"], out / "xfm", "t1_to_epi")
    epi_to_t1 = keep_transforms(epi_reg["invtransforms"], out / "xfm", "epi_to_t1")

    # MNI -> EPI chain: first-listed transform is nearest the output (EPI) grid
    mni_to_epi = t1_to_epi + mni_to_t1
    pd.Series({"mni_to_epi": mni_to_epi, "epi_to_mni": epi_to_t1 + t1_to_mni,
               "mni_to_epi_invert": [False] * len(mni_to_epi)}).to_json(out / "xfm" / "chains.json")

    ho_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm", data_dir=str(DATA / "nilearn"))
    ho_sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm", data_dir=str(DATA / "nilearn"))
    to_epi = lambda img, interp: ants.apply_transforms(epi_ref, img, mni_to_epi, interpolator=interp)
    cort_epi = to_epi(ants.from_nibabel_nifti(_as_nifti(ho_cort.maps)), "genericLabel")
    sub_epi = to_epi(ants.from_nibabel_nifti(_as_nifti(ho_sub.maps)), "genericLabel")
    brain_epi = to_epi(mni_mask, "nearestNeighbor")
    gm_epi = ants.apply_transforms(epi_ref, gm_t1, t1_to_epi, interpolator="linear")
    t1_epi = ants.apply_transforms(epi_ref, t1_brain, t1_to_epi, interpolator="linear")

    sub_ids = [ho_sub.labels.index(name) for name in SUBCORTICAL]
    deep = np.isin(sub_epi.numpy(), sub_ids)
    mask = ((gm_epi.numpy() > GM_THRESHOLD) | deep) & (brain_epi.numpy() > 0)

    for name, img in [("t1_n4", t1), ("t1_brainmask", t1_mask), ("t1_gm_prob", gm_t1),
                      ("epi_ref", epi_ref), ("epi_brainmask", brain_epi), ("epi_gm_prob", gm_epi),
                      ("epi_t1", t1_epi), ("epi_ho_cort", cort_epi), ("epi_ho_sub", sub_epi)]:
        ants.image_write(img, str(out / f"{name}.nii.gz"))
    ants.image_write(epi_ref.new_image_like(mask.astype(np.float32)), str(out / "epi_analysis_mask.nii.gz"))

    pd.DataFrame({"index": range(len(ho_cort.labels)), "label": ho_cort.labels}).to_csv(out / "labels_ho_cort.csv", index=False)
    pd.DataFrame({"index": range(len(ho_sub.labels)), "label": ho_sub.labels}).to_csv(out / "labels_ho_sub.csv", index=False)
    print(f"anat done: {int(mask.sum())} voxels in analysis mask "
          f"({int((deep & mask).sum())} subcortical)")


def _as_nifti(img):
    import nibabel as nib
    return nib.load(img) if isinstance(img, (str, Path)) else img


# ---------------------------------------------------------------- runs stage

def affine_to_six(path):
    """ANTs 12-parameter affine -> 3 rotations (rad) + 3 translations (mm)."""
    import ants
    from scipy.spatial.transform import Rotation
    p = np.asarray(ants.read_transform(path).parameters)
    rot = Rotation.from_matrix(p[:9].reshape(3, 3)).as_euler("xyz")
    return np.concatenate([rot, p[9:12]])


def process_run(args):
    sub, task, run, path = args
    out = outdir(sub)
    target = out / "runs" / f"task-{task}_run-{run:02d}.npy"
    if target.exists():
        return f"{task} {run:02d}: cached"

    import ants
    from nilearn import signal

    ref = ants.image_read(str(out / "epi_ref.nii.gz"))
    mask = ants.image_read(str(out / "epi_analysis_mask.nii.gz")).numpy() > 0
    mc = ants.motion_correction(load_bold(path), fixed=ref, type_of_transform="BOLDRigid")

    motion = np.array([affine_to_six(f[0]) for f in mc["motion_parameters"]])
    ts = mc["motion_corrected"].numpy()[mask].T  # (time, voxels)
    ants.image_write(ants.get_average_of_timeseries(mc["motion_corrected"]),
                     str(out / "runs" / f"task-{task}_run-{run:02d}_mean.nii.gz"))
    del mc

    confounds = np.hstack([motion, np.vstack([np.zeros(6), np.diff(motion, axis=0)])])
    clean = signal.clean(ts, detrend=True, standardize="zscore_sample",
                         confounds=confounds, t_r=TR)
    np.save(target, clean[TRS_PER_CLIP:].astype(np.float32))

    cols = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]
    mdf = pd.DataFrame(motion, columns=cols)
    mdf["fd"] = mc_fd(motion)
    mdf.to_csv(out / "runs" / f"task-{task}_run-{run:02d}_motion.tsv", sep="\t", index=False)
    return f"{task} {run:02d}: max FD {mdf.fd.max():.2f} mm, mean FD {mdf.fd.mean():.3f} mm"


def mc_fd(motion, radius=50.0):
    """Power et al. framewise displacement; rotations converted on a 50 mm sphere."""
    d = np.abs(np.diff(motion, axis=0))
    return np.concatenate([[0.0], (d[:, :3] * radius).sum(1) + d[:, 3:].sum(1)])


def run_runs(sub, workers):
    threads = max(1, (os.cpu_count() or 4) // workers)
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    jobs = [(sub, task, run, p) for task, run, p in bold_runs(sub)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for msg in pool.map(process_run, jobs):
            print(msg, flush=True)


# ------------------------------------------------------------ assemble stage

def clip_table(sub, task):
    """events.tsv rows that survive dropping each run's repeated first clip."""
    rows = []
    for f in sorted((BIDS / f"sub-{sub}" / "func").glob(f"sub-{sub}_task-{task}_run-*_events.tsv")):
        ev = pd.read_csv(f, sep="\t").iloc[1:].copy()
        ev["run"] = int(f.name.split("run-")[1][:2])
        rows.append(ev)
    ev = pd.concat(rows, ignore_index=True)
    ev["genre"] = ev["genre"].str.strip("'")
    ev["clip_id"] = ev["genre"] + "." + ev["track"].astype(str).str.zfill(5) + "@" + ev["start"].round(2).astype(str)
    ev["tr_start"] = np.arange(len(ev)) * TRS_PER_CLIP
    return ev


def run_assemble(sub):
    import nibabel as nib
    out = outdir(sub)
    for task in ("Training", "Test"):
        runs = sorted((out / "runs").glob(f"task-{task}_run-*.npy"))
        resp = np.concatenate([np.load(r) for r in runs if "_mean" not in r.name])
        np.save(out / f"Resp_{task}.npy", resp)
        ev = clip_table(sub, task)
        assert len(ev) * TRS_PER_CLIP == resp.shape[0], (task, len(ev), resp.shape)
        ev.to_csv(out / f"clips_{task}.csv", index=False)
        print(f"Resp_{task}: {resp.shape}")

    # average the 4 presentations of each test clip, aligned on clip onset
    test = np.load(out / "Resp_Test.npy", mmap_mode="r")
    ev = clip_table(sub, "Test")
    order = list(dict.fromkeys(ev["clip_id"]))
    mean = np.stack([np.mean([test[s:s + TRS_PER_CLIP] for s in ev.loc[ev.clip_id == c, "tr_start"]], axis=0)
                     for c in order]).reshape(-1, test.shape[1])
    np.save(out / "Resp_Test_Mean.npy", mean.astype(np.float32))
    pd.Series(order, name="clip_id").to_csv(out / "clips_Test_Mean.csv", index=False)
    print(f"Resp_Test_Mean: {mean.shape}")

    # per-voxel table: index into the EPI grid + atlas labels
    mask = nib.load(out / "epi_analysis_mask.nii.gz").get_fdata() > 0
    ijk = np.argwhere(mask)
    cort = nib.load(out / "epi_ho_cort.nii.gz").get_fdata()[mask].astype(int)
    subc = nib.load(out / "epi_ho_sub.nii.gz").get_fdata()[mask].astype(int)
    cl = pd.read_csv(out / "labels_ho_cort.csv")["label"].tolist()
    sl = pd.read_csv(out / "labels_ho_sub.csv")["label"].tolist()
    vox = pd.DataFrame(ijk, columns=["i", "j", "k"])
    vox["ho_cort"] = [cl[c] for c in cort]
    vox["ho_sub"] = [sl[s] for s in subc]
    vox["subcortical"] = vox["ho_sub"].isin(SUBCORTICAL)
    vox.to_csv(out / "voxels.csv", index_label="voxel")
    print(f"voxels: {len(vox)} ({int(vox.subcortical.sum())} subcortical)")


# ----------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default="001")
    ap.add_argument("--stage", choices=["anat", "runs", "assemble", "all"], default="all")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.stage in ("anat", "all"):
        run_anat(a.sub)
    if a.stage in ("runs", "all"):
        run_runs(a.sub, a.workers)
    if a.stage in ("assemble", "all"):
        run_assemble(a.sub)


if __name__ == "__main__":
    main()
