"""Paper figures for the perception-1 run (Task 9). Nothing is fitted here.

  hierarchy        held-out R2 (95% CI) for L0, L0+L1, L2 (listener mean, range), theory
  dissociation     acoustic vs structural R2 per rating, at L0 and per listener at L2, H2 verdicts
  projection_null  per listener x rating: the H3 random-projection null with the L2 R2 marker
  psychophysics    ITD and ILD vs azimuth for 4 cochlear channels, Woodworth ITD overlaid
  generalisation   H4: PMEmo Pearson r (95% CI) for the page readout and descriptive models

Reads data/perception/results.json (schema: src/perception/run.py docstring); the psychophysics
panel needs only the ear model.

    python -m src.perception.figures [--results PATH] [--out DIR]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from src import features as F
from src.figures import AXIS, BLUE, INK, MUTED, ORANGE, SECONDARY, SURFACE, style, title
from src.perception.run import RATINGS, alpha_at_edge

DPI = 300
TOP_MARK = "†"
FOOTNOTE = TOP_MARK + " ridge alpha at an end of the grid"
# verdict text never reuses the acoustic / structural BLUE / ORANGE
VERDICT_COLOR = {"supported": INK, "not supported": "#9b2c3a", "inconclusive": MUTED}
OUT_DEFAULT = F.ROOT / "figures" / "perception"
RESULTS_DEFAULT = F.ROOT / "data" / "perception" / "results.json"


def _save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(out / f"{name}.png")


def _top(rec):
    """Ridge alpha at either end of the frozen grid (run.alpha_at_edge, the results.md predicate)."""
    return bool(rec) and alpha_at_edge(rec.get("alpha"))


def _mark_top(ax, x, y):
    ax.annotate(TOP_MARK, (x, y), xytext=(4, 0), textcoords="offset points",
                fontsize=8, color=SECONDARY, va="center")


def _footnote(fig, used, y=-0.01):
    if used:
        fig.text(0.01, y, FOOTNOTE, fontsize=7.5, color=SECONDARY, ha="left", va="top")


def _verdict_text(ax, text, verdict, y=1.0):
    ax.text(0.99, y, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            color=VERDICT_COLOR.get(verdict, SECONDARY))


# ---------------------------------------------------------------- hierarchy

def hierarchy(res, out):
    m = res["models"]
    rows = ["L0", "L0+L1", "L2", "theory"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    used = False
    for ax, rating in zip(axes, RATINGS):
        style(ax, grid="x")
        for i, key in enumerate(rows):
            y = len(rows) - 1 - i
            if key == "L2":
                vals = np.array([m["L2"][s][rating]["r2"] for s in m["L2"]], float)
                mid, lo, hi = vals.mean(), vals.min(), vals.max()
                top = False
                ax.plot([lo, hi], [y, y], color=BLUE, lw=2, solid_capstyle="butt")
                off = np.linspace(-0.24, 0.24, len(vals))  # spread listeners vertically so they do not overprint
                ax.plot(vals, y + off, "o", color=BLUE, ms=3, alpha=0.8, zorder=2)
                for s, v in zip(m["L2"], vals):  # dagger at the listener's own tick
                    if _top(m["L2"][s][rating]):
                        used = True
                        _mark_top(ax, v, y + 0.18)
                col = BLUE
            else:
                r = m[key][rating]
                mid, lo, hi, top, col = r["r2"], r["r2_lo"], r["r2_hi"], _top(r), INK
                ax.plot([lo, hi], [y, y], color=INK, lw=1.4)
            ax.plot(mid, y, "D" if key == "L2" else "o", color=col, mfc=SURFACE if key == "L2" else col,
                    ms=5, zorder=3)
            if top:
                used = True
                _mark_top(ax, max(hi, mid), y)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(range(len(rows)), rows[::-1])
        ax.set_xlabel("held-out $R^2$")
        title(ax, rating)
    _footnote(fig, used)
    _save(fig, out, "hierarchy")


# ---------------------------------------------------------------- dissociation

def dissociation(res, out):
    m, h2 = res["models"], res["hypotheses"]["H2"]
    subs = res["meta"]["listeners"]
    labels = ["L0"] + [f"L2 {s}" for s in subs]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4), sharey=True)
    used = False
    for k, (ax, rating) in enumerate(zip(axes, RATINGS)):
        style(ax, grid="x")
        pairs = [(m["L0_acoustic"][rating], m["L0_structural"][rating])]
        pairs += [(m["L2_acoustic"][s][rating], m["L2_structural"][s][rating]) for s in subs]
        for i, (a, b) in enumerate(pairs):
            y = len(labels) - 1 - i
            for rec, col, dy in ((a, BLUE, 0.13), (b, ORANGE, -0.13)):
                lo, hi = rec.get("r2_lo", rec["r2"]), rec.get("r2_hi", rec["r2"])
                ax.plot([lo, hi], [y + dy] * 2, color=col, lw=1.2)
                ax.plot(rec["r2"], y + dy, "o", color=col, ms=4.5)
                if _top(rec):
                    used = True
                    _mark_top(ax, max(hi, rec["r2"]), y + dy)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(range(len(labels)), labels[::-1])
        ax.set_xlabel("held-out $R^2$")
        title(ax, rating)
        d = h2["L0"][rating]
        _verdict_text(ax, f"diff {d['diff']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]", d["verdict"], y=1.04)
        if k == 1:  # per-listener H2 verdicts (both ratings) beside the right panel
            ax.annotate("H2 (both ratings)", (1.02, len(labels) - 0.5), xycoords=("axes fraction", "data"),
                        fontsize=7, color=SECONDARY, va="bottom", fontweight="bold")
            for i, v in enumerate([h2["L0"]["verdict"]] + [h2["L2"]["listeners"][s]["verdict"] for s in subs]):
                ax.annotate(v, (1.02, len(labels) - 1 - i), xycoords=("axes fraction", "data"),
                            fontsize=7, color=VERDICT_COLOR.get(v, SECONDARY), va="center")
    axes[0].plot([], [], "o", color=BLUE, label="acoustic")
    axes[0].plot([], [], "o", color=ORANGE, label="structural")
    fig.legend(handles=axes[0].get_lines()[-2:], labels=["acoustic", "structural"], frameon=False,
               fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, labelcolor=SECONDARY)
    fig.text(0.5, 1.0, "H2 L0: " + res["verdicts"]["H2_L0"] + "     H2 L2: " + res["verdicts"]["H2_L2"],
             ha="center", fontsize=8, color=SECONDARY)
    fig.subplots_adjust(right=0.86, wspace=0.12)
    _footnote(fig, used)
    _save(fig, out, "dissociation")


# ---------------------------------------------------------------- projection null

def projection_null(res, out):
    h3, subs = res["hypotheses"]["H3"], res["meta"]["listeners"]
    fig, axes = plt.subplots(len(RATINGS), len(subs), figsize=(2.2 * len(subs), 4.8), squeeze=False)
    for i, rating in enumerate(RATINGS):
        for j, s in enumerate(subs):
            ax = axes[i, j]
            style(ax, grid="y")
            d = h3[rating]["listeners"][s]
            ax.hist(d["null"], bins=20, color=MUTED, alpha=0.7, edgecolor=SURFACE)
            ax.axvline(d["null_p95"], color=SECONDARY, lw=1, ls="--")
            ax.axvline(d["r2"], color=BLUE if d["pass"] else ORANGE, lw=2)
            title(ax, s + " " + rating)
            ax.set_xlabel("$R^2$")
            ax.set_yticks([])
            ax.xaxis.set_major_locator(MaxNLocator(3))
            ax.ticklabel_format(axis="x", useOffset=False)
            ax.tick_params(axis="x", labelsize=7)
        _verdict_text(axes[i, -1], "H3 " + rating + ": " + h3[rating]["verdict"], h3[rating]["verdict"], y=1.25)
    fig.tight_layout()
    fig.legend(handles=[Line2D([], [], color=BLUE, lw=2, label="L2 passes (at or above null p95)"),
                        Line2D([], [], color=ORANGE, lw=2, label="L2 fails (below null p95)"),
                        Line2D([], [], color=SECONDARY, lw=1, ls="--", label="null 95th pct")],
               frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3,
               labelcolor=SECONDARY)
    _save(fig, out, "projection_null")


# ---------------------------------------------------------------- psychophysics

def psychophysics(out):
    from src.ear.brainstem import listen
    from src.ear.constants import EAR_SR
    from src.ear.room import Source, head_itd_s
    az = np.arange(-90, 91, 30)
    sig = np.random.default_rng(0).standard_normal(int(0.4 * EAR_SR)) * 0.1
    itd, ild, cfs = [], [], None
    for a in az:
        e = listen([Source(sig, float(a), 1.0)])
        itd.append(np.nanmedian(e.cues["itd_s"], axis=1))
        ild.append(np.median(e.cues["ild_db"], axis=1))
        cfs = e.left.cfs
    itd, ild = np.array(itd) * 1e6, np.array(ild)
    chans = [int(np.argmin(np.abs(cfs - f))) for f in (250, 500, 1500, 5000)]
    cols = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    fine = np.linspace(-90, 90, 91)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, data, ylab in ((axes[0], itd, "ITD ($\\mu$s)"), (axes[1], ild, "ILD (dB)")):
        style(ax, grid="both")
        for c, col in zip(chans, cols):
            ax.plot(az, data[:, c], "o-", color=col, ms=4, lw=1.3, label=f"{cfs[c]:.0f} Hz")
        ax.set_xlabel("azimuth (deg, + left)")
        ax.set_ylabel(ylab)
        ax.set_xticks(az)
    axes[0].plot(fine, [head_itd_s(float(a)) * 1e6 for a in fine], color=INK, lw=1, ls="--", label="Woodworth")
    title(axes[0], "ITD")
    title(axes[1], "ILD")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=SECONDARY)
    fig.tight_layout()
    _save(fig, out, "psychophysics")


# ---------------------------------------------------------------- generalisation

def generalisation(res, out):
    h4, pm, subs = res["hypotheses"]["H4"], res["pmemo_models"], res["meta"]["listeners"]
    rows = [("page", s) for s in subs] + [("L2", s) for s in subs] + [(k, None) for k in ("L0", "L0+L1", "theory")]
    mdl = res["models"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), sharey=True)
    used = False
    for ax, rating in zip(axes, RATINGS):
        style(ax, grid="x")
        for i, (key, s) in enumerate(rows):
            y = len(rows) - 1 - i
            if key == "page":
                rec = h4[rating]["listeners"][s]
            elif s is not None:
                rec = pm[key][s][rating]
            else:
                rec = pm[key][rating]
            primary = key == "page"
            col = BLUE if primary else MUTED
            ax.plot([rec["lo"], rec["hi"]], [y, y], color=col, lw=1.5 if primary else 1.1)
            ax.plot(rec["r"], y, "o", color=col, ms=5 if primary else 4)
            # H4 models are the DEAM-train fits: alpha comes from results["models"]
            fit = None if key == "theory" else (mdl[key][s][rating] if s else mdl[key][rating])
            if _top(fit):
                used = True
                _mark_top(ax, max(rec["hi"], rec["r"]), y)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks(range(len(rows)), [f"{k} {s}" if s else k for k, s in rows][::-1])
        ax.set_xlabel("PMEmo Pearson $r$")
        title(ax, rating)
        _verdict_text(ax, "H4 " + rating + ": " + h4[rating]["verdict"], h4[rating]["verdict"], y=1.04)
    fig.subplots_adjust(wspace=0.1)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="page readout (primary)"),
                        Line2D([], [], marker="o", ls="", color=MUTED, label="descriptive")],
               frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2,
               labelcolor=SECONDARY)
    _footnote(fig, used, y=-0.08)
    _save(fig, out, "generalisation")


# ---------------------------------------------------------------- entry

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    res = json.loads(Path(a.results).read_text(encoding="utf-8"))
    hierarchy(res, out)
    dissociation(res, out)
    projection_null(res, out)
    psychophysics(out)
    generalisation(res, out)


if __name__ == "__main__":
    main()
