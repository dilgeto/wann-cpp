#!/usr/bin/env python3
"""
Macro figures that stack two conductance occupancies of the paper noise figures: 25 % of g_max (top two rows: PCM, ReRAM-CMO)
and 100 % of g_max (bottom two rows: PCM, ReRAM-CMO), tasks in columns. One figure per noise type:

  noise_drift_occ25_occ100.png       retained performance vs. time since programming, noise scale 1
  noise_intensity_occ25_occ100.png   retained performance vs. noise scale (all mechanisms), right after programming

Same data, aggregation and drawing code as graficos/paper/memristors/noise_occupancy/occ25|occ100/noise_{drift,intensity}.png
(plot_paper_noise.py + plot_paper_noise_occupancy.py); this script only lays the two occupancies out in one figure, with one
legend, column titles only on top and the x label only at the bottom. Rows of the same device and column share the y range so the
two occupancies can be compared directly. Reads data only; nothing is simulated or trained.

Outputs (default): graficos/paper/memristors/noise_occupancy/<name>.png and <name>_caption.md.
Usage: python eval_results/plot_paper_noise_occ_macro.py [--occupancies 0.25 1.0] [--kinds drift intensity] [--sgrl-name SG]
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import plot_paper_noise as P
import plot_paper_noise_occupancy as O
from plot_style import JOURNAL_FULL_W, TIME_TICK_LABELS, setup_journal_style, setup_publication_style

OUT = _ROOT / "graficos" / "paper" / "memristors" / "noise_occupancy"
JOURNAL_NAME = {"drift": "figura_05", "intensity": "figura_04"}     # names in graficos/paper/send_to_journal


def name_of(occ):
    return f"occ{O.tag(occ)}"


def occ_text(occ):
    return f"{occ * 100:g} %"


JOURNAL = False      # set from --journal: draw at 190 mm wide with >= 7 pt text and thinner marks


def build_axes(nocc, ndev, ntask):
    nr = nocc * ndev
    size = (JOURNAL_FULL_W, 1.95 * nr + 0.75) if JOURNAL else (13, 3.1 * nr + 0.9)
    fig, axs = plt.subplots(nr, ntask, figsize=size, squeeze=False, sharex="col")
    for j, (_, lab) in enumerate(P.TASKS):
        axs[0][j].set_title(lab)
    return fig, axs


def draw_lines(ax, d, task, dev, xs, getter, rng):
    """P.draw_lines with mark sizes that depend on the output size (identical data and statistics)."""
    thin, thick, ms = (0.4, 1.4, 3.2) if JOURNAL else (0.7, 2.4, 6)
    for src in P.SRC:
        cols = {}
        for k, x in enumerate(xs):
            for rs, val in getter(task, src, dev, x).items():
                cols.setdefault(rs, {})[k] = val
        if not cols:
            continue
        tab = pd.DataFrame(cols).T.reindex(columns=range(len(xs)))
        for _, row in tab.iterrows():
            ax.plot(range(len(xs)), row.to_numpy(), color=P.STYLE[src]["color"], lw=thin, alpha=0.22 if JOURNAL else 0.18, zorder=2)
        ci = np.array([P.boot_ci(tab[k].dropna().to_numpy(), rng) for k in range(len(xs))])
        ax.fill_between(range(len(xs)), ci[:, 0], ci[:, 1], color=P.STYLE[src]["color"], alpha=0.15, lw=0, zorder=3)
        ax.plot(range(len(xs)), tab.mean(axis=0).to_numpy(), color=P.STYLE[src]["color"], lw=thick,
                marker=P.STYLE[src]["marker"], ms=ms, zorder=4)
    ax.axhline(100, color="0.5", lw=0.6 if JOURNAL else 1, ls=":")


def finish(fig, axs, occs, devs, xlabel):
    nd = len(devs)
    for j in range(axs.shape[1]):                      # same y range for the same device and task across occupancies
        for k in range(nd):
            col = [axs[o * nd + k][j] for o in range(len(occs))]
            lo, hi = min(a.get_ylim()[0] for a in col), max(a.get_ylim()[1] for a in col)
            for a in col:
                a.set_ylim(lo, hi)
    for ax in axs[:, 0]:
        ax.set_ylabel("Retained performance (%)")
    for ax in axs[-1]:
        ax.set_xlabel(xlabel)
    for i in range(axs.shape[0]):
        dev = devs[i % nd]
        axs[i][0].annotate(P.DEV_LABEL[dev], xy=(-0.36 if JOURNAL else -0.30, 0.5), xycoords="axes fraction", rotation=90,
                           va="center", ha="center", color=P.DEV_COLOR[dev], fontweight="bold", fontsize=7.5 if JOURNAL else 13)
    h = [Line2D([], [], color=P.STYLE[s]["color"], marker=P.STYLE[s]["marker"], lw=1.4 if JOURNAL else 2.2,
                ms=3.5 if JOURNAL else 7, label=P.LABEL[s]) for s in P.SRC]
    fig.legend(handles=h, loc="lower center", ncol=len(P.SRC), frameon=False, bbox_to_anchor=(0.5, 0.0),
               columnspacing=1.2 if JOURNAL else 2.0, handlelength=1.8 if JOURNAL else 2.0)
    fig.tight_layout(rect=(0.095 if JOURNAL else 0.06, 0.04 if JOURNAL else 0.035, 1, 1), h_pad=0.6 if JOURNAL else 1.08,
                     w_pad=0.6 if JOURNAL else 1.08)
    # occupancy of each row group: bold label at the far left and a thin separator between groups
    for o, occ in enumerate(occs):
        top = axs[o * nd][0].get_position().y1
        bot = axs[o * nd + nd - 1][0].get_position().y0
        fig.text(0.015 if JOURNAL else 0.012, (top + bot) / 2, f"{occ_text(occ)} of $g_{{max}}$", rotation=90, va="center", ha="center",
                 fontweight="bold", fontsize=8 if JOURNAL else 14)
        if o:
            y = (axs[o * nd - 1][0].get_position().y0 + top) / 2
            fig.add_artist(Line2D([0.01, 0.995], [y, y], color="0.6", lw=1.0, ls="--", transform=fig.transFigure))


def make(kind, occs, ds):
    devs = ds[occs[0]].devs
    fig, axs = build_axes(len(occs), len(devs), len(P.TASKS))
    rng = np.random.default_rng(0 if kind == "intensity" else 1)
    for o, occ in enumerate(occs):
        d = ds[occ]
        for k, dev in enumerate(devs):
            for j, (task, _) in enumerate(P.TASKS):
                ax = axs[o * len(devs) + k][j]
                if kind == "drift":
                    xs = [P.T0[dev], 3600.0, 86400.0, 31536000.0]
                    getter = lambda tk, s, dv, x: d.vec(tk, s, dv, "all", 1.0, x)
                    labels = [TIME_TICK_LABELS.get(int(x), str(x)) for x in xs]
                else:
                    xs = P.SCALES
                    getter = lambda tk, s, dv, x: d.vec(tk, s, dv, "all", x, P.T0[dv])
                    labels = [f"{s:g}" for s in xs]
                draw_lines(ax, d, task, dev, xs, getter, rng)
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels(labels)
                ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]), top=min(ax.get_ylim()[1], P.Y_CAP))
    finish(fig, axs, occs, devs, "Time since programming" if kind == "drift" else "Noise scale")
    return fig


def caption(kind, occs, ds):
    two = " and ".join(occ_text(o) for o in occs)
    common = O.common_for(None).replace("at the occupancy given in the figure",
                                        f"at the occupancy of each row group ({occ_text(occs[0])} of g_max in the top two rows, "
                                        f"{occ_text(occs[1])} in the bottom two)")
    if kind == "drift":
        head = ("**noise_drift_occ25_occ100** — Retained performance as a function of the time elapsed since programming, at noise "
                f"scale 1, with the conductance occupancy of g_max set to {two} (top and bottom pair of rows); within each pair, "
                "rows are devices (PCM, ReRAM-CMO) and columns are tasks. The first point is the initial time of each device model "
                "(PCM: 0; ReRAM-CMO: 1 s), then 1 hour, 1 day and 1 year. ")
    else:
        head = ("**noise_intensity_occ25_occ100** — Retained performance as a function of the noise scale (all mechanisms scaled "
                f"together), with the conductance occupancy of g_max set to {two} (top and bottom pair of rows); within each pair, "
                "rows are devices (PCM, ReRAM-CMO) and columns are tasks. Evaluated right after programming (PCM: t = 0; "
                "ReRAM-CMO: t = 1 s). At scale 0 PCM is 100 % by construction; for ReRAM-CMO scale 0 still includes the g_min floor "
                "of the differential pair (8.9 % of g_max), so it can differ from 100 %. ")
    tail = ("Thin lines: individual trained networks; thick line and band: mean over networks and 95 % bootstrap CI. Dotted line: "
            "100 % (noise-free). Rows of the same device and task share the y axis across the two occupancies. Time points "
            "are categorical (not to scale). ") if kind == "drift" else (
            "Thin lines: individual trained networks; thick line and band: mean over networks and 95 % bootstrap CI. Dotted line: "
            "100 % (noise-free). Rows of the same device and task share the y axis across the two occupancies. ")
    return (head + tail + common + P.pending_note(ds[occs[0]])).replace("SGRL", P.SG_NAME).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--occupancies", type=float, nargs=2, default=[0.25, 1.0], help="top and bottom occupancy (fractions of g_max)")
    ap.add_argument("--kinds", nargs="+", default=["drift", "intensity"], choices=["drift", "intensity"])
    ap.add_argument("--sgrl-name", default="SG", help="how the surrogate-gradient method is named in the figure/caption")
    ap.add_argument("--occ-dir", type=Path, default=_SCRIPT_DIR / "paper_noise_occ")
    ap.add_argument("--old-dir", type=Path, default=_SCRIPT_DIR)
    ap.add_argument("--cmo-dir", type=Path, default=_SCRIPT_DIR / "paper_noise")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--journal", action="store_true",
                    help="draw at the final journal size (190 mm wide, >= 7 pt text, embedded fonts) and write vector PDF + 600 dpi PNG "
                         "to <journal-dir>/figura_0N (drift = 05, intensity = 04) instead of --out")
    ap.add_argument("--journal-dir", type=Path, default=_ROOT / "graficos" / "paper" / "send_to_journal")
    a = ap.parse_args()
    global JOURNAL
    JOURNAL = a.journal
    a.cache_dir = a.occ_dir / "merged"
    (setup_journal_style if a.journal else setup_publication_style)()
    P.SG_NAME = a.sgrl_name
    P.COMMON_ORIG = P.COMMON
    nm0 = pd.read_csv(a.occ_dir / "car_replicas" / "models.csv")
    if (nm0.source == "sgrl_64_occ").any():
        P.SRC = ["wann", "sgrl_con", "sgrl_neu", "sgrl_64"]
    P.LABEL = {k: v.replace("SGRL", a.sgrl_name) for k, v in P.LABEL.items()}
    occs = list(a.occupancies)
    ds = {o: O.load(o, Namespace(old_dir=a.old_dir, cmo_dir=a.cmo_dir, occ_dir=a.occ_dir, cache_dir=a.cache_dir)) for o in occs}
    a.out.mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        fig = make(kind, occs, ds)
        stem = f"noise_{kind}_{name_of(occs[0])}_{name_of(occs[1])}"
        if a.journal:
            a.journal_dir.mkdir(parents=True, exist_ok=True)
            for ext in (".pdf", ".png"):
                fig.savefig(a.journal_dir / (JOURNAL_NAME[kind] + ext))
        else:
            fig.savefig(a.out / f"{stem}.png")
        plt.close(fig)
        (a.out / f"{stem}_caption.md").write_text(caption(kind, occs, ds).replace(f"noise_{kind}_occ25_occ100", stem), encoding="utf-8")
        print("wrote", (a.journal_dir / JOURNAL_NAME[kind]) if a.journal else a.out / f"{stem}.png")


if __name__ == "__main__":
    main()
