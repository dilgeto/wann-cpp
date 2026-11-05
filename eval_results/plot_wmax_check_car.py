#!/usr/bin/env python3
"""
Figuras del control de w_max de wmax_check_car.py (WANN-Car, red ganadora rank00_seed06, peso 1.0): cómo cambia
la retención al ruido según qué fracción de g_max ocupa el peso. Lee eval_results/car_wmax_check/summary.csv (y,
como referencia, los modelos principales de SGRL de eval_results/car_wann_vs_sgrl/). No simula nada.

  <prefijo>_time.png        retención vs. tiempo desde la programación (escala de ruido 1), una línea por ocupación
                            del rango (w_max); líneas punteadas = los SGRL principales de la comparación original.
  <prefijo>_occupancy.png   retención vs. ocupación del rango de conductancia (12,5-100 %), una línea por condición
                            de ruido; marca el nivel 1 de ODIN (14 %).

Prefijo por defecto `car_wmax_check`, distinto de car_wann_vs_sgrl_*/car_replicas_*: no pisa las figuras anteriores.

Uso:
  python eval_results/plot_wmax_check_car.py
"""
import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR.parent)
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_style import TIME_TICK_LABELS, setup_publication_style, tagged_dir

YEAR = 31536000.0
SHORT = {"pcm": "PCM", "reramcmo": "ReRAM CMO/HfOx"}
# ocupación -> azul secuencial (más oscuro = más conductancia): un solo tono, orden claro
OCC_COLORS = {8.0: "#9ECAE1", 4.0: "#4292C6", 2.0: "#2171B5", 1.0: "#08306B"}
OCC_MARKERS = {8.0: "o", 4.0: "s", 2.0: "^", 1.0: "D"}
# colores de los métodos SGRL, los mismos de plot_noise_compare.py (referencias punteadas)
SGRL = {"sgrl": ("#E69F00", "SGRL 64×64"), "sgrl_con": ("#009E73", "SGRL same connections"),
        "sgrl_neu": ("#CC79A7", "SGRL same neurons")}
PLANS = {"pcm": ([0.0, YEAR], ["0\n(just programmed)", "1 year"]),
         "reramcmo": ([1.0, 86400.0, YEAR], ["1 s", "1 day", "1 year"])}


def occ(w):
    return 100.0 / w


def sgrl_refs():
    d = Path("eval_results/car_wann_vs_sgrl")
    if not (d / "aggregated.csv").is_file():
        return None, {}
    ag, mm = pd.read_csv(d / "aggregated.csv"), pd.read_csv(d / "models.csv")
    return ag, {r.source: r.model_id for r in mm[mm.main == 1].itertuples() if r.source in SGRL}


def cell(res, m, s, t, w):
    r = res[(res.model == m) & np.isclose(res.scale, s) & np.isclose(res.t, t) & np.isclose(res.wmax, w)]
    return None if r.empty else r.iloc[0]


def fig_time(res, ag, mains, out, wmaxs):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharey=True)
    for ax, (dev, (ts, labels)) in zip(axes, PLANS.items()):
        xs = np.arange(len(ts))
        for w in wmaxs:
            rows = [cell(res, dev, 1.0, t, w) for t in ts]
            y = np.array([r.ret for r in rows])
            ax.fill_between(xs, [r.lo for r in rows], [r.hi for r in rows], color=OCC_COLORS[w], alpha=0.16, linewidth=0)
            ax.plot(xs, y, color=OCC_COLORS[w], marker=OCC_MARKERS[w], lw=2.4, ms=7, zorder=4)
        if ag is not None:
            for src, (col, _) in SGRL.items():
                v = [ag[(ag.model_id == mains[src]) & (ag.noise_model == dev) & (ag.mechanism == "all")
                        & np.isclose(ag.scale, 1.0) & np.isclose(ag.t_inference, t)].ret_mean.iloc[0] for t in ts]
                ax.plot(xs, v, color=col, ls="--", lw=1.8, marker="x", ms=7, zorder=3)
        ax.axhline(100, color="0.5", lw=1, ls=":")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.25, len(ts) - 0.75)
        ax.set_title(f"{SHORT[dev]} (noise scale 1)", fontsize=14)
        ax.set_xlabel("Time since programming")
    axes[0].set_ylabel("Retained performance (% of noise-free)")
    axes[0].set_ylim(0, 112)
    handles = [Line2D([], [], color=OCC_COLORS[w], marker=OCC_MARKERS[w], lw=2.4, ms=7,
                      label=f"WANN, weight uses {occ(w):.1f}% of g_max (w_max = {w:g})") for w in wmaxs]
    if ag is not None:
        handles += [Line2D([], [], color=c, ls="--", lw=1.8, marker="x", ms=7, label=f"{n} (original comparison)")
                    for c, n in SGRL.values()]
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, -0.03, "CarTrack. Same WANN network (rank00 seed06) and shared weight (1.0) in every WANN line; only the "
             "weight-to-conductance mapping (w_max) changes. Bands: 95% bootstrap CI over 60 chips.",
             ha="center", va="top", fontsize=10, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def fig_occupancy(res, out, wmaxs):
    conds = [("reramcmo", 1.0, YEAR, "ReRAM CMO, scale 1, 1 year", "#D55E00", "o", "-"),
             ("reramcmo", 1.0, 86400.0, "ReRAM CMO, scale 1, 1 day", "#E69F00", "s", "-"),
             ("reramcmo", 3.0, 1.0, "ReRAM CMO, scale 3, 1 s", "#009E73", "^", "-"),
             ("reramcmo", 0.0, 1.0, "ReRAM CMO, no noise (g_min floor only)", "#56B4E9", "D", "-"),
             ("pcm", 1.0, YEAR, "PCM, scale 1, 1 year", "#0072B2", "v", "--")]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    xs = np.array([occ(w) for w in wmaxs])
    order = np.argsort(xs)
    for m, s, t, lab, col, mk, ls in conds:
        rows = [cell(res, m, s, t, w) for w in wmaxs]
        y = np.array([r.ret for r in rows])[order]
        lo = np.array([r.lo for r in rows])[order]
        hi = np.array([r.hi for r in rows])[order]
        ax.fill_between(xs[order], lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.plot(xs[order], y, color=col, marker=mk, lw=2.4, ls=ls, ms=7, label=lab, zorder=4)
        ax.annotate(f"{y[0]:.0f}%", (xs[order][0], y[0]), textcoords="offset points", xytext=(-6, -14), ha="right",
                    fontsize=9, color=col)
    ax.axvline(100 / 7, color="0.35", lw=1.2, ls="-.")
    ax.text(100 / 7, 4, " ODIN level 1\n (w_max = 7, 14%)", fontsize=10, color="0.3", va="bottom")
    ax.axhline(100, color="0.5", lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs[order])
    ax.set_xticklabels([f"{x:g}%" for x in xs[order]])
    ax.minorticks_off()
    ax.set_xlim(xs.min() * 0.8, xs.max() * 1.1)
    ax.set_ylim(0, 112)
    ax.set_xlabel("Fraction of the conductance range (g_max) used by the shared weight")
    ax.set_ylabel("Retained performance (% of noise-free)")
    ax.set_title("WANN-Car: noise robustness depends on where the weight sits in the range", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    fig.text(0.5, -0.01, "CarTrack. Same network (rank00 seed06) and shared weight (1.0); only w_max changes (8, 4, 2, 1 → 12.5, 25, 50, 100%). "
             "Bands: 95% bootstrap CI over 60 chips.", ha="center", va="top", fontsize=9.5, color="0.35")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("eval_results/car_wmax_check"))
    ap.add_argument("--out-dir", default="graficos/Racing Car")
    ap.add_argument("--prefix", default="car_wmax_check")
    ap.add_argument("--flat", action="store_true", help="no crear la subcarpeta ruido/<tag>/ (organización antigua)")
    a = ap.parse_args()
    if not (a.data_dir / "summary.csv").is_file():
        sys.exit(f"Error: no existe {a.data_dir / 'summary.csv'}. Corré wmax_check_car.py primero.")
    res = pd.read_csv(a.data_dir / "summary.csv")
    wmaxs = sorted(res.wmax.unique(), reverse=True)           # 8, 4, 2, 1: ocupación creciente
    setup_publication_style()
    ag, mains = sgrl_refs()
    out = tagged_dir(a.out_dir, a.prefix, a.flat)
    fig_time(res, ag, mains, out / f"{a.prefix}_time.png", wmaxs)
    fig_occupancy(res, out / f"{a.prefix}_occupancy.png", wmaxs)


if __name__ == "__main__":
    main()
