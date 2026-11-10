#!/usr/bin/env python3
"""
Boxplot of the percentage performance loss of the SG networks (surrogate
gradient, dilgeto/surrogate-gradient-rl) when run on the ODIN chip (E4: LIF layers
on the ZCU104 board, closed loop), relative to the original network on the PC (E0).
Two boxes per task, one per SG family:
  runs           -> same number of connections as the WANN
  runs_neuronas  -> same number of neurons as the WANN
Figure text is in English (publication format).

Data: ODIN-SG-WANN-datos/sg/datos_graficos/datos/01_E0_vs_E4_placa_lazo_cerrado_3_tareas/redes.csv
  (one row per network; rendimiento_E4_pct column already computed).

  performance (%) = 100 * (R - floor) / (R_original - floor)
  loss (%)        = 100 - performance
  floor: -500 Acrobot, -200 discrete MountainCar, 0 racing car.

Loss 0 % = same as original; 100 % = achieves nothing (floor). May be negative
(improvement) or >100 % (racing car with return below 0). MountainCar excludes
one network (runs_neuronas rank0 s101) whose original is already at the floor
(percentage undefined).

Usage:  python eval_results/plot_sg_odin_loss.py [--data PATH] [--out PNG]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from plot_style import setup_publication_style

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = (_REPO_ROOT.parent / "ODIN-SG-WANN-datos" / "sg" / "datos_graficos" / "datos"
                / "01_E0_vs_E4_placa_lazo_cerrado_3_tareas" / "redes.csv")
DEFAULT_OUT = _REPO_ROOT / "graficos" / "sg_odin_perdida_rendimiento_boxplot.png"
DEFAULT_OUT_BEST = _REPO_ROOT / "graficos" / "sg_odin_perdida_rendimiento_boxplot_mejor_config.png"

TASKS = [("Acrobot", "Acrobot"), ("MountainCar", "Discrete\nMountain Car"), ("Coche", "Racing car")]
# (columna familia, etiqueta, color, desplazamiento) — Okabe-Ito, seguro para daltonismo
FAMILIES = [("runs", "Same connections as WANN", "#0072B2", -0.19),
            ("runs_neuronas", "Same neurons as WANN", "#E69F00", +0.19)]
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e3"
BOX_W = 0.28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--best-config", action="store_true",
                    help="per (task, family) keep only the hyperparameter configuration (rank0/1/2) with the "
                         "highest mean ORIGINAL (E0, PC) reward over its 11 seeds; the loss on ODIN is "
                         "computed afterwards, only on those 11 networks (selection never looks at the loss)")
    a = ap.parse_args()
    if a.out is None:
        a.out = DEFAULT_OUT_BEST if a.best_config else DEFAULT_OUT
    setup_publication_style()

    df = pd.read_csv(a.data)
    chosen = {}
    if a.best_config:
        m0 = df.groupby(["tarea", "familia", "modelo"]).recompensa_E0_pc.mean()
        for (t, f), g in m0.groupby(level=[0, 1]):
            chosen[(t, f)] = g.droplevel([0, 1]).idxmax()
            print(f"best config {t:12s} {f:14s} -> {chosen[(t, f)]} (E0 mean {g.max():.2f}; "
                  + ", ".join(f"{k[2]} {v:.2f}" for k, v in g.items()) + ")")
        df = df[[chosen[(t, f)] == mo for t, f, mo in zip(df.tarea, df.familia, df.modelo)]]
    df["perdida"] = 100.0 - df["rendimiento_E4_pct"]
    df = df.dropna(subset=["perdida"])

    fig, ax = plt.subplots(figsize=(11, 6.8))
    rng = np.random.default_rng(0)
    counts = {}

    for fam, _, color, off in FAMILIES:
        for i, (task, _) in enumerate(TASKS):
            v = df.loc[(df.tarea == task) & (df.familia == fam), "perdida"].to_numpy()
            counts[(task, fam)] = len(v)
            x = i + off
            bp = ax.boxplot([v], positions=[x], widths=BOX_W, patch_artist=True, showfliers=False,
                            whis=1.5, medianprops=dict(color=INK, lw=2.2),
                            whiskerprops=dict(color=MUTED, lw=1.2), capprops=dict(color=MUTED, lw=1.2),
                            boxprops=dict(edgecolor=color, lw=1.2))
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_alpha(0.30)
            # each dot = one network (all of them, outliers included)
            ax.scatter(x + rng.uniform(-BOX_W * 0.32, BOX_W * 0.32, len(v)), v, s=13, color=color,
                       alpha=0.8, edgecolors="white", linewidths=0.4, zorder=3)
    ax.axhline(0, color=INK, lw=1.3, zorder=1)
    ax.axhline(100, color=MUTED, lw=1, ls="--", zorder=1)

    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([lab for _, lab in TASKS], fontsize=12)
    ax.set_ylabel("Performance loss on ODIN (%)", fontsize=12)
    ax.set_xlim(-0.62, len(TASKS) - 0.38)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, lw=1, alpha=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(fc=c, ec=c, alpha=0.5, label=lab) for _, lab, c, _ in FAMILIES],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=11)
    fig.subplots_adjust(top=0.97, bottom=0.17, left=0.09, right=0.97)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print("saved:", a.out)
    stats = {}
    for (t, _) in TASKS:
        for fam, _, _, _ in FAMILIES:
            v = df.loc[(df.tarea == t) & (df.familia == fam), "perdida"].to_numpy()
            stats[(t, fam)] = v
            print(f"{t:12s} {fam:14s} n={len(v):2d} median={np.median(v):6.1f}  "
                  f"p25={np.percentile(v,25):6.1f}  p75={np.percentile(v,75):6.1f}  "
                  f"min={v.min():6.1f}  max={v.max():6.1f}")

    # caption (explanatory text lives here, not inside the figure)
    cap = ("**Performance loss of surrogate-gradient (SG) spiking networks when run on the ODIN chip.** "
           "Blue: networks with the same number of connections as the WANN; orange: same number of neurons. "
           + ("Per task and family, the hyperparameter configuration with the highest mean original (PC) "
              "reward over its 11 seeds was selected first (" + "; ".join(
                  f"{lab.replace(chr(10), ' ')}: " + " / ".join(f"{chosen[(t, fam)]}" for fam, *_ in FAMILIES)
                  for t, lab in TASKS) + " for connections / neurons), and the loss was computed afterwards. "
              if a.best_config else "All 33 networks per family and task (3 configurations × 11 seeds). ")
           + "Each dot is one network (mean of 20 episodes); boxes show median and interquartile range, "
           "whiskers extend to 1.5 × IQR. Loss = 100 % − performance, with performance = "
           "100·(R − floor)/(R_original − floor), where R_original is the network on the PC (float32, snnTorch neuron) "
           "and R is the reward with the LIF layers running on the ZCU104 board (closed loop, integer weights). "
           "Floor: −500 (Acrobot), −200 (Mountain Car), 0 (racing car). 0 % = same as the original, "
           "100 % = task not achieved, negative = better than the original. The networks were not retrained for ODIN. "
           + ("" if a.best_config else "Mountain Car excludes one network (same-neurons family) whose original is already at the floor. ")
           + "\n\nMedian loss (connections / neurons): " + "; ".join(
               f"{lab.replace(chr(10), ' ')} " + " / ".join(
                   f"{np.median(stats[(t, fam)]):.1f} % (n = {len(stats[(t, fam)])})" for fam, *_ in FAMILIES)
               for t, lab in TASKS) + ".\n")
    cap_path = a.out.with_name(a.out.stem + "_caption.md")
    cap_path.write_text(cap, encoding="utf-8")
    print("caption:", cap_path)


if __name__ == "__main__":
    main()
