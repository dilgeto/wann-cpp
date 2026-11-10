#!/usr/bin/env python3
"""
Boxplot of the percentage performance loss of the WANN networks adapted to ODIN
(measured on the ZCU104 board) relative to the ORIGINAL WANN network in software.
Companion of plot_sg_odin_loss.py (same loss definition and style). Figure text in English.

Inputs
  - ODIN board means: ODIN-SG-WANN-datos/wann/datos_graficos/{03,04,05}_*/resumen.csv
      car           03: rank00_seed00..10 (car_ttfs_first_spike_discrete), episodes 100-199
      mountain_car  04: disc_mc_small_first_spike_discrete/rank01_seed00..10, episodes 100-199
      acrobot       05: acrobot_small_first_spike/rank01_seed00..10, episodes 0-99
  - Software reference: eval_results/wann_odin_reference_episodes.csv
      (made by eval_wann_odin_reference.py: original network + its winner weight, same episode indices)

  performance (%) = 100 * (R_odin - floor) / (R_software - floor)
  loss (%)        = 100 - performance
  floor: -500 Acrobot, -200 discrete MountainCar, 0 racing car.
Networks whose software reward is at/below the floor (ratio undefined) are excluded.
Loss 0 % = same as original software network; <0 = better on ODIN; 100 % = achieves nothing.

Usage: python eval_results/plot_wann_odin_loss.py [--odin-dir DIR] [--ref CSV] [--out PNG]
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_style import setup_publication_style

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ODIN = _REPO_ROOT.parent / "ODIN-SG-WANN-datos" / "wann" / "datos_graficos"
DEFAULT_REF = _REPO_ROOT / "eval_results" / "wann_odin_reference_episodes.csv"
DEFAULT_OUT = _REPO_ROOT / "graficos" / "wann_odin_perdida_rendimiento_boxplot.png"

TASKS = [("acrobot", "Acrobot", -500.0), ("mountain_car", "Discrete\nMountain Car", -200.0),
         ("car", "Racing car", 0.0)]
BOX = "#009E73"      # Okabe-Ito green: WANN (SG figure uses blue/orange)
INK, MUTED, GRID = "#1a1a1a", "#555555", "#e3e3e3"


def load_odin(d: Path) -> pd.DataFrame:
    car = pd.read_csv(d / "03_coche_11_redes_2026-11-08" / "resumen.csv")
    car["task"], car["net"] = "car", car.red
    mc = pd.read_csv(d / "04_mountain_car_33_redes_2026-11-08" / "resumen.csv")
    mc = mc[mc.red.str.startswith("disc_mc_small_first_spike_discrete/")].copy()
    mc["task"], mc["net"] = "mountain_car", mc.red.str.split("/").str[1]
    ac = pd.read_csv(d / "05_acrobot_11_redes_2026-11-08" / "resumen.csv")
    ac["task"], ac["net"] = "acrobot", ac.red.str.split("/").str[1]
    return pd.concat([x[["task", "net", "media_placa"]] for x in (car, mc, ac)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odin-dir", type=Path, default=DEFAULT_ODIN)
    ap.add_argument("--ref", type=Path, default=DEFAULT_REF)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    setup_publication_style()

    sw = (pd.read_csv(a.ref).groupby(["task", "net"]).reward_software.mean().rename("r_sw").reset_index())
    m = load_odin(a.odin_dir).merge(sw, on=["task", "net"], how="inner")
    floor = {t: f for t, _, f in TASKS}
    m["floor"] = m.task.map(floor)
    ok = m.r_sw > m.floor
    excluded = m[~ok]
    m = m[ok].copy()
    m["loss"] = 100.0 - 100.0 * (m.media_placa - m.floor) / (m.r_sw - m.floor)

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    rng = np.random.default_rng(0)
    data = [m.loc[m.task == t, "loss"].to_numpy() for t, _, _ in TASKS]
    bp = ax.boxplot(data, positions=range(len(TASKS)), widths=0.45, patch_artist=True, showfliers=False,
                    whis=1.5, medianprops=dict(color=INK, lw=2.2),
                    whiskerprops=dict(color=MUTED, lw=1.2), capprops=dict(color=MUTED, lw=1.2),
                    boxprops=dict(edgecolor=BOX, lw=1.2))
    for p in bp["boxes"]:
        p.set_facecolor(BOX)
        p.set_alpha(0.30)
    for i, v in enumerate(data):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color=BOX, alpha=0.85,
                   edgecolors="white", linewidths=0.5, zorder=3)  # each dot = one network

    ax.axhline(0, color=INK, lw=1.3, zorder=1)
    ax.axhline(100, color=MUTED, lw=1, ls="--", zorder=1)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels([lab for _, lab, _ in TASKS], fontsize=12)
    ax.set_ylabel("Performance loss on ODIN (%)", fontsize=12)
    ax.set_xlim(-0.6, len(TASKS) - 0.4)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, lw=1, alpha=1)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    fig.subplots_adjust(top=0.97, bottom=0.1, left=0.1, right=0.97)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, facecolor="white")
    print("saved:", a.out)
    for (t, _, _), v in zip(TASKS, data):
        print(f"{t:13s} n={len(v):2d} median={np.median(v):6.1f}  p25={np.percentile(v,25):6.1f}  "
              f"p75={np.percentile(v,75):6.1f}  min={v.min():6.1f}  max={v.max():6.1f}")

    # caption (explanatory text lives here, not inside the figure)
    ex = "; ".join(f"{r.task.replace('_', ' ')} {r.net.split('_')[-1]} (software reward {r.r_sw:.1f}, at or below the floor)"
                   for r in excluded.itertuples())
    cap = ("**Performance loss of WANN spiking networks adapted to ODIN.** Each dot is one network "
           "(11 per task, mean of 100 episodes; boxes show median and interquartile range, whiskers extend to "
           "1.5 × IQR). Loss = 100 % − performance, with performance = 100·(R_ODIN − floor)/(R_software − floor), "
           "where R_software is the original WANN network evaluated in software (its winner shared weight) and "
           "R_ODIN is the adapted network running on the ZCU104 board. Floor: −500 (Acrobot), −200 (Mountain Car), "
           "0 (racing car). 0 % = same as the original, 100 % = task not achieved, negative = better than the original. "
           "Episodes 0–99 (Acrobot) and 100–199 (Mountain Car, racing car). Racing car: a single ODIN configuration "
           "(calibrated on seed 06) was applied to all networks"
           + (f"; excluded because the ratio is undefined: {ex}" if ex else "") + ".\n\n"
           "Median loss (n): " + "; ".join(f"{lab.replace(chr(10), ' ')} {np.median(v):.1f} % ({len(v)})"
                                          for (_, lab, _), v in zip(TASKS, data)) + ".\n")
    cap_path = a.out.with_name(a.out.stem + "_caption.md")
    cap_path.write_text(cap, encoding="utf-8")
    print("caption:", cap_path)


if __name__ == "__main__":
    main()
