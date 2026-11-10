#!/usr/bin/env python3
"""
Publication figures for the conductance-occupancy study (w_max), built from eval_results/wmax_study/<task>_wmax_check/
(summary.csv, networks.csv, pareto_mean_<device>.csv; produced by wmax_check_all.py / pareto_wmax.py). Reads data only:
nothing is simulated or trained.

Occupancy = fraction of g_max at which the shared weight is written (w_max = weight / occupancy). Lower occupancy means
lower synaptic conductance, i.e. lower static power (proxy). Retention = (return - floor) / (noise-free return - floor)
x 100, per network; the replicate unit is the WANN network (n = 11).

Devices: PCM (g_max 25 µS) and ReRAM-CMO with the Falcone et al. range (8-90 µS) only.
Figures (graficos/paper/occupancy/<name>.png + <name>_caption.md; English, no title/footnotes inside the image):
  retention_vs_occupancy      retention at 1 year vs occupancy, one panel per task, one line per device model
  networks_above_threshold    fraction of networks keeping >= 90 % of their noise-free return, vs occupancy
  retention_heatmap           retention over time x occupancy (devices in rows, tasks in columns)
  per_network_distribution    every network's retention at 1 year, vs occupancy (devices in rows, tasks in columns)
  weight_occupancy_map        Pareto-style map: performance for each (shared weight, occupancy) configuration

Usage:  python eval_results/plot_wmax_publication.py            (from anywhere)
"""
import os
import sys
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
from matplotlib.ticker import FixedLocator, FixedFormatter

from plot_style import setup_publication_style

DATA = _SCRIPT_DIR / "wmax_study"
OUT = _ROOT / "graficos" / "paper" / "occupancy"
YEAR = 31536000.0
TASKS = [("car", "CarTrack"), ("disc_mc", "Mountain Car"), ("acrobot", "Acrobot")]
DEVICES = ["pcm", "reramcmo_falcone"]
# Conductance ranges (PCM 25 µS; ReRAM-CMO 8-90 µS) are given in the captions, not in the figure.
DEV_LABEL = {"pcm": "PCM", "reramcmo": "ReRAM-CMO (1–25 µS)", "reramcmo_falcone": "ReRAM-CMO"}
DEV_COLOR = {"pcm": "#0072B2", "reramcmo": "#009E73", "reramcmo_falcone": "#009E73"}   # Okabe-Ito (project colour of ReRAM-CMO)
DEV_MARK = {"pcm": "o", "reramcmo": "s", "reramcmo_falcone": "^"}
TASK_COLOR = {"car": "#D55E00", "disc_mc": "#56B4E9", "acrobot": "#CC79A7"}
TASK_LABEL = dict(TASKS)
OCC = [0.0625, 0.125, 0.25, 0.5, 1.0]
OCC_LABEL = ["6.25", "12.5", "25", "50", "100"]
XLAB = "Occupancy of $g_{max}$ (%, ∝ static power)"
RLAB = "Retained performance (% of noise-free)"
TIME_TICKS = [(1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"), (2592000, "30 days"),
              (YEAR, "1 year")]


def load():
    S, N = {}, {}
    for t, _ in TASKS:
        d = DATA / f"{t}_wmax_check"
        S[t] = pd.read_csv(d / "summary.csv")
        N[t] = pd.read_csv(d / "networks.csv")
    return S, N


def sel(df, model, scale, t):
    return df[(df.model == model) & np.isclose(df.scale, scale) & np.isclose(df["t"], t)]


def occ_axis(ax):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(OCC))
    ax.xaxis.set_major_formatter(FixedFormatter(OCC_LABEL))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlim(0.0625 / 1.15, 1.15)


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    (OUT / f"{name}_caption.md").write_text(caption.strip() + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.png")


def device_legend(fig, loc="lower center", ncol=2, extra=None, y=-0.02):
    h = [Line2D([0], [0], color=DEV_COLOR[d], marker=DEV_MARK[d], lw=2, ms=7, label=DEV_LABEL[d]) for d in DEVICES]
    if extra:
        h += extra
    fig.legend(handles=h, loc=loc, ncol=ncol, frameon=False, bbox_to_anchor=(0.5, y))


# ---------------------------------------------------------------- 1. retention vs occupancy
def fig_retention(S, N):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    nets = {}
    for ax, (t, lab) in zip(axs, TASKS):
        for d in DEVICES:
            r = sel(S[t], d, 1.0, YEAR).sort_values("occ")
            ax.fill_between(r.occ, r.lo, r.hi, color=DEV_COLOR[d], alpha=0.15, lw=0)
            ax.plot(r.occ, r.ret, color=DEV_COLOR[d], marker=DEV_MARK[d], lw=2, ms=7)
        ax.axhline(100, color="0.5", lw=1, ls=":")
        ax.axhline(90, color="0.5", lw=1, ls="--")
        occ_axis(ax)
        ax.set_xlabel(XLAB)
        ax.set_title(lab)
    axs[0].set_ylabel(RLAB)
    axs[0].set_ylim(-15, 145)
    device_legend(fig, y=-0.06)
    fig.tight_layout()
    rows = []
    for t, lab in TASKS:
        for d in DEVICES:
            r = sel(S[t], d, 1.0, YEAR).set_index("occ")
            rows.append(f"{lab}/{DEV_LABEL[d]}: {r.ret[0.0625]:.0f}% at 6.25% vs {r.ret[1.0]:.0f}% at 100%")
    cap = ("**retention_vs_occupancy** — Performance retained 1 year after programming (noise scale 1) as a function of the "
           "fraction of g_max used by the shared weight, one panel per task and one line per device model. Mean over WANN "
           "networks (n = 11; CarTrack n = 10 because network 2 has a negative noise-free return, for which retention is "
           "undefined), band = 95 % bootstrap CI over networks. Dotted line: 100 % (noise-free); dashed line: 90 %. "
           "Retention = (return − floor)/(noise-free return − floor); floors 0 / −200 / −500 (CarTrack / Mountain Car / "
           "Acrobot); each network uses its own best shared weight. Values above 100 % mean the noisy network scored higher "
           "than its own noise-free evaluation. PCM: g_max = 25 µS; ReRAM-CMO: g_min–g_max = 8–90 µS "
           "(range of the Falcone et al. fit assumed by aihwkit); its g_min floor is 8.9 % of g_max, so occupancies below "
           "that are not representable and the low-occupancy collapse of ReRAM-CMO partly reflects this floor. Static power is proxied by weight-branch conductance only (neuron gain compensation and "
           "periphery not counted).\n\nValues (6.25 % vs 100 %): " + "; ".join(rows) + ".")
    save(fig, "retention_vs_occupancy", cap)


# ---------------------------------------------------------------- 2. fraction of networks above threshold
def fig_fraction(S, N, thr=90.0):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    rows = []
    for ax, (t, lab) in zip(axs, TASKS):
        for d in DEVICES:
            r = sel(N[t], d, 1.0, YEAR).dropna(subset=["ret"])
            g = r.groupby("occ").ret.agg(lambda x: (x >= thr).mean() * 100)
            ax.plot(g.index, g.values, color=DEV_COLOR[d], marker=DEV_MARK[d], lw=2, ms=7)
            rows.append(f"{lab}/{DEV_LABEL[d]}: " + ", ".join(f"{o*100:g}%→{v:.0f}%" for o, v in g.items()))
        occ_axis(ax)
        ax.set_xlabel(XLAB)
        ax.set_title(lab)
        ax.set_ylim(-4, 104)
    axs[0].set_ylabel(f"Networks retaining ≥ {thr:g} % of\nnoise-free return (% of networks)")
    device_legend(fig, y=-0.06)
    fig.tight_layout()
    cap = (f"**networks_above_threshold** — Share of WANN networks that keep at least {thr:g} % of their noise-free return "
           "1 year after programming (noise scale 1), vs. fraction of g_max used by the shared weight. Unlike the mean "
           "retention, this exposes how many individual networks survive, not just the average. n = 11 networks per task "
           "(CarTrack: 10 with defined retention). Per-network best shared weight.\n\n" + "\n".join(f"- {r}" for r in rows))
    save(fig, "networks_above_threshold", cap)


# ---------------------------------------------------------------- 3. heatmap time x occupancy
def fig_heatmap(S, N):
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 5.8), sharex=True, sharey=True)
    cmap = plt.get_cmap("Blues").copy()
    im = None
    for i, d in enumerate(DEVICES):
        for j, (t, lab) in enumerate(TASKS):
            ax = axs[i, j]
            r = S[t][(S[t].model == d) & np.isclose(S[t].scale, 1.0) & (S[t]["t"] > 0)]
            piv = r.pivot_table(index="occ", columns="t", values="ret").sort_index()
            ts = piv.columns.values
            edges = np.log10(ts)
            e = np.concatenate([[edges[0] - (edges[1] - edges[0]) / 2], (edges[1:] + edges[:-1]) / 2,
                                [edges[-1] + (edges[-1] - edges[-2]) / 2]])
            im = ax.pcolormesh(e, np.arange(len(OCC) + 1), piv.values, cmap=cmap, vmin=0, vmax=100, shading="flat")
            ax.set_yticks(np.arange(len(OCC)) + 0.5)
            ax.set_yticklabels(OCC_LABEL)
            ax.grid(False)
            if i == 0:
                ax.set_title(lab)
            if j == 0:
                ax.set_ylabel(DEV_LABEL[d], color=DEV_COLOR[d], fontweight="bold", fontsize=11)
            if i == len(DEVICES) - 1:
                ax.set_xticks([np.log10(x) for x, _ in TIME_TICKS])
                ax.set_xticklabels([l for _, l in TIME_TICKS], rotation=35, ha="right")
                ax.set_xlabel("Time since programming")
    fig.tight_layout(rect=(0.02, 0, 0.93, 1))
    fig.text(0.008, 0.5, "Occupancy of $g_{max}$ (%)", rotation=90, va="center", fontsize=13)
    cax = fig.add_axes([0.945, 0.17, 0.015, 0.66])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(RLAB + "\n(clipped at 100)")
    cap = ("**retention_heatmap** — Mean retained performance (noise scale 1) over time since programming (x, log scale, 1 s "
           "to 1 year; 55 time points) and conductance occupancy (y), for each device model (rows) and task (columns). "
           "Mean over WANN networks (n = 11; CarTrack n = 10). Darker = more performance retained; values above 100 % are "
           "clipped to the darkest shade. Lower rows use less conductance. It separates the two failure modes: drift "
           "(left-to-right degradation at low occupancy in ReRAM-CMO) versus a static penalty already present at t = 1 s "
           "(Acrobot, PCM).")
    save(fig, "retention_heatmap", cap)


# ---------------------------------------------------------------- 5. per-network distribution
def fig_pernet(S, N):
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.4), sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for i, d in enumerate(DEVICES):
        for j, (t, lab) in enumerate(TASKS):
            ax = axs[i, j]
            r = sel(N[t], d, 1.0, YEAR).dropna(subset=["ret"])
            for k, o in enumerate(OCC):
                v = r[np.isclose(r.occ, o)].ret.values
                x = np.full(len(v), k) + rng.uniform(-0.18, 0.18, len(v))
                ax.scatter(x, np.clip(v, -20, 200), s=26, color=DEV_COLOR[d], alpha=0.7, edgecolors="white", lw=0.4)
                ax.hlines(np.median(v), k - 0.3, k + 0.3, color="0.15", lw=2)
            ax.axhline(100, color="0.5", lw=1, ls=":")
            ax.axhline(90, color="0.5", lw=1, ls="--")
            ax.set_xticks(range(len(OCC)))
            ax.set_xticklabels(OCC_LABEL)
            if i == 0:
                ax.set_title(lab)
            if i == len(DEVICES) - 1:
                ax.set_xlabel(XLAB)
            if j == 0:
                ax.set_ylabel(DEV_LABEL[d], color=DEV_COLOR[d], fontweight="bold", fontsize=11)
    axs[0, 0].set_ylim(-25, 205)
    fig.tight_layout(rect=(0.02, 0, 1, 1))
    fig.text(0.008, 0.5, RLAB, rotation=90, va="center", fontsize=13)
    cap = ("**per_network_distribution** — Retention of each individual WANN network (dot; n = 11, CarTrack n = 10) 1 year "
           "after programming, noise scale 1, for each occupancy (x), device model (rows) and task (columns). Black bar = "
           "median over networks; dotted/dashed lines = 100 % / 90 %. Values are clipped to [−20, 200] % for display. "
           "Shows inter-network heterogeneity hidden by the mean.")
    save(fig, "per_network_distribution", cap)


# ---------------------------------------------------------------- 6. weight x occupancy map
def fig_weightmap(S, N):
    devs = ["pcm", "reramcmo_falcone"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.6), sharex=True)
    im = None
    for i, d in enumerate(devs):
        for j, (t, lab) in enumerate(TASKS):
            ax = axs[i, j]
            p = pd.read_csv(DATA / f"{t}_wmax_check" / f"pareto_mean_{d}.csv")
            piv = p.pivot_table(index="level", columns="occ", values="perf").sort_index()
            fr = p.pivot_table(index="level", columns="occ", values="on_front", aggfunc="max").reindex_like(piv)
            im = ax.imshow(piv.values, cmap="Blues", vmin=0, vmax=100, aspect="auto", origin="lower")
            ax.grid(False)
            for a, lv in enumerate(piv.index):
                for b, o in enumerate(piv.columns):
                    if fr.loc[lv, o]:
                        ax.plot(b, a, marker="*", color="#D55E00", ms=15, mec="white", mew=0.8)
            ax.set_xticks(range(len(piv.columns)))
            ax.set_xticklabels([f"{o*100:g}" for o in piv.columns])
            ax.set_yticks(range(len(piv.index)))
            ax.set_yticklabels([f"{v:g}" for v in piv.index])
            if i == 0:
                ax.set_title(lab)
            if i == len(devs) - 1:
                ax.set_xlabel(XLAB)
            if j == 0:
                ax.set_ylabel(f"{DEV_LABEL[d]}\nShared weight", color=DEV_COLOR[d], fontweight="bold", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Performance under noise, mean over 0–1 year\n(% of the network's best noise-free return)")
    cap = ("**weight_occupancy_map** — Performance under PCM / ReRAM-CMO (8–90 µS) drift, averaged over time (0 to 1 year, "
           "noise scale 1) and over the 11 networks, for every (shared weight, occupancy) configuration. Each network is "
           "evaluated with all six shared weights of WEIGHT_VALS, which is the weight-agnostic degree of freedom: the same "
           "topology can be programmed with any of them. Performance is normalised by the network's best noise-free return "
           "across its six weights, so weights are comparable. Stars: non-dominated configurations (no other configuration "
           "has both less conductance and higher performance). Acrobot evaluates shared weights 0.5–3 in its own grid.")
    save(fig, "weight_occupancy_map", cap)


if __name__ == "__main__":
    setup_publication_style()
    S, N = load()
    for f in (fig_retention, fig_fraction, fig_heatmap, fig_pernet, fig_weightmap):
        f(S, N)
