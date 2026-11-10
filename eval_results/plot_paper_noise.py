#!/usr/bin/env python3
"""
Paper figures for noise types and ablations: WANN vs. SGRL (same connections / same neurons), float32 (no
quantization), on PCM and ReRAM-CMO (Falcone et al. range, 8-90 uS). Reads data only; nothing is simulated or trained.

Data (11 trained networks per method; unit of replication = network, not chip):
  * PCM and the noise-free baseline: eval_results/<task>_replicas/sweep_all.csv (computed earlier; PCM does not
    depend on the ReRAM-CMO range).
  * ReRAM-CMO (8-90 uS): eval_results/paper_noise/<task>_replicas/sweep_all.csv, produced by
    eval_results/run_paper_noise_falcone.sh. The earlier ReRAM-CMO data (1-25 uS) and ReRAM Wan2022 are NOT used.
    If the new CMO data is missing, the figures are drawn with PCM only (and the script says so).
The two sources are merged into eval_results/paper_noise/<task>_merged/ (cache) and passed through the same
per-network aggregation as stats_noise_replicas.py (WANN: one ODIN level per network chosen noise-free with the
reserved seed 999; retention = (return - floor) / (noise-free return - floor) x 100).

Figures (graficos/paper/noise/<name>.png + <name>_caption.md; English; no title/footnotes inside the image):
  noise_intensity   retention vs. noise scale (devices in rows, tasks in columns)
  noise_drift       retention vs. time since programming, noise scale 1
  noise_boxplots    distribution over networks at noise scales 1 and 3 (stars: Mann-Whitney vs. WANN, BH-FDR)
  noise_ablation    one noise mechanism at a time (programming / read / drift) at noise scale 3

Usage:  python eval_results/plot_paper_noise.py
"""
import json
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
from scipy import stats

from plot_style import TIME_TICK_LABELS, setup_publication_style
from stats_noise_replicas import network_table

OLD = _SCRIPT_DIR            # eval_results/<task>_replicas (PCM + noise-free)
NEW = _SCRIPT_DIR / "paper_noise"   # eval_results/paper_noise/<task>_replicas (ReRAM-CMO, Falcone range)
OUT = _ROOT / "graficos" / "paper" / "noise"
TASKS = [("car", "CarTrack"), ("disc_mc", "Mountain Car"), ("acrobot", "Acrobot")]
SRC = ["wann", "sgrl_con", "sgrl_neu"]
LABEL = {"wann": "WANN", "sgrl_con": "SGRL (same connections)", "sgrl_neu": "SGRL (same neurons)"}
STYLE = {"wann": dict(color="#D55E00", marker="o"), "sgrl_con": dict(color="#CC79A7", marker="s"),
         "sgrl_neu": dict(color="#4D4D4D", marker="^")}          # Okabe-Ito; blue/green are reserved for the devices
DEV_LABEL = {"pcm": "PCM", "reramcmo": "ReRAM-CMO"}
DEV_COLOR = {"pcm": "#0072B2", "reramcmo": "#009E73"}
T0 = {"pcm": 0.0, "reramcmo": 1.0}              # default evaluation time of each device (intensity / ablation for CMO)
T_ABL = {"pcm": 86400.0, "reramcmo": 1.0}
SCALES = [0.0, 1.0, 2.0, 3.0, 5.0]
MECH = {"prog": "Programming", "read": "Read", "drift": "Drift"}
Y_CAP = 150.0


# ------------------------------------------------------------------------------------------------ data
def merge(task):
    old, new = OLD / f"{task}_replicas", NEW / f"{task}_replicas"
    keep = lambda df: df[df.source.isin(SRC)]
    a = keep(pd.read_csv(old / "sweep_all.csv"))
    parts = [a[a.noise_model.isin(["none", "pcm"])]]
    have_cmo = (new / "sweep_all.csv").is_file()
    if have_cmo:
        b = keep(pd.read_csv(new / "sweep_all.csv"))
        b = b[b.noise_model == "reramcmo"]
        miss = set(a.model_id) - set(b.model_id)
        if miss:
            print(f"[{task}] WARNING: {len(miss)} networks have no ReRAM-CMO (Falcone) rows yet "
                  f"(e.g. {sorted(miss)[:2]}); the run may be incomplete.")
        parts.append(b)
    else:
        print(f"[{task}] ReRAM-CMO (Falcone) data not found in {new}: drawing PCM only.")
    df = pd.concat(parts, ignore_index=True)
    m = pd.read_csv(old / "models.csv")
    m = m[m.source.isin(SRC)]
    dst = NEW / f"{task}_merged"
    dst.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst / "sweep_all.csv", index=False)
    m.to_csv(dst / "models.csv", index=False)
    pr = old / "protocol.json"
    if pr.is_file():
        (dst / "protocol.json").write_text(pr.read_text())
    net, base, floor = network_table(dst, "best")
    devs = ["pcm"] + (["reramcmo"] if have_cmo else [])
    return dict(net=net, base=base, floor=floor, devs=devs, models=m)


class D:
    def __init__(self):
        self.t = {t: merge(t) for t, _ in TASKS}
        self.devs = [d for d in ["pcm", "reramcmo"] if all(d in v["devs"] for v in self.t.values())]

    def vec(self, task, src, dev, mech, scale, t):
        """ret_pct per network (index = replica_seed) for one cell; scale 0 of PCM = 100 by definition."""
        net = self.t[task]["net"]
        if scale == 0.0 and dev == "pcm":
            b = self.t[task]["base"]
            g = b[b.source == src].dropna(subset=["replica_seed"]).set_index("replica_seed")
            return pd.Series(100.0, index=g.index)
        g = net[(net.source == src) & (net.noise_model == dev) & (net.mechanism == mech)
                & np.isclose(net.scale, scale) & np.isclose(net.t_inference, t)]
        return g.dropna(subset=["ret_pct"]).set_index("replica_seed")["ret_pct"]


def boot_ci(v, rng, n=2000):
    v = np.asarray(v, float)
    if len(v) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n, len(v)))
    return tuple(np.percentile(v[idx].mean(axis=1), [2.5, 97.5]))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        run = min(run, p[i] * n / rank)
        q[i] = run
    return q


def star(q):
    return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"


# ------------------------------------------------------------------------------------------------ helpers
def grid(nrows, ncols=3, h=3.6, **kw):
    fig, axs = plt.subplots(nrows, ncols, figsize=(13, h * nrows + 0.8), squeeze=False, **kw)
    for j, (_, lab) in enumerate(TASKS):
        axs[0][j].set_title(lab)
    return fig, axs


def row_label(axs, devs):
    for i, d in enumerate(devs):
        axs[i][0].annotate(DEV_LABEL[d], xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90, va="center",
                           ha="center", color=DEV_COLOR[d], fontweight="bold", fontsize=13)


def legend(fig, boxes=False, y=0.0):
    if boxes:
        h = [Line2D([], [], color=STYLE[s]["color"], lw=7, alpha=0.6, label=LABEL[s]) for s in SRC]
    else:
        h = [Line2D([], [], color=STYLE[s]["color"], marker=STYLE[s]["marker"], lw=2.2, ms=7, label=LABEL[s]) for s in SRC]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, y))


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    (OUT / f"{name}_caption.md").write_text(caption.strip() + "\n", encoding="utf-8")
    print("wrote", OUT / f"{name}.png")


def n_nets(d):
    out = {}
    for t, lab in TASKS:
        net = d.t[t]["net"]
        out[lab] = {s: int(net[net.source == s].replica_seed.nunique()) for s in SRC}
    return out


COMMON = ("Retention = (return − floor)/(noise-free return − floor) × 100 for each trained network, with floors 0 / −200 / "
          "−500 (CarTrack / Mountain Car / Acrobot); a network is the unit of replication (n = 11 per method and "
          "task), each network averaged over 30 chips × 1 read. WANN: one ODIN integer level "
          "(1–7, weight = level, w_max = 7) per network, chosen noise-free with an independent seed; SGRL: float32 "
          "(not quantized) networks with the same number of connections or of neurons as the WANN network. Devices: PCM "
          "(g_max = 25 µS) and ReRAM-CMO (g_min–g_max = 8–90 µS, the range of the Falcone et al. fit assumed by aihwkit). "
          "Noise scale 1 = literature default of the device model. For ReRAM-CMO, the g_min floor (8 µS = 8.9 % of g_max) "
          "already alters the noise-free network (noise scale 0; the differential pair cannot go below g_min), so part of "
          "the retention loss or gain on that device is due to the floor and not to stochastic noise. ")


def pending_note(d):
    return "" if "reramcmo" in d.devs else (" **ReRAM-CMO is not drawn yet**: its Falcone-range sweep "
                                           "(`eval_results/run_paper_noise_falcone.sh`) has not been run.")


# ------------------------------------------------------------------------------------------------ figures
def draw_lines(ax, d, task, dev, xs, getter, rng):
    for src in SRC:
        cols = {}
        for k, x in enumerate(xs):
            v = getter(task, src, dev, x)
            for rs, val in v.items():
                cols.setdefault(rs, {})[k] = val
        if not cols:
            continue
        tab = pd.DataFrame(cols).T.reindex(columns=range(len(xs)))
        for _, row in tab.iterrows():
            ax.plot(range(len(xs)), row.to_numpy(), color=STYLE[src]["color"], lw=0.7, alpha=0.18, zorder=2)
        ci = np.array([boot_ci(tab[k].dropna().to_numpy(), rng) for k in range(len(xs))])
        ax.fill_between(range(len(xs)), ci[:, 0], ci[:, 1], color=STYLE[src]["color"], alpha=0.15, lw=0, zorder=3)
        ax.plot(range(len(xs)), tab.mean(axis=0).to_numpy(), color=STYLE[src]["color"], lw=2.4,
                marker=STYLE[src]["marker"], ms=6, zorder=4)
    ax.axhline(100, color="0.5", lw=1, ls=":")


def fig_intensity(d):
    rng = np.random.default_rng(0)
    nr = len(d.devs)
    fig, axs = grid(nr, sharex=True)
    for i, dev in enumerate(d.devs):
        for j, (t, _) in enumerate(TASKS):
            ax = axs[i][j]
            draw_lines(ax, d, t, dev, SCALES, lambda tk, s, dv, x: d.vec(tk, s, dv, "all", x, T0[dv]), rng)
            ax.set_xticks(range(len(SCALES)))
            ax.set_xticklabels([f"{s:g}" for s in SCALES])
            ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]), top=min(ax.get_ylim()[1], Y_CAP))
            if i == nr - 1:
                ax.set_xlabel("Noise scale")
    for ax in axs[:, 0]:
        ax.set_ylabel("Retained performance (%)")
    row_label(axs, d.devs)
    legend(fig, y=-0.02)
    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    cap = ("**noise_intensity** — Retained performance as a function of the noise scale (all mechanisms scaled together) "
           "for WANN and two SGRL variants, one panel per task (columns) and device (rows). Thin lines: individual "
           "trained networks; thick line and band: mean over networks and 95 % bootstrap CI. Evaluated right after "
           "programming (PCM: t = 0; ReRAM-CMO: t = 1 s). Dotted line: 100 % (noise-free). At scale 0 PCM is 100 % by "
           "construction; for ReRAM-CMO scale 0 still includes the g_min floor of the differential pair (8.9 % of "
           "g_max), so it can differ from 100 %. " + COMMON + pending_note(d))
    save(fig, "noise_intensity", cap)


def fig_drift(d):
    rng = np.random.default_rng(1)
    nr = len(d.devs)
    fig, axs = grid(nr, sharex=True)
    for i, dev in enumerate(d.devs):
        ts = [T0[dev], 3600.0, 86400.0, 31536000.0]
        for j, (t, _) in enumerate(TASKS):
            ax = axs[i][j]
            draw_lines(ax, d, t, dev, ts, lambda tk, s, dv, x: d.vec(tk, s, dv, "all", 1.0, x), rng)
            ax.set_xticks(range(len(ts)))
            ax.set_xticklabels([TIME_TICK_LABELS.get(int(x), str(x)) for x in ts])
            ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]), top=min(ax.get_ylim()[1], Y_CAP))
            if i == nr - 1:
                ax.set_xlabel("Time since programming")
    for ax in axs[:, 0]:
        ax.set_ylabel("Retained performance (%)")
    row_label(axs, d.devs)
    legend(fig, y=-0.02)
    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    cap = ("**noise_drift** — Retained performance as a function of the time elapsed since programming, at noise scale 1, "
           "for WANN and two SGRL variants (columns: tasks; rows: devices). The first point is the initial time of each "
           "device model (PCM: 0; ReRAM-CMO: 1 s), then 1 hour, 1 day and 1 year. Thin lines: individual networks; thick "
           "line and band: mean and 95 % bootstrap CI over networks. Time points are categorical (not to scale). " +
           COMMON + pending_note(d))
    save(fig, "noise_drift", cap)


def _clip(ax, x, v, top, color):
    over = v[v > top]
    if len(over):
        ax.scatter([x] * len(over), [top - 1.5] * len(over), marker="^", s=24, color=color, edgecolor="white",
                   lw=0.4, zorder=5, clip_on=False)


def box_fig(d, name, rows, cats, cell_of, row_title, xlabels, caption_head):
    """rows: list of row keys; cats: list of category keys (x axis); cell_of(task, src, row, cat) -> Series."""
    rng = np.random.default_rng(2)
    nr, n = len(rows), len(SRC)
    w = 0.8 / n
    fig, axs = grid(nr, h=3.3, sharey=True)
    tests = []
    for r, row in enumerate(rows):
        for j, (t, _) in enumerate(TASKS):
            for c, cat in enumerate(cats):
                ref = cell_of(t, "wann", row, cat).to_numpy()
                for src in SRC[1:]:
                    v = cell_of(t, src, row, cat).to_numpy()
                    if len(ref) > 1 and len(v) > 1:
                        p = stats.mannwhitneyu(ref, v, alternative="two-sided").pvalue
                        tests.append(((r, j, c, src), p))
    q = dict(zip([k for k, _ in tests], bh([p for _, p in tests]))) if tests else {}
    allv = [cell_of(t, s, row, cat).to_numpy() for t, _ in TASKS for s in SRC for row in rows for cat in cats]
    allv = [v for v in allv if len(v)]
    lo = min([0.0] + [float(v.min()) for v in allv]) - 3
    top = max([112.0] + [min(float(v.max()), Y_CAP) for v in allv]) + 12
    for r, row in enumerate(rows):
        for j, (t, _) in enumerate(TASKS):
            ax = axs[r][j]
            for c, cat in enumerate(cats):
                for i, src in enumerate(SRC):
                    v = cell_of(t, src, row, cat).to_numpy()
                    if not len(v):
                        continue
                    x = c + (i - (n - 1) / 2) * w
                    bp = ax.boxplot([v], positions=[x], widths=w * 0.82, patch_artist=True, showfliers=False,
                                    medianprops=dict(color="black", lw=1.6), whiskerprops=dict(lw=1), capprops=dict(lw=1))
                    bp["boxes"][0].set(facecolor=STYLE[src]["color"], alpha=0.45, edgecolor=STYLE[src]["color"])
                    ax.scatter(x + rng.uniform(-w * 0.28, w * 0.28, len(v)), np.minimum(v, top - 1.5), s=14,
                               color=STYLE[src]["color"], edgecolor="white", lw=0.4, zorder=4)
                    _clip(ax, x, v, top - 12, STYLE[src]["color"])
                    k = (r, j, c, src)
                    if k in q:
                        y = min(v[v <= top - 12].max() if (v <= top - 12).any() else 0, top - 12) + 3
                        ax.text(x, y, star(q[k]), ha="center", va="bottom", fontsize=10 if star(q[k]) != "ns" else 8,
                                color="0.25" if star(q[k]) != "ns" else "0.55")
            ax.set_xticks(range(len(cats)))
            ax.set_xticklabels([xlabels(c) for c in cats])
            ax.axhline(100, color="0.5", lw=1, ls=":")
            ax.set_ylim(lo, top)
        axs[r][0].annotate(row_title(row), xy=(-0.30, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="center",
                           fontweight="bold", fontsize=12)
    for ax in axs[:, 0]:
        ax.set_ylabel("Retained performance (%)")
    legend(fig, boxes=True, y=-0.02)
    fig.tight_layout(rect=(0.03, 0.05, 1, 1))
    med = []
    for t, lab in TASKS:
        for row in rows:
            for cat in cats:
                med.append(f"| {lab} | {row_title(row)} | {xlabels(cat)} | " + " | ".join(
                    f"{np.median(cell_of(t, s, row, cat)):.0f}" if len(cell_of(t, s, row, cat)) else "–" for s in SRC) + " |")
    table = ("\n\nMedian retention over networks (%):\n\n| Task | Row | Column | WANN | SGRL (same connections) | "
             "SGRL (same neurons) |\n|---|---|---|---|---|---|\n" + "\n".join(med))
    cap = (caption_head + " Box: quartiles and median over networks; points: individual networks; triangles at the top: "
           "networks above the displayed range. Stars above SGRL boxes: Mann–Whitney U test versus WANN (two-sided, "
           f"Benjamini–Hochberg correction over the {len(tests)} comparisons of this figure): * q < .05, ** q < .01, "
           "*** q < .001, ns = not significant. " + COMMON + pending_note(d) + table)
    save(fig, name, cap)


def fig_boxplots(d):
    cell = lambda t, s, sc, dev: d.vec(t, s, dev, "all", sc, T0[dev])
    box_fig(d, "noise_boxplots", [1.0, 3.0], d.devs, cell, lambda sc: f"Noise scale {sc:g}", lambda dv: DEV_LABEL[dv],
            "**noise_boxplots** — Retention of each trained network at noise scale 1 (top) and 3 (bottom), per device (x) and "
            "task (columns), right after programming (PCM: t = 0; ReRAM-CMO: t = 1 s).")


def fig_ablation(d):
    cell = lambda t, s, dev, mech: d.vec(t, s, dev, mech, 3.0, T_ABL[dev])
    box_fig(d, "noise_ablation", d.devs, list(MECH), cell, lambda dv: DEV_LABEL[dv], lambda m: MECH[m],
            "**noise_ablation** — Ablation of the noise mechanisms: only one mechanism (programming, read or drift) is active "
            "at noise scale 3 while the others are off, per device (rows) and task (columns). PCM at t = 1 day, ReRAM-CMO "
            "at t = 1 s.")


def stats_csv(d):
    rows = []
    for t, lab in TASKS:
        for dev in d.devs:
            for src in SRC:
                for sc in (1.0, 3.0):
                    v = d.vec(t, src, dev, "all", sc, T0[dev])
                    rows.append(dict(task=t, device=dev, method=src, mechanism="all", scale=sc, t=T0[dev], n=len(v),
                                     mean=v.mean(), median=v.median(), sd=v.std()))
                for mech in MECH:
                    v = d.vec(t, src, dev, mech, 3.0, T_ABL[dev])
                    rows.append(dict(task=t, device=dev, method=src, mechanism=mech, scale=3.0, t=T_ABL[dev], n=len(v),
                                     mean=v.mean(), median=v.median(), sd=v.std()))
    out = NEW / "paper_noise_summary.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


if __name__ == "__main__":
    setup_publication_style()
    NEW.mkdir(parents=True, exist_ok=True)
    d = D()
    print("devices drawn:", d.devs, "| networks per method:", n_nets(d))
    fig_intensity(d)
    fig_drift(d)
    fig_boxplots(d)
    fig_ablation(d)
    stats_csv(d)
