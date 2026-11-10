#!/usr/bin/env python3
"""
Figuras del estudio de w_max con las 11 redes WANN (salida de wmax_check_all.py), una pareja por tarea:

  <tarea>_wmax_check_time.png        retención vs. tiempo desde la programación (escala de ruido 1; PCM y ReRAM-CMO),
                                     una línea por ocupación del rango de conductancia.
  <tarea>_wmax_check_occupancy.png   retención vs. ocupación de g_max, una línea por condición de ruido.
  <tarea>_wmax_check_tradeoff.png    ocupación mínima (y ahorro de conductancia estática) con que se conserva >= 95 / 90 %
                                     de retención, según el tiempo desde la programación.

Cada punto es la media sobre las 11 redes (la unidad de réplica es la red); la banda es el IC95 % bootstrap sobre
redes. Retención = (retorno - piso) / (retorno sin ruido - piso) × 100, medida con el nivel ODIN propio de cada red.
No simula nada: lee eval_results/<tarea>_wmax_check/summary.csv (y networks.csv para --show-networks).

Las figuras van a graficos/<Tarea>/ruido/wmax_check/ (misma convención que el resto de figuras de ruido).

Uso:
  python eval_results/plot_wmax_check.py --task all
  python eval_results/plot_wmax_check.py --task car --show-networks     # además, una línea tenue por red
"""
import argparse
import os
import textwrap
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

from plot_style import setup_publication_style, tagged_dir

YEAR = 31536000.0
TASKS = {"car": ("CarTrack", "Racing Car"), "disc_mc": ("Mountain Car (discrete)", "Mountain Car"),
         "acrobot": ("Acrobot", "Acrobot")}
SHORT = {"pcm": "PCM (g_max 25 µS)", "reramcmo": "ReRAM-CMO, assumed range\n(1-25 µS)",
         "reramcmo_falcone": "ReRAM-CMO, Falcone et al. range\n(8-90 µS)"}
DEV_ORDER = ["pcm", "reramcmo", "reramcmo_falcone"]
CMO_LABEL = {"reramcmo": "ReRAM CMO (assumed 1-25 µS)", "reramcmo_falcone": "ReRAM CMO (Falcone 8-90 µS)"}


def present(res):
    return [d for d in DEV_ORDER if d in set(res.model)]
# ocupación -> azul secuencial (más oscuro = más conductancia); se asigna por rango, así sirve para cualquier lista
BLUES = ["#C6DBEF", "#9ECAE1", "#4292C6", "#2171B5", "#08306B"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
TICKS = [(0, "0"), (1, "1 s"), (60, "1 min"), (3600, "1 h"), (86400, "1 day"), (604800, "1 week"),
         (2592000, "30 days"), (YEAR, "1 year")]


def occ_style(occs):
    occs = sorted(occs)
    cols = [BLUES[round(i * (len(BLUES) - 1) / max(len(occs) - 1, 1))] for i in range(len(occs))]
    return {o: (c, MARKERS[i % len(MARKERS)]) for i, (o, c) in enumerate(zip(occs, cols))}


def cell(res, m, s, t, o):
    r = res[(res.model == m) & np.isclose(res.scale, s) & np.isclose(res.t, t) & np.isclose(res.occ, o)]
    return None if r.empty else r.iloc[0]


def foot(label, n, levels, band="95% bootstrap CI over networks"):
    lv = ", ".join(f"{l:g}×{(levels == l).sum()}" for l in sorted(levels.unique()))
    return (f"{label}. {n} WANN networks (training seeds), each with its own best shared weight ({lv}); only the "
            f"conductance the weight is written at changes (w_max = weight / occupancy; lower occupancy = lower "
            f"conductance = lower static power). Bands: {band}.")


def ylim_for(values_lo, values_hi, pad_frac=0.05, include=None):
    """Límites Y ajustados a los datos (con margen), sin recortar nada por encima o por debajo de 100 %."""
    lo, hi = float(np.nanmin(values_lo)), float(np.nanmax(values_hi))
    if include is not None:
        lo, hi = min(lo, include), max(hi, include)
    pad = max((hi - lo) * pad_frac, 1.0)
    return lo - pad, hi + pad


def fig_time(res, label, n, levels, out, nets=None, zoom_floor=60.0):
    """Fila superior: rango completo (ajustado a medias y bandas, incluso > 100 %). Fila inferior: zoom sobre las
    curvas que se mantienen por encima de `zoom_floor` % (la zona donde se decide cuánto se puede bajar)."""
    occs = sorted(res.occ.unique())
    sty = occ_style(occs)
    devs = present(res)
    fig, axes = plt.subplots(2, len(devs), figsize=(6.2 * len(devs), 8.6), sharex="col", squeeze=False,
                             gridspec_kw=dict(height_ratios=[1, 1]))
    sub_all = res[np.isclose(res.scale, 1.0) & res.model.isin(devs)]
    for k, dev in enumerate(devs):
        sub = res[(res.model == dev) & np.isclose(res.scale, 1.0)]
        ts = sorted(sub.t.unique())
        for r_i, ax in enumerate((axes[0][k], axes[1][k])):
            for o in occs:
                rows = [cell(res, dev, 1.0, t, o) for t in ts]
                ok = [r is not None for r in rows]
                if not any(ok):
                    continue
                tt = np.array([t for t, kk in zip(ts, ok) if kk])
                rr = [r for r in rows if r is not None]
                col, mk = sty[o]
                if nets is not None:
                    for _, g in nets[(nets.model == dev) & np.isclose(nets.scale, 1.0) & np.isclose(nets.occ, o)].groupby("seed"):
                        g = g.sort_values("t")
                        ax.plot(g.t, g.ret, color=col, alpha=0.13, lw=0.8)
                ax.fill_between(tt, [r.lo for r in rr], [r.hi for r in rr], color=col, alpha=0.12, linewidth=0)
                ax.plot(tt, [r.ret for r in rr], color=col, marker=mk, lw=2.0, ms=4.5 if len(tt) > 30 else 5.5, zorder=4)
            ax.axhline(100, color="0.5", lw=1, ls=":")
            ax.set_xscale("symlog", linthresh=1.0, linscale=0.35)
            tk = [(t, l) for t, l in TICKS if ts and min(ts) <= t <= max(ts)]
            ax.set_xticks([t for t, _ in tk])
            ax.set_xticklabels([l for _, l in tk], rotation=40, ha="right")
            if ts:
                ax.set_xlim(0, max(ts) * 1.08)        # symlog: sin esto el eje reserva media figura para t < 0
            ax.minorticks_off()
            ax.grid(alpha=0.3)
            ax.tick_params(labelbottom=(r_i == 1))
    # filas con eje Y compartido entre dispositivos (comparables), ajustado a los datos
    lo, hi = ylim_for(sub_all.lo, sub_all.hi, include=100.0)
    keep = sub_all[sub_all.ret >= zoom_floor]
    if len(keep):
        zl, zh = ylim_for(keep.ret, keep.ret, pad_frac=0.12, include=None)
        zh = max(zh, float(keep.ret.max()) + 1.0)
    else:
        zl, zh = lo, hi
    for k in range(len(devs)):
        axes[0][k].set_ylim(lo, hi)
        axes[1][k].set_ylim(zl, zh)
        axes[0][k].set_title(f"{SHORT[dev := devs[k]]}\nnoise scale 1", fontsize=13)
        axes[1][k].set_xlabel("Time since programming")
        if k:
            axes[0][k].tick_params(labelleft=False)
            axes[1][k].tick_params(labelleft=False)
        axes[0][k].axhspan(zl, zh, color="0.5", alpha=0.07, zorder=0)       # región ampliada abajo
    axes[0][0].set_ylabel("Retained performance (% of noise-free)\nfull range")
    axes[1][0].set_ylabel(f"zoom (curves ≥ {zoom_floor:g}%)")
    handles = [Line2D([], [], color=sty[o][0], marker=sty[o][1], lw=2.2, ms=6,
                      label=f"weight uses {o * 100:g}% of g_max") for o in occs]
    fig.tight_layout(rect=[0, 0.1, 1, 1], h_pad=1.2)
    fig.legend(handles=handles, loc="lower center", ncol=len(occs), bbox_to_anchor=(0.5, 0.035))
    fig.text(0.5, 0.0, textwrap.fill(foot(label, n, levels), 190), ha="center", va="bottom", fontsize=9, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def fig_occupancy(res, label, n, levels, out, cmo="reramcmo"):
    occs = sorted(res.occ.unique())
    cl = CMO_LABEL[cmo].replace("ReRAM CMO", "CMO")
    conds = [(cmo, 1.0, YEAR, f"{cl}, scale 1, 1 year", "#D55E00", "o", "-"),
             (cmo, 1.0, 86400.0, f"{cl}, scale 1, 1 day", "#E69F00", "s", "-"),
             (cmo, 3.0, 1.0, f"{cl}, scale 3, 1 s", "#009E73", "^", "-"),
             (cmo, 0.0, 1.0, f"{cl}, no noise (g_min floor only)", "#56B4E9", "D", "-"),
             ("pcm", 1.0, YEAR, "PCM, scale 1, 1 year", "#0072B2", "v", "--"),
             ("pcm", 1.0, 86400.0, "PCM, scale 1, 1 day", "#7F7F7F", "x", "--")]
    xs = np.array(occs) * 100
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ylo, yhi = [], []
    for m, s, t, lab, col, mk, ls in conds:
        rows = [cell(res, m, s, t, o) for o in occs]
        if any(r is None for r in rows):
            continue
        y, lo, hi = (np.array([getattr(r, k) for r in rows]) for k in ("ret", "lo", "hi"))
        ylo.append(lo.min()); yhi.append(hi.max())
        ax.fill_between(xs, lo, hi, color=col, alpha=0.13, linewidth=0)
        ax.plot(xs, y, color=col, marker=mk, lw=2.3, ls=ls, ms=7, label=lab, zorder=4)
        ax.annotate(f"{y[0]:.0f}%", (xs[0], y[0]), textcoords="offset points", xytext=(-6, -14), ha="right",
                    fontsize=9, color=col)
    ax.axhline(100, color="0.5", lw=1, ls=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:g}%" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(xs.min() * 0.8, xs.max() * 1.1)
    if ylo:
        ax.set_ylim(*ylim_for(ylo, yhi, include=100.0))      # se ajusta a lo dibujado, incluso > 100 %
    ax.set_xlabel("Fraction of g_max used by the shared weight  (∝ synaptic conductance, i.e. static power)")
    ax.set_ylabel("Retained performance (% of noise-free)")
    ax.set_title(f"WANN-{label}: noise robustness depends on where the weight sits in the range", fontsize=14)
    ax.legend(loc="lower right", fontsize=9.5)
    fig.text(0.5, -0.01, foot(label, n, levels), ha="center", va="top", fontsize=9.5, color="0.35", wrap=True)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def fig_tradeoff(tr, label, n, levels, out):
    devs = [d for d in DEV_ORDER if d in set(tr.model)]
    fig, axes = plt.subplots(1, len(devs), figsize=(6.8 * len(devs), 5.8), sharey=True, squeeze=False)
    axes = axes[0]
    cols = {95.0: "#08306B", 90.0: "#E69F00"}
    for ax, dev in zip(axes, devs):
        sub = tr[(tr.model == dev) & np.isclose(tr.scale, 1.0)]
        for thr, col in cols.items():
            g = sub[np.isclose(sub.thr, thr)].sort_values("t")
            if g.empty:
                continue
            cap = 1.25                                    # "no alcanzada" (ni siquiera a 100 %): se dibuja arriba
            y = g.min_occ_median.replace(np.inf, cap).to_numpy() * 100
            lo = g.min_occ_q25.replace(np.inf, cap).to_numpy() * 100
            hi = g.min_occ_q75.replace(np.inf, cap).to_numpy() * 100
            ax.fill_between(g.t, lo, hi, color=col, alpha=0.14, linewidth=0)
            ax.plot(g.t, y, color=col, marker="o", ms=5, lw=2.2, label=f"keeps ≥ {thr:g}% of noise-free performance")
        ax.set_xscale("symlog", linthresh=1.0, linscale=0.35)
        ax.set_yscale("log", base=2)
        ts = sorted(sub.t.unique())
        tk = [(t, l) for t, l in TICKS if ts and min(ts) <= t <= max(ts)]
        ax.set_xticks([t for t, _ in tk])
        ax.set_xticklabels([l for _, l in tk], rotation=40, ha="right")
        if ts:
            ax.set_xlim(0, max(ts) * 1.08)
        yt = [6.25, 12.5, 25, 50, 100, 125]
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{v:g}%" if v <= 100 else "not reached" for v in yt])
        ax.minorticks_off()
        ax.axhline(100, color="0.5", lw=1, ls=":")
        ax.set_title(f"{SHORT[dev]}\nnoise scale 1", fontsize=13)
        ax.set_xlabel("Time since programming")
    axes[0].set_ylabel("Minimum fraction of g_max needed\n(median over networks; band = IQR)")
    axes[0].legend(loc="lower right", fontsize=10)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.text(0.5, -0.01, foot(label, n, levels, "interquartile range over networks") + " Static-power saving = 1 − occupancy (conductance of the weight branch only; "
             "neuron-side gain compensation and periphery not counted).", ha="center", va="top", fontsize=9.5,
             color="0.35", wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=list(TASKS) + ["all"], default=["all"])
    ap.add_argument("--data-root", default="eval_results", help="carpeta que contiene <tarea>_wmax_check/")
    ap.add_argument("--out-root", default="graficos", help="graficos/<Tarea>/ruido/wmax_check/")
    ap.add_argument("--prefix-suffix", default="wmax_check")
    ap.add_argument("--show-networks", action="store_true", help="en la figura de tiempo, una línea tenue por red")
    ap.add_argument("--flat", action="store_true")
    a = ap.parse_args()
    setup_publication_style()
    for task in (list(TASKS) if "all" in a.task else a.task):
        d = Path(a.data_root) / f"{task}_wmax_check"
        if not (d / "summary.csv").is_file():
            print(f"[{task}] no existe {d / 'summary.csv'}; corré wmax_check_all.py primero (se omite)")
            continue
        label, gdir = TASKS[task]
        res = pd.read_csv(d / "summary.csv")
        levels = pd.read_csv(d / "levels.csv")["level"]
        nets = pd.read_csv(d / "networks.csv") if a.show_networks and (d / "networks.csv").is_file() else None
        n = int(res.n.max())
        prefix = f"{task}_{a.prefix_suffix}"
        out = tagged_dir(Path(a.out_root) / gdir, prefix, a.flat)
        fig_time(res, label, n, levels, out / f"{prefix}_time.png", nets)
        for cmo in [d for d in present(res) if d != "pcm"]:
            suf = "" if cmo == "reramcmo" else "_falcone"
            fig_occupancy(res, label, n, levels, out / f"{prefix}_occupancy{suf}.png", cmo)
        if (d / "tradeoff.csv").is_file():
            fig_tradeoff(pd.read_csv(d / "tradeoff.csv"), label, n, levels, out / f"{prefix}_tradeoff.png")


if __name__ == "__main__":
    main()
