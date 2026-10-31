#!/usr/bin/env python3
"""
Figuras de `run_noise_compare.py --replicas`: las 11 semillas de revalidación de cada método bajo ruido de
memristores. Lee <data-dir>/{sweep_all.csv, models.csv, protocol.json} (y, si existen, los contrastes de
stats_noise_replicas.py); no simula nada.

Cada punto de dato es UNA RED ENTRENADA (retención media sobre sus chips), no un chip: lo que se dibuja y se
contrasta es la variación entre semillas de entrenamiento, que es la que permite hablar de métodos. Retención =
(retorno - piso)/(retorno sin ruido de ESA red - piso)*100 (piso: Car 0, Mountain Car -200, Acrobot -500).

--group float   WANN y los tres SGRL en float32:        <prefijo>_{intensity,boxplots,drift,ablation}.png
--group quant   WANN y los tres SGRL cuantizados (PTQ); cada media lleva su contraparte float32 punteada, y se añade
                <prefijo>q_quant_effect.png (cada red cuantizada frente a SU red float, pareado por semilla).

Figuras
  intensity   retención y retorno vs. escala de ruido (t por defecto): una línea fina por red, media gruesa con
              banda IC 95 % bootstrap sobre redes.
  boxplots    LA figura principal: distribución de las 11 redes por método y dispositivo, a escala 1 y 3. Sobre cada
              caja de SGRL: significancia frente a WANN (Mann-Whitney, BH-FDR): * q<.05, ** q<.01, *** q<.001, ns.
  drift       retención vs. tiempo desde la programación (escala 1) en PCM y ReRAM-CMO.
  ablation    un mecanismo activo a escala 3 (programación / lectura / deriva), distribución entre redes.
  quant_effect (solo quant) dos filas: retorno sin ruido float -> cuantizado, y retención a escala 3 por dispositivo.

Uso:
  python eval_results/plot_noise_replicas.py --data-dir eval_results/car_replicas --out-dir "graficos/Racing Car" --prefix car_replicas
"""
import argparse
import json
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

from plot_noise_compare import METHODS, SHORT, INTENSITY_T, MECH_TITLE
from plot_style import TIME_TICK_LABELS, setup_publication_style
from stats_noise_replicas import FLOAT_OF, network_table

GROUPS = {"float": ["wann", "sgrl", "sgrl_con", "sgrl_neu"], "quant": ["wann", "sgrlq", "sgrlq_con", "sgrlq_neu"]}
DEVICES = ("pcm", "reram", "reramcmo")
STAR = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


class Data:
    def __init__(self, data_dir: Path, group: str, wann_view: str = "best"):
        self.wann_view = wann_view
        self.net, self.base, self.floor = network_table(data_dir, wann_view)
        self.models = pd.read_csv(data_dir / "models.csv")
        self.sources = [s for s in GROUPS[group] if s in set(self.models.source)]
        self.group = group
        sp = data_dir / ("stats_replicas_vs_wann.csv" if wann_view == "best" else f"stats_replicas_vs_wann_{wann_view}.csv")
        self.stats = pd.read_csv(sp) if sp.is_file() else None
        qp = data_dir / "stats_replicas_quant.csv"
        self.qstats = pd.read_csv(qp) if qp.is_file() else None

    def label(self, src: str) -> str:
        m = self.models[self.models.source == src]
        n = int(m.replica_seed.nunique()) if "replica_seed" in m else len(m)
        if src == "wann":
            odin = "weight_value" in m and m.weight_value.notna().any()
            how = (("best ODIN level (noise-free pick) each" if odin else "best shared weight each")
                   if self.wann_view == "best" else ("mean over ODIN levels 1-7" if odin else "mean over shared weights"))
            return f"WANN ({n} seeds, {how})"
        h = str(m.hidden.iloc[0]).replace("x", "×") if "hidden" in m and pd.notna(m.hidden.iloc[0]) else ""
        what = {"": "", "con": ", same connections", "neu": ", same neurons"}[src.split("_")[1] if "_" in src else ""]
        q = f", {int(m.quant_bits.iloc[0])}-bit" if src in FLOAT_OF and pd.notna(m.quant_bits.iloc[0]) else ""
        return f"SGRL {h}{what}{q} ({n} seeds)"

    def cell(self, src, nm, mech, scale, t, col="ret_pct"):
        """Vector (ordenado por semilla) de una celda, una entrada por red."""
        g = self.net[(self.net.source == src) & (self.net.noise_model == nm) & (self.net.mechanism == mech)
                     & np.isclose(self.net.scale, scale) & np.isclose(self.net.t_inference, t)]
        return g.sort_values("replica_seed")[["model_id", "replica_seed", col]].dropna()

    def base_vec(self, src):
        g = self.base[self.base.source == src].sort_values("replica_seed")
        return g[["model_id", "replica_seed", "ret_abs"]]

    def star(self, src, nm, mech, scale, t):
        if self.stats is None or src == "wann":
            return ""
        r = self.stats[(self.stats.method == src) & (self.stats.noise_model == nm) & (self.stats.mechanism == mech)
                       & np.isclose(self.stats.scale, scale) & np.isclose(self.stats.t_inference, t)]
        if r.empty:
            return ""
        q = float(r.q_bh.iloc[0])
        for thr, s in STAR:
            if q < thr:
                return s
        return "ns"


def boot_ci(v, rng, n=2000):
    if len(v) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(v), size=(n, len(v)))
    return tuple(np.percentile(v[idx].mean(axis=1), [2.5, 97.5]))


def legend_handles(d: Data, with_ref: bool):
    h = [Line2D([], [], color=METHODS[s]["color"], lw=2.6, marker=METHODS[s]["marker"], ms=6, label=d.label(s))
         for s in d.sources]
    h.append(Line2D([], [], color="0.5", lw=0.9, alpha=0.6, label="each trained network (one line per seed)"))
    if with_ref:
        h.append(Line2D([], [], color="0.3", lw=1.2, ls="--", label="float32 counterpart (mean)"))
    return h


def save(fig, out: Path, handles, note, ncol=2):
    fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.005))
    fig.text(0.5, -0.045, note, ha="center", va="top", fontsize=10, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def draw_series(ax, d: Data, rng, xs, getter, ref=False):
    """getter(src, x) -> Series (por semilla) o None. Una línea fina por red y la media con banda entre redes."""
    for src in d.sources:
        cols = {}
        for x in xs:
            v = getter(src, x)
            if v is None or v.empty:
                continue
            for mid, val in zip(v.model_id, v.iloc[:, -1]):
                cols.setdefault(mid, {})[x] = val
        if not cols:
            continue
        tab = pd.DataFrame(cols).T.reindex(columns=xs)
        for _, row in tab.iterrows():
            ax.plot(xs, row.to_numpy(), color=METHODS[src]["color"], lw=0.8, alpha=0.22, zorder=2)
        mean = tab.mean(axis=0).to_numpy()
        ci = np.array([boot_ci(tab[x].dropna().to_numpy(), rng) for x in xs])
        ax.fill_between(xs, ci[:, 0], ci[:, 1], color=METHODS[src]["color"], alpha=0.16, linewidth=0, zorder=3)
        ax.plot(xs, mean, color=METHODS[src]["color"], lw=2.4, marker=METHODS[src]["marker"], ms=6, zorder=4)
        if ref and src in FLOAT_OF:
            fsrc = FLOAT_OF[src]
            ftab = {}
            for x in xs:
                v = getter(fsrc, x)
                if v is not None and not v.empty:
                    ftab[x] = v.iloc[:, -1].mean()
            if ftab:
                ax.plot(list(ftab), list(ftab.values()), color=METHODS[src]["color"], lw=1.4, ls="--", zorder=3)


def fig_intensity(d: Data, out: Path, note, rng):
    scales = [0.0, 1.0, 2.0, 3.0, 5.0]
    fig, axes = plt.subplots(2, 3, figsize=(16.2, 8.8), sharex=True, squeeze=False)
    refs = d.group == "quant"
    for j, dev in enumerate(DEVICES):
        t = INTENSITY_T[dev]
        for r, col in enumerate(("ret_pct", "ret_abs")):
            def getter(src, sc, dev=dev, t=t, col=col):
                if sc == 0.0:
                    if dev == "reramcmo":
                        return d.cell(src, dev, "all", 0.0, t, col)
                    v = d.base_vec(src)
                    if col == "ret_pct":
                        v = v.assign(ret_abs=100.0)
                    return v
                return d.cell(src, dev, "all", sc, t, col)
            ax = axes[r][j]
            draw_series(ax, d, rng, scales, getter, ref=refs)      # las referencias float salen de d.net, no de d.sources
            if r == 1:
                axes[r][j].set_xlabel("Noise scale (1.0 = literature default)")
        axes[0][j].set_title(f"{SHORT[dev]}\n(t = {TIME_TICK_LABELS.get(int(t), t)})", fontsize=14)
        axes[0][j].axhline(100, color="0.5", lw=1, ls=":")
    axes[0][0].set_ylabel("Retained performance (% of noise-free)")
    axes[1][0].set_ylabel("Mean episode reward")
    for ax in axes[0]:
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
    fig.suptitle("Retention across trained networks vs. noise intensity", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.11, 1, 0.96])
    save(fig, out, legend_handles(d, refs), note)


def fig_boxplots(d: Data, out: Path, note):
    scales = (1.0, 3.0)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8), sharey=True, squeeze=False)
    n = len(d.sources)
    w = 0.8 / n
    rng = np.random.default_rng(1)
    for k, sc in enumerate(scales):
        ax = axes[0][k]
        for i, src in enumerate(d.sources):
            for j, dev in enumerate(DEVICES):
                v = d.cell(src, dev, "all", sc, INTENSITY_T[dev]).ret_pct.to_numpy()
                if len(v) == 0:
                    continue
                x = j + (i - (n - 1) / 2) * w
                bp = ax.boxplot([v], positions=[x], widths=w * 0.82, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=1.6), whiskerprops=dict(lw=1), capprops=dict(lw=1))
                bp["boxes"][0].set(facecolor=METHODS[src]["color"], alpha=0.45, edgecolor=METHODS[src]["color"])
                ax.scatter(x + rng.uniform(-w * 0.28, w * 0.28, len(v)), v, s=16, color=METHODS[src]["color"],
                           edgecolor="white", linewidths=0.4, zorder=4)
                st = d.star(src, dev, "all", sc, INTENSITY_T[dev])
                if st:
                    ax.text(x, max(v.max(), 0) + 3, st, ha="center", va="bottom", fontsize=10 if st != "ns" else 8,
                            color="0.25" if st != "ns" else "0.55")
        ax.set_xticks(range(len(DEVICES)))
        ax.set_xticklabels([SHORT[x] for x in DEVICES])
        ax.axhline(100, color="0.5", lw=1, ls=":")
        ax.set_title(f"Noise scale {sc:g}", fontsize=14)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]), top=max(112, ax.get_ylim()[1]))
    axes[0][0].set_ylabel("Retained performance (% of noise-free), one point per network")
    handles = [Line2D([], [], color=METHODS[s]["color"], lw=6, alpha=0.6, label=d.label(s)) for s in d.sources]
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    save(fig, out, handles, note + " Stars: SGRL vs WANN, Mann-Whitney U (BH-FDR): * q<.05, ** q<.01, *** q<.001, ns.", ncol=2)


def fig_drift(d: Data, out: Path, note, rng):
    plans = {"pcm": [0.0, 3600.0, 86400.0, 31536000.0], "reramcmo": [1.0, 3600.0, 86400.0, 31536000.0]}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharey=True, squeeze=False)
    refs = d.group == "quant"
    for j, (dev, ts) in enumerate(plans.items()):
        xs = list(range(len(ts)))

        def getter(src, x, dev=dev, ts=ts):
            return d.cell(src, dev, "all", 1.0, ts[x])
        draw_series(axes[0][j], d, rng, xs, getter, ref=refs)
        axes[0][j].set_xticks(xs)
        axes[0][j].set_xticklabels([TIME_TICK_LABELS.get(int(t), str(t)) for t in ts])
        axes[0][j].set_xlabel("Time since programming")
        axes[0][j].set_title(f"{SHORT[dev]} (noise scale 1)", fontsize=14)
        axes[0][j].axhline(100, color="0.5", lw=1, ls=":")
        axes[0][j].set_ylim(bottom=min(0.0, axes[0][j].get_ylim()[0]))
    axes[0][0].set_ylabel("Retained performance (% of noise-free)")
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    save(fig, out, legend_handles(d, refs), note)


def fig_ablation(d: Data, out: Path, note):
    plans = {"pcm": 86400.0, "reramcmo": 1.0}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharey=True, squeeze=False)
    n = len(d.sources)
    w = 0.8 / n
    rng = np.random.default_rng(2)
    for j, (dev, t) in enumerate(plans.items()):
        ax = axes[0][j]
        for i, src in enumerate(d.sources):
            for m, mech in enumerate(MECH_TITLE):
                v = d.cell(src, dev, mech, 3.0, t).ret_pct.to_numpy()
                if len(v) == 0:
                    continue
                x = m + (i - (n - 1) / 2) * w
                bp = ax.boxplot([v], positions=[x], widths=w * 0.82, patch_artist=True, showfliers=False,
                                medianprops=dict(color="black", lw=1.6))
                bp["boxes"][0].set(facecolor=METHODS[src]["color"], alpha=0.45, edgecolor=METHODS[src]["color"])
                ax.scatter(x + rng.uniform(-w * 0.28, w * 0.28, len(v)), v, s=14, color=METHODS[src]["color"],
                           edgecolor="white", linewidths=0.4, zorder=4)
                st = d.star(src, dev, mech, 3.0, t)
                if st:
                    ax.text(x, max(v.max(), 0) + 3, st, ha="center", va="bottom", fontsize=10 if st != "ns" else 8,
                            color="0.25" if st != "ns" else "0.55")
        ax.set_xticks(range(len(MECH_TITLE)))
        ax.set_xticklabels(list(MECH_TITLE.values()))
        ax.axhline(100, color="0.5", lw=1, ls=":")
        ax.set_title(f"{SHORT[dev]} ablation, noise scale 3, t = {TIME_TICK_LABELS.get(int(t), t)}", fontsize=13)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]), top=max(112, ax.get_ylim()[1]))
    axes[0][0].set_ylabel("Retained performance (% of noise-free)")
    handles = [Line2D([], [], color=METHODS[s]["color"], lw=6, alpha=0.6, label=d.label(s)) for s in d.sources]
    fig.tight_layout(rect=[0, 0.2, 1, 1])
    save(fig, out, handles, note, ncol=2)


def fig_quant_effect(d: Data, out: Path, note):
    fams = [s for s in ("sgrlq", "sgrlq_con", "sgrlq_neu") if s in d.sources]
    if not fams:
        return print("Sin modelos cuantizados.")
    rows = [("Noise-free\nmean reward", None)] + [(f"Retention (scale 3)\n{SHORT[dev]}", dev) for dev in DEVICES]
    fig, axes = plt.subplots(len(rows), len(fams), figsize=(5.0 * len(fams), 3.1 * len(rows)), squeeze=False)
    for c, q in enumerate(fams):
        f = FLOAT_OF[q]
        ref_of = dict(zip(d.models.model_id, d.models.ref_model_id))   # red cuantizada -> su red float (misma semilla)
        for r, (lab, dev) in enumerate(rows):
            ax = axes[r][c]
            if dev is None:
                vf = d.base_vec(f).set_index("model_id")["ret_abs"]
                vq = d.base_vec(q).set_index("model_id")["ret_abs"]
            else:
                vf = d.cell(f, dev, "all", 3.0, INTENSITY_T[dev]).set_index("model_id")["ret_pct"]
                vq = d.cell(q, dev, "all", 3.0, INTENSITY_T[dev]).set_index("model_id")["ret_pct"]
            vq.index = [ref_of.get(m, m) for m in vq.index]
            both = pd.concat([vf.rename("f"), vq.rename("q")], axis=1).dropna()
            for _, row in both.iterrows():
                ax.plot([0, 1], [row.f, row.q], color=METHODS[q]["color"], lw=0.9, alpha=0.45, marker="o", ms=3)
            ax.plot([0, 1], [both.f.mean(), both.q.mean()], color=METHODS[q]["color"], lw=3, marker=METHODS[q]["marker"], ms=8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["float32", "quantized"])
            ax.set_xlim(-0.3, 1.3)
            if c == 0:
                ax.set_ylabel(lab, fontsize=10)
            if r == 0:
                ax.set_title(d.label(q).replace(f" ({len(both)} seeds)", ""), fontsize=11)
            if d.qstats is not None and dev is not None:
                s = d.qstats[(d.qstats.method == q) & (d.qstats.noise_model == dev) & (d.qstats.mechanism == "all")
                             & np.isclose(d.qstats.scale, 3.0) & np.isclose(d.qstats.t_inference, INTENSITY_T[dev])]
                if not s.empty:
                    ax.text(0.5, 0.04, f"Wilcoxon q={float(s.q_bh.iloc[0]):.3f}  ({int(s.n_worse.iloc[0])}/{int(s.n.iloc[0])} worse)",
                            transform=ax.transAxes, ha="center", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.suptitle("Effect of weight quantization: each trained network, float32 vs. quantized (paired by seed)",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.0, note, ha="center", va="bottom", fontsize=9.5, color="0.35")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--prefix", default=None)
    ap.add_argument("--group", choices=list(GROUPS), default="float")
    ap.add_argument("--wann-view", choices=["best", "mean"], default="best",
                    help="WANN: un peso por red elegido sin ruido (def.) o el promedio de sus pesos")
    ap.add_argument("--env-label", default=None)
    a = ap.parse_args()
    proto = json.loads((a.data_dir / "protocol.json").read_text()) if (a.data_dir / "protocol.json").is_file() else {}
    label = a.env_label or proto.get("task_label", a.data_dir.name)
    prefix = (a.prefix or a.data_dir.name) + ("q" if a.group == "quant" else "") + ("" if a.wann_view == "best" else f"_{a.wann_view}")
    out = Path(a.out_dir or "graficos")
    setup_publication_style()
    d = Data(a.data_dir, a.group, a.wann_view)
    if len(d.sources) < 2:
        sys.exit(f"Error: el grupo {a.group} no tiene métodos suficientes en {a.data_dir / 'models.csv'}.")
    nets = {s: int(d.models[d.models.source == s].replica_seed.nunique()) for s in d.sources}
    print(f"[{label}] grupo {a.group}: redes por método {nets}")
    rng = np.random.default_rng(0)
    note = (f"{label}. One point/line = one trained network (mean over chips); bands: 95% bootstrap CI of the mean "
            f"across networks. Retention relative to each network's own noise-free return"
            + (f" (floor {d.floor:g} = failure)." if d.floor else "."))
    fig_intensity(d, out / f"{prefix}_intensity.png", note, rng)
    fig_boxplots(d, out / f"{prefix}_boxplots.png", note)
    fig_drift(d, out / f"{prefix}_drift.png", note, rng)
    fig_ablation(d, out / f"{prefix}_ablation.png", note)
    if a.group == "quant":
        fig_quant_effect(d, out / f"{prefix}_quant_effect.png", note)


if __name__ == "__main__":
    main()
