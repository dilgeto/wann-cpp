#!/usr/bin/env python3
"""
Boxplots SIN RUIDO del efecto de cuantizar los pesos a 3 bits: WANN y las tres familias de SGRL, con un punto por red
entrenada (11 semillas por método). Una figura por tarea: graficos/<Tarea>/ruido/quant_noisefree/<tarea>_quant_noisefree.png

  SGRL   float32 vs. 3 bits (PTQ esquema `odin`): retorno medio por red, de ../surrogate-gradient-rl/runs_quant/odin/
         quantization.csv. Las dos cajas de cada familia usan las MISMAS redes y los MISMOS episodios (comparación pareada);
         bajo cada par se anota cuántas de las 11 redes empeoran al cuantizar.
  WANN   (a) "peso elegido en el entrenamiento": el peso compartido de WEIGHT_VALS que el entrenamiento dejó como ganador de cada
             red (su _best.wi), medido con episodios NUEVOS en la revalidación (eval_results/*_best.csv). NO es el mejor de los 6
             sobre esos episodios: la elección del entrenamiento puede no sostenerse al re-medir (maldición del ganador).
         (b) "nivel entero 1-7": cada red con el nivel de mejor desempeño sin ruido, elegido con episodios independientes
             (semilla reservada 999) y medido con otros (celda `none` de eval_results/<tarea>_replicas). Es el equivalente
             de "cuantizado" para WANN: todas sus sinapsis valen ±nivel. Incluye niveles (4, 6, 7) que no estaban entre los 6
             pesos originales, y se elige con un criterio mejor, por eso puede superar a (a).
Los retornos son los originales del entorno (sin shaping). No simula nada: solo lee CSV ya calculados.

Uso:
  python eval_results/plot_quant_noisefree.py
"""
import argparse
import json
import os
import re
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
from matplotlib.patches import Patch

from plot_noise_compare import METHODS
from plot_style import setup_publication_style, tagged_dir
from stats_noise_replicas import network_table

TASKS = {
    "car": dict(label="CarTrack", graficos="Racing Car", eval_csvs=["eval_results/car_ttfs_first_spike_discrete/car_best.csv"]),
    "disc_mc": dict(label="Mountain Car (discrete)", graficos="Mountain Car",
                    eval_csvs=sorted(Path("eval_results").glob("disc_mc_*_discrete/mountain_car_best.csv"))),
    "acrobot": dict(label="Acrobot", graficos="Acrobot", eval_csvs=["eval_results/eval_p3_weights/acrobot_best.csv"]),
}
RUN_RE = re.compile(r"^(?P<task>.+?)-(?:p3-(?P<model>rank\d+)|f2-(?P<trial>t\d+))-s(?P<seed>\d+)$")
FAMILIES = [("sgrl", "SGRL 64×64"), ("sgrl_con", "SGRL same connections"), ("sgrl_neu", "SGRL same neurons")]


def sgrl_pairs(task: str, models: pd.DataFrame, qcsv: pd.DataFrame) -> dict:
    """{fuente: DataFrame(seed, float, q3)} con el retorno medio por red antes/después de cuantizar."""
    out = {}
    for src, _ in FAMILIES:
        rows = []
        for r in models[models.source == src].itertuples():
            ck = Path(r.path)
            m = RUN_RE.match(ck.parent.name)
            fam, mod = ck.parent.parent.name, (m["model"] or m["trial"])
            q = qcsv[(qcsv.family == fam) & (qcsv.task == m["task"]) & (qcsv.model == mod) & (qcsv.seed == int(m["seed"]))]
            f, b = q[q.bits == 32], q[q.bits == 3]
            if len(f) and len(b):
                rows.append(dict(seed=int(m["seed"]), float=float(f.return_mean.iloc[0]), q3=float(b.return_mean.iloc[0])))
        out[src] = pd.DataFrame(rows)
    return out


def wann_original(task: str, proto: dict) -> pd.Series:
    w = proto["wann"]
    d = pd.concat([pd.read_csv(c) for c in TASKS[task]["eval_csvs"]], ignore_index=True)
    if "reward_type" in d.columns and (d.reward_type == "original").any():
        d = d[d.reward_type == "original"]
    d = d[(d.run_key == w["run_key"]) & (d["rank"] == w["rank"])].sort_values("seed_idx")
    return d.set_index("seed_idx")["reward"]


def box(ax, x, v, color, hatch=None, alpha=0.5, width=0.34):
    bp = ax.boxplot([v], positions=[x], widths=width, patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.8), whiskerprops=dict(lw=1), capprops=dict(lw=1))
    bp["boxes"][0].set(facecolor=color, alpha=alpha, edgecolor=color, hatch=hatch)
    rng = np.random.default_rng(int(round(x * 100)) + 1000)
    ax.scatter(x + rng.uniform(-width * 0.3, width * 0.3, len(v)), v, s=22, color=color, edgecolor="white", linewidths=0.5, zorder=4)


def drop_pct(a: pd.Series, b: pd.Series, floor: float, min_den: float):
    """% de la distancia al piso que se pierde al pasar de `a` a `b`, por red: (a-b)/(a-floor)*100. Positivo = peor.
    Redes cuyo valor de partida está a menos de `min_den` del piso (o por debajo) quedan fuera: el cociente no tiene sentido."""
    den = a - floor
    ok = den > min_den
    return ((a - b) / den * 100.0).where(ok), int((~ok).sum())


def pct_figure(task, spec, proto, pairs, wo, wl_s, models, out, hid):
    floor = float(proto["ret_floor"])
    min_den = 0.05 * abs(floor) if floor else 25.0         # 5 % del rango (o 25 de reward en Car, piso 0)
    groups, excluded, rows = [], {}, []
    for src, name in FAMILIES:
        p = pairs[src]
        if p.empty:
            continue
        d, n_ex = drop_pct(p["float"].reset_index(drop=True), p.q3.reset_index(drop=True), floor, min_den)
        groups.append((src, name, d.dropna().to_numpy()))
        excluded[src] = n_ex
    wsel = wo.rename_axis("seed").sort_index().reset_index(drop=True)          # seed_idx 0..10 -> posición
    wlv = wl_s.reset_index(drop=True)
    dw, n_exw = drop_pct(wsel, wlv, floor, min_den)
    groups.insert(0, ("wann", "WANN", dw.dropna().to_numpy()))
    excluded["wann"] = n_exw
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    xs = {"wann": 0.0, "sgrl": 1.4, "sgrl_con": 2.4, "sgrl_neu": 3.4}
    for src, name, v in groups:
        c = "0.45" if src == "wann" else METHODS[src]["color"]
        box(ax, xs[src], v, c, hatch="\\\\" if src == "wann" else None, alpha=0.5, width=0.5)
        rows.append(dict(task=task, group=src, n=len(v), n_excluded=excluded[src], median_pct=float(np.median(v)),
                         mean_pct=float(np.mean(v)), q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)),
                         n_worse=int((v > 0).sum())))
    ax.axhline(0, color="0.4", lw=1, ls=":")
    # el eje se ajusta a las cajas y a SGRL; los puntos de WANN que quedan fuera (ganancias enormes en redes que partían
    # casi del piso) se marcan con una flecha en el borde y su valor, en vez de aplastar el resto
    vals = np.concatenate([v for _, _, v in groups])
    lo = min(v.min() for src, _, v in groups if src != "wann")
    wq1 = np.percentile(dict((s_, v) for s_, _, v in groups)["wann"], 25)
    ylo = max(vals.min(), min(lo, wq1) - 15)
    yhi = vals.max() + 8
    ax.set_ylim(ylo - 6, yhi)
    for src, _, v in groups:
        out_pts = v[v < ylo]
        if len(out_pts):
            ax.annotate(f"{len(out_pts)} network(s): " + ", ".join(f"{x:.0f}%" for x in sorted(out_pts)), (xs[src] + 0.12, ylo - 4),
                        ha="left", va="center", fontsize=9, color="0.25")
            ax.plot([xs[src]], [ylo - 4], marker="v", ms=8, color="0.25", zorder=6)
    ax.set_xticks([xs[k] for k, *_ in groups])
    labels = []
    for src, name, v in groups:
        base_lab = "WANN\n(not quantized:\nlevel vs training pick)" if src == "wann" else (
            name if src == "sgrl" else f"{name}\n{str(hid[src]).replace('x', '×')}")
        labels.append(f"{base_lab}\n{int((v > 0).sum())}/{len(v)} worse" + (f"\n({excluded[src]} excluded)" if excluded[src] else ""))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Reward lost, % of the distance to the failure floor\n(positive = worse)")
    ax.set_title(f"{spec['label']}: reward lost when weights go to 3 bits (no noise), one point per network", fontsize=12.5)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.text(0.5, 0.0, f"loss = (reward_before - reward_after) / (reward_before - floor) x 100, floor = {floor:g}. SGRL: float32 -> 3-bit, same network and episodes.\n"
             "WANN (gray, hatched): training-chosen weight -> best integer level; negative values are gains, not a quantization effect. "
             "Triangles: points below the axis.", ha="center", va="bottom", fontsize=8.5, color="0.35")
    fig.savefig(out / f"{task}_quant_noisefree_pct.png")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(Path(f"eval_results/{task}_replicas") / "quant_noisefree_pct_summary.csv", index=False)
    print(pd.DataFrame(rows).round(2).drop(columns="task").to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sg-repo", default="../surrogate-gradient-rl")
    ap.add_argument("--quant-csv", default=None, help="def.: <sg-repo>/runs_quant/odin/quantization.csv")
    ap.add_argument("--out-dir-base", default="graficos")
    ap.add_argument("--flat", action="store_true")
    a = ap.parse_args()
    qpath = Path(a.quant_csv or Path(a.sg_repo) / "runs_quant" / "odin" / "quantization.csv")
    if not qpath.is_file():
        sys.exit(f"Error: no existe {qpath}")
    qcsv = pd.read_csv(qpath)
    setup_publication_style()
    for task, spec in TASKS.items():
        D = Path(f"eval_results/{task}_replicas")
        if not (D / "sweep_all.csv").is_file():
            print(f"[{task}] sin {D}; se omite")
            continue
        models = pd.read_csv(D / "models.csv")
        proto = json.loads((D / "protocol.json").read_text())
        pairs = sgrl_pairs(task, models, qcsv)
        wo = wann_original(task, proto)
        _, base, _ = network_table(D, "best")
        wl_s = base[base.source == "wann"].set_index("replica_seed").ret_abs.sort_index()   # nivel elegido sin ruido, por red
        wl = wl_s.to_numpy()
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
        xs = {"wann": 0.0, "sgrl": 1.6, "sgrl_con": 2.8, "sgrl_neu": 4.0}
        d = 0.21
        box(ax, xs["wann"] - d, wo.to_numpy(), METHODS["wann"]["color"], alpha=0.30)
        box(ax, xs["wann"] + d, wl, METHODS["wann"]["color"], hatch="///", alpha=0.65)
        ax.text(xs["wann"], 0.015, "", transform=ax.get_xaxis_transform())
        notes = {}
        for src, name in FAMILIES:
            p = pairs[src]
            if p.empty:
                continue
            c = METHODS[src]["color"]
            box(ax, xs[src] - d, p["float"].to_numpy(), c, alpha=0.30)
            box(ax, xs[src] + d, p.q3.to_numpy(), c, hatch="///", alpha=0.65)
            notes[src] = f"{int((p.q3 < p['float']).sum())}/{len(p)} worse"
        ax.set_xticks([xs[k] for k in ("wann", "sgrl", "sgrl_con", "sgrl_neu")])
        hid = {s: models[models.source == s].hidden.iloc[0] for s, _ in FAMILIES}
        ax.set_xticklabels(["WANN"] + [(f"{n}" if s == "sgrl" else f"{n}\n{str(hid[s]).replace('x', '×')}") + f"\n{notes.get(s, '')}"
                                       for s, n in FAMILIES])
        ax.set_ylabel("Mean episode reward (original, no noise)")
        ax.set_title(f"{spec['label']}: effect of 3-bit weights without noise, one point per trained network", fontsize=13)
        handles = [Patch(facecolor="0.6", alpha=0.35, edgecolor="0.4", label="SGRL float32 / WANN weight chosen by training (one of 6, re-measured)"),
                   Patch(facecolor="0.4", alpha=0.65, edgecolor="0.3", hatch="///", label="SGRL 3-bit / WANN integer level 1-7 (best, chosen on independent episodes)")]
        fig.tight_layout(rect=[0, 0.12, 1, 1])
        fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.03))
        fig.text(0.5, 0.0, "11 networks per box, one point each. 'worse' = networks whose reward drops when quantized to 3 bits. "
                 "WANN is not quantized: its integer levels (1-7) are the equivalent.", ha="center", va="bottom", fontsize=8.5, color="0.35")
        out = tagged_dir(Path(a.out_dir_base) / spec["graficos"], f"{task}_quant_noisefree", a.flat)
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{task}_quant_noisefree.png")
        plt.close(fig)
        pct_figure(task, spec, proto, pairs, wo, wl_s, models, out, hid)
        print(f"[{task}] redes: WANN {len(wo)}/{len(wl)}, " + ", ".join(f"{s} {len(p)}" for s, p in pairs.items()) + f" -> {out / (task + '_quant_noisefree.png')}")


if __name__ == "__main__":
    main()
