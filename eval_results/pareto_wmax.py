#!/usr/bin/env python3
"""
Frente de Pareto consumo vs. robustez al ruido (drift) sobre las configuraciones (peso compartido × ocupación).
Lee eval_results/<tarea>_wmax_check/networks.csv de `wmax_check_all.py --weights all`. No simula nada.

Idea: cada red WANN se puede programar con cualquiera de los 6 pesos compartidos de WEIGHT_VALS, y ese peso se puede
escribir a distinta fracción de g_max (ocupación). Una configuración (peso, ocupación) tiene
  * consumo teórico  ∝ ocupación (la conductancia de cada sinapsis es ocupación × g_max; todas valen ±el mismo
                       peso, así que la conductancia total es ∝ ocupación × nº de sinapsis; se muestra relativo al 100 %);
  * robustez         = rendimiento ABSOLUTO bajo ruido, normalizado por una referencia común de la red:
                       perf = (retorno - piso) / (mejor retorno sin ruido de esa red entre sus 6 pesos - piso) × 100.
    Normalizar por la referencia de la red (y no por el retorno sin ruido de cada peso) es lo que permite comparar
    pesos entre sí: un peso malo sin ruido no parece "robusto" por conservar poco.
El frente es el conjunto de configuraciones no dominadas: ninguna otra tiene a la vez menos consumo y más perf.

Métrica de robustez (--metric), a escala de ruido 1 salvo `stress`; todas hasta --horizon (def. 1 año):
  mean    media de perf sobre la grilla de tiempos (≈ área bajo la curva en log t): qué tan bien aguanta la deriva
  end     perf a 1 año
  worst   mínimo de perf sobre la grilla (el peor instante)
  stress  perf a 1 año con escala de ruido 3 (más ruido)

Se calcula por red (frente de cada una, tenue) y agregado: media de perf sobre las 11 redes por configuración
(los 6 pesos significan lo mismo en todas las redes de una tarea), con IC95 % bootstrap sobre redes, y el frente de
esas medias. Salidas: graficos/<Tarea>/ruido/wmax_check/<tarea>_wmax_check_pareto_<metric>.png y
eval_results/<tarea>_wmax_check/pareto_<metric>_<dispositivo>.csv (todas las configuraciones, con `on_front`).

Límites: el consumo es un proxy (conductancia de la rama del peso); no cuenta la compensación de ganancia en la
neurona (se supone ideal), ni periferia. En ReRAM-CMO el piso g_min = 1 µS (4 % de g_max) hace irrepresentables las
ocupaciones ≲ 4 %: su límite bajo puede venir del piso y no del ruido.

Uso:
  python eval_results/pareto_wmax.py --task all
  python eval_results/pareto_wmax.py --task car --metric end
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

from plot_style import setup_publication_style, tagged_dir

YEAR = 31536000.0
TASKS = {"car": ("CarTrack", "Racing Car"), "disc_mc": ("Mountain Car (discrete)", "Mountain Car"),
         "acrobot": ("Acrobot", "Acrobot")}
DEVICES = {"pcm": "PCM", "reramcmo": "ReRAM CMO/HfOx"}
METRIC_TITLE = {"mean": "mean over time (0 to {h})", "end": "at {h}", "worst": "worst instant (0 to {h})",
                "stress": "at 1 year, noise scale 3"}
GREENS = ["#FDE725", "#7AD151", "#22A884", "#2A788E", "#414487", "#440154"]     # un color por peso (viridis)
MARKS = ["o", "s", "^", "D", "v", "P"]


def load(task: str, root: Path) -> pd.DataFrame:
    f = root / f"{task}_wmax_check" / "networks.csv"
    if not f.is_file():
        sys.exit(f"Error [{task}]: no existe {f}. Corré `wmax_check_all.py --weights all` primero.")
    df = pd.read_csv(f)
    if df.groupby("seed")["level"].nunique().min() < 2:
        sys.exit(f"Error [{task}]: {f} tiene un solo peso por red; el Pareto necesita `--weights all` (los 6 pesos).")
    return df


def perf_table(df: pd.DataFrame, metric: str, device: str, horizon: float) -> pd.DataFrame:
    """Una fila por (red, peso, ocupación): perf [%] de la métrica elegida."""
    ref = df[df.model != "none"].drop_duplicates(["seed", "level"]).groupby("seed").base.max()   # mejor retorno sin ruido
    d = df[(df.model == device)].copy()
    d["ref"] = d.seed.map(ref)
    d["perf"] = np.where(d.ref > d.floor, (d.ret_abs - d.floor) / (d.ref - d.floor) * 100.0, np.nan)
    key = ["seed", "level", "occ"]
    if metric == "stress":
        g = d[np.isclose(d.scale, 3.0) & np.isclose(d.t, YEAR)]
        return g.groupby(key).perf.mean().reset_index()
    d = d[np.isclose(d.scale, 1.0) & (d.t <= horizon + 1e-9)]
    if metric == "end":
        d = d[np.isclose(d.t, horizon)]
        return d.groupby(key).perf.mean().reset_index()
    agg = "mean" if metric == "mean" else "min"
    inner = d.groupby(key + ["t"]).perf.mean().reset_index()
    return inner.groupby(key).perf.agg(agg).reset_index()


def pareto_mask(cost: np.ndarray, perf: np.ndarray) -> np.ndarray:
    """No dominados: minimizar cost, maximizar perf (dominado si otro es <= en cost y >= en perf, y mejor en alguno)."""
    n = len(cost)
    keep = np.ones(n, bool)
    for i in range(n):
        dom = (cost <= cost[i]) & (perf >= perf[i]) & ((cost < cost[i]) | (perf > perf[i]))
        if dom.any():
            keep[i] = False
    return keep


def boot_ci(x: np.ndarray, rng, n=5000):
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return tuple(np.percentile(x[idx].mean(axis=1), [2.5, 97.5]))


def aggregate(pt: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for (lv, oc), g in pt.groupby(["level", "occ"]):
        x = g.perf.to_numpy()
        lo, hi = boot_ci(x, rng)
        rows.append(dict(level=lv, occ=oc, n=int((~np.isnan(x)).sum()), perf=np.nanmean(x), lo=lo, hi=hi))
    ag = pd.DataFrame(rows)
    ag["on_front"] = pareto_mask(ag.occ.to_numpy(), ag.perf.to_numpy())
    return ag.sort_values(["occ", "level"]).reset_index(drop=True)


def draw(ax, pt, ag, levels, show_nets):
    col = {lv: GREENS[round(i * (len(GREENS) - 1) / max(len(levels) - 1, 1))] for i, lv in enumerate(levels)}
    mk = {lv: MARKS[i % len(MARKS)] for i, lv in enumerate(levels)}
    if show_nets:                       # frente de cada red, tenue
        for _, g in pt.groupby("seed"):
            f = g[pareto_mask(g.occ.to_numpy(), g.perf.fillna(-1e9).to_numpy())].sort_values("occ")
            ax.step(f.occ * 100, f.perf, where="post", color="0.55", alpha=0.35, lw=0.9, zorder=1)
    for r in ag.itertuples():
        ax.errorbar(r.occ * 100, r.perf, yerr=[[r.perf - r.lo], [r.hi - r.perf]], fmt="none", ecolor=col[r.level],
                    alpha=0.35, lw=1.1, zorder=2)
        ax.scatter(r.occ * 100, r.perf, s=95 if r.on_front else 55, color=col[r.level], marker=mk[r.level],
                   edgecolor="k" if r.on_front else "none", linewidth=1.4, zorder=4 if r.on_front else 3)
    f = ag[ag.on_front].sort_values("occ")
    ax.step(f.occ * 100, f.perf, where="post", color="k", lw=2.0, zorder=3)
    ax.set_xscale("log", base=2)
    occs = sorted(ag.occ.unique())
    ax.set_xticks([o * 100 for o in occs])
    ax.set_xticklabels([f"{o * 100:g}%" for o in occs])
    ax.minorticks_off()
    ax.set_xlim(min(occs) * 100 * 0.8, 100 * 1.15)
    return col, mk


def fig_pareto(task, label, df, metric, horizon, show_nets, out, csv_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    levels = sorted(df.level.unique())
    summ = {}
    for ax, (dev, name) in zip(axes, DEVICES.items()):
        pt = perf_table(df, metric, dev, horizon)
        if pt.empty:
            ax.set_title(f"{name}: sin datos")
            continue
        ag = aggregate(pt)
        col, mk = draw(ax, pt, ag, levels, show_nets)
        ax.set_title(name, fontsize=14)
        ax.set_xlabel("Relative conductance (∝ static power)")
        ax.grid(alpha=0.3)
        summ[dev] = ag
        csv = csv_dir / f"pareto_{metric}_{dev}.csv"
        csv.parent.mkdir(parents=True, exist_ok=True)
        ag.to_csv(csv, index=False)
        f = ag[ag.on_front].sort_values("occ")
        print(f"[{task}] {name:15s} frente ({metric}): "
              + "  ".join(f"w{r.level:g}@{r.occ * 100:g}% → {r.perf:.1f}" for r in f.itertuples()))
    axes[0].set_ylabel(f"Performance under noise, {METRIC_TITLE[metric].format(h='1 year')}\n"
                       "(% of the network's best noise-free return)")
    hs = [Line2D([], [], marker=MARKS[i % len(MARKS)], ls="", color=GREENS[round(i * (len(GREENS) - 1) / max(len(levels) - 1, 1))],
                 ms=9, label=f"shared weight {lv:g}") for i, lv in enumerate(levels)]
    hs += [Line2D([], [], color="k", lw=2, label="Pareto front (non-dominated)")]
    if show_nets:
        hs += [Line2D([], [], color="0.55", lw=1, label="per-network fronts")]
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.legend(handles=hs, loc="lower center", ncol=len(hs) if len(hs) <= 4 else 4, bbox_to_anchor=(0.5, 0.045))
    n = df.seed.nunique()
    fig.text(0.5, 0.0, f"{label}. {n} WANN networks; each point = one (shared weight, occupancy) configuration, mean over networks "
             "(bars: 95% bootstrap CI over networks). Noise scale 1 unless stated; consumption is a proxy (100% = every weight written at g_max; weight-branch "
             "conductance only; neuron gain compensation and periphery not counted).",
             ha="center", va="bottom", fontsize=9, color="0.35", wrap=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"Guardado: {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=list(TASKS) + ["all"], default=["all"])
    ap.add_argument("--metric", choices=list(METRIC_TITLE), default="mean")
    ap.add_argument("--horizon", type=float, default=YEAR, help="último tiempo (s) considerado (def.: 1 año)")
    ap.add_argument("--data-root", default="eval_results")
    ap.add_argument("--out-root", default="graficos")
    ap.add_argument("--show-networks", action="store_true", help="dibuja además el frente de cada red (tenue)")
    ap.add_argument("--flat", action="store_true")
    a = ap.parse_args()
    setup_publication_style()
    for task in (list(TASKS) if "all" in a.task else a.task):
        label, gdir = TASKS[task]
        df = load(task, Path(a.data_root))
        prefix = f"{task}_wmax_check_pareto_{a.metric}"
        out = tagged_dir(Path(a.out_root) / gdir, f"{task}_wmax_check", a.flat) / f"{prefix}.png"
        fig_pareto(task, label, df, a.metric, a.horizon, a.show_networks, out, Path(a.data_root) / f"{task}_wmax_check")


if __name__ == "__main__":
    main()
