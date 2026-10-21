#!/usr/bin/env python3
"""
weight_agnosticism.py — curva reward-vs-peso compartido de TODAS las redes
originales de fase 3 (4 run_keys × 11 seeds por tarea), incluidos pesos FUERA
de la grilla de entrenamiento, para compararla con el agnosticismo de pesos del
paper original de WANN (Gaier & Ha, 2019, Tabla 1 y Fig. 4).

No entrena nada: solo llama a wann_eval_weights_<task> con --weights (evaluación
de la red guardada con una lista arbitraria de pesos compartidos).

Por qué fuera de la grilla: el paper entrena con 6 pesos fijos pero mide el
"random shared weight" con pesos continuos U(-2,2) que la red NO vio al buscar.
Evaluar solo en los 6 pesos de entrenamiento (lo que guarda eval_p3_weights.py)
es una medida dentro de muestra y sobreestima el agnosticismo.

Las seeds de evaluación (EVAL_SEED0..) son distintas de las del bootstrap/
ablation (0..99) para no reusar episodios con los que se eligió a la ganadora.

Subcomandos:
  run      barre los pesos DENSE de cada red (paralelo; escribe
           eval_results/weight_agnosticism/<task>_sweep.csv, formato largo)
  analyze  tablas (CSV + markdown) y figuras (graficos/...)

Uso:
  python eval_results/weight_agnosticism.py run --jobs 10
  python eval_results/weight_agnosticism.py analyze
"""
import argparse
import concurrent.futures
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_p3_weights import TASKS, best_config_per_run_key  # noqa: E402  (ancla CWD a la raíz)
from plot_style import setup_publication_style  # noqa: E402

OUT_DIR = Path("eval_results/weight_agnosticism")
EVAL_SEED0 = 1000
N_SEEDS = 50
# Grilla densa común a las 3 tareas. Contiene los 6 pesos de entrenamiento de
# cada tarea (acrobot 0.5..3 paso .5; car/mountain_car 0.5,1,2,3,5,8) más pesos
# intermedios y extrapolados (0.25, 3.5, 4, 6, 7, 10, ...).
DENSE = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0,
         5.0, 6.0, 7.0, 8.0, 10.0]

# run_keys originales (los 4 de eval_results/eval_p3_weights/*_best.csv).
# Las variantes posteriores (_40ms, _pop1024, _wide, ...) quedan fuera a propósito.
RUN_KEYS = {
    "acrobot":      ["acrobot_small_first_spike", "acrobot_small_rate_argmax",
                     "acrobot_ttfs_first_spike", "acrobot_ttfs_rate_argmax"],
    "mountain_car": ["mountain_car_small_first_spike", "mountain_car_small_rate_argmax",
                     "mountain_car_ttfs_first_spike", "mountain_car_ttfs_rate_argmax"],
    "car":          ["car_small_first_spike", "car_small_rate",
                     "car_ttfs_first_spike", "car_ttfs_rate"],
}
META = {  # graf: carpeta de graficos/; floor: reward de fallo total (norm = 0)
    "acrobot":      {"graf": "Acrobot",      "title": "Acrobot",                   "floor": -500.0, "ceil": 0.0},
    "mountain_car": {"graf": "Mountain Car", "title": "Mountain Car (discrete)",   "floor": -200.0, "ceil": 0.0},
    "car":          {"graf": "Racing Car",   "title": "Racing Car",                "floor": 0.0,    "ceil": 1462.0},
}
# Paper, Tabla 1 (WANN): random shared weight / tuned shared weight (media, 100 trials).
PAPER = {"Swing Up": (515, 723), "CarRacing": (375, 608), "Biped": (51, 261)}


def RK_SHORT(rk: str) -> str:
    """run_key sin el prefijo de tarea (car_/acrobot_/mountain_car_)."""
    for pre in ("mountain_car_", "acrobot_", "car_"):
        if rk.startswith(pre):
            return rk[len(pre):]
    return rk


RUNKEY_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito, orden fijo


def norm(task: str, r):
    m = META[task]
    return np.clip((np.asarray(r) - m["floor"]) / (m["ceil"] - m["floor"]), 0.0, None)


def sweep_one(task: str, model: dict, omp: int, timeout: int | None) -> pd.DataFrame | None:
    td = TASKS[task]
    seeds = ",".join(str(EVAL_SEED0 + i) for i in range(N_SEEDS))
    cmd = [td["executable"], "-f", str(model["model_path"]), "-d", td["base_config"],
           "-p", str(model["cfg_path"]), "--seeds", seeds, "--reward", td["reward"],
           "--nreps", "1", "--weights", ",".join(str(w) for w in DENSE)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {model['run_key']} seed{model['seed_idx']}", file=sys.stderr)
        return None
    if p.returncode != 0:
        print(f"  [FAIL] {model['run_key']} seed{model['seed_idx']}\n{p.stderr}", file=sys.stderr)
        return None
    wide = pd.read_csv(io.StringIO(p.stdout))
    long = wide.melt(id_vars="seed", var_name="k", value_name="reward")
    long["weight"] = long["k"].str[1:].astype(int).map(dict(enumerate(DENSE)))
    long = long.drop(columns="k")
    long.insert(0, "seed_idx", model["seed_idx"])
    long.insert(0, "rank", model["rank"])
    long.insert(0, "run_key", model["run_key"])
    long.insert(0, "task", task)
    return long


def cmd_run(a):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for task in a.tasks:
        out = OUT_DIR / f"{task}_sweep.csv"
        if out.exists() and not a.force:
            print(f"[{task}] {out} ya existe (usa --force para rehacer)")
            continue
        groups = best_config_per_run_key(TASKS[task]["screening_prefix"], set(RUN_KEYS[task]))
        models = [m for rk in RUN_KEYS[task] for m in groups.get(rk, [])]
        print(f"[{task}] {len(models)} redes × {len(DENSE)} pesos × {N_SEEDS} seeds "
              f"({len(models) * len(DENSE) * N_SEEDS} episodios)", flush=True)
        frames = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            futs = {ex.submit(sweep_one, task, m, a.omp, a.timeout): m for m in models}
            for i, f in enumerate(concurrent.futures.as_completed(futs), 1):
                df = f.result()
                if df is not None:
                    frames.append(df)
                print(f"  [{i}/{len(models)}]", end="\r", flush=True)
        print()
        pd.concat(frames, ignore_index=True).to_csv(out, index=False)
        print(f"[{task}] -> {out}")


# ── Análisis ────────────────────────────────────────────────────────────────
def zones(task: str):
    g = TASKS[task]["weight_vals"]
    lo, hi = min(g), max(g)
    inside = [w for w in DENSE if lo <= w <= hi and w not in g]   # interpolación
    outside = [w for w in DENSE if (w < lo or w > hi)]            # extrapolación
    return g, inside, outside


def per_net(task: str, df: pd.DataFrame) -> pd.DataFrame:
    g, inside, outside = zones(task)
    rows = []
    for (rk, si), d in df.groupby(["run_key", "seed_idx"]):
        prof = d.groupby("weight").reward.mean()
        n = pd.Series(norm(task, prof.values), index=prof.index)
        best = n.max()
        ratio = lambda ws: (n[ws].mean() / best) if best > 0 else np.nan  # noqa: E731
        rows.append({
            "task": task, "run_key": rk, "seed_idx": si,
            "best_weight": n.idxmax(), "best_reward": prof.max(),
            "best_norm": best,
            "ratio_grid": ratio(g),                 # dentro de muestra (6 pesos de entrenamiento)
            "ratio_interp": ratio(inside),          # fuera de grilla pero dentro del rango entrenado
            "ratio_range": ratio([w for w in DENSE if min(g) <= w <= max(g)]),
            "ratio_all": ratio(DENSE),              # todo, incluida extrapolación
            "frac_ok_range": float((n[[w for w in DENSE if min(g) <= w <= max(g)]] >= 0.5 * best).mean())
                              if best > 0 else np.nan,
            "frac_ok_all": float((n >= 0.5 * best).mean()) if best > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def plot_task(task: str, df: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_publication_style()
    g, _, _ = zones(task)
    rks = RUN_KEYS[task]
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2), sharey=True, sharex=True)
    for ax, rk, col in zip(axes, rks, RUNKEY_COLORS):
        d = df[df.run_key == rk]
        prof = d.groupby(["seed_idx", "weight"]).reward.mean().unstack("weight")
        for _, row in prof.iterrows():
            ax.plot(row.index, row.values, color=col, alpha=0.22, lw=1.0)
        med = prof.median()
        ax.plot(med.index, med.values, color=col, lw=2.4, marker="o", ms=4, label="median over 11 nets")
        ax.plot(g, [0.015] * len(g), "|", color="0.25", ms=9, mew=1.2,
                transform=ax.get_xaxis_transform())
        ax.axvspan(min(g), max(g), color="0.5", alpha=0.07, lw=0)
        ax.set_xscale("log")
        ax.set_xticks([0.25, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
        ax.set_title(RK_SHORT(rk).replace("_", " "), fontsize=12)
        ax.set_xlabel("shared weight")
    axes[0].set_ylabel("mean reward (50 episodes)")
    fig.suptitle(f"{META[task]['title']}: reward vs shared weight, all 44 original networks "
                 f"(shaded = trained weight range, ticks = the 6 training weights)",
                 fontsize=13, fontweight="bold", y=1.04)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def plot_vs_paper(nets: pd.DataFrame, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_publication_style()
    labels = [("acrobot", "Acrobot"), ("mountain_car", "Mountain Car\n(discrete)"), ("car", "Racing Car")]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    rng = np.random.default_rng(0)
    for x, (t, lab) in enumerate(labels):
        for key, col, off, name in (("ratio_grid", "#999999", -0.17, "in-sample (6 training weights)"),
                                    ("ratio_interp", "#0072B2", 0.17, "off-grid, inside trained range")):
            v = nets[nets.task == t][key].dropna().values
            ax.scatter(x + off + rng.uniform(-0.07, 0.07, len(v)), v, s=14, color=col, alpha=0.5, lw=0,
                       label=name if x == 0 else None)
            ax.hlines(np.median(v), x + off - 0.13, x + off + 0.13, color=col, lw=2.6)
    px = 3.4
    for i, (name, (rnd, tuned)) in enumerate(PAPER.items()):
        ax.scatter(px + i * 0.6, rnd / tuned, marker="D", s=60, color="#D55E00",
                   label="paper WANN (random shared / tuned shared)" if i == 0 else None)
        ax.annotate(f"{rnd / tuned:.2f}", (px + i * 0.6, rnd / tuned), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=10, color="0.2")
    ax.set_xticks(list(range(3)) + [px + i * 0.6 for i in range(3)])
    ax.set_xticklabels([l for _, l in labels] + list(PAPER.keys()))
    ax.axvline(2.9, color="0.8", lw=1)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("mean over weights / best weight")
    ax.set_title("Weight agnosticism: this work vs Gaier & Ha (2019)")
    ax.legend(loc="upper right", fontsize=9)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def cmd_analyze(a):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    allnets = []
    for task in a.tasks:
        f = OUT_DIR / f"{task}_sweep.csv"
        if not f.exists():
            print(f"[{task}] falta {f}; corre `run` primero", file=sys.stderr)
            continue
        df = pd.read_csv(f)
        nets = per_net(task, df)
        allnets.append(nets)
        plot_task(task, df, Path("graficos") / META[task]["graf"] / f"{task}_weight_agnosticism.png")
        prof = df.groupby("weight").reward.mean()
        print(f"\n== {task}  ({len(nets)} redes)")
        print(nets.groupby("run_key")[["ratio_grid", "ratio_interp", "ratio_range", "ratio_all",
                                       "frac_ok_range"]].median().round(2).to_string())
        print("TOTAL mediana:", nets[["ratio_grid", "ratio_interp", "ratio_range", "ratio_all",
                                      "frac_ok_range", "frac_ok_all"]].median().round(2).to_dict())
        print("best_weight (conteo):", nets.best_weight.value_counts().sort_index().to_dict())
        print("reward medio por peso (todas las redes):", prof.round(1).to_dict())
    if allnets:
        nets = pd.concat(allnets, ignore_index=True)
        nets.to_csv(OUT_DIR / "per_network_summary.csv", index=False)
        summ = nets.groupby("task")[["ratio_grid", "ratio_interp", "ratio_range", "ratio_all",
                                     "frac_ok_range", "frac_ok_all"]].median().round(3)
        summ.to_csv(OUT_DIR / "task_summary.csv")
        plot_vs_paper(nets, Path("graficos") / "weight_agnosticism_vs_paper.png")
        print("\n", summ.to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("run", cmd_run), ("analyze", cmd_analyze)):
        p = sub.add_parser(name)
        p.add_argument("--tasks", nargs="+", default=list(RUN_KEYS), choices=list(RUN_KEYS))
        p.set_defaults(fn=fn)
        if name == "run":
            p.add_argument("--jobs", type=int, default=8)
            p.add_argument("--omp", type=int, default=1)
            p.add_argument("--timeout", type=int, default=1800)
            p.add_argument("--force", action="store_true")
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
