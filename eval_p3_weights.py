#!/usr/bin/env python3
"""
Evalúa cada modelo resultante de fase 3 (Acrobot, Mountain Car discreto,
Racing Car) con N seeds distintas (default 10); en cada seed se prueban los
6 pesos compartidos de esa tarea, y luego se promedia cada peso a través de
las seeds. Acrobot y Mountain Car discreto se evalúan con la recompensa
original (sin shaping); Car se evalúa con la recompensa shaped. Ver --reward
para forzar un modo distinto.

Requiere los binarios wann_eval_weights_{acrobot,car,disc_mc} compilados:
  cd build && ninja wann_eval_weights_acrobot wann_eval_weights_car wann_eval_weights_disc_mc

Tareas (--task): acrobot | mountain_car | car
  "mountain_car" aquí es Mountain Car discreto (internamente sigue siendo la
  tarea "disc_mc" del código — ver comentario junto a TASKS más abajo).

Uso:
  python eval_p3_weights.py                       # las 3 tareas, todos los run_keys encontrados
  python eval_p3_weights.py --task acrobot         # solo acrobot
  python eval_p3_weights.py --task mountain_car    # solo Mountain Car discreto
  python eval_p3_weights.py --seeds 10 --jobs 8
"""
import argparse
import concurrent.futures
import io
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

# Claves públicas del script (lo que el usuario pide con --task). "mountain_car"
# es la tarea que en el código sigue llamándose "disc_mc" (Mountain Car
# discreto, Gymnasium MountainCar-v0): su ejecutable, config y directorios
# screening_full/log siguen usando el nombre "disc_mc" porque el código fuente
# no fue renombrado — solo la etiqueta con la que nos referimos a ella cambió.
# No confundir con el otro task "mountain_car" del código (rl-tools
# MountainCarContinuous-v0, ya evaluado en screening_full/mountain_car_*),
# que este script no toca.
TASKS: dict[str, dict] = {
    "acrobot": {
        "executable":       "./build/wann_eval_weights_acrobot",
        "base_config":       "p/acrobot_snn.json",
        "weight_vals":       [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        "screening_prefix":  "acrobot",
        "reward":            "original",
    },
    "mountain_car": {
        "executable":       "./build/wann_eval_weights_disc_mc",
        "base_config":       "p/disc_mc_snn.json",
        "weight_vals":       [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "screening_prefix":  "mountain_car",
        "reward":            "original",
    },
    "car": {
        "executable":       "./build/wann_eval_weights_car",
        "base_config":       "p/car_snn.json",
        "weight_vals":       [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "screening_prefix":  "car",
        "reward":            "shaped",
    },
}

MODEL_RE = re.compile(r"^rank(\d+)_seed(\d+)$")


def find_models(prefix: str) -> list[dict]:
    """Locate every phase-3 model under screening_full/<prefix>_*/p3_configs."""
    cfg_dirs = sorted(Path("screening_full").glob(f"{prefix}_*/p3_configs"))
    cfg_dirs += sorted(Path("screening_full").glob(f"{prefix}/p3_configs"))

    models = []
    for cfg_dir in cfg_dirs:
        run_key = cfg_dir.parent.name
        for cfg_path in sorted(cfg_dir.glob("rank*_seed*.json")):
            m = MODEL_RE.match(cfg_path.stem)
            if not m:
                continue
            model_path = Path("log") / f"full_p3_{run_key}" / f"{cfg_path.stem}_best.out"
            if not model_path.exists():
                print(f"  [SKIP] modelo no encontrado: {model_path}", file=sys.stderr)
                continue
            models.append({
                "run_key":    run_key,
                "rank":       int(m.group(1)),
                "seed_idx":   int(m.group(2)),
                "model_path": model_path,
                "cfg_path":   cfg_path,
            })
    return models


def model_size(model_path: Path) -> tuple[int, int]:
    """Return (n_neurons, n_connections) from a network file exported by
    Ind::exportNet: N rows of N weight columns + 1 activation column;
    a weight is "nan" or 0.0 when there's no connection (same convention as
    Ind::importNet)."""
    with open(model_path) as f:
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    n_neurons = len(rows)
    n_connections = 0
    for row in rows:
        for tok in row[:n_neurons]:
            if tok == "nan":
                continue
            if float(tok) != 0.0:
                n_connections += 1
    return n_neurons, n_connections


def eval_model(task: str, td: dict, model: dict, seeds: list[int],
               omp: int, timeout: int | None) -> tuple[dict, pd.DataFrame] | tuple[None, None]:
    """Run wann_eval_weights_{task} for one model. Returns (row, raw) where
    row has the per-weight mean/std across `seeds` and raw is the underlying
    per-seed CSV (columns: seed, w0..w{N-1}) so the winning (model, weight)
    can later report its individual per-seed evaluations."""
    seeds_arg = ",".join(str(s) for s in seeds)
    cmd = [td["executable"],
           "-f", str(model["model_path"]),
           "-d", td["base_config"],
           "-p", str(model["cfg_path"]),
           "--seeds", seeds_arg,
           "--reward", td["reward"]]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {model['run_key']}/{model['cfg_path'].stem}", file=sys.stderr)
        return None, None

    if proc.returncode != 0:
        print(f"  [FAIL] {model['run_key']}/{model['cfg_path'].stem}\n{proc.stderr}",
              file=sys.stderr)
        return None, None

    raw = pd.read_csv(io.StringIO(proc.stdout))
    w_cols = [c for c in raw.columns if c != "seed"]
    means = raw[w_cols].mean()
    stds  = raw[w_cols].std().fillna(0.0)  # std is NaN when only 1 seed was evaluated

    n_neurons, n_connections = model_size(model["model_path"])

    row = {
        "task":        task,
        "reward":      td["reward"],
        "run_key":     model["run_key"],
        "rank":        model["rank"],
        "seed_idx":    model["seed_idx"],
        "n_seeds_eval": len(raw),
        "n_neurons":     n_neurons,
        "n_connections": n_connections,
    }
    for i, wc in enumerate(w_cols):
        wval = td["weight_vals"][i]
        row[f"w{i}_{wval:g}_mean"] = means[wc]
        row[f"w{i}_{wval:g}_std"]  = stds[wc]
    row["mean_across_weights"] = means.mean()

    # Hiperparámetros con los que se optimizó este modelo (screening_full/*/p3_configs),
    # para poder promediar por configuración en el *_summary.csv.
    try:
        hp = json.loads(model["cfg_path"].read_text())
    except Exception:
        hp = {}
    for k, v in hp.items():
        if k == "save_mod":
            continue
        row[f"hp_{k}"] = v

    return row, raw


def run_task(task: str, seeds: list[int], jobs: int, omp: int,
             timeout: int | None, out_dir: Path,
             reward_override: str | None = None) -> tuple[pd.DataFrame, list[dict]]:
    td = dict(TASKS[task])
    if reward_override:
        td["reward"] = reward_override
    if not Path(td["executable"]).exists():
        print(f"ERROR: {td['executable']} no existe. Compilar primero "
              f"(cd build && ninja {Path(td['executable']).name}).", file=sys.stderr)
        return pd.DataFrame(), []

    prefix = td["screening_prefix"]
    models = find_models(prefix)
    if not models:
        print(f"[{task}] no se encontraron modelos de fase 3 en "
              f"screening_full/{prefix}_*/p3_configs.")
        return pd.DataFrame(), []

    print(f"\n[{task}] {len(models)} modelos encontrados, "
          f"evaluando con {len(seeds)} seeds cada uno (reward={td['reward']})...")

    results: list[dict] = []
    raw_by_key: dict[tuple, pd.DataFrame] = {}
    done = [0]
    lock = threading.Lock()

    def worker(m: dict) -> tuple[dict, pd.DataFrame] | tuple[None, None]:
        r, raw = eval_model(task, td, m, seeds, omp, timeout)
        with lock:
            done[0] += 1
            status = f"mean={r['mean_across_weights']:.4f}" if r is not None else "FAIL"
            print(f"  [{done[0]:3d}/{len(models)}] "
                  f"{m['run_key']}/{m['cfg_path'].stem}  {status}")
        return r, raw

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for r, raw in pool.map(worker, models):
            if r is not None:
                results.append(r)
                raw_by_key[(r["run_key"], r["rank"], r["seed_idx"])] = raw

    df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_weight_means.csv"
    df.to_csv(out_path, index=False)
    print(f"[{task}] resultados → {out_path}  ({len(df)}/{len(models)} exitosos)")

    bests: list[dict] = []
    if not df.empty:
        summary = summarize_by_config(df)
        summary_path = out_dir / f"{task}_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"[{task}] resumen por configuración → {summary_path}  "
              f"({len(summary)} configuraciones)")

        bests = best_per_run_key(task, td, df, raw_by_key)
        if bests:
            best_path = out_dir / f"{task}_best.csv"
            pd.DataFrame(bests).to_csv(best_path, index=False)
            print(f"[{task}] mejor modelo/peso por run_key → {best_path}  "
                  f"({len(bests)} run_keys)")

    return df, bests


def summarize_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Group per-model rows by hyperparameter configuration (run_key + rank —
    the same config is retrained across several phase-3 seeds) and average
    the per-weight means across those seeds."""
    group_keys = ["run_key", "rank"]
    w_mean_cols = [c for c in df.columns if re.match(r"^w\d+_.*_mean$", c)]
    agg_cols = w_mean_cols + ["mean_across_weights"]
    hp_cols = [c for c in df.columns if c.startswith("hp_")]

    stats = df.groupby(group_keys)[agg_cols].agg(["mean", "std"])
    stats.columns = [f"{col}_{stat}" for col, stat in stats.columns]
    stats = stats.fillna(0.0)  # std is NaN when a config has a single model

    n_models = df.groupby(group_keys).size().rename("n_models")
    hp_first = df.groupby(group_keys)[hp_cols].first() if hp_cols else None

    summary = pd.concat([n_models, stats] + ([hp_first] if hp_first is not None else []),
                        axis=1).reset_index()
    summary = summary.sort_values("mean_across_weights_mean", ascending=False)
    return summary


def best_model_weight(task: str, td: dict, df: pd.DataFrame,
                       raw_by_key: dict[tuple, pd.DataFrame]) -> dict | None:
    """Within `df` (every model × weight evaluated), find the single
    highest-reward (model, weight) combination. reward/reward_std are the
    mean/std of that weight's reward across the evaluation seeds, and
    eval_seed_* carries each individual evaluation (one per seed) for that
    winning weight, taken from raw_by_key[(run_key, rank, seed_idx)]."""
    if df.empty:
        return None

    w_mean_cols = [c for c in df.columns if re.match(r"^w\d+_.*_mean$", c)]
    melted = df.melt(id_vars=["run_key", "rank", "seed_idx"],
                      value_vars=w_mean_cols, var_name="w_col", value_name="reward_value")
    best = melted.loc[melted["reward_value"].idxmax()]
    wi = int(re.match(r"^w(\d+)_", best["w_col"]).group(1))
    std_col = best["w_col"][:-len("_mean")] + "_std"

    best_row = df.loc[(df["run_key"] == best["run_key"]) &
                       (df["rank"] == best["rank"]) &
                       (df["seed_idx"] == best["seed_idx"])].iloc[0]

    result = {
        "task":         task,
        "run_key":      best["run_key"],
        "rank":         int(best["rank"]),
        "seed_idx":     int(best["seed_idx"]),
        "weight_index": wi,
        "weight_value": td["weight_vals"][wi],
        "reward":       best["reward_value"],
        "reward_std":   best_row[std_col],
        "reward_type":  td["reward"],
        "n_neurons":     int(best_row["n_neurons"]),
        "n_connections": int(best_row["n_connections"]),
    }

    raw = raw_by_key.get((best["run_key"], int(best["rank"]), int(best["seed_idx"])))
    if raw is not None:
        w_col = f"w{wi}"
        for _, r in raw.sort_values("seed").iterrows():
            result[f"eval_seed_{int(r['seed'])}"] = r[w_col]

    return result


def best_per_run_key(task: str, td: dict, df: pd.DataFrame,
                      raw_by_key: dict[tuple, pd.DataFrame]) -> list[dict]:
    """Best (model, weight) combination for each encoder/decoder run_key
    found for `task` (e.g. acrobot_ttfs_first_spike, acrobot_small_rate_argmax, ...)."""
    if df.empty:
        return []
    bests = []
    for run_key in sorted(df["run_key"].unique()):
        sub = df.loc[df["run_key"] == run_key]
        b = best_model_weight(task, td, sub, raw_by_key)
        if b is not None:
            bests.append(b)
    return bests


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), default=None,
                    help="Solo esta tarea (default: las 3)")
    ap.add_argument("--seeds", type=int, default=10,
                    help="Cantidad de seeds por modelo (default: 10)")
    ap.add_argument("--seed0", type=int, default=0,
                    help="Primera seed de evaluación (default: 0)")
    ap.add_argument("--jobs", type=int, default=8,
                    help="Modelos evaluados en paralelo (default: 8)")
    ap.add_argument("--omp", type=int, default=None,
                    help="OMP_NUM_THREADS por corrida (default: cpu_count // jobs)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Timeout por modelo en segundos (default: sin límite)")
    ap.add_argument("--out-dir", default="eval_p3_weights", dest="out_dir",
                    help="Directorio de salida (default: eval_p3_weights/)")
    ap.add_argument("--reward", choices=["shaped", "original"], default=None,
                    help="Forzar shaped u original para todas las tareas corridas "
                         "(default por tarea: acrobot=original, mountain_car=original, car=shaped)")
    args = ap.parse_args()

    omp   = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    tasks = [args.task] if args.task else list(TASKS)
    out_dir = Path(args.out_dir)

    bests: list[dict] = []
    for task in tasks:
        _, task_bests = run_task(task, seeds, args.jobs, omp, args.timeout, out_dir, args.reward)
        bests.extend(task_bests)

    print(f"\n{'='*66}")
    print("  Mejor modelo y peso por run_key (encoder/decoder)")
    print(f"{'='*66}")
    if not bests:
        print("  (sin resultados)")
    for b in bests:
        print(f"  [{b['task']}] run_key={b['run_key']}  rank={b['rank']}  "
              f"seed_idx={b['seed_idx']}  peso={b['weight_value']:g} "
              f"(w{b['weight_index']})  reward({b['reward_type']})="
              f"{b['reward']:.4f} ± {b['reward_std']:.4f}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
