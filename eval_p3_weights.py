#!/usr/bin/env python3
"""
Evalúa cada modelo resultante de fase 3 (Acrobot, Mountain Car discreto,
Racing Car) con N seeds distintas (default 10); en cada seed se prueban los
6 pesos compartidos de esa tarea (getDistFitness), y luego se promedia cada
peso a través de las seeds.

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
    },
    "mountain_car": {
        "executable":       "./build/wann_eval_weights_disc_mc",
        "base_config":       "p/disc_mc_snn.json",
        "weight_vals":       [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "screening_prefix":  "disc_mc",
    },
    "car": {
        "executable":       "./build/wann_eval_weights_car",
        "base_config":       "p/car_snn.json",
        "weight_vals":       [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
        "screening_prefix":  "car",
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


def eval_model(task: str, td: dict, model: dict, seeds: list[int],
               omp: int, timeout: int | None) -> dict | None:
    """Run wann_eval_weights_{task} for one model, return the per-weight
    mean across `seeds` (each seed already averages over the 6 weights)."""
    seeds_arg = ",".join(str(s) for s in seeds)
    cmd = [td["executable"],
           "-f", str(model["model_path"]),
           "-d", td["base_config"],
           "-p", str(model["cfg_path"]),
           "--seeds", seeds_arg]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {model['run_key']}/{model['cfg_path'].stem}", file=sys.stderr)
        return None

    if proc.returncode != 0:
        print(f"  [FAIL] {model['run_key']}/{model['cfg_path'].stem}\n{proc.stderr}",
              file=sys.stderr)
        return None

    df = pd.read_csv(io.StringIO(proc.stdout))
    w_cols = [c for c in df.columns if c != "seed"]
    means = df[w_cols].mean()

    row = {
        "task":        task,
        "run_key":     model["run_key"],
        "rank":        model["rank"],
        "seed_idx":    model["seed_idx"],
        "n_seeds_eval": len(df),
    }
    for i, wc in enumerate(w_cols):
        wval = td["weight_vals"][i]
        row[f"w{i}_{wval:g}_mean"] = means[wc]
    row["mean_across_weights"] = means.mean()
    return row


def run_task(task: str, seeds: list[int], jobs: int, omp: int,
             timeout: int | None, out_dir: Path) -> pd.DataFrame:
    td = TASKS[task]
    if not Path(td["executable"]).exists():
        print(f"ERROR: {td['executable']} no existe. Compilar primero "
              f"(cd build && ninja {Path(td['executable']).name}).", file=sys.stderr)
        return pd.DataFrame()

    prefix = td["screening_prefix"]
    models = find_models(prefix)
    if not models:
        print(f"[{task}] no se encontraron modelos de fase 3 en "
              f"screening_full/{prefix}_*/p3_configs.")
        return pd.DataFrame()

    print(f"\n[{task}] {len(models)} modelos encontrados, "
          f"evaluando con {len(seeds)} seeds cada uno...")

    results: list[dict] = []
    done = [0]
    lock = threading.Lock()

    def worker(m: dict) -> dict | None:
        r = eval_model(task, td, m, seeds, omp, timeout)
        with lock:
            done[0] += 1
            status = f"mean={r['mean_across_weights']:.4f}" if r is not None else "FAIL"
            print(f"  [{done[0]:3d}/{len(models)}] "
                  f"{m['run_key']}/{m['cfg_path'].stem}  {status}")
        return r

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for r in pool.map(worker, models):
            if r is not None:
                results.append(r)

    df = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_weight_means.csv"
    df.to_csv(out_path, index=False)
    print(f"[{task}] resultados → {out_path}  ({len(df)}/{len(models)} exitosos)")
    return df


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
    args = ap.parse_args()

    omp   = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    tasks = [args.task] if args.task else list(TASKS)
    out_dir = Path(args.out_dir)

    for task in tasks:
        run_task(task, seeds, args.jobs, omp, args.timeout, out_dir)


if __name__ == "__main__":
    main()
