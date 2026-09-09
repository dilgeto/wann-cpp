#!/usr/bin/env python3
"""
bootstrap_auto_lib.py — lógica compartida por bootstrap_compare_{car,acrobot,
mountain_car}_auto.py: dado un run_key de una corrida de fase 3 ya
entrenada, revalida sus modelos (reusa eval_p3_weights.run_task), elige el
(seed, peso) ganador, y evalúa la SNN ganadora contra N episodios para el
bootstrap (bootstrap_compare_lib.run_comparison hace la estadística).

Cada script de tarea solo necesita aportar su propio eval_ann() (el
baseline ANN difiere por tarea: binario compilado para Car, DQN/PPO de
Stable-Baselines3 en otro venv para Acrobot/Mountain Car) — todo lo demás
(revalidación, selección de ganador, evaluación de la SNN, CLI, bootstrap)
es idéntico entre tareas y vive acá una sola vez.

No se entrena nada acá — solo revalida y evalúa modelos ya entrenados.
"""
import argparse
import io
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from bootstrap_compare_lib import build_arg_parser, run_comparison
from eval_p3_weights import TASKS, run_task


def build_auto_arg_parser(doc: str, task: str) -> argparse.ArgumentParser:
    """Parser base (bootstrap_compare_lib.build_arg_parser, --out-dir sin
    default fijo) + los flags propios del paso de revalidación. Cada script
    de tarea le agrega encima sus flags de ANN (--ann-*, --dqn-*, etc.)."""
    ap = build_arg_parser(doc, None)
    ap.add_argument("--run-key", required=True, dest="run_key",
                    help="run_key de la corrida de fase 3 a validar (el mismo "
                         "usado en --tag/run_key de screening_full.py).")
    ap.add_argument("--seeds", type=int, default=11,
                    help="Seeds de revalidación en el paso 1 (default: 11)")
    ap.add_argument("--nreps", type=int, default=11,
                    help="Episodios promediados por seed en el paso 1 (default: 11)")
    ap.add_argument("--eval-jobs", type=int, default=4, dest="eval_jobs",
                    help="Modelos evaluados en paralelo en el paso 1 (default: 4)")
    ap.add_argument("--eval-omp", type=int, default=None, dest="eval_omp",
                    help="OMP_NUM_THREADS por corrida en el paso 1 "
                         "(default: cpu_count // eval_jobs)")
    ap.add_argument("--eval-timeout", type=int, default=None, dest="eval_timeout",
                    help="Timeout por run_key en el paso 1 (default: sin límite)")
    ap.add_argument("--eval-out-dir", default=None, dest="eval_out_dir",
                    help="Directorio de salida del paso 1 "
                         "(default: eval_results/<run_key>/)")
    ap.add_argument("--skip-revalidation", action="store_true", dest="skip_revalidation",
                    help=f"Si ya existe <eval-out-dir>/{task}_best.csv de una corrida "
                         f"anterior, reusarlo y saltar directo al bootstrap.")
    return ap


def revalidate(task: str, run_key: str, seeds: list[int], nreps: int, jobs: int,
               omp: int, timeout: int | None, out_dir: Path) -> pd.DataFrame:
    """Paso 1: reutiliza eval_p3_weights.run_task() para revalidar todos los
    modelos del rank ganador de run_key. Devuelve el DataFrame completo
    (una fila por seed de entrenamiento)."""
    td = TASKS[task]
    if not Path(td["executable"]).exists():
        print(f"ERROR: {td['executable']} no existe. Compilar primero "
              f"(cd build && ninja {Path(td['executable']).name}).", file=sys.stderr)
        sys.exit(1)

    print(f"── Paso 1/2: revalidando modelos de {run_key} "
          f"({len(seeds)} seeds × {nreps} nreps) ──")
    df = run_task(task, seeds, nreps, jobs, omp, timeout, out_dir,
                  reward_override=None, run_keys={run_key})
    if df.empty:
        print(f"ERROR: no se encontraron modelos de fase 3 para run_key={run_key} "
              f"(¿corriste 'screening_full.py --mode phase3 --tag ...' para esta "
              f"corrida?).", file=sys.stderr)
        sys.exit(1)
    return df


def pick_winner(df: pd.DataFrame) -> pd.Series:
    """Elige la fila (modelo/peso) con mayor reward; avisa si el segundo
    lugar queda dentro de 1 std del ganador (empate técnico)."""
    ranked = df.sort_values("reward", ascending=False).reset_index(drop=True)
    winner = ranked.iloc[0]
    print(f"\nGanador: run_key={winner['run_key']}  rank={winner['rank']}  "
          f"seed_idx={winner['seed_idx']}  peso={winner['weight_value']:g}  "
          f"reward={winner['reward']:.4f} ± {winner['reward_std']:.4f}")

    if len(ranked) > 1:
        runner_up = ranked.iloc[1]
        gap = winner["reward"] - runner_up["reward"]
        if gap < winner["reward_std"]:
            print(f"[AVISO] Empate técnico: seed_idx={runner_up['seed_idx']} "
                  f"(peso={runner_up['weight_value']:g}) quedó a {gap:.4f} del "
                  f"ganador — reward={runner_up['reward']:.4f} ± "
                  f"{runner_up['reward_std']:.4f}, dentro de 1 std. Considerá "
                  f"correr con más --seeds/--nreps si esta decisión es sensible.")
    return winner


def eval_snn(task: str, winner: pd.Series, n: int, seed0: int, omp: int,
            timeout: int | None) -> np.ndarray:
    """Corre n episodios del modelo ganador — un episodio por seed
    (--nreps 1), con el peso que la revalidación registró como mejor.
    Idéntico entre tareas: solo cambia TASKS[task] (ejecutable/config/reward)."""
    td = TASKS[task]
    run_key, rank, seed_idx = winner["run_key"], int(winner["rank"]), int(winner["seed_idx"])
    weight_index = int(winner["weight_index"])
    model_path = Path("log") / f"full_p3_{run_key}" / f"rank{rank:02d}_seed{seed_idx:02d}_best.out"
    cfg_path   = (Path("screening_full") / run_key / "p3_configs"
                  / f"rank{rank:02d}_seed{seed_idx:02d}.json")

    seeds = ",".join(str(seed0 + i) for i in range(n))
    cmd = [td["executable"],
           "-f", str(model_path),
           "-d", td["base_config"],
           "-p", str(cfg_path),
           "--seeds", seeds,
           "--reward", td["reward"],
           "--nreps", "1",
           "--episode-detail", "--weight-index", str(weight_index)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Falló la evaluación de la SNN")

    df = pd.read_csv(io.StringIO(proc.stdout))
    return df["reward"].to_numpy(dtype=float)


def run_auto_comparison(task: str, args: argparse.Namespace,
                        eval_ann_fn: Callable[[int, int, int | None], np.ndarray],
                        ann_desc: str, ann_label: str, task_label: str) -> None:
    """Orquesta los 2 pasos completos y llama a run_comparison(). eval_ann_fn
    recibe (n, seed0, timeout) — cualquier otro parámetro específico del ANN
    (ruta de modelo, venv, etc.) va ya cerrado sobre la función via
    functools.partial/lambda en el script de la tarea."""
    eval_out_dir = Path(args.eval_out_dir or f"eval_results/{args.run_key}")
    best_csv = eval_out_dir / f"{task}_best.csv"

    if args.skip_revalidation and best_csv.exists():
        print(f"── Paso 1/2: --skip-revalidation, reusando {best_csv} ──")
        df = pd.read_csv(best_csv)
    else:
        eval_jobs = args.eval_jobs
        eval_omp  = args.eval_omp or max(1, (os.cpu_count() or 4) // eval_jobs)
        seeds     = list(range(args.seed0, args.seed0 + args.seeds))
        df = revalidate(task, args.run_key, seeds, args.nreps, eval_jobs, eval_omp,
                        args.eval_timeout, eval_out_dir)

    winner = pick_winner(df)

    omp = args.omp or (os.cpu_count() or 4)
    if args.out_dir is None:
        args.out_dir = f"bootstrap_results/{args.run_key}"

    print(f"\n── Paso 2/2: bootstrap SNN vs ANN ({args.n} episodios, "
          f"{args.resamples} resamples) ──")
    print(f"Evaluando SNN ({args.run_key} rank={int(winner['rank'])} "
          f"seed_idx={int(winner['seed_idx'])} peso={winner['weight_value']:g} "
          f"[índice {int(winner['weight_index'])}]): {args.n} episodios...")
    rewards_snn = eval_snn(task, winner, args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN ({ann_desc}): {args.n} episodios...")
    rewards_ann = eval_ann_fn(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle=f"{task_label} ({args.run_key}) — SNN vs ANN ({ann_label}), "
                           f"bootstrap no pareado",
                  plot_stem=f"{args.run_key}_snn_vs_ann", ann_label=f"ANN ({ann_label})")
