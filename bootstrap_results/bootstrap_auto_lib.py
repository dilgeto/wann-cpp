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
import concurrent.futures
import io
import os
import re
import subprocess
import sys
import threading
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
    ap.add_argument("--run-key", default=None, dest="run_key",
                    help="run_key de la corrida de fase 3 a validar (el mismo "
                         "usado en --tag/run_key de screening_full.py). "
                         "Requerido salvo que se pase --model (modelo suelto, "
                         "entrenado fuera del pipeline de screening_full).")
    ap.add_argument("--model", default=None, dest="model_path",
                    help="Ruta a un *_best.out entrenado fuera del pipeline de "
                         "screening_full (sin rank/seed_idx/p3_configs — p.ej. "
                         "una corrida suelta de wann_<task> con -o algún "
                         "prefijo). Mutuamente excluyente con --run-key/"
                         "--model-dir; salta la revalidación/selección de "
                         "ganador por completo y bootstrapea directamente ese "
                         "modelo.")
    ap.add_argument("--model-dir", default=None, dest="model_dir",
                    help="Directorio con varias corridas sueltas seed<NN>_best.out "
                         "(mismo patrón que -o log/<dir>/seed<NN> al entrenar "
                         "wann_<task> a mano N veces, sin pasar por "
                         "screening_full) — revalida las que encuentre y "
                         "bootstrapea la de mayor reward, igual que --run-key "
                         "pero sin requerir p3_configs. Mutuamente excluyente "
                         "con --run-key/--model.")
    ap.add_argument("--config", default=None, dest="config_path",
                    help="Config base para evaluar --model (default: la "
                         "config base de la tarea, p/<task>_snn.json). Ignorado "
                         "sin --model.")
    ap.add_argument("--override", default=None, dest="override_path",
                    help="Override JSON (archivo o string inline) aplicado sobre "
                         "--config, por si la corrida ad-hoc usó hiperparámetros "
                         "SNN distintos de la config base (ventana, tau, etc.). "
                         "Ignorado sin --model.")
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
    ap.add_argument("--seed-idx", type=int, default=None, dest="seed_idx",
                    help="Fija el bootstrap a este modelo puntual (la seed de "
                         "entrenamiento seed_idx dentro del rank ganador de "
                         "--run-key) en vez de dejar que pick_winner() elija "
                         "automáticamente la fila de mayor reward entre las "
                         "revalidadas. Útil para aislar un modelo específico "
                         "(p.ej. controlar por su propia ventana/hiperparámetros) "
                         "en vez de comparar contra el que gane por azar de "
                         "revalidación.")
    ap.add_argument("--weight-index", type=int, default=None, dest="weight_index_override",
                    help="Junto con --seed-idx: evalúa ese modelo con este índice "
                         "de peso compartido en vez del que la revalidación "
                         "registró como mejor para esa seed. Junto con --model: "
                         "índice de peso a evaluar (default: se infiere del "
                         "<model>.wi si existe, como el que deja wann_<task>).")
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


def select_model(df: pd.DataFrame, seed_idx: int, weight_index: int | None) -> pd.Series:
    """Fija el bootstrap a un modelo puntual (la seed de entrenamiento
    seed_idx dentro del rank ya revalidado) en vez de dejar que pick_winner()
    elija automáticamente la fila de mayor reward. weight_index (opcional)
    sobreescribe qué peso compartido evaluar para ese modelo — el archivo de
    red no está atado a un peso, así que cualquier índice es válido, no solo
    el que la revalidación registró como mejor para esa seed."""
    matches = df[df["seed_idx"] == seed_idx]
    if matches.empty:
        raise SystemExit(f"seed_idx={seed_idx} no está entre los revalidados "
                          f"({sorted(df['seed_idx'].unique())}).")
    model = matches.iloc[0].copy()
    print(f"\nModelo fijado manualmente: run_key={model['run_key']}  rank={model['rank']}  "
          f"seed_idx={model['seed_idx']}  peso={model['weight_value']:g}  "
          f"reward={model['reward']:.4f} ± {model['reward_std']:.4f}")

    if weight_index is not None and weight_index != model["weight_index"]:
        new_weight_value = TASKS[model["task"]]["weight_vals"][weight_index]
        print(f"  peso sobreescrito: índice {model['weight_index']} "
              f"(peso={model['weight_value']:g}) → índice {weight_index} "
              f"(peso={new_weight_value:g})")
        model["weight_index"] = weight_index
        model["weight_value"] = new_weight_value
    return model


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


def eval_snn_manual(task: str, model_path: Path, config_path: Path,
                    override_path: str | None, weight_index: int, n: int,
                    seed0: int, omp: int, timeout: int | None) -> np.ndarray:
    """Como eval_snn, pero para un modelo entrenado fuera del pipeline de
    screening_full (sin rank/seed_idx ni p3_configs) — p.ej. una corrida
    suelta de wann_<task> con -o algún prefijo (log/<prefix>_best.out)."""
    td = TASKS[task]
    seeds = ",".join(str(seed0 + i) for i in range(n))
    cmd = [td["executable"],
           "-f", str(model_path),
           "-d", str(config_path)]
    if override_path:
        cmd += ["-p", override_path]
    cmd += ["--seeds", seeds,
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


def run_manual_comparison(task: str, args: argparse.Namespace,
                          eval_ann_fn: Callable[[int, int, int | None], np.ndarray],
                          ann_desc: str, ann_label: str, task_label: str,
                          target_label: str, plot_suffix: str) -> None:
    """Bootstrap directo de un --model suelto contra el ANN, sin pasar por
    revalidación/pick_winner (esos asumen la estructura de screening_full)."""
    td = TASKS[task]
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"--model no existe: {model_path}")
    config_path = Path(args.config_path) if args.config_path else Path(td["base_config"])

    weight_index = args.weight_index_override
    if weight_index is None:
        wi_path = model_path.with_suffix(".wi")
        if not wi_path.exists():
            raise SystemExit(f"--weight-index requerido: no se encontró {wi_path} "
                              f"para inferirlo automáticamente.")
        weight_index = int(wi_path.read_text().strip())
        print(f"Peso inferido de {wi_path}: índice {weight_index}")
    weight_value = td["weight_vals"][weight_index]

    print(f"\nModelo manual: {model_path}  config={config_path}"
          f"{f'  override={args.override_path}' if args.override_path else ''}"
          f"  peso=índice {weight_index} (valor={weight_value:g})")

    omp = args.omp or (os.cpu_count() or 4)
    if args.out_dir is None:
        args.out_dir = f"bootstrap_results/{model_path.stem}"

    print(f"\n── Bootstrap {target_label} vs ANN ({args.n} episodios, "
          f"{args.resamples} resamples) ──")
    print(f"Evaluando {target_label} ({model_path.name}, peso={weight_value:g}): "
          f"{args.n} episodios...")
    rewards_target = eval_snn_manual(task, model_path, config_path, args.override_path,
                                     weight_index, args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN ({ann_desc}): {args.n} episodios...")
    rewards_ann = eval_ann_fn(args.n, args.seed0, args.timeout)

    run_comparison(rewards_target, rewards_ann, args,
                  suptitle=f"{task_label} ({model_path.stem}) — {target_label} vs ANN "
                           f"({ann_label}), bootstrap no pareado",
                  plot_stem=f"{model_path.stem}_{plot_suffix}", ann_label=f"ANN ({ann_label})",
                  snn_label=target_label)


DIR_MODEL_RE = re.compile(r"^seed(\d+)_best$")


def find_dir_models(model_dir: Path) -> list[dict]:
    """Encuentra seed<NN>_best.out sueltos en model_dir (el patrón que deja
    -o <model_dir>/seed<NN> al correr wann_<task> a mano N veces, sin pasar
    por screening_full), emparejados con su propio seed<NN>_best.wi."""
    models = []
    for out_path in sorted(model_dir.glob("seed*_best.out")):
        m = DIR_MODEL_RE.match(out_path.stem)
        if not m:
            continue
        wi_path = out_path.with_suffix(".wi")
        if not wi_path.exists():
            print(f"  [SKIP] sin .wi: {out_path}", file=sys.stderr)
            continue
        models.append({
            "seed_idx":     int(m.group(1)),
            "model_path":   out_path,
            "weight_index": int(wi_path.read_text().strip()),
        })
    return models


def fetch_episode_detail_dir(task: str, model_path: Path, config_path: Path,
                             override_path: str | None, weight_index: int,
                             seeds: list[int], nreps: int, omp: int,
                             timeout: int | None) -> pd.DataFrame | None:
    """Como eval_p3_weights.fetch_episode_detail, pero apuntando directo a un
    model_path/config_path explícitos en vez de derivarlos de run_key/rank."""
    td = TASKS[task]
    cmd = [td["executable"], "-f", str(model_path), "-d", str(config_path)]
    if override_path:
        cmd += ["-p", str(override_path)]
    cmd += ["--seeds", ",".join(str(s) for s in seeds),
            "--reward", td["reward"], "--nreps", str(nreps),
            "--episode-detail", "--weight-index", str(weight_index)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] {model_path}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"  [FAIL] {model_path}\n{proc.stderr}", file=sys.stderr)
        return None
    return pd.read_csv(io.StringIO(proc.stdout))


def revalidate_dir(task: str, model_dir: Path, config_path: Path, override_path: str | None,
                   seeds: list[int], nreps: int, jobs: int, omp: int,
                   timeout: int | None, out_dir: Path) -> pd.DataFrame:
    """Paso 1 para --model-dir: revalida cada seed<NN>_best.out encontrado en
    model_dir contra la MISMA config/override (comparten hiperparámetros, a
    diferencia de --run-key donde cada rank tiene los suyos) — equivalente a
    correr eval_p3_weights.py a mano para un directorio que no tiene
    p3_configs. Devuelve un DataFrame con las mismas columnas que
    eval_p3_weights.run_task() para poder reusar pick_winner()/select_model()."""
    td = TASKS[task]
    if not Path(td["executable"]).exists():
        print(f"ERROR: {td['executable']} no existe. Compilar primero "
              f"(cd build && ninja {Path(td['executable']).name}).", file=sys.stderr)
        sys.exit(1)

    models = find_dir_models(model_dir)
    if not models:
        print(f"ERROR: no se encontraron seed*_best.out en {model_dir}.", file=sys.stderr)
        sys.exit(1)

    print(f"── Paso 1/2: revalidando {len(models)} modelos de {model_dir} "
          f"({len(seeds)} seeds × {nreps} nreps) ──")

    rows: list[dict] = []
    lock = threading.Lock()
    done = [0]

    def worker(m: dict) -> dict | None:
        ep_df = fetch_episode_detail_dir(task, m["model_path"], config_path, override_path,
                                         m["weight_index"], seeds, nreps, omp, timeout)
        if ep_df is None:
            return None
        seed_means = ep_df.groupby("seed")["reward"].mean()
        row = {
            "task":          task,
            "run_key":       model_dir.name,
            "rank":          0,
            "seed_idx":      m["seed_idx"],
            "weight_index":  m["weight_index"],
            "weight_value":  td["weight_vals"][m["weight_index"]],
            "reward_type":   td["reward"],
            "reward":        float(seed_means.mean()),
            "reward_std":    float(seed_means.std()),
        }
        with lock:
            done[0] += 1
            print(f"  [{done[0]:3d}/{len(models)}] seed_idx={m['seed_idx']}  "
                  f"peso={row['weight_value']:g}  "
                  f"reward={row['reward']:.4f}±{row['reward_std']:.4f}")
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for row in pool.map(worker, models):
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{task}_best.csv"
    df.to_csv(best_path, index=False)
    print(f"[{task}] {model_dir} → {best_path}  ({len(df)}/{len(models)} exitosos)")
    if df.empty:
        print(f"ERROR: ningún modelo de {model_dir} pudo revalidarse.", file=sys.stderr)
        sys.exit(1)
    return df


def eval_snn_dir(task: str, model_dir: Path, config_path: Path, override_path: str | None,
                 winner: pd.Series, n: int, seed0: int, omp: int,
                 timeout: int | None) -> np.ndarray:
    """Paso 2 para --model-dir: como eval_snn_manual, pero reconstruye
    model_path desde model_dir + seed_idx del ganador."""
    model_path = model_dir / f"seed{int(winner['seed_idx']):02d}_best.out"
    return eval_snn_manual(task, model_path, config_path, override_path,
                           int(winner["weight_index"]), n, seed0, omp, timeout)


def run_dir_comparison(task: str, args: argparse.Namespace,
                       eval_ann_fn: Callable[[int, int, int | None], np.ndarray],
                       ann_desc: str, ann_label: str, task_label: str,
                       target_label: str, plot_suffix: str) -> None:
    """--model-dir: revalida todas las seed<NN>_best.out de un directorio
    suelto, elige la de mayor reward (o la que fije --seed-idx) y bootstrapea
    esa contra el ANN — el equivalente de --run-key para corridas que no
    pasaron por screening_full (sin p3_configs)."""
    td = TASKS[task]
    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"--model-dir no es un directorio: {model_dir}")
    config_path = Path(args.config_path) if args.config_path else Path(td["base_config"])

    eval_out_dir = Path(args.eval_out_dir or f"eval_results/{model_dir.name}")
    best_csv = eval_out_dir / f"{task}_best.csv"

    if args.skip_revalidation and best_csv.exists():
        print(f"── Paso 1/2: --skip-revalidation, reusando {best_csv} ──")
        df = pd.read_csv(best_csv)
    else:
        eval_jobs = args.eval_jobs
        eval_omp  = args.eval_omp or max(1, (os.cpu_count() or 4) // eval_jobs)
        seeds     = list(range(args.seed0, args.seed0 + args.seeds))
        df = revalidate_dir(task, model_dir, config_path, args.override_path, seeds,
                            args.nreps, eval_jobs, eval_omp, args.eval_timeout, eval_out_dir)

    winner = (select_model(df, args.seed_idx, args.weight_index_override)
              if args.seed_idx is not None else pick_winner(df))

    omp = args.omp or (os.cpu_count() or 4)
    if args.out_dir is None:
        args.out_dir = f"bootstrap_results/{model_dir.name}"

    print(f"\n── Paso 2/2: bootstrap {target_label} vs ANN ({args.n} episodios, "
          f"{args.resamples} resamples) ──")
    print(f"Evaluando {target_label} ({model_dir.name} seed_idx={int(winner['seed_idx'])} "
          f"peso={winner['weight_value']:g} [índice {int(winner['weight_index'])}]): "
          f"{args.n} episodios...")
    rewards_target = eval_snn_dir(task, model_dir, config_path, args.override_path, winner,
                                  args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN ({ann_desc}): {args.n} episodios...")
    rewards_ann = eval_ann_fn(args.n, args.seed0, args.timeout)

    run_comparison(rewards_target, rewards_ann, args,
                  suptitle=f"{task_label} ({model_dir.name}) — {target_label} vs ANN "
                           f"({ann_label}), bootstrap no pareado",
                  plot_stem=f"{model_dir.name}_{plot_suffix}", ann_label=f"ANN ({ann_label})",
                  snn_label=target_label)


def run_auto_comparison(task: str, args: argparse.Namespace,
                        eval_ann_fn: Callable[[int, int, int | None], np.ndarray],
                        ann_desc: str, ann_label: str, task_label: str,
                        eval_target_fn: Callable[[str, "pd.Series", int, int, int, int | None], np.ndarray] | None = None,
                        target_label: str = "SNN",
                        target_desc: str | None = None,
                        plot_suffix: str = "snn_vs_ann") -> None:
    """Orquesta los 2 pasos completos y llama a run_comparison(). eval_ann_fn
    recibe (n, seed0, timeout) — cualquier otro parámetro específico del ANN
    (ruta de modelo, venv, etc.) va ya cerrado sobre la función via
    functools.partial/lambda en el script de la tarea.

    eval_target_fn (default: eval_snn — el simulador SNN por software) tiene
    la misma firma que eval_snn: (task, winner, n, seed0, omp, timeout) ->
    rewards. Pasar uno propio (p.ej. eval_odin en
    bootstrap_compare_car_odin_auto.py) reusa el mismo paso 1 de
    revalidación/selección de ganador pero evalúa el modelo ganador en otro
    backend en vez del simulador de software — target_label/target_desc
    ajustan las etiquetas de la salida/gráfico en consecuencia."""
    modes_given = sum(x is not None for x in (args.run_key, args.model_path, args.model_dir))
    if modes_given > 1:
        raise SystemExit("--run-key, --model y --model-dir son mutuamente excluyentes.")

    if args.model_path is not None or args.model_dir is not None:
        if eval_target_fn is not None:
            raise SystemExit("--model/--model-dir no soportan un eval_target_fn "
                              "custom (p.ej. backend ODIN) — solo el simulador "
                              "SNN estándar.")
        if args.model_path is not None:
            run_manual_comparison(task, args, eval_ann_fn, ann_desc, ann_label,
                                  task_label, target_label=target_label,
                                  plot_suffix=plot_suffix)
        else:
            run_dir_comparison(task, args, eval_ann_fn, ann_desc, ann_label,
                               task_label, target_label=target_label,
                               plot_suffix=plot_suffix)
        return
    if args.run_key is None:
        raise SystemExit("Se requiere --run-key (o --model/--model-dir para "
                          "modelos sueltos).")
    if args.weight_index_override is not None and args.seed_idx is None:
        raise SystemExit("--weight-index requiere --seed-idx (fija primero qué "
                          "modelo evaluar).")

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

    winner = (select_model(df, args.seed_idx, args.weight_index_override)
              if args.seed_idx is not None else pick_winner(df))

    omp = args.omp or (os.cpu_count() or 4)
    if args.out_dir is None:
        args.out_dir = f"bootstrap_results/{args.run_key}"

    target_fn = eval_target_fn or eval_snn
    target_desc = target_desc or target_label

    print(f"\n── Paso 2/2: bootstrap {target_label} vs ANN ({args.n} episodios, "
          f"{args.resamples} resamples) ──")
    print(f"Evaluando {target_desc} ({args.run_key} rank={int(winner['rank'])} "
          f"seed_idx={int(winner['seed_idx'])} peso={winner['weight_value']:g} "
          f"[índice {int(winner['weight_index'])}]): {args.n} episodios...")
    rewards_target = target_fn(task, winner, args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN ({ann_desc}): {args.n} episodios...")
    rewards_ann = eval_ann_fn(args.n, args.seed0, args.timeout)

    run_comparison(rewards_target, rewards_ann, args,
                  suptitle=f"{task_label} ({args.run_key}) — {target_label} vs ANN ({ann_label}), "
                           f"bootstrap no pareado",
                  plot_stem=f"{args.run_key}_{plot_suffix}", ann_label=f"ANN ({ann_label})",
                  snn_label=target_label)
