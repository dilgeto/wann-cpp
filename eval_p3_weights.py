#!/usr/bin/env python3
"""
Para CADA combinación encoder/decoder (run_key) de una tarea (Acrobot,
Mountain Car discreto, Racing Car) — típicamente 4 por tarea —, toma su
propia mejor configuración de hiperparámetros (el rank con mayor fitPeak DE
ENTRENAMIENTO promedio entre sus seeds: el máximo real entre los 6 pesos
compartidos, última fila de cada
log/full_p3_<run_key>/rank<NN>_seed<NN>_stats.out — no una re-evaluación) y
revalida sus 11 modelos (uno por seed de entrenamiento), cada uno con el
peso que el entrenamiento registró como mejor para ese modelo (su archivo
_best.wi). Cada (modelo, peso) se revalida con N seeds distintas
(default 11), promediando nreps episodios por seed (default 11 — ver
--nreps; sobreescribe alg_nReps solo para esta evaluación, no afecta el
entrenamiento).

Guarda, por tarea (una fila por modelo — hasta 4 run_keys × 11 seeds):
  {task}_best.csv          — modelo/peso elegido,
                              fitPeak/fitTop/fitTopOrig de entrenamiento,
                              reward/reward_std (media±std de las seeds,
                              cada una ya promediada sobre sus episodios,
                              usando SOLO el peso ganador — su _best.wi) y
                              peak_reward/peak_reward_std (media±std sobre
                              el total de episodios de todas las seeds
                              juntas, sin agrupar primero por seed).
                              mean_across_6weights/_std: mismo modelo/seeds,
                              pero promediando los 6 pesos compartidos (no
                              solo el ganador) — para comparar el peso
                              ganador contra el promedio general de la red.
                              peak_seed_idx queda solo como referencia: cuál
                              seed tuvo el mejor promedio individual.
  {task}_best_episodes.csv — los nreps episodios individuales de cada seed
                              evaluada (run_key, seed, episode, reward),
                              SOLO del peso ganador (no del barrido de 6).

Acrobot y Mountain Car discreto se evalúan con la recompensa original (sin
shaping); Car se evalúa con la recompensa shaped. Ver --reward para forzar
un modo distinto.

Requiere los binarios wann_eval_weights_{acrobot,car,disc_mc} compilados:
  cd build && ninja wann_eval_weights_acrobot wann_eval_weights_car wann_eval_weights_disc_mc

Tareas (--task): acrobot | mountain_car | car
  "mountain_car" aquí es Mountain Car discreto (internamente sigue siendo la
  tarea "disc_mc" del código — ver comentario junto a TASKS más abajo).

Uso:
  python eval_p3_weights.py                       # las 3 tareas, todos los run_keys encontrados
  python eval_p3_weights.py --task acrobot         # solo acrobot
  python eval_p3_weights.py --task mountain_car    # solo Mountain Car discreto
  python eval_p3_weights.py --seeds 11 --nreps 11
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

# Columnas de DataGatherer::save() (ver src/DataGatherer.cpp): xScale, fitMed,
# fitMax(elite), fitTop(best/récord), fitPeak(máximo real entre pesos),
# nodeMed, connMed, fitTopOrig. NOTA: en p3_validation.csv la columna
# "peak_fitness" en realidad lee fitTop (screening_reduce.py:_read_peak), no
# fitPeak — aquí SÍ se usa el fitPeak real, tomado directo del _stats.out.
STATS_COLS = ["evals", "fitMed", "fitMax", "fitTop", "fitPeak", "nodeMed",
              "connMed", "fitTopOrig"]

MODEL_RE = re.compile(r"^rank(\d+)_seed(\d+)$")


def find_models(prefix: str, run_keys: set[str] | None = None) -> list[dict]:
    """Locate every phase-3 model under screening_full/<prefix>_*/p3_configs.
    If run_keys is given, only those exact run_key directories are used
    (instead of every screening_full/<prefix>_* match) — for revalidating a
    single ad-hoc run (e.g. a '_wide' tag) without re-running every official
    encoder/decoder combination for the task."""
    if run_keys:
        cfg_dirs = sorted(Path("screening_full") / rk / "p3_configs" for rk in run_keys)
        cfg_dirs = [d for d in cfg_dirs if d.is_dir()]
    else:
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


def read_last_stats(log_prefix: Path) -> dict | None:
    """Read the last row of <log_prefix>_stats.out (one row per generation,
    see DataGatherer::save). Returns None if the file is missing/empty."""
    stats_path = Path(f"{log_prefix}_stats.out")
    if not stats_path.exists():
        return None
    try:
        df = pd.read_csv(stats_path, header=None)
    except Exception:
        return None
    if df.empty:
        return None
    df.columns = STATS_COLS[:df.shape[1]]
    last = df.iloc[-1]
    return {c: float(last[c]) for c in df.columns}


def read_best_wi(log_prefix: Path) -> int | None:
    """Read <log_prefix>_best.wi — the shared-weight index the training run
    recorded as best for the elite individual at that save point."""
    wi_path = Path(f"{log_prefix}_best.wi")
    if not wi_path.exists():
        return None
    try:
        return int(wi_path.read_text().strip())
    except ValueError:
        return None


def _group_models_by_config(prefix: str, run_keys: set[str] | None = None) -> dict[tuple[str, int], list[dict]]:
    """Group every phase-3 model under `prefix` by (run_key, rank), attaching
    each model's own weight index (_best.wi) and last-generation training
    stats (_stats.out). Skips models missing either file. This does NOT
    re-evaluate anything; it only reads existing training logs."""
    models = find_models(prefix, run_keys)
    by_group: dict[tuple[str, int], list[dict]] = {}
    for m in models:
        log_prefix = Path("log") / f"full_p3_{m['run_key']}" / f"rank{m['rank']:02d}_seed{m['seed_idx']:02d}"
        stats = read_last_stats(log_prefix)
        wi = read_best_wi(log_prefix)
        if stats is None or wi is None:
            continue
        by_group.setdefault((m["run_key"], m["rank"]), []).append({**m, "weight_index": wi, **stats})
    return by_group


def best_config_per_run_key(prefix: str, run_keys: set[str] | None = None) -> dict[str, list[dict]]:
    """For EVERY run_key (encoder/decoder combination) found under `prefix`,
    pick its own best rank (hyperparameter config) — the one with the
    highest MEAN training-time fitPeak across its own seeds (the real
    max-across-weights fitness, read from the last row of each model's
    _stats.out). Unlike a single global winner, this returns one winning
    group PER run_key.

    Returns {run_key: [models...]} — one dict per seed_idx within that
    run_key's winning rank, each carrying its own weight index (that seed's
    own _best.wi)."""
    by_group = _group_models_by_config(prefix, run_keys)
    if not by_group:
        return {}

    run_keys = sorted({rk for rk, _ in by_group})
    result: dict[str, list[dict]] = {}
    for run_key in run_keys:
        candidates = {k: v for k, v in by_group.items() if k[0] == run_key}
        best_key = max(candidates, key=lambda k: sum(c["fitPeak"] for c in candidates[k]) / len(candidates[k]))
        result[run_key] = candidates[best_key]
    return result

    best_key = max(by_group, key=lambda k: sum(c["fitPeak"] for c in by_group[k]) / len(by_group[k]))
    return by_group[best_key]


def fetch_episode_detail(td: dict, best: dict, seeds: list[int], nreps: int,
                          omp: int, timeout: int | None) -> pd.DataFrame | None:
    """Run wann_eval_weights_{task} in --episode-detail mode for a fixed
    (model, weight), across every seed in `seeds`. Returns a DataFrame with
    columns run_key, seed, episode, reward — the individual episodes behind
    each seed's average."""
    run_key, rank, seed_idx, wi = (best["run_key"], best["rank"],
                                    best["seed_idx"], best["weight_index"])
    model_path = Path("log") / f"full_p3_{run_key}" / f"rank{rank:02d}_seed{seed_idx:02d}_best.out"
    cfg_path   = (Path("screening_full") / run_key / "p3_configs"
                  / f"rank{rank:02d}_seed{seed_idx:02d}.json")

    cmd = [td["executable"],
           "-f", str(model_path),
           "-d", td["base_config"],
           "-p", str(cfg_path),
           "--seeds", ",".join(str(s) for s in seeds),
           "--reward", td["reward"],
           "--nreps", str(nreps),
           "--episode-detail", "--weight-index", str(wi)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] episode-detail {run_key}", file=sys.stderr)
        return None

    if proc.returncode != 0:
        print(f"  [FAIL] episode-detail {run_key}\n{proc.stderr}", file=sys.stderr)
        return None

    df = pd.read_csv(io.StringIO(proc.stdout))
    df.insert(0, "train_seed_idx", seed_idx)  # cuál de los 11 modelos entrenados es este
    df.insert(0, "run_key", run_key)
    return df


def fetch_weights_sweep(td: dict, model: dict, seeds: list[int], nreps: int,
                        omp: int, timeout: int | None) -> pd.DataFrame | None:
    """Run wann_eval_weights_{task} in aggregate mode (sweeps los 6 pesos
    compartidos, sin --weight-index) para un modelo a través de cada seed en
    `seeds`. Devuelve un DataFrame: columnas seed, w0..w{N-1} (cada celda ya
    promediada sobre nreps episodios)."""
    run_key, rank, seed_idx = model["run_key"], model["rank"], model["seed_idx"]
    model_path = Path("log") / f"full_p3_{run_key}" / f"rank{rank:02d}_seed{seed_idx:02d}_best.out"
    cfg_path   = (Path("screening_full") / run_key / "p3_configs"
                  / f"rank{rank:02d}_seed{seed_idx:02d}.json")

    cmd = [td["executable"],
           "-f", str(model_path),
           "-d", td["base_config"],
           "-p", str(cfg_path),
           "--seeds", ",".join(str(s) for s in seeds),
           "--reward", td["reward"],
           "--nreps", str(nreps)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"  [TIMEOUT] weights-sweep {run_key}", file=sys.stderr)
        return None

    if proc.returncode != 0:
        print(f"  [FAIL] weights-sweep {run_key}\n{proc.stderr}", file=sys.stderr)
        return None

    return pd.read_csv(io.StringIO(proc.stdout))


def run_task(task: str, seeds: list[int], nreps: int, jobs: int, omp: int,
             timeout: int | None, out_dir: Path,
             reward_override: str | None = None,
             run_keys: set[str] | None = None) -> pd.DataFrame:
    td = dict(TASKS[task])
    if reward_override:
        td["reward"] = reward_override
    if not Path(td["executable"]).exists():
        print(f"ERROR: {td['executable']} no existe. Compilar primero "
              f"(cd build && ninja {Path(td['executable']).name}).", file=sys.stderr)
        return pd.DataFrame()

    prefix = td["screening_prefix"]
    groups = best_config_per_run_key(prefix, run_keys)
    if not groups:
        where = f"run_key(s) {sorted(run_keys)}" if run_keys else f"screening_full/{prefix}_*/"
        print(f"[{task}] no se encontraron modelos/stats de fase 3 en {where} "
              f"(o su carpeta log/full_p3_* correspondiente).")
        return pd.DataFrame()

    print(f"\n[{task}] {len(groups)} combinaciones encoder/decoder encontradas — "
          f"revalidando cada una con su propia mejor config de hiperparámetros:")
    models: list[dict] = []
    for run_key, group in groups.items():
        mean_fit_peak = sum(m["fitPeak"] for m in group) / len(group)
        rank = group[0]["rank"]
        print(f"  - {run_key}: rank={rank}  "
              f"(mean fitPeak entre sus {len(group)} seeds de entrenamiento = {mean_fit_peak:.4f})")
        models.extend(group)

    print(f"[{task}] revalidando {len(models)} modelos en total "
          f"({len(groups)} combinaciones × hasta 11 seeds de entrenamiento c/u) "
          f"con {len(seeds)} seeds × {nreps} episodios cada uno (reward={td['reward']})...")

    rows: list[dict] = []
    episode_dfs: list[pd.DataFrame] = []
    lock = threading.Lock()
    done = [0]

    def worker(w: dict) -> tuple[dict | None, pd.DataFrame | None]:
        ep_df = fetch_episode_detail(td, w, seeds, nreps, omp, timeout)
        if ep_df is None:
            return None, None

        seed_means = ep_df.groupby("seed")["reward"].mean()
        peak_seed = int(seed_means.idxmax())  # solo informativo: la seed de mejor promedio
        all_episodes = ep_df["reward"]        # las seeds × nreps episodios, todos juntos
        n_neurons, n_connections = model_size(w["model_path"])

        # Barrido de los 6 pesos compartidos (mismas seeds/nreps), para
        # comparar el peso ganador contra el promedio general de la red.
        sweep_df = fetch_weights_sweep(td, w, seeds, nreps, omp, timeout)
        if sweep_df is not None:
            sweep_vals = sweep_df[[c for c in sweep_df.columns if c != "seed"]].to_numpy().ravel()
            mean_6weights, std_6weights = float(sweep_vals.mean()), float(sweep_vals.std())
        else:
            mean_6weights = std_6weights = None

        row = {
            "task":                  task,
            "run_key":               w["run_key"],
            "rank":                  w["rank"],
            "seed_idx":              w["seed_idx"],
            "weight_index":          w["weight_index"],
            "weight_value":          td["weight_vals"][w["weight_index"]],
            "reward_type":           td["reward"],
            "training_fitPeak":      w["fitPeak"],
            "training_fitTop":       w["fitTop"],
            "training_fitTopOrig":   w.get("fitTopOrig"),
            "reward":                float(seed_means.mean()),
            "reward_std":            float(seed_means.std()),
            "mean_across_6weights":      mean_6weights,
            "mean_across_6weights_std":  std_6weights,
            "peak_seed_idx":         peak_seed,
            "peak_reward":           float(all_episodes.mean()),
            "peak_reward_std":       float(all_episodes.std()),
            "n_neurons":             n_neurons,
            "n_connections":         n_connections,
        }
        with lock:
            done[0] += 1
            sweep_str = f"{mean_6weights:.4f}" if mean_6weights is not None else "FAIL"
            print(f"  [{done[0]:3d}/{len(models)}] {w['run_key']} seed_idx={w['seed_idx']}  "
                  f"peso={row['weight_value']:g}  training_fitPeak={w['fitPeak']:.4f}  "
                  f"reward(mejor peso)={row['reward']:.4f}±{row['reward_std']:.4f}  "
                  f"reward(6 pesos)={sweep_str}")
        return row, ep_df

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        for row, ep_df in pool.map(worker, models):
            if row is not None:
                rows.append(row)
                episode_dfs.append(ep_df)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_path = out_dir / f"{task}_best.csv"
    df.to_csv(best_path, index=False)
    print(f"[{task}] modelos de las {len(groups)} combinaciones encoder/decoder → {best_path}  "
          f"({len(df)}/{len(models)} exitosos)")

    if episode_dfs:
        episodes_path = out_dir / f"{task}_best_episodes.csv"
        pd.concat(episode_dfs, ignore_index=True).to_csv(episodes_path, index=False)
        print(f"[{task}] episodios individuales → {episodes_path}")

    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=list(TASKS), default=None,
                    help="Solo esta tarea (default: las 3)")
    ap.add_argument("--seeds", type=int, default=11,
                    help="Cantidad de seeds de revalidación (default: 11)")
    ap.add_argument("--seed0", type=int, default=0,
                    help="Primera seed de evaluación (default: 0)")
    ap.add_argument("--nreps", type=int, default=11,
                    help="Episodios promediados por seed (default: 11). "
                         "Sobreescribe alg_nReps SOLO para esta evaluación "
                         "(--nreps del binario) — no toca p/*.json ni el "
                         "entrenamiento.")
    ap.add_argument("--jobs", type=int, default=4,
                    help="Modelos (seeds de entrenamiento) evaluados en paralelo (default: 4)")
    ap.add_argument("--omp", type=int, default=None,
                    help="OMP_NUM_THREADS por corrida (default: cpu_count // jobs)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Timeout por run_key en segundos (default: sin límite)")
    ap.add_argument("--out-dir", default="eval_p3_weights", dest="out_dir",
                    help="Directorio de salida (default: eval_p3_weights/)")
    ap.add_argument("--reward", choices=["shaped", "original"], default=None,
                    help="Forzar shaped u original para todas las tareas corridas "
                         "(default por tarea: acrobot=original, mountain_car=original, car=shaped)")
    ap.add_argument("--run-key", action="append", default=None, dest="run_keys",
                    help="Restringe a este/estos run_key(s) exactos (ej. "
                         "car_ttfs_first_spike_wide) en vez de TODOS los "
                         "screening_full/<prefix>_* de la tarea — repetir la "
                         "opción para pasar varios. Requiere --task. Útil para "
                         "revalidar una corrida ad-hoc sin re-evaluar las "
                         "combinaciones oficiales.")
    args = ap.parse_args()

    if args.run_keys and not args.task:
        ap.error("--run-key requiere --task (para saber a qué TASKS[...] pertenece)")

    omp   = args.omp or max(1, (os.cpu_count() or 4) // args.jobs)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    tasks = [args.task] if args.task else list(TASKS)
    out_dir = Path(args.out_dir)
    run_keys = set(args.run_keys) if args.run_keys else None

    dfs: list[pd.DataFrame] = []
    for task in tasks:
        dfs.append(run_task(task, seeds, args.nreps, args.jobs, omp,
                            args.timeout, out_dir, args.reward, run_keys))

    print(f"\n{'='*66}")
    print("  Modelos revalidados: por cada run_key (encoder/decoder), su")
    print("  propio rank de mayor fitPeak promedio, uno por seed de entrenamiento")
    print(f"{'='*66}")
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    if all_df.empty:
        print("  (sin resultados)")
    for _, b in all_df.iterrows():
        print(f"  [{b['task']}] run_key={b['run_key']}  rank={b['rank']}  "
              f"seed_idx={b['seed_idx']}  peso={b['weight_value']:g}  "
              f"training_fitPeak={b['training_fitPeak']:.4f}  "
              f"reward={b['reward']:.4f} ± {b['reward_std']:.4f}")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
