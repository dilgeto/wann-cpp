#!/usr/bin/env python3
"""
bootstrap_compare.py (car_lpp_repro) — bootstrap de uno de los 3 reruns de
run.sh (lexicographic_parsimony=true sobre el mejor config de
car_ttfs_first_spike, seeds 0/100/200) contra el baseline ANN (PPO nativo,
racing-car-ppo). Mismo cálculo que bootstrap_results/bootstrap_compare_car_auto.py
(y comparte bootstrap_compare_lib.run_comparison para CSV/gráfico en el mismo
formato), pero sin pasar por la convención run_key/rank/seed_idx de
screening_full.py — estos repro_car_best_lpp_s* vienen de wann_car directo
(run.sh de este directorio), no de screening_full.py, así que no hay
p3_configs/rankNN_seedNN.json ni log/full_p3_<run_key>/ para revalidar.

No se entrena nada acá: solo evalúa el modelo ya entrenado (log/repro_car_best_lpp_s<seed>_best.out)
contra el ANN.

Uso:
  python experiments/car_lpp_repro/bootstrap_compare.py --seed 100
  python experiments/car_lpp_repro/bootstrap_compare.py --seed 0 --n 200 --resamples 20000
"""
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "bootstrap_results"))
os.chdir(_REPO_ROOT)

from bootstrap_compare_lib import build_arg_parser, run_comparison  # noqa: E402

EXECUTABLE  = "./build/wann_eval_weights_car"
BASE_CONFIG = "p/car_snn.json"
OVERRIDE    = "experiments/car_lpp_repro/override_lpp.json"
REWARD      = "shaped"  # "shaped" == "original" para Car (sin potential-based shaping)

# Mismo baseline ANN fijo que bootstrap_compare_car_auto.py.
ANN_EVAL_BIN = "../racing-car-ppo/build/evaluate"
ANN_MODEL    = "../racing-car-ppo/models/model_seed10.h5"


def eval_snn(train_seed: int, n: int, seed0: int, omp: int, timeout: int | None) -> np.ndarray:
    prefix = f"repro_car_best_lpp_eps1_s{train_seed}"
    model_path = Path("log") / f"{prefix}_best.out"
    wi_path    = Path("log") / f"{prefix}_best.wi"
    if not model_path.exists() or not wi_path.exists():
        sys.exit(f"ERROR: falta {model_path} o {wi_path} — corré "
                 f"experiments/car_lpp_repro/run.sh primero.")
    weight_index = int(wi_path.read_text().strip())

    seeds = ",".join(str(seed0 + i) for i in range(n))
    cmd = [EXECUTABLE, "-f", str(model_path), "-d", BASE_CONFIG, "-p", OVERRIDE,
           "--seeds", seeds, "--reward", REWARD, "--nreps", "1",
           "--episode-detail", "--weight-index", str(weight_index)]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Falló la evaluación de la SNN")

    df = pd.read_csv(io.StringIO(proc.stdout))
    print(f"  peso usado: índice {weight_index} (record de entrenamiento, {prefix}_best.wi)")
    return df["reward"].to_numpy(dtype=float)


def eval_ann(n: int, seed0: int, timeout: int | None) -> np.ndarray:
    rewards = np.empty(n, dtype=float)
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(n):
            seed = seed0 + i
            csv_path = Path(tmpdir) / f"eval_{seed}.csv"
            proc = subprocess.run([ANN_EVAL_BIN, ANN_MODEL, str(csv_path), str(seed)],
                                  capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError(f"Falló la evaluación del PPO (seed {seed})")
            traj = pd.read_csv(csv_path)
            rewards[i] = traj["reward"].sum()
    return rewards


def main() -> None:
    ap = build_arg_parser(__doc__, None)
    ap.add_argument("--seed", type=int, default=100, choices=[0, 100, 200],
                    help="Cuál de los 3 reruns de run.sh (repro_car_best_lpp_eps1_s<seed>), default: 100")
    args = ap.parse_args()
    if args.out_dir is None:
        args.out_dir = f"bootstrap_results/car_lpp_repro_s{args.seed}"

    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN (repro_car_best_lpp_eps1_s{args.seed}): {args.n} episodios...")
    rewards_snn = eval_snn(args.seed, args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (PPO nativo, {ANN_MODEL}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle=f"Racing Car (car_lpp_repro s{args.seed}) — SNN vs ANN (PPO), "
                           f"bootstrap no pareado",
                  plot_stem=f"car_lpp_repro_s{args.seed}_snn_vs_ann",
                  ann_label="PPO", snn_label="SNN")


if __name__ == "__main__":
    main()
