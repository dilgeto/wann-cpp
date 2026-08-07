#!/usr/bin/env python3
"""
bootstrap_compare_car_no_inhibitory.py — Compara el agente SNN campeón de
Racing Car (car_ttfs_first_spike rank=2 seed_idx=0) contra una VARIANTE del
mismo modelo donde todas las conexiones inhibitorias fueron deshabilitadas
(puestas en 0), dejando el resto de la topología y el peso compartido
intactos. La comparación de desempeño es contra el ANN (PPO nativo), igual
que bootstrap_compare_car.py — así ambos lados quedan en la misma escala y
se puede leer el costo de eliminar la inhibición mirando ambos out-dir.

El archivo del modelo campeón (log/full_p3_car_ttfs_first_spike/
rank02_seed00_best.out) NO se modifica: la variante se generó como una copia
aparte (log/car_ttfs_first_spike_rank02_seed00_no_inhibitory.out) con las 39
entradas -1.0 (inhibitorias) puestas en 0.0 y las 54 entradas +1.0
(excitatorias) intactas — mismo N de nodos (61), mismos NaN estructurales,
mismo vector de activación por nodo.

Escribe en un directorio de salida separado (bootstrap_car_no_inhibitory/)
para no pisar los resultados de bootstrap_compare_car.py (bootstrap_car/).

No se entrena nada acá: solo se evalúa la política ya modificada.

Uso:
  python bootstrap_compare_car_no_inhibitory.py                       # N=100, 10000 resamples
  python bootstrap_compare_car_no_inhibitory.py --n 200 --resamples 20000
  python bootstrap_compare_car_no_inhibitory.py --out-dir mis_resultados/
"""
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_compare_lib import build_arg_parser, run_comparison

# ── Modelo SNN: campeón car_ttfs_first_spike con inhibitorias deshabilitadas ─
SNN_EXECUTABLE  = "./build/wann_eval_weights_car"
SNN_BASE_CONFIG = "p/car_snn.json"
SNN_RUN_KEY     = "car_ttfs_first_spike"
SNN_RANK        = 2
SNN_SEED_IDX    = 0
SNN_LABEL       = "car_ttfs_first_spike rank=2 seed_idx=0 (sin conexiones inhibitorias)"
SNN_MODEL       = Path("log/car_ttfs_first_spike_rank02_seed00_no_inhibitory.out")
SNN_OVERRIDE    = Path(f"screening_full/{SNN_RUN_KEY}/p3_configs/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}.json")
SNN_WEIGHT_INDEX = 5     # mismo peso que el campeón original (ver bootstrap_compare_car.py)
SNN_WEIGHT_VALUE = 8.0   # valor real del peso compartido (weight_vals[SNN_WEIGHT_INDEX])
SNN_REWARD_TYPE  = "shaped"   # == "original" para Car (sin potential-based shaping)

# ── Modelo ANN (PPO nativo rl-tools, racing-car-ppo) ─────────────────────────
ANN_EVAL_BIN = "../racing-car-ppo/build/evaluate"
ANN_MODEL    = "../racing-car-ppo/models/model_seed10.h5"


def eval_snn(n: int, seed0: int, omp: int, timeout: int | None) -> np.ndarray:
    """Corre n episodios de la SNN sin conexiones inhibitorias — un episodio
    por seed (--nreps 1), con el mismo peso compartido que el campeón."""
    seeds = ",".join(str(seed0 + i) for i in range(n))
    cmd = [SNN_EXECUTABLE,
           "-f", str(SNN_MODEL),
           "-d", SNN_BASE_CONFIG,
           "-p", str(SNN_OVERRIDE),
           "--seeds", seeds,
           "--reward", SNN_REWARD_TYPE,
           "--nreps", "1",
           "--episode-detail", "--weight-index", str(SNN_WEIGHT_INDEX)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(omp)

    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Falló la evaluación de la SNN")

    df = pd.read_csv(io.StringIO(proc.stdout))
    return df["reward"].to_numpy(dtype=float)


def eval_ann(n: int, seed0: int, timeout: int | None) -> np.ndarray:
    """Corre n episodios del PPO nativo (rl-tools) vía el binario evaluate,
    uno por seed: cada corrida escribe una trayectoria a CSV; el retorno del
    episodio es la suma de su columna reward (misma convención que
    validate_seeds.sh, que ya usa este mismo binario para reportar
    episode_return)."""
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
    ap = build_arg_parser(__doc__, "bootstrap_car_no_inhibitory")
    args = ap.parse_args()
    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN ({SNN_LABEL}, peso={SNN_WEIGHT_VALUE:g} "
          f"[índice {SNN_WEIGHT_INDEX}]): {args.n} episodios...")
    rewards_snn = eval_snn(args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (PPO nativo, {ANN_MODEL}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle="Racing Car (campeón sin inhibitorias) — SNN vs ANN (PPO nativo), bootstrap no pareado",
                  plot_stem="car_no_inhibitory_snn_vs_ann", ann_label="ANN (PPO)")


if __name__ == "__main__":
    main()
