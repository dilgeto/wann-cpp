#!/usr/bin/env python3
"""
bootstrap_compare_car_auto.py — dado el run_key de una corrida de fase 3 de
Racing Car, revalida sus modelos, elige el (seed, peso) ganador, y corre el
bootstrap contra el baseline ANN (PPO nativo rl-tools, racing-car-ppo). Toda
la lógica de revalidación/selección/bootstrap es compartida con Acrobot y
Mountain Car — ver bootstrap_auto_lib.py. Acá solo vive lo específico de
Car: cómo evaluar el ANN (binario compilado ../racing-car-ppo/build/evaluate,
no un modelo de Stable-Baselines3 como las otras dos tareas).

No se entrena nada acá: solo se evalúan modelos ya entrenados. "shaped" ==
"original" para Car (SnnCarTask.cpp no aplica potential-based shaping), así
que la comparación es directa.

Uso:
  python bootstrap_compare_car_auto.py --run-key car_ttfs_first_spike
  python bootstrap_compare_car_auto.py --run-key car_ttfs_first_spike_pop1024_neuro \\
      --seeds 21 --nreps 21 --n 200 --resamples 20000
  python bootstrap_compare_car_auto.py --run-key car_ttfs_first_spike_wide \\
      --skip-revalidation
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_auto_lib import build_auto_arg_parser, run_auto_comparison

# ── Modelo ANN de referencia (PPO nativo rl-tools, racing-car-ppo) ───────────
# Fijo para todas las variantes de Car — mismo baseline en toda comparación.
ANN_EVAL_BIN = "../racing-car-ppo/build/evaluate"
ANN_MODEL    = "../racing-car-ppo/models/model_seed10.h5"


def make_eval_ann(ann_bin: str, ann_model: str):
    def eval_ann(n: int, seed0: int, timeout: int | None) -> np.ndarray:
        """Corre n episodios del PPO nativo (rl-tools) vía el binario evaluate,
        uno por seed: cada corrida escribe una trayectoria a CSV; el retorno
        del episodio es la suma de su columna reward."""
        rewards = np.empty(n, dtype=float)
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n):
                seed = seed0 + i
                csv_path = Path(tmpdir) / f"eval_{seed}.csv"
                proc = subprocess.run([ann_bin, ann_model, str(csv_path), str(seed)],
                                      capture_output=True, text=True, timeout=timeout)
                if proc.returncode != 0:
                    print(proc.stderr, file=sys.stderr)
                    raise RuntimeError(f"Falló la evaluación del PPO (seed {seed})")
                traj = pd.read_csv(csv_path)
                rewards[i] = traj["reward"].sum()
        return rewards
    return eval_ann


def main() -> None:
    ap = build_auto_arg_parser(__doc__, "car")
    ap.add_argument("--ann-bin", default=ANN_EVAL_BIN, dest="ann_bin",
                    help=f"Binario de evaluación del ANN (default: {ANN_EVAL_BIN})")
    ap.add_argument("--ann-model", default=ANN_MODEL, dest="ann_model",
                    help=f"Modelo ANN (default: {ANN_MODEL})")
    args = ap.parse_args()

    run_auto_comparison("car", args, make_eval_ann(args.ann_bin, args.ann_model),
                        ann_desc=f"PPO nativo, {args.ann_model}",
                        ann_label="PPO", task_label="Racing Car")


if __name__ == "__main__":
    main()
