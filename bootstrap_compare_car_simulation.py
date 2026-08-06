#!/usr/bin/env python3
"""
bootstrap_compare_car_simulation.py — Compara el agente SNN (WANN+SNN) de la
corrida independiente "car_simulation" (log/car_simulation_best.out, wann_car
con la config base p/car_snn.json sin overrides de screening) contra el
agente ANN (PPO nativo, rl-tools/C++) mediante bootstrap NO pareado.

Variante de bootstrap_compare_car.py que usa el modelo "car_simulation" en
vez del campeón de la fase de screening (car_ttfs_first_spike rank=2
seed_idx=0), y escribe en un directorio de salida separado
(bootstrap_car_simulation/) para no pisar los resultados de
bootstrap_compare_car.py (bootstrap_car/).

No se entrena nada acá: solo se evalúan las dos políticas ya entrenadas.
Los agentes corren sobre implementaciones de entorno distintas (SNN vía el
binario wann_eval_weights_car, ANN vía el binario evaluate de
racing-car-ppo), así que las muestras se tratan como independientes — no hay
pareo episodio a episodio, aunque por defecto se usan los mismos números de
seed para ambos. Ambos usan la MISMA física/recompensa de fondo (rl-tools
CarTrack), y para esta tarea "shaped" == "original" (SnnCarTask.cpp no aplica
potential-based shaping, a diferencia de Acrobot/Mountain Car), así que la
comparación es directa.

Modelo SNN: log/car_simulation_best.out, entrenado directamente con
p/car_snn.json (sin overrides de hiperparámetros de screening) vía
`./wann_car -o log/car_simulation`. Peso compartido registrado como mejor:
índice 5 → 8.0 (ver log/car_simulation_best.wi y SnnCarTask::WEIGHT_VALS).
Modelo ANN: ../racing-car-ppo/models/model_seed10.h5 (PPO nativo rl-tools,
evaluado con el binario ya compilado ../racing-car-ppo/build/evaluate).

La lógica de bootstrap/estadística/gráficos/guardado es compartida — ver
bootstrap_compare_lib.py.

Uso:
  python bootstrap_compare_car_simulation.py                       # N=100, 10000 resamples
  python bootstrap_compare_car_simulation.py --n 200 --resamples 20000
  python bootstrap_compare_car_simulation.py --out-dir mis_resultados/
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

# ── Modelo SNN: corrida independiente "car_simulation" ───────────────────────
SNN_EXECUTABLE  = "./build/wann_eval_weights_car"
SNN_BASE_CONFIG = "p/car_snn.json"
SNN_LABEL       = "car_simulation"
SNN_MODEL       = Path("log/car_simulation_best.out")
SNN_WEIGHT_INDEX = 5     # ver log/car_simulation_best.wi
SNN_WEIGHT_VALUE = 8.0   # valor real del peso compartido (weight_vals[SNN_WEIGHT_INDEX])
SNN_REWARD_TYPE  = "shaped"   # == "original" para Car (sin potential-based shaping)

# ── Modelo ANN (PPO nativo rl-tools, racing-car-ppo) ─────────────────────────
ANN_EVAL_BIN = "../racing-car-ppo/build/evaluate"
ANN_MODEL    = "../racing-car-ppo/models/model_seed10.h5"


def eval_snn(n: int, seed0: int, omp: int, timeout: int | None) -> np.ndarray:
    """Corre n episodios de la SNN "car_simulation" — un episodio por seed
    (--nreps 1), con el peso que el entrenamiento registró como mejor."""
    seeds = ",".join(str(seed0 + i) for i in range(n))
    cmd = [SNN_EXECUTABLE,
           "-f", str(SNN_MODEL),
           "-d", SNN_BASE_CONFIG,
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
    ap = build_arg_parser(__doc__, "bootstrap_car_simulation")
    args = ap.parse_args()
    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN ({SNN_LABEL} peso={SNN_WEIGHT_VALUE:g} "
          f"[índice {SNN_WEIGHT_INDEX}]): {args.n} episodios...")
    rewards_snn = eval_snn(args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (PPO nativo, {ANN_MODEL}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle="Racing Car (car_simulation) — SNN vs ANN (PPO nativo), bootstrap no pareado",
                  plot_stem="car_simulation_snn_vs_ann", ann_label="ANN (PPO)")


if __name__ == "__main__":
    main()
