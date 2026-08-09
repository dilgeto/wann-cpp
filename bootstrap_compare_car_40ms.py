#!/usr/bin/env python3
"""
bootstrap_compare_car_40ms.py — Compara el agente SNN campeón de la corrida
"car_ttfs_first_spike_40ms" (misma configuración ganadora de Fase 2 que el
campeón oficial car_ttfs_first_spike, pero con SIM_WINDOW_MS=40.0 en vez de
20.0 — ver WANN_CAR_SIM_WINDOW_MS en include/wann/SnnCarTask.h y
run_car_40ms_phase3.sh) contra el agente ANN (PPO nativo, rl-tools/C++)
mediante bootstrap NO pareado.

No se entrena nada acá: solo se evalúan las dos políticas ya entrenadas.
Los agentes corren sobre implementaciones de entorno distintas (SNN vía el
binario wann_eval_weights_car, ANN vía el binario evaluate de
racing-car-ppo), así que las muestras se tratan como independientes — no hay
pareo episodio a episodio, aunque por defecto se usan los mismos números de
seed para ambos. Ambos usan la MISMA física/recompensa de fondo (rl-tools
CarTrack), y para esta tarea "shaped" == "original" (SnnCarTask.cpp no aplica
potential-based shaping, a diferencia de Acrobot/Mountain Car), así que la
comparación es directa.

Modelo SNN: dentro de run_key=car_ttfs_first_spike_40ms (única combinación
corrida), el modelo específico (una de las 11 seeds de entrenamiento) con
mayor "reward" — la revalidación fresca de eval_p3_weights.py --run-key
car_ttfs_first_spike_40ms --out-dir eval_p3_weights_40ms (promedio a través
de 11 seeds de evaluación):
  run_key=car_ttfs_first_spike_40ms  rank=0  seed_idx=4  peso=2.0
  (n_neurons=35, n_connections=56 — ver eval_p3_weights_40ms/car_best.csv)
Modelo ANN: ../racing-car-ppo/models/model_seed10.h5 (PPO nativo rl-tools,
evaluado con el binario ya compilado ../racing-car-ppo/build/evaluate).

IMPORTANTE: SNN_EXECUTABLE (./build/wann_eval_weights_car) usa actualmente
SIM_WINDOW_MS=40.0 como default (ver include/wann/SnnCarTask.h) — es el mismo
binario que evaluaría el campeón oficial car_ttfs_first_spike (entrenado a
20ms), así que NO usar este binario para revalidar ese modelo sin antes
volver el default a 20.0 y recompilar.

La lógica de bootstrap/estadística/gráficos/guardado es compartida — ver
bootstrap_compare_lib.py.

Uso:
  python bootstrap_compare_car_40ms.py                       # N=100, 10000 resamples
  python bootstrap_compare_car_40ms.py --n 200 --resamples 20000
  python bootstrap_compare_car_40ms.py --out-dir mis_resultados/
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

# ── Modelo SNN ganador de la corrida "40ms" (ver eval_p3_weights_40ms/car_best.csv) ─
SNN_EXECUTABLE  = "./build/wann_eval_weights_car"
SNN_BASE_CONFIG = "p/car_snn.json"
SNN_RUN_KEY     = "car_ttfs_first_spike_40ms"
SNN_RANK        = 0
SNN_SEED_IDX    = 4
SNN_MODEL       = Path(f"log/full_p3_{SNN_RUN_KEY}/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}_best.out")
SNN_OVERRIDE    = Path(f"screening_full/{SNN_RUN_KEY}/p3_configs/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}.json")
SNN_WEIGHT_INDEX = 2     # índice de peso — el de mayor reward (revalidación fresca) entre los 11 modelos
SNN_WEIGHT_VALUE = 2.0   # valor real del peso compartido (weight_vals[SNN_WEIGHT_INDEX])
SNN_REWARD_TYPE  = "shaped"   # == "original" para Car (sin potential-based shaping)

# ── Modelo ANN (PPO nativo rl-tools, racing-car-ppo) ─────────────────────────
ANN_EVAL_BIN = "../racing-car-ppo/build/evaluate"
ANN_MODEL    = "../racing-car-ppo/models/model_seed10.h5"


def eval_snn(n: int, seed0: int, omp: int, timeout: int | None) -> np.ndarray:
    """Corre n episodios de la SNN ganadora — un episodio por seed
    (--nreps 1), con el peso que el entrenamiento registró como mejor."""
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
    ap = build_arg_parser(__doc__, "bootstrap_car_40ms")
    args = ap.parse_args()
    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN ({SNN_RUN_KEY} rank={SNN_RANK} seed_idx={SNN_SEED_IDX} "
          f"peso={SNN_WEIGHT_VALUE:g} [índice {SNN_WEIGHT_INDEX}]): {args.n} episodios...")
    rewards_snn = eval_snn(args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (PPO nativo, {ANN_MODEL}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle="Racing Car (ventana 40ms) — SNN vs ANN (PPO nativo), bootstrap no pareado",
                  plot_stem="car_40ms_snn_vs_ann", ann_label="ANN (PPO)")


if __name__ == "__main__":
    main()
