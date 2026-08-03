#!/usr/bin/env python3
"""
bootstrap_compare_acrobot.py — Compara el mejor agente SNN (WANN+SNN) de
Acrobot contra el agente ANN (DQN, Stable-Baselines3) mediante bootstrap NO
pareado.

No se entrena nada acá: solo se evalúan las dos políticas ya entrenadas.
Los agentes corren sobre implementaciones de entorno distintas (SNN vía
rl-tools, ANN vía Gymnasium), así que las muestras se tratan como
independientes — no hay pareo episodio a episodio, aunque por defecto se usan
los mismos números de seed para ambos.

Modelo SNN: dentro de la configuración ganadora de Acrobot (mayor
training_fitPeak promedio entre configuración de hiperparámetros +
encoder/decoder, ver eval_p3_weights.py), el modelo específico (una de las
11 seeds de entrenamiento) con mayor "reward" — la revalidación fresca de
eval_p3_weights.py (promedio a través de 11 seeds de evaluación), más
robusta que training_fitPeak (una sola medición de una sola generación):
  run_key=acrobot_small_first_spike  rank=1  seed_idx=1  peso=2.5
Modelo ANN: ../dqn-Acrobot-v1/ (Stable-Baselines3 DQN, vía RL-Zoo/HF hub).

La lógica de bootstrap/estadística/gráficos/guardado es compartida — ver
bootstrap_compare_lib.py.

Uso:
  python bootstrap_compare_acrobot.py                       # N=100, 10000 resamples
  python bootstrap_compare_acrobot.py --n 200 --resamples 20000
  python bootstrap_compare_acrobot.py --out-dir mis_resultados/
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_compare_lib import build_arg_parser, run_comparison

# ── Modelo SNN ganador (ver eval_p3_weights.py / acrobot_best.csv) ───────────
SNN_EXECUTABLE  = "./build/wann_eval_weights_acrobot"
SNN_BASE_CONFIG = "p/acrobot_snn.json"
SNN_RUN_KEY     = "acrobot_small_first_spike"
SNN_RANK        = 1
SNN_SEED_IDX    = 1
SNN_MODEL       = Path(f"log/full_p3_{SNN_RUN_KEY}/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}_best.out")
SNN_OVERRIDE    = Path(f"screening_full/{SNN_RUN_KEY}/p3_configs/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}.json")
SNN_WEIGHT_INDEX = 4     # índice de peso — el de mayor reward (revalidación fresca) entre los 11 modelos
SNN_WEIGHT_VALUE = 2.5   # valor real del peso compartido (weight_vals[SNN_WEIGHT_INDEX])
SNN_REWARD_TYPE  = "original"

# ── Modelo ANN (DQN, Stable-Baselines3) ──────────────────────────────────────
DQN_VENV_PY   = "/home/dilget/Tesis/ppo-MountainCar-v0/.venv/bin/python3"
DQN_MODEL_ZIP = "../dqn-Acrobot-v1/dqn-Acrobot-v1.zip"
GYM_ENV_ID    = "Acrobot-v1"

# Script que corre en el venv de Stable-Baselines3 (gym viejo + gymnasium +
# shimmy instalados ahí) para evaluar el DQN. Los objetos de la corrida
# original (learning_rate/lr_schedule/exploration_schedule) no cargan en
# Python moderno (bytecode de Python 3.7) pero no hacen falta para predict().
_DQN_RUNNER = """
import sys
import gymnasium as gym
from stable_baselines3 import DQN

model = DQN.load(
    "{model_zip}",
    custom_objects={{
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
    }},
)
env = gym.make("{env_id}")

rewards = []
for i in range({n}):
    seed = {seed0} + i
    obs, _ = env.reset(seed=seed)
    done = False
    total = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        done = terminated or truncated
    rewards.append(total)

print(",".join(str(r) for r in rewards))
"""


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
    """Corre n episodios del DQN (ANN) vía Stable-Baselines3, en el venv que
    tiene sb3+gymnasium+shimmy instalados (no el mismo que el resto del
    proyecto usa para graficar/analizar)."""
    script = _DQN_RUNNER.format(model_zip=DQN_MODEL_ZIP, env_id=GYM_ENV_ID, n=n, seed0=seed0)
    proc = subprocess.run([DQN_VENV_PY, "-c", script],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Falló la evaluación del DQN")

    last_line = proc.stdout.strip().splitlines()[-1]
    return np.array([float(x) for x in last_line.split(",")], dtype=float)


def main() -> None:
    ap = build_arg_parser(__doc__, "bootstrap_acrobot")
    args = ap.parse_args()
    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN ({SNN_RUN_KEY} rank={SNN_RANK} seed_idx={SNN_SEED_IDX} "
          f"peso={SNN_WEIGHT_VALUE:g} [índice {SNN_WEIGHT_INDEX}]): {args.n} episodios...")
    rewards_snn = eval_snn(args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (DQN, {DQN_MODEL_ZIP}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle="Acrobot — SNN vs ANN (DQN), bootstrap no pareado",
                  plot_stem="acrobot_snn_vs_ann", ann_label="ANN (DQN)")


if __name__ == "__main__":
    main()
