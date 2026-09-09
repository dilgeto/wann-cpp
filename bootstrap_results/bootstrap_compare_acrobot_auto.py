#!/usr/bin/env python3
"""
bootstrap_compare_acrobot_auto.py — dado el run_key de una corrida de fase 3
de Acrobot, revalida sus modelos, elige el (seed, peso) ganador, y corre el
bootstrap contra el baseline ANN (DQN, Stable-Baselines3). Toda la lógica de
revalidación/selección/bootstrap es compartida con Car y Mountain Car — ver
bootstrap_auto_lib.py. Acá solo vive lo específico de Acrobot: cómo evaluar
el DQN (corre en un venv aparte con sb3+gymnasium+shimmy instalados, vía
subprocess — no el venv de este proyecto).

No se entrena nada acá: solo se evalúan modelos ya entrenados. Acrobot usa
reward "original" (sin potential-based shaping) para esta comparación —
ver TASKS["acrobot"] en eval_p3_weights.py.

Uso:
  python bootstrap_results/bootstrap_compare_acrobot_auto.py --run-key acrobot_small_first_spike
  python bootstrap_results/bootstrap_compare_acrobot_auto.py --run-key acrobot_ttfs_rate_argmax \\
      --seeds 21 --nreps 21 --n 200 --resamples 20000
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

# Ancla CWD y sys.path a la raíz del repo (un nivel arriba de
# bootstrap_results/) — ver el mismo bloque en bootstrap_compare_car_auto.py.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval_results"))  # eval_p3_weights.py vive ahí
os.chdir(_REPO_ROOT)

from bootstrap_auto_lib import build_auto_arg_parser, run_auto_comparison

# ── Modelo ANN de referencia (DQN, Stable-Baselines3) ────────────────────────
# El venv del proyecto no tiene sb3/shimmy — se corre en un venv aparte
# (mismo que usa bootstrap_compare_mountain_car.py) vía subprocess.
DQN_VENV_PY   = "../ppo-MountainCar-v0/.venv/bin/python3"
DQN_MODEL_ZIP = "../dqn-Acrobot-v1/dqn-Acrobot-v1.zip"
GYM_ENV_ID    = "Acrobot-v1"

# Objetos de la corrida original (learning_rate/lr_schedule/exploration_schedule)
# no cargan en Python moderno (bytecode de Python 3.7 viejo) pero no hacen
# falta para predict() — se descartan vía custom_objects.
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


def make_eval_ann(venv_py: str, model_zip: str, env_id: str):
    def eval_ann(n: int, seed0: int, timeout: int | None) -> np.ndarray:
        """Corre n episodios del DQN (ANN) vía Stable-Baselines3, en el venv
        que tiene sb3+gymnasium+shimmy instalados."""
        script = _DQN_RUNNER.format(model_zip=model_zip, env_id=env_id, n=n, seed0=seed0)
        proc = subprocess.run([venv_py, "-c", script],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError("Falló la evaluación del DQN")

        last_line = proc.stdout.strip().splitlines()[-1]
        return np.array([float(x) for x in last_line.split(",")], dtype=float)
    return eval_ann


def main() -> None:
    ap = build_auto_arg_parser(__doc__, "acrobot")
    ap.add_argument("--dqn-venv-py", default=DQN_VENV_PY, dest="dqn_venv_py",
                    help=f"Python del venv con Stable-Baselines3 (default: {DQN_VENV_PY})")
    ap.add_argument("--dqn-model", default=DQN_MODEL_ZIP, dest="dqn_model",
                    help=f"Modelo DQN .zip (default: {DQN_MODEL_ZIP})")
    ap.add_argument("--env-id", default=GYM_ENV_ID, dest="env_id",
                    help=f"Id de Gymnasium (default: {GYM_ENV_ID})")
    args = ap.parse_args()

    run_auto_comparison("acrobot", args,
                        make_eval_ann(args.dqn_venv_py, args.dqn_model, args.env_id),
                        ann_desc=f"DQN, {args.dqn_model}",
                        ann_label="DQN", task_label="Acrobot")


if __name__ == "__main__":
    main()
