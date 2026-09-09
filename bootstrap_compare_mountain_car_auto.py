#!/usr/bin/env python3
"""
bootstrap_compare_mountain_car_auto.py — dado el run_key de una corrida de
fase 3 de Mountain Car discreto (--task mountain_car en eval_p3_weights.py
mapea al ejecutable/config disc_mc — "Mountain Car" a secas siempre es el
discreto en este proyecto, ver CLAUDE.md), revalida sus modelos, elige el
(seed, peso) ganador, y corre el bootstrap contra el baseline ANN (PPO +
VecNormalize, Stable-Baselines3). Toda la lógica de
revalidación/selección/bootstrap es compartida con Car y Acrobot — ver
bootstrap_auto_lib.py. Acá solo vive lo específico de Mountain Car: cómo
evaluar el PPO (venv aparte con sb3+gymnasium+shimmy, y VecNormalize porque
el PPO se entrenó con norm_obs=True).

No se entrena nada acá: solo se evalúan modelos ya entrenados. Mountain Car
discreto usa reward "original" para esta comparación — ver
TASKS["mountain_car"] en eval_p3_weights.py.

Uso:
  python bootstrap_compare_mountain_car_auto.py --run-key mountain_car_small_first_spike
  python bootstrap_compare_mountain_car_auto.py --run-key mountain_car_ttfs_rate_argmax \\
      --seeds 21 --nreps 21 --n 200 --resamples 20000
"""
import subprocess
import sys

import numpy as np

from bootstrap_auto_lib import build_auto_arg_parser, run_auto_comparison

# ── Modelo ANN de referencia (PPO + VecNormalize, Stable-Baselines3) ─────────
PPO_VENV_PY      = "../ppo-MountainCar-v0/.venv/bin/python3"
PPO_MODEL_ZIP    = "../ppo-MountainCar-v0/ppo-MountainCar-v0.zip"
PPO_VECNORM_PATH = "../ppo-MountainCar-v0/vec_normalize.pkl"
GYM_ENV_ID       = "MountainCar-v0"

# A diferencia del DQN de Acrobot, este PPO se entrenó con VecNormalize
# (norm_obs=True), así que hay que envolver el entorno igual en evaluación —
# de lo contrario la política recibe observaciones fuera de la escala con la
# que fue entrenada y el resultado no es representativo.
_PPO_RUNNER = """
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

model = PPO.load(
    "{model_zip}",
    custom_objects={{
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }},
)

venv = DummyVecEnv([lambda: gym.make("{env_id}")])
venv = VecNormalize.load("{vecnorm_path}", venv)
venv.training = False
venv.norm_reward = False

rewards = []
for i in range({n}):
    seed = {seed0} + i
    venv.seed(seed)
    obs = venv.reset()
    done = False
    total = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_arr, _ = venv.step(action)
        total += float(reward[0])
        done = bool(done_arr[0])
    rewards.append(total)

print(",".join(str(r) for r in rewards))
"""


def make_eval_ann(venv_py: str, model_zip: str, vecnorm_path: str, env_id: str):
    def eval_ann(n: int, seed0: int, timeout: int | None) -> np.ndarray:
        """Corre n episodios del PPO (ANN) vía Stable-Baselines3 + VecNormalize,
        en el venv que tiene sb3+gymnasium+shimmy instalados."""
        script = _PPO_RUNNER.format(model_zip=model_zip, env_id=env_id,
                                    vecnorm_path=vecnorm_path, n=n, seed0=seed0)
        proc = subprocess.run([venv_py, "-c", script],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError("Falló la evaluación del PPO")

        last_line = proc.stdout.strip().splitlines()[-1]
        return np.array([float(x) for x in last_line.split(",")], dtype=float)
    return eval_ann


def main() -> None:
    ap = build_auto_arg_parser(__doc__, "mountain_car")
    ap.add_argument("--ppo-venv-py", default=PPO_VENV_PY, dest="ppo_venv_py",
                    help=f"Python del venv con Stable-Baselines3 (default: {PPO_VENV_PY})")
    ap.add_argument("--ppo-model", default=PPO_MODEL_ZIP, dest="ppo_model",
                    help=f"Modelo PPO .zip (default: {PPO_MODEL_ZIP})")
    ap.add_argument("--vecnorm-path", default=PPO_VECNORM_PATH, dest="vecnorm_path",
                    help=f"vec_normalize.pkl (default: {PPO_VECNORM_PATH})")
    ap.add_argument("--env-id", default=GYM_ENV_ID, dest="env_id",
                    help=f"Id de Gymnasium (default: {GYM_ENV_ID})")
    args = ap.parse_args()

    run_auto_comparison("mountain_car", args,
                        make_eval_ann(args.ppo_venv_py, args.ppo_model,
                                     args.vecnorm_path, args.env_id),
                        ann_desc=f"PPO, {args.ppo_model}",
                        ann_label="PPO", task_label="Mountain Car (discreto)")


if __name__ == "__main__":
    main()
