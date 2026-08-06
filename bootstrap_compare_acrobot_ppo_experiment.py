#!/usr/bin/env python3
"""
bootstrap_compare_acrobot_ppo_experiment.py — EXPERIMENTO: igual que
bootstrap_compare_acrobot.py, pero comparando contra PPO (../ppo-Acrobot-v1)
en vez de DQN (../dqn-Acrobot-v1), para ver cuánto cambian los resultados.

No modifica bootstrap_compare_acrobot.py ni ningún archivo de la tesis.

Uso:
  python bootstrap_compare_acrobot_ppo_experiment.py
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_compare_lib import build_arg_parser, run_comparison

# ── Modelo SNN ganador (idéntico a bootstrap_compare_acrobot.py) ────────────
SNN_EXECUTABLE  = "./build/wann_eval_weights_acrobot"
SNN_BASE_CONFIG = "p/acrobot_snn.json"
SNN_RUN_KEY     = "acrobot_small_first_spike"
SNN_RANK        = 1
SNN_SEED_IDX    = 1
SNN_MODEL       = Path(f"log/full_p3_{SNN_RUN_KEY}/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}_best.out")
SNN_OVERRIDE    = Path(f"screening_full/{SNN_RUN_KEY}/p3_configs/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}.json")
SNN_WEIGHT_INDEX = 4
SNN_WEIGHT_VALUE = 2.5
SNN_REWARD_TYPE  = "original"

# ── Modelo ANN (PPO + VecNormalize, Stable-Baselines3) ───────────────────────
PPO_VENV_PY       = "/home/dilget/Tesis/ppo-MountainCar-v0/.venv/bin/python3"
PPO_MODEL_ZIP     = "../ppo-Acrobot-v1/ppo-Acrobot-v1.zip"
PPO_VECNORM_PATH  = "../ppo-Acrobot-v1/vec_normalize.pkl"
GYM_ENV_ID        = "Acrobot-v1"

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


def eval_snn(n: int, seed0: int, omp: int, timeout: int | None) -> np.ndarray:
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
    script = _PPO_RUNNER.format(model_zip=PPO_MODEL_ZIP, env_id=GYM_ENV_ID,
                                vecnorm_path=PPO_VECNORM_PATH, n=n, seed0=seed0)
    proc = subprocess.run([PPO_VENV_PY, "-c", script],
                          capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError("Falló la evaluación del PPO")

    last_line = proc.stdout.strip().splitlines()[-1]
    return np.array([float(x) for x in last_line.split(",")], dtype=float)


def main() -> None:
    ap = build_arg_parser(__doc__, "bootstrap_acrobot_ppo_experiment")
    args = ap.parse_args()
    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN ({SNN_RUN_KEY} rank={SNN_RANK} seed_idx={SNN_SEED_IDX} "
          f"peso={SNN_WEIGHT_VALUE:g} [índice {SNN_WEIGHT_INDEX}]): {args.n} episodios...")
    rewards_snn = eval_snn(args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (PPO, {PPO_MODEL_ZIP}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle="Acrobot — SNN vs ANN (PPO), bootstrap no pareado",
                  plot_stem="acrobot_snn_vs_ppo_experiment", ann_label="ANN (PPO)")


if __name__ == "__main__":
    main()
