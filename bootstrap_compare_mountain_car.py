#!/usr/bin/env python3
"""
bootstrap_compare_mountain_car.py — Compara el mejor agente SNN (WANN+SNN) de
Mountain Car discreto contra el agente ANN (PPO, Stable-Baselines3) mediante
bootstrap NO pareado.

No se entrena nada acá: solo se evalúan las dos políticas ya entrenadas.
Los agentes corren sobre implementaciones de entorno distintas (SNN vía
rl-tools, ANN vía Gymnasium), así que las muestras se tratan como
independientes — no hay pareo episodio a episodio, aunque por defecto se usan
los mismos números de seed para ambos.

Modelo SNN: entre las 4 combinaciones encoder/decoder de Mountain Car
discreto (cada una con su propia mejor configuración de hiperparámetros,
ver eval_p3_weights.py), la de mayor "reward" — la revalidación fresca de
eval_p3_weights.py (promedio a través de 11 seeds de evaluación) — de su
propio mejor modelo (una de sus 11 seeds de entrenamiento). NOTA: por un
problema de nombres al correr el entrenamiento, estos modelos quedaron
guardados bajo el prefijo "mountain_car_*" en vez de "disc_mc_*" — son
igualmente el modelo discreto (ann_nOutput=3, ejecutable wann_disc_mc),
confirmado por el usuario.
  run_key=mountain_car_small_first_spike  rank=2  seed_idx=10  peso=1.0
Modelo ANN: ../ppo-MountainCar-v0/ (Stable-Baselines3 PPO + VecNormalize,
vía RL-Zoo/HF hub). A diferencia del DQN de Acrobot, este PPO fue entrenado
con normalize_kwargs={"norm_obs": True}, así que hay que cargar también
vec_normalize.pkl y normalizar las observaciones antes de predict().

La lógica de bootstrap/estadística/gráficos/guardado es compartida — ver
bootstrap_compare_lib.py.

Uso:
  python bootstrap_compare_mountain_car.py                       # N=100, 10000 resamples
  python bootstrap_compare_mountain_car.py --n 200 --resamples 20000
  python bootstrap_compare_mountain_car.py --out-dir mis_resultados/
"""
import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_compare_lib import build_arg_parser, run_comparison

# ── Modelo SNN ganador (ver eval_p3_weights.py / mountain_car_best.csv) ──────
SNN_EXECUTABLE  = "./build/wann_eval_weights_disc_mc"
SNN_BASE_CONFIG = "p/disc_mc_snn.json"
SNN_RUN_KEY     = "mountain_car_small_first_spike"
SNN_RANK        = 2
SNN_SEED_IDX    = 10
SNN_MODEL       = Path(f"log/full_p3_{SNN_RUN_KEY}/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}_best.out")
SNN_OVERRIDE    = Path(f"screening_full/{SNN_RUN_KEY}/p3_configs/rank{SNN_RANK:02d}_seed{SNN_SEED_IDX:02d}.json")
SNN_WEIGHT_INDEX = 1     # índice de peso — el de mayor reward (revalidación fresca) entre los 11 modelos
SNN_WEIGHT_VALUE = 1.0   # valor real del peso compartido (weight_vals[SNN_WEIGHT_INDEX])
SNN_REWARD_TYPE  = "original"

# ── Modelo ANN (PPO + VecNormalize, Stable-Baselines3) ───────────────────────
PPO_VENV_PY       = "/home/dilget/Tesis/ppo-MountainCar-v0/.venv/bin/python3"
PPO_MODEL_ZIP     = "../ppo-MountainCar-v0/ppo-MountainCar-v0.zip"
PPO_VECNORM_PATH  = "../ppo-MountainCar-v0/vec_normalize.pkl"
GYM_ENV_ID        = "MountainCar-v0"

# Script que corre en el venv de Stable-Baselines3 para evaluar el PPO. A
# diferencia del DQN de Acrobot, este modelo se entrenó con VecNormalize
# (norm_obs=True), así que hay que envolver el entorno igual en evaluación
# — de lo contrario la política recibe observaciones fuera de la escala con
# la que fue entrenada y el resultado no es representativo. clip_range
# también se descarta vía custom_objects (bytecode viejo, no hace falta
# para predict()).
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
    """Corre n episodios del PPO (ANN) vía Stable-Baselines3 + VecNormalize,
    en el venv que tiene sb3+gymnasium+shimmy instalados."""
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
    ap = build_arg_parser(__doc__, "bootstrap_mountain_car")
    args = ap.parse_args()
    omp = args.omp or (os.cpu_count() or 4)

    print(f"Evaluando SNN ({SNN_RUN_KEY} rank={SNN_RANK} seed_idx={SNN_SEED_IDX} "
          f"peso={SNN_WEIGHT_VALUE:g} [índice {SNN_WEIGHT_INDEX}]): {args.n} episodios...")
    rewards_snn = eval_snn(args.n, args.seed0, omp, args.timeout)

    print(f"Evaluando ANN (PPO, {PPO_MODEL_ZIP}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_snn, rewards_ann, args,
                  suptitle="Mountain Car discreto — SNN vs ANN (PPO), bootstrap no pareado",
                  plot_stem="mountain_car_snn_vs_ann", ann_label="ANN (PPO)")


if __name__ == "__main__":
    main()
