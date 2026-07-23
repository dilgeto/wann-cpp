#!/usr/bin/env python3
"""
weight_sweep_eval.py - Evalua una red YA ENTRENADA (best.out de phase3/final)
en cada uno de los 6 pesos del sweep de WANN, usando N seeds/episodios por
peso, y reporta el promedio y std por peso.

Es el "nivel 1" recalculado desde cero para un modelo puntual: el pipeline de
entrenamiento (screening_full.py) descarta el vector de 6 rewards por peso y
solo guarda el promedio final (columna original_fitness / peak_fitness). Este
script reconstruye ese vector llamando al binario wann_*_eval (ya compilado)
una vez por peso, reutilizando evalEpisodes() en vez de reentrenar nada.

Requiere los binarios *_eval compilados en build/ (make/cmake --build ya
corrido) y el archivo de la red guardada (*_best.out de log/full_p3_.../).

Uso:
  # Config de fase 3 (parametros del override) + config base de la tarea
  python weight_sweep_eval.py --task acrobot \
      --net log/full_p3_acrobot_small_rate_argmax/rank00_seed00_best.out \
      --override screening_full/acrobot_small_rate_argmax/p3_configs/rank00_seed00.json \
      --n 10 --seed 0 \
      --out screening_full/acrobot_small_rate_argmax/weight_sweep_best.csv

  # Sin override (usa el config base de la tarea tal cual)
  python weight_sweep_eval.py --task mountain_car \
      --net log/snn_mountain_car_best.out --n 10 --seed 0 --out sweep.csv
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

# ── Definicion por tarea (ejecutable eval, config base, pesos del sweep) ─────
# Los valores de weight_vals deben coincidir exactamente con WEIGHT_VALS en
# el .cpp de cada tarea (Snn*Task.cpp); N_WEIGHTS=6 en las cuatro tareas.
TASK_EVAL: dict[str, dict] = {
    "acrobot": {
        "executable":  "./build/wann_acrobot_eval",
        "base_config": "p/acrobot_snn.json",
        "weight_vals": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
    },
    "car": {
        "executable":  "./build/wann_car_eval",
        "base_config": "p/car_snn.json",
        "weight_vals": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    },
    "mountain_car": {
        "executable":  "./build/wann_mountain_car_eval",
        "base_config": "p/mountain_car_snn.json",
        "weight_vals": [0.5, 1.0, 1.5, 2.0, 5.0, 8.0],
    },
    "disc_mc": {
        "executable":  "./build/wann_disc_mc_eval",
        "base_config": "p/disc_mc_snn.json",
        "weight_vals": [0.5, 1.0, 2.0, 3.0, 5.0, 8.0],
    },
}

_SUMMARY_RE = re.compile(
    r"\[(Shaped|Original)\]\s+Media=(-?[\d.]+)\s+StdDev=(-?[\d.]+)\s+"
    r"Min=(-?[\d.]+)\s+Max=(-?[\d.]+)"
)


def build_merged_config(base_config: str, override: str | None, tmp_path: Path) -> Path:
    """Combina el config base de la tarea con el override de phase3 (si se da)
    y lo escribe a tmp_path. El binario *_eval solo acepta UN archivo (-d)."""
    merged = json.load(open(base_config))
    if override is not None:
        merged.update(json.load(open(override)))
    tmp_path.write_text(json.dumps(merged, indent=2))
    return tmp_path


def run_one_weight(executable: str, net: str, cfg: Path, weight: float,
                    n_episodes: int, seed: int) -> dict:
    cmd = [executable, "-f", net, "-d", str(cfg),
           "-w", str(weight), "-n", str(n_episodes), "-s", str(seed)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{executable} fallo (exit {proc.returncode}) con peso={weight}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    row: dict = {"weight_val": weight}
    for kind, mean, std, mn, mx in _SUMMARY_RE.findall(proc.stdout):
        prefix = "shaped" if kind == "Shaped" else "orig"
        row[f"{prefix}_mean"] = float(mean)
        row[f"{prefix}_std"]  = float(std)
        row[f"{prefix}_min"]  = float(mn)
        row[f"{prefix}_max"]  = float(mx)

    if "shaped_mean" not in row or "orig_mean" not in row:
        raise RuntimeError(
            f"No se pudo parsear el resumen de {executable} para peso={weight}. "
            f"Salida completa:\n{proc.stdout}"
        )
    return row


def sweep(task: str, net: str, override: str | None, base: str | None,
          exe: str | None, n_episodes: int, seed: int) -> pd.DataFrame:
    td = TASK_EVAL[task]
    executable  = exe or td["executable"]
    base_config = base or td["base_config"]

    if not Path(executable).exists():
        print(f"ERROR: {executable} no encontrado. Compila primero (build/).", file=sys.stderr)
        sys.exit(1)
    if not Path(net).exists():
        print(f"ERROR: red no encontrada: {net}", file=sys.stderr)
        sys.exit(1)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        cfg_path = build_merged_config(base_config, override, Path(tf.name))

    rows = []
    for wi, weight in enumerate(td["weight_vals"]):
        print(f"  [{wi+1}/{len(td['weight_vals'])}] peso={weight} "
              f"({n_episodes} episodios, seed base={seed})...")
        row = run_one_weight(executable, net, cfg_path, weight, n_episodes, seed)
        row["weight_idx"] = wi
        rows.append(row)

    cfg_path.unlink(missing_ok=True)

    cols = ["weight_idx", "weight_val",
            "shaped_mean", "shaped_std", "shaped_min", "shaped_max",
            "orig_mean", "orig_std", "orig_min", "orig_max"]
    return pd.DataFrame(rows)[cols]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=list(TASK_EVAL))
    ap.add_argument("--net", required=True, help="Red entrenada (*_best.out)")
    ap.add_argument("--override", default=None,
                     help="JSON de overrides de phase3 (screening_full/.../p3_configs/rankXX_seedXX.json). "
                          "Se funde sobre el config base de la tarea.")
    ap.add_argument("--base", default=None, help="Override del config base de la tarea")
    ap.add_argument("--exe", default=None, help="Override del ejecutable *_eval")
    ap.add_argument("--n", type=int, default=10, help="Episodios (seeds) por peso (default: 10)")
    ap.add_argument("--seed", type=int, default=0, help="Seed base (episodios usan seed..seed+n-1)")
    ap.add_argument("--out", required=True, help="CSV de salida")
    args = ap.parse_args()

    df = sweep(args.task, args.net, args.override, args.base, args.exe,
               args.n, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{df.to_string(index=False, float_format=lambda x: f'{x:.2f}')}")
    print(f"\nResultados -> {out_path}")


if __name__ == "__main__":
    main()
