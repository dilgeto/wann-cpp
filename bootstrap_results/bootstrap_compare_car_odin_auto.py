#!/usr/bin/env python3
"""
bootstrap_compare_car_odin_auto.py — igual que bootstrap_compare_car_auto.py
(revalida el run_key de Car, elige el (seed, peso) ganador, bootstrap contra
el ANN), pero evalúa el modelo ganador en el core ODIN real (ZCU104/PYNQ) en
vez del simulador de software. Corre EN el board (necesita /dev/mem con
permisos de root para hablarle a ODIN vía OdinDriver, y el overlay ya cargado
— ver odin_bootstrap.py).

La selección del (seed, peso) ganador sigue siendo por software
(wann_eval_weights_car, igual que siempre) — solo el paso final de evaluación
de N episodios cambia de "correr en el simulador" a "exportar + correr en
ODIN". No se entrena nada acá tampoco.

Uso (en el ZCU104, con el overlay ya cargado vía odin_bootstrap.py):
  sudo python3 bootstrap_results/bootstrap_compare_car_odin_auto.py \\
      --run-key car_ttfs_first_spike_pruned --window-ms 20 --n 30
  sudo python3 bootstrap_results/bootstrap_compare_car_odin_auto.py \\
      --run-key car_ttfs_first_spike_40ms --skip-revalidation
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "eval_results"))
os.chdir(_REPO_ROOT)

from bootstrap_auto_lib import build_auto_arg_parser, run_auto_comparison
from bootstrap_compare_car_auto import ANN_EVAL_BIN, ANN_MODEL, make_eval_ann
from eval_p3_weights import TASKS

ODIN_EXPORT_BIN = "./build/wann_car_odin_export"
ODIN_EVAL_BIN   = "./build/wann_car_odin_eval"


def make_eval_odin(window_ms: float):
    def eval_odin(task: str, winner: pd.Series, n: int, seed0: int, omp: int,
                 timeout: int | None) -> np.ndarray:
        """Exporta el modelo ganador a config ODIN y corre n episodios reales
        en hardware — un proceso por corrida (--csv una sola vez con -n n),
        igual de eficiente que el paso equivalente en software ya que el
        cuello de botella real es el hardware, no el overhead de proceso."""
        assert task == "car", "bootstrap_compare_car_odin_auto.py es solo para car"
        for b in (ODIN_EXPORT_BIN, ODIN_EVAL_BIN):
            if not Path(b).exists():
                print(f"ERROR: {b} no existe. Compilar con "
                      f"-DWANN_ODIN_HW=ON primero.", file=sys.stderr)
                sys.exit(1)

        td = TASKS[task]
        run_key, rank, seed_idx = winner["run_key"], int(winner["rank"]), int(winner["seed_idx"])
        weight_index = int(winner["weight_index"])
        model_path = Path("log") / f"full_p3_{run_key}" / f"rank{rank:02d}_seed{seed_idx:02d}_best.out"

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_json = Path(tmpdir) / "odin_config.json"
            export_cmd = [ODIN_EXPORT_BIN, "-f", str(model_path), "-d", td["base_config"],
                         "-w", str(weight_index), "-k", run_key, "-o", str(cfg_json)]
            proc = subprocess.run(export_cmd, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError("Falló wann_car_odin_export (¿nodo con signo "
                                   "mixto sin split, o red > 256 neuronas?)")

            eval_cmd = [ODIN_EVAL_BIN, "-c", str(cfg_json), "-d", td["base_config"],
                       "-n", str(n), "-s", str(seed0), "-m", str(window_ms), "--csv"]
            proc = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=timeout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise RuntimeError("Falló wann_car_odin_eval sobre hardware real "
                                   "(¿corriste como root? ¿odin_bootstrap.py ya cargó "
                                   "el overlay? ¿OdinRegisters.h tiene las direcciones "
                                   "reales?)")
            df = pd.read_csv(pd.io.common.StringIO(proc.stdout))
            return df["reward"].to_numpy(dtype=float)
    return eval_odin


def main() -> None:
    ap = build_auto_arg_parser(__doc__, "car")
    ap.add_argument("--ann-bin", default=ANN_EVAL_BIN, dest="ann_bin")
    ap.add_argument("--ann-model", default=ANN_MODEL, dest="ann_model")
    ap.add_argument("--window-ms", type=float, default=40.0, dest="window_ms",
                    help="SIM_WINDOW_MS del modelo exportado — no viene en "
                         "car_snn.json, es un #define de compilación en "
                         "SnnCarTask.h (default: 40, la base del repo). Un "
                         "modelo podado puede usar otra ventana (p.ej. 20).")
    args = ap.parse_args()

    run_auto_comparison(
        "car", args, make_eval_ann(args.ann_bin, args.ann_model),
        ann_desc=f"PPO nativo, {args.ann_model}", ann_label="PPO", task_label="Racing Car",
        eval_target_fn=make_eval_odin(args.window_ms),
        target_label="ODIN", target_desc="ODIN (hardware real)",
        plot_suffix="odin_vs_ann")


if __name__ == "__main__":
    main()
