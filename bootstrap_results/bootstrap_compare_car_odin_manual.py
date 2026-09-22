#!/usr/bin/env python3
"""
bootstrap_compare_car_odin_manual.py — bootstrap ODIN vs ANN para UN modelo
car ya elegido a mano (un .out/.wi puntual, p.ej. una red podada a mano que
no viene de un run_key de screening_full.py --mode phase3). A diferencia de
bootstrap_compare_car_odin_auto.py, esto NO revalida ni elige un ganador
entre varios seeds/pesos — asume que ya sabes qué modelo y qué peso querés
evaluar, y va directo al bootstrap. Corre EN el board (root, overlay ya
cargado vía odin_bootstrap.py) — ver bootstrap_compare_car_odin_auto.py para
esos requisitos.

No se entrena nada acá tampoco.

Uso:
  sudo python3 bootstrap_results/bootstrap_compare_car_odin_manual.py \\
      --model-file /home/xilinx/.../ttfs_first_spike_pruned.out \\
      --window-ms 20 --n 30
  # --weight-index explícito en vez de leerlo del .wi de al lado:
  sudo python3 bootstrap_results/bootstrap_compare_car_odin_manual.py \\
      --model-file red.out --weight-index 5 --window-ms 20 --n 30 \\
      --out-dir bootstrap_results/red_pruned_odin
"""
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

from bootstrap_compare_car_auto import ANN_EVAL_BIN, ANN_MODEL, make_eval_ann
from bootstrap_compare_car_odin_auto import run_odin_episodes
from bootstrap_compare_lib import build_arg_parser, run_comparison
from eval_p3_weights import TASKS


def main() -> None:
    ap = build_arg_parser(__doc__, default_out_dir=None)
    ap.add_argument("--model-file", required=True, dest="model_file",
                    help="Red car ya entrenada (.out, formato exportNet/importNet).")
    ap.add_argument("--weight-index", type=int, default=None, dest="weight_index",
                    help="Índice en SnnCarTask::WEIGHT_VALS (0..5) a usar. "
                         "Default: leer <model_file sin extensión>.wi (el "
                         "mismo archivo que escribe DataGatherer::save).")
    ap.add_argument("--base-config", default=TASKS["car"]["base_config"], dest="base_config")
    ap.add_argument("--ann-bin", default=ANN_EVAL_BIN, dest="ann_bin")
    ap.add_argument("--ann-model", default=ANN_MODEL, dest="ann_model")
    ap.add_argument("--window-ms", type=float, default=40.0, dest="window_ms",
                    help="SIM_WINDOW_MS/snn_window_ms con el que se entrenó "
                         "este modelo (default: 40, la base del repo). Un "
                         "modelo podado puede usar otra ventana (p.ej. 20).")
    args = ap.parse_args()

    model_path = Path(args.model_file)
    if not model_path.exists():
        sys.exit(f"ERROR: {model_path} no existe.")

    weight_index = args.weight_index
    if weight_index is None:
        wi_path = model_path.with_suffix(".wi")
        if not wi_path.exists():
            sys.exit(f"ERROR: no se dio --weight-index y no existe {wi_path}. "
                     f"Indicá uno de los dos.")
        weight_index = int(wi_path.read_text().strip())
        print(f"weight_index leído de {wi_path}: {weight_index}")

    if args.out_dir is None:
        args.out_dir = f"bootstrap_results/{model_path.stem}_odin"

    print(f"\n── Evaluando ODIN (hardware real): {model_path}  "
          f"weight_index={weight_index}  window_ms={args.window_ms}  "
          f"{args.n} episodios...")
    rewards_odin = run_odin_episodes(model_path, weight_index, args.base_config,
                                     args.window_ms, args.n, args.seed0, args.timeout)

    eval_ann = make_eval_ann(args.ann_bin, args.ann_model)
    print(f"Evaluando ANN (PPO nativo, {args.ann_model}): {args.n} episodios...")
    rewards_ann = eval_ann(args.n, args.seed0, args.timeout)

    run_comparison(rewards_odin, rewards_ann, args,
                  suptitle=f"Racing Car ({model_path.name}) — ODIN vs ANN (PPO), "
                           f"bootstrap no pareado",
                  plot_stem=f"{model_path.stem}_odin_vs_ann",
                  ann_label="ANN (PPO)", snn_label="ODIN")


if __name__ == "__main__":
    main()
