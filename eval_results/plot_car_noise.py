#!/usr/bin/env python3
"""
Grafica la robustez a ruido de memristores de una red car ya barrida con
wann_car_noise (ver BINARIOS.md §4d): recompensa media ± desvío estándar
entre chips, en función de uno de los tres ejes de intensidad (--x-col:
prog_noise_scale por defecto, o read_noise_scale/drift_scale para una
corrida que ablacionó ESE mecanismo en particular — ver --prog-noise-scale/
--read-noise-scale/--drift-scale de wann_car_noise), una línea por
noise_model. Para comparar los tres modelos en una unidad común (no
noise_scale, que no significa lo mismo en cada uno) usar
plot_car_noise_rmse.py en su lugar.

Reusa summarize_car_noise.summarize() para la agregación (mismo cálculo que
el resumen en texto de ese script), así que un cambio ahí se refleja acá sin
duplicar la lógica. Texto del gráfico en inglés y estilo de publicación
(fuentes grandes, dpi 300, paleta segura para daltonismo) vía plot_style.py.

Uso:
  python eval_results/plot_car_noise.py \
      --csv log/car_ttfs_first_spike_noise_sweep.csv \
      --out "graficos/Racing Car/car_ttfs_first_spike_noise_robustness.png"
"""
import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR.parent)           # ancla CWD a la raíz del repo
sys.path.insert(0, str(_SCRIPT_DIR))   # para importar summarize_car_noise/plot_style

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_car_noise import load_rows, summarize
from plot_style import MODEL_COLORS as COLORS, MODEL_LABELS as LABELS, setup_publication_style

X_AXIS_LABELS = {
    "prog_noise_scale": "Programming noise scale (1.0 = IBM/literature default)",
    "read_noise_scale": "Read noise scale (1.0 = IBM/literature default)",
    "drift_scale": "Drift scale (1.0 = IBM/literature default)",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                     help="uno o más CSV crudos de wann_car_noise (se concatenan)")
    ap.add_argument("--out", required=True, help="ruta del PNG de salida")
    ap.add_argument("--reward", choices=["shaped", "original"], default="shaped")
    ap.add_argument("--x-col", default="prog_noise_scale",
                     choices=["prog_noise_scale", "read_noise_scale", "drift_scale"],
                     help="qué eje de intensidad graficar en X (default: prog_noise_scale)")
    ap.add_argument("--title", default="Reward vs. memristor noise intensity")
    ap.add_argument("--show", action="store_true",
                     help="además de guardar, abrir una ventana interactiva")
    args = ap.parse_args()

    setup_publication_style()
    df = load_rows(args.csv)
    summary = summarize(df)
    col = f"reward_{args.reward}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, group in summary.groupby("noise_model"):
        group = group.sort_values(args.x_col)
        color = COLORS.get(model)
        label = LABELS.get(model, model)
        ax.plot(group[args.x_col], group[f"{col}_mean"], "-o", color=color,
                linewidth=2, markersize=5, label=label)
        ax.fill_between(group[args.x_col],
                         group[f"{col}_mean"] - group[f"{col}_std"],
                         group[f"{col}_mean"] + group[f"{col}_std"],
                         color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel(X_AXIS_LABELS[args.x_col])
    ax.set_ylabel(f"Mean episode reward ({args.reward})")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title(args.title)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Guardado: {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
