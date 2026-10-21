#!/usr/bin/env python3
"""
Mapa de calor escala de ruido × tiempo desde la programación, para UN modelo
de ruido (pcm, reram o reramcmo — "simple" no depende del tiempo, no aplica).
Muestra interacciones que dos curvas 1D por separado (plot_car_noise.py,
plot_car_noise_time.py) no pueden: por ejemplo, si la deriva solo empieza a
importar cuando el ruido de programación ya es alto.

Necesita un CSV de grilla completa (wann_car_noise corrido con la lista
completa de --t-inference en cada valor de --noise-scale) — ver
run_car_noise_sweeps.py --grid. Texto del gráfico en inglés y estilo de
publicación (fuentes grandes, dpi 300) vía plot_style.py — viridis ya es
seguro para daltonismo, no hace falta cambiarlo.

Uso:
  python eval_results/plot_car_noise_heatmap.py \
      --csv log/car_ttfs_first_spike_grid_sweep.csv --noise-model pcm \
      --out "graficos/Racing Car/car_ttfs_first_spike_heatmap_pcm.png"
"""
import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR.parent)
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from summarize_car_noise import load_rows, summarize
from plot_style import MODEL_LABELS as LABELS, TIME_TICK_LABELS as TIME_TICKS, setup_publication_style


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                     help="uno o más CSV crudos de wann_car_noise con grilla completa (se concatenan)")
    ap.add_argument("--noise-model", required=True, choices=["pcm", "reram", "reramcmo"])
    ap.add_argument("--out", required=True, help="ruta del PNG de salida")
    ap.add_argument("--reward", choices=["shaped", "original"], default="shaped")
    ap.add_argument("--weight", type=float, default=None,
                     help="si el CSV tiene varios pesos (--weight-indices), elegir cuál graficar; "
                          "por defecto usa el primero que encuentra")
    ap.add_argument("--title", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    setup_publication_style()
    df = load_rows(args.csv)
    df = df[df["noise_model"] == args.noise_model]
    if df.empty:
        sys.exit(f"Error: no hay filas con noise_model={args.noise_model} en el CSV.")

    weights = sorted(df["weight"].unique())
    weight = args.weight if args.weight is not None else weights[0]
    if len(weights) > 1 and args.weight is None:
        print(f"Aviso: el CSV tiene varios pesos {weights}; graficando weight={weight} "
              f"(pasar --weight para elegir otro)", file=sys.stderr)
    df = df[df["weight"] == weight]

    summary = summarize(df)
    col = f"reward_{args.reward}_mean"

    scales = sorted(summary["prog_noise_scale"].unique())
    times = sorted(summary["t_inference"].unique())
    grid = np.full((len(times), len(scales)), np.nan)
    for _, row in summary.iterrows():
        i = times.index(row["t_inference"])
        j = scales.index(row["prog_noise_scale"])
        grid[i, j] = row[col]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.pcolormesh(np.arange(len(scales) + 1), np.arange(len(times) + 1), grid,
                       cmap="viridis", shading="flat")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"Mean episode reward ({args.reward})")

    ax.set_xticks(np.arange(len(scales)) + 0.5)
    ax.set_xticklabels([f"{s:g}×" for s in scales])
    ax.set_yticks(np.arange(len(times)) + 0.5)
    ax.set_yticklabels([TIME_TICKS.get(t, f"{t:g}s") for t in times])
    ax.set_xlabel("Noise scale")
    ax.set_ylabel("Time since programming")

    label = LABELS[args.noise_model]
    ax.set_title(args.title or f"{label}: reward vs. noise scale and time")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Guardado: {out_path}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
