#!/usr/bin/env python3
"""
Grafica recompensa vs. RMSE efectivo de los pesos (columna `weight_rmse` de
wann_car_noise: error cuadrático medio entre el peso realmente realizado tras
el ruido y el nominal, normalizado por w_max) — a diferencia de
plot_car_noise.py (que grafica contra noise_scale, una unidad DISTINTA por
modelo), esta es la métrica común entre simple/pcm/reram/reramcmo: un RMSE de
0.1 significa lo mismo sin importar qué modelo lo produjo.

weight_rmse es continuo (cambia con cada sorteo de ruido), así que no se
agrupa como en summarize_car_noise.py — se bina en cuantiles por modelo y se
grafica la media ± desvío de recompensa por bin, con los puntos crudos de
fondo (alpha bajo) para no esconder la dispersión real. Texto del gráfico en
inglés y estilo de publicación (fuentes grandes, dpi 300, paleta segura para
daltonismo) vía plot_style.py.

Uso:
  python eval_results/plot_car_noise_rmse.py \
      --csv log/car_ttfs_first_spike_noise_sweep.csv \
      --out "graficos/Racing Car/car_ttfs_first_spike_rmse_robustness.png"
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
import pandas as pd

from summarize_car_noise import load_rows
from plot_style import MODEL_COLORS as COLORS, MODEL_LABELS as LABELS, setup_publication_style


def binned_trend(x: pd.Series, y: pd.Series, n_bins: int) -> pd.DataFrame:
    """Bins por cuantil de x (no por rango fijo: weight_rmse no se distribuye
    uniforme) -> media/std de x e y por bin."""
    try:
        bin_id = pd.qcut(x, q=n_bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["x_mean", "y_mean", "y_std"])
    g = pd.DataFrame({"x": x, "y": y, "bin": bin_id}).groupby("bin", observed=True)
    out = g.agg(x_mean=("x", "mean"), y_mean=("y", "mean"),
                y_std=("y", lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")))
    return out.sort_values("x_mean").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                     help="uno o más CSV crudos de wann_car_noise (se concatenan)")
    ap.add_argument("--out", required=True, help="ruta del PNG de salida")
    ap.add_argument("--reward", choices=["shaped", "original"], default="shaped")
    ap.add_argument("--bins", type=int, default=12, help="cuantiles de weight_rmse por modelo")
    ap.add_argument("--title", default="Reward vs. effective weight error")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    setup_publication_style()
    df = load_rows(args.csv)
    if "weight_rmse" not in df.columns:
        sys.exit("Error: el CSV no tiene la columna weight_rmse — generado con un "
                  "wann_car_noise viejo. Volvé a correr el barrido con el binario actual.")

    col = f"reward_{args.reward}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for model, group in df.groupby("noise_model"):
        if model == "none":
            continue
        color = COLORS.get(model)
        label = LABELS.get(model, model)
        ax.scatter(group["weight_rmse"], group[col], color=color, alpha=0.08, s=14,
                   linewidths=0)
        trend = binned_trend(group["weight_rmse"], group[col], args.bins)
        if trend.empty:
            continue
        ax.plot(trend["x_mean"], trend["y_mean"], "-o", color=color,
                linewidth=2, markersize=5, label=label)
        ax.fill_between(trend["x_mean"], trend["y_mean"] - trend["y_std"],
                         trend["y_mean"] + trend["y_std"], color=color, alpha=0.15, linewidth=0)

    ax.set_xlabel(r"Effective weight RMSE ($|w_{realized} - w_{nominal}| \, / \, w_{max}$)")
    ax.set_ylabel(f"Episode reward ({args.reward})")
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
