#!/usr/bin/env python3
"""
Grafica cómo cambia la recompensa con el TIEMPO transcurrido desde la
programación (--t-inference), a intensidad de ruido fija (--noise-scale no
varía entre puntos — por defecto 1.0, el ajuste de IBM/literatura). Es el
otro corte posible del barrido de wann_car_noise, complementario a
plot_car_noise.py (que fija t_inference y varía --noise-scale).

El modelo "simple" (DeviceModel/NoisyCrossbar) queda afuera a propósito: no
tiene ningún término de deriva — su read_weights() ignora t_inference por
completo — así que un barrido de tiempo le daría una línea plana sin
significado (sólo el ruido de lectura re-muestreado, sin relación con el
tiempo). PCM y ReRAM CMO/HfOx tienen deriva continua (cualquier t_inference
sirve, aunque en ReRAM CMO/HfOx específicamente t_inference=0 es un bypass
total: devuelve el valor ya programado sin cambios, ni deriva ni ruido de
lectura — distinto de PCM, donde incluso a t=0 hay algo de ruido de lectura
base). ReRAM Wan2022 solo tiene 3 instantes medidos, sin ley continua ni
bypass — pasar cualquier otro valor lanza una excepción.

Eje X en escala symlog (lineal cerca de 0, log más allá de linthresh) porque
los t_inference típicos van de 0 a ~10^6-10^7 segundos y "0" no tiene
logaritmo. Texto del gráfico en inglés y estilo de publicación (fuentes
grandes, dpi 300, paleta segura para daltonismo) vía plot_style.py.

Uso:
  python eval_results/plot_car_noise_time.py \
      --csv log/car_ttfs_first_spike_time_sweep.csv \
      --out "graficos/Racing Car/car_ttfs_first_spike_time_robustness.png"
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

from summarize_car_noise import load_rows, summarize
from plot_style import (MODEL_COLORS as COLORS, MODEL_LABELS as LABELS,
                        TIME_TICK_LABELS as TIME_TICKS, setup_publication_style)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                     help="uno o más CSV crudos de wann_car_noise (se concatenan)")
    ap.add_argument("--out", required=True, help="ruta del PNG de salida")
    ap.add_argument("--reward", choices=["shaped", "original"], default="shaped")
    ap.add_argument("--title", default="Reward vs. time since programming")
    ap.add_argument("--show", action="store_true",
                     help="además de guardar, abrir una ventana interactiva")
    args = ap.parse_args()

    setup_publication_style()
    df = load_rows(args.csv)
    df = df[df["noise_model"] != "simple"]
    if df.empty:
        sys.exit("Error: no quedaron filas tras excluir noise_model=simple "
                  "(¿el CSV solo tenía ese modelo?).")

    summary = summarize(df)
    col = f"reward_{args.reward}"

    fig, ax = plt.subplots(figsize=(8, 5.5))
    used_ticks = set()
    for model, group in summary.groupby("noise_model"):
        group = group.sort_values("t_inference")
        color = COLORS.get(model)
        label = LABELS.get(model, model)
        ax.plot(group["t_inference"], group[f"{col}_mean"], "-o", color=color,
                linewidth=2, markersize=6, label=label)
        ax.fill_between(group["t_inference"],
                         group[f"{col}_mean"] - group[f"{col}_std"],
                         group[f"{col}_mean"] + group[f"{col}_std"],
                         color=color, alpha=0.15, linewidth=0)
        used_ticks.update(group["t_inference"].tolist())

    LINTHRESH = 60
    ax.set_xscale("symlog", linthresh=LINTHRESH)

    # Funde ticks que caen a una distancia menor a linthresh (p. ej. t=0 de
    # PCM y t=1s de ReRAM): a esta escala son indistinguibles en pantalla y
    # dos etiquetas tan juntas solo se superponen. Se queda con una por grupo
    # (preferencia: el valor con etiqueta conocida en TIME_TICKS, si no el
    # más chico).
    sorted_ticks = sorted(used_ticks)
    groups, current = [], [sorted_ticks[0]]
    for t in sorted_ticks[1:]:
        if t - current[-1] < LINTHRESH:
            current.append(t)
        else:
            groups.append(current)
            current = [t]
    groups.append(current)

    ticks, labels = [], []
    for g in groups:
        rep = next((t for t in g if t in TIME_TICKS), g[0])
        ticks.append(rep)
        labels.append(TIME_TICKS.get(rep, str(rep)))
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    # "1 day"/"2 days" quedan a sólo 0.3 décadas de distancia (86400 vs
    # 172800s) y sus textos colisionan en horizontal sin esto.
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.set_xlabel("Time since programming")
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
