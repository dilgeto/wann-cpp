#!/usr/bin/env python3
"""
graph_mean_training.py — Curvas de entrenamiento promediadas sobre las N
semillas de una configuración (run_key + rank) de Phase 3.

Mismo estilo que la figura "_training.png" de graph.py (fitness a lo largo
de la evolución + complejidad de la red), pero cada curva es el promedio de
todos los rank<rank>_seed*_stats.out encontrados para ese run_key.

Uso:
    python graph_mean_training.py --run-key acrobot_small_first_spike --rank 1 \
        --title "Acrobot Signed+First_spike" \
        --out graficos/Acrobot/acrobot_signed_first_spike_mean_training.png

    # elegir manualmente otra configuración (otro rank) si no se quiere la
    # ganadora de Phase 3 (la que aparece en eval_p3_weights/<task>_best.csv):
    python graph_mean_training.py --run-key acrobot_small_first_spike --rank 0
"""

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--run-key", required=True,
                    help="Nombre del run (ej. acrobot_small_first_spike)")
parser.add_argument("--rank", type=int, required=True,
                    help="Rank de la configuración a promediar (ej. el de la fila "
                         "ganadora en eval_p3_weights/<task>_best.csv)")
parser.add_argument("--log-dir", default="log",
                    help="Directorio base de logs (default: log)")
parser.add_argument("--title", default=None,
                    help="Prefijo a mostrar en el título (default: <run_key> rank<NN>)")
parser.add_argument("--out", default=None,
                    help="Ruta de salida del PNG (default: <run_key>_mean_training.png)")
args = parser.parse_args()

rank_p  = f"{args.rank:02d}"
run_dir = os.path.join(args.log_dir, f"full_p3_{args.run_key}")
pattern = os.path.join(run_dir, f"rank{rank_p}_seed*_stats.out")
files   = sorted(glob.glob(pattern))

if not files:
    print(f"No se encontraron stats en {pattern}")
    raise SystemExit(1)

all_stats = [np.loadtxt(f, delimiter=",") for f in files]
min_len   = min(len(s) for s in all_stats)
stacked   = np.stack([s[:min_len, :7] for s in all_stats])   # (n_seeds, min_len, 7)
mean_stats = stacked.mean(axis=0)

gens      = np.arange(min_len)
fit_med   = mean_stats[:, 1]
fit_elite = mean_stats[:, 2]
fit_best  = mean_stats[:, 3]
fit_peak  = mean_stats[:, 4]
node_med  = mean_stats[:, 5]
conn_med  = mean_stats[:, 6]

TASK = args.title if args.title is not None else f"{args.run_key} rank{rank_p}"
N_SEEDS = len(files)

AXIS_LABEL_FONTSIZE = 13

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"{TASK} — Curvas de entrenamiento (promedio de {N_SEEDS} entrenamientos)",
            fontsize=12)

ax = axes[0]
ax.plot(gens, fit_med,   label="Mediana población", color="steelblue",  alpha=0.5, linewidth=1)
ax.plot(gens, fit_elite, label="Elite (mejor gen.)", color="darkorange", linewidth=1.2)
ax.plot(gens, fit_best,  label="Best (récord)",      color="green",      linewidth=1.5)
ax.plot(gens, fit_peak,  label="Peak (mejor peso)",  color="red",        linewidth=1.5, linestyle="--")
ax.set_xlabel("Generación", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Fitness (reward)", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_title("Fitness a lo largo de la evolución")
ax.legend(fontsize=13)
ax.grid(True, alpha=0.3)

ax  = axes[1]
ax2 = ax.twinx()
ln1 = ax.plot( gens, node_med, label="Nodos (mediana)", color="purple", linewidth=1.5)
ln2 = ax2.plot(gens, conn_med, label="Conns (mediana)", color="teal",   linewidth=1.5, linestyle="--")
ax.set_xlabel("Generación", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Nodos", color="purple", fontsize=AXIS_LABEL_FONTSIZE)
ax2.set_ylabel("Conexiones", color="teal", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_title("Complejidad de la red (mediana poblacional)")
lns = ln1 + ln2
ax.legend(lns, [l.get_label() for l in lns], fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = args.out or f"{args.run_key}_mean_training.png"
plt.savefig(out, dpi=150)
print(f"Guardado: {out}")
