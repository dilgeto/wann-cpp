#!/usr/bin/env python3
"""
Visualización de resultados WANN+SNN.

Uso:
    python graph.py
    python graph.py --prefix log/snn_mountain_car --nInput 2 --nOutput 1
    python graph.py --prefix log/snn_car --nInput 9 --nOutput 2 --save
"""

import argparse
import glob
import os
import re
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # cambiar a "Agg" si no hay display (servidor)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--prefix",  default="log/snn_mountain_car",
                    help="Prefijo de los archivos de log")
parser.add_argument("--nInput",  type=int, default=2,
                    help="Número de entradas del WANN (sin bias)")
parser.add_argument("--nOutput", type=int, default=1,
                    help="Número de salidas del WANN")
parser.add_argument("--save",    action="store_true",
                    help="Guardar figuras como PNG en lugar de mostrarlas")
args = parser.parse_args()

PREFIX   = args.prefix
N_INPUT  = args.nInput
N_OUTPUT = args.nOutput
TASK     = os.path.basename(PREFIX)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Curvas de entrenamiento
# ─────────────────────────────────────────────────────────────────────────────
stats_file = PREFIX + "_stats.out"
if not os.path.exists(stats_file):
    print(f"No se encontró {stats_file}")
    exit(1)

stats = np.loadtxt(stats_file, delimiter=",")
# columnas: evals, fitMed, fitMax(elite), fitTop(best), fitPeak, nodeMed, connMed
gens      = np.arange(len(stats))
fit_med   = stats[:, 1]
fit_elite = stats[:, 2]
fit_best  = stats[:, 3]
fit_peak  = stats[:, 4]
node_med  = stats[:, 5]
conn_med  = stats[:, 6]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle(f"{TASK} — Curvas de entrenamiento", fontsize=12)

ax = axes[0]
ax.plot(gens, fit_med,   label="Mediana población", color="steelblue",  alpha=0.5, linewidth=1)
ax.plot(gens, fit_elite, label="Elite (mejor gen.)", color="darkorange", linewidth=1.2)
ax.plot(gens, fit_best,  label="Best (récord)",      color="green",      linewidth=1.5)
ax.plot(gens, fit_peak,  label="Peak (mejor peso)",  color="red",        linewidth=1.5, linestyle="--")
ax.set_xlabel("Generación")
ax.set_ylabel("Fitness (reward)")
ax.set_title("Fitness a lo largo de la evolución")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax  = axes[1]
ax2 = ax.twinx()
ln1 = ax.plot( gens, node_med, label="Nodos (mediana)", color="purple", linewidth=1.5)
ln2 = ax2.plot(gens, conn_med, label="Conns (mediana)", color="teal",   linewidth=1.5, linestyle="--")
ax.set_xlabel("Generación")
ax.set_ylabel("Nodos", color="purple")
ax2.set_ylabel("Conexiones", color="teal")
ax.set_title("Complejidad de la red (mediana poblacional)")
lns = ln1 + ln2
ax.legend(lns, [l.get_label() for l in lns], fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
if args.save:
    out = PREFIX + "_training.png"
    plt.savefig(out, dpi=150); print(f"Guardado: {out}")
else:
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 2. Evolución del frente de Pareto
# ─────────────────────────────────────────────────────────────────────────────
pareto_dir   = PREFIX + "_pareto"
pareto_files = sorted(glob.glob(os.path.join(pareto_dir, "*.out"))) \
               if os.path.isdir(pareto_dir) else []

if pareto_files:
    n_snap    = min(5, len(pareto_files))
    indices   = np.round(np.linspace(0, len(pareto_files) - 1, n_snap)).astype(int)
    snapshots = [pareto_files[i] for i in indices]
    colors    = plt.cm.viridis(np.linspace(0.1, 0.9, n_snap))

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, fpath in zip(colors, snapshots):
        gen_num = int(re.search(r"(\d+)\.out$", fpath).group(1))
        d = np.loadtxt(fpath, delimiter=",")
        if d.ndim == 1: d = d.reshape(1, -1)
        ax.scatter(d[:, 2], d[:, 0], color=col, alpha=0.4, s=12, label=f"Gen {gen_num}")

    ax.set_xlabel("nConn")
    ax.set_ylabel("Fitness medio")
    ax.set_title(f"{TASK} — Evolución del frente de Pareto")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if args.save:
        out = PREFIX + "_pareto_evolution.png"
        plt.savefig(out, dpi=150); print(f"Guardado: {out}")
    else:
        plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualización del mejor individuo
# ─────────────────────────────────────────────────────────────────────────────
best_file = PREFIX + "_best.out"
if os.path.exists(best_file):
    rows = []
    with open(best_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = []
            for tok in line.split(","):
                tok = tok.strip()
                if tok:
                    vals.append(float("nan") if tok.lower() == "nan" else float(tok))
            if vals:
                rows.append(vals)

    N   = len(rows)
    W   = np.full((N, N), np.nan)
    act = []
    for r, row in enumerate(rows):
        for c in range(N):
            W[r, c] = row[c]
        act.append(int(row[N]))

    act_names = {1: "RS", 2: "FS", 3: "CH", 4: "LTS",
                 5: "IB", 6: "RES", 7: "FS", 8: "RS", 9: "RS", 10: "CH"}
    n_hid = max(N - 1 - N_INPUT - N_OUTPUT, 0)
    node_labels = (
        ["bias"]
        + [f"in{i}" for i in range(1, N_INPUT + 1)]
        + [f"h{i}"  for i in range(1, n_hid + 1)]
        + [f"out{i}" for i in range(1, N_OUTPUT + 1)]
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{TASK} — Mejor individuo (N={N} nodos)", fontsize=12)

    # Heatmap
    ax = axes[0]
    display = np.where(np.isnan(W), 0.0, W)
    im = ax.imshow(display, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for r in range(N):
        for c in range(N):
            if np.isnan(W[r, c]):
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                             facecolor="lightgrey", edgecolor="none"))
    plt.colorbar(im, ax=ax, label="Peso (gris = sin conexión)")
    ax.set_xticks(range(N)); ax.set_xticklabels(node_labels[:N], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(N)); ax.set_yticklabels(node_labels[:N], fontsize=7)
    ax.set_xlabel("Nodo destino"); ax.set_ylabel("Nodo origen")
    ax.set_title("Matriz de conexiones")

    # Grafo por capas
    ax = axes[1]
    ax.axis("off")
    ax.set_title("Topología de la red")

    layer_nodes = {
        "bias":   [0],
        "input":  list(range(1, 1 + N_INPUT)),
        "hidden": list(range(1 + N_INPUT, 1 + N_INPUT + n_hid)),
        "output": list(range(1 + N_INPUT + n_hid, N)),
    }
    layer_x   = {"bias": 0.0, "input": 0.5, "hidden": 1.5, "output": 2.5}
    layer_col = {"bias": "gold", "input": "cornflowerblue",
                 "hidden": "mediumseagreen", "output": "tomato"}

    pos = {}
    for layer, nodes_l in layer_nodes.items():
        xpos  = layer_x[layer]
        total = max(len(nodes_l), 1)
        for k, idx in enumerate(nodes_l):
            ypos       = (k - (total - 1) / 2.0) * 0.7
            pos[idx]   = (xpos, ypos)
            ax.add_patch(plt.Circle((xpos, ypos), 0.18,
                                    color=layer_col[layer], zorder=3))
            lbl = node_labels[idx] if idx < len(node_labels) else str(idx)
            ax.text(xpos, ypos, lbl, ha="center", va="center",
                    fontsize=6, zorder=4, fontweight="bold")
            ax.text(xpos, ypos - 0.27, act_names.get(act[idx], "?"),
                    ha="center", va="top", fontsize=5, color="dimgrey")

    for src in range(N):
        for dst in range(N):
            if np.isnan(W[src, dst]) or W[src, dst] == 0.0:
                continue
            x0, y0 = pos.get(src, (0, 0))
            x1, y1 = pos.get(dst, (0, 0))
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->",
                                        color="navy" if W[src, dst] > 0 else "crimson",
                                        lw=1.0, shrinkA=11, shrinkB=11))

    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    pad   = 0.5
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(color="gold",           label="Bias"),
        mpatches.Patch(color="cornflowerblue", label="Input"),
        mpatches.Patch(color="mediumseagreen", label="Hidden"),
        mpatches.Patch(color="tomato",         label="Output"),
    ], loc="lower right", fontsize=7)

    plt.tight_layout()
    if args.save:
        out = PREFIX + "_network.png"
        plt.savefig(out, dpi=150); print(f"Guardado: {out}")
    else:
        plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 4. Distribución final de la población
# ─────────────────────────────────────────────────────────────────────────────
if pareto_files:
    last_file = pareto_files[-1]
    gen_last  = int(re.search(r"(\d+)\.out$", last_file).group(1))
    d = np.loadtxt(last_file, delimiter=",")
    if d.ndim == 1: d = d.reshape(1, -1)
    fitness = d[:, 0]; fitmax = d[:, 1]; nconn = d[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle(f"{TASK} — Población final (gen {gen_last})", fontsize=12)

    h = axes[0].hist2d(nconn, fitness, bins=20, cmap="YlOrRd")
    plt.colorbar(h[3], ax=axes[0], label="Individuos")
    axes[0].set_xlabel("nConn"); axes[0].set_ylabel("Fitness medio")
    axes[0].set_title("Densidad: fitness vs conectividad")

    sc = axes[1].scatter(fitness, fitmax, s=8, alpha=0.4, c=nconn, cmap="viridis")
    plt.colorbar(sc, ax=axes[1], label="nConn")
    axes[1].set_xlabel("Fitness medio (todos los pesos)")
    axes[1].set_ylabel("Peak fitness (mejor peso)")
    axes[1].set_title("Fitness medio vs peak (color = nConn)")
    axes[1].grid(True, alpha=0.3)
    diag = np.linspace(min(fitness.min(), fitmax.min()),
                       max(fitness.max(), fitmax.max()), 50)
    axes[1].plot(diag, diag, "k--", alpha=0.3, linewidth=0.8, label="mean = peak")
    axes[1].legend(fontsize=7)

    plt.tight_layout()
    if args.save:
        out = PREFIX + "_final_pop.png"
        plt.savefig(out, dpi=150); print(f"Guardado: {out}")
    else:
        plt.show()
