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
import collections
import graphviz
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
parser.add_argument("--title",   default=None,
                    help="Prefijo a mostrar en los títulos de los gráficos (por defecto, se deriva de --prefix)")
args = parser.parse_args()

PREFIX   = args.prefix
N_INPUT  = args.nInput
N_OUTPUT = args.nOutput
TASK     = args.title if args.title is not None else os.path.basename(PREFIX)

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
ax.legend(fontsize=13)
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
ax.legend(lns, [l.get_label() for l in lns], fontsize=13)
ax.grid(True, alpha=0.3)

plt.tight_layout()
if args.save:
    out = PREFIX + "_training.png"
    plt.savefig(out, dpi=150); print(f"Guardado: {out}")
else:
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 1b. Acercamiento a la región inicial donde ocurre la mejora (figura aparte):
#     el resto del entrenamiento suele quedar plano y no aporta información
#     visual, comprimiendo la parte relevante de la curva anterior.
# ─────────────────────────────────────────────────────────────────────────────
init_val, final_val = fit_best[0], fit_best[-1]
rng = final_val - init_val
if abs(rng) > 1e-9 and len(gens) > 2:
    threshold = init_val + 0.95 * rng
    idx95 = int(np.argmax(fit_best >= threshold)) if rng > 0 else int(np.argmax(fit_best <= threshold))
    zoom_end = min(len(gens) - 1, max(idx95 + 5, int(idx95 * 1.3)))
    if zoom_end > 2:
        sl = slice(0, zoom_end + 1)
        fig_z, ax_z = plt.subplots(figsize=(8, 4.5))
        ax_z.plot(gens[sl], fit_med[sl],   label="Mediana población", color="steelblue",  alpha=0.5, linewidth=1)
        ax_z.plot(gens[sl], fit_elite[sl], label="Elite (mejor gen.)", color="darkorange", linewidth=1.2)
        ax_z.plot(gens[sl], fit_best[sl],  label="Best (récord)",      color="green",      linewidth=1.5)
        ax_z.plot(gens[sl], fit_peak[sl],  label="Peak (mejor peso)",  color="red",        linewidth=1.5, linestyle="--")
        ax_z.set_xlim(0, zoom_end)
        ax_z.set_xlabel("Generación")
        ax_z.set_ylabel("Fitness (reward)")
        ax_z.set_title(f"{TASK} — Acercamiento a las primeras {zoom_end} generaciones", fontsize=11)
        ax_z.legend(fontsize=10)
        ax_z.grid(True, alpha=0.3)
        plt.tight_layout()
        if args.save:
            out_z = PREFIX + "_training_zoom.png"
            plt.savefig(out_z, dpi=150); print(f"Guardado: {out_z}")
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
    ax.legend(fontsize=13)
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

    n_conn = int(np.sum(~np.isnan(W) & (W != 0.0)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{TASK} — Mejor individuo (N={N} nodos, {n_conn} conexiones)", fontsize=12)

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

    # Grafo por capas — Graphviz con profundidad topológica real
    node_type = {0: "bias"}
    for i in range(1, 1 + N_INPUT):
        node_type[i] = "input"
    for i in range(1 + N_INPUT, 1 + N_INPUT + n_hid):
        node_type[i] = "hidden"
    for i in range(1 + N_INPUT + n_hid, N):
        node_type[i] = "output"

    output_set = {i for i, t in node_type.items() if t == "output"}
    layer_fill = {"bias": "gold", "input": "cornflowerblue",
                  "hidden": "mediumseagreen", "output": "tomato"}

    # Longest-path depth por orden topológico (Kahn)
    succs_map = [[] for _ in range(N)]
    in_deg    = [0] * N
    for src in range(N):
        for dst in range(N):
            if not np.isnan(W[src, dst]) and W[src, dst] != 0.0:
                succs_map[src].append(dst)
                in_deg[dst] += 1

    queue = collections.deque(n for n in range(N) if in_deg[n] == 0)
    depth = [0] * N
    while queue:
        u = queue.popleft()
        for v in succs_map[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # Las salidas siempre van al frente, más allá de cualquier nodo no-salida
    max_pre = max((depth[n] for n in range(N) if n not in output_set), default=0)
    for n in output_set:
        depth[n] = max_pre + 1

    # Agrupar nodos por su profundidad
    layer_groups: dict[int, list] = collections.defaultdict(list)
    for node in range(N):
        layer_groups[depth[node]].append(node)

    dot = graphviz.Digraph(engine="dot")
    dot.attr(rankdir="LR", bgcolor="white", size="8,6!")
    dot.attr("node", shape="circle", style="filled", fontsize="9",
             width="0.5", fixedsize="true")

    for layer_idx in sorted(layer_groups):
        with dot.subgraph() as sg:
            sg.attr(rank="same")
            for idx in layer_groups[layer_idx]:
                lbl   = node_labels[idx] if idx < len(node_labels) else str(idx)
                aname = act_names.get(act[idx], "?")
                sg.node(str(idx), label=f"{lbl}\n{aname}",
                        fillcolor=layer_fill[node_type[idx]])

    max_w = float(np.nanmax(np.abs(W))) or 1.0
    for src in range(N):
        for dst in range(N):
            w = W[src, dst]
            if np.isnan(w) or w == 0.0:
                continue
            color = "crimson" if w > 0 else "navy"
            pw    = f"{max(0.5, 3.0 * abs(w) / max_w):.2f}"
            dot.edge(str(src), str(dst), color=color, penwidth=pw)

    # Pedir posiciones al engine de Graphviz sin rasterizar (formato plain)
    plain = dot.pipe(format="plain").decode()
    node_pos  = {}
    node_size = {}
    for line in plain.split("\n"):
        parts = line.split()
        if len(parts) >= 5 and parts[0] == "node":
            try:
                n = int(parts[1])
                node_pos[n]  = (float(parts[2]), float(parts[3]))
                node_size[n] = float(parts[4]) / 2.0   # radio = ancho / 2 (pulgadas)
            except ValueError:
                pass

    ax = axes[1]
    ax.axis("off")
    ax.set_title("Topología de la red")

    # Aristas (debajo de los nodos)
    for src in range(N):
        for dst in range(N):
            w = W[src, dst]
            if np.isnan(w) or w == 0.0:
                continue
            if src not in node_pos or dst not in node_pos:
                continue
            x0, y0 = node_pos[src]
            x1, y1 = node_pos[dst]
            r_src  = node_size.get(src, 0.25)
            r_dst  = node_size.get(dst, 0.25)
            dx, dy = x1 - x0, y1 - y0
            dist   = np.hypot(dx, dy)
            color  = "crimson" if w > 0 else "navy"
            lw     = max(0.4, 2.0 * abs(w) / max_w)
            if dist < 1e-6:
                ax.add_patch(plt.Circle((x0 + r_src * 1.2, y0), r_src * 0.6,
                                        fill=False, lw=lw, color=color, zorder=1))
                continue
            xs = x0 + dx / dist * r_src
            ys = y0 + dy / dist * r_src
            xe = x1 - dx / dist * r_dst
            ye = y1 - dy / dist * r_dst
            ax.annotate("", xy=(xe, ye), xytext=(xs, ys),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                        mutation_scale=8))

    # Nodos (encima de las aristas)
    for idx, (x, y) in sorted(node_pos.items()):
        r      = node_size.get(idx, 0.25)
        fcolor = layer_fill[node_type.get(idx, "hidden")]
        ax.add_patch(plt.Circle((x, y), r, color=fcolor, zorder=3,
                                linewidth=0.6, edgecolor="grey"))
        lbl   = node_labels[idx] if idx < len(node_labels) else str(idx)
        aname = act_names.get(act[idx], "?")
        ax.text(x, y + 0.04, lbl,   ha="center", va="center",
                fontsize=6, zorder=4, fontweight="bold")
        ax.text(x, y - 0.12, aname, ha="center", va="top",
                fontsize=5, zorder=4, color="dimgrey")

    if node_pos:
        xs_all = [p[0] for p in node_pos.values()]
        ys_all = [p[1] for p in node_pos.values()]
        pad    = max(node_size.values(), default=0.25) + 0.3
        ax.set_xlim(min(xs_all) - pad, max(xs_all) + pad)
        ax.set_ylim(min(ys_all) - pad, max(ys_all) + pad)
    ax.set_aspect("equal")
    ax.legend(handles=[
        mpatches.Patch(color="gold",           label="Bias"),
        mpatches.Patch(color="cornflowerblue", label="Input"),
        mpatches.Patch(color="mediumseagreen", label="Hidden"),
        mpatches.Patch(color="tomato",         label="Output"),
    ], loc="lower right", fontsize=12)

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
    axes[1].legend(fontsize=12)

    plt.tight_layout()
    if args.save:
        out = PREFIX + "_final_pop.png"
        plt.savefig(out, dpi=150); print(f"Guardado: {out}")
    else:
        plt.show()
