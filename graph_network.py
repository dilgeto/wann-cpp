#!/usr/bin/env python3
"""
Visualización interactiva de la red WANN con pyvis.

Uso:
    python graph_network.py
    python graph_network.py --prefix log/snn_car --nInput 9 --nOutput 2
"""

import argparse
import os
import collections
import webbrowser
import numpy as np
from pyvis.network import Network

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
args = parser.parse_args()

PREFIX   = args.prefix
N_INPUT  = args.nInput
N_OUTPUT = args.nOutput
TASK     = os.path.basename(PREFIX)

# ─────────────────────────────────────────────────────────────────────────────
# Leer peso matrix
# ─────────────────────────────────────────────────────────────────────────────
best_file = PREFIX + "_best.out"
if not os.path.exists(best_file):
    print(f"No se encontró {best_file}")
    exit(1)

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
    + [f"in{i}"  for i in range(1, N_INPUT + 1)]
    + [f"h{i}"   for i in range(1, n_hid + 1)]
    + [f"out{i}" for i in range(1, N_OUTPUT + 1)]
)

# ─────────────────────────────────────────────────────────────────────────────
# Tipos de nodo y colores
# ─────────────────────────────────────────────────────────────────────────────
node_type = {0: "bias"}
for i in range(1, 1 + N_INPUT):
    node_type[i] = "input"
for i in range(1 + N_INPUT, 1 + N_INPUT + n_hid):
    node_type[i] = "hidden"
for i in range(1 + N_INPUT + n_hid, N):
    node_type[i] = "output"

output_set = {i for i, t in node_type.items() if t == "output"}

layer_color = {
    "bias":   {"background": "#FFD700", "border": "#B8860B"},
    "input":  {"background": "#6495ED", "border": "#1E3A8A"},
    "hidden": {"background": "#3CB371", "border": "#1A5C3A"},
    "output": {"background": "#FF6347", "border": "#8B1A00"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Profundidad topológica (longest-path desde fuentes)
# ─────────────────────────────────────────────────────────────────────────────
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

max_pre = max((depth[n] for n in range(N) if n not in output_set), default=0)
for n in output_set:
    depth[n] = max_pre + 1

# Posiciones en píxeles
layer_groups: dict[int, list] = collections.defaultdict(list)
for node in range(N):
    layer_groups[depth[node]].append(node)

X_SPACING = 220
Y_SPACING = 100
pos: dict[int, tuple] = {}
for d, nodes_in_layer in layer_groups.items():
    total = len(nodes_in_layer)
    for k, idx in enumerate(nodes_in_layer):
        pos[idx] = (d * X_SPACING, (k - (total - 1) / 2.0) * Y_SPACING)

# ─────────────────────────────────────────────────────────────────────────────
# Construir grafo pyvis
# ─────────────────────────────────────────────────────────────────────────────
net = Network(height="800px", width="100%", directed=True,
              bgcolor="#f9f9f9", font_color="#222222")
net.toggle_physics(False)

max_w = float(np.nanmax(np.abs(W))) or 1.0

for idx in range(N):
    x, y  = pos[idx]
    lbl   = node_labels[idx] if idx < len(node_labels) else str(idx)
    aname = act_names.get(act[idx], "?")
    ntype = node_type.get(idx, "hidden")
    col   = layer_color[ntype]
    title = (f"<b>{lbl}</b><br>"
             f"tipo: {ntype}<br>"
             f"neurona SNN: {aname}<br>"
             f"índice: {idx}")
    net.add_node(idx,
                 label=f"{lbl}\n{aname}",
                 title=title,
                 x=x, y=y,
                 color={"background": col["background"],
                        "border":     col["border"],
                        "highlight":  {"background": "#FFFFFF",
                                       "border":     col["border"]}},
                 size=22,
                 font={"size": 10, "face": "monospace", "bold": True},
                 borderWidth=1.5,
                 borderWidthSelected=3,
                 physics=False)

for src in range(N):
    for dst in range(N):
        w = W[src, dst]
        if np.isnan(w) or w == 0.0:
            continue
        color = "#1a3a8a" if w > 0 else "#b22222"
        width = max(0.5, 4.0 * abs(w) / max_w)
        net.add_edge(src, dst,
                     color={"color": color, "highlight": "#FF8C00"},
                     width=width,
                     title=f"peso: {w:+.4f}",
                     arrows="to",
                     smooth={"type": "curvedCW", "roundness": 0.15})

net.set_options("""{
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": {"enabled": true},
    "tooltipDelay": 100
  },
  "layout": {"improvedLayout": false}
}""")

# ─────────────────────────────────────────────────────────────────────────────
# Guardar y abrir
# ─────────────────────────────────────────────────────────────────────────────
out_html = PREFIX + "_network.html"
try:
    net.write_html(out_html)
except AttributeError:
    net.save_graph(out_html)

# Forzar modo claro sin importar la preferencia del navegador/SO
with open(out_html, "r") as f:
    html = f.read()
html = html.replace(
    "<head>",
    '<head>\n  <meta name="color-scheme" content="light">'
    '\n  <style>:root { color-scheme: light; }</style>',
    1,
)
with open(out_html, "w") as f:
    f.write(html)

print(f"Red guardada en: {out_html}")
webbrowser.open("file://" + os.path.abspath(out_html))
