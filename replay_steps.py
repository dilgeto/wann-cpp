#!/usr/bin/env python3
"""
Figura estática de "steps" de un replay: a la izquierda la trayectoria del
agente con varios instantes superpuestos coloreados con un degradado (color
inicial en el primer paso, color final en el último), a la derecha la
recompensa acumulada con el mismo degradado.

Soporta los tres entornos: MountainCar, Acrobot y Car (rl-tools). El entorno
se infiere del número de columnas del CSV, salvo que se indique --env.

Uso:
    python replay_steps.py --env mountain_car --csv log/snn_mountain_car_best_replay.csv
    python replay_steps.py --env acrobot       --csv log/snn_acrobot_best_replay.csv
    python replay_steps.py --env car           --csv log/snn_car_best_replay.csv
    python replay_steps.py --csv log/snn_car_best_replay.csv --n-steps 10 --save
    python replay_steps.py --csv log/snn_car_best_replay.csv --every 5
    python replay_steps.py --csv log/snn_car_best_replay.csv --color-start "#fde725" --color-end "#440154"
"""

import argparse
import os
import re

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--env", choices=["mountain_car", "acrobot", "car"], default=None,
                    help="Entorno del replay (por defecto se infiere del número de columnas del CSV)")
parser.add_argument("--csv", default=None,
                    help="CSV de trayectoria (un episodio)")
parser.add_argument("--n-steps", type=int, default=8,
                    help="Cantidad de instantes a superponer en la trayectoria (default: 8); "
                         "a mayor cantidad, menor el salto entre instantes")
parser.add_argument("--every", type=int, default=None,
                    help="Salto fijo en pasos entre instantes mostrados (ej. 5 = un instante cada 5 pasos). "
                         "Si se indica, reemplaza --n-steps")
parser.add_argument("--cmap", default="viridis",
                    help="Colormap para el degradado temporal (ignorado si se pasan --color-start/--color-end; default: viridis)")
parser.add_argument("--color-start", default=None,
                    help="Color del primer instante mostrado (junto con --color-end reemplaza --cmap)")
parser.add_argument("--color-end", default=None,
                    help="Color del último instante mostrado (junto con --color-start reemplaza --cmap)")
parser.add_argument("--track", default=None,
                    help="Ruta al track.h de rl-tools (solo env=car)")
parser.add_argument("--title", default=None,
                    help="Prefijo del título de la figura")
parser.add_argument("--save", action="store_true",
                    help="Guardar la figura como PNG en lugar de mostrarla")
parser.add_argument("--out", default=None,
                    help="Ruta de salida del PNG (default: <csv>_steps.png)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Cargar trayectoria
# ─────────────────────────────────────────────────────────────────────────────
ENV_COLS = {5: "mountain_car", 9: "acrobot", 13: "car"}
ENV_DEFAULT_CSV = {
    "mountain_car": "log/snn_mountain_car_best_replay.csv",
    "acrobot":      "log/snn_acrobot_best_replay.csv",
    "car":          "log/snn_car_best_replay.csv",
}

csv_path = args.csv or (ENV_DEFAULT_CSV[args.env] if args.env else None)
if csv_path is None:
    parser.error("Debes indicar --csv (o --env para usar el CSV por defecto)")
if not os.path.exists(csv_path):
    print(f"No se encontró {csv_path}")
    exit(1)

traj = np.loadtxt(csv_path, delimiter=",", skiprows=1)
if traj.ndim == 1:
    traj = traj[np.newaxis, :]

env = args.env or ENV_COLS.get(traj.shape[1])
if env is None:
    parser.error(f"No se pudo inferir --env desde {traj.shape[1]} columnas; indícalo explícitamente")

steps_col = traj[:, 0].astype(int)
reward    = traj[:, -1]
cum_rew   = np.cumsum(reward)
n_frames  = len(traj)

TASK = args.title if args.title is not None else {
    "mountain_car": "MountainCar",
    "acrobot":      "Acrobot",
    "car":          "CarTrack",
}[env]

# ─────────────────────────────────────────────────────────────────────────────
# Instantes a superponer y su color (degradado desde el primer paso mostrado
# hasta el último; más informativo que un degradado de transparencia).
# ─────────────────────────────────────────────────────────────────────────────
if args.every:
    step_gap = max(1, args.every)
    idxs = np.arange(0, n_frames, step_gap)
    if idxs[-1] != n_frames - 1:
        idxs = np.append(idxs, n_frames - 1)
else:
    n_snap = max(2, min(args.n_steps, n_frames))
    idxs   = np.unique(np.linspace(0, n_frames - 1, n_snap).astype(int))

if args.color_start and args.color_end:
    cmap = LinearSegmentedColormap.from_list("steps_gradient", [args.color_start, args.color_end])
else:
    cmap = plt.get_cmap(args.cmap)

colors = cmap(np.linspace(0.0, 1.0, len(idxs)))

# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.6), facecolor="white",
                                        gridspec_kw={"width_ratios": [1.3, 1]})
fig.suptitle(f"{TASK} — Steps de la trayectoria", fontsize=18, fontweight="bold")

# ── Panel izquierdo: trayectoria con degradado de color ─────────────────────
if env == "mountain_car":
    POS_MIN, POS_MAX = -1.2, 0.6
    GOAL_X = 0.45
    hill_x = np.linspace(POS_MIN - 0.05, POS_MAX + 0.05, 300)
    hill_y = np.sin(3.0 * hill_x) * 0.45 + 0.55

    ax_left.plot(hill_x, hill_y, color="saddlebrown", linewidth=2.0, zorder=1)
    ax_left.fill_between(hill_x, hill_y - 0.3, hill_y, color="peru", alpha=0.4, zorder=0)
    ax_left.axvline(GOAL_X, color="green", linewidth=1.5, linestyle="--", alpha=0.7, label="Meta")

    positions = traj[:, 1]
    for k, i in enumerate(idxs):
        pos = positions[i]
        y   = np.sin(3.0 * pos) * 0.45 + 0.55
        ax_left.plot(pos, y, "o", color=colors[k], markersize=11,
                     markeredgecolor="black", markeredgewidth=0.4, zorder=5)

    ax_left.set_xlim(POS_MIN - 0.05, POS_MAX + 0.05)
    ax_left.set_ylim(0.0, 1.15)
    ax_left.set_xlabel("Posición (m)", fontsize=13)
    ax_left.set_ylabel("Altura", fontsize=13)
    ax_left.legend(fontsize=12, loc="upper left")

elif env == "acrobot":
    L = 1.0
    cos_th1, sin_th1 = traj[:, 1], traj[:, 2]
    cos_th2, sin_th2 = traj[:, 3], traj[:, 4]
    th1 = np.arctan2(sin_th1, cos_th1)
    th2 = np.arctan2(sin_th2, cos_th2)

    ax_left.axhline(1.0, color="green", linewidth=1.2, linestyle="--", alpha=0.6, label="Meta")
    ax_left.plot(0, 0, "ko", markersize=6, zorder=5)

    for k, i in enumerate(idxs):
        x1, y1 = L * np.sin(th1[i]), -L * np.cos(th1[i])
        x2 = x1 + L * np.sin(th1[i] + th2[i])
        y2 = y1 - L * np.cos(th1[i] + th2[i])
        c = colors[k]
        ax_left.plot([0, x1], [0, y1], "o-", color=c, linewidth=3, markersize=6, zorder=4)
        ax_left.plot([x1, x2], [y1, y2], "s-", color=c, linewidth=3, markersize=5, zorder=4)
        ax_left.plot(x2, y2, "*", color=c, markersize=14,
                     markeredgecolor="black", markeredgewidth=0.4, zorder=6)

    ax_left.set_xlim(-2.2, 2.2)
    ax_left.set_ylim(-2.2, 2.2)
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("x (m)", fontsize=13)
    ax_left.set_ylabel("y (m)", fontsize=13)
    ax_left.legend(fontsize=12, loc="upper right")

else:  # car
    BOUND   = 2.5
    EXTENT  = [-BOUND, BOUND, -BOUND, BOUND]
    CAR_LEN = 0.18

    TRACK_SEARCH_PATHS = [
        "../snn-simulator/external/rl-tools/include/rl_tools/rl/environments/car/track.h",
        os.path.expanduser("~/Tesis/snn-simulator/external/rl-tools/include/rl_tools/rl/environments/car/track.h"),
    ]
    track_file = args.track
    if track_file is None:
        for p in TRACK_SEARCH_PATHS:
            if os.path.exists(p):
                track_file = p
                break

    track_img = None
    if track_file and os.path.exists(track_file):
        with open(track_file) as f:
            content = f.read()
        values = re.findall(r'\b(true|false)\b', content)
        if len(values) == 10000:
            track_img = np.array([v == "true" for v in values], dtype=bool).reshape(100, 100)
        else:
            print(f"Advertencia: track.h con {len(values)} valores (esperaba 10000)")
    else:
        print("Advertencia: no se encontró track.h — fondo en blanco")

    if track_img is not None:
        rgb = np.ones((100, 100, 3))
        rgb[track_img]  = [0.75, 0.92, 0.72]
        rgb[~track_img] = [0.88, 0.88, 0.88]
        ax_left.imshow(rgb, extent=EXTENT, origin="upper", aspect="equal", zorder=0)
    else:
        ax_left.set_facecolor("#f0f0f0")

    xs, ys, mus = traj[:, 1], traj[:, 2], traj[:, 3]
    ax_left.plot(xs, ys, color="gray", alpha=0.35, linewidth=1.0, zorder=1)

    for k, i in enumerate(idxs):
        cx, cy, mu = xs[i], ys[i], mus[i]
        dx, dy = CAR_LEN * np.cos(mu), CAR_LEN * np.sin(mu)
        c = colors[k]
        ax_left.annotate("", xy=(cx + dx, cy + dy), xytext=(cx - dx * 0.3, cy - dy * 0.3),
                          arrowprops=dict(arrowstyle="simple,head_width=0.5,tail_width=0.2",
                                          color=c, lw=1.5), zorder=6)

    ax_left.plot(xs[0],  ys[0],  "o", color="black", markersize=7, zorder=5, label="Inicio")
    ax_left.plot(xs[-1], ys[-1], "s", color="black", markersize=7, zorder=5, label="Fin")

    ax_left.set_xlim(-BOUND - 0.1, BOUND + 0.1)
    ax_left.set_ylim(-BOUND - 0.1, BOUND + 0.1)
    ax_left.set_aspect("equal")
    ax_left.set_xlabel("x (m)", fontsize=13)
    ax_left.set_ylabel("y (m)", fontsize=13)
    ax_left.legend(fontsize=12, loc="upper right")

ax_left.set_title("Trayectoria (color = avance temporal)", fontsize=14)
ax_left.grid(True, alpha=0.2, linewidth=0.4)
ax_left.tick_params(labelsize=11)

# ── Panel derecho: recompensa acumulada con el mismo degradado ──────────────
points   = np.array([steps_col, cum_rew]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)
seg_colors = cmap(np.linspace(0.0, 1.0, max(1, len(segments))))
lc = LineCollection(segments, colors=seg_colors, linewidth=1.8, zorder=2)
ax_right.add_collection(lc)

for k, i in enumerate(idxs):
    ax_right.plot(steps_col[i], cum_rew[i], "o", color=colors[k], markersize=8,
                  markeredgecolor="black", markeredgewidth=0.6, zorder=3)

ax_right.set_xlim(0, max(steps_col[-1], 1))
y_pad = 0.05 * max(abs(cum_rew.min()), abs(cum_rew.max()), 1.0)
ax_right.set_ylim(min(cum_rew.min(), 0) - y_pad, max(cum_rew.max(), 0) + y_pad)
ax_right.set_xlabel("Paso", fontsize=13)
ax_right.set_ylabel("Reward acumulado", fontsize=13)
ax_right.set_title("Recompensa acumulada", fontsize=14)
ax_right.grid(True, alpha=0.3)
ax_right.tick_params(labelsize=11)

if args.save:
    out_path = args.out or (os.path.splitext(csv_path)[0] + "_steps.png")
    fig.savefig(out_path, dpi=150)
    print(f"Guardado: {out_path}")
else:
    plt.show()
