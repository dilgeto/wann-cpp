#!/usr/bin/env python3
"""
Animación del episodio del Car con la mejor red WANN+SNN.

Modo un episodio:
  python replay.py --csv log/snn_car_best_replay.csv

Modo evolución (secuencia de generaciones guardadas cada 256 gens):
  python replay.py --dir log/snn_car_replay/
  python replay.py --dir log/snn_car_replay/ --save
"""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ─────────────────────────────────────────────────────────────────────────────
# Argumentos
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",   default="log/snn_car_best_replay.csv",
                    help="CSV de trayectoria (modo un episodio)")
parser.add_argument("--dir",   default=None,
                    help="Directorio con gen_XXXX.csv para modo evolución")
parser.add_argument("--track", default=None,
                    help="Ruta al track.h de rl-tools")
parser.add_argument("--save",  action="store_true",
                    help="Guardar como MP4")
parser.add_argument("--fps",   type=int,   default=30)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--pause", type=int,   default=20,
                    help="Frames de pausa entre generaciones (default: 20)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Cargar trayectorias
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return data

if args.dir:
    csv_files = sorted(glob.glob(os.path.join(args.dir, "gen_*.csv")))
    if not csv_files:
        print(f"No se encontraron gen_*.csv en {args.dir}")
        exit(1)
    trajs = []
    for f in csv_files:
        m = re.search(r'gen_(\d+)', os.path.basename(f))
        gen = int(m.group(1)) if m else 0
        trajs.append((gen, load_csv(f)))
    print(f"Cargadas {len(trajs)} trayectorias de {args.dir}")
else:
    if not os.path.exists(args.csv):
        print(f"No se encontró {args.csv}")
        print("Primero corre:  ./build/wann_car_replay")
        exit(1)
    trajs = [(0, load_csv(args.csv))]

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de frames: (traj_idx, step_idx)
# Se agrega una pausa al final de cada generación excepto la última.
# ─────────────────────────────────────────────────────────────────────────────
PAUSE = args.pause
frame_map = []
for ti, (gen, traj) in enumerate(trajs):
    for si in range(len(traj)):
        frame_map.append((ti, si))
    if ti < len(trajs) - 1:
        for _ in range(PAUSE):
            frame_map.append((ti, len(traj) - 1))

total_frames = len(frame_map)

# ─────────────────────────────────────────────────────────────────────────────
# Precomputar velocidad y reward acumulado por trayectoria
# ─────────────────────────────────────────────────────────────────────────────
# Columnas: step,x,y,mu,vx,vy,omega,lidar_l,lidar_c,lidar_r,throttle,steering,reward
speeds    = [np.hypot(t[:, 4], t[:, 5]) for _, t in trajs]
cum_rews  = [np.cumsum(t[:, 12])        for _, t in trajs]

# Límites globales para que los ejes no salten entre generaciones.
g_speed_max = max(sp.max() for sp in speeds) * 1.1 + 0.1
all_cr = np.concatenate(cum_rews)
g_cr_min = min(all_cr.min() * 1.05, 0)
g_cr_max = max(all_cr.max() * 1.05, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Track (track.h de rl-tools)
# ─────────────────────────────────────────────────────────────────────────────
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
        print(f"Track cargado desde: {track_file}")
    else:
        print(f"Advertencia: track.h con {len(values)} valores (esperaba 10000)")
else:
    print("Advertencia: no se encontró track.h — fondo en blanco")

# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────
BOUND  = 2.5
EXTENT = [-BOUND, BOUND, -BOUND, BOUND]
CAR_LEN = 0.18

fig = plt.figure(figsize=(13, 6), facecolor="white")
title_obj = fig.suptitle("WANN+SNN — CarTrack", fontsize=13, fontweight="bold")

gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.45, wspace=0.35,
                      left=0.06, right=0.97, top=0.90, bottom=0.08)

ax_track  = fig.add_subplot(gs[:, 0])
ax_reward = fig.add_subplot(gs[0, 1])
ax_speed  = fig.add_subplot(gs[1, 1])
ax_action = fig.add_subplot(gs[2, 1])

# ── Panel track ──────────────────────────────────────────────────────────────
if track_img is not None:
    rgb = np.ones((100, 100, 3))
    rgb[track_img]  = [0.75, 0.92, 0.72]
    rgb[~track_img] = [0.88, 0.88, 0.88]
    ax_track.imshow(rgb, extent=EXTENT, origin="upper", aspect="equal", zorder=0)
else:
    ax_track.set_facecolor("#f0f0f0")

ax_track.set_xlim(-BOUND - 0.1, BOUND + 0.1)
ax_track.set_ylim(-BOUND - 0.1, BOUND + 0.1)
ax_track.set_xlabel("x (m)", fontsize=9)
ax_track.set_ylabel("y (m)", fontsize=9)
ax_track.set_aspect("equal")
ax_track.grid(True, alpha=0.2, linewidth=0.4)

# Trayectoria completa (fantasma) — se actualiza en cada generación
ghost_line, = ax_track.plot([], [], color="steelblue", alpha=0.20, linewidth=1.0, zorder=1)
start_dot,  = ax_track.plot([], [], "go", markersize=7, zorder=5, label="Inicio")
end_dot,    = ax_track.plot([], [], "rs", markersize=7, zorder=5, label="Fin")
ax_track.legend(fontsize=7, loc="upper right")

trail_line, = ax_track.plot([], [], color="royalblue", linewidth=1.5, zorder=2)
car_arrow   = ax_track.annotate("", xy=(0, 0), xytext=(0, 0),
    arrowprops=dict(arrowstyle="simple,head_width=0.5,tail_width=0.2",
                    color="crimson", lw=1.5),
    zorder=6)
step_text = ax_track.text(0.02, 0.97, "", transform=ax_track.transAxes,
                           fontsize=8, va="top", ha="left",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# ── Panel reward ─────────────────────────────────────────────────────────────
ax_reward.set_xlim(0, 1)
ax_reward.set_ylim(g_cr_min, g_cr_max)
ax_reward.set_xlabel("Paso", fontsize=8)
ax_reward.set_ylabel("Reward acum.", fontsize=8)
ax_reward.set_title("Reward acumulado", fontsize=9)
ax_reward.grid(True, alpha=0.3)
reward_line, = ax_reward.plot([], [], color="darkorange", linewidth=1.5)
reward_dot,  = ax_reward.plot([], [], "o", color="darkorange", markersize=5)

# ── Panel velocidad ───────────────────────────────────────────────────────────
ax_speed.set_xlim(0, 1)
ax_speed.set_ylim(0, g_speed_max)
ax_speed.set_xlabel("Paso", fontsize=8)
ax_speed.set_ylabel("||v|| (m/s)", fontsize=8)
ax_speed.set_title("Velocidad", fontsize=9)
ax_speed.grid(True, alpha=0.3)
speed_line, = ax_speed.plot([], [], color="purple", linewidth=1.5)
speed_dot,  = ax_speed.plot([], [], "o", color="purple", markersize=5)

# ── Panel acción ──────────────────────────────────────────────────────────────
ax_action.set_xlim(-0.5, 1.5)
ax_action.set_ylim(-1.2, 1.2)
ax_action.set_xticks([0, 1])
ax_action.set_xticklabels(["Acelerador", "Dirección"], fontsize=8)
ax_action.set_ylabel("Valor", fontsize=8)
ax_action.set_title("Acciones", fontsize=9)
ax_action.axhline(0, color="gray", linewidth=0.6, linestyle="--")
action_bars = ax_action.bar([0, 1], [0, 0], color=["forestgreen", "steelblue"],
                             width=0.5, alpha=0.8)

# ─────────────────────────────────────────────────────────────────────────────
# Animación
# ─────────────────────────────────────────────────────────────────────────────
cur_ti = [-1]  # índice de trayectoria activa

def update(frame):
    ti, si = frame_map[frame]
    gen, traj = trajs[ti]

    xs        = traj[:, 1];  ys       = traj[:, 2]
    mus       = traj[:, 3]
    throttles = traj[:, 10]; steerings = traj[:, 11]
    steps_col = traj[:, 0].astype(int)
    n_steps   = len(steps_col)
    sp        = speeds[ti]
    cr        = cum_rews[ti]

    # Al cambiar de generación: actualizar elementos estáticos.
    if ti != cur_ti[0]:
        cur_ti[0] = ti

        ghost_line.set_data(xs, ys)
        start_dot.set_data([xs[0]],  [ys[0]])
        end_dot.set_data(  [xs[-1]], [ys[-1]])

        ax_reward.set_xlim(0, n_steps)
        ax_speed.set_xlim(0, n_steps)

        label = f"Gen {gen:04d}  ({ti+1}/{len(trajs)})" if len(trajs) > 1 else "CarTrack"
        title_obj.set_text(f"WANN+SNN — {label}")
        ax_track.set_title(f"Trayectoria — {label}", fontsize=10)

        # Limpiar líneas del episodio anterior.
        trail_line.set_data([], [])
        reward_line.set_data([], [])
        speed_line.set_data([], [])

    i = si

    # Carro
    cx, cy, mu = xs[i], ys[i], mus[i]
    dx = CAR_LEN * np.cos(mu);  dy = CAR_LEN * np.sin(mu)
    car_arrow.set_position((cx, cy))
    car_arrow.xy     = (cx + dx,        cy + dy)
    car_arrow.xytext = (cx - dx * 0.3,  cy - dy * 0.3)

    trail_line.set_data(xs[max(0, i-60):i+1], ys[max(0, i-60):i+1])

    step_text.set_text(
        f"Paso: {steps_col[i]:3d}/{n_steps}\n"
        f"v: {sp[i]:.2f} m/s\n"
        f"Reward: {cr[i]:.1f}"
    )

    reward_line.set_data(steps_col[:i+1], cr[:i+1])
    reward_dot.set_data( [steps_col[i]],  [cr[i]])

    speed_line.set_data(steps_col[:i+1], sp[:i+1])
    speed_dot.set_data( [steps_col[i]],  [sp[i]])

    action_bars[0].set_height(throttles[i]);  action_bars[0].set_y(min(0, throttles[i]))
    action_bars[1].set_height(steerings[i]);  action_bars[1].set_y(min(0, steerings[i]))

    return (ghost_line, start_dot, end_dot, trail_line, car_arrow, step_text,
            reward_line, reward_dot, speed_line, speed_dot, *action_bars)

# Índice del primer frame de cada generación (para saltar con ←/→)
gen_starts = []
for ti in range(len(trajs)):
    for fi, (t, _) in enumerate(frame_map):
        if t == ti:
            gen_starts.append(fi)
            break

state = {'paused': False, 'speed': args.speed, 'frame': 0}

def frame_gen():
    while True:
        yield min(state['frame'], len(frame_map) - 1)
        if not state['paused'] and state['frame'] < len(frame_map) - 1:
            state['frame'] += 1

def on_key(event):
    if event.key == ' ':
        state['paused'] = not state['paused']
    elif event.key in ('up', '+', '='):
        state['speed'] = min(state['speed'] * 2.0, 32.0)
        anim.event_source.interval = max(1, int(1000 / args.fps / state['speed']))
    elif event.key in ('down', '-'):
        state['speed'] = max(state['speed'] / 2.0, 0.125)
        anim.event_source.interval = max(1, int(1000 / args.fps / state['speed']))
    elif event.key == 'right' and len(trajs) > 1:
        cur = frame_map[min(state['frame'], len(frame_map) - 1)][0]
        if cur + 1 < len(gen_starts):
            state['frame'] = gen_starts[cur + 1]
    elif event.key == 'left' and len(trajs) > 1:
        cur = frame_map[min(state['frame'], len(frame_map) - 1)][0]
        state['frame'] = gen_starts[max(cur - 1, 0)]

interval_ms = max(1, int(1000 / args.fps / args.speed))

if args.save:
    anim = FuncAnimation(fig, update, frames=total_frames,
                         interval=interval_ms, blit=False)
    out_mp4 = (args.dir or args.csv).rstrip("/").replace(".csv", "") + "_evolution.mp4"
    writer = FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(out_mp4, writer=writer, dpi=150)
    print(f"Guardado: {out_mp4}")
else:
    fig.canvas.mpl_connect('key_press_event', on_key)
    anim = FuncAnimation(fig, update, frames=frame_gen(),
                         interval=interval_ms, blit=False, save_count=total_frames)
    print("Controles: Espacio=pausa  ↑/↓=velocidad  ←/→=generación anterior/siguiente")
    plt.show()
