#!/usr/bin/env python3
"""
Animación del MountainCar con la mejor red WANN+SNN.

Modo un episodio:
  python replay_mountain_car.py --csv log/snn_mountain_car_best_replay.csv

Modo evolución (secuencia de generaciones):
  python replay_mountain_car.py --dir log/snn_mountain_car_replay/
  python replay_mountain_car.py --dir log/snn_mountain_car_replay/ --save
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
parser.add_argument("--csv",   default="log/snn_mountain_car_best_replay.csv")
parser.add_argument("--dir",   default=None)
parser.add_argument("--save",  action="store_true")
parser.add_argument("--fps",   type=int,   default=30)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--pause", type=int,   default=20,
                    help="Frames de pausa entre generaciones (default: 20)")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Cargar trayectorias
# Columnas: step,position,velocity,action,reward
# ─────────────────────────────────────────────────────────────────────────────
def load_csv(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[np.newaxis, :] if data.ndim == 1 else data

if args.dir:
    csv_files = sorted(glob.glob(os.path.join(args.dir, "gen_*.csv")))
    if not csv_files:
        print(f"No se encontraron gen_*.csv en {args.dir}"); exit(1)
    trajs = [(int(re.search(r'gen_(\d+)', os.path.basename(f)).group(1)), load_csv(f))
             for f in csv_files]
    print(f"Cargadas {len(trajs)} trayectorias de {args.dir}")
else:
    if not os.path.exists(args.csv):
        print(f"No se encontró {args.csv}"); exit(1)
    trajs = [(0, load_csv(args.csv))]

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de frames
# ─────────────────────────────────────────────────────────────────────────────
PAUSE = args.pause
frame_map = []
for ti, (gen, traj) in enumerate(trajs):
    for si in range(len(traj)):
        frame_map.append((ti, si))
    if ti < len(trajs) - 1:
        for _ in range(PAUSE):
            frame_map.append((ti, len(traj) - 1))

# ─────────────────────────────────────────────────────────────────────────────
# Superficie del hill: y = sin(3*x) * 0.45 + 0.55
# ─────────────────────────────────────────────────────────────────────────────
POS_MIN, POS_MAX = -1.2, 0.6
hill_x = np.linspace(POS_MIN - 0.05, POS_MAX + 0.05, 300)
hill_y = np.sin(3.0 * hill_x) * 0.45 + 0.55
GOAL_X = 0.45

# Precomputar reward acumulado
cum_rews = [np.cumsum(t[:, 4]) for _, t in trajs]
all_cr   = np.concatenate(cum_rews)
g_cr_min = min(all_cr.min() * 1.05, 0)
g_cr_max = max(all_cr.max() * 1.05, 1)

# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(11, 5), facecolor="white")
title_obj = fig.suptitle("WANN+SNN — MountainCar", fontsize=13, fontweight="bold")

gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.5, wspace=0.35,
                      left=0.07, right=0.97, top=0.88, bottom=0.10)

ax_hill   = fig.add_subplot(gs[:, 0])
ax_reward = fig.add_subplot(gs[0, 1])
ax_action = fig.add_subplot(gs[1, 1])

# ── Panel colina ──────────────────────────────────────────────────────────────
ax_hill.plot(hill_x, hill_y, color="saddlebrown", linewidth=2.0, zorder=1)
ax_hill.fill_between(hill_x, hill_y - 0.3, hill_y, color="peru", alpha=0.4, zorder=0)
ax_hill.axvline(GOAL_X, color="green", linewidth=1.5, linestyle="--", alpha=0.7, label="Meta")
ax_hill.set_xlim(POS_MIN - 0.05, POS_MAX + 0.05)
ax_hill.set_ylim(0.0, 1.15)
ax_hill.set_xlabel("Posición (m)", fontsize=9)
ax_hill.set_ylabel("Altura", fontsize=9)
ax_hill.set_aspect("auto")
ax_hill.grid(True, alpha=0.2, linewidth=0.5)
ax_hill.legend(fontsize=8, loc="upper left")

car_dot,  = ax_hill.plot([], [], "o", color="crimson", markersize=10, zorder=5)
step_text  = ax_hill.text(0.03, 0.97, "", transform=ax_hill.transAxes,
                           fontsize=8, va="top",
                           bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

# ── Panel reward ──────────────────────────────────────────────────────────────
ax_reward.set_xlim(0, 1)
ax_reward.set_ylim(g_cr_min, g_cr_max)
ax_reward.set_xlabel("Paso", fontsize=8)
ax_reward.set_ylabel("Reward acum.", fontsize=8)
ax_reward.set_title("Reward acumulado", fontsize=9)
ax_reward.grid(True, alpha=0.3)
reward_line, = ax_reward.plot([], [], color="darkorange", linewidth=1.5)
reward_dot,  = ax_reward.plot([], [], "o", color="darkorange", markersize=5)

# ── Panel acción ──────────────────────────────────────────────────────────────
ax_action.set_xlim(-0.5, 0.5)
ax_action.set_ylim(-1.2, 1.2)
ax_action.set_xticks([0])
ax_action.set_xticklabels(["Fuerza"], fontsize=8)
ax_action.set_ylabel("[-1, 1]", fontsize=8)
ax_action.set_title("Acción", fontsize=9)
ax_action.axhline(0, color="gray", linewidth=0.6, linestyle="--")
action_bar = ax_action.bar([0], [0], color="steelblue", width=0.4, alpha=0.8)[0]

# ─────────────────────────────────────────────────────────────────────────────
# Animación
# ─────────────────────────────────────────────────────────────────────────────
cur_ti = [-1]

def update(frame):
    ti, si = frame_map[frame]
    gen, traj = trajs[ti]
    cr        = cum_rews[ti]
    steps_col = traj[:, 0].astype(int)
    n_steps   = len(steps_col)
    positions = traj[:, 1]
    velocities= traj[:, 2]
    actions   = traj[:, 3]

    if ti != cur_ti[0]:
        cur_ti[0] = ti
        ax_reward.set_xlim(0, n_steps)
        reward_line.set_data([], [])
        label = f"Gen {gen:04d}  ({ti+1}/{len(trajs)})" if len(trajs) > 1 else "MountainCar"
        title_obj.set_text(f"WANN+SNN — {label}")
        ax_hill.set_title(f"Trayectoria — {label}", fontsize=10)

    i = si
    pos = positions[i]
    car_y = np.sin(3.0 * pos) * 0.45 + 0.55

    car_dot.set_data([pos], [car_y])

    step_text.set_text(
        f"Paso: {steps_col[i]:3d}/{n_steps}\n"
        f"Pos: {pos:.3f} m\n"
        f"Vel: {velocities[i]:.4f} m/s\n"
        f"R: {cr[i]:.2f}"
    )

    reward_line.set_data(steps_col[:i+1], cr[:i+1])
    reward_dot.set_data([steps_col[i]], [cr[i]])

    action_bar.set_height(actions[i])
    action_bar.set_y(min(0, actions[i]))

    return (car_dot, step_text, reward_line, reward_dot, action_bar)

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
    anim = FuncAnimation(fig, update, frames=len(frame_map),
                         interval=interval_ms, blit=False)
    out_mp4 = (args.dir or args.csv).rstrip("/").replace(".csv", "") + "_evolution.mp4"
    writer = FFMpegWriter(fps=args.fps, bitrate=1800)
    anim.save(out_mp4, writer=writer, dpi=150)
    print(f"Guardado: {out_mp4}")
else:
    fig.canvas.mpl_connect('key_press_event', on_key)
    anim = FuncAnimation(fig, update, frames=frame_gen(),
                         interval=interval_ms, blit=False, save_count=len(frame_map))
    print("Controles: Espacio=pausa  ↑/↓=velocidad  ←/→=generación anterior/siguiente")
    plt.show()
