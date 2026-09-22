#!/usr/bin/env python3
"""
Animación 3D del episodio del L2F (Crazyflie) con la mejor red WANN+SNN.

pos_err/vel_err son relativos al punto de la trayectoria Lissajous objetivo
en cada paso, no posición absoluta — por eso el dron se dibuja en el marco
de referencia del objetivo (el objetivo queda fijo en el origen).

Modo un episodio:
  python replay_l2f.py --csv log/snn_l2f_best_replay.csv

Modo evolución (secuencia de generaciones guardadas cada REPLAY_INTERVAL):
  python replay_l2f.py --dir log/snn_l2f_replay/
  python replay_l2f.py --dir log/snn_l2f_replay/ --save
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
parser.add_argument("--csv",   default="log/snn_l2f_best_replay.csv",
                    help="CSV de trayectoria (modo un episodio)")
parser.add_argument("--dir",   default=None,
                    help="Directorio con gen_XXXX.csv para modo evolución")
parser.add_argument("--save",  action="store_true",
                    help="Guardar como MP4")
parser.add_argument("--fps",   type=int,   default=30)
parser.add_argument("--speed", type=float, default=1.0)
parser.add_argument("--pause", type=int,   default=20,
                    help="Frames de pausa entre generaciones (default: 20)")
parser.add_argument("--step",  type=int,   default=1,
                    help="Pasos del entorno por frame (default: 1). "
                         "L2F corre a 100Hz -> --step 3 --fps 30 da tiempo real aprox.")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Cargar trayectorias
# Columnas: step,pos_err_x,pos_err_y,pos_err_z,qw,qx,qy,qz,
#           vel_err_x,vel_err_y,vel_err_z,wx,wy,wz,
#           rotor0,rotor1,rotor2,rotor3,reward
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
        print("Primero corre:  ./build/wann_l2f_replay")
        exit(1)
    trajs = [(0, load_csv(args.csv))]

# ─────────────────────────────────────────────────────────────────────────────
# Mapa de frames: (traj_idx, step_idx)
# ─────────────────────────────────────────────────────────────────────────────
PAUSE = args.pause
STEP = max(1, args.step)
frame_map = []
for ti, (gen, traj) in enumerate(trajs):
    for si in range(0, len(traj), STEP):
        frame_map.append((ti, si))
    if ti < len(trajs) - 1:
        for _ in range(PAUSE):
            frame_map.append((ti, len(traj) - 1))

total_frames = len(frame_map)

# ─────────────────────────────────────────────────────────────────────────────
# Precomputar magnitudes por trayectoria
# ─────────────────────────────────────────────────────────────────────────────
pos_errs  = [t[:, 1:4]           for _, t in trajs]
quats     = [t[:, 4:8]           for _, t in trajs]   # qw,qx,qy,qz
ang_vels  = [np.linalg.norm(t[:, 11:14], axis=1) for _, t in trajs]
rotors    = [t[:, 14:18]         for _, t in trajs]
cum_rews  = [np.cumsum(t[:, 18]) for _, t in trajs]

g_pos_max = max(np.abs(p).max() for p in pos_errs) * 1.2 + 0.05
g_wmax    = max(w.max() for w in ang_vels) * 1.1 + 0.1
all_cr = np.concatenate(cum_rews)
g_cr_min = min(all_cr.min() * 1.05, 0)
g_cr_max = max(all_cr.max() * 1.05, 1)

def quat_to_axes(qw, qx, qy, qz):
    """Ejes del cuerpo (x,y,z) expresados en el marco mundo, como columnas
    de la matriz de rotación asociada al cuaternión."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx**2 + qy**2)],
    ])
    return R[:, 0], R[:, 1], R[:, 2]  # ejes x,y,z del cuerpo en coords mundo

# ─────────────────────────────────────────────────────────────────────────────
# Figura
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 7), facecolor="white")
title_obj = fig.suptitle("WANN+SNN — L2F (Crazyflie)", fontsize=13, fontweight="bold")

gs = fig.add_gridspec(3, 2, width_ratios=[2, 1], hspace=0.5, wspace=0.35,
                      left=0.03, right=0.97, top=0.90, bottom=0.08)

ax_traj   = fig.add_subplot(gs[:, 0], projection="3d")
ax_reward = fig.add_subplot(gs[0, 1])
ax_angvel = fig.add_subplot(gs[1, 1])
ax_rotors = fig.add_subplot(gs[2, 1])

# ── Panel trayectoria 3D ─────────────────────────────────────────────────────
ax_traj.set_xlim(-g_pos_max, g_pos_max)
ax_traj.set_ylim(-g_pos_max, g_pos_max)
ax_traj.set_zlim(-g_pos_max, g_pos_max)
ax_traj.set_xlabel("err x (m)", fontsize=8)
ax_traj.set_ylabel("err y (m)", fontsize=8)
ax_traj.set_zlabel("err z (m)", fontsize=8)
ax_traj.scatter([0], [0], [0], color="green", s=40, marker="x", label="Objetivo (Lissajous)")
ax_traj.legend(fontsize=7, loc="upper right")

ghost_line, = ax_traj.plot([], [], [], color="steelblue", alpha=0.20, linewidth=1.0)
trail_line, = ax_traj.plot([], [], [], color="royalblue", linewidth=1.5)
drone_dot   = ax_traj.scatter([0], [0], [0], color="crimson", s=30)

AXIS_LEN = 0.06
body_x_line, = ax_traj.plot([], [], [], color="red",   linewidth=2.0)
body_y_line, = ax_traj.plot([], [], [], color="green", linewidth=2.0)
body_z_line, = ax_traj.plot([], [], [], color="blue",  linewidth=2.0)

step_text = ax_traj.text2D(0.02, 0.97, "", transform=ax_traj.transAxes,
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

# ── Panel velocidad angular ──────────────────────────────────────────────────
ax_angvel.set_xlim(0, 1)
ax_angvel.set_ylim(0, g_wmax)
ax_angvel.set_xlabel("Paso", fontsize=8)
ax_angvel.set_ylabel("||w|| (rad/s)", fontsize=8)
ax_angvel.set_title("Velocidad angular", fontsize=9)
ax_angvel.grid(True, alpha=0.3)
angvel_line, = ax_angvel.plot([], [], color="purple", linewidth=1.5)
angvel_dot,  = ax_angvel.plot([], [], "o", color="purple", markersize=5)

# ── Panel rotores ─────────────────────────────────────────────────────────────
ax_rotors.set_xlim(-0.5, 3.5)
ax_rotors.set_ylim(-1.2, 1.2)
ax_rotors.set_xticks([0, 1, 2, 3])
ax_rotors.set_xticklabels(["r0", "r1", "r2", "r3"], fontsize=8)
ax_rotors.set_ylabel("Comando", fontsize=8)
ax_rotors.set_title("Rotores", fontsize=9)
ax_rotors.axhline(0, color="gray", linewidth=0.6, linestyle="--")
rotor_bars = ax_rotors.bar([0, 1, 2, 3], [0, 0, 0, 0],
                            color=["forestgreen", "steelblue", "goldenrod", "indianred"],
                            width=0.6, alpha=0.85)

# ─────────────────────────────────────────────────────────────────────────────
# Animación
# ─────────────────────────────────────────────────────────────────────────────
cur_ti = [-1]

def update(frame):
    ti, si = frame_map[frame]
    gen, traj = trajs[ti]

    pe   = pos_errs[ti]
    q    = quats[ti]
    steps_col = traj[:, 0].astype(int)
    n_steps   = len(steps_col)
    wnorm = ang_vels[ti]
    rot   = rotors[ti]
    cr    = cum_rews[ti]

    if ti != cur_ti[0]:
        cur_ti[0] = ti

        ghost_line.set_data(pe[:, 0], pe[:, 1])
        ghost_line.set_3d_properties(pe[:, 2])

        ax_reward.set_xlim(0, n_steps)
        ax_angvel.set_xlim(0, n_steps)

        label = f"Gen {gen:04d}  ({ti+1}/{len(trajs)})" if len(trajs) > 1 else "L2F"
        title_obj.set_text(f"WANN+SNN — {label}")
        ax_traj.set_title(f"Error de posición vs. objetivo — {label}", fontsize=10)

        trail_line.set_data([], [])
        trail_line.set_3d_properties([])
        reward_line.set_data([], [])
        angvel_line.set_data([], [])

    i = si

    px, py, pz = pe[i]
    ex, ey, ez = quat_to_axes(*q[i])

    drone_dot._offsets3d = ([px], [py], [pz])

    lo = max(0, i - 60)
    trail_line.set_data(pe[lo:i+1, 0], pe[lo:i+1, 1])
    trail_line.set_3d_properties(pe[lo:i+1, 2])

    body_x_line.set_data([px, px + AXIS_LEN * ex[0]], [py, py + AXIS_LEN * ex[1]])
    body_x_line.set_3d_properties([pz, pz + AXIS_LEN * ex[2]])
    body_y_line.set_data([px, px + AXIS_LEN * ey[0]], [py, py + AXIS_LEN * ey[1]])
    body_y_line.set_3d_properties([pz, pz + AXIS_LEN * ey[2]])
    body_z_line.set_data([px, px + AXIS_LEN * ez[0]], [py, py + AXIS_LEN * ez[1]])
    body_z_line.set_3d_properties([pz, pz + AXIS_LEN * ez[2]])

    step_text.set_text(
        f"Paso: {steps_col[i]:3d}/{n_steps}\n"
        f"||err||: {np.linalg.norm(pe[i]):.3f} m\n"
        f"Reward: {cr[i]:.1f}"
    )

    reward_line.set_data(steps_col[:i+1], cr[:i+1])
    reward_dot.set_data( [steps_col[i]],  [cr[i]])

    angvel_line.set_data(steps_col[:i+1], wnorm[:i+1])
    angvel_dot.set_data( [steps_col[i]],  [wnorm[i]])

    for bar, val in zip(rotor_bars, rot[i]):
        bar.set_height(val)
        bar.set_y(min(0, val))

    return (ghost_line, trail_line, drone_dot, body_x_line, body_y_line, body_z_line,
            step_text, reward_line, reward_dot, angvel_line, angvel_dot, *rotor_bars)

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
