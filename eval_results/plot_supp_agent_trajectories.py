#!/usr/bin/env python3
"""
Supplementary figures: for each task (Acrobot, Mountain Car, Racing Car) the 11 phase-3 networks (one per training seed
of the best hyperparameter configuration of the paper's WANN family) drawn next to ONE episode of the agent they control.
The agent's path is coloured by the time step at which it was visited; a strip along the bottom of the figure gives the
step each colour stands for.

Networks (same family/rank as the paper's WANN figures; replicate unit = trained network, n = 11):
  Acrobot       acrobot_small_first_spike_w8           rank02   (shared weights {0.5,1,2,3,5,8})
  Mountain Car  disc_mc_small_rate_argmax_discrete     rank01   (discrete actions)
  Racing Car    car_ttfs_first_spike_discrete          rank00
Each network is replayed with its own saved best shared weight (the one the revalidation in eval_results/*_best.csv uses),
and the episode shown is the REPRESENTATIVE one: among the 121 revalidation episodes of the network (11 evaluation seeds x 11 episodes, at its own best
shared weight; returns in eval_results/*_best_episodes.csv) the one whose return is closest to the network's revalidated mean return; so the
return printed on every panel is one of the episodes behind the numbers in the paper. The replay is done by
build/wann_replay_weights_<task> (evaluation only: nothing is trained) and cached as CSV in
eval_results/supp_trajectories/<task>/seedNN_evalMM.csv (network seed NN, evaluation seed MM); delete the cache (or pass --refresh) to regenerate it.

Output (graficos/paper/send_to_journal/): suppl_figure_S1_acrobot.{pdf,png}, suppl_figure_S2_mountain_car.{pdf,png},
suppl_figure_S3_racing_car.{pdf,png} + suppl_figures_caption.md.   Style: the journal style of the main figures
(plot_style.setup_journal_style: 190 mm wide, Liberation Sans, vector PDF with embedded fonts).

Usage:  python eval_results/plot_supp_agent_trajectories.py [--refresh]     (from anywhere)
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

from plot_style import JOURNAL_FULL_W, setup_journal_style
from supp_caption import caption_height, draw_caption

OUT = _ROOT / "graficos" / "paper" / "send_to_journal"
CACHE = _ROOT / "eval_results" / "supp_trajectories"
CMAP = plt.get_cmap("viridis")     # perceptually uniform and colour-blind safe; early = dark, late = bright

# node / edge colours (Okabe & Ito palette, same family as plot_style.MODEL_COLORS)
C_BIAS, C_IN, C_HID, C_OUT = "#E69F00", "#56B4E9", "#BBBBBB", "#009E73"
C_EXC, C_INH = "#0072B2", "#CC79A7"

TASKS = {
    "acrobot": dict(
        # w8 networks (shared-weight table {0.5,1,2,3,5,8}, 2026-11-23), winner of eval_results/acrobot_w8/acrobot_best.csv.
        # The pre-w8 figure used acrobot_small_first_spike rank01 with wann_replay_weights_acrobot_legacy (old table
        # {0.5,1,1.5,2,2.5,3}); its replay cache stays in supp_trajectories/acrobot/, hence the separate cache dir here.
        label="Acrobot", fig="suppl_figure_S1_acrobot", binary="wann_replay_weights_acrobot", base="p/acrobot_snn.json",
        run_key="acrobot_small_first_spike_w8", rank=2, csv="eval_results/acrobot_w8/acrobot_best.csv", cache="acrobot_w8"),
    "disc_mc": dict(
        label="Mountain Car", fig="suppl_figure_S2_mountain_car", binary="wann_replay_weights_disc_mc",
        base="p/disc_mc_snn.json", run_key="disc_mc_small_rate_argmax_discrete", rank=1,
        csv="eval_results/disc_mc_small_rate_argmax_discrete/mountain_car_best.csv"),
    "car": dict(
        label="Racing Car", fig="suppl_figure_S3_racing_car", binary="wann_replay_weights_car", base="p/car_snn.json",
        run_key="car_ttfs_first_spike_discrete", rank=0, csv="eval_results/car_ttfs_first_spike_discrete/car_best.csv"),
}
N_NETS = 11
COMMON = ("These are the 11 networks behind the reported WANN+SNN results (Figs. 1, 3, 4 and 5; one per training seed of the best hyperparameter configuration); Net 1 is the network ablated in Fig. 2. They are ordered from best to worst by revalidated mean return (11 evaluation seeds × 11 episodes; “mean”). "
          "Left: network topology (inputs on the left, outputs on the right, hidden neurons by depth; edge colour: synapse polarity). "
          "Right: {desc} in one representative episode (the one, among the network's 121 revalidation episodes, whose return is closest to its mean; “episode” is its return) at its best shared weight; "
          "colour is the time step (strip below, common scale for all panels).")
CAPTIONS = {
    "acrobot": ("Fig. S1. Acrobot: the 11 evolved WANN+SNN networks and their behaviour.",
                "the path of the tip of the second link (dot: pivot; circle: start; dashed line: goal height)"),
    "disc_mc": ("Fig. S2. Mountain Car (discrete actions): the 11 evolved WANN+SNN networks and their behaviour.",
                "the position of the car on the hill (circle: start; dashed line: goal; points are lifted with time so that repeated passes stay visible)"),
    "car": ("Fig. S3. Racing Car: the 11 evolved WANN+SNN networks and their behaviour.",
            "the path of the car over the track (grey; circle: start; laps after the first overlap the earlier ones)"),
}


# --------------------------------------------------------------------------------------------- data
def load_models(task):
    t = TASKS[task]
    d = pd.read_csv(t["csv"])
    d = d[(d["run_key"] == t["run_key"]) & (d["rank"] == t["rank"])]
    d = d.sort_values("reward", ascending=False).reset_index(drop=True)      # best -> worst revalidated mean return
    assert len(d) == N_NETS, (task, len(d))
    return d


def model_paths(t, seed_idx):
    stem = f"rank{t['rank']:02d}_seed{seed_idx:02d}"
    return (Path("log") / f"full_p3_{t['run_key']}" / f"{stem}_best.out",
            Path("screening_full") / t["run_key"] / "p3_configs" / f"{stem}.json")


def merged_config(t, cfg_path):
    cfg = json.loads(Path(t["base"]).read_text())
    cfg.update(json.loads(Path(cfg_path).read_text()))
    return cfg


def replay(task, row, eval_seed, rep, refresh):
    """Run build/wann_replay_weights_<task> for episode `rep` of evaluation seed `eval_seed` (cached). Returns the trajectory DataFrame."""
    t = TASKS[task]
    csv = CACHE / t.get("cache", task) / f"seed{int(row.seed_idx):02d}_eval{eval_seed:02d}_rep{rep:02d}.csv"
    if refresh or not csv.exists():
        net, cfg = model_paths(t, int(row.seed_idx))
        csv.parent.mkdir(parents=True, exist_ok=True)
        cmd = [f"./build/{t['binary']}", "-f", str(net), "-d", t["base"], "-p", str(cfg),
               "--weight-index", str(int(row.weight_index)), "--seed", str(eval_seed), "--nreps", "11", "-o", str(csv)]
        if rep:
            cmd += ["--rep", str(rep)]
        subprocess.run(cmd, check=True, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
    return pd.read_csv(csv)


_EPISODES = {}


def episodes_table(task):
    """Returns of the 121 revalidation episodes of every network (winning weight): eval_results/*_best_episodes.csv."""
    if task not in _EPISODES:
        d = pd.read_csv(TASKS[task]["csv"].replace("_best.csv", "_best_episodes.csv"))
        _EPISODES[task] = d[d["run_key"] == TASKS[task]["run_key"]]
    return _EPISODES[task]


def representative(task, row, refresh):
    """Episode, among the network's 121 revalidation episodes, whose return is closest to its revalidated mean return."""
    ep = episodes_table(task)
    ep = ep[ep["train_seed_idx"] == int(row.seed_idx)].reset_index(drop=True)
    assert len(ep) == 121 and abs(ep["reward"].mean() - float(row.reward)) < 1e-6 * max(1.0, abs(float(row.reward))), (task, row.seed_idx, len(ep))
    k = int((ep["reward"] - float(row.reward)).abs().idxmin())
    es, rp = int(ep["seed"][k]), int(ep["episode"][k])
    df = replay(task, row, es, rp, refresh)
    assert abs(episode_return(task, df) - float(ep["reward"][k])) < 1e-3 * max(1.0, abs(float(ep["reward"][k]))), \
        (task, row.seed_idx, es, rp, episode_return(task, df), ep["reward"][k])      # the replay reproduces the evaluated episode
    return es, rp, df, ep["reward"].to_numpy()


def episode_return(task, df):
    # Acrobot / Mountain Car: environment reward = -1 per step; Racing Car: reward column is the environment return
    return float(df["reward"].sum()) if task == "car" else -float(len(df))


# --------------------------------------------------------------------------------------------- network drawing
def read_net(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append([float("nan") if tok.strip().lower() == "nan" else float(tok) for tok in line.split(",") if tok.strip()])
    n = len(rows)
    return np.array([r[:n] for r in rows])


def layout(W, n_in, n_out):
    """Node positions: x = longest-path depth (inputs left, outputs right), y ordered by barycentre sweeps."""
    N = len(W)
    out_ids = list(range(N - n_out, N))
    edges = [(s, d) for s in range(N) for d in range(N) if np.isfinite(W[s, d]) and W[s, d] != 0.0]
    succ, indeg = collections.defaultdict(list), [0] * N
    for s, d in edges:
        succ[s].append(d)
        indeg[d] += 1
    depth = [0] * N
    queue = collections.deque(n for n in range(N) if indeg[n] == 0)
    while queue:
        u = queue.popleft()
        for v in succ[u]:
            depth[v] = max(depth[v], depth[u] + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)
    inputs = list(range(0, 1 + n_in))
    for n in inputs:
        depth[n] = 0
    max_pre = max(depth[n] for n in range(N) if n not in out_ids)
    for n in out_ids:
        depth[n] = max_pre + 1
    layers = collections.defaultdict(list)
    for n in range(N):
        layers[depth[n]].append(n)
    order = {n: i for L in layers.values() for i, n in enumerate(L)}
    preds = collections.defaultdict(list)
    for s, d in edges:
        preds[d].append(s)
    ypos = {}

    def assign():
        for L in layers.values():
            m = len(L)
            for i, n in enumerate(L):
                ypos[n] = 1.0 - (i + 0.5) / m
    for L in layers.values():
        L.sort(key=lambda n: n if depth[n] in (0, max_pre + 1) else 0)
    assign()
    hidden_layers = [k for k in sorted(layers) if 0 < k <= max_pre]
    for sweep in range(4):                      # barycentre ordering of hidden layers (fewer crossings)
        for k in hidden_layers:
            layers[k].sort(key=lambda n: np.mean([ypos[p] for p in preds[n]]) if preds[n] else 0.5, reverse=True)
        assign()
    xs = {n: depth[n] / (max_pre + 1) for n in range(N)}
    return {n: (xs[n], ypos[n]) for n in range(N)}, edges


def draw_network(ax, W, n_in, n_out, lw=0.35, ms=7):
    N = len(W)
    pos, edges = layout(W, n_in, n_out)
    used = {n for e in edges for n in e}
    for s, d in edges:
        (x0, y0), (x1, y1) = pos[s], pos[d]
        ax.plot([x0, x1], [y0, y1], color=C_EXC if W[s, d] > 0 else C_INH, lw=lw, alpha=0.55, zorder=1, solid_capstyle="round")
    kind = lambda n: "bias" if n == 0 else "in" if n <= n_in else "out" if n >= N - n_out else "hid"
    col = dict(bias=C_BIAS, **{"in": C_IN}, out=C_OUT, hid=C_HID)
    for n in range(N):
        k = kind(n)
        x, y = pos[n]
        unused = (k in ("in", "bias")) and n not in used
        ax.scatter([x], [y], s=ms, c=col[k], edgecolors="0.25", linewidths=0.3, zorder=3, alpha=0.35 if unused else 1.0)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")


# --------------------------------------------------------------------------------------------- trajectory drawing
def time_lines(x, y, t, tmax, lw):
    pts = np.column_stack([x, y])
    seg = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(seg, cmap=CMAP, norm=plt.Normalize(0, tmax), linewidths=lw, capstyle="round", zorder=3)
    lc.set_array(t[:-1])
    return lc


_TRACK = None


def track_image():
    global _TRACK
    if _TRACK is None:
        f = Path("../snn-simulator/external/rl-tools/include/rl_tools/rl/environments/car/track.h")
        vals = re.findall(r"\b(true|false)\b", f.read_text())
        assert len(vals) == 10000
        _TRACK = np.array([v == "true" for v in vals], dtype=bool).reshape(100, 100)
    return _TRACK


def draw_agent(task, ax, df, tmax):
    t = df["step"].to_numpy(float)
    if task == "car":
        b = 2.5
        rgb = np.where(track_image()[..., None], np.array([0.80, 0.80, 0.80]), np.array([1.0, 1.0, 1.0]))
        ax.imshow(rgb, extent=[-b, b, -b, b], origin="upper", zorder=0, interpolation="nearest")
        ax.add_collection(time_lines(df["x"].to_numpy(), df["y"].to_numpy(), t, tmax, 1.6))
        ax.scatter(df["x"].iloc[0], df["y"].iloc[0], s=9, marker="o", c="white", edgecolors="k", linewidths=0.6, zorder=5)
        ax.set_xlim(-b, b)
        ax.set_ylim(-b, b)
        ax.set_aspect("equal")
    elif task == "acrobot":
        c1, s1, c2, s2 = (df[k].to_numpy() for k in ("cos_th1", "sin_th1", "cos_th2", "sin_th2"))
        p1x, p1y = s1, -c1                                         # elbow (link length 1; theta_1 = 0 hangs down)
        p2x, p2y = p1x + (s1 * c2 + c1 * s2), p1y - (c1 * c2 - s1 * s2)   # tip
        ax.axhline(1.0, color="0.45", lw=0.6, ls=(0, (3, 2)), zorder=1)   # goal: tip above this line
        # only the (smoothly interpolated) trace of the tip, coloured by time step
        tf = np.linspace(t[0], t[-1], 6 * len(t))
        xf, yf = PchipInterpolator(t, p2x)(tf), PchipInterpolator(t, p2y)(tf)
        ax.add_collection(time_lines(xf, yf, tf, tmax, 1.6))
        ax.scatter(p2x[0], p2y[0], s=9, marker="o", c="white", edgecolors="k", linewidths=0.6, zorder=5)
        ax.scatter([0], [0], s=7, c="k", zorder=6)   # pivot
        ax.set_xlim(-2.1, 2.1)
        ax.set_ylim(-2.1, 1.55)
        ax.set_aspect("equal")
    else:  # discrete Mountain Car
        xs = np.linspace(-1.25, 0.65, 300)
        hill = lambda x: 0.45 * np.sin(3 * x) + 0.55
        ax.fill_between(xs, hill(xs), -0.15, color="0.85", lw=0, zorder=0)
        ax.plot(xs, hill(xs), color="0.55", lw=0.6, zorder=1)
        ax.axvline(0.5, ymin=0, ymax=1, color="0.45", lw=0.6, ls=(0, (3, 2)), zorder=1)   # goal position
        x = df["position"].to_numpy()
        # lift each visit slightly with time so that returns over the same stretch of hill stay visible
        y = hill(x) + 0.02 + 0.55 * t / tmax
        ax.add_collection(time_lines(x, y, t, tmax, 1.5))
        ax.scatter(x[0], y[0], s=9, marker="o", c="white", edgecolors="k", linewidths=0.6, zorder=5)
        ax.set_xlim(-1.25, 0.65)
        ax.set_ylim(-0.15, 1.65)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(0.4)
        s.set_color("0.6")


# --------------------------------------------------------------------------------------------- figure
def nice_ceil(v):
    for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000):
        if v <= step * 5 or step == 1000:
            return int(np.ceil(v / step) * step)


def make_figure(task, refresh):
    t = TASKS[task]
    models = load_models(task)
    trajs, nets, rets, dims, means, chosen, spread = [], [], [], [], [], [], []
    for row in models.itertuples():
        es, rp, df_ep, all_rets = representative(task, row, refresh)
        trajs.append(df_ep)
        chosen.append((es, rp))
        spread.append((all_rets.min(), all_rets.max()))
        net_path, cfg_path = model_paths(t, int(row.seed_idx))
        W = read_net(net_path)
        cfg = merged_config(t, cfg_path)
        nets.append((W, int(cfg["ann_nInput"]), int(cfg["ann_nOutput"])))
        rets.append(episode_return(task, trajs[-1]))
        means.append(float(row.reward))
        dims.append((len(W), int(np.sum(np.isfinite(W) & (W != 0)))))
    longest = max(len(d) for d in trajs)
    tmax = nice_ceil(longest)

    n_rows, n_cols = 6, 2
    ctitle, cdesc = CAPTIONS[task]
    ctext = COMMON.format(desc=cdesc)
    cap_h = caption_height(ctitle, ctext, JOURNAL_FULL_W)
    fig_h = 8.3 + cap_h
    fig = plt.figure(figsize=(JOURNAL_FULL_W, fig_h))
    grid = fig.add_gridspec(n_rows, n_cols, hspace=0.26, wspace=0.07, left=0.012, right=0.988, top=1 - 0.23 / fig_h,
                            bottom=(cap_h + 0.70) / fig_h)
    for i in range(N_NETS):
        r, c = divmod(i, n_cols)
        cell = grid[r, c].subgridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.04)
        ax_n = fig.add_subplot(cell[0])
        ax_a = fig.add_subplot(cell[1])
        W, n_in, n_out = nets[i]
        draw_network(ax_n, W, n_in, n_out)
        draw_agent(task, ax_a, trajs[i], tmax)
        ax_n.set_title(f"Net {i + 1}", loc="left", fontsize=7.5, fontweight="bold", pad=2)
        ax_n.set_title(f"{dims[i][0]} neurons · {dims[i][1]} synapses", loc="right", fontsize=6.5, pad=2, color="0.25")
        mfmt = f"{means[i]:.0f}" if task == "car" else f"{means[i]:.1f}"
        ax_a.set_title(f"mean {mfmt} · episode {rets[i]:.0f}", loc="right", fontsize=6.5, pad=2, color="0.25")
    # 12th cell: legend
    r, c = divmod(N_NETS, n_cols)
    ax_l = fig.add_subplot(grid[r, c])
    ax_l.axis("off")
    handles = [Line2D([], [], marker="o", ls="", mfc=C_BIAS, mec="0.25", mew=0.3, ms=4.5, label="bias"),
               Line2D([], [], marker="o", ls="", mfc=C_IN, mec="0.25", mew=0.3, ms=4.5, label="input (faded: unconnected)"),
               Line2D([], [], marker="o", ls="", mfc=C_HID, mec="0.25", mew=0.3, ms=4.5, label="hidden"),
               Line2D([], [], marker="o", ls="", mfc=C_OUT, mec="0.25", mew=0.3, ms=4.5, label="output"),
               Line2D([], [], color=C_EXC, lw=1.4, label="excitatory synapse"),
               Line2D([], [], color=C_INH, lw=1.4, label="inhibitory synapse"),
               Line2D([], [], marker="o", ls="", mfc="white", mec="k", mew=0.6, ms=4, label="start of the episode")]
    ax_l.legend(handles=handles, loc="center left", ncol=1, frameon=False, fontsize=7, handletextpad=0.6, labelspacing=0.45,
                borderaxespad=0.0, bbox_to_anchor=(0.02, 0.5))
    # time strip along the bottom
    ax_t = fig.add_axes([0.18, (cap_h + 0.33) / fig_h, 0.64, 0.13 / fig_h])
    ax_t.imshow(np.linspace(0, 1, 512)[None, :], aspect="auto", cmap=CMAP, extent=[0, tmax, 0, 1])
    ax_t.set_yticks([])
    ticks = np.linspace(0, tmax, 6) if tmax % 5 == 0 else [0, tmax]
    ax_t.set_xticks(ticks)
    ax_t.set_xlabel("Time step of the episode (colour of the agent's path)", fontsize=7.5, labelpad=2)
    ax_t.tick_params(axis="x", length=2.5, width=0.6, labelsize=7, pad=1.5)
    for s in ax_t.spines.values():
        s.set_linewidth(0.4)
    draw_caption(fig, ctitle, ctext)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{t['fig']}.pdf")
    fig.savefig(OUT / f"{t['fig']}.png")
    plt.close(fig)
    print(f"wrote {OUT / t['fig']}.pdf/.png   (episode lengths {[len(d) for d in trajs]}, colour scale 0-{tmax})")
    return dict(task=task, lengths=[len(d) for d in trajs], returns=rets, tmax=tmax, weights=models["weight_value"].tolist(),
                seeds=models["seed_idx"].tolist(), means=means, chosen=chosen, spread=spread, caption=(ctitle, ctext))


def write_caption(info):
    """Plain-text copy of the captions drawn inside the figures, plus the Net -> training seed mapping."""
    L = ["# Supplementary Figures S1-S3 - captions (the same text is drawn inside each figure)", ""]
    for i in info:
        title, text = i["caption"]
        L += [f"**{title}** {text}", "",
              "Net -> training seed (revalidated mean return): " + "; ".join(
                  f"Net {k + 1} = seed {sd} (mean {m:.1f}; episode {c[1]} of evaluation seed {c[0]}, return {r:.0f})"
                  for k, (sd, m, c, r) in enumerate(zip(i["seeds"], i["means"], i["chosen"], i["returns"]))), ""]
    (OUT / "suppl_figures_caption.md").write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-run the replay binaries even if the CSV cache exists")
    ap.add_argument("--task", choices=list(TASKS), nargs="*", default=list(TASKS))
    a = ap.parse_args()
    setup_journal_style()
    plt.rcParams["axes.grid"] = False
    infos = [make_figure(t, a.refresh) for t in a.task]
    if len(infos) == len(TASKS):
        write_caption(infos)


if __name__ == "__main__":
    main()
