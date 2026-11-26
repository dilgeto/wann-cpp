#!/usr/bin/env python3
"""
Unified ablation heatmaps (Acrobot | Mountain Car | Racing Car) in one
side-by-side figure. Reads the grids already produced by ablation_study.py
(eval_results/ablation/*_ablation_grid.csv) — runs nothing.

Output: graficos/paper/wann/ablation_heatmaps.png (+ _caption.md)
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_p3_weights import TASKS  # noqa: E402  (also anchors CWD to repo root)
from plot_style import JOURNAL_FULL_W, setup_journal_style, setup_publication_style  # noqa: E402

ABL = Path("eval_results/ablation")
OUT = Path("graficos/paper/wann")

# (panel title, TASKS key, grid-csv stem, networks-csv, run_key)
PANELS = [
    # Acrobot: `_w8` search (shared weights {0.5,1,2,3,5,8} since 2026-11-23). The previous panel was
    # ("acrobot", "ablation_networks.csv", "acrobot_small_first_spike"), old table {0.5..3}.
    ("Acrobot", "acrobot", "acrobot_small_first_spike_w8",
     "ablation_networks_acrobot_small_first_spike_w8.csv", "acrobot_small_first_spike_w8"),
    ("Mountain Car", "mountain_car", "disc_mc_small_rate_argmax_discrete",
     "ablation_networks_disc_mc_small_rate_argmax_discrete.csv", "disc_mc_small_rate_argmax_discrete"),
    ("Racing Car", "car", "car_ttfs_first_spike_discrete",
     "ablation_networks_car_ttfs_first_spike_discrete.csv", "car_ttfs_first_spike_discrete"),
]

ROWS = ["normal", "no_inh", "drop_exc", "flip_inh",
        "type1", "type2", "type3", "type4", "type5", "type6", "shuffle"]
LABELS = {
    "normal": "Normal", "no_inh": "No inhibitory", "drop_exc": "Control: K random exc. removed",
    "flip_inh": "Inhibitory → excitatory",
    "type1": "All RS", "type2": "All FS", "type3": "All CH", "type4": "All LTS",
    "type5": "All IB", "type6": "All RZ", "shuffle": "Control: types shuffled",
}
N_POL = 3  # rows after "normal" that belong to the polarity study


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", action="store_true",
                    help="journal version: panels stacked vertically at 190 mm wide, >= 7 pt text, embedded fonts; writes vector PDF + "
                         "600 dpi PNG to --journal-out instead of the default PNG (the caption is still written to the default folder)")
    ap.add_argument("--journal-out", type=Path, default=Path("graficos/paper/send_to_journal/figura_02"), help="stem, no extension")
    a = ap.parse_args()
    jr = a.journal
    (setup_journal_style if jr else setup_publication_style)()
    plt.rcParams["axes.grid"] = False
    if jr:
        fig, axes = plt.subplots(3, 1, figsize=(JOURNAL_FULL_W, 8.9))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharey=True)
    info = []
    for ax, (title, task, stem, netcsv, run_key) in zip(axes, PANELS):
        g = pd.read_csv(ABL / f"{stem}_ablation_grid.csv")
        nets = pd.read_csv(ABL / netcsv)
        win = int(nets[nets["run_key"] == run_key]["weight_index"].iloc[0])
        wis = sorted(g["wi"].unique())
        nm = {k: g[g["wi"] == k]["normal_mean"].iloc[0] for k in wis}

        def cell(c, k):
            return nm[k] if c == "normal" else nm[k] + g[(g["condition"] == c) & (g["wi"] == k)]["delta"].iloc[0]

        def sig(c, k):
            return c != "normal" and bool(g[(g["condition"] == c) & (g["wi"] == k)]["significant"].iloc[0])

        M = np.array([[cell(c, k) for k in wis] for c in ROWS])
        S = np.array([[sig(c, k) for k in wis] for c in ROWS])
        im = ax.imshow(M, cmap="YlGnBu", vmin=M.min(), vmax=M.max(), aspect="auto")
        wv = TASKS[task]["weight_vals"]
        ax.set_xticks(range(len(wis)))
        ax.set_xticklabels([f"{wv[k]:g}" for k in wis])
        ax.set_yticks(range(len(ROWS)))
        ax.set_yticklabels([LABELS[c] for c in ROWS])
        ax.set_xlabel("Shared weight value")
        ax.set_title(title)
        span = (M.max() - M.min()) or 1.0
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                dark = (M[i, j] - M.min()) / span > 0.55
                ax.text(j, i, f"{M[i, j]:.0f}" + ("*" if S[i, j] else ""), ha="center", va="center",
                        fontsize=7 if jr else 9, color="white" if dark else "black")
        ax.axhline(0.5, color="black", lw=0.8 if jr else 1.2)
        ax.axhline(N_POL + 0.5, color="black", lw=0.8 if jr else 1.2)
        wj = wis.index(win)  # winning weight: box around its "Normal" cell
        ax.add_patch(plt.Rectangle((wj - 0.5, -0.5), 1, 1, fill=False, ec="#D55E00", lw=1.6 if jr else 2.5))
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7 if jr else 10)
        info.append((title, run_key, wv[win], len(wis)))
    axes[0].set_ylabel("")
    fig.tight_layout(h_pad=0.8 if jr else 1.08)
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / "ablation_heatmaps.png"
    if jr:
        a.journal_out.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(a.journal_out.with_suffix(ext))
        png = a.journal_out.with_suffix(".pdf")
    else:
        fig.savefig(png)
    plt.close(fig)

    cap = [
        "**Ablation heatmaps of the winning SNN-WANN network of each task.** "
        "Each cell is the mean episode reward (n = 100 episodes, same seeds as the bootstrap) of the "
        "winning network after an ablation, evaluated at each of the six shared-weight values; "
        "darker = better, and each panel has its own colour scale. "
        "Rows, top to bottom (within each panel): the unmodified network; synaptic-polarity study (inhibitory synapses removed; "
        "control removing the same number K of random excitatory synapses; inhibitory synapses turned "
        "excitatory); neuron-type study (every neuron set to one Izhikevich type — RS, FS, CH, LTS, IB, RZ; "
        "control with types shuffled across neurons). Control rows are the mean over 30 random draws. "
        "An asterisk marks cells whose paired 95% bootstrap CI of the difference to the normal network "
        "excludes zero. The orange box marks the shared weight at which the network was selected. "
        "Panels: " + "; ".join(f"{t} ({k}, selected w = {w:g})" for t, k, w, _ in info) + ".",
    ]
    (OUT / "ablation_heatmaps_caption.md").write_text("\n".join(cap) + "\n")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
