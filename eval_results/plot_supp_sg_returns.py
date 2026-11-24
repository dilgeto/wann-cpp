#!/usr/bin/env python3
"""
Supplementary figure S4: noise-free environment return of the surrogate-gradient (SG) networks only, one row per task
(Acrobot, Mountain Car (discrete), Racing Car), one box per SG family (64x64, same connections as the WANN nets, same neurons
as the WANN nets), one point per trained network (11 seeds per family). Return = mean environment return per network in
software (float32), i.e. the "before quantisation" values of ../surrogate-gradient-rl/runs_quant/odin/quantization.csv that
the paper's Figure 3 starts from (same 3 x 11 networks, same episodes). Reads CSVs only: nothing is simulated or trained.

Output: graficos/paper/send_to_journal/suppl_figure_S4_sg_returns.{pdf,png} (journal style, 190 mm wide) + _caption.md
        eval_results/suppl_sg_returns_summary.csv

Usage:  python eval_results/plot_supp_sg_returns.py [--sg-repo ../surrogate-gradient-rl]     (from anywhere)
"""
import argparse
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(_SCRIPT_DIR.parent)
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_noise_compare import METHODS
from plot_quant_noisefree import FAMILIES, box, sgrl_pairs
from plot_style import JOURNAL_FULL_W, setup_journal_style
from supp_caption import caption_height, draw_caption

OUT = Path("graficos/paper/send_to_journal")
ORDER = [("acrobot", "Acrobot"), ("disc_mc", "Mountain Car"), ("car", "Racing Car")]
XS = {"sgrl": 0.0, "sgrl_con": 1.0, "sgrl_neu": 2.0}
CAP_TITLE = "Fig. S4. Return of the surrogate-gradient (SG) networks."
CAP_TEXT = ("These are the SG networks compared with WANN+SNN in the reported results (Figs. 3, 4 and 5). Environment return of the 11 SG networks of each family (64×64 hidden units; same number of connections as the WANN "
            "networks; same number of neurons), one point per network, in software (float32, no noise). Boxes: quartiles and "
            "median; whiskers: 1.5 × interquartile range.")
XLAB = {"sgrl": "SG\n64×64", "sgrl_con": "SG\nsame\nconnections", "sgrl_neu": "SG\nsame\nneurons"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sg-repo", default="../surrogate-gradient-rl")
    a = ap.parse_args()
    q = pd.read_csv(Path(a.sg_repo) / "runs_quant" / "odin" / "quantization.csv")
    setup_journal_style()
    plt.rcParams["axes.grid"] = True
    fw = JOURNAL_FULL_W * 0.62
    cap_h = caption_height(CAP_TITLE, CAP_TEXT, fw)
    fig_h = 7.0 + cap_h
    fig, axs = plt.subplots(3, 1, figsize=(fw, fig_h), sharex=True)
    rows, hid = [], {}
    for ax, (task, label) in zip(axs, ORDER):
        models = pd.read_csv(f"eval_results/{task}_replicas/models.csv")
        pairs = sgrl_pairs(task, models, q)
        hid[task] = {s: str(models[models.source == s].hidden.iloc[0]).replace("x", "×") for s, _ in FAMILIES}
        allv = []
        for src, _ in FAMILIES:
            v = pairs[src]["float"].to_numpy()
            assert len(v) == 11, (task, src, len(v))
            box(ax, XS[src], v, METHODS[src]["color"], alpha=0.5, width=0.5)
            allv.append(v)
            rows.append(dict(task=task, family=src, n=len(v), median=np.median(v), mean=v.mean(), q1=np.percentile(v, 25),
                             q3=np.percentile(v, 75), min=v.min(), max=v.max()))
        allv = np.concatenate(allv)
        pad = 0.06 * (allv.max() - allv.min())
        ax.set_ylim(allv.min() - pad, allv.max() + pad)
        ax.set_ylabel(f"{label}\nreturn")
        ax.set_xlim(-0.55, 2.55)
    axs[-1].set_xticks(list(XS.values()))
    axs[-1].set_xticklabels([XLAB[s] for s in XS])
    fig.tight_layout(rect=(0, cap_h / fig_h, 1, 1))
    draw_caption(fig, CAP_TITLE, CAP_TEXT)
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "suppl_figure_S4_sg_returns"
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
    plt.close(fig)
    s = pd.DataFrame(rows)
    s.to_csv("eval_results/suppl_sg_returns_summary.csv", index=False)
    print(s.round(1).to_string(index=False))

    (OUT / "suppl_figure_S4_caption.md").write_text(f"**{CAP_TITLE}** {CAP_TEXT}\n", encoding="utf-8")
    print("->", stem)


if __name__ == "__main__":
    main()
