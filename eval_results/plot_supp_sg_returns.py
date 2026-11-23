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

OUT = Path("graficos/paper/send_to_journal")
ORDER = [("acrobot", "Acrobot"), ("disc_mc", "Mountain Car"), ("car", "Racing Car")]
XS = {"sgrl": 0.0, "sgrl_con": 1.0, "sgrl_neu": 2.0}
XLAB = {"sgrl": "SG\n64×64", "sgrl_con": "SG\nsame\nconnections", "sgrl_neu": "SG\nsame\nneurons"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sg-repo", default="../surrogate-gradient-rl")
    a = ap.parse_args()
    q = pd.read_csv(Path(a.sg_repo) / "runs_quant" / "odin" / "quantization.csv")
    setup_journal_style()
    plt.rcParams["axes.grid"] = True
    fig, axs = plt.subplots(3, 1, figsize=(JOURNAL_FULL_W * 0.62, 7.0), sharex=True)
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
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / "suppl_figure_S4_sg_returns"
    for ext in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(ext))
    plt.close(fig)
    s = pd.DataFrame(rows)
    s.to_csv("eval_results/suppl_sg_returns_summary.csv", index=False)
    print(s.round(1).to_string(index=False))

    med = "; ".join(f"{lab.splitlines()[0]}: " + " / ".join(f"{s[(s.task == t) & (s.family == f)]['median'].iloc[0]:.1f}" for f in XS)
                    for t, lab in ORDER)
    sizes = "; ".join(f"{lab.splitlines()[0]}: {hid[t]['sgrl_con']} / {hid[t]['sgrl_neu']}" for t, lab in ORDER)
    (OUT / "suppl_figure_S4_caption.md").write_text(
        "**Figure S4. Return of the surrogate-gradient (SG) networks.** One row per task (Acrobot, Mountain Car with discrete "
        "actions, Racing Car) and one box per SG family: SG with 64×64 hidden units, SG with the same number of connections as the "
        "WANN networks, and SG with the same number of neurons as the WANN networks. Each point is one trained network (11 "
        "training seeds per family); boxes show median, interquartile range and whiskers at 1.5×IQR. Return is the mean "
        "environment return of each network in software (float32, no noise, no quantisation) over the evaluation episodes used in "
        "the quantisation study (Figure 3); higher is better. Medians (SG 64×64 / same connections / same neurons): "
        f"{med}. Hidden sizes of the same-connections / same-neurons networks: {sizes}.\n", encoding="utf-8")
    print("->", stem)


if __name__ == "__main__":
    main()
