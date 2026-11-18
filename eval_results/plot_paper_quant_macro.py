#!/usr/bin/env python3
"""
Macro figure (3 tasks stacked: Acrobot, Mountain Car, Racing Car) of the reward lost by each method when its network is moved to
low-precision hardware, no memristor noise:

  SG x3  float32 -> 3-bit weights (PTQ, scheme `odin`), software, same network and same episodes. Identical to the SG boxes of
           graficos/<Tarea>/ruido/quant_noisefree/<tarea>_quant_noisefree_pct.png (same data, same loss, same exclusions).
  WANN     two boxes: (1) software, the box of the original _pct figures: training-chosen shared weight -> best integer level 1-7
           (not a quantization effect; negative values are gains from a better selection), and (2) software -> ODIN chip
           (ZCU104 board), each network calibrated for REWARD (reward-oriented calibration).

Loss (%) = (R_before - R_after) / (R_before - floor) * 100  (positive = worse, 0 = no loss, 100 = task not accomplished).
Floors: Acrobot -500, Mountain Car -200, Racing Car 0.

WANN data: ../ODIN-SG-WANN-datos/comparacion_sg_wann/perdida_sg_vs_wann_2026-11-11.csv (column perdida_pct, rows metodo == wann), restricted
to the same WANN family as the quantization figures: acrobot_small_first_spike r01 (small_fs), disc_mc_small_rate_argmax_discrete r01
(small_ra), car_ttfs_first_spike_discrete r00. Nothing is simulated: the script only reads CSVs already computed.

Outputs (default): graficos/paper/fpga/quant_vs_odin_macro.png, its _caption.md and eval_results/quant_vs_odin_macro_summary.csv.
Usage: python eval_results/plot_paper_quant_macro.py [--odin-csv CSV] [--out PNG]
"""
import argparse
import json
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
from plot_quant_noisefree import FAMILIES, TASKS, box, drop_pct, sgrl_pairs, wann_original
from plot_style import JOURNAL_FULL_W, setup_journal_style, setup_publication_style
from stats_noise_replicas import network_table

ODIN_CSV = Path("../ODIN-SG-WANN-datos/comparacion_sg_wann/perdida_sg_vs_wann_2026-11-11.csv")
# macro-figure order -> (task key of plot_quant_noisefree, tarea/red-prefix in the ODIN CSV, panel label)
ORDER = [("acrobot", "acrobot", "small_fs/", "Acrobot"),
         ("disc_mc", "mountain_car", "small_ra/", "Mountain Car\n(discrete)"),
         ("car", "coche", "", "Racing Car")]
XS = {"wann": 0.0, "wann_odin": 1.0, "sgrl": 2.4, "sgrl_con": 3.4, "sgrl_neu": 4.4}
WANN_GRAY = "0.45"


def wann_odin_loss(df: pd.DataFrame, tarea: str, prefix: str) -> np.ndarray:
    w = df[(df.metodo == "wann") & (df.tarea == tarea) & df.red.str.startswith(prefix)]
    return w.perdida_pct.to_numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--odin-csv", type=Path, default=ODIN_CSV)
    ap.add_argument("--sg-repo", default="../surrogate-gradient-rl")
    ap.add_argument("--out", type=Path, default=Path("graficos/paper/fpga/quant_vs_odin_macro.png"))
    ap.add_argument("--journal", action="store_true",
                    help="draw at the final journal size (190 mm wide, >= 7 pt text, embedded fonts) and write vector PDF + 600 dpi PNG "
                         "to --journal-out instead of --out (the caption and summary CSV are still written next to --out)")
    ap.add_argument("--journal-out", type=Path, default=Path("graficos/paper/send_to_journal/figura_03"), help="stem, no extension")
    a = ap.parse_args()
    odin = pd.read_csv(a.odin_csv)
    qcsv = pd.read_csv(Path(a.sg_repo) / "runs_quant" / "odin" / "quantization.csv")
    (setup_journal_style if a.journal else setup_publication_style)()

    fig, axs = plt.subplots(3, 1, figsize=(JOURNAL_FULL_W, 7.6) if a.journal else (8.2, 9.6), sharex=True)
    rows, hid_all = [], {}
    for ax, (task, tarea, prefix, label) in zip(axs, ORDER):
        D = Path(f"eval_results/{task}_replicas")
        models = pd.read_csv(D / "models.csv")
        proto = json.loads((D / "protocol.json").read_text())
        floor = float(proto["ret_floor"])
        min_den = 0.05 * abs(floor) if floor else 25.0
        pairs = sgrl_pairs(task, models, qcsv)
        hid_all[task] = {s: str(models[models.source == s].hidden.iloc[0]).replace("x", "×") for s, _ in FAMILIES}

        groups = []
        # WANN original: training-chosen weight -> best integer level 1-7 (same computation as plot_quant_noisefree.pct_figure)
        wo = wann_original(task, proto)
        _, base, _ = network_table(D, "best")
        wl_s = base[base.source == "wann"].set_index("replica_seed").ret_abs.sort_index()
        dw, n_exw = drop_pct(wo.rename_axis("seed").sort_index().reset_index(drop=True), wl_s.reset_index(drop=True), floor, min_den)
        groups.append(("wann", dw.dropna().to_numpy(), n_exw))
        groups.append(("wann_odin", wann_odin_loss(odin, tarea, prefix), 0))
        for src, _ in FAMILIES:
            p = pairs[src]
            d, n_ex = drop_pct(p["float"].reset_index(drop=True), p.q3.reset_index(drop=True), floor, min_den)
            groups.append((src, d.dropna().to_numpy(), n_ex))

        for src, v, n_ex in groups:
            if src == "wann":
                box(ax, XS[src], v, METHODS["wann"]["color"], hatch="///", alpha=0.5, width=0.5)
            elif src == "wann_odin":
                box(ax, XS[src], v, METHODS["wann"]["color"], alpha=0.5, width=0.5)
            else:
                box(ax, XS[src], v, METHODS[src]["color"], alpha=0.5, width=0.5)
            rows.append(dict(task=task, group=src, n=len(v), n_excluded=n_ex, median_pct=float(np.median(v)),
                             q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)), min_pct=float(v.min()),
                             max_pct=float(v.max()), n_worse=int((v > 0).sum())))
        allv = np.concatenate([v for _, v, _ in groups])
        span = allv.max() - allv.min()
        ax.set_ylim(min(allv.min() - 0.06 * span, -0.04 * span), allv.max() + 0.06 * span)
        ax.axhline(0, color="0.4", lw=1, ls=":", zorder=1)
        if allv.max() >= 100:
            ax.axhline(100, color="0.4", lw=1, ls="--", zorder=1)
        ax.set_ylabel(f"{label}\nreward lost (%)")
        ax.set_xlim(-0.6, 5.0)

    axs[-1].set_xticks(list(XS.values()))
    axs[-1].set_xticklabels(["WANN\n(software)", "WANN\non ODIN", "SG\n64×64", "SG\nsame\nconnections", "SG\nsame\nneurons"],
                            fontsize=7 if a.journal else 11)
    fig.tight_layout()      # no legend: what each box is lives in the x tick labels and in the caption
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.journal:
        a.journal_out.parent.mkdir(parents=True, exist_ok=True)
        for ext in (".pdf", ".png"):
            fig.savefig(a.journal_out.with_suffix(ext))
    else:
        fig.savefig(a.out)
    plt.close(fig)

    summ = pd.DataFrame(rows)
    summ.to_csv("eval_results/quant_vs_odin_macro_summary.csv", index=False)
    print(summ.round(2).to_string(index=False))
    write_caption(a.out.with_name(a.out.stem + "_caption.md"), summ, hid_all)
    print("->", a.out)


def write_caption(path: Path, s: pd.DataFrame, hid: dict):
    def m(task, g):
        r = s[(s.task == task) & (s.group == g)].iloc[0]
        return r
    names = {"wann": "WANN software", "wann_odin": "WANN on ODIN", "sgrl": "SG 64×64", "sgrl_con": "SG same connections", "sgrl_neu": "SG same neurons"}
    tasks = [("acrobot", "Acrobot"), ("disc_mc", "Mountain Car"), ("car", "Racing Car")]
    med = "; ".join(f"{t}: " + " / ".join(f"{m(k, g).median_pct:.1f}" for g in names) for k, t in tasks)
    nn = "; ".join(f"{t}: " + " / ".join(f"{int(m(k, g).n)}" for g in names) for k, t in tasks)
    worse = "; ".join(f"{t}: " + " / ".join(f"{int(m(k, g).n_worse)}" for g in names) for k, t in tasks)
    excl = [f"{t} {names[g]} ({int(m(k, g).n_excluded)})" for k, t in tasks for g in names if m(k, g).n_excluded]
    sizes = "; ".join(f"{t}: {hid[k]['sgrl_con']} / {hid[k]['sgrl_neu']}" for k, t in tasks)
    path.write_text(
        "**Figure X. Reward lost when the networks are moved to low-precision hardware (no memristor noise), per task.**\n"
        "Each marker is one trained network; boxes show the quartiles, the line the median, and whiskers extend to 1.5 times the "
        "interquartile range. Loss is 100·(R_before − R_after)/(R_before − R_floor), with floors of −500 (Acrobot), −200 (Mountain "
        "Car, discrete) and 0 (Racing Car): 0% (dotted line) means no loss, 100% (dashed line, shown when reached) means the task is "
        "not accomplished, and negative values mean the network scores higher after the change. "
        "WANN, software (hatched): not a quantization effect; WANN weights are not quantized, every synapse takes the single shared "
        "value. R_before is the shared weight chosen by training (one of the six, re-measured on new episodes) and R_after the best "
        "integer level 1-7 (the ODIN weight levels), chosen on independent episodes and measured on others; negative values are gains "
        "from a better selection, not from the change of precision. "
        "WANN on ODIN (plain blue): networks moved from software to the ODIN chip (ZCU104 board), each one calibrated to maximise "
        "reward (ODIN neuron parameters chosen per network; weights and connectivity unchanged); R_after is the board mean "
        "over 100 episodes. "
        "SG (plain): surrogate-gradient SNNs with 64×64 hidden units, with the same connections as the WANN networks, and with the "
        "same neurons as the WANN networks; R_before is the float32 network and R_after the same network with its weights rounded to 3 "
        "bits (post-training quantization, biases kept in float32), evaluated on the same episodes. "
        "The three kinds of loss are therefore different interventions: WANN on ODIN includes the change of neuron model and its "
        "reward calibration (which can exceed the software reward, hence negative losses), SG's is weight rounding alone. "
        f"Hidden sizes of the SG same-connections / same-neurons networks: {sizes}. "
        f"Medians (WANN software / WANN on ODIN / SG 64×64 / SG same connections / SG same neurons), %: {med}. "
        f"Networks per box: {nn}. Networks with a positive loss: {worse}. "
        + ("Networks excluded because their starting reward lies within 5% of the range of the floor (loss undefined): "
           + ", ".join(excl) + ". " if excl else "")
        + "Racing Car, WANN on ODIN: one of 11 networks (seed02) is excluded because its software reward is at or below the floor. "
        "Evaluation episodes differ between WANN (ODIN) and SG (software); within each method, before and after use the same "
        "episodes. WANN networks: acrobot_small_first_spike rank01, disc_mc_small_rate_argmax_discrete rank01 and "
        "car_ttfs_first_spike_discrete rank00, 11 seeds each.\n", encoding="utf-8")
    print("->", path)


if __name__ == "__main__":
    main()
