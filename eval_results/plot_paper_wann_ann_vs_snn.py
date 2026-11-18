#!/usr/bin/env python3
"""
Paper figures, section "WANN": classical-ANN WANN (../wann) vs spiking WANN (this repo). Both sides ran the same
three-phase hyperparameter optimisation (screening_reduce -> screening_full phase 2 -> phase 3) and the same
revalidation protocol (eval_p3_weights.py: 11 training seeds x 11 evaluation seeds x 11 episodes, each network at
its own best shared weight, environment reward). Reads data only: nothing is simulated or trained.

Compared networks (replicate unit = trained network, n = 11 per family and task):
  ANN  ../wann/eval_results/<task>/<task>_best.csv                      (car rank02, acrobot rank02, disc_mc rank00)
  SNN  eval_results/<run_key>/..._best.csv (the 11 networks used in the paper: car_ttfs_first_spike_discrete r00,
       disc_mc_small_first_spike_discrete r01, acrobot_small_first_spike r01)

Figures -> graficos/paper/wann/<name>.png + <name>_caption.md (English, no title/footnotes inside the image):
  ann_vs_snn_performance     per-network revalidated return (best shared weight), box + points
  ann_vs_snn_agnosticism     weight agnosticism: floor-adjusted mean over the 6 shared weights / best weight
  ann_vs_snn_complexity      connections vs return per network
  ann_vs_snn_training        running-best training return vs genomes evaluated (median + IQR over seeds)
  ann_vs_snn_hyperopt        full-budget (phase 2) trial scores of the hyperparameter search + top-3 configs over seeds
  ann_vs_snn_best_episodes   episode returns of the single best network of each family (bootstrap-comparison data)
Also writes graficos/paper/wann/ann_vs_snn_stats.csv (medians, means, bootstrap CI of the difference, Mann-Whitney).

Usage:  python eval_results/plot_paper_wann_ann_vs_snn.py            (from anywhere)
"""
import os
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
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
from scipy import stats

from plot_style import setup_publication_style

ANN_ROOT = _ROOT.parent / "wann"
OUT = _ROOT / "graficos" / "paper" / "wann"
RNG = np.random.default_rng(0)
N_BOOT = 10000

C_ANN, C_SNN = "#0072B2", "#D55E00"
LAB_ANN, LAB_SNN = "WANN (ANN)", "WANN (SNN)"
TASKS = [  # key, label, floor, ann dir, snn run_key, snn best csv, snn mean-across-weights column
    ("car", "CarTrack", 0.0, "car", "car_ttfs_first_spike_discrete",
     "eval_results/car_ttfs_first_spike_discrete/car_best.csv"),
    ("disc_mc", "Mountain Car", -200.0, "disc_mc", "disc_mc_small_first_spike_discrete",
     "eval_results/disc_mc_small_first_spike_discrete/mountain_car_best.csv"),
    ("acrobot", "Acrobot", -500.0, "acrobot", "acrobot_small_first_spike",
     "eval_results/eval_p3_weights/acrobot_best.csv"),
]
SNN_RANK = {"car": 0, "disc_mc": 1, "acrobot": 1}


def load_nets(task, ann_dir, run_key, snn_csv):
    a = pd.read_csv(ANN_ROOT / "eval_results" / ann_dir / f"{ann_dir}_best.csv")
    a = a.rename(columns={"mean_across_weights": "mean_w"})
    s = pd.read_csv(_ROOT / snn_csv)
    s = s[(s["run_key"] == run_key) & (s["rank"] == SNN_RANK[task])].rename(
        columns={"mean_across_6weights": "mean_w"})
    assert len(a) == 11 and len(s) == 11, (task, len(a), len(s))
    return a.sort_values("seed_idx").reset_index(drop=True), s.sort_values("seed_idx").reset_index(drop=True)


def boot_diff_ci(x, y, stat=np.mean):
    """95 % percentile bootstrap CI of stat(y) - stat(x), resampling networks within each family."""
    d = [stat(RNG.choice(y, len(y))) - stat(RNG.choice(x, len(x))) for _ in range(N_BOOT)]
    return np.percentile(d, [2.5, 97.5])


def strip(ax, pos, vals, color, marker, width=0.28):
    ax.boxplot([vals], positions=[pos], widths=width * 1.6, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="none", edgecolor=color, linewidth=1.6),
               medianprops=dict(color=color, linewidth=2.2), whiskerprops=dict(color=color, linewidth=1.4),
               capprops=dict(color=color, linewidth=1.4), manage_ticks=False)
    jit = RNG.uniform(-width * 0.7, width * 0.7, len(vals))
    ax.scatter(pos + jit, vals, s=34, color=color, marker=marker, alpha=0.8, edgecolor="white",
               linewidth=0.6, zorder=3)


def legend_handles():
    return [Line2D([], [], marker="o", ls="", color=C_ANN, label=LAB_ANN, markersize=8),
            Line2D([], [], marker="s", ls="", color=C_SNN, label=LAB_SNN, markersize=8)]


def save(fig, name, caption):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    (OUT / f"{name}_caption.md").write_text(caption.strip() + "\n")
    print("wrote", OUT / f"{name}.png")


def main():
    setup_publication_style()
    data = {}
    for key, label, floor, ann_dir, run_key, csv in TASKS:
        data[key] = load_nets(key, ann_dir, run_key, csv)

    # ---------------- statistics table ----------------
    rows = []
    for key, label, floor, *_ in TASKS:
        a, s = data[key]
        ra, rs = a["reward"].to_numpy(), s["reward"].to_numpy()
        lo, hi = boot_diff_ci(ra, rs)
        lom, him = boot_diff_ci(ra, rs, np.median)
        u = stats.mannwhitneyu(rs, ra, alternative="two-sided")
        ag_a = (a["mean_w"] - floor) / (a["reward"] - floor)
        ag_s = (s["mean_w"] - floor) / (s["reward"] - floor)
        rows.append(dict(task=label, n=11, ann_median=np.median(ra), snn_median=np.median(rs),
                         ann_mean=ra.mean(), snn_mean=rs.mean(), diff_mean_snn_minus_ann=rs.mean() - ra.mean(),
                         diff_mean_ci_lo=lo, diff_mean_ci_hi=hi, diff_median_ci_lo=lom, diff_median_ci_hi=him,
                         mannwhitney_p=u.pvalue, ann_agnosticism_median=ag_a.median(),
                         snn_agnosticism_median=ag_s.median(),
                         ann_conn_median=a["n_connections"].median(), snn_conn_median=s["n_connections"].median(),
                         ann_neurons_median=a["n_neurons"].median(), snn_neurons_median=s["n_neurons"].median()))
    st = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    st.to_csv(OUT / "ann_vs_snn_stats.csv", index=False)
    S = {r.task: r for r in st.itertuples()}

    # ---------------- 1. performance ----------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, (key, label, floor, *_ ) in zip(axes, TASKS):
        a, s = data[key]
        strip(ax, 0, a["reward"].to_numpy(), C_ANN, "o")
        strip(ax, 1, s["reward"].to_numpy(), C_SNN, "s")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ANN", "SNN"]); ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel(label, fontweight="bold"); ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Return (best shared weight)")
    save(fig, "ann_vs_snn_performance", f"""
**ann_vs_snn_performance** — Revalidated return of the evolved WANN networks, ANN (blue) vs SNN (orange), one panel per task. Each point is one trained network (n = 11 per family; one per training seed) evaluated at its own best shared weight over 11 evaluation seeds x 11 episodes with the environment reward (no shaping); boxes show median and interquartile range, whiskers 1.5 x IQR. Both families use the winning configuration of the same three-phase hyperparameter optimisation (ANN: ../wann rank 2 / 2 / 0 for CarTrack / Acrobot / Mountain Car; SNN: car_ttfs_first_spike_discrete rank 0, acrobot_small_first_spike rank 1, disc_mc_small_first_spike_discrete rank 1, i.e. TTFS/first-spike for CarTrack and population/first-spike for the others). Medians ANN vs SNN: CarTrack {S['CarTrack'].ann_median:.0f} vs {S['CarTrack'].snn_median:.0f}; Mountain Car {S['Mountain Car'].ann_median:.1f} vs {S['Mountain Car'].snn_median:.1f}; Acrobot {S['Acrobot'].ann_median:.1f} vs {S['Acrobot'].snn_median:.1f}. Difference of means SNN − ANN (95 % bootstrap CI over networks, 10 000 resamples): CarTrack {S['CarTrack'].diff_mean_snn_minus_ann:.0f} [{S['CarTrack'].diff_mean_ci_lo:.0f}, {S['CarTrack'].diff_mean_ci_hi:.0f}]; Mountain Car {S['Mountain Car'].diff_mean_snn_minus_ann:.1f} [{S['Mountain Car'].diff_mean_ci_lo:.1f}, {S['Mountain Car'].diff_mean_ci_hi:.1f}]; Acrobot {S['Acrobot'].diff_mean_snn_minus_ann:.1f} [{S['Acrobot'].diff_mean_ci_lo:.1f}, {S['Acrobot'].diff_mean_ci_hi:.1f}]. Higher is better in all panels (Mountain Car and Acrobot returns are negative). Values in ann_vs_snn_stats.csv.""")

    # ---------------- 2. agnosticism ----------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, (key, label, floor, *_ ) in zip(axes, TASKS):
        a, s = data[key]
        for pos, v, color, marker, hollow in ((0, a["reward"], C_ANN, "o", False), (1, a["mean_w"], C_ANN, "o", True),
                                              (2.4, s["reward"], C_SNN, "s", False), (3.4, s["mean_w"], C_SNN, "s", True)):
            strip(ax, pos, v.to_numpy(), color, marker)
            if hollow:  # mean over weights: redraw the points hollow
                ax.collections[-1].set_facecolor("white"); ax.collections[-1].set_edgecolor(color)
                ax.collections[-1].set_linewidth(1.6)
        ax.set_xticks([0, 1, 2.4, 3.4]); ax.set_xticklabels(["best", "mean", "best", "mean"]); ax.set_xlim(-0.7, 4.1)
        ax.set_xlabel(label, fontweight="bold"); ax.grid(axis="x", visible=False)
        ax.text(0.5, -0.2, "ANN", transform=ax.get_xaxis_transform(), ha="center", color=C_ANN, fontsize=12)
        ax.text(2.9, -0.2, "SNN", transform=ax.get_xaxis_transform(), ha="center", color=C_SNN, fontsize=12)
    axes[0].set_ylabel("Return")
    ag = {k: st.loc[st.task == l].iloc[0] for k, l, *_ in TASKS}
    def m(k, f): return ag[k][f]
    save(fig, "ann_vs_snn_agnosticism", f"""
**ann_vs_snn_agnosticism** — Weight agnosticism of each evolved network: return at the network's own best shared weight ("best", filled) next to the return averaged over all six training shared weights ("mean", hollow), ANN (blue) and SNN (orange), one panel per task. One point per network (n = 11 per family), box = median and IQR. The closer "mean" is to "best", the less the network depends on the selected weight; the gap is the price of not tuning the weight. The weight grids differ between families (ANN: {{−2, −1, −0.5, 0.5, 1, 2}} with sign; SNN: {{0.5, 1, 2, 3, 5, 8}} synaptic magnitude), so the comparison is of how flat each family's return is over its own grid, not of identical weight values; the six weights are the training ones, so this is in-sample agnosticism and overstates out-of-grid agnosticism. Median floor-adjusted ratio (R_mean − floor)/(R_best − floor), floors 0 / −200 / −500, ANN vs SNN: CarTrack {m('car','ann_agnosticism_median')*100:.0f} % vs {m('car','snn_agnosticism_median')*100:.0f} %; Mountain Car {m('disc_mc','ann_agnosticism_median')*100:.0f} % vs {m('disc_mc','snn_agnosticism_median')*100:.0f} %; Acrobot {m('acrobot','ann_agnosticism_median')*100:.0f} % vs {m('acrobot','snn_agnosticism_median')*100:.0f} % (the ratio is not plotted because it is unstable when a network's best-weight return is near the floor). A network can have mean above best because "best" is the weight chosen during training and revalidated on different seeds.""")

    # ---------------- 3. complexity ----------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    for ax, (key, label, floor, *_ ) in zip(axes, TASKS):
        a, s = data[key]
        ax.scatter(a["n_connections"], a["reward"], s=44, color=C_ANN, marker="o", alpha=0.85, edgecolor="white")
        ax.scatter(s["n_connections"], s["reward"], s=44, color=C_SNN, marker="s", alpha=0.85, edgecolor="white")
        ax.set_xscale("log"); ax.xaxis.set_minor_formatter(NullFormatter()); ax.set_xlabel(f"Connections\n{label}", fontweight="bold")
    axes[0].set_ylabel("Return (best shared weight)")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.07), frameon=False)
    save(fig, "ann_vs_snn_complexity", f"""
**ann_vs_snn_complexity** — Final network size (number of enabled connections, log scale) against revalidated return, one point per trained network (n = 11 per family), per task. Median connections ANN vs SNN: CarTrack {S['CarTrack'].ann_conn_median:.0f} vs {S['CarTrack'].snn_conn_median:.0f}; Mountain Car {S['Mountain Car'].ann_conn_median:.0f} vs {S['Mountain Car'].snn_conn_median:.0f}; Acrobot {S['Acrobot'].ann_conn_median:.0f} vs {S['Acrobot'].snn_conn_median:.0f}. Median neurons (hidden + I/O) ANN vs SNN: CarTrack {S['CarTrack'].ann_neurons_median:.0f} vs {S['CarTrack'].snn_neurons_median:.0f}; Mountain Car {S['Mountain Car'].ann_neurons_median:.0f} vs {S['Mountain Car'].snn_neurons_median:.0f}; Acrobot {S['Acrobot'].ann_neurons_median:.0f} vs {S['Acrobot'].snn_neurons_median:.0f}. Sizes are counted by the same routine in both repositories on the exported network file (nodes incl. inputs/outputs; non-zero matrix entries). Caveat: the ANN code base predates the 2026-10-23 fix of disabled hidden-to-hidden genes being re-expressed as enabled; ANN training and evaluation both use the expressed matrix, so the evaluated ANN network is self-consistent, but its connection count includes those re-enabled edges.""")

    # ---------------- 4. training curves ----------------
    def curves(path_glob_dir, prefix):
        out = []
        for f in sorted(Path(path_glob_dir).glob(f"{prefix}_seed*_stats.out")):
            d = np.loadtxt(f, delimiter=",")
            out.append((d[:, 0], d[:, 4]))  # genomes evaluated, running-best peak fitness (shaped, best over the 6 weights)
        return out

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    YLIM = {"car": (0, 1600), "disc_mc": (-200, -65), "acrobot": (-130, -30)}
    info = []
    for ax, (key, label, floor, ann_dir, run_key, _) in zip(axes, TASKS):
        a, s = data[key]
        ar, sr = int(a["rank"].iloc[0]), SNN_RANK[key]
        cs = {"ANN": curves(ANN_ROOT / "log" / f"full_p3_{ann_dir}", f"rank{ar:02d}"),
              "SNN": curves(_ROOT / "log" / f"full_p3_{run_key}", f"rank{sr:02d}")}
        for fam, color in (("ANN", C_ANN), ("SNN", C_SNN)):
            xmax = max(c[0][-1] for c in cs[fam])
            grid = np.linspace(0, xmax, 400)
            M = np.full((len(cs[fam]), len(grid)), np.nan)
            for i, (x, y) in enumerate(cs[fam]):
                idx = np.searchsorted(x, grid, side="right") - 1
                ok = (grid >= x[0]) & (grid <= x[-1])
                M[i, ok] = y[idx[ok]]
            med = np.nanmedian(M, axis=0)
            q1, q3 = np.nanpercentile(M, 25, axis=0), np.nanpercentile(M, 75, axis=0)
            ax.plot(grid / 1e3, med, color=color, lw=2.2)
            ax.fill_between(grid / 1e3, q1, q3, color=color, alpha=0.2, lw=0)
            info.append((label, fam, len(cs[fam]), xmax))
        ax.set_xscale("log"); ax.set_ylim(*YLIM[key])
        ax.set_xlabel(f"Genomes (x10$^3$)\n{label}", fontweight="bold")
    axes[0].set_ylabel("Running-best peak training fitness")
    axes[0].legend(handles=[Line2D([], [], color=C_ANN, lw=2.5, label=LAB_ANN),
                            Line2D([], [], color=C_SNN, lw=2.5, label=LAB_SNN)], loc="lower right")
    runs = "; ".join(f"{l} {f}: {n} runs, up to {x/1e3:.0f}k genomes" for l, f, n, x in info)
    save(fig, "ann_vs_snn_training", f"""
**ann_vs_snn_training** — Training progress of the selected configuration of each family: peak training fitness of the running-best individual (best over the six shared weights, as logged by the DataGatherer, column fitPeak; includes the reward-shaping term, scale 10 for Acrobot and Mountain Car in both families, none for CarTrack) against the cumulative number of genomes evaluated (population 480 in both families; each genome is evaluated at 6 shared weights x 6 repetitions in both). Genome axis is logarithmic and the y axis is truncated (early values reach about −200 for Mountain Car and −500 for Acrobot in the SNN). Line = median over the 11 training seeds, band = interquartile range; curves are step functions resampled on a common grid and each family is drawn up to the longest run. Number of runs and range: {runs}. These are selection-inflated training values (maximum over weights and over the evolving population, shaped fitness), not revalidated environment returns; compare with ann_vs_snn_performance for unbiased values. The unshaped fitTopOrig column is not used because the two code bases do not log it equivalently. Wall-clock cost per genome is much higher for the SNN (spiking simulation) and is not shown.""")

    # ---------------- 5. hyperparameter optimisation ----------------
    P2 = {"car": (ANN_ROOT / "screening_full/car", _ROOT / "screening_full/car_ttfs_first_spike_discrete"),
          "disc_mc": (ANN_ROOT / "screening_full/disc_mc", _ROOT / "screening_full/disc_mc_small_first_spike_discrete"),
          "acrobot": (ANN_ROOT / "screening_full/acrobot", _ROOT / "screening_full/acrobot_small_first_spike")}
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), gridspec_kw=dict(height_ratios=[1, 1]))
    notes = []
    for j, (key, label, floor, *_ ) in enumerate(TASKS):
        da, ds = P2[key]
        ax = axes[0, j]
        for pos, d, color, marker in ((0, da, C_ANN, "o"), (1, ds, C_SNN, "s")):
            v = pd.read_csv(d / "p2_results.csv")["peak_fitness"].to_numpy()
            strip(ax, pos, v, color, marker)
            notes.append((label, "ANN" if pos == 0 else "SNN", len(v), np.median(v), v.max()))
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ANN", "SNN"]); ax.set_xlim(-0.6, 1.6)
        ax.grid(axis="x", visible=False)
        ax = axes[1, j]
        for pos, d, color, marker in ((-0.12, da, C_ANN, "o"), (0.12, ds, C_SNN, "s")):
            p3 = pd.read_csv(d / "p3_summary.csv").sort_values("rank").head(3)
            ax.errorbar(p3["rank"] + pos, p3["mean"], yerr=p3["std"], fmt=marker, color=color,
                        capsize=3, ms=7, lw=1.4)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["1st", "2nd", "3rd"]); ax.set_xlim(-0.6, 2.6)
        ax.set_xlabel(label, fontweight="bold"); ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Phase-2 trial peak fitness")
    axes[1, 0].set_ylabel("Phase-3 return over 11 seeds")
    fig.legend(handles=legend_handles(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.97), frameon=False)
    nt = "; ".join(f"{l} {f}: {n} trials, median {m:.0f}, max {mx:.0f}" for l, f, n, m, mx in notes)
    save(fig, "ann_vs_snn_hyperopt", f"""
**ann_vs_snn_hyperopt** — Effect of the hyperparameter optimisation in each family. Top row: peak fitness (best individual over the six shared weights, shaped training fitness) of every phase-2 full-budget trial of the search, one point per sampled configuration (box = median and IQR). Bottom row: phase-3 re-runs of the three best phase-2 configurations over 11 training seeds each, mean ± s.d. over seeds of the peak training fitness (shaped, selection-inflated; columns mean/std of p3_summary.csv). Searched hyperparameters: ANN — probMoo, addConn, addNode, enable, mutAct, initEnable, cull/elite ratio, tournament size; SNN — the same plus excitatory-toggle probability and, for the SNN, the window, tau_exc/tau_inh and TTFS threshold microparameters (so the SNN search space is larger). Trial counts and statistics: {nt}. Trial counts differ between families, so spread in the top row reflects both search-space size and budget, not only sensitivity to hyperparameters. Shaping scale is identical (10) for Acrobot and Mountain Car in both families.""")

    # ---------------- 6. best-network episodes ----------------
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.2))
    parts = []
    for ax, (key, label, floor, *_ ) in zip(axes, TASKS):
        f = ANN_ROOT / "bootstrap_results" / "snn_vs_wann" / key / "rewards.csv"
        r = pd.read_csv(f)
        ra = r[r["agent"] == "wann"]["reward"].to_numpy(); rs = r[r["agent"] == "wann_snn"]["reward"].to_numpy()
        strip(ax, 0, ra, C_ANN, "o", 0.3); strip(ax, 1, rs, C_SNN, "s", 0.3)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ANN", "SNN"]); ax.set_xlim(-0.6, 1.6)
        ax.set_xlabel(label, fontweight="bold"); ax.grid(axis="x", visible=False)
        ci = pd.read_csv(f.parent / "ci_results.csv").iloc[0]
        parts.append(f"{label}: mean ANN {ra.mean():.1f} vs SNN {rs.mean():.1f}, difference SNN − ANN "
                     f"{ci['diff_point']:.1f} [{ci['diff_ci_lo']:.1f}, {ci['diff_ci_hi']:.1f}] (SNN − ANN)")
    axes[0].set_ylabel("Episode return")
    save(fig, "ann_vs_snn_best_episodes", f"""
**ann_vs_snn_best_episodes** — Episode returns (100 evaluation seeds, one episode each, environment reward) of the single highest-scoring revalidated network of each family. This is the data of ../wann/bootstrap_results/bootstrap_compare_snn_vs_wann.py; the unpaired bootstrap there (10 000 resamples, 95 % CI) is reported in its ci_results.csv: {" | ".join(parts)}. The winner of each family is selected as the maximum over many networks and weights, so this comparison is optimistic and the per-network figure ann_vs_snn_performance (all 11 networks) is the unbiased one. Positive differences mean the SNN scored higher.""")

    print(st.round(3).T.to_string())


if __name__ == "__main__":
    main()
