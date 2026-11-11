#!/usr/bin/env python3
"""
Paper figures that combine conductance occupancy with the noise types: WANN vs. SGRL (same connections / same neurons),
float32, on PCM and ReRAM-CMO (Falcone range 8-90 uS), at occupancies 6.25 / 12.5 / 25 / 50 / 100 % of g_max.
Reads data only; nothing is simulated or trained. Data (produced by eval_results/run_paper_noise_occupancy.sh):

  eval_results/paper_noise_occ/<task>_replicas/   WANN (source wann_occ) at every occupancy and SGRL (sgrl_*_occ) at
                                                  6.25-50 %, PCM + ReRAM-CMO
  eval_results/<task>_replicas/                   SGRL at 100 % (its default mapping) and the noise-free baselines, PCM
  eval_results/paper_noise/<task>_replicas/       SGRL at 100 %, ReRAM-CMO (Falcone)

Occupancy = fraction of g_max at which the programmed weight sits. WANN: every weight is +/- the network's ODIN level
(chosen noise-free), written at that fraction of g_max (w_max = weight / occupancy). SGRL: the largest |w| of each layer
at that fraction (all other weights lower). Retention = (return - floor)/(noise-free return - floor) x 100 per network.

Outputs (graficos/paper/noise_occupancy/):
  noise_vs_occupancy_pcm.png, noise_vs_occupancy_reramcmo.png   retention vs. occupancy for each noise type (columns)
                                                                and task (rows): the figure that shows how occupancy
                                                                interacts with each mechanism
  occ<NN>/noise_{intensity,drift,boxplots,ablation}.png          the four noise-type figures of plot_paper_noise.py at
                                                                one fixed occupancy (NN = 6p25, 12p5, 25, 50, 100)
and eval_results/paper_noise_occ/summary_by_occupancy.csv.

Usage:
  python eval_results/plot_paper_noise_occupancy.py
  python eval_results/plot_paper_noise_occupancy.py --occupancies 0.5 1     # only some levels
"""
import argparse
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
from matplotlib.ticker import FixedFormatter, FixedLocator

import plot_paper_noise as P
from plot_style import setup_publication_style
from stats_noise_replicas import network_table

# SRC is read from P.SRC at call time (it gains sgrl_64 when that data exists)
KEEP = ["source", "model_id", "noise_model", "mechanism", "scale", "t_inference", "seed", "chip", "read", "weight_rmse",
        "ret", "prog_noise_scale", "read_noise_scale", "drift_scale"]
YEAR = 31536000.0
OCCS_ALL = [0.0625, 0.125, 0.25, 0.5, 1.0]
VARIANT = {"sgrl_con": "sgrl_con_occ", "sgrl_neu": "sgrl_neu_occ", "sgrl_64": "sgrl_64_occ"}
FULL_FROM_OCC = {"sgrl_64"}   # SGRL 64x64 (new methodology): simulated at 100 % too, no earlier data to reuse


def tag(o):
    return f"{o * 100:g}".replace(".", "p")


class OD(P.D):
    """P.D (vec(), devs) over per-task tables built for one occupancy."""

    def __init__(self, t, devs):
        self.t, self.devs = t, devs


def build(task, occ, a):
    old, cmo, new = a.old_dir / f"{task}_replicas", a.cmo_dir / f"{task}_replicas", a.occ_dir / f"{task}_replicas"
    if not (new / "sweep_all.csv").is_file():
        sys.exit(f"No data in {new}. Run eval_results/run_paper_noise_occupancy.sh (on the cluster) first.")
    ns, nm = pd.read_csv(new / "sweep_all.csv"), pd.read_csv(new / "models.csv")
    close = lambda m: np.isclose(m.occupancy.astype(float), occ)
    frames, models = [], []
    wm = nm[(nm.source == "wann_occ") & close(nm)]
    frames.append(ns[ns.model_id.isin(wm.model_id)].assign(source="wann"))
    models.append(wm.assign(source="wann")[["model_id", "source", "replica_seed"]])
    have_cmo = bool((ns.noise_model == "reramcmo").any())
    has64 = bool((nm.source == "sgrl_64_occ").any())
    for s in ("sgrl_con", "sgrl_neu") + (("sgrl_64",) if has64 else ()):
        if np.isclose(occ, 1.0) and s not in FULL_FROM_OCC:
            om = pd.read_csv(old / "models.csv")
            om = om[om.source == s]
            parts = [pd.read_csv(old / "sweep_all.csv").query("source == @s and noise_model in ['none', 'pcm']")]
            if (cmo / "sweep_all.csv").is_file():
                cs = pd.read_csv(cmo / "sweep_all.csv")
                parts.append(cs[(cs.source == s) & (cs.noise_model == "reramcmo")])
            frames.append(pd.concat(parts).assign(source=s))
            models.append(om.assign(source=s)[["model_id", "source", "replica_seed"]])
        else:
            sm = nm[(nm.source == VARIANT[s]) & close(nm)] if "occupancy" in nm else nm.iloc[0:0]
            if sm.empty:
                print(f"[{task}] no {s} data at occupancy {occ * 100:g}%: that method is skipped there.")
                continue
            frames.append(ns[ns.model_id.isin(sm.model_id)].assign(source=s))
            models.append(sm.assign(source=s)[["model_id", "source", "replica_seed"]])
    df = pd.concat(frames, ignore_index=True)[KEEP]
    dst = a.cache_dir / f"{task}_occ{tag(occ)}"
    dst.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst / "sweep_all.csv", index=False)
    pd.concat(models).to_csv(dst / "models.csv", index=False)
    pr = old / "protocol.json"
    if pr.is_file():
        (dst / "protocol.json").write_text(pr.read_text())
    net, base, floor = network_table(dst, "best")
    devs = ["pcm"] + (["reramcmo"] if have_cmo else [])
    return dict(net=net, base=base, floor=floor, devs=devs)


def load(occ, a):
    t = {task: build(task, occ, a) for task, _ in P.TASKS}
    return OD(t, [d for d in ["pcm", "reramcmo"] if all(d in v["devs"] for v in t.values())])


def common_for(occ=None):
    """P.COMMON with the WANN/SGRL sentence rewritten for the occupancy mapping (occ=None: generic wording)."""
    c = P.COMMON_ORIG
    i, j = c.index("WANN: one ODIN"), c.index("Devices:")
    at = "at the occupancy given in the figure" if occ is None else f"at {occ * 100:g} % of g_max"
    extra = (", and 64×64 networks (hyperparameters from the same two-phase screening as the other SGRL variants)"
             if "sgrl_64" in P.SRC else "")
    mid = (f"WANN: the ODIN integer level of each network (chosen noise-free with an independent seed) with every weight "
           f"written {at} (w_max = weight / occupancy); SGRL: float32 (not quantized) networks with the same number of "
           f"connections or of neurons as the WANN network{extra}, with the largest |w| of each layer written {at}. ")
    return c[:i] + mid + c[j:]


# ------------------------------------------------------------------------------------------------ summary figure
COND = {"pcm": [("prog", 3.0, 86400.0, "Programming noise only (scale 3)"), ("read", 3.0, 86400.0, "Read noise only (scale 3)"),
                ("drift", 3.0, 86400.0, "Drift only (scale 3, 1 day)"), ("all", 1.0, YEAR, "All noise, 1 year (scale 1)")],
        "reramcmo": [("all", 0.0, 1.0, "No noise (g_min floor only)"), ("prog", 3.0, 1.0, "Programming noise only (scale 3)"),
                     ("read", 3.0, 1.0, "Read noise only (scale 3)"), ("all", 1.0, YEAR, "All noise, 1 year (scale 1)")]}


def summary(ds, occs, a):
    rng = np.random.default_rng(3)
    rows = []
    for dev in ds[occs[0]].devs:
        fig, axs = plt.subplots(3, 4, figsize=(14, 9), squeeze=False, sharex=True)
        for r, (task, lab) in enumerate(P.TASKS):
            for c, (mech, sc, t, title) in enumerate(COND[dev]):
                ax = axs[r][c]
                if r == 0:
                    ax.set_title(title, fontsize=12)
                for src in P.SRC:
                    xs, ms, lo, hi = [], [], [], []
                    for o in occs:
                        v = ds[o].vec(task, src, dev, mech, sc, t)
                        if len(v) == 0:
                            continue
                        ci = P.boot_ci(v.to_numpy(), rng)
                        xs.append(o); ms.append(v.mean()); lo.append(ci[0]); hi.append(ci[1])
                        rows.append(dict(task=task, device=dev, method=src, occupancy=o, mechanism=mech, scale=sc, t=t,
                                         n=len(v), mean=v.mean(), median=v.median(), ci_lo=ci[0], ci_hi=ci[1]))
                    if xs:
                        st = P.STYLE[src]
                        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.15, lw=0)
                        ax.plot(xs, ms, color=st["color"], marker=st["marker"], lw=2.2, ms=6)
                ax.axhline(100, color="0.5", lw=1, ls=":")
                ax.set_xscale("log", base=2)
                ax.xaxis.set_major_locator(FixedLocator(OCCS_ALL))
                ax.xaxis.set_major_formatter(FixedFormatter(["6.25", "12.5", "25", "50", "100"]))
                ax.xaxis.set_minor_locator(FixedLocator([]))
                ax.set_xlim(0.0625 / 1.15, 1.15)
                if r == 2:
                    ax.set_xlabel("Occupancy of $g_{max}$ (%)")
            axs[r][0].set_ylabel(f"{lab}\nRetained performance (%)")
        P.legend(fig, y=-0.01)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        name = f"noise_vs_occupancy_{dev}"
        cap = (f"**{name}** — Retained performance on {P.DEV_LABEL[dev]} as a function of the occupancy of g_max, for each "
               "noise mechanism applied alone (columns) and all of them together after 1 year, per task (rows). Lines: mean "
               "over networks; bands: 95 % bootstrap CI over networks. WANN: every weight written at that fraction of g_max "
               "(w_max = weight/occupancy); SGRL: the largest |w| of each layer at that fraction. Lower occupancy = lower "
               "conductance = lower static power (proxy). "
               + common_for() + P.pending_note(ds[occs[0]]))
        P.save(fig, name, cap)
    out = a.occ_dir / "summary_by_occupancy.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--occ-dir", type=Path, default=_SCRIPT_DIR / "paper_noise_occ")
    ap.add_argument("--old-dir", type=Path, default=_SCRIPT_DIR, help="folder with <task>_replicas (PCM, SGRL at 100%%)")
    ap.add_argument("--cmo-dir", type=Path, default=_SCRIPT_DIR / "paper_noise", help="folder with <task>_replicas (CMO, SGRL at 100%%)")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=_ROOT / "graficos" / "paper" / "noise_occupancy")
    ap.add_argument("--occupancies", type=float, nargs="+", default=OCCS_ALL)
    ap.add_argument("--no-per-occupancy", action="store_true", help="only the occupancy-vs-noise summary figures")
    a = ap.parse_args()
    a.cache_dir = a.cache_dir or a.occ_dir / "merged"
    setup_publication_style()
    P.COMMON_ORIG = P.COMMON
    nm0 = pd.read_csv(a.occ_dir / 'car_replicas' / 'models.csv')
    if (nm0.source == 'sgrl_64_occ').any():
        P.SRC = ['wann', 'sgrl_con', 'sgrl_neu', 'sgrl_64']
    occs = sorted(a.occupancies)
    ds = {o: load(o, a) for o in occs}
    P.OUT = a.out
    P.COMMON = common_for()
    summary(ds, occs, a)
    if not a.no_per_occupancy:
        for o in occs:
            P.OUT = a.out / f"occ{tag(o)}"
            P.COMMON = common_for(o)
            for f in (P.fig_intensity, P.fig_drift, P.fig_boxplots, P.fig_ablation):
                f(ds[o])


if __name__ == "__main__":
    main()
