#!/usr/bin/env python3
"""
Single-occupancy version of the paper noise figures (figura_04D / figura_05D): one conductance occupancy (default 25 % of
g_max), rows = devices (PCM, ReRAM-CMO), columns = tasks, plus one extra WANN line (orange, dashed): networks trained with
the narrower shared-weight table {0.5, 1, 1.5, 2, 2.5, 3}, in every task whose data exist (--extra-reps):
  Acrobot       acrobot_replicas_pre_w8  (run_key acrobot_small_first_spike, kept there when the _w8 nets replaced them)
  Racing Car    car_replicas_w3          (car_ttfs_first_spike_w3; eval_results/run_paper_noise_w3.sh, on the cluster)
  Mountain Car  disc_mc_replicas_w3      (disc_mc_small_rate_argmax_w3; same script)
A task whose folder is missing (in --occ-dir, or in --cmo-aihwkit-dir when given) is skipped with a message, so with only
the Acrobot data the figure is the original Acrobot-only figura_04D/05D. The main WANN line (vermillion) is, in every task,
the networks trained with {0.5, 1, 2, 3, 5, 8}.

Same data, aggregation and drawing code as plot_paper_noise_occ_macro.py (figura_04C/05C at 25 %); the extra line is a
separate network table (its own noise-free ODIN level per network), so it does not change the other lines' data.
Reads data only; nothing is simulated or trained.

  noise_intensity_occ25_<w3 tag>     retained performance vs. noise scale (all mechanisms), right after programming
  noise_drift_occ25_<w3 tag>         retained performance vs. time since programming, noise scale 1
  (<w3 tag> = acrobot_w3 when only Acrobot has the extra line, else w3)

Usage (journal set, ReRAM-CMO single mapping like figura_04C/05C):
  python eval_results/plot_paper_noise_occ_single.py --journal --zoom \
      --cmo-aihwkit-dir eval_results/paper_noise_occ_single_aihwkit --cmo-mapping single --journal-suffix D
"""
import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
os.chdir(_ROOT)
sys.path.insert(0, str(_SCRIPT_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import plot_paper_noise as P
import plot_paper_noise_occ_macro as M
import plot_paper_noise_occupancy as O
from plot_style import setup_journal_style, setup_publication_style

EXTRA = "wann_w3"                      # source name of the extra 0.5-3 line
EXTRA_REPS = {"acrobot": "acrobot_replicas_pre_w8", "car": "car_replicas_w3", "disc_mc": "disc_mc_replicas_w3"}
EXTRA_STYLE = dict(color="#E69F00", marker="v", ls="--")   # Okabe-Ito orange; dashed + own marker as redundant encoding


def load(occ, a):
    ns = Namespace(old_dir=a.old_dir, cmo_dir=a.cmo_dir, occ_dir=a.occ_dir, cache_dir=a.cache_dir,
                   cmo_aihwkit_dir=a.cmo_aihwkit_dir)
    d = O.load(occ, ns)
    for task, rep in a.extra.items():
        x = O.build(task, occ, ns, rep=rep)
        t = d.t[task]
        for k in ("net", "base"):
            w = x[k][x[k].source == "wann"].assign(source=EXTRA)
            t[k] = pd.concat([t[k], w], ignore_index=True)
    return d


def extra_available(a) -> dict:
    """task -> replicas folder of the extra 0.5-3 line, only for the tasks whose data exist."""
    want = dict(kv.split("=", 1) for kv in a.extra_reps)
    bad = set(want) - {t for t, _ in P.TASKS}
    if bad:
        sys.exit(f"--extra-reps: unknown task(s) {sorted(bad)} (use acrobot / car / disc_mc)")
    out = {}
    for task, _ in P.TASKS:               # figure column order
        rep = want.get(task)
        if not rep:
            continue
        dirs = [a.occ_dir / rep] + ([a.cmo_aihwkit_dir / rep] if a.cmo_aihwkit_dir is not None else [])
        missing = [str(d) for d in dirs if not (d / "sweep_all.csv").is_file()]
        if missing:
            print(f"[{task}] no 0.5-3 data ({', '.join(missing)}): no extra line in that column.")
            continue
        out[task] = rep
    return out


def extra_tasks_text(extra) -> str:
    return ", ".join(lab for t, lab in P.TASKS if t in extra)


def caption(kind, occ, d, stem, extra_tasks):
    o = M.occ_text(occ)
    if kind == "drift":
        head = (f"**{stem}** — Retained performance as a function of the time elapsed since programming, at noise scale 1, with "
                f"every weight written at {o} of g_max; rows are devices (PCM, ReRAM-CMO) and columns are tasks. The first "
                "point is the initial time of each device model (PCM: 0; ReRAM-CMO: 1 s), then 1 hour, 1 day and 1 year. ")
    else:
        head = (f"**{stem}** — Retained performance as a function of the noise scale (all mechanisms scaled together), with "
                f"every weight written at {o} of g_max; rows are devices (PCM, ReRAM-CMO) and columns are tasks. Evaluated "
                "right after programming (PCM: t = 0; ReRAM-CMO: t = 1 s). " + (P.CMO_SCALE0 if P.CMO_MAPPING else
                "At scale 0 PCM is 100 % by construction; for ReRAM-CMO scale 0 still includes the g_min floor "
                "of the differential pair (8.9 % of g_max), so it can differ from 100 %. "))
    extra = ""
    if list(extra_tasks) == ["acrobot"]:          # the original Acrobot-only figure, text unchanged
        extra = ("WANN (vermillion, solid): networks evolved with the shared-weight set {0.5, 1, 2, 3, 5, 8}, in every task. "
                 "Acrobot only, orange dashed line: the Acrobot WANN networks evolved with the narrower set "
                 "{0.5, 1, 1.5, 2, 2.5, 3} (same encoder/decoder, an earlier screening), evaluated with the identical "
                 "protocol (own noise-free ODIN level per network, same occupancy and noise). ")
    elif extra_tasks:
        where = ("in every task" if len(extra_tasks) == len(P.TASKS)
                 else f"{extra_tasks_text(extra_tasks)} only")
        extra = ("WANN (vermillion, solid): networks evolved with the shared-weight set {0.5, 1, 2, 3, 5, 8}, in every task. "
                 f"Orange dashed line ({where}): WANN networks evolved with the narrower set {{0.5, 1, 1.5, 2, 2.5, 3}}, "
                 "with the same encoder/decoder as the solid line but their own hyperparameter screening, evaluated with "
                 "the identical protocol (own noise-free ODIN level per network, same occupancy and noise). ")
    tail = ("Thin lines: individual trained networks; thick line and band: mean over networks and 95 % bootstrap CI. Dotted line: "
            "100 % (noise-free). ") + ("Time points are categorical (not to scale). " if kind == "drift" else "")
    if M.ZOOM:
        tail += ("The y axis of each panel is fitted to its mean curves and 95 % CI bands, so it differs between panels; "
                 "panels whose axis does not start at 0 are marked “y zoomed”, and individual networks outside the axis "
                 "limits are not drawn. ")
    return (head + extra + tail + O.common_for(occ) + P.pending_note(d)).replace("SGRL", P.SG_NAME).strip() + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--occupancy", type=float, default=0.25, help="occupancy (fraction of g_max)")
    ap.add_argument("--kinds", nargs="+", default=["drift", "intensity"], choices=["drift", "intensity"])
    ap.add_argument("--sgrl-name", default="SG")
    ap.add_argument("--occ-dir", type=Path, default=_SCRIPT_DIR / "paper_noise_occ")
    ap.add_argument("--old-dir", type=Path, default=_SCRIPT_DIR)
    ap.add_argument("--cmo-dir", type=Path, default=_SCRIPT_DIR / "paper_noise")
    ap.add_argument("--extra-reps", nargs="*", default=[f"{t}={r}" for t, r in EXTRA_REPS.items()], metavar="TASK=FOLDER",
                    help="replicas folder (in --occ-dir and --cmo-aihwkit-dir) of the extra 0.5-3 WANN line, per task; tasks "
                         "without data are skipped. Pass the flag with no value for no extra line")
    ap.add_argument("--zoom", action="store_true", help="fit each panel's y axis to its mean curves and CI bands")
    O.add_cmo_args(ap)
    ap.add_argument("--out", type=Path, default=M.OUT)
    ap.add_argument("--journal", action="store_true", help="190 mm wide, vector PDF + 600 dpi PNG to <journal-dir>/figura_0N<suffix>")
    ap.add_argument("--journal-dir", type=Path, default=_ROOT / "graficos" / "paper" / "send_to_journal")
    ap.add_argument("--journal-suffix", default="D")
    a = ap.parse_args()
    M.JOURNAL, M.ZOOM = a.journal, a.zoom
    M.OCC_LABEL = False            # the occupancy goes in the caption / manuscript
    O.check_cmo_args(ap, a)
    a.cache_dir = (a.cmo_aihwkit_dir or a.occ_dir) / "merged"
    (setup_journal_style if a.journal else setup_publication_style)()
    P.SG_NAME = a.sgrl_name
    P.COMMON_ORIG = P.COMMON
    nm0 = pd.read_csv(a.occ_dir / "car_replicas" / "models.csv")
    if (nm0.source == "sgrl_64_occ").any():
        P.SRC = ["wann", "sgrl_con", "sgrl_neu", "sgrl_64"]
    a.extra = extra_available(a)
    if a.extra:
        P.SRC = ["wann", EXTRA] + P.SRC[1:]
        P.STYLE[EXTRA] = EXTRA_STYLE
        P.LABEL["wann"] = "WANN, weights 0.5–8"
        P.LABEL[EXTRA] = ("WANN, weights 0.5–3" if len(a.extra) == len(P.TASKS)
                          else f"WANN, weights 0.5–3 ({extra_tasks_text(a.extra)})")
    P.LABEL = {k: v.replace("SGRL", a.sgrl_name) for k, v in P.LABEL.items()}
    occ = a.occupancy
    d = load(occ, a)
    a.out.mkdir(parents=True, exist_ok=True)
    for kind in a.kinds:
        fig = M.make(kind, [occ], {occ: d})
        stem = f"noise_{kind}_{M.name_of(occ)}" + ({(): "", ("acrobot",): "_acrobot_w3"}.get(tuple(a.extra), "_w3"))
        if a.journal:
            a.journal_dir.mkdir(parents=True, exist_ok=True)
            for ext in (".pdf", ".png"):
                fig.savefig(a.journal_dir / (M.JOURNAL_NAME[kind] + a.journal_suffix + ext))
        else:
            fig.savefig(a.out / f"{stem}.png")
        plt.close(fig)
        (a.out / f"{stem}_caption.md").write_text(caption(kind, occ, d, stem, list(a.extra)), encoding="utf-8")
        print("wrote", (a.journal_dir / (M.JOURNAL_NAME[kind] + a.journal_suffix)) if a.journal else a.out / f"{stem}.png")


if __name__ == "__main__":
    main()
