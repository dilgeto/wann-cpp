#!/usr/bin/env python3
"""
Software reference for the WANN networks that were adapted to ODIN
(ODIN-SG-WANN-datos/wann/): evaluates the ORIGINAL WANN network (wann-cpp binaries
wann_eval_weights_<task>, evaluation only, no training) on the same episode indices
used on ODIN, with each network's winner shared weight (.wi file), and writes the
per-episode rewards to eval_results/wann_odin_reference_episodes.csv.

Networks (11 per task):
  car          car_ttfs_first_spike_discrete        rank00_seed00..10   episodes 100-199  (shaped reward)
  mountain_car disc_mc_small_first_spike_discrete   rank01_seed00..10   episodes 100-199  (original reward)
  acrobot      acrobot_small_first_spike            rank01_seed00..10   episodes 0-99     (original reward)

Episode index e -> `--seeds e --nreps 1` (episodeSeed = e*10000 + wi*100). The ODIN-side
adapted network (config C: recalibrated parameters) is not identical to the original, so
episodes are NOT compared one by one; the plot compares per-network means.

Usage: python eval_results/eval_wann_odin_reference.py [--omp 8]
"""
import argparse
import os
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

os.chdir(Path(__file__).resolve().parent.parent)

TASKS = {
    "car": dict(exe="./build/wann_eval_weights_car", base="p/car_snn.json",
                run_key="car_ttfs_first_spike_discrete", rank=0, reward="shaped", e0=100),
    "mountain_car": dict(exe="./build/wann_eval_weights_disc_mc", base="p/disc_mc_snn.json",
                         run_key="disc_mc_small_first_spike_discrete", rank=1, reward="original", e0=100),
    "acrobot": dict(exe="./build/wann_eval_weights_acrobot", base="p/acrobot_snn.json",
                    run_key="acrobot_small_first_spike", rank=1, reward="original", e0=0),
}
N_EPISODES, N_NETS = 100, 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--omp", type=int, default=8)
    ap.add_argument("--out", default="eval_results/wann_odin_reference_episodes.csv")
    a = ap.parse_args()
    rows = []
    for task, t in TASKS.items():
        seeds = ",".join(str(t["e0"] + i) for i in range(N_EPISODES))
        for s in range(N_NETS):
            stem = f"rank{t['rank']:02d}_seed{s:02d}"
            net = Path("log") / f"full_p3_{t['run_key']}" / f"{stem}_best.out"
            wi = int((Path("log") / f"full_p3_{t['run_key']}" / f"{stem}_best.wi").read_text().split()[0])
            cfg = Path("screening_full") / t["run_key"] / "p3_configs" / f"{stem}.json"
            cmd = [t["exe"], "-f", str(net), "-d", t["base"], "-p", str(cfg), "--seeds", seeds,
                   "--reward", t["reward"], "--nreps", "1", "--episode-detail", "--weight-index", str(wi)]
            env = dict(os.environ, OMP_NUM_THREADS=str(a.omp))
            p = subprocess.run(cmd, capture_output=True, text=True, env=env)
            if p.returncode != 0:
                raise SystemExit(f"FAIL {task} {stem}\n{p.stderr}")
            d = pd.read_csv(StringIO(p.stdout))
            for _, r in d.iterrows():
                rows.append(dict(task=task, run_key=t["run_key"], net=stem, weight_index=wi,
                                 episode=int(r["seed"]), reward_software=r["reward"]))
            print(f"{task:12s} {stem} wi={wi} mean={d.reward.mean():8.2f}", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print("saved:", a.out)


if __name__ == "__main__":
    main()
