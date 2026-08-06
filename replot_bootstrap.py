#!/usr/bin/env python3
"""
replot_bootstrap.py — Regenera solo el PNG de un bootstrap_compare_*.py ya
corrido, leyendo rewards.csv y bootstrap_samples.csv desde --out-dir. No
re-evalúa episodios ni re-muestrea: sirve para iterar rápido sobre cambios
de estilo/escala en make_plots().

Uso:
  python replot_bootstrap.py bootstrap_car \
      --suptitle "Racing Car — SNN vs ANN (PPO nativo), bootstrap no pareado" \
      --plot-stem car_snn_vs_ann --ann-label "ANN (PPO)"
"""
import argparse
from pathlib import Path

import pandas as pd

from bootstrap_compare_lib import make_plots


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", help="Directorio con rewards.csv y bootstrap_samples.csv")
    ap.add_argument("--suptitle", required=True)
    ap.add_argument("--plot-stem", required=True, dest="plot_stem")
    ap.add_argument("--ann-label", default="ANN", dest="ann_label")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    rewards_df = pd.read_csv(out_dir / "rewards.csv")
    boot_df = pd.read_csv(out_dir / "bootstrap_samples.csv")
    ci_df = pd.read_csv(out_dir / "ci_results.csv").iloc[0]

    rewards_snn = rewards_df.loc[rewards_df["agent"] == "snn", "reward"].to_numpy()
    rewards_ann = rewards_df.loc[rewards_df["agent"] == "ann", "reward"].to_numpy()
    boot_ratio_signed = boot_df["boot_ratio_pct_signed"].to_numpy()

    plot_path = out_dir / f"{args.plot_stem}.png"
    make_plots(rewards_snn, rewards_ann, boot_ratio_signed,
              ci_df["ratio_signed_ci_lo_pct"], ci_df["ratio_signed_ci_hi_pct"], ci_df["ci_level"],
              plot_path, args.suptitle, args.ann_label)
    print(f"Regenerado: {plot_path}")


if __name__ == "__main__":
    main()
