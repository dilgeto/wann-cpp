#!/usr/bin/env python3
"""
Control de sensibilidad al mapeo peso -> conductancia (w_max) para la red WANN ganadora de Car.

Pregunta: ¿el colapso de WANN-Car con la deriva de ReRAM-CMO es propio de la red o consecuencia de cómo se
mapea su peso? En wann_car_noise, `w_max` es el peso que se programa en g_max; por defecto es el mayor de
WEIGHT_VALS (8), y el peso ganador es 1, así que ocupa solo el 12,5 % del rango de conductancia. Con la deriva
de CMO (aditiva, en µS, independiente del nivel) un peso en baja conductancia queda muy distorsionado en
términos relativos; si ocupara el fondo de escala (w_max = 1) el mismo corrimiento sería ~25 veces menor en
relación al peso.

Se evalúa la MISMA red y el MISMO peso (1.0) con w_max en {8, 4, 2, 1} (el peso ocupa 12,5 / 25 / 50 / 100 %
de g_max), con el protocolo de la comparación original (30 chips x 3 lecturas x 2 semillas, 6 episodios por
fila). w_max = 8 reproduce las celdas de eval_results/car_wann_vs_sgrl (control de regresión). Solo evalúa.

Uso:
  python eval_results/wmax_check_car.py --jobs 12
  python eval_results/wmax_check_car.py --summary-only
"""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd

NET = "log/full_p3_car_ttfs_first_spike_discrete/rank00_seed06_best.out"
OVERRIDE = "screening_full/car_ttfs_first_spike_discrete/p3_configs/rank00_seed06.json"
CONFIG = "p/car_snn.json"
BINARY = "./build/wann_car_noise"
WEIGHT = 1.0
YEAR = 31536000.0
# (dispositivo, escala, t): lo mínimo que responde la pregunta; PCM como contraste (deriva multiplicativa)
CELLS = [("reramcmo", 0.0, 1.0), ("reramcmo", 1.0, 1.0), ("reramcmo", 1.0, 86400.0), ("reramcmo", 1.0, YEAR),
         ("reramcmo", 3.0, 1.0), ("pcm", 1.0, 0.0), ("pcm", 1.0, YEAR)]
REF_DIR = Path("eval_results/car_wann_vs_sgrl")


def fmt(x):
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def commands(wmaxs, out: Path, seeds):
    jobs = [("none", 0.0, 0.0, 8.0)]       # sin ruido: no depende de w_max
    jobs += [(m, s, t, w) for w in wmaxs for (m, s, t) in CELLS]
    for m, s, t, w in jobs:
        for seed in seeds:
            stem = f"{m}_s{fmt(s)}_t{fmt(t)}_w{fmt(w)}_seed{seed}"
            cmd = [BINARY, "-f", NET, "-d", CONFIG, "-p", OVERRIDE, "--weight", fmt(WEIGHT), "--w-max", fmt(w),
                   "--mode", "per-synapse", "--noise-model", m, "--n-chips", "30", "--reads-per-chip", "3",
                   "--t-inference", fmt(t), "--n-levels", "0", "--reramcmo-g-max", "25", "--reramcmo-g-min", "1",
                   "--seeds", str(seed)]
            if m != "none":
                cmd += ["--noise-scale", fmt(s)]
            yield stem, cmd, (m, s, t, w, seed)


def run_one(stem, cmd, out: Path):
    part = out / f"{stem}.csv"
    if part.is_file():
        return stem, True
    env = dict(os.environ, OMP_NUM_THREADS="1")
    with open(part.with_suffix(".tmp"), "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env, text=True)
    if rc.returncode != 0:
        print(f"[FALLÓ] {stem}: {rc.stderr[-300:]}", flush=True)
        return stem, False
    os.replace(part.with_suffix(".tmp"), part)
    return stem, True


def boot_ci(x, rng, n=4000):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def summarize(out: Path, wmaxs, seeds):
    rng = np.random.default_rng(0)
    rows = []
    for stem, _, (m, s, t, w, seed) in commands(wmaxs, out, seeds):
        p = out / f"{stem}.csv"
        if not p.is_file():
            continue
        rows.append(pd.read_csv(p).assign(model=m, scale=s, t=t, wmax=w))
    df = pd.concat(rows, ignore_index=True)
    chip = df.groupby(["model", "scale", "t", "wmax", "seed", "chip"])["reward_original"].mean().reset_index()
    base = chip[chip.model == "none"].reward_original.mean()
    print(f"retorno sin ruido de la red (peso {WEIGHT:g}): {base:.1f}   (retención = retorno / ese valor)\n")
    out_rows = []
    for (m, s, t, w), g in chip[chip.model != "none"].groupby(["model", "scale", "t", "wmax"]):
        x = g.reward_original.to_numpy()
        lo, hi = boot_ci(x, rng)
        out_rows.append(dict(model=m, scale=s, t=t, wmax=w, n=len(x), ret=x.mean() / base * 100,
                             lo=lo / base * 100, hi=hi / base * 100, ret_abs=x.mean()))
    res = pd.DataFrame(out_rows)
    res.to_csv(out / "summary.csv", index=False)
    # referencias ya calculadas (comparación original): misma red/peso con w_max=8, y los SGRL principales
    ref = None
    if (REF_DIR / "aggregated.csv").is_file():
        ref = pd.read_csv(REF_DIR / "aggregated.csv")
        mm = pd.read_csv(REF_DIR / "models.csv")
        mains = {r.source: r.model_id for r in mm[mm.main == 1].itertuples()}
    tl = {1.0: "1 s", 0.0: "0", 86400.0: "1 día", YEAR: "1 año"}
    print("Retención % [IC95] de la MISMA red WANN (peso 1) según cuánta conductancia ocupa su peso (w_max):")
    hdr = f"{'dispositivo / escala / t':26s}" + "".join(f"w_max={fmt(w):>2s} ({100 / w:>5.1f}% de g_max)   " for w in wmaxs)
    print(hdr + (" | SGRL 64×64   conex.   neur." if ref is not None else ""))
    for (m, s, t) in CELLS:
        line = f"{m:8s} s={s:g} t={tl.get(t, t):6s}   "
        for w in wmaxs:
            r = res[(res.model == m) & (res.scale == s) & np.isclose(res.t, t) & (res.wmax == w)]
            line += (f"{r.ret.iloc[0]:6.1f} [{r.lo.iloc[0]:5.1f},{r.hi.iloc[0]:5.1f}]       " if len(r) else " " * 28)
        if ref is not None:
            vals = []
            for src in ("sgrl", "sgrl_con", "sgrl_neu"):
                r = ref[(ref.model_id == mains[src]) & (ref.noise_model == m) & (ref.mechanism == "all")
                        & np.isclose(ref.scale, s) & np.isclose(ref.t_inference, t)]
                vals.append(f"{r.ret_mean.iloc[0]:6.1f}" if len(r) else "   n/a")
            line += " | " + "  ".join(vals)
        print(line)
    if ref is not None:   # control de regresión: w_max = 8 debe reproducir wann_w1 de la comparación original
        diffs = []
        for (m, s, t) in CELLS:
            r8 = res[(res.model == m) & (res.scale == s) & np.isclose(res.t, t) & (res.wmax == 8.0)]
            r0 = ref[(ref.model_id == "wann_w1") & (ref.noise_model == m) & (ref.mechanism == "all")
                     & np.isclose(ref.scale, s) & np.isclose(ref.t_inference, t)]
            if len(r8) and len(r0):
                diffs.append(abs(r8.ret_abs.iloc[0] - r0["mean"].iloc[0]))
        print(f"\nControl de regresión (w_max = 8 vs. comparación original, wann_w1): |dif| máxima de la media = "
              f"{max(diffs):.3g} sobre {len(diffs)} celdas")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wmax", type=float, nargs="+", default=[8.0, 4.0, 2.0, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--jobs", type=int, default=12)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/car_wmax_check"))
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        todo = [(stem, cmd) for stem, cmd, _ in commands(a.wmax, a.out_dir, a.seeds)]
        print(f"{len(todo)} trabajos (hasta {a.jobs} a la vez); los ya hechos se saltan.", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
            res = list(ex.map(lambda sc: run_one(sc[0], sc[1], a.out_dir), todo))
        fails = [s for s, ok in res if not ok]
        if fails:
            sys.exit(f"{len(fails)} trabajos fallaron")
    summarize(a.out_dir, a.wmax, a.seeds)


if __name__ == "__main__":
    main()
