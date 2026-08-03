#!/usr/bin/env python3
"""
bootstrap_two_sample.py - Bootstrap de dos muestras (independientes) para
comparar recompensas por episodio de tu agente SNN contra un benchmark ANN,
cuando tienes los episodios completos de ambos lados (CSVs de una columna
"reward", ej. car_snn.csv vs car_ann.csv).

A diferencia de bootstrap_compare.py (que trata al benchmark como un numero
fijo), aqui se remuestrea CADA lado por separado, asi la incertidumbre de
ambas muestras entra al resultado.

Muestras independientes (no pareadas): se remuestrea cada CSV por su cuenta,
con su propio tamano, sin importar si tienen distinto numero de episodios.

Modo 1 - diferencia de medias (SNN - ANN):
  python bootstrap_two_sample.py --a car_snn.csv --b car_ann.csv

Modo 2 - objetivo "SNN >= pct%% de ANN" (ej. 90%%), propagando incertidumbre
de ambos lados (target se recalcula en cada remuestreo a partir de ANN,
no de un numero fijo):
  python bootstrap_two_sample.py --a car_snn.csv --b car_ann.csv --pct 0.9
"""

import argparse

import numpy as np
import pandas as pd


def load_rewards(path, col):
    df = pd.read_csv(path)
    vals = df[col].dropna().values.astype(float)
    if len(vals) < 2:
        raise SystemExit(f"{path}: solo {len(vals)} episodios validos; se necesitan al menos 2.")
    return vals


def bootstrap_means(samples, n_boot, rng):
    samples = np.asarray(samples, dtype=float)
    return rng.choice(samples, size=(n_boot, len(samples)), replace=True).mean(axis=1)


def scaled_target(benchmark, pct):
    """Umbral 'al menos pct% del benchmark', consciente del signo (mayor
    siempre es mejor, sin importar si la recompensa es negativa)."""
    return benchmark - (1 - pct) * np.abs(benchmark)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="CSV de tu agente (ej. car_snn.csv)")
    ap.add_argument("--b", required=True, help="CSV del benchmark (ej. car_ann.csv)")
    ap.add_argument("--col", default="reward", help="columna a usar (default: reward)")
    ap.add_argument("--a-name", default=None)
    ap.add_argument("--b-name", default=None)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--pct", type=float, default=None,
                     help="si se indica (ej. 0.9), evalua el objetivo "
                          "'A alcanza al menos pct%% de B' en vez de la diferencia de medias")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    name_a = args.a_name or args.a
    name_b = args.b_name or args.b

    a = load_rewards(args.a, args.col)
    b = load_rewards(args.b, args.col)

    rng = np.random.default_rng(args.seed)
    boot_a = bootstrap_means(a, args.n_boot, rng)
    boot_b = bootstrap_means(b, args.n_boot, rng)

    print(f"{name_a}: n={len(a)} media={a.mean():.2f}")
    print(f"{name_b}: n={len(b)} media={b.mean():.2f}")

    if args.pct is not None:
        target = scaled_target(boot_b, args.pct)
        diff = boot_a - target
        lo_one_sided = np.percentile(diff, (1 - args.ci) * 100)
        alpha_result = np.mean(diff < 0)
        reaches = lo_one_sided >= 0
        estado = "SI" if reaches else "NO"
        print(f"objetivo: {name_a} >= {args.pct*100:.0f}% de {name_b}")
        print(f"  {estado} se alcanza (cota inferior {int(args.ci*100)}% de una cola "
              f"de la diferencia = {lo_one_sided:.2f}, "
              f"alpha resultante={alpha_result:.4f}, se exige alpha<={1 - args.ci:.2f})")
        return

    diff = boot_a - boot_b
    lo, hi = np.percentile(diff, [(1 - args.ci) / 2 * 100, (1 + args.ci) / 2 * 100])
    p = 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))
    inside = lo <= 0 <= hi
    verdict = "sin diferencia significativa" if inside else \
        (f"{name_a} SUPERA a {name_b}" if diff.mean() > 0 else f"{name_a} por DEBAJO de {name_b}")
    print(f"diferencia ({name_a} - {name_b}) = {diff.mean():.2f} | "
          f"IC{int(args.ci*100)}% = [{lo:.2f}, {hi:.2f}]")
    print(f"  {verdict} (p={p:.4f}, 0 {'dentro' if inside else 'fuera'} del IC de la diferencia)")


if __name__ == "__main__":
    main()
