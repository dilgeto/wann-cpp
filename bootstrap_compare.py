#!/usr/bin/env python3
"""
bootstrap_compare.py - Bootstrap de percentiles para comparar las recompensas
de tu agente (salida de `screening_full.py --mode phase3`, columna
original_fitness de p3_validation.csv) contra scores fijos publicados por
otros trabajos (ANN, Gymnasium).

No se necesita reentrenar: usa las N seeds que ya corriste en phase3.

Objetivo tipo "alcanzar al menos el X% de un benchmark" (--pct):
  El umbral NO se calcula multiplicando el score tal cual, porque eso
  invierte el sentido en tareas de recompensa negativa (Mountain Car,
  Acrobot: mas cercano a 0 = mejor). En su lugar:
      target = benchmark - (1 - pct) * abs(benchmark)
  Con benchmark=350 (positivo) y pct=0.9  -> target=315 (necesitas >=315).
  Con benchmark=-110 (negativo) y pct=0.9 -> target=-121 (necesitas >=-121,
  es decir, se permite hasta 10% mas de magnitud de penalizacion).

Uso:
  # Comparacion directa contra el score exacto reportado
  python bootstrap_compare.py \
      --csv screening_full/mountain_car_ttfs_rate_argmax/p3_validation.csv \
      --rank 0 --ann -110 --gym -100

  # Objetivo: alcanzar al menos el 90% del score de stable-baselines3
  python bootstrap_compare.py \
      --csv screening_full/mountain_car_ttfs_rate_argmax/p3_validation.csv \
      --rank 0 --ann -110 --pct 0.9
"""

import argparse

import numpy as np
import pandas as pd


def bootstrap_ci(samples, n_boot=10000, ci=0.95, rng=None):
    rng = rng or np.random.default_rng()
    samples = np.asarray(samples, dtype=float)
    boot_means = rng.choice(samples, size=(n_boot, len(samples)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return samples.mean(), lo, hi, boot_means


def p_value_vs_benchmark(boot_means, benchmark):
    p_below = np.mean(boot_means < benchmark)
    p_above = np.mean(boot_means > benchmark)
    return 2 * min(p_below, p_above)


def scaled_target(benchmark, pct):
    """Umbral 'al menos pct% del benchmark', consciente del signo de la
    recompensa (mayor = mejor siempre, sin importar si es negativa)."""
    return benchmark - (1 - pct) * abs(benchmark)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="p3_validation.csv de screening_full.py")
    ap.add_argument("--rank", type=int, default=0, help="rank de la config (0 = mejor)")
    ap.add_argument("--col", default="original_fitness",
                     help="columna a usar como recompensa (default: original_fitness)")
    ap.add_argument("--ann", type=float, default=None, help="score publicado del benchmark ANN")
    ap.add_argument("--gym", type=float, default=None, help="score publicado del benchmark Gymnasium")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--pct", type=float, default=None,
                     help="si se indica (ej. 0.9), en vez de comparar contra el "
                          "score exacto evalua el objetivo 'alcanzar al menos pct%% del benchmark' "
                          "(consciente del signo, ver docstring)")
    ap.add_argument("--seed", type=int, default=0, help="semilla del RNG del bootstrap (reproducibilidad)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    sub = df[df["rank"] == args.rank].dropna(subset=[args.col])
    samples = sub[args.col].values
    n = len(samples)
    if n < 2:
        raise SystemExit(f"Solo {n} muestras para rank {args.rank}; se necesitan al menos 2.")

    rng = np.random.default_rng(args.seed)
    mean, lo, hi, boot_means = bootstrap_ci(samples, n_boot=args.n_boot, ci=args.ci, rng=rng)

    print(f"n = {n} seeds | media = {mean:.2f} | IC{int(args.ci * 100)}% = [{lo:.2f}, {hi:.2f}]")

    # cota inferior de una sola cola (percentil alpha), para el chequeo de objetivo
    lo_one_sided = np.percentile(boot_means, (1 - args.ci) * 100)

    for name, bench in (("ANN", args.ann), ("Gymnasium", args.gym)):
        if bench is None:
            continue

        if args.pct is not None:
            target = scaled_target(bench, args.pct)
            reaches = lo_one_sided >= target
            estado = "SI" if reaches else "NO"
            # alpha resultante: nivel de significancia de una cola al que el
            # objetivo queda justo en el borde del intervalo (menor = mas confianza)
            alpha_result = np.mean(boot_means < target)
            print(f"  objetivo >= {args.pct*100:.0f}% de {name} ({bench:.2f}) "
                  f"-> target={target:.2f}: {estado} se alcanza "
                  f"(cota inferior {int(args.ci*100)}% de una cola = {lo_one_sided:.2f}, "
                  f"alpha resultante={alpha_result:.4f}, se exige alpha<={1 - args.ci:.2f})")
            continue

        p = p_value_vs_benchmark(boot_means, bench)
        inside = lo <= bench <= hi
        if inside:
            verdict = "sin diferencia significativa"
        elif mean > bench:
            verdict = "tu agente SUPERA"
        else:
            verdict = "tu agente por DEBAJO de"
        print(f"  vs {name} ({bench:.2f}): {verdict} "
              f"(p={p:.4f}, benchmark {'dentro' if inside else 'fuera'} del IC)")


if __name__ == "__main__":
    main()
