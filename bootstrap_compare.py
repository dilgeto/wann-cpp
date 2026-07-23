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

  # Comparacion con incertidumbre del benchmark (--ann-std y --ann-n de SB3,
  # p.ej. evaluate_policy con n_eval_episodes=10 -> mean_reward, std_reward):
  # en vez de tratar el ANN como una constante, se simula su distribucion
  # muestral de la media ~ Normal(mean_reward, std_reward/sqrt(n_eval_episodes))
  # y se bootstrapea la diferencia (tu_agente - ANN).
  python bootstrap_compare.py \
      --csv screening_full/mountain_car_ttfs_rate_argmax/p3_validation.csv \
      --rank 0 --ann -110 --ann-std 25 --ann-n 10
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


def diff_bootstrap(boot_means, bench_mean, bench_std, bench_n, rng):
    """Simula la distribucion muestral de la media del benchmark como
    Normal(bench_mean, SE) con SE = bench_std / sqrt(bench_n) (bench_std es
    la dispersion episodio-a-episodio, no el SE de la media), y devuelve la
    distribucion bootstrap de la diferencia (tu_agente - benchmark)."""
    se_bench = bench_std / np.sqrt(bench_n)
    bench_boot_means = rng.normal(bench_mean, se_bench, size=len(boot_means))
    return boot_means - bench_boot_means, se_bench


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="p3_validation.csv de screening_full.py")
    ap.add_argument("--rank", type=int, default=0, help="rank de la config (0 = mejor)")
    ap.add_argument("--col", default="original_fitness",
                     help="columna a usar como recompensa (default: original_fitness)")
    ap.add_argument("--ann", type=float, default=None, help="score publicado del benchmark ANN (mean_reward)")
    ap.add_argument("--ann-std", type=float, default=None,
                     help="std_reward del ANN (opcional). Si se entrega junto con --ann-n, "
                          "propaga la incertidumbre del benchmark en vez de tratarlo como constante")
    ap.add_argument("--ann-n", type=int, default=None,
                     help="n_eval_episodes usado para --ann-std (requerido junto con --ann-std)")
    ap.add_argument("--gym", type=float, default=None, help="score publicado del benchmark Gymnasium (mean_reward)")
    ap.add_argument("--gym-std", type=float, default=None,
                     help="std_reward del benchmark Gymnasium (opcional, ver --ann-std)")
    ap.add_argument("--gym-n", type=int, default=None,
                     help="n_eval_episodes usado para --gym-std (requerido junto con --gym-std)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--pct", type=float, default=None,
                     help="si se indica (ej. 0.9), en vez de comparar contra el "
                          "score exacto evalua el objetivo 'alcanzar al menos pct%% del benchmark' "
                          "(consciente del signo, ver docstring)")
    ap.add_argument("--seed", type=int, default=0, help="semilla del RNG del bootstrap (reproducibilidad)")
    args = ap.parse_args()

    if (args.ann_std is None) != (args.ann_n is None):
        raise SystemExit("--ann-std y --ann-n deben entregarse juntos.")
    if (args.gym_std is None) != (args.gym_n is None):
        raise SystemExit("--gym-std y --gym-n deben entregarse juntos.")

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

    for name, bench, bench_std, bench_n in (
        ("ANN", args.ann, args.ann_std, args.ann_n),
        ("Gymnasium", args.gym, args.gym_std, args.gym_n),
    ):
        if bench is None:
            continue

        if args.pct is not None:
            if bench_std is not None:
                print(f"  [nota] --{name.lower()}-std se ignora en modo --pct "
                      f"(el objetivo usa el punto estimado del benchmark).")
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

        if bench_std is not None:
            # Propaga la incertidumbre del benchmark: simula su distribucion
            # muestral de la media y bootstrapea la diferencia (tu_agente - benchmark).
            diffs, se_bench = diff_bootstrap(boot_means, bench, bench_std, bench_n, rng)
            diff_mean = mean - bench
            d_lo, d_hi = np.percentile(diffs, [(1 - args.ci) / 2 * 100, (1 + args.ci) / 2 * 100])
            p = 2 * min(np.mean(diffs < 0), np.mean(diffs > 0))
            inside = d_lo <= 0 <= d_hi
            if inside:
                verdict = "sin diferencia significativa"
            elif diff_mean > 0:
                verdict = "tu agente SUPERA"
            else:
                verdict = "tu agente por DEBAJO de"
            print(f"  vs {name} ({bench:.2f} ± SE={se_bench:.2f}, "
                  f"std={bench_std:.2f}, n={bench_n}): {verdict} "
                  f"(diferencia={diff_mean:.2f}, IC{int(args.ci*100)}%=[{d_lo:.2f}, {d_hi:.2f}], "
                  f"p={p:.4f}, 0 {'dentro' if inside else 'fuera'} del IC de la diferencia)")
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
