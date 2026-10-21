#!/usr/bin/env python3
"""
Resume el CSV que imprime `wann_car_noise` (ver BINARIOS.md §4d): una fila por
(seed, chip, read, t_inference) con el reward promediado sobre alg_nReps
episodios. Este script no corre ningún binario ni simula nada — solo agrega
filas ya calculadas.

Agrupa por (mode, noise_model, weight, t_inference, prog_noise_scale,
read_noise_scale, drift_scale) y reporta,
para reward_shaped y reward_original, DOS niveles de varianza por separado
(no uno solo mezclado):

  - entre-chips (`*_between_chip_std`): cuánto cambia el reward promedio de
    un chip al promedio de otro — "mala suerte de programación/D2D", lo que
    le importaría a un fabricante preguntando por rendimiento de yield.
  - dentro-de-chip (`*_within_chip_std`): cuánto varía el reward entre
    lecturas repetidas del MISMO chip (mismo sorteo de programación/D2D/
    deriva) — ruido de lectura puro + variabilidad de la política. Queda en
    NaN si el CSV se corrió con `--reads-per-chip 1` (no hay de dónde
    sacarlo).

Además: intervalos de confianza por bootstrap de la media (remuestreando a
nivel de CHIP, no de fila — una lectura repetida del mismo chip no es una
observación independiente, remuestrear filas sueltas subestimaría la
incertidumbre real) y percentiles 10/50/90 de los promedios por chip, para
graficar una banda de percentiles en vez de asumir normalidad.

Si el mismo archivo (o alguno de los pasados) tiene filas con
noise_model == "none" para el mismo (mode, weight), las usa como baseline y
agrega delta_mean / delta_pct respecto de ese baseline sin ruido.

Uso:
  python eval_results/summarize_car_noise.py --csv log/car_noise_pcm.csv
  python eval_results/summarize_car_noise.py --csv log/car_noise_none.csv log/car_noise_pcm.csv \
      --out eval_results/car_noise_summary.csv
  python eval_results/summarize_car_noise.py --csv log/car_noise_pcm.csv \
      --success-threshold 500   # agrega frac_chips_above (sobre reward_shaped)
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ancla CWD a la raíz del repo — mismo patrón que eval_p3_weights.py y los
# scripts de bootstrap_results/, para que rutas relativas (--out por defecto,
# etc.) resuelvan igual sin importar desde dónde se invoque este script.
os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bootstrap_results"))
from bootstrap_compare_lib import percentile_ci  # noqa: E402

GROUP_COLS = ["mode", "noise_model", "weight", "t_inference",
              "prog_noise_scale", "read_noise_scale", "drift_scale"]
REWARD_COLS = ["reward_shaped", "reward_original"]
CHIP_COLS = ["seed", "chip"]  # identidad de una instancia de chip: (seed,chip)


def load_rows(csv_paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        if not Path(p).is_file():
            sys.exit(f"Error: no existe {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    missing = set(GROUP_COLS + REWARD_COLS + CHIP_COLS) - set(df.columns)
    if missing:
        sys.exit(
            f"Error: el CSV no tiene las columnas esperadas de wann_car_noise "
            f"(faltan: {sorted(missing)}). ¿Es un CSV viejo? Volvé a generarlo "
            f"con el binario actual."
        )
    return df


def bootstrap_mean_ci(chip_means: np.ndarray, n_resamples: int,
                       rng: np.random.Generator, ci: float = 0.95) -> tuple[float, float]:
    """IC por percentiles de la media, remuestreando CHIPS (no filas) con
    reemplazo — respeta que varias lecturas del mismo chip no son
    independientes entre sí."""
    n = len(chip_means)
    if n < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = chip_means[idx].mean(axis=1)
    return percentile_ci(boot_means, ci=ci)


def summarize(df: pd.DataFrame, n_resamples: int = 10000, bootstrap_seed: int = 0,
              success_threshold: float | None = None) -> pd.DataFrame:
    rows = []
    baseline = df[df["noise_model"] == "none"]
    rng = np.random.default_rng(bootstrap_seed)

    for keys, group in df.groupby(GROUP_COLS, dropna=False):
        mode, noise_model, weight, t_inference, prog_scale, read_scale, drift_scale = keys
        row = {
            "mode": mode, "noise_model": noise_model,
            "weight": weight, "t_inference": t_inference,
            "prog_noise_scale": prog_scale, "read_noise_scale": read_scale,
            "drift_scale": drift_scale,
            "n": len(group),
        }
        per_chip_groups = group.groupby(CHIP_COLS)
        row["n_chips"] = per_chip_groups.ngroups
        row["reads_per_chip"] = len(group) / per_chip_groups.ngroups

        for col in REWARD_COLS:
            # Nivel "fila completa" (lo que ya había) — se mantiene para no
            # romper a plot_car_noise.py / plot_car_noise_time.py.
            mean = group[col].mean()
            std = group[col].std(ddof=1) if len(group) > 1 else float("nan")
            sem = std / (len(group) ** 0.5) if len(group) > 1 else float("nan")
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_sem"] = sem

            # Dos niveles de varianza, por chip.
            per_chip_mean = per_chip_groups[col].mean()  # una fila por (seed,chip)
            per_chip_std = per_chip_groups[col].std(ddof=1)  # NaN si ese chip tuvo 1 sola lectura

            row[f"{col}_between_chip_std"] = (
                per_chip_mean.std(ddof=1) if len(per_chip_mean) > 1 else float("nan")
            )
            row[f"{col}_within_chip_std"] = per_chip_std.mean(skipna=True)

            p10, p50, p90 = per_chip_mean.quantile([0.10, 0.50, 0.90])
            row[f"{col}_p10"] = p10
            row[f"{col}_p50"] = p50
            row[f"{col}_p90"] = p90

            ci_lo, ci_hi = bootstrap_mean_ci(per_chip_mean.to_numpy(), n_resamples, rng)
            row[f"{col}_ci_lo"] = ci_lo
            row[f"{col}_ci_hi"] = ci_hi

            if success_threshold is not None:
                row[f"{col}_frac_chips_above_{success_threshold:g}"] = float(
                    (per_chip_mean >= success_threshold).mean())

            if noise_model != "none":
                base = baseline[(baseline["mode"] == mode) & (baseline["weight"] == weight)]
                if not base.empty:
                    base_mean = base[col].mean()
                    row[f"{col}_delta_mean"] = mean - base_mean
                    row[f"{col}_delta_pct"] = (
                        (mean - base_mean) / base_mean * 100.0 if base_mean != 0 else float("nan")
                    )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(GROUP_COLS).reset_index(drop=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", nargs="+", required=True,
                     help="uno o más CSV de wann_car_noise (se concatenan)")
    ap.add_argument("--out", default=None,
                     help="si se da, además guarda el resumen en este CSV")
    ap.add_argument("--resamples", type=int, default=10000,
                     help="remuestreos del bootstrap de la media (default: 10000)")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--success-threshold", type=float, default=None,
                     help="si se da, agrega frac_chips_above_<umbral> (sobre reward_shaped y "
                          "reward_original) — fracción de chips cuyo promedio queda por encima")
    args = ap.parse_args()

    df = load_rows(args.csv)
    summary = summarize(df, n_resamples=args.resamples, bootstrap_seed=args.bootstrap_seed,
                        success_threshold=args.success_threshold)

    with pd.option_context("display.max_rows", None, "display.max_columns", None,
                            "display.width", 200):
        print(summary.to_string(index=False))

    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"\nGuardado en {args.out}")


if __name__ == "__main__":
    main()
