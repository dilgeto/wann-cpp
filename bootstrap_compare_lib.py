#!/usr/bin/env python3
"""
bootstrap_compare_lib.py — funciones compartidas por los scripts
bootstrap_compare_{acrobot,mountain_car,...}.py: estadística descriptiva,
bootstrap NO pareado (con y sin ajuste de signo), intervalos de confianza
por percentiles, gráficos, y guardado de resultados a CSV.

No se entrena ni evalúa nada acá — cada script de tarea provee sus propias
funciones eval_snn()/eval_ann() y llama a run_comparison() con los vectores
de reward ya calculados.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def describe(rewards: np.ndarray) -> dict:
    return {
        "n_episodes": len(rewards),
        "mean":       float(np.mean(rewards)),
        "std":        float(np.std(rewards, ddof=1)),
        "median":     float(np.median(rewards)),
        "min":        float(np.min(rewards)),
        "max":        float(np.max(rewards)),
    }


def bootstrap_unpaired(rewards_snn: np.ndarray, rewards_ann: np.ndarray,
                       n_resamples: int, rng: np.random.Generator):
    """Bootstrap NO pareado: cada lado se remuestrea con reemplazo por su
    cuenta (mismo tamaño que su muestra original), de forma independiente."""
    n_snn, n_ann = len(rewards_snn), len(rewards_ann)
    idx_snn = rng.integers(0, n_snn, size=(n_resamples, n_snn))
    idx_ann = rng.integers(0, n_ann, size=(n_resamples, n_ann))

    boot_mean_snn = rewards_snn[idx_snn].mean(axis=1)
    boot_mean_ann = rewards_ann[idx_ann].mean(axis=1)
    boot_diff  = boot_mean_snn - boot_mean_ann
    boot_ratio = boot_mean_snn / boot_mean_ann * 100.0
    return boot_mean_snn, boot_mean_ann, boot_diff, boot_ratio


def signed_performance_pct(snn_vals, ann_vals):
    """% de rendimiento de SNN respecto a ANN, consciente del signo: 100%
    cuando snn==ann, >100% cuando SNN es mejor (más cercano a 0 o mayor),
    <100% cuando es peor. Válido tanto para recompensas positivas (donde
    coincide exactamente con el ratio literal 100*snn/ann, ya que
    abs(ann)==ann cuando ann>0) como negativas (Acrobot/Mountain Car: más
    negativo=peor), donde el ratio literal 100*snn/ann se invierte de
    sentido. Misma idea que scaled_target() en bootstrap_compare.py,
    aplicada a un ratio en vez de a un umbral."""
    return 100.0 * (1.0 + (snn_vals - ann_vals) / np.abs(ann_vals))


def percentile_ci(x: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - ci
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


# Tamaños de fuente centralizados — subir acá si hace falta más letra grande.
FONT_TITLE    = 15
FONT_LABEL    = 13
FONT_LEGEND   = 11
FONT_TICK     = 11
FONT_SUPTITLE = 18


def make_plots(rewards_snn: np.ndarray, rewards_ann: np.ndarray,
              boot_ratio_signed: np.ndarray, ci_lo: float, ci_hi: float, ci: float,
              out_path: Path, suptitle: str, ann_label: str = "ANN") -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Mismo rango de bins para ambos histogramas superiores, así el eje X
    # queda alineado y las barras son directamente comparables.
    combined_min = min(rewards_snn.min(), rewards_ann.min())
    combined_max = max(rewards_snn.max(), rewards_ann.max())
    shared_bins = np.linspace(combined_min, combined_max, 21)

    ax = axes[0, 0]
    ax.hist(rewards_snn, bins=shared_bins, color="#d62728", edgecolor="black", alpha=0.85,
            label=f"Episodios individuales (n={len(rewards_snn)})")
    ax.set_title("Rewards SNN (episodios)", fontsize=FONT_TITLE)
    ax.set_xlabel("Reward", fontsize=FONT_LABEL)
    ax.set_ylabel("Frecuencia", fontsize=FONT_LABEL)
    ax.axvline(rewards_snn.mean(), color="black", linestyle="--", linewidth=1,
               label=f"media={rewards_snn.mean():.2f}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")

    ax = axes[0, 1]
    ax.hist(rewards_ann, bins=shared_bins, color="#4C72B0", edgecolor="black", alpha=0.85,
            label=f"Episodios individuales (n={len(rewards_ann)})")
    ax.set_title(f"Rewards {ann_label} (episodios)", fontsize=FONT_TITLE)
    ax.set_xlabel("Reward", fontsize=FONT_LABEL)
    ax.set_ylabel("Frecuencia", fontsize=FONT_LABEL)
    ax.axvline(rewards_ann.mean(), color="black", linestyle="--", linewidth=1,
               label=f"media={rewards_ann.mean():.2f}")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")

    # Mismo eje X e Y en ambos histogramas superiores para que la
    # comparación SNN vs ANN sea visualmente directa.
    shared_xlim = (combined_min, combined_max)
    shared_ylim = (0, max(axes[0, 0].get_ylim()[1], axes[0, 1].get_ylim()[1]))
    for ax in (axes[0, 0], axes[0, 1]):
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)

    ax = axes[1, 0]
    ax.hist(boot_ratio_signed, bins=40, color="#808080", edgecolor="black", alpha=0.85,
            label="Remuestreos bootstrap")
    ax.axvline(90.0, color="crimson", linestyle="--", linewidth=1.5,
               label="Objetivo mínimo (90%)")
    ax.axvline(ci_lo, color="black", linestyle=":", linewidth=1.3,
               label=f"IC {int(round(ci * 100))}% (percentiles {(1 - ci) / 2 * 100:.1f}/{(1 + ci) / 2 * 100:.1f})")
    ax.axvline(ci_hi, color="black", linestyle=":", linewidth=1.3)
    ax.set_title(f"Distribución bootstrap: % rendimiento SNN vs {ann_label}\n(consciente del signo)",
                fontsize=FONT_TITLE)
    ax.set_xlabel("% de rendimiento", fontsize=FONT_LABEL)
    ax.set_ylabel("Frecuencia", fontsize=FONT_LABEL)
    # Espacio reservado arriba para que la leyenda no tape las barras del
    # histograma, sea cual sea la forma de la distribución.
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
    ax.legend(fontsize=FONT_LEGEND, loc="upper left")

    ax = axes[1, 1]
    bp = ax.boxplot([rewards_snn, rewards_ann], tick_labels=["SNN", ann_label],
                    patch_artist=True, widths=0.6, medianprops=dict(color="black"))
    for patch, color in zip(bp["boxes"], ["#d62728", "#4C72B0"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_alpha(0.85)
    ax.set_title("Comparación de episodios", fontsize=FONT_TITLE)
    ax.set_ylabel("Reward", fontsize=FONT_LABEL)

    for ax in axes.flat:
        ax.tick_params(axis="both", labelsize=FONT_TICK)

    fig.suptitle(suptitle, fontweight="bold", fontsize=FONT_SUPTITLE)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def build_arg_parser(doc: str, default_out_dir: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100,
                    help="Episodios a evaluar por agente (default: 100)")
    ap.add_argument("--seed0", type=int, default=0,
                    help="Primera seed de entorno (default: 0)")
    ap.add_argument("--resamples", type=int, default=10000,
                    help="Remuestreos bootstrap (default: 10000, mínimo 10000)")
    ap.add_argument("--ci", type=float, default=0.95,
                    help="Nivel de confianza (default: 0.95)")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="Semilla del generador aleatorio del bootstrap (default: 0)")
    ap.add_argument("--omp", type=int, default=None,
                    help="OMP_NUM_THREADS para el binario SNN (default: cpu_count)")
    ap.add_argument("--timeout", type=int, default=None,
                    help="Timeout en segundos por evaluación (default: sin límite)")
    ap.add_argument("--out-dir", default=default_out_dir, dest="out_dir",
                    help=f"Directorio de salida (default: {default_out_dir}/)")
    return ap


def run_comparison(rewards_snn: np.ndarray, rewards_ann: np.ndarray, args: argparse.Namespace,
                   suptitle: str, plot_stem: str, ann_label: str = "ANN") -> None:
    """Calcula estadísticas + bootstrap, imprime el resumen, guarda los CSV
    y genera el gráfico — igual para cualquier tarea."""
    if args.resamples < 10000:
        raise SystemExit("--resamples debe ser al menos 10000.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats_snn = describe(rewards_snn)
    stats_ann = describe(rewards_ann)
    ratio_point        = stats_snn["mean"] / stats_ann["mean"] * 100.0
    ratio_signed_point = float(signed_performance_pct(stats_snn["mean"], stats_ann["mean"]))

    rng = np.random.default_rng(args.rng_seed)
    boot_mean_snn, boot_mean_ann, boot_diff, boot_ratio = bootstrap_unpaired(
        rewards_snn, rewards_ann, args.resamples, rng)
    boot_ratio_signed = signed_performance_pct(boot_mean_snn, boot_mean_ann)

    diff_ci_lo, diff_ci_hi = percentile_ci(boot_diff, args.ci)
    ratio_ci_lo, ratio_ci_hi = percentile_ci(boot_ratio, args.ci)
    ratio_signed_ci_lo, ratio_signed_ci_hi = percentile_ci(boot_ratio_signed, args.ci)
    prob_ge_90 = float(np.mean(boot_ratio_signed >= 90.0))
    meets_90 = ratio_signed_ci_lo >= 90.0
    # alpha resultante: nivel de significancia de una cola al que el objetivo
    # (>=90%) queda justo en el borde del intervalo bootstrap (menor = más
    # confianza) — misma idea que alpha_result en bootstrap_compare.py.
    alpha_result = float(np.mean(boot_ratio_signed < 90.0))

    # ── Salida por consola ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {suptitle}")
    print(f"{'='*70}")
    print(f"  SNN  (n={stats_snn['n_episodes']}): "
          f"media={stats_snn['mean']:.4f} ± {stats_snn['std']:.4f}  "
          f"mediana={stats_snn['median']:.4f}  min={stats_snn['min']:.4f}  max={stats_snn['max']:.4f}")
    print(f"  {ann_label}  (n={stats_ann['n_episodes']}): "
          f"media={stats_ann['mean']:.4f} ± {stats_ann['std']:.4f}  "
          f"mediana={stats_ann['median']:.4f}  min={stats_ann['min']:.4f}  max={stats_ann['max']:.4f}")
    print(f"  % rendimiento consciente del signo (100%=igual, >100%=SNN mejor) = "
          f"{ratio_signed_point:.2f}%")
    print(f"  IC{int(args.ci*100)}% bootstrap del % rendimiento (con signo) = "
          f"[{ratio_signed_ci_lo:.2f}%, {ratio_signed_ci_hi:.2f}%]")
    print(f"  [ref.] % rendimiento literal (mean SNN / mean {ann_label} × 100, SIN ajuste "
          f"de signo — engañoso con reward negativo) = {ratio_point:.2f}%  "
          f"IC{int(args.ci*100)}%=[{ratio_ci_lo:.2f}%, {ratio_ci_hi:.2f}%]")
    print(f"  IC{int(args.ci*100)}% bootstrap de la diferencia (SNN-{ann_label}) = "
          f"[{diff_ci_lo:.4f}, {diff_ci_hi:.4f}]")
    print(f"  P(bootstrap) de que % rendimiento (con signo) >= 90% = {prob_ge_90:.4f}")
    print(f"  ¿Límite inferior del IC del % rendimiento (con signo) >= 90%? "
          f"{'SÍ' if meets_90 else 'NO'} ({ratio_signed_ci_lo:.2f}% {'>=' if meets_90 else '<'} 90%)")
    print(f"  alpha resultante (P bootstrap de que % rendimiento < 90%) = {alpha_result:.4f}  "
          f"(se exige alpha<={1 - args.ci:.2f} para IC{int(args.ci*100)}%)")
    print(f"{'='*70}")

    # ── Guardar resultados ───────────────────────────────────────────────
    rewards_df = pd.concat([
        pd.DataFrame({"agent": "snn", "episode": np.arange(len(rewards_snn)),
                     "seed": args.seed0 + np.arange(len(rewards_snn)), "reward": rewards_snn}),
        pd.DataFrame({"agent": "ann", "episode": np.arange(len(rewards_ann)),
                     "seed": args.seed0 + np.arange(len(rewards_ann)), "reward": rewards_ann}),
    ], ignore_index=True)
    rewards_df.to_csv(out_dir / "rewards.csv", index=False)

    boot_df = pd.DataFrame({
        "boot_mean_snn": boot_mean_snn,
        "boot_mean_ann": boot_mean_ann,
        "boot_diff":     boot_diff,
        "boot_ratio_pct_literal": boot_ratio,
        "boot_ratio_pct_signed":  boot_ratio_signed,
    })
    boot_df.to_csv(out_dir / "bootstrap_samples.csv", index=False)

    summary_df = pd.DataFrame([
        {"agent": "snn", **stats_snn},
        {"agent": "ann", **stats_ann},
    ])
    summary_df.to_csv(out_dir / "summary_stats.csv", index=False)

    ci_df = pd.DataFrame([{
        "ratio_literal_point_pct": ratio_point,
        "ratio_literal_ci_lo_pct": ratio_ci_lo,
        "ratio_literal_ci_hi_pct": ratio_ci_hi,
        "ratio_signed_point_pct":  ratio_signed_point,
        "ratio_signed_ci_lo_pct":  ratio_signed_ci_lo,
        "ratio_signed_ci_hi_pct":  ratio_signed_ci_hi,
        "diff_point":        stats_snn["mean"] - stats_ann["mean"],
        "diff_ci_lo":        diff_ci_lo,
        "diff_ci_hi":        diff_ci_hi,
        "prob_ratio_signed_ge_90": prob_ge_90,
        "alpha_result":            alpha_result,
        "meets_90pct_target":      meets_90,
        "ci_level":          args.ci,
        "n_resamples":       args.resamples,
        "n_episodes":        args.n,
    }])
    ci_df.to_csv(out_dir / "ci_results.csv", index=False)

    plot_path = out_dir / f"{plot_stem}.png"
    make_plots(rewards_snn, rewards_ann, boot_ratio_signed,
              ratio_signed_ci_lo, ratio_signed_ci_hi, args.ci,
              plot_path, suptitle, ann_label)

    print(f"\nResultados guardados en {out_dir}/:")
    print("  rewards.csv, bootstrap_samples.csv, summary_stats.csv, ci_results.csv")
    print(f"  {plot_path}")
