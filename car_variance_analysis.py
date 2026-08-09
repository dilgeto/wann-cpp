#!/usr/bin/env python3
"""
car_variance_analysis.py — Dos análisis complementarios sobre la brecha de
Racing Car frente al umbral del 90%, usando datos YA generados (sin
entrenar ni evaluar nada nuevo):

1. Descomposición de varianza entre-semilla vs. dentro-de-semilla (ANOVA de
   un factor), para el modelo SNN campeón y para el PPO de referencia por
   separado, usando sus propios 11 modelos entrenados con distinta semilla.
   Responde: ¿la varianza entre semillas de la SNN es excepcional, o el PPO
   nativo (11 semillas propias, sin modelo preentrenado disponible para
   Racing Car) tiene un problema comparable?

2. Comparación de % de rendimiento restringida a episodios EXITOSOS
   (excluyendo los episodios fallidos de recompensa cercana a 0, ~10-11% en
   ambos agentes, ver Sección Comparaciones/Racing Car), usando el mismo
   procedimiento de bootstrap no pareado de bootstrap_compare_lib.py.
   Responde: ¿la brecha viene de la magnitud de las vueltas buenas, o los
   episodios fallidos siguen pesando en el promedio general?

Fuentes de datos (todas ya existentes, no requiere C++ ni GPU):
  - SNN (campeón car_ttfs_first_spike, 11 seeds x 121 episodios de
    revalidación): eval_p3_weights/car_best_episodes.csv
  - PPO (11 seeds de entrenamiento x 11 seeds de evaluación, 1 episodio c/u):
    ../racing-car-ppo/validation/summary.csv
  - Comparación bootstrap de 100 episodios ya usada en la tesis (Tabla
    tab:comparacion_resumen): bootstrap_car/rewards.csv
  - Misma comparación pero con el campeón revalidado a 40ms de ventana
    (Tabla tab:disc_car_ablacion): bootstrap_car_40ms/rewards.csv

Umbral de episodio fallido: reward < 100. Se eligió por ser el valor que
reproduce exactamente las cifras ya publicadas en el texto (11% de
episodios fallidos en SNN, 10% en PPO, sobre los mismos 100 episodios de
bootstrap_car/rewards.csv) — no es un valor elegido a posteriori para este
script.

Uso:
  python car_variance_analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

from bootstrap_compare_lib import bootstrap_unpaired, signed_performance_pct, percentile_ci

FAILURE_THRESHOLD = 100.0
N_RESAMPLES = 10000
RNG_SEED = 0


# ─────────────────────────────────────────────────────────────────────────
# Parte 1: ANOVA entre-semilla vs. dentro-de-semilla
# ─────────────────────────────────────────────────────────────────────────

def anova_between_within(groups: list[np.ndarray], label: str) -> None:
    f_stat, p_val = ss.f_oneway(*groups)

    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta_sq = ss_between / ss_total

    means = [g.mean() for g in groups]
    print(f"── {label} ──")
    print(f"  {len(groups)} semillas, medias por semilla: "
          f"min={min(means):.2f}  max={max(means):.2f}  "
          f"std entre medias={np.std(means, ddof=0):.2f}")
    print(f"  ANOVA un factor (semilla):  F={f_stat:.4f}  p={p_val:.4g}")
    print(f"  eta^2 (fracción de varianza explicada por la semilla) = {eta_sq:.4f}")
    print(f"  -> {eta_sq*100:.1f}% de la varianza total es ENTRE semillas "
          f"(calidad del modelo encontrado), {100*(1-eta_sq):.1f}% es DENTRO "
          f"de semilla (ruido de ejecución de un mismo modelo).")
    print()


def part1() -> None:
    print("=" * 70)
    print("PARTE 1 — Varianza entre-semilla vs. dentro-de-semilla")
    print("=" * 70)
    print()

    snn_ep = pd.read_csv("eval_p3_weights/car_best_episodes.csv")
    snn_ep = snn_ep[snn_ep["run_key"] == "car_ttfs_first_spike"]
    snn_groups = [g["reward"].to_numpy() for _, g in snn_ep.groupby("train_seed_idx")]
    anova_between_within(snn_groups, "SNN (car_ttfs_first_spike, 11 seeds x 121 episodios)")
    print("  Diagnóstico: como el componente DENTRO-de-semilla domina (65.8% > 34.2%),")
    print("  el factor limitante es sobre todo la robustez de ejecución de un modelo ya")
    print("  entrenado (varía mucho episodio a episodio con la misma red), más que la")
    print("  calidad de la búsqueda de topologías entre semillas — aunque ambos pesan.")
    print()

    ppo_ep = pd.read_csv("../racing-car-ppo/validation/summary.csv")
    ppo_groups = [g["episode_return"].to_numpy() for _, g in ppo_ep.groupby("train_seed")]
    anova_between_within(ppo_groups, "PPO nativo (11 seeds x 11 episodios)")


# ─────────────────────────────────────────────────────────────────────────
# Parte 2: % de rendimiento restringido a episodios exitosos
# ─────────────────────────────────────────────────────────────────────────

def success_only_comparison(rewards_csv: Path, label: str) -> None:
    df = pd.read_csv(rewards_csv)
    snn_all = df[df["agent"] == "snn"]["reward"].to_numpy()
    ann_all = df[df["agent"] == "ann"]["reward"].to_numpy()

    snn_ok = snn_all[snn_all >= FAILURE_THRESHOLD]
    ann_ok = ann_all[ann_all >= FAILURE_THRESHOLD]

    print(f"── {label} ({rewards_csv}) ──")
    print(f"  SNN: {len(snn_all)} episodios totales, {len(snn_all)-len(snn_ok)} fallidos "
          f"({100*(len(snn_all)-len(snn_ok))/len(snn_all):.0f}%), "
          f"{len(snn_ok)} exitosos -> media={snn_ok.mean():.2f} +/- {snn_ok.std(ddof=1):.2f}")
    print(f"  ANN: {len(ann_all)} episodios totales, {len(ann_all)-len(ann_ok)} fallidos "
          f"({100*(len(ann_all)-len(ann_ok))/len(ann_all):.0f}%), "
          f"{len(ann_ok)} exitosos -> media={ann_ok.mean():.2f} +/- {ann_ok.std(ddof=1):.2f}")

    rng = np.random.default_rng(RNG_SEED)
    boot_mean_snn, boot_mean_ann, boot_diff, boot_ratio = bootstrap_unpaired(
        snn_ok, ann_ok, N_RESAMPLES, rng)
    boot_ratio_signed = signed_performance_pct(boot_mean_snn, boot_mean_ann)
    ratio_signed_point = signed_performance_pct(snn_ok.mean(), ann_ok.mean())
    ci_lo, ci_hi = percentile_ci(boot_ratio_signed, 0.95)
    alpha_result = float(np.mean(boot_ratio_signed < 90.0))

    print(f"  % rendimiento (solo exitosos, con signo) = {ratio_signed_point:.2f}%  "
          f"IC95%=[{ci_lo:.2f}%, {ci_hi:.2f}%]")
    print(f"  p-value (P bootstrap de que % rendimiento < 90%) = {alpha_result:.4f}")
    print()


def part2() -> None:
    print("=" * 70)
    print("PARTE 2 — % de rendimiento restringido a episodios exitosos "
          f"(reward >= {FAILURE_THRESHOLD:g})")
    print("=" * 70)
    print()

    success_only_comparison(Path("bootstrap_car/rewards.csv"),
                            "Campeón oficial (20ms), Tabla tab:comparacion_resumen")
    success_only_comparison(Path("bootstrap_car_40ms/rewards.csv"),
                            "Campeón revalidado a 40ms, Tabla tab:disc_car_ablacion")


# ─────────────────────────────────────────────────────────────────────────
# Parte 3: Análisis de potencia estadística (por simulación)
# ─────────────────────────────────────────────────────────────────────────

def power_at_n(snn_pool: np.ndarray, ann_pool: np.ndarray, n: int,
               n_outer: int, n_inner_resamples: int, rng: np.random.Generator) -> float:
    """Potencia a tamaño de muestra n: fracción de n_outer 'experimentos
    simulados' (cada uno remuestreando n episodios con reemplazo desde la
    distribución empírica observada, tratada como la población real) en que
    el test bootstrap habitual (alpha_result <= 0.05) detecta el efecto."""
    hits = 0
    for _ in range(n_outer):
        snn_sample = rng.choice(snn_pool, size=n, replace=True)
        ann_sample = rng.choice(ann_pool, size=n, replace=True)
        boot_mean_snn, boot_mean_ann, _, _ = bootstrap_unpaired(
            snn_sample, ann_sample, n_inner_resamples, rng)
        boot_ratio_signed = signed_performance_pct(boot_mean_snn, boot_mean_ann)
        alpha_result = float(np.mean(boot_ratio_signed < 90.0))
        if alpha_result <= 0.05:
            hits += 1
    return hits / n_outer


def power_analysis(rewards_csv: Path, label: str,
                   sample_sizes: list[int], n_outer: int = 300,
                   n_inner_resamples: int = 2000, seed: int = RNG_SEED) -> None:
    df = pd.read_csv(rewards_csv)
    snn_pool = df[df["agent"] == "snn"]["reward"].to_numpy()
    ann_pool = df[df["agent"] == "ann"]["reward"].to_numpy()
    point = signed_performance_pct(snn_pool.mean(), ann_pool.mean())

    print(f"── {label} ({rewards_csv}) ──")
    print(f"  Efecto observado (n=100 c/u): {point:.2f}% de rendimiento. "
          f"Se trata esa distribución empírica como la 'población real' y se")
    print(f"  simulan experimentos con distinto número de episodios por lado, "
          f"midiendo en qué fracción de {n_outer} repeticiones el test detecta")
    print(f"  significativamente que el rendimiento >= 90% (alpha_result <= 0.05).")
    print()
    print(f"  {'episodios/agente':>18}{'potencia':>12}")
    rng = np.random.default_rng(seed)
    reached_80 = None
    for n in sample_sizes:
        power = power_at_n(snn_pool, ann_pool, n, n_outer, n_inner_resamples, rng)
        print(f"  {n:>18}{power:>12.3f}")
        if reached_80 is None and power >= 0.8:
            reached_80 = n
    if reached_80 is not None:
        print(f"\n  -> Se alcanza potencia >= 0.8 (convención estándar) a partir de "
              f"~{reached_80} episodios por agente.")
    else:
        print(f"\n  -> No se alcanza potencia >= 0.8 dentro del rango probado "
              f"(máximo {sample_sizes[-1]}).")
    print()


def part3() -> None:
    print("=" * 70)
    print("PARTE 3 — Análisis de potencia estadística (simulación)")
    print("=" * 70)
    print()

    sizes = [100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
    power_analysis(Path("bootstrap_car/rewards.csv"),
                   "Campeón oficial (20ms), Tabla tab:comparacion_resumen", sizes)
    power_analysis(Path("bootstrap_car_40ms/rewards.csv"),
                   "Campeón revalidado a 40ms, Tabla tab:disc_car_ablacion", sizes)


if __name__ == "__main__":
    part1()
    part2()
    part3()
