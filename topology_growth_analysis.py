#!/usr/bin/env python3
"""
topology_growth_analysis.py — Tres análisis complementarios sobre el
crecimiento topológico posterior a la convergencia del fitness (Discusión,
"Simplicidad estructural de las redes obtenidas"), usando datos YA
generados (curvas _stats.out de Fase 3 y snapshots periódicos de la red,
sin entrenar ni evaluar nada nuevo):

1. ¿Es un patrón del algoritmo o solo de la semilla ganadora? Repite el
   chequeo de "última generación con cambio de fitTop" sobre las 11 semillas
   de Fase 3 de cada una de las 3 configuraciones campeonas (no solo la
   semilla elegida como campeona), para ver si el estancamiento-con-
   crecimiento es un patrón general o un caso particular de la semilla que
   ganó.
2. ¿El crecimiento posterior agrega estructura nueva o redundante? Compara
   la composición de tipos de neurona (comportamiento de Izhikevich) y la
   proporción excitatoria/inhibitoria entre un snapshot cercano al punto de
   convergencia y la red final, para Mountain Car y Racing Car (los 2 casos
   con un patrón de estancamiento limpio).
3. ¿Qué hiperparámetros predicen más o menos crecimiento posterior a la
   convergencia? Correlación de Spearman entre los hiperparámetros
   ganadores y el crecimiento relativo de conexiones después del último
   cambio de fitTop, a través de las 12 combinaciones tarea/codificador/
   decodificador (cada una con su propia configuración y semilla campeona).

Uso:
  python topology_growth_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as ss

STATS_COLS = ["evals", "fitMed", "fitMax", "fitTop", "fitPeak",
              "nodeMed", "connMed", "fitTopOrig"]

HP_COLS = ["alg_probMoo", "prob_addConn", "prob_addNode", "prob_enable",
           "prob_mutAct", "prob_toggleExcitatory", "prob_initEnable",
           "select_cullRatio", "select_eliteRatio", "select_tournSize"]

CHAMPIONS = {
    "acrobot":      ("acrobot_small_first_spike",      1, 1),
    "mountain_car": ("mountain_car_small_first_spike",  2, 10),
    "car":          ("car_ttfs_first_spike",            2, 0),
}

ALL_RUN_KEYS = {
    "acrobot": ["acrobot_small_first_spike", "acrobot_small_rate_argmax",
                "acrobot_ttfs_first_spike", "acrobot_ttfs_rate_argmax"],
    "mountain_car": ["mountain_car_small_first_spike", "mountain_car_small_rate_argmax",
                     "mountain_car_ttfs_first_spike", "mountain_car_ttfs_rate_argmax"],
    "car": ["car_small_first_spike", "car_small_rate",
            "car_ttfs_first_spike", "car_ttfs_rate"],
}


def load_stats(run_key: str, rank: int, seed_idx: int) -> pd.DataFrame:
    path = Path(f"log/full_p3_{run_key}/rank{rank:02d}_seed{seed_idx:02d}_stats.out")
    df = pd.read_csv(path, header=None, names=STATS_COLS)
    df["gen"] = range(1, len(df) + 1)
    return df


def last_change_gen(df: pd.DataFrame) -> int:
    """Última generación en que fitTop cambió de valor."""
    changed = df[df["fitTop"] != df["fitTop"].shift(1)]
    return int(changed["gen"].iloc[-1])


def parse_network(path: Path) -> tuple[int, int, int, list[int]]:
    """Lee un .out exportado por Ind::exportNet.
    Retorna (n_neuronas, n_excitatorias, n_inhibitorias, lista_de_activaciones)."""
    with open(path) as f:
        rows = [line.rstrip("\n").split(",") for line in f if line.strip()]
    n = len(rows)
    n_exc = n_inh = 0
    acts = []
    for row in rows:
        for tok in row[:n]:
            if tok == "nan":
                continue
            v = float(tok)
            if v == 1.0:
                n_exc += 1
            elif v == -1.0:
                n_inh += 1
        acts.append(int(float(row[n])))
    return n, n_exc, n_inh, acts


# ─────────────────────────────────────────────────────────────────────────
# Parte 1: ¿patrón del algoritmo o de la semilla ganadora?
# ─────────────────────────────────────────────────────────────────────────

def part1() -> None:
    print("=" * 70)
    print("PARTE 1 — Estancamiento-con-crecimiento a través de las 11 semillas")
    print("=" * 70)
    print()

    for task, (run_key, rank, champ_seed) in CHAMPIONS.items():
        print(f"── {task} ({run_key}, rank={rank}) ──")
        print(f"  {'seed':>5}{'último cambio fitTop':>22}{'nodeMed conv->final':>24}"
              f"{'connMed conv->final':>24}{'crec. conexiones':>18}")
        for seed_idx in range(11):
            try:
                df = load_stats(run_key, rank, seed_idx)
            except FileNotFoundError:
                continue
            lg = last_change_gen(df)
            conv_row = df.iloc[lg - 1]
            final_row = df.iloc[-1]
            conn_growth = (final_row["connMed"] / conv_row["connMed"] - 1) * 100 \
                if conv_row["connMed"] > 0 else float("nan")
            marker = " <- campeona" if seed_idx == champ_seed else ""
            print(f"  {seed_idx:>5}{lg:>22}"
                  f"{f'{conv_row.nodeMed:.0f} -> {final_row.nodeMed:.0f}':>24}"
                  f"{f'{conv_row.connMed:.0f} -> {final_row.connMed:.0f}':>24}"
                  f"{conn_growth:>+17.0f}%{marker}")
        print()


# ─────────────────────────────────────────────────────────────────────────
# Parte 2: ¿estructura nueva o redundante?
# ─────────────────────────────────────────────────────────────────────────

ACT_NAMES = {1: "REGULAR_SPIKING", 2: "FAST_SPIKING", 3: "CHATTERING",
             4: "LOW_THRESHOLD_SPIKING", 5: "INTRINSICALLY_BURSTING",
             6: "RESONATOR", 7: "FAST_SPIKING", 8: "REGULAR_SPIKING",
             9: "REGULAR_SPIKING", 10: "CHATTERING"}


def part2() -> None:
    print("=" * 70)
    print("PARTE 2 — Composición estructural: snapshot de convergencia vs. final")
    print("=" * 70)
    print()

    cases = [
        ("mountain_car", "mountain_car_small_first_spike", 2, 10, 30),
        ("car",          "car_ttfs_first_spike",            2, 0,  40),
        ("acrobot",      "acrobot_small_first_spike",       1, 1,  200),
    ]

    for task, run_key, rank, seed_idx, snap_gen in cases:
        base = Path(f"log/full_p3_{run_key}/rank{rank:02d}_seed{seed_idx:02d}_best")
        snap_path = base / f"{snap_gen:04d}.out"
        final_path = Path(f"log/full_p3_{run_key}/rank{rank:02d}_seed{seed_idx:02d}_best.out")

        n_s, exc_s, inh_s, acts_s = parse_network(snap_path)
        n_f, exc_f, inh_f, acts_f = parse_network(final_path)

        print(f"── {task} ({run_key}) — snapshot gen {snap_gen} vs. final ──")
        print(f"  Neuronas:      {n_s:>4}  ->  {n_f:>4}   (+{n_f-n_s})")
        print(f"  Excitatorias:  {exc_s:>4}  ->  {exc_f:>4}   (+{exc_f-exc_s})")
        print(f"  Inhibitorias:  {inh_s:>4}  ->  {inh_f:>4}   (+{inh_f-inh_s})")
        ratio_s = exc_s / max(inh_s, 1)
        ratio_f = exc_f / max(inh_f, 1)
        print(f"  Razón exc/inh: {ratio_s:.2f}  ->  {ratio_f:.2f}")

        types_s = pd.Series([ACT_NAMES.get(a, "?") for a in acts_s]).value_counts()
        types_f = pd.Series([ACT_NAMES.get(a, "?") for a in acts_f]).value_counts()
        all_types = sorted(set(types_s.index) | set(types_f.index))
        print(f"  Tipos de neurona (comportamiento Izhikevich):")
        for t in all_types:
            cs, cf = types_s.get(t, 0), types_f.get(t, 0)
            print(f"    {t:<24} {cs:>4}  ->  {cf:>4}")
        n_new_types = len(set(types_f.index) - set(types_s.index))
        print(f"  Tipos de neurona NUEVOS que no estaban en el snapshot: {n_new_types}")
        print()


# ─────────────────────────────────────────────────────────────────────────
# Parte 3: hiperparámetros vs. crecimiento posterior a la convergencia
# ─────────────────────────────────────────────────────────────────────────

def part3() -> None:
    print("=" * 70)
    print("PARTE 3 — Hiperparámetros vs. crecimiento posterior a la convergencia")
    print("=" * 70)
    print()

    rows = []
    for task, run_keys in ALL_RUN_KEYS.items():
        best_csv = pd.read_csv(f"eval_p3_weights/{task}_best.csv")
        for run_key in run_keys:
            g = best_csv[best_csv["run_key"] == run_key]
            if g.empty:
                continue
            best = g.loc[g["reward"].idxmax()]
            rank, seed_idx = int(best["rank"]), int(best["seed_idx"])

            cfg_path = Path(f"screening_full/{run_key}/p3_configs/"
                            f"rank{rank:02d}_seed{seed_idx:02d}.json")
            hp = json.loads(cfg_path.read_text())

            df = load_stats(run_key, rank, seed_idx)
            lg = last_change_gen(df)
            conv_row = df.iloc[lg - 1]
            final_row = df.iloc[-1]
            if conv_row["connMed"] <= 0:
                continue
            conn_growth_pct = (final_row["connMed"] / conv_row["connMed"] - 1) * 100
            gen_frac = lg / len(df)

            row = {"run_key": run_key, "conn_growth_pct": conn_growth_pct,
                  "conv_gen_frac": gen_frac}
            for c in HP_COLS:
                row[c] = hp.get(c)
            rows.append(row)

    df_hp = pd.DataFrame(rows)
    print(df_hp[["run_key", "conv_gen_frac", "conn_growth_pct"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()
    print(f"N = {len(df_hp)} combinaciones tarea/codificador/decodificador")
    print()
    print(f"── Spearman: hiperparámetro vs. %% de crecimiento de conexiones "
          f"tras la convergencia ──")
    print(f"  {'Parámetro':<28}{'rho':>8}{'p-val':>10}  sig")
    results = []
    for c in HP_COLS:
        rho, p = ss.spearmanr(df_hp[c], df_hp["conn_growth_pct"])
        results.append((c, rho, p))
    for c, rho, p in sorted(results, key=lambda x: abs(x[1]), reverse=True):
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {c:<28}{rho:>+8.3f}{p:>10.4f}  {sig}")


if __name__ == "__main__":
    part1()
    part2()
    part3()
