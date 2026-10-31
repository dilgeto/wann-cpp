#!/usr/bin/env python3
"""
Contraste estadístico, celda por celda, entre el modelo principal de WANN y el de cada variante de
SGRL, a partir de lo que consolida run_noise_compare.py (sweep_all.csv + models.csv + protocol.json).
No simula nada.

Qué se compara y con qué unidad de réplica
  - Métrica: la retención de las figuras, (retorno - piso)/(retorno sin ruido de ESE modelo - piso)*100,
    calculada por CHIP (media de las lecturas de un mismo chip: las lecturas de un chip no son
    independientes entre sí). El retorno sin ruido de cada modelo se toma como constante (como en las figuras).
  - Réplicas: los chips ((semilla, chip), 60 por celda con el protocolo por defecto), independientes
    entre modelos y dentro de un modelo.
  - Una "celda" es (dispositivo, mecanismo, escala, tiempo). Se excluyen la celda sin ruido y las de
    escala 0 de PCM/ReRAM (retención 100 % por construcción). La de escala 0 de ReRAM-CMO se mantiene: el
    piso g_min ya perturba los pesos.
  - Diferencia = método - WANN, en puntos porcentuales (igual que la fila 2 de los heatmaps).

Pruebas
  - t de Welch (varianzas distintas) sobre la retención por chip; es la prueba principal.
  - Mann-Whitney como chequeo de robustez (las retenciones por chip pueden ser sesgadas o bimodales).
  - IC 95 % de la diferencia por bootstrap remuestreando chips de cada modelo por separado.
  - Corrección por comparaciones múltiples: Benjamini-Hochberg (FDR) sobre TODAS las celdas × métodos de la
    tarea (q < --alpha). "Relevante" = significativa y |diferencia| >= --min-effect pp (umbral elegido, no
    una regla universal).

Qué NO prueba esto: las inferencias son sobre ESAS redes concretas (un modelo WANN con su peso, un modelo SGRL
por variante) a través de instancias de hardware. No hay réplica de redes entrenadas, así que no permite
afirmar nada sobre WANN o SGRL "como métodos". (SGRL 64×64 trae 5 semillas, pero solo su modelo principal
entra aquí.)

Salida: <data-dir>/stats_vs_wann.csv (todas las celdas) e impresión de un resumen. plot_noise_compare.py lo
usa, si existe, para atenuar en los heatmaps las celdas no significativas.

Uso:
  python eval_results/stats_noise_compare.py                       # las tres tareas
  python eval_results/stats_noise_compare.py --task acrobot --min-effect 3
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

import numpy as np
import pandas as pd
from scipy import stats

TASKS = ("car", "disc_mc", "acrobot")
METHODS = ("sgrl", "sgrl_con", "sgrl_neu", "sgrlq", "sgrlq_con", "sgrlq_neu")
FLOAT_OF = {"sgrlq": "sgrl", "sgrlq_con": "sgrl_con", "sgrlq_neu": "sgrl_neu"}
CELL = ["noise_model", "mechanism", "scale", "t_inference"]
DEVICES = ("pcm", "reram", "reramcmo")


def retention_by_chip(df: pd.DataFrame, models: pd.DataFrame, floor: float) -> pd.DataFrame:
    main = models[models.main == 1][["model_id", "source"]]
    df = df[df.model_id.isin(main.model_id)]
    cm = (df.groupby(["model_id", "source"] + CELL + ["seed", "chip"], sort=False)["ret"].mean().reset_index())
    base = cm[cm.noise_model == "none"].groupby("model_id")["ret"].mean()
    cm["base"] = cm.model_id.map(base)
    cm["ret_pct"] = np.where(cm.base > floor, (cm["ret"] - floor) / (cm.base - floor) * 100.0, np.nan)
    return cm[cm.noise_model != "none"]


def boot_diff_ci(x, y, rng, n_boot):
    ix = rng.integers(0, len(x), size=(n_boot, len(x)))
    iy = rng.integers(0, len(y), size=(n_boot, len(y)))
    d = y[iy].mean(axis=1) - x[ix].mean(axis=1)
    return np.percentile(d, [2.5, 97.5])


def compare_task(task: str, data_dir: Path, a) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "sweep_all.csv")
    models = pd.read_csv(data_dir / "models.csv")
    proto = json.loads((data_dir / "protocol.json").read_text()) if (data_dir / "protocol.json").is_file() else {}
    floor = float(proto.get("ret_floor", 0.0))
    cm = retention_by_chip(df, models, floor)
    groups = {k: g["ret_pct"].to_numpy() for k, g in cm.groupby(["source"] + CELL, sort=False)}
    rng = np.random.default_rng(a.seed)
    rows = []
    cells = cm[CELL].drop_duplicates().itertuples(index=False, name=None)
    for cell in cells:
        nm, mech, scale, t = cell
        if scale == 0 and nm in ("pcm", "reram"):
            continue
        x = groups.get(("wann",) + cell)
        if x is None:
            continue
        x = x[~np.isnan(x)]
        for src in METHODS:
            y = groups.get((src,) + cell)
            if y is None:
                continue
            y = y[~np.isnan(y)]
            if len(x) < a.min_chips or len(y) < a.min_chips:
                continue
            diff = y.mean() - x.mean()
            if np.ptp(x) == 0 and np.ptp(y) == 0:
                pw = pm = 1.0 if diff == 0 else 0.0
            else:
                pw = stats.ttest_ind(y, x, equal_var=False).pvalue
                try:
                    pm = stats.mannwhitneyu(y, x, alternative="two-sided").pvalue
                except ValueError:
                    pm = 1.0
            lo, hi = boot_diff_ci(x, y, rng, a.boot)
            sp = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
            rows.append(dict(task=task, noise_model=nm, mechanism=mech, scale=scale, t_inference=t, method=src,
                             n_wann=len(x), n_method=len(y), mean_wann=x.mean(), mean_method=y.mean(), diff_pp=diff,
                             ci_lo=lo, ci_hi=hi, cohen_d=(diff / sp if sp > 0 else np.nan), p_welch=pw, p_mwu=pm))
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit(f"Error [{task}]: ninguna celda con >= {a.min_chips} chips por modelo; no hay nada que contrastar "
                 f"(¿protocolo con pocos chips? usá --min-chips).")
    out["q_bh"] = stats.false_discovery_control(out.p_welch.fillna(1.0).to_numpy(), method="bh")
    out["q_bh_mwu"] = stats.false_discovery_control(out.p_mwu.fillna(1.0).to_numpy(), method="bh")
    out["sig"] = out.q_bh < a.alpha
    out["relevant"] = out.sig & (out.diff_pp.abs() >= a.min_effect)
    out["direction"] = np.where(~out.sig, "n.s.", np.where(out.diff_pp > 0, "method > WANN", "WANN > method"))
    out.to_csv(data_dir / "stats_vs_wann.csv", index=False)
    return out


def summarize(out: pd.DataFrame, task: str):
    print(f"\n=== {task}: {len(out)} contrastes ({out.noise_model.nunique()} dispositivos × celdas × 3 variantes), "
          f"FDR {out.q_bh.lt(0.05).mean() * 100:.0f} % significativos")
    agree = ((out.q_bh < 0.05) == (out.q_bh_mwu < 0.05)).mean() * 100
    print(f"    acuerdo Welch vs Mann-Whitney en significancia: {agree:.0f} %")
    g = out.groupby(["noise_model", "method"])
    t = g.apply(lambda d: pd.Series({
        "celdas": len(d),
        "WANN>método": int((d.direction == "WANN > method").sum()),
        "método>WANN": int((d.direction == "method > WANN").sum()),
        "n.s.": int((d.direction == "n.s.").sum()),
        "relevantes": int(d.relevant.sum())}), include_groups=False)
    print(t.to_string())


def quant_effect(task: str, data_dir: Path, a) -> pd.DataFrame | None:
    """Efecto de cuantizar: cada modelo cuantizado frente a SU versión en float, celda por celda. En
    noise_eval.py las dos redes tienen las mismas formas de tensor y por tanto el MISMO flujo de números
    aleatorios por (semilla, chip): los chips están apareados, y se usa t pareada + bootstrap de las
    diferencias por chip. Diferencia = cuantizado - float, en puntos de retención (cada uno respecto a su
    propio retorno sin ruido: separa 'cuánto ruido aguanta' de 'cuánto pierde solo por cuantizar')."""
    models = pd.read_csv(data_dir / "models.csv")
    if not set(FLOAT_OF) & set(models.source):
        return None
    df = pd.read_csv(data_dir / "sweep_all.csv")
    proto = json.loads((data_dir / "protocol.json").read_text()) if (data_dir / "protocol.json").is_file() else {}
    cm = retention_by_chip(df, models, float(proto.get("ret_floor", 0.0)))
    cm = cm.set_index(["source"] + CELL + ["seed", "chip"])["ret_pct"]
    rng = np.random.default_rng(a.seed)
    rows = []
    cells = cm.reset_index()[CELL].drop_duplicates().itertuples(index=False, name=None)
    for cell in cells:
        if cell[2] == 0 and cell[0] in ("pcm", "reram"):
            continue
        for q, f in FLOAT_OF.items():
            try:
                xq, xf = cm.loc[(q,) + cell], cm.loc[(f,) + cell]
            except KeyError:
                continue
            d = (xq - xf.reindex(xq.index)).dropna().to_numpy()
            if len(d) < a.min_chips:
                continue
            pv = stats.ttest_1samp(d, 0.0).pvalue if d.std() > 0 else (1.0 if d.mean() == 0 else 0.0)
            idx = rng.integers(0, len(d), size=(a.boot, len(d)))
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            rows.append(dict(task=task, noise_model=cell[0], mechanism=cell[1], scale=cell[2], t_inference=cell[3],
                             method=q, n_chips=len(d), diff_pp=d.mean(), ci_lo=lo, ci_hi=hi, p_paired=pv))
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    out["q_bh"] = stats.false_discovery_control(out.p_paired.fillna(1.0).to_numpy(), method="bh")
    out["sig"] = out.q_bh < a.alpha
    out["relevant"] = out.sig & (out.diff_pp.abs() >= a.min_effect)
    out.to_csv(data_dir / "stats_quant_vs_float.csv", index=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--base-dir", type=Path, default=Path("eval_results"))
    ap.add_argument("--alpha", type=float, default=0.05, help="nivel de FDR (Benjamini-Hochberg)")
    ap.add_argument("--min-effect", type=float, default=5.0, help="|diferencia| mínima (pp) para llamarla relevante")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--min-chips", type=int, default=3, help="chips mínimos por modelo y celda para contrastar")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for task in a.task:
        d = a.base_dir / f"{task}_wann_vs_sgrl"
        if not (d / "sweep_all.csv").is_file():
            sys.exit(f"Error: no existe {d / 'sweep_all.csv'}")
        out = compare_task(task, d, a)
        summarize(out, task)
        print(f"    -> {d / 'stats_vs_wann.csv'}")
        qe = quant_effect(task, d, a)
        if qe is not None:
            print(f"    efecto de cuantizar (pareado, {len(qe)} contrastes): "
                  f"{int(qe.relevant.sum())} celdas con |dif| >= {a.min_effect:g} pp significativas -> {d / 'stats_quant_vs_float.csv'}")


if __name__ == "__main__":
    main()
