#!/usr/bin/env python3
"""
Contraste estadístico A NIVEL DE RED para lo que produce `run_noise_compare.py --replicas` (las 11 semillas
de revalidación de cada método). Aquí la unidad de réplica es la RED ENTRENADA, no el chip: por eso, a
diferencia de stats_noise_compare.py (un modelo por método, chips como réplicas), estos contrastes sí hablan
de MÉTODOS —WANN frente a SGRL— y no solo de modelos concretos.

Por red y celda (dispositivo, mecanismo, escala, tiempo): retención = (retorno medio sobre los chips - piso) /
(retorno sin ruido de ESA red - piso) * 100. Por celda y método SGRL, se compara su vector de 11 retenciones con el
de WANN (11 redes, cada una con su mejor peso compartido):
  - Mann-Whitney U (prueba principal, sin supuestos de normalidad con n = 11), t de Welch como chequeo;
  - efecto: diferencia de medianas y de medias (método - WANN, en puntos de retención), IC 95 % de la diferencia
    de medias por bootstrap sobre redes, y probabilidad de superioridad P(método > WANN) (0,5 = sin diferencia);
  - corrección Benjamini-Hochberg (FDR) sobre todas las celdas × métodos de la tarea.
Efecto de cuantizar: cada red cuantizada frente a SU red en float (misma semilla de entrenamiento), prueba de
Wilcoxon de rangos con signo pareada (n = 11) + BH.

Limitaciones: n = 11 por grupo da poca potencia (el mínimo p exacto de Mann-Whitney con 11 vs 11 es ~1e-6, pero
diferencias moderadas con dispersión alta no se detectan); WANN usa para cada semilla el peso de mejor
revalidación (sesgo de selección dentro de cada red, a favor de WANN); las 11 semillas son de UNA configuración
(el mejor rank/trial de cada familia), no de todas.

Salida: <data-dir>/stats_replicas_vs_wann.csv y stats_replicas_quant.csv, y un resumen en pantalla.
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

CELL = ["noise_model", "mechanism", "scale", "t_inference"]
SELECT_SEED = 999      # semilla reservada para elegir el mejor nivel ODIN de WANN sin ruido (ver run_noise_compare.py)
METHODS = ("sgrl", "sgrl_con", "sgrl_neu", "sgrlq", "sgrlq_con", "sgrlq_neu")
FLOAT_OF = {"sgrlq": "sgrl", "sgrlq_con": "sgrl_con", "sgrlq_neu": "sgrl_neu"}


def network_table(data_dir: Path, wann_view: str = "best"):
    """Una fila por (red, celda): retorno medio sobre chips, retención y retorno sin ruido de la red.

    WANN trae varios pesos compartidos por red: los niveles ODIN enteros 1..7 (weight_value en models.csv; modo
    por defecto) o los de WEIGHT_VALS (is_best_weight). Vista de WANN:
      best  UN peso por semilla, elegido SIN ruido: con niveles ODIN, el de mayor retorno en la celda `none` con la
            semilla reservada 999 (episodios independientes de todas las demás celdas); con WEIGHT_VALS, el de la
            revalidación original. Fijado antes de ver el ruido (def.);
      mean  el promedio de los pesos de cada semilla: robustez de la red sin elegir peso.
    En ambos casos hay UNA fila por red (semilla) y celda: la unidad de réplica es la red."""
    df = pd.read_csv(data_dir / "sweep_all.csv")
    models = pd.read_csv(data_dir / "models.csv")
    sel = df[df.seed == SELECT_SEED].groupby("model_id")["ret"].mean()      # retorno sin ruido para ELEGIR nivel
    df = df[df.seed != SELECT_SEED]                                          # el resto del análisis no la usa
    proto = json.loads((data_dir / "protocol.json").read_text()) if (data_dir / "protocol.json").is_file() else {}
    floor = float(proto.get("ret_floor", 0.0))
    chip = df.groupby(["model_id", "source"] + CELL + ["seed", "chip"], sort=False)["ret"].mean().reset_index()
    net = chip.groupby(["model_id", "source"] + CELL, sort=False)["ret"].agg(["mean", "count"]).reset_index()
    net = net.rename(columns={"mean": "ret_abs", "count": "n_chips"})
    base = net[net.noise_model == "none"].set_index("model_id")["ret_abs"]
    net["base"] = net.model_id.map(base)
    net["ret_pct"] = np.where(net.base > floor, (net.ret_abs - floor) / (net.base - floor) * 100.0, np.nan)
    net = net.merge(models[["model_id"] + [c for c in ("replica_seed", "ref_model_id", "is_best_weight", "weight_value")
                                           if c in models.columns]], on="model_id", how="left")
    is_w = net.source == "wann"
    odin = "weight_value" in net.columns and net.loc[is_w, "weight_value"].notna().any()
    multi = odin or ("is_best_weight" in net.columns and net.loc[is_w, "is_best_weight"].notna().any())
    if multi:
        w = is_w
        if odin and wann_view == "best":
            if sel.empty:
                raise ValueError("no hay celdas `none` con la semilla 999 para elegir el nivel ODIN de WANN")
            pick = (pd.DataFrame({"model_id": sel.index, "sel": sel.values}).merge(models[["model_id", "replica_seed"]])
                    .sort_values("sel", ascending=False).drop_duplicates("replica_seed"))
            chosen = set(pick.model_id)
            net = net[~w | net.model_id.isin(chosen)]
            net.attrs["wann_levels"] = {int(r.replica_seed): float(models.set_index("model_id").weight_value[r.model_id])
                                        for r in pick.itertuples()}
        elif wann_view == "best":
            net = net[~w | (net.is_best_weight == 1)]
        elif wann_view == "mean":
            agg = (net[w].groupby(["replica_seed"] + CELL, sort=False)
                   .agg(ret_abs=("ret_abs", "mean"), base=("base", "mean"), ret_pct=("ret_pct", "mean"),
                        n_chips=("n_chips", "first")).reset_index())
            agg["source"] = "wann"
            agg["model_id"] = agg.replica_seed.map(lambda k: f"wann_s{int(k):02d}_mean")
            net = pd.concat([net[~w], agg], ignore_index=True)
        else:
            raise ValueError("wann_view: best | mean")
    return net[net.noise_model != "none"], net[net.noise_model == "none"], floor


def boot_mean_diff(x, y, rng, n_boot):
    ix = rng.integers(0, len(x), size=(n_boot, len(x)))
    iy = rng.integers(0, len(y), size=(n_boot, len(y)))
    return np.percentile(y[iy].mean(axis=1) - x[ix].mean(axis=1), [2.5, 97.5])


def methods_vs_wann(task, net, a) -> pd.DataFrame:
    rng = np.random.default_rng(a.seed)
    rows = []
    groups = {k: g.sort_values("replica_seed").ret_pct.dropna().to_numpy() for k, g in net.groupby(["source"] + CELL)}
    for cell in net[CELL].drop_duplicates().itertuples(index=False, name=None):
        if cell[2] == 0 and cell[0] in ("pcm", "reram"):
            continue
        x = groups.get(("wann",) + cell)
        if x is None or len(x) < 3:
            continue
        for src in METHODS:
            y = groups.get((src,) + cell)
            if y is None or len(y) < 3:
                continue
            u = stats.mannwhitneyu(y, x, alternative="two-sided")
            pw = stats.ttest_ind(y, x, equal_var=False).pvalue if (x.std() > 0 or y.std() > 0) else 1.0
            lo, hi = boot_mean_diff(x, y, rng, a.boot)
            rows.append(dict(task=task, noise_model=cell[0], mechanism=cell[1], scale=cell[2], t_inference=cell[3],
                             method=src, n_wann=len(x), n_method=len(y), median_wann=np.median(x),
                             median_method=np.median(y), mean_wann=x.mean(), mean_method=y.mean(),
                             diff_median_pp=np.median(y) - np.median(x), diff_mean_pp=y.mean() - x.mean(),
                             ci_lo=lo, ci_hi=hi, prob_superiority=u.statistic / (len(x) * len(y)),
                             p_mwu=u.pvalue, p_welch=pw))
    out = pd.DataFrame(rows)
    if out.empty:
        sys.exit(f"Error [{task}]: ninguna celda con >= 3 redes por método.")
    out["q_bh"] = stats.false_discovery_control(out.p_mwu.fillna(1.0).to_numpy(), method="bh")
    out["sig"] = out.q_bh < a.alpha
    out["relevant"] = out.sig & (out.diff_mean_pp.abs() >= a.min_effect)
    out["direction"] = np.where(~out.sig, "n.s.", np.where(out.diff_mean_pp > 0, "method > WANN", "WANN > method"))
    return out


def quant_effect(task, net, base, a) -> pd.DataFrame | None:
    if "ref_model_id" not in net.columns or not net.source.isin(FLOAT_OF).any():
        return None                       # corrida sin modelos cuantizados
    ref = net[net.source.isin(FLOAT_OF)][["model_id", "source", "ref_model_id"] + CELL + ["ret_pct"]]
    if ref.empty:
        return None
    flt = net.set_index(["model_id"] + CELL)["ret_pct"]
    rows = []
    for key, g in ref.groupby(["source"] + CELL):
        src, cell = key[0], tuple(key[1:])
        q = g.set_index("ref_model_id")["ret_pct"]
        f = pd.Series({m: flt.get((m,) + cell, np.nan) for m in q.index})
        d = (q - f).dropna()
        if len(d) < 3 or (cell[2] == 0 and cell[0] in ("pcm", "reram")):
            continue
        try:
            pw = stats.wilcoxon(d).pvalue if (d != 0).any() else 1.0
        except ValueError:
            pw = 1.0
        rows.append(dict(task=task, noise_model=cell[0], mechanism=cell[1], scale=cell[2], t_inference=cell[3],
                         method=src, n=len(d), median_diff_pp=d.median(), mean_diff_pp=d.mean(),
                         n_worse=int((d < 0).sum()), p_wilcoxon=pw))
    out = pd.DataFrame(rows)
    if out.empty:
        return None
    out["q_bh"] = stats.false_discovery_control(out.p_wilcoxon.to_numpy(), method="bh")
    out["sig"] = out.q_bh < a.alpha
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, nargs="+", default=[Path(f"eval_results/{t}_replicas") for t in ("car", "disc_mc", "acrobot")])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--wann-view", choices=["best", "mean"], default="best",
                    help="best = un peso por red elegido sin ruido (def.); mean = promedio de los pesos de cada red")
    ap.add_argument("--min-effect", type=float, default=5.0, help="|diferencia de medias| mínima (pp) para 'relevante'")
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for d in a.data_dir:
        if not (d / "sweep_all.csv").is_file():
            sys.exit(f"Error: no existe {d / 'sweep_all.csv'}")
        task = d.name.replace("_replicas", "")
        net, base, floor = network_table(d, a.wann_view)
        suf = "" if a.wann_view == "best" else f"_{a.wann_view}"
        out = methods_vs_wann(task, net, a)
        out.to_csv(d / f"stats_replicas_vs_wann{suf}.csv", index=False)
        print(f"\n=== {task} (WANN: {a.wann_view}): {len(out)} contrastes de red (n=11 vs 11), {out.sig.mean() * 100:.0f} % significativos (BH {a.alpha})")
        t = out.groupby(["noise_model", "method"]).apply(lambda g: pd.Series({
            "celdas": len(g), "WANN>método": int((g.direction == "WANN > method").sum()),
            "método>WANN": int((g.direction == "method > WANN").sum()), "n.s.": int((g.direction == "n.s.").sum()),
            "relevantes": int(g.relevant.sum())}), include_groups=False)
        print(t.to_string())
        qe = quant_effect(task, net, base, a)
        if qe is not None:
            if a.wann_view == "best":
                qe.to_csv(d / "stats_replicas_quant.csv", index=False)
            print(f"    efecto de cuantizar (Wilcoxon pareado): {int(qe.sig.sum())}/{len(qe)} celdas significativas")
        print(f"    -> {d / f'stats_replicas_vs_wann{suf}.csv'}")


if __name__ == "__main__":
    main()
