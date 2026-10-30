#!/usr/bin/env python3
"""
Compara la robustez a ruido de memristores del mejor modelo WANN de cada tarea
(wann_{car,acrobot,disc_mc}_noise) contra checkpoints de gradientes surrogados
de ../surrogate-gradient-rl (scripts/noise_eval.py), con el MISMO protocolo en
ambos lados, y lo hace todo en paralelo (un solo planificador para las tareas).

Solo evalúa, nunca entrena. Orquesta (1) los barridos, (2) la consolidación en
un CSV normalizado por tarea y (3) las figuras de plot_noise_compare.py.

Métodos que se comparan (todos entrenados SIN ruido)
----------------------------------------------------
  wann      la mejor red WANN de la tarea (con sus 6 pesos compartidos)
  sgrl      SGRL 64×64 de la búsqueda Optuna original (runs_default/, 5 mejores semillas)
  sgrl_con  SGRL con tantas CONEXIONES como la red WANN (runs/ + screening/), solo el mejor modelo
  sgrl_neu  SGRL con tantas NEURONAS como la red WANN (runs_neuronas/ + screening_neuronas/), solo el mejor
Con --sg-quant-bits B se añaden las versiones CUANTIZADAS (PTQ por redondeo de los pesos, esquema `odin`:
B bits de magnitud + signo, ver sgrl/quantize.py) de esos tres modelos principales: sgrlq, sgrlq_con, sgrlq_neu.
Los checkpoints cuantizados son los que escribe ../surrogate-gradient-rl/scripts/quantize_eval.py en
runs_quant/<esquema>/<carpeta de runs>/<corrida>/w<B>/ (primero cuantiza, DESPUÉS se inyecta el ruido; los
sesgos quedan en float, como en el modelo de ruido). WANN no se cuantiza: todas sus sinapsis valen ±w (una
sola magnitud), así que ya es representable con un nivel. Las figuras de este grupo se escriben aparte
(<tarea>_wann_vs_sgrlq<B>_*.png) con la versión sin cuantizar como referencia punteada.
Para las dos variantes, "el mejor" = el rank de la fase 3 con mayor media de validación
(screening*/<tarea>/p3_summary.json) y, dentro de él, la semilla de mayor final_eval
(--sgv-top). --sg-variants controla cuáles se incluyen (def.: con neu; sin valores = ninguna).
Los resultados ya existentes de wann/sgrl en --out-dir se reutilizan (partes reanudables): relanzar
solo calcula los modelos nuevos. Si en el clúster faltan esas partes, el --dry-run lo muestra
(columna "hechos") antes de gastar cómputo.

Modo --replicas: TODAS las semillas de revalidación (11) de la configuración elegida de cada método
---------------------------------------------------------------------------------------------------
El modo normal evalúa un modelo "principal" por método (el mejor, con el sesgo de selección que eso trae).
Con --replicas se evalúan las 11 redes entrenadas de cada método —WANN (cada semilla con SU mejor peso
compartido, el de la revalidación), SGRL 64×64, SGRL por conexiones y por neuronas— y, con --sg-quant-bits,
también todas sus versiones cuantizadas. Con 11 redes por método hay réplica real: se puede contrastar
MÉTODOS (unidad = red) y no solo modelos concretos. Un protocolo reducido hace viable el costo: por defecto
30 chips × 1 lectura × 1 semilla por celda y 27 celdas (intensidad 1/2/3/5, deriva a 1 h/1 día/1 año,
ablación a escala 3; --replica-cells full usa las 191 celdas de siempre). De WANN se evalúan los 6 pesos
compartidos de cada red. Por defecto (--replica-wann-weights odin) son los niveles ENTEROS 1..7, los únicos pesos
que ODIN puede guardar (el signo lo da la sinapsis excitadora/inhibidora; peso simulado = nivel, w_max = 7, igual
que wann_car_odin_finetune), y NO los de WEIGHT_VALS (0.5, 1.5, 2.5 y 8 no son representables). El análisis
principal usa, por semilla, el nivel de mejor desempeño SIN ruido medido con episodios independientes (celda
`none` con la semilla 999, que no se usa en ninguna otra celda); se puede ver también el promedio de los niveles
(--wann-view mean). Salida aparte en
eval_results/<tarea>_replicas/ (otro protocolo: no se mezcla con lo anterior) y sus propias figuras y
contrastes (plot_noise_replicas.py, stats_noise_replicas.py). Uso:
  python eval_results/run_noise_compare.py --replicas --sg-quant-bits 3 --jobs 300

Tareas y modelos (se eligen solos; ver --dry-run)
-------------------------------------------------
  car      WANN: mejor modelo de car_ttfs_first_spike_discrete      SGRL: carrace-f2
  disc_mc  WANN: mejor de las búsquedas disc_mc_*_discrete          SGRL: mountaincar-f2
           ("Mountain Car" = el discreto, ver CLAUDE.md)
  acrobot  WANN: mejor de las búsquedas acrobot_* (la original)     SGRL: acrobot-f2
"Mejor" = mayor recompensa revalidada (la columna `reward` de los *_best.csv de
eval_p3_weights.py), la misma regla que usa el bootstrap (pick_winner). En
SGRL: el trial con mayor media de validación y sus --sg-top mejores semillas.

Qué se evalúa (todo en una sola pasada, por tarea)
--------------------------------------------------
  intensidad   --noise-scales a t fijo (pcm 0 s, reram 1 s, reramcmo 1 s)
  tiempo       t desde la programación, a escala 1.0
  grilla       escala × tiempo (heatmaps)
  ablación     un mecanismo activo cada vez (programación / lectura / deriva)
               en pcm y reramcmo; se omite con --no-ablation
  base         --noise-model none (referencia sin ruido para la retención)
Las celdas repetidas entre estos barridos se calculan una sola vez.

Protocolo común (lo que hace comparables ambos lados)
-----------------------------------------------------
  - mismo entorno y mismos modelos de ruido (el puerto de SGRL está validado
    contra memristor_noise.hpp); las tres tareas de SGRL son las de wann-cpp;
  - mismo generador por (semilla, chip): programar un chip, luego
    --reads-per-chip lecturas con ruido fresco;
  - cada fila promedia --episodes episodios (por defecto alg_nReps del config);
  - recompensa ORIGINAL del entorno (SGRL nunca usa shaping al evaluar).
  Diferencias intrínsecas que no se pueden igualar: en WANN cada sinapsis vale
  ±(peso compartido) y w_max es el mayor peso de la tarea (max WEIGHT_VALS),
  mientras que SGRL tiene pesos entrenados con w_max = max|W| por capa; el ruido
  relativo cae distinto sobre cada red.

Paralelismo
-----------
Cada trabajo es UNA celda (modelo, escala, t, mecanismo) para UNA semilla de
UN modelo, de unas 90 filas con los valores por defecto (≲1 min WANN, ≲5 min
SGRL): suficientemente fino para llenar --jobs hebras y suficientemente grueso
para que el arranque de Python/torch (~3 s) sea <2 %. Los trabajos más caros
(SGRL) salen primero. --jobs-sgrl limita cuántos procesos Python corren a la
vez: cada uno carga torch (del orden de 0.5 GB de RAM; ver --dry-run). Las
estimaciones de tiempo son cotas superiores (se midieron en car, la tarea con
episodios más largos).

Reanudable: cada trabajo escribe <out-dir>/parts/<modelo>/<celda>.csv de forma
atómica; al relanzar se saltan los que ya existen (--rerun los rehace). Un
trabajo que falla no aborta el resto: se registra en failed_jobs.txt y, tras
consolidar lo hecho, el script sale con código 1 sin graficar (--plot-anyway).

Uso:
  python eval_results/run_noise_compare.py --dry-run
  python eval_results/run_noise_compare.py --jobs 300                      # las 3 tareas, 4 métodos
  python eval_results/run_noise_compare.py --task acrobot --sg-top 1 --wann-weights 1
  python eval_results/run_noise_compare.py --skip-run                      # solo consolidar y graficar
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Ancla CWD a la raíz del repo — mismo patrón que run_car_noise_sweeps.py.
REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

REAL_MODELS = ["pcm", "reram", "reramcmo"]
# t por defecto de cada modelo (coincide con wann_*_noise y con SGRL NoiseConfig.t)
INTENSITY_T = {"simple": 0.0, "pcm": 0.0, "reram": 1.0, "reramcmo": 1.0}
ABLATION_MECHS = ("prog", "read", "drift")
MECH_FLAG = {"prog": "--prog-noise-scale", "read": "--read-noise-scale", "drift": "--drift-scale"}
# Costo aproximado por fila (s de CPU, una hebra); solo para ordenar y estimar. Medido: WANN en car;
# SGRL 64x64 en car (16 pasos de simulación por paso de entorno); las variantes de car son ~8x más caras
# por tener 43 pasos de simulación; en acrobot y Mountain Car el costo por fila es mínimo (~0.2-0.3 s).
COST_PER_ROW = {"wann": 0.8, "sgrl": 3.2}


def row_cost(kind: str, source: str, task: str) -> float:
    if kind == "wann":
        return COST_PER_ROW["wann"]
    if task == "car":
        return COST_PER_ROW["sgrl"] if source in ("sgrl", "sgrlq") else 7.5
    return 0.3

# ret_floor: recompensa del peor desenlace (fracaso total); la "retención" se mide entre este
# piso y el retorno sin ruido del propio modelo. car: sin piso negativo (0 ⇒ retorno/base);
# acrobot: -1 por paso hasta 500; Mountain Car discreto: -1 por paso hasta 200.
TASKS = {
    "car": dict(
        label="CarTrack", binary="./build/wann_car_noise", base_config="p/car_snn.json",
        sg_study="carrace-f2", sg_task="car", ret_floor=0.0, graficos="Racing Car", prefix="car_wann_vs_sgrl",
        eval_csvs=["eval_results/car_ttfs_first_spike_discrete/car_best.csv"]),
    "disc_mc": dict(
        label="Mountain Car (discrete)", binary="./build/wann_disc_mc_noise", base_config="p/disc_mc_snn.json",
        sg_study="mountaincar-f2", sg_task="disc_mc", ret_floor=-200.0, graficos="Mountain Car", prefix="disc_mc_wann_vs_sgrl",
        eval_csvs=sorted(glob.glob("eval_results/disc_mc_*_discrete/mountain_car_best.csv"))),
    "acrobot": dict(
        label="Acrobot", binary="./build/wann_acrobot_noise", base_config="p/acrobot_snn.json",
        sg_study="acrobot-f2", sg_task="acrobot", ret_floor=-500.0, graficos="Acrobot", prefix="acrobot_wann_vs_sgrl",
        eval_csvs=["eval_results/eval_p3_weights/acrobot_best.csv"]),
}

# Variantes de SGRL con la red del tamaño de la de WANN: carpeta de corridas y de screening (relativas a --sg-repo).
VARIANTS = {
    "con": dict(runs="runs", screening="screening", what="mismas conexiones"),
    "neu": dict(runs="runs_neuronas", screening="screening_neuronas", what="mismas neuronas"),
}


def kind_of(source: str) -> str:
    """'wann' o 'sgrl' (cualquier variante de SGRL): determina costo, ejecutable y tope --jobs-sgrl."""
    return "sgrl" if source.startswith("sgrl") else "wann"


NORMALIZED_COLS = ["source", "model_id", "noise_model", "mechanism", "scale", "t_inference", "seed",
                   "chip", "read", "weight_rmse", "ret", "prog_noise_scale", "read_noise_scale",
                   "drift_scale"]


def fmt(x: float) -> str:
    """Número para la línea de comandos y los nombres de archivo (sin notación científica)."""
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def csv_floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


# --------------------------------------------------------------------------------------------
# Celdas del experimento
# --------------------------------------------------------------------------------------------
def replica_cells(a) -> list[tuple]:
    """Celdas del protocolo reducido de --replicas (27): sin ruido; intensidad a escalas 1/2/3/5 por dispositivo
    (t por defecto); deriva a escala 1 (PCM y CMO a 1 h/1 día/1 año, ReRAM a 1 día); ablación a escala 3
    (PCM a ablation_t, CMO a 1 s) y la escala 0 de CMO (efecto del piso g_min)."""
    cells = {("none", 0.0, 0.0, "all"), ("reramcmo", 0.0, 1.0, "all")}
    for m in REAL_MODELS:
        for sc in (1.0, 2.0, 3.0, 5.0):
            cells.add((m, sc, INTENSITY_T[m], "all"))
    for m in ("pcm", "reramcmo"):
        for t in (3600.0, 86400.0, 31536000.0):
            cells.add((m, 1.0, t, "all"))
    cells.add(("reram", 1.0, 86400.0, "all"))
    for m, t in (("pcm", a.ablation_t), ("reramcmo", 1.0)):
        for mech in ABLATION_MECHS:
            cells.add((m, 3.0, t, mech))
    return sorted(cells)


def build_cells(a) -> list[tuple]:
    if a.replicas and a.replica_cells == "reduced":
        return replica_cells(a)
    """Celdas únicas (noise_model, scale, t, mechanism). mechanism='all' = los tres mecanismos
    siguen a `scale`; 'prog'/'read'/'drift' = solo ese vale `scale` y los otros dos 0."""
    cells = {("none", 0.0, 0.0, "all")}
    models = REAL_MODELS + (["simple"] if a.with_simple else [])
    times = {"pcm": a.time_points_pcm, "reram": a.time_points_reram, "reramcmo": a.time_points_reramcmo}
    for m in models:
        for s in a.noise_scales:
            cells.add((m, s, INTENSITY_T[m], "all"))           # intensidad
    for m in REAL_MODELS:
        for s in a.noise_scales:
            for t in times[m]:
                cells.add((m, s, t, "all"))                    # grilla escala×tiempo
        for t in times[m]:
            cells.add((m, 1.0, t, "all"))                      # tiempo a escala 1.0
    if not a.no_ablation:
        for m, t in (("pcm", a.ablation_t), ("reramcmo", 1.0)):
            cells.add((m, 0.0, t, "all"))                      # punto s=0 de las tres curvas
            for mech in ABLATION_MECHS:
                for s in a.noise_scales:
                    if s > 0:
                        cells.add((m, s, t, mech))
    return sorted(cells)


def scale_flags(cell, noise_scale_flag: str) -> list[str]:
    """Flags de escala; son los mismos nombres y semántica en wann_*_noise y noise_eval.py."""
    model, s, _t, mech = cell
    if model == "none":
        return []
    if mech == "all":
        return [noise_scale_flag, fmt(s)]
    flags = []
    for k, flag in MECH_FLAG.items():
        flags += [flag, fmt(s if k == mech else 0.0)]
    return flags


# --------------------------------------------------------------------------------------------
# Selección de modelos
# --------------------------------------------------------------------------------------------
def read_alg_nreps(base_cfg: Path, override: str | None) -> int | None:
    n = None
    for p in (base_cfg, Path(override) if override else None):
        if p and p.is_file():
            try:
                n = json.loads(p.read_text()).get("alg_nReps", n)
            except json.JSONDecodeError:
                pass
    return int(n) if n else None


def pick_wann_best(task: str, spec: dict, a) -> dict:
    """Mejor (run_key, rank, seed) por recompensa revalidada, entre los *_best.csv de la tarea."""
    if a.wann_net:    # selección manual (solo con una tarea)
        return dict(net=a.wann_net, override=a.wann_override, run_key="(manual)", rank=-1, seed_idx=-1,
                    weight_index=a.wann_winner_weight, reward=float("nan"), reward_std=float("nan"))
    csvs = [c for c in spec["eval_csvs"] if Path(c).is_file()]
    if not csvs:
        sys.exit(f"Error [{task}]: no hay CSV de revalidación ({spec['eval_csvs']}). "
                 f"Corré bootstrap_results/*_auto.py o eval_p3_weights.py, o pasá --wann-net.")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    if "reward_type" in df.columns and (df.reward_type == "original").any():
        df = df[df.reward_type == "original"]          # la recompensa comparable con SGRL
    if a.wann_run_key:
        df = df[df.run_key.str.contains(a.wann_run_key)]
        if df.empty:
            sys.exit(f"Error [{task}]: ningún run_key coincide con {a.wann_run_key!r}.")
    best = df.sort_values("reward", ascending=False).iloc[0]
    rk, r, s = best.run_key, int(best["rank"]), int(best.seed_idx)
    net = f"log/full_p3_{rk}/rank{r:02d}_seed{s:02d}_best.out"
    override = f"screening_full/{rk}/p3_configs/rank{r:02d}_seed{s:02d}.json"
    for p in (net, override):
        if not Path(p).is_file():
            sys.exit(f"Error [{task}]: no existe {p}")
    return dict(net=net, override=override, run_key=rk, rank=r, seed_idx=s,
                weight_index=int(best.weight_index), reward=float(best.reward),
                reward_std=float(best.reward_std))


def hidden_of(run_dir: Path) -> str:
    """Capas ocultas del actor (p. ej. '64x64') leídas del config.json de la corrida."""
    try:
        return "x".join(str(h) for h in json.loads((run_dir / "config.json").read_text())["args"]["hidden"])
    except (OSError, KeyError, json.JSONDecodeError):
        return "?"


def select_sgrl(task: str, spec: dict, a) -> list[dict]:
    repo = Path(a.sg_repo).resolve()
    default_dir = a.sg_default_dir or ("runs_default" if (repo / "runs_default").is_dir() else "runs")
    runs_dir = repo / default_dir
    ckpt_name = f"{a.sg_ckpt}.pt"
    study = a.sg_study or spec["sg_study"]
    if a.sg_runs:
        dirs = [Path(d).resolve() for d in a.sg_runs]
    else:
        val_path = runs_dir / f"{study}-validation.json"
        if not val_path.is_file():
            sys.exit(f"Error [{task}]: no existe {val_path}. Pasá --sg-study o --sg-runs.")
        val = json.loads(val_path.read_text())
        trial = a.sg_trial if a.sg_trial != "auto" else max(val, key=lambda k: val[k]["mean"])
        if trial not in val:
            sys.exit(f"Error [{task}]: el trial {trial} no está en {val_path.name} (hay: {sorted(val, key=int)}).")
        scored = []
        for d in sorted(runs_dir.glob(f"{study}-t{trial}-s*")):
            fe = d / "final_eval.json"
            if fe.is_file():
                scored.append((json.loads(fe.read_text())["return_mean"], d))
        if not scored:
            sys.exit(f"Error [{task}]: sin corridas con final_eval.json para {study}-t{trial} en {runs_dir}.")
        scored.sort(key=lambda x: -x[0])
        dirs = [d for _, d in scored[:a.sg_top]]
    out = []
    for d in dirs:
        ck = d / ckpt_name
        if not ck.is_file():
            sys.exit(f"Error [{task}]: no existe {ck}")
        fe = d / "final_eval.json"
        ret = json.loads(fe.read_text())["return_mean"] if fe.is_file() else float("nan")
        out.append(dict(model_id=f"sgrl_{d.name}", source="sgrl", label=f"SGRL {d.name}", path=str(ck),
                        train_return=ret, weight_index="", hidden=hidden_of(d)))
    return out


def select_sgrl_variant(task: str, spec: dict, a, variant: str) -> list[dict]:
    """Mejor(es) modelo(s) de una variante de SGRL con la red del tamaño de la de WANN: el rank de la
    fase 3 con mayor media de validación y, dentro de él, las semillas de mayor final_eval."""
    v = VARIANTS[variant]
    repo = Path(a.sg_repo).resolve()
    sg_task = spec["sg_task"]
    summ = repo / getattr(a, f"sg_{variant}_screening") / sg_task / "p3_summary.json"
    runs = repo / getattr(a, f"sg_{variant}_runs")
    if not summ.is_file():
        sys.exit(f"Error [{task}]: no existe {summ} (variante {variant}: {v['what']}). Revisá --sg-{variant}-screening.")
    val = json.loads(summ.read_text())
    rank = max(val, key=lambda k: val[k]["mean"])
    scored = []
    for d in sorted(runs.glob(f"{sg_task}-p3-rank{rank}-s*")):
        fe = d / "final_eval.json"
        if fe.is_file() and (d / f"{a.sg_ckpt}.pt").is_file():
            scored.append((json.loads(fe.read_text())["return_mean"], d))
    if not scored:
        sys.exit(f"Error [{task}]: sin corridas {sg_task}-p3-rank{rank}-s* con {a.sg_ckpt}.pt en {runs}.")
    scored.sort(key=lambda x: -x[0])
    out = []
    for ret, d in scored[:a.sgv_top]:
        out.append(dict(model_id=f"sgrl_{variant}_{d.name}", source=f"sgrl_{variant}",
                        label=f"SGRL {v['what']} {d.name}", path=str(d / f"{a.sg_ckpt}.pt"),
                        train_return=ret, weight_index="", hidden=hidden_of(d),
                        rank=int(rank), val_mean=float(val[rank]["mean"]), val_std=float(val[rank].get("std", float("nan")))))
    return out


QUANT_OF = {"sgrl": "sgrlq", "sgrl_con": "sgrlq_con", "sgrl_neu": "sgrlq_neu"}


def replica_seed_of(name: str) -> int | None:
    import re
    m = re.search(r"-s(\d+)$", name)
    return int(m.group(1)) if m else None


def select_wann_replicas(task: str, spec: dict, a, best: dict) -> list[dict]:
    """Las 11 semillas de revalidación de la configuración (run_key, rank) ganadora; cada red con SU mejor peso."""
    if best["rank"] < 0:
        sys.exit("Error: --replicas necesita que la red WANN se elija sola (sin --wann-net).")
    df = pd.concat([pd.read_csv(c) for c in spec["eval_csvs"] if Path(c).is_file()], ignore_index=True)
    if "reward_type" in df.columns and (df.reward_type == "original").any():
        df = df[df.reward_type == "original"]
    df = df[(df.run_key == best["run_key"]) & (df["rank"] == best["rank"])].sort_values("seed_idx")
    n_w = int(json.loads(Path(spec["base_config"]).read_text()).get("alg_nVals", 6))
    levels = [int(x) for x in a.odin_levels.split(",") if x.strip()]
    out = []
    for r in df.itertuples():
        net = f"log/full_p3_{r.run_key}/rank{int(r.rank):02d}_seed{int(r.seed_idx):02d}_best.out"
        ov = f"screening_full/{r.run_key}/p3_configs/rank{int(r.rank):02d}_seed{int(r.seed_idx):02d}.json"
        for f in (net, ov):
            if not Path(f).is_file():
                sys.exit(f"Error [{task}]: no existe {f}")
        # el peso "mejor" es el de la revalidación SIN ruido (fijado antes de ver el ruido); con
        # --replica-wann-weights all se evalúan además los demás pesos compartidos de cada red
        if a.replica_wann_weights == "odin":
            for L in levels:
                out.append(dict(model_id=f"wann_s{int(r.seed_idx):02d}_L{L}", source="wann",
                                label=f"WANN {r.run_key} rank{int(r.rank):02d}-seed{int(r.seed_idx):02d} (ODIN level {L})",
                                path=net, net=net, override=ov, weight_value=float(L), w_max=a.wann_wmax,
                                train_return=float(r.reward), replica_seed=int(r.seed_idx)))
            continue
        for wi in (range(n_w) if a.replica_wann_weights == "original" else [int(r.weight_index)]):
            out.append(dict(model_id=f"wann_s{int(r.seed_idx):02d}_w{wi}", source="wann",
                            label=f"WANN {r.run_key} rank{int(r.rank):02d}-seed{int(r.seed_idx):02d} (w idx {wi})",
                            path=net, net=net, override=ov, weight_index=wi,
                            is_best_weight=int(wi == int(r.weight_index)),
                            train_return=float(r.reward), replica_seed=int(r.seed_idx)))
    return out


def select_sgrl_quant(task: str, a, models: list[dict], all_models: bool = False) -> list[dict]:
    """Versión cuantizada del modelo principal (el primero) de cada fuente SGRL ya elegida; con all_models, de TODOS."""
    repo = Path(a.sg_repo).resolve()
    out = []
    bases = [(src, qsrc, base) for src, qsrc in QUANT_OF.items()
             for base in ([b for b in models if b["source"] == src] if all_models
                          else [next((b for b in models if b["source"] == src), None)]) if base is not None]
    for src, qsrc, base in bases:
        ck = Path(base["path"])
        run, runs_dir = ck.parent.name, ck.parent.parent.name        # <carpeta de runs>/<corrida>/final.pt
        qdir = repo / a.sg_quant_dir / a.sg_quant_scheme / runs_dir / run / f"w{a.sg_quant_bits}"
        for f in (f"{a.sg_ckpt}.pt", "config.json"):
            if not (qdir / f).is_file():
                sys.exit(f"Error [{task}]: falta {qdir / f}. Generalo con scripts/quantize_eval.py de {repo.name} "
                         f"(--scheme {a.sg_quant_scheme} --bits {a.sg_quant_bits}) o revisá --sg-quant-dir.")
        out.append(dict(model_id=f"{qsrc}_w{a.sg_quant_bits}_{run}", source=qsrc,
                        label=f"{base['label']} ({a.sg_quant_bits}-bit {a.sg_quant_scheme})",
                        path=str(qdir / f"{a.sg_ckpt}.pt"), train_return=float("nan"), weight_index="",
                        hidden=base.get("hidden", ""), quant_bits=a.sg_quant_bits, quant_scheme=a.sg_quant_scheme,
                        ref_model_id=base["model_id"], replica_seed=base.get("replica_seed")))
    return out


class TaskCtx:
    """Todo lo que depende de la tarea: modelos elegidos, rutas de salida, binario, episodios."""

    def __init__(self, task: str, a):
        self.name, self.spec = task, TASKS[task]
        spec = self.spec
        self.wann = pick_wann_best(task, spec, a)
        self.binary = spec["binary"]
        self.base_config = spec["base_config"]
        if not Path(self.binary).is_file() and not a.skip_run:
            sys.exit(f"Error [{task}]: no existe {self.binary}. Compilar primero: ./build.sh")
        self.episodes = a.episodes or read_alg_nreps(Path(self.base_config), self.wann["override"]) or 6
        suffix = "replicas" if a.replicas else "wann_vs_sgrl"
        self.out_dir = Path(a.out_dir) / f"{task}_{suffix}" if a.out_dir else Path("eval_results") / f"{task}_{suffix}"
        self.graficos = a.graficos_dir or f"graficos/{spec['graficos']}"
        self.prefix = spec["prefix"]
        w = self.wann
        wlabel = f"WANN {w['run_key']} rank{w['rank']:02d}-seed{w['seed_idx']:02d}" if w["rank"] >= 0 else "WANN (manual)"
        if a.replicas:
            a.sg_top = a.sgv_top = 999            # todas las semillas del trial/rank elegido
        if a.replicas:
            self.models = select_wann_replicas(task, spec, a, w) + select_sgrl(task, spec, a)
        else:
            self.models = [dict(model_id=f"wann_w{wi}", source="wann", label=f"{wlabel} (w idx {wi})",
                                path=w["net"], train_return=w["reward"], weight_index=wi)
                           for wi in a.wann_weights] + select_sgrl(task, spec, a)
        for variant in a.sg_variants:
            self.models += select_sgrl_variant(task, spec, a, variant)
        if a.replicas:   # réplica = el nº de semilla de entrenamiento de cada red SGRL
            for m in self.models:
                if m["source"] != "wann":
                    m["replica_seed"] = replica_seed_of(Path(m["path"]).parent.name)
        if a.sg_quant_bits:
            self.models += select_sgrl_quant(task, a, self.models, all_models=a.replicas)

    def describe(self):
        w = self.wann
        print(f"[{self.name}] WANN: {w['run_key']} rank{w['rank']:02d} seed{w['seed_idx']:02d}, peso idx "
              f"{w['weight_index']} (reward revalidado {w['reward']:.1f} ± {w['reward_std']:.1f}); "
              f"{self.episodes} episodios por fila.")
        for src in sorted({m["source"] for m in self.models if m["source"] != "wann"}):
            sg = [m for m in self.models if m["source"] == src]
            extra = (f", rank {sg[0]['rank']} (validación {sg[0]['val_mean']:.1f})" if "rank" in sg[0]
                     else f", {sg[0]['quant_bits']}-bit {sg[0]['quant_scheme']}" if "quant_bits" in sg[0] else "")
            names = [Path(m["path"]).parent.name for m in sg]
            shown = ", ".join(names) if len(names) <= 5 else f"{names[0]} … {names[-1]}"
            print(f"    {src:9s} {len(sg)} checkpoint(s), oculta {sg[0]['hidden']}{extra}: {shown}")


# --------------------------------------------------------------------------------------------
# Trabajos
# --------------------------------------------------------------------------------------------
class Job:
    def __init__(self, ctx, model, cell, seed, cmd, part, rows):
        self.ctx, self.source, self.model, self.cell, self.seed = ctx, model["source"], model, cell, seed
        self.kind = kind_of(self.source)
        self.cmd, self.part, self.rows = cmd, part, rows
        self.tmp = part.with_suffix(".csv.tmp")
        self.err = part.with_suffix(".err")
        self.cost = rows * row_cost(self.kind, self.source, ctx.name) + (3.0 if self.kind == "sgrl" else 0.2)  # + arranque

    @property
    def label(self):
        m, s, t, mech = self.cell
        return f"[{self.ctx.name}] {self.model['model_id']} {m}/{mech} scale={fmt(s)} t={fmt(t)} seed={self.seed}"


SELECT_SEED = 999      # semilla de chips/episodios SOLO para elegir el mejor nivel de WANN sin ruido (celda `none`)


def seeds_for(a, model, cell) -> list[int]:
    if a.replicas and a.replica_wann_weights == "odin" and model["source"] == "wann" and cell[0] == "none":
        return list(a.seeds) + [SELECT_SEED]
    return list(a.seeds)


def cell_stem(cell, seed) -> str:
    m, s, t, mech = cell
    return f"{m}_{mech}_s{fmt(s)}_t{fmt(t)}_seed{seed}"


def wann_cmd(a, ctx, model, cell, seed) -> list[str]:
    m, s, t, _ = cell
    w = ctx.wann
    net, override = model.get("net", w["net"]), model.get("override", w["override"])
    cmd = [ctx.binary, "-f", net, "-d", ctx.base_config]
    if override:
        cmd += ["-p", override]
    weight = ["--weight", fmt(model["weight_value"]), "--w-max", fmt(model["w_max"])] if "weight_value" in model \
        and pd.notna(model.get("weight_value")) else ["--weight-index", str(model["weight_index"])]
    cmd += weight + ["--mode", "per-synapse", "--noise-model", m,
            "--n-chips", str(a.n_chips), "--reads-per-chip", str(a.reads_per_chip),
            "--t-inference", fmt(t), "--n-levels", str(a.n_levels),
            "--reramcmo-g-max", fmt(a.reramcmo_g_max), "--reramcmo-g-min", fmt(a.reramcmo_g_min),
            "--seeds", str(seed)]
    return cmd + scale_flags(cell, "--noise-scale")


def sgrl_cmd(a, ctx, model, cell, seed, out_path: Path) -> list[str]:
    m, s, t, _ = cell
    cmd = [a.sg_python, str(Path(a.sg_repo).resolve() / "scripts" / "noise_eval.py"), model["path"],
           "--seeds", str(seed), "--n-chips", str(a.n_chips), "--reads-per-chip", str(a.reads_per_chip),
           "--episodes", str(ctx.episodes), "--device", a.sg_device, "--noise-model", m,
           "--t-inference", fmt(t), "--n-levels", str(a.n_levels),
           "--reramcmo-g-max", fmt(a.reramcmo_g_max), "--reramcmo-g-min", fmt(a.reramcmo_g_min),
           "--out", str(out_path)]
    return cmd + scale_flags(cell, "--noise-scales")


def build_jobs(a, ctx: TaskCtx, cells: list[tuple]) -> list[Job]:
    jobs = []
    rows = a.n_chips * a.reads_per_chip
    for model in ctx.models:
        mdir = ctx.out_dir / "parts" / model["model_id"]
        for cell in cells:
            for seed in seeds_for(a, model, cell):
                part = mdir / f"{cell_stem(cell, seed)}.csv"
                if model["source"] == "wann":
                    cmd = wann_cmd(a, ctx, model, cell, seed)
                else:
                    cmd = sgrl_cmd(a, ctx, model, cell, seed, part.with_suffix(".csv.tmp"))
                jobs.append(Job(ctx, model, cell, seed, cmd, part, rows))
    return jobs


# --------------------------------------------------------------------------------------------
# Planificador: hasta --jobs procesos a la vez, como mucho --jobs-sgrl de ellos de SGRL
# --------------------------------------------------------------------------------------------
def child_env() -> dict:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"   # una hebra por proceso: el paralelismo es entre procesos
    return env


def launch(job: Job, env) -> subprocess.Popen:
    job.part.parent.mkdir(parents=True, exist_ok=True)
    err = open(job.err, "w")
    if job.source == "wann":
        out = open(job.tmp, "w")           # wann_*_noise escribe el CSV por stdout
    else:
        out = subprocess.DEVNULL           # noise_eval.py escribe su --out (job.tmp)
    try:
        p = subprocess.Popen(job.cmd, stdout=out, stderr=err, env=env)
    finally:
        err.close()                        # el hijo conserva su copia: así no se acumulan descriptores
        if out is not subprocess.DEVNULL:
            out.close()
    return p


def finish_ok(job: Job) -> str | None:
    """Valida y publica la parte. Devuelve None si está bien o el motivo del fallo."""
    if not job.tmp.is_file():
        return "no produjo CSV"
    n = sum(1 for _ in open(job.tmp)) - 1
    if n != job.rows:
        return f"filas inesperadas: {n} (esperadas {job.rows})"
    os.replace(job.tmp, job.part)
    job.err.unlink(missing_ok=True)
    return None


def run_jobs(jobs: list[Job], a) -> list[tuple[Job, str]]:
    todo = [j for j in jobs if a.rerun or not j.part.is_file()]
    skipped = len(jobs) - len(todo)
    for j in todo:
        j.tmp.unlink(missing_ok=True)
    todo.sort(key=lambda j: -j.cost)       # lo más caro primero: reduce la cola final
    print(f"Trabajos: {len(jobs)} en total, {skipped} ya hechos, {len(todo)} por correr "
          f"(hasta {a.jobs} a la vez, {a.jobs_sgrl} de SGRL).")
    env, running, failures = child_env(), {}, []
    n_sgrl, done, t0, last_print = 0, 0, time.time(), 0.0

    def stop_all(*_):
        for p in running:
            p.terminate()
        sys.exit("\nInterrumpido: se terminaron los procesos hijos; lo ya terminado queda en parts/ "
                 "(relanzá el mismo comando para reanudar).")
    signal.signal(signal.SIGTERM, stop_all)

    try:
        while todo or running:
            while todo and len(running) < a.jobs:
                idx = next((i for i, j in enumerate(todo) if j.kind != "sgrl" or n_sgrl < a.jobs_sgrl), None)
                if idx is None:
                    break
                job = todo.pop(idx)
                running[launch(job, env)] = (job, time.time())
                n_sgrl += job.kind == "sgrl"
            time.sleep(0.25)
            for p in list(running):
                job, start = running[p]
                rc = p.poll()
                if rc is None and time.time() - start > a.timeout:
                    p.kill()
                    p.wait()
                    rc, why = -9, f"timeout ({a.timeout} s)"
                elif rc is None:
                    continue
                else:
                    why = None
                del running[p]
                n_sgrl -= job.kind == "sgrl"
                done += 1
                if why is None and rc == 0:
                    why = finish_ok(job)
                elif why is None:
                    tail = job.err.read_text()[-400:].strip() if job.err.is_file() else ""
                    why = f"código {rc}: {tail}"
                if why:
                    failures.append((job, why))
                    print(f"  [FALLÓ] {job.label}: {why}", flush=True)
            if time.time() - last_print > 30:
                last_print = time.time()
                print(f"  {done}/{done + len(todo) + len(running)} hechos, {len(running)} corriendo, "
                      f"{len(failures)} fallidos, {(last_print - t0) / 60:.1f} min", flush=True)
    except KeyboardInterrupt:
        for p in running:
            p.terminate()
        sys.exit("\nInterrumpido (Ctrl-C): lo ya terminado queda en parts/; relanzá para reanudar.")
    print(f"Listo en {(time.time() - t0) / 60:.1f} min; {len(failures)} fallidos.")
    return failures


# --------------------------------------------------------------------------------------------
# Consolidación
# --------------------------------------------------------------------------------------------
def load_part(job: Job) -> pd.DataFrame:
    df = pd.read_csv(job.part)
    ret = df["reward_original"] if job.source == "wann" else df["return_mean"]
    m, s, t, mech = job.cell
    out = pd.DataFrame({
        "source": job.source, "model_id": job.model["model_id"], "noise_model": m, "mechanism": mech,
        "scale": s, "t_inference": df["t_inference"], "seed": df["seed"], "chip": df["chip"],
        "read": df["read"], "weight_rmse": df["weight_rmse"], "ret": ret,
        "prog_noise_scale": df["prog_noise_scale"], "read_noise_scale": df["read_noise_scale"],
        "drift_scale": df["drift_scale"]})
    return out[NORMALIZED_COLS]


def consolidate(ctx: TaskCtx, jobs: list[Job], a) -> int:
    mine = [j for j in jobs if j.ctx is ctx]
    frames = [load_part(j) for j in mine if j.part.is_file()]
    missing = sum(1 for j in mine if not j.part.is_file())
    if not frames:
        print(f"[{ctx.name}] nada que consolidar.")
        return missing
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(ctx.out_dir / "sweep_all.csv", index=False)
    mf = pd.DataFrame(ctx.models)
    # principal: WANN = el peso ganador; SGRL = la mejor semilla (la primera seleccionada)
    mf["main"] = 0
    wm = mf[mf.source == "wann"]
    pick = wm[wm.weight_index == ctx.wann["weight_index"]]
    if len(wm):
        mf.loc[(pick.index[0] if len(pick) else wm.index[0]), "main"] = 1
    if a.replicas:
        mf["main"] = 1                  # réplicas: no hay modelo principal, todas las redes cuentan
    for src in ([] if a.replicas else mf.source.unique()):   # cada variante de SGRL: su primer modelo es el principal
        if src != "wann":
            mf.loc[mf[mf.source == src].index[0], "main"] = 1
    mf.to_csv(ctx.out_dir / "models.csv", index=False)
    print(f"[{ctx.name}] consolidado: {ctx.out_dir / 'sweep_all.csv'} ({len(df)} filas, {len(frames)} partes, "
          f"{missing} faltantes) y models.csv")
    return missing


def input_files(ctx: TaskCtx, a) -> list[tuple[str, Path]]:
    """Todos los archivos que deciden QUÉ se evalúa, como (etiqueta, ruta). Las rutas de SGRL se rotulan
    'sgrl:<ruta relativa a --sg-repo>' para que la etiqueta sea la misma en cualquier máquina."""
    sg_repo = Path(a.sg_repo).resolve()
    files = [("wann:" + str(ctx.wann["net"]), Path(ctx.wann["net"])), ("wann:" + ctx.base_config, Path(ctx.base_config))]
    if ctx.wann["override"]:
        files.append(("wann:" + str(ctx.wann["override"]), Path(ctx.wann["override"])))
    for m in ctx.models:                      # --replicas: cada semilla de WANN trae su propia red y override
        if m["source"] == "wann" and "net" in m:
            files += [("wann:" + m["net"], Path(m["net"])), ("wann:" + m["override"], Path(m["override"]))]
    files += [("wann:" + c, Path(c)) for c in ctx.spec["eval_csvs"]]       # deciden cuál es "el mejor"
    default_dir = a.sg_default_dir or ("runs_default" if (sg_repo / "runs_default").is_dir() else "runs")
    study = a.sg_study or ctx.spec["sg_study"]
    sel = [sg_repo / default_dir / f"{study}-validation.json"]                   # decide el trial de 64x64
    for v in a.sg_variants:                                                      # deciden el rank de cada variante
        sel.append(sg_repo / getattr(a, f"sg_{v}_screening") / ctx.spec["sg_task"] / "p3_summary.json")
    for m in ctx.models:
        if m["source"] != "wann":
            d = Path(m["path"]).parent
            sel += [Path(m["path"]), d / "config.json"]
            sel += [d / f for f in ("final_eval.json", "quant.json") if (d / f).is_file()]
    for f in sel:
        try:
            rel = f.resolve().relative_to(sg_repo)
        except ValueError:
            rel = f
        files.append((f"sgrl:{rel}", f))
    return files


def write_manifest(ctxs, a, out: Path):
    """Estilo sha256sum: '<sha256>  <bytes>  <etiqueta>'. Corré lo mismo en el PC y en el clúster y compará con diff."""
    import hashlib
    lines = []
    for ctx in ctxs:
        lines.append(f"# tarea {ctx.name}")
        for label, f in input_files(ctx, a):
            if f.is_file():
                lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.stat().st_size:>9d}  {label}")
            else:
                lines.append(f"{'FALTA':64s}  {'-':>9s}  {label}")
    out.write_text("\n".join(lines) + "\n")
    miss = sum(l.startswith("FALTA") for l in lines)
    print(f"Manifiesto: {out} ({len(lines) - len(ctxs)} archivos, {miss} faltantes)")


def check_protocol(ctx: TaskCtx, a):
    """Las partes se nombran por (modelo, celda, semilla), NO por n_chips/lecturas/episodios ni por la
    red: relanzar con otro protocolo sobre la misma carpeta mezclaría resultados incompatibles."""
    p = ctx.out_dir / "protocol.json"
    if not p.is_file() or a.rerun:
        return
    old = json.loads(p.read_text())
    diffs = []
    for k in ("n_chips", "reads_per_chip", "n_levels", "reramcmo_g_max", "reramcmo_g_min"):
        if str(getattr(a, k)) != old.get("args", {}).get(k):
            diffs.append(f"{k}: {old.get('args', {}).get(k)} -> {getattr(a, k)}")
    if old.get("episodes") != ctx.episodes:
        diffs.append(f"episodes: {old.get('episodes')} -> {ctx.episodes}")
    if old.get("wann", {}).get("net") != ctx.wann["net"]:
        diffs.append(f"red WANN: {old.get('wann', {}).get('net')} -> {ctx.wann['net']}")
    # Se compara "<corrida>/<archivo>" y no la ruta completa: mover runs/ a runs_default/ no cambia el modelo.
    tail = lambda path: "/".join(Path(path).parts[-2:])
    old_paths = {m["model_id"]: m["path"] for m in old.get("models", [])}
    for m in ctx.models:
        if m["model_id"] in old_paths and tail(old_paths[m["model_id"]]) != tail(m["path"]):
            diffs.append(f"{m['model_id']}: {old_paths[m['model_id']]} -> {m['path']}")
    if diffs:
        sys.exit(f"[{ctx.name}] {ctx.out_dir} ya tiene resultados con otro protocolo:\n  " + "\n  ".join(diffs) +
                 "\nUsá otra --out-dir (recomendado) o --rerun para rehacerlo todo; si no, se mezclarían partes incompatibles.")


# --------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=list(TASKS), default=list(TASKS),
                    help="tareas a comparar (def.: las tres)")
    g = ap.add_argument_group("WANN")
    g.add_argument("--wann-run-key", default=None, help="restringe la búsqueda del mejor modelo a run_keys que contengan este texto")
    g.add_argument("--wann-weights", default="0,1,2,3,4,5",
                   help="índices de peso compartido a evaluar (WANN es agnóstica al peso: se muestran todos)")
    g.add_argument("--wann-net", default=None, help="red .out manual (solo con una tarea; si no, se elige la mejor)")
    g.add_argument("--wann-override", default=None, help="override JSON de esa red (con --wann-net)")
    g.add_argument("--wann-winner-weight", type=int, default=1, help="índice del peso ganador (con --wann-net)")
    g = ap.add_argument_group("Gradientes surrogados")
    g.add_argument("--sg-repo", default="../surrogate-gradient-rl")
    g.add_argument("--sg-python", default=None, help="def.: <sg-repo>/.venv/bin/python")
    g.add_argument("--sg-study", default=None, help="def.: carrace-f2 / mountaincar-f2 / acrobot-f2 según la tarea")
    g.add_argument("--sg-trial", default="auto", help="trial de Optuna; auto = el de mayor media de validación")
    g.add_argument("--sg-top", type=int, default=5, help="mejores semillas (por final_eval) del trial")
    g.add_argument("--sg-ckpt", choices=["final", "best"], default="final")
    g.add_argument("--sg-runs", nargs="+", default=None, help="carpetas de corrida explícitas (solo con una tarea)")
    g.add_argument("--sg-device", default="cpu")
    g.add_argument("--sg-default-dir", default=None,
                   help="carpeta (dentro de --sg-repo) de las corridas 64×64 originales; def.: runs_default si existe, si no runs")
    g.add_argument("--sg-variants", nargs="*", choices=list(VARIANTS), default=list(VARIANTS),
                   help="variantes con la red del tamaño de la de WANN: con (mismas conexiones), neu (mismas neuronas); "
                        "sin valores = ninguna (def.: las dos)")
    g.add_argument("--sgv-top", type=int, default=1, help="semillas por variante (def.: 1 = solo el mejor modelo)")
    g.add_argument("--sg-quant-bits", type=int, default=0,
                   help="si > 0, añade las versiones cuantizadas (B bits) de los modelos principales de SGRL; 0 = no (def.)")
    g.add_argument("--sg-quant-scheme", default="odin", choices=["odin", "symmetric"])
    g.add_argument("--sg-quant-dir", default="runs_quant", help="carpeta (en --sg-repo) con los checkpoints de quantize_eval.py")
    g.add_argument("--sg-con-runs", default="runs", help="corridas de la variante por conexiones")
    g.add_argument("--sg-con-screening", default="screening")
    g.add_argument("--sg-neu-runs", default="runs_neuronas", help="corridas de la variante por neuronas")
    g.add_argument("--sg-neu-screening", default="screening_neuronas")
    g = ap.add_argument_group("Protocolo de ruido (igual en ambos lados)")
    g.add_argument("--replicas", action="store_true",
                   help="evaluar las 11 semillas de cada método (ver el docstring); salida en <tarea>_replicas/")
    g.add_argument("--replica-wann-weights", choices=["odin", "original", "best"], default="odin",
                   help="--replicas: pesos compartidos de WANN por red. odin (def.) = niveles ENTEROS 1..7, los únicos que "
                        "ODIN puede guardar (peso simulado = nivel, w_max = 7); original = los 6 de WEIGHT_VALS (0.5..8, "
                        "algunos no representables en ODIN); best = solo el mejor de la revalidación original")
    g.add_argument("--odin-levels", default="1,2,3,4,5,6,7", help="niveles de magnitud de WANN con --replica-wann-weights odin")
    g.add_argument("--wann-wmax", type=float, default=7.0, help="w_max de WANN con pesos odin (nivel 7 = g_max)")
    g.add_argument("--replica-cells", choices=["reduced", "full"], default="reduced",
                   help="celdas del modo --replicas: reduced = 27 (def.), full = las 191 de siempre")
    g.add_argument("--seeds", type=int, nargs="+", default=None, help="def.: 0 1 (0 con --replicas)")
    g.add_argument("--n-chips", type=int, default=None, help="def.: 30")
    g.add_argument("--reads-per-chip", type=int, default=None, help="def.: 3 (1 con --replicas)")
    g.add_argument("--episodes", type=int, default=None, help="def.: alg_nReps del config de WANN")
    g.add_argument("--n-levels", type=int, default=0)
    g.add_argument("--noise-scales", default="0,0.5,1,2,3,4,5")
    g.add_argument("--time-points-pcm", default="0,60,300,3600,21600,86400,604800,2592000,31536000")
    g.add_argument("--time-points-reram", default="1,86400,172800", help="deben ser EXACTAMENTE estas 3 claves")
    g.add_argument("--time-points-reramcmo", default="0,60,300,3600,21600,86400,604800,2592000,31536000")
    g.add_argument("--ablation-t", type=float, default=86400.0, help="t (s) de la ablación en PCM")
    g.add_argument("--no-ablation", action="store_true")
    g.add_argument("--with-simple", action="store_true", help="incluir el modelo 'simple' (placeholder) en intensidad")
    g.add_argument("--reramcmo-g-max", type=float, default=25.0)
    g.add_argument("--reramcmo-g-min", type=float, default=1.0)
    g = ap.add_argument_group("Ejecución")
    g.add_argument("--jobs", type=int, default=500, help="procesos simultáneos en total")
    g.add_argument("--jobs-sgrl", type=int, default=None,
                   help="tope de procesos Python/torch simultáneos (def.: --jobs); ~0.5 GB de RAM cada uno")
    g.add_argument("--timeout", type=int, default=3600, help="segundos por trabajo")
    g.add_argument("--out-dir", default=None, help="carpeta base (def.: eval_results); cada tarea va a <base>/<tarea>_wann_vs_sgrl")
    g.add_argument("--graficos-dir", default=None, help="def.: graficos/<Tarea> (Racing Car / Mountain Car / Acrobot)")
    g.add_argument("--rerun", action="store_true", help="rehacer también las partes ya existentes")
    g.add_argument("--skip-run", action="store_true", help="no correr nada: solo consolidar y graficar")
    g.add_argument("--skip-plots", action="store_true")
    g.add_argument("--plot-anyway", action="store_true", help="graficar aunque haya trabajos fallidos")
    g.add_argument("--dry-run", action="store_true", help="solo imprimir el plan y la estimación")
    g.add_argument("--manifest", type=Path, default=None,
                   help="escribe la lista (con sha256) de los archivos que deciden qué modelos se evalúan, y sale; "
                        "corrélo en el PC y en el clúster y compará con diff")
    a = ap.parse_args()

    if len(a.task) > 1 and (a.wann_net or a.sg_runs or a.sg_study or a.graficos_dir):
        sys.exit("Error: --wann-net/--sg-runs/--sg-study/--graficos-dir valen para UNA tarea (--task X).")
    a.wann_weights = [int(x) for x in a.wann_weights.split(",") if x.strip()]
    a.n_chips = a.n_chips or 30
    a.reads_per_chip = a.reads_per_chip or (1 if a.replicas else 3)
    a.seeds = a.seeds or ([0] if a.replicas else [0, 1])
    a.noise_scales = csv_floats(a.noise_scales)
    for k in ("time_points_pcm", "time_points_reram", "time_points_reramcmo"):
        setattr(a, k, csv_floats(getattr(a, k)))
    a.jobs_sgrl = a.jobs if a.jobs_sgrl is None else a.jobs_sgrl
    if a.sg_python is None:
        venv = Path(a.sg_repo) / ".venv" / "bin" / "python"
        # absolute() y NO resolve(): el python del venv es un symlink al intérprete base, y al
        # resolverlo se pierde el pyvenv.cfg y, con él, los paquetes del venv (torch, snntorch).
        a.sg_python = str(venv.absolute()) if venv.is_file() else sys.executable

    ctxs = [TaskCtx(t, a) for t in a.task]
    cells = build_cells(a)
    jobs = []
    for ctx in ctxs:
        ctx.describe()
        jobs += build_jobs(a, ctx, cells)

    if a.manifest:
        write_manifest(ctxs, a, a.manifest)
        return

    rows = a.n_chips * a.reads_per_chip
    todo = [j for j in jobs if a.rerun or not j.part.is_file()]
    cpu = {k: sum(j.cost for j in todo if j.kind == k) for k in COST_PER_ROW}
    print(f"\nPlan: {len(ctxs)} tareas × {len(ctxs[0].models)} modelos × {len(cells)} celdas × {len(a.seeds)} semillas "
          f"= {len(jobs)} trabajos de {rows} filas; {len(jobs) - len(todo)} ya hechos, {len(todo)} por correr.")
    print("Por modelo (partes hechas / totales):")
    for ctx in ctxs:
        for m in ctx.models:
            mine = [j for j in jobs if j.ctx is ctx and j.model is m]
            print(f"  [{ctx.name}] {m['model_id']:42s} {sum(j.part.is_file() for j in mine):4d}/{len(mine)}")
    # pared ≥ máx(total/jobs, SGRL/jobs_sgrl, trabajo más largo): el tope de SGRL deja hebras ociosas
    # cuando WANN (la parte barata) ya terminó, así que cuenta como cota inferior propia.
    longest = max((j.cost for j in todo), default=0.0)
    wall = max(sum(cpu.values()) / a.jobs, cpu["sgrl"] / min(a.jobs, a.jobs_sgrl), longest)
    print(f"CPU pendiente estimada (cota superior): WANN {cpu['wann'] / 3600:.1f} h + SGRL {cpu['sgrl'] / 3600:.1f} h; con "
          f"{a.jobs} procesos ({min(a.jobs, a.jobs_sgrl)} de SGRL) ≈ {wall / 60:.0f} min de pared (el trabajo más "
          f"largo ≈ {longest / 60:.1f} min).")
    print(f"Memoria máxima aprox. de SGRL: {min(a.jobs, a.jobs_sgrl)} procesos × ~0.5 GB "
          f"= {min(a.jobs, a.jobs_sgrl) * 0.5:.0f} GB.")
    if a.dry_run:
        for ctx in ctxs:
            for j in [j for j in jobs if j.ctx is ctx][:1] + [j for j in jobs if j.ctx is ctx and j.source == "sgrl_con"][:1]:
                print(f"[dry-run] {j.label}\n   $ {' '.join(j.cmd)}")
        return

    for ctx in ctxs:
        check_protocol(ctx, a)
    for ctx in ctxs:
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        (ctx.out_dir / "protocol.json").write_text(json.dumps(
            dict(created=datetime.now().isoformat(timespec="seconds"), task=ctx.name, task_label=ctx.spec["label"],
                 ret_floor=ctx.spec["ret_floor"], wann=ctx.wann, episodes=ctx.episodes,
                 args={k: str(v) for k, v in vars(a).items()}, models=ctx.models, n_cells=len(cells)),
            indent=1, ensure_ascii=False))

    failures = [] if a.skip_run else run_jobs(jobs, a)
    if failures:
        by_ctx = {}
        for j, why in failures:
            by_ctx.setdefault(j.ctx, []).append((j, why))
        for ctx, fl in by_ctx.items():
            with open(ctx.out_dir / "failed_jobs.txt", "w") as f:
                for j, why in fl:
                    f.write(f"{j.label}: {why}\n   $ {' '.join(j.cmd)}\n")
            print(f"[{ctx.name}] fallos en {ctx.out_dir / 'failed_jobs.txt'}; relanzá el mismo comando para reintentarlos.")
    bad = False
    for ctx in ctxs:
        missing = consolidate(ctx, jobs, a)
        if a.skip_plots:
            continue
        if missing and not a.plot_anyway:
            print(f"[{ctx.name}] {missing} trabajos sin resultado: no se grafica (usá --plot-anyway para forzar).")
            bad = True
            continue
        if a.replicas:
            for view in ("best", "mean"):
                subprocess.run([sys.executable, "eval_results/stats_noise_replicas.py", "--data-dir", str(ctx.out_dir),
                                "--wann-view", view], check=True)
            for grp in ("float",) + (("quant",) if a.sg_quant_bits else ()):
                subprocess.run([sys.executable, "eval_results/plot_noise_replicas.py", "--data-dir", str(ctx.out_dir),
                                "--out-dir", ctx.graficos, "--prefix", f"{ctx.name}_replicas", "--group", grp], check=True)
            continue
        subprocess.run([sys.executable, "eval_results/stats_noise_compare.py", "--task", ctx.name,
                        "--base-dir", str(ctx.out_dir.parent)], check=True)
        subprocess.run([sys.executable, "eval_results/plot_noise_compare.py", "--data-dir", str(ctx.out_dir),
                        "--out-dir", ctx.graficos, "--prefix", ctx.prefix], check=True)
        if a.sg_quant_bits:
            subprocess.run([sys.executable, "eval_results/plot_noise_compare.py", "--data-dir", str(ctx.out_dir),
                            "--out-dir", ctx.graficos, "--prefix", f"{ctx.prefix}q{a.sg_quant_bits}",
                            "--sources", "wann", "sgrlq", "sgrlq_con", "sgrlq_neu"], check=True)
    if failures or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
