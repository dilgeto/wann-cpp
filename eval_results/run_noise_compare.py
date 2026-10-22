#!/usr/bin/env python3
"""
Compara la robustez a ruido de memristores del mejor modelo WANN de cada tarea
(wann_{car,acrobot,disc_mc}_noise) contra checkpoints de gradientes surrogados
de ../surrogate-gradient-rl (scripts/noise_eval.py), con el MISMO protocolo en
ambos lados, y lo hace todo en paralelo (un solo planificador para las tareas).

Solo evalúa, nunca entrena. Orquesta (1) los barridos, (2) la consolidación en
un CSV normalizado por tarea y (3) las figuras de plot_noise_compare.py.

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
  python eval_results/run_noise_compare.py --jobs 500 --jobs-sgrl 250      # las 3 tareas
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
# Costo aproximado por fila (s de CPU, una hebra), medido en car; solo para ordenar y estimar.
COST_PER_ROW = {"wann": 0.8, "sgrl": 3.2}

# ret_floor: recompensa del peor desenlace (fracaso total); la "retención" se mide entre este
# piso y el retorno sin ruido del propio modelo. car: sin piso negativo (0 ⇒ retorno/base);
# acrobot: -1 por paso hasta 500; Mountain Car discreto: -1 por paso hasta 200.
TASKS = {
    "car": dict(
        label="CarTrack", binary="./build/wann_car_noise", base_config="p/car_snn.json",
        sg_study="carrace-f2", ret_floor=0.0, graficos="Racing Car", prefix="car_wann_vs_sgrl",
        eval_csvs=["eval_results/car_ttfs_first_spike_discrete/car_best.csv"]),
    "disc_mc": dict(
        label="Mountain Car (discrete)", binary="./build/wann_disc_mc_noise", base_config="p/disc_mc_snn.json",
        sg_study="mountaincar-f2", ret_floor=-200.0, graficos="Mountain Car", prefix="disc_mc_wann_vs_sgrl",
        eval_csvs=sorted(glob.glob("eval_results/disc_mc_*_discrete/mountain_car_best.csv"))),
    "acrobot": dict(
        label="Acrobot", binary="./build/wann_acrobot_noise", base_config="p/acrobot_snn.json",
        sg_study="acrobot-f2", ret_floor=-500.0, graficos="Acrobot", prefix="acrobot_wann_vs_sgrl",
        eval_csvs=["eval_results/eval_p3_weights/acrobot_best.csv"]),
}

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
def build_cells(a) -> list[tuple]:
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


def select_sgrl(task: str, spec: dict, a) -> list[dict]:
    repo = Path(a.sg_repo).resolve()
    ckpt_name = f"{a.sg_ckpt}.pt"
    study = a.sg_study or spec["sg_study"]
    if a.sg_runs:
        dirs = [Path(d).resolve() for d in a.sg_runs]
    else:
        val_path = repo / "runs" / f"{study}-validation.json"
        if not val_path.is_file():
            sys.exit(f"Error [{task}]: no existe {val_path}. Pasá --sg-study o --sg-runs.")
        val = json.loads(val_path.read_text())
        trial = a.sg_trial if a.sg_trial != "auto" else max(val, key=lambda k: val[k]["mean"])
        if trial not in val:
            sys.exit(f"Error [{task}]: el trial {trial} no está en {val_path.name} (hay: {sorted(val, key=int)}).")
        scored = []
        for d in sorted((repo / "runs").glob(f"{study}-t{trial}-s*")):
            fe = d / "final_eval.json"
            if fe.is_file():
                scored.append((json.loads(fe.read_text())["return_mean"], d))
        if not scored:
            sys.exit(f"Error [{task}]: sin corridas con final_eval.json para {study}-t{trial} en {repo / 'runs'}.")
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
                        train_return=ret, weight_index=""))
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
        self.out_dir = Path(a.out_dir) / f"{task}_wann_vs_sgrl" if a.out_dir else Path("eval_results") / f"{task}_wann_vs_sgrl"
        self.graficos = a.graficos_dir or f"graficos/{spec['graficos']}"
        self.prefix = spec["prefix"]
        w = self.wann
        wlabel = f"WANN {w['run_key']} rank{w['rank']:02d}-seed{w['seed_idx']:02d}" if w["rank"] >= 0 else "WANN (manual)"
        self.models = [dict(model_id=f"wann_w{wi}", source="wann", label=f"{wlabel} (w idx {wi})",
                            path=w["net"], train_return=w["reward"], weight_index=wi)
                       for wi in a.wann_weights] + select_sgrl(task, spec, a)

    def describe(self):
        w = self.wann
        sg = [m for m in self.models if m["source"] == "sgrl"]
        print(f"[{self.name}] WANN: {w['run_key']} rank{w['rank']:02d} seed{w['seed_idx']:02d}, peso idx "
              f"{w['weight_index']} (reward revalidado {w['reward']:.1f} ± {w['reward_std']:.1f}); "
              f"SGRL: {len(sg)} checkpoints ({', '.join(Path(m['path']).parent.name for m in sg)}); "
              f"{self.episodes} episodios por fila.")


# --------------------------------------------------------------------------------------------
# Trabajos
# --------------------------------------------------------------------------------------------
class Job:
    def __init__(self, ctx, model, cell, seed, cmd, part, rows):
        self.ctx, self.source, self.model, self.cell, self.seed = ctx, model["source"], model, cell, seed
        self.cmd, self.part, self.rows = cmd, part, rows
        self.tmp = part.with_suffix(".csv.tmp")
        self.err = part.with_suffix(".err")
        self.cost = rows * COST_PER_ROW[self.source]

    @property
    def label(self):
        m, s, t, mech = self.cell
        return f"[{self.ctx.name}] {self.model['model_id']} {m}/{mech} scale={fmt(s)} t={fmt(t)} seed={self.seed}"


def cell_stem(cell, seed) -> str:
    m, s, t, mech = cell
    return f"{m}_{mech}_s{fmt(s)}_t{fmt(t)}_seed{seed}"


def wann_cmd(a, ctx, model, cell, seed) -> list[str]:
    m, s, t, _ = cell
    w = ctx.wann
    cmd = [ctx.binary, "-f", w["net"], "-d", ctx.base_config]
    if w["override"]:
        cmd += ["-p", w["override"]]
    cmd += ["--weight-index", str(model["weight_index"]), "--mode", "per-synapse", "--noise-model", m,
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
            for seed in a.seeds:
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
                idx = next((i for i, j in enumerate(todo) if j.source != "sgrl" or n_sgrl < a.jobs_sgrl), None)
                if idx is None:
                    break
                job = todo.pop(idx)
                running[launch(job, env)] = (job, time.time())
                n_sgrl += job.source == "sgrl"
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
                n_sgrl -= job.source == "sgrl"
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
    sm = mf[mf.source == "sgrl"]
    if len(sm):
        mf.loc[sm.index[0], "main"] = 1
    mf.to_csv(ctx.out_dir / "models.csv", index=False)
    print(f"[{ctx.name}] consolidado: {ctx.out_dir / 'sweep_all.csv'} ({len(df)} filas, {len(frames)} partes, "
          f"{missing} faltantes) y models.csv")
    return missing


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
    g = ap.add_argument_group("Protocolo de ruido (igual en ambos lados)")
    g.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    g.add_argument("--n-chips", type=int, default=30)
    g.add_argument("--reads-per-chip", type=int, default=3)
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
    a = ap.parse_args()

    if len(a.task) > 1 and (a.wann_net or a.sg_runs or a.sg_study or a.graficos_dir):
        sys.exit("Error: --wann-net/--sg-runs/--sg-study/--graficos-dir valen para UNA tarea (--task X).")
    a.wann_weights = [int(x) for x in a.wann_weights.split(",") if x.strip()]
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

    rows = a.n_chips * a.reads_per_chip
    cpu = {s: sum(j.cost for j in jobs if j.source == s) for s in COST_PER_ROW}
    print(f"\nPlan: {len(ctxs)} tareas × {len(ctxs[0].models)} modelos × {len(cells)} celdas × {len(a.seeds)} semillas "
          f"= {len(jobs)} trabajos de {rows} filas.")
    print(f"CPU estimada (cota superior): WANN {cpu['wann'] / 3600:.1f} h + SGRL {cpu['sgrl'] / 3600:.1f} h; con "
          f"{a.jobs} procesos ≈ {sum(cpu.values()) / a.jobs / 60:.0f} min de pared (el trabajo más largo ≈ "
          f"{max(j.cost for j in jobs) / 60:.1f} min).")
    print(f"Memoria máxima aprox. de SGRL: {min(a.jobs, a.jobs_sgrl)} procesos × ~0.5 GB "
          f"= {min(a.jobs, a.jobs_sgrl) * 0.5:.0f} GB.")
    if a.dry_run:
        for ctx in ctxs:
            for j in [j for j in jobs if j.ctx is ctx][:1] + [j for j in jobs if j.ctx is ctx and j.source == "sgrl"][:1]:
                print(f"[dry-run] {j.label}\n   $ {' '.join(j.cmd)}")
        return

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
        subprocess.run([sys.executable, "eval_results/plot_noise_compare.py", "--data-dir", str(ctx.out_dir),
                        "--out-dir", ctx.graficos, "--prefix", ctx.prefix], check=True)
    if failures or bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
