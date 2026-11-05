#!/usr/bin/env python3
"""
Estudio de w_max (fracción de g_max que ocupa el peso compartido) para las 11 redes WANN de cada tarea
(CarTrack, Mountain Car discreto, Acrobot), con una grilla de tiempos densa. SOLO EVALÚA (no entrena): lanza
wann_<tarea>_noise, una celda por proceso, y deja un CSV por celda (reanudable).

Pregunta: ¿cuánto se puede reducir la conductancia con que se programan los pesos (y con ello el consumo estático
teórico, ∝ conductancia) antes de que el ruido sináptico del memristor degrade el desempeño? Generaliza
wmax_check_car.py (una red, un peso, 4 tiempos) al protocolo de las réplicas:
  * Redes: las 11 semillas de revalidación de la configuración ganadora, las mismas de eval_results/<tarea>_replicas/
    (rutas de net/override de su models.csv / protocol.json).
  * Peso de cada red (--weights best, por defecto): SU peso compartido ganador de WEIGHT_VALS, el de la revalidación
    sin ruido (columnas weight_index/weight_value de los *_best.csv; mismo criterio que select_wann_replicas con
    --replica-wann-weights best). Con --weights odin: su mejor nivel ODIN entero 1..7 elegido con la semilla 999 de
    <tarea>_replicas/sweep_all.csv (no simula). Queda en <out>/levels.csv.
  * Ocupación: el peso ocupa `occ` de g_max, es decir w_max = peso / occ (occ en {6,25, 12,5, 25, 50, 100 %}).
    OJO con el sentido: `w_max` es el peso que se escribe como g_max, así que BAJAR la ocupación (SUBIR w_max) es
    lo que baja la conductancia y el consumo. Parametrizar por ocupación y no por w_max absoluto es lo que permite
    comparar redes con pesos distintos. La red simulada es la misma (peso lógico fijo); solo cambia dónde cae el
    peso dentro del rango del chip. Supone que la neurona compensa la escala (lectura w = g·w_max/g_max); ese costo
    de periferia no está contado. Con ReRAM-CMO, g_min = 1 µS impone un piso: a ocupación ≲ 4 % (g < g_min) el peso
    no es representable (la celda de escala 0 lo aísla).
  * --weights all (para el frente de Pareto de eval_results/pareto_wmax.py): cada red se evalúa con LOS 6 pesos
    compartidos de WEIGHT_VALS (no solo con el ganador) × todas las ocupaciones. ~6× más procesos. Aquí solo se
    escribe networks.csv (sin summary/tradeoff, que asumen un peso por red).
  * Celdas (por red y ocupación): PCM y ReRAM-CMO a escala 1 en toda la grilla de tiempos (--times), más CMO a
    escala 3 y a escala 0 (piso g_min) con t = 1 s (figura de ocupación), y escala 3 a 1 año ("más ruido" para el
    Pareto). Más una celda `none` por red
    (sin ruido, línea base de la retención; no depende de w_max).
  * Protocolo de ruido: --n-chips 30 --reads-per-chip 1 (el de las réplicas); la unidad de réplica es la RED
    (n = 11), no el chip. Retención = (retorno - piso) / (retorno sin ruido - piso) × 100, con el piso de la
    tarea (car 0, acrobot -500, Mountain Car -200; se lee de <tarea>_replicas/protocol.json).

Salida (por tarea), en eval_results/<tarea>_wmax_check/:
  parts/s<SS>_L<L>/<celda>.csv   salida cruda de cada proceso (se saltan los ya hechos al relanzar)
  levels.csv                     red -> peso elegido (columna `level`)
  tradeoff.csv                   ocupación mínima que mantiene >= 95 / 90 % de retención, por celda (ahorro = 1 - occ)
  networks.csv                   una fila por (red, dispositivo, escala, t, ocupación): retorno y retención
  summary.csv                    por (dispositivo, escala, t, ocupación): media sobre las 11 redes e IC95 % bootstrap
  failed_jobs.txt                solo si algún proceso falla (el resto continúa; el script sale con código 1)

Uso (desde la raíz del repo, con ./build/wann_*_noise compilados):
  python eval_results/wmax_check_all.py --dry-run                       # plan, niveles y nº de procesos; no simula
  python eval_results/wmax_check_all.py --jobs 200                      # las 3 tareas
  python eval_results/wmax_check_all.py --task car --jobs 64             # una tarea
  python eval_results/wmax_check_all.py --summary-only                  # solo consolidar lo ya calculado
Luego las figuras:  python eval_results/plot_wmax_check.py --task all

Cada proceso corre con OMP_NUM_THREADS=1; --jobs es el nº de procesos simultáneos (≈ núcleos). Con los valores por
defecto son ~(11 redes × 5 ocupaciones × (2×18 + 4) celdas + 11) ≈ 2200 procesos (×6 con --weights all) por tarea, de ≲1 min cada uno
(CarTrack, el más lento) y mucho menos en las otras dos.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
os.chdir(REPO)

import numpy as np
import pandas as pd

YEAR = 31536000.0
# Grilla de tiempos desde la programación (s): 0, 1 s ... 1 año, ~2 puntos por década. Cualquier t >= 0 es válido
# para PCM y ReRAM-CMO (ReRAM Wan2022 solo admite t tabulados y por eso no se usa aquí).
TIMES = [0, 1, 10, 60, 300, 900, 3600, 7200, 21600, 43200, 86400, 172800, 604800, 1209600, 2592000, 7776000,
         15768000, 31536000]
OCCUPANCIES = [0.0625, 0.125, 0.25, 0.5, 1.0]
THRESHOLDS = [95.0, 90.0]     # retención mínima aceptable para el análisis de ahorro (tradeoff.csv)
DEVICES = ["pcm", "reramcmo"]
SELECT_SEED = 999     # semilla reservada para elegir el nivel sin ruido (igual que run_noise_compare.py)
RUN_SEED = 0          # semilla de chips/episodios de las celdas con ruido (la de las réplicas)

TASKS = {
    "car": dict(binary="./build/wann_car_noise", config="p/car_snn.json", graficos="Racing Car",
                eval_csvs=["eval_results/car_ttfs_first_spike_discrete/car_best.csv"]),
    "disc_mc": dict(binary="./build/wann_disc_mc_noise", config="p/disc_mc_snn.json", graficos="Mountain Car",
                    eval_csvs=sorted(glob.glob("eval_results/disc_mc_*_discrete/mountain_car_best.csv"))),
    "acrobot": dict(binary="./build/wann_acrobot_noise", config="p/acrobot_snn.json", graficos="Acrobot",
                    eval_csvs=["eval_results/eval_p3_weights/acrobot_best.csv"]),
}


def fmt(x) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def occ_tag(occ: float) -> str:
    return f"o{int(round(occ * 1000))}"          # 0.125 -> o125, 1.0 -> o1000 (sin puntos en el nombre de archivo)


def data_dir(task: str) -> Path:
    return Path(f"eval_results/{task}_replicas")


def out_dir(task: str, a) -> Path:
    return Path(a.out_dir) / f"{task}_wmax_check" if a.out_dir else Path(f"eval_results/{task}_wmax_check")


# --------------------------------------------------------------------------------------------
# Redes y niveles
# --------------------------------------------------------------------------------------------
def check_files(task: str, best: pd.DataFrame) -> pd.DataFrame:
    for r in best.itertuples():
        for f in (r.net, r.override):
            if not Path(f).is_file():
                sys.exit(f"Error [{task}]: no existe {f} (¿falta sincronizar log/ o screening_full/ en el clúster?)")
    return best[["seed", "model_id", "net", "override", "level", "sel"]].reset_index(drop=True)


SRC = {"car": "src/SnnCarTask.cpp", "disc_mc": "src/SnnDiscMCTask.cpp", "acrobot": "src/SnnAcrobotTask.cpp"}


def weight_vals(task: str) -> list[float]:
    """WEIGHT_VALS de la tarea, leídos del fuente (una sola verdad con lo que entrenó/evalúa el binario)."""
    txt = Path(SRC[task]).read_text()
    m = re.search(r"WEIGHT_VALS\[N_WEIGHTS\]\s*=\s*\{([^}]*)\}", txt)
    if not m:
        sys.exit(f"Error [{task}]: no pude leer WEIGHT_VALS de {SRC[task]}")
    return [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]


def pick_networks(task: str, weights: str = "best") -> pd.DataFrame:
    """Una fila por semilla de WANN: red, override y peso (columna `level`).
    best: peso ganador de WEIGHT_VALS de la revalidación sin ruido; odin: mejor nivel entero 1..7 sin ruido."""
    d = data_dir(task)
    if weights == "all":                       # las 11 redes × los 6 pesos de WEIGHT_VALS
        b = pick_networks(task, "best")
        wv = weight_vals(task)
        out = pd.concat([b.assign(level=w, sel=np.nan) for w in wv], ignore_index=True)
        return out.sort_values(["seed", "level"]).reset_index(drop=True)
    if weights == "best":
        pj = d / "protocol.json"
        if not pj.is_file():
            sys.exit(f"Error [{task}]: falta {pj} (de él sale la configuración ganadora: run_key y rank).")
        w = json.loads(pj.read_text())["wann"]
        df = pd.concat([pd.read_csv(c) for c in TASKS[task]["eval_csvs"] if Path(c).is_file()], ignore_index=True)
        if "reward_type" in df.columns and (df.reward_type == "original").any():
            df = df[df.reward_type == "original"]
        df = df[(df.run_key == w["run_key"]) & (df["rank"] == w["rank"])].sort_values("seed_idx")
        if df.empty:
            sys.exit(f"Error [{task}]: no hay filas de {w['run_key']} rank {w['rank']} en {TASKS[task]['eval_csvs']}")
        best = pd.DataFrame({
            "seed": df.seed_idx.astype(int).to_numpy(),
            "model_id": [f"wann_s{int(x):02d}" for x in df.seed_idx],
            "net": [f"log/full_p3_{w['run_key']}/rank{int(w['rank']):02d}_seed{int(x):02d}_best.out" for x in df.seed_idx],
            "override": [f"screening_full/{w['run_key']}/p3_configs/rank{int(w['rank']):02d}_seed{int(x):02d}.json"
                         for x in df.seed_idx],
            "level": df.weight_value.astype(float).to_numpy(), "sel": df.reward.astype(float).to_numpy()})
        return check_files(task, best)
    for f in ("models.csv", "sweep_all.csv"):
        if not (d / f).is_file():
            sys.exit(f"Error [{task}]: falta {d / f} (salida de run_noise_compare.py --replicas). Hace falta para "
                     f"saber qué 11 redes usar y qué nivel ODIN tiene cada una.")
    models = pd.read_csv(d / "models.csv")
    models = models[models.source == "wann"]                       # el mapeo ODIN; wann_fs repite red y nivel
    sweep = pd.read_csv(d / "sweep_all.csv", usecols=["model_id", "seed", "noise_model", "ret"])
    sel = sweep[(sweep.seed == SELECT_SEED) & (sweep.noise_model == "none")].groupby("model_id")["ret"].mean()
    if sel.empty:
        sys.exit(f"Error [{task}]: {d / 'sweep_all.csv'} no tiene celdas `none` con la semilla {SELECT_SEED}.")
    m = models.assign(sel=models.model_id.map(sel)).dropna(subset=["sel"])
    best = m.sort_values("sel", ascending=False).drop_duplicates("replica_seed").sort_values("replica_seed")
    best = best.rename(columns={"replica_seed": "seed", "weight_value": "level"})
    best["seed"] = best["seed"].astype(int)
    best["level"] = best["level"].astype(float)
    return check_files(task, best)


def cells(times: list[float]) -> list[tuple]:
    """(dispositivo, escala, t) con ruido, sin repetir."""
    out = [(dev, 1.0, float(t)) for dev in DEVICES for t in times]
    for extra in [("reramcmo", 3.0, 1.0), ("reramcmo", 0.0, 1.0),          # figura de ocupación
                  ("pcm", 3.0, YEAR), ("reramcmo", 3.0, YEAR)]:            # "más ruido" a 1 año (Pareto)
        if extra not in out:
            out.append(extra)
    return out


# --------------------------------------------------------------------------------------------
# Trabajos
# --------------------------------------------------------------------------------------------
def stem_of(dev: str, scale: float, t: float, occ: float | None) -> str:
    return f"none_seed{RUN_SEED}" if dev == "none" else f"{dev}_s{fmt(scale)}_t{fmt(t)}_{occ_tag(occ)}_seed{RUN_SEED}"


def jobs_for(task: str, nets: pd.DataFrame, a, root: Path):
    spec = TASKS[task]
    cl = cells(a.times)
    for r in nets.itertuples():
        pdir = root / "parts" / f"s{r.seed:02d}_L{fmt(r.level)}"
        base = [spec["binary"], "-f", r.net, "-d", spec["config"], "-p", r.override, "--weight", fmt(r.level),
                "--mode", "per-synapse", "--n-chips", str(a.n_chips), "--reads-per-chip", str(a.reads_per_chip),
                "--n-levels", "0", "--reramcmo-g-max", fmt(a.reramcmo_g_max), "--reramcmo-g-min", fmt(a.reramcmo_g_min),
                "--seeds", str(RUN_SEED)]
        yield pdir / f"{stem_of('none', 0, 0, None)}.csv", \
            base + ["--w-max", fmt(r.level), "--noise-model", "none", "--t-inference", "0"]
        for occ in a.occupancy:
            wmax = r.level / occ
            for dev, s, t in cl:
                yield pdir / f"{stem_of(dev, s, t, occ)}.csv", \
                    base + ["--w-max", repr(round(wmax, 6)), "--noise-model", dev, "--t-inference", fmt(t),
                            "--noise-scale", fmt(s)]


def run_one(part: Path, cmd: list[str], timeout: int):
    if part.is_file():
        return part, True, ""
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_name(part.name + ".tmp")
    env = dict(os.environ, OMP_NUM_THREADS="1")
    try:
        with open(tmp, "w") as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return part, False, f"timeout {timeout}s"
    if rc.returncode != 0:
        tmp.unlink(missing_ok=True)
        return part, False, rc.stderr[-300:]
    os.replace(tmp, part)
    return part, True, ""


# --------------------------------------------------------------------------------------------
# Consolidación
# --------------------------------------------------------------------------------------------
def boot_ci(x: np.ndarray, rng, n: int = 10000):
    idx = rng.integers(0, len(x), size=(n, len(x)))
    return np.percentile(x[idx].mean(axis=1), [2.5, 97.5])


def summarize(task: str, nets: pd.DataFrame, a, root: Path):
    floor = 0.0
    pj = data_dir(task) / "protocol.json"
    if pj.is_file():
        floor = float(json.loads(pj.read_text()).get("ret_floor", 0.0))
    rows, missing = [], 0
    for r in nets.itertuples():
        pdir = root / "parts" / f"s{r.seed:02d}_L{fmt(r.level)}"
        base_f = pdir / f"{stem_of('none', 0, 0, None)}.csv"
        if not base_f.is_file():
            missing += 1
            continue
        base = pd.read_csv(base_f)["reward_original"].mean()
        for occ in a.occupancy:
            for dev, s, t in cells(a.times):
                p = pdir / f"{stem_of(dev, s, t, occ)}.csv"
                if not p.is_file():
                    missing += 1
                    continue
                df = pd.read_csv(p)
                ret = df.groupby(["seed", "chip"])["reward_original"].mean().mean()     # media por chip y luego global
                pct = (ret - floor) / (base - floor) * 100.0 if base > floor else np.nan
                rows.append(dict(task=task, seed=r.seed, level=r.level, model=dev, scale=s, t=t, occ=occ,
                                 wmax=r.level / occ, ret_abs=ret, base=base, floor=floor, ret=pct,
                                 weight_rmse=df["weight_rmse"].mean()))
    if not rows:
        print(f"[{task}] no hay resultados que consolidar en {root / 'parts'}")
        return None
    net = pd.DataFrame(rows)
    net.to_csv(root / "networks.csv", index=False)
    print(f"[{task}] {len(net)} filas red×peso×celda → {root / 'networks.csv'}")
    if a.weights == "all":
        print(f"    (--weights all: sin summary/tradeoff; usá eval_results/pareto_wmax.py --task {task})")
        return None
    rng = np.random.default_rng(0)
    out = []
    for (m, s, t, o), g in net.groupby(["model", "scale", "t", "occ"]):
        x = g.ret.dropna().to_numpy()
        if len(x) == 0:
            continue
        lo, hi = boot_ci(x, rng)
        out.append(dict(model=m, scale=s, t=t, occ=o, n=len(x), ret=x.mean(), sd=x.std(ddof=1) if len(x) > 1 else np.nan,
                        lo=lo, hi=hi, median=float(np.median(x)), ret_abs=g.ret_abs.mean()))
    res = pd.DataFrame(out)
    res.to_csv(root / "summary.csv", index=False)
    # Ahorro: ocupación MÍNIMA (la conductancia programada es ∝ ocupación, y el consumo estático ∝ conductancia) con la
    # que cada red conserva >= thr % de retención; la mediana sobre redes cuenta las que nunca llegan como "no alcanzada".
    tr = []
    for (m, s_, t), g in net.groupby(["model", "scale", "t"]):
        for thr in THRESHOLDS:
            mins = []
            for _, gg in g.groupby("seed"):
                ok = gg[gg.ret >= thr].occ
                mins.append(ok.min() if len(ok) else np.inf)
            mins = np.array(mins)
            fin = mins[np.isfinite(mins)]
            tr.append(dict(model=m, scale=s_, t=t, thr=thr, n=len(mins), n_ok=len(fin),
                           min_occ_median=float(np.median(mins)), min_occ_q25=float(np.percentile(mins, 25)),
                           min_occ_q75=float(np.percentile(mins, 75)),
                           saving_median=float(1 - np.median(mins)) if np.isfinite(np.median(mins)) else np.nan))
    pd.DataFrame(tr).to_csv(root / "tradeoff.csv", index=False)
    exp = len(nets) * (1 + len(a.occupancy) * len(cells(a.times)))
    print(f"[{task}] {len(net)} filas red×celda de {exp - len(nets)} esperadas"
          f"{f' ({missing} partes faltan)' if missing else ''}; escrito {root / 'summary.csv'}")
    # resumen corto en consola: CMO y PCM a 1 año, por ocupación
    for m in DEVICES:
        sub = res[(res.model == m) & (res.scale == 1.0) & np.isclose(res.t, YEAR)].sort_values("occ")
        if len(sub):
            print(f"    {m:9s} escala 1, 1 año: " + "  ".join(f"{r.occ * 100:g}% → {r.ret:5.1f} [{r.lo:5.1f},{r.hi:5.1f}]"
                                                              for r in sub.itertuples()))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", nargs="+", choices=list(TASKS) + ["all"], default=["all"])
    ap.add_argument("--weights", choices=["best", "odin", "all"], default="best",
                    help="best: peso compartido ganador de cada red (WEIGHT_VALS); odin: mejor nivel entero 1..7; "
                         "all: los 6 pesos de WEIGHT_VALS por red (para el Pareto)")
    ap.add_argument("--occupancy", type=float, nargs="+", default=OCCUPANCIES,
                    help="fracción de g_max que ocupa el peso (w_max = nivel / ocupación); def.: 0.0625 0.125 0.25 0.5 1.0")
    ap.add_argument("--times", type=float, nargs="+", default=TIMES, help="tiempos (s) desde la programación")
    ap.add_argument("--n-chips", type=int, default=30)
    ap.add_argument("--reads-per-chip", type=int, default=1)
    ap.add_argument("--reramcmo-g-max", type=float, default=25.0)
    ap.add_argument("--reramcmo-g-min", type=float, default=1.0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 8, help="procesos simultáneos (1 hebra OpenMP c/u)")
    ap.add_argument("--timeout", type=int, default=3600, help="segundos máximos por proceso")
    ap.add_argument("--out-dir", default="", help="raíz de salida (def.: eval_results/); crea <tarea>_wmax_check/")
    ap.add_argument("--dry-run", action="store_true", help="muestra plan y niveles, no simula")
    ap.add_argument("--summary-only", action="store_true", help="solo consolidar lo ya calculado")
    a = ap.parse_args()
    tasks = list(TASKS) if "all" in a.task else a.task
    a.times = sorted(set(a.times))
    a.occupancy = sorted(set(a.occupancy))
    if any(o <= 0 or o > 1 for o in a.occupancy):
        sys.exit("Error: --occupancy debe estar en (0, 1].")

    plan = {}
    for task in tasks:
        nets = pick_networks(task, a.weights)
        root = out_dir(task, a)
        root.mkdir(parents=True, exist_ok=True)
        nets.to_csv(root / ("levels_all.csv" if a.weights == "all" else "levels.csv"), index=False)
        jobs = list(jobs_for(task, nets, a, root))
        todo = [j for j in jobs if not j[0].is_file()]
        plan[task] = (nets, root, jobs, todo)
        lv = (f"{nets.seed.nunique()} redes × pesos {sorted(nets.level.unique())}" if a.weights == "all" else
              f"{len(nets)} redes; pesos elegidos (sin ruido): "
              + " ".join(f"s{r.seed:02d}:w{fmt(r.level)}" for r in nets.itertuples()))
        print(f"[{task}] {lv}"
              + f"\n        {len(jobs)} procesos ({len(jobs) - len(todo)} ya hechos, {len(todo)} por hacer) "
              f"→ {root}", flush=True)
    if a.dry_run:
        print(f"\n(dry-run) ocupaciones {[f'{o * 100:g}%' for o in a.occupancy]}, {len(a.times)} tiempos, "
              f"dispositivos {DEVICES}; ejemplo de comando:")
        task = tasks[0]
        print("  " + " ".join(plan[task][2][1][1]))
        return

    failed = []
    if not a.summary_only:
        for task in tasks:
            nets, root, jobs, todo = plan[task]
            if not todo:
                continue
            t0 = time.time()
            print(f"\n[{task}] lanzando {len(todo)} procesos (hasta {a.jobs} a la vez) …", flush=True)
            done = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=a.jobs) as ex:
                futs = [ex.submit(run_one, p, c, a.timeout) for p, c in todo]
                for f in concurrent.futures.as_completed(futs):
                    part, ok, err = f.result()
                    done += 1
                    if not ok:
                        failed.append((task, part, err))
                        print(f"[FALLÓ] {part}: {err}", flush=True)
                    if done % 200 == 0 or done == len(todo):
                        print(f"  [{task}] {done}/{len(todo)}  ({(time.time() - t0) / 60:.1f} min)", flush=True)
    for task in tasks:
        nets, root, _, _ = plan[task]
        summarize(task, nets, a, root)
    if failed:
        for task in {f[0] for f in failed}:
            with open(out_dir(task, a) / "failed_jobs.txt", "w") as fh:
                fh.writelines(f"{p}\t{e.strip()}\n" for t, p, e in failed if t == task)
        sys.exit(f"{len(failed)} procesos fallaron (ver failed_jobs.txt); relanzá el mismo comando para completarlos.")


if __name__ == "__main__":
    main()
