#!/usr/bin/env python3
"""
Orquesta los barridos de ruido de memristores de wann_car_noise (ver
BINARIOS.md §4d) y genera las figuras correspondientes de una sola vez:

  1. Barrido de INTENSIDAD (--noise-scale, t_inference fijo) ->
     plot_car_noise.py -> "<prefix>_noise_robustness.png"
  2. Barrido de TIEMPO (--t-inference, noise_scale fijo en 1.0, sin el
     modelo "simple" porque no depende del tiempo) ->
     plot_car_noise_time.py -> "<prefix>_time_robustness.png"
  3. Con --grid: una grilla completa escala×tiempo (pcm/reram/reramcmo, no
     "simple") -> plot_car_noise_heatmap.py -> "<prefix>_heatmap_<modelo>.png"

Solo evalúa episodios con una red ya entrenada (vía wann_car_noise) — nunca
entrena. Requiere el binario compilado: cd build && ninja wann_car_noise (o
./build.sh).

--noise-scale es un escalar por invocación del binario, así que el barrido de
intensidad hace una llamada por (modelo, valor de escala). --t-inference en
cambio acepta una lista separada por comas en UNA sola invocación, así que el
barrido de tiempo (y cada fila de la grilla) hace solo una llamada por
(modelo, escala) con la lista completa de tiempos.

Con --weight-indices (en vez de --weight-index) se repiten los barridos de
arriba para cada peso compartido de la lista, concatenados en el mismo CSV
(la columna `weight` ya los distingue) — la premisa de WANN es desempeño
estable *a través* de pesos, así que vale la pena ver si eso también se
sostiene bajo ruido, no solo para un peso fijo.

Uso:
  python eval_results/run_car_noise_sweeps.py
  python eval_results/run_car_noise_sweeps.py --noise-scales 0,1,2,3,4,5 --jobs 8
  python eval_results/run_car_noise_sweeps.py --weight-indices 1,3,5
  python eval_results/run_car_noise_sweeps.py --grid --noise-scales 0,1,2,3,4,5
  python eval_results/run_car_noise_sweeps.py --net log/otra_red.out \
      --weight-index 2 --prefix otra_red
  python eval_results/run_car_noise_sweeps.py --skip-plots   # solo genera los CSV
"""
import argparse
import builtins
import concurrent.futures
import functools
import io
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

# Sin esto, el orden de los mensajes de progreso sale mezclado al capturar
# stdout de un proceso que a su vez lanza subprocesos (el buffering por
# bloques de Python cuando stdout no es una terminal).
print = functools.partial(builtins.print, flush=True)

# Ancla CWD a la raíz del repo — mismo patrón que eval_p3_weights.py y
# summarize_car_noise.py.
os.chdir(Path(__file__).resolve().parent.parent)

NOISE_MODELS = ["simple", "pcm", "reram", "reramcmo"]
BINARY = "./build/wann_car_noise"


def base_cmd(args, weight_index: int) -> list[str]:
    cmd = [BINARY, "-f", args.net, "-d", args.config,
           "--weight-index", str(weight_index),
           "--mode", args.mode,
           "--n-chips", str(args.n_chips),
           "--reads-per-chip", str(args.reads_per_chip),
           "--reramcmo-g-max", str(args.reramcmo_g_max),
           "--reramcmo-g-min", str(args.reramcmo_g_min),
           "--seeds", args.seeds]
    if args.override:
        cmd += ["-p", args.override]
    return cmd


def run_one(label: str, cmd: list[str], timeout: int) -> tuple[str, str]:
    """Corre un wann_car_noise y devuelve (label, stdout crudo)."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        sys.exit(f"[TIMEOUT] {label}: {' '.join(cmd)}")
    if proc.returncode != 0:
        sys.exit(f"[FAIL] {label}\n$ {' '.join(cmd)}\n{proc.stderr}")
    return label, proc.stdout


def combine_csv(outputs: list[tuple[str, str]], out_path: Path) -> None:
    """Concatena varios stdout de wann_car_noise (cada uno con su propio
    encabezado) en un solo CSV, imprimiendo progreso a medida que llegan."""
    parts = []
    for label, stdout in outputs:
        print(f"  ok: {label} ({len(stdout.splitlines()) - 1} filas)")
        parts.append(stdout)
    df = pd.concat([pd.read_csv(io.StringIO(p)) for p in parts], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Guardado: {out_path} ({len(df)} filas)")


def run_many(jobs_list: list[tuple[str, list[str]]], jobs: int, timeout: int,
             dry_run: bool) -> list[tuple[str, str]]:
    if dry_run:
        for label, cmd in jobs_list:
            print(f"[dry-run] {label}: {' '.join(cmd)}")
        return []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(lambda lc: run_one(lc[0], lc[1], timeout), jobs_list))


def weight_indices(args) -> list[int]:
    if args.weight_indices:
        return [int(w) for w in args.weight_indices.split(",")]
    return [args.weight_index]


def intensity_sweep(args) -> list[tuple[str, list[str]]]:
    jobs_list = []
    for wi in weight_indices(args):
        for model in NOISE_MODELS:
            for scale in args.noise_scales.split(","):
                cmd = base_cmd(args, wi) + ["--noise-model", model, "--noise-scale", scale]
                if model == "reram":
                    cmd += ["--t-inference", "1"]
                elif model == "reramcmo":
                    # t_inference=0 es un bypass total en este modelo (ver
                    # memristor_noise.hpp) — 1s para ejercitar deriva+lectura.
                    cmd += ["--t-inference", "1"]
                jobs_list.append((f"intensity wi={wi} model={model} scale={scale}", cmd))
    return jobs_list


# Modelos con eje de tiempo continuo (cualquier t_inference sirve) — ReRAM
# Wan2022 queda afuera de esta lista porque solo acepta sus 3 claves exactas,
# se maneja aparte en time_sweep()/grid_sweep().
CONTINUOUS_TIME_MODELS = ["pcm", "reramcmo"]


def time_sweep(args) -> list[tuple[str, list[str]]]:
    # "simple" no depende de t_inference (no modela deriva) — se excluye,
    # igual que en plot_car_noise_time.py.
    jobs_list = []
    for wi in weight_indices(args):
        for model in CONTINUOUS_TIME_MODELS:
            times = args.time_points_pcm if model == "pcm" else args.time_points_reramcmo
            jobs_list.append((f"time wi={wi} model={model}", base_cmd(args, wi) +
                               ["--noise-model", model, "--noise-scale", "1.0",
                                "--t-inference", times]))
        jobs_list.append((f"time wi={wi} model=reram", base_cmd(args, wi) +
                           ["--noise-model", "reram", "--noise-scale", "1.0",
                            "--t-inference", args.time_points_reram]))
    return jobs_list


def grid_sweep(args) -> list[tuple[str, list[str]]]:
    # Grilla completa escala×tiempo: una llamada por (modelo, escala), cada
    # una con la LISTA completa de tiempos de ese modelo (--t-inference
    # acepta CSV) — no una por (modelo,escala,tiempo), que sería N veces más
    # invocaciones para el mismo dato.
    jobs_list = []
    for wi in weight_indices(args):
        model_times = [("pcm", args.time_points_pcm), ("reram", args.time_points_reram),
                        ("reramcmo", args.time_points_reramcmo)]
        for model, times in model_times:
            for scale in args.noise_scales.split(","):
                cmd = base_cmd(args, wi) + ["--noise-model", model, "--noise-scale", scale,
                                            "--t-inference", times]
                jobs_list.append((f"grid wi={wi} model={model} scale={scale}", cmd))
    return jobs_list


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="log/full_p3_car_ttfs_first_spike/ttfs_first_spike_pruned.out")
    ap.add_argument("--config", default="p/car_snn.json")
    ap.add_argument("--override", "-p", default=None)
    ap.add_argument("--weight-index", type=int, default=5)
    ap.add_argument("--weight-indices", default=None,
                     help="CSV de índices (0..5) — si se da, reemplaza --weight-index y repite "
                          "cada barrido para cada uno, concatenados (columna 'weight' los distingue)")
    ap.add_argument("--mode", default="per-synapse", choices=["shared", "per-synapse"])
    ap.add_argument("--n-chips", type=int, default=30)
    ap.add_argument("--reads-per-chip", type=int, default=1)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--noise-scales", default="0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0",
                     help="CSV de valores de --noise-scale (una llamada al binario por valor)")
    ap.add_argument("--time-points-pcm",
                     default="0,60,300,3600,21600,86400,604800,2592000,31536000",
                     help="CSV de --t-inference para PCM (ley continua: cualquier valor sirve). "
                          "Grilla log de 60 s a 1 año, más el 0 como referencia 'recién programado' "
                          "(convención de Le Gallo et al. 2023 para evaluar drift de aihwkit)")
    ap.add_argument("--time-points-reram", default="1,86400,172800",
                     help="CSV de --t-inference para ReRAM (debe ser EXACTAMENTE estas 3 claves)")
    ap.add_argument("--time-points-reramcmo",
                     default="0,60,300,3600,21600,86400,604800,2592000,31536000",
                     help="CSV de --t-inference para ReRAM CMO/HfOx (deriva continua, igual que PCM; "
                          "t_inference=0 es un bypass total en este modelo, ver memristor_noise.hpp)")
    ap.add_argument("--reramcmo-g-max", type=float, default=25.0,
                     help="g_max (µS) de ReRAM CMO/HfOx — sin default confiable en aihwkit, "
                          "pasar mediciones reales si se tienen")
    ap.add_argument("--reramcmo-g-min", type=float, default=1.0,
                     help="g_min (µS) de ReRAM CMO/HfOx — sin default confiable en aihwkit; "
                          "al ser > 0, la rama 'apagada' de cada par diferencial siempre queda "
                          "flotando en al menos g_min, aun sin ruido")
    ap.add_argument("--prefix", default="car_ttfs_first_spike",
                     help="prefijo para los CSV (log/) y PNG (graficos/Racing Car/)")
    ap.add_argument("--log-dir", default="log")
    ap.add_argument("--graficos-dir", default="graficos/Racing Car")
    ap.add_argument("--jobs", type=int, default=4, help="invocaciones de wann_car_noise en paralelo")
    ap.add_argument("--timeout", type=int, default=1800, help="segundos por invocación")
    ap.add_argument("--grid", action="store_true",
                     help="además, corre la grilla completa escala×tiempo y genera los heatmaps")
    ap.add_argument("--skip-intensity", action="store_true")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-plots", action="store_true", help="solo generar los CSV, no las figuras")
    ap.add_argument("--dry-run", action="store_true", help="solo imprimir los comandos, no correr nada")
    args = ap.parse_args()

    if not args.dry_run and not Path(BINARY).is_file():
        sys.exit(f"Error: no existe {BINARY}. Compilar primero: cd build && ninja wann_car_noise "
                  f"(o ./build.sh).")

    noise_csv = Path(args.log_dir) / f"{args.prefix}_noise_sweep.csv"
    time_csv = Path(args.log_dir) / f"{args.prefix}_time_sweep.csv"
    grid_csv = Path(args.log_dir) / f"{args.prefix}_grid_sweep.csv"
    noise_png = Path(args.graficos_dir) / f"{args.prefix}_noise_robustness.png"
    time_png = Path(args.graficos_dir) / f"{args.prefix}_time_robustness.png"

    if not args.skip_intensity:
        print("== Barrido de intensidad (--noise-scale) ==")
        outputs = run_many(intensity_sweep(args), args.jobs, args.timeout, args.dry_run)
        if outputs:
            combine_csv(outputs, noise_csv)

    if not args.skip_time:
        print("== Barrido de tiempo (--t-inference) ==")
        outputs = run_many(time_sweep(args), args.jobs, args.timeout, args.dry_run)
        if outputs:
            combine_csv(outputs, time_csv)

    if args.grid:
        print("== Grilla escala×tiempo ==")
        outputs = run_many(grid_sweep(args), args.jobs, args.timeout, args.dry_run)
        if outputs:
            combine_csv(outputs, grid_csv)

    if args.dry_run or args.skip_plots:
        return

    print("== Figuras ==")
    title = f"{args.prefix} ({Path(args.net).stem}, {args.mode})"
    if noise_csv.is_file():
        subprocess.run([sys.executable, "eval_results/plot_car_noise.py",
                         "--csv", str(noise_csv), "--out", str(noise_png),
                         "--title", title], check=True)
    if time_csv.is_file():
        subprocess.run([sys.executable, "eval_results/plot_car_noise_time.py",
                         "--csv", str(time_csv), "--out", str(time_png),
                         "--title", title], check=True)
    if args.grid and grid_csv.is_file():
        for model in ("pcm", "reram", "reramcmo"):
            heatmap_png = Path(args.graficos_dir) / f"{args.prefix}_heatmap_{model}.png"
            subprocess.run([sys.executable, "eval_results/plot_car_noise_heatmap.py",
                             "--csv", str(grid_csv), "--noise-model", model,
                             "--out", str(heatmap_png), "--title", title], check=True)


if __name__ == "__main__":
    main()
