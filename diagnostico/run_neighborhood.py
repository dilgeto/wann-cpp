#!/usr/bin/env python3
"""Lanza wann_car_neighborhood sobre los snapshots de varias corridas.

Para cada corrida (directorio ``<prefix>_snap`` con ``gen_XXXXX.json``) y cada
generación pedida elige el snapshot más cercano y analiza su élite. Guarda cada
análisis en ``<out-dir>/<corrida>_g<gen>_*.csv`` (los lee neighborhood_summary.py).

POR DEFECTO NO SIMULA NADA: solo imprime el plan. Modos:
  (sin flags)   imprime qué se analizaría.
  --estimate    corre el binario con --dry-run (enumera vecinos, NO simula) y suma
                los episodios que costaría el lote. Barato: úsalo antes de --run.
  --run         ejecuta el análisis (simula; es un lote de cómputo real).

Uso típico:
  python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank00/seed*_snap' \\
      --override diag/rank00.json --gens 25,100,250,last --estimate
  python diagnostico/run_neighborhood.py ... --run --jobs 3 --omp 60 \\
      --extra "--n2-mids 20 --n2-if-stuck"

-d/-p (--base/--override) deben ser los MISMOS del entrenamiento. Los análisis ya
hechos (existe <prefijo>_parents.csv) se saltan salvo con --force.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import diag_lib as dl

dl.anchor_to_repo_root()

SNAP_RE = re.compile(r"gen_(\d+)\.json$")
EST_RE = re.compile(r"vecinos N1:\s*(\d+)\s*\(~(\d+) episodios\)")


def pick_snapshots(snap_dir: Path, wanted: list[str]) -> list[tuple[int, Path]]:
    snaps = sorted((int(m.group(1)), p) for p in snap_dir.glob("gen_*.json")
                   if (m := SNAP_RE.search(p.name)))
    if not snaps:
        return []
    chosen: dict[int, Path] = {}
    for w in wanted:
        target = snaps[-1][0] if w == "last" else int(w)
        g, p = min(snaps, key=lambda s: (abs(s[0] - target), s[0]))
        chosen[g] = p
    return sorted(chosen.items())


def build_cmd(args, snap: Path, prefix: Path, dry: bool) -> list[str]:
    cmd = [args.binary, "-i", str(snap), "-d", args.base, "-o", str(prefix)]
    if args.override:
        cmd += ["-p", args.override]
    cmd += shlex.split(args.extra)
    if dry:
        cmd.append("--dry-run")
    return cmd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snap-glob", required=True, help="patrón de directorios *_snap")
    ap.add_argument("--runs", default="", help="solo estas corridas, separadas por coma (p. ej. seed08,seed12); "
                    "el nombre es el del directorio sin el sufijo _snap")
    ap.add_argument("--gens", default="25,100,250,last", help="generaciones a analizar ('last' = la última)")
    ap.add_argument("--base", default="p/car_snn.json", help="-d del binario")
    ap.add_argument("--override", default="", help="-p del binario (el JSON del entrenamiento)")
    ap.add_argument("--out-dir", default="log/diag_neighborhood")
    ap.add_argument("--binary", default="build/wann_car_neighborhood")
    ap.add_argument("--extra", default="", help="flags extra del binario, entre comillas")
    ap.add_argument("--jobs", type=int, default=1, help="análisis en paralelo")
    ap.add_argument("--omp", type=int, default=0, help="OMP_NUM_THREADS por análisis (default: núcleos // jobs)")
    ap.add_argument("--estimate", action="store_true", help="enumera con --dry-run y suma episodios")
    ap.add_argument("--run", action="store_true", help="ejecuta el análisis (simula)")
    ap.add_argument("--force", action="store_true", help="rehace análisis que ya existen")
    args = ap.parse_args()

    dirs = sorted(Path(p) for p in glob.glob(args.snap_glob) if Path(p).is_dir())
    if args.runs:
        wanted_runs = {r.strip() for r in args.runs.split(",") if r.strip()}
        names = {d.name[:-5] if d.name.endswith("_snap") else d.name: d for d in dirs}
        missing = sorted(wanted_runs - set(names))
        if missing:
            print(f"--runs: no encuentro {missing} entre {sorted(names)[:5]}...", file=sys.stderr)
            return 1
        dirs = [names[r] for r in sorted(wanted_runs)]
    if not dirs:
        print(f"Sin directorios para {args.snap_glob!r}", file=sys.stderr)
        return 1
    if not Path(args.binary).exists():
        print(f"No existe {args.binary}: compila con ./build.sh", file=sys.stderr)
        return 1
    wanted = [w.strip() for w in args.gens.split(",") if w.strip()]
    out_dir = Path(args.out_dir)

    jobs: list[tuple[str, int, Path, Path]] = []      # (corrida, gen, snapshot, prefijo)
    for d in dirs:
        run = d.name[:-5] if d.name.endswith("_snap") else d.name
        for g, snap in pick_snapshots(d, wanted):
            jobs.append((run, g, snap, out_dir / f"{run}_g{g:05d}"))
    if not jobs:
        print("Los directorios no contienen snapshots gen_*.json", file=sys.stderr)
        return 1

    todo = [j for j in jobs
            if args.force or not (Path(f"{j[3]}_parents.csv").exists()
                                  and Path(f"{j[3]}_parents.csv").stat().st_size > 0)]
    print(f"{len(dirs)} corridas x gens {wanted} -> {len(jobs)} análisis "
          f"({len(jobs) - len(todo)} ya hechos, {len(todo)} pendientes)")

    if args.estimate:
        total_ep = total_nb = 0
        for run, g, snap, prefix in todo:
            tmp = out_dir / "_estimate" / f"{run}_g{g:05d}"
            tmp.parent.mkdir(parents=True, exist_ok=True)
            res = subprocess.run(build_cmd(args, snap, tmp, dry=True),
                                 capture_output=True, text=True)
            m = EST_RE.search(res.stdout)
            if res.returncode != 0 or not m:
                print(f"  {run} g{g}: no se pudo estimar\n{res.stdout[-300:]}{res.stderr[-300:]}")
                continue
            nb, ep = int(m.group(1)), int(m.group(2))
            total_nb += nb
            total_ep += ep
            print(f"  {run} g{g:<5} {nb:>5} vecinos  ~{ep:>7} episodios")
        print(f"\nTotal pendiente: {total_nb} vecinos, ~{total_ep:,} episodios de cribado N1 "
              "(+ ~10% de confirmación por defecto, + N2 si lo activas con --extra).")
        print("Referencia: una generación de entrenamiento con popSize 480 son ~17,000 episodios.")
        return 0

    if not args.run:
        for run, g, snap, prefix in todo:
            print("  " + " ".join(shlex.quote(c) for c in build_cmd(args, snap, prefix, dry=False)))
        print("\nNo se ejecutó nada. Usa --estimate para costear (sin simular) o --run para lanzar.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    omp = args.omp or max(1, (os.cpu_count() or 1) // max(1, args.jobs))
    env = {**os.environ, "OMP_NUM_THREADS": str(omp)}
    print(f"Ejecutando con --jobs {args.jobs} x OMP_NUM_THREADS={omp}")

    def one(j) -> tuple[str, int, int]:
        run, g, snap, prefix = j
        with open(f"{prefix}.log", "w") as log:
            rc = subprocess.run(build_cmd(args, snap, prefix, dry=False),
                                stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        return run, g, rc

    failed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for i, fut in enumerate(concurrent.futures.as_completed([pool.submit(one, j) for j in todo]), 1):
            run, g, rc = fut.result()
            failed += rc != 0
            print(f"  [{i}/{len(todo)}] {run} g{g}: {'ok' if rc == 0 else f'FALLÓ (rc={rc}, ver {out_dir}/{run}_g{g:05d}.log)'}")
    print(f"Listo. Resume con: python diagnostico/neighborhood_summary.py --dir {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
