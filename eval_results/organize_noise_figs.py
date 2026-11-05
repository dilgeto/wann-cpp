#!/usr/bin/env python3
"""
Ordena las figuras de ruido de memristores que ya están sueltas en graficos/<Tarea>/ en subcarpetas por tag:

    graficos/<Tarea>/ruido/<tag>/<tarea>_<tag>_<figura>.png

El tag es el prefijo de la figura sin el nombre de la tarea (car_replicas_fs_boxplots.png -> tag `replicas_fs`).
Solo toca los grupos de ruido generados por run_noise_compare.py / wmax_check_car.py:
  wann_vs_sgrl*, replicas*, wmax_check*  (los nombres de archivo no cambian, así que siguen siendo autoexplicativos).
No toca las demás figuras del proyecto (entrenamiento, redes, bootstrap, ablación de neuronas, ni las
car_ttfs_first_spike_* del primer arnés de ruido). Por defecto SOLO MUESTRA el plan; --apply mueve de verdad.
Se niega a sobrescribir si el destino ya existe. Escribe un INDICE.md en cada graficos/<Tarea>/ruido/.

Uso:
  python eval_results/organize_noise_figs.py            # plan
  python eval_results/organize_noise_figs.py --apply    # mover
  python eval_results/organize_noise_figs.py --undo     # devolver todo a graficos/<Tarea>/ (plano)
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import figure_tag

TASK_DIRS = ["Racing Car", "Mountain Car", "Acrobot"]
GROUP = re.compile(r"^(car|disc_mc|acrobot)_(wann_vs_sgrl|replicas|wmax_check)")
DESC = {
    "wann_vs_sgrl": "Comparación original: mejor modelo de WANN (peso de WEIGHT_VALS) vs. mejor modelo de cada SGRL (64×64, mismas conexiones, mismas neuronas).",
    "wann_vs_sgrlq3": "Igual, con las versiones de SGRL cuantizadas a 3 bits (ODIN).",
    "replicas": "11 semillas por método; WANN con niveles ODIN enteros 1-7 tal cual (peso = nivel, w_max = 7); SGRL en float32.",
    "replicas_fs": "11 semillas por método; WANN en dos mapeos: nivel ODIN tal cual y fondo de escala; SGRL en float32.",
    "replicasq": "11 semillas; WANN (nivel ODIN tal cual) vs. SGRL cuantizado a 3 bits, con su versión float32 punteada.",
    "replicas_fsq": "11 semillas; WANN en dos mapeos vs. SGRL cuantizado a 3 bits.",
    "replicas_mean": "Réplicas con WANN promediado sobre sus niveles (sin elegir peso).",
    "wmax_check": "Control de mapeo en Car: misma red y peso, con el peso ocupando 12,5 / 25 / 50 / 100 % del rango de conductancia.",
}


def plan(base: Path):
    moves = []
    for td in TASK_DIRS:
        d = base / td
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix == ".png" and GROUP.match(f.name):
                stem = re.sub(r"_(intensity|boxplots|drift|ablation(_[a-z]+)?|quant_effect|noise_robustness|time_robustness|"
                              r"rmse_robustness|heatmap_[a-z]+|summary|occupancy|time)$", "", f.stem)
                moves.append((f, d / "ruido" / figure_tag(stem) / f.name))
    return moves


def write_index(base: Path):
    for td in TASK_DIRS:
        r = base / td / "ruido"
        if not r.is_dir():
            continue
        lines = [f"# Figuras de ruido de memristores — {td}", "",
                 "Una subcarpeta por tag (nombre = prefijo de las figuras sin la tarea).", ""]
        for t in sorted(p for p in r.iterdir() if p.is_dir()):
            n = len(list(t.glob("*.png")))
            lines.append(f"- **{t.name}/** ({n} figuras): {DESC.get(t.name, '(sin descripción)')}")
        (r / "INDICE.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, default=Path("graficos"))
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", action="store_true")
    a = ap.parse_args()
    if a.undo:
        moves = [(f, f.parents[2] / f.name) for td in TASK_DIRS for f in sorted((a.base / td / "ruido").glob("*/*.png"))]
    else:
        moves = plan(a.base)
    clashes = [(s, d) for s, d in moves if d.exists()]
    for s, d in moves:
        print(f"{'!! YA EXISTE ' if d.exists() else ''}{s.relative_to(a.base)}  ->  {d.relative_to(a.base)}")
    print(f"\n{len(moves)} archivos; {len(clashes)} conflictos.")
    if clashes:
        sys.exit("Hay destinos que ya existen: no se mueve nada.")
    if not a.apply and not a.undo:
        print("Solo plan (usa --apply para mover).")
        return
    for s, d in moves:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    if a.undo:
        for td in TASK_DIRS:                       # limpia las carpetas vacías y el índice
            r = a.base / td / "ruido"
            for x in sorted(r.glob("**/*"), reverse=True) if r.is_dir() else []:
                if x.name == "INDICE.md":
                    x.unlink()
                elif x.is_dir() and not any(x.iterdir()):
                    x.rmdir()
            if r.is_dir() and not any(r.iterdir()):
                r.rmdir()
    else:
        write_index(a.base)
    print("Hecho.")


if __name__ == "__main__":
    main()
