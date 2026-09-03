#!/usr/bin/env python3
"""
Poda neuronas ocultas "muertas" de un modelo WANN+SNN exportado (_best.out)
y guarda el resultado como una segunda versión del archivo.

Una neurona oculta se considera muerta cuando ninguna de sus conexiones
entrantes activas es excitatoria (peso > 0), es decir, nunca puede cruzar
el umbral de disparo del modelo de Izhikevich partiendo de reposo (este
proyecto crea las neuronas con i_offset=0, noise=0, ver neuron.cpp) y por
lo tanto nunca emite spikes. Al no disparar nunca, sus conexiones salientes
no tienen ningún efecto sobre el resto de la red, así que se puede eliminar
sin cambiar el comportamiento expresado por el genoma.

La detección es iterativa (punto fijo): al quitar una neurona muerta puede
quedar otra neurona con solo entradas inhibitorias/nulas, que antes parecía
viva porque su única fuente "excitatoria" era, en realidad, una neurona que
nunca disparaba.

Advertencia importante: la regla es SOLO estructural (mira el signo de los
pesos) y no simula la dinámica. Para neuronas LOW_THRESHOLD_SPIKING (act=4)
existe "rebound spiking": el modelo de Izhikevich con b=0.25 puede disparar
al liberarse de una hiperpolarización fuerte, así que una entrada puramente
inhibitoria no garantiza que esté muerta. RESONATOR (act=6) también tiene
dinámica subumbral no trivial. Por eso, por defecto, este script NO poda esos
tipos de neurona aunque cumplan la regla: los reporta como "candidatas, no
podadas" para que se verifiquen empíricamente (por ejemplo comparando el
fitness del modelo original vs. el podado con el binario de evaluación
correspondiente, p.ej. wann_eval_car). Usar --include-rebound-prone para
forzar su poda si se acepta ese riesgo.

Uso:
    python prune_dead_neurons.py --prefix log/snn_car --nInput 9 --nOutput 2
    python prune_dead_neurons.py --in log/snn_car_best.out --nInput 9 --nOutput 2
    python prune_dead_neurons.py --prefix log/snn_car --config p/car_snn.json
    python prune_dead_neurons.py --prefix log/snn_car --config p/car_snn.json --dry-run
"""

import argparse
import json
import math
import os
import sys

ACT_NAMES = {
    1: "REGULAR_SPIKING", 2: "FAST_SPIKING", 3: "CHATTERING",
    4: "LOW_THRESHOLD_SPIKING", 5: "INTRINSICALLY_BURSTING", 6: "RESONATOR",
    7: "FAST_SPIKING", 8: "REGULAR_SPIKING", 9: "REGULAR_SPIKING",
    10: "CHATTERING",
}
ACT_SHORT = {1: "RS", 2: "FS", 3: "CH", 4: "LTS", 5: "IB", 6: "RES",
             7: "FS", 8: "RS", 9: "RS", 10: "CH"}

# Tipos con dinámica que puede disparar sin entrada excitatoria neta
# (post-inhibitory rebound / subumbral). Ver docstring del módulo.
REBOUND_PRONE_IDS = {4, 6}


def parse_net(path):
    """Lee un archivo _best.out (N filas x (N+1) columnas).

    Devuelve:
      tokens: lista de N filas, cada una con N+1 strings crudos (tal cual
              estaban en el archivo, para poder reescribirlo sin pérdida
              de precisión ni reformateos).
      W:      matriz N x N de floats (nan donde la conexión está deshabilitada
              o no aplica).
      act:    lista de N ids de activación (int).
    """
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [t.strip() for t in line.split(",")]
            tokens.append(row)

    N = len(tokens)
    for r, row in enumerate(tokens):
        if len(row) != N + 1:
            raise ValueError(
                f"{path}: fila {r} tiene {len(row)} columnas, se esperaban "
                f"{N + 1} (N={N} pesos + 1 activación)")

    def to_float(tok):
        return float("nan") if tok.lower() == "nan" else float(tok)

    W = [[to_float(tokens[r][c]) for c in range(N)] for r in range(N)]
    act = [int(round(to_float(tokens[r][N]))) for r in range(N)]
    return tokens, W, act


def find_dead_hidden(alive, hidden_idx, W, act, include_rebound_prone, max_iter):
    """Punto fijo: devuelve (removed, flagged, rounds).

    removed: lista de (idx, ronda) de neuronas ocultas eliminadas.
    flagged: lista de idx de neuronas rebound-prone que cumplían la regla
             pero se dejaron (no podadas) porque son rebound-prone y
             include_rebound_prone=False.
    """
    removed = []
    flagged = set()

    for rnd in range(1, max_iter + 1):
        candidates = []
        for h in hidden_idx:
            if h not in alive:
                continue
            has_excitatory_in = False
            for i in alive:
                if i == h:
                    continue
                w = W[i][h]
                if not math.isnan(w) and w > 0.0:
                    has_excitatory_in = True
                    break
            if not has_excitatory_in:
                candidates.append(h)

        if not candidates:
            return removed, sorted(flagged), rnd - 1

        pruned_this_round = False
        for h in candidates:
            if act[h] in REBOUND_PRONE_IDS and not include_rebound_prone:
                flagged.add(h)
                continue
            alive.discard(h)
            removed.append((h, rnd))
            flagged.discard(h)
            pruned_this_round = True

        if not pruned_this_round:
            # Todos los candidatos restantes son rebound-prone y no se
            # incluyen: no hay más para hacer.
            return removed, sorted(flagged), rnd

    return removed, sorted(flagged), max_iter


def main():
    ap = argparse.ArgumentParser(
        description="Poda neuronas ocultas muertas (solo-inhibitorias) de un "
                    "modelo WANN+SNN exportado, y guarda una 2da versión.")
    ap.add_argument("--in", dest="in_path", default=None,
                    help="Ruta al archivo .out de origen")
    ap.add_argument("--prefix", default=None,
                    help="Alternativa a --in: usa <prefix>_best.out (mismo "
                         "convenio que graph_network.py)")
    ap.add_argument("--out", dest="out_path", default=None,
                    help="Ruta de salida (default: <entrada>_pruned.out)")
    ap.add_argument("--nInput", type=int, default=None,
                    help="Número de entradas del WANN (sin bias)")
    ap.add_argument("--nOutput", type=int, default=None,
                    help="Número de salidas del WANN")
    ap.add_argument("--config", default=None,
                    help="JSON de hiperparámetros (p.ej. p/car_snn.json) del "
                         "que leer ann_nInput / ann_nOutput si no se pasan "
                         "explícitamente")
    ap.add_argument("--include-rebound-prone", action="store_true",
                    help="También podar neuronas LOW_THRESHOLD_SPIKING/"
                         "RESONATOR que cumplan la regla estática (riesgo: "
                         "pueden disparar por rebound aunque solo reciban "
                         "inhibición). Por defecto se dejan y solo se "
                         "reportan como candidatas.")
    ap.add_argument("--max-iter", type=int, default=50,
                    help="Máximo de rondas del punto fijo (default: 50)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo reporta, no escribe el archivo de salida")
    args = ap.parse_args()

    if not args.in_path and not args.prefix:
        ap.error("pasá --in <archivo.out> o --prefix <prefijo>")
    in_path = args.in_path or (args.prefix + "_best.out")

    n_input, n_output = args.nInput, args.nOutput
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        n_input = n_input if n_input is not None else cfg.get("ann_nInput")
        n_output = n_output if n_output is not None else cfg.get("ann_nOutput")
    if n_input is None or n_output is None:
        ap.error("faltan --nInput/--nOutput (pasalos directo o vía --config)")

    if not os.path.exists(in_path):
        print(f"No se encontró {in_path}", file=sys.stderr)
        sys.exit(1)

    tokens, W, act = parse_net(in_path)
    N = len(tokens)

    if n_input + 1 + n_output > N:
        print(f"nInput/nOutput inconsistentes con N={N} filas del archivo "
              f"(bias+inputs={n_input + 1} + outputs={n_output} > N={N})",
              file=sys.stderr)
        sys.exit(1)
    input_idx = set(range(0, n_input + 1))       # bias (0) + inputs
    output_idx = set(range(N - n_output, N))
    hidden_idx = set(range(n_input + 1, N - n_output))

    alive = set(range(N))
    removed, flagged, rounds = find_dead_hidden(
        alive, hidden_idx, W, act, args.include_rebound_prone, args.max_iter)

    kept = sorted(alive)
    n_hidden_before = len(hidden_idx)
    n_hidden_after = len(hidden_idx) - len(removed)

    print(f"Entrada:  {in_path}  (N={N}, hidden={n_hidden_before})")
    print(f"Punto fijo: {rounds} ronda(s)")
    if removed:
        print(f"\nNeuronas ocultas eliminadas ({len(removed)}):")
        for idx, rnd in removed:
            print(f"  idx={idx:4d}  act={act[idx]:2d} ({ACT_SHORT.get(act[idx],'?')})"
                  f"  ronda={rnd}")
    else:
        print("\nNo se encontraron neuronas ocultas muertas.")

    if flagged:
        print(f"\nCandidatas NO podadas por ser rebound-prone ({len(flagged)}):")
        for idx in flagged:
            print(f"  idx={idx:4d}  act={act[idx]:2d} ({ACT_SHORT.get(act[idx],'?')})"
                  f"  -> verificar empíricamente (rebound spiking) o correr "
                  f"con --include-rebound-prone si se acepta el riesgo")

    print(f"\nHidden: {n_hidden_before} -> {n_hidden_after}   "
          f"Nodos totales: {N} -> {len(kept)}")

    if args.dry_run:
        print("\n(--dry-run) no se escribió ningún archivo.")
        return

    out_path = args.out_path
    if not out_path:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_pruned{ext}"

    with open(out_path, "w") as f:
        for r in kept:
            row = [tokens[r][c] for c in kept] + [tokens[r][N]]
            f.write(",".join(row) + "\n")

    print(f"\nGuardado: {out_path}")
    print("Recordá verificar el resultado (p.ej. comparando el fitness "
          "reportado por el binario de evaluación de la tarea con el .out "
          "original vs. el podado) antes de descartar el original.")


if __name__ == "__main__":
    main()
