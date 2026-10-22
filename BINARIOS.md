# Cómo correr cada binario

Todos los binarios quedan en `build/` (`./build.sh` los compila; requiere
`../snn-simulator` ya compilado). Los comandos asumen que se ejecutan **desde la
raíz del repo**, porque los defaults (`p/*.json`, `log/...`) son rutas relativas.

Convenciones comunes:

- `-d` = config base (`p/<tarea>_snn.json`), `-p` = overrides encima de la base.
  `-p` acepta una ruta a un JSON **o** un JSON inline: `-p '{"maxGen":1}'`.
- Los binarios de entrenamiento escriben todo bajo `log/<prefix>_*` (`-o prefix`).
- El paralelismo sobre individuos usa OpenMP: `OMP_NUM_THREADS=N ./wann_car ...`.
- "Peso" = uno de los `alg_nVals` valores del peso compartido
  (`WEIGHT_VALS = {0.5, 1, 2, 3, 5, 8}`). `.wi` guarda el índice ganador.
- Cualquier gráfico/figura se genera con un script de Python (matplotlib, mismo
  estilo que `graph.py`/`generate_graphs.sh`, salida a `graficos/<Tarea>/`) —
  nunca como página HTML/Artifact — salvo que se pida explícitamente lo
  contrario (ver `CLAUDE.md`).

Terminología: "Mountain Car" a secas = discreto (`disc_mc`, `MountainCar-v0`).
El continuo es `mountain_car` (siempre "continuous Mountain Car").

---

## 1. Entrenamiento (evolución WANN)

Binarios: `wann_car`, `wann_acrobot`, `wann_disc_mc`, `wann_mountain_car`
(continuous), `wann_l2f`, `wann_snn` (Pendulum) y `wann_train` (sin SNN, stub).

```bash
./build/wann_<tarea> [-d base.json] [-p overrides.json|'{"k":v}'] [-o prefix] [-s seed] [-v]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-d` | Config base | `p/<tarea>_snn.json` (`wann_train`: `p/default_wan.json`) |
| `-p` | Overrides (archivo o JSON inline) | ninguno |
| `-o` | Prefijo de salida → `log/<prefix>_*` | `snn_<tarea>` (`wann_train`: `test`) |
| `-s` | Semilla | `42` |
| `-v` | Escribe `log/<prefix>_debug.log` (estado SNN del mejor cada `save_mod`). No existe en `wann_snn` ni `wann_train` | off |

`early_stop_patience` (en el JSON) solo funciona en `wann_car`.

`snapshot_interval` (en el JSON, default `0` = apagado, solo `wann_car`): si es
`> 0`, escribe `log/<prefix>_lineage.csv` (cada generación: padre, operador y
fitness de cada individuo) y, cada `snapshot_interval` generaciones más la
última, `log/<prefix>_snap/gen_XXXXX.json` con los genomas completos de la
población y sus recompensas. Es solo registro: no consume números aleatorios,
así que la misma semilla da la misma corrida con o sin él. Lo consume
`wann_car_neighborhood` (sección 4b).

### Reproducir un entrenamiento del screening (car, ttfs, first_spike)

Los hiperparámetros ganadores del screening están en
`screening_full/car_ttfs_first_spike/p3_configs/rank<RR>_seed<SS>.json`
(solo los 10 hiperparámetros afinados + `save_mod`, `snn_encoder`, `snn_decoder`).
El screening los aplica sobre `p/car_snn.json` con la semilla
`rank*10000 + seed_idx*100`:

```bash
OMP_NUM_THREADS=64 ./build/wann_car \
    -d p/car_snn.json \
    -p screening_full/car_ttfs_first_spike/p3_configs/rank00_seed00.json \
    -o mi_corrida -s 0
```

(`rank00_seed00` → semilla 0; `rank02_seed03` → semilla `20000 + 300 = 20300`.)

`screening_full.py` pasa además `--early-stop-patience 100` por defecto; esos
JSON no lo traen. Para replicarlo, mergéalo en un override:

```bash
python3 -c "import json;c=json.load(open('screening_full/car_ttfs_first_spike/p3_configs/rank00_seed00.json'));c['early_stop_patience']=100;json.dump(c,open('p/mi_override.json','w'),indent=2)"
./build/wann_car -d p/car_snn.json -p p/mi_override.json -o mi_corrida -s 0
```

### Archivos que produce (`log/`)

| Archivo | Contenido |
|---------|-----------|
| `<prefix>_stats.out` | Por generación: evals acumuladas, fitMed, fitMax, fitTop, fitPeak, nodeMed, connMed, fitTopOrig |
| `<prefix>_best.out` / `.wi` / `.gen` | Mejor red, índice de peso ganador, generación del guardado |
| `<prefix>_best/NNNN.out` | Snapshot del mejor por generación guardada |
| `<prefix>_pareto/NNNN.out` | Población (fitness, fitMax, nConn) para el frente de Pareto |
| `<prefix>_replay/gen_NNNN.csv` | Trayectoria del élite (solo `wann_car`) |
| `<prefix>_time.log` | Tiempo total / por generación (solo `wann_car`) |
| `<prefix>_mutstats.csv` | Por generación: qué mutación se eligió/aplicó y qué objetivo usó `probMoo` (solo `wann_car`) |
| `<prefix>_mutstats_summary.csv` | Totales y porcentajes de lo anterior (solo `wann_car`) |

---

## 2. Evaluación de la mejor red — `wann_<tarea>_eval`

Binarios: `wann_car_eval`, `wann_acrobot_eval`, `wann_disc_mc_eval`,
`wann_mountain_car_eval`, `wann_l2f_eval`.

```bash
./build/wann_car_eval [-f red.out] [-d config.json] [-w peso|best] [-n episodios] \
                      [-s seed] [-t steps] [-S best|N] [-o salida.csv] [-i nInput]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada (`.out`; el `.wi` se busca junto a ella) | `log/snn_<tarea>_best.out` |
| `-d` | Config | `p/<tarea>_snn.json` |
| `-w` | `best` = lee el `.wi` (o prueba todos los pesos si no existe); un número = peso fijo (se usa el índice más cercano) | `best` |
| `-n` | Nº de episodios | `10` |
| `-s` | Seed base | `0` |
| `-t` | Pasos máx. por episodio (**solo car**) | default de la tarea |
| `-S` | Guardar la trayectoria de un episodio: `best` (mejor shaped) o índice `N` | no guarda |
| `-o` | CSV de la trayectoria de `-S` | `<red>_ep<N>_replay.csv` |
| `-i` | Sobrescribe `ann_nInput` del config (**solo car, acrobot, disc_mc**) | del config |

Imprime cada episodio (shaped y original) y un resumen media/desv/min/max.

```bash
./build/wann_car_eval -f log/mi_corrida_best.out -d p/car_snn.json -n 30 -s 0 -S best
```

---

## 3. Replay de un episodio — `wann_car_replay`, `wann_l2f_replay`

Corre **un** episodio y exporta la trayectoria a CSV para visualizar.

```bash
./build/wann_car_replay [-f best.out] [-d config.json] [-w peso|best] [-s seed] [-o salida.csv]
./build/wann_l2f_replay [-f best.out] [-d config.json] [-w idx|best]  [-s seed] [-o salida.csv]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada | `log/snn_<car\|l2f>_best.out` |
| `-d` | Config | `p/<car\|l2f>_snn.json` |
| `-w` | `best` = evalúa todos los pesos y elige el mejor. En **car** un valor de peso (p.ej. `5.0`); en **l2f** un índice de `WEIGHT_VALS` | `best` |
| `-s` | Seed | `0` |
| `-o` | CSV de salida | `<red>_replay.csv` |

```bash
./build/wann_car_replay -f log/mi_corrida_best.out -d p/car_snn.json -w best -s 0
```

---

## 4. Barrido de pesos compartidos — `wann_eval_weights_<tarea>`

Binarios: `wann_eval_weights_acrobot`, `wann_eval_weights_car`,
`wann_eval_weights_disc_mc` (la tarea va fija en compilación). Imprime a stdout
un CSV con una fila por seed. Lo orquesta `eval_results/eval_p3_weights.py`.

```bash
./build/wann_eval_weights_car -f red.out -d p/car_snn.json --seeds 0,1,2,3 \
    [-p overrides.json] [--reward shaped|original] \
    [--episode-detail --weight-index N] [--nreps N]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada (**obligatorio**) | — |
| `-d` | Config (**obligatorio**) | — |
| `--seeds` | Lista de seeds separadas por coma (**obligatorio**) | — |
| `-p` | Overrides del config (archivo) | ninguno |
| `--reward` | `shaped` (como en entrenamiento) u `original` (sin shaping) | `shaped` |
| `--episode-detail` | En vez del promedio por (seed, peso), imprime cada episodio individual (`seed,episode,reward`). Requiere `--weight-index` | off |
| `--weight-index` | Índice de peso (≥ 0) para `--episode-detail` | — |
| `--nreps` | Sobrescribe `alg_nReps` solo en memoria | del config |
| `--ablate-inhibitory` | Ablación post-hoc: `w<0 → 0` (quita las sinapsis inhibitorias) | off |
| `--flip-inhibitory` | `w<0 → +|w|` (inhibitorias pasan a excitatorias, misma topología) | off |
| `--drop-random-exc K` | Quita K sinapsis excitatorias al azar (control de `--ablate-inhibitory` con K = nº de inhibitorias) | off |
| `--force-neuron-type N` | Todas las neuronas (entradas, bias, ocultas, salidas) pasan al tipo N (1..6) | off |
| `--shuffle-types` | Permuta los tipos entre neuronas (mismo histograma) | off |
| `--ablate-seed S` | Semilla de los dos controles aleatorios | 0 |

Las ablaciones solo transforman la red ya cargada (no tocan el archivo ni el
entrenamiento) y escriben un resumen en stderr. Las orquesta
`eval_results/ablation_study.py` (`check` / `run` / `analyze`): estudio de
ablación de polaridad y tipo de neuron sobre la red ganadora del bootstrap de
cada tarea, con las mismas seeds de episodio que el bootstrap (pareado).
Salidas en `eval_results/ablation/` y `graficos/<Tarea>/<run_key>_ablation*.png`.

---

## 4b. Análisis de vecindad — `wann_car_neighborhood`

**Solo evalúa, no entrena.** Dado un individuo, enumera todos los genomas a una
mutación de distancia (N1: cada `addConn`, `addNode`, `enable`, `mutAct` y
`toggleExcitatory` posible), los evalúa con las **mismas semillas que el padre**
y reporta cuántos mejoran, son neutros o empeoran, por operador. También da la
probabilidad exacta por mutación de caer en un vecino que mejora. Sirve para
distinguir "óptimo local estricto" de "hay vecinos mejores que la evolución no
encuentra".

```bash
./build/wann_car_neighborhood -i <snapshot.json | red_best.out> [-d config.json] [-p overrides]
    [-o prefijo] [--who elite|top:K|idx:N,N|all] [--seed S] [--max-per-op N]
    [--noise-seeds R] [--eps E] [--confirm K] [--confirm-seeds V]
    [--n2-mids M] [--n2-per-mid S] [--n2-if-stuck] [--climb K] [--rng-seed S] [--dry-run]
```

`-d`/`-p` deben ser **los mismos que en el entrenamiento** (probabilidades de
operadores, `ann_actRange`, parámetros SNN).

| Flag | Significado | Default |
|------|-------------|---------|
| `-i` | `*.json`: snapshot de `snapshot_interval` (genomas completos, semillas de entrenamiento). Otro: `*_best.out` (genoma **reconstruido**, aproximado; ver `GenomeIO.h`) | obligatorio |
| `--who` | Individuos del snapshot a analizar (`elite` = mayor fitness medio) | `elite` |
| `--seed` | Fuerza la semilla base de evaluación (individuo i: `S*10000+i`) | la del snapshot; `0` con `.out` |
| `--max-per-op` | Submuestrea N vecinos por operador (`0` = todos). `weight` en el CSV reescala las probabilidades | `0` |
| `--noise-seeds` | Re-evalúa el padre con R semillas para medir el ruido; sin `--eps`, `eps = 2·sd` | `4` |
| `--eps` | Umbral fijo: `|d_mean| <= E` es neutro | `2·sd` del ruido |
| `--confirm` | Confirmación en dos etapas: el cribado de N1 usa **una** semilla por vecino, y elegir los mejores de ~1000 evaluaciones ruidosas sobreestima su ventaja (maldición del ganador). Re-evalúa los K vecinos con mayor `d` que superen `eps`, junto al padre, con semillas nuevas; se confirman si la ventaja media supera 2 errores estándar. `0` = apagada. Con ella activa, `--climb` y `--n2-if-stuck` usan solo vecinos confirmados. Si hay más de K candidatos, `P(mejora confirmada)` es cota inferior | `20` |
| `--confirm-seeds` | Semillas nuevas por candidato (≥ 2, hace falta una varianza). Costo extra ≈ `(K+1) × V × 6 × alg_nReps` episodios | `5` |
| `--n2-mids`, `--n2-per-mid` | Muestrea N2: M intermedios (ponderados por su probabilidad de aparecer) y S vecinos de cada uno. Separa mejoras alcanzables por un intermedio neutro de las que exigen cruzar un valle | `0`, `50` |
| `--n2-if-stuck` | Solo corre N2 si N1 no tiene ningún vecino que mejore | off |
| `--climb` | Ascenso voraz: hasta K pasos al mejor vecino que supere `eps`, con semilla nueva en cada paso | `0` |
| `--dry-run` | Solo cuenta vecinos por operador; no simula | off |

**Ojo con `eps`.** Por defecto `eps = 2·sd` del fitness del *padre* entre semillas, que mide el
ruido de evaluar un individuo con otra semilla. Para comparar un vecino con su padre **con la
misma semilla** ese umbral es demasiado alto: en las corridas de `diag_rank02` salió en ~0.4 del
fitness del padre, mientras que el mejor vecino ganaba solo +1 a +20 y el percentil 99 de los Δ
era ~+4. Un resultado "ningún vecino supera eps" solo descarta mejoras grandes. Para mirar
mejoras pequeñas usa `--eps` bajo (p. ej. `--eps 0`) junto con `--confirm` y más semillas de
confirmación, y revisa la distribución de `d_mean` en `_neighbors.csv`.

Salidas (`-o`, default `log/neighbors_<archivo>`): `_neighbors.csv` (una fila por
individuo evaluado, con la recompensa por peso), `_parents.csv` (resumen por
padre/paso), `_ops.csv` (desglose por operador) y `_confirm.csv` (candidatos
re-evaluados: `d` del cribado vs `d` validado ± error estándar). Costo ≈ `nº vecinos × 6 pesos ×
alg_nReps` episodios por padre; el binario lo imprime antes de simular.
`OMP_NUM_THREADS` aplica igual que en el entrenamiento.

```bash
# vecindad del élite de una generación (N1 completa) y N2 si está atascado
./build/wann_car_neighborhood -i log/mi_corrida_snap/gen_00400.json \
    -d p/car_snn.json -p mi_override.json --n2-mids 20 --n2-if-stuck
```

---

## 4c. Diagnóstico de estancamiento y dependencia de la semilla — `diagnostico/`

Scripts de Python (solo numpy/pandas; matplotlib opcional) que trabajan sobre lo que
escribe `wann_car` con `snapshot_interval > 0` y sobre las salidas de
`wann_car_neighborhood`. Ninguno entrena. Todos anclan el cwd a la raíz del repo y
escriben por defecto en `diag_results/`.

```bash
# 1) tras el entrenamiento (log/diag_rank00/seed*_lineage.csv y seed*_snap/)
python diagnostico/lineage_analysis.py --glob 'log/diag_rank00/seed*_lineage.csv' --out diag_results/lineage
python diagnostico/early_prediction.py --glob 'log/diag_rank00/seed*_lineage.csv' --out diag_results/prediction

# 2) vecindad sobre los snapshots: primero costear (no simula), luego lanzar
python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank00/seed*_snap' \
    --override diag/rank00.json --gens 25,100,250,last --estimate
python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank00/seed*_snap' \
    --override diag/rank00.json --gens 25,100,250,last --run --jobs 3 --omp 60 \
    --extra "--n2-mids 20 --n2-if-stuck"

# 3) resumen, agrupando semillas en terciles de nivel final
python diagnostico/neighborhood_summary.py --dir log/diag_neighborhood \
    --runs-csv diag_results/lineage_runs.csv
```

| Script | Qué responde |
|--------|--------------|
| `lineage_analysis.py` | Cuándo se "decide" una corrida: nº de ancestros de la gen 0 vivos por generación, cuándo queda uno solo, ancestro común más reciente de la población final, percentil del ancestro del élite final, largo del estancamiento y tasa de hijos que superan al élite por operador (fase temprana vs tardía). Salidas `_runs/_curves/_ops.csv` |
| `early_prediction.py` | Cuánto predice lo medido en la generación g (fitness del élite, récord, top 5 %, nº de conexiones) el nivel final, entre corridas: Spearman con IC 95 % bootstrap. Salidas `_prediction.csv/.png` |
| `run_neighborhood.py` | Elige el snapshot más cercano a cada generación pedida y lanza `wann_car_neighborhood`. **Por defecto solo imprime el plan**; `--estimate` cuenta episodios sin simular; `--run` ejecuta. Salta lo ya hecho (`--force` rehace) |
| `confirm_report.py` | ¿Las "mejoras confirmadas" son reales o azar? Por generación: % confirmado frente al esperado por azar (t de Student), p binomial, Δ medio en cribado vs validado, y cuántos resisten Bonferroni por padre (y con efecto ≥ 2 % del fitness). Sirve además como **control positivo**: las generaciones tempranas, donde el élite sí sube, deben mostrar señal |
| `neighborhood_summary.py` | Clasifica cada padre (óptimo local en N1 / mejoras no confirmadas / mejoras confirmadas) y, con N2, si la salida pasa por un intermedio neutro o por un valle. Agrupa por terciles con `--runs-csv` |

Notas: el fitness de cada generación se evalúa con otra semilla y los élites se
re-evalúan, así que el récord acumulado incluye evaluaciones con suerte; por eso el
"nivel final" es la media del élite en las últimas 20 generaciones. Con menos de ~8
corridas los intervalos de correlación son muy anchos (los scripts avisan).

---

## 4d. Robustez a ruido de memristores — `wann_car_noise`

**Solo evalúa, no entrena.** Mide qué tan robusta es una red `car` ya
entrenada al ruido de memristores (escritura/programación del chip + lectura
en cada inferencia), usando los modelos de `memristor_noise.hpp` (puertos
literales de IBM `aihwkit`: PCM — Nandakumar et al., ICECS 2019 —, ReRAM —
Wan et al., Nature 2022 — y ReRAM CMO/HfOx — Falcone et al., IBM Research
Europa-Zúrich). Dos modos activables con `--mode`:

- `shared`: un único valor ruidoso para **toda** la red (como ya evalúa
  `wann_eval_weights_car`/`evalEpisodes`, pero con el escalar perturbado por
  ruido en vez de fijo) — barato, no ve fragilidad sinapsis-a-sinapsis.
- `per-synapse`: cada conexión habilitada recibe su propio valor ruidoso,
  muestreado de forma independiente — el escenario realista de N memristores
  físicos distintos. Requiere soporte de peso-por-sinapsis agregado a
  `Network` en `../snn-simulator` (`setSynapseWeight`/`step()`/`run(duration)`
  sin escalar — adiciones puras, no tocan el `step(double)` de siempre).

```bash
./build/wann_car_noise -f red.out -d p/car_snn.json [-p overrides.json] \
    (--weight-index N | --weight VAL) \
    [--mode shared|per-synapse] [--noise-model none|simple|pcm|reram|reramcmo] \
    [--n-chips N] [--reads-per-chip N] [--t-inference t1,t2,...] \
    [--w-max V] [--noise-scale F] \
    [--prog-noise-scale F] [--read-noise-scale F] [--drift-scale F] \
    [--n-levels N] [--reramcmo-g-max V] [--reramcmo-g-min V] \
    --seeds s0,s1,...
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada (**obligatorio**) | — |
| `-d` | Config (**obligatorio**) | — |
| `--seeds` | Lista de seeds separadas por coma (**obligatorio**) | — |
| `--weight-index` / `--weight` | Uno de los dos, **obligatorio**: índice en `WEIGHT_VALS` (`0..5`) o un valor de peso directo | — |
| `-p` | Overrides del config (archivo) | ninguno |
| `--mode` | `shared` o `per-synapse` | `shared` |
| `--noise-model` | `none` (sin ruido — chequeo de regresión), `simple` (placeholder de dos coeficientes), `pcm`, `reram`, `reramcmo` | `none` |
| `--n-chips` | Nº de instancias de "chip" simuladas (cada una re-sortea ruido de escritura) | `100` |
| `--reads-per-chip` | Nº de lecturas (inferencias) por chip (ruido de lectura fresco en cada una). Pasar > 1 es necesario para que `summarize_car_noise.py` pueda separar varianza entre-chips de varianza dentro-de-chip | `1` |
| `--t-inference` | Segundos desde la programación, CSV. Solo importa para `pcm`/`reram`/`reramcmo` (deriva). `reram` exige que coincida exactamente con una clave de `ReRamWan2022DeviceModel::coeff_dic`; `reramcmo` acepta cualquier valor (deriva continua), pero `0` es un bypass total (sin deriva ni ruido de lectura) | `0` (pcm); `1,86400,172800` (reram); `1` (reramcmo) |
| `--w-max` | Conductancia máxima del chip en unidades de peso lógico (mapea `|peso| = w_max` → `g_max`) | máximo de `WEIGHT_VALS` (`8.0`) |
| `--noise-scale` | Multiplicador de intensidad sobre los tres mecanismos del modelo elegido a la vez (`0` = sin ruido, `1` = los valores por defecto del header, `2` = el doble). Sin efecto con `--noise-model none` | `1.0` |
| `--prog-noise-scale` / `--read-noise-scale` / `--drift-scale` | Desglosan `--noise-scale` por mecanismo, para ablación uno-a-la-vez (p. ej. `--read-noise-scale 0 --drift-scale 0` mide SOLO ruido de programación/D2D). Si no se pasan, cada uno cae en `--noise-scale`. `reram` (Wan2022) no distingue programación de deriva/lectura en su modelo — un solo valor (`--prog-noise-scale`, o `--noise-scale` si no se da) cubre los tres; `reramcmo` sí los separa, como PCM | sin pasar → `--noise-scale` |
| `--n-levels` | Cuantiza la conductancia a N niveles distinguibles al programar (par diferencial, cada rama por separado) — modela resolución de programación finita, no viene de ningún paper citado | `0` (continuo) |
| `--reramcmo-g-max` / `--reramcmo-g-min` | Conductancia máx/mín (µS) del modelo ReRAM CMO/HfOx — **sin valor por defecto confiable en aihwkit** (su `__init__` exige pasarlos; acá los defaults son solo ilustrativos, no de Falcone et al.). `g_min > 0` por defecto: la rama "apagada" de cada par diferencial siempre queda flotando en al menos `g_min`, incluso sin ruido — es una propiedad física del modelo, no un bug (ver `memristor_noise.hpp`) | `25.0` / `1.0` |

Salida: CSV a stdout, una fila por `(seed, chip, read, t_inference, escalas)`
con el reward promediado sobre `alg_nReps` episodios —
`seed,chip,read,t_inference,mode,noise_model,weight,prog_noise_scale,read_noise_scale,drift_scale,weight_rmse,reward_shaped,reward_original`.
`weight_rmse` es el RMSE entre el peso realmente realizado (tras el ruido) y
el nominal, normalizado por `w_max` — a diferencia de `noise_scale`, es la
**misma unidad en los cuatro modelos**, así que sirve de eje común para
comparar `simple`/`pcm`/`reram`/`reramcmo` entre sí (ver `plot_car_noise_rmse.py`).
Sin agregación estadística en C++: usar `eval_results/summarize_car_noise.py`
sobre el CSV resultante.

```bash
# chequeo de regresión: deben coincidir EXACTAMENTE (verificado al implementar)
./build/wann_car_noise -f red.out -d p/car_snn.json --weight-index 1 \
    --mode shared      --noise-model none --n-chips 1 --reads-per-chip 1 --seeds 0
./build/wann_car_noise -f red.out -d p/car_snn.json --weight-index 1 \
    --mode per-synapse --noise-model none --n-chips 1 --reads-per-chip 1 --seeds 0

# ablación: SOLO ruido de programación/D2D (lectura y deriva apagadas)
./build/wann_car_noise -f red.out -d p/car_snn.json --weight-index 1 \
    --mode per-synapse --noise-model pcm \
    --prog-noise-scale 1 --read-noise-scale 0 --drift-scale 0 \
    --n-chips 30 --reads-per-chip 3 --seeds 0,1 > log/car_noise_prog_only.csv

# resolución de programación finita: solo 8 niveles de conductancia distinguibles
./build/wann_car_noise -f red.out -d p/car_snn.json --weight-index 1 \
    --mode per-synapse --noise-model pcm --n-levels 8 \
    --n-chips 30 --reads-per-chip 1 --seeds 0,1 > log/car_noise_8levels.csv

# resumen: media, std TOTAL, std entre-chips, std dentro-de-chip, percentiles
# 10/50/90, IC 95% por bootstrap (remuestreando chips) y fracción de chips
# por encima de un umbral
python eval_results/summarize_car_noise.py --csv log/car_noise_prog_only.csv \
    --success-threshold 500

# las figuras de una sola vez (barrido de --noise-scale, de --t-inference, y
# opcionalmente la grilla 2D completa), orquestando wann_car_noise por vos —
# ver eval_results/run_car_noise_sweeps.py
python eval_results/run_car_noise_sweeps.py \
    --net log/full_p3_car_ttfs_first_spike/ttfs_first_spike_pruned.out \
    --weight-index 5 --reads-per-chip 3 --prefix car_ttfs_first_spike --jobs 4

# eje común entre modelos (RMSE efectivo) — no necesita --grid
python eval_results/plot_car_noise_rmse.py --csv log/car_ttfs_first_spike_noise_sweep.csv \
    --out "graficos/Racing Car/car_ttfs_first_spike_rmse_robustness.png"
```

`run_car_noise_sweeps.py` corre varios barridos y genera las figuras
correspondientes en una sola invocación:

- **Intensidad** (`--noise-scales`, default `0,0.5,...,5.0`): una llamada a
  `wann_car_noise` por `(noise_model, noise_scale)` — `--noise-scale` es un
  escalar, no admite lista — con `t_inference` fijo (`0` para simple/pcm, `1`
  para reram/reramcmo). Resultado: `<prefix>_noise_sweep.csv` +
  `eval_results/plot_car_noise.py` → `<prefix>_noise_robustness.png`.
- **Tiempo** (`--time-points-pcm`/`--time-points-reram`/`--time-points-reramcmo`):
  acá sí una sola llamada por modelo (`--t-inference` sí admite lista), con
  `noise_scale` fijo en `1.0`. El modelo "simple" queda afuera (no depende del
  tiempo). `--time-points-pcm`/`--time-points-reramcmo` por defecto son una
  grilla log de 60 s a 1 año más el 0 como referencia "recién programado"
  (`0,60,300,3600,21600,86400,604800,2592000,31536000` — convención de Le
  Gallo et al. 2023 para evaluar drift con aihwkit). `--time-points-reram`
  queda fijo en sus 3 únicas claves medidas. Resultado: `<prefix>_time_sweep.csv`
  + `eval_results/plot_car_noise_time.py` → `<prefix>_time_robustness.png`.
- **Grilla 2D** (`--grid`): combina los dos ejes — una llamada por
  `(noise_model, noise_scale)` con la lista COMPLETA de tiempos de ese
  modelo (pcm/reram/reramcmo, no "simple"). Resultado: `<prefix>_grid_sweep.csv`
  + `eval_results/plot_car_noise_heatmap.py` (uno por modelo) →
  `<prefix>_heatmap_{pcm,reram,reramcmo}.png` — muestra interacciones que las
  dos curvas 1D no pueden (p. ej. si la deriva solo importa cuando el ruido de
  programación ya es alto).

`--weight-indices 1,3,5` (en vez de `--weight-index`) repite cualquiera de
los barridos de arriba para cada peso compartido de la lista, concatenados
en el mismo CSV (la columna `weight` los distingue) — la premisa de WANN es
desempeño estable a través de pesos; esto permite ver si eso se sostiene
también bajo ruido. `--jobs N` corre invocaciones de `wann_car_noise` en
paralelo (son procesos independientes, no usan OpenMP internamente).
`--dry-run` solo imprime los comandos; `--skip-plots` genera únicamente los
CSV.

Cualquier gráfico/figura sobre estos resultados se genera con un script de
Python (`eval_results/plot_car_noise{,_time,_rmse,_heatmap}.py`,
`eval_results/run_car_noise_sweeps.py`, o uno nuevo del mismo estilo), nunca
como página HTML/Artifact — ver la nota en la cabecera de este documento.

---

### 4d-bis. WANN vs. gradientes surrogados bajo ruido — `run_noise_compare.py`

Compara el mejor modelo WANN de cada tarea con checkpoints de
`../surrogate-gradient-rl` (`scripts/noise_eval.py`; mismos modelos de ruido,
mismo generador por (semilla, chip), mismo entorno) en una sola pasada y en
paralelo. Solo evalúa. Las tres tareas:

| Tarea | WANN (binario) | Mejor modelo (mayor `reward` revalidado) | SGRL |
|---|---|---|---|
| `car` | `wann_car_noise` | `car_ttfs_first_spike_discrete` | `carrace-f2` |
| `disc_mc` (Mountain Car discreto) | `wann_disc_mc_noise` | mejor de `disc_mc_*_discrete` | `mountaincar-f2` |
| `acrobot` | `wann_acrobot_noise` | mejor de `acrobot_*` (búsqueda original) | `acrobot-f2` |

`wann_acrobot_noise` y `wann_disc_mc_noise` son el mismo `main_car_noise.cpp` con la tarea
fijada en compilación (`NOISE_TASK_ACROBOT`/`NOISE_TASK_DISC_MC`; mismo motivo que
`wann_eval_weights_<tarea>`) y requieren `evalEpisodesPerSynapse` de `SnnAcrobotTask`/
`SnnDiscMCTask`. Chequeo de regresión hecho: con `--noise-model none`, `--mode per-synapse`
da exactamente lo mismo que `--mode shared` en ambas tareas.

Cada trabajo es una celda (modelo, escala, t, mecanismo) de una semilla de un modelo; el
planificador (uno solo para todas las tareas) mantiene `--jobs` procesos vivos y como mucho
`--jobs-sgrl` de Python/torch (~0.5 GB cada uno). Es reanudable (`parts/` atómico) y un trabajo
fallido no aborta el resto.

```bash
python eval_results/run_noise_compare.py --dry-run                      # plan, modelos elegidos y estimación
python eval_results/run_noise_compare.py --jobs 500 --jobs-sgrl 250     # las tres tareas
python eval_results/run_noise_compare.py --task acrobot --sg-top 1      # una sola tarea
python eval_results/plot_noise_compare.py --data-dir eval_results/acrobot_wann_vs_sgrl   # solo figuras
```

Por defecto: la red ganadora con los 6 pesos compartidos y las 5 mejores semillas del mejor
trial de SGRL; 30 chips × 2 semillas × 3 lecturas; `--episodes` = `alg_nReps` (6). Salida en
`eval_results/<tarea>_wann_vs_sgrl/` (`sweep_all.csv`, `models.csv`, `aggregated.csv`,
`protocol.json`) y figuras `graficos/<Tarea>/<tarea>_wann_vs_sgrl_*.png` (intensidad, tiempo,
RMSE, heatmaps y ablación por dispositivo).

La comparación justa es la *retención*: `(retorno − piso)/(retorno sin ruido − piso)`, con
piso = recompensa del peor desenlace (car 0; Mountain Car discreto −200; Acrobot −500; las dos
últimas dan −1 por paso). Se usa la recompensa *original* (la shaped difiere en acrobot y
disc_mc). Diferencia que no se puede igualar: en WANN cada sinapsis vale ±(peso compartido) y
`w_max` es el mayor `WEIGHT_VALS` de la tarea (el valor por defecto de `--w-max`), mientras que
SGRL usa pesos entrenados con `w_max = max|W|` por capa, así que el ruido relativo cae distinto.

## 5. Exportar a ODIN — `wann_car_odin_export`

Convierte una red car ya evolucionada a un JSON de configuración para el core
ODIN (ver `ODIN_DEPLOYMENT.md`). Corre en x86, no necesita hardware.

```bash
./build/wann_car_odin_export [-f red.out] [-d config.json] [-w weight_index] \
                             [-k run_key] [-o salida.json] [--thr 0-7]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-f` | Red guardada | `log/snn_car_best.out` |
| `-d` | Config | `p/car_snn.json` |
| `-w` | Índice de peso compartido; si falta se lee el `.wi` junto a la red (error si no existe) | `.wi` |
| `-k` | `run_key` (metadato) | vacío |
| `-o` | JSON de salida | `odin_config.json` |
| `--thr` | Fuerza el umbral IZH (0–7, 3 bits) de todas las neuronas | valores de `izhParamsFor()` |

Verificar antes de exportar una red con decoder `population_vector` (no está
comprobado).

## 6. Evaluar sobre hardware ODIN — `wann_car_odin_eval`

Solo existe si se compiló con `-DWANN_ODIN_HW=ON` (ARM/Linux, ZCU104/PYNQ). En
cada power cycle hay que correr antes `python3 odin_bootstrap.py /ruta/odin.bit`.

```bash
./build/wann_car_odin_eval [-c odin_config.json] [-d config.json] [-n episodios] \
                           [-s seed] [-m window_ms] [--csv] [--profile] [--steps N]
```

| Flag | Significado | Default |
|------|-------------|---------|
| `-c` | Config exportada por `wann_car_odin_export` | `odin_config.json` |
| `-d` | Config de hiperparámetros | `p/car_snn.json` |
| `-n` | Nº de episodios | `10` |
| `-s` | Seed | `0` |
| `-m`, `--window-ms` | Ventana SNN en ms; **debe coincidir con `snn_window_ms` del entrenamiento** (avisa si no se pasa) | `40` |
| `--csv` | stdout = solo `episode,reward` (el resto a stderr) | off |
| `--profile` | Desglose de tiempo entorno vs. camino ODIN | off |
| `--steps` | Pasos por episodio | `1000` |

---

## 7. Pipeline de screening (Python, orquesta los binarios)

```bash
python screening_reduce.py --task car --encoder ttfs --decoder first_spike --rounds 4 --n 30 --jobs 8 --omp 64
python screening_full.py   --task car --encoder ttfs --decoder first_spike --mode both --n 20 --top 3 --seeds 11 --jobs 3 --omp 190
python eval_results/eval_p3_weights.py --task car --seeds 11
bash generate_graphs.sh car
python bootstrap_results/bootstrap_compare_car_auto.py --run-key car_ttfs_first_spike
```

`--jobs` × `--omp` deben ajustarse al número de cores (se multiplican).
