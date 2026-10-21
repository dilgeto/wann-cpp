# Diagnóstico: estancamiento y dependencia de la semilla en WANN (car)

Resumen del análisis de vecindad y linaje pedido para entender (a) por qué el
entrenamiento de `wann_car` se estanca mucho antes de `maxGen`, y (b) por qué el
resultado final depende tanto de la semilla inicial. Cubre 30 corridas de
entrenamiento (`log/diag_rank02/`, config `rank02` de
`screening_full/car_ttfs_first_spike/`, `popSize` 480, 1024 generaciones,
`snn_encoder`/`snn_decoder` = `ttfs`/`first_spike`) y el análisis de vecindad
construido sobre ellas. Todo lo reportado aquí es reproducible con los scripts de
`diagnostico/` y los datos en `diag_results/` y `log/diag_*`.

## 1. Qué se construyó

- **`snapshot_interval`** (`Hyperparams.h`, solo `main_car.cpp`): registro por
  generación de qué individuo viene de qué padre y con qué operador
  (`log/<prefix>_lineage.csv`), y snapshot de la población completa cada N
  generaciones (`log/<prefix>_snap/gen_XXXXX.json`). Puramente registro: no
  consume el RNG, no altera el entrenamiento (verificado, sección 2).
- **`wann_car_neighborhood`**: dado un individuo, enumera *todos* los genomas a
  una mutación de distancia (N1: cada `addConn`, `addNode`, `enable`, `mutAct`,
  `toggleExcitatory` posible, con su probabilidad exacta), los evalúa con la
  misma semilla que el padre, y reporta mejores/neutros/peores por operador.
  Incluye una etapa de confirmación (re-evaluación con semillas nuevas), N2
  muestreado y ascenso voraz. `include/wann/Neighborhood.h` reimplementa los
  operadores de `Wann` (privados) para poder enumerarlos.
- **`diagnostico/`** (Python): `lineage_analysis.py` (coalescencia y
  ancestros), `early_prediction.py` (correlación de lo temprano con el
  resultado final), `run_neighborhood.py` (lanza el binario en lote),
  `neighborhood_summary.py` y `confirm_report.py` (agregan resultados y separan
  señal real de ruido estadístico).

Detalle de uso de cada uno en `BINARIOS.md` §4b/§4c.

## 2. Verificación de la metodología

- **Reproducibilidad del registro**: se corrió la misma semilla (60
  generaciones) con y sin `snapshot_interval`; `stats.out` salió idéntico
  (`log/diag_check/`).
- **Paridad evaluación/entrenamiento**: en los 180+ análisis de vecindad
  corridos, la recompensa recalculada del padre coincidió exactamente
  (`max|dif|=0`) con la guardada en el snapshot, en el 100% de los casos.
- **Control positivo del criterio de confirmación**: se repitió el análisis de
  vecindad en generaciones tempranas (25, 50, 100), donde el élite todavía
  mejora, para confirmar que la herramienta detecta mejoras reales cuando
  existen y que su ausencia en generaciones tardías no es un artefacto de
  insensibilidad (sección 4).
- **Corrección del criterio de confirmación**: el criterio inicial
  ("media − 2·error estándar > 0") no corregía por comparaciones múltiples;
  con `eps` bajo y decenas de candidatos por padre confirmaba ~3.9% de 827
  candidatos, estadísticamente indistinguible del 3.8% esperado por puro azar.
  Se corrigió a un t-test de una cola con Bonferroni entre los candidatos de
  cada padre, más un piso de efecto práctico (`--min-effect`, default 0.2% del
  fitness del padre) — ver `src/main_car_neighborhood.cpp`.

## 3. Dependencia de la semilla

30 corridas idénticas salvo la semilla, mismos hiperparámetros:

| | Valor |
|---|---|
| Nivel final (mediana del élite, últimas 20 gens) | 241.8 |
| Rango | 176.3 – 427.3 (2.4×) |
| Coeficiente de variación | 0.32 |

El resultado final es **bimodal**: 6 de 30 corridas (20%) llegan a ~405–427; el
resto no pasa de 325. El hueco entre ambos grupos (82 puntos) es más del doble
del siguiente hueco más grande (37).

**Sin crossover estructural** (`prob_crossover = 0` en esta config), cada
individuo desciende de un único ancestro de la generación 0, y la diversidad
solo puede perderse:

- Mediana: un solo fundador domina desde la generación **14** (rango 6–634).
- El ancestro común más reciente de toda la población final aparece en la
  generación **987.5** (mediana), de 1023.
- El ancestro del élite final estaba en el top 10% de fitness de la generación
  0 en el **77%** de las corridas (azar puro ≈ 10%) — pero eso **no predice**
  el nivel final (Spearman ρ ≈ 0.03, no significativo).

**El resultado se decide entre las generaciones ~25 y ~100**, no antes:
Spearman entre el fitness del élite en la generación *g* y el nivel final,
sobre las 30 corridas (`diagnostico/early_prediction.py`):

| gen | 5 | 10 | 25 | 50 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|---|
| ρ | 0.14 | 0.15 | 0.61 | 0.79 | 0.89 | 0.93 | 0.96 |

Un dato adicional sugerente (no confirmado con más análisis): las conexiones
del élite en la generación 5 correlacionan **negativamente** con el resultado
final (ρ = −0.54, IC 95% [−0.75, −0.24], n=30) — arrancar con menos estructura
se asocia a mejor resultado.

## 4. El estancamiento es real, no un artefacto de la herramienta

Análisis de vecindad N1 sobre el élite de 12 corridas (las 6 de nivel más alto
y las 6 más bajo), en generaciones 25, 50, 75, 100, 150, 200, 250, 300, 350,
400 y 1023 (`log/diag_neighborhood_control/`, `log/diag_neighborhood_mid/`,
`log/diag_neighborhood_eps0/`, `diag_results/confirm_gap_report.csv`):

- **Control positivo confirmado**: en las generaciones 25, 50 y 100 (grupo de
  nivel alto), la fracción de candidatos confirmados excede claramente el azar
  (p ≈ 1e-9 a 1e-19 vs. ~3.8% esperado). La herramienta detecta mejoras reales
  cuando las hay.
- **La transición no es un corte limpio, es un cambio de régimen entre la gen
  ~100 y ~150**: de señal densa y clara (varios confirmados por corrida, p
  extremo) a señal escasa. De 150 en adelante (150, 200, 250, 300, 350, 400,
  1023 — 7 puntos × 6 corridas de nivel alto = 42 combinaciones), solo 5
  tuvieron algún candidato confirmado por el propio criterio del binario
  (Bonferroni + efecto mínimo 0.2%), y en 3 de esas 7 generaciones (200, 300 y
  1023 para el grupo alto) no hubo ninguna. Donde sí aparece algo, el efecto
  real va de 0.34% a 2.8% del fitness del padre — nunca cero, pero raro y
  pequeño frente a la densidad de la ventana temprana.
- **Ejemplo concreto**: `seed22`, que escapa recién hacia la generación ~680
  (sección 3), ya tenía en la generación 150 varios vecinos confirmados de
  ~0.34–0.36% de efecto (un `addNode` repetido y un `toggleExcitatory`) —
  consistente con la imagen de acumulación lenta de pasos pequeños, no de un
  golpe de suerte único, y con que la evolución tarda en explotarlos incluso
  cuando existen.
- **Incluso donde hay señal real, el efecto es casi siempre minúsculo**: de
  3102 candidatos re-evaluados con 10 semillas nuevas (todo el rango 25–1023),
  la mayoría de las confirmaciones reales quedan bajo 1% del fitness del
  padre; unas pocas (`seed19` en 250 y 350, `seed08` en 1023) llegan a 2–7%.
  Nunca un salto grande y frecuente — consistente con evolución incremental.
- **N2 muestreado** (36 000 nietos, 20 intermedios × 50 nietos por padre): solo
  6.3% con Δ>0 en el cribado; el mayor Δ en cualquier padre fue +32 sobre
  fitness de 150–430.

## 5. Qué pasa con la estructura mientras tanto

Sobre las 30 corridas completas, en las mismas 6 generaciones
(`log/diag_neighborhood_neutral/`), cada mutación se clasifica como "idéntica"
si produce exactamente la misma recompensa que el padre en los 6 pesos
compartidos (misma semilla):

| gen | 25 | 50 | 75 | 100 | 400 | 1023 |
|---|---|---|---|---|---|---|
| % mutaciones que sí cambian el comportamiento | 54.9 | 41.0 | 36.4 | 32.6 | 20.6 | **11.0** |
| Conexiones del élite (mediana) | 14.5 | 18 | 20 | 20.5 | 34.5 | 57.5 |

Al final del entrenamiento, **9 de cada 10 mutaciones no cambian nada**, pero
la red sigue creciendo sin pausa. Por operador, en la generación 1023: `enable`
sigue siendo casi siempre efectivo (57.5% idénticas); `addConn`, `addNode`,
`mutAct` y `toggleExcitatory` son idénticas en 85–92% de los casos.

**Hallazgo central**: la fracción de mutaciones vivas en la *última*
generación correlaciona fuertemente con el nivel final (ρ = 0.68, IC 95%
[0.39, 0.86], n=30) — y **no es un efecto de tamaño**: no correlaciona con el
número de conexiones (ρ = −0.04), y el número de conexiones por sí solo
predice mal el resultado final (ρ = 0.28, no significativo). Las 6 corridas de
nivel alto retienen 12–24% de mutaciones vivas al final, contra ~9% de mediana
en el resto, repartidas en todo el rango de tamaños de red.

## 6. Conclusión

1. **Sin crossover, la coalescencia es rápida e inevitable** (mediana: un solo
   fundador desde la gen 14): la búsqueda es, de hecho, 480 caminatas
   independientes que colapsan a una sola muy pronto.
2. **El resultado se decide en una ventana temprana** (gen ~25–100), no por
   cuál individuo arranca mejor (el líder de la gen 0 no predice nada), sino
   por qué estructura queda fijada en esa ventana.
3. **Después de esa ventana, el estancamiento es real**: no hay vecinos a un
   paso que mejoren más allá del ruido, confirmado con un control positivo que
   sí detecta mejoras cuando existen.
4. **La red sigue creciendo aunque deje de mejorar**: la gran mayoría de las
   mutaciones dejan de tener efecto (89% al final), pero no se eliminan ni se
   penalizan lo suficiente — se acumula estructura inerte.
5. **Cuánta estructura inerte se acumula, no cuánto crece la red, distingue las
   corridas buenas de las malas**: retener más "superficie mutable" al final
   correlaciona con mejor resultado, independientemente del tamaño.

Esto es consistente con deriva neutral en un algoritmo sin crossover, donde la
presión de parsimonia (objetivo 1/nConn, activo con probabilidad
`alg_probMoo`) no distingue conexiones que aportan de conexiones que son puro
lastre estructural.

## 7. Limitaciones

- Todo el análisis de vecindad (secciones 4) usa **un solo individuo por
  corrida y generación** (el élite), no la población completa de 480.
- Todo es sobre **una configuración** (`rank02`) y **una tarea** (`car`,
  encoder `ttfs`, decoder `first_spike`); no se ha repetido con otras.
- El análisis de vecindad detallado (control positivo, confirmación) usa 12
  corridas; el de neutralidad usa las 30. Los intervalos de confianza de
  varias correlaciones son anchos.
- Es observacional: no se ha probado causalidad, solo correlación entre
  corridas ya hechas.
- Los dos problemas del núcleo documentados en `CLAUDE.md`
  ("Known issues to review") — `addConn` nunca conecta la capa oculta más
  profunda a una salida, y conexiones oculto→oculto deshabilitadas que
  reviven en la red exportada — no se han corregido y podrían interactuar con
  estos hallazgos.

## 8. Cómo reproducir

```bash
# linaje y predicción temprana (barato, ya hecho)
python diagnostico/lineage_analysis.py --glob 'log/diag_rank02/seed*_lineage.csv' --out diag_results/lineage
python diagnostico/early_prediction.py --glob 'log/diag_rank02/seed*_lineage.csv' --out diag_results/prediction

# vecindad N1 + confirmación (cómputo real; ver diagnostico/run_neighborhood.py --estimate primero)
python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank02/seed*_snap' --runs <lista> \
    --gens <gens> --override screening_full/car_ttfs_first_spike/p3_configs/rank02_seed00.json \
    --out-dir <dir> --run --extra "--eps 0 --confirm 100 --confirm-seeds 10"
python diagnostico/confirm_report.py --dirs <dir> --runs-csv diag_results/lineage_runs.csv --split-at 400

# neutralidad (barato: --confirm 0, sin re-evaluación)
python diagnostico/run_neighborhood.py --snap-glob 'log/diag_rank02/seed*_snap' \
    --gens 25,50,75,100,400,1023 --override .../rank02_seed00.json \
    --out-dir log/diag_neighborhood_neutral --run --extra "--confirm 0"
```

## 9. Próximo paso

Se probó una primera intervención: podar (deshabilitar) conexiones sin camino
estructural de entrada a salida, activable/desactivable vía hiperparámetro,
sin tocar el resto del algoritmo. Implementada, verificada correcta contra un
genoma armado a mano con estructura desconectada, y verificada reversible
(con el flag apagado, `stats.out` salía byte a byte idéntico al original).

Pero probada contra ~87 000 conexiones de 4 poblaciones reales
(`log/diag_rank02/`, hasta la generación 1023) y contra corridas de
entrenamiento nuevas de hasta 300 generaciones, **nunca encontró nada que
podar**. La razón: los 5 operadores de `Wann` nunca pueden producir una
conexión habilitada y estructuralmente desconectada — `mutAddNode` siempre
inserta el nodo nuevo dentro de un camino ya válido, y ningún operador
deshabilita una conexión previamente viva salvo `mutAddNode` al dividirla
(lo cual preserva el camino). La neutralidad del 89% medida en la sección 5
no es desconexión estructural: es una conexión conectada cuyo efecto se
pierde por otra razón (dinámica de disparos, saturación del decodificador),
que solo se detecta simulando.

Por eso se revirtió (2026-09-28): el código no quedó en el repo. Quedan dos
rutas abiertas, ninguna implementada aún:

- Cambiar qué cuenta el objetivo `1/nConn` de la selección (NSGA-II) para que
  solo compute conexiones vivas, en vez de podar el genoma directamente.
- Detectar neutralidad real simulando cada cierto número de generaciones
  (probar si desactivar una conexión cambia el fitness) y podar esas — ataca
  exactamente lo medido, a costo de más cómputo por generación.
