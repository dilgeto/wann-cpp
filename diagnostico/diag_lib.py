"""Helpers compartidos por los scripts de diagnóstico de esta carpeta (linaje,
predicción temprana, vecindad). Solo numpy/pandas.

Formato de entrada: ``log/<prefix>_lineage.csv`` que escribe ``wann_car`` cuando el
JSON trae ``snapshot_interval > 0``. Columnas:
``gen,idx,parent,parent_b,op,applied,fitness,fit_max,n_conn``.
  - ``parent`` indexa la población de la generación anterior (-1 en la gen 0).
  - ``op`` = operador (0 addConn, 1 addNode, 2 enable, 3 mutAct, 4 toggle);
    -1 = élite copiado sin cambios (y toda la gen 0).
  - ``fitness`` = media sobre los pesos compartidos, evaluada con la semilla de ESA
    generación: un mismo genoma cambia de valor entre generaciones (ruido).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OP_NAMES = ["addConn", "addNode", "enable", "mutAct", "toggleExcitatory"]


def anchor_to_repo_root() -> None:
    """Igual que bootstrap_results/*.py: mismo comportamiento desde cualquier cwd."""
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


@dataclass
class Lineage:
    name:    str
    parent:  np.ndarray   # [G, P] int
    op:      np.ndarray   # [G, P] int
    applied: np.ndarray   # [G, P] int
    fitness: np.ndarray   # [G, P] float
    n_conn:  np.ndarray   # [G, P] int

    @property
    def n_gen(self) -> int:
        return self.parent.shape[0]

    @property
    def pop(self) -> int:
        return self.parent.shape[1]

    @property
    def last_gen(self) -> int:
        return self.n_gen - 1

    def elite_idx(self) -> np.ndarray:
        """Índice del individuo de mayor fitness en cada generación (como main_car)."""
        return self.fitness.argmax(axis=1)

    def elite_fit(self) -> np.ndarray:
        return self.fitness.max(axis=1)

    def running_best(self) -> np.ndarray:
        """Récord acumulado de fitness: lo que usa early_stop_patience (fitTop)."""
        return np.maximum.accumulate(self.elite_fit())

    def final_level(self, last: int = 20) -> float:
        """Nivel final robusto: media del fitness del élite en las últimas `last` gens.
        El récord acumulado premia evaluaciones con suerte; esto no."""
        return float(self.elite_fit()[-last:].mean())


def load_lineage(path: str | Path) -> Lineage:
    """Carga un *_lineage.csv. Descarta una última generación incompleta (corrida
    interrumpida) y exige que las generaciones sean contiguas desde 0."""
    path = Path(path)
    df = pd.read_csv(path)
    counts = df.groupby("gen").size()
    pop = int(counts.loc[0]) if 0 in counts.index else 0
    if pop == 0:
        raise ValueError(f"{path}: sin generación 0 completa")

    complete = []
    for g in range(int(df["gen"].max()) + 1):
        if counts.get(g, 0) == pop:
            complete.append(g)
        else:
            break
    df = df[df["gen"].isin(complete)].sort_values(["gen", "idx"])
    shape = (len(complete), pop)

    idx = df["idx"].to_numpy().reshape(shape)
    if not (idx == np.arange(pop)).all():
        raise ValueError(f"{path}: índices de individuo inesperados")

    name = path.name
    for suffix in ("_lineage.csv", ".csv"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return Lineage(
        name=name,
        parent=df["parent"].to_numpy(dtype=np.int64).reshape(shape),
        op=df["op"].to_numpy(dtype=np.int64).reshape(shape),
        applied=df["applied"].to_numpy(dtype=np.int64).reshape(shape),
        fitness=df["fitness"].to_numpy(dtype=np.float64).reshape(shape),
        n_conn=df["n_conn"].to_numpy(dtype=np.int64).reshape(shape),
    )


# --------------------------------------------------------------------------- linaje
def founders(lin: Lineage) -> np.ndarray:
    """founders[g, i] = índice, en la gen 0, del ancestro de i en la gen g."""
    f = np.empty_like(lin.parent)
    f[0] = np.arange(lin.pop)
    for g in range(1, lin.n_gen):
        f[g] = f[g - 1][lin.parent[g]]
    return f


def ancestor_line(lin: Lineage, g_end: int, i_end: int) -> np.ndarray:
    """line[g] = índice del ancestro en la gen g del individuo i_end de la gen g_end."""
    line = np.empty(g_end + 1, dtype=np.int64)
    line[g_end] = i_end
    for g in range(g_end, 0, -1):
        line[g - 1] = lin.parent[g][line[g]]
    return line


def mrca_gen(lin: Lineage) -> float:
    """Última generación en que UN solo individuo es ancestro de toda la población
    final (ancestro común más reciente). NaN si en la gen 0 sobreviven varios."""
    ancestors = np.arange(lin.pop)
    for g in range(lin.last_gen, 0, -1):
        ancestors = np.unique(lin.parent[g][ancestors])
        if len(ancestors) == 1:
            return float(g - 1)
    return float("nan")


def percentile_in_gen(lin: Lineage, g: int, i: int) -> float:
    """Fracción de la población de la gen g con fitness estrictamente menor al de i."""
    return float((lin.fitness[g] < lin.fitness[g, i]).mean())


# ----------------------------------------------------------------------- estadística
def spearman(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """(rho, lo, hi, n) con IC percentil por bootstrap sobre las corridas."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    rho = spearman(x, y)
    if n < 3 or not np.isfinite(rho):
        return rho, float("nan"), float("nan"), n
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        s = rng.integers(0, n, n)
        r = spearman(x[s], y[s])
        if np.isfinite(r):
            boots.append(r)
    if len(boots) < 10:
        return rho, float("nan"), float("nan"), n
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rho, float(lo), float(hi), n
