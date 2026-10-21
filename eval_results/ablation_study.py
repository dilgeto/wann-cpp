#!/usr/bin/env python3
"""
ablation_study.py — ablaciones post-hoc de las dos contribuciones del método
(polaridad de las sinapsis y tipo de neurona Izhikevich) sobre la red GANADORA
del bootstrap de cada tarea (acrobot, mountain_car discreto, car).

No entrena nada: solo evalúa la red guardada con wann_eval_weights_<task> y
los flags de ablación de src/main_eval_weights.cpp.

Diseño (pareado con el bootstrap): mismas seeds de episodio que
bootstrap_compare_*_auto.py (seed0=0, n=100, un episodio por seed, --nreps 1),
así que la condición "normal" reproduce el `snn` de bootstrap_results/
bootstrap_<task>/rewards.csv (se verifica con `check`). Cada condición se evalúa
con los 6 pesos compartidos (un barrido por invocación), no solo con el ganador.

Estudio 1 — polaridad
  normal        red original
  no_inh        se eliminan las sinapsis inhibitorias (w<0 -> 0)
  drop_exc      CONTROL: se eliminan K excitatorias al azar, K = nº de inhibitorias
                (--draws sorteos) — separa "importa la inhibición" de "importa
                tener menos conexiones"
  flip_inh      las inhibitorias pasan a excitatorias (misma topología)
Estudio 2 — tipo de neurona
  type1..type6  TODAS las neuronas (entradas, bias, ocultas, salidas) del tipo
                1..6 de ann_actRange (RS, FS, CH, LTS, IB, RZ)
  shuffle       CONTROL: tipos permutados entre neuronas (--draws sorteos)

Subcomandos:
  check     re-evalúa "normal" y lo compara con bootstrap_results/.../rewards.csv
  run       evalúa todas las condiciones (reanuda: salta las ya presentes en el CSV)
  analyze   tablas (CSV + markdown) y figuras (graficos/<Tarea>/)

Uso:
  python eval_results/ablation_study.py check
  python eval_results/ablation_study.py run --jobs 4 --draws 30
  python eval_results/ablation_study.py analyze
"""
import argparse
import concurrent.futures
import io
import os
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_p3_weights import TASKS  # noqa: E402  (también ancla CWD a la raíz del repo)

OUT_DIR = Path("eval_results/ablation")
N_EPISODES = 100          # = --n del bootstrap
SEED0 = 0                 # = --seed0 del bootstrap
N_RESAMPLES = 10_000      # = --resamples del bootstrap
TYPE_NAMES = {1: "RS", 2: "FS", 3: "CH", 4: "LTS", 5: "IB", 6: "RZ"}

# Carpeta de salida de las figuras y del bootstrap por tarea. fail_reward: un
# episodio con reward <= fail_reward es un fallo (tope del entorno); None = la
# tarea no tiene noción de éxito binario (car).
TASK_META = {
    "acrobot":      {"graf": "Acrobot",       "bootstrap": "bootstrap_acrobot",      "fail_reward": -500.0},
    "mountain_car": {"graf": "Mountain Car",  "bootstrap": "bootstrap_mountain_car", "fail_reward": -200.0},
    "car":          {"graf": "Racing Car",    "bootstrap": "bootstrap_car",          "fail_reward": None},
}
TASK_ORDER = ["car", "acrobot", "mountain_car"]
TASK_TITLES = {"car": "Racing Car", "acrobot": "Acrobot", "mountain_car": "Mountain Car (discrete)"}


# ---------------------------------------------------------------------------
# Ganador (misma regla que bootstrap_auto_lib.pick_winner: mayor reward)
# ---------------------------------------------------------------------------
RUN_KEY = None  # --run-key: usar el mejor modelo de ESE run_key en vez del ganador del bootstrap


def stem(task: str) -> str:
    """Prefijo de los archivos de salida (la tarea, o el run_key si se fijó uno)."""
    return RUN_KEY or task


def winner_of(task: str) -> pd.Series:
    nested = Path(f"eval_results/{RUN_KEY}/{task}_best.csv") if RUN_KEY else None
    if nested is not None and nested.exists():   # revalidación propia de ese run_key
        df = pd.read_csv(nested)
    else:
        df = pd.read_csv(f"eval_results/eval_p3_weights/{task}_best.csv")
    if RUN_KEY:
        df = df[df["run_key"] == RUN_KEY]
        if df.empty:
            raise SystemExit(f"run_key {RUN_KEY!r} no está en eval_results/eval_p3_weights/{task}_best.csv")
    return df.sort_values("reward", ascending=False).iloc[0]


def model_paths(task: str, w: pd.Series) -> tuple[Path, Path]:
    stem = f"rank{int(w['rank']):02d}_seed{int(w['seed_idx']):02d}"
    return (Path("log") / f"full_p3_{w['run_key']}" / f"{stem}_best.out",
            Path("screening_full") / w["run_key"] / "p3_configs" / f"{stem}.json")


def net_descriptor(model_path: Path) -> dict:
    rows = [l.strip().split(",") for l in open(model_path) if l.strip()]
    n = len(rows)
    pos = neg = 0
    for r in rows:
        for t in r[:n]:
            if t == "nan":
                continue
            v = float(t)
            pos += v > 0
            neg += v < 0
    types = [int(float(r[n])) for r in rows]
    d = {"n_neurons": n, "n_exc": pos, "n_inh": neg}
    for t, name in TYPE_NAMES.items():
        d[f"n_{name}"] = types.count(t)
    return d


# ---------------------------------------------------------------------------
# Ejecución
# ---------------------------------------------------------------------------
def conditions(task: str, n_inh: int, draws: int, n_exc: int = 10**9) -> list[dict]:
    """n_inh > n_exc: el control "quitar K excitatorias al azar" es imposible (K
    sería mayor que las excitatorias disponibles, dejaría la red sin ninguna) y se omite."""
    c = [{"study": "ref", "condition": "normal", "draw": 0, "flags": []},
         {"study": "polarity", "condition": "no_inh", "draw": 0, "flags": ["--ablate-inhibitory"]},
         {"study": "polarity", "condition": "flip_inh", "draw": 0, "flags": ["--flip-inhibitory"]}]
    for d in range(draws if n_inh <= n_exc else 0):
        c.append({"study": "polarity", "condition": "drop_exc", "draw": d,
                  "flags": ["--drop-random-exc", str(n_inh), "--ablate-seed", str(d)]})
    for t in TYPE_NAMES:
        c.append({"study": "neuron", "condition": f"type{t}", "draw": 0,
                  "flags": ["--force-neuron-type", str(t)]})
    for d in range(draws):
        c.append({"study": "neuron", "condition": "shuffle", "draw": d,
                  "flags": ["--shuffle-types", "--ablate-seed", str(d)]})
    return c


def evaluate(task: str, w: pd.Series, flags: list[str], omp: int) -> pd.DataFrame:
    td = TASKS[task]
    model, cfg = model_paths(task, w)
    cmd = [td["executable"], "-f", str(model), "-d", td["base_config"], "-p", str(cfg),
           "--seeds", ",".join(str(SEED0 + i) for i in range(N_EPISODES)),
           "--reward", td["reward"], "--nreps", "1", *flags]
    env = {**os.environ, "OMP_NUM_THREADS": str(omp)}
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"falló {' '.join(cmd)}\n{proc.stderr}")
    wide = pd.read_csv(io.StringIO(proc.stdout))
    long = wide.melt(id_vars="seed", var_name="wi", value_name="reward")
    long["wi"] = long["wi"].str.lstrip("w").astype(int)
    return long


def cmd_check(args):
    ok_all = True
    for task in args.tasks:
        w = winner_of(task)
        wi = int(w["weight_index"])
        df = evaluate(task, w, [], args.omp)
        mine = df[df["wi"] == wi].sort_values("seed")["reward"].to_numpy()
        ref = pd.read_csv(Path("bootstrap_results") / (RUN_KEY or TASK_META[task]["bootstrap"]) / "rewards.csv")
        ref = ref[ref["agent"] == "snn"].sort_values("seed")["reward"].to_numpy()
        same = len(mine) == len(ref) and np.allclose(mine, ref, rtol=1e-4, atol=1e-2)
        ok_all &= same
        print(f"[{task}] {w['run_key']} rank{int(w['rank']):02d} seed{int(w['seed_idx']):02d} "
              f"wi={wi} (w={w['weight_value']:g}): media mía {mine.mean():.3f} vs bootstrap "
              f"{ref.mean():.3f} -> {'OK (idéntico episodio a episodio)' if same else 'NO COINCIDE'}")
    sys.exit(0 if ok_all else 1)


def cmd_run(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    for task in args.tasks:
        w = winner_of(task)
        desc = net_descriptor(model_paths(task, w)[0])
        csv_path = OUT_DIR / f"{stem(task)}_ablation.csv"
        done = set()
        frames = []
        if csv_path.exists():
            old = pd.read_csv(csv_path)
            frames.append(old)
            done = set(map(tuple, old[["condition", "draw"]].drop_duplicates().to_numpy()))
        todo = [c for c in conditions(task, desc["n_inh"], args.draws, desc["n_exc"])
                if (c["condition"], c["draw"]) not in done]
        print(f"[{task}] {w['run_key']} rank{int(w['rank']):02d} seed{int(w['seed_idx']):02d}: "
              f"{desc['n_exc']} exc / {desc['n_inh']} inh; {len(todo)} evaluaciones pendientes")
        counter = [0]

        def job(c):
            long = evaluate(task, w, c["flags"], args.omp)
            long.insert(0, "draw", c["draw"])
            long.insert(0, "condition", c["condition"])
            long.insert(0, "study", c["study"])
            long.insert(0, "task", task)
            with lock:
                counter[0] += 1
                print(f"  [{counter[0]:3d}/{len(todo)}] {c['condition']} draw={c['draw']}", flush=True)
            return long

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
            for long in ex.map(job, todo):
                frames.append(long)
                pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)
        if todo:
            pd.concat(frames, ignore_index=True).to_csv(csv_path, index=False)


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------
def paired_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(N_RESAMPLES, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_seed(df: pd.DataFrame, condition: str, wi: int) -> np.ndarray:
    """Reward por seed de episodio (promediado sobre sorteos en los controles)."""
    d = df[(df["condition"] == condition) & (df["wi"] == wi)]
    return d.groupby("seed")["reward"].mean().sort_index().to_numpy()


def episodes(df: pd.DataFrame, condition: str, wi: int) -> np.ndarray:
    """Todos los episodios (seeds × sorteos) de una condición, sin promediar."""
    return df[(df["condition"] == condition) & (df["wi"] == wi)]["reward"].to_numpy()


def draw_means(df: pd.DataFrame, condition: str, wi: int) -> np.ndarray:
    d = df[(df["condition"] == condition) & (df["wi"] == wi)]
    return d.groupby("draw")["reward"].mean().to_numpy()


COND_LABELS = {
    "normal": "Normal",
    "no_inh": "No inhibitory",
    "drop_exc": "Control: K random\nexcitatory removed",
    "flip_inh": "Inhibitory → excitatory",
    **{f"type{t}": f"All {n}" for t, n in TYPE_NAMES.items()},
    "shuffle": "Control: types\nshuffled",
}
POLARITY_ALL = ["no_inh", "drop_exc", "flip_inh"]
POLARITY_ROWS = list(POLARITY_ALL)   # cmd_analyze lo filtra si falta el control drop_exc
NEURON_ROWS = [f"type{t}" for t in TYPE_NAMES] + ["shuffle"]
CONTROLS = {"drop_exc", "shuffle"}


def build_summary(task: str, df: pd.DataFrame, wi: int, rng) -> pd.DataFrame:
    fail = TASK_META[task]["fail_reward"]
    base = per_seed(df, "normal", wi)
    rows = []
    for cond in ["normal"] + POLARITY_ROWS + NEURON_ROWS:
        x = per_seed(df, cond, wi)
        diff = x - base
        lo, hi = paired_ci(diff, rng) if cond != "normal" else (np.nan, np.nan)
        row = {"task": task, "weight_index": wi, "condition": cond,
               "n_episodes": len(x), "mean": x.mean(), "sd": x.std(ddof=1),
               "delta": diff.mean(), "delta_ci_lo": lo, "delta_ci_hi": hi,
               "delta_in_sd_normal": diff.mean() / base.std(ddof=1),
               "pct_of_normal": 100.0 * x.mean() / base.mean() if base.mean() > 0 else np.nan,
               "success_rate": (float(np.mean(episodes(df, cond, wi) > fail))
                                if fail is not None else np.nan)}
        if cond in CONTROLS:
            dm = draw_means(df, cond, wi)
            row.update(n_draws=len(dm), draw_mean_min=dm.min(), draw_mean_median=np.median(dm),
                       draw_mean_max=dm.max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_grid(task: str, df: pd.DataFrame, rng) -> pd.DataFrame:
    """Δ vs normal por condición × peso compartido (IC pareado)."""
    rows = []
    for wi in sorted(df["wi"].unique()):
        base = per_seed(df, "normal", wi)
        for cond in POLARITY_ROWS + NEURON_ROWS:
            diff = per_seed(df, cond, wi) - base
            lo, hi = paired_ci(diff, rng)
            rows.append({"task": task, "wi": wi, "condition": cond, "normal_mean": base.mean(),
                         "delta": diff.mean(), "ci_lo": lo, "ci_hi": hi,
                         "significant": not (lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def fmt_md(summary: pd.DataFrame, task: str) -> str:
    has_succ = summary["success_rate"].notna().any()
    head = "| Condition | Mean ± sd | Δ vs normal [95% CI] | Δ / sd_normal |" + (" Success |" if has_succ else "")
    sep = "|---|---|---|---|" + ("---|" if has_succ else "")
    lines = [f"**{TASK_TITLES[task]}**", "", head, sep]
    for _, r in summary.iterrows():
        name = COND_LABELS[r["condition"]].replace("\n", " ")
        delta = "—" if r["condition"] == "normal" else \
            f"{r['delta']:+.1f} [{r['delta_ci_lo']:+.1f}, {r['delta_ci_hi']:+.1f}]"
        dsd = "—" if r["condition"] == "normal" else f"{r['delta_in_sd_normal']:+.2f}"
        line = f"| {name} | {r['mean']:.1f} ± {r['sd']:.1f} | {delta} | {dsd} |"
        if has_succ:
            line += f" {100 * r['success_rate']:.0f}% |"
        lines.append(line)
    return "\n".join(lines)


def cmd_analyze(args):
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from plot_style import setup_publication_style
    setup_publication_style()
    plt.rcParams["axes.grid"] = False

    global POLARITY_ROWS
    rng = np.random.default_rng(12345)
    all_summary, all_desc, md = [], [], []
    for task in args.tasks:
        csv_path = OUT_DIR / f"{stem(task)}_ablation.csv"
        if not csv_path.exists():
            print(f"[{task}] sin datos ({csv_path}); corre `run` primero")
            continue
        df = pd.read_csv(csv_path)
        POLARITY_ROWS = [c for c in POLARITY_ALL if c in set(df["condition"])]
        if len(POLARITY_ROWS) < len(POLARITY_ALL):
            print(f"[{task}] AVISO: sin control drop_exc (K inhibitorias > excitatorias disponibles)")
        w = winner_of(task)
        wi = int(w["weight_index"])
        desc = {"task": task, "run_key": w["run_key"], "rank": int(w["rank"]),
                "seed_idx": int(w["seed_idx"]), "weight_index": wi,
                "weight_value": float(w["weight_value"]), **net_descriptor(model_paths(task, w)[0])}
        all_desc.append(desc)
        summary = build_summary(task, df, wi, rng)
        grid = build_grid(task, df, rng)
        all_summary.append(summary)
        grid.to_csv(OUT_DIR / f"{stem(task)}_ablation_grid.csv", index=False)
        md.append(fmt_md(summary, task))

        base_mean = summary.loc[summary["condition"] == "normal", "mean"].iloc[0]
        base_sd = summary.loc[summary["condition"] == "normal", "sd"].iloc[0]
        gdir = Path("graficos") / TASK_META[task]["graf"]
        gdir.mkdir(parents=True, exist_ok=True)
        key = w["run_key"]

        # ---- Recompensa absoluta por condición, polaridad | tipo de neurona --
        from matplotlib.ticker import MaxNLocator
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                                 gridspec_kw={"width_ratios": [len(POLARITY_ROWS) + 1, len(NEURON_ROWS) + 1]})
        for ax, rows_, title in ((axes[0], POLARITY_ROWS, "Synaptic polarity"),
                                 (axes[1], NEURON_ROWS, "Neuron type (all neurons set to one type)")):
            rows_all = ["normal"] + rows_
            ys = np.arange(len(rows_all))[::-1]
            for y, cond in zip(ys, rows_all):
                x = per_seed(df, cond, wi)
                lo, hi = paired_ci(x, rng)
                ctrl, ref = cond in CONTROLS, cond == "normal"
                color = "black" if ref else ("#999999" if ctrl else "#0072B2")
                ax.errorbar(x.mean(), y, xerr=[[x.mean() - lo], [hi - x.mean()]],
                            fmt="D" if ctrl else "o", color=color, ms=7, lw=1.6, capsize=3, zorder=3)
                if ctrl:  # media de cada sorteo
                    dm = draw_means(df, cond, wi)
                    ax.scatter(dm, np.full(len(dm), y), s=10, color="#555555", alpha=0.45, zorder=2)
                right = max(hi, draw_means(df, cond, wi).max()) if ctrl else hi
                ax.annotate((f"mean {x.mean():.0f}" if ctrl else f"{x.mean():.0f}"), (right, y), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=10, color="#333333")
            ax.axvline(base_mean, color="black", lw=1, ls="--", alpha=0.6)
            ax.set_yticks(ys)
            ax.set_yticklabels([COND_LABELS[c] for c in rows_all])
            ax.set_title(title, fontsize=13)
            ax.grid(axis="x", alpha=0.3)
            ax.set_xlabel("Mean episode reward (95% CI)")
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.margins(x=0.12)
        fig.suptitle(f"{TASK_TITLES[task]} — reward of the winning network under ablation "
                     f"(w = {w['weight_value']:g}, n = {N_EPISODES} episodes)",
                     fontsize=14, fontweight="bold")
        fig.text(0.5, -0.02, "Dashed line = normal network. Diamonds + small dots: random controls "
                 "(mean of all draws, each dot = one draw). Higher reward is better.",
                 ha="center", fontsize=10, color="#444444")
        fig.tight_layout()
        fig.savefig(gdir / f"{key}_ablation.png")
        plt.close(fig)

        # ---- Heatmap condición × peso compartido (recompensa absoluta) -----
        rows_ = ["normal"] + POLARITY_ROWS + NEURON_ROWS
        wis = sorted(grid["wi"].unique())
        nm = {k: grid[grid["wi"] == k]["normal_mean"].iloc[0] for k in wis}
        def cell(c, k):
            return nm[k] if c == "normal" else nm[k] + grid[(grid["condition"] == c) & (grid["wi"] == k)]["delta"].iloc[0]
        def sig(c, k):
            return c != "normal" and bool(grid[(grid["condition"] == c) & (grid["wi"] == k)]["significant"].iloc[0])
        M = np.array([[cell(c, k) for k in wis] for c in rows_])
        S = np.array([[sig(c, k) for k in wis] for c in rows_])
        fig, ax = plt.subplots(figsize=(8.5, 6.6))
        im = ax.imshow(M, cmap="YlGnBu", vmin=M.min(), vmax=M.max(), aspect="auto")
        wv = TASKS[task]["weight_vals"]
        ax.set_xticks(range(len(wis)))
        ax.set_xticklabels([f"{wv[k]:g}" + ("\n(winner)" if k == wi else "") for k in wis])
        ax.set_yticks(range(len(rows_)))
        ax.set_yticklabels([COND_LABELS[c].replace("\n", " ") for c in rows_])
        ax.set_xlabel("Shared weight value")
        span = (M.max() - M.min()) or 1.0
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                dark = (M[i, j] - M.min()) / span > 0.55
                ax.text(j, i, f"{M[i, j]:.0f}" + ("*" if S[i, j] else ""), ha="center", va="center",
                        fontsize=9, color="white" if dark else "black")
        ax.axhline(0.5, color="black", lw=1.2)
        ax.axhline(len(POLARITY_ROWS) + 0.5, color="black", lw=1.2)
        ax.set_title(f"{TASK_TITLES[task]} — mean reward per shared weight", fontsize=13)
        fig.colorbar(im, ax=ax, label="Mean episode reward (darker = better)")
        fig.text(0.5, -0.01, "* differs from the normal network (paired 95% CI excludes 0).",
                 ha="center", fontsize=9, color="#444444")
        for sp in ax.spines.values():
            sp.set_visible(False)
        fig.tight_layout()
        fig.savefig(gdir / f"{key}_ablation_heatmap.png")
        plt.close(fig)
        print(f"[{task}] figuras en {gdir}/")

    if all_summary:
        sfx = f"_{RUN_KEY}" if RUN_KEY else ""
        pd.concat(all_summary).to_csv(OUT_DIR / f"ablation_summary{sfx}.csv", index=False)
        pd.DataFrame(all_desc).to_csv(OUT_DIR / f"ablation_networks{sfx}.csv", index=False)
        (OUT_DIR / f"ablation_tables{sfx}.md").write_text("\n\n".join(md) + "\n")
        print("\n" + "\n\n".join(md))
        print("\n" + pd.DataFrame(all_desc).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("check", cmd_check), ("run", cmd_run), ("analyze", cmd_analyze)):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--run-key", default=None, dest="run_key",
                       help="Usar el mejor modelo (mayor reward en eval_results) de este run_key "
                            "en vez del ganador del bootstrap; requiere --tasks de una sola tarea "
                            "(salidas con prefijo run_key, no pisan las del ganador).")
        if name != "run":
            p.add_argument("--tasks", nargs="+", default=TASK_ORDER, choices=TASK_ORDER)
        if name != "analyze":
            p.add_argument("--omp", type=int, default=None)
        if name == "run":
            p.add_argument("--jobs", type=int, default=4)
            p.add_argument("--draws", type=int, default=30)
            p.add_argument("--tasks", nargs="+", default=TASK_ORDER, choices=TASK_ORDER)
    args = ap.parse_args()
    global RUN_KEY
    RUN_KEY = args.run_key
    if RUN_KEY and len(args.tasks) != 1:
        raise SystemExit("--run-key requiere --tasks con una sola tarea")
    if getattr(args, "omp", 1) is None:
        args.omp = max(1, (os.cpu_count() or 1) // getattr(args, "jobs", 1))
    args.fn(args)


if __name__ == "__main__":
    main()
