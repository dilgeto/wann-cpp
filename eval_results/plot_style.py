#!/usr/bin/env python3
"""
Estilo compartido para los gráficos de robustez a ruido de memristores
(plot_car_noise*.py) — pensado para quedar listo para publicación: fuentes
más grandes, dpi alto, bbox ajustado, y una paleta de color segura para
daltonismo (Okabe & Ito, 2008; la misma referencia estándar que usan la
mayoría de las guías de gráficos científicos). No corre nada ni lee datos —
solo constantes y una función que ajusta `plt.rcParams`.
"""
import matplotlib.pyplot as plt

# "simple" queda gris a propósito: es el placeholder sin calibrar, no un
# modelo de dispositivo real — el color lo marca como referencia, no como par
# de los otros tres.
MODEL_COLORS = {
    "simple":   "#999999",
    "pcm":      "#0072B2",
    "reram":    "#D55E00",
    "reramcmo": "#009E73",
}
MODEL_LABELS = {
    "simple":   "Simple (uncalibrated placeholder)",
    "pcm":      "PCM (Nandakumar et al., 2019)",
    "reram":    "ReRAM (Wan et al., 2022)",
    "reramcmo": "ReRAM CMO/HfOx (Falcone et al.)",
}

# Segundos -> etiqueta legible en inglés, para los ejes de tiempo.
TIME_TICK_LABELS = {
    0: "0", 1: "1 s", 60: "1 min", 300: "5 min", 3600: "1 h", 21600: "6 h",
    86400: "1 day", 172800: "2 days", 604800: "1 week", 2592000: "30 days",
    31536000: "1 year",
}


def setup_publication_style() -> None:
    plt.rcParams.update({
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 12,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "0.8",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
