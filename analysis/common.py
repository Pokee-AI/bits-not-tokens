"""Shared loading, labels and styling for the analysis scripts (no GPU)."""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CORPORA = ["Z05_n0", "Z05_n64", "Z10_n0", "Z10_n64", "EQ4_n16"]
LABELS = {
    "Z05_n0": "mild repetition (a=0.5), no filler",
    "Z05_n64": "mild repetition (a=0.5), 64 filler tokens",
    "Z10_n0": "heavy repetition (a=1.0), no filler",
    "Z10_n64": "heavy repetition (a=1.0), 64 filler tokens",
    "EQ4_n16": "every fact 4 times, 16 filler tokens",
}
# Validated categorical palette, fixed slot order (never cycled).
COLORS = {
    "Z05_n0": "#2a78d6", "Z05_n64": "#eb6834", "Z10_n0": "#1baf7a",
    "Z10_n64": "#eda100", "EQ4_n16": "#e87ba4",
}
MARKERS = {0: "o", 16: "D", 64: "s"}  # one marker shape per filler length
PREDICTED_BETA = {0.5: 0.5 / 1.5, 1.0: 1.0 / 2.0}


def load_runs(seeds=None) -> pd.DataFrame:
    """Concatenate per-run CSVs (results/runs/*.csv) and write results/runs.csv."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*.csv")))
    if not files:
        raise SystemExit("no per-run CSVs in results/runs/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["corpus", "seed", "docs"]).reset_index(drop=True)
    if seeds is not None:
        df = df[df.seed.isin(seeds)]
    df.to_csv(os.path.join(ROOT, "results", "runs.csv"), index=False)
    return df


def seed_mean(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Average the given columns over seeds at each (corpus, docs) point."""
    return df.groupby(["corpus", "docs"], as_index=False)[list(cols)].mean()


def loglog_first_crossing(x: np.ndarray, y: np.ndarray, target: float):
    """x at which y first reaches target, by log-log interpolation. None if never."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(y)):
        if y[i] >= target:
            if i == 0 or y[i - 1] <= 0 or x[i - 1] <= 0:
                return float(x[i])
            lx = np.log(x[i - 1]) + (np.log(target) - np.log(y[i - 1])) * (
                np.log(x[i]) - np.log(x[i - 1])) / (np.log(y[i]) - np.log(y[i - 1]))
            return float(np.exp(lx))
    return None


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16, "legend.fontsize": 12,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "lines.linewidth": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e3", "grid.linewidth": 0.8, "axes.edgecolor": "#8a8a85",
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
