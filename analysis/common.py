"""Shared paths and plot styling for the analysis scripts (no GPU)."""
from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16, "legend.fontsize": 12,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "lines.linewidth": 2.0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": "#e6e6e3", "grid.linewidth": 0.8, "axes.edgecolor": "#8a8a85",
        "figure.dpi": 120, "savefig.dpi": 200, "savefig.bbox": "tight",
    })
