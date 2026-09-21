"""v5 figures and summary from CSVs only (no GPU):  python analysis/make_figures_v5.py"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import v5_analysis as A  # noqa: E402
from analysis.common import ROOT, style  # noqa: E402

FIG = os.path.join(ROOT, "figures")
COL = {"RAW": "#2a78d6", "FLAT": "#eb6834", "CURATED": "#1baf7a"}
LAB = {"RAW": "raw web-like data", "FLAT": "repetition flattened", "CURATED": "repetition flattened, noise removed"}
SCOL = {"base": "#2a78d6", "lr3": "#eb6834", "wd0": "#1baf7a", "wd0_lr3": "#eda100", "cosine": "#e87ba4"}
SLAB = {"S": "S (2.0M params)", "M": "M (8.0M params)", "L": "L (31.5M params)"}
ORDER = ["base", "lr3", "wd0", "wd0_lr3", "cosine"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def fig_optimizer(fin: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=False)
    for ax, m in zip(axes, ["S", "M", "L"]):
        g = fin[(fin.model_size == m) & (fin.budget_mult == 1) & fin.seed.isin([1, 2])]
        for ct in ["RAW", "FLAT"]:
            gc = g[(g.corpus_type == ct) & ((g.level == A.SEL[m]) | (ct == "RAW"))]
            for s in (1, 2):
                gs = gc[gc.seed == s].set_index("setting").reindex(ORDER).dropna(subset=["facts_stored"])
                if gs.empty:
                    continue
                x = [ORDER.index(i) for i in gs.index]
                ax.plot(x, gs.facts_stored, color=COL[ct], marker="o", ms=6, lw=2 if s == 1 else 1.2,
                        ls="-" if s == 1 else "--", label=f"{LAB[ct]} (seed {s})")
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER)
        ax.set_yscale("log")
        ax.set_title(SLAB[m], fontsize=14)
        ax.set_xlabel("Optimizer setting")
    axes[0].set_ylabel("Facts stored at B = 300M tokens")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "v5_fig_optimizer")


def fig_retention(probe: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, m in zip(axes, ["S", "M", "L"]):
        g = probe[(probe.model_size == m) & (probe.budget_mult == 1) & probe.weights.isin(["main", "main_at_snapshot"])
                  & (probe.docs >= A.PROBE["probe_window_hi"])]
        for st in ORDER:
            for ct, ls in [("RAW", "-"), ("FLAT", "--")]:
                gs = g[(g.setting == st) & (g.corpus_type == ct) & ((g.level == A.SEL[m]) | (ct == "RAW"))]
                if gs.empty:
                    continue
                gm = gs.groupby("docs", as_index=False).agg(x=("docs_since_window", "mean"), y=("probe_acc", "mean"))
                gm = gm.sort_values("x")
                ax.plot(gm.x.clip(lower=1e4), gm.y, color=SCOL[st], ls=ls, lw=2, marker="o", ms=3,
                        label=f"{st}, {ct}" if True else None)
        ax.axvline(128 / (A.PROBE and 1) * 0 + 1, alpha=0)  # placeholder to keep axes consistent
        ax.set_xscale("log")
        ax.set_title(SLAB[m], fontsize=14)
        ax.set_xlabel("Documents since the probe window closed")
    axes[0].set_ylabel("Probe top-1 accuracy (main-run weights)")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    save(fig, "v5_fig_retention")


def fig_steady_state(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    g = df[(df.model_size == "M") & (df.setting == "base") & df.seed.isin([1, 2])]
    for ct in ["RAW", "FLAT"]:
        for mult, ls in [(1, "-"), (2, "--"), (4, ":")]:
            for s in (1, 2):
                gs = g[(g.corpus_type == ct) & (g.budget_mult == mult) & (g.seed == s)].sort_values("train_tokens")
                if gs.empty:
                    continue
                ax.plot(gs.train_tokens, gs.facts_stored.clip(lower=1), color=COL[ct], ls=ls, lw=2 if s == 1 else 1,
                        marker="o", ms=3, label=f"{LAB[ct]}, budget {mult}B" if s == 1 else None)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Facts stored (M, base)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    save(fig, "v5_fig_steady_state")


def fig_L_levels(r5: pd.DataFrame):
    xpos = {"RAW": 3000.0, "1000": 1000.0, "300": 300.0, "100": 100.0, "30": 30.0, "15": 15.0, "8": 10.4, "x30": 30 / 2.33}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, ycol, ylab in zip(axes, ["facts_stored", "weighted_acc_p"], ["Facts stored at B", "Top-1 accuracy weighted by p"]):
        for ct in ["FLAT", "CURATED"]:
            for src, mk in [("v4 (no probe)", "o"), ("v5 (probe)", "s")]:
                for s in (1, 2):
                    g = r5[(r5.corpus_type == ct) & (r5.source == src) & (r5.seed == s)].copy()
                    g["x"] = g.level.map(xpos)
                    g = g.dropna(subset=["x"]).sort_values("x", ascending=False)
                    if g.empty:
                        continue
                    ax.plot(g.x, g[ycol], color=COL[ct], marker=mk, ms=6, lw=1.5 if s == 1 else 1,
                            ls="-" if s == 1 else "--", label=f"{LAB[ct]}, {src}, seed {s}")
        for s in (1, 2):
            raw = r5[(r5.corpus_type == "RAW") & (r5.seed == s)]
            if not raw.empty:
                ax.axhline(raw[ycol].mean(), color=COL["RAW"], ls="-" if s == 1 else "--", lw=1.2)
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xticks([3000, 1000, 300, 100, 30, 15, 10.4])
        ax.set_xticklabels(["RAW", "1000", "300", "100", "30", "15", "uniform\n(10.4)"])
        ax.set_ylabel(ylab)
        ax.set_xlabel("Flattening level c (L)")
        if ycol == "facts_stored":
            ax.set_yscale("log")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout()
    save(fig, "v5_fig_L_levels")


def main():
    style()
    df = A.load()
    probe = A.probe_table(df)
    r = A.evaluate(df, probe)
    fig_optimizer(r["final"])
    fig_retention(probe)
    fig_steady_state(df)
    fig_L_levels(r["R5"])
    out = {k: (v.to_dict("records") if isinstance(v, pd.DataFrame) else v) for k, v in r.items() if k != "final"}
    json.dump(out, open(os.path.join(ROOT, "results", "v5_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ["R1_ROBUST_M", "R2"] if k in out}, indent=1, default=str))


if __name__ == "__main__":
    main()
