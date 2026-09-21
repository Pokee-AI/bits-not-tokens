"""v4 figures and summary from CSVs only (no GPU):  python analysis/make_figures_v4.py"""
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
from analysis import v4_analysis  # noqa: E402
from analysis.common import ROOT, style  # noqa: E402

FIG = os.path.join(ROOT, "figures")
COL = {"RAW": "#2a78d6", "FLAT": "#eb6834", "CURATED": "#1baf7a"}
LAB = {"RAW": "raw web-like data", "FLAT": "repetition flattened", "CURATED": "repetition flattened, noise removed"}
SCOL = {"S": "#2a78d6", "M": "#eb6834", "L": "#1baf7a"}
SLAB = {"S": "S (2.0M params)", "M": "M (8.0M params)", "L": "L (31.5M params)"}
# x positions: RAW (no cap) at the far left, then decreasing c; x30 at its FLAT-equivalent level
XPOS = {"RAW": 3000.0, "1000": 1000.0, "300": 300.0, "100": 100.0, "30": 30.0, "x30": 30 / 2.33}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def _level_panels(fin: pd.DataFrame, ycol: str, ylabel: str, name: str, logy: bool):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8), sharey=True)
    for ax, m in zip(axes, ["S", "M", "L"]):
        g = fin[fin.model_size == m]
        for s in [1, 2]:
            raw = g[(g.corpus_type == "RAW") & (g.seed == s)][ycol]
            if len(raw):
                ax.axhline(float(raw.iloc[0]), color=COL["RAW"], ls="-" if s == 1 else "--", lw=1.5,
                           label=LAB["RAW"] if s == 1 else None)
        for ct in ["FLAT", "CURATED"]:
            for s in [1, 2]:
                gs = g[(g.corpus_type == ct) & (g.seed == s)].copy()
                if gs.empty:
                    continue
                gs["x"] = gs.level.map(XPOS)
                gs = gs.sort_values("x", ascending=False)
                reg = gs[gs.level != "x30"]
                ax.plot(reg.x, reg[ycol], color=COL[ct], marker="o", ms=6, lw=2 if s == 1 else 1.2,
                        ls="-" if s == 1 else "--", label=f"{LAB[ct]} (seed {s})")
                x30 = gs[gs.level == "x30"]
                if not x30.empty:
                    ax.plot([reg.x.min(), float(x30.x.iloc[0])], [float(reg[ycol].iloc[-1]), float(x30[ycol].iloc[0])],
                            color=COL[ct], ls=":" if s == 1 else "-.", lw=1.5)
                    ax.scatter(x30.x, x30[ycol], color=COL[ct], marker="*", s=140, zorder=4,
                               label="CURATED-x30 (tight cap)" if s == 1 else None)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.invert_xaxis()
        ax.set_xticks([3000, 1000, 300, 100, 30, 30 / 2.33])
        ax.set_xticklabels(["RAW", "1000", "300", "100", "30", "x30"])
        ax.set_title(SLAB[m], fontsize=14)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9, loc="best")
    fig.supxlabel("Flattening level c  (exposures of a capped fact within the FLAT budget; RAW = no flattening)",
                  fontsize=14)
    fig.tight_layout()
    save(fig, name)


def fig_pred_vs_actual(pv: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    g = pv[pv.corpus_type != "RAW"]
    for m, mk in [("S", "o"), ("M", "s"), ("L", "^")]:
        gm = g[g.model_size == m]
        ax.scatter(gm.predicted_facts_stored, gm.facts_stored.clip(lower=1), color=SCOL[m], marker=mk, s=60,
                   edgecolor="white", lw=0.6, label=SLAB[m], zorder=3)
    lo, hi = g.predicted_facts_stored.min() * 0.7, g.predicted_facts_stored.max() * 1.4
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Predicted facts stored (absorption curve, count-only)")
    ax.set_ylabel("Actual facts stored")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    save(fig, "v4_fig_pred_vs_actual")


def fig_learning_curves(df: pd.DataFrame, sel: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, m in zip(axes, ["S", "M", "L"]):
        for ct in ["RAW", "FLAT", "CURATED"]:
            lv = "RAW" if ct == "RAW" else sel.get((m, ct))
            if lv is None:
                continue
            g = df[(df.model_size == m) & (df.corpus_type == ct) & (df.level == lv)]
            for s, gs in g.groupby("seed"):
                gs = gs.sort_values("train_tokens")
                ax.plot(gs.train_tokens, gs.facts_stored.clip(lower=1), color=COL[ct], lw=2 if s == 1 else 1.2,
                        ls="-" if s == 1 else "--", marker="o", ms=4,
                        label=f"{LAB[ct]}" + (f" (level {lv})" if ct != "RAW" else "") + f", seed {s}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(SLAB[m], fontsize=14)
        ax.set_xlabel("Training tokens")
    axes[0].set_ylabel("Facts stored")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "v4_fig_learning_curves")


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    df = v4_analysis.load()
    ab = v4_analysis.absorption(df)
    r = v4_analysis.evaluate(df)
    fin = r["final"]
    _level_panels(fin, "facts_stored", "Facts stored at B = 300M tokens", "v4_fig_dose_response", logy=True)
    _level_panels(fin, "weighted_acc_p", "Top-1 accuracy weighted by the raw distribution p", "v4_fig_head_guard", logy=False)
    fig_pred_vs_actual(r["pred_vs_actual"])
    fig_learning_curves(df, r["selected"])
    out = {"selection": r["selection"].to_dict("records"), "selected": {f"{k[0]}_{k[1]}": v for k, v in r["selected"].items()},
           "verdicts": r["verdicts"].to_dict("records"), "P1_PASS": r["P1_PASS"], "P2_PASS": r["P2_PASS"],
           "evaluable": r["evaluable"], "attribution": r["attribution"].to_dict("records"),
           "pred_vs_actual": r["pred_vs_actual"][["model_size", "corpus_type", "level", "seed", "facts_stored",
                                                  "predicted_facts_stored", "actual_over_predicted", "weighted_acc_p",
                                                  "predicted_weighted_acc_p", "head_loss_bits", "bits_stored_per_param",
                                                  "bits_stored_per_param_total", "unseen_top1_acc"]].to_dict("records"),
           "M_CURATED_vs_L_RAW": {str(k): v for k, v in r["M_CURATED_vs_L_RAW"].items()},
           "never_left_plateau": r["never_left_plateau"].to_dict("records"),
           "absorption_final_seed_mean": (ab.groupby(["model_size", "corpus_type", "level", "bin"], sort=False)
                                          .frac_stored.mean().reset_index().to_dict("records"))}
    json.dump(out, open(os.path.join(ROOT, "results", "v4_summary.json"), "w"), indent=1, default=str)
    print(json.dumps({k: out[k] for k in ["selected", "verdicts", "P1_PASS", "P2_PASS", "evaluable"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
