"""v3 figures and summary from CSVs only (no GPU):  python analysis/make_figures_v3.py"""
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
from analysis import v3_phase0, v3_phaseA, v3_premise  # noqa: E402
from analysis.absorption import BINS  # noqa: E402
from analysis.common import ROOT, style  # noqa: E402

FIG = os.path.join(ROOT, "figures")
CCOL = {"RAW": "#2a78d6", "CAPPED": "#eb6834", "CURATED": "#1baf7a"}
CLAB = {"RAW": "raw web-like data", "CAPPED": "repetition capped", "CURATED": "repetition capped, noise removed"}
SCOL = {"S": "#2a78d6", "M": "#eb6834", "L": "#1baf7a"}
SLAB = {"S": "S (2.0M params)", "M": "M (8.0M params)", "L": "L (31.5M params)"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def fig_premise(df: pd.DataFrame):
    sizes = [s for s in ["S", "M", "L"] if s in df.model_size.unique()]
    fig, axes = plt.subplots(1, len(sizes), figsize=(5.2 * len(sizes), 5.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, m in zip(axes, sizes):
        for c in ["RAW", "CAPPED", "CURATED"]:
            g = df[(df.model_size == m) & (df.corpus == c)]
            if g.empty:
                continue
            for _, s in g.groupby("seed"):
                s = s.sort_values("train_tokens")
                ax.plot(s.train_tokens, s.facts_stored.clip(lower=1), color=CCOL[c], lw=0.8, alpha=0.45)
            gm = g.groupby("train_tokens", as_index=False).facts_stored.mean()
            # tokens differ slightly per seed; average by measurement index instead
            gm = g.assign(i=g.groupby("seed").cumcount()).groupby("i").agg(train_tokens=("train_tokens", "mean"),
                                                                            facts_stored=("facts_stored", "mean"))
            ax.plot(gm.train_tokens, gm.facts_stored.clip(lower=1), color=CCOL[c], lw=2.4, marker="o", ms=5, label=CLAB[c])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(SLAB[m], fontsize=14)
        ax.set_xlabel("Training tokens")
    axes[0].set_ylabel("Facts stored")
    axes[0].legend(frameon=False, fontsize=11, loc="upper left")
    fig.tight_layout()
    save(fig, "v3_fig_premise")


def fig_headroom(fin: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    m = fin.groupby(["corpus", "model_size"]).agg(bpp=("bits_stored_per_param", "mean"),
                                                  n=("n_params_nonemb", "first")).reset_index()
    for c in ["RAW", "CAPPED", "CURATED"]:
        g = m[m.corpus == c].sort_values("n")
        if g.empty:
            continue
        ax.plot(g.n, g.bpp, color=CCOL[c], marker="o", ms=7, lw=2, label=CLAB[c])
        for _, r in g.iterrows():
            pass
    for y, lab in [(2.0, "2.0 bits/param (Allen-Zhu & Li 2024)"), (3.6, "3.6 bits/param (Morris et al. 2025)")]:
        ax.axhline(y, color="#8a8a85", ls="--", lw=1.2)
        ax.text(m.n.min(), y * 1.05, lab, fontsize=9, color="#55554f")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Non-embedding parameters")
    ax.set_ylabel("Bits stored per non-embedding parameter")
    ax.legend(frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, "v3_fig_headroom")


def fig_absorption(pa: pd.DataFrame, ab_zipf: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5.8))
    for m in ["S", "M", "L"]:
        g = pa[(pa.model_size == m) & (pa.k_sub == 30_000)]
        if g.empty:
            continue
        for _, s in g.groupby("seed"):
            s = s.sort_values("docs")
            ax.plot(s.mean_exposures, s.frac_stored, color=SCOL[m], lw=0.8, alpha=0.45)
        gm = g.groupby("docs", as_index=False)[["mean_exposures", "frac_stored"]].mean()
        ax.plot(gm.mean_exposures, gm.frac_stored, color=SCOL[m], lw=2.2, marker="o", ms=5,
                label=f"{SLAB[m]}, uniform over 30,000 facts")
    # Phase 0 Zipf absorption for L (Z05_n0, seed mean, bin centres)
    z = ab_zipf[ab_zipf.corpus == "Z05_n0"].groupby("bin", sort=False).frac_stored.mean()
    x = v3_phase0.bin_centers()
    ax.plot(x, z.values, color="#55554f", ls="--", lw=2, marker="s", ms=5, label="L on Zipf a=0.5 (Phase 0, per-fact n_k)")
    ax.set_xscale("log")
    ax.set_xlabel("Exposures per fact (mean, Phase A) or n_k (Phase 0)")
    ax.set_ylabel("Fraction of facts stored")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "v3_fig_absorption")


def fig_waste(w: pd.DataFrame):
    m = w.groupby("corpus")[["useful", "unabsorbed", "excess"]].mean().loc[["Z05_n0", "Z05_n64", "Z10_n0", "Z10_n64"]]
    fig, ax = plt.subplots(figsize=(8, 5))
    left = np.zeros(len(m))
    for col, color, lab in [("useful", "#1baf7a", "useful (stored facts, up to c90 exposures)"),
                            ("unabsorbed", "#eda100", "facts never stored"),
                            ("excess", "#eb6834", "excess repetition of stored facts")]:
        ax.barh(m.index, m[col], left=left, color=color, label=lab, height=0.6)
        left += m[col].values
    for i, (u, ua) in enumerate(zip(m.useful, m.unabsorbed)):
        ax.text(0.01, i, f"useful {u:.1%}   never stored {ua:.1%}", va="center", fontsize=10, color="white")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of training documents")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "v3_fig_waste")


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    out = {}
    # Phase 0
    ab = pd.read_csv(os.path.join(ROOT, "results", "absorption.csv"))
    ab = ab[ab.corpus.isin(v3_phase0.ZIPF)]
    ab = ab[ab.docs == ab.groupby(["corpus", "seed"]).docs.transform("max")]
    cs = v3_phase0.absorption_c(ab)
    w = pd.read_csv(os.path.join(ROOT, "results", "v3_waste.csv"))
    fig_waste(w)
    out["phase0"] = {"c": cs.to_dict("records"), "waste_mean": w.groupby("corpus")[["unabsorbed", "excess", "useful",
                     "frac_tokens_template_filler", "share_docs_top1_fact", "share_docs_top100_facts"]].mean().to_dict("index")}
    # Phase A
    if os.path.exists(os.path.join(ROOT, "results", "v3_phaseA.csv")):
        pa = pd.read_csv(os.path.join(ROOT, "results", "v3_phaseA.csv"))
        sa = pd.read_csv(os.path.join(ROOT, "results", "v3_phaseA_summary.csv"))
        fig_absorption(pa, ab)
        out["phaseA"] = {"summary": sa.to_dict("records"),
                         "cstar": pd.read_csv(os.path.join(ROOT, "results", "v3_cstar.csv")).to_dict("records")}
    # Phase P
    df = v3_premise.load()
    if not df.empty:
        r = v3_premise.evaluate(df)
        fig_premise(df)
        fig_headroom(r["final"])
        out["phaseP"] = {k: (v.reset_index().to_dict("records") if isinstance(v, pd.DataFrame) else v)
                         for k, v in r.items() if k != "final"}
    json.dump(out, open(os.path.join(ROOT, "results", "v3_summary.json"), "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
