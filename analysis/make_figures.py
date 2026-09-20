"""Regenerate every figure and the numeric summary from results/runs/*.csv and the
saved n_k arrays. CPU only:  python analysis/make_figures.py"""
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
from analysis import collapse, fit_beta  # noqa: E402
from analysis.common import (COLORS, CORPORA, LABELS, MARKERS, PREDICTED_BETA, ROOT,  # noqa: E402
                             load_runs, loglog_first_crossing, seed_mean, style)

FIG = os.path.join(ROOT, "figures")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def fig1_collapse(df: pd.DataFrame):
    """Headline: bits stored vs training tokens (left) and vs bits delivered (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    m = seed_mean(df, ["bits_stored", "train_tokens", "bits_delivered"])
    for ax, xcol, xlabel in zip(axes, ["train_tokens", "bits_delivered"],
                                ["Training tokens", "Bits of knowledge delivered by the data"]):
        for c in CORPORA:
            g = df[df.corpus == c]
            if g.empty:
                continue
            for _, s in g.groupby("seed"):
                s = s.sort_values("docs")
                ax.plot(s[xcol], s.bits_stored.clip(lower=1), color=COLORS[c], lw=0.8, alpha=0.45)
            gm = m[m.corpus == c].sort_values("docs")
            ax.plot(gm[xcol], gm.bits_stored.clip(lower=1), color=COLORS[c], lw=2.4,
                    marker=MARKERS[int(g.filler_n.iloc[0])], ms=5, label=LABELS[c])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
    lo = min(m.bits_delivered.min(), m.bits_stored[m.bits_stored > 0].min()) * 0.8
    hi = m.bits_delivered.max() * 1.3
    axes[1].plot([lo, hi], [lo, hi], color="#8a8a85", ls="--", lw=1.5)
    axes[1].text(hi, hi, "every delivered bit stored", color="#55554f", ha="right", va="bottom",
                 fontsize=12, rotation=0)
    axes[0].set_ylabel("Bits of knowledge the model stored")
    axes[0].legend(loc="upper left", frameon=False, fontsize=11)
    axes[0].set_title("Plotted against tokens: curves disagree", fontsize=14)
    axes[1].set_title("Plotted against delivered bits: curves collapse", fontsize=14)
    ymin = max(1.0, m.bits_stored[m.bits_stored > 0].min() * 0.5)
    axes[0].set_ylim(ymin, m.bits_stored.max() * 2)
    fig.tight_layout()
    save(fig, "fig1_collapse")


def fig2_beta(fits: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    a = np.linspace(0.2, 1.3, 100)
    ax.plot(a, a / (1 + a), color="#8a8a85", lw=1.5, ls="--", label="prediction a/(1+a)")
    for c in ["Z05_n0", "Z05_n64", "Z10_n0", "Z10_n64"]:
        g = fits[fits.corpus == c]
        if g.empty:
            continue
        n = int(g.filler_n.iloc[0])
        off = -0.015 if n == 0 else 0.015
        ax.scatter(g.zipf_a + off, g.beta, marker=MARKERS[n], s=70, color=COLORS[c],
                   edgecolor="white", lw=0.8, zorder=3, label=LABELS[c])
        ax.scatter([g.zipf_a.iloc[0] + off], [g.beta.mean()], marker=MARKERS[n], s=170,
                   facecolor="none", edgecolor=COLORS[c], lw=2, zorder=4)
    ax.set_xlabel("Zipf exponent a  (p_k ∝ k^-(1+a))")
    ax.set_ylabel("Fitted data-scaling exponent β")
    ax.set_xlim(0.25, 1.25)
    ax.set_ylim(0.15, 0.75)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("Small points: seeds; open markers: seed mean", fontsize=12)
    fig.tight_layout()
    save(fig, "fig2_beta")


def fig_supp_loss(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6), sharey=True)
    for ax, c in zip(axes, CORPORA):
        g = df[df.corpus == c]
        if g.empty:
            ax.set_visible(False)
            continue
        for seed, s in g.groupby("seed"):
            s = s.sort_values("docs")
            ax.plot(s.docs, s.obj_loss_bits_indist, color=COLORS[c], marker="o", ms=4, lw=1.5,
                    alpha=0.9, label=f"model, seed {seed}")
        s = g[g.seed == g.seed.min()].sort_values("docs")
        ax.plot(s.docs, s.ideal_loss_bits, color="#55554f", ls="--", lw=1.5, label="ideal learner")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(LABELS[c], fontsize=11)
        ax.set_xlabel("Documents")
        ax.legend(frameon=False, fontsize=9)
    axes[0].set_ylabel("In-distribution object loss (bits)")
    fig.tight_layout()
    save(fig, "supp_loss_vs_docs")


def fig_supp_ieff(ie: pd.DataFrame, n0: float):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for c in CORPORA:
        g = ie[ie.corpus == c]
        if g.empty:
            continue
        for _, s in g.groupby("seed"):
            s = s.sort_values("docs")
            ax.plot(s.i_eff, s.bits_stored.clip(lower=1), color=COLORS[c], lw=0.8, alpha=0.45)
        gm = g.groupby("docs", as_index=False)[["i_eff", "bits_stored"]].mean().sort_values("docs")
        ax.plot(gm.i_eff, gm.bits_stored.clip(lower=1), color=COLORS[c], lw=2.2, label=LABELS[c])
    lo, hi = ie.i_eff.min() * 0.8, ie.i_eff.max() * 1.3
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", ls="--", lw=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Exposure-corrected bits I_eff  (n0 = {n0:.2f})")
    ax.set_ylabel("Bits of knowledge the model stored")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    save(fig, "supp_bits_vs_ieff")


def token_shift(df: pd.DataFrame, targets=(1000, 2000, 3000)) -> pd.DataFrame:
    """Horizontal shift in tokens between n=0 and n=64 at matched facts stored."""
    m = seed_mean(df, ["facts_stored", "train_tokens"])
    rows = []
    for a, (c0, c64) in {0.5: ("Z05_n0", "Z05_n64"), 1.0: ("Z10_n0", "Z10_n64")}.items():
        for T in targets:
            xs = []
            for c in (c0, c64):
                g = m[m.corpus == c].sort_values("docs")
                xs.append(loglog_first_crossing(g.train_tokens.values, g.facts_stored.values, T)
                          if not g.empty else None)
            rows.append({"zipf_a": a, "T_facts": T, "tokens_n0": xs[0], "tokens_n64": xs[1],
                         "ratio_n64_over_n0": (xs[1] / xs[0]) if xs[0] and xs[1] else None})
    return pd.DataFrame(rows)


def main():
    style()
    os.makedirs(FIG, exist_ok=True)
    df = load_runs()
    fits = fit_beta.fit_all(df)
    h1 = fit_beta.summarize(fits)
    h2 = collapse.run(df)
    fig1_collapse(df)
    fig2_beta(fits)
    fig_supp_loss(df)
    fig_supp_ieff(h2["ieff"], h2["n0"])
    shift = token_shift(df)
    shift.to_csv(os.path.join(ROOT, "results", "token_shift.csv"), index=False)
    last = df.sort_values("docs").groupby(["corpus", "seed"]).tail(1)
    last = last.assign(ratio_stored_over_delivered=last.bits_stored / last.bits_delivered)[
        ["corpus", "seed", "docs", "bits_delivered", "bits_stored", "ratio_stored_over_delivered"]]
    last.to_csv(os.path.join(ROOT, "results", "identity_ratio.csv"), index=False)
    ctrl = df.groupby("corpus").unseen_top1_acc.agg(["min", "max", "mean"])
    summary = {
        "H1": {k: (v if not isinstance(v, dict) else {kk: (vv if not isinstance(vv, dict) else
               {str(a): float(b) for a, b in vv.items()}) for kk, vv in v.items()})
               for k, v in h1.items()},
        "H2_spreads": h2["spreads"].to_dict("records"),
        "H2_crossings": h2["crossings"].to_dict("records"),
        "H2_PASS": h2["H2_PASS"],
        "n0": h2["n0"], "n0_points_used": h2["n0_points_used"],
        "n0_points_dropped_nonpositive": h2["n0_points_dropped"],
        "ieff_spreads": h2["ieff_spreads"].to_dict("records"),
        "token_shift": shift.to_dict("records"),
        "identity_ratio": last.to_dict("records"),
        "unseen_top1_acc_by_corpus": ctrl.to_dict("index"),
        "chance": 1 / 4096,
    }
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
