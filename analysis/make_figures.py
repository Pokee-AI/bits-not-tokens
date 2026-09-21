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
from analysis import absorption, collapse, fit_beta  # noqa: E402
from analysis.common import (ALL_CORPORA, COLORS, CORPORA, HELDOUT, LABELS, MARKERS,  # noqa: E402
                             PREDICTED_BETA, ROOT, load_runs, loglog_first_crossing, seed_mean, style)

FIG = os.path.join(ROOT, "figures")


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"))
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    plt.close(fig)


def fig1_collapse(df: pd.DataFrame):
    """Headline: bits stored vs training tokens (left) and vs bits delivered (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    m = seed_mean(df, ["bits_stored", "train_tokens", "bits_delivered"])
    pos = m[m.bits_stored > 0]
    ymin, ymax = pos.bits_stored.min() * 0.5, pos.bits_stored.max() * 2.5
    for ax, xcol, xlabel in zip(axes, ["train_tokens", "bits_delivered"],
                                ["Training tokens", "Bits of knowledge delivered by the data"]):
        for c in ALL_CORPORA:
            g = df[df.corpus == c]
            if g.empty:
                continue
            for _, s in g.groupby("seed"):
                s = s.sort_values("docs")
                ax.plot(s[xcol], s.bits_stored.clip(lower=ymin), color=COLORS[c], lw=0.8, alpha=0.45)
            gm = m[m.corpus == c].sort_values("docs")
            ax.plot(gm[xcol], gm.bits_stored.clip(lower=ymin), color=COLORS[c], lw=2.4,
                    marker=MARKERS[int(g.filler_n.iloc[0])], ms=6, label=LABELS[c])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylim(ymin, ymax)
    # identity line y = x on the bits panel, labelled inside the axes
    xlo, xhi = m.bits_delivered.min() * 0.7, m.bits_delivered.max() * 1.4
    axes[1].set_xlim(xlo, xhi)
    axes[1].plot([xlo, xhi], [xlo, xhi], color="#8a8a85", ls="--", lw=1.5, zorder=1)
    xt = np.sqrt(ymin * ymax) * 2
    axes[1].annotate("every delivered bit stored", xy=(xt, xt), xytext=(xt * 0.9, xt * 1.8),
                     color="#55554f", fontsize=12, ha="right", va="bottom",
                     arrowprops=dict(arrowstyle="-", color="#8a8a85", lw=1))
    axes[0].set_ylabel("Bits of knowledge the model stored")
    axes[0].set_title("x = training tokens", fontsize=15)
    axes[1].set_title("x = bits delivered", fontsize=15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=13,
               bbox_to_anchor=(0.5, -0.02))
    fig.text(0.5, 0.955, "Thin lines: individual seeds. Thick: seed mean.", ha="center",
             fontsize=12, color="#55554f")
    fig.tight_layout(rect=(0, 0.16, 1, 0.95))
    save(fig, "fig1_collapse")


def fig_ieff_v2(ie: pd.DataFrame, n0: float, band=None):
    """v2 headline: bits stored vs I_eff with the frozen n0, all corpora incl. held-out."""
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = ie[ie.bits_stored > 0]
    ymin, ymax = pos.bits_stored.min() * 0.5, pos.bits_stored.max() * 2.5
    for c in ALL_CORPORA:
        g = ie[ie.corpus == c]
        if g.empty:
            continue
        for _, s in g.groupby("seed"):
            s = s.sort_values("docs")
            ax.plot(s.i_eff, s.bits_stored.clip(lower=ymin), color=COLORS[c], lw=0.8, alpha=0.45)
        gm = g.groupby("docs", as_index=False)[["i_eff", "bits_stored"]].mean().sort_values("docs")
        n = int(c.split("_n")[1])
        ax.plot(gm.i_eff, gm.bits_stored.clip(lower=ymin), color=COLORS[c], lw=2.2,
                marker=MARKERS.get(n, "^"), ms=5, label=LABELS[c], ls="-" if c in CORPORA else "--")
    lo, hi = ie.i_eff.min() * 0.7, ie.i_eff.max() * 1.4
    ax.plot([lo, hi], [lo, hi], color="#8a8a85", ls=":", lw=1.5)
    if band:
        ax.axvspan(band[0], band[1], color="#e6e6e3", alpha=0.6, zorder=0)
        ax.axhline(300 * 12, color="#8a8a85", lw=1, ls="-.")
        ax.text(band[0] * 1.05, ymin * 1.6, "H2v2 band\nat T = 300", fontsize=10, color="#55554f")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(f"Exposure-corrected bits I_eff = 12 Σ(1 − e^(−n_k/{n0:.2f}))  (n0 frozen)")
    ax.set_ylabel("Bits of knowledge the model stored")
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("Solid: v1 corpora (n0 fitted on the Zipf ones). Dashed: held-out.", fontsize=12)
    fig.tight_layout()
    save(fig, "fig_ieff_v2")


def fig3_absorption(ab: pd.DataFrame):
    """Fraction of delivered facts stored vs exposure count, per corpus (last point, seed mean)."""
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bins = [b for b in ab["bin"].unique()]
    xpos = {b: i for i, b in enumerate(bins)}
    for c in ALL_CORPORA:
        g = ab[ab.corpus == c]
        if g.empty:
            continue
        gm = g.groupby("bin", sort=False).agg(frac=("frac_stored", "mean"), n=("n_facts", "mean")).reset_index()
        gm = gm[gm.n >= 50]  # bins with fewer than 50 facts are too noisy to show
        n = int(c.split("_n")[1])
        ax.plot([xpos[b] for b in gm["bin"]], gm.frac.clip(lower=0), color=COLORS[c], lw=2,
                marker=MARKERS.get(n, "^"), ms=6, label=LABELS[c], ls="-" if c in CORPORA else "--")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins)
    ax.set_xlabel("Exposures of the fact during training (n_k)")
    ax.set_ylabel("Fraction of delivered facts stored (top-1, chance-corrected)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_title("At each run's final measurement point; bins with < 50 facts omitted", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_absorption")


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
    # ---- v2: frozen n0, held-out corpora, absorption curve (PASS_CRITERIA_v2.md)
    nk = collapse.load_nk(df)
    v2 = collapse.h2v2(df, nk)
    v2["crossings"].to_csv(os.path.join(ROOT, "results", "h2v2_crossings.csv"), index=False)
    v2["spreads"].to_csv(os.path.join(ROOT, "results", "h2v2_spreads.csv"), index=False)
    fig_ieff_v2(v2["ieff"], v2["n0"], v2["verdict"].get("band"))
    ab = absorption.compute()
    if not ab.empty:
        fig3_absorption(ab)
    det = collapse.determinism_check(df)
    det.to_csv(os.path.join(ROOT, "results", "determinism_check.csv"), index=False)
    # exploratory only: refit n0 with the held-out corpora included (never used for the verdict)
    n0_explore, _, _, _ = collapse.fit_n0(df, nk, exclude=("EQ4_n16",))
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
        "n0_excl_eq4_diagnostic": h2["n0_excl_eq4"],
        "ieff_spreads_excl_eq4_diagnostic": h2["ieff_spreads_excl_eq4"].to_dict("records"),
        "token_shift": shift.to_dict("records"),
        "identity_ratio": last.to_dict("records"),
        "unseen_top1_acc_by_corpus": ctrl.to_dict("index"),
        "chance": 1 / 4096,
        "v2": {"n0_frozen": v2["n0"], "spreads": v2["spreads"].to_dict("records"),
               "crossings": v2["crossings"].to_dict("records"), "verdict": v2["verdict"],
               "n0_refit_incl_heldout_EXPLORATORY": n0_explore,
               "determinism_check": det.to_dict("records"),
               "absorption_last_point_seed_mean": (ab.groupby(["corpus", "bin"], sort=False)
                                                   .frac_stored.mean().unstack("bin").to_dict("index")
                                                   if not ab.empty else {})},
    }
    json.dump(summary, open(os.path.join(ROOT, "results", "summary.json"), "w"), indent=1,
              default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
