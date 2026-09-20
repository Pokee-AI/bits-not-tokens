"""H2: spread of the x at which each corpus reaches T facts stored, on tokens vs bits;
plus the exposure-corrected axis I_eff with a single fitted n0 (section 8)."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import CORPORA, ROOT, load_runs, loglog_first_crossing, seed_mean  # noqa: E402
from synth.information import i_eff  # noqa: E402

TARGETS = (1000, 2000, 3000)
AXES = {"tokens": "train_tokens", "bits": "bits_delivered"}


def crossings(df: pd.DataFrame) -> pd.DataFrame:
    """Seed-averaged curves; x at first crossing of T for each corpus and axis."""
    m = seed_mean(df, ["facts_stored", "train_tokens", "bits_delivered"])
    rows = []
    for T in TARGETS:
        for c in CORPORA:
            g = m[m.corpus == c].sort_values("docs")
            if g.empty:
                continue
            r = {"T": T, "corpus": c}
            for ax, col in AXES.items():
                r[f"x_{ax}"] = loglog_first_crossing(g[col].values, g.facts_stored.values, T)
            r["reached"] = r["x_tokens"] is not None
            rows.append(r)
    return pd.DataFrame(rows)


def spreads(cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for T, g in cross.groupby("T"):
        r = g[g.reached]
        row = {"T": T, "n_corpora": len(r), "excluded": ",".join(g[~g.reached].corpus) or "-"}
        for ax in AXES:
            v = r[f"x_{ax}"].astype(float)
            row[f"spread_{ax}"] = v.max() / v.min() if len(v) >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def load_nk(df: pd.DataFrame) -> dict:
    """{(corpus, seed, docs): n_k} from artifacts/<run>/nk_<docs>.npz."""
    out = {}
    for (c, s), g in df.groupby(["corpus", "seed"]):
        for d in g.docs:
            f = os.path.join(ROOT, "artifacts", f"{c}_seed{s}", f"nk_{d}.npz")
            if os.path.exists(f):
                out[(c, s, int(d))] = np.load(f)["n_k"]
    return out


def fit_n0(df: pd.DataFrame, nk: dict):
    """Single n0 across all corpora and seeds: minimise the residual of a straight-line
    fit of log(bits_stored) on log(I_eff). Points with bits_stored <= 0 are excluded."""
    pts = [(k, v) for k, v in nk.items()]
    d = df.set_index(["corpus", "seed", "docs"])
    y = np.array([d.loc[k, "bits_stored"] for k, _ in pts], float)
    keep = y > 0
    logy = np.log(y[keep])
    nks = [v for (_, v), k in zip(pts, keep) if k]

    def rss(log_n0):
        x = np.log([i_eff(v, np.exp(log_n0)) for v in nks])
        a, b = np.polyfit(x, logy, 1)
        return float(np.sum((logy - (a * x + b)) ** 2))

    res = minimize_scalar(rss, bounds=(np.log(0.01), np.log(1000.0)), method="bounded")
    n0 = float(np.exp(res.x))
    rows = [{"corpus": k[0], "seed": k[1], "docs": k[2], "i_eff": i_eff(v, n0),
             "bits_stored": d.loc[k, "bits_stored"], "facts_stored": d.loc[k, "facts_stored"]}
            for k, v in pts]
    return n0, pd.DataFrame(rows), int(keep.sum()), int((~keep).sum())


def ieff_spreads(ie: pd.DataFrame) -> pd.DataFrame:
    m = ie.groupby(["corpus", "docs"], as_index=False)[["i_eff", "facts_stored"]].mean()
    rows = []
    for T in TARGETS:
        xs = {}
        for c in CORPORA:
            g = m[m.corpus == c].sort_values("docs")
            if g.empty:
                continue
            x = loglog_first_crossing(g.i_eff.values, g.facts_stored.values, T)
            if x is not None:
                xs[c] = x
        v = np.array(list(xs.values()))
        rows.append({"T": T, "spread_ieff": v.max() / v.min() if len(v) >= 2 else np.nan,
                     "n_corpora": len(v)})
    return pd.DataFrame(rows)


def run(df: pd.DataFrame):
    cross = crossings(df)
    sp = spreads(cross)
    cross.to_csv(os.path.join(ROOT, "results", "h2_crossings.csv"), index=False)
    sp.to_csv(os.path.join(ROOT, "results", "h2_spreads.csv"), index=False)
    nk = load_nk(df)
    n0, ie, n_used, n_dropped = fit_n0(df, nk)
    ie.to_csv(os.path.join(ROOT, "results", "ieff.csv"), index=False)
    isp = ieff_spreads(ie)
    isp.to_csv(os.path.join(ROOT, "results", "ieff_spreads.csv"), index=False)
    row3 = sp[sp["T"] == 3000].iloc[0] if (sp["T"] == 3000).any() else None
    h2 = bool(row3 is not None and row3.spread_tokens >= 10 and row3.spread_bits <= 2)
    return {"crossings": cross, "spreads": sp, "n0": n0, "ieff": ie, "ieff_spreads": isp,
            "n0_points_used": n_used, "n0_points_dropped": n_dropped, "H2_PASS": h2}


if __name__ == "__main__":
    out = run(load_runs())
    print(out["crossings"].to_string(index=False))
    print(out["spreads"].to_string(index=False))
    print("n0", out["n0"], "points", out["n0_points_used"], "dropped", out["n0_points_dropped"])
    print(out["ieff_spreads"].to_string(index=False))
    print("H2_PASS", out["H2_PASS"])
