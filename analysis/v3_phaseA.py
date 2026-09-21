"""v3 Phase A: plateau length, c50/c90 on dense uniform data, c* and W per model size."""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ROOT  # noqa: E402
from analysis.v3_phase0 import c_at  # noqa: E402

PLATEAU_BITS = 11.5
CSTAR_CAP = 400
W_CAP = 1_000_000


def load_phase_a() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ROOT, "results", "v3_phaseA", "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df.to_csv(os.path.join(ROOT, "results", "v3_phaseA.csv"), index=False)
    return df


def plateau(run_dir: str):
    """First probe step with in-distribution loss < 11.5 bits -> (step, docs); None if never."""
    p = pd.read_csv(os.path.join(run_dir, "probe.csv"))
    below = p[p.obj_loss_bits_indist < PLATEAU_BITS]
    if below.empty:
        return None, None
    return int(below.step.iloc[0]), int(below.docs.iloc[0])


def per_run(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (c, s, m), g in df.groupby(["corpus", "seed", "model_size"]):
        g = g.sort_values("docs")
        k = int(g.k_sub.iloc[0])
        d = os.path.join(ROOT, "artifacts", f"{c}_seed{s}_{m}")
        pstep, pdocs = plateau(d)
        x = g.docs.values / k
        y = g.frac_stored.values
        c50, c90 = c_at(0.5, x, y), c_at(0.9, x, y)
        if pdocs is not None:
            xa = np.maximum(g.docs.values - pdocs, 1) / k  # exposures after the plateau ended
            c50a, c90a = c_at(0.5, xa, y), c_at(0.9, xa, y)
        else:
            c50a = c90a = None
        last = g.iloc[-1]
        rows.append({"model_size": m, "k_sub": k, "seed": s, "plateau_steps": pstep, "plateau_docs": pdocs,
                     "c50": c50, "c90": c90, "c50_after_plateau": c50a, "c90_after_plateau": c90a,
                     "frac_stored_final": last.frac_stored, "mean_exposures_final": last.mean_exposures,
                     "bits_per_param_final": last.bits_stored_per_param,
                     "bits_per_param_total_final": last.bits_stored_per_param_total,
                     "n_params_nonemb": last.n_params_nonemb})
    out = pd.DataFrame(rows).sort_values(["model_size", "k_sub", "seed"])
    out.to_csv(os.path.join(ROOT, "results", "v3_phaseA_summary.csv"), index=False)
    return out


def c_star_and_w(summary: pd.DataFrame, z05_plateau_docs: dict | None = None) -> pd.DataFrame:
    """c* = c90 (from start) on K_sub = 30,000 seed mean, ceil to 10, cap 400; fallback K_sub = 10,000.
    W = 2 x longest plateau (docs) over the K_sub = 30,000 runs, ceil to 50,000, cap 1,000,000."""
    rows = []
    for m, g in summary.groupby("model_size"):
        src, cval, note = None, None, ""
        for k in (30_000, 10_000):
            gk = g[g.k_sub == k]
            if not gk.empty and gk.c90.notna().all():
                src, cval = k, float(gk.c90.mean())
                break
        if cval is None:
            cstar, note = CSTAR_CAP, "90 % not reached at K_sub = 30,000 or 10,000; c* set to the cap"
        else:
            cstar = min(CSTAR_CAP, int(math.ceil(cval / 10.0)) * 10)
            if cstar == CSTAR_CAP:
                note = f"c90 = {cval:.1f} exceeds the cap; c* capped at {CSTAR_CAP}"
        g30 = g[g.k_sub == 30_000]
        longest = g30.plateau_docs.max() if g30.plateau_docs.notna().any() else None
        left = bool(g.plateau_docs.notna().any())
        if longest is None:
            W, wnote = None, "no K_sub = 30,000 run left the plateau"
        else:
            W = int(math.ceil(2 * longest / 50_000)) * 50_000
            wnote = ""
            if W > W_CAP:
                wnote = f"2 x longest plateau = {W:,} exceeds the cap; W capped at {W_CAP:,}"
                W = W_CAP
        rows.append({"model_size": m, "c90_source_k_sub": src, "c90_seed_mean": cval, "c_star": cstar,
                     "longest_plateau_docs_k30": longest, "W": W, "any_run_left_plateau": left,
                     "note": "; ".join(n for n in (note, wnote) if n)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = load_phase_a()
    s = per_run(df)
    print(s.round(3).to_string(index=False))
    cw = c_star_and_w(s)
    cw.to_csv(os.path.join(ROOT, "results", "v3_cstar.csv"), index=False)
    print(cw.to_string(index=False))
