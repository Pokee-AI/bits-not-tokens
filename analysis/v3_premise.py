"""v3 Phase P: assemble results/v3_premise.csv and evaluate P1 / P2 (PASS_CRITERIA_v3.md)."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ROOT  # noqa: E402

SIZES = ["S", "M", "L"]
CORPORA = ["RAW", "CAPPED", "CURATED"]
EVALUABLE_MIN = 100
CAP_ADEQ_MIN = 0.7


def load() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ROOT, "results", "v3_phaseP", "*.csv")))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.sort_values(["model_size", "corpus", "seed", "train_tokens"]).reset_index(drop=True)
    df.to_csv(os.path.join(ROOT, "results", "v3_premise.csv"), index=False)
    return df


def final_points(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["model_size", "corpus", "seed"]).train_tokens.idxmax()]


def forgetting_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Exploratory diagnostic (not pre-registered): for each capped run, facts that first
    reached c* between consecutive measurement points, and the fraction of them stored at the
    final point; plus the fraction stored among facts still below c* at the end."""
    import glob as _glob
    rows = []
    for (m, c, s), g in df[df.corpus.isin(["CAPPED", "CURATED"])].groupby(["model_size", "corpus", "seed"]):
        d = os.path.join(ROOT, "artifacts", f"{c}_seed{s}_{m}")
        cstar = int(g.cap.iloc[0])
        pts = sorted(int(os.path.basename(f)[3:-4]) for f in _glob.glob(os.path.join(d, "nk_*.npz")))
        if not pts:
            continue
        nk_final = np.load(os.path.join(d, f"nk_{pts[-1]}.npz"))["n_k"]
        hit = np.unpackbits(np.load(os.path.join(d, f"hits_{pts[-1]}.npz"))["hit"])[: len(nk_final)].astype(bool)
        prev = np.zeros(len(nk_final), bool)
        tokens = dict(zip(g.docs, g.train_tokens))
        for i, p in enumerate(pts):
            nk = np.load(os.path.join(d, f"nk_{p}.npz"))["n_k"]
            capped = nk >= cstar
            new = capped & ~prev
            rows.append({"model_size": m, "corpus": c, "seed": s, "point": i, "docs": p,
                         "train_tokens": tokens.get(p, np.nan), "facts_newly_capped": int(new.sum()),
                         "retained_at_end": float(hit[new].mean()) if new.any() else np.nan})
            prev = capped
        below = (nk_final > 0) & (nk_final < cstar)
        rows.append({"model_size": m, "corpus": c, "seed": s, "point": "below_cstar_at_end", "docs": pts[-1],
                     "train_tokens": tokens.get(pts[-1], np.nan), "facts_newly_capped": int(below.sum()),
                     "retained_at_end": float(hit[below].mean()) if below.any() else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(ROOT, "results", "v3_forgetting_EXPLORATORY.csv"), index=False)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    fin = final_points(df)
    mean = fin.groupby(["model_size", "corpus"]).facts_stored.mean().unstack("corpus")
    per_seed = fin.pivot_table(index=["model_size", "corpus"], columns="seed", values="facts_stored")
    evaluable = {m: bool(m in mean.index and "RAW" in mean.columns and mean.loc[m, "RAW"] >= EVALUABLE_MIN)
                 for m in SIZES}
    ratios = {}
    for m in SIZES:
        if m in mean.index:
            r = mean.loc[m]
            ratios[m] = {"CURATED_over_RAW": r.get("CURATED", np.nan) / r.get("RAW", np.nan),
                         "CAPPED_over_RAW": r.get("CAPPED", np.nan) / r.get("RAW", np.nan),
                         "CURATED_over_CAPPED": r.get("CURATED", np.nan) / r.get("CAPPED", np.nan)}
    p1_sizes = {m: (ratios[m]["CURATED_over_RAW"] >= 3.0) for m in SIZES if evaluable.get(m) and m in ratios}
    p1 = bool(evaluable.get("L") and p1_sizes and all(p1_sizes.values()))
    p2 = bool(evaluable.get("L") and "S" in mean.index and "L" in mean.index
              and mean.loc["S", "CURATED"] >= mean.loc["L", "RAW"])
    cap_flags = fin[fin.corpus.isin(["CAPPED", "CURATED"])][["model_size", "corpus", "seed", "cap_adequacy"]].copy()
    cap_flags["cap_limited"] = cap_flags.cap_adequacy < CAP_ADEQ_MIN
    # never left the plateau = in-distribution loss above 11.5 bits at EVERY measurement point
    mn = df.groupby(["model_size", "corpus", "seed"]).obj_loss_bits_indist.min().reset_index()
    never_left = mn[mn.obj_loss_bits_indist > 11.5]
    pred = pd.read_csv(os.path.join(ROOT, "results", "v3_predicted_storable.csv"))
    pv = fin.merge(pred[["model_size", "corpus", "predicted_facts_reaching_cstar", "predicted_ratio_CURATED_over_RAW"]],
                   on=["model_size", "corpus"], how="left")
    headroom = fin.groupby(["model_size", "corpus"])[["bits_stored_per_param", "bits_stored_per_param_total"]].mean()
    forgetting = forgetting_curve(df)
    return {"final": fin, "mean_facts_stored": mean, "per_seed_facts_stored": per_seed, "evaluable": evaluable,
            "ratios": ratios, "P1_by_size": p1_sizes, "P1_PASS": p1, "P2_PASS": p2,
            "S_CURATED_vs_L_RAW": (float(mean.loc["S", "CURATED"]) if "S" in mean.index else None,
                                   float(mean.loc["L", "RAW"]) if "L" in mean.index else None),
            "cap_adequacy": cap_flags, "never_left_plateau": never_left, "predicted_vs_actual": pv,
            "headroom": headroom, "forgetting_curve_EXPLORATORY": forgetting}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = load()
    if df.empty:
        raise SystemExit("no Phase P results yet")
    r = evaluate(df)
    for k in ["mean_facts_stored", "per_seed_facts_stored", "cap_adequacy", "never_left_plateau", "headroom"]:
        print(k); print(r[k].round(4).to_string())
    print("evaluable", r["evaluable"]); print("ratios", r["ratios"])
    print("P1 by size", r["P1_by_size"], "P1", r["P1_PASS"], "P2", r["P2_PASS"], r["S_CURATED_vs_L_RAW"])
    print(r["predicted_vs_actual"][["model_size", "corpus", "seed", "facts_stored", "facts_with_at_least_cstar_exposures",
                                    "predicted_facts_reaching_cstar", "cap_adequacy"]].round(3).to_string(index=False))
