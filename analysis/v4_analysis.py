"""v4: assemble results/v4_premise.csv, absorption curves, level selection, P1/P2/P3."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.absorption import BINS, bin_label  # noqa: E402
from analysis.common import ROOT  # noqa: E402

SIZES = ["S", "M", "L"]
LEVELS = ["1000", "300", "100", "30", "x30"]
P3_TOL = 0.02
EVALUABLE_MIN = 100
CHANCE = 1 / 4096


def load() -> pd.DataFrame:
    """v4 runs plus the v3 RAW runs (reused; determinism verified)."""
    files = sorted(glob.glob(os.path.join(ROOT, "results", "v4_runs", "*.csv")))
    raw_v4 = [f for f in files if os.path.basename(f).startswith("RAW_")]
    # RAW: prefer the v4 rerun (adds weighted_acc_p / head_loss_bits); assert it equals the v3 rows
    raw = sorted(glob.glob(os.path.join(ROOT, "results", "baseline_raw", "RAW_seed*_*.csv")))
    for f in raw_v4:
        v3 = os.path.join(ROOT, "results", "baseline_raw", os.path.basename(f))
        if os.path.exists(v3):
            a, b = pd.read_csv(v3), pd.read_csv(f)
            for col in ["docs", "train_tokens", "facts_delivered", "facts_stored", "obj_loss_bits_indist", "unseen_top1_acc"]:
                assert np.array_equal(a[col].values, b[col].values), f"RAW rerun differs from v3 in {col}: {f}"
    have = {os.path.basename(f) for f in raw_v4}
    raw = [f for f in raw if os.path.basename(f) not in have]
    df = pd.concat([pd.read_csv(f) for f in files + raw], ignore_index=True)
    df.loc[df.corpus == "RAW", "corpus_type"] = "RAW"
    df.loc[df.corpus == "RAW", "level"] = "RAW"
    df["level"] = df["level"].astype(str)
    df = df.sort_values(["model_size", "corpus_type", "level", "seed", "train_tokens"]).reset_index(drop=True)
    df.to_csv(os.path.join(ROOT, "results", "v4_premise.csv"), index=False)
    return df


def final_points(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby(["model_size", "corpus_type", "level", "seed"]).train_tokens.idxmax()].copy()


def absorption(df: pd.DataFrame) -> pd.DataFrame:
    fin = final_points(df)
    rows = []
    for _, r in fin.iterrows():
        run = f"{r.corpus}_seed{int(r.seed)}_{r.model_size}" + ("" if r.corpus_type == "RAW" else f"_{r.level}")
        d = os.path.join(ROOT, "artifacts", run)
        f_nk, f_h = os.path.join(d, f"nk_{int(r.docs)}.npz"), os.path.join(d, f"hits_{int(r.docs)}.npz")
        if not (os.path.exists(f_nk) and os.path.exists(f_h)):
            continue
        nk = np.load(f_nk)["n_k"]
        hit = np.unpackbits(np.load(f_h)["hit"])[: len(nk)].astype(bool)
        for lo, hi in BINS:
            m = (nk >= lo) & (nk <= hi)
            rows.append({"model_size": r.model_size, "corpus_type": r.corpus_type, "level": r.level, "seed": int(r.seed),
                         "bin": bin_label(lo, hi), "n_facts": int(m.sum()),
                         "frac_stored": (hit[m].mean() - CHANCE) if m.any() else np.nan})
    csv = os.path.join(ROOT, "results", "v4_absorption.csv")
    if not rows:  # no artifacts/ (fresh clone): use the committed table
        return pd.read_csv(csv)
    out = pd.DataFrame(rows)
    out.to_csv(csv, index=False)
    return out


def evaluate(df: pd.DataFrame) -> dict:
    fin = final_points(df)
    key = ["model_size", "corpus_type", "level", "seed"]
    fs = fin.set_index(key).facts_stored
    wa = fin.set_index(key).weighted_acc_p
    pred = pd.read_csv(os.path.join(ROOT, "results", "v4_predicted.csv"))
    pred["level"] = pred["level"].astype(str)
    sel, rows = {}, []
    for m in SIZES:
        for ct in ["FLAT", "CURATED"]:
            eligible = []
            for lv in LEVELS:
                k1 = (m, ct, lv, 1)
                if k1 not in fs.index:
                    continue
                p3_s1 = wa[k1] >= wa[(m, "RAW", "RAW", 1)] - P3_TOL
                rows.append({"model_size": m, "corpus_type": ct, "level": lv, "facts_stored_s1": fs[k1],
                             "weighted_acc_p_s1": wa[k1], "raw_weighted_acc_p_s1": wa[(m, "RAW", "RAW", 1)],
                             "passes_P3_on_seed1": bool(p3_s1)})
                if p3_s1:
                    eligible.append((fs[k1], lv))
            sel[(m, ct)] = max(eligible)[1] if eligible else None
    selection = pd.DataFrame(rows)
    # verdicts on seed 2 at the selected level
    verdict_rows = []
    evaluable = {m: bool(fs.get((m, "RAW", "RAW", 2), 0) >= EVALUABLE_MIN) for m in SIZES}
    p1_by_size, p3_by_size = {}, {}
    for m in SIZES:
        lv = sel[(m, "CURATED")]
        raw2, raw_wa2 = fs[(m, "RAW", "RAW", 2)], wa[(m, "RAW", "RAW", 2)]
        if lv is None:
            verdict_rows.append({"model_size": m, "selected_CURATED_level": "NO ELIGIBLE LEVEL", "raw_s2": raw2})
            p1_by_size[m] = "FAIL (NO ELIGIBLE LEVEL)"
            continue
        k2 = (m, "CURATED", lv, 2)
        cur2, cur_wa2 = fs[k2], wa[k2]
        p3 = bool(cur_wa2 >= raw_wa2 - P3_TOL)
        ratio = cur2 / raw2
        p3_by_size[m] = p3
        p1_by_size[m] = ("PASS" if ratio >= 3.0 else "FAIL") if p3 else "FAIL (HEAD LOSS)"
        if not evaluable[m]:
            p1_by_size[m] = "NOT EVALUABLE"
        verdict_rows.append({"model_size": m, "selected_CURATED_level": lv, "selected_FLAT_level": sel[(m, "FLAT")],
                             "raw_s2": raw2, "curated_s2": cur2, "ratio_s2": ratio, "raw_weighted_acc_s2": raw_wa2,
                             "curated_weighted_acc_s2": cur_wa2, "P3_s2": p3, "P1": p1_by_size[m],
                             "evaluable": evaluable[m]})
    verdicts = pd.DataFrame(verdict_rows)
    p1 = bool(evaluable["L"] and all(v == "PASS" for m, v in p1_by_size.items() if evaluable[m]))
    lvS = sel[("S", "CURATED")]
    p2 = None
    if evaluable["L"] and lvS is not None:
        p2 = bool(fs[("S", "CURATED", lvS, 2)] >= fs[("L", "RAW", "RAW", 2)] and p3_by_size.get("S", False))
    # descriptive tables
    tab = fin.pivot_table(index=["model_size", "corpus_type", "level"], columns="seed",
                          values=["facts_stored", "weighted_acc_p", "head_loss_bits", "bits_stored_per_param"])
    attrib = []
    for m in SIZES:
        for lv in ["1000", "300", "100", "30"]:
            for s in [1, 2]:
                r, f, c = fs.get((m, "RAW", "RAW", s)), fs.get((m, "FLAT", lv, s)), fs.get((m, "CURATED", lv, s))
                attrib.append({"model_size": m, "level": lv, "seed": s, "FLAT_over_RAW": f / r if f is not None else np.nan,
                               "CURATED_over_FLAT": c / f if (c is not None and f) else np.nan,
                               "CURATED_over_RAW": c / r if c is not None else np.nan})
    pv = fin.merge(pred[["model_size", "corpus_type", "level", "predicted_facts_stored", "predicted_ratio_to_RAW",
                         "predicted_weighted_acc_p"]], on=["model_size", "corpus_type", "level"], how="left")
    pv["actual_over_predicted"] = pv.facts_stored / pv.predicted_facts_stored
    m_vs_l = {s: (fs.get(("M", "CURATED", sel[("M", "CURATED")], s)) if sel[("M", "CURATED")] else None,
                  fs.get(("L", "RAW", "RAW", s))) for s in [1, 2]}
    never_left = df.groupby(["model_size", "corpus_type", "level", "seed"]).obj_loss_bits_indist.min()
    never_left = never_left[never_left > 11.5]
    return {"final": fin, "selection": selection, "selected": sel, "verdicts": verdicts, "P1_PASS": p1, "P2_PASS": p2,
            "evaluable": evaluable, "table": tab, "attribution": pd.DataFrame(attrib), "pred_vs_actual": pv,
            "M_CURATED_vs_L_RAW": m_vs_l, "never_left_plateau": never_left.reset_index()}


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = load()
    ab = absorption(df)
    r = evaluate(df)
    print(r["selection"].round(4).to_string(index=False))
    print(r["verdicts"].round(4).to_string(index=False))
    print("P1", r["P1_PASS"], "P2", r["P2_PASS"], "evaluable", r["evaluable"], "M vs L", r["M_CURATED_vs_L_RAW"])
    print(r["table"].round(3).to_string())
    print(r["attribution"].round(3).to_string(index=False))
    print(r["pred_vs_actual"][["model_size", "corpus_type", "level", "seed", "facts_stored", "predicted_facts_stored",
                               "actual_over_predicted", "weighted_acc_p", "predicted_weighted_acc_p", "head_loss_bits"]]
          .round(3).to_string(index=False))
    print("never left plateau:", r["never_left_plateau"].to_dict("records"))
