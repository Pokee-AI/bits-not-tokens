"""v5: assemble results/v5_runs.csv and results/v5_probe.csv; evaluate R1-R5 (CRITERIA_v5.md)."""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ROOT  # noqa: E402

B = 300_000_000
SIZES = ["S", "M", "L"]
SETTINGS = ["base", "lr3", "wd0", "wd0_lr3", "cosine"]
SEL = {"S": "300", "M": "100", "L": "30"}
PROBE = yaml.safe_load(open(os.path.join(ROOT, "configs", "v5.yaml")))


def run_name(r) -> str:
    tag = f"{r.model_size}_{r.setting}"
    if r.budget_tokens > B:
        tag += f"_{int(round(r.budget_tokens / B))}B"
    if r.corpus_type != "RAW" and str(r.level) not in (SEL[r.model_size],):
        tag += f"_{r.level}"
    return f"{r.corpus}_seed{int(r.seed)}_{tag}"


def load() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ROOT, "results", "v5_runs", "*.csv")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["level"] = df["level"].astype(str)
    df["budget_mult"] = (df.budget_tokens / B).round().astype(int)
    df["run"] = [run_name(r) for r in df.itertuples()]
    df = df.sort_values(["model_size", "setting", "budget_mult", "corpus_type", "level", "seed", "train_tokens"]).reset_index(drop=True)
    df.to_csv(os.path.join(ROOT, "results", "v5_runs.csv"), index=False)
    return df


def final_points(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.groupby("run").train_tokens.idxmax()].copy()


def probe_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for run, g in df.groupby("run"):
        f = os.path.join(ROOT, "artifacts", run, "probe_retention.csv")
        if not os.path.exists(f):
            continue
        p = pd.read_csv(f)
        r0 = g.iloc[0]
        p["run"], p["model_size"], p["setting"], p["corpus_type"], p["level"], p["seed"], p["budget_mult"] = \
            run, r0.model_size, r0.setting, r0.corpus_type, r0.level, int(r0.seed), int(r0.budget_mult)
        p["docs_since_window"] = p.docs - PROBE["probe_window_hi"]
        rows.append(p)
    csv = os.path.join(ROOT, "results", "v5_probe.csv")
    if not rows:  # no artifacts/ (fresh clone): use the committed table
        return pd.read_csv(csv)
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(csv, index=False)
    return out


def half_life(p: pd.DataFrame):
    """Documents since window end until probe accuracy falls to half its window-end value
    (main-run weights only; linear interpolation). None if not reached."""
    m = p[p.weights.isin(["main", "main_at_snapshot"]) & (p.docs >= PROBE["probe_window_hi"])].sort_values("docs")
    if m.empty:
        return None, None
    a0 = float(m.probe_acc.iloc[0])
    x, y = m.docs_since_window.values.astype(float), m.probe_acc.values
    for i in range(1, len(y)):
        if y[i] <= a0 / 2:
            t = x[i - 1] + (y[i - 1] - a0 / 2) * (x[i] - x[i - 1]) / (y[i - 1] - y[i]) if y[i - 1] != y[i] else x[i]
            return a0, float(t)
    return a0, None


def evaluate(df: pd.DataFrame, probe: pd.DataFrame) -> dict:
    fin = final_points(df)
    key = ["model_size", "setting", "budget_mult", "corpus_type", "level", "seed"]
    fs = fin.set_index(key)
    out = {}
    # ---- R1
    r1 = []
    for m in SIZES:
        for st in SETTINGS:
            raw = fin[(fin.model_size == m) & (fin.setting == st) & (fin.budget_mult == 1) & (fin.corpus_type == "RAW") & fin.seed.isin([1, 2])]
            fl = fin[(fin.model_size == m) & (fin.setting == st) & (fin.budget_mult == 1) & (fin.corpus_type == "FLAT") & (fin.level == SEL[m]) & fin.seed.isin([1, 2])]
            if raw.empty or fl.empty:
                continue
            ratio = fl.facts_stored.mean() / raw.facts_stored.mean()
            dw = fl.weighted_acc_p.mean() - raw.weighted_acc_p.mean()
            r1.append({"model_size": m, "setting": st, "raw_mean": raw.facts_stored.mean(), "flat_mean": fl.facts_stored.mean(),
                       "ratio": ratio, "raw_s1": raw[raw.seed == 1].facts_stored.iloc[0] if (raw.seed == 1).any() else np.nan,
                       "raw_s2": raw[raw.seed == 2].facts_stored.iloc[0] if (raw.seed == 2).any() else np.nan,
                       "flat_s1": fl[fl.seed == 1].facts_stored.iloc[0] if (fl.seed == 1).any() else np.nan,
                       "flat_s2": fl[fl.seed == 2].facts_stored.iloc[0] if (fl.seed == 2).any() else np.nan,
                       "weighted_acc_raw": raw.weighted_acc_p.mean(), "weighted_acc_flat": fl.weighted_acc_p.mean(),
                       "passes": bool(ratio >= 1.5 and dw >= -0.02)})
    r1 = pd.DataFrame(r1)
    out["R1"] = r1
    out["R1_ROBUST_M"] = bool(len(r1[r1.model_size == "M"]) == 5 and r1[r1.model_size == "M"].passes.all())
    # ---- R2
    fb = fin[(fin.model_size == "M") & (fin.setting == "base") & (fin.corpus_type == "FLAT") & (fin.level == "100") & fin.seed.isin([1, 2])]
    r2 = fb.groupby("budget_mult").facts_stored.mean()
    rb = fin[(fin.model_size == "M") & (fin.setting == "base") & (fin.corpus_type == "RAW") & fin.seed.isin([1, 2])].groupby("budget_mult").facts_stored.mean()
    if 1 in r2.index and 4 in r2.index:
        growth = r2[4] / r2[1]
        out["R2"] = {"flat_B": r2[1], "flat_2B": r2.get(2, np.nan), "flat_4B": r2[4], "ratio_4B_over_B": growth,
                     "within_15pct": bool(abs(growth - 1) <= 0.15), "growth_exponent_tokens": float(np.log(growth) / np.log(4)),
                     "raw_B": rb.get(1, np.nan), "raw_2B": rb.get(2, np.nan), "raw_4B": rb.get(4, np.nan)}
    # ---- R3
    r3 = []
    for run, p in probe.groupby("run"):
        g = df[df.run == run].iloc[0]
        a0, hl = half_life(p)
        wd_ts = np.inf if g.weight_decay == 0 else 128.0 / (g.lr * g.weight_decay)
        r3.append({"run": run, "model_size": g.model_size, "setting": g.setting, "corpus_type": g.corpus_type,
                   "level": g.level, "seed": int(g.seed), "budget_mult": int(g.budget_mult), "acc_window_end": a0,
                   "half_life_docs": hl, "wd_timescale_docs": wd_ts, "lr": g.lr, "weight_decay": g.weight_decay})
    out["R3"] = pd.DataFrame(r3)
    # ---- R4
    r4 = []
    for m in SIZES:
        base = fin[(fin.model_size == m) & (fin.setting == "base") & (fin.budget_mult == 1)]
        raw = base[base.corpus_type == "RAW"].set_index("seed").facts_stored
        for ct in ["FLAT", "CURATED"]:
            x = base[(base.corpus_type == ct) & (base.level == SEL[m])].set_index("seed").facts_stored
            ratios = {s: x[s] / raw[s] for s in x.index if s in raw.index}
            r25 = np.array([ratios[s] for s in sorted(ratios) if s in (2, 3, 4, 5)])
            if len(r25) >= 2:
                ci = stats.t.interval(0.95, len(r25) - 1, loc=r25.mean(), scale=stats.sem(r25))
            else:
                ci = (np.nan, np.nan)
            r4.append({"model_size": m, "corpus_type": ct, "level": SEL[m], "seed1_ratio": ratios.get(1, np.nan),
                       "n_seeds_2to5": len(r25), "mean_2to5": r25.mean() if len(r25) else np.nan,
                       "ci95_lo": ci[0], "ci95_hi": ci[1], "per_seed": {int(k): float(v) for k, v in ratios.items()}})
    out["R4"] = pd.DataFrame(r4)
    # ---- R5: L levels (v4 levels without probe + v5 C levels with probe); base setting
    v4 = pd.read_csv(os.path.join(ROOT, "results", "v4_premise.csv"))
    v4["level"] = v4.level.astype(str)
    v4f = v4.loc[v4.groupby(["model_size", "corpus_type", "level", "seed"]).train_tokens.idxmax()]
    v4L = v4f[v4f.model_size == "L"][["corpus_type", "level", "seed", "facts_stored", "weighted_acc_p", "head_loss_bits"]].assign(source="v4 (no probe)")
    v5L = fin[(fin.model_size == "L") & (fin.setting == "base") & (fin.budget_mult == 1) & fin.seed.isin([1, 2])][
        ["corpus_type", "level", "seed", "facts_stored", "weighted_acc_p", "head_loss_bits"]].assign(source="v5 (probe)")
    r5 = pd.concat([v4L, v5L], ignore_index=True)
    raw_w = {(s, src): r5[(r5.corpus_type == "RAW") & (r5.seed == s) & (r5.source == src)].weighted_acc_p.mean()
             for s in (1, 2) for src in r5.source.unique()}
    r5["P3_eligible"] = [bool(r.weighted_acc_p >= raw_w.get((r.seed, r.source), np.nan) - 0.02) if r.corpus_type != "RAW" else True
                         for r in r5.itertuples()]
    out["R5"] = r5.sort_values(["corpus_type", "source", "seed", "level"])
    # ---- probe perturbation check: v5 base vs v4 (same cells, same seeds), RAW and FLAT-sel
    pert = []
    for m in SIZES:
        for ct in ["RAW", "FLAT", "CURATED"]:
            lv = "RAW" if ct == "RAW" else SEL[m]
            for s in (1, 2):
                a = v4f[(v4f.model_size == m) & (v4f.corpus_type == ct) & (v4f.level == lv) & (v4f.seed == s)]
                b = fin[(fin.model_size == m) & (fin.setting == "base") & (fin.budget_mult == 1) & (fin.corpus_type == ct) & (fin.level == lv) & (fin.seed == s)]
                if not a.empty and not b.empty:
                    pert.append({"model_size": m, "corpus_type": ct, "seed": s, "v4_facts_stored": float(a.facts_stored.iloc[0]),
                                 "v5_facts_stored": float(b.facts_stored.iloc[0]),
                                 "rel_change": float(b.facts_stored.iloc[0] / a.facts_stored.iloc[0] - 1)})
    out["probe_perturbation"] = pd.DataFrame(pert)
    # ---- cosine vs base: absorption by when exposures occurred (early vs last third), n_k in 17-64
    timing = []
    for run, g in df.groupby("run"):
        r = g.iloc[0]
        if r.setting not in ("base", "cosine") or r.budget_mult != 1:
            continue
        d = os.path.join(ROOT, "artifacts", run)
        pts = sorted(int(os.path.basename(f)[3:-4]) for f in glob.glob(os.path.join(d, "nk_*.npz")))
        if not pts or not os.path.exists(os.path.join(d, f"seen_{pts[-1]}.npz")):
            continue
        D = pts[-1]
        nk = np.load(os.path.join(d, f"nk_{D}.npz"))["n_k"]
        hit = np.unpackbits(np.load(os.path.join(d, f"hits_{D}.npz"))["hit"])[: len(nk)].astype(bool)
        sn = np.load(os.path.join(d, f"seen_{D}.npz"))
        pk = np.load(os.path.join(d, "probe_keys.npy"))
        mid = (sn["first_seen"].astype(np.int64) + sn["last_seen"].astype(np.int64)) / 2
        ok = (nk >= 17) & (nk <= 64)
        ok[pk] = False
        for name, lo, hi in [("early_third", 0, D / 3), ("last_third", 2 * D / 3, D)]:
            m = ok & (mid >= lo) & (mid < hi)
            timing.append({"run": run, "model_size": r.model_size, "setting": r.setting, "corpus_type": r.corpus_type,
                           "seed": int(r.seed), "window": name, "n_facts": int(m.sum()),
                           "frac_stored": float(hit[m].mean() - 1 / 4096) if m.any() else np.nan})
    tcsv = os.path.join(ROOT, "results", "v5_timing.csv")
    if timing:
        out["timing"] = pd.DataFrame(timing)
        out["timing"].to_csv(tcsv, index=False)
    else:  # no artifacts/ (fresh clone): use the committed table
        out["timing"] = pd.read_csv(tcsv)
    out["never_left_plateau"] = df.groupby("run").obj_loss_bits_indist.min().pipe(lambda s: s[s > 11.5]).reset_index()
    out["final"] = fin
    return out


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    df = load()
    probe = probe_table(df)
    r = evaluate(df, probe)
    for k in ["R1", "R3", "R4", "R5", "probe_perturbation", "timing", "never_left_plateau"]:
        print(f"== {k}")
        print(r[k].round(4).to_string(index=False) if isinstance(r[k], pd.DataFrame) else r[k])
    print("R1 ROBUST (M):", r["R1_ROBUST_M"]); print("R2:", r.get("R2"))
