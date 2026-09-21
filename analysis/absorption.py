"""Absorption curve: fraction of delivered facts stored vs exposure count n_k, per corpus.

Uses artifacts/<run>/hits_<docs>.npz (per-fact top-1 hits) and nk_<docs>.npz. For the v1
corpora the hits come from the `_rerun` runs (same seeds); see PASS_CRITERIA_v2.md.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.common import ALL_CORPORA, ROOT  # noqa: E402

BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 10**9)]
CHANCE = 1 / 4096


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">={lo}")


def run_dirs(corpus: str):
    """{seed: artifact dir with hits}; prefers <corpus>_seed<s>_rerun, else <corpus>_seed<s>."""
    out = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "artifacts", f"{corpus}_seed*"))):
        base = os.path.basename(d)
        tag = base[len(corpus) + 5:]
        if "_" in tag and not tag.endswith("_rerun"):
            continue
        seed = int(tag.split("_")[0])
        if glob.glob(os.path.join(d, "hits_*.npz")):
            if seed not in out or base.endswith("_rerun"):
                out[seed] = d
    return out


def absorption_at(d: str, docs: int) -> list[dict]:
    nk = np.load(os.path.join(d, f"nk_{docs}.npz"))["n_k"]
    hit = np.unpackbits(np.load(os.path.join(d, f"hits_{docs}.npz"))["hit"])[: len(nk)].astype(bool)
    rows = []
    for lo, hi in BINS:
        m = (nk >= lo) & (nk <= hi)
        n = int(m.sum())
        rows.append({"bin": bin_label(lo, hi), "n_lo": lo, "n_facts": n,
                     "frac_stored": (hit[m].mean() - CHANCE) if n else np.nan})
    return rows


def compute(points: str = "last") -> pd.DataFrame:
    rows = []
    for c in ALL_CORPORA:
        for seed, d in run_dirs(c).items():
            docs_all = sorted(int(os.path.basename(f)[5:-4]) for f in glob.glob(os.path.join(d, "hits_*.npz")))
            sel = [docs_all[-1]] if points == "last" else docs_all
            for docs in sel:
                for r in absorption_at(d, docs):
                    rows.append({"corpus": c, "seed": seed, "docs": docs, **r})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(ROOT, "results", "absorption.csv"), index=False)
    return df


if __name__ == "__main__":
    df = compute()
    pd.set_option("display.width", 200)
    piv = df.groupby(["corpus", "bin"], sort=False).frac_stored.mean().unstack("bin")
    print(piv.round(3).to_string())
