"""Exposure-count bins for absorption curves (P(stored) versus n_k)."""
from __future__ import annotations

BINS = [(1, 1), (2, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64), (65, 128), (129, 256), (257, 10**9)]
CHANCE = 1 / 4096


def bin_label(lo, hi):
    return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">={lo}")
