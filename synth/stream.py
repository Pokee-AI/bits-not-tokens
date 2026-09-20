"""Document streams (Zipf / EQ) and the exact information counter n_k.

A stream is fully determined by (world, corpus config, run seed). Its state can be
snapshotted and restored so a cooldown branch consumes exactly the documents the
main run will consume next.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from synth.information import zipf_probs
from synth.world import World


@dataclass
class Batch:
    docs: np.ndarray  # int32 [B, L], PAD at the end of each row
    keys: np.ndarray  # int64 [B]
    obj_pos: np.ndarray  # int32 [B]
    n_tokens: int  # non-PAD tokens in this batch


class DocStream:
    """Base: draws keys, templates and filler; maintains n_k and a document counter."""

    def __init__(self, world: World, filler_n: int, run_seed: int):
        self.world = world
        self.filler_n = filler_n
        self.run_seed = run_seed
        self.rng = np.random.default_rng([run_seed, 0xB175])
        self.n_k = np.zeros(world.n_facts, dtype=np.int32)
        self.docs_seen = 0
        self.tokens_seen = 0

    # -- subclass hook -------------------------------------------------------
    def _draw_keys(self, n: int) -> np.ndarray:
        raise NotImplementedError

    # -- public --------------------------------------------------------------
    def next_batch(self, n: int) -> Batch:
        keys = self._draw_keys(n)
        templ = self.rng.integers(0, self.world.n_templates, size=n)
        docs, obj_pos, n_tok = self.world.build_docs(keys, templ, self.filler_n, self.rng)
        # Exact information counter: n_k counts consumptions of each fact.
        np.add.at(self.n_k, keys, 1)
        self.docs_seen += n
        self.tokens_seen += n_tok
        return Batch(docs=docs, keys=keys, obj_pos=obj_pos, n_tokens=n_tok)

    @property
    def facts_delivered(self) -> int:
        return int(np.count_nonzero(self.n_k))

    def snapshot(self) -> dict:
        return {
            "rng": copy.deepcopy(self.rng.bit_generator.state),
            "n_k": self.n_k.copy(),
            "docs_seen": self.docs_seen,
            "tokens_seen": self.tokens_seen,
        }

    def restore(self, snap: dict) -> None:
        self.rng.bit_generator.state = copy.deepcopy(snap["rng"])
        self.n_k = snap["n_k"].copy()
        self.docs_seen = snap["docs_seen"]
        self.tokens_seen = snap["tokens_seen"]


class ZipfStream(DocStream):
    """Facts sampled i.i.d. with p_k proportional to k^-(1+a) over popularity ranks."""

    def __init__(self, world: World, a: float, filler_n: int, run_seed: int):
        super().__init__(world, filler_n, run_seed)
        self.a = a
        self.p = zipf_probs(world.n_facts, a)
        self.cdf = np.cumsum(self.p)
        self.cdf[-1] = 1.0

    def _draw_keys(self, n: int) -> np.ndarray:
        u = self.rng.random(n)
        ranks = np.searchsorted(self.cdf, u, side="right")
        ranks = np.minimum(ranks, self.world.n_facts - 1)
        return self.world.rank_to_key[ranks].astype(np.int64)


class EqStream(DocStream):
    """Every fact exactly `repeats` times, one global shuffle drawn from the run seed."""

    def __init__(self, world: World, repeats: int, filler_n: int, run_seed: int):
        super().__init__(world, filler_n, run_seed)
        self.repeats = repeats
        order_rng = np.random.default_rng([run_seed, 0xE0])
        self.order = order_rng.permutation(np.repeat(np.arange(world.n_facts, dtype=np.int64), repeats))
        self.max_docs = len(self.order)

    def _draw_keys(self, n: int) -> np.ndarray:
        if self.docs_seen + n > self.max_docs:
            raise StopIteration("EQ corpus exhausted")
        return self.order[self.docs_seen : self.docs_seen + n]

    # docs_seen is the position in the fixed order, so the base snapshot suffices.


def make_stream(world: World, corpus: dict, run_seed: int) -> DocStream:
    if corpus["sampling"] == "zipf":
        return ZipfStream(world, float(corpus["zipf_a"]), int(corpus["filler_n"]), run_seed)
    if corpus["sampling"] == "eq":
        return EqStream(world, int(corpus["repeats"]), int(corpus["filler_n"]), run_seed)
    raise ValueError(corpus["sampling"])
