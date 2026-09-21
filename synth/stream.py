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


# ----------------------------------------------------------------------------------------
# v3 streams (Experiment Brief v3)
class ProbStream(DocStream):
    """Facts sampled i.i.d. from an arbitrary probability vector over popularity ranks."""

    def __init__(self, world: World, p: np.ndarray, filler_n: int, run_seed: int):
        super().__init__(world, filler_n, run_seed)
        self.p = np.asarray(p, dtype=np.float64)
        self.cdf = np.cumsum(self.p)
        self.cdf[-1] = 1.0

    def _draw_ranks(self, n: int) -> np.ndarray:
        u = self.rng.random(n)
        return np.minimum(np.searchsorted(self.cdf, u, side="right"), self.world.n_facts - 1)

    def _draw_keys(self, n: int) -> np.ndarray:
        return self.world.rank_to_key[self._draw_ranks(n)].astype(np.int64)


class UniformSubsetStream(DocStream):
    """Uniform (with replacement) over the first k_sub facts of the popularity permutation."""

    def __init__(self, world: World, k_sub: int, filler_n: int, run_seed: int):
        super().__init__(world, filler_n, run_seed)
        self.k_sub = int(k_sub)
        self.subset_keys = world.rank_to_key[: self.k_sub].astype(np.int64)

    def _draw_keys(self, n: int) -> np.ndarray:
        return self.subset_keys[self.rng.integers(0, self.k_sub, size=n)]


class CappedStream(ProbStream):
    """Shifted-Zipf stream with a per-fact repetition cap (v3 CAPPED / CURATED).

    The first `warmup_docs` documents are drawn uncapped. Afterwards the sampling CDF is
    rebuilt over facts with n_k < cap every `block` documents, and within a block any draw
    that would exceed the cap is rejected and redrawn. Raises StopIteration when no fact
    is left below the cap.
    """

    def __init__(self, world: World, p: np.ndarray, cap: int, warmup_docs: int, filler_n: int,
                 run_seed: int, block: int = 50_000):
        super().__init__(world, p, filler_n, run_seed)
        self.cap, self.warmup_docs, self.block = int(cap), int(warmup_docs), int(block)
        self.base_cdf = self.cdf.copy()
        self.block_id = -1  # id of the block the current cdf was built for; -1 = warm-up cdf
        self.n_uncapped = world.n_facts

    def _current_block(self) -> int:
        return -1 if self.docs_seen < self.warmup_docs else (self.docs_seen - self.warmup_docs) // self.block

    def _rebuild(self, block_id: int) -> None:
        if block_id < 0:
            self.cdf = self.base_cdf
        else:
            # n_k is indexed by key; p by rank -> mask ranks whose key is capped
            open_ = self.n_k[self.world.rank_to_key] < self.cap
            self.n_uncapped = int(open_.sum())
            if self.n_uncapped == 0:
                raise StopIteration("every fact reached the cap")
            p = np.where(open_, self.p, 0.0)
            self.cdf = np.cumsum(p / p.sum())
            self.cdf[-1] = 1.0
        self.block_id = block_id

    def _draw_keys(self, n: int) -> np.ndarray:
        b = self._current_block()
        if b != self.block_id:
            self._rebuild(b)
        if b < 0:
            return super()._draw_keys(n)
        out = np.empty(n, dtype=np.int64)
        got = 0
        pending: dict[int, int] = {}  # key -> draws accepted in this batch (not yet in n_k)
        while got < n:
            cand = super()._draw_keys(n - got)
            for k in cand:
                k = int(k)
                if self.n_k[k] + pending.get(k, 0) < self.cap:
                    out[got] = k
                    got += 1
                    pending[k] = pending.get(k, 0) + 1
        return out

    def snapshot(self) -> dict:
        s = super().snapshot()
        s["cdf"] = self.cdf.copy()
        s["block_id"] = self.block_id
        s["n_uncapped"] = self.n_uncapped
        return s

    def restore(self, snap: dict) -> None:
        super().restore(snap)
        self.cdf = snap["cdf"].copy()
        self.block_id = snap["block_id"]
        self.n_uncapped = snap["n_uncapped"]


def make_stream_v3(world: World, corpus: dict, run_seed: int, cap: int | None = None,
                   warmup_docs: int | None = None) -> DocStream:
    from synth.information import shifted_zipf_probs
    kind = corpus["sampling"]
    if kind == "uniform_subset":
        return UniformSubsetStream(world, int(corpus["k_sub"]), int(corpus["filler_n"]), run_seed)
    p = shifted_zipf_probs(world.n_facts, float(corpus["zipf_a"]), float(corpus["zipf_q"]))
    if kind == "shifted_zipf":
        return ProbStream(world, p, int(corpus["filler_n"]), run_seed)
    if kind == "capped":
        assert cap is not None and warmup_docs is not None, "capped stream needs c* and W"
        return CappedStream(world, p, cap, warmup_docs, int(corpus["filler_n"]), run_seed,
                            int(corpus.get("block", 50_000)))
    return make_stream(world, corpus, run_seed)
