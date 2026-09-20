"""The synthetic world theta: facts, popularity permutation, templates, vocabulary.

Everything here is a deterministic function of the world seed (0 for all real runs).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OBJ_BITS = 12  # log2(4096)


@dataclass(frozen=True)
class Vocab:
    """Disjoint token ranges in the order fixed by the brief."""

    n_templ_tok: int = 64
    n_subj_tok: int = 256
    n_rel_tok: int = 20
    n_obj_tok: int = 4096
    n_fill_tok: int = 1024

    PAD: int = 0
    BOS: int = 1
    EOS: int = 2

    @property
    def templ0(self) -> int:
        return 3

    @property
    def subj0(self) -> int:
        return self.templ0 + self.n_templ_tok

    @property
    def rel0(self) -> int:
        return self.subj0 + self.n_subj_tok

    @property
    def obj0(self) -> int:
        return self.rel0 + self.n_rel_tok

    @property
    def fill0(self) -> int:
        return self.obj0 + self.n_obj_tok

    @property
    def raw_size(self) -> int:
        return self.fill0 + self.n_fill_tok

    @property
    def size(self) -> int:
        """Vocabulary size rounded up to a multiple of 64."""
        return ((self.raw_size + 63) // 64) * 64


class World:
    """K facts (s, r) -> o, a popularity permutation, and 64 templates."""

    def __init__(
        self,
        world_seed: int = 0,
        n_subjects: int = 50_000,
        n_relations: int = 20,
        n_objects: int = 4096,
        n_templates: int = 64,
        templ_len_min: int = 4,
        templ_len_max: int = 8,
        vocab: Vocab | None = None,
    ):
        self.world_seed = world_seed
        self.n_subjects = n_subjects
        self.n_relations = n_relations
        self.n_objects = n_objects
        self.n_facts = n_subjects * n_relations
        self.n_templates = n_templates
        self.vocab = vocab or Vocab()
        assert n_objects == self.vocab.n_obj_tok
        assert n_relations <= self.vocab.n_rel_tok
        assert n_subjects <= self.vocab.n_subj_tok ** 2

        rng = np.random.default_rng(world_seed)
        # Object of each fact, indexed by key = s * n_relations + r.
        self.objects = rng.integers(0, n_objects, size=self.n_facts, dtype=np.int32)
        # rank_to_key[i] is the key with popularity rank i + 1 (rank 1 = most popular).
        self.rank_to_key = rng.permutation(self.n_facts).astype(np.int32)
        # Templates: 64 rows padded with -1, lengths uniform in {4..8}.
        self.templ_len = rng.integers(templ_len_min, templ_len_max + 1, size=n_templates).astype(np.int32)
        self.templ_max = templ_len_max
        self.templates = np.full((n_templates, templ_len_max), -1, dtype=np.int32)
        for t in range(n_templates):
            self.templates[t, : self.templ_len[t]] = self.vocab.templ0 + rng.integers(
                0, self.vocab.n_templ_tok, size=self.templ_len[t]
            )

    # --- key helpers -------------------------------------------------------
    def key_to_subject_relation(self, key):
        key = np.asarray(key)
        return key // self.n_relations, key % self.n_relations

    def subject_tokens(self, s):
        s = np.asarray(s)
        v = self.vocab
        return v.subj0 + s // v.n_subj_tok, v.subj0 + s % v.n_subj_tok

    def decode_subject(self, hi_tok, lo_tok):
        v = self.vocab
        return (np.asarray(hi_tok) - v.subj0) * v.n_subj_tok + (np.asarray(lo_tok) - v.subj0)

    def max_doc_len(self, filler_n: int) -> int:
        # BOS + template(<=8) + subj_hi + subj_lo + rel + obj + filler + EOS
        return 1 + self.templ_max + 4 + filler_n + 1

    def build_docs(self, keys, templ_ids, filler_n: int, rng: np.random.Generator | None):
        """Vectorised document builder.

        Returns (docs int32 [B, L] padded with PAD at the end, obj_pos int32 [B], n_tokens int64).
        Layout of each row: BOS template subj_hi subj_lo relation object filler EOS PAD...
        """
        v = self.vocab
        keys = np.asarray(keys, dtype=np.int64)
        templ_ids = np.asarray(templ_ids, dtype=np.int64)
        B = keys.shape[0]
        L = self.max_doc_len(filler_n)
        s, r = self.key_to_subject_relation(keys)
        hi, lo = self.subject_tokens(s)
        tl = self.templ_len[templ_ids]

        # Fixed-slot layout, -1 marks an empty template slot; then compact.
        full = np.full((B, L), -1, dtype=np.int32)
        full[:, 0] = v.BOS
        full[:, 1 : 1 + self.templ_max] = self.templates[templ_ids]
        c = 1 + self.templ_max
        full[:, c] = hi
        full[:, c + 1] = lo
        full[:, c + 2] = v.rel0 + r
        full[:, c + 3] = v.obj0 + self.objects[keys]
        if filler_n > 0:
            assert rng is not None
            full[:, c + 4 : c + 4 + filler_n] = v.fill0 + rng.integers(
                0, v.n_fill_tok, size=(B, filler_n), dtype=np.int32
            )
        full[:, c + 4 + filler_n] = v.EOS

        keep = full >= 0
        cols = np.cumsum(keep, axis=1) - 1
        rows = np.broadcast_to(np.arange(B)[:, None], (B, L))
        docs = np.full((B, L), v.PAD, dtype=np.int32)
        docs[rows[keep], cols[keep]] = full[keep]
        obj_pos = (1 + tl + 3).astype(np.int32)
        n_tokens = int(keep.sum())
        return docs, obj_pos, n_tokens
