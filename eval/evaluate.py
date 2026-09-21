"""Evaluators (section 6). All object losses in bits, on the object token only.

The softmax is restricted to the 4,096 object tokens, so an unseen fact costs exactly
12 bits under a uniform guess and top-1 chance is exactly 1/4096 (matching the
chance term and the ideal-learner closed form). Full-vocabulary NLL is logged too.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

from synth.information import shifted_zipf_probs, zipf_probs
from synth.world import OBJ_BITS, World

LN2 = math.log(2.0)


class Evaluator:
    def __init__(self, world: World, corpus: dict, eval_seed: int, n_indist: int, n_control: int,
                 batch: int, device):
        self.world, self.device, self.batch = world, device, batch
        self.n_control = n_control
        rng = np.random.default_rng(eval_seed)
        K = world.n_facts
        # 1. In-distribution keys from the corpus's own fact distribution (uniform for EQ).
        kind = corpus["sampling"]
        if kind in ("zipf", "shifted_zipf", "capped", "flat"):
            # in-distribution = the corpus's base fact distribution (the uncapped one for capped streams)
            p = (zipf_probs(K, float(corpus["zipf_a"])) if kind == "zipf"
                 else shifted_zipf_probs(K, float(corpus["zipf_a"]), float(corpus["zipf_q"])))
            cdf = np.cumsum(p)
            ranks = np.minimum(np.searchsorted(cdf, rng.random(n_indist), side="right"), K - 1)
            self.indist_keys = world.rank_to_key[ranks].astype(np.int64)
        elif kind == "uniform_subset":
            self.indist_keys = world.rank_to_key[rng.integers(0, int(corpus["k_sub"]), size=n_indist)].astype(np.int64)
        else:
            self.indist_keys = rng.integers(0, K, size=n_indist, dtype=np.int64)
        self.indist_templ = rng.integers(0, world.n_templates, size=n_indist)
        # 2/3/4. One fixed template per fact, and a fixed order for choosing control facts.
        self.templ_of_key = rng.integers(0, world.n_templates, size=K)
        self.head_keys = world.rank_to_key[:1000].astype(np.int64)  # v4: 1,000 most popular facts
        self.control_order = rng.permutation(K)
        self.obj_lo, self.obj_hi = world.vocab.obj0, world.vocab.obj0 + world.vocab.n_obj_tok

    @torch.no_grad()
    def _object_scores(self, model, keys, templ):
        """Returns (nll_bits restricted, nll_bits full vocab, top1_correct) as numpy arrays."""
        docs, obj_pos, _ = self.world.build_docs(keys, templ, 0, None)
        N = len(keys)
        nll_r = np.empty(N, dtype=np.float64)
        nll_f = np.empty(N, dtype=np.float64)
        hit = np.empty(N, dtype=bool)
        for i in range(0, N, self.batch):
            d = torch.from_numpy(docs[i:i + self.batch].astype(np.int64)).to(self.device)
            p = torch.from_numpy(obj_pos[i:i + self.batch].astype(np.int64)).to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = model(d)
            lg = logits[torch.arange(d.shape[0], device=self.device), p - 1].float()
            tgt = d[torch.arange(d.shape[0], device=self.device), p]
            full = F.cross_entropy(lg, tgt, reduction="none")
            obj = lg[:, self.obj_lo:self.obj_hi]
            restricted = F.cross_entropy(obj, tgt - self.obj_lo, reduction="none")
            hit_b = obj.argmax(dim=1) == (tgt - self.obj_lo)
            nll_r[i:i + self.batch] = (restricted / LN2).cpu().numpy()
            nll_f[i:i + self.batch] = (full / LN2).cpu().numpy()
            hit[i:i + self.batch] = hit_b.cpu().numpy()
        return nll_r, nll_f, hit

    @torch.no_grad()
    def probe_loss(self, model, n: int = 20_000) -> float:
        """Cheap in-distribution object loss (bits) on the first n in-distribution documents."""
        model.eval()
        nll, _, _ = self._object_scores(model, self.indist_keys[:n], self.indist_templ[:n])
        model.train()
        return float(nll.mean())

    def evaluate(self, model, n_k: np.ndarray) -> dict:
        model.eval()
        out = {}
        # 1. In-distribution object loss.
        nll_r, nll_f, hit_i = self._object_scores(model, self.indist_keys, self.indist_templ)
        out["obj_loss_bits_indist"] = float(nll_r.mean())
        out["obj_loss_bits_indist_fullvocab"] = float(nll_f.mean())
        # v4: top-1 accuracy weighted by the corpus's base distribution (the in-dist set is sampled from it)
        out["weighted_acc_p"] = float(hit_i.mean())
        h_nll, _, _ = self._object_scores(model, self.head_keys, self.templ_of_key[self.head_keys])
        out["head_loss_bits"] = float(h_nll.mean())
        # 2. Facts stored over delivered facts.
        delivered = np.flatnonzero(n_k > 0).astype(np.int64)
        nd = len(delivered)
        out["facts_delivered"] = nd
        out["bits_delivered"] = OBJ_BITS * nd
        hit_all = np.zeros(self.world.n_facts, dtype=bool)
        nll_all = np.zeros(self.world.n_facts, dtype=np.float32)
        if nd:
            d_nll, _, d_hit = self._object_scores(model, delivered, self.templ_of_key[delivered])
            d_soft = np.maximum(0.0, OBJ_BITS - d_nll).sum()
            d_hits = int(d_hit.sum())
            hit_all[delivered] = d_hit
            nll_all[delivered] = d_nll
        else:
            d_soft, d_hits = 0.0, 0
        out["_hit"], out["_nll"] = hit_all, nll_all  # per-fact arrays, consumed by the trainer
        # 3. Control: fixed random sample of unseen facts.
        unseen = self.control_order[n_k[self.control_order] == 0][: self.n_control]
        c_nll, _, c_hit = self._object_scores(model, unseen.astype(np.int64), self.templ_of_key[unseen])
        out["unseen_top1_acc"] = float(c_hit.mean())
        out["unseen_loss_bits"] = float(c_nll.mean())
        c_soft_mean = float(np.maximum(0.0, OBJ_BITS - c_nll).mean())
        out["delivered_top1_acc"] = d_hits / nd if nd else 0.0
        out["facts_stored"] = d_hits - nd / self.world.n_objects
        out["bits_stored"] = OBJ_BITS * out["facts_stored"]
        out["bits_stored_soft"] = float(d_soft - nd * c_soft_mean)
        model.train()
        return out
