import numpy as np

from eval.evaluate import Evaluator
from synth.world import World


def test_flat_corpus_indist_set_is_drawn_from_p():
    """The in-distribution set must come from the base distribution p for flat corpora."""
    import torch
    w = World(0)
    corp = {"sampling": "flat", "zipf_a": 0.5, "zipf_q": 1000, "filler_n": 0}
    ev = Evaluator(w, corp, 12345, n_indist=20_000, n_control=100, batch=1024, device=torch.device("cpu"))
    ranks = np.argsort(w.rank_to_key)[ev.indist_keys]  # key -> rank
    assert np.median(ranks) < 20_000  # p-weighted: half the mass sits in the first ~few thousand ranks
    corp_u = {"sampling": "eq", "repeats": 4, "filler_n": 0}
    ev_u = Evaluator(w, corp_u, 12345, n_indist=20_000, n_control=100, batch=1024, device=torch.device("cpu"))
    assert np.median(np.argsort(w.rank_to_key)[ev_u.indist_keys]) > 300_000
