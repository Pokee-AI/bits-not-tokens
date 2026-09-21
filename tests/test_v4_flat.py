import numpy as np

from synth.information import capped_exposures, flatten, shifted_zipf_probs, solve_tau
from synth.stream import FlatStream
from synth.world import World


def test_tau_reproduces_c_and_no_zero_probability():
    p = shifted_zipf_probs(1_000_000, 0.5, 1000)
    for n_stat, c in [(10_400_000, 1000), (10_400_000, 300), (10_400_000, 100), (10_400_000, 30), (24_000_000, 30)]:
        tau = solve_tau(p, n_stat, c)
        assert abs(capped_exposures(p, tau, n_stat) - c) / c < 0.005
        q = flatten(p, tau)
        assert q.min() > 0 and abs(q.sum() - 1) < 1e-12
        assert (p > tau).sum() > 0  # some facts are actually capped


def _flat(seed=1, warmup=3_000, n_stat=60_000, c=25):
    w = World(0, n_subjects=200, n_relations=20)  # K = 4000
    p = shifted_zipf_probs(w.n_facts, 0.5, 50)
    tau = solve_tau(p, n_stat, c)
    return w, p, tau, FlatStream(w, p, tau, warmup, filler_n=2, run_seed=seed)


def test_empirical_capped_exposures_and_budget():
    w, p, tau, s = _flat()
    warmup, n_stat = 3_000, 60_000
    budget_tokens = (warmup + n_stat) * (12 + 2)  # mean 12 + filler 2 tokens per doc
    while s.tokens_seen < budget_tokens:
        s.next_batch(128)
    assert budget_tokens <= s.tokens_seen < budget_tokens + 128 * w.max_doc_len(2)
    capped = w.rank_to_key[p > tau]
    exp_warm = s.warmup_docs * p[p > tau]
    n_stat_actual = s.docs_seen - s.warmup_docs
    exp_total = exp_warm + n_stat_actual * tau / np.minimum(p, tau).sum()
    ratio = s.n_k[capped].mean() / exp_total.mean()
    assert abs(ratio - 1) < 0.02, ratio


def test_flat_stream_deterministic_and_switches_once():
    _, _, _, a = _flat(seed=3)
    _, _, _, b = _flat(seed=3)
    for _ in range(40):
        assert np.array_equal(a.next_batch(128).docs, b.next_batch(128).docs)
    snap = a.snapshot()
    x = [a.next_batch(128).docs for _ in range(10)]
    a.restore(snap)
    assert all(np.array_equal(u, v) for u, v in zip(x, [a.next_batch(128).docs for _ in range(10)]))
    _, _, _, c = _flat(seed=4)
    assert not np.array_equal(c.next_batch(128).docs, b.next_batch(128).docs)
