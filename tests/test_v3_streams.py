import numpy as np
import pytest

from synth.information import expected_facts_delivered, shifted_zipf_probs
from synth.stream import CappedStream, ProbStream, UniformSubsetStream
from synth.world import World


def test_shifted_zipf_closed_form():
    w = World(0)
    p = shifted_zipf_probs(w.n_facts, 0.5, 1000)
    assert p[0] < 6e-4  # top fact about 0.05 % of draws
    s = ProbStream(w, p, 0, run_seed=5)
    N = 200_000
    for _ in range(N // 10_000):
        s.next_batch(10_000)
    expected = expected_facts_delivered(p, N)
    assert abs(s.facts_delivered - expected) / expected < 0.01


def test_uniform_subset():
    w = World(0)
    s = UniformSubsetStream(w, 500, 0, run_seed=1)
    for _ in range(20):
        s.next_batch(1000)
    seen = np.flatnonzero(s.n_k)
    assert set(seen) <= set(w.rank_to_key[:500].tolist())
    assert len(seen) == 500


def _capped(seed=1, cap=5, warmup=2000, block=1000):
    w = World(0, n_subjects=200, n_relations=20)  # K = 4000
    p = shifted_zipf_probs(w.n_facts, 0.5, 50)
    return w, CappedStream(w, p, cap=cap, warmup_docs=warmup, filler_n=2, run_seed=seed, block=block)


def test_cap_never_exceeded_after_warmup_and_budget_respected():
    w, s = _capped()
    budget = 40_000
    n_after_warmup_over_cap_at_warmup_end = None
    while s.tokens_seen < budget:
        s.next_batch(128)
        if s.docs_seen >= s.warmup_docs and n_after_warmup_over_cap_at_warmup_end is None:
            # facts that exceeded the cap during warm-up are allowed to stay above it
            n_after_warmup_over_cap_at_warmup_end = s.n_k.copy()
    nk0 = n_after_warmup_over_cap_at_warmup_end
    # after warm-up, no fact gains a draw once it is at/above the cap
    assert np.all((s.n_k <= s.cap) | (s.n_k == nk0))
    assert np.all(s.n_k[nk0 < s.cap] <= s.cap)
    # the token budget is respected to within one batch
    assert budget <= s.tokens_seen < budget + 128 * w.max_doc_len(2)


def test_capped_stream_deterministic_and_snapshot_exact():
    _, a = _capped(seed=3)
    _, b = _capped(seed=3)
    for _ in range(60):
        assert np.array_equal(a.next_batch(128).docs, b.next_batch(128).docs)
    snap = a.snapshot()
    x = [a.next_batch(128).docs for _ in range(30)]  # crosses a block boundary
    a.restore(snap)
    y = [a.next_batch(128).docs for _ in range(30)]
    assert all(np.array_equal(u, v) for u, v in zip(x, y))
    _, c = _capped(seed=4)
    assert not np.array_equal(c.next_batch(128).docs, b.next_batch(128).docs)


def test_capped_stream_ends_when_everything_capped():
    _, s = _capped(cap=2, warmup=0, block=500)
    with pytest.raises(StopIteration):
        for _ in range(10_000):
            s.next_batch(64)
    assert np.all(s.n_k <= 2)
