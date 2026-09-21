import numpy as np

from synth.information import shifted_zipf_probs
from synth.stream import ProbStream, probe_keys_for, probe_schedule
from synth.world import World

CFG = {"probe_n_facts": 50, "probe_rank_lo": 1000, "probe_rank_hi": 3000, "probe_window_lo": 20_000,
       "probe_window_hi": 30_000, "probe_exposures": 20, "probe_measure_every": 5_000}


def _stream(seed):
    w = World(0, n_subjects=200, n_relations=20)  # K = 4000
    pk = probe_keys_for(w, CFG)
    p = shifted_zipf_probs(w.n_facts, 0.5, 50)
    p[np.argsort(w.rank_to_key)[pk]] = 0.0
    p /= p.sum()
    st = ProbStream(w, p, 0, seed)
    pos, facts = probe_schedule(pk, CFG, seed)
    st.attach_probe(pk, pos, facts)
    return w, pk, st


def test_probe_facts_only_in_window_and_exactly_n_times():
    w, pk, st = _stream(1)
    total = 40_000
    while st.docs_seen < total:
        st.next_batch(128)
    assert np.all(st.n_k[pk] == CFG["probe_exposures"])
    assert st.first_seen[pk].min() >= CFG["probe_window_lo"]
    assert st.last_seen[pk].max() < CFG["probe_window_hi"]
    # first/last seen are consistent with n_k for every delivered fact
    d = st.n_k > 0
    assert np.all(st.first_seen[d] <= st.last_seen[d]) and np.all(st.last_seen[~d] == -1)


def test_probe_stream_deterministic_and_snapshot_exact():
    _, _, a = _stream(3)
    _, _, b = _stream(3)
    for _ in range(200):
        assert np.array_equal(a.next_batch(128).keys, b.next_batch(128).keys)
    snap = a.snapshot()
    x = [a.next_batch(128).keys for _ in range(40)]
    a.restore(snap)
    assert all(np.array_equal(u, v) for u, v in zip(x, [a.next_batch(128).keys for _ in range(40)]))
    _, _, c = _stream(4)
    for _ in range(200):
        c.next_batch(128)
    assert not np.array_equal(c.n_k, a.n_k)


def test_probe_keys_are_world_seeded_and_in_rank_range():
    w = World(0, n_subjects=200, n_relations=20)
    pk1, pk2 = probe_keys_for(w, CFG), probe_keys_for(w, CFG)
    assert np.array_equal(pk1, pk2) and len(set(pk1.tolist())) == CFG["probe_n_facts"]
    ranks = np.argsort(w.rank_to_key)[pk1]
    assert ranks.min() >= CFG["probe_rank_lo"] and ranks.max() < CFG["probe_rank_hi"]
