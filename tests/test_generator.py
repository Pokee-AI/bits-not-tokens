import numpy as np
import pytest

from synth.information import expected_facts_delivered, zipf_probs
from synth.stream import EqStream, ZipfStream
from synth.world import World


def test_generator_determinism():
    w1, w2 = World(0), World(0)
    assert np.array_equal(w1.objects, w2.objects)
    assert np.array_equal(w1.rank_to_key, w2.rank_to_key)
    assert np.array_equal(w1.templates, w2.templates)
    s1 = ZipfStream(w1, 0.5, 8, run_seed=1)
    s2 = ZipfStream(w2, 0.5, 8, run_seed=1)
    for _ in range(3):
        b1, b2 = s1.next_batch(64), s2.next_batch(64)
        assert np.array_equal(b1.docs, b2.docs)
        assert np.array_equal(b1.keys, b2.keys)
    # World is identical across run seeds; the stream differs.
    s3 = ZipfStream(World(0), 0.5, 8, run_seed=2)
    assert np.array_equal(s3.world.objects, w1.objects)
    assert not np.array_equal(s3.next_batch(64).docs, s1.next_batch(64).docs)
    # Snapshot / restore reproduces the same documents.
    snap = s1.snapshot()
    a = s1.next_batch(32).docs
    s1.restore(snap)
    assert np.array_equal(s1.next_batch(32).docs, a)


def test_closed_form_facts_delivered():
    w = World(0)
    a, N = 0.5, 200_000
    s = ZipfStream(w, a, 0, run_seed=7)
    for _ in range(N // 10_000):
        s.next_batch(10_000)
    assert s.docs_seen == N
    expected = expected_facts_delivered(zipf_probs(w.n_facts, a), N)
    assert abs(s.facts_delivered - expected) / expected < 0.01


def test_document_format():
    w = World(0)
    v = w.vocab
    n = 5
    s = ZipfStream(w, 1.0, n, run_seed=3)
    b = s.next_batch(256)
    L = w.max_doc_len(n)
    assert b.docs.shape == (256, L)
    for i in range(256):
        row, key, op = b.docs[i], b.keys[i], b.obj_pos[i]
        assert row[0] == v.BOS
        subj, rel = w.key_to_subject_relation(key)
        # object sits at BOS + template + subj_hi + subj_lo + relation
        assert row[op] == v.obj0 + w.objects[key]
        assert row[op - 1] == v.rel0 + rel
        assert w.decode_subject(row[op - 3], row[op - 2]) == subj
        assert 4 <= op - 4 <= 8  # template length
        assert np.all((row[1:op - 3] >= v.templ0) & (row[1:op - 3] < v.subj0))
        fill = row[op + 1 : op + 1 + n]
        assert np.all((fill >= v.fill0) & (fill < v.fill0 + v.n_fill_tok))
        assert row[op + 1 + n] == v.EOS
        assert np.all(row[op + 2 + n :] == v.PAD)
    # Filler length is exactly n: non-PAD count = 6 + template + n
    nonpad = (b.docs != v.PAD).sum(axis=1)
    assert np.array_equal(nonpad, b.obj_pos - 4 + 6 + n)
    assert b.n_tokens == nonpad.sum()
    # PAD is masked from the loss.
    import torch
    from train.model import lm_loss
    logits = torch.randn(256, L - 1, v.size)
    tgt = torch.from_numpy(b.docs[:, 1:].astype(np.int64))
    loss = lm_loss(logits, tgt, pad_id=v.PAD)
    # Perturbing logits at PAD target positions must not change the loss ...
    logits2 = logits.clone()
    logits2[tgt == v.PAD] += torch.randn_like(logits2[tgt == v.PAD]) * 10
    assert torch.allclose(loss, lm_loss(logits2, tgt, pad_id=v.PAD))
    # ... while perturbing a non-PAD position does.
    logits3 = logits.clone()
    logits3[tgt != v.PAD] += 1.0 * torch.randn_like(logits3[tgt != v.PAD])
    assert not torch.allclose(loss, lm_loss(logits3, tgt, pad_id=v.PAD))


def test_eq4_every_fact_exactly_four_times():
    w = World(0, n_subjects=100, n_relations=20)  # K = 2000
    s = EqStream(w, repeats=4, filler_n=2, run_seed=1)
    assert s.max_docs == 8000
    while s.docs_seen < s.max_docs:
        s.next_batch(500)
    assert np.all(s.n_k == 4)
    with pytest.raises(StopIteration):
        s.next_batch(1)
