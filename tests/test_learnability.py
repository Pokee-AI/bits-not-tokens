"""Test 5: the pipeline can store facts. Small world, uniform sampling, tiny model."""
import numpy as np
import pytest
import torch

from eval.evaluate import Evaluator
from synth.stream import EqStream
from synth.world import World
from train.model import GPT, lm_loss


@pytest.mark.slow
def test_learnability_smoke():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world = World(0, n_subjects=100, n_relations=20)  # K = 2000
    corpus = {"sampling": "eq", "repeats": 60, "filler_n": 0}
    stream = EqStream(world, repeats=60, filler_n=0, run_seed=1)  # uniform, 120k docs
    torch.manual_seed(0)
    model = GPT(world.vocab.size, world.max_doc_len(0), n_layers=2, d_model=128, n_heads=4).to(device)
    opt = torch.optim.AdamW(model.param_groups(0.1), lr=1e-3, betas=(0.9, 0.95))
    for step in range(stream.max_docs // 128):
        b = stream.next_batch(128)
        x = torch.from_numpy(b.docs.astype(np.int64)).to(device)
        for g in opt.param_groups:
            g["lr"] = 1e-3 * min(1.0, (step + 1) / 100)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = lm_loss(model(x[:, :-1]), x[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    ev = Evaluator(world, corpus, 123, n_indist=2000, n_control=100, batch=1024, device=device)
    # all facts delivered -> control set is empty; evaluate delivered accuracy directly
    nll, _, hit = ev._object_scores(model, np.flatnonzero(stream.n_k > 0), ev.templ_of_key[stream.n_k > 0])
    acc = hit.mean()
    print("top1 on delivered:", acc, "mean nll bits", nll.mean())
    assert acc > 0.95
