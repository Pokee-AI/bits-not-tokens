"""One training run = (corpus config, run seed). Single process, single GPU.

Main run at constant LR; at each measurement point D_i a cooldown branch resumes from
a snapshot taken at 0.9*D_i, trains on the same documents with LR decaying linearly to
zero, is evaluated and discarded (section 5.1).
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.evaluate import Evaluator  # noqa: E402
from synth.information import ideal_loss_bits_eq, ideal_loss_bits_zipf, zipf_probs  # noqa: E402
from synth.stream import make_stream  # noqa: E402
from synth.world import World  # noqa: E402
from train.model import GPT, lm_loss  # noqa: E402

CSV_COLS = [
    "corpus", "zipf_a", "filler_n", "seed", "lr", "docs", "train_tokens",
    "facts_delivered", "bits_delivered",
    "obj_loss_bits_indist", "ideal_loss_bits",
    "facts_stored", "bits_stored", "bits_stored_soft", "unseen_top1_acc",
    "train_loss_nats", "wall_seconds", "git_commit", "config_hash",
    # extras (not in the brief's schema, appended at the end)
    "obj_loss_bits_indist_fullvocab", "unseen_loss_bits", "delivered_top1_acc", "cooldown",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def measurement_points(cfg: dict, max_docs: int, batch: int) -> list[int]:
    pts = np.geomspace(cfg["meas_min_docs"], cfg["meas_max_docs"], cfg["meas_points"])
    pts = [int(round(p / batch)) * batch for p in pts]
    pts = [p for p in pts if p <= max_docs]
    if not pts or pts[-1] != max_docs:
        pts.append(max_docs)
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="configs/<name>.yaml")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=None, help="override configs/common.yaml lr")
    ap.add_argument("--max-docs", type=int, default=None, help="override corpus max_docs")
    ap.add_argument("--tag", default="", help="suffix for result/artifact names")
    ap.add_argument("--no-cooldown", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--log-every", type=int, default=500)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs", "common.yaml")))
    corpus = yaml.safe_load(open(args.corpus))
    if args.lr is not None:
        cfg["lr"] = args.lr
    max_docs = args.max_docs or int(corpus["max_docs"])
    batch = int(cfg["batch_size"])
    max_docs = (max_docs // batch) * batch
    lr = float(cfg["lr"])
    run_name = f"{corpus['name']}_seed{args.seed}{('_' + args.tag) if args.tag else ''}"
    full_cfg = {"common": cfg, "corpus": corpus, "seed": args.seed, "max_docs": max_docs,
                "cooldown": not args.no_cooldown}
    config_hash = hashlib.sha256(json.dumps(full_cfg, sort_keys=True).encode()).hexdigest()[:12]
    commit = git_commit()
    art_dir = os.path.join(root, "artifacts", run_name)
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(os.path.join(root, "results", "runs"), exist_ok=True)
    csv_path = os.path.join(root, "results", "runs", f"{run_name}.csv")
    json.dump(full_cfg, open(os.path.join(art_dir, "config.json"), "w"), indent=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    world = World(cfg["world_seed"], cfg["n_subjects"], cfg["n_relations"], cfg["n_objects"],
                  cfg["n_templates"])
    stream = make_stream(world, corpus, args.seed)
    filler_n = int(corpus["filler_n"])
    doc_len = world.max_doc_len(filler_n)
    model = GPT(world.vocab.size, doc_len, cfg["n_layers"], cfg["d_model"], cfg["n_heads"],
                cfg["mlp_ratio"]).to(device)
    counts = model.param_counts()
    print(f"[{run_name}] params {counts} vocab {world.vocab.size} doc_len {doc_len} "
          f"device {device} lr {lr} max_docs {max_docs} commit {commit} cfg {config_hash}", flush=True)
    opt = torch.optim.AdamW(model.param_groups(cfg["weight_decay"]), lr=lr,
                            betas=tuple(cfg["betas"]), fused=device.type == "cuda")
    fwd = torch.compile(model) if args.compile else model
    evaluator = Evaluator(world, corpus, cfg["eval_seed"], cfg["eval_indist_docs"],
                          cfg["eval_control_facts"], cfg["eval_batch"], device)
    if corpus["sampling"] == "zipf":
        p_zipf = zipf_probs(world.n_facts, float(corpus["zipf_a"]))
        ideal = lambda n: ideal_loss_bits_zipf(p_zipf, n)  # noqa: E731
    else:
        ideal = lambda n: ideal_loss_bits_eq(world.n_facts, int(corpus["repeats"]), n)  # noqa: E731

    points = measurement_points(cfg, max_docs, batch)
    print(f"[{run_name}] measurement points: {points}", flush=True)
    warmup = int(cfg["warmup_steps"])
    clip = float(cfg["grad_clip"])
    pad = world.vocab.PAD
    t0 = time.time()
    step = 0
    recent = []

    def train_step(lr_now: float) -> float:
        nonlocal step
        b = stream.next_batch(batch)
        x = torch.from_numpy(b.docs.astype(np.int64)).to(device, non_blocking=True)
        for g in opt.param_groups:
            g["lr"] = lr_now
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = fwd(x[:, :-1])
        loss = lm_loss(logits, x[:, 1:], pad_id=pad)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        step += 1
        return loss

    def main_lr(s: int) -> float:
        return lr * min(1.0, (s + 1) / warmup)

    def run_main_until(target_docs: int):
        last = None
        while stream.docs_seen < target_docs:
            last = train_step(main_lr(step))
            if step % args.log_every == 0:
                l = last.item()
                recent.append(l)
                el = time.time() - t0
                print(f"[{run_name}] step {step} docs {stream.docs_seen} loss {l:.4f} "
                      f"lr {main_lr(step):.2e} {stream.docs_seen / el:.0f} docs/s wall {el:.0f}s",
                      flush=True)
        return last

    def snapshot():
        return {"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "opt": copy.deepcopy(opt.state_dict()), "stream": stream.snapshot(), "step": step,
                "rng": torch.get_rng_state()}

    def restore(snap):
        nonlocal step
        model.load_state_dict(snap["model"])
        opt.load_state_dict(snap["opt"])
        stream.restore(snap["stream"])
        step = snap["step"]
        torch.set_rng_state(snap["rng"])

    def write_row(row: dict):
        new = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in CSV_COLS})

    for D in points:
        start = int(round(cfg["cooldown_fraction"] * D / batch)) * batch  # docs in the cooldown
        branch_from = D - start
        run_main_until(branch_from if not args.no_cooldown else D)
        loss_main = float(np.mean(recent[-4:])) if recent else float("nan")
        if not args.no_cooldown:
            snap = snapshot()
            n_branch = (D - stream.docs_seen) // batch
            lr0 = main_lr(step)
            last = None
            for t in range(n_branch):
                last = train_step(lr0 * (1.0 - t / n_branch))
            assert stream.docs_seen == D, (stream.docs_seen, D)
            loss_main = float(last.item())
        ev = evaluator.evaluate(model, stream.n_k)
        np.savez_compressed(os.path.join(art_dir, f"nk_{D}.npz"), n_k=stream.n_k)
        row = {"corpus": corpus["name"], "zipf_a": corpus.get("zipf_a", ""), "filler_n": filler_n,
               "seed": args.seed, "lr": lr, "docs": D, "train_tokens": stream.tokens_seen,
               "ideal_loss_bits": ideal(D), "train_loss_nats": loss_main,
               "wall_seconds": time.time() - t0, "git_commit": commit, "config_hash": config_hash,
               "cooldown": int(not args.no_cooldown), **ev}
        write_row(row)
        print(f"[{run_name}] MEASURE docs {D} tokens {row['train_tokens']} "
              f"delivered {ev['facts_delivered']} indist {ev['obj_loss_bits_indist']:.3f} "
              f"ideal {row['ideal_loss_bits']:.3f} stored {ev['facts_stored']:.0f} "
              f"soft {ev['bits_stored_soft']:.0f} unseen_acc {ev['unseen_top1_acc']:.2e} "
              f"wall {row['wall_seconds']:.0f}s", flush=True)
        if not args.no_cooldown:
            if D == points[-1]:
                break  # the main run's last 10% is never evaluated
            restore(snap)
            del snap
    print(f"[{run_name}] DONE wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
