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
from synth.stream import make_stream, make_stream_v3  # noqa: E402
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
    # v3 extras
    "model_size", "n_params_nonemb", "n_params_total", "bits_stored_per_param",
    "bits_stored_per_param_total", "k_sub", "frac_stored", "mean_exposures", "cap",
    "warmup_docs", "facts_with_at_least_cstar_exposures", "cap_adequacy", "budget_tokens",
    "n_uncapped",
]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__)),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def measurement_points(cfg: dict, max_docs: int, batch: int) -> list[int]:
    pts = list(np.geomspace(cfg["meas_min_docs"], cfg["meas_max_docs"], cfg["meas_points"]))
    ratio = pts[1] / pts[0]
    while pts[-1] * ratio < max_docs:  # corpora larger than meas_max_docs: continue the sequence
        pts.append(pts[-1] * ratio)
    pts = [int(round(p / batch)) * batch for p in pts]
    pts = [p for p in pts if p <= max_docs]
    if not pts or pts[-1] != max_docs:
        pts.append(max_docs)
    return pts


def token_points(budget: int, n: int = 8, start_div: int = 300) -> list[int]:
    """v3 Phase P: n log-spaced token counts from budget/start_div to budget."""
    return [int(round(x)) for x in np.geomspace(budget / start_div, budget, n)]


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
    # v3 options (Experiment Brief v3)
    ap.add_argument("--model", default="L", help="S | M | L from configs/models.yaml")
    ap.add_argument("--budget-tokens", type=int, default=None,
                    help="v3 Phase P: stop at this many non-PAD tokens; 8 log-spaced points from B/300")
    ap.add_argument("--cap", type=int, default=None, help="v3 capped stream: c*")
    ap.add_argument("--warmup-docs", type=int, default=None, help="v3 capped stream: W")
    ap.add_argument("--probe-every", type=int, default=0,
                    help="v3 Phase A: log in-distribution object loss every N steps to artifacts/<run>/probe.csv")
    ap.add_argument("--save-final", action="store_true", help="save final weights to artifacts/<run>/final.pt")
    ap.add_argument("--results-subdir", default="runs", help="results/<subdir>/<run>.csv")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs", "common.yaml")))
    corpus = yaml.safe_load(open(args.corpus))
    v3 = corpus.get("sampling") in ("uniform_subset", "shifted_zipf", "capped") or args.budget_tokens
    msize = yaml.safe_load(open(os.path.join(root, "configs", "models.yaml")))[args.model]
    if args.model != "L":
        lr_file = os.path.join(root, "configs", "v3_lr.yaml")
        if os.path.exists(lr_file):
            cfg["lr"] = yaml.safe_load(open(lr_file)).get(args.model, cfg["lr"])
    if args.lr is not None:
        cfg["lr"] = args.lr
    max_docs = args.max_docs or int(corpus["max_docs"])
    batch = int(cfg["batch_size"])
    max_docs = (max_docs // batch) * batch
    lr = float(cfg["lr"])
    run_name = f"{corpus['name']}_seed{args.seed}{('_' + args.tag) if args.tag else ''}"
    full_cfg = {"common": cfg, "corpus": corpus, "seed": args.seed, "max_docs": max_docs,
                "cooldown": not args.no_cooldown}
    if v3 or args.model != "L":
        full_cfg["v3"] = {"model": args.model, "model_cfg": msize, "budget_tokens": args.budget_tokens,
                          "cap": args.cap, "warmup_docs": args.warmup_docs}
    config_hash = hashlib.sha256(json.dumps(full_cfg, sort_keys=True).encode()).hexdigest()[:12]
    commit = git_commit()
    art_dir = os.path.join(root, "artifacts", run_name)
    os.makedirs(art_dir, exist_ok=True)
    os.makedirs(os.path.join(root, "results", args.results_subdir), exist_ok=True)
    csv_path = os.path.join(root, "results", args.results_subdir, f"{run_name}.csv")
    json.dump(full_cfg, open(os.path.join(art_dir, "config.json"), "w"), indent=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    world = World(cfg["world_seed"], cfg["n_subjects"], cfg["n_relations"], cfg["n_objects"],
                  cfg["n_templates"])
    stream = (make_stream_v3(world, corpus, args.seed, args.cap, args.warmup_docs) if v3
              else make_stream(world, corpus, args.seed))
    filler_n = int(corpus["filler_n"])
    doc_len = world.max_doc_len(filler_n)
    model = GPT(world.vocab.size, doc_len, msize["n_layers"], msize["d_model"], msize["n_heads"],
                msize["mlp_ratio"]).to(device)
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
    elif corpus["sampling"] == "eq":
        ideal = lambda n: ideal_loss_bits_eq(world.n_facts, int(corpus["repeats"]), n)  # noqa: E731
    else:
        ideal = lambda n: float("nan")  # noqa: E731

    # progress unit: documents (v1/v2, Phase A) or non-PAD tokens (Phase P budget)
    unit = "tokens" if args.budget_tokens else "docs"
    progress = (lambda: stream.tokens_seen) if unit == "tokens" else (lambda: stream.docs_seen)
    points = token_points(args.budget_tokens) if unit == "tokens" else measurement_points(cfg, max_docs, batch)
    probe_path = os.path.join(art_dir, "probe.csv")
    if args.probe_every:
        open(probe_path, "w").write("step,docs,tokens,obj_loss_bits_indist\n")
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

    exhausted = [False]

    def run_main_until(target: int):
        last = None
        while progress() < target and not exhausted[0]:
            try:
                last = train_step(main_lr(step))
            except StopIteration:
                exhausted[0] = True  # capped stream: every fact reached c*
                print(f"[{run_name}] stream exhausted at docs {stream.docs_seen} tokens {stream.tokens_seen}",
                      flush=True)
                break
            if args.probe_every and step % args.probe_every == 0:
                pl = evaluator.probe_loss(model)
                open(probe_path, "a").write(f"{step},{stream.docs_seen},{stream.tokens_seen},{pl:.5f}\n")
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
        if unit == "docs":
            branch_from = D - int(round(cfg["cooldown_fraction"] * D / batch)) * batch
        else:
            branch_from = int(round((1 - cfg["cooldown_fraction"]) * D))
        run_main_until(branch_from if not args.no_cooldown else D)
        loss_main = float(np.mean(recent[-4:])) if recent else float("nan")
        if not args.no_cooldown and not exhausted[0]:
            snap = snapshot()
            if unit == "docs":
                n_branch = (D - stream.docs_seen) // batch
            else:  # tokens: estimate the number of batches from the running tokens/batch
                n_branch = max(1, int(round((D - stream.tokens_seen) / (stream.tokens_seen / max(step, 1)))))
            lr0 = main_lr(step)
            last = None
            for t in range(n_branch):
                try:
                    last = train_step(lr0 * (1.0 - t / n_branch))
                except StopIteration:
                    exhausted[0] = True
                    break
            if unit == "docs" and not exhausted[0]:
                assert stream.docs_seen == D, (stream.docs_seen, D)
            if last is not None:
                loss_main = float(last.item())
        D = progress()  # actual position (tokens: within one batch of the target)
        ev = evaluator.evaluate(model, stream.n_k)
        np.savez_compressed(os.path.join(art_dir, f"nk_{D}.npz"), n_k=stream.n_k)
        # per-fact top-1 hits and NLL (bits, fp16) over all K facts; undelivered facts are 0
        np.savez_compressed(os.path.join(art_dir, f"hits_{D}.npz"),
                            hit=np.packbits(ev["_hit"]), nll_bits=ev["_nll"].astype(np.float16))
        row = {"corpus": corpus["name"], "zipf_a": corpus.get("zipf_a", ""), "filler_n": filler_n,
               "seed": args.seed, "lr": lr, "docs": stream.docs_seen, "train_tokens": stream.tokens_seen,
               "ideal_loss_bits": ideal(stream.docs_seen), "train_loss_nats": loss_main,
               "wall_seconds": time.time() - t0, "git_commit": commit, "config_hash": config_hash,
               "cooldown": int(not args.no_cooldown), **ev}
        row.update({"model_size": args.model, "n_params_nonemb": counts["non_embedding"],
                    "n_params_total": counts["total"],
                    "bits_stored_per_param": ev["bits_stored"] / counts["non_embedding"],
                    "bits_stored_per_param_total": ev["bits_stored"] / counts["total"],
                    "budget_tokens": args.budget_tokens or ""})
        if corpus.get("sampling") == "uniform_subset":
            row.update({"k_sub": corpus["k_sub"], "frac_stored": ev["facts_stored"] / corpus["k_sub"],
                        "mean_exposures": stream.docs_seen / corpus["k_sub"]})
        if args.cap is not None:
            hit = ev["_hit"]
            at_cap = stream.n_k >= args.cap
            row.update({"cap": args.cap, "warmup_docs": args.warmup_docs,
                        "facts_with_at_least_cstar_exposures": int(at_cap.sum()),
                        "cap_adequacy": (float(hit[at_cap].mean()) - 1 / world.n_objects) if at_cap.any() else "",
                        "n_uncapped": getattr(stream, "n_uncapped", "")})
        ev.pop("_hit"), ev.pop("_nll")
        write_row({k: v for k, v in row.items() if not k.startswith("_")})
        print(f"[{run_name}] MEASURE docs {D} tokens {row['train_tokens']} "
              f"delivered {ev['facts_delivered']} indist {ev['obj_loss_bits_indist']:.3f} "
              f"ideal {row['ideal_loss_bits']:.3f} stored {ev['facts_stored']:.0f} "
              f"soft {ev['bits_stored_soft']:.0f} unseen_acc {ev['unseen_top1_acc']:.2e} "
              f"wall {row['wall_seconds']:.0f}s", flush=True)
        if exhausted[0] or D >= points[-1]:
            break  # the main run's last 10% is never evaluated
        if not args.no_cooldown:
            restore(snap)
            del snap
    if args.save_final:
        torch.save({"model": model.state_dict(), "config": full_cfg, "docs": stream.docs_seen,
                    "tokens": stream.tokens_seen, "step": step}, os.path.join(art_dir, "final.pt"))
    print(f"[{run_name}] DONE wall {time.time() - t0:.0f}s docs {stream.docs_seen} tokens {stream.tokens_seen}",
          flush=True)


if __name__ == "__main__":
    main()
