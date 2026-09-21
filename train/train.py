"""One training run = (corpus config, run seed). Single process, single GPU.

Main run at constant LR; at each measurement point D_i a cooldown branch resumes from
a snapshot taken at 0.9*D_i, trains on the same documents with LR decaying linearly to
zero, is evaluated and discarded (section 5.1).
"""
from __future__ import annotations

import argparse
import copy
import math
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
from synth.stream import make_stream, make_stream_v3, make_stream_v5, probe_keys_for  # noqa: E402
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
    # v4 extras
    "corpus_type", "level", "tau", "n_facts_at_cap", "weighted_acc_p", "head_loss_bits",
    # v5 extras
    "setting", "weight_decay", "schedule", "mid_schedule", "probe_acc_branch", "probe_acc_main_at_snapshot",
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
    # v4 options (Experiment Brief v4.1)
    ap.add_argument("--tau", type=float, default=None, help="v4 flat stream: flattening threshold")
    ap.add_argument("--level", default="", help="v4: flattening level label (1000/300/100/30/x30)")
    ap.add_argument("--corpus-type", default="", help="v4: RAW | FLAT | CURATED")
    ap.add_argument("--stop-after-points", type=int, default=None,
                    help="stop after this many measurement points (determinism checks)")
    # v5 options (Experiment Brief v5.1)
    ap.add_argument("--v5", action="store_true", help="v5: attach the retention probe (configs/v5.yaml)")
    ap.add_argument("--probe-config", default="configs/v5.yaml", help="probe config (smoke tests use a small one)")
    ap.add_argument("--wd", type=float, default=None, help="override weight decay (v5 optimizer ablation)")
    ap.add_argument("--schedule", default="constant", choices=["constant", "cosine"],
                    help="cosine: 100-step warmup then cosine to 10%% of peak at --budget-tokens; no cooldown branches")
    ap.add_argument("--setting", default="", help="v5 optimizer-setting label (base/wd0/lr3/wd0_lr3/cosine)")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = yaml.safe_load(open(os.path.join(root, "configs", "common.yaml")))
    corpus = yaml.safe_load(open(args.corpus))
    v3 = corpus.get("sampling") in ("uniform_subset", "shifted_zipf", "capped", "flat") or args.budget_tokens
    if args.tau is not None:
        corpus["tau"] = args.tau
    msize = yaml.safe_load(open(os.path.join(root, "configs", "models.yaml")))[args.model]
    if args.model != "L":
        lr_file = os.path.join(root, "configs", "v3_lr.yaml")
        if os.path.exists(lr_file):
            cfg["lr"] = yaml.safe_load(open(lr_file)).get(args.model, cfg["lr"])
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.wd is not None:
        cfg["weight_decay"] = args.wd
    if args.schedule == "cosine":
        args.no_cooldown = True  # cosine runs have no cooldown branches (brief v5.1 §4)
    probe_cfg = yaml.safe_load(open(os.path.join(root, args.probe_config))) if args.v5 else None
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
    if args.tau is not None:
        full_cfg["v4"] = {"tau": args.tau, "level": args.level, "corpus_type": args.corpus_type}
    if args.v5:
        full_cfg["v5"] = {"probe": probe_cfg, "setting": args.setting, "weight_decay": cfg["weight_decay"],
                          "schedule": args.schedule}
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
    if args.v5:
        stream = make_stream_v5(world, corpus, args.seed, probe_cfg, args.warmup_docs)
        probe_keys = probe_keys_for(world, probe_cfg)
        is_probe = np.zeros(world.n_facts, dtype=bool)
        is_probe[probe_keys] = True
        np.save(os.path.join(art_dir, "probe_keys.npy"), probe_keys)
        retention_path = os.path.join(art_dir, "probe_retention.csv")
        open(retention_path, "w").write("docs,tokens,step,weights,probe_acc\n")
        next_probe_docs = [int(probe_cfg["probe_window_hi"])]
    else:
        stream = (make_stream_v3(world, corpus, args.seed, args.cap, args.warmup_docs) if v3
                  else make_stream(world, corpus, args.seed))
        is_probe = None
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
        w = lr * min(1.0, (s + 1) / warmup)
        if args.schedule == "cosine":  # cosine in token space: exactly 10 % of peak at the budget
            frac = min(1.0, stream.tokens_seen / args.budget_tokens)
            return w * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        return w

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
            if args.v5 and stream.docs_seen >= next_probe_docs[0]:
                acc = evaluator.probe_accuracy(model, probe_keys)
                open(retention_path, "a").write(f"{stream.docs_seen},{stream.tokens_seen},{step},main,{acc:.5f}\n")
                next_probe_docs[0] += int(probe_cfg["probe_measure_every"])
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

    for pi, D in enumerate(points):
        if unit == "docs":
            branch_from = D - int(round(cfg["cooldown_fraction"] * D / batch)) * batch
        else:
            branch_from = int(round((1 - cfg["cooldown_fraction"]) * D))
        run_main_until(branch_from if not args.no_cooldown else D)
        loss_main = float(np.mean(recent[-4:])) if recent else float("nan")
        probe_main = ""
        if args.v5 and not args.no_cooldown:
            probe_main = evaluator.probe_accuracy(model, probe_keys)  # main-run weights at snapshot time
            open(retention_path, "a").write(f"{stream.docs_seen},{stream.tokens_seen},{step},main_at_snapshot,{probe_main:.5f}\n")
        if not args.no_cooldown and not exhausted[0]:
            snap = snapshot()
            if unit == "docs":
                n_branch = (D - stream.docs_seen) // batch
            else:  # tokens: estimate the number of batches from the running tokens/batch
                n_branch = max(1, int(round((D - stream.tokens_seen) / (stream.tokens_seen / max(step, 1)))))
            lr0 = main_lr(step)
            last = None
            t = 0
            while (t < n_branch) if unit == "docs" else (progress() < D):
                try:
                    last = train_step(lr0 * (1.0 - min(t, n_branch - 1) / n_branch))
                except StopIteration:
                    exhausted[0] = True
                    break
                t += 1
            if unit == "docs" and not exhausted[0]:
                assert stream.docs_seen == D, (stream.docs_seen, D)
            if last is not None:
                loss_main = float(last.item())
        D = stream.docs_seen  # artifacts are keyed by documents (tokens: within one batch of target)
        ev = evaluator.evaluate(model, stream.n_k, exclude=is_probe)
        np.savez_compressed(os.path.join(art_dir, f"nk_{D}.npz"), n_k=stream.n_k)
        if args.v5:
            np.savez_compressed(os.path.join(art_dir, f"seen_{D}.npz"), first_seen=stream.first_seen,
                                last_seen=stream.last_seen)
            probe_branch = evaluator.probe_accuracy(model, probe_keys)
            open(retention_path, "a").write(f"{D},{stream.tokens_seen},{step},"
                                            f"{'main' if args.no_cooldown else 'branch'},{probe_branch:.5f}\n")
        # per-fact top-1 hits and NLL (bits, fp16) over all K facts; undelivered facts are 0
        np.savez_compressed(os.path.join(art_dir, f"hits_{D}.npz"),
                            hit=np.packbits(ev["_hit"]), nll_bits=ev["_nll"].astype(np.float16))
        row = {"corpus": corpus["name"], "zipf_a": corpus.get("zipf_a", ""), "filler_n": filler_n,
               "seed": args.seed, "lr": lr, "docs": stream.docs_seen, "train_tokens": stream.tokens_seen,
               "ideal_loss_bits": ideal(stream.docs_seen), "train_loss_nats": loss_main,
               "wall_seconds": time.time() - t0, "git_commit": commit, "config_hash": config_hash,
               "cooldown": int(not args.no_cooldown), **ev}
        row.update({"corpus_type": args.corpus_type, "level": args.level, "tau": args.tau if args.tau is not None else "",
                    "n_facts_at_cap": getattr(stream, "n_at_cap", "")})
        if args.v5:
            row.update({"setting": args.setting, "weight_decay": cfg["weight_decay"], "schedule": args.schedule,
                        "mid_schedule": int(args.schedule == "cosine" and pi < len(points) - 1),
                        "probe_acc_branch": probe_branch, "probe_acc_main_at_snapshot": probe_main})
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
        if exhausted[0] or pi == len(points) - 1:
            break  # the main run's last 10% is never evaluated
        if args.stop_after_points and pi + 1 >= args.stop_after_points:
            print(f"[{run_name}] stopping after {pi + 1} measurement points (--stop-after-points)", flush=True)
            break
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
