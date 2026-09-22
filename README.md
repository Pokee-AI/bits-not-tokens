# Bits, not tokens

**What a small language model stores from a corpus is set by how often each fact is
repeated — its exposure *rate* — not by how many bits the corpus delivers, and not by how
many tokens it costs.** In a synthetic world of 1,000,000 random 12-bit facts, flattening a
web-like (Zipf) repetition profile stores 2.0× (2M-parameter model), 2.8× (8M) and 4.5–5.1×
(31M) more facts from the same 300M-token budget, with no loss on the raw distribution;
removing noise tokens buys more documents but no more facts; and a stationary stream reaches
its steady state within ~10⁷ documents and does not grow with 2× or 4× longer training.
The advantage holds under five optimizer recipes; forgetting is finite without weight decay
and scales with the learning rate. (Figures: `figures/v4_fig_dose_response.png`,
`figures/v5_fig_optimizer.png`; numbers: `RESULTS_V4.md`, `RESULTS_V5.md`.)

This repository is the complete record of five pre-registered rounds, including three
that failed. Criteria files were committed before each round's runs and never edited;
`DEVIATIONS.md` lists every departure from the briefs, including two operational restarts.

## The five rounds

| round | question | brief | criteria (frozen before runs) | results | verdict |
|---|---|---|---|---|---|
| v1 | Does "bits delivered" collapse learning curves; is the loss exponent set by the data? | `briefs/v1.md` | `PASS_CRITERIA.md` | `RESULTS.md` | **H1 FAIL, H2 FAIL** — a fact needs ~50 exposures to be stored; a corpus with no heavily repeated fact stores nothing |
| v2 | Does an exposure-corrected axis (I_eff, frozen n0) collapse held-out corpora? | `briefs/v2.md` | `PASS_CRITERIA_v2.md` | `RESULTS.md` (v2 section) | **FAIL** — 16 or 32 exposures of every fact store 0 %; absorption is a sigmoid in exposures and depends on the rest of the corpus |
| v3 | The premise: does capping repetition at what the learner needs store several times more? | `briefs/v3.md` | `PASS_CRITERIA_v3.md`, `C_STAR.md` | `RESULTS_V3.md` | **P1 FAIL, P2 FAIL** — a hard cap causes total forgetting of every capped fact (a sliding window, not accumulation) |
| v4 | The premise, made stationary: flatten the distribution instead of capping | `briefs/v4.md` | `PASS_CRITERIA_v4.md`, `results/v4_predicted.csv` | `RESULTS_V4.md` | **P1 FAIL (3× bar met by the 31M model only, 5.1×), P2 FAIL**; absorption is a function of exposure *rate*; noise removal buys nothing |
| v5 | Is v4 a property of the learner or of the optimizer recipe? | `briefs/v5.md` | `CRITERIA_v5.md` | `RESULTS_V5.md` | **R1 ROBUST** under all recipes; **R2 steady state** confirmed; half-life scales with 1/lr, not 1/(lr·wd) |

Model: GPT-2-style decoder (L: 10 layers, d = 512, 31.5M non-embedding parameters; S and M
in `configs/models.yaml`), one document per row, AdamW, single pass. World seed 0
throughout; every run is determined by (corpus config, run seed) and was verified bit-exact
on rerun.

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m pytest -q                       # CPU tests (< 1 min); add -m slow for the GPU learnability test
.venv/bin/python analysis/make_figures_v4.py        # regenerates the v4 figures and results/v4_summary.json from CSVs, no GPU
```

`make_figures.py`, `make_figures_v3.py` and `make_figures_v5.py` do the same for the other
rounds. To rerun training: `scripts/run_all.sh "1 2"` launches the v1 matrix on idle GPUs;
a single run is `train/train.py --corpus configs/<name>.yaml --seed 1` (v3–v5 runs take the
extra flags recorded in `scripts/ops/queue_*.txt`).

## Layout

```
briefs/             the experiment briefs, v1–v5, as given (with the amendments each round adopted)
PASS_CRITERIA*.md   pre-registered pass criteria (v1, v2, v3, v4); CRITERIA_v5.md; C_STAR.md (v3 constants)
RESULTS*.md         results per round; DEVIATIONS.md lists every departure from the briefs
configs/            corpora (v1/v2 in configs/, v3–v5 in configs/v3/), model sizes, probe settings
synth/              world (facts, templates, vocabulary), document streams, closed forms
train/              model and trainer (constant LR + cooldown branches; token budgets; probe; schedules)
eval/               evaluators (object loss, facts stored, head guard, unseen-fact control, probe)
analysis/           fit/collapse (v1–v2), phase 0/A/premise (v3), v4 and v5 analyses, figure scripts
results/            one CSV per run under results/*runs*/, merged CSVs, fits, predictions, summaries
figures/            all figures (png + pdf)
tests/              generator, closed forms, document format, learnability, streams, probe
scripts/            run_all.sh, launch.sh; scripts/ops/ holds queue files and server helpers
artifacts/, logs/   not in git: per-run n_k / per-fact hit / exposure-time arrays and run logs
```

Object losses are in bits over the 4,096-object softmax (`DEVIATIONS.md` #1). The synthetic
world contains only arbitrary random facts: nothing here is about generalisation, reasoning
or compressible structure.

License: Apache 2.0.
