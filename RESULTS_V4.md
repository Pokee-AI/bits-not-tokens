# Results v4: the premise test, made stationary (Experiment Brief v4.1)

Pre-registered in `PASS_CRITERIA_v4.md` (commit `312fc5a`) and `results/v4_predicted.csv`
(commit `0dc61b9`), both before any v4 run. Everything regenerates from CSVs with
`python analysis/make_figures_v4.py`. Runs: 54 v4 (2 corpus types × 4 levels × 3 sizes × 2
seeds, plus CURATED-x30 × 3 sizes × 2 seeds) and 6 RAW reruns (for the two new columns);
all completed, none crashed. Unseen-fact control at chance everywhere (means 2.3e-4 –
2.6e-4, chance 2.44e-4; max 1.7e-3 on a CURATED-x30 point with a small residual control set).
Wall time: FLAT/RAW ≈ 15 min, CURATED ≈ 36 min per run.

**Verdicts: P1 FAIL, P2 FAIL, P3 passes at every selected level.** Flattening the raw
distribution helps every size (up to 5.1× facts stored for L at equal tokens, with the
head intact), but the pre-registered 3× bar is met by L only (S 2.27×, M 2.77× on seed 2).
The mechanism is unambiguous: **absorption depends on a fact's exposure *rate*, not its
exposure count.** FLAT tracks the count-only prediction to within 12 % at every level;
CURATED gets 2.3× more exposures per fact from the same tokens and stores no more than
FLAT, because a stationary stream reaches a steady state after ~10M documents and further
exposures at the same rate add nothing.

## Determinism check

RAW-S seed 1 rerun with `--stop-after-points 2`: both rows identical to v3 in every column.
The six RAW reruns (for `weighted_acc_p` / `head_loss_bits`) are asserted equal to the v3
rows in docs, tokens, facts delivered, facts stored, in-distribution loss and control
accuracy before use (`analysis/v4_analysis.load`).

## Levels (`results/v4_levels.csv`)

τ is defined on FLAT's stationary document count and shared by CURATED (so CURATED-c is the
same distribution with 2.3× the documents). L and M share τ (same W = 300,000); S differs
slightly (W = 1,000,000).

| level | τ (L, M) | Z | facts at cap | exposures of a capped fact: FLAT / CURATED (incl. warm-up) |
|---|---|---|---|---|
| 1000 | 6.52e-5 | 0.680 | 2,971 | 1,052 / 2,424 |
| 300 | 1.12e-5 | 0.388 | 11,878 | 319 / 730 |
| 100 | 2.01e-6 | 0.209 | 39,473 | 107 / 244 |
| 30 | 2.60e-7 | 0.090 | 156,827 | 32 / 73 |
| x30 (CURATED only) | 4.15e-8 | 0.034 | 535,963 | – / 31 |

N = 10,714,285 documents (FLAT, RAW) or 25,000,000 (CURATED) at B = 300M tokens.

## Level selection (seed 1) and verdicts (seed 2)

Levels passing the P3 guard on seed 1 (`weighted_acc_p` ≥ RAW − 0.02): S: 1000, 300 (both
types); M: 1000, 300, 100 (both types); L: 1000, 300, 100, 30 (both types). x30 fails P3 at
every size (head lost: `head_loss_bits` 9.6–12.0). Selected (most facts stored on seed 1
among eligible): **S → 300, M → 100, L → 30** for both FLAT and CURATED.

| size | RAW seed 2 | CURATED (selected) seed 2 | ratio | `weighted_acc_p` RAW → CURATED | P3 | **P1** |
|---|---|---|---|---|---|---|
| S | 16,955 | 38,518 (level 300) | **2.27×** | 0.769 → 0.858 | pass | **FAIL** |
| M | 46,014 | 127,281 (level 100) | **2.77×** | 0.863 → 0.936 | pass | **FAIL** |
| L | 67,942 | 348,507 (level 30) | **5.13×** | 0.890 → 0.960 | pass | PASS |

All three sizes evaluable (RAW ≥ 100 facts). **P1: FAIL** (holds for L only). **P2: FAIL** —
S on CURATED-300 stores 38,518 vs L on RAW 67,942 (0.57×). Seed-1 values for the same
cells: S 36,916 vs 17,559 (2.10×); M 128,311 vs 45,370 (2.83×); L 353,924 vs 68,026 (5.20×).
Descriptive: **M on CURATED-100 (127,281) beats L on RAW (67,942) by 1.87×** — the 4× size
gap is bridged; the 16× gap (S vs L) is not.

One confirmation seed, so no confidence intervals. For context, v3's seed-to-seed relative
spread in facts stored was 0.1 % (L RAW / CURATED), 1.4–2.7 % (M), 3.5 % (S RAW) and up to
18 % (S CURATED, near zero); in v4 the two seeds agree to ≤ 5 % everywhere except the
collapsed S-30 / x30 cells.

## Dose-response (`figures/v4_fig_dose_response.png`, `results/v4_premise.csv`)

Facts stored at B, seed 1 / seed 2:

| size | RAW | FLAT-1000 | FLAT-300 | FLAT-100 | FLAT-30 | CUR-1000 | CUR-300 | CUR-100 | CUR-30 | CUR-x30 |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 17,559 / 16,955 | 23,163 / 22,839 | 34,443 / 35,345 | 33,111 / 32,856 | 276 / 25 | 26,106 / 26,231 | 36,916 / 38,518 | 44,186 / 45,251 | 1,564 / 1,141 | 184 / 54 |
| M | 45,370 / 46,014 | 62,329 / 61,595 | 89,760 / 92,299 | 130,540 / 129,561 | 114,253 / 116,663 | 61,101 / 60,724 | 86,118 / 84,927 | 128,311 / 127,281 | 145,634 / 136,431 | 27,029 / 27,126 |
| L | 68,026 / 67,942 | 93,558 / 96,161 | 139,192 / 138,798 | 209,127 / 206,945 | 304,725 / 288,910 | 75,919 / 76,319 | 114,216 / 113,541 | 173,238 / 166,481 | 353,924 / 348,507 | 43,063 / 117,883 |

`weighted_acc_p` (seed 1; RAW: S 0.772, M 0.863, L 0.891) rises with flattening until the
cliff: S 0.807 / 0.842 / **0.531** / 0.001 (FLAT-1000/300/100/30); M 0.890 / 0.916 / 0.935 /
**0.491**; L 0.913 / 0.936 / 0.955 / 0.916; CURATED-x30: 0.000 / 0.038 / 0.063. `head_loss_bits`
at the cliff: S-100 4.5–5.8, M-30 6.6–7.0, L-x30 9.6–10.1 (head gone); L-30 0.7 (head kept).
The cliff sits where the capped facts' exposure rate drops below what the model can hold:
about 100 exposures per 10.7M documents for S, 30 for M, and below 30 for L.

Attribution (seed mean):

| size | level | FLAT / RAW | CURATED / FLAT | CURATED / RAW |
|---|---|---|---|---|
| S | 1000 / 300 / 100 / 30 | 1.33 / 2.02 / 1.91 / 0.01 | 1.14 / 1.08 / 1.36 / — | 1.52 / 2.19 / 2.59 / 0.08 |
| M | 1000 / 300 / 100 / 30 | 1.36 / 1.99 / 2.85 / 2.53 | 0.98 / 0.94 / 0.98 / 1.22 | 1.33 / 1.87 / 2.80 / 3.09 |
| L | 1000 / 300 / 100 / 30 | 1.40 / 2.04 / 3.06 / 4.37 | 0.80 / 0.82 / 0.82 / 1.18 | 1.12 / 1.68 / 2.50 / 5.17 |

Flattening alone accounts for essentially the whole gain. Removing the 16 filler tokens —
which buys 2.3× the documents and 2.3× the exposures per fact from the same token budget —
adds nothing for M and *costs* ~20 % for L at levels 1000–100.

## Predicted versus actual (`figures/v4_fig_pred_vs_actual.png`)

Actual / predicted facts stored (seed 1, seed 2):

| size | RAW | FLAT-1000 | FLAT-300 | FLAT-100 | FLAT-30 | CUR-1000 | CUR-300 | CUR-100 | CUR-30 | CUR-x30 |
|---|---|---|---|---|---|---|---|---|---|---|
| L | 1.00, 1.00 | 1.07, 1.10 | 1.11, 1.10 | 1.12, 1.11 | 1.00, 0.94 | 0.50, 0.50 | 0.52, 0.52 | 0.54, 0.51 | 0.67, 0.66 | 0.06, 0.15 |
| M | 0.99, 1.01 | 1.06, 1.04 | 1.05, 1.08 | 1.02, 1.02 | 0.59, 0.61 | 0.58, 0.58 | 0.57, 0.56 | 0.57, 0.56 | 0.38, 0.36 | 0.05, 0.05 |
| S | 1.02, 0.98 | 1.06, 1.05 | 1.11, 1.14 | 0.85, 0.84 | 0.02, 0.00 | 0.66, 0.66 | 0.65, 0.68 | 0.52, 0.54 | 0.02, 0.01 | 0.00, 0.00 |

FLAT lands on the count-only prediction (within 12 % at every level where the head is
kept), so under stationary sampling at a fixed run length the absorption curve transfers
across distributions: the rest of the corpus does not matter. CURATED lands at half the
prediction at every level. The reason is visible at matched *documents* rather than
matched tokens: at ~11M documents CURATED-L-100 has 200,013 facts and FLAT-L-100 209,127;
CURATED-M-100 126,271 vs FLAT-M-100 130,540; CURATED-L-30 303,066 vs FLAT-L-30 304,725.
CURATED then trains for another 14M documents and gains nothing (L-100 ends at 173,238,
M-100 at 128,311). Its endpoint absorption curve is shifted right — L-100 facts with 9–16
exposures are stored 6 % of the time in CURATED vs 53 % in FLAT, 17–32 exposures 26 % vs
91 % — because a fact with 16 exposures over 25M documents has 2.3× the spacing of one with
16 over 10.7M. The count-only assumption fails in exactly one direction: what a fact needs
is a *rate* of exposures (per ~10⁷ documents), and once a stationary stream has run for a
few of the learner's memory horizons, the stored set is the set of facts above that rate
and stops growing. Predictions made from the FLAT absorption curve applied to exposure
rate would put CURATED-c on top of FLAT-c, which is where it is.

## Bits stored per non-embedding parameter (total-parameter value ≈ 0.7–0.9× these)

| size | RAW | best FLAT | best CURATED | reference |
|---|---|---|---|---|
| S | 0.10 | 0.21 (300) | 0.26 (100, head guard fails) / 0.22 (300) | 2.0 (Allen-Zhu & Li 2024), 3.6 (Morris et al. 2025) |
| M | 0.07 | 0.20 (100) | 0.21 (30, head guard fails) / 0.19 (100) | |
| L | 0.03 | 0.11 (30) | 0.13 (30) | |

Every value is 8–70× below the 2.0 reference; the models are far from a capacity ceiling.
S does not saturate — it collapses (level 30) rather than flattening.

## What v4 settles

1. Flattening a web-like distribution helps at every size, monotonically until a
   size-dependent cliff: 1.3–2.0× at c = 1000–300 for all sizes, up to 5.1× for L at c = 30,
   with the head retained (P3 passes). The pre-registered 3× bar is met only by L.
2. Removing noise tokens buys documents but not facts: at the same τ, CURATED equals FLAT
   at matched documents and does not improve with 2.3× more of them.
3. Absorption is a function of exposure rate, not count. Under stationary sampling the
   count-only absorption curve transfers exactly across distributions of the same run
   length (FLAT), and fails by a factor equal to the run-length ratio when the run is
   longer (CURATED).
4. A small model on flattened data can match a 4× larger model on raw data (M-CURATED-100
   1.87× L-RAW), but not a 16× larger one (S-CURATED 0.57× L-RAW): the small model's
   exposure-rate threshold (≈ 100 exposures per 10.7M documents) is higher than the
   flattening that would be needed.

## What this does not show

Arbitrary random 12-bit facts only, so nothing about generalisation or reasoning; toy scale
(2M–31.5M non-embedding parameters, 300M tokens, single pass); one confirmation seed, no
confidence intervals; one learning rate per size; one warm-up rule; the shifted-Zipf
"web-like" corpus is a stand-in. The premise concerns compressible structure, which this
world does not contain. The exposure-rate result identifies a memory horizon of order 10⁷
documents for this learner but does not identify its cause (AdamW's decoupled decay has a
timescale of 1/(lr·wd) = 33K steps ≈ 4.3M documents here; interference from concurrently
learned facts is the other candidate); a weight-decay ablation would separate them.

## Deviations

`DEVIATIONS.md` entries 22–30. Two queue restarts (28: evaluator in-distribution set for
flat corpora; 30: results directory deleted by the code-sync script) cost ~1.5 GPU-hours
and changed no training; the final queue ran once under the committed code with no reruns.
