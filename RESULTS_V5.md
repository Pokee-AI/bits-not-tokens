# Results v5: is the v4 result robust? (Experiment Brief v5.1)

Pre-registered in `CRITERIA_v5.md` (commit `a8cca50`, before any v5 run). Regenerate with
`python analysis/make_figures_v5.py`. 87 runs (A 44, B 27, C 8, D 8), all completed, none
crashed, none rerun, every run left the plateau. Every v5 run carries the retention probe
(2,000 facts × 100 exposures in documents 3.2M–4.27M, replacing ordinary draws; probe facts
excluded from `facts_stored`). Unseen-fact control at chance. Wall time ≈ 3.3 h on 8 GPUs.

**Short answer to "learner or recipe?": the v4 finding is a property of the learner.** The
FLAT-sel / RAW advantage survives every optimizer setting (R1 ROBUST), a stationary stream
is at steady state by B (R2), and forgetting is finite without weight decay and scales with
the learning rate, not with 1/(lr·wd) (R3). What the recipe *does* change is the absolute
level: every alternative to v4's constant-LR / wd 0.1 recipe stores more facts (up to 1.7×
for L under a cosine schedule), while the relative gain from flattening moves by less than
15 %.

## R1 — the effect survives the recipe: ROBUST

FLAT-sel / RAW at the final point (seeds 1, 2 mean; per-seed in `results/v5_summary.json`).
Criterion: ratio ≥ 1.5 and FLAT's `weighted_acc_p` ≥ RAW's − 0.02.

| size | setting | RAW | FLAT-sel | ratio | weighted acc RAW → FLAT | passes |
|---|---|---|---|---|---|---|
| M | base | 45,400 | 129,577 | **2.85** | 0.862 → 0.935 | yes |
| M | lr3 | 61,619 | 154,474 | **2.51** | 0.882 → 0.945 | yes |
| M | wd0 | 52,572 | 149,174 | **2.84** | 0.875 → 0.943 | yes |
| M | wd0_lr3 | 67,235 | 164,819 | **2.45** | 0.889 → 0.948 | yes |
| M | cosine | 66,818 | 170,837 | **2.56** | 0.892 → 0.950 | yes |
| S | base / wd0 / cosine | 17,419 / 17,892 / 22,143 | 35,540 / 35,546 / 41,945 | 2.04 / 1.99 / 1.89 | all up | yes |
| L | base / wd0 / cosine | 67,413 / 82,510 / 115,164 | 300,275 / 371,643 / 473,229 | 4.45 / 4.50 / 4.11 | all up | yes |

**ROBUST for M (5/5), and 3/3 for S and L.** The ratio is least under the recipes that
retain the most (M: 2.45 at wd0_lr3 vs 2.85 at base), because better retention helps RAW's
mid-tail slightly more than FLAT's; it never approaches 1.5. Seed-to-seed spread of the
ratio ≤ 4 % in every cell.

## R2 — steady state: CONFIRMED

M, base, seeds 1–2 mean: FLAT-sel stores 129,577 at B, 135,899 at 2B, **126,856 at 4B**
(4B / B = 0.98, within 15 %; growth exponent in tokens −0.015). RAW: 45,400 / 46,422 /
44,353. Doubling and quadrupling the run changes nothing (`figures/v5_fig_steady_state.png`).

## R3 — what sets the horizon (descriptive)

Probe half-life on the main-run weights: documents after the window closes until probe
accuracy falls to half its window-end value (window-end accuracy 1.00 for M and L, 0.85–0.96
for S). Mean over seeds (range); weight-decay timescale 128/(lr·wd) in documents alongside.

| size | setting | lr, wd | half-life RAW | half-life FLAT-sel | 128/(lr·wd) |
|---|---|---|---|---|---|
| M | base | 3e-4, 0.1 | 0.33M (0.32–0.34) | 0.36M (0.36–0.37) | 4.27M |
| M | wd0 | 3e-4, 0 | 0.41M | 0.52M (0.50–0.54) | ∞ |
| M | lr3 | 1e-4, 0.1 | 1.22M | 1.80M (1.78–1.82) | 12.8M |
| M | wd0_lr3 | 1e-4, 0 | 1.42M | 2.01M | ∞ |
| M | cosine | 3e-4 peak, 0.1 | 0.67M | 0.84M | 4.27M (at peak) |
| L | base | 3e-4, 0.1 | 0.35M (0.34–0.35) | 0.49M (0.25–0.80) | 4.27M |
| L | wd0 | 3e-4, 0 | 0.45M | 1.23M (1.19–1.26) | ∞ |
| L | cosine | | 0.77M | 3.74M | 4.27M |
| S | base | 3e-3, 0.1 | 0.25M | 0.25M | 0.43M |
| S | wd0 | 3e-3, 0 | 0.26M | 0.26M | ∞ |
| S | cosine | | 0.23M | 0.23M | 0.43M |

Plainly: **the half-life scales with 1/lr, not with 1/(lr·wd).** Dividing the learning rate
by 3 lengthens it 3.7× (RAW) to 4.9× (FLAT) for M; removing weight decay entirely lengthens
it only 1.3–1.4× for M, 1.3–2.5× for L and 1.03× for S, and it stays finite — so forgetting
is dominated by interference from ongoing updates, with decoupled decay a minor
contributor (largest for L). Under cosine the half-life is longer because the learning rate
is already falling during the retention period. In absolute terms the unrehearsed
half-life (0.25–0.5M documents at base) is an order of magnitude shorter than the ~10⁷-
document steady-state horizon inferred in v4; the two are different quantities — v4's
horizon concerns facts that keep being rehearsed at a low rate, the probe measures a fact
that is never seen again — and v4's inference stands as a rate threshold, not a decay time.
CURATED-S's half-life (63K documents, 4× shorter than FLAT-S) is the one anomaly;
CURATED-M and -L match their FLAT counterparts.

## R4 — intervals (descriptive)

Ratios at the v4-selected levels, base recipe. Seeds 2–5 for FLAT (4 seeds, df = 3); CURATED
has v5 seeds 3–5 only (v4's seed 2 lacks the probe and is not mixed in), so df = 2.

| size | FLAT-sel / RAW: seed 1 | seeds 2–5 mean [95 % t-CI] | CURATED-sel / RAW: seeds 3–5 mean [95 % CI] |
|---|---|---|---|
| S (level 300) | 2.08 | **1.98 [1.82, 2.15]** | **2.18 [1.89, 2.47]** |
| M (level 100) | 2.86 | **2.84 [2.80, 2.89]** | **2.82 [2.73, 2.91]** |
| L (level 30) | 4.42 | **4.47 [4.39, 4.54]** | **5.10 [5.06, 5.15]** |

Per-seed ratios are in `results/v5_summary.json`. Seed 1 sits inside every interval, so the
v4 level selection did not inflate the reported ratios. The probe perturbation moved
`facts_stored` by −3 % to +7 % relative to the probe-free v4 runs of the same cells (median
|Δ| ≈ 1 %; `probe_perturbation` in the summary), identical across arms.

## R5 — L's optimum (descriptive)

Levels 15 and "8" (infeasible; run as fully uniform, c = 10.4 — DEVIATIONS.md #31) added to
v4's 1000 / 300 / 100 / 30 (v4 runs have no probe; v5's base RAW and FLAT-30 reproduce them
within 3.5 %). Seed 1 / seed 2:

| level | FLAT facts stored | FLAT weighted acc | CURATED facts stored | CURATED weighted acc |
|---|---|---|---|---|
| 30 (v4 / v5) | 304,725 / 288,910 (v5: 301,536 / 299,015) | 0.916 / 0.900 | 353,924 / 348,507 | 0.965 / 0.960 |
| 15 | 141,780 / 13,884 | 0.253 / 0.028 | 251,337 / 242,364 | 0.429 / 0.419 |
| uniform (10.4) | 4,954 / 1,513 | 0.006 / 0.001 | 112,101 / 46,561 | 0.106 / 0.046 |

RAW `weighted_acc_p` is 0.89, so levels 15 and uniform fail the head guard by a wide margin
(`head_loss_bits` 8–11: the head is gone) and FLAT-15 is unstable across seeds. **The best
P3-eligible level for L is 30 in both seeds and both corpus types**; L's cliff lies between
30 and 15 exposures per 10.7M documents.

## Cosine versus base: when a fact's exposures happened

Fraction stored among facts with 17–64 exposures (probe facts excluded), split by the
midpoint of their first and last exposure (`first_seen` / `last_seen`): first third of the
run versus last third. Seed mean.

| size | corpus | base: early third → last third | cosine: early third → last third |
|---|---|---|---|
| L | RAW | 0.00 → 0.93 | 0.75 → 1.00 |
| L | FLAT-30 | 0.00 → 0.65 | 0.93 → 1.00 |
| M | RAW | 0.00 → 0.60 | 0.00 → 1.00 |
| M | FLAT-100 | 0.00 → 0.75 | 0.00 → 1.00 |
| S | RAW | 0.00 → 0.12 | 0.00 → 0.50 |
| S | FLAT-300 | 0.00 → 0.15 | 0.00 → 0.19 |

Under constant LR, a fact whose 17–64 exposures fell in the first third of the run is gone
by the end at every size; what the model holds at the end is what it saw recently. A cosine
schedule lets L keep 75–93 % of such early facts (the learning rate is low by the time they
would be overwritten) but does nothing for M or S, whose interference is too strong at any
LR the schedule passes through. This is why cosine raises L-RAW from 67K to 115K facts.

## Is the v4 memory horizon a property of the learner or of the recipe?

Of the learner, with the recipe setting its scale. Every recipe forgets an unrehearsed fact
within a few hundred thousand to a few million documents; removing weight decay does not
remove forgetting (interference does the work), and the horizon stretches in proportion to
1/lr. Because the horizon is finite under every recipe, the steady-state picture of v4 holds
under every recipe: R2 shows no growth from B to 4B, and R1 shows the flattening gain at
every setting. What changes with the recipe is how many facts sit above the rate threshold —
lower or decaying learning rates lengthen the horizon and store 1.2–1.7× more in absolute
terms, for RAW and FLAT alike — so v4's absolute numbers were at the low end of what this
learner can do, and its ratios were, if anything, slightly high (base gives the largest
FLAT / RAW ratio of the five recipes).

## Deviations

`DEVIATIONS.md` entries 31–34. Notable: FLAT-8 infeasible (run as uniform, c = 10.4); cosine
computed in token space; main-weights retention curve ends at 0.9 B.

## What this does not show

This world contains only arbitrary random facts: nothing about generalisation, reasoning or
compressible structure. Toy scale (2M–31.5M non-embedding parameters, single pass, ≤ 1.2B
tokens); intervals from four (FLAT) or three (CURATED) seeds; one LR per size with two
ablated values; one probe design (100 exposures in a fixed window). The half-life is for a
100-exposure burst with no rehearsal and will differ for other exposure histories.
