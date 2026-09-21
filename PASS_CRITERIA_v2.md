# Pass criteria v2 (held-out test of the x-axis claim)

Written after the v1 runs (PASS_CRITERIA.md, unchanged; outcome H1 FAIL / H2 FAIL in
RESULTS.md) and committed before any v2 run is launched. Everything below is frozen from
the fifteen existing v1 runs (`results/runs/*.csv`, `artifacts/*/nk_*.npz`).

## Frozen exposure-corrected axis

I_eff = 12 · Σ_k (1 − exp(−n_k / n0)), with **n0 = 9.88** (9.877276), fitted by least
squares on log(bits_stored) vs log(I_eff) over the twelve Zipf runs (Z05_n0, Z05_n64,
Z10_n0, Z10_n64 × seeds 1, 2, 3; 144 points, none dropped), EQ4 excluded.

A single n0 is used for both filler levels. Fitting per filler level gives n0 = 3.51
(n = 0) and 23.3 (n = 64), but the residual landscape is nearly flat: at n0 = 9.88 the
rms log-residual is within 4 % (n = 0) and 2 % (n = 64) of each level's own optimum.
That is not "clearly inadequate", so one n0 is frozen. No other free parameters.
n0 is not refitted after the held-out runs; any refit reported later is labelled exploratory.

## Held-out corpora

| name | fact sampling | filler n | documents | K |
|---|---|---|---|---|
| EQ16_n0  | every fact exactly 16 times, one global shuffle | 0  | 16,000,000 | 1,000,000 (full) |
| EQ16_n64 | every fact exactly 16 times, one global shuffle | 64 | 16,000,000 | 1,000,000 (full) |
| EQ32_n0  | every fact exactly 32 times, one global shuffle | 0  | 32,000,000 | 1,000,000 (full) |

Seeds 1 and 2. Same world (seed 0), model, optimizer, batch size, LR = 3e-4, cooldown
protocol and evaluators as v1. Measurement points: the v1 log-spaced points (30,000 →
10,000,000, 12 points, batch-aligned), continued with the same ratio (×1.696) while below
the corpus size, plus the corpus size. Projected wall time (v1 throughput ≈ 7,000–8,500
docs/s incl. cooldown branches): EQ16 ≈ 45–50 min, EQ32 ≈ 80–90 min, all under 3 h, so
the full K = 1,000,000 is used.

## Targets

T ∈ {100, 200, 300} facts stored. Every existing corpus with signal reaches 300 in
every seed (per-seed maxima: Z05_n0 ≥ 3,497; Z05_n64 ≥ 3,261; Z10_n0 ≥ 420; Z10_n64
≥ 368). EQ4_n16 is excluded from the choice of T: its maximum seed-mean facts_stored is
34, inside the noise of the chance correction (facts_delivered / 4096 ≈ 244, σ ≈ 16),
so it does not reach any usable target and is reported as "never reaches T".

## H2v2 (pass / fail)

For each corpus (seed-averaged curve, log-log interpolation, as in v1) find the x at
which facts_stored first reaches T, for x = training tokens and x = I_eff (n0 frozen
above). Let [L, U] be the min and max of x_Ieff over the four Zipf corpora at T = 300.

**PASS** if, at T = 300, every held-out EQ corpus that reaches T has x_Ieff within
[L / 2, 2 U]. A held-out corpus that never reaches T = 300 is a FAIL for that corpus and
is stated as such. Report, for T = 100, 200, 300, the spread (max / min) in tokens and in
I_eff over all corpora that reach T, including the EQ corpora, and name the corpora that
do not.

## H1 (descriptive, no pass / fail)

Report fitted β per corpus (same fit as v1: log loss vs log documents over points with
0.05 ≤ loss ≤ 6 bits) next to the ideal-learner β over the same document range, per seed.

## Absorption curve (descriptive)

For each corpus and measurement point, the fraction of delivered facts whose object is
the model's top-1 prediction, binned by exposure count n_k (bins 1, 2, 3–4, 5–8, 9–16,
17–32, 33–64, 65–128, 129–256, ≥ 257), chance-corrected by subtracting 1/4096. Per-fact
top-1 hits are saved at every measurement point for the v2 runs; the v1 corpora are
re-run with the same seeds to obtain them (v1 CSVs stay the record; reruns are compared
to them as a determinism check).
