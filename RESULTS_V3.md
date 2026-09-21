# Results v3: testing the premise (Experiment Brief v3)

Pre-registered in `PASS_CRITERIA_v3.md` (commit `9a9813c`, before Phase A) and `C_STAR.md`
(commit `93aadca`, after Phase A, before Phase P). All numbers regenerate from CSVs with
`python analysis/make_figures_v3.py`. Runs: 6 LR-sweep, 14 Phase A, 18 Phase P; all
completed, none crashed, none rerun. Unseen-fact control at chance everywhere (Phase P
means 2.3e-4 – 2.5e-4, max 5.5e-4; Phase A mean 2.4e-4; chance 2.44e-4).

**Verdicts: P1 FAIL, P2 FAIL. Every capped run is cap-limited** (cap adequacy 0.003–0.15,
criterion 0.7). The reason is not the value of c* but what capping does: **a fact that
stops being repeated is forgotten.** In every CAPPED and CURATED run, facts that reached
c* exposures before the last measurement interval are stored at 0.0 % at the end of
training; the models' in-distribution loss on the corpus head rises from ~4 bits early in
training to 12.2–13.0 bits (worse than uniform) by the end. The "excess repetition" that
Phase 0 counts as waste is the rehearsal that keeps RAW's head stored.

## Phase 0: token-waste accounting (v1 Zipf runs, 3 seeds each)

Absorption curve from per-fact top-1 hits at 10M documents (`results/absorption.csv`):

| corpus | c50 (exposures) | c90 | unabsorbed | excess (> c90 on stored facts) | useful | template+filler tokens | top-1 fact share of docs | top-100 share |
|---|---|---|---|---|---|---|---|---|
| Z05_n0  | 43.0 (41–44) | 93.1 (90–99)  | 1.5 % | **96.8 %** | 1.7 % | 50.2 % | 38.3 % | 92.4 % |
| Z05_n64 | 48.3 (44–50) | 99.8 (91–110) | 1.6 % | 96.7 % | 1.8 % | 92.1 % | 38.3 % | 92.4 % |
| Z10_n0  | 42.9 (42–45) | 79.8 (79–81)  | 0.2 % | **99.6 %** | 0.3 % | 50.2 % | 60.8 % | 99.4 % |
| Z10_n64 | 56.0 (53–60) | 85.4 (84–87)  | 0.2 % | 99.5 % | 0.3 % | 92.1 % | 60.8 % | 99.4 % |

(`results/v3_waste.csv`, `figures/v3_fig_waste.png`; per-seed c50/c90 in parentheses.)

## Phase A: dense-uniform calibration (14 runs; `results/v3_phaseA_summary.csv`)

LR sweep (`results/v3_lr_sweep.csv`, U30000, 1M docs, seed 1): S 3e-3 (25,901 stored; 1e-3
25,700; 3e-4 22,708), M 3e-4 (29,967; 1e-3 29,735; 3e-3 11,491). L keeps 3e-4.

| model | non-emb params | K_sub | plateau (docs; steps) | c50 | c90 | c90 after plateau | stored at end | bits/param at end |
|---|---|---|---|---|---|---|---|---|
| S | 2,045,568 | 10,000 | 128,000; 1,000 (both) | 18.6 / 16.8 | 25.6 / 24.0 | 12.6 / 10.6 | 100 % | 0.059 |
| S | | 30,000 | 512,000; 4,000 (both) | 29.9 / 27.8 | 39.0 / 38.7 | 21.4 / 21.1 | 99.9 % | 0.176 |
| M | 7,993,152 | 10,000 | 128,000; 1,000 | 18.5 / 19.0 | 27.2 / 30.0 | 14.0 / 16.5 | 100 % | 0.015 |
| M | | 30,000 | 128,000; 1,000 | 16.7 / 17.1 | 22.7 / 22.9 | 18.3 / 18.5 | 100 % | 0.045 |
| L | 31,524,864 | 10,000 | 128,000; 1,000 | 12.2 / 11.5 | 20.7 / 19.3 | 6.3 / 4.9 | 100 % | 0.004 |
| L | | 30,000 | 128,000; 1,000 | 14.0 / 12.9 | 21.4 / 20.9 | 17.0 / 16.5 | 100 % | 0.011 |
| L | | 100,000 | 3,584,000 / 4,352,000; 28,000 / 34,000 | 45.3 / 45.4 | 56.0 / 56.3 | 5.4 / 4.4 | 100 % | 0.038 |

(seed 1 / seed 2.) Every run left the plateau. **c\* = 40 (S), 30 (M), 30 (L); W = 1,000,000
(S; the cap binds, 2 × 512,000 = 1,050,000), 300,000 (M), 300,000 (L)** — `C_STAR.md`.
Note that dense-uniform c90 (21–39) is 3–4× below the Zipf c90 of Phase 0 (80–100).

## Phase P: the premise test (18 runs, B = 300M tokens; `results/v3_premise.csv`)

Facts stored at the final point (seed 1 / seed 2 / mean):

| model | RAW | CAPPED | CURATED |
|---|---|---|---|
| S | 17,559 / 16,955 / **17,257** | 11,501 / 10,392 / **10,947** | 10,567 / 8,823 / **9,695** |
| M | 45,370 / 46,014 / **45,692** | 76,140 / 71,785 / **73,962** | 118,697 / 121,936 / **120,316** |
| L | 68,026 / 67,942 / **67,984** | 258,018 / 251,683 / **254,850** | 413,102 / 412,689 / **412,895** |

Evaluability: RAW ≥ 100 facts for S, M and L — all three evaluable.

- **P1 (CURATED ≥ 3× RAW for every evaluable size): FAIL.** CURATED / RAW = 0.56 (S),
  2.63 (M), 6.07 (L). Holds for L only; S is *worse* on CURATED than on RAW.
- **P2 (S on CURATED ≥ L on RAW): FAIL.** 9,695 vs 67,984 (0.14×).
- **Cap adequacy (fraction stored among facts that reached c\*):** L CAPPED 0.150 / 0.146,
  L CURATED 0.129 / 0.128, M CAPPED 0.062 / 0.057, M CURATED 0.042 / 0.043, S CAPPED
  0.013 / 0.013, S CURATED 0.004 / 0.003. **All twelve capped runs are cap-limited** (< 0.7).
  For comparison, RAW facts with ≥ c\* exposures are stored at 0.98 / 0.97 (L), 0.92 / 0.91
  (M), 0.62 / 0.61 (S). Per the pre-registered rule, the under-storing of the capped
  corpora is attributed to the cap, not to the premise.
- No run failed to leave the plateau (every run's in-distribution loss fell below 11.5 bits
  at an early measurement point).

Descriptive:

| model | CAPPED / RAW | CURATED / CAPPED | CURATED / RAW measured | CURATED / RAW predicted (facts reaching c\*) |
|---|---|---|---|---|
| S | 0.63 | 0.89 | 0.56 | 13.0 |
| M | 1.62 | 1.63 | 2.63 | 17.6 |
| L | 3.75 | 1.62 | 6.07 | 17.6 |

Predicted vs actual facts reaching c\* (`results/v3_predicted_storable.csv`, computed before
launch; actual seed 1 / seed 2): RAW 26,125 → 26,040 / 26,061 (S), 32,016 → 31,962 / 31,974
(M, L); CAPPED 111,792 → 111,483 / 111,534 (S), 165,189 → 164,860 / 164,713 (M, L); CURATED
339,753 → 337,761 / 337,591 (S), 564,328 → 559,685 / 559,467 (M, L). The stream did what the
prediction said; the model did not store what the stream delivered.

Headroom (bits stored per non-embedding parameter; total-parameter value in parentheses):

| model | RAW | CAPPED | CURATED | reference |
|---|---|---|---|---|
| S | 0.101 (0.070) | 0.064 (0.044) | 0.057 (0.039) | 2.0 (Allen-Zhu & Li 2024), 3.6 (Morris et al. 2025) |
| M | 0.069 (0.057) | 0.111 (0.093) | 0.181 (0.151) | |
| L | 0.026 (0.024) | 0.097 (0.089) | 0.157 (0.144) | |

Every value is 10–80× below the 2.0 reference. S does not flatten at a capacity ceiling
under CURATED; it peaks at 18,433 facts (26M tokens) and then *declines* to 9,695 while
in-distribution loss climbs from 3.8 to 12.3 bits. That is forgetting, not saturation.

## Why the capped corpora fail: forgetting (exploratory diagnostic, not pre-registered)

`results/v3_forgetting_EXPLORATORY.csv`, `figures/v3_fig_forgetting_EXPLORATORY.png`. For each
capped run, facts are grouped by the measurement interval in which they reached c\* (and
were therefore never drawn again); the table gives the fraction of each group stored at the
*end* of training (seed 1; seed 2 within 0.01).

| reached c\* by token count → | ≤ 1M | 2.3M | 5.1M | 11.5M | 26M | 59M | 133M | 300M (last interval) | still below c\* at end |
|---|---|---|---|---|---|---|---|---|---|
| L CURATED | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.185 | **0.775** |
| L CAPPED  | 0.004 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.242 | 0.290 |
| M CURATED | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.061 | 0.217 |
| S CURATED | 0.000 | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.006 | 0.014 |

Every fact whose repetitions were cut off more than one measurement interval before the end
is gone. What the capped models "store" at the end is the set of facts they are currently
being shown (the sliding window of uncapped facts), and a fraction of those capped in the
last interval. The window is wider for larger models, which is why CURATED / RAW grows with
size (0.56 → 2.6 → 6.1): L's window at 300M tokens holds ~413K facts; RAW-L holds 68K facts
that it keeps rehearsing. In-distribution loss on the base head at the end: RAW 1.42 / 1.78 /
2.77 bits (L / M / S), CAPPED 12.4 / 12.3 / 12.3, CURATED 13.0 / 12.4 / 12.2 — the capped
models predict the head's objects *confidently wrong*.

Two things this diagnostic cannot separate: whether a larger c\* would only delay the
forgetting (the Phase 0 facts stored at 65–128 exposures were sampled i.i.d. throughout
training, so their last exposure was always recent — there is no evidence anywhere in v1–v3
of a fact surviving ~10⁸ tokens without rehearsal), and whether spaced rehearsal at a low
rate (rather than a hard cap) would retain facts at lower token cost than RAW's repetition.
That is the experiment the premise actually needs.

## What this does not show

Arbitrary random 12-bit facts only (no generalisation, no reasoning, no shared structure
between facts); toy scale (2M–31.5M non-embedding parameters, 300M tokens, single pass);
two seeds, no confidence intervals; one learning rate per size; one cap rule (hard cut at
c\*, no rehearsal schedule); the shifted-Zipf "web-like" corpus is a stand-in, not web data.

## Deviations

See `DEVIATIONS.md` entries 13–19 (v3). Not run: the optional §8 diagnostic (no
confirmation wait occurred, so the `--save-final` rerun it needed was not made).
