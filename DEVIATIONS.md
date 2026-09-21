# Deviations from the brief

Each entry: what changed, why, and when it was decided relative to the affected runs.
None of these changed after the real runs were launched.

1. **Object loss softmax restricted to the 4,096 object tokens** (decided before the smoke run).
   The brief asks for losses "on the object token only". We compute the NLL of the true
   object under the softmax over the object-token range, so an unseen fact costs exactly
   12 bits under a uniform guess and top-1 chance is exactly 1/4096 — matching the chance
   term `facts_delivered / 4096` and the ideal-learner closed form. The full-vocabulary
   NLL is logged as an extra column (`obj_loss_bits_indist_fullvocab`); it differs from the
   restricted one by < 0.02 bits after the first measurement point.
2. **Measurement points rounded to multiples of the batch size (128)**, e.g. 29,952 instead
   of 30,000, so that snapshot/branch boundaries fall on batch boundaries and the branch
   consumes exactly the documents the main run would. 10,000,000 and 4,000,000 are
   multiples of 128 and are unchanged.
3. **The main run's last 10 % (0.9 D_last to D_last) at constant LR is never trained**: only
   the cooldown branch covers it, and the branch is what gets evaluated. Saves time,
   changes nothing that is measured.
4. **Empty control set at the last EQ4 point.** At D = 4,000,000 every one of the 1,000,000
   facts has been delivered, so the "20,000 facts with n_k = 0" control set is empty;
   `unseen_top1_acc` and `bits_stored_soft` are NaN there. At D = 3,477,760 only 294
   unseen facts remain, so that point's control is noisy (one hit = 3.4e-3).
5. **Results are written per run** (`results/runs/<corpus>_seed<seed>.csv`) to avoid
   concurrent appends; `analysis/make_figures.py` (and `load_runs`) concatenates them
   into `results/runs.csv`. Four extra columns are appended after the brief's schema:
   `obj_loss_bits_indist_fullvocab, unseen_loss_bits, delivered_top1_acc, cooldown`.
6. **`torch.compile` not used.** It worked on the first try but steady-state throughput
   was identical to eager (~8,500 docs/s; the step is launch-bound at these sequence
   lengths), so eager is used for simplicity. `--compile` remains available.
7. **"Fit to the ideal-learner closed form over the same range"** is implemented as the
   fit over the same *document* points that the model fit uses (the 0.05–6 bit window
   applied to the model loss). For a = 1.0 the ideal loss is below 0.05 bits at every
   point, so applying the loss window to the ideal curve itself would give no fit.
8. **Seed 3 was run for all corpora** (the brief allows this when GPUs are free). It is
   included in every seed average; per-seed values are reported.
9. **Diagnostic n0 fit excluding EQ4** (secondary analysis only). The pre-registered
   single-n0 fit across all corpora is reported as specified; because EQ4 stores ~0 bits
   at every point, no n0 makes it collapse and the fit runs to the lower bound. A second
   fit excluding EQ4 is reported alongside and labelled as a diagnostic.
10. **LR sweep runs used the full cooldown-branch protocol** to 1,000,000 documents (the
    brief does not say whether the sweep should use cooldown; using it makes the
    selection criterion match the real measurement).

## v2 notes (PASS_CRITERIA_v2.md)

11. **Prediction recorded 2026-09-21 01:45 UTC, before any v2 result was available.** With
    n0 = 9.88 frozen, I_eff credits one exposure with 1 − e^(−1/9.88) = 9.6 % of a fact. At
    T = 300 the Zipf corpora reach 300 stored facts at I_eff ≈ 6.5K–8.9K bits (band
    [3.2K, 17.7K]) because they have delivered only ~1,000 distinct facts by then. An EQ
    corpus has ~30,000 once-seen facts at its first measurement point (I_eff ≈ 35K) and
    stores nothing until exposures reach several per fact (I_eff in the millions). So
    H2v2 cannot pass as written, whatever the model does; the criterion is kept and the
    outcome reported. The absorption curve is expected to show near-zero storage at 1–4
    exposures, i.e. a threshold-like curve rather than the concave exponential assumed by
    I_eff. Any alternative exposure weighting fitted from it is exploratory.
12. **Reruns of the v1 corpora** (`--tag rerun`, same seeds) exist only to obtain per-fact
    hits for the absorption curve. v1 CSVs remain the record; `load_runs` ignores tagged
    CSVs, and `analysis/collapse.determinism_check` reports v1-vs-rerun differences.
