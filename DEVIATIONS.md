# Deviations from the briefs

Each entry: what changed, why, and when it was decided relative to the affected runs.
Entries are numbered as they were recorded across all design rounds; entries 2–21 concern
the archived rounds and live on branch `archive/rounds-1-3`. The entries below are unchanged.

## Conventions inherited from earlier rounds

1. **Object loss softmax restricted to the 4,096 object tokens** (decided before the smoke run).
   The brief asks for losses "on the object token only". We compute the NLL of the true
   object under the softmax over the object-token range, so an unseen fact costs exactly
   12 bits under a uniform guess and top-1 chance is exactly 1/4096 — matching the chance
   term `facts_delivered / 4096` and the ideal-learner closed form. The full-vocabulary
   NLL is logged as an extra column (`obj_loss_bits_indist_fullvocab`); it differs from the
   restricted one by < 0.02 bits after the first measurement point.

## v4 notes (Experiment Brief v4.1; PASS_CRITERIA_v4.md)

22. **`--stop-after-points N`** trainer flag, used only for the RAW-S determinism check (the
    8 token points are defined by B, so a smaller budget would not reproduce the rows).
23. **Unit test on empirical capped exposures** compares against the corpus's own
    `N_stat · τ / Z` plus expected warm-up exposures (for CURATED-c with the shared τ that is
    ≈ 2.3c, not c), within 2 %.
24. **CURATED-x30 is placed on the dose-response x-axis at its FLAT-equivalent level**
    (30 / 2.33 ≈ 13, i.e. the c that FLAT would need for the same τ) with a distinct marker.
25. **`weighted_acc_p`** is top-1 accuracy on the existing 200,000-document in-distribution
    set (sampled from p with the fixed eval seed, the same set for every corpus);
    **`head_loss_bits`** adds the 1,000 most popular facts (fixed template per fact) to the
    evaluation. No other evaluator change. The v4 CURATED config is named `CURATEDF` in
    `configs/v3/` to avoid clashing with v3's capped `CURATED`; results label it CURATED.
26. **Warm-up boundary is rounded up to a batch** (a batch that starts inside the warm-up is
    drawn entirely from p), as in v3.
27. **The absorption curve for predictions pools both v3 RAW seeds** (the brief says seed
    mean; pooling per-fact hits is the same estimator with fewer empty bins).
28. **Evaluator bug caught 25 % into the v4 queue and the queue restarted.** The in-distribution
    eval set was drawn from p only for `zipf` / `shifted_zipf` / `capped` corpora; for the new
    `flat` corpora it fell through to uniform-over-all-facts, so `obj_loss_bits_indist` and
    `weighted_acc_p` (the P3 guard) were mis-measured on the first 12 v4 runs (`facts_stored`,
    `head_loss_bits`, hits and the control were unaffected; training is independent of the
    eval set). Fixed (`eval/evaluate.py`), all running v4 jobs stopped, their outputs moved to
    `results/v4_runs_invalid_evaluator/` and `artifacts/v4_invalid_evaluator/`, and the full
    54-run queue relaunched from scratch under the fixed code. Runs are deterministic, so the
    relaunch reproduces the same training; no run, seed or level was dropped.
29. **RAW rerun under v4 code to log `weighted_acc_p` and `head_loss_bits`** (the v3 RAW CSVs
    predate these columns and P3 needs RAW's `weighted_acc_p` per size and seed). Training is
    deterministic (checked on RAW-S seed 1, deviation 22), so all pre-existing columns are
    identical; the analysis asserts equality with the v3 rows before using the rerun.
30. **v4 queue restarted a second time (08:15 UTC).** The rsync-based code push used
    `--delete` scoped to the repository and removed `results/v4_runs/` on the server (28
    finished-run CSVs) because the local copy had not pulled them; artifacts and logs were
    unaffected. Runs are deterministic, so the full queue (54 + 6 RAW reruns) was relaunched
    under unchanged code; `scripts/sync_remote.sh` now pushes via `git push` (which never
    touches untracked files) and `scripts/pull_results.sh` pulls results additively.

## v5 notes (Experiment Brief v5.1; CRITERIA_v5.md)

31. **FLAT-8 (Part C) is infeasible within the budget**: fully uniform sampling over the
    1,000,000 facts already gives 10.4 exposures per fact in FLAT's stationary phase, so no τ
    reaches c = 8. The cell is run as the fully uniform limit (τ → 0; every fact "at the cap")
    and reported at its true level, c = 10.4 (CURATED at the same τ: 24.7). Level 15 is feasible
    (399,904 facts at the cap).
32. **Cosine schedule is computed in token space**: lr(t) = peak · (0.1 + 0.9 · ½(1 + cos(π · tokens/B)))
    after the 100-step linear warm-up, so it reaches exactly 10 % of peak at B regardless of
    the small variation in tokens per batch.
33. **Probe accuracy on the main-run weights is logged at window end and every 500,000
    documents thereafter, but the main run stops at 0.9 × B** (its last 10 % is covered only
    by the cooldown branch, as in v1–v4), so the retention curve on main weights ends at
    0.9 × B; the branch-weight value at B is logged separately.
34. **Probe facts have probability zero in the ordinary stream (renormalised) but keep the
    v4 τ**; the removed mass is ≈ 3 × 10⁻⁴ of p, so the flattened distribution is unchanged
    to that precision.
