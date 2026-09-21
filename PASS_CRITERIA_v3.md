# Pass criteria v3 (section 6 of Experiment Brief v3, as amended, copied verbatim)

Committed before Phase A. Never edited afterwards.

All comparisons use the final measurement point and the mean over seeds; per-seed values
are also reported. Corpora: RAW (shifted Zipf, no cap, filler 16), CAPPED (shifted Zipf,
after warm-up W a fact that has reached c* exposures is never drawn again, filler 16),
CURATED (same capped sampling, filler 0). CAPPED caps repetitions per fact; it does not
deduplicate documents. c* and W per model size are frozen in `C_STAR.md` before any
Phase P run. Budget B = 300,000,000 non-PAD training tokens per run.

- **Evaluability.** A model size is *evaluable* if its RAW run stores at least 100 facts
  (seed mean) at the final point. A run that never leaves the plateau counts as storing
  zero facts; it is not rerun with changed settings, and it is noted prominently.
- **P1 (equal tokens).** For every evaluable model size, CURATED stores at least 3x as
  many facts as RAW. PASS only if the 31M model (L) is evaluable, P1 holds for every
  evaluable size, and no size is counted as a pass because it was not evaluable.
- **P2 (size equivalence).** The S model (about 2M non-embedding parameters) trained on
  CURATED stores at least as many facts as the L model (about 31M) trained on RAW.
  Requires L evaluable; a non-evaluable L is never counted as a pass.
- **Cap adequacy (assumption check).** c* is measured on dense uniform data (Phase A) and
  applied to Zipf-shaped streams; that transfer is an explicit assumption. For every
  CAPPED and CURATED run report, among facts that reached c* exposures, the fraction
  stored. Below 0.7 the run is flagged **cap-limited**, and an under-storing capped
  corpus is attributed to the cap, not to the premise.
- **Descriptive, no pass/fail.**
  - Attribution: CAPPED / RAW and CURATED / CAPPED ratios of facts stored, per model size,
    separating the effect of capping repetition from the effect of removing noise.
  - Predicted ratio: the CURATED / RAW ratio implied by `results/v3_predicted_storable.csv`
    (facts reaching c* within B) next to the measured ratio, so a 3x result when 10x was
    predicted is visible as a shortfall.
  - Headroom: bits stored per non-embedding parameter (total-parameter value alongside)
    for every (corpus, size), next to the reference values 2.0 (Allen-Zhu and Li, 2024) and
    3.6 (Morris et al., 2025) bits per parameter. If S flattens under CURATED, report the
    bits per parameter at which it flattens (capacity ceiling).
  - Predicted versus actual: facts stored against `facts_with_at_least_cstar_exposures`.
- With two seeds there are no confidence intervals. Report per-seed values and the range.
