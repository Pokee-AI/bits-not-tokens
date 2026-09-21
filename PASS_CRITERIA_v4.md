# Pass criteria v4 (section 6 of Experiment Brief v4.1, copied verbatim)

Committed before any v4 training run. Never edited afterwards.

**Level selection.** For each model size and corpus type, consider only the levels that pass the P3 head-loss guard on **seed 1** (against RAW seed 1). Among those, the selected level is the one with the most facts stored at B on seed 1. If no level passes P3 on seed 1, report NO ELIGIBLE LEVEL for that size and corpus type, and P1 for that size is FAIL. All pass/fail comparisons then use **seed 2 only**, at the selected level, against RAW seed 2. Seed 1 numbers are reported but decide nothing except the selection.

- **P1 (equal tokens), same bar as v3.** CURATED at the selected level stores at least 3x the facts RAW stores. A model size is evaluable only if RAW stores at least 100 facts. PASS requires L to be evaluable and the ratio to hold at every evaluable size.
- **P2 (size equivalence), same bar as v3.** S on CURATED (selected level, seed 2) stores at least as many facts as L on RAW (seed 2). Evaluable only if L on RAW stores at least 100 facts.
- **P3 (no head loss), required for P1 and P2 to count.** Checked again on seed 2: at the selected level, `weighted_acc_p` is no more than 0.02 (absolute) below RAW's at the same size and seed. If P3 fails for a size, that size's P1 result is reported as FAIL (HEAD LOSS).
- **Descriptive, no pass/fail.**
  - Dose-response: facts stored and `weighted_acc_p` versus level c, per size and corpus type, both seeds, with RAW as the reference.
  - Attribution: FLAT / RAW (flattening alone) and CURATED / FLAT (noise removal) at each level.
  - Predicted versus actual facts stored for every run (section 5), and the actual/predicted ratio.
  - M on CURATED versus L on RAW (a 4x size gap), alongside P2's 16x gap.
  - Bits per parameter against the 2.0 and 3.6 reference values.
  - Plateau: any run whose in-distribution loss never drops below 11.5 bits counts as storing zero facts and is not rerun.
- With one confirmation seed there are no confidence intervals. Say so, and report v3's observed seed-to-seed spread next to the verdicts for context.
