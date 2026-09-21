# Criteria v5 (section 7 of Experiment Brief v5.1, copied verbatim)

Committed before any v5 run. Never edited afterwards. v5 is characterization, not a premise
test; v4's verdicts stand whatever v5 shows.

- **R1 (the effect survives the recipe).** For M, under every one of the five optimizer settings, FLAT-sel stores at least 1.5x the facts RAW stores (seed mean, final point), and FLAT-sel's `weighted_acc_p` is no more than 0.02 below RAW's. ROBUST if it holds for all five; otherwise list the settings where it fails. Report the same for S and L on their three settings.
- **R2 (steady state).** For M under `base`, FLAT-sel facts stored at 4B is within 15% of its value at B (seed mean). If it grows by more than 15%, report NOT A STEADY STATE and give the growth exponent in tokens.
- **R3 (what sets the horizon), descriptive.** Probe half-life per setting and size, next to the weight-decay timescale 1/(lr x wd) steps = 128/(lr x wd) documents (infinite for `wd0`). Absolute agreement is not expected, since one is a half-life and the other an e-folding time of the weights; what matters is the scaling across settings (for example, whether `lr3` lengthens the half-life by about 3x). State plainly whether half-life scales with 1/(lr x wd), with 1/lr alone, or with neither. For `wd0`, a finite half-life means forgetting comes from interference, not decay.
- **R4 (intervals), descriptive.** FLAT-sel / RAW and CURATED-sel / RAW ratios per size as mean and 95% t-interval over seeds 2 to 5 (3 degrees of freedom, so expect wide intervals), with seed 1 shown separately. If GPUs are idle at the end, seeds 6 and 7 may be added for all Part B cells at once, never selectively.
- **R5 (L's optimum), descriptive.** Facts stored and `weighted_acc_p` versus level for L including the two new levels; identify the best P3-eligible level per seed.
- Under `cosine`, also report facts stored per setting against `base`: does a decayed schedule retain more of what was seen earlier in the run? Compare the absorption curve's dependence on *when* a fact's exposures occurred (early third versus last third of the run), using the saved `first_seen` / `last_seen` arrays.
