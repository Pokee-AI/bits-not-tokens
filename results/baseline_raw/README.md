# RAW baselines

`RAW_seed{1,2}_{S,M,L}.csv` are the raw-corpus (shifted-Zipf, filler 16, 300M-token budget)
baseline runs for the three model sizes. They were trained in the earlier design round
("v3" in the frozen results and criteria files; branch `archive/rounds-1-3`) under the same
code, configuration and seeds used by v4. RESULTS_V4.md's determinism check is the evidence:
a rerun of RAW-S seed 1 under the v4 code reproduced these rows exactly, and the six RAW reruns
made in v5 (which add the `weighted_acc_p` / `head_loss_bits` columns) are asserted equal to
these files in every shared column before use (`analysis/v4_analysis.load`).
