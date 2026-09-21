# C_STAR.md — frozen per-model-size constants for Phase P (Experiment Brief v3 §4.3)

Computed from Phase A (`results/v3_phaseA_summary.csv`, `results/v3_cstar.csv`), committed
after Phase A and before any Phase P run. Never edited afterwards.

| model | non-embedding params | c90 (K_sub = 30,000, seed mean, exposures from start) | **c\*** | longest plateau (K_sub = 30,000) | 2 × plateau | **W** |
|---|---|---|---|---|---|---|
| S | 2,045,568 | 38.83 (seeds: 38.96, 38.70) | **40** | 512,000 docs (both seeds) | 1,050,000 | **1,000,000** — the 1,000,000-document cap binds |
| M | 7,993,152 | 22.77 (22.67, 22.86) | **30** | 128,000 docs (both seeds) | 256,000 | **300,000** |
| L | 31,524,864 | 21.18 (21.42, 20.94) | **30** | 128,000 docs (both seeds) | 256,000 | **300,000** |

Rules applied: c* = c90 from the K_sub = 30,000 runs, rounded up to the next multiple of 10,
capped at 400 (no size needed the K_sub = 10,000 fallback; no size hit the cap). W = 2 × the
longest plateau length in documents over the K_sub = 30,000 runs, rounded up to the next
50,000, capped at 1,000,000 documents; W is identical for CAPPED and CURATED. Plateau length =
first probe step (every 1,000 steps) with in-distribution object loss < 11.5 bits.

Stop condition: every Phase A run left the plateau (the L / K_sub = 100,000 runs took
3.58M and 4.35M documents; they do not enter c* or W). Phase P proceeds for S, M and L.

Context for the cap-adequacy check (PASS_CRITERIA_v3.md): on dense uniform data c90 is
21–39 exposures, whereas on the v1 Zipf corpora it is 80–100 (Phase 0). c* is therefore
tight relative to Zipf-shaped absorption; the capped streams concentrate each fact's
exposures in time (a sliding head), which is closer to the dense regime, but whether that
suffices is what cap adequacy measures.
