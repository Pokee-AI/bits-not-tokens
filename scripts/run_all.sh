#!/usr/bin/env bash
# Rerun the whole experiment: the v4 matrix (54 flattened-corpus runs + 6 RAW baselines),
# then the v5 robustness matrix (87 runs), across idle GPUs, one run per GPU.
# Each run is deterministic given (corpus config, seed); results land in results/v4_runs
# and results/v5_runs, artifacts in artifacts/, logs in logs/.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs
scripts/ops/run_queue.sh scripts/ops/queue_v4_all.txt
scripts/ops/run_queue.sh scripts/ops/queue_v5.txt
