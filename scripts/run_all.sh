#!/usr/bin/env bash
# Launch the run matrix across idle GPUs, one run per GPU. Runs whose per-run CSV
# already exists are skipped. usage: scripts/run_all.sh "1 2" [extra args]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
seeds=${1:-"1 2"}; shift || true
corpora="Z05_n0 Z05_n64 Z10_n0 Z10_n64 EQ4_n16"
# idle = no compute process and < 1 GiB used
mapfile -t idle < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
  | awk -F', ' '$2 < 1024 {print $1}')
echo "idle GPUs: ${idle[*]}"
i=0
for seed in $seeds; do for c in $corpora; do
  [[ -f "$ROOT/results/runs/${c}_seed${seed}.csv" ]] && { echo "skip $c seed $seed (csv exists)"; continue; }
  [[ $i -ge ${#idle[@]} ]] && { echo "no idle GPU left for $c seed $seed; launch later"; continue; }
  "$ROOT/scripts/launch.sh" "${idle[$i]}" "$c" "$seed" "$@"
  i=$((i + 1))
done; done
