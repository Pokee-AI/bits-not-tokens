#!/usr/bin/env bash
# Run a queue of jobs (one per line: "<corpus> <seed> [extra train.py args]") across idle
# GPUs, one job per GPU. usage: nohup scripts/run_queue.sh queue.txt > logs/queue.log &
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
declare -A pid_on_gpu
mapfile -t jobs < "$1"
i=0
while [[ $i -lt ${#jobs[@]} ]]; do
  for g in $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2 < 1024 {print $1}'); do
    [[ $i -ge ${#jobs[@]} ]] && break
    p=${pid_on_gpu[$g]:-}
    [[ -n $p ]] && kill -0 "$p" 2>/dev/null && continue   # our own job still starting up
    read -r corpus seed extra <<< "${jobs[$i]}"
    # shellcheck disable=SC2086
    out=$("$ROOT/scripts/launch.sh" "$g" "$corpus" "$seed" $extra)
    echo "$(date -u +%FT%TZ) $out"
    pid_on_gpu[$g]=$(sed -n 's/.*pid \([0-9]*\).*/\1/p' <<< "$out")
    i=$((i + 1))
    sleep 20
  done
  sleep 30
done
echo "$(date -u +%FT%TZ) queue drained"
