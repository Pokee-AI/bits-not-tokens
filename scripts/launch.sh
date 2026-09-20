#!/usr/bin/env bash
# Launch one run on one GPU under nohup, logging to logs/.
# usage: scripts/launch.sh <gpu> <corpus_name> <seed> [extra train.py args...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gpu=$1; corpus=$2; seed=$3; shift 3
tag=""
for a in "$@"; do [[ $prev == "--tag" ]] && tag="_$a"; prev=$a; done
log="$ROOT/logs/${corpus}_seed${seed}${tag}.log"
cd "$ROOT"
CUDA_VISIBLE_DEVICES=$gpu nohup .venv/bin/python train/train.py --corpus "configs/${corpus}.yaml" \
  --seed "$seed" "$@" > "$log" 2>&1 &
echo "gpu $gpu -> $corpus seed $seed $* (pid $!, log $log)"
