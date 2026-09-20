#!/usr/bin/env bash
# Push the repo (code + results, not venv/artifacts) to the GPU server.
# Source and destination are both explicit repo directories; --delete is scoped to
# the remote bits-not-tokens/ directory only.
set -euo pipefail
REMOTE=${REMOTE:-USER@GPU_HOST}
KEY=${KEY:-$HOME/.ssh/google_compute_engine}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$ROOT/PASS_CRITERIA.md" ] || { echo "refusing: $ROOT is not the repo root"; exit 1; }
rsync -az --delete --exclude .venv --exclude artifacts --exclude __pycache__ --exclude .pytest_cache \
  --exclude 'logs/*.log' -e "ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes" \
  "$ROOT/" "$REMOTE:~/bits-not-tokens/"
