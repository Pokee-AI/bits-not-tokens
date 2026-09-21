#!/usr/bin/env bash
# Pull results/ artifacts/ (and logs/) from the GPU server. Additive only: never deletes.
set -euo pipefail
REMOTE=${REMOTE:-USER@GPU_HOST}
KEY=${KEY:-$HOME/.ssh/google_compute_engine}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
for d in results artifacts logs; do
  rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes" "$REMOTE:~/bits-not-tokens/$d/" "$ROOT/$d/"
done
