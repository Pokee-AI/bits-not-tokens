#!/usr/bin/env bash
# Push committed code to the GPU server via git (never touches untracked files there:
# results/, artifacts/, logs/ on the server are the source of truth and are only ever pulled).
set -euo pipefail
REMOTE=${REMOTE:-USER@GPU_HOST}
KEY=${KEY:-$HOME/.ssh/google_compute_engine}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes"
git remote get-url gpu >/dev/null 2>&1 || git remote add gpu "$REMOTE:bits-not-tokens"
ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$REMOTE" 'cd ~/bits-not-tokens && git config receive.denyCurrentBranch updateInstead'
git push gpu HEAD:"$(git branch --show-current)"
