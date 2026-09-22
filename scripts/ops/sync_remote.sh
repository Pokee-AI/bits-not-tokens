#!/usr/bin/env bash
# Push committed code to the GPU server via git (never touches untracked files there:
# results/, artifacts/, logs/ on the server are the source of truth and are only ever pulled).
# Requires REMOTE=user@host (and optionally KEY=path to the ssh key).
set -euo pipefail
: "${REMOTE:?set REMOTE=user@host}"
KEY=${KEY:-$HOME/.ssh/id_ed25519}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export GIT_SSH_COMMAND="ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes"
git remote get-url gpu >/dev/null 2>&1 || git remote add gpu "$REMOTE:bits-not-tokens"
ssh -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes "$REMOTE" 'cd ~/bits-not-tokens && git config receive.denyCurrentBranch updateInstead'
git push gpu HEAD:"$(git branch --show-current)"
if git remote get-url origin >/dev/null 2>&1; then git push origin HEAD; fi
