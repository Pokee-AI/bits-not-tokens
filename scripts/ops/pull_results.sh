#!/usr/bin/env bash
# Pull results/ artifacts/ logs/ from the GPU server. Additive only: never deletes.
# Requires REMOTE=user@host (and optionally KEY=path to the ssh key).
set -euo pipefail
: "${REMOTE:?set REMOTE=user@host}"
KEY=${KEY:-$HOME/.ssh/id_ed25519}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for d in results artifacts logs; do
  rsync -az -e "ssh -i $KEY -o IdentitiesOnly=yes -o BatchMode=yes" "$REMOTE:~/bits-not-tokens/$d/" "$ROOT/$d/"
done
