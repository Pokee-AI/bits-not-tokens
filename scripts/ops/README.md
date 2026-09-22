# Operational scripts

Used to drive the experiment on a multi-GPU server; not needed to read or regenerate results.

- `run_queue.sh <queue.txt>` — runs a queue of jobs (one per line: `<corpus> <seed> [train.py args]`)
  across idle GPUs, one job per GPU, via `../launch.sh`.
- `queue_*.txt` — the exact job lists that produced each round's runs (v2 reruns, v3 Phase A / P,
  v4, v5), for provenance.
- `sync_remote.sh` / `pull_results.sh` — code push (git) and results pull (rsync, additive) between a
  workstation and the GPU server; require `REMOTE=user@host`.
- `stop_v4.sh` / `restart_v4.sh` — stop and relaunch the v4 queue (used for the two v4 restarts
  recorded in `DEVIATIONS.md` #28 and #30).
