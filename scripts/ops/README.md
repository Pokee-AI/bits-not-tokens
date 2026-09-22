# Operational scripts

Used to drive the experiment on a multi-GPU server; not needed to read or regenerate results.

- `run_queue.sh <queue.txt>` — runs a queue of jobs (one per line: `<corpus> <seed> [train.py args]`)
  across idle GPUs, one job per GPU, via `launch.sh`.
- `launch.sh <gpu> <corpus> <seed> [args]` — one run on one GPU under nohup, log in `logs/`.
- `queue_v4.txt`, `queue_v4_raw.txt`, `queue_v4_all.txt`, `queue_v5.txt` — the exact job lists
  that produced the v4 and v5 runs, for provenance (`scripts/run_all.sh` replays them).
- `stop_v4.sh` / `restart_v4.sh` — stop and relaunch the v4 queue (used for the two v4 restarts
  recorded in `DEVIATIONS.md` #28 and #30).
