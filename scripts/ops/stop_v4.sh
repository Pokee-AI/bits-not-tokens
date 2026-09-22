#!/usr/bin/env bash
# Stop v4 queue runners and v4 training jobs (only those from this repo's v4 configs).
for p in $(pgrep -f "run_queue.sh scripts/queue_v4"); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f "train/train.py --corpus configs/v3/(FLAT|CURATEDF|RAW).yaml"); do kill "$p" 2>/dev/null; done
sleep 8
echo "train procs: $(pgrep -fc 'train/train.py')  queue procs: $(pgrep -fc 'run_queue.sh')"
