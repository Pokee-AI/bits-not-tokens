#!/usr/bin/env bash
# Archive partial v4 outputs and relaunch the full v4 queue (54 + 6 RAW reruns).
set -u
cd "$(dirname "$0")/../.."
ts=$(date -u +%Y%m%dT%H%M)
mkdir -p "logs/v4_archive_$ts" "artifacts/v4_archive_$ts" "results/v4_archive_$ts"
mv logs/FLAT_seed*_*.log logs/CURATEDF_seed*_*.log logs/RAW_seed*_[SML].log logs/queue_v4*.log "logs/v4_archive_$ts/" 2>/dev/null
for d in artifacts/FLAT_seed* artifacts/CURATEDF_seed*; do [ -e "$d" ] && mv "$d" "artifacts/v4_archive_$ts/"; done
[ -d results/v4_runs ] && mv results/v4_runs "results/v4_archive_$ts/"
cat scripts/ops/queue_v4.txt scripts/ops/queue_v4_raw.txt > scripts/ops/queue_v4_all.txt
nohup scripts/ops/run_queue.sh scripts/ops/queue_v4_all.txt > logs/queue_v4.log 2>&1 &
sleep 2
echo "relaunched $(wc -l < scripts/ops/queue_v4_all.txt) jobs"
