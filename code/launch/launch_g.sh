#!/bin/bash
# Starts the "g" scheduler (queue_q4g.json) detached. Own children only; STOP file: sched/STOP_G.
cd /root/turgutino/jev || exit 2
export HF_HOME=/root/autodl-tmp/turgutino/jev/hf
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/root/turgutino/jev/tmp
export XDG_CACHE_HOME=/root/turgutino/jev/cache
export TRITON_CACHE_DIR=/root/turgutino/jev/triton
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=0
export MIN_FREE_HF_GB=1.0
export SCHED_TAG=g
export MAX_TOTAL_MIB=22500
export MAX_PARALLEL=1
ls -d "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" 2>&1
if [ -e sched/G_LOG.txt ] || [ -e sched/state_g.json ]; then echo "g scheduler files already exist - not starting"; exit 3; fi
setsid nohup ./venv/bin/python scheduler_max4.py /root/turgutino/jev queue_q4g.json > sched/g.out 2>&1 < /dev/null &
sleep 12
echo "--- started"; ps -eo pid,etime,args | grep -E "scheduler_max4" | grep -v grep | cut -c1-120
echo "--- G_LOG"; tail -n 5 sched/G_LOG.txt 2>/dev/null | cut -c1-200
