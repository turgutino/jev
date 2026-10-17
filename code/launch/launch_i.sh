#!/bin/bash
# Waiter: starts the "i" scheduler (queue_q4i.json) only after the "g" scheduler (pid 871425) has ended
# (finished OR died), so the two queues never share the GPU. Own processes only.
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
export SCHED_TAG=i
export MAX_TOTAL_MIB=22500
export MAX_PARALLEL=1
if [ -e sched/I_LOG.txt ] || [ -e sched/state_i.json ]; then echo "i scheduler files already exist - not starting"; exit 3; fi
echo "$(date '+%F %T') waiting for g scheduler pid 871425"
while kill -0 871425 2>/dev/null; do sleep 30; done
echo "$(date '+%F %T') g scheduler ended; starting i"
exec ./venv/bin/python scheduler_max4.py /root/turgutino/jev queue_q4i.json
