#!/usr/bin/env bash
# Prompt audit + repeat speed test. Writes ONLY inside the existing run dir. No kill/rm/sudo. Run detached.
RUN=/root/autodl-tmp/turgutino/projects/jev/run_20261015_161348
MODEL=Qwen/Qwen3-0.6B
cd "$RUN" || exit 2
export HF_HOME="$RUN/hf" HF_HUB_CACHE="$RUN/hf/hub" HF_DATASETS_CACHE="$RUN/hf/datasets"
export PIP_CACHE_DIR="$RUN/pip" TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
export HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY="hf-mirror.com,.hf-mirror.com,localhost,127.0.0.1"
PY="$RUN/venv/bin/python"
LOG="$RUN/logs/audit"; mkdir -p "$LOG" "$RUN/results/promptaudit"
S="$RUN/AUDIT_STATUS.txt"
say() { echo "$(date '+%F %T') $*" | tee -a "$S"; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }

say "audit start; GPU: $(gpu)"
for PV in default paperA paperB; do
  say "variant $PV start; GPU: $(gpu)"
  PV=$PV nice -n 19 timeout 3000 "$PY" "$RUN/prompt_audit.py" "$MODEL" "$RUN/results/promptaudit/$PV" > "$LOG/$PV.log" 2>&1
  say "variant $PV exit $?"
done
say "speed test repeat start; GPU: $(gpu)"
nice -n 19 timeout 2400 "$RUN/venv/bin/jevforge" train --model "$MODEL" --mix intent --max-examples 640 --max-steps 4 \
  --decontaminate none --save-every 0 --out "$RUN/results/speedtest2" > "$LOG/speedtest2.log" 2>&1
say "speed test repeat exit $?; GPU: $(gpu)"
grep -i "s/step\|sec/step\|it/s\|s/it" "$LOG/speedtest2.log" | tail -n 5 | tee -a "$S"
say "AUDIT DONE"
touch "$RUN/AUDIT_DONE"
