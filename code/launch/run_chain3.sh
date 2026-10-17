#!/usr/bin/env bash
# Chain v3 = chain v2 minus the Banking77 baseline (already done in results/chain2/base) and with a corrected guard.
# Guard: our own stage is stopped ONLY if total GPU memory >= GUARD_MIB AND a compute process that is not ours (different session id)
# is listed by nvidia-smi. Our own memory alone never trips it. New names only (CHAIN3_*, results/chain3, logs/chain3).
# Writes ONLY inside the run dir. Never kills anything except its own stage. No sudo, no API keys.
# Stop politely: touch $RUN/CHAIN_STOP3
RUN=/root/autodl-tmp/turgutino/projects/jev/run_20261015_161348
MODEL=Qwen/Qwen3-0.6B
GUARD_MIB=23500
cd "$RUN" || exit 2
export HF_HOME="$RUN/hf" HF_HUB_CACHE="$RUN/hf/hub" HF_DATASETS_CACHE="$RUN/hf/datasets"
export PIP_CACHE_DIR="$RUN/pip" TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
export HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY="hf-mirror.com,.hf-mirror.com,localhost,127.0.0.1"
PY="$RUN/venv/bin/python"; JF="$PY $RUN/jf_paper.py"
LOG="$RUN/logs/chain3"; mkdir -p "$LOG"
OUT="$RUN/results/chain3"; mkdir -p "$OUT"
S="$RUN/CHAIN3_STATUS.txt"
MYSID=$(ps -o sid= -p $$ | tr -d ' ')
say() { echo "$(date '+%F %T') $*" | tee -a "$S"; }
gpum() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
foreign_gpu_pids() { # pids using the GPU whose session id is not ours
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
    s=$(ps -o sid= -p "$p" 2>/dev/null | tr -d ' '); [ -n "$s" ] && [ "$s" != "$MYSID" ] && echo "$p"
  done; }
stopped() { [ -f "$RUN/CHAIN_STOP3" ] && { say "CHAIN_STOP3 found; finishing"; touch "$RUN/CHAIN3_DONE"; exit 0; }; }
stage() { # name timeout cmd...
  local n="$1" t="$2"; shift 2; stopped
  say "STAGE $n start (timeout ${t}s); GPU: $(gpu); foreign GPU pids: [$(foreign_gpu_pids | tr '\n' ' ')]"
  nice -n 19 timeout "$t" "$@" > "$LOG/$n.log" 2>&1 &
  local pid=$! peak=0 m f
  while kill -0 "$pid" 2>/dev/null; do
    m=$(gpum); m=${m:-0}
    [ "$m" -gt "$peak" ] 2>/dev/null && peak=$m
    if [ "$m" -ge "$GUARD_MIB" ] 2>/dev/null; then
      f=$(foreign_gpu_pids | tr '\n' ' ')
      if [ -n "$f" ]; then
        say "GUARD: total GPU memory ${m} MiB >= ${GUARD_MIB} and foreign GPU pids [$f]; stopping OUR stage $n (pid $pid) and ending the chain"
        touch "$RUN/CHAIN_STOP3"; kill "$pid" 2>/dev/null
      fi
    fi
    sleep 20
  done
  wait "$pid"; local rc=$?
  say "STAGE $n exit $rc (124 = timeout, 143 = stopped); peak total GPU memory seen: ${peak} MiB"
}
REASON="clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"

say "chain3 start (paperA prompt, $MODEL); GPU: $(gpu); session $MYSID"
# --- Reason-9k (control; target 282 steps; sources that fail to download are skipped by jevforge and logged in the train log) ---
stage reason9k_train 21600 $JF train --model "$MODEL" --mix "$REASON" --max-examples 9000 --label "$MODEL LoRA reason-9k paperA" --out "$OUT/reason9k"
AD="$OUT/reason9k"; [ -f "$AD/adapter_model.safetensors" ] || AD="$OUT/reason9k/checkpoint-last"
if [ -d "$AD" ]; then
  stage reason9k_jevbench 5400 $JF jevbench --model "$MODEL" --adapter "$AD" --label "$MODEL reason-9k paperA" --out "$OUT/reason9k_eval"
  stage reason9k_banking77 3000 $JF eval --model "$MODEL" --adapter "$AD" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/reason9k_eval"
else say "no reason9k adapter"; fi

# --- Intent-1ep (the intent trap; 2984 steps) ---
stage intent_train 72000 $JF train --model "$MODEL" --mix intent --label "$MODEL LoRA intent-1ep paperA" --out "$OUT/intent1ep"
AD="$OUT/intent1ep"; [ -f "$AD/adapter_model.safetensors" ] || AD="$OUT/intent1ep/checkpoint-last"
if [ -d "$AD" ]; then
  stage intent_jevbench 5400 $JF jevbench --model "$MODEL" --adapter "$AD" --label "$MODEL intent-1ep paperA" --out "$OUT/intent1ep_eval"
  stage intent_banking77 3000 $JF eval --model "$MODEL" --adapter "$AD" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/intent1ep_eval"
else say "no intent adapter"; fi
say "CHAIN3 DONE"
touch "$RUN/CHAIN3_DONE"
