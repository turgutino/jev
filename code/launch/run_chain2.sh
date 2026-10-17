#!/usr/bin/env bash
# Chain v2: Banking77 base, Reason-9k train+eval, Intent-1ep train+eval; all with prompt paperA; Qwen3-0.6B; seed = repo default.
# New names only (CHAIN2_*, results/chain2, logs/chain2): nothing from earlier attempts is touched.
# Writes ONLY inside the run dir. Never kills anything except its own stage. No sudo, no API keys.
# Stop politely: touch $RUN/CHAIN_STOP2
# Guard: if total GPU memory used >= GUARD_MIB (other users' jobs + ours), our own running stage is stopped and the chain ends.
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
LOG="$RUN/logs/chain2"; mkdir -p "$LOG"
OUT="$RUN/results/chain2"; mkdir -p "$OUT"
S="$RUN/CHAIN2_STATUS.txt"
say() { echo "$(date '+%F %T') $*" | tee -a "$S"; }
gpum() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' '; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
stopped() { [ -f "$RUN/CHAIN_STOP2" ] && { say "CHAIN_STOP2 found; finishing"; touch "$RUN/CHAIN2_DONE"; exit 0; }; }
stage() { # name timeout cmd...
  local n="$1" t="$2"; shift 2; stopped
  say "STAGE $n start (timeout ${t}s); GPU: $(gpu)"
  nice -n 19 timeout "$t" "$@" > "$LOG/$n.log" 2>&1 &
  local pid=$! peak=0 m
  while kill -0 "$pid" 2>/dev/null; do
    m=$(gpum); m=${m:-0}
    [ "$m" -gt "$peak" ] 2>/dev/null && peak=$m
    if [ "$m" -ge "$GUARD_MIB" ] 2>/dev/null; then
      say "GUARD: total GPU memory ${m} MiB >= ${GUARD_MIB}; stopping OUR stage $n (pid $pid) and ending the chain"
      touch "$RUN/CHAIN_STOP2"; kill "$pid" 2>/dev/null
    fi
    sleep 20
  done
  wait "$pid"; local rc=$?
  say "STAGE $n exit $rc (124 = timeout, 143 = stopped); peak total GPU memory seen: ${peak} MiB"
}
REASON="clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"

say "chain2 start (paperA prompt, $MODEL); GPU: $(gpu)"
stage base_banking77 3000 $JF eval --model "$MODEL" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/base"

# --- Reason-9k (control; 282 steps) ---
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
say "CHAIN2 DONE"
touch "$RUN/CHAIN2_DONE"
