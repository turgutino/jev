#!/usr/bin/env bash
# Chain: Reason-9k train+eval, then Intent-1ep train+eval, all with prompt paperA. Qwen3-0.6B, seed = repo default.
# Writes ONLY inside the run dir. No kill/rm/sudo, no API keys. Stop politely with: touch $RUN/CHAIN_STOP
RUN=/root/autodl-tmp/turgutino/projects/jev/run_20261015_161348
MODEL=Qwen/Qwen3-0.6B
cd "$RUN" || exit 2
export HF_HOME="$RUN/hf" HF_HUB_CACHE="$RUN/hf/hub" HF_DATASETS_CACHE="$RUN/hf/datasets"
export PIP_CACHE_DIR="$RUN/pip" TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache"
export HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY="hf-mirror.com,.hf-mirror.com,localhost,127.0.0.1"
PY="$RUN/venv/bin/python"; JF="$PY $RUN/jf_paper.py"
LOG="$RUN/logs/chain"; mkdir -p "$LOG" "$RUN/results/chain"
S="$RUN/CHAIN_STATUS.txt"
say() { echo "$(date '+%F %T') $*" | tee -a "$S"; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
stopped() { [ -f "$RUN/CHAIN_STOP" ] && { say "CHAIN_STOP found; finishing"; touch "$RUN/CHAIN_DONE"; exit 0; }; }
stage() { # name timeout cmd...
  local n="$1" t="$2"; shift 2; stopped
  say "STAGE $n start (timeout ${t}s); GPU: $(gpu)"
  nice -n 19 timeout "$t" "$@" > "$LOG/$n.log" 2>&1
  say "STAGE $n exit $? (124 = timeout)"
}
REASON="clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"
OUT="$RUN/results/chain"

say "chain start (paperA prompt, $MODEL)"
# Banking77 baseline (paper reports it); non-fatal if the command differs
stage base_banking77 3000 $JF eval --model "$MODEL" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/base"

# --- Reason-9k (control; 282 steps) ---
stage reason9k_train 14400 $JF train --model "$MODEL" --mix "$REASON" --max-examples 9000 --label "$MODEL LoRA reason-9k paperA" --out "$OUT/reason9k"
AD="$OUT/reason9k"; [ -f "$AD/adapter_model.safetensors" ] || AD=""
if [ -n "$AD" ]; then
  stage reason9k_jevbench 5400 $JF jevbench --model "$MODEL" --adapter "$AD" --label "$MODEL reason-9k paperA" --out "$OUT/reason9k_eval"
  stage reason9k_banking77 3000 $JF eval --model "$MODEL" --adapter "$AD" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/reason9k_eval"
else say "no reason9k adapter"; fi

# --- Intent-1ep (the intent trap; 2984 steps, GPU-share dependent: roughly 6-12 h) ---
stage intent_train 50400 $JF train --model "$MODEL" --mix intent --label "$MODEL LoRA intent-1ep paperA" --out "$OUT/intent1ep"
AD="$OUT/intent1ep"; [ -f "$AD/adapter_model.safetensors" ] || AD="$OUT/intent1ep/checkpoint-last"
if [ -d "$AD" ]; then
  stage intent_jevbench 5400 $JF jevbench --model "$MODEL" --adapter "$AD" --label "$MODEL intent-1ep paperA" --out "$OUT/intent1ep_eval"
  stage intent_banking77 3000 $JF eval --model "$MODEL" --adapter "$AD" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/intent1ep_eval"
else say "no intent adapter"; fi
say "CHAIN DONE"
touch "$RUN/CHAIN_DONE"
