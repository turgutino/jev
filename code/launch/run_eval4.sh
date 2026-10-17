#!/usr/bin/env bash
# Evaluate the intent-1ep checkpoint-last (training stopped at ~step 3100 of 3228 by the shared-GPU guard) with the paperA prompt.
# New names only (EVAL4_*, results/chain4). Our process is capped at CAP_GIB of CUDA memory (shared GPU). Writes only inside the run dir.
RUN=/root/autodl-tmp/turgutino/projects/jev/run_20261015_161348
MODEL=Qwen/Qwen3-0.6B
cd "$RUN" || exit 2
export HF_HOME="$RUN/hf" HF_HUB_CACHE="$RUN/hf/hub" HF_DATASETS_CACHE="$RUN/hf/datasets"
export TMPDIR="$RUN/tmp" XDG_CACHE_HOME="$RUN/cache" HF_ENDPOINT="https://hf-mirror.com" HF_HUB_DISABLE_PROGRESS_BARS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY="hf-mirror.com,.hf-mirror.com,localhost,127.0.0.1" CAP_GIB=6
JF="$RUN/venv/bin/python $RUN/jf_paper2.py"
AD="$RUN/results/chain3/intent1ep/checkpoint-last"
OUT="$RUN/results/chain4"; LOG="$RUN/logs/chain4"; mkdir -p "$OUT" "$LOG"
S="$RUN/EVAL4_STATUS.txt"
say() { echo "$(date '+%F %T') $*" | tee -a "$S"; }
gpu() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }
say "eval4 start; GPU: $(gpu); adapter: $(ls -la --time-style=long-iso $AD/adapter_model.safetensors | cut -c1-80)"
[ -f "$AD/adapter_model.safetensors" ] || { say "adapter missing"; touch "$RUN/EVAL4_DONE"; exit 0; }
say "STAGE intent_jevbench start"
nice -n 19 timeout 3000 $JF jevbench --model "$MODEL" --adapter "$AD" --label "$MODEL intent-1ep(step~3100/3228) paperA" --out "$OUT/intent1ep_eval" > "$LOG/intent_jevbench.log" 2>&1
say "STAGE intent_jevbench exit $?; GPU: $(gpu)"
say "STAGE intent_banking77 start"
nice -n 19 timeout 3000 $JF eval --model "$MODEL" --adapter "$AD" --tasks banking77 --limit 1000 --readouts jev --out "$OUT/intent1ep_eval" > "$LOG/intent_banking77.log" 2>&1
say "STAGE intent_banking77 exit $?; GPU: $(gpu)"
say "EVAL4 DONE"; touch "$RUN/EVAL4_DONE"
