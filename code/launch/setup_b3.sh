#!/usr/bin/env bash
# Third step of the second-server setup: the jevforge code was copied from the first server (same commit); torch is installed.
# Installs the pinned dependencies and jevforge (editable) into /root/turgutino/jev/venv. No deletions, no sudo.
BASE=/root/turgutino/jev
cd "$BASE" || exit 2
LOG="$BASE/setup3.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
free_gb() { df -BG --output=avail "$1" | tail -n 1 | tr -dc '0-9'; }
export PIP_NO_CACHE_DIR=1 TMPDIR="$BASE/tmp" PYTHONDONTWRITEBYTECODE=1
PIP="$BASE/venv/bin/pip"
say "setup3 start; root free $(free_gb /root) GB"
say "pinned dependencies"
$PIP install transformers==5.19.0 peft==0.21.2 datasets==3.6.0 accelerate==1.15.0 numpy==2.5.4 safetensors==0.8.0 tokenizers==0.23.3 huggingface_hub==2.2.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple > "$BASE/pip_deps.log" 2>&1 || { say "deps install failed"; exit 8; }
say "jevforge editable"
( cd "$BASE/jevforge" && $PIP install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple ) > "$BASE/pip_jev.log" 2>&1 || { say "jevforge install failed"; exit 9; }
say "versions"; $PIP list 2>/dev/null | grep -i -E "^(torch|transformers|peft|datasets|numpy|accelerate|safetensors|tokenizers|jevforge|huggingface)" | tee -a "$LOG"
"$BASE/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())" 2>&1 | tee -a "$LOG"
"$BASE/venv/bin/python" -c "import jevforge, jevforge.cli; print('jevforge import OK')" 2>&1 | tee -a "$LOG"
say "disk after: root free $(free_gb /root) GB, venv $(du -sh "$BASE/venv" | cut -f1)"
say "SETUP3 DONE"
touch "$BASE/SETUP3_DONE"
