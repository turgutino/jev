#!/usr/bin/env bash
# Setup of the jevforge environment on the second server (professor-shared, 4090 D 24 GB). Mirrors the first server's versions.
# Writes ONLY to /root/turgutino/jev (root FS: venv, code, logs) and /root/autodl-tmp/turgutino/jev (model/dataset caches).
# No sudo, no deletions, no API keys. Aborts if free disk gets too low.
BASE=/root/turgutino/jev
HFD=/root/autodl-tmp/turgutino/jev
cd "$BASE" || exit 2
LOG="$BASE/setup.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
free_gb() { df -BG --output=avail "$1" | tail -n 1 | tr -dc '0-9'; }
guard() { local f; f=$(free_gb /root); [ "$f" -lt "$1" ] && { say "ABORT: root disk free ${f} GB < $1"; exit 3; }; return 0; }
export PIP_NO_CACHE_DIR=1 TMPDIR="$BASE/tmp" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$HFD/hf"
say "setup start; root free $(free_gb /root) GB, autodl-tmp free $(free_gb /root/autodl-tmp) GB"
guard 12
say "venv"
/root/miniconda3/bin/python -m venv "$BASE/venv" || { say "venv failed"; exit 4; }
PIP="$BASE/venv/bin/pip"
say "torch 2.14.1 cu126"
$PIP install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126 > "$BASE/pip_torch.log" 2>&1 || { say "torch install failed"; exit 5; }
say "torch done; root free $(free_gb /root) GB"
guard 4
say "clone jevforge @ 8986e52 (+ submodule 1bcc55e)"
git clone --recurse-submodules https://github.com/pedroliveiraAI/LLMtoSystemOne.git "$BASE/jevforge" > "$BASE/git_clone.log" 2>&1 || { say "clone failed"; exit 6; }
( cd "$BASE/jevforge" && git checkout -q 8986e521b8220dc41ff4de44fb418dc0da02afea && git submodule update --init --recursive -q ) >> "$BASE/git_clone.log" 2>&1 || { say "checkout failed"; exit 7; }
say "jevforge HEAD: $(cd "$BASE/jevforge" && git rev-parse HEAD); jevbench: $(cd "$BASE/jevforge/third_party/jevbench" && git rev-parse HEAD)"
say "pinned dependencies"
$PIP install transformers==5.19.0 peft==0.21.2 datasets==3.6.0 accelerate==1.15.0 numpy==2.5.4 safetensors==0.8.0 tokenizers==0.23.3 huggingface_hub==2.2.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple > "$BASE/pip_deps.log" 2>&1 || { say "deps install failed"; exit 8; }
guard 3
say "jevforge editable"
( cd "$BASE/jevforge" && $PIP install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple ) > "$BASE/pip_jev.log" 2>&1 || { say "jevforge install failed"; exit 9; }
say "versions"; $PIP list 2>/dev/null | grep -i -E "^(torch|transformers|peft|datasets|numpy|accelerate|safetensors|tokenizers|jevforge|huggingface)" | tee -a "$LOG"
"$BASE/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())" 2>&1 | tee -a "$LOG"
say "disk after: root free $(free_gb /root) GB, venv $(du -sh "$BASE/venv" | cut -f1)"
say "SETUP DONE"
touch "$BASE/SETUP_DONE"
