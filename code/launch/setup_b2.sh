#!/usr/bin/env bash
# Continuation of setup_b.sh on the second server: torch is already installed in /root/turgutino/jev/venv.
# Clone over HTTP/1.1 (the first attempt hit a transient HTTP/2 framing error) with retries, then pinned deps and jevforge.
# Writes only to /root/turgutino/jev and /root/autodl-tmp/turgutino/jev. No deletions, no sudo.
BASE=/root/turgutino/jev
HFD=/root/autodl-tmp/turgutino/jev
cd "$BASE" || exit 2
LOG="$BASE/setup2.log"
say() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
free_gb() { df -BG --output=avail "$1" | tail -n 1 | tr -dc '0-9'; }
export PIP_NO_CACHE_DIR=1 TMPDIR="$BASE/tmp" PYTHONDONTWRITEBYTECODE=1
PIP="$BASE/venv/bin/pip"
say "setup2 start; root free $(free_gb /root) GB"
DIR="$BASE/jevforge_src"
ok=0
for try in 1 2 3 4; do
  if [ -d "$DIR/.git" ]; then ok=1; break; fi
  say "clone try $try (HTTP/1.1)"
  git -c http.version=HTTP/1.1 clone --recurse-submodules https://github.com/pedroliveiraAI/LLMtoSystemOne.git "$DIR" > "$BASE/git_clone2_$try.log" 2>&1 && { ok=1; break; }
  [ -d "$DIR" ] && { say "partial clone dir left by try $try; using a new directory name"; DIR="$BASE/jevforge_src_t$try"; }
  sleep 20
done
[ "$ok" = 1 ] || { say "clone failed after retries"; exit 6; }
say "cloned into $DIR"
( cd "$DIR" && git checkout -q 8986e521b8220dc41ff4de44fb418dc0da02afea && git -c http.version=HTTP/1.1 submodule update --init --recursive -q ) >> "$BASE/git_clone2.log" 2>&1 || { say "checkout failed"; exit 7; }
say "jevforge HEAD: $(cd "$DIR" && git rev-parse HEAD); jevbench: $(cd "$DIR/third_party/jevbench" && git rev-parse HEAD)"
echo "$DIR" > "$BASE/JEVFORGE_DIR"
say "pinned dependencies"
$PIP install transformers==5.19.0 peft==0.21.2 datasets==3.6.0 accelerate==1.15.0 numpy==2.5.4 safetensors==0.8.0 tokenizers==0.23.3 huggingface_hub==2.2.0 \
  -i https://pypi.tuna.tsinghua.edu.cn/simple > "$BASE/pip_deps.log" 2>&1 || { say "deps install failed"; exit 8; }
say "jevforge editable"
( cd "$DIR" && $PIP install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple ) > "$BASE/pip_jev.log" 2>&1 || { say "jevforge install failed"; exit 9; }
say "versions"; $PIP list 2>/dev/null | grep -i -E "^(torch|transformers|peft|datasets|numpy|accelerate|safetensors|tokenizers|jevforge|huggingface)" | tee -a "$LOG"
"$BASE/venv/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())" 2>&1 | tee -a "$LOG"
say "disk after: root free $(free_gb /root) GB, venv $(du -sh "$BASE/venv" | cut -f1)"
say "SETUP2 DONE"
touch "$BASE/SETUP2_DONE"
