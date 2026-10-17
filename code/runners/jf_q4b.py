"""jevforge command with optional 4-bit (QLoRA-style) base model and optional cap on answer options per training question.
jevforge itself is NOT edited: the loaders / build_mix are replaced at run time (same technique as jf_max.py).
Env:
  PROMPT_VARIANT  paperA (default) | default        prompt wording (see jf_max.py)
  Q4=1            load the base model in 4-bit nf4 (bitsandbytes, double quant, bf16 compute), for training AND evaluation
  MAX_OPTS=N      training only: every question with more than N options keeps its gold option + N-1 random others (seeded)
  FORCE_MODE=m    force jevforge's scoring mode (tree|cache|naive) instead of the automatic check, so a transient OOM can never silently switch the mode\n  CAP_GIB=G       hard cap on this process's CUDA memory (so it cannot starve other jobs on the same GPU)
Deviations from the paper's setup (must be reported): 4-bit base weights; fewer options per training question.
Usage: python jf_q4.py <jevforge args...>"""
import hashlib
import os
import random
import sys

import torch

import jevforge.jev as J
import jevforge.prompt as P
import jevforge.train as T

TAIL = "Select one option. Answer only with its bracketed numeric identifier."


def paperA(state, options, instruction=None):
    opts = [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]
    return "\n".join([f"State: {str(state).strip()}", "Options:"] + opts + [TAIL])


VARIANT = os.environ.get("PROMPT_VARIANT", "paperA")
if VARIANT == "paperA":
    P.format_user_message = paperA
elif VARIANT != "default":
    sys.exit(f"unknown PROMPT_VARIANT {VARIANT}")

if os.environ.get("CAP_GIB") and torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, float(os.environ["CAP_GIB"]) * 2 ** 30 / total), 0)

if os.environ.get("Q4") == "1":
    def load_q4(model_id, adapter=None, device=None, dtype=None, revision=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                 bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, quantization_config=bnb,
                                                     dtype=torch.bfloat16, device_map={"": 0})
        if adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        return model, tok

    J.load_model_and_tokenizer = load_q4
    T.load_model_and_tokenizer = load_q4

if os.environ.get("MAX_OPTS"):
    import jevforge.data.mix as M
    from jevforge.data.schema import Example

    MAXN = int(os.environ["MAX_OPTS"])
    _build = M.build_mix

    def _cap(e):
        if len(e.options) <= MAXN:
            return e
        seed = int(hashlib.sha256((e.state + "|".join(e.options[:3])).encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        others = [i for i in range(len(e.options)) if i != e.gold]
        keep = sorted(rng.sample(others, MAXN - 1) + [e.gold])
        return Example(e.state, [e.options[i] for i in keep], keep.index(e.gold), e.instruction, e.source)

    def build_capped(*a, **k):
        exs = _build(*a, **k)
        out = [_cap(e) for e in exs]
        n = sum(1 for x, y in zip(exs, out) if x is not y)
        print(f"[jf_q4] capped options to {MAXN} in {n}/{len(exs)} training questions "
              f"(max options before: {max(len(e.options) for e in exs)})", flush=True)
        return out

    M.build_mix = build_capped

if os.environ.get("FORCE_MODE"):
    _mode = os.environ["FORCE_MODE"]
    J.select_mode = lambda model, enc: _mode
    print(f"[jf_q4b] scoring mode forced to {_mode}", flush=True)
print(f"[jf_q4] prompt={VARIANT} Q4={os.environ.get('Q4', '0')} MAX_OPTS={os.environ.get('MAX_OPTS', '-')} "
      f"CAP_GIB={os.environ.get('CAP_GIB', '-')}", flush=True)

from jevforge.cli import main  # noqa: E402

main(sys.argv[1:])
