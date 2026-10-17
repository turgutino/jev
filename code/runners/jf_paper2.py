"""Same as jf_paper.py (paperA prompt, jevforge untouched) plus a hard cap on this process's CUDA memory, so on a shared GPU
our process fails with OOM instead of squeezing another user's job. Cap = CAP_GIB (default 6) GiB.
Usage: CAP_GIB=6 python jf_paper2.py <jevforge args...>"""
import os
import sys

import torch

import jevforge.prompt as P

if torch.cuda.is_available():
    total = torch.cuda.get_device_properties(0).total_memory
    cap = float(os.environ.get("CAP_GIB", "6")) * 2**30
    torch.cuda.set_per_process_memory_fraction(min(1.0, cap / total), 0)

TAIL = "Select one option. Answer only with its bracketed numeric identifier."


def paperA(state, options, instruction=None):
    opts = [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]
    return "\n".join([f"State: {str(state).strip()}", "Options:"] + opts + [TAIL])


P.format_user_message = paperA
from jevforge.cli import main  # noqa: E402

main(sys.argv[1:])
