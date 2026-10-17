"""jf_polite.py + env PROMPT_VARIANT (paperA = paper-literal prompt, default = jevforge's own prompt). jevforge is not edited.
Pause points: before every training example (jevforge.train.encode) and before every scored item (Jev.scores).
Env: POLITE_NEED_MIB, POLITE_RESERVE_MIB, POLITE_DUTY, POLITE_LOG, PROMPT_VARIANT (default paperA)."""
import os
import sys

import jevforge.prompt as P
import jevforge.train as T
from jevforge.jev import Jev

import polite_gpu

TAIL = "Select one option. Answer only with its bracketed numeric identifier."


def paperA(state, options, instruction=None):
    opts = [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]
    return "\n".join([f"State: {str(state).strip()}", "Options:"] + opts + [TAIL])


VARIANT = os.environ.get("PROMPT_VARIANT", "paperA")
if VARIANT == "paperA":
    P.format_user_message = paperA
elif VARIANT != "default":
    sys.exit(f"unknown PROMPT_VARIANT {VARIANT}")
print(f"[jf_polite2] prompt variant: {VARIANT}", flush=True)

G = polite_gpu.install_torch_governor(polite_gpu.policy_from_env(), log_path=os.environ.get("POLITE_LOG"))

_encode = T.encode
_scores = Jev.scores


def encode_hook(*a, **k):
    G.checkpoint()
    return _encode(*a, **k)


def scores_hook(self, *a, **k):
    G.checkpoint()
    return _scores(self, *a, **k)


T.encode = encode_hook
Jev.scores = scores_hook

from jevforge.cli import main  # noqa: E402

main(sys.argv[1:])
