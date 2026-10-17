"""jevforge command with the paper-literal prompt (PROMPT_VARIANT=paperA, default) or jevforge's own prompt (PROMPT_VARIANT=default).
No governor, no throttling, no memory cap: maximum speed (second server, reserved for us). jevforge itself is not edited.
Usage: python jf_max.py <jevforge args...>"""
import os
import sys

import jevforge.prompt as P

TAIL = "Select one option. Answer only with its bracketed numeric identifier."


def paperA(state, options, instruction=None):
    opts = [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]
    return "\n".join([f"State: {str(state).strip()}", "Options:"] + opts + [TAIL])


VARIANT = os.environ.get("PROMPT_VARIANT", "paperA")
if VARIANT == "paperA":
    P.format_user_message = paperA
elif VARIANT != "default":
    sys.exit(f"unknown PROMPT_VARIANT {VARIANT}")
print(f"[jf_max] prompt variant: {VARIANT}", flush=True)

from jevforge.cli import main  # noqa: E402

main(sys.argv[1:])
