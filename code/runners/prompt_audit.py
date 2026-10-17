"""Prompt-sensitivity driver for the JevBench baseline (H0 audit). Does NOT edit jevforge: it replaces
jevforge.prompt.format_user_message at run time, then calls the normal `jevforge jevbench` command.
Usage: PV=default|paperA|paperB python prompt_audit.py MODEL OUT_DIR
 default = jevforge default prompt (what stage A used)
 paperA  = paper Sec.3.2 literal: "State: {s}" / "Options:" lines / "Select one option. Answer only with its bracketed numeric identifier." (item instruction dropped)
 paperB  = paperA but the item's own instruction kept after the state
The exact newline layout of the paper template and JevBench's real prompt are UNKNOWN; these are candidates only."""
import os
import sys

import jevforge.prompt as P

PV = os.environ.get("PV", "default")
TAIL = "Select one option. Answer only with its bracketed numeric identifier."


def _opts(options):
    return [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]


def paperA(state, options, instruction=None):
    return "\n".join([f"State: {str(state).strip()}", "Options:"] + _opts(options) + [TAIL])


def paperB(state, options, instruction=None):
    mid = [(instruction or P.DEFAULT_INSTRUCTION).strip()]
    return "\n".join([f"State: {str(state).strip()}"] + mid + ["Options:"] + _opts(options) + [TAIL])


if PV == "paperA":
    P.format_user_message = paperA
elif PV == "paperB":
    P.format_user_message = paperB
elif PV != "default":
    sys.exit(f"unknown PV {PV}")

from jevforge.cli import main  # noqa: E402

model, out = sys.argv[1], sys.argv[2]
main(["jevbench", "--model", model, "--label", f"{model} prompt={PV}", "--out", out])
