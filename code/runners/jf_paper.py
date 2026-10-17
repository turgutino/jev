"""Run any jevforge CLI command with the paper-literal prompt (paperA, see prompt_audit.py). Does NOT edit jevforge.
Usage: python jf_paper.py <jevforge args...>   e.g.  python jf_paper.py train --model M --mix intent --out DIR
The same replacement is active for training and evaluation, so train and test prompts match."""
import sys

import jevforge.prompt as P

TAIL = "Select one option. Answer only with its bracketed numeric identifier."


def paperA(state, options, instruction=None):
    opts = [f"[{i}] {str(o).strip()}" for i, o in enumerate(options, start=1)]
    return "\n".join([f"State: {str(state).strip()}", "Options:"] + opts + [TAIL])


P.format_user_message = paperA
from jevforge.cli import main  # noqa: E402

main(sys.argv[1:])
