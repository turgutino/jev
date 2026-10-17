# Replication of the "intent trap" of LLM-as-Jev (arXiv 2610.02076 v3)

Start with **[REPORT.md](REPORT.md)** (results, deviations, limitations). Everything in the report that is marked [V] can be re-computed from this repository.

## Layout

| path | content |
|---|---|
| `REPORT.md` | the report (English) |
| `docs/REPRO_PLAN_preregistration.md` | hypotheses, metrics, decision rules and deviations log, written before the first training run |
| `analysis/analyze_jevbench_run.py` | per-item JevBench analysis (accuracy, ECE, "yes" count, catch-all mass, paired test); validated against the harness `summary.json` |
| `analysis/analyze_4b_full.py` → `results_4b.md` | Qwen3.5-4B: tables, paired tests, "yes"/catch-all, failure analysis, temperature scaling |
| `analysis/analyze_06b.py` → `results_06b.md` | Qwen3-0.6B: tables and paired tests |
| `results/server_A`, `results/server_B` | per-item outputs (`results.jsonl` for JevBench, `banking77.jsonl`), `summary.json`, `train_log.jsonl`, `run.json` of every reported run. No adapters and no training data (they are large / regenerable) |
| `code/runners/` | thin drivers around `jevforge` (monkey-patches only, jevforge itself is untouched): `jf_max.py`, `jf_q4b.py` (4-bit loader, option cap, forced scoring mode, memory cap), prompt variants |
| `code/schedulers/` | job queue runners used on the shared GPU servers (`scheduler_max4.py` is the last one) |
| `code/queues/` | queue generators and queue files (`queue_q4f/g/i.json` are the final 4B queues; earlier files are superseded) |
| `code/launch/` | shell scripts used to start chains/queues |

## Re-computing the numbers

Needs Python 3 with `numpy` only.

```bash
python analysis/analyze_06b.py out_06b.md     # ~1 min
python analysis/analyze_4b_full.py out_4b.md  # ~2 min (bootstrap + cross-validation)
diff out_4b.md analysis/results_4b.md         # identical
```

## Re-running the experiments (not needed to check the numbers)

1. `git clone https://github.com/pedroliveiraAI/LLMtoSystemOne` at commit `8986e52` (JevBench submodule `1bcc55e`) and install it in a venv (`code/launch/setup_b*.sh` show what we did on the servers; they contain site-specific paths).
2. Edit the paths at the top of the scheduler/queue files, then e.g. `PROMPT_VARIANT=paperA Q4=1 MAX_OPTS=20 python code/runners/jf_q4b.py train --model Qwen/Qwen3.5-4B --mix intent --max-examples 3000 --batch-size 30 --out results/q4/i100`.
3. Evaluate with `jf_q4b.py jevbench --adapter <dir> --out <dir>` and `jf_q4b.py eval --tasks banking77 --limit 1000 --readouts jev --out <dir>`.

Environment variables of the runners: `PROMPT_VARIANT` (`paperA` = paper-literal prompt, `default` = jevforge's own), `Q4=1` (nf4 4-bit base), `MAX_OPTS=N` (≤ N options per training question),
`CAP_GIB` (CUDA memory cap), `FORCE_MODE` (scoring mode, `cache`).

## Provenance and honesty notes

* No API keys, credentials, hostnames or ports are stored in this repository.
* The `code/` folder is the code as it was used; some files are superseded or only used for failed attempts (see the report, §8). Failed jobs are visible in the queue files, their outputs are not part of the reported results.
* Paper numbers quoted in the report were read from the paper's PDF text by us; the PDF itself is not redistributed here.
