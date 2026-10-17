"""Generate queue.json (full plan) and test_queue.json (tiny scheduler test). Paths are relative to the run dir (scheduler cwd).
Needs (MiB): 0.6B train 9000 (measured peak 7777 + 15%), 0.6B jevbench 3500, 0.6B banking77 6000 (an earlier 6 GB cap worked),
4B numbers are GUESSES (train 16000, eval 11000) until the probe job has measured them; the queue stops after the probe."""
import json

M06, M4 = "Qwen/Qwen3-0.6B", "Qwen/Qwen3.5-4B"
REASON = "clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"
Q = "results/q"
jobs = []


def add(id, args, marker, need, **extra):
    jobs.append({"id": id, "args": args, "marker": marker, "need_mib": need, **extra})


def train(id, model, mix_args, out, need, label, seed=None, idle=False):
    args = ["train", "--model", model, *mix_args, "--label", label, "--out", f"{Q}/{out}"]
    if seed:
        args += ["--seed", str(seed)]
    add(id, args, f"{Q}/{out}/adapter_model.safetensors", need, **({"idle_only": True} if idle else {}))


def evals(prefix, model, adapter, out, need_jb, need_b77, label, idle=False, env=None):
    ex = {**({"idle_only": True} if idle else {}), **({"env": env} if env else {})}
    ad = ["--adapter", adapter] if adapter else []
    add(f"{prefix}_jevbench", ["jevbench", "--model", model, *ad, "--label", label, "--out", f"{Q}/{out}"],
        f"{Q}/{out}/jevbench/summary.json", need_jb, **ex)
    add(f"{prefix}_b77", ["eval", "--model", model, *ad, "--label", label, "--tasks", "banking77", "--limit", "1000",
                          "--readouts", "jev", "--out", f"{Q}/{out}"], f"{Q}/{out}/eval/summary.json", need_b77, **ex)


I100 = ["--mix", "intent", "--max-examples", "3000", "--batch-size", "30"]
R9K = ["--mix", REASON, "--max-examples", "9000"]
INT = ["--mix", "intent"]
T06, JB06, B06 = 9000, 3500, 6000
T4, E4 = 16000, 11000          # GUESSES until the probe has run

# --- 1) 0.6B quick jobs -------------------------------------------------------------------------------------------------
evals("q01_i1ep96", M06, "results/chain3/intent1ep/checkpoint-last", "i1ep96_eval", JB06, B06, "0.6B intent-1ep ckpt~3100 paperA")
jobs.pop(0)   # the JevBench result of this checkpoint already exists (chain4); only Banking77 (1000 items) is missing
train("q02_i100_06", M06, I100, "i100_06", T06, "0.6B intent-100 paperA")
evals("q03_i100_06", M06, f"{Q}/i100_06", "i100_06_eval", JB06, B06, "0.6B intent-100 paperA")

# --- 2) 4B memory probe; scheduler exits after it so the guessed needs can be corrected ---------------------------------
add("q10_4b_probe", ["train", "--model", M4, "--mix", "intent", "--max-examples", "64", "--batch-size", "8", "--max-steps", "2",
                     "--decontaminate", "none", "--save-every", "0", "--out", f"{Q}/probe4b"],
    f"{Q}/probe4b/adapter_model.safetensors", 21000, idle_only=True, pause_queue_after=True)

# --- 3) 4B main experiment (paper's 4B setting) ---------------------------------------------------------------------------
evals("q11_4b_base_default", M4, None, "4b_base_default", E4, E4, "4B base jevforge-default prompt", idle=True, env={"PROMPT_VARIANT": "default"})
evals("q12_4b_base_paperA", M4, None, "4b_base_paperA", E4, E4, "4B base paperA", idle=True)
train("q13_4b_i100", M4, I100, "4b_i100", T4, "4B intent-100 paperA", idle=True)
evals("q14_4b_i100", M4, f"{Q}/4b_i100", "4b_i100_eval", E4, E4, "4B intent-100 paperA", idle=True)
train("q15_4b_r9k", M4, R9K, "4b_r9k", T4, "4B reason-9k paperA", idle=True)
evals("q16_4b_r9k", M4, f"{Q}/4b_r9k", "4b_r9k_eval", E4, E4, "4B reason-9k paperA", idle=True)

# --- 4) 0.6B completion and extra seeds -------------------------------------------------------------------------------------
train("q20_i1ep_full_06", M06, INT, "i1ep_full_06", T06, "0.6B intent-1ep full paperA")
evals("q21_i1ep_full_06", M06, f"{Q}/i1ep_full_06", "i1ep_full_06_eval", JB06, B06, "0.6B intent-1ep full paperA")
for s, tag in ((20261001, "s2"), (20261002, "s3")):
    train(f"q22_r9k_{tag}_06", M06, R9K, f"r9k_{tag}_06", T06, f"0.6B reason-9k {tag} paperA", seed=s)
    evals(f"q23_r9k_{tag}_06", M06, f"{Q}/r9k_{tag}_06", f"r9k_{tag}_06_eval", JB06, B06, f"0.6B reason-9k {tag} paperA")
train("q24_i1ep_s2_06", M06, INT, "i1ep_s2_06", T06, "0.6B intent-1ep s2 paperA", seed=20261001)
evals("q25_i1ep_s2_06", M06, f"{Q}/i1ep_s2_06", "i1ep_s2_06_eval", JB06, B06, "0.6B intent-1ep s2 paperA")

# --- 5) very long, last: 4B full intent epoch (can be removed from the file) ------------------------------------------------
train("q40_4b_i1ep", M4, INT, "4b_i1ep", T4, "4B intent-1ep paperA", idle=True)
evals("q41_4b_i1ep", M4, f"{Q}/4b_i1ep", "4b_i1ep_eval", E4, E4, "4B intent-1ep paperA", idle=True)

json.dump(jobs, open("queue.json", "w", encoding="utf-8"), indent=1)
ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids)), "duplicate job ids"
print(len(jobs), "jobs written to queue.json")

# tiny scheduler test: two small evaluations (no training), second one fails on purpose to test retry/skip logic
test = [
    {"id": "t_eval_a", "args": ["jevbench", "--model", M06, "--limit", "5", "--label", "sched-test-a", "--out", f"{Q}_test/a"],
     "marker": f"{Q}_test/a/jevbench/summary.json", "need_mib": 3500},
    {"id": "t_fail_b", "args": ["jevbench", "--model", "no/such-model", "--limit", "5", "--out", f"{Q}_test/b"],
     "marker": f"{Q}_test/b/jevbench/summary.json", "need_mib": 3500},
    {"id": "t_eval_c", "args": ["jevbench", "--model", M06, "--limit", "5", "--label", "sched-test-c", "--out", f"{Q}_test/c"],
     "marker": f"{Q}_test/c/jevbench/summary.json", "need_mib": 3500},
]
json.dump(test, open("test_queue.json", "w", encoding="utf-8"), indent=1)
print(len(test), "jobs written to test_queue.json")
