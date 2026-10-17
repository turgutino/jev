"""queue_q4e.json: 4B experiments with a 4-bit (QLoRA-style) base and <=20 options per training question (runner jf_q4b.py).
Scoring mode is forced to 'cache' (a transient OOM must never switch the mode silently). One job at a time.
Order: JevBench evaluations right after each training; ALL Banking77 evaluations at the end with a 10 GiB cap (a 77-option question needs
more than 8 GiB in cache mode; at the end the GPU is less crowded). Training caps 8 GiB (a 2-step test peaked at ~6.7 GB incl. context).
Deviations from the paper (report them): 4-bit nf4 base weights; <=20 options per training question; fla kernels installed; 4B only;
Intent-300 (an intermediate point between the paper's 100 steps and 1 epoch) is an addition."""
import json

M4 = "Qwen/Qwen3.5-4B"
Q = "results/q4"
REASON = "clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"
jobs = []
BASE_ENV = {"Q4": 1, "PROMPT_VARIANT": "paperA", "FORCE_MODE": "cache"}


def add(id, args, marker, env, requires=None):
    j = {"id": id, "runner": "jf_q4b.py", "args": args, "marker": marker, "need_mib": 9500, "env": {**BASE_ENV, **env}}
    if requires:
        j["requires"] = requires
    jobs.append(j)


def jevbench(prefix, adapter, out, label):
    ad = ["--adapter", adapter] if adapter else []
    add(f"{prefix}_jevbench", ["jevbench", "--model", M4, *ad, "--label", label, "--out", f"{Q}/{out}"],
        f"{Q}/{out}/jevbench/summary.json", {"CAP_GIB": 8}, f"{adapter}/adapter_model.safetensors" if adapter else None)


def b77(prefix, adapter, out, label):
    ad = ["--adapter", adapter] if adapter else []
    add(f"{prefix}_b77", ["eval", "--model", M4, *ad, "--label", label, "--tasks", "banking77", "--limit", "1000", "--readouts", "jev",
                          "--out", f"{Q}/{out}"], f"{Q}/{out}/eval/summary.json", {"CAP_GIB": 10},
        f"{adapter}/adapter_model.safetensors" if adapter else None)


def train(id, mix_args, out, label):
    add(id, ["train", "--model", M4, *mix_args, "--label", label, "--out", f"{Q}/{out}"], f"{Q}/{out}/adapter_model.safetensors",
        {"MAX_OPTS": 20, "CAP_GIB": 8})


jevbench("a1c_base", None, "base_c", "4B 4-bit base paperA (cache forced)")
train("a2_i100", ["--mix", "intent", "--max-examples", "3000", "--batch-size", "30"], "i100", "4B 4-bit intent-100 (opts<=20) paperA")
jevbench("a3_i100", f"{Q}/i100", "i100_eval", "4B 4-bit intent-100 paperA")
train("a4_r9k", ["--mix", REASON, "--max-examples", "9000"], "r9k", "4B 4-bit reason-9k (opts<=20) paperA")
jevbench("a5_r9k", f"{Q}/r9k", "r9k_eval", "4B 4-bit reason-9k paperA")
train("a6_i300", ["--mix", "intent", "--max-examples", "9000", "--batch-size", "30"], "i300", "4B 4-bit intent-300 (opts<=20) paperA")
jevbench("a7_i300", f"{Q}/i300", "i300_eval", "4B 4-bit intent-300 paperA")
b77("b1_base", None, "base_c", "4B 4-bit base paperA (cache forced)")
b77("b2_i100", f"{Q}/i100", "i100_eval", "4B 4-bit intent-100 paperA")
b77("b3_r9k", f"{Q}/r9k", "r9k_eval", "4B 4-bit reason-9k paperA")
b77("b4_i300", f"{Q}/i300", "i300_eval", "4B 4-bit intent-300 paperA")

ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids))
json.dump(jobs, open("queue_q4e.json", "w", encoding="utf-8"), indent=1)
print(len(jobs), "jobs in queue_q4e.json")
for j in jobs:
    print(f"{j['id']:18s} cap={j['env']['CAP_GIB']} opts={j['env'].get('MAX_OPTS', '-')} requires={j.get('requires', '-')}")
