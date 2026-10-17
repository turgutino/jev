"""queue_q4.json: 4B experiments with the 4-bit (QLoRA-style) base and at most 20 options per training question (jf_q4.py).
Order: 4-bit baseline, Intent-100 (+evals), Reason-9k (+evals), Intent-300 (+evals; an intermediate point between the paper's 100 steps and 1 epoch).
Declared 'need' is only for packing (the scheduler runs one job at a time here, MAX_TOTAL_MIB small). Training jobs cap their CUDA memory (CAP_GIB).
Deviations from the paper (report them): 4-bit nf4 base weights; <=20 options per training question; fla kernels installed; 4B only."""
import json

M4 = "Qwen/Qwen3.5-4B"
Q = "results/q4"
REASON = "clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"
jobs = []
RUN = "jf_q4.py"


def add(id, args, marker, need, env=None, requires=None):
    j = {"id": id, "runner": RUN, "args": args, "marker": marker, "need_mib": need,
         "env": {"Q4": 1, "PROMPT_VARIANT": "paperA", **(env or {})}}
    if requires:
        j["requires"] = requires
    jobs.append(j)


def evals(prefix, adapter, out, label):
    ad = ["--adapter", adapter] if adapter else []
    req = f"{adapter}/adapter_model.safetensors" if adapter else None
    add(f"{prefix}_jevbench", ["jevbench", "--model", M4, *ad, "--label", label, "--out", f"{Q}/{out}"],
        f"{Q}/{out}/jevbench/summary.json", 9500, {"CAP_GIB": 12}, req)
    add(f"{prefix}_b77", ["eval", "--model", M4, *ad, "--label", label, "--tasks", "banking77", "--limit", "1000", "--readouts", "jev",
                          "--out", f"{Q}/{out}"], f"{Q}/{out}/eval/summary.json", 9500, {"CAP_GIB": 12}, req)


def train(id, mix_args, out, label):
    add(id, ["train", "--model", M4, *mix_args, "--label", label, "--out", f"{Q}/{out}"], f"{Q}/{out}/adapter_model.safetensors", 9500,
        {"MAX_OPTS": 20, "CAP_GIB": 9})


evals("a1_base", None, "base", "4B 4-bit base paperA")
train("a2_i100", ["--mix", "intent", "--max-examples", "3000", "--batch-size", "30"], "i100", "4B 4-bit intent-100 (opts<=20) paperA")
evals("a3_i100", f"{Q}/i100", "i100_eval", "4B 4-bit intent-100 paperA")
train("a4_r9k", ["--mix", REASON, "--max-examples", "9000"], "r9k", "4B 4-bit reason-9k (opts<=20) paperA")
evals("a5_r9k", f"{Q}/r9k", "r9k_eval", "4B 4-bit reason-9k paperA")
train("a6_i300", ["--mix", "intent", "--max-examples", "9000", "--batch-size", "30"], "i300", "4B 4-bit intent-300 (opts<=20) paperA")
evals("a7_i300", f"{Q}/i300", "i300_eval", "4B 4-bit intent-300 paperA")

ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids))
json.dump(jobs, open("queue_q4.json", "w", encoding="utf-8"), indent=1)
print(len(jobs), "jobs in queue_q4.json")
for j in jobs:
    print(f"{j['id']:16s} need={j['need_mib']} env={j['env']} requires={j.get('requires', '-')}")
