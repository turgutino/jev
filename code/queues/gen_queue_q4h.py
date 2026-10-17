"""queue_q4h.json: second seed (20260923) for the two 4-bit 4B training runs that carry the main comparison
(Intent-100 and Reason-9k), each followed by its JevBench evaluation. Same settings as a2_i100 / a4_r9k (nf4 base, <=20 options,
Reason-9k with --max-len 450, paperA prompt, scoring mode forced to cache); only --seed differs. Everything goes to NEW directories
(results/q4h/...). It is started by a waiter that first waits until the 'g' scheduler (queue_q4g.json) has ended, so the two
queues never share the GPU.
Purpose: a single seed cannot tell a real difference from run-to-run noise; two seeds show how large that noise is."""
import json

M4 = "Qwen/Qwen3.5-4B"
H = "results/q4h"
REASON = "clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"
SEED = "20260923"
ENV = {"Q4": 1, "PROMPT_VARIANT": "paperA", "FORCE_MODE": "cache"}
jobs = []


def add(id, args, marker, env, requires=None):
    j = {"id": id, "runner": "jf_q4b.py", "args": args, "marker": marker, "need_mib": 9500, "env": env}
    if requires:
        j["requires"] = requires
    jobs.append(j)


def train(id, mix_args, out, label):
    add(id, ["train", "--model", M4, *mix_args, "--seed", SEED, "--label", label, "--out", f"{H}/{out}"],
        f"{H}/{out}/adapter_model.safetensors", {**ENV, "MAX_OPTS": 20, "CAP_GIB": 20})


def jevbench(id, adapter, out, label):
    add(id, ["jevbench", "--model", M4, "--adapter", adapter, "--label", label, "--out", f"{H}/{out}"],
        f"{H}/{out}/jevbench/summary.json", {**ENV, "CAP_GIB": 20}, f"{adapter}/adapter_model.safetensors")


train("h1_i100_s2", ["--mix", "intent", "--max-examples", "3000", "--batch-size", "30"], "i100_s2",
      "4B 4-bit intent-100 (opts<=20) paperA seed2")
jevbench("h2_i100_s2_jevbench", f"{H}/i100_s2", "i100_s2_eval", "4B 4-bit intent-100 paperA seed2")
train("h3_r9k_s2", ["--mix", REASON, "--max-examples", "9000", "--max-len", "450"], "r9k_s2",
      "4B 4-bit reason-9k (opts<=20) paperA seed2")
jevbench("h4_r9k_s2_jevbench", f"{H}/r9k_s2", "r9k_s2_eval", "4B 4-bit reason-9k paperA seed2")

ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids))
json.dump(jobs, open("queue_q4h.json", "w", encoding="utf-8"), indent=1)
print(len(jobs), "jobs in queue_q4h.json")
for j in jobs:
    print(f"{j['id']:22s} cap={j['env']['CAP_GIB']} opts={j['env'].get('MAX_OPTS', '-')} requires={j.get('requires', '-')}")
