"""queue_q4i.json (replaces the waiting queue_q4h.json; same four seed-2 jobs, now preceded by a bf16 test).
Why: the paper-faithful bf16 Intent-100 job (d1_bf16_i100, no option cap) failed with CUDA out-of-memory three times on the
24 GB GPU, ALONE (22 GiB cap, tried to allocate 490 MiB in attention repeat_kv). So the bf16 test is repeated with the same
<=20-options cap that the 4-bit runs use: then the only difference to the 4-bit Intent-100 run is the numeric precision of the base
weights (bf16 vs nf4). retry_args adds --max-len 450 from the second try on (as for Reason-9k) in case a long example still overflows.
Then the seed-2 repeats of the two 4-bit runs (Intent-100, Reason-9k). New output directories only (results/q4i, results/q4h)."""
import json

M4 = "Qwen/Qwen3.5-4B"
H = "results/q4h"
E = "results/q4i"
REASON = "clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter"
SEED = "20260923"
ENV4 = {"Q4": 1, "PROMPT_VARIANT": "paperA", "FORCE_MODE": "cache"}
ENVB = {"PROMPT_VARIANT": "paperA", "FORCE_MODE": "cache"}
jobs = []


def add(id, args, marker, env, requires=None, retry=None):
    j = {"id": id, "runner": "jf_q4b.py", "args": args, "marker": marker, "need_mib": 9500, "env": env}
    if requires:
        j["requires"] = requires
    if retry:
        j["retry_args"] = retry
    jobs.append(j)


def train(id, base, mix_args, out, label, env, retry=None):
    add(id, ["train", "--model", M4, *mix_args, "--label", label, "--out", f"{base}/{out}"],
        f"{base}/{out}/adapter_model.safetensors", env, retry=retry)


def jevbench(id, base, adapter, out, label, env):
    add(id, ["jevbench", "--model", M4, "--adapter", adapter, "--label", label, "--out", f"{base}/{out}"],
        f"{base}/{out}/jevbench/summary.json", env, f"{adapter}/adapter_model.safetensors")


# bf16 (no quantisation) with the same <=20-options cap as the 4-bit runs
train("e1_bf16_i100_c20", E, ["--mix", "intent", "--max-examples", "3000", "--batch-size", "30"], "bf16_i100_c20",
      "4B bf16 intent-100 (opts<=20) paperA", {**ENVB, "MAX_OPTS": 20, "CAP_GIB": 22}, retry=["--max-len", "450"])
jevbench("e2_bf16_i100_c20_jevbench", E, f"{E}/bf16_i100_c20", "bf16_i100_c20_eval", "4B bf16 intent-100 (opts<=20) paperA",
         {**ENVB, "CAP_GIB": 22})

# second seed of the two 4-bit runs
train("h1_i100_s2", H, ["--mix", "intent", "--max-examples", "3000", "--batch-size", "30", "--seed", SEED], "i100_s2",
      "4B 4-bit intent-100 (opts<=20) paperA seed2", {**ENV4, "MAX_OPTS": 20, "CAP_GIB": 20})
jevbench("h2_i100_s2_jevbench", H, f"{H}/i100_s2", "i100_s2_eval", "4B 4-bit intent-100 paperA seed2", {**ENV4, "CAP_GIB": 20})
train("h3_r9k_s2", H, ["--mix", REASON, "--max-examples", "9000", "--max-len", "450", "--seed", SEED], "r9k_s2",
      "4B 4-bit reason-9k (opts<=20) paperA seed2", {**ENV4, "MAX_OPTS": 20, "CAP_GIB": 20})
jevbench("h4_r9k_s2_jevbench", H, f"{H}/r9k_s2", "r9k_s2_eval", "4B 4-bit reason-9k paperA seed2", {**ENV4, "CAP_GIB": 20})

ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids))
json.dump(jobs, open("queue_q4i.json", "w", encoding="utf-8"), indent=1)
print(len(jobs), "jobs in queue_q4i.json")
for j in jobs:
    print(f"{j['id']:28s} cap={j['env']['CAP_GIB']} q4={j['env'].get('Q4', 0)} opts={j['env'].get('MAX_OPTS', '-')} "
          f"retry={j.get('retry_args', '-')} requires={j.get('requires', '-')}")
