"""queue_q4g.json: re-run of the 4B jobs that died of CUDA out-of-memory on 2026-10-17 (the GPU was shared with the 0.6B jobs then),
plus a bf16 (non-quantised, no option cap) 4B Intent-100 test. Everything goes to NEW output directories (results/q4g/...,
results/q4/i300b); nothing existing is overwritten. The GPU is empty now, so the memory caps are large. One job at a time.
Order: Intent-300 retrain + JevBench, bf16 baseline and bf16 Intent-100 (+JevBench), then the four Banking77 evaluations.
Deviations from the paper (report them): the 4-bit jobs use nf4 base weights and <=20 options per training question;
the bf16 jobs (d*) use neither; fla kernels installed; single seed; scoring mode forced to 'cache'."""
import json

M4 = "Qwen/Qwen3.5-4B"
G = "results/q4g"
Q = "results/q4"
RUN = "jf_q4b.py"
Q4_ENV = {"Q4": 1, "PROMPT_VARIANT": "paperA", "FORCE_MODE": "cache"}
BF_ENV = {"PROMPT_VARIANT": "paperA", "FORCE_MODE": "cache"}
jobs = []


def add(id, args, marker, env, requires=None):
    j = {"id": id, "runner": RUN, "args": args, "marker": marker, "need_mib": 9500, "env": env}
    if requires:
        j["requires"] = requires
    jobs.append(j)


def jevbench(id, adapter, out, label, env, cap):
    ad = ["--adapter", adapter] if adapter else []
    add(id, ["jevbench", "--model", M4, *ad, "--label", label, "--out", f"{G}/{out}"], f"{G}/{out}/jevbench/summary.json",
        {**env, "CAP_GIB": cap}, f"{adapter}/adapter_model.safetensors" if adapter else None)


def b77(id, adapter, out, label):
    ad = ["--adapter", adapter] if adapter else []
    add(id, ["eval", "--model", M4, *ad, "--label", label, "--tasks", "banking77", "--limit", "1000", "--readouts", "jev",
             "--out", f"{G}/{out}"], f"{G}/{out}/eval/summary.json", {**Q4_ENV, "CAP_GIB": 20},
        f"{adapter}/adapter_model.safetensors" if adapter else None)


# 4-bit: Intent-300 again (same settings as the failed a6_i300), new output directory
add("c1_i300b", ["train", "--model", M4, "--mix", "intent", "--max-examples", "9000", "--batch-size", "30",
                 "--label", "4B 4-bit intent-300 (opts<=20) paperA, rerun", "--out", f"{Q}/i300b"],
    f"{Q}/i300b/adapter_model.safetensors", {**Q4_ENV, "MAX_OPTS": 20, "CAP_GIB": 20})
jevbench("c2_i300b_jevbench", f"{Q}/i300b", "i300b_eval", "4B 4-bit intent-300 paperA", Q4_ENV, 20)

# bf16, no option cap: the closest to the paper
jevbench("d3_bf16_base_jevbench", None, "bf16_base", "4B bf16 base paperA (cache forced)", BF_ENV, 22)
add("d1_bf16_i100", ["train", "--model", M4, "--mix", "intent", "--max-examples", "3000", "--batch-size", "30",
                     "--label", "4B bf16 intent-100 (no option cap) paperA", "--out", f"{G}/bf16_i100"],
    f"{G}/bf16_i100/adapter_model.safetensors", {**BF_ENV, "CAP_GIB": 22})
jevbench("d2_bf16_i100_jevbench", f"{G}/bf16_i100", "bf16_i100_eval", "4B bf16 intent-100 paperA", BF_ENV, 22)

# Banking77 (4-bit), new output directories
b77("b1_base", None, "b77_base", "4B 4-bit base paperA (cache forced)")
b77("b2_i100", f"{Q}/i100", "b77_i100", "4B 4-bit intent-100 paperA")
b77("b3_r9k", f"{Q}/r9k", "b77_r9k", "4B 4-bit reason-9k paperA")
b77("b4_i300b", f"{Q}/i300b", "b77_i300b", "4B 4-bit intent-300 paperA")

ids = [j["id"] for j in jobs]
assert len(ids) == len(set(ids))
json.dump(jobs, open("queue_q4g.json", "w", encoding="utf-8"), indent=1)
print(len(jobs), "jobs in queue_q4g.json")
for j in jobs:
    print(f"{j['id']:24s} cap={j['env']['CAP_GIB']} q4={j['env'].get('Q4', 0)} opts={j['env'].get('MAX_OPTS', '-')} requires={j.get('requires', '-')}")
