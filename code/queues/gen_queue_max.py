"""queue_max.json (full plan for the second server, max-speed scheduler) from queue_b.json, plus test_queue_max.json (tiny test).
Changes vs queue_b.json: drop the 4B probe (peak memory is recorded per job instead), drop idle_only / pause_queue_after, mark all
4B jobs exclusive, make every evaluation wait for its adapter ("requires"), and add --grad-ckpt as a retry argument for 4B training."""
import json

b = json.load(open("queue_b.json", encoding="utf-8"))
out = []
for j in b:
    if j["id"] == "q10_4b_probe":
        continue
    j = {k: v for k, v in j.items() if k not in ("idle_only", "pause_queue_after")}
    is4b = "Qwen/Qwen3.5-4B" in j["args"]
    if is4b:
        j["exclusive"] = True
        if j["args"][0] == "train":
            j["retry_args"] = ["--grad-ckpt"]
    if "--adapter" in j["args"]:
        j["requires"] = j["args"][j["args"].index("--adapter") + 1] + "/adapter_model.safetensors"
    out.append(j)
ids = [j["id"] for j in out]
assert len(ids) == len(set(ids))
json.dump(out, open("queue_max.json", "w", encoding="utf-8"), indent=1)
print(len(out), "jobs in queue_max.json")

M06 = "Qwen/Qwen3-0.6B"
Q = "results/qt"
test = [
    {"id": "t1_train", "args": ["train", "--model", M06, "--mix", "intent", "--max-examples", "64", "--batch-size", "8", "--max-steps", "2",
                                "--decontaminate", "none", "--save-every", "0", "--out", f"{Q}/t1"],
     "marker": f"{Q}/t1/adapter_model.safetensors", "need_mib": 9000},
    {"id": "t2_eval_adapter", "args": ["jevbench", "--model", M06, "--adapter", f"{Q}/t1", "--limit", "5", "--label", "t2", "--out", f"{Q}/t2"],
     "marker": f"{Q}/t2/jevbench/summary.json", "need_mib": 3500, "requires": f"{Q}/t1/adapter_model.safetensors"},
    {"id": "t3_fail", "args": ["jevbench", "--model", "no/such-model", "--limit", "5", "--out", f"{Q}/t3"],
     "marker": f"{Q}/t3/jevbench/summary.json", "need_mib": 3500},
    {"id": "t4_eval_dep_on_fail", "args": ["jevbench", "--model", M06, "--adapter", f"{Q}/t3", "--limit", "5", "--out", f"{Q}/t4"],
     "marker": f"{Q}/t4/jevbench/summary.json", "need_mib": 3500, "requires": f"{Q}/t3/adapter_model.safetensors"},
    {"id": "t5_eval_base", "args": ["jevbench", "--model", M06, "--limit", "5", "--label", "t5", "--out", f"{Q}/t5"],
     "marker": f"{Q}/t5/jevbench/summary.json", "need_mib": 3500},
]
json.dump(test, open("test_queue_max.json", "w", encoding="utf-8"), indent=1)
print(len(test), "jobs in test_queue_max.json")
