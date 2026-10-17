"""queue_b.json for the second server = queue.json without q01-q03 (those stay on the first server).
Order is unchanged: 4B memory probe (queue stops after it for review), 4B baselines / Intent-100 / Reason-9k, then 0.6B full intent epoch
and seeds, 4B full intent epoch last. All paths are relative to the run dir (/root/turgutino/jev)."""
import json

q = json.load(open("queue.json", encoding="utf-8"))
b = [j for j in q if not j["id"].startswith(("q01", "q02", "q03"))]
assert b[0]["id"] == "q10_4b_probe"
b[0]["need_mib"] = 22000          # empty GPU: let the probe use almost everything so the real peak can be measured
json.dump(b, open("queue_b.json", "w", encoding="utf-8"), indent=1)
print(len(b), "jobs in queue_b.json; first:", b[0]["id"], "; last:", b[-1]["id"])
