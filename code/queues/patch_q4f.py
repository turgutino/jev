import json
q = json.load(open("queue_q4f.json", encoding="utf-8"))
for j in q:
    if j["id"] == "a4_r9k":
        j["env"]["CAP_GIB"] = 9
json.dump(q, open("queue_q4f.json", "w", encoding="utf-8"), indent=1)
a4 = [j for j in q if j["id"] == "a4_r9k"][0]
print(a4["args"][-6:], a4["env"])
