"""Full analysis of the 4B runs (JevBench-231 and Banking77-1000) for the 'intent trap' study.
Uses the validated per-item helpers of analyze_jevbench_run.py (ECE with 10 bins = harness definition, sign test, bootstrap).

Sections
 1. accuracy / ECE of every run, next to the harness summary.json (sanity check)
 2. paired tests: tuned vs the baseline of the SAME precision (4-bit vs 4-bit, bf16 vs bf16) and seed-to-seed noise
 3. 'yes' bias on the binary yes/no items and catch-all ('none of these' style) options on JevBench
 4. failure analysis: per-family accuracy, most confident wrong answers, lost/won items; Banking77 top confusions
 5. post-hoc temperature scaling: T fitted on Banking77 and applied to JevBench (transfer), and 5-fold CV inside JevBench.
    T only rescales confidence (argmax unchanged), so accuracy stays the same and only ECE moves.
Usage: python analyze_4b_full.py [OUT.md]
Everything printed is computed from the files in ROOT; nothing is typed in by hand."""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_jevbench_run as A  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "results" / "server_B"
JB = {
    "4bit_base": "q4/base_c/jevbench",
    "4bit_i100": "q4/i100_eval/jevbench",
    "4bit_i100_s2": "q4h/i100_s2_eval/jevbench",
    "4bit_i300": "q4g/i300b_eval/jevbench",
    "4bit_r9k": "q4/r9k_eval/jevbench",
    "4bit_r9k_s2": "q4h/r9k_s2_eval/jevbench",
    "bf16_base": "q4g/bf16_base/jevbench",
    "bf16_i100c20": "q4i/bf16_i100_c20_eval/jevbench",
}
B77 = {
    "4bit_base": "q4g/b77_base/eval",
    "4bit_i100": "q4g/b77_i100/eval",
    "4bit_r9k": "q4g/b77_r9k/eval",
    "4bit_i300": "q4g/b77_i300b/eval",
}
PAIRS = [  # (baseline, tuned)
    ("4bit_base", "4bit_i100"), ("4bit_base", "4bit_i100_s2"), ("4bit_base", "4bit_i300"),
    ("4bit_base", "4bit_r9k"), ("4bit_base", "4bit_r9k_s2"), ("bf16_base", "bf16_i100c20"),
    ("4bit_i100", "4bit_i100_s2"), ("4bit_r9k", "4bit_r9k_s2"),   # seed-to-seed noise
    ("4bit_i100", "4bit_r9k"),                                    # Intent vs broader data
]
B77_PAIRS = [("4bit_base", "4bit_i100"), ("4bit_base", "4bit_i300"), ("4bit_base", "4bit_r9k"), ("4bit_i100", "4bit_r9k")]
rng = np.random.default_rng(0)
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


# ---------------------------------------------------------------- loading
def load_jb(name):
    p = ROOT / JB[name] / "results.jsonl"
    return A.load(p) if p.exists() else None


def load_b77(name):
    p = ROOT / B77[name] / "banking77.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    probs = np.array([r["jev"] for r in rows], dtype=float)
    gold = np.array([r["gold"] for r in rows])
    return {"probs": probs, "gold": gold, "conf": probs.max(1), "pred": probs.argmax(1), "correct": probs.argmax(1) == gold}


def jb_arrays(rows):
    ids = sorted(rows)
    conf = np.array([A.conf_and_correct(rows[i])[0] for i in ids], dtype=float)
    corr = np.array([bool(rows[i]["correct"]) for i in ids])
    return ids, conf, corr


def paired(ca, cb, conf_a, conf_b, n_boot=4000):
    won, lost = int((~ca & cb).sum()), int((ca & ~cb).sum())
    p = A.sign_test(won, lost)
    idx = np.arange(len(ca))
    d_acc, d_ece = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx))
        d_acc.append(cb[s].mean() - ca[s].mean())
        d_ece.append(A.ece(conf_b[s], cb[s].astype(float), 10) - A.ece(conf_a[s], ca[s].astype(float), 10))
    q = lambda d: tuple(np.percentile(d, [2.5, 97.5]))  # noqa: E731
    return {"won": won, "lost": lost, "p": p, "dacc": cb.mean() - ca.mean(), "dacc_ci": q(d_acc),
            "dece": A.ece(conf_b, cb.astype(float), 10) - A.ece(conf_a, ca.astype(float), 10), "dece_ci": q(d_ece)}


def verdict(ci):
    return "differs from 0" if (ci[0] > 0 or ci[1] < 0) else "CI contains 0"


# ---------------------------------------------------------------- 1. summary
JBR = {k: load_jb(k) for k in JB}
JBR = {k: v for k, v in JBR.items() if v is not None}
say("# 4B analysis (all numbers computed from the per-item files)\n")
say("## 1. JevBench-231: accuracy and ECE (10 bins) -- ours vs harness summary.json\n")
say("| run | n | acc | ECE10 | harness acc | harness ECE | conf on wrong | wrong with conf>=0.9 |")
say("|---|---|---|---|---|---|---|---|")
for k, rows in JBR.items():
    s = A.summarize(rows)
    sj = json.loads((ROOT / JB[k] / "summary.json").read_text(encoding="utf-8"))
    he = sj["ece"]["ece"] if isinstance(sj["ece"], dict) else sj["ece"]
    say(f"| {k} | {s['n']} | {s['acc']:.4f} | {s['ece10']:.4f} | {sj['accuracy']:.4f} | {he:.4f} | {s['conf_wrong']:.3f} | {s['wrong_ge90']:.1%} |")

B77R = {k: load_b77(k) for k in B77}
B77R = {k: v for k, v in B77R.items() if v is not None}
if B77R:
    g0 = next(iter(B77R.values()))["gold"]
    assert all((v["gold"] == g0).all() for v in B77R.values()), "Banking77 questions differ between runs"
    say("\n## 1b. Banking77 (1000 questions, 77 options): accuracy and ECE (10 bins)\n")
    say("| run | acc | ECE10 | mean confidence | mean conf - acc |")
    say("|---|---|---|---|---|")
    for k, v in B77R.items():
        e = A.ece(v["conf"], v["correct"].astype(float), 10)
        say(f"| {k} | {v['correct'].mean():.4f} | {e:.4f} | {v['conf'].mean():.3f} | {v['conf'].mean() - v['correct'].mean():+.3f} |")

# ---------------------------------------------------------------- 2. paired
say("\n## 2. Paired tests on JevBench (A = baseline of the same precision, B = tuned; 95% bootstrap CI, 4000 resamples)\n")
say("| A -> B | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |")
say("|---|---|---|---|---|---|")
for a, b in PAIRS:
    if a not in JBR or b not in JBR:
        continue
    ids_a, conf_a, corr_a = jb_arrays(JBR[a])
    ids_b, conf_b, corr_b = jb_arrays(JBR[b])
    assert ids_a == ids_b
    r = paired(corr_a, corr_b, conf_a, conf_b)
    say(f"| {a} -> {b} | {r['won']} / {r['lost']} | {r['p']:.3f} | {r['dacc']:+.3f} [{r['dacc_ci'][0]:+.3f}, {r['dacc_ci'][1]:+.3f}] "
        f"| {r['dece']:+.3f} [{r['dece_ci'][0]:+.3f}, {r['dece_ci'][1]:+.3f}] | {verdict(r['dece_ci'])} |")

if B77R:
    say("\n## 2b. Paired tests on Banking77\n")
    say("| A -> B | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |")
    say("|---|---|---|---|---|---|")
    for a, b in B77_PAIRS:
        if a not in B77R or b not in B77R:
            continue
        x, y = B77R[a], B77R[b]
        r = paired(x["correct"], y["correct"], x["conf"], y["conf"])
        say(f"| {a} -> {b} | {r['won']} / {r['lost']} | {r['p']:.4f} | {r['dacc']:+.3f} [{r['dacc_ci'][0]:+.3f}, {r['dacc_ci'][1]:+.3f}] "
            f"| {r['dece']:+.3f} [{r['dece_ci'][0]:+.3f}, {r['dece_ci'][1]:+.3f}] | {verdict(r['dece_ci'])} |")

# ---------------------------------------------------------------- 3. yes bias / catch-all
say("\n## 3. 'yes' bias and catch-all options (JevBench)\n")
say("'yes' bias is measured on the binary yes/no items: predicted yes, gold yes (= correct yes + wrong no), mean P(yes). "
    "Catch-all = options named other/unknown/none/none of these/...; we report how many items have one, the mean probability mass on it, "
    "and how often it is the prediction.\n")
say("| run | binary items | predicted yes | gold yes | mean P(yes) | items with catch-all | mean catch-all mass | predicted catch-all |")
say("|---|---|---|---|---|---|---|---|")
for k, rows in JBR.items():
    s = A.summarize(rows)
    say(f"| {k} | {s['binary_items']} | {s['yes_pred']} | {s['yes_gold']} | {s['mean_p_yes']:.3f} | {s['items_with_catchall']} | "
        f"{s['mean_catchall_mass']:.3f} | {s['pred_catchall']} |")

# ---------------------------------------------------------------- 4. failures
say("\n## 4. Failure analysis\n")
say("### 4a. Accuracy and ECE per family (JevBench)\n")
fams = sorted({r["family"] for r in next(iter(JBR.values())).values()})
hdr = "| family | n | " + " | ".join(JBR) + " |"
say(hdr)
say("|---|---|" + "---|" * len(JBR))
for f in fams:
    cells = []
    n = 0
    for k, rows in JBR.items():
        sub = [r for r in rows.values() if r["family"] == f]
        n = len(sub)
        cells.append(f"{np.mean([bool(r['correct']) for r in sub]):.2f}")
    say(f"| {f} | {n} | " + " | ".join(cells) + " |")
say("\n(accuracy per family; cells are per run, same order as the header)\n")

say("### 4b. Most confident wrong answers (top 8 per tuned run)\n")
for k in ("4bit_i100", "4bit_i300", "4bit_r9k", "bf16_i100c20"):
    if k not in JBR:
        continue
    wrong = [(max((r.get("probs") or {"-": 0}).values()), tid, r["family"], r["predicted"]) for tid, r in JBR[k].items() if not r["correct"]]
    wrong.sort(reverse=True)
    say(f"**{k}** ({len(wrong)} wrong):")
    for conf, tid, fam, pred in wrong[:8]:
        say(f"- {tid} ({fam}) predicted `{pred}` with confidence {conf:.3f}")
    say()

say("### 4c. Items won/lost versus the matching baseline, by family\n")
say("| A -> B | family | won | lost |")
say("|---|---|---|---|")
for a, b in (("4bit_base", "4bit_i100"), ("4bit_base", "4bit_i300"), ("4bit_base", "4bit_r9k"), ("bf16_base", "bf16_i100c20")):
    if a not in JBR or b not in JBR:
        continue
    cnt = defaultdict(lambda: [0, 0])
    for tid in JBR[a]:
        ca, cb = bool(JBR[a][tid]["correct"]), bool(JBR[b][tid]["correct"])
        f = JBR[a][tid]["family"]
        cnt[f][0] += (not ca) and cb
        cnt[f][1] += ca and (not cb)
    for f in sorted(cnt):
        if cnt[f] != [0, 0]:
            say(f"| {a} -> {b} | {f} | {cnt[f][0]} | {cnt[f][1]} |")

if B77R:
    say("\n### 4d. Banking77: most frequent confusions (gold index -> predicted index) and confidence\n")
    for k, v in B77R.items():
        wrong = ~v["correct"]
        pairs = Counter(zip(v["gold"][wrong].tolist(), v["pred"][wrong].tolist())).most_common(5)
        say(f"- **{k}**: {wrong.sum()} wrong; mean confidence on wrong {v['conf'][wrong].mean():.3f}; wrong with conf>=0.9: "
            f"{(v['conf'][wrong] >= .9).mean():.1%}; top confusions " + ", ".join(f"{g}->{p} (x{c})" for (g, p), c in pairs))


# ---------------------------------------------------------------- 5. temperature scaling
def conf_T(probs_rows, T):
    """Confidence of the argmax after p -> p^(1/T) renormalised. probs_rows: list of arrays."""
    out = []
    for p in probs_rows:
        q = np.power(np.clip(p, 1e-12, None), 1.0 / T)
        out.append(q.max() / q.sum())
    return np.array(out)


def bce(conf, corr):
    c = np.clip(conf, 1e-6, 1 - 1e-6)
    return float(-(corr * np.log(c) + (1 - corr) * np.log(1 - c)).mean())


def fit_T(prob_rows, corr, grid=np.exp(np.linspace(np.log(0.3), np.log(8), 60))):
    best = min(grid, key=lambda T: bce(conf_T(prob_rows, T), corr.astype(float)))
    return float(best)


def jb_prob_rows(rows):
    ids = sorted(rows)
    return [np.array(list((rows[i].get("probs") or {"-": 1.0}).values()), dtype=float) for i in ids], \
        np.array([bool(rows[i]["correct"]) for i in ids])


say("\n## 5. Post-hoc temperature scaling (ECE10 on JevBench; accuracy unchanged by construction)\n")
say("T is chosen to minimise the cross-entropy of 'is the top answer right' (confidence of the top option after p^(1/T), renormalised). "
    "Transfer: T fitted on the same model's Banking77 answers, applied to its JevBench answers (the OOD direction of the paper). "
    "CV: T fitted on 4/5 of JevBench, ECE measured on the other 1/5 (5 folds, pooled), averaged over 20 random splits.\n")
say("| run | ECE10 raw | T from Banking77 | ECE10 after (transfer) | ECE10 after (5-fold CV in JevBench) |")
say("|---|---|---|---|---|")
for k in JBR:
    rows_p, corr = jb_prob_rows(JBR[k])
    raw = A.ece(np.array([p.max() / p.sum() for p in rows_p]), corr.astype(float), 10)
    tb, ece_tr = None, None
    if k in B77R:
        v = B77R[k]
        tb = fit_T(list(v["probs"]), v["correct"])
        ece_tr = A.ece(conf_T(rows_p, tb), corr.astype(float), 10)
    idx = np.arange(len(corr))
    ece_reps = []
    for rep in range(20):   # 20 different random 5-fold splits (fixed seeds, independent of the order of the runs)
        folds = np.array_split(np.random.default_rng(1000 + rep).permutation(idx), 5)
        pooled_conf = np.zeros(len(corr))
        for f in folds:
            tr = np.setdiff1d(idx, f)
            T = fit_T([rows_p[i] for i in tr], corr[tr])
            pooled_conf[f] = conf_T([rows_p[i] for i in f], T)
        ece_reps.append(A.ece(pooled_conf, corr.astype(float), 10))
    ece_cv = float(np.mean(ece_reps))
    say(f"| {k} | {raw:.4f} | {('%.2f' % tb) if tb else 'n/a'} | {('%.4f' % ece_tr) if ece_tr is not None else 'n/a'} | {ece_cv:.4f} |")

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written to {sys.argv[1]}]")
