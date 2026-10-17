"""Analysis of the Qwen3-0.6B runs (JevBench-231 and Banking77-1000), same helpers and definitions as analyze_4b_full.py.
Baseline = untrained 0.6B with the paper-literal prompt (paperA) from the prompt audit. All tuned runs also use paperA.
Sections: accuracy/ECE table with the paper's numbers for reference, paired tests tuned vs baseline, 'yes' bias and catch-all.
Usage: python analyze_06b.py [OUT.md]   (all numbers are computed from the per-item files, nothing typed by hand)"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze_jevbench_run as A  # noqa: E402
rng = np.random.default_rng(0)


def paired(ca, cb, conf_a, conf_b, n_boot=4000):
    """Same definition as in analyze_4b_full.py: sign test on won/lost items, bootstrap CIs of the accuracy and ECE10 differences."""
    won, lost = int((~ca & cb).sum()), int((ca & ~cb).sum())
    idx = np.arange(len(ca))
    d_acc, d_ece = [], []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx))
        d_acc.append(cb[s].mean() - ca[s].mean())
        d_ece.append(A.ece(conf_b[s], cb[s].astype(float), 10) - A.ece(conf_a[s], ca[s].astype(float), 10))
    q = lambda d: tuple(np.percentile(d, [2.5, 97.5]))  # noqa: E731
    return {"won": won, "lost": lost, "p": A.sign_test(won, lost), "dacc": cb.mean() - ca.mean(), "dacc_ci": q(d_acc),
            "dece": A.ece(conf_b, cb.astype(float), 10) - A.ece(conf_a, ca.astype(float), 10), "dece_ci": q(d_ece)}


def verdict(ci):
    return "differs from 0" if (ci[0] > 0 or ci[1] < 0) else "CI contains 0"


ROOT_A = Path(__file__).resolve().parent.parent / "results" / "server_A"
ROOT_B = Path(__file__).resolve().parent.parent / "results" / "server_B"
JB = {  # name -> (root, relative dir)   all tuned runs use the paperA prompt, seed 20260922 unless stated
    "base_paperA": (ROOT_A, "promptaudit/paperA/jevbench"),
    "base_default": (ROOT_A, "promptaudit/default/jevbench"),
    "intent100": (ROOT_A, "q/i100_06_eval/jevbench"),
    "intent1ep_96pct_ckpt": (ROOT_A, "chain4/intent1ep_eval/jevbench"),
    "intent1ep_full_seed1": (ROOT_B, "q/i1ep_full_06_eval/jevbench"),
    "intent1ep_full_seed2": (ROOT_B, "q/i1ep_s2_06_eval/jevbench"),
    "reason9k_seed1": (ROOT_A, "chain3/reason9k_eval/jevbench"),
    "reason9k_seed2": (ROOT_B, "q/r9k_s2_06_eval/jevbench"),
    "reason9k_seed3": (ROOT_B, "q/r9k_s3_06_eval/jevbench"),
}
B77 = {
    "base_paperA": (ROOT_A, "chain2/base/eval"),
    "intent100": (ROOT_A, "q/i100_06_eval/eval"),
    "intent1ep_96pct_ckpt": (ROOT_A, "q/i1ep96_eval/eval"),
    "intent1ep_full_seed1": (ROOT_B, "q/i1ep_full_06_eval/eval"),
    "intent1ep_full_seed2": (ROOT_B, "q/i1ep_s2_06_eval/eval"),
    "reason9k_seed1": (ROOT_A, "chain3/reason9k_eval/eval"),
    "reason9k_seed2": (ROOT_B, "q/r9k_s2_06_eval/eval"),
    "reason9k_seed3": (ROOT_B, "q/r9k_s3_06_eval/eval"),
}
PAPER = {"base": (56.7, 0.278, 22.2), "intent1ep": (53.7, 0.204, 64.0), "reason9k": (55.0, 0.118, 60.4)}
TUNED = [k for k in JB if k not in ("base_paperA", "base_default")]
OUT = []


def say(s=""):
    OUT.append(s)
    print(s)


def load_jb(name):
    root, rel = JB[name]
    p = root / rel / "results.jsonl"
    return A.load(p) if p.exists() else None


def load_b77(name):
    root, rel = B77[name]
    p = root / rel / "banking77.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    probs = np.array([r["jev"] for r in rows], dtype=float)
    gold = np.array([r["gold"] for r in rows])
    return {"conf": probs.max(1), "correct": probs.argmax(1) == gold, "gold": gold}


def arrays(rows):
    ids = sorted(rows)
    return ids, np.array([A.conf_and_correct(rows[i])[0] for i in ids], dtype=float), np.array([bool(rows[i]["correct"]) for i in ids])


JBR = {k: load_jb(k) for k in JB}
JBR = {k: v for k, v in JBR.items() if v is not None}
say("# Qwen3-0.6B analysis (all numbers computed from the per-item files)\n")
say("Paper (arXiv 2610.02076 v3), 0.6B: original 56.7 / ECE .278 / Banking77 22.2; Intent-1ep 53.7 / .204 / 64.0; Reason-9k 55.0 / .118 / 60.4.\n")
say("## 1. JevBench-231\n")
say("| run | acc | ECE10 | harness acc | harness ECE | predicted yes (of 74 binary; gold yes 35) | mean P(yes) | predicted catch-all (of 24 items) | mean catch-all mass |")
say("|---|---|---|---|---|---|---|---|---|")
for k, rows in JBR.items():
    s = A.summarize(rows)
    root, rel = JB[k]
    sj = json.loads((root / rel / "summary.json").read_text(encoding="utf-8"))
    he = sj["ece"]["ece"] if isinstance(sj["ece"], dict) else sj["ece"]
    say(f"| {k} | {s['acc']:.4f} | {s['ece10']:.4f} | {sj['accuracy']:.4f} | {he:.4f} | {s['yes_pred']} | {s['mean_p_yes']:.3f} | "
        f"{s['pred_catchall']} | {s['mean_catchall_mass']:.3f} |")

B77R = {k: load_b77(k) for k in B77}
B77R = {k: v for k, v in B77R.items() if v is not None}
say("\n## 1b. Banking77 (1000 questions)\n")
say("| run | acc | ECE10 | mean confidence |")
say("|---|---|---|---|")
for k, v in B77R.items():
    say(f"| {k} | {v['correct'].mean():.4f} | {A.ece(v['conf'], v['correct'].astype(float), 10):.4f} | {v['conf'].mean():.3f} |")

say("\n## 2. Paired tests vs the untrained 0.6B (paperA), JevBench (95% bootstrap CI, 4000 resamples)\n")
say("| tuned run | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |")
say("|---|---|---|---|---|---|")
ids0, conf0, corr0 = arrays(JBR["base_paperA"])
for k in TUNED:
    if k not in JBR:
        continue
    ids1, conf1, corr1 = arrays(JBR[k])
    assert ids0 == ids1
    r = paired(corr0, corr1, conf0, conf1)
    say(f"| {k} | {r['won']} / {r['lost']} | {r['p']:.3f} | {r['dacc']:+.3f} [{r['dacc_ci'][0]:+.3f}, {r['dacc_ci'][1]:+.3f}] | "
        f"{r['dece']:+.3f} [{r['dece_ci'][0]:+.3f}, {r['dece_ci'][1]:+.3f}] | {verdict(r['dece_ci'])} |")

say("\n## 2b. Paired tests vs the untrained 0.6B (paperA), Banking77\n")
say("| tuned run | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |")
say("|---|---|---|---|---|---|")
b0 = B77R["base_paperA"]
for k in TUNED:
    if k not in B77R:
        continue
    b1 = B77R[k]
    assert (b0["gold"] == b1["gold"]).all()
    r = paired(b0["correct"], b1["correct"], b0["conf"], b1["conf"])
    say(f"| {k} | {r['won']} / {r['lost']} | {r['p']:.4f} | {r['dacc']:+.3f} [{r['dacc_ci'][0]:+.3f}, {r['dacc_ci'][1]:+.3f}] | "
        f"{r['dece']:+.3f} [{r['dece_ci'][0]:+.3f}, {r['dece_ci'][1]:+.3f}] | {verdict(r['dece_ci'])} |")

if len(sys.argv) > 1:
    Path(sys.argv[1]).write_text("\n".join(OUT), encoding="utf-8")
    print(f"\n[written to {sys.argv[1]}]")
