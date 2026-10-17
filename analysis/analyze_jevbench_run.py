"""Analyse JevBench-231 per-item results (jevforge/jevbench harness results.jsonl) for the 'intent trap' study.
Computes, from the per-item probabilities: accuracy, ECE (10 bins = harness definition, and 15 bins), mean confidence on wrong
answers, share of wrong answers with conf >= 0.9, 'yes' count on the binary yes/no items, P(yes), catch-all probability mass.
With two runs it adds a paired comparison (won/lost, exact sign test, bootstrap CIs).
Usage: python analyze_jevbench_run.py RUN_A_results.jsonl [RUN_B_results.jsonl]    (A = baseline, B = tuned)
Validation: for a run directory containing summary.json the script prints the harness numbers next to ours."""
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

CATCH = re.compile(r"^(other|unknown|none|none_of_these|none of these|none_of_the_above|unsure|unclear|neither|not_sure)$", re.I)
rng = np.random.default_rng(0)


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["task_id"]: r for r in rows}


def conf_and_correct(r):
    probs = r.get("probs") or {}
    if not probs:
        return None, bool(r.get("correct"))
    return max(probs.values()), bool(r.get("correct"))


def ece(conf, corr, bins):
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(conf)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            e += m.sum() / n * abs(conf[m].mean() - corr[m].mean())
    return float(e)


def summarize(rows):
    ids = sorted(rows)
    cc = [conf_and_correct(rows[i]) for i in ids]
    keep = [(c, k) for c, k in cc if c is not None]
    conf = np.array([c for c, _ in keep])
    corr = np.array([float(k) for _, k in keep])
    out = {"n": len(keep), "acc": corr.mean(), "ece10": ece(conf, corr, 10), "ece15": ece(conf, corr, 15),
           "conf_wrong": conf[corr == 0].mean() if (corr == 0).any() else float("nan"),
           "wrong_ge90": float((conf[corr == 0] >= .9).mean()) if (corr == 0).any() else float("nan")}
    yes = gold_yes = binary = 0
    pyes = []
    for i in ids:
        r = rows[i]
        probs = r.get("probs") or {}
        if set(probs) == {"yes", "no"}:
            binary += 1
            pred, ok = r["predicted"], bool(r["correct"])
            yes += pred == "yes"
            gold_yes += (pred == "yes") == ok
            pyes.append(probs["yes"])
    out.update(binary_items=binary, yes_pred=yes, yes_gold=gold_yes, mean_p_yes=float(np.mean(pyes)) if pyes else float("nan"))
    mass, pred_catch, with_catch = [], 0, 0
    for i in ids:
        probs = rows[i].get("probs") or {}
        cl = [k for k in probs if CATCH.match(str(k))]
        if cl:
            with_catch += 1
            mass.append(sum(probs[k] for k in cl))
            pred_catch += rows[i]["predicted"] in cl
    out.update(items_with_catchall=with_catch, mean_catchall_mass=float(np.mean(mass)) if mass else float("nan"),
               pred_catchall=pred_catch)
    return out


def fmt(name, s):
    print(f"{name}: n={s['n']} acc={s['acc']:.4f} ECE10={s['ece10']:.4f} ECE15={s['ece15']:.4f} "
          f"conf_wrong={s['conf_wrong']:.3f} wrong>=.9={s['wrong_ge90']:.1%} | binary={s['binary_items']} yes_pred={s['yes_pred']} "
          f"(gold yes={s['yes_gold']}) mean P(yes)={s['mean_p_yes']:.3f} | catch-all items={s['items_with_catchall']} "
          f"mean mass={s['mean_catchall_mass']:.3f} predicted={s['pred_catchall']}")


def sign_test(won, lost):
    n = won + lost
    if n == 0:
        return 1.0
    k = min(won, lost)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def compare(a, b):
    ids = sorted(set(a) & set(b))
    ca = np.array([bool(a[i]["correct"]) for i in ids])
    cb = np.array([bool(b[i]["correct"]) for i in ids])
    won, lost = int((~ca & cb).sum()), int((ca & ~cb).sum())
    print(f"paired on {len(ids)} items: won {won}, lost {lost}, sign test p={sign_test(won, lost):.3f}")
    conf_a = np.array([conf_and_correct(a[i])[0] for i in ids])
    conf_b = np.array([conf_and_correct(b[i])[0] for i in ids])
    d_acc, d_ece = [], []
    idx = np.arange(len(ids))
    for _ in range(4000):
        s = rng.choice(idx, len(idx))
        d_acc.append(cb[s].mean() - ca[s].mean())
        d_ece.append(ece(conf_b[s], cb[s].astype(float), 10) - ece(conf_a[s], ca[s].astype(float), 10))
    for name, d in (("acc diff (B-A)", d_acc), ("ECE10 diff (B-A)", d_ece)):
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f"{name}: {np.mean(d):+.4f}  95% bootstrap CI [{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    paths = sys.argv[1:]
    runs = []
    for p in paths:
        rows = load(p)
        runs.append(rows)
        fmt(Path(p).parent.parent.name, summarize(rows))
        sj = Path(p).parent / "summary.json"
        if sj.exists():
            s = json.loads(sj.read_text(encoding="utf-8"))
            e = s["ece"]["ece"] if isinstance(s["ece"], dict) else s["ece"]
            print(f"   harness summary.json: accuracy={s['accuracy']:.4f} ({s['n_correct']}/{s['n_scorable']}) ece={e:.4f}")
    if len(runs) == 2:
        compare(runs[0], runs[1])
