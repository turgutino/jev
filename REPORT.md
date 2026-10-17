# Replicating the "intent trap" of LLM-as-Jev (arXiv 2610.02076 v3) — report

Status: draft for review, 2026-10-17. Single author, small compute (one 24 GB GPU per run).
Legend used throughout: **[V]** = computed by us from the per-item files in this repository (re-run `analysis/*.py` to check),
**[I]** = our interpretation, **[U]** = unverified / open question.

## 1. Summary

Task: fine-tune a small decision model (LLM-as-Jev: LoRA on Qwen3-0.6B and Qwen3.5-4B) on intent-only data and check the claimed
side effects — higher in-domain accuracy, but worse calibration (ECE) and more confident wrong answers out of domain, a "yes" bias,
and avoiding "none of these" options — then look at the failures and try fixes.

What we found:

1. **Qwen3-0.6B reproduces the paper's pattern closely** [V]. The untrained model with the paper-literal prompt gives JevBench
   56.7 % / ECE 0.277 (paper: 56.7 % / .278). Intent-1ep lowers ECE (0.277 → 0.224 / 0.233, paper .204), raises the "yes" count
   (61 → 64 / 68 of 74 binary items) and lowers the mass on catch-all options (0.144 → 0.096 / 0.111). Banking77 accuracy rises from 19 % to 57–58 %.
   Balanced "Reason-9k" data lowers ECE a lot in all 3 seeds (−0.13 … −0.19, paired CI excludes 0).
2. **Qwen3.5-4B: the intent trap shows up on Banking77 and for the long intent run, but not for the short intent runs on JevBench** [V].
   * Banking77 (in-domain): every tuned model gains 4.3–5.0 accuracy points (sign test p < 0.001) and becomes over-confident:
     ECE 0.064 → 0.124 (Intent-100), 0.180 (Intent-300), 0.118 (Reason-9k); all differences are significant.
   * JevBench (out of domain): Intent-300 raises ECE by +0.048 (95 % CI [+0.006, +0.086]), 30 % of its wrong answers have confidence ≥ 0.9
     (baseline 10 %), and it predicts "yes" on 62 instead of 55 binary items. Intent-100 (two seeds) and the bf16 Intent-100 do **not** differ
     significantly from their baselines in accuracy or ECE. We see **no accuracy drop** on JevBench (the paper reports −3.9 and −6.5 points).
3. **The paper's remedy (balanced data) is not robust for 4B on JevBench** [V]. Reason-9k seed 1 improved accuracy (+5.2, p = 0.017) and ECE (−0.048),
   seed 2 did not (+2.6, +0.008, both CIs contain 0); the two seeds differ in ECE by 0.056 (significant). Single-seed 4B conclusions are therefore unreliable.
4. **Our 4B baseline only matches the paper with a different prompt than the one we trained with** [V]. With jevforge's default prompt the bf16 baseline is
   79.2 % / ECE 0.035 / 40 "yes" (paper 81.4 % / .057 / 36). With the paper-literal prompt ("paperA", used for all our 4B fine-tuning) it is 72.7 % / 0.051 / 51 "yes".
   For 0.6B it is the other way round. Our 4B fine-tuning results are therefore **not** a faithful replication of the paper's 4B setting (see §6 and §8).
5. **Post-hoc temperature scaling is a cheap fix** [V]: with T fitted on Banking77 and applied to JevBench, ECE falls 0.157 → 0.097 (Intent-300),
   0.106 → 0.090 (Intent-100), 0.061 → 0.048 (Reason-9k); the fitted T grows with intent training (1.14 base, 1.35, 1.59). Only 231 test items, so indicative.

## 2. What the paper claims (numbers as read from the PDF text, single seed in the paper)

| model / config | JevBench acc | ECE | "yes" (of 74 binary) | Banking77 |
|---|---|---|---|---|
| 4B original | 81.4 | .057 | 36 | 69.0 |
| 4B Intent-100 | 77.5 | .092 | 55 | 74.6 |
| 4B Intent-1ep | 74.9 | .134 | 49 | 76.7 |
| 4B Reason-9k | 81.4 | .067 | 44 | 75.9 |
| 0.6B original | 56.7 | .278 | 34 | 22.2 |
| 0.6B Intent-1ep | 53.7 | .204 | 48 | 64.0 |
| 0.6B Reason-9k | 55.0 | .118 | 41 | 60.4 |

Hypotheses and decision rules were written down **before** the first training run (`docs/REPRO_PLAN_preregistration.md`).
JevBench has only 231 items; the paper's own bootstrap sd is about 3 accuracy points, so accuracy differences below ~3 points are noise.

## 3. Setup

* **Implementation:** `jevforge` (third-party, MIT; commit 8986e52, JevBench submodule 1bcc55e), not the authors' code. We audited it by reading it; it was
  **not** validated numerically against the authors' checkpoints [U]. jevforge itself was not edited; our driver scripts patch the prompt, the 4-bit loader
  and the option cap at run time (`code/runners/`).
* **Models:** Qwen/Qwen3-0.6B and Qwen/Qwen3.5-4B. LoRA r=16, alpha=32, all linear layers, lr 1e-4, jevforge's tree loss with KL anchors, default seed 20260922
  (other seeds listed below).
* **Data mixes** (jevforge `--mix`): *Intent-1ep* = the intent mix for one epoch (3228 steps at batch 32 in our logs); *Intent-100* = 3000 intent examples, batch 30
  (100 steps; stratified by source — the paper does not say how it samples); *Intent-300* = 9000 intent examples, batch 30 (300 steps; our addition, between Intent-100 and 1 epoch);
  *Reason-9k* = CLINC150, MASSIVE, Bitext + ReClor, LogiQA2, CosmosQA, PIQA, aNLI, ProofWriter, 9000 examples = 282 steps.
* **Evaluation:** JevBench public split (231 items; 74 binary yes/no items of which 35 have gold "yes"; 24 items with a catch-all option) and 1000 Banking77 questions
  (77 options; `--limit 1000`). ECE with 10 equal-width bins (the harness definition; our recomputation equals the harness `summary.json` for every run).
  Paired comparison on the same items: exact sign test on won/lost items; 95 % bootstrap intervals (4000 resamples) for accuracy and ECE differences.
  "Yes" bias = number of predicted "yes" on the 74 binary items (our definition; the paper's exact definition is not known to us [U]).
* **Hardware:** one 24 GB GPU (23.5 GiB usable) per run, sometimes shared with our other jobs. All runs 2026-10-15 … 2026-10-17.

## 4. Baseline check (is our pipeline comparable to the paper?)

| model, prompt | JevBench acc | ECE | "yes" | Banking77 acc / ECE | paper |
|---|---|---|---|---|---|
| 0.6B, paperA | 56.7 | .277 | 61 | 19.4 / .447 | 56.7 / .278 / 34 / 22.2 |
| 0.6B, jevforge default | 59.3 | .232 | 52 | n/a | |
| 4B bf16, paperA | 72.7 | .052 | 51 | 69.3 / .039 | 81.4 / .057 / 36 / 69.0 |
| 4B bf16, jevforge default | 79.2 | .035 | 40 | 70.5 / .065 | |
| 4B 4-bit, paperA | 69.7 | .109 | 55 | 66.3 / .064 | |

[V] Pre-registered tolerance for "baseline reproduced": |acc − paper| ≤ 3 points and |ECE − paper| ≤ 0.03. It is met by 0.6B+paperA (accuracy and ECE)
and by 4B+default (79.2 vs 81.4; .035 vs .057). It is **not** met by 4B+paperA (−8.7 points) nor by the "yes" counts of 0.6B (61 vs 34).
[I] The best prompt differs per model, and the exact JevBench prompt of the paper is not known to us (the paper's template cannot contain the item-specific question as drawn).
All fine-tuning below uses the paperA prompt for training and evaluation. The 4B baseline under paperA is lower than the paper's, so absolute 4B numbers should not be compared with the paper's.

## 5. Results — Qwen3-0.6B (paperA prompt; paired tests are against the untrained 0.6B)

| run | JevBench acc | ECE | "yes" | ECE diff vs base [95 % CI] | Banking77 acc | B77 ECE |
|---|---|---|---|---|---|---|
| untrained | 56.7 | .277 | 61 | | 19.4 | .447 |
| Intent-100 | 55.8 | .171 | 61 | −0.106 [−0.152, −0.062] | 47.4 | .076 |
| Intent-1ep, seed 1 | 55.0 | .224 | 64 | −0.053 [−0.103, −0.002] | 58.1 | .176 |
| Intent-1ep, seed 2 | 54.5 | .233 | 68 | −0.044 [−0.100, +0.007] | 57.1 | .196 |
| Intent-1ep, 96 % checkpoint of an early run | 53.7 | .228 | 68 | −0.049 [−0.103, +0.001] | 59.2 | .136 |
| Reason-9k, seed 1 | 56.7 | .143 | 46 | −0.133 [−0.190, −0.079] | 47.1 | .094 |
| Reason-9k, seed 2 | 61.0 | .090 | 46 | −0.187 [−0.229, −0.117] | 49.5 | .088 |
| Reason-9k, seed 3 | 59.7 | .108 | 46 | −0.169 [−0.211, −0.106] | 47.7 | .114 |

[V] Accuracy differences on JevBench are all within noise (sign tests p ≥ 0.15). On Banking77 every tuned model gains 28–40 points (p < 0.001) and its ECE falls
(the untrained model is badly mis-calibrated there). Mean P(yes) on the binary items: 0.778 untrained, 0.82–0.87 after Intent-1ep, 0.57–0.58 after Reason-9k.
Mean probability on catch-all options: 0.144 untrained, 0.089–0.111 after Intent-1ep, 0.108–0.157 after Reason-9k; the number of predicted catch-alls is 3 in every run.
[I] This matches the paper's 0.6B pattern (ECE does not rise, "yes" rises, Reason-9k helps) and the pre-registered hypothesis H2.
Banking77 accuracy after tuning is below the paper's (57–59 vs 64.0 for Intent-1ep; 47–49 vs 60.4 for Reason-9k) [V].

## 6. Results — Qwen3.5-4B

All "4-bit" runs use nf4 base weights (QLoRA-style), ≤ 20 options per training question and paperA. Only one bf16 run was possible (see §8). Baselines are matched by precision.

### JevBench-231

| run | acc | ECE | "yes" | conf. on wrong | wrong with conf ≥ .9 | paired vs matching baseline: won/lost, acc diff [CI], ECE diff [CI] |
|---|---|---|---|---|---|---|
| 4-bit baseline | 69.7 | .109 | 55 | .641 | 10.0 % | |
| 4-bit Intent-100, seed 1 | 71.0 | .106 | 53 | .643 | 10.4 % | 9/6, +1.3 [−1.7, +4.8], −0.003 [−0.042, +0.030] |
| 4-bit Intent-100, seed 2 | 70.6 | .095 | 54 | .644 | 13.2 % | 11/9, +0.9 [−3.0, +4.8], −0.015 [−0.045, +0.028] |
| 4-bit Intent-300 | 69.7 | .157 | 62 | .743 | 30.0 % | 10/10, 0.0 [−3.9, +3.9], **+0.048 [+0.006, +0.086]** |
| 4-bit Reason-9k, seed 1 | 74.9 | .061 | 56 | .618 | 8.6 % | 17/5 (p = .017), **+5.2 [+1.3, +9.1]**, **−0.048 [−0.083, −0.008]** |
| 4-bit Reason-9k, seed 2 | 72.3 | .117 | 55 | .613 | 7.8 % | 12/6, +2.6 [−0.9, +6.1], +0.008 [−0.045, +0.035] |
| bf16 baseline | 72.7 | .051 | 51 | .617 | 7.9 % | |
| bf16 Intent-100 (≤ 20 options) | 74.9 | .083 | 54 | .663 | 15.5 % | 10/5, +2.2 [−0.9, +5.6], +0.031 [−0.018, +0.059] |

Seed-to-seed: Intent-100 s1→s2: ECE −0.011 [−0.041, +0.031] (noise level); Reason-9k s1→s2: ECE **+0.056 [+0.003, +0.080]**, accuracy −2.6 [−6.1, +0.9] [V].
Intent-100 vs Reason-9k (seed 1): ECE −0.045 [−0.077, −0.002], accuracy +3.9 [−0.4, +7.8] [V].

### Banking77 (1000 questions, 4-bit)

| run | acc | ECE | mean confidence | wrong with conf ≥ .9 | paired vs baseline: won/lost, acc diff, ECE diff [CI] |
|---|---|---|---|---|---|
| baseline | 66.3 | .064 | .727 | 9.8 % | |
| Intent-100 | 70.8 | .124 | .831 | 20.9 % | 67/22 (p < .001), +4.5 [+2.8, +6.3], +0.060 [+0.037, +0.075] |
| Intent-300 | 70.6 | .180 | .885 | 37.1 % | 78/35 (p = .0001), +4.3 [+2.1, +6.5], +0.115 [+0.089, +0.133] |
| Reason-9k | 71.3 | .118 | .831 | 22.6 % | 70/20 (p < .001), +5.0 [+3.2, +6.8], +0.053 [+0.031, +0.069] |

Intent-100 vs Reason-9k on Banking77: no difference (ECE −0.006 [−0.020, +0.008]) [V]. Banking77 numbers are single-seed (no 4B Banking77 repeats).

### "Yes" bias and catch-all options (4B)

Predicted "yes" out of 74 (gold 35): 4-bit baseline 55, Intent-100 53 / 54, Intent-300 **62**, Reason-9k 56 / 55; bf16 baseline 51, bf16 Intent-100 54. Mean P(yes): 0.656 → 0.673 / 0.693 (Intent-100),
**0.784** (Intent-300), 0.662 / 0.664 (Reason-9k) [V]. Catch-all options: mean mass 0.282 (baseline) vs 0.326 / 0.296 (Intent-100), 0.279 (Intent-300), 0.263 / 0.267 (Reason-9k);
predicted catch-alls 7 → 8 / 6 / 6 / 6 / 7 — **no suppression of "none of these" is visible for 4B** [V]. A "yes" bias exists in all of our 4B baselines already (51–55 vs gold 35), unlike the paper's baseline (36); only with the default prompt is it close (40) [V].

### Checklist against the paper's claims

| claim | 0.6B | 4B |
|---|---|---|
| in-domain accuracy up | yes (+28…+40, p < .001) [V] | yes (+4.3…+5.0, p < .001) [V] |
| JevBench accuracy down | no (−1…−3, n.s.) [V] | no (0…+2, n.s.) [V] |
| ECE up | no, down (as the paper's 0.6B) [V] | Banking77: yes, strongly; JevBench: only Intent-300 (+0.048); Intent-100 n.s. [V] |
| more confident wrong answers | n/a | Intent-300: 10 % → 30 % of errors with conf ≥ .9; Banking77 10 % → 21–37 % [V] |
| "yes" bias up | yes (61 → 64–68) [V] | Intent-300 only (55 → 62) [V] |
| catch-all avoided | mass 0.144 → 0.089–0.111, counts unchanged [V] | not seen [V] |
| balanced data helps | yes, 3/3 seeds [V] | one seed yes, one seed no [V] |

## 7. Failure analysis and improvement attempts

**Where the model fails (4B, JevBench)** [V]: the typical confident error is a "yes" on a binary item (fact, judge_hard, adequacy; e.g. easy-fact-07, hard-sol-b-judge_hard-02/-08 are among the
five most confident errors of every tuned run). `temporal_numeric` is at 13–40 % accuracy for every model. After tuning, the `fact` family gets worse (0.67 → 0.50–0.58, only 12 items) while
`intent`/`extraction`/`routing` items improve slightly. On Banking77 the same confusable intent pairs persist (gold index → predicted index 38→21, 43→38 appear in every run), while the share of wrong answers
made with ≥ 0.9 confidence rises with intent training. Full per-family tables and the lists of most confident errors: `analysis/results_4b.md`.

**Improvements tried.** (a) *Balanced data* (Reason-9k): see §5/§6 — robust at 0.6B, not robust at 4B/JevBench. (b) *Post-hoc temperature scaling* (our own objective: cross-entropy of "top answer is correct" after p → p^(1/T)):
T fitted on the same model's Banking77 answers and applied to JevBench (accuracy unchanged by construction):

| 4B 4-bit run | ECE raw | T | ECE after (transfer) | ECE after (5-fold CV inside JevBench, mean of 20 splits) |
|---|---|---|---|---|
| baseline | .109 | 1.14 | .093 | .071 |
| Intent-100 | .106 | 1.35 | .090 | .096 |
| Intent-300 | .157 | 1.59 | .097 | .056 |
| Reason-9k | .061 | 1.35 | .048 | .050 |

[V] T > 1 grows with intent training (1.14 → 1.35 → 1.59), consistent with growing over-confidence. [U] Only 231 items; the CV column moves by about ±0.01 with the split; no confidence interval computed.
The untested idea of a label-smoothing term on already-known training items (listed in the pre-registration) was **not** tried.

## 8. Deviations from the paper and limitations (please read)

1. **jevforge instead of the authors' code**; not numerically validated; JevBench version pinned by jevforge (1bcc55e) may differ from the paper's [U].
2. **Prompt.** All fine-tuning used the paper-literal prompt (paperA, State/Options/Select one option…). For 4B the jevforge default prompt reproduces the paper's baseline better (§4) but **no 4B model was fine-tuned or evaluated after fine-tuning with it** (not done).
3. **4B fine-tuning is not paper-faithful:** nf4 4-bit base weights, ≤ 20 options per training question (gold + random others, seeded), Reason-9k with `--max-len 450` (drops longer examples; count not recorded), scoring mode forced to "cache", flash-linear-attention kernels installed.
   The paper-faithful bf16 run without the option cap ran out of memory **three times even alone on the 24 GB GPU** [V]; the one bf16 run we have (Intent-100) uses the ≤ 20-option cap.
4. **Not done:** 4B Intent-1ep (full epoch; the bf16 attempt failed and was not retried in 4-bit); 4B Reason-9k in bf16; bf16 Banking77 after tuning; any 4B Banking77 repeat seed; the "known-item label smoothing" idea; LoRA ensembles.
5. **Seeds:** 4B has two seeds only for 4-bit Intent-100 and Reason-9k (seed 2 = 20260923); 0.6B has Reason-9k × 3 and Intent-1ep × 2 (seeds 20260922 / 20261001 / 20261002) plus one early Intent-1ep run that was evaluated from a checkpoint at about 96 % of training. All other configurations are single-seed ("indicative").
6. **Multiple comparisons:** about 30 paired tests are reported without correction; use the CIs as descriptive. 231 JevBench items → the paper's own sd of ~3 accuracy points applies.
7. **"Yes" definition** and **Banking77 sample** (1000 questions) are ours; baseline "yes" counts differ from the paper (§4).
8. **Shared GPU:** some runs shared the GPU with other jobs of ours; several jobs failed with out-of-memory and were re-run unchanged or with a documented change (4B Reason-9k: cap 9 GiB + `--max-len 450`; 4B Intent-300: re-run after a crash at step 170/300). Numerics are not affected by sharing, but all re-runs are listed in `code/queues/`.
9. The temperature-scaling objective and the CV protocol are ours; they are not taken from the paper.

## 9. Conclusion (what we are willing to claim)

* 0.6B: replication of the paper's pattern, including the baseline numbers [V].
* 4B: in-domain accuracy gain with over-confidence is clear [V]; an out-of-domain ECE increase appears only for the longer Intent-300 run (and with the same sign, but not significant, for bf16 Intent-100) [V]; the absence of a JevBench accuracy drop and of catch-all suppression
  is a **difference from the paper** in our setting [V], which may be caused by our prompt, our 4-bit/option-cap deviations or by genuine variance [U].
* The paper's remedy works at 0.6B; at 4B it works for one seed out of two [V]. Temperature scaling reduces ECE cheaply [V, indicative].
* **Recommended next step (not done):** repeat the 4B experiments with the default prompt in bf16 (≤ 20 options) — Intent-100, Intent-300 and Reason-9k with ≥ 2 seeds — because that is where our baseline matches the paper.

## 10. Files

* `code/` runners, schedulers, queue generators, launch scripts; `code/queues/queue_q4f.json`, `queue_q4g.json`, `queue_q4i.json` are the final 4B queues (later files supersede earlier ones).
* `results/server_A`, `results/server_B` per-item results (`results.jsonl`, `banking77.jsonl`), `summary.json`, training logs of every reported run (no adapters, no training data).
* `analysis/analyze_jevbench_run.py` (validated against the harness summaries), `analyze_4b_full.py`, `analyze_06b.py` and their outputs `results_4b.md`, `results_06b.md`. Re-run from the repository root: `python analysis/analyze_4b_full.py out.md`.
* `docs/REPRO_PLAN_preregistration.md` hypotheses, decision rules and deviations log written before the runs.
