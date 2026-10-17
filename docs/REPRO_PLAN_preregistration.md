# Reproduction plan (written BEFORE any training run, 2026-10-15) - "intent trap" of LLM-as-Jev (arXiv 2610.02076 v3)

> Note for readers of this repository: this is the plan as written on 2026-10-15, before the GPU training runs (including the notes appended that day); only this note was added for the repository. It mentions files of a related project
> (`obs/`, `research/txt/`, the "Laya" observation H4) that are **not** part of this repository. Outcomes against these hypotheses are in `REPORT.md`, section 6.

Purpose of this file: fix hypotheses, metrics and decision rules in advance so results cannot be re-interpreted after the fact.
Anything not listed here and done later must be logged under "Deviations" with a reason.

## Task (professor)
Reproduce the phenomenon: a decision model fine-tuned on a narrow intent-classification corpus gains in-domain
(Banking77) but, out of domain (JevBench public-231), shows worse calibration (ECE up), confident wrong answers,
"yes" bias, and avoids "none of these". Then observe failures; only afterwards try improvements.

## What the paper itself reports (verified in the PDF text; single seed, ECE without CI)
| model / config | JevBench acc | ECE | "yes" count (74 binary items) | Banking77 |
|---|---|---|---|---|
| 4B original | 81.4 | .057 | 36 | 69.0 |
| 4B Intent-100 | 77.5 (sign test 12/21, p=.16) | .092 | 55 | 74.6 |
| 4B Intent-1ep | 74.9 (8/23, p=.01) | .134 | 49 | 76.7 |
| 4B Reason-9k | 81.4 (14/14) | .067 | 44 | 75.9 |
| 0.6B original | 56.7 | .278 | 34 | 22.2 |
| 0.6B Intent-1ep | 53.7 (22/29, p=.40) | .204 | 48 | 64.0 |
| 0.6B Reason-9k | 55.0 | .118 | 41 | 60.4 |
Inner Jev paper: JevBench-231 bootstrap sd is about 3 accuracy points, so differences < ~3 points are noise.

## Hypotheses
- H0 (pipeline): our baseline reproduces the paper's baseline. Accept if |acc - paper| <= 3 points AND |ECE - paper| <= 0.03 (tolerance is OUR choice, justified by the ~3-point bootstrap sd). If not accepted -> debug the pipeline BEFORE any training; do not interpret later results.
- H1 (4B, main): Intent-1ep lowers JevBench acc, raises ECE, raises confidence on wrong answers, raises "yes" count, lowers catch-all probability, versus the 4B baseline; Banking77 rises.
- H2 (0.6B): Intent-1ep raises "yes" count (paper: 34 -> 48) but does NOT raise ECE (paper: .278 -> .204). Observing the paper's pattern counts as a replication; observing ECE rise is a NEW finding.
- H3 (control): Reason-9k (balanced) restores accuracy to baseline level; "yes" stays above baseline (paper: 44 vs 36).
- H4 (Laya observation, already measured separately): shipped temperatures are not task-appropriate; a temperature < 1 harms (see obs/REPORT_laya.md).

## Design
- Models: Qwen3-0.6B first (cheap, validates the pipeline), then Qwen3.5-4B (needs ~8 GB download via hf-mirror; fit in 24 GB NOT yet verified; use --grad-ckpt if needed).
- Implementation: jevforge (independent MIT implementation of the paper's objective; audited by reading, NOT numerically validated). Settings = paper defaults: LoRA r16/a32 all-linear, lr 1e-4, batch 32, 1 epoch, lambda=1, seed 20260922 (repo default) plus at least one extra seed per main configuration if time allows.
- Mixes (definitions checked against the PDF text, Table 3/5 and line 1175): 
  * Intent-1ep = `--mix intent` (CLINC150, MASSIVE, Bitext, CommonsenseQA, HellaSwag; ~95,461 examples, 2,984 steps, batch 32).
  * Reason-9k = 1,000 examples from EACH of 9 datasets (CLINC150, MASSIVE, Bitext + ReClor, LogiQA2, CosmosQA, PIQA, aNLI, ProofWriter), 282 steps = `--mix clinc150,massive,bitext,reclor,logiqa2,cosmos_qa,piqa,anli,proofwriter --max-examples 9000` (build_mix gives equal quota 9000//9 = 1000 per source). NOTE: jevforge's own comment/constant REASON_PER_SOURCE=1500 and `--mix general` do NOT equal Reason-9k.
  * Intent-100 (4B only) = 3,000 intent examples, global batch 30, 100 steps: `--mix intent --max-examples 3000 --batch-size 30`; the paper does not say whether its 3,000 are stratified, ours would be (600 per source) -> deviation.
- Speed test first (4 steps) -> projected hours; real run only if projected*1.15 <= 6 h per run.
- Never put API keys on the server. Write only inside our own folder; no deletions/overwrites.

## Metrics (all computed from stored per-item probabilities, not only from summaries)
Accuracy with bootstrap 95% CI and paired sign test vs baseline (won/lost); ECE (15 bins, with bootstrap CI);
mean confidence on wrong answers; share of wrong answers with confidence >= 0.9; "yes" count on the 74 binary items and
P(yes) shift per item; mean probability on catch-all options; accuracy on the 37 long items; Banking77 accuracy;
confidence histogram. Extra: our 7-dataset calibration suite (obs/) for base vs tuned, if time allows.

## Decision rules
- "Reproduced" (H1): acc drop >= 3 points AND ECE increase >= 0.03 AND "yes" count increase, in the same direction as the paper, for the 4B model; report CIs.
- "Partly reproduced": "yes"/catch-all effects present, ECE or accuracy not.
- "Not reproduced": none of the above beyond noise. A negative result is reported as such.
- With a single seed per configuration the conclusion is labelled "single-seed, indicative".

## Improvements (only after observing; each compared on the SAME items with the SAME metrics)
1. Post-hoc temperature >= 1 fitted on held-out items, cross-fitted (cheap; paper literature: competitive).
2. Balanced data (Reason mix) - known fix in the paper.
3. CogCalib-style idea (OURS, untested): label-smoothing term applied only to training items the base model already answers correctly with high probability ("known"), inside the tree loss.
4. For comparison only, if time: LoRA ensemble / Laplace-LoRA (needs a different codebase; not planned for tonight).

## Threats to validity (to be repeated in the report)
Single seed; jevforge is third-party and unvalidated against the authors' numbers; JevBench version pinned by jevforge (commit 1bcc55e) may differ from the paper's v1.4.2.2; 231 items only (sd ~3 points); our hardware (1x 4090 D) and per-example loop may change numerics; bf16 vs fp32.

## Analysis code status
research/analyze_jevbench_run.py reproduces the harness summary.json EXACTLY on the repo's sample runs (acc .5887/.5714, ECE10 .2372/.1238; 74 binary items, 35 gold "yes").
Early observation on the repo's own tiny CPU run (Qwen3-0.6B, 94 steps, ~3% data): "yes" predictions 51 -> 61, mean catch-all mass .129 -> .053, ECE .237 -> .124 (CI of the ECE difference excludes 0), accuracy -1.8 points (p=.57). Direction matches the paper's 0.6B pattern (yes up, catch-all down, ECE down).
H0 WARNING (pre-registered rule): that same baseline run already deviates from the paper's 0.6B baseline: ECE .237 vs .278 (diff .041 > our .03 tolerance) and "yes" 51 vs 34, while accuracy 58.9 vs 56.7 is within 3 points. So H0 would NOT be accepted on that run; our own baseline run on the GPU server must be compared again (possible causes, untested: dtype, prompt template/readout details, harness version).

PROMPT AUDIT (jevforge/prompt.py vs paper Sec. 3.2, read 2026-10-15): paper template = "State: {s}" / "Options: [1] {o1} ... [K] {oK}" / "Select one option. Answer only with its bracketed numeric identifier." + assistant prefill "Best answer: [" with thinking disabled. jevforge = "{state}" (no "State:" label) / item instruction (JevBench passes each item's own `instructions`; default "Choose the best option.") / "Options:" + "[i] text" lines / "Answer only with its bracketed numeric identifier."; same prefill. Differences: no "State:" label, instruction sentence differs, item-specific instruction is inserted. The paper's template cannot contain the item-specific question as drawn, so the exact JevBench prompt is UNKNOWN to us. Candidate cause of the baseline mismatch (untested). PLAN: run the baseline with BOTH the jevforge default prompt and a paper-literal variant (monkeypatch format_user_message in our own driver script, no repo edit) and report the sensitivity of ECE and "yes" count to prompt wording; do not claim a faithful baseline reproduction until one variant meets the H0 tolerance.

## Deviations log (append below with date and reason)
- 2026-10-15: jevforge's hub.py comment says Reason-9k = 1.5k per reasoning set; the PDF (Table 5: 33% intent; line 1175: 1,000 per dataset, 282 steps) says 1,000 per dataset incl. 3 intent sets. We follow the PDF (explicit 9-source list). Source: research/txt/llm_as_jev_2610.02076.txt lines ~1027-1033, 1175-1176.
- 2026-10-15: Intent-100 sampling (stratified vs random) is unspecified in the paper; ours is stratified by source.
