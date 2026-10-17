# 4B analysis (all numbers computed from the per-item files)

## 1. JevBench-231: accuracy and ECE (10 bins) -- ours vs harness summary.json

| run | n | acc | ECE10 | harness acc | harness ECE | conf on wrong | wrong with conf>=0.9 |
|---|---|---|---|---|---|---|---|
| 4bit_base | 231 | 0.6970 | 0.1091 | 0.6970 | 0.1091 | 0.641 | 10.0% |
| 4bit_i100 | 231 | 0.7100 | 0.1058 | 0.7100 | 0.1058 | 0.643 | 10.4% |
| 4bit_i100_s2 | 231 | 0.7056 | 0.0945 | 0.7056 | 0.0945 | 0.644 | 13.2% |
| 4bit_i300 | 231 | 0.6970 | 0.1573 | 0.6970 | 0.1573 | 0.743 | 30.0% |
| 4bit_r9k | 231 | 0.7489 | 0.0613 | 0.7489 | 0.0613 | 0.618 | 8.6% |
| 4bit_r9k_s2 | 231 | 0.7229 | 0.1172 | 0.7229 | 0.1172 | 0.613 | 7.8% |
| bf16_base | 231 | 0.7273 | 0.0512 | 0.7273 | 0.0512 | 0.617 | 7.9% |
| bf16_i100c20 | 231 | 0.7489 | 0.0826 | 0.7489 | 0.0826 | 0.663 | 15.5% |

## 1b. Banking77 (1000 questions, 77 options): accuracy and ECE (10 bins)

| run | acc | ECE10 | mean confidence | mean conf - acc |
|---|---|---|---|---|
| 4bit_base | 0.6630 | 0.0642 | 0.727 | +0.064 |
| 4bit_i100 | 0.7080 | 0.1238 | 0.831 | +0.123 |
| 4bit_r9k | 0.7130 | 0.1176 | 0.831 | +0.118 |
| 4bit_i300 | 0.7060 | 0.1795 | 0.885 | +0.179 |

## 2. Paired tests on JevBench (A = baseline of the same precision, B = tuned; 95% bootstrap CI, 4000 resamples)

| A -> B | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |
|---|---|---|---|---|---|
| 4bit_base -> 4bit_i100 | 9 / 6 | 0.607 | +0.013 [-0.017, +0.048] | -0.003 [-0.042, +0.030] | CI contains 0 |
| 4bit_base -> 4bit_i100_s2 | 11 / 9 | 0.824 | +0.009 [-0.030, +0.048] | -0.015 [-0.045, +0.028] | CI contains 0 |
| 4bit_base -> 4bit_i300 | 10 / 10 | 1.000 | +0.000 [-0.039, +0.039] | +0.048 [+0.006, +0.086] | differs from 0 |
| 4bit_base -> 4bit_r9k | 17 / 5 | 0.017 | +0.052 [+0.013, +0.091] | -0.048 [-0.083, -0.008] | differs from 0 |
| 4bit_base -> 4bit_r9k_s2 | 12 / 6 | 0.238 | +0.026 [-0.009, +0.061] | +0.008 [-0.045, +0.035] | CI contains 0 |
| bf16_base -> bf16_i100c20 | 10 / 5 | 0.302 | +0.022 [-0.009, +0.056] | +0.031 [-0.018, +0.059] | CI contains 0 |
| 4bit_i100 -> 4bit_i100_s2 | 6 / 7 | 1.000 | -0.004 [-0.035, +0.026] | -0.011 [-0.041, +0.031] | CI contains 0 |
| 4bit_r9k -> 4bit_r9k_s2 | 5 / 11 | 0.210 | -0.026 [-0.061, +0.009] | +0.056 [+0.003, +0.080] | differs from 0 |
| 4bit_i100 -> 4bit_r9k | 16 / 7 | 0.093 | +0.039 [-0.004, +0.078] | -0.045 [-0.077, -0.002] | differs from 0 |

## 2b. Paired tests on Banking77

| A -> B | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |
|---|---|---|---|---|---|
| 4bit_base -> 4bit_i100 | 67 / 22 | 0.0000 | +0.045 [+0.028, +0.063] | +0.060 [+0.037, +0.075] | differs from 0 |
| 4bit_base -> 4bit_i300 | 78 / 35 | 0.0001 | +0.043 [+0.021, +0.065] | +0.115 [+0.089, +0.133] | differs from 0 |
| 4bit_base -> 4bit_r9k | 70 / 20 | 0.0000 | +0.050 [+0.032, +0.068] | +0.053 [+0.031, +0.069] | differs from 0 |
| 4bit_i100 -> 4bit_r9k | 24 / 19 | 0.5424 | +0.005 [-0.008, +0.018] | -0.006 [-0.020, +0.008] | CI contains 0 |

## 3. 'yes' bias and catch-all options (JevBench)

'yes' bias is measured on the binary yes/no items: predicted yes, gold yes (= correct yes + wrong no), mean P(yes). Catch-all = options named other/unknown/none/none of these/...; we report how many items have one, the mean probability mass on it, and how often it is the prediction.

| run | binary items | predicted yes | gold yes | mean P(yes) | items with catch-all | mean catch-all mass | predicted catch-all |
|---|---|---|---|---|---|---|---|
| 4bit_base | 74 | 55 | 35 | 0.656 | 24 | 0.282 | 7 |
| 4bit_i100 | 74 | 53 | 35 | 0.673 | 24 | 0.326 | 8 |
| 4bit_i100_s2 | 74 | 54 | 35 | 0.693 | 24 | 0.296 | 6 |
| 4bit_i300 | 74 | 62 | 35 | 0.784 | 24 | 0.279 | 6 |
| 4bit_r9k | 74 | 56 | 35 | 0.662 | 24 | 0.263 | 6 |
| 4bit_r9k_s2 | 74 | 55 | 35 | 0.664 | 24 | 0.267 | 7 |
| bf16_base | 74 | 51 | 35 | 0.599 | 24 | 0.298 | 7 |
| bf16_i100c20 | 74 | 54 | 35 | 0.653 | 24 | 0.299 | 7 |

## 4. Failure analysis

### 4a. Accuracy and ECE per family (JevBench)

| family | n | 4bit_base | 4bit_i100 | 4bit_i100_s2 | 4bit_i300 | 4bit_r9k | 4bit_r9k_s2 | bf16_base | bf16_i100c20 |
|---|---|---|---|---|---|---|---|---|---|
| adequacy | 12 | 0.58 | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | 0.58 | 0.75 |
| adversarial | 6 | 0.83 | 0.83 | 0.67 | 0.83 | 0.83 | 0.67 | 0.83 | 0.83 |
| ambiguous | 7 | 0.43 | 0.43 | 0.57 | 0.43 | 0.57 | 0.57 | 0.57 | 0.71 |
| extraction | 24 | 0.88 | 0.92 | 0.92 | 0.96 | 0.92 | 0.96 | 0.88 | 0.92 |
| fact | 12 | 0.67 | 0.58 | 0.58 | 0.50 | 0.50 | 0.50 | 0.67 | 0.50 |
| intent | 24 | 0.96 | 1.00 | 0.96 | 1.00 | 1.00 | 1.00 | 0.96 | 1.00 |
| judge_hard | 17 | 0.47 | 0.53 | 0.53 | 0.47 | 0.53 | 0.53 | 0.59 | 0.59 |
| long_policy | 19 | 0.47 | 0.47 | 0.42 | 0.42 | 0.68 | 0.47 | 0.63 | 0.58 |
| multi_hop | 18 | 0.56 | 0.56 | 0.61 | 0.61 | 0.67 | 0.67 | 0.56 | 0.50 |
| ordinal | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| policy | 12 | 0.92 | 0.92 | 1.00 | 0.83 | 1.00 | 1.00 | 0.83 | 0.92 |
| probability | 10 | 0.40 | 0.40 | 0.30 | 0.40 | 0.40 | 0.40 | 0.40 | 0.40 |
| routing | 12 | 0.92 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| routing_hard | 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| temporal_numeric | 15 | 0.13 | 0.13 | 0.20 | 0.13 | 0.27 | 0.07 | 0.20 | 0.40 |
| tool_selection | 12 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| tradeoff | 6 | 0.50 | 0.50 | 0.33 | 0.50 | 0.33 | 0.50 | 0.50 | 0.50 |
| trap | 8 | 0.88 | 0.88 | 0.75 | 0.62 | 0.88 | 0.88 | 0.88 | 0.88 |

(accuracy per family; cells are per run, same order as the header)

### 4b. Most confident wrong answers (top 8 per tuned run)

**4bit_i100** (67 wrong):
- easy-fact-07 (fact) predicted `yes` with confidence 0.963
- hard-sol-b-judge_hard-02 (judge_hard) predicted `yes` with confidence 0.958
- hard-opus-b-ambiguous-07 (ambiguous) predicted `over_limit` with confidence 0.951
- hard-opus-a-temporal_numeric-06 (temporal_numeric) predicted `yes` with confidence 0.940
- original-adequacy-03-0 (adequacy) predicted `yes` with confidence 0.933
- hard-sol-b-judge_hard-08 (judge_hard) predicted `yes` with confidence 0.924
- easy-fact-11 (fact) predicted `yes` with confidence 0.915
- easy-fact-05 (fact) predicted `yes` with confidence 0.881

**4bit_i300** (70 wrong):
- easy-fact-07 (fact) predicted `yes` with confidence 0.989
- hard-sol-b-judge_hard-02 (judge_hard) predicted `yes` with confidence 0.988
- hard-opus-a-temporal_numeric-06 (temporal_numeric) predicted `yes` with confidence 0.980
- hard-sol-b-judge_hard-08 (judge_hard) predicted `yes` with confidence 0.974
- hard-opus-c-long_policy-11 (long_policy) predicted `yes` with confidence 0.974
- hard-sol-a-trap-13 (trap) predicted `yes` with confidence 0.974
- hard-opus-b-ambiguous-07 (ambiguous) predicted `over_limit` with confidence 0.974
- original-adequacy-03-0 (adequacy) predicted `yes` with confidence 0.971

**4bit_r9k** (58 wrong):
- easy-fact-07 (fact) predicted `yes` with confidence 0.971
- easy-fact-11 (fact) predicted `yes` with confidence 0.953
- hard-sol-b-judge_hard-02 (judge_hard) predicted `yes` with confidence 0.953
- hard-sol-b-judge_hard-08 (judge_hard) predicted `yes` with confidence 0.924
- hard-opus-b-ambiguous-07 (ambiguous) predicted `over_limit` with confidence 0.914
- original-adequacy-03-0 (adequacy) predicted `yes` with confidence 0.894
- hard-opus-c-long_policy-11 (long_policy) predicted `yes` with confidence 0.881
- easy-fact-03 (fact) predicted `yes` with confidence 0.852

**bf16_i100c20** (58 wrong):
- hard-sol-b-judge_hard-02 (judge_hard) predicted `yes` with confidence 0.958
- hard-opus-c-long_policy-11 (long_policy) predicted `yes` with confidence 0.958
- hard-sol-b-judge_hard-08 (judge_hard) predicted `yes` with confidence 0.958
- hard-opus-b-ambiguous-07 (ambiguous) predicted `over_limit` with confidence 0.950
- easy-fact-07 (fact) predicted `yes` with confidence 0.947
- hard-opus-a-temporal_numeric-09 (temporal_numeric) predicted `yes` with confidence 0.924
- hard-opus-a-probability-07 (probability) predicted `resolved_first_contact` with confidence 0.917
- original-adequacy-03-0 (adequacy) predicted `yes` with confidence 0.905

### 4c. Items won/lost versus the matching baseline, by family

| A -> B | family | won | lost |
|---|---|---|---|
| 4bit_base -> 4bit_i100 | adequacy | 1 | 0 |
| 4bit_base -> 4bit_i100 | extraction | 1 | 0 |
| 4bit_base -> 4bit_i100 | fact | 0 | 1 |
| 4bit_base -> 4bit_i100 | intent | 1 | 0 |
| 4bit_base -> 4bit_i100 | judge_hard | 1 | 0 |
| 4bit_base -> 4bit_i100 | long_policy | 2 | 2 |
| 4bit_base -> 4bit_i100 | multi_hop | 2 | 2 |
| 4bit_base -> 4bit_i100 | temporal_numeric | 1 | 1 |
| 4bit_base -> 4bit_i300 | adequacy | 1 | 0 |
| 4bit_base -> 4bit_i300 | extraction | 2 | 0 |
| 4bit_base -> 4bit_i300 | fact | 0 | 2 |
| 4bit_base -> 4bit_i300 | intent | 1 | 0 |
| 4bit_base -> 4bit_i300 | long_policy | 1 | 2 |
| 4bit_base -> 4bit_i300 | multi_hop | 2 | 1 |
| 4bit_base -> 4bit_i300 | policy | 1 | 2 |
| 4bit_base -> 4bit_i300 | routing | 1 | 0 |
| 4bit_base -> 4bit_i300 | temporal_numeric | 1 | 1 |
| 4bit_base -> 4bit_i300 | trap | 0 | 2 |
| 4bit_base -> 4bit_r9k | adequacy | 1 | 0 |
| 4bit_base -> 4bit_r9k | ambiguous | 1 | 0 |
| 4bit_base -> 4bit_r9k | extraction | 1 | 0 |
| 4bit_base -> 4bit_r9k | fact | 0 | 2 |
| 4bit_base -> 4bit_r9k | intent | 1 | 0 |
| 4bit_base -> 4bit_r9k | judge_hard | 1 | 0 |
| 4bit_base -> 4bit_r9k | long_policy | 5 | 1 |
| 4bit_base -> 4bit_r9k | multi_hop | 3 | 1 |
| 4bit_base -> 4bit_r9k | policy | 1 | 0 |
| 4bit_base -> 4bit_r9k | routing | 1 | 0 |
| 4bit_base -> 4bit_r9k | temporal_numeric | 2 | 0 |
| 4bit_base -> 4bit_r9k | tradeoff | 0 | 1 |
| bf16_base -> bf16_i100c20 | adequacy | 2 | 0 |
| bf16_base -> bf16_i100c20 | ambiguous | 1 | 0 |
| bf16_base -> bf16_i100c20 | extraction | 1 | 0 |
| bf16_base -> bf16_i100c20 | fact | 0 | 2 |
| bf16_base -> bf16_i100c20 | intent | 1 | 0 |
| bf16_base -> bf16_i100c20 | long_policy | 1 | 2 |
| bf16_base -> bf16_i100c20 | multi_hop | 0 | 1 |
| bf16_base -> bf16_i100c20 | policy | 1 | 0 |
| bf16_base -> bf16_i100c20 | temporal_numeric | 3 | 0 |

### 4d. Banking77: most frequent confusions (gold index -> predicted index) and confidence

- **4bit_base**: 337 wrong; mean confidence on wrong 0.517; wrong with conf>=0.9: 9.8%; top confusions 38->21 (x9), 43->38 (x9), 56->64 (x8), 7->27 (x8), 33->32 (x7)
- **4bit_i100**: 292 wrong; mean confidence on wrong 0.658; wrong with conf>=0.9: 20.9%; top confusions 38->21 (x8), 43->38 (x7), 44->21 (x7), 40->37 (x7), 25->14 (x6)
- **4bit_r9k**: 287 wrong; mean confidence on wrong 0.675; wrong with conf>=0.9: 22.6%; top confusions 43->38 (x10), 38->21 (x9), 7->27 (x7), 56->64 (x6), 35->27 (x6)
- **4bit_i300**: 294 wrong; mean confidence on wrong 0.748; wrong with conf>=0.9: 37.1%; top confusions 38->21 (x9), 43->38 (x8), 25->14 (x8), 44->21 (x7), 7->27 (x6)

## 5. Post-hoc temperature scaling (ECE10 on JevBench; accuracy unchanged by construction)

T is chosen to minimise the cross-entropy of 'is the top answer right' (confidence of the top option after p^(1/T), renormalised). Transfer: T fitted on the same model's Banking77 answers, applied to its JevBench answers (the OOD direction of the paper). CV: T fitted on 4/5 of JevBench, ECE measured on the other 1/5 (5 folds, pooled), averaged over 20 random splits.

| run | ECE10 raw | T from Banking77 | ECE10 after (transfer) | ECE10 after (5-fold CV in JevBench) |
|---|---|---|---|---|
| 4bit_base | 0.1091 | 1.14 | 0.0925 | 0.0714 |
| 4bit_i100 | 0.1058 | 1.35 | 0.0901 | 0.0964 |
| 4bit_i100_s2 | 0.0945 | n/a | n/a | 0.0652 |
| 4bit_i300 | 0.1573 | 1.59 | 0.0973 | 0.0557 |
| 4bit_r9k | 0.0613 | 1.35 | 0.0475 | 0.0500 |
| 4bit_r9k_s2 | 0.1172 | n/a | n/a | 0.0465 |
| bf16_base | 0.0512 | n/a | n/a | 0.0580 |
| bf16_i100c20 | 0.0826 | n/a | n/a | 0.0594 |