# Qwen3-0.6B analysis (all numbers computed from the per-item files)

Paper (arXiv 2610.02076 v3), 0.6B: original 56.7 / ECE .278 / Banking77 22.2; Intent-1ep 53.7 / .204 / 64.0; Reason-9k 55.0 / .118 / 60.4.

## 1. JevBench-231

| run | acc | ECE10 | harness acc | harness ECE | predicted yes (of 74 binary; gold yes 35) | mean P(yes) | predicted catch-all (of 24 items) | mean catch-all mass |
|---|---|---|---|---|---|---|---|---|
| base_paperA | 0.5671 | 0.2768 | 0.5671 | 0.2768 | 61 | 0.778 | 3 | 0.144 |
| base_default | 0.5931 | 0.2318 | 0.5931 | 0.2318 | 52 | 0.678 | 3 | 0.133 |
| intent100 | 0.5584 | 0.1707 | 0.5584 | 0.1707 | 61 | 0.721 | 3 | 0.111 |
| intent1ep_96pct_ckpt | 0.5368 | 0.2280 | 0.5368 | 0.2280 | 68 | 0.839 | 3 | 0.089 |
| intent1ep_full_seed1 | 0.5498 | 0.2237 | 0.5498 | 0.2237 | 64 | 0.822 | 3 | 0.096 |
| intent1ep_full_seed2 | 0.5455 | 0.2333 | 0.5455 | 0.2333 | 68 | 0.869 | 3 | 0.111 |
| reason9k_seed1 | 0.5671 | 0.1433 | 0.5671 | 0.1433 | 46 | 0.583 | 3 | 0.132 |
| reason9k_seed2 | 0.6104 | 0.0895 | 0.6104 | 0.0895 | 46 | 0.582 | 3 | 0.108 |
| reason9k_seed3 | 0.5974 | 0.1083 | 0.5974 | 0.1083 | 46 | 0.570 | 3 | 0.157 |

## 1b. Banking77 (1000 questions)

| run | acc | ECE10 | mean confidence |
|---|---|---|---|
| base_paperA | 0.1940 | 0.4466 | 0.641 |
| intent100 | 0.4740 | 0.0756 | 0.483 |
| intent1ep_96pct_ckpt | 0.5920 | 0.1365 | 0.724 |
| intent1ep_full_seed1 | 0.5810 | 0.1756 | 0.757 |
| intent1ep_full_seed2 | 0.5710 | 0.1955 | 0.767 |
| reason9k_seed1 | 0.4710 | 0.0938 | 0.565 |
| reason9k_seed2 | 0.4950 | 0.0884 | 0.566 |
| reason9k_seed3 | 0.4770 | 0.1142 | 0.591 |

## 2. Paired tests vs the untrained 0.6B (paperA), JevBench (95% bootstrap CI, 4000 resamples)

| tuned run | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |
|---|---|---|---|---|---|
| intent100 | 12 / 14 | 0.845 | -0.009 [-0.052, +0.035] | -0.106 [-0.152, -0.062] | differs from 0 |
| intent1ep_96pct_ckpt | 16 / 23 | 0.337 | -0.030 [-0.082, +0.022] | -0.049 [-0.103, +0.001] | CI contains 0 |
| intent1ep_full_seed1 | 16 / 20 | 0.618 | -0.017 [-0.069, +0.035] | -0.053 [-0.103, -0.002] | differs from 0 |
| intent1ep_full_seed2 | 19 / 24 | 0.542 | -0.022 [-0.078, +0.035] | -0.044 [-0.100, +0.007] | CI contains 0 |
| reason9k_seed1 | 19 / 19 | 1.000 | +0.000 [-0.052, +0.052] | -0.133 [-0.190, -0.079] | differs from 0 |
| reason9k_seed2 | 25 / 15 | 0.154 | +0.043 [-0.009, +0.100] | -0.187 [-0.229, -0.117] | differs from 0 |
| reason9k_seed3 | 21 / 14 | 0.311 | +0.030 [-0.022, +0.078] | -0.169 [-0.211, -0.106] | differs from 0 |

## 2b. Paired tests vs the untrained 0.6B (paperA), Banking77

| tuned run | won / lost | sign test p | acc diff [CI] | ECE diff [CI] | ECE verdict |
|---|---|---|---|---|---|
| intent100 | 299 / 19 | 0.0000 | +0.280 [+0.250, +0.310] | -0.371 [-0.400, -0.332] | differs from 0 |
| intent1ep_96pct_ckpt | 407 / 9 | 0.0000 | +0.398 [+0.365, +0.429] | -0.310 [-0.338, -0.276] | differs from 0 |
| intent1ep_full_seed1 | 404 / 17 | 0.0000 | +0.387 [+0.355, +0.420] | -0.271 [-0.301, -0.239] | differs from 0 |
| intent1ep_full_seed2 | 393 / 16 | 0.0000 | +0.377 [+0.345, +0.408] | -0.251 [-0.283, -0.220] | differs from 0 |
| reason9k_seed1 | 291 / 14 | 0.0000 | +0.277 [+0.248, +0.307] | -0.353 [-0.378, -0.319] | differs from 0 |
| reason9k_seed2 | 319 / 18 | 0.0000 | +0.301 [+0.270, +0.332] | -0.358 [-0.383, -0.322] | differs from 0 |
| reason9k_seed3 | 300 / 17 | 0.0000 | +0.283 [+0.253, +0.313] | -0.332 [-0.359, -0.302] | differs from 0 |