# Qwen4Exp prefill: two bit-identical kernels, and a 1K/8K/16K ladder

Date: 2026-10-13 UTC. Host: AMD Radeon 8060S / gfx1151, machine
`55ea6c509d0b49eea8de7094a1023668`, ROCm 10.0.0 (TheRock). Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, `production` profile, BF16 KV, prefill
chunk 4096, compact capture, warm PLE cache, one warmup and three measured
repetitions per case, 128 decode transitions. Tracked-clean tree at
`1a5f0069a`.

## What changed

Two kernels that change launch geometry and memory traffic only. Both are
bit-identical to what they replace and run in every profile.

- **Attention block scoring.** One 16-query x 16-block tile per workgroup
  from LDS-staged operands, replacing one 32-thread workgroup per (query,
  block) pair with four threads doing work. Kernel time per 4096-token
  prefill: 129 to 26 ms.
- **Hyper-connection sigmoid and gated mean.** The sigmoid is applied inside
  the gated-mean pass and the gate, which nothing reads afterwards, is not
  written. Kernel time per 4096-token prefill: 338 to 175 ms.

## Canonical 12-case protocol

`scripts/qwen4exp_canonical_ar_bench.py hipengine --prefill-chunk-size 4096
--capture-mode compact --ple-cache-mode warm --execution-profile production`.
Artifact: [`hipengine-production.json`](hipengine-production.json). Baseline:
the same protocol at `6d8306d1e` earlier the same day
([artifact](../2026-10-14-qwen4exp-production-f16-composition/hipengine-production.json)).

| Prompt tokens | Baseline prefill tok/s | Prefill tok/s | Change | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 769.2 | **779.4** | +1.33% | 20.94 |
| 1024 | 932.9 | **977.9** | +4.82% | 20.18 |
| 4096 | 1108.4 | **1190.3** | +7.39% | 20.59 |

All 12 cases are deterministic and produce the baseline's output token ids
([`regression-check.json`](regression-check.json), run with
`--require-identical-output`). Per-case prefill medians: p512 774.6 / 777.9 /
795.1 / 769.8, p1024 968.4 / 982.8 / 991.4 / 969.1, p4096 1187.3 / 1191.8 /
1192.7 / 1189.9 for code / general_en / general_ja / mixed_ja_en.

## 1K / 8K / 16K ladder

Fixture `benchmarks/fixtures/qwen4exp_canonical_ar_p1024_p8192_p16384.json`
(sha256 `352beade…`), built by the same recipe as the canonical fixture; its
four 1024-token cases are token-identical to the canonical ones. Same command
with `--fixture`. Artifact:
[`hipengine-production-p1024-p8192-p16384.json`](hipengine-production-p1024-p8192-p16384.json).

| Prompt tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 1024 | 978.7 | 20.18 |
| 8192 | **1158.8** | 20.53 |
| 16384 | **1133.1** | 20.43 |

All 12 cases are deterministic. Per-case prefill medians: p8192 1156.8 /
1159.8 / 1160.4 / 1158.3, p16384 1132.4 / 1133.5 / 1134.8 / 1132.1. This is
the first measurement at these lengths, so there is no earlier row to compare
with, and the external engines have not been measured at 8K or 16K.

## Kernel attribution after these changes

`rocprofv3 --kernel-trace` on one code-p4096 prefill
([`attribution-code-p4096.json`](attribution-code-p4096.json); the trace
window holds two prefills, figures below are per prefill): 3409 ms of kernel
time, against 3657 before these two kernels and 2373 for gufo `v0.12.0`
measured on this host on 2026-10-12.

| Largest kernels | ms | Launches |
| --- | ---: | ---: |
| F16 dense wide projections | 649 | 302 |
| Routed gate/up GEMM | 409 | 47 |
| F16 dense wide, hyper-connection down (K 10240, N 320) | 249 | 96 |
| Routed down GEMM (Q5_1) | 246 | 43 |
| Tiled GDN recurrence | 245 | 36 |
| Scalar F32 GEMV, 4- and 48-column projections | 223 | 216 |
| Fused sigmoid gated mean | 175 | 96 |
| Dense wide with F32 input, hyper-connection up | 154 | 96 |
| F32 to F16 activation conversion | 118 | 328 |
| Grouped RMS norm with F16 publication | 117 | 70 |

## Regression check

`scripts/qwen4exp_rate_regression_check.py --baseline <artifact> --candidate
<artifact>` compares two runs of the same fixture on the same host: weighted
prefill and decode per prompt length within a tolerance (3% by default),
determinism, and optionally identical output token ids. The two artifacts in
this directory are the baselines for the canonical and ladder fixtures.
