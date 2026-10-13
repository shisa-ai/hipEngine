# Qwen4Exp `fast` profile: small-matrix F16 GEMM for the indexer projections and the router

Date: 2026-10-13 UTC. Host: AMD Radeon 8060S / gfx1151, AMD Ryzen AI Max+
395, 128 GB unified memory, machine `55ea6c509d0b49eea8de7094a1023668`,
kernel `7.2.9-2-cachyos`, ROCm 10.0.0 (TheRock), `tuned-adm` profile
`accelerator-performance`, `scaling_governor=performance`. Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, four shards.

The opt-in `fast` execution profile moves two groups of small projections
whose weights are stored uncompressed onto a new F16-operand WMMA GEMM
(`hipengine/kernels/hip_gfx1100/linear/dense_f16_gemm.hip`): FP32
activations and FP32 or BF16 weights are rounded to F16 while staged in LDS,
multiplied with F16 WMMA and F32 accumulation, and written as FP32.

- **Indexer query and key projections** (BF16 weights, 512 x 2560 and
  128 x 2560, 12 attention layers): previously a scalar FP32 row-at-a-time
  GEMV, 285 ms per 4096-token prefill; now 30 ms.
- **Router logits** (FP32 weights, 512 x 2560, 48 layers): previously the
  fastest FP32 producer at 295 ms per 4096-token prefill.

`fast` changes arithmetic and is **not certified against the production
numerical envelope**. The router change in particular rounds its operands to
F16, so expert selection can differ from the FP32 producers on near-ties.

## Prefill and decode (tok/s)

Canonical protocol
([`docs/CROSS-ENGINE-PROTOCOL.md`](../../../docs/CROSS-ENGINE-PROTOCOL.md)):
committed 12-case exact-token fixture, prefill-only wall, 1 warmup + 3
measured repetitions, median per case, equal-weight mean; decode over 128
transitions; BF16 KV, chunk 4096, compact capture, warm PLE. All rows were
measured on this host in the same boot; the `production` row is the run
recorded in
[`../2026-10-13-qwen4exp-fast-routed-f16/`](../2026-10-13-qwen4exp-fast-routed-f16/README.md).

| Profile | p512 | p1024 | p4096 | Decode at p4096 |
| --- | ---: | ---: | ---: | ---: |
| `production` | 306.4 | 330.0 | 335.2 | 20.57 |
| `fast` before this change | 684.8 | 754.4 | 754.7 | 20.60 |
| `fast`, indexer projections only | 707.8 | 788.5 | 794.2 | 20.59 |
| `fast`, indexer projections and router | **707.3** | **811.4** | **825.0** | 20.57 |
| `fast` over `production` | 2.31x | 2.46x | 2.46x | unchanged |

| Case | production prefill | fast prefill | ratio | production decode | fast decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 304.9 | 702.8 | 2.31x | 20.92 | 20.92 |
| code-p1024 | 328.2 | 803.9 | 2.45x | 20.17 | 20.16 |
| code-p4096 | 335.1 | 823.2 | 2.46x | 20.57 | 20.57 |
| general_en-p512 | 306.8 | 706.9 | 2.30x | 20.93 | 20.92 |
| general_en-p1024 | 331.2 | 814.9 | 2.46x | 20.17 | 20.16 |
| general_en-p4096 | 335.5 | 825.7 | 2.46x | 20.57 | 20.57 |
| general_ja-p512 | 309.7 | 719.6 | 2.32x | 20.93 | 20.92 |
| general_ja-p1024 | 332.2 | 821.6 | 2.47x | 20.17 | 20.16 |
| general_ja-p4096 | 335.3 | 826.1 | 2.46x | 20.58 | 20.57 |
| mixed_ja_en-p512 | 304.4 | 699.7 | 2.30x | 20.93 | 20.92 |
| mixed_ja_en-p1024 | 328.3 | 805.1 | 2.45x | 20.17 | 20.16 |
| mixed_ja_en-p4096 | 334.9 | 824.8 | 2.46x | 20.57 | 20.58 |

All 12 cases improve; every run is repeat-deterministic. The router change
is neutral at 512 tokens and positive at 1024 and 4096.

## Scope of the GEMM

The dispatch moves a projection only when it has at least 64 rows, a K
extent that is a multiple of 32, FP32 activations and output, and at least
96 output columns. Narrower matrices stay on the scalar GEMV. Single-process
sweep on the `fast` profile, best of two samples per arm (diagnostic, tok/s):

| Minimum output columns | code-p512 | code-p4096 |
| --- | ---: | ---: |
| GEMM off | 682.1 | 754.5 |
| 96 (selected) | 703.8 | 791.5 |
| 32 (adds the 48-column GDN gate projections) | 690.8 | 803.1 |
| 1 (adds the 4-column hyper-connection inject) | 591.9 | 778.8 |

## Quality characterization against the strict teacher

18-prompt battery, 32-step free greedy trajectories, top-20 logprobs, 2
repeats, scored against the hipEngine `strict` teacher captured on
2026-10-12. Score: [`quality-score.json`](quality-score.json).

| Profile | Greedy identity (exact / 18) | Aligned top-1 agreement | Top-20 mean KL | Top-20 max KL | Repeats exact |
| --- | ---: | ---: | ---: | ---: | --- |
| `production` | 15 | 1.00 | 0.0052 | 0.62 | yes |
| `fast` before this change | 12 | 1.00 | 0.0362 | 0.94 | yes |
| `fast`, this change | 13 | 1.00 | 0.0561 | 1.22 | yes |

The KL columns are the truncated top-20 proxy of the cross-engine protocol,
not the full-vocabulary teacher-forced KL of the production envelope. Under
the same scoring on the same host, halo-box `strix-llama.cpp` master
`6aa81b88e` measured 11/18 with mean KL 0.241 and gufo `v0.12.0` 13/18.

## Per-family kernel time (code-p4096, `rocprofv3 --kernel-trace`)

One profiled capture of the indexer-only composition (router still on the
FP32 warp-reduce producer), holding a warmup and a measured prefill; figures
are the window halved. Kernel total 5145 ms against a 5231 ms wall.
Attribution: [`prof/attribution.json`](prof/attribution.json). The
`production` and gufo columns are the 2026-10-12 pass-1 capture.

| Family (ms per prefill) | production | fast | gufo `v0.12.0` |
| --- | ---: | ---: | ---: |
| QSA attention | 1151.9 | 1167 | 112.5 |
| Dense projections | 3045.1 | 1281 | 841.0 |
| Routed experts | 4621.1 | 698 | 812.3 |
| GDN (plus 155 ms for its norm/gate kernel) | 593.5 | 587 | 208.2 |
| MoE router and reduce | 712.1 | 448 (295 of it router logits) | 6.1 |
| Hyper-connection elementwise (sigmoid, gated mean, write, norm) | in 1581.0 | about 580 | in 325.0 |
| **Kernel total** | **12185.5** | **5145** | **2372.8** |

`dense_f16_gemm_f32_out_kernel<unsigned short, 128, 64, 32>` ran 24 times per
prefill for 30.3 ms.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md), HIPENGINE_HIP_ARCH=gfx1151.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
RES=benchmarks/results/2026-10-13-qwen4exp-fast-indexer-f16-gemm
P1=benchmarks/results/2026-10-12-qwen4exp-cross-engine-pass1/raw

python scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root "$MODEL" --output $RES/hipengine-fast.json \
  --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
  --execution-profile fast

python scripts/cross_engine_quality_gate.py hipengine --model-root "$MODEL" \
  --engine-label hipengine-fast --profile fast \
  --output $RES/quality-hipengine-fast.json
python scripts/cross_engine_quality_gate.py score \
  --teacher $P1/quality-hipengine-strict-teacher.json \
  --engines $P1/quality-hipengine-production.json $RES/quality-hipengine-fast.json \
  --model-root "$MODEL" --output $RES/quality-score.json

python -m pytest tests/test_gpu_dense_f16_gemm.py
```
