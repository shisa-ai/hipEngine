# Qwen4Exp `fast` profile: WMMA prefill attention

Date: 2026-10-13 UTC. Host: AMD Radeon 8060S / gfx1151, AMD Ryzen AI Max+
395, 128 GB unified memory, machine `55ea6c509d0b49eea8de7094a1023668`,
kernel `7.2.9-2-cachyos`, ROCm 10.0.0 (TheRock), `tuned-adm` profile
`accelerator-performance`, `scaling_governor=performance`. Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, four shards.

The opt-in `fast` execution profile runs the prefill attention of the 12
sparse-attention layers on a WMMA kernel
(`hipengine/kernels/hip_gfx1100/attention/qwen4_exp_qsa_wmma.hip`): queries,
keys and values as F16 fragments on the matrix units, F32 score and output
accumulation, online softmax. It replaces two scalar FP32 kernels, one for
the dense causal window and one for the masked sparse window. The paged BF16
K/V cache is gathered to contiguous F16 rows and each query's selected
positions are folded into a block bitmask before the launch. Decode
attention is unchanged.

`fast` changes arithmetic and is **not certified against the production
numerical envelope**. Its figures are `fast` figures, not production figures.

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
| `fast` before this change | 707.3 | 811.4 | 825.0 | 20.57 |
| `fast`, this change | **728.9** | **868.4** | **1002.2** | 20.58 |
| `fast` over `production` | 2.38x | 2.63x | 2.99x | unchanged |

| Case | production prefill | fast prefill | ratio | production decode | fast decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 304.9 | 724.2 | 2.38x | 20.92 | 20.93 |
| code-p1024 | 328.2 | 860.4 | 2.62x | 20.17 | 20.17 |
| code-p4096 | 335.1 | 1000.2 | 2.99x | 20.57 | 20.58 |
| general_en-p512 | 306.8 | 728.7 | 2.38x | 20.93 | 20.93 |
| general_en-p1024 | 331.2 | 872.6 | 2.64x | 20.17 | 20.17 |
| general_en-p4096 | 335.5 | 1002.6 | 2.99x | 20.57 | 20.58 |
| general_ja-p512 | 309.7 | 742.1 | 2.40x | 20.93 | 20.93 |
| general_ja-p1024 | 332.2 | 879.8 | 2.65x | 20.17 | 20.17 |
| general_ja-p4096 | 335.3 | 1004.2 | 2.99x | 20.58 | 20.58 |
| mixed_ja_en-p512 | 304.4 | 720.6 | 2.37x | 20.93 | 20.93 |
| mixed_ja_en-p1024 | 328.3 | 860.9 | 2.62x | 20.17 | 20.17 |
| mixed_ja_en-p4096 | 334.9 | 1001.7 | 2.99x | 20.57 | 20.58 |

All 12 cases improve; the run is repeat-deterministic. The gain grows with
prompt length because attention cost grows with context while the other
families are linear in tokens.

For reference, the external engines measured on this host with the same
protocol on 2026-10-12: halo-box `strix-llama.cpp` master `6aa81b88e`
759.3 / 1109.1 / 1446.4 and gufo `v0.12.0` 1095.0 / 1482.2 / 1736.4 tok/s,
with decode at p4096 of 25.1 and 28.1 tok/s
([pass-1 artifact](../2026-10-12-qwen4exp-cross-engine-pass1/README.md)).

## Quality characterization against the strict teacher

18-prompt battery, 32-step free greedy trajectories, top-20 logprobs, 2
repeats, scored against the hipEngine `strict` teacher captured on
2026-10-12. Score: [`quality-score.json`](quality-score.json).

| Profile | Greedy identity (exact / 18) | Aligned top-1 agreement | Top-20 mean KL | Top-20 max KL | Repeats exact |
| --- | ---: | ---: | ---: | ---: | --- |
| `production` | 15 | 1.00 | 0.0052 | 0.62 | yes |
| `fast` before this change | 13 | 1.00 | 0.0561 | 1.22 | yes |
| `fast`, this change | 13 | 1.00 | 0.0450 | 1.29 | yes |

The KL columns are the truncated top-20 proxy of the cross-engine protocol,
not the full-vocabulary teacher-forced KL of the production envelope. Under
the same scoring on the same host, halo-box measured 11/18 with mean KL
0.241 and gufo 13/18. The battery's prompts are short, so it exercises the
dense attention window more than the sparse one; the sparse window is
covered by the kernel test and by the 4096-token canonical cases'
determinism, not yet by a long-context quality gate.

## Kernel correctness

`tests/test_gpu_qwen4exp_qsa_wmma.py`: against a float64 reference, dense
causal windows with and without prior context (including a single-row
launch), and masked sparse windows where each query sees a random subset of
complete four-token blocks plus its own incomplete tail block, with shuffled
selection lists and a permuted block table. Relative L2 error <= 2e-2, no
write past the launched rows, and the BF16-to-F16 gather is bit-exact.

## Per-family kernel time (code-p4096, `rocprofv3 --kernel-trace`)

One profiled capture holding a warmup and a measured 4096-token prefill;
figures are the window halved. Kernel total 4049 ms against a 4133 ms wall
(profiled rate 991.0 tok/s). Attribution:
[`prof/attribution.json`](prof/attribution.json). The `production` and gufo
columns are the 2026-10-12 pass-1 capture.

| Family (ms per prefill) | production | fast | gufo `v0.12.0` |
| --- | ---: | ---: | ---: |
| Attention kernels (dense + sparse windows) | about 995 | 112 | 112.5 (whole family) |
| Attention block scoring and K/V gather, mask build | 130 | 132 | |
| Dense projections | 3045.1 | 1283 | 841.0 |
| Routed experts | 4621.1 | 692 | 812.3 |
| GDN | about 749 | 688 | 208.2 |
| Hyper-connection elementwise (sigmoid, gated mean, write, norms) | in 1581.0 | 582 | in 325.0 |
| Router logits | 564 | 87 | in 6.1 |
| MoE scatter and reduce | 148 | 152 | |
| **Kernel total** | **12185.5** | **4049** | **2372.8** |

`WmmaCausalAttentionKernel<4, 16, true, false>` (sparse window) ran 12 times
per prefill for 74.3 ms and `<16, 16, false, false>` (dense
window) 12 times for 37.2 ms; `PagedKvToF16Kernel` and
`SelectionMaskKernel` ran 12 times each for 0.3 and 2.7 ms.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md), HIPENGINE_HIP_ARCH=gfx1151.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
RES=benchmarks/results/2026-10-14-qwen4exp-fast-qsa-wmma
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

python -m pytest tests/test_gpu_qwen4exp_qsa_wmma.py
```
