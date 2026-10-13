# Qwen4Exp `fast` profile: wide-tile routed-expert F16 GEMMs

Date: 2026-10-13. Host: AMD Radeon 8060S / gfx1151, AMD Ryzen AI Max+ 395,
128 GB unified memory, machine `55ea6c509d0b49eea8de7094a1023668`, kernel
`7.2.9-2-cachyos`, ROCm 10.0.0 (TheRock), `tuned-adm` profile
`accelerator-performance`, `scaling_governor=performance`. Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, four shards.

Two execution profiles of the same tree, measured back to back on the same
host with the canonical protocol
([`docs/CROSS-ENGINE-PROTOCOL.md`](../../../docs/CROSS-ENGINE-PROTOCOL.md)):
committed 12-case exact-token fixture, prefill-only wall, 1 warmup + 3
measured repetitions, median per case, equal-weight mean; decode over 128
transitions; BF16 KV, chunk 4096, compact capture, warm PLE.

- **`production`** (the shipped default): routed experts on the exact
  grouped chains (iu8 WMMA with risk tracking plus a sparse exact repair).
- **`fast`** (opt-in): the same plan with every grouped routed-expert
  dispatch on the wide-tile routed F16 GEMMs
  (`hipengine/kernels/hip_gfx1100/quant/qwen4_exp_routed_f16.hip`). They read
  the raw GGUF expert tensors, decode each weight tile to F16 once per 48-128
  grouped rows, and accumulate in F32 with no repair pass. No second weight
  copy is resident.

`fast` changes arithmetic and is **not certified against the production
numerical envelope**. Its figures are `fast` figures, not production figures.

## Prefill and decode (tok/s)

| Profile | p512 | p1024 | p4096 | Decode at p4096 |
| --- | ---: | ---: | ---: | ---: |
| `production` | 306.4 | 330.0 | 335.2 | 20.57 |
| `fast` | **466.3** | **502.2** | **495.1** | 20.59 |
| change | +52.2% | +52.2% | +47.7% | unchanged |

Per case (prefill median of 3, decode median of 3):

| Case | production prefill | fast prefill | change | production decode | fast decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 304.9 | 463.5 | +52.1% | 20.92 | 20.93 |
| code-p1024 | 328.2 | 499.9 | +52.3% | 20.17 | 20.17 |
| code-p4096 | 335.1 | 495.0 | +47.7% | 20.57 | 20.58 |
| general_en-p512 | 306.8 | 467.0 | +52.3% | 20.93 | 20.93 |
| general_en-p1024 | 331.2 | 503.4 | +52.0% | 20.17 | 20.17 |
| general_en-p4096 | 335.5 | 495.3 | +47.7% | 20.57 | 20.59 |
| general_ja-p512 | 309.7 | 471.8 | +52.3% | 20.93 | 20.92 |
| general_ja-p1024 | 332.2 | 505.8 | +52.2% | 20.17 | 20.17 |
| general_ja-p4096 | 335.3 | 495.3 | +47.7% | 20.58 | 20.59 |
| mixed_ja_en-p512 | 304.4 | 462.9 | +52.1% | 20.93 | 20.93 |
| mixed_ja_en-p1024 | 328.3 | 499.9 | +52.3% | 20.17 | 20.17 |
| mixed_ja_en-p4096 | 334.9 | 494.8 | +47.7% | 20.57 | 20.59 |

All 12 cases improve; both profiles are repeat-deterministic
(`all_cases_deterministic: true`). Decode is unchanged because the routed
GEMMs serve grouped prefill dispatches only; single-token decode keeps its
owners. The 129-token greedy output is identical between the two profiles in
3 of 12 cases.

## Quality characterization against the strict teacher

18-prompt battery (mtp-bench 10 + GDN heldouts 8), 32-step free greedy
trajectories, top-20 logprobs, 2 repeats, scored by
`scripts/cross_engine_quality_gate.py` against the hipEngine `strict`
teacher captured on 2026-10-12
(`../2026-10-12-qwen4exp-cross-engine-pass1/raw/quality-hipengine-strict-teacher.json`).
Score: [`quality-score.json`](quality-score.json).

| Profile | Greedy identity (exact / 18) | Aligned top-1 agreement | Top-20 mean KL | Top-20 max KL | Repeats exact |
| --- | ---: | ---: | ---: | ---: | --- |
| `production` | 15 | 1.00 | 0.0052 | 0.62 | yes |
| `fast` | 13 | 1.00 | 0.0550 | 2.19 | yes |

The KL columns are the truncated top-20 proxy defined in the cross-engine
protocol: comparable between rows scored the same way, not comparable to the
full-vocabulary teacher-forced KL of the production envelope. Under the same
scoring on the same host, the external engines measured 11/18 with mean KL
0.241 (halo-box `strix-llama.cpp` master `6aa81b88e`) and 13/18 with no
logprob surface (gufo `v0.12.0`); see
[`../2026-10-12-qwen4exp-cross-engine-pass1/README.md`](../2026-10-12-qwen4exp-cross-engine-pass1/README.md).

## Kernel correctness

`tests/test_gpu_qwen4exp_routed_f16.py`: against the F32 CPU reference on
synthetic raw GGUF weights, for Q4_K and Q5_K paired gate/up (tiles 64 and
128) and Q5_1, Q8_0, Q4_K, Q5_K single projections (tiles 16, 48, 64), with
empty, sub-tile, exact-tile and multi-tile expert buckets and output widths
that are not tile multiples. Relative L2 error <= 6e-3 (BF16 output rounding
is about 2e-3), unit-variance softmax KL <= 0.05, argmax agreement >= 90%,
and no write past the grouped rows.

## Per-family kernel time (code-p4096, `rocprofv3 --kernel-trace`)

One profiled `fast` capture holding a warmup and a measured 4096-token
prefill; figures are the window halved (the two prefills are identical), and
the halved kernel sum of 8245 ms is 0.997 of the measured 8273.7 ms
wall. The `production` and gufo columns are the 2026-10-12 pass-1 capture on
the same host and case. Attribution: [`prof/attribution.json`](prof/attribution.json).

| Family (ms per prefill) | production | fast | gufo `v0.12.0` |
| --- | ---: | ---: | ---: |
| Routed experts, gate/up | 2654.6 | 420.9 | 812.3 (fused with down) |
| Routed experts, down | 1966.5 | 273.9 | |
| Dense projections | 3045.1 | 3056.2 | 841.0 |
| Hyper-connection | 1581.0 | 1589.4 | 325.0 |
| QSA attention | 1151.9 | 1152.5 | 112.5 |
| MoE router and reduce | 712.1 | 711.6 | 6.1 |
| GDN | 593.5 | 590.1 | 208.2 |
| **Kernel total** | **12185.5** | **8245.0** | **2372.8** |

The routed kernels ran under their expected names:
`RoutedF16GEMMKernel<WeightType 12, 128, 128, 2, true>` (Q4_K paired gate/up,
47 launches per prefill), `<13, 128, 128, 2, true>` (Q5_K, 1), `<7, 128, 64,
2, false>` (Q5_1 down, 43), `<8, 128, 64, 2, false>` (Q8_0 down, 5), and
`RoutedTileMapKernel` (96). Routed experts now cost 695 ms per prefill
against 4621 ms under `production`; the other families are unchanged, as
expected.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md), HIPENGINE_HIP_ARCH=gfx1151.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
RES=benchmarks/results/2026-10-13-qwen4exp-fast-routed-f16
P1=benchmarks/results/2026-10-12-qwen4exp-cross-engine-pass1/raw

for PROFILE in production fast; do
  python scripts/qwen4exp_canonical_ar_bench.py hipengine \
    --model-root "$MODEL" --output $RES/hipengine-$PROFILE.json \
    --prefill-chunk-size 4096 --capture-mode compact --ple-cache-mode warm \
    --execution-profile $PROFILE
done

python scripts/cross_engine_quality_gate.py hipengine --model-root "$MODEL" \
  --engine-label hipengine-fast --profile fast \
  --output $RES/quality-hipengine-fast.json
python scripts/cross_engine_quality_gate.py score \
  --teacher $P1/quality-hipengine-strict-teacher.json \
  --engines $P1/quality-hipengine-production.json $RES/quality-hipengine-fast.json \
  --model-root "$MODEL" --output $RES/quality-score.json

python -m pytest tests/test_gpu_qwen4exp_routed_f16.py
```
