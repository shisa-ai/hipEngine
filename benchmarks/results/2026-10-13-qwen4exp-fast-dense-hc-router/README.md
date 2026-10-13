# Qwen4Exp `fast` profile: all-layer F16 dense route, hyper-connection wide route, warp-reduce router

Date: 2026-10-13. Host: AMD Radeon 8060S / gfx1151, AMD Ryzen AI Max+ 395,
128 GB unified memory, machine `55ea6c509d0b49eea8de7094a1023668`, kernel
`7.2.9-2-cachyos`, ROCm 10.0.0 (TheRock), `tuned-adm` profile
`accelerator-performance`, `scaling_governor=performance`. Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, four shards.

The opt-in `fast` execution profile gains three route changes on top of its
wide-tile routed-expert GEMMs. All three use kernels that were already
registered in this tree; `fast` widens where they run:

- **Dense Q8_0 projections on the F16 wide route at every layer.**
  `production` certifies that route at layers 13-47 and runs the exact F32
  column-tile chain below; `fast` runs it at 0-47. This also carries the
  hyper-connection down projection (K 10240, N 320).
- **Hyper-connection up projection on the F16 wide route** instead of the
  exact F32 fused sigmoid/mean kernel.
- **Router logits on the warp-reduce F32 producer** instead of the ordered
  DPP tile. Both are FP32; they differ in reduction order.

`fast` changes arithmetic and is **not certified against the production
numerical envelope**. Its figures are `fast` figures, not production figures.

## Prefill and decode (tok/s)

Canonical protocol
([`docs/CROSS-ENGINE-PROTOCOL.md`](../../../docs/CROSS-ENGINE-PROTOCOL.md)):
committed 12-case exact-token fixture, prefill-only wall, 1 warmup + 3
measured repetitions, median per case, equal-weight mean; decode over 128
transitions; BF16 KV, chunk 4096, compact capture, warm PLE. The `production`
row is the run of the same day, host and protocol recorded in
[`../2026-10-13-qwen4exp-fast-routed-f16/`](../2026-10-13-qwen4exp-fast-routed-f16/README.md);
the `fast` row was measured about two hours later in the same boot.

| Profile | p512 | p1024 | p4096 | Decode at p4096 |
| --- | ---: | ---: | ---: | ---: |
| `production` | 306.4 | 330.0 | 335.2 | 20.57 |
| `fast`, routed-expert GEMMs only | 466.3 | 502.2 | 495.1 | 20.59 |
| `fast`, this change | **684.8** | **754.4** | **754.7** | 20.60 |
| `fast` over `production` | 2.23x | 2.29x | 2.25x | unchanged |

| Case | production prefill | fast prefill | ratio | production decode | fast decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 304.9 | 680.5 | 2.23x | 20.92 | 20.94 |
| code-p1024 | 328.2 | 748.4 | 2.28x | 20.17 | 20.18 |
| code-p4096 | 335.1 | 753.6 | 2.25x | 20.57 | 20.60 |
| general_en-p512 | 306.8 | 684.0 | 2.23x | 20.93 | 20.94 |
| general_en-p1024 | 331.2 | 757.1 | 2.29x | 20.17 | 20.19 |
| general_en-p4096 | 335.5 | 755.4 | 2.25x | 20.57 | 20.60 |
| general_ja-p512 | 309.7 | 697.2 | 2.25x | 20.93 | 20.94 |
| general_ja-p1024 | 332.2 | 762.7 | 2.30x | 20.17 | 20.18 |
| general_ja-p4096 | 335.3 | 755.6 | 2.25x | 20.58 | 20.60 |
| mixed_ja_en-p512 | 304.4 | 677.4 | 2.23x | 20.93 | 20.94 |
| mixed_ja_en-p1024 | 328.3 | 749.4 | 2.28x | 20.17 | 20.18 |
| mixed_ja_en-p4096 | 334.9 | 754.4 | 2.25x | 20.57 | 20.60 |

All 12 cases improve; the run is repeat-deterministic
(`all_cases_deterministic: true`).

## How the routes were chosen

Single-process sweeps on `code-p4096`, `fast` profile, routes toggled after
bind, best of two samples per arm (diagnostic, tok/s):

| Arm (cumulative) | Prefill | Kept |
| --- | ---: | --- |
| routed-expert GEMMs only | 494.1 | baseline |
| + dense F16 wide route at layers 0-47 | 612.0 | yes |
| + int8-WMMA hyper-connection down and up | 679.8 | superseded below |
| hyper-connection down on the wide route instead of int8 WMMA | 685.8 | yes |
| hyper-connection up on the wide route instead of int8 WMMA | 717.1 | yes |
| router on the warp-reduce producer | 718.4 (from 686.6) | yes |
| all kept routes together | 754.0 | |

Measured and not kept, each relative to the arm it was added to: QSA flash
prefill at all layers (679.3 to 663.1, and 754.0 to 734.6 in the later
sweep), GDN peer prefill at all layers (663.1 to 615.6), staged
hyper-connection (no change), the int8 MMQ dense route for the remaining
dense shapes (no change).

## Quality characterization against the strict teacher

18-prompt battery (mtp-bench 10 + GDN heldouts 8), 32-step free greedy
trajectories, top-20 logprobs, 2 repeats, scored by
`scripts/cross_engine_quality_gate.py` against the hipEngine `strict`
teacher captured on 2026-10-12. Score: [`quality-score.json`](quality-score.json).

| Profile | Greedy identity (exact / 18) | Aligned top-1 agreement | Top-20 mean KL | Top-20 max KL | Repeats exact |
| --- | ---: | ---: | ---: | ---: | --- |
| `production` | 15 | 1.00 | 0.0052 | 0.62 | yes |
| `fast`, routed-expert GEMMs only | 13 | 1.00 | 0.0550 | 2.19 | yes |
| `fast`, this change | 12 | 1.00 | 0.0362 | 0.94 | yes |

The KL columns are the truncated top-20 proxy of the cross-engine protocol:
comparable between rows scored the same way, not comparable to the
full-vocabulary teacher-forced KL of the production envelope. Under the same
scoring on the same host, halo-box `strix-llama.cpp` master `6aa81b88e`
measured 11/18 with mean KL 0.241 and gufo `v0.12.0` 13/18 with no logprob
surface.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md), HIPENGINE_HIP_ARCH=gfx1151.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
RES=benchmarks/results/2026-10-13-qwen4exp-fast-dense-hc-router
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
```
