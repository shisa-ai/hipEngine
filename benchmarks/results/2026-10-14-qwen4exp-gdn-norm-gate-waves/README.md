# Qwen4Exp GDN: eight-wave norm/gate (all profiles) and the tiled recurrence at every layer (`fast`)

Date: 2026-10-13 UTC. Host: AMD Radeon 8060S / gfx1151, AMD Ryzen AI Max+
395, 128 GB unified memory, machine `55ea6c509d0b49eea8de7094a1023668`,
kernel `7.2.9-2-cachyos`, ROCm 10.0.0 (TheRock), `tuned-adm` profile
`accelerator-performance`, `scaling_governor=performance`. Model: unsloth
`Qwen3.8-Flash-Next` `UD-Q4_K_XL`, four shards.

Two changes to the gated-delta-network (GDN) layers' prefill:

1. **Eight-wave norm/gate, every profile, bit-identical.** The per-head RMS
   norm, weight and sigmoid gate after the tiled recurrence launched one
   32-thread workgroup per (token, value head): 196,608 workgroups per layer
   at 4096 tokens, about 10.4 ms per layer. The new kernel runs the same
   arithmetic with eight pairs per workgroup, one per wave: 1.3 ms per layer.
2. **Tiled recurrence at every GDN layer, `fast` only.** `production`
   certifies the tiled multi-column recurrence for layers 27-47 and runs the
   register-state recurrence below (about 20.5 ms per layer against 6.8);
   `fast` runs the tiled recurrence at all 36 GDN layers. The two
   recurrences differ in summation order.

## Prefill and decode (tok/s)

Canonical protocol
([`docs/CROSS-ENGINE-PROTOCOL.md`](../../../docs/CROSS-ENGINE-PROTOCOL.md)):
committed 12-case exact-token fixture, prefill-only wall, 1 warmup + 3
measured repetitions, median per case, equal-weight mean; decode over 128
transitions; BF16 KV, chunk 4096, compact capture, warm PLE. The two rows
marked "this change" were measured back to back, `fast` first.

| Profile | p512 | p1024 | p4096 | Decode at p4096 |
| --- | ---: | ---: | ---: | ---: |
| `production` before (same boot, about eight hours earlier) | 306.4 | 330.0 | 335.2 | 20.57 |
| `production`, this change | 306.6 | 331.4 | 337.2 | 20.55 |
| `fast` before | 728.9 | 868.4 | 1002.2 | 20.58 |
| `fast`, this change | **769.8** | **935.1** | **1104.8** | 20.60 |
| `fast` over `production`, this change | 2.51x | 2.82x | 3.28x | unchanged |

`production` generates identical tokens in all 12 cases before and after
(the kernel is bit-identical), and its prefill moves +0.07 / +0.42 / +0.60%,
which is inside this host's cross-run variation over hours; the direct
evidence for the kernel is the profile below. `fast` improves in all 12
cases (+5.6 / +7.7 / +10.2% in aggregate).

| Case | production prefill | fast prefill | ratio | production decode | fast decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| code-p512 | 305.5 | 763.8 | 2.50x | 20.92 | 20.94 |
| code-p1024 | 329.6 | 925.4 | 2.81x | 20.16 | 20.19 |
| code-p4096 | 336.9 | 1102.8 | 3.27x | 20.54 | 20.60 |
| general_en-p512 | 306.0 | 769.7 | 2.52x | 20.92 | 20.95 |
| general_en-p1024 | 332.8 | 940.1 | 2.82x | 20.16 | 20.18 |
| general_en-p4096 | 337.4 | 1105.5 | 3.28x | 20.54 | 20.60 |
| general_ja-p512 | 310.5 | 785.2 | 2.53x | 20.92 | 20.94 |
| general_ja-p1024 | 333.3 | 947.9 | 2.84x | 20.16 | 20.19 |
| general_ja-p4096 | 337.4 | 1107.2 | 3.28x | 20.54 | 20.60 |
| mixed_ja_en-p512 | 304.3 | 760.4 | 2.50x | 20.92 | 20.94 |
| mixed_ja_en-p1024 | 330.0 | 927.0 | 2.81x | 20.16 | 20.19 |
| mixed_ja_en-p4096 | 337.0 | 1103.6 | 3.27x | 20.55 | 20.60 |

## Kernel evidence (code-p4096, `rocprofv3 --kernel-trace`, `fast`)

Per prefill, window halved; kernel total 3657 ms against a 3744 ms wall
(profiled rate 1094.1 tok/s). Attribution:
[`prof/attribution.json`](prof/attribution.json).

| GDN kernel (ms per prefill) | Before | After |
| --- | ---: | ---: |
| Register-state recurrence, 21 layers | 430.6 | 0 (not launched) |
| Tiled recurrence | 102.4 (15 layers) | 245.4 (36 layers) |
| Norm/gate after the tiled recurrence | 155.5 (15 launches, per-pair) | 47.6 (36 launches, eight-wave) |
| Convolution | 53.8 | 54.0 |
| **GDN total** | **742** | **347** |

Remaining kernel time by family: dense projections 1283 ms, routed
experts 694, hyper-connection elementwise chain 583, GDN 347,
attention with block scoring 302. gufo `v0.12.0` measured 2373 ms
total on this host and case on 2026-10-12.

## Quality characterization against the strict teacher (`fast`)

18-prompt battery, 32-step free greedy trajectories, top-20 logprobs, 2
repeats. Score: [`quality-score.json`](quality-score.json).

| Profile | Greedy identity (exact / 18) | Aligned top-1 agreement | Top-20 mean KL | Top-20 max KL | Repeats exact |
| --- | ---: | ---: | ---: | ---: | --- |
| `production` | 15 | 1.00 | 0.0052 | 0.62 | yes |
| `fast` before this change | 13 | 1.00 | 0.0450 | 1.29 | yes |
| `fast`, this change | 14 | 1.00 | 0.0493 | 1.70 | yes |

The KL columns are the truncated top-20 proxy of the cross-engine protocol.
`fast` is not certified against the production numerical envelope.

## Kernel correctness

`tests/test_gpu_qwen4exp_gdn_norm_gate_waves.py`: the eight-wave kernel's
output is bit-identical to the per-pair parent on five shapes, including
pair counts that do not fill the last workgroup, with no write past the
buffer. The existing GDN test files pass unchanged.

## Reproducing

```bash
# Environment: TheRock ROCm clean-process wrapper (docs/ENVS.md), HIPENGINE_HIP_ARCH=gfx1151.
MODEL=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
RES=benchmarks/results/2026-10-14-qwen4exp-gdn-norm-gate-waves
P1=benchmarks/results/2026-10-12-qwen4exp-cross-engine-pass1/raw

for PROFILE in fast production; do
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

python -m pytest tests/test_gpu_qwen4exp_gdn_norm_gate_waves.py
```
