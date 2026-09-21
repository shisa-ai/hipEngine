# Hoisting the wide Q8_0 prefill activation conversion: what it is worth

The promoted dense prefill route at layers 16-47
(`hipengine_gguf_q8_0_dense_wide256_f32_f32_out`, `docs/QWEN4EXP-STATUS.md` §4)
converts its activation from f32 to f16 inside the K loop, during LDS staging.
The conversion therefore runs once per column block, and the f32 activation is
re-read from global memory the same number of times.

This record measures the alternative: convert the activation once into a scratch
buffer, then run the same tile against an f16 activation. It answers the first
question in `docs/QWEN4EXP-STATUS.md` §5 priority 1 with a measured composite
rather than a load-path probe.

## Result

| | ms per 4096-token prefill | share of the wide route |
| --- | ---: | ---: |
| Promoted route: conversion inside the K loop | 1385 | 100% |
| Conversion pass, then f16-input tile | **794** | 57.4% |
| **Saving** | **591** | **42.6%** |

Every configuration produced **bit-identical** output between the two activation
ABIs: both put the same f16 bytes in LDS for the same values, so the change moves
work without moving arithmetic.

The saving is 3.4% of the 17.2 s `code-p4096` prefill wall
(`benchmarks/results/2026-09-17-qwen4exp-shipped-default-attribution`), on the
one role family this route owns.

## The packet is calibrated against the engine

Per-launch times from this packet and from the shipped default's own
role-marked capture (`code-p4096`, 1024-token chunks, 1056 wide launches):

| Role | Engine, ms/launch | Packet f32-in, ms/launch | Packet composite, ms/launch |
| --- | ---: | ---: | ---: |
| `attn_qkv` | 2.500 | 2.626 | 1.392 |
| `attn_q` | 2.733 | 2.832 | 1.689 |
| `attn_output` | 2.468 | 2.284 | 1.087 |
| `ssm_out` | 2.387 | 2.374 | 1.103 |
| `attn_gate` | 1.544 | 1.586 | 0.945 |
| `hc_attn_down` | 2.001 | 1.929 | 1.296 |
| `hc_ffn_down` | 1.961 | 1.911 | 1.292 |
| `shared_gate` | 0.345 | 0.242 | 0.194 |
| `shared_up` | 0.249 | 0.240 | 0.194 |
| `attn_k` | 0.319 | 0.237 | 0.197 |
| `attn_v` | 0.249 | 0.235 | 0.198 |
| `shared_down` | 0.149 | 0.159 | 0.141 |
| **total over 1056 launches** | **1404** | **1385** | **794** |

The engine total and the packet total agree to 1.4%, so the per-shape times here
stand in for the engine's own launches at these shapes. Two roles differ by more
than 10%: `attn_k`/`attn_v` (engine 35% slower) and `shared_gate` (engine 43%
slower). The `attn_k`/`attn_v` gap is the tail instantiation
(`q8_0_dense_wide_kernel<float, 128, 256, 64, 64, true>`): the engine's capture
shows 512 tail-path launches against 1600 tail-free ones, and this packet always
measures the tail-free path. Both are small in absolute terms, and the aggregate
agrees.

## What the conversion pass costs

| Shape | Cast, ms | Bytes moved | Implied bandwidth |
| --- | ---: | ---: | ---: |
| K=2560, 1024 rows | 0.034 | 15.7 MB | 462 GB/s |
| K=6144, 1024 rows | 0.171 | 37.7 MB | 220 GB/s |
| K=10240, 1024 rows | 0.276 | 62.9 MB | 228 GB/s |

The pass reads the f32 activation and writes f16, so the bytes moved are
`rows * K * 6`. The two larger shapes run at DRAM speed. The K=2560 row is above
the 256 GB/s DRAM figure because its 10.5 MB source activation is still in cache
from the previous repetition, so treat 462 GB/s as an upper bound. The largest
cast is 0.276 ms against a 1.1 ms saving on the same shape.

## Where the saving comes from

| Role | Launches | f32-in total, ms | Composite total, ms | Saving, ms |
| --- | ---: | ---: | ---: | ---: |
| `attn_qkv` | 96 | 252.1 | 128.2 | 124.0 |
| `hc_attn_down` | 128 | 246.8 | 139.8 | 107.0 |
| `hc_ffn_down` | 128 | 244.6 | 140.1 | 104.6 |
| `ssm_out` | 96 | 227.9 | 125.2 | 102.8 |
| `attn_gate` | 96 | 152.2 | 86.5 | 65.8 |
| `attn_q` | 32 | 90.6 | 53.3 | 37.3 |
| `attn_output` | 32 | 73.1 | 41.5 | 31.6 |
| `shared_gate` | 128 | 31.0 | 23.8 | 7.2 |
| `shared_up` | 128 | 30.7 | 23.9 | 6.9 |
| `shared_down` | 128 | 20.4 | 18.8 | 1.5 |
| `attn_k` | 32 | 7.6 | 6.6 | 1.0 |
| `attn_v` | 32 | 7.5 | 6.6 | 0.9 |
| **total** | **1056** | **1384.7** | **794.1** | **590.6** |

Launch counts are the shipped default's own census (24 layers for `attn_qkv`,
`attn_gate` and `ssm_out`; 32 for the `hc_*`, `shared_*` and `shared_down` roles;
8 for the full-attention roles) times the four 1024-token chunks of a 4096-token
prefill.

Two pairs share one live activation — `attn_qkv` with `attn_gate`, and
`shared_gate` with `shared_up` — so the table charges each pair half a cast. No
other sharing is assumed: no reuse across chunks, layers, or requests.

## How this was measured

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
PY=$ENV_PREFIX/bin/python
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPENGINE_HIP_ARCH=gfx1151

$PY scripts/qwen4exp_dense_wide_f16_activation.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 512,1024 --repetitions 24 --weight-buffers 8 \
  --output benchmarks/results/2026-09-17-q8-dense-f16-activation/artifact.json
```

`--weight-buffers 8` rotates eight copies of the weight tensor across
repetitions, so each launch reads weights that the previous launches evicted.
One copy stays in cache and measures a kernel the engine never runs: the same
run with `--weight-buffers 1` (`artifact-l2-resident.json`) reports 1292 ms
instead of 1385 ms for the f32 arm, 7% optimistic. The rotated run is the one
reported above because it matches the engine's own per-launch times.

Timing is the median of 24 per-repetition HIP event pairs on one stream. The
composite is two launches on one stream, so the second launch's cost is inside
the measurement. The scratch buffer is allocated once per shape and reused, and
the weights are real Q8_0 blocks read from the model at a layer where the route
runs (16 for the roles present at every layer from 16, 19 for the
every-fourth-layer full-attention roles).

## What this does not establish

- **No engine measurement.** The route is not wired into the runner. The 591 ms
  is per-launch timing summed over the census's launch counts, not a prefill
  wall. Wiring it needs a scratch buffer, one cast per distinct live activation,
  invalidation on producer writes, and the strict fallback left registered.
- **No numerical gate.** Bit-identity is checked between the two activation
  ABIs at these shapes on this host, which is stronger than the profile gate for
  this change but is not the calibrated production envelope
  (`docs/EXECUTION-PROFILES.md`).
- **The tile is unchanged.** This is `dense_wide256` (128 columns x 256 rows,
  BK 64) in both arms; no other tile was measured against the f16 ABI.
- **One host.** AMD Radeon 8060S (`gfx1151`), Framework Desktop, 125 GB unified
  memory, ROCm 7.15.0 / hipcc 7.15.26333.

## Estimate provenance

The 590.6 ms headline is the census-weighted aggregate of separately timed
medians with the conversion pass *shared* across the consumers of one
activation. The measured composite arm (conversion then tile, no sharing)
gives 553.9 ms, and the assumed sharing is worth only 7.5 ms of the difference.
Read the honest range as **553.9-590.6 ms**, and read the per-launch figure as
the one that matters for an implementation that converts per call: the
implementation consequence is that per-call conversion into reusable
runner-owned scratch captures essentially the whole opportunity, and a general
activation cache with invalidation machinery is not a prerequisite for it. This
is a kernel-and-launch-count aggregate, not a prefill wall time; the engine
measurement is the route A/B in
[`../2026-09-17-q8-dense-f16-activation-hoist/`](../2026-09-17-q8-dense-f16-activation-hoist/).
