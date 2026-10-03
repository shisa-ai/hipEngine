# gfx1151 shared-expert gate+up: wave-strided Q8T16 loop

Date 2026-10-03. Model `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, quant `gguf_ud_q4_k_m`,
host gfx1151 Strix Halo Radeon 8060S, shipping AR route.

At `rows == 1` the Q8T16 shared gate+up pair is routed through
`q8_0_t16_dual_split_gemv_kernel` (`WAVE_BLOCK_INDEX = true`) instead of
`q8_0_t16_dual_gemv_kernel`. Both enumerate the identical `(lane, block)` set in
the identical order, so the change is bit-identical; only the address arithmetic
differs.

## Why it was worth doing

The shared-expert gate+up stream was the worst-bandwidth stage in the decode step
that was not already explained: 87.1 MiB per token at 103 GB/s against 199 GB/s
for the main dense GEMVs in the same step. The pair launches as grid `(64, 1)`
with 128 threads — `(out_a + out_b) / T16_COLS` — which is under one block per
compute unit on a 40-CU part, and its loop recomputes `k / 32` and the tile
address every iteration.

## Isolated kernel A/B

Cold stream (pools larger than L2/MALL), median of repeats, target tree
(`isolated-kernel-ab.json`):

| variant | gate+up (grid 64, 128 thr) |
| --- | ---: |
| dual concat (previous production) | 18.9 us, 117.9 GB/s |
| **dual_split (new)** | 15.6 us, **142.5 GB/s** |
| two singles (fallback) | 30.0 us, 74.4 GB/s |

**+20.8%.** `np.array_equal` on the uint16 outputs is True.

| variant | down (grid 128, 128 thr) |
| --- | ---: |
| single (previous production) | 10.2 us, 109.1 GB/s |
| dual_split 2x1024 | 9.8 us, 114.3 GB/s |

The `down` variant was measured and deliberately not taken: its isolated gain is
small and the real stage showed no delta, so routing it would be a
shape-specific hack rather than a general improvement.

Ceiling references on the same host: a single 8192x2048 Q8_0 GEMV reaches
181.7 GB/s at 512 blocks (the shared-expert shapes are fixed at 64 and 128
blocks), and a 1 GiB device-to-device copy reaches 210.2 GB/s.

## Real decode step

Same-session A/B with the GPU stage recorder, one persistent session, warmup 1,
measured 1, prompt 512, 32 decode tokens. The baseline arm redirects only the
`rows == 1` pair-concat call back to the concat kernel by monkeypatch, so both
arms run the same tree and the same session shape (`real-step-ab.json`):

| stage | baseline us/32tok | patched | delta |
| --- | ---: | ---: | ---: |
| **decode_ffn_moe_shared_gate_up** | **30182.3** | **26006.3** | **-13.84%** |
| decode_ffn_moe_shared_down | 15897.9 | 15631.0 | -1.68% |
| decode_ffn_moe_selected | 125426.6 | 125368.2 | -0.05% |
| decode_ffn_moe_router | 24464.8 | 24499.3 | +0.14% |
| decode_ffn_moe_combine | 14968.2 | 14966.7 | -0.01% |

Decode **53.1877 -> 53.5820 tok/s (+0.741%)**. The stage saving is 130.5
us/token against an 18.8 ms step, a 0.69% prediction, so the transfer to the
step is complete. Two earlier independent sessions read 942.1 -> 812.9 and
941.2 -> 812.8 us/token (-13.7% and -13.6%).

## Bit-identity

- concat: `for (k = tid; k < in_features; k += blockDim.x)`, then
  `block_idx = k / 32`, `lane = k & 31`.
- split: `lane = tid & 31`, `wave = tid >> 5`,
  `for (block_idx = wave; block_idx < blocks_per_row; block_idx += waves)`, then
  `k = block_idx * 32 + lane`.

With 128 threads there are 4 warps, so the concat enumeration `k = tid + 128j`
maps to `block_idx = tid/32 + 4j`, `lane = tid & 31` — the split enumeration.
Per-thread FP32 accumulation order and the reduction tree (one
`GGUF_T16_WAVE_REDUCE` plus a 4-sub-group sum) are therefore identical, and a
warp reads one Q8_0 block per iteration in both.

Register pressure is also identical: VGPR 56, SGPR 128, LDS 512, 128 threads for
both kernels, from the saved kernel trace. There is no occupancy mechanism by
which the new kernel could regress.

## Tests

- 268 passed: `test_unit_gguf_linear_dispatch.py`,
  `test_unit_gguf_gemv_decode_dispatch.py`,
  `test_unit_qwen35_gguf_compact_moe_gemv_routing.py`,
  `test_unit_qwen35_gguf_iq_dispatch.py`,
  `test_unit_qwen35_gguf_compact_moe_wmma_routing.py`.
- 38 passed: `tests/test_gpu_gguf_q8_0_t16_gemv_decode.py`, CPU-oracle-gated.

No `docs/EXECUTION-PROFILES.md` numerical gate is required: this is a
launch-routing change with no arithmetic change.

## Limits

One host, one model, one quant, one profile. The change is bit-identical with
identical register pressure and access pattern, so it is not gfx1151-specific by
mechanism, and gfx1151 compiles the gfx1100 kernel bodies so both backends share
the path — but the performance magnitude is measured on gfx1151 only and is
**unverified on gfx1100/W7900**. The isolated rates are synthetic cold-stream
measurements; the -13.84% stage delta and the +0.741% decode delta are the
load-bearing numbers.

The remaining gap from 142.5 to the ~181 GB/s ceiling needs more than 64 blocks,
which is not bit-identically reachable at this shape: a K-split reassociates the
FP32 sums, `col8` reads strided (108 GB/s, slower), and a row-tile wastes rows at
rows=1. A wider-block split variant would change the reduction tree from 4 to 8
sub-groups and needs the production numerical gate.

## Reproduce

```bash
source /tmp/reprof/env.sh && export PYTHONPATH=/home/lhl/hipEngine
$PY /tmp/reprof/shexp_ab.py

$PY /tmp/reprof/run_shexp_ab.py baseline \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --quant gguf_ud_q4_k_m \
  --prompt-length 512 --decode-tokens 32 --warmup-runs 1 --measured-runs 1 \
  --persistent-session --gpu-stage-timings --public-ar-profile
$PY /tmp/reprof/run_shexp_ab.py patched <same args>
```

Note: the `hipengine` editable install points at a different tree
(`/home/lhl/hipEngine-main`), so out-of-repo probe scripts must set
`PYTHONPATH=/home/lhl/hipEngine` or they silently measure the wrong commit.
