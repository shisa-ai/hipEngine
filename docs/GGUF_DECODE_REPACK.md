# P9.H2 GGUF Decode Repack Design

Date: 2026-05-19
Scope: qwen35moe / Qwen3.6-35B-A3B-UD-Q4_K_M / gfx1100 resident decode

## Status

Implementation progress:

- P9.H3a registered the four T16 quant keys and added bit-lossless CPU
  materializers/inverses for Q5T16, Q6T16, and Q8T16; Q4T16 reuses P9.C13.
- P9.H3b added the resident materialization opt-in
  `HIPENGINE_GGUF_DECODE_REPACK=1` / `decode_repack=True`, which plans and
  copies T16 `tiles` allocations without raw expert sidecars for covered
  qwen35moe slots.
- P9.H3c added the first HIP consumer: Q8T16 single and dual dense/shared
  rows=1 GEMV decode kernels plus registry/runtime single-dispatch support.
- P9.H3d added selected-MoE Q4T16 direct/compact dual gate-up and Q5T16/Q6T16
  direct/compact down GEMV kernels, plus resident routing through the `tiles`
  allocations.
- P9.H3e added dense Q6T16 lm-head GEMV (`BF16 -> F32/BF16`) so current
  decode-repack profiles no longer rely on the legacy Q6_K `prefill_out`
  fallback.

This is the high-priority design for tasks #50/#51. It responds to the P9.B7
finding that the current rows=1 GGUF decode opt-in is wired but not useful:

- 512/128 graph decode stayed at `63.033 tok/s` vs the `>=95 tok/s` target and
  the task #16/current `62.557 tok/s` baseline.
- 512/16-minus-512/0 rocprof decode deltas showed selected `pack8_gemv` kernels
  active, but legacy raw-GGUF `prefill_out` kernels still consumed a large
  bucket (`72.960 ms` graph / `94.725 ms` eager).
- P9.E2 rejected the true WMMA+GEMV opt-in combination (`KL 5.993`, top-1
  `5.43%`). Task #49 therefore safety-disables those qwen35moe resident fast
  paths unless `HIPENGINE_GGUF_ALLOW_UNSAFE_QWEN35MOE_FASTPATHS=1` is set.

Conclusion: another launch/graph or raw-GGUF dequant-on-the-fly kernel tweak is
not the right next move. The resident path needs a decode-friendly replacement
layout, with no duplicate large raw+packed residency.

## Current inventory and memory budget

Target model metadata from
`/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`:

```text
layers=40, hidden=2048, vocab=248320
expert_count=256, experts_per_token=8, expert_ffn=512, shared_ffn=512
layer pattern: 30 linear-attention layers + 10 full-attention layers
```

Relevant resident weight bytes by slot:

| Slot | Count | Quant | Raw bytes | Raw GiB | Shape |
| --- | ---: | --- | ---: | ---: | --- |
| `ffn_gate_exps` | 40 | Q4_K | 6,039,797,760 | 5.6250 | `[256, 512, 2048]` |
| `ffn_up_exps` | 40 | Q4_K | 6,039,797,760 | 5.6250 | `[256, 512, 2048]` |
| `ffn_down_exps` | 37 | Q5_K | 6,829,342,720 | 6.3594 | `[256, 2048, 512]` |
| `ffn_down_exps` | 3 | Q6_K | 660,602,880 | 0.6152 | `[256, 2048, 512]` |
| dense/shared Q8_0 projections used by attention/linear-attn/shared FFN | 90 | Q8_0 | 1,359,790,080 | 1.2659 | mixed |

All model tensors total `20.604 GiB`; the current resident 512/128 bench peaked
at about `20.886 GiB` tracked (`21.35 GiB` HIP sampled). The deployment envelope
is 24 GiB-class, so we have roughly `2.6-3.1 GiB` headroom in the current single
request shape. Duplicating expert weights is not viable: duplicating only the
Q4 gate+up experts would add `11.25 GiB`.

## Chosen layout: replacement tile-major decode slabs

Use replacement layouts that can serve decode first and later prefill kernels,
not sidecar layouts that duplicate the raw GGUF tensors.

### Naming

Use one layout family with quant-specific plugin keys:

```text
gguf_q4_k_t16_v1
gguf_q5_k_t16_v1
gguf_q6_k_t16_v1
gguf_q8_0_t16_v1
```

`T16` means one slab stores 16 adjacent output columns for one K block, with K
metadata and quant payload arranged for coalesced rows=1 decode loads. It is not
limited to WMMA; it is the resident storage format that both decode and future
small-row/prefill kernels should consume.

### Q4_K selected gate/up: reuse Q4T16

Task #43/#44 already selected and prototyped the Q4T16 layout:

```text
tiles[expert, out_tile16, k_block, 2368]
```

Per 16-column / 256-K block slab:

| Field | Bytes | Notes |
| --- | ---: | --- |
| `d[16]` fp16 | 32 | raw `d` per output column |
| `dmin[16]` fp16 | 32 | raw `dmin` per output column |
| unpacked scales `[8,16]` uint8 | 128 | avoids per-call 12-byte scale/min unpack |
| unpacked mins `[8,16]` uint8 | 128 | same |
| q4 nibbles `[8 subblocks, 32 K lanes, 16 columns]` packed 2/byte | 2048 | contiguous by K lane then output tile |
| **total** | **2368** | raw is `16 * 144 = 2304` bytes, overhead `+2.78%` |

Full Q4 gate+up replacement cost:

```text
raw Q4 gate+up = 11.2500 GiB
Q4T16 gate+up = 11.5625 GiB
persistent overhead if raw is replaced = +0.3125 GiB
```

### Q5_K selected down: Q5T16

Use the same slab axes:

```text
tiles[expert, out_tile16, k_block, 2880]
```

Per 16-column / 256-K block slab:

| Field | Bytes | Notes |
| --- | ---: | --- |
| `d[16]` fp16 | 32 | raw `d` |
| `dmin[16]` fp16 | 32 | raw `dmin` |
| unpacked scales `[8,16]` uint8 | 128 | from the raw 12-byte scale/min field |
| unpacked mins `[8,16]` uint8 | 128 | same |
| q4 low nibbles | 2048 | Q4-style low bits |
| qh high bit | 512 | one high bit per K value |
| **total** | **2880** | raw is `16 * 176 = 2816` bytes, overhead `+2.27%` |

Full Q5 selected-down replacement cost:

```text
raw Q5 down = 6.3594 GiB
Q5T16 down ~= 6.5039 GiB
persistent overhead if raw is replaced ~= +0.1445 GiB
```

### Q6_K selected down: Q6T16

Q6_K already stores signed scale bytes directly, so a tile-major reorder can be
byte-neutral:

```text
tiles[expert, out_tile16, k_block, 3360]
```

Per 16-column / 256-K block slab:

| Field | Bytes | Notes |
| --- | ---: | --- |
| `d[16]` fp16 | 32 | raw super-block scale |
| scales `[16 groups,16 columns]` int8 | 256 | per 16-K group |
| ql low nibbles | 2048 | low four bits |
| qh high two bits | 1024 | high two bits |
| **total** | **3360** | raw is `16 * 210 = 3360`, overhead `0` |

Full Q6 selected-down replacement cost is approximately byte-neutral
(`0.6152 GiB`).

### Q8_0 dense/shared projections: Q8T16

Use the same 16-output-column slab for dense Q8_0 projections, with shape
adapted per tensor:

```text
tiles[out_tile16, k_block32, 544]
```

Per 16-column / 32-K block slab:

| Field | Bytes | Notes |
| --- | ---: | --- |
| `d[16]` fp16 | 32 | raw Q8_0 scale |
| q8 values `[32 K,16 columns]` int8 | 512 | output-tile-major |
| **total** | **544** | raw is `16 * 34 = 544`, overhead `0` |

This replaces raw Q8_0 for attention, linear-attention, and shared-expert
projections that currently hit `gguf_q8_0_prefill_out_kernel` /
`gguf_k_dual_prefill_out_kernel` in decode profiles. The Q8_0 token embedding
path may keep its existing raw/embedding layout initially because it is not a
major decode bucket.

## Persistent memory plan

If all targeted tensors are replaced rather than duplicated:

| Family | Raw GiB | Replacement GiB | Delta GiB |
| --- | ---: | ---: | ---: |
| Q4_K selected gate+up | 11.2500 | 11.5625 | +0.3125 |
| Q5_K selected down | 6.3594 | ~6.5039 | +0.1445 |
| Q6_K selected down | 0.6152 | 0.6152 | 0.0000 |
| Q8_0 dense/shared listed above | 1.2659 | 1.2659 | 0.0000 |
| Q6_K lm-head | byte-neutral | byte-neutral | 0.0000 |
| **Total targeted** | **19.4905** | **~19.9475** | **~+0.4570** |

Expected tracked peak if the rest of the runtime is unchanged:

```text
current tracked peak                 ~= 20.886 GiB
replacement-layout persistent delta  ~= +0.457 GiB
expected tracked peak                ~= 21.34 GiB
headroom to 24 GiB                   ~= 2.66 GiB
```

Do **not** implement this as a sidecar that keeps raw expert tensors resident.
Raw+packed for just Q4/Q5/Q6 experts would exceed the 24 GiB-class budget.

## Replacement policy

1. Add a resident materialization mode, not a post-load sidecar:

   ```text
   HIPENGINE_GGUF_DECODE_REPACK=1
   resident_decode_layout="t16_v1"
   ```

2. For covered qwen35moe slots, materialize only the replacement allocation into
   `Qwen35GGUFDeviceWeight`; do not allocate the raw GGUF bytes.

3. Store enough metadata to identify the source tensor and reconstruct raw bytes
   in CPU tests, but the runtime should not depend on raw reconstruction.

4. Fallback behavior:
   - If a covered tensor has no T16 kernel for the requested operation, fail
     loudly in the resident fast-path mode rather than silently allocating raw
     duplicate storage.
   - A debug-only `HIPENGINE_GGUF_DECODE_REPACK_KEEP_RAW=1` may keep raw bytes
     for one-layer kernel bring-up, but any benchmark artifact using it must be
     `blocked_memory` / not promotable.
   - The current #49 safety gate remains in place until the full P9.E2 contract
     passes with `effective_wmma_prefill=true` and/or `effective_gemv_decode=true`.

5. lm-head status: P9.B7 originally allowed Q6_K lm-head logits pack8 fallback,
   but P9.H3e now materializes `root.lm_head` as `gguf_q6_k_t16_v1` in
   decode-repack mode and routes logits through dense Q6T16 GEMV. Treat any
   remaining legacy Q6_K `prefill_out` in decode traces as a regression unless
   a future artifact documents an explicit fallback.

## Expected kernel families

Keep registry invariants: no `if backend == ...` or `if quant == ...` in engine
or model code. Add new layouts/quant variants and resolve through the existing
four-axis registry.

Initial kernels for #51:

| Layer key | Quant key | Variant | Purpose |
| --- | --- | --- | --- |
| `moe_linear` | `gguf_q4_k_t16_v1` | `selected_dual_t16_gemv_decode_compact_bf16_bf16_out` | selected gate+up for 8 routed experts/token |
| `moe_linear` | `gguf_q5_k_t16_v1` | `selected_t16_gemv_decode_compact_bf16_bf16_out` | selected down projection |
| `moe_linear` | `gguf_q6_k_t16_v1` | `selected_t16_gemv_decode_compact_bf16_bf16_out` | selected down projection for Q6 layers |
| `linear` | `gguf_q8_0_t16_v1` | `t16_gemv_decode_bf16_bf16_out` | dense/shared single projection |
| `linear` | `gguf_q8_0_t16_v1` | `t16_dual_gate_up_gemv_decode_bf16_bf16_out` | shared-expert gate+up dual projection |
| `linear` | `gguf_q6_k_t16_v1` | `t16_gemv_decode_bf16_f32_out` | lm-head logits |

Kernel shape guidance:

- One wave/block should compute one output tile (`16` columns) for one routed
  expert/lane or dense row; use 64/128 threads only if the K loop benefits after
  profiling.
- Load quant metadata once per output tile/K block, not once per scalar output.
- Keep accumulation in FP32 and output BF16 to match the current resident path.
- Preserve graph-capture safety: no host copies, no shape-dependent allocations,
  and all scratch zeroing uses device memset/kernels.
- Start with deterministic K order matching the CPU/reference kernels; if a
  faster order changes results, it must still pass P9.E2 before promotion.

## Integration sequence

1. **Materializer/oracle tests**
   - Q4T16 already exists; extend the pattern to Q5T16/Q6T16/Q8T16.
   - Add exact inverse tests for Q5/Q6/Q8 raw reconstruction.
   - Add CPU reference GEMV routines that consume the T16 layouts directly and
     match existing GGUF dequant outputs.

2. **Standalone HIP kernel tests**
   - Tiny synthetic fixtures for Q4 selected dual, Q5/Q6 selected, Q8 single,
     and Q8 dual.
   - Compare against `kernels/cpu_reference` with KL <= 0.05 and top-1 >= 90%,
     plus stricter numeric max-abs where current tests already use it.

3. **Runtime materialization**
   - Add T16 layout constants to `qwen35_gguf_materialize.py`.
   - Add device weight specs for `gguf_q*_t16_v1`.
   - Add dispatch tests proving rows=1 decode selects T16 variants when the
     resident decode layout is enabled.

4. **P9.E2 safety release**
   - Enable qwen35moe fast paths only when the T16 coverage is complete for the
     target shape.
   - Re-run `scripts/qwen35_gguf_p9_e2e_correctness.py` full 512/128x3.
   - Promotion requires `passed=true` and artifact `fastpath_safety` showing the
     relevant `effective_*` flags are true, not a safety fallback.

5. **P9.B7 retention**
   - 512/128 graph replay decode median over 3 runs.
   - Target: `>=95 tok/s` vs current `63.033 tok/s` and task #16/current
     `62.557 tok/s` baseline.
   - `rocprofv3 --kernel-trace` (full if possible, otherwise 512/16 decode-delta
     plus explicit explanation) must show T16 decode kernels dominating and
     legacy `prefill_out` absent from decode (including lm-head unless a future
     artifact explicitly reinstates that fallback).
   - Peak tracked memory must remain below the 24 GiB-class budget; record both
     tracked and HIP sampled peaks.

## Acceptance criteria for #51/#52

- P9.E2 512/128x3 passes with effective fast paths enabled, not safety-disabled.
- 512/128 graph replay decode median is `>=95 tok/s` over 3 measured runs.
- Correctness artifact, wall-clock artifact, and rocprof evidence are retained.
- Persistent tracked peak is `<=22 GiB` for the 512/128 shape and must not rely
  on raw+packed duplication. If a W7900 run differs, report exact hardware.
- The benchmark rollup marks the row accepted only if both correctness and
  performance pass; otherwise retain a blocked artifact with the exact failing
  kernel/memory bucket.

## Non-goals / rejected shortcuts

- Do not re-enable the current raw-GGUF WMMA/GEMV opt-ins globally; #49 disables
  them because they failed P9.E2.
- Do not revive the old full expert sidecar as a default. It duplicates too much
  expert storage for the 24 GiB-class budget.
- Do not chase another launch-bound sweep of raw `prefill_out`/`pack8_gemv`
  kernels before the layout replacement exists; P9.B7 showed the kernel mix, not
  graph overhead, is the limiter.
- Do not add model/quant conditionals to dispatch. New layouts must enter via
  registered quant/layout plugins and explicit resident materialization mode.
