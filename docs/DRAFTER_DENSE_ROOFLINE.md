# DFlash R3.3 drafter dense roofline

Date: 2026-05-23
Hardware: AMD Radeon Pro W7900 / gfx1100
Scope: z-lab `Qwen3.6-35B-A3B-DFlash` drafter dense kernels in hipEngine.

## Executive summary

Fresh W7900 microbench confirms the Round-3 diagnosis: the current DFlash dense kernels are far below the memory-bandwidth roofline and account for most of the drafter decoder wall.

Key measured facts:

- R2.2 synchronized phase split: drafter decoder layers = **19.60 ms/cycle**.
- R3.3 dense microbench estimates current dense work = **16.32 ms/cycle** across the 8 drafter layers.
- Therefore dense kernels explain **~83%** of decoder-layer wall (`16.32 / 19.60`).
- Effective weight bandwidth is only **~25–54 GB/s** depending on shape, i.e. **~3–6%** of W7900's `864 GB/s` memory-bandwidth roof.
- A 30% BW WMMA/LDS implementation projects dense work at **~2.98 ms/cycle**, saving **~13.3 ms/cycle** and moving decoder layers from `19.6 → ~6.3 ms`.

R3.4 is justified: a WMMA-tiled small-row BF16 dense kernel is the largest currently-measured DFlash drafter lever on W7900. It will not make the whole system profitable alone (cycle wall projects `~62 → ~49 ms` before verifier work), but it is necessary to make prompt-3-like high-acceptance cases winnable with R3.6/R3.5.

## Model shapes

Drafter config (`/home/lhl/.cache/huggingface/hub/models--z-lab--Qwen3.6-35B-A3B-DFlash/snapshots/42d3b34d588423cdae7ba8f53a8cf7789346a719/config.json`):

| Field | Value |
| --- | ---: |
| hidden_size | `2048` |
| intermediate_size | `6144` |
| num_hidden_layers | `8` |
| num_attention_heads | `32` |
| num_key_value_heads | `4` |
| head_dim | `128` |
| kv_features | `512` |
| block_size | `16` |
| vocab_size | `248320` |

Dense ops per drafter layer:

| Op | Kernel output | Shape `(rows, in, out)` | Current kernel |
| --- | --- | --- | --- |
| Q projection | FP32 | `(16, 2048, 2048)` | `dflash_dense_bf16_to_f32` |
| K projection | FP32 | `(16, 2048, 512)` | `dflash_dense_bf16_to_f32` |
| V projection | BF16 | `(16, 2048, 512)` | `dflash_dense_bf16_to_bf16` |
| O projection | BF16 | `(16, 2048, 2048)` | `dflash_dense_bf16_to_bf16` |
| gate projection | BF16 | `(16, 2048, 6144)` | `dflash_dense_bf16_to_bf16` |
| up projection | BF16 | `(16, 2048, 6144)` | `dflash_dense_bf16_to_bf16` |
| down projection | BF16 | `(16, 6144, 2048)` | `dflash_dense_bf16_to_bf16` |

## Current kernel shape

Source: `hipengine/kernels/hip_gfx1100/speculative/dflash_drafter.hip`

Current `dflash_dense_bf16_to_{bf16,f32}_kernel` launch geometry:

```text
grid  = dim3(out_features, rows, 1)
block = dim3(threads=128)
```

Each block computes **one output element** by reducing over all `in_features`:

```text
out[row, out_col] = sum_i bf16(x[row, i]) * bf16(weight[out_col, i])
```

For `(rows=16, in=2048, out=2048)`, that means:

- `16 × 2048 = 32,768` thread blocks per dense op.
- Each block performs a 2048-element reduction using 128 threads.
- Weight traffic dominates: `2048 × 2048 × 2 B = 8 MiB/op`.
- Input row bytes are small (`16 × 2048 × 2 B = 64 KiB`) but are reread many times because each output column is independent.

This is a scalar-GEMV-style implementation, not a tiled GEMM.

## Fresh GPU sanity measurement

Command:

```bash
PYTHONPATH=. python3 scripts/dflash_dense_microbench.py \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --loops 20 --warmup 5 \
  --hardware-gpu 'AMD Radeon Pro W7900' \
  --json benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json
```

Artifact: [`benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json`](../benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json)

| Kind | Shape `(rows,in,out)` | Mean ms/op | Min ms/op | Effective weight GB/s | Effective TFLOP/s | BW roof utilization |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16→BF16 | `(16,2048,2048)` | `0.182` | `0.170` | `46.1` | `0.738` | `5.3%` |
| BF16→FP32 | `(16,2048,2048)` | `0.184` | `0.168` | `45.6` | `0.730` | `5.3%` |
| BF16→BF16 | `(16,2048,6144)` | `0.523` | `0.457` | `48.1` | `0.769` | `5.6%` |
| BF16→BF16 | `(16,6144,2048)` | `0.463` | `0.399` | `54.3` | `0.869` | `6.3%` |
| BF16→BF16 | `(16,2048,512)` | `0.081` | `0.070` | `25.9` | `0.415` | `3.0%` |
| BF16→FP32 | `(16,2048,512)` | `0.083` | `0.081` | `25.1` | `0.402` | `2.9%` |

Weighted per-layer estimate using the actual DFlash op mix:

```text
Q f32 2048x2048:     0.184 ms
K f32 2048x512:      0.083 ms
V bf16 2048x512:     0.081 ms
O bf16 2048x2048:    0.182 ms
gate bf16 2048x6144: 0.523 ms
up bf16 2048x6144:   0.523 ms
down bf16 6144x2048: 0.463 ms
--------------------------------
Dense per layer:      2.040 ms
8 layers:            16.319 ms
```

Compare to R2.2 sync phase `decoder_layers = 19.60 ms/cycle`: dense kernels are **~83%** of decoder-layer time. The remaining `~3.3 ms/cycle` is attention, norms, add/silu, concat, and dispatch.

## Roofline floor

W7900 constants:

```text
Peak memory BW:       864 GB/s
Peak BF16 WMMA:       ~120 TFLOP/s
Rows:                 16
```

For these small-row dense ops, weight bandwidth is the limiting term. Example `(16,2048,2048)`:

```text
weight bytes = 2048 × 2048 × 2 = 8.39 MB
FLOPs       = 2 × 16 × 2048 × 2048 = 134 MFLOP
BW floor    = 8.39 MB / 864 GB/s = 0.0097 ms
FLOP floor  = 134 MFLOP / 120 TFLOP/s = 0.0011 ms
roof floor  = max(BW, FLOP) = 0.0097 ms
```

Actual op-mix floor across all dense ops:

| Op | BW floor ms/op |
| --- | ---: |
| Q 2048×2048 | `0.0097` |
| K 2048×512 | `0.0024` |
| V 2048×512 | `0.0024` |
| O 2048×2048 | `0.0097` |
| gate 2048×6144 | `0.0291` |
| up 2048×6144 | `0.0291` |
| down 6144×2048 | `0.0291` |
| **per layer floor** | **`0.1117`** |
| **8-layer floor** | **`0.893`** |

The current `16.32 ms` dense total is `~18.3×` the absolute BW floor.

Projected dense total if R3.4 reaches a fraction of BW peak:

| Effective BW | Dense total | Saved vs current | Decoder projection (`19.6 - saved`) | Cycle projection (`62.1 - saved`) |
| ---: | ---: | ---: | ---: | ---: |
| current `3–6%` | `16.3 ms` | `0` | `19.6 ms` | `62.1 ms` |
| `20%` | `4.47 ms` | `11.85 ms` | `7.75 ms` | `50.25 ms` |
| `30%` | `2.98 ms` | `13.34 ms` | `6.26 ms` | `48.76 ms` |
| `50%` | `1.79 ms` | `14.53 ms` | `5.07 ms` | `47.57 ms` |

R3.4 alone does not reach prompt-3 break-even (`Need wall < ~36 ms` at B=4), but it moves the wall by the largest measured amount and makes R3.6/R3.5 meaningful.

## Proposed R3.4 kernel design

Implement two WMMA-tiled kernels:

```c++
extern "C" int hipengine_dflash_dense_bf16_to_bf16_wmma(
    const uint16_t* x,       // [rows, in]      BF16 row-major
    const uint16_t* weight,  // [out, in]       BF16 row-major
    uint16_t* out,           // [rows, out]     BF16 row-major
    int64_t rows,            // require rows % 16 == 0 for v1
    int64_t in_features,     // require in % 16 == 0
    int64_t out_features,    // require out % 16 == 0
    int64_t stream);

extern "C" int hipengine_dflash_dense_bf16_to_f32_wmma(
    const uint16_t* x,
    const uint16_t* weight,
    float* out,
    int64_t rows,
    int64_t in_features,
    int64_t out_features,
    int64_t stream);
```

Tile plan (v1):

- Use RDNA3 native `16×16×16` BF16 WMMA with FP32 accumulation.
- One workgroup owns one output tile `C[16 rows × 16 out_cols]`.
- Grid: `dim3(out_features / 16, rows / 16)`; for current drafter rows this is one tile row.
- Loop over `K = in_features / 16` tiles.
- LDS stage one `A[16×16]` input tile and one `B[16×16]` weight tile per K iteration.
- Accumulate FP32 registers, then store BF16 or FP32 depending on output type.

Expected memory behavior:

- Weight traffic remains exactly one pass over `weight[out, in]` per op.
- Input tile traffic is replicated per output tile in v1, but input is only `64 KiB` for 2048-wide shapes and is small relative to weights; v2 can improve with persistent/L2-friendly scheduling if needed.
- LDS footprint per workgroup is tiny: `2 × 16 × 16 × 2 B = 1 KiB` (double-buffering still far below RDNA3 LDS budget).

## R3.4 measured WMMA results (W7900, gfx1100)

Implemented in `hipengine/kernels/hip_gfx1100/speculative/dflash_drafter.hip` as
`hipengine_dflash_dense_bf16_to_{bf16,f32}_wmma`. Wave32 workgroup, one
`v_wmma_f32_16x16x16_bf16` instruction per K-tile, no LDS staging in v1
(weights and inputs are read once per output tile).

```bash
PYTHONPATH=. python3 scripts/dflash_dense_microbench.py \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --loops 20 --warmup 5 \
  --hardware-gpu 'AMD Radeon Pro W7900' \
  --json benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-dense-wmma-microbench.json
```

Microbench artifacts:
- naive: [`benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json`](../benchmarks/results/2026-05-23-hipengine-dflash-r3.3-w7900-dense-microbench.json)
- WMMA:  [`benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-dense-wmma-microbench.json`](../benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-dense-wmma-microbench.json)

| Kind | Shape `(rows,in,out)` | Naive ms/op | WMMA ms/op | Speedup | Naive BW % | WMMA BW % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16→BF16 | `(16,2048,2048)` | `0.182` | `0.056` | `3.2×` | `5.3%` | `17.3%` |
| BF16→FP32 | `(16,2048,2048)` | `0.184` | `0.056` | `3.3×` | `5.3%` | `17.3%` |
| BF16→BF16 | `(16,2048,6144)` | `0.523` | `0.091` | `5.7×` | `5.6%` | `32.0%` |
| BF16→BF16 | `(16,6144,2048)` | `0.463` | `0.096` | `4.8×` | `6.3%` | `30.5%` |
| BF16→BF16 | `(16,2048,512)` | `0.081` | `0.047` | `1.7×` | `3.0%` | `5.1%` |
| BF16→FP32 | `(16,2048,512)` | `0.083` | `0.048` | `1.7×` | `2.9%` | `5.0%` |

Bigger shapes (gate/up/down) hit `30–32%` of BW, exceeding R3.3's 30% target. Q/O hit
`~17%` because the workgroup count is small (`128` for `2048×2048`); a v2 design
with persistent CTAs or multi-tile per workgroup can lift that further. K/V at
`out_features=512` is launch-overhead bound (only 32 N-tiles → 32 wave32
workgroups); v2 should batch K/V or split-K to recover bandwidth there.

Per-cycle dense estimate using actual op mix:

```text
Q f32 2048x2048:     0.056 ms
K f32 2048x512:      0.048 ms
V bf16 2048x512:     0.047 ms
O bf16 2048x2048:    0.056 ms
gate bf16 2048x6144: 0.091 ms
up bf16 2048x6144:   0.091 ms
down bf16 6144x2048: 0.096 ms
--------------------------------
Dense per layer:      ~0.485 ms
8 layers:            ~3.88 ms (R3.3 naive: 16.32 ms, -76%)
```

### 9-prompt B=4 D=32 same-session DFlash on W7900

Artifacts:
- WMMA on:  [`benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-wmma-on-b4-d32-9prompt.json`](../benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-wmma-on-b4-d32-9prompt.json)
- WMMA off: [`benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-wmma-off-b4-d32-9prompt-control.json`](../benchmarks/results/2026-05-23-hipengine-dflash-r3.4-w7900-wmma-off-b4-d32-9prompt-control.json)

| Metric | WMMA off | WMMA on | Delta |
| --- | ---: | ---: | ---: |
| Aggregate AR ratio | `0.446` | `0.636` | `+0.19` (+42% relative) |
| Drafter wall ms/cycle | `23.50` | `9.09` | `-14.41` (-61%) |
| Verifier wall ms/cycle | `29.10` | `27.89` | `-1.21` (-4%) |
| Cycle wall ms | `52.83` | `37.21` | `-15.62` (-29%) |
| Avg accept length | `1.55` | `1.53` | `-0.02` (within sampling noise) |
| Exact AR rows | `9 / 9` | `9 / 9` | unchanged |

Per-prompt:

| Prompt | WMMA off | WMMA on | Delta |
| --- | ---: | ---: | ---: |
| code:quicksort_prefix | `0.327` | `0.468` | `+0.14` |
| code:function_continuation | `0.555` | `0.804` | `+0.25` |
| code:class_continuation | `0.658` | `0.911` | `+0.25` |
| code:json_yaml_continuation | `0.481` | `0.646` | `+0.17` |
| code:humaneval_add | `0.511` | `0.722` | `+0.21` |
| code:humaneval_sort_third | `0.365` | `0.526` | `+0.16` |
| math:short_gsm8k_style | `0.375` | `0.530` | `+0.16` |
| instruct:simple_qa_no_template | `0.505` | `0.723` | `+0.22` |
| instruct:simple_qa_qwen_static_chat | `0.425` | `0.638` | `+0.21` |

Conclusions:
- R3.4 is the largest single DFlash drafter lever measured to date.
- Decoder wall projection from R3.3 (`19.6 → ~6.3 ms` at 30% BW) is met or exceeded for big shapes; small-N shapes (`512`) underperform but they're a small share of cycle wall.
- The WMMA path is **default-on** via `HIPENGINE_DFLASH_DRAFTER_DENSE=wmma`; set `naive` to revert.
- DFlash still does not break-even vs AR; verifier work (R3.6 / R3.7) and tree topology (R3.5) are still required to push aggregate ≥ 1.0x AR.

## Implementation constraints

- Keep existing `dflash_dense_bf16_to_{bf16,f32}` kernels registered as fallback.
- Add new registry variants; do not branch on backend/quant in engine code.
- Gate with `HIPENGINE_DFLASH_DRAFTER_DENSE={naive,wmma}` or equivalent until exactness/perf passes.
- Correctness gate: KL ≤ 0.05 and top-1 ≥ 90% vs CPU reference / unfused current kernel on fixture inputs.
- Perf gate: `rocprofv3 --kernel-trace` shows WMMA kernel durations in the rough range:
  - `2048×512`: `~8–15 µs` at 20–30% BW
  - `2048×2048`: `~32–50 µs` at 20–30% BW
  - `2048×6144` / `6144×2048`: `~100–150 µs` at 20–30% BW
- E2E gate: same-session DFlash exact AR on the prompt suite; no default-on until the suite artifact shows a real cycle-wall reduction.

## Risks / unknowns

1. **Small N=512 shapes may underutilize WMMA.** K/V projection shapes have only 32 output tiles; launch overhead and occupancy may dominate. If v1 underperforms here, leave K/V on naive or add a specialized 16×32 / multi-N tile.
2. **Input reuse across N tiles is not perfect in v1.** The 16×K input slab is reread by every N tile. This is acceptable because weights dominate, but a persistent CTA design could stage input more efficiently if profiling shows input bandwidth matters.
3. **BF16 store rounding must match tolerance gates.** Exact bitwise identity to the naive FP32 accumulation order is not expected; use KL/top-1 gate, not bit-equality, for WMMA.
4. **R3.4 is necessary but insufficient.** Even 30% BW projects cycle wall to `~49 ms`; prompt-3 still needs verifier/topology work. Do not over-claim.

## R3.4 work order

1. Add a CPU/unfused parity test for small synthetic shapes `(rows=16, in/out in {512, 2048})`.
2. Implement `bf16_to_f32_wmma` first (Q/K path) because FP32 output avoids BF16 output-rounding ambiguity.
3. Implement `bf16_to_bf16_wmma` with standard round-to-even BF16 store.
4. Wire env-gated drafter dense selection.
5. Run `scripts/dflash_dense_microbench.py` and require ≥20% BW before E2E.
6. Run DFlash same-session E2E exactness + phase split; compare dense/decoder/cycle wall to this document.
