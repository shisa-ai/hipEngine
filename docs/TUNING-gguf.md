# GGUF Tuning Plan

Date: 2026-06-16
Branch/worktree: `gguf-tuning` / `/home/lhl/hipEngine-gguf-tuning`
Scope: Qwen3.6-35B-A3B GGUF on GPU1/gfx1100 (`AMD Radeon RX 7900 XTX`, 24 GiB-class) as the active eval/testbed, with W7900 rows kept as comparison references and the 0.8B GGUF fixtures kept as fast correctness sentinels.

## Thesis

Our GGUF hot path is HIP/C++ and resident-weight based. Once a model is loaded,
there should be no structural reason for hipEngine GGUF to trail either local
`~/llama.cpp/` or the native PARO path on the same gfx1100 GPU. The gap should be
closed the same way the recent MTP/DFlash work moved from a slow but correct
path to retained wins: lock a same-suite baseline, profile the exact phase that
is slow, keep every exact non-regressive micro-win, and reject attractive
launch-reduction ideas when the profile says they move work to a slower bucket.

This file is the active GGUF-specific tuning playbook and punchlist. The running `pi-multiloop` lane is `gguf-tuning/run-20260615-103446`; keep iteration notes here and detailed evidence in `WORKLOG.md`.

Active gates:

- **Primary acceptance gates:** GPU1 `512/128` and `4K/128`, measuring both prefill and decode.
- **Promotion check:** run `128K/128` before claiming a default-path GGUF win; if GPU1 cannot fit, record the blocker and rerun on the W7900 only as an explicitly labeled fallback.
- **Correctness/memory:** stable generated IDs/logits on the gates, targeted GGUF tests green, and no raw+packed duplicate residency or unexplained peak-memory growth.

It complements:

- [`GGUF.md`](GGUF.md) — loader/runtime status and GGUF format notes.
- [`GGUF_DECODE_REPACK.md`](GGUF_DECODE_REPACK.md) — T16 decode-repack layout.
- [`KERNELS.md`](KERNELS.md) — kernel catalog and port/playbook requirements.
- [`ROOFLINE.md`](ROOFLINE.md) — RDNA3/W7900 roofline and do-not-chase rules.
- [`BENCHMARK.md`](BENCHMARK.md) — promotion/evidence contract.
- [`MTP.md`](MTP.md), [`DFLASH.md`](DFLASH.md), and [`MEGAKERNEL.md`](MEGAKERNEL.md) — the recent tuning pattern to copy.

## Compact goal

Close the GGUF gap without weakening the architecture:

1. **Beat local `~/llama.cpp/` matched GGUF rows** on the same model/quant,
   backend/device, prompt token IDs, KV dtype, context, and run environment.
   Re-measure current `~/llama.cpp/` first; the README rows are comparison
   anchors, not a substitute for a fresh matched run.
2. **Reach PARO-class decode for c=1 short/mid shapes** where quant/model
   differences do not make the comparison meaningless. Primary acceptance is
   GPU1 `512/128` and `4K/128`; later promotion must also survive `128K/128`.
   Use the parent PARO c=1 W7900 rows at `512/128`, `4K/128`, and `32K/128` as
   target anchors, but refresh same-host PARO rows whenever they fit the active
   device.
3. **Keep the GGUF value proposition:** no torch hot path, no llama.cpp FFI shim
   on the hot path, no backend/quant branches in model/dispatch code, and no
   raw+packed duplicate residency in promoted paths.
4. **Restore 24 GiB-class viability where possible.** The active GPU1 Q4_K_S
   gate baseline fits `512/128` and `4K/128` at `21.335 GiB` tracked peak
   (`21.954 GiB` sampled HIP used), but the stale W7900 full-sweep diagnostic
   peaked at `25.108 GiB`. Promoted consumer-card rows need a specific memory
   plan or must be labeled W7900-only.

## Current scorecard to explain, not yet the final baseline

The table below stitches together the latest documented rows. Treat it as the
first hypothesis map; the first tuning sprint must rerun the relevant rows from
this branch/worktree under one clean TheRock ROCm 7.13 environment before making
new claims.

| Workload | hipEngine GGUF Q4_K_S, W7900 TheRock 7.13 | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M | PARO parent c=1 target |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `2262.097` prefill / `109.347` decode / `25.108 GiB` | `2436.049` / `85.487` / `21.125 GiB` | `1816.927` / `127.515` / `20.844 GiB` | `2696.4` / `116.05` / `18.80 GiB` |
| 4K/128 | `2544.475` / `99.873` / `25.108 GiB` | `2176.905` / `87.375` / `21.197 GiB` | `1705.093` / `120.163` / `20.969 GiB` | `2741.5` / `113.05` / `21.64 GiB` |
| 32K/128 | `1878.052` / `86.486` / `25.108 GiB` | `1496.409` / `76.994` / `21.738 GiB` | `1128.554` / `98.073` / `21.533 GiB` | `1880` / `98.8` / `21.37 GiB` |
| 128K/128 | `995.295` / `58.066` / `25.108 GiB` | `710.213` / `57.341` / `23.605 GiB` | `480.539` / `64.478` / `23.596 GiB` | `914` / `62.6` / `27.42 GiB` |

Sources: `benchmarks/README.md` 2026-06-14/15 rows and the source-lineage
PARO table. Caveats:

- The current GGUF row is Q4_K_S, while the stale llama.cpp comparison rows are
  Q4_K_M. Rerun both Q4_K_S and Q4_K_M when possible.
- The GGUF README rows are retained diagnostics and explicitly carry
  `performance_claim=false`; promotion requires the full `docs/BENCHMARK.md`
  evidence and correctness gate.
- The user's current observation that GGUF is slower than local `~/llama.cpp/`
  takes precedence over stale table interpretation. Refresh the local
  `~/llama.cpp/` rows before deciding whether the primary gap is HIP, Vulkan, or
  PARO.

Initial read: GGUF decode is already ahead of older llama.cpp HIP rows but behind
llama.cpp Vulkan and PARO on most c=1 decode shapes, and it uses more memory.
That points to kernel shape/layout, dispatch mix, and residency policy rather
than file parsing or Python host code.

## Active GPU1 gate baseline

Established on 2026-06-15 from `/home/lhl/hipEngine-gguf-tuning` with
`HIP_VISIBLE_DEVICES=1`, TheRock HIP `7.13.26162-1140233ffe`, cached HIP builds,
`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`, `HIPENGINE_GGUF_DECODE_REPACK=1`, bulk prefill,
WMMA prefill, and GEMV decode. Raw JSON lives at
`/tmp/hipengine-gguf-tuning-gpu1-acceptance.json`; commit only compact retained
artifacts after a tuning candidate is accepted.

| Workload | Prefill tok/s median | Decode tok/s median | Tracked peak | Sampled HIP used peak | Correctness sanity |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/128 | `1647.390` | `127.012` | `21.335 GiB` | `21.952 GiB` | stable final token `220`, finite logits |
| 4K/128 | `1855.806` | `115.805` | `21.335 GiB` | `21.954 GiB` | stable final token `570`, finite logits |

Configured targeted GGUF guard tests also passed (`154 passed`). The primary
multiloop metric is the minimum gate decode rate, currently `115.805 tok/s` after
G-D3 relaxed the selected single-output T16 GEMV launch-bound default from `4` to
`2` on top of the retained selected dual+SiLU and selected-WMMA launch-bound
changes.

## Active GPU1 profile findings

G-M2 paired `rocprofv3` captures were taken initially on 2026-06-15 with
`decode_tokens=16` and refreshed for the retained-default 4K path on 2026-06-16.
Prefill-only traces are used to strip the prefill prefix from decode traces. Raw
CSV/summary files live under `/tmp/hipengine-gguf-tuning/20260615-gpu1-q4ks-*`
and `/tmp/hipengine-gguf-tuning/20260616-gpu1-q4ks-retained-4k-d16-nowarmup`; the
compact retained-refresh artifact is
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-retained-4k-rocprof-diagnostic.json`.

| Shape | Phase | Total kernel time | Top buckets |
| --- | --- | ---: | --- |
| 512/16 | prefill | `289.487 ms` | selected dual Q4_K WMMA `111.306 ms` (`38.45%`); dense Q8_0 WMMA `54.196 ms` (`18.72%`); GDN prefill recurrent `41.706 ms` (`14.41%`) |
| 512/16 | decode | `131.242 ms` (`8.203 ms/token`) | dense Q8_0 T16 GEMV `50.137 ms` (`38.20%`); selected dual Q4_K T16 GEMV `17.468 ms` (`13.31%`); lm-head Q6 T16 `10.158 ms` (`7.74%`) |
| 4K/16 | prefill | `2094.198 ms` | selected dual Q4_K WMMA `800.455 ms` (`38.22%`); GDN prefill recurrent `351.231 ms` (`16.77%`); dense Q8_0 WMMA `329.473 ms` (`15.73%`); full-attn prefill `87.719 ms` (`4.19%`) |
| 4K/16 | decode | `138.513 ms` (`8.657 ms/token`) | dense Q8_0 T16 GEMV `49.758 ms` (`35.92%`); selected dual Q4_K T16 GEMV `17.154 ms` (`12.38%`); full-attn decode `13.912 ms` (`10.04%`); lm-head Q6 T16 `10.087 ms` (`7.28%`) |

G-M3 resource census on the refreshed 4K/16 trace is recorded in
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-resource-census-diagnostic.json`.
`qwen35_gguf_rocprof_summary.py` now emits VGPR/SGPR/scratch/LDS/workgroup/grid
summaries per bucket and per kernel.

| Phase | Bucket | Scratch | VGPR | LDS | Read |
| --- | --- | ---: | ---: | ---: | --- |
| prefill | selected dual Q4_K WMMA | `676 B` | `256` | `0` | dominant prefill bucket is register-pressure/spill-limited; inspect ISA/codegen before more launch-bound pokes. |
| prefill | dense Q8_0 WMMA | `0/8 B` | `96/128/192` | `0` | mostly scratch-free, with one tiny-spill shape; do tile-specific census before changing tile policy. |
| prefill | full-attn prefill | `3200 B` | `256` | `0` | AOTriton/full-attn spill exists but is only `~4%` of the 4K prefill trace. |
| decode | dense Q8_0 T16 GEMV | `0 B` | `64` | `512` | top decode bucket is scratch-free; blind launch-bound retuning has little evidence now. |
| decode | selected dual Q4_K T16 GEMV | `0 B` | `200` | `1024` | scratch-free but high VGPR; candidate only if ISA shows an easy pressure cut. |
| decode | Q6 lm-head T16 GEMV | `0 B` | `72` | `512` | scratch-free; prior Q6 d-load/lb variants stayed no-hold. |

Retained notes:

- **G-P1 launch-bound default retained (2026-06-15).** Lowering
  `HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS` from `2` to `1` in the selected Q4_K
  T16 WMMA prefill kernel kept gate IDs stable, kept tracked peak at
  `21.335 GiB`, and moved the GPU1 gate medians to `1658.695 / 126.334 tok/s`
  (`512/128`) and `1850.311 / 115.114 tok/s` (`4K/128`).
- **G-D3 selected dual+SiLU GEMV launch-bound retained (2026-06-15).** Relaxing
  `q4_k_t16_selected_dual_silu_direct_gemv_kernel` from
  `__launch_bounds__(256, 2)` to `__launch_bounds__(256, 1)` preserved gate IDs,
  kept tracked peak at `21.335 GiB`, and moved the GPU1 gate medians to
  `1644.664 / 126.993 tok/s` (`512/128`) and `1851.330 / 115.703 tok/s`
  (`4K/128`).
- **G-D3 selected down GEMV launch-bound retained (2026-06-15).** Relaxing
  `qk_t16_selected_direct_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 2)` preserved gate IDs, kept tracked peak at
  `21.335 GiB`, and moved the GPU1 gate medians to `1647.390 / 127.012 tok/s`
  (`512/128`) and `1855.806 / 115.805 tok/s` (`4K/128`). Final promotion still
  needs the `128K/128` memory/throughput check.

Initial focused lanes from evidence:

1. **G-D2 first for decode:** dense Q8_0 T16 GEMV is the dominant decode bucket
   on both gates (`~37-38%` of decode kernel time), so launch-bound/tile-shape
   tuning has the cleanest same-suite decode upside.
2. **G-P1 first for prefill:** selected dual Q4_K WMMA prefill is the largest
   prefill bucket on both gates (`~38%`), so any prefill push should start there
   before chasing smaller glue kernels.
3. **Secondary decode checks:** full-attention decode matters at 4K (`10.10%`)
   and lm-head Q6 T16 is stable at `~7.4-7.7%`; keep them as follow-ups after
   the Q8_0 T16 decode bucket is audited.

No-hold notes:

- **G-P1 selected-down T16 prefill launch-bound=1 rejected (2026-06-15).**
  Relaxing `gguf_k_t16_selected_prefill` (selected single-output/down WMMA
  prefill) from `__launch_bounds__(32, 2)` to `__launch_bounds__(32, 1)`
  preserved generated IDs and memory and improved the short gate
  (`512/128` prefill/decode `1647.390 / 127.012 -> 1654.917 / 127.066 tok/s`),
  but regressed the retained `4K/128` gate (`1855.806 / 115.805 ->
  1850.369 / 115.758 tok/s`). Keep selected-down T16 WMMA prefill at `32,2`.
- **G-P2 AOTriton prefill threshold=1024 rejected (2026-06-15).** Raising
  the default GGUF AOTriton prefill threshold from `512` to `1024` preserved
  memory and the `4K/128` generated ID, but changed the `512/128` final token
  (`220 -> 318`) and dropped short-prefill throughput (`1647.390 ->
  1516.829 tok/s`). Keep the GGUF AOTriton prefill threshold default at `512`.
- **G-P2 Q8 T16 prefill launch-bound=4 rejected (2026-06-15).** Relaxing
  `gguf_q8_0_t16_prefill_wmma_kernel` from `__launch_bounds__(32, 8)` to
  `__launch_bounds__(32, 4)` preserved generated IDs and memory and improved
  prefill (`512/128` `1647.390 -> 1655.202 tok/s`, `4K/128` `1855.806 ->
  1859.871 tok/s`), but regressed both retained decode medians (`127.012 ->
  126.995 tok/s`, `115.805 -> 115.610 tok/s`). Keep Q8_0 T16 prefill WMMA at
  `32,8` until a shape policy can protect decode.
- **G-P2 Q8 T16 prefill tile_n=16 rejected (2026-06-15).** Changing the default
  T16 WMMA prefill tile policy so rows `<2048` used `tile_n=16` preserved IDs
  and memory, but regressed both prefill medians (`1647.390 -> 1621.701 tok/s`,
  `1855.806 -> 1806.887 tok/s`) and both retained decode medians (`127.012 ->
  126.991 tok/s`, `115.805 -> 115.624 tok/s`). Keep the default `tile_n=32` for
  rows `>=32`.
- **G-P2 Q8 T16 dual gate+up WMMA prefill rejected (2026-06-16).** Adding a
  resident Q8_0 T16 dual gate+up WMMA prefill kernel/wrapper/dispatch path
  preserved generated IDs, memory, and bit-exact focused dual-vs-single tests,
  but regressed retained prefill (`512/128` `1647.390 -> 1621.518 tok/s`,
  `4K/128` `1855.806 -> 1850.689 tok/s`) and the `4K/128` decode gate
  (`115.805 -> 115.764 tok/s`) despite improving `512/128` decode
  (`127.012 -> 127.189 tok/s`). Keep Q8_0 T16 rows>1 gate/up prefill on the
  existing singleton WMMA path unless a tile/codegen profile proves a fused
  variant avoids the prefill loss.
- **G-P4 chunk-min=8192 rejected (2026-06-15).** Raising the
  auto-chunk minimum from `1025` to `8192` disabled chunking for the 4K gate and
  preserved generated IDs, but raised tracked peak (`21.335 -> 21.416 GiB`),
  dropped `512/128` prefill (`1647.390 -> 1592.474 tok/s`), and regressed both
  retained decode medians (`127.012 -> 126.952 tok/s`, `115.805 ->
  115.617 tok/s`) despite improving `4K/128` prefill (`1855.806 ->
  1864.260 tok/s`). Keep the auto-chunk minimum at `1025`.
- **G-P4 MoE-only chunk=2048 rejected (2026-06-15).** Raising only the default
  auto-tuned MoE prefill chunk from `1024` to `2048` preserved generated IDs and
  memory and improved `4K/128` prefill (`1855.806 -> 1857.311 tok/s`), but left
  `512/128` prefill flat/slightly lower and regressed both retained decode
  medians (`127.012 -> 126.959 tok/s`, `115.805 -> 115.717 tok/s`). Keep the
  mid-context MoE chunk at `1024`.
- **G-P4 linear/MoE chunk=2048 rejected (2026-06-15).** Raising the default
  auto-tuned linear/MoE prefill chunks from `1024` to `2048` preserved generated
  IDs and memory and improved prefill (`512/128` `1647.390 -> 1661.258 tok/s`,
  `4K/128` `1855.806 -> 1859.452 tok/s`), but regressed both retained decode
  medians (`127.012 -> 126.874 tok/s`, `115.805 -> 115.670 tok/s`). Keep the
  mid-context linear/MoE chunks at `1024` until a shape-specific policy can
  protect decode.
- **G-P3/G-P4 full-attn query chunk=2048 rejected (2026-06-15).** Lowering the
  auto-tuned full-attention query chunk from `4096` to `2048` reduced tracked
  peak memory (`21.335 -> 20.689 GiB`), but changed the `4K/128` final token
  (`570 -> 15`) and regressed throughput (`4K/128` prefill `1855.806 ->
  1814.443 tok/s`, decode `115.805 -> 115.546 tok/s`; `512/128` decode
  `127.012 -> 126.881 tok/s`). Keep the mid-context full-attention query chunk
  at `4096`; smaller query chunks need an exactness fix before memory-policy
  promotion.
- **G-D3 selected SiLU maxThreads=128 rejected (2026-06-15).** Matching
  `q4_k_t16_selected_dual_silu_direct_gemv_kernel` launch bounds to its actual
  128-thread launch (`256,1 -> 128,1`) preserved generated IDs and memory, but
  regressed the retained selected-down gate (`115.805 -> 115.669 tok/s`) and
  lowered `512/128` prefill to `1597.871 tok/s`. Keep the selected dual+SiLU T16
  GEMV launch bound at `256,1`.
- **G-D3 selected down launch-bound=1 rejected (2026-06-15).** Further
  relaxing `qk_t16_selected_direct_gemv_kernel` from `__launch_bounds__(128, 2)`
  to `__launch_bounds__(128, 1)` preserved generated IDs and memory, but
  regressed the retained selected-down gate (`115.805 -> 115.570 tok/s`) and
  lowered both prefill medians. Keep selected-down T16 GEMV at `128,2`.
- **G-D3 selected down launch-bound=3 rejected (2026-06-15).** Tightening
  `qk_t16_selected_direct_gemv_kernel` from `__launch_bounds__(128, 2)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and nudged
  `512/128` decode up (`127.012 -> 127.040 tok/s`), but regressed `512/128`
  prefill (`1647.390 -> 1618.596 tok/s`), `4K/128` prefill (`1855.806 ->
  1850.923 tok/s`), and the retained `4K/128` decode gate (`115.805 ->
  115.617 tok/s`). Keep selected-down T16 GEMV at `128,2`.
- **G-D2 Q8 launch-bound=3 rejected (2026-06-15).** Relaxing all four Q8_0 T16
  GEMV decode kernels from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and nudged
  `512/128` decode up (`127.012 -> 127.112 tok/s`), but regressed the retained
  `4K/128` gate (`115.805 -> 115.704 tok/s`) and lowered 4K prefill. Keep Q8_0
  T16 GEMV decode at `128,4`.
- **G-D2 Q8 dual-split launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_dual_split_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  `512/128` prefill (`1647.390 -> 1664.017 tok/s`), but regressed both retained
  decode gates (`127.012 -> 126.960 tok/s`, `115.805 -> 115.688 tok/s`) and
  `4K/128` prefill (`1855.806 -> 1853.900 tok/s`). Keep all Q8_0 T16 GEMV
  decode variants at `128,4` until variant-level codegen evidence says
  otherwise.
- **G-D2 Q8 single launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  `512/128` decode (`127.012 -> 127.190 tok/s`) and `4K/128` prefill
  (`1855.806 -> 1857.904 tok/s`), but regressed `512/128` prefill sharply
  (`1647.390 -> 1624.324 tok/s`) and the retained `4K/128` decode gate
  (`115.805 -> 115.740 tok/s`). Keep the single Q8_0 T16 GEMV variant at
  `128,4`.
- **G-D2 Q8 dual launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_dual_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  `512/128` decode (`127.012 -> 127.140 tok/s`) plus `4K/128` prefill
  (`1855.806 -> 1859.269 tok/s`), but regressed `512/128` prefill
  (`1647.390 -> 1645.535 tok/s`) and the retained `4K/128` decode gate
  (`115.805 -> 115.692 tok/s`). Keep the dual Q8_0 T16 GEMV variant at
  `128,4`.
- **G-D2 Q8 triple-split launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_triple_split_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  prefill (`512/128` `1647.390 -> 1665.444 tok/s`, `4K/128` `1855.806 ->
  1856.539 tok/s`), but regressed both retained decode gates (`127.012 ->
  126.937 tok/s`, `115.805 -> 115.633 tok/s`). Keep the triple-split Q8_0 T16
  GEMV variant at `128,4`.
- **G-D2 Q8 launch-bound=5 rejected (2026-06-15).** Tightening all four Q8_0
  T16 GEMV decode kernels from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 5)` preserved generated IDs and memory, but regressed
  both retained decode medians (`127.012 -> 126.870 tok/s`, `115.805 ->
  115.565 tok/s`) and did not improve prefill. Keep Q8_0 T16 GEMV decode at
  `128,4`.
- **G-D2 Q8 half-pointer d-load rejected (2026-06-15).** Loading all four Q8_0
  T16 GEMV decode variants' fp16 `d[16]` scales through
  `reinterpret_cast<const half_t*>(tile)[col]` instead of `fp16_bytes_to_float`
  preserved generated IDs and memory, but regressed both prefill medians
  (`1647.390 -> 1626.168 tok/s`, `1855.806 -> 1853.323 tok/s`) and both decode
  medians (`127.012 -> 126.941 tok/s`, `115.805 -> 115.560 tok/s`). Keep the
  byte/union `fp16_bytes_to_float` loads for Q8_0 T16 GEMV.
- **G-D2 scale broadcast rejected (2026-06-15).** Replacing per-lane Q8_0 T16
  scale loads with `__shfl` broadcast preserved correctness but regressed the
  gate metric from `114.602` to `106.905 tok/s` (`-6.7%`) and reduced both gate
  decode medians. The original per-lane scale loads are faster on this GPU;
  do not retry this exact change without new ISA/rocprof evidence.
- **G-D2 block64 rejected (2026-06-15).** Launching Q8_0 T16 GEMV with 64
  threads instead of 128 passed the synthetic Q8 fixture but changed full-model
  gate tokens (`220 -> 97799`, `570 -> 28944`) and still regressed decode vs the
  original baseline (`114.602 -> 111.797 tok/s` primary metric). Keep the 128-
  thread launch unless a new accumulation-order-safe variant is proven.
- **G-D2 Q8 launch-bound=2 rejected (2026-06-15).** Relaxing the Q8_0 T16 GEMV
  `__launch_bounds__(128, 4)` minimum-block count to `2` preserved generated IDs
  and memory, but slightly regressed the current lb1 gate metric
  (`115.114 -> 115.085 tok/s`) and hurt `512/128` prefill noise. Keep the Q8_0
  T16 GEMV launch-bound at `128,4` unless an occupancy/code-object census shows
  a clearer pressure problem.
- **G-D5 Q6 launch-bound=3 rejected (2026-06-15).** Relaxing the Q6_K T16
  lm-head GEMV from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory, but regressed
  both gate decode medians versus the retained selected-down lb2 row
  (`127.012 -> 126.932 tok/s`, `115.805 -> 115.631 tok/s`). Keep Q6_K T16
  GEMV at `128,4`.
- **G-D5 Q6 launch-bound=2 rejected (2026-06-15).** Relaxing the Q6_K T16
  lm-head GEMV `__launch_bounds__(128, 4)` minimum-block count to `2` preserved
  generated IDs and memory, but regressed the current lb1 gate metric
  (`115.114 -> 115.013 tok/s`) and lowered `512/128` prefill to `1635.484 tok/s`.
  Keep Q6_K T16 GEMV at `128,4`.
- **G-D5 Q6 launch-bound=5 rejected (2026-06-15).** Tightening the Q6_K T16
  lm-head GEMV from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 5)` preserved generated IDs and memory, but regressed
  both retained decode medians (`127.012 -> 126.784 tok/s`, `115.805 ->
  115.604 tok/s`) and lowered both prefill medians. Keep Q6_K T16 GEMV at
  `128,4`.
- **G-D5 Q6 per-tile d preload rejected (2026-06-15).** Preloading the Q6_K T16
  lm-head GEMV per-column `d[16]` and scales pointer outside the `k` loop
  preserved generated IDs and memory and improved `512/128` prefill
  (`1647.390 -> 1656.158 tok/s`), but regressed `4K/128` prefill
  (`1855.806 -> 1855.017 tok/s`) and the retained `4K/128` decode gate
  (`115.805 -> 115.579 tok/s`). Keep the original in-loop `d` loads unless an
  ISA/occupancy profile proves the preload no longer costs the 4K path.
- **G-D5 Q6 half-pointer d-load rejected (2026-06-15).** Loading the Q6_K T16
  lm-head GEMV per-column `d` scale through
  `reinterpret_cast<const half_t*>(tile + Q6_T16_D_OFFSET)[col]` preserved
  generated IDs and memory and slightly improved `512/128` decode
  (`127.012 -> 127.089 tok/s`), but regressed `512/128` prefill
  (`1647.390 -> 1593.584 tok/s`), `4K/128` prefill
  (`1855.806 -> 1852.882 tok/s`), and the retained `4K/128` decode gate
  (`115.805 -> 115.640 tok/s`). Keep the byte/union `fp16_bytes_to_float` Q6
  `d` load unless a disassembly-guided variant avoids the prefill/4K loss.
- **G-D4 split decode threshold=8192 rejected (2026-06-15).** Raising the
  full-attention split/gate fused decode threshold from `1024` to `8192` forced
  the `4K/128` gate onto the direct context + gate-mul path and looked faster
  (`115.805 -> 125.592 tok/s`), but changed the final token (`570 -> 263`).
  Keep the split threshold default at `1024`; direct 4K decode is
  correctness-invalid until fixed.
- **G-D4 split decode threshold=512 rejected (2026-06-15).** Lowering the
  full-attention split/gate fused decode threshold from `1024` to `512` routed
  the short gate through the warp-split path and preserved memory, but changed
  the `512/128` final token (`220 -> 17`) and regressed decode sharply
  (`127.012 -> 113.635 tok/s`); `4K/128` decode also slipped
  (`115.805 -> 115.742 tok/s`). Keep the split threshold default at `1024`.
- **G-D4 grouped-GQA min-context=512 rejected (2026-06-15).** Lowering the
  grouped GQA split decode default threshold from `4096` to `512` enabled the
  grouped path on the short gate and preserved generated IDs and memory, but
  regressed both retained decode medians (`127.012 -> 126.935 tok/s`,
  `115.805 -> 115.652 tok/s`) and sharply lowered `512/128` prefill. Keep the
  grouped GQA min-context default at `4096`.
- **G-D4 GQA warp-split default rejected (2026-06-15).** Disabling the grouped
  GQA split full-attention decode default preserved generated IDs and memory, but
  regressed `4K/128` decode sharply (`115.114 -> 112.597 tok/s`). Keep grouped
  GQA split decode enabled for mid-context GGUF unless a later shape-specific
  threshold proves otherwise.
- **G-H2 graph4 rejected (2026-06-15).** `--graph-steps-per-replay 4` reused the
  existing multi-step capture support and kept the 4K token stable, but changed
  the `512/128` final token (`220 -> 11`). Treat GGUF multi-step graph replay as
  correctness-blocked until a fixture proves the captured position/token state
  advances exactly across replay groups.

## What we copy from the MTP/DFlash/megakernel successes

The MTP sprint moved from `0.758x / 27.8 ms` to a retained `1.023x / 14.134 ms`
row by treating every cycle component as measurable. DFlash accepted a `1.231x`
27B dense row by adding an online confidence gate only after exactness and
same-session AR baselines were locked. The megakernel work prevented a bad
default by proving a seductive one-launch FFN kernel was slower on the GPU.

For GGUF, copy these rules:

1. **Same-suite or it does not count.** Every candidate compares against the
   same branch, model file, prompt tokens, context, warmups, GPU, ROCm stack, and
   correctness fixture.
2. **Profile before changing.** Start each lane with a `rocprofv3 --kernel-trace`
   plus bucket summary. If a candidate does not target a measured bucket, park it.
3. **Keep exact micro-wins.** Small retained wins from MTP were additive
   (`0.04-0.11 ms/cycle` was worth keeping). GGUF should keep small exact wins
   when the same benchmark suite is non-regressive.
4. **Re-profile when the operating point moves.** MTP's B=1 retune found wins
   that were no-holds at B=3. GGUF must re-check candidates separately for
   `512`, `4K`, `32K`, and c>N shapes.
5. **Do not worship launch count.** The FFN megakernel removed launches but lost
   grid parallelism and became `2.66x` slower than production. GGUF fusions must
   preserve occupancy/coalescing and remove real memory traffic, not just calls.
6. **Reject memory mirages.** Sidecars or raw+packed duplication that make a
   benchmark impossible on 24 GiB cards are blocked even if W7900 runs.
7. **Promote only with rollback semantics.** Exact, non-regressive performance
   paths become default; default-off paths need a concrete blocker in
   `docs/REFACTOR.md` or this file.

## Baseline refresh protocol

Run these before editing kernels. Use a clean shell and do not let profiled
processes spawn `hipcc`. The active tuning loop uses GPU1
(`HIP_VISIBLE_DEVICES=1`), which maps to the 24 GiB-class testbed on this host;
verify the sysfs card name before llama.cpp peak-memory runs.

```bash
# From /home/lhl/hipEngine-gguf-tuning
ROOT=/home/lhl/mambaforge/envs/therock
PY=$ROOT/bin/python3.12
$ROOT/bin/hipcc --version > /tmp/hipengine-hipcc-version-713.txt
```

### hipEngine GGUF sweep

```bash
GGUF_S=/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$GGUF_S" --quant gguf_q4_k_s \
  --workloads 512/128 4K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-gate-hipengine-q4ks.json

# Promotion/final check, after a candidate survives the primary gates.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$GGUF_S" --quant gguf_q4_k_s \
  --workloads 128K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-final-128k-hipengine-q4ks.json
```

Also run Q4_K_M when it fits the intended target:

```bash
GGUF_M=/home/lhl/hipEngine/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$GGUF_M" --quant gguf_q4_k_m \
  --workloads 512/128 4K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-gate-hipengine-q4km.json
```

### llama.cpp refresh

Use the checked-out local binaries so we compare against what the user is
actually seeing:

```bash
MODEL=/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
HIP_VISIBLE_DEVICES=1 python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench \
  --model "$MODEL" --backend hip \
  --workloads 512/128 4K/128 \
  --repetitions 3 --ngl 99 --flash-attn 1 \
  --cache-type-k f16 --cache-type-v f16 --poll 10 --card-name card0 \
  --extra-args "-dev ROCm0" \
  --output benchmarks/results/<date>-gpu1-gguf-tuning-baseline-llamacpp-hip-q4ks.json

python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  --model "$MODEL" --backend vulkan \
  --workloads 512/128 4K/128 \
  --repetitions 3 --ngl 99 --flash-attn 1 \
  --cache-type-k f16 --cache-type-v f16 --poll 10 --card-name card0 \
  --extra-args "-dev Vulkan0" \
  --output benchmarks/results/<date>-gpu1-gguf-tuning-baseline-llamacpp-vulkan-q4ks.json
```

### PARO c=1 reference on the same host

```bash
PARO=/home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine paro --model "$PARO" --backend hip_gfx1100 \
  --shared-expert-format packed_paro_w4 --token-id 9707 \
  --workloads 512/128 4K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 4 \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-baseline-hipengine-paro.json
```

### Correctness gates for GGUF tuning

Use the narrowest gate first, then the full model gate before retaining a perf
row:

```bash
# Dense 0.8B and 35B smoke fixture path.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
  PYTHONPATH=. python3 scripts/qwen35_gguf_e2e_correctness.py --repeat 2

# qwen35moe safety gate when touching 35B GGUF kernels/materialization.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
  PYTHONPATH=. python3 scripts/qwen35_gguf_p9_e2e_correctness.py

# Targeted bundles for decode-repack / T16 work.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
  python3 -m pytest \
    tests/test_gguf_t16_repack.py \
    tests/test_gguf_q8_0_t16_gemv_decode.py \
    tests/test_gguf_t16_selected_gemv_decode.py \
    tests/test_gguf_q6_k_t16_gemv_decode.py \
    tests/test_gguf_gemv_decode_dispatch.py \
    tests/test_qwen35_gguf_compact_moe_gemv_routing.py -q
```

### Profile capture and summary

Warm the JIT outside the profiler, then trace only cached runs. Store raw CSVs
under `/tmp`; commit only compact summaries.

```bash
RUN=/tmp/hipengine-gguf-tuning/<date>-q4ks-512
mkdir -p "$RUN"

# Warmup/build outside rocprofv3.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_gguf_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
  --quant gguf_q4_k_s --prompt-length 512 --decode-tokens 16 \
  --persistent-session --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json "$RUN/warmup.json"

# Trace a short decode window for kernel mix.
rocprofv3 --kernel-trace -d "$RUN/rocprof" -o q4ks512 -f csv -- \
  env HIP_VISIBLE_DEVICES=1 \
      HIPENGINE_GGUF_DECODE_REPACK=1 \
      HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
      PYTHONPATH=. \
      "$PY" scripts/qwen35_gguf_bench.py \
        --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf \
        --quant gguf_q4_k_s --prompt-length 512 --decode-tokens 16 \
        --persistent-session --force-bulk-prefill --bulk-prefill-attention-mode bulk \
        --use-wmma-prefill --use-gemv-decode \
        --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
        --json "$RUN/profiled.json"

# Summarize the CSV into a compact artifact.
python3 scripts/qwen35_gguf_rocprof_summary.py \
  --csv "$RUN/rocprof/q4ks512_kernel_trace.csv" \
  --tokens-decode 16 \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-rocprof-summary.json
```

## Tuning lanes

Use stable IDs in commits, artifacts, and `WORKLOG.md`.

### M lane — Measurement and attribution

| ID | Candidate | Why | Acceptance |
| --- | --- | --- | --- |
| G-M0 | Refresh hipEngine GGUF, llama.cpp HIP/Vulkan, and PARO rows from this worktree. | The current scorecard mixes quant variants and dates. | Compact JSON artifacts for each runner, exact commands, clean environment metadata. |
| G-M1 | Build a matched-token prompt fixture shared by hipEngine and llama.cpp. | Avoid token/template drift when comparing decode and MTP-bearing GGUFs. | Same token IDs and prompt length proven in artifacts. |
| G-M2 | Produce per-shape GGUF rocprof bucket summaries for `512`, `4K`, `32K`, `128K`. | P9.C showed selected-MoE and Q8 buckets moved with shape; do not optimize blind. | `qwen35_gguf_rocprof_summary.py` artifacts list top buckets, dispatches, VGPR/scratch, and legacy fallback presence. |
| G-M3 | Add/refresh code-object occupancy and scratch census for hot GGUF kernels. | Decode is memory-bound and occupancy-sensitive; any scratch on hot kernels is a bug. | Hot kernels report `Scratch_Size=0`, acceptable VGPR, and no unexpected LDS expansion. |
| G-M4 | Memory residency census by tensor family/layout. | GPU1 gates fit but leave only ~2 GiB free, and stale full-sweep rows exceeded 24 GiB; duplicate layouts hide in totals. | Artifact breaks down raw, T16, KV, scratch, graph, and sampler allocations. |

### D lane — Decode throughput

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-D1 | Eliminate any legacy `*_prefill_out*` or raw GGUF fallback from decode traces. | `GGUF_DECODE_REPACK.md` says remaining legacy fallback is a regression, especially lm-head. | rocprof summary shows T16/GEMV decode buckets and no unplanned legacy decode buckets. |
| G-D2 | Tune T16 GEMV launch bounds / tile shape for Q8T16, Q4T16, Q5T16, Q6T16 at rows=1. | T16 is the right replacement layout but may not be at the best occupancy point. | Same-suite decode improvement, `Scratch_Size=0`, no correctness regression. |
| G-D3 | Router + selected-MoE decode fusion where it removes real memory traffic. | MTP router fusion was a large retained win; GGUF MoE routing still has small kernels per layer. | Generated IDs/logits stable; launch count and wall improve; no extra HBM staging. |
| G-D4 | Direct full-attention short-context producer/reduce path parity with PARO direct-gate work. | MTP one-split direct gate saved wall exactly; GGUF full-attn decode may carry similar split/reduce overhead. | Exact full-attention fixture; `512`/`4K` decode improved or no-held. |
| G-D5 | lm-head Q6T16/argmax fusion or top-1-only path. | Final logits are a large dense projection; avoid full materialization if only greedy token is needed. | Greedy token/logprob semantics preserved; sampling/logprob paths fall back explicitly. |
| G-D6 | c>N GGUF serial bridge replacement with true resident batch path. | PARO c=4/c=8 shows large aggregate gains; GGUF currently has more serial behavior. | Separate c>N correctness/provenance gate; not required for c=1 promotion. |

### P lane — Prefill throughput

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-P1 | Revisit selected-MoE raw GGUF-K WMMA redesign from P9.C with the latest T16/repack context. | P9.C ended with a `~30 ms` 512/0 target-bucket gap; shallow sidecars regressed, but a deeper layout may still be needed. | Target bucket moves materially without >24 GiB duplicate storage. |
| G-P2 | Shape-specific Q8_0 shared/dense WMMA schedule. | Q8_0 bucket is still large; P9.C1 showed shape-specific tile rules mattered. | 512/0 and 4K/0 prefill both non-regressive; code path remains registered by quant/layout key. |
| G-P3 | Full-attention prefill glue parity with PARO/AOTriton path. | Long-context prefill is chunk/attention sensitive. | 32K/128 and 128K/128 prefill improve without decode/memory regression. |
| G-P4 | Chunk auto-tune and memory budget policy for Q4_K_S/Q4_K_M. | Current GPU1 gates fit at `21.335 GiB`, but final `128K/128` may need chunk/KV/scratch policy to stay inside 24 GiB. | Same throughput class with lower peak, or clear W7900-only label. |

### H lane — Host/runtime and graph replay

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-H1 | Cache resident tensor views, scratch handles, and per-layer dispatch objects in GGUF like MTP verifier caches. | MTP host caches removed real milliseconds without math changes. | Profile shows host window reduction, exact outputs unchanged. |
| G-H2 | Audit graph capture/replay setup for per-token validation or recapture overhead. | Graph replay helped PARO only a few percent; GGUF may still be doing avoidable host work. | Wall improves with same kernel time, or candidate is rejected. |
| G-H3 | Collapse ctypes wrapper overhead only if measured >3% of decode wall. | C++ host loop is not worth doing without proof. | Before/after host-marker or Python profile evidence. |

### L lane — Layout and memory

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-L1 | Make replacement-layout materialization truly replace raw tensors for all promoted T16 paths. | Decode-repack is only useful if it does not duplicate model weights. | Allocation census proves no raw+packed duplicate for covered tensors. |
| G-L2 | Q4_K_M/Q4_K_S model-specific residency plan. | Q4_K_S saves selected-down bytes but current run still peaks high. | Peak target under 24 GiB for 512/128 and 4K/128, or documented blocker. |
| G-L3 | Decide whether any sidecar is worth keeping. | P9.C5 side metadata and expert pack8 sidecar were slower or too large. | Only retained if replacement, not duplicate, and E2E faster. Otherwise add `docs/REFACTOR.md` removal note. |

## First sprint checklist

1. Create GPU1 gate artifacts for hipEngine GGUF Q4_K_S/Q4_K_M, local llama.cpp
   HIP/Vulkan, and PARO where it fits on TheRock 7.13. Keep W7900 comparison
   artifacts only when GPU1 cannot fit a required final/promotion shape.
2. Generate `G-M2` rocprof bucket summaries for at least `512/128` and `4K/128`.
3. Answer these from data before editing kernels:
   - Is the gap primarily decode kernels, prefill kernels, host overhead, or memory/chunk policy?
   - Which kernel family is the top decode bucket after T16 repack: selected MoE, dense Q8/Q6, full attention, router, lm-head, or glue?
   - Are any legacy GGUF raw/prefill-out kernels still running in measured decode?
   - How much peak memory is raw tensors, T16 tensors, KV, scratch, and graph capture?
4. Pick **one** highest-share bucket and run a focused multiloop-style pass:
   hypothesis -> code -> correctness gate -> profile -> keep/revert/log. Primary
   pass/fail is `512/128` plus `4K/128`; `128K/128` is the final promotion gate.
5. If a change is exact and same-suite non-regressive, make it default and
   update artifacts/rollups. If it stays gated, record the blocker.

## Promotion policy

A GGUF tuning change is promoted only when all of the following hold:

- Relevant GGUF correctness fixtures pass, including qwen35moe when 35B paths are touched.
- `rocprofv3 --kernel-trace` confirms the intended kernel(s) ran and no unexpected fallback dominates.
- Same-suite benchmark improves prefill and/or decode on the GPU1 `512/128` and `4K/128` gates, or removes memory/launch/KV overhead without throughput regression.
- Benchmark artifact follows `docs/BENCHMARK.md` and rollup updates are made for accepted performance rows.
- No torch hot-path import, no llama.cpp hot-path FFI, no model/dispatch `if backend == ...` or `if quant == ...` branch.
- No unbounded duplicate residency; GPU1 `128K/128` is checked before promotion, and W7900-only rows are labeled as such.

## Do-not-chase list for this lane

- **Do not** start with another raw-GGUF `prefill_out` micro-tweak if the profile
  still shows replacement-layout coverage gaps.
- **Do not** re-enable unsafe qwen35moe fast paths unless the P9.E2-style gate
  passes with effective fast paths actually enabled.
- **Do not** copy llama.cpp kernels blindly. Use llama.cpp for block math,
  subgroup-shape ideas, and baselines; hipEngine kernels must fit the raw-pointer
  ABI, registry, and resident layout.
- **Do not** assume wave64/Vulkan subgroup behavior maps to HIP. Prove it with a
  tensor-level fixture and `rocprofv3` evidence.
- **Do not** treat a single synthetic 512-token prompt as a quality proof. It is
  a speed sentinel; promotion still needs deterministic generated IDs/logits and
  the documented correctness gate.

## Open questions

- Is the user's latest `~/llama.cpp/` faster than the stale README HIP rows, and
  if so, which commit/build flag changed the target?
- Does Q4_K_S or Q4_K_M become the canonical tuning target for hipEngine GGUF?
- Can Q4_K_S fit the 24 GiB-class envelope after replacement-layout and scratch
  cleanup, or is Q4_K_M the consumer-card default?
- Which GGUF path should get native sampling/logprob parity first: c=1 greedy
  only, c=1 sampled, or c>N batch?
- Is any approximate/relaxed GGUF math acceptable, or do we keep strict GGML
  dequant parity for all promoted rows?
