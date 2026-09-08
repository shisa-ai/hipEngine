# Raw IQ Dense Kernel Optimization Campaign

_Status: active. Optimization ledger for `gguf_iq_dense_strict_kernel`, the
raw dense IQ/Q3_K projection kernel that serves the published Qwen3.8-27B
`UD-Q4_K_M` / `UD-Q4_K_S` files. Last updated: 2026-09-08._

Companion to [`OPTIMIZE-KERNEL-IQ2_XS.md`](OPTIMIZE-KERNEL-IQ2_XS.md), which
covers the *selected-MoE* IQ2_XS path. This file covers the **dense** path.
See [`KERNELS.md`](KERNELS.md) for the catalog, [`ROOFLINE.md`](ROOFLINE.md)
for the RDNA3 model, and [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) for
the strict/production contracts.

## Campaign reprioritized: read this before picking up the queue

On the **same model, same GPU, same harness, same 512-token shape**, the plain
`Qwen3.8-27B-Q4_K_S.gguf` prefills at **160.5 tok/s** while
`Qwen3.8-27B-UD-Q4_K_M.gguf` prefills at **13.77** — 11.2x more GPU time by
kernel trace (3,320 ms vs 37,299 ms), so the gap is device work, not host.

The cause is a route property, not a kernel-quality property. The plain file
reaches the WMMA GEMM prefill family (`gguf_q4_t16_dense_wmma_prefill_*`, plus
rocBLAS Tensile); the UD file is served entirely by GEMV-shaped kernels — and
**even its Q4_K tensors** land on `gguf_q4_k_pack8_prefill_out` rather than the
WMMA prefill kernels, because those require the T16 repacked layouts while UD
tensors are resident as `LAYOUT_RAW_GGUF` or pack8. A GEMV-shaped prefill
re-reads the weight matrix once per row block; a GEMM reads it once.

Separately, the same harness measures the plain file at 160.5 where the
scoreboard publishes **396.1** for that model/quant/GPU, so there is a second
route/protocol gap on `Qwen35GGUFResidentSession.prefill(use_bulk=True)` —
likely the opt-in `use_wmma_prefill`. The true UD headroom is therefore larger
than 12x.

**Consequence for this campaign.** Everything below optimizes inside the slow
regime. The row tile bought 1.79x by cutting the number of weight passes 8x;
routing UD prefill onto the GEMM family changes the exponent rather than the
constant. The decode queue (B4, further LDS work) is worth tens of percent and
is deprioritized under:

1. Reconstruct the published 396.1 protocol and re-measure both files under it.
2. Find why UD Q4_K tensors miss `gguf_q4_t16_dense_wmma_prefill_*` — a
   materialization/layout question in `qwen35_gguf_materialize.py` and
   `qwen35_gguf_consumer_surface.py`, not a kernel one, and the cheapest of the
   three because those kernels already exist and are qualified.
3. A5: a dense entry point over the existing IQ integer-MMQ prefill kernels.

**Root cause found, statically.** `plan_qwen35_gguf_materialization` computes
one model-wide boolean — `use_decode_repack = requested and not
gguf_ar_decode_repack_veto(...)`, where the veto is `any(type in {IQ2_XS,
IQ3_XXS, IQ4_XS})` over all AR layer tensors. The UD files' IQ4_XS tensors trip
it, which strips the T16/x8/planar layouts from **all 554 rank-2 tensors**. So
the UD files plan **0.0%** of their weight bytes into an optimized layout, against
94.8% for the plain file. Lifting it to a per-tensor predicate would put 44.2%
(K_M) / 34.4% (K_S) onto already-qualified kernels *and* save ~1 GB of VRAM,
because the `q4_k_pack8` fallback expands to ~6 bpw.

The plan, the ordered queue, and the queued GPU validation are in
[`UD-OPTIMIZED-ROUTE-PLAN.md`](UD-OPTIMIZED-ROUTE-PLAN.md). Full evidence:
`worklog/entries/20260908T042726.067135Z-lhl-ud-prefill-route-gap-9240cf.md`
and `…T060056.351273Z-lhl-ud-repack-veto-root-cause-c022cc.md`.

## Measurement protocol (read before adding a row)

Every wall-clock number in this campaign so far comes from **zbook**, a
power- and thermal-limited laptop (Radeon 8060S / gfx1151). Two rules follow.

**Sequential single-variant runs are not a valid screen.** Two runs of
byte-identical machine code measured 58.8 and 64.8 GB/s on the same tensor.
Use interleaved A/B: build each candidate into its own cache entry, alternate
their launches inside one process so both see the same clock and thermal
state, and report min-of-N.

**The residual floor is +/-5%.** In an interleaved table, quants whose ISA is
unchanged across arms (IQ3_S, Q3_K when the change touches only IQ4 or sign
tables) read the noise directly. Always keep such a control in the table.
Anything under 5% needs an instruction-count or occupancy argument to be
retained, and is recorded as inconclusive rather than as a win.

Harnesses: `scratchpad/ab_leaf.py` (interleaved leaf bandwidth, rows=1),
`scratchpad/rowbatch_bench.py` (row-slab sweep), `scratchpad/ud_e2e_ab.py`
(same-session end-to-end A/B on the public path).

## What the kernel is bound by

Measured on gfx1151, IQ4_XS `bf16_bf16_out`, T=8:

- 43 VGPRs, **zero scratch**, the full **16 waves/SIMD**. There is no
  occupancy or register problem to solve.
- ~920 instructions issued per inner body to retire 16 FMAs, roughly a fifth
  of them `s_waitcnt`/`s_delay_alu`.

Because occupancy is maximal, VMEM latency is well hidden and the limiter is
**VALU issue**. The campaign's central lesson so far follows from that:

> **Removing a memory operand only pays if it does not add arithmetic.**

## Ledger

### Retained

| Item | Change | Evidence | Commit |
| --- | --- | --- | --- |
| B2 | `iq_signs[128]` -> `i \| ((popc(i)&1)<<7)` | loads 50->34 on IQ3_XXS/IQ2_XS; wall clock inside the floor | `0d20ab81b` |
| B5 | IQ2_XS magnitude ternary -> one packed immediate | IQ2_XS VALU 569->562 | `0d20ab81b` |
| A1 | `R` rows-per-block tile: decode each weight once, reuse across R activations | kernel 2.97-4.05x at rows>=8; end-to-end prefill 1.73x | `0de4d22da` |
| A7 | memoize the loaded CDLL in `launch()` | 10.7 us/call removed from every projection | `0de4d22da` |
| B3 | stage `iq3_grid`, `iq3_xxs_grid`, `IQ2_XS_GRID_PACKED` in LDS | IQ3_S +10.5/+10.4%, IQ3_XXS +7.9/+8.0% across two runs; loads and VALU both fall | `b3a5dbbf7` |

All four are **bit-exact**. The row tile is exact because `R` only changes
which rows share a workgroup: per-`(row, column)` k ownership, FMA order,
wave32 shuffle tree and serial wave-0..3 sum are untouched.

### Rejected, with the measurement kept

| Item | Change | Why rejected |
| --- | --- | --- |
| B1 | `iq4_values[16]` -> packed immediates + `v_perm_b32` gather | removes 16 `global_load_i8` per body but adds ~112 VALU; measured **-5.0%** on the IQ4_XS leaf. The kernel is VALU bound at full occupancy, so an L1-resident byte load is the cheaper operand. |
| R=16 | 16 rows per block | hits the 256-VGPR cap and spills 48 B/lane. R=8 is the retained maximum. |
| B3 (IQ2_S) | stage the 8 KB `iq2_s_grid` | occupancy 16 -> 12 waves/SIMD, measured **-10.6%**. The footprint costs more than the 16 loads it removes. |

The B1 rejection is recorded in the kernel source itself so it is not
re-derived. Note it also overturned this campaign's own initial prediction,
which had ranked B1 as the best value-per-line item available.

### Open queue

| Item | Change | Note |
| --- | --- | --- |
| B4 | payload byte loads -> dword loads | changes the k->lane mapping, so it breaks the declared strict contract. Authorized as a **production-profile** variant this phase, with the strict tile as the registered fallback. |
| A5 | dense entry point over the existing selected-shape IQ integer-MMQ prefill kernels | `gguf_iq_source_mmq_prefill.hip` (IQ4_XS/IQ3_XXS) and `gguf_iq2_xs_mmq_prefill.hip` already exist and are correctness-tested in MoE shape. |
| A6 | host trace of the prefill loop | the ~12 s host vs ~1.2 s GPU dense-era figure predates the row tile; with prefill now 1.73x faster on device, the host share is proportionally larger. |
| — | decode (rows=1) | untouched by the row tile and still the open target: IQ2_XS 41-45 GB/s against Q5_K/Q6_K at 125-147 GB/s on the same host. |

## Revisit on the desktop gfx1151

Re-measurement is cheap; record contingent results rather than over-trusting
laptop rows.

**Direction safe, magnitude provisional**

- Row-slab speedups are far outside the floor, so the win is real; the exact
  ratios are not final.
- **R=8 over R=4 is only 10-13%**, and R=8 costs occupancy (10 vs 16
  waves/SIMD). The crossover may move on a host with different clock
  behaviour. Re-sweep before treating R=8 as final.

**Inconclusive here**

- B2/B5 wall clock (+4.9% IQ2_XS, +1.1% IQ3_XXS) is inside the floor; they are
  retained on instruction count. Confirm sign and size.
- B1's **-5.0% is exactly the floor**. The ISA supports the direction, but one
  clean pass would settle it. If the desktop shows it neutral or positive, the
  VALU-bound model above needs revisiting and so does the B3 reasoning that
  depends on it.
- The end-to-end harness here reports decode 4.23 tok/s where the 2026-09-08
  raw-Q5/Q6 row reports 6.39 from a different harness. **Reconcile the
  protocols** so future UD rows are comparable rather than only
  self-consistent.

**Not yet swept**

- `T` x `R` jointly. T=8 was tuned at R=1; the best column tile may differ once
  R=8 spends registers on row accumulators.
- Per-role slab selection. The shipped policy is one global function of `rows`;
  the raw-K family instead qualifies specific `(quant, output, K, N)` shapes.
- R=8 at rows between 2 and 8, where the slab is partly padding.

## Known unrelated breakage

`tests/test_qwen35_gguf_iq_dispatch.py` does not collect: a circular import
between `hipengine/generation/concurrency2.py` and
`hipengine/generation/registry.py`. Reproduced at HEAD with this campaign's
changes reverted, so it is pre-existing and outside this campaign's scope.
