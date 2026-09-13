# Decode graph attribution (2026-09-11) - the four-arm pure-kernel census

`scripts/gguf_decode_graph_rocprof_driver.py` under `rocprofv3
--kernel-trace`, graph-replay mode (the published decode protocol), 32
measured tokens after a 512-token prefill; per-kernel pure durations from
the trailing device-timeline window. Full per-kernel table (top 40 per
arm) in `four-arm-per-kernel.json`.

| arm | kernels/tok | pure ms/tok | wall ms/tok (bench) |
| --- | ---: | ---: | ---: |
| UD K_M | 898 | 35.23 | 38.2 |
| plain K_M | 822 | 27.78 | 29.3 |
| UD K_S | 918 | 36.16 | 38.8* |
| plain K_S | 950 | 26.16 | 30.5 |

*profiled-process wall; the bench medians are 26.20/25.53 UD.

The UD-vs-plain pure-kernel gap: +7.45 ms/token (K_M), +10.0 (K_S);
graph-node overhead adds ~1 ms more for UD (898 vs 822 launches/token).

## The gap's structure (K_M numbers; K_S in parens where different)

1. **IQ4_XS local32: 8.17 (10.87) ms/tok, 117 (172) launches.** UD-only
   family (plain has no IQ4_XS). The dominant shapes run as singles:
   61/tok `kernel<2>` gate/up at 84 us + 30/tok `kernel<4>` down at 67 us.
   No gate+up dual exists for IQ4_XS decode - plain's Q4 dense FFN runs a
   fused dual+SiLU (`q4_k_t16_dense_dual_local32_silu`, 8.43 ms for MORE
   elements). Lever: an IQ4_XS dual local32 + SiLU owner sharing the
   activation read (est. -2 to -3 ms) plus launch-count reduction.
2. **Q5 MoE selected GEMV: 7.60 (4.06) ms/tok, 113 (68) launches.** UD's
   Q5_K expert population is larger (131 tensors vs plain's 48), and the
   `qk_t16_selected_direct_gemv<u16,5>` costs 74.5 us/launch where plain's
   Q4 local32 single does the same-shaped work at ~33 us. Lever: a
   local32/optimized selected owner for Q5 (needs a why-is-it-2.3x study
   first: unpack cost vs layout).
3. **Strict IQ GEMV: 4.64 (7.94) ms/tok, 18 (39) launches.** The dense
   IQ4_NL/IQ3_S/IQ2 tensors have no fast decode owner; the strict kernel
   runs 199-392 us/launch for shapes local32 does at 67-84 us (~4x).
   Lever: an IQ4_NL local32 sibling (byte codebook - the local32 design's
   fused LUT applies directly) and IQ3_S/IQ2_S variants; est. -3.4 (-6+)
   ms.
4. Q6 planar: UD is FASTER (2.08 vs 6.24) - the qmicro planar owner wins;
   attention/GDN/norms are at parity (2.4/0.9/1.5 ms).

Combined realistic recovery: -8 to -12 ms pure + graph-overhead reduction
-> UD decode 26.2 -> low-30s tok/s, near plain parity (34.1). This is the
decode campaign's ranked backlog. Quality: every owner replacement re-runs
the tokenized category suite (the local32 route itself remains approximate
and stays gated; the K_M max 2.81e-2 diagnostic remains documented).
