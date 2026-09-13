# Split/scale IQ local32 decode admission (2026-09-11, decode lever 2b)

Candidate: all six dense-IQ quants on the local32 decode owner
(hip_gfx1100). Incumbent: all-strict decode (the gate script's fixed
arm pair; the production incumbent for the split quants is strict).

The category gate is seed-independent (fixed fixture prompts extended
by the incumbent's own greedy continuation), so the per-seed runs are
numerically identical; the three runs per state are kept as reruns.

- k_s-cat-{7,11,23}.json - UD-Q4_K_S: PASS (mean 9.96e-5, p99 1.41e-3,
  max 6.60e-3, top-1 99.83%). K_S keeps the full route: IQ3_S 3.0 ms/tok
  strict -> local32.
- k_m-cat-7-unpinned-fail.json - UD-Q4_K_M with all four IQ3_S slots
  routed: FAIL (max 5.19e-2, mixed_ja_en single-position outlier).
- k_m-cat-11-pin3-fail.json - three ffn_down slots pinned, ffn_gate
  still routed: FAIL (max 1.785e-1, mixed_ja_en outlier, p99 3.4e-3).
- k_m-cat-{7,11,23}-pinall.json / k_m-cat-23.json - all four IQ3_S
  slots pinned via GGUF_IQ_DENSE_DECODE_STRICT_SLOTS: PASS (mean
  1.666e-4, p99 2.731e-3, max 3.947e-2) - bit-identical to the
  NL-round admitted state, i.e. the pins restore exactly the shipped
  XS+NL decode plus strict IQ3_S.
- k_{m,s}-{7,11,23}-flat.json - the default (non-category) screen on
  the unpinned route, all PASS (K_M mean 2.7e-5; auxiliary).

Natural-prompt probes (corrected arm protocol): see
../../local32-decode-gate/split-route-2026-09-11/.
