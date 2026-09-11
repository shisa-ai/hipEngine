# Final decode campaign rollup (2026-09-11) — validation and census

The decode campaign's four levers are landed and the final stack is
re-measured end to end: tokenized category gates on the final stack,
the four-arm decode re-measure (graph-replay protocol, XTX, medians of
3, all arms same-conditions), and the pure-kernel re-census via
`scripts/gguf_decode_graph_rocprof_driver.py` + the new
`scripts/gguf_decode_census_summary.py`.

## Campaign ledger (decode, XTX wall tok/s, 512/128)

| lever | commit | K_M | K_S |
| --- | --- | ---: | ---: |
| campaign start (post prefill-parity) | — | 26.20 | 25.53 |
| 1. IQ4_XS gate/up dual local32 + SiLU | `5337f439a` | 26.46 | 25.66 |
| 2. IQ4_NL local32 | `4fc21cf04` | 27.67 | 26.77 |
| 2b. split/scale IQ (IQ3_S/XXS/IQ2_S/XS; K_M IQ3_S pinned) | `b723aa2b8` | 27.70 | 28.41 |
| 3. Q5 dense local32 single | `1570712a1` | **29.98** | **29.49** |

Cumulative: **K_M +14.4%, K_S +15.5%**; prefill unchanged throughout.

## Four-arm decode (GPU1 XTX, medians of 3, same conditions)

| Model | Decode plain | Decode UD | prefill plain | prefill UD |
| --- | ---: | ---: | ---: | ---: |
| `UD-Q4_K_M` | **35.37** | **29.98** | 983-988 | 901-912 |
| `UD-Q4_K_S` | **36.66** | **29.49** | 945-947 | 890-900 |

Decode ratios: K_M **0.848x**, K_S **0.804x** (campaign start: 0.769x /
0.836x).

Note: the plain artifacts **also gained** from lever 3 — plain K_S
carries 60 Q5_K tensors and re-measures 36.66 vs the campaign-start
30.55; the local32 owner serves any Q5_K tensor at the routed shapes,
so the four-arm gap comparison stays same-conditions (both arms
changed) rather than UD-only: plain K_M 34.12 -> 35.37, plain K_S
30.55 -> 36.66 on the same protocol.

## Pure-kernel census (XTX, 32-step window)

The census window is sliced on the trailing 32 `advance_decode_position`
kernels (the per-step position-advance marker) — the pure-gap heuristic
mis-slices plain-arm traces whose decode burst carries host-side
replay stalls, which once made "pure" exceed wall. On XTX:

| arm | pure ms/tok | launches/tok | campaign start (old census) |
| --- | ---: | ---: | ---: |
| UD K_M (final) | **30.26** | 831 | 35.23 / 898 |
| UD K_M (pre-lever-3) | 32.03 | 831 | — |
| UD K_S (final) | **30.02** | 839 | 36.16 / 918 |

UD pure decode fell 4.97 (K_M) / 6.14 (K_S) ms/token across the
campaign. The plain arms crash under rocprofv3 on the XTX (JIT-compile
interception faults, reproducible); their W7900 censuses read 30.48
(plain K_M) / 28.73 (plain K_S) ms per token at 796/920 launches per
token, but the matching W7900 UD runs came out inflated past wall (a
profiler-serialization artifact on the slower card with the 831-launch
UD stack), so the cross-arm pure gap is not published from mixed-card
traces — the wall four-arm above is the same-conditions comparison.

### Attribution corrections and remaining levers

- The campaign-start census keyed kernels on a truncated name, so the
  whole direct-GEMV family merged under the `<u16,5>` Q5 label: the
  "7.60 ms / 113 launches" figure was the family total. On the final
  stack the Q5 dense singles are 69 launches (52.2 us in-situ) with 44
  same-family launches on other quants/shapes.
- **Remaining decode levers (final census):** (a) the Q5 **MoE
  selected-expert** path — 44 launches / 2.22 ms per token (K_M) and 18
  / 0.46 ms (K_S) still run the direct GEMV under the
  `selected_t16_gemv` ABI (a different dispatch than the C1-covered
  dense singles); a selected-ABI local32 owner is the natural next
  lever. (b) Q3_K strict decode — 7 launches / 1.43 ms (K_M), 12 /
  2.43 ms (K_S); the superblock layout has no local32 lane geometry
  yet. (c) the Q5 gate/up dual (9 launches / 1.95 ms on the direct
  dual; two local32 singles + SiLU already beat it, a local32 dual is
  the Q4 pattern).

## Final-stack category gates

`scripts/gguf_ud_combined_stack_gate.py --category-heldout` on the
final stack (seed 7; the category screen is seed-independent - fixed
fixture prompts extended by the incumbent's own greedy continuation -
and the per-seed runs in the lever archives are byte-identical):

- **UD-Q4_K_M: PASS** - pooled mean 1.377e-4, p95 ~4.98e-4, p99
  1.759e-3, max 1.484e-2, top-1 99.91%.
- **UD-Q4_K_S: PASS** - pooled mean 1.446e-4, p99 1.884e-3, max
  2.014e-2, top-1 99.91%.

### The K_M max-row diagnostic trail (documented per the campaign
contract)

The local32-family accumulation-order tail on K_M compounded across
the levers and was gated, not amended: 2.81e-2 (prefill campaign end,
Q3_K prefill pin) -> 3.95e-2 (IQ4_NL admission) -> 5.19e-2 unpinned /
1.785e-1 partial-pin (split/scale round) -> **pinned 3.947e-2** (all
four K_M IQ3_S slots pinned strict via
`GGUF_IQ_DENSE_DECODE_STRICT_SLOTS`) -> **1.484e-2 on the final
stack**. Every approximate owner carries its own admission evidence
(synthetic A/B + real-fixture test + probe/category gates) recorded in
its lever's worklog entry and result directory.
