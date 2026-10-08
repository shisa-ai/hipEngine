# Qwen4Exp T16 Tile Residency — Design

Status: design (2026-10-09); implementation not started. This document is the
committed follow-on named by ledger rows T16-GU and T16-DOWN: *materialize the
T16 tiles at load, free the raw weights they replace, making the full window
fit*.

## Why

Both T16 candidates are gated behind X-7 with complete evidence chains, but
their **windows** are bounded by today's sidecar-cache integration:

- The gate/up A/B and gate ran a **30-of-46-layer** window (the gate used a
  declared 24-layer window): the full Q4_K gate/up tile set does not fit
  beside the resident raw weights in 120 GiB UMA.
- The down route's sidecar happened to fit all 43 Q5_1 layers (smaller set),
  but the same sidecar pattern is what bounded the gate/up window.

The bounded window is a measurement compromise, not a property of the design:
the tile sets are **bit-lossless rearrangements of the raw bytes**, so the
tile totals equal the raw totals exactly.

## Measured accounting (this model, UD-Q4_K_XL, 4-shard GGUF)

| Family | Raw (GiB) | T16 tiles (GiB) | Layers |
| --- | ---: | ---: | ---: |
| `ffn_gate_exps` + `ffn_up_exps` (Q4_K) | 42.38 | 42.38 | 48 (96 tensors) |
| `ffn_down_exps` Q5_1 | ~26.3 | ~26.3 | 43 |
| `ffn_down_exps` Q8_0 (no T16 route) | ~3.0 | — | 5 |
| **Combined routed experts** | **71.73** | **~68.7** | |

**The key property: materialize-tiles-and-free-the-raw is net-zero memory.**
Today's sidecar keeps both resident (raw 42.38 + tiles 42.38 = 84.76 GiB for
gate/up alone), which is what breaks the full window. Freeing the raw that
the tiles replace gives the full 46-layer gate/up window and the full
43-layer down window at exactly the footprint the model already occupies.

## Design

### 1. The tile stage on the weight binding

Each routed-expert weight binding (`expert_gate`, `expert_up`,
`expert_down`) grows an optional allocation stage `"t16"` — a
`DeviceTensorAllocation` holding the tile buffer — alongside today's
`"raw"`. The runner's T16 branches then check the binding, not a
module-level sidecar:

```
tiles = weights["expert_gate"].allocations.get("t16")   # binding-owned
```

This removes the `_T16_*_TILES` sidecar caches, their raw-pointer keys (which
freed raw pointers would invalidate), and their cross-session lifetime
hazards in one move. The sidecar populate scripts remain as diagnostics.

### 2. The load-path materialization (production profile)

In `materialize_qwen4_exp_weights`, after a routed-expert raw tensor is
loaded (and only when the production profile's T16 window selects the
layer):

1. read the raw tensor bytes (host);
2. `repack_gguf_q4_k_tile16` / `repack_gguf_q5_1_tile16` (host, NumPy);
3. upload the tile buffer, attach it as the binding's `"t16"` stage;
4. `allocation("raw").free()` and drop the `"raw"` stage from the binding.

Ordering: gate and up materialize before their raws are freed (the pair
shares the dual kernel's two tile buffers); down is independent.

### 3. The strict fallback contract

The binding-owned tile stage changes what "fallback" means when the raw is
freed:

- **Within the production profile**, a layer whose raw is freed has no
  in-memory strict chain. The registered strict fallback stays available two
  ways: (a) **the strict profile itself** — strict materializes raw, not
  tiles, so the exact chains remain one profile switch away (this is the
  architectural answer; EXECUTION-PROFILES already treats strict as the
  debugging oracle, not a production hot path); (b) **re-materialization
  from the GGUF on disk** — the same lazy pattern `Qwen4ExpPLEMMapTable`
  already models (one memmap owner, dequantize/repack on demand). (b) is the
  in-process escape hatch if a production run ever needs to fall back
  per-layer; it is slow by design and that is acceptable for a fallback.
- **Cache-miss fallback inside the runner branch** disappears for resident
  layers: the branch is keyed on the binding's own stage, so there is no
  miss state — the layer either has tiles (T16 route) or has raw (production
  chain). Layers outside the window keep raw and the production chain.

### 4. Window selection

The window is a production-profile constant (the same form as
`PRODUCTION_Q8_DENSE_WIDE_PREFILL_LAYERS`): all 48 MoE layers for gate/up
(where Q4_K holds), all 43 Q5_1 layers for down. Partial windows remain
selectable for A/B comparability with the committed 2026-10-08 evidence.

### 5. Admission sequencing

No numerical re-gating is implied by residency itself — the arithmetic is
the already-gated T16 arithmetic, and the full-window gate result is
expected to match the partial-window class (the gate/up gate passed every
global aggregate on its 24-layer window; the down gate passed every global
aggregate and category on its full 43-layer window). What must re-run if
X-7 admits the candidates:

1. the full-window engine A/B (the +6.53% gate/up figure was measured on
   30/46 layers; the full window's figure should be measured, not
   extrapolated);
2. the layer2 gate at the full window (the existing artifacts bound it);
3. the standard memory/capacity probes at the new peak (which is unchanged
   net, but the allocation *shape* changes: one tile buffer instead of one
   raw buffer per layer — fragmentation and staging must be re-probed).

Residency alone changes no default: it is the mechanism that makes the full
window *available*; the X-7 decision governs whether the T16 routes turn on.

## Non-goals

- No change to the T16 kernels, layouts, or their evidence chains.
- No change to the X-7 decision domain or the prefill_last threshold.
- No Q8_0-down T16 route (those five layers stay on the production chain).
- No decode-path change (the decode owners read raw layouts today; a decode
  T16 variant is a separate unit if the ledger ever ranks it).
