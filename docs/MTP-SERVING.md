---
status: current
owns: The what-works-where MTP serving policy matrix across models, backends, sampling modes, concurrency widths, and automatic-versus-explicit admission, plus the per-request force semantics.
---

# MTP serving policy matrix

This page is the current, complete answer to "what MTP works where" — per
model, backend, sampling mode, concurrency width, and request mode — and the
per-request force semantics. It reflects the retained declarations and
evidence rows in `hipengine/models/*.py` and the resolver in
`hipengine/speculative/serving.py`; reprofiling updates land here as
measured (gfx1151 first, gfx1100 separately, with no cross-arch transfer).

## How a request is admitted

Three inputs decide, in order:

1. **Request mode.** `"speculative_mtp": false` always forces AR. `true` is an
   *explicit* request: it must be admitted by capability or fail with a named
   reason — it is never silently downgraded. Omitted leaves the *automatic*
   policy to decide.
2. **Model-plugin scope.** `speculative_mtp_serving_implementations` declares
   what the kernels implement (backend, KV storage, sampling modes, max depth
   per width). Explicit requests may admit on a declaration alone — an
   implementation contract is not a performance claim. `speculative_mtp_serving_evidence`
   rows are *retained measured scopes*; automatic admission requires an
   `automatic_eligible` row (or automatic-eligible declaration) matching the
   cell. An evidence row never withholds a declared implementation from an
   explicit request.
3. **Physical cell.** The backend package's cell table and the resolved
   capability bound the (width, depth) pair the adapter will actually run; a
   cell nothing lists falls to whole-group AR rather than running unmeasured
   work. Resources, context room, graph availability, and the circuit breaker
   are checked before mutation.

An admitted scope always has the exact strict AR fallback; circuit-breaker or
operator rollback resolves to AR and the response reports the substitution
(`mtp_circuit_breaker_open` / `mtp_operator_rollback`) instead of hiding it.

## Per-request force semantics

| Request | Behavior |
| --- | --- |
| `speculative_mtp: false` | AR, always, every model |
| `speculative_mtp: true` | Explicit: admitted by declaration or evidence for the loaded artifact's cell, or HTTP 501 naming the missing capability; never a silent downgrade |
| omitted | Automatic policy: only `automatic_eligible` scopes matching the cell; otherwise AR |
| depth (`speculative_mtp_num_tokens`) | Explicit requests may ask up to the declared/evidenced maximum; deeper is refused by name, not clamped silently |
| width | The due group forms only within the declaration's `max_group_rows` and the backend's listed cells; a wider explicit group is refused by name (`*_group_above_offered_width` or the physical-bucket reason) |
| breaker/rollback open | AR with the named substitution reason |

PARO supports a single resident request with one draft candidate, greedy
sampling, and BF16 KV. Sampled requests and wider groups are refused by name.

## The matrix (current head)

### Dense Qwen (Qwen3.6 / Qwen3.8 GGUF, `Qwen35GGUFModel`)

| Backend | KV | Width/depth (declared) | Sampling | Automatic today | Explicit today |
| --- | --- | --- | --- | --- | --- |
| gfx1100 | bf16 | C1 K≤7; group C2–C8 K≤3 | greedy_fast, sampled | Qwen3.6 Q4_K_M: C1/K3, C2/K2 (dual-promotion rows); Qwen3.8: **no automatic cells** (withdrawn, see below) | all declared cells, incl. Qwen3.8 C2/K2–K3 and C8/K3 rows |
| gfx1100 | int8_per_token_head | C1 K≤7; group C2–C4 K≤3 | greedy_fast, sampled | declaration-scoped (no rows of its own) | yes |
| gfx1151 | bf16 | C1 K≤7; group C2–C4 K≤3 | greedy_fast, sampled | Qwen3.8 strict C1/K3 (natural25, 1.6445× AR) and cap4-realized C1; **sampled C1–C4/K3** | yes, incl. production C1/C2 rows (explicit-only) |
| gfx1151 | int8_per_token_head | C1 K≤7; group C2–C4 K≤3 | greedy_fast, sampled | declaration-scoped | yes |

Sampled admission on dense is storage-blind and keyed on the artifact's
weight quant through `sampled_route_qualified_for`; the device accept samples
from the target's own law over verifier row logits.

**Qwen3.8 gfx1100 automatic withdrawal (evidence-referenced).** The
better-MTP campaign measured every retained cell losing to current AR, so the
fingerprinted automatic policy selects K0 (plain AR) for Qwen3.8 on gfx1100:
all 20 measured C1–C8 cells miss the win threshold, the C2/K2, C2/K3 and
C8/K3 rows stay explicit-only, and W7900 C1 evidence was withdrawn (its runs
used the legacy singleton target). This is the decision the gfx1100 reprofile
task re-validates on current head — see
[`QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md`](campaigns/QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md)
and the S5/S6 closure in [`SPECDEC2.md`](reference/SPECDEC2.md). gfx1151 is
governed separately and retains its automatic keys above.

### MoE Qwen (Qwen3.6 GGUF, `Qwen35MoeGGUFModel`)

| Backend | KV | Width/depth | Sampling | Automatic today | Explicit today |
| --- | --- | --- | --- | --- | --- |
| gfx1100 | bf16 | C1–C2, K≤2 | greedy_fast, sampled | Q4_K_M C1/K2 and C2/K2 (dual-promotion rows) | yes; **sampled resolves at c1–c2** (c1 via the native-cycle device accept, c2 via the dense adapter's packed per-row sampled accept), c3+ refused by name (`moe_group_above_offered_width`) |
| gfx1151 | bf16 | not declared | — | no (no declaration, no evidence) | refused by name until the capability is established — not assumed |

Sampled MTP at c=N runs through the dense adapter that serves MoE groups at
capacity > 1: the packed target asks for each row's logits span and draws
per-row tokens with the same sorted sampler as AR **before any commit**
(`_sample_packed_target_rows`), with per-prefix state for history processors
and a host fallback. A sampled row reaching a lane that cannot decide it
raises by name (`MoeSampledRouteUnsupportedError` on the C1 adapter's own
width-2 lane; `sampled physical verification requires resident device
results` on the host-batch lane) and is contained, never silently decided by
argmax. The c2 sampled cell's GPU qualification on gfx1100 is the recorded
gate for promoting it beyond explicit-only.

### PARO (W4, `Qwen35ParoMoeModel`)

| Backend | Width/depth | Sampling | Automatic | Explicit today |
| --- | --- | --- | --- | --- |
| gfx1100 | C1, K=1 | greedy only | declaration-scoped (no evidence rows yet) | **resolvable through the OpenAI route** via the `paro_w4_gfx1100_c1_native_chain` declaration; sampled is refused by name (`mtp_sampling_unsupported`) until the decider lands |
| gfx1151 | C1, K=1 | greedy only | declaration-scoped | yes, via `paro_w4_gfx1151_c1_native_chain`; sampled is refused by name (`mtp_sampling_unsupported`) |

Both backends use the shared staged adapter. The gfx1151 public-generation
checks confirm real speculative cycles, repeatable outputs, and cleanup.
The existing bulk verifier can produce different greedy trajectories from
ordinary decoding; cross-route token identity is not promised. See the
[adapter validation record](../benchmarks/results/2026-10-23-gfx1151-paro-mtp-adapter/validation.json)
for the comparisons and numerical diagnostics. This repair does not change
kernel arithmetic or publish an MTP throughput result.

### Gemma 4 (assistant-head MTP)

| Backend | Width/depth | Sampling | Automatic | Explicit |
| --- | --- | --- | --- | --- |
| gfx1100 | C1, K≤8 | greedy_fast, sampled | yes (declaration, W7900 evidence) | yes |
| gfx1151 | C1, K≤8 | greedy_fast, sampled | yes (declaration) | yes |

Gemma 4 serves through the whole-request assistant-head provider
(`gemma4_mtp.py`), not the staged adapter seam; both blocking and streaming
routes reach the provider and refuse by name when unconfigured.

### Compact DMS

Speculative serving is refused by name with its clearing setting
(`speculative_mtp_serving='off'`); the config gate is the recorded boundary,
not a missing implementation. Dense INT8 and BF16 KV are unaffected.

## Recorded gaps (each with an owner)

These are the cells between "declared and working" and "every model, every
common scenario" — each is in flight, and this section shrinks as they land:

- ~~**PARO explicit through the server**~~ — **closed 2026-10-16**: the
  plugin now declares the C1 scope and the generator exposes the resolver, so
  an explicit PARO request resolves through the OpenAI route (greedy, gfx1100)
  and every out-of-scope axis is refused by name.
- **PARO sampled MTP** — the declaration stays greedy-only until the sampled
  decider is real: the device sampled accept (the shared native-spec
  `sampled_accept`/workspace machinery the dense/MoE lanes use) ports behind a
  graph-warmup gate. The shared adapter runs on gfx1100 and gfx1151; a sampled
  decider would need validation on each backend. Owner: the PARO sampled-decider unit.
- **MoE sampled at c=N** — closed as misrecorded: the production c2 lane
  (dense adapter) already decides sampled rows exactly per-row before commit;
  the "C1-only" reading came from the C1 adapter's own width-2 lane, which
  production never routes to. Remaining: GPU qualification of the c2 sampled
  cell on gfx1100 before it is promoted beyond explicit-only.
- **C8 speculative owner** — no qualified C8-width speculative frontier
  exists on gfx1151, so explicit MTP at concurrency 8 has no physical path
  there. Current-head ladder (2026-10-16, 8060S, UD-Q4_K_M): AR c1 9.95 /
  c4 11.05 / **c8 11.13 tok/s**, with the M5 production rule routing c5-c8
  due batches to whole-batch AR before any priming - confirmed on current
  head. No measurable C8 MTP path exists to evaluate (no native R16/R32
  bucket; the legacy arm serves c1 only), so qualification is a kernel
  campaign: develop the native bucket + oracle, then beat both the sweep
  decomposition and the 11.13 baseline under the full S5/S6 gates. The
  1.741x c1 MTP win on the same hardware is the campaign's opening signal.
  gfx1100 is separate: its explicit-only C8/K3 row keeps explicit MTP at c8
  forceable there. Owner: the C8 campaign.
- **MoE on gfx1151** — no declaration or evidence; explicit stays refused by
  name until measured, per the no-transfer rule.

## Reprofile status

- **gfx1151**: reprofiled 2026-10-16 on the current head (Radeon 8060S;
  artifacts in `benchmarks/results/2026-10-16-gfx1151-qwen38-udq4km-*`).
  Measured: whole-request native MTP at C1/K3 on the host-resident
  Qwen3.8-27B-UD-Q4_K_M runs **1.741x true AR** (17.32 vs 9.95 tok/s, greedy
  IDs exact 30/30) — but automatic still selects AR for that artifact because
  it has no gfx1151 evidence row (its plugin rows are gfx1100-scoped), and
  the retained strict rows cannot engage because the gfx1151 cell table lists
  production widths/depths only. The skip currently in force is
  evidence-absence, not a measured loss. Follow-ups: a production-C1/K3
  promotion packet for the resident UD artifact, and a strict cell-table
  decision (add the entry or withdraw the strict rows).
- **gfx1100**: separate reprofile on the W7900 host follows; no gfx1151 rate,
  threshold, or fingerprint transfers.

Evidence artifacts for every retained row live under `benchmarks/results/`
(cited per row in `hipengine/models/qwen35.py` and `gemma4.py`); the per-arch
campaign ledgers (`SPECDEC2.md`, `SPECDEC2-PERF*.md`,
`MTP-CONCURRENCY2-DUAL-PROMOTION.md`) own the measurement history.
