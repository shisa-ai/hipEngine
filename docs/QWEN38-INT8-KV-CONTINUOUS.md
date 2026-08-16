# Qwen3.8 INT8 KV Continuous-Batching Campaign

Status: **active; `IKV-C0` completed on 2026-08-16 and `IKV-C1` is next.**
Planning baseline was local commit `c791ca3c9`; merge commit `6cff90213`
integrated the 94 tracked `origin/main` commits before runtime implementation.
Artifact/backend/target capability identity now fails closed before compact
c>N work.

This campaign turns the retained Qwen3.8 c1 capacity result into an honest,
compact, no-BF16-mirror c>N serving route. It does **not** build a second
scheduler. The current shared GGUF owner already provides live admission,
chunked work selection, stable request/session/KV identity, cancellation hooks,
compaction, reclaim, request-budget-sized device-KV allocation, retryable pool
rejection, and detailed telemetry. Its no-mirror policy/resource work must plug
into the active Generation-2 ownership and global-arena interfaces in
[`CONCURRENCY2.md`](CONCURRENCY2.md) as those phases land. The missing feature is
packed no-mirror residency and block-table-aware prefill for multiple rows, a
direct row-batched INT8 decode consumer, complete memory admission, and
artifact-scoped quality promotion.

Related authorities:

- [`PLAN.md`](PLAN.md) — architecture and registry invariants.
- [`KVCACHE.md`](KVCACHE.md) — K1 storage, accuracy, and capacity evidence.
- [`CONCURRENCY2.md`](CONCURRENCY2.md) — active scheduler, resource-ledger, and
  global-KV ownership design.
- [`CONCURRENCY.md`](CONCURRENCY.md) — retained legacy resident/c>N evidence
  gates used during migration.
- [`KERNELS.md`](KERNELS.md) — kernel catalog, lineage, and port rules.
- [`TESTING.md`](TESTING.md) — RED/GREEN and CPU-reference gates.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence and anti-gaming contract.
- [`REFACTOR.md`](REFACTOR.md) — mirrored-route removal seam.
- [`Qwen3.8 compact evidence`](../benchmarks/results/2026-08-16-qwen38-27b-actual-context-quality-w7900.json).

---

## 1. Campaign objective

Deliver a production-shaped Qwen3.8 GGUF route that can keep multiple requests
resident with genuinely compact FP32-scale per-token/head INT8 K/V, consume
that INT8 K/V directly during decode, use only bounded transient exact-prefill
oracle ownership, admit or reject the complete request resource budget before
HIP OOM, and preserve exact resident lifecycle semantics.

The campaign has two closure levels:

1. **Explicit compact-capacity closure:** artifact/backend-qualified c2/c4
   no-mirror serving is correct, lifecycle-clean, and materially smaller than
   BF16. A declared serial c1-per-row model step is allowed at this level, but
   no throughput claim is allowed.
2. **Retained native-c>N closure:** row-batched INT8 attention runs under the
   expected kernel names, is non-regressive on the same suite, and replaces the
   temporary serial route for supported widths. Only this level may be promoted
   as efficient INT8 continuous batching.

BF16 remains supported/default throughout the campaign. A rejected
artifact/backend combination stays rejected even if another file with the same
model name passes.

## 2. Frozen baseline: what is already true

### 2.1 Artifact and backend identity

The passing local gfx1100 file is:

| Field | Value |
| --- | --- |
| Path | `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` |
| Size | `17,106,773,984` bytes |
| SHA-256 | `7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b` |
| Weight quant | `gguf_q4_k_m` |
| Attention shape | 24 Q heads / 4 KV heads / head dim 256 / 16 full-attention layers |
| Qualified backend | `hip_gfx1100` |
| KV candidate | uniform `int8_per_token_head`, FP32 per-token/head K/V scales |

The newer unintegrated `origin/main` gfx1151 campaign at commit `50a0390db`
tested a **different** file:

| Field | Value |
| --- | --- |
| Size | `17,106,775,008` bytes |
| SHA-256 | `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Backend | `hip_gfx1151` |
| Result | pure INT8 rejected at complete 1K/8; minimum-prompt top-1 `77.78%` |

The local pass does not override the gfx1151 rejection, and the gfx1151
rejection does not invalidate the local gfx1100 file. Runtime admission must be
bound to immutable artifact identity and backend evidence, never inferred from
filename, `Qwen3.8`, or 24Q/4KV/D256 geometry.

### 2.2 Retained local evidence

| Boundary | Measured result | Campaign meaning |
| --- | --- | --- |
| Complete quality | 512/8 and 4K/16 pass; complete 4K aggregate top-1 `99.47%` | Local artifact/backend candidate is numerically viable. |
| Bounded long quality | `mixed_v1` 129,024/16: mean/max KL `0.0000104/0.0001354`, 100% top-1, no mirror | Strong long control, not a complete long suite. |
| XTX c1 physical ceiling | 129,024 total tokens pass at `23.962624 GiB`; next pages stall; 130,048 OOM | 126K is a physical ceiling, not a service recommendation. |
| XTX service recommendation | four natural 112K requests pass at `23.322876 GiB` | Keep 112K as the repeated-natural c1 setting. |
| W7900 c1 | model-native 262,144 tokens pass at `29.441 GiB` | Direct c1 storage and attention are real. |
| W7900 short c>N | c2/c4/c8 pass 8K; all reject 8,448 in resident preparation | Scheduler works; direct no-mirror packed attention does not. |
| Controlled SSE | staggered c1->c4 is 4/4 exact; occupancy `0->1->4->3->2->1->0`; admitted/reclaimed `4/4`; zero final ownership | Short mirrored live admission/reclaim is closed on gfx1100. |
| Short INT8 storage | `25,296,896` bytes/page with BF16 mirrors versus `8,519,680` no-mirror | Current c>N path is not an INT8 memory saving. |

### 2.3 Existing implementation that must be reused

- `SubmitPollTextGenerator -> ResidentEngineLoop ->
  Qwen35GGUFResidentModelRunner` is the production owner.
- `reserve_admission()` allocates the complete prompt-plus-output KV page budget
  before scheduler slot publication and releases it on reclaim.
- `DeviceChunkedKVPool` grows/shrinks real backing chunks, supports shared
  prefixes/refcounts, and emits retryable high-water rejection.
- `KVLiveSpans` already carries per-row base offsets, live counts, token
  positions, block tables, eviction masks, and scale metadata.
- Batch INT8 writers already exist.
- Direct 24Q/4KV/D256 c1 INT8 split-K attention is CPU-reference and
  `rocprofv3` gated.
- The resident owner already has an honest `_step_native_serial()` fallback and
  route/fallback telemetry.
- Layer-outer exact c1 prefill already reuses one BF16 oracle pair and releases
  it before decode.

## 3. Exact missing boundaries

### 3.1 Packed no-mirror admission is deliberately blocked

The following fail-closed seams are current behavior, not accidental test
failures:

- `Qwen35GGUFResidentModelRunner._reserve_sessions()` validates every c>N
  session pool through `_packed_ar_kv_layout_for_sessions()`.
- `_packed_ar_kv_layout_for_sessions()` rejects any INT8 layer without a BF16
  mirror.
- `_packed_full_attention_scratch_for_layer()` likewise rejects direct INT8
  packed prefill/decode scratch without a mirror.
- Shifted dynamic-pool allocations use block-table-aware packed prefill, so
  merely sending decode through `_step_native_serial()` is insufficient.

The first compact serial milestone therefore needs a correct single-row
block-table-aware no-mirror prefill path as well as c1 decode dispatch.

### 3.2 Direct INT8 attention is c1-shaped

`qwen35_paged_attn_decode_int8_gqa_splitk_gate_bf16_spans()` has no `rows`
parameter and its reducer is one-row shaped. BF16 packed AR has row-batched
context/split-K variants; INT8 does not. The batch writer and `KVLiveSpans` ABI
do not make the consumer batch-capable by themselves.

### 3.3 Admission accounts for pages, not the complete request peak

Current pool reservation covers persistent KV pages but does not atomically
budget every possible owner:

- INT8 payload and scale planes;
- BF16 mirrors when the fallback route is selected;
- packed row workspace and split-K partials;
- full hidden and transient prefill-oracle ownership;
- graph-pinned pages/workspaces;
- a configured whole-device reserve.

This is why retryable 429 pool rejection exists while some XTX c>N shapes can
still reach HIP OOM outside the pool estimate.

### 3.4 The allocator intentionally does not extend requests across chunks

`DeviceChunkedKVPool` requires one request's pages to remain in one backing
chunk because attention uses one base pointer plus an int32 block table. There
is no request-extension API, and arbitrary late growth cannot guarantee
contiguity.

Per-token page growth is **not** a campaign prerequisite. Keep request-budget-
sized upfront reservation. If future elastic request extension becomes a goal,
it requires either capacity credits backed by one reserved chunk or a
pointer-table/chunk-aware attention ABI as a separate architecture campaign.

## 4. Non-goals

- Do not rewrite the resident scheduler or replace it with
  `_generate_greedy_batch`.
- Do not claim generic Qwen3.8 support from one artifact/backend combination.
- Do not add filename, backend, quant, or model-name branches to engine/model
  hot paths; use registered capabilities and kernel variants.
- Do not implement per-token KV allocation growth.
- Do not reopen prompt-selected layer maps, token-conditioned branches, recent
  tails, clipping, block16, or Hadamard families that failed their frozen gates
  unless a materially new input-independent representation signal is approved.
- Do not combine MTP, graph promotion, prefix-cache promotion, DMS, or c>8
  native kernels with the initial compact c>N milestone.
- Do not count a BF16 mirror route as an INT8 memory saving.

## 5. Milestone plan

### IKV-C0 — integrate source and lock capability identity

**Status: completed 2026-08-16.**

**Purpose:** prevent incompatible Qwen3.8 artifacts/backends from inheriting
one another's quality decision.

Retained implementation:

- Merge commit `6cff90213` preserves both campaign histories and all focused
  mapping/MTP/server/geometry/attention tests.
- `hipengine.models.kv_capabilities` computes a demand-driven full-file SHA-256
  and resolves a complete immutable key through `Qwen35GGUFModel` evidence.
- The exact `7b2aec...` gfx1100 FP32-scale contract is qualified; the distinct
  `7e78da...` gfx1151 contract remains rejected; unknown artifacts or contract
  mismatches fall back to BF16.
- `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED=1` and the historical
  `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1` permit explicitly labeled,
  non-promotable diagnostics without changing the evidence decision.
- `/ready`, `/v1/models`, and `/v1/hipengine/capabilities` expose capability ID,
  full artifact identity, requested/effective storage, evidence artifact,
  runtime action, diagnostic override, and promotion eligibility.

Original work contract:

1. Integrate the local campaign commits with the 94 tracked `origin/main`
   commits before editing high-conflict runtime/kernel files.
2. Preserve both Qwen3.8 quality artifacts and their exact model identities.
3. Define a model-plugin capability record keyed by immutable content
   fingerprint, backend/target, weight quant, KV layout, scale dtype/granularity,
   and quality artifact.
4. Make unsupported or unknown combinations fail closed to BF16/default; keep
   explicit diagnostic overrides clearly labeled and non-promotable.
5. Expose the resolved capability identity in readiness/artifact provenance.

Gate:

- The local `7b2aec...` gfx1100 candidate resolves to its explicit INT8
  capability.
- The `7e78da...` gfx1151 candidate remains rejected.
- Same filename or geometry with an unknown fingerprint does not inherit either
  result.
- No engine/dispatch backend or quant branch is added.

Stop rule: no compact c>N implementation begins until this gate passes after
integration.

### IKV-C1 — compact serial c>N correctness milestone

**Purpose:** isolate no-mirror storage, shifted block-table prefill, allocator,
and resident lifecycle from new batched attention math.

Work:

1. Permit multiple artifact-qualified no-mirror sessions without claiming
   packed native decode.
2. Add a single-row block-table-aware no-mirror prefill route for shifted
   device-pool allocations.
3. Execute each active row through the existing direct c1 INT8 attention via
   `_step_native_serial()`.
4. Record `kv_attention_source=int8_direct`, logical C, physical width 1, and
   `serial_decode_fallback=true`.
5. Preserve stable request/session/KV identity across admission, retirement,
   compaction, cancellation, and survivor continuation.

Required evidence:

- c2 and c4 no-mirror layout audits with zero persistent BF16 bytes.
- Exact independent-c1 token/logit/state/KV comparison.
- Staggered SSE admission, cancellation, reclaim, and zero final ownership.
- Request-budget-sized pool allocations remain one backing chunk each.
- No performance or native-c>N claim.

Removal rule: this route is a temporary correctness fallback. Remove or demote
it after IKV-C2 passes the same suite non-regressively; retain only an explicit
registered numerical fallback when required by the fused-kernel fallback
contract.

### IKV-C2 — row-batched direct INT8 split-K attention

**Purpose:** provide the missing efficient packed c>N consumer.

Kernel/runtime work:

1. Add a registered row-batched per-token/head INT8 GQA split-K producer for
   24Q/4KV/D256 with row-shaped `KVLiveSpans`.
2. Add a row-batched gated reducer with explicit row/gate/output strides.
3. Keep c1 wrappers and unfused/numerical fallbacks.
4. Wire the new variant into
   `_run_full_attention_decode_batch_layer_rows()` without backend/quant
   branches in engine/model dispatch.
5. Emit truthful native packed versus serial-fallback manifests.

Primitive gate:

- c1/c2/c4/c8 against `kernels/cpu_reference/`.
- Ragged contexts, sparse active masks, non-zero base rows, page boundaries,
  and mixed live counts.
- KL <= `0.05` and top-1 >= `90%` on fixture logits.
- Bit/exact comparison against independently run c1 sessions where the runtime
  contract requires exactness.
- `rocprofv3 --kernel-trace` proving the expected producer and reducer names,
  plausible durations, and intended row geometry.

Stop rule: a kernel that requires prompt-specific selection, loses the
correctness gate, or reads BF16 mirrors is rejected regardless of speed.

### IKV-C3 — global prefill ownership and workspace plan

**Purpose:** prevent N long-context sessions from replicating enough hidden/
oracle scratch to erase the KV saving.

Work:

1. Model persistent KV/scales, active decode, pending prefill, packed split-K,
   full hidden, oracle, and graph ownership together.
2. Prefer one scheduler-owned/shared prefill workspace when ownership and
   synchronization permit it.
3. Preserve layer-outer exact prefill by either pausing decode for a bounded
   layer phase or yielding at layer boundaries; do not silently revert to N
   full-length oracle pairs.
4. Keep a numerically equivalent unfused/chunk-outer fallback.
5. Emit the selected prefill lifetime/workspace plan and its owned bytes.

Gate:

- c2/c4 modeled and measured peak is lower than matched BF16.
- No concurrent session can overwrite another request's hidden/oracle/KV.
- Fairness/TTFT impact is recorded; a memory win cannot hide unbounded decode
  starvation.

### IKV-C4 — complete admission accounting and overload behavior

**Purpose:** reject requests before untracked allocations reach HIP OOM.

The admission estimate must include:

```text
persistent INT8 K/V
+ K/V scale planes
+ selected BF16 mirror bytes (normally zero for this campaign)
+ packed attention workspace
+ prefill hidden/oracle workspace share
+ graph-pinned ownership
+ configured whole-device reserve
```

Work:

- Bind the estimate to the selected capability and prefill/workspace plan.
- Preserve request-budget-sized upfront page allocation.
- Reject before allocation with the existing retryable public overload
  taxonomy; 429 versus 503 is a separate API policy decision.
- Report estimated versus measured bytes and the rejecting resource.

Gate:

- Controlled pressure tests return exact retryable rejects, never HTTP 500 or
  HIP OOM.
- Recovery accepts a later request after reclaim.
- Pool, graph, workspace, and session owners drain to zero.

### IKV-C5 — lifecycle, elastic-pool, and telemetry closure

Required lifecycle matrix:

- simultaneous and staggered c1/c2/c4/c8 arrivals;
- c1->c4 join during observed decode;
- mixed prompt lengths and chunked prefill/decode fairness;
- cancellation before admission, during prefill, and during decode;
- c4->c3->c2->c1 survivor compaction with independent-c1 state/KV equality;
- default pool grow, idle shrink, graph invalidation/regrow, overload, recovery,
  shutdown, and zero final ownership.

Required explicit telemetry:

- `kv_attention_source = int8_direct | bf16_mirror`;
- logical active C and physical execution width;
- native packed versus serial fallback;
- prefill lifetime/workspace plan;
- persistent INT8, scale, and mirror bytes;
- admission-estimated and measured current/peak bytes;
- admitted/reclaimed/rejected/cancelled counters and active mask.

A field must describe the loaded engine/capability, not a hardcoded endpoint
default.

### IKV-C6 — artifact-scoped full quality and capacity qualification

Run in this order for every candidate artifact/backend; stop at the first
required failure:

1. complete 512/8 BF16-teacher-forced suite;
2. complete 1K/8 transfer when another backend/artifact has failed there;
3. complete 4K/16 suite;
4. bounded 32K/16 and 64K/16 controls;
5. complete multi-category long-context suite at the largest practical shared
   shape;
6. category-heldouts and natural server prompts;
7. XTX and W7900 c1/c2/c4/c8 capacity brackets with actual failure
   classification.

Every row records exact model fingerprint, backend, weight quant, KV layout,
scale policy, BF16 mirror bytes, intended kernel ownership, and final resource
ownership. No candidate may be selected from the evaluation prompts.

### IKV-C7 — economics, promotion, and cleanup

Only after C0-C6 pass:

- Compare direct INT8 c1/c2/c4/c8 against matched BF16 and the temporary serial
  compact route on the same hardware/protocol.
- Report prefill, decode, SSE goodput/SLO, complete wall, launch families,
  persistent/current/peak bytes, and failure/recovery behavior.
- A capacity-only explicit route may be retained if it is correct and materially
  smaller but slower; it is not called efficient or promoted to default.
- Native-c>N promotion requires same-suite exactness, lower memory, and
  non-regressive retained performance with all lifecycle gates passing.
- Remove the mirrored short route when no supported caller needs it; preserve a
  separately registered unfused/numerical fallback.
- Update `docs/REFACTOR.md`, benchmark rollups, artifacts, and immutable
  worklogs with accepted and rejected outcomes.

## 6. Validation matrix

| Level | Required shapes/evidence |
| --- | --- |
| Policy/capability | passing fingerprint, rejected fingerprint, unknown fingerprint, backend mismatch, quant/layout/scale mismatch |
| Writer/attention primitive | rows 1/2/4/8; contexts 1/255/256/257/1K/8K+; ragged live counts; sparse masks; non-zero base rows |
| Direct model | independent-c1 versus serial c>N versus packed c>N token/logit/hidden/state/KV equality |
| Quality | complete 512/1K/4K, bounded long controls, complete long categories where practical, heldouts |
| Memory | exact payload/scale/mirror/workspace bytes; tracked and whole-device current/peak; no shadow |
| Lifecycle | stagger, chunked prefill fairness, cancellation, survivor compaction, reclaim, grow/shrink, overload/recovery, shutdown |
| Kernel ownership | cached build plus `rocprofv3` producer/reducer trace under expected names |
| Performance | repeated same-session BF16/serial/direct controls; no single-run or profiler-topline claim |

New HIP tests require an explicit ROCm availability guard. After an isolated
failure in a completed broad run, follow the focused-repair rule in
[`TESTING.md`](TESTING.md) rather than automatically repeating the broad suite.

## 7. Campaign stop rules

- **Quality failure:** stop that artifact/backend/layout at the first required
  failed shape. Do not tune to the failing prompt or transfer a pass from
  another file/backend.
- **Mirror detected:** classify the row as mirrored fallback, not compact INT8.
- **Wrong kernel:** treat it as registry/dispatch failure before editing math.
- **OOM outside estimate:** stop throughput work and repair admission ownership.
- **Memory-negative c>N:** repair workspace/prefill ownership before promotion.
- **Serial fallback:** retain only as explicit correctness evidence; never
  headline its aggregate throughput as native batching.
- **Metric win with correctness/lifecycle failure:** reject and revert.
- **Branch divergence:** integrate current `origin/main` and rerun affected
  gates before retaining a high-conflict runtime/kernel unit.

## 8. Execution ledger

| Milestone | State | Dependency | Exit |
| --- | --- | --- | --- |
| IKV-C0 integration + capability identity | `completed` | approved campaign | passing/rejected/unknown identities resolve correctly |
| IKV-C1 compact serial c>N | `ready` | C0 | no-mirror c2/c4 lifecycle exact |
| IKV-C2 row-batched INT8 attention | `blocked` | C0, C1 oracle | CPU/model/trace gates pass |
| IKV-C3 shared prefill ownership | `blocked` | C1/C2 measured ownership | c2/c4 remains memory-positive |
| IKV-C4 complete admission | `blocked` | C3 byte plan | pressure rejects before OOM and recovers |
| IKV-C5 lifecycle/telemetry | `blocked` | C2-C4 | cancellation/grow/shrink/overload matrix passes |
| IKV-C6 quality/capacity | `blocked` | C2-C5 | artifact-scoped complete gates pass |
| IKV-C7 economics/promotion | `blocked` | C6 | retained decision and cleanup published |

The next executable unit is **IKV-C1**: compact no-mirror c2/c4 through the
declared serial c1-per-row correctness route, including shifted block-table-aware
prefill. It makes no native-c>N throughput claim.
