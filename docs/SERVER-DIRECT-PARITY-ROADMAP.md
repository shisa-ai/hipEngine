# Server/direct parity: review and implementation roadmap

Review date: 2026-09-10. Reviewed source: `ff217c9d1`, including the oracle
repair, its regression tests, and the dead KV-import removal. This is a
code/evidence review and proposed execution order, not new GPU qualification.

Amended 2026-09-10 with a follow-up read-only code trace at the same base
(`be99b8cce` is docs-only over `ff217c9d1`, so the runtime is unchanged). The
amendment answers this review's open direct-arm question in F1, names the
second AOTriton admission site, and records the slab-count and oracle-lifetime
mechanisms in F3/F5. It adds no measurement.

Primary scope: Qwen3.8-27B Q4_K_M, gfx1100, dense BF16 and uniform
INT8-per-token/head KV with FP32 scales, autoregressive serving. The immediate
problem is the compact INT8 server route. Shared service, scheduler, accounting,
sampling, and prefix boundaries were also reviewed. Other models/backends,
MTP, DMS, tool-specific HTTP semantics, and security are not certified here.

The architecture and numerical authorities remain [PLAN.md](PLAN.md),
[EXECUTION-PROFILES.md](EXECUTION-PROFILES.md), and [TESTING.md](TESTING.md).
This roadmap complements the [capacity campaign](QWEN38-27B-GFX1100-24GB-CAPACITY.md)
and [INT8 continuous-batching campaign](QWEN38-INT8-KV-CONTINUOUS.md); it does
not authorize another scheduler, an unqualified codec, or a profile change.

## 1. Executive conclusion

The server is not simply the direct engine plus HTTP. Its prefill executor,
KV workspace reservation, physical address space, and scheduling boundaries
are different. There are three separate problems:

1. **Prefill compute:** a measured slow attention kernel is selected on the
   repaired compact INT8 packed path. Reusing the direct path's admitted
   attention implementation is the first speed task.
2. **Memory:** the server reserves a second context-sized KV region and uses
   per-layer prefill oracles sized against the whole pool. Layer-outer execution
   fixes the oracle lifetime, but removing the redundant reservation and
   reconciling address spaces are also needed for the direct memory slope.
3. **Serving/decode:** C1 already uses the direct session step. Remaining C1
   overhead needs a matched measurement, not a replacement decoder. Compact
   INT8 C>1 still needs the independently planned row-batched consumer.
   Long monolithic prefill also blocks the sole service driver.

The target is one set of admitted compute primitives and ownership contracts
used through multiple surfaces. Exact HTTP wall-time equality is not a useful
promise: transport, token handling, and fairness have costs. Equal KV geometry,
no redundant context-proportional storage, and near-direct model-step time are
reasonable requirements.

## 1.1 Execution status (2026-09-10, end of day)

Status levels are distinguished per the review: implementation landed /
diagnostic passed / qualified. P0 (telemetry) and P4 (lease removal,
allocator corrected) are landed; P1-P3 have diagnostics passed with packet
gates outstanding; P2 is an observation harness, not an admission preflight;
P5's C1 baseline is landed and P7's capacity bracket is tier-1
allocation-validity only. Stage table (W7900 GPU0, 27B Q4_K_M,
int8_per_token_head + FP32 scales, C1, real server route, 2,048-row prompt
unless noted):

| Stage | Packed prefill | Decode boundary | Server-route transients @16K/32K declared | Commit |
| --- | ---: | ---: | --- | --- |
| Session start | 67 tok/s | unmeasured | unmeasured | - |
| P1 slot-local AOTriton | 677 tok/s (10.1x) | - | unmeasured | `04b2fccf5` |
| P0 telemetry freeze | 677 | - | oracle 2.0/4.0 GiB (16 owners); lease 0.51/1.02; workspace 0.95; hidden 0.27/0.43; pool high-water 0.58/1.09 | `664298f84` |
| P2 server-faithful probe | 677 | - | same, now measured while-live on the real route (in-process oracle peak capture) | `5a49e5bbe` |
| P3 layer-outer executor | 732 | - | oracle 0.125/0.25 GiB (1 owner); bitwise parity vs scalar at 2,048 and 1,500 rows | `117ca5cab` |
| P4 lease removal | 677-732 | - | lease 0/0; pinned 0/0; pool high-water 0.07; chunk-outer oracle halves to 1.0 GiB (pool-backing coupling); determinism verified | `0aded03b1` |
| P5 t1 decode ladder | - | R0 35.15 ms; packed-entry R1 35.22 ms (ratio 1.002, private sessions - executor-entry cost only, not server pool/scheduler execution) | - | `2cc99ad75` |
| P7 t1 capacity | - | - | tier-1 allocation-validity bracket passes 65,536-155,648 DECLARED contexts (one 2,048-row request each; not full-length completions; no cross-card direct comparison) | `c7eb76b88` |
| P3 promotion | - | - | trace identity passed (AOTriton + native, as predicted); default reverted to OFF pending the packet gates (reviewer F4) | `bf2267029` + corrective |
| P7 t2 publish | - | - | paired reps: ratio median 1.004 (one matched 8K diagnostic; cross-rep variance 89-104% under shared-host contention, so 'consistently within 5%' is not established); SSE observation is delivery cadence, not isolated transport cost; kernel-family attribution landed | `9e28487cb`, `131d4219d` |
| P6a/P6c resumable prefill | - | per-poll layer segment; checkpoint `next_layer` + derived ping-pong phase; oracle released at every segment boundary | - | `35b61413b` |
| P6b suspended-state ownership | - | - | dedicated hidden-plane + linear-state buffers copied out at each yield and back on resume; suspended owner counted in prefill-transient telemetry; freed on completion, failure, and row reclaim | `--` (this unit) |
| P6d harness repair | - | - | - | `--` (this unit) |
| P6e service proof | - | bounded command acknowledgement and cancel delay while a long prefill is in flight; survivor text byte-identical over 192 characters | - | `f6578280a`, `53f134a7e`, `1a663ef9a` |
| P6f layer-boundary state gate | - | GPU proof: the segmented resumable prefill's committed per-layer direct INT8 K/V, its scales, and the linear conv/recurrent state fingerprint identically to the one-shot reference over all 64 layers at 3,072 committed rows, and again at 3,172 rows with a short 100-row final round; the gate's first run exposed an uninitialized-scale-tail comparison defect in the harness itself | - | `dfd31004d` + ragged follow-up |

Combined P3+P4 at 32K declared: ~1.6 GiB of route transients vs ~6.4 GiB at
the P0 baseline (-75%), measured with the executor flag enabled. The
layer-outer executor's diagnostics all passed (bitwise parity at 2,048/1,500
rows, wall A/B at 1K/2K/4K/8K within 2%, trace identity, server probe,
decode handoff) and cancellation cleanup is CPU-tested, but the remaining
packet gates are outstanding (aliasing on this executor; the
layer-boundary/state, exact KV/control, and reachable ragged-final-round gates
were closed on GPU 2026-09-11 by `scripts/gguf_resumable_prefill_gpu_proof.py`,
whose `layer_boundary_state` gate also passes on a ragged prompt - multi-slot
unequal prompts decline by the executor's slot-stability guard), so the default
is OFF (reviewer finding 4);
`HIPENGINE_GGUF_PACKED_LAYER_OUTER=1` enables it. The lease removal is default
on for the C1/prefix-off/MTP-off route with `HIPENGINE_GGUF_PACKED_KV_LEASE=1`
as the rollback.

Remaining open packets: P6 (P6a/P6b/P6c landed opt-in - the resumable layer-outer
checkpoint, its suspended-state ownership, and the CPU tests; P6e GPU proof on
the serial INT8 route **landed 2026-09-11 and now passes on the resumable route
itself** (eleven gates - ten passing and the native-sampling gate explicitly
`skipped` with its blocker named - including a cancellation-delay sweep and a
host-sampling arm; see the coverage table in the P6 section), and P6f's
layer-boundary/state, exact KV/control, and reachable ragged-final-round gates
closed on GPU 2026-09-11 while aliasing on this executor remains outstanding, so
the route stays behind
`HIPENGINE_GGUF_PACKED_LAYER_OUTER` until P6f closes), P5
remainder (HIP-API/queue-gap attribution and graph-capture amortization; the
IKV-C2 C>1 row-batched consumer landed and is promoted to physical c4 on the
W7900 gfx1100 Qwen3.8-27B INT8 artifact as of 2026-09-11, so the capacity>1
lease gate is no longer waiting on it), and P7 remainder (the C>1 and P6-gated
acceptance items). The C-width
baseline harness is repaired (per-lane measured counts, measured model-step and
route/fallback deltas, a real same-server serial control, no decode-only rate)
and has now been run live before and after the IKV-C2 promotion. The C1-scoped
publish is landed: paired speed
parity, decode boundary, memory exactness, capacity ceiling, trace identity,
and the HTTP/SSE transport budget.

## 2. Findings, ordered by impact

### F1 - P1: long packed INT8 prefill selects the wrong performance regime

**Measured attribution:** the existing W7900 8,192-row, 16,384-capacity trace
attributes 110.989 of 121.385 seconds of summed prefill kernel time to
`qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel`. The copy-removal A/B
has effectively unchanged wall time and identical final logits/continuations.
See the [attribution artifact](../benchmarks/results/2026-09-10-w7900-int8-packed-paged-attn-attribution.json)
and [removal artifact](../benchmarks/results/2026-09-10-w7900-int8-packed-kv-import-removal.json).
Those are existing diagnostic measurements, not a new server throughput claim.

**Code:** `Qwen35GGUFResidentModelRunner._prefill_native_chunk()` buffers
`int8_direct` work into `_prefill_native_row()`, which invokes
`prefill_batch_native()`. Slot-local full attention calls the common
`_run_full_attention_prefill_layer_aotriton()`, but
`_gguf_slot_local_prefill_allow_aotriton()` disables AOTriton on transient
oracle layers unless the new flag is enabled.

The native kernel walks history per query/head. Its launcher reserves
`4 * (max_context_len + threads + head_dim)` LDS bytes per block. The Python
wrapper switches to global scores only when this exceeds 65,536 bytes, or
when the explicit global-score variant is selected. That safety fallback does
not prevent the already measured mid-context performance collapse.

The high device-time share is strong attribution. LDS occupancy loss is a
code-supported mechanism hypothesis; the exact occupancy/counter explanation
and a universal quadratic/cubic scaling law are not established by eight
points. Summed kernel durations divided by an unprofiled run's wall time also
do not establish an exact same-run GPU-busy percentage.

**Important correction:** the normal scalar bulk path calls the same
`_run_full_attention_prefill_layer_aotriton()` with AOTriton allowed.
`use_wmma_prefill=True` selects projection kernels; it is not evidence that
the scalar attention core is a separate in-tree WMMA score-GEMM implementation.
Trace the direct arm before describing its actual attention kernel family.

**Direct arm traced (2026-09-10, code only).** `_prefill_bulk()` is
layer-outer and, for `FULL_ATTENTION`, calls the same
`_run_full_attention_prefill_layer_aotriton()` with `allow_aotriton` left at
its default `True` and no per-call `aotriton_min_tokens` override. For INT8-KV
sessions on this geometry `_full_attention_prefill_scratch_for_layer()`
replaces only `key_cache`/`value_cache` with the BF16 oracle pair and leaves
`append_spans`/`prefill_spans` BF16, reaching the INT8 store through
`retained_*` alone; the wrapper's `direct_int8_prefill` term is therefore
False. `_gguf_aotriton_prefill_allowed()` has no gfx1100 capability entry and
returns its `True` default. So the direct arm's attention family is AOTriton
compact varlen (`_gguf_aotriton_prefill_mode()`, v3 by default) above the
512-row crossover and the same native paged kernel below it. This confirms the
correction above: there is no separate in-tree WMMA score-GEMM on the direct
BF16-oracle arm, and the server route differs from it by kernel selection
rather than by arithmetic contract.

**Both admission sites, and what the flag actually changes.** Two places
restrict AOTriton on this route, and only one of them gates admission.
`_prefill_batch_native_single_slab()` clears `force_aotriton_slots` when the
route is `direct_int8_prefill`; that drops the per-slot `aotriton_min_tokens=1`
override only, returning those slots to the standard 512-row crossover, so it
is not by itself a block. `_gguf_slot_local_prefill_allow_aotriton()` is the
gate: because `direct_int8_prefill` is False on transient-oracle layers for the
reason traced above, that helper supplies the only False term in the wrapper's
`use_aotriton` conjunction. Enabling the flag is consequently expected to admit
AOTriton on slabs at or above 512 rows while sub-512 slabs keep the native
kernel, including the tail slab of any prompt that is not a multiple of the
slab width. That is the same mixed pattern the admitted direct arm already runs
(`_chunk_ranges()` merges a tail only below `min_chunk_size=2`), so it is not a
new hazard; it does mean the P1 trace check must expect two kernel identities
and the P1 numerical packet must include a prompt whose final slab is under
512 rows. Expected admission is a code reading, not a measurement.

**Action:** re-qualify the existing AOTriton route first. A new paged/tiled
attention kernel is a conditional fallback project, not the first assumption.
Keep the registered strict arithmetic fallback and the default-off flag until
the declared profile gate passes.

### F2 - P1: the server pays for a second full-context KV reservation

`configure_engine_loop()` in
[`generation/qwen35_gguf.py`](../hipengine/generation/qwen35_gguf.py) constructs:

- Request pool capacity: `capacity * ceil(max_positions / 256)`, optionally
  capped by the configured high-water limit.
- Workspace capacity: `capacity * max(request_pages, 1024 / 256)`.
- Actual allocation: request capacity **plus** workspace capacity.

`create_global_device_kv_pool()` allocates that backing immediately, then
`lease_workspace()` pins the workspace pages. This is a real extra residency
cost, not just a bookkeeping reservation. Initial/chunk/low-water knobs do not
make this global route physically elastic: `GlobalDeviceKVPool.shrink_idle()`
is a no-op.

The slot-local prefill path now neither imports nor exports packed full-KV
history, and singleton decode uses the request's direct session. Nevertheless
it still creates `_GGUFPackedTargetState` and reserves the full workspace
capacity. Disabling copies did not remove their destination allocation.

**Action:** separate execution scratch/recurrent state from canonical KV
backing. For the no-packed-KV route, represent the absence of a packed KV
consumer in the resolved workspace plan. Do not merely unpin pages that a
fallback or graph can still address. BF16 packed decode, fragmented-page
fallbacks, and MTP must retain whatever their own declared consumers require.

### F3 - P1: corrected oracle ownership amplifies the pool size

`_int8_prefill_oracle_capacity_positions()` returns
`max(scratch.max_positions, backing.pages * block_size)`.
`_int8_prefill_oracle_cache_for_layer()` now correctly retains a separate pair
per INT8 layer during a multi-slab packed call. With the server's enlarged
pool, each of those pairs is enlarged too.

There is another avoidable combination: `_allocate_bulk_prefill_workspace()`
still allocates the full-capacity hidden owner from the
`layer_outer_shared_oracle` lifetime plan, while packed execution overrides
only oracle keying and remains chunk-outer. Thus packed execution can pay
both the layer-outer hidden allocation and chunk-outer oracle allocation.
The runtime fix is correct containment; its old lifetime estimate is not an
accurate estimate of the realized packed route.

**Slab count is a backend policy input, not a constant.**
`_prefill_batch_native_impl()` sets `oracle_per_layer = len(chunks) > 1`, and
the chunk count follows `_prefill_scratch_rows()` through
`_gguf_dense_prefill_scratch_row_cap()`. For this geometry
`GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_POLICIES` in
[`kernels/hip_gfx1100/__init__.py`](../hipengine/kernels/hip_gfx1100/__init__.py)
declares a single `max_rows_by_capacity` entry, so the cap resolves to 1,024
rows at every capacity at or above 1,024. Every prompt longer than 1,024 rows
is therefore multi-slab and takes the per-layer pair count, while sub-slab
prompts already take the one-pair path on today's code. That gives P3 a cheap
in-tree contrast for the plan/allocation binding check (compare realized
oracle owners for a sub-1,024 prompt against a 2,048 one), and it makes the row
cap an input the inspectable plan should name. Raising the cap is not a free
lever: the same policy's comment records that 1,024-row chunks measured faster
than 4,096-row chunks, and wider slabs enlarge the hidden and scratch owners
that P3 is trying to bound.

**Action:** make executor schedule, oracle count, address space, hidden
lifetime, and resource claim agree in one inspectable plan. A layer-outer
packed executor removes the per-layer pair count. A request-logical oracle
view removes dependence on unrelated physical pool capacity where applicable.

Do not blindly shorten the shared helper's allocation. The slot-local code
already distinguishes oracle spans from retained physical INT8 spans through
`_int8_retained_prefill_spans()`, but other packed consumers use physical
tables. Prove each writer/reader's address mapping, including shifted and
fragmented pages, before changing bounds. Keep retained payload/scales
addressed through scheduler-owned `KVLiveSpans`.

### F4 - P1: admission does not guarantee the complete request peak

`reserve_admission()` reserves prompt-plus-output KV pages and binds a session.
It does not reserve the newly realized per-layer BF16 oracle peak or all lazy
prefill workspace growth. Passing page admission therefore does not prove
that the following prefill can allocate its working set.

Startup does not close this gap. `prepare_request_scratch()` caps its warm
prompt at 128 tokens and skips packed AR warmup at C1. It does not exercise
the multi-slab oracle lifetime. The direct `gguf_capacity_probe.py` calls
scalar `session.prefill(use_bulk=True)`, not the server's executor.

**Action:** add a complete executor-derived transient claim and an
allocation-only server-mode preflight before publishing admission/readiness
claims. Account for shared scratch once, simultaneous live prefills explicitly,
graphs and pool backing separately, and reserve a declared device margin.
Use existing generic resource/admission interfaces. A failed preflight must
reject cleanly without corrupting survivors.

### F5 - P1 for mixed serving: internal slabs are not service yield points

The compact INT8 `_prefill_native_chunk()` fallback defers computation until
the last scheduler chunk, then executes the whole prompt inside one call.
The sampled full-row route likewise executes a complete prefill.
`EngineService._drive()` drains commands only between `poll()` calls.

Consequences: scheduler token quanta do not bound model work on these routes;
long prefill can delay other requests' decode, admission, cancellation, and
control calls. A faster attention kernel reduces the stall but does not
restore bounded service progress. The default service command timeout is
30 seconds, which is relevant when a single prefill call outlasts it.

**The blocking boundary is oracle lifetime, and it is narrower than it
looks.** `prefill_batch_native()`'s `finally` releases every session's INT8
prefill oracle buffers and clears `_int8_prefill_oracle_per_layer` on each call
return. Within one call the oracles already survive all internal slabs, which
is why the runner can chunk an 8,192-row prompt internally while
`_prefill_native_chunk()` cannot chunk it across scheduler ticks. The
checkpointable owner P6 needs is therefore an extension of an existing
cross-slab lifetime rather than a new capability. It must still keep the
release-on-failure guarantee the `finally` currently provides, which is
precisely the part that makes the extension non-trivial.

**Action:** define resumable prefill work with a real execution boundary.
For a layer-outer route, checkpoint layer index, chunk offset, hidden ownership,
current layer oracle, recurrent-state progress, and the intended final commit.
Do not pretend that processing a chunk of one layer completes those prompt
tokens through the model. Scratch rebinding and safe device completion must
precede interleaving.

C1 bulk mode can be the throughput reference. Mixed-serving qualification
also needs a maximum quantum and cancellation/decode-latency gate.

### F6 - P2: packed-workspace telemetry double-counts shared owners

`observability_snapshot()` sums `packed_workspace_nbytes()` across resident
sessions. `_resident_sessions()` deduplicates session identities, not workspace
owners. `resident_slot_view()` explicitly shares `_packed_ws_state`, and
`packed_workspace_nbytes()` walks that shared state for every view.

A CPU-only reproduction with a real shared `_PackedWorkspaceState` reports
2,048 bytes for two views of one 1,024-byte allocation. This can inflate the
reported workspace by resident capacity without any additional allocation.
Separately, the helper walks `.buffers` but omits separately owned
`full_attn_split_growth_buffers`.

The metric also intentionally excludes pool-leased KV planes. It cannot be
used as a proxy for workspace lease size. Per-request KV audits describe
referenced page bytes; when prefixes share pages, summing those references is
not unique physical residency either.

**Action:** publish owner-deduplicated resident bytes, borrowed/leased bytes,
referenced request bytes, transient peak, and total pool bytes as distinct
quantities. Include growth owners. Add real shared-view and prefix-alias
fixtures, not only independently allocated fake sessions.

### F7 - P2: C1 decode and C>1 INT8 have different remaining problems

For one physical row, `_step_native_rows()` selects `_step_native_serial()`,
which calls `session.step()` or `_step_native_c1_graph()`. This is already
the direct compute path. `_resolve_decode_graph_min_replay_steps()` rejects
non-BF16 KV, so both direct and server INT8 are eager under this policy.
INT8 graph support is not necessary to match the existing eager direct path.

For multiple compact INT8 requests, physical width is bounded by the
artifact's qualified consumer. `_qualified_kv_decode_batch_route()` fails
closed to width one without a registered qualified row-batched variant.
The serving fallback is intentionally serial. Increasing a row limit or
turning on a graph flag is not a substitute for the missing consumer.

Potential C1 host costs to measure, not yet assigned a throughput percentage:
per-tick `mem_get_info()` from `loop_barrier()`, physical-group/manifest
construction, sampling, per-token stream telemetry, token tuple construction,
completion processing, and output backpressure.
`_native_stream_chunk()` rebuilds the accumulated token tuple each step.
These are more relevant to long-output decode than the disproven KV import.

Do not chase a presumed one-millisecond sleep: the public LLM constructs
`EngineService(..., idle_wait_seconds=0.0)`.

### F8 - P2: prefix and sampling routes need separate parity rows

On a reused greedy prefix, `_prefill_native_chunk()` consumes the unmatched
suffix by repeated `session.step()`. This preserves state but can make a long
cache-hit suffix much slower than batched prefill. It is not evidence against
the no-prefix fix, and must not be hidden by an aggregate hit-rate metric.

Native GPU sampling and host sampling have different logits readback paths.
Graph eligibility also excludes some sampled/mixed cases. Start with greedy
AR and then gate each public sampling class; do not claim parity for all
requests from the greedy row.

### F9 - P2: existing diagnostic harnesses do not close server parity

- `gguf_prefill_route_ab.py` compares fresh private sessions. It exercises the
  packed entry but not the actual shared pool, resident capacity, shifted
  admission, service loop, or HTTP.
- `gguf_packed_kv_import_profile.py` continues with `session.step()`, not a
  server decode group. Its copy/hash result does not establish packed decode
  transition parity or live-server throughput.
- The A/B agreement switch checks generated IDs, not aligned intermediate
  state or the complete numerical profile gate. A requested agreement gate
  can also return success when only one arm was selected.
- Allocator peak counters in same-process A/Bs are cumulative unless reset;
  a later lower-memory arm can inherit an earlier arm's peak. Fresh processes
  or scoped allocation events are required for independent peaks.
- The import profiler advances one RNG across requested shapes. Running
  `--rows 8192` does not generate the same 8K input as the last case in
  `--rows 1024,2048,4096,8192`. Use explicit prompt IDs and hashes for matched
  timing/control runs. This does not erase the kernel-family attribution.
- Some docstrings still repeat the withdrawn GDN-arithmetic explanation.
  Correct historical descriptions in new notes without editing immutable logs.

## 3. Define "direct" before comparing

Public `LLM` is not the raw baseline: `LLM._get_text_generator()` constructs
`SubmitPollTextGenerator` and `EngineService`. A direct Python `llm.generate()`
versus HTTP comparison isolates transport, not the resident-versus-raw gap.

Create one matched case manifest and these five measurement boundaries:

| Boundary | What runs | Difference isolated |
| --- | --- | --- |
| R0 | Raw `Qwen35GGUFResidentSession.prefill/step`, current shipping selectors | Direct compute, memory, and arithmetic reference |
| R1 | Session with actual server pool binding/slot views and server prefill entry | Executor, layout, allocation, shifted ownership |
| R2 | `SubmitPollTextGenerator` / resident runner without service wrapper | Scheduling, admission, per-token host processing |
| R3 | `EngineService` / public LLM, token-ID inputs | Driver, commands, collectors, backpressure |
| R4 | HTTP blocking and SSE with exact same prompt IDs and output budget | HTTP parsing, transport, serialization, consumer behavior |

The controls must fix model hash, physical host/GPU identity, source and dirty
state, compiler/build, execution profile and selected manifest, quant/KV
policy, rounded capacity, prompt IDs, output count, prefix/MTP mode, sampling,
graph mode, cache history, and startup reserve. Default differences are not
performance regressions.

Measure load, warmup, prefill, capture, first-token publication, steady decode,
terminal drain, and end-to-end client wall separately. A raw graph replay
that reads all tokens only at the end is an offline ceiling, not the baseline
for SSE that publishes each token. Use one-step replay/readback in both arms
for the streaming-equivalent comparison. Exclude the prefill sample from the
timed decode denominator consistently.

## 4. Memory model and target

For the exact pure-INT8 FP32-scale geometry:

```text
KV bytes/token = 16 layers * 2(K,V) * 4 heads * (256 int8 bytes + 4 scale bytes)
               = 33,280 bytes = 32.5 KiB
KV bytes/page  = 256 * 33,280 = 8,519,680 bytes = 8.125 MiB
BF16 KV/token  = 65,536 bytes = 64 KiB
One BF16 oracle pair / position = 4,096 bytes = 4 KiB
One BF16 hidden plane / position = 5,120 * 2 = 10 KiB
```

These values were checked against `_qwen35_gguf_kv_page_bytes()` with fake
host geometry. They exclude weights, recurrent state, workspace, metadata,
alignment, mirrors, and driver overhead.

Let `C` be configured resident capacity, `L` the rounded per-request capacity,
`P` total pool positions, and `Npf` the number of simultaneously live oracle
sets. With no high-water cap and L >= 1024, current server `P ~= 2*C*L`.
Current multi-slab oracle bytes are `64 KiB * P * Npf`. Do not substitute
`Npf=C` automatically: the present serial full-prompt route normally executes
one such prefill call at a time.

At C1, pure INT8, one live multi-slab prefill, and one aliased hidden plane,
the following are **partial analytic coefficients, not measured total slopes**:

| Major context-proportional term | Raw layer-outer direct | Current packed server |
| --- | ---: | ---: |
| Canonical KV backing | 32.5 KiB/L-token | 32.5 KiB/L-token |
| Additional packed-KV reservation | 0 | 32.5 KiB/L-token |
| BF16 oracle | 4 KiB/L-token | 128 KiB/L-token |
| Hidden plane | 10 KiB/L-token | 10 KiB/L-token |
| Subtotal | 46.5 KiB/L-token | 203 KiB/L-token |

INT64 token buffers and other workspace/metadata are additional. At L=32,768
the subtotal difference alone is about 4.891 GiB. This is a formula-based
explanation of why the gap can be large, not an XTX capacity prediction.
The low-level private-session oracle fix's +0.94 GiB at L=16,384 must not be
extrapolated unchanged to a server pool of a different size.

The measurement ledger must distinguish:

1. Unique resident allocation owners, including alternate weight layouts.
2. Pool capacity bytes, counted once, with occupied/free/pinned/leased subsets.
3. Request-referenced KV bytes, allowing explicitly identified shared pages.
4. Shared compute scratch, per-slot recurrent state, graph state, hidden planes,
   sampler storage, and oracle owners at their actual live boundaries.
5. Tracked current/peak, sampled whole-card current/peak, and unexplained
   residual, in separate accounting domains.

Sweep declared capacity with a fixed short multi-slab prompt to expose
reservation slope; sweep actual prompt with fixed declared capacity to expose
lazy allocation and execution transitions. Separately vary configured C and
live occupancy. Page alignment and route thresholds make piecewise fits
necessary. Never divide total VRAM by tokens and call it KV bytes/token.

**Target:** same canonical KV coefficient and same necessary prefill working
set as the matching raw execution contract; no server-only duplicate
full-context KV/oracle slope. A documented bounded host/control and per-slot
overhead is acceptable. Physical fit and operational reserve remain separate.

## 5. Implementation packets and exit gates

Every packet is one or more small validated commits with an immutable handoff.
Do not postpone a complete, validated unit's commit until a later campaign.

### P0 - Freeze measurement and repair accounting

Deliver:

- The R0-R4 case manifest, exact token-ID fixtures, route/shape counters,
  source/compiler/variant provenance, stage timing, and independent peaks.
- Fix F6 using allocation-owner identity, including growth buffers. Expose
  workspace lease pages/bytes independently of workspace allocation bytes.
- Add oracle count/capacity/address-space and hidden-owner bytes to stage
  captures. Report actual executor mode separately from legacy lifetime plan.
- Make asserted comparison modes fail closed on a missing/failed arm or
  mismatched prompt/shape/profile. Remove stale attribution text.

Gate: CPU alias/growth/lease fixtures reconcile exactly; a small GPU canary
reconciles owners and route identities. Publish an explicitly unexplained
residual rather than fitting it to the desired conclusion.

### P1 - Restore an efficient full-attention prefill route

Deliver: qualify the existing slot-local transient-oracle AOTriton route on
the repaired code. Compare with both the corrected packed parent and R0.
Use the common layer implementation; resolve admission through existing
backend/variant policy instead of adding an engine-side special case.

Test single/multi/tail chunks, nonzero starts, shifted contiguous bases, and
the noncontiguous fallback. Check causal Q/K length alignment, scale writes,
positions, final hidden/logits, and decode continuation. A raw AOTriton
pointer cannot consume arbitrary physical pages without a proven logical
oracle, rebase, or gather. Merely flipping the flag is not qualification.

Gate: applicable strict/production numerical and task packet, deterministic
repeats, exact ownership, expected attention kernel in trace, and same-host
repeated wall improvement at 1K/2K/4K/8K plus a safe longer point. If it wins,
promote in its qualified scope and keep the strict fallback.

If blocked: isolate the exact cause first. Consider the already registered
global-score strict variant as a bounded diagnostic; measure occupancy and
launch count. A tiled paged BF16 kernel using `KVLiveSpans` is the subsequent
kernel project if arbitrary-page or arithmetic constraints prevent reuse. That
project has an in-tree structural donor in
`qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_wmma_spans` (BF16 WMMA score
GEMM with per-tile exp and online max, 2026-09-09), whose score-GEMM and
epilogue structure is the reusable part. Reuse the structure only: its K/V
source, scale handling, and measured numerics belong to the oracle-free
INT8-read route. Do not confuse oracle-free INT8-read prefill with this
BF16-oracle route.

### P2 - Build a server-faithful allocation preflight

Deliver: extend the existing probe infrastructure with actual resident owner,
pool/workspace leases, rounded capacity, sampling mode, and effective KV
capability. Allocate the selected route's worst live working set before
expensive math. Where lazy behavior remains, use the shortest request that
crosses its relevant boundary, including at least two internal slabs.

The probe must cover later context-dependent partial/global-score growth,
not assume that any 2K execution allocates every long-context buffer. Report
last pass/first allocation failure, stage, peak, reserve, and clean teardown.
Integrate the same resource model into admission; avoid a separate estimator
that can drift from the executor.

Gate: a short allocation probe predicts one safe full-depth confirmation;
forced allocation failure rejects cleanly, returns to baseline, and does not
break an existing survivor. Bracket capacity with allocation probes only.
Keep old server ceilings unqualified until this route is exercised.

### P3 - Match the direct prefill memory lifetime

Deliver in separable steps:

- A layer-outer execution plan using one oracle pair per required live
  prefill owner and the direct hidden-plane lifetime. Reuse the common layer
  functions, not a duplicated model forward implementation.
- Explicit oracle-logical versus retained-physical views. Size each allocation
  to the address range its consumer can actually access.
- Binding plan/allocated-owner checks so packed execution cannot silently
  realize 16 oracle pairs while reporting a one-pair plan.

Gate: layer-boundary hidden/state comparisons, full logits under the declared
profile, exact KV/control and cancellation cleanup, tail/ragged/shifted page
fixtures, and measured removal of the expected owner bytes. Validate aliasing
on the new executor itself. Preserve the corrected chunk-outer fallback.

This fixes memory, not automatically F1 or service fairness. Checkpointable
execution belongs in P6 before broad mixed-serving promotion.

### P4 - Remove unused packed KV and duplicate execution scratch

Deliver: separate the currently combined `_GGUFPackedTargetState` needs:
canonical request KV, packed fallback KV, linear state, row metadata, and
compute scratch. For eligible C1/slot-local AR, borrow request storage and
retain only the necessary row/linear scratch. Reuse compatible bulk and
packed scratch according to proven live ranges; do not free/reallocate it on
every reclaim.

Evaluate capability-dependent workspace leases, not a blanket zero lease.
For native C>1, move consumers toward request-owned KV row views so a second
full-history store is not required. MTP's provisional state has different
ownership and must not alias committed KV.

Gate: no server-only second context-sized KV reservation for the scoped
route; unique-owner census matches the planned delta; no hot-path allocation
churn after warmup; fragmented pages, refill, group membership changes, and
graph invalidation are exact. Graph pointers must remain valid across reuse.

### P5 - Close singleton decode overhead; then native INT8 C>1

Start the decode measurement part of P0 immediately; it need not wait for
all memory work. After each executor change, repeat only affected gates.

Deliver for C1:

- R0-R4 one-token-boundary timing with the same eager/graph mode and sampler.
- Kernel-family, HIP API, queue-gap, D2H/H2D, telemetry, and host CPU attribution.
- Narrow fixes for measured overhead: for example move memory telemetry to
  bounded cadence/stage events if its per-tick cost is material, avoid
  repeated full-history token construction, and preserve bounded queues.
- Explicit graph capture cost and amortization; do not compare an amortized
  raw graph with a cold server request without reporting that distinction.

For C>1, implement/qualify the existing IKV-C2 consumer project, including
row-batched split-K producer/reducer, scale-aware per-request KV views,
deterministic row isolation, and physical-width evidence. Compare against the
same direct batch shape or report the serial aggregate reference honestly.
INT8 graphs are a separate later optimization, not a prerequisite for C1
eager parity.

Gate: same compute route and numerical contract; no unexplained model-step
gap; exact cancellation/refill and C1<->C2/C4 transitions. Report per-request
latency and aggregate throughput separately.

Status 2026-09-11: landed and promoted on the W7900 gfx1100 Qwen3.8-27B INT8
artifact. The primitive gate is bit-exact against the CPU reference and
independent c1 for c1/c2/c4/c8; the packed-AR model gate is token-exact with
max KL 0.0 and top-1 1.0 over four rows by four steps; the transition gate
holds a 4 -> 2 -> 4 lane schedule exact per row at every step with excluded
retired lanes byte-frozen; and a cache-only `rocprofv3` trace shows one batch
producer plus one strided reducer launch for all four rows. On the live server
the measured same-server serial control shows aggregate complete-request
throughput of ~1.23x (C2) and ~1.40x (C4) over the serial rate, with
`serial_decode_fallback_steps` at zero. Per-request latency still grows with
width and is reported separately from aggregate throughput.

Per-request latency at C>N is a published figure (4.95 s at C1, 7.97 s at C2,
14.1 s at C4 concurrent, against a 5.06 s serial control at C4), so 1.42x
aggregate throughput costs 2.80x per-request latency and the tradeoff is
visible rather than implied. Cancellation safety for a formed packed group is
pinned by tests over the chunk-change flush that writes a group's accumulated
packed state back before the group re-forms, including a newcomer reusing a
freed lane at equal width. The packed-ownership trace has a committed generator
that resolves the capability snapshot live, so the artifact cannot record a
width the registry has moved past. A cancellation racing an in-flight *packed
prefill* is the P6 resumable-prefill intersection rather than a separate IKV-C2
gap.

Still open in this packet: the C1 decode HIP-API/queue-gap, D2H/H2D, telemetry,
and host-CPU attribution, and explicit graph-capture cost and amortization.

### P6 - Restore bounded prefill/decode service and cover public modes

Deliver the resumable prefill owner described in F5, integrated with the
existing scheduler. Bulk and fair modes must expose actual GPU work progress,
not just prompt tokens buffered in Python. Define a bound on non-preemptible
work before admission, cancellation, and decode can run again.

Cover a long arrival while a short request decodes; cancellation during
prefill; refill and sparse survivors; BF16 and compact INT8; native and host
sampling; long prefix suffixes; blocking and SSE; slow/disconnected consumers.
Extend prefix suffix prefill to a qualified chunked path only after its state
ownership is proven. Keep unqualified MTP/DMS combinations separate.

Gate: unchanged resource conservation, correct survivors, bounded command
acknowledgement/cancel delay, explicit p95/p99 decode gap/TTFT SLOs, no
unbounded output queue, and no hidden serial fallback in native claims.

**Coverage status as of 2026-09-11** (evidence:
`benchmarks/results/2026-09-11-w7900-p6e-cancel-refill-service-proof.json`,
eleven gates - ten passing and the native-sampling gate explicitly `skipped` with its blocker named - `passed: true`, `performance_claim: false`):

| P6 coverage item | Status |
| --- | --- |
| Long arrival while a short request decodes | Covered (blocking and SSE arms) |
| Cancellation during prefill | Covered; swept at 600/2000/3500 ms, acknowledgement flat within 1% |
| Refill and sparse survivors | Covered; all five survivors byte-identical to the reference across **192 characters** (96 decode tokens), one shared sha256 |
| Blocking and SSE | Covered |
| Slow / disconnected consumers | Covered |
| Compact INT8 | Covered end to end on the resumable route |
| Host sampling | Covered; the host route is asserted by route counter, not inferred from text |
| Native sampling | **Blocked by a real defect.** `HIPENGINE_QWEN35_NATIVE_SAMPLER` gates the native GPU sampler (`qwen35_gguf.py:10206`) and is off by default; with it off, a native-shaped request silently falls back to the host sampler. With it **on**, the packed native sampler has capacity 0, so the request fails `400 invalid_request: "packed native sampler row 0 exceeds capacity 0"` and the engine then times out and closes. The harness carries the arm as `--native-sampling-arm` (default off, because running it closes the engine) and its gate reports `skipped: true` with this blocker named. Reproduce: `HIPENGINE_QWEN35_NATIVE_SAMPLER=1 python3 scripts/gguf_p6e_cancel_refill_proof.py --native-sampling-arm` |
| BF16 | **Scope clarification, not a gap.** The resumable executor is only attempted when the session's `kv_attention_source` is `int8_direct` (`qwen35_gguf.py:8094`), so the BF16 route cannot use this mechanism. Its bounded yield comes from the default route and is measured by the seven non-engagement gates, which pass under BF16. "BF16 resumable coverage" therefore has no referent: there is no BF16 resumable path to cover |
| Long prefix suffixes | **Intentionally fail-closed.** `_prefill_resumable_int8_chunk` returns False for `row.prefix_reused_tokens` (`qwen35_gguf.py:8204`), because shared-prefix admission needs incremental prefill the layer-outer executor does not provide. A qualified chunked path requires proving its state ownership first |
| Layer budgets other than 4 | **Not a settable knob.** The budget is derived, not configured: `budget = max(1, ceil(remaining_layers / remaining_polls))` with `remaining_polls = ceil(remaining_tokens / chunk_len)` (`qwen35_gguf.py:8244-8248`). What varies it is prompt length and prefill chunk size, so a budget sweep means varying those, not a flag |

The service proof must be run with `kv_storage="int8_per_token_head"`; the harness
defaults to it and refuses to run the workload otherwise, because a run on the
default `auto` layout resolves to BF16 and produces green gates about a route that
does not contain the resumable mechanism.

### P7 - Publish the parity and capacity result

Proposed engineering acceptance targets, to freeze before the qualifying run:

- Warm C1 prefill and decode model-stage median at least 95% of matched R0
  throughput, with paired repetitions and noise reported; no material tail
  regression hidden by the median.
- Exact canonical KV bytes/page and zero unnecessary server-only linear
  context-memory term for the scoped route. Explain remaining fixed/per-slot
  differences by unique owners rather than a permissive total-VRAM ratio.
- HTTP blocking/SSE overhead separately budgeted in milliseconds/token and
  TTFT; no promise of literal transport-free wall parity.
- Operational context tested with the same reserve, output horizon, workload,
  actual server route, and physical card as the direct reference.

Use all four mtp-bench prompt categories and category-heldouts for the task
gate, even for an AR performance path; use synthetic shapes for mechanism and
allocation tests only. Vary page tails, prompt lengths, output horizons and
pool history. Run each GPU as an independent lane.

Update the compact artifact, benchmark README/date, changelog, public
settings, and applicable plan/campaign state only after qualification.
Run the benchmark export check. Remove or ledger obsolete flags/fallbacks in
[REFACTOR.md](REFACTOR.md). A provisional C1 result is not closure for C>1,
prefix, sampled, MTP, or mixed-serving modes.

## 6. Immediate assignment for the coder

1. Continue the corrected-route AOTriton gate already in progress. The direct
   attention family is now recorded in F1 (AOTriton compact varlen above the
   512-row crossover); what remains is verifying the page/address contracts and
   covering both kernel identities in the trace and numerical packets.
2. In the next accounting unit, fix shared-owner workspace telemetry and add
   oracle/lease/hidden live-stage counters plus the server allocation probe.
3. Implement layer-outer ownership and remove the unused full-context packed
   KV requirement as separate, independently gated memory changes.
4. Capture the R0-R4 C1 decode baseline before optimizing host code. Follow
   the existing IKV-C2 campaign for native compact C>1.
5. Do not declare server parity until real yield points and the mixed-request
   lifecycle packet pass.

No new attention port, complete context ladder, generic HTTP rewrite,
per-token allocator, or speculative feature campaign is justified before
these smaller discriminating steps.

## 7. Test handoff for implementation

These are instructions for the implementing coder, not tests certified by
this review. Check current source and test names before running: other agents
are changing the gate and decode harnesses concurrently. Coordinate GPU
ownership, prewarm outside the profiler, and use the cache-only recipe in
[HARNESSES.md](../benchmarks/HARNESSES.md). Do not run every bundle after
every change.

### Existing test anchors

| Packet | Existing tests to extend or run |
| --- | --- |
| P0 accounting and comparison | `tests/test_gguf_packed_workspace_stability.py`, `tests/test_kvcache_global_device_pool.py`, `tests/test_kvcache_global_pool.py`, `tests/test_benchmark_matrix.py`, `tests/test_exact_token_benchmark.py` |
| P1 attention admission | `tests/test_qwen35_gguf_slot_local_aotriton_admission.py`, `tests/test_gguf_int8_prefill_oracle_per_layer.py`, `tests/test_gguf_packed_kv_import_skip.py`, `tests/test_gguf_device_kv_binding.py` |
| P2-P4 resource/lifetime changes | `tests/test_qwen35_gguf_int8_kv_policy.py`, `tests/test_gguf_int8_prefill_oracle_per_layer.py`, `tests/test_int8_layer_outer_hidden_alias.py`, `tests/test_gguf_packed_workspace_stability.py`, `tests/test_gguf_packed_verify_layout.py`, `tests/test_gguf_device_kv_binding.py` |
| P5 decode and sampler | `tests/test_generation_qwen35_gguf_sampling.py`, `tests/test_gguf_packed_execution_manifest.py`, `tests/test_gguf_packed_decode_graph.py`, `tests/test_qwen38_int8_batch_decode_gate.py`, `tests/test_qwen38_int8_kv_capability.py` |
| P6 service and lifecycle | `tests/test_generation_engine_service.py`, `tests/test_generation_engine_loop_burst.py`, `tests/test_qwen38_int8_server_context_soak.py`, plus the affected nodes in `tests/test_server_api.py` |

An existing test filename is not proof of coverage for a new route. Add RED
fixtures for the actual gaps: shared-owner sums, omitted growth owners,
missing comparison arms, transient admission denial, executor/plan mismatch,
logical versus physical oracle addressing, and yielding during real prefill.
Tests that invoke HIP or build kernels need an explicit availability guard.

For the narrow oracle/import contract, the existing command is:

```bash
python3 -m pytest -q \
  tests/test_gguf_int8_prefill_oracle_per_layer.py \
  tests/test_gguf_packed_kv_import_skip.py
```

Service tests should be run with visible node progress and a bounded
diagnostic timeout, for example:

```bash
timeout 180s python3 -m pytest -vv -o faulthandler_timeout=60 \
  tests/test_generation_engine_service.py
```

The timeout is a diagnostic bound, not a product SLO or a passing verdict.
Capture the stalled node/stack and isolate it before rerunning the bundle.
Preserve the initial result plus focused repairs; follow the repository's
no-automatic-broad-rerun rule.

### Required boundary matrix

Freeze exact inputs and expected routes before measurement. At minimum:

- Prompt rows around page, attention-admission, and slab boundaries:
  255/256/257, 511/512/513, 1023/1024/1025, 2048/2049, 4096 and 8192.
  Verify actual configured thresholds rather than assuming those constants.
- Fixed short multi-slab prompt at increasing declared capacity, and increasing
  prompt length at fixed capacity. Include the global-score fallback boundary
  and deferred workspace growth without using full prompts to bracket memory.
- C1 at configured capacity 1 and at wider configured capacity; then supported
  C2/C4 and their return-to-C1 transitions. Keep unsupported widths explicit.
- Identity, shifted contiguous, fragmented, and reused-prefix page layouts;
  ragged prompt lengths, partial final slabs, and successive request reuse.
- Greedy output horizons 8 for diagnostics, 128 for steady decode, and a
  longer horizon such as 512 for token-processing and lifecycle costs.
  Force the intended count through supported ignore-EOS controls and verify
  actual tokens; do not silently count early EOS as horizon completion.
- Prefill cancellation, exception/allocation failure, survivor decode,
  refill, graph invalidation, slow consumer, disconnect, and clean drain.

Do not take the full Cartesian product blindly. Each packet selects its
binding axes and states which others are unchanged, unsupported, or deferred.
Full task qualification uses the complete four-category suite and heldouts,
not these synthetic shapes.

### GPU and release evidence

Reuse `scripts/gguf_prefill_route_ab.py` and
`scripts/gguf_packed_kv_import_profile.py` only within their diagnostic scope.
The corrected-route numerical adapter
`scripts/execution_profile_gguf_int8_direct_prefill_gate.py` must identify
whether it tests oracle-free INT8 reads or BF16-oracle AOTriton admission;
its filename alone does not establish the route.
Use the applicable profile evaluator and independent reference fixtures.

For a new kernel, run its declared RED/GREEN and CPU-reference gate, then
cache-only kernel-trace smoke with expected identity, plausible duration,
and manifest provenance. Existing `tests/test_qwen38_int8_batch_attention_gpu.py`
and `tests/test_qwen35_int8_prefill_attention_gpu.py` are relevant only if the
changed consumer is actually exercised by them.

R0-R4 timing needs a matched harness/manifest, not a ratio assembled from
unrelated existing runs. Reuse `scripts/exact_token_generation.py`,
`scripts/benchmark_matrix.py`, `scripts/gguf_live_server_bench.py`, and the
allocation-ledger helpers where their contracts apply; add missing adapters
before calling the comparison complete.

At milestone closure, run the repository's full `uv run pytest -v` and named
performance/profile gates after coordinating with other workers. Documentation
publication runs `python3 scripts/worklog.py check` and
`python3 scripts/sync_benchmark_readme.py --check`. A new full all-green run is
not required merely to document this review.

## 8. Review validation and remaining uncertainty

The review inspected the HTTP service entry, public LLM wrapping, engine
service/loop, GGUF generation adapter, resident session/prefill/decode,
workspace and global pool ownership, attention wrapper/kernel, and relevant
campaign/harness evidence. Source references above are symbol-based so they
survive line movement.

CPU-only checks on the reviewed tree:

- `python3 -m pytest -q tests/test_gguf_int8_prefill_oracle_per_layer.py tests/test_gguf_packed_kv_import_skip.py`
  passed all 19 tests.
- Shared-holder reproduction using `object.__new__(Qwen35GGUFResidentSession)`,
  two sessions sharing `_packed_ws_state`, and one fake
  `DeviceBuffer(4096, 1024)` confirmed the 2,048-byte summed counter.
- `_qwen35_gguf_kv_page_bytes()` with 16 INT8 layers, 4 KV heads, head dim 256,
  FP32 per-token/head scales returned 8,519,680 bytes/page.
- A fake 32,768-position session backed by 256 pages of 256 tokens returned
  65,536 oracle-capacity positions.
- The separate command
  `python3 -m pytest -q tests/test_gguf_packed_workspace_stability.py tests/test_generation_engine_service.py tests/test_generation_engine_loop_burst.py`
  stopped reporting progress after 32 passing-test markers. No completed
  verdict or failing-node attribution was obtained before the user requested
  read-only review. This is incomplete validation, not an all-green bundle
  or a diagnosed product defect. The coder should inspect any surviving
  review process before starting another run; no further tests were launched
  for the documentation closeout.

No new GPU run, server capacity certification, or HTTP throughput measurement
was performed for this review. Existing profiler attribution does not prove
every kernel's occupancy mechanism or every server shape's behavior.
The uncommitted census artifact and benchmark README work belong to another
lane and were not edited. Runtime/gate implementation work remains with the
coder; this document does not claim its pending experiments have passed.
