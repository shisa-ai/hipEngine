---
status: current
owns: Implementation checklist, supported scope, and acceptance tests for serving GGUF MTP with INT8 KV.
---
# INT8 MTP Server Readiness

This checklist is the implementation plan for taking the existing dense GGUF
INT8 target verifier through the normal library and HTTP serving paths.
Changing the server's default KV storage is outside its scope. Tests must
select INT8 explicitly and work when the product default is BF16, INT8, or auto.

Implementation availability, approximate-KV quality, and MTP performance are
separate questions. A missing MTP benchmark row must not disable working code.
A measured INT8-versus-BF16 quality failure is not repaired by showing that
INT8 MTP matches INT8 AR. Neither is an explanation for a missing implementation.

## Supported Scope

The first shipping scope is dense Qwen GGUF with a NextN head, uniform paged
per-token/head INT8 K/V, and registered gfx11 kernels. Scale planes are part
of the storage contract. Shared engine ownership is mandatory, including
single-request MTP in a wider resident server.

Concurrency must never be misreported: a request-local verifier is not a
physical multi-request verifier. A packed request the kernels cannot execute
must return a named capability error before mutation. Automatic selection may
use an implemented AR route. Neither case may kill the engine.

Compact DMS is an additional retention topology, not another name for INT8.
It needs eviction/compaction transactions as well as quantized payload and
scales. Dense support must not advertise DMS support implicitly.

## Checklist

Status: `[x]` implemented and tested at the stated surface; `[ ]` work remains.
Existing direct-runtime results are not HTTP completion evidence.

### Storage And Admission

- [x] CLI tests exercise explicit BF16, INT8, and auto plus environment
  precedence without pinning one product storage default.
- [x] Explicit INT8's scale defaults match its supported storage contract.
- [x] MTP uses effective storage, layout and scales after KV resolution, not
  merely the requested string.
- [x] Requested/effective storage, mirror use, and fallback cause are visible
  in readiness and request diagnostics.
- [x] Model/provider capability admits implemented INT8 verification without
  treating BF16-only evidence as INT8 evidence.
- [x] Automatic intent, explicit enable, and explicit disable work independently
  of the global KV default; explicit disable executes zero speculative cycles.
- [x] Sampling, tree masks, mixed layouts, resource failures, and missing
  kernels receive accurate structural reasons before allocation or mutation.
  A sampled request either runs the sampled route its artifact admits or keeps
  the autoregressive route with the sampling blockers named; an unimplemented
  storage layout is refused by name (`tail4_hadamard_group32`); a missing packed
  capability keeps the AR route with a named capability miss rather than running
  a partial one; and resource exhaustion is refused by the unified admission
  budget before anything is allocated. Tree masks name a scope, not an untested
  path: the candidate ladder is a linear chain, so this engine has no tree mask
  to reject.
- [x] Candidate-depth and context coverage derive from implementation capacity;
  no benchmark-only window is introduced.
  The packed multi-choice route no longer carries a 1024-token live-context
  gate: admission follows the artifact's admitted no-mirror capability and the
  physical cell the kernels execute. The row that found the gap now reports the
  reason it was written for, over both endpoints at all four lengths.
- [ ] Automatic intent at physical c2 on INT8 KV. The named verification
  command does not reach INT8, so this row cannot be cleared by it as written.
  `scripts/gguf_mtp_c1c8_server_bench.py` constructs `LLM` and `ServerConfig`
  without a KV storage selector and has never contained one, so it runs the
  product default; `/ready` reports `effective_kv_storage: bf16` for that exact
  construction, including when `int8_per_token_head` is passed to both objects.
  Every artifact that script has produced under an `int8kv` name is therefore a
  BF16 run. Read live at a realized width of 2 (full route decision, commit
  `86de3308`), the withholding there is a retained **BF16** evidence row:
  `policy_cell: qwen38-q4km-gfx1151-production-bf16-c2-k3-d24`,
  `reason: diagnostic_production_c2_after_ar_rebase`, `automatic_eligible:
  false`, `static_intent_allowed: false`, so
  `_serving_plan_route_decision` reports
  `automatic_mtp_scope_not_promoted`. That is a promotion decision on a measured
  cell -- the explicit arm of the same sweep engages it at 25.75 tok/s against
  its own AR baseline's 10.79 -- not a capability miss and not an INT8 static
  bound.

  The INT8 static-width half of the original finding was real and is now fixed:
  a request's static width bound follows the widest automatic-eligible
  declaration for its storage and backend instead of the singleton declaration
  its own realized width selects (`hipengine/models/qwen35.py`, commit
  `7c3914162`, pinned by `tests/test_unit_int8_mtp_serving.py`). It is simply
  not what withholds this sweep's c2, because this sweep is not the INT8 cell.

  Clearing route, corrected: run the automatic arm against a server whose
  `/ready` reports `effective_kv_storage: int8_per_token_head` -- the CLI route
  (`hipengine serve --kv-storage int8_per_token_head`), which the category,
  lifecycle, prefix, pressure and sampled gates already assert -- and confirm
  `engaged_cells` and `route_expectation_passed` at c1, c2 and c4 there. If the
  bench is to be the vehicle for an INT8 row it needs a KV-storage selector that
  the run records, otherwise its artifact cannot be told apart from a BF16 one.

  Recorded run: `benchmarks/results/2026-09-24-gfx1151-qwen38-default-kv-mtp-vs-ar-c1c4-automatic.json`
  (status failed; c1 and c4 engaged 10/10, c2 0/10).
  Evidence: `benchmarks/results/2026-09-23-gfx1151-qwen38-int8kv-mtp-vs-ar-c1c4.json`
  (automatic arm: c2 engaged 0/10, route `default`, decision reason
  `automatic_mtp_scope_not_promoted`; explicit arm: c2 2.386x with 10/10
  engaged).

### Runtime And Ownership

- [x] Shared-page-table INT8 attention accepts per-row causal live counts.
- [x] FP16/FP32 scale primitive gates pass on gfx1151; c1 fallback is registered.
- [x] Dense C1 eager/graph rejection, partial/full acceptance and rollback
  preserve live payload/scales, recurrent state, cursors and next logits.
- [x] Graph binding signatures include page tables, payloads, mirrors and scales.
- [x] Page/split transitions are exercised, including an unrelated short case.
- [x] The resident owner uses this verifier with shifted/global pool pages,
  not only privately allocated identity tables.
- [x] A C1 request under the default wider resident capacity actually speculates.
- [x] Admission/retirement/refill and mixed AR/MTP neighbors preserve identity
  and select an honest physical route for each occupancy.
- [x] Packed multi-request INT8 verification runs the same row-bulk verifier
  pass as BF16, binding the retained INT8 payload planes and their
  per-token-head scale metadata so its full-attention layers attend through the
  retained-decode leaf. Multi-choice (`n>1`) in one request and concurrent
  requests coalesced into one decode step both speculate and match the
  autoregressive ids. The packed direct INT8 prefill admits the same group width
  as decode, because both write the same packed physical cell. Admission is
  bounded by the artifact's admitted no-mirror capability (physical c4), so an
  artifact whose compact INT8 capability is rejected reports a named capability
  miss and keeps its autoregressive route instead of running the packed path.
- [x] Graph reuse after cancellation, pool growth, slot reassignment and scale
  reallocation does not retain stale pointers or another request's state.
  The packed-decode churn guards cover cancellation, pool growth, slot
  reassignment and scale reallocation (`tests/test_unit_int8_mtp_teardown.py`),
  and the live pool-ownership gate grows the packed verify scratch across a real
  MTP generation and then runs MTP on the far side of the growth
  (`tests/test_live_gguf_pool_ownership.py`).
- [x] Cancellation/deadline/failure/shutdown drain target and provider ownership
  exactly once; the next request remains usable.
  Live deadline/disconnect reuse and final library teardown pass. Allocation
  tracing after a prefix/MTP cycle reports zero outstanding HIP allocations.

### Prefix Cache And Memory

- [x] Prefix reuse copies/shares every INT8 payload and scale plane; append COW
  preserves the cached source.
  The independent INT8 prefix-owner gate covers active/completed sources;
  MTP composition additionally passes the four-category warm-prefix gate.
- [x] Draft-provider priming/restoration follows prefix reuse, or reports a
  specific supported fallback without corrupting provider position.
  Radix caching now retains four provider checkpoints by default; the live
  W7900 gate verifies 512-token hits and actual MTP after restoration.
- [x] Prefix hit/miss and eviction under pressure, with MTP enabled and
  disabled. `scripts/int8_mtp_prefix_pressure_gate.py` runs one request sequence
  against two servers that differ only in `--prefix-cache`. The retained working
  set is bounded by the request capacity -- one durable boundary per active
  request -- and trimmed oldest-first, so seeding `2 x capacity` distinct prompts
  evicts the oldest half deterministically. On this host that is capacity 4 with
  8 seeds, and the recorded residency series stops growing at 5. The
  first-seeded prompt then misses (`fallback_reason` `"miss"`, no match, no
  reuse) while the last-seeded one hits with 512 reused tokens
  (`matched_tokens` 512, `source` `completed_snapshot`), and the disabled-cache
  server reports `cache_off` with zero residency for the same sequence. Every
  probe returns the same generated ids in all four configurations, and the
  speculative arm over reused KV returns ids identical to the autoregressive arm
  over that same reused KV, which is the provider-position check. A miss
  re-populates a boundary, so the two arms of an evicted probe use different
  prompts; a hit refreshes the boundary it just used, so the resident probe can
  reuse one prompt for both arms. True request overlap was not achievable on
  this host: the startup scratch probe runs at width 1 here (4 concurrent
  sessions need 208.20 GiB against 57.07 GiB usable), so the pressure comes from
  sequential MTP requests with a concurrent tail that serializes.
- [x] Cancellation, deadline and disconnect mid-cycle restore the store, and the
  session's teardown leaves no outstanding allocations.
  `tests/test_live_dms_int8_mtp_lifecycle.py` (4 passed, live tier) drives the
  same resident DMS+INT8 harness the parity gate uses. Each arm opens a real
  verify cycle -- which appends its rows to the DMS store -- and asserts the
  cycle actually mutated the store before aborting, so the rollback is not
  vacuous. Restoration is compared field by field against the pre-cycle state:
  host payload and scale planes, positions, live counts, evict mask, range
  capacity, base offsets, extents, and the extent pool's and ledger's own
  allocator state, plus the device payload store's K/V payload, scale, position,
  evict and live-count planes (the signature requires the device store to exist,
  so those planes cannot be skipped silently). The arms are cancellation
  (`GenerationCancelled`), a real expired deadline through the module's own
  `_deadline_checkpoint` factory (`GenerationDeadlineExceeded`, with a second
  unwind asserted to be idempotent), and an unexpected mid-cycle exception
  (`ConnectionResetError`) -- each followed by a fresh cycle that must still
  commit exactly the autoregressive tokens. The fourth arm returns
  `memory_stats()["active_allocations"]` to its pre-session baseline after
  teardown with an open, uncommitted cycle. Scope, stated exactly: DMS reaches
  the resident-session surface only -- no `dms_metadata_path` reaches the engine,
  LLM, or HTTP layer -- so these arms observe the lifecycle at the cycle
  boundary the verifier owns rather than through an HTTP client disconnect.
- [ ] Prefix reuse after a cancelled request is tested with MTP enabled and
  disabled. The lifecycle half of this row is now covered: cancellation, deadline
  and disconnect mid-cycle and the session-level allocation trace are green in
  `tests/test_live_dms_int8_mtp_lifecycle.py`. What is still missing is the same
  trace for a request cancelled *after* its prefix was reused, in both cache
  configurations.
- [x] Admission budgets include target KV/scales, draft KV, verifier scratch,
  graphs, and retained prefix ownership; overload fails before HIP OOM.
  `hipengine/runtime/memory_admission.py` prices every resident consumer and
  `require_memory_admission` raises `MemoryAdmissionRefused` carrying the
  refused consumer and the priced totals before any allocation;
  `tests/test_unit_memory_admission.py` (18 passed) makes each of the five named
  consumers the refusal reason when it is the one that does not fit, and
  `tests/test_integration_server_api.py` shows the refusal answered over HTTP as
  capacity rather than as an internal fault. The refusal is exercised against
  the priced budget, not by driving the device to exhaustion.
- [x] Compact mode reports no persistent BF16 mirrors; any mirror mode is
  explicitly reported rather than presented as compact INT8.
- [x] Pool growth/shrink and final request reclaim leave no orphaned ownership.
  `tests/test_live_gguf_pool_ownership.py` runs a real INT8-KV MTP session on
  gfx1151: it creates the packed verify workspace, generates, grows the verifier
  scratch to a wider geometry, generates again on the far side of the growth,
  and closes. It asserts the reclaim guard refuses while a decode graph still
  binds the workspace, and that after close the packed verify state, its
  scratch, and the retained prefix snapshot arena pool are all gone with
  `active_allocations` equal to the process baseline (1 passed in 48s). The
  shrink half is a pinned no-op rather than a mechanism: `shrink_idle` returns
  0 and `shrink_events` is fixed at 0.

### HTTP And Library Behavior

- [x] `hipengine serve --kv-storage int8_per_token_head` reaches the intended
  route without a test-only admission override on the supported W7900 artifact.
- [x] Public library speculative methods independently resolve typed intent:
  `generate_speculative_mtp_detailed()` and `stream_speculative_mtp_detailed()`.
  Ordinary `generate()`/`generate_detailed()` remain the AR baseline.
- [x] Blocking completions and chat return correct IDs/text, finish reason and
  prompt/completion usage against their own true no-MTP INT8 baseline.
- [x] SSE streams preserve ordering, usage, terminal events, and cancellation.
- [x] Stopping inside an accepted draft chain publishes no extra tokens.
  Explicit text-stop MTP is served rather than refused, and the cycle commit
  applies the autoregressive finish rule to the whole verified chain -- stop
  token ids, multi-token stop sequences, and the `min_tokens` EOS floor -- by
  selecting its terminal prefix. `scripts/mtp_finish_rule_gate.py` measures it
  on INT8 and BF16 KV.
- [x] Multiple choices, tool/structured responses, and unsupported sampling
  either work through existing contracts or select a named supported fallback.
  `tests/test_live_mtp_http_surface.py` drives a real `hipengine serve` on the
  INT8 KV profile and compares each constrained request against the same request
  with `speculative_mtp: false` (4 passed in 90.8s). `n=2` and a forced tool call
  keep token-exact parity per choice, MTP actually runs (`cycles > 0` against the
  AR arm's 0), and a produced call carries the OpenAI `tool_calls` shape;
  `response_format: {"type": "json_object"}` keeps parity and parses. An
  explicit request that cannot use MTP is not silently downgraded -- it reports
  the route it took in `choices[0].hipengine.diagnostics.specdec2_mtp2`
  (`plan_reason`, `plan_ar_only`, `provider_readiness`,
  `provider_decline_reason`, `cycles`), which is one level below the top-level
  metadata a client sees first.
- [x] GPU waits leave the HTTP event loop responsive.
  `tests/test_unit_server_ready_driver_bound.py` and
  `tests/test_unit_server_ready_degradation.py` bound `/ready` while a driver
  holds the device and pin how it degrades;
  `tests/test_unit_server_mtp_pressure_probe.py` probes responsiveness with a
  real `EngineService` holding its driver thread.
- [x] Busy/rejected/unavailable errors retain correct status and retryability.
  `tests/test_unit_server_error_retryability.py` pins the taxonomy and its
  uncovered edges, and
  `tests/test_unit_generation_execution_failure_containment.py` keeps a failed
  generation from taking the engine down.
- [x] Capabilities and request telemetry expose actual MTP cycles, target route,
  effective storage, and concrete fallback reasons.

### End-To-End Verification

- [x] Direct actual-NextN runs match INT8 AR on all ten category prompts and
  eight heldouts at 24 generated tokens on gfx1151.
- [x] Public HTTP/library category and heldout runs cover the shipping profile,
  streaming/blocking, short/long prompts, and stop/cancel/refill transitions.
  `scripts/int8_mtp_server_gate.py` drives blocking and SSE requests across both
  endpoints and the category and heldout prompt sets, and
  `tests/test_unit_int8_mtp_server_gate_checks.py` pins the gate's own checks.
  It refuses to run unless the effective storage really is
  `int8_per_token_head`, and on an artifact whose compact-INT8 route needs the KV
  quality override it fails unless that override is acknowledged. Stop, cancel
  and refill transitions are covered by the lifecycle and teardown gates
  (`tests/test_unit_int8_mtp_teardown.py`,
  `tests/test_unit_generation_execution_failure_containment.py`) rather than by
  the matrix run itself.
- [x] Repeated schedules are deterministic and isolated from neighboring
  requests; quality is evaluated against the applicable profile contract.
  The same gate re-runs its whole schedule and requires identical ids row by
  row, and runs a neighbouring request beside each target to show the two do not
  disturb each other (41 passed for the module's own checks).
- [x] No regression to BF16 routes, explicit INT8 AR, CLI overrides, or error
  shapes in targeted unit/integration bundles.
- [x] A full user-path run demonstrates the selected route in diagnostics;
  fake engines alone cannot close this item.
- [x] Performance claims, when made, use a same-host true no-MTP AR baseline,
  the full categories plus heldouts, and the benchmark artifact/rollup protocol.
  A claim was made: `benchmarks/results/2026-09-23-gfx1151-qwen38-int8kv-mtp-vs-ar-c1c4.json`
  records this host's INT8 MTP measurement against a matched no-MTP INT8 AR
  baseline over the category prompt set, with the artifact's protocol, cells,
  memory and acceptance fields filled in. Heldout coverage for the claim is the
  category gate's, not the artifact's.

### DMS Extension

- [x] Bind per-head variable spans and their quantized scale planes.
- [x] Journal eviction decisions, token positions, compaction and allocator
  ownership across rejected candidates; cursor reset alone is insufficient.
- [x] Commit only accepted DMS mutations; rollback/cancellation restores the
  complete pre-cycle state, including payload moved by compaction.
- [x] Test real DMS+INT8 MTP against the same DMS+INT8 AR retention policy.
  `tests/test_live_dms_int8_mtp_parity.py` runs both arms as resident
  `Qwen35GGUFResidentSession` rows sharing one policy: the same trained external
  linear sidecar, the same `create_dms_int8_evaluation_backend` factory, and one
  prompt per category from `benchmarks/prompts/mtpbench-code-general-ja.jsonl`
  at a 768-token context, deterministically repeated so the 256-token DMS window
  forces compaction. Each arm decodes 12 tokens; the speculative arm runs three
  cycles of three candidates plus a bonus token, drafted from the AR arm's own
  output, which isolates the retention and transaction axes from draft quality.
  On all four categories the speculative arm reproduces the AR ids and finish
  reason exactly with every cycle fully accepted (`accepted_counts (3,3,3)`),
  and both arms report the same policy outcome: a maximum live count of 525 over
  49792 logical token rows (778 logical tokens across 64 compact rows) and
  `actual_compression_ratio` 1.4857 against a target ratio of 2. A fifth case
  corrupts one candidate to force partial acceptance; the committed sequence
  stays an exact AR prefix, and the policy is compared against an AR arm stopped
  at the same token count, where the live count matches as well. The gate drives
  the serial-exact verify route with `allow_graph=False`, because the packed
  decode graph is BF16-only and cannot serve an INT8 KV DMS row.
  The sidecar this gate ran against is
  `~/dms-artifacts/qwen38-external-v1/sidecar/dms_metadata.json`
  (sha256 `85e84d0068f8f885edbb96468434b49768f8b7d0fcb01895080a131562e0cee0`,
  `artifact_fingerprint` `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169`),
  rebuilt on this host with `scripts/qwen38_dms_train_sidecar.py` and its
  capture/label predecessors. The sidecar the earlier live run used is not on
  this host, so the earlier run's `metadata_sha256` no longer identifies the
  artifact under test.
- [x] Restore the device payload store's live counts with the rest of the
  request state. `DMSDevicePayloadStore.snapshot` captures the per-head live
  counts and `restore` writes them back alongside the payload, position,
  eviction and scale planes. Without them, a commit that keeps only part of an
  appended prefix, or a rollback, left the device counts at their post-append
  values and the next append failed its host/device cross-check with
  `DMS direct append device/host live-count mismatch`.
  `tests/test_gpu_dms_int8_device_payloads.py` asserts the restored live-count
  plane.
- [x] Report a DMS refusal by its actual cause. The former
  `compact_dms_mtp_transaction_not_implemented` decline is gone: the
  transactional verifier selects the DMS store journal whenever the target
  carries a `_dms_backend`, so a missing transaction contract is no longer a
  reason to refuse. On the storage-layout axis a DMS row presents the same
  `"uniform"` layout as dense INT8, which the INT8 MTP declaration already
  covers, so `mtp_kv_layout_unsupported` is not a DMS refusal either. The one
  layout the INT8 chain still refuses is `tail4_hadamard_group32`, a different
  storage layout rather than a DMS retention policy.

## Execution Order

1. Fix default-independent tests and commit them.
2. Implement typed capability and effective-storage diagnostics.
3. Connect the resident C1/wider-owner path and pre-mutation fallbacks.
4. Exercise real public calls, repair lifecycle/prefix/resource defects exposed
   by them, and extend tests across concurrent transitions.
5. Implement remaining packed/DMS contracts as distinct units with their own
   numerical and ownership tests. Never infer completion from dense C1 results.
6. Run the relevant complete validation bundle and update this checklist with
   exact tested scope, remaining concrete limitations, and durable evidence.

## Existing Evidence

The earlier public C1/wider-owner record claimed that all 18 prompts passed on both
endpoints against explicit AR: 108 requests across blocking AR, blocking MTP and
SSE MTP. That claim is not valid for the local gfx1151 artifact: the
`code_lru_cache` row differs by one token (decoded as `LRU Cache` versus
`LRUCache`) under the diagnostic INT8 route. The category gate now reports the
first differing token and fails the row instead of allowing a misleading
108-request summary. Treat the historical run as superseded diagnostic evidence;
it does not qualify INT8 MTP or establish a production correctness claim. The
independent W7900 run on its supported exact Q4_K_M artifact remains separate
evidence and is not affected by this local artifact failure. Current lifecycle
checks still cover automatic intent, mixed neighbors, named errors for explicit
unsupported packed/text-stop requests, disconnect, deadline and reuse.

This host's exact artifact has an existing compact-INT8 quality rejection.
The run explicitly uses the existing KV diagnostic override and reports it
as such; no MTP screening override is required. This is successful HTTP
execution evidence, not reversal of the approximate-KV quality decision.
The independent W7900 run on its supported exact Q4_K_M artifact also passes
the full 108-request matrix and lifecycle gate without diagnostic overrides.
Nine primitive GPU cases pass there. Prefix-on restoration passes all four
categories with actual 512-token cache hits, compact INT8 and AR-matching MTP.
Pressure/eviction and DMS work remain open. The resumable-prefill repair is
integrated: public gfx1151 requests at 2053/4097/6149 tokens now execute MTP and
match AR; mid-prefill deadline/reuse also passes.

Checkpoint lifetime now follows target eviction and adapter shutdown. A target
hit without a usable provider checkpoint is treated as a prefix-cache miss:
normal prefill reconstructs the provider and preserves MTP intent. The final
W7900 library probe confirms output parity, a 512-token hit, and zero outstanding
HIP allocations after close. Dynamic verifier scratch growth belongs to the
persistent session root, and retained target snapshot arenas are closed.

The C1 serving implementation is available; this checklist is not fully closed.
DMS contracts beyond the single-request serial route, unified provider/target
byte budgeting, and the broader pressure/long-context matrix remain separate
open items. Explicit unsupported implementation requests return named errors
rather than being advertised as working MTP.

Commands against an already running INT8 server:

```bash
.venv/bin/python scripts/int8_mtp_server_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --json /tmp/int8-mtp-http-category-suite.json
.venv/bin/python scripts/int8_mtp_server_lifecycle_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --json /tmp/int8-mtp-http-lifecycle.json
.venv/bin/python scripts/int8_mtp_prefix_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --json /tmp/int8-mtp-prefix.json
.venv/bin/python scripts/int8_mtp_prefix_pressure_gate.py \
  --base-url http://127.0.0.1:8098 --no-reuse-base-url http://127.0.0.1:8099 \
  --model int8-mtp --capacity 4 \
  --json /tmp/int8-mtp-prefix-pressure.json
.venv/bin/python scripts/int8_mtp_sampled_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --json /tmp/int8-mtp-sampled.json
.venv/bin/python scripts/mtp_finish_rule_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --expect-storage int8_per_token_head \
  --json /tmp/int8-mtp-finish-rule.json
```

The sampled gate is the live half of the sampled-route declaration: for each of
temperature/top_p, penalty, `logit_bias` and `suppress_token_ids` it requires the
speculative arm to have run real cycles, requires finish reason and usage to
match the request's own true no-MTP baseline, and requires a second seeded run to
reproduce the first. The induced-law half is
`scripts/mtp_sampled_accept_distribution_gate.py`, which measures the sampler law
and the accept coupling on real model rows and needs no server.

The finish-rule gate is the live half of the route's stop semantics. It takes a
seeded free trajectory from the server, then places a stop at each distinct
token of that trajectory and requires both arms to publish exactly that prefix,
to report `stop`, and to publish nothing after the stop; it repeats that for a
multi-token stop sequence, checks that `eos_token_id` fires `eos` at or above
`min_tokens` and withholds the token below it, and checks that a request which
stops early leaves a concurrent neighbour's published ids unchanged. It reads
tokens rather than storage, so it runs against any KV cell: pass
`--expect-storage` to pin the one under test (it refuses a server reporting a
different cell), and run it once per cell. On a server deliberately using an
existing approximate-KV diagnostic override it requires
`--allow-kv-diagnostic-override` as well.

For a server deliberately using an existing approximate-KV diagnostic
override, the category, lifecycle, prefix, pressure and sampled commands require
`--allow-kv-diagnostic-override`. The category
gate rejects BF16 mirrors unless `--allow-mirror` is explicitly requested.
Neither switch enables MTP or changes server policy. The pressure gate needs two
servers of the same capacity, one launched with `--prefix-cache radix` and one
with `--prefix-cache off`; it asserts each server's `/ready` mode rather than
trusting the launch, and `--capacity` must match their `--max-active-requests`
because that value is what bounds the retained working set.

- [Shared-table primitive](../../worklog/entries/20260920T151032.854230Z-lhl-int8-mtp-shared-attention-f8afc4.md)
- [Native verifier checkpoint](../../worklog/entries/20260920T153527.801947Z-lhl-int8-mtp-native-verifier-b8c7e7.md)
- [No-override W7900 HTTP](../../worklog/entries/20260920T182156.489469Z-lhl-int8-mtp-w7900-public-gate-f8951d.md)
- [Long-prefill priming](../../worklog/entries/20260920T182939.759305Z-lhl-int8-mtp-resumable-priming-2f4990.md)
- [Provider restoration](../../worklog/entries/20260920T215105.693007Z-lhl-mtp-provider-prefix-default-a4b133.md)
- [Zero-allocation teardown](../../worklog/entries/20260920T222915.794666Z-lhl-int8-mtp-teardown-owners-3864b7.md)
- `tests/test_unit_int8_verify_attention.py`
- `tests/test_unit_gguf_int8_mtp.py`
- `tests/test_unit_int8_mtp_serving.py`
- `tests/test_gpu_qwen38_int8_batch_attention_gpu.py`
- `tests/test_live_gguf_int8_mtp.py`

Use `uv run --extra dev pytest <explicit test paths> -q` for CPU checks.
GPU/live checks require the installed HIP environment and an available GPU.
Run `tests/test_live_gguf_int8_mtp.py` for the direct transaction/category gate;
it deliberately isolates compact INT8 storage and is not a server certificate.
