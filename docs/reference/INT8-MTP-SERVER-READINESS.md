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
physical multi-request verifier. An unsupported packed layout must select an
implemented, observable fallback before mutation. It must not kill the engine.

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
- [ ] Sampling, tree masks, mixed layouts, resource failures, and missing
  kernels receive accurate structural reasons before allocation or mutation.
- [ ] Candidate-depth and context coverage derive from implementation capacity;
  no benchmark-only window is introduced.

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
- [x] Packed multi-request INT8 verification, or its explicit pre-mutation
  fallback, preserves accepted-prefix state and does not invoke BF16 readers.
  Current route is the named `packed_int8_mtp_not_implemented` AR fallback,
  not a packed INT8 MTP implementation.
- [ ] Graph reuse after cancellation, pool growth, slot reassignment and scale
  reallocation does not retain stale pointers or another request's state.
- [ ] Cancellation/deadline/failure/shutdown drain target and provider ownership
  exactly once; the next request remains usable.

### Prefix Cache And Memory

- [ ] Prefix reuse copies/shares every INT8 payload and scale plane; append COW
  preserves the cached source.
- [ ] Draft-provider priming/restoration follows prefix reuse, or reports a
  specific supported fallback without corrupting provider position.
- [ ] Prefix hit/miss, eviction under pressure, and reuse after cancellation are
  tested with MTP enabled and disabled.
- [ ] Admission budgets include target KV/scales, draft KV, verifier scratch,
  graphs, and retained prefix ownership; overload fails before HIP OOM.
- [x] Compact mode reports no persistent BF16 mirrors; any mirror mode is
  explicitly reported rather than presented as compact INT8.
- [ ] Pool growth/shrink and final request reclaim leave no orphaned ownership.

### HTTP And Library Behavior

- [ ] `LLM.generate()` and `hipengine serve --kv-storage int8_per_token_head`
  reach the intended route without a test-only admission override.
- [x] Blocking completions and chat return correct IDs/text, finish reason and
  prompt/completion usage against their own true no-MTP INT8 baseline.
- [x] SSE streams preserve ordering, usage, terminal events, and cancellation;
  stopping inside an accepted draft chain publishes no extra tokens.
- [ ] Multiple choices, tool/structured responses, and unsupported sampling
  either work through existing contracts or select a named supported fallback.
- [ ] GPU waits leave the HTTP event loop responsive.
- [ ] Busy/rejected/unavailable errors retain correct status and retryability.
- [x] Capabilities and request telemetry expose actual MTP cycles, target route,
  effective storage, and concrete fallback reasons.

### End-To-End Verification

- [x] Direct actual-NextN runs match INT8 AR on all ten category prompts and
  eight heldouts at 24 generated tokens on gfx1151.
- [ ] Public HTTP/library category and heldout runs cover the shipping profile,
  streaming/blocking, short/long prompts, and stop/cancel/refill transitions.
- [ ] Repeated schedules are deterministic and isolated from neighboring
  requests; quality is evaluated against the applicable profile contract.
- [ ] No regression to BF16 routes, explicit INT8 AR, CLI overrides, or error
  shapes in targeted unit/integration bundles.
- [ ] A full user-path run demonstrates the selected route in diagnostics;
  fake engines alone cannot close this item.
- [ ] Performance claims, when made, use a same-host true no-MTP AR baseline,
  the full categories plus heldouts, and the benchmark artifact/rollup protocol.
  Lack of a speed measurement does not itself prevent feature enablement.

### DMS Extension

- [ ] Bind per-head variable spans and their quantized scale planes.
- [ ] Journal eviction decisions, token positions, compaction and allocator
  ownership across rejected candidates; cursor reset alone is insufficient.
- [ ] Commit only accepted DMS mutations; rollback/cancellation restores the
  complete pre-cycle state, including payload moved by compaction.
- [ ] Test real DMS+INT8 MTP against the same DMS+INT8 AR retention policy.
- [x] Until those operations exist, report the missing DMS transaction contract
  explicitly and retain supported AR behavior.

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

The public C1/wider-owner run uses production-profile Qwen3.8-27B Q4_K_M on
gfx1151 with capacity four, FP32 scales, uniform INT8 and prefix caching off.
All 18 prompts pass on both endpoints against explicit AR: 108 requests across
blocking AR, blocking MTP and SSE MTP. Separate lifecycle checks pass automatic
intent, mixed neighbors, n=2 fallback, stop, disconnect, deadline and reuse.

This host's exact artifact has an existing compact-INT8 quality rejection.
The run explicitly uses the existing KV diagnostic override and reports it
as such; no MTP screening override is required. This is successful HTTP
execution evidence, not reversal of the approximate-KV quality decision.
The no-override supported-artifact run, prefix-on validation, and DMS work
are still open.

Commands against an already running INT8 server:

```bash
.venv/bin/python scripts/int8_mtp_server_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --json /tmp/int8-mtp-http-category-suite.json
.venv/bin/python scripts/int8_mtp_server_lifecycle_gate.py \
  --base-url http://127.0.0.1:8098 --model int8-mtp \
  --json /tmp/int8-mtp-http-lifecycle.json
```

For a server deliberately using an existing approximate-KV diagnostic
override, both commands require `--allow-kv-diagnostic-override`. The category
gate rejects BF16 mirrors unless `--allow-mirror` is explicitly requested.
Neither switch enables MTP or changes server policy.

- [Shared-table primitive](../../worklog/entries/20260920T151032.854230Z-lhl-int8-mtp-shared-attention-f8afc4.md)
- [Native verifier checkpoint](../../worklog/entries/20260920T153527.801947Z-lhl-int8-mtp-native-verifier-b8c7e7.md)
- `tests/test_unit_int8_verify_attention.py`
- `tests/test_unit_gguf_int8_mtp.py`
- `tests/test_gpu_qwen38_int8_batch_attention_gpu.py`
- `tests/test_live_gguf_int8_mtp.py`

Use `uv run --extra dev pytest <explicit test paths> -q` for CPU checks.
GPU/live checks require the installed HIP environment and an available GPU.
Run `tests/test_live_gguf_int8_mtp.py` for the direct transaction/category gate;
it deliberately isolates compact INT8 storage and is not a server certificate.
