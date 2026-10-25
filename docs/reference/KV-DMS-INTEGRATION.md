---
status: current
owns: Production KV and compact-DMS integration, request/session ownership, adapter boundaries, and remaining integration limits.
---
# KV and DMS integration

See [Architecture](../ARCHITECTURE.md) for subsystem navigation and
[DMS serving](DMS-SERVING.md) for user configuration. This reference distinguishes
production integration from shared interfaces that are not universal entry points.

## Owners and actual consumers

| Owner | Contract | Production consumer or boundary |
| --- | --- | --- |
| `hipengine/kvcache/backend.py` | Named resource claims, leases, storage views, `KVCacheBackend` | Shared contract; not proof that every family runner delegates through it |
| `hipengine/kvcache/pool.py`, `hipengine/kvcache/device_global.py` | Dense page ownership, arbitrary-page tables, workspace leases and pool growth | Qwen GGUF resident session/adapter integrates these with model-specific planes |
| `hipengine/kvcache/ledger.py`, `hipengine/kvcache/global_pool.py` | Resource accounting and format-neutral pool machinery | Shared backend/admission implementations and contract tests |
| `hipengine/kvcache/spans.py` | `KVLiveSpans`: offsets, live counts, token positions and eviction mask | Attention/KV-write liveness interface; dense retention fills uniform spans, compact retention varies them |
| `hipengine/runtime/qwen35_gguf_kv.py` | Physical Qwen KV layout, plane accounting, backing allocation/release | Dense session allocation and pool growth; no scheduler or retention policy |
| `hipengine/runtime/qwen35_gguf_planning.py` | Shared geometry/policy scratch and capacity plan | Real scratch allocator and serving capacity estimation |
| `hipengine/kvcache/dms.py:DMSCompactBackend` | Compact extent reservation, live state, append/pack, transactions and reclaim | Qwen session prefill/decode and DMS transaction integration |
| `hipengine/kvcache/dms_device.py` | Device compact payload/position storage and staging | Compact backend/session execution |
| `hipengine/kvcache/dms_sidecar.py` | Sidecar loading, projection and retention decisions | External-DMS session collector and decode decision path |
| `hipengine/kvcache/dms.py:DMSAdmissionManager`, `DMSCompactResidentRunnerAdapter` | Generic runner integration through `KVBatchView` | Tested library adapter; not the Qwen public adapter's entry point |
| `hipengine/runtime/dms_prefix.py`, `hipengine/runtime/dms_prefill_cache.py` | Owned compact prompt snapshots and partial-prefill reuse | Metadata-path Qwen DMS serving; not dense radix-page sharing |
| `hipengine/generation/qwen35_gguf.py` | Shared registration, admission, prefill/decode dispatch and request retirement | Ordinary dense KV and both Qwen DMS entry points |
| `hipengine/generation/qwen35_gguf_dms.py` | Typed API restrictions, request-session acquisition and typed observability | `DMSConfig`; delegates execution/lifecycle rather than duplicating it |

`DMSCompactBackend` implements the resource/view/transaction methods in the shared
backend contract. The production Qwen session nevertheless owns its integration:
it creates the backend, reserves the compact request, invokes pack/append and
reclaims the backend on session close. Instantiating the generic wrapper is not
an alternative spelling for this existing path; its runner-view interface must
be integrated and tested before replacing production calls.

## Two public selections, one configuration source

`configure_dms` records one canonical generator sidecar path for either public
selection. A typed `DMSConfig` also records its requested prefill and decision
modes. `_prepared_session_kv_kwargs` is the single source of session DMS options,
including codec factory. Supplying `dms` and `dms_metadata_path` together is a
configuration error at the LLM/server boundary, not a late duplicate-key error.

Both paths use shared session acquisition for policy binding, automatic context
selection, capacity reporting and generator session/profile configuration. The
typed adapter uses a distinct acquisition pool name, but compact sessions are
closed on retirement and never returned as reusable dense slots. Shared weights
outlive those request sessions.

The entry points retain different supported compositions; normalization does not
silently enable features. See the [selection table](DMS-SERVING.md#entry-points-and-supported-compositions).

## Request lifecycle

1. **Register:** validate the complete request cohort before publishing rows or
   advancing the batch ID. Context validation uses prepared available or busy
   sessions; it does not depend on the existence of a dense pool. Apply the
   compact store's effective output limit before computing required positions.
   Unsupported prompts shorter than the convolution width fail here: the current
   DMS capture implementation requires bulk prefill, whose minimum is declared by
   the model geometry. Zero-output requests do not claim KV.
2. **Admit:** the scheduler owns admission timing. Dense retention reserves and
   binds a pool allocation. Compact retention assigns an exclusive session lease
   to the row without binding a dense global pool. A duplicate claim is an error.
3. **Prefill:** scheduler chunks retain normal request accounting. The compact
   capture route executes at the complete-prompt boundary and passes explicit
   bulk semantics. Sidecar collection and packing are session/backend work, not
   another queue. Deadlines bracket the device call.
4. **Decode/commit:** compact decode consumes its retention state and live spans.
   The dense fixed-address graph path is not substituted for changing compact
   extents. The scheduler publishes completed tokens through the existing path.
5. **Cancel/reclaim:** common row cleanup releases suspended work, prefix and
   graph references, then retires only the row's lease. An unadmitted row owns no
   lease and cannot close a neighbor's or idle session.
6. **Retire:** a reusable dense session resets and returns to the available list
   after its allocation detaches. A compact session closes: reset cannot recreate
   released dense-prefill planes or erase compact retention history. No
   reset-then-close-all pass remains in the typed adapter.
7. **Shutdown:** active rows retire, remaining available sessions close, then the
   generator releases shared model/provider assets.

The engine request ID and compact-store row ID are different namespaces. A
singleton Qwen session uses `dms_store_request_id == 0`; provider journals map the
engine request to that session-local owner rather than using the engine ID as a
compact row index.

## Speculative and prefix boundaries

The metadata-path route has its own DMS prompt cache and existing transactional
MTP integration. Its prefix-cache setting selects owned compact snapshots, not
the dense radix cache's page leases. Snapshots can outlive the request session;
the cache owns their release. Provider checkpoints must refer to the same target
prefix and legal hidden-state boundary.

Speculative commit spans more than compact KV. The target's recurrent state,
provider state, positions, output selection and sampling state must agree at the
commit point. `hipengine/speculative/transaction.py` owns shared checkpoint-journal
lifetime where applicable; DMS append/store journals and family adapters supply
their concrete restore/commit operations. The
[shared speculative consumer matrix](SHARED-SPECULATIVE-OPTIMIZATION.md#source-backed-consumer-matrix)
identifies implementations that use shared journals versus owner-local packed
transactions. Do not layer a second undo owner over an already transactional path.

## Verification and limits

`tests/test_unit_dms_request_ownership.py` checks early refusal, atomic cohort
publication, admitted versus queued cancellation, fresh session reuse, duplicate
admission, zero output, canonical configuration, shared auto-context acquisition
and equal output-limit selection across the two entry points.

Adjacent contracts live in `tests/test_unit_dms_serving.py`,
`tests/test_unit_dms_prefix.py`, `tests/test_unit_dms_prefill_cache.py`,
`tests/test_unit_dms_cycle_transaction_journal.py`,
`tests/test_unit_qwen35_gguf_dms_output_clamp.py` and
`tests/test_unit_gguf_admission_context.py`. Fake-runtime tests establish ownership,
not attention accuracy or trained retention quality.

Public lifecycle validation can use a geometry-compatible synthetic sidecar with
an explicit untrained-fixture label. Such a run can establish routing,
cancellation/refill, refusal recovery and resource release. It cannot establish
learned retention quality; that needs a real trained sidecar and its quality
protocol. `tests/test_live_dms_public_serving.py` names the trained-sidecar lane's
resources and skips when they are absent.

## Remaining boundaries

- The typed API still exposes BF16, one active request, no prefix reuse and no
  speculative serving. The metadata-path integration is a distinct existing
  composition; do not infer every cross-product is supported.
- Generic `KVCacheBackend` wrappers are not uniformly adopted by model runners.
  A future migration must replace a real integration, not add another adapter
  alongside the current one.
- DMS prefill remains a complete-prompt capture operation. Cancellation between
  internal layers, packed multi-request DMS and dense-graph reuse need concrete
  implementations; documentation or a protocol alone does not provide them.
- Prefill scratch pricing still has execution-backed sizing callbacks. Canonical,
  provisional and cached recurrent-state ownership is the next broader runner
  boundary, separate from physical KV layout.
- Historical measurements in [KVCACHE.md](KVCACHE.md),
  [DMS.md](DMS.md) and [CONCURRENCY2.md](CONCURRENCY2.md) describe their dated
  configurations. Their old product-status paragraphs do not override this
  production integration map or current capability checks.
