---
status: current
owns: Architecture navigation, production subsystem ownership, dependency boundaries, and the distinction between kernel selection and request lifecycle.
---
# Architecture and subsystem boundaries

Start here to find the owner of a change. [PLAN.md](PLAN.md) owns architectural
invariants and the roadmap; this page describes the implemented boundaries and
links to their contracts. Kernel selection is only one boundary: request
scheduling, cache ownership and speculative transactions have separate owners.

## Kernel selection versus model and subsystem composition

The kernel registry key is **`(backend, layer, quant, variant)`**:

| Key component | Selects | Source |
| --- | --- | --- |
| Backend | Hardware implementation and launch support | `hipengine/kernels/backends.py`, peer backend trees |
| Layer | Operation or fused operation family | `hipengine/kernels/registry.py`, layer registrations |
| Quant | Operand representation and supported numerical format | Quant plugins, loading contracts and kernel registration |
| Variant | Implementation within the other three coordinates | Kernel registry plus execution-profile selection |

Model plugins separately declare model geometry, layer sequence, tensor mapping
and generation factories. A model family is not a fifth kernel-key field.
Execution profiles select variants; they are not another registry axis either.
See [PLAN: extensibility](PLAN.md#extensibility-design) and
[execution profiles](EXECUTION-PROFILES.md).

These keys answer **which implementation can execute an operation**. They do not
own admission queues, cache leases, cancellation, retained prefixes or speculative
commit points. Those are subsystem contracts. Do not encode a lifecycle policy as
a new kernel variant or add model/backend branches to the scheduler.

## Production request path

```text
LLM / HTTP transport
  -> model-selected generator and construction owner
  -> EngineService -> SubmitPollTextGenerator -> ResidentEngineLoop
  -> BatchScheduler and model-family resident adapter
  -> model executor + session/state owners
  -> resolved kernel implementations

Resident adapters coordinate with:
  KV storage/retention owners      reserve, bind, publish, reclaim
  speculative target/provider     propose, verify, accept, commit or rollback
  sampling and finish handling    request-local randomness, stops and outputs
```

This is an ownership map, not a promise that every family uses identical class
names or every request executes every stage. Streaming, speech and multimodal
families expose additional boundaries. In particular, "runner" can name either
a scheduler-facing resident adapter or a lower-level model executor; inspect its
callers rather than inferring ownership from the filename.

## Subsystem map

| Subsystem | Owns | Must delegate | Start here |
| --- | --- | --- | --- |
| Public API and transport | Request validation, HTTP errors, transport backpressure, disconnect signals | Device scheduling and model state | `hipengine/llm.py`, `hipengine/server/api.py`; [API](API.md) |
| Construction | Temporary ownership until generator, driver and service publication; setup-error cleanup | Model loading and declared capabilities | `hipengine/generation/construction.py`; [construction evidence](campaigns/MAIN-PATH-OWNERSHIP-EVIDENCE.md) |
| Generation engine | Request identity, scheduling, cancellation, outputs and scheduler commit boundaries | Format-specific memory pricing, layer arithmetic and draft execution | `hipengine/generation/engine_service.py`, `hipengine/generation/engine_loop.py`, `hipengine/generation/batch_scheduler.py`; [Generation 2](reference/CONCURRENCY2.md) |
| Model/loading/quant | Architecture, tensor mapping, materialization and invocation contracts | Request queues and cross-request state | `hipengine/models/`, `hipengine/loading/`, `hipengine/quant/`; [PLAN](PLAN.md#extensibility-design) |
| Family runtime | Weights, layer execution, request/session state and model-specific transaction operations | Shared admission, lease accounting and speculative-cycle policy | `hipengine/runtime/`; [Qwen runtime boundaries](reference/QWEN-GGUF-RUNTIME.md) |
| KV storage and retention | Pools, leases, live spans, payload layouts, retention state, prefix references and reclamation | Attention computation and public queues | `hipengine/kvcache/`; [KV/DMS integration](reference/KV-DMS-INTEGRATION.md) |
| Speculative decoding | Candidate/frontier contracts, cycle accounting, shared checkpoint/graph ownership and capability-bounded depth | Provider arithmetic, hidden alignment and target-state copy/restore | `hipengine/speculative/`; [shared consumers](reference/SHARED-SPECULATIVE-OPTIMIZATION.md), [SPECDEC2](reference/SPECDEC2.md) |
| Sampling and finish | Per-request sampling history/randomness, constraints, stop/EOS and output accounting | Model logits and device implementations | `hipengine/generation/sampling.py`, `hipengine/generation/finish.py`, `hipengine/speculative/sampling.py` |
| Device/core | Device buffers, streams, graph primitives, copies and compiler/build interfaces | Model policy and request admission | `hipengine/core/`; [testing contracts](TESTING.md) |

The shared engine does not replace HTTP backpressure: the transport can reject or
queue clients while the scheduler separately admits device work. Conversely,
BF16, INT8 and DMS storage must not each introduce another request scheduler.

## State and lifetime boundaries

| Lifetime | Typical resources | Release/transfer boundary |
| --- | --- | --- |
| Model/load | Materialized weights, shared provider uploads, selected implementations | Generator/runtime shutdown after their borrowers |
| Resident execution | Prepared scratch, slot views, reusable graph buckets, dense pool backing | Resident owner teardown; not necessarily every request completion |
| Request | Lease, logical position, compact retention state, sampling history | Completion, cancellation or contained failure reclaims only that request |
| Work item | Transient prefill/verification buffers and resource claims | Completion or failure of the work that acquired them |
| Transaction | Provisional target/provider state, append journals and selected-row state | Accepted commit or rollback; publication follows successful state operations |
| Prefix cache | Detached snapshots, referenced immutable pages or owned compact payloads | Cache eviction/pressure/close, independently of the source request |

A speculative transaction is larger than KV: recurrent/convolution state,
provider caches, position cursors and sampling state can also participate.
`hipengine/speculative/transaction.py` provides shared records and completed-
checkpoint journal ownership. Model adapters supply legal checkpoint/restore
operations; they must not use dense append truncation for a state that needs a
real recurrent-state restore. Packed routes can retain their own transaction
owner instead of acquiring a second undo journal.

## What is shared, and what is not

- **KV:** shared pools, resource claims, prefix ownership and `KVLiveSpans` are
  real reusable machinery. Physical layouts and model state remain specialized.
  `KVCacheBackend` and the generic dense/DMS resident wrappers are tested library
  contracts, but the Qwen production route does not uniformly enter through those
  wrappers. Its sessions integrate the compact backend directly. See the
  [production consumer map](reference/KV-DMS-INTEGRATION.md).
- **DMS:** retention decisions change live positions and compact extents; DMS is
  not just an INT8 codec. The sidecar/model geometry contract and compact-store
  operations remain separate from scheduler identity and session retirement.
- **Speculation:** prompt chunk validation, completed checkpoints, graph-cache
  lifecycle, packing, acceptance/accounting and depth planning can be shared.
  NextN, assistant heads and DFlash still own different hidden/cache semantics.
  The [consumer matrix](reference/SHARED-SPECULATIVE-OPTIMIZATION.md#source-backed-consumer-matrix)
  records actual adopters and non-applicable mechanisms. A module's existence is
  not evidence that every provider uses it.
- **Model execution:** Qwen hybrid GGUF/PARO, Qwen4 experimental, Gemma, Laguna
  and other families retain separate executors. Share a component only after its
  inputs, state ownership and numerical contract match; do not replace them with
  a universal runner full of family branches.

## Where to put a change

1. Identify whether it changes implementation selection, request policy,
   resource lifetime, storage semantics or model arithmetic.
2. Find the owner in the table; trace one production caller and its teardown.
3. For sharing, identify at least two compatible consumers and the old duplicate
   path to remove. Prefer explicit inputs and composition to mixins with hidden
   shared attributes.
4. Test refusal before publication, cancellation/neighbor isolation, partial
   failure, commit/rollback and cleanup as applicable. Exercise the public route;
   source-only tests or simulator coverage do not establish serving integration.
5. Keep numerical changes separate from ownership refactors. Run the relevant
   [testing](TESTING.md) and [optimization gates](OPTIMIZATION.md) when applicable.

## Scope of the current cleanup

The Qwen cleanup separated physical KV storage and scratch/capacity planning,
then integrated DMS configuration, session acquisition, early refusal and request
retirement with the existing Qwen lifecycle. It did not implement packed DMS,
make the typed DMS API support every metadata-path composition, unify every model
runtime, or replace all production adapters with `KVCacheBackend` wrappers.
Remaining boundaries and validation limitations are recorded in
[KV/DMS integration](reference/KV-DMS-INTEGRATION.md#remaining-boundaries).
