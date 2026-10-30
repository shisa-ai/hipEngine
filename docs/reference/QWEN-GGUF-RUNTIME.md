---
status: current
owns: Qwen hybrid GGUF runtime module responsibilities, layer execution inputs, diagnostic capture and timing, KV and resident-state storage, and scratch/capacity planning.
---
# Qwen hybrid GGUF runtime structure

`qwen35` names the GGUF architecture family, not a restriction to a model release.
`hipengine/loading/qwen35_gguf.py:qwen35_gguf_config_from_metadata` accepts `qwen35` and
`qwen35moe`; newer model releases using those identifiers use the same family.
Qwen4 experimental models have a separate runtime and are not covered here.

Paths in this document are repository-relative.

## Responsibilities

| Owner | Responsibility | Does not own |
| --- | --- | --- |
| `hipengine/generation/qwen35_gguf.py` | Public generator configuration, resident model adapter, request admission and serving assets | Individual kernel arithmetic |
| `hipengine/runtime/qwen35_gguf_runner.py:Qwen35GGUFFullStackRunner` | Loaded target weights, selected execution primitives and full-stack layer execution | Public request scheduling |
| `hipengine/runtime/qwen35_gguf_runner.py:Qwen35GGUFResidentSession` | Request/session state, prefill/decode/verification orchestration, pool growth coordination and graph capture/synchronization | Physical KV plane allocation formulas or graph pin bookkeeping |
| `hipengine/runtime/qwen35_gguf_graph_lifetime.py` | Decode-handle registry, KV pin accounting and retained submission-context closure over explicit resources | Session mutation, capture policy, kernel selection or synchronization |
| `hipengine/runtime/qwen35_gguf_layer_attn.py` | Scalar linear/full attention and shared norm/alpha-beta leaves over explicit weights, geometry, views and selected operations | Request admission, registry policy, graph lifetime or bulk attention orchestration |
| `hipengine/runtime/qwen35_gguf_layer_ffn.py` | Dense and MoE feed-forward execution for scalar and bulk rows over explicit weights, scratch and selected operations | Session state, speculative acceptance or kernel-selection policy |
| `hipengine/runtime/qwen35_gguf_capture.py` | Host diagnostic snapshots, summary schemas and selected-buffer readback | Token setup, layer execution, session mutation or synchronization |
| `hipengine/runtime/qwen35_gguf_timing.py` | Stage timing accumulation, event handles and wall-clock marker storage | Choosing interval boundaries or scheduling production work |
| `hipengine/runtime/qwen35_gguf_probe.py` | Standalone one-layer projection diagnostic and its temporary resources | Full-model generation or resident-session ownership |
| `hipengine/runtime/qwen35_gguf_results.py` | Shared next-token result contract | Probe execution or device resources |
| `hipengine/runtime/qwen35_gguf_planning.py` | Geometry/storage-policy scratch plan, resolved KV layout construction, resident byte breakdown and capacity search | Device allocation, loaded executors, environment selection or kernel dispatch |
| `hipengine/runtime/qwen35_gguf_prefill_plan.py` | Prefill field sizes, owner order, lifetime placement and alias views from explicit geometry and resolved route policy | Host arrays, device allocation, environment lookup or kernel selection |
| `hipengine/runtime/qwen35_gguf_state.py` | Canonical root/slot bindings, provisional verifier buffers and reusable rollback storage | Layer execution, graph ordering or speculative cycle decisions |
| `hipengine/runtime/qwen35_gguf_prefix_state.py` | Cache-owned prefix snapshots, arena recycling and copy-failure cleanup | Boundary admission, KV leases or request scheduling |
| `hipengine/runtime/qwen35_gguf_kv.py` | Resolved KV layout validation, backing and plane views, logical/physical byte accounting, allocation and release | Runner/session construction, storage-policy selection, admission budgets, graph scheduling or kernel dispatch |
| `hipengine/runtime/qwen35_gguf_nextn.py`, `hipengine/runtime/qwen35_gguf_mtp.py` | NextN draft execution and transactional speculative integration | Canonical target session ownership |
| `hipengine/runtime/gguf_packed_decode_graph.py`, `hipengine/runtime/gguf_native_spec_cycle.py` | Packed decode graph replay and native speculative-cycle coordination | Public generator construction |

The `qwen35_gguf_runner.py` filename describes only part of its contents. It still
contains session orchestration, scratch allocation, result contracts and
stateful capture setup as well as the full-stack executor. Its module description
now identifies production execution and session orchestration rather than
bring-up probes. Keeping the import path avoids a repository-wide rename before
these responsibilities have separate owners.

## Physical KV storage boundary

The execution module resolves a storage policy into `Qwen35GGUFKVChunkLayout`.
It checks executor lifetime and hands only the model configuration, resolved
layout and runtime to the storage allocator. The allocator does not receive
weights, a runner, a session or a model identity.

The storage module owns:

- `Qwen35GGUFKVChunkLayout`: per-layer dtype, scale and mirror invariants.
- `Qwen35GGUFKVChunkBacking`: payload/scale planes, page addressing and layerwise
  release of no-longer-needed full-attention backing.
- `VmmPlaneBuffer`: a logical plane view over a virtual-memory arena owner.
- Page/plane sizing and granularity-aligned physical commitment accounting.
- Ordinary and virtual-memory allocation, backing assembly and partial-allocation
  rollback, followed by release through the allocating runtime.

Session code still owns decisions about when to grow a pool, whether a request
fits, publishing pointer tables and descriptors, and rollback of a failed growth
transaction. Storage allocation alone does not publish a request or mutate its
logical page ownership. The existing `KVLiveSpans` kernel interface is unchanged.

The old layout/backing type imports from `qwen35_gguf_runner` are aliases to the
storage definitions, not duplicate classes. Existing `isinstance` and serialized
class lookup remain supported. Private allocator functions now take configuration
rather than a runner. Tests that replace allocation/free functions patch the
storage module for plane memory and the execution module for execution scratch;
patching the executor's allocator no longer intercepts both responsibilities.

## Scratch and capacity planning boundary

`hipengine/runtime/qwen35_gguf_planning.py` owns `_FullStackScratchPlan` and its builder,
`qwen35_gguf_resident_breakdown`, and `estimate_qwen35_gguf_kv_capacity`.
Both `_FullStackScratch.allocate` and capacity estimation consume the same plan;
there is no second set of allocation formulas in the estimator.

Inputs are model configuration, explicit geometry, storage policy, context and
slot counts. Planning does not take a loaded executor, allocate device memory,
read environment selectors or choose kernels. `materialize=False` returns sizes
with every host-array field set to `None`; `materialize=True` builds each set of
host metadata and rotary-position tables once for allocation. The ordered owner
sizes, alignment, mirror threshold and byte-accounting rules are shared.

`hipengine/runtime/qwen35_gguf_prefill_plan.py` separately owns the prefill layout.
Its immutable plan records ordered device owners, physical views, logical offsets,
lifetimes and dense in-place aliases. Dedicated, single-arena and owner-slot
layouts use the same plan for allocation and exact byte estimation. Disabled
planes own no memory; dedicated metadata and optional KV caches remain separate
owners. The planner neither materializes host arrays nor imports a device runtime.

The execution module resolves route policy before planning, then allocates the
owners and uploads metadata. One failure handler releases acquired owners once.
`_gguf_prefill_scratch_nbytes` sums the shared plan rather than executing the
allocator against fake memory or interpolating between capacities. Block tables
and split buffers follow rounded block counts; rotary tables follow prefill rows.
No object-identity cache can reuse a stale policy or geometry result. DMS head-major
KV sizing also shares its admission/size helper with the real allocator.

Serving capacity adapters resolve the same architecture, geometry and memory
chunk policy as session construction before choosing prefill rows. The plan prices
initial owners, not later split-query growth; workspaces track those additional
owners separately. Context-dependent layer selection and speculative workspace
costs still enter capacity search through caller-supplied callbacks.

Existing imports from `qwen35_gguf_runner` are aliases to the planning definitions.
Serving capacity consumers import the planning owner directly. Diagnostics that
patch the private mirror threshold must patch both execution policy and planning;
the two modules must price and execute the same diagnostic layout.

## Resident state and transaction storage

`qwen35_gguf_state.py` separates storage lifetime from session orchestration:

- `CanonicalResidentState` holds the scratch owner, active scratch view, cursor
  and KV binding. Only the root releases scratch owners; a resident slot borrows
  its scratch and has an independent cursor and KV lease. KV pools retain
  physical backing ownership.
- `VerifyBlockStorage` and `LinearStateStorage` own provisional verifier rows
  and producer initial-state snapshots. Growth allocates a complete replacement
  before publication and old-owner release. Allocation failure releases partial
  replacements and preserves the previous buffers and capacity.
- `LinearRollbackStorage` owns reusable device backups over an explicit borrowed
  `CanonicalLinearState`. It does not decide acceptance or create another journal:
  existing speculative transactions still order capture, commit and restore.
  Active producer captures prevent initial-state buffer replacement or release.
  Verifier construction releases its journal if later setup fails. Session close
  checks root and slot capture leases before mutating resources, so an invalid
  close order leaves the owners intact for a correctly ordered retry.

Session properties preserve existing field names as accessors into these owners,
not duplicate values. Slot construction creates separate local storage and never
inherits the root's prefix arena pool. Root and slot teardown use the same local
verifier cleanup; packed-workspace state remains deliberately shared across slots.
Graph invalidation and synchronization precede publication of packed-workspace
replacements and release of their old buffers.

Packed verification's captured/deferred initial-state binding uses a one-slot
view for a singleton, including a nonzero borrowed resident slot. Unsegmented
Conv/GDN leaves read their pointer bases directly, so this view supplies both
the correct offset and per-slot byte extent; its local state index is zero.
Multi-session segmented leaves instead consume slab pointers plus explicit
state indices. Candidate state rows remain provisional and do not mutate any
resident slot before acceptance commits.

`qwen35_gguf_prefix_state.py` owns snapshot allocation and copying. Capture takes
explicit per-layer buffers, KV identity metadata and a pool, not a session. The
session checks the current boundary, available continuation capacity, KV coverage
and dirty packed state before calling it. The cache owns each returned snapshot;
closing it returns its arena to the pool or frees it if the pool has closed.
Failed asynchronous copies must retire before their arena can be reused. If
synchronization fails, the pool holds the arena separately until a successful
close retry. Slot teardown cannot close a root's pool.

Session code still owns kernel launches, cursor-device updates, KV bind/unbind
publication and graph capture/synchronization. `qwen35_gguf_graph_lifetime.py`
owns graph handles, KV pins and retained submission contexts. Pin/unpin receive
current allocation and pool explicitly; slot views have independent bookkeeping.
Session compatibility properties expose the same mutable containers until capture
sites migrate to owner methods. Prefix clone publishes its cursor only after
required deferred-state flushing succeeds. The numerical copy order and existing
speculative journal interfaces are unchanged.

## Layer execution inputs

`qwen35_gguf_layer_attn.py` owns scalar linear and full attention, including the
shared attention-norm and alpha/beta projection leaves. `qwen35_gguf_layer_ffn.py`
owns the feed-forward schedule used by scalar decode, bulk prefill and verifier
rows. Dense, scalar MoE and multi-row MoE keep their separate execution schedules.

The functions receive resolved layer weights, model configuration, geometry,
scratch or per-layer views, runtime, and selected operations. They do not receive
a runner or resident session. Lazy selectors preserve library and variant
resolution at the point of use; the runner retains registry policy and its caches.
Existing runner methods delegate to these functions rather than keeping a second
copy of their arithmetic.

Full-attention position updates and split-workspace growth remain explicit
operations over borrowed session scratch. DMS-owned attention is selected before
ordinary full-attention views are built. The session still owns that DMS route,
KV publication and graph lifetime. Bulk attention and layer composition remain in
the runner; moving the shared feed-forward half does not move those schedules.

Profiler markers for nested MoE calls wrap the execution module's functions.
The runner's outer attention methods remain the scalar attention marker boundary.
Instrumentation must patch the owner that actually executes, not assume that an
internal call passes through a compatibility method.

## Diagnostics and timing

`qwen35_gguf_capture.py` owns diagnostic snapshots, their summary dictionaries and
host readback over explicit configuration, dimensions, pointers, scratch and copy
operations. It does not import the runner, launch kernels or mutate a session.
Session capture methods retain input validation, embedding and position setup,
preceding-layer execution, state mutation and stream synchronization. Only then do
they delegate buffer copying and report construction. Router tracing still captures
its input before each layer executes and captures its output afterward.

`qwen35_gguf_timing.py` owns event and device wall-clock recorder resources. The
production caller still places each marker and decides when to resolve timings.
Disabled recorders do not allocate or synchronize. Event timing and wall-clock
timing remain separate measurements; aliases and negative-event accounting retain
their existing meanings. Diagnostic tools patch the timing module's primitives,
not the runner's allocator, when intercepting recorder storage.

`qwen35_gguf_probe.py` owns the standalone partial projection probe, including its
weight and temporary-buffer lifetime. It is not a full attention layer or a
production generation route. It declares no expert-call intents when loading its
projection-only selection, so MoE artifacts do not require unused expert partners.
`qwen35_gguf_results.py` owns the next-token result
shared by the probe and full-model execution. Legacy runner imports are aliases,
not duplicate implementations. Source-audit tools inspect both the session's
reader binding and the capture module before reporting the host-copy dtype. A
copied runner source must include the matching capture module beside it; the
source audit raises an error if that required source is missing.

## Scope of the remaining runner

Bulk attention orchestration and full-stack composition remain in the runner;
their distinct schedules are not folded into scalar attention. Session mutation,
graph capture/replay, production hidden-seed sinks and diagnostic execution setup
remain beside the state they control. These are not replaced with a diagnostic
proxy or mixins that share implicit session attributes.

## Validation targets

`tests/test_unit_qwen35_gguf_kv_storage.py` protects configuration-only allocation,
exact allocation order and byte counts, partial-plane rollback, lifetime checks,
type identity and the direct dependency boundary. The storage module must not
import execution, generation or kernel modules.

The existing suites cover session binding and integration:

```bash
uv run pytest \
  tests/test_unit_qwen35_gguf_kv_storage.py \
  tests/test_unit_gguf_device_kv_binding.py \
  tests/test_unit_gguf_vmm_kv_pool.py \
  tests/test_unit_gguf_global_pool_growth.py \
  tests/test_unit_gguf_packed_workspace_stability.py \
  tests/test_unit_gguf_packed_verify_layout.py \
  tests/test_gpu_dms_layerwise_kv_release.py -q
```

`tests/test_unit_qwen35_gguf_scratch_plan_boundary.py` covers size-only versus
materialized planning, byte equality, owner order, mirror thresholds, precision
and invalid policies. `tests/test_unit_qwen35_gguf_planning_integration.py` verifies
that allocation and estimation use the same planner and that planning has no
direct execution/device-allocation dependency. Existing capacity tests compare
planned sizes with the actual allocator using fake device buffers.

`tests/test_unit_qwen35_gguf_prefill_plan.py` covers the pure prefill layout;
`tests/test_unit_qwen35_gguf_prefill_plan_integration.py` checks exact estimator
bytes at block and route boundaries, resolved chunk policies, policy changes and
absence of size-only host/device materialization. The allocation-failure tests in
`tests/test_unit_qwen35_gguf_prefill_scratch_cleanup.py` require each acquired
owner to be released once after allocation or upload failure across all layouts.
Existing prefill liveness and packed-workspace tests protect aliases, metadata
and stable workspace ownership.

`tests/test_unit_qwen35_gguf_state_ownership.py` injects first, middle and last
allocation failures into real session methods. `tests/test_unit_qwen35_gguf_state_storage.py`
checks canonical borrowing, rollback replacement, active capture leases and clone
publication failure. `tests/test_unit_qwen35_gguf_prefix_state.py` covers arena
capture, reuse, copy failures, synchronization failure and delayed cleanup.
Existing prefix, packed-workspace and MTP tests exercise their integrated consumers.

`tests/test_unit_qwen35_gguf_layer_execution.py` checks the independent execution
boundary, launch order and arguments, minimal leaf inputs, branch-dependent
selectors, split-buffer growth and DMS override. Existing dispatch, mixed-precision
and selected-expert tests exercise the runner adapters. Projection-level probes
patch the attention module's launch symbols; FFN selected-operation tables retain
the runner's launch symbols. `tests/test_unit_gguf_decode_rocprof.py` checks marker
ownership and restoration for both current and historical execution owners.

`tests/test_unit_qwen35_gguf_capture.py` checks readback order, shapes, precision,
optional dense/MoE and linear/full-attention inputs, type aliases and source-audit
bindings. Existing hidden-seed and capture-script tests protect session mutation,
preceding-layer setup and artifact schemas. Timing and probe unit tests use fake
runtime operations to check resource failure paths without requiring a GPU.

Most of these tests use fake runtime allocations; the GPU node has an explicit
availability guard. Public real-model generation remains a separate integration
check. This refactor does not change arithmetic, kernel selection or defaults.
