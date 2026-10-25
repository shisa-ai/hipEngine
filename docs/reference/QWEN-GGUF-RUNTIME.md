---
status: current
owns: Qwen hybrid GGUF runtime module responsibilities, KV and resident-state storage, scratch/capacity planning boundaries, and remaining execution/diagnostics separation.
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
| `hipengine/runtime/qwen35_gguf_runner.py:Qwen35GGUFResidentSession` | Request/session state, prefill/decode/verification orchestration, pool growth coordination and graph lifetime | Physical KV plane allocation formulas |
| `hipengine/runtime/qwen35_gguf_planning.py` | Geometry/storage-policy scratch plan, resolved KV layout construction, resident byte breakdown and capacity search | Device allocation, loaded executors, environment selection or kernel dispatch |
| `hipengine/runtime/qwen35_gguf_prefill_plan.py` | Prefill field sizes, owner order, lifetime placement and alias views from explicit geometry and resolved route policy | Host arrays, device allocation, environment lookup or kernel selection |
| `hipengine/runtime/qwen35_gguf_state.py` | Canonical root/slot bindings, provisional verifier buffers and reusable rollback storage | Layer execution, graph ordering or speculative cycle decisions |
| `hipengine/runtime/qwen35_gguf_prefix_state.py` | Cache-owned prefix snapshots, arena recycling and copy-failure cleanup | Boundary admission, KV leases or request scheduling |
| `hipengine/runtime/qwen35_gguf_kv.py` | Resolved KV layout validation, backing and plane views, logical/physical byte accounting, allocation and release | Runner/session construction, storage-policy selection, admission budgets, graph scheduling or kernel dispatch |
| `hipengine/runtime/qwen35_gguf_nextn.py`, `hipengine/runtime/qwen35_gguf_mtp.py` | NextN draft execution and transactional speculative integration | Canonical target session ownership |
| `hipengine/runtime/gguf_packed_decode_graph.py`, `hipengine/runtime/gguf_native_spec_cycle.py` | Packed decode graph replay and native speculative-cycle coordination | Public generator construction |

The `qwen35_gguf_runner.py` filename describes only part of its contents. It still
contains session orchestration, scratch allocation, result contracts, hidden-state
capture and diagnostics as well as the full-stack executor. Its module description
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

`qwen35_gguf_prefix_state.py` owns snapshot allocation and copying. Capture takes
explicit per-layer buffers, KV identity metadata and a pool, not a session. The
session checks the current boundary, available continuation capacity, KV coverage
and dirty packed state before calling it. The cache owns each returned snapshot;
closing it returns its arena to the pool or frees it if the pool has closed.
Failed asynchronous copies must retire before their arena can be reused. If
synchronization fails, the pool holds the arena separately until a successful
close retry. Slot teardown cannot close a root's pool.

Session code still owns kernel launches, cursor-device updates, KV bind/unbind
publication and graph lifetime. Prefix clone publishes its cursor only after
required deferred-state flushing succeeds. The numerical copy order and existing
speculative journal interfaces are unchanged.

## Remaining separation

The following are candidates, not implemented modules or a mandate to move code
by line count:

1. **Layer execution:** isolate attention and feed-forward execution only after
   defining their weight, scratch and selected-kernel inputs. Preserve registry
   selection rather than introducing backend-specific dispatch ladders.
2. **Diagnostics:** separate one-layer probes, capture formatting and profiling
   from production contracts without removing callers used by correctness gates.

Do not replace one large class with mixins that all read and mutate the same
implicit attributes. A module boundary is useful when its inputs, lifetime and
failure responsibilities can be stated independently.

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

Most of these tests use fake runtime allocations; the GPU node has an explicit
availability guard. Public real-model generation remains a separate integration
check. This refactor does not change arithmetic, kernel selection or defaults.
