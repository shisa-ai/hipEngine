---
status: current
owns: Qwen hybrid GGUF runtime module responsibilities, KV storage and scratch/capacity planning boundaries, and remaining execution/session separation.
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

The execution module retains real scratch allocation and route-specific sizing
adapters. In particular, `_gguf_prefill_scratch_nbytes` still runs the existing
prefill allocator against a fake runtime, and its result enters capacity planning
through a callback. Extracting that prefill allocator's geometry and lifetime
planning is separate work; this boundary does not claim all workspace sizing is
independent of execution. Context-dependent layer selection and speculative
workspace costs also enter as explicit caller-supplied callbacks.

Existing imports from `qwen35_gguf_runner` are aliases to the planning definitions.
Serving capacity consumers import the planning owner directly. Diagnostics that
patch the private mirror threshold must patch both execution policy and planning;
the two modules must price and execute the same diagnostic layout.

## Remaining separation

The following are candidates, not implemented modules or a mandate to move code
by line count:

1. **Prefill scratch planning:** replace the execution-backed sizing adapter with
   an explicit plan shared by the prefill allocator and estimator, after defining
   its route and lifetime inputs.
2. **Resident state and transactions:** define ownership of canonical state,
   provisional verification state, prefix snapshots and rollback journals before
   extracting the session's methods.
3. **Layer execution:** isolate attention and feed-forward execution only after
   defining their weight, scratch and selected-kernel inputs. Preserve registry
   selection rather than introducing backend-specific dispatch ladders.
4. **Diagnostics:** separate one-layer probes, capture formatting and profiling
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

Most of these tests use fake runtime allocations; the GPU node has an explicit
availability guard. Public real-model generation remains a separate integration
check. This refactor does not change arithmetic, kernel selection or defaults.
