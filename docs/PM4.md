# In-Tree Retained-PM4 Submission

> **Status (2026-08-07):** implementation in progress. P0 documentation, P1
> exact graph inspection, P2 direct public-HSA AQL, P3 retained gfx1100 PM4,
> P4 lifecycle-reproducer safe controls, and P5 production GGUF graph integration
> are complete. P6 measurement/optimization is in progress. Native HIP graphs
> remain the package default. Explicit PM4 selection fails closed, and reset-prone
> submit/recreate stress remains unrun pending a separate warning and approval.

This document defines hipEngine's plan for a small, torch-free, in-tree
retained-PM4 transport. The transport is intended to preserve the useful launch
amortization demonstrated by Redline while removing Redline as a runtime and
build dependency. It also provides a much smaller reproducer for the gfx1100
address-zero VM-fault lifecycle reported as
[ROCm/ROCm#6529](https://github.com/ROCm/ROCm/issues/6529).

This is not a promise that arbitrary PM4 is a stable ROCm API. AMD exposes the
public HSA/ROCr queue and executable APIs and publishes the vendor-AQL PM4-IB
packet layout in `aqlprofile`, but the compute register stream inside that IB is
architecture-specific. Admission is therefore per architecture, conservative,
and fail closed.

## Decision

Build the transport in-tree.

- Keep `hipgraph` as the default and correctness baseline.
- Add explicit `aql` and `pm4` transports behind a backend registration.
- Implement gfx1100 first; do not infer gfx1151 or gfx12 safety from gfx1100.
- Inspect already captured native HIP graphs rather than interposing HIP module
  registration or rewriting kernel launch wrappers.
- Load the exact embedded HSACO through public HSA executable APIs.
- Own one persistent ROCr queue and retained resources per process/GPU transport
  context rather than recreating them per replay.
- Reject any graph, kernel ABI, architecture, or lifecycle state that the first
  implementation cannot prove.
- Never fall back to HIP after an explicit PM4 submission has begun.

The intended production surface is about 2–3k lines of native/Python runtime
code plus tests and the lifecycle reproducer. It is deliberately not a generic
graph framework, compiler, profiler, or HIP interposer.

## Implementation status

| Phase | Status | Current evidence |
| --- | --- | --- |
| P0 documentation/contract | Complete | This document, PLAN cross-link, and WORKLOG decision |
| P1 exact graph inspection | Complete | Bounded ELF64/bundle/MessagePack parsing, exact kernarg packing, deterministic DAG validation, and live gfx1100 `smoke_add` reconciliation |
| P2 direct public-HSA AQL | Complete | Exact PCI-BDF agent match, public executable load, persistent queue, checked packet publication/wait/teardown, and bit-exact smoke |
| P3 retained gfx1100 PM4 | Complete | Strict descriptor admission, conservative PM4 tape, vendor-AQL IB, two bit-exact safe replays, and no fallback |
| P4 lifecycle reproducer | Complete (safe controls) | Reuse, recreate/no-submit, HSA/HIP allocation, timestamps, queue-first quarantine, and complete per-cycle JSON; reset-prone submit/recreate stress implemented but intentionally unrun |
| P5 production graph integration | Complete | Registry-selected session-owned transport, persistent context across graph generations, p512/d3 exact eager/HIP/AQL/PM4 token-state-KV-logit gate, cancellation/close, zero fallback, and exact memory recovery |
| P6 performance/promotion | In progress | Clean p512/d128 baseline: PM4 synchronized replay is 6.699% lower wall than HIP graph, but first capture makes the complete request 3.310% slower; optimize setup before any promotion |

## Goals and non-goals

### Goals

1. Replace hundreds of AQL kernel packets in a captured decode graph with one
   vendor-AQL packet pointing to one retained PM4 indirect buffer (IB).
2. Reuse hipEngine's existing native HIP graph capture as the graph-description
   frontend.
3. Remove runtime dependencies on Redline, Rust/Cargo, PyO3, mold, Radiowave,
   and `LD_PRELOAD` interposition.
4. Preserve the exact JIT-built device code, launch geometry, kernargs, and DAG
   ordering used by the native graph.
5. Make transport choice explicit, inspectable, architecture-keyed, and easy to
   switch for A/B and issue isolation.
6. Produce a compact same-HSACO HIP/direct-AQL/PM4 lifecycle reproducer with a
   complete resource-generation ledger.
7. Retain the existing torch-free runtime invariant.

### Non-goals for the first implementation

- Replacing HIP allocation, streams, events, BLAS, or kernel compilation.
- Supporting arbitrary HIP graph node types.
- Supporting scratch/private segments, dynamic call stacks, every implicit
  user-SGPR layout, partial workgroups, cooperative launch, or device enqueue.
- Providing multi-queue overlap, CU partitioning, GPU timestamps, or a generic
  PM4 optimizer in the first correctness milestone.
- Claiming gfx1151, gfx12, APU, or non-AMD portability.
- Treating process isolation, retry, or native shadow execution as a production
  recovery mechanism for a GPU VM fault.
- Copying Redline wholesale into this repository.

## Why retained PM4 is worth owning

hipEngine's decode paths are graph-heavy. A current production-sized Qwen3.6
GGUF graph contains hundreds of kernel nodes. Native HIP graph replay removes
Python launch overhead but still leaves runtime/driver graph traversal and
packet construction. A retained PM4 IB materializes the register and dispatch
stream once, then each replay publishes one small vendor packet and rings one
HSA doorbell.

The prior narrow W7900 Redline experiment established feasibility, not a
package-default claim:

| Property | Observed result |
| --- | --- |
| Workload | Qwen3.6-35B-A3B UD-Q4_K_M, p512/d128, c=1, one repeated-token graph workload |
| Captured topology | 627 kernel nodes |
| Correctness | Bit-identical final logits and token IDs; KL 0.0; top-1 1.0 |
| Native HIP graph | 92.812 tok/s median decode |
| Retained PM4 | 100.357 tok/s median decode |
| Narrow delta | +8.129% decode; +6.747% steady post-load wall |
| Persistent control | 512 PM4 launches, exact output, 0.095% range/median |
| Promotion status | Blocked by a separate recreate-heavy address-zero GPU fault and missing broad gates |

The evidence says that submission work is material enough to pursue and that a
large real graph can be lowered exactly. It does **not** prove natural-prompt,
long-context, concurrent, cancellation, shutdown, cross-architecture, or broad
model performance.

## What the in-tree path removes

The Redline integration proved useful ideas but carried a much larger product
surface than hipEngine needs:

| Redline surface | Needed in hipEngine? | In-tree replacement |
| --- | --- | --- |
| Rust workspace and Cargo build | No | Existing Python build cache compiles one small C++ C-ABI DSO |
| PyO3 control module | No | `ctypes` over the native C ABI |
| `mold` build assumption/workaround | No | Existing compiler/build planning |
| `LD_PRELOAD` HIP interposer | No | Public HIP graph inspection after capture |
| `__hipRegisterFunction` interception | No | `hipKernelNameRefByPtr` plus `dladdr` |
| Generic graph IR/planner/partitioner | No | A strict kernel-only DAG compiler |
| Radiowave recipe/tuning framework | No | Conservative barriers first; measured in-tree tuning later |
| Multi-backend benchmark framework | No | hipEngine tests and benchmark protocols |
| Multi-queue/CU-mask machinery | Not initially | One persistent queue per selected GPU |
| Generic ROCr wrapper | Partly | Only the public HSA calls and ownership types required here |
| PM4/AQL packet knowledge | Yes | Minimal architecture-keyed encoder with provenance and goldens |

Redline remains a read-only behavioral and provenance reference at commit
`33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e`. Its relevant PM4/ROCr sources are
Apache-2.0. AMD's vendor packet reference is the `aqlprofile` source named below.
No external checkout is modified by this work.

## Feasibility result: no interposer is required

A live W7900 spike inspected hipEngine's existing `smoke_add` native HIP graph
using public APIs:

- `hipGraphGetNodes` enumerated the graph.
- `hipGraphNodeGetType` identified a kernel node.
- `hipGraphKernelNodeGetParams` returned grid, block, function, dynamic shared
  memory, and stable argument-value pointers.
- `hipKernelNameRefByPtr` returned
  `hipengine_smoke_add_f32_kernel`.
- `dladdr` mapped the host function pointer to the exact JIT DSO:
  `/home/lhl/.cache/hipengine/build/.../smoke_add.so`.
- The DSO's `.hip_fatbin` section began with
  `__CLANG_OFFLOAD_BUNDLE__` and contained the exact AMDGPU code object and
  complete `amdhsa.kernels` metadata.

The observed graph contained one node with grid `(1,1,1)`, block `(256,1,1)`,
and four argument values. The three device pointers and scalar token count
matched the original launch exactly. The DSO symbol and embedded HSACO also
matched the captured function.

That gives a simpler frontend than Redline's interposer:

```text
existing hipEngine eager launches
        |
        v
hipStreamBeginCapture ... hipStreamEndCapture
        |
        +--> native hipGraph (still available as baseline)
        |
        v
public HIP graph inspector
        |
        +--> node DAG, host function, geometry, argument values
        +--> dladdr -> exact JIT DSO -> .hip_fatbin -> exact gfx1100 HSACO
        v
strict PM4 graph compiler
```

No existing `hipLaunchKernelGGL` wrapper needs to change merely to discover a
captured graph.

## Architectural invariants

The PM4 work must preserve all repository-wide invariants and adds these:

1. **Native HIP remains the oracle.** PM4 output is compared with the same
   graph's native HIP execution before any performance result is retained.
2. **Exact code object.** HIP and HSA consume bytes extracted from the same JIT
   DSO. Recompiling a nominally equivalent kernel is not sufficient.
3. **Exact graph generation.** A PM4 executable is bound to one graph topology,
   DSO fingerprint set, kernel metadata set, launch geometry set, kernarg bytes,
   and device identity. Any change rebuilds it.
4. **Persistent ownership.** Queue, completion signal, executable, reader,
   HSACO storage, kernargs, IB, and every encoded pointee outlive all submissions
   that can reference them.
5. **One architecture, one registration.** gfx1100 admission does not imply
   gfx1151 or gfx12 admission.
6. **Fail closed before submit.** Unsupported nodes, metadata, hidden args,
   scratch, symbols, register state, target IDs, or pointers reject PM4
   instantiation.
7. **Fail stop after submit.** Timeout, queue error, or transport-state
   corruption marks the transport unusable. It does not execute a native shadow
   graph and does not free packet pointees without proven retirement.
8. **No backend branch in model code.** Submission selection occurs through a
   transport registry/capability boundary.
9. **No performance shortcut around synchronization.** The first path uses
   conservative ordering. Barriers can only be relaxed after exact dependency
   and profiler evidence.
10. **No benchmark gaming.** Promotion uses the full relevant prompt/category
    suite and heldouts, not the repeated-token feasibility row.

## High-level design

```text
Python, torch-free

  HipRuntime capture
       |
       v
  GraphInspector -------------------> HipGraphManifest
       |                                - topological kernel nodes
       |                                - DSO + HSACO hashes
       |                                - symbols and kernarg layouts
       |                                - launch geometry/dependencies
       v
  KernargPacker --------------------> exact per-node bytes
       |
       v
  SubmissionRegistry
       |             |              |
       | hipgraph    | aql          | pm4
       v             v              v
  hipGraphExec   architected    native C ABI
                 AQL packets         |
                                     v
                              ROCrContext (one GPU)
                                - exact HSA agent
                                - persistent queue
                                - executables/readers
                                - kernarg allocations
                                - completion signal
                                - retained gfx1100 IB
```

The frontend stays in Python because graph and JIT metadata are already exposed
there and it keeps the native ABI small. Queue ownership, atomics, HSA callbacks,
packet publication, waits, and PM4 encoding live in C++.

## Planned source layout

```text
hipengine/core/pm4/
  __init__.py          public errors, manifests, and factory
  graph.py             public HIP graph inspection/topological validation
  elf.py               bounded ELF64, clang bundle, note, and symbol parsing
  msgpack.py           bounded subset needed by AMDGPU metadata
  kernarg.py           exact explicit/hidden kernarg packing
  transport.py         registry adapter and Python/native ownership
  native.cpp           public HSA/ROCr ownership and C ABI
  pm4_gfx1100.cpp      gfx1100 packet/register lowering
  build.py             deterministic native DSO build/cache plan

scripts/pm4_lifecycle_repro.py

tests/
  test_pm4_elf.py
  test_pm4_kernarg.py
  test_pm4_graph.py
  test_pm4_packets.py
  test_pm4_transport.py
  test_pm4_gpu.py       explicit HIP-availability skip
```

Names may be collapsed if a smaller implementation is clearer, but boundaries
between parsing, graph compilation, native ownership, and architecture encoding
must remain visible.

## Graph extraction and validation

### HIP APIs

`HipRuntime` gains lazy typed bindings for:

- `hipGraphGetNodes`
- `hipGraphGetEdges`
- `hipGraphNodeGetType`
- `hipGraphKernelNodeGetParams`
- `hipKernelNameRefByPtr`
- `hipGetDevice`
- `hipDeviceGetPCIBusId` (or an equivalent exact PCI identity API)

The inspector performs two-call count/fill enumeration where required, verifies
that counts remain stable, owns copies of all returned values, and includes the
node handle only for diagnostics. It never treats node enumeration order as
execution order.

### Supported graph shape

The first compiler accepts only a non-empty DAG of kernel nodes. It constructs a
deterministic topological order from `hipGraphGetEdges`, rejecting:

- cycles;
- edges containing unknown handles;
- duplicate/ambiguous nodes;
- memcpy, memset, host, event, child-graph, empty, external semaphore, or other
  node types;
- graph mutation during inspection;
- disconnected nodes unless a deterministic topological order remains valid
  and conservative serialization is explicitly recorded.

Independent native graph nodes are serialized in stable topological order in
the first one-queue PM4 tape. This may give up overlap but cannot invent a data
race. Multi-queue lowering is a later, separate gate.

### DSO and code-object identity

For every kernel node:

1. Resolve the kernel name with `hipKernelNameRefByPtr`.
2. Resolve the containing JIT DSO with `dladdr`.
3. Parse the little-endian ELF64 section table with strict bounds.
4. Extract exactly one applicable `.hip_fatbin` image.
5. Parse the classic clang offload bundle and select exactly one AMDGPU target
   whose `gfx...` architecture matches the selected device.
6. Require an ELF64 AMDGPU code object and hash both the DSO and selected HSACO.
7. Deduplicate executable loads by selected-HSACO SHA-256.

Host entries and nonmatching device entries are ignored. Zero matches, multiple
matching device images, CCOB/compressed formats not yet implemented, malformed
ranges, files changed during read, or a DSO outside the known JIT artifact set
all fail closed.

### AMDGPU metadata

The selected HSACO's `NT_AMDGPU_METADATA` ELF note (type 32, name `AMDGPU`) is a
MessagePack map. The parser needs only a bounded, defensive subset sufficient
to read:

- `amdhsa.kernels`
- `.name` and `.symbol`
- `.kernarg_segment_size` and `.kernarg_segment_align`
- `.group_segment_fixed_size`
- `.private_segment_fixed_size`
- `.args[].offset`
- `.args[].size`
- `.args[].value_kind`

The parser enforces byte, nesting, entry-count, integer, range, overlap, and
segment-size limits. It requires one exact kernel match, normally the `.kd`
loader symbol. There is no guessed sequential-pointer fallback in the
production PM4 path.

### Kernarg packing

`hipGraphKernelNodeGetParams` supplies either:

- `kernelParams`: pointers to explicit argument values; or
- HIP's bounded `extra` key/value protocol containing a prepacked launch buffer
  and size.

For `kernelParams`, metadata offsets and sizes determine each copy. Hidden
fields are synthesized only from a strict allowlist:

- `hidden_block_count_{x,y,z}`
- `hidden_group_size_{x,y,z}`
- `hidden_dynamic_lds_size`
- `hidden_grid_dims`
- zero-valued global offsets/service pointers whose zero contract is explicitly
  admitted for a dedicated queue

Unknown hidden fields reject the node rather than defaulting silently to zero.
The packed segment size must agree with both metadata and the public HSA loader
query. Explicit argument count, non-null source pointers, field bounds,
non-overlap, alignment, and maximum segment size are validated.

Kernargs are copied once at PM4 instantiation. Graphs whose pointer/scalar values
are intentionally updated after instantiation need an explicit update/rebuild
API and are not silently treated as static.

## Native ROCr core

### Why C++ with a C ABI

Queue publication requires exact C layouts and acquire/release atomics. A small
native DSO gives those semantics without introducing a package dependency or
placing low-level pointer lifetime in Python. The exported API uses only fixed
width integers, byte spans, opaque handles, status codes, and caller-provided
error buffers; Python uses `ctypes`.

The DSO is built lazily through hipEngine's deterministic build cache, with the
active ROCm headers/compiler and `libhsa-runtime64`. Importing hipEngine or the
PM4 module does not initialize HSA or touch a GPU.

### Process/GPU ownership

One `RocrContext` owns one selected GPU:

```text
RocrContext
  RuntimeLease
  exact HSA agent (matched to HIP PCI domain:bus:device.function)
  queue + error callback state
  completion signal
  kernarg/executable-memory pool selection
  Executable[]
    owned HSACO bytes
    hsa_code_object_reader
    hsa_executable
    resolved Kernel[]
  GraphExec[]
    kernarg allocations
    indirect-buffer allocation
    encoded PM4 words
    generation/telemetry state
```

`hsa_init`/`hsa_shut_down` are process-refcounted inside the native DSO. Agent
selection rejects ordinal-only ambiguity and verifies profile, queue type/range,
wavefront, name, and PCI identity. The HIP and HSA visibility maps are recorded
because `ROCR_VISIBLE_DEVICES` may remap physical devices.

### Exact executable loading

For each unique HSACO:

1. `hsa_code_object_reader_create_from_memory`
2. `hsa_executable_create_alt`
3. `hsa_executable_load_agent_code_object`
4. `hsa_executable_freeze`
5. `hsa_executable_get_symbol_by_name`
6. `hsa_executable_symbol_get_info` for kernel object, kernarg size/alignment,
   group segment, private segment, and dynamic call stack

The owned HSACO bytes and reader outlive the executable. The executable and
resolved kernel metadata outlive every graph using them.

The HSA kernel object is a loaded descriptor address, not directly the code
entry used by `COMPUTE_PGM_LO`. The encoder locates the `.kd` descriptor in the
HSACO ELF symbol table, reads its 64-byte descriptor, and computes the loaded
code entry from the descriptor's signed entry offset. It preserves
`compute_pgm_rsrc1`, `compute_pgm_rsrc2`, `compute_pgm_rsrc3`, and kernel code
properties from that descriptor, cross-checking public loader metadata.

### Allocations

Kernarg and IB memory come from a fine-grained/global HSA pool admitted for the
selected agent, with explicit alignment and access checks. The IB allocation is
GPU executable-command-visible and 4-byte aligned. Every allocation has a type,
address, size, generation, owner, creation status, last-submit generation, and
retirement status in the diagnostic ledger.

The first production context retains its queue, completion signal, executable
set, kernargs, and IB for its complete lifetime. A failed retirement does not
free pointees that hardware may still reference; it reports and quarantines
(or intentionally leaks on process exit) rather than creating a use-after-free.

## gfx1100 PM4 lowering

The initial encoder is registered only for exact `gfx1100`. gfx1100 uses the
legacy gfx10/gfx11 compute register map, but this fact does not authorize a
broad `gfx11*` registration.

The minimum command vocabulary is:

- `PACKET3_ACQUIRE_MEM`
- `PACKET3_SET_SH_REG`
- `PACKET3_DISPATCH_DIRECT`
- `PACKET3_EVENT_WRITE` with `CS_PARTIAL_FLUSH`

Timestamp commands (`COPY_DATA`/`RELEASE_MEM`) remain disabled by default.
The lifecycle reproducer can explicitly wrap a retained tape with a dedicated
16-byte timestamp allocation so timestamp-resource creation/retirement can be
compared against the unprofiled arm; timestamp lifecycle is one suspect in
#6529.

### Dispatch state

For each kernel the tape programs, at minimum:

- `COMPUTE_PGM_LO/HI`
- `COMPUTE_PGM_RSRC1/2`
- `COMPUTE_PGM_RSRC3`
- `COMPUTE_TMPRING_SIZE`
- `COMPUTE_NUM_THREAD_X/Y/Z`
- `COMPUTE_RESOURCE_LIMITS`
- `COMPUTE_USER_DATA_0...` for the admitted HSA user-SGPR layout
- `DISPATCH_DIRECT` workgroup counts and initiator

Static plus dynamic LDS is rounded to gfx1100's 512-byte allocation granule and
inserted into `PGM_RSRC2`. Grid work-items must be exactly divisible by each
workgroup dimension. Code entry must be nonzero and 256-byte aligned.

The initial supported kernel descriptor has:

- zero private-segment bytes;
- no dynamic call stack;
- wave32;
- only optional private-segment-buffer and required kernarg-segment-pointer
  implicit user SGPRs;
- at most 16 user-SGPR dwords.

A requested private-segment buffer is initialized conservatively to zero only
when the descriptor contract and zero-scratch restriction permit it. Every
other implicit SGPR property rejects instantiation.

### Register-state elision

A correctness-first encoder emits all required state. A second pure encoder
tracks register values inside one tape and elides only byte-identical writes to
the same register. The packet golden tests cover both modes. Stateful elision is
not enabled in production until it is bit-exact on the GPU smoke and real graph.

### Memory ordering

The first tape uses conservative boundaries:

1. System acquire at the HIP-to-HSA ownership boundary.
2. For every producer/consumer edge, wait for compute idle and perform the
   conservative same-agent global acquire/writeback/invalidate sequence.
3. End the entire IB with `CS_PARTIAL_FLUSH` before the vendor packet may signal
   completion.

This intentionally leaves performance available. Later tuning may classify an
edge as VMEM-only and use a narrower acquire, or prove that an adjacent dispatch
needs only the compute-idle boundary. Such changes are math/runtime changes and
require packet goldens, native parity, profiler proof, and full graph gates.

### Vendor AQL PM4-IB packet

One 64-byte vendor-specific AQL packet contains one `INDIRECT_BUFFER` packet
pointing at the retained PM4 words and one HSA completion signal. The layout and
constants must match AMD's
`aqlprofile/src/core/amd_aql_pm4_ib_packet.h`:

- type-zero vendor packet with barrier publication semantics;
- PM4 `PACKET3_INDIRECT_BUFFER` header;
- low/high IB address;
- bounded dword count plus valid/temporal bits;
- completion signal in bytes 56–63.

Queue code reserves an absolute packet ID, writes bytes 4–63 first, then
release-publishes the 32-bit header word, and finally writes the absolute final
packet ID to the doorbell. The queue read index is loaded with acquire semantics
when proving capacity. These are native atomic operations, not Python memory
writes.

## Submission transports

A backend-keyed transport registry exposes one common lifecycle:

```python
capture(stream) -> graph
instantiate(graph) -> executable
launch(executable, stream) -> None
destroy(executable, graph) -> None
provenance(executable) -> dict
```

Initial registrations:

| Key | Meaning |
| --- | --- |
| `hipgraph` | Existing native HIP graph instantiate/launch; default everywhere |
| `aql` | One architected AQL kernel-dispatch packet per node; diagnostic oracle |
| `pm4` | One retained architecture-specific PM4 IB; explicit only |

Selection is exposed consistently through a CLI option and environment value:

```text
--submission-transport hipgraph|aql|pm4
HIPENGINE_SUBMISSION_TRANSPORT=hipgraph|aql|pm4
```

The environment value is parsed once at owner construction, not read inside a
per-token hot loop. Production/model code asks the registry for the configured
backend and transport capability; it does not branch on `backend == ...` or
`quant == ...`. The GGUF resident session retains one native submission context
per explicit transport, so graph pointer/topology generations replace only the
executable while the matched ROCr queue persists until session/context close.

### Interoperation with HIP streams

The public HSA queue is not the caller's HIP stream. The first safe integration
therefore uses an explicit boundary:

1. synchronize the capture/replay HIP stream before direct-AQL/PM4 submission;
2. publish the vendor packet;
3. wait with a finite timeout for completion;
4. only then return to work that may run on HIP.

This makes initial `aql`/`pm4` launch synchronous and preserves correctness at
the cost of overlap. It is admitted only in graph paths that already synchronize
at the replay boundary. External semaphores or asynchronous cross-queue
integration are a later project, not assumed.

### Fallback policy

- Default/automatic selection may stay on `hipgraph` when PM4 capability checks
  fail **before** any PM4 packet is submitted.
- Explicit `pm4` selection reports the rejection and does not silently choose
  HIP.
- Once a PM4 executable has submitted, any timeout, queue error, generation
  mismatch, or unusable state is fail stop for that executable/context.
- Native shadow launch, retry, subprocess restart, and graph-family process
  splitting are diagnostics only and can never be called recovery.

## Lifecycle and error handling

Each replay follows:

1. Verify context, queue, graph, executable, pointer, and generation state.
2. Reset the completion signal only after the preceding submission completed.
3. Wait for ring capacity with a finite host deadline.
4. Copy/publish the vendor packet and ring the doorbell.
5. Wait for completion with a finite deadline while polling callback fault
   state.
6. Record queue read/write indices and completion status.
7. Mark the generation retired only after completion is proven.

On timeout or HSA callback error:

- mark the graph and context unusable;
- record operation and teardown errors independently;
- attempt queue inactivation;
- do not free signal/kernarg/IB/executable storage without proof that pending
  access stopped;
- expose a machine-readable ledger suitable for issue reports;
- never perform an unbounded wait in a destructor.

Normal destruction order is:

1. prove no graph is in flight;
2. inactivate and destroy the queue;
3. destroy completion signals;
4. free IB and kernarg allocations;
5. destroy executables;
6. destroy readers;
7. release owned HSACO bytes;
8. release the HSA runtime lease.

Every non-success status is retained. Destructors do not erase the first error.

## Standalone lifecycle reproducer

`scripts/pm4_lifecycle_repro.py` uses one tiny HSACO and the same input/output
buffers across three transports:

- HIP launch/native graph;
- direct architected AQL kernel dispatch;
- vendor-AQL retained PM4 IB.

The default command runs four safe correctness/reuse cycles. The implementation
also supports queue/resource/buffer reuse or recreation, submit versus
create/drop-only, HIP versus fine-grained HSA allocations, optional PM4 GPU
timestamps, and queue-first bounded-generation quarantine. Every direct-AQL or
PM4 submit arm that recreates executable packet resources is rejected unless
`--ack-reset-risk` is present—even one cycle exercises the suspected retirement
boundary. `--stress` selects 128 submit/recreate cycles and therefore also
requires that acknowledgement. The destructive arms are implemented but have
not been run.

### Controls

| Axis | Arms |
| --- | --- |
| Transport | HIP, direct AQL, retained PM4 |
| Queue | persistent reuse, recreate per cycle, create/drop without submit |
| Packet resources | reuse or recreate signal/kernarg/IB |
| Profiling | no timestamps first; optional timestamp resource arm later |
| Allocation | HSA-only; mixed HIP buffers with HSA dispatch |
| Ownership | one dispatch queue; separate ownership/dispatch queue diagnostic later |
| Retirement | immediate release; bounded generation quarantine |
| Workload | one known-good kernel; controlled kernel-family changes only later |

### Required ledger

For every process and cycle record:

- process, physical PCI BDF, HIP ordinal, HSA agent handle/name, gfx target;
- ROCm/HIP/HSA versions and source commit;
- queue generation/id/type/size/base, doorbell handle/value, read/write indices;
- completion handle, initial/final value, wait status, timeout;
- executable, reader, symbol, kernel object, code entry, HSACO hash;
- kernarg/IB/buffer/timestamp addresses, sizes, and generation numbers;
- packet ID, header, IB address/dword count, submit and retirement status;
- create, inactivate, destroy, signal-destroy, and free statuses;
- exact first failing cycle and last known-good cycle;
- output hash/value versus CPU and HIP oracles.

Sensitive model/process memory and raw coredumps are not committed or published.
The native provenance records both the raw observable doorbell signal value and
the last host-published absolute doorbell value; the latter is the authoritative
publication ledger because a consumed doorbell may read back as zero. It also
records HSA ABI version, completion value, packet ID/count/header/timeout,
reader/executable handles, and per-dispatch symbol, kernel object, relocated code
entry, loader alignment, geometry, and kernarg address/size. Full-address ledgers
remain local issue evidence and are compacted before publication.

### Interpretation matrix

| Result | Strongest supported inference |
| --- | --- |
| Direct AQL passes, PM4 fails | PM4 encoding, IB visibility, fence, or vendor packet path |
| Both AQL and PM4 fail only on recreation | Queue/signal/allocation/executable generation lifecycle |
| Persistent reuse passes; recreate fails | Retirement/reuse mechanism, not dispatch count alone |
| Create/drop without submit fails | Queue lifecycle is sufficient; kernel execution is not required |
| Unprofiled passes; timestamp arm fails | Timestamp commands/buffer lifetime becomes the leading differentiator |
| HSA-only passes; mixed HIP allocation fails | HIP/ROCr interoperation is required |
| Quarantine changes failure threshold | Delayed reference/address reuse is strongly implicated |
| In-tree PM4 reproduces #6529 | Redline's generic framework/interposer is eliminated; focus moves to PM4/ROCr/amdgpu |
| In-tree PM4 does not reproduce | Implementation differences become bisection leads; it does not prove the defect fixed |

This reproducer targets #6529's gfx1100 address-zero SQC-data VM fault and MES
reset chain. It must not be presented as a reproducer for #6437's distinct
repeated-128K gfx1151 queue no-progress stall.

## C ABI sketch

The exact names may change, but the ownership contract should remain this
small:

```c
typedef struct he_pm4_context he_pm4_context;
typedef struct he_pm4_executable he_pm4_executable;
typedef struct he_pm4_buffer he_pm4_buffer;
typedef struct he_pm4_node he_pm4_node; /* fixed-layout, ABI-size checked */

int he_pm4_context_create(const char *pci_bdf, const char *gfx_target,
                          he_pm4_context **out, char *error, size_t error_size);
int he_pm4_executable_create_ex(he_pm4_context *context,
                                const he_pm4_node *nodes, size_t node_count,
                                uint32_t flags, he_pm4_executable **out,
                                char *error, size_t error_size);
int he_pm4_launch_aql(he_pm4_executable *executable, uint64_t timeout_ns,
                      char *error, size_t error_size);
int he_pm4_launch_pm4(he_pm4_executable *executable, uint64_t timeout_ns,
                      char *error, size_t error_size);
int he_pm4_context_retire_queue(he_pm4_context *context,
                                char *error, size_t error_size);
int he_pm4_buffer_create(he_pm4_context *context, size_t bytes,
                         he_pm4_buffer **out, uint64_t *address,
                         char *error, size_t error_size);
int he_pm4_executable_destroy(he_pm4_executable *executable,
                              char *error, size_t error_size);
int he_pm4_buffer_destroy(he_pm4_buffer *buffer,
                          char *error, size_t error_size);
int he_pm4_context_destroy(he_pm4_context *context,
                           char *error, size_t error_size);
```

Diagnostic query functions expose copied JSON/records rather than native
pointers. Every destroy operation returns a status; Python context managers call
explicit close and make close idempotent.

## Implementation phases and acceptance gates

### P0 — Documentation and frozen contract

- This document and PLAN cross-link land first.
- Freeze default-off/fail-closed policy, architecture admission, and lifecycle
  safety rules.
- Record source/provenance references.

**Gate:** document reread; no GPU run.

### P1 — Exact graph/HSACO/kernarg inspection

- Add HIP graph APIs and pure bounded ELF/bundle/MessagePack parsers.
- Produce an immutable manifest from `smoke_add` and a multi-node graph.
- Add malformed-input, unsupported-node, topology, hidden-arg, and DSO identity
  tests.

**Gate:** deterministic CPU tests plus a guarded live inspection smoke. The
manifest must reconcile every node, edge, symbol, geometry, and explicit value.

### P2 — Public-HSA direct AQL smoke

- Build the native DSO and exact HSA agent matcher.
- Load selected HSACO, resolve symbols, allocate/copy kernargs, publish standard
  kernel-dispatch packets, and wait safely.
- Compare HIP and direct-AQL outputs for fresh and reused inputs.

**Gate:** CPU packet goldens, guarded W7900 smoke, exact output, finite timeout,
clean normal teardown, and kernel-trace proof of the expected kernel.

### P3 — gfx1100 retained PM4 smoke

- Add descriptor parsing, strict ABI admission, conservative PM4 lowering,
  vendor packet publication, persistent queue/IB/signal ownership, and
  diagnostics.
- Keep timestamps and stateful register elision off.

**Gate:** PM4 dword goldens, architecture-negative tests, exact HIP/AQL/PM4
output over repeated fresh inputs, expected kernel trace, no native fallback,
and clean resource recovery.

### P4 — Lifecycle reproducer

- Add safe reuse/create-drop controls and complete generation ledger.
- Implement recreate, mixed-allocation, timestamp, and quarantine arms.

**Gate:** safe controls pass. Any reset-prone recreate run requires a separate
warning/approval and kernel-journal collection plan. A VM fault is reported as a
fault, never retried into a passing aggregate.

### P5 — One production graph integration

- Integrate one kernel-only GGUF decode graph whose caller already synchronizes
  after replay.
- Keep transport explicit and default `hipgraph`.
- Rebuild on pointer/topology generation changes.

**Gate:** exact native-HIP/PM4 final logits, token IDs, state/KV checks, graph
reuse, cancellation/close, memory recovery, and transport provenance. No broad
performance claim yet.

**Result (W7900/gfx1100):** complete. `scripts/pm4_gguf_decode_gate.py` captured
one 627-node/17-HSACO p512 GGUF decode graph and replayed it three times through
native HIP graph, direct AQL, and retained PM4 against the same eager state
oracle. All token IDs, per-step FP32 hidden plus Conv/GDN state and live BF16
K/V fingerprints, and all 248,320 final FP32 logits were bit exact. Direct AQL
consumed 1,881 queue packets; retained PM4 consumed three vendor packets for the
same 25,707-dword/102,828-byte IB. Both native routes recorded zero fallback,
zero callback status, complete retirement, a no-submit cancellation generation,
checked context close, and exact recovery of the retained 2 MiB allocation.
Evidence:
`benchmarks/results/2026-08-07-gfx1100-in-tree-pm4-gguf-p5-correctness.json`.
Destructive submit/recreate stress remains intentionally unrun.

### P6 — Performance and conservative optimization

Measure native HIP graph, direct AQL, conservative PM4, and state-elided PM4 in
counterbalanced order on the same loaded model and graph. Attribute:

- host launch call wall;
- synchronized replay wall;
- GPU kernel-family total;
- PM4 dword/register-write count;
- HSA queue submissions and waits;
- end-to-end p512/d128 throughput.

Only then consider register-write elision and narrower dependency boundaries.
The canonical focused harness is `scripts/pm4_graph_bench.py`: one loaded model,
one stable-pointer graph per transport, exact reset/rearm, rotating order, and
separate host-call/synchronized/capture-inclusive metrics. Native call wall is
blocking by contract and includes its stream drain plus finite wait; synchronized
replay is therefore the primary cross-transport comparison.

**Baseline result (W7900/gfx1100, clean `bfc658195`):** the canonical one-warmup,
five-round p512/d128 run is exact across all warmup/measured final tokens,
recurrent/KV state, and final logits. PM4 synchronized replay improves
**10.747345 -> 10.027337 ms/token (-6.699%)**, or **93.046 -> 99.727 tok/s
(1.0718x)**, while direct AQL is **11.460463 ms/token**. PM4 first capture is
**192.119 ms** versus HIP graph **46.475 ms**; after charging capture, PM4 is
**11.528268 versus 11.110428 ms/token (+3.761%)**, and complete request wall is
**3.310% slower**. The estimated capture break-even is about 202 decode tokens.
Keep PM4 explicit and optimize setup before p512/d128 promotion. `rocprofv3`
does not decode nested dispatches inside the vendor-AQL IB; it records no inner
PM4 kernel rows, so device attribution needs direct-AQL tracing or retained GPU
IB timestamps rather than a false kernel-family sum. Evidence:
`benchmarks/results/2026-08-08-gfx1100-in-tree-pm4-graph-baseline.json`.

The first exact opt-in candidate, `HIPENGINE_PM4_STATEFUL_REGISTERS=1`, ports the
already-frozen register-state oracle into the native encoder. It always emits
the first value for every SH register and only elides later identical values;
kernarg pointer dwords still change per node. The p512/d3 production gate remains
bit exact and the tape falls **25,707 -> 18,100 dwords (-29.591%)**. The corrected
same-loaded-session harness gives conservative and stateful PM4 separate queues
and counterbalances them against HIP graph. Its tracked-clean one-warmup/five-
round p512/d128 result cuts **25,666 -> 18,079 dwords (-29.560%)** and improves
**10.044991 -> 9.989421 ms/token (-0.553%, 5/5 paired wins; -0.611% paired
median)** with exact shared tokens, recurrent/KV state, and logits plus zero
fallback and clean teardown. This remains an opt-in candidate pending the
broader PM4 promotion gates, not package-default evidence. Evidence:
`benchmarks/results/2026-08-08-gfx1100-in-tree-pm4-stateful-register-elision.json`.

Setup attribution on the post-review 626-node p512/d3 stateful graph identifies
**126.023 ms** in graph inspection and **42.044 ms** in native instantiation
inside **205.819 ms** total capture; native subphases are **28.287 ms** for 626
kernarg allocations, **6.936 ms** for all 17 HSA module load/freezes, and only
**0.144 ms** for PM4 encoding. Caching immutable kernel name/DSO/metadata
resolution by repeated HIP function pointer within one inspection is exact and
cuts inspection **126.023 -> 86.590 ms (-31.290%)** and total capture
**205.819 -> 164.959 ms (-19.852%)** in the directional run; a committed-clean
repeat is exact at **85.811 ms** inspection and **166.750 ms** capture. Replacing
626 separately rounded HSA allocations (2,564,096 bytes) with one aligned
200,240-byte logical/200,704-byte allocated kernarg slab then cuts native
instantiation **44.167 -> 12.760 ms (-71.110%)**, kernarg staging plus allocation
to **0.286 ms**, and separately measured total capture **166.750 -> 142.550 ms
(-14.512%)** despite 5.520 ms adverse inspection variance. Module load/freeze is
not the high-leverage target; DSO read/extract/hash/metadata owns **51.555 ms**
of remaining clean inspection. Identity-checked immutable DSO reuse by
`(path, gfx, device, inode, size, mtime, ctime)` makes a subsequent exact graph
capture pay **0.057 ms** rather than **57.407 ms** for DSO loading; in the
same-process directional run, inspection is **91.699 -> 30.448 ms** and capture
**145.042 -> 75.914 ms**. Eight-way cold DSO loading regressed that phase
**55.061 -> 63.072 ms** and was removed. Slab/cache publication and a
same-session p512/d128 end-to-end comparison remain pending.

A wait-only dependency experiment retained each compute-idle `EVENT_WRITE` but
removed the following `ACQUIRE_MEM`. It reduced the 626-node stateful tape
**18,079 -> 13,079 dwords (-27.656%)** and diagnostic replay
**9.948 -> 9.809 ms/token (-1.392%)**, but is **rejected**: recurrent/KV state
and final logits diverged after three steps even though token IDs coincidentally
matched. Every submission retired and teardown stayed clean, so this is direct
evidence that a cache acquire/invalidation is semantically mandatory, not a
lifecycle artifact. The flag and implementation were removed. Evidence:
`benchmarks/results/2026-08-08-gfx1100-pm4-wait-only-dependency-rejected.json`.

The narrower default-off `HIPENGINE_PM4_LOCAL_CACHE_DEPENDENCIES=1` candidate
retains both the compute-idle event and `ACQUIRE_MEM`, but changes GCR control
from `GLK_INV|GLV_INV|GL1_INV|GL2_INV|GL2_WB` (`0xc380`) to the local
`GLK_INV|GLV_INV|GL1_INV` subset (`0x0380`). The prior completion barrier makes
producer writes visible in coherent GL2; consumers still invalidate scalar,
vector, and GL1 caches. It is bit-exact on the 626-node p512/d3 gate and on a
counterbalanced p512/d128 run. The latter improves stateful replay
**10.022598 -> 9.936004 ms/token (-0.864%, 5/5 paired wins)** and throughput
**99.775 -> 100.644 tok/s (1.0087x)** with identical tokens, recurrent/KV state,
and all logits. The tracked-clean three-way publication confirms global-to-local
**10.052766 -> 9.964358 ms/token (-0.879%, 5/5)** and local PM4 is **7.104%**
faster than HIP graph replay. Cold local capture is **132.858 ms**, down
**31.425%** from the prior 193.739 ms stateful artifact; cold capture-inclusive
p512/d128 is now within **0.329 ms / 0.023%** of HIP graph instead of about 3.5%
slower. Tape size remains 18,079 dwords. Evidence:
`benchmarks/results/2026-08-08-gfx1100-pm4-setup-local-cache-clean.json`. This is
retained as an explicit candidate pending the broader promotion matrix.

`scripts/pm4_promotion_gate.py` executes that matrix in one resident session. It
uses HIP graph as the exact oracle for every prompt in the complete
`mtpbench-code-general-ja` category suite and `gdn-prefill-category-heldouts`,
adds a 4K context stress generation, reuses one PM4 queue across all graph
rebuilds, performs no-submit and retired-after-submit cancellation closes, and
requires child-ledger drain, context/session shutdown, and memory recovery. The
one-prompt/64-token harness smoke passes four PM4 generations with exact
seed/final tokens, recurrent/KV state, and all logits. The tracked-clean full
run passes **19/19 exact cases** across every named category/heldout plus 4K,
with 21 graph generations, 58 retired submissions, both cancellation paths,
zero live children, and clean context shutdown. Its only aggregate false
failure was measuring memory before the first 4K lazy prefill workspace; the
harness now warms the declared maximum shape before taking the memory baseline.
A tracked-clean focused one-natural+4K rerun then passes memory recovery at the
known **4 MiB** first-use delta together with both cancellation paths, four more
graph generations, and context shutdown. Per focused-repair policy these form
one accepted promotion matrix; evidence:
`benchmarks/results/2026-08-08-gfx1100-pm4-promotion-matrix.json`. The full
command is:

```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 GPU_MAX_HW_QUEUES=1 \
HIPENGINE_HIP_ARCH=gfx1100 PYTHONPATH=. \
python3 scripts/pm4_promotion_gate.py \
  --steps 3 --context-stress-length 4096 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached --json /tmp/hipengine-pm4-promotion.json
```

**Gate:** bit-exact or repository correctness thresholds, all required prompt
categories/heldouts for a retained claim, every named lifecycle gate, exact
benchmark command/hardware/source evidence, compact artifact, rollup, and
changelog update.

### P7 — Broader graph admission

Apply separately to proposal graphs, target/verifier graphs, and eventually a
combined speculative tape only where pointer and transaction ownership are
stable. Admit gfx1151 only with a peer encoder registration, its own packet
proof, and its own lifecycle/correctness/performance gates.

## Test matrix

### CPU deterministic tests

- ELF64 section/symbol/note bounds and malformed corpus.
- Classic clang bundle target selection and ambiguity rejection.
- Bounded MessagePack decoding and AMDGPU kernel matching.
- Explicit and hidden kernarg field packing, overlap, alignment, size, null,
  unknown-kind, and `extra` protocol cases.
- DAG topological order, cycles, independent nodes, unsupported node types, and
  graph mutation.
- gfx1100 kernel descriptor parsing and code-entry relocation.
- PM4 `PACKET3` headers, register offsets, LDS rounding, workgroup counts,
  initiator, acquire/flush, and stateful elision goldens.
- Vendor AQL packet bytes and publication header.
- Architecture, scratch, dynamic-callstack, user-SGPR, wave-size, partial-grid,
  and null-address rejection.
- Ownership state machine, timeout, double-close, operation-plus-teardown error,
  quarantine, and no-fallback tests using a fake native ABI.
- Registry selection/default/explicit rejection tests.

### Guarded GPU tests

Every test that loads HIP/HSA or runs a kernel first probes
`libamdhip64.so`/`libhsa-runtime64.so` and skips cleanly without ROCm.

1. HIP graph inspection reconciliation.
2. HIP versus direct AQL smoke output.
3. HIP versus PM4 smoke output with fresh inputs each replay.
4. Same graph reuse, signal reset, and teardown.
5. Kernel trace showing the expected symbol.
6. Architecture-negative selection.
7. Targeted real decode graph native/PM4 equality.
8. Lifecycle close/cancellation/recovery.

A new/ported compute kernel correctness threshold remains KL <= 0.05 and top-1
>= 90%; transport parity should normally be bit exact because device code and
kernargs are identical.

## Promotion policy

`pm4` remains explicit and default-off until all are true:

1. The minimal safe reproducer and retained-queue controls are stable.
2. #6529's recreate/lifecycle risk has an owned root cause or the required
   recreate stress passes on both gfx1100 discrete cards under the declared
   protocol.
3. Strict proof records one PM4 submission and zero native fallback for every
   claimed launch.
4. Natural prompt/category and heldout correctness passes.
5. 512 and 4K contexts, graph rebuild/regrow, repeated reuse, cancellation,
   server shutdown, and memory recovery pass.
6. The measured end-to-end workload improves without a correctness or lifecycle
   regression.
7. The package can report architecture, transport, source, HSACO hashes, graph
   fingerprint, and lifecycle status.

Even after gfx1100 promotion, other architectures retain `hipgraph` until their
independent encoder and gates pass.

## Refactor/removal triggers

During development, `HIPENGINE_SUBMISSION_TRANSPORT`, the direct-AQL diagnostic,
conservative/stateful PM4 comparison, and lifecycle controls are intentional.
After promotion:

- remove comparison-only environment aliases once the canonical CLI/config path
  is stable;
- retain `hipgraph` as the required portable fallback and correctness oracle;
- retain direct AQL while it remains useful for PM4 versus ROCr isolation;
- remove timestamp/quarantine experiments if they neither reproduce nor
  differentiate #6529;
- collapse packet/state diagnostics only after the lifecycle issue is closed and
  an equivalent machine-readable ledger remains.

These triggers must be mirrored in `docs/REFACTOR.md` when temporary flags or
paths are actually introduced.

## Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| PM4 is architecture-specific and not a stable HSA contract | Exact gfx target registration, packet goldens, default-off, fail closed |
| Wrong graph order | Use edges/topological sort; conservative serialization; reject ambiguity |
| Wrong kernarg layout/hidden fields | Require AMDGPU metadata and loader agreement; no guessed production fallback |
| Stale graph pointers | Fingerprint values/generations; explicit rebuild/update contract |
| HIP/HSA device mismatch | Match physical PCI BDF and record visibility remapping |
| Missing cache/fence operation | Conservative per-edge idle/acquire first; relax only with proof |
| Queue timeout followed by unsafe free | Mark unusable, inactivate, quarantine/leak unretired pointees, report both errors |
| Recreating #6529 resets the GPU | Safe modes by default; separate approval and journal plan for destructive stress |
| Narrow benchmark overfit | Full categories/heldouts and exact native baseline before promotion |
| New hard dependency | Native DSO uses existing ROCm installation and stdlib `ctypes`; no new Python/Rust package |
| Redline code/license confusion | Clean small implementation with explicit provenance; preserve notices for any adapted code |

## Provenance and primary references

Implementation must cite exact source/commit in file headers or comments where
packet/register/lifecycle logic is adapted.

- Redline read-only reference:
  `warpfront/redline@33683f3d4f302a6c56bcc7a4c33ab8be3262dd2e`
  - `crates/redline-rocr/src/pm4_gfx10.rs`
  - `crates/redline-rocr/src/packet.rs`
  - `crates/redline-rocr/src/runtime.rs`
  - `crates/redline-hipgraph/src/metadata.rs`
  - `crates/redline-dispatch/src/aql/replay.rs`
- AMD vendor AQL packet and public HSA-header reference:
  `ROCm/rocm-systems@c0430a50286200ab0562f4733445cdee6e48d416`,
  especially `projects/aqlprofile/src/core/amd_aql_pm4_ib_packet.h` and the
  public ROCr HSA headers.
- Public ROCr/HSA headers from the active ROCm installation.
- AMDGPU code object ABI and kernel descriptor documentation corresponding to
  the active compiler/code-object version.
- gfx10/gfx11 packet/register definitions corresponding to the admitted target.
- Functional issue:
  [ROCm/ROCm#6529](https://github.com/ROCm/ROCm/issues/6529).
- Distinct long-context progress issue, not this reproducer:
  [ROCm/ROCm#6437](https://github.com/ROCm/ROCm/issues/6437).

## Final success criterion

The project succeeds when hipEngine can, without Redline or interposition,
inspect one of its own captured kernel-only graphs, load the exact same gfx1100
HSACO through public HSA, replay the exact graph through one retained PM4 IB,
match native HIP output, survive the declared lifecycle gates, expose complete
transport proof, and deliver a measured non-regressive end-to-end improvement.
Until then, native HIP graph replay remains the default.
