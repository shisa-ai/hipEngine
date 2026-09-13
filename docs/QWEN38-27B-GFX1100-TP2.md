# Tensor parallelism: TP=N architecture, Qwen3.8-27B TP2 bring-up

Status: implementation plan; no tensor-parallel engine exists yet and no TP
speedup is claimed. Packet 0 topology/collective screening, Packet 1 rank-bound
transport, and Packet 2 shard planning are measured or implemented; see
"Measured status" below.
Reviewed: 2026-09-14 against source `a9aa29c364601620cf86b3824479d808adae46ad`.
The filename is retained for existing links. Infrastructure targets single-host
TP=N; the first topology to qualify is W7900 + RX 7900 XTX.
N-rank design is not N-rank certification.

## Objective and decision rule

Implement torch-free tensor parallelism, first qualifying TP2 for Qwen3.8-27B GGUF
`Q4_K_M` on one host containing a W7900 and an RX 7900 XTX, both gfx1100,
connected through PCIe. Optimize **one active request's decode latency**, not
aggregate throughput from two independent requests. Include multi-token
prediction (MTP) as a separately qualified path.

Start with equal weight shards and replicated hidden activations. Measure PCIe
collective latency before implementing the full model. Partition the dense
multilayer perceptron (MLP) first, then the hybrid attention layers. Keep the
small sequential MTP draft on one GPU initially; run target verification across
both. Do not assume either TP2 or MTP will win.

Planning targets, not predictions or minimum promotion thresholds:

- TP2 autoregressive (AR, no MTP) decode: aim for at least 1.3x the faster
  same-host single-GPU AR arm; 1.5x is a stretch target.
- TP2 + MTP: beat TP2 AR **and** the best measured single-GPU configuration,
  including single-GPU MTP. Report both ratios, not their assumed product.
- Keep every correctness-qualified, non-regressive measured improvement in its
  validated scope, even below these targets. If PCIe latency prevents a win,
  preserve useful scoped improvements and document the blocker; do not enable
  a slower TP2 or MTP route by default.

Out of scope: pipeline/expert/sequence parallelism, multi-host execution,
training, MoE, quant-format changes, and a speculative algorithm replacement.
Do not divert this campaign into DFlash or independent two-request throughput.

## Implementation facts and dependencies

Notation: N is the number of TP ranks; TP1 means one GPU;
C is the number of active requests; K is draft
candidate depth, with K0 meaning no MTP. Q/K/V are attention query/key/value;
KV is the key/value cache. RCCL is AMD's collective communication library;
P2P means peer-to-peer device transfer. GEMV/GEMM are matrix-vector/matrix-matrix
multiplication; H2D/D2H mean host-to-device/device-to-host copies. EOS means
end-of-sequence; OOM means out of memory. Other numerical terms follow
`EXECUTION-PROFILES.md`.

Read these before coding:

| Source | Relevant fact or contract |
| --- | --- |
| [`PLAN.md`](PLAN.md#multi-gpu-strategy) | Host-owned sharding and communication; the former minimal TP sketch is not an implementation or a validated effort estimate. |
| `hipengine/distributed/__init__.py` | Still empty at review; no working TP engine to extend. |
| `hipengine/server/api.py` | Capability metadata advertises world size 1 and no tensor-parallel support. Do not advertise TP2 before integration gates pass. |
| `hipengine/core/{hip,device,memory,tensor}.py` | Existing HIP/device primitives to audit for explicit device ownership. |
| `hipengine/core/runtime.py`, `hipengine/core/pm4/{transport,graph}.py` | Backend-neutral runtime protocol and registered HIP/native submission transports; submission transport is not collective transport. |
| `hipengine/generation/{engine_loop,engine_service}.py`, `hipengine/dispatch/` | One model-owning loop and scheduler; TP ranks are not independent requests or replica engines. |
| `hipengine/kvcache/{backend,ledger,global_pool,graph_binding}.py` | `KVCacheBackend`, atomic resource claims/deltas, generation-checked storage and graph bindings. |
| `hipengine/loading/qwen35_gguf{,_materialize}.py` | Metadata, mixed GGUF tensor formats, full/linear attention mapping, and materialization. |
| `hipengine/runtime/qwen35_gguf_runner.py` | Resident target, recurrent state, prefill, and packed verification. |
| `hipengine/runtime/qwen35_gguf_{mtp,nextn}.py`, `hipengine/speculative/gguf_mtp.py` | Draft/target integration; also inspect `transaction.py`, `native_cycle.py`, and `verify_graph.py` in `hipengine/speculative/`. |
| `hipengine/generation/qwen35_gguf_mtp2{,_registry}.py`, `hipengine/speculative/{provider,frontier,interfaces}.py` | Current provider/frontier integration; `mtp2` names the speculative implementation, not tensor-parallel degree two. |
| [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md) | Normative arithmetic, determinism, ownership, and test gates. |
| [`BENCHMARK.md`](BENCHMARK.md), [`benchmarks/README.md`](../benchmarks/README.md) | Evidence and matched-baseline rules. Historical rates are not TP2 denominators. |

Qwen3.8 uses the Qwen3.5-family implementation here. It is not an ordinary
all-softmax transformer: linear-attention layers carry convolution and Gated
DeltaNet (GDN) recurrent state. Generate the exact layer/head/state inventory
from the supplied GGUF; do not hardcode dimensions from the model name.

### Review findings and current boundaries

- **TP2-only orchestration would become architectural debt.** Use rank vectors,
  explicit partitions, and collective schedules, not paired fields or `1 - rank`.
  The two-rank peer optimization is a transport capability, not the generic API.
- **Device labels are not device ownership.** `DeviceBuffer` stores pointer/size;
  `DeviceRuntime` has no device-selection contract, and `get_hip_runtime()` is a
  process singleton. A shared library binding can remain shared, but allocation
  and launch ownership must be verified.
- **Serving must use the current loop and KV contracts.** A separate TP scheduler
  or one complete runner per GPU duplicates request ownership and cannot insert
  the required intra-layer collectives.
- **Replay and profile policy are first-class dependencies.** Native PM4 replay
  currently drains its HIP stream and waits on its native queue. Its kernel-only
  graph inspection does not establish RCCL interoperability. Existing profile
  certificates cover TP1, not distributed sums.
- **The September 6 MTP handoff is historical, not a current capability map.**
  The gfx1100 capability table now contains physical C1 and width/depth entries,
  while model-local evidence independently controls admission. Backend capability,
  explicit legacy MTP, physical provider engagement, and automatic qualification
  are different facts. Inspect `hipengine/models/qwen35.py`,
  `hipengine/kernels/hip_gfx1100/__init__.py`, and resolved serving evidence at
  implementation time. No TP1 result qualifies TP=N.

The recommendation remains column/row sharding with replicated hidden
activations and RCCL first. Whether this is fastest on the actual PCIe host
requires measurement; the review does not establish a performance result.

## Measured status

Packets 0-2 are measured on the target host (Ryzen 9 5950X, W7900 at
`0000:0d:00.0` + RX 7900 XTX at `0000:10:00.0`, both `gfx1100`, separate CPU root
ports, PCIe 4.0 x16 confirmed under load). Artifacts live under
`benchmarks/results/tp2_*.json`; the numbers and their scope are in
`benchmarks/README.md` and the worklog entry for the unit.

- **Peer DMA is unavailable on this host, for two independent reasons.**
  `hipDeviceCanAccessPeer` is false in both directions and
  `hipDeviceEnablePeerAccess` fails with HIP error 101: both cards expose a
  256 MB BAR even though the kernel advertises a resize attribute
  (`resource0_resize`). Independently, every bridge between the two cards — the
  CPU root ports `00:03.1`/`00:03.2` and the downstream bridges `0c:00.0`/
  `0f:00.0` — has ACS redirection enabled (`ACSCtl` sets `SrcValid+`,
  `ReqRedir+`, `CmpltRedir+`, `UpstreamFwd+`), which sends peer TLPs to the root
  complex instead of forwarding them. Both cards are trained at PCIe 4.0 x16
  (`LnkSta: Speed 16GT/s, Width x16`; `pp_dpm_pcie`'s `x8` is a DPM capability
  table, not the live link), so link width is not the limiter. Collectives
  therefore host-stage at ~8 GB/s of payload.
  **Host-level action**: peer DMA needs Resizable BAR / Above 4G Decoding in
  firmware *and* ACS redirection turned off on the path between the cards; the
  second is an IOMMU-isolation tradeoff and is the human lead's call, not a
  benchmark-time change. Until both are in place the peer-copy path in "Runtime
  and communication" is not a candidate and bf16 transport is the prefill
  default (2.593 -> 1.309 ms per 1024-row all-reduce).
- **Collective latency is affordable for decode.** A 20 KB all-reduce costs
  28-32 us marginal inside one group; 36 collectives per token is 1.0-1.15
  ms/token against 33.5-35.8 ms/token of single-GPU decode. Group enqueue already
  satisfies the "first rank's collective must not block the second rank's
  enqueue" requirement, and threaded enqueue is measurably worse, so no host
  threads are needed for enqueue.
- **A TP2 rank holds half the KV pool.** At 8192 context a rank claims 258 MiB
  of KV against 514 MiB for the whole pool (16 full-attention layers, 2 of the 4
  KV heads, 32 KiB per token, plus 2 MiB of `KVLiveSpans`), and 1032 against 2056
  MiB at 32768. The per-rank head partition is the weight planner's own
  `partition_groups` result, so a rank's KV heads are exactly the heads its
  weights serve and N=3 is refused with the planner's message. Claims are
  all-or-nothing across ranks.
- **RCCL work can be captured into a HIP graph, up to a size limit.** With
  communicator creation outside capture and each rank's whole chain captured on
  its own stream, 40/40 probes across chain depths 1/4/8/16/24 replayed
  bit-identically to the graph-disabled result, and replay is 20-30% faster than
  eager enqueue (a 24-op group: 1.05-1.19 -> 0.72-0.88 ms). The limit matters:
  49 captured nodes (24 ops plus their producer/consumer memsets) works, 65
  nodes (32 ops) faults the device with a memory access error, so a TP2 decode
  step's 36 collectives must be split across at least two graphs rather than
  captured as one.
- **The GDN value-head axis is tiled.** GGUF linear-attention weights use
  llama.cpp's reordering (`k_head = v_head % ssm_group_count`), so a shard plan
  must split the key-head axis contiguously and cut each value tile the same way.
  A contiguous value-head split silently pairs value heads with the wrong key
  heads and is not a legal plan. `hipengine/loading/qwen35_gguf_shards.py` owns
  this mapping, and `gdn_head_map` exposes the rank-local head mapping a kernel
  needs.
- **Admissible degrees are model geometry, not a free choice.** Qwen3.8-27B
  `Q4_K_M` admits N=1, 2, and 4: N=3 fails on the 16-group GDN key-head axis and
  N=8 fails because 17408 MLP input columns per rank is not a 256-element quant
  block. Both are refused before allocation.
- **Byte preservation is verified end to end.** All 851 autoregressive tensors
  (15.65 GiB) round-trip bit-exactly at N=1, 2, and 4, with original quant blocks
  copied verbatim.

Still unimplemented: rank-local runner adapter, distributed KV composition,
local kernel registry resolution at halved head counts, nonblocking
communicator/abort with a supervised diagnostic fallback, MTP on one rank, and
vocabulary sharding.

## Architecture to implement

### TP=N plan and integration boundaries

Resolve one immutable distributed plan at construction, before weight allocation.
These are proposed responsibilities, not existing public APIs:

| Owner | Responsibility |
| --- | --- |
| `hipengine/distributed/` plan/context | Ordered unique devices, N, control/draft owners, rank-local contexts, communicator lifetime, collective schedule, topology fingerprint. |
| Model plugin / shard planner | Logical tensor axes, full-attention and GDN head maps, layer boundaries, aliases, admissible degrees. |
| Quant/materialization plugin | Byte-preserving extraction, block alignment, local strides, preflight/consumer checks, rank-local repacks and sidecars. |
| Distributed runner adapter | Consume one scheduler `WorkItem`; enqueue local layer segments on all ranks and collectives; expose one logical runner/result to the existing loop. |
| Distributed KV composition | Aggregate rank-local pools/leases/views behind the existing backend contract; own distributed state transactions. |
| Local four-axis kernel registry | Resolve actual local dimensions/profile scopes; no TP degree or transport as a fifth axis. |
| Collective transport | Rank-group broadcast and all-reduce-sum, completion/errors, optional correctness/sampling gather; no model or scheduler policy. |
| Submission transport registry | HIP graph or independently qualified native replay for local compute; not an RCCL replacement. |

Bind model/quant/KV identities, rank order and physical devices, global-to-local
head/channel maps, replicated tensors, local layout/variant manifests,
communication dtype/algorithm, ordered reduction boundaries, graph policy and
evidence scope. Serialize deterministically and hash the plan. Use per-layer
partitions where geometries differ; require exact coverage without gaps/overlaps
except declared replication. Reject impossible degrees before allocation.
Generic TP=N does not mean every integer degree works for every model.

Start with balanced aligned ranges for arbitrary admissible N. Test N=1,2,3,4
on synthetic CPU fixtures, including non-power-of-two and uneven legal
partitions. Do not generalize `hidden/2` or a pair-specific reduce tree.
Initially require compatible homogeneous backend/architecture families;
different gfx1100 card capacities are allowed. Mixed HIP/CUDA or architecture
groups need separate qualification. N=1 bypasses communicator construction and
preserves the existing TP1 runner.

One existing model-owning engine loop decides admission, row maps, sampling,
finish and cancellation. It sends the same logical work/positions/active-mask
schedule to every rank. Physical pointers and head-local KV dimensions differ;
request identity and causal visibility do not. TP degree N, physical request
width C, and verifier rows R are independent axes.

### KV pools, transactions, and admission

Expose one composite `KVCacheBackend` with rank-qualified pool IDs in its
`KVPoolPlan`. The existing ledger reserves one atomic `ResourceClaimSet`
spanning every rank, including communication workspace and transaction snapshots.
Never admit from aggregate free VRAM: each rank's claims must fit. Failed
reservation/preparation releases all acquired resources before another admission;
inject failures after each rank's acquisition in tests.

Each local attention call consumes local `KVLiveSpans` and `KVStorageView`
through a rank-local prepared view. Preserve the common backend interface with
a distributed adapter/internal view bundle; do not concatenate foreign-device
pointers into one kernel view. Head-local offsets can differ, but logical
positions, visibility and prefix ownership must agree. Prefix keys include the
distributed layout identity. Disable prefix/tiering/eviction compositions until
qualified rather than inheriting TP1 capability claims.

Use a group transaction ID and epoch for AR mutation as well as MTP. Prepare
all ranks, enqueue agreed work, check completion, then publish one committed
prefix/output. Do not expose a sampled token before its state commits everywhere.
This is a logical commit protocol, not a requirement to copy all KV/recurrent
state on every AR token. AR may mutate in place and invalidate the group on
failure; speculative rollback requires its declared snapshots/journal.
Recoverable pre-launch errors unwind all ranks; asynchronous device/collective
failure poisons the group and fails its requests. Partially advanced groups are
not reusable. Cancellation takes effect at an agreed boundary, never by skipping
one rank's next collective. Reclaim only after all consumers/graphs complete.

### Runtime and communication

Use one process, one explicit device context and nonblocking compute stream per
rank, and persistent per-device workspaces. A rank is one participating GPU.
Enqueue all ranks' work before waiting. Establish reductions **inside each layer**, before
the next consumer; running two complete forwards and reducing afterward is
incorrect. Prototype with grouped RCCL calls through `ctypes` (`librccl.so`),
without importing torch. Validate group launch semantics so the first rank's
collective cannot block the second rank from being enqueued.

For each segment: enqueue local producers for every rank, issue matching
collectives for all communicators inside one group, close the group, then enqueue
dependent consumers. Start with collectives on the compute stream for ordering.
A separate communication stream is an optimization only when independent work
can overlap; require producer/completion events and stable buffer lifetimes.

RCCL exports NCCL-compatible `nccl*` names. Bind against installed headers and
version, not guessed enums or opaque-struct sizes. Group initialization separately
from collectives. Successful blocking-mode `ncclGroupEnd` establishes enqueue,
not device completion; nonblocking communicator/group progress needs explicit
status polling. Do not enqueue dependent kernels until nonblocking group enqueue
has completed on every communicator. Validate kind, sequence, count, dtype, root
and communicator epoch before issuing a group. Uneven weight shards still produce equal
`rows * hidden_size` all-reduce counts. Provide asynchronous-error/timeout handling
and communicator abort. Do not promise Python recovery from a hung driver:
a supervisor must be able to terminate a poisoned worker.

The serving transport must use a version-qualified nonblocking communicator/
abort path and poll stream progress plus asynchronous errors under a deadline,
not wait indefinitely in `hipStreamSynchronize`. Never abort while another
thread is inside a communicator call. Older blocking-only bring-up belongs in
a supervised diagnostic process, not the public serving path.

Prevalidate static collective schedules at plan/capture construction and validate
dynamic row/epoch metadata once per work item. Do not add Python tensor inspection,
D2H probes, or host barriers inside every production layer. GPU stream ordering
carries layer dependencies; only the request/transaction boundary needs host
completion before externally publishing results.

Use RCCL as the reference communication implementation. Screen a specialized
two-rank peer-copy plus local-sum path only if measured small-message latency
justifies it. It must have explicit producer/consumer events, reusable-buffer
lifetime rules, and stress tests; peer accessibility alone does not establish
ordering or coherent polling semantics. Do not begin with persistent GPU
spin-wait barriers. Keep a registered strict fallback for any fused kernel.
A host-staged transport is a diagnostic fallback, not an assumed fast path.

Each allocation, stream, event, library handle, graph executable, kernel-module
handle, and workspace must belong to a device. Audit cache keys and teardown,
including thread-local current-device state. Compile artifacts may be shared
by architecture where safe; loaded handles and graph/device pointers must not
be reused across devices accidentally. TP configuration chooses a distributed
plan; model and dispatch code must not grow backend/quant string branches.

Use rank-bound runtime owners over existing library bindings, with scoped
current-device selection/restoration for allocation, launch, capture and teardown.
Carry device identity through buffers, arenas, tensors and allocation accounting;
validate it at wrapper boundaries. Audit BLAS handles, ctypes caches, native-cycle
descriptors, sampler buffers, PM4 queues and environment-derived settings.
Existing profile binders use process-wide settings: resolve one consistent group
profile before rank construction, never toggle environment variables per rank.
Add host threads or a native enqueue loop only after measuring launch skew;
multiprocessing is a separate decision if single-process constraints cannot be
resolved.

### Weight and state ownership

In this table a matrix is written mathematically as `W[out, in]`, independent
of GGUF storage order. Column-parallel means split output features; row-parallel
means split input features and sum partial outputs.

| Component | Initial TP2 ownership | Communication / correctness requirement |
| --- | --- | --- |
| Hidden activations, residual, RMSNorm | Replicated | Norm over the complete hidden vector locally. Add residual and any bias once, after the sum. |
| MLP gate/up | Shard matching intermediate channels | Keep paired gate/up slices aligned; nonlinear product is local. |
| MLP down | Shard matching input channels | Sum full-hidden partial outputs once per MLP. |
| Full-attention Q/K/V, query gate and norms | Shard complete query-head groups with their KV heads; duplicate required KV heads only when grouping requires it | Preserve grouped-query head mapping, rotary dimensions, gate layout, and per-head norms. No hidden all-gather inside attention. |
| Full-attention output | Shard input head channels | Sum full-hidden partial outputs before residual. |
| KV cache | Local owned heads, with only necessary KV-head replication | Keep `KVLiveSpans` ABI, positions, masks, and logical page ownership consistent on both ranks. Do not shard sequence positions. |
| Linear-attention QKV/gate, alpha/beta, convolution, GDN state | Shard complete independent state/head groups | Map repeated Q/K groups to their value heads explicitly. Slice all associated parameters and state together. Prove norm axes are local; otherwise replicate the dependent group or add an explicit reduction. |
| Linear-attention `ssm_out` | Shard input channels matching GDN output | Sum full-hidden partial outputs before residual. |
| Embedding and final output norm | Rank-owned embedding, replicated final norm initially | Broadcast input embeddings; tied weight aliases must remain correct. Benchmark replication only if memory and saved communication justify it. |
| Target vocabulary head | One owner for first correctness implementation; then shard vocabulary rows | Greedy: reduce local `(max logit, global token ID)` with deterministic tie-breaking, not full vocabulary gathers. Full distributions need a separate correct path. |
| MTP NextN block and proposal head | One designated draft owner initially | Reuse replicated verified hidden state and exact token embeddings; broadcast candidates and commit decisions. Include head weights/aliases in memory accounting. |

This table applies to each of N ranks. Preserve global query-to-KV mapping;
when N exceeds KV-head count, replicate required KV heads and reject unsupported
local kernel mappings. For GDN inventory `ssm_group_count`, `ssm_time_step_rank`,
`ssm_state_size` and derived value dimension from config. Replicate repeated Q/K
groups when value-head partitions require them; do not simply divide both head
counts by N. Test gate/Q interleaving and convolution channel layout separately.

For vocabulary sharding, gather N local `(score, global ID)` candidates to the
control owner, select deterministically, and broadcast the winning ID as the
first implementation. This does not require a custom RCCL pair-reduction
operator. Full-logit numerical probes and initially supported stochastic
sampling gather the complete vocabulary on the owner outside timed greedy runs.
Keep one authoritative per-request RNG/processor state; shard-local selection is
valid only for processors whose semantics are proven to commute with that split.

Produce identical replicated post-reduction values on every rank: sum partials,
apply output bias once, add residual once, convert dtype at the declared boundary.
In an MLP-only diagnostic replicated attention outputs are already complete and
must not be summed again. FP32 communication after BF16 partial-output rounding
is not FP32 partial accumulation: qualify the actual kernel/dtype chain or add
an FP32-output variant.

The model inventory must specify every GGUF tensor's logical axes, byte/block
alignment, per-rank ranges, replications, and local dimensions. `Q4_K_M` is a
mixed-tensor preset, not a promise that every matrix uses Q4. Cutting a packed
row on its reduction axis may require block-aligned repacking and local stride
changes. Preserve original quantized blocks/scales without dequantizing and
requantizing to manufacture shards. If an axis cannot be partitioned safely,
replicate that operation first and record its serial cost.
Keep existing materializer metadata-only preflight, precision-contraction checks
and native-consumer validation. Slice original blocks first, then create only
needed rank-local T16/planar/other sidecars; existing TP1 repacks may not admit
the same cuts. Replicated-output operations need no sum; owner-only operations
need broadcast.

Use the RX 7900 XTX's 24 GB as the limiting rank budget, not half the combined
VRAM. Record actual free memory on both cards. Budget weights, draft copies,
KV, recurrent snapshots, communication buffers, prefill/verification workspaces,
graph buckets, and allocator headroom before loading. Start 50:50. Consider
unequal, block/head-aligned partitions only after per-rank timings show a
persistent imbalance; the W7900's larger VRAM does not imply faster compute.

### Numerical contract

A row-parallel split changes floating-point reduction order. Do not claim
bitwise single-GPU parity just because the weight bytes are unchanged.

- Write a CPU split-matrix/split-state oracle with a declared fixed reduction
  schedule. Validate indexing and tensor/state reconstruction independently of
  the optimized GPU path; compare intermediate layer boundaries to TP1.
- Preserve exact control/ownership in every profile. Verify repeat determinism
  for a fixed device pair, rank order, shard manifest, collective algorithm,
  dtype, and execution schedule.
- Preserve the registered TP1 strict fallback. Do not register a TP2 route as
  public `strict` unless it meets that profile's reference arithmetic contract.
  A deterministic split oracle alone does not certify public strict parity.
- Topology admission precedes profile fallback. A strict TP1 primitive can serve
  a local shard only with compatible shape/layout; it does not certify distributed
  sums. Keep an unfused distributed reference chain for localization and the TP1
  model as teacher. Reject explicit strict/batch-invariant TP=N without a
  certified group plan. Never silently load a full TP1 model on one rank or
  change degree mid-session; TP1 fallback requires a separately admitted session.
- Extend cold-path certification and serving evidence with the distributed
  manifest before public admission. The current default TP1 `production`
  selection cannot authorize TP=N by key coincidence. Record selected collective
  algorithm/protocol and rank-local variants; fixed rank order alone does not
  prove repeat determinism.
- Qualify changed arithmetic as `production` only through the full calibrated
  strict-teacher mean/tail/max KL, top-1, BF16-relative, isolation, and task gates
  in `EXECUTION-PROFILES.md`. The CPU-reference KL ≤ 0.05 / top-1 ≥ 90% smoke
  floor is necessary for kernels, not sufficient for promotion.
- Start with FP32 partial-output accumulation/communication for the oracle;
  measure the cost. Narrower transport or fused reduction is a separate
  numerical candidate, not an invisible communication optimization.

## Ordered coder punchlist

Each packet ends with focused tests, an immutable worklog entry, and a scoped
commit. New names below are proposed deliverables, not existing commands/APIs.
Packet 0's topology/microbench screen may share minimal primitives with Packet 1.
The final break-even decision uses Packet 2's validated local shapes and shard
kernel timings before committing to Packet 3; do not build duplicate temporary
communication wrappers or treat estimated half-model time as measured evidence.

### Packet 0 — Establish the host and break-even budget

- [ ] Record physical host identity; GPU UUID/PCI bus IDs and rank mapping;
  ROCm, driver, RCCL, compiler; NUMA placement; CPU affinity; display load;
  clocks/power/temperature; PCIe negotiated generation/width under load and
  root-complex topology. Record IOMMU/ACS settings without changing security
  settings merely to improve a benchmark.
- [ ] Check HIP is available and both devices are gfx1100. Test peer access
  independently in both directions, actual verified device-to-device copies,
  and bidirectional traffic. Identify staging rather than inferring P2P from
  nominal PCIe bandwidth. Missing direct P2P is a measured risk, not proof of
  impossibility.
- [ ] Inventory `/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` (or record replacement
  path/hash), all tensor formats, layer types, MTP block, hidden/head/state
  dimensions, and per-device memory budget. Bind the shard manifest to its hash.
- [ ] Add a guarded `scripts/tp_collective_bench.py` and tests accepting an
  ordered device list (proposed interface). Measure warm
  latency p50/p95/p99 and bandwidth for broadcast and sum, for FP32 and any
  proposed transport dtype: payload `rows * hidden_size * dtype_bytes`, rows
  1, 2, 3, 4, 5, plus actual prefill chunks. Test rank orders and many sequential
  reductions with a local producer/consumer, not just isolated copies.
- [ ] Measure matched TP1 AR and engaged TP1 MTP on each GPU, one at a time on
  this host, using the full category suite. Profile per-layer and head time.
  Do not substitute a result from another host, even with the same GPU model.
- [ ] Write a break-even artifact and a go/no-go decision before Packet 3.
- [ ] Use Tier-1 allocation probes before full-prompt capacity testing. Record
  actual TP1 HIP/PM4 submission transport and resolved profile; measure local
  shard-shaped kernels and rank enqueue skew before extrapolating.

For each segment between synchronization boundaries estimate:

`TP=N segment time ≈ max(local time over ranks) + exposed reduction + launch/synchronization cost`.

Sum segment costs across layers and add embeddings, head, serial operations,
and sampling. Do not take one maximum over an entire layer when ranks can
arrive at its two reductions with different skews. For a
fully sharded dense layer budget two full-hidden sums (attention output and
MLP down); an MLP-only prototype needs one. Derive the total from the actual
layer inventory. Communication is small in bytes at one row but repeated
through every layer; a dependency-bound sum cannot be hidden behind its own
consumer. Use measured shard kernel times rather than assuming half of TP1.
Stop full-model expansion if the optimistic bound cannot beat the faster TP1
arm; investigate the dominant measured cost, not blind kernel tuning.

### Packet 1 — Make device ownership and collectives safe

- [ ] Implement minimal distributed config/context and a transport interface in
  `hipengine/distributed/`; add only missing HIP device/peer operations in core.
  Keep world-size-one behavior unchanged and RCCL optional until TP is requested.
- [ ] Resolve the N-rank plan and composite KV pool/claim set. Preserve the
  model-owning scheduler; add a distributed runner adapter, not N schedulers.
- [ ] Audit core, weight loading, runtime workspaces, graphs, native cycle ABI,
  sampler, and global caches for implicit device zero/default stream ownership.
- [ ] Add CPU/mock tests for rank/device mismatch, invalid topology and dtype,
  failed initialization, partial teardown, and unsupported configurations.
  Add two-GPU guarded tests for sums, event ordering, repeated buffer reuse,
  concurrent streams, and cleanup. One-rank failure must not hang the process;
  use watchdog/abort behavior and explicit request failure, not silent recovery
  from half-committed distributed state.
- [ ] Establish a graph-disabled correctness baseline. Test RCCL capture support
  explicitly before graph integration; a Python loop that synchronizes every
  rank/layer is an oracle implementation, not the intended fast path.
  Even the oracle must enqueue all participants before a collective wait.

### Packet 2 — Build shard manifests and CPU reconstruction tests

- [ ] Add immutable shard descriptors and GGUF shard materialization. Validate
  axis/block alignment, head grouping, local shapes, aliases, offsets, and
  per-device memory before allocations. Reject unsupported layouts clearly.
- [ ] Test byte-preserving reconstruction for every quant type actually present,
  column and row cuts, output/gate pairing, tied heads, and shard boundaries.
- [ ] Test MLP, full-attention head mapping, convolution/GDN recurrence, and
  output projection against independent small fixtures, including warm state,
  multiple tokens, odd/unsupported dimensions, and snapshot restoration.
- [ ] Confirm a rank never allocates a second full target model as an accidental
  materialization intermediate. Account for host-side conversion copies too.
- [ ] Exercise N=1/2/3/4 CPU plans, uneven legal cuts, KV-head replication,
  impossible degrees, metadata-only refusals and aggregate claim rollback.
  Two-GPU success cannot mark N=3/4 hardware support complete.

### Packet 3 — Integrate AR in incremental boundaries

- [ ] Implement MLP-only TP2 with replicated attention as a diagnostic first
  slice. Confirm one down-projection reduction per layer and exact single
  residual addition; use boundary probes to localize drift.
- [ ] Add full-attention head sharding and then linear-attention group/state
  sharding. Verify cold and warm trajectories separately; sharding only QKV
  without its convolution/GDN state is not a complete implementation.
- [ ] Connect prefill, one-token decode, positions, KV allocation, reset, EOS,
  and resource teardown. Prefill must produce the same rank-local state layout
  consumed by decode, including chunk boundaries and long contexts.
- [ ] Run full-logit numerical gates and same-schedule repeats. Establish true
  TP2 AR with all draft allocation/execution disabled as the MTP denominator.
- [ ] Measure head cost. Add vocabulary-row sharding only if worthwhile. For
  greedy output preserve deterministic global tie-breaking; for stochastic
  sampling initially use a correct full-logit gather on one owner and one RNG
  stream, or explicitly declare the mode unsupported. Distributed top-k/top-p
  and speculative probability normalization require their own correctness gate.

Exit: a correct end-to-end TP2 AR runner, documented profile scope, two-GPU
memory/timing evidence, and a decision on whether to proceed with optimization.

### Packet 4 — Reduce exposed PCIe and host overhead

- [ ] Trace both ranks with cached builds. Attribute local GEMV/GEMM, GDN,
  collectives, rank skew, idle gaps, H2D/D2H copies, and host launch cost.
  Use common host wall for end-to-end latency; do not subtract unsynchronized
  timestamps from different GPUs as if they shared a clock.
- [ ] Capture stable per-device graph segments and collectives only where
  supported. Keep event dependencies explicit and verify repeated replay,
  address lifetimes, and graph invalidation. Do not assume one cross-device
  HIP graph works, or discard graphs without measuring the lost TP1 benefit.
- [ ] Use the existing submission registry. Qualify RCCL capture for the whole
  communicator group, consistent replay order and capture failure on any rank.
  Native PM4 `NativeGraphSubmission.launch()` currently waits on the caller's
  HIP stream and native completion; sequential rank calls can serialize compute.
  Do not export RCCL internal kernels into kernel-only PM4 manifests or assume
  HSA writes are ordered by HIP-stream events. A native TP path needs explicit
  cross-queue completion and independent stress/capture gates. Explicit
  unsupported transport requests fail closed; report automatic HIP selection.
- [ ] Compare RCCL to peer-copy/local-sum for actual payloads and the full layer
  chain. If justified, fuse local sum with residual/norm without double-adding
  residual; follow kernel catalog, lineage, strict fallback, numerical, and
  `rocprofv3 --kernel-trace` gates for each new kernel.
- [ ] Tune actual rank-local GEMV and verifier shapes through the four-axis
  registry. Smaller shards may hit different performance regimes than TP1.
- [ ] Screen aligned unequal shards only if rank skew warrants it; retest the
  complete manifest and numerical contract for each split. Drop rejected paths
  or record precise removal conditions in `REFACTOR.md`.

### Packet 5 — Make MTP economically useful at one active request

- [ ] Start with one draft owner and TP2 target verification. Benchmark each
  GPU as draft owner; do not choose by VRAM size alone. Keep hidden seeds,
  proposal weights, and recurrent draft state resident there. Do not copy
  full-vocabulary logits to the CPU each proposal step in the optimized path.
- [ ] Broadcast candidate IDs/positions and a single authoritative accept,
  reject, bonus-token, EOS, and committed-prefix decision. Both ranks must
  execute collectives in the same order even on early rejection/cancellation.
- [ ] Verify the chain as one multi-row target pass, amortizing collective
  latency across verifier rows. A single request with K candidates is not K
  concurrent requests. Record the actual verifier row count (including any
  root/bonus convention), graph bucket, and engaged route.
- [ ] Make target KV and convolution/GDN snapshots transactional on both ranks.
  Test rejection at every depth, all accepted, zero accepted, EOS in a proposal,
  context/chunk boundaries, repeated rollback, cancellation, and clean drain.
  Commit the same prefix everywhere; discard all rejected state and preserve
  the verified hidden seed at the exact boundary expected by NextN. Restore or
  advance the draft owner's recurrent state consistently with that prefix too;
  target rollback alone is insufficient.
- [ ] Establish actual width-one provider engagement and TP-specific evidence
  admission without weakening production admission. If needed, add a test-only
  screening mode that labels unqualified cells and cannot enable public automatic
  MTP.
  Zero engaged cycles invalidate an MTP speed claim.
- [ ] Screen K=0,1,2,3,4 only where implemented and actually engaged. Measure
  draft, candidate broadcast, verifier, head, acceptance, rollback/commit,
  collective count/bytes, and committed tokens per cycle. Qualification must
  cover all categories and heldouts, not the best acceptance prompt.
- [ ] Only after this baseline, compare sharding the draft itself or overlapping
  genuinely independent work. A sequential one-block draft may lose more to
  PCIe than it gains; target/draft data dependencies forbid assuming free overlap.
- [ ] Extend serving-evidence keys to distinguish TP degree, device pair/rank
  map, shard/transport and variant manifests, context, verifier shape, K,
  sampling and profile. Never reuse a TP1 qualification record for TP2.
- [ ] Use existing provider/frontier/transaction integration. The native
  single-device cycle descriptor is not a distributed ABI: use per-rank
  descriptors and group-owned commit coordination before optimizing host control.
  Apply sampling constraints/processors before local argmax when sharding the
  vocabulary. Raw-logit top-1 is not valid for arbitrary public sampling options.
  Mask padded vocabulary rows and define global-ID tie behavior.

Economics check: `MTP time per committed token = cycle wall / committed tokens`.
Include all draft, verification, communication, and recovery work. MTP wins only
when this is below matched TP2 AR time per token. Acceptance alone is not a win;
TP2 may accelerate AR more than the serial draft, reducing the optimal K.
Keep K0 as the automatic fallback outside qualified winning scopes.

### Packet 6 — Qualify the public path and publish the result

- [ ] Wire explicit TP configuration through model construction, `LLM.generate`,
  and serving. Reject unsupported hardware/model/profile/sampling combinations
  before loading; expose the resolved topology and MTP engagement in diagnostics.
- [ ] Publish capability per model/profile/degree/transport, not a global flag
  inferred from visible GPUs. An explicit ordered device list defines rank
  mapping; validate its length against any degree selector. Mock tests and TP2
  evidence do not authorize public N>2 hardware support.
- [ ] Test streaming/non-streaming output, cancellation, request reuse, error
  propagation, and final memory/ownership drain. Initial support may admit only
  one active request; reject/queue extra work explicitly rather than silently
  using an unqualified multi-request plan.
- [ ] Run the matrix below with exact commands and clean provenance. Promote
  winning qualified TP2/MTP paths within explicit two-GPU selection; do not
  silently claim a second GPU for an ordinary single-device invocation.
- [ ] Save compact artifacts under `benchmarks/results/`, update benchmark
  README date/rows and changelog for every retained measurement, run the README
  export check, and record blockers and cleanup conditions durably.

## Binding benchmark and correctness matrix

Use `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, all `code`,
`general_en`, `general_ja`, and `mixed_ja_en` prompts, plus category-heldouts
fixed before tuning. Extend existing GGUF suite/server harnesses rather than
inventing TP command-line flags in documentation before they exist. The new
harness must emit exact reproducible invocations and resolved route metadata.

| Arm | Purpose |
| --- | --- |
| W7900 TP1 AR / TP1 MTP | Same-host control and draft economics. |
| RX 7900 XTX TP1 AR / TP1 MTP | Other single-device control; determine the faster valid baseline per shape. |
| Pair TP2 AR | True no-MTP denominator; also compare with best TP1 AR. |
| Pair TP2 + MTP at each engaged K | Compare with TP2 AR and best valid TP1 AR/MTP. |

Hold model hash, quant, profile, KV/state storage, prompts, tokenization,
sampling, context/output lengths, and compilation policy fixed across matched
arms. Different shard manifests are expected and must be recorded. If a TP1
arm cannot fit a shape, report OOM/capacity advantage separately; no invented
speed ratio. Keep the unused GPU idle during TP1 controls.

Predeclare these shapes, or document a resource-driven change before measuring:

- Primary concurrency C=1, context lengths 128, 512, 2048, 8192 tokens;
  generated horizons 128 and 512. Run the repository's short canonical protocol
  as a compatibility row, not a substitute for sustained decode.
- Use deterministic context construction/token truncation for the complete
  category suite; disclose lengths after tokenization. Add boundary fixtures
  for page, prefill-chunk and verifier transitions. Test larger contexts only
  inside the measured per-rank memory envelope.
- Greedy first. Gate seeded sampling separately before advertising it; require
  correct target/proposal distributions and acceptance semantics, not greedy
  token equality as a proxy for stochastic correctness.
- At least three balanced paired repetitions after warmup and stable clocks,
  both arm orders. Expand repetitions when the uncertainty overlaps no gain.
  Run correctness probes separately so full-logit collection does not distort
  the performance path.

Report per-prompt/category and aggregate decode tok/s, committed token count,
wall time, time to first token, inter-token p50/p95/p99, prefill rate, peak VRAM
per rank, MTP engagement/acceptance and cycle breakdown. Distinguish bursty MTP
stream emission latency from amortized time per committed token. Report spread
and repeatability, not a best-run rate. Pair every artifact with exact host,
GPU topology, commands, model hash, profile/variant/shard manifests, source
revision, warmup/repetitions and correctness evidence.

Correctness progression: CPU fixtures → two-GPU primitives → layer boundaries
→ full-model teacher-forced numerical trajectories → AR generation → MTP
transactions → public lifecycle. TP2 MTP must obey the target sampling contract;
require greedy AR/MTP equality where the declared arithmetic contract binds it,
and the full production-profile gates where width-dependent arithmetic is
allowed. Never treat distributed ownership divergence as numerical drift.

Add explicit HIP/two-device skips to hardware tests so CPU-only CI remains
usable. Run focused tests for each packet, the applicable deterministic bundle
for shared host changes, and the milestone gate from `TESTING.md` for closure.
Prebuild outside `rocprofv3`, use compiler-version/cache-only execution, and
profile final leaf processes rather than the multi-process suite parent.

## Completion audit

- [ ] Both ranks demonstrably execute local target shards and the expected
  reductions; no hidden full-model duplicate or CPU-staged fast-path claim.
- [ ] The complete shard/state, numerical, determinism, MTP transaction and
  public lifecycle gates pass in the advertised scope.
- [ ] Same-host evidence separates TP2 AR speedup, incremental MTP speedup, and
  total speedup over the best single-GPU option. Losing cells select K0/TP1
  appropriately; unsupported cells fail closed.
- [ ] Every retained improvement is enabled in its validated scope, or a
  concrete blocker is documented. No blanket 1.3x hurdle discards smaller wins.
- [ ] Campaign checkboxes, immutable handoff, benchmark artifacts/rollups,
  architecture notes, and cleanup ledger match the implementation. If no TP2
  win is possible on this PCIe topology, close with a measured negative result
  and its limiting costs rather than claiming success from capacity alone.

## External API checks

Reviewed against AMD official documentation on 2026-09-14. Recheck the installed
version in Packet 0: documentation does not certify this Radeon topology's peer
access, capture behavior or throughput.

- HIP multi-device management: device selection, stream/event ownership and
  possible host staging without peer access.
  `https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/multi_device.html`
- RCCL API library: rank collectives and single-thread group semantics.
  `https://rocm.docs.amd.com/projects/rccl/en/docs-6.4.2/api-reference/api-library.html`
- RCCL group calls: group enqueue versus stream completion.
  `https://rocmdocs.amd.com/projects/rccl/en/develop/userguide/source/api/group.html`
- RCCL fault tolerance and communicator progress: nonblocking progress, abort
  ownership, and deadline/error polling.
  `https://rocm.docs.amd.com/projects/rccl/en/latest/how-to/fault-tolerance.html`
  `https://rocmdocs.amd.com/projects/rccl/en/develop/userguide/source/api/comms.html`
