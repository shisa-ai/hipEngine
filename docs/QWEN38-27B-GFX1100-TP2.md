# Tensor parallelism: TP=N architecture, Qwen3.8-27B TP2 bring-up

Status: implementation plan; no public tensor-parallel support and no TP
speedup is claimed. Packet 0 topology/collective screening, Packet 1 rank-bound
transport, and Packet 2 shard planning are measured or implemented, and an
MLP-only TP2 diagnostic session (`MlpTP2GenerationSession`: replicated
attention/GDN, sharded MLP, staged partial exchange) runs the full model; its
per-GPU TP1 correctness controls were re-qualified on 2026-09-16 over an
18-prompt/831-position teacher-forced suite. See "Measured status" below.
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

**External cross-engine checkpoint (2026-09-19).** The same
`Qwen3.8-27B-Q4_K_M.gguf` on the same host, c=1, 512-token prompt, 128 decode
tokens, f16 KV, no speculative decoding, measured against the
`llama.cpp-rdna3-opt` RDNA3 fork at build `15995a1`:

| Route | Prefill tok/s | Decode tok/s | Device memory |
| --- | ---: | ---: | ---: |
| llama.cpp TP=1 (W7900) | 941.8 | 30.44 | 15.38 GiB |
| hipEngine TP1 (W7900, bulk prefill) | 875.8 | 30.76 | 17.70 GiB |
| llama.cpp TP=2 `-sm tensor` | 1474.6 | 41.30 | 7.75 GiB/rank |
| hipEngine TP2 | 39.5 | 38.8 | 22.37 GiB/rank |

Decode is at parity with the fork's tensor split (38.8 vs 41.3 tok/s) and
single-card prefill/decode are within 7%/1%. Two gaps are quantified rather
than assumed: TP2 prefill is 0.045x this project's own single-card bulk route
because the session is still token-serial (the P1/P2 plan below), and TP2 does
not reduce per-rank residency — 22.37 GiB/rank measured against 7.83 GiB/rank
of planned resident weights from the shard manifest, leaving only 1.58 GiB free
on the 24 GiB XTX rank. Full protocol, commands, and artifacts:
[`benchmarks/HISTORY.md`](../benchmarks/HISTORY.md) "Qwen3.8-27B dense Q4_K_M
TP1/TP2 vs llama.cpp RDNA3 fork" and
[`2026-09-19-w7900-qwen38-27b-tp1-tp2-hipengine-vs-llamacpp.json`](../benchmarks/results/2026-09-19-w7900-qwen38-27b-tp1-tp2-hipengine-vs-llamacpp.json).

Packets 0-2 are measured on the target host (Ryzen 9 5950X, W7900 at
`0000:0d:00.0` + RX 7900 XTX at `0000:10:00.0`, both `gfx1100`, separate CPU root
ports, PCIe 4.0 x16 confirmed under load). Artifacts live under
`benchmarks/results/tp2_*.json`; the numbers and their scope are in
`benchmarks/README.md` and the worklog entry for the unit.

- **Peer DMA is unavailable on this host.** `hipDeviceCanAccessPeer` is false in
  both directions and `hipDeviceEnablePeerAccess` fails with HIP error 101. Both
  cards expose a 256 MB BAR even though the kernel advertises a resize attribute
  (`resource0_resize`), and every bridge between the two cards — the CPU root
  ports `00:03.1`/`00:03.2` and the downstream bridges `0c:00.0`/`0f:00.0` — has
  ACS redirect bits set (`ACSCtl` reads `SrcValid+ TransBlk- ReqRedir+
  CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans-`), which redirects peer TLPs
  to the root complex instead of forwarding them. Both cards are trained at PCIe
  4.0 x16 (`LnkSta: Speed 16GT/s, Width x16`; `pp_dpm_pcie`'s `x8` is a DPM
  capability table, not the live link), so link width is not the limiter.
  Collectives therefore host-stage at ~8 GB/s of payload.
  **Host-level action**: these two observations are consistent with the failure,
  but neither is a proven cause and their sufficiency is unverified — the root
  complex can forward a transaction even with ACS redirect bits set, and a
  resized BAR alone does not establish a working cross-root-port path. Changing
  either one means changing firmware or IOMMU isolation on a shared host, so it
  is the human lead's call and not a benchmark-time change. Until such a change
  is measured, the peer-copy path in "Runtime and communication" is not a
  candidate, and the bf16 transport measurements below (2.593 -> 1.309 ms per
  1024-row all-reduce) are transport-level diagnostics rather than a qualified
  model-level prefill default.
- **Collective latency is the binding constraint for decode, and RCCL's eager path is
too expensive for it.** The shard inventory fixes the count at **128 row-split
tensors per token** (two per transformer block across 64 autoregressive blocks:
64 MLP down projections, 48 GDN state-output projections, 16 attention output
projections). In a *dependent* chain, where each reduction runs in its own group
and its result feeds the next layer, a 20 KB fp32 all-reduce costs **177.3 us per
reduction**, so a decode token spends **22.69 ms** in exposed collectives
(`benchmarks/results/2026-09-14-w7900-tp2-dependent-reduction-chain.json`).
Against the faster matched single-GPU arm (35.36 tok/s, 28.28 ms/token) that is
80% of a whole token before any weight is read, and the break-even projection
lands at **0.61-0.68x** - TP2 slower than one GPU, not faster. RCCL's per-step
groups must get **2.4-4.5x cheaper** for the eager path to break even. A
two-rank host-staged exchange with batched submission reaches **42.9 us** and
projects **0.99-1.17x**; that projection is conditional - the artifact is
uncertified and no local shard kernel or model-level measurement backs it - but
it is the only structure measured here that reaches 1.0x at all, and RCCL's
per-step groups do not.

  **Four transport levers are now measured, and the host orchestration of the
  exchange is the largest single one.** Every structure below reproduces the
  closed form `seed * 2 ** depth`, which only holds if every reduction consumed
  its predecessor, so these are per-layer costs rather than deferrable ones.

  | Per-reduction structure | Cost | 128 reductions | Speedup at 0% fixed share |
  | --- | ---: | ---: | ---: |
  | RCCL, one group per reduction, with an intermediate device copy | 177.6 us | 22.73 ms | 0.680x |
  | RCCL, one group per reduction, copy removed | 153.9 us | 19.70 ms | 0.738x |
  | RCCL, copy removed, replayed from a captured graph | 147.0 us | 18.82 ms | 0.757x |
  | Host exchange, one rank at a time (submit, wait, submit, wait) | 70.3 us | 9.00 ms | 1.022x |
  | Host exchange, both ranks submitted before either is awaited | 40.0 us | 5.12 ms | 1.181x |
  | Host exchange, same protocol driven from a native C++ loop | **20.5 us** | **2.62 ms** | **1.328x** |

  The intermediate device copy costs **23.6 us per reduction**, so the copy-free
  protocol is the right default for any RCCL-based chain. Host submission is only
  **6.9 us** - that is what replaying the same device structure from a captured
  graph removes - so RCCL's per-reduction cost is device-side protocol, not
  Python or ctypes overhead. Collapsing N dependent reductions into a single
  group saves 119.6 us, which is why that structure is fast and why it cannot
  carry a layer dependency.

  **Host orchestration is worth 28.8 us per reduction on its own.** The serial
  exchange performs four host waits per reduction - submit and wait for rank 0,
  then for rank 1, then the same for the return copies - and the two transfers of
  a pair never overlap. Submitting both device-to-host copies before awaiting
  either, and not awaiting the return copies at all, costs two waits per
  reduction and moves the projection from **0.88-1.02x to 0.99-1.17x**. The
  return copies need no host wait: the device-to-host copy for a step is
  enqueued after the return copy that read the same staging slot on that rank's
  stream, so the wait the host already performs is also the slot-reuse guard.

  Host orchestration is worth **29.3 us per reduction**, split by a third arm
  that keeps the batching and restores the return wait: **17.7 us is rank
  batching and 11.6 us is dropping the return wait**. The instrumented arm's wall
  clock and phases cover the same steps, and they account for 41.9 of 44.3 us, so
  the residual is 2.3 us of helper and loop overhead rather than a missing phase.
  The phases now name a **15.2 us device-context entry/exit term** that earlier
  counters omitted entirely - more than either copy submission - so no native
  ceiling follows from subtracting phases. Copies are probed per rank: 13.6-15.1 us
  for a single copy on an idle stream, 8.2-9.9 us per copy when 16 are submitted
  back to back inside one event pair. Both are probe observations, not a trace of
  the chain's critical path, and rank 1 is consistently the slower card. Each
  staged byte crosses PCIe twice, so a peer-DMA path would halve the copy term,
  and peer DMA is unavailable on this host.

  The Python arm's projection clears 1.0x at 0%, 10% and 20% fixed-cost share and
  reaches 0.994x at 30%. Driving the identical protocol from a native C++ loop
  costs **20.45 us per reduction instead of 40.00**, and its projection clears 1.0x
  in every row and reaches **1.328x at 0% fixed share**, above the design's 1.3x
  target. The comparison is now **matched rather than artifact-imported**:
  `scripts/tp_staged_exchange_native_ab.py` reruns both arms in one session over
  the same depth ladder with alternating order per repetition, and reports the
  ratio only after both arms agree on payload, depths, protocol, physical devices
  and repetition count (artifact
  `benchmarks/results/2026-09-14-w7900-tp2-staged-exchange-native-ab.json`). The
  term-by-term picture is not a simple subtraction: submission and device scoping
  fall from 26.3 to 3.3 us and the host sum from 6.6 to 2.0 us, but the exposed
  wait **rises** from 8.9 to 14.9 us because earlier submission changes what is
  exposed. These are still **conditional projections**: the artifacts
  record `certified: false` with no shard-kernel evidence, the native runner is a
  standalone program rather than engine code, and the fixed-cost share is an
  assumption. The transport budget is now measured as affordable; nothing here is
  a qualified model result. The 1.3x target is an aspiration, and the design accepts smaller
  qualified wins; whether the exchange can supply one is undecided until the local
  segments and an in-chain copy trace exist. The native runner's timed and
  verified passes share one `step` implementation, so the structure that is timed
  is the structure that is checked: both ranks are read back over the whole
  5120-element vector, nonfinite values fail the run explicitly, and the timed
  recurrence is verified exactly at every depth through 64 while depth 128 is
  reported as fp32-saturating rather than passed.

  Capturing the dependency-bearing chain works: 128 reductions, one native group
  each, captured into a HIP graph and replayed, with the closed form reproduced
  and a `0xFF`-poisoned tail buffer proving the replay did the work. It saves
  6.9 us per reduction, which is the measured host submission cost. Capturing the
  whole chain into a *single* group is both invalid (the reductions collapse) and
  unsafe on this host (a `Memory access fault` at depth 32), so it is opt-in and
  excluded from a default run.

  An earlier estimate of 3.66-4.48 ms (11-13% of a token) used a marginal
  measured from collectives that were *not* dependent: a single group holding the
  whole chain costs 34-37 us per step, but that structure cannot carry a layer
  dependency - the reductions collapse (depth 4 returns the depth-2 value, depth
  16 returns roughly the depth-10 value) and the closed form is not reached. The
  per-step structure, which can carry the dependency, costs 177.3 us - 4.8x more.
  The earlier figure is superseded, not merely refined.

  Group enqueue already satisfies the "first rank's collective must not block
the second rank's enqueue" requirement, and threaded enqueue is measurably
worse, so no host threads are needed for enqueue.
- **A TP2 rank holds half the KV pool.** At 8192 context a rank claims 258 MiB
  of KV against 514 MiB for the whole pool (16 full-attention layers, 2 of the 4
  KV heads, 32 KiB per token from the declared 256-wide K and V planes, plus
  2 MiB of `KVLiveSpans` under the dense-policy layout of four 32-bit fields per
  token per layer), and 1032 against 2056 MiB at 32768. Geometry is
  backend-owned: `key_length` and `value_length` are read separately from the
  config, so unequal K/V widths are sized correctly and an inferred value width
  is recorded as inferred; the metadata term comes from a declared
  `KvSpansLayout` (`paged_uniform`, `per_head_variable`, `sliding_ring`, or the
  historical `dense_policy`) because those modes carry different tensors and
  only some of them are per token. The per-rank head partition is the weight
  planner's own `partition_groups` result, so a rank's KV heads are exactly the
  heads its weights serve, and N=3 is refused with the planner's message. When a
  group exceeds the KV-head count the assignment replicates by *block* -
  consecutive ranks share a head - so each rank still holds the KV heads
  covering its own query-head block; round-robin replication would hand a rank a
  head its queries never attend to.

  Admission is all-or-nothing and ledger-owned: each rank's KV region is a
  stable `KVPoolPlan` of byte pools, `reserve_group_kv` takes a provisional
  reservation in every rank's `ResourceLedger` and commits only when all ranks
  reserved, a failure on any rank rolls back the earlier holds, and a second
  group cannot claim the same rank's pools. `claim_all` drives the device
  allocator after the ledger commits.
- **RCCL work can be captured into a HIP graph, up to a size limit.** With
  communicator creation outside capture and each rank's whole chain captured on
  its own stream, 40/40 probes across chain depths 1/4/8/16/24 replayed
  bit-identically to the graph-disabled result, and replay is 20-30% faster than
  eager enqueue (a 24-op group: 1.05-1.19 -> 0.72-0.88 ms). The limit matters:
  49 captured nodes (24 ops plus their producer/consumer memsets) works, 65
  nodes (32 ops) faults the device with a memory access error, so a TP2 decode
  step's 128 collectives must be split across several graphs rather than captured
  as one.
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

Use RCCL as the reference communication implementation. The specialized two-rank
screen has now been run, and it is the only structure measured on this host that
projects a win: a page-locked host exchange with a host-side sum, both ranks
submitted before either is awaited, costs **42.9 us per reduction** against
**153.7 us** for the copy-free RCCL per-step chain, both dependency-verified
(`benchmarks/results/2026-09-14-w7900-tp2-dependent-reduction-chain.json`). The
submission structure is the largest single lever in the transport - one rank at a
time costs 71.7 us - and the return copies need no host wait, because the
device-to-host copy that follows them on the same stream is already awaited and
is therefore also the staging-slot reuse guard.

It is a measured candidate rather than a fast path: it still must have explicit
producer/consumer events, reusable buffer lifetime rules, and stress tests before
serving, and its per-reduction cost is 30.8 us of copy time plus 26.4 us of host
work. Peer accessibility alone does not establish ordering or coherent polling
semantics. Do not begin with persistent GPU spin-wait barriers. Keep a registered
strict fallback for any fused kernel. A host-staged transport must be qualified
as a serving path before it is a default; today it is a screen result.

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

## Rank-local bulk TP2 prefill design (2026-09-17)

The TP2 session currently drives prefill token-by-token
(`MlpTP2GenerationSession._forward_token(..., kind='prefill')`), while the
optimized TP1 product path uses the resident bulk prefill schedule. The
token-serial schedule is a registered fallback, not the bulk path. The following
is the minimal bounded plan for rank-local batched/bulk TP2 prefill using
in-tree primitives. It preserves actual sharded TP2 execution: the MLP stays
column/row sharded with the cross-rank reduction, and the optimized TP1
denominator is not bypassed by running a full TP1 model and copying hidden
state.

### Model geometry (from the supplied GGUF, not hardcoded by name)

`/models/gguf/Qwen3.8-27B-Q4_K_M.gguf` (SHA-256
`7b2aec3b9ababdfd75aa17552ee95607d866e44decf547f6f12fcef85cc89f1b`), 64 layers:
48 `linear_attention` (GDN) and 16 `full_attention` at interval 4.

| Axis | Value |
| --- | --- |
| hidden / ffn / vocab | 5120 / 17408 / 248320 |
| full attention | 24 q heads, 4 kv heads, key/value length 256, output width 6144 |
| linear attention | `ssm_inner_size` 6144, `ssm_group_count` 16, `ssm_time_step_rank` 48, `ssm_state_size` 128, `ssm_conv_kernel` 4, `linear_qkv_width` 10240 |
| MLP shard (N=2) | `per_rank_ffn` 8704 (17408/2, 34x256 block-aligned) |
| vocab head shard (N=2) | 124160 rows/rank |

Replicated per rank in the current diagnostic: attention and GDN (with the
explicit tiled value-head mapping) and the final norm. Sharded: MLP gate/up
(column, output/intermediate axis) and down (row, input axis), and the target
vocab head. Residual and post-attention norm are replicated; the residual is
added exactly once after the reduction.

### State and metadata to preserve

- **KV**: `KVLiveSpans` `(base_offsets, live_counts, token_positions,
  evict_mask)` stays the attention ABI. Bulk prefill writes all prompt rows for
the full-attention layers; logical positions and causal visibility must match
the serial schedule. Per-rank KV stays the rank's owned heads.
- **Conv state**: `(ssm_conv_kernel, linear_qkv_width)` = `(4, 10240)` per
  linear layer; final value after the chunk must equal the serial final state
  (up to the declared arithmetic contract).
- **GDN recurrent state**: `(ssm_time_step_rank, ssm_state_size, ssm_state_size)`
  = `(48, 128, 128)` per linear layer; the chunk's final recurrent state is the
  decode entry state.
- **Residual/hidden chain**: per-chunk `(rows, hidden)` bf16 residual stream;
  the swap/ownership discipline must match the eager layer loop.
- **Exchange payload**: down partials are `(rows, hidden)` f32 (or the declared
  partial dtype) per layer; the reduction is over the shard axis, not rows.

### Bounded packets

1. **P0 (this unit) — CPU planning + reference + RED tests.** A pure module
   (`hipengine/distributed/tp2_prefill.py`) that plans per-layer batched shapes,
   chunk boundaries, MLP shard dims, KV/conv/GDN final-state shapes, and an
   independent numpy reference for the batched sharded MLP reduction. RED CPU
   tests assert ownership, shape, row independence, chunk boundaries, and
   reference equality. No device calls, no arithmetic change.
2. **P1 — Batched rank-local MLP shard.** Extend `MlpShardRank`/`MlpShardGroup`
   to `rows > 1`: row-major persistent buffers, prefill GEMM launches
   (`use_gemv_decode=False`) for gate/up/SiLU/down, and a batched reduction
   payload `rows * hidden`. The strict token-serial path stays registered.
3. **P2 — Batched replicated attention/GDN prefill per rank.** Reuse the
   resident prefill primitives (`_run_linear_attention_prefill_layer_rows`,
   full-attention prefill) per rank with KV span writes and final conv/GDN
   state, then the prefill -> graph-decode transition (state reset, positions,
   liveness, multi-GPU streams). **P2 blocker found (2026-09-17):** those
   resident functions are full-width monolithic layer functions — they run
   attention/GDN *and* the full-width MLP *and* the residual in one call
   (`qwen35_gguf_runner.py` `_run_linear_attention_prefill_layer_rows` ~L8015,
   `_run_full_attention_prefill_layer_aotriton` ~L4553). Calling them from TP2
   would be exactly the forbidden full-TP1 forward + hidden-state copy. P2 must
   first factor the attention/GDN prefill subgraph out of the monolithic layer
   function (preserving its `commit_final_linear_state` / chunk-metadata
   behavior) and then run the sharded batched MLP + reduction, or add a
   sharded-MLP layer variant; this is a decomposition unit, not a wiring unit.
4. **P3 — Validation.** Bounded GPU probes at the saved 64-token prompt /
   position 146 plus at least one category-heldout control, then the full D128
   sustained gate only after the failure is repaired. Kernel numerics that are
   not bit-exact are evaluated against the production contract, not rejected.

### P1 status (2026-09-17)

P1 is implemented and CPU+GPU validated at the MLP level, and the whole bulk
route is still partial (no batched attention/GDN yet, no end-to-end quality
run).

- `MlpShardRank` gained an opt-in integer `rows` capacity and an
  `active_rows` count; `write_input`, `write_input_from_device`, `read_input`,
  `read_partial`, and `forward_partial` either use the declared active count
  exactly or reject the call before any launch. Non-integer/bool row values and
  over-capacity values are rejected.
- `MlpShardGroup`, `StagedExchangeTransport`, and `CompiledStagedExchangeTransport`
  carry the capacity; the staged exchange stages and sums exactly the active
  rows and zeroes the inactive tail, and the group casts the full capacity so
  the bf16 output tail is explicit zeros. The capacity-1 single-row route keeps
  calling the original ABIs unchanged.
- The compiled host driver (`staged_exchange_host.cpp`) gained
  `tp2_staged_reduce_rows` / `tp2_staged_reduce_at_rows` and a capacity-rows
  create argument; the device-graph exchange stays the single-row decode route
  and rejects `rows != 1`.
- **Dispatch finding:** the real launcher rewrites `rows>1` to
  `t16_wmma_prefill` (Q4_K gate/up) and `t16_gemv_rowtile` (Q6_K down) leaves
  that `resolve_gguf_linear_dispatch` does not name, so the shard preflight is a
  dtype/layout support check, not a rows-aware proof. Q4_K t16 has no f32-output
  dispatch surface and fails before launch.
- **GPU evidence:** `scripts/tp2_batched_prefill_probe.py` on the real GGUF
  shard quants (Q4_K t16 gate/up, Q6_K t16 qmicro planar down) at capacity 4
  passes batch-composition invariance for active rows 1–4 (active 1 exact;
  2–4 ≤ 1.9e-04 relative) and zeroes the inactive tail, for both the Python and
  compiled drivers (`benchmarks/results/2026-09-17-w7900-tp2-batched-prefill-probe.json`).

### P2 status (2026-09-17)

The rank-local bulk prefill path is implemented end to end and opt-in
(`MlpTP2GenerationSession(bulk_prefill=True, bulk_prefill_rows=N)`), but its GPU
diagnostic **fails the unchanged production numerical envelope**, so it stays
default-off and the token-serial route remains the committed prefill schedule.

- **Composition:** each layer runs the attention/GDN prefill helper, the
  post-attention norm+residual helper, one batched sharded MLP exchange, and one
  residual add per rank, on per-rank bulk scratch. Full-attention layers read
  the shared decode KV cache; linear-attention layers read and commit the shared
  decode conv/recurrent state, so the prefill state is visible to the captured
  graph decode. The bulk MLP group shares the decode group's uploaded shard
  weights (`owns_weights=False`) so `close` cannot double-free them, and every
  bulk buffer is appended to its rank's `_extra_buffers`.
- **Order:** `bulk_prefill` builds the decode graph schedule FIRST, then zeroes
  state, then writes the prompt KV/GDN state, so capture/warmup cannot overwrite
  the prefill state. Every call zeroes state, so the route is repeatable and
  never inherits a previous sequence. An over-capacity prompt is rejected before
  any allocation or launch (chunked bulk prefill is not implemented).
- **CPU tests:** `tests/test_unit_distributed_tp2_generate.py` covers call
  order, exactly one MLP+residual per layer, over-capacity rejection before
  launch, weight-sharing/no-double-free, state reset/repeatability, the
  prefill→decode transition position, the schedule-before-state order, and
  poison-on-failure.
- **GPU evidence:** `scripts/tp2_bulk_prefill_diagnostic.py` runs the saved
  teacher protocol (prefill the prompt, then feed the saved forced-decode
  tokens) on both a bulk and a token-serial arm of the same session. Measured
  pre-fix on `92e9d5b98` in a clean checkout, bulk-vs-teacher max KL was
  **0.48714** on `mixed_ja_en_translate` and **0.05573** on the heldout
  `heldout_mixed_summary`; the token-serial arm of the same session scored
  **0.551179** and **0.033202** against the same teacher, so the bulk route was
  far outside the token-serial route's own envelope. The bounded
  first-divergence investigation localized the bulk prefill's first numerical
  difference to the **layer-0 linear-attention output**, and a separate
  control/metadata defect was fixed on the way: the capacity-sized bulk scratch
  was passed to the full-attention prefill helper, which derives its row count
  from `scratch.rows`, so a 64-token prompt on a 200-row capacity ran the
  full-attention layers at 200 rows and wrote KV for 200 positions;
  `bulk_prefill` now narrows the scratch with `for_chunk(0, rows, rows)` exactly
  like the resident bulk caller. That fix does not change the failing-prompt
  logits (byte-identical sha256)
  (`benchmarks/results/2026-09-17-w7900-tp2-bulk-prefill-diagnostic.json`).
- **Direct teacher comparison and root cause (2026-09-17):** the failing quality
  gate compares TP2 bulk against the **resident TP1 bulk teacher**, so
  `scripts/tp2_bulk_vs_resident_layer0.py` compares those two directly on the
  same 64-token prompt in one process under the production profile, reading each
  layer-0 field immediately after its own producer and on its own stream, with
  layouts derived from the producer (`scripts/tp2_layer0_capture.py`; the bulk
  prefill scratch reports `allocation_mode=dedicated`, and the earlier
  `conv_out`/`linear_qkv` captures had been dtype-misread as bf16). With
  identical layer input (sha256 `306c076e…`), identical layer-0 weights,
  identical all-zero initial conv/recurrent state and the same
  `chain_compact_peer_wave32` GDN mode, the earliest differing operation was the
  **Q6_K `attn_qkv` projection**: `linear_qkv` (bf16 `[64, 10240]`) differed by
  rel **3.73e-03** / max_abs 0.25 on a 67.0 peak, while `linear_z` (the Q4_K
  `attn_gate` projection, bf16 `[64, 6144]`) from the **same launch group** was
  **bit-identical**. Verified cause: `MlpTP2GenerationSession.bulk_prefill`
  called the shared resident layer helpers directly, while
  `Qwen35GGUFResidentSession` wraps its whole bulk prefill in eleven
  session-scoped GGUF linear dispatch owners. Six of those are plain
  process-global toggles that participate in `launch_gguf_linear`'s dispatch
  resolution **and its dispatch cache key**, so the same Q6_K weight at
  `rows=64` resolved `t16_wmma_prefill_bf16_bf16_out` on the teacher and
  `t16_gemv_decode_bf16_bf16_out` on the candidate. The Q4_K `attn_gate` shape
  already resolved its WMMA variant without the context, which is why it stayed
  bit-identical.
- **Dispatch-context fix (2026-09-17):**
  `resident_prefill_dispatch_session` in
  `hipengine/runtime/qwen35_gguf_runner.py` is now the single source of those
  six device-free owners (`q8_t16_two_wave_prefill`, `wmma_prefill`,
  `gemv_decode`, `q8_t16_dual_wmma_prefill`,
  `q4_pack8_dual_wmma_silu_prefill`, `q4_t16_unequal_pair_prefill`). The
  resident session enters it once for its whole bulk prefill; the TP2 bulk path
  enters it **per rank** around each rank's attention/GDN helper and
  post-attention norm+residual helper, and once around the shard-group forward
  (whose owners are rank-invariant and hold no device pointers). The five
  remaining resident owners (f16 staging, Q6 integer MMQ, IQ dense MMQ, Q8 MMQ
  workspace, Q6 f16 rocBLAS) bind session-owned scratch and stay with the
  resident session. The shipped `wmma_prefill` policy moved next to the toggles
  it feeds (`resident_session_wmma_prefill_default`), so the resident sessions
  and the route that replaces their bulk prefill read one policy. The single-row
  decode route is untouched: it passes `use_gemv_decode` explicitly per launch
  and never enters the bulk owners.
- **Post-fix layer-0 result (2026-09-17):** all **15** layer-0 producer fields
  are now **bit-identical** between the resident TP1 bulk teacher and the TP2
  bulk candidate on identical input, weights and initial state; the candidate's
  dispatch resolve log shows exactly one bulk context, equal to the teacher's
  (`benchmarks/results/2026-09-17-w7900-tp2-bulk-vs-resident-tp1-layer0.json`).
- **Post-fix end-to-end result (2026-09-17):** the same-script, same-command A/B
  shows the saved failing prompt improving sharply — `mixed_ja_en_translate`
  bulk-vs-teacher max KL 0.487140 → **0.167053** (−65.7%), mean KL 0.00682607 →
  **0.00255761** (−62.5%), p99 KL 0.20779 → **0.0913498** (−56.0%), top-1
  0.992188 → **1.0** with no flipped rows, and bulk-vs-serial max KL 0.551179 →
  **0.0258852** (−95.3%). On the heldout `heldout_mixed_summary` the same change
  improves p95 KL 0.00117821 → **0.000687013**, p99 KL 0.0374066 → **0.0187383**
  and top-1 0.96875 → **0.984375**, but worsens max KL 0.0557287 → **0.260776**
  and mean KL 0.00100139 → **0.00240681**. The token-serial route is
  bit-identical before and after on both prompts, and two independent post-fix
  runs produced identical bulk and serial logits SHA-256s.
  (`benchmarks/results/2026-09-17-w7900-tp2-bulk-prefill-dispatch-context-ab.json`)
- **Blocker:** both prompts still exceed the production envelope (max KL ≤ 0.05,
  mean KL ≤ 0.001, p99 KL ≤ 0.02), so the bulk route stays opt-in and
  default-off and is not promoted. Bit-identical layer-0 intermediates do **not**
  by themselves prove that every end-to-end difference came from the dispatch
  context; the A/B above is the controlled intervention, and it shows the
  remaining gap is no longer a single monotone defect — the heldout's max/mean
  regressed while its p95/p99/top-1 improved. The next experiment is to localize
  the remaining gap **downstream of layer 0** (the sharded MLP chain route,
  whose Q6_K down projection changed to `t16_wmma_prefill` when the context was
  wired in, and later-layer accumulation) before any envelope or promotion
  claim. The bulk head projection is deliberately left outside the dispatch
  context (it is not a resident layer helper, and the Q6_K planar
  `t16_wmma_prefill_bf16_f32_out` leaf is unregistered); both are recorded in
  `docs/REFACTOR.md`. Do not relax the envelope.
- **Layer-0 MLP boundary measured (2026-09-17):** the sharded MLP chain is not a
  defect source. `scripts/tp2_bulk_vs_resident_layer0.py` now also captures the
  MLP half, and `scripts/tp2_layer0_mlp_reference_check.py` compares it against
  an independent numpy f32 reference built from the dequantized layer-0 weights.
  On the same 64-token prompt, with bit-identical `post_norm`/`residual` and a
  bit-identical concatenated activation (1,114,112 cells, both rank slices), the
  route's schedule-internal steps are exact: the staged f32 reduce equals the sum
  of the staged bf16 partials to rel **5.1e-11**, `cast` is exactly the bf16
  rounding of `reduced`, and `out` is exactly `bf16(residual + cast)` on both
  ranks. The whole remaining difference sits in the down projection:
  `cast` vs the teacher's `ffn_down` is max_abs **0.125** / rel **6.90e-03**
  (110,629 of 327,680 cells, 17,021 beyond one bf16 ULP). The reference
  decomposition attributes it: f32 partials vs the full-width f32 projection
  differ by rel **1.05e-07**, the bf16 partial boundary costs max_abs **0.0309**
  / rel **1.70e-03**, and each route's own output deviates from that same f32
  reference by 1-2 bf16 ULPs (teacher **0.125**, candidate **0.0625**), so the
  gap is the two down kernels' f32 accumulation order plus the bf16 partial
  boundary — not slicing, layout, kernel selection, or state ownership. Evidence:
  `benchmarks/results/2026-09-17-w7900-tp2-layer0-mlp-boundary.json`.
- **Remaining work (2026-09-17):** with layer 0 explained, the end-to-end gap is
  accumulated bf16-level association drift over 64 layers, and it is
  prompt-dependent: the heldout fails against **both** TP1 references post-fix
  (teacher max KL 0.261, token-serial 0.182), while the saved prompt is inside
  the envelope against token-serial (max 0.0259, mean 4.67e-04, p99 0.0164,
  top-1 1.0) and outside it against the resident teacher (0.167). The failure is
  carried by a few near-tie rows (heldout p95 6.9e-04 against max 0.261). The
  next decision is arithmetic, not defect-hunting: either remove the introduced
  rounding by staging f32 down partials where a registered f32 partial consumer
  exists, or accept the bf16 partial boundary and requalify the envelope on the
  full mtp-bench category suite. Do not relax the envelope.
- **f32 down partial measured and rejected (2026-09-17):** the arithmetic
  question above was answered by building the missing half of it. The
  rank-parallel down projection can now write its unrounded f32 accumulator for
  **both** down quant families — `q4_k_t16`
  `dense_single_local32_bf16_f32_out` (a wider store on the existing local32
  owner; 32 of the model's 64 `ffn_down` tensors are Q4_K, which is why the
  group's single staging dtype had to be bf16) and the pre-existing `q6_k_t16`
  `t16_gemv_decode_bf16_f32_out` — selected per session by
  `decode_partial_dtype="f32"`. The route is arithmetically exact: on layer 0
  each rank's f32 partial matches an independent f64 oracle to max_abs **5.0e-08**
  / **7.7e-08** and the two-rank f32 sum matches the resident TP1 f32 down output
  to **1.9e-09**; the transport control is exact too (bf16 partials with the
  device spin-sum and with the host staged sum give **byte-identical** logits in
  separate processes, and the f32 arm reproduces byte-identically across
  processes). End to end on the same 2 prompts x 128 forced-decode rows it makes
  agreement with the resident TP1 teacher **worse**, not better: max KL
  **0.108406 -> 0.732890** and **0.046893 -> 0.242473**, mean KL **0.0014286 ->
  0.0084696** and **0.0010041 -> 0.0028254**, top-1 **1.000 -> 0.9766** and
  **0.9844 -> 0.9766** with 3 flipped rows each. `decode_partial_dtype="bf16"`
  therefore stays the shipped schedule; the f32 option remains implemented,
  registered, documented and fail-closed (`f32` requires `reduce_mode="host"`,
  and no `hip_gfx1100` kernel is registered for the rows>1 f32 variant, so a
  multirow f32 request cannot acquire a bf16 GPU store) as a qualified
  alternative and bisection control.
- **Why the boundary fix did not move the end-to-end number (2026-09-17):**
  widening the measurement to a 2x2 of prefill route x partial dtype shows the
  metric is not ordered by distance from the teacher. Serial prefill with the
  bf16 partial is best on every reported metric and sits at the envelope edge
  rather than far outside it (heldout mean KL 0.0010041 against the 0.001 bar,
  p99 0.0288 against 0.02, max 0.0469 inside the 0.05 ceiling, top-1 0.9844;
  `mixed_ja_en_translate` mean 0.0014286, p99 0.0398, max 0.1084). Bulk prefill
  makes it worse (mean KL 0.0024068 / 0.0025576); serial with the f32 partial has
  the worst max KL on `mixed_ja_en_translate` (0.7329) while bulk with f32 is the
  better of the two f32 arms there. p95 stays within 5.7e-04..1.02e-03 across all
  four arms while mean/p99/max move by 2-6x. With the layer-0 boundary
  reproducing the teacher to 1.9e-09 and each route's own repeat reproducing
  bit-identically (the bulk arm also reproduces an earlier session's recorded
  numbers and logits sha256 exactly), the TP2-vs-teacher tail is bf16-ULP-level
  chaos amplified over 64 layers: every route perturbation moves it by more than
  the envelope width, so no single arithmetic term decides the comparison.
- **The sustained failure is implementation spread, not a TP2 defect
  (2026-09-17):** the 128-step failure on `mixed_ja_en_translate` at decode index
  82 was measured against the product's own alternative route. Four arms ran over
  the same teacher-forced prefix on one revision, one host, one identity
  (`scripts/tp2_prefill_schedule_failure_probe.py`; the recorded identity diff is
  empty, and the `tp1-bulk` self-check arm reproduced the teacher's rows exactly,
  max KL 0.0). The teacher is hipEngine TP1 with bulk prefill; **TP1 with
  token-serial prefill — no TP2 code involved — breaches the same 0.05 max-KL
  ceiling at the same decode index 82 (0.2993) and reaches 0.6092 by index 93**,
  while TP2 token-serial reaches 0.1084 and TP2 bulk prefill 0.1671. TP2 is
  closer to the teacher than that TP1 route on mean (1.429e-03 vs 7.241e-03),
  p99 (0.0398 vs 0.2211), max (0.1084 vs 0.6092) and top-1 (1.000 vs 1.000),
  and at the failing position TP2 sits 10.5x closer to a *same-schedule* TP1
  control (0.028379) than that control sits to the teacher (0.299279 — the same
  number the earlier TP1-only localization recorded, to six decimals).
  The mechanism is measured, not inferred: the breaching positions are the flat
  ones. At index 82 the teacher's top-1 holds 0.4866 of the mass with a 0.6233
  top-2 logit gap and entropy 1.2390, against neighbours at top-1 probability
  >= 0.99 and gaps of 5-21, and **every arm keeps the same top-1 token** (248046)
  there. 2-3 of 128 positions exceed 0.01 KL and p95 stays inside
  5.66e-04..9.88e-04 for every arm. A ~0.1-1.0 max-abs logit difference between
  two legitimate implementations therefore moves a lot of probability mass
  exactly where the model is undecided: over a 128-step forced horizon the
  absolute max-KL ceiling is a bit-exactness test in disguise, and the shipped
  TP1 product path fails it too. Evidence:
  `benchmarks/results/2026-09-17-tp2-prefill-schedule-failure-probe.json` (+ its
  `.worst-rows.npz`, the full logit rows at each comparison's worst position).
- **Consequence for the sustained gate (2026-09-17, decided):** the discriminating
  sustained measurements are mean/p95 KL, top-1 agreement, and a
  **same-schedule** implementation-spread comparison, not the raw max against a
  single bulk-prefill reference. The absolute max ceiling is not relaxed: instead
  the comparison **horizon** is declared, which is the mechanism
  `docs/EXECUTION-PROFILES.md` already uses for the C2/C3 packets. Section 6.5 of
  that document now declares **D=42** for this model/host/suite, with the
  three-part test it rests on (the reference reproduces itself exactly;
  materially different implementations breach the same prompts at the same
  depths; the breaching rows are flat and keep the reference's top-1 token).
  Both harnesses are horizon-aware: `--horizon D` gates the first D rows of each
  prompt and always records the full-horizon envelope as an unscored diagnostic,
  so a shorter horizon cannot make a tail disappear from an artifact. Measured at
  that horizon on one revision (18 prompts, 756 scored rows per arm, three
  bit-identical repeats each): both ranks' bulk-prefill TP1 arms are
  bit-identical to the reference (max KL 0.0, top-1 100%) and TP2 token-serial
  scores mean 9.442e-05 / p95 4.411e-04 / p99 1.029e-03 / max 4.955e-03 /
  top-1 99.74% (worst category 99.4%), so **the sustained numerical gate passes
  in the advertised scope D<=42** with 10x or better margin on max/p99; the same
  artifact records TP2's unscored 0.268571 max KL over the full 128 rows. Nothing
  is promoted by the declaration itself: `decode_partial_dtype`
  stays `bf16` and token-serial prefill stays the shipped prefill route. The bulk
  candidate's own arithmetic effect is bounded by
  `tp2-serial_vs_tp2-bulk` = 0.0258852 max KL with top-1 1.000 and no position
  over the ceiling on that prompt — inside the ceiling on the schedule-difference
  axis — but its distance to the single bulk teacher grows 1.4-2.3x on
  mean/p95/p99/max relative to the serial arm, and its own horizon is 42 against
  the shipped route's 45, so retaining it still needs the full mtp-bench
  category suite.
- **The sustained failure is suite-wide and has a measured horizon
  (2026-09-17):** the same comparison was run over the whole 18-prompt teacher
  suite (2,304 teacher-forced positions per arm, one revision, one host; arms one
  session at a time, the two TP2 schedules compared across passes through
  `--store-rows`/`--compare-rows`). **Every arm satisfies the full production
  numeric envelope on every prompt up to D=42** — TP1 token-serial (no TP2 code),
  TP2 token-serial and TP2 bulk — and TP2 token-serial reaches **D=45**. Beyond
  that they breach on the *same four prompts at the same decode depths*:
  43/46/43 on `heldout_mixed_review`, 51/51/51 on `mixed_ja_en_review`, 80/80 on
  `heldout_mixed_summary` (TP2 arms), 83/83/83 on `mixed_ja_en_translate`. The
  reference arm reproduced itself exactly (max KL 0.0 on all 2,304 positions,
  18/18 prompts), so the depths are a property of the reference trajectory's
  logit geometry, not of any candidate's arithmetic. Over the suite, TP1
  token-serial scores mean KL 6.912e-04 / worst prompt mean 7.241e-03 / worst max
  0.6092 / min top-1 0.9844 and TP2 token-serial **3.688e-04 / 2.334e-03 / 0.2686
  / 0.9844**, i.e. the shipped TP2 route is the best of the three, and counting
  every envelope statistic (including the 0.99 top-1 bar) TP1 token-serial fails
  5 of 18 prompts, TP2 token-serial 4 and TP2 bulk 5. The large-KL positions are
  the flat ones: the reference's top-1 probability has median 0.9951 across the
  suite while the 27 rows over 0.01 KL have median 0.7988 (max 0.9421) and a
  median top-2 gap of 1.66 against 5.73. So **D=128 is past every route's
  authorized horizon** and its verdict at that depth is not a TP2 discriminator.
  The decision is whether to declare the horizon (D<=42 any route, D<=45 shipped
  TP2 — the framework `docs/EXECUTION-PROFILES.md` already uses), to read the
  numeric envelope beyond it as diagnostic, or to leave the gate as it is and
  record that no route qualifies at D=128 on this host. Evidence:
  `benchmarks/results/2026-09-17-tp2-prefill-schedule-suite-spread.json`.
- **The comparison basis is the open question (2026-09-17):** the TP2 gate scores
  against **hipEngine TP1's own logits** (`quality-tp1-d0.json`, arm `tp1-d0`),
  while hipEngine's production quality basis is the independent llama.cpp BF16
  teacher protocol (`scripts/qwen38_llama_teacher.py`, `TEACHER_STEPS=9`). A
  route-parity metric between two legitimately different arithmetic routes cannot
  separate a TP2 defect from legitimate route difference — which is exactly what
  the 1.9e-09 layer-0 boundary result demonstrates. The next decision is the
  qualification basis: score TP2 against the independent teacher capture on the
  mtp-bench protocol and gate its own oracle-KL against the same envelope, or
  declare a TP2-specific envelope from its own repeat/isolation distribution, or
  close the route difference by moving the sharded leaves into the resident leaf
  family. Evidence:
  `benchmarks/results/2026-09-17-w7900-tp2-f32-down-partial-ab.json`,
  mechanism check `scripts/tp2_mlp_slice_e2e.py --down-output-dtype`,
  `docs/REFACTOR.md` (the knob and the dtype-blind rewrite gap).

Before any kernel port: run `scripts/check_lineage.py`, check `docs/KERNELS.md`,
and register a strict fallback. No new kernel unless a concrete missing
primitive is identified; no backend/quant dispatch branches; no Torch on the
hot path.

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
- [x] Write a break-even artifact and a go/no-go decision before Packet 3.
  **Decision: no-go on the RCCL eager path; the host-staged exchange clears the
  design's qualified-win bar.** `benchmarks/results/tp2_break_even_per_step.json`
  projects 0.61-0.68x against the faster matched TP1 arm with RCCL's
  per-reduction groups. Removing the intermediate device copy gives 0.66-0.74x
  (`.../tp2_break_even_per_step_alternating.json`), replaying the same structure
  from a captured graph gives 0.68-0.76x
  (`.../tp2_break_even_per_step_alternating_graph.json`), and replacing RCCL with
  a page-locked host exchange gives 0.87-1.02x
  (`.../tp2_break_even_staged_exchange_host_sync.json`). Batching that exchange's
  submission so both ranks are in flight before either is awaited gives
  **0.99-1.17x** (`.../tp2_break_even_staged_exchange_batched.json`) - above 1.0x
  at 0%, 10% and 20% fixed-cost share. Packet 3 does not start from the RCCL
  result; see "Measured status".
- [x] Profile where the TP1 step's device time goes, on both cards. A rocprofv3
  attribution puts weight reads at **25.95 ms/step (W7900) and 21.77 ms/step
  (XTX)** - 77% of each card's own token time - with attention + GDN + sampler +
  copy at **1.578 and 1.441 ms/step** beside them, and the two cards' 1.19x
  weight-read ratio matching their 1.20x TP1 token-rate ratio. This is a device-time
  profile, **not** a fixed-cost share: dispatch intervals overlap (the harness
  reports a 1.916 overlap ratio), the family sums are not additive, and the fixed
  term includes launch and scheduling cost that the table does not measure. The
  0-30% sensitivity range therefore stands. Each step issues ~811 launches, which a
  TP2 group would carry on both ranks.
- [ ] Measure the shard-shaped segments that selective sharding needs, on both
  cards: half-intermediate MLP through the engine's own GEMV dispatch, plus
  replicated and single-owner attention and GDN at full and half head counts, with
  local layout, output dtype, residual/norm boundaries and the relevant context, and
  representative `compute -> exchange -> compute` sequences rather than an idle
  buffer exchange. This needs a shard-shaped weight layout, which the engine does
  not have yet. Sharding only the MLP halves the exposed reduction count but
  **restores the full attention and GDN work on each rank** relative to full TP,
  and the slower rank can determine that segment's completion time. Single-owner
  attention is not free either: it needs the corresponding result broadcast. The
  decision rule is per segment, `TP1 segment time > slowest shard time + exposed
  communication`, and one global fixed-cost share cannot resolve it.
- [ ] Run a native C++ A/B of the exact batched protocol with preallocated
  pinned slots, views/descriptors and sum scratch, preserving ordering, device
  ownership and numerical semantics, and compare against the Python arm on total
  latency. The Python phase counters do not bound the native benefit: they start
  inside the device-scope helper, exclude context entry/exit and other Python
  work, and cover a different population from the ladder slope. Earlier
  submission can also change the exposed wait, so that term is not constant
  across implementations. The benefit is unmeasured, not bounded by a
  subtraction.
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
- [x] Resolve the N-rank plan and composite KV pool/claim set. Preserve the
  model-owning scheduler; add a distributed runner adapter, not N schedulers.
  `hipengine/distributed/kv.py` resolves geometry from the model config and a
  declared `KvSpansLayout`, builds a per-rank `KVPoolPlan`, and reserves the
  group's claims through the existing `ResourceLedger` (provisional hold on
  every rank, commit only when all reserved, rollback otherwise). The
  distributed runner adapter is still Packet 3 work.
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

- [x] Implement MLP-only TP2 with replicated attention as a diagnostic first
  slice. Confirm one down-projection reduction per layer and exact single
  residual addition; use boundary probes to localize drift. (Commits
  75f75e335, e4403443c; full-model e2e with all production gates passing.)
- [ ] Add full-attention head sharding and then linear-attention group/state
  sharding. Verify cold and warm trajectories separately; sharding only QKV
  without its convolution/GDN state is not a complete implementation.
  (NOT DONE: the current runner replicates attention and GDN per rank with
  per-rank GDN/conv scratch zeroing; stale-state discipline is pinned by
  teacher-forced-after-generation tests, but that is replicated attention,
  not attention sharding. The vocabulary head IS row-sharded, commit
  7c2a7fb66.)
- [x] Connect prefill, one-token decode, positions, KV allocation, reset, EOS,
  and resource teardown. Prefill must produce the same rank-local state layout
  consumed by decode, including chunk boundaries and long contexts. (The
  e2e harness drives prefill, decode, EOS, and teardown end to end.)
- [x] Run full-logit numerical gates and same-schedule repeats. Establish true
  TP2 AR with all draft allocation/execution disabled as the MTP denominator.
  (Mean KL 3.945e-04, max KL 2.365e-03, top-1 100% vs both per-GPU TP1
  controls; same-schedule repeats bit-exact.)
- [x] Measure head cost. Add vocabulary-row sharding only if worthwhile. For
  greedy output preserve deterministic global tie-breaking; for stochastic
  sampling initially use a correct full-logit gather on one owner and one RNG
  stream, or explicitly declare the mode unsupported. Distributed top-k/top-p
  and speculative probability normalization require their own correctness gate.
  (Head = 1.04 GB/token, the largest single read; row-sharded 124,160/rank,
  -3.2% decode p50, bit-identical greedy tokens, commit 7c2a7fb66. The
  session is greedy-only; stochastic modes are explicitly unsupported.)

Exit: a correct end-to-end TP2 AR runner, documented profile scope, two-GPU
memory/timing evidence, and a decision on whether to proceed with optimization.

### Packet 4 — Reduce exposed PCIe and host overhead

- [x] Trace both ranks with cached builds. Attribute local GEMV/GEMM, GDN,
  collectives, rank skew, idle gaps, H2D/D2H copies, and host launch cost.
  Use common host wall for end-to-end latency; do not subtract unsynchronized
  timestamps from different GPUs as if they shared a clock. (rocprof
  kernel-trace wedges on full-stack sessions - blocked loop iteration with
  symptoms; HIP-event attribution instead: wall 24.37 ms/step with layers
  22.73, tail 1.25, metadata 0.31, rank skew 0.06 ms/step - spans include
  device-side waits and gaps (especially around the polled exchange), so
  the near-100%-device figure is an upper bound on true execution, not a
  proven floor; attention dominates the ~350 us graphed layer.)
- [x] Capture stable per-device graph segments and collectives only where
  supported. Keep event dependencies explicit and verify repeated replay,
  address lifetimes, and graph invalidation. Do not assume one cross-device
  HIP graph works, or discard graphs without measuring the lost TP1 benefit.
  (Per-(layer, rank) captured segments promoted as the default, commit
  914112cf6: -46% decode p50 vs eager, capture at the capacity bound,
  bit-identical to eager at exact bounds.)
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
- [x] Tune actual rank-local GEMV and verifier shapes through the four-axis
  registry. Smaller shards may hit different performance regimes than TP1.
  (Fused gate/up+SiLU shard route promoted, commit 3d7254090; down-shard
  GEMV variant sweep no-go: the incumbent is the family's best at
  (rows=1, in=4352, out=5120), all variants bit-identical, commit
  79288288b. Verifier shapes are Packet 5 MTP work.)
- [x] Screen aligned unequal shards only if rank skew warrants it; retest the
  complete manifest and numerical contract for each split. Drop rejected paths
  or record precise removal conditions in `REFACTOR.md`. (Measured rank skew
  is 0.06 ms/step - unequal shards are not warranted.)

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

### Default-path decode cell and the traffic accounting (2026-09-17)

Three matched repeats of `scripts/tp2_mlp_generate_e2e.py` (4 prompts x 16
greedy decode transitions, all three arms in one session on one revision) put
the default TP2 route at decode p50 **24.155 ms/token** against the same-run
resident TP1 controls at **31.753** (W7900) and **26.329** (RX 7900 XTX)
ms/token: **1.315x** and **1.090x**. The three repeats span 24.155-24.167
ms/token, and every arm keeps the recorded teacher gates (mean KL 3.945e-04,
max KL 2.365e-03, top-1 100%). The harness now records the route each arm
actually resolved rather than the raw CLI argument, because the previous
artifacts stored `None` for the TP1 arms' forced eager/host control schedule and
for the head shard that the TP2 default enables. This is a matched diagnostic
cell, not a product speedup: the horizon is 16 transitions and the sustained
numerical gate is still the open blocker.

`scripts/tp2_traffic_accounting.py` derives the per-token weight traffic from
the same GGUF index and the degree-2 shard plan:

| quantity | value |
| --- | ---: |
| resident TP1 control, per token | 15.652 GiB |
| TP2 current route, per rank per token | 9.676 GiB |
| of which MLP (sharded) | 9.650 GiB -> 4.825 GiB/rank |
| of which attention (replicated) | 3.301 GiB |
| of which GDN/SSM (replicated) | 1.062 GiB |
| of which head (sharded) | 0.971 GiB -> 0.486 GiB/rank |
| cross-rank reductions per token | 64 |
| resident achieved bandwidth, slow rank (W7900) | 529.3 GB/s |
| TP2 achieved, per rank | 430.1 GB/s |
| implied wall at the slow rank's bandwidth | 19.629 ms |
| measured wall | 24.155 ms |

Two readings follow, and they re-order the remaining speed work:

- **4.53 ms/token is not explained by weight traffic.** The TP2 route moves
  61.8% of the resident control's bytes but takes 76.1% of the W7900 control's
  wall, so its per-byte efficiency is 81% of the resident route's. That gap -
  the in-graph exchange, rank imbalance, shard-shape kernel efficiency and any
  exposed GDN latency - is larger than what finishing the sharding would buy,
  and it is the first thing to attribute.
- **Finishing Packet 3's attention/GDN sharding is a small speed lever, not a
  large one.** It cuts per-rank traffic 1.38x (9.676 -> 8.646 GiB for the slow
  rank, 17.54 ms implied at the same bandwidth) but adds 64 cross-rank
  reductions per token to the 64 already paid, so the net gain is bounded by
  roughly 2.1 ms minus the extra reduction cost. Its justification is capacity
  and correctness, not throughput.

### Where the unexplained milliseconds are (2026-09-17)

`scripts/tp2_stage_device_attribution.py` now records per-`(layer, rank)`
HIP-event spans (and takes the reduction owner as an argument), so the aggregate
`layers` span splits by layer index and layer type. On the default device-side
reduction, 16 hand-driven steps:

| quantity | rank 0 (W7900) | rank 1 (RX 7900 XTX) |
| --- | ---: | ---: |
| per-layer median | 0.350 ms | 0.349 ms |
| per-layer min / max | 0.327 / 0.386 | 0.307 / 0.386 |
| full-attention blocks (n=16) | 0.345 ms | 0.357 ms |
| GDN/linear-attention blocks (n=48) | 0.350 ms | 0.348 ms |
| per-layer rank skew | median 0.008, max 0.056 ms | |
| achieved bandwidth, 162.3 MB/rank/layer | 464 GB/s | 466 GB/s |
| same rank's measured resident bandwidth | 529.3 GB/s | 637.7 GB/s |

Two conclusions, and both close off a hypothesis:

- **There is no layer-type structure.** Full-attention and GDN blocks cost the
  same per layer to within noise (and the ordering flips between ranks), and the
  per-layer spread is 18% with no layer-index pattern. So the deficit is not
  "GDN is slow", not "full attention is slow", and not a specific layer.
- **Both ranks run ~12% below their own resident per-byte efficiency, uniformly.**
  The per-layer weight-traffic floor is 306.7 us at the slow rank's measured
  529.3 GB/s; the measured median is 350 us, i.e. 464 GB/s achieved. The
  host-reduction control isolates rank 1 (which does not wait on its peer in
  host mode) at 287 us / 566 GB/s, 89% of its own 637.7 GB/s. That is **3.09
  ms/token** of the 24.155 ms wall, and it is a registry/kernel-efficiency
  target at the shard shapes, not a schedule target.

The same runs also decompose the transport: the host-side reduction exposes
**3.178 ms/token** between layers (the device reduction collapses that to
**0.307 ms**) at a cost of **0.492 ms/token** inside the captured graphs, which
is the measured basis for the device-side default.

Artifacts: `benchmarks/results/2026-09-17-w7900-tp2-stage-attribution-device-reduce.json`
and `...-host-reduce.json`.

#### The per-layer time is one streaming half plus one latency-bound half (2026-09-17)

Splitting a layer into its two captured halves - the attention/GDN half and the
rest (post-attention norm, D2D input copy, MLP shard chain, residual) - and
replaying each from a **DRAM rotation** over eight probe layers (four of each
type) gives the split the whole-layer number cannot:

| layer type | attention half | rest half | sum | real-loop layer |
| --- | ---: | ---: | ---: | ---: |
| full attention (n=4) | 229 us / 61.4 MB = **260 GB/s** | 163 us / 86.7 MB = 504 GB/s | 392 us | 345 us |
| GDN (n=4) | 183 us / 84.5 MB = **436 GB/s** | 167 us / 86.7 MB = 493 GB/s | 350 us | 350 us |

(Rank 0, W7900. Rank 1, RX 7900 XTX: full attention 207 us at 286 GB/s and
143 us at 559 GB/s; GDN 156 us at 506 GB/s and 143 us at 545 GB/s. The rotation
matters: one layer's weights fit the 96 MB Infinity Cache, so a single-layer
replay measures L2.)

A second audit measures every projection alone, with the model's real weights
through the production launch entry points, per launch, same rotation protocol:

| projection | shape (out x in) | W7900 | RX 7900 XTX |
| --- | --- | ---: | ---: |
| `attn_q` | 12288 x 5120 | 35.4 MB, 59.9 us, **591 GB/s** | 66.6 us, 532 GB/s |
| `attn_k` + `attn_v` pair | 1024 x 5120 each (Q4_K + planar-Q6) | 7.2 MB, 32.8 us, **221 GB/s** | 30.6 us, 237 GB/s |
| `attn_output` | 5120 x 6144 | 17.7 MB, 34.3 us, 516 GB/s | 33.2 us, 534 GB/s |
| `attn_qkv` + `attn_gate` pair | 10240/6144 x 5120 | 60.7 MB, 92.9 us, 654 GB/s | 78.4 us, 774 GB/s |
| `ssm_out` | 5120 x 6144 | 21.6 MB, 40.0 us, 540 GB/s | 33.1 us, 653 GB/s |
| `ffn_gate` / `ffn_up` (TP1 width) | 17408 x 5120 | 50.1 MB, 86.8 us, 577 GB/s | 68.7 us, 730 GB/s |
| `ffn_down` (TP1 width) | 5120 x 17408 | 73.1 MB, 105.4 us, 694 GB/s | 90.0 us, 812 GB/s |

Three conclusions, and they redirect the campaign:

- **The GEMV kernels are not the deficit.** Every projection runs at 516-694
  GB/s on the W7900 and 532-813 GB/s on the XTX, i.e. at or above the resident
  route's own blended 529.3 / 637.7 GB/s. The shard shapes are not slower than
  the full-width ones, and no projection is stuck in a bad variant.
- **The attention half is latency-bound, not bandwidth-bound.** Subtracting the
  measured projection times from the attention half leaves **89 us per
  full-attention layer** and **50 us per GDN layer** of small-kernel time - the
  input norm, head norm and RoPE, KV write, attention core and gate multiply,
  alpha/beta, conv, and the GDN recurrence. Across 16 + 48 layers that is
  **3.82 ms/token**, 15.8% of the 24.155 ms wall, and both ranks pay all of it
  because the attention/GDN weights are replicated.
- **The worst single projection was the smallest one, and it was a policy
  miss, not a kernel miss.** The `attn_k`/`attn_v` pair moved 7.2 MB in 45.8 us
  (158 GB/s) because a pair launch at `out=1024` is fixed-cost-bound; 16 of
  those were 0.73 ms/token. The exact block-parallel narrow K/V pair kernel was
  already registered on gfx1100 - only the two policy declarations were missing
  (the c1 table entry routing the shape's Q4_K K singleton to the col4 sibling
  the pair composes from, and the shape capability itself), so both ranks now run
  one fused launch of `narrow_col4_planar_pair_bf16_bf16_out` (V is planar-Q6 in
  this quantization): **32.8 us on the W7900 and 30.6 us on the RX 7900 XTX**,
  saving 13.0 and 16.1 us in each of the 16 full-attention layers (**0.21 / 0.26
  ms/token** isolated). The graphed schedule already hid part of the launch
  cost, so the measured wall moves by less: **TP2 p50 24.155 -> 24.025 ms/token
  (-0.54%)**, TP1 W7900 31.753 -> 31.713, TP1 XTX 26.329 -> 26.256, with every
  repeat below the previous cell's range and bit-identical logits.
  The remaining lever there is a Q4_K `q`+`k`+`v` triple, which is now worth
  less than it looked: `attn_q` already runs at 594 GB/s and the fused pair's
  marginal rate is close to it, so folding K/V into the Q launch mostly moves
  the same fixed cost rather than removing it.

#### The per-layer launch inventory (2026-09-18)

`scripts/tp2_decode_kernel_inventory.py` runs the production decode step inside
one ROCTX region bounded by device synchronizations and rolls up a `rocprofv3`
kernel trace per rank. The per-rank split is verified rather than assumed: one
probe launch per rank with a distinct element count maps the trace's `Agent_Id`
to a rank, and an unverifiable mapping is reported as unavailable. Eight decode
steps on one revision, both ranks:

| quantity | rank 0 (W7900) | rank 1 (RX 7900 XTX) |
| --- | ---: | ---: |
| kernel launches per step | 901 | 901 |
| device copies per step | 132 | 132 |
| launches per layer | 14.06 | 14.06 |
| kernel time per step | 19.949 ms | 20.502 ms |
| small kernels (<10 us mean) | 1.914 ms/step | 1.483 ms/step |

The profiled region wall is 26.367 ms/step against the unprofiled 24.025
ms/token default-path cell, so kernel time is quoted from the trace while the
wall stays the e2e measurement.

Two results change the plan, and both are negative for hypotheses the earlier
sections raised:

- **The norm/residual chain is already fused.** The per-layer trace shows one
  `gguf_norm_fixed5120_wave256_kernel<true, false>` (the `kAddResidual`
  instantiation, i.e. `add_rmsnorm`), one `<false, false>` plain input norm and
  one `gguf_bf16_add_kernel` - 3 launches where a naive reading of the
  architecture would predict 4 (two norms plus two residual adds). The capture
  path already calls the runner's resolved `add_rmsnorm` leaf
  (`_add_norm_kernel`, `tp2_generate.py`), and the remaining add is the
  exchange epilogue. There is no unfused norm pair to fuse.
- **The exchange is not on the critical path, and its cost is rank skew being
  absorbed.** `tp2_dev_spin_add_bf16` on the pacer rank (W7900) is min 3.72 /
  p50 4.08 / p90 4.36 us per layer, while on the XTX it is min 4.24 / p50 66.18
  us: the faster rank spends ~62 us per layer waiting for the slower rank's
  partial. The add itself is ~4 us of 20 KB, so the mean is not work. Removing
  the exchange would not remove that time; the pacer is rank 0's kernel time plus
  intra-graph gaps.

What the inventory does locate, per rank per layer, is 14.06 launches:
~4.75 GEMV projections, 2.75 attention/GDN core kernels, 3 norm/add, 1.5
f32-weight dense GEMV, 2 exchange (`tp2_dev_publish_flag` + `tp2_dev_spin_add_bf16`)
and 2.06 device copies. The two remaining launch-count levers are therefore the
exchange pair and the copies, not the norm chain; the projections that carry the
bytes are already at their measured bandwidth.

Artifact: `benchmarks/results/2026-09-18-w7900-tp2-decode-kernel-inventory.json`.

#### The split is even; the two cards are not (2026-09-18)

The per-layer inventory also settles a question the earlier projection audit
raised but did not act on. Measured per kernel, both ranks, identical call
counts, the RX 7900 XTX (rank 1) is faster on **every** streaming kernel:

| kernel | calls/step | W7900 us | XTX us | XTX/W7900 |
| --- | ---: | ---: | ---: | ---: |
| `q4_k_t16_dense_single_local32` | 136 | 37.45 | 30.30 | 0.81 |
| `q4_k_t16_dense_dual_local32_silu` | 64 | 79.09 | 64.58 | 0.82 |
| `q6_k_t16_qmicro_planar` | 56 | 64.43 | 54.50 | 0.85 |
| `q5_k_t16_dense_single_local32` | 48 | 42.69 | 33.66 | 0.79 |
| `dense_gemv_bf16_f32w` | 96 | 4.84 | 4.29 | 0.89 |
| GDN recurrence, norms, conv | 224 | - | - | 0.88-0.93 |

This is not an in-situ artifact: the projection audit measured the same ratio
in isolation (577 vs 730 GB/s on `ffn_gate`/`ffn_up`), and the resident TP1
controls are 31.753 ms (W7900) against 26.329 ms (XTX).

The cost of splitting evenly shows up in one kernel. `tp2_dev_spin_add_bf16`
takes 4.07 us per layer on the pacer and 65.35 us on rank 1, so rank 1
finishes its faster work and then spins for 4.182 ms/step waiting for rank 0.
Removing the wait from both sides gives the work times: **19.689 ms** (rank 0)
and **16.320 ms** (rank 1). The even split makes the slower card the pacer
while the faster card idles.

`scripts/tp2_split_balance_plan.py` turns that into a split. It reads the
shard manifest and the inventory, separates each rank's streaming time from
its fixed time and its spin time, and solves for the byte fraction that
equalizes the two ranks:

| quantity | rank 0 (W7900) | rank 1 (XTX) |
| --- | ---: | ---: |
| streaming rate | 432.0 GiB/s | 526.6 GiB/s |
| fixed work | 2.347 ms/step | 2.092 ms/step |
| spin wait (excluded from cost) | 0.260 ms/step | 4.182 ms/step |
| streamed bytes | 7.492 GiB/step | 7.492 GiB/step |
| movable (MLP) bytes | 4.825 GiB/step | 4.825 GiB/step |

The model reproduces the measured pacer work time exactly - it predicts
19.689 ms/step for the even split against 19.949 - 0.260 = 19.689 measured -
and then predicts **17.937 ms/step** for a split of **0.417 / 0.583** of the
MLP bytes: a **1.752 ms/step** saving, about 7.3% of the 24.025 ms/token wall.

Two constraints define the scope, and both are deliberate:

- **Only the MLP projections are eligible.** `ffn_gate`/`ffn_up` split output
  rows and `ffn_down` splits the input-feature axis, so an uneven split there
  is a geometry change with no head semantics. Every attention and GDN tensor
  is head-structured, and `partition_groups` refuses uneven splits for them on
  purpose: rank `r` owns query heads `[r * q_per_rank, (r + 1) * q_per_rank)`
  and must load exactly the KV heads those queries attend to. An uneven split
  there would be silently wrong attention, not an error. The eligible MLP pool
  is 9.65 GiB of the 14.98 GiB streamed per step, which is enough to reach
  balance without touching them.
- **`ffn_down` changes arithmetic.** Its split point moves the summation
  grouping of the two ranks' partials, so this is a production-profile change
  requiring the full KL / top-1 gate, not a bit-exactness argument. The
  `ffn_gate`/`ffn_up` part is bit-exact (independent output rows).

The cost is memory on the faster card: rank 1 gains 0.851 GiB of resident
weights (7.009 to 7.860 GiB), which it has room for in 24 GB, at some reduction
in the maximum context that the capacity ladder would have to re-measure. Rank
0 drops the same amount (8.646 to 7.795 GiB), so the pair ends balanced in
resident bytes as well as in time.

Status: plan and calculator only. No split has been changed yet.

Artifact: `benchmarks/results/2026-09-18-w7900-tp2-split-balance-plan.json`.

The consequence for the plan: the traffic-implied 19.629 ms floor assumes every
byte moves at the resident route's *blended* rate, and the resident route only
reaches that rate because 61.7% of its bytes are MLP. The TP2 route's byte mix
is 49.9% MLP and 45.1% replicated attention/GDN, so its blended rate is lower by
construction. Further weight sharding cuts bytes but leaves the 3.82 ms/token
latency term untouched; reducing the per-layer launch count in the attention/GDN
half is the larger remaining lever, and it helps the resident route too.

Artifacts: `benchmarks/results/2026-09-17-w7900-tp2-layer-halves-dram-rotation.json`,
`benchmarks/results/2026-09-17-w7900-decode-projection-bandwidth-audit.json`.

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
