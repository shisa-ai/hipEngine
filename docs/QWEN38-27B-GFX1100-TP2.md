# Qwen3.8-27B: TP2 on W7900 + RX 7900 XTX

Status: implementation plan; no two-GPU measurements or runtime support claimed.

## Objective and decision rule

Implement torch-free two-way tensor parallelism (TP2) for Qwen3.8-27B GGUF
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

Notation: TP1 means one GPU; C is the number of active requests; K is draft
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
| `hipengine/distributed/__init__.py` | Empty at campaign creation; no working TP engine to extend. |
| `hipengine/server/api.py` | Capability metadata advertises world size 1 and no tensor-parallel support. Do not advertise TP2 before integration gates pass. |
| `hipengine/core/{hip,device,memory,tensor}.py` | Existing HIP/device primitives to audit for explicit device ownership. |
| `hipengine/loading/qwen35_gguf{,_materialize}.py` | Metadata, mixed GGUF tensor formats, full/linear attention mapping, and materialization. |
| `hipengine/runtime/qwen35_gguf_runner.py` | Resident target, recurrent state, prefill, and packed verification. |
| `hipengine/runtime/qwen35_gguf_{mtp,nextn}.py`, `hipengine/speculative/gguf_mtp.py` | Draft/target integration; also inspect `transaction.py`, `native_cycle.py`, and `verify_graph.py` in `hipengine/speculative/`. |
| [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md), [`TESTING.md`](TESTING.md) | Normative arithmetic, determinism, ownership, and test gates. |
| [`BENCHMARK.md`](BENCHMARK.md), [`benchmarks/README.md`](../benchmarks/README.md) | Evidence and matched-baseline rules. Historical rates are not TP2 denominators. |

Qwen3.8 uses the Qwen3.5-family implementation here. It is not an ordinary
all-softmax transformer: linear-attention layers carry convolution and Gated
DeltaNet (GDN) recurrent state. Generate the exact layer/head/state inventory
from the supplied GGUF; do not hardcode dimensions from the model name.

The [2026-09-06 MTP matrix handoff](../worklog/entries/20260906T050447.321364Z-lhl-gfx1100-mtp-ck-matrix-blocked-8dc5cc.md)
records two prerequisites: the physical width-one path can fall back to no MTP,
and unqualified width/depth cells cannot engage without existing serving
evidence. It also records qualified multi-request cells losing to their matched
AR arms. These are not single-request TP2 measurements, but they rule out using
old acceptance or speedup numbers as proof of a new MTP gain. Recheck source at
campaign start because another unit may resolve these blockers.

## Architecture to implement

### Runtime and communication

Use one process, one explicit device context and compute stream per rank, and
persistent per-device workspaces. A rank is one participating GPU. Enqueue both
ranks' work before waiting. Establish reductions **inside each layer**, before
the next consumer; running two complete forwards and reducing afterward is
incorrect. Prototype with grouped RCCL calls through `ctypes` (`librccl.so`),
without importing torch. Validate group launch semantics so the first rank's
collective cannot block the second rank from being enqueued.

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

The model inventory must specify every GGUF tensor's logical axes, byte/block
alignment, per-rank ranges, replications, and local dimensions. `Q4_K_M` is a
mixed-tensor preset, not a promise that every matrix uses Q4. Cutting a packed
row on its reduction axis may require block-aligned repacking and local stride
changes. Preserve original quantized blocks/scales without dequantizing and
requantizing to manufacture shards. If an axis cannot be partitioned safely,
replicate that operation first and record its serial cost.

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
- [ ] Add a guarded `scripts/tp2_collective_bench.py` and tests. Measure warm
  latency p50/p95/p99 and bandwidth for broadcast and sum, for FP32 and any
  proposed transport dtype: payload `rows * hidden_size * dtype_bytes`, rows
  1, 2, 3, 4, 5, plus actual prefill chunks. Test rank orders and many sequential
  reductions with a local producer/consumer, not just isolated copies.
- [ ] Measure matched TP1 AR and engaged TP1 MTP on each GPU, one at a time on
  this host, using the full category suite. Profile per-layer and head time.
  Do not substitute a result from another host, even with the same GPU model.
- [ ] Write a break-even artifact and a go/no-go decision before Packet 3.

For each segment between synchronization boundaries estimate:

`TP2 segment time ≈ max(rank0 local time, rank1 local time) + exposed reduction + launch/synchronization cost`.

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
- [ ] Compare RCCL to peer-copy/local-sum for actual payloads and the full layer
  chain. If justified, fuse local sum with residual/norm without double-adding
  residual; follow kernel catalog, lineage, strict fallback, numerical, and
  `rocprofv3 --kernel-trace` gates for each new kernel.
- [ ] Tune local half-width GEMV and verifier shapes through the four-axis
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
- [ ] Resolve width-one engagement and evidence-admission blockers without
  weakening production admission. If still needed, add a test-only screening
  mode that labels unqualified cells and cannot enable public automatic MTP.
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

Economics check: `MTP time per committed token = cycle wall / committed tokens`.
Include all draft, verification, communication, and recovery work. MTP wins only
when this is below matched TP2 AR time per token. Acceptance alone is not a win;
TP2 may accelerate AR more than the serial draft, reducing the optimal K.
Keep K0 as the automatic fallback outside qualified winning scopes.

### Packet 6 — Qualify the public path and publish the result

- [ ] Wire explicit TP configuration through model construction, `LLM.generate`,
  and serving. Reject unsupported hardware/model/profile/sampling combinations
  before loading; expose the resolved topology and MTP engagement in diagnostics.
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
