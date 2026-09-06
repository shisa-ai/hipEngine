# Qwen3.8-27B: capacity on a 24 GB RX 7900 XTX

Status: capacity measurement and optimization plan; no new fit limit claimed.

## 1. Objective and deliverables

Determine exactly which Qwen3.8-27B configurations fit and remain useful on one
24 GB RX 7900 XTX (`gfx1100`), for both one long request and multiple concurrent
requests. Compare `Q4_K_M` and `Q4_K_S` weights, BF16 and genuinely compact INT8
key/value (KV) caches, and AR-only versus MTP-capable/active execution. AR means
autoregressive decode without speculation; MTP means multi-token prediction.
Here **BF16 refers to KV storage, not full BF16 model weights**.

The result is a measured capacity map, not a single “maximum context” number:

1. **C1 map:** maximum total sequence budget and recommended prompt/output
   budgets for each weight quant, KV format and MTP mode.
2. **C=N map:** maximum simultaneous resident and physically batched requests
   at fixed per-request budgets, plus maximum per-request context at fixed
   resident capacity. Distinguish native batched execution from serial fallback.
3. **Memory ledger and improvements:** unique weight bytes, optional MTP bytes,
   state, KV/scales/mirrors, graph storage and transient peaks; remove avoidable
   allocation without concealing latency or quality costs.
4. **User settings:** exact supported launch/request settings and measured
   recommendations for an idle inference card and a card with a declared
   display/workstation memory reserve. Include throughput, latency and quality
   beside capacity, and identify unsupported combinations explicitly.

Use the actual XTX for fit, high-water and failure claims. A W7900 is useful for
code development or inspecting an oversized allocation, but subtracting its
tracked allocation from 24 GB is not an XTX fit test. The
[TP2 campaign](QWEN38-27B-GFX1100-TP2.md) is separate; no second GPU may hide
this card's limit.

Native C1/C>1 MTP and K4-K7 functional work belongs to
[`QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md`](QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md).
Compact INT8 continuous-batching implementation belongs to
[`QWEN38-INT8-KV-CONTINUOUS.md`](QWEN38-INT8-KV-CONTINUOUS.md).
Reuse and qualify those paths on the XTX; do not build another scheduler or
claim a blocked path fits because an estimate says it should.

## 2. Starting evidence: what it does and does not establish

Read [`KVCACHE.md`](KVCACHE.md), the
[INT8 continuous campaign](QWEN38-INT8-KV-CONTINUOUS.md), and
[`benchmarks/README.md`](../benchmarks/README.md) before measuring. These are
historical starting points; Packet 0 recovers exact source/commands and freezes
current behavior on the XTX.

| Starting observation | Interpretation / required check |
| --- | --- |
| The supplied report quotes 15.9 GiB weights, c5 at 23.3 GiB and c6 at 24.2 GiB for 512-token prompts. | Recover the allocation metric, source revision, output/context reservation, MTP loading and resident capacity. A fit estimate from another card is not a measured XTX boundary. GGUF file bytes are not necessarily resident device-weight bytes. |
| The [2026-09-06 direct c1-c8 sweep](../benchmarks/results/2026-09-06-gfx1100-qwen38-q4km-direct-c1c8-sweep.json) reports W7900 tracked peaks under 512 prompt / 128 output tokens. | This is a different protocol from server measurements. Its c6 peak is about 23.7 GiB, unlike the quoted 24.2 GiB. Do not average these values or declare a memory regression without matching the owner and workload. Neither number alone proves c6 works on an XTX. |
| The supplied old C1 128-prompt/24-output reference is 18.618 GiB. | Treat it as an unverified regression anchor until its artifact, source, route and metric are recovered. One matched current C1 run is the first cheap check, not a conclusion from the newer 512/128 curve. |
| Historical XTX BF16 C1: 32K operational, about 21.9 GiB whole-card peak; 52K near the physical limit. | See `KVCACHE.md`. These are AR server results with specific pool/reserve settings, not proof of current Generation-2 C1 or MTP capacity. |
| Historical compact INT8 C1: four natural 112K requests passed at about 23.3 GiB; 126K was a near-zero-headroom physical ceiling. | The [dedicated qualification](../worklog/entries/20260815T182245.795089Z-lhl-qwen38-dedicated-context-qualification-59a182.md) is idle-card, max-active=1, AR-only, prefix-cache-off/MTP-off. The [later context audit](../worklog/entries/20260816T071413.192306Z-lhl-qwen38-int8-context-quality-concurrency-dfa478.md) separates physical ceiling from repeatable service. Very short generated suffixes do not certify long-output use at the same prompt length. |
| Artifact-scoped no-mirror INT8 logical c2/c4 exists with serial physical-C1 execution. | `hipengine/models/qwen35.py` names `explicit_no_mirror_c1_direct_c4_serial`; the [INT8 campaign](QWEN38-INT8-KV-CONTINUOUS.md) distinguishes this from native C>1. Direct batched-attention tests/harnesses also exist; inspect integration and evidence rather than assuming old status text proves absence of a kernel. |

The old “width or depth, not both” summary describes a resource tradeoff, not a
fixed law for this engine. KV grows with aggregate live/reserved tokens, while
state and workspaces may grow with resident capacity or execution rows. Reducing
avoidable per-slot or per-graph storage can move both boundaries.

INT8 evidence is bound to exact model file, backend, quant, scale/layout and
route. A passing `Q4_K_M` artifact does not qualify a different `Q4_K_S` file.
The same filename on gfx1151 has previously identified different bytes and
rejected INT8 quality. Do not transfer that lane's pass/fail or memory policy.

## 3. Define “fits” before searching for a limit

Notation:

- C: requests actually active in a physical step; N: configured resident slots.
  Also record offered requests and actual groups: eight submitted requests
  processed as two c4 waves are not native c8.
- L: prompt tokens; D: requested maximum output tokens; S: total sequence
  budget including any required root, bonus, lookahead and guard slots.
- K: draft depth, K0 for AR. Logical verifier rows are `R=sum(1+k_i)`;
  record padded rows P too. A graph/workspace sized for K7 may cost memory
  even when a particular request only asks for K1.
- H: explicitly configured whole-card safety headroom in bytes. Use actual
  device-reported total/free memory, not an assumed 24.000 GiB capacity.

All artifacts report bytes and GiB (`bytes / 2^30`); distinguish marketed GB.
In context sizes, 1K means 1,024 tokens; a label such as D24 means 24 output
tokens. OOM means out of memory. Classify each attempted cell:

| Classification | Required evidence |
| --- | --- |
| Estimate only | Byte model; no claim that loading or execution passed. |
| Unsupported/policy-rejected | Stable reason before mutation, with requested and effective route/format/depth. Not an out-of-memory result. |
| Load-only fit | Startup and weight materialization pass; prefill/decode unproven. |
| Execution fit | Full requested prompt/output work completes, measured high water below actual free budget, correct output/state and teardown. |
| Operational fit | Repeated natural workloads plus cancellation/refill/pressure tests complete under the declared H and latency contract. |
| Capacity rejection / OOM / stall | Name the actual failure stage, last completed operation and sizes. A timeout is not an OOM or a numerical failure. |

Publish **largest observed pass**, **first tested failure**, and **operational
recommendation** separately. An untested gap, model context limit, policy cap,
quality rejection or latency limit is not a physical VRAM ceiling.

Predeclare two separate service contracts:

- Dedicated idle-card inference: measure idle driver usage and choose H before
  the search. A proposed initial reserve is 512 MiB, labeled as a service policy,
  not a hardware constant or a reason to discard a lower-memory improvement.
- Display/workstation coexistence: record observed display usage and an explicit
  additional reserve (for example 2 GiB). Test with that reserve; do not promise
  coexistence with arbitrary future desktop applications. Do not double-count
  measured external usage and the remaining reserve.

A diagnostic physical-ceiling row may go below those margins with bounded
watchdogs; it is not the recommended serving configuration. Never reserve less
than the requested output budget or hide paging into host memory to claim fit.

## 4. Memory accounting that explains the boundary

For each cell collect aligned, **unique physical allocations** and their
lifetimes. Count arena backing once, not backing plus every tensor view. Separate
requested/live bytes, reserved arena/pool bytes, and actual whole-card usage.

`peak = max_over_time(weights + optional_MTP + KV_and_scales + recurrent_state + live_workspaces + graph_storage + runtime_overhead)`

Add external card usage and the declared remaining reserve when deciding fit.
Do not sum independent stage peaks that never coexist. Conversely, tracked HIP
allocations can miss runtime/library/module/graph allocations or external users.
Reconcile allocation instrumentation, `hipMemGetInfo` and whole-card sampling;
record polling interval and use allocation hooks/stage-boundary readings so a
short load/repack peak is not missed between polls.

| Memory class | What to inventory |
| --- | --- |
| Target weights | GGUF source bytes versus unique resident payload, expanded/repacked layouts, device metadata, root head/embedding aliases, mapped-host ranges. |
| MTP static assets | NextN block/projections, proposal head, borrowed target-root aliases, host-dequantized copies and device copies. Separate shared pointers from duplicate physical payloads. |
| Persistent request state | Convolution/GDN state, canonical hidden/token rows, per-slot caches and scheduler resources. Recurrent state is not KV and is not halved by INT8 KV. |
| KV and metadata | K/V payload, scale dtype/granularity, BF16 mirrors, page tables/spans, page rounding, unused reserved pages, free chunks, prefix-cache pages and references. |
| Prefill transient | Chunk-dependent activations, attention temporaries, exact BF16 INT8-prefill oracle, staging/repack buffers, logits and prompt-hidden capture for MTP. |
| Decode/verify transient | Execution-row scratch, output head/logits, split reductions, R/P-sized activations, speculative journals/checkpoints and provider repair. |
| Graphs and caches | Cached graph/workspace buckets per width/depth/context, graph-pinned pages, retired references and allocator reserve. |
| Non-device costs | Host resident/pinned memory, mmap/GTT placement and transfers; these are not free VRAM savings if they stall generation or pressure system memory. |

Existing anchors: `hipengine/loading/qwen35_gguf_residency.py` provides planned
weight census and physical-range alias/duplicate audits; `hipengine/core/memory.py`
tracks allocations and arenas. Extend these rather than adding a second byte
counter disconnected from ownership. Inspect live callers before attributing
legacy helpers such as `_load_mtp_serving_assets()` to the staged product path.

### KV slope sanity check

Derive actual geometry from each artifact. For the documented 16 full-attention
layers, 4 KV heads and head dimension 256:

- BF16 K+V: `2 * 16 * 4 * 256 * 2 = 65,536 bytes/token` (64 KiB).
- INT8 K+V plus one FP32 scale per token/head for each of K and V:
  `2 * 16 * 4 * (256 + 4) = 33,280 bytes/token` (32.5 KiB).
- At 512 stored tokens this is 32 MiB versus 16.25 MiB per request, before
  page rounding/metadata and output/lookahead reservation. A 32K BF16 cache
  alone is 2 GiB. These are payload calculations, not measured process peaks.

If a c1-c8 peak grows around 0.85 GiB per request at 512 prompt tokens, most of
that growth cannot be explained by the prompt KV payload alone. Investigate
reserved S, private state/workspace, graphs, repeated weights and MTP allocation;
do not call it irreducible KV growth. Measure the slope against **reserved**
tokens and N as well as actual live C/L/D.

Compact INT8 is only compact if the steady route consumes it without a persistent
BF16 shadow. An INT8 payload plus scales plus BF16 mirror can use more memory
than BF16 alone. One bounded reusable prefill oracle is different from a mirror
retained per layer/request through decode; measure both peak and final residency.

### First measured XTX points (2026-09-06)

Partial Packet 0 freeze plus an early C1 boundary probe. Recorded here as data,
not as packet completion: the model-file census and the 18.618 GiB anchor
recovery are still open, so no Packet 0 box is checked.

Controls captured: host `epyc`, RX 7900 XTX as GPU index 1, device-reported
total 25,753,026,560 B (23.984 GiB), free 23.949 GiB idle, ROCm
7.2.53211-3d9ef42, `GPU_MAX_HW_QUEUES=1`, W7900 idle on GPU 0. Not yet
captured: GPU UUID/PCI ID in-artifact, display load, other GPU processes,
model-file hash and tensor census.

Route: `python -m hipengine.server` with `--max-active-requests 1`, explicit
`--kv-storage`, one completion per point, whole-card peak sampled from
`rocm-smi` on GPU 1. AR only; MTP not loaded. Prefix cache and graph mode not
recorded, so these are not yet stage-complete evidence.

| KV storage | `--max-context-tokens` | Result | Whole-card peak |
| --- | ---: | --- | ---: |
| `bf16` | 2,048 | starts and completes | 21.869 GiB |
| `bf16` | 3,072 | starts and completes | 23.328 GiB |
| `bf16` | 4,096 | HIP OOM during warmup | not reached |
| `bf16` | 8,192 | HIP OOM during warmup | not reached |
| `int8_per_token_head` | 2,048 | starts and completes | 21.869 GiB |
| `int8_per_token_head` | 3,072 | starts and completes | 23.328 GiB |
| `int8_per_token_head` | 4,096 | HIP OOM during warmup | not reached |

Every failure is `HipError: HIP error 2: out of memory` raised during eager
warmup, before any request is issued, so the boundary is a startup reservation
limit at that declared context and not a live-token limit.

Two observations that bear on the hypotheses above:

1. **The measured slope is roughly 22x the BF16 KV payload.** The two BF16
   successes differ by 1.459 MiB per declared token (2,048 -> 21.869 GiB,
   3,072 -> 23.328 GiB), against the 64 KiB/token K+V payload computed in the
   slope check. Per that section this must not be called KV growth; the excess
   is unattributed and is the target of Packet 2. The intercept is 18.951 GiB,
   of which 15.932 GiB is the GGUF file size, leaving ~3.019 GiB of unexplained
   fixed residency that the same packet should account for.
2. **INT8 shows no compaction at this stage.** `int8_per_token_head` and `bf16`
   produce identical peaks at both successful contexts and fail at the same
   4,096. This is consistent with the persistent-BF16-shadow failure mode named
   above, and it means the INT8 rows in section 2 cannot be treated as
   reproduced. Packet 3 owns the qualification.

These points do not resolve the memory-regression question. They neither
recover the 18.618 GiB anchor nor reconstruct the historical 32K BF16 / 112K
INT8 configurations, so the historical rows remain unmatched rather than
refuted. The withdrawn scoreboard figure of 21.869 GiB peak with 2.115 GiB
headroom reproduces exactly here at 2,048 declared tokens, which is a lead for
Packet 0's provenance recovery rather than evidence of a 16x regression.

Artifact:
[`XTX c1 context ceiling`](../benchmarks/results/2026-09-06-rx7900xtx-qwen38-c1-context-ceiling.json).

## 5. Ordered coder punchlist

Each packet ends with focused tests, an immutable worklog entry and a scoped
commit. Names for new harnesses/outputs below are proposed deliverables, not
existing CLI flags. No GPU benchmark was run to write this plan.

### Packet 0 — Freeze the XTX and resolve the memory-regression question

- [ ] Record physical host, GPU UUID/PCI ID, device-reported total/free memory,
  driver/ROCm/compiler, queue settings, display load and other GPU processes.
  Bind only the XTX; do not change another user's display/process settings.
- [ ] Hash and inventory the exact `Q4_K_M` and `Q4_K_S` files: layer/head
  geometry, every tensor format, root aliases, trailing NextN/MTP availability,
  and source bytes. Missing artifacts or MTP tensors are explicit blockers,
  not permission to silently substitute another model.
- [ ] Recover the supplied 18.618 GiB C1 128/24 reference. Re-run that exact
  route/shape once on current XTX source as the narrow regression check. If
  there is a matched difference, capture load/idle/prefill/decode allocation
  deltas before bisecting. If provenance is missing, report an unmatched anchor,
  not a regression or a repaired regression.
- [ ] Measure current 512/128 C1 through direct engine and real service owner,
  then separate C=N measurements. Record N, maximum sequence length, actual
  groups, prefix cache, graph mode and all MTP allocations in every run.
- [ ] Build a read-only byte estimate from the census and runtime policies;
  confirm it against measured allocations before using it to prune the sweep.

Exit: exact model identities, current XTX controls and an explained-or-explicitly
unmatched old memory anchor. Do not run a broad suite merely to answer whether
one old C1 number regressed.

### Packet 1 — Add stage-complete capacity evidence and admission checks

- [ ] Add a `scripts/qwen38_xtx_capacity.py` harness or extend the existing
  residency/context-soak harnesses. Emit one machine-readable row per attempt
  with command/source/host/model identity, requested/effective modes, C/N/L/D/S/K,
  physical groups, allocation ledger, stage peaks, timings, correctness and
  pass/failure classification. Save progress before each expensive stage.
- [ ] Instrument cold load/materialization, post-load idle, first request,
  prefill, steady decode, MTP verify/repair, width/context transitions, drain
  and teardown. Run fresh-process and warmed/reused-owner cases separately.
- [ ] Exercise N=1/2/4/8 with C=1 and full occupancy to separate upfront slot
  reservation from active execution cost. Sweep one axis at a time: context,
  output allowance, K and graph-cache history. Track cumulative cached buckets.
- [ ] Reconcile the Generation-2 resource ledger against complete peaks,
  including MTP R/P scratch and temporary INT8 BF16 oracles. Reserve before
  publishing slots or mutating state; reject/queue safely when the full claim
  cannot fit. Preserve request-budget-sized KV reservation unless a separately
  proven pool growth mechanism exists.
- [ ] Add CPU ledger tests for page rounding, shared aliases, peak lifetime
  overlap, mirror bytes, MTP optionality, graph retention and insufficient
  headroom. Hardware tests need explicit HIP availability guards.

Exit: measured peak agrees with named owners plus a recorded residual; pressure
produces bounded admission failure rather than an uncontrolled HIP OOM.

### Packet 2 — Measure and minimize the AR-only memory floor

Compare three explicit modes with the same target model/profile:

| Mode | Required distinction |
| --- | --- |
| AR-only, MTP not loaded | No NextN device weights, proposal state, speculative journals/workspaces/graphs or prompt-hidden capture owned solely for MTP. |
| MTP-capable but K0 | Record assets cached at startup or after prior use, even if zero MTP cycles execute. This is not the first mode. |
| Active MTP at K | Prove actual engagement and measure static plus C/K/R/P-dependent incremental bytes. |

- [ ] Trace optional load/materialization through the real owner; prove an
  AR-only request does not eagerly allocate assets merely because its GGUF
  includes NextN tensors. Metadata scanning or file mmap is not device residency.
- [ ] Deduplicate tied target/draft heads and embeddings through existing
  alias-aware loading. Inventory alternate quant payloads and free one-time
  conversion staging only after the consuming kernel/graph lifetime ends.
  Never count the same alias as a new allocation or free a borrowed owner.
- [ ] Audit per-slot scratch versus per-execution-group scratch, temporary
  full-vocabulary logits, rotary tables and long prompt-hidden arrays. Reuse or
  bound sequential scratch where ownership permits; avoid N copies of a
  workspace used by only one staged operation. Keep per-request GDN state distinct.
- [ ] Bound graph/workspace caching and reclaim safe retired buckets/arenas
  after synchronization. Test N-wide warmup followed by C1 and the reverse;
  capacity recommendations must not assume a process never served a wider row.
- [ ] Make true no-MTP loading an explicit supported capacity configuration
  if it is not already one. Lazy load/unload is optional and must be atomic:
  check the full activation budget, do not free graph-referenced assets, and
  fail/decline before mutation if MTP cannot be enabled. Permanent AR-only
  mode is preferable to unmeasured per-cycle loading across PCIe.

Exit: an exact current AR-only memory floor and measured MTP-residency delta,
with current throughput/latency controls and no lost ownership.

### Packet 3 — Qualify BF16 versus compact INT8, especially C>1

- [ ] Resolve `kv_storage`, scale dtype/granularity, effective attention source,
  mirror policy and artifact evidence from runtime telemetry, not requested
  flags. Start with BF16 and `int8_per_token_head`/FP32 scales where supported.
  Do not report a BF16 fallback as an INT8 pass.
- [ ] Measure INT8 payload/scales and zero persistent BF16 mirror after prefill.
  Attribute transient exact-prefill oracle size/lifetime, scratch and graph
  differences. A smaller steady cache that OOMs during prefill does not fit.
- [ ] Audit the current direct row-batched kernel, tests and
  `scripts/qwen38_int8_batch_decode_gate.py`, then trace actual service selection.
  Reuse the INT8 continuous campaign's ownership work. Separate compact serial
  residency from native packed no-mirror attention/prefill/graph/MTP support;
  a kernel test alone does not establish all those combinations.
- [ ] Close missing native compact C>1 boundaries in coordination with that
  campaign, including block-table-aware prefill, shifted pools, scale planes,
  graph replay, cancellation and reclaim. Publish serial capacity results only
  under their honest route label while native support is incomplete.
- [ ] Gate INT8 numerical/task quality against the same-weight BF16 KV teacher,
  including long prompts, decode horizons and all category/heldout scopes.
  MTP with INT8 additionally needs exact accept/commit/rollback ownership and
  distribution/trajectory gates; BF16 MTP evidence cannot authorize it.
- [ ] Qualify `Q4_K_S` INT8 independently; do not reuse `Q4_K_M` artifact hashes
  or gfx1151 decisions. Keep unsupported cases explicit instead of invoking an
  unverified-long override and presenting the outcome as supported serving.

Exit: per-artifact BF16/INT8 route and capacity evidence with mirror/serial
status, applicable quality gates and named integration blockers. No blanket
“INT8 doubles context” or “INT8 supports native c8” claim.

### Packet 4 — Evaluate smaller weights and other capacity candidates

- [ ] Compare `Q4_K_S` versus `Q4_K_M` unique resident weights and cold-load
  peaks under identical C/N/context/MTP settings. Derive savings from actual
  tensor formats/materialization, not file size or the preset suffix alone.
- [ ] Measure speed as well as capacity: a smaller quant can select slower
  projection kernels or require another resident layout. Qualify model quality
  with common teacher/task/category/heldout evidence and report the quantization
  tradeoff separately from same-quant implementation drift. Cross-quant token
  equality is not a required proof of quality.
- [ ] Price bounded prefill chunks, final-output-only prefill heads, reusable
  INT8 oracles and selective prompt-hidden capture against complete prefill
  wall, time to first token and MTP readiness. Do not drop intermediate verifier
  scores or necessary hidden seeds using a prefill-only optimization.
- [ ] Audit FP32 recurrent-state/workspace lifetimes before experimenting with
  lower precision. FP16/BF16 recurrent storage is a distinct numerical candidate,
  not an INT8 KV setting; require the full profile and long-trajectory gates.
- [ ] Measure prefix caching only as a separate shared-prefix workload. Report
  unique physical pages, logical context, copy-on-write and eviction pressure.
  Unrelated-prompt capacity is the baseline; do not count duplicated prompt
  tokens as additional physical capacity.
- [ ] If the ledger justifies it, screen existing mapped-host embedding/root
  placement with explicit host/pinned-memory and PCIe cost. Keep it separate
  from fully device-resident recommendations. No silent weight/KV offload.

Priority is exact lifetime/alias/workspace savings, then `Q4_K_S` and compact
INT8. More aggressive weights (for example existing Q3 paths), sub-INT8 KV,
KV eviction/DMS and general offload are optional follow-ups, not prerequisites.
They require their own model-quality or attention-semantics gates and must not
be conflated with lossless capacity improvements or full-context retention.

### Packet 5 — Search the C1 and C=N capacity boundaries

Use a staged sweep rather than every Cartesian combination at the longest
context. Validate a small supported case before increasing one resource axis.

- [ ] Primary quant/KV matrix: `Q4_K_M` / `Q4_K_S` × BF16 / compact INT8,
  each with AR-only, MTP-capable K0 and actual active-MTP rows. Unsupported
  quant/KV/MTP combinations get reasons, not fabricated measurements.
- [ ] C1: screen total budgets 4K, 8K, 16K, 32K, 64K, 96K, 112K and 128K
  within the model limit and supported envelope. Refine the last pass/first
  failure in page-aligned increments. Only extend beyond 128K if smaller-weight
  configurations and the byte estimate justify it; stop at the model limit.
- [ ] For each total budget S, use both short and sustained output allowances
  (D=24,128,512) with L reduced accordingly. Include at least one natural
  long-output workload and record actual output/EOS. A request that stops early
  may prove reserved memory, but does not prove the full decode horizon.
- [ ] C=N: start at 512 prompt / 128 output, N=1,2,3,4,5,6,7,8 where admitted;
  explicitly test the quoted c5/c6 boundary on the XTX. At each supported N,
  increase per-request budget through 1K,2K,4K,8K,16K then refine. Record both
  total logical tokens and unique physical KV pages. Wider N is conditional on
  measured savings and native-route support, not required to answer this card.
- [ ] Test N=8/C=1 and gradual fill/retirement separately from N=C=1. Include
  mixed short/long requests and heterogeneous output budgets, then sparse
  survivors and refill. An eight-slot configured owner may reserve more than a
  single-slot owner even when only one request is active.
- [ ] Measure MTP K1/K2/K3 where actually supported. Add K4-K7 as the better-MTP
  campaign enables them; until then record a functional dependency, not an
  out-of-memory boundary. Estimate and later verify larger R/P storage,
  including C8/K7 R64/P66 when rows6 padding applies. Low acceptance does not
  excuse non-functional deeper support, but it can make AR the recommendation.
- [ ] Run each near-boundary attempt in a fresh bounded subprocess, then repeat
  viable operational rows on a reused owner with changing graph buckets. Stop
  the ladder on OOM/stall, record the stage and check GPU health before retry;
  no repeated long hanging probes or automatic destructive GPU reset.

Exit: two distinct frontiers (C1 context and C=N context/width) for each supported
configuration, with a bracketed limit or explicit reason the ceiling is unknown.

### Packet 6 — Operational soak, quality and publication

Use `benchmarks/prompts/mtpbench-code-general-ja.jsonl`, every `code`,
`general_en`, `general_ja` and `mixed_ja_en` prompt, plus category-heldouts fixed
before tuning. Document deterministic long-context construction and retain
actual token counts. Teacher fixtures may be collected on a larger device when
necessary for quality evaluation, but capacity and performance claims still
require matched XTX execution.

- [ ] Test proposed recommendations on repeated natural conversations, full
  category/heldout suites and realistic prompt lengths. Use different prompts,
  not only repeated-token capacity fill. Separate numerical/task evaluation
  from timed fit runs so collecting full reference logits does not create the
  product's apparent memory requirement.
- [ ] Exercise simultaneous admission, staggered arrival, cancellation, refill,
  prefix sharing/eviction, MTP activation/deactivation where supported, and
  pressure above the limit. Fail/queue before mutation; exact ownership, output
  usage and zero final leaks are required. A smaller static run is not a soak.
- [ ] Pair memory candidates with current same-host throughput, prefill/first-
  token latency, inter-token p50/p95/p99 and fairness controls. MTP speed claims
  require true no-MTP AR with the same model/quant/KV/C/N/workload, not a disabled
  verifier loop. Keep automatic MTP off where it loses or is unqualified.
- [ ] Publish both dedicated-card and declared-display-reserve recommendations;
  retain smaller wins in scope even if a configuration still misses a desired
  context/width. Never lower a quality gate or silently reduce output allowance
  merely to label a configuration “fits.”
- [ ] Save compact artifacts and exact commands, update benchmark README/date,
  changelog and relevant public capacity text, and run the README export check.
  Replace ambiguous W7900-to-XTX inferences only when this campaign has the
  corresponding XTX evidence; do not rewrite immutable old worklogs.

## 6. Implementation and validation anchors

| Existing surface | Campaign use |
| --- | --- |
| `hipengine/loading/qwen35_gguf_residency.py`, `qwen35_gguf_materialize.py`, `qwen35_gguf_nextn_materialize.py` in the same directory | Source/resident byte census, tied aliases, alternate payload and load-transient audits. |
| `hipengine/core/memory.py`, `hipengine/runtime/workspace.py`, `hipengine/runtime/qwen35_gguf_runner.py` | Allocations, arena/workspace liveness, recurrent state, INT8 mirrors/oracles and packed target scratch. |
| `hipengine/generation/qwen35_gguf.py`, `hipengine/generation/qwen35_gguf_mtp2.py`, `hipengine/speculative/serving.py` | Actual service owner, optional draft assets, physical groups and speculative resource claims. |
| `hipengine/kvcache/ledger.py`, `device_global.py`, `global_pool.py`, `graph_binding.py` in the same directory | Atomic admission, pool capacity, shared pages and graph-pinned ownership. |
| `hipengine/models/kv_capabilities.py`, `hipengine/models/qwen35.py` | Artifact-scoped KV and MTP qualification; preserve unrelated models/backends. |
| `scripts/gguf_residency_g6.py`, `scripts/qwen38_int8_server_context_soak.py`, `scripts/qwen38_int8_batch_decode_gate.py`, `scripts/gguf_packed_ar_bench.py` | Reusable census, context/soak, direct-INT8 correctness and direct-AR controls; validate current parser and route before reuse. |
| `tests/test_qwen35_gguf_residency.py`, `tests/test_kvcache_resource_ledger.py`, `tests/test_kvcache_global_device_pool.py`, `tests/test_qwen38_int8_kv_capability.py`, `tests/test_qwen38_int8_server_context_soak.py` | Focused regression anchors; add lifecycle/peak tests for each changed owner. |

Follow [`TESTING.md`](TESTING.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)
and [`BENCHMARK.md`](BENCHMARK.md). Exact allocation/lifetime changes require
exact tokens and ownership in their declared scope. Arithmetic, KV or weight
quantization changes additionally require their applicable numerical and task
quality gates; a CPU-reference smoke threshold is not full promotion evidence.
New kernels need catalog/lineage review, an oracle/RED test, a registered strict
fallback and cached-build `rocprofv3 --kernel-trace` proof on the XTX.

Keep the runtime torch-free and plugin-based. Coordinate high-conflict owner,
registry and server files. Use an isolated clean worktree for benchmarks if
other agents are editing shared code. Track temporary flags, serial fallback
removal and cache policies in [`REFACTOR.md`](REFACTOR.md).

## 7. Required output tables and completion audit

One detailed row per attempted configuration includes:

`host + GPU + source + model hash + weight quant + profile/variant + requested/effective KV/scales/mirror + MTP residency/engaged K + route/groups + C/N/L/D/S + prefix policy + graph mode/history + unique weights + MTP bytes + KV/scales/state + peak stage + tracked/reserved/whole-card peaks + headroom H + quality/lifecycle + speed/latency + exact command + classification`.

Publish two compact user tables, backed by those rows:

- **One request:** quant, KV, MTP mode/depth, total budget, maximum prompt at
  stated output allowance, peak/headroom, latency/rate, operational versus
  physical limit, and exact supported settings.
- **Concurrent requests:** quant, KV, MTP, resident N, actual native C/groups,
  per-request prompt/output budget, aggregate unique KV, peak/headroom,
  throughput/per-request latency, prefix assumptions and exact settings.

- [ ] Quoted 128/24 and 512/128 memory differences are matched and explained,
  or explicitly unmatched; no unsupported regression claim.
- [ ] Each required quant/KV/AR/MTP axis has current XTX measurements or an
  explicit source-grounded integration/quality blocker. Estimates, serial
  fallback, native execution and actual OOM remain distinguishable.
- [ ] C1 long context and C=N capacity have separate same-host boundaries,
  output reservations, memory ledgers and operational recommendations.
- [ ] AR-only omits unnecessary MTP assets; active MTP and compact INT8 claims
  prove the requested route, effective depth and mirror status.
- [ ] Every retained memory improvement is enabled within its validated scope
  or has a concrete blocker, with quality, latency, ownership and clean drain.
- [ ] Deferred native C1/deeper MTP or compact C>1 work is linked to its owning
  campaign; capacity estimates cannot close those functional requirements.
- [ ] Artifacts, public tables, architecture links, cleanup ledger and immutable
  handoff agree. The result states what fits now, what is recommended, and what
  remains unmeasured—without treating nominal spare VRAM as a guarantee.
