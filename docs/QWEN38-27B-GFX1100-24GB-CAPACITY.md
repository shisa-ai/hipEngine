# Qwen3.8-27B: capacity on a 24 GB RX 7900 XTX

Status: measurement and optimization plan. Initial startup probes are recorded
below; current operational context and concurrency limits are not qualified.

## 1. Scope and required results

Measure Qwen3.8-27B on one RX 7900 XTX (`gfx1100`) across:

- `Q4_K_M` and `Q4_K_S` weights;
- BF16 KV, compact INT8 KV, and qualified FastDMS eviction;
- AR-only with MTP unloaded, MTP assets retained at K0, and active MTP;
- one long request and multiple resident/physically batched requests.

AR is autoregressive decode without speculation; MTP is multi-token prediction.
BF16 here describes the key/value (KV) cache, not full BF16 model weights.

Publish separate tables for **C1 context capacity** and **concurrent capacity**.
Each recommendation must name prompt/output limits, actual execution route,
peak memory, reserve, quality gate, throughput, latency and reproducible settings.
Report the largest observed pass, first tested failure and operational setting
separately. A W7900 allocation estimate does not establish an XTX fit limit.

Coordinate implementation with the existing campaigns:

| Owner | Boundary |
| --- | --- |
| [`CONCURRENCY2.md`](CONCURRENCY2.md) | Shared KV pool, admission and request lifecycle. Do not add another scheduler or private full-capacity KV pool per request. |
| [`INT8 continuous batching`](QWEN38-INT8-KV-CONTINUOUS.md) | Compact INT8 consumers, prefill and native batch integration. |
| [`Better MTP`](QWEN38-27B-GFX1100-CONCURRENCY2-BETTER-MTP.md) | Native C1 and K1-K7 functionality. This campaign measures their XTX memory cost as they become available. |
| [`DMS.md`](DMS.md) | Trained eviction policy, compact storage, quality and product integration. Include its capacity potential and remaining prerequisites here. |
| [`TP2`](QWEN38-27B-GFX1100-TP2.md) | Separate two-GPU work; cannot supply memory for an XTX-only claim. |

## 2. Evidence audit

### Historical controls

The supplied 18.618 GiB C1 128-prompt/24-output reference still needs its source,
command, owner and metric recovered. It cannot establish a regression against
a different workload or a whole-card measurement.

The [2026-09-06 W7900 direct sweep](../benchmarks/results/2026-09-06-gfx1100-qwen38-q4km-direct-c1c8-sweep.json)
uses 512 prompt / 128 output tokens and reports tracked allocations, including
about 23.7 GiB at c6. The supplied c5/c6 estimates of 23.3/24.2 GiB use a different
or unresolved measurement basis. Neither is a measured XTX c6 boundary.

Historical XTX results in [`KVCACHE.md`](KVCACHE.md) include a 32K BF16 operational
row and 52K near-capacity row. The [112K INT8 qualification](../worklog/entries/20260815T182245.795089Z-lhl-qwen38-dedicated-context-qualification-59a182.md)
completed four natural requests sequentially with MTP and prefix caching off.
The [later audit](../worklog/entries/20260816T071413.192306Z-lhl-qwen38-int8-context-quality-concurrency-dfa478.md)
separates that setting from the near-zero-headroom 126K ceiling. These older
configurations have not been reconstructed on the current owner. Short output
suffixes also do not qualify a long-output request at the same prompt length.

### Initial XTX probes, 2026-09-06

The [aggregate artifact](../benchmarks/results/2026-09-06-rx7900xtx-qwen38-c1-context-ceiling.json)
reports host `epyc` and RX 7900 XTX at GPU index 1; W7900 is GPU 0. The original
document additionally recorded ROCm 7.2.53211-3d9ef42,
`GPU_MAX_HW_QUEUES=1`, total 25,753,026,560 bytes (23.984375 GiB), and
23.949 GiB free at idle. These additional fields need per-run provenance. Device UUID/PCI
mapping, model hash, resolved profile, graph/prefix settings, effective KV route
and allocation census were not captured in that artifact.

The points predate the checked-in
[`gguf_context_ceiling_probe.py`](../scripts/gguf_context_ceiling_probe.py).
The artifact says an equivalent ad-hoc script collected them; source identity
and per-point commands must be recovered before claiming exact reproduction.

| Requested KV | Configured maximum context | Reported result | Sampled whole-card high water |
| --- | ---: | --- | ---: |
| BF16 | 2,048 | Server started; completion reported | 21.869 GiB |
| BF16 | 3,072 | Server started; completion reported | 23.328 GiB |
| BF16 | 4,096 | Warmup OOM reported | 16.900 GiB, incomplete sampling |
| BF16 | 8,192 | Warmup OOM reported | 17.041 GiB, incomplete sampling |
| INT8 per-token/head | 2,048 | Server started; completion reported | 21.869 GiB |
| INT8 per-token/head | 3,072 | Server started; completion reported | 23.328 GiB |
| INT8 per-token/head | 4,096 | Warmup OOM reported | 18.269 GiB, incomplete sampling |

OOM means out of memory. The artifact attributes the failures to HIP error 2
during eager warmup, before a request. Its individual rows say `REQUEST_FAILED`;
the original logs are needed to reconcile those labels with the warmup finding.
The failed samples are not the allocation peaks at failure.

**What the data supports:** this reported startup configuration passed at 3,072
and failed at 4,096 declared tokens. No sampled memory difference appeared
between the requested KV formats at the two passing points.

**What it does not establish:** a 3K live-context ceiling, an AR-only memory
floor, an INT8 mirror, or a regression against the historical 32K/112K routes.
The artifact's `regression_found` status and replacement-ceiling wording exceed
its own stated limitations. Recover the controls before using those conclusions
in public capacity tables; do not substitute the older limits as current facts.

### Probe limitations and required repairs

Source review of the checked-in probe found:

- The request uses `"a " * (context // 2)` and 16 output tokens by default.
  It neither counts prompt tokens nor validates returned usage. Configured
  context is not demonstrated live context. A response containing `"text"`
  passes without checking generated count, content or finish reason.
- VRAM is polled before readiness and once after the blocking completion call,
  **not during prefill/decode**. The default interval is five seconds, and each
  `rocm-smi` call can add delay. Even successful samples are lower bounds on
  the true peak. No paired evidence supports a fixed 0.2–0.9 GiB gap from the
  tracked allocator.
- `--gpu` records an ordinal and sets `HIP_VISIBLE_DEVICES`; it does not verify
  the runtime UUID/PCI mapping against the sampled card. The environment is
  inherited, and an existing server on the port is not explicitly ruled out.
- Requested KV is recorded, not effective storage/scales/mirrors. The command
  does not explicitly disable MTP or audit its allocations. A K0 default is
  not proof that MTP assets were never loaded.
- Any response containing `hiperror` is classified as OOM, even if it is a
  different HIP error. Timeouts, readiness failures, process exits and HIP
  error 2 need distinct classifications backed by stage logs.

Existing pure-helper tests do not cover these end-to-end limitations. Packet 1
repairs the harness before new operational claims. No GPU rerun was performed
for this documentation audit.

Converting GiB to MiB, the two-point slope is
`(23.328 - 21.869) * 1024 / (3072 - 2048) = 1.459 MiB` per
additional **declared** token, or **23.344×** the 64 KiB BF16 payload below.
This is not an allocation attribution. The extrapolated 18.951 GiB intercept
is not measured fixed residency; subtracting the 15.932 GiB GGUF file size does
not identify runtime overhead. Resident weight payloads may differ from file
bytes. Do not extrapolate a precise token ceiling from two rounded samples.

## 3. Shared pool and memory model

Use C for active requests, N for configured resident slots, L for prompt tokens,
D for maximum output tokens and S for the complete reservation, including
lookahead/guard slots. K is draft depth; K0 is AR. Verifier logical rows are
`R=sum(1+k_i)`; record actual padded rows P. Context 1K means 1,024 tokens.
Report memory in bytes and GiB (`bytes / 2^30`).

### Existing shared ownership

Compatible requests share one backend-declared global KV pool set. This shares
**capacity**, not necessarily token contents: unrelated requests still need
distinct live pages. Prefix reuse is a separate copy-on-write/refcount contract.
BF16 and INT8 storage cannot be reinterpreted in place as each other's pages.
See [`CONCURRENCY2.md`](CONCURRENCY2.md#sharing-domain-and-backend-replacement).

`hipengine/generation/qwen35_gguf.py` already selects
`create_global_device_kv_pool` when available. Its pool setup derives request
capacity from N and context, then adds a leased packed-workspace region sized
for at least eight slots and at least 1,024 positions per slot. Trace whether
this setup runs in the measured configuration and inventory its page format
and backing bytes. It is a concrete reservation candidate, **not yet an
explanation of the measured slope**. The legacy chunked-pool fallback and
private session preparation must be labelled if reached.

Report separately:

1. Unique physical pool backing, including leased workspace planes.
2. Live request pages/extents, prefix-retained pages, free reusable capacity,
   alignment/fragmentation and admission credits.
3. Other resident state, scratch and graphs, including any redundant private KV.

Reclaiming pages inside a preallocated pool increases reusable capacity without
necessarily decreasing `rocm-smi` usage. Prove capacity savings by fitting more
live work into the same backing, or the same work into a smaller load-time pool.
Do not count arena backing plus its tensor views twice or sum request maxima
as though each owned a separate full pool.

### Allocation ledger

Measure simultaneous lifetimes, not the sum of independent stage peaks:

`process_peak = max_t(weights + MTP + pool_backing + recurrent_state + workspaces + graphs + runtime_allocations)`

Reconcile tracked requested/reserved bytes with device free memory and whole-card
sampling. Add external usage and the declared reserve once. Include cold load,
repack, prefill, verification, graph-cache growth, cancellation and teardown.

| Class | Required audit |
| --- | --- |
| Weights | Unique device payload versus source bytes, expanded/repacked layouts, root/head aliases, temporary conversion copies. |
| MTP | NextN weights, borrowed roots, provider state, hidden capture, journals and R/P-sized scratch; distinguish unloaded, cached K0 and engaged K. |
| KV | Pool planes, scales, BF16 mirrors, metadata, retained/free pages and workspace leases. |
| Hybrid state | Per-request convolution/Gated DeltaNet (GDN) state and checkpoints. KV compression does not reduce this state. |
| Transients | Prefill chunks/oracles, logits, projection scratch, graph-pinned buffers and cached buckets. |
| Host memory | Resident/pinned memory, mmap/GTT placement and PCIe transfers; label offloaded configurations separately. |

Reuse `hipengine/loading/qwen35_gguf_residency.py` for planned/physical weight
census and aliases, `hipengine/core/memory.py` for allocations, and the
Generation-2 ledger/pool telemetry for shared backing and claims.

For the documented 16 full-attention layers, four KV heads and head dimension
256, dense payload per stored token is:

- BF16: `2 * 16 * 4 * 256 * 2 = 65,536 bytes` (64 KiB).
- INT8 with FP32 K/V scales per token/head:
  `2 * 16 * 4 * (256 + 4) = 33,280 bytes` (32.5 KiB).

At 512 stored tokens this is 32 MiB versus 16.25 MiB per request, excluding
rounding, metadata and output reservation. Verify geometry for each artifact.
An INT8 cache retained alongside BF16 is not compact. Audit effective dispatch
and both persistent and transient bytes before assigning a cause to equal peaks.

## 4. FastDMS capacity target

Dynamic Memory Sparsification (DMS) removes selected KV history. The in-tree
FastDMS path includes trained sidecar tooling, compact pools/extents, transactional
pack/append/eviction and device attention. The
[Generation-2 DMS status](CONCURRENCY2.md#c2-7--fastdms-topology-and-codec-composition)
records fixture-backed c1-c32 device/lifecycle qualification and INT8 composition,
but leaves integrated model/product qualification open.

[`DMS.md`](DMS.md) records an exact-budget trained policy with an 8,192-token
protected window, qualified on source-disjoint 32K/128K gfx1151 suites. Reported
live-cell compression is about 1.60×/1.882×. Its older branch/merge notes are
historical; use current source for implementation presence. Those results do
not qualify the XTX, another quant artifact, or concurrent MTP serving.

### What “2× saving” means

For a target retaining half the eligible history, context T and protected
window W, the ideal retained tokens per head are:

`live = min(T,W) + ceil(max(0,T-W)/2)`

Thus compression approaches 2× for T much larger than W; it is not 2× at short
context and never means half the total model VRAM. Using W=8,192 and BF16
payload at the geometry above:

| Context | Ideal live tokens/head | KV compression | Dense payload | Compact payload |
| --- | ---: | ---: | ---: | ---: |
| 8K | 8,192 | 1.000× | 0.500 GiB | 0.500 GiB |
| 32K | 20,480 | 1.600× | 2.000 GiB | 1.250 GiB |
| 128K | 69,632 | 1.882× | 8.000 GiB | 4.250 GiB |

These are payload calculations, not XTX capacity results. Predictor bytes,
metadata, extent granularity and reserve add overhead. For shorter contexts a
smaller protected window needs new quality evidence; do not replace the qualified
window with 256 merely to project nearly 2× compression.

A trained predictor alone is insufficient. Savings require compact storage that
releases or reuses evicted cells, no persistent dense shadow, and bounded
prefill peaks. Dense-prefill-then-compact can lower decode residency while still
failing to load/prefill on 24 GB. In a fixed shared pool, fewer occupied extents
must translate into additional admissions or a smaller pool plan.

DMS and INT8 are different axes: DMS keeps fewer tokens; INT8 stores each retained
token more compactly. Their ideal long-context payload savings can approach
4× relative to dense BF16, but scales/window/metadata reduce this and the
composition needs independent quality and kernel qualification. Neither
compression factor applies to weights, recurrent state or the whole process.
DMS changes attention history; it is not lossless full-context retention.

## 5. Ordered implementation packets

Commit each tested unit with a worklog entry. Hardware tests need HIP guards;
new kernels require lineage/catalog review, an oracle, registered strict fallback
and cached-build profiler evidence. Preserve torch-free/plugin boundaries and
coordinate shared-file edits with the INT8, MTP and DMS owners.

### Packet 0 — Establish matched XTX controls

- [ ] Record model hashes/tensor census, XTX UUID/PCI identity, driver/compiler,
  profile/variants, environment, display/other usage and actual total/free bytes.
- [ ] Recover the 18.618 GiB 128/24 and historical 32K/112K commands and routes.
  Run a narrow matched control first; mark unrecoverable anchors unmatched.
- [ ] Freeze direct versus service-owner 512/128 controls, C/N, S, pool plan,
  prefix policy, graph mode and MTP residency. Verify device selection at runtime.
- [ ] Recover original probe logs/commands; correct unsupported public ceiling
  or regression interpretations without modifying recorded sample values.

### Packet 1 — Repair the probe and account for shared reservations

- [ ] Tokenize prompts and record actual L, requested/actual D, usage and finish
  reason. Complete the intended horizon or label it early-stop; do not equate
  configured context with live tokens. Validate response schema and correctness.
- [ ] Sample through load, prefill and decode concurrently with the request;
  record interval/gaps and allocation-stage peaks. Treat samples as lower bounds
  unless transient peaks are covered by instrumentation.
- [ ] Verify child/device/port identity; freeze inherited configuration. Record
  effective KV/scales/mirrors, MTP allocation/engagement, profile, pool/workspace
  plan, graph/prefix settings, full command, source and exit/stage logs.
- [ ] Classify OOM only from matching error evidence; test other HIP errors,
  process exits, readiness/request timeouts and malformed/short responses.
- [ ] Compare N=1/2/4/8 at C=1 and full occupancy. Attribute global request pages,
  leased workspace, private preparation, recurrent state and cached graphs.
  Trace the eight-slot workspace minimum before changing it.
- [ ] Test atomic admission against complete claims, including temporary/MTP
  peaks. Shared free capacity, not a per-request private cache maximum, decides
  aggregate fit. Retain bounded rejection and exact reclaim under pressure.

### Packet 2 — Reduce exact allocation costs

- [ ] Prove AR-only omits NextN weights and MTP-only state/graphs/hidden capture.
  Measure MTP-capable K0 both before and after active MTP; do not call it unloaded.
- [ ] Deduplicate actual weight aliases and release conversion staging safely.
  Count resident payloads, not GGUF size, as the device-weight baseline.
- [ ] Right-size shared workspace and prefill scratch by supported execution
  shapes; reuse sequential scratch without merging per-request GDN state.
  Preserve graph pointer lifetimes and overlap constraints.
- [ ] Bound cached graph/workspace buckets and safe retirement. Test wide-to-C1
  and C1-to-wide histories. Report reusable pool capacity separately from memory
  returned to the device allocator.
- [ ] Provide a true AR-only configuration. Optional lazy MTP activation must
  reserve its full peak before mutation; unloading must not free borrowed or
  in-flight graph assets. Do not load/unload weights per decode cycle.

### Packet 3 — Qualify smaller weights and compact INT8

- [ ] Compare actual resident and load-peak `Q4_K_M`/`Q4_K_S` bytes, throughput
  and quality. Inventory MTP tensor availability and preserve artifact identity.
- [ ] Prove effective compact INT8 has no persistent BF16 shadow. Bound and
  reuse prefill oracles; a smaller steady cache with an oversized prefill peak
  does not fit. Account for scales, pool planes and graph scratch.
- [ ] Inspect existing `qwen38_int8_batch_decode_gate.py` and service selection.
  Separate serial compact residency from native no-mirror batched prefill,
  decode, graphs and MTP. Coordinate missing integration with the INT8 campaign.
- [ ] Gate each quant's INT8 against its same-weight BF16 teacher. Do not reuse
  `Q4_K_M`/gfx1151 evidence for `Q4_K_S`/XTX. MTP composition needs its own
  acceptance, selected-prefix and provider/target rollback tests.
- [ ] Price bounded prefill, selective output heads and hidden capture against
  first-token latency. Lower-precision recurrent state is a separate numerical
  candidate, not a KV switch. Never omit required verifier scores.

### Packet 4 — Qualify FastDMS capacity on the shared pool

- [ ] Inventory the trained sidecar, hash/model/quant binding, protected window,
  calibration and actual eviction policy. Missing or mismatched qualification
  fails closed; training alone does not authorize serving.
- [ ] Measure dense BF16, compact no-evict and trained DMS BF16 through the same
  owner. Record logical tokens, per-layer/head survivors, allocated extents,
  free capacity, fragmentation and all transient/metadata/predictor bytes.
- [ ] Trace streaming compact prefill and direct compact decode. Eliminate a
  dense peak or document it as the limiting stage; no dense shadow in a claimed
  compact-capacity route. Verify shared-pool credits recover after eviction.
- [ ] Run native C1 then C2/C4/C8 where supported, heterogeneous lengths,
  protected-window boundaries, cancellation, pressure and refill. DMS prefix
  sharing remains off until snapshot/overlay semantics qualify; sharing pool
  capacity does not authorize sharing divergent evicted histories.
- [ ] Gate the trained policy against dense and no-evict controls on all
  categories/heldouts and long trajectories. Add MTP provisional-state/eviction
  rollback before combining them; rejected drafts must not evict committed KV.
- [ ] Evaluate DMS+INT8 only after independent codec/topology gates. Measure
  actual compression and quality; do not multiply nominal factors into a fit
  claim. Keep scope-specific failures linked to the DMS campaign.

### Packet 5 — Measure context and concurrency limits

- [ ] Sweep supported quant × KV/topology × MTP combinations, classifying each
  as estimate, unsupported, load-only, execution fit, operational fit, OOM or
  stall. Preserve requested/effective modes and actual physical groups.
- [ ] C1: start below the reported startup boundary, then test 4K/8K/16K/32K/
  64K/96K/112K/128K total budgets as supported. Refine last pass/first failure
  at page-aligned steps; extend only with a justified byte estimate and within
  the model context limit. Do not infer a physical ceiling from a policy cap.
- [ ] Budget D=24/128/512 and reduce L to leave output/lookahead space. Include
  natural long-output cases; an early EOS proves only the work actually done.
- [ ] Concurrent baseline: 512 prompt/128 output at N=C=1 through 8, including
  the disputed c5/c6 boundary. Increase per-request budgets through 1K/2K/4K/
  8K/16K where possible. Add N=8/C=1, mixed lengths, gradual fill and survivors.
- [ ] Measure K1-K3 where engaged; include K4-K7 as their owning campaign
  qualifies them. A functional blocker is not a memory ceiling. Record R/P and
  peak journals/scratch, including possible C8/K7 R64/P66.
- [ ] Use bounded fresh processes near failure, then repeated reused-owner
  tests with changing buckets. Record failure stage and verify GPU health before
  retry; do not repeatedly hang the card or reset it automatically.

### Packet 6 — Select operational settings and publish

- [ ] Freeze dedicated-card and display-reserve contracts before measurement.
  Suggested initial reserves are 512 MiB and an additional 2 GiB respectively;
  these are policy choices, not hardware constants. Avoid double-counting usage.
- [ ] Run the full `benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite
  (`code`, `general_en`, `general_ja`, `mixed_ja_en`) and fixed category-heldouts.
  Document long-context construction; repeated-token probes are not task gates.
- [ ] Test admission, cancellation, refill, overload, teardown and relevant
  prefix/MTP transitions. Require exact ownership, usage and clean drain.
- [ ] Compare complete memory and latency/throughput on the same XTX, with
  paired repeated controls. MTP needs true no-MTP AR; arithmetic, weight/KV
  quantization and DMS eviction need their applicable numerical/task gates.
  Collect reference logits separately from timed capacity runs.
- [ ] Report physical fit separately from useful service. Keep exact savings
  with non-regressive controls; report explicit tradeoffs for smaller quant,
  mapped-host placement or slower compact routes. More aggressive quantization,
  sub-INT8 codecs and general offload remain separate optional experiments.
- [ ] Update compact artifacts, benchmark README/date, changelog and public
  settings. Run `scripts/sync_benchmark_readme.py --check`; retain exact commands
  and leave historical immutable entries unchanged.

## 6. Validation anchors and completion

Use the relevant tests under `tests/` for residency, resource ledger/global pool,
INT8 capability/batch decode, MTP transactions, DMS extents/device rollback and
`test_gguf_context_ceiling_probe.py`. Add end-to-end harness tests for the gaps
in section 2. Normative gates are [`TESTING.md`](TESTING.md),
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) and [`BENCHMARK.md`](BENCHMARK.md).
Track temporary routes/flags in [`REFACTOR.md`](REFACTOR.md).

Each result must include host/GPU identity, source/command, model/sidecar hashes,
profile/variants, quant/KV/topology, effective MTP/route, C/N/L/D/S/K/R/P, pool
plan and cache history, stage peaks, tracked/reserved/sampled bytes, headroom,
quality/lifecycle evidence and throughput/latency.

- [ ] Old/current comparisons are matched or explicitly unresolved. No declared
  context is presented as a demonstrated live-token limit.
- [ ] Shared backing, occupied/free capacity and workspace leases reconcile
  with peaks; AR-only, INT8 mirror status and MTP engagement are measured.
- [ ] Separate C1 and concurrent tables report largest pass, first failure and
  operational prompt/output settings with actual physical execution labels.
- [ ] FastDMS has an XTX capacity/quality result or a named integration/sidecar
  blocker. Its eligible-history compression is not reported as total-VRAM saving.
- [ ] Qualified improvements are enabled in scope; unsupported/losing automatic
  MTP choices remain K0. Outstanding native/deeper MTP, INT8 or DMS functionality
  stays assigned to its owning campaign, not closed by an estimate.
- [ ] Artifacts, public claims, plan and immutable handoff agree with measured
  evidence and stated limitations.
