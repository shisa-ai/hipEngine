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

- [x] Record model hashes/tensor census, XTX UUID/PCI identity, driver/compiler,
  profile/variants, environment, display/other usage and actual total/free bytes.
  Recorded 2026-09-06: Q4_K_M 17,106,773,984 B / 866 tensors,
  Q4_K_S 16,121,359,328 B / 866 tensors with inventory hashes, XTX PCI
  `0000:10:00.0` unique_id `0xcc4d02090dc9c3ff`, ROCm 7.2.4 userspace on driver
  7.1.3-2-cachyos. Artifact:
  `results/2026-09-06-rx7900xtx-capacity-packet0-controls.json`.
- [x] Recover the 18.618 GiB 128/24 and historical 32K/112K commands and routes.
  Run a narrow matched control first; mark unrecoverable anchors unmatched.
  Recovered 2026-09-06 with per-anchor statuses (soak 112K INT8 RECOVERED with
  harness+measured commits, 126K page-aligned probes RECOVERED, BF16 32K graph
  row PARTIALLY RECOVERED); unsupported public interpretations corrected in the
  same unit. Artifact:
  `results/2026-09-06-rx7900xtx-capacity-packet0-controls.json`.
- [x] Freeze direct versus service-owner 512/128 controls, C/N, S, pool plan,
  prefix policy, graph mode and MTP residency. Verify device selection at runtime.
  Frozen 2026-09-06: direct-probe and service-owner 512/128 controls with
  C/N=1, MTP off, prefix off, one cold server per point, runtime device
  cross-check. Artifact:
  `results/2026-09-06-rx7900xtx-startup-boundary-repaired-probe.json`.
- [x] Recover original probe logs/commands; correct unsupported public ceiling
  or regression interpretations without modifying recorded sample values.
  Done 2026-09-06: original anchors recovered with commands where available;
  the published 32K/112K context rows were withdrawn and the INT8 "no memory
  saving" row was re-interpreted as an unengaged BF16 fallback, with recorded
  samples left untouched.

### Packet 1 — Repair the probe and account for shared reservations

- [x] Tokenize prompts and record actual L, requested/actual D, usage and finish
  reason. Complete the intended horizon or label it early-stop; do not equate
  configured context with live tokens. Validate response schema and correctness.
  Done 2026-09-06 (`caacf3bcf`): the probe fits the prompt to the exact token
  target through the server tokenizer and validates choices/usage/finish
  reason/authoritative token IDs before accepting a sample.
- [x] Sample through load, prefill and decode concurrently with the request;
  record interval/gaps and allocation-stage peaks. Treat samples as lower bounds
  unless transient peaks are covered by instrumentation.
  Done 2026-09-06: 20 ms sysfs `VramSampler` spans startup through request, and
  warmup-failure sampling was corrected (the prior artifact's 16.9 GiB was
  incomplete sampling; true 4,096 failure peak 23.951 GiB).
- [x] Verify child/device/port identity; freeze inherited configuration. Record
  effective KV/scales/mirrors, MTP allocation/engagement, profile, pool/workspace
  plan, graph/prefix settings, full command, source and exit/stage logs.
  Done 2026-09-06: the probe resolves the card by PCI id, requires the server
  readiness payload to report the same device, and records effective KV/MTP/
  pool/graph state with the full command; the INT8-with-FP16-scales fallback
  (`runtime_action fallback_bf16`) is now classified instead of misreported.
- [x] Classify OOM only from matching error evidence; test other HIP errors,
  process exits, readiness/request timeouts and malformed/short responses.
  Done 2026-09-06: OOM matches only `out of memory` / `hipErrorOutOfMemory` /
  `HIP error 2`; other HIP errors, process exits, readiness/request timeouts
  and malformed/short responses are classified separately.
- [x] Compare N=1/2/4/8 at C=1 and full occupancy. Attribute global request pages,
  leased workspace, private preparation, recurrent state and cached graphs.
  Trace the eight-slot workspace minimum before changing it.
  Traced 2026-09-06: the floor lives in `_packed_verify_union_geometry` and the
  global-pool lease in `configure_engine_loop`; serving slot requests (MTP
  widths, packed group layouts, prefill widths) are all bounded by
  `max_active_requests`, and the only >1-slot C1 path (B3 C2-shadow ABI) has no
  callers. Right-sized both sites to the honest capacity (see Packet 2).
- [x] Test atomic admission against complete claims, including temporary/MTP
  peaks. Shared free capacity, not a per-request private cache maximum, decides
  aggregate fit. Retain bounded rejection and exact reclaim under pressure.
  Passed 2026-09-06 on the XTX (BF16 KV, contexts 512/1024/2048, max-active 3):
  all six rescaled workloads pass route/correctness/SLO gates; the pressure
  contract holds — a live 2048-token row completes while a 1024-token candidate
  receives retryable 429 `engine_busy` with exact admission metadata
  (requested=5 vs capacity=39 units, the complete-claim accounting), then the
  pool shrinks/regrows with fresh block ids, graphs rebind, memory recovers and
  ownership drains exactly. The gate gained `--required-contexts` rescaling and
  KV-policy plumbing; the dual-session exactness harness does not fit at 3.1K
  max-sequence on 24 GB (prepared owner + reference session > card), and the
  continuous packed owner remains fail-closed to BF16 KV (INT8 requests serve
  exact but serially). Artifact:
  `results/2026-09-06-rx7900xtx-capacity-packet1-admission-pressure.json`.

### Packet 2 — Reduce exact allocation costs

- [ ] Prove AR-only omits NextN weights and MTP-only state/graphs/hidden capture.
  Measure MTP-capable K0 both before and after active MTP; do not call it unloaded.
  AR-only omission proven 2026-09-06: with MTP serving off the materialization
  plan omits the 4 NextN tensors (0.052 GiB source bytes), the resident census
  holds 851 tensors / 15.992 GiB with zero NextN bytes, and the draft provider
  is lazily acquired and pooled (never load/unloaded per decode cycle) — so
  true AR-only is the default configuration. Evidence:
  `results/2026-09-06-rx7900xtx-capacity-ar-only-nextn-omission.json`.
  The K0-off "before" points are measured 2026-09-07: N=2 request peaks
  21.307 GiB at context 1024 and 21.578 GiB at context 1536 (BF16 KV,
  post-scratch-cap; pre-cap 1536 was 22.574). Evidence:
  `results/2026-09-07-rx7900xtx-capacity-k0-mtp-off.json`. The MTP-armed
  opt_in mode additionally cannot reach readiness: its startup scratch-probe
  engine-service child hangs at width 2 (`STARTUP_SCRATCH_PROBE ... engine
  service command timed out`) — same subsystem as the enabled KeyError. The
  K0->K resident delta and the opt_in K0 state are blocked by that MTP-serving
  warmup bug — owned by the MTP campaign; re-measure with the probe pair when
  it serves.
- [x] Deduplicate actual weight aliases and release conversion staging safely.
  Count resident payloads, not GGUF size, as the device-weight baseline.
  Done 2026-09-06: the planned residency census reports 851 logical tensors,
  0 aliases, 15.995 GiB planned resident vs 15.932 GiB GGUF file bytes, and a
  malloc-site attribution run mapped every >=64 MiB live load allocation to a
  planned weight (the 32 x 69.73 MiB group is the `ffn_down` Q6_K_T16 set) or
  the 350.81 MiB prefill-scratch liveness arena — zero retained conversion
  staging. Baseline: resident payload bytes. Evidence:
  `results/2026-09-06-rx7900xtx-capacity-packet2-workspace-ledger.json`.
- [x] Right-size shared workspace and prefill scratch by supported execution
  shapes; reuse sequential scratch without merging per-request GDN state.
  Preserve graph pointer lifetimes and overlap constraints.
  Workspace done 2026-09-06: lease + union geometry follow the real serving
  capacity instead of the 8-slot floor; measured on the XTX at BF16/3072/N=1:
  load 19.742 -> 18.429 GiB, request peak 23.328 -> 20.998 GiB (-2.330, -10.0%),
  lease 96 -> 12 pages, transients 3.586 -> 2.569 GiB; INT8 fp32 route verified
  serving under the smaller lease (peak 20.841 GiB). Prefill scratch rows done
  2026-09-06: a gfx1100 geometry policy (`GGUF_DENSE_PREFILL_SCRATCH_ROW_CAP_
  POLICIES`, dense H5120 Q4_K_M, min_capacity 1024 -> max_rows 1024) bounds the
  session scratch that previously scaled with the declared context (~1 MiB per
  declared token vs the 64 KiB/token BF16 KV payload); measured on the XTX:
  BF16 3,840 request peak 21.820 -> 19.375 GiB (-11.2%), BF16 declared-context
  boundary 3,840 -> 40,960 tokens (10.7x), INT8 fp32 5,120-class passes at
  19.290 GiB, d512 solid concurrency N=3 -> N=4; exactness: a 2,650-token
  greedy server request is token-identical capped vs uncapped (the 1,024-row
  chunks route through the registered exact F16/rocBLAS pair producer;
  full production-profile KL/top-1 gates remain a Packet 6 item). Bucket
  bounding and retirement remain open; the 512/128 arm (declared 768) is
  unchanged and still scratch-dominated below the 1,024-row threshold.
- [x] Bound cached graph/workspace buckets and safe retirement. Test wide-to-C1
  and C1-to-wide histories. Report reusable pool capacity separately from memory
  returned to the device allocator. In-session interleave stability is
  enforced under the capacity-honest union (2026-09-06: repaired the stale
  interleave fake that predated the serving-cap right-size — the fake now
  declares `max_batch_size`, and a contract test pins the first packed
  allocation to the declared cap; failure window bisected to b753495b4).
  Done 2026-09-07: the device pool stats now carry cumulative
  `retired_pages`/`retired_bytes` (pages returned to the device allocator,
  distinct from `free_pages` reusable capacity), exposed as
  `hipengine_kv_pool_retired_pages_total`/`_bytes_total`; and a one-session
  history probe drove wide(1,535 tok) <-> C1(63 tok) alternation on the XTX
  (BF16, N=2, ctx 2048): all five phases validate exactly, whole-card usage is
  flat at 22,224 MiB across every shape change (zero creep), the default pool
  is born at full serving capacity (32 pages = 512 MiB, grow 0 / shrink 0 /
  retired 0), parking 16 pages (256 MiB) as reusable capacity with 16 pages
  pinned by resident sessions — retirement is grow-scenario machinery
  (CPU-verified) and the default path never grows. Evidence:
  `results/2026-09-07-rx7900xtx-capacity-pool-history-wide-c1.json`.
- [ ] Provide a true AR-only configuration. Optional lazy MTP activation must
  reserve its full peak before mutation; unloading must not free borrowed or
  in-flight graph assets. Do not load/unload weights per decode cycle.

### Packet 3 — Qualify smaller weights and compact INT8

- [x] Compare actual resident and load-peak `Q4_K_M`/`Q4_K_S` bytes, throughput
  and quality. Inventory MTP tensor availability and preserve artifact identity.
  Bytes done 2026-09-06: Q4_K_S saves 0.918 GiB of file bytes but parks more
  device memory (resident +0.176 GiB, request peak +2.061 GiB pre-scratch-cap)
  because it stores 8 ffn_down, 3 attn_qkv and 1 attn_v as dense BF16 where
  Q4_K_M stores Q6_K_T16 layouts. Throughput and MTP availability done
  2026-09-07: same-session matched pairs on the XTX at 512/128 (BF16 KV,
  persistent session, graph-replay decode) — decode ties (~34.0 tok/s both
  routes) while Q4_K_S prefills 38.4% slower on the shipping AR route (602.5
  vs 978.7 tok/s; default route 320.1 vs 963.3), and both files carry the
  identical 4 `blk.64.nextn` tensors so MTP availability is unchanged.
  Evidence: `results/2026-09-07-rx7900xtx-capacity-q4ks-q4km-throughput.json`
  and `results/2026-09-06-rx7900xtx-capacity-q4ks-vs-q4km-bytes.json`.
  Q4_K_M remains the default; per-quant quality gates stay with the
  teacher-gating item (no reuse of Q4_K_M/gfx1151 evidence for Q4_K_S/XTX).
- [ ] Prove effective compact INT8 has no persistent BF16 shadow. Bound and
  reuse prefill oracles; a smaller steady cache with an oversized prefill peak
  does not fit. Account for scales, pool planes and graph scratch.
- [x] Inspect existing `qwen38_int8_batch_decode_gate.py` and service selection.
  Separate serial compact residency from native no-mirror batched prefill,
  decode, graphs and MTP. Coordinate missing integration with the INT8 campaign.
  Inspection and gate run recorded 2026-09-07. Serial compact residency is
  qualified: IKV-C1 no-mirror audits report zero persistent BF16 payload bytes
  and the 2026-09-06 XTX probe points are serial c=1 per-token/head INT8
  (FP32 scales) at 2,048/3,072/4,096 ok (request peaks 21.35/22.60/23.75 GiB)
  and 5,120 OOM at startup. Batched decode (IKV-C2) is now model-level proven
  on the XTX: the gate passes exactly (token-exact, max KL 0.0, top-1 1.0,
  zero hidden and zero state/KV-scale mismatches over 4 rows x 4 steps vs
  independent c1 oracles) with every step routed `kv_live_spans_int8_batch`,
  physical_rows 4 and zero host row iterations under the
  `per_token_head_gqa_splitk_gate_bf16_batch_strided_spans` variant; the
  artifact capability still admits c1, so this is a pre-promotion gate and
  capability promotion belongs to the INT8 campaign
  (`results/2026-09-07-rx7900xtx-ikv-c2-batch-decode-gate.json`). Not
  integrated: batched prefill stays IKV-C3-blocked (the gate prefetches
  prompts through independent scalar sessions by design); packed decode
  graphs hard-require BF16 KV (`gguf_packed_decode_graph` raises
  NotImplementedError), so all INT8 decode is eager; c=1 decode graphs admit
  INT8 only in the tail4-Hadamard-group32 layout, not the per-token/head
  FP32-scale layout the probe points use; and the MTP native spec target
  graph N1 requires BF16 KV, so INT8+MTP is unproven on both axes. Capacity
  consequence: INT8 buys residency at serial widths today; batched-eager
  decode is proven but graph and prefill ownership must land before INT8
  batched serving is throughput-eligible.
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
  Refined 2026-09-06 at 256-token page-aligned steps after the workspace
  right-size, with clean ownership on every point: BF16 last pass 3,840
  (3,328/3,584/3,840 pass at 21.324/21.525/21.820 GiB; 4,096 still fails eager
  warmup); qualified INT8 fp32 last pass 4,864 (4,352/4,608/4,864 pass at
  21.841/21.843/21.849 GiB; 5,120 still first fails). 4K+ totals are
  card-infeasible at this model size; 8K-128K rows are policy-capped
  unreachable, not measured ceilings. Artifact:
  `results/2026-09-06-rx7900xtx-capacity-c1-boundary-refinement.json`.
  Superseded same day by the prefill scratch row cap (that scratch, not KV
  bytes, was the binding constraint below 4K): BF16 last pass **40,960**
  declared tokens (4,096/8,192/16,384/32,768 pass at 19.379/19.955/21.098/
  23.162 GiB; 40,960 razor-thin at 23.973; 45,056 first fails startup OOM at
  23.943 GiB); qualified INT8 fp32 reaches **15,872** (20.349 GiB) before a
  pre-existing functional blocker — request-time HIP error 1 (invalid
  argument) in (15,872, 16,128], reproduced uncapped on the base tree — not a
  memory ceiling. Artifact:
  `results/2026-09-06-rx7900xtx-capacity-scratch-row-cap.json`.
- [ ] Budget D=24/128/512 and reduce L to leave output/lookahead space. Include
  natural long-output cases; an early EOS proves only the work actually done.
- [ ] Concurrent baseline: 512 prompt/128 output at N=C=1 through 8, including
  the disputed c5/c6 boundary. Increase per-request budgets through 1K/2K/4K/
  8K/16K where possible. Add N=8/C=1, mixed lengths, gradual fill and survivors.
  Measured 2026-09-06 on the XTX (BF16 KV, page-aligned 768-token context,
  one cold server per point, repaired probe, capacity-honest lease): N=1/2/4/
  5/6 pass at 18.748/20.859/22.822/23.357/23.676 GiB request peaks with exact
  accounting and clean ownership; N=7/8 fail eager warmup (STARTUP_SCRATCH_
  PROBE OOM, sampled failure peaks 23.972/23.955 GiB). The c5/c6 boundary is
  resolved: last pass **N=6**, per-request slope ~0.99 GiB — superseding the
  withdrawn "four to five" estimate, which predated the repaired probe and
  the workspace right-size. N=3 not separately measured (monotone between
  N=2 and N=4). Wider per-request budgets are bounded by the C1 boundaries
  (after the scratch row cap: BF16 40,960 / INT8 fp32 15,872 functional-
  blocked). At the D=512 budget (512-token prompts, page-aligned 1,280
  context) the scratch row cap moves the solid last pass from **N=3** to
  **N=4** (23.821 GiB, clean): N=5 is marginal (serves the horizon; idle
  residue 159 MiB vs the 128 MiB gate, reproduced), N=6 fails warmup OOM at
  23.976 GiB; N=3 improved 22.692 -> 22.129 GiB (~-0.19 GiB/session). The
  512/128 arm (declared 768, below the 1,024-row cap threshold) is unchanged:
  last pass **N=6**, per-request slope ~0.99 GiB, N=7/8 fail eager warmup
  (STARTUP_SCRATCH_PROBE OOM, sampled failure peaks 23.972/23.955 GiB) — its
  slope is still prefill-scratch-dominated and is the recorded next lever
  (sub-1,024 row cap or served-chunk sizing, gated on the GDN 512-row chunk
  qualification). Artifact:
  `results/2026-09-06-rx7900xtx-capacity-scratch-row-cap.json`. Artifact:
  `results/2026-09-06-rx7900xtx-capacity-concurrency-d512.json`. Artifact:
  `results/2026-09-06-rx7900xtx-capacity-concurrency-512-128.json`.
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
