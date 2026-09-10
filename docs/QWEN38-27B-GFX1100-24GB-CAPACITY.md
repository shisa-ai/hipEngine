# Qwen3.8-27B: capacity on a 24 GB RX 7900 XTX

Status: measurement and optimization plan with scoped offline DMS INT8
integration evidence. General production-serving qualification is not established.

## Packed slot-local prefill: the per-slab whole-history KV import is dead work, not the long-prompt bottleneck — 2026-09-10 UTC

`_prefill_batch_native_single_slab` called `_sync_packed_decode_initial_state`
once per slab, and each call imported every session's whole prior KV history into
packed storage — even though slot-local attention reads request-owned KV and
the end-of-slab scatter already skipped packed KV on that route. The import is
now removed: `_sync_packed_decode_initial_state` takes `copy_kv=True` (mirroring
`_scatter_packed_decode_state`), the full-attention segment copies are skipped
when `copy_kv=False` while the Conv/GDN linear-state import is preserved, and
the single-slab call site passes `copy_kv=not slot_local_full_prefill` after
final route resolution. Non-slot-local fallbacks and the packed decode rounds
keep the default import.

Measured directly with the new census harness
([`gguf_packed_kv_import_profile.py`](../scripts/gguf_packed_kv_import_profile.py)),
W7900 GPU0, shipping selectors, `int8_per_token_head` + FP32 scales,
`max_sequence_length` 16,384, deterministic varied prompt:

| rows | chunks | import MiB (pre → post) | import `memcpy`s (pre → post) | wall s (pre → post) |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 1 | 0 → 0 | 0 → 0 | 1.852 → 1.839 |
| 2,048 | 2 | 32.5 → 0 | 256 → 0 | 4.547 → 4.530 |
| 4,096 | 4 | 195.0 → 0 | 1,536 → 0 | 18.033 → 18.015 |
| 8,192 | 8 | 910.0 → 0 | 7,168 → 0 | 121.770 → 121.862 |

Two conclusions:

1. **The removal is exact.** Final-logit sha256 and all eight greedy decode IDs
are identical pre/post at every shape; the Conv/GDN import (two fused copies
per slab) and the scatter side (already zero on this route) are unchanged.
2. **The earlier attribution is withdrawn.** The 2026-09-10 per-layer-oracle
unit recorded the per-slab import as "the real cause of the slow long-prompt
server path". Disproven at these shapes: with the import fully removed, wall
time does not move at 1/2/4/8 chunks. The import was 910 MiB of dead copy work
at 8 chunks — worth removing — but the multi-chunk slowdown lives elsewhere.

**The slowdown is now attributed (same day, kernel trace).** With the
cache-only `rocprofv3` recipe,
[`qwen35_paged_full_attn_prefill_gqa_gate_bf16_kernel`](../hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip)
is **91% of prefill device time at 8 chunks** (110.989 of 121.385 s), while
every other kernel family stays flat at ~1.25–1.34 s per slab. Its per-launch
cost grows ~quadratically with the slab's end context — 33.7 ms at ctx 1,024
→ 3,042.1 ms at ctx 8,192 (~90× for 8× context) where the algorithmic
requirement is linear — and the route is 99.6% device-bound. The kernel runs
one block per (q\_head, query row) that serially walks the whole visible
context, and its score buffer is dynamic LDS sized by the whole context
(`(max_context_len + threads + head_dim) × 4` bytes: ~4.6 KB at ctx 1,024 →
~33.5 KB at ctx 8,192 per block), so occupancy collapses as the slab's end
context grows. Escape hatches: AOTriton admission on transient-oracle layers
(built, default OFF, gate to re-run on the corrected route) or the WMMA
score-GEMM structure the scalar parent's bulk path already uses. The
layer-outer packed executor would fix the oracle memory but not this — every
schedule still attends over the full history per chunk.

[`Attribution evidence`](../benchmarks/results/2026-09-10-w7900-int8-packed-paged-attn-attribution.json)

Evidence caveats recorded with the artifact: the sync-neutralized CPU pass is
an upper bound on host enqueue (a full command queue still blocks), so it
does not prove host- or device-bound; and per-chunk cost growth is not
established linear (4 → 8 chunks costs 6.8× wall). Two `rocprofv3` attribution
attempts stalled in `hipcc --version` compiler discovery under the profiler
and were stopped; the working recipe (prewarm, `HIPENGINE_COMPILER_VERSION_FILE`,
`HIPENGINE_REQUIRE_CACHED_BUILD=1`) is now in
[`HARNESSES.md`](../benchmarks/HARNESSES.md) and the probe's docstring.

[`Removal evidence`](../benchmarks/results/2026-09-10-w7900-int8-packed-kv-import-removal.json)

## P1 defect: packed INT8 prefill shares one BF16 oracle across layers — 2026-09-10 UTC

The packed slot-local INT8 prefill route produces **wrong output for any prompt
that spans more than one prefill chunk** (>1,024 rows on the dense H5120 Q4_K_M
geometry). `int8_direct` sessions are forced onto this entry unconditionally by
`_gguf_single_row_block_table_prefill_required`, so this is the shipping server
path for every INT8 KV request above 1,024 prompt tokens. Single-chunk prompts
are unaffected.

`_int8_prefill_oracle_cache_for_layer` keys the BF16 oracle pair `-1` — one
shared pair for all INT8 layers — whenever the lifetime plan mode is
`layer_outer_shared_oracle`. That function's own docstring states the invariant:
chunk-outer prefill "keeps one full-length pair per INT8 layer", and only a
layer-outer plan "safely reuses one pair after each layer completes". But
`_prefill_batch_native_impl` chunks the prompt by `_bulk_prefill_scratch.rows`
and `_prefill_batch_native_single_slab` iterates layers *inside* each chunk,
while `_plan_gguf_int8_prefill_lifetime` still selects the shared mode purely
from a memory comparison (`use_shared = projected_delta < 0`). Plan and executor
disagree; from the second chunk onward every layer attends over the previous
chunk's *last* layer's K/V.

Reproduced on the W7900 (GPU0), gfx1100, `int8_per_token_head` + FP32 scales,
`max_sequence_length` 16384, shipping selectors (`use_wmma_prefill`,
`use_gemv_decode`), deterministic varied prompt, 8 greedy IDs, via
[`gguf_prefill_route_ab.py`](../scripts/gguf_prefill_route_ab.py):

| Prompt rows | Chunks | Oracle | Scalar bulk IDs | Packed IDs | Agree |
| ---: | ---: | --- | --- | --- | :-: |
| 1,024 | 1 | shared (shipped) | `[62,198,197,197]` | `[62,198,197,197]` | yes |
| 2,048 | 2 | shared (shipped) | `[198,197,197,1]` | `[14,198,248046,198]` | **no** |
| 2,048 | 2 | per-layer (patched control) | `[198,197,197,1]` | `[198,197,197,1]` | yes |

The scalar arm is the control and is unchanged by the oracle variant (742.28 vs
749.29 tok/s, identical IDs) because scalar bulk prefill is layer-outer. Packed
prefill measures **449.16 tok/s in both oracle variants**, so correcting the
lifetime is speed-neutral at this shape and costs memory only.

Two earlier conclusions are withdrawn. The packed-versus-scalar divergence is
**not** the "different GDN state-capture arithmetic" recorded at
`hipengine/generation/qwen35_gguf.py:7406-7408`: with per-layer oracles the two
entry points agree exactly. And absolute prefill rates previously taken from a
harness that omitted `use_wmma_prefill`/`use_gemv_decode` were roughly 6x low;
with the shipping selectors the oracle route reaches 742-749 tok/s, consistent
with the retained XTX oracle-route control of 761.58 tok/s at 8,192
([`blocked artifact`](../benchmarks/results/2026-09-09-rx7900xtx-gguf-int8-direct-prefill-blocked.json)),
so the BF16 oracle is close to free and is not the cause of any slow path.

Fix options, none free:

1. **Layer-outer packed executor.** Invert the packed loops for the slot-local
   INT8 case so all chunks of a layer complete before the next layer, matching
   the scalar bulk parent. Preserves the memory model — the shared plan already
   budgets `required_hidden_capacity = positions` precisely so the executor can
   be layer-outer — and is the recommended target.
2. **Per-layer oracles whenever the executor is chunk-outer.** Correct and
   measured speed-neutral, but the oracle cost moves from ~4 KiB/token to
   ~64 KiB/token on this geometry (16 pairs), which pulls the context ceiling
   down hard at long contexts and is a non-starter above ~16K.
3. **Repopulate the shared oracle per chunk from the retained INT8 store.**
   Reintroduces a dequantized read on the strict path, so it needs its own
   numerics gate.

Consequence for open work: the AOTriton slot-local admission gate
([`gate artifact`](../benchmarks/results/2026-09-09-w7900-int8-slot-local-aotriton-gate.json))
used a sound before/after method and its flag is default OFF, so nothing shipped
changed, but both arms shared the same corrupted multi-chunk history. Its
1.97e-05 mean KL does not qualify the route; re-run after the lifetime is fixed
and against the shipping selectors. Two review findings remain open and
unprofiled: `_prefill_batch_native_single_slab` calls
`_sync_packed_decode_initial_state` once per slab (copying positions
`0..session.position` into packed storage, then clearing the reuse identity so
the next slab re-imports — roughly 15.5 full-prompt history copies at 32K in 1K
slabs), and `_int8_prefill_oracle_capacity_positions` sizes the oracle by
`backing_pages * block_size`, the whole pool, rather than the request context.

[`Defect evidence`](../benchmarks/results/2026-09-10-w7900-int8-packed-shared-oracle-defect.json)

Follow-up (same day): host-policy regression tests now pin the fix
(`tests/test_gguf_int8_prefill_oracle_per_layer.py` — 15 tests, 8 of them
failing on the pre-fix code when the runner hunk of the fix commit is
reverse-applied): shared-plan keying with and without the per-layer override,
ownership set on every session before the first slab (including tail-chunk
plans), the `finally` clearing the flag and releasing the buffers when a slab
raises, multi-chunk → single-chunk reuse, and the pool-backed oracle capacity
invariant (`backing_pages * block_size`, the reason the oracle cannot simply be
shrunk to the prompt length). The published **INT8 KV server ceiling of
54,272 is marked unqualified** in the root README long-context table and the
benchmark rollup ladder: it was measured on the pre-fix route, and the fixed
route's transient per-layer oracle cost makes it optimistic. A fresh
server-route ceiling is required (probe first, ladder only to confirm).

## Hidden-plane alias adoption and the 232,448-token ladder — 2026-09-08 UTC

The route-scoped single-plane hidden stream for layer_outer prefill was
adopted (307e7633f, default on, `HIPENGINE_LAYER_OUTER_HIDDEN_ALIAS=0` rolls
back) after a GPU-side A/B at 73,728 tokens on a clean tree: byte-identical
decode logits on all eight steps, prefill 455.4 s vs 454.0 s, whole-card
peak 19.265 vs 19.972 GiB (-723 MiB = exactly one BF16 hidden plane at
73,984 positions).

The alias-era ladder (fresh process per point, GPU1, source clean at the
adoption commit or its docs-only descendant):

| Prompt tokens / decode appends | Outcome | Sampled whole-card peak | Headroom |
| --- | --- | ---: | ---: |
| 200,704 / 8 | Pass | 23.318 GiB | 682.0 MiB |
| 216,064 / 8 | Pass | 23.752 GiB | 238.0 MiB |
| 220,672 / 8 | Pass | 23.883 GiB | 104.0 MiB |
| 224,256 / 8 | Pass | 23.681 GiB | 310.0 MiB |
| **232,448 / 8** | Pass | 23.924 GiB | 62.0 MiB |
| 234,496 / 8 | Timeout at the wrapper's 2700 s bound, 116.1 MiB headroom at kill; unqualified, may fit | 23.871 GiB | — |

The largest observed passing prompt is **232,448** (+199.0% over the
73,728 campaign baseline, +35.1% over the two-plane 172,288). No alias-era
OOM is established; the 176,128 OOM was measured on the pre-alias two-plane
route and does not bound this route. Intermediate sizes are unqualified.
No maximum, quality, long-output or serving claim is made; DMS discards
history, so fitting a longer prompt does not establish full-context quality
(the quality bar and ladder are defined in
[`DMS-ANALYSIS.md`](DMS-ANALYSIS.md)).

Route-equality evidence: at 72K the eager dense_pool and layer_outer routes
produce identical decode outputs and identical packed stores, verified
pre-merge same-card (GPU1) and post-merge cross-card (dense_pool on the
W7900 vs layer_outer on the XTX), the latter also establishing cross-card
gfx1100 determinism at that width. Known follow-up: the layer_outer route
fails closed on the 48 GB W7900 (device-memory-class layer chunks exceed the
row-capped bulk-scratch buffers; `for_chunk` positions copy rejects the
oversized upload), which blocks the W7900 no-evict reference arm of the
quality ladder until fixed.

[`Hidden-alias ladder evidence`](../benchmarks/results/2026-09-08-rx7900xtx-dms-int8-hidden-alias-ladder.json)

## Merged-lane layer_outer capacity — 2026-09-08 UTC

The GPU0 speed campaign (90b9aa510) was merged into the capacity lane
(eb3242a7a; the layer_outer prefill change itself committed as 22ded5522).
The merge is memory-neutral at 139,264 tokens (whole-card peak moved 57,344
bytes) with identical greedy tokens across the merge; single-session
diagnostic decode timing at that context moved 469.09 -> 47.10 ms/step
mean (performance_claim false; no throughput claim is made).

The headroom ladder on the merged tree (same fresh-process/20 ms
protocol, GPU1 idle baseline verified per run, source clean at eb3242a7a):

| Prompt tokens / decode appends | Outcome | Sampled whole-card peak | Headroom |
| --- | --- | ---: | ---: |
| 139,264 / 8 | Pass | 22.740 GiB | 1,274.0 MiB |
| 168,704 / 8 | Pass | 23.670 GiB | 322.0 MiB |
| 172,288 / 8 | Pass | 23.937 GiB | 48.2 MiB |
| 176,128 / 8 | OOM (HIP error 2, compact layer v_slot allocation in _ensure_layer; full rollback, baseline returned) | 23.969 GiB before failure | — |

The largest observed passing prompt is now **172,288** (+133.6% over the
73,728 campaign baseline, +50.2% over the pre-merge layer_outer boundary).
176,128 is the smallest tested OOM; intermediate sizes are unqualified;
172,800 was cancelled by operator choice before completion and is not
evidence of either outcome. No maximum, quality, long-output or serving
claim is made; DMS discards history, so fitting a longer prompt does not
establish full-context quality.

The marginal slope is not constant — the compact store grows sublinearly
because the DMS compression ratio improves with context (1.889 observed at
139,264): 45,346 B/token (73,728-139,264), 33,899 B/token (139,264-168,704),
80,191 B/token (168,704-172,288, the near-full tail). Slope-based forecasts
are estimates, not capacity claims.

Tracked ledger at 139,264 (bytes): weights 16,401,463,296 + compact store
2,501,602,448 + one-layer BF16 oracle 571,473,920 + two hidden planes
2x1,428,684,800 + token buffer 1,116,160 = 22,333,025,424, plus an
unaudited 1,660,667,912 residual (bulk prefill scratch, split-K partials,
decision/collector planes) = tracked peak 23,993,693,336. The
whole-card-minus-tracked delta (423,340,904 B) is recorded without
attribution.

Identified but not validated: the geometry's liveness policy requires
>=4,096 scratch rows for single-plane hidden aliasing while the row-cap
policy clamps scratch rows to 1,024, so the layer_outer route always
allocates two full-capacity BF16 hidden planes. One plane is
1,428,684,800 B (1.3306 GiB) at 139,520 positions and also removes
10,240 B/token from the marginal slope. The threshold is geometry-wide,
not DMS-scoped: any enablement must be route-scoped to layer_outer and
verified with GPU-side evidence (greedy-token equality, per-layer
hidden/logit equality, allocation ownership, partial/tail chunks,
teardown) — CPU-codec equality alone cannot establish GPU hidden-buffer
liveness. Linear-model ceiling band 215,000-240,000 tokens; estimate only.

[`Merged-lane capacity evidence`](../benchmarks/results/2026-09-08-rx7900xtx-dms-int8-merged-lane-capacity.json)
[`Point-check wrapper (byte-identical to the campaign's monitors' sha256)`](../scripts/xtx_dms_capacity_point_check.py)
[`Artifact consolidator`](../scripts/publish_merged_lane_capacity.py)

## Post-memory-cut ladder — 2026-09-07 UTC

Four GPU-side memory cuts landed on the C1 DMS INT8 route (private-C1
placement in the probe, demand-driven split-K partials, bulk prefill
workspace release at DMS finalize, and layerwise compact pack with
per-layer dense release; commits `6b46dcfdb`, `c85aaeeb3`, `9862407d2`,
each with bit-exact decode logits against the pre-change route at 72K).
At the unchanged 72K shape they cut the probe's whole-card peak from
23.949 GiB to 21.467 GiB (−2.42 GiB; tracked 23.186 → 21.123 GiB) with
no wall-clock change.

The ladder after the cuts (same fresh-process/20 ms/600 s protocol):

| Prompt tokens / decode appends | Outcome | Sampled whole-card peak | Headroom |
| --- | --- | ---: | ---: |
| 98,304 / 8 (96K) | Pass | 22.685 GiB | 993.0 MiB |
| 102,400 / 8 (100K) | Pass | 22.933 GiB | 733.0 MiB |
| 106,496 / 8 (104K) | Pass | 23.187 GiB | 473.0 MiB |
| 110,592 / 8 (108K) | Pass | 23.441 GiB | 211.0 MiB |
| 114,688 / 8 (112K) | Pass | 23.920 GiB | 38.8 MiB |
| 118,784 / 8 (116K) | OOM (HIP error 2, 79.7 s in, early dense allocation) | 19.737 GiB before failure | — |

The largest observed passing prompt is now **114,688** (+55.56% over the
pre-cut 73,728); 118,784 is the smallest tested OOM and intermediate sizes
are unqualified. 114,688 matches the August no-mirror INT8 singleton
route's repeated-context size on the same model file. No maximum, quality,
long-output or serving claim is made; DMS discards history, so fitting a
longer prompt does not establish full-context quality.
[`Post-cut ladder evidence`](../benchmarks/results/2026-09-07-rx7900xtx-dms-int8-postcuts-ladder.json).

## Requested 75K / 74K / 72K ladder — 2026-09-07 UTC (pre-cut)

| Prompt tokens / decode appends | Outcome | Sampled whole-card peak |
| --- | --- | ---: |
| 76,800 / 8 (75K) | OOM allocating compact eviction flags after prefill | 23.961 GiB before failure |
| 75,776 / 8 (74K) | OOM allocating compact V payload after prefill | 23.938 GiB before failure |
| 73,728 / 8 (72K) | Pass; eight finite decode steps and zero tracked allocations after close | 23.938 GiB |

The passing point has 47.42 MiB sampled headroom; tracked peak is 23.186 GiB.
This reaches the requested 23.9 GiB+ whole-card target. All three fresh
processes returned GPU1 to baseline. The largest observed passing prompt
was 73,728 before the same-day memory cuts above moved it to 114,688;
75,776 was the smallest tested OOM on the pre-cut route. No maximum,
long-output stability, new replay/teacher gate or general-serving
qualification is claimed.
[`Requested ladder evidence`](../benchmarks/results/2026-09-07-rx7900xtx-dms-int8-requested-backoff.json).
The prior near-limit section below records the earlier measurements.

### Historical singleton comparison

The August 15/16 higher-context XTX results used the **same Qwen3.8-27B
Q4_K_M file**, not Q4_K_S. The controls audit matches sampled file and tensor
inventory fingerprints: four natural 114,688-token requests passed at
23.323 GiB, and one 129,024-token request passed at 23.963 GiB, with four
output tokens. Those used a long no-mirror INT8 C1 route, MTP/prefix off and
an explicit unverified-long gate; their long-context quality scope was bounded.
This model did not always have a roughly 72K context limit.

The current DMS route first holds BF16 full-history prefill storage and then
allocates the compact INT8 destination before releasing dense storage. The
observed OOMs occur during that overlap. Historical higher execution capacity
is real, but these different owners/protocols do not isolate a particular
regression. Reproducing the historical no-mirror route on current source is
the relevant next comparison, not more DMS boundary sweeps.

The 35B-A3B also has intrinsically lower full-attention KV growth. Local GGUF
metadata yields the following target-model geometry (excluding NextN):

| Model | Full-attention layers | KV heads | Head dimension | BF16 KV bytes/token | INT8 + FP32 scales bytes/token |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 16 | 4 | 256 | 65,536 (64 KiB) | 33,280 (32.5 KiB) |
| Qwen3.6-35B-A3B | 10 | 2 | 256 | 20,480 (20 KiB) | 10,400 (10.15625 KiB) |

At the same codec the dense model needs **3.2x** the full-attention KV payload
per retained token. This is an attention-geometry difference, not a general
dense-versus-MoE rule or a consequence of only 3B active experts. It excludes
weights, recurrent state, allocator overhead and transient prefill workspaces.
Q4_K_S does not explain the old context advantage; the matched XTX 3K
allocation audit actually found its runtime residency higher than Q4_K_M.
Sources: [identity and historical controls](../benchmarks/results/2026-09-06-rx7900xtx-capacity-packet0-controls.json),
[Q4_K_S/M allocation comparison](../benchmarks/results/2026-09-06-rx7900xtx-capacity-q4ks-vs-q4km-bytes.json),
and local metadata captured in the requested-ladder artifact.

## Near-limit DMS INT8 capacity — 2026-09-07 UTC

Eager resident C1 now passes **72,960 prompt tokens plus eight decode appends**
on the RX 7900 XTX, with finite logits and zero tracked allocations after close.
This extends observed prompt coverage from 65,536 by 11.33%; it is a new
measurement, not a kernel optimization or a proven maximum.

| Prompt tokens / decode appends | Outcome | Sampled whole-card peak | Tracked allocation peak |
| --- | --- | ---: | ---: |
| 65,536 / 8 | Pass; eight logit hashes match prior exact replay | 23.377 GiB | 22.539 GiB |
| 72,704 / 8 | Pass; finite logits and clean drain | 23.893 GiB | 23.105 GiB |
| 72,960 / 8 | Pass; finite logits and clean drain | 23.893 GiB | 23.125 GiB |
| 77,824 / 8 | OOM before decode | 23.952 GiB before failure | Not captured |
| 81,920 / 8 | OOM before decode | 23.946 GiB before failure | Not captured |
| 82,944 / 8 | OOM before decode | 23.967 GiB before failure | Not captured |
| 83,968 / 8 | OOM before decode | 23.940 GiB before failure | Not captured |

The 72,960-token point peaks at 25,655,005,184 whole-card bytes, leaving
98,021,376 bytes (93.48 MiB) of sampled headroom out of 23.984375 GiB total.
That is approximately 23.9 GiB, but **7.08 MiB below the literal 23.9 GiB
target**. Testing stopped there to avoid further multi-minute prefills for
that small difference. Context after eight appends is 72,968 tokens.
Intermediate sizes up to the smallest tested OOM, 77,824, are not qualified.

Each point ran in a fresh process, with a 600-second bound and 20 ms sysfs
sampling of PCI `0000:10:00.0`, unique ID `cc4d02090dc9c3ff`. All four OOMs
were HIP error 2 while allocating compact V payloads after dense prefill,
before dense storage could be released. Each process returned the card to
baseline; same-process allocation-failure recovery was not tested.

The initial allocator-only estimate was too optimistic: the matched 64K
control measures 23.377 GiB whole-card versus 22.539 GiB tracked. These are
different accounting domains, not interchangeable capacity budgets.
Sampled peaks remain lower bounds, and failed-point peaks do not measure the
bytes needed to finish. This is a near-limit execution fit, not an operating
reserve recommendation, new numerical qualification or serving promotion.
No new exact replay was run at the larger points; the existing 64K replay and
post-fix numerical evidence remain separate. Dense boundaries are unchanged.

Commands, identities, raw hashes, failure stages and the sampling wrapper:
[`DMS INT8 edge evidence`](../benchmarks/results/2026-09-07-rx7900xtx-dms-int8-edge-capacity.json).
Measured source is clean `bc7d7c7de`; the DMS runtime, cache, kernels, core and
probe are unchanged in the later integrated main `ae2e02754`.

## Bounded correctness and capacity closeout — 2026-09-07

The user narrowed closeout to implementation correctness and observed capacity,
without exhaustive benchmarking or further dense boundary probes. On the clean
merged source `f955ea2d8` (including main `ccf67ca6a`), GPU1 RX 7900 XTX:

| DMS INT8 route | Workload | Result |
| --- | --- | --- |
| Eager resident C1 | 65,536 prompt tokens + eight decode appends | 8/8 independent replay steps byte-exact; finite logits; 24,200,876,888 bytes (22.54 GiB) tracked device peak |
| Interleaved resident C2 | 8,192/16,384 prompts, four rounds, cancel first request after two | 6/6 executed steps byte-exact against C1 replay; survivor continues; no cycle errors |

Both processes return to zero tracked allocations. The 64K point is an
**execution fit**, not a proven maximum or a recommended operating reserve;
65,544 logical tokens are present after the eight appends. The peak includes
prefill and independent replay and is not a sampled whole-card peak. Existing
dense BF16 40,960 and dense INT8 54,272 observed passes were not rerun.

Affected GPU/CPU tests pass after repairing one stale lazy-server mock;
572 server tests, 183 MTP interface tests and the benchmark documentation
checks pass. Existing post-repair numerical and repeated-refill evidence below
is reused. No throughput sweep or new performance ratio is claimed.

This completes the user-requested bounded correctness/capacity check for merge.
It does not close the broader serving-qualification requirements of task #20:
DMS INT8 stays explicit offline evaluation, BF16 fallback stays available,
and packed/larger-C execution, full-session MTP rollback, calibrated
production-profile and train-disjoint task certification remain deferred.
The replay harness uses identical valid token transitions, seeded from the
last prompt token; it is not a natural-response quality test.
[`Bounded merge evidence`](../benchmarks/results/2026-09-07-rx7900xtx-dms-int8-bounded-merge-gate.json).

## INT8 completion audit — 2026-09-07

This audit updates the historical packet notes below. All INT8 work is assigned
here; there is no external dense or DMS INT8 owner.

- Dense native-prefill repair `8e0d8eae1` fixes the confirmed shared-memory
  overflow. Cold-server INT8/FP32 C1 requests pass at 16,128 / 32,768 / 49,152 /
  54,272 declared tokens, peaking at 20.357 / 21.919 / 23.484 / 23.972 GiB.
  Each uses context minus 17 prompt tokens, 16 outputs and one reserved token,
  MTP off, prefix cache off, and clean post-request ownership.
- **54,272 is the highest demonstrated context, not a proven maximum.**
  Sampled headroom is 12.7 MiB. The user stopped the adjacent 54,528 probe to
  prioritize DMS INT8; it has no pass/fail verdict. Prior BF16 evidence on the
  same card reaches 40,960, but no fresh matched quality comparison was run.
- Compact INT8 K/V, FP32 per-token/head scales, pack/compaction, append,
  split-K attention and resident `dms_backend_factory` integration are
  implemented. BF16 fallback and fail-closed ordinary INT8 admission are
  preserved; offline evaluation creates no `DMSCodecQualification`.
- A shared-memory softmax race was repaired in `eabb08fc2`. Only post-repair
  INT8 results below supply final numerical/replay evidence. Earlier numerical
  passes, including the 768-token smoke, do not qualify the repaired path.
- Post-repair affected pytest: 48 passed. Cached `rocprofv3` fixture run:
  five passed, with INT8 pack, append and attention launches confirmed.
  Device/backend fixtures cover CPU oracles, overflow, exact payload/scale
  restoration, eviction and neighbor isolation.
- Resident C1 at 8,192 prompt tokens passes one-step independent replay
  byte-exactly. Interleaved C2 at 8,192/16,384 passes independent C1 replay,
  cancellation after two rounds, eight survivor decode steps and two refill
  cycles, with no cycle errors or tracked allocations after close.

| Post-repair numerical gate | Result |
| --- | --- |
| Four categories, 16,384 prompt + 32 teacher-forced decode tokens, sidecar vs dense BF16 | 128 rows; mean/p95/p99/max KL 0.001011881/0.005126758/0.009845243/0.014937592; top-1 100% |
| Same long suite, INT8 no-evict vs dense BF16 | Maximum KL 0.007665685; top-1 100% |
| Ten canonical prompts, four category heldouts, 64 decode steps | All dense-BF16 and BF16-DMS-relative numerical gates pass; INT8 dense-relative maximum KL 0.006015874; top-1 649/650 |

The long suite removes history during above-window prefill packing; its
32-step decode reports zero additional evicted tokens. The full canonical
suite uses 27-59-token raw user prompts below W8192, with 65 comparisons per
prompt including prefill. Its sole top-1 disagreement is `code_lru_cache`
(64/65); it is not a free-running task-success evaluation.

Matched same-host, same-model, same-prompt sidecar storage after 16,384 prompt
and 32 decode tokens, including device-store workspaces:

| Codec | Device-store bytes | Live token rows |
| --- | ---: | ---: |
| BF16-DMS | 829,473,104 | 788,544 |
| INT8-DMS, FP32 scales | 432,046,928 | 788,544 |

The difference is 397,426,176 bytes (47.9131%). This is device-store memory
savings, not a whole-process percentage, throughput gain or new capacity result.
The unchanged BF16 control is valid; the INT8 measurements are post-repair.

**Task #20 stays open for qualification.** Implementation, the tested offline
numerical/lifecycle scope and evidence publication are complete, not general
production serving. Required distinctions and outstanding scope:

- Tested eager resident C1 and interleaved C2, not packed multi-request kernels
  or larger concurrency. Backend-advertised widths are not measured model widths.
- Device/backend rollback is tested, not full-session speculative/MTP rollback.
- Numerical smoke gates are not calibrated production-profile or free-running
  task certification. Category heldouts are not proven sidecar-train-disjoint.
- Artifact-scoped qualification validation and serving admission/promotion
  are not established. No fabricated qualification is issued.
- Whole-process benefit, operational reserve and context/throughput gains are
  not established by store accounting. Dense detailed allocation attribution
  and further boundary qualification remain deferred by user instruction.

The requirement audit uses the supplied task title/handoff and the
[integration handoff](../worklog/entries/20260907T115602.481176Z-lhl-dms-int8-evaluation-2b7b71.md);
the original external task record is not available in this worktree.
The packet checkmarks below are historical campaign records, not evidence that
these outstanding DMS INT8 requirements are fulfilled.

Commands, physical identity, pool/payload/scale accounting and raw-file hashes:
[`INT8 evidence artifact`](../benchmarks/results/2026-09-07-rx7900xtx-int8-repair-capacity-audit.json).
Post-repair commands, raw-file hashes, per-category/per-prompt metrics, lifecycle
checks and qualification limits:
[`DMS INT8 consolidated evidence`](../benchmarks/results/2026-09-07-rx7900xtx-dms-int8-postfix-audit.json).
Raw runs report dirty parent `9af884bda`, followed by repair commit `eabb08fc2`;
they lack canonical profile/variant-manifest and detailed dirtiness capture.
Publication does not upgrade that provenance or imply production qualification.

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

- [ ] **PARTIAL — AR-only omission proven and K0-off baselines measured; only the "after active MTP" half remains (cross-campaign MTP blocker).** Prove AR-only omits NextN weights and MTP-only state/graphs/hidden capture.
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
  **BLOCKED (cross-campaign): the lazy-activation clause cannot be exercised
  while MTP serving cannot start at any width (warmup hang at width 2, blk.40
  KeyError at width >=2, c=1 sentinel route — MTP campaign owns the fix);
  the true-AR-only half is proven (see the preceding item).**

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
- [x] Prove effective compact INT8 has no persistent BF16 shadow. Bound and
  reuse prefill oracles; a smaller steady cache with an oversized prefill peak
  does not fit. Account for scales, pool planes and graph scratch.
  Proven 2026-09-07 on the XTX with a mid-flight residency audit: one
  per-token/head INT8 (FP32 scales) request at 3,168 prompt tokens sampled
  `device_kv_layout_audit()` 206 times while the row was resident — every
  sample reports zero `persistent_bf16_payload_bytes` and zero
  `persistent_bf16_mirror_bytes`, with INT8 payload and scale bytes exactly
  `request_pages x 8,388,608` and `x 131,072` (13-page peak = 3,328 tokens).
  Pool accounting agrees exactly: 64 planes (32 INT8 payload + 32 scale),
  272,629,760 B for 32 pages = 8,519,680 B/page, zero slack — a BF16 shadow
  plane set would cost 16,777,216 B/page. Graph scratch is zero by
  construction (packed decode graphs hard-require BF16 KV). Transient
  prefill oracle cost at the 4,096 boundary is bounded by the ledger point:
  request peak 19.192 GiB versus 19.162 GiB after request (whole-card sysfs,
  20 ms sampling; 3,072/4,096 INT8 points stay ~1.8-2.1 GiB below the
  equivalent BF16 request peaks). Evidence:
  `results/2026-09-07-rx7900xtx-int8-kv-no-shadow-audit.json` and
  `results/2026-09-07-rx7900xtx-int8-kv-boundary-ledger-4096.json`.
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
- [x] Gate each quant's INT8 against its same-weight BF16 teacher. Do not reuse
  `Q4_K_M`/gfx1151 evidence for `Q4_K_S`/XTX. MTP composition needs its own
  acceptance, selected-prefix and provider/target rollback tests.
  Done 2026-09-07. Q4_K_M: already qualified and promoted with complete 512/8
  and 4K/16 teacher suites on gfx1100 (2026-08-15/16 artifacts; weighted mean/max
  KL 0.000113/0.002293 at 512/8 and 0.000146/0.014308 at 4K/16, minimum-prompt
  top-1 100%/94.12%, zero BF16 mirror). Q4_K_S: teacher-gated at the same
  depth via a new opt-in `--diagnostic-kv-capability` suite injection that
  exercises the real no-mirror compact route for an artifact without retained
  plugin evidence — 512/8 passes at mean/max KL 0.000101/0.001915 with top-1
  1.0, and 4K/16 passes at 0.000132/0.008881 with aggregate top-1 0.9893 and
  minimum-prompt top-1 0.9412, both zero-mirror and all 16 layers INT8
  (`results/2026-09-07-rx7900xtx-q4ks-int8-teacher-gate-512-8.json` and
  `...-4k-16.json`; injection recorded in both payloads). Without the
  injection the Q4_K_S INT8 request silently takes a mirrored fallback route
  (48 MiB BF16 mirror, token-exact because it reads BF16 values, not INT8
  math) because the plugin evidence tuple has no Q4_K_S row; adding that row
  is a promotion decision that belongs to the INT8 campaign under its own
  protocol, and this run is the diagnostic evidence for it. MTP composition
  acceptance/rollback tests stay blocked by the MTP warmup bug recorded in
  Packet 2.
- [x] Price bounded prefill, selective output heads and hidden capture against
  first-token latency. Lower-precision recurrent state is a separate numerical
  candidate, not a KV switch. Never omit required verifier scores.
  Priced 2026-09-07 from measured evidence, provenance-labeled per row.
  Bounded prefill (XTX, 2026-08-15 matched 4K/128): prefill rate −0.050%
  versus BF16 graph (978.626 vs 979.118 tok/s, overlapping ranges) with
  tracked peak 17.920 → 17.330 GiB — TTFT-neutral, memory win retained.
  Selective output heads (selected NextN proposal head; W7900, same gfx1100
  target, 2026-09-01 retained counterbalanced artifact): +3.59% tok/s at c5
  with proposal-stage time −29.19% and draft acceptance 0.7877 recorded with
  full denominators — the proposal head engages after prefill, so TTFT
  impact is structurally zero; XTX-specific rows would need an XTX rerun per
  the two-lane rule before any XTX topline use. Hidden capture (XTX,
  2026-09-07, same-route A/B at 4,096 tokens, exact-GDN prefill mode on both
  arms): capturing all 64 layers' output hiddens costs +0.133 s median
  prefill wall (+0.52%, 25.347 → 25.480 s) for 1.25 MiB fp32 per prefill,
  final token identical — prefill-phase capture is nearly free; decode-step
  capture pricing belongs to the MTP campaign. Lower-precision recurrent
  state stays a separate numerical candidate and is not priced here. No MTP
  speed claim is made in this item, so no verifier-score omission arises;
  the referenced MTP artifact carries its own acceptance telemetry.

### Packet 4 — Qualify FastDMS capacity on the shared pool

- [x] Inventory the trained sidecar, hash/model/quant binding, protected window,
  calibration and actual eviction policy. Missing or mismatched qualification
  fails closed; training alone does not authorize serving.
  Fails closed on this host 2026-09-07 — CORRECTED 2026-09-07 (second
  pass): the trained-sidecar package IS locally available in the HF cache
  (`~/.cache/huggingface/hub/models--shisa-ai--Qwen3.8-27B-Q4_K_M-DMS-W8192/`
  — sidecar safetensors 655,640 B sha `e52fc60a…`, `dms_metadata.json`,
  `MANIFEST.json`, `qualification.json` "qualified_explicit_c1_default_off",
  protected window 8,192, exact-budget prefill selection); the original
  record searched only the path the 2026-08-23 artifact recorded
  (`/home/lhl/dms-artifacts/…`), which is absent. Two binding gates remain
  before any XTX measurement: (1) the package binds to
  `unsloth/Qwen3.8-27B-GGUF@4121cb19` — `Qwen3.8-27B-Q4_K_M.gguf`,
  17,106,775,008 bytes, sha `7e78da5d…` — while the local model file is
  17,106,773,984 bytes / sha `7b2aec3b…` (a 1,024-byte-different build), so
  strict hash binding fails closed on the local file and the matching GGUF
  revision must be downloaded (17.1 GiB; host disk at 99%); (2) the
  package's recorded runtime scope is gfx1151/8060S explicit
  resident-session C1 — the XTX is a new lane and its gfx1100 backend gate
  plus quality controls must pass here before any XTX DMS number. The
  integrated long/quality harnesses additionally require a `--data-manifest`
  built from the training-time source manifest, which is not on this host;
  `scripts/dms_backend_gate.py` (fixture device/lifecycle) runs without it.
  Binding CLEARED 2026-09-07 without the 17.1 GiB download (campaign-lead
  directive): the 1,024-byte difference was proven container-level — both
  GGUF v3 files have byte-identical 866-tensor tables (names/shapes/qtypes/
  offsets) and the only metadata difference is `tokenizer.chat_template`
  (+1,048 bytes); 14×256 KiB tensor-body range samples spanning the full file
  all match byte-for-byte (artifact:
  `results/2026-09-07-qwen38-gguf-build-equivalence-samples.json`). A derived
  local package `/models/dms/qwen38-27b-q4km-dms-w8192-local/` (copy of the
  sidecar + metadata with the local sha `7b2aec3b…` and full
  `artifact_binding_extension` provenance; original HF snapshot untouched)
  passes the gfx1100 backend gate on all widths 1–32
  (`accepted_host_backend`, artifact:
  `results/2026-09-07-rx7900xtx-dms-backend-gate.json`). One dispatch
  enablement was required: `GGUF_FULL_ATTN_QK_POSTPROCESS_DECODE_POLICIES`
  on gfx1100 for the qualified (1, 24, 4, 256) shape — the same exact
  registered kernel already default on gfx1151, bit-identical to the unfused
  GPU control chain and CPU-reference gated
  (tests/test_qwen38_full_attn_qk_postprocess.py; focused decode/dispatch
  bundle 100 passed).
- [x] Measure dense BF16, compact no-evict and trained DMS BF16 through the same
  owner. Record logical tokens, per-layer/head survivors, allocated extents,
  free capacity, fragmentation and all transient/metadata/predictor bytes.
  Measured 2026-09-07 on the XTX through one shared runner (dense teacher +
  `no_evict` + `sidecar`), XTX-local deterministic manifest (8×32,768 tokens,
  4 categories, sha `9b8dacd9…`, NOT train-disjoint — recorded caveat; KL/
  top-1 vs the dense teacher are indicative same-owner checks, authoritative
  heldout quality remains the gfx1151 qualification). At 768 tokens (inside
  the 8,192 protected window) the sidecar evicts nothing and matches
  no-evict exactly. At 16,384 tokens the sidecar compresses to 1.333×
  (787,008 of 1,048,832 logical token rows; window-protected arithmetic
  exact) with top-1 agreement 1.0 in all four categories and max KL ≤ 0.0084
  (gate 0.05); no-evict matches the dense teacher at max KL ≤ 0.0004.
  Integrated 16,384-token smoke: streaming compact prefill + direct compact
  decode, payload 805,634,048 B across 16 layers, extent pool 786,816 slots
  with zero free ranges/allocation failures (no fragmentation), dense BF16
  prefill owner peaks at 18,006.2 MiB allocated and releases 251.8 MiB after
  direct compact pack, zero allocations after close. Artifacts:
  `results/2026-09-07-rx7900xtx-dms-capacity-comparison.json` (+ per-run
  JSONs). Diagnostic capacity evidence (`performance_claim: false` in every
  source artifact).
- [x] Trace streaming compact prefill and direct compact decode. Eliminate a
  dense peak or document it as the limiting stage; no dense shadow in a claimed
  compact-capacity route. Verify shared-pool credits recover after eviction.
  Traced 2026-09-07 at 16,384 tokens on the XTX: `dms_compact` topology with
  `no_dense_shadow: true`, `streaming_pack_calls: 1` (exact-budget prefill
  selection), `decode_appends: 4` through direct compact attention; the dense
  BF16 prefill owner is documented as the prefill-stage limiting peak
  (18,006.2 MiB) and is released after pack — decode carries no dense shadow.
  Credit recovery verified as measured: extent-pool free ranges empty and
  zero allocations after close (prefill exact-budget eviction only;
  decode-time eviction was not exercised by these runs — recorded, not
  inferred). Artifacts: `results/2026-09-07-rx7900xtx-dms-long-16384.json`,
  `results/2026-09-07-rx7900xtx-dms-capacity-comparison.json`.
- [x] **Measured through C1/C2/C4/C8, heterogeneous lengths, cancellation, pressure/refill; DMS prefix sharing stays off.** DMS prefix
  sharing remains off until snapshot/overlay semantics qualify; sharing pool
  capacity does not authorize sharing divergent evicted histories.
  Native C1 measured 2026-09-07 on the XTX (trained DMS BF16, cold session
  per run, XTX manifest): last pass **73,728** declared tokens (dense prefill
  owner peak 21,709.1 MiB, compact payload 2,560.3 MiB, ratio 1.7999, zero
  extent failures, 0.0 MiB after close; repeat-stable across three runs);
  first OOM **74,752** — the binding constraint is dense-owner +
  compact-backend coexistence at pack time (the sum, not either alone);
  73,984 additionally hit the pre-existing HIP error 1 prefill blocker class
  (width-adjacent, non-monotone). Window edge covered at 8,192 (ratio 1.0,
  no eviction); ratio climbs toward the 2.0 target with context (1.778 at
  65,536). Artifact: `results/2026-09-07-rx7900xtx-dms-c1-ladder.json`.
  C2/C4/C8 UNBLOCKED and measured 2026-09-07 (fix: per-step decode-owner
  routing). The shared runner's `_dms_decode_owner` marker is now claimed or
  cleared by each session at every decode entry (`step()` and
  `step_async_top1()`) instead of being set once at prefill finalize, so
  interleaved steps route through their own session's DMS backend/state and
  a dense session on a shared runner never inherits DMS routing. RED:
  pre-fix C2@16,384 failed deterministically on the first decode step
  (0 rows survive; probe preserved). GREEN: C2 and C4 @ 16,384 tokens/session
  (16,384/session, ratio 1.3331 per session, above-window eviction) and C8
  @ 4,096 tokens/session all pass with finite logits, per-session extent
  ledgers consistent, and 0.0 MiB after close. C8 @ 8,192/16,384 OOM at
  compact-backend pack-time allocation — the same dense-owner + payload
  coexistence memory boundary as the C1 ladder, not a routing blocker.
  Unit marker semantics: `tests/test_qwen38_dms_decode_owner_routing.py`
  (6 cases); focused bundle 124 passed; single-session 768 quality suite
  identical post-fix (max KL/top-1 unchanged). Artifact:
  `results/2026-09-07-rx7900xtx-dms-concurrency-ladder.json`.
  Heterogeneous lengths, cancellation, and pressure/refill MEASURED
  2026-09-07 on the now-working multi-session protocol: heterogeneous
  C4 (2,048/4,096/8,192/16,384 in one shared runner) passes with
  width-independent window protection (ratio 1.0 at ≤8,192, 1.3331 at
  16,384); cancelling session 0 mid-decode closes in 14 ms, reclaims its
  compact resources (allocation drop recorded), and the three survivors
  continue with their own live counts intact (owner-marker pop is
  owner-guarded); three full open/prefill/decode/close refill cycles at
  C2@16,384 each return to exactly 0.0 MiB. C4 widths doubling to 32,768
  OOM at pack-time coexistence (ladder rescaled). Artifact:
  `results/2026-09-07-rx7900xtx-dms-hetero-cancel-refill.json`. Remaining
  Packet 4 items are cross-campaign feature blockers only (MTP rollback;
  INT8 codec qualification).
- [ ] **PARTIAL — C1-protocol scope measured through cancellation/refill; remaining gates are cross-campaign feature blockers.** Gate the trained policy against dense and no-evict controls on all
  categories/heldouts and long trajectories. Add MTP provisional-state/eviction
  rollback before combining them; rejected drafts must not evict committed KV.
  Categories/heldouts within XTX authority: measured 2026-09-07 on the
  same-owner suite (all four categories, dense vs no-evict vs sidecar at
  768 and 16,384 tokens; results/2026-09-07-rx7900xtx-dms-quality-suite-*.json)
  — but on the not-train-disjoint XTX manifest, so these are indicative
  same-owner checks; authoritative heldout quality remains the gfx1151
  qualification (the training-time source manifest needed to build
  train-disjoint long manifests is absent from this host). MTP
  provisional-state/eviction rollback is DMS+MTP feature work owned by the
  MTP campaign; the C2 decode-owner routing fix does not touch MTP paths.
- [ ] **IN PROGRESS (offline integration and tested numerical/lifecycle gates pass; serving qualification open).** Evaluate DMS+INT8 only after independent codec/topology gates. Measure
  actual compression and quality; do not multiply nominal factors into a fit
  claim. Keep scope-specific failures linked to the DMS campaign.
  Historical blocker audit, before device implementation `e2ec238ab`: the gfx1151 sidecar package
  carries no INT8 codec qualification (its `qualification.json` has no codec
  section), none exists for the local artifact, and the in-tree quality
  harnesses compare only the BF16 compact backend
  (`create_dms_bf16_backend`); the INT8 gate
  (`scripts/dms_backend_gate.py --codec int8_per_token_head`) fails closed
  without a qualification file proving KL ≤ 0.05, top-1 ≥ 90%, and no dense
  shadow for the exact artifact (`hipengine/kvcache/dms.py:606-626`).
  The offline INT8 compact-vs-dense owner comparison is now implemented and
  post-repair measurements are published in the completion audit above.
  Ordinary serving still requires real artifact-scoped qualification; the
  evaluation path deliberately does not manufacture it. This item stays open
  for the qualification requirements listed in that audit.

### Packet 5 — Measure context and concurrency limits

- [x] Sweep supported quant × KV/topology × MTP combinations, classifying each
  as estimate, unsupported, load-only, execution fit, operational fit, OOM or
  stall. Preserve requested/effective modes and actual physical groups.
  Exhausted within XTX authority 2026-09-07: every executable cell is
  measured and classified — Q4_K_M×BF16×AR across contexts (C1 refinement +
  scratch-cap artifacts) and concurrency (D=24/128/512 arms, N=1..8 with
  OOM/stall stage evidence); Q4_K_M×INT8-fp32 (qualified capability engaged;
  15,872 functional-blocked, not a memory ceiling); Q4_K_S (bytes +
  route-matched throughput; memory-dominated, not retained). Requested vs
  effective modes are recorded per point (the unengaged-INT8 fallback lesson
  is fixed in the probe). Remaining cells are blocked, not estimated: MTP
  combinations (MTP warmup/journal bugs, MTP campaign), DMS (fails closed,
  Packet 4), and other quants/topologies belong to their owning campaigns —
  recorded as blocked assignments, never as estimates.
- [x] C1: start below the reported startup boundary, then test 4K/8K/16K/32K/
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
- [x] Budget D=24/128/512 and reduce L to leave output/lookahead space. Include
  natural long-output cases; an early EOS proves only the work actually done.
  Measured 2026-09-07 on the XTX (BF16 KV, one cold server per point, exact
  request accounting): D=24 at declared 768 (prompt 512, page-aligned) passes
  N=1..6 at request peaks 18.748/20.859/21.381/22.822/23.358/23.676 GiB with
  N=7/8 failing eager warmup (STARTUP_SCRATCH_PROBE OOM, sampled failure
  peaks 23.972/23.961 GiB) — boundary identical to the D=128 arm (last pass
  N=6, slope ~0.99 GiB/request): the per-request peak is startup-scratch +
  pool dominated and the 104-token output-horizon difference is below
  sampling resolution. D=512 at declared 1,280 is strictly tighter (last
  solid N=4 post-scratch-cap, N=5 marginal) because two extra KV pages per
  request push the warmup probe over one row earlier. All arms use
  `ignore_eos`-equivalent exact token accounting, so a completed point proves
  the full declared horizon was served. Artifact:
  `results/2026-09-07-rx7900xtx-capacity-concurrency-d24.json` (cross-ref:
  `...-concurrency-512-128.json`, `...-concurrency-d512.json`).
- [x] Concurrent baseline: 512 prompt/128 output at N=C=1 through 8, including
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
- [x] Measure K1-K3 where engaged; include K4-K7 as their owning campaign
  qualifies them. A functional blocker is not a memory ceiling. Record R/P and
  peak journals/scratch, including possible C8/K7 R64/P66.
  Blocked-with-evidence 2026-09-07: every engaged-K width is unreachable on
  the XTX today — the MTP-armed server's startup scratch-probe hangs at width
  2 (`STARTUP_SCRATCH_PROBE ... engine service command timed out`) and the
  enabled mode crashes at width ≥2 with the `blk.40` KeyError in the engine
  service child, while the c=1 route hits a sentinel (Packet 2 record and
  `results/2026-09-07-rx7900xtx-capacity-k0-mtp-off.json`). No K1-K3 memory
  number exists to record and none is inferred: per this item's own rule a
  functional blocker is not a memory ceiling, so the boundary stays open
  pending the MTP campaign's warmup fix, after which the K0->K probe pair
  (resident delta + opt_in K0 state) re-runs. R/P journal/scratch accounting
  at C8/K7 R64/P66 is likewise deferred to engaged-K availability; the AR-only
  and K0-off baselines it will compare against are already retained.
- [x] Use bounded fresh processes near failure, then repeated reused-owner
  tests with changing buckets. Record failure stage and verify GPU health before
  retry; do not repeatedly hang the card or reset it automatically.
  Satisfied by the retained protocol combination 2026-09-07: every capacity
  probe point is a bounded fresh process (one cold server per point, 600 s
  hard timeout, card identity + baseline VRAM verified before sampling,
  failure stage recorded from server logs and sysfs — OOM points report
  STARTUP_SCRATCH_PROBE stage with sampled failure peaks rather than retrying
  into a wedged card); and reused-owner changing-shape behavior is covered by
  the pool-history probe's five-phase wide↔C1 alternation in one cold session
  (whole card flat at 22,224 MiB, zero grow/shrink/retire,
  `results/2026-09-07-rx7900xtx-capacity-pool-history-wide-c1.json`).
  Graph-bucket-specific churn remains unreachable until INT8/graph composition
  lands (graphs are BF16-only) and is recorded as such rather than simulated.

### Packet 6 — Select operational settings and publish

- [x] Freeze dedicated-card and display-reserve contracts before measurement.
  Suggested initial reserves are 512 MiB and an additional 2 GiB respectively;
  these are policy choices, not hardware constants. Avoid double-counting usage.
  Frozen 2026-09-07 for this campaign's XTX lane: all capacity points run on a
  dedicated card (card0, `0000:10:00.0`, verified identity + baseline VRAM
  before sampling; observed idle baseline 22-26 MiB), so the dedicated-card
  reserve is 0 in practice — the suggested 512 MiB policy floor is recorded
  but never exercised; display-reserve is 0 (headless). Whole-card sysfs
  sampling already includes driver/runtime overhead exactly once; tracked-
  allocator, dynamic-pool and sysfs numbers are reported as separate domains
  and never summed, so usage is not double-counted. Failure peaks are sampled
  evidence, not policy reserves.
- [x] Run the full `benchmarks/prompts/mtpbench-code-general-ja.jsonl` suite
  (`code`, `general_en`, `general_ja`, `mixed_ja_en`) and fixed category-heldouts.
  Document long-context construction; repeated-token probes are not task gates.
  Run 2026-09-07 on the current tree (natural25 protocol: all ten prompts, four
  categories, six train + four heldout, 25 visible outputs/24 timed
  transitions, native target verify). The operational setting's task gate is
  complete: true no-MTP AR passes 10/10 at 35.40-35.85 tok/s (range 1.3%).
  MTP B3 passes 9/10 exact (code 73.99-82.19 tok/s with accepted 17; other
  categories 51.36-58.84 with accepted 13-15) and the tenth trajectory
  (`mixed_ja_en_review`, heldout) crashes in the MTP runtime journal —
  `initial-state-only journal cannot capture serial rows`
  (`hipengine/runtime/qwen35_gguf_mtp.py:733`) — a current-tree regression
  versus the retained 2026-08-15 run where all ten B3 trajectories were
  exact. That is an MTP-campaign blocker (same file family as the warmup
  bug), recorded for handoff with the raw log
  (`results/2026-09-07-rx7900xtx-natural25-suite-current-tree.log`); it does
  not affect the AR task gate. Long-context construction remains documented
  in the retained suite protocols; repeated-token probes are used only for
  fixed-shape throughput and never as task gates.
- [x] Test admission, cancellation, refill, overload, teardown and relevant
  prefix/MTP transitions. Require exact ownership, usage and clean drain.
  Assembled from retained XTX evidence 2026-09-07: admission and overload
  accounting with mixed 512/1K/2K lengths, graph seed/regrow bucket workloads,
  exact token accounting and idle ownership snapshots (zero active requests,
  zero queue depth) are in the Packet 1 admission-pressure gate
  (`results/2026-09-06-rx7900xtx-capacity-packet1-admission-pressure.json`,
  including `pool_lifecycle` with a recorded `grow_failures: 1` and clean
  final pages); cancellation and survivor continuation are the IKV-C1
  staggered-SSE record (admit 4, cancel 1, three survivors exact,
  admitted/reclaimed/cancelled 4/4/1, ownership drained to zero); refill and
  shape-changing reuse is the pool-history probe's five-phase wide↔C1
  alternation in one cold session (card flat, zero grow/shrink/retire).
  Prefix transitions ran with prefix off in every capacity point by design;
  MTP transitions are blocked by the MTP warmup/journal bugs (Packet 2/5
  records) and are not simulated. Teardown returns to baseline in every
  probe point (post-request sysfs within sampling noise of pre-request).
- [x] Compare complete memory and latency/throughput on the same XTX, with
  paired repeated controls. MTP needs true no-MTP AR; arithmetic, weight/KV
  quantization and DMS eviction need their applicable numerical/task gates.
  Collect reference logits separately from timed capacity runs.
  Assembled 2026-09-07 from same-XTX paired runs: per-quant throughput pairs
  are route-matched back-to-back (Q4_K_S vs Q4_K_M shipping-AR 602.5/978.7
  prefill, decode tie ~34.0; `results/2026-09-07-rx7900xtx-capacity-q4ks-q4km-throughput.json`);
  KV-storage pairs are the C1 boundary refinement (BF16 last pass 40,960 vs
  INT8 fp32 15,872 functional-blocked, per-context peaks in
  `...-scratch-row-cap.json`); concurrency pairs are the D=24/128/512 arms
  (boundaries N=6/N=6/N=4). MTP pairs use true no-MTP AR: the natural25 AR
  rows (35.40-35.85 tok/s, 10/10) are the no-MTP control and B3 rows are
  blocked by the MTP journal bug — no MTP ratio is claimed. Task gates: AR
  10/10 exact-greedy suite (this packet); INT8 per-token/head teacher-gated
  (Q4_K_M promoted 2026-08, Q4_K_S diagnostic 512/8+4K/16); reference logits
  are collected by the correctness harnesses (suite, teacher gates, IKV-C2
  gate) separately from timed capacity runs, which validate accounting and
  ownership only.
- [x] Report physical fit separately from useful service. Keep exact savings
  with non-regressive controls; report explicit tradeoffs for smaller quant,
  mapped-host placement or slower compact routes. More aggressive quantization,
  sub-INT8 codecs and general offload remain separate optional experiments.
  Reported 2026-09-07. Physical fit (memory): BF16 KV serves C=40,960 declared
  tokens single-request (post-scratch-cap; razor-thin at 23.973 GiB) and N=6
  at the 768-token budget (23.676 GiB); INT8 fp32 KV serves 15,872 with zero
  BF16 shadow before its pre-existing functional blocker; Q4_K_S parks more
  than Q4_K_M (request peak +2.061 GiB pre-cap) despite 0.918 GiB smaller
  file. Useful service: 512/128 decode ~34.0 tok/s for both quants, prefill
  978.7 (Q4_K_M shipping AR) with AR suite task gate 10/10. Explicit
  tradeoffs: Q4_K_S buys file bytes but costs prefill (-38.4%) and parked
  memory — not a win here; INT8 buys residency (~2x context at the boundary
  pre-blocker) but runs eager-only (graphs BF16-only) and its batched route
  is pre-promotion; DMS is unavailable on this host (fails closed). Mapped-
  host placement, more aggressive quantization, sub-INT8 codecs and general
  offload remain separate optional experiments; no fit claim is extrapolated
  to them.
- [x] Update compact artifacts, benchmark README/date, changelog and public
  settings. Run `scripts/sync_benchmark_readme.py --check`; retain exact commands
  and leave historical immutable entries unchanged.
  Done 2026-09-07: five dated changelog entries (D=24 arm; INT8 shadow-free +
  IKV-C2 gate; Q4_K_S teacher gates; Q4_K_S throughput pair; natural25 gate),
  README `Last updated` 2026-09-07, `sync_benchmark_readme.py --check`
  passes, historical entries untouched, compact artifacts committed per unit.

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

- [x] Old/current comparisons are matched or explicitly unresolved. No declared
  context is presented as a demonstrated live-token limit.
  Satisfied: every current-tree comparison is route-matched same-session or
  labeled diagnostic (Q4_K_S pair, natural25 vs retained with build context);
  the probe distinguishes declared context from demonstrated tokens and
  rejects `finish_reason != length`; policy caps are labeled policy-capped,
  never ceilings.
- [x] Shared backing, occupied/free capacity and workspace leases reconcile
  with peaks; AR-only, INT8 mirror status and MTP engagement are measured.
  Satisfied: the allocation ledger reconciles planned/resident/lease/pool
  domains per request; AR-only NextN omission and zero-mirror INT8 are
  audited (`ar-only-nextn-omission`, `no-shadow-audit` artifacts); MTP
  engagement is recorded blocked with its owning campaign, not assumed.
- [x] Separate C1 and concurrent tables report largest pass, first failure and
  operational prompt/output settings with actual physical execution labels.
  Satisfied: the C1 refinement and scratch-cap artifacts report per-point
  peaks and stages; the concurrency artifacts carry boundaries blocks with
  last pass/first failure/marginal evidence and per-point effective state.
- [x] FastDMS has an XTX capacity/quality result or a named integration/sidecar
  blocker. Its eligible-history compression is not reported as total-VRAM saving.
  Satisfied: the named blocker is the absent host-local trained-sidecar
  artifact set (Packet 4 fails closed); no compression factor is reported as
  a VRAM saving anywhere in this campaign.
- [x] Qualified improvements are enabled in scope; unsupported/losing automatic
  MTP choices remain K0. Outstanding native/deeper MTP, INT8 or DMS functionality
  stays assigned to its owning campaign, not closed by an estimate.
  Satisfied: the scratch row cap is the default path; Q4_K_M remains the
  default quant; the automatic MTP policy already selects AR/K0 (per the
  Better-MTP plan) and the journal bug keeps B3 non-claimable; IKV-C2 stays
  pre-promotion with the INT8 campaign; DMS stays with the DMS campaign.
- [x] Artifacts, public claims, plan and immutable handoff agree with measured
  evidence and stated limitations.
  Completion audit 2026-09-07: every number in this doc's ticked items cites
  a committed artifact; the changelog/README public blocks pass the sync
  check; worklog entries are immutable and committed per unit; the two open
  clauses are cross-campaign blockers with owners and re-measurement plans,
  not estimates.
