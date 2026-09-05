# Concurrency and Continuous Batching

Last updated: 2026-08-17.

> **Superseded for active architecture planning.** The Generation-2 request,
> scheduler, and shared-KV redesign is now in
> [`CONCURRENCY2.md`](CONCURRENCY2.md). This file remains the retained c=N
> kernel/resident-runner roadmap and evidence history. Its unresolved work is
> migrated or delegated in the Generation-2 migration ledger; do not add new
> active queue items here.

This document was the source-of-truth roadmap and punchlist for making `c=N` a
first-class model pipeline in hipEngine. Its destination was fully native
single-GPU continuous batching for:

- GGUF and PARO model paths;
- gfx1100 and gfx1151 backends;
- prompt prefill, decode, row retirement, cancellation, streaming, and reclaim;
- greedy and normal per-row sampling.

This is a planning document, not an experiment diary. Results appear here only
as one-line status summaries with pointers to `WORKLOG.md` or retained artifacts.
Detailed commands, measurements, negative probes, and bisection history belong
in `WORKLOG.md` and `benchmarks/results/`.

Related source-of-truth documents:

- [`PLAN.md`](PLAN.md) — architecture invariants and the batch-shaped runtime
  design.
- [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) — strict, production, and
  batch-invariant numerical, ownership, and determinism contracts.
- [`BENCHMARK.md`](BENCHMARK.md) — c=N correctness and performance acceptance
  protocols.
- [`ROADMAP.md`](ROADMAP.md) — project-level phase ordering.
- [`KVCACHE.md`](KVCACHE.md) — KV policy and future DMS work.
- [`PREFILL.md`](PREFILL.md) — prefill implementations and boundaries.
- [`SAMPLING.md`](SAMPLING.md) — normal sampling support.
- [`NATIVE_SPEC_CYCLE.md`](NATIVE_SPEC_CYCLE.md) — canonical speculative-cycle
  milestone glossary and current MTP/provider performance scorecard.
- [`ENVS.md`](ENVS.md) — supported runtime knobs.

## Goal

hipEngine must treat an active group of requests as one canonical runtime shape,
not as N independent c=1 sessions hidden behind a server coalescer.

When this roadmap is complete:

1. One model-owning engine loop outlives individual API calls.
2. Requests may be admitted while other requests are decoding.
3. Prefill and decode are scheduled as peer work classes.
4. One native model step advances every active decode row once.
5. Finished, cancelled, and disconnected rows are reclaimed at commit points
   without stopping live neighbors.
6. Physical slots may be sparse, compacted, and reused without changing stable
   request identity.
7. Per-row sampling parameters and stop conditions coexist in one active group.
8. KV allocation, prefix reuse, graph buckets, and metrics are owned by the same
   loop.
9. The same contracts work for GGUF/PARO and gfx1100/gfx1151 without backend,
   quant, or execution-profile branches in engine/model code.
10. Every run declares `strict`, `production`, or `batch_invariant`; exact
    control/state ownership binds at every width, while arithmetic and
    cross-composition equality follow the declared profile.

The first retained milestone is not “the server accepts concurrent requests.”
It is a correctness- and profiler-proven native c=N model step. The final
milestone is live admission/reclaim around that step.

## Terminology and claim levels

| Term | Meaning |
| --- | --- |
| HTTP concurrency | Multiple client requests are in flight. Backend execution may still be serial. |
| Static prompt-list batch | One API call contains multiple prompts and runs one fixed batch from start to finish. |
| Serial bridge | Batch-shaped host metadata, but complete c=1 model work runs row by row. Correctness diagnostic only. |
| Exact hybrid | Some operations are batch-shaped while explicitly identified c1-exact row operations remain. May be a production-safe fallback, but is not a fully native throughput claim. |
| Native c=N model step | One model-step contract consumes `[C, ...]` state and advances all active rows without complete c=1 session/layer replay or host-side per-row model loops. |
| Native group width | Largest number of rows advanced by one native model step. Two serial c4 groups are not native c8. |
| Continuous batching | Requests are admitted, prefilled, decoded, finished, and reclaimed while other requests remain live under one model-owning loop. |
| First-class c=N pipeline | Native c=N is the canonical internal path. c=1 is the `C=1` instance, not a separate architecture. |
| Retained c=N row | Declared execution-profile correctness, lifecycle, profiler, scaling/SLO, memory, and observability gates all pass under `docs/BENCHMARK.md`. |

### Fully native does not require one giant kernel

A native model step may launch multiple kernels and may preserve independent-row
math inside a batch-shaped kernel. It must not:

- create one resident c=1 session per active row;
- execute a Python loop over complete layers or complete model steps per row;
- call the c=1 layer runner once per row and label the result native;
- claim c8 when execution is two serial c4 groups;
- hide per-row/split/serial fallbacks from artifact and profiler metadata.

Row-local or width-specific arithmetic is allowed when launched through an
honest c-aware batch interface and reported under its execution profile. Strict
retains its exact boundary; production may reassociate only after the complete
strict-teacher/dynamic/task gate. Performance promotion still requires the
profiler to show that the route scales better than the serial bridge.

## Current truth

**hipEngine now has correctness-retained OpenAI continuous membership for both
gfx11 GGUF paths, but not project-wide production continuous batching.** One
long-lived runner, fixed reusable session identities, and a scheduler-owned
request-sized policy-shaped device-KV pool admit controlled submissions while a neighbor
decodes, commit bounded prompt chunks, stream row-owned token events, cancel or
retire rows, and reuse exact resources while survivor token/Conv/GDN/live-KV
state remains c1-exact. Public blocking `LLM.generate()` and OpenAI SSE drive
the same configured loop. Historical gfx1100 D4/D5/E3/F1 close streaming,
lifecycle, arbitrary-C lowering, and compaction correctness. The current
2026-08-08 source-pinned packet additionally restores context-qualified physical-C8 admission for the
p512/d128 envelope while preserving the unspecified/long-context C4 safety
cap. gfx1151 independently closes direct native-c2/c4/c8 correctness/scaling,
admission, two mid-flight disconnects, streaming, request/KV metrics, fallback
accounting, final ownership, E3 arbitrary-C lowering, and the full F1 server
packet. Current direct native-c8 graph model steps are scaling-retained at
**266.479** and **133.251 aggregate tok/s** on gfx1100 and gfx1151. gfx1100's
current C13 SSE row is **129.507 tok/s** with **189/189** static rows exact, and
live C8→C13 is **122.860 tok/s / 1,664/1,664 IDs** exact with zero fallback;
gfx1151 retains 73.235 at C13. The current gfx1100 direct/server decision is
recorded in the
[`context-scoped C8 server refresh`](../benchmarks/results/2026-08-08-gfx1100-context-scoped-c8-server-refresh.json).
Explicit PARO direct c2 steps are retained at
121.923 and 79.237 aggregate tok/s on gfx1100 and gfx1151; gfx1151 also retains
true physical c4/c8 at 100.209/99.943 aggregate tok/s. gfx1151 now also retains
its PARO resident owner and package default: stable model slots, chunked prefill,
native c2/c4/c8 decode, cancellation/reuse, blocking OpenAI accounting, and
concurrent exact SSE share the backend-neutral production loop. Blocking F1
c1/c2/c4/c8 is 47.124/51.962/60.323/61.253 aggregate tok/s with all 68 rows
exact; SSE c1/c2/c4/c8/serial-c8 is 36.327/38.666/42.471/41.487/35.633 with
all 100 rows exact. gfx1151 GGUF additionally retains occupancy-adaptive
physical c1/c2/c4/c8 execution with stable request/state/KV ownership and one
true resident physical-C8 group. The clean corrected-window BF16 server packet
is **44.137/59.258/74.105/83.771 tok/s** blocking and
**43.307/58.579/72.872/81.609** exact SSE at p512/d128; delayed C8 is
**66.563**, all 117 rows are exact, and final graph/KV/workspace ownership is
zero. The matched explicit mirrored-INT8 packet is
**44.225/60.598/74.631/83.408** blocking and
**42.759/55.128/71.284/81.140** exact SSE, with delayed C8 **65.034**. Its
full 11-prompt/99-position fidelity gate remains KL=0 with 100% top-1. The
<=8192 INT8 route is not default or memory-saving; tail4, direct/no-mirror INT8
attention, and longer c>N INT8 remain fail closed. F5 retains deterministic sampled
blocking/SSE c4 and `n=3` on packed model ticks, exact generated-token/logprob
metadata, stop/request-EOS, strict tool forcing, and fail-closed structured
results with zero serial/resident fallback. Its first opt-in prefix slice also
shares an active exact-current 256-token GGUF page with byte-exact survivor
state, moving p256+s1 continuation TTFT **249.269 -> 21.188 ms (11.765x)** and
live pages **4 -> 3** on gfx1151. A separate correctness gate resets the source
before admission and restores the same boundary from a cache-owned snapshot with
byte-exact state/KV and refcounts **1->1->2->1->0**. Its matched continuation
TTFT is **249.446 -> 22.013 ms (11.332x)** at an exact **72,089,600-byte** cache
residency cost and no unique-page saving. Broader historical boundaries,
sampled reuse, promotable agentic prefix economics, gfx1100 PARO owner
symmetry/c4/c8, and general prefix/continuation reuse remain open. The gfx1100
deterministic prefix correctness transfer and agentic 2K/8K evaluation are
closed, but the measured route is not promotable.
Long-context pressure, matched external comparisons, and project-wide production
promotion remain open.

The merged UD-Q3_K_M branch also retains a separate direct/native GPU1
RX 7900 XTX result: exact C=2/4/8 IDs and full logits, native graph/profiler
provenance, and **207.780/211.177 aggregate tok/s** at C8 512/4K. Its synchronous
prompt-list scheduler, compaction, and admission evidence remain useful direct
oracles; current persistent-server ownership and the broader agentic conclusions
below remain the governing production contract. See
[`2026-07-21-gpu1-q3-native-cn-retained.json`](../benchmarks/results/2026-07-21-gpu1-q3-native-cn-retained.json).

### Production OpenAI serving objective

The next target is not merely a wider retained batch. It is one production
server configuration that preserves the fastest profile-qualified c1 route
when only one request is live and increases physical width only when occupancy
and the latency policy justify it. A concurrency-capable server must not make a lone
request pay for masked physical c8 execution. Stable request identity, device
state, and KV ownership must survive c1/c2/c4/c8 bucket changes without
re-prefill or row migration visible to the client.

“Retain c1 performance” means near-direct transition performance at occupancy
one; it does not mean every request receives c1 throughput while sharing one
GPU. Under load, the objective is maximum generated-token goodput subject to
explicit TTFT/ITL SLOs. Direct model-step, backend transition, complete HTTP,
and complete SSE cycle walls remain separate timing scopes.

The production closure order is:

1. re-certify the current shared engine/server tree on gfx1151 GGUF;
2. select exact c1/c2/c4/c8 execution from live occupancy in one owner;
3. profile and remove host/graph/kernel overhead without regressing c1 or c>N;
4. pass mixed-length, continuous-arrival, overload, cancellation, and soak gates;
5. close sampled/API-path, prefix/KV-reuse, and long-context memory-pressure
   coverage;
6. run matched same-GGUF llama.cpp concurrency plus clearly qualified
   vLLM/SGLang serving comparisons.

This is a focused production-serving program, not a claim that broad
vLLM/SGLang feature parity already exists. The constrained greedy
Q4_K_M/BF16-KV loop is real continuous batching; production parity additionally
requires adaptive low-occupancy dispatch, broad request-shape coverage,
cache/memory economics, and SLO-based external evidence.

### Qwen3.8 pure-INT8 route boundary

The W7900 Qwen3.8-27B `Q4_K_M` pure-INT8 capacity soak used non-streaming
blocking requests. Its compatible-sampling HTTP queue submitted offered c2 as
one c2 call, c4 as c2+c2, and c8 as c1+c3+c4. Those are valid exact-request,
call-shape, memory, and lifecycle rows, but they do not test decode-time
admission, retirement, or slot refill. The raw `/ready`
`continuous_decode=false` value was also a telemetry bug: readiness called the
capability helper without the loaded engine, while the engine-aware
capabilities endpoint already reported controlled membership correctly.

Source audit confirms that this GGUF generator does construct the shared
resident owner: `SubmitPollTextGenerator` wraps
`Qwen35GGUFResidentModelRunner`, whose scheduler admits requests, interleaves
bounded prefill/decode work, plans occupancy-adaptive physical groups, compacts
slots, reclaims completed rows, and binds request-sized allocations from the
dynamic device-KV pool. The server launches independent controlled tasks for
SSE requests, while the blocking endpoint intentionally coalesces one static
call at a time.

A separate W7900 controlled-SSE gate now qualifies that short-route lifecycle
at 512 prompt + 24 decode tokens. After resident metrics observed c1 in decode,
three new SSE requests joined; sampled occupancy moved
`0 -> 1 -> 4 -> 3 -> 2 -> 1 -> 0`, all four outputs exactly matched independent
c1 token oracles, admission/reclaim counters advanced `4/4`, and final
active rows, occupied slots, and refcounted pages were zero. Thus <=8K mirrored
Qwen3.8 INT8 has measured live admission, occupancy-adaptive decode, retirement,
and reclaim. The first attempted audit is explicitly rejected: its stagger
helper still sent blocking requests and zero configured warmups also made its
generic gate false, so it proved only HTTP queue arrival.

The storage blocker remains real. All offered widths pass 8K/request on W7900,
but 8.25K fails resident preparation because packed AR direct no-mirror INT8
attention is not admitted. The <=8K route retains BF16 mirrors and cannot be a
compact-INT8 capacity claim. The telemetry and basic live-admission/reclaim
items are closed.

The approved next campaign is
[`QWEN38-INT8-KV-CONTINUOUS.md`](QWEN38-INT8-KV-CONTINUOUS.md):

1. `IKV-C0` integrates current origin and adds fail-closed artifact/backend/
   target/quant/layout/scale capability identity;
2. `IKV-C1` proves no-mirror c2/c4 through a declared serial c1-per-row route,
   including shifted block-table-aware prefill;
3. `IKV-C2` adds the true row-batched INT8 split-K producer/reducer;
4. `IKV-C3/C4` shares/models prefill ownership and admits KV, scales, mirrors,
   workspace, oracle, graphs, and reserve before allocation;
5. `IKV-C5/C6` closes cancellation, grow/shrink, overload/recovery, explicit
   source telemetry, and artifact-scoped quality/capacity;
6. `IKV-C7` runs matched BF16/serial/direct economics and promotes or rejects.

Evidence:
[`Qwen3.8 actual context, W7900 quality, and concurrency frontier`](../benchmarks/results/2026-08-16-qwen38-27b-actual-context-quality-w7900.json).

### W7900 coding-agent evaluation conclusions (A1–A6)

The frozen A1–A6 program is complete for Qwen3.6-35B-A3B UD-Q4_K_M/BF16-KV
on gfx1100. These are the concurrency and serving conclusions; detailed
commands and raw measurements remain in `WORKLOG.md` and the linked artifacts.

| Stage | Measured conclusion | Retained decision |
| --- | --- | --- |
| A1 deterministic baseline | Exact active-SSE goodput at C1/C4/C8 is **16.239/15.995/16.020 tok/s** for small, **15.100/15.231/15.036** for growing, and **4.127/4.629/4.339** for medium. Only medium C4 provides a material gain (**1.122x C1**); short/growing wider groups are flat. | Preserve c1 at low occupancy and require a medium-C4 guard for wider-route promotion. [`A1 artifact`](../benchmarks/results/2026-07-21-w7900-agentic-a1-repeated-baseline.json). |
| A2 prefix reuse | Deterministic active/completed p2048/p8192 state/KV/lifecycle is exact, but paired radix C1 goodput regresses **64.19%/65.63%/26.64%** and buffered tool-ready p50 worsens **181.90%/196.09%/38.81%** for small/growing/medium. | Keep `HIPENGINE_PREFIX_CACHE=off`; radix remains an explicit diagnostic, not a production concurrency feature. [`A2 decision`](../benchmarks/results/2026-07-21-w7900-agentic-a2-prefix-decision.json). |
| A3 native GPU sampling | The gfx1100 sampler is fixed-seed/state/KV/API exact and native rows reduce full-vocabulary logits D2H to zero. The frozen auto-tool route nevertheless fails turn-1 strict-envelope validity; all valid specific/required tool rows remain explicit host fallback. | Keep `HIPENGINE_QWEN35_NATIVE_SAMPLER=0`; no C1/C4/C8 timing or speedup claim is eligible until a native-eligible tool-valid route exists. [`A3 blocked artifact`](../benchmarks/results/2026-07-22-w7900-agentic-a3-native-sampler-blocked.json). |
| A4 routing/SLO | All **8 candidates x 3 balanced repetitions** complete (**288 requests / 8,640 IDs**), but no candidate passes every exactness and SLO gate. The package control misses TTFT p95 once (**10.983 s > 10 s**); faster alternatives produce nine late p512/d48 ID divergences after 20–24 exact IDs. | Keep `protect_decode:256/burst-1` and the zero-ms window. Diagnostic gains up to **63.81%** are invalid for promotion. [`A4 decision`](../benchmarks/results/2026-07-22-w7900-agentic-a4-routing-decision.json). |
| A5 pressure/soak | After closing admission, disconnect, cancellation-drain, and pre-start reservation races, all **9/9** workloads pass over **122 requests**: 108 completions, 12 exact retryable rejects, one disconnect, and one deadline. Queue/KV/stream/graph/workspace memory is bounded, the 80-second soak is 40/40 exact, GPU0 is exclusive, and final ownership is zero. | Retain the corrected bounded lifecycle and unchanged package defaults. This is bounded SLO/reliability evidence, not a multi-day or comparative performance claim. [`A5 closure`](../benchmarks/results/2026-07-22-w7900-agentic-a5-pressure-soak-closure.json). |
| A6 automatic-tool quality | Only **10/48** turns complete successfully; valid-call/correct-tool is **18/48**, exact arguments/external-oracle pass is **16/48**, safe patch success is **0/6**, and external-test selection is **8/8**. Repeated outputs are exact, so failures are reproducible rather than timing noise. | Autonomous tool-call formation is the dominant production blocker. This is a no-performance synthetic quality diagnostic. [`A6 artifact`](../benchmarks/results/2026-07-22-w7900-agentic-a6-broad-quality.json). |

The A4 row is pre-fix historical evidence, not a current numerical-drift
candidate. Commit `29a0c75d6` repaired authoritative-token/position publication
through the width transition and a clean follow-up passed 560/560 layer
comparisons plus 360/360 IDs. Its old `+63.81%` screen remains invalid; the
current campaign requires a fresh post-fix strict/production A4 SLO baseline.

The retained gfx1100 coding-agent configuration is therefore cache off, native
sampler off, `protect_decode:256/burst-1`, and a zero-ms generation window.
Wider execution, prefix reuse, or native sampling may be promoted only after the
same frozen correctness and SLO gates pass; diagnostic speed cannot override
invalid IDs or tool envelopes.

The next closure order is:

1. **Completed:** model-general tokenizer-aware constrained tool/JSON decoding,
   including exact dynamic/forced close-queue handling and fail-closed fallback;
2. reproduce and repair the late A4 p512/d48 physical-width transition while
   comparing every survivor's state and logical block-table-ordered live KV;
3. rerun the unchanged A3 and A4 protocols, promoting a default only if every
   correctness, ownership, C1, medium-C4, and SLO gate passes.

The completed first item masks tokenizer-decoded candidates before processed
argmax or host sampling. Tool-enabled chat admits either normal text in `auto`
or one canonical `<tool_call>` envelope with a declared name and root-object
JSON arguments; `required` and specific-tool requests admit only the envelope.
EOS is withheld until the selected branch is valid, partial close markers and
budget-fitting deterministic suffixes use the existing forced-token queue, and
every forced token is revalidated against the same grammar. Missing tokenizer
callbacks, token-decode failures, empty token text, and an empty constrained
candidate set fail closed. Native GPU sampling and raw-target MTP remain blocked
for these rows, so this is a host processed-logits correctness path, not an A3
performance promotion. JSON Schema values still use the existing strict
post-generation validator; this unit enforces envelope/name/root JSON syntax
and does not claim schema-guided argument quality or a rerun of A6.

Do not tune seeds, token IDs, prompts, candidate ordering, or grammar branches
to these frozen outputs. The repair must generalize across the committed
repository/general/multilingual quality families and category heldouts.

### Model/backend coverage

| Model path | Backend | Current c>N status | Production behavior | First missing gate |
| --- | --- | --- | --- | --- |
| GGUF Q4_K_M / BF16 KV | gfx1151 | `retained` direct native-c2/c4/c8 graph model steps, continuous membership, honest arbitrary-C lowering, and occupancy-adaptive OpenAI scaling: F0 physical-c2 repair passes 10,240/10,240 all-layer comparisons while prior 188,080 direct/category and 134,160 E3 comparisons remain retained; F3 singleton-indexed GDN refreshes direct c1/c2/c4/c8 to 50.335/78.552/108.050/133.251 aggregate tok/s with repeatable exact trajectories and 748 packed-native / 0 row-local / 0-copy c8 dispatches; the clean current p512/d128 real-Uvicorn packet is 44.137/59.258/74.105/83.771 blocking and 43.307/58.579/72.872/81.609 exact SSE at c1/c2/c4/c8, with delayed C8 66.563, all 117 rows exact, and one physical C8; F4 selects scoped package-default `fair:256`, passes all nine real-Uvicorn workloads, and sustains 120/120 soak requests at 43.314 exact SLO-goodput tok/s; F5 passes repeated seeded sampled blocking/SSE c4 and `n=3`, exact generated-token/logprob reconstruction, stop/request-EOS, strict tool and structured-result validation, packed c2/c4 route gates, memory recovery, and full owner drain with zero serial/resident fallback; its opt-in active-current p256+s1 prefix slice is byte-exact and moves continuation TTFT 249.269 -> 21.188 ms (11.765x) while saving one 5 MiB live page, and completed-source snapshot restore is byte-exact through source reset plus refcounts 1->1->2->1->0, moving TTFT 249.446 -> 22.013 ms (11.332x) at 72,089,600-byte cache residency | Public blocking calls and OpenAI SSE share one configured model-owning loop with reusable c8-capable sessions and real BF16 device KV; occupancy selects physical c1/c2/c4/c8 while stable request/state/KV ownership is unchanged; backend c1 reaches 95.625% of same-process direct; gfx1151 packed AR selects exact singleton-indexed GDN with segmented fallback and gfx1100 stays segmented pending transfer; active-current and completed-source greedy exact-boundary prefix reuse are explicit `radix` opt-ins and execute shared suffixes through exact c1; completed mode retains bounded cache-owned device snapshots and explicit eviction; C>8 remains multiple declared groups, never native c9/c13 | Reduce partial-width burst ramp and steady server/stream overhead while keeping the primary SSE and F4 SLO gates; broaden historical boundaries/LRU pressure and sampled prefix reuse; transfer sampled/API and prefix economics to gfx1100 separately |
| GGUF Q4_K_M / INT8 KV | gfx1151 | `retained` for explicit short-context uniform `int8_per_token_head` c1/c2/c4/c8 continuous serving: clean corrected-window p512/d128 blocking is **44.225/60.598/74.631/83.408 tok/s**, exact SSE is **42.759/55.128/71.284/81.140**, all **117/117** server rows are exact, delayed C8 is **65.034 tok/s**, and the full 11-prompt p512/d8 gate passes 99/99 positions at KL=0 / 100% top-1 | The shared owner carries INT8 K/V + FP16 per-token/head scales + bounded BF16 attention mirrors through dynamic allocation, packed import/scatter, prefill, decode, reclaim, and policy-locked server preparation for rounded contexts <=8192; C8 drains 65/65 owners, zero refs/pins, all slots free, and packed workspace to zero | Remove the bounded mirrors and independently gate direct INT8 attention before claiming memory savings; close strict high-C SLO; gate tail4, longer contexts, gfx1100, sampling, prefix reuse, and external serving separately |
| GGUF Q4_K_M / BF16 KV | gfx1100 | `retained` current source-pinned direct c1/c2/c4/c8 at **98.263/148.944/209.304/266.479 aggregate tok/s** with c1 HIP graph, c2/c4/c8 canonical PM4, exact repeatable trajectories, zero fallback/callback error, full retirement, and zero tracked bytes after close; real OpenAI SSE c1/c8/c9/c13/serial-c13 is **72.169/158.542/137.001/129.507/55.868 tok/s**, all **189/189 rows / 24,192 IDs** are exact, and live C8→C13 is **122.860 tok/s / 1,664 IDs** exact with zero fallback and complete ownership recovery | Public blocking calls and OpenAI SSE share one configured model-owning loop, reusable sessions, bounded queues, real BF16 device KV, and arbitrary-C manifests; registry-owned context capacity admits 13 logical short-context rows through declared physical groups no wider than C8, while unspecified or >768-position contexts and explicit MTP retain the established C4 cap; C>8 remains multiple declared groups | Broaden context/pressure and cancellation coverage around the short-context C13 boundary; normal sampling, prefix reuse, compaction tuning, and any higher resident cap remain separately gated |
| GGUF Q5_K/Q6_K/Q8_0 / BF16 KV | gfx1100/gfx1151 | Not executed end to end under c>N | c1 | Run quant-specific direct, profiler, lifecycle, and scaling gates |
| PARO W4 / BF16 KV | gfx1151 | `retained`: explicit direct native-c2/c4/c8 is **79.237/100.209/99.943 aggregate tok/s** with all 5,754 direct IDs exact; blocking OpenAI F1 is **47.124/51.962/60.323/61.253** at c1/c2/c4/c8 with 68/68 exact rows; real FastAPI SSE c1/c2/c4/c8/serial-c8 is **36.327/38.666/42.471/41.487/35.633** with 100/100 exact rows, plus 72/72 exact c8 stress rows | Public blocking/OpenAI requests share one fixed-capacity owner with chunked prefill, stable device-state/KV slots, authoritative generated-token accounting, profile partitions, and package-default native c2/c4/c8; explicit `=0` rollback flags preserve exact serial fallback | Close gfx1100 owner c4/c8 independently; broaden sampled groups, context/KV coverage, and only then consider graph replay |
| PARO W4 / BF16 KV | gfx1100 | `retained` for the explicit direct c2 model step: canonical selected-batch passes p512/d128 repetition, all-layer hidden/Conv/GDN/KV/NumPy-context, uniform/ragged EOS+cancel c2→c1 with inactive state/KV immutability, the ten-prompt category/heldout suite, primitive and profiler gates, and zero fallback layers; median is **121.923 aggregate tok/s**, **+5.09% vs c1** and **+20.81% vs serial c2** | Public blocking/OpenAI sessions remain width-1; the explicit native-c2 retained/default route resolves selected-batch, while grouped-compact remains an exact slower diagnostic | Generalize one physical c4/c8 algorithm without stacking c2 groups, then attach retained widths to the shared model-owning loop |
| PARO W4 / INT8 KV | gfx1100/gfx1151 | Not started | Width-1 | BF16 native path first |

### Implemented scaffolding — not production-loop evidence

The following components exist and have focused CPU/host tests:

- `ResidentBatchScheduler` request ids, physical slots, pending/admitted state,
  active masks, finish reasons, and reclaim callbacks;
- `ResidentEngineLoop.submit/poll/cancel` around an abstract runner;
- prompt-list and `n>1` API lowering plus per-request output queues;
- `ChunkedKVPool` host-model checks plus `DeviceChunkedKVPool` callback-owned
  real device growth, shrink, refcounts, graph pins, and stable block identity;
- `RadixCache` host-side prefix indexing and copy-on-write metadata;
- per-row sampling parameter blocks and EOS/reclaim metadata;
- graph-bucket, request, and pool observability schemas;
- Prometheus endpoint plumbing.

The gfx1100 GGUF D4 path now owns real device state, request-sized BF16 KV, and
bounded OpenAI token delivery through live prefill/decode/reclaim transitions.
It proves requests can enter, leave, disconnect, time out, reuse sessions/pages,
and drain through app shutdown while neighbors remain exact. D5 exports the
full live-loop metric set from one lock-consistent snapshot. E3/F1 retain honest
arbitrary-C lowering plus repeated real OpenAI burst scaling and one exact
c8→c13 live-admission trace. The unchanged backend-neutral loop independently
passes gfx1151 E1 admission, masked joined decode, disconnect/reclaim, row-owned
streaming, real OpenAI SSE, request/KV metrics, fallback accounting, and final
shutdown ownership while survivor state/KV remains independent-c1 exact.
Both GGUF backends retain continuous-membership correctness and real server
throughput scaling. PARO implements the same runner contract without a PARO-only
scheduler; gfx1151 now retains live admission/reclaim, repeated real OpenAI
blocking/SSE scaling, and package-default c2/c4/c8, while gfx1100 owner symmetry
remains open.

### Result pointers

- Host scaffolding and its limitations: `WORKLOG.md`, **2026-07-13 — Re-baseline
  PARO and GGUF concurrency**.
- gfx1151 GGUF natural-suite token equality: the same 2026-07-13 WORKLOG entry
  and `benchmarks/results/2026-07-13-gfx1151-gguf-natural10-cn-token-equality.json`.
- gfx1151 GGUF exact packed state/KV lifecycle: `WORKLOG.md`, **2026-07-13 —
  Make GGUF packed AR state/KV exact through c4 lifecycle**, and
  `benchmarks/results/2026-07-13-gfx1151-gguf-packed-ar-exact-lifecycle.json`.
- gfx1151 PARO selected-batch c2 retained transfer: `WORKLOG.md`, **2026-07-18
  — Retain gfx1151 PARO selected-batch c2**, and
  `benchmarks/results/2026-07-18-gfx1151-paro-g2-selected-batch-c2-retained.json`.
- gfx1151 PARO true physical c4/c8 direct retention, resident OpenAI ownership,
  and G5 blocking/SSE package-default promotion: `WORKLOG.md`, **2026-07-18 —
  Retain gfx1151 PARO direct native c2/c4/c8**, **Close gfx1151 PARO resident
  OpenAI correctness**, and **Retain gfx1151 PARO resident server scaling**, with
  `benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json`,
  `benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json`,
  `benchmarks/results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json`,
  `benchmarks/results/2026-07-18-gfx1151-paro-g5-c8-sse-repeatability.json`, and
  `benchmarks/results/2026-07-18-gfx1151-paro-g5-default-openai-c4.json`.
- gfx1151 GGUF unchanged native-c2/c4/c8 direct correctness, retained
  profiler/scaling, and live-loop symmetry: `WORKLOG.md`, **2026-07-17 — Retain
  gfx1151 GGUF direct concurrency correctness**, **Retain gfx1151 native-c8
  profiler and scaling**, and **Close gfx1151 E1 live-loop symmetry**, plus
  `benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json`,
  `benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json`,
  and `benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-live-loop-closure.json`.
- Current-main gfx1151 GGUF F0 physical-c2 repair and focused direct/server/
  profiler recertification: `WORKLOG.md`, **2026-07-19 — Repair gfx1151 GGUF
  physical-c2 long-horizon exactness** and **Re-certify current-main gfx1151
  GGUF serving**, plus
  `benchmarks/results/2026-07-19-gfx1151-gguf-f0-c2-fixed256-correctness.json`
  and
  `benchmarks/results/2026-07-19-gfx1151-gguf-f0-current-main-recertification.json`.
- gfx1151 GGUF F2 occupancy-adaptive c1/c2/c4/c8 owner retention:
  `WORKLOG.md`, **2026-07-19 — Retain occupancy-adaptive GGUF serving**, and
  `benchmarks/results/2026-07-19-gfx1151-gguf-f2-occupancy-adaptive-serving.json`.
- gfx1151 GGUF F4 production load/SLO and F5 sampled OpenAI API closure:
  `WORKLOG.md`, **2026-07-19 — Retain gfx1151 GGUF production load SLO** and
  **Close gfx1151 GGUF sampled OpenAI API paths**, plus
  `benchmarks/results/2026-07-19-gfx1151-gguf-f4-production-load-slo.json` and
  `benchmarks/results/2026-07-19-gfx1151-gguf-f5-sampled-openai-api.json`.
- gfx1151 GGUF explicit short mirrored-INT8 continuous ownership, full
  multi-prompt fidelity, C1/C2/C4/C8 scaling, and delayed-C8 lifecycle:
  `WORKLOG.md`, **2026-07-19 — Retain mirrored INT8 GGUF continuous serving**,
  and `benchmarks/results/2026-07-19-gfx1151-gguf-mirrored-int8-continuous-concurrency.json`.
- gfx1151 E3 honest C13 eager/graph grouping, middle-hole cancellation/admission,
  inactive state/KV immutability, and explicit optional-compaction graph/resource
  safety: `WORKLOG.md`, **2026-07-18 — Retain gfx1151 E3 arbitrary-C
  correctness**, and
  `benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json`.
- gfx1100 Phase-A c1/serial-c2/c4 controls and package-c4 route inventory:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF Phase-A controls**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-phase-a-controls.json`.
- gfx1100 B1 c2/c4 primitive gates, package-policy inventory, and cached-only
  packed-c2 route trace: `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF B1
  primitive and route preflight**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b1-preflight.json`.
- gfx1100 B2 strict-exact c2/c4, sparse, ragged, and all-layer lifecycle:
  `WORKLOG.md`, **2026-07-16 — Retain clean gfx1100 GGUF B2 direct lifecycle**,
  and `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b2-direct-lifecycle.json`.
- gfx1100 B3/B4 all-row 512/128 and prompt-diversity closure: `WORKLOG.md`,
  **2026-07-16 — Close gfx1100 GGUF B3 standard lifecycle** and **Close
  gfx1100 GGUF B4 prompt diversity and Phase B**, plus
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b3-standard-lifecycle.json`
  and `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-b4-category-lifecycle.json`.
- gfx1100 C1 runtime/profiler hybrid census: `WORKLOG.md`, **2026-07-16 — Close
  gfx1100 GGUF C1 hybrid-boundary census**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c1-hybrid-census.json`.
- gfx1100 C2 indexed recurrent closure and repeated Phase-B equality:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF C2 recurrent linear
  attention**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c2-recurrent-closure.json`.
- gfx1100 C3 c-aware family census and copy-free steady metadata closure:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF C3 model boundaries**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c3-model-boundaries-closure.json`.
- gfx1100 C4 state-bound graph replay and retained direct native-c4 scaling:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF C4 native graph scaling**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json`.
- gfx1100 D1 long-lived GGUF model/session ownership and clean cross-call c2
  equality: `WORKLOG.md`, **2026-07-16 — Attach D1 to persistent GGUF
  model/session state**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d1-resident-model-runner-closure.json`.
- gfx1100 D2 bounded live admission, packed-group cancellation, max-token
  retirement, exact session reuse, and survivor state/KV equality: `WORKLOG.md`,
  **2026-07-16 — Close gfx1100 GGUF D2 live scheduling**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d2-live-lifecycle-closure.json`.
- gfx1100 D3 real BF16 device-KV admission, atomic high water, graph-pinned
  pointer lifetime, exact tracked memory contraction, and 3→6→3→12→3 lifecycle:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF D3 device KV**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d3-device-kv-pool-closure.json`.
- gfx1100 D4 bounded OpenAI streaming, full-queue live admission, row-local
  disconnect/timeout, SSE completion, and two-phase shutdown reclaim:
  `WORKLOG.md`, **2026-07-16 — Close gfx1100 GGUF D4 OpenAI streaming**, and
  `benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d4-openai-streaming-closure.json`.
- gfx1100 D5 live scheduler/latency/KV/graph/route observability:
  `WORKLOG.md`, **2026-07-17 — Close gfx1100 GGUF D5 live observability**, and
  `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-d5-live-observability-closure.json`.
- gfx1100 E2 true physical-c8 masks, graph replay, lifecycle equality, profiler
  census, and retained direct scaling: `WORKLOG.md`, **2026-07-17 — Close native
  c8 ragged, cancellation, and p512/d128 equality** plus **Retain clean native
  c8 profiler and scaling**, and
  `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-correctness.json`
  plus `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json`.
- gfx1100 E3 arbitrary-C eager/graph equality, sparse cancellation/admission,
  inactive state/KV immutability, and optional-compaction resource/graph safety:
  `WORKLOG.md`, **2026-07-17 — Retain clean gfx1100 arbitrary-C correctness** and
  **Retain clean optional-compaction correctness**, plus
  `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json`.
- gfx1100 F1 real OpenAI p512/128-output burst and c8→c13 live-admission
  scaling: `WORKLOG.md`, **2026-07-17 — Retain real OpenAI arbitrary-C server
  scaling**, and
  `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json`.
- Current gfx1100 context-qualified admission, source-pinned direct refresh, and
  real OpenAI C1/C8/C9/C13 plus live C8→C13 packet: `WORKLOG.md`, **2026-08-08 —
  Restore short-context C8/C13 OpenAI topline**, and
  `benchmarks/results/2026-08-08-gfx1100-context-scoped-c8-server-refresh.json`.
- gfx1151 E1/E3/F1 direct native-c8, continuous membership, arbitrary-C plus
  explicit-compaction correctness, and real OpenAI p512/128-output burst/live
  scaling: `WORKLOG.md`, **2026-07-17 — Retain gfx1151 native c8 scaling** plus
  **2026-07-18 — Preserve gfx1151 packed attention after the concurrency merge**
  and **Retain gfx1151 real OpenAI arbitrary-C server scaling**, with
  `benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json`,
  `benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json`,
  and
  `benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json`.
- gfx1100 PARO native-c2 first divergence: `WORKLOG.md`, **2026-07-17 —
  Localize the first W7900 PARO native-c2 divergence**, and
  `benchmarks/results/2026-07-17-gfx1100-paro-g2-native-c2-first-divergence.json`.
- gfx1100 PARO dense-order c2 token/hidden/state/KV progress and exact-kernel
  trace: `WORKLOG.md`, **2026-07-17 — Close PARO c2 short-context arithmetic
  drift**, and
  `benchmarks/results/2026-07-17-gfx1100-paro-g2-native-c2-dense-order-progress.json`.
- gfx1100 PARO grouped-compact c2 selected-MoE stage closure: `WORKLOG.md`,
  **2026-07-17 — Close PARO grouped-compact c2 arithmetic**, and
  `benchmarks/results/2026-07-17-gfx1100-paro-g2-grouped-moe-stage-closure.json`.
- gfx1100 PARO c2 direct/lifecycle/repetition/scaling closure: `WORKLOG.md`,
  **2026-07-18 — Close gfx1100 PARO native-c2 correctness**, and
  `benchmarks/results/2026-07-18-gfx1100-paro-g2-native-c2-correctness-scaling.json`.
- gfx1100 PARO selected-batch c2 retained promotion: `WORKLOG.md`, **2026-07-18
  — Retain gfx1100 PARO selected-batch c2**, and
  `benchmarks/results/2026-07-18-gfx1100-paro-g2-selected-batch-c2-retained.json`.
- Historical PARO c1-c8 catalog and lifecycle: `docs/BENCHMARK.md` §PARO c1-c8
  exact concurrency matrix.
- Historical c>N graph replay and output-tiled GEMV: `WORKLOG.md`, **2026-06-08
  — C3.0b-5 (piece D): c>1 decode graph capture + replay (correctness GREEN)**
  and **2026-06-08 — concurrency C3.0c output-tiled GEMV decode throughput: NULL
  (dispatch-bound)**; neither is a retained production c>N row.

## Architectural contract

### One owner

The production engine loop is the only owner of:

- admission and pending queues;
- request-id ↔ physical-slot maps;
- active masks and batch shape;
- prompt cursors and decode positions;
- canonical KV allocation and mutation;
- Conv/GDN recurrent state and model scratch;
- sampling, completion, streaming events, and reclaim;
- graph-bucket capture/invalidation;
- dynamic-pool grow/shrink decisions.

Server adapters may validate and enqueue. They may not own a second scheduler or
hold a request-lifetime model lock.

### Canonical batch state

The canonical internal shape is batch-aware even at C=1:

```text
request_ids[C]                 stable logical identity
slots[C]                       physical resident slots
active_mask[C]
token_ids[C]
positions[C]
context_lengths[C]
finish_flags[C]
hidden[C, hidden_size]
logits[C, vocab_size]          or row-tiled equivalent
conv_state[num_linear_layers, slots, ...]
gdn_state[num_linear_layers, slots, ...]
kv_spans[layer, C]             KVLiveSpans
sampler_params[C]
```

Request identity and slot identity must never be conflated. Kernels receive row
maps or state indices when routed lanes do not map directly to physical slots.

### Runner contract

A production model runner must expose equivalent operations to:

```python
prefill_batch(batch_state, prompt_slab, *, commit) -> PrefillResult
decode_batch(batch_state, *, commit) -> DecodeResult
compact_batch(batch_state, new_row_map) -> None
reclaim_slots(slot_ids) -> None
```

The exact Python names may differ. Required semantics do not:

- one call owns one batch-shaped model transition;
- canonical state mutates only at commit points;
- every fallback is explicit in execution metadata;
- GGUF and PARO implement the same scheduler-facing contract;
- backend-specific launch geometry comes from package capabilities/registry
  variants, never engine branches.

### Work classes and commit points

Each loop tick performs one work class:

| Work class | Action | Commit |
| --- | --- | --- |
| `RECLAIM` | Release finished/cancelled row state and KV | Free slots/pages; emit completion |
| `ADMIT` | Move pending requests into free slots | Publish slot/request map |
| `PREFILL_CHUNK` | Advance one or more prompt chunks | Prompt cursor, model state, KV |
| `DECODE_STEP` | Advance every decode-ready active row once | Token, position, state, KV |
| `VERIFY_STEP` | Future SpecDec target verification | Accepted rows only |
| `PACK_STEP` | Future DMS compaction | Compact KV only |

Default ordering is `RECLAIM → ADMIT → choose(PREFILL_CHUNK, DECODE_STEP)`.
The generic policy remains `protect_decode`; registry-selected model/backend/
quant defaults may refine it only after an independent production load gate.
The gfx1151 Qwen GGUF Q4_K_M F4 packet supplies that evidence for `fair:256`,
and the gfx1100 Q4_K_M path now declares the same scoped default
(`fair:256/burst-1`) after the W7900 Qwen3.8-27B 16K server measured width-4
packed AR decode under `fair` while the `protect_decode` default serialized
every concurrent request's decode before the next prefill (zero packed decode
steps at c4/c8). The explicit env pin remains the rollback for configurations
that must stay on `protect_decode` (the A4 frozen UD-Q4_K_M gates).

Cancellation, disconnect, EOS, max-tokens, timeout, and errors all converge on
`RECLAIM`. Mid-kernel cancellation never mutates canonical state.

### KV contract

- `KVLiveSpans` is the only attention/KV-write ABI.
- The scheduler allocates KV; kernels and model code do not.
- A logical block id remains stable for its lifetime.
- While a block is resident and referenced by a captured graph, its device-memory backing
  pointer is pinned.
- Pool shrink may free only unreferenced, graph-unpinned tail chunks.
- Future KV tier movement must update allocator indirection and invalidate or
  rebind affected graph buckets before execution. “Stable block id” does not
  mean a captured stale pointer remains valid after a tier move.
- Prefix sharing remains default-off until a real model loop passes shared-prefix
  lifecycle and savings gates.

This resolves the apparent conflict between current append-only device-memory allocation
and future KVTC pointer movement.

### Graph contract

Graph buckets include every axis that changes captured work:

```text
(model plan, backend, execution profile + variant-manifest hash, C bucket,
 context/page bucket, active mask, KV dtype/layout, prefill/decode mode,
 top-k/experts, replay length)
```

A cache lookup hit is not a replay claim. Retained evidence requires a replayed
kernel path in `rocprofv3`, positive replay counts, and matching bucket metadata.
Bucket invalidation is required after pointer-changing pool/tier events.

### Sampling contract

The active group may contain incompatible row parameters:

- temperature;
- top-k and top-p;
- repetition penalty;
- seed/RNG state;
- stop tokens;
- max-new-token budget.

Greedy exactness is the bring-up oracle. Normal sampling becomes production only
when deterministic seeded row-local tests and distribution checks pass. The
sampler may use multiple kernels, but not one complete LM-head/sampler host loop
per row in a native throughput claim.

### Backpressure and fairness

The loop enforces:

- maximum active requests;
- maximum pending requests;
- current KV admission capacity;
- maximum prefill tokens per tick;
- bounded streaming queues;
- explicit admission blocker reasons.

A slow streaming client pauses only its row at a commit point. It must not block
other rows or hold a model/session lock.

## Correctness and evidence gates

`docs/BENCHMARK.md` is authoritative. This section summarizes the concurrency
requirements so roadmap items have local exit criteria.

### Gate 1 — primitive kernels

For every backend, row count, KV dtype, execution profile, and kernel family
used by the runner:

- KV append destination/ownership metadata is exact; strict value bytes match
  their oracle, while production values satisfy the declared primitive
  tolerance;
- batch attention vs independent strict c1 satisfies the declared strict or
  production tolerance;
- CPU/NumPy outer oracle tolerance passes;
- repeated A/A determinism passes where required;
- `rocprofv3 --kernel-trace` confirms the intended kernel, workgroup, resources,
  and positive duration.

### Gate 2 — direct model profile gate

For greedy sampling and SpecDec disabled:

- request/slot/token/position/mask/`KVLiveSpans`/state ownership and lifecycle
  metadata are exact;
- logits and all recorded states are finite;
- `strict` generated tokens, hidden/state boundaries, and live K/V prefixes
  match N independent strict c1 sessions at the declared exact boundaries;
- `production` compares c>N full logits against strict at identical teacher-
  forced contexts and passes calibrated mean/p95/p99/max KL, top-1 by category/
  shape/transition, same-schedule determinism, same-width neighbor isolation,
  BF16-relative checks where available, and task non-inferiority; numerical
  Conv/GDN/KV values may drift but ownership and transactions may not;
- `batch_invariant` preserves the fixed-seed request result across supported
  slots, neighbors, widths, admission order, cancellation, and compaction; and
- no serial bridge or undeclared fallback is present.

Cross-width free-running generated-ID equality is binding for strict and
batch-invariant and diagnostic for production.

### Gate 3 — lifecycle and isolation

At minimum:

- steady c2 and c4;
- ragged prompt lengths;
- front, middle, and tail row retirement;
- c4→c3→c2→c1 sparse-slot survival;
- cancellation of one row while neighbors continue;
- one newly admitted row while existing rows continue;
- prefill/decode interleaving;
- standard all-row prompt 512 / decode 128.

A short-horizon token-only suite is supportive evidence, not lifecycle closure.
Every profile requires exact maps, positions, masks, `KVLiveSpans`, commit/
rollback destinations, reclaim, and neighbor isolation. Strict/batch-invariant
add their result-equality contract; production uses strict-teacher numerical
quality at the transition contexts.

### Gate 4 — native execution

Artifacts and profiles must prove:

- one native group has the claimed width;
- no complete c1 session/layer replay per row;
- no host per-row metadata allocation/copy in steady decode;
- c-aware projection, attention, state, MoE, and sampler paths are named;
- fallbacks are empty or explicitly disqualify the row;
- graph replay evidence is real when replay is claimed.

### Gate 5 — retained performance

Each retained c=N row records:

- execution profile/schema, selected and strict-fallback manifest hashes,
  teacher source, and whether generated-ID equality is binding or diagnostic;
- exact model fingerprint, quant, KV dtype, backend, hardware, command, and clean
  source revision;
- c1 and serial-bridge baselines from the same protocol;
- aggregate tok/s and per-request tok/s;
- aggregate/c1, per-request/c1, and aggregate/serial ratios;
- TTFT and inter-token-latency p50/p95;
- occupancy over time and row-count transitions;
- tracked allocator/KV/workspace memory;
- profiler family durations and launch counts;
- admission/completion timestamps and finish reasons.

A native row must beat the serial bridge and c1 aggregate to become a scaling
claim. There is no arbitrary fixed speedup target for an individual qualified
route; exact and production-profile non-regressive wins are kept under the
repository performance policy. Changing the public default profile separately
uses the material SLO-goodput/c1 guard in `EXECUTION-PROFILES.md`.

### Prompt coverage

Do not tune concurrency to one repeated-token prompt. Before promotion, run:

- deterministic raw-token 512/128 fixtures for strict equality or production
  strict-teacher comparison;
- all categories in `benchmarks/prompts/mtpbench-code-general-ja.jsonl`;
- ragged and sparse lifecycle fixtures;
- category-heldout prompts when sampling or route selection changes.

## Roadmap

The ordering is deliberate: transfer the strongest existing GGUF correctness
anchor to W7900, make and measure one c4 group as fully native, attach the live
scheduler on W7900, then generalize across backends and widths. PARO follows the
same pipeline once the GGUF contracts are proven.

### Phase A — refresh the roadmap and baselines

**Objective:** establish current-HEAD W7900 controls without changing kernels.

Punchlist:

- [x] Replace the historical concurrency notebook with this roadmap.
- [x] Record clean gfx1100 c1 GGUF 512/128 package-default baseline.
- [x] Record gfx1100 c2/c4 serial-bridge controls with exact accounting.
- [x] Capture a current packed c4 route/dispatch inventory before modifying it.
- [x] Confirm ROCm/compiler/model fingerprint and cached-build discipline.
- [x] Add a compact branch kickoff entry to `WORKLOG.md`.

Exit: one compact baseline artifact names every current fallback and supplies the
control for Phase B. No c>N performance claim.

### Phase B — gfx1100 GGUF c4 exact lifecycle

**Objective:** transfer the gfx1151 packed Q4_K_M/BF16 correctness anchor to
W7900 before optimizing or integrating the live loop.

B1. Primitive and route preflight:

- [x] Run c2/c4 KV append and full-attention primitive gates on gfx1100.
- [x] Confirm the production gfx1100 package defaults used by packed prefill and
      decode, including router, device metadata, GDN, quant, and graph policy.
- [x] Trace one packed step and verify every intended registry route.

B2. Direct lifecycle oracle:

- [x] Run `scripts/gguf_packed_ar_state_oracle.py` on gfx1100 for steady c2/c4.
- [x] Run c4→c3→c2→c1 middle-hole retirement.
- [x] Run ragged `[512,64,64,64]` and short equal-length prompts.
- [x] Add/enable per-layer hidden capture for the packed route.
- [x] Compare every generated token, 30 Conv/GDN families, 10 live-KV families,
      and per-layer hidden outputs against independent c1.

B3. Standard protocol capacity:

- [x] Make all-row c4 prompt-512/decode-128 fit through a correct chunked packed
      prefill path. The old 768-row hidden-slab ceiling is not an acceptable
      substitute for the standard gate.
- [x] Prove no row is silently serialized because the packed slab is full.
- [x] Pass the complete 512/128 lifecycle gate with exact accounting.

B4. Prompt diversity:

- [x] Run the full 10-prompt category suite for at least three repeats.
- [x] Add heldouts if route selection or sampling behavior changes.

Exit: gfx1100 GGUF c4 qualifies as `exact_hybrid` for
tokens/hidden/state/KV/lifecycle. It must remain `performance_claim=false` until
C4 replay and scaling pass.

### Phase C — fully native GGUF c4

**Objective:** remove exact-hybrid row replay from one c4 model step while
preserving Phase B equality.

C1. Make the fallback boundary observable:

- [x] Emit one execution manifest per layer family: projection, Conv/GDN,
      full-attention, MoE/FFN, LM head, and sampler.
- [x] Count host row loops, per-row launches, metadata copies, synchronizations,
      and scalar fallbacks.
- [x] Add profiler buckets for packed native work versus exact row-local work.

Clean `88f10724` attributes one p512 steady c4 transition's 1,386 dispatches to
**840 exact-row-local** launches (480 projection, 240 Conv/GDN, 120 RMSNorm) and
**546 packed-native** launches. The exact-row-local bucket accounts for 41.10%
of instrumented GPU duration. The runtime manifest reports 30 host row-loop
sites/120 row iterations, nine H2D copies, one four-value D2H token read, two
synchronizations, and zero steady state import/scatter or scalar fallback. The
single profiled transition is a route census, not a throughput measurement.

C2. Close recurrent linear attention:

- [x] Add a RED fixture that captures the first packed-vs-c1 state/hidden drift.
- [x] Implement/register a c-aware exact Conv/GDN decode route for gfx1100.
- [x] Preserve c1 arithmetic order or document and pass the numerical contract.
- [x] Mutate packed state in place with slot/segment metadata; do not scatter a
      complete state slab after every step.
- [x] Keep the unfused primitive chain as the required fallback.

Clean `f6e8363e` profiles one p512 steady c4 transition as **756 packed-native**
dispatches and **zero exact-row-local** dispatches, down from C1's 1,386 total
(**-45.45%**) and 840 row-local launches. The manifest reports zero host
model-row loops/iterations, zero steady state import/scatter, and the same nine
small H2D copies plus one vector D2H read. Indexed Conv and segmented FP32-output
GDN each execute 30 times with zero scratch. Clean lifecycle/category gates at
`f6e8363e`/`799d29b9` pass **20,640/20,640** p512/d128 and **560/560** sparse
shrink hidden comparisons plus **1,350/1,350** category tokens and
**54,000/54,000** category hidden comparisons. The route remains
`exact_hybrid`; profiler durations are diagnostic and no throughput/native-c4
claim is made. See the compact [C2 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c2-recurrent-closure.json).

C3. Close remaining per-row model work:

- [x] Make full-attention append/context/O/post consume sparse c4 spans without
      complete row replay.
- [x] Confirm MoE routing, selected experts, expert dispatch, and reduction are
      truly c-aware at c4.
- [x] Make LM head and greedy sampler return one token per live row without
      full-vocab host readback.
- [x] Move steady-step metadata preparation to persistent device buffers.

Clean `db1ce640` profiles one p512 steady c4 transition as **749 packed-native**
dispatches and **zero exact-row-local** dispatches. Full attention launches 10
context plus 10 KV-write kernels at row extent four; all 40 MoE layers launch
32 selected lanes for gate-up/down plus one c4 combine; one c4 Q6 rowtile feeds
two row-wise argmax stages and a single four-i32 D2H read. The new registry-
resolved metadata producer replaces eight H2D uploads with one 64-thread,
16-VGPR, zero-scratch/LDS kernel, leaving exactly the token H2D and token-vector
D2H copy dispatches. Clean p512/c4/d128 and sparse c4→c1 gates pass
**20,640/20,640** and **560/560** layer-hidden comparisons with exact tokens,
Conv/GDN state, and live KV. The route remains correctness-only `exact_hybrid`;
C4 owns graph replay, repeated replay equality, scaling, and any native-c4 or
performance claim. See the compact [C3 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c3-model-boundaries-closure.json).

C4. Replay, measure, and prove:

- [x] Capture/replay at least c1/c2/c4 decode buckets after eager equality passes.
- [x] Pass Phase B again with replay enabled and disabled.
- [x] Trace the full c4 step and prove zero undeclared c1 layer/session fallback.
- [x] Run same-protocol c1/c2/c4 plus explicit chunked-c8 controls.
- [x] Report aggregate/per-request throughput, TTFT/ITL, memory, occupancy, and
      profiler family/launch accounting.
- [x] Compare native c4 against c1 and the serial bridge in one clean session.

Clean graph/eager p512/c4/d128 each pass **20,640/20,640** hidden comparisons
plus exact tokens, Conv/GDN state, and live KV. Clean sparse graph replay is
exact through c4→c3→c2→c1 (**560/560** hidden rows). The marker-sliced c4 graph
launch contains **747 packed-native / 0 exact-row-local / 0 copy** dispatches,
with one device-position metadata launch and one token-feedback commit. In one
clean shared-runner 1+3 packet, c1/c2/c4/chunked-c8/serial-c4 aggregate decode
is **84.907/126.909/184.993/183.900/84.140 tok/s**. Native c4 is **2.179x** c1
and **2.199x** serial-c4; its per-request rate is **46.248 tok/s**. The tradeoff
is explicit: c4 model-step ITL p50/p95 is **21.641/21.888 ms**, TTFT is
**2.027/2.031 s**, and four-session tracked/HIP-used peaks are
**23.396/23.823 GiB**. Chunked c8 remains two serialized c4 groups and is not a
native-c8 claim. See the compact [C4 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json).

Exit: satisfied for the direct gfx1100 GGUF c4 decode model step. Phase D still
owns production model-loop admission, fairness, cancellation, reclaim, and
streaming delivery.

### Phase D — gfx1100 production model-owning continuous loop

**Objective:** connect the proven and measured gfx1100 c4 GGUF model step to one
loop that survives API calls and changes membership while generation is active.

D1. Runner ownership:

- [x] Add one long-lived GGUF runner implementing the scheduler-facing prefill,
      decode, compact, and reclaim contracts.
- [x] Move model/session creation outside `generate()` calls.
- [x] Remove the request-lifetime generation lock from the model execution path.
- [x] Make `SubmitPollTextGenerator` adapt to the shared loop rather than create
      a loop per call.

Clean `285c20b4` W7900 evidence compares the old direct resident control, the
first shared-loop call, and a second call for two independent 16-token/d4 rows.
All three trajectories are exact (`[9707,9707,9707,9707]` and
`[9708,9709,9708,9709]`); all four preallocated session identities remain
stable and all four return idle after each call. The final route records three
native c2 model steps with no serial fallback. Focused host validation is **78
passed, 4 skipped** and includes context-pool rebuild, greedy packed/c1,
sampled and zero-token compatibility, commit, compact, and reclaim contracts. D1 does
not claim live admission: each synchronous caller still drives loop ticks, and
real prompt execution occurs at the final prefill commit rather than being
interleaved with live decode. See the compact [D1 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d1-resident-model-runner-closure.json).

D2. Real scheduling:

- [x] Admit a request while another row is decoding.
- [x] Interleave bounded prefill chunks with decode according to the selected
      policy.
- [x] Retire EOS/max-token rows and reuse slots without stopping live neighbors.
- [x] Cancel/disconnect one row without cancelling or corrupting its group.
- [x] Preserve deterministic token/state/KV equality across membership changes.

Clean `b51bd688` W7900 evidence uses four controlled p512 requests, 256-token
prefill chunks, the `protect_ttft` policy, and one c4 resident pool. A emits two
tokens before B is admitted; B executes `256+256`, joins A, runs one native c2
step, and is cancelled while packed state is dirty. C reuses B's exact session,
executes `256+256`, joins A, and retires at max-tokens after another native c2
step. D reuses that session again and is cancelled after its first 256-token
chunk. A remains live throughout and matches its independent c1 trajectory at
all eight tokens. A after B cancel, B before reset, A after C retire, C before
reset, A across D cancel, and final A each have zero Conv/GDN or live-KV hash
mismatches against the corresponding c1/stability oracle. All four sessions
return idle; focused host validation is **380 passed, 4 skipped** and includes
EOS, max-token, cancel, disconnect, and timeout reclaim paths. This advances only
the non-streaming gfx1100 GGUF model loop to `continuous_eq_ok`; D3 device KV
allocation and D4 server streaming remain open. See the compact [D2 closure
artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d2-live-lifecycle-closure.json).

D3. Device KV pool:

- [x] Connect scheduler admission to real paged KV allocation.
- [x] Grow only at admission barriers; shrink only idle, free, unpinned chunks.
- [x] Enforce high-water rejection without partial request mutation.
- [x] Audit allocator-visible and tracked current/peak memory.
- [x] Pass burst→steady→idle→burst with pointer/graph validity checks.

Clean `367cf7a5` W7900 evidence binds each p512/d2 request to three real
5,242,880-byte BF16 pages before slot publication. A state-bound graph pins all
three tail pages, replays the exact `[9708,9708]` c1 trajectory with zero
Conv/GDN/live-KV mismatch, and is invalidated before the tail shrinks. The
allocator executes **3→6→3→12→3 pages**, preserves every live representative
pointer, and regrows with fresh logical ids `6..14`; tracked first grow and
shrink deltas are exactly **15,728,640 bytes** and the 12-page pool high water is
**62,914,560 bytes**. Sampled HIP used bytes are retained separately and show
allocator granularity rather than exact byte-for-byte contraction. A real-HIP
three-page high-water probe rejects the next request before lease/slot
publication with pages/refcounts unchanged. The full A/B/C/D D2 lifecycle also
repeats through real allocation with every token and all six state/KV checks
exact, B→C reusing ids `3..5` plus the same backing/session, D regrowing ids
`6..8`, and the pool ending at three free unpinned pages. Focused clean host
validation is **419 passed, 4 skipped**. D3 adds no throughput claim and supports
request-sized allocation for BF16 KV only. See the compact [D3 closure
artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d3-device-kv-pool-closure.json).

D4. API and streaming:

- [x] Route both `LLM.generate()` and the OpenAI server through the same loop.
- [x] Stream token events per request without holding model/session locks.
- [x] Bound slow-client queues and isolate backpressure by row.
- [x] Support request cancellation, timeout, disconnect, and shutdown drain.
- [x] Keep non-streaming prompt-list behavior as a compatibility adapter, not a
      second execution architecture.

Clean `f03957cc` W7900 evidence fills A's two-chunk HTTP queue before B submits,
then admits and prefills B while A remains live and executes one native c2
A+B decode. A finishes all eight exact `[9707]` tokens; B's consumed `[9708]`
prefix and both rows' Conv/GDN/live-KV state match independent c1 before B
reclaims as `disconnect`. A real OpenAI completion emits four valid SSE payloads,
usage, completion, and `[DONE]`; a 1 ms request deadline emits
`deadline_exceeded` plus `[DONE]`. Forced shutdown trips the producer token and
returns HTTP active/queued counts to zero; the app's immediately following
long-lived runner close reclaims the still-resident three-page row and leaves
zero scheduler rows, sessions, or KV-pool ownership. Host validation covers
queue overflow isolation and graceful/forced ownership; the clean hardware gate
passes in **78.158 s**. This closes D4 at correctness-only `continuous_eq_ok`—it
adds no server throughput, TTFT, ITL, or concurrent-kernel claim. See the compact
[D4 closure artifact](../benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-d4-openai-streaming-closure.json).

D5. Observability:

- [x] Export pending/admitted/active counts and physical bucket occupancy.
- [x] Export prefill/decode/reclaim work counts and scheduler policy.
- [x] Export request queue, TTFT, inter-token, service, and completion latency.
- [x] Export KV current/high-water bytes, pages, refcounts, grow/shrink/failures.
- [x] Export graph hit/capture/replay/invalidation counts by bucket.
- [x] Include route/fallback manifests in benchmark artifacts.

Clean `7ab8eb3b` W7900 evidence scrapes the real Prometheus endpoint at c1, live
c2, post-reclaim, graph-capture, graph-replay, and graph-invalidation barriers.
A and B fill independent two-chunk queues, remain simultaneously admitted in
**2/4** physical slots with **6** request-owned pages, execute native c2, and
finish as length/disconnect with exact **16/16** and **4/4** token prefixes plus
zero state/KV mismatches. The completed rows populate all five bounded latency
families with queue/TTFT/ITL/service/completion counts **2/2/18/2/2**; these
values validate instrumentation and are not a performance claim.

A scheduler-owned c1 row then captures stable bucket `600fea75…` at the admitted
**24-replay** gfx1100 threshold. Its 24 graph tokens and final state/KV are
independent-c1 exact; `/metrics` records **1 capture, 24 hits/replays, 3 pinned
pages**, then **1 invalidation, 0 entries, 0 refcounted/pinned pages**. The
retained artifact carries the full bucket key, additive route/fallback counts,
and complete two-row packed execution manifest. All **12/12** closure checks
pass in **84.048 s**; host validation is **487 passed** plus **387 passed, 4
skipped**. This closes Phase D at observable correctness-only
`continuous_eq_ok` with no server throughput, concurrent-kernel, or native-c8
claim. See the compact [D5 closure artifact](../benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-d5-live-observability-closure.json).

Exit: live requests enter and leave one real gfx1100 GGUF model loop, with exact
lifecycle evidence and no per-call inner-loop ownership.

### Phase E — cross-backend GGUF and native c8

**Objective:** make GGUF concurrency a backend-capability path rather than a
one-device special case, then raise the native group width.

E1. gfx1151 symmetry:

- [x] Run the Phase B/C gates unchanged on gfx1151.
- [x] Register gfx1151-specific launch variants only where measurement requires
      them; none were required, and runner/scheduler code remains backend-neutral.
- [x] Replace the current gfx1151 exact-hybrid boundaries with the same native
      model-step contract.
- [x] Run the Phase D live-loop lifecycle gate unchanged on gfx1151.
- [x] Keep separate gfx1100 and gfx1151 artifacts and route catalogs.

Clean tracked `80bdf6a3` validates the merged shared-gfx11 implementation on the
Radeon 8060S without a gfx1151-specific runtime or kernel edit. Primitive
c2/c4/c8 is exact; short eager/graph, ragged, and sparse c8→c1 cases pass; c4
and true physical-c8 eager/graph p512/d128 contribute **124,400/124,400** exact
all-layer comparisons; and the 18-prompt category plus heldout suite contributes
another **54,000/54,000** across three deterministic repeats. The complete
direct packet retains **188,080** hidden comparisons with zero token,
Conv/GDN/live-KV, or hidden mismatch and manifests zero complete-c1 replay,
row-local model loop, or subgraph invocation. Clean detached `d0195221` then
proves one physical c8 graph replay with **748 packed-native / 0 row-local / 0
copy dispatches** and all full-attention, selected-MoE, exact `6+2` Q6 LM-head,
sampler, and metadata checks passing. The same-session p512/d128 packet measures
c1/c2/c4/native-c8/chunked-c8/serial-c4 at
**50.277/72.104/102.597/127.902/102.606/50.206 aggregate tok/s**. Native c8 is
**2.544x c1**, **1.247x c4+c4 (+24.65%)**, and **2.548x** the serial-c4 rate;
per-request c8 is **15.988 tok/s**, and its higher TTFT/ITL/memory costs remain
explicit. This retains the direct gfx1151 model-step packet without an
architecture-specific kernel/runtime edit. The unchanged Phase-D loop then
admits B while A decodes, runs two live rows in one masked physical-c4 bucket
(`1100`), disconnects/reclaims B and D while A/C remain independent-c1
state/KV exact, streams row-owned chunks, completes real OpenAI SSE with usage
and `[DONE]`, exposes request/KV ownership metrics, records zero serial/resident
fallback, and drains scheduler/session/KV ownership to zero. This closes E1 at
correctness-only `continuous_eq_ok`; it adds no server throughput, TTFT, ITL,
or concurrent-kernel claim. Evidence:
`benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json`,
`benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json`,
and `benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-live-loop-closure.json`.

E2. Native widths:

- [x] Support physical buckets c1/c2/c4/c8 with active masks.
- [x] Run one true native c8 model step; c4+c4 may remain an explicit fallback
      but cannot qualify as c8.
- [x] Pass c8 steady, ragged, sparse, cancellation, and 512/128 equality.
- [x] Validate non-edge survivors through c8→c1 retirement without compaction.
- [x] Validate optional compaction separately with state/KV hashes at every move.

Clean `4089de11` eager and this E2 graph-control probe use one physical eight-row
model step rather than c4+c4. The graph probe captures and replays one c8 bucket
twice with exact tokens, Conv/GDN/live-KV, and **960/960** all-layer hidden
comparisons versus eight independent c1 references. Its manifest reports
`physical_rows=8`, all eight lanes active, zero complete-c1 session/layer replay,
zero host model-row loops/subgraph invocations, and zero steady metadata copies.
This satisfies only the true-native-step item. The subsequent eager masked-
bucket probe keeps one physical c8 workspace across active masks
`11111111 → 10110111 → 10100101 → 00100100 → 00000100`, covering non-edge
c8→c6→c4→c2→c1 retirement without compaction. Tokens, every survivor's
all-layer hidden output, and every physical session's Conv/GDN/live-KV state—including
all retired lanes—remain exact across **1,160/1,160** comparisons. Inactive
lanes carry `-1` positions, zero live counts, and no state import/scatter. The
mask-aware graph-control helpers also keep inactive block rows, token feedback,
recording, and cursors inert. Five mask-specific physical-c8 graphs replay the
same c8→c1 sequence with **1,160/1,160** exact hidden comparisons; the cumulative
all-active c8 graph regression remains **960/960** exact. Ragged p16/23 graph
replay is **1,600/1,600** exact. A capacity-eight `protect_ttft` live loop cancels
non-edge lane 3 after one native c8 step, then runs two masked physical-c8 steps;
cancelled/survivor tokens and every Conv/GDN/live-KV hash match production-route
c1 controls, with zero fallback and final ownership zero. The standard p512/d128
eager and graph paths each pass **41,280/41,280** hidden comparisons plus exact
tokens/state/KV; graph captures once and replays 128 times with zero steady
copies. This closes the physical-bucket, no-compaction retirement, and complete
E2 equality-suite items. Clean `52b0db25` then retains one physical-c8 graph at
**246.872 aggregate tok/s** (**30.859 per request**), **2.888x c1** and
**1.349x c4+c4 (+34.89%)**. All measured trajectories repeat and match across
native/chunked controls. The marker-sliced replay is **748 packed-native / 0
row-local / 0 copies** and passes full-attention, selected-MoE, exact `6+2` Q6
LM-head, sampler, and metadata census checks. E3 separately validates optional
compaction and arbitrary-C; no server-performance or gfx1151 claim is inferred
from the E2 packet alone. Evidence:
`benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-correctness.json`
and `benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json`.

E3. Arbitrary request counts:

- [x] Lower arbitrary live C into one or more declared physical buckets.
- [x] Prove tail buckets and masked lanes cannot mutate inactive state or KV.
- [x] Record logical C, physical bucket width, number of groups, and active mask
      in every artifact.
- [x] Establish policy for C>8 from measurement: wider native buckets, multiple
      groups, or both. Never report multiple groups as one native width.

Both gfx11 E3 packets lower logical C13 as physical c8 plus sparse physical c8
(`11111111 + 11111000`), never native c13. gfx1100 short and p512/d128
eager/graph paths pass **135,200/135,200** all-layer comparisons overall;
gfx1151's clean transfer passes **134,160/134,160** across short graph plus
p512/d128 eager/graph. Tokens, Conv/GDN, and live BF16 KV are exact on both.
Middle-hole cancellation at slots 2/10 produces `11011111 + 11011000`; tail
admission restores C13 without fallback. On each backend, explicit compaction
performs nine real moves, preserves every survivor hash, allocation, block id,
and device pointer, closes both sparse graphs with **2/2** invalidations, and
admits newcomers at slots 11/12. The real server packets retain logical
c1/c8/c9/c13/serial-c13 at **25.583/136.122/88.592/111.380/31.708 aggregate
tok/s** on gfx1100 and, after gfx1151 F2 retention,
**43.033/86.942/77.302/73.235/43.066** on gfx1151. Grouped C13 is
**4.354x/1.702x** logical-c1 and **3.513x/1.701x** serial. Both
C9 drops versus C8 establish the retained policy: multiple declared groups
above eight, no wider native bucket yet. Compaction remains explicit and
carries no automatic-policy/performance claim. Evidence:
`benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json`,
`benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json`,
`benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json`,
and
`benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json`.

Exit: GGUF Q4_K_M/BF16 has exact native c1/c2/c4/c8 model steps, live admission,
honest arbitrary-C lowering, and retained real OpenAI burst/live-admission
scaling on both gfx1100 and gfx1151.

### Phase F — c1-preserving production GGUF concurrency

**Objective:** one OpenAI server configuration keeps the fastest exact c1 route
at occupancy one, expands into exact native groups under load, and maximizes
goodput within explicit TTFT/ITL SLOs without benchmark gaming.

F0. Current-tree freshness gate:

- [x] Re-run the narrow gfx1151 direct, lifecycle, real SSE, and profiler packet
      on current `main` after the shared PARO owner/API changes.
- [x] Record the realized occupancy-one physical route; masked c8 is a control,
      not the desired production c1 policy.
- [x] Preserve the completed F1 evidence if a focused repair is sufficient; do
      not rerun unrelated expensive gates automatically.

Clean current-main `ef46ee8c` passes the focused F0 packet after repairing the
physical-c2 gfx1151 attention reduction. Direct p512/d128
c1/c2/c4/c8/chunked-c8/serial-c4 is
**50.291/72.262/102.663/128.075/102.724/50.235 aggregate tok/s** with every
cross-route trajectory exact and at most **0.039%** rate stdev/median. Current
c8 remains one physical group, **2.547x c1** and **+24.68%** over c4+c4; the
cached route census is **748 packed-native / 0 row-local / 0 copies**. Real SSE
logical c1/c8/c9/c13/serial-c13 is
**15.798/86.358/57.691/73.065/43.116 aggregate tok/s**, all **189/189** rows are
exact, and live c8→c13 emits **1,664/1,664** exact IDs at **71.675 tok/s** before
ownership drains. The occupancy-one server route is explicitly
`physical_widths=[8]`, mask `10000000`, and **127 masked-c8 decode steps / zero
native-c1 steps**. F0 therefore refreshes the retained baseline and closes
freshness only; it does not satisfy F2. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-f0-current-main-recertification.json`.

F1. Retention packet:

- [x] Run c1/c2/c4/c8 prompt-512/decode-128 on gfx1100 and gfx1151.
- [x] Include same-protocol c1, serial bridge, exact hybrid, and native rows.
- [x] Run burst/live-admission traces with per-request latency percentiles.
- [x] Run the complete prompt-category suite plus heldouts.
- [x] Update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, compact artifacts,
      and `WORKLOG.md` for every retained promotion.

Both gfx11 F1 slices are retained: existing direct c1/c2/c4/c8 native/control
rows and each 18-prompt category+heldout anchor join clean real SSE burst
packets. Each packet preserves **189/189** exact resident prompt IDs, output
IDs, usage, finish metadata, and scheduler timestamps. Maximum static variance
is **1.299%** on gfx1100 and current F0 **0.485%** on gfx1151. Controlled live
runs observe c8 before tail admission, reach C13, emit **1,664/1,664** exact IDs
at **107.284/71.675 aggregate tok/s**, and drain request/session ownership to
zero.
Server timing is complete SSE cycle wall and remains separate from direct
graph-step timing.

F2. Occupancy-adaptive execution:

- [x] Use the exact c1 graph/GEMV route for one live request in the same owner
      that can later execute c2/c4/c8.
- [x] Change physical groups without changing logical request IDs, re-prefilling
      survivors, reallocating stable state/KV, or weakening cancellation/reclaim.
- [x] Choose sparse masking, declared partitions, or compaction from measured
      cost and SLO policy; never label multiple groups as one wider native group.
- [x] Keep occupancy-one backend transition performance within **5%** of the
      same-process direct c1 control, with HTTP/SSE overhead reported separately.
- [x] Retain current c8/C13 exactness and aggregate scaling while improving the
      low-occupancy route.

Clean `190f208c` closes F2. Stable scheduler slots, request IDs, state, and KV
stay fixed while dense ephemeral execution rows select c1/c2/c4/c8. The retained
p512/d128 SSE packet is **43.033/86.942/77.302/73.235/43.066 aggregate tok/s**
at c1/c8/c9/c13/serial-c13 with all **189/189** rows exact and at most **0.524%**
rate variance. Occupancy-one transitions reach **50.610 vs 52.926 direct tok/s
(95.625%)**; complete SSE c1 improves **2.724x** over F0's masked-c8 control.
C9 improves **33.99%** via c8+c1, while c8/C13 are non-regressive. Clean live
C2→C8 and C4→C8 traces preserve **256/256** IDs each, and live C13 preserves
**1,664/1,664** at **71.891 tok/s**; every run drains ownership. A clean owner
c1 gate crosses the graph break-even and proves capture/replay/reclaim. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-f2-occupancy-adaptive-serving.json`.

F3. Profile-directed tuning order:

1. remove host synchronization and metadata transfers;
2. amortize launch/graph dispatch cost;
3. improve small-C projection and LM-head geometry;
4. improve c-aware Conv/GDN and full attention;
5. improve MoE grouping/routing/selected-expert utilization;
6. overlap pool/admission work only after exact ownership is stable.

- [x] Attribute launch count and GPU time by family for c1/c2/c4/c8.
- [x] Keep exact non-regressive wins and promote them to package defaults.
- [ ] Record rejected and neutral probes in `WORKLOG.md`, not this roadmap.
- [ ] Remove obsolete experiment flags through `docs/REFACTOR.md` after defaults
      settle.

Clean `3a0fe188` closes the width-family census. Physical c2/c4/c8 record
**747/747/748 packed-native dispatches**, zero row-local or steady-copy work,
and zero unclassified kernels. Dense projections plus selected-MoE/combine
account for **64.51%/68.24%/68.78%** of diagnostic GPU time, so F3 proceeds to
those kernel families rather than more host-movement work. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-f3-width-family-census.json`.

Clean `e99b228b` retains the first profile-directed kernel win. Packed gfx1151
AR now uses a one-token-per-row indexed GDN sibling with the arbitrary-segment
kernel preserved as fallback and gfx1100 left unchanged. Canonical p512/d128
direct c2/c4/c8 improves **+8.71%/+5.25%/+4.04%** to
**78.552/108.050/133.251 aggregate tok/s**; c1 is structurally unchanged at
**50.335**. All measured trajectories repeat exactly, and the required c8 trace
keeps **748 packed-native / 0 row-local / 0-copy** dispatches while diagnostic
Conv/GDN time falls **8.230 -> 4.038 ms (-50.94%)**. Existing F2 server-wall
rows remain retained but were not remeasured. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-f3-singleton-gdn-retained.json`.

F4. Production workload and SLO gate:

- [x] Add static, ragged, burst, and controlled continuous-arrival workloads
      with mixed prompt/output lengths and generated-token correctness.
- [x] Report goodput, queue delay, TTFT, ITL, and end-to-end p50/p95/p99 together
      with occupancy, KV/workspace memory, rejects, and finish reasons.
- [x] Exercise cancellation, disconnect, overload, bounded backpressure, idle
      recovery, and a sustained soak without leaking request/session ownership.
- [x] Tune prefill/decode token budgets and fairness from measured SLO curves,
      not one fixed prompt or one headline aggregate rate.

Clean `ecaf14d5` closes F4 on Radeon 8060S/gfx1151, Qwen3.6-35B-A3B
Q4_K_M/BF16-KV, and real localhost Uvicorn SSE. The four-candidate frozen sweep
selects scoped package-default `fair:256` at **46.527 exact SLO-goodput tok/s**,
**+6.73%** over passing `fair:128`. All nine workloads pass the declared
queue-p99/TTFT-p95/ITL-p99/end-to-end-p95 SLOs: cancellation preserves **6/6**
normal neighbors plus one post-token disconnect and one distinct 408 timeout;
overload returns **16 exact completions + 16 exact 429 rejects**; and the
60-second soak completes **120/120 at 43.314 goodput tok/s**. Memory and all
scheduler/runner/KV ownership drain. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-f4-production-load-slo.json`.

F5. Serving economics and comparison closure:

- [x] Close deterministic seeded per-row sampling and disclose exact serial/API
      fallbacks for logprobs, stop strings, tools, structured output, and n>1.
- [x] Retain the first active-current greedy p256+s1 reuse slice on gfx1151 with
      real device-KV refcounts/COW/reclaim, byte-exact state/KV, and measured
      TTFT/live-page benefit.
- [x] Retain completed-source p256+s1 correctness/economics on gfx1151 with
      cache-owned device snapshots, source reset before admission, explicit
      eviction, exact **1->1->2->1->0** ownership, and disclosed residency cost.
- [ ] Generalize prefix/continuation reuse to broader historical boundaries,
      sampled requests, LRU pressure, and graph-safe lifecycle; retain it only
      with measured benefit and no survivor regression.
- [ ] Validate mixed 1K/4K/32K and feasible longer contexts, grow/shrink, capacity
      rejection, and supported non-BF16 KV policies.
- [ ] Run same-GGUF llama.cpp c1/c2/c4/c8/C13 with matched generated-token and
      server timing boundaries; qualify different-quant vLLM/SGLang rows.

Clean `7871c088` closes the gfx1151 sampled/API slice through one real localhost
Uvicorn owner using package-default scheduling. Two repeated four-row blocking
waves and two repeated four-row SSE waves are deterministic and reconstruct
exact text plus selected/top logprobs; repeated `n=3` preserves exact choice
indexes and usage. Request EOS, stop, strict forced-tool parsing, and strict
root-object schema handling pass, with malformed structured output explicitly
rejected as `schema_violation`. The sampled waves record physical c2/c4 work,
**26 total host-sampler requests**, **88 packed model steps**, and **zero
serial-decode or resident-request fallback** across the full packet. Memory and
all scheduler/model-runner/KV ownership drain. `host_sampling_required` is an
honest sampler-placement disclosure, not a serial model fallback. This is a
correctness/serving-path result, not a throughput claim; gfx1100 transfer,
general prefix reuse, long-context/non-BF16 pressure, and external comparisons
remain open. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-f5-sampled-openai-api.json`.

The first prefix/KV-reuse slice is retained only as explicit
`HIPENGINE_PREFIX_CACHE=radix` on gfx1151. A live greedy 256-token source shares
its exact-current page with a one-token-suffix continuation; the clone boundary,
output, all 60 Conv/GDN and 20 live K/V components, source-first reclaim, final
survivor state, and four teacher-forced transitions are byte-exact (`KL=0.0`,
top-1 `100%`). A clean one-warmup/three-pair benchmark charges synchronized
continuation admission through first token and moves median TTFT **249.269 ->
21.188 ms (11.765x, -91.50%)**, with 3/3 exact 256-token hits, no fallback, and
live pages **4 -> 3** (**5,242,880 bytes**). Paired HIP-current deltas are
`0/203,423,744/0` bytes, median zero, so no process/GPU-current reduction is
claimed. The fixed-capacity backing is preallocated in both modes. Default
remains `off`: broader boundary/LRU/graph pressure is unmeasured, sampled reuse
is unsupported, packed shared-prefix suffix arithmetic is not byte-exact to c1,
the completed cache carries explicit residency cost, and the separately measured
gfx1100 agentic economics reject promotion. The completed-source gate
resets/unbinds the source before admission, restores a 66,846,720-byte
cache-owned hybrid snapshot, and reproduces boundary/output/all Conv/GDN/live-KV
and four teacher-forced steps byte-exactly (`KL=0`, top-1
`100%`), and drains refs through **1->1->2->1->0**. Its clean one-warmup/three-pair
packet moves continuation TTFT **249.446 -> 22.013 ms (11.332x, -91.18%)** with
3/3 exact snapshot hits. Unique continuation pages stay **2 -> 2**; retained
snapshot + page residency is exactly **72,089,600 bytes**, tracked allocation is
+66,846,720 bytes, and paired HIP current is **+62,914,560 bytes**. Evidence:
`benchmarks/results/2026-07-19-gfx1151-gguf-active-prefix-reuse-correctness.json`,
`benchmarks/results/2026-07-19-gfx1151-gguf-active-prefix-reuse-economics.json`,
`benchmarks/results/2026-07-19-gfx1151-gguf-completed-prefix-reuse-correctness.json`,
and
`benchmarks/results/2026-07-19-gfx1151-gguf-completed-prefix-reuse-economics.json`.
The matching gfx1100/W7900 correctness transfer now passes both active-current
and completed-source p256+s1: output, all Conv/GDN/live-KV bytes, four
teacher-forced steps (`KL=0`, top-1 100%), refcount/COW, cache eviction, and
final drain are exact. This p256 transfer is correctness-only; the separate
agentic p2048/p8192 economics and pressure packet is closed and rejects
promotion. Evidence:
`benchmarks/results/2026-07-21-gfx1100-gguf-active-prefix-reuse-correctness.json`
and
`benchmarks/results/2026-07-21-gfx1100-gguf-completed-prefix-reuse-correctness.json`.

Exit: one package-default GGUF OpenAI owner preserves near-direct c1 performance
at low occupancy, retains exact native scaling under load, and passes the
production workload/SLO, cache/KV, sampling, long-context, and matched external
comparison gates on gfx1100 and gfx1151.

### Phase G — PARO native concurrency

**Objective:** implement the same first-class pipeline for PARO after GGUF has
proven the runner, scheduler, KV, lifecycle, and evidence contracts.

PARO work may begin earlier when it does not destabilize the active GGUF closure
set, but it must reuse rather than fork the production loop.

G1. Re-establish c2 controls:

- [x] Run current-HEAD gfx1100 PARO c1 and c2 exact/serial controls. Clean
      p512/d128 at `ff4e21d2`: serial c2 matches 274/274 recorded IDs; direct
      native c2 first diverges at generated index 2 and remains rejected.
- [x] Re-run the gfx1151 c2 route through the full retained packet. Clean
      `778c7a70` selected-batch passes direct p512/d128 repetition, all-layer
      hidden/Conv/GDN/KV/NumPy-context, uniform/ragged lifecycle, ten prompts,
      primitive, auto-default, profiler, and scaling without a target-specific
      code change.
- [ ] Separate graph/eager policy per backend using registered capabilities.
- [ ] Preserve true width-1 fail-closed behavior for unsupported groups.

G2. Fully native c2:

- [x] Bisect the first hidden divergence with a reusable layer/stage comparator.
      On gfx1100, L1/L2 were green and full-attention layer 3 first failed at
      native batch context because append-relative block rows were reused for
      absolute decode addressing. Separate cached append/decode tables close
      that physical-row alias.
- [x] Close the next short-context arithmetic boundary with one physical-c2
      batch-grid kernel that follows dense c1 reduction order. Clean `32de8d08`
      p512/w8/d128 matches 274/274 IDs; the full L40/d3 hidden/Conv/GDN/KV and
      NumPy-context gate is exact; cached rocprof records the expected c2 kernel.
      This left selected-c1 MoE and lifecycle open.
- [x] Close grouped-compact selected-MoE arithmetic at c2. The old FP16 compact
      GEMV path inherited 128-thread wrapper defaults while selected-c1 uses 64;
      matching that reduction geometry makes every routed stage bit-exact. Clean
      grouped p512/d128 repeats, full L40/d3, uniform/ragged EOS+cancel c2→c1,
      and inactive state/KV immutability now pass with 40 grouped layers, zero
      fallback layers, and no hidden/state/KV bit drift.
- [x] Replace row-local full-attention and selected-c1 hybrid boundaries with
      exact c-aware routes on gfx1100. Full attention is native, and canonical
      `selected_batch` is one rows=2 selected-kernel transition with zero
      row/layer replay, unlike the separately named per-row fallback.
- [x] Close gfx1100 batch-GEMV QKV/Z/O/FFN output projections at c2; the clean
      all-layer gate and retained validator name every admitted batch path and
      report no row chunks or fallback layers.
- [x] Close Conv/GDN segmented state mutation and selected-expert MoE at c2.
- [x] Pass 512/128 direct and shrinking-lifecycle equality on both backends.
- [x] Trace true c2 selected-batch steps with no rowchunk/serial model fallback
      on both gfx11 backends. gfx1100 records 1,306 dispatches; gfx1151 records
      1,598. Both are `eq_ok` and contain the exact c2 context plus selected
      projection families.

Clean detached `778c7a70` transfers the retained algorithm unchanged to the
Radeon 8060S. Three direct p512/d128 runs are
**79.163/79.228/79.218 aggregate tok/s** (median **79.218**, **+11.87%** over
c1 graph and **+20.81%** over serial c2), all **274/274** recorded IDs per run
are exact, and auto resolves all 40 layers to selected-batch with zero fallback.
The all-layer, three lifecycle, primitive, ten-prompt/**330/330 ID**, and cached
profiler gates pass. The trace contains one exact c2 context dispatch and ten
selected projection dispatches. This G2 snapshot retained only the direct c2
model step; G3 below supersedes its c4/c8 status, while public/OpenAI ownership
still remains width-1. Evidence:
`benchmarks/results/2026-07-18-gfx1151-paro-g2-selected-batch-c2-retained.json`.

G3. Native c4/c8:

- [x] Generalize the c2 algorithms rather than stacking c2 groups and calling
      them c4/c8 on gfx1151.
- [ ] Pass the complete c1-c8 exact matrix from `docs/BENCHMARK.md`; retained
      dispatch widths are intentionally c2/c4/c8, while c3/c5/c6/c7 fail closed
      through exact partitioning.
- [x] Pass sparse c8→c1 lifecycle and cancellation without compaction; every
      retired row is immutable and all surviving token/state/KV hashes match c1.
- [ ] Capture/replay validated width/context buckets where replay wins.
- [ ] Retain backend-specific c1/c2/c4/c8 rows on both targets; gfx1151 is now
      retained, while gfx1100 remains at c2.

Clean equivalent-tree commits `e175e28f` (measured) / `8c8cc15e` (pushed)
retain true physical gfx1151 c4/c8. Three p512/d128 processes measure c2/c4/c8
at **79.237/100.209/99.943 aggregate tok/s** with <=0.054% stdev/median;
c4/c8 are **+41.52%/+41.14%** over c1. c8 aggregate throughput is
**0.265% below c4**, showing a real c4 bandwidth plateau, but its median model
step remains **0.183% faster than two sequential c4 steps**. All **5,754/5,754**
canonical IDs, all ten category/heldout prompts at both c4 and c8, the L40/d3
state/KV/NumPy oracle, three sparse c8→c1 lifecycle variants, primitives, and a
4,644-dispatch c8 kernel trace pass. Evidence:
`benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json`.

G4. Attach PARO to the production loop:

- [x] Implement the shared runner contract without a PARO-only scheduler.
- [ ] Exercise live admission, chunked prefill, decode, cancellation, reclaim,
      streaming, and device KV ownership on both backends. gfx1151 is retained;
      gfx1100 remains open.
- [x] Route `LLM.generate()` and server requests through the same model-owning
      loop.
- [x] Promote gfx1151 only after the same Gate 1–5 packet used by GGUF passes;
      retain the identical requirement for gfx1100.

The gfx1151 correctness packet uses one fixed-capacity session with stable model
slots independent of scheduler compaction. A 512-token live-admission sequence
keeps slot 0 exact for 16 tokens while slot 1 is cancelled after one visible
token and immediately reused for an exact 8-token result; seven native c2 calls
execute with no fallback reason and all five admissions/reclaims drain. Real
OpenAI blocking c2 reports exact 1024+16 accounting, while two concurrent
512/8 SSE requests each match the independent eight-ID baseline, emit eight
deltas plus done/usage/`[DONE]`, count cumulative streamed tokens 1..8, and each
executes five native c2 steps. Fair admission intentionally also records two
serial steps per stream; that transition is truthful rather than a fallback
concealment. The shared host/server gate passes **1,018 tests**. G4 correctness
evidence is
`benchmarks/results/2026-07-18-gfx1151-paro-g4-resident-openai-correctness.json`.
G5 subsequently retains gfx1151 blocking c1/c2/c4/c8 at
**47.124/51.962/60.323/61.253 aggregate tok/s** and real SSE
c1/c2/c4/c8/serial-c8 at **36.327/38.666/42.471/41.487/35.633**, with every
measured row exact and package-default c2/c4/c8 routing. Evidence:
`benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json` and
`benchmarks/results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json`.
gfx1100 owner symmetry remains open.

Exit: PARO W4/BF16 has retained native continuous batching on gfx1100 and
gfx1151.

### Phase H — coverage and advanced composition

These items follow the Q4_K_M/PARO-W4 BF16 core. They must not broaden the active
closure set prematurely.

- [ ] GGUF Q5_K, Q6_K, and Q8_0 c=N route/equality coverage.
- [ ] PARO INT8 KV c=N coverage.
- [ ] Seeded temperature/top-k/top-p/repetition-penalty sampling.
- [x] Active-current greedy p256+s1 prefix reuse through opt-in `RadixCache`
      with real device-pool refcounts/COW and exact survivor lifecycle on gfx1151.
- [x] Completed-source greedy p256+s1 cache-owned snapshot restore/economics on
      gfx1151, with 11.332x TTFT, exact eviction/final drain, and 72,089,600-byte
      cache residency explicitly disclosed.
- [x] Active-current and completed-source greedy p256+s1 correctness transfer to
      gfx1100, including exact output/state/KV, COW/refcounts, eviction, and drain.
- [x] gfx1100 deterministic agentic p2048/p8192 active/completed prefix
      correctness, C1 economics, LRU/pressure lifecycle, eviction, and final
      drain; the performance decision rejects promotion and keeps cache off.
- [ ] Sampled prefix reuse and broader graph-safe historical boundaries.
- [ ] Long-context 4K/32K/128K admission and memory-pressure policies.
- [ ] Graph-pool invalidation under real grow/shrink events.
- [ ] DMS/KVTC tier movement under the stable-id/rebind contract.
- [ ] MTP/DFlash verify/commit/scatter as a new scheduler work class.
- [ ] Tensor-parallel compatibility under `docs/TENSOR_PARALLEL.md`.

MTP/DFlash throughput must use a true no-MTP AR baseline and the complete
multi-prompt acceptance suite. Speculative work never weakens the AR c=N gates.

### Future campaigns — route-cap follow-up and true continuous batching

These campaigns follow Phase H. They target the boundary between “c=N as a
fixed resident shape” and “concurrency as a function of live KV capacity”.
Until they close, the plain-AR route keeps its registry-owned fixed caps.

**Route-cap follow-up (near-term).** The server's plain-AR concurrency is
clamped by the `*_SERVER_PLAIN_AR_MAX_ACTIVE_REQUESTS` package capabilities
(currently C4 on gfx1100 against the measured 4K/C8 resident-session OOM, C8
on gfx1151, and 13 logical rows only inside the 768-position envelope), and
`--max-active-requests` is applied as `min(flag, route cap)`. This clamp is
why a c8 burst runs as a width-4 resident bucket. The campaign:

- [ ] Re-qualify the gfx1100 C4 cap at 16K/256K contexts under real KV pool
      pressure (measured page high-water, graph pins, and the OOM boundary at
      C4/C6/C8); record the context-qualified mapping the same way the
      768→13 envelope was recorded.
- [ ] Decide whether `--max-active-requests` should widen the route cap,
      narrow it only, or be rejected when it exceeds the registry value; keep
      the registry value authoritative for memory.
- [ ] Keep the 13/768-position envelope and any new context entry as
      separate, independently gated capabilities.

**True continuous batching (the destination).** The fixed c=N row caps are a
memory-safety approximation. The target is admission governed by live
device-KV pool state — free pages, graph-pinned pages, and the configured
whole-device reserve — so the engine stops caring about `c=N` limits and
shares/evicts KV intelligently instead:

- [ ] Admission by KV budget: admit a row when its projected page need
      (prompt + max_tokens + reserve) fits the live pool, replacing the fixed
      active-row cap as the primary gate; keep the row cap only as a
      degenerate backstop for unprojectable requests.
- [ ] KV sharing: extend refcounted page sharing from the opt-in p256+s1
      prefix slice to general shared prefixes and system-prompt reuse, with
      COW/refcount lifecycle and exact survivor-state gates.
- [ ] KV eviction: add bounded eviction of completed/low-priority resident KV
      (page-level, never graph-pinned or live prefix-shared pages) so a burst
      can reclaim capacity from finished neighbors without dropping live
      rows; eviction must invalidate or rebind affected graph buckets per the
      KV contract.
- [ ] Prefill co-admission: let one tick admit several bounded prefill chunks
      (the scheduler currently emits one row per `next_prefill_work` tick, so
      `protect_decode` can serialize an entire group); keep `fair`
      interleaving as the default and re-derive the tick bounds.
- [ ] Re-certify Gates 1–5 (lifecycle, cancellation, sparse retirement,
      ownership drain) and the production load/SLO workloads under
      capacity-adaptive admission before removing any fixed cap.

No item may be claimed as “continuous batching” until a mixed-arrival
overload trace shows live admission + decode + eviction + reclaim in one
loop with exact survivor state/KV and bounded memory; the current C4/C8
caps remain the production boundary until then.

## Active execution queue

Work this list in order unless a measured blocker is recorded in `WORKLOG.md`.
The active lane is deliberately narrow even though the production closure has
multiple later gates.

1. **Completed — C4:** prove one fully native replayable c4 model step.
2. **Completed — C4:** publish the direct c1/c2/c4 and chunked-c8 performance packet.
3. **Completed — D1:** attach that c4 step to one long-lived gfx1100 model runner.
4. **Completed — D2:** close live admission, retirement, and cancellation on W7900.
5. **Completed — E1:** retain the same model-step and live-loop gates on gfx1151.
6. **Completed — E2:** retain one true physical-c8 gfx1100 model step.
7. **Completed on both gfx11 targets — E3/F1:** arbitrary-C lowering, explicit
   optional-compaction correctness, repeated real SSE burst scaling, and
   live-admission latency are retained on both targets without weakening the
   direct gate.
8. **Completed on gfx1151 — G3/G5:** retain PARO physical c2/c4/c8 and the
   package-default resident blocking/SSE owner; gfx1100 symmetry is separate.
9. **Completed — F0:** current-main gfx1151 GGUF direct/server/profiler is
   recertified; occupancy one is confirmed as masked physical c8.
10. **Completed — F2:** retain c1-preserving occupancy-adaptive c1/c2/c4/c8
    execution in one long-lived gfx1151 GGUF owner.
11. **Completed — F3 plus profile requalification:** singleton-indexed
    packed-AR GDN remains retained. The former blanket Q8T16 rejection conflated
    deterministic free-running ID divergence with control correctness: a fresh
    ZBook-local 1,050-row strict-teacher gate is bit-identical, so current A/B evidence now
    retains all-projection rowtiling only at physical c4/c8; c2 remains direct
    after a measured `1.795%` regression.
12. **Completed — F4:** clean real-Uvicorn static/ragged/fixed/Poisson,
    scheduler-owned timeout/disconnect isolation, exact overload, recovery, and
    120-request/60-second soak all pass under scoped package-default `fair:256`.
13. **Completed on gfx1151 — F5 sampled/API:** repeated seeded blocking/SSE and
    `n=3` use packed c2/c4 model ticks with exact API metadata, explicit host-
    sampler disclosure, and zero serial/resident fallback.
14. **Completed on gfx1151 — F5 active/completed prefix economics:** explicit
    active-current p256+s1 reuse is byte-exact, 11.765x faster to first token,
    and saves one 5 MiB live page; completed-source reuse is byte-exact through
    **1->1->2->1->0** and 11.332x faster at 72,089,600-byte cache residency.
15. **Completed on gfx1100 — coding-agent A1–A6:** deterministic concurrency,
    prefix economics/lifecycle, native-sampler correctness/tool preflight,
    routing/SLO, bounded pressure/soak, and broad automatic-tool quality are
    published. Prefix, native-sampler, and A4 routing candidates remain rejected.
16. **Completed — constrained tools:** tokenizer-aware host candidate masking
    now enforces canonical tool envelopes, declared names, root-object JSON,
    exact forced/dynamic close queues, and fail-closed fallback.
17. **Active — A4 exactness:** repair the late p512/d48 physical-width transition
    divergence against independent c1 state/KV trajectories.
18. **Then — A3/A4 rerun:** rerun the frozen W7900 gates and promote only a
    candidate that passes every correctness, ownership, C1/medium-C4, and SLO
    requirement.
19. **Completed — IKV-C0:** integrated the divergent Qwen3.8 branches and
    locked INT8 admission to full-file artifact/backend/target/quant/layout/scale
    evidence with BF16 fallback and explicit provenance.
20. **Next — IKV-C1:** qualify compact no-mirror c2/c4 through an honest serial
    c1-per-row route with shifted block-table-aware prefill; make no throughput
    claim.
21. **Then — IKV-C2:** add and trace the row-batched 24Q/4KV/D256 INT8 split-K
    producer/reducer, preserving c1 and unfused fallbacks.
22. **Then — IKV-C3/C4:** prevent N-copy prefill scratch from erasing savings
    and reject the complete resource budget before HIP OOM.
23. **Then — IKV-C5/C6:** close cancellation, grow/shrink, overload/recovery,
    direct-vs-mirror telemetry, and artifact-scoped quality/capacity.
24. **Then — IKV-C7:** run matched BF16/serial/direct economics, promote or
    reject, and remove the mirrored seam when safe.
25. **After IKV — F5/H:** broaden sampled prefix reuse and graph-safe historical
    boundaries, close other long-context/memory pressure, and run matched
    external serving comparisons.
26. **Future — route cap / true continuous batching:** qualify the
    context-qualified route-cap mapping under KV-pool pressure and build
    KV-budget admission plus shared/evictable resident KV so concurrency is
    governed by live KV capacity instead of fixed c=N row caps. See “Future
    campaigns — route-cap follow-up and true continuous batching”.

Do not label C>8 grouping, general prefix caching, DMS, speculative
integration, or external-engine parity from the retained gfx11 GGUF results;
each keeps its own gate and artifact.

## Coverage ledger

Use only these status values:

- `not_started` — no current-HEAD evidence on that backend;
- `primitive_ok` — kernel primitives pass, no model equality;
- `token_diag` — generated-token equality only; no hidden/state/KV lifecycle proof;
- `direct_eq_ok` — direct steady/sparse/ragged token/hidden/state/KV equality passes,
  but the complete standard lifecycle/prompt gate is still open;
- `exact_hybrid` — the complete direct lifecycle gate passes with declared row-local work;
- `native_eq_ok` — native direct/lifecycle equality passes;
- `continuous_eq_ok` — live admission/reclaim equality passes;
- `retained` — Gate 1–5 artifact is promoted.

| Path | gfx1100 | gfx1151 | Target |
| --- | --- | --- | --- |
| GGUF Q4_K_M / BF16, c2 | `direct_eq_ok` | `retained` | `retained` |
| GGUF Q4_K_M / BF16, c4 | `retained` | `retained` | `retained` |
| GGUF Q4_K_M / BF16, c8 native group | `retained` | `retained` | `retained` |
| GGUF Q4_K_M / BF16, live admission | `retained` | `retained` | `retained` |
| GGUF Q4_K_M / BF16, arbitrary-C lowering | `retained` | `retained` | `retained` |
| GGUF Q4_K_M / BF16, sampled OpenAI API | `direct_eq_ok` | `retained` | `retained` |
| Qwen3.8 `7b2aec...` / no-mirror INT8 c1 (explicit artifact) | `retained` | `not_started` | `retained` |
| Qwen3.8 `7b2aec...` / short mirrored INT8 live admission (diagnostic) | `continuous_eq_ok` | `not_started` | `retained` |
| Qwen3.8 `7b2aec...` / direct no-mirror INT8 c>N (IKV-C0-C7) | `not_started` | `not_started` | `retained` |
| PARO W4 / BF16, c2 | `retained` | `retained` | `retained` |
| PARO W4 / BF16, c4 | `not_started` | `retained` | `retained` |
| PARO W4 / BF16, c8 | `not_started` | `retained` | `retained` |
| PARO W4 / BF16, live admission | `not_started` | `retained` | `retained` |

Update this table only when the named gate changes. A new timing alone does not
advance status.

## Implementation guardrails

### Registry boundaries

- No `if backend == ...` or `if quant == ...` in engine, scheduler, or model
  dispatch code.
- Backend/quant capability selection belongs in the four-axis plugin registry or
  package metadata.
- gfx1100 and gfx1151 may register different launch variants behind the same
  model-step contract.
- Every fused route retains a registered strict unfused chain. Production
  composites may reassociate only under a certified profile manifest.

### Correctness before tuning

- Add a RED fixture before changing model math whenever practical.
- Preserve independent strict c1 sessions as the arithmetic/control oracle until
  c=N becomes canonical and a frozen fixture exists.
- Never keep a strict win that regresses required parity or a production win
  that fails any control, tail-KL, category, determinism, isolation, or task
  gate.
- Fail closed to a certified strict route for unsupported/uncertified shapes;
  never silently enter an unvalidated native or production route.

### Honest execution labels

Every route emits:

```text
execution_profile
execution_profile_schema
variant_manifest_sha256
strict_manifest_sha256
generated_id_equality_binding
logical_concurrency
physical_bucket_width
native_group_width
native_group_count
active_mask
prefill_route
decode_route
state_route
attention_route
moe_route
sampler_route
serial_or_row_fallbacks
graph_bucket_key
graph_replay_count
```

`native_caware_decode=true` is an execution label, not a correctness or
throughput claim. The coverage status and artifact gates determine the claim.

### Memory accounting

Report both:

1. tracked allocator/pool bytes; and
2. process/GPU-visible current and peak memory.

The difference is unattributed runtime/driver/workspace overhead and remains
visible. Prefix sharing claims report bytes avoided, extra metadata, refcounts,
and copy-on-write cost.

INT8 c>N admission additionally reports and budgets persistent payload, K/V
scale planes, any BF16 mirrors, packed/split-K workspace, the selected prefill
hidden/oracle workspace share, graph-pinned ownership, and a configured
whole-device reserve. A page-only estimate is incomplete even when pool
high-water rejection works. Estimated versus measured bytes and the rejecting
resource are required campaign evidence.

### No benchmark gaming

- Do not specialize to fixed prompts, token IDs, candidate IDs, or measured
  completion patterns.
- Do not tune only the first layer or one synthetic row shape and generalize the
  claim.
- Do not compare c=N aggregate throughput against c1 without per-request and
  serial-bridge ratios.
- Do not promote diagnostics with stale hardware, a different model fingerprint,
  or undeclared fallback paths.

## Failure handling

| Failure | Action |
| --- | --- |
| Primitive mismatch | Stop model work; add/fix the primitive RED fixture. |
| First undeclared hidden drift | Capture the earliest layer/stage; verify it is the declared T1/T2 boundary before tuning downstream kernels. |
| Token equality but state/KV ownership mismatch | Reject lifecycle gate in every profile; token coincidence is insufficient. |
| Production ID divergence | Hold teacher contexts fixed and localize mean/tail KL, margin, transition, isolation, and task quality; do not call it a state bug or a pass from IDs alone. |
| Exact route slower than serial | Keep as correctness anchor, profile family/launch walls, and do not promote. |
| Graph/eager mismatch | Disable replay for that backend/bucket and keep eager canonical. |
| Pool pointer changes under replay | Invalidate/rebind graph before reuse; treat stale execution as correctness failure. |
| Unsupported C/context/KV dtype | Fail closed to declared groups or c1 and record the fallback. |
| Cancellation harms neighbors | Reject production loop; cancellation must be row-scoped at commit. |
| OOM/high-water rejection | Reject admission atomically; do not partially mutate request or KV state. |

## Documentation and result discipline

For each meaningful iteration:

1. Put commands, measurements, diagnosis, and next action in a new immutable
   `worklog/entries/` file.
2. Put compact machine-readable evidence in `benchmarks/results/` when the
   benchmark protocol applies.
3. Add only a one-line result pointer here if it changes current truth,
   coverage status, phase exit, or queue order.
4. Update `benchmarks/README.md` and `benchmarks/CHANGELOG.md` only for retained
   benchmark rows under the repository evidence policy.
5. Update `docs/REFACTOR.md` when an experiment flag, duplicate path, or fallback
   should be removed after promotion.

Do not append experiment narratives, profiler tables, or per-iteration queues to
this file. Git history and immutable `worklog/entries/` preserve them.

## Definition of done

This roadmap is complete only when all of the following are true for both GGUF
and PARO on both gfx1100 and gfx1151:

- [ ] c1/c2/c4/c8 native model steps pass the declared strict, production, and
      batch-invariant gates, with exact control/state/KV ownership in all three.
- [ ] Sparse retirement, cancellation, compaction, and new admission pass while
      neighbors remain live.
- [ ] One model-owning loop serves `LLM.generate()` and the OpenAI server.
- [ ] Prefill/decode scheduling, real device KV allocation, reclaim, streaming,
      and backpressure are integrated.
- [ ] Greedy and normal per-row sampling are supported under their gates.
- [ ] Profiler evidence proves the claimed native widths and replay paths.
- [ ] Retained c=N rows beat c1 aggregate and the serial bridge without
      regressing per-request latency or memory outside documented tradeoffs.
- [ ] The first-class path is package-default; obsolete bridges/flags are removed
      or have concrete blockers in `docs/REFACTOR.md`.

Until then, the honest project claim is: **gfx1100 and gfx1151 GGUF have
retained native-c4/c8 direct model steps, honest arbitrary-C physical-group
lowering, and real OpenAI burst/live-admission scaling. Both gfx11 PARO targets
have retained explicit direct native-c2 model steps; gfx1151 additionally
retains true physical c4/c8 and a package-default resident blocking/SSE owner.
gfx1151 GGUF occupancy-adaptive c1/c2/c4/c8 serving and its production
load/SLO gate are retained, including scheduler-owned timeout/disconnect row
isolation and a 120-request soak. Opt-in gfx1151 active-current greedy p256+s1
prefix reuse is byte-exact and economics-retained; completed-source p256+s1
snapshot restore is also correctness/economics-retained at 11.332x TTFT with
exact eviction/final drain and 72,089,600-byte cache residency. gfx1100 now
passes the matching active/completed p256+s1 correctness transfer, without a
performance claim; its deterministic agentic p2048/p8192 economics/lifecycle
packet is also closed but rejects promotion. Broader historical boundaries and
sampled reuse remain open. Model-general tokenizer-aware constrained tool
sampling is now implemented on the host processed-logits path. The gfx1100 PARO
owner c4/c8, A4 width-transition exactness, long-context memory-pressure
coverage, and complete project-wide production continuous batching remain in
progress.**
