# gfx1151 PARO/GGUF Optimization Ledger

Last updated: 2026-07-11.

Status: active ledger. The P0 foundation (`SOL-E1` through `SOL-E5`, `SOL-B1`,
`SOL-M1`, and `SOL-D1`) plus `SOL-S2` is accepted on top of the
`7ea21e98b097` release-default baseline.
The gfx1151 HIP/Vulkan v2
matrix was measured at `ca241dae795d` with `hipengine_dirty=false`; the PARO
true-c1 shrinking gate was measured at `0c1845170955` with
`hipengine_dirty=false`.

This is the active coordinator for making the PARO and GGUF paths correct,
fast, memory-efficient, and scalable on gfx1151 without regressing gfx1100. It
consolidates the next work from:

- `MTP-LLAMACPP-PARITY.md`, the current GGUF MTP parity dashboard;
- `PARO-GGUF-MTP-TRANSFER.md`, the current PARO transfer queue;
- `HIP-vs-VULKAN.md`, the current compiler/runtime decision dashboard;
- `TUNING-gfx1151.md`, `TUNING-gguf.md`, and `CONCURRENCY.md`;
- the 2026-07-10 current-HEAD code and evidence audit.

`PLAN.md` remains the architecture source of truth. `BENCHMARK.md`
and `TESTING.md` remain the promotion contracts. Each dashboard links a
byte-for-byte `*-HISTORY.md` snapshot containing its dated lab notebook and
implementation record. This file owns the cross-cutting ordering,
prerequisites, and completion state.

## Scope And Completion

In scope:

- PARO and GGUF AR prefill/decode on gfx1151 and gfx1100;
- HTTP concurrency, true backend row width, continuous shrinking, and sparse
  resident slots;
- GGUF MTP and PARO MTP/DFlash routing, verifier economics, and commit paths;
- architecture-specific tuning through registry/config profiles;
- a bounded, corrected HIP/Vulkan comparison that can guide production work;
- memory residency, launch/synchronization, and host/device transfer costs.

Out of scope until a retained profile activates them:

- a broad Vulkan backend;
- generic hand-written ISA;
- speculative kernel rewrites without an exposed end-to-end bucket;
- new approximation modes or accuracy trades;
- prompt-specific acceptance tuning of any kind.

"Tried everything" has a bounded meaning: every unconditional item in this
ledger is accepted, rejected, or blocked with evidence, and every conditional
item is either run because its activation trigger fired or parked with that
trigger recorded. It does not mean enumerating arbitrary kernel variants.

Status values:

| Status | Meaning |
| --- | --- |
| `open` | Ready once its dependencies are satisfied. |
| `blocked` | A named prerequisite prevents useful work. |
| `conditional` | Run only when its activation trigger is present in a corrected profile. |
| `in_progress` | The current logical unit; name it in `WORKLOG.md`. |
| `accepted` | Correctness, end-to-end, artifact, rollup, and commit gates passed. |
| `rejected` | The premise was tested and did not pass; preserve the artifact and reason. |
| `parked` | Do not retry until the stated premise changes. |

Landing instrumentation is not `accepted` for a performance item.
"Done" means the exit gate in this document passed.

## Current Evidence Snapshot

The table names the source revision for each result.

| Area | Current defensible result | Qualification / immediate consequence |
| --- | --- | --- |
| GGUF MTP on gfx1151 | `llama-compat` B2 reports `71.52 tok/s` versus llama.cpp HIP `71.91 tok/s`, with hipEngine stage wall `14.005` versus `14.269 ms/output`. | This is an opt-in direct-commit/dp4a compatibility contract, not exact/default semantics. Keep it as a replication lane, not the production default. |
| Exact/default GGUF MTP | Fixed 10-cycle B5 reports `61.98` versus AR `54.79 tok/s`. | Natural `max_tokens=24` loses at B1/B2/B5: `52.13/52.04/50.65` versus AR `54.80`. Fixed-cycle rows do not close production MTP economics. |
| MTP server routing | After normalizing to generated token IDs, current evidence still favors MTP at c1/c2 and AR at c4/c8. | `SOL-E1`/`SOL-E2` fix future all-choice denominators and copied batch timing; `SOL-S2` now proves that a c8 client run under the current cap is two four-request queue/backend groups rather than one width-8 verifier. Prior absolute rates remain invalid and need a rerun under all contracts. |
| Exact server measurement | `SOL-E1` carries exact IDs through every choice; `SOL-E2` gives timing payloads explicit scope/row/owner metadata; `SOL-S2` separately records the request-scoped route cap, queue request/prompt grouping, actual backend calls/widths, and target verifier rows. | `mtp-bench.py` fails closed on incomplete shape groups and counts each timing owner and queue group once. Retokenized visible text remains non-authoritative; historical server rows predate these contracts. |
| Canonical artifact provenance | `SOL-E3` gives server, retained PARO, GGUF category/true-AR, and HIP/Vulkan micro artifacts one torch-free schema with dynamic backend/arch/device identity, separate staged/unstaged/untracked state, and content-derived model fingerprints. | New retained rows must contain a valid `hipengine_artifact_provenance` v1 block and an existing model fingerprint where a model ran. Legacy provenance remains diagnostic until rerun. |
| PARO c>N | Retained gfx1100 direct rows exist for c4/c8. The gfx1151 c1-c8 timing rows at `4175dabf`/`02aec604` used a batch-shaped width-1 oracle and cannot select routing. At `0c184517` with `hipengine_dirty=false`, serial c8-to-c1 passes 8/8 rows against independent c1; native decode fails 0/8, first mismatch at c8 generated token index 2. | Production greedy and sampled batches use exact width-1 sessions. Reopen every gfx1151 native width; localize the common c8 divergence before width-specific tuning. |
| GGUF eager correctness / gfx1100 refresh | On gfx1151, SOL-G1 proves the exact Q4_K_M `[9707] * 512` continuation matches llama.cpp and is byte-exact for four eager hidden/Conv/GDN/KV transitions. The last W7900 diagnostic remains about `654 prefill / 35.8 decode tok/s`. | Repetition of `9707` is valid model behavior, but the W7900 row is still `performance_claim=false` and predates the hardware-local oracle/provenance contract. Rerun both gates on W7900 before using the rate. |
| HIP versus Vulkan | The gfx1151 timing-contract v2 matrix at `ca241dae` records `hipengine_dirty=false`, retains 22/22 comparisons, and separates `serial_latency` from `independent_throughput`. Serialized production slices are mostly HIP-favored; synthetic packed dot and dispatch retain Vulkan leads. | Keep HIP as the production backend. gfx1100 still needs the same bounded v2 matrix; Q6 lm-head remains incomparable because the implementations use different math/layouts. |

The first sprint is therefore measurement and routing correctness, followed by
GGUF recovery and PARO shape safety. New speculative kernels come later.

## Non-Negotiable Gates

Every optimization unit must satisfy all applicable gates:

1. **Exact workload identity.** Record model fingerprint/revision, quant, KV
   dtype, prompt token IDs or prompt-suite hash, context/generation lengths,
   concurrency, choices, sampling mode, and speculative mode.
2. **Exact runtime identity.** Record configured and resolved backend, target
   arch, GPU, ROCm/HIP/compiler versions, build profile, hipEngine commit, and
   full dirty state including staged and untracked files.
3. **Correct denominators.** Count generated token IDs across every choice.
   Keep visible-text re-tokenization only as a separately named compatibility
   diagnostic. Never use it as backend throughput.
4. **Owned timing.** Every timing payload declares `timing_scope`,
   stable `batch_id` where applicable, and `group_rows`.
   Batch timing is counted once, never once per choice.
5. **Correctness before speed.** Math/kernel changes need RED coverage where
   practical, the relevant CPU/reference or generated-token oracle, finite
   outputs, and the documented KL/top-1 gate. State/KV/graph changes also need
   multi-step state equality, not only final top-1.
6. **End-to-end before promotion.** A microbenchmark or profiler sub-window may
   justify keeping an exact low-level win, but it does not support a server or
   engine throughput claim without the matching end-to-end gate.
7. **Architecture isolation.** A gfx1100 result cannot select a gfx1151 default,
   or vice versa. Unverified architecture rows are explicit.
8. **No hidden fallback.** Artifacts record requested and effective attention,
   linear-attention, MoE, projection, sampler, graph, and speculative modes.
9. **No benchmark gaming.** MTP/acceptance work uses the complete
   `mtpbench-code-general-ja.jsonl` category suite plus held-outs and
   a true same-protocol AR baseline.
10. **Atomic retention.** Accepted performance work updates the compact artifact,
    `benchmarks/README.md`, `benchmarks/CHANGELOG.md`,
    `WORKLOG.md`, and default route in the same logical unit unless a
    concrete blocker is logged.

## Canonical Accounting Contract

The server and benchmark harnesses must distinguish four shapes:

- HTTP concurrency: number of simultaneous client requests;
- choices `n`: outputs requested by one HTTP request;
- backend group width `C`: live requests advanced together;
- verifier rows `V`: flattened speculative rows processed by the target.

Non-streaming hipEngine responses expose these as
`hipengine.generation_shape` schema v1. The route cap is an object with
`scope="queue_requests"`; it is never interpreted as a backend-row or verifier
limit. `queue_group` records the coalesced HTTP-request count, total prompt
rows, and this response item's row slice. `backend_groups[]` records each
actual generator call and any internal width split. `verifier_rows` is the sum
of target rows across those backend calls. The harness deduplicates repeated
response copies by `queue_group.id` and requires every group item exactly once.

Required generated-work fields:

| Field | Definition |
| --- | --- |
| `choice_generated_token_ids` | Exact token IDs emitted by each choice. |
| `choice_generated_tokens` | Length of that choice's exact ID list. |
| `total_generated_tokens` | Sum across all choices and requests. |
| `draft_tokens` / `accepted_draft_tokens` | Speculative work only; never substituted for visible output. |
| `target_rows` | Target model rows actually evaluated. |
| `retokenized_visible_tokens` | Optional decoded-text diagnostic, clearly non-authoritative. |

Exact-token direct/server comparisons also retain
`hipengine.prompt_token_accounting`: input type, per-row token-ID SHA-256,
per-row lengths, and total prompt tokens. The raw rows enter the common
`GenerationRequest` and bypass PARO/GGUF tokenizers; the hash echo and generated
ID oracle must match before timing is comparable.

Required timing scopes:

| Scope | Examples | Aggregation |
| --- | --- | --- |
| `choice` | Per-choice stop/sample/output handling | Sum only when measuring total per-choice work. |
| `batch` | Packed prefill, native decode step, draft/verify/commit phase | Deduplicate by `batch_id`. |
| `request` | Queue delay, TTFT, request wall | Report distribution; do not sum into GPU work. |
| `client` | Whole benchmark wall/makespan | Denominator for aggregate server generated tok/s. |

Primary server metrics:

```text
aggregate_generated_tok_s = total_generated_tokens / client_makespan_seconds
per_request_generated_tok_s = request_generated_tokens / request_wall_seconds
backend_batch_decode_tok_s = dedup_batch_generated_tokens / dedup_batch_decode_seconds
```

Also report TTFT, inter-token latency, completion latency, makespan, and
p50/p95. A batch-wide timing copied to six choices must still contribute once.

Required provenance:

```text
configured_backend, resolved_backend, target_arch, device_name
model_path, model_revision, model_fingerprint, quant, kv_dtype
hipengine_commit, staged_dirty, unstaged_dirty, untracked_dirty
rocm_version, hipcc_version, build_profile, exact command and env
timing_protocol, warmups, repetitions, profiler identity/status
```

The existing stronger dirty-tree handling in
`scripts/gguf_mtp_category_bench.py` should become shared
infrastructure rather than being reimplemented inconsistently.

## Architecture And Shape Identity

gfx1151 may reuse gfx1100 source bodies, but it must not reuse gfx1100 semantic
identity. The resolved backend and target arch must flow through generator,
runner/session, build, registry resolve, tuning selection, telemetry, and
artifact creation.

Do not mechanically relocate every physical import from
`hipengine.kernels.hip_gfx1100`. Those modules are the shared source
lineage used by the gfx1151 alias layer. Remove semantic hard-codes instead:

- generator registry keys fixed to `hip_gfx1100`;
- model defaults fixed to `hip_gfx1100`;
- registry `resolve()` calls fixed to `hip_gfx1100`;
- wrapper/build defaults that ignore the resolved target;
- capability/provenance surfaces that report configured `auto` instead
  of the resolved backend.

Use one immutable architecture tuning profile selected at model/session build
time. It may contain chunk sizes, workgroups, rowtile limits, route caps,
attention splits, and graph policies. It must be keyed through registry/config
composition, not hot-path `if backend == ...` branches.

Any c>N algorithm decision uses at least:

```text
resolved backend + target arch + model fingerprint + quant + KV dtype
+ rows + context bucket + mode + active-mask shape
+ attention + linear-attention + MoE + projection + sampler + graph variants
```

An unknown key falls back to the serial/exact route and reports why.

## PARO/GGUF Parity Audit

This table prevents a win in one path from being forgotten in the other while
also preventing incompatible quant kernels from being copied blindly.

| Surface | PARO today | GGUF today | Required comparison / transfer |
| --- | --- | --- | --- |
| Backend identity | PARO has gfx1100/gfx1151 factories and carries backend/target arch. | `SOL-B1` tags resident GGUF models and every weight with the resolved backend; embedding, linear/fused-linear, router, GDN, and compact/sidecar MoE resolves rebind shared gfx1100 source templates to that identity. A live gfx1151 public smoke retained `hip_gfx1151` through generator/runner/model/weights and generated token ID `11`. | Backend identity is complete; establish the corrected baseline matrix before architecture-specific tuning. |
| Prefill chunking | gfx1151 all-256 chunking was a large diagnostic win. | SOL-G2 certifies the raw-Q/K exact GDN chain across segment/chunk boundaries, but clean G3 walls reject it at 512/4K (+5.19%/+6.70%). | Keep fused; retune chunks against the selected fused route unless a materially different exact-chain scheduler is proposed. |
| Decode graph | PARO has graph/bucket infrastructure, with path-specific evidence. | Fast GGUF graph was retired after third-and-later replay state corruption; SOL-G1 supplies the exact eager control and SOL-G4 profiles it in 24 decode-only marker windows. | Recapture by full shape/state key in G5 only against the retained eager wall/profile. |
| c>N decode | PARO has native multi-row paths, owned batch timing, and retained gfx1100 c4/c8 direct rows. | GGUF server has packed AR/verify work, exact all-choice IDs, and owned batch timing, but route width/shape identity remain incomplete. | Run the same c1-c8 and shrinking matrix on both paths. |
| Sparse slots | Runtime accepts sorted sparse physical slots. | Resident MTP slots are tracked, but actual group width must be exposed. | Remove generator compact-from-zero gating and test holes/reclaim. |
| Full attention | PARO uses shape-specific native/rowchunk bridges; several widths remain diagnostic. | GGUF AR/verify paths have separate packed behavior. | Bucket context, row width, reducer, KV ABI, and fallback separately. |
| GDN/linear state | PARO has segmented multi-row state work plus shape-specific fallbacks. | GGUF eager state is exact; the raw-Q/K split passes G2 but loses balanced G3 full-prefill wall at both primary contexts. | Keep fused as the GGUF default and preserve the exact chain as the required unfused fallback/bisection path. |
| MoE | Selected-c1 is already a true multi-row algorithm for even widths; grouped compact covers other diagnostics. | GGUF uses Q*_K/T16/X8 and dp4a-specific selected paths. | Transfer row/group policy and measurement, not quant kernel bodies. |
| Projection | PARO catalogs candidates, but evidence mixes architectures and row-only bounds. | GGUF replacement layouts and selected/dense paths are quant-specific. | Key catalogs by full identity; compare true weight reuse versus row-GEMV. |
| LM-head/sampler | PARO has batched LM-head evidence at some widths and serial fallback elsewhere. | GGUF uses Q6 rowtile/chunks; large rowtiles collapse. | Profile full lm-head + reduction + readback before new fusion. |
| Speculative lifecycle | PARO DFlash has coarse timing and graph-shape telemetry, not a current real GPU row. | GGUF has packed verify and deferred scatter work; copied group timing now has one stable owner, while verifier-row/actual-group identity remains open. | Complete shape identity, then compare accepted-row scatter, tail discard, commit, and sync. |
| Startup/cache | PARO/GGUF both contain warmup and resident-cache ideas. | Coverage and artifact identity differ. | Record cold, warmed, cache-hit, and shape-miss behavior explicitly. |
| Memory residency | PARO packed rows are near the consumer-card target. | SOL-G6 cleanly audits gfx1151 Q4_K_M p512/d128 at 21.478 GiB owned/tracked: 733 unique sources, no raw+replacement duplicates or enabled optional sidecars, and 2.522 GiB margin to 24 GiB. | Keep the replacement-only default; context-specific long-KV capacity remains a separate policy gate. |

GGUF Q*_K, T16/X8, q8_1/dp4a, and Q6 LM-head kernels are not PARO
`w4_paro` kernels. Transfer scheduler, lifecycle, shape, graph,
warmup, and accounting lessons. Transfer device math only after matching the
layout and profiled bottleneck.

## P0 Foundation Punchlist

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-E1` | Carry exact generated IDs/counts through `GenerationOutput` and OpenAI responses; aggregate every choice in `mtp-bench.py`. | `accepted`: PARO/GGUF outputs carry exact IDs; completion/chat `n=6` and retokenization-mismatch regressions pass; `mtp-bench.py` validates and aggregates all rows; API/benchmark semantics are documented. | none | Retokenization-mismatch and `n=6` regressions prove exact all-choice totals; usage semantics are documented. |
| `SOL-E2` | Add `batch_id`, `group_rows`, `timing_scope`, and timing owner; deduplicate batch metrics in harnesses. | `accepted`: the telemetry contract defaults unscoped timing to explicit choice ownership and requires complete batch ownership; PARO/GGUF live c2 groups expose one owner; `mtp-bench.py` rejects malformed ownership and deduplicates copied batch walls by ID. | none | Synthetic duplicate payload and live PARO/GGUF group tests count batch wall once. |
| `SOL-E3` | Create shared artifact/provenance helpers; detect backend/arch dynamically; include full dirty state and model fingerprint. | `accepted`: the stdlib/torch-free collector emits `hipengine_artifact_provenance` v1 for server, retained PARO, GGUF category/true-AR, and micro artifacts; dynamic gfx1151 identity resolves from `auto`; staged, unstaged, untracked, snapshot-revision, file, directory, and missing-model cases have regressions. | none | Server, PARO retained, GGUF, and micro artifacts satisfy one schema; staged/untracked tests pass. |
| `SOL-E4` | Repair dashboards: remove `performance_claim=false` rows from "Current fastest," correct server token headlines where raw IDs suffice, and mark timing rows awaiting rerun. | `accepted`: the canonical and root current-topline tables contain only retained speculative rows; stale/false-claim model, capacity, and concurrency numbers are replaced by linked rerun notices; historical server rows explicitly await an exact-ID/scoped-timing rerun. | E1-E3 | Current tables contain only eligible rows; diagnostics remain linked in a separate section. |
| `SOL-E5` | Add an exact-token server benchmark route shared by PARO/GGUF direct and server runs. | `accepted`: raw token rows are a common generation input; OpenAI `int[]`/`int[][]` prompts preserve IDs through batching and `n`; prompt hashes/counts and exact generated IDs are exposed; the shared tool/schema fail closed on parity; live gfx1151 PARO 512/128 direct/HTTP matched all IDs. | E1, E3 | 512/128 token-ID prompts produce the same prompt IDs and generated-ID oracle through direct and HTTP paths. |
| `SOL-B1` | Register GGUF for `hip_gfx1151` and thread resolved backend/target through generator, runner/session, registry resolves, builds, capabilities, and telemetry. | `accepted`: backend packages have a refreshable registration hook; the GGUF generator/runner defaults resolve `auto`; resident models/weights carry the concrete backend; embedding, single/fused linear, router, GDN, and compact/sidecar MoE registry paths use it. AST regression rejects literal gfx1100 resolver arguments, lazy gfx1151 reconstruction passes, and a live public gfx1151 smoke generated ID `11` with all layers tagged gfx1151. | E3 | gfx1151 factory/dispatch/build tests pass; no semantic gfx1100 resolver key remains on the selected path. |
| `SOL-B2` | Add registry/config-owned architecture tuning profiles without changing defaults. | `blocked` | B1, baseline matrix | Empty/equal profiles are behavior-identical; future gfx1151 values require same-device evidence. |
| `SOL-M1` | Add one matrix driver/report that joins exact tokens, scoped timings, path variants, latency, memory, and profiler summaries. | `accepted`: manifest/schema v1 normalizes PARO/GGUF direct/server rows, recomputes rates from exact IDs, deduplicates timing owners, preserves backend/verifier shapes, attaches memory/profiler artifacts, and rejects forged denominators or cross-scope ratios; the four-surface contract and real SOL-E5 PARO diagnostic smoke pass. | E1-E3, E5 | One artifact can compare direct/server and PARO/GGUF without manual denominator repair. |
| `SOL-D1` | Split the three source docs into a short current dashboard and dated lab notebook/history; reconcile stale concurrency and "Done" wording. | `accepted`: MTP parity, PARO transfer, and HIP/Vulkan are 94/87/87-line current dashboards with retained/diagnostic/open/blocked language; their 6,812/597/2,602-line notebooks are linked `*-HISTORY.md` files whose blob hashes exactly match the originals. | E4 | Each current dashboard contains only eligible results and open blockers; historical diagnostics remain linked and unchanged. |

The P0 foundation is accepted. Run the first baseline matrix before any
architecture tuning; do not combine backend plumbing with kernel tuning.

## Baseline Matrix

Run the first baseline immediately after the P0 foundation is green and before
performance or routing changes. gfx1151 is local first; gfx1100/W7900 is a
separate rerun when that hardware is available. Never merge the architectures
into one dispatch decision.

### AR Correctness And Throughput

| Matrix | Required rows | Purpose |
| --- | --- | --- |
| Short concurrency | c1-c8, prompt 512 / decode 128, every integer width | Find odd-width/c6 holes and record the actual backend group. |
| Mid-context concurrency | c1/c2/c4/c8, prompt 4K / decode 128 | Exercise attention/context buckets without exploding matrix cost. |
| Long context | c1 first at 32K/64K/128K; c2/c4 only after c1 is green and memory-safe | Validate chunk/KV policy and architecture-specific memory limits. |
| Dynamic shrink | Start c8 and force completion/cancel transitions through c7...c1 | Prove state/KV/slot correctness under live shape changes. |
| Sparse slots | Holes at front/middle/tail with sorted physical slots | Prove native decode is not accidentally compact-from-zero only. |
| Ragged context | Mixed prompt lengths within one group | Exercise per-row positions, spans, attention, and graph keys. |
| Sampling | Greedy first; then supported per-row normal sampling | Keep sampler correctness separate from core AR bring-up. |

For every c>N row report aggregate/c1, per-request/c1, native/serial, actual
group histogram, active occupancy, TTFT, inter-token latency, completion
p50/p95, makespan, memory, and exact generated-ID equality versus independent
c1.

### Speculative Economics

Use all categories in
`benchmarks/prompts/mtpbench-code-general-ja.jsonl` plus held-outs.

| Matrix | Required rows | Purpose |
| --- | --- | --- |
| Natural short horizon | c1/c2/c3/c4/c8, `max_tokens=24`, true AR and MTP | Establish immediate auto-routing policy with actual backend group widths. |
| Longer horizon | At least `max_tokens=64/128` for routes that survive natural24 | Measure setup amortization and avoid fixed-cycle conclusions. |
| Context buckets | Short and 4K first; longer only if route remains positive | Decide whether routing depends on attention/context cost. |
| Budget | Exact/default budgets first; compat is a separately labeled lane | Never merge accuracy-traded and exact economics. |
| PARO DFlash | Same prompt categories, same target AR protocol | Compare verifier/drafter lifecycle rather than headline from another path. |

Required speculative metrics include visible outputs/cycle, accepted/output,
target rows/output, draft/verify/commit wall, group width, route decision reason,
and a true no-spec AR baseline.

## GGUF Recovery And Optimization

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-G1` | Build a teacher-forced token, hidden, recurrent-state, and KV oracle for eager GGUF decode across at least four steps. | `accepted` on gfx1151 at `c941c158`: the exact Q4_K_M `[9707] * 512` prompt and five-token continuation match llama.cpp; production bulk/eager tokens match; positions 513-516 are finite and byte-exact against fresh serial prefixes across every layer output, 30 Conv/GDN pairs, and 10 live K/V pairs. The old W7900 performance row still requires a hardware-local rerun. | E3, B1 | Repeated-token/current eager behavior is classified as correct or localized to the first divergent layer/state. |
| `SOL-G2` | Add explicit GDN prefill `fused|chain|auto` diagnostic selection. Reproduce the 17-token mismatch and bisect first hidden/recurrent divergence. | `accepted` at `332f01f8`: the RED localized normalized-Q/K materialization as the first layer-0 recurrent divergence. The GGUF-only raw-Q/K-plus-scale exact split passes 6/6 clean gfx1151 cases: greeting, 512, 1024/1025 segment threshold, and 4095/4096 four-chunk boundary. Sampled tokens, FP32 hidden seeds, and resident Conv/GDN state are byte-exact; greeting/512 all-layer rows are exact. Focused tests pass 48/48 and the expected zero-scratch kernels appear in a cached-only trace. | G1 | Chain matches target tokens/state at short, 512, 4K, segment, and chunk boundaries. |
| `SOL-G3` | Promote the split prepare + segmented-k2 + RMSNorm chain only if same-run wall wins. | `rejected` at clean `ad773eba`: exact timed tokens and the linked G2 state/trace gates pass, but four balanced repetitions show chain `1248.436` vs fused `1186.842 ms` at 512 (+5.19%) and `10870.022` vs `10187.300 ms` at 4K (+6.70%). Fused remains default. | G2 | Exact state/tokens plus prefill wall win on both primary contexts; expected kernel trace present. |
| `SOL-G4` | Bisect correct eager decode against the last fast revision and profile the correct route by layer family. | `accepted` on gfx1151 at clean `5f4c6561`: p512/d128 exact eager is 49.285 tok/s; direct-parent `4499fb13` is the 17.799 -> 54.963 tok/s (+208.79%, 3.088x) library-cache boundary; current p8 is 55.208 tok/s. Twenty-four exact ROCTX windows yield the decode-only Amdahl table below. W7900 remains blocked on hardware. | G1 | Correct eager baseline, first performance-changing revision, and Amdahl table are recorded. |
| `SOL-G5` | Rebuild correct graph replay by full shape/state key; test third-and-later replay explicitly. | `accepted` on gfx1151 at clean `7f611fe3`: the production graph is exact for all 128 hidden/Conv/GDN/KV/token checkpoints and capture-inclusive wall improves same-run eager `20.3343 -> 20.3115 ms/token` (+0.112% throughput). Admission is gfx1151-only, non-streaming c1 greedy, and at least 128 remaining transitions. W7900 remains blocked on hardware. | G4 | Eager/graph hidden, recurrent state, KV, and tokens match over long replay; wall beats eager. |
| `SOL-G6` | Audit replacement layout residency and eliminate raw+packed duplicates where the replacement path is complete. | `accepted` on gfx1151 at clean `d70c9464`: 733 unique source tensors map to one resident layout each, with zero raw+replacement duplicates and zero enabled optional sidecars. The p512/d128 BF16-KV production graph session is 21.478 GiB owned/tracked (2.522 GiB under 24 GiB); graph/exec adds 0 tracked bytes and 308 KiB sampled HIP residency. G5 is linked by SHA-256 for exact speed non-regression. | E3 | Allocation census names raw/packed/KV/scratch/graph bytes; 24 GiB-class goals are checked without speed regression. |
| `SOL-G7` | Tune gfx1151 chunk, workgroup, rowtile, attention split, and route thresholds. | `blocked` | B1-B2, G2-G4, matrix | Same-device exact A/B selects profile values; gfx1100 remains unchanged. |
| `SOL-G8` | Replace GGUF serial/row-replay concurrency with a true resident multi-row AR path across c1-c8 and sparse slots. | `blocked` | G4, baseline matrix | Exact c1-c8, shrink, sparse-slot, and profiler gates pass with aggregate scaling. |
| `SOL-G9` | Narrow HIP Q4 selected-dual recovery using source/layout/reduction/waitcnt changes. | `parked`: corrected V6 serialized Q4 is parity/HIP-favored (`0.922x-0.973x` Vulkan/HIP), so the activation premise is false; G4 also identifies dense Q8 and selected-MoE GEMV as the dominant production families. | corrected V6 result and real profile | Activate only if serialized matched Q4 still favors Vulkan and Q4 is material in production wall. |
| `SOL-G10` | Four-wave Q6 verifier LM-head rowtile: each wave owns four output columns to reduce accumulators. | `parked`: G4 attributes 10.06% of eager GPU time to Q6 LM-head, behind dense Q8 and selected-MoE; the exact small-B rowtile is already retained, while the later rowtile+top1 server experiment was flat/rejected. The “Q6 remains dominant” trigger is false. | E2, corrected profile | Activate only if Q6 head remains dominant; R6/R8/R12 show no spills, exact output, lower GPU event time, and server wall win. |

Do not restore the old GGUF graph as a shortcut. SOL-G1 proves the repeated
`9707` stream is valid on gfx1151; it still does not make a stale timing row
eligible or replace the full prompt-suite quality/performance gates.

### SOL-G4 gfx1151 eager Amdahl

The retained audit uses clean `5f4c6561`, exact Q4_K_M, BF16 KV,
`[9707] * 512`, four decode warmup steps, and 24 synchronized eager steps.
Only kernels fully contained in each ROCTX step range enter the table. Kernel
sum is **18.402 ms/token** versus **20.766 ms/token** profiled host wall
(**88.62%**); the unprofiled four-run p512/d128 median is **20.290 ms/token**.

| Family | GPU us/token | GPU share | Overall if family 2x | 4x | Infinite |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense Q8_0 GEMV | 8142.998 | 44.25% | 1.284x | 1.497x | 1.794x |
| Selected-MoE GEMV | 3996.701 | 21.72% | 1.122x | 1.195x | 1.277x |
| Full-attention core/KV | 1965.390 | 10.68% | 1.056x | 1.087x | 1.120x |
| Q6 LM-head/argmax | 1851.998 | 10.06% | 1.053x | 1.082x | 1.112x |
| MoE router | 807.582 | 4.39% | 1.022x | 1.034x | 1.046x |
| GDN/linear attention | 705.191 | 3.83% | 1.020x | 1.030x | 1.040x |
| RMSNorm/RoPE | 532.355 | 2.89% | 1.015x | 1.022x | 1.030x |
| Dense BF16 GEMV | 192.909 | 1.05% | 1.005x | 1.008x | 1.011x |
| MoE combine/SiLU | 165.271 | 0.90% | 1.005x | 1.007x | 1.009x |
| Other + embedding + copies/fills | 41.641 | 0.23% | 1.001x | 1.002x | 1.002x |

The old 2026-06-29 whole-process profile included prefill and warmup and is not
the G4 Amdahl source. The full current table, top kernels, VGPR/scratch counts,
commands, trace hashes, and linked G1 SHA are in
[`2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json`](../benchmarks/results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json).

### SOL-G5 gfx1151 production graph

The retained clean `7f611fe3` audit exercises the production
`capture_decode_graph()` API rather than a benchmark-local reconstruction. Its
key covers backend/target, model and quant identity, layer/KV layout, route and
sampler, resident weight and buffer pointers, recording options, state
generation, and a bounded context/replay window. Eager and stable graph replay
match byte-for-byte through all 128 third-and-later launches; conservative
state-generation recapture also passes 128/128 but is too slow to retain.

| Route | Median ms/token | Throughput | Decision |
| --- | ---: | ---: | --- |
| Same-run eager | 20.3343 | 49.178 tok/s | Control |
| State-bound graph, capture inclusive | 20.3115 | 49.233 tok/s | **Retained, +0.112%** |
| Per-token state-generation recapture | 35.4290 | 28.225 tok/s | Rejected |

The graph row charges one capture/instantiate plus final destroy to each
128-token window. The strict 128-transition gfx1151 threshold and eager
fallbacks remain part of the claim. Full commands, samples, key, provenance,
and checkpoint hashes are in
[`2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json`](../benchmarks/results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

### SOL-G6 gfx1151 replacement residency

The clean `d70c9464` census runs the production p512/d128 graph session and
audits both the materialization plan and live owned buffers. All **733** source
tensors have one planned resident layout; there are **zero** same-source
raw+replacement duplicates and **zero** enabled optional raw/X8 sidecars.

| Resident family | GiB | Notes |
| --- | ---: | --- |
| Replacement weights | 20.461 | Q4/Q5/Q6/Q8 T16 replacements; no optional X8 sidecar |
| Required raw GGUF | 0.503 | Device token embedding, not a duplicate |
| Dense weights/metadata | 0.097 | F32/BF16 residents |
| Decode scratch | 0.080 | 15 MiB BF16 KV, 63.75 MiB linear state, metadata/other |
| Session/prefill buffers | 0.337 | 0.330 GiB bulk-prefill scratch dominates |
| **Owned/tracked total** | **21.478** | **2.522 GiB below the 24 GiB gate** |

The production `record_steps=0` graph owns no tracked `DeviceBuffer`; the
synchronized live-minus-closed `hipMemGetInfo` delta is **315,392 bytes**
(308 KiB) for HIP graph/exec internals. Tracked allocations return exactly to
their pre-load baseline after session close. The artifact links the accepted
G5 SHA-256 for 128-launch exactness and capture-inclusive speed non-regression;
G6 itself makes no new throughput claim. Full allocation maps, close deltas,
commands, and clean provenance are in
[`2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json`](../benchmarks/results/2026-07-11-sol-g6-gfx1151-gguf-residency-audit.json).

## PARO Concurrency And Optimization

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-P1` | Run the exact c1-c8 512/128 matrix and publish a full shape/algorithm catalog per architecture/model. | `blocked`: the prior gfx1151 c2-c8 timing matrix used a batch-shaped width-1 oracle. At `0c184517` with `hipengine_dirty=false`, native c8 fails independent c1 at generated token index 2. gfx1100 remains stale. | P0 foundation; native divergence bisect | Every width is green or explicitly serial; no gfx1100 evidence silently selects gfx1151. |
| `SOL-P2` | Run c8->c1 EOS/cancel shrink, ragged contexts, and sparse-slot transitions. | `blocked`: at `0c184517` with `hipengine_dirty=false`, gfx1151 serial c8-to-c1 passes 8/8 rows and native passes 0/8. A full c3-to-sparse-c2 serial run passes and native fails. Ragged contexts remain unrun; resume after the common native divergence is localized in P1. | P1 | Per-row state/KV/output identity matches independent c1; no group-wide cancellation. |
| `SOL-P3` | Remove the generator's compact-from-zero gate and use sorted sparse physical slots accepted by `step_batch_native()`. | `blocked`: CPU/runtime addressing supports sorted sparse slots, but native sparse decode is correctness-red. Production uses width-1 sessions, so holes do not select an uncertified native route. | P2 RED tests; native correctness | Holey live groups stay native and exact; no serial fallback solely because of slot holes. |
| `SOL-P4` | Make selected-c1 MoE a named multi-row algorithm and compare it with grouped-compact at every supported width. | `blocked` | P1, scoped profiler | Full-layer/server wall, routed-lane counts, and correctness select the mode; c6's current advantage is rechecked. |
| `SOL-P5` | Close odd-width and c6 attention/linear/MoE/projection/sampler shapes with full identity keys. | `blocked`: the prior c3/c5/c6/c7 green rows used the invalid batch-shaped oracle. The common native c8 path fails before any width transition, so width-specific promotion evidence is reopened. | P1 | c3/c5/c6/c7 are retained-safe or serial; unproven rowchunk/grouped routes cannot auto-select. |
| `SOL-P6` | Benchmark c6 direct versus sequential c4+c2 splitter with all-choice counts and latency distributions. | `blocked`: the splitter is default-off; neither direct c6 nor c4+c2 is independent-c1-certified on gfx1151. | E1-E2, P1 | Keep a splitter only for an explicitly chosen latency objective; aggregate throughput, makespan, and p95 are non-regressive for that policy. |
| `SOL-P7` | Capture/replay decode buckets keyed by active rows, context, mask, variants, and replay length. | `blocked` | P1-P5 | Cache hit/miss/fallback telemetry is complete; exact replay improves server wall for retained shapes. |
| `SOL-P8` | Retune gfx1151 prefill chunks, AOTriton threshold, projection, and sampler modes after the route is shape-safe. | `blocked` | B2, P1-P5 | Same-device profile and end-to-end matrix select values without gfx1100 regression. |
| `SOL-P9` | Replace row-parallel GEMV with weight-reusing MMQ/GEMM/WMMA/grouped algorithms where c>N profiles justify it. | `conditional` | P1 profiler | Activate per family when weight reload/occupancy is material; prove c1 non-regression and c>N aggregate wall win. |

The c6 splitter is an opt-in diagnostic policy. The schema-1 c2-c8 profile has
`performance_claim=false` and an invalid batch-shaped oracle, so the loader
rejects it. Production uses `scheduler_true_c1_fallback`. A schema-2 profile
must pass independent-c1 packed-prefill, sparse-slot, and shrinking gates before
the planner can select native groups.

For c9-c16, `scripts/qwen35_batch_retained_bench.py
--batch-decode-execution=profile_partitioned` remains a diagnostic driver. Use
`--batch-decode-execution=serial` for the matched fallback control and
`direct_native` only for correctness localization. Production c9+ requests use
reused width-1 sessions until an accepted schema-2 profile exists.

## MTP, DFlash, And Routing

| ID | Work | Status | Dependencies | Exit gate |
| --- | --- | --- | --- | --- |
| `SOL-S1` | Move `auto` MTP choice from per-request eligibility to the realized backend group. | `blocked` | E1-E2, natural matrix | Initial policy is c1/c2 MTP, c4+ AR, c3 measured; explicit opt-in always requests MTP. Policy records reason/group/horizon. |
| `SOL-S2` | Record route cap, actual backend group, queue grouping, and verifier rows separately. | `accepted`: non-streaming server responses emit `generation_shape` v1 with a request-scoped cap, queue-group ID/request/prompt counts and item slice, actual backend call widths, and deduplicated verifier rows; `mtp-bench.py` validates complete groups. The c8 regression produces two c4 queue/backend groups, never a width-8 verifier row, and the opt-in c6 splitter reports c4+c2 calls. | E2 | A c8 client row cannot be mistaken for a width-8 verifier row. |
| `SOL-S3` | Add context/output-length buckets and EWMA hysteresis only after static policy is stable. | `blocked` | S1 retained | Online policy beats/equals static on held-out full-suite traffic without prompt-conditioned branches. |
| `SOL-S4` | Run a real PARO DFlash row using the landed coarse phase and graph-shape telemetry. | `open` | E1-E3 | Same-session AR, exact output, phase coverage, and shape hit/miss data identify the dominant parent bucket. |
| `SOL-S5` | Compare GGUF deferred accepted-row scatter/tail discard with PARO verifier commit/canonicalization. | `conditional` | S4 profile | Activate only if commit/scatter/sync is material; exact state/KV and cycle/server wall must improve. |
| `SOL-S6` | Add true draft-side batching and/or wider verifier groups. | `conditional` | S1-S4 profile | Activate only if current phase serialization/group caps dominate; retain on full suite and server wall. |
| `SOL-S7` | Re-evaluate LM-head/top1 fusion, readback, and sampler boundaries. | `conditional` | corrected scoped profile | Existing generic fusion/readback probes stay rejected unless a changed shape exposes the bucket again. |

Do not retry generic LM-head fusion, deferred readback, rowtile+top1, broad
route-cap increases, or confidence policies merely because attribution moved.
Require a changed premise and name it in the new artifact.

## HIP/Vulkan Measurement Repair

### Current Claim Classification

| Evidence | Current use |
| --- | --- |
| Dispatch/grid floor | Retained v2: Vulkan/HIP `1.162x-16.789x` serialized and `1.116x-150.459x` independent. This is runtime/submission evidence, not compiler evidence. |
| Geometry/reduction/sampler/two-stage | Retained v2: HIP wins or is mixed under required ordering; Vulkan wins independent overlap. Timing mode is part of the workload. |
| Synthetic packed dot | Retained v2 Vulkan lead: `3.052x-3.243x` serialized and `3.840x-4.272x` independent. |
| Production Q4/Q6/Q8 slices | Retained v2: serialized Q4 is parity/HIP-favored, Q6 is about `1.82x` HIP-favored combined, and dense Q8 is HIP-favored on every serialized row. |
| Q6 lm-head HIP T16 versus Vulkan q8_1/X8 | Blocked: different math/layouts, so no cross-backend ratio is permitted. |
| ISA dot4/VOPD/waitcnt/spill counts | Structural evidence remains valid. |

### Harness Contract V2

Implemented at `ca241dae` with `hipengine_dirty=false`; the executable contract is in
`benchmarks/micro/timing_contract.py`, the v2 schema, and the HIP/Vulkan runner
headers. The retained bounded artifact is
`benchmarks/micro/results/gfx1151/strix-halo/2026-07-10-hip-vulkan-timing-v2-bounded.json`.

1. Use `serial_latency` and `independent_throughput` modes.
2. In serial mode, add compute-to-compute execution and memory dependencies
   between every Vulkan repetition, including WAW and read-to-next-write hazards.
3. In independent mode, use disjoint outputs and compare with a HIP
   multi-stream/independent-graph path.
4. Record Vulkan GPU timestamps and host submit-to-fence wall separately.
   Record HIP event time and equivalent host wall.
5. Include `reps=1` plus a burst, and equalize warmup by dispatch count.
6. Validate the actual timed N-repetition command, not a separate one-dispatch
   command. Run Vulkan synchronization validation outside timing.
7. Match input bytes, selected IDs, algorithm/layout, output dtype, workgroup,
   cache state, and hot versus rotating working sets.
8. Extend the micro result schema with timing mode, dependency contract,
   timestamp metadata, memory flags, commit/dirty state, and claim eligibility.

### Bounded Rerun

| ID | Family | Corrected gfx1151 anchors | Status |
| --- | --- | --- | --- |
| `SOL-V1` | Harness/schema | Implement the contract above and a dependency litmus test. | `accepted` at `ca241dae` |
| `SOL-V2` | Dispatch | counts 1/50/941, grids 1/8192, reps 1 and burst, both modes/timings. | `accepted` on gfx1151 |
| `SOL-V3` | Geometry | K 512/8192, rows 1/8, wg 64/128/256. | `accepted` on gfx1151 |
| `SOL-V4` | Sampler | top-1/top-k8, rows 1/8, vocab 32768, wg 64/128/256. | `accepted` on gfx1151 |
| `SOL-V5` | Dot/memory | q8/q4 N=32768; coalesced-4 plus gather control; wg 64/128/256. | `accepted` on gfx1151 |
| `SOL-V6` | Q4 selected-dual | Active production layout, 4x32, 2048->512, HIP/Vulkan wg 64/128/256. | `accepted`: parity/HIP-favored |
| `SOL-V7` | Q6 selected-down | rows 8, 512->2048, wg 64/128/256. | `accepted`: HIP-favored |
| `SOL-V8` | Two-stage control | K 32768, rows 1/8, wg 128/256, split 4, serialized. | `accepted` on gfx1151 |
| `SOL-V9` | HIP independent control | Multi-stream/disjoint-output throughput against Vulkan independent mode. | `accepted` on gfx1151 |
| `SOL-V10` | gfx1100 portability | Repeat corrected anchors and every gfx1151 delta above 5%. | `blocked` on W7900 hardware |
| `SOL-V11` | Q6 LM-head | Same math/layout at rows 1/8, 2048->152064. | `blocked`: no matched math/layout implementation |
| `SOL-V12` | Production Vulkan probe | Persistent registry Q4/sampler object and real engine wall. | `parked`: corrected production slices do not justify backend work |

The gfx1151 matrix is complete. Run V10 on W7900 before transferring any ratio.
The gfx1151 result does not justify an LLVM issue, inline ISA program, or Vulkan
registry path.

## Profiling And Optimization Loop

For every performance item:

1. Select the highest actionable end-to-end or verified sub-window bucket.
2. State the hypothesis, affected shape keys, baseline artifact, expected
   movement, and stop condition in `WORKLOG.md`.
3. Add the narrow RED oracle before math/state changes, or log why RED-first is
   impractical.
4. Make one logical change. Keep a registered unfused/exact fallback.
5. Run the narrow correctness gate, expected-kernel trace, and same-suite A/B.
6. Use at least three timing samples for retention and apply variance rules.
7. If exact and non-regressive, promote the default and commit the artifact and
   rollups. If rejected, record the measured reason and remove or ledger the
   temporary path in `REFACTOR.md`.
8. Refresh this ledger status/result link before taking the next item.

Prioritize by wall reduction, but retain exact cycle-wall, verified sub-window,
launch-count, and H2D/D2H improvements even when aggregate variance hides a
small compounding win, as required by the project evidence policy.

## Execution Order

| Order | Work package | Items | Why now |
| ---: | --- | --- | --- |
| 1 | Exact accounting and provenance | E1-E3, S2 | All later routing and wall decisions depend on correct denominators. |
| 2 | Exact server route and backend identity | E5, B1 | Makes gfx1151 GGUF and direct/server comparisons real. |
| 3 | Harness/report and evidence cleanup | M1, E4, D1 | Establishes one trustworthy dashboard. |
| 4 | Current-HEAD baseline matrix | P1-P2 plus GGUF AR/spec matrices | Separates architecture, row, context, and lifecycle failures. |
| 5 | gfx1100 GGUF recovery | G1-G6 | Largest plausible performance recovery; correctness first. |
| 6 | PARO shape-safe native batching | P3-P8, then P9 if activated | Converts diagnostics into deployable c>N behavior. |
| 7 | Batch-aware speculative routing | S1-S4, then conditional S5-S7 | Uses corrected economics rather than per-request guesses. |
| 8 | Targeted non-backend kernel work | Activated P9/S items; G10 is parked | Only profiled, production-shaped kernels enter here. |
| 9 | HIP/Vulkan portability rerun | V10 | gfx1151 V1-V9 are complete; W7900 remains. |
| 10 | Backend/ISA decision | V11-V12 if activated; G9 is parked | Broad backend work requires a corrected production gate. |

Cross-GPU work is not allowed to block useful local gfx1151 progress, but no
gfx1100/gfx1151 shared default is promoted without both architectures or an
explicit architecture-specific profile.

## Definition Of Sprint Closure

The optimization ledger can be called complete only when:

- all P0 accounting, provenance, backend identity, and dashboard items pass;
- PARO and GGUF have exact current-HEAD baselines with trustworthy denominators;
- every PARO c1-c8 width and c8->c1 transition is retained-safe or explicitly
  serial with a named blocker;
- gfx1100 GGUF repeated-token/state behavior is resolved and the correct eager
  path is profiled; GDN chain and graph are accepted or rejected with evidence;
- MTP `auto` uses actual group/horizon economics and explicit opt-in
  remains available;
- a real PARO DFlash profile either activates or parks each transfer candidate;
- every currently relevant conditional kernel is accepted, rejected, or parked
  by its trigger;
- HIP/Vulkan retained timing language is rebuilt from synchronized measurements,
  with throughput and latency kept separate;
- current dashboards contain only eligible claims, while rejected/diagnostic
  history remains discoverable.

The next PARO GPU unit is `SOL-P1`: localize the native c8 token-index-2
divergence with teacher-forced hidden, linear-state, KV, and token comparisons.
Do not resume c3/c5/c7 or c>8 performance tuning until that common path matches
independent c1.

The next PARO GPU unit is `SOL-P1`. G7/G8 now wait on the corrected baseline
matrix and B2 profile plumbing; G9/G10 are parked because their triggers did
not fire, so there is no independent GGUF unit ahead of P1.
G2/G3 establish fused as the exact, measured prefill default. The matrix driver
is ready, but its first clean repeated PARO/GGUF baseline is measurement work,
not evidence implied by these correctness gates.
