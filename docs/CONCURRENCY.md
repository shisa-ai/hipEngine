# Concurrency and Continuous Batching

Last updated: see git log.

This document is the working guide for turning the current single-request
resident runtime into a vLLM-style continuous-batching serving path on a
single GPU. It covers what serving looks like when this work is done, what is
implemented today, the contracts the implementation must satisfy (engine loop,
elastic KV pool, prefix sharing, per-row sampler, streaming, observability),
and the benchmark gates a retained c>N row must pass.

Tensor parallelism (TP), expert parallelism (EP), compact DMS, and speculative
decoding (MTP / DFlash / EAGLE3) are **out of scope here** and live in their
own feature branches and docs. Concurrency-side decisions that must not paint
those follow-ons into a corner are called out in
§[Forward-compatibility guardrails](#forward-compatibility-guardrails).

Related source-of-truth docs:

- [`PLAN.md`](PLAN.md) — architecture invariants, long-form concurrent decode
  design, and §Multi-GPU Strategy for the TP/EP follow-on.
- [`BENCHMARK.md`](BENCHMARK.md) — evidence policy and c=N benchmark gates.
- [`KVCACHE.md`](KVCACHE.md) — dense INT8 KV capacity path and the compact DMS
  roadmap (the next-feature DMS plan).
- [`PREFILL.md`](PREFILL.md) — native/compact prefill details.
- [`ENVS.md`](ENVS.md) — knobs introduced by this doc.

## Definitions

| Term | Meaning |
| --- | --- |
| HTTP concurrency | Multiple client requests are in flight at the server at once. This can still be serialized internally. |
| Prompt-list batching | One API call carries multiple prompts, e.g. OpenAI completions `prompt=[...]`. Counts as true c>N only if the generator advances those prompts together. |
| c>N decode | `N` independent live requests each advance one target token in the same model step. |
| Continuous batching | The scheduler can admit, prefill, decode, finish, compact, and reclaim requests while other requests keep running, under a single long-lived engine loop. |
| Engine loop | One long-lived scheduler tick driving admission, prefill, decode, verify, reclaim, and pool resize across all active requests. |
| Elastic KV pool | Dense paged KV backed by an allocator that can grow and shrink between admission cycles up to a high-water cap. |
| Append-only block id | Allocator contract: a block id, once issued, keeps a fixed device pointer until freed; growth issues fresh ids past the current high water. |
| Prefix sharing | Multiple requests share refcounted KV pages for a common token prefix via a radix-tree index; the first divergent token forces a copy-on-write fork. |
| KVTC (KV tiered cache) | Future-direction multi-tier KV storage: hot HBM pages → pinned host RAM → optional NVMe spill, behind the same `KVLiveSpans` and block-id contracts so block ids stay stable across tier moves. KVTC is a follow-on feature branch, not in CONCURRENCY scope. |
| Per-row sampler | Sampling parameters (temperature, top-k, top-p, repetition penalty, seed, stop tokens) are independent per active row. |
| Packed/native prefill | Multiple prompt rows packed into one prefill slab and launched through row-shaped kernels. |
| Serial bridge | A correctness-first path with batch-shaped slots/KV metadata but active rows execute through the c=1 layer path. Diagnostics only; not a throughput claim. |

## Destination state

When this work is done, hipEngine on a single W7900 (or any single supported
GPU) runs as:

- One long-lived **engine loop** (one background driver thread under `hipengine
  serve` and `LLM.generate()`) admits new HTTP requests mid-stream up to
  *current* pool capacity, grows the **elastic KV pool** toward a high-water
  cap when load demands, and shrinks back toward a low-water floor when idle.
- The loop interleaves **chunked prefill** with **decode** under an explicit
  prefill-vs-decode policy; finished requests are reclaimed at the next commit
  point without waiting for the longest active request.
- Common token prefixes are shared via **refcounted KV pages**; the first
  divergent token forks a request onto fresh pages. `n>1` lowers to N
  scheduler requests with a shared prefix.
- The **per-row sampler** lets requests with different temperature, top-k,
  top-p, repetition penalty, and stop tokens decode together in one step.
- **Streaming** and non-streaming traffic share the same loop and the same
  reclaim path; cancellation, client disconnect, EOS, and max-tokens are one
  unified path.
- **Per-request and per-pool observability** is exported on `/metrics`
  (Prometheus) and recorded in retained benchmark artifacts.

Primary target workloads are agentic loops and long-context multi-turn
chat; both depend on heavy prefix sharing across requests and across turns.
`n>1` lowering is the third major prefix-sharing consumer.

Single-GPU, single-process, single-rank. Multi-GPU TP, DMS compact KV,
RadixCache eviction policies under variable-span KV, multi-tier KV storage
(KVTC), and speculative decoding are explicit follow-on feature branches.

## Current answer

**hipEngine now has most host-side continuous-batching scaffolding in code, but
it still must not claim true retained c>N throughput.** Qwen/PARO BF16 c=2/c=4/c=8
generated-token equality vs independent c=1 is green, and c=2/c=4/c=8 projection
dispatch evidence selects the row-bounded native projection candidate from one
combined catalog. C2 now deliberately keeps the no-flag sampler on
`serial_lm_head` after an eight-run batched-sampler audit reproduced intermittent
`[137,104]` / `[82,137]` flakes; c4/c8 keep row-aware `batched_lm_head`. The
remaining hard gates are accepted retained scaling/graph replay evidence,
full-native c4/c8 attention without rowchunk diagnostics, and any residual
fallback labels.

What is in place:

- The server and `LLM.generate()` paths have prompt-list batching, `n>1`
  lowering, streaming through per-request queues, request ids, per-row seeds,
  and Prometheus metrics hooks.
- `SubmitPollTextGenerator` and `ResidentEngineLoop` provide a persistent
  `submit`/`poll`/`cancel` driver around `ResidentBatchScheduler` for tests and
  host integration, with `RECLAIM → ADMIT → PREFILL/DECODE` tick policy,
  per-request completion metadata, graph-bucket bookkeeping, and unified cancel,
  disconnect, EOS, max-token, and timeout reclaim.
- The KV/prefix scaffolding exists: `ChunkedKVPool` grows/shrinks in chunks,
  keeps append-only block ids, reports current admission capacity, supports
  shared-prefix refcounts and copy-on-write forks, and `RadixCache` indexes
  block-aligned token prefixes.
- Per-row sampling parameters and per-row EOS/reclaim are represented in the
  scheduler; artifact/schema gates prevent serial bridges, fallback execution,
  non-native sampler metadata, or incomplete timing/profiler payloads from being
  promoted as accepted retained c>N rows.

What is still not green:

- The retained Qwen/PARO native c>N decode path is experimental. BF16 primitive
  c=2/4/8 KV append/full-attention correctness passes, and generated-token
  equality now passes for the c=2/c=4/c=8 512/128 gates under the
  correctness-first auto projection path (no-selected batch metadata using the
  128-thread batch-GEMV QKV/Z diagnostic for c=2/c=4/c=8), native segmented
  linear state, batch-GEMV/Marlin linear output, grouped-compact MoE for
  c<=8 auto decode, native full-attention decode for c=2, and native
  row-chunked full-attention diagnostics for c=4/c=8. The c=2/c=4/c=8 profiler
  preflights now capture compact `rocprofv3 --kernel-trace` summaries with
  native batch attention, KV write, and `batch_argmax_stage{1,2}` sampler
  kernels. These artifacts remain
  blocked for retained/scaling claims until the native batch linear/full-attention/
  projection/native-dispatch paths, graph-replay profiler evidence, and scaling evidence are
  green.
- Hidden-state bisection now separates generated-token equality from hidden drift:
  focused L4/L8 controls keep tokens green, and the selected-c1 output replay
  diagnostic now consumes the segmented state's gated `recurrent_bf16` instead of
  recasting raw `recurrent_out`. With that fix, L1 selected-c1 projection/native
  segmented state passes with both selected-c1 and native output, and the matched
  L8 selected-c1 projection/native segmented state/selected-output + per-row-full
  probe is hidden/token green
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-selected-out-fixed-perrow-full-atol4e-3-focus1269.json`).
  The matching L8 native-output probe remains red at the old row-0 token idx-13
  boundary and first hidden failure step 6 / row 1
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-native-out-perrow-full-atol4e-3-focus1269.json`),
  but forcing the batch-GEMV output fallback clears that selected-projection/
  native-state/per-row-full probe
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`).
  With native projections restored and the same batch-GEMV output fallback, tokens
  stay green but hidden still fails at step 11 with first layer-0 stage drift at
  `qkv`
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-native-proj-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`).
  A new row-aware batch-GEMV QKV/Z projection diagnostic reduces that layer-0
  `qkv` drift below hidden tolerance but still leaves the same hidden-only step-11
  failure
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-batch-gemv-proj-state-out-perrow-full-atol4e-3-focus1269.json`).
  Forcing token-1 QKV/Z exactly while leaving native A/B projection, native
  segmented state, batch-GEMV output, and per-row full attention also stays
  hidden-only red at step 11
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-qkvz-native-ab-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`).
  Conversely, forcing token-1 A/B while leaving QKV/Z on the batch-GEMV path also
  stays hidden-only red at step 11
  (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-batch-gemv-qkvz-selected-ab-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`).
  The selected-all projection control is green, so both QKV/Z bit exactness and
  A/B exactness/amplification remain in the linear-projection closure set. A
  projection-split refresh added selected-rotary-input, selected-QKV-only, and
  selected-Z-only diagnostics; all three still fail the c=2 512/128 generated
  gate at row 1 / prefix 104 (`/tmp/hipengine-e2e-native-c2-512-128-selected-qkvz-input-native-ab-state-marlin-out-perrow-moe-full-per-row.json`,
  `/tmp/hipengine-e2e-native-c2-512-128-selected_qkv-native-other-state-marlin-out-perrow-moe-full-per-row.json`,
  `/tmp/hipengine-e2e-native-c2-512-128-selected_z-native-other-state-marlin-out-perrow-moe-full-per-row.json`),
  so the remaining c=2 projection fallback is the complete selected-c1 QKV/Z
  pair rather than rotary input alone or only one of QKV/Z.
  Fully native full-attention decode under the selected-projection/
  batch-GEMV-output control still fails (`/tmp/hipengine-e2e-native-c2-512-128-selected-qkvz-native-state-marlin-out-perrow-linear-moe-native_full.json`:
  prefixes `[82,137]`). Keeping native full-attention input/QKV/context, native
  post-attention, and batch layer-copy while using batch-GEMV full-attention
  output plus per-row full-attention MoE was generated-token green for c=2/c=4/c=8
  (`/tmp/hipengine-e2e-native-c{2,4,8}-512-128-selected-linear-native-full-batch-gemv-output-moe-native-post.json`, all rows prefix `137`).
  The narrower batched selected-c1 MoE default first removed the per-row linear and
  per-row full-attention MoE blockers while keeping c=2/c=4/c=8 equality green
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-batched-selected-moe-matrix/summary.json`),
  and the follow-up selected-c1 MoE matrix recorded `moe_decode_path=selected_c1_batch`,
  `moe_selected_c1_fallback_layers=0`, and no selected-c1 MoE decode blocker
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-selected-moe-promoted-matrix/summary.json`).
  The grouped-compact MoE blocker is now cleared for the c<=8 decode shape by
  using compact-row selected AWQ GEMV for grouped gate/up/down plus row-aware
  shared-expert GEMV: the all-grouped c=2/c=4/c=8 512/128 matrix is generated-token
  green with all rows prefix `137`, and the independent c=1 `native_batch` hidden
  oracle is hidden/token green for c=2 layer-limit 40 decode step 0
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-grouped-moe-small-gemv-matrix/summary.json`).
  The retained-bench and hidden-bisect defaults now promote `grouped_compact` for
  global, linear-attention, and full-attention MoE subpaths; the default matrix
  stays c=2/c=4/c=8 generated-token green and the default hidden oracle stays green
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-grouped-moe-default-matrix/summary.json`).
  This is still correctness-only evidence: no retained throughput/scaling claim is
  made until native projection/dispatch, profiler, and scaling gates pass.
  The promoted selected-QKV/Z metadata matrix now removes the generic QKV/Z
  projection decode blocker while keeping `native_caware_decode=false` and
  retained validation blocked until a real native projection path/projection-dispatch
  artifact exists
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-selected-qkvz-promoted-matrix/summary.json`).
  The retained-bench artifact builder now also removes the stale `blocked until
  generated-token equality passes` reason once the artifact itself proves
  generated-token equality; the c=2/c=4/c=8 evidence matrix remains green and
  non-retained
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-equality-gate-promoted-matrix/summary.json`).
  The same artifact-sanitizing pass now removes the stale BF16/context<1024
  full-attention support caveat when the child artifact records native full-attention
  decode, BF16 KV, and max context below 1024; the c=2/c=4/c=8 matrix remains
  green and non-retained
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-context-gate-promoted-matrix/summary.json`).
  At that point the C2.3/C2.4 blockers were native QKV/Z bit exactness and
  projection-dispatch/retained-evidence closure; row-aware batch-GEMV linear/full-attention outputs,
  grouped-compact MoE for c<=8 decode, and the selected-QKV/Z correctness
  path were green and no longer blocked correctness-default equality. A post-grouped
  projection refresh rejected `batch`, `batch_gemv`, `batch_gemv_selected_ab`,
  and `selected_ab` as replacements for selected-QKV/Z at c=2 512/128: all four
  stayed generated-token red with prefixes `[82,104]`
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-projection-red-probe/summary.json`).
  A follow-up temporary dual-GEMV/planar-copy probe for the `batch_gemv` QKV/Z
  path improved only row 1 (`[82,104] -> [82,137]`) while the minimum prefix stayed
  82, and the selected-rotary-input + dual-GEMV variant remained `[82,104]`;
  layer-1 hidden probes still showed non-bit-exact QKV/Z projection drift under
  tolerance and hidden parity red, so no runtime code was retained
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-dual-gemv-projection-red-probe/summary.json`).
  Adding selected-c1 A/B replay to that same temporary dual-GEMV QKV/Z path made
  row 0 worse (`[82,104] -> [0,137]`) and left the layer-1 hidden oracle red, so
  A/B exactness alone does not isolate the QKV/Z batch-GEMV path as green
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-dual-gemv-selected-ab-red-probe/summary.json`).
  A clean-tree hidden-control rerun reproduced layer-limit 40 hidden/token parity
  green for both default selected-QKV/Z and full selected-c1 projection replay;
  layer-limit 1 remains an intermediate hidden-only diagnostic mismatch, not a
  final l40 failure. The same pass exposed a primary c=2 512/128 reproducibility
  flake (`[82,137]` then immediate repeat `[137,137]`), so the selected fallback
  is still correctness-only rather than retained-ready
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-selected-projection-hidden-repro/summary.json`).
  A follow-up confidence sweep returned five consecutive c=2 `[137,137]` repeats
  and a fresh c=2/c=4/c=8 equality matrix with all rows prefix 137, restoring
  confidence in the fallback while keeping native projection/dispatch blocked
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-selected-fallback-repeat-confidence/summary.json`).
  A temporary direct-planar dual-GEMV QKV/Z probe wrote QKV and Z outputs in one
  launch without the row-interleaved temporary-copy step, but `batch_gemv` still
  failed c=2 512/128 at `[82,104]`; the selected-QKV/Z fallback stayed `[137,137]`
  after reverting the code, so no runtime code was retained
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-planar-dual-gemv-projection-red-probe/summary.json`).
  Forcing exact selected-c1 linear state while keeping QKV/Z on `batch_gemv`
  also stayed red at `[82,104]`, while the selected-QKV/Z fallback control stayed
  `[137,137]`, so the blocker is not repaired by token-1 state replay in isolation
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-batch-gemv-selected-state-red-probe/summary.json`).
  The focused QKV/Z reduction-order fix now runs the explicit `batch_gemv` QKV/Z
  diagnostic with 128-thread single GEMVs instead of 64-thread launches. After one
  reproduced c=2 matrix flake, immediate c=2 repeats and a fresh c=2/c=4/c=8
  matrix were generated-token green with all rows prefix 137
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-batch-gemv-qkvz-threads128-matrix/summary.json`).
  This removes the explicit batch-GEMV QKV/Z correctness blocker but remains
  non-retained. A follow-up default-promotion pass now resolves c<=8 auto
  projection to the no-selected `batch` path; the no-flag c=2 primary gate and a
  fresh c=2/c=4/c=8 matrix are green with `linear_attention_projection_path=native_batch`
  and `native_caware_decode=true`
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-auto-batch-projection-default-matrix/summary.json`).
  Retained/scaling claims are still blocked by projection-dispatch evidence,
  graph-replay profiler data, c1/serial baselines, and benchmark gates. A focused
  L8 hidden-state isolation rerun with the no-selected `batch` projection default
  still fails at decode step 11 when full attention is fully native, but forcing
  only full-attention decode to the per-row diagnostic path makes the same
  projection/state/output/MoE controls hidden-green; this moves the remaining
  hidden-only blocker from linear QKV/Z projection to native full-attention decode
  evidence
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-full-attention-hidden-isolation/summary.json`).
  A narrower follow-up proved that forcing only full-attention context/gate to the
  per-row diagnostic path is sufficient to make the same L8 probe hidden/token
  green, narrowing the blocker to native full-attention context/gate decode
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-full-attention-context-isolation/summary.json`).
  A KV-append-only diagnostic reproduced the native hidden mismatch exactly while
  the context/gate-only diagnostic stayed green, so the next native fix target is
  the batch context/gate path rather than standalone KV append
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-full-attention-context-vs-kvappend-isolation/summary.json`).
  The raw batched paged-context primitive is green at the real multi-block shape
  (`rows=2`, context 513/524, 64 Q heads, 8 KV heads, head_dim 128), and a
  QKV-prep-only per-row diagnostic still reproduces the hidden mismatch; the
  remaining issue is therefore native context/gate integration with model
  scratch/cache/gate tensors, not standalone primitive context arithmetic or QKV
  prep
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-full-attention-context-primitive-realshape/summary.json`).
  A new gate-only diagnostic (`--batch-decode-attn-gate-path per_row`) keeps
  native batch context while applying the sigmoid gate per row; it still fails
  hidden parity, so the blocker is before/at native batch context output
  integration rather than the contiguous batch gate multiply alone
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-full-attention-gate-split-isolation/summary.json`).
  The complementary context-only split (`--batch-decode-attn-context-path
  per_row_context_only`) replays only token-1 context rows and then uses the
  normal batch gate; it is hidden/token green, confirming the batch gate is fine
  when fed per-row context outputs and pinning the blocker to native batch
  context output integration before gate
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-full-attention-context-only-split/summary.json`).
  Follow-up batch-context staging diagnostics (`batch_temp_output` and
  `batch_compact_cache`) stayed hidden-red, ruling out query_raw destination
  aliasing and simple all-slots cache compaction as sufficient fixes; in the
  same run the no-flag native generated-token equality gate passed for c=2,
  c=4, and c=8 at 512/128 with min equal-prefix `137` on every row
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-c2-c4-c8-equality-and-context-staging/summary.json`).
  The first retained-scaling prerequisites after that gate are now recorded for
  c=8: c=1 native 512/128 baseline `133.91` tok/s and c=8 serial-bridge
  baseline `104.43` aggregate tok/s / `13.05` per-request tok/s, both as
  non-retained baseline artifacts validated by the retained helper
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-c8-scaling-baselines/summary.json`).
  The c=8 profiler prerequisite is also captured now: a `rocprofv3
  --kernel-trace` run recorded expected native batch context, KV append,
  post-MoE batch combine, and batched sampler kernels; the retained validation
  artifact sees `profiler.status=captured`, complete scaling ratios, primitive
  c=8 correctness, and generated-token equality green, but remains blocked by
  projection-dispatch/other retained gates, so no throughput claim is retained
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-c8-profile-evidence/summary.json`).
  A focused projection-dispatch evidence artifact now measures the native c=8
  `batch` projection candidate against the explicit selected-c1/row-GEMV
  projection baseline (`1.1446x` aggregate and per-request) and the retained
  runner selects `projection_dispatch.selected_candidate=batch` with path
  `benchmark_accepted_caware_projection` when the artifact is supplied; remaining
  retained blockers are batch-level eligibility/provenance/observability rather
  than missing projection candidate evidence
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-c8-projection-dispatch-evidence/summary.json`).
  An all-available-evidence c=8 validation now uses repo-relative output and
  compiler-version provenance, so those earlier path blockers are gone
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-c8-all-evidence-blockers/summary.json`).
  A follow-up exact-command `rocprofv3` recapture removes the stale profiler
  command `--json` / `--compiler-version-file` mismatches as well; graph bucket
  stats now distinguish cache lookup hits from actual replay-kernel hits, so the
  exact-profile artifact records `hits=1` but `replay_kernel_hits=0`, matching
  the raw trace with no graph/replay kernel. The retained gate now also records
  observed decode context buckets (`512` and `768` for the 512/128 run), removing
  the per-request bucket context-axis blockers. Stable fixed-slot block identity
  audit now passes as well. It stays blocked only by batch-level
  `throughput_claim_eligible=false`
  (`benchmarks/results/2026-06-02-hipengine-qwen35-native-c8-exact-profile/summary.json`).
  A later c=8 auto refresh keeps generated-token equality green only by using
  the per-row full-attention fallback with selected-c1 MoE; forcing the native
  batch full-attention branch with selected-c1 MoE remains red, and an input /
  QKV / context-gate / KV-append / output / post-attention / persistent-c1
  branch-isolation sweep did not repair it
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-full-attn-branch-isolation/summary.json`).
  A native row-chunk diagnostic then showed that c=8 rows 0-3 are green while
  rows 4-7 are red, and a derived c=4 fixture containing only original rows
  4-7 reproduces the same native-full red prefixes while the per-row-full
  control is green; the remaining blocker is therefore prompt/window-sensitive
  native full-attention behavior, not only c=8 batch size or physical slot id
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-full-attn-row-window-isolation/summary.json`).
  The current c=8 auto path now uses native full-attention row chunks of 2
  instead of the older per-row full-attention fallback; c=2/c=4/c=8 equality
  remains green, c2 pair controls are mostly green, c3 windows over original
  rows 4..7 reproduce the native-full red pattern, and row-chunk-2 controls are
  green. This is still non-retained because row chunking is a diagnostic
  fallback (`native_caware_decode=false`)
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-rowchunk2-auto-pair-sweep/summary.json`).
  The current c=8 auto profiler now confirms that rowchunk2 launches native
  batch full-attention context kernels and zero per-row full-attention context
  kernels, with generated-token equality green in that profiled run
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-rowchunk2-profiler/summary.json`).
  A follow-up confidence/stage sweep found intermittent c=8 equality flakes in
  both the rowchunk2 auto path and explicit per-row-full fallback, while c3
  rows4/5/6 stayed red under every partial native-branch diagnostic and green
  only under the broad per-row-full branch; c8 remains correctness-only and not
  retained-ready
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-stage-c8-confidence/summary.json`).
  A fresh sampler repeat matrix did not reproduce the flake: explicit c8
  rowchunk2+batched-LM-head, rowchunk2+serial-LM-head, and
  per-row-full+serial-LM-head each passed 3/3 repeats at `[137 x8]`. This
  narrows the flake away from a deterministic sampler-mode mismatch, but the
  prior flakes and diagnostic fallback labels still block retained claims
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-sampler-repeat-matrix/summary.json`).
  A row-chunk size sweep then showed the prompt/window-sensitive safe grouping:
  chunk sizes 1 and 2 are green for original c8 plus derived rows4..7 c4 and
  rows4..6 c3, while chunk size 3 or larger reproduces the native-full red
  pattern. The current largest green native-context grouping for those prompts
  is therefore 2 rows; full native c8/c4 rows4..7 remains blocked
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-full-attn-rowchunk-size-sweep/summary.json`).
  A rows4/5/6 permutation sweep showed the native c3 failure prefixes follow
  logical prompt identity rather than row position (`row4=11`, `row5=60`,
  `row6=117` under all tested orderings), while rowchunk2 controls stay green
  for every ordering. This makes a simple output-row-position alias unlikely
  and keeps the target on prompt/window-sensitive native full-attention behavior
  when three or more logical rows share one full-attention group
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-full-attn-row-permutation/summary.json`).
  The retained bench auto path now caps covered c=3/c=4/c=8 full-attention
  groups at rowchunk2 by default. Candidate validation passed derived rows4..6
  c3, derived rows4..7 c4, original c4, and original c8 at `[137]` equal-prefix
  under auto with no per-row full-attention fallback. This is a correctness
  default only: rowchunk2 still marks `native_caware_decode=false`, and full
  native grouping >=3 remains the retained blocker
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-auto-rowchunk2-c3-c4/summary.json`).
  Hidden-bisect now supports rowchunk trace diagnostics. For derived rows4..6
  c3, L4 raw `attn_context` parity is green for full native, rowchunk2, and
  per-row controls; in the L40 trace window around generated index 11, the first
  native full-attention failure is already `attn_input_pre_qkv` before
  `attn_context`. The context drift there is therefore inherited, not proof that
  the context kernel is the first faulty stage
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-full-attn-context-trace-rowchunk/summary.json`).
  A fresh current-commit L40 trace over decode steps 0..11 reproduces derived
  rows4/5/6 c3 full-native generated-token red prefixes `[11,60,117]`, while
  rowchunk2 remains green at `[137,137,137]`. The earliest full-attention stage
  drift is already decode step 0 / layer 7 / `attn_input_pre_qkv` / row 0, before
  context/output/O-projection; the earliest traced drift overall is a hidden-only
  linear-attention `attn_input`/conv mismatch at decode step 0 / layer 4, so the
  generated-token blocker stays pinned to prompt/window-sensitive full-attention
  grouping >=3 rather than late output copy
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-full-attn-trace0-12/summary.json`).
  Forcing the full-attention input RMSNorm/pre-QKV setup to the per-row
  diagnostic path does not move those c3 prefixes (`[11,60,117]` unchanged),
  and a short L40 trace keeps the same first full-attention mismatch at decode
  step 0 / layer 7 / `attn_input_pre_qkv` / row 0. The rowchunk2 control stays
  generated-token green and clears the post-RMSNorm/QKV-prep stages in the same
  short trace, so input RMSNorm alone is eliminated as the grouping>=3 fix; the
  next target remains the upstream hidden/row-group interaction before or at the
  full-attention pre-QKV boundary
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-perrow-input-red/summary.json`).
  Forcing full-attention QKV prep to per-row scratch also does not move the c3
  prefixes (`[11,60,117]` unchanged) and keeps the same first L40 short-trace
  mismatch at decode step 0 / layer 7 / `attn_input_pre_qkv` / row 0. Forcing
  independent per-row full-attention layer scratch is not a fix either: prefixes
  worsen to `[11,11,40]`, and the hidden trace produces non-finite values before
  JSON serialization. QKV/scratch replay alone is therefore eliminated; the next
  concrete diagnostic should compare native-full vs rowchunk2 inputs at the
  layer-7 pre-QKV boundary or inspect the upstream hidden/row-group interaction
  that rowchunk2 changes
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-qkv-scratch-red/summary.json`).
  A clean post-commit c2/c4/c8 matrix under current auto defaults passed
  generated-token equality at equal-prefix 137 for every row: c2 remains full
  native, while c4/c8 use native rowchunk2. This confirms the covered c>1
  correctness path at the current commit, but c4/c8 remain non-retained because
  rowchunk2 reports `native_caware_decode=false` and retained projection/throughput
  evidence was not yet attached at that point
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-clean-c2-c4-c8-auto-matrix/summary.json`).
  A fresh post-change c8 profiler smoke confirms the current auto rowchunk2 path
  still launches native batch full-attention context kernels (16,320 calls,
  1.865 s aggregate) and zero per-row full-attention context kernels; batched
  sampler argmax kernels are present and generated-token equality remains green.
  This is runtime evidence only, not a retained throughput claim
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-auto-rowchunk2-final-profiler/summary.json`).
  A focused c2 repeat sweep after a red verify flake found five green repeats
  and one red repeat at `[23,137]`; the red row0 first diverges at generated
  index 23 with batch token `1156` where independent c1 expected `1879`. This
  makes c2 full-native equality intermittently flaky at the current commit and
  blocks retained readiness despite frequent immediate green repeats
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-repeat-flake/summary.json`).
  Retained-bench generated-token equality artifacts now include compact
  `mismatch_summaries` fields with first mismatch token/windows and row-alias
  candidates. A current-commit c2 512/128 repeat capture after adding that
  diagnostic ran 16/16 green at `[137,137]` with empty `mismatch_summaries`; this
  restores current confidence but does not prove the intermittent flake is fixed
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-mismatch-summary-green16/summary.json`). A later exact active-command
  repeat after the c3..c8/c9 diagnostic work ran five consecutive repeats plus
  the final active verify green at `[137,137]`, again increasing confidence
  without converting it into a retained throughput claim
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-repeat-current217/summary.json`). A later active verify during the rowchunk2 MoE-boundary sweep
  flaked red again at `[103,137]` before the immediate final verify returned
  `[137,137]`, so the intermittent c2 flake risk remains open
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c5c9-rowchunk2-moe-boundary/summary.json`). A follow-up c2 stress check ran five default selected-c1-MoE
  repeats and five explicit grouped-compact-MoE repeats; all ten were green at
  `[137,137]`, so the flake is not trivially reproduced by toggling c2 MoE mode
  in isolated repeat runs
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-default-grouped-repeat220/summary.json`). A sampler split likewise ran five default batched-LM-head
  sampler repeats and five explicit serial-LM-head repeats, all green at
  `[137,137]`; sampler mode alone is not an immediate trigger
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-sampler-repeat221/summary.json`). A post-c9 stress sequence then reproduced the exact
  c2 `[103,137]` flake shape: three cycles of red c9 grouped-compact rowchunk2
  followed by the active c2 default command produced c2 prefixes
  `[137,137]`, `[103,137]`, `[137,137]`. The flake is therefore reproducible
  under post-c9/high-row grouped stress, but still nondeterministic and not yet
  isolated to MoE vs rowchunk/high-row stress
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-after-c9-stress222/summary.json`). Replacing only that preceding c9 MoE path with selected-c1 while
  keeping the same c9 fixture and rowchunk2 full-attention ran five green c9
  controls (`[137]*9`) followed by five green active c2 runs (`[137,137]` each),
  narrowing the suspect toward c9 grouped-compact/red-path stress rather than
  high-row rowchunk2 alone, though not proving causality
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-after-c9-selectedc1-control223/summary.json`). A covered c8 grouped-compact rowchunk2 control likewise ran five green
  c8 controls (`[137]*8`) followed by five green active c2 runs (`[137,137]`
  each), so grouped-compact MoE at the covered c8 boundary is not enough to
  reproduce the c2 flake; the suspect narrows further toward the c9
  grouped-compact boundary/red path
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-after-c8-grouped-control224/summary.json`). Repeating the red c9 grouped stress but making the following c2 run use
  explicit `serial_lm_head` sampling kept all five c2 controls green
  (`[137,137]`), narrowing the observed post-c9 trigger toward interaction with
  the default batched LM-head sampler path rather than upstream c2 hidden/decode
  state alone
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-serial-after-c9-grouped-control225/summary.json`). The correctness-first c2 default now leaves c2 on `serial_lm_head`
  instead of auto-enabling the retained batched-sampler artifact until that
  explicit batched path is fixed; three post-c9 grouped-stress cycles kept the
  active no-flag c2 default green (`[137,137]`) with sampler metadata confirming
  `serial_lm_head`, while c4/c8 auto sampler evidence remains enabled
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-default-serial-after-c9-fix226/summary.json`). A post-demotion no-flag c2/c4/c8 matrix confirms the default gate is still
  generated-token green throughout: c2 uses `serial_lm_head` and c4/c8 use
  retained `batched_lm_head` evidence plus rowchunk2 full-attention diagnostics
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-default-sampler-demotion-matrix227/summary.json`). A new diagnostic `--batch-sample-argmax-mode=serial_per_row` keeps the
  batched LM-head projection/norm but resolves each row's argmax with the serial
  per-row kernel. Under the same red c9 grouped stress, three explicit c2
  `batched_lm_head` + `serial_per_row` argmax controls stayed green (`[137,137]`)
  and recorded a sampler blocker, narrowing the explicit c2 batched-sampler
  flake toward `batch_argmax_f32` rather than batched projection/norm logits
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-serial-argmax-after-c9-control228/summary.json`). A follow-up `--batch-sample-argmax-audit` now runs `batch_argmax_f32`,
  then serial per-row argmax over the same batched logits, and accumulates
  same-logits parity stats while recording a sampler blocker. Across five more
  red c9 grouped-stress cycles, explicit c2 `batched_lm_head` with audited
  `batch_argmax_f32` stayed green (`[137,137]`) and checked 680 decode steps /
  1360 row argmaxes with zero batch-vs-serial token mismatches. This does not
  reproduce the historical `[103,137]` flake and rules out a deterministic
  batch-argmax reduction disagreement on those audited logits; the explicit c2
  batched sampler remains blocked until the historical flake is reproduced under
  audit or otherwise fixed
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-batch-argmax-audit-after-c9-229/summary.json`). A current no-audit rerun of the actual explicit c2 batched sampler also did
  not reproduce the historical flake: five red c9 grouped-stress cycles were
  each followed by c2 `batched_lm_head`/batch-argmax runs that stayed green
  (`[137,137]`) with `native_row_aware_lm_head=true` and empty sampler blockers.
  This strengthens the timing/transient diagnosis but does not by itself restore
  c2 `batched_lm_head` as the default because the iter222 `[103,137]` flake
  artifact remains contrary evidence
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-batched-noaudit-after-c9-repeat232/summary.json`). A post-diagnostic no-flag c2/c4/c8 equality matrix confirms the active
  default gate is still generated-token green: c2 `[137,137]` uses
  `serial_lm_head` with full-native `native_caware_decode=true`, while c4/c8 use
  retained `batched_lm_head` sampler evidence and rowchunk2 full-attention
  diagnostics with prefixes `[137]*4` and `[137]*8`. The new argmax audit fields
  are inactive by default; c4/c8 remain correctness-only because rowchunk2 still
  reports `native_caware_decode=false`
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-post-argmax-audit-default-matrix230/summary.json`). Forcing grouped-compact MoE after the sampler diagnostics keeps the same
  c2/c4/c8 equality gate green: c2 `[137,137]`, c4 `[137]*4`, and c8 `[137]*8`,
  with all 40 layers reporting `moe_decode_path=grouped_compact` and zero
  selected-c1 fallback layers. This refreshes the more-native MoE correctness
  checkpoint; the remaining c4/c8 blocker stays rowchunked/full-native full
  attention rather than MoE or sampler instrumentation
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-grouped-moe-post-audit-matrix231/summary.json`). A later required active c2 verify during the c4 context-producer
  refresh reproduced the intermittent serial-sampler/default c2 flake once at
  `[103,137]` with `selected_c1_batch` MoE and `native_caware_decode=true`; an
  immediate repeat returned `[137,137]`. Treat c2 equality stability as still open
  even when the latest repeat is green
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c4-context-producer-refresh236/summary.json`). A c2 split after that flake ran the exact default selected-c1/native-context
  command green 3/3 and grouped-compact/native-context green 3/3, while the
  `per_row_context_only` diagnostic deterministically reproduced the same row0
  `[103,137]` / token-6007-vs-1483 signature 3/3. The final active verify was
  green `[137,137]`; the flake now points toward a full-attention context-path
  signature, not sampler or MoE mode alone, but why native-context metadata can
  intermittently land there remains unresolved
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-stability-split237/summary.json`). The context-only diagnostic now uses paged spans for each replayed
  row instead of the dense single-row context kernel after batch KV append;
  this eliminates that focused c2 blocker (`[103,137]` before, `[137,137]`
  3/3 after, active verify `137`)
  (`benchmarks/results/2026-06-04-hipengine-qwen35-native-c2-contextonly-paged-fix238/summary.json`). The combined per-row context+gate diagnostic now also routes replayed
  contexts into `query_raw` and applies the batch gate kernel; it moved from
  iter238 red `[103,137]` to `[137,137]` 3/3 with active verify `137`. Both
  diagnostic paths remain non-retained by metadata, but their c2 parity blockers
  are eliminated
  (`benchmarks/results/2026-06-04-hipengine-qwen35-native-c2-contextgate-batch-gate-fix239/summary.json`). For rows>2 the per-row context diagnostic must keep row-local
  context+gate rather than c2's batch-gate replay: after this row-aware split,
  c2 per-row context stays `[137,137]` and c4 grouped/no-rowchunk per-row
  context returns to `[137]*4`, while native c4 remains red at
  `[137,137,137,118]`
  (`benchmarks/results/2026-06-04-hipengine-qwen35-native-c4-contextgate-row-aware-fix240/summary.json`).
  The context-only diagnostic now mirrors the row-count split: c2 keeps paged
  context plus batch gate, but rows>2 use row-local dense context plus row-local
  gate. That moves c4 grouped/no-rowchunk context-only from `[137,137,137,118]`
  to `[137]*4`; native c4 still remains red, narrowing the retained blocker to
  native batch context/gate coupling rather than row-local context math
  (`benchmarks/results/2026-06-04-hipengine-qwen35-native-c4-contextonly-row-gate-fix241/summary.json`).
  A follow-up split added explicit dense-vs-paged context-only diagnostics: c4
  grouped/no-rowchunk row-local dense context is green at
  `[137]*4`, while row-local paged context reproduces the native red
  `[137,137,137,118]`; c2 keeps the current split because dense-only c2 is red
  `[103,137]` while the existing c2 context-only path remains `[137,137]`
  (`benchmarks/results/2026-06-04-hipengine-qwen35-native-c4-dense-vs-paged-context-fix242/summary.json`).
  Real-shape random-input primitive probes for c4/c8 then showed the paged batch
  context kernel exactly matches independent paged c1 (`max_abs=0.0`)
  and differs from dense c1 by only ~2e-08 at Qwen-like shape/context lengths,
  so the E2E red path is not a broad primitive shape bug; next compare
  actual model-layer context tensors or captured failing-window inputs
  (`benchmarks/results/2026-06-04-hipengine-qwen35-primitive-realshape-context-parity-fix243/summary.json`).
  Hidden-bisect now exposes the same explicit dense-vs-paged context-only
  controls. On actual c4 no-rowchunk model tensors, row-local dense context is
  green at step 0 and in the long failing window, but row-local paged context is
  hidden-red at the first generated token: both paged kernels match their own
  NumPy softmax oracles, while batch-vs-c1 NumPy context first differs at layer
  7 from a single current-token KV-prefix hash mismatch at position 512. The
  long paged trace later becomes token-red (row0 index 82 / row1 index 104), so
  the blocker is model-trajectory/current-token KV sensitivity on the paged
  context path, not broad primitive arithmetic
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hidden-c4-model-context-dense-vs-paged-244/summary.json`).
  A selected-layer dense-context override then proved this is a producer chain:
  all-paged fails first at layer 7; making only layer 3 dense moves the first
  context/KV-prefix failure to layer 11; making only layer 7 dense does not move
  the layer-7 failure; and making layers 3+7 dense moves the first failure to
  layer 15. Thus the first retained no-rowchunk producer is layer-3 paged
  context output feeding downstream current-token KV, and each subsequent paged
  full-attention layer can repeat the same pattern
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hidden-c4-paged-layer-override-245/summary.json`).
  A c4 step0 prefix sweep confirmed that the first failure monotonically advances
  to the next still-paged full-attention layer (none→7, 3→11, 3+7→15,
  3+7+11→19, 3+7+11+15→23), while making all full-attention producers dense
  (`3,7,11,15,19,23,27,31,35,39`) closes the hidden/context/KV-prefix blocker
  at step0. This keeps the c4 no-rowchunk target on the paged-context producer
  mode across full-attention layers, not sampler or later token feedback
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hidden-c4-paged-prefix-dense-sweep-246/summary.json`).
  The dense-context producer can still use the normal batch gate: c4 step0 is
  green with row-local dense context plus batch gate, while row-local paged
  context and native batch paged context both fail at the same layer-7
  current-token KV/context point. This eliminates the gate kernel/downstream
  suffix as the step0 cause when dense contexts feed it, leaving the paged
  context producer/addressing mode as the focused blocker
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hidden-c4-dense-context-batch-gate-247/summary.json`).
  The same split holds across the historical long window: c4 no-rowchunk
  dense context plus batch gate stays `eq_ok` for warmup8+decode112, matching
  the dense row-local gate control, while row-local paged context and native
  batch paged context both become token-red at row0 index82 and show the same
  traced decode-step117 layer-3 KV/context divergence. This keeps the long-run
  blocker on paged context production rather than batch-gate feedback
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hidden-c4-long-dense-context-batch-gate-248/summary.json`).
  Retained-bench now exposes the same dense-context+batch-gate diagnostic. On
  c4 no-rowchunk 512/128 with grouped-compact MoE and native full-attention O
  output, dense context plus batch gate is generated-token green at
  `[137]*4`; the native paged-context control remains red at
  `[137,137,137,118]`. The output path is still coupled: forcing batch-GEMV O
  with the same dense-context diagnostic is red at `[137,104,137,137]`, so the
  green correctness-only no-rowchunk fallback is dense context plus native O,
  not dense context alone
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c4-dense-context-batch-gate-equality-249/summary.json`).
  The c4 no-rowchunk fallback is not yet a c8 fallback: c8 no-rowchunk dense
  context plus batch gate plus native O is red at
  `[137,137,137,137,45,11,83,137]`, and dense context plus row-local gate has
  the same prefixes. Native paged no-rowchunk/native-O remains red at
  `[137,137,137,118,45,31,68,137]`; rowchunk4 and selected-c1 MoE contrasts
  remain red. The current rowchunk2 control stays generated-token green at
  `[137]*8`, so c8 still needs the rowchunk2 cap while the dense-context/native-O
  no-rowchunk fallback is c4-only
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c8-dense-context-batch-gate-frontier-250/summary.json`).
  Repackaging original c8 rows4..7 as a compact c4 fixture confirms the residual
  c8 rows are prompt-identity/full-attention row-grouping issues rather than an
  eight-row scheduler/sampler effect: the compact tail quartet is red under
  no-rowchunk dense-context+batch-gate/native-O (`[45,11,83,137]`) and native
  paged/native-O (`[45,31,68,137]`), but green under rowchunk2 and rowchunk1
  native-paged controls (`[137]*4`)
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c4-tail-rows-rowchunk2-isolation-251/summary.json`).
  A rowchunk3 frontier pass on the same compact tail quartet remains red:
  native-paged rowchunk3 is `[45,11,137,137]` and dense-context+batch-gate
  rowchunk3 is `[45,11,83,137]`, while the rowchunk2 control remains `[137]*4`.
  Thus the tail quartet's safe correctness cap is at most two rows per
  full-attention group; rowchunk3 is not enough
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c4-tail-rowchunk3-frontier-252/summary.json`).
  A c8 rowchunk2 native-O frontier adds
  `--batch-decode-full-attn-output-path native_row_chunk` to bypass the automatic
  rowchunk batch-GEMV O repair: the batch/default control stays green at
  `[137]*8`, but forced native rowchunk O is red at
  `[137,137,137,118,45,137,68,137]`. Rowchunk2 grouping alone is therefore not
  sufficient for retained c8; the rowchunk batch-GEMV O repair remains a separate
  correctness requirement until native rowchunk O is fixed
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c8-rowchunk2-native-o-frontier-253/summary.json`).
  Compacting each original c8 rowchunk2 pair into standalone c2 fixtures shows
  the native-O issue is prompt-pair sensitive even without rowchunk slicing: rows
  0..1 are green (`[137,137]`), but rows 2..3, 4..5, and 6..7 are red at
  `[137,118]`, `[45,137]`, and `[68,137]` with `native_caware_decode=true` and
  no blockers. Forcing batch-GEMV full-attention O makes every compact c2 pair
  green at `[137,137]`, so row-aware batch-GEMV O is the focused correctness
  repair for these prompt pairs, not just a rowchunk copy workaround
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c8-rowchunk2-pair-native-o-isolation-254/summary.json`).
  The rows=2 default now auto-selects row-aware batch-GEMV full-attention O
  (`batch_gemv_auto`) unless a native/per-row/GEMV diagnostic is explicitly
  forced. Under the default `--batch-decode-full-attn-output-path batch`, every
  compact c2 prompt pair is green at `[137,137]`; the explicit native diagnostic
  preserves the red native-O signal for rows 2..7 (`[137,118]`, `[45,0]`,
  `[68,137]`). Active c2 512/128 still reports `[137,137]` with first full layer
  output path `batch_gemv_auto`
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c2-auto-gemv-o-default-255/summary.json`).
  A refreshed default c=2/c=4/c=8 equality matrix after the c2 GEMV-O default is
  green vs independent c1 for every row: c2 `[137,137]` with
  `batch_gemv_auto` and `native_caware_decode=true`, c4 `[137]*4`, and c8
  `[137]*8`. c4/c8 still use rowchunk2 full attention plus rowchunk batch-GEMV O
  and remain correctness-only/blocked for retained performance eligibility
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-default-equality-after-c2-gemv-256/summary.json`).
  The c4 auto path now avoids rowchunk/O blockers by choosing no rowchunk,
  row-local dense context with the batch gate, and grouped-compact MoE. The
  refreshed default matrix remains green: c2 `[137,137]`, c4 `[137]*4` with only
  the dense-context batch-gate blocker, and c8 `[137]*8` still on rowchunk2 plus
  rowchunk batch-GEMV O. This is still correctness-only, with no retained/scaling
  claim
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-c4-no-rowchunk-default-257/summary.json`).
  Rowchunked two-row groups now inherit the accepted rows=2 `batch_gemv_auto` O
  default instead of reporting a separate O projection blocker. The default
  matrix stays green (`c2 [137,137]`, `c4 [137]*4`, `c8 [137]*8`); c8 still uses
  rowchunk2 full attention, but its first full-attention layer output path is
  `native_batch_row_chunks_with_batch_gemv_auto` and only the rowchunk blocker
  remains
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-rowchunk-auto-gemv-output-258/summary.json`).
  The auto MoE path now uses grouped-compact for the retained-claim correctness
  frontier instead of selected-c1 speed diagnostics. The default matrix remains
  green (`c2 [137,137]`, `c4 [137]*4`, `c8 [137]*8`), with
  `moe_grouped_compact_layers=40` and `moe_selected_c1_fallback_layers=0` for
  every row count. c2 now has no decode blockers; c4 still has the dense-context
  batch-gate blocker and c8 still has the rowchunk full-attention blocker. A
  no-rowchunk c8 dense-context trial stayed red (`min=11`), so rowchunk removal
  remains unresolved
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-grouped-moe-default-259/summary.json`).
  A focused `rocprofv3 --kernel-trace` capture on the active c2 512/128 default
  now provides required runtime evidence, still without a throughput/scaling
  claim. The profiled run stayed generated-token green (`[137,137]`) with
  grouped-compact MoE, `batch_gemv_auto` full-attention O, `native_caware_decode=true`,
  and no decode blockers. The trace ran on `HIP_VISIBLE_DEVICES=1` / RX 7900 XTX
  and includes positive-duration native c2 kernels for batched paged full-attn
  context, batched KV append, grouped-compact MoE group kernels, linear-attention
  segment decode, and GEMV/Marlin projections. Raw CSV remains under `/tmp`; the
  compact evidence artifact is committed
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c2-profiler-evidence-260/summary.json`).
  The primitive GPU correctness gate now also has repo-relative c=2/c=4/c=8
  artifacts on `HIP_VISIBLE_DEVICES=1` / RX 7900 XTX. All three pass append K/V,
  batch-vs-c1 attention, batch-vs-numpy tolerance, and append+attention AA
  checks with `attn_batch_vs_c1_max_abs=0.0` and `attn_batch_aa_max_abs=0.0`.
  This is correctness/runtime evidence only and still makes no throughput claim
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-primitive-gpu-correctness-261/summary.json`).
  Fresh c2 scaling references are now captured under `benchmarks/results/` for
  the same GPU1/XTX 512/128 shape: c1 native baseline `134.05` tok/s and c2
  serial bridge `110.35` aggregate tok/s / `55.18` per-request tok/s. A retained
  c2 attach audit loads both references plus the repo-relative primitive c2 gate
  with `scaling.complete=true` and equality still `[137,137]`; ratios are
  `aggregate_vs_c1=0.8617` and `aggregate_vs_serial_bridge=1.0467`, so this is
  baseline evidence only and not a retained throughput claim
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c2-baseline-references-262/summary.json`).
  A current GPU1/XTX c=2/c=4/c=8 matrix with explicit row-aware
  `batched_lm_head` sampler evidence now passes generated-token equality vs
  independent c1 for every row (`[137]` prefixes throughout), with c2 fully
  native/c-aware, c4 still dense-context batch-gated, and c8 still rowchunked;
  this is correctness-only and not retained throughput/scaling evidence
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-batched-sampler-equality-263/summary.json`).
  The current c2 path also reloads the combined projection-dispatch catalog with
  all current c2 references attached: it selects
  `benchmark_accepted_caware_projection` /
  `gemv_awq_selected_dual_pack8_strided_c2`, has empty projection/batch/decode
  blockers, keeps equality `[137,137]`, and loads primitive/profiler/scaling
  evidence; it remains non-retained because aggregate-vs-c1 scaling is still
  below 1.0
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c2-projection-dispatch-264/summary.json`).
  A follow-up current c2 `rocprofv3 --kernel-trace` retained audit closes the
  profiler-promotion evidence fields too: profiler source/command provenance,
  expected/trace/duration kernel names, graph histogram, selected projection
  kernel evidence, and native `batch_argmax` sampler durations all validate;
  the retained decision is now blocked only by
  `scaling.ratios.aggregate_vs_c1 <= 1.0`, so no throughput claim is made
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c2-profiler-promotion-265/summary.json`).
  A current c=2/c=4/c=8 frontier rerun attaches the combined c-aware projection
  catalog plus primitive and row-aware sampler evidence for every control row:
  equality remains green (`c2 [137,137]`, `c4 [137]*4`, `c8 [137]*8`) and
  projection/sampler blockers are empty. The remaining c4/c8 frontier is still
  full-attention grouping: c4 is green only through the dense-context batch-gate
  fallback, c8 is green through rowchunk2, and rowchunk3 probes stay red
  (`c4 [137,104,137,137]`, `c8 [82,137,137,137,45,11,137,137]`), with no
  throughput claim
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c248-full-attn-frontier-266/summary.json`).
  A selected-layer dense-context+batch-gate override now narrows the c4
  no-rowchunk fallback: native paged context is red at `[137,137,137,118]`,
  layers `3` and `3,7` remain red, but layers `3,7,11` are generated-token
  green in three runs (`[137]*4` each) with accepted c-aware projection and
  row-aware sampler evidence. Adding layer `15` without `19` is reproducibly red
  (`0/2`, `[137,137,137,118]`), while `3,7,11,15,19+` stays green. This moves
  the c4 blocker from an all-full-attention-layer dense fallback to a small
  producer-layer frontier, still correctness-only and not retained
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c4-dense-context-layer-sweep-267/summary.json`).
  A pair matrix around that frontier shows the transition is recovery-window
  sensitive rather than simply monotonic: base `3,7,11` stays green; `+15` is red
  (`[137,137,137,118]`); `+19`, `+15+19`, and `+15+23` are green (`+15+23` in
  `2/2` runs); but `+15+27`, `+15+31`, `+15+35`, and `+15+39` remain red. Thus
  the layer-15 dense-context transition needs an early downstream producer
  (`19` or `23`) to recover row3 token-118 drift; later full-attention layers are
  too late
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c4-dense-layer-pair-matrix-268/summary.json`).
  Hidden tracing now exposes the same selected-layer batch-gate override. A red
  `3,7,11,15` trace reproduces row3 token index 118, while green
  `3,7,11,15,19` is token-clean. Both traces still show hidden drift from
  decode step 0 and the same decode-step117 context/KV signatures (first
  full-context KV-prefix failure layer 7 row0 at positions 512..629,
  `attn_context` bit-drift green, full-attention stage rollup first failing at
  layer 3 `mlp_input`). Layer 19 therefore recovers the token trajectory without
  eliminating the earlier hidden/context-KV drift, keeping the next target on
  the downstream recovery boundary around layers 15→19/23 rather than the
  earliest hidden drift alone
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hidden-c4-dense-recovery-trace-269/summary.json`).
  The retained-bench c4 auto default now uses this selected-layer override
  instead of the prior all-full-attention-layer dense-context fallback: a no-flag
  c4 512/128 run with accepted c-aware projection, primitive correctness, and
  row-aware `batched_lm_head` evidence is green (`[137]*4`), records
  `batch_decode_attention_dense_context_batch_gate_layers="3,7,11"`, and leaves
  only the selected-layer dense-context blocker
  (`benchmarks/results/2026-06-04-hipengine-qwen35-retained-c4-selected-layer-default-270/summary.json`).
  A fresh post-promotion current-default c=2/c=4/c=8 matrix is generated-token
  green at `[137]*rows` on GPU1/XTX with accepted projection evidence and
  primitive correctness attached for each row count: c2 is full native/c-aware,
  c4 uses selected dense-context batch-gate layers `3,7,11`, and c8 remains
  native rowchunk2
  (`benchmarks/results/2026-06-04-hipengine-qwen35-current-default-c248-after-c4-selected-271/summary.json`).
  The no-flag sampler default now also covers c2: three explicit c2
  `batched_lm_head` repeats were green (`[137,137]`), and a post-change
  current-default c=2/c=4/c=8 matrix is green with row-aware `batched_lm_head`
  on every row count. c2 now has no decode or sampler blockers; c4/c8 remain
  correctness-only due to selected dense-context and rowchunk attention labels,
  respectively
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c2-batched-sampler-default-272/summary.json`).
  The c4 selected dense-context fallback is now narrower again: subset sweep
  showed `11` and `7,11` are red, `3,11` is green in `3/3` runs, and the
  no-flag c4 default is green at `[137]*4` with
  `batch_decode_attention_dense_context_batch_gate_layers="3,11"`. c8 rowchunk4
  plus selected dense-context probes remained red, so c8 stays rowchunk2
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c4-selected-layer-minimum-273/summary.json`).
  A final current-default refresh after both the c2 sampler and c4 layer-minimum
  changes is green for c=2/c=4/c=8 (`[137]*rows`) with row-aware
  `batched_lm_head` on all three row counts, c2 full-native/c-aware with empty
  decode/sampler/projection blockers, c4 selected dense-context layers `3,11`,
  and c8 rowchunk2
  (`benchmarks/results/2026-06-04-hipengine-qwen35-current-default-c248-final-274/summary.json`).
  A hard-window c4 fixture built from original rows4..7 shows the selected-layer
  c4 default is prompt/window-sensitive rather than a universal c4 fix: current
  `3,11` default is red (`[45,59,137,137]`), all tested no-rowchunk selected-
  layer expansions remain red, and native rowchunk2 is the only green control
  (`[137]*4`). The default is therefore retained for the primary first-four c4
  gate, while the rows4..7/c8 boundary stays on rowchunk2
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hard-c4-full-attn-boundary-275/summary.json`).
  The correctness-first auto default now rowchunks c4 as well as c3/c5..c8 and
  disables the implicit selected-layer c4 fallback unless explicitly requested:
  hard rows4..7 c4 turns green (`[137]*4`), primary c4 stays green, and the
  current c=2/c=4/c=8 matrix remains green with c4/c8 carrying only the rowchunk
  full-attention blocker
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c4-auto-rowchunk2-default-276/summary.json`).
  A c8 no-rowchunk output-path frontier then rejected simple O-path swaps as a
  rowchunk2 replacement: native paged context plus forced batch-GEMV O is red at
  `[82,137,137,137,45,11,137,137]`, while dense-context+batch-gate with either
  forced batch-GEMV O or per-row O is red at `[137,104,137,137,45,11,83,137]`.
  The same iteration's no-flag c=2/c=4/c=8 defaults remain green, so the current
  c8 blocker stays rowchunk/full-attention grouping rather than a standalone
  full-attention O projection choice
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c8-no-rowchunk-output-frontier-277/summary.json`).
  A follow-up rowchunk2 pairing permutation sweep shows the current c8 rowchunk2
  correctness is not only a property of the original contiguous row pairs:
  original order, pair-swapped order `[0,2,1,3,4,6,5,7]`, and head/tail cross
  order `[0,4,1,5,2,6,3,7]` all pass generated-token equality at `[137]*8`.
  This eliminates a narrow contiguous-pair-only suspicion while leaving the
  rowchunk2 diagnostic blocker itself in place
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c8-rowchunk2-pairing-permutation-278/summary.json`).
  A six-run repeat/stress pass over the same original, pair-swapped, and
  head/tail-cross fixtures stayed green in every run (`[137]*8` each), giving
  confidence that the current rowchunk2 c8 path is stable across those mixed
  row orderings even though rowchunk2 remains a diagnostic blocker
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c8-rowchunk2-stress-repeat-279/summary.json`).
  Moving the hard tail rows to the front and fully reversing row order keeps
  rowchunk2 green (`[137]*8` for `[4,5,6,7,0,1,2,3]` and `[7,6,5,4,3,2,1,0]`),
  while the same tail-front fixture without rowchunk is red at
  `[45,31,40,137,137,137,137,118]`. This pins rowchunk2 as the correctness
  boundary rather than a row-position artifact
  (`benchmarks/results/2026-06-04-hipengine-qwen35-c8-tailfront-rowchunk2-280/summary.json`).
  A current c8 no-flag rowchunk2 profiler refresh also stays generated-token
  green (`[137]*8`) with grouped-compact MoE, accepted c-aware projection, and
  row-aware `batched_lm_head`; the trace contains native batch full-attention
  context, native batch KV append, grouped-MoE, c-aware projection, and batched
  sampler argmax kernels. This is runtime/profiler evidence only because
  rowchunk2 still reports `native_caware_decode=false`
  (`benchmarks/results/2026-06-04-hipengine-qwen35-current-c8-rowchunk2-profiler-281/summary.json`).
  The matching current c4 rowchunk2 profiler refresh is likewise green
  (`[137]*4`) and contains the same expected native batch attention/KV,
  grouped-MoE, c-aware projection, segmented-state, and batched sampler kernels;
  it is also runtime/profiler evidence only because rowchunk2 remains diagnostic
  (`benchmarks/results/2026-06-04-hipengine-qwen35-current-c4-rowchunk2-profiler-282/summary.json`).
  A post-profiler current no-flag c2/c4/c8 refresh stays generated-token green
  (`c2 [137,137]`, `c4 [137]*4`, `c8 [137]*8`), confirming the current profiler
  evidence work did not perturb the primary equality gates
  (`benchmarks/results/2026-06-04-hipengine-qwen35-current-c248-post-profiler-refresh-283/summary.json`).
  The hard rows4..7 c4 gate also remains green under the current no-flag
  rowchunk2 default (`[137]*4`), while explicit no-rowchunk native full
  attention stays red (`[45,31,68,137]`) and no-rowchunk dense-context+batch-gate
  stays red (`[45,11,83,137]`); the hard c4 boundary therefore remains
  rowchunk/full-attention grouping, not selected dense context or metadata drift
  (`benchmarks/results/2026-06-04-hipengine-qwen35-hard-c4-post-profiler-refresh-284/summary.json`).
  A required-runtime baseline refresh then captured compact reusable c1 and c2
  scheduler-serial-bridge scaling references on GPU1/XTX and attached the raw
  references to a current c2 retained run. The c2 run stayed generated-token
  green (`[137,137]`) and `scaling.complete=true`, but the aggregate native/c1
  ratio was only `0.822` while native/serial-bridge was `1.049`, so this is
  baseline evidence only and still not a retained throughput/scaling claim
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-c1-serial-baseline-refresh-285/summary.json`).
  The same baseline-evidence path now covers c4 rowchunk2: a compact c4
  scheduler-serial-bridge reference is correctness-passed but blocked at
  `108.524` aggregate / `27.131` per-request tok/s, and a current c4 retained
  run with compact c1+c4-serial references stays green (`[137]*4`) with
  `scaling.complete=true`. Its aggregate native/c1 and native/serial ratios are
  `1.053` and `1.303`, respectively, but this is still evidence-only because
  the retained run is rowchunk2 with `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-serial-baseline-refresh-286/summary.json`).
  C8 now has the same baseline evidence: the compact c8 scheduler-serial-bridge
  reference is correctness-passed but blocked at `111.890` aggregate / `13.986`
  per-request tok/s, and a current c8 retained run with compact c1+c8-serial
  references stays green (`[137]*8`) with `scaling.complete=true`. Its aggregate
  native/c1 and native/serial ratios are `1.272` and `1.527`, respectively, but
  this is still evidence-only because c8 is also rowchunk2 with
  `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-serial-baseline-refresh-287/summary.json`).
  A focused long-context primitive probe for rows4/rows8 at 512..640 live tokens
  is green: batch paged KV append/context decode matches independent c1 exactly
  (`attn_batch_vs_c1_max_abs=0.0`, A/A `0.0`) and matches NumPy/dense-c1 within
  about `1.4e-8`. The rowchunk2 E2E blocker is therefore not reproduced by the
  isolated BF16 paged full-attention context/KV primitives alone; it likely sits
  in E2E full-attention composition/state/ordering
  (`benchmarks/results/2026-06-05-hipengine-qwen35-long-context-primitive-parity-288/summary.json`).
  A hard rows4..7 c4 no-rowchunk output frontier then ruled out a simple O-path
  swap as the rowchunk replacement: rowchunk2 remains green (`[137]*4`), native
  no-rowchunk O stays red (`[45,31,68,137]`), and forcing row-aware batch-GEMV O
  or per-row O both stay red (`[45,11,137,137]`). This keeps the hard-c4 target
  on coupled E2E full-attention composition/state/ordering rather than the O
  projection alone
  (`benchmarks/results/2026-06-05-hipengine-qwen35-hard-c4-output-frontier-289/summary.json`).
  A layer-scoped rowchunk diagnostic narrows that hard-c4 target: with row-aware
  batch-GEMV O, no-rowchunk is red (`[45,11,137,137]`), rowchunking only layer 3
  is red (`[45,58,137,137]`), rowchunking layers 3+7 is red
  (`[137,109,137,137]`), but rowchunking layers 3+7+11 is green on the hard
  rows4..7 fixture (`[137]*4`). Single/paired controls for 11, 7+11, and 3+11
  stay red
  (`benchmarks/results/2026-06-05-hipengine-qwen35-hard-c4-rowchunk-layer-scope-290/summary.json`).
  The following default-promotion check showed the first three layers are not
  enough for the original c4/c8 fixtures (`[137,137,137,118]` for c4 and
  `[137,137,137,118,137,137,137,137]` for c8), but rowchunking the first four
  full-attention producer layers (3,7,11,15) with row-aware batch-GEMV O keeps
  original c4, hard rows4..7 c4, and original c8 generated-token green. The
  correctness-first c4/c8 auto path now uses that narrower layer scope, while it
  remains diagnostic/non-retained because `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-c8-auto-layer-scope-291/summary.json`).
  C4 selected-layer profiler evidence is green and shows the expected native
  batch full-attention context/KV append kernels, grouped-compact MoE, accepted
  c-aware projection, row-aware output GEMV, segmented-state, and batched sampler
  kernels. A c8 selected-layer profiler was also captured, but later repeat
  stress reproduced a selected-layer c8 mismatch, so c8 was demoted back to the
  older all-layer rowchunk2 correctness-first default. This remains
  profiler/correctness evidence only and not a throughput/scaling claim
  (`benchmarks/results/2026-06-05-hipengine-qwen35-current-c4-layer-scope-profiler-292/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-current-c8-layer-scope-profiler-293/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-alllayer-demotion-297/summary.json`).
  Current c=2/c=4/c=8 no-flag generated-token equality is green vs independent
  c1 for every row (`c2 [137,137]`, `c4 [137]*4`, `c8 [137]*8`); c2 is
  full-native/c-aware, c4 uses selected rowchunk layers `[3,7,11,15]` plus
  row-aware batch-GEMV full-attention O, and c8 is back on all-layer rowchunk2
  for profiler-backed correctness. C8 first-four was not repeat-stable, first-
  five was red in 5/5 runs (`[137,137,137,137,137,31,137,137]`), first-six was
  green in normal 5/5 repeats but failed under `rocprofv3` with prefixes
  `[137,137,137,137,0,0,137,137]`; the current all-layer c8 profiler is green
  and has expected kernels. Follow-up c3/c5/c6/c7 coverage has c3/c5/c6 using
  selected layers `[3,7,11,15]`, while c7 stays on all-layer rowchunk2 after a
  selected-layer c7 repeat produced a mismatch. This remains a correctness-only
  diagnostic path
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c36-auto-layer-scope-295/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c5-auto-layer-scope-296/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-alllayer-demotion-297/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-first6-layer-scope-298/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-alllayer-profiler-299/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-current-c248-post-c8-profiler-300/summary.json`).
  A later active-loop c2 512/128 stability audit confirms the active verify
  invocation itself is currently repeat-green: five no-flag c2 runs all match
  independent c1 for all 137 generated tokens (`[137,137]` each), with full
  native/c-aware decode, `batched_lm_head`, and no decode blockers. The artifact
  remains correctness-only (not a retained throughput/scaling claim), so current
  correctness work should target c4/c8/full-attention evidence rather than
  another c2 mismatch hunt
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-active-stability-301/summary.json`).
  A fresh current c=2/c=4/c=8 baseline-attachment refresh is green for every
  row at prefix 137 and now loads compact c1/serial baselines, primitive GPU
  correctness, and accepted c-aware projection-dispatch evidence under the
  post-c8-profiler no-flag defaults. The retained bench now also auto-loads the
  committed projection-dispatch catalog when no explicit artifact/env override is
  supplied, so the active no-flag c2 verify uses the same accepted c-aware
  projection path instead of intermittently falling back to row-GEMV. C2 is full
  native/c-aware with grouped MoE and accepted projection but remains below c1
  on aggregate scaling; c4/c8 beat their serial bridges in this evidence run
  but remain diagnostic because
  rowchunked full attention keeps `native_caware_decode=false` (c4 selected
  layers `[3,7,11,15]`, c8 all rowchunk layers). A follow-up no-arg validation
  proves the default catalog path selects accepted projection candidates for
  c2/c4/c8 and keeps the accepted final c2/c4/c8 equality matrix green; it also
  records one transient initial c2 flake (`[61,137]`) before five immediate c2
  repeats returned `[137,137]`, so c2 equality is currently green but not claimed
  repeat-stability-closed. A focused current-c8 stress audit then ran three
  c8-all-layer-rowchunk2 no-arg cycles, each immediately followed by c2 no-arg;
  all c8 controls stayed `[137]*8` and all post-c8 c2 runs stayed `[137,137]`,
  with a final c4 spot check `[137]*4`. This does not close the intermittent c2
  flake, but it narrows the latest risk away from the current c8 default itself.
  A hard rows4..7 c4 refresh under the same no-explicit-projection defaults kept
  the selected rowchunk layers `[3,7,11,15]` green at `[137]*4`, while an explicit
  no-rowchunk native control remained red at `[45,31,68,137]`. A c8 first-seven
  rowchunk-scope probe (`[3,7,11,15,19,23,27]`) passed two normal controls at
  `[137]*8` but failed under `rocprofv3` at `[137,0,137,137,137,137,137,137]`,
  so the profiler-backed c8 default stayed all-layer rowchunk2. A follow-up c8
  first-eight probe (`[3,7,11,15,19,23,27,31]`) then failed normal repeat at
  `[137,137,137,19,137,137,137,137]`; during the same required matrix, c4's
  selected first-four default flaked at `[137,137,137,118]`, while two explicit
  all-layer c4 controls were green. The correctness-first auto default therefore
  demotes c4 to all-layer rowchunk2 as well; the post-demotion c2/c4/c8 matrix is
  green (`[137]*rows`). A fresh post-demotion c4 all-layer `rocprofv3` run also
  stays green at `[137]*4` with expected native batch full-attention context/KV,
  grouped-MoE, c-aware projection, row-aware output GEMV, segmented-state, and
  batched sampler kernels. A post-demotion baseline attachment refresh keeps
  c2/c4/c8 green with primitive correctness and c1/serial scaling references
  loaded; the immediately following active c2 verify flaked once at `[82,137]`
  before three active c2 repeats recovered to `[137,137]`. A focused repeat audit
  then reproduced the batched-sampler instability in two of eight c2 runs
  (`[137,104]`, `[82,137]`), so the correctness-first no-flag c2 sampler is
  demoted back to `serial_lm_head`; eight post-demotion c2 repeats are green at
  `[137,137]`, while c4/c8 still use `batched_lm_head` and stay green. A fresh
  post-demotion baseline attachment now records the same current defaults with
  c1/serial references and primitive correctness loaded: c2 `[137,137]` uses
  `serial_lm_head`, c4 `[137]*4` and c8 `[137]*8` keep `batched_lm_head`, and all
  rows select the accepted c-aware projection candidates automatically. A fresh
  c2 `rocprofv3 --kernel-trace` under the serial-sampler default also stays
  `[137,137]` and contains expected native full-attention context/KV append,
  grouped-MoE, c-aware projection, row-aware output GEMV, linear segment decode,
  and serial argmax kernels, with batched argmax absent. Current c4 and c8
  all-layer rowchunk2 profiler refreshes stay `[137]*4` / `[137]*8` and contain
  the expected native context/KV, grouped-MoE, c-aware projection, row-aware
  output GEMV, linear segment decode, and batched argmax kernels. C2 remains
  below aggregate-vs-c1, while c4/c8 remain diagnostic despite serial-bridge
  speedups because all-layer rowchunk keeps `native_caware_decode=false`. A final
  current-default repeat matrix passed three c2/c4/c8 cycles (`c2 [137,137]`,
  `c4 [137]*4`, `c8 [137]*8`) with the same sampler/projection defaults. A
  current forced c2 `batched_lm_head` + `serial_per_row` argmax diagnostic then
  passed 7/8 repeats but reproduced a red `[61,137]` run, so the explicit c2
  batched-sampler flake is not isolated to `batch_argmax_f32` alone. Adding a
  per-row final RMSNorm/cast diagnostic before the same batched LM-head
  projection plus serial argmax makes c2 8/8 green at `[137,137]`; the same
  per-row final-norm diagnostic with the real batch argmax is also 8/8 green.
  This clears the row-aware LM-head projection plus `batch_argmax_f32` suffix
  under the per-row final-norm repair. A finer split then shows neither single
  per-row repair is sufficient: batch RMSNorm + per-row cast passed 7/8 but
  failed `[82,0]`, and per-row RMSNorm + batch cast passed 7/8 but failed
  `[137,0]`. Thus both batched final RMSNorm and batched final cast can
  participate in the intermittent c2 batched-sampler instability. A follow-up
  attempted to promote c2's no-flag default to batched LM-head plus per-row norm
  and per-row cast, but rejected it after 7/8 repeats passed and one failed at
  `[137,104]`; c2 therefore stays `serial_lm_head` by default. A new
  LM-head-suffix audit reruns each row through the serial c=1 LM-head projection
  from the same normalized BF16 row; full-batch c2 final norm/cast plus this
  audit was 8/8 green at `[137,137]` with 136 audited steps, 272 audited rows,
  zero projection+argmax mismatches, and max value delta 0.0 per repeat. A
  follow-up final-norm/cast suffix audit reruns each row through serial c=1 final
  RMSNorm, FP16→BF16 cast, LM-head projection, and argmax from the same hidden
  row; full-batch c2 final norm/cast plus this audit was also 8/8 green at
  `[137,137]`, again with 136 audited steps, 272 audited rows, zero mismatches,
  and max value delta 0.0 per repeat. A serial suffix-fence variant that reruns
  the same serial suffix with host reads but without output comparison was also
  8/8 green at `[137,137]` with 136 fenced steps, 272 fenced rows, and 544 host
  reads per repeat. A kernel-only suffix fence (same serial suffix kernels but
  no serial host readback) was likewise 8/8 green at `[137,137]` with 136 fenced
  steps, 272 fenced rows, and zero suffix host reads per repeat. A narrower
  LM-head+argmax-only kernel fence from the already-normalized BF16 rows was not
  sufficient: it passed 7/8 repeats but failed once at `[0,137]` with row 0
  diverging at token 0. The complementary final RMSNorm+FP16→BF16 cast-only
  kernel fence (no LM-head, argmax, host readback, or output comparison) was
  8/8 green at `[137,137]` with 136 fenced steps, 272 fenced rows, and zero host
  reads per repeat. A final RMSNorm-only kernel fence (no cast, LM-head, argmax,
  host readback, or output comparison) was also 8/8 green at `[137,137]` with
  the same 136 fenced steps, 272 fenced rows, and zero host reads per repeat.
  Running the same serial RMSNorm kernel into a dedicated temp buffer, instead
  of the sampler's shared norm scratch, was also 8/8 green at `[137,137]`. A
  final FP16→BF16 cast-only kernel into a dedicated temp buffer was likewise 8/8
  green at `[137,137]`; however, a one-element final-cast temp-buffer fence was
  not sufficient, passing only 6/8 repeats with failures at `[82,137]` and
  `[137,104]`; a 64-element prefix fence was also not sufficient (5/8, failures
  at `[137,104]`, `[137,104]`, and `[82,104]`), and a 96-element prefix fence was
  not sufficient (7/8, one `[0,104]` failure), while 100-, 104-, 112-, 128-,
  256-, and 1024-element prefix final-cast temp-buffer fences were each 8/8 green
  at `[137,137]`. A sync-only fence (extra `device_synchronize` calls, no
  kernels, host readback, or output comparison) was also not sufficient: it
  passed 7/8 repeats but failed once at `[137,0]`. This clears the full sampler
  suffix for observed hidden inputs and suggests the remaining intermittent c2
  batched-sampler issue is upstream of the suffix or sensitive to the amount /
  pattern of lightweight final-suffix kernel work touching temp memory (all-green
  threshold above 96 and at/below 100 cast elements in this workload) rather than
  sampler norm-scratch clobbering, host readback, LM-head-only fencing, a minimal
  one-element launch, or extra synchronization alone. An opt-in non-blocking
  stabilizer variant (`--batch-sample-stabilize-cast-elems`) that records
  `native_row_aware_lm_head=true` and leaves sampler blockers empty was then
  tested: 100 elements was still not sufficient (7/8, one `[137,104]` failure),
  while 112 elements was 8/8 green at `[137,137]`. The no-flag c2 bench default
  was then promoted to `batched_lm_head` with the same 112-element stabilizer when
  the repo c2 sampler-equality artifact validates; the active c2 verify and four
  additional no-sampler-flag repeats were all green at `[137,137]` while recording
  `native_row_aware_lm_head=true` and empty sampler blockers. A later c1/serial
  baseline-attachment rerun found that 112 was still marginal under the retained
  evidence path (`[137,104]` once), so the default c2 stabilizer was raised to 128;
  explicit 128 was 4/4 green and the no-flag default with c1+serial baselines was
  also 4/4 green at `[137,137]`. A current c2 `rocprofv3 --kernel-trace` rerun then
  exposed that 128 was still marginal under profiler instrumentation (`[137,23]`),
  so the default c2 stabilizer was raised to 256; forced 256 and no-flag default256
  profiler runs were green at `[137,137]`, and two no-flag default256 c1+serial
  baseline-attachment checks were also green. A current no-flag c2/c4/c8 equality
  refresh then found that c8 without a stabilizer was still intermittent
  (`[137,31]` and `[0]` failures in two no-flag c8 runs), while forced c8
  stabilize-cast=256 was 3/3 green. The no-flag c8 default now also enables the
  256-element stabilizer; after that change, c2/c4/c8 all passed generated-token
  equality vs independent c=1 at `[137,137]`, `[137]*4`, and `[137]*8` respectively
  (c8 4/4 repeats), with `batched_lm_head`, `native_row_aware_lm_head=true`, and
  empty sampler blockers (c2/c8 use 256-element stabilizers; c4 does not).
  Follow-up c2 `rocprofv3 --kernel-trace` smokes stayed green at `[137,137]`
  (first 112, then current 256) and captured native batch full-attention/KV,
  grouped-MoE, accepted c-aware projection, row-aware output GEMV, linear segment
  decode, batch residual combine, `batch_argmax_stage{1,2}`, and FP16→BF16
  final-cast/stabilizer kernels. A current c2 default256 baseline/profiler
  attachment loaded c1, c2 serial bridge, primitive c2 correctness, and
  retained-gate-shaped profiler references; equality stayed `[137,137]`, scaling
  was complete, and native c-aware decode / batch blockers were green, but the row
  remains non-retained because aggregate-vs-c1 is still below 1.0 and the profiler
  was captured from the no-baseline current default smoke rather than the exact
  retained-attachment command.
  The matching c4 and c8 no-flag profiler smokes stayed green at `[137]*4` and
  `[137]*8` with row-aware `batched_lm_head`, accepted c-aware projection,
  rowchunked native batch full-attention/KV, grouped-MoE, row-aware output GEMV,
  linear segment decode, batch residual combine, `batch_argmax_stage{1,2}`, and
  final-cast kernels; after the c8 stabilizer promotion, the current c8 default256
  profiler smoke was refreshed and stayed green at `[137]*8` with the same expected
  native/projection/sampler kernel families and rowchunked full-attention blockers.
  Matching current c4/c8 baseline/profiler attachments then loaded c1,
  serial-bridge, primitive correctness, and retained-gate-shaped profiler
  references; generated-token equality stayed `[137]*4` / `[137]*8` and scaling
  was complete, but both rows remain non-retained because rowchunked full attention
  keeps `native_caware_decode=false` and the profilers were captured from
  no-baseline current default smokes rather than the exact retained-attachment
  commands.
  This is correctness/runtime evidence, not a retained
  throughput/scaling claim. The immediately following post-audit no-flag c2/c4/c8 controls recovered green
  (`[137,137]`, `[137]*4`, `[137]*8`). No retained throughput/scaling claim is made
  (`benchmarks/results/2026-06-05-hipengine-qwen35-current-c248-baseline-attach-302/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-noarg-c248-post-projection-default-303/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-to-c2-stress-304/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-hard-c4-current-default-305/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-first7-profiler-boundary-306/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-first8-c4-demotion-307/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c4-alllayer-profiler-308/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-post-demotion-baseline-attach-309/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-sampler-demotion-310/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-post-c2-sampler-baseline-attach-311/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-serial-profiler-312/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-alllayer-profiler-313/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c4-alllayer-profiler-314/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-final-default-repeat-315/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-batched-lm-serial-argmax-current-316/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-sampler-perrow-norm-317/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-perrow-norm-batch-argmax-318/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-sampler-norm-cast-split-319/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-default-batched-perrow-reject-320/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-lm-head-audit-321/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-norm-audit-322/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-suffix-fence-323/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-suffix-kernel-fence-324/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-lm-head-kernel-fence-325/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-norm-kernel-fence-326/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-rmsnorm-kernel-fence-327/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-sync-fence-328/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-rmsnorm-temp-fence-329/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-temp-fence-330/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-tiny-fence-331/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-elems1024-fence-332/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-elems64-256-fence-333/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-elems128-fence-334/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-elems96-112-fence-335/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-elems104-fence-336/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-final-cast-elems100-fence-337/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-stabilize-cast100-112-338/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-default-stabilize112-339/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c248-default-equality-340/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-stabilized-batched-profiler-341/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c4-default-profiler-342/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-default-profiler-343/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-default128-baseline-344/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-default256-profiler-345/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c2-default256-baseline-349/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c4-default-baseline-350/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-default-stabilizer-346/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-default256-profiler-347/summary.json`,
  `benchmarks/results/2026-06-05-hipengine-qwen35-c8-default256-baseline-348/summary.json`).
  A focused c4 rowchunk-scope rerun then re-promotes c4 from all-layer rowchunk2
  to the selected first-four full-attention producer layers `[3,7,11,15]` under
  the current stabilized sampler/projection defaults. Explicit first-four c4
  rowchunk was 4/4 green, post-change no-flag c4 default was 4/4 green
  (`[137]*4`), while explicit no-rowchunk native and first-three controls failed
  deterministically at row3/token118. This narrows the c4 full-attention
  diagnostic blocker but does not make a retained performance/scaling claim:
  selected rowchunk layers still report `native_caware_decode=false`; at that
  point c8 stayed on all-layer rowchunk2 until a selected-layer path became
  profiler-stable
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-rowchunk-first4-351/summary.json`).
  The matching c4 first-four profiler/baseline attachment then stayed green at
  `[137]*4` with c1, c4 serial-bridge, primitive c4 correctness, and a fresh
  retained-gate-shaped `rocprofv3` summary loaded. Scaling references are complete
  (`aggregate_vs_c1=1.1297`, `aggregate_vs_serial_bridge=1.3987`), and the trace
  contains expected native batch full-attention/KV, grouped-MoE, accepted c-aware
  projection, row-aware output GEMV, linear segment decode, batch residual combine,
  batched argmax, and final-cast kernels. This remains required runtime evidence
  only because selected rowchunk layers keep `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first4-profiler-baseline-352/summary.json`).
  A follow-up c4 layer-subset probe narrows the primary c4 rowchunk scope again:
  layer 11 alone passed 5/5 normal repeats, the layer-11 `rocprofv3` smoke stayed
  `[137]*4`, and the post-change no-flag c4 default was 2/2 green with only
  layer 11 rowchunked. Single-layer controls for 3 and 15 were red (layer15-only
  failed `[82,137,137,137]`), so layer 11 was the narrowest proven c4 scope for
  the primary first-four fixture. This remains correctness/runtime evidence only
  because that one selected rowchunk layer still keeps `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-layer11-rowchunk-357/summary.json`).
  The matching c4 layer-11 baseline/profiler attachment stayed green at
  `[137]*4` with c1, c4 serial-bridge, primitive c4 correctness, and the layer-11
  profiler summary loaded. Scaling references are complete
  (`aggregate_vs_c1=1.1395`, `aggregate_vs_serial_bridge=1.4107`), but this is
  still evidence-only because layer-11 rowchunk keeps `native_caware_decode=false`
  and the profiler was captured from a no-baseline smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-layer11-baseline-358/summary.json`).
  A focused c8 first-nine rowchunk probe then narrows the c8 full-attention
  diagnostic scope from all ten full-attention producer layers to
  `[3,7,11,15,19,23,27,31,35]`: explicit first-nine repeats were 3/3 green,
  the `rocprofv3` smoke stayed `[137]*8`, and the post-change no-flag c8 default
  was 2/2 green with the expected native batch/projection/sampler kernel families.
  This remains correctness/runtime evidence only because selected rowchunk layers
  still keep `native_caware_decode=false`; prior first-seven/first-eight c8
  selected scopes were unstable/red, so first-nine is the current narrowest
  profiler-backed c8 scope
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-first9-rowchunk-353/summary.json`).
  The matching c8 first-nine baseline/profiler attachment stayed green at
  `[137]*8` with c1, c8 serial-bridge, primitive c8 correctness, and the first-
  nine profiler summary loaded. Scaling references are complete
  (`aggregate_vs_c1=1.3066`, `aggregate_vs_serial_bridge=1.5690`), but the row
  remains evidence-only because selected rowchunk layers keep
  `native_caware_decode=false`. The immediately following active c2 verify
  flaked once at `104` and an explicit post-c8 repeat reproduced `[137,104]`,
  before two more repeats recovered to `[137,137]`; c2 remains green for final
  measurement but the intermittent sampler-stability risk is still open
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-first9-baseline-354/summary.json`).
  The c2 no-flag sampler stabilizer was then raised from 256 to 512 elements:
  forced c2 stabilize512 was 4/4 green, a current c8 first-nine control stayed
  `[137]*8`, the following four no-flag c2 default512 repeats were 4/4 green at
  `[137,137]`, and the final active c2 verify printed 137. This is a
  correctness-stability workaround, not a retained throughput claim
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-default512-stabilizer-355/summary.json`).
  The matching c2 default512 profiler/baseline attachment then stayed green at
  `[137,137]` with c1, c2 serial-bridge, primitive c2 correctness, and a fresh
  retained-gate-shaped profiler summary loaded. Batch/decode/sampler/projection
  blockers are empty and `native_caware_decode=true`; scaling references are
  complete (`aggregate_vs_c1=0.8382`, `aggregate_vs_serial_bridge=1.0690`). This
  refreshes required runtime evidence after the stabilizer change, but c2 remains
  non-retained because aggregate-vs-c1 is below 1.0 and the profiler was captured
  from a no-baseline smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-default512-profiler-baseline-356/summary.json`).
  The c8 first-nine baseline/profiler attachment was then refreshed under the
  current default512 c2 stabilizer. The current no-flag c8 row stayed green at
  `[137]*8` with c1, c8 serial-bridge, primitive c8 correctness, and the first-
  nine profiler summary loaded; the following active c2 verify stayed `[137,137]`
  with `stabilize_cast_elems=512`. Scaling references remain complete
  (`aggregate_vs_c1=1.2984`, `aggregate_vs_serial_bridge=1.5591`), but this is
  still evidence-only because first-nine rowchunk keeps `native_caware_decode=false`
  and the profiler was captured from a no-baseline smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-first9-baseline-refresh-359/summary.json`).
  A hard rows4..7 c4 recheck then invalidated the c4 layer-11-only default as a
  prompt/window-stable auto policy: layer 11 alone failed `[45,11,68,137]`, while
  all-layer rowchunk2 passed `[137]*4`. The no-flag c4 default was demoted back
  to all-layer rowchunk2; post-change hard rows4..7 and primary first-four c4
  both stayed `[137]*4`. This is a correctness-first demotion, not a retained
  throughput/scaling claim, and c4 still has the full-attention rowchunk blocker
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-hard-demotion-360/summary.json`).
  The c2 batched-sampler default was then demoted back to `serial_lm_head` after
  the token-104 flake reproduced even with a wider 1024-element stabilizer. Four
  forced 1024 runs and four temporary no-flag default1024 runs were green, but
  the required active verify immediately afterward failed `[137,104]`. Removing
  c2 from the no-flag sampler-evidence defaults produced six serial no-flag
  repeats plus the final active verify at `[137,137]`, with
  `native_caware_decode=true` and no decode blockers. This is correctness-first
  c2 stability evidence, not a retained throughput/scaling claim
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-sampler-redemotion-361/summary.json`).
  A current no-flag c=2/c=4/c=8 equality refresh after the c2 and c4 demotions is
  green for every row: c2 `[137,137]` with `native_caware_decode=true` and
  `serial_lm_head`, c4 `[137]*4` with all-layer rowchunk2, and c8 `[137]*8` with
  first-nine rowchunk2 plus `stabilize_cast_elems=256`. This refreshes the
  correctness gate under the current defaults only; c4/c8 still have rowchunk
  blockers and no retained throughput/scaling claim is made
  (`benchmarks/results/2026-06-05-hipengine-qwen35-current-c248-final-after-demotions-362/summary.json`).
  The matching current c2 serial-sampler baseline/profiler attachment stayed
  green at `[137,137]` with c1, c2 serial-bridge, primitive c2 correctness, and
  serial-sampler profiler evidence loaded. Batch/decode/projection blockers are
  empty and `native_caware_decode=true`; scaling references are complete
  (`aggregate_vs_c1=0.8441`, `aggregate_vs_serial_bridge=1.0767`). This is still
  evidence-only because aggregate-vs-c1 remains below 1.0 and the profiler source
  is a serial-sampler summary rather than a retained-gate-shaped batched-sampler
  trace
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-serial-baseline-final-365/summary.json`).
  The matching current c4 all-layer baseline/profiler attachment stayed green at
  `[137]*4` with c1, c4 serial-bridge, primitive c4 correctness, and the all-layer
  profiler summary loaded. Scaling references are complete
  (`aggregate_vs_c1=1.0666`, `aggregate_vs_serial_bridge=1.3205`), but this is
  still evidence-only because all-layer rowchunk keeps `native_caware_decode=false`
  and the profiler was captured from a no-baseline smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-alllayer-baseline-refresh-363/summary.json`).
  A focused hard-window c4 first-nine rowchunk probe then narrowed the current
  c4 default from all ten full-attention producer layers to
  `[3,7,11,15,19,23,27,31,35]`: explicit first-nine and post-change no-flag
  runs are green at `[137]*4` for both the primary first-four fixture and the
  hard rows4..7 fixture. This keeps the c4 rowchunk blocker
  (`native_caware_decode=false`) but leaves only final full-attention producer
  layer 39 native, matching the current c8 first-nine boundary
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first9-hard-default-366/summary.json`).
  A post-promotion current c=2/c=4/c=8 equality refresh is final-green after a
  c2 repeat recovery: the first c2 attempt flaked at `[137,0]`, then three c2
  repeats plus the required active verify recovered to `[137,137]`; c4 stayed
  `[137]*4` on the first-nine default and c8 stayed `[137]*8` on first-nine.
  This validates the current final matrix while keeping the residual c2 transient
  risk open and making no retained throughput/scaling claim
  (`benchmarks/results/2026-06-05-hipengine-qwen35-current-c248-after-c4-first9-367/summary.json`).
  The matching current c4 first-nine baseline/profiler attachment then stayed
  green at `[137]*4` with c1, c4 serial-bridge, primitive c4 correctness, and a
  fresh first-nine profiler summary loaded. Scaling references are complete
  (`aggregate_vs_c1=1.0725`, `aggregate_vs_serial_bridge=1.3279`), but this is
  still evidence-only because selected first-nine rowchunk keeps
  `native_caware_decode=false` and the profiler was captured from a no-baseline
  smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first9-baseline-profiler-368/summary.json`).
  A follow-up c4 first-eight probe narrowed the c4 default again to
  `[3,7,11,15,19,23,27,31]`: explicit first-eight and post-change no-flag runs
  are green at `[137]*4` for both the primary first-four fixture and the hard
  rows4..7 fixture, and the post-change c=2/c=4/c=8 matrix is green. This kept
  the c4 rowchunk blocker (`native_caware_decode=false`) but left final layers
  35 and 39 native
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first8-default-371/summary.json`).
  A next c4 first-seven probe narrowed the c4 default to
  `[3,7,11,15,19,23,27]`: explicit first-seven and post-change no-flag runs are
  green at `[137]*4` for both primary and hard rows4..7, and the post-change
  c=2/c=4/c=8 matrix is green. This kept the c4 rowchunk blocker but left layers
  31, 35, and 39 native
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first7-default-372/summary.json`).
  A next c4 first-six probe narrowed the c4 default to
  `[3,7,11,15,19,23]`: explicit first-six and post-change no-flag runs are green
  at `[137]*4` for both primary and hard rows4..7, and the post-change
  c=2/c=4/c=8 matrix is green. This keeps the c4 rowchunk blocker but leaves
  layers 27, 31, 35, and 39 native
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first6-default-373/summary.json`).
  A first-five follow-up (`[3,7,11,15,19]`) is not prompt-stable: primary stayed
  `[137]*4`, but hard rows4..7 failed at `[137,31,137,137]`. The current
  first-six default stayed green on the hard fixture and the c=2/c=4/c=8 matrix
  stayed green, so first-six is the current c4 boundary and was not narrowed
  further
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first5-red-boundary-374/summary.json`).
  A sparse five-layer follow-up that kept layer 23 but dropped layer 19
  (`[3,7,11,15,23]`) also failed: primary was `[137,137,137,118]` and hard
  rows4..7 was `[137,72,137,137]`. The current first-six default stayed green on
  both fixtures and the c=2/c=4/c=8 matrix stayed green, so layer 19 is currently
  required along with layer 23 for prompt-stable c4 equality
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-sparse5-red-boundary-376/summary.json`).
  A sparse five-layer follow-up that kept layers 19 and 23 but dropped layer 15
  (`[3,7,11,19,23]`) kept primary green at `[137]*4` but failed hard rows4..7 at
  `[137,31,137,137]`. The current first-six default stayed green on both
  fixtures and the c=2/c=4/c=8 matrix stayed green, so layer 15 is currently
  required for prompt-stable c4 hard-window equality
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-sparse-drop15-boundary-377/summary.json`).
  A sparse five-layer follow-up that kept layers 15, 19, and 23 but dropped
  layer 11 (`[3,7,15,19,23]`) kept primary green at `[137]*4` but failed hard
  rows4..7 at `[137,72,137,137]`. One current first-six hard control flaked at
  `[137,137,137,0]`, but three immediate repeats recovered to `[137]*4` and the
  c=2/c=4/c=8 matrix stayed green. Thus layer 11 is currently required for
  prompt-stable c4 hard-window equality, with residual transient risk noted
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-sparse-drop11-boundary-378/summary.json`).
  A sparse five-layer follow-up that kept layers 3, 11, 15, 19, and 23 but
  dropped layer 7 (`[3,11,15,19,23]`) kept primary green at `[137]*4` but failed
  hard rows4..7 broadly at `[45,72,83,137]`. The current first-six default stayed
  green on both fixtures and the c=2/c=4/c=8 matrix stayed green, so layer 7 is
  currently required for prompt-stable c4 hard-window equality
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-sparse-drop7-boundary-379/summary.json`).
  A sparse five-layer follow-up that kept layers 7, 11, 15, 19, and 23 but
  dropped layer 3 (`[7,11,15,19,23]`) failed primary at `[137,137,137,118]` and
  hard rows4..7 broadly at `[45,58,68,137]`. The current first-six default stayed
  green on both fixtures and the c=2/c=4/c=8 matrix stayed green, so layer 3 is
  currently required for prompt-stable c4 equality
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-sparse-drop3-boundary-380/summary.json`).
  The current c4 first-six default now also has refreshed profiler/scaling
  evidence: a no-baseline `rocprofv3` smoke stayed `[137]*4` with expected native
  batch full-attention context/KV, grouped-MoE, accepted c-aware projection,
  row-aware output GEMV, and batched sampler kernels; the retained-bench
  attachment loaded c1, c4 serial-bridge, primitive c4 correctness, and that
  profiler while preserving `[137]*4`. Scaling references are complete
  (`aggregate_vs_c1=1.1075`, `aggregate_vs_serial_bridge=1.3711`), but this is
  still evidence-only because first-six rowchunk keeps `native_caware_decode=false`
  and the profiler was captured from a no-baseline smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c4-first6-profiler-refresh-375/summary.json`).
  The matching current c8 first-nine baseline/profiler attachment also stayed
  green at `[137]*8` with c1, c8 serial-bridge, primitive c8 correctness, and the
  first-nine profiler summary loaded. Scaling references are complete
  (`aggregate_vs_c1=1.3060`, `aggregate_vs_serial_bridge=1.5683`), but this is
  still evidence-only because selected first-nine rowchunk keeps
  `native_caware_decode=false` and the profiler was captured from a no-baseline
  smoke
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-first9-baseline-final-364/summary.json`).
  A fresh current c8 first-nine profiler recapture later reproduced row3/token118
  red (`[137,137,137,118,137,137,137,137]`), so the c8 default was demoted back
  to all-layer rowchunk2. The all-layer profiler control, retained-bench
  attachment, and post-change c=2/c=4/c=8 matrix are green; c8 scaling references
  remain complete (`aggregate_vs_c1=1.2751`, `aggregate_vs_serial_bridge=1.5311`).
  This is still correctness/runtime evidence only because all-layer rowchunk keeps
  `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-alllayer-redemotion-369/summary.json`).
  A non-prefix c8 all-minus-layer35 probe (`[3,7,11,15,19,23,27,31,39]`) failed
  at `[137,137,137,0,137,137,137,137]`, while the current all-layer control and
  c=2/c=4/c=8 matrix stayed green. This establishes layer 35 as required for the
  current c8 all-layer rowchunk boundary
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop35-boundary-381/summary.json`).
  The all-minus-layer31 scope (`[3,7,11,15,19,23,27,35,39]`) was then promoted
  as the no-flag c8 default after three explicit green repeats and a post-change
  no-flag c8/c2/c4/c8 matrix all stayed at prefix 137. This narrows the c8
  rowchunk diagnostic by leaving layer 31 native, but it is still evidence-only
  because selected rowchunk layers keep `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop31-promotion-382/summary.json`).
  The next drop27/drop31 scope (`[3,7,11,15,19,23,35,39]`) was then promoted as
  the no-flag c8 default after three explicit green repeats and a post-change
  no-flag c8/c2/c4/c8 matrix all stayed at prefix 137. This leaves layers 27 and
  31 native while rowchunking 35/39, but remains evidence-only because selected
  rowchunk layers keep `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop27-promotion-383/summary.json`).
  A follow-up drop-layer23 attempt (`[3,7,11,15,19,35,39]`) failed at
  `[137,137,137,137,137,31,137,137]`, while the current drop27/drop31 c8 default
  and c=2/c=4/c=8 matrix stayed green. Thus layer 23 is currently required for
  the c8 rowchunk boundary
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop23-boundary-384/summary.json`).
  A follow-up drop-layer19 attempt (`[3,7,11,15,23,35,39]`) failed at
  `[137,137,137,118,137,72,137,137]`; the first c=2/c=4/c=8 matrix had one
  transient current-c8 flake at `[137,137,137,137,0,137,137,137]`, but the
  direct current-c8 control plus three immediate repeats were green and a full
  retry matrix recovered. Thus layer 19 is currently required, with residual
  transient c8 risk noted
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop19-boundary-385/summary.json`).
  A follow-up drop-layer15 attempt (`[3,7,11,19,23,35,39]`) failed at
  `[137,137,137,137,137,31,137,137]`, while the current drop27/drop31 c8 default
  and c=2/c=4/c=8 matrix stayed green. Thus layer 15 is currently required for
  the c8 rowchunk boundary
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop15-boundary-386/summary.json`).
  A follow-up drop-layer11 attempt (`[3,7,15,19,23,35,39]`) failed at
  `[137,137,137,137,137,72,137,137]`, while the current drop27/drop31 c8 default
  and c=2/c=4/c=8 matrix stayed green. Thus layer 11 is currently required for
  the c8 rowchunk boundary
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop11-boundary-387/summary.json`).
  A follow-up drop-layer7 attempt (`[3,11,15,19,23,35,39]`) failed at
  `[137,137,137,137,45,72,83,137]`, while the current drop27/drop31 c8 default
  and c=2/c=4/c=8 matrix stayed green. Thus layer 7 is currently required for
  the c8 rowchunk boundary
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop7-boundary-388/summary.json`).
  Re-probing drop-layer39 under the narrower drop27/drop31 default then stayed
  3/3 green with `[3,7,11,15,19,23,35]`; the no-flag c8 default was promoted to
  that list and the post-change c=2/c=4/c=8 matrix stayed green. This leaves
  layers 27, 31, and 39 native, but remains evidence-only because selected
  rowchunk layers keep `native_caware_decode=false`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop39-promotion-389/summary.json`).
  A follow-up current-default drop-layer35 / c4-aligned first-six attempt
  (`[3,7,11,15,19,23]`) was green twice but failed the third repeat at
  `[137,137,137,137,137,58,137,137]`; the current c8 default and c=2/c=4/c=8
  matrix stayed green. Thus layer 35 remains required for repeat-stable c8
  equality under the current narrowed default
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop35-current-boundary-390/summary.json`).
  The final current-default drop-layer3 attempt (`[7,11,15,19,23,35]`) failed at
  `[137,137,137,118,45,58,68,137]`, while the current c8 default and c=2/c=4/c=8
  matrix stayed green. Thus layer 3 is currently required, closing the current
  c8 selected-layer boundary at `[3,7,11,15,19,23,35]`
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-drop3-boundary-391/summary.json`).
  Fresh profiler/runtime evidence now covers that narrowed no-flag c8 default:
  a `rocprofv3 --kernel-trace` baseline/profiler attachment stayed `[137]*8`,
  loaded c1, c8 serial-bridge, primitive c8 correctness, and expected native
  batch/projection/sampler kernels, and the final attachment remained green with
  scaling references complete (`aggregate_vs_c1=1.3236`,
  `aggregate_vs_serial_bridge=1.5895`). The post-profiler c=2/c=4/c=8 matrix
  stayed `[137,137]`, `[137]*4`, `[137]*8`, and active c2 verify stayed 137.
  This remains correctness/runtime evidence only because selected rowchunk keeps
  `native_caware_decode=false`; no retained throughput/scaling claim is made
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c8-current-profiler-refresh-392/summary.json`).
  A follow-up c2 full-hidden sampler-stabilizer stress showed why c2 is still
  not re-promoted to the no-flag `batched_lm_head` default: explicit c2
  `batched_lm_head` with `stabilize_cast_elems=2048` was 20/20 green (including
  four c8→c2 cycles) with native row-aware sampler metadata and no sampler
  blockers, but a temporary no-flag promotion candidate reproduced the old
  row0/token82 flake once in 8 runs and a subsequent serial no-flag active
  verify transiently printed 104 before recovery repeats returned to 137. The
  c2 default therefore remains `serial_lm_head`; this is sampler-stability
  evidence only and no retained throughput/scaling claim is made
  (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-fullcast-sampler-stress-393/summary.json`).
  An earlier post-c8 stability stress ran three then-current no-flag c8 all-layer
  default → c2 default cycles. Every c8 run stayed `[137]*8`, every following c2
  run stayed `[137,137]` with `serial_lm_head` and `native_caware_decode=true`,
  and the required active c2 verify also stayed `[137,137]`. This adds runtime
  confidence after the iter367 `[137,0]` c2 flake, but the residual transient
  risk remains open and no retained throughput/scaling claim is made
  (`benchmarks/results/2026-06-05-hipengine-qwen35-post-c8-c2-stress-370/summary.json`).
  A grouped-compact auto-MoE promotion was tried but rolled back: the first
  primitive-attached repeat kept c2 green but rejected c4 at `[137,137,137,0]`
  and c8 at `[137,137,137,137,137,137,0,137]`. Repo-relative primitive GPU
  correctness artifacts now pass and are attached for c=2/c=4/c=8; with auto MoE
  restored to `selected_c1` for c<=8, generated-token equality remains green for
  every row at prefix 137 and `primitive_batch_correctness` is loaded/passed for
  each row count. Remaining blockers are c4/c8 rowchunked full attention,
  selected-c1 batch MoE metadata, row-GEMV projection dispatch without accepted
  c-aware evidence, and missing profiler/scaling/baseline artifacts
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-primitive-attached/summary.json`).
  The current selected-c1 c2 path now has a captured `rocprofv3 --kernel-trace`
  artifact attached to a 512/128 retained-bench rerun: equality remains
  `[137,137]`, primitive correctness is loaded/passed, and profiler provenance /
  kernel-duration blockers are absent. A follow-up c2 rerun attaches the
  combined c2/c4/c8 projection-dispatch catalog and now selects
  `benchmark_accepted_caware_projection` with empty projection blockers while
  preserving `[137,137]` equality, primitive correctness, and profiler evidence.
  This is still not a throughput claim; the artifact remains blocked by
  selected-c1 MoE retained metadata and aggregate-vs-c1 scaling
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-current-projection-dispatch/summary.json`).
  The current c4 correctness-first rowchunk2/selected-c1 path also has attached
  baseline references plus a captured `rocprofv3 --kernel-trace` retained-bench
  rerun: equality remains `[137,137,137,137]`, primitive correctness is
  loaded/passed, profiler provenance / kernel-duration blockers are absent, and
  scaling references are complete. A follow-up c4 rerun now attaches the combined
  c2/c4/c8 projection-dispatch catalog and selects
  `benchmark_accepted_caware_projection` / `gemv_awq_selected_dual_pack8_strided_c4`
  with empty projection blockers while preserving equality and profiler evidence.
  This is still not a throughput claim because c4 remains blocked by rowchunked
  full attention (`native_caware_decode=false`) and selected-c1 MoE retained
  metadata
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c4-current-projection-dispatch/summary.json`).
  The same evidence is now attached for c8: the current rowchunk2/selected-c1
  512/128 retained-bench rerun keeps equality green at
  `[137,137,137,137,137,137,137,137]`, primitive correctness is loaded/passed,
  profiler provenance / kernel-duration blockers are absent, and c1 / serial
  scaling references are complete. A follow-up c8 rerun now attaches the combined
  c2/c4/c8 projection-dispatch catalog and selects
  `benchmark_accepted_caware_projection` / `gemv_awq_selected_dual_pack8_strided_c8`
  with empty projection blockers while preserving full equality in the final
  artifact. It remains non-retained for the same reasons: rowchunked full
  attention (`native_caware_decode=false`) and selected-c1 MoE metadata. The c8
  projection-dispatch reruns showed pre-final equality instability, so keep
  repeatability under observation before any promotion
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-current-projection-dispatch/summary.json`).
  A grouped-compact MoE rerun with the same combined projection-dispatch catalog
  keeps c2/c4/c8 generated-token equality green
  (`[137,137]`, `[137,137,137,137]`, and
  `[137,137,137,137,137,137,137,137]`), reports
  `moe_decode_path=grouped_compact`, `moe_grouped_compact_layers=40`,
  `moe_selected_c1_fallback_layers=0`, and keeps projection blockers empty. This
  removes the selected-c1 MoE metadata blocker for correctness evidence
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-grouped-projection-equality/summary.json`).
  The fully native-caware c2 grouped/projection path now also has a captured
  `rocprofv3 --kernel-trace` retained-bench rerun: equality remains `[137,137]`,
  primitive correctness is loaded/passed, grouped MoE and projection blockers are
  absent, and profiler provenance / kernel-duration blockers are absent. This is
  still not a throughput claim because aggregate-vs-c1 scaling is below the
  retained threshold. The c4 and c8 grouped/projection rowchunk2 paths now have
  the same profiler-backed evidence: c4 equality remains `[137,137,137,137]`,
  c8 equality remains `[137,137,137,137,137,137,137,137]`, primitive correctness
  is loaded/passed, grouped MoE / projection / profiler blockers are absent, and
  the only batch blockers are rowchunked full-attention /
  `native_caware_decode=false`. A three-run c8 repeatability check then kept the
  same rowchunk2 evidence path green (`[137]*8` on every run), so the c8
  rowchunk2 repeatability caveat is closed for this path; full-native attention
  remains the promotion/performance-claim blocker
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-grouped-projection-profile/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c4-grouped-projection-profile/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-grouped-projection-profile/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-grouped-projection-repeatability/summary.json`).
  Long-context primitive checks at retained-like context length 520 are green for
  c3, c4, and c8 BF16 paged KV append plus batched full-attention context decode,
  including the Qwen/PARO-real full-attention geometry (`num_q_heads=16`,
  `num_kv_heads=2`, `head_dim=256`): batch-vs-independent-c1 max_abs and batch
  A/A max_abs are both `0.0`, so the standalone append/context primitive is
  eliminated at the exact grouped>=3 threshold and larger retained row counts
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-long-context-primitive-attn/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c48-long-context-primitive-attn/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c348-realshape-primitive-attn/summary.json`).
  A c4/c8 full-attention output-path contrast first showed the row-aware
  `batch_gemv` O-projection path was required for the green rowchunk2
  grouped/projection path: `batch_gemv` kept c4/c8 equality green, while forcing
  native `batch` output made the same workload red (`c4=[137,137,137,118]`,
  `c8=[137,137,137,118,45,137,68,137]`). The same contrast at c3 kept both
  output paths green (`[137,137,137]`), so the native batch O-output bug was a
  c4/c8 issue rather than the exact grouped>=3 threshold
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c48-fullattn-output-batch-control/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-fullattn-output-path-contrast/summary.json`). A rowchunk3
  follow-up narrowed the split further: c4 with native `batch` O-output was green
  under rowchunk3 (`[137,137,137,137]`), but c4 full-native/no-chunk native output
  was still red (`[137,137,137,118]`), c4 rowchunk3 plus `batch_gemv` output was
  red (`[82,137,137,137]`), and c8 rowchunk3 remained red for both output modes
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c4-c8-rowchunk3-output-contrast/summary.json`). The current
  runtime first kept native batch O projection on the leading row chunk while
  forcing row-aware `batch_gemv` only for nonzero multi-row chunks; later compact
  derived-row evidence showed the leading multi-row chunk can also be red. Runtime
  now forces row-aware `batch_gemv` for every multi-row rowchunk, records
  `native_batch_with_row_chunk_batch_gemv`, and preserves c4/c8 rowchunk2 equality
  (`[137]*4`, `[137]*8`). Retained-bench still uses `batch` as the default
  full-attention output path, but rowchunk diagnostics repair multi-row chunks
  internally. A no-chunk full-native contrast with output `batch_gemv` is still
  red (`c4=[82,137,137,137]`, `c8=[82,137,137,137,45,11,137,137]`), so forcing
  GEMV O projection alone is not the retained full-native fix. A post-fix
  full-native refresh keeps that boundary: c4 native context+native O is red
  (`[137,137,137,118]`), forced batch-GEMV O alone is red
  (`[82,137,137,137]`), c4 context-only replay with native O remains green
  (`[137]*4`), and context-only plus batch-GEMV O is red
  (`[137,104,137,137]`); c8 stays red for batch-GEMV O and context-only plus
  batch-GEMV O (`[82,137,137,137,45,11,137,137]` and
  `[137,104,137,137,45,11,83,137]`). The remaining retained blocker is native
  grouped full-attention context/pre-QKV/no-rowchunk behavior, not the rowchunk O
  path (`benchmarks/results/2026-06-03-hipengine-qwen35-native-fullnative-context-output-refresh235/summary.json`). A c4
  context-producer refresh keeps the native batch context signature red
  (`[137,137,137,118]`) even through temp-output and compact-cache diagnostics,
  while per-row context+gate and per-row context-only replay are both green
  (`[137]*4`). That eliminates KV cache addressing and context output-buffer
  aliasing for c4; target the native grouped context producer / pre-QKV setup
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c4-context-producer-refresh236/summary.json`). This is
  correctness-only evidence: rowchunk full attention remains
  `native_caware_decode=false`, and full-native no-chunk grouped>=3 attention is
  still the retained blocker
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c48-tail-gemv-output-fix/summary.json`,
  `benchmarks/results/2026-06-03-hipengine-qwen35-native-c48-default-output-batch/summary.json`). A full-native/no-chunk
  c4 isolation matrix now makes the first single-stage native-context blocker
  green: forcing full-attention context replay to per-row, either context+gate or
  context-only, restores c4 equality (`[137]*4`), while O-output-only,
  post-attention-only, gate-only, KV-append-only, and MoE-only variants stay red.
  The same context replay does not fully fix c8 (`[137,137,137,137,45,11,83,137]`),
  so c4 is isolated to native batched context and c8 has an additional row-count
  interaction (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c48-fullnative-context-isolation/summary.json`). A
  c4→c8 threshold sweep with the same full-native/no-chunk context-only replay
  shows the break starts exactly at the fifth active row: c4 is green, c5 is
  already `[137,137,137,137,45]`, then c6/c7/c8 extend the same tail pattern.
  Rows 0..3 remain stable after the c4 context blocker is bypassed; rows>=5 need
  a separate full-native row-count fix
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c45678-context-threshold/summary.json`). A c5 follow-up adds
  per-row MoE, post-attention, O projection, KV append, input, QKV, scratch, and
  input+QKV on top of context-only replay; all stay red on row 4. Explicit
  rowchunk2, however, restores c5/c6/c7 equality (`[137]*5`, `[137]*6`,
  `[137]*7`) with the rowchunk-GEMV metadata. The rows>=5 residual is therefore a
  coupled full-attention row-group/no-rowchunk interaction that rowchunk2 avoids,
  not a simple single-stage suffix or pre-QKV fallback
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c5-c7-rowchunk2-rescue/summary.json`). A current default-output
  rowchunk boundary sweep sharpens the cap: rowchunk1/2 are green for c5/c8;
  rowchunk3 is green while the non-leading segment has at most two rows (c4/c5)
  but red once the second segment has rows 3..5 (c6/c7/c8); rowchunk4/full-native
  is red as soon as a leading four-row group is formed (c4/c5/c8). Until the
  grouped full-attention producer is fixed, the safe diagnostic cap is therefore
  leading group <=3 and later groups <=2, which is exactly why runtime auto stays
  at rowchunk2 (`benchmarks/results/2026-06-03-hipengine-qwen35-native-rowchunk-boundary-c4c8/summary.json`). Runtime/retained-bench auto now promotes rowchunk2 to the
  contiguous c3..c8 range; with no explicit rowchunk flag, c5 and c7 are green
  and c6 is green across three repeats (`[137]*rows`) with rowchunk-GEMV metadata
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c567-auto-rowchunk2/summary.json`). A follow-up current-default
  gate keeps c2/c4/c8 green, but a derived c9 fixture is red for default/no-rowchunk
  and explicit rowchunk2/rowchunk1 controls, so c9 stays outside the auto cap
  until it gets a separate row-count fix
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-default-c9-cap/summary.json`). A c9 frontier split then shows the extra blocker is MoE rather
  than linear projection: grouped-compact MoE stays red under rowchunk2 with
  auto/native/batch-GEMV linear controls, while `selected_c1` MoE plus rowchunk2
  turns c9 green (`[137]*9`); full-native/no-rowchunk remains red even with
  `selected_c1` MoE, and rowchunk1 is still tail-red
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c9-selectedc1-moe-frontier/summary.json`). The same `selected_c1`
  MoE + rowchunk2 diagnostic also leaves the required c4/c8 gates green
  (`[137]*4`, `[137]*8`) while active c2 remains `[137,137]`, making it a
  correctness-only cross-row-count control rather than a retained/default
  promotion (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-selectedc1-moe-rowchunk2-control/summary.json`). A rowchunk2 MoE boundary sweep confirms
  grouped-compact MoE is green for c5/c6/c7/c8 and first fails at the derived c9
  row count; selected-c1 MoE restores c9 equality, so the post-c8 boundary is a
  grouped-compact MoE row-count issue layered on top of the rowchunk2 diagnostic
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c5c9-rowchunk2-moe-boundary/summary.json`). A current post-sampler-audit rerun corrects/supersedes the earlier
  derived-row rowchunk2-green assumption: original c4/c8 rowchunk2 with
  grouped-compact MoE still passes (`[137]*4`, `[137]*8`), but compact derived
  rows4..6/rows4..7 were red under rowchunk2 (`[45,137,137]` and
  `[45,137,137,137]`), and selected-c1 MoE did not repair that compact
  derived-row case (`[11,105,137]` / `[11,105,137,137]`)
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-rowchunk2-derived-rows-post-audit233/summary.json`). The
  follow-up output-projection fix shows the issue was the leading multi-row
  rowchunk's native O projection, not prompt content, slot compaction, or MoE:
  compact rows4..7 rowchunk1 was green, explicit rowchunk2 `batch_gemv` O-output
  was green, and after forcing `batch_gemv` for every multi-row rowchunk the
  compact c3/c4 derived-row gates are green (`[137]*3`, `[137]*4`) while original
  c4/c8 stay green (`[137]*4`, `[137]*8`)
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-rowchunk-output-gemv-leading-fix234/summary.json`).
  A focused c8 rowchunk4 full-attention audit keeps the accepted projection
  metadata but raises the native chunk size from 2 to 4; it is correctness-red
  (`[137,137,137,137,11,60,117,137]`), and per-row input/QKV/context/gate/output
  plus compact-cache/temp-output/scratch diagnostics do not restore full equality.
  This keeps the correctness cap at rowchunk2 and leaves native grouped
  full-attention for row groups of four as the active blocker
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c8-rowchunk4-projection-audit/summary.json`).
  A current rows4/5/6 c3 contrast keeps the full-native 512/128 run red
  (`[45,11,137]`) while rowchunk2 is green (`[137,137,137]`). The paired L40
  decode-step-0 trace shows native full attention first fails at layer 7
  `attn_input_pre_qkv`; a later trace-coverage fix showed rowchunk2 had not
  originally captured the pre-QKV/QKV callback stages there, so do not treat the
  older rowchunk2 pre-QKV-green reading as valid. The generated-token contrast
  still keeps the grouping>=3 fix target on upstream hidden row-group interaction
  / native grouped pre-QKV setup rather than sampler or late output
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-rowchunk-boundary-contrast/summary.json`, corrected by `benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-rowchunk-tracefix/summary.json`).
  The apparent duplicate `block_table_rows` in native c3 full-attention metadata
  are not the bug: `_batch_full_spans` encodes row-relative physical-slot offsets
  because kernels add the compact active-row base internally. Active c2 is green
  with the same expected duplicate rows, c3 native is red with formula-consistent
  rows, and rowchunk2's tail chunk correctly shifts to slot-2 offsets. Block-table
  construction is therefore eliminated as the c3 grouping>=3 suspect
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-block-table-audit/summary.json`).
  Context-side diagnostics are also not the fix: forcing native c3 full-attention
  context through compact-cache copies or through a temporary context output
  buffer leaves rows4/5/6 red at `[45,11,137]`, and paired L40 decode-step-0
  traces keep the first full-attention failure at layer 7 `attn_input_pre_qkv`
  (`max_abs=0.015625`). This eliminates context-cache copying/output-buffer
  aliasing and keeps the target upstream of context at hidden/pre-QKV grouped
  setup
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-context-diagnostics-red/summary.json`).
  The post-attention add/RMSNorm boundary is eliminated as well: forcing only
  native c3 full-attention post-attention through the per-row diagnostic path
  keeps rows4/5/6 red at `[45,11,137]`, and the paired L40 decode-step-0 trace
  still first fails at layer 7 `attn_input_pre_qkv` row 0 (`max_abs=0.015625`).
  The grouping>=3 target therefore remains upstream native grouped hidden /
  pre-QKV setup rather than context cache/output aliasing or post-attention
  residual/RMSNorm handoff
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-postattn-red/summary.json`).
  A layer-limit transition probe over L4-L8 localizes the first over-tolerance
  hidden transition for both native and rowchunk2 to layer limit 8 (the second
  full-attention layer, index 7); L4-L7 stay hidden/token green. After rowchunk
  trace coverage was fixed to capture post-input-RMSNorm and QKV callback stages,
  both native full and rowchunk2 first fail at `attn_input_pre_qkv`. Rowchunk2
  therefore keeps 512/128 generated-token equality green by changing later
  trajectory, not by making the L8 pre-QKV hidden trace parity-green
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-layer7-transition/summary.json`, corrected by `benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-rowchunk-tracefix/summary.json`).
  Using the retained-compatible `--c1-decode-path native_batch` oracle and the
  retained warmup, a trace over the first native token-divergence window
  separates the paths cleanly: full-native remains retained-red `[45,11,137]`
  and is hidden/token red at row 1 index 11, while rowchunk2 remains retained-green
  `[137,137,137]` and hidden/token green in the same L40 trace window. The first
  native traced full-attention mismatch at that window is already layer 7 `input`,
  so by the token flip the bad trajectory has propagated into the layer input;
  rowchunk2 fixes that trajectory by the divergence window even though early L8
  pre-QKV hidden drift can appear under narrower probes
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c3-token-window-c1native/summary.json`).
  Runtime auto now selects native full-attention row chunks of 2 for covered
  rows `{3,4,8}` when `HIPENGINE_QWEN35_BATCH_DECODE_FULL_ATTN_ROW_CHUNK_SIZE`
  is unset, while explicit env values still override and c=2 remains one native
  batch. Clean 512/128 equality is green for c3 rows4/5/6, c4, and c8 under
  the rowchunk2 path (`[137...]` in each case), but the path still records
  `native_caware_decode=false` and retained/performance blockers
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-rowchunk2-auto-c3c4c8/summary.json`).
  c2 scaling-baseline references are now present: c1 native and c2 serial-bridge
  artifacts load into the c2 retained bench, `scaling.complete=true`, and ratios
  are populated (`aggregate_vs_c1=0.9298`, `aggregate_vs_serial_bridge=1.1990`),
  but the row remains blocked on throughput-claim/profiler/rollup gates and makes
  no retained performance claim
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-baselines/summary.json`).
  A current c2 retained profiler rerun now captures `rocprofv3 --kernel-trace`
  evidence with grouped-compact MoE, explicit batched LM-head sampler evidence,
  generated-token equality `[137,137]`, complete scaling refs, and synthesized
  profiler duration fields from a compact kernel trace. This removes the stale
  profiler-not-captured, sampler-profiler-command, and selected-c1 MoE blockers
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-profiler-current/summary.json`).
  A follow-up c2 projection-dispatch artifact measures native-batch projection
  at `110.30` aggregate tok/s vs selected-c1/row-GEMV projection at `101.99`
  aggregate tok/s (`1.0814x`) and validates a `gemv_awq_selected_dual_pack8_strided`
  candidate with profiler evidence; `projection_dispatch.path` is now
  `benchmark_accepted_caware_projection`, batch blockers are empty, and
  generated-token equality remains `[137,137]`. The profiled retained row is
  still blocked, with no performance claim, because aggregate native c2 remains
  below the c1 baseline (`aggregate_vs_c1=0.8372`) and retained-bench promotion
  now blocks before schema validation when aggregate scaling does not beat c1
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-projection-dispatch/summary.json`).
  A c2 output-path audit then checked native output promotion under the same
  projection-dispatch evidence: default batch-GEMV outputs remain fastest and
  green (`112.45` tok/s, `[137,137]`), native full-attention O alone is green but
  slower (`109.27` tok/s), native linear output alone is correctness-red
  (`[82,137]`), and all-native linear/full output is green but slower than the
  serial bridge (`99.91` tok/s). No output default is promoted
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c2-output-path-audit/summary.json`).
  c4/c8 projection-dispatch evidence is now also green under the rowchunk2
  equality path: native projection beats selected-c1/row-GEMV by `1.2149x` at c4
  and `1.2927x` at c8, and runtime metadata selects
  `benchmark_accepted_caware_projection` with full generated-token equality for
  both; retained throughput remains blocked because rowchunk2 full attention is
  `native_caware_decode=false`
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c48-projection-dispatch/summary.json`).
  The c2 and c4/c8 projection evidence is also merged into one combined
  projection-dispatch catalog. Runtime validation with that single artifact keeps
  c2/c4/c8 generated-token equality green (`[137]` for every row) and selects
  row-bounded `gemv_awq_selected_dual_pack8_strided_c{2,4,8}` candidates for the
  matching row counts. This is a dispatch/catalog validation only: c2 still does
  not beat the c1 scaling baseline, and c4/c8 still rely on rowchunk2 full
  attention, so no retained c>N performance claim is made
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-c248-projection-dispatch-catalog/summary.json`).
  A follow-up full-attention group-size audit with that combined projection
  catalog confirms the threshold: c4/c8 rowchunk2 controls stay green, but
  rowchunk3 and full-native grouping reproduce the same failures (`c4` prefixes
  `[82,137,137,137]`; `c8` prefixes `[82,137,137,137,45,11,137,137]`). Projection
  dispatch is therefore eliminated from the c4/c8 full-native failure, and the
  next native attention target is the grouped pre-QKV/hidden interaction that
  appears when a full-attention group contains three or more rows
  (`benchmarks/results/2026-06-03-hipengine-qwen35-native-fullnative-projection-c4c8-audit/summary.json`).
  A focused c4 hidden-bisect over the first rowchunk3 token-divergence window
  keeps rowchunk2 hidden/token green vs the retained-compatible native-c1 oracle,
  while rowchunk3 is hidden/token red: row0's first token mismatch is generated
  sequence index 82, and the traced rowchunk3 full-attention failures are already
  present at decode step 80 (`layer7 attn_input_pre_qkv`, then later layer15
  input/output rows0..2). This pins the grouped>=3 issue to the full-attention
  trajectory before the token flip, not to projection dispatch or late sampler
  metadata
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-window/summary.json`).
  The earlier decode-step trace confirms the same rowchunk3 run is hidden-red
  before any token mismatch: rowchunk2 is hidden/token green for decode steps
  0..3, while rowchunk3 remains token-green but has first final-hidden mismatch
  at decode step 2 / generated index 3 / row0, with full-attention trace failures
  already starting at decode step 0 / layer7 `attn_input_pre_qkv` for rows0..2.
  The later row0 token flip is downstream of this early grouped>=3 hidden
  trajectory divergence
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-early/summary.json`).
  A c4 layer-limit transition probe adds that the grouped>=3 issue is visible
  before it escapes the layer-8 final hidden tolerance: rowchunk3 remains
  hidden/token green at layer limits 4 and 8, but it already has strict bit drift
  by layer limit 4 and over-tolerance full-attention trace failures at layer
  limit 8, starting at decode step 0 / generated index 1 / layer7
  `attn_input_pre_qkv` for rows0..2. Rowchunk2 has no corresponding stage
  failures
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-layer-transition/summary.json`).
  Adding the intermediate layer limits changes the c4 diagnosis again: rowchunk2
  remains generated-token green and has no full-attention stage failures through
  L4-L8, but its final hidden oracle is over tolerance at L5/L7 and recovers at
  L6/L8. Rowchunk3 inherits the L5 hidden drift, becomes token-red under the
  layer-limit-6 truncated oracle (row1 first mismatch at generated index 4), has
  rowchunk3-specific layer4 QKV/Z linear projection drift for rows0..2, and still
  first fails the L8 full-attention trace at layer7 `attn_input_pre_qkv`. This
  makes the grouped>=3 issue observable before the second full-attention layer in
  layer-limited trajectories while rowchunk2 remains the generated-token-green
  control
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-linear-bridge/summary.json`).
  Forcing the same L4-L8 bridge through selected-c1 linear projections does not
  change the rowchunk3 shape: rowchunk2 still has zero linear-projection drift
  and no full-attention stage failures, while rowchunk3 still has layer4 QKV/Z
  drift at L5, the same L6 truncated-token failure, and the same L8 layer7
  `attn_input_pre_qkv` failure. This eliminates the native batched linear
  projection kernel as the cause of that drift; the projection differences are
  inherited from the grouped>=3 layer3 output/hidden trajectory
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-selected-projection-bridge/summary.json`).
  Forcing full-attention layer-copy and post-attention add/RMSNorm to per-row
  diagnostics also does not change the shape: rowchunk2 stays generated-token
  green with zero linear-projection drift and no full-attention stage failures,
  while rowchunk3 keeps the L5 layer4 QKV/Z drift, L6 truncated-token failure,
  and L8 layer7 `attn_input_pre_qkv` failure. The grouped>=3 c4 drift is
  therefore not caused by the layer3 output-to-layer4 copy/RMSNorm handoff; the
  bad trajectory is already in the layer3 full-attention output values that feed
  the next linear layer
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-handoff/summary.json`).
  A producer split then tested whether the layer3 producer can be repaired by a
  single per-row subpath. Per-row context-only replay is not the rowchunk3 fix:
  rowchunk2 stays generated-token green with no full-attention stage failures,
  but rowchunk3 keeps the L5 layer4 QKV/Z drift, L6 truncated-token failure, and
  L8 layer7 `attn_input_pre_qkv`/`attn_context` failures. Forcing only the
  full-attention O projection to per-row is also not a safe fix: rowchunk3 loses
  the L6 token failure but remains hidden/stage-red, and the rowchunk2 control
  becomes token-red at L8. The remaining target is the coupled context/gate/O/MoE
  layer3 producer path rather than batch-GEMV O alone or native batch context
  arithmetic alone
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-producer-split/summary.json`).
  Forcing only the full-attention-layer MoE suffix to per-row selected-c1 also
  does not move rowchunk3: rowchunk2 remains the generated-token-green control
  with no projection drift or full-attention stage failures, while rowchunk3
  keeps the L5 layer4 QKV/Z drift, L6 truncated-token failure, and L8 layer7
  `attn_input_pre_qkv`/`attn_context` failures. Full-attention grouped-compact
  MoE alone is eliminated; the remaining target stays upstream/in the coupled
  context/gate/O producer behavior before MoE
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-moe/summary.json`).
  Forcing only the full-attention gate multiply to per-row also does not move
  rowchunk3: rowchunk2 remains the generated-token-green control with no
  projection drift or full-attention stage failures, while rowchunk3 keeps the L5
  layer4 QKV/Z drift, L6 truncated-token failure, and L8 layer7
  `attn_input_pre_qkv`/`attn_context` failures. Gate alone is eliminated; the
  remaining grouped>=3 target stays in upstream/coupled context/O behavior before
  or around gate rather than the gate multiply itself
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-gate/summary.json`).
  Forcing context+gate and O together to per-row is also not a safe retained fix:
  it removes the rowchunk3 L6 generated-token failure, but rowchunk3 remains
  hidden/stage-red with L5 layer4 QKV/Z drift and L8 layer7
  `attn_input_pre_qkv`/`attn_context` failures, while the rowchunk2 control
  becomes token-red at L8 and gains QKV/Z drift under the O-bearing suffix. The
  combined suffix is eliminated as a fix; the useful signal is that the
  rowchunk3 token failure can be moved by the O-bearing suffix, but not without
  poisoning rowchunk2 or fixing the inherited hidden trajectory
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-context-gate-o/summary.json`).
  Forcing only the full-attention KV append to per-row likewise does not change
  rowchunk3: rowchunk2 remains generated-token green with no projection drift or
  full-attention stage failures, while rowchunk3 keeps L5 layer4 QKV/Z drift,
  the L6 truncated-token failure, and L8 layer7 `attn_input_pre_qkv`/
  `attn_context` failures. KV append alone is eliminated; the grouped>=3 target
  stays after append in the context/output interaction or in the upstream
  hidden/grouping trajectory
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-kv-append/summary.json`).
  Forcing the full-attention context+gate branch to per-row without the O suffix
  also does not change rowchunk3: rowchunk2 remains generated-token green with no
  projection drift or full-attention stage failures, while rowchunk3 keeps L5
  layer4 QKV/Z drift, the L6 truncated-token failure, and L8 layer7
  `attn_input_pre_qkv`/`attn_context` failures. Context+gate alone is eliminated;
  the previously observed token movement requires the O-bearing suffix and is not
  a hidden-trajectory fix
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-context-gate/summary.json`).
  Interleaving per-row KV append with per-row context+gate also does not change
  rowchunk3: rowchunk2 remains generated-token green with no projection drift or
  full-attention stage failures, while rowchunk3 keeps L5 layer4 QKV/Z drift,
  the L6 truncated-token failure, and L8 layer7 `attn_input_pre_qkv`/
  `attn_context` failures. Append/context ordering is eliminated; the issue is
  not a phased append-then-context ordering artifact
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-perrow-append-context-order/summary.json`).
  Forcing native batch context to write into a fresh FP32 temp buffer before
  copying into the normal context scratch also does not change rowchunk3:
  rowchunk2 remains generated-token green with no projection drift or
  full-attention stage failures, while rowchunk3 keeps L5 layer4 QKV/Z drift,
  the L6 truncated-token failure, and L8 layer7 `attn_input_pre_qkv`/
  `attn_context` failures. Context output aliasing is eliminated; the issue is
  not caused by in-place/native batch context writes into the standard
  `query_raw` buffer
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-batch-temp-context/summary.json`).
  Forcing native batch context through compact copied per-row KV caches/block
  tables also does not change rowchunk3: rowchunk2 remains generated-token green
  with no projection drift or full-attention stage failures, while rowchunk3 keeps
  L5 layer4 QKV/Z drift, the L6 truncated-token failure, and L8 layer7
  `attn_input_pre_qkv`/`attn_context` failures. Rowchunk cache/block-table layout
  is eliminated; the issue is not caused by the original rowchunk cache/table
  layout seen by the batch context kernel
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-batch-compact-cache/summary.json`).
  The row-chunk diagnostic branch now forwards full-attention force flags and
  chunk-local per-row context/append tuples into each chunked runtime call; before
  that fix, rowchunk force-flag artifacts could record fallback metadata without
  actually exercising the fallback inside the chunk. The forwarded pre-QKV rerun
  confirms the same shape with trustworthy execution: forcing full-attention
  input RMSNorm and QKV prep/scratch to per-row does not change rowchunk3;
  rowchunk2 remains generated-token green with no projection drift and no
  full-attention stage failures or bit drift, while rowchunk3 keeps L5 layer4
  QKV/Z drift, the L6 generated-token failure, and L8 layer7
  `attn_input_pre_qkv`/`attn_context` failures. Pre-QKV setup remains eliminated;
  the grouped>=3 issue is downstream/inherited from the layer3 trajectory rather
  than unforwarded diagnostic metadata
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-preqkv/summary.json`).
  The forwarded context+gate rerun is also red: forcing only the full-attention
  context+gate branch to per-row poisons the rowchunk2 control at L6, leaves
  rowchunk3 token-red at L6, and gives both paths L5 layer4 QKV/Z drift. Unlike
  the whole-suffix rerun, context+gate alone does not create the huge layer3
  context mismatch; full-attention stage failures remain delayed until L8.
  Context+gate alone is eliminated as a retained-compatible hidden/token repair,
  and pre-forwarding context/gate evidence is superseded
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-context-gate/summary.json`).
  The forwarded context-only rerun is likewise unsafe: swapping only the context
  kernel to per-row with the batch gate poisons rowchunk2 at L6, leaves rowchunk3
  token-red at L6, and gives both paths L5 layer4 QKV/Z drift. It does not create
  the huge layer3 context mismatch from the whole-suffix rerun, but it is
  eliminated as a retained-compatible repair and as a trustworthy oracle for
  native batch context arithmetic
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-context-only/summary.json`).
  The forwarded context-staging rerun also preserves the same shape for both
  `batch_temp_output` and `batch_compact_cache`: rowchunk2 stays token-green with
  no projection drift or full-attention stage failures, while rowchunk3 still has
  L5 layer4 QKV/Z drift, the L6 token failure, and L8 layer7
  `attn_input_pre_qkv`/query/`attn_context`/`mlp_input` failures. Fresh FP32
  context-output staging and compact copied row-cache/block-table context
  execution are eliminated under trustworthy forwarded execution; the blocker is
  not simple context-output aliasing nor the original rowchunk cache/table layout
  seen by the batch context kernel
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-context-staging/summary.json`).
  The forwarded gate-only rerun keeps rowchunk2 token-green with no projection
  drift or full-attention stage failures, while rowchunk3 still has L5 layer4
  QKV/Z drift, the L6 token failure, and L8 layer7
  `attn_input_pre_qkv`/query/`attn_context`/`mlp_input` failures. Gate multiply
  alone is eliminated as the grouped>=3 cause and as a standalone gate mismatch
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-gate/summary.json`).
  The forwarded O-projection-only rerun keeps the rowchunk2 control token-green
  with no projection drift or full-attention stage/bit failures, but rowchunk3
  still has L5 layer4 QKV/Z drift, the L6 token failure, and L8 layer7
  `attn_input_pre_qkv`/query/`attn_context`/`mlp_input` failures. The layer3 O
  projection stage itself stays within tolerance, with only tiny under-tolerance
  bit drift on rowchunk3. O projection alone is eliminated as the grouped>=3
  cause; the remaining failure is inherited from the rowchunk3 layer3 trajectory
  into the next linear layer rather than a standalone O mismatch
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-o-proj/summary.json`).
  The forwarded context+gate+O rerun is also not a repair and no longer
  reproduces the pre-forwarding rowchunk3 token movement: forcing both the
  context/gate branch and O projection to per-row poisons rowchunk2 at L6, leaves
  rowchunk3 token-red at L6, gives both paths L5 layer4 QKV/Z drift, and leaves
  both paths with L8 layer7 `attn_input_pre_qkv`/query/`attn_context`/
  `mlp_input` failures. The O-bearing suffix is eliminated as a
  retained-compatible fix under trustworthy forwarded execution, and the earlier
  token-movement signal is superseded
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-context-gate-o/summary.json`).
  The forwarded layer-copy-only rerun also keeps rowchunk2 token-green with no
  projection drift or full-attention stage failures, while rowchunk3 still has L5
  layer4 QKV/Z drift, the L6 token failure, and L8 layer7
  `attn_input_pre_qkv`/query/`attn_context`/`mlp_input` failures. The output-to-
  next-layer copy/handoff is eliminated as the grouped>=3 cause and as a
  copy/aliasing bug
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-layer-copy/summary.json`).
  The forwarded full-attention-MoE-only rerun also keeps rowchunk2 token-green
  with no projection drift or full-attention stage failures, while rowchunk3
  still has L5 layer4 QKV/Z drift, the L6 token failure, and L8 layer7
  `attn_input_pre_qkv`/query/`attn_context`/`mlp_input` failures. Full-attention
  grouped-compact MoE alone is eliminated under trustworthy forwarded execution;
  the remaining failure is inherited before/around the context-output producer
  rather than from the MoE suffix
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-full-moe/summary.json`).
  The forwarded KV-append-only rerun has the same retained-compatible shape:
  rowchunk2 stays token-green with no projection drift or full-attention stage
  failures, while rowchunk3 still has L5 layer4 QKV/Z drift, the L6 token
  failure, and L8 layer7 `attn_input_pre_qkv`/query/`attn_context`/`mlp_input`
  failures. The layer3 full-attention trace stays within tolerance except for
  tiny under-tolerance bit drift. KV append/live-span mutation alone is
  eliminated as the grouped>=3 cause
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-kv-append/summary.json`).
  The forwarded post-attention-only rerun also keeps rowchunk2 token-green with
  no projection drift or full-attention stage failures, while rowchunk3 still has
  L5 layer4 QKV/Z drift, the L6 token failure, and L8 layer7
  `attn_input_pre_qkv`/query/`attn_context`/`mlp_input` failures. Post-attention
  add/RMSNorm alone is eliminated as the grouped>=3 cause and as a standalone
  residual/RMSNorm handoff bug
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-post-attn/summary.json`).
  The forwarded whole-suffix rerun is decisively red: forcing per-row KV append,
  context+gate, O, post-attention add/RMSNorm, MoE, and suffix interleaving makes
  rowchunk2 token-red at L6, leaves rowchunk3 token-red at L6, gives both paths L5
  layer4 QKV/Z drift, and is already over tolerance in layer3
  `attn_context`/residual/`mlp_input` (context max_abs about 9.03). Whole-suffix
  per-row interleaving is eliminated as a retained-compatible path and as a
  hidden-parity repair; the earlier pre-forwarding suffix token movement is
  superseded
  (`benchmarks/results/2026-06-03-hipengine-qwen35-hidden-c4-rowchunk-forwarded-suffix-interleaved/summary.json`).
  c=8 native A/B projection is green under the selected-QKV/Z diagnostic, while
  paged KV row setup and the segmented state update itself under selected
  projections remain lower on the list. C2.3 and retained/performance evidence
  remain the priority; C2.4/C2.5 generated-token equality is green only under
  the non-retained correctness fallback defaults.
- Long-context c>N still uses a per-row split-K fallback label; no long-context
  native c>N claim is allowed until the split-K reducer is row-aware.
- INT8 c>N parity, aggregate-vs-c1 scaling, graph replay buckets,
  residual-serial-loop removal, graph-replay profiler closure, and retained
  scoreboard promotion remain open performance/coverage work; native output
  promotion is not the current c2 scaling fix.

### Decode performance levers (2026-06-07 measured)

Native c>1 decode is far from GPU-efficiency limits and scales poorly: measured
medians (RX 7900 XTX, BF16-KV PARO, 512/128, batched_lm_head) are c1 133, c2 102
(0.77x c1, *worse* than c1), c4 138 (1.04x), c8 192 tok/s (1.44x). A
`rocprofv3 --kernel-trace` differential
(`benchmarks/results/2026-06-07-hipengine-qwen35-decode-dispatch-profile-484/`)
localized the cause to **dispatch count**, not just missing graph replay:

- Each decode step fires thousands of tiny kernels: **917/step at c1, 5370 at c4,
  9362 at c8** (c8 = 10.2x c1, worse than the 8x row count). Per-dispatch
  MEC/SPI overhead on kernels this short dominates the wall.
- **Graph replay is already on at c=1 by default** (clean delta: 134 tok/s with
  vs 110 eager = **+21.6%** at 512 context) yet c1 decode is still <=~37%
  GPU-utilized. So graph replay removes per-dispatch *host* latency / per-step
  host overhead but cannot remove per-dispatch *device* overhead.
- **c>1 decode graph replay is not wired** (`graph_bucket_stats.replay_kernel_hits=0`):
  the batch step is not device-resident (per-step host token list + host
  positions + per-layer host->device block-table copy in `_batch_full_spans`),
  so it cannot be captured/replayed as-is.

Levers, in priority:

1. **Wire c-aware (c>1) decode graph replay** — make the batch step
   device-resident (device token feedback via `batch_lm_out_index`, on-device
   position/context advance, persistent device block tables/spans), then
   capture/replay like the c=1 path. Bounded refactor; recovers the eager
   penalty (>=~20%, likely more at c>1 where 9362 dispatches/step overwhelm
   eager host issue).
2. **Reduce decode dispatch count via kernel fusion** — 917/step at c1 is high;
   fewer launches help c=1 and c>1 beyond graph replay.
3. **Output-column-tiled c>1 GEMM — its own big lift.** A separate kernel-family
   effort that loads each weight tile once and reuses it across all `c` columns
   (caching the quantized activation once), plus per-expert token batching for
   MoE. This both cuts dispatch count and amortizes weight loads, and is the
   only lever toward the c>1 roofline (`docs/ROOFLINE.md` §3.2). Track as a
   large standalone workstream, **not** part of the graph-replay change.

## Readiness matrix

| Layer | Current status | Evidence / code | Blocks retained c>N |
| --- | --- | --- | --- |
| OpenAI server | Non-streaming compatible requests still coalesce through `_GenerationBatcher`; streaming and non-streaming now share request accounting, `n>1` lowers to multiple choices with request ids, and `/metrics` is available behind `--metrics prometheus` / `HIPENGINE_METRICS=prometheus`. | `hipengine/server/api.py:_GenerationBatcher`, `_choice_request_id`, `_row_seeds_for_request`, `_render_prometheus_metrics`; `pytest -q tests/test_server_api.py -q`. | Coalescer can be demoted once native c>N equality/perf is green; no retained throughput claim comes from HTTP coalescing alone. |
| Public `LLM.generate()` / loop adapter | The public generator can be wrapped by `SubmitPollTextGenerator`, preserving outputs while exercising submit/poll semantics in tests. | `hipengine/generation/engine_loop.py:SubmitPollTextGenerator`; `pytest -q tests/test_generation_batch_scheduler.py -q`. | Native Qwen/PARO c>N decode equality and retained benchmark evidence. |
| Engine loop / scheduler | `ResidentEngineLoop` and `ResidentBatchScheduler` own pending/admitted queues, slots, active masks, compact prefill slabs, decode work, graph bucket keys, completion routing, and unified reclaim. | `hipengine/generation/engine_loop.py:ResidentEngineLoop`; `hipengine/generation/batch_scheduler.py`; scheduler tests. | Runtime equality/perf gates, not host-loop shape. |
| Prefill | BF16 compact/native prompt-list prefill is live; scheduler tests cover chunk/policy plumbing. INT8 retained c>N prefill remains blocked. | `prefill_native_packed`, `CompactPromptSlab`, `scripts/qwen35_batch_packed_prefill_correctness.py`; `tests/test_generation_batch_scheduler.py`. | INT8 c>N parity and retained end-to-end equality. |
| Decode runtime | Safe/diagnostic paths remain non-claiming: serial bridge rows and experimental native rows are blocked/rejected unless generated-token equality and native execution metadata pass. c=2/c=4/c=8 512/128 generated equality is green with the no-selected auto `batch` projection path (128-thread batch-GEMV QKV/Z under native metadata), native segmented linear state, batch-GEMV/Marlin linear output, grouped-compact MoE using compact-row selected GEMV for c<=8 decode, c=2 native full-attention as one batch, and c=3/c=4/c=8 native full-attention rowchunk2 plus row-aware batch-GEMV full-attention output; retained-bench artifacts now drop the stale generated-equality and BF16/context<1024 full-attention support blockers only after embedding passing equality/native-context evidence. | `step_batch_serial`, `step_batch_native`, `_sample_batch_from_hidden`, `batch_execution_metadata`; retained/hidden-bisect artifacts cited in C2. | Projection-dispatch/retained evidence and retained benchmark/profiler evidence. |
| Sampler | `PerRowSamplingParams` and sampler blocks exist; native `batched_lm_head` now uses a row-aware `hipengine_batch_argmax_f32` launch and is evidence-gated by generated-token equality plus sampler provenance. The retained bench's no-flag c=2/4/8 diagnostic path now auto-attaches the repo-retained sampler equality artifacts so primary equality probes exercise the row-aware sampler instead of the older serial LM-head loop. | `hipengine/generation/batch_scheduler.py:PerRowSamplingParams`; `hipengine.dispatch.sampling`; `hipengine/kernels/hip_gfx1100/linear/lm_head.*`; `scripts/qwen35_batch_retained_bench.py`; `tests/test_lm_head_plan.py`; `tests/test_generation_batch_scheduler.py`. | Retained native throughput is still blocked by projection/dispatch/profiler/graph-replay gates, not by the sampler launch itself. |
| Attention / KV primitives | BF16 batched paged KV append and batched full-attention context decode pass c=1/2/4/8 primitive correctness. Split-K long-context decode is labeled per-row fallback. | `scripts/qwen35_batch_correctness.py`; `/tmp/hipengine-multiloop-c{2,4,8}-correctness.json`; attention dispatch tests. | Row-aware split-K reducer; INT8 end-to-end gate. |
| MoE / quant kernels | The c<=8 decode path now uses grouped-compact MoE without per-row selected-c1 replay: compact sorted expert rows run through selected AWQ GEMV for gate/up/down, the shared expert uses row-aware GEMV, and `lane_to_row` is cleared before rebuilding the inverse combine map. c=2/c=4/c=8 generated-token equality is green with all-grouped MoE, the c=2 layer-limit-40 hidden oracle is hidden/token green vs independent c=1 native-batch, and the retained-bench/hidden-bisect defaults now use grouped MoE for global/linear/full-attention subpaths. | `hipengine/runtime/qwen35_paro.py`; `tests/test_qwen35_decode_state.py`; `scripts/qwen35_batch_retained_bench.py`; `scripts/qwen35_batch_hidden_bisect.py`; `benchmarks/results/2026-06-02-hipengine-qwen35-native-grouped-moe-default-matrix/summary.json`. | Retained benchmark/profiler/scaling evidence; larger prefill/grouped WMMA throughput remains separate from the c<=8 decode correctness gate. |
| KV pool | Chunked grow/shrink, append-only block ids, current admission capacity, prefix refcounts, and copy-on-write forks are implemented in host tests. | `hipengine/kvcache/pool.py:ChunkedKVPool`, `admit_with_shared_prefix`, `fork_copy_on_write`; `pytest -q tests/test_kvcache_policy.py -q`. | Device/runtime retained equality and perf, not the host allocator contract. |
| Prefix / radix cache | `RadixCache` indexes block-aligned token prefixes; server exposes prefix-cache mode and `n>1` lowering uses distinct row seeds/request ids. | `hipengine/kvcache/radix.py:RadixCache`; `hipengine/server/api.py`; kvcache/server tests. | Broader retained coverage and future DMS/KVTC policy work; no flat prefix-LRU peer path. |
| Observability | Completion artifacts and `/metrics` include request/pool counters; graph-bucket stats exist for scheduler observability, and the c=2/c=4/c=8 profiler preflights are captured as compact JSON rather than retaining raw rocprof dumps. | `CompletedRequest.to_json_dict`, `KVPoolStats.to_json_dict`, `GraphBucketCache`, `_render_prometheus_metrics`; server/scheduler tests; `benchmarks/results/2026-06-02-hipengine-qwen35-native-c{2,4,8}-profiler-preflight/profiler-c{2,4,8}.json`. | Accepted retained rows still need graph-replay profiler evidence and accepted benchmark rollup promotion. |

DMS / compact KV serving status lives in [`KVCACHE.md`](KVCACHE.md) and is not
mirrored in this matrix.

## Engine-loop contract

The engine loop is the single owner of admission, work scheduling, KV
allocation, sampling, completion, reclaim, and pool resize in the host-side
scheduler contract. The FastAPI adapter now keeps only a short `session_lock`
in `hipengine/server/api.py` for model/session preparation; request-lifetime
`engine.generate(...)` calls are serialized by the batcher worker rather than a
coarse server lock, which is an adapter safety rail and not evidence that the C4
loop scaffolding is absent.

### C1 lock-scope audit

Current server lock scope after the C4/C5 host work:

- Startup eager-load holds `session_lock` only around LLM construction,
  resident-session preparation, context-budget validation, and capacity logging;
  the warmup `engine.generate(...)` call runs after the lock is released.
- Non-streaming requests call `generate(...)`, which holds `session_lock` only
  for lazy LLM construction, resident-context preparation, sampling
  construction, optional `n>1` row-seed lowering, and context-budget validation,
  then enqueue into `_GenerationBatcher`.
- `_GenerationBatcher._run_group(...)` no longer accepts or holds a generation
  lock. It owns an event-loop queue/worker and calls
  `engine.generate(tuple(prompts), sampling)` outside `session_lock`, so no
  request lifetime is covered by a server lock.
- Streaming chat holds `session_lock` for preparation only, then routes through
  `_GenerationBatcher.stream(...)`. The batcher owns a per-request queue, so
  streaming no longer directly bypasses the batcher through `engine.stream(...)`.

The remaining native-throughput blocker is correctness/performance, not host API
shape: server endpoints can be thinned further only after the resident path has
native c>N generated-token equality, retained execution metadata, and accepted
benchmark evidence. Until then, the batcher worker serializes grouped calls to
avoid concurrent mutation of shared KV, linear-attention recurrent state, hidden
buffers, scratch, and sampler state without holding a request-lifetime server
lock.

### Public interface (target)

```python
request_id = engine.submit(
    prompt_tokens: Sequence[int],
    sampling: SamplingParams,
    max_new_tokens: int,
    stream: bool = False,
) -> int

events = engine.poll(timeout: float | None = None) -> list[Event]
# Event(request_id, kind: 'token' | 'finish' | 'error', payload)

ok = engine.cancel(request_id: int) -> bool
```

`LLM.generate()` and the OpenAI server become thin adapters over this surface.
Both streaming and non-streaming traffic call the same `submit/poll/cancel`.

### Work classes

Each engine tick picks **one** of the following work classes for the next
kernel-launch sequence:

| Class | Action | Commit at end |
| --- | --- | --- |
| `ADMIT` | Move pending requests into active slots up to current pool capacity. Try one pool grow per cycle if grow-on-admission is enabled. | New slot table |
| `PREFILL_CHUNK` | Run one chunked prefill step over one or more admitted requests. | Per-request prompt cursor; KV append |
| `DECODE_STEP` | One token of decode for every active request whose prefill is done. | Per-request token; KV append |
| `RECLAIM` | Free KV pages, refcounts, scratch from finished/cancelled requests. | Free list; pool shrink eligibility |
| `VERIFY_STEP` *(SpecDec, later)* | One target-verify pass over draft rows. | Accept-list; transactional KV commit |
| `PACK_STEP` *(DMS, later)* | One streaming-pack sweep over a finished prefill layer/chunk. | Compact KV append; dense scratch release |

Default per-tick policy: `RECLAIM` → `ADMIT` → choose between `PREFILL_CHUNK`
and `DECODE_STEP` under the **prefill-vs-decode policy** (see below). Verify
and pack classes are inserted by SpecDec / DMS feature branches without
changing the loop contract.

### Commit points

KV mutation, generated-token delivery, streaming event emission, and
cancellation are honored **only at commit points**. A commit point is the
boundary between two work-class steps. Mid-step mutations are scratch.

This is what protects KV writes from being torn by mid-step cancellations,
gives SpecDec a clean accept/rollback gate, and lets DMS pack between active
requests' decode steps without races.

### Prefill-vs-decode policy

| Policy | Behavior | Default |
| --- | --- | --- |
| `protect_decode` | Decode always wins when any active request can decode. Prefill chunks fill remaining cycles up to a token budget. | yes |
| `protect_ttft` | Prefill wins for any newly admitted request until its first decode token. | — |
| `fair` | Round-robin between prefill and decode. Token-equivalent budgets are a later latency/metrics refinement. | — |

Knobs: `HIPENGINE_PREFILL_DECODE_POLICY` / `--prefill-decode-policy`
(default `protect_decode`, vLLM-equivalent default that minimizes
inter-token-latency regressions for active requests),
`HIPENGINE_MAX_ACTIVE_REQUESTS` / `--max-active-requests`,
`HIPENGINE_MAX_PENDING_REQUESTS` / `--max-pending-requests`, and
`HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` / `--max-prefill-chunk-tokens`.

## Dynamic KV pool

Continuous batching's admission policy is a function of current KV capacity.
A fixed startup-sized pool either wastes VRAM that could hold extra slots or
caps `C` for no reason. The pool must size against actual load.

### Allocator contract

- **Block id is permanent.** Once a block id `b` is allocated, its backing
  device pointer never changes. `KVLiveSpans` and captured `hipGraph` buckets
  that reference `b` stay valid until `b` is freed.
- **Growth is append-only.** New block ids come from chunks allocated past the
  current high-water mark. Existing live blocks are never relocated.
- **Shrink frees from the free list only.** A block is freeable iff its
  refcount is zero *and* no captured graph bucket has recorded a pointer for
  it. Shrink trims tail chunks; the high-water mark is monotonic during steady
  state.
- **Allocation granularity is a chunk.** `hipMalloc` happens in chunks of
  `kv_pool_chunk_pages` (default 128 pages or ≥ 64 MiB equivalent, whichever
  is larger), then sub-allocated into block ids. Avoids `hipMalloc` storms
  under bursty admission.
- **All allocation goes through the scheduler.** No path in dispatch / model /
  kernel code allocates KV pages directly. Admission is the only producer.

### Sizing policy

| Knob | Default | Notes |
| --- | --- | --- |
| `kv_pool_initial_bytes` | auto = v0.2.2 startup estimate | First chunk allocation. |
| `kv_pool_low_water_bytes` | `kv_pool_initial_bytes` | Pool never shrinks below this. |
| `kv_pool_high_water_bytes` | `min(free_after_weights * 0.9, kv_pool_initial_bytes * 4)` | Pool never grows above this. |
| `kv_pool_chunk_pages` | 128 (or ≥ 64 MiB equivalent) | Grow granularity. |
| `kv_pool_idle_grace_seconds` | 60 | Time below `low_water + chunk` before shrinking. |
| `kv_pool_grow_on_admission` | true | If false, admission rejects when the pool is full instead of trying to grow. |

CLI: `--kv-pool-{initial,low-water,high-water,chunk-pages,idle-grace}-*`.
Env: `HIPENGINE_KV_POOL_*`. Document in `docs/ENVS.md`.

### Admission rule (every cycle)

1. If the request fits in free pages → admit.
2. Else if `kv_pool_grow_on_admission` and `current_bytes + chunk_bytes ≤
   high_water_bytes` and `hipMemGetInfo()` permits → grow one chunk; admit.
3. Else queue with an explicit `admission_blocked_reason`
   (`kv_capacity_high_water_reached` / `device_oom` / etc.).

### Shrink rule (background, between scheduler ticks)

1. If `free_bytes > low_water_bytes + chunk_bytes` continuously for
   `idle_grace_seconds` → free one tail chunk.
2. Never free a chunk containing a non-zero-refcount block, regardless of idle
   time (protects refcounted prefix pages).

### Admission accounting

- `KVPolicy.admission_cap()` returns **current** free-page equivalents, not
  startup capacity. Dense fixed-page policy returns `free_pages`; DMS (later)
  returns compact-live-token capacity over current free pages.
- The pending queue carries a `kv_pages_needed_estimate` per request, computed
  from `prompt_tokens + max_new_tokens` at submit time and revised as actual
  decode positions advance.
- Admission decisions run after the current step's `RECLAIM` so that finishing
  requests free pages before the next admit attempt.

### Acceptance for a dynamic-pool-enabled c>N row

In addition to the existing benchmark gates:

- Pool grew and shrank on a designed burst+idle workload (artifact records
  ≥1 `grow_event` and ≥1 `shrink_event`), or the run fit in the initial chunk
  and the artifact says so explicitly.
- `kv_pool_grow_events ≤ ceil((peak_bytes - initial_bytes) / chunk_bytes)`
  (no `hipMalloc` storms).
- Debug check: no block-id pointer changed during the run.
- Memory audit: tracked allocator peak ≤
  `kv_pool_high_water_bytes + non_kv_baseline_bytes`.

## KV sharing: RadixCache (+ KVTC forward-compat)

Refcounted block ids unlock prefix sharing across requests. Prefix sharing is
the first non-trivial reduction of KV bytes per active request and a
prerequisite for cheap `n>1` lowering. The structure is RadixCache; flat
block-LRU is explicitly not implemented as a peer (rationale below).
Multi-tier KV storage (KVTC) is a follow-on feature branch; CONCURRENCY work
must honor the KVTC ABI guardrails so that branch lands cleanly later.

### Refcount semantics

- Every block id carries a refcount; default 1 when first written by a
  request.
- A second request that walks the same prompt prefix into an existing
  refcounted block increments the refcount and reuses the block id.
- A request finishing (`RECLAIM`) decrements refcounts on its block ids.
- A block is freeable when refcount reaches zero.
- A captured graph bucket holding a pointer for a block keeps the *chunk*
  alive against shrink (but not against free).

### Copy-on-write fork

- Two requests share a block until one of them writes a token that diverges
  from the other's path.
- At divergence, the diverging request gets a fresh block id (allocated under
  the same admission rule as any new write), copies the shared prefix's last
  partial block if needed, and continues independently.
- The original shared block stays refcounted on the non-diverging path.

### Why radix and not flat block-hash LRU

The primary target workloads — agentic loops, multi-turn chat, `n>1`
sampling — are all tree-structured: branches off a common root, where flat
block-hash LRU only catches one path at a time. RadixCache catches sharing
at every branch point, including partial-block edges. The ~200 LoC delta
over a flat structure (per [`PLAN.md`](PLAN.md)) is well-spent for this
workload mix; carrying two prefix schemes also doubles the surface area
where prefix sharing × dynamic pool × cancellation can interact badly, so
flat prefix-LRU is not implemented as a peer.

### Prefix index

Knob: `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}`.
Default `off` until correctness gates pass; then `radix`. Pick `off` to
disable prefix reuse entirely.

### Tiered storage (KVTC, future feature branch)

KVTC (KV tiered cache) is the planned multi-tier storage layer that sits
under prefix sharing: hot pages stay in HBM, cold but refcounted prefix
pages spill to pinned host RAM, and very cold session state spills to
NVMe / disk. KVTC is **not** in CONCURRENCY scope; it is called out here
so that the contracts in C2 / C4 / C5 do not preclude it. The reference
designs are vLLM v0.6+ CPU offload and SGLang hierarchical cache.

Rough tier roadmap (sketch, not committed in this doc):

| Tier | Storage | Latency | Use |
| --- | --- | --- | --- |
| T0 | Device HBM | ns | Active live KV; hot prefix nodes. |
| T1 | Pinned host RAM | µs (PCIe DMA) | Refcounted but cold prefix pages; admission-eligible without recompute. |
| T2 | NVMe / disk | ms | Session save/restore; very cold long prefixes. |

ABI requirements that CONCURRENCY work must already honor for KVTC to
land cleanly later are listed in
§[Forward-compatibility guardrails](#forward-compatibility-guardrails)
under "Don't break KVTC."

### `n>1` lowering

- The API layer accepts `n > 1` by submitting N scheduler requests with the
  same prompt tokens and a per-call seed offset.
- The prefix cache shares prompt KV across the N requests until the first
  divergent sampled token (immediate, for distinct seeds).
- Output is collected via N `request_id`s and returned to the client under the
  OpenAI `n` schema.
- This is the first user of prefix sharing in production and the natural
  staging ground for the contract.

### What's deliberately deferred

- **Eviction under variable-span KV (DMS).** Per-sequence eviction overlays
  for shared prefix blocks are an open design point; until then DMS disables
  prefix sharing (see [`KVCACHE.md`](KVCACHE.md) Phase K2).
- **Disk session save/restore.** Possible follow-on; ABI-compatible with the
  block-id-stable contract.

## Streaming, cancellation, and reclaim

### Per-request output queue

- Each active request owns a bounded token queue (default 64 tokens).
- The streaming adapter (SSE for `/v1/chat/completions` and
  `/v1/completions`) pulls from the queue and emits OpenAI-format events.
- When the queue is full (slow client), the request's slot is paused at the
  next commit point. It does not block other requests' decode steps.
- Knob: `HIPENGINE_STREAM_QUEUE_DEPTH` / `--stream-queue-depth`.

### Cancellation paths

| Trigger | Effect |
| --- | --- |
| `engine.cancel(request_id)` | Marked at next commit; slot is reclaimed. |
| Client disconnect (SSE) | Same as `cancel`. |
| EOS token sampled | Same as `cancel` with `finish_reason="stop"`. |
| `max_new_tokens` reached | Same as `cancel` with `finish_reason="length"`. |
| Per-request timeout (optional) | Same as `cancel` with `finish_reason="timeout"`. |

All five funnel through the same `RECLAIM` work class. There is one reclaim
implementation, not five.

### In-flight semantics

- Cancel during prefill: drop at the next chunk boundary.
- Cancel during decode: drop at the next step boundary.
- Cancel during verify (SpecDec, later): discard the verify journal; no
  canonical KV mutation.
- Cancel during pack (DMS, later): finish the in-flight pack; drop at its
  natural boundary.

Mid-step cancellation is never honored. This is what keeps KV mutation atomic
and what lets graph capture buckets stay valid across cancels.

## Per-row sampler and `n>1`

The coalescer's "compatible sampling key" requirement is a current-runtime
limitation, not a target architecture. Continuous batching needs the sampler
to accept per-row parameters in one kernel launch.

- Logits computed per row in one `w8a16_linear_bf16_f32_out` launch (current
  code path; already row-shaped).
- Sampling reads a **per-row params block** instead of scalar params:
  - `temperature[C]`
  - `top_k[C]` (or `0` = greedy)
  - `top_p[C]` (or `1.0` = no top-p)
  - `repetition_penalty[C]`
  - `seed[C]`
  - `stop_token_id[C][K_STOP_MAX]`
- Per-row EOS handling: the sampler emits a `done` flag per row when a stop
  token matches; the scheduler reclaims that row at the next commit.
- The submission-time coalescer (`_GenerationBatcher`) becomes redundant once
  the engine loop is live; remove it or keep it as a cold-path latency
  optimization for empty-pool startup bursts only.

`n>1` then lowers naturally: N submissions of the same prompt with distinct
seeds, shared prefix until the first divergent token.

## Observability contract

### Per-request fields (recorded in completion event and `/metrics`)

| Field | Meaning |
| --- | --- |
| `queue_seconds` | Time between `submit` and first `ADMIT`. |
| `prefill_seconds` | Wall time spent in `PREFILL_CHUNK` ticks owned by this request. |
| `decode_seconds` | Wall time spent in `DECODE_STEP` ticks where this row is active. |
| `tokens_generated` | Sampled tokens (excluding seed). |
| `kv_pages_owned` | Pages currently refcounted to this request at finish. |
| `kv_pages_peak` | Peak pages referenced by this request during its lifetime. |
| `kv_pool_bytes_at_admit` | Pool size when the request was admitted. |
| `bucket_key` | Decode graph bucket the request ran under most. |
| `admission_blocked_reason` | If queued; one of `kv_capacity_high_water_reached`, `pending_queue_full`, `device_oom`, …. |
| `finish_reason` | `stop`, `length`, `cancel`, `timeout`, `error`. |

### Per-pool counters

| Field | Meaning |
| --- | --- |
| `kv_pool_current_bytes` | Allocator-visible KV pool size right now. |
| `kv_pool_high_water_observed` | Largest size the pool has reached. |
| `kv_pool_grow_events` | Successful chunk allocations. |
| `kv_pool_grow_failures` | Allocations that hit `device_oom` or `high_water`. |
| `kv_pool_shrink_events` | Tail-chunk frees. |
| `kv_pool_free_pages` | Free pages right now. |
| `kv_pool_refcounted_pages` | Pages whose refcount > 1 (prefix sharing). |

### Per-bucket counters

| Field | Meaning |
| --- | --- |
| `graph_bucket_entries` | Distinct keys currently captured. |
| `graph_bucket_hits` | Replays since last reset. |
| `graph_bucket_misses` | Uncaptured fallbacks. |
| `graph_bucket_miss_reason` | `new_shape`, `chunk_added`, `mask_changed`, …. |
| `step_kernel_seconds` | Histogram of kernel-wall time per step. |

### `/metrics` endpoint

- Prometheus text format, when running `hipengine serve`.
- Knob: `HIPENGINE_METRICS` / `--metrics` in `{off, prometheus}`. Default `off`
  until C5.
- Per-request fields are exposed as histograms; per-pool / per-bucket as
  gauges and counters.

## Quant / model coverage matrix under c>N

The four-axis registry means every `(model, quant, KV dtype)` triple needs its
own c>N validation. This matrix tracks coverage; rows without a green retained
cell at c=2/4/8 are not c>N-eligible regardless of the engine-loop work.

| (model, quant, KV) | c=1 long | c=2 512/128 | c=4 512/128 | c=8 512/128 |
| --- | --- | --- | --- | --- |
| Qwen3.5/PARO × w4_paro × BF16 | retained | eq_ok *(blocked fallback)* | eq_ok *(blocked fallback)* | eq_ok *(blocked fallback)* |
| Qwen3.5/PARO × w4_paro × INT8/per-token-head | retained (capacity) | not_started | not_started | not_started |
| GGUF × Q4_K × BF16 | retained | not_started | not_started | not_started |
| GGUF × Q5_K × BF16 | retained | not_started | not_started | not_started |
| GGUF × Q6_K × BF16 | retained | not_started | not_started | not_started |
| GGUF × Q8_0 × BF16 | retained | not_started | not_started | not_started |
| W8A16 dense × BF16 | partial | not_started | not_started | not_started |

Status legend: `not_started`, `primitive_ok` (kernel correctness only),
`eq_ok` (generated-token equality vs c=1, blocked on protocol shape or
correctness fallbacks), `retained` (accepted retained row),
`rejected_correctness` (equality failed).

GGUF c>N coverage is required for the repo's namesake quant path. It can
follow the Qwen3.5/PARO equality template once the engine loop and per-row
sampler are live.

## Benchmark eligibility gates

A c>N row is not eligible for `accepted` status until all of these pass:

1. `scripts/qwen35_batch_correctness.py --rows N` passes for the exact
   primitive families used by the runner: `append_key_mismatch=0`,
   `append_value_mismatch=0`, `attn_batch_vs_c1_max_abs=0.0`, and
   `0.0 <= attn_batch_vs_numpy_max_abs <= 2e-5`.
2. The resident batch runner emits generated-token IDs equal to N independent
   c=1 resident runs for the same fixed prompts with greedy sampling and
   SpecDec disabled.
3. The artifact records scheduler occupancy, active mask shape, graph bucket
   key, KV policy, packed-prefill status, compaction events, and whether any
   serial bridge remains.
4. Continuous-batching rows include admission/completion timestamps and
   per-request p50/p95 latency in addition to aggregate tok/s.
5. Performance summaries show both aggregate tok/s and per-request tok/s.
   Never compare c=N aggregate to c=1 without also showing aggregate/c1 and
   per-request/c1 ratios.
6. **(dynamic pool)** Pool grew and shrank on a burst+idle workload, or the
   run fit in the initial chunk and the artifact says so. `grow_events ≤
   ceil((peak_bytes - initial_bytes) / chunk_bytes)`.
7. **(stable block id)** Debug check asserts no block-id pointer changed
   during the run.
8. **(prefix sharing, when enabled)** Shared-prefix workload artifact shows
   KV-byte savings *and* per-request TTFT drop vs the same workload with
   prefix sharing off.

## Bite-sized implementation queue

The phase ladder below is the source of truth for C0..C5. This queue expands
those phase items into implementation-sized packets for a multiloop or a human
coder. A good packet is one logical commit with a narrow test/bench gate and a
WORKLOG entry when it changes runtime behavior, correctness, or performance.
Do not check a packet merely because code exists; check it only when the
listed acceptance gate has passed and the parent phase item can cite it.

Recommended order: finish the C2 correctness packets before C3/C4/C5 feature
work, because continuous batching and KV sharing only matter once native c>N
emits the same tokens as independent c=1. For multiloop progress, count open
or partial checkboxes in this queue only; the phase ladder below stays as the
roll-up/status view.

### C0 packets — make diagnostics durable

- [x] **C0.1 c-sweep CLI.** Add `hipengine bench c-sweep` (or an equivalent
      `scripts/qwen35_batch_c_sweep.py`) that runs c=1/2/4/8 primitive,
      serial-bridge, and native-diagnostic commands from one config without
      copy/paste loops. Acceptance: JSON summary records every command,
      status, artifact path, and dirty git state. Evidence:
      `hipengine bench c-sweep --dry-run ...` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C0.2 artifact schema guard.** Add a CPU test/helper that rejects c>N
      diagnostic JSON missing `workload.native_compact_prefill`,
      `execution.batch_execution.native_compact_prefill`,
      `native_caware_decode` as an execution flag, a correctness/status field,
      and `throughput_claim_eligible`. Acceptance: failing fixture proves the
      guard catches a missing field. Evidence: `scripts/qwen35_batch_artifact_schema.py`
      plus `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C0.3 promote current diagnostics.** Move or regenerate the current
      c=2 accepted/rejected diagnostic JSONs under `benchmarks/results/` with
      `status=blocked` or `rejected_correctness` as appropriate. Acceptance:
      `WORKLOG.md` links exact commands and no scoreboard row is added unless
      `status=accepted`. Evidence:
      `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-32-blocked.json`
      and `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-128-rejected-correctness.json`.

### C1 packets — keep current integration safe

- [x] **C1.1 lock scope audit.** Trace the server/generator mutation paths
      formerly protected by `generation_lock`; document which session state is
      still non-reentrant. Acceptance: a focused test or review note proves the
      lock is narrow enough for C1 and names the exact blocker for native
      request-level concurrency. Evidence: §C1 lock-scope audit plus
      `hipengine/server/api.py` code refs and `pytest -q tests/test_server_api.py -q`.
- [x] **C1.2 API rejection contract.** Keep `n>1` rejected until C5 and add
      regression coverage if missing for completions and chat. Acceptance:
      server tests prove `n>1` returns the intended 4xx while prompt-list
      batching still works. Evidence: `pytest -q tests/test_server_api.py -q`.

### C2 packets — native BF16 c>N correctness first

- [x] **C2.1 remove compatibility shim.** Remove the generator
      `batch_execution_metadata(...)` `TypeError` compatibility path once all
      call sites use the settled signature. Acceptance: targeted generator and
      resident-layout tests pass. Evidence: commit removing the shim plus
      `pytest -q tests/test_generation_qwen35_paro.py tests/test_qwen35_resident_batch_layout.py -q`.
- [x] **C2.2 hidden-state bisection harness.** Add a HIP-guarded diagnostic
      that compares c=2 native vs independent c=1 hidden tensors after each
      layer and optionally after sub-stages (attention, selected MoE, shared
      expert, combine, LM head). Acceptance: the harness can reproduce the
      current L40 c=2 512/128 divergence earlier than generated-token idx 87.
      Evidence: `scripts/qwen35_batch_hidden_bisect.py` plus
      `python3 scripts/qwen35_batch_hidden_bisect.py --fixture /tmp/hipengine-prebench/fixtures/qwen36_paro_8x512_prompt_ids.json --prompt-length 512 --batch-size 2 --decode-tokens 16 --max-layers 8 --layer-limits 8 --max-sequence-length 1024 --json /tmp/hipengine-hidden-bisect-L8-512-16.json`
      emitted `status=mismatch_found`, first hidden mismatch at generated
      index 1, and first token mismatch at row 0 index 13 (< 87). CPU guard
      coverage: `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C2.3 native linear-attention parity.** Root-cause the remaining
      c=2 hidden/generated drift isolated by the selected-QKV/Z projection
      controls; native segmented conv/GDN/recurrent state,
      batch-GEMV/Marlin linear output, and native A/B projections for c=2/c=4/c=8 are
      token-green under selected-QKV/Z, whose correctness metadata no longer carries
      a generic projection decode blocker; the current retained blockers are native QKV/Z
      projections, grouped-compact linear MoE under this shape, and native
      full-attention parity. Row-aware batch-GEMV/Marlin linear output is green and no longer blocks, although the older fused/native linear output path remains a negative control. Acceptance: C2.2 reports hidden equality for the failing fixture on
      native linear-attention projection/output paths and generated-token equality
      stays green without selected-c1 linear replay.
      Progress: decode batch rows now use grouped compact MoE
      scratch for `tokens>1` instead of selected-MoE c1 wrappers, and
      decode-execution metadata reports `moe_decode_path`/`moe_decode_rows`,
      grouped-compact and selected-c1 fallback layer counts, and per-layer
      decode traces so retained gates and diagnostics can distinguish batched
      selected-c1 MoE from true per-row fallback paths; CPU coverage now locks the
      token-major routed-lane → token-row, selected-expert → expert-group,
      sorted routing-weight, weighted selected-branch accumulation,
      lane-to-sorted-row helper semantics used by grouped MoE combine metadata,
      selected-c1 projection/state replay inputs, selected-c1/batch-GEMV
      linear-output diagnostic metadata plus blockers, hidden-bisect
      all-selected-c1 CLI/workload metadata, and retained-schema rejection for
      diagnostic per-layer linear decode/projection/state/selected-c1+batch-GEMV
      output, full-attention boundary, per-layer trace-source, batch/decode
      aggregate trace-summary paths via a shared deny-list, diagnostic
      CLI/env/structured-metadata overrides via a shared constants fragment list,
      shared hidden-bisect diagnostic evidence fragments, shared diagnostic hidden-trace
      correctness fields, decode-execution trace fields outside execution,
      future trace-key fragments in metadata/commands, and shared diagnostic
      profiler kernel-name fragments,
      but
      `/tmp/hipengine-hidden-bisect-L1-8-512-1-grouped.json` still reports the
      first hidden mismatch at layer-limit 6 (row 0, generated index 1), and the
      old row-0 token idx-13 mismatch remains. Latest traced diagnostic
      `/tmp/hipengine-hidden-bisect-L6-512-1-traced.json` still emits
      `status=mismatch_found` at layer-limit 6 (row 0, generated index 1,
      `max_abs=0.00146484375`, no token mismatch) and now copies the failing
      step's `batch_decode_execution.layer_executions` into
      `correctness.first_hidden_mismatch`; that trace shows native batch
      full-attention at layer 3 and grouped-compact MoE on all six decoded
      layers, keeping C2.3 focused on the linear-attention+MoE layer rather than
      sampler, selected-c1 fallback, or split-K paths. The follow-up artifact
      `/tmp/hipengine-hidden-bisect-L6-512-1-maxdim.json` uses the richer hidden
      comparison schema and localizes the top row-0 difference to hidden dim
      1269 (`batch=0.8564453125`, `c1=0.85498046875`, signed diff
      `+0.00146484375`), giving the native linear-attention parity fix a stable
      coordinate to inspect across projection/state/output and MoE control
      traces. The latest
      top-diff artifact `/tmp/hipengine-hidden-bisect-L6-512-1-topdiff.json`
      adds `elements_over_atol=1` and the top eight hidden-coordinate diffs to
      each row comparison; row 0's only over-tolerance element is still dim 1269
      while row 1 remains within tolerance despite bit-level drift. A paired
      L5/L6 run at `/tmp/hipengine-hidden-bisect-L5-L6-512-1-topdiff.json`
      confirms L5 hidden/token equality still passes (`max_abs≤0.00048828125`,
      `elements_over_atol=0` for both rows) and L6 is the first failing layer,
      with the same row-0 dim-1269 single over-tolerance element. The refreshed
      transition artifact `/tmp/hipengine-hidden-bisect-L5-L6-512-1-transition.json`
      records this as `correctness.first_failing_layer_transition` with
      `previous_green_layer_limit=5`, `failing_layer_limit=6`,
      `adjacent_layer_limits=true`, and the embedded first-hidden-mismatch plus
      native decode trace. The refreshed row-scoped artifact
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-transition-rows.json` also tags
      the transition as `failure_modes=["hidden"]`,
      `hidden_failure_rows=[0]`, and `token_failure_rows=[]`, keeping the
      current C2.3 target to a row-0 hidden-state divergence before the longer
      decode token mismatch. The execution-scoped refresh
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-transition-exec.json` lifts the
      failing and previous-green layer execution records into the transition;
      both are `linear_attention` layers with `moe_decode_path=grouped_compact`
      and `full_attention_decode_path=not_applicable`, isolating the first red
      boundary to the layer-5 linear-attention/grouped-MoE decode output. The
      focus refresh `/tmp/hipengine-hidden-bisect-L5-L6-512-1-focus.json` adds
      `first_hidden_mismatch_focus` for row 0 / dim 1269; that coordinate is
      the failing layer's top diff but is not present in the previous-green L5
      row-0 top-diff list, narrowing the jump to the L6 layer output. The
      row-focus refresh `/tmp/hipengine-hidden-bisect-L5-L6-512-1-rowfocus.json`
      adds per-row focus lists for that coordinate: at L6, row 0 is the only
      hidden-failing row (`abs_diff=0.00146484375`, over tolerance) while row 1
      shares dim 1269 as its top diff but remains within tolerance
      (`abs_diff=0.00048828125`); at L5, neither row has dim 1269 in its
      top-diff list. A selected-c1 MoE probe at
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-selected-c1-moe.json` forces
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_MOE=1` and records
      `moe_decode_path=selected_c1_forced`/`moe_grouped_compact_layers=0`; the
      L6 row-0 hidden mismatch persists at dim 1269 (`max_abs=0.001953125`, no
      token mismatch), so the reduced failure is not cleared by bypassing the
      grouped-compact WMMA MoE path. The per-row-linear probe at
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-per-row-linear.json` forces
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_LINEAR=1`; it records
      `linear_attention_decode_path=selected_c1_per_row_fallback`,
      `native_caware_decode=false`, and the same L6 row-0 dim-1269 hidden
      failure (`max_abs=0.00146484375`, no token mismatch), so the reduced
      failure is also not cleared by replacing the batch linear-attention
      segment path with per-row c=1 linear layers. The per-row full-attention
      probe at `/tmp/hipengine-hidden-bisect-L5-L6-512-1-per-row-full-attn.json`
      forces `HIPENGINE_QWEN35_BATCH_FULL_ATTN_NATIVE=0` through
      `--batch-decode-full-attn-path per_row`; it records
      `full_attention_decode_path=per_row_context_fallback`,
      `native_full_attention_layers=0`, and again preserves the L6 row-0
      dim-1269 hidden failure (`max_abs=0.00146484375`, no token mismatch), so
      the reduced failure is not cleared by replacing the native batch
      full-attention layer either. Two tolerance probes then bracket the scale
      of the drift without changing the retained correctness gate:
      `/tmp/hipengine-hidden-bisect-L5-L6-512-1-atol2e-3.json` passes L5/L6 at
      `hidden_atol=0.002` (`status=eq_ok`, no token mismatch, L6 row-0 dim
      1269 still the top diff at `0.00146484375`), while
      `/tmp/hipengine-hidden-bisect-L1-8-512-1-atol2e-3.json` first fails at
      layer-limit 8 on the same row/dim after the next full-attention layer
      (`max_abs=0.002197265625`, previous-green layer-limit 7 has
      `max_abs=0.001953125`, no token mismatch). That means the strict 1e-3 L6
      report is a small BF16-scale drift, but it monotonically grows enough by
      L8 to remain a real hidden-state blocker before full 40-layer equality.
      A selected-coordinate trace at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-focus1269.json`
      adds `--focus-hidden-flat-index 1269` so every row/layer records that
      coordinate even when it is outside the top-diff list: row 0 is exact at
      L5, jumps to `0.001953125` at L6 and remains there through L7, then grows
      to `0.00244140625` at L8; row 1 stays at or below `0.0009765625` and no
      token mismatch appears. The all-per-row variant
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-focus1269.json`
      combines selected-c1 MoE, per-row linear attention, and per-row
      full-attention fallbacks (`moe_grouped_compact_layers=0`,
      `moe_selected_c1_fallback_layers=8`); it still fails first at L8 on row 0
      dim 1269 (`max_abs=0.002197265625`, no token mismatch). The prefill-aware
      refresh
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-prefill-focus1269.json`
      adds final-prefill hidden comparisons for the same run; compact prefill
      vs independent c=1 final hidden passes at every L5-L8 limit under
      `hidden_atol=0.002` (L8 row-0 dim 1269 is only `0.0009765625`), while
      the first decode step still reaches `0.002197265625`. The linear-state
      refresh
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-state-focus1269.json`
      compares compact-prefill slot linear states with independent c=1 states
      before decode; final hidden remains green, but `prefill_linear_state_passed=false`
      for every L5-L8 limit at `state_atol=1e-6` (for example L6 layer-5
      recurrent state `max_abs=0.0072229355573654175`, conv state
      `max_abs=0.0078125`). The row-scoped refresh
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-state-rows-focus1269.json`
      adds per-row state maxima: at L6/layer 5, row 0 is the larger recurrent
      offender (`0.0072229355573654175` vs row 1 `0.0025315284729003906`),
      while conv is large for both rows (`0.0078125`). A per-segment linear
      prefill diagnostic at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-perseg-prefill-all-per-row-focus1269.json`
      forces `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_LINEAR=1` and
      records `linear_attention_prefill_path=per_segment`; it still leaves
      `prefill_linear_state_passed=false` and the same L8 row-0 dim-1269 decode
      failure (`max_abs=0.002197265625`, no token mismatch). This keeps the
      live fix target on packed-prefill linear-state materialization / slot-state
      contents, especially row-0 recurrent state, not final prefill hidden,
      segment state-index ordering alone, or any single native batch decode
      subpath. A pre-linear-input trace at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-all-per-row-inputs-focus1269.json`
      shows the state drift is input-driven before those later linear layers:
      `prefill_linear_input_passed=false` from L5 onward, with the first bad
      layer-4 input already after full-attention layer 3 (row 0
      `max_abs=0.0059814453125` at prompt token 10 dim 751; row 1
      `0.00305938720703125` at token 176 dim 1237). Layer-5/layer-6 inputs
      then grow (`0.00951385498046875` and `0.01171875` row-0 maxima), while
      final prefill hidden still passes and the L8 decode failure remains row 0
      dim 1269 (`0.002197265625`, no token mismatch). A follow-up full-attn
      prefill diagnostic at
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-perseg-fullprefill-all-per-row-inputs-focus1269.json`
      forces `HIPENGINE_QWEN35_PACKED_PREFILL_FORCE_PER_SEGMENT_FULL_ATTN=1`
      with local block tables, per-slot caches, and c=1-style AOTriton prefill;
      that run clears the L5-L8 hidden/token gate (`status=eq_ok`) and clears
      `prefill_linear_input_passed` for every L5-L8 limit. L5 still reports
      state diffs under the strict `1e-6` state probe, but L6-L8 state summaries
      also pass. The retained-path fix then switches packed-varlen full-attention
      prefill to the AOTriton compact-varlen attention kernel using contiguous
      scratch K/V plus per-segment max sequence lengths; the follow-up retained
      artifact
      `/tmp/hipengine-hidden-bisect-L5-L8-512-1-atol2e-3-packed-aotriton-all-per-row-inputs-focus1269.json`
      records `full_attention_prefill_path=packed_varlen_aotriton` with no
      forced blockers, `status=eq_ok`, and green `prefill_linear_input`,
      `prefill_linear_state`, hidden, and token gates for every L5-L8 limit.
      The longer L8/16 refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-focus1269.json`
      confirms the old row-0 generated-token idx-13 mismatch is gone
      (`token_passed=true`) but still finds a multi-step hidden drift at decode
      step 6 / generated index 7 (`row 0`, dim 1269,
      `max_abs=0.02685546875`) with native c-aware decode. An all-per-row
      decode variant at
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-all-per-row-focus1269.json`
      also keeps tokens green but shifts the first hidden mismatch to decode
      step 11 / generated index 12 (`row 0`, dim 1543,
      `max_abs=0.010440826416015625`), so C2.3 is not closed yet. A decode-state
      trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-decode-states-focus1269.json`
      adds per-step compact-vs-c1 conv/recurrent state summaries; retained
      native c-aware decode still fails hidden equality at step 11 (`row 0`,
      dim 1167, `max_abs=0.0157470703125`) and reports strict state drift from
      step 0. The matching all-per-row trace
      `/tmp/hipengine-hidden-bisect-L8-512-16-packed-aotriton-all-per-row-decode-states-focus1269.json`
      is green (`status=eq_ok`, `decode_linear_state_passed=true`). A one-native-
      subpath sweep then narrows the retained target further: native grouped MoE
      alone stays green at
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-moe-only-decode-states-focus1269.json`,
      while native linear-attention decode alone fails at step 2 / generated
      index 3 (`row 1`, dim 1073, `max_abs=0.00799560546875`) with strict
      decode-state drift from step 0, and native full-attention decode alone
      fails at step 6 / generated index 7 (`row 0`, dim 1269,
      `max_abs=0.02734375`). A native-linear input-trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-linear-only-decode-inputs-focus1269.json`
      shows `decode_linear_input_passed=false` only later (first input mismatch:
      step 12 / generated index 13, layer 6, row 0 dim 585,
      `max_abs=0.00811767578125`); inputs are still green through the first
      hidden mismatch at step 2, while strict state drift starts at step 0. The
      matching all-per-row input trace
      `/tmp/hipengine-hidden-bisect-L8-512-16-all-per-row-decode-inputs-focus1269.json`
      is green. Decode execution metadata now records
      `linear_attention_segment_metadata` (`cu_seqlens` and `state_indices`),
      and the latest row-1/segment probe
      `/tmp/hipengine-hidden-bisect-L8-512-16-c1-batch-segments-decode-metadata-focus1269.json`
      still fails hidden equality with `rows=1`, `state_indices=[0]`, and
      `decode_linear_input_passed=true` (first hidden mismatch: step 11 / dim
      1543, `max_abs=0.010478973388671875`). The matching rows=1 forced-c1
      linear wrapper at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c1-forced-linear-per-row-decode-focus1269.json`
      is green (`status=eq_ok`, hidden/token/state/input gates all true),
      confirming that the divergence is inside the native segment linear-decode
      wrapper/kernel path rather than row setup, full-attention fallback, or
      later selected-c1 MoE. The retained singleton bridge now defaults rows=1
      batch decode through the specialized c1 linear kernel; the refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c1-single-row-c1-linear-focus1269.json`
      is also green with `linear_attention_decode_path=single_row_c1` and the
      same `state_indices=[0]`. A grouped-MoE + native-linear c=2 probe at
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-linear-grouped-decode-metadata-focus1269.json`
      records `state_indices=[0,1]` and fails later at the old row-0 idx-13
      token boundary. A post-singleton c=2 control refresh shows the correctness
      bridge boundaries explicitly: c1-linear + per-row full attention at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-full-per-row-after-singleton-focus1269.json`
      is green (`status=eq_ok`, all decode input/state/hidden/token gates true),
      while c1-linear + native full attention at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-full-native-after-singleton-focus1269.json`
      still fails at decode step 6 / generated index 7 (`row 0`, dim 1269,
      `max_abs=0.027587890625`, tokens still green). This is not only a
      full-layer grouped-MoE artifact: the earlier selected-c1/full-only control
      `/tmp/hipengine-hidden-bisect-L8-512-16-native-full-only-decode-states-focus1269.json`
      also failed at step 6 / row 0 dim 1269 (`max_abs=0.02734375`) with
      `token_passed=true`. The reduced prefill drift is fixed, all-per-row
      fallback, grouped compact MoE, selected-c1 full-layer MoE, singleton row setup,
      and segment state-index mapping are not the blocker at this shape; the
      next C2.3 targets are c>1 native full-attention decode and c>1 native
      linear segment decode, with the per-row linear/full fallbacks serving only
      as non-retained correctness controls until native paths pass. The
      full-attention I/O trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-full-attn-io-trace-focus1269.json`
      adds per-step input/output summaries from
      `scripts/qwen35_batch_hidden_bisect.py` and shows the first native-full
      mismatch at decode step 6 / generated index 7, layer 3 `output` (`row 0`,
      `max_abs=0.008148193359375`, tokens still green), before the layer-limit
      hidden max reaches row 0 dim 1269 at `0.027587890625`. The substage
      trace refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-full-attn-substages-focus1269.json`
      extends that schema with `attn_input`, `gated_attn`, and `o_proj` stage
      gates plus a compact `first_mismatch`; its first substage failure is
      earlier, at decode step 0 / generated index 1, layer 7 `attn_input`
      (`row 0`, dim 1269, `max_abs=0.015625`). The matching layer-3-only
      control at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-full-attn-substages-focus1269.json`
      is green across all full-attention substages. The compact linear-first
      refresh at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-first-mismatch-focus1269.json`
      adds `decode_linear_inputs.first_mismatch` and
      `decode_linear_states.first_mismatch`; it shows the strict state trace
      first diverges earlier at decode step 0, layer 4 `conv` row 0
      (`max_abs=0.0078125`), while visible linear input drift first appears at
      decode step 6, layer 4 row 0 dim 1504 (`max_abs=0.008148193359375`). The
      worst-diff refresh at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-worst-diff-focus1269.json`
      keeps the layer-3-only run green but exposes its largest native-full
      subthreshold drift at layer 3 `output` row 0 dim 1269
      (`max_abs=0.00048828125`). The matching L8 artifact
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-worst-diff-focus1269.json`
      shows worst drift later in layer 7 `attn_input` (`max_abs=0.3984375`) and
      layer-4 `conv` state (`max_abs=0.390625`). Zero-tolerance controls then
      bracket the bit-exactness issue: before the first full-attention layer,
      `/tmp/hipengine-hidden-bisect-L3-512-16-c2-strict-before-full-focus1269.json`
      is exact (`status=eq_ok` with `hidden_atol=0`), and the all-per-row L4
      control
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-strict-all-per-row-focus1269.json`
      is also exact, while native-full L4
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-strict-native-full-focus1269.json`
      fails immediately at generated index 1 (`token_passed=true`, row 0 dim
      1269, `max_abs=0.00048828125`). The strict transition artifact
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-strict-transition-focus1269.json`
      now records `previous_green_layer_limit=3`, `adjacent_layer_limits=true`,
      and compact trace summaries. The gate/context transition refresh at
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-gate-context-transition-focus1269.json`
      shows `input`, `attn_input`, and `gate` are exact, while `attn_context`
      is the first mismatching substage at layer 3 (`max_abs=2.1604321002960205`,
      all 4096 row-0 context elements over zero tolerance). The final layer
      output drift is still row 0 dim 1269 (`max_abs=0.00048828125`). The
      metadata refresh at
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-metadata-transition-focus1269.json`
      records the failing runtime layer with `positions=[512,512]`,
      `decode_live_counts=[513,513]`, `block_table_rows=[[0,1,2,3],[0,1,2,3]]`,
      and `attn_context_trace_source=attention_scratch.query_raw`. Matching
      model-shape primitive controls now fill paged rows across block boundaries
      and cover the 16-Q/2-KV/head-dim-256 path:
      `/tmp/hipengine-multiloop-c2-modelshape-primitive-correctness.json`
      (`context_lens=513,512`, with dense-c1 comparison) and
      `/tmp/hipengine-multiloop-c2-modelshape-primitive-correctness-513x2.json`
      (`context_lens=513,513`) both pass with append mismatches zero,
      `attn_batch_vs_c1_max_abs=0.0`, and NumPy max abs
      `1.4901161193847656e-08`; the dense short-context c1 comparison is also
      tolerance-green (`attn_batch_vs_dense_c1_max_abs=1.862645149230957e-08`).
      The KV-tail trace refresh at
      `/tmp/hipengine-hidden-bisect-L3-L4-512-16-c2-kv-tail-transition-focus1269.json`
      added BF16 cache samples for `first`, `previous`, and `current` tokens;
      the multipoint refresh at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-kv-multipoint-focus1269.json`
      now samples `first`, `page0_last`, `page1_first`, `previous`, and `current`
      positions (`[0,255,256,511,512]` for the failing step). The failing layer's
      `decode_full_kv_samples` still passes at zero tolerance (`bit_mismatch=0`,
      worst `max_abs=0.0`) while `attn_context` still fails.
      The query refresh at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-query-focus1269.json` also showed
      the FP32 `query` launch input was exact (`max_abs=0.0`) while
      `attn_context` remained the first mismatching substage (`max_abs=2.1604321002960205`).
      The context-oracle refresh found the launch-path asymmetry: c1 slot spans
      advertised `max_live_count=max_sequence_length=1024`, which routed the
      513-token c1 reference through split-K while native c=2 used the live
      513-token batch context path. `hipengine/runtime/qwen35_paro_runner.py`
      now keeps host `position_arr`/`context_arr` current and `_slot_full_spans`
      uses those live counts. Evidence:
      `/tmp/hipengine-hidden-bisect-L4-512-1-c2-context-oracle-live-max-focus1269.json`
      has exact input/query and NumPy-oracle-green context (`batch_context_vs_numpy`
      `5.960464477539062e-07`, `c1_context_vs_numpy` `2.384185791015625e-06`,
      `batch_numpy_vs_c1_numpy=0.0`), and
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-context-oracle-live-max-atol1e-3-focus1269.json`
      is `status=eq_ok` with `token_passed=true`, `hidden_passed=true`, and
      `decode_full_context_oracle.passed=true`. L8 still remains open because
      the later linear-attention state drift reaches layer 7 context (`batch_numpy_vs_c1_numpy`
      `0.5023813247680664` at decode step 10 in
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-full-native-live-max-atol1e-3-focus1269.json`).
      The tolerance-transition refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-tolerance-transition-focus1269.json`
      makes this distinction explicit: top-level `first_hidden_bit_drift` is the
      L4 strict-only native-full drift (`passed_under_atol=true`, `max_abs=0.00048828125`),
      while `first_tolerance_hidden_mismatch` and
      `first_failing_layer_transition.hidden_mismatch_kind=over_atol` point to
      L8 decode step 6 / row 0 dim 1269 (`max_abs=0.027587890625`). The transition
      now includes `decode_full_context_oracle` in its trace summaries. The
      linear-state focus refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-linear-state-focus1269.json`
      adds `first_hidden_mismatch_linear_state_focus`; at the first over-tolerance
      hidden mismatch, layers 0-2 conv/recurrent state diffs are zero while layer 4
      has row-0 `conv max_abs=0.390625` and `recurrent max_abs=0.021587848663330078`.
      The first-state focus refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-first-linear-state-focus1269.json`
      adds `first_linear_state_mismatch_focus`; it shows the earliest state drift
      in the L8 failing run is decode step 0 / layer 0 `recurrent` row 0
      (`max_abs=0.001646714168600738`) while that step's hidden row and layer-0
      decode input are still tolerance/exact green. The state-focus refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus2e-3-focus1269.json`
      adds `first_linear_state_mismatch_over_focus_atol`; with `state_atol=0`
      and `state_focus_atol=0.002`, the first focus-threshold state drift is
      decode step 0 / layer 4 `conv` row 0 (`max_abs=0.0078125`) while that
      step's hidden row and layer-4 decode input remain under the hidden
      tolerance (`max_abs=0.00048828125`). The history refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-history2e-3-focus1269.json`
      adds `first_linear_state_mismatch_over_focus_atol_history`; the layer-4
      conv row remains over the 0.002 focus threshold from step 0, and at the
      first over-tolerance hidden step (decode step 6) the same row jumps to
      `state max_abs=0.390625` while `decode_linear_input.max_abs=0.0081787109375`
      and hidden row dim 1269 reaches `0.027587890625`. The same-index refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-same-index2e-3-focus1269.json`
      tracks the original layer-4 conv index `[64, 3]`; it is only `0.015625` at
      step 6 while the row max moved to `[4852, 3]`, so the amplification is not
      one persistent component. The delta refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-delta2e-3-focus1269.json`
      adds previous-state and update-delta comparisons; at step 6 the previous
      state row is still tiny (`max_abs=0.0078125`) but the update delta is already
      large (`max_abs=0.390625`, top index `[4852, 3]`). The execution refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-exec2e-3-focus1269.json`
      lifts the focused step/layer execution into the history: step 6 is layer 4
      `linear_attention` over rows `[0,1]` / slots `[0,1]` with
      `linear_attention_decode_path=selected_c1_per_row_fallback` and
      `native_caware_decode=false`. The row-map refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-state-focus-rowmap2e-3-focus1269.json`
      records `linear_attention_row_state_map=[{row:0,slot:0,state_index:0},{row:1,slot:1,state_index:1}]`
      and matching `state_indices=[0,1]`, so the step-6 drift is not a row/slot
      metadata swap. The producer refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-layer4-input-producer-focus1269.json`
      ties that layer-4 input drift to the preceding layer-3 full-attention block:
      stages `input`, `attn_input`, `gate`, `query`, `attn_context`, `gated_attn`,
      and `o_proj` are green, while final `output` is over tolerance
      (`max_abs=0.0081787109375`, dim 1504) under `native_batch` full-attention
      plus `grouped_compact` MoE. The FP16 output-delta refresh at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-output-minus-oproj-fp16-focus1269.json`
      fixes the trace-value conversion and records `output_minus_o_proj` as the first
      bad delta too (`max_abs=0.0081787109375`, dim 1504), while `o_proj` itself
      remains green (`max_abs=6.103515625e-05`). The post-attention component refresh
      at `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-post-attn-components-focus1269.json`
      traces `residual` and `mlp_input`: layer-3 `residual` is green
      (`max_abs=0.000244140625`, no elements over tolerance), but `mlp_input` is
      the first bad stage (`max_abs=0.00390625`, dim 100). The RMSNorm-oracle refresh
      at `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-rmsnorm-oracle-focus1269.json`
      infers the post-attention RMSNorm transform from the c=1 residual/mlp pair;
      applying it to the c=2 residual leaves only two over-tolerance FP16-ulp
      differences (`max_abs=0.001953125`, dims 135/2012), so the residual's small
      green drift explains much of `mlp_input` but not the last one-ulp gap. The
      per-row full-attention control at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-perrow-fullattn-focus1269.json`
      disables both native linear segments and native full-attention; it still
      reports `status=mismatch_found`, but the first hidden over-atol case moves to
      layer-limit 4 / decode step 1 / row 1 with only one element over tolerance
      (`max_abs=0.00146484375` at focus dim 1269) under `per_row_context_fallback`.
      This means native full-attention/post-attention is not the only c>N equality
      source; small FP16 state drift from the per-row fallback can also cross the
      strict `hidden_atol=0.001` gate. The tolerance-sensitivity refresh separates
      those regimes: the all-per-row control passes generated-token and hidden
      equality at `hidden_atol=0.004` in
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-perrow-fullattn-atol4e-3-focus1269.json`,
      while native full-attention is still over tolerance at the same threshold in
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-fullattn-atol4e-3-focus1269.json`
      (`max_abs=0.027587890625`, 345 elements over). Token/hidden classification:
      the `atol=0.002` all-per-row control is hidden-only fail (`token_passed=true`,
      `first_token_mismatch=null`); the `atol=0.004` all-per-row control is
      token+hidden pass; and the `atol=0.004` native-full control is again
      hidden-only fail (`token_passed=true`, `first_token_mismatch=null`), now
      also emitted as top-level `correctness.failure_modes=["hidden"]` in the
      native-full artifact. The selected-c1 MoE control at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-fullattn-selected-c1-moe-atol4e-3-focus1269.json`
      keeps native full-attention but bypasses grouped-compact MoE; it remains
      hidden-only red at L8 (`max_abs=0.02734375`, 346 elements over, tokens green),
      so grouped-compact MoE is not the source of the large native-full drift.
      The row-count refresh of that same artifact adds per-layer and top-level
      row-failure summaries: L4 has `failure_modes=[]`, `hidden_failure_rows=[]`
      but strict bit drift on both rows, while L8 has `failure_modes=["hidden"]`,
      `hidden_failure_rows=[0,1]`, and `token_failure_rows=[]`; top-level
      `correctness.row_failure_summary` matches hidden rows `[0,1]`, strict rows
      `[0,1]`, and token rows `[]`. The diagnostic schema now also emits
      `decode_full_attention.stage_failure_summary` with per-stage failing rows
      and a compact `first_failure` record,
      `decode_full_context_oracle.comparison_failure_summary` with per-comparison
      row/failure rollups, and the top-level
      `correctness.decode_full_context_oracle_failure_summary` aggregate, so
      native-full artifacts can tell attention-context,
      `mlp_input`/post-attention, and final hidden/token failures apart; CPU
      coverage lives in `test_hidden_bisect_summary_embeds_batch_decode_execution_trace`.
      A new diagnostic switch,
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_POST_ATTN=1` (or hidden-bisect
      `--batch-decode-post-attn-path per_row`), routes only the c>N
      full-attention decode post-attention add/RMSNorm boundary through token-1
      row kernels, labels the decode as a diagnostic fallback, and is covered by
      `test_qwen35_resident_run_layers_batch_decode_can_force_per_row_post_attention_probe`.
      Combined input+post boundary diagnostic metadata is covered by
      `test_qwen35_resident_run_layers_batch_decode_combined_full_attention_boundary_probes_are_non_native`,
      which asserts both per-layer and top-level native-caware flags stay false.
      The first focused artifact,
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-perrow-postattn-atol4e-3-focus1269.json`,
      remains hidden-only red (`token_passed=true`, `failure_modes=["hidden"]`):
      L4 final hidden/token stays green but full-attention substage drift is visible,
      and L8 first fails at decode step 2 / row 1 / dim 1073 (`max_abs=0.008148193359375`).
      Therefore the batch post-attention add/RMSNorm boundary is not the sole
      native-full blocker. A stricter core-isolation control,
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-full-core-perrow-linear-postattn-selected-c1-atol4e-3-focus1269.json`,
      keeps native full-attention decode but forces per-row linear, selected-c1 MoE,
      and per-row post-attention; it still fails hidden-only at L8 decode step 6 / row 0
      / dim 1269 (`max_abs=0.02734375`) while L4 stays green. The refreshed
      top-level context-oracle rollup shows L8 only fails `batch_numpy_vs_c1_numpy`
      (`first_failure`: decode step 0 / row 0 / context dim 2812,
      `max_abs=0.00811624526977539`; worst at step 10, `max_abs=0.4993577003479004`),
      while `batch_context_vs_numpy` and `c1_context_vs_numpy` pass. The added
      top-level full-attention stage rollup now shows L8 first fails at `attn_input`
      (decode step 0 / row 0 / dim 1269, `max_abs=0.015625`); raw stage `input`
      first fails only later at decode step 6, and `attn_context` first fails as
      fp32 at step 0 / dim 2812 (`max_abs=0.008107900619506836`). That means
      the context kernel matches its own oracle; the next C2.3 target is the
      layer-7 attention-input RMSNorm/QKV preparation or state feeding, not raw
      hidden input copy or softmax context math. A diagnostic
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_INPUT=1` control
      (hidden-bisect `--batch-decode-attn-input-path per_row`) now forces just
      the full-attention input RMSNorm through token-1 row kernels. The refreshed
      probe,
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-native-full-core-perrow-attninput-linear-postattn-selected-c1-atol4e-3-focus1269.json`,
      remains hidden-only red: L8 still first fails final hidden at decode step 6
      / row 0 / dim 1269 (`max_abs=0.02734375`), and the post-layer `attn_input`
      trace still first fails at L8 decode step 0 / row 0 / dim 1269
      (`max_abs=0.015625`). The immediate `attn_input_pre_qkv` plus new
      `attn_input_after_rotate`, `attn_input_after_project`, and
      `attn_input_after_prepare` traces all pass at L4 and L8, so the input
      RMSNorm/rotate/project/prepare path is not overwriting `attn_input`; the
      existing post-layer `attn_input` trace is polluted after those stages and
      should not drive the fix. C1 input-scratch tracing is now symmetric with
      native c=2 and includes `attn_input_pre_qkv`, `attn_input_after_rotate`,
      `attn_input_after_project`, and `attn_input_after_prepare`. The refreshed
      probe shows L4 producer stages now pass exactly, so the earlier L4
      `q_proj_key_after_project` signal was missing-c1-trace noise. The first
      full-attention stage drift is now L8 decode step 0 / row 0 at
      `attn_input_pre_qkv` (`max_abs=0.015625`, dim 1269), propagating through
      Q/K preparation (`q_proj_key_after_project` `max_abs=0.0078125`,
      `query_after_prepare` `max_abs=0.005970478057861328`, `key_after_prepare`
      `max_abs=0.005676984786987305`). The hidden-bisect context oracle now also
      stores per-token CRC32 hashes for the full BF16 K/V prefix and emits
      `correctness.decode_full_context_kv_prefix_failure_summary`, covered by
      `test_hidden_bisect_summary_embeds_batch_decode_execution_trace`. It now
      additionally snapshots post-prefill full-KV prefix hashes before any decode
      write and emits `correctness.prefill_full_kv_prefix_failure_summary`, also
      covered by that CPU test. K/V prefix comparisons now carry compact
      `mismatch_positions` summaries (first/last sampled positions, span width,
      and tail-window counts) so prompt-tail/current-token failures can be
      compared across repeats without dumping full K/V. The refreshed
      prefill-aware probe at
      `/tmp/hipengine-hidden-bisect-L4-L8-512-16-c2-prefill-kv-prefix-native-full-core-atol4e-3-focus1269.json`
      is still hidden-only red, has green post-prefill K/V prefix hashes in that
      run, and localizes the decode-time prefix/sample failure to L8 step 0 /
      layer 7 / row 0 current token 512. An L4-only repeat
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-prefill-kv-prefix-repeat-atol4e-3-focus1269.json`
      caught a pre-decode prompt-tail hash failure at layer 3 / row 0 token 500
      that the decode prefix then inherited; the position-summary rerun
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-kv-position-summary-atol4e-3-focus1269.json`
      was green and records an empty tail window (`tail_start=496`,
      `tail_mismatch_count=0`) for the same probe shape. The new hidden-bisect
      `--repeat-runs` mode then recorded two green repeats at
      `/tmp/hipengine-hidden-bisect-L4-512-16-c2-repeat-rollup-atol4e-3-focus1269.json`
      (`status_counts={"eq_ok":2}`, no prefix/sample failed repeats). Prefix
      hash failures now embed the first mismatching token's first eight BF16
      words; the L8 native-full rerun at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-kv-token-samples-atol4e-3-focus1269.json`
      is hidden-only red and shows current-token position 512 at layer 3 / row 0
      with a one-word key-token sample delta (`16166` vs `16167`) while
      post-prefill prefix hashes remain green. The follow-up current-source
      audit at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-current-source-check-v2-atol4e-3-focus1269.json`
      shows both batch and c1 current-token cache writes match their local
      producer traces exactly (`batch_cache_vs_source` and `c1_cache_vs_source`
      bit-mismatch zero), while `batch_source_vs_c1_source` already differs at
      layer 3 / step 0. The promoted top-level rollup at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-current-source-rollup-atol4e-3-focus1269.json`
      reports `correctness.decode_full_kv_current_source_failure_summary` with
      failed kinds `key,value`, first failure `batch_source_vs_c1_source`, and
      layer-limit status without opening the full layer dump. The bit-aware
      stage-context rerun at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-current-source-bitstage-atol4e-3-focus1269.json`
      is hidden/token green for that short L8 probe but still exposes exact-bit
      current-source drift: `key_after_prepare` itself is under the hidden
      tolerance (`max_abs=0.0029807090759277344`), the first hidden-tolerance
      producer failure is `attn_input_pre_qkv` (`max_abs=0.0078125`, flat index
      859, `bit_mismatch=1488`), and the first exact-bit producer drift is
      already at full-attention `input` (`bit_mismatch=1531`). The follow-up
      rollup at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-full-attn-bit-drift-rollup-atol4e-3-focus1269.json`
      emits `correctness.decode_full_attention_bit_drift_summary`; in that run
      the first exact-bit drift is again layer 3 / row 0 full-attention `input`
      (`bit_mismatch=1574`, hidden-atol pass), while the first hidden-atol stage
      failure remains `attn_input_pre_qkv`. The linear-input rollup at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-input-bit-drift-rollup-atol4e-3-focus1269.json`
      emits `correctness.decode_linear_input_bit_drift_summary` and shows the
      earliest exact-bit handoff drift at decode step 0 / layer 1 / row 0
      (`bit_mismatch=1092`, hidden-atol pass), before layer 3 full-attention
      input consumes the drift. The handoff-summary rerun at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-input-handoff-summary-atol4e-3-focus1269.json`
      is hidden/token green but exposes the same exact-bit handoff drift in
      `correctness.decode_linear_input_bit_drift_summary.first_handoff`: target
      layer 1 consumes producer layer 0, whose execution is
      `layer_type=linear_attention` / `linear_attention_decode_path=native_batch_segments`
      with `cu_seqlens=[0,1,2]` and `state_indices=[0,1]`; there is no
      full-attention producer trace for that layer type. The output-to-input
      handoff-copy check at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-handoff-copy-check-atol4e-3-focus1269.json`
      adds `correctness.decode_linear_handoff_summary` and shows the same layer
      0→1 / row 0 exact-bit drift (`bit_mismatch=1092`) is already present in
      the layer-0 native linear-attention producer output, while both batch and
      c1 output→target-input copies are exact (`copy_passed=true`,
      `first_copy_mismatch=null`). The per-row linear replay at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-per-row-replay-atol4e-3-focus1269.json`
      forces `--batch-decode-linear-path per_row`; layer 0→1 is then exact
      (`bit_mismatch=0`) and the first exact-bit producer drift shifts to layer
      4→5 / row 0 (`bit_mismatch=1343`, hidden-atol pass), while the full probe
      still fails hidden only. The selected-c1 MoE replay at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-selected-c1-moe-replay-nativeflag-atol4e-3-focus1269.json`
      keeps `linear_attention_decode_path=native_batch_segments` but forces
      `moe_decode_path=selected_c1_forced`; layer 0→1 still drifts
      (`bit_mismatch=1079`), so the early drift is not a grouped-compact MoE
      reducer artifact. Selected-c1 MoE diagnostics now mark
      `native_caware_decode=false` in workload and layer execution metadata to
      avoid overclaiming. The linear-stage trace at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-stage-trace-atol4e-3-focus1269.json`
      adds `correctness.decode_linear_stage_bit_drift_summary` and shows the
      first layer-0 native `batch_segments` drift starts at the linear-attention
      `qkv` projection (`bit_mismatch=622`, `max_abs=0.0078125`, flat index
      2274) while layer-0 `attn_input` is still exact; downstream `z`,
      `conv_out`, `recurrent_out`, `out_proj`, residual/MLP/output then drift.
      The selected-c1 projection replay at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-selected-c1-projection-atol4e-3-focus1269.json`
      forces token-1 QKV/Z/A/B projections before native segmented state
      updates and marks `native_caware_decode=false`; it eliminates layer-0
      QKV/Z drift, shifting the first layer-0 stage mismatch to
      `recurrent_out` (`bit_mismatch=4096`, `max_abs=2.7936763763427734`), so
      the remaining native red is in segmented conv/GDN/recurrent state update
      rather than projection, copy, or MoE. The selected-c1 projection+state
      replay at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-selected-c1-proj-state-atol4e-3-focus1269.json`
      forces token-1 QKV/Z/A/B, conv/GDN/recurrent, and output projection while
      keeping grouped-compact MoE; it is hidden/token green (`status=eq_ok`,
      `correctness.passed=true`) but non-retained (`native_caware_decode=false`)
      and leaves only tiny exact-bit layer-0 output drift (`bit_mismatch=541`,
      `max_abs=3.0517578125e-05`). Splitting the output path at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-selected-c1-state-native-out-atol4e-3-focus1269.json`
      keeps selected-c1 projection/state replay but forces native batched output
      projection (`linear_attention_output_path=batch_from_f32`); it goes hidden
      red again, with first layer-0 exact drift at `out_proj` (`bit_mismatch=312`,
      hidden-atol pass) and first layer-0 hidden-atol stage failure at
      `mlp_input` (`max_abs=0.0078125`). The batch-GEMV output variant at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-selected-c1-state-batch-gemv-out-atol4e-3-focus1269.json`
      bypasses the row>1 AWQ prefill output kernel and reduces layer-0
      `out_proj` drift to one bit (`bit_mismatch=1`, `max_abs=2.384185791015625e-07`),
      but that older selected-c1-state probe still eventually goes hidden red and
      is superseded by the refreshed native-state output audit below. The
      all-selected-c1 control
      at `/tmp/hipengine-hidden-bisect-L8-512-16-c2-linear-all-selected-c1-atol4e-3-focus1269.json`
      forces selected-c1 MoE in addition to selected-c1 projection/state/output
      and regresses to hidden red (`status=mismatch_found`, first hidden mismatch
      step 6 / dim 1269), so grouped-compact MoE is not the residual blocker and
      should remain the preferred MoE path while fixing linear-attention parity.
      A selected-c1 output replay audit found the earlier native-state/
      selected-output probes were invalid: segmented state replay writes raw
      recurrent values to `scratch.recurrent_out` and the gated lowp activation
      to `scratch.recurrent_bf16`, while the token-1 output replay path recast
      `recurrent_out`. `project_linear_attention_prefill_rows_out_fp16` now
      routes selected-output diagnostics after native segmented state through the
      existing lowp `recurrent_bf16` path; CPU coverage is
      `test_qwen35_decode_state_selected_output_uses_prefill_lowp_after_segment_state`.
      With that fix, L1 selected-c1 projection/native segmented state passes both
      selected-output and native-output controls:
      `/tmp/hipengine-hidden-bisect-L1-512-4-c2-selected-proj-native-state-selected-out-fixed-focus1269.json`
      and
      `/tmp/hipengine-hidden-bisect-L1-512-4-c2-selected-proj-native-state-native-out-focus1269.json`
      are both `status=eq_ok`; the selected-c1 projection/state/native-output L1
      control
      `/tmp/hipengine-hidden-bisect-L1-512-4-c2-selected-proj-state-native-out-focus1269.json`
      is also `status=eq_ok`. The matched L8 selected-c1 projection/native-state/
      selected-output + per-row-full probe
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-selected-out-fixed-perrow-full-atol4e-3-focus1269.json`
      is hidden/token green, while selected-c1 projection/native-state/native-
      output + per-row-full at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-native-out-perrow-full-atol4e-3-focus1269.json`
      still fails at the old row-0 token idx-13 boundary with first hidden failure
      at decode step 6 / row 1 (`max_abs=0.01242828369140625`). The refreshed
      batch-GEMV output fallback probe
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`
      clears that selected-projection/native-state/per-row-full L8 gate
      (`status=eq_ok`, hidden/token green), proving the fused native batch output
      kernel is the red branch for that controlled path. Restoring native
      QKV/Z/A/B projections while keeping batch-GEMV output and per-row full
      attention at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-native-proj-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`
      leaves tokens green but hidden-only red (`status=mismatch_found`, first
      hidden failure step 11 / row 0 `max_abs=0.010463714599609375`), with the
      first layer-0 linear-stage bit drift at `qkv` (`max_abs=0.0078125`, flat
      index 2274). The code now exposes a diagnostic row-aware GEMV projection
      path (`--batch-decode-linear-projection-path batch_gemv`, backed by
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_GEMV_LINEAR_PROJECTIONS`); the matched
      probe
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-batch-gemv-proj-state-out-perrow-full-atol4e-3-focus1269.json`
      reduces first layer-0 `qkv` drift to `max_abs=0.0009765625` under the
      0.004 hidden tolerance but remains hidden-only red at step 11 / row 0
      (`max_abs=0.010431289672851562`). A narrower QKV/Z-only selected-c1
      diagnostic path (`--batch-decode-linear-projection-path selected_qkv_z`,
      backed by `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_QKVZ`)
      makes layer-0 `qkv` and `z` exact while keeping native A/B projection,
      native segmented state, batch-GEMV output, and per-row full attention; the
      matched probe
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-qkvz-native-ab-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`
      is still hidden-only red at step 11 / row 0 (`max_abs=0.010393142700195312`).
      The complementary A/B-only selected-c1 diagnostic
      (`--batch-decode-linear-projection-path batch_gemv_selected_ab`, backed by
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_SELECTED_C1_LINEAR_AB` plus the
      batch-GEMV QKV/Z override) keeps generated tokens green but is also
      hidden-only red at step 11 / row 0
      (`/tmp/hipengine-hidden-bisect-L8-512-16-c2-batch-gemv-qkvz-selected-ab-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`,
      current rerun `max_abs=0.010467529296875`); layer-0 `qkv`/`z` drift is
      under 0.004
      but still bit-different (`qkv` `max_abs=0.0009765625`, `bit_mismatch=5`),
      and the first red stage is the same large `recurrent_out` amplification.
      Hidden-bisect now emits `decode_linear_projection_bit_drift_summary` as a
      strict QKV/Z bit-exactness rollup; the same batch-GEMV-QKVZ/selected-A-B
      probe records `bit_exact=false`, `drift_stages=["qkv", "z"]`, and
      `passed_under_atol=false` once all decode steps/layers are considered;
      it also records `over_atol_drift_stages=["qkv", "z"]`, the same lists in
      `layer_limits=[{"layer_limit": 8, ...}]`, plus the first over-tolerance
      projection drift (`qkv`, layer 1, step 0, row 0), separating the layer-0
      under-tolerance first bit drift from later over-tolerance projection
      amplification. A narrow multi-limit probe at
      `/tmp/hipengine-hidden-bisect-L1-L2-512-16-c2-batch-gemv-qkvz-selected-ab-state-batch-gemv-out-perrow-full-atol4e-3-focus1269.json`
      is hidden/token green (`status=eq_ok`) while reporting
      `first_over_atol_layer_limit={"layer_limit": 2, ...}` plus per-limit
      drift/under-atol/over-atol stage counts: `layer_limit=1` QKV/Z under-atol
      drift only and `layer_limit=2` QKV/Z over-atol drift, confirming projection
      amplification begins at the second retained linear layer before it becomes
      an end-to-end hidden mismatch.
      A compact comparison artifact
      `/tmp/hipengine-hidden-bisect-L1-L2-512-16-c2-projection-route-compare.json`
      now compares that A/B-selected route against the complementary
      selected-QKV/Z/native-A/B route. Both routes stay hidden/token green and
      agree that the first over-tolerance projection limit is L2. Its compact
      classification is
      `same_first_over_atol_location_with_record_and_drift_delta`: the first
      over-tolerance record has the same location in both routes (`layer_limit=2`,
      layer 1, step 0, row 0, QKV, flat index 857) with only a one-bit-count
      difference (`3722` vs `3721`). The compare helper now accepts top-level
      and per-layer-limit expectation flags for those classifications/statuses,
      the exact first-over-atol coordinate, and the one-bit mismatch delta; the
      artifact records the passed expectation set under `expectations` plus
      SHA256/size metadata for both source artifacts, and the compare helper can
      require those fingerprints with `--expect-artifact-sha256` /
      `--expect-artifact-size-bytes`, so this diagnostic can be rechecked
      mechanically without a bespoke JSON assertion snippet. They differ at L1:
      selected-QKV/Z/native-A/B has no QKV/Z drift while the A/B-selected route
      has under-tolerance QKV/Z drift. Together with the green selected-all
      projection control, this means both QKV/Z bit exactness and A/B exactness
      are still required for this controlled path; do not promote the batch-GEMV
      QKV/Z path just because an early direct stage max error is under hidden
      tolerance.
      Re-enabling native full-attention decode
      on the selected-projection/native-state/batch-GEMV-output control at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-batch-gemv-out-native-full-atol4e-3-focus1269.json`
      keeps generated tokens green but is hidden-only red (`status=mismatch_found`,
      first hidden failure step 6 / row 1 `max_abs=0.0124053955078125`), with
      `native_full_attention_layers=2` and no per-row-full blocker. Forcing only
      the full-attention input/RMSNorm boundary per row at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-batch-gemv-out-native-full-perrow-input-atol4e-3-focus1269.json`
      does not clear that native-full path (`status=mismatch_found`, tokens green,
      first hidden failure step 6 / row 0 `max_abs=0.027587890625`). Combining
      per-row input and post-attention boundaries at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-batch-gemv-out-native-full-perrow-input-post-atol4e-3-focus1269.json`
      also remains hidden-only red (`status=mismatch_found`, tokens green, first
      hidden failure step 6 / row 0 `max_abs=0.02734375`). Forcing selected-c1
      linear state replay as well at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-state-batch-gemv-out-native-full-atol4e-3-focus1269.json`
      still leaves native full-attention hidden-only red (`status=mismatch_found`,
      tokens green, first hidden failure step 6 / row 0 `max_abs=0.027099609375`).
      Decode-execution metadata now also lifts linear-attention projection,
      state, and output route choices plus full-attention input/context/post
      boundary choices to top-level `linear_attention_projection_path`,
      `linear_attention_state_path`, `linear_attention_output_path`,
      `full_attention_input_decode_path`,
      `full_attention_context_decode_path`, and `post_attention_decode_path`;
      retained-bench and accepted-artifact gates reject non-native top-level
      values, so a diagnostic fallback cannot hide behind missing or stale
      per-layer traces. A context-only diagnostic switch now exists for the next
      isolation step:
      hidden-bisect `--batch-decode-attn-context-path per_row` sets
      `HIPENGINE_QWEN35_BATCH_DECODE_FORCE_PER_ROW_FULL_ATTN_CONTEXT`, routes the
      native-full context/gate stage through slot-specific c=1 spans, records
      `full_attention_context_decode_path=per_row_context_gate_fallback`, and is
      rejected by retained-artifact gates; CPU coverage is
      `test_qwen35_decode_state_decode_batch_full_attention_can_force_per_row_context`,
      `test_qwen35_resident_run_layers_batch_decode_can_force_per_row_full_attention_context_probe`,
      and hidden-bisect dry-run / retained-schema tests. The first context-only
      probe at
      `/tmp/hipengine-hidden-bisect-L8-512-16-c2-selected-proj-native-state-batch-gemv-out-native-full-perrow-context-atol4e-3-focus1269.json`
      remains hidden-only red (`status=mismatch_found`, tokens green, first
      hidden failure step 2 / row 1 `max_abs=0.008087158203125`), with the
      context trace source fixed to `attention_scratch.query_raw`; context-oracle
      numeric checks pass for `batch_context_vs_numpy` and `c1_context_vs_numpy`
      but remain red for `batch_numpy_vs_c1_numpy` (key/value prefix-hash checks
      are also red). The failure is therefore not resolved by either boundary
      fallback alone, both boundaries together, diagnostic linear-state replay,
      or per-row context/gate replay. This shifts the
      current C2.3/C2.4 target to native A/B projection exactness/amplification,
      native batched output fallback/retention, and native full-attention hidden
      parity; raw `recurrent_out` stage summaries from segmented state are not
      equivalent to token-1 decode post-gate `recurrent_out` and must not be
      used as a closure/blocker signal. A refreshed L8 context split under the
      selected-projection/native-state/batch-GEMV-output native-full control
      confirms the narrower context-only split is still hidden-only red; adding
      per-row KV append or per-row QKV prep does not clear it. The per-row full-
      attention positive control remains hidden/token green, so the blocker is
      still native batch context/output integration, not the standalone KV append
      or QKV prep branch
      (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-fullattn-context-split-394/summary.json`).
      The apparent narrower pre-QKV+context green split was deconfounded by
      follow-up source/GPU checks. `--batch-decode-attn-scratch-path per_row`
      enters the broad `force_per_row_layer_scratch` branch and replays the full
      full-attention layer per row, so the iter395/396 green results are not
      proof that native batch context can remain active. The true QKV temp-
      scratch diagnostic is `--batch-decode-attn-qkv-path per_row`: QKV-temp only
      remains token-green but hidden-red, and fresh true input+QKV-temp+context
      variants without the broad layer-scratch flag also remain hidden-red with
      batch-GEMV O, per-row O, and context-only controls. A fresh broad layer-
      scratch positive control is still hidden/token green. The new
      `per_row_batch_scratch` diagnostic then replayed each row through the c1
      full-attention layer while using row views of the normal batch scratch and
      also cleared hidden/token parity. Thus independent scratch allocation is
      not required for the green path. The follow-up `per_row_attn_batch_moe`
      diagnostic keeps grouped compact MoE in batch but replays full-attention
      attention/post work per row with batch scratch row views; it is also
      hidden/token green. The next `per_row_attn_batch_post_moe` split keeps
      batch post-attention and grouped compact MoE, replaying only attention/O
      per row, and is also hidden/token green. The follow-up
      `per_row_attn_batch_o_post_moe` split keeps batch O projection, batch
      post-attention, and grouped compact MoE, replaying only the pre-O
      full-attention sequence per row, and is also hidden/token green. The
      context/gate split shows `per_row_preqkv_append_context_gate_batch_o_post_moe`
      (per-row context+gate, batch O/post/MoE) is hidden/token green, while the
      same pre-QKV+append sequence with batch context/gate and the per-row
      context + batch-gate split are hidden-red. The next existing-diagnostic
      split shows batch pre-QKV/append + per-row paged context/gate is still
      hidden-red; adding per-row append, per-row QKV+append, and even
      phase-separated per-row input+QKV+append+context/gate remain hidden-red.
      However batch pre-QKV with interleaved per-row append+context/gate is
      hidden/token green, and a follow-up L8+L40 hidden-bisect keeps that
      diagnostic green through all 40 layers on the short decode workload. The
      blocker therefore narrows to append+context/gate interleaving before batch
      O/post/MoE, not batch pre-QKV, not merely which subkernels are per-row,
      and not O projection, post-attention, c1 MoE replay, or independent
      scratch allocation
      (`benchmarks/results/2026-06-05-hipengine-qwen35-c2-scratch-semantics-397/summary.json`,
      `benchmarks/results/2026-06-05-hipengine-qwen35-c2-true-preqkv-context-398/summary.json`,
      `benchmarks/results/2026-06-05-hipengine-qwen35-c2-batch-view-layer-scratch-399/summary.json`,
      `benchmarks/results/2026-06-05-hipengine-qwen35-c2-perrow-attn-batch-moe-400/summary.json`,
      `benchmarks/results/2026-06-05-hipengine-qwen35-c2-perrow-attn-batch-post-moe-401/summary.json`,
      `benchmarks/results/2026-06-05-hipengine-qwen35-c2-perrow-attn-batch-o-post-moe-402/summary.json`,
      `benchmarks/results/2026-06-05-hipengine-qwen35-c2-perrow-context-gate-split-403/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-preqkv-append-context-order-404/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-append-context-interleave-405/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-l40-append-context-interleave-406/summary.json`).
      Bridging that green diagnostic into the full 512/128 retained bench keeps
      generated-token equality green at `[137,137]`, but only with per-row
      KV-append/context/interleave blockers and `native_caware_decode=false`; the
      no-flag retained bench is also generated-token green and native, while
      no-flag, batch-temp-context, and compact-cache-context hidden-bisect
      contrasts remain hidden-red. The retained-default hidden-bisect projection
      path also clears at L8+L40 with only the same per-row KV-append/context plus
      interleaved ordering fallback, so the blocker is not selected-projection
      specific and not batch linear projection/state/output. A phase-isolation
      pass then shows per-row KV append followed by phased native batch context
      is still hidden-red, native rowchunk1 alone is still hidden-red, and the
      retained-default interleaved append/context fallback stays hidden/token
      green through L40 warmup8+decode32. A rowchunk1 + same per-row
      append/context flags contrast stayed hidden-red because the rowchunk branch
      sliced to `tokens=1` and then fell through to the batch append/context
      path rather than the `tokens>1` append+context interleave branch. The
      diagnostic branch now honors append+context interleaving for token-1
      chunks too; after that runtime diagnostic fix, rowchunk1 + the same
      per-row append/context flags is hidden/token green at L8+L40 and stays
      hidden/token green through L40 warmup8+decode112. The full retained 512/128
      bench with the same rowchunk1 diagnostic flags is still deterministically
      token-red at `[137,104]` while the no-flag active default remains green;
      that narrows the remaining rowchunk1 diagnostic gap away from traced
      hidden/full-attention parity and toward full E2E/sampler or post-hidden
      feedback. The interleaving fallback is therefore correctness evidence and
      a row-bounded diagnostic path, not yet a retained/native fix
      (`benchmarks/results/2026-06-06-hipengine-qwen35-c2-retained-interleave-bridge-407/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-retained-default-interleave-hidden-408/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-append-context-phase-isolation-409/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-rowchunk1-callorder-410/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-rowchunk1-interleave-fix-411/summary.json`,
      `benchmarks/results/2026-06-06-hipengine-qwen35-c2-rowchunk1-retained-bridge-412/summary.json`).
      The next target is retained projection/output/full-attention parity without
      diagnostic flags; do not change paged-KV writer code yet. Do not re-open
      row setup, native linear segment metadata, output trace/copy semantics, or
      grouped MoE output yet.
- [x] **C2.4 full c=2 BF16 512/128 equality.** Re-run the full 40-layer c=2
      512/128 retained protocol with `serial_lm_head` default and no serial
      decode bridge. Acceptance: generated-token equality vs two c=1 sessions
      passes; if timing is still not retained, artifact is `blocked` for a
      non-correctness reason. Evidence: `/tmp/hipengine-e2e-native-c2-512-128.json`
      reports `generated_token_equality.passed=true`, `prefix_lengths=[137,137]`,
      `workload.batch_decode_linear_path=batch_segments`, `workload.batch_decode_moe_path=selected_c1`,
      selected-QKV/Z linear projections with native A/B, native segmented linear
      state, batch-GEMV/Marlin linear output, batched selected-c1 MoE, and
      native full-attention with row-aware batch-GEMV output; status remains
      `blocked` because diagnostic fallbacks prevent retained/perf claims. The
      native segmented-state control is green for c=2/c=4/c=8 with selected-c1
      projections/output (`/tmp/hipengine-e2e-native-c2-c4-c8-selected-proj-native-state-selected-out-perrow-moe-full-per-row-matrix.json`),
      the batch-GEMV/Marlin output control is green for c=2/c=4/c=8
      (`/tmp/hipengine-e2e-native-c2-c4-c8-selected-proj-native-state-marlin-out-perrow-moe-full-per-row-matrix.json`),
      native A/B projections are now green for c=2/c=4/c=8 with selected-QKV/Z
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-auto-qkvz-equality-matrix/summary.json`).
      Therefore selected-c1 state, selected-c1 output replay, and full selected-c1
      A/B projection replay are no longer part of the c=2/c=4/c=8
      correctness-default gate. The batched selected-c1 MoE matrix
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-batched-selected-moe-matrix/summary.json`)
      removes per-row linear/full-attention MoE replay from the c=2/c=4/c=8
      equality default. The follow-up selected-rotary-input, selected-QKV-only,
      and selected-Z-only diagnostics all fail c=2 at the same row-1 prefix-104
      boundary, so selected-c1 QKV/Z remains required as a complete correctness pair.
      The promoted selected-QKV/Z matrix
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-selected-qkvz-promoted-matrix/summary.json`)
      keeps c=2/c=4/c=8 prefixes at `137` without the generic QKV/Z decode blocker,
      while still marking the path non-retained/non-native for projection-dispatch acceptance.
      The equality-gate promoted matrix
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-equality-gate-promoted-matrix/summary.json`)
      keeps the same prefixes and removes the stale generated-equality blocker from
      child `batch_execution.blockers` only when the embedded equality comparison passed.
      The context-gate promoted matrix
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-context-gate-promoted-matrix/summary.json`)
      keeps the same prefixes and also removes the stale BF16/context<1024 full-attention
      support caveat only when the child artifact records native full-attention decode
      under BF16 KV with max context below 1024.
- [x] **C2.5 c=4/c=8 BF16 equality.** Extend the same gate to c=4 and c=8.
      Acceptance: generated-token equality passes for both shapes, with
      aggregate/per-request scaling fields recorded even if not yet optimized.
      Evidence: `/tmp/hipengine-e2e-native-c2-c4-c8-equality-matrix.json`
      reports c=2/c=4/c=8 generated equality green with min equal-prefix `137`
      for every row; the child retained artifacts live under
      `/tmp/hipengine-e2e-native-c2-c4-c8-equality-matrix/` and remain
      non-retained/blocking while auto selected-QKV/Z projection,
      row-aware batch-GEMV/Marlin linear output, batched selected-c1 MoE, and native
      full-attention decode with row-aware batch-GEMV full-attention output defaults
      are active. The linear-output batch-GEMV blocker was eliminated by
      `benchmarks/results/2026-06-02-hipengine-qwen35-native-linear-batchgemv-output-matrix/summary.json`,
      and the batch-GEMV full-attention output blocker was eliminated by
      `benchmarks/results/2026-06-02-hipengine-qwen35-native-fullattn-batchgemv-output-matrix/summary.json`;
      both keep c=2/c=4/c=8 prefixes at `137` and remove their output-path blockers from child artifacts. The
      batched selected-c1 MoE matrix
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-batched-selected-moe-matrix/summary.json`)
      also keeps c=2/c=4/c=8 prefixes at `137` and removes the per-row MoE blockers while leaving the selected-c1 MoE diagnostic blocker active. Progress:
      primitive GPU correctness now has c=4 and c=8 artifacts at
      `/tmp/hipengine-multiloop-c4-correctness.json` (`rows=4`,
      `context_lens=[1,2,3,4]`) and `/tmp/hipengine-multiloop-c8-correctness.json`
      (`rows=8`, `context_lens=[1,2,3,4,1,2,3,4]`); both artifacts report
      `append_*_mismatch=0`, `append_batch_aa_*_mismatch=0`,
      `attn_batch_vs_c1_max_abs=0.0`, `attn_batch_aa_max_abs=0.0`,
      `device.env.HIP_VISIBLE_DEVICES=1`, `device_name=AMD Radeon RX 7900 XTX`,
      and `passed=true`; exact commands:
      `HIP_VISIBLE_DEVICES=1 python3 scripts/qwen35_batch_correctness.py --rows 4 --json /tmp/hipengine-multiloop-c4-correctness.json`
      and
      `HIP_VISIBLE_DEVICES=1 python3 scripts/qwen35_batch_correctness.py --rows 8 --json /tmp/hipengine-multiloop-c8-correctness.json`;
      combined c-sweep dry-run coverage locks per-category env-prefixed command text/argv with `HIP_VISIBLE_DEVICES=1`, per-category command text/argv schema and synchronization including optional-row all-row checks, Python launcher validation including optional-row all-row checks, script/category matching including optional-row all-row checks, batch-size schema/list membership, planned-status, returncode, duration, output-tail/condition-field, and `git_dirty` dry-run lifecycle bindings including optional-row all-row checks, and `--rows` binding, artifact/`--json`, direct-symlink, parent-component, and symlink-parent rejection, output-dir containment, and category-filename binding, model/fixture argv binding, optional INT8/GGUF `--rows`/batch-size bindings, INT8 device-env, `--rows`/batch-size, model/fixture, primary artifact/`--json`, stale/missing/blank auxiliary artifact path, and duplicate diagnostic-flag bindings across all optional rows, GGUF device-env and primary artifact/`--json` bindings, template fixture/backend/quant/decode stale/missing/blank/duplicate flags across all optional rows, and all-c>1 quant ordering, workload-shape argv binding, cache/compiler/summary/profiler cached-build bool and trace/kernel-name-padding/source path-safety binding including profiler-source alias typed non-empty/parent-component/direct-symlink/symlink-parent/parent-file rejection, and retained gate/precondition/primitive+scaling-alias argv/path-safety binding including typed non-empty, blank, parent-component, direct-symlink, symlink-parent, and existing-directory alias regressions, the primitive
      `primitive-c1/c2/c4/c8.json` command text, argv,
      `--rows`, `--seed 1234`, `--json` labels, typed top-level summary schema/exact keyset, commands collection schema/per-category planned-row keysets and extra-key rejection, typed status label, typed non-empty timestamp, typed git provenance object/keyset/status list, typed dry-run mode, typed output-dir root, typed non-empty unique batch-size plan/order,
      top-level options object schema/keyset, typed optional diagnostic include flags/count toggles, optional INT8/GGUF row-count and category-rollup invariants, stop-on-failure and seed options, typed cached-build and compiler-version-file options, typed projection-dispatch artifact option, typed model/fixture workload labels, typed workload shape options, typed command/exact completed-command count bounds, typed status/category/condition rollup containers/kind-status labels/non-bool/non-string/non-negative/exact-match counts plus optional-row rollup drift checks, typed empty skipped/failed rollup containers/object entry keysets/value matching, and rejects
      missing/empty/blank/stale per-category command text/argv including optional-row all-row checks, wrong device env including INT8/GGUF all-row optional checks,
      missing/blank/unknown/known-mismatched/wrong script/category including optional-row all-row checks, per-category missing/malformed/unlisted
      `batch_size` including optional-row all-row checks, missing non-baseline argv `--batch-size`/`--rows` labels, c=1 native-baseline batch labels, per-category missing/unknown/non-planned dry-run status including optional-row all-row checks,
      per-category missing/malformed/non-null returncode, per-category missing/malformed/non-finite/nonzero duration,
      per-category output-tail/all condition-field variants,
      per-category unknown command schema keys and missing/malformed/stale `git_dirty`, all including optional-row all-row checks, per-category duplicate/blank/missing
      `--json` labels including optional-row all-row checks, missing/mismatched optional INT8/GGUF category counts including all-row status/category rollup drift, optional INT8/GGUF `--rows`/batch-size mismatches including INT8/GGUF all-row checks, optional INT8/GGUF artifact/`--json` desyncs including INT8/GGUF all-row primary artifacts, optional INT8/GGUF c-specific artifact filename mismatches including INT8/GGUF all-row checks, outside-output-dir optional INT8/GGUF artifacts including INT8/GGUF all-row checks, missing optional INT8/GGUF planned rows across all optional rows, stale INT8 model/fixture labels, missing/blank/stale-c/stale-name optional INT8 auxiliary artifact flags across all optional rows, and duplicate INT8 diagnostic flags, plus missing/blank/stale/duplicate GGUF template flags, duplicate/swapped optional GGUF quant rows across each c>1 group,
      duplicate `--rows`/`--seed` labels,
      blank inline/missing/non-integer `--rows`/`--seed` values,
      CLI/summary parent-directory/symlinked/symlink-parent/non-directory-parent `output_dir`, missing/blank/non-string
      `artifact_path`, parent-directory artifact paths, direct-symlink artifacts, symlink-parent artifact paths, non-directory artifact parents, and existing-directory artifact paths including non-optional and optional all-row checks, or
      outside-output-dir artifacts, wrong artifact filenames,
      artifact-path/argv `--json`, `--rows`, or `--seed` labels in
      `test_batch_c_sweep_can_plan_combined_int8_and_gguf_diagnostics`.
      The primitive script also re-runs
      the batched KV append and batched full-attention context kernels on the
      same inputs and emits A/A determinism fields (`append_batch_aa_*` and
      `attn_batch_aa_max_abs`) that must be zero for `passed=true`; retained
      bench loading, accepted-artifact schema validation, and c-sweep retained
      preconditions now reject primitive gate artifacts missing those zero A/A
      fields. Primitive correctness
      artifacts also stamp the visible HIP device/env (for example GPU1/XTX via
      `HIP_VISIBLE_DEVICES=1`) so GPU re-baseline runs carry hardware provenance
      directly in the JSON; retained-bench loading, accepted-artifact schema,
      and c-sweep retained preconditions now reject accepted/passing primitive
      gates without valid device metadata, and accepted artifacts require that
      primitive device name to match `hardware.gpu` (with retained-bench hardware
      context resolved from the visible HIP device). Retained-bench command
      labels also preserve `HIP_VISIBLE_DEVICES=1` as an `env` prefix for the
      benchmark, primitive correctness reference, and profiled command; the c=1
      baseline artifact self-binds `artifact_path` and its benchmark label,
      HIP/ROCm hardware metadata, and software dirty-state plus the serial-bridge baseline benchmark
      label/hardware metadata preserve the same visible GPU1/XTX provenance;
      retained-bench scaling and c-sweep
      preconditions reject scaling references whose concurrency or
      prompt/decode shape labels mismatch the retained workload, whose status is
      unusable for scaling, whose reason field is non-null, whose native or
      baseline rates are non-positive/non-finite, whose aggregate rate is inconsistent with per-request rate times concurrency, whose visible-device name conflicts with `hardware.gpu`, whose visible-device env is absent despite a matching benchmark command env, whose visible-device env conflicts, whose device-stamped artifact lacks software provenance, whose benchmark command label is absent, whose benchmark command env is missing/conflicts with the retained command env, or whose command launches the wrong baseline script. Schema and c-sweep launch validators
      accept that prefix while still requiring the retained benchmark script
      after it. Accepted artifact validation now also
      requires those command prefixes whenever retained hardware/primitive device
      metadata stamps `HIP_VISIBLE_DEVICES`, so GPU1 re-baseline JSON cannot
      strip the device-selection env from reproducibility commands. The c-sweep
      planner also prefixes planned/executed subprocess argv with the visible HIP
      device env, so summary JSON records the actual GPU1 selection instead of
      relying on inherited shell state; retained-bench, c-sweep, primitive
      correctness, strict artifact/validation-summary JSON input and output,
      hidden-bisect diagnostics/output cloning,
      hidden-artifact comparison input/output/record canonicalization, INT8/GGUF
      diagnostic templates, c-sweep/retained-bench/fixture input/output JSON,
      serial/packed/sparse correctness smokes, and serial-bridge baseline JSON
      serializers now reject non-finite values instead of emitting `NaN`/`Infinity`; c-sweep, retained-bench, and
      accepted-artifact gates reject profiler/correctness command labels that
      drop or change that env prefix. The c=2/c=4/c=8 profiler preflights now record
      compact native-batch kernel timing, including `batch_argmax_stage{1,2}`;
      the next retained/scaling step is graph-replay profiler evidence and native
      batch linear/full-attention/projection closure without non-retained per-row
      correctness fallbacks.
- [x] **C2.6 slot-validation and long-context fallback guards.** Add CPU
      structural tests for invalid slot orders/duplicates/out-of-range ids,
      INT8 KV rejection, and the current `max_context >= 1024` per-row split-K
      fallback until row-aware split-K is live. Acceptance: tests fail if the
      experimental path silently accepts unsupported shapes or routes long
      contexts through a false native c>N reducer. Evidence:
      `test_qwen35_resident_step_batch_native_rejects_invalid_sparse_slots`,
      `test_qwen35_resident_step_batch_native_rejects_int8_kv_when_experimental`,
      `test_qwen35_resident_step_batch_native_accepts_long_context_for_splitk_fallback`,
      `test_qwen35_resident_run_layers_batch_decode_uses_per_row_splitk_fallback_for_long_context`,
      and `pytest -q tests/test_qwen35_resident_batch_layout.py -q`.
- [ ] **C2.7 row-aware split-K full attention.** Make full-attention decode
      and reduction consume per-row spans for `max_context >= 1024` before any
      long-context c>N claim. Acceptance: primitive correctness plus a
      generated-token diagnostic at a long-context shape. Progress: the host
      long-context rejection is removed and split-K contexts now route through
      the existing per-row split-K fallback, with `/tmp/hipengine-hidden-bisect-L4-1024-1-splitk.json`
      showing reduced L4 1024/1 generated-token and hidden equality vs
      independent c=1. Batch execution metadata now records
      `decode_execution.full_attention_decode_path=per_row_splitk_fallback` and
      forces `native_caware_decode=false` when that fallback is used; the retained
      bench payload mirrors that execution flag, and retained-bench plus accepted
      artifact schema now require `decode_execution.full_attention_decode_path=native_batch`,
      `decode_execution.native_caware_decode=true`, and both aggregate and per-layer
      full-attention contexts below 1024, so artifacts cannot overclaim
      long-context native decode by hiding split-K in per-layer traces. The item remains open until the split-K reducer
      itself is row-aware/native c>N.
- [x] **C2.8 append-only block-id contract.** Prevent block ids from changing
      backing pointer during a live request; add a debug/memory-audit test.
      Acceptance: the test would fail on pointer mutation or id reuse.
      Evidence: `FixedPagedKVPolicy(...).register(block_pointer_map=...)` plus
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C2.9 live admission cap.** Make `KVPolicy.admission_cap()` return
      current free capacity rather than startup capacity. Acceptance: fake
      policy/scheduler tests show reclaim changes admission capacity before the
      next admit. Evidence: `pytest -q tests/test_kvcache_policy.py -q`.

### C3 packets — widen kernel/model coverage

**Decode throughput packets (2026-06-07 dispatch-count profile).** Native c>1
decode is dispatch-bound (917/5370/9362 dispatches per step at c1/c4/c8; c1 is
already <=~37% GPU-utilized *with* graph replay on). See "Decode performance
levers" above and
`benchmarks/results/2026-06-07-hipengine-qwen35-decode-dispatch-profile-484/`.

- [x] **C3.0a Per-step host-overhead trim.** Cache the static `(rows,slots)`
      block table in `_batch_full_spans` (skip the per-layer-per-step host
      build + synchronous host->device copy) and reuse persistent device
      token/position scratch instead of malloc/free per step. Acceptance:
      generated tokens byte-identical (c4/c8) / c2 `serial_lm_head` stable 137,
      decode tok/s up. Evidence:
      `benchmarks/results/2026-06-07-hipengine-qwen35-decode-host-trim-485/`
      (c8 192->230 tok/s +19.9%, c4 +15.9%, c2 +14.7%, c1 unaffected).
- [x] **C3.0b Wire c-aware (c>1) decode graph replay — LANDED, correctness
      GREEN; throughput small/neutral (the regime is device-dispatch-bound, as
      the bounded-headroom note predicted).** The native batch decode step is
      now device-resident (token feedback via `batch_lm_out_index`, device
      batched LM-head argmax, on-stream `advance_decode_positions_i64`,
      persistent per-`(rows,slots)` block tables + segment metadata), and
      `capture_batch_decode_graph` / `Qwen35ParoBatchDecodeGraph` capture one
      step and replay it per token like the c=1 path.
      *Result (2026-06-08).* Replayed tokens **byte-identical to the eager
      device-resident reference at c2/c4/c8** (`scripts/qwen35_batch_decode_graph_smoke.py`,
      two fresh sessions because PARO linear-attn recurrent state advances each
      step); eager-default and device-resident `native_batch_vs_independent_c1`
      stay green (0 mismatches, c4/c8). Back-to-back replay vs eager (decode 32,
      single-run): c2 +1.5%, c4 +2.0%, c8 −0.5% — small/neutral, because the
      eager device-resident path is already host-lean after C3.0a + C3.0b-1, so
      replay only removes residual host-side per-dispatch launch latency, not
      the per-dispatch **device** overhead the c>1 regime is bound by (matches
      the c=1 ~37%-GPU-utilized finding and C3.0c's null bandwidth result).
      Pieces delivered: C3.0b-1 persistent segment metadata (cu_seqlens/state),
      A device-token-fed batch embedding (`_set_batch_token_embeddings_from_ptr`),
      B device batched-LM-head argmax write (`_write_batch_next_tokens_device`),
      C/E `_step_batch_from_device_tokens` (+ `step_batch_native device_resident`
      mode + `--device-resident-decode`), D `capture_batch_decode_graph`.
      *Follow-ups (deferred, lower priority):* multi-rep retained-bench replay
      throughput medians (the device-resident decode path can now feed a
      replay-driven retained row); longer-context / ≥1024-ctx split-K capture
      (current gate is <1024 non-split); device-resident per-step token
      recording kernel to drop the replay_collect host readback.
- [~] **C3.0c Output-column-tiled c>1 GEMM kernel family — DONE for the dense
      pack8 projections (single+dual), but a NULL end-to-end throughput result
      (the regime is dispatch-bound, not bandwidth-bound — see Result below).**
      Replace the c>1 GEMV-per-column projection/MoE path
      with an output-column-tiled GEMM that loads each weight tile once and
      reuses it across all `c` columns (caching the quantized activation once),
      plus per-expert token batching for the MoE GEMMs. This both cuts decode
      dispatch count and amortizes weight loads, and is the only lever toward
      the c>1 roofline (`docs/ROOFLINE.md` §3.2). Treat as a standalone
      kernel-family workstream with its own RED/GREEN correctness gate + rocprof
      evidence, **not** part of the graph-replay change. Acceptance: per-step
      decode kernel time scales sub-linearly with `c` (weight reuse measurable
      in rocprof), generated tokens unchanged, aggregate decode tok/s improves
      materially vs C3.0b.

      *Design (scoped 2026-06-07).* The current c>1 projection/expert kernel is
      `gemv_awq_selected_dual_pack8_strided_kernel`
      (`kernels/hip_gfx1100/quant/paro_awq_gemv.hip`). Its grid is
      `(out_pack, x_rows*lanes_per_x_row)` and `x_row = row / lanes_per_x_row`,
      so each x_row gets its own blocks that **re-stream the full `qweight`** —
      the weight is read `c` times for `c` columns. Output-tiling = one block per
      output pack loads each weight tile once and accumulates all `c` columns
      (writing `c` outputs), so weight bytes/token stop scaling with `c`.
      Two sub-problems, do dense first:
      (1) **Dense projections** (attention QKV/O, linear-attn in/out, router,
      shared expert; `selected[row]` is a single shared weight) amortize cleanly
      across the `c` columns — the tractable first win.
      (2) **MoE expert GEMMs** are the active-weight bulk but route divergently
      (c=8 can touch ~64 experts), so amortization only applies within an
      expert's gathered tokens (grouped_compact already sorts by expert); the
      per-expert GEMM must batch its small `M_e` decode tokens. Note the parent
      `~/amd-gpu-tuning/MOE_KERNEL_DESIGN.md` covers **prefill** large-M MoE and
      states decode uses the M=1 kernel, so there is no direct port — small-M
      (c=2..8) decode GEMM is new kernel-family R&D. Per AGENTS.md that R&D
      belongs in `~/amd-gpu-tuning/` (rocprof iteration, device-code gotchas);
      hipEngine's side is the CPU-reference correctness gate (KL<=0.05 /
      top-1>=90% vs `kernels/cpu_reference/`, plus byte-identical generated
      tokens vs the current GEMV path) and the port once a stable kernel exists.

      *Progress (2026-06-07) — in-repo (fork boundary changed; kernel work now
      done in this tree, AGENTS.md left unedited to avoid merge conflicts).*
      - **Single pack8 output-tiled GEMV: DONE + wired + exercised.** New
        `gemv_awq_pack8_output_tiled_kernel<scalar_t, qweight_transposed, int C>`
        (`paro_awq_gemv.hip`): grid `(out_packed)`, one block loads each weight
        pack once into `acc[C][8]` registers and FMAs all C columns; reduction
        order identical to the per-row kernel so it is **byte-exact**. Templated
        C in {2,4,8}, bf16+fp16, both layouts (strided + transposed). Gate
        `tests/test_paro_awq_output_tiled_gemv.py` **96/96 PASS**.
      - **Decode uses the TRANSPOSED layout.** A projection-call-count probe
        (c2, native decode) shows per-step: dual_transposed 2200,
        pack8_transposed 1800, marlin_k 1600, selected_dual_transposed 1200,
        selected_transposed 1200. The strided `.qweight` single GEMV is NOT on
        the decode hot path; `project_pack8`'s transposed branch and the
        QKV/Z (`force_gemv`) + shared-down (`use_batch_gemv`) sites are.
      - Wired the transposed single output-tiled into those hot sites
        (`tokens in {2,4,8}`, env kill-switch `HIPENGINE_DISABLE_PACK8_OUTPUT_TILED`).
        Probe confirms `gemv_awq_pack8_output_tiled_transposed_fp16` = 1000
        calls/step (per-row 1800→800). Generated tokens byte-identical ON vs OFF
        e2e. Throughput NOT claimed (crude wall A/B noise-dominated; needs proper
        protocol; single-projection is a small ~5.3 ms/step slice).
      - **DUAL pack8 output-tiled: DONE** (`gemv_awq_dual_pack8_output_tiled_kernel`,
        transposed+strided × bf16/fp16; gate 192/192). Wired into the dual decode
        hot sites (linear-attn KV/QK, full-attn QKV+Z, shared gate+up); probe shows
        400 dual output-tiled calls/c2-step (per-row 2200→1800; rotate-fused + some
        linear-attn dual paths still on other kernels).
      - **Result (2026-06-08) — NULL throughput, dispatch-bound.** Warmed decode
        tok/s ON vs OFF (`HIPENGINE_DISABLE_PACK8_OUTPUT_TILED`), per-c prompt
        (prompt×c=512), decode 128, median of 3: c2 129.1→130.5 (+1.1%), c4
        178.7→176.0 (−1.6%), c8 203.4→203.0 (−0.2%) — all within run noise.
        Output-tiling cuts weight **bandwidth** but not **dispatch count** (both
        per-row and output-tiled are one launch per projection), and c>1 decode
        here is dispatch-bound, so it does not move throughput. The kernels are
        correct + a real bandwidth reduction and are kept (kill-switch-gated);
        their payoff requires C3.0b dispatch-count reduction (graph replay +
        fusion → bandwidth-bound regime) and/or longer contexts. Artifact:
        `benchmarks/results/2026-06-08-hipengine-qwen35-output-tiled-gemv-c2c4c8-decode/`.
      - **Deferred follow-ups (lower priority than C3.0b):** re-measure at longer
        contexts (agentic-focus cases) once a ≥2048-token fixture / multi-prompt
        path exists, full c-sweep; extend output-tiling to the rotate-staged dual
        and the selected (MoE) transposed kernels.

- [ ] **C3.1 INT8 KV c>N parity.** Validate batched INT8 KV append/decode
      end-to-end with the same generated-token gates as BF16. Acceptance:
      c=2 512/128 INT8 artifact is equality-green or explicitly
      `rejected_correctness` with first mismatch. Progress:
      `scripts/qwen35_batch_int8_diagnostic.py` emits the schema-checked
      blocked template `/tmp/hipengine-int8-c2-diagnostic.json` with the future
      retained-bench command and explicit blockers (`compact c>N native prefill`
      and `step_batch_native` INT8 rejection). The c-sweep planner includes this
      template behind `--include-int8`, producing an `int8_native_diagnostic`
      command in `/tmp/hipengine-c-sweep-int8-plan/summary.json`; dry-run summary
      tests now assert `options.include_int8=true`, `command_count=7`, and an
      `int8_native_diagnostic` category count. The blocked template also records
      the CPU-reference and HIP-required primitive layer-accuracy commands from
      `scripts/qwen35_kv_int8_accuracy.py` for prompt/context-boundary
      `512,513`, so C3.1 handoff artifacts name both the generated-token gate
      and the lower-level INT8 KV accuracy gate before promotion; c-sweep
      standalone and combined `--include-int8` rows now bind HIP device env
      prefixes, model/fixture labels, row/artifact labels, and c-specific
      future-retained, CPU primitive, and HIP primitive JSON paths and reject
      stale labels, duplicate INT8 model/fixture/diagnostic flags, or summaries
      where the `int8_native_diagnostic` rows no longer agree with
      `options.include_int8`;
      combined
      `--include-int8 --include-gguf` dry-run coverage locks the c=2/c=4/c=8
      INT8 rows, artifact filenames, HIP device env prefixes, model/fixture
      labels, future-retained JSON labels, CPU/HIP primitive evidence labels,
      and row labels before each GGUF quant fan-out, including stale c=8
      model/fixture/future/primitive/artifact/row-label/env-prefix rejection
      plus duplicate c=8 INT8 model/fixture/diagnostic flag rejection. The future
      retained-bench command now carries `--int8-kv-primitive-cpu-json` and
      `--int8-kv-primitive-hip-json`, `scripts/qwen35_kv_int8_accuracy.py`
      self-describes written JSON paths, and accepted c>N artifact schema
      requires loaded, self-matching CPU-reference plus HIP `--require-int8-hip`
      INT8 primitive layer-accuracy evidence whose retained-bench/profiler
      command flags are bound to those artifact paths before any
      `int8_per_token_head` retained row can validate. The INT8 diagnostic
      template now also preserves `HIP_VISIBLE_DEVICES` in the future retained
      command and both primitive layer-accuracy commands, with
      `test_int8_cN_diagnostic_template_records_blocked_c2_gate` covering the
      GPU1/XTX command-label handoff. The item remains open because
      blocked-before-execution is not an accepted C3.1 terminal status.
- [x] **C3.2 per-row `KVLiveSpans` everywhere.** Audit full-attention decode,
      KV append, and storage-dtype wrappers for scalar `(block_table,
      context_len)` shortcuts. Acceptance: tests cover BF16 and INT8 per-row
      spans. Evidence: BF16/INT8 c>N `FixedPagedKVPolicy.batch_spans(...)`
      plus paged-KV-write/full-attention dispatch route checks in
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C3.3 linear-attention `[C]` state.** Remove c1 aliases from
      conv/recurrent state update paths and use active masks + slot ids.
      Acceptance: c=2 state fixtures compare against two c=1 references.
      Evidence: `_run_layers_batch_decode(...)` passes whole `[C]` conv/
      recurrent state plus `state_indices` slot ids, and
      `pytest -q tests/test_qwen35_resident_batch_layout.py -q` covers a
      c=2 `(slots=(0, 2))` state-index fixture against `_slot_linear_state(...)`
      c=1 reference views.
- [ ] **C3.4 c-aware projection dispatch.** Keep c=1 on GEMV/Marlin-K while
      routing c=2/4/8 to MMQ/GEMM/WMMA candidates only when they beat row-GEMV.
      Acceptance: dispatch tests prove thresholds and benchmark artifacts show
      aggregate/per-request ratios. Progress: `hipengine.dispatch.projection`
      now exposes a tested c-aware projection policy: c=1 is pinned to row-GEMV,
      c>N candidates must name a non-row-GEMV projection kernel and require accepted benchmark evidence with aggregate and
      per-request speedups over row-GEMV, missing/slow/rejected/self-row-GEMV evidence falls
      back to row-GEMV with explicit blockers, and
      `ProjectionDispatchEvidence.from_json_dict(...)`,
      `ProjectionDispatchCandidate.from_json_dict(...)`,
      `projection_dispatch_candidates_from_json(...)`,
      `projection_dispatch_candidates_from_artifact(...)`, and
      `plan_projection_dispatch_from_artifact(...)` schema-check retained
      artifact candidate/evidence lists before the policy can consume them, including rejecting row-GEMV-named retained candidates and duplicate candidate names so `selected_candidate` cannot be ambiguous;
      projection speedup evidence must reference a non-symlink regular artifact under
      `benchmarks/results/` whose resolved target stays inside the active
      results tree (with symlink parents rejected), retained promotion rejects non-JSON/non-regular/symlinked evidence artifacts before scoring candidates, and, when accepted, must beat row-GEMV on both
      aggregate and per-request ratios; accepted c>N artifact schema rejects
      malformed optional `projection_dispatch_candidates` metadata plus non-JSON/non-regular/symlinked projection evidence paths; accepted c>N
      artifact schema now requires `execution.batch_execution.projection_dispatch`
      to name an evidence-backed non-row-GEMV c-aware path whose selected
      candidate is present in `projection_dispatch_candidates` and profiler
      expected/trace/duration kernel names; retained native batch metadata records a `projection_dispatch` row-GEMV fallback
      with an explicit blocker when no c-aware projection candidate is available, can load optional `HIPENGINE_QWEN35_PROJECTION_DISPATCH_ARTIFACT` retained candidate metadata when available only if candidate evidence artifacts are loadable/accepted/self-describing and in-bounds, and retained-bench/c-sweep can pass that metadata with `--projection-dispatch-artifact` while fail-closing non-`benchmarks/results/`, symlinked/non-regular, invalid, or missing candidate artifacts plus missing/out-of-bounds candidate evidence artifacts before an expensive run, validating that c-sweep summaries keep the flag on c>N retained commands and off c=1 baselines, and copying schema-checked candidates into the artifact payload; retained bench now blocks promotion before schema validation unless projection dispatch names an evidence-backed non-row-GEMV c-aware candidate present in `projection_dispatch_candidates` with matching row bounds, selection, retained artifact path, an accepted same-row evidence artifact JSON carrying self-matching `artifact_path`/`source_artifact_path` plus matching >1 aggregate/per-request row-GEMV speedup ratios, evidence, and profiler expected/trace/duration kernel names; the 2026-06-03 c2 projection artifact validates the `gemv_awq_selected_dual_pack8_strided` candidate (`1.0814x` vs selected-c1/row-GEMV projection) with profiled runtime metadata selecting `benchmark_accepted_caware_projection`; the c4/c8 projection artifact validates row-bounded candidates for the same variant (`1.2149x` at c4, `1.2927x` at c8) with runtime metadata selecting `benchmark_accepted_caware_projection` for both; and the combined c2/c4/c8 catalog proves one `--projection-dispatch-artifact` can select the matching row-bounded candidate while preserving full generated-token equality for c=2/c=4/c=8. The
      item remains open for retained aggregate scaling and full native c4/c8
      attention; c=2/c4/c8 projection dispatch now uses selected c-aware paths
      but is still not a retained throughput claim.
- [x] **C3.5 GGUF c>N template.** Port the Qwen/PARO equality template to
      GGUF Q4_K/Q5_K/Q6_K/Q8_0. Acceptance: at least one GGUF c=2 diagnostic
      reaches an unambiguous `eq_ok`, `blocked`, or `rejected_correctness`
      status with exact command. Evidence: `scripts/qwen35_batch_gguf_diagnostic.py`
      emitted `/tmp/hipengine-gguf-c2-diagnostic.json` with `status=blocked`
      and exact command `python3 scripts/qwen35_batch_gguf_diagnostic.py --fixture tests/fixtures/gguf/qwen35_0_8b_q4_k_m_e2e.json --rows 2 --backend hip_gfx1100 --quant gguf_q4_k_m --max-new-tokens 4`; the template also preserves
      `HIP_VISIBLE_DEVICES` in its native c>N and independent c=1 command labels
      for GPU1/XTX re-baseline runs. The c-sweep planner now has
      `--include-gguf`, which adds blocked GGUF c>N diagnostic commands for
      Q4_K_M/Q5_K_M/Q6_K/Q8_0 at c>1 while preserving and binding the visible
      HIP device env, and rejects stale GGUF fixture/backend/quant/decode-length/row
      labels plus stale env-prefix labels across all four standalone quant rows,
      duplicate GGUF diagnostic flags across all four standalone quant rows,
      duplicate/swapped/unsupported quant order, artifact-path/argv `--json`
      links and artifact filename metadata across all four standalone quant rows, or
      summaries where GGUF diagnostic rows no longer agree with
      `options.include_gguf`; combined
      `--include-int8 --include-gguf` dry-run coverage locks the c=2/c=4/c=8
      four-quant artifact filenames, HIP device env prefixes, fixture/backend
      labels, and decode-length labels
      after each INT8 row, including stale c=8
      fixture/backend/decode/quant/artifact/env-prefix rejection and duplicate
      c=8 GGUF fixture/diagnostic flag rejection;
      covered by `test_gguf_cN_diagnostic_template_records_blocked_c2_command`
      and `test_batch_c_sweep_can_plan_gguf_blocked_diagnostics` in
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] **C3.6 native LM-head/sampler launch.** Replace the per-row
      `serial_lm_head` loop with a native row-aware LM-head/argmax only after
      C2 equality is green. Acceptance: c=2/4/8 equality stays green with
      `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=batched_lm_head` or successor.
      Progress: `hipengine.dispatch.sampling` now gates `batched_lm_head` behind
      explicit c>N generated-token equality evidence and a retained artifact path
      under `benchmarks/results/`; `_sample_batch_from_hidden(...)` records the
      sampler decision and falls back to `serial_lm_head` when evidence is
      missing, failed, wrong-row, reports serial sampler metadata, points outside retained artifacts, or resolves
      through a symlink outside the active results tree, and accepted c>N artifact
      schema requires a native sampler decision with requested mode `batched_lm_head`, row count and equality row count matching `workload.concurrency`, green retained equality
      evidence plus no blockers, and dispatch/retained bench now block promotion before schema validation unless sampler metadata records an explicitly requested native row-aware batched LM-head decision with matching rows/equality rows, a retained equality artifact whose core dispatch, retained-bench, and accepted-schema path loaders reject non-JSON/non-regular/symlinked evidence before reading and whose JSON reports non-blank self-matching `artifact_path`/`source_artifact_path` plus generated-token equality vs independent c=1 (`passed=true`, `skipped=false`, matching non-empty typed integer batch/c1 sequence lists, empty mismatches) at the same row count, matching profiler expected/trace/duration evidence for a native batch sampler/LM-head kernel that is not serial/per-row/fallback, and no blockers, so setting the mode cannot silently create a
      native sampler claim before same-concurrency equality and profiler evidence are green. The retained bench now exposes explicit
      `--batch-sample-mode/--batch-sample-eq-*` flags, and GPU1 / RX 7900 XTX
      c=2/c=4/c=8 512/128 runs with `--batch-sample-mode batched_lm_head`
      stayed generated-token equal (all rows prefix 137) using same-row sampler
      equality artifacts under `benchmarks/results/2026-06-02-hipengine-qwen35-c{2,4,8}-native-batch-sampler-equality.json`.
      After the retained-bench default-evidence change, the no-flag c=2/c=4/c=8
      equality matrix is also green with row-aware sampler metadata
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-default-sampler-equality-matrix/summary.json`).
      A follow-up output-labeled no-flag matrix keeps c=2/c=4/c=8 equality
      green while recording the active linear-output route as `batch_gemv`
      in the summary (`benchmarks/results/2026-06-02-hipengine-qwen35-native-default-output-labeled-equality-matrix/summary.json`).
      A two-repeat no-flag matrix also passes all six c=2/c=4/c=8 runs
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-repeat2-equality-matrix/summary.json`),
      but the loop primary verifier still produced one transient c=2 prefix-82
      sample before two no-code-change reruns returned 137, so deterministic
      stability is still not claimed. The compact fingerprint artifact
      `benchmarks/results/2026-06-02-hipengine-qwen35-native-c2-primary-variability/summary.json`
      preserves the prefix-82/prefix-104 mismatch windows and confirms the
      failing sample still used `batched_lm_head` plus `batch_gemv` output.
      The explicit `batched_lm_head` control matrix with matching same-row
      sampler equality artifacts also passes c=2/c=4/c=8
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-explicit-batched-sampler-control-matrix/summary.json`),
      confirming the non-defaulted native sampler path remains green. The
      explicit `serial_lm_head` sampler control matrix also passes c=2/c=4/c=8
      (`benchmarks/results/2026-06-02-hipengine-qwen35-native-serial-sampler-control-matrix/summary.json`),
      isolating the intermittent primary issue toward row-aware sampler
      stability or sampler/decode interaction; neither control is a retained
      throughput/scaling claim.

### C4 packets — continuous scheduler and dynamic KV pool

- [x] **C4.1 engine-loop skeleton.** Introduce long-lived
      `submit/poll/cancel` driver around existing resident sessions, initially
      using fake/CPU tests and the serial bridge. Acceptance: requests can be
      admitted, decoded, finished, and reclaimed without a one-call lifetime.
      Evidence: `hipengine/generation/engine_loop.py` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.2 adapter migration.** Lower `LLM.generate()` and non-streaming
      server endpoints onto `submit+poll` while preserving current outputs.
      Acceptance: existing generator/server tests pass and prompt-list
      batching still routes by request id. Evidence:
      `SubmitPollTextGenerator` wraps resolved model generators in
      `LLM._get_text_generator()`, prompt order/row-seed coverage in
      `pytest -q tests/test_generation_batch_scheduler.py tests/test_llm_generate.py -q`,
      and server prompt-list batching remains covered by
      `pytest -q tests/test_server_api.py -q`.
- [x] **C4.3 tick policy.** Implement `RECLAIM → ADMIT → choose(PREFILL_CHUNK,
      DECODE_STEP)` with `protect_decode` default. Acceptance: scheduler tests
      cover decode protection and TTFT/fair alternatives. Evidence:
      `ResidentEngineLoop(prefill_decode_policy=...)` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.4 chunked KV pool.** Add chunked allocation, grow-on-admission,
      idle shrink, and high/low-water knobs behind fake-runtime tests first.
      Acceptance: burst+idle fixture records at least one grow and shrink or
      explicitly records that the initial chunk sufficed. Evidence:
      `hipengine/kvcache/pool.py` plus `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C4.5 pool/env docs.** Add CLI/env knobs for `HIPENGINE_KV_POOL_*` and
      `HIPENGINE_PREFILL_DECODE_POLICY` and document them in `docs/ENVS.md`.
      Acceptance: CLI/env tests and docs agree on defaults. Evidence:
      `add_engine_loop_config_args(...)`, `docs/ENVS.md`, and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.6 streaming through loop.** Route streaming completions through
      per-request token queues instead of bypassing the batcher. Acceptance:
      streaming and non-streaming share reclaim/cancel tests. Evidence:
      `_GenerationBatcher.stream(...)` per-request queues, single-row chat
      streaming routed through that batcher instead of `engine.stream`, and
      `pytest -q tests/test_server_api.py -q`.
- [x] **C4.7 unified reclaim.** Make cancel, disconnect, EOS, max-tokens, and
      timeout converge on one `RECLAIM` path. Acceptance: each finish reason
      frees KV/scratch exactly once in tests. Evidence:
      `ResidentBatchScheduler.cancel/disconnect/timeout(...)`, generated-token
      `stop`/`length` reclaim, and `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C4.8 non-compact-slot native decode.** Extend native decode beyond
      compact `0..C-1` slots after scheduler compaction/reclaim. Acceptance:
      generated-token equality passes with a deliberately sparse/compacted
      slot schedule. Evidence: `step_batch_native(...)` now accepts sorted
      sparse physical slots, `_batch_full_spans(...)` maps slot ids into
      row-relative KV block tables, `pytest -q tests/test_qwen35_resident_batch_layout.py -q`,
      and `python3 scripts/qwen35_batch_sparse_slot_correctness.py --json /tmp/hipengine-sparse-slot-L1.json`
      shows generated-token equality vs independent c=1 for a cancel-middle
      active slot history `[[0, 2], [0, 2]]`.
- [x] **C4.9 observability fields.** Record per-request and per-pool fields in
      completion/artifact metadata. Acceptance: tests assert queue/prefill/
      decode seconds, KV pages, bucket key, admission blocker, and finish
      reason are present. Evidence: `CompletedRequest.to_json_dict()`,
      `KVPoolStats.to_json_dict()`, accepted-artifact schema checks, and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.

### C5 packets — prefix sharing, per-row sampling, `n>1`, metrics

- [x] **C5.1 block refcounts.** Add block-id refcounts and reuse accounting.
      Acceptance: shared-prefix admission increments/decrements refcounts and
      reclaim only frees zero-refcount blocks. Evidence:
      `ChunkedKVPool.admit_with_shared_prefix(...)`, prefix reuse counters, and
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C5.2 RadixCache.** Implement the token-id trie with
      `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}`. Acceptance:
      prefix-hit/miss tests cover partial-block edges and cancellation.
      Evidence: `hipengine/kvcache/radix.py`, `HIPENGINE_PREFIX_CACHE`,
      `hipengine serve --prefix-cache`, and
      `pytest -q tests/test_kvcache_policy.py tests/test_server_api.py -q`.
- [x] **C5.3 copy-on-write fork.** Fork fresh pages at the first divergent
      token while preserving shared prefix pages. Acceptance: two diverging
      requests keep prefix bytes shared and produce independent suffix KV.
      Evidence: `ChunkedKVPool.fork_copy_on_write(...)`, COW fork counters,
      and `pytest -q tests/test_kvcache_policy.py -q`.
- [x] **C5.4 `n>1` lowering.** Replace API rejection with N scheduler
      requests sharing a prompt prefix and distinct seeds. Acceptance:
      OpenAI-compatible responses preserve `n` semantics and request IDs.
      Evidence: server completion/chat `n` lowering, distinct `row_seeds`,
      per-choice `request_id`, and `pytest -q tests/test_server_api.py -q`.
- [x] **C5.5 per-row sampler.** Land per-row temperature/top-k/top-p/
      repetition-penalty/seed/stop-token handling. Acceptance: incompatible
      sampling params decode together and deterministic seeds are stable.
      Evidence: `PerRowSamplingParams`, `SamplerParamsBlock`, and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C5.6 per-row EOS/reclaim.** Finish rows independently inside a batch.
      Acceptance: one row can finish while others keep decoding and its KV is
      reclaimed at the next commit point. Evidence:
      `ResidentBatchScheduler(reclaim_callback=...)` and
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] **C5.7 metrics endpoint.** Add Prometheus `/metrics` behind
      `HIPENGINE_METRICS` / `--metrics`. Acceptance: metrics are additive and
      include request, pool, and graph-bucket counters. Evidence:
      `ServerConfig(metrics="prometheus")`, `hipengine.server.__main__ --metrics`,
      `docs/ENVS.md`, and `pytest -q tests/test_server_api.py -q`.
- [x] **C5.8 retained-row enforcement.** Make the bench harness enforce gates
      for timestamps, p50/p95 latency, dynamic pool, stable block id, and
      prefix-sharing savings before `status=accepted`. Acceptance: a fixture
      missing any required field cannot be accepted. Evidence:
      `scripts/qwen35_batch_artifact_schema.py` accepted-row gates now also
      require non-skipped generated-token equality vs independent c=1 with exact
      `batch_sequences == c1_sequences` lists whose row count matches
      `workload.concurrency`, whose per-row token counts match seed +
      `workload.warmup_decode_tokens` + `workload.gen_tokens_per_request`, and
      whose seed prefixes and measured-decode suffixes match
      `execution.seed_tokens` / `execution.generated_tokens`, whose
      `execution.completed` rows cover every request, carry prompt-token counts
      matching `workload.prompt_lengths`, and match generated-token and
      finish-reason records, no mismatches, and a passing primitive c>N GPU
      correctness JSON whose self-reported `artifact_path` and `rows` values match
      the retained reference path and `workload.concurrency`, plus
      retained-bench allocator/memory evidence merge/blockers in
      `test_qwen35_retained_allocator_memory_evidence_from_stats`,
      `test_qwen35_retained_memory_payload_uses_bench_evidence`, and
      `test_qwen35_retained_memory_evidence_blockers_cover_required_fields`,
      and `pytest -q tests/test_generation_batch_scheduler.py -q`.

### Performance packets — run only after correctness is green

- [ ] **P1 baseline bundle.** Establish c=1, serial bridge c=2/4/8, first
      green uncaptured native c>N, and primitive microbench baselines.
      Acceptance: artifacts include exact commands, hardware, correctness,
      aggregate/per-request ratios, and dirty-state. Progress: retained native
      c>N artifacts now require explicit c=1 and serial-bridge baseline JSONs
      before `performance_claim=true`; `scripts/qwen35_batch_c_sweep.py` wires
      those paths into the planned retained command (`--c1-baseline-json`,
      `--serial-bridge-json`) and now also passes the matching
      `--primitive-correctness-json` path (`primitive-cN.json`) plus the planned
      `--profiler-json` path (`profiler-cN.json`) so a green generated-token run
      without primitive GPU correctness or captured profiler evidence remains
      blocked instead of becoming a throughput claim, and retained bench now blocks promotion before schema validation when profiler artifact/trace/command/profiler-json/output-json/warmup-inclusive workload-shape/model-fixture/cached-build/reference-artifact/KV-policy provenance, symlink-escaped retained artifact/reference/profiler paths, exact rocprof separator count, pre-separator rocprof executable/option binding/uniqueness, rocprof separator/profiled-command binding, post-separator retained-flag binding/uniqueness, self-contained artifact command labels, concrete profiler trace paths, all concrete profiler command-label validation, profiler-command generated-equality gating, trace kernel names, expected kernel names, explicit profiler capture-status and expected-kernel-present verdicts, unique native-batch profiler kernel-name lists, expected-kernel trace membership, positive kernel-duration evidence, total-duration arithmetic, per-kernel duration-share arithmetic, duration-category total/share arithmetic, or CPU-side bottleneck total/share arithmetic are missing or inconsistent. Real c-sweep runs now skip
      retained native diagnostics if the matching primitive, c=1 baseline,
      serial-bridge, or profiler-summary artifact is missing, failed, has a
      mismatched profiler artifact path, row count, or prompt/decode shape, or
      lacks required row/shape/kernel/CPU-side bottleneck labels; the c=1 PARO
      bench now emits a
      first-class `workload` object with `concurrency=1`, prompt/decode token
      counts, and KV policy, and retained scaling summaries carry c=1/serial
      baseline `status`/`reason`, `workload_concurrency`, prompt/decode
      labels, and shared required baseline/ratio key names, and the retained precondition records include the resolved baseline
      and profiler status/reason/c-sweep-precondition-and-schema-checked matching command/output-format/existing non-symlink+symlink-parent-free+directory-parent trace-dir/trace-file paths with readable kernel rows/exactly one kernel-trace CSV and trace-declared kernel-name/duration agreement/unique kernel-name/kernel-duration set-agreement/provenance fields (`profiler_trace_synthesized_fields` in c-sweep preconditions, mandatory post-run c-sweep retained-artifact cross-checks with validated persisted summary rollups and singular failed `postcondition` records, shared c-sweep/schema retained-bench unique/reference flags plus primitive/correctness-reference command flag allow-lists, shared c-sweep/retained-bench kernel-trace CSV column aliases, and shared schema/retained-bench/c-sweep synthesized-field allow-lists for `profiler.synthesized_fields` in retained artifacts; c-sweep and retained-bench load kernel names and durations from trace CSVs), structured retained/c=1/serial/primitive/profiler reference artifact paths (including retained-bench, c-sweep, and accepted-artifact-schema scaling-reference `reference_artifact_path` self-binding for c=1/serial source JSONs, retained-bench/source-schema `source_artifact_path` self-binding for primitive correctness and profiler JSONs with retained-bench profiler provenance blockers, c-sweep `profiler_source_artifact_path` precondition/postcondition self-binding, plus a c-sweep primitive precondition `primitive_artifact_path`, each validator-checked against the retained command's matching gate path),
      structured cached-build flags, structured model/fixture/run-shape labels, shared primitive-correctness schema/seed/shape/tolerance gates, aggregate/per-request rates, profiler native-batch-only schema-enforced stripped kernel-name duration/share keys plus shared schema/c-sweep/retained-bench profiler duration and CPU-side bottleneck category key sets, and
      schema-checked kernel-row-derived category totals/shares plus CPU-side bottleneck totals/shares, so c-sweep preconditions and artifact schema
      validation reject c>N rows compared against missing, failed/unusable,
      reason-bearing, ambiguous, or wrong-shape baselines; the sweep writes `command_count`,
      `completed_command_count`, an `options` block, per-retained-command
      `preconditions`, `status_counts`, `category_status_counts`,
      `retained_precondition_counts`, and `skipped_preconditions` summary rollups
      for planned/passed/skipped/failed rows, has persisted-summary coverage and summary-validator checks for
      typed validator/CLI root object/schema/version plus pre-run/persisted exact summary/option key sets, typed dry-run/run-option booleans/non-blank typed model-fixture/deterministic-seed/workload-shape labels, parseable timezone-aware timestamps, typed pre-run/persisted non-empty batch-size-list and exact dry-run planned/skipped/simple-executed-row key sets plus dry-run/skipped-row status/duration/output/condition/postcondition semantics, stop-on-failure terminal-row semantics, exact git provenance key set plus non-blank dirty-state/status provenance, non-empty command list, known command key set and command identity fields (including non-blank category/artifact-path/command/strict-python-executable/non-empty-primitive+retained-argv-flag-value/argv, retained-flag uniqueness, parent-traversal-free artifact-path/`--json` and artifact `output_dir`, non-symlink artifact-path/`output_dir`/parent containment plus category/batch filename identity, batch-size/argv, run-shape argv, category/script consistency, fully-passed summary artifact regular-file/non-symlink existence, finite execution-duration metadata, precondition/postcondition scope/path integrity (including retained gate argv presence/parent-traversal-free non-symlink filename bindings), and status/returncode/precondition/postcondition consistency (including non-skipped failed-gate and zero-return passed-postcondition rejection)), typed derived command counts/dry-run+non-stop completeness/order, known condition key set plus exact minimal-failed/primitive-correctness/scaling-reference/profiler-summary precondition and passed/failed retained-postcondition key sets plus failed retained source/synthesized-evidence-free malformed-source plus typed JSON output-dir-bound non-symlink/symlink-parent-free parent-traversal-free source-provenance/source-mismatch and synthesized-field evidence known/unique/precondition-binding/pairing/typing and condition-entry schema/non-blank failed-reason shape, primitive-command rows/seed plus primitive-correctness schema/seed/fixture-shape/typed-context-lens/NumPy-oracle precondition fields, profiler-shape/command-path/strict-command-executable/single-command-separator/bidirectional-rocprof-retained-option-placement/rocprof-option-value/rocprof-option-uniqueness/profiled-command/profiled-command-flags/profiled-command-flag-value/profiled-command-flag-uniqueness/command-kernel-trace-flag/command-output-format/artifact-ref/parent-traversal-free non-symlink serial+native matched compiler-version/cache-required build-cache option+argv/precondition-synth-field/trace-kernel/trace-kernel-uniqueness/expected-kernel-uniqueness/trace-path/trace-path-canonical-containment/parent-traversal-free trace-dir+trace-file/non-symlink trace-dir+trace-file parent-containment/trace-file-extension/trace-file-uniqueness/trace-duration/category-arithmetic/CPU-bottleneck-arithmetic, and CLI-reported non-blank typed pre-run output-dir/summary-json/compiler-version paths plus persisted summary output-dir/compiler-version paths plus model-fixture/typed deterministic-seed/workload-shape plus validate-summary-json path/symlink-parent and persisted scaling-label/rate/arithmetic checks, passed retained-row postcondition presence/reason-shape/synthesized-field precondition-binding checks, and retained native gate/postcondition-kind checks, typed status/category-status/precondition/postcondition rollups (including empty-command, unknown-count top/leaf-label, nonnegative-count, and bool-count tamper checks plus `qwen35_batch_c_sweep.py --validate-summary-json`), typed exact-key command-derived skipped-precondition/failed-postcondition rollups, and skipped retained rows retaining both the complete `preconditions` list, first-failed reason/bounded-output-tail evidence, and
      typed singular first-failed `precondition` / `postcondition` (rejecting stale/stray/type-drifted singular entries when no condition failed or no matching condition list exists, and binding failed-postcondition output tails to the postcondition reason), and has unit coverage confirming
      usable references allow the retained command to run. Accepted artifact
      schema and c-sweep preconditions share and reject baseline statuses known
      to be unusable for claims (`missing`, `invalid_json`, `failed`,
      `rejected`, and `rejected_correctness`) before a c>N row can be promoted.
- [ ] **P2 graph replay buckets.** Add decode hipGraph capture/replay buckets
      by `(C, context bucket, active mask, KV dtype, layer plan, top-k/experts,
      replay length)`. Acceptance: bucket hit/miss stats and profiler evidence
      show replay for common shapes. Progress: graph-bucket stats now serialize
      `entries`, `hits`, `misses`, `replay_hit_rate`, miss-reason counts, and typed-integer kernel-time
      histogram buckets from `GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS`; retained bench profiler summaries populate those buckets, invalid hit/miss/replay-rate stats, internally inconsistent entry coverage or miss-reason totals, plus missing or unknown-bucket histogram observations block promotion, and retained accepted-artifact schema requires non-empty known-bucket kernel-time
      histogram evidence plus those
      observability fields plus actual/accepted-schema-validated replay shape-key axes (`context_bucket`,
      `kv_storage_dtype`, `layer_plan`, `top_k`, `experts_per_token`,
      `replay_steps`, `draft_depth`, and `tree_shape`, with the context bucket
      covering the workload prompt length and the KV/layer axes matching the
      retained workload; per-request `bucket_key` observability strings include
      the same mode/C/context/mask, KV/layer, and top-k/expert/replay/draft
      axes, stale pre-axis strings block promotion, and retained/schema gates
      reject bucket-key shape, KV dtype, layer-plan, or decode-axis values that
      differ from the workload/scheduler shape; see `BatchShapeKey`,
      `_record_decode_graph_bucket_metadata`, the serial-bridge shape payload
      helpers,
      `test_batch_shape_key_includes_context_bucket_mask_and_mode`,
      `test_resident_scheduler_completion_observability_and_pool_counters`,
      `test_resident_scheduler_decode_bucket_key_uses_workload_axes`,
      `test_qwen35_retained_records_decode_graph_bucket_metadata`,
      `test_qwen35_retained_graph_replay_stats_blockers_require_hit_evidence`,
      `test_qwen35_retained_graph_replay_profiler_evidence_blockers_require_graph_duration`,
      `test_qwen35_retained_request_observability_blockers_cover_row_evidence`,
      `test_qwen35_batch_diagnostic_artifact_schema_enforces_accepted_row_gates`, and
      `test_qwen35_batch_serial_shape_key_payloads_include_workload_axes`), positive profiler `graph_replay` expected-kernel/duration/category/share evidence whenever replay hits are positive, and per-bucket histogram observation counts whose totals and bucket labels cover both replay hits and profiler kernel-duration evidence before a c>N row can be promoted; `/metrics` exposes a
      hit/miss-derived replay-hit-rate gauge plus labeled miss-reason and known kernel-time-bucket counters for live runs.
- [ ] **P3 remove residual serial loops.** Remove full-attention per-row
      fallback, per-row metadata allocation, per-row LM-head launches, and
      Python per-layer dispatch from steady-state native decode. Acceptance:
      profiler summaries show the removed bottleneck and equality remains
      green. Progress: accepted/performance-claim c>N artifact schema now rejects
      serial-bridge paths, non-scheduler-owned execution, non-full-native, wrong-path, wrong-layer-limit, or unsupported-layer-bearing prefill plans, non-empty batch/prefill/decode-execution blockers, row executions labeled `serial`/`fallback`, missing or wrong-shape
      decode-execution row/slot/context/layer-count plus grouped-compact MoE path/row/layer-count metadata, missing or stale per-layer decode traces for native full-attention/grouped-MoE layers, native-batch decode contexts at or beyond 1024 before row-aware split-K lands, non-`native_batch` full-attention decode paths,
      per-row full-attention decode fallbacks, non-native sampler metadata, sampler requested-mode mismatches, sampler row/equality-row mismatches, and failed or wrong-row sampler equality artifacts,
      runtime short-context native full-attention metadata now reports the retained-compatible `native_batch` path,
      and retained bench now blocks promotion before schema validation for the same serial/fallback batch/decode metadata,
      so residual serial loops cannot be promoted as retained rows while this
      item remains open.
- [ ] **P4 MoE/projection scaling.** Group routed lanes by expert and switch
      c=2/4/8 projections/MoE to kernels that beat row-GEMV. Acceptance:
      c=8 aggregate decode improves vs both c=1 and the serial bridge, with
      per-request ratios reported.
- [ ] **P5 retained scoreboard update.** Only after accepted artifacts exist,
      update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and compact
      JSON artifacts under `benchmarks/results/`. Acceptance: every perf claim
      cites correctness gate, profiler status, exact command, and hardware.
      Progress: accepted/performance-claim c>N artifacts now fail schema
      validation unless they include the retained-bench envelope (`schema=3`,
      `status=accepted`, `performance_claim=true`, `decision.accepted=true`,
      `decision.reason=correctness/protocol passed`, matching `run_tag`,
      timezone timestamp, canonical summary, matching `rows`, and the native-path
      plus split-K-scope `notes`), fully native scheduler-owned
      batch/prefill/decode-execution metadata with empty blockers, the known
      full-native prefill path, null unsupported-layer fields, positive native
      full-attention layer evidence, decode rows/slots plus grouped-compact MoE
      decode rows matching `workload.concurrency` with positive grouped-compact
      layer count and zero selected-c1 fallback layers, decode context covering
      `workload.prompt_tokens_per_request` while staying below the open row-aware
      split-K threshold, and prefill layer limits matching `workload.max_layers`,
      workload native prefill/decode flags set, workload scheduler labels matching the execution path, per-layer decode traces matching the global native full-attention and grouped-MoE layer counts, projection evidence artifact JSON reporting accepted same-row evidence with self-matching `artifact_path`/`source_artifact_path` and matching >1 aggregate/per-request row-GEMV speedup ratios, sampler requested mode `batched_lm_head` plus rows/equality rows matching `workload.concurrency` and sampler equality artifact JSON reporting self-matching `artifact_path`/`source_artifact_path` and the same rows with generated-token equality vs independent c=1 (`passed=true`, `skipped=false`, matching batch/c1 sequence lists, empty mismatches),
      generated-token equality sequence lists matching `workload.concurrency`
      rows and seed + warmup + measured decode token counts per row, with
      `execution.seed_tokens` / `execution.generated_tokens` matching the seed
      prefix and measured-decode equality suffixes, and `execution.completed`
      rows covering every request with prompt-token counts matching
      `workload.prompt_lengths` plus matching generated-token and finish-reason
      records,
      primitive GPU correctness reference typed script schema (`schema=1`), source `artifact_path` matching the retained reference path, typed row count matching
      `workload.concurrency` in both source and summary preconditions,
      reference-/c-sweep-gated typed deterministic `seed=1234` provenance,
      deterministic typed fixture-shape
      metadata (`block_size`, `max_context_len`, `num_q_heads`, `num_kv_heads`,
      `head_dim`) in summary preconditions, reference-/c-sweep-gated typed
      per-row `context_lens` fixture coverage,
      reference-/c-sweep-gated typed source zero append-mismatch counters,
      reference-/c-sweep-gated exact-zero batch-vs-c1 attention error, and
      reference-/c-sweep-gated finite nonnegative NumPy-oracle attention error ≤ 2e-5,
      full 40-layer workload labels with concrete model/quant/KV storage dtype
      plus matching KV policy metadata,
      aggregate token labels and per-row prompt lengths matching per-request
      shape times concurrency, full-row admission/completion/per-request
      observability with finite admission/completion timestamps, completion
      after admission, finite nonnegative per-row timing, matching row ids,
      and latency samples matching completion-minus-admission plus derived
      percentiles (`p50` median, `p95 >= p50`) for every row in
      `workload.concurrency`, memory batch/sequence/KV-policy metadata
      matching workload shape, finite nonnegative allocator peak bytes,
      dynamic-pool evidence plus finite nonnegative counters,
      stable block-id audit, and prefix-sharing savings,
      execution scheduler metadata with decode shape-key active mask
      length/count matching workload concurrency plus graph-bucket entry/hit/miss arithmetic, positive replay hits, matching replay-hit-rate, and positive profiler graph-replay expected-kernel/duration/share evidence, positive finite
      aggregate/per-request throughput whose c-sweep scaling precondition
      concurrency and run-shape labels are typed and match `workload`, and whose
      native scaling copy matches the primary measurements, all
      required positive scaling ratios that mathematically match usable same-shape
      c=1 and usable same-shape/same-concurrency serial bridge
      baselines, aggregate ratios vs both references that beat 1.0,
      accepted-artifact schema checks for positive finite throughput and
      decode-step timing samples,
      a retained-benchmark command starting with a Python invocation of
      `scripts/qwen35_batch_retained_bench.py` with a top-level artifact path under
      `benchmarks/results/` matched by the
      retained benchmark/profiler `--json` outputs, explicit `--model` /
      `--fixture` plus `--batch-size`, `--prompt-length`, `--decode-tokens`, and
      `--max-layers` matching workload shape fields and baseline/correctness
      reference paths matching the retained artifact payload,
      a correctness-reference command that names generated-token equality vs
      independent c=1, with an embedded Python invocation of
      `scripts/qwen35_batch_correctness.py` whose own argv carries only
      `qwen35_batch_correctness.py` flags with unique `--rows` / `--seed` / `--json`,
      `--rows` matching `workload.concurrency`, `--seed` matching
      `correctness.primitive_batch_correctness.seed`, and `--json` matching
      `correctness.primitive_batch_correctness.artifact_path`, and a concrete
      `rocprofv3 --kernel-trace` profiler command targeting
      `scripts/qwen35_batch_retained_bench.py` after the rocprof `--` separator,
      with unique rocprof-only flags (`--kernel-trace`, `--output-format csv`, `-d`) before
      that separator, and unique retained shape/artifact/reference/cached-build
      flags validated from the post-separator profiled command segment
      (`--model`, `--fixture`, `--batch-size`, `--prompt-length`,
      `--decode-tokens`, `--max-layers`, `--c1-baseline-json`, `--serial-bridge-json`,
      `--primitive-correctness-json`, `--profiler-json`,
      `--compiler-version-file`, `--require-cached-build`), typed profiler
      precondition workload/warmup/layer labels matching the retained command,
      benchmark/profiler `--json` outputs plus primitive/scaling/compiler-version
      artifact paths resolving under the current `benchmarks/results/` tree with
      no parent traversal spelling and explicit `.json` regular-file references
      with no symlink file or parent-directory components, and the retained bench can now attach a
      captured profiler summary via `--profiler-json` / `--profiler-command`,
      synthesize `profiler.total_kernel_duration_ns`,
      `profiler.kernel_duration_shares`, `profiler.kernel_duration_categories_ns`,
      `profiler.kernel_duration_category_shares`, and CPU-side bottleneck
      summaries from per-kernel durations and retained wall-clock timings when
      the summary omits them, and require `--profiler-json` to match
      `profiler.artifact_path`; accepted artifacts now schema-check retained-payload
      benchmark rollup declarations (`artifact_path`, matching `source_artifact_path`,
      `benchmarks/README.md`, and `benchmarks/CHANGELOG.md`) while the post-run `validate_cn_diagnostic_rollup_evidence` gate
      (or the CLI `python3 scripts/qwen35_batch_artifact_schema.py <artifact>
      --rollup-evidence --summary-json
      benchmarks/results/<artifact-stem>-rollup-check.json`) verifies live
      `benchmarks/README.md` and `benchmarks/CHANGELOG.md` both mention the
      retained artifact path, the README carries `Last updated: YYYY-MM-DD`,
      and the changelog carries a same-line dated `YYYY-MM-DD` artifact entry whose date matches README `Last updated` and includes numeric old→new metric plus percent-delta evidence before promotion, writes self-validating
      schema-versioned (`schema=1`) closed-key pass/fail regular `.json` summary file evidence (canonical relative or absolute current-repo paths) resolving under the current repo `benchmarks/results/` for both `--summary-json` writes and existing-file `--validation-summary` rechecks before filesystem write/read attempts (no parent traversal spellings/escapes, symlink targets/parents, external repo paths, non-file targets, or non-directory parents), binds all schema and rollup summary write/recheck relative paths to the retained/source artifact location with write- and recheck-specific diagnostics (including failed summaries with no retained `artifact_path`), clears `status`, `performance_claim`, and `benchmark_rollup` from failed summaries while rejecting stale success/rollup fields, keeps rollup metadata out of generic artifact-schema summaries (even while preserving passed `status`/`performance_claim` labels), requires passed summaries to keep `status=accepted` and `performance_claim=true` consistent, requires passed rollup summaries to assert `status=accepted`, `performance_claim=true`, and closed-key canonical README/CHANGELOG rollup metadata, rejects malformed/extra-key rollup metadata, requires every validation-summary source/retained artifact path (including rollup metadata copies) to be repo-relative under `benchmarks/results/` with no parent traversal spelling and passed/rollup-bearing summaries to carry a non-null retained `artifact_path` with no nested copied-prefix, and can
      recheck those summaries with `--validation-summary`,
      exact environment capture command entries for `rocminfo | grep -E 'Name:|gfx' | head -4`,
      `rocm-smi --showmeminfo vram --showuse --showtemp`, `hipcc --version`,
      `git rev-parse HEAD`, and `git diff --quiet`,
      concrete non-empty hardware `gpu`/`arch` fields with `gpu` identifying an
      AMD/Radeon/Instinct device and `arch` formatted as a
      `gfx*` architecture string plus successful
      `hardware.rocminfo`/`hardware.rocm_smi` capture objects whose commands
      include the retained capture fragments (`rocminfo | grep -E`, `Name:|gfx`,
      `head -4`, and the `rocm-smi --showmeminfo vram --showuse --showtemp` flags),
      whose `rocminfo` output includes a `Name:` marker plus the recorded arch,
      and whose `rocm_smi` output includes GPU and VRAM markers,
      clean full-commit software fields (`software.hipengine_dirty == false`)
      plus a non-empty `hipcc_version` string containing a hipcc/HIP/clang version marker,
      and captured profiler evidence with a
      `profiler.artifact_path` under `benchmarks/results/`, profiler trace
      files canonically contained under `profiler.trace_dir`, native batch
      expected kernel names and duration/share-map keys as non-empty strings (no
      serial/per-row/fallback labels) present with every duration-map entry,
      including extra trace-listed entries, carrying positive finite numeric evidence,
      `profiler.total_kernel_duration_ns`
      matching the duration-map sum, exact per-kernel duration-share keys/values
      matching `duration / total`, finite exact-key category duration/share buckets for
      attention/MoE/projection/sampling/graph/other, finite exact-key CPU-side bottleneck
      duration/share totals, plus an accepted non-row-GEMV
      `projection_dispatch` decision whose selected candidate is listed with
      matching retained speedup evidence and appears in profiler expected/trace/duration kernel names, plus native sampler/LM-head expected/trace/duration profiler evidence.
      The scoreboard item remains open until accepted
      artifacts exist and the benchmark rollups are updated.

## Phase ladder

The phase ladder is the ground truth for c>N progress. Each phase has a
"definition of done" and a checklist. Boxes that are checked have shipped
commits in `git log`; unchecked are open work.

### C0 — keep diagnostics honest

Definition of done: every c>N number on disk is unambiguously labeled
`serial_bridge`, `experimental`, or `retained`.

- [x] Generate c≤8 prompt fixtures for 512/128 and 4K/128 diagnostics.
- [x] Run c=1/2/4/8 primitive correctness on GPU0.
- [x] Run c=1/2/4/8 scheduler serial bridge diagnostics and record blocked
      status.
- [x] Promote `rejected_correctness` as a distinct status in the retained
      bench harness so failing-equality rows are not silently `blocked`.
- [x] Add a `hipengine bench c-sweep` subcommand that runs the full
      diagnostic sweep without copy/paste loops.
- [x] Ensure every diagnostic artifact distinguishes
      `workload.native_compact_prefill`,
      `execution.batch_execution.native_compact_prefill`,
      `native_caware_decode` (execution flag, not correctness),
      a correctness-pass/status field, and `throughput_claim_eligible`.

### C1 — server and generator integration

Definition of done: prompt-list and short-window HTTP coalescing reach the
batch generator; `n>1` rejected; streaming unchanged.

- [x] Batch-capable Qwen/PARO generator path for prompt lists with scheduler
      request ids, physical slots, packed prefill slabs, and output routing.
- [x] Coalesce compatible non-streaming server generations into one
      prompt-list `LLM.generate()` call.
- [x] Preserve a narrow safety lock only around non-reentrant
      model/session mutation until the session is proven concurrency-safe.
- [x] Keep `n>1` rejected at the API layer until C5 lowers it to N
      scheduler requests.

### C2 — native c>N prefill/decode green

Definition of done: full 40-layer Qwen/PARO BF16 c=2/4/8 512/128 emits
generated-token IDs equal to independent c=1 runs, with no serial decode
bridge and `throughput_claim_eligible=true`. Append-only block-id contract
in place even though pool growth lands in C4.

- [x] Native compact packed BF16 prefill via `next_compact_prefill_slabs(...)`
      + `prefill_native_packed(...)`.
- [x] Guard `step_batch_native` behind
      `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE=1` and default
      `_sample_batch_from_hidden` to `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=
      serial_lm_head` until row-aware sampler lands.
- [x] Document `HIPENGINE_QWEN35_EXPERIMENTAL_NATIVE_BATCH_DECODE` and
      `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE` in `docs/ENVS.md`.
- [x] Remove stale compatibility glue once the guarded native API is
      settled (removed the `batch_execution_metadata(...)` `TypeError`
      shim in the generator).
- [x] Add HIP-guarded reduced-shape equality diagnostics that do **not**
      require full 40 layers, so failures can be bisected in CI/dev
      environments with ROCm. Keep full 40-layer 512/128 as the retained
      benchmark gate.
- [x] Re-run guarded current-default c=2 equality after `serial_lm_head`:
      L40 512/32 passes as a reduced diagnostic, but full L40 512/128 is
      still `rejected_correctness` at row 0 idx 87
      (`/tmp/hipengine-retained/guarded-L40-c2-512-128-current.json`).
- [x] Promote current c=2 accepted/rejected diagnostic artifacts under
      `benchmarks/results/` before using them as retained evidence.
- [ ] Root-cause and fix native linear-attention c>N drift: current
      auto selected-QKV/Z projection + native segmented state +
      batch-GEMV/Marlin output controls isolate native QKV/Z projections,
      grouped-compact MoE under this shape, native/fused output retention,
      and native full-attention parity as the
      remaining blockers.
- [ ] Add row-aware split-K full-attention decode/reduce before any
      long-context c>N claim (`max_context ≥ 1024`). The current long-context
      diagnostic uses the per-row split-K fallback, not a row-aware batch reducer.
- [x] Add CPU-runnable structural tests for the experimental env gate,
      INT8 KV rejection, default/invalid sample mode, and
      `throughput_claim_eligible=false` for guarded diagnostics.
- [x] Extend structural tests for invalid-slot and long-context rejection;
      sorted sparse slots are now accepted and covered by C4.8.
- [x] **Append-only block id contract.** Make the KV allocator's block id
      permanent for its lifetime. Remove any path that reuses a block id at
      a different pointer. Add a debug check that fails on pointer mutation.
- [x] **Live admission cap.** `KVPolicy.admission_cap()` returns *current*
      free capacity, not startup capacity.

### C3 — kernel coverage

Definition of done: every retained `(model, quant, KV)` row in the coverage
matrix has at least one green retained c>N cell on the 512/128 protocol.

- [ ] Validate batched INT8 KV append/decode paths with the same gates as
      BF16; require generated-token equality.
- [x] Make full-attention decode consume per-row `KVLiveSpans` for all
      retained KV storage dtypes.
- [x] Make linear-attention conv/recurrent state updates consume
      `[C, ...]` state, active masks, and slot ids; remove c1 aliases.
- [ ] Validate grouped-compact MoE metadata for c=2/4/8 after native
      linear-attention parity is fixed; batched selected-c1 MoE is correctness-green
      for c<=8 but is not the retained throughput target.
- [x] Keep c=1 GEMV dispatch separate from c>N MMQ/GEMM/WMMA candidates.
      Evidence: `plan_projection_dispatch(...)` pins c=1 to `row_gemv_c1`
      even when a faster c-aware candidate is supplied, c>N candidates require
      accepted benchmark evidence with aggregate and per-request speedups over
      row-GEMV, missing/slow/rejected evidence falls back to row-GEMV, and
      accepted c>N artifacts must record an evidence-backed non-row-GEMV
      `execution.batch_execution.projection_dispatch` decision before any
      projection throughput claim; covered by
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] Validate GGUF Q4_K/Q5_K/Q6_K/Q8_0 c=2/4/8 with the same gates.
      Progress: `scripts/qwen35_batch_c_sweep.py --include-gguf` now plans
      blocked GGUF c>N diagnostics for all four template quants at c>1 with
      `HIP_VISIBLE_DEVICES` preserved in the command labels, and validates that
      summaries keep the template fixture, exact per-c quant order/set, and artifact filenames bound to those commands; this is planning
      coverage only and does not close the item because generated-token equality
      artifacts for c=2/4/8 are still missing.
- [ ] Native row-aware LM-head + sampler: replace the per-row argmax loop
      and prepare for per-row sampling params (C5 finishes this).

### C4 — scheduler-owned engine loop + dynamic KV pool

Definition of done: one long-lived background driver runs
`submit/poll/cancel`, ticks the work classes, grows/shrinks the KV pool, and
routes both streaming and non-streaming through the same path. `LLM.generate()`
becomes a `submit+poll` adapter.

- [x] Promote the resident runner from static prompt-list batches to a
      scheduler-owned engine loop that persists beyond one
      `LLM.generate()` call. Evidence: `ResidentEngineLoop` in
      `hipengine/generation/engine_loop.py` plus
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] Implement `submit(prompt_tokens, sampling, max_new_tokens, stream) →
      request_id`, `poll(timeout) → events`, `cancel(request_id) → bool`.
      Evidence: `ResidentEngineLoop.submit/poll/cancel` and scheduler reclaim
      coverage in `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [x] Lower `LLM.generate()` and OpenAI server endpoints to
      `submit + poll + cancel`. Evidence: `SubmitPollTextGenerator` wraps
      public text generation, server non-streaming calls use the shared
      batcher/generator path, and
      `pytest -q tests/test_generation_batch_scheduler.py tests/test_server_api.py -q`.
- [x] Implement the per-tick policy: `RECLAIM → ADMIT → choose(PREFILL_CHUNK,
      DECODE_STEP)`; default `protect_decode`.
- [x] Add `kv_pool_chunk_pages` chunked underlying allocation with one chunk
      at startup. Evidence: `ChunkedKVPool(..., chunk_pages=...)` in
      `hipengine/kvcache/pool.py` and
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] Add grow-on-admission up to `kv_pool_high_water_bytes`, one attempt per
      admit cycle; record `grow_events` / `grow_failures`. Evidence:
      `ChunkedKVPool.allocate(...)` grow counters and
      `pytest -q tests/test_kvcache_policy.py -q`.
- [x] Add idle shrink down to `kv_pool_low_water_bytes` with
      `kv_pool_idle_grace_seconds`; never free a chunk holding a non-zero
      refcount. Evidence: `ChunkedKVPool.shrink_idle(...)` plus refcounted-tail
      coverage in `pytest -q tests/test_kvcache_policy.py -q`.
- [x] Add CLI/env knobs `--kv-pool-{initial,low-water,high-water,
      chunk-pages,idle-grace}-*`,
      `HIPENGINE_KV_POOL_*`,
      `HIPENGINE_PREFILL_DECODE_POLICY` / `--prefill-decode-policy`;
      document in `docs/ENVS.md`.
- [x] Add a burst-then-idle acceptance fixture that exercises grow and
      shrink and records the events. Evidence:
      `test_chunked_kv_pool_grows_and_shrinks_on_burst_idle` in
      `tests/test_kvcache_policy.py`.
- [x] Add a memory-audit test that fails if a block id's backing pointer
      changes mid-run. Evidence:
      `test_fixed_paged_policy_audits_append_only_block_pointers` in
      `tests/test_kvcache_policy.py`.
- [x] Narrow or remove the coarse `generation_lock`; any remaining lock
      protects only non-reentrant session mutation, not the lifetime of a
      generated batch. Evidence: `hipengine/server/api.py` now uses
      `session_lock` only for LLM construction/preparation/context-budget mutation,
      `_GenerationBatcher` no longer accepts a lock, and
      `test_generation_batcher_default_zero_window_queues_without_lifetime_lock`
      plus `pytest -q tests/test_server_api.py -q` cover the batcher path.
- [x] Route server streaming through the engine loop and the per-request
      token queue; the streaming path no longer bypasses the batcher.
- [x] Unify cancel / disconnect / EOS / max-tokens / timeout into one
      `RECLAIM` path.
- [x] Per-request observability fields (queue/prefill/decode seconds,
      kv pages owned/peak, bucket key, admission_blocked_reason,
      finish_reason).
- [x] Per-pool observability counters
      (current_bytes, high_water_observed, grow/shrink events, free pages,
      refcounted pages).
- [x] Extend native decode correctness to non-compact slots after
      scheduler compaction/reclaim moves requests; sorted sparse slots are
      supported by explicit physical slot ids.

### C5 — KV sharing, per-row sampler, `n>1`, `/metrics`

Definition of done: refcounted prefix sharing on by default; per-row sampler
in code; `n>1` lowered to N scheduler requests; Prometheus `/metrics`
endpoint live; retained c>N rows include all gates above.

- [x] Add block-id refcounts; admission increments refcount when reusing
      an existing block on a matched prefix.
- [x] Implement RadixCache trie index over token ids; expose
      `HIPENGINE_PREFIX_CACHE` / `--prefix-cache` in `{off, radix}` with
      default `off` until acceptance gates pass.
- [x] Implement copy-on-write fork at first divergent token.
- [x] **KVTC ABI guardrail.** Block ids returned by the allocator must be
      stable across hypothetical tier moves; refcount and eviction state
      must attach to the radix node rather than the block pointer. KVTC
      itself ships in a separate feature branch. Evidence:
      `PrefixCacheEntryState` exposes pointer-independent radix-node metadata
      (`block_ids`, `owner_request_ids`, `refcount`, `eviction_state`),
      `RadixCache.mark_entry_eviction_state(...)` updates tier/eviction state
      without rewriting block ids, and
      `test_radix_cache_entry_state_is_pointer_independent_kvtc_guardrail`
      covers stable block ids across tier-state changes and cancellation.
- [x] Lower `n > 1` at the API layer to N `submit()` calls with the same
      prompt tokens and distinct seeds; collect output by `request_id`;
      remove the `n>1 → 400` rejection.
- [x] Land the per-row sampler params block (temperature, top-k, top-p,
      repetition penalty, seed, stop tokens) in one launch.
- [x] Per-row EOS handling drives `RECLAIM` per-row, not per-batch.
- [x] Remove or demote the submission-time coalescer
      (`_GenerationBatcher`) to a cold-path optimization. Evidence: default
      `ServerConfig.generation_batch_window_ms` and
      `HIPENGINE_GENERATION_BATCH_WINDOW_MS`/`--generation-batch-window-ms`
      remain `0`, `_GenerationBatcher` applies no intentional zero-window delay
      and no longer holds a request-lifetime generation lock, and
      `test_generation_batcher_default_zero_window_queues_without_lifetime_lock`
      plus `test_metrics_prefix_cache_and_generation_batch_cli_env_defaults`
      cover the default/opt-in path.
- [x] Add Prometheus `/metrics` endpoint;
      knob `HIPENGINE_METRICS` / `--metrics` in `{off, prometheus}`;
      default `off` until coverage is broad.
- [ ] Per-bucket graph-cache observability
      (entries, hits, misses, miss reason, kernel-time histogram). Progress:
      `GraphBucketCache.stats.to_json_dict()` now includes miss-reason counts
      and zero-filled, fixed-key `GRAPH_KERNEL_TIME_HISTOGRAM_BUCKETS` histogram
      buckets; retained/serial scripts emit that shape; retained bench validates
      decode shape-key axes (including context-bucket coverage for the workload
      prompt length), merges integer profiler kernel durations into the fixed
      histogram schema, and blocks promotion when shape keys are invalid,
      hit/miss/replay-rate stats are invalid, fixed histogram buckets are missing,
      no known-bucket observations remain, or unknown buckets appear.
      Accepted-artifact schema shares the runtime bucket taxonomy and requires
      context-bucket workload coverage plus fixed-bucket histogram observations
      that cover profiler kernel-duration evidence for accepted rows, and
      `/metrics` exports labeled miss-reason counters plus zero-filled counters
      for every known kernel-time bucket while filtering unknown bucket labels and
      malformed scalar/mapping values (bool, non-finite, negative, or
      non-numeric); the item remains open until real replay profiler evidence
      populates kernel-time buckets.
- [x] Retained-row gates 4 (admission/completion timestamps + p50/p95) and
      6/7/8 (dynamic pool + stable block id + prefix sharing artifact)
      enforced by the bench harness.

## Performance gates and optimization work

The phase ladder above is organized by *what is enabled* (correctness,
engine loop, sharing). Performance-scaling work runs **inside** each phase
after the correctness gate for that phase is green. This section collects
the shared performance contract; cite a specific phase when scheduling a
performance item.

### Baseline artifacts

Establish these before optimizing anything:

- c=1 native prefill/decode for the retained shapes.
- c=2/4/8 serial bridge diagnostics.
- First green uncaptured native c>N rows (no graph replay).
- Primitive/kernel microbenchmarks for attention, KV append, MoE,
  projection, and LM-head sampling.

### Scaling reported on every retained c>N row

- `prefill_tok_s_aggregate / c1_prefill_tok_s`.
- `decode_tok_s_aggregate / c1_decode_tok_s`.
- `decode_tok_s_per_request / c1_decode_tok_s`.
- p50/p95 first-token latency and inter-token latency.
- Active-batch occupancy over time.

### Target throughput envelope

- Decode aggregate speedup vs c=1 and vs the serial bridge. Per
  [`PLAN.md`](PLAN.md), c=8 decode plausibly lands around 2-4× c=1
  aggregate when kernels reuse enough work; do not promise 8×.
- Prefill aggregate scaling vs c=1 by keeping prompt rows packed, avoiding
  per-request Python loops, and using AOTriton/WMMA paths where they beat
  row-GEMV.

### Optimization checklist (overlay onto C2-C5)

- [ ] Add hipGraph capture/replay buckets for decode by `(C, context
      bucket, active mask, KV dtype, layer plan, top-k/experts, replay
      length)`, with an uncaptured fallback for rare shapes.
- [x] Add graph-bucket cache hit/miss and replay statistics to artifacts.
      Evidence: `GraphBucketStats.to_json_dict()` serializes `entries`,
      `hits`, `misses`, `replay_hit_rate`, `miss_reasons`, and zero-filled
      fixed-bucket `kernel_time_histogram_ns`; `scripts/qwen35_batch_retained_bench.py`
      and serial diagnostics emit `decode_shape_key` / `graph_bucket_stats`, and the retained bench merges profiler kernel durations into that fixed-bucket histogram;
      accepted-artifact schema requires those fields, the complete fixed bucket
      key set, and non-empty known-bucket histogram observations for accepted rows
      using the runtime bucket taxonomy; `/metrics` exports graph-bucket counters,
      filters kernel-time buckets to that taxonomy, and rejects malformed graph/KV
      scalar and mapping values before Prometheus export;
      covered by
      `test_graph_bucket_cache_clear_resets_entries_and_counters`,
      `test_qwen35_retained_records_decode_graph_bucket_metadata`,
      `test_qwen35_batch_diagnostic_artifact_schema_enforces_accepted_row_gates`,
      `test_metrics_endpoint_filters_malformed_graph_bucket_scalars`,
      `test_metrics_endpoint_filters_malformed_kv_pool_scalars`,
      and `test_metrics_endpoint_is_opt_in_and_additive`.
- [ ] Eliminate residual serial loops on the native path after correctness
      is green:
  - full-attention per-row fallback;
  - per-row host metadata allocation/free;
  - per-row LM-head/argmax launches where a batched launch is correct;
  - Python per-layer dispatch overhead inside steady-state decode.
- [ ] c-aware projection dispatch thresholds:
  - c=1 stays on tuned GEMV/Marlin-K paths;
  - c=2/4/8 use MMQ/GEMM/WMMA-style kernels when they beat row-GEMV;
  - c>16 prefers GEMM/WMMA and grouped MoE designs over widening c1
    GEMV wrappers.
- [ ] MoE routed-lane reuse:
  - group lanes by expert;
  - use compact/WMMA grouped kernels when routed lanes justify it;
  - measure router, group-scatter, gate/up, down, shared expert, and
    combine time separately.
- [ ] Memory traffic and workspace reuse:
  - preallocate per-bucket scratch instead of allocating per step;
  - avoid host-device copies for metadata that can be updated on device;
  - keep JIT builds out of profiler runs with `require_cached`;
  - track peak allocator/KV/workspace bytes in artifacts.
- [x] Backpressure and fairness policies once the scheduler is continuous:
  - max active requests, max queued requests, max prefill chunk tokens are
    represented by `EngineLoopConfig.max_active_requests`,
    `max_pending_requests`, and `max_prefill_chunk_tokens`, exposed as
    `HIPENGINE_MAX_ACTIVE_REQUESTS`, `HIPENGINE_MAX_PENDING_REQUESTS`,
    `HIPENGINE_MAX_PREFILL_CHUNK_TOKENS` plus matching CLI flags, and wired
    through `ResidentEngineLoop` / `ResidentBatchScheduler`; excess queued
    submissions are rejected before admission.
  - prefill-vs-decode policy protects decode latency by default
    (`protect_decode`; see §Engine-loop contract) with `protect_ttft` and
    `fair` alternatives.
  - per-row sampling params (`PerRowSamplingParams` / `SamplerParamsBlock`)
    let incompatible temperature/top-k/top-p/repetition/stop-token rows stay in
    one decode launch instead of starving behind compatibility groups.
    Evidence: `test_resident_batch_scheduler_enforces_pending_queue_limit`,
    `test_engine_loop_cli_env_defaults_match_docs`,
    `test_engine_loop_cli_env_overrides`,
    `test_resident_engine_loop_prefill_decode_policies`, and
    `test_resident_scheduler_per_row_sampler_block_keeps_incompatible_rows_together`.
- [x] Profiler summaries for accepted rows: expected kernel names,
      duration/share for attention, MoE, projection, sampling, graph
      replay, and any CPU-side bottleneck. Evidence: accepted-artifact schema
      requires native batch expected kernel names, per-kernel durations/shares
      matching `kernel_durations_ns / total_kernel_duration_ns`, category
      duration/share buckets for attention/MoE/projection/sampling/graph/other,
      and CPU-side bottleneck durations/shares whose totals match; retained-bench
      ingestion synthesizes totals, shares, category buckets, and CPU-side
      bottleneck summaries when profiler summaries provide durations; covered by
      `pytest -q tests/test_generation_batch_scheduler.py -q`.
- [ ] Only update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and
      `benchmarks/results/` for retained rows with correctness green,
      protocol shape satisfied, and profiler evidence. Rejected/blocked
      diagnostics stay useful but are not scoreboard entries.

## Out of scope (C6 onward)

C6 onward is out of scope for this doc:

- **TP / EP** (separate feature branch). Design in [`PLAN.md`](PLAN.md)
  §Multi-GPU Strategy. CONCURRENCY contracts are designed to be TP-safe;
  see §Forward-compatibility guardrails.
- **DMS compact KV serving** (separate feature branch). Roadmap in
  [`KVCACHE.md`](KVCACHE.md) Phase K2. CONCURRENCY contracts are designed to
  be DMS-safe; see §Forward-compatibility guardrails.
- **KVTC tiered KV storage** (separate feature branch). HBM → pinned host
  RAM → NVMe / disk. CONCURRENCY contracts are designed to be KVTC-safe;
  see §Forward-compatibility guardrails.
- **Speculative decoding** (separate feature branches). MTP, DFlash,
  EAGLE3; designs in [`MTP.md`](MTP.md), [`DFLASH.md`](DFLASH.md),
  [`SPECULATIVE-DECODE.md`](SPECULATIVE-DECODE.md).

## Forward-compatibility guardrails

CONCURRENCY-side decisions that the TP, DMS, SpecDec, and KVTC feature
branches depend on. The work in C0..C5 must already satisfy these; they are
not new tasks.

### Don't break TP

- **Scheduler / admission / sampler are single-owner.** TP rank-0 will own
  the engine loop; workers tick in lockstep. Do not put admission, sampling,
  or pool-resize decisions inside per-rank code.
- **hipGraph bucket keys are rank-agnostic.** Same
  `(C, context bucket, active mask, KV dtype, layer plan, top-k/experts,
  replay length)`. Per-rank capture/replay is fine; key derivation isn't
  per-rank.
- **All-reduce points are loop-visible.** Reductions happen after `o_proj`,
  after `down_proj`, after the shared expert
  (per [`PLAN.md`](PLAN.md) §Multi-GPU Strategy). Don't fold reductions into
  kernel internals where the loop can't see them.
- **KV is replicated per rank first, sharded later.** `KVLiveSpans` is
  per-rank; the scheduler does not assume rank-shared KV. The dynamic pool
  is per-rank; admission uses `min(per_rank_admission_cap)`.
- **No `if backend == "hip_tp_*"` branches in dispatch.** TP variants
  register as `(backend, layer, quant, variant)` tuples.

### Don't break DMS

- **`KVLiveSpans` stays the only attention / KV-write ABI.** No
  `(block_table, context_len)` shortcuts anywhere in the c>N decode path.
- **`KVPolicy.admission_cap()` is the scheduler's capacity unit.** Today
  returns dense-page capacity; DMS will return compact-live-token capacity.
  Continuous batching must not assume page == 1-token equivalent.
- **KV mutation is transactional.** Canonical KV updates only at scheduler
  commit points; verify/spec rows write scratch/journal. DMS evictions need
  the same commit-point gate.
- **Eviction-aware prefix sharing is a separate decision.** When prefix
  sharing lands in C5, either disable it under DMS or design per-sequence
  eviction overlays. Don't blind-share under variable-span eviction.
- **The engine loop must allow a `PACK_STEP` work class to be inserted
  between active requests' decode steps.** Don't model the loop as a strict
  `prefill ; decode_until_done` macro-pattern.

### Don't break SpecDec

- **Verify rows commit only at scheduler commit points.** Canonical KV (dense
  or compact) is updated only on accept; rejects discard scratch.
- **`DraftBatch` metadata is the verify ABI.** Verification kernels consume
  `request_id`, candidate token(s), parent position, draft depth, optional
  tree parent, and active mask. No c=1 chain shortcuts.
- **`VERIFY_STEP` is a peer work class.** The loop's per-tick policy can
  schedule verify steps without changing the contract.

### Don't break KVTC

- **Block id stays stable across tier moves.** A block id `b`'s id and
  refcount are preserved when its backing pointer moves between HBM,
  pinned host RAM, or NVMe. Consumers ask the allocator for the current
  pointer rather than caching it. This is a strict extension of the
  append-only block-id contract in §Dynamic KV pool.
- **Refcount and eviction state live on the radix node, not the block
  pointer.** Tier moves do not change prefix-sharing topology.
- **Tier moves happen only at scheduler commit points.** No mid-kernel
  tier promotion or demotion; no torn `KVLiveSpans` reads.
- **`KVLiveSpans` is unchanged by tiering.** A tier move is a pointer swap
  inside the allocator, not a span rewrite.
- **The `/metrics` schema is extensible.** When KVTC lands it adds
  counters like `kv_tier_promotions_total{tier}` and
  `kv_tier_demotions_total{tier}`; CONCURRENCY's per-pool counter block
  must be additive, not restructured.

## GPU0 diagnostic evidence

Most historical scratch fixtures and artifacts live under `/tmp/hipengine-prebench/`
and `/tmp/hipengine-retained/`. The current c=2 native-decode review artifacts
are promoted under `benchmarks/results/` as blocked/rejected diagnostics:

- `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-32-blocked.json`
- `benchmarks/results/2026-05-27-hipengine-qwen35-paro-c2-native-l40-512-128-rejected-correctness.json`

These are not scoreboard rows because neither has `status=accepted`.

Primitive c=1/2/4/8 correctness (BF16 batched KV append + batched paged
full-attention decode):

| c | append key mismatch | append value mismatch | attn batch-vs-c1 max abs |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0.0 |
| 2 | 0 | 0 | 0.0 |
| 4 | 0 | 0 | 0.0 |
| 8 | 0 | 0 | 0.0 |

Scheduler serial bridge sweep (Qwen3.6/PARO-35B-A3B, W7900, 40 layers, INT8
KV, prompt 512 + 4K, decode 128):

| Shape | Decode aggregate tok/s | Decode per-request tok/s |
| --- | ---: | ---: |
| c=1 512/128 | 102.12 | 102.12 |
| c=2 512/128 | 102.32 | 51.16 |
| c=4 512/128 | 101.47 | 25.37 |
| c=8 512/128 | 100.30 | 12.54 |
| c=1 4K/128 | 99.98 | 99.98 |
| c=8 4K/128 | 98.88 | 12.36 |

Aggregate decode stays flat while per-request falls as `1/c` — the signature
of the serial bridge. Full per-row artifacts:
`/tmp/hipengine-prebench/scheduler/qwen36-paro-cC-{512,4k}-128-serial-bridge.json`.

Experimental native decode (commit `86e6fa2`) currently has two distinct
correctness signals:

- Pre-workaround batched LM-head L8 512/32 rejected at row 0 idx 13
  (`/tmp/hipengine-retained/guarded-L8-512-32.json`). Switching to
  `HIPENGINE_QWEN35_BATCH_SAMPLE_MODE=serial_lm_head` fixes that reduced
  512/32 drift: `/tmp/hipengine-retained/eq-{L8,L40}-512-32-serialsample.json`
  passed, and the current-default rerun
  `/tmp/hipengine-retained/guarded-L40-512-32-current.json` also passed
  equality (`status=blocked` only because 32 decode tokens is reduced).
- Full 40-layer c=2 512/128 still rejects on current tip with the
  `serial_lm_head` default:
  `/tmp/hipengine-retained/guarded-L40-c2-512-128-current.json`, row 0 idx 87
  (`batch=271`, `c1=1165`), `throughput_claim_eligible=false`. The later
  hidden-bisect controls narrowed the reduced L8 failure beyond token-only
  sweeps: selected-c1 projection/state/output replay is hidden/token green only
  with grouped-compact MoE left active, while forcing selected-c1 MoE regresses;
  the next correctness step is native linear-attention segmented
  conv/GDN/recurrent state plus native batched output projection parity.

## What not to claim yet

Do not describe any current row as:

- true c=2/4/8 serving throughput;
- continuous batching;
- radix/prefix-cache reuse;
- compact/DMS KV serving;
- c-aware decode graph replay;
- dynamic KV pool growth/shrink;
- KVTC / tiered KV storage.

The correct phrasing for current diagnostics is:

> c>N scheduler serial bridge diagnostic: batch-shaped slots and KV metadata,
> but active rows execute serially through the c=1 layer path. Aggregate
> decode throughput remains roughly c=1, so the row is blocked/non-retained.
