---
status: current
owns: Gemma 4 26B-A4B gfx1100 single-request optimization plan, measurement contract, candidate sequence, and completion criteria.
---
# Gemma 4 26B-A4B optimization campaign

## Current correctness status — 2026-09-26

The key-slice attention implementation failed the corrected teacher-forced
numerical gate. Earlier entries below claiming that its 1023-row comparison
qualified the default are historical and invalid as promotion evidence: that
comparison did not engage split decode. The repaired evaluator records actual
route selection and distinguishes numerical rows from complete promotion.
See `benchmarks/results/2026-09-26-gemma4-split-layer-cache-trace.json`.

The correctness repair replaces key-slice partial sums with dimension
partitioning, preserving the single-kernel ascending-key accumulation order.
The recorded 1023-row strict comparison has zero KL and 100% top-1 agreement;
public-generation parity also passes. This is a correctness fix, not a speed
win over the numerically failing implementation. The depth4 candidate measured
43.74 decode tok/s at 1024 prompt / 128 output on the lane specified below.
The complete repair evidence and validation limitations belong in its worklog
entry; these checks do not close G4/G5 or establish task quality.

The measured optimization loop stopped after iteration 58 under the
three-failed-candidates/re-profile rule. Its closure is recorded in
`worklog/entries/20260926T114943.870421Z-lhl-gemma4-dimension-tranche-close-31c043.md`.

## Objective and scope

Improve real Gemma 4 26B-A4B GGUF text-inference latency and throughput without
regressing output correctness or request ownership. Start with the working
`gemma4` branch at `1ea13cf8c`, not an isolated kernel harness. The primary lane
is host `epyc`, physical GPU1, RX 7900 XTX, `hip_gfx1100`, the existing
`gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` artifact, greedy decoding, and BF16 KV.

The user requested this campaign after the text-inference review. The milestone
notes below record its execution history; the current status above takes
precedence over historical acceptance claims.

Success means lower public request latency and improved prefill/decode costs,
with the selected default path verified through `LLM.generate()` and
`hipengine serve`. There is no invented throughput target or minimum percentage
win. Same-artifact llama.cpp is an engine comparison; Qwen3.6-35B-A3B on the
same GPU is a product reference, not an arithmetic oracle or a model-quality
comparison.

First tranche: single-request inference through 4096 prompt tokens with 128
outputs, plus correctness checks at chunk boundaries and longer contexts.
Exclude multi-GPU, new quantizations, MTP, tool parsing, multimodal input,
non-greedy sampling, and unrelated backend work. Native incremental streaming
and contexts beyond the existing attention limit are follow-on work unless
profiling makes their underlying changes necessary for this tranche. Do not
change model semantics or public defaults merely to improve a benchmark.

Normative rules remain in [OPTIMIZATION](../OPTIMIZATION.md),
[BENCHMARK](../BENCHMARK.md), [EXECUTION-PROFILES](../EXECUTION-PROFILES.md), and
[TESTING](../TESTING.md). This plan does not supersede them.

## Starting evidence and missing measurements

The [review closure](../../worklog/entries/20260923T091939.935292Z-lhl-gemma4-review-closure-474f1e.md)
records working direct generation, chunked prefill, server chat/SSE, and the
request/ownership/geometry fixes. Its smoke timings combine prefill and decode,
exclude loading, and are single samples. They are not the campaign baseline.
Existing W7900 Qwen rows use another GPU and a different protocol; do not quote
ratios against them.

`Gemma4Runner.forward()` already separates prefill blocks from one-token
forwards, but `scripts/gemma4_real_generate.py` times the combined request.
`scripts/gemma4_llamacpp_compare.py` is a CPU first-token correctness diagnostic,
not a GPU engine-performance comparator. No separated Gemma campaign harness is
provided yet. The harness, comparator adapter, and evaluator described below
are deliverables, not commands that already exist.

The implementation has concrete profiling candidates:

- `hipengine/runtime/gemma4.py::_forward_block` computes and uploads per-layer
  RoPE tables and masks on the host, then copies full logits to the host and
  applies softcapping there.
- Every prefill block computes its final norm/output head, although only the
  last block's logits reach the caller.
- `gemma4_project` delegates quantized projections to `launch_gguf_linear`.
  Record the effective kernel selected for each row count and weight type;
  do not assume the optimized Qwen path is automatically selected.
- Gemma attention stores all live-key scores in shared memory. Decode now has
  a dedicated bit-exact kernel (barrier rounds batched across a tile of keys,
  `tokens == 1` routing in `attention_symbol`), landed under G3; see worklog
  entry `20260924T102206.088801Z-lhl-gemma4-a651b0.md`. The 512-wide-head
  resource ceiling on live-key count is unchanged for the block kernel and
  its tiled decode twin still keys off the same shared-bytes check.

These are hypotheses, not measured bottlenecks.

## Measurement contract

### Freeze before tuning

Create a compact baseline artifact with the full source commit, clean/dirty
state, model identity/fingerprint, tokenizer/template identity, effective
execution profile and variant manifest, compiler/runtime versions, GPU identity,
power/clock settings, visibility variables, and exact commands. Hashes identify
evidence; never use them as runtime admission rules. Pin the baseline commit
and preserve it for paired runs; do not reset another worker's files.

Use `env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1` on `epyc` and verify that
logical device zero is the RX 7900 XTX before each measurement session. Check
ROCm liveness and competing GPU work. Keep clocks, power policy, environment,
KV policy, cache state, and model artifact unchanged across each pair. Run
engines sequentially, not concurrently on the GPU. Separate cold load/JIT from
warm resident measurements and exclude profiler overhead from headline timing.

### Workload matrix and accounting

| Scope | Workloads | Purpose |
| --- | --- | --- |
| Primary timing | Exact 128, 512, 1024, 4096 prompt tokens; 128 generated tokens; one request | Short and medium contexts, separate prefill/decode/wall costs |
| Boundary correctness | 1, 127, 511, 512, 513, 793, 1023, 1024, 1025 prompt tokens | Chunk thresholds, both sides, and unrelated shapes |
| Long correctness | 8191 prompt tokens plus one output at capacity 8192; small fixtures at the LDS boundary | Capacity/position accounting without claiming long-context speed |
| Public chat | Multi-category natural prompts; thinking on/off; stop naturally | Template, EOS, reasoning, request-wall latency |
| Public server | Regular and SSE versions of matched chat requests | Serving overhead and answer/reasoning parity |

The primary matrix fixes capacity at 8192 and prefill block at 512 initially.
Any block-size sweep is a separate named experiment, including memory cost.
Generate and save exact token IDs before timing; match prompt IDs within each
same-model engine comparison. Assert actual prompt/output counts. Fixed-length
runs explicitly ignore EOS in both arms; natural-EOS runs report the actual
count and are never silently mixed with fixed-length rows. No padding output
counts or forwarding an unused final token.

Report all timing phases, not just a single tokens/s field:

- Load, JIT/preparation, and warmup separately from measured requests.
- Prefill latency and input tokens/s, including the last-row output head.
- First-token latency, with sampler time and its timing boundary stated.
- Decode latency and tokens/s with the exact denominator: after the first token
  comes from prefill, 128 outputs require 127 subsequent decode forwards.
- Public request wall time and generated tokens divided by that time, explicitly
  labeled as including prefill. Record reasoning and answer tokens separately.
- HTTP time to first visible content and completion time. The existing SSE
  adapter buffers generation; its first event is not GPU time to first token.
- Allocator peak, device-memory scope, scratch/staging/KV bytes, and allocation,
  transfer, synchronization, and kernel-launch counts when instrumented.

Use explicit HIP synchronization around diagnostic GPU phase timing. Preserve
untouched public wall measurements to detect instrumentation effects. Validate
phase/count accounting with deterministic fake-runner tests and a real request.

Use one full-shape warmup and at least three measured samples per arm. Candidate
confirmation uses four balanced baseline/candidate pairs, alternating AB/BA
order (two of each), and stores every sample plus median/p95/min/max/stdev. Freeze the same
prompt suite and heldouts before tuning. Repeated-token inputs may supplement
kernel diagnostics but cannot replace natural multi-category public workloads.
Follow BENCHMARK's variance rejection rule; investigate noisy results instead
of selecting the best run or repeating until a desired result appears.

### Comparators

1. **Incumbent hipEngine:** same artifact, GPU, inputs, profile, request settings,
   and harness; this decides whether an optimization improved the implementation.
2. **llama.cpp HIP:** same Gemma GGUF, physical GPU and exact input token IDs,
   greedy output, matched thinking/template/BOS/EOS and context settings.
   Record backend/KV-type differences where exact matching is unavailable.
   Pin its source/build and reproduce exact commands; do not assume the CPU
   first-token comparison script launches this engine.
3. **Qwen3.6-35B-A3B GGUF UD-Q4_K_M:** same RX 7900 XTX, timing boundaries,
   workload sizes, greedy mode and BF16 KV where supported. It uses its own
   template/tokenizer and is a different-model latency reference. Probe capacity
   first; if it cannot fit the requested configuration, record that resource
   result rather than silently changing GPU or quantization. ParoQuant W4 is an
   optional separately labeled comparison, not a substitute for the GGUF row.

## Correctness and acceptance

Establish the incumbent against existing CPU/GPU fixtures and the independent
HF/llama.cpp checks before treating it as a teacher. A matching incumbent can
still share an implementation bug. Freeze teacher tokens, prompt categories,
heldouts, task scoring, and paired non-inferiority margins before evaluating
arithmetic candidates. Use the full `mtpbench-code-general-ja.jsonl` category
set plus heldouts; this is prompt coverage, not speculative decoding.

For exact host/ownership work, require unchanged logits/IDs under the declared
same-schedule contract, request accounting, and resource lifetime checks.
For changed arithmetic, use the full production profile, not an ad hoc bitwise
veto: 500–1000 shared-chain teacher-forced full-vocabulary rows, global and
per-category/shape/transition metrics, three deterministic repeats, finite
state, and task/heldout checks. The normative KL limits are mean <= 1e-3,
p95 <= 5e-3, p99 <= 2e-2, maximum <= 5e-2; top-1 >= 99% overall and >= 97% in
every scope. Rows above 2e-2 require explicit diagnosis. Include BF16-relative
checks where a teacher is available; missing teacher data is not fabricated.
The CPU-reference outer floor alone cannot promote changed arithmetic.

Preserve strict registered fallbacks for fused or reassociated variants.
Exercise reset/reuse, failed allocation, cancellation/cleanup, chunk boundaries,
sliding/global attention, tied/untied heads, and unequal FFN widths as applicable.
Device argmax must preserve tie-breaking, softcap semantics, EOS and stop rules;
softcap monotonicity in real arithmetic alone does not prove floating-point
argmax parity.

A candidate is kept when its declared correctness/profile gates pass and paired
measurements show improvement in the targeted phase or public wall time without
a confirmed regression in another primary workload. No minimum percentage gain
is required. If another row worsens, rerun that paired row with bounded extra
samples to distinguish noise; do not average a real regression away. Document
sub-window wins honestly when aggregate latency is flat. An ambiguous result
is inconclusive, not an automatic promotion or rejection. Freeze any automated
noise/regression decision rule in the baseline artifact before launching a loop.

Accepted changes ship on the ordinary path. Confirm selected kernels, sampler,
and fallback reason through `LLM.generate()` or `hipengine serve`; a private
harness-only win is not shipped. Do not add model-name allowlists or default-off
flags because a configuration has not been benchmarked.

## Execution milestones

- [ ] **G0 — Harness and baseline.** Implement separated phase timing and public
  wall measurement with accounting tests; freeze evaluator and provenance;
  measure the matrix and same-GPU comparators. Record unsupported comparator
  configurations honestly. No kernel tuning before the baseline exists.
  *Status 2026-09-24: harness, matrix, boundary rows, both comparators, and
  the consolidated artifact are published
  (`benchmarks/results/2026-09-24-gemma4-26b-a4b-g0-baseline.json`, commit
  `8a3d35193`). The evaluator-freeze sub-item is deferred to immediately
  before the first changed-arithmetic candidate — host/exact candidates and
  profiling need only the already-passing public-path ID parity — and the
  deferral is recorded in that artifact's `limitations`.
- [ ] **G1 — Attribution.** Prebuild outside `rocprofv3`, supply the compiler
  version file and require cached builds. Trace one short and one long primary
  shape with plausible named dispatches; attribute GPU time, host gaps,
  transfers, synchronization, allocation and output-head work. Rank candidates
  by measured contribution. Retain compact summaries, not raw profiler dumps.
  *Status 2026-09-24: both shapes traced and ranked; compact summaries and the
  recipe fix live in worklog entry
  `20260924T091834.106937Z-lhl-gemma4-optimization-6a8d54.md` (raw dumps not
  committed). Headline: decode is device-bound — attention is 79.2% of decode
  device time at KV~4096 (150.86 ms/step) and 24% at KV~129; q4_k 23% and
  pack8 19% at KV~129; pack8 runs 410 launches/step. Open sub-item: transfer
  attribution (kernel-trace reports no copies; one `--hip-api-trace` run
  remains), so this box stays unchecked.
- [ ] **G2 — Host and redundant-work candidates.** Evaluate intermediate-block
  output-head avoidance, RoPE/mask reuse or device generation, bounded scratch
  reuse, and device greedy sampling only when G1 supports them. One hypothesis
  per atomic unit; RED/GREEN tests before implementation where practical.
- [ ] **G3 — Dominant kernels.** Tune the measured projection/MoE/attention
  bottleneck, with in-tree kernels and registry dispatch. Explore row-batched
  projection routes, decode specialization or tiled attention in measured order.
  Graph capture/fusion follows stable pointer and state ownership, not before it.
  *Status 2026-09-24: decode attention specialization landed — a bit-exact
  decode kernel batches barrier rounds across 8 keys and halves the reduction
  barrier cost (0.263 vs 0.458 ms/launch microbenchmark on the RX 7900 XTX).
  Campaign row at 1024p/128o: decode 16.09 -> 20.48 tok/s, prefill unchanged,
  public-path token-id parity exact. Design history, refuted variants and the
  divergent-shuffle gotcha are in worklog entry
  `20260924T102206.088801Z-lhl-gemma4-a651b0.md`. Remaining G3 candidates in
  measured order: q4_k decode GEMV registration (changed arithmetic — needs the
  deferred evaluator freeze), pack8 launch count, then re-profile.
  *Status 2026-09-24 (iteration 15, final): the teacher-forced evaluator is
  frozen and the multi-block attention split was built, measured (0.102 vs
  0.263 ms/launch) and **rejected at the gate** (kl_max 0.304 > 0.05; bf16-KV
  amplification) — evidence row
  `2026-09-24-gemma4-26b-a4b-attention-split-rejected.json`. The q4_k decode
  re-route's selection point was located and closed in the same span: it is
  stacked behind a default-off diagnostic env, a missing Q5_1 compact down
  kernel (no table entry in any family) and an `expert_ffn % 256` guard the
  artifact's ffn=704 fails — full analysis in worklog entry
  `20260924T111851.082630Z-lhl-gemma4-f98888.md`. At the original 15-iteration
  cap the loop stood at 23.1517 tok/s (+43.9% vs the 16.0931 baseline);
  **2026-09-24 lead decision: the cap is raised to 500 iterations and the
  loop restarted** — G4/G5 and the Q5_1 kernel campaign remain in scope.
  *Status 2026-09-24 (iterations 16-18): the compact Q5_1 down GEMV landed as
  default-off capability, and the dense pack8-GEMV decode rewrite
  (`HIPENGINE_GGUF_GEMV_DECODE`) was **rejected at the production gate**: it is
  not a bit-exact drop-in — the decode kernel walks eight consecutive k per
  thread with the per-block scale hoisted, while the legacy pack8 kernel walks
  unit stride, so per-thread partial sums cover different k-sets and a small
  fraction of bf16 outputs land one ULP apart (32/1023 top-1 flips, kl_max
  2.447 vs the 5e-2 bar) with no measurable speed win. Attribution is complete:
  a resolve-level probe shows this single substitution is the whole candidate
  (205 launches change, nothing else), the forced-rewrite arm is bit-identical
  to the env-on arm, and the default path is bit-identical to the frozen
  baseline. Evidence row
  `2026-09-24-gemma4-26b-a4b-pack8-decode-rejected.json`; mechanism pinned by
  `tests/test_gpu_gguf_q8_0_pack8_gemv_decode_parity.py`. Because the compact
  route's session guard is this same switch, evaluating compact on its own
  merits requires decoupling that guard.
  *Status 2026-09-25 (iteration 19): the measured order changed. A decode-phase
  `rocprofv3` trace diff puts `gemma4_attention_decode_kernel` at **24.0 ms of
  the 45.7 ms token (53%)**, the MoE decode linears at 6.2 ms combined (13.6%),
  dense projections 3.9 ms, norms 1.4 ms, and host dispatch across 1116
  launches at 8.0 ms (7.2 us each). Two explanations for the attention cost were closed by
  measurement: it is not KV-bandwidth-bound (an all-zero keep-mask, which
  removes every K/V load, is only 11% faster - **withdrawn in iteration 20: the
  all-masked build produces NaN weights, so `weight == 0.0f` never fired and no
  V load was actually removed**) and not barrier-bound (widening
  the launcher's tile ceiling from 8 to 32 gives tile 8 = 598.5 us against
  16 = 1076.8 and 32 = 722.5 at sliding/1024 - a regression, reverted). The
  kernel is latency-bound at a 16-block grid with one 2-byte dependent load per
  thread per key, so the lever is memory-level parallelism in the tile walk.
  Evidence row `2026-09-25-gemma4-26b-a4b-decode-bottleneck-profile.json`
  (amended with a labeled `corrections` block), tool
  `scripts/gemma4_attention_decode_bench.py`, worklog entry
  `20260925T193532.806869Z-lhl-gemma4-2b8481.md`.
  *Status 2026-09-25 (iteration 20): an eight-point ablation study of the decode
  attention kernel closed the remaining cost hypotheses and **changed the
  mechanism**. Re-measured correctly, K/V loads are ~0% of the sliding geometry
  (591 vs 600-708 us at keys=1024) and 26% of the full geometry (871 vs 1186 us).
  Removing the 256-lane reduction tree (585 us), all four tile-loop barriers
  (684 us), the mask byte load (1118 us, worse), the publish broadcast and the
  running-max chain (696 us) each leaves the time within noise of baseline, and a
  minimal tile body carrying no loads and no dot product at all still costs 525
  us. Cost is ~5.3 us per 8-key tile (~0.5-0.6 us per key) and is **invariant to
  the work inside the key iteration**, so the 256-thread / 8-warp /
  5-barrier-per-8-keys structure is itself the cost and micro-optimization inside
  the existing loop cannot win. Next candidate: one warp per (token, head) with
  32 lanes x 8 dims, 16-byte vector loads, no shared-memory traffic and no
  barriers, reproducing the 256-lane tree with 5 shuffle rounds plus 3 in-lane
  rounds in identical association order so the result stays bit-exact. Evidence
  row `2026-09-25-gemma4-26b-a4b-decode-attention-ablation.json`, worklog entry
  `20260925T194545.594823Z-lhl-gemma-4-decode-attention-ablation-cost-is-invari-cad8cd.md`.
  *Status 2026-09-25 (iteration 21): **accepted** — the warp-per-head decode
  kernel with a shuffle-exact 256-lane tree, selected for `head_dim == 512`. The
  block kernel's tree spans eight warps, so every key costs a shared partial
  row, three LDS tree rounds, a publish/broadcast pair and a barrier sequence;
  the new kernel puts all 256 tree lanes inside one warp (lane `w` owns tree
  lanes `[8w, 8w+8)`), turning strides 128/64/32/16/8 into shuffle lane
  distances 16/8/4/2/1 and strides 4/2/1 into in-lane adds — same pairs, same
  order, so it is **bit-exact** (parity suite extended with the four real
  geometries, keys 1..8192, both dtypes) and needs no production-profile gate.
  Measured 851-870 us vs the block kernel's 1186 us on the full geometry at
  keys=1024 (28%), 1489 vs 2210 us at keys=2048. At `head_dim == 256` the same
  kernel loses (822 vs 652-700 us) because one warp per (token, head) is 16
  warps against the block kernel's 128, so the launcher keeps the block kernel
  there; a grid-scaling run (`--tokens` 1/2/4/8/16 -> 695/1097/1093/1653/1116 us)
  shows the machine is 16x idle and the block kernel's remaining cost is
  per-block latency. Campaign metric **24.5903-24.6025 tok/s vs 23.1517
  (+6.2%)** with public-path parity true; decode improved on all four rows
  (128p 43.12 -> 44.08, 512p 31.26 -> 32.63, 4096p 12.62 -> 14.48), prefill
  within noise, public wall improved on all four. The iteration's first
  candidate — branchless batched tile loads — measured 910 vs 695 us and was
  reverted. Evidence row `2026-09-25-gemma4-26b-a4b-warp-decode-accepted.json`,
  worklog entry
  `20260925T201309.614359Z-lhl-gemma-4-decode-attention-warp-per-head-kernel-wi-c36c82.md`.
  Remaining in measured order: the sliding geometry still runs the block kernel
  at ~0.6 us per key with the machine 16x idle — the next candidate is eight
  warps per (token, head) with a strided key class per warp and the two
  cross-warp reductions hoisted to once per launch; then the 8.0 ms/token host
  gap from iteration 19.
  *Status 2026-09-25 (iteration 22): **accepted, and it supersedes iteration 21**.
  Eight warps share one (token, head); warp `w` owns the keys `w, w+8, w+16, ...`
  and reduces each of their logits with the shuffle-exact 256-lane tree, so pass
  1 has no LDS traffic and no barriers, while the two cross-warp reductions
  leave the key walk (each warp publishes its running maximum; the row maximum
  is the order-independent `fmaxf` of the eight). Pass 2 and pass 3 stay the
  block kernel's own partitions verbatim, so the kernel remains **bit-exact**
  (parity suite green at both real geometries, keys 1..8192, both dtypes) and
  needs no production-profile gate. It beats the block kernel on **both**
  geometries — 421-428 us vs 652-700 us at sliding/1024 and 665-696 us vs
  1186 us at full/1024 — so it replaces the warp-per-head kernel, which is
  deleted as dead code rather than kept as an unselected variant. Campaign
  metric **33.0213 tok/s from 24.5964 (+34.3%)**, and **+105% over the 16.0931
  baseline**; public-path parity true. All four rows improved: 128p 44.08 ->
  47.63, 512p 32.63 -> 39.94, 4096p 14.48 -> 21.45, prefill within noise, public
  wall improved on every row. Evidence row
  `2026-09-25-gemma4-26b-a4b-key-class-decode-accepted.json`, worklog entry
  `20260925T205258.017905Z-lhl-gemma-4-decode-attention-key-class-kernel-keeps-59134e.md`.
  Position against the comparators: the same-artifact llama.cpp reference is
  68.92 tok/s, so the engine is at 48% of it (23% at the campaign baseline); the
  campaign target of 70% of the Qwen3.6-35B-A3B reference (114.57 tok/s) is
  80.2 tok/s. Next in measured order: pass 3 of the decode attention kernel,
  which still walks every key once per lane with a single 2-byte V load and a
  shared-logit read and has never been restructured; then the 8.0 ms/token host
  gap from iteration 19.
  *Iteration 22 closing diagnostic (timing-only ablation, V loads removed from
  pass 3 of the key-class kernel, `--keys 1024 --iters 30`):* sliding 427.9 ->
  183.4 us, full 696.0 -> 204.1 us. Pass 3 is therefore **57% of the sliding
  kernel and 71% of the full one** (244 us and 492 us of the 1024-key launch),
  and passes 1 and 2 together are only 183-204 us. Pass 3 reads 16.78 MB unique
  at full in 492 us (34 GB/s) while the same loop with the loads removed moves
  100.66 MB issued in 204 us (493 GB/s), so it is latency-bound on a 2-byte
  per-key V load inside a dependent accumulate chain, not bandwidth-bound. The
  next candidate is an **order-preserving register prefetch** in pass 3 (load
  key `j + n*threads` into a register while accumulating key `j`, so the
  ascending-j accumulation order is untouched and the kernel stays bit-exact),
  which is the same register-and-no-branch shape that made pass 1 fast. Pass 3
  has never been restructured in this campaign.
  *Status 2026-09-25 (iteration 23): **accepted**. Pass 3 is now pipelined and
  geometry-selected. `gemma4_decode_pass3<kDimsPerLane, kPrefetch, scalar_t>` is
  factored out of the key-class kernel: it loads the V value of key `j + 4` into
  a register while accumulating key `j` (the `weight == 0` skip the block kernel
  uses was what stopped the compiler from hoisting the loads, so the walk had
  been serialising at memory latency), and when `head_dim >= 2 * threads` lane L
  owns the dimensions 2L and 2L+1 and issues one `gemma4_pair2` load - 4 bytes
  for 16-bit KV - instead of two 2-byte loads. The sliding geometry
  (`head_dim` 256 = threads) keeps the scalar load, because a two-dimension
  mapping there idles half the block: 336.9 us two-wide against 318.9 us scalar.
  Depth 4 is the swept optimum (2/4/8/16 -> 384.1/315.7/336.0/337.6 us sliding,
  578.1/419.3/464.4/465.3 us full). Only *when* a load is issued and which lane
  owns which dimension changed, never the per-dimension ascending-j order, so
  the kernel stays **bit-exact** (parity and geometry suites green, both load
  paths covered). Microbenchmark: sliding 427.9 -> 318.9 us and full 696.0 ->
  339.8 us at keys=1024, full 1174 -> 509.5 us at keys=2048. Campaign metric
  **33.0213 -> 41.9049 tok/s (+26.9%)**, **+160% over the 16.0931 baseline**;
  all four rows improved (128p 47.63 -> 48.56, 512p 39.94 -> 46.43, 4096p 21.45
  -> 25.78), prefill unchanged within noise, public wall improved or held, public
  parity true. The engine is now at **61% of the same-artifact llama.cpp
  reference** (68.92 tok/s), from 23% at the campaign baseline. Evidence row
  `2026-09-25-gemma4-26b-a4b-pass3-pipeline-accepted.json`, worklog entry
  `20260925T210516.062297Z-lhl-gemma-4-decode-attention-pass-3-prefetched-v-loa-21a053.md`.
  Next in measured order: passes 1 and 2 are the larger half again (183-204 us of
  the 319-340 us launch, and pass 2 walks every key per lane through LDS), then
  the 8.0 ms/token host gap from iteration 19 (1116 launches x 7.2 us), which is
  now comparable to the whole attention cost.
  *Iteration 24 closing diagnostic (launch census through the public
  `LLM.generate()` path, counting every kernel launch for 1024 prompt + 32
  output):* **21,813 launches**, of which only 18 distinct symbols are launched
  through the ctypes path - the rest are hipblaslt. Per layer per decode step
  the elementwise kernels dominate the count: 5x `rmsnorm_f32w`, 2x
  `head_rmsnorm_f32w`, 2x `add_rmsnorm_scale`, 1x `rmsnorm_weightless`, 1x
  `partial_rotary`, 1x `gelu_tanh_mul_split`, 1x `router_prescale`, 1x
  `router_logits` = **14 tiny elementwise launches per layer per step**, or
  ~420 per step across 30 layers, before any GEMM. At the ~6-7 us host cost per
  launch measured in iteration 19 that is the bulk of the 8.0 ms/token host gap,
  which is now comparable to the whole attention kernel (8.5 ms of a 22.9 ms
  step). Two candidate directions, in order of expected size: (1) capture the
  decode step as a HIP graph - `hipengine/core/hip.py` already exposes
  `stream_begin_capture`, graph instantiation and `hipGraphLaunch`. **Correction
  (iteration 45):** the claim that "nothing calls them" is stale -
  `hipengine/core/pm4/transport.py:528` calls `graph_instantiate`, the `hipgraph`
  transport is an implemented submission path, and `docs/REFACTOR.md` records its
  measured result (+0.81%/+0.68% at 64 tokens, +1.36%/+1.40% at 32). The
  machinery exists and is exercised; what remains unmeasured is only its value at
  this campaign's rows, which the paragraph below correctly scopes to the
  short-prompt ones; (2) fuse the
  per-layer elementwise chains. **Correction (iteration 27):**
  `HIPENGINE_FUSED_RMSNORM_ROTATE` is not a Gemma 4 lever at all - it belongs to
  the Qwen3.5/PARO MTP verifier path (`hipengine/runtime/qwen35_paro.py`) - and
  its default-off is supported by a recorded observed regression, not by the
  "pending verifier economics" phrase ENVS.md uses: it stayed bit-exact while
  worsening the verifier kernel 13.41 -> 14.09 ms/pass (+5.0%) and the host
  window 18.45 -> 19.05 ms/pass, because the one-block-per-row RMSNorm reduction
  serialises the per-group rotate (`docs/REFACTOR.md`, `benchmarks/CHANGELOG.md`,
  `benchmarks/results/2026-06-08-hipengine-mtp-m15.4-fused-rmsnorm-rotate-neutral.json`).
  A recorded regression is a legal cause under Product Defaults, so this flag is
  not a violation and not a Gemma 4 candidate; fusing Gemma 4's own elementwise
  chain would be a new kernel, not a flag flip. Measure before either: the census above counts launches, not
  their host cost, and no trace-based attribution has been possible since
  rocprofv3 began hanging on this box.
  *Status 2026-09-25 (iteration 25): **accepted**, and it closes the host-gap
  hypothesis rather than confirming it.* Decomposing the per-launch host cost
  found the gap was not in the HIP call: a bare ctypes call with prebuilt args
  costs **1.84 us** and the `signed_kernel_fn` lookup 0.53 us, but the launcher
  wrapper cost **11.81 us**. The extra ~9.5 us was `build_hip` re-deriving its
  cache key on every launch, because the kernel launchers resolve their library
  per call (they pass no `library=`): `build_gemma4_norm(load=True)` alone was
  7.97 us, of which `_resolve_compiler_version` 2.05, the target-arch read 0.85,
  cache-root resolution 0.48, path normalisation ~2.2, and the rest call glue.
  `os.environ.get` costs 0.463 us per read on POSIX (the mapping fsencodes the
  key on every access), so the 8-12 environment reads were the largest single
  term. A fast path in front of the loaded-library cache, keyed on the raw
  request plus a `_BUILD_ENV_KEYS` environment signature, takes the wrapper to
  **5.27 us** and removes **6.4 ms of host time per decode step** (989 build
  calls per step x 6.5 us; 31,647 fast-path hits and 0 slow-path derivations
  across a 1024p+32o generate). Campaign rows: 1024p 43.6297 -> **43.9827
  (+0.8%)**, 512p 47.32 -> 48.1231 (+1.7%), 128p 48.4910 -> **51.9685 (+7.2%)**,
  4096p 28.49 -> 28.5889 (flat), prefill unchanged, public wall improved or held,
  `public_path_parity=true` throughout. The gradient is the finding: **host
  dispatch is hidden behind GPU execution at long prompts and exposed only where
  per-step GPU work is small**, so iteration 19's 8.0 ms/token host gap is not on
  the critical path at the primary row - removing 6.4 ms/token there bought
  +0.8%, which means the remaining 22.9 ms step at 1024p is GPU work. Evidence
  row `2026-09-25-gemma4-26b-a4b-host-dispatch-fastpath-accepted.json`, worklog
  entry
  `20260925T214512.713190Z-lhl-gemma-4-host-dispatch-build-fast-path-removes-6-00f1cb.md`.
  Next: GPU-side work at the primary row, in the order iteration 19's trace
  implies - attention (8.5 ms of the 22.9 ms step, with pass 1's shuffle tree and
  the pass-3 pipeline already improved in iterations 23-24), the MoE decode
  linears (6.2 ms), the dense projections (3.9 ms). HIP graph capture remains
  worth measuring for the short-prompt rows, where host cost is still exposed,
  but it should be scoped by its 128p/512p benefit rather than by the old 8 ms
  figure. (The `HIPENGINE_FUSED_RMSNORM_ROTATE` recommendation in this paragraph
  is withdrawn in the iteration-27 correction above: that flag is Qwen3.5/PARO
  MTP only, and its default-off rests on a recorded regression.)
  *Status 2026-09-25 (iteration 26): **rejected**, and it corrects the diagnosis
  of pass 1's tree.* Iteration 24 measured the tree at 108 us of a 288 us sliding
  launch and iteration 24's 1024-thread regression showed more warps do not help,
  which pointed at a serialised dependent chain inside each warp: the two keys per
  tile interleave their K loads and dot products but then call
  `gemma4_warp_tree` twice, each a 5-level chain of 40 shuffles. A
  `gemma4_warp_tree2` that issues both keys' shuffles per level - bit-exact, same
  register footprint, same kKeysPerTile=2 - was **neutral at sliding and slower
  at full** (sliding 288.4 -> 290.7 us, full 270.1 -> 280.2 us at keys=1024, full
  399 -> 420.6 us at keys=2048) while the campaign row stayed flat
  (43.9827 -> 44.0065, +0.05%). Reverted. The arithmetic says why: 1024 keys x 40
  shuffles x 16 warps is 655k shuffles in 108 us, about 5 cycles per shuffle per
  warp, which is the block's 16 warps saturating its 4 SIMD pipes - so the chain
  was already overlapped (both calls are inlined and independent, and the
  scheduler interleaves them) and the tree is **block-level pipe throughput**, not
  latency. That also explains iteration 24's 1024-thread regression: more warps
  cannot add SIMD pipes to a block. The real limiter is structural and was
  confirmed in the launcher: `grid = tokens * num_heads`, so a decode step
  launches **16 blocks on a 96-CU GPU** with 512 threads each. Pass 1's tree cost
  is therefore irreducible *within one block* and the candidate that can move it
  is **split-K**: divide each head's key range across N blocks and combine their
  (max, denominator, weighted-V) partials with an online-softmax rescale. That is
  changed arithmetic - it reorders the weighted sums - so it needs the
  production-profile gate from `docs/EXECUTION-PROFILES.md`, not parity, and the
  evaluator for it is `scripts/gemma4_teacher_forced_gate.py`. It is also the
  first candidate in this campaign whose payoff comes from occupancy rather than
  from removing work: at 16 of 96 CUs there is room for roughly 4x before other
  limits bind, against an attention share of 8.5 ms in a 22.9 ms step. Evidence row
  *Iteration 27 correction, and the withdrawn recommendation:* the paragraph
  above names split-K as the next candidate on the strength of the block-count
  probe. That recommendation is **withdrawn as written**, because iteration 15
  already built it: `2026-09-24-gemma4-26b-a4b-attention-split-rejected.json`
  records the multi-block attention split at **0.102 vs 0.263 ms/launch (2.6x
  faster, consistent with this probe's headroom) rejected at the production
  gate on kl_max 0.304 against the 0.05 bar**, attributed to bf16-KV
  amplification. The unapplied prototype is
  `/mnt/nvme1/gemma4-eval/gemma4-attention-split-candidate.patch` and it targets
  the pre-iteration-21 block kernel. So the speed is real and known, and what
  fails is the numerics - which the probe cannot see and the gate can.
  *The refinement worth testing instead:* the rejected split gave every slice its
  own max, so each slice's `expf` arguments differ from the single-kernel row max
  and the rescale amplifies bf16 error. A two-phase split keeps the arithmetic
  change to the **association order of the weighted-V sum alone**: phase 1 runs
  the existing kernel unchanged to produce the logits, the row max and the
  denominator (16 blocks, bit-identical to today), and phase 2 splits only pass 3
  across N blocks, each slice accumulating `expf(logit - row_max) * v` over its
  key range with the *shared* row max. The combine is then a plain sum of the
  slice partials divided by the already-final denominator - no per-slice rescale,
  no second `expf`. Every summand's weight is bit-identical to the incumbent
  path; only the f32 accumulation order over the slice's keys changes, which is
  a ~1e-7 relative perturbation rather than a rescale. Pass 3 is the part worth
  splitting anyway: iteration 22's ablation put it at 57% of the sliding kernel
  and 71% of the full one, while passes 1 and 2 together are only 183-204 us. This
  is the variant to build and gate; the per-slice-max variant stays rejected.
  `2026-09-25-gemma4-26b-a4b-tree2-interleave-rejected.json`.
  *Implemented 2026-09-25 (iteration 28).* The two-phase split is built and on the
  default path. Phase 1 is the existing class kernel stopping after pass 2 and
  writing its weights plus a `(denominator, row max)` header; phase 2 is
  `gemma4_attention_decode_slice_kernel`, which accumulates the weighted V sum
  over its slice of the key range; a combine sums the slices in ascending order
  and divides by phase 1's denominator. `decode_slices(keys)` keeps 1 slice below
  512 keys (the single-kernel path, unchanged), 2 from 513-1024 and 4 from 1025
  up, so the metric context's sliding layers run 32 blocks and its full layers
  32-64 - the probe's saturation point. The workspace is per-stream and
  grow-only; every launch the runner makes is on the default stream. Contract:
  phase 1's denominator and every summand's weight are the incumbent path's own
  f32 values, so the sole change is the association order of the f32 weighted
  sum. Measured on the artifact's two real geometries at 1024, 2055 and 8192
  keys, the split agrees with the single-kernel path to 1e-4 relative in f32 and
  within one bf16 ulp, where the incumbent parity tests assert bit-equality. The
  speed mechanism is the rejected per-slice-max split's (0.102 vs 0.263
  ms/launch) with the rescale removed, and the open question is whether
  association alone clears the kl_max bar.
  *Accepted 2026-09-26 (iteration 28): the slice policy was wrong and the
  acceptance rows caught it.* The first shipped policy split from 513 keys. The
  512p row then measured 48.12 -> 46.31, and a paired interleaved A/B (single
  48.0433 and 47.9830 against split 46.5579 and 46.5885 tok/s) confirmed it: a
  512-key context is 3.0% *slower* split than unsplit, because the split's fixed
  cost - a weights round trip through global memory, two extra kernel launches
  and a combine - is not paid back at that length. Raising the entry threshold
  to 1024 keys fixed it (512p back to 48.0158), but the first attempt at the
  fix also held every slice at 512+ keys, which the primary row itself rejected:
  4 slices over keys 1025-1151 measure 45.3015 tok/s against 41.7934 with 2. The
  final policy is therefore an entry threshold of 1024 keys and then aggressive
  growth (doubling while a slice still carries more than 512 keys, capped at 4).
  Final rows, incumbent -> candidate: **1024p 43.9827 -> 45.3015 (+3.0%)**,
  4096p 28.5889 -> 31.7066 (**+10.9%**), 512p 48.1231 -> 48.0158 (flat), 128p
  51.9685 -> 51.9240 (flat). The per-launch A/B, taken with the arms interleaved
  (the first pass was discarded: it reported the split 2.2x slower at
  sliding/1024, which contradicts the end-to-end row, and was taken while a peer
  job held 10.7 GiB of the device), puts sliding/1024 at -17%, sliding/8192 at
  -21%, full/8192 at -20% and full/1024 at +20%. That last cell was the one loss, and
  it is now closed as not worth pursuing: re-measured with both GPUs idle (arms
  interleaved, two passes each), the full geometry at 1024 keys loses 3.5%, not
  the 20% the contended pass reported, and full/2048 is already a 16% win. The
  remaining loss is 5 of 30 layers at 9.6 us each - about 48 us per step, or
  +0.2% end-to-end - against the cost of plumbing head_dim into the selection
  policy. That plumbing was done in iteration 30 for the narrow head only, after
  the read range made it load-bearing (see below); iteration 36 then retired the
  wide-geometry rule as well, having measured it to be wrong in the same
direction. The clean-device pass also puts every other cell 4-10% above its
  contended reading: sliding/1024 -14%, sliding/2048 -24%, sliding/8192 -24%,
  full/8192 -19%. The same A/B answers iteration 27's open
  question - the weighted layer mix predicts ~0.9 ms saved against the 0.72 ms
  measured, so the ~2.4 ms that iteration 22's 57-71% pass-3 share implied was an
  overestimated share rather than an inefficient kernel.
  Evidence row
  `2026-09-26-gemma4-26b-a4b-two-phase-split-accepted.json`.
  *Accepted 2026-09-26 (iteration 30): the sliding layers were walking keys the
  mask had already zeroed, and fixing it forced a correction to the slice policy
  above.* A sliding layer's keep-mask zeroes every key outside its window, but the
  mask is full width, so the kernel walked the whole live context: at context
  4096, 3073 of 4224 keys per layer whose weight is exactly zero, on 25 of the 30
  layers. The walk is what costs - the decode kernel's time tracks `keys`, not
  the number of live keys - and iteration 24's census of per-layer launches had
  not looked at the key range itself. `gemma4_layer_forward_bf16` gained
  `key_begin`, which moves the key, value and mask pointers forward together and
  shortens `keys`; the kernel is untouched, so there is no new ABI. The change is
  **bit-exact rather than close**: a masked key contributes `exp(-inf) = 0` to
  both reductions, the row maximum is unchanged because the dropped entries were
  `-inf`, and removing zero-valued terms does not reorder the survivors. So no
  arithmetic changed and no gate was owed - the 128-token generation is
  byte-identical to the incumbent arm, and `public_path_parity` is true on all
  four rows. Only a one-row block may skip: a prefill block's rows sit at
  different positions and its mask rows are strided by the full key count, so the
  single pointer offset would read the wrong mask row. The public-path check
  confirmed that prefill blocks (rows=512) never skipped.
  **The first measurement was a 5.5% regression (45.2913 -> 42.8098), and the
  cause was the slice policy.** Handing sliding layers exactly 1024 keys instead
  of the whole live context dropped them from 4 slices to 2, because the rule
  kept more than 512 keys per slice - a floor that is wrong for a 256-wide head,
  whose key tile holds twice as many keys per stage. That cost 1.3 ms per step,
  which is the whole regression. `decode_slices` now takes `head_dim`: narrow
  heads take 4 slices as soon as the 1024-key entry threshold is reached
  (measured 192.2 us against 244.4 for 2 at 1024 keys, paired with two passes per
  arm), while the 512-wide `attention_k_eq_v` layers keep the doubling rule
  unchanged. A dead loop after the function's `return`, left by the earlier
  policy edit, was removed in the same pass.
  Final rows, incumbent -> candidate: **1024p 45.3015 -> 45.8427 (+1.2%)**,
  4096p 31.7066 -> **42.7662 (+34.9%)**, 512p 48.0158 -> 47.9667 and 128p
  51.9240 -> 51.7789 flat. The last two are not merely unregressed but untouched:
  with fewer live keys than the window, `key_begin` is 0 and the layer passes
  byte-identical pointer and key arguments to the kernel. The gradient is the
  point - the skipped fraction is what scales, 0-128 of 1152 keys at the metric
  row against 3073 of 4224 at 4096p - so the long-context row moved from 42.8% to
  53.3% of the Qwen stop condition while the metric row gained 1.2%. Intended
  path confirmed through the public `LLM.generate()` surface by spying on both
  selection functions: 525 sliding-layer calls skipped keys (1..7, growing with
  context) and selected `head_dim=256 keys=1024 slices=4`, the 5 full layers
  stayed at 0, and prefill never skipped. Evidence row
  `2026-09-26-gemma4-26b-a4b-sliding-read-range-accepted.json`; the campaign's
  rollup for the three acceptances since 2026-09-24 was owed and is now in
  `benchmarks/CHANGELOG.md` and `benchmarks/README.md`.
  *Rejected 2026-09-26 (iteration 36): the 4-slice cap was never measured and 16
  is much better, but the change makes a binding correctness metric worse.* The
  policy above says more slices is better until the cap and never justified the
  cap; it was an argument, not a measurement. Paired interleaved arms, one pass
  per arm per run, us per launch:

  ===========  ======  ======  ======  ======  ======
  geometry     keys       4       8      16      32
  ===========  ======  ======  ======  ======  ======
  sliding       1024   192.0   168.0   159.3   165.6
  sliding       2048   322.5   288.5   264.3   265.7
  sliding       8192   865.9   708.6   617.4   632.0
  full          1024   227.9   189.8   197.6   197.8
  full          2048   357.4   297.9   292.4   290.6
  full          8192  1043.4   667.5   658.7   609.9
  ===========  ======  ======  ======  ======  ======

  16 is optimal or within 4% at every point but the full geometry at 8192 keys,
  where 32 is 8% better; the doubling rule this function also carried kept more
  than 512 keys per slice and returned 2 slices at exactly 1024 keys, where 64
  keys per slice is the fastest configuration measured. The speed is real:
  **1024p 45.8427 -> 48.2630 (+5.3%)**, 4096p 42.7662 -> 45.9349 (+7.4%), 512p
  and 128p flat, and it passes the production gate against its own incumbent
  (kl_max 0.005212 against the 0.05 bar, zero top-1 flips).

  It is rejected anyway, because it amplifies a correctness failure that this
  iteration's evaluator fix exposed. The two-phase split **as shipped, at 4
  slices, already fails the production kl_max bar against the strict path**:
  kl_max 0.055589 against 0.05, on 1 of 1023 rows. At 16 slices the breach grows
  to 2 rows and kl_max 0.150817. Every other limit passes in every arm
  (kl_mean 9.2e-05 / 2.1e-04 against 0.001, kl_p95 4.0e-06 / 3.5e-06 against
  0.005, kl_p99 4.1e-05 / 3.8e-05 against 0.02, top-1 rate 1.0 with zero flips
  against 0.99). On the outlier rows the teacher is near-deterministic and the
  decision never moves - row 864: top-1 6605 -> 6605, teacher p_top1 0.995465
  against the split's 0.999999; row 602: top-1 6605 -> 6605, teacher p_top1
  0.988141 against 0.999995 - so the divergence is in the tail, and KL from a
  0.995-peaked distribution to a slightly sharper one is dominated by the
  residual spread over the remaining 262143 tokens. The max|delta logit| on
  those rows is 18-20, which is far above f32 association error and is the thing
  to understand next.

  The slice table is kept because the +5.3% is recoverable as soon as that tail
  behaviour is resolved; the policy itself is reverted to the pre-change rule
  (4 slices for a 256-wide head, the doubling rule capped at 4 for a 512-wide
  head). Evidence row
  `2026-09-26-gemma4-26b-a4b-split-16-slices-rejected.json`.

  **The gate itself was broken, and this iteration fixed it.** `capture_chain`
  fed the prompt one token at a time from an empty cache, so key counts ran 1,
  2, 3, ... 1023 - one token below the split's 1024-key entry threshold - and
  `decode_slices` returned 1 for every row: the gate compared the single-kernel
  path with itself and reported `kl_max` of exactly 0.0. The 2026-09-25 split
  gate result (kl_max 0.006746) therefore measured the sliding read range, not
  the split, and the split was promoted to the default path on a gate that could
  not see it. The evaluator now takes `--prefill` (ids pushed through the cache
  in one forward before scoring) and `--slices` (pin the split for one arm; 1
  selects the strict path), records both in the baseline, and refuses to compare
  arms whose scored key range differs. The three arms above are the first real
  gate verdicts this path has had.

  *Diagnostic 2026-09-26 (iteration 37): what the breach actually is.* The
  rejected candidate's row-level record said `max|delta logit|` was 18-20 on the
  outlier rows, which reads as a large arithmetic difference. It is not, and
  neither is the breach a near-tie effect. Against the same frozen strict
  baseline, 1023 paired rows at keys 1025-2047, the shipped 4-slice split gives:

  ==================  =========  =========  =========  =========  =========
  statistic              p50        p90        p99      p99.9        max
  ==================  =========  =========  =========  =========  =========
  row KL              2.592e-08  1.040e-06  3.955e-05  3.611e-02  5.559e-02
  max|delta logit|        1.085      2.797      7.803     22.586     22.586
  ==================  =========  =========  =========  =========  =========

  A one-logit difference is the **median**, not the exception: 571 of 1023 rows
  exceed 1 and 37 exceed 5. It is almost always invisible in KL, because it sits
  in the tail of a peaked distribution - the row with the largest difference of
  all (row 863, 22.586) has KL 0.0000, and `corr(kl, max|delta logit|)` is only
  0.288. The two are largely independent, so the earlier reading of that figure
  as the explanation for the breach was wrong. What the breach actually is:

  - **Two rows of 1023 exceed 1e-3 KL, and one exceeds the 0.05 bar.** The
    distribution is not a broad drift: p50 2.592e-08 and p99 3.955e-05 sit two
    to four orders of magnitude inside their bars, and only the 99.9th
    percentile and the maximum cross any line.
  - **Top-1 never flips on any row.** The candidate comes out *sharper* on the
    outlier rows (p_top1 0.995465 -> 0.999999), so its top logits differ from
    strict's by 5-15, which is far above f32 association error and is what makes
    the tail collapse.
  - **The KL is carried by mid-rank tokens, not the decision.** On row 864 the
    largest single contribution is at rank 7220 of 262144; the top-1 token
    contributes a *negative* 8% of the total.
  - **The divergence is shared across slice counts.** 4 slices and 16 slices
    differ from strict by comparable amounts and in the same direction, and the
    difference between the two split arms is small (kl_max 0.0556 against
    0.1508). A difference that is largely identical at 4 and 16 slices must come
    from the code the two share, not from how the key range is divided: either
    phase 1 - which the design claims is the incumbent kernel's own passes 1 and
    2, and which nothing has verified bit-for-bit - or the combine's division by
    phase 1's denominator.
  - **The prompt is cycled prose.** `exact_prompt_ids` repeats a six-sentence
    corpus to reach 2048 tokens, so the rows that breach are repeated-context
    rows where the model is legitimately near-certain. `kl_max` is a single
    order statistic, and on a peaked reference it is dominated by the residual
    tail: a candidate whose tail is a few orders of magnitude smaller scores a
    large KL while agreeing on every token.

  What that leaves for the campaign is a decision rather than a measurement, and
  it is recorded here rather than taken: either phase 1 is made bit-identical to
  the single kernel's pass 2 (a kernel-level test, and the next experiment -
  compare the split's phase-1 weights against the single kernel's, rather than
  comparing end-to-end logits), or the applicability of an absolute `kl_max` bar
  to a peaked reference on a greedy workload is re-derived with the lead, or the
  strict path returns as the default. The `kl_mean`, `kl_p95`, `kl_p99` and
  top-1 bars - the ones a greedy workload actually depends on - pass in every
  arm with room to spare.

  *Diagnostic 2026-09-26 (iteration 38): the mechanism, and it is not a defect.*
  The split's divergence is a pure association reordering of the weighted-V sum,
  which is what the acceptance record claims it is. Established by reading the
  kernel and by one discriminating measurement.

  - **Phase 1 is faithful.** `gemma4_attention.hip` pass 2 computes
    `weight = expf(logits_s[j] - row_max)`, writes it back into `logits_s[j]`,
    and reduces the denominator with the tree pinned to
    `kGemma4ReduceThreads` (256) so its order matches the block kernel's. The
    split then stores exactly those weights (`split_weights[head_index * keys + j]
    = logits_s[j]`) and a header of `(denominator, row_max)`. Every slice's
    summands are therefore the incumbent path's own f32 values, as designed.
  - **It is not a precision round-trip.** The workspace is `float` throughout
    (`float* __restrict__ split_weights`, and the sizing function multiplies by
    `sizeof(float)`).
  - **It is not the slice count.** Measured against the same strict baseline:
    2 slices kl_max 0.1765 with 2 rows over the bar, 4 slices 0.0556 with 1,
    8 slices 0.0559 with 2, 16 slices 0.1508 with 2; median `max|delta logit|`
    is 1.03-1.15 in *every* arm including 2; and the per-row KL correlates
    0.790/0.987/0.821 between 2/8/16 slices and 4. Row 864 sits at
    0.056-0.060 in all four. A 2-slice combine can differ from the single
    kernel's chain by one rounding step, so a shared one-logit difference at 2
    slices localises the source to the split's existence rather than its width.
  - **What is left is the association order, and it cannot be removed.**
    Pass 3 is a strictly sequential f32 chain over ascending keys; any split of
    that range computes partial sums in parallel and combines them, which
    reassociates the sum by construction. A slice cannot reproduce a sequential
    dependency chain over the whole prefix without giving up the parallelism
    that is the split's entire purpose, and a higher-precision accumulator would
    still not be bit-identical to the strict f32 chain. So the split's arithmetic
    difference from strict is irreducible, and the only lever is how much it
    matters.
  - **How much it matters:** p50 row KL 2.592e-08, p90 1.040e-06, p99 3.955e-05
    against bars of 1e-3, 5e-3 and 2e-2; top-1 rate 1.0 with zero flips on all
    1023 rows; 2 rows over 1e-3 and 1 over the 0.05 max bar. The one-logit
    median difference is real but sits in the tail of a peaked distribution, and
    on the breaching rows the model's decision does not move at all.

  So the campaign's choice is between accepting the split under the aggregate
  bars plus top-1 and treating an absolute `kl_max` as inapplicable to a
  reordering-class change, and returning to the strict path. That is a lead
  decision, recorded here with the mechanism rather than taken. Making phase 2
  bit-identical is not available; that option is closed with the reason above.
  *Diagnostic 2026-09-26 (iteration 31): the MoE decode linears are now the
  largest single item in the step and the achieved-bandwidth baseline did not
  exist.* With attention down to roughly 8.5 ms and the host gap cut by 6.4 ms,
  the MoE's 6.2 ms/step from iteration 19's profile is **28% of the ~21.8 ms
  step**. A shape-substituted run of
  `scripts/gguf_q4_k_moe_ffn_fused_microbench.py` (hidden 2816, ffn 768 - 704 is
  not a multiple of 256 and the synthetic-weight fixture refuses it, so 768 is
  the nearest legal substitute at ~9% wider; the real kernels handle 704, so that
  is a fixture limit, not a kernel one) puts the production unfused chain at
  **0.3002 ms per layer for 31.35 MB, ~104 GB/s or 12% of the RX 7900 XTX's
  peak**. The memory roofline for those bytes is ~35 us per layer, against 300 us
  here and 207 us per layer in production: **the MoE is 6-9x off the roofline and
  is latency-bound, not bandwidth-bound**.
  Two things follow. The **fused megakernel is refuted** - 0.5545 ms against
  0.3002, so consolidating three launches into one is 1.85x *slower* at these
  shapes, and it applies SiLU where Gemma 4's MoE is `gelu_tanh`, which is why
  `gemma4_moe.py` registers its own GEGLU. And the campaign's own next candidate
  is re-scoped rather than unblocked: iteration 15 recorded the q4_k decode
  re-route behind three obstacles, and the `expert_ffn % 256` guard splits
  differently than recorded - the gate_up projection's in_features is 2816
  (11 x 256, legal) and only the down projection's 704 is not. The decode-shaped
  kernels do exist, in the t16 family (`gguf_t16_selected_gemv`, registered under
  `gguf_q4_k_t16_v1` and siblings), but they read a **repacked** weight layout,
  so reaching them means a loader-side repack plus the routing change - G4/G5
  scope, not a one-iteration edit. Diagnostic only: no product path changed, no
  row moved. Evidence row
  `2026-09-26-gemma4-26b-a4b-moe-bandwidth-probe.json`.
  *Diagnostic 2026-09-26 (iteration 32): the launch structure is not the MoE's
  problem, and the cheap refactor is refuted.* `gemma4_experts.py` states its own
  structure - "one launch per non-empty expert with a pointer offset into the
  stacked expert weights, not one launch per token-lane" - so at decode, where
  top_k 8 makes all 8 selected experts non-empty, a layer issues **8 gate_up
  GEMVs and 8 down GEMVs**, plus a memset, three grouping kernels, a gather and
  the weighted accumulate. Sixteen GEMV launches per layer looked like the
  obvious lever, and at ~5.3 us of host cost each it is ~2.5 ms per step of
  dispatch work.
  It is the wrong lever, for two measured reasons. First, each launch carries
  ~13 us of GPU work (0.207 ms per layer over 16 launches), so the host cost is
  *hidden* behind execution - the same finding iteration 25 reached from the
  other direction. Second, and decisively, the microbench's **selected** form
  does the same layer in 3 launches and reaches **104 GB/s, the same ~11% of
  peak** as the per-expert form. Two different launch structures, one bandwidth.
  So neither fusing the launches nor skipping the grouping pipeline (which is
  provably redundant at one token, where every lane carries the same hidden row)
  would move the number: **the cost is the kernel's memory access pattern**, and
  the fix is the vectorized/repacked route, not the orchestration. The decode
  fast path is recorded here as refuted rather than left as an attractive
  unmeasured idea.
  The dense projections were checked the same way and are also not mis-routed:
  both `gemma4_project` and `gemma4_project_experts_selected` go through the
  GGUF dispatch, which selects on row count, so rows == 1 already takes the
  decode branch. Their 3.9 ms is kernel quality too, and the same repack applies.
  Diagnostic only: no product path changed, no row moved.

  *Next candidate 2026-09-26 (after iteration 38): measure the t16 decode GEMV
  before spending a loader project on it.* Iteration 31 named the repacked t16
  route as the MoE's fix and scoped it as a loader-side repack plus a routing
  change, and the campaign has been treating that as the next real lever. The
  assumption underneath it has never been tested: **the t16 decode kernels have
  never been measured at Gemma 4's shapes.** Iteration 31 measured the production
  unfused chain (0.3002 ms/layer, ~104 GB/s) and iteration 32 measured the
  selected form (3 launches, the same ~104 GB/s); the t16 family's speed is
  inferred from the Qwen3.5/3.8 lanes it was built for, not measured here.

  That matters because the repack is a multi-iteration loader job. The decode
  family is `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.py` -
  `gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out` and its natural, tile8,
  parallel and pairreuse variants - and it is registered under
  `gguf_q4_k_t16_v1`, but it reads a repacked weight layout and is reached today
  only through the Qwen profiles (`qwen36_gguf_gfx1100_profiles`,
  `qwen38_gguf_profiles`). Reaching it for Gemma 4 means building a Gemma 4
  repack path, which is exactly the kind of work that should not be started on an
  assumed number.

  The measurement is cheap and decisive: a synthetic t16-layout fixture at the
  MoE decode shapes (hidden 2816, ffn 704/1408, top_k 8, 128 experts), run
  against the same harness iteration 31 used, compared to the production
  selected GEMV's 104 GB/s. If t16 is substantially faster the repack is
  justified and should be planned as its own unit; if it is also ~104 GB/s the
  repack is refuted before it is built, and the MoE needs a different answer.
  `scripts/gguf_q4_k_moe_ffn_fused_microbench.py` has no t16 arm today, so the
  fixture is the work item. Recorded rather than started here: it is a new unit,
  and the two diagnostics above closed this one.

  *Diagnostic 2026-09-26 (iteration 39): the MoE is two different quants, and the
  repack can only ever cover one of them.* A metadata scan of the real artifact
  (`scan_gguf`, no GPU) settles the eligibility question before any fixture work,
  and the answer is not what the campaign's record assumes.

  - **`ffn_gate_up_exps.weight`, ggml_shape (2816, 1408, 128): Q4_K in 29 of 30
    layers, Q5_K in one.** The campaign's Q4_K description of the MoE is correct
    *for this tensor*. It is **t16-eligible**: the repack shape is rank 3,
    `out_features` 1408 is divisible by 16, and `in_features` 2816 is divisible
    by 256, so `bytes_per_row` 1584 is a multiple of the Q4_K block.
  - **`ffn_down_exps.weight`, ggml_shape (704, 2816, 128): Q5_1 in 29 of 30
    layers, Q8_0 in one.** It is **not t16-eligible, and cannot be**: no t16
    repack shape is registered for Q5_1 at all, and the single Q8_0 layer's
    registered shape is rank 2 (dense), while 704 is not divisible by 256.
  - **The reason is structural, not policy.** Q4_K's block is 256 elements, so a
    704-element row cannot be Q4_K at all; the down projection is Q5_1 (block 32)
    because that is what can represent it. No amount of repack work makes the
    Q4_K t16 route cover `ffn_down_exps`.
  - **So the repack's ceiling is `ffn_gate_up_exps`: 59.8% of the MoE's 14.42 GB
    of expert weights**, not all of it. That is still a large prize, but the
    campaign should hold the number it is actually chasing.

  It also corrects the measurement baseline underneath iteration 31. That
  microbench ran a Q4_K chain at **ffn 768** - a 9%-wider substitute for
  gate_up's real 1408 - and gave the *down* arm Q4_K weights, but the production
  down projection is Q5_1. So the down projection, 40% of the MoE's expert bytes,
  **has no production-shaped bandwidth measurement at all**. Its headroom is a
  Q5_1-specific question rather than a repack one, and iteration 33 already
  showed it resolving to `selected_gemv_bf16_bf16_out`, the same kernel the
  104 GB/s figure came from. The t16 measurement stays the next candidate, now
  scoped to gate_up's real shape. Diagnostic only: no product path changed, no
  row moved. Evidence row
  `2026-09-26-gemma4-26b-a4b-moe-repack-eligibility.json`.

  *Refinement 2026-09-26 (after iteration 39): the down projection is measurable
  at its real shape, and it is the larger unmeasured prize.* Iteration 31's
  harness refused `ffn_len` 704 and substituted 768, and recorded that as "a
  fixture limit, not a kernel one". The scan above explains *why* the fixture
  refused it and why the substitute was the wrong shape to reach for: that
  harness packs **Q4_K**, whose block is 256 elements, so a 704-element row
  cannot be represented - the same reason the artifact's own down projection is
  **Q5_1**, whose block is 32, and 704 = 22 x 32 exactly. So a Q5_1 arm measures
  the production tensor at its true shape with no substitute, and it is worth
  measuring first: `ffn_down_exps` is 5.79 GB against gate_up's 8.63 GB, so at
  top_k 8 of 128 experts it is **362 MB/token** against gate_up's 539 MB/token,
  40% of the MoE's expert bytes. At the roofline that is 0.377 ms/token against
  the ~2.5 ms the campaign's 6.2 ms/step MoE budget implies for it, so the same
  6-7x headroom is likely present and has never been confirmed. The t16 question
  and the Q5_1 question are independent: t16 can only ever address gate_up, and
  Q5_1 has no t16 shape, so the down projection's answer has to come from a
  Q5_1-shaped kernel measurement rather than from a repack. New campaign-scoped
  script `scripts/gemma4_moe_q5_1_bandwidth.py` is the work item.

  *Diagnostic 2026-09-26 (iteration 40): the production dispatch table, and the
  MoE issues two launches per layer rather than sixteen.* A probe on a real
  `LLM.generate()` (`scripts/gemma4_moe_dispatch_probe.py`) records what the
  expert path actually runs. It also corrects the method the campaign has been
  using to ask: the experts resolve through `registry.resolve` with
  `_SELECTED_VARIANT` inside `gemma4_project_experts_selected`, **not** through
  `resolve_gguf_linear_dispatch` - so a probe patching the GGUF linear dispatch
  alone reports dense projections and **zero experts**, which is how this was
  found.

  - **Both expert projections are served by the selected family.** gate_up is
    `gguf_q4_k` at (2816 -> 1408, 128 experts) and down is **`gguf_q5_1` at
    (704 -> 2816, 128 experts)**, both at **rows = 8**. Iteration 33's 783
    `gguf_q4_k` and 783 `gguf_q5_1` resolutions were therefore the *expert*
    tensors, not the dense projections they were attributed to.
  - **rows = 8 is one token's top_k, and the selected family batches all 8 into
    a single launch.** Per decode step the MoE issues **2 selected GEMVs per
    layer**, not the 16 that iteration 32's record assumes. Measured: 87
    launches of each across 30 layers, the excess over 60 coming from the single
    layer whose experts are Q5_K/Q8_0. Iteration 32's *conclusion* - that the
    cost is the kernel's memory access pattern rather than the orchestration -
    survives, and this is its production-side confirmation.
  - **The bandwidth arithmetic, on production shapes.** gate_up stores 287 MB per
    layer and down 193 MB, 480 MB stored; one token reads 8/128 of it, **30.0 MB
    per layer, 0.90 GB per token**. At the 960 GB/s roofline that is **0.94
    ms/token**. Production spends 0.207 ms per layer (iteration 19's profile),
    which is **145 GB/s - 15% of peak, 6.5x off the roofline.**
  - **Iteration 31's microbench was a fair proxy after all.** Its fixture
    measured 31.35 MB against a real per-token read of 30.0 MB, and its 104 GB/s
    sits *below* production's 145 GB/s. The synthetic stand-in was right-sized.
  - **The remaining question is kernel-internal, and it is now the only one
    left.** Launch overhead, routing and fusion are all refuted by measurement,
    and the Q5_1 down projection is confirmed to take the same kernel. So why
    does `selected_gemv_bf16_bf16_out` reach 15% of peak with 8 batched expert
    rows and one launch per projection? The remaining candidates are
    vectorization and occupancy over Q4_K's 144-byte and Q5_1's 24-byte blocks -
    a G4/G5 kernel-quality job, not a repack and not a flag. Diagnostic only: no
    product path changed, no row moved. Evidence row
    `2026-09-26-gemma4-26b-a4b-moe-dispatch-table.json`.

  *Diagnostic 2026-09-26 (iteration 41): the expert path has untested registered
  variants, and one constant is why they are unreachable.* The MoE's remaining
  candidate is kernel-internal, so the cheap question is whether a faster
  *registered* variant already exists - the move that found a real 13% win for
  the dense projections in iteration 34.

  - **`gguf_q5_1` has three decode variants that have never been A/B'd**:
    `selected_gemv_logical256_t128_bf16_bf16_out`,
    `selected_gemv_logical256_t64_bf16_bf16_out` and
    `selected_gemv_wave64_bf16_bf16_out`. This is the **down projection**, 40% of
    the MoE's expert bytes, and their names describe tiling and wave-size
    choices - which is exactly the "vectorization and occupancy" hypothesis.
  - **`gguf_q4_k` has twelve**, including `selected_dual_gemv_bf16_bf16_out` and
    `selected_pack8_gemv_bf16_bf16_out`.
  - **None of them can be reached by the obvious edit.** Registration is
    per-quant, and `selected_gemv_bf16_bf16_out` is the *only* variant registered
    for all four quants the expert path uses (q4_k, q5_1, q5_k, q8_0). The q5_1
    alternatives are q5_1-only; `selected_pack8_gemv_bf16_bf16_out` covers
    q4_k/q5_k/q8_0 but not q5_1; `selected_dual_gemv_bf16_bf16_out` is q4_k only.
    So `_SELECTED_VARIANT` is a single constant *because* it has to serve every
    quant, and pointing it at a q5_1 variant would silently drop the q4_k path to
    the per-expert offset fallback - a different route, not an isolated A/B.
  - **The legal route is a registry-keyed preference, not a quant branch.** A
    preference list resolved per candidate variant through `is_registered` picks
    the best variant each quant *declares support for*, which is a check on
    kernel capability rather than a quant-name test, and is the shape
    `gemma4_project_experts_selected` already uses to decide whether the selected
    family serves the weight at all. That is the next candidate and it is a
    product change with a measurable A/B and a guard, not another diagnostic.
    Diagnostic only here: no product path changed, no row moved.

  *Measured 2026-09-26 (iteration 42): both registered gguf_q5_1 decode variants
  are about twice as slow, and the variant-reuse line is closed for the MoE.* The
  A/B the previous entry called for was run, and it lost decisively.

  - **Arm A**, `selected_gemv_logical256_t128_bf16_bf16_out` on the down
    projection only: **24.7049 tok/s, -46.2%.**
  - **Arm B**, `selected_gemv_wave64_bf16_bf16_out` on the down projection only:
    **24.3347 tok/s, -47.0%.**
  - **Control**, the incumbent: **45.9396 tok/s.**
  - Both arms land at the same ~24.5, which points at the exact-logical-256 path
    being untuned for this shape rather than at a difference between the two
    variants.
  - **The isolation was proven, not assumed.** The switch was a variant
    preference list resolved per candidate through `is_registered`, so each quant
    takes the first variant it declares support for with no quant-name branch.
    With the incumbent first the dispatch probe's table is byte-identical to
    iteration 40's; with a q5_1 variant first, `gguf_q5_1` resolves to the new
    variant while `gguf_q4_k`, `gguf_q5_k` and `gguf_q8_0` all still resolve to
    the incumbent, checked with the same loop production runs.
  - **The machinery was reverted.** With the variants refuted it had no effect,
    and `gemma4_experts.py` is byte-identical to HEAD (empty `git diff`), so this
    iteration's product change is zero.
  - **The MoE's cheap-candidate space is now exhausted.** Fused megakernel
    (iteration 31), launch structure (32), routing (33) and registered-variant
    reuse (33 for q4_k, 42 for q5_1) are all refuted by measurement, which
    satisfies the three-failed-candidates rule, and the re-profile it calls for
    has been done (iteration 40). What remains is either authoring a new selected
    GEMV for these quants - a G4/G5 kernel job - or closing the MoE line. That is
    a lead-level investment decision, and it now sits alongside the split's
    pending accept-or-return decision. Evidence row
    `2026-09-26-gemma4-26b-a4b-q5_1-variant-ab-rejected.json`.

  *Analysis 2026-09-26 (iteration 43): where the target actually sits, and the
  sliding-window line closed.* Two things were settled, one negative and one
  arithmetic.

  - **The sliding window is already honored, so there is no lever and no bug.**
    The artifact carries `sliding_window = 1024` with 25 of 30 layers sliding,
    and the sliding layers also use a *different* RoPE (`dimension_count_swa`
    256 against 512, `freq_base_swa` 10000 against 1000000). `runtime/gemma4.py`
    honors all of it: `key_length_swa` and `swa_rope` are selected per layer
    type, `sliding_window` is passed to the attention, and `_sliding_read_range`
    computes `max(0, live - window)` for decode so the kernel does not walk keys
    outside the window. A windowed read that is arithmetic-preserving was the
    last unblocked idea on attention; it is already implemented.
  - **The campaign target sits above the reference implementation.** The target
    is 70% of Qwen3.6-35B-A3B's 114.57, i.e. **80.2 tok/s**. The doc also records
    llama.cpp at **68.92 tok/s on this same artifact and hardware**, so the target
    is **16.4% above** what the reference implementation achieves, and the
    comparison is not like-for-like: Qwen3.6-35B-A3B has fewer active parameters
    (A3B against A4B).
  - **The reachable envelope from the work already identified.** Current 45.94
    (step 21.77 ms, 66.7% of llama.cpp). Taking the MoE from its production 6.2 ms
    to its measured 0.94 ms roofline saves 5.26 ms -> 60.58 tok/s (87.9%). Adding
    the split's measured +5.3% -> **63.79 tok/s, 92.6% of llama.cpp**, step 15.68
    ms. That is an *upper bound* on that path, because it assumes a MoE kernel
    that does not exist yet; the split half is measured.
  - **Recommendation, presented not taken:** re-base the campaign target on the
    same-artifact llama.cpp reference, where 92.6% is a near-term goal from work
    already identified, rather than on 70% of a different model's throughput.
    Reaching 80.2 would require beating llama.cpp on attention and the dense
    projections as well. Evidence row
    `2026-09-26-gemma4-26b-a4b-target-gap-arithmetic.json`.

  *Correction 2026-09-26 (iteration 44): the split is already the default, so its
  gain is banked and the open decision is keep-or-rollback.* The campaign has
  been describing the split as unrealized - "if the split is accepted, the
  reverted slice table is worth +5.3% at the metric row" - and that framing is
  wrong. Checking the runtime rather than the narrative:

  - **It is on, unconditionally.** `gemma4_attention.py:312` calls
    `slices = decode_slices(key_count, head_dim)` with no guard, and the file
    contains no `HIPENGINE_*` environment gate. The slice table is
    `keys < 1024 -> 1`, `head_dim <= 256 -> 4`, and otherwise 2 up to 1024 keys
    and 4 from 2048.
  - **The metric already contains it.** `decode_slices`' own docstring records
    "the row measures 45.33 tok/s with 4 slices over keys 1025-1151", which
    matches this campaign's measured 45.06-45.94 range on this tree. The current
    45.9 tok/s **is** a split number, not a pre-split one.
  - **So the decision is keep-or-rollback, not promote-or-not.** Rolling it back
    would cost roughly 5% of the metric. Iterations 36-38 established that it
    breaches the absolute `kl_max` bar (2 of 1023 rows above 1e-3) through
    irreducible association reordering, while the aggregate bars pass with room
    (p50 row KL 2.592e-08, p99 3.955e-05) and top-1 is 1.0 with zero flips. The
    question for the lead is whether an absolute `kl_max` applies to a
    reordering-class change at all, and the cost of saying no is now correctly
    stated as a rollback cost rather than a foregone gain.
  - **One defect fixed while checking.** `decode_slices` carried an unreachable
    second copy of its slice logic after an unconditional `return`. Removed;
    behaviour is provably unchanged and the slice table was re-verified across
    keys 512-8192 at both head dims. Evidence row
    `2026-09-26-gemma4-26b-a4b-split-is-default-banked.json`.

  *Analysis 2026-09-26 (iteration 45): the profile is closed, and the last slice
  is launch latency with no cheap fix.* The re-profile the three-failed-candidates
  rule requires is now complete: every part of the 21.8 ms decode step is
  attributed.

  - **The elementwise chain is not a bandwidth problem.** The 14 elementwise
    launches per layer (iteration 26's census) each read and write the 2816-wide
    bf16 hidden row, so the whole chain moves about **4.7 MB per token** -
    **0.005 ms** at the roofline.
  - **The 3.2 ms unattributed slice is GPU-side launch latency.** 420 launches at
    5-8 us of gap each is **2.10-3.36 ms**, which closes the profile:
    18.6 + 3.2 = 21.8 ms. Iteration 25 already established host cost is hidden at
    this row, so these are inter-kernel GPU gaps, not launcher cost.
  - **Fusing it is refuted twice, and the composite that exists is Qwen's.**
    `docs/REFACTOR.md` records two bit-exact, launch-reducing fusions that lost on
    wall time: the Qwen rmsnorm+rotate (13.41 -> 14.09 ms/pass, because the
    one-block-per-row RMSNorm reduction serialises the per-group rotate) and
    `HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED` (932 -> 922 calls/pass, 26.925 ->
    27.010 ms/cycle). The registered
    `split_qgate+head_rmsnorm+partial_rotary` composite is `gguf_qwen35_*`, keyed
    to a `qwen35_position_f32` variant, so it is not reusable here.
  - **Graphs are already implemented, measured, and scoped elsewhere.** The
    `hipgraph` transport is live and `docs/REFACTOR.md` records +0.68% to +1.40%,
    and line 450 above already scopes it to the short-prompt rows where host is
    exposed. The stale "nothing calls them" claim in the iteration-26 entry is
    corrected in place.
  - **So reducing this slice needs new engineering, not a candidate off the
    shelf**: fusion that does not serialise a reduction before a dependent
    transform, or a graph that helps where host is already hidden. Evidence row
    `2026-09-26-gemma4-26b-a4b-profile-gap-closed-launch-bubbles.json`.
  *Diagnostic 2026-09-26 (iteration 33): the routing hypothesis is closed, and
  the campaign's three-cheap-candidates rule is now in play.* A resolve-level
  probe over a real 1024-token `LLM.generate()` - spying on
  `kernels.registry.resolve` and its `gguf_linear` binding - shows every decode
  projection already taking a decode-shaped GEMV: `gguf_q4_k`,
  `gguf_q5_1` and `gguf_q5_k` all resolve to
  `selected_gemv_bf16_bf16_out` (783 / 783 / 27 resolutions), and `gguf_q8_0`
  takes `pack8_gemv_bf16_bf16_out` at decode while its prefill rows take
  `exact_prefill_tile16x4_bf16_bf16_out` (205 each).
  That `gguf_q4_k` selection is **the same kernel iteration 31 measured at 104
  GB/s** - `gguf_q4_k_selected_gemv_bf16_bf16_out` - so the measurement and the
  production path are the same code and the loop is closed. There is no faster
  registered variant to re-route to: `local32_fixed_meta_gemv_decode` exists for
  q4_k but only with f32 output, which the bf16 scratch cannot consume, and the
  `pack8_gemv_decode_*` alternative was rejected at the gate in iteration 18
  (kl_max 2.447, 32/1023 top-1 flips) with no measurable speed win in the first
  place. This also corrects iteration 15's note, which described the q4_k decode
  path as running a prefill-shaped kernel; it does not.
  **Three cheap candidates are now closed by measurement rather than argument:**
  the fused MoE megakernel (1.85x slower at these shapes, and the wrong
  activation), the launch-structure refactor (the host cost is hidden and the
  selected form reaches the same 104 GB/s in 3 launches instead of 16), and the
  routing change (every projection is already on a decode GEMV). What remains is
  **kernel work, not dispatch work**: either a Q4_K/Q5_1 decode GEMV that reads
  the plain block layout with real memory-level parallelism, or the t16 repack
  that makes the existing vectorized family reachable. The campaign's
  three-failed-candidates rule therefore applies - the next step is a re-profile
  or a scope decision, not another routing probe.
  *Diagnostic 2026-09-26 (iteration 34): that closure was premature, and a live
  candidate came out of testing it.* A direct A/B of the two Q4_K decode GEMVs at
  a dense projection shape (rows 1, in 2816, out 4096, 6.49 MB of Q4_K, paired
  passes) gives **plain `gemv_bf16_bf16_out` 0.0582 ms (111.6 GB/s) against
  `selected_gemv_bf16_bf16_out` 0.0659 ms (98.5 GB/s) - the plain kernel is
  13.3% faster, and the two are bit-identical (0 bf16 ulp)**. So there *is* a
  faster registered variant; iteration 33's "no faster registered variant
  exists" was wrong.
  The cause is a declared routing table, not a fallback:
  `hipengine/loading/gguf_selected_contract.py` lists
  `("Q4_K", "linear", "gguf_q4_k", "selected_gemv_bf16_bf16_out", ...)` in
  `RAW_SELECTED_CONSUMERS`, and the resolve probe shows the dense projections
  requesting exactly that variant. A dense projection has no expert indirection
  to justify the selected ABI. **The contract file is out of this loop's scope
  and is shared with other models, so it was not changed.** The in-scope route is
  `gemma4_project` in `kernels/hip_gfx1100/gemma4/gemma4_layer.py`, which can pass
  `registered_variant=` to `launch_gguf_linear` (the hint is consumed by
  `_registered_variant_dispatch`, which is row-conditioned); it is also the right
  seam because `gemma4_project` carries the dense projections while the routed
  experts go through `gemma4_project_experts_selected`.
  **The candidate to land next:** prefer the plain GEMV for Gemma 4's dense
  projections at decode, guarded by an `is_registered` check cached per quant so
  the check does not run per launch. Worth roughly 13% of the dense projections'
  3.9 ms, about 0.5 ms per step or 2% of the metric, and it is bit-identical so
  it needs token-id parity rather than a numerical gate. Not landed here because
  it needs the full validation set - verify, guard, and the other primary rows -
  which does not fit in the iteration that found it.
  *Reverted 2026-09-26 (iteration 35): the microbenchmark win did not transfer,
  and the change cost 10.4%.* Passing
  `registered_variant="gemv_bf16_bf16_out"` from `gemma4_project` was built,
  unit-tested and measured end-to-end: **45.8427 -> 41.0714 tok/s (-10.4%)**,
  with token ids still identical to the incumbent arm, so the change was correct
  and simply much slower. The hint is not a free preference. `launch_gguf_linear`
  has five dispatch paths gated on `registered_variant is None`
  (lines 3108/3123/3186/3229/3267), and passing a variant suppresses all of
  them; the host cache is not the cause, because `registered_variant` is part of
  the cache key at line 3343, so this is a kernel-selection effect rather than a
  re-planning cost. A 10.4% step loss is roughly a 55% slowdown of the dense
  projections, which is far more than one shape's 13.3% could explain, so the
  plain GEMV must be substantially slower at the other projection shapes (the
  k and v projections are far narrower than the q projection the A/B used).
  **The lesson is the exact inverse of iteration 34's:** there, a cheap
  measurement beat a paragraph of reasoning; here, a cheap measurement at *one
  shape* was over-extrapolated to the whole model. A shape-local microbenchmark
  can refute a claim, but it cannot establish an end-to-end win, and the
  end-to-end row is what decides. The candidate is closed for this loop: the
  reachable fix is in the shared caller contract
  (`hipengine/loading/gguf_selected_contract.py`) or in the dispatch's own
  preference logic, both outside this loop's scope, and the honest follow-up is
  a note to that owner rather than a narrower hint from here.
  *Gate verdict 2026-09-25: **passed**, and comfortably.* Against the frozen
  8192-context baseline over all 1023 teacher-forced rows: **kl_max 0.006746**
  (bar 0.05), kl_mean 9.03e-06 (0.001), kl_p95 5.25e-06 (0.005), kl_p99
  6.12e-05 (0.02), **top-1 rate 1.0 with zero flips** (bar 0.99). That is 45x
  under the rejected per-slice-max variant's 0.304, which is the whole point of
  the two-phase design: with no per-slice max there is no rescale to amplify
  bf16 error, so what remains is the association order of an f32 sum. Verdict
  artifact `benchmarks/results/2026-09-26-gemma4-26b-a4b-two-phase-split-gate.json`;
  the split is now the default decode path above 512 keys and
  `decode_slices`'s thresholds are pinned by a unit test.
  *The gate ran on GPU0 (W7900), not GPU1, and why that is sound.* Three
  attempts on GPU1 died in `hipMalloc` - a peer worker's job holds ~6.9 GiB of
  its 24 GiB, leaving less than the model's footprint (the same OOM reproduces
  at context 4096, so the weights rather than the KV cache are what do not
  fit, and the split's own workspace is under 1 MiB). GPU0 is a W7900: the same
  gfx1100 ISA, the same compiled kernel, so the arithmetic is identical and a
  numerics verdict transfers. What does *not* transfer is performance - the
  campaign's timing rows must still be measured on GPU1, whose clocks and
  bandwidth differ. The gate's verdict JSON carries no provenance block, so the
  device is recorded here and in the worklog entry instead.
  *Measured 2026-09-25 (iteration 28):* **45.4170 tok/s** median of 3 samples at
  1024p/128o against the incumbent's 43.9827 - **+3.26%**, 22.74 -> 22.02 ms per
  step, 0.72 ms saved. That is far below what the probe's 3.3x per-head scaling
  at 32 blocks implies if pass 3 carried iteration 22's 57-71% share: halving a
  57% pass 3 would have saved ~2.4 ms, not 0.72. So either the real pass-3 share
  of the production step is near 17% rather than the ablation's figure, or the
  slice kernel is less efficient per unit work than the whole-kernel probe
  suggested. `scripts/gemma4_attention_decode_bench.py` can separate those two
  in one bounded run at the real geometry, and it should be run before any
  further split tuning. The split is kept because it is measured-positive and
  correct, not because the mechanism is understood yet.
  *Iteration 26 closing diagnostic (`scripts/gemma4_attention_scale_probe.py`,
  sliding geometry head_dim=256 keys=1024, 30 iterations):* varying the head count
  varies the number of launched blocks, since `grid = tokens * num_heads` and each
  head reads its own KV rows. Measured - **16 heads/16 blocks 282.1 us (17.63
  us/head), 32/32 172.4 (5.39), 64/64 164.2 (2.56), 128/128 205.9 (1.61)**. Per-head
  time improves **6.9x** from 16 to 64 blocks while per-launch time falls only
  1.7x, and 128 blocks is worse in total time than 64, so the device saturates
  somewhere around 64 blocks. At the production 16 blocks the kernel moves 16 MB
  per launch in 282 us - about **57 GB/s of a ~960 GB/s peak, on 16 of 96 CUs**.
  The decode attention kernel is therefore parallelism-starved, not
  traffic-bound or reduction-bound, and split-K is the candidate with the right
  shape: it adds blocks while keeping total traffic constant, which is strictly
  better than what this probe does (its traffic grows with the head count). The
  measured ceiling is the 64-block row: a 4-way split should take a decode step's
  sliding layers from ~282 us toward ~165 us plus the combine, against an
  attention share of 8.5 ms in a 22.9 ms step.
  *Status 2026-09-25 (iteration 24): **accepted**. Pass 1's shuffle tree was the
  measured bottleneck (clean ablations, each keeping the data dependencies alive:
  replacing the tree with in-lane adds saved 108 us of 318 at sliding, skipping
  the pass-1 key loop 158 us, while the K loads cost only 25-31 us and the pass-2
  loop and the LDS logit store cost nothing). It is latency-bound, not
  throughput-bound - 40 warp-wide shuffles per key in a dependent chain with only
  8 warps per block and 16 blocks per decode step - so the block is now **512
  threads with 16 key classes and keys-per-tile 2** (swept 2/4/8 ->
  290.3/294.1/303.1 us sliding, 271.2/280.4/422.2 us full; 1024 threads was
  worse on both geometries). The denominator reduction is **pinned to the block
  kernel's 256 lanes** via `kGemma4ReduceThreads`: pass 2 runs on group 0 and the
  other 256-thread groups execute `gemma4_decode_tree1` with an empty sum only to
  reach its barriers, because the tree's addition order depends on its lane count
  and a 512-thread block would otherwise round the denominator differently
  (measured: 3.3% of elements off by 1-2 bf16 ulps). Parity caught both
  512-thread defects before any row was taken. Microbenchmark: sliding 318.9 ->
  288.4 us and full 339.8 -> 270.1 us at keys=1024, full 509.5 -> 399 us at
  keys=2048. Campaign metric **41.9049 -> 43.6297 tok/s (+4.1%)**, **+171% over
  the 16.0931 baseline**; 512p 46.43 -> 47.32, 4096p 25.78 -> 28.49, 128p 48.56
  -> 48.20 (inside the 47.6-48.8 band that row has held for three iterations,
  re-measured with a bounded paired rerun), prefill unchanged, public parity
  true. The engine is at **63% of the same-artifact llama.cpp reference** (68.92
  tok/s). Evidence row `2026-09-25-gemma4-26b-a4b-wide-block-accepted.json`,
  worklog entry
  `20260925T212657.692407Z-lhl-gemma-4-decode-attention-512-thread-block-for-th-423bf9.md`.
  Next: pass 1's tree still issues 40 shuffles per key and any cheaper reduction
  changes the addition order, which makes it a changed-arithmetic candidate that
  needs the production-profile gate rather than parity; and the 8.0 ms/token host
  gap from iteration 19 (1116 launches x 7.2 us) is now larger than the whole
  attention kernel.
- [ ] **G4 — Integrated confirmation.** Repeat the primary paired matrix,
  correctness/heldouts and live chat/SSE; verify no hidden fallback. Re-measure
  llama.cpp and Qwen with the frozen comparison contract and report differences
  without conflating model quality or hardware.
- [ ] **G5 — Closure.** Publish accepted evidence and rejected/inconclusive
  candidates, update the model guide, and leave a clean committed worktree with
  explicit follow-on work. Run the milestone `--suite all` gate once; use the
  repository's focused-repair rule for isolated failures rather than blindly
  repeating the entire suite.

After each measured candidate, record the hypothesis, exact diff/commit, profile,
metric samples, quality packet, selection evidence and keep/reject/inconclusive
reason. Three unsuccessful candidates in one bottleneck trigger re-profiling
before further variants. If no supported win remains after that reassessment,
close the tranche with its measurements; do not manufacture progress. A hung
GPU, external ownership conflict, or broken oracle is a blocker to repair and
record, not permission to reset unrelated work or weaken correctness.

  *Diagnostic 2026-09-26 (iteration 59): the prefill expert projections, and the
  same open bar decision at 3.4x the stake.* The prefill bottleneck was not
  attention after all. Routing Gemma's MoE prefill through the repo's existing
  compact WMMA owners - the ones the Qwen35 MoE production path already uses -
  moved a 1024-token prefill from 8.523 s to 2.544 s. Three things were needed:
  a fused-stride argument on the dual owner (a fused `gate_up` tensor stores both
  halves in one allocation, so the expert stride is `out_features_a +
  out_features_b` while each half still indexes from its own origin), and two new
  compensated owner variants described below.

  Measured on the W7900 lane, same artifact, greedy, bf16 KV, 1024 prompt
  tokens, 60 layer calls, via `scripts/gemma4_prefill_census.py`:

  | expert route | prefill s | tok/s | vs baseline | teacher-forced gate |
  | --- | ---: | ---: | ---: | --- |
  | selected GEMV (incumbent) | 8.523 | 120.1 | - | kl_max 0.000000, pass |
  | grouped rowbatch8 (default) | 6.982 | 146.7 | +22% | kl_max 0.000000, pass |
  | compensated WMMA | 2.544 | 402.5 | +235% | kl_max 0.060867, **fail** |
  | plain WMMA | 2.236 | 458.0 | +281% | kl_max 0.182137, **fail** |

  - **The grouped owner is exact, and that is what ships.** The Q5_1 down
    projection is 17.8% of the incumbent prefill (1238 ms over 58 calls) and the
    grouped rowbatch8 owner serves it in the same time as the selected GEMV
    *bit-for-bit*: the gate reports kl_max 0.000000, not merely below the bar.
    Its gate_up counterpart is unchanged because no grouped owner is registered
    for raw Q4_K, so `auto` keeps the selected GEMV there. This arm is the
    default and is worth +22% prefill at zero numerical cost.
  - **The WMMA breach is fp16 weight rounding, and compensating it is cheap.**
    The plain owners convert each dequantised weight straight to fp16 (~2^-11
    relative). A compensated form carries `hi = fp16(w)` plus
    `lo = fp16(w - hi)` and issues a second WMMA per k-tile, which represents the
    weight to ~2^-22. That moved kl_max from 0.182137 to 0.060867 for 14% more
    time (2.236 -> 2.544 s). The activation operand needs no compensation: bf16
    is a subset of fp16, so the conversion is exact.
  - **Both projections contribute, so no exact subset exists.** Pinning only the
    fused gate+up to WMMA gives kl_max 0.174212; pinning only the down projection
    gives 0.162814; both plain gives 0.182137. Each alone is nearly as bad as the
    pair.
  *Diagnostic 2026-09-26 (iteration 60): the fused `gate_up` joins the default
  route, bit-for-bit.* Iteration 59's grouped owner served only the down
  projection; the fused Q4_K `gate_up` stayed on the selected GEMV and was the
  largest single item in the default prefill (3951 ms of 6982 ms, 56.7%, 68.1 ms
  per call) because the selected dual GEMV runs one CTA per (row, output column)
  and re-reads every weight row once per compact row. The repository already had
  a grouped rowbatch8 *dual* owner for Q4_K, with a test pinning it bit-for-bit
  to the strict selected dual, but Gemma could not reach it for two reasons: the
  probe used the single-output grouped variant name, and the dual owner takes two
  weight tensors and two output buffers while Gemma's `gate_up` is one allocation
  holding both halves.

  Threading `output_row_stride` and `expert_stride_rows` through the grouped dual
  owner (both defaulting to zero, which is the two-tensor convention, so every
  existing variant's arithmetic and layout is untouched) made the fused form
  reachable. A fused `gate_up` passes the fused width for both strides, offsets
  the up half's output pointer one gate width into the fused row, and offsets the
  up half's weight pointer one half into the fused allocation - the last of which
  is the convention `gemma4_project_experts_wmma_dual` already used and is
  required, because the owner strides experts by the fused width while indexing
  each side from its own origin.

  | default-route family | before ms | after ms |
  | --- | ---: | ---: |
  | fused `gate_up` | 3951 (selected GEMV) | 1452 (grouped dual) |
  | Q5_1 down (grouped) | 1238 | 1101 |
  | attention prefill | 917 | 901 |
  | dense Q8_0 | 570 | 575 |
  | layer-29 MoE | 205 | 201 |
  | **layer total** | **6982** | **4242** |

  Prefill 6.982 -> 4.242 s on the W7900 lane (146.7 -> 241.4 tok/s) and 154.0 ->
  253.0 tok/s on the RX 7900 XTX lane; first-token latency 6.66 -> 4.05 s, decode
  flat at 43.8357. **The route is bit-identical, not approximate**: the gate
  reports `kl_max` of exactly 0.000000 over 1023 rows with top-1 rate 1.0 and
  zero flips, so the prefill win carries no numerical cost and owes no promotion
  gate. The grouped dual owner measures 25.0 ms per call against the selected
  dual's 68.1 (2.7x). Cumulative for the campaign's prefill column, 128.5 ->
  253.0 tok/s (+97%).

  This does not touch iteration 59's open decision. The WMMA arm is still the
  faster route (402.5 tok/s against 253.0) and still breaches the absolute
  `kl_max` bar on 1 of 1023 rows; `auto` now gets a larger exact share of the
  gap, which reduces what that decision is worth in absolute terms but does not
  resolve it.

  *Diagnostic 2026-09-26 (iteration 61): the grouped owner was input-bound, and
  the loop nest was the reason.* Iteration 60 left the fused gate+up owner at
  22.2 ms per call. A scaling probe of that owner on the real artifact settled
  what it was bound by: device time is linear in compact rows (1.90x / 3.64x /
  7.09x / 15.86x for 2x / 4x / 8x / 16x the rows), which is the signature of a
  kernel that re-walks the input once per output column. The arithmetic agrees -
  `out_features * rows * in_features * 2` bytes is 65 GB for one call against
  2.8 GB of weight traffic, about 1.5 TB/s of L2 reads and roughly one 2-byte
  input load per FMA, at 5% of fp32 peak.

  The per-output association is a fixed 128-thread tree over each thread's
  ascending column partials, and `ROW_BATCH` does not appear in it, so the loop
  nest could be swapped to load the input row batch once per k-block and reuse it
  across `OUT_BATCH` output columns without changing any output's summation
  order. That is a bit-identical change, and it measures as one.

  | shape | ms per call | VGPRs | waves/SIMD |
  | --- | ---: | ---: | ---: |
  | row-batch `<8, 1>` | 44.3 | 66 | 8 |
  | amortized `<8, 4>` | **15.8** | 173 | 8 |
  | amortized `<4, 8>` | 21.8 | 230 | 6 |

  `<4, 8>` halves the input traffic again but doubles the weight passes and drops
  occupancy, so it loses. Register pressure is what stops the shape going wider:
  the accumulator array is `2 * OUT_BATCH * ROW_BATCH` f32.

  | default-route family | iteration 60 ms | iteration 61 ms |
  | --- | ---: | ---: |
  | fused `gate_up` (grouped dual) | 1452 | 471 |
  | Q5_1 down (grouped) | 1101 | 1130 |
  | attention prefill | 901 | 905 |
  | dense Q8_0 | 575 | 552 |
  | **layer total** | **4242** | **3348** |

  Prefill 4.242 -> 3.360 s on the W7900 lane (241.4 -> 304.8 tok/s) and 253.0 ->
  319.0 tok/s on the RX 7900 XTX lane; first-token latency 4.05 -> 3.21 s, decode
  flat at 43.9149. The gate reports `kl_max` of exactly 0.000000 over 1023 rows
  with top-1 rate 1.0, so no promotion gate applies. Cumulative for the
  campaign's prefill column, 128.5 -> 319.0 tok/s (+148%).

  The same probe now points at the Q5_1 down owner (1130 ms, 33.7%, the largest
  single item): it has the identical nest, with the differences noted in the
  worklog entry. The WMMA arm's 402.5 tok/s is no longer ahead of the exact route
  and its open `kl_max` decision is correspondingly worth less.

  *Diagnostic 2026-09-26 (iteration 62): the down projection is metadata-bound,
  not input-bound.* Iteration 61's swap applied to the Q5_1 down owner (1130 ms,
  33.7%, the largest item left) but bought only 51.3 -> 49.8 ms on its own. The
  owner is bound by `dequant_q5_1` re-reading each weight block's `d`, `m` and
  `qh` words from global for every (thread, column) pair: four loads per eight
  FMAs, with all 32 lanes that share a block re-reading the same header.
  Preloading those words into shared memory once per block is the lever, and it
  only pays with the amortized nest because the slab then serves four output
  columns. Both together:

  | default-route family | iteration 61 ms | iteration 62 ms |
  | --- | ---: | ---: |
  | attention prefill | 905 | 923 |
  | Q5_1 down (grouped) | 1130 | 890 |
  | dense Q8_0 | 552 | 560 |
  | fused `gate_up` (grouped dual) | 471 | 469 |
  | **layer total** | **3348** | **3130** |

  Prefill 3.360 -> 3.130 s on the W7900 lane (304.8 -> 327.0 tok/s) and 319.0 ->
  341.0 tok/s on the RX 7900 XTX lane; first-token latency 3.21 -> 3.00 s, decode
  flat at 43.8715. `kl_max` is exactly 0.000000 over 1023 rows, and a direct A/B
  of the two owners shows 0 of 23,068,672 output elements differing. Cumulative
  for the campaign's prefill column, 128.5 -> 341.0 tok/s (+165%).

  Two process notes worth keeping. Shape selection needs interleaved medians:
  back-to-back passes of the *same* configuration drifted 45% on this machine, so
  the first, non-interleaved sweep concluded a 1.42x win that did not exist and
  the interleaved one found the real 1.32x. And the census caught a probe-order
  bug the isolated A/B could not: the Gemma probe listed the row-batch variant
  first, so the amortized owner resolved but was never reached, which showed up
  as "1.29x in isolation, no change in situ" until the list order was fixed.

  *Diagnostic 2026-09-26 (iteration 63): attention prefill, and why it is not a
  loop swap.* Attention prefill is the largest item left (923 ms of 3130 ms,
  29.5%, 15.4 ms per layer) at roughly 190 GFLOPS. It is not structurally the
  same problem as the MoE owners, so the amortization that paid there does not
  apply directly.

  `gemma4_attention_prefill_kernel` runs one CTA per (token, head) -- 11264 CTAs
  at 1024 tokens and 11 heads -- and pass 1 walks all `keys` one at a time, each
  key costing a 256-thread `gemma4_attn_block_sum`: a store, a barrier, eight
  barrier-separated tree rounds, and one more barrier after the call, so about
  ten barriers per key and about 10^4 barriers per CTA. Each thread issues one
  useful FMA per key against roughly thirty instructions of reduction and loop
  overhead, which is what 190 GFLOPS looks like.

  Two bit-identical fixes are available and neither is a nest swap:

  - Batch the key reductions. Compute a tile of `KEY_TILE` key partials into
    registers, store them into a `KEY_TILE x blockDim` slab, and run the tree
    for the whole tile at once. The per-key tree keeps its exact 256 leaves and
    pairing, so the published bits do not move, and the barrier count falls by
    `KEY_TILE`. The blocker is shared memory, not arithmetic: `logits_s` is
    `keys` floats, which is 62 KB at the 15616-key geometry
    `tests/test_gpu_gemma4_attention_geometry.py` exercises at `head_dim` 512, so
    the extra slab has to be sized against the geometry rather than fixed, and
    `gemma4_attention_shared_bytes` has to agree with the launcher.
  - Replace the last five tree rounds (strides 16, 8, 4, 2, 1) with
    `__shfl_down` over the lanes that the LDS tree would have paired. Same
    pairing, same order, so also bit-identical, and it removes half the barriers
    in pass 1 without any shared-memory cost. The first three rounds (strides
    128, 64, 32) cross warps and have to stay in LDS.

  Neither is a candidate that can be judged by the isolated A/B used for the
  MoE owners: attention has no bit-equality test against a slower owner, so the
  evidence has to be the parity suite plus the teacher-forced gate. Iteration 64
  found that both fixes were already written, in the decode family, and that the
  missing piece was the launcher's routing rather than the kernel.

  **Iteration 87: two attempts at the MoE dp4a gate_up, both wrong, both caught
  by the gate, both reverted.** Iteration 86 established that a grouped dp4a
  owner exists for the MoE gate_up and that the Gemma path has most of the
  plumbing. Wiring it produced two failures, and the way they failed is the
  useful part.

  **Attempt 1: non-finite logits.** I built a *separate* tile plan with
  ``qwen35_moe_mmq32_tile_map``, reasoning that a 32-row owner needs a 32-row
  plan. The reference does not do that: ``qwen4_exp_runner`` builds the plan with
  ``qwen35_moe_wmma_tile_map`` and uses ``tile_rows = 32`` only to *validate*
  ``wmma_total_rows`` against the tile capacity. The 32 in the owner's name is a
  row count it checks, not a different plan. The wrong expert/tile mapping
  produced non-finite logits, and the census in the same run reported
  **801.75 tok/s, +22.6%** -- a fast wrong answer, measured and believed for the
  few minutes it took the gate to run.

  **Attempt 2: plausible but wrong.** Switching to the existing 16-row plan (and
  dropping the extra buffers, since the plan is shared) still fails the gate:
  ``kl_max`` 29.29 against a 0.05 bar, ``kl_mean`` 0.764 against 0.001. This is
  worse than attempt 1 in the way that matters -- the numbers are finite and
  plausible, so nothing but the gate would have flagged them.

  **What I did not do, and should have.** I read the wrapper's *signature* and
  the reference's *call site*, and inferred the contract from them. Neither
  states what ``compact_to_source`` means, what output layout the owner writes,
  or what expert stride it assumes. The reference shows what qwen4 passes, not
  why it is correct there, and there is a concrete difference I noticed and
  dismissed: qwen4 passes **separate allocations** for gate and up
  (``weights["expert_gate"]`` and ``weights["expert_up"]``), while Gemma's fused
  expert tensor gives one ``base_ptr`` plus a half offset. Assuming those are
  interchangeable is exactly the kind of inference that needs the kernel's own
  documentation, not a call site.

  **The transferable point.** Both failures were caught by the gate and neither
  by the census. The census measures speed; it cannot see a wrong answer, and
  the +22.6% it reported was attached to a kernel writing infinities. Any future
  attempt at this route starts by reading the owner's contract in the kernel
  source, and treats the census as a speed instrument only after the gate is
  green.

  The MoE dp4a route is unproven, not disproven: the owner's shape constraints
  fit, the plumbing is nearly all present, and the prize is real. Reverted;
  the grouped fp32 owners stay on the MoE line.

  **Iteration 88: the MoE dp4a route is closed after four attempts, all wrong,
  all reverted -- and the census reported a win on every one of them.** Iteration
  87 recorded two attempts. Two more followed, each after reading the reference
  more carefully, and each failing the same way.

  **Attempt 3: separate expert-start arrays.** Reading the launcher showed it
  takes ``expert_start_compact`` *and* ``expert_start_mmq32`` as distinct
  arguments, and ``qwen4_exp_runner`` passes ``group_expert_start`` (compact) for
  the first and ``group_wmma_expert_start`` (plan) for the second. My earlier
  attempts had passed one array for both, in opposite directions: attempt 1 used
  the padded array as compact, attempt 2 used the compact array as padded. Fixed
  to match the reference exactly. Result: non-finite.

  **Attempt 4: padded-capacity buffers.** Reading the reference's *allocations*
  -- which I had never done, having only read its call -- showed its identity map
  and Q8_1 workspace are sized by ``compact_capacity``, not by the unpadded row
  count. Since the tile plan pads rows to a 32-row boundary, sizing by ``lanes``
  lets the owner index past the end for padding rows, and an uninitialized int64
  row index is a wild gather. Sized both by a padded-capacity bound and filled
  the identity over the padded range. Result: non-finite.

  **The pattern, which is the real finding.** Each round I found a genuine
  discrepancy, fixed it, and the failure mode never changed. Four rounds of that
  is not a sequence of near-misses; it means my model of the contract is wrong in
  a way I have not identified, and reading the reference one detail at a time is
  narrowing it too slowly to converge.

  **And the instrument was actively misleading.** Every broken attempt produced a
  large apparent speedup on the census: 801.75 tok/s (+22.6%) on attempt 1,
  747.49 (+24.8%) on attempt 3, 744.60 (+24.3%) on attempt 4, against a 598.9
  baseline. A broken path is *faster*, because it is not doing the work
  correctly, so the census did not merely fail to catch four wrong kernels -- it
  recommended all four. An instrument that rewards the failure it cannot see is
  worse than no instrument.

  **What to do instead, next time.** The contract has at least five interacting
  arrays (packed activations, compact-to-source map, two expert-start arrays, the
  tile-expert map, two weight pointers, and an output layout). Iterating that
  through a ten-minute end-to-end gate costs forty minutes to learn one bit per
  round. A kernel-level numerical oracle -- run the MMQ owner against the fp32
  grouped owner on random inputs and compare, ~30 seconds per cycle -- would have
  localized the error to weights, plan, or output layout in the first round and
  made the other three unnecessary. Build the direct oracle before touching the
  end-to-end path.

  The route is unproven, not disproven: the owner's shape constraints fit, the
  plumbing is nearly all present, and the apparent prize (~25%) is large enough
  to be worth a properly instrumented attempt. It is not worth a fifth
  inference. Reverted; ``gemma4_experts.py`` is byte-identical to HEAD.

  **Iteration 89: the two-plane Q8 MMQ passes the gate; its speed is a wash, and
  the apparent regression was my own measurement error.** Iteration 88 closed the
  MoE dp4a route and named the dense Q8_0 linears as the next target, on the
  campaign's own roadmap ("attention and the dense Q8_0 linears are the larger
  remaining share and neither has been routed through a WMMA owner yet") plus the
  observation that ``dense:gguf_q8_0`` runs at roughly a third of the memory
  roofline. Reading the policy resolver showed the dense MMQ path is *already*
  engaged by default as the "retained three-plane d4x3 exact chain", and that a
  two-plane variant exists behind ``HIPENGINE_QWEN4_EXP_Q8_MMQ_PLANES=2``,
  documented as "+1.4x on the dense legs, quantization-level drift pending
  envelope qualification".

  **The gate result, which is the durable finding: the two-plane variant passes
  with large margin.** ``kl_max`` 0.00642 against a 0.05 bar (7.8x), ``kl_mean``
  1.62e-05 against 1e-3 (62x), zero failed checks. The "drift pending envelope
  qualification" caveat attached to that env var since it was introduced is now
  discharged: the faster dense-MMQ variant is numerically qualified against the
  same teacher-forced reference every other arm is judged by. The default is
  unchanged at three planes, because a variant with no measured speed benefit is
  not worth an arithmetic change -- but the qualification is no longer the reason
  to hold it back.

  **The speed is a wash, and I first read it as a 1.1% regression.** The census
  reported 646.79 against a 653.68 baseline. Every family I had printed in the
  run was *faster*, which is impossible, so I went to the JSON artifact for the
  full split and then to the baseline's own JSON -- and found the real
  comparison, because the 1553.7 ms ``layer_total`` I had been reasoning from was
  the *broken* MoE run's, not the clean baseline's. Against the correct baseline
  every family moves by the same small amount: dense Q8_0 +4.9, gate_up +4.5,
  attention +4.9, down +2.1, ``moe_misc`` exactly 0.0. Attention prefill and the
  MoE bookkeeping kernels do not touch the Q8 MMQ path at all, so a uniform shift
  across them is not a kernel effect. It is host contention: I ran this census in
  parallel with a teacher-forced gate on the other GPU, and the gate is
  CPU-heavy.

  **Two process rules fall out, and both cost real time here.** First, capture
  the census whole: I piped it through ``grep`` at capture time, which is why the
  first pass had no family split to explain its own result -- the ``--json``
  artifact had it all along. Second, do not run the census concurrently with
  another GPU job. The contention effect here is about 1%, the same size as the
  keep/kill threshold, and it flipped the sign of this comparison. A single-lane
  census is the only one whose numbers can decide a keep.

  Evidence: ``/mnt/nvme1/lhl/gemma4-captures/census-mmq-xtx.json`` (baseline),
  ``census-mmq2plane-xtx.json`` (two-plane), ``/tmp/gate-mmq2plane.json``.

  **Iteration 90: the kernel-level oracle found the MoE dp4a root cause in five
  30-second runs, after four ten-minute gate cycles could not.** Iteration 88
  closed the route and named the missing instrument. This iteration built it:
  ``scripts/gemma4_moe_owner_oracle.py`` wraps the production grouped-dual owner
  to capture its real arguments on a live forward, then replays those exact
  arguments through the known-good fp32 owner and through the Q4_K ds4 MMQ owner
  and compares. It paid for itself immediately.

  **The four failed attempts varied arguments that do not matter.** All eight
  combinations of identity fill (unpadded vs padded), expert-start array (compact
  vs plan), and plan builder (``qwen35_moe_mmq32_tile_map`` vs
  ``qwen35_moe_wmma_tile_map``) produce the *identical* wrong answer --
  ``max_abs`` 3.91504 to five digits, on every one of them. So
  ``compact_to_source``, the expert-start arrays, and the plan choice are all
  irrelevant to the failure. Every hypothesis those four attempts were built on
  was a variable that changes nothing.

  **The error is a uniform scale, not an indexing error.** With a valid plan the
  best-fit scale is 0.062 -- the MMQ output is ~16x too large -- and *zero* rows
  land within 1e-2 of the reference, while a half-swap test is no better than the
  direct comparison. No row is right and every row is wrong by the same factor:
  that is a wrong *input format*, not a wrong index.

  **The oracle also reproduced the end-to-end failure mode exactly.** An early
  version of it called the MMQ owner before ``build_plan``, leaving the
  tile-expert map uninitialized; that run reported non-finite output, and a
  repeat of the *same* call reported finite output instead -- non-determinism
  from reading uninitialized device memory. That is the attempt-2 failure: the
  plan buffers are only built when ``gate_up_wmma or down_wmma`` is set, which is
  false by default, so attempt 2 fed the owner an uninitialized tile map. A
  silently-uninitialized plan is indistinguishable from a wrong one at the gate,
  and it cost a full cycle to learn.

  **The root cause is a misleading import alias.** ``qwen4_exp_runner`` line 123
  reads::

      gguf_q8_1_mmq_ds4_pack_bf16_d4x3 as gguf_q8_1_mmq_ds4_pack_bf16,

  so the call site I copied -- ``gguf_q8_1_mmq_ds4_pack_bf16(...)`` -- does not
  run the generic single-plane pack. It runs the three-plane one. The pack I
  called is documented as "Pack BF16 activations as primary DS4 plus two residual
  DS4 planes", and the owner's registered variants are all ``d4x3``. Feeding the
  owner a single-plane pack drops both residual planes, which is exactly the
  uniform-scale signature the oracle measured, and the three-plane format also
  needs a larger workspace than the 144-bytes-per-128-block I allocated -- which
  is what faulted the GPU when the three-plane pack was tried.

  **Cost accounting, because this is the whole argument for the instrument.**
  Four end-to-end attempts: roughly 40 minutes of gate cycles, each learning one
  bit, all four bits irrelevant. Five oracle runs: roughly three minutes total,
  producing the failure class, the eliminated variables, the reproduced
  end-to-end defect, and the root cause. The next step is mechanical rather than
  inferential -- read ``struct block_q8_1_mmq_ds4`` for the three-plane stride,
  size the workspace, re-run the oracle expecting ``scale_k`` ~ 1.0 and
  ``rows_ok`` ~ 100%, then wire it and gate it.

  Evidence: ``scripts/gemma4_moe_owner_oracle.py``; captures
  ``rows=4096 experts=128 in=2816 half=704 fused=1408``.

  **Iteration 91: the MoE dp4a root cause is a missing weight_pack for Q4_K --
  the route is a feature, not a wiring fix.** Iteration 90 built the oracle and
  named the pack format as the likely cause. Testing that properly eliminated it
  and every other wiring hypothesis, and landed on the weights.

  **The evidence chain, in the order the oracle produced it.**

  1. All eight combinations of identity fill (unpadded/padded) x expert-start
     array (compact/plan) x plan builder (mmq32/wmma) give the *identical* wrong
     answer, ``max_abs`` 3.91504 to five digits. The mapping is not the fault.
  2. ``got_absmax`` is 2.24 (RAW) and 1.96 (X8) against the reference's 2.125:
     the magnitudes are right. The apparent "16x scale error" from the
     least-squares fit was an artifact -- a scale fitted to uncorrelated data is
     meaningless, and I read it as a finding for one iteration. What actually
     holds is: right magnitudes, uncorrelated values.
  3. No row permutation: for a sample of output rows the nearest reference row is
     1.85 away, 0/8 matched. The rows are not shuffled.
  4. RAW vs X8 with a *valid* plan differ (3.915 vs 3.920), so the layout
     template does matter -- but neither is right. The earlier "RAW equals X8"
     reading was invalid: that comparison ran before ``build_plan``, so both
     sides were reading an uninitialized tile map.
  5. Halves as-is vs swapped: all four combinations ~3.9 with 0% rows correct.
     The half order is not the fault.

  Right magnitudes, uncorrelated values, insensitive to mapping, ordering,
  template, and half order. That leaves the weights themselves.

  **The root cause.** ``hipengine/runtime/gguf_q8_mmq_sidecars.py`` resolves a
  ``weight_pack`` kernel at variant ``mmq_kmajor76`` per quant key: the MMQ
  owners consume a K-major-packed *sidecar* tensor, not the raw GGUF blocks.
  That variant is registered for exactly one quant -- ``gguf_q8_0`` -- and the
  sidecar builder is imported by exactly one runner, ``qwen4_exp_runner``.
  Gemma's runtime never builds sidecars, and no Q4_K ``mmq_kmajor76`` pack exists
  to build. So every attempt fed raw Q4_K blocks to an owner expecting a packed
  layout, which produces plausible-magnitude, uncorrelated output -- precisely
  what was measured. The same registry explains why the *dense* Q8_0 MMQ path
  works and is default-on: its pack exists.

  **What that means for the route.** The MoE dp4a prize (~25% of prefill, the
  MoE line sitting at 71% of prefill against ~6% of its memory floor) requires
  writing a ``weight_pack``/``mmq_kmajor76`` kernel for ``gguf_q4_k`` and wiring
  sidecar construction into Gemma's runtime -- a kernel-development task with a
  load-time repack and its memory cost, not a dispatch fix. That is a materially
  different decision from the one the earlier attempts implied, and it is the
  decision the lead now has to make.

  **Also real, and cheap to fix separately:** the oracle reproduced attempt 2's
  end-to-end failure exactly by calling the owner before ``build_plan``. The
  plan buffers are only built when ``gate_up_wmma or down_wmma`` is set, which is
  false by default, so any future attempt that reaches for the plan without
  building it gets an uninitialized tile map -- non-deterministic output that
  looks like a correctness bug rather than a missing initialization.

  Evidence: ``scripts/gemma4_moe_owner_oracle.py`` (updated with the layout,
  half-order, and row-permutation probes and with the plan built before the
  probes, which is what made them valid).

  **Iteration 92: the real root cause -- the MMQ dual owner wants two
  independently-addressed weight tensors, and Gemma's is fused.** Iteration 91
  concluded from the ``weight_pack`` registry that the owner needs a
  ``mmq_kmajor76`` sidecar. That was wrong, and reading the kernel showed why: the
  wrapper I call is templated ``Q4_K_LAYOUT_RAW``, and the launcher branches
  ``if constexpr (WEIGHT_LAYOUT == Q4_K_LAYOUT_X8)`` to a different kernel. A RAW
  owner reads raw GGUF blocks. The registry inference was the same mistake the
  four failed attempts made -- reasoning from a name instead of the code.

  **What the body actually does** (``gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_body``,
  weight addressing)::

      const int64_t weight_row_bytes = blocks_per_weight_row * Q4_K_BLOCK_BYTES;
      const int64_t expert_bytes = local_out_features * weight_row_bytes;
      const uint8_t* expert_base = qweight + expert_id * expert_bytes;

  ``local_out_features`` is the *per-half* width -- 704, not the fused 1408. So
  ``qweight_a`` and ``qweight_b`` are two independently-addressed tensors, each
  with an expert stride of ``in_features * out_features_half``. Gemma's fused
  expert tensor is ``[expert][gate 704 | up 704]`` with an expert stride of
  ``in_features * 1408``. Every expert except 0 therefore reads from the wrong
  place, and ``qweight_b`` reads offset 0 for expert 0 -- the gate's data -- so no
  output row can be fully correct even for the first expert.

  **That accounts for most of the measurements.** Right magnitudes, because the
  bytes are real weight data, just the wrong expert's. Uncorrelated values,
  because a different expert's weights multiply the same activations. Insensitive
  to the tile map, identity fill, expert-start arrays, plan builder, layout
  template, and half order, because none of them change the stride that is wrong.

  **One measurement it does not yet account for, stated rather than smoothed
  over.** For expert 0 both tensors resolve to their true starts -- ``qweight_a``
  at 0 and ``qweight_b`` at ``base_ptr + half_bytes`` -- so *if* expert 0 owned
  any output rows, those rows would be correct and ``rows_ok`` would be at least
  0.8% (32 of 4096 rows). It measured exactly 0.0%, and the row-match probe found
  row 0 wrong. The likely resolution is that output rows are in *tile* order, so
  row 0 belongs to whatever expert the first tile holds, which need not be expert
  0 -- but that is an assumption, not a measurement. The validation below should
  also print ``tile_expert[0]`` so it is settled rather than assumed. If the first
  tile *is* expert 0 and its rows are still wrong, then a second difference exists
  and the stride analysis is incomplete.

  **The fix is a de-interleave, not a repacking kernel.** The owner needs
  ``[all experts][gate]`` and ``[all experts][up]`` as separate tensors; Gemma
  stores ``[expert][gate | up]``. Converting one to the other is a pure
  expert-level copy -- no layout transform, no kmajor76, nothing like the
  kernel-development task iteration 91 described. The gate_up tensor is about
  285 MB at these shapes, so the cost is a load-time copy and that much extra
  device memory.

  **And the diagnosis is directly checkable before any production change:** the
  oracle can build the two half tensors by copying each expert's halves into two
  expert-major buffers, re-run the same owner, and expect ``rows_ok`` to go from
  0.0% to ~100%. If it does not, the stride analysis is wrong and the route
  closes for good. That check is the next action.

  **Iteration 93: the two-tensor diagnosis is validated in kind -- splitting the
  fused weights drops the error to quantization scale, with a localized residual
  left.** Iteration 33 read the owner's body and concluded it wants two
  independently-addressed weight tensors. This iteration tested that by building
  them: for each expert, copy the gate half and the up half out of Gemma's fused
  tensor into two expert-major buffers (half = 1,115,136 bytes, expert =
  2,230,272 bytes at these shapes) and re-run the same owner.

  **The result, and it is progress rather than success.** Direct error falls from
  3.91504 to 1.97070, with both halves improving equally (gate 1.92334, up
  1.97070). On row 0 the best alignment is row-major (confirming the output is
  not tile-major, as the 32-wide column-tile grid had suggested), and the
  per-column-tile error profile is the informative part: **median 0.0415** against
  a reference absmax of 2.125 -- about 2%, which is quantization scale for Q4_K
  weights against Q8_1 activations -- with a few outlier tiles reaching 0.4658.
  So the split is the right structural fix and most of the output is now correct.
  ``rows_ok`` stays 0.0% only because the strict per-row 1e-2 bar is below the
  residual outliers.

  **The earlier 0.0%-including-expert-0 puzzle is settled by measurement.**
  ``tile_expert[0] = 12``: output row 0 belongs to expert 12, not expert 0, so no
  output row belongs to the one expert whose pointers happened to resolve
  correctly under the fused layout. The assumption I flagged in iteration 33's
  correction is now a measurement, and the stride diagnosis explains the original
  0.0% without exception.

  **What is left, and the probe that settles it.** A residual of ~2% median with
  sparse 20% outliers is not the weight stride -- that is fixed -- and it is
  larger than quantization noise should be, so a smaller addressing difference
  remains. ``block_q8_1_mmq_ds4`` is four (scale, sum) half2 pairs followed by 128
  int8 quants, i.e. four 32-feature sub-blocks, and the column tile is also 32
  wide. If the residual is in the *activation* packing, the per-tile error pattern
  will be the same across rows; if it is in the *weight* side, the pattern will
  vary per row. Comparing the per-tile error profile across several rows
  distinguishes the two in one run. That is the next action, and the fix is
  expected to be a variant choice or an offset in the pack rather than a new
  kernel.

  Evidence: ``scripts/gemma4_moe_owner_oracle.py`` (now builds the split tensors
  and prints ``tile_expert[0]``, the row-major/tile-major comparison, and the
  per-tile error profile).

  **Iteration 95: the residual is row-dependent, so it is on the activation
  side -- and my stated hypothesis had the implication backwards.** Iteration 34
  left a ~2% median residual with sparse 20% outliers and asked whether the
  per-tile error profile is row-independent (activation packing) or row-dependent
  (weight side). That framing was wrong, and the measurement says so plainly.

  The implication runs the other way. A *weight-side* error -- wrong weight rows,
  columns, or values -- hits the same output columns for every row, so the
  per-tile error profile would be identical across rows: row-independent. An
  *activation-side* error mis-packs one row's activations, so the error differs
  from row to row: row-dependent.

  **Measured, split weights:** ``mean`` 0.1705, ``across_row_std`` 0.3001, ratio
  1.76, and the worst tile per row is ``[20, 11, 18, 6, 21, 5, 11, 10]`` -- eight
  different indices. The error varies per row by more than its own mean, so the
  residual is on the activation side. For contrast the fused case measures ratio
  0.02: a single uniform profile, which is what a stride error looks like.

  **This narrows the remaining work to the activation pack.** The weights are
  structurally right (the two-tensor split took the error from 3.9 to quantization
  scale), and the leftover is in how the row's activations are quantized and
  laid out for the owner -- consistent with ``block_q8_1_mmq_ds4`` being four
  32-feature (scale, sum) sub-blocks, which is also the column-tile width. The
  candidates are the pack variant (the generic single-plane pack versus the d4x3
  and f32 forms) and the sub-block or row offset within it. Worth noting for
  calibration: both paths read the *same* raw Q4_K weights, so the only intended
  difference is bf16 activations becoming Q8_1, which should cost a few tenths of
  a percent -- not the 20% outliers seen here. Those outliers are a real defect,
  not quantization noise.

  Evidence: ``scripts/gemma4_moe_owner_oracle.py`` (per-tile profile across eight
  rows, with the across-row variance and the worst-tile index per row).

  **Iteration 96: the MoE route is close but not passing, and this is the point
  to decide whether to keep spending on it.** Two more probes narrowed the
  residual, and neither produced a gate-passing path.

  **The activation pack variant is eliminated.** The generic single-plane pack and
  the three-plane ``d4x3`` pack produce *bit-identical* output with the split
  weights: direct 1.97070, median row error 1.84277, same to five digits. As with
  the earlier RAW/X8 comparison, identical output from two different inputs means
  the difference is not reaching the result. The pack form is not the residual.

  **It is not a tile-to-expert pairing mismatch either.** Zero of 128 32-row tiles
  land within 1e-2 of the reference. A pairing disagreement would leave the tiles
  whose assignment happens to coincide correct, so 0/128 rules it out.

  **What the residual actually looks like.** Row 0's per-column-tile error has
  median 0.0415 -- about 2% of the reference absmax, quantization scale -- while
  that same row's maximum is 1.84, and the median *row* maximum across the tensor
  is 1.84277. So within each row most columns are already correct and a small
  number are badly wrong, with the bad set varying from row to row. That is not
  the signature of an addressing error, which would be uniform; it is the
  signature of a small number of wrong *activation* contributions per row. The
  strongest remaining candidate is the ``block_q8_1_mmq_ds4`` sub-block handling
  -- four 32-feature (scale, sum) pairs -- for particular positions within a row.

  **Status and the decision.** What is established: the owner needs two
  independently-addressed weight tensors, and splitting the fused tensor takes the
  error from 3.915 to 1.971 and puts the bulk of the output at quantization scale.
  What is not: a path that passes the gate. ``rows_ok`` is 0.0% and no tile is
  clean, so the split alone is necessary but not sufficient.

  This route has now consumed eight iterations (iterations 28 through 35 plus the
  oracle work) without moving the production metric, which remains 653.68 against
  llama.cpp's 3910. The prize is real and the largest remaining one -- the MoE line
  is 71% of prefill at roughly 6% of its memory floor -- but the remaining defect
  is a small numerical one in activation packing rather than the structural
  problem the earlier iterations were chasing, and further progress needs either
  the specific sub-block fix or a working qwen4 Q4_K reference to diff against.
  Whether to keep spending on it or redirect to another prefill target is the
  lead's call, and this entry records it as a call rather than deciding it.

  Evidence: ``scripts/gemma4_moe_owner_oracle.py`` (pack-variant comparison,
  per-tile good/bad counts, expert ids of good tiles).

  **Iteration 97: weight traffic is not the bottleneck anywhere -- every family
  runs 20-100x above its own weight ceiling.** Eight iterations went into routing
  the MoE gate_up through a dp4a owner on the premise that the MoE line was far
  above its memory floor. That premise was never measured. This iteration measured
  it, and it does not survive.

  ``scripts/gemma4_weight_bytes_census.py`` wraps the projection owners during a
  forward and records the weight tensor bytes each call touches, giving a
  per-family traffic ceiling (tensor bytes / 864 GB/s). Against the 653.68
  census's timings::

      family                        time      tensor bytes   ceiling   ratio
      moe_grouped_dual:gguf_q4_k    417.9 ms      16.6 GB      19.2 ms    22x
      moe_grouped:gguf_q5_1         435.5 ms      11.0 GB      12.8 ms    34x
      dense:gguf_q8_0               394.3 ms       3.5 GB       4.0 ms    98x
      whole prefill                1570.3 ms      32.9 GB      38.0 ms    41x

  **Two things follow.** First, a grouped or selected owner touches only the
  activated experts' rows, so its tensor size is a *ceiling* on its traffic, not
  its actual traffic -- dividing measured time by these numbers produced
  impossible bandwidths (32 TB/s), which is how the ceiling/floor confusion was
  caught. Second, and more important, even the *ceiling* is 20-100x below the
  measured time. No family is anywhere near memory-bound on weights.

  **The dense line is the cleanest case**, because each call reads its whole
  weight: 12.255 MB per call at a 961 us mean, so roughly 18 MB moved per call
  counting activations. That is about 19 GB/s, or 2% of device peak. These are
  memory-*inefficient* kernels, not memory-*bound* ones.

  **What this means for the campaign.** The MoE dp4a work, the weight_pack
  questions, the layout templates, and the dense MMQ two-plane variant all target
  weight traffic. If weight traffic is not the constraint, none of them can
  deliver the headroom they promise, and the eight iterations spent on the MoE
  route were aimed at the wrong bottleneck -- which is consistent with the route
  never producing a measurable win. The headroom is in kernel *efficiency*:
  occupancy, access pattern, tile shape, and the per-call overheads that the
  census currently buries in ``unattributed`` (33-70 ms, including the Q8_1
  activation packing that the MMQ path adds per call). That is a different
  optimization target, and it should be chosen deliberately rather than by
  continuing down the layout path.

  Caveat stated plainly: these are tensor sizes, not measured traffic. The
  read fraction needs a profiler (``rocprofv3``) or a routing histogram, not
  arithmetic. The dense number is the trustworthy one because its read fraction is
  1.0 by construction.

  Evidence: ``scripts/gemma4_weight_bytes_census.py``.

  **Iteration 97b: per-shape census -- the dense linears run at 1.5% of memory
  bandwidth, and I cannot yet say which owner is running them.**
  ``scripts/gemma4_prefill_shape_census.py`` reuses the prefill census's HIP-event
  timing but labels each call by shape and adds bytes, GB/s, and TF/s. The family
  view had been hiding the distribution, and the distribution is the story::

      shape                                              calls   tot ms   mean us   MB/call  GB/s  TF/s
      moe_grouped:gguf_q5_1 rows=4096 k=704  n=2816 e=128    58    473.2    8158.6   219.15  26.9   2.0
      moe_grouped_dual:q4_k rows=4096 k=2816 n=1408 e=128    58    451.3    7780.6   320.08  41.1   4.2
      dense:gguf_q8_0 r=512 k=2816 n=2112                   120    107.0     891.7    11.37  12.7   6.8
      dense:gguf_q8_0 r=512 k=2816 n=4096                    50     74.1    1482.4    19.33  13.0   8.0
      dense:gguf_q8_0 r=512 k=2816 n=2048                   100     73.7     736.9    11.11  15.1   8.0
      dense:gguf_q8_0 r=512 k=2112 n=2816                    60     63.6    1059.9    11.37  10.7   5.7
      dense:gguf_q8_0 r=512 k=4096 n=2816                    50     55.4    1108.5    19.33  17.4  10.7

  **Measured, and solid.** Every dense call is 512 rows. The dense line achieves
  10.7-17.4 GB/s -- about 1.5% of the 864 GB/s peak -- at 5.7-10.7 TF/s. The MoE
  families do better but are still low: 26.9-41.1 GB/s at 2.0-4.2 TF/s. The two
  MoE shapes alone are 924 ms of the 1570 ms layer total, 59%.

  A 512x2816x2048 projection moves 11.4 MB, which at peak bandwidth is 13 us. It
  takes 737 us: 57x off the memory roofline, and roughly 15x off an INT8 compute
  roofline. It is inefficient at both, so neither roofline is the binding
  constraint.

  **Not asserted, and this matters before any kernel conclusion.** The label is by
  *quant*, not by *owner*. ``_NATIVE_ROWTILE_CHUNK_MAX_ROWS = 512`` carries the
  comment "rows >= 512 is the bulk-prefill regime and stays on WMMA", so these
  512-row dense batches may be running the WMMA path rather than the Q8_0 MMQ
  path -- in which case the finding is that the WMMA dense path is slow and the
  MMQ path is not engaged for these shapes at all, which is a different problem
  with a different fix. Which owner serves these shapes has to be established
  first; the shape census deliberately does not guess.

  **Why this matters for the campaign's direction.** Every lever pursued so far --
  the MoE dp4a route, the weight_pack questions, the layout templates, the
  two-plane MMQ variant -- assumes weight traffic is the constraint. At 1.5% of
  bandwidth on the dense line and ~3-5% on the MoE line, it cannot be. Whatever
  the owner turns out to be, the headroom is in kernel efficiency, and the first
  question is which kernel is running.

  Evidence: ``scripts/gemma4_prefill_shape_census.py``.

  **Iteration 98: the two MoE kernels holding 59% of prefill now have names; the
  dense dispatch is still unidentified.** A probe wrapping
  ``hipengine.kernels.registry.resolve`` during one 1024-token prefill records 126
  resolutions total, and they name the MoE owners exactly::

      58  moe_linear|gguf_q4_k|selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out
      58  moe_linear|gguf_q5_1|selected_grouped_prefill_pair2_fold128_bf16_bf16_out

  Those are the 451 ms gate_up and the 473 ms down shapes -- 59% of layer time --
  so the largest target is now two named kernels rather than two families. The
  names also correct a labelling habit: the census calls them ``moe_grouped_*``
  because that is the wrapper it patches, but the registered variant is a
  ``selected_*`` hybrid.

  **The dense line does not resolve through this path at all.** Of 126
  resolutions, only two are ``linear|gguf_q8_0`` (both ``selected_gemv``, a decode
  variant); the 410 dense prefill calls resolve nothing. They use a cached
  dispatch, so this probe cannot name their owner, and the question from iteration
  97b -- MMQ or WMMA for the 512-row dense shapes -- remains open. It needs either
  the cached dispatch instrumented or the selection code read directly. Stating it
  as open is the point: the alternative is inferring an owner from a constant's
  comment, which is how the kmajor76 sidecar conclusion went wrong.

  **And the profiler route is closed.** The campaign already records that
  ``rocprofv3`` began hanging on this box, so kernel attribution here has to come
  from instrumentation. Worth remembering before proposing a profiling pass: the
  documented tool does not run.

  Evidence: ``/tmp/gemma4_variant_probe.py`` (probe, not committed -- the two
  variant names above are the finding).

  **Iteration 99: the gate_up MoE kernel is L2-bound on activation re-reads, and
  the lever is the output-column block. This revises iteration 97b.**
  ``gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel<8, 4, true, true, true,
  true>`` launches grid ``((out_features + 3) / 4, num_experts)`` = 352 x 128 =
  45,056 CTAs of 128 threads, so **each CTA owns one expert, four output columns,
  and all live rows**. The activation tensor is therefore re-read once per
  output-column group: ``out_features / 4`` = **352 times**.

  Traffic, gate_up at 4096 rows x 2816 K x 1408 N::

      activations   4096 x 2816 x 2 B = 23.0 MB, re-read 352x  = 8.10 GB
      weights       285 MB, read once (each column belongs to one CTA) = 0.29 GB
      total                                                     = 8.39 GB

  8.39 GB in the measured 7.8 ms is **~1.1 TB/s apparent** -- above the 864 GB/s
  HBM peak, which is only possible because the 23 MB activation tensor fits inside
  the 96 MB Infinity Cache. So the kernel is **L2-bandwidth-bound on activation
  re-reads**, at roughly a quarter of the L2 rate. It is not compute-bound
  (16.2 G MACs is ~530 us at an INT8 roofline, 15x below measured) and it is not
  weight-bound (285 MB is 330 us).

  **The lever is the output-column block, and it is a template parameter**::

      out-block   activation traffic   weights   total    predicted
          4 (now)          8.10 GB     0.29 GB  8.39 GB   7.8 ms (measured)
         16               2.02 GB     0.29 GB  2.31 GB   ~2.1 ms
         32               1.01 GB     0.29 GB  1.30 GB   ~1.2 ms

  A 16-wide block predicts ~2.1 ms against 7.8 ms measured, and the shape runs 58
  times per prefill, so the stake is roughly 325 ms of a 1570 ms layer total.

  **This corrects iteration 97b.** That entry concluded the kernels were
  "inefficient at both" rooflines and that weight traffic could not be the
  constraint. The first half was right and the second half was under-specified:
  the constraint is *activation* traffic, and the re-read factor comes from the
  grid geometry, not from the kernel's inner loop. Note also that weight traffic
  becomes material again once blocking is fixed -- at out-block 32 the weights are
  285 MB of 1.30 GB, 22% of the total. So the weight-layout work is not wasted; it
  is **downstream of** the blocking fix rather than an alternative to it. That is a
  cleaner ordering than either entry had.

  **Caveat.** The L2-residency argument is inferred from the apparent bandwidth
  exceeding HBM peak, not measured. It is consistent, but the direct check is the
  A/B itself: instantiate a wider out-block, confirm it is bit-identical, and read
  the time. If the wider block does not win, the re-read model is wrong and this
  entry should be corrected by a new one rather than quietly dropped.

  Evidence: ``gguf_q4_k_selected_prefill.hip:2680`` (launcher geometry);
  ``gguf_q4_k_selected_prefill.py:1593`` (registration).

  **Iteration 100: A/B confirms the input-traffic model and fits its cost; the
  gate_up kernel is 39% fixed cost. Predicts 1.4-1.8x from a wider out_batch.**
  ``gemma4_experts`` inserts the amortized dual owner only while ``in_features <=
  _GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES``. Lowering that constant to 2048 makes
  the same forward fall back to the non-amortized row-batch owner on identical
  shapes, which is a 4x spread in input traffic on the same kernel family. Both
  arms measured with the shape census, one 1024-token prefill each::

      arm  resolved variant                                  in-traffic   gate_up
      A    selected_dual_grouped_rowbatch8_out4_amortized    8.12 GB      7742 us
      B    selected_dual_grouped_rowbatch8                   32.5 GB     21824 us

  4x the input traffic costs 2.82x the time, so the kernel is **partially**
  traffic-bound. Two points fit a linear model::

      time = 3052 us fixed + 577 us/GB of traffic

  The slope is ~1.73 TB/s effective, **above the 864 GB/s HBM peak** -- independent
  corroboration of iteration 99's L2 inference, from a different experiment.

  **The fixed term is the more interesting number: 3052 us of the current 7742 us,
  39%, does not move with traffic at all.** That is compute/issue cost, and it is
  the floor this lever can reach. Predicted gate_up at wider out-blocks, from the
  fit rather than from a bandwidth argument::

      out_batch    in-traffic   predicted   vs now
          4 (now)    8.12 GB      7742 us    1.00x  (measured)
          8          4.06 GB      5562 us    1.39x
         16          2.03 GB      4391 us    1.76x
         32          1.02 GB      3802 us    2.04x
      asymptote     0            3052 us    2.54x

  **Why this justifies building the wider block.** The earlier case for it rested
  on an inferred L2-residency argument. This rests on a measured two-point fit
  whose slope is independently corroborated, and it bounds the payoff: no wider
  block can beat 2.54x on this kernel, because the fixed cost does not shrink.

  **The down projection did not move**: 7864 us in arm B against 8116 us in arm A,
  a 3% difference within run-to-run noise. The gate covers only the *dual* owner,
  so the 470 ms ``selected_grouped_prefill_pair2_fold128`` kernel was untouched by
  the switch and its own geometry is still unexamined. It is the single largest
  shape in the prefill and the next thing to read.

  Evidence: ``scripts/gemma4_amortized_ab.py`` (the A/B, committed).

  **Iteration 101: the input-traffic model is FALSIFIED. Wider out-blocks make the
  grouped gate_up slower, including at identical register cost. Iterations 99 and
  100 are superseded by this entry.**

  Three arms of ``gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel``, same
  shape (4096 rows, k=2816, n=1408, 128 experts), same 1024-token prefill, each
  measured after clearing the family build cache::

      template      accumulators   input traffic   gate_up
      <8, 4> now        64           8.12 GB       7742 us
      <4, 8>            64           4.06 GB      11756 us   1.52x slower
      <8, 8>           128           4.06 GB      14971 us   1.93x slower

  **Why this falsifies the model.** ``<4, 8>`` halves the input traffic that the
  amortized nest exists to reduce, at an accumulator footprint of 2 * 8 * 4 = 64
  floats -- *identical* to the production owner -- and it is 1.52x slower. If
  input traffic bound this kernel, halving it could not cost 52%. ``<8, 8>``
  halves traffic and doubles accumulators, and is slower again, so register
  pressure is a real second effect but not the primary one.

  **What iteration 100 actually measured.** Arm B there was the *non-amortized*
  nest, which differs from arm A in code structure -- columns outermost, rows
  re-walked -- as well as in traffic. The 2.82x ratio conflated the two, and
  fitting ``3052 us + 577 us/GB`` across that pair attributed a structural
  difference to traffic. The slope looked corroborated because it exceeded HBM
  peak, which is consistent with L2 but is not evidence that traffic *binds*.
  Iteration 99's L2 inference rested on the same conflation.

  **What the three arms do show.** They differ in per-CTA work and grid size, and
  time rises monotonically as per-CTA work rises::

      template   CTAs     FMAs/thread   gate_up
      <8, 4>     45056       2816       7742 us
      <4, 8>     22528       5625      11756 us
      <8, 8>     22528       5625      14971 us

  Total FMAs are identical across all three. So the cost tracks *how the work is
  divided*, not how much traffic it moves: fewer, longer-running CTAs hide load
  latency worse. The nest is **latency/issue-bound**. That is consistent with the
  39% "fixed" term iteration 100 isolated, and it is the term that a wider block
  cannot touch -- which is why the asymptote it predicted was never reachable.

  **What this closes and what it leaves.** The wider-out-block lever on the exact
  grouped owner is closed: two configurations tested, both regressions, one at
  equal register cost. It does not reopen weight traffic either -- the weights are
  already read exactly once per column. The remaining gap is issue/latency-bound
  structure, which is what a different compute path addresses rather than a
  retune of this one. The campaign already records the compensated WMMA owners at
  2.7x on a 1024-token prefill, held off the default path by the arithmetic gate;
  this measurement says the exact grouped nest has little left in it by retuning.

  **Reverted.** ``gguf_q4_k_selected_prefill.hip`` is back to ``<8, 4>`` and the
  reverted binary reproduces 7752 us against the 7742 us baseline, 0.1%. The
  rejection is recorded in a comment at the launcher so the next reader does not
  repeat the experiment.

  Evidence: ``scripts/gemma4_amortized_ab.py`` (unchanged, reused for all three
  arms).

  **Iteration 102: the expert-grid retune is neutral, and the WMMA MoE prize
  measures +11.6% on today's code, not the 2.7x the campaign records.**

  Two results, one negative and one that changes a pending decision.

  **Negative: the down projection's expert grid is not the constraint.**
  ``launch_q51_pair2`` caps ``grid.y`` at 64, so with Gemma's 128 experts each CTA
  walked two experts: 22528 CTAs at 360K MACs each, for a shape that is the
  single largest in the prefill (470 ms). Uncapping it to ``experts`` doubles the
  CTA count at half the work per CTA -- the direction iteration 101's two arms
  both favoured -- and measures 8079 us against a 7827-8165 us baseline range.
  No effect. Shared memory on that launcher is only ~4.6 KB, so the cap was never
  an occupancy limit, and the kernel is indifferent to it. Reverted; worktree
  clean.

  That is the third retune of these two kernels with no win: wider out-block at
  two accumulator footprints (iteration 101), and a wider expert grid here. Both
  kernels run ~30-60x below compute peak and do not respond to blocking or grid
  geometry. Whatever limits them is not the shape of the launch.

  **Decision-relevant: the WMMA arm is worth ~12%, not 270%.** The campaign
  records the compensated WMMA MoE owners as "2.7x faster on a 1024-token
  prefill", held off the default path by the arithmetic gate. Measured today,
  same box, same 1024-token prefill, via ``HIPENGINE_GEMMA4_MOE_PREFILL``::

      auto    prefill_s=1.711354   prefill_tps=598.36
      wmma    prefill_s=1.533729   prefill_tps=667.65

  That is **1.116x, +11.6%**. The arm is still faster, so the record is not
  wrong about direction, but the magnitude does not reproduce. This matters
  because the open lead decision is whether an absolute ``kl_max`` bar applies to
  a reordering-class change, and the cost side of that trade was priced at 2.7x.
  At 1.12x it reads differently: a modest win that still breaches an absolute bar
  on 1 of 1023 rows. The ruling is now low-stakes rather than the largest
  available lever.

  **Measurement caveat, and it is not small.** Production measured 598.36 tok/s
  here against the 653.68 recorded as the loop's current metric, and
  ``layer_total`` 1699.9 ms against ~1570 ms measured earlier in this same
  session. That is 8-9% environmental drift between runs. Two consequences:
  single-run comparisons below ~10% are not reliable and need repeats, and the
  iteration-101 rejections (1.52x, 1.93x) stand comfortably above that floor
  while a 1.12x result sits inside it. The WMMA number above is one run each and
  should be repeated before it is relied on; it is recorded as measured, not as
  settled.

  **Where this leaves the objective.** Three retunes dead, the gated WMMA prize
  ~12%, and the two MoE kernels ~30-60x below compute peak and unresponsive to
  the knobs tried. Reaching llama.cpp's 3910 tok/s from ~600 needs ~6.5x. No
  available knob supplies that; it needs a different kernel structure rather than
  more retuning of these two.

  Evidence: ``scripts/gemma4_prefill_census.py`` (both arms);
  ``/tmp/census_{auto,wmma}.txt``.

  **Iteration 103: the down projection re-reads its input 4x per CTA -- read from
  the code, not inferred. It is the gate_up's `AMORTIZE_INPUT` fix, never applied.**

  ``q5_1_selected_grouped_prefill_pair2_bf16_kernel`` at Gemma's down geometry
  (FOLD128, no pair-reduce, no register cache, no row publish) runs this nest::

      for (offset = 0; offset < 8; offset += 2) {          // 4 passes, 2 columns
        load metadata for 2 columns; __syncthreads();
        for (base = begin; base < end; base += R) {        // rows, R = 8
          for (column = threadIdx.x; column < k; column += L) {   // k, L = 256
            x0 = input[(base+r)*k + column];               // input load
            ...FMA...

  **The whole row-by-k traversal is inside the offset loop**, so every CTA walks
  its expert's input slice once per offset pass -- four times. This is exactly the
  shape the gate_up's ``AMORTIZE_INPUT`` exists to remove, and this kernel never
  got that treatment.

  Traffic at Gemma's down geometry (4096 rows, k=704, n=2816, 128 experts)::

      CTAs per expert       n / 8            = 352
      offset passes         per CTA          = 4
      rows per expert                        = 32
      input bytes           352*4*32*704*2   = 63.4 MB per expert
      total                 x 128 experts    = 8.1 GB

  Against a 5.8 MB input tensor that is a **~1400x re-read**, four times worse
  than the gate_up's 352x. At the measured 8.08 ms it is **~1.0 TB/s apparent**,
  and the time matches the traffic almost exactly.

  **The weight dequantization is not the problem** -- the hypothesis this
  iteration started from is wrong. ``w0[p]``/``w1[p]`` are hoisted out of the row
  loop, so each weight is dequantized once and reused across all R=8 rows, the
  same amortization the gate_up has. Only the input re-read differs.

  **The fix is the nest swap the gate_up already has.** Swapping amortized-vs-not
  is the one *unconfounded* direction in iteration 100's comparison -- arm A
  against arm B varied code structure and traffic together, but the swap itself
  measured 2.82x. The constraint is that reusing input across columns needs one
  accumulator per (column x row): 8 columns x 8 rows = 128 accumulators is the
  configuration that regressed 1.93x on the gate_up. The 64-accumulator options
  are 4 columns x 8 rows (traffic halves to ~4 GB) and 8 columns x 4 rows (input
  read once, ~2 GB, at the cost of halving the dequant amortization).

  Predicted from the traffic: 8.1 GB -> ~4 GB or ~2 GB, so ~4 ms or ~2 ms
  against 8.08 ms measured, on a shape worth 470 ms of the prefill.

  **Why this is not iteration 99-100 again.** Those inferred a mechanism from a
  two-point fit whose slope looked corroborated because it exceeded HBM peak; the
  inference was wrong and cost two iterations. Here the re-read is *in the code*
  -- the nest is literally written that way -- and the traffic model is a
  multiplication over constants that are all visible. The apparent-bandwidth
  agreement is corroboration, not the argument. The A/B is still the test, and if
  a swapped nest does not win, this entry is wrong and should be corrected by a
  new one.

  Evidence: ``qwen4_exp_q5_1.hip:515`` (kernel), ``:590-612`` (the nest),
  ``:1798`` (launcher).

  **Iteration 104: the down kernel's input re-read is real and measurable, but
  worth ~4% -- not the 2x the traffic model predicted. Fourth falsification.**

  Iteration 103 read the down kernel's nest and found the row-by-k traversal
  inside the offset loop, giving 4 input passes per CTA and 1408 passes per
  expert. This iteration built the fix: ``COLS`` (output columns per pass) as a
  template parameter threaded through all 16 sites that hardcoded the 2-column
  pair, with the fold128 entry instantiated at ``COLS=4``.

  The design is a **clean controlled test**, unlike the earlier attempts: the CTA
  count is unchanged at 352 x 64, the accumulator footprint is unchanged at
  64 floats, the work per CTA is unchanged. Only the input passes per expert
  change, 1408 -> 704, halving input traffic from 8.1 GB to 4.05 GB.

  Measured, three samples with the unchanged gate_up as a run-level control::

      arm        down us (3 samples)              gate_up us (control)
      COLS=2     7827 7827 8073 8080 8116 8159 8165 8165   7742 7752 7757 7797
      COLS=4     7719 7743 7764                            7708 7722 7739

  All three COLS=4 samples fall below the COLS=2 minimum, and the control says
  the box was ~1% fast in those runs, so the effect is real: **~4%, not 2x**.
  Four percent of a 450 ms shape is ~1% of the prefill.

  **The traffic model fails a fourth time, and this time cleanly.** Iterations
  99/100 inferred it from a two-point fit; iteration 101 refuted it for the
  gate_up; iteration 103 found a genuine code-level re-read and predicted 2x from
  it; halving that re-read moves the time by 4%. Input traffic is not the binding
  constraint in either MoE kernel, and the apparent-bandwidth agreement that
  looked like corroboration three times running was coincidence.

  **Reverted.** A ~1% overall gain does not justify an unmeasured change to a
  kernel path shared with Qwen3.5, and ``git restore`` returns the file exactly to
  HEAD. The measurement is recorded here so a later iteration can take it with
  proper Qwen coverage; the template parameter is the whole change and it is
  specified above.

  **What four failures say together.** Across these two kernels, these levers are
  now measured dead: wider out-block at two accumulator footprints (101), wider
  expert grid (102), and the input re-read itself (104). Nothing that changes how
  the work is *shaped* moves these kernels, and nothing that changes how much
  traffic they move does either. Both sit at 2-4 TFLOPS against a device with
  ~61 TFLOPS of fp32 and ~123 of WMMA fp16. The remaining explanation is the
  inner loop's instruction mix and latency structure, which is where the next
  effort has to look -- or a different compute path entirely, where the measured
  prize is the ~12% from iteration 102 rather than the 2.7x the record claims.

  Evidence: ``scripts/gemma4_amortized_ab.py`` (reused); reverted change described
  above.

  **Iteration 105: the platform is healthy and the 8-9% drift is a 12% clock
  swing. Kernel conclusions stand; A/B method has to change.**

  Four shape and traffic retunes failed to move the MoE kernels, and both land at
  ~7.7 ms per call despite very different shapes (16.2 G MACs against 8.1 G). A
  uniform shortfall across unrelated kernels is what a throttled or low-clocked
  device produces, so this iteration checked the platform state directly with
  ``rocm-smi`` under a real prefill load::

      t=4s    junction 32C   sclk    0 MHz   power  13 W   idle (model loading)
      t=8s    junction 42C   sclk 2588 MHz   power 267 W   active
      t=12s   junction 37C   sclk 2850 MHz   power  80 W
      t=16s   junction 38C   sclk    0 MHz   power  10 W
      t=20s   junction 73C   sclk 2500 MHz   power 240 W   active
      t=24s+  junction ~37C  sclk    0 MHz   power  14 W   idle (model loading)

  **The platform is healthy.** Under load the device holds 2588-2850 MHz, which is
  at or above the W7900's nominal boost, draws 267 W against a 295 W limit, and
  reaches 73 C junction. There is no throttling and no low-power state. So the
  kernels do saturate the device and the 20-40x-off-roofline finding stands -- the
  remaining gap is genuinely kernel inefficiency, not platform state.

  **A first reading of this data was wrong and is worth recording.** Sampling
  only from t=22s showed sclk 0 MHz and a flat junction, which read as "the device
  is idle, so the kernels are occupancy-bound". That was an artifact: the A/B
  script spends most of its wall time loading the model, so the samples missed the
  GPU windows entirely. Sampling from t=0 across back-to-back runs caught two
  267 W / 240 W bursts. The lesson is that a plausible mechanism (occupancy) plus
  an unvalidated instrument (sampling that missed the load window) produced a
  confident wrong conclusion in one step -- the same failure mode as iterations
  99-100, in a different costume.

  **The drift is explained, and it is thermal.** The device boosts to **2850 MHz**
  when cool and settles to **~2500 MHz** hot. That is a **~12% swing**, which
  brackets the 8-9% run-to-run variation measured repeatedly in this session
  (production 598.36 tok/s against a recorded 653.68; layer_total 1699.9 ms
  against ~1570 ms earlier in the same session). It is not a mystery and it is not
  noise in the ordinary sense -- it is a clock state that depends on how recently
  the GPU ran hard.

  **What this means for method.** Comparisons below ~10% are unreliable in the
  order the campaign has been running them, which is single-arm before/after:
  the first arm runs cool and fast, the second runs hot and slow. Consequences
  already visible in the record: the ``<8, 4>`` vs ``<4, 8>`` rejections (1.52x,
  1.93x) are far above the band and stand; the COLS=4 result (~4%, iteration 104)
  sits inside it and was correctly reverted rather than kept on a weak margin.
  Going forward, either interleave the arms or warm the device to a steady state
  before measuring, and state the clock state with any sub-10% claim.

  **Cheap practical note.** Model loading dominates wall time -- roughly 10-13% of
  a measurement run is GPU-busy. A GPU-bound microbenchmark that reuses loaded
  weights would make iteration several times cheaper than the current
  load-model-per-arm loop.

  Evidence: ``rocm-smi --showclocks --showtemp --showpower`` sampled at 4 s
  intervals across three back-to-back ``scripts/gemma4_amortized_ab.py`` runs;
  ``/tmp/clock_{1,2,3}.txt``.

  **Iteration 86: the MoE line has a grouped dp4a owner, and the Gemma path is
  already most of the way to it.** The dense win in iteration 85 leaves the two
  grouped MoE owners as the largest target by a wide margin -- ``moe_grouped``
  (q5_1, 450.3 ms) and ``moe_grouped_dual`` (q4_k, 443.1 ms) are 57% of the
  remaining prefill -- and the MoE dispatch consults the MMQ session zero times.

  **The owners exist, and their names mislead.** ``moe_linear`` carries dp4a
  owners for exactly these quants. The one that matters is
  ``gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out``:
  despite the ``selected_`` prefix it takes ``expert_start_compact_ptr`` -- the
  compacted, grouped expert layout -- and it is a *dual* owner writing both
  halves of the fused gate_up row, which is the same shape as the
  ``moe_grouped_dual`` owner it would replace. The prefix is legacy naming, not
  a statement about the layout.

  **Feasibility is checked, not assumed.** ``_check_mmq32_common`` requires
  ``in_features % 256 == 0`` and ``out_features % 32 == 0``; Gemma's expert
  gate_up is 2816 in (11 x 256) by 704 per half (22 x 32), and
  ``mmq_total_rows % 32`` holds by construction from the tile plan. Every
  constraint is satisfied.

  **The Gemma MoE path already has the hard parts.** ``Gemma4ExpertScratch``
  carries ``expert_start``, ``wmma_expert_start``, ``wmma_tile_expert`` and
  ``wmma_total``, and ``gemma4_experts_forward_bf16`` already builds the tile
  plan from them (lines 272-274) and already has ``packed_hidden``, the
  compacted activation batch. What is missing is the DS4-Q8_1 activation
  workspace and the routing.

  The reference wiring is ``qwen4_exp_runner.py:3794``, and it is three calls:
  ``gguf_q4_k_q8_1_mmq_ds4_pack_bf16`` to quantize the compacted activations
  into the ds4 workspace, ``qwen35_moe_wmma_tile_map`` to build the plan (with
  ``wmma_total_rows`` read back from the device and validated against the tile
  capacity), then the owner itself. The routing belongs in
  ``gemma4_experts_forward_bf16`` rather than in
  ``gemma4_project_experts_grouped_dual``, because that function receives
  ``expert_start_ptr`` but not the scratch the plan lives in.

  **Why this is worth doing: the MoE line is dequant-bound, like the dense line
  was.** Its expert weights are about 850 MB per layer, all 128 experts being
  active across a 1024-token batch, so a prefill reads roughly 49 GB of expert
  weights against a 51 ms floor at 960 GB/s -- and it takes 893 ms. It is at 6%
  of its memory floor and near 10% of peak, which is the same
  instruction-throughput wall iteration 81 diagnosed on the dense line, and the
  same fix applies.

  Two things are not established and should not be assumed. The q5_1 *down*
  projection (450.3 ms, the other half of the MoE line) has only a
  ``selected_`` dp4a owner in the registry, not a ``compact32`` one, so it may
  not be applicable to the grouped path at all. And per iteration 85's lesson,
  the only measurement that counts here is the model's own census: an isolated
  crossover with constructed activations is what produced a wrong answer twice.

  **Iteration 85: the MMQ dense line lands, +14.4% prefill, and it is the
  campaign's first non-exact win.** Iteration 84 wired the guarded d4x3 chain
  and measured a regression, then reverted. The regression was not the chain: it
  was the threshold, and the measurement that settles it is the guard's queue
  rate on real weights.

  **The guard's queue rate, real GGUF Q8_0 weights, 512 rows:**

  | threshold | queued | differing from the exact owner |
  | --- | ---: | ---: |
  | 1e-8 | 0.0% | 271-985 |
  | 1e-7 | 0.0% | 118-484 |
  | 1e-6 | 0.2-0.3% | 0-12 |
  | **1e-5** | **1.5-2.5%** | **0** |
  | 3e-5 | 3.8-6.2% | 0 |
  | 1e-4 | 10-16% | 0 |
  | 1e-2 | 100% | 0 |
  | 1.0 | 100% | 0 |

  Two things this settles. The guard queues *more* as the threshold rises, which
  is the opposite of what the name suggests and is why iteration 84's
  ``threshold=1.0`` run queued everything and measured 211 tok/s -- the "sparse"
  correction became a full exact recompute. And 1e-5 is the setting where the
  chain is both exact and cheap: repairing 1.5-2.5% of the elements at
  ``in_features`` MACs each costs about 2% of the GEMM, while repairing 10-16%
  costs about what the GEMM costs. **The Qwen policies already used 1e-5.** I
  changed it to 1e-4 without evidence, and that one number was the whole
  regression.

  **Measured on the XTX census, same script and artifact:**

  | | before | after |
  | --- | ---: | ---: |
  | ``dense:gguf_q8_0`` | 548.9 ms | **389.4 ms** (1.41x) |
  | ``layer_total`` | 1781.6 ms | **1553.7 ms** |
  | prefill | 571.42 | **653.68 tok/s** (+14.4%) |

  W7900, same script: ``dense`` 605 -> 424.8 ms, prefill 522.5 -> 598.9 tok/s
  (+14.6%). In the shape census every admitted shape improved 1.2x-1.8x --
  (4096,2816)@512 1976 -> 1109 us, (8192,2816)@512 3873 -> 2307 us -- and
  (2112,2816), which the ``min_rows`` map excludes because 2112 is not a
  multiple of 128, is untouched at 1071 -> 1089 us.

  **This is the first change in this campaign that is not bit-identical, and
  that is the honest headline.** The teacher-forced gate returns ``passed:
  true``, ``failed: []``, with ``kl_max`` 0.0064 against a 0.05 bar (7.8x
  margin), ``kl_mean`` 1.6e-05 against 0.001 (62x), ``kl_p99`` 8.1e-05, and zero
  top-1 flips over 1023 x 262144 logits. It is not 0.0, which every previous
  landing in this campaign was. The isolated per-element comparison showed 0
  differing at 1e-5 because it used synthetic activations; the model's real
  activations drift slightly and the guard does not queue every drifting
  element. So the arithmetic requirement is met by the execution-profile gate
  and by nothing else: this path is a bounded-deviation production candidate,
  not an exact one, and the gate's own ``promotion_qualified`` is ``False``
  because the category, task and isolation gates are separate and have not been
  run.

  **Iteration 84: the MMQ chain is wired end to end and is a measured
  regression on this model. Reverted.** Iteration 83's crossover said 1.5-2.25x.
  That number did not survive contact with the real weights, and this records
  both the wiring that works and why the estimate was wrong.

  **The wiring, which is the reusable part.** The shared linear dispatch reads
  the policy *off the session* (``session.policy(rows, in, out)`` and
  ``session.policy.risk_threshold``) rather than re-resolving one from the
  registry, so a model can supply its own ``Q8MMQPrefillPolicy`` and keep its
  own crossover map and threshold. That matters here because this GGUF is
  ``UD-Q4_K_XL`` -- the same file type as the Qwen4Exp model that already owns
  the ``gguf_ud_q4_k_xl`` registry key, so registering a Gemma policy under it
  would have collided. The session is established by wrapping
  ``Gemma4Runner._forward_block`` (split into a thin wrapper over
  ``_forward_block_inner`` so the layer loop needed no re-indentation), with a
  workspace sized from the policy's own widest shape rather than from the block,
  plus a 4-byte risk counter and a ``rows * widest_out`` index queue, all
  allocated through ``_alloc`` so they are freed with the runner. That wiring
  works: the dispatch did select the MMQ owners, as the timings below show.

  **The measurement, on the same instrumented prefill as iteration 83's shape
  census (559.5 tok/s, dense 654.8 ms at baseline):**

  | config | tok/s | dense ms | (2816,2112) @512 | (2816,8192) @512 |
  | --- | ---: | ---: | ---: | ---: |
  | exact (baseline) | 559.5 | 654.8 | 1084.7 us | 3830.3 us |
  | MMQ, ``risk_threshold=1e-4`` | 516.3 | 854.6 | 2087.1 us | 5960.8 us |
  | MMQ, ``risk_threshold=1.0`` | 211.3 | 3699.7 | 7201.8 us | 29539.5 us |

  Every admitted shape got slower, and raising the threshold made it far worse
  rather than better -- the opposite of what the guard's design implies. Why is
  not established; the cost is not simply "how many rows the correction
  repairs", which is what iteration 83 assumed.

  **Why the crossover estimate failed, which is the transferable lesson.** The
  isolated harness used ``make_q8_0_weight`` synthetic weights; the model uses
  the GGUF's own. The guard's repair rate is a function of the weight values --
  it estimates each row's drift and queues the rows it expects to exceed the
  threshold -- so synthetic weights with a benign drift pattern measure a chain
  that never fires its expensive half. Iteration 83 even saw the symptom and
  dismissed it: passing ``inf`` measured 0.13x, which was written up as "the
  pathological case" instead of as the first evidence that the correction's cost
  is the whole story. A crossover measurement for a *guarded* path has to use
  the real weights, because the guard's cost is data-dependent by construction.

  Reverted; the exact tile16x4 owner stays on the dense line. The arithmetic
  route is still the right direction -- llama.cpp's 25%-of-peak is the existence
  proof -- but not through this chain as it stands, and not on the strength of a
  synthetic-weight crossover.

  **Iteration 83: the MMQ d4x3 chain is bit-identical and 1.5-2.25x on this
  model's dense shapes.** Iteration 82 found the dp4a owners registered but
  unadmitted; this measures the admission contract's ``min_rows`` crossover and
  tests the exactness claim instead of assuming it.

  **The dense line's real shapes.** Wrapping ``gemma4_project`` for a 1024-token
  prefill gives the shape census. Prefill runs in 512-row chunks with a 64-row
  tail, every dense projection is ``gguf_q8_0``, and seven shapes carry about
  88% of the line: (2816, 2112) 130.2 ms, (2816, 2048) 101.9, (4096, 2816)
  98.8, (2816, 4096) 98.6, (2112, 2816) 67.6, (8192, 2816) 38.7, (2816, 8192)
  38.3. That is the target list, and it is why the Qwen policy's
  ``max_out_features=8192`` fits here too.

  **Crossover, measured with the full chain timed (quantize, MMQ, sparse
  correction) against the exact tile16x4 owner, and bit-compared:**

  | shape | rows | exact ms | chain ms | speedup | differing elements |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 4096, 2816 | 512 | 2.026 | 0.990 | **2.05x** | 18 |
  | 2816, 4096 | 512 | 2.135 | 1.228 | **1.74x** | 0 |
  | 8192, 2816 | 512 | 4.051 | 1.800 | **2.25x** | 162 |
  | 2816, 4096 | 64 | 0.272 | 0.498 | 0.55x | 0 |
  | 8192, 2816 | 64 | 0.552 | 1.225 | 0.45x | 19 |

  (the 1e-5 rows above; at ``risk_threshold=1e-4`` every row repairs and the
  differing count is 0 at 1.47x-2.00x.)

  **``risk_threshold`` is the exactness knob, and that is the finding.** The
  guard estimates each row's drift and queues the ones it thinks will exceed the
  threshold; the sparse correction then recomputes exactly those rows. At
  ``1e-5`` the estimate misses a few rows -- 18 to 162 elements of 1.4 million
  differ from the exact owner -- and at ``1e-4`` it misses none. So the chain is
  exact at 1e-4 and merely close at 1e-5, which is a different claim than "d4x3
  is the exact chain": it is exact *at a threshold*, and the threshold is
  per-model data, not a constant to inherit. An earlier run in this iteration
  passed ``inf`` and measured 0.13x, because flagging every row turns the
  "sparse" correction into a full recompute -- the guard's cost is only
  meaningful against a realistic threshold.

  **Eligibility.** ``2112`` is not a multiple of 128, so ``(2112, 2816)`` -- 10%
  of the dense line, 67.6 ms -- cannot use this path at all. That is a capability
  miss in the d4 packing, not an admission gate, and it needs either a 32-wide
  variant or a retained exact owner for that shape.

  **Shape of the win.** The eligible shapes at 512 rows carry about 507 ms of
  the dense line; at 1.7x that is roughly 200 ms, about 11% of prefill, and it
  is bit-identical rather than reassociated. The MMQ owners are registered at
  layer ``linear`` only, so the MoE grouped owners (49% of prefill) have no
  dp4a route today and are the larger prize if the same chain can be built for
  ``moe_grouped_dual``.

  **Iteration 82: the arithmetic route is already in the tree, unwired for
  this model.** Iteration 81 concluded the dense Q8_0 line is limited by
  instruction throughput -- a dequant step per weight on top of the FMA that
  consumes it -- and that the fix is an integer dot-product path. That path
  exists.

  ``gguf_q8_0_mmq_prefill.py`` registers 44 MMQ owners, including, at layer
  ``linear`` for quant ``gguf_q8_0``,
  ``mmq128_prefill_q8_1_d4x3_guarded_bf16_bf16_out`` -- the same bf16-in,
  bf16-out ABI the dense line already uses. What is missing is the policy:
  ``resolve_q8_mmq_prefill_policy("gguf_q8_0")`` returns ``None``, because
  policies are registered per model quant and only the Qwen quants
  (``gguf_ud_q3_k_m``, ``gguf_ud_q4_k_xl``) have one.

  ``Q8MMQPrefillPolicy`` is the admission contract: a per-shape ``min_rows``
  crossover map keyed by ``(hidden, out_features)``, ``max_rows``,
  ``risk_threshold``, ``max_out_features``, and ``planes``. The chain is
  guarded: the session carries a bounded risk counter and index queue, so rows
  whose arithmetic would drift are detected and repaired rather than silently
  wrong. Its own docstring distinguishes the chains -- ``planes=2`` (d4x2) is
  the +1.4x candidate with "quantization-level drift pending envelope
  qualification", while "the default stays the retained three-plane d4x3 exact
  chain".

  **That distinction matters more than the speed.** If d4x3 is exact by
  construction and by gate, then enabling it is not a changed-arithmetic
  promotion at all -- it is a faster path to the same numbers, and the
  promotion rule for reassociating candidates does not apply. That is a
  question for measurement, not for argument.

  Why it is worth the integration: on this artifact llama.cpp's 3910 tok/s
  over a 1024-token prompt at ~4B active parameters is about 31 TFLOPS, ~25%
  of the XTX's bf16 peak. The exact path's 571 tok/s is 4.6 TFLOPS, ~3.7%. The
  gap is arithmetic efficiency, and llama.cpp closes it with dp4a.

  The work, in order: measure the MMQ-versus-exact crossover per Gemma dense
  shape to fill ``min_rows`` (the policy comment records that narrow and short
  shapes lose once boundary repair is included, so the map is not uniform);
  register the policy for this model's quants; establish the session in the
  Gemma runner with a workspace and the risk buffers; then measure both lanes
  and run the teacher-forced gate. Every piece of this is already built and
  measured on another model -- what is missing is the wiring and this model's
  crossover numbers.

  **Iteration 81: the dense Q8_0 line is instruction-bound, not traffic-bound.**
  A diagnostic iteration, recorded because it closes off a whole direction that
  looked promising on paper.

  ``dense:gguf_q8_0`` is 548.9 ms of the XTX prefill (30.8%) over 410 calls at a
  1338 us mean, and the census attributes it by wrapping the Python launch site
  (``census.wrap(gemma4_layer, "gemma4_project", dense_label)``), so those are
  ``gemma4_project`` calls carrying a q8_0 weight rather than a trace family.

  Pricing one representative shape -- [1024, 2816] x [2816, 2816], 8.1 GFLOP in
  1.34 ms -- gives 6.0 TFLOPS, about 10% of the XTX's bf16 peak and 20% of fp32.
  Its unique bytes are 8.4 MB of weights plus 5.8 MB of activations, which is
  15 us at HBM speed against 1340 us measured: 90x off the memory floor. So the
  re-read of a weight tile by every row group is *not* what limits this line,
  even though the amplification ratio (about 88x) makes it look like the
  obvious target. L2 absorbs those re-reads.

  What limits it is instruction throughput: every weight needs a dequant step
  (scale multiply, int8 to float convert) on top of the fp32 FMA that consumes
  it, roughly doubling the instruction count per MAC. That is a vectorisation
  and datatype problem, not a tiling one, and the fix that addresses it is an
  integer dot-product path (dp4a/MMQ, llama.cpp's own arithmetic) rather than a
  larger register tile. Chasing the tile would have cost a kernel rewrite to
  recover something L2 was already hiding.

  **Iteration 80: pass 3 is bounded by the mask's last kept key.** The kernel
  already contained the argument this generalises: ``key_begin``'s docstring
  shortens the key range of a one-row block because a dropped key contributes
  exactly zero to both reductions and the surviving terms keep their ascending
  order. That argument is per row, not per block, so the kernel can read the
  bound out of the caller's own mask rather than assuming a shape.

  Pass 3 walks every key for each of the row's dimensions, and the tail past the
  last kept key is dead for that row and contributes weight 0 to every one of
  them -- 0.0 * v is 0.0 and adding 0.0 to a float is identity, so skipping those
  terms is exact. ``last_active`` is a block max-reduce over ``mask_row``,
  reusing ``max_s`` after ``row_max`` was reduced from it and pass 2's tree
  barriers had retired the reads. Because the bound is read from the mask rather
  than assumed causal, it holds for the windowed sliding layers as well as the
  full causal ones; pass 3 takes the bound as its ``keys`` argument, so the
  change is at the call site plus the reduction.

  Measured, W7900 census, same script and artifact: ``attention_prefill`` 227.1
  -> 214.7 ms (-12.5, -5.5%), prefill 513.4 -> 522.5 tok/s (+1.8%). Exactness:
  the teacher-forced gate returns ``kl_max`` 0.0 with ``passed: true`` and
  ``failed: []`` over 1023 x 262144 float32 logits.

  **The estimate was 57 ms and the measurement is 12.5 ms.** I priced pass 3 by
  its iteration count -- half a causal row's walk -- but the prefetch pipeline
  had already made that walk latency-tolerant and the existing ``weight == 0``
  skip was already suppressing the V loads. What was left to save was loop
  overhead, not memory traffic. The same mistake as the Q5_K probe in iteration
  79, in the opposite direction: a count of iterations is not a cost when a
  pipeline or a skip has already hidden most of them.

  A note on the gate script: its output flag is ``--out``, not ``--json``. The
  first invocation of this arm died on argument parsing and wrote nothing, while
  exiting 0 and being reported as a success. A background task's exit status is
  not evidence that its command was valid.

  **Iteration 79: the Q5_K grouped dual owner lands, 6.3x on the two outlier
  layers.** Iteration 78 established that the port was worth writing and exact in
  principle; this is it, built and measured.

  Q5_K and Q4_K share a superblock prefix -- d, dmin, and the same 12-byte 6-bit
  scale packing -- and the same nibble order in ``qs``. Q5_K adds a 32-byte
  ``qh`` slab holding each weight's fifth bit, at bit ``subblock`` of
  ``qh[lane]``, with ``qh`` not advanced between subblocks. So the port is one
  helper plus a ``Q5K`` template parameter on the decode, the row stride and the
  metadata slab: ``gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel`` is now
  instantiated ``<8, 4, true, true, true, true>`` and exposed as
  ``hipengine_gguf_q5_k_selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out``.

  **One real bug caught by reading rather than by a failing test.** The kernel
  hardcodes ``Q4_K_BLOCK_BYTES`` in three places -- the row stride and both sides
  of the metadata slab build. For Q5_K those read the wrong offsets, so the slab
  would have been filled with another block's scales and mins: wrong numbers, no
  fault, and a bit-comparison against the selected GEMV is the only thing that
  would have caught it after the fact. All three are now quant-aware. A first
  pass also templated a call site inside ``gguf_q4_k_selected_dual_grouped_pair2_bf16_kernel``,
  a different and untemplated kernel, and was reverted.

  Measured on the W7900 census, same script and artifact:

  | family | before | after |
  | --- | ---: | ---: |
  | ``moe_selected:gguf_q5_k`` | 155.9 | **0.0** |
  | ``moe_grouped_dual:gguf_q5_k`` | -- | **24.9** |
  | ``layer_total`` | 2121.0 | **1982.4** (-138.6) |
  | prefill | 480.3 | **513.4 tok/s** |

  The two layers' ``gate_up`` went from 155.9 ms to 24.9 ms, 6.3x, and the
  selected family for that quant is empty. The other families are unmoved
  (``moe_grouped_dual:gguf_q4_k`` 488.5 -> 484.9, ``moe_selected:gguf_q8_0``
  49.1 -> 47.5).

  **Exactness, end to end:** the teacher-forced gate against the pre-change
  ``fold128`` capture returns ``kl_max`` 0.0 over 1023 x 262144 float32 logits
  with ``failed: []`` and ``passed: true``. That is stronger than the isolated
  bit-comparison in iteration 78 -- it is the whole model's next-token
  distribution, unchanged.

  A note on the estimate, since it moved twice. The projection from the census
  was ~139 ms; the isolated harness then said 1.55x and 55 ms, which was wrong
  because its synthetic expert layout is not the model's; the census measured
  138.6 ms after all. The isolated probe is the right tool for *exactness* and a
  poor one for *throughput* when the weight layout differs from the real one.

  **Iteration 78: the port to serve the two outlier layers is exact and is
  mechanical.** Iteration 77 left one open question that decided whether writing
  a grouped owner for Q5_K/Q8_0 was worth anything: the grouped owners are
  documented bit-identical to *each other*, never to the selected GEMV they would
  replace, so a ported owner might have been unpromotable no matter how fast.

  Measured directly. Q8_0 at Gemma's down geometry (8192 compact rows, 128
  experts, in 704, out 2816), ``selected_gemv_bf16_bf16_out`` against
  ``selected_grouped_gemv_bf16_bf16_out``:

  ```
  differing: 0 / 23068672        BIT-IDENTICAL
  ```

  So the two families do share one arithmetic -- same thread-to-column map, same
  256-thread tree per output -- and a ported ``out4_amortized`` owner inherits
  that, which is what makes it promotable to the exact default rather than another
  numerical candidate.

  The port itself is mechanical rather than a rewrite. The Q4_K dual kernel
  (``gguf_q4_k_selected_prefill.hip`` line 191,
  ``gguf_q4_k_selected_dual_grouped_rowbatch_bf16_kernel``) decodes through small
  helpers -- ``gguf_q4_k_scale``, ``_min``, ``_quant``, ``_weight``, and a paired
  ``q4_k_weight_pair128`` -- and Q5_K differs from Q4_K in exactly one place: the
  32-byte ``qh`` slab carrying each weight's fifth bit, on the same superblock
  layout and the same 6-bit scale packing. A ``gguf_q5_k_*`` set of the same
  helpers plus an instantiation is the whole change.

  Worth doing for the ``gate_up`` and not for the ``down``. The 58 layers run the
  dual amortized owner at this geometry in 488.5 ms, about 8.4 ms per layer, while
  the two outlier layers' Q5_K ``gate_up`` costs 77.97 ms per call -- so a ported
  owner is worth roughly 139 ms, 7.2% of the 1939 ms default path. Their Q8_0
  ``down`` is a different story: at in 704 the selected GEMV is already 24.75 ms
  per call, against 40.2 ms for the q5_1 amortized owner at the same geometry, so
  there the incumbent is the faster shape and a port would lose.

  **Iteration 77: the two outlier layers, settled by the registry rather than by
  inference.** The 205 ms those two layers cost (10.6% of the default path) had
  been attributed twice by reasoning about which owners "should" exist. The
  registry says it directly, and the earlier reasoning was wrong in both
  directions.

  What is registered at ``moe_linear``:

  | quant | owners | grouped/rowbatch/fold forms |
  | --- | ---: | --- |
  | ``gguf_q4_k`` | 37 | all of them |
  | ``gguf_q5_1`` | 22 | all of them |
  | ``gguf_q5_k`` | 13 | **none** |
  | ``gguf_q8_0`` | 0 | **none** |

  So the two layers fall back to the selected GEMV for a real reason: the grouped
  owners exist for the two quants that serve the other 58 layers and for neither
  of these. At layer ``linear``, ``gguf_q5_k`` registers 101 variants and
  ``gguf_q8_0`` 68, but the grouped ones among them
  (``selected_grouped_row4_gemv``, ``selected_grouped_gemv``) are the
  column-per-CTA shape already measured at 2.6-2.8x *slower* than the incumbent
  in iteration 75 -- they re-read the activation row batch once per output column.
  The owner that would win is the ``out4_amortized`` shape, which exists only for
  q4_k/q5_1 and binds a quant-specific symbol in its own module
  (``qwen4_exp_q5_1_...``, ``gguf_q4_k_selected_prefill``), so serving these two
  layers means porting that loop nest to Q5_K/Q8_0's block decode -- new kernel
  work, not a registration or an adapter. Whether the ported owner would be
  bit-identical to the selected GEMV is also open: the grouped owners are
  documented as bit-identical to *each other*, not to the selected path.

  **A method error worth recording.** The first pass at this question tested each
  variant with ``try: _ensure_linear_kernel_registered(key)`` and read "no
  exception" as "registered". It is not: that helper does not raise for a key it
  cannot serve, so every variant looked served for every quant and the answer came
  out inverted -- "the owners exist, the probe is rejecting them". The
  ``resolve()`` call in the isolated probe is what exposed it, returning
  ``MissingKernelError`` for a key the earlier test had just "passed". Any future
  check of this kind has to read ``registered_keys()`` after the ensure, which is
  what the table above does.

  **Iteration 76: the WMMA arm re-measured, and a correction to how it was being
  described.** The arm's number had been quoted from a pre-session measurement for
  several iterations. Re-measured on the XTX with the same bench, the same 1024
  prompt and the same artifact as the exact path:

  | config | prefill tok/s | vs exact | ``kl_max`` (bar 0.05) |
  | --- | ---: | ---: | ---: |
  | exact, current | **525.06** | 1.00x | 0.0, passes |
  | MoE-only (``GEMMA4_MOE_PREFILL=wmma``) | 562.25 | 1.07x | 0.0609 |
  | MoE+dense (``+GGUF_WMMA_PREFILL=1``) | **757.60** | 1.44x | 0.161 |
  | llama.cpp HIP 8cfc315 | 3910 | 7.4x | -- |

  Two things this settles. First, the arm is worth **1.44x**, not the
  campaign-changing factor it had been described as: promoted in full it lands at
  758 against llama.cpp's 3910, still 5.2x short. Earlier iterations in this
  campaign framed the arm as *the* route to the objective's target; that framing
  was wrong and this table is the correction. Second, the two opt-ins are
  complementary and must be quoted together -- the 736.84 figure came from
  ``MoE+dense``, so a re-measurement with the MoE opt-in alone reads 562.25 and
  looks like a 24% regression that did not happen. The arm's own progress across
  this session is 736.84 -> 757.60, which is the attention tile skip (1.21x on a
  kernel both paths share), not a change to any WMMA owner.

  The gate status is unchanged and still applies: the arm's captures were taken
  before this session's two changes, but both are bit-identical (the teacher-forced
  gate returns ``kl_max`` 0.0 for them), so the baseline the arm was scored
  against has not moved. ``gate-pf-moe-only`` fails only ``kl_max``, at 0.0609 --
  1.22x over -- with ``kl_mean`` 8.58e-05 against a 0.001 bar, ``kl_p95`` 7.70e-06
  against 0.005, ``kl_p99`` 9.89e-05 against 0.02, and zero top-1 flips in 1023
  rows. ``gate-pf-dense-only`` is 0.180 and ``gate-pf-full`` 0.161.

  What follows from the table is that the decision space is not "exact versus
  WMMA". Neither side reaches llama.cpp: the exact path passes every threshold at
  525, the arm buys 1.44x at 758 and fails one threshold by 3.2x. The 3910 figure
  comes from int8 dp4a MMQ with quantized activations -- a 4x-per-instruction
  inner loop that no bf16 arrangement reproduces. That family is in-tree
  (``mmq128_prefill_q8_1_*``, ``q8_1_dp4a_grouped_*``) but is entered by the
  Qwen35 GGUF runner through a workspace and risk-count session rather than a
  flag, and quantizing activations is a larger arithmetic step than the WMMA arm
  already failing the bar. Reaching the objective's target is an integration and
  numerical-policy decision, not a remaining scheduling one.

  **Iteration 75: the row4 GEMV is the wrong tool for the two outlier layers, and
  the dense tile ceiling is real.** Two closed questions, both now measured.

  *Row4 GEMV.* The earlier iteration recorded a GPU fault when driving
  ``selected_grouped_row4_gemv_bf16_bf16_out`` at the Q5_K ``gate_up`` shape and
  stopped rather than wire it in on a guess. The fault was in the *probe*: it
  sized Q5_K rows at 144 bytes, which is Q4_K's superblock, so the kernel walked
  past the end of a buffer that should have been 176 bytes per 256 weights. The
  kernel's own row derivation is ``x_row[r] = row[r] / (rows / x_rows)``, which
  the identity mapping satisfies, and the call is legal. It is also the wrong
  kernel: 205.5 ms for the Q5_K shape against the incumbent selected GEMV's 78.0,
  and 70.0 against 24.8 for Q8_0. Its grid is one CTA per (output column, expert)
  with the rows walked inside, so every one of the 1408 column-CTAs re-reads the
  same activation rows -- 65 GB of x traffic per call at 316 GB/s. It is a
  decode-shaped kernel, and the two layers still need a new row-batched grouped
  owner for q5_k/q8_0 rather than an adapter.

  *Dense tile ceiling.* The tiled exact owner's ``static_assert`` capped
  COL_TILE x ROW_TILE at 64 accumulators. Traffic there is
  ``(rows / ROW_TILE) * W + (out / COL_TILE) * X``, so at the dense 2112x2816
  shape the incumbent 16x4 moves 2.38 GB (1.62 GB of it weights re-read once per
  row tile) and a 16x8 shape would move 1.57 GB at the same weight traffic and
  half the x. That is a 34% traffic cut, so the cap was worth testing directly.
  16x8 was instantiated, and it is **bit-identical to 16x4** -- confirming the
  tile shape does not touch the association, since the reduction tree depends on
  the 128-thread block, not on the tile.

  It is also slower. Achieved bandwidth falls from ~930 GB/s to ~490-565 GB/s
  across the three dense shapes, so the traffic model is not the binding
  constraint at that point: 128 fp32 accumulators plus the weight and x register
  sets put the kernel past the occupancy knee, and the lost memory-level
  parallelism costs more than the traffic saved. 16x4 holds the three dense
  shapes at 2.53-2.57 ms and 4.7-4.8 TFLOPS. The 64-accumulator bound is a
  measured ceiling, not a stale guess, and the 16x8 instantiation was reverted
  rather than left registered as an unselected candidate.

  Both of these say the same thing about the remaining exact path: every large
  family is now bounded by something that is not a scheduling mistake -- dense by
  occupancy against traffic, q5_1 down by the chronology-pinned LDS tree, gate_up
  by input reuse that the dual owners already capture, attention by passes 2 and
  3 walking every key. The gap to llama.cpp's 3910 tok/s is the arithmetic route
  (int8 dp4a with quantized activations), which no amount of exact scheduling
  closes.

  **Iteration 74: the key-class attention kernel paid for the whole square on a
  causal block. 1.21x on attention, bit-identical.** The prefill launcher already
  routes multi-token blocks to the decode-class family (measured 421 us against
  the block kernel's 652-700 at keys 1024, head_dim 256), so attention was not
  running the older schedule. The waste was inside that kernel: pass 1 computed
  every key's dot product and *then* discarded the masked ones
  (``active[t] ? reduced * scale : -INFINITY``). The block kernel it replaced
  guards the dot with ``if (mask_row[j] != 0)``, so the key-class kernel was
  strictly worse at exactly the geometry prefill uses.

  The caller hands the layer a ``(rows, keys)`` keep mask -- causal on full
  layers, windowed on sliding ones -- so the dead keys are knowable per row, but
  the existing shortcut cannot use them: ``key_begin`` shortens the key range by
  moving the K/V/mask pointers forward, which is only valid for a one-row block
  because a multi-row block has a different bound per row. The kernel therefore
  needed a per-row tile skip. A warp's tile covers ``j0 + t * warps``, so for a
  causal mask every tile with ``j0 > token`` is entirely dead, which is half the
  tiles of the average row when tokens == keys.

  The skipped tiles still write ``-INFINITY`` into their ``logits_s`` slots: pass
  2 reads that array for every key, so the write is what keeps the change
  arithmetic-free rather than merely approximately so. ``running_max`` is
  untouched because ``fmaxf(x, -INFINITY)`` is a no-op, and pass 2 adds
  ``expf(-INFINITY - row_max) = 0`` at the same ascending-j step, so no surviving
  term moves. This is the same principle ``key_begin``'s docstring already
  states -- a masked key contributes zero to both reductions.

  Measured on the W7900 census (same script, same artifact, back to back):

  | family | before | after | delta |
  | --- | ---: | ---: | ---: |
  | ``attention_prefill`` | 294.9 | 244.5 | **-50.4** |
  | ``layer_total`` | 2177.0 | 2121.0 | -56.0 |
  | ``dense:gguf_q8_0`` | 605.2 | 604.4 | -0.8 |
  | ``moe_grouped:gguf_q5_1`` | 497.0 | 496.1 | -0.9 |
  | ``moe_grouped_dual:gguf_q4_k`` | 491.8 | 488.5 | -3.3 |
  | ``moe_selected:gguf_q5_k`` | 155.9 | 155.9 | 0.0 |

  Every other family is flat within 3 ms, so the 50 ms is the attention kernel
  and not a clock excursion. Attention is 244.5 ms against 4.9 ms per call
  before, 4.07 ms after.

  **Exactness:** the teacher-forced gate against the pre-change ``fold128``
  capture returns ``kl_max`` 0.0 over 1023 x 262144 float32 logits, ``passed``
  true, zero top-1 flips -- with the baseline recording attention ``.hip`` sha
  ``df40ebd7...`` against the candidate's ``9e5dbd00...``, so the gate is
  certifying a changed source as exactly equivalent rather than re-scoring an
  unchanged one.

  What the skip does not reach is the rest of the kernel: pass 2 still walks
  every key with ``expf``, and pass 3 still walks every key with the V loads
  (its ``weight == 0`` guard stops the loads but not the walk). Both are O(keys)
  without a dot, which is why the win is 1.21x and not the 2x the dead half of
  the triangle would allow.

  **Iteration 73: the row4 GEMV adapter for the two outlier layers does not
  reduce to an argument reorder.** The two Q5_K/Q8_0 layers (205 ms, 9.4%) have no
  grouped owner at ``moe_linear``, but ``gguf_k_gemv`` registers
  ``selected_grouped_row4_gemv_bf16_bf16_out`` for both quants at layer
  ``linear``, which is the expert-grouped form with four-row weight reuse. Its
  signature differs from the grouped probe's by one argument
  (``lane_to_row_ptr``) and by passing ``x_rows`` and ``rows`` separately, so the
  adapter looked like a reorder: the Gemma grouped path has ``expert_start``
  already, and the selected path it would replace passes ``x_rows = rows = lanes``
  (``gemma4_experts.py`` line 368), so identity was the obvious mapping.

  It is not. Driving the kernel in isolation at the Q5_K ``gate_up`` shape
  (8192 compact rows, 128 experts, in 2816, out 1408, 285 MB of Q5_K blocks) with
  ``lane_to_row = NULL`` and ``x_rows = rows = 8192`` **faults** with a GPU memory
  access fault. ``lane_to_row`` itself is not the cause: ``gguf_k_gemv.hip`` line
  1405 guards it and substitutes the identity when it is null. So the mismatch is
  in the row-mapping semantics -- what the kernel derives from ``x_rows``,
  ``rows`` and the expert offsets when the compact activation is already gathered
  and there is no dense source row to map back to.

  That makes this job a kernel-semantics investigation rather than the small
  adapter the last iteration recorded, so it stops here rather than being wired in
  on a guess. The other exact job -- several query rows per attention CTA sharing
  one K/V pass, 295 ms at 13.5%, bit-identical because each row keeps its own
  logit tree and its own pass-3 order -- remains open and is new kernel work.

  **Iteration 72: the Qwen35 dual forms, reached and rejected; and the exact path's
  three limits.** Iteration 71 could not call the Qwen35-era dual owners because
  they reject the fused-stride keywords the Gemma probe passes. They accept a
  simpler call, so they were swept that way:

  | owner | ms | bits |
  | --- | ---: | --- |
  | ``rowbatch8_out4_amortized`` (incumbent) | 38.24 | ref |
  | ``pair2`` | 43.63 | **differs (5,761,400 elements)** |
  | ``expertgrid64`` | 46.90 | **differs (5,761,401)** |
  | ``expertgrid64_bundle`` | 43.54 | **differs (5,761,400)** |
  | ``rowbatch8`` (one output column) | 123.38 | identical |

  Slower *and* different: the Qwen35 dual forms are neither a schedule upgrade
  nor the same arithmetic at this layout, so the fused ``gate_up`` keeps its
  incumbent. Unlike the Q5_1 down -- where the Qwen35-era ``pair2_fold128`` was
  both 1.84x faster and bit-identical -- nothing transfers here.

  With that, all three large exact lines have a measured limit rather than an
  assumed one:

  - **Grouped Q5_1 down** (497 ms): taken to ``pair2_fold128``, 1.80x. The
    incumbent family's own ladder is exhausted and the remaining cost is the
    chronology-pinned LDS tree.
  - **Fused Q4_K gate_up** (492 ms): incumbent wins every comparable sweep, the
    Qwen35 dual forms are slower and inexact, and the one-output-column form is
    3.2x slower. Input reuse is worth more here than anywhere else because the
    half width is 704 against a 2816 input.
  - **Dense Q8_0** (605 ms, the largest single item): the tiled owner's design is
    capped at 64 accumulators per thread by a ``static_assert``, so ``16x4`` *is*
    the ceiling and all three ladder points (8x2, 8x4, 16x4) were already
    measured into the selection thresholds. The ``rowtile`` family is a rows<=8
    decode domain, and every faster route for this shape (``wmma_prefill``,
    ``mmq128``, ``iu8_wmma``) changes arithmetic.

  What is left on the exact path is two specific jobs, both needing new code
  rather than a selection change: the two Q5_K/Q8_0 layers (205 ms, 9.4%) need a
  grouped owner or an adapter for the existing ``linear``-layer row4 GEMV, and
  attention (295 ms, 13.5%) is bandwidth-shaped and wants several query rows per
  CTA sharing one K/V pass, which is bit-identical because each row keeps its own
  logit tree and its own pass-3 order.

  **Iteration 71: the same sweep on the fused `gate_up`, and why it does not
  transfer.** Iteration 70's win came from a Qwen35-era owner the probe never
  tried, so the fused Q4_K `gate_up` -- the second-largest item -- was swept the
  same way at its real geometry (8192 compact rows, 128 experts, in 2816, half
  width 704, fused width 1408), with the same bit-equality check:

  | owner | ms | bits |
  | --- | ---: | --- |
  | ``rowbatch8_out4_amortized`` (incumbent) | 38.25 | ref |
  | ``rowbatch8`` (one output column) | 124.22 | identical |
  | ``pair2``, ``expertgrid64``, ``expertgrid64_bundle`` | -- | rejected on ABI |

  The incumbent wins here, and by a wide margin over the one-output-column form:
  the fused tensor's half width is 704 against an input of 2816, so the amortized
  owner's input reuse is worth far more than it is on the down projection, whose
  input is only 704 wide. The Qwen35-era dual forms are *not* drop-in: all seven
  ``moe_linear`` dual variants have wrappers, but the pair2/expertgrid64/bundle
  family rejects the fused-stride keywords the Gemma probe passes
  (``output_row_stride``, ``expert_stride_rows``), so they take a different
  calling convention rather than being a schedule flag. Reaching them needs an
  adapter or a signature-preserving wrapper, not a probe reorder, which is why
  this iteration ends without a change.

  **Iteration 70: the Qwen35 paired/folded Q5_1 owner, and the largest exact win of
  the campaign (1.80x on the biggest item).** The campaign goal asks for the
  Qwen35 kernel tuning to be used where it helps. The Gemma probe only ever tried
  the amortized and row-batch variants, so the Qwen35-era ``pair2`` family -- five
  registered Q5_1 owners with the same call ABI -- had never been measured at
  Gemma's geometry. It was measured here, in isolation, on the real shape (8192
  compact rows, 128 experts, in 704, out 2816, nine reps, median of the middle
  five):

  | owner | ms | bits vs incumbent |
  | --- | ---: | --- |
  | ``rowbatch8_out4_amortized`` (incumbent) | 40.16 | ref |
  | ``rowbatch8`` (one output column) | 51.22 | identical |
  | ``rowbatch8_out8`` | 45.02 | identical |
  | ``rowbatch8_out8_expertgrid64`` | 50.56 | identical |
  | ``pair2`` | 28.56 | identical |
  | **``pair2_fold128``** | **21.83** | identical |
  | ``pair2_fold128_pair`` | 25.83 | identical |

  All seven are bit-identical to each other on real Q5_1 blocks, so this is a
  schedule choice with no numerical exposure -- the probe now prefers
  ``pair2_fold128`` ahead of both row-batch variants when the width is inside the
  paired form's 4096-feature fold limit, and falls through for anything else.

  Measured through the model, W7900 lane:

  | | before | after |
  | --- | ---: | ---: |
  | grouped Q5_1 down | 893.8 ms | **497.0 ms** (1.80x) |
  | layer total | 2568 ms | **2177 ms** |
  | prefill | 397.0 tok/s | **468.0 tok/s** |
  | prefill, campaign bench | 2.68 s (382.2) | **2.29 s (448.0)** |

  XTX lane, campaign bench: 2.45 s (417.8) -> **2.03 s (504.1)**. Decode is
  unchanged on both lanes (XTX 43.87 against 43.91).

  Exactness: a teacher-forced capture of the new path against the pre-change
  default capture (``base-pf1024.npz``, 1023 rows, prompt 2048, prefill 1024) is
  **bitwise identical over 1023 x 262144 float32 logits**. 171 GPU tests in the
  ``grouped``/``q5_1`` selection pass. The down owner's share of the layer falls
  from 34.8% to 22.8%.

  **Iteration 69: both lanes, measured, so the llama.cpp gap has a current basis.**
  The campaign's prefill column is the XTX lane and the llama.cpp HIP reference of
  3910 tok/s was taken there, while iterations 64-67 were measured on the W7900
  lane. Both lanes now, same artifact, same 1024/128 shape, three samples with
  one warmup:

  | lane | path | prefill | first token | decode |
  | --- | --- | ---: | ---: | ---: |
  | RX 7900 XTX | default (exact) | 2.45 s (417.8 tok/s) | 2.45 s | 43.91 tok/s |
  | RX 7900 XTX | WMMA arm | 1.39 s (736.9 tok/s) | 1.39 s | 43.87 tok/s |
  | W7900 | default (exact) | 2.68 s (382.2 tok/s) | 2.68 s | 39.49 tok/s |
  | W7900 | WMMA arm | 1.68 s (611.4 tok/s) | 1.68 s | 39.41 tok/s |

  Against the campaign's starting point the exact path is 128.5 -> 417.8 tok/s on
  the XTX lane (3.25x). Against llama.cpp's 3910 tok/s on that lane the exact path
  is 9.4x away and the WMMA arm 5.3x, which is the number the goal's "no problem
  to beat llama.cpp" has to be read against: no exact-path increment reaches it,
  and the arm that halves the distance is the one blocked on the promotion gate.

  **Iteration 68: the tuning-guide review, as a checklist against measured state.**
  The campaign goal asks for ``docs/LESSONS-LEARNED.md`` and
  ``docs/RDNA3-TUNING-GUIDE.md`` to be reviewed before the prefill work. The
  guide had not been cited anywhere in this document, so the relevant rules are
  recorded here against what was actually measured, in the guide's own section
  numbers:

  | guide rule | state in this campaign |
  | --- | --- |
  | 5.2 Amortize weights across rows | Applied. The MoE owners run row-batch 8 across 4 output columns (iterations 4-6), and the dual ``gate_up`` shares its input slab. The rule's own tradeoff -- accumulator state against occupancy -- is what the ``out4``/``out8`` variant ladder exists to explore. |
  | 5.6 Use the cheapest correct reduction | Applied where the chronology allows it. Attention moved to the decode family's intra-warp shuffle tree in iteration 64, bit-identically, for 2.8x. The grouped Q5_1 down cannot: its 256-leaf butterfly is the published association, so the shuffle route is a gate candidate, not an exact edit. |
  | 5.7 Make LDS earn every barrier | **Measured, and it fails the guide's filter.** The down owner stores a 32 KB FP32 accumulator plane per row batch -- exactly the "holds complete FP32 partials while reducing output parallelism" failure the section names -- and spends about 96 LDS operations per 96 FMAs, four times the FMA cost. The section permits LDS for "a reduction that cannot remain wave-local", which is what pins it here, so the design is guide-compliant and the 2.1 TFLOPS is the price of the chronology rather than a scheduling mistake. Iteration 65's rejected wave-sync change is the measurement behind that sentence. |
  | 5.8 Choose WMMA by useful tile occupancy | Measured as an arm, not adopted: 611.4 tok/s against 382.2 on the same lane. Blocked on the promotion gate, iteration 67. |
  | 5.9 Treat integer dot/MMQ as a layout decision | Not applied to this model. ``q8_1_dp4a_grouped`` and the ``mmq128_*`` families exist but are scoped to model-plugin workspace sessions, and for Gemma they would quantize activations to Q8_1 -- a larger arithmetic change than the WMMA arm that is already failing the gate. |
  | 6.4 Prefill attention and linear layers | Attention is done (iteration 64). The linear layers are the item the campaign is still on. |
  | 6.5 MoE execution | Applied: grouped owners replace per-expert sequences, routing stays device-resident, and ``gate_up`` is fused with a strict fallback. The remaining gap is the two quants with no grouped owner at all, iteration 66. |
  | 4.x Evidence discipline | Iteration 67 found the gate harness scoring every row with a single-token forward, which had made a prefill arm look numerically free. The guide's 4.7 "confirm that work volume held constant" is the rule that caught it. |

  The guide's 5.3 "optimize resident waves, not block count" and 5.11 "make cache
  policy follow reuse distance" have not been exercised for this model and are
  the two rules with the least coverage in the table above.

  **Iteration 67: the promotion gate, measured per arm, and the shape of the breach.**
  Two harness facts had to be fixed before any prefill arm could be judged, and the
  first one invalidates the way the arm was screened earlier in this campaign.

  *The teacher-forced chain scores every row with a single-token forward.*
  ``_run_chain`` pushes ``prefill`` ids through one multi-token ``runner.forward``
  and then calls ``runner.forward([ids[position]])`` once per scored row. A gate
  run without ``--prefill`` therefore never executes a multi-token prefill at all:
  the KL it reports is the *decode* path's, and an arm that only changes prefill
  kernels gates at ``kl_max`` exactly 0.0 while saying nothing. That is why
  ``HIPENGINE_GGUF_WMMA_PREFILL=1`` looked numerically free at 63 rows. Every
  prefill claim needs ``--prefill`` large enough to exercise the multi-token
  kernels; ``--prompt 2048 --prefill 1024`` gives 1023 scored rows at keys
  1025..2047, which also clears the decode split's 1024-key entry threshold and
  retires the ``split_not_exercised`` failure.

  *Arms, measured against one frozen default-path capture on the same tree*
  (``/mnt/nvme1/lhl/gemma4-captures/base-pf1024.npz``, 1023 rows, prompt 2048,
  prefill 1024; bars are max 0.05, mean 1e-3, p95 5e-3, p99 2e-2, top-1 0.99):

  | arm | kl_max | kl_mean | kl_p95 | kl_p99 | top-1 flips | verdict |
  | --- | ---: | ---: | ---: | ---: | ---: | --- |
  | dense only (`GGUF_WMMA_PREFILL`) | 0.18002 | 2.29e-4 | 1.38e-5 | -- | 0 | fail on max |
  | MoE only (`GEMMA4_MOE_PREFILL=wmma`) | 0.06087 | 8.58e-5 | 7.70e-6 | -- | 0 | fail on max |
  | both | 0.16108 | 2.11e-4 | 9.70e-6 | 6.32e-5 | 0 | fail on max |

  The MoE-only row reproduces this campaign's earlier 0.060867 to six digits, so
  the two measurements are the same arm and the earlier one was taken with
  ``--prefill``. The dense Q8_0 WMMA route is the *larger* offender, not the MoE
  route: 0.180 against 0.061.

  **The breach is two rows, not a distribution.** Recomputing per-row KL from a
  matched capture of the full arm: the 1023 rows have max 0.16108, mean 2.11e-4,
  p95 9.70e-6, p99 6.32e-5, and **exactly two rows exceed 0.05** (row 602 at
  0.16108, row 864 at 0.05191). The third-worst row is 0.00023 -- a factor of 700
  below the second -- and no row anywhere in the chain flips its top-1 token.
  Every threshold but the absolute maximum passes, and passes by two to three
  orders of magnitude: the mean is 4.7x under its bar, p95 515x, p99 316x, top-1
  100% against a 99% floor. So there is no promotable *subset* of the arm to
  carve out -- each piece fails the same absolute bar on the same kind of
  outlier -- and the decision the arm waits on is a single threshold's
  interpretation, not a missing kernel or a missing measurement.

  **Iteration 66: where the remaining prefill headroom actually lives, and what the
  WMMA arm is worth today.** With attention fixed, the exact path's remaining
  items were re-derived from the 2568 ms census. Every one of them is either
  already at the best schedule its association allows, or has no exact owner at
  all:

  | item | ms | share | exact-path state |
  | --- | ---: | ---: | --- |
  | grouped Q5_1 down | 894 | 34.8% | LDS-issue-bound at the pinned 256-leaf tree; ~2.1 TFLOPS |
  | dense Q8_0 | 581 | 22.6% | already `exact_prefill_tile16x4`; ~11.4 TFLOPS |
  | fused grouped Q4_K `gate_up` | 475 | 18.5% | already the amortized out4 owner; ~7.9 TFLOPS |
  | attention prefill | 331 | 12.9% | fixed in iteration 64 |
  | two Q5_K/Q8_0 layers | 203 | 7.9% | **no grouped owner is registered for either quant** |

  The last row is the sharp one: `moe_linear` has 7 grouped variants for
  `gguf_q4_k` and 12 for `gguf_q5_1`, and **zero for `gguf_q5_k` and
  `gguf_q8_0`**. The two layers whose experts use those quants therefore run the
  selected per-row GEMV, which re-reads each expert's weight once per compact
  row: 0.85 TFLOPS on the Q5_K `gate_up` (76.7 ms for 64.9 GFLOP) against the
  grouped Q4_K owner's 7.9. That is a real, exact win of roughly 150 ms, but it
  is a new kernel family -- one grouped owner per missing quant -- not a routing
  change.

  **The WMMA arm, re-measured on the same lane and the same day:**

  | 1024 prompt / 128 output, W7900 lane | prefill | first token | decode |
  | --- | ---: | ---: | ---: |
  | default (exact) path | 2.68 s (382.2 tok/s) | 2.68 s | 39.49 tok/s |
  | `HIPENGINE_GGUF_WMMA_PREFILL=1` + `HIPENGINE_GEMMA4_MOE_PREFILL=wmma` | **1.68 s (611.4 tok/s)** | 1.68 s | 39.41 tok/s |

  The arm is now worth **1.6x** rather than the 1.3x it was before iteration 64,
  because the attention fix lifts both arms while the arm's own kernels were
  never the binding constraint on that line. Its numerical status is unchanged:
  the earlier 1023-row measurement put `kl_max` at 0.060867 against the absolute
  0.05 bar, and a 63-row screen re-run today returns `kl_max` 0.0 with 0 top-1
  flips but does not exercise the arm at all -- the gate reports
  `split_not_exercised`, and at 63 rows the WMMA routes do not engage, so that
  screen says nothing about the arm either way. Promotion still needs a
  long-chain capture against the arm.

  **Iteration 65: the Q5_1 down owner is LDS-throughput-bound, not barrier-bound;
  the wave-sync change was rejected.** With attention fixed, the grouped Q5_1 down
  is the largest item (894 ms of 2568 ms, 34.8%) and the least efficient: at
  Gemma's real geometry -- 8192 compact rows, in_features 704, out_features 2816
  -- it sustains about 2.1 TFLOPS against the fused Q4_K ``gate_up`` owner's 7.9
  at in_features 2816, out_features 1408. Its inner loop is a 256-lane in-place
  LDS tree per (output column, row) pair: OUT_BATCH x ROW_BATCH = 32 trees per
  row batch, each publishing 256 partials and running eight barrier-separated
  strides. That is about 96 LDS operations per 96 FMAs, and the tree's slots are
  pinned by the bit-exactness contract -- the published association is the
  256-leaf butterfly, so the reduction cannot be re-laid-out into warp shuffles
  without becoming a changed-arithmetic candidate.

  Hypothesis tested: the eight block-wide barriers per tree batch are the cost,
  so strides below 32 -- which only pair lanes inside warp 0, one stride after
  that same wave wrote the slots -- can use a wave-level sync instead.
  ``q5_1_tree_sync`` was added and applied to both tree sites, with the trailing
  barrier kept workgroup-wide to separate the readers from the next publish.

  Result: **rejected.** Q5_1 down 893.8 -> 923.3 ms (+3.3%), layer total 2568 ->
  2606 ms, prefill 397.0 -> 391.3 tok/s. The change was bit-exact -- a
  teacher-forced capture of 63 x 262144 float32 logits against the pre-change
  kernel is byte-identical -- so this is a clean performance verdict, not a
  correctness one. The kernel is limited by LDS issue throughput rather than
  barrier latency: at 32 floats per cycle per CU the tree's ~96 LDS operations
  per thread per row batch cost roughly four times the 96 FMAs it accompanies,
  which is where the 2.1 TFLOPS comes from. Reverted; the file is byte-identical
  to its committed state.

  Consequence for the next attempt: the down owner's headroom needs a different
  reduction layout (32 partials per output reduced inside a wave, which removes
  the LDS traffic entirely), and that changes the association, so it is a
  production-profile candidate judged by the gate rather than an exact edit.

  **Iteration 64: multi-token prefill now runs the decode family's batched-barrier
  kernels, bit-identically.** Iteration 63 identified attention prefill as the
  largest remaining item and assumed it needed a new kernel. It did not: the
  key-class family the decode step already uses is a general ``(tokens, keys)``
  kernel -- ``blockIdx.x`` is ``token * num_heads + head`` and the mask row is
  ``keep_mask + token * keys``, exactly as in the block kernel -- and it already
  carries both of the fixes iteration 63 sketched, the ``kTile`` key batching and
  the ``__shfl_down`` replacement for the intra-warp tree rounds. Only the
  launcher's routing kept it away from prefill.

  ``launch_gemma4_attention_prefill`` now sends ``tokens > 1`` through
  ``launch_gemma4_attention_decode`` first and keeps the block kernel as the
  fallback for geometries the family does not cover, so no new arithmetic was
  written. ``gemma4_attention_shared_bytes`` reports the larger of the two
  routes' requirements, because the key-class kernel holds the logits plus a
  256-lane partial per 256-thread group plus one max slot per warp -- keys + 512
  + 16 floats at its 512-thread block -- which is 32 bytes more than the block
  kernel at a narrow head and less at a wide one.

  W7900 lane, 1024 prompt / 128 output, ``scripts/gemma4_campaign_bench.py``:

  | | before | after |
  | --- | ---: | ---: |
  | prefill | 3.130 s (327.0 tok/s) | **2.68 s (382.2 tok/s)** |
  | first token | 3.00 s | 2.68 s |
  | decode | 39.49 tok/s | 39.49 tok/s |

  ``scripts/gemma4_prefill_census.py`` at 1024 tokens: attention prefill 923 ->
  **331 ms** (2.8x), layer total 3130 -> 2568 ms, so attention falls from 29.5%
  to 12.9% of the layer. Cumulative for the campaign's prefill column on this
  lane, 128.5 -> 382.2 tok/s.

  Exactness: a direct kernel-level A/B -- the same 64x64 causal block, head_dim
  256, 4 heads, run against the stashed pre-change ``.hip`` and against this one
  -- is **65536 of 65536 bf16 output elements identical**, 0 mismatches. That is
  a stronger statement than the teacher-forced gate's ``kl_max``: the two paths
  produce the same bytes, not merely the same distribution. The frozen baseline
  ``.npz`` the gate compares against is not on disk in this tree, so the gate was
  not re-run; ``capture`` on the pre-change tree followed by ``gate`` reproduces
  it if a KL row is wanted. 63 attention tests pass
  (``test_unit_gemma4_attention_geometry``, ``..._routing``, ``..._scratch``,
  ``test_gpu_gemma4_attention_geometry``).

  - **The residual 0.060867 is reduction association, and it is irreducible.**
    Splitting the K accumulation across two independent f32 accumulators moved
    kl_max to 0.081599 - same class, different draw, not an improvement. The
    grouped owner measures exactly 0.0 because it reproduces the strict
    reduction order, which is a property of that kernel, not of the arithmetic.
  - **Everything except the absolute bar passes with large margin.** At kl_max
    0.060867 the arm reports kl_mean 8.58e-05 against 1e-3 (12x), kl_p95 and
    kl_p99 9.89e-05 against 5e-3 and 2e-2 (50x and 200x), and top-1 rate 1.0
    with zero flips on all 1023 scored rows.

  This is the decision already recorded above for the key-slice attention split,
  at a much larger stake: 0.0556 kl_max blocked a 1.1x attention win, 0.0609
  kl_max blocks a 3.4x prefill win. The mechanism is the same - a reordering-class
  arithmetic change that passes every aggregate bar and never moves a decision,
  failing only an absolute `kl_max` order statistic on 1 of 1023 rows of a peaked
  reference. The campaign does not take that decision here. The WMMA arms are
  reachable through `HIPENGINE_GEMMA4_MOE_PREFILL=wmma` (compensated) or
  `wmma_plain` (diagnostic), `auto` keeps the exact routes, and the clearing
  command is a lead ruling that an absolute `kl_max` does not apply to
  reordering-class changes - at which point `wmma` becomes the default.

  Evidence: `/tmp/gemma4-gate-{selected,grouped,wmma,wmma_gate,wmma_down,comp}.json`
  and `/tmp/gemma4-census-{comp,grouped}.json` from this iteration; the census
  family split is in the worklog entry. Kernel-level parity for the compensated
  owners is in `tests/test_gpu_gguf_q4_k_selected_wmma_prefill.py` and
  `tests/test_gpu_qwen4_exp_q5_1_selected.py`, which assert the compensated arm
  is strictly closer to the strict reference than the plain one.

  What this leaves as the next prefill target, at the compensated-WMMA rate:
  attention prefill 959 ms (37.9%), dense Q8_0 625 ms (24.7%), fused gate+up
  577 ms (22.8%), layer-29 MoE 204 ms (8.1%), down 82 ms (3.2%). llama.cpp's
  HIP prefill on the same file is 3910 tok/s, so the MoE routing closes part of
  an 8.5x gap; attention and the dense Q8_0 linears are the larger remaining
  share and neither has been routed through a WMMA owner yet.

## Commands available now

The G0 harness exists: `scripts/gemma4_campaign_bench.py` separates prefill,
decode, first-token and public wall time, verifies public-path token-id
parity, and records a memory row; `scripts/gemma4_llamacpp_reference_bench.py`
runs the same-artifact llama.cpp comparator. To profile either with
`rocprofv3`, precompute the compiler version outside the profiler:

```bash
hipcc --version > /tmp/hipcc_version.txt
env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 \
  HIPENGINE_REQUIRE_CACHED_BUILD=1 \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc_version.txt PYTHONPATH=. \
  rocprofv3 --kernel-trace --output-format json -d <out-dir> -o trace.json -- \
  .venv/bin/python scripts/gemma4_campaign_bench.py \
  --prompt <P> --output <O> --samples 1 --warmup 0
```

Without the version-file pair the profiled first forward spawns
`<compiler> --version` and deadlocks under the profiler — that failure is
already on record in the G1 worklog entry above.

```bash
# From the gemma4 worktree, with the model artifact available:
env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 PYTHONPATH=. \
  .venv/bin/python scripts/gemma4_real_generate.py \
  --artifact /mnt/nvme1/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
  --context 8192 --max-tokens 96 --long-probe-words 700 \
  --out /tmp/gemma4-campaign-functional.jsonl

env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=1 \
  .venv/bin/python -m pytest tests/test_unit_gemma4_*.py tests/test_gpu_gemma4_*.py -q

.venv/bin/python -m pytest tests/test_unit_gemma4_chat.py \
  tests/test_integration_server_api.py tests/test_integration_poolside_v1_reasoning.py \
  tests/test_unit_qwen4exp_chat_protocol.py -q
```

Exact benchmark, correctness and comparator commands are documented in the
G0 baseline artifact and the G1 worklog entry above. Before kernel work, run
the lineage check in OPTIMIZATION;
the review's missing external peer checkout is a known environment issue to
resolve or explicitly scope, not a successful lineage result. Use the Tier-1
allocation probe for capacity questions before full-prompt tests.

## Deliverables and handoff

Each accepted performance result updates `benchmarks/results/`,
`benchmarks/README.md` and `benchmarks/CHANGELOG.md` together, with a new immutable
worklog entry and source commit. Keep raw logs/traces and model weights outside
Git. Inventory temporary flags and rejected routes in `docs/REFACTOR.md` and
run the audit gate; do not raise its budget.

No performance row is added by this planning commit. The review recorded two
unrelated default-suite failures and a published-command documentation failure;
carry their exact evidence forward without claiming the repository is globally
green. Fix newly introduced failures within the candidate that caused them.
