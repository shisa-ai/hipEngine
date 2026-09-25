---
status: current
owns: Gemma 4 26B-A4B gfx1100 single-request optimization plan, measurement contract, candidate sequence, and completion criteria.
---
# Gemma 4 26B-A4B optimization campaign

## Objective and scope

Improve real Gemma 4 26B-A4B GGUF text-inference latency and throughput without
regressing output correctness or request ownership. Start with the working
`gemma4` branch at `1ea13cf8c`, not an isolated kernel harness. The primary lane
is host `epyc`, physical GPU1, RX 7900 XTX, `hip_gfx1100`, the existing
`gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` artifact, greedy decoding, and BF16 KV.

The user requested this campaign after the text-inference review. This document
creates the plan; no tuning run or autonomous loop has started. The first
execution milestone is a validated measurement harness and baseline. Kernel
changes follow measured attribution, not the hypotheses listed below.

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
  `stream_begin_capture`, graph instantiation and `hipGraphLaunch`, and nothing
  calls them, so the machinery exists and is unexercised; (2) fuse the
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
  the read range made it load-bearing (see below); the wide geometry still runs
  this rule. The clean-device pass also puts every other cell 4-10% above its
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
