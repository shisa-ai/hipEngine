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
