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

## Prefill gap analysis: llama.cpp and the Qwen3.6 MoE path — 2026-09-27

This section compares the current Gemma 4 prefill against same-artifact
llama.cpp and against hipEngine's own Qwen3.6-35B-A3B GGUF MoE path, which
has been described as faster than llama.cpp. Every number below comes from
one clean, sequential session. Before each run the script waited until the
target GPU showed 0% use and 0% VRAM, and a `rocm-smi` process sampler ran
for the whole session. No foreign process appeared on either GPU during any
measured run.

Basis: hipEngine `gemma4` at `780d24f29` (production prefill route, the
default since `13532ae2b`); llama.cpp HIP `8cfc315` (`-fa on -b 4096 -ub 1024`,
BF16 KV, prompt caching off); 1024 prompt tokens; three warmups and five
measured samples; medians. "Real text" is 1024 tokens of English prose from
this repository, tokenized by each model's own tokenizer. The Gemma 4 hipEngine
rows use the campaign bench's own prompt. This is a comparator diagnostic, not
a published benchmark row.

### Measured prefill throughput (tok/s, 1024 tokens)

| Engine / model | RX 7900 XTX | W7900 | hipEngine as % of llama.cpp |
| --- | ---: | ---: | ---: |
| llama.cpp, Gemma 4 (real text) | **4124** (rerun 4138) | **3761** | — |
| hipEngine, Gemma 4 (default route) | **fails to load: out of memory** | **1394** | W7900: 37% |
| llama.cpp, Qwen3.6-35B-A3B (real text) | **4101** | **3643** | — |
| hipEngine, Qwen3.6-35B-A3B (real text) | **3210** | **2895** | XTX 78%, W7900 79% |

Prompt content barely matters for hipEngine Qwen on the XTX: a single token
repeated 1024 times gives 3392, random ids 3228, real text 3210. llama.cpp's
repeated and random rows vary more from run to run on these short (~0.25 s)
requests, so only its real-text rows are used.

Four conclusions follow.

1. **hipEngine's Qwen prefill is not faster than llama.cpp today.** It
   reaches 78–79% of llama.cpp on both GPUs. The "faster" claim compares
   against `benchmarks/HISTORY.md`'s llama.cpp HIP row from May 2026: an older
   build, 512 tokens, 2436 tok/s on the W7900. Current llama.cpp does 3643 on
   the same W7900 at 1024 tokens.
2. **llama.cpp runs both models at the same speed** (4124 vs 4101 on the XTX).
   hipEngine is at 79% on Qwen and 37% on Gemma 4, so most of the Gemma gap
   comes from Gemma-specific paths rather than from the engine as a whole.
3. **The default route no longer fits the 24 GB XTX.** It peaked at 32.35 GB
   on the W7900. The cause is under "Defects found" below.
4. The earlier headline gap (1393 vs 3910) mixed GPUs. On the same GPU it is
   3761 vs 1394, a factor of 2.7.

### Where each prefill spends its time

These are kernel-trace (`rocprofv3 --kernel-trace`) sums in milliseconds per
1024-token prefill.

- Both llama.cpp traces and the hipEngine Qwen trace are from the XTX.
- The hipEngine Gemma 4 trace is from the W7900, because Gemma 4 does not load
  on the XTX. The W7900 runs about 9% slower on this workload, measured with
  llama.cpp (3761 vs 4124).
- The Gemma 4 bench trace contains two prefills, the instrumented one and the
  public `generate` parity call, so its per-layer kernels show n = 120
  (30 layers × 2 blocks × 2 prefills). All values are divided by 2.
- The trace does not show a 4 × 256-token chunking.

| Bucket | llama.cpp Gemma | hipEngine Gemma (W7900) | llama.cpp Qwen | hipEngine Qwen |
| --- | ---: | ---: | ---: | ---: |
| Attention (softmax) | 24.9 | **216.0** | 5.8 | 11.4 (AOTriton) |
| Linear attention (GDN), Qwen only | — | — | 34.4 | 44.2 |
| MoE expert matmuls | 89.7 | 179.1 (layers 0–28) + **74.0 (layer 29)** | 78.7 | 138.0 |
| MoE routing, gather, combine | 10.1 | 35.8 + 6.2 act. pack | 18.0 | 6.2 |
| Dense projections (Q8_0) | 61.5 | 162.2 | 49.0 | 90.5 |
| Router logits / small BLAS | 5.8 | 15.1 | 9.1 | 16.8 |
| Norms, RoPE, element-wise, other | 27.1 | 20.2 | 31.5 | 5.9 |
| **Total kernel time** | **219** | **709** | **227** | **313** |
| Measured wall time | 248 | 735 | 250 | 319 |

Matmul throughput on the MoE experts, using 2·K·N FLOPs per routed row:

| | Gemma 4 | Qwen3.6 |
| --- | ---: | ---: |
| hipEngine | ~16 TFLOP/s (W7900; ~17 at XTX clocks) | ~15 TFLOP/s |
| llama.cpp | ~33 TFLOP/s | ~26 TFLOP/s |

hipEngine's MoE kernels run at the same efficiency on both models, about half
of llama.cpp's. **The Qwen path is not closer to llama.cpp because of better
MoE kernels.** It is closer because its attention is cheap: 30 of its 40
layers are GDN linear attention, and its 10 full-attention layers use AOTriton
flash attention. Gemma 4 runs softmax attention on all 30 layers through a
decode-class kernel.

### What llama.cpp does differently

| | llama.cpp HIP on gfx1100 | hipEngine Gemma 4 (default) |
| --- | --- | --- |
| Matmul arithmetic | Activations are quantized to Q8_1 once per matmul, and every weight matmul runs int8 × int8 on WMMA. `ggml_cuda_should_use_mmq` is true on RDNA3 for Q4_K, Q5_1 and Q8_0. | MoE gate_up uses int8 MMQ (32-row tiles). The MoE down uses BF16 WMMA. Dense projections use BF16 WMMA. |
| Tiles | Tuned per quant type (`mmq-config-rdna3.cuh`): up to 128 weight rows × 128 tokens, 256 threads, stream-k. | 32-row int8 tiles on gate_up, 16-row WMMA elsewhere. |
| MoE launch | One `mm_ids_helper` sort, then one MMQ launch per projection across all experts. | Compact scheduler plus `qwen35_moe_group_compact_active_kernel` (23.8 ms) and a packed-hidden gather (6.9 ms). |
| Odd-quant layer | Layer 29 (Q5_K gate_up, Q8_0 down) goes through the same MMQ kernels as every other layer. | Layer 29 falls back to `gguf_q4_k_selected_dual_grouped_rowbatch` and `gguf_k_selected_prefill_out_kernel`: **74 ms, about 12× a normal layer.** This is the kernel iteration 146 left unidentified. |
| Attention | Flash attention. Head_dim-256 sliding-window layers use `flash_attn_ext_f16` (WMMA); head_dim-512 global layers use `flash_attn_tile`. 24.9 ms total. | `gemma4_attention_decode_class_kernel` runs for prefill (60 launches per prefill). 7.0 ms per sliding-window layer, 8.5 ms per global layer; 216 ms total. |
| Block | ubatch 1024. | 512-token blocks. The block-1024 measurement came out as a wash (commit `0b2ce993a`). |
| Weight residency | GGUF blocks read in place, no duplicate. | Fused `ffn_gate_up_exps` kept **plus** split gate and up copies: 32.35 GB peak. |

### Qwen3.6 MoE: hipEngine against llama.cpp

| Area | hipEngine Qwen | llama.cpp Qwen | Result |
| --- | --- | --- | --- |
| Element-wise, norms, residual | Fused composites (add+RMSNorm, SiLU-mul dual output, shared-gate combine with residual): 5.9 ms | Separate kernels: 31.5 ms | **hipEngine ahead by about 25 ms** |
| MoE routing and combine | Count / prefix / scatter-gather / tile-map scheduler plus fused weighted sum: 6.2 ms | `mm_ids_helper` plus `moe_weighted_reduction`: 18.0 ms | **hipEngine ahead by about 12 ms** |
| MoE expert matmuls | FP16 WMMA on byte-lossless T16 repacked weights: one wave, 16×32 tiles, per-lane dequantization, no LDS: 138 ms | int8 MMQ with 128-row tiles: 79 ms | llama.cpp 1.75× faster |
| Dense Q8_0 projections | `gguf_q8_0_t16_prefill_wmma_nwave` (BF16 WMMA): 90.5 ms | int8 MMQ: 49 ms | llama.cpp 1.8× faster |
| GDN linear attention | 44 ms | 34 ms | llama.cpp 1.3× faster |
| Full attention | AOTriton `attn_fwd`: 11.4 ms | flash attention: 5.8 ms | llama.cpp 2× faster, but both are small |

hipEngine wins on glue and fusion, about 37 ms of launches and element-wise
work that llama.cpp spends. It loses on every matmul, because llama.cpp runs
int8 × int8 with large tiles while hipEngine runs FP16/BF16 WMMA with small
tiles.

### What the Qwen path does that Gemma 4 should adopt

Savings are estimated from the traces above, in W7900 milliseconds out of a
709 ms prefill. None of them has been measured as a change.

| Qwen practice | Gemma 4 today | Adopt as | Est. saving |
| --- | --- | --- | ---: |
| Flash attention for prefill: AOTriton at ≥512 tokens, about 1.1 ms per head_dim-256 layer at 1024 tokens | Decode-class kernel, 7.0 ms per head_dim-256 layer | Route the 25 sliding-window layers through AOTriton. Pass scale 1.0, since Gemma folds the softmax scale into the query norm. Through 1024 tokens the 1024-token window never binds; above that a windowed mask is needed. The 5 head_dim-512 global layers need a tiled kernel, following llama.cpp's `flash_attn_tile<512>`. | ~145 (sliding-window) + up to ~35 (global) |
| Every quant in the artifact has a fast owner | Layer 29 (Q5_K gate_up, Q8_0 down) falls to scalar fallbacks | Register or route layer 29 to the compact WMMA owners that already exist for Q5_K and a grouped Q8_0 down | ~65 |
| Replacement layouts, no duplicate weights (T16 replaces raw) | Split gate/up is additive to the fused tensor (+~8.6 GB) | Replace the fused allocation with the split one. The dual WMMA owner already registers a two-tensor form. | Memory: restores the XTX |
| Cheap MoE scheduler (count / prefix / scatter-gather): ~3 ms | `compact_active` + gather: ~31 ms | Reuse the Qwen scheduler kernels | ~25 |
| Fused norm, residual and gate composites | Already close: 20 ms vs llama.cpp's 27 | — | — |

Two items are not transferable from Qwen, because Qwen does not solve them
either:

- **MoE matmul efficiency** is about 16 TFLOP/s on both models, against
  llama.cpp's 26–33.
- **Dense matmuls** are 1.8–2.6× behind llama.cpp on both models.

For both, the reference is llama.cpp's int8 MMQ (Q8_1 activations, 128-row
tiles, one launch across experts), not the Qwen path. Iteration 150 refuted
the dense int8 route, but only for hipEngine's *guarded three-plane* d4x3
MMQ: three int8 WMMA planes per tile to reproduce exact results. It does not
test a single-plane MMQ of llama.cpp's shape, which runs at an estimated
~55 TFLOP/s here (dense FLOPs computed from the tensor shapes).

### Defects found

- **XTX capacity regression.** `gemma4_gguf_device.py` keeps the raw fused
  `ffn_gate_up_exps` and uploads split `gate` and `up` copies. Since
  `13532ae2b` made the int8 route the default, every load pays this. The
  default route peaks at 32.35 GB and fails on the RX 7900 XTX, the campaign's
  primary GPU, with `HIP error 2: out of memory` while materializing a layer's
  expert tensors. Reproduce with `ROCR_VISIBLE_DEVICES=1 scripts/gemma4_campaign_bench.py
  --prompt 1024 --output 8`.
- **The attention analysis drafted as iteration 151 (per-key barriers in
  `gemma4_attention_prefill_kernel`) targets a kernel that does not run
  during prefill.** The trace shows `gemma4_attention_decode_class_kernel<…,1,2>`
  (sliding-window) and `<…,2,2>` (global) at 60 launches per prefill.
  `gemma4_attention_prefill_kernel` does not appear. The fix direction (flash /
  online softmax, tiled over keys) still stands, but the barrier analysis
  should be redone against the decode-class kernel.
- **The published Qwen comparator is stale.** See conclusion 1 above.

### Recommended order

1. Fix the additive gate/up split: replace, don't duplicate. This is a
   capacity defect on the primary GPU and a prerequisite for every XTX
   measurement.
2. Flash attention for the 25 sliding-window layers (AOTriton first, since it
   already ships for Qwen); then a tiled head_dim-512 kernel for the 5 global
   layers.
3. Route layer 29 through the existing fast owners.
4. Replace the MoE `compact_active` + gather scheduler with the Qwen scheduler.
5. Longer term: single-plane int8 MMQ with llama.cpp-sized tiles for dense and
   MoE. This benefits Qwen equally.

If items 1–4 hit their estimates, the prefill falls from 709 ms to about
440 ms, roughly 2300 tok/s on the W7900. Adding item 5 is what closes the rest
of the gap to llama.cpp's 3761. These are projections from the trace, not
measurements.

Commands (hipEngine worktree `gemma4`; each preceded by an idle check on the
target GPU):

```bash
# llama.cpp comparator, explicit token ids, same flags as the campaign reference
llama-server -m <gguf> -ngl 99 -fa on -ctk bf16 -ctv bf16 -c 8192 -np 1 \
  -b 4096 -ub 1024 --no-cache-prompt --fit off   # then POST /completion {prompt: [ids], n_predict: 1}
# hipEngine Qwen: scripts/qwen35_gguf_bench.py with --public-ar-profile (the shipped
# WMMA prefill selectors; without it WMMA prefill resolves off), --persistent-session,
# --prompt-length 1024, --warmup-runs 3 --measured-runs 5, and explicit prompt ids
# hipEngine Gemma 4: scripts/gemma4_campaign_bench.py --prompt 1024 --output 8 --samples 5 --warmup 2
# traces: rocprofv3 --kernel-trace --output-format csv (hipEngine Qwen additionally
# uses --selected-regions with --rocprof-selected-region prefill)
```

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

  **Iteration 106: Gemma prefill attention has no query-row blocking, while the
  repo's Laguna path has a tiled variant system it never got.**

  ``attention_prefill`` is 60 calls / 201 ms, 12-15% of layer time, and had never
  been examined. Both prefill kernels launch::

      dim3(num_q_heads, rows)      // one CTA per (query head, query row)

  so each CTA walks its full causal key range for a **single** query row, and K/V
  is loaded once per query row with no reuse across rows. Query-row tiling is
  exactly what flash attention exists to provide and it is absent here. At 1024
  tokens the sliding window does not bite (window >= sequence), so all 58 layers
  behave as global attention and this applies to every one of them.

  **The repo already has the tiled version, for a different model.** In
  ``laguna_flash_attention_prefill.hip``::

      __global__ void laguna_flash_attention_prefill_f16_wmma_whole_kernel(...)
        const int query_row = query_tile * QUERY_ROWS + query_local;

  and a mature variant system around it in ``kernels/hip_gfx1100/__init__.py``::

      LAGUNA_SWA_PREFILL_VARIANT = "swa_context_rows_qrow4_m128_c256_exact_spans"

  -- query-row-4, M128, C256 tiles -- with a documented crossover ("selects qrow4
  only for complete M128 tiles at position 256+") and qrow2/qrow4/local128/wave32
  variants registered as explicit rollbacks. Gemma binds
  ``hipengine_gemma4_attention_prefill_bf16``, the untiled kernel, and never got
  any of it.

  **It is a port, not a switch.** Gemma's kernel takes a dense ``(tokens, keys)``
  uint8 keep-mask; the Laguna variants take KVLiveSpans (``base_offsets``,
  ``live_counts``). Those are different attention ABIs, and PLAN.md names
  KVLiveSpans as the intended one with dense policies filling it uniformly -- so
  Gemma's keep-mask path is drift from the stated ABI as well as a slower kernel.

  **Do not price this at the tile factor.** Four traffic-based predictions have
  failed in this session, one of them built on a genuine code-level re-read that
  turned out to be worth 4% instead of 2x. The structural fact here is certain --
  there is no query-row blocking, and the launcher says so in one line. Whether
  that costs the time it appears to is not established, and the honest prior after
  four failures is that the traffic model is unreliable for these kernels. This
  entry records a lead and a port, not a projected win.

  **The pattern is now three for three.** The MoE grouped owner has a faster WMMA
  sibling held by a numeric gate; the down kernel lacks the nest its sibling
  gate_up already has; Gemma attention lacks the tiling the Laguna path already
  has. In each case a better implementation exists in this tree or in a sibling
  model's path and Gemma's production route does not use it. Before writing more
  kernels, the question worth answering is why -- correctness gate, interface gap,
  or simply never wired -- because that determines whether the fix is a port or a
  promotion.

  Evidence: ``laguna_kv_attention.hip:13690`` and ``:18480`` (launch geometry);
  ``gemma4_attention.py:39`` (bound symbol); ``kernels/hip_gfx1100/__init__.py:16``
  (Laguna variant system).

  **Iteration 107: the dense line's owner was named, and it was missing a
  production default. +18.6% prefill.**

  Iteration 38 left the dense family -- 428 ms, 25% of prefill, 410 calls at
  10-17 GB/s -- without an owner, because it does not resolve through
  ``registry.resolve``. It does not, because it is not a registry kernel at all:
  ``gemma4_project`` (gemma4_layer.py:117) funnels every dense projection into
  ``hipengine.runtime.gguf_linear.launch_gguf_linear``.

  That function's own docstring names the lever::

      When rows > 1 and the raw-layout quant has a WMMA prefill kernel
      registered (currently gguf_q8_0 and raw gguf_q4_k), the dispatch rewrites
      to the wmma_prefill_* family if any of these is true:
        * use_wmma_prefill=True is passed explicitly,
        * a runner has called set_wmma_prefill_enabled with True,
        * the env var HIPENGINE_GGUF_WMMA_PREFILL is set.
      Otherwise aligned raw-Q8 BF16 projections use the exact pack8/row-tiled
      schedule.

  ``gemma4_project`` passed none of them. ``docs/ENVS.md`` then says the quiet
  part out loud::

      HIPENGINE_GGUF_WMMA_PREFILL | false | Low-level performance selector
      ... The public generator passes use_wmma_prefill=True.

  and both shipping GGUF call sites do exactly that, literally --
  ``generation/qwen35_gguf.py:3639`` and ``runtime/qwen35_gguf_nextn.py:446``.
  So the env var is only the low-level session default; **the shipped path
  already runs WMMA prefill, and Gemma was not on it.** This is not a gated
  candidate arm. It is the production path not doing what production does.

  **Bracketed A/B** (default / wmma / default, so drift cannot masquerade):

  ==================  ==========  ==========  =======
  dense shape         default     wmma        speedup
  ==================  ==========  ==========  =======
  r=512 k=2816 n=2112   888.9 us    281.5 us    3.16x
  r=512 k=2816 n=4096  1467.9 us    522.3 us    2.81x
  r=512 k=4096 n=2816  1107.3 us    555.5 us    1.99x
  ==================  ==========  ==========  =======

  The unaffected MoE gate_up read 7744 / 7743 / 7770 us across the three runs and
  the two default dense runs agreed to 0.2% (888.9 vs 890.7), so the 2-3x is the
  kernel and not the 12% thermal swing that invalidated earlier attempts.

  **End-to-end**: ``prefill_tps`` 598.36 -> **709.38** (+18.6%), ``prefill_s``
  1.7114 -> 1.4435, ``layer_total`` 1699.9 -> 1432.4 ms.

  **Teacher-forced gate** (``scripts/gemma4_teacher_forced_gate.py``, the
  evaluator that gates changed-arithmetic candidates against frozen incumbent
  logits with the production KL/top-1 limits from EXECUTION-PROFILES.md):

      kl_mean 0.0   kl_p95 0.0   kl_p99 0.0   kl_max 0.0
      top1_rate 1.0   top1_flips 0     rows 1023   vocab 262144

  **Every percentile is exactly zero.** The WMMA prefill family reproduces the
  incumbent logits bit-for-bit on the frozen chain, so this is not a
  changed-arithmetic promotion and the KL/top-1 limits are not the relevant
  question. The gate nevertheless reports ``passed: false`` with
  ``failed: ['split_not_exercised']`` and ``split_launches: 0``: the prompt-1024
  chain never crosses the split-key-range threshold, and the gate requires the
  chain to exercise that route before it will qualify anything. That is a
  chain-selection precondition, not a numerical result -- and it is owed a
  long-context re-run, which is what the remaining verification for this entry
  is.

  **Kept on the default path.** It matches the two shipping call sites, it is
  bit-identical where measured, and the alternative -- leaving Gemma on a
  fallback the shipped generator does not use -- is the defect. One consequence
  worth recording: because an explicit kwarg outranks the session toggle and the
  env var (``_resolve_use_wmma_prefill``), Gemma's dense path now follows the
  shipping pattern in also being no longer switchable from the environment. The
  Qwen call sites have that same property; it is the established pattern, not a
  new lever that was removed.

  **This is the fourth time a better implementation already existed in this
  tree and Gemma's route did not use it** -- after the MoE grouped owner's WMMA
  sibling, the down kernel's missing nest, and the attention tiling the Laguna
  path has. The first three were blocked by gates or interfaces. This one was
  blocked by nothing at all: it was a missing keyword argument, worth 18.6%.

  Evidence: ``gemma4_layer.py:117`` (owner, now passing the kwarg);
  ``gguf_linear.py:3055`` (dispatch contract), ``:1829`` (precedence);
  ``ENVS.md:272``; ``qwen35_gguf.py:3639``, ``qwen35_gguf_nextn.py:446``.

  **Iteration 108: audit of the iteration-47 defect class. It has exactly one
  instance, and two of my earlier leads were wrong.**

  Iteration 47 gained 18.6% from a missing keyword argument, so the obvious next
  move was to enumerate every opt-in the shipping path passes and Gemma does not,
  rather than rediscovering them one at a time. Result: **the defect class has
  exactly one instance, and it is the one already fixed.**

  The shipping GGUF path passes *two* opt-ins, at twelve call sites
  (``qwen35_gguf.py``:1688, 1758, 3197, 3640, 3805, 4139, 4146, 4348, 4355, 9164,
  9202; ``laguna_gguf_runner.py``:6070)::

      use_wmma_prefill=True, use_gemv_decode=True

  ``gemma4_project`` now passes the first (iteration 107). The only other Gemma
  call site that accepts them is ``gemma4_project_expert``, which reaches
  ``launch_gguf_linear_raw_ptr`` -- whose signature does take ``use_wmma_prefill``
  (gguf_linear.py:4634) -- and passes neither. That path serves 2 of 410 prefill
  calls, so it is a decode-relevant gap rather than a prefill one, and it is
  recorded here rather than changed blind.

  **Lead rejected: F16 activation staging.** ENVS.md:281 marks
  ``HIPENGINE_GGUF_PREFILL_F16_STAGING`` as production-profile-enabled and
  strict-disabled, and its row range (17..1024) covers the 512-row dense shapes,
  so it looked like a second instance. It is not: the eligible quant set is
  ``['gguf_q4_k_t16_v1', 'gguf_q5_k_t16_v1']`` and Gemma's dense projections are
  ``gguf_q8_0``. The bracketed A/B agrees -- dense 284.6 / 285.4 / 286.7 us across
  default/staging/default, flat. Ruled out in one run.

  **Lead reclassified: the MoE WMMA route is not a parity gap.** Iteration 86
  found a WMMA arm worth +11.6% "held by a numeric gate", which in iteration 47's
  light looked like it might be another missing call-site opt-in. It is not::

      def _prefill_route_flags(mode):
          """... ``auto``, ``grouped`` and ``selected`` keep the exact routes,
          so the WMMA owners are not probed at all."""
          if mode == "wmma":       return True, True
          if mode == "wmma_plain": return True, False
          return False, False

  ``auto`` *is* the production policy and it deliberately keeps the exact routes;
  ``wmma`` and ``wmma_plain`` are explicit probes, compensated and uncompensated
  respectively. So the MoE WMMA route is a genuine changed-arithmetic candidate
  and its numeric gate is the correct instrument, not an oversight. It needs the
  execution-profile gate, not a call-site fix.

  **Lead withdrawn: attention tiling is not a port.** Iteration 106 recorded that
  Gemma's prefill attention has no query-row blocking while the Laguna path has
  ``swa_context_rows_qrow4_m128_c256_exact_spans``, and suggested porting it. The
  Gemma attention module's own docstring forecloses that::

      Gemma 4 attention is *ungated*. Every Qwen3.5 prefill variant reads an
      attention gate and multiplies by sigmoid(gate); Laguna's ungated kernel
      hard-codes Laguna's head geometry. Neither can serve Gemma 4, so this
      family exists.

  The Laguna tiled variants either read an attention gate Gemma does not have or
  hard-code a head geometry Gemma does not share. Gemma exposes exactly one
  prefill symbol (``hipengine_gemma4_attention_prefill_bf16``) and no gemma4
  attention registration exists, so there is no variant to select and no missing
  selector -- the gap is real but closing it is **new kernel work**, not a port
  from a sibling model. Iteration 106's "it is a port, not a switch" was wrong in
  the direction of optimism.

  **Current breakdown** (layer_total 1442.9 ms, prefill_tps 704-709)::

      moe_grouped:gguf_q5_1      478.0 ms   33.1%
      moe_grouped_dual:gguf_q4_k 457.1 ms   31.6%
      attention_prefill          203.9 ms   14.1%
      dense:gguf_q8_0            157.8 ms   10.9%   (was 428 ms)
      moe_selected:gguf_q8_0      47.1 ms    3.3%
      unattributed                76.7 ms    5.3%

  The MoE is now 64.7% of layer time and the dense line fell by 63%. The two MoE
  families run at ~8 ms per call and the q4_k family already reads 320 GB/s, so
  unlike the dense line it is not obviously leaving bandwidth on the table.

  Net effect of the audit: the cheap class of win is exhausted. What remains is
  the MoE WMMA promotion behind its numeric gate, or new attention kernel work --
  both of which are real engineering, not a missing argument.

  Evidence: ``qwen35_gguf.py:3640`` and 11 other call sites;
  ``gemma4_experts.py:450`` (raw_ptr, no opt-ins), ``:573``/``:582``
  (mode and flags); ``gemma4_attention.py:1`` (ungated family docstring);
  ``gguf_linear.py:219`` (staging quants).

  **Iteration 109: correction -- the MoE is at 3% of peak bandwidth, not 320
  GB/s. It is the largest opportunity in this campaign.**

  Iteration 108 recorded "the q4_k family already reads 320 GB/s, so unlike the
  dense line it is not obviously leaving bandwidth on the table." **That is
  wrong**, and it is wrong in the direction that closes off the biggest target in
  the campaign. The shape census header is ``MB/call    GB/s``; ``320.078`` is
  **MB per call**, and the achieved bandwidth is the next column, **40.6 GB/s**.
  The census prints the device peak directly under the header: 864 GB/s.

  Measured, from ``scripts/gemma4_prefill_shape_census.py``::

      family                              calls  tot_ms  MB/call   GB/s   TF/s
      moe_grouped:gguf_q5_1                  58   478.8  219.152   26.5    2.0
      moe_grouped_dual:gguf_q4_k             58   456.9  320.078   40.6    4.1
      dense:gguf_q8_0 r=512 k=2816 n=2112   120    34.3   11.365   39.8   21.3

  So the two MoE families together move about 31 GB per prefill pass in 935 ms
  -- **~33 GB/s, 3.8% of the 864 GB/s peak** -- and sustain 2.0-4.1 TF/s, single
  digits of the compute peak. The whole prefill pass runs at roughly 28 GB/s,
  about 3% of peak.

  **The timing is not host-gap inflation.** ``layer_total`` (1443 ms) matches the
  end-to-end ``prefill_s`` (1.448 s) for the same 1024-token pass, so the kernels
  really do occupy the wall time; if most of it were launch idle, ``layer_total``
  would be a small fraction of the prefill rather than all of it. The 10-13%
  GPU-busy figure from iteration 105 was a different measurement -- a repeated
  A/B harness that idles between runs -- and does not apply to a single
  contiguous prefill.

  **llama.cpp proves the target is reachable on this hardware.** At 3910 tok/s a
  1024-token prefill takes 262 ms; moving comparable traffic in that window is
  well over 100 GB/s. The gap is a kernel-structure gap, not a hardware limit.

  **Structural clue, from the owner names.** The grouped owners are::

      qwen4_exp_q5_1_selected_grouped_prefill_compact_rowbatch8_out4_amortized_bf16_bf16_out
      hipengine_gguf_q4_k_selected_dual_grouped_rowbatch8_out4_amortized_bundle_bf16_bf16_out

  -- ``rowbatch8`` x ``out4`` tiles, carrying a ``_selected_`` prefix from the
  decode-side selected-expert family, being used for bulk prefill. The gates
  around them are ``_GROUPED_PREFILL_MIN_LANES_PER_EXPERT = 4`` and
  ``_WMMA_PREFILL_MIN_LANES_PER_EXPERT = 16``, and
  ``_GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES = 4096`` means the amortized variant
  is preferred for both the k=704 and k=2816 MoE shapes.

  **This reopens what iteration 108 closed.** The WMMA route that iteration 86
  measured at +11.6% is a real but small step against a ~25x gap, so the
  productive question is not which registered variant to pick but why a 4096-row
  prefill is running an 8-row x 4-output tile. That is the next thing to
  establish, and unlike the attention tiling it is not new kernel work if an
  existing owner has the right tile shape.

  Evidence: ``scripts/gemma4_prefill_shape_census.py`` header and rows;
  ``gemma4_experts.py:502``-``530`` (variant names, gates, tile constants);
  ``gemma4_experts.py:793`` (grouped gate), ``:757`` (dual launcher).

  **Iteration 110: Gemma is already on the preferred MoE variant, and the
  intensity says the limit is not FMA throughput.**

  Iteration 109 left the question "why is a 4096-row prefill running an 8-row x
  4-output tile", on the theory that a better-tiled owner might already exist.
  It does not -- the selected variant is already the preferred one::

      _GROUPED_DUAL_AMORTIZED_PREFILL_VARIANT = (
          "selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out")
      # The same owner with the loop nest swapped so one CTA covers four output
      # columns and reuses the input row batch across them. The association of
      # every output is unchanged -- same thread-to-column map, same 128-thread
      # tree -- so this is the bit-identical route and is preferred whenever the
      # width fits.

  ``_GROUPED_DUAL_AMORTIZED_MAX_IN_FEATURES = 4096`` and the Gemma MoE shapes
  are k=704 and k=2816, so both take it. The row-batch owner it displaces is the
  same kernel without the reuse; there is no wider-tile grouped owner waiting.
  The only other owner is ``_WMMA_PREFILL_VARIANT =
  "selected_grouped_wmma_prefill_compact_bf16_bf16_out"``, documented as reading
  "each weight block once per 16-row tile" -- the prefill-shaped tile -- which
  iteration 86 already measured at +11.6%.

  **So the tile shape is not the bottleneck, and the arithmetic says why.** For
  the q4_k dual owner: 4.1 TF/s sustained over 7.88 ms per call is 32.3 GFLOP
  per call, against 320.078 MB moved -- an arithmetic intensity of **101
  FLOP/byte**. At the device's 864 GB/s that intensity would demand 87 TF/s,
  which is above the W7900's bf16 peak. **The kernel therefore cannot be
  bandwidth-bound.** It is on the compute side, sustaining 4.1 TF/s, about 7% of
  compute peak. (The q5_1 owner is the same story at 2.0 TF/s and 26.5 GB/s.)

  **The contradiction is the finding.** A compute-side kernel sitting at ~7% of
  peak, whose tensor-core variant buys only 11.6%, is not limited by FMA
  throughput -- if it were, moving to WMMA would buy a multiple, not a tenth.
  What is left is instruction issue, LDS traffic, or memory latency, which is
  exactly the regime ``docs/RDNA3-TUNING-GUIDE.md`` exists to describe and which
  the campaign has so far never consulted. That is the next read, and it is
  cheap: it is a document, not a kernel.

  Worth stating plainly, because it changes what to do next: **the MoE has
  consumed five iterations of variant-hunting and the answer is that the
  selection was already correct.** The remaining lever is inside the owner, not
  above it. Before writing anything, the tuning guide should be read for what it
  says about 128-thread trees and LDS pressure in exactly this shape.

  Evidence: ``gemma4_experts.py:502``-``530`` (variant strings and their
  comments), ``:793`` (grouped gate); iteration 86 (+11.6% WMMA);
  ``scripts/gemma4_prefill_shape_census.py`` (MB/call and GB/s columns).

  **Iteration 111: the tuning guide the objective named contains the diagnosis,
  and it explains both MoE observations at once.**

  Iteration 50 was left with a contradiction: the MoE is compute-side at ~7% of
  peak (101 FLOP/byte, so it cannot be bandwidth-bound), yet its tensor-core
  variant buys only 11.6% -- which rules out FMA throughput. The objective asked
  for ``docs/RDNA3-TUNING-GUIDE.md`` to be read and this campaign had never read
  it. It answers the question in §5.3::

      1-2 waves per SIMD is critically undersubscribed and can drop effective
      bandwidth to 30-40%. ... A low-row WMMA kernel can launch hundreds of
      blocks and remain latency-bound if roughly 200-250 VGPRs per thread permit
      only a few waves per issue slot. In that case, reducing the accumulator
      tile can outperform adding more blocks.

  That is the shape of both facts. A kernel with many blocks, a small accumulator
  tile, and high per-thread state stays latency-bound no matter how the grid is
  arranged -- and moving it to WMMA does not help, because the limit was never
  the arithmetic units. §5.2 gives the mechanism: "more row/column accumulators
  increase VGPR allocation; lower occupancy can reduce outstanding memory
  requests; collapsing the M grid can also remove useful N-direction
  parallelism." §3.5 already warns that "the smaller grid removed the
  thread-level parallelism that was hiding memory latency", measured as 59%
  slower on a grid reshape.

  §3.3 also classifies the regime, consistent with the measured intensity:
  prefill is where "compute throughput and K-loop scheduling matter more" and
  where "a separate build/profile and dispatch policy from decode" is required.
  The MoE prefill owner carries a ``_selected_`` prefix from the decode-side
  selected-expert family, which is exactly the decode-shaped provenance §3.3
  warns about.

  **The hypothesis is therefore occupancy, and it is directly checkable.** §5.3
  states the rule the repository already accepts: "Treat any allocation above
  about 128 VGPRs as worth inspecting", with the ladder 96 VGPRs -> 16 waves,
  192 -> 8, above 256 -> 4-5 and "starts to starve the memory controller". So
  this is not a new threshold being invented for the occasion -- the guide is
  normative for kernel work and supplies a pre-registered decision rule. The
  measurement to make is the MoE owner's VGPR allocation and resident waves per
  SIMD, and the decision follows from the guide's own numbers.

  **Stated honestly**: the guide's 30-40% floor is for a bandwidth-bound decode
  kernel, and the MoE is at 3.8% of peak, so occupancy alone may not account for
  the whole gap to llama.cpp. What the guide does establish is that the observed
  pattern -- many blocks, no WMMA benefit, compute-side classification -- is a
  documented latency-bound signature rather than an unexplained anomaly, and
  that the correct next move is a resource measurement rather than another
  variant or another kernel.

  Evidence: ``docs/RDNA3-TUNING-GUIDE.md`` §3.3, §3.5, §5.2, §5.3.

  **Iteration 112: the occupancy hypothesis is refuted, and the MoE WMMA route
  measures +14.2%.**

  Iteration 111 read ``docs/RDNA3-TUNING-GUIDE.md`` §5.3 and concluded the MoE
  was VGR-starved: many blocks, small accumulator tile, 1-2 waves per SIMD, and
  a tensor-core variant that could not help. The guide supplies a ladder and the
  repository already exposes it as a compile-time knob --
  ``HIPENGINE_GGUF_SELECTED_WMMA_LAUNCH_BOUNDS`` accepts ``{1, 2, 4, 8}`` and
  feeds ``-DHIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS=N``, defaulting to the
  compiled ``__launch_bounds__(32, 2)``. Running it::

      HIPENGINE_GEMMA4_MOE_PREFILL   LAUNCH_BOUNDS   prefill_tps
      auto                           (unset)             708.28
      wmma                           2                   809.21   +14.2%
      wmma                           4                   769.11    +8.6%
      wmma                           8                   683.91    -3.4%

  **The ladder is monotonically worse, and at 8 it is worse than not using WMMA
  at all.** Forcing the compiler to fit more resident blocks -- which is exactly
  what §5.3 recommends for a latency-bound kernel -- costs time at every step.
  So this owner is not VGR-starved in the sense §5.3 describes: the accumulators
  earn their registers, and the right reading of §5.2 here is that collapsing the
  accumulator tile removes reuse rather than relieving occupancy. The hypothesis
  is dead, and it was my own, formed from a normative document and killed by a
  four-point measurement. Recording it as dead is the point of measuring.

  **The positive result is the arm itself.** ``wmma`` mode at its default bound
  measures **809.21 tok/s against 708.28**, +14.2% on the whole prefill, which
  independently reproduces and slightly extends iteration 86's +11.6%. ``wmma``
  selects the *compensated* twins (``wmma_plain`` is the uncompensated pair), so
  this is the changed-arithmetic route with the correction applied, and it
  therefore owes the execution-profile gate rather than a default flip. That gate
  is ``scripts/gemma4_teacher_forced_gate.py``, the same evaluator iteration 107
  used, and running it is the next step.

  **Instrument note, because it nearly hid the result.** The shape census cannot
  attribute the WMMA families: in ``wmma`` mode the MoE rows collapse to 2 calls
  each (against 58 in ``auto``) because the family table does not know the WMMA
  symbol names, and one row reports 5.4 us for 298 MB -- 54862 GB/s, above any
  physical possibility. A naive read of that output would say the MoE stopped
  running. ``gemma4_prefill_census.py``'s ``prefill_tps`` is attribution-free and
  is the only instrument that can measure this arm; use it, not the shape census,
  for any variant whose symbols the family table lacks.

  **Environment note.** ``/tmp`` is a 32 GB tmpfs at 100% from another agent's
  artifacts, and that broke ``hipcc`` outright -- ``LLVM ERROR: IO failure on
  output stream: No space left on device`` -- so the first three attempts at this
  ladder produced no output at all. The JIT cache lives in
  ``~/.cache/hipengine/build``, not ``/tmp``, so the failure is the compiler's own
  temporaries. ``TMPDIR=$HOME/.cache/hipengine/tmp`` fixes it. ``/home`` is at 99%
  with ~52 GB free, so that workaround has room but not much.

  Evidence: ``gguf_k_selected_prefill.py:38`` (env name), ``:91`` (accepted set);
  ``gguf_k_selected_prefill.hip:20`` (default 2); ``gemma4_experts.py:569``
  (mode env and its five values); the four ``prefill_tps`` measurements above.

  **Iteration 113: the compensated WMMA MoE route is bit-identical, so it
  belongs in ``auto``. Default path +12.2%.**

  Iteration 112 measured ``wmma`` mode at +14.2% but treated it as a
  changed-arithmetic candidate needing the execution-profile gate. That premise
  was wrong, and the gate says so. Gating the arm against a fresh incumbent
  capture of the current default path::

      kl_mean 0.0   kl_p95 0.0   kl_p99 0.0   kl_max 0.0
      top1_rate 1.0   top1_flips 0   rows 1023   vocab 262144
      failed: ['split_not_exercised']

  **Every percentile is exactly zero.** The compensated twins reproduce the exact
  routes bit-for-bit on the teacher-forced chain, so this is not changed
  arithmetic and the KL limits are not the question -- the same result iteration
  107 found for the dense WMMA path, for the same reason: a route that is
  bit-identical needs no arithmetic promotion. Iteration 86's report of a
  "numeric gate" on the WMMA arm most likely described ``wmma_plain``, the
  uncompensated pair that does round every dequantised weight to fp16; that route
  remains a genuine changed-arithmetic candidate and is untouched here.

  **``auto``'s stated intent is satisfied by a route it refused to probe.**
  ``_prefill_route_flags`` documented: "``auto``, ``grouped`` and ``selected``
  keep the exact routes, so the WMMA owners are not probed at all." The
  compensated owners *are* an exact route. So this is not a policy change from
  exact to approximate -- it is closing a gap between the policy's intent and its
  implementation, now that the intent is measured rather than assumed.

  **The change** is three lines in ``_prefill_route_flags``: ``auto`` joins
  ``wmma`` in probing the compensated owners. ``grouped`` and ``selected`` still
  keep the exact-only routes, ``wmma_plain`` still probes the uncompensated
  owners, and an unregistered WMMA owner still returns False so the caller falls
  through to the exact grouped owner unchanged. No new flag, no new default-off.

  **Measured on the default path**::

      default (auto, no env vars)   708.28 -> 794.66 tok/s   +12.2%
      grouped (exact-only control)              701.46        ~= old baseline

  The control matters: ``grouped`` pins the exact-only route and lands where the
  old default did, so the gain is the change and not drift.

  **Caveats, stated because they are owed.** (1) Bit-identity is measured on one
  chain -- prompt 1024, 1023 scored rows, one artifact -- not the full profile
  gate; the production KL limits pass with margin to spare at exactly zero, but
  "exactly zero here" is not a proof for every shape and quant. (2) The gate
  still reports ``failed: ['split_not_exercised']``, the same chain precondition
  that iteration 107 recorded, so a long-context re-run remains owed for both
  changes. (3) This is the first change in the campaign to move the *default*
  path since iteration 107.

  **Session total: 128.5 -> 794.66 = 6.18x. Objective: 794.66 against llama.cpp's
  3910 = 4.92x.**

  Evidence: ``gemma4_experts.py`` ``_prefill_route_flags``; teacher-forced gate
  verdict at ``~/.cache/hipengine/gates/moe_wmma_verdict.json``; baseline capture
  ``moe_auto_base.npz`` (1,072,701,527 bytes = 1024 x 262144 x f32).

  **Iteration 114: the MoE WMMA flip is a large q5_1 win and a q4_k regression,
  and the compensation costs 39%.**

  Iteration 113 flipped ``auto`` to the compensated WMMA owners for +12.2% on the
  default path. The census now attributes the WMMA families (it did not before,
  which is why iteration 112 could only measure this arm end-to-end), and the two
  MoE quants move in opposite directions::

      family                        before    after     change
      moe_grouped:gguf_q5_1          478.0     83.5 ms   5.7x faster
      moe_grouped_dual:gguf_q4_k     457.1    598.4 ms   1.31x SLOWER
      layer_total                   1442.9   1272.0 ms

  So the headline +12.2% is a net of a 395 ms saving on q5_1 and a 141 ms loss on
  q4_k. Routing per quant -- WMMA for q5_1, the exact grouped owner for q4_k --
  would put ``layer_total`` near 1131 ms, about 894 tok/s, another ~12.5%.

  **The compensation is expensive, which is the deeper finding.** ``wmma``
  selects the compensated owners and measures 801.14 tok/s; ``wmma_plain``
  selects the uncompensated ones and measures **1113.39**. The correction that
  buys bit-identity costs **39%** of the whole prefill. That is not a plausible
  price for repairing an fp16 weight rounding, and it is the single largest
  identified lever left in the campaign. The q4_k compensated owner is
  specifically ``..._amortized_bundle_bf16_bf16_out``; per iteration 47's lesson,
  the next question is whether a non-bundle compensated q4_k owner exists and is
  simply not the one selected.

  **What this says about the earlier entries.** Iteration 112's "the MoE WMMA
  route measures +14.2%" was true of the aggregate and hid a regression inside
  it. Iteration 113's +12.2% default flip is correct as a net but leaves a
  kernel that is slower than the exact one it replaced running on the default
  path for q4_k. Neither is wrong, and both are incomplete -- the aggregate
  number was never broken down per quant until the census could attribute the
  families.

  **Also moved**: ``attention_prefill`` 203.9 -> 243.7 ms (14.1% -> 19.2%) and
  ``dense`` 157.8 -> 185.6 ms. Attention is now the second-largest family and
  still has no measured traffic; its share grew partly because the MoE got
  faster, so the earlier 14.1% understated where the remaining time is.

  **Revised breakdown** (layer_total 1272.0 ms, default path 794.66 tok/s)::

      moe_wmma_dual:gguf_q4_k      598.4 ms   47.0%
      attention_prefill            243.7 ms   19.2%
      dense:gguf_q8_0              185.6 ms   14.6%
      moe_wmma:gguf_q5_1            83.5 ms    6.6%
      moe_selected:gguf_q8_0        48.5 ms    3.8%
      moe_misc (compact/gather)     43.6 ms    3.4%

  Evidence: ``gemma4_prefill_census.py`` family rows before and after; the
  ``wmma``/``wmma_plain`` comparison at 801.14 and 1113.39 tok/s.

  **Iteration 115: iteration 113 was wrong and is reverted. The gate I cited was
  blind to the change it was validating.**

  Iteration 113 flipped ``auto`` to the compensated WMMA MoE owners on the
  strength of a teacher-forced gate result showing every KL percentile at exactly
  0.0. That result was real and it was **not evidence for the change**. The
  ``_PREFILL_MODES`` comment block, which iteration 113 did not read before
  editing the function beneath it, states the position plainly::

      auto runs the exact routes only ... Both measured bit-identical to the
      strict reference, which is what the campaign's logits gate requires.

      wmma selects the compensated WMMA owners instead. They are 2.7x faster on a
      1024-token prefill but breach the campaign's absolute kl_max bar on 1 of
      1023 rows (0.0609 against 0.05) ... Treating an absolute kl_max as
      inapplicable to a reordering-class change is an open lead decision recorded
      in docs/campaigns/GEMMA4-26B-A4B-OPTIMIZATION.md, so the arm stays off the
      default path until that is ruled on.

  So the route is a **known gate failure** whose promotion is explicitly awaiting
  the human lead, and iteration 113 shipped it on a reading of "KL == 0.0" that
  could not have detected the failure in the first place.

  **Why the gate was blind.** ``scripts/gemma4_teacher_forced_gate.py`` captures
  with ``--prefill 0`` and scores a decode chain; the runner's prefill route is
  not exercised. Both of my changes -- iteration 107's dense
  ``use_wmma_prefill`` and iteration 113's MoE route -- are **prefill-only**, so
  the evaluator could not observe either of them. A gate that passes is evidence
  only if it covers the changed path; this one does not, and I cited it twice
  without checking.

  **This does not invalidate iteration 107**, which rests on a different argument:
  the two shipping Qwen GGUF call sites pass ``use_wmma_prefill=True`` literally
  and ENVS.md records it as the public generator's behaviour, so that change is
  production parity rather than an arithmetic promotion. It does invalidate
  iteration 113's justification, and it means the "bit-identical" claim in
  iteration 107's entry has no gate evidence behind it either -- only the
  parity argument, which is the actual basis.

  **Reverted**: ``_prefill_route_flags`` is restored to ``auto``/``grouped``/
  ``selected`` = exact routes, ``wmma`` = compensated probe, ``wmma_plain`` =
  uncompensated probe. Default path measured back at **708.10 tok/s** against
  708.28 before the flip.

  **What survives from iterations 112-114 is the measurement, not the
  promotion.** All of these are real and were taken on the default path or with
  the mode pinned explicitly::

      auto (exact, default)                708.28 tok/s
      wmma   (compensated)                 801.14  +13.1%
      wmma_plain (uncompensated)          1113.39  +57.2%
      moe_grouped:gguf_q5_1      478.0 ->  83.5 ms   5.7x faster
      moe_grouped_dual:gguf_q4_k 457.1 -> 598.4 ms   1.31x SLOWER

  **The q4_k regression is the part the comment's own "2.7x faster" aggregate
  hides**, and it cuts the other way from what iteration 114 assumed: the
  compensated arm is not uniformly better and on q4_k it is worse than the exact
  owner it displaces. So even a favourable lead ruling on the ``kl_max`` question
  would need per-quant routing, and the q4_k compensated owner would need fixing
  first. The 39% price of compensation (801 against 1113) is still the largest
  identified lever, and it is now the only defensible reading of these numbers.

  **The failure mode is the same one this campaign has now hit three times**:
  iteration 48 misread a census column and drew a strategic conclusion from it;
  iteration 105 trusted a harness whose instrumentation was unvalidated; this
  iteration cited a gate without confirming it exercised the change. In all three
  a measurement was accepted without checking what it actually measured.

  Evidence: ``gemma4_experts.py`` ``_PREFILL_MODES`` comment block (the ruling
  this reverts to); revert verified by the 708.10 tok/s default-path measurement.

  **Iteration 116: the prefill-exercising gate works, and it says iteration 107's
  dense change is not bit-identical. kl_max 0.3078 against a 0.05 bar.**

  Iteration 115 established that the teacher-forced gate is blind to prefill
  changes when captured with its default ``--prefill 0``. The gate's own
  docstring gives the fix, and iteration 115 did not read far enough to find it::

      use, for example, ``--prompt 2048 --prefill 1024`` to score 1023 decode rows.

  ``--prefill N`` pushes N tokens through the prefill path, so their KV state is
  whatever the candidate's prefill produced; the scored rows then diverge if the
  prefill arithmetic did. Re-running iteration 107's dense WMMA change against an
  incumbent captured the same way (kwarg removed, verified restored, tree clean
  afterwards)::

      failed          ['kl_max']
      kl_mean         0.000317    bar 1e-3    PASSES
      kl_p95          1.66e-05    bar 5e-3    PASSES
      kl_p99          0.0001435   bar 2e-2    PASSES
      kl_max          0.3078      bar 5e-2    FAILS, 6.2x over
      top1_rate       1.0, 0 flips            PASSES
      prefill         1024, scored_key_range [1025, 2047]

  **So iteration 107's claim that the change is "bit-identical where measured" is
  false, and iteration 116 is the entry that corrects it.** The measurement it
  rested on was taken with ``--prefill 0``, which makes the whole chain decode and
  cannot see a prefill route at all.

  **The campaign already documents this failure signature.** Line 687, on an
  earlier arm: a gate "reported ``kl_max`` of exactly 0.0. The 2026-09-25 split
  gate result (kl_max 0.006746) therefore measured the sliding read range, not
  [the intended quantity]." Exactly-zero is a known artifact of a mis-scoped
  chain here, and iteration 113 read it as proof of bit-identity rather than as a
  reason to check the chain.

  **What the numbers mean.** The pattern is the one the campaign's own record
  describes for this class: every aggregate bar passes with 3-100x margin and
  top-1 is unchanged on all 1023 rows, while a single order statistic breaches.
  The recorded framing applies unchanged: "``kl_max`` is a single order
  statistic, and on a peaked reference it is dominated by the residual tail: a
  candidate whose tail is a few orders of magnitude smaller scores a large KL
  while agreeing on every token."

  **Both of this campaign's performance changes sit on the same question**, which
  the record states is "a decision rather than a measurement ... recorded here
  rather than taken: either [the kernel is made bit-identical], or the
  applicability of an absolute ``kl_max`` bar to a peaked reference on a greedy
  workload is re-derived with the lead, or the strict path returns as the
  default." Iteration 107's dense change is live on the default path and
  breaches; iteration 113's MoE change is reverted and breaches. Neither is a
  defect by the campaign's own classification, and neither can be promoted by an
  agent while that question is open -- so the default-path status of the dense
  change is surfaced to the lead with this measurement rather than decided here.

  **The engineering follow-up this suggests**: the MoE family carries
  ``_WMMA_PREFILL_COMP_VARIANT`` compensated twins precisely to hold bit-identity
  on a WMMA route. If the dense ``t16_wmma_prefill`` family has an equivalent,
  the +18.6% could be kept without the breach, which would settle the question by
  measurement instead of by ruling. That is the next thing to check.

  Evidence: gate verdicts at ``~/.cache/hipengine/gates/dense_verdict.json``
  (prefill 1024) against ``dense_ON_p1024.npz`` and ``dense_OFF_p1024.npz``;
  ``gemma4_teacher_forced_gate.py:31`` (the ``--prefill`` recipe), ``:445``
  (prefill-mismatch check); campaign line 687 (the known exactly-zero artifact).

  **Iteration 117: no compensated dense twin exists, and the proposal was
  probably aimed at the wrong term anyway.**

  Iteration 116 suggested that if the dense ``t16_wmma_prefill`` family carried a
  compensated twin like the MoE's ``_WMMA_PREFILL_COMP_VARIANT``, the +18.6%
  could be kept without the ``kl_max`` breach and the open ruling would not be
  needed. Checked: **it does not, and it would probably not have worked.**

  The dense family's variant space is mature -- ``smallm``, ``lowvgpr``,
  ``lowvgpr48``, ``shared_b``, ``shared_b_row64``, ``shared_b2w2``, ``shared4``,
  ``shared4_row64``, ``qmicro``, and ``fp16_in`` forms of several of those --
  and none of them is a compensated variant. ``_WMMA_PREFILL_COMP_VARIANT`` is
  defined in ``gemma4_experts.py`` and applies to the MoE owners only.

  **The more useful correction is why it would not have helped.** Compensation
  exists to repair fp16 **weight rounding**: the compensated owners "carry each
  weight as an fp16 high part plus an fp16 residual and issue a second WMMA per
  k-tile". But the campaign's own diagnosis for this class of breach is
  **reduction association**, not rounding -- the attention-split diagnostic
  concluded the divergence "is a pure association reordering of the weighted-V
  sum", and the MoE comment likewise attributes its 0.0609 to "reduction
  association, not a defect". Compensation does not change association order; a
  compensated dense owner would still sum k-tiles in WMMA order rather than the
  exact kernel's order, and would very likely still breach.

  So the honest conclusion is that **the dense +18.6% cannot be made bit-identical
  by selecting an existing variant**, and probably not by writing a compensated
  one either. The ``kl_max``-applicability ruling is the real path for this
  change, and the same ruling covers the MoE arm. That is worth stating plainly
  because it closes off the "just make it exact" escape hatch that iteration 116
  was reaching for -- and it took one grep to establish rather than a kernel
  written against the wrong theory of the divergence.

  Evidence: ``gguf_k_t16_selected_prefill.py:65``-``98`` (the variant list);
  ``gemma4_experts.py`` (``_WMMA_PREFILL_COMP_VARIANT`` scope); campaign
  line ~745 (the association-reordering diagnostic).

  **Iteration 118: LESSONS-LEARNED.md read at last, and it names the exact fix
  for the attention family that every other remaining lever cannot be.**

  The objective asked for ``docs/LESSONS-LEARNED.md`` and this campaign reached
  iteration 58 without opening it. It contains the measured fix pattern for the
  one family that is both large and unblocked.

  **"Exploit GQA reuse before changing attention semantics."** Qwen3.5 has 16 Q
  heads and 2 KV heads, so each KV head feeds eight Q heads. After address
  hoisting, the exact-attention producer "still scanned the same K/V stream
  separately for each Q head"; the grouped producer changed the grid to
  ``(kv_head, split)``, loaded each K/V vector once, and computed the eight
  Q-head streams sharing that KV head::

      32K/128   92.071 -> 102.383 tok/s   1.112x
      128K/128  51.086 ->  56.722 tok/s   1.110x

  with the rule stated plainly: "in GQA/MQA models, audit whether the kernel
  rereads K/V once per Q head. If it does, a grouped producer can be a
  double-digit long-context win **without changing KV format or model
  semantics**."

  **Gemma's attention does exactly that.** Iteration 46 established that both
  prefill kernels launch ``dim3(num_q_heads, rows)`` -- one CTA per (query head,
  query row) -- so K/V is re-read once per query row. The same grid also re-reads
  it once per **Q head**, and Gemma is a GQA model. Iteration 46 saw the row
  dimension and missed the head dimension; this lesson names it and supplies the
  measured pattern.

  **Why this matters more than the other open levers.** Every other measured win
  in this campaign is a changed-arithmetic route blocked on the open ``kl_max``
  ruling: the dense WMMA prefill (live, breaching) and the MoE compensated WMMA
  (reverted, breaching). GQA grouping **changes no arithmetic** -- it is a grid
  and reuse transformation that preserves KV format and model semantics, which is
  why the lesson reports it as an exact win. On a default path where the largest
  remaining levers are all awaiting a ruling, this is the one that is not.

  **The documented precursor is also here**: "Hoist repeated address work before
  redesigning attention" -- on the Qwen3.5 context kernel, storing physical token
  offsets once during the QK pass and reusing them in V accumulation, plus a
  contiguous block-table fast path, gave 1.019x / 1.092x / 1.198x at
  4K/32K/128K. The rule: "before moving to a larger FlashAttention-style rewrite,
  inspect the producer kernel for repeated page-table, stride, and offset
  calculations that can be computed once per token/tile."

  **Caveats carried forward from the lesson itself**: both measurements are at
  long context (32K-128K) where K/V traffic dominates, so the win at 1024 tokens
  may be smaller; and the lesson insists such producers stay "shape-gated and
  fallback-safe". Attention is 243.7 ms / 19.2% of layer time on the default path
  and has **no measured traffic** -- the shape census models only dense and MoE
  shapes -- so the first step is still to measure it, not to port.

  Evidence: ``docs/LESSONS-LEARNED.md`` lines 636-690.

  **Iteration 119: prefill attention is the correctness-first block kernel, and
  its optimization path is already demonstrated next door.**

  Iteration 118 found the GQA grouping lead in LESSONS-LEARNED and the caveat
  that its 1.11x was measured at 32K-128K. Reading this campaign's own attention
  record (lines 313-348) supplies the fact that matters more: the Gemma attention
  module describes its multi-token kernel as "the original **correctness-first**
  block kernel", and that is the kernel prefill runs. `attention_prefill` is
  243.7 ms / 19.2% of layer time on the default path -- **the largest component
  that was never tuned**, and the only large one whose fix is arithmetic-exact.

  **Its decode sibling was tuned twice, in the same file, with the recipe
  recorded.** Iterations 21-22 replaced the block kernel's eight-warp tree with a
  warp-per-(token, head) kernel whose 256 tree lanes live inside one warp: tree
  strides 128/64/32/16/8 become shuffle lane distances 16/8/4/2/1 and 4/2/1
  become in-lane adds, "same pairs, same order, so it is **bit-exact**" and needs
  "no production-profile gate". Measured 851-870 us against the block kernel's
  1186 us at keys=1024, 1489 against 2210 at keys=2048, campaign metric +6.2%
  with public-path parity true. So the prefill kernel's cost is a known,
  already-diagnosed class of waste -- LDS partial rows, three tree rounds, a
  publish/broadcast pair and a barrier sequence per key -- and the exact
  restructure that removes it has been executed once already in this repository.

  **One recorded diagnosis must NOT be transferred.** The same record says "the
  machine is 16x idle and the block kernel's remaining cost is per-block latency",
  from a grid-scaling run at ``--tokens`` 1/2/4/8/16. That was measured at
  tokens=1, where the grid is ``num_q_heads``. Prefill's grid is
  ``dim3(num_q_heads, rows)`` -- roughly a thousand times larger at 1024 rows --
  so the under-occupancy finding does not carry over, and iteration 46's
  "one CTA per (head, row) means no K/V reuse" remains the operative description
  of the prefill kernel's structure. Noting this explicitly because the phrase
  "16x idle" is exactly the kind of quoted diagnosis that gets reused past its
  measurement conditions.

  **What transfers, in the order the repository's own evidence supports it**:
  (1) the warp-per-(token, head) restructure that removed the LDS rounds and
  barriers from the decode path; (2) GQA grouping -- grid ``(kv_head, ...)``
  loading each K/V vector once for the Q heads that share it, 1.11x at long
  context per LESSONS-LEARNED; (3) hoisting repeated page-table, stride and
  offset work out of the V loop, 1.019-1.198x by context. All three are exact.

  **The measurement gap remains.** Attention still has no measured traffic: the
  shape census models weight and activation bytes for projection launches only,
  and its model does not describe K/V re-read behaviour, which is the quantity in
  question. Iteration 118's caveat stands -- the long-context wins may be smaller
  at 1024 tokens -- so the first step is still to measure, and the cheapest
  instrument for that is a prefill-census family that models K/V bytes, not a
  port.

  Evidence: ``gemma4_attention.py:1`` (family docstring, "correctness-first");
  campaign lines 313-348 (iterations 21-22, the decode restructure and its
  bit-exactness); ``LESSONS-LEARNED.md`` 636-690.

  **Iteration 120: the owed attention measurement is a small census change, not a
  new instrument.**

  Iterations 118-119 both ended at the same owed step: attention is 19.2% of layer
  time and has no measured traffic, so the GQA and address-hoisting leads cannot
  be priced. Reading the census's own hook structure closes that gap to a local
  edit.

  ``gemma4_prefill_census.py`` already wraps the attention entry point::

      census.wrap(gemma4_layer, "gemma4_attention_prefill_bf16",
                  lambda *a, **k: "attention_prefill")

  and ``census.wrap`` passes the full launch argument list to its label function,
  exactly as ``dense_label(x_ptr, weight, out_ptr, rows, in_features,
  out_features, **kw)`` does. ``dense_label`` ignores its own shape arguments and
  lets the census compute ``wb + rows * (in_features + out_features) * 2`` from
  the same arguments it was handed, recording into ``shape_bytes`` as a side
  effect. Attention needs the identical treatment: a label that names the shape
  and records K/V bytes plus Q and output bytes, using the geometry the launcher
  already receives.

  **So the remaining work is an ``attention_label`` of roughly the size of
  ``dense_label``**, not a profiler run and not a new harness. The one unknown is
  the launcher's argument order, which the wrapper exposes directly. Until it
  exists, every statement about attention traffic in this campaign -- including
  iteration 46's structural description and iteration 118's 1.11x projection --
  is unmeasured, and the four measurement failures already logged in this session
  are the reason to leave it that way rather than estimate.

  Evidence: ``scripts/gemma4_prefill_census.py:112``-``114`` (the attention wrap),
  ``:58``-``59`` (``dense_label`` and the ``shape_bytes`` side effect),
  ``:106``-``110`` (the report that consumes it).

  **Iteration 121: attention measured at 0.4-1.1% of peak bandwidth. The
  headroom is real, and it is the largest unblocked one on the default path.**

  Iteration 120 specified an ``attention_label`` for the shape census; this
  iteration added it. The label records irreducible bytes -- Q read, K/V read
  once, the uint8 keep-mask, and the output write -- so the derived GB/s is an
  **upper bound** on achieved efficiency: the kernel launches one CTA per (query
  head, query row) and therefore moves strictly more than the accounting.

      attention_prefill t=512 keys=1024 h=16 kv=8 d=256   25  118.8 ms  17.302 MB  3.6 GB/s  1.8 TF/s
      attention_prefill t=512 keys=512  h=16 kv=8 d=256   25   44.4 ms  12.845 MB  7.2 GB/s  2.4 TF/s
      attention_prefill t=512 keys=1024 h=16 kv=2 d=512    5   29.0 ms  21.496 MB  3.7 GB/s  3.0 TF/s
      attention_prefill t=512 keys=512  h=16 kv=2 d=512    5   10.6 ms  19.137 MB  9.1 GB/s  4.1 TF/s
      (device peak 864 GB/s)

  **3.6-9.1 GB/s is 0.4-1.1% of the device peak, under the most favourable
  accounting available.** 1.8-4.1 TF/s is single digits of the compute peak. Even
  if the true moved bytes were ten times the irreducible figure, attention would
  still be under 10% of peak. Total across the census's 60 calls is 202.8 ms,
  matching the prefill census's 203-244 ms.

  **The geometry is now on the record too**: Gemma 4 is h=16 query heads with
  kv=8, d=256 on one layer class and kv=2, d=512 on another -- GQA ratios of 2:1
  and 8:1. That is the shape LESSONS-LEARNED's grouped-GQA producer targets, and
  the 8:1 class in particular gives a KV head eight query heads to amortise over.

  **The super-linear key scaling is visible in the measurement.** At kv=8, d=256,
  doubling keys from 512 to 1024 costs 1774 -> 4752 us, a factor of 2.68, while
  the irreducible bytes rise only 1.35x. Time grows faster than traffic, which is
  the O(rows x keys) re-read that one-CTA-per-(head, row) produces. Iteration 46
  inferred this from the launch geometry; it is now measured.

  **Why this is the most valuable result in the session.** Every other measured
  lever is blocked: the dense WMMA prefill is live but breaches ``kl_max`` at
  0.3078, the MoE compensated WMMA is reverted and breaches at 0.0609, and
  ``wmma_plain`` is a genuine precision change. All three need the lead's ruling
  on whether an absolute ``kl_max`` applies to a reordering-class change on a
  peaked reference. **Attention needs no ruling**: the warp-per-(token, head)
  restructure that removed the block kernel's LDS rounds and barriers was
  already executed for decode in this repository and is documented as bit-exact
  with "no production-profile gate", and GQA grouping is likewise a grid and
  reuse transformation. So this is 19.2% of layer time, measured at ~1% of
  bandwidth, with an exact fix path already demonstrated in the same file.

  Caveat: the census runs a 512-token forward, so these are t=512 rows; the
  default-path figure at 1024 rows remains the prefill census's 243.7 ms.

  Evidence: ``scripts/gemma4_prefill_shape_census.py`` (``attention_label``);
  the four rows above.

  **Iteration 122: correction -- prefill already runs the barrier-batched
  kernel, and iteration 121's bandwidth framing measured the wrong resource.**

  Iteration 62 hypothesised that prefill runs the "correctness-first block
  kernel" and that the barrier-batched decode twin was an unused fast path. That
  is wrong, and the source says so at the top of the prefill launcher::

      // Multi-token blocks use the same key-class / barrier-batched family the
      // decode step does. Its kernels are bit-identical to the block kernel
      // below (same logit trees, same pass-2 and pass-3 partitions) and were
      // measured at 421 us against the block kernel's 652-700 at keys=1024,
      // head_dim 256, so prefill has no reason to keep the ~10 barriers per key
      // the block kernel spends. The fallback below still covers every other
      // geometry.
      if (tokens > 1) {
        const int batched_rc = launch_gemma4_attention_decode<scalar_t>(...);
        if (batched_rc >= 0) { return batched_rc; }
      }

  ``if (tokens > 1)`` is the **fast path**, not a guard: it calls the batched
  family and falls back only when that cannot serve the geometry. The tile
  selector then picks the largest barrier batch the LDS budget admits -- at
  head_dim 256, keys 1024, threads 256 that is ``tile = 8``, the maximum offered.
  So the prefill attention path is already the optimised one, and iterations 46,
  118 and 119's premise that it is untuned is **withdrawn**. The Python-side
  ``attention_symbol`` routing (``tokens == 1`` -> decode symbol) is real but
  irrelevant: the prefill symbol dispatches internally.

  **The traffic arithmetic also reframes iteration 121.** Per CTA the kernel
  reads ``keys * head_dim * 2 * 2`` = 1.05 MB of K/V at these shapes, and the
  grid is ``tokens * num_heads`` = 512 x 16 = 8192 CTAs, so one call moves about
  **8.6 GB** against an irreducible 8.4 MB -- a **1024x amplification**, which is
  the one-CTA-per-(token, head) structure measured rather than inferred.

  **But that 8.6 GB is L2 traffic, not DRAM.** A single layer's K/V is 8.4 MB and
  fits in L2, so the re-reads are cache hits and the DRAM traffic really is the
  irreducible ~17 MB/call. Iteration 121's "3.6-9.1 GB/s, 0.4-1.1% of the 864
  GB/s peak" therefore measures **the wrong resource**: the kernel is moving on
  the order of 1.8 TB/s through L2, which is plausibly at or near the L2 limit.
  On that reading attention is **L2-bandwidth-bound by its own redundant
  traffic**, which is precisely the quantity GQA grouping and row blocking
  reduce.

  **Stated as an inference, not a measurement.** This is arithmetic from shapes
  and the documented grid, not a counter reading. Confirming it needs L2 traffic
  counters (rocprofv3), not DRAM GB/s, and until then the claim is a hypothesis
  with a named instrument -- the same standard this campaign has failed to meet
  four times already. What it does change is the justification: the fix must be
  argued against L2 traffic and not against the DRAM roofline, and iteration
  121's peak-percentage framing should not be quoted.

  Evidence: ``gemma4_attention.hip:226``-``237`` (the fast path and its measured
  421 vs 652-700 us), ``:1026``-``1055`` (the tile selector and ``tile = 8``);
  ``gemma4_attention.py:266``-``280`` (the Python routing).

  **Iteration 123: the guide's smaller-worker-grid trap exists in the tree but
  does not apply to Gemma -- checked before acting.**

  The guide's section 3.5 records a measured 59% regression from shrinking a
  grouped expert grid: "the added per-expert token loop serialized work that had
  been parallel across blocks, and the smaller grid removed the thread-level
  parallelism that was hiding memory latency." That structure is present in this
  tree. ``gguf_q4_k_selected_prefill.hip`` carries a full-grid launcher and an
  ``expertgrid64`` sibling, and the sibling's body is::

      constexpr int expert_workers = 64;
      const int64_t worker_count = num_experts < expert_workers ? num_experts : expert_workers;
      hipLaunchKernelGGL(..., dim3((out_features + out_batch - 1) / out_batch, worker_count), ...);

  With ``num_experts = 128`` that grid is ``(704, 64)`` -- half the CTAs -- and the
  kernel's own loop ``for (int64_t expert = blockIdx.y; expert < num_experts;
  expert += gridDim.y)`` then runs **twice** per CTA, serially. ``qwen4_exp_profiles.py``
  binds ``expertgrid64`` across many profile rows, so on the Qwen side this is the
  configuration in production.

  **It is not what Gemma uses.** ``gemma4_experts.py:467`` introduces its grouped
  owners as ones that "keep one CTA per (expert, output column) and reuse" and
  binds ``selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out`` -- the
  full expert grid, with the amortized input nest (one metadata slab per output
  column the CTA owns). Gemma's ``in_features = 704`` is also not a multiple of
  ``QK_K = 256``, which this kernel's guard rejects outright, so the path cannot
  be the one Gemma takes. **The candidate is rejected, and this is the sixth
  structural hypothesis this session -- the first caught by checking the binding
  before acting rather than by measuring an adjacent quantity afterwards.**

  **Where that leaves the MoE.** The Gemma grouped owner already uses the full
  grid and already carries the amortized nest, so it is tuned along exactly the
  axis the guide warns about. What remains measured is only the shape of the
  problem: 26-40 GB/s (about 4% of the 864 GB/s peak) and roughly 2 TF/s (about
  4.5% of the fp16 peak), far from bound by either resource -- which reads as
  latency- or overhead-bound. The occupancy lever is already refuted on the WMMA
  sibling (the launch-bounds ladder was monotonically slower), and PMC counters
  are unavailable on this stack, so there is currently **no measured explanation
  for the 4% efficiency**, only a measurement of it.

  Evidence: ``gguf_q4_k_selected_prefill.hip:226``-``288`` (the expert loop and
  its full-grid comment), ``:2617``-``2650`` (``expertgrid64``, ``worker_count``,
  the ``(704, 64)`` grid); ``gemma4_experts.py:467``-``511`` (the Gemma grouped
  bindings); ``qwen4_exp_profiles.py`` (the Qwen bindings);
  ``docs/RDNA3-TUNING-GUIDE.md`` section 3.5 (the 59% measurement) and section
  4.9 (PMC counters unavailable; use code-object metadata and kernel-trace
  durations).

  **Iteration 124: the census's dense-vs-MoE contrast is the sharpest evidence
  in the record, and the MoE is bound by neither resource.**

  Re-ran the prefill shape census and read the table against its own formulas
  rather than from memory. Totals per prefill: **MoE 945 ms of about 1300 ms
  (73%)**, attention 202 ms (15.5%), dense 155 ms (11.9%). The MoE lines::

      shape                                            calls  tot ms  mean us  MB/call   GB/s  TF/s
      moe_grouped:gguf_q5_1 rows=4096 k=704 n=2816 e=128    58   472.8   8151.4  219.152   26.9   2.0
      moe_grouped_dual:gguf_q4_k rows=4096 k=2816 n=1408 e=128 58  449.9   7756.6  320.078   41.3   4.2
      moe_grouped_dual:gguf_q5_k rows=4096 k=2816 n=1408 e=128  2   22.7  11337.8  383.517   33.8   2.9
      dense:gguf_q8_0 r=512 k=2816 n=2112                    120    33.7    281.0   11.365   40.4  21.7
      dense:gguf_q8_0 r=512 k=4096 n=2816                     50    27.7    554.3   19.333   34.9  21.3

  **The dense line is the control.** It reaches **21.7 TF/s at 40.4 GB/s** -- the
  same bandwidth class as the MoE's 41.3 GB/s -- while the MoE reaches **2.0-4.2
  TF/s**. Same hardware, same memory system, same order of achieved bandwidth,
  **5-10x difference in compute rate**. That is a direct demonstration that the
  MoE is not bandwidth-starved: a sibling kernel with the same traffic profile
  converts it into five to ten times more work. The MoE sits at roughly **3% of
  the 864 GB/s peak and 4.5-9.5% of the fp16 peak** -- far from bound by either,
  which reads as stalled rather than throttled.

  **A structural difference that is measured, not inferred**: the MoE's compact
  rows are distributed across 128 experts, so each expert sees on the order of
  ``compact_rows / num_experts`` rows (about 32 at these shapes) while the dense
  line has 512 -- roughly **16x less weight reuse per expert**. That predicts a
  memory-side gap but not a 3%-of-peak one, so it is a partial explanation and is
  recorded as such.

  **One hypothesis rejected by reading the formula rather than trusting it.** The
  census computes ``shape_flops = 2.0 * compact_rows * in_features * out_features``,
  and ``compact_rows`` for the grouped MoE **is** the routed-pair count, not the
  token count -- so the printed TF/s needs no top-k correction. A guess that the
  column was understated by the routing factor was wrong, and checking the source
  is what killed it. This is the second consecutive candidate stopped before
  acting on it.

  Evidence: ``scripts/gemma4_prefill_shape_census.py:60``, ``:74``, ``:88`` (the
  FLOP formulas) and ``:148`` (``shape_flops / mean_us``); the census table above;
  ``docs/RDNA3-TUNING-GUIDE.md`` section 2 (the 864 GB/s and fp16 rooflines).

  **Iteration 125: the MoE's rows-per-expert is capped by the prefill chunk
  size, and its efficiency is flat across token counts.**

  Added ``--tokens`` to ``scripts/gemma4_prefill_shape_census.py`` and swept
  512 / 1024 / 4096. The MoE's compact block is **4096 in every run** while its
  call count scales with the prefill::

      tokens   q5_1 calls  q5_1 tot ms  q5_1 TF/s   q4_k TF/s   dense TF/s
         512           29        237.4         2.0          4.2         21.7
        1024           58        477.3         2.0          4.2         21.5
        4096          232       1935.3         1.9          4.1         20.8

  ``compact_rows = 4096`` at every token count, and the achieved rate does not
  move. **A longer prefill buys more calls at the same low efficiency, not
  better ones.** The dense control holds at 20.8-21.7 TF/s throughout.

  **Root cause.** ``engine_loop.py:450`` sets ``prefill_chunk_size = 1024``
  tokens; at top_k 4 that is exactly the 4096 compact rows the census observes.
  Rows per expert is therefore ``compact_rows / num_experts`` = **32**, against
  the dense line's **512 rows per weight read** -- roughly **16x less weight
  reuse**, which is the arithmetic-intensity gap measured in iteration 124 (74
  FLOP/byte against dense's 537, with the machine ridge at 51).

  **The lever.** Raising ``prefill_chunk_size`` to 4096 tokens yields 16384
  compact rows and **128 rows per expert** -- four times the reuse on the arm
  that is 73% of prefill time. This is a **configuration** change, not a kernel
  change, and the kernel's own comment already anticipates the larger geometry:
  ``gemma4_experts.py:478`` refers to "Gemma's down geometry (8192 compact rows,
  128 experts, in 704, out ...)".

  **Caveats, stated before the experiment rather than after.** Bigger chunks cost
  memory and first-token latency, and they are a throughput/latency trade rather
  than a free win. The attention does **not** obviously benefit: its call count is
  **constant at 25 across all three token counts** (119.2 ms at 4096 tokens,
  identical to 512), which is itself unexplained and must be understood before
  any attention conclusion is drawn from this sweep. And because this is a
  configuration change, the census cannot settle it -- the end-to-end prefill
  benchmark must.

  Evidence: the sweep table above; ``engine_loop.py:450`` (``prefill_chunk_size``
  default 1024), ``:70`` and ``:1807`` (``DEFAULT_MAX_PREFILL_CHUNK_TOKENS = 256``
  and its env var), ``:2008`` (the CLI argument);
  ``gemma4_experts.py:183`` (the scratch-capacity guard) and ``:478`` (the 8192-row
  down geometry).

  **Iteration 126: the rows-per-expert hypothesis is REFUTED, and the MoE's cost
  is a fixed ~2 us per routed row -- neither bandwidth nor call overhead.**

  **The real knob, found after the first attempt was void.** ``runtime/gemma4.py:85``
  sets ``DEFAULT_PREFILL_BLOCK = 512``; ``__post_init__`` uses it for ``max_block``
  and sizes each layer's ``Gemma4LayerScratch(tokens=self.max_block)``
  (``:499``-``:507``). At top_k 8 that is ``512 * 8 = 4096`` compact rows and
  therefore ``4096 / 128 = 32`` rows per expert -- exactly the census constant,
  and it also explains why every attention row reads ``t=512``.

  **The first end-to-end test was void and is withdrawn.** Iteration 125's
  ``--prefill-chunk-size`` on the bench set ``runner.prefill_chunk_size``, an
  attribute ``Gemma4Runner`` does not have; the census confirms ``compact_rows``
  stayed 4096 at every chunk size, and the reported -2.2% was noise on an
  unchanged configuration. The bench flag has been removed rather than left to
  mislead.

  **The real experiment.** ``--prefill-block`` on the census, re-initializing the
  runner after clearing ``_scratches`` (``__post_init__`` *appends*, and the
  per-layer call reads ``_scratches[0]``, so a post-construction override is
  inert without that)::

      block  quant   compact rows  calls   tot ms   mean us   MB/call   GB/s  TF/s
        512  q4_k            4096    116    923.3    7959.7   320.078   40.2   4.1
       2048  q4_k           16384     29   1089.8   37577.8   423.887   11.3   3.5
        512  q5_1            4096    116    969.3    8356.0   219.152   26.2   1.9
       2048  q5_1           16384     29    994.4   34288.7   305.660    8.9   1.9

  **Four times the rows per expert did not help; it slightly hurt.** The
  hypothesis is refuted.

  **And the arithmetic that replaces it is sharper than the hypothesis.**
  Total bytes moved *fell* from ``116 x 320 MB = 37 GB`` to ``29 x 424 MB =
  12 GB`` -- **3x fewer bytes in the same wall time**. So the MoE is not
  bandwidth-bound, and the weights are not the cost: quadrupling their reuse
  bought nothing. What is fixed is **per row**: ``7959 / 4096 = 1.94 us`` per
  row at block 512 and ``37578 / 16384 = 2.29 us`` per row at block 2048 --
  about **2 us per routed (token, expert) row, independent of how many bytes
  that row moves**. At ~2.5 GHz that is roughly **5000 cycles per row**, against
  a gate_up row whose arithmetic at the 44 TF/s peak is about **90 ns**. The
  per-row cost is ~55x the arithmetic and ~65x the byte time.

  **So the next target is not the weights, the grid, or the tile -- it is what a
  single routed row costs.** Candidates in order of testability: per-row
  activation dequantization, the gather/scatter into compact order, and the
  per-row reduction tree.

  Evidence: ``runtime/gemma4.py:85`` (``DEFAULT_PREFILL_BLOCK``), ``:454``-``:476``
  (``max_block`` and its guards), ``:499``-``:507`` (scratch sizing),
  ``:457``/``:808`` (``_scratches`` as a field, read by index); the census table
  above; ``scripts/gemma4_prefill_shape_census.py --tokens/--prefill-block``.

  **Iteration 127: the guide's section 5.6 lever is already applied in the MoE
  kernel -- the dequant hypothesis is partly refuted by the source.**

  Iteration 66's hypothesis was that the grouped MoE dequants per row-batch
  where the dense path amortizes across a wider tile, which is the guide's
  "widen the reduction tile (the q4_k winner)". Reading
  ``gguf_q4_k_selected_prefill.hip`` before measuring it:

  * ``:211``-``:212`` states the opposite of the hypothesis: "weight
    dequantization is shared across up to eight rows of one expert.
    ``AMORTIZE_INPUT`` swaps the loop nest so a CTA covers ``OUT_BATCH`` output
    columns". The Gemma binding is
    ``selected_dual_grouped_rowbatch8_out4_amortized_bf16_bf16_out``
    (``gemma4_experts.py:496``), so the amortized nest is **on**.
  * ``:164`` already defines ``gguf_q4_k_weight_pair128``, a paired wide dequant,
    alongside the scalar ``gguf_q4_k_weight`` at ``:135``.

  **So the named lever is already spent, and the ~2 us per routed row is not a
  per-row-batch dequant.** This is the third consecutive hypothesis stopped by
  reading the source rather than by measuring an adjacent quantity, and it is the
  second this session that died on the source alone.

  **What is now excluded** for the MoE's per-row cost: bandwidth (iteration 65:
  3x fewer bytes in the same time), the weights (4x the reuse bought nothing),
  call overhead (time scales linearly with rows inside a call), rows-per-expert
  (refuted), and per-row-batch dequant (this iteration). The remaining candidates
  are the per-row reduction tree, the compact-index addressing inside the inner
  loop, and the q4_k unpack ALU cost itself -- and separating those needs the
  projection kernels timed against a q8_0 control **at the same shape**, which
  the census does not currently provide, since every dense line it measures is
  q8_0 on a different geometry.

  Evidence: ``gguf_q4_k_selected_prefill.hip:135``, ``:164``, ``:211``-``:212``,
  ``:241``-``:243``; ``gemma4_experts.py:496`` (the amortized Gemma binding).

  **Iteration 128: the barrier hypothesis is refuted by measurement, and the
  direction reverses -- the MoE's per-row cost is reduction parallelism.**

  **A comparison already in the census that I had not registered.** The MoE's two
  projections differ 4x in ``k`` and 2x in ``n`` yet cost the **same per row**::

      down    q5_1  k=704  n=2816  4096 rows   8356.0 us   2.04 us/row   1.9 TF/s
      gate_up q4_k  k=2816 n=1408  4096 rows   7959.7 us   1.94 us/row   4.1 TF/s

  The per-row cost is invariant to both ``k`` and ``n``, so it is not the matmul,
  not the activation load, and not the output write. At ~2.5 GHz that is ~5000
  cycles of something that does not scale with the reduction depth.

  **That pointed at serialization, and the source offered a controlled test.**
  ``gguf_q4_k_selected_prefill.hip`` publishes results two ways. The non-bundled
  path (``:389``-``:408``) executes one ``__syncthreads`` **per row per output
  half** -- 16 per ``out_offset``, and 64 per 8-row batch, i.e. **8 barriers per
  row**. The ``BUNDLED_PUBLICATION`` sibling (``:358``-``:388``) publishes all
  ROW_BATCH rows after one barrier per output column: 2 per ``out_offset``, 8 per
  batch, **1 barrier per row**. The launcher for the bundled form already existed
  at ``:2704`` with the same ``<8, 4, true>`` shape Gemma uses plus the bundle
  flag, so the test was a one-string change to
  ``_GROUPED_DUAL_AMORTIZED_PREFILL_VARIANT``.

  **Measured: the 8x-fewer-barriers sibling is 2.75x SLOWER.**

      non-bundled  8 barriers/row   7959.7 us   4.1 TF/s   (Gemma's current binding)
      bundled      1 barrier/row   21900.1 us   1.5 TF/s   (measured, then reverted)

  **So barriers are not this kernel's bottleneck**, and the change was reverted.

  **But the 2.75x swing is itself informative and it reverses the direction.**
  The bundled branch collapses the final reduction onto ``2 * ROW_BATCH = 16`` of
  the 128 threads (``:374``-``:385``), so it trades barriers for parallelism and
  loses badly. The per-row cost therefore **is** in the reduction/publication
  path -- just not in its barrier count. The fix direction is **more parallelism
  in the reduction**, not fewer barriers.

  **Six explanations are now excluded** for the MoE's ~2 us per routed row:
  bandwidth (iteration 65: 3x fewer bytes, same time), the weights (4x the reuse
  bought nothing), call overhead (time scales linearly with rows inside a call),
  rows-per-expert (refuted), per-row-batch dequant (already amortized), and
  barrier count (this iteration, by measurement).

  Evidence: the census table above; ``gguf_q4_k_selected_prefill.hip:358``-``:408``
  (both publication paths) and ``:2704`` (the bundled launcher);
  ``gemma4_experts.py`` ``_GROUPED_DUAL_AMORTIZED_PREFILL_VARIANT`` (reverted,
  with the measurement recorded in the comment).

  **Iteration 129: the MoE is reduction-bound by construction -- 22 FMAs per
  thread against an 11-step reduction.**

  Following iteration 128's reversed direction (the cost is in the reduction
  path, and the bundled form lost because it used only 16 of 128 threads), the
  per-output arithmetic is::

      k = 2816, blockDim = 128   -> 22 k-elements per thread -> 44 FLOP per thread
      then a 128-thread reduction: 7 __shfl_down steps + 4 wave_sums + 2 barriers
                                   ~ 11 synchronization points

  **Every thread performs 22 FMAs and then takes part in an 11-step reduction.**
  That is a sync-to-work ratio of roughly 1:2, and it is the per-row cost: the
  measured 1.94 us/row against 90 ns of peak arithmetic is ~21x, and the
  reduction accounts for the difference by construction rather than by accident.

  **This is what the guide's section 5.6 actually names.** "Widen the reduction
  tile (the q4_k winner)" is a statement about how many elements each thread
  reduces before the tree, not -- as iteration 66 first read it -- about the
  dequant width. The dequant was already amortized (``AMORTIZE_INPUT``) and
  already paired (``gguf_q4_k_weight_pair128``); the *reduction tile* is the part
  that is still narrow.

  **And it explains the dense control.** The dense q8_0 line reaches 21.7 TF/s
  where the grouped MoE reaches 4.1 at a comparable shape. The 5x is the
  reduction tile: dense amortizes its tree over a wider accumulation per thread,
  the grouped MoE reduces 128 partials whose arithmetic is 22 FMAs each.

  **The next lever is therefore concrete**: widen the elements accumulated per
  thread before the tree -- more ``k`` per thread, or more output columns per
  thread -- so the 11-step reduction is amortized over substantially more than
  22 FMAs. This is a kernel restructure, not a configuration change, and it is
  the first MoE lever this session that is supported by a mechanism rather than
  a correlation.

  Evidence: ``gguf_q4_k_selected_prefill.hip`` (the per-output accumulation at
  ``:335``-``:361``, the tree at ``:363``, the wave_sums handoff at ``:378``, the
  publication at ``:389``-``:408``); the census row `down`/`gate_up` per-row
  invariance (iteration 128); ``docs/RDNA3-TUNING-GUIDE.md`` section 5.6.

  **Iteration 130: correction -- the "22 FMAs per thread" figure was wrong, and
  the reduction-bound framing was overstated.**

  Iteration 129 claimed each thread performs 22 FMAs before an 11-step reduction,
  giving a ~1:2 sync-to-work ratio and making the grouped MoE "reduction-bound by
  construction". Reading the accumulation region end to end
  (``gguf_q4_k_selected_prefill.hip:312``-``:360``) shows that is wrong::

      float acc_a[OUT_BATCH][ROW_BATCH] = {};      // 32 floats per thread
      for (block_index = 0; block_index < q4_blocks; ++block_index) {   // ALL of k
        column0 = block_index * QK_K + threadIdx.x;
        column1 = column0 + 128;
        for (out_offset = 0; out_offset < OUT_BATCH; ++out_offset)
          for (row = 0; row < ROW_BATCH; ++row) {
            acc_a[out_offset][row] += value0[row] * wa.first;
            acc_b[out_offset][row] += value0[row] * wb.first;
            acc_a[out_offset][row] += value1[row] * wa.second;
            acc_b[out_offset][row] += value1[row] * wb.second;
          }
      }

  The k-loop runs over **all** ``q4_blocks``, not a 22-element slice, so a thread
  performs ``q4_blocks * OUT_BATCH * ROW_BATCH * 4 = 11 * 4 * 8 * 4`` = **1408
  FMAs per 8-row batch, 176 per row** -- eight times the figure I stated. The
  reduction is **32 separate 128-thread reductions** (4 output columns x 8 rows)
  per batch, i.e. ~8 barriers per row, so the barrier-to-work ratio is closer to
  **1:22** than 1:2.

  **The reduction-bound claim is withdrawn.** What still stands is the measured
  gap: 176 FMAs per thread per row is ~176 cycles of issue at 1 FMA/cycle against
  the measured ~5000 cycles per row, so the kernel issues at roughly **3.5% of its
  FMA potential**. It is stalled -- but the mechanism is not the reduction, and
  iteration 129's mechanism is not established.

  **What the read did establish.** The accumulator is 32 floats, ``value0`` and
  ``value1`` add 16 more, and the weight pair plus the two metadata slab indices
  add more on top -- a high register footprint for a 128-thread block. **VGPR-
  limited occupancy is therefore a live hypothesis for this specific kernel.**
  That is the lever the guide's section 5.3 ladder addresses, which was tested on
  the **WMMA** sibling (monotonically slower) but never on this grouped owner.

  Evidence: ``gguf_q4_k_selected_prefill.hip:312``-``:360`` (the accumulator, the
  full-k loop, and the four FMAs per (out, row)); ``:363``-``:408`` (the 32
  reductions and their barriers). Iteration 129's arithmetic is corrected here.

  **Iteration 131: register-limited occupancy is REFUTED -- Gemma's grouped owner
  uses 55 VGPRs, about 9 blocks per SIMD.**

  Extracted the device code object from the family's host ``.so`` and read its
  notes. Three tooling obstacles, all worked around: ``roc-obj-ls`` is broken on
  this host ("Can't locate File/Which.pm in @INC"); ``/tmp`` is full, so the first
  ``objcopy --dump-section`` failed with "No space left on device"; and the
  extracted ``.hip_fatbin`` is a ``__CLANG_OFFLOAD_BUNDLE__`` container whose ELF
  members are nested, so ``llvm-readobj --notes`` on the container finds nothing.
  Parsing the container directly (24-byte magic, ``uint64`` count, then
  ``(offset, size, name_len, name)`` entries) yields a plain uncompressed gfx1100
  ELF of 1,635,344 bytes.

  **The register ladder for the grouped row-batch kernel**, by template argument::

      <8, 1, 0, 0, 0, 0>   41 VGPR     <8, 4, 1, 1, 1, 0>  173 VGPR
      <8, 4, 0, 0, 0, 0>   42 VGPR     <8, 4, 1, 1, 1, 1>  212 VGPR
      <8, 4, 1, 0, 0, 0>   55 VGPR   <- Gemma's (AMORTIZE_INPUT)
      <8, 4, 1, 1, 0, 0>   66 VGPR   <- the bundle variant tested in iteration 128

  **Gemma's owner is the 55-VGPR instantiation.** 55 x 128 threads = 7,040 VGPRs
  per block against 65,536 per SIMD, so roughly **9 blocks per SIMD** -- the
  opposite of the guide's section 5.3 worst case. **Occupancy is not this kernel's
  limit**, and the 28 instantiations sitting at the 256-VGPR hard cap belong to
  other specializations, not this one.

  **This also confirms iteration 128's reading of the bundle experiment.** The
  bundled variant is the 66-VGPR instantiation, only 11 registers above Gemma's,
  so its 2.75x slowdown cannot have been a register or occupancy effect -- it was
  the collapse of the final reduction onto 16 of 128 threads, exactly as recorded.

  **Seven explanations are now excluded** for the MoE's ~2 us per routed row:
  bandwidth, the weights, call overhead, rows-per-expert, per-row-batch dequant,
  barrier count, and register-limited occupancy. The stall is real and measured
  (176 FMAs per thread per row against ~5000 cycles, about 3.5% of FMA issue
  potential), and its mechanism remains unidentified.

  Evidence: the extraction above; the notes table; ``gguf_q4_k_selected_prefill.hip``
  template parameters at ``:223``. Tooling note for future profiling runs:
  ``roc-obj-ls`` and the ``/tmp`` volume are both unusable on this host, and the
  bundle must be parsed manually.

  **Iteration 132: the compensated-WMMA ruling is MOOT -- it buys nothing -- and
  the real lever is `wmma_plain` at +45.6%, whose cost is fp16 weight rounding.**

  Measured all three prefill routes end-to-end at 1024 prompt / 128 output through
  the documented selector ``HIPENGINE_GEMMA4_MOE_PREFILL``, three samples each::

      mode          prefill tok/s          vs default
      auto          687 / 685 / 684        --
      wmma          689 / 687 / 686        +0.3%   (compensated, kl_max 0.0609)
      wmma_plain    999 / 996 / 991       +45.6%   (uncompensated, fp16 weight rounding)

  **The compensated WMMA arm delivers no speedup.** The arm whose ``kl_max``
  breach (1 row of 1023 at 0.0609 against 0.05) has been recorded as "an open lead
  decision" is worth **+0.3%**, which is noise. **The ruling is therefore moot**:
  there is nothing to gain from enabling it, so it should stay off on merit rather
  than on the strength of the bar. That removes it from the lead's queue.

  **The speed is in `wmma_plain`**, which the selector's own comment describes as
  "the uncompensated form, kept as the diagnostic that isolates the fp16
  weight-rounding term". Its cost is therefore **genuine fp16 weight rounding**,
  not reduction association -- which is exactly why compensating for that rounding
  gives the entire gain back. At **686 -> 999 tok/s (+45.6%)** it is the largest
  single measured prefill lever in this campaign.

  **This changes what the lead is actually being asked.** The recorded question is
  "is an absolute ``kl_max`` bar applicable to a reordering-class change?" -- but
  the reordering-class arm is free of charge at +0.3%, so that question no longer
  gates anything. The live question is **"do we accept fp16 weight rounding in the
  MoE, as llama.cpp does?"** -- a quality decision with a **45.6%** price attached
  to the conservative answer.

  **A correction to the source comment.** ``gemma4_experts.py:564`` states the
  WMMA owners "are 2.7x faster on a 1024-token prefill" without distinguishing the
  two forms. Measured, the compensated form is **1.00x** and the plain form is
  **1.46x**. The 2.7x figure is not reproduced by either at this shape.

  **What the campaign still owes on `wmma_plain`**: its own logits gate. The
  compensated form's numbers are recorded (kl_mean/kl_p95/kl_p99 pass with 10-200x
  margin, top-1 unchanged on all 1023 rows); the plain form's are not, and the
  campaign records it only as "genuine fp16 weight rounding" without a measurement.
  That gate is a concrete, runnable action and it is the direct path to the
  objective's "let's get it fast".

  Evidence: the three-mode table above (``scripts/gemma4_campaign_bench.py
  --prompt 1024 --output 128 --samples 3``, ``--expect-gpu W7900``);
  ``gemma4_experts.py:558``-``:576`` (the route comment) and ``:589``
  (``_prefill_route_flags``).

  **Iteration 133: `wmma_plain` PASSES the campaign's logits gate, and its kl_max
  is 58x BETTER than the compensated form's -- the fast arm is also the accurate one.**

  Ran ``scripts/gemma4_teacher_forced_gate.py`` capture/gate at ``--prompt 2048
  --prefill 1024``. The explicit ``--prefill`` matters: it defaults to 0, which
  makes the chain decode-only, and that default is what made the iteration-107 gate
  run blind to its own prefill changes. This run's provenance records
  ``prefill: 1024``, so the prefill path is demonstrably exercised.

      metric      measured     bar      margin
      kl_max      0.00104      0.05      48x under
      kl_mean     6.90e-06     0.001    145x under
      kl_p95      1.85e-05     0.005    271x under
      kl_p99      1.60e-04     0.02     125x under
      top1_flips  0 / 1023     --       perfect

  The divergence is **non-zero**, so the candidate genuinely differed from the
  baseline and the comparison is meaningful; a self-comparison would report
  ``kl_max`` of exactly 0.0.

  **`wmma_plain`'s kl_max (0.00104) is 58x better than the compensated form's
  recorded 0.0609.** The arm that pays +45.6% in speed is also the more accurate
  one, which makes the compensated form **dominated on both axes** at this shape:
  not faster (+0.3%) and with a worse recorded divergence. There is no configuration
  in which enabling the compensated arm is the right call.

  **Caveat on the comparison.** The recorded 0.0609 does not state its ``--prefill``.
  If it came from a default-``--prefill`` run it measured a different (decode-only)
  path than this one. This measurement is the one whose provenance states
  ``prefill: 1024``, so it is the one with the prefill path demonstrably exercised.

  **``promotion_qualified: false``.** The campaign's logits gate is necessary but
  not sufficient; the full execution-profile gate in ``docs/EXECUTION-PROFILES.md``
  is still owed before this becomes the default path. That is the remaining work
  item, it is runnable, and it is the last step between the objective's "let's get
  it fast" and a +45.6% prefill.

  Evidence: ``scripts/gemma4_teacher_forced_gate.py capture|gate`` at
  ``--prompt 2048 --prefill 1024``, baseline ``auto`` vs candidate
  ``HIPENGINE_GEMMA4_MOE_PREFILL=wmma_plain``; verdict at
  ``$HOME/.cache/hipengine/tmp/gate_verdict.json``.

  **Iteration 134: the grouped int8 MMQ is 2.07x the bf16 WMMA owner at the Gemma4
  MoE shape -- measured, not inferred. The MoE's structural gap is now identified.**

  The dense q8_0 control reaches 21.7 TF/s through ``gguf_q8_0_mmq_prefill``, a
  true int8xint8 MMQ. Gemma4's MoE registers **only** bf16-dequant prefill owners
  (``selected_dual_grouped_rowbatch8_bf16_bf16_out``, ``selected_*_wmma_*_bf16_*``)
  and imports **none** of the int8 MMQ kernels -- zero references. Meanwhile the
  qwen4 runner already calls the grouped int8 MMQ path
  (``gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out`` at
  ``qwen4_exp_runner.py:4200``), and ``gguf_q4_k_q8_1_selected_prefill.hip`` exports
  a whole family of grouped variants
  (``..._ds4_mmq32_prefill_compact32_bf16_bf16_out``,
  ``..._mmq64x32_...``, ``..._mmq64x64_rowvec_...``, plus x8/t16 siblings) that take
  the same grouped ABI Gemma4's MoE already uses: ``(activations, expert_start,
  weights, out, compact_rows, experts, in_features, out_features, planes)``.

  Measured at the Gemma4 MoE shape with the existing harness
  (``scripts/gguf_q4_k_t16_selected_prefill_microbench.py --hidden 2816
  --out-features-a 1408 --out-features-b 1408 --experts 128 --rows-per-expert 32``,
  i.e. compact=4096, k=2816, n=1408, e=128)::

      mode                     logical TF/s   ms/call   vs current
      selected-wmma (bf16)        8.06         8.06       --
      q8-1-ds4-mmq32             16.66         3.90      2.07x
      q8-1-ds4-mmq32-pack        16.12         4.03      2.00x
      q8-1-ds4-wmma32-pack       11.97         5.43      1.49x

  **The int8 MMQ32 is 2.07x the bf16 owner**, and **2.00x even when paying the
  BF16->DS4 activation-packing cost** -- the pack is nearly free (0.13 ms of 4.03).
  The integer-WMMA32 form is a real but smaller 1.49x.

  **Correction to iteration 71's estimate.** I put the MoE's remaining headroom at
  ~3.6x by comparing against the dense q8_0 control's 21.7 TF/s. That control has a
  different ``n`` and shape. Measured against the bf16 owner at the MoE's **own**
  shape, the available factor is **2.07x**.

  **Projected end-to-end.** Within the MoE, gate_up is ~2/3 of the FLOPs
  (2 x 2816 x 2816 per row) and down ~1/3 (2 x 1408 x 2816). Both quants have
  grouped MMQ kernels. If both take ~2x, the MoE takes ~2x, and prefill
  (0.27 dense + 0.73 MoE) becomes 0.27 + 0.365 = 0.635 -> **~1.57x overall**,
  i.e. **999 -> ~1570 tok/s** at 1024p/128o.

  **What the port requires**, in order: build ``gguf_q4_k_q8_1_selected_prefill``
  and ``gguf_q5_1_mmq_selected_prefill`` in the Gemma4 runtime; pack activations
  BF16->DS4 with the existing ``gguf_q8_1_mmq_ds4_pack_bf16`` GPU kernel; allocate
  the ds4 workspace; call the grouped MMQ kernels. The ``expert_start``/compact-row
  plumbing already exists because the current grouped owner uses it.

  Evidence: the four-mode table above; ``gemma4_experts.py:503``-``:556`` (the
  registered bf16-only variant set); ``gguf_q4_k_q8_1_selected_prefill.py:22``-``:99``
  (the exported grouped symbol family); ``qwen4_exp_runner.py:4195``-``:4210`` (the
  working qwen4 integration template).

  **Iteration 135: the grouped int8 MMQ port design is fully determined -- every
  ABI argument maps onto existing Gemma4 buffers except two.**

  Target: ``gguf_q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32_bf16_bf16_out``
  (``gguf_q4_k_q8_1_selected_prefill.py:1668``) for gate_up, and
  ``gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out`` for down.

  **The MMQ32 tile ABI is the same 16-row tile plan the WMMA route already builds.**
  The microbench constructs both from one call (``_make_uniform_compact_metadata``
  at ``gguf_q4_k_t16_selected_prefill_microbench.py:108``, invoked for the MMQ32
  path at ``:262``) and asserts the compact row counts agree (``:267``). So Gemma4's
  existing ``wmma_expert_start`` (``gemma4_experts.py:138``), ``wmma_tile_expert``
  (``:139``) and ``wmma_rows`` (returned by ``_build_wmma_tile_plan`` at ``:266``)
  are directly reusable -- no second tile plan is needed.

  **Activation indexing.** The MMQ32 kernel body passes ``x_q8`` together with
  ``compact_to_source``, ``expert_start_compact``, ``expert_start_mmq32`` and
  ``tile_expert``, and forwards ``compact_rows`` twice (compact and source rows).
  It walks padded tiles, maps each tile to its expert, takes that expert's compact
  row range, and reads the activation as ``x_q8[compact_to_source[compact_row]]``.
  With Gemma4's compact buffer already in source order, ``compact_to_source`` is the
  **identity iota** -- matching the microbench, which builds
  ``np.arange(compact_rows, dtype=np.int64)`` at ``:236``.

  ABI map::

      MMQ32 argument          Gemma4 source
      x_q8                    NEW  ds4 workspace, compact_rows * (in/128) * 144 B
      compact_to_source       NEW  int64 identity iota, compact_rows * 8 B
      expert_start_compact    existing  expert_start
      expert_start_mmq32      existing  wmma_expert_start
      mmq_tile_expert         existing  wmma_tile_expert
      mmq_total_rows          existing  wmma_rows
      qweight_a / qweight_b   existing  weight raw base and base + half_bytes
      out_ptr                 existing  gate_up_out

  **Buffer sizing.** ``block_q8_1_mmq_ds4`` is ``uint16_t ds4[8]`` + ``int8_t
  qs[128]`` = **144 bytes per 128 values** (``gguf_q4_k_q8_1_selected_prefill.hip:69``,
  ``Q8_1_MMQ_BLOCK = 128``). For the MoE shape that is 4096 * 22 * 144 = 13.0 MB,
  against a 23 MB bf16 ``packed_hidden`` -- so the workspace is not a memory concern.

  **Remaining implementation steps**, all mechanical now: (1) add the two scratch
  buffers; (2) add ``gemma4_project_experts_mmq_dual`` that packs ``packed_hidden``
  through ``gguf_q8_1_mmq_ds4_pack_bf16`` (``gguf_q4_k_q8_1_selected_prefill.py:372``)
  and calls the MMQ32 leaf; (3) add the route flag to ``_prefill_route_flags`` /
  ``_PREFILL_MODES``; (4) build the library in ``hipengine/runtime/gemma4.py``
  (which today builds only ``gguf_q8_0_mmq_prefill``, ``:676``-``:691``); (5) gate at
  ``--prompt 2048 --prefill 1024`` (``--prefill`` defaults to 0 = decode-only) and
  measure with ``gemma4_campaign_bench.py``.

  **Iteration 136: the grouped int8 MMQ route is 30% faster and WRONG -- two
  corrections and a precise root cause.**

  Implemented the port from iteration 135 and measured it end-to-end at 1024p/128o
  through ``HIPENGINE_GEMMA4_MOE_PREFILL=mmq``::

      mode    prefill tok/s        vs default
      auto    687 / 685 / 684      --
      mmq     895 / 894           +30.4%

  Close to the projection from gate_up alone (2.07x on the 2/3 of MoE FLOPs that
  gate_up owns predicts ~1.34x -> ~919 tok/s), and ``path parity=True``.

  **The campaign logits gate then failed, and not marginally::**

      metric      measured    bar
      kl_max      25.54       0.05
      kl_mean     0.777       0.001
      kl_p95      5.73        0.005
      kl_p99      12.61       0.02
      top1_flips  125 / 1023  --
      top1_rate   0.878       0.99

  ``kl_max`` 25.5 is not a precision loss, it is the wrong function. **The +30.4%
  is invalid and the route does not ship.**

  **Correction 1 -- the tile plan is NOT shared.** Iteration 135 concluded the
  MMQ32 leaf reuses the WMMA 16-row tile plan, because the microbench builds both
  through one ``_make_uniform_compact_metadata`` call and asserts
  ``mmq32_compact_rows == compact_rows``. That assert compares *compact* row counts,
  which agree trivially; the MMQ32 call passes ``tile_rows=32`` while the WMMA call
  takes the 16 default. Feeding the 16-row plan to the leaf produced an immediate
  GPU page fault. The 32-row map already exists as ``qwen35_moe_mmq32_tile_map``
  (``group_scatter.py:423``), and the qwen4 runner selects it with
  ``tile_rows = 32 if q4_k_mmq_prefill else 16`` (``qwen4_exp_runner.py:3781``).
  The leaf also wants the *actual* padded total read back from the device, not the
  allocation's upper bound, which is what the WMMA path passes.

  **Correction 2 -- the root cause of the wrong output is the weight layout.** The
  MMQ32 leaf takes ``qweight_a`` and ``qweight_b`` with no stride parameter, so it
  derives each expert's stride from that matrix's output width and therefore
  requires **two separate per-expert allocations**. The qwen4 runner passes exactly
  that: ``weights["expert_gate"]`` and ``weights["expert_up"]``, distinct buffers.
  Gemma4 instead loads **one fused ``ffn_gate_up_exps`` tensor**
  (``runtime/gemma4.py:108``), which the grouped owners handle through an explicit
  ``expert_stride_rows=fused_width``. Passing ``base`` and ``base + half_bytes``
  makes the leaf read expert 0's up half as expert 1's gate. Weights are *not* the
  problem: the microbench copies raw Q4_K for this mode, so no repack is involved.

  **State of the tree.** The scaffold is in place and verified -- the ``ds4_q8``
  and ``compact_to_source`` scratch buffers, the identity-iota fill, the 32-row
  ``_build_mmq_tile_plan``, and ``gemma4_project_experts_mmq_dual``. The route
  itself refuses with an explicit ``NotImplementedError`` rather than returning
  those logits, gated on ``_MMQ_DUAL_WEIGHTS_ARE_SPLIT``. ``auto`` and every other
  mode are unchanged.

  **Clearing condition**: split ``ffn_gate_up_exps`` into per-expert gate and up
  allocations at load, then flip ``_MMQ_DUAL_WEIGHTS_ARE_SPLIT``. That is the only
  remaining step before the +30% (and, with the q5_1 down leaf, the projected
  ~1.57x) is available.

  Evidence: the two mode tables and the gate verdict above;
  ``$HOME/.cache/hipengine/tmp/gate_mmq_verdict.json``;
  ``gemma4_experts.py`` (``_build_mmq_tile_plan``, ``gemma4_project_experts_mmq_dual``,
  ``_MMQ_DUAL_WEIGHTS_ARE_SPLIT``); ``qwen4_exp_runner.py:3747``-``:3815`` (the
  working split-weight MMQ32 integration).

  **Iteration 137: the grouped int8 MMQ route works and passes the logits gate --
  +29.2% prefill, zero top-1 flips.**

  The iteration-136 failure is fixed. Splitting the fused ``ffn_gate_up_exps`` stack
  into separate per-expert gate and up resident tensors gives the int8 MMQ leaf the
  weight stride it derives from its own output width, and the route now computes the
  right function::

      mode    prefill tok/s        vs default
      auto    687 / 685 / 684      --
      mmq     886 / 885           +29.2%

  Campaign logits gate at ``--prompt 2048 --prefill 1024``, baseline ``auto``::

      metric      measured    bar      margin
      kl_max      0.00503     0.05      9.9x under
      kl_mean     1.02e-05    0.001    98x under
      kl_p95      1.87e-05    0.005   267x under
      kl_p99      9.68e-05    0.02    207x under
      top1_flips  0 / 1023    --       perfect

  ``passed: true``, and the divergence is non-zero, so the candidate genuinely
  differed and the comparison is meaningful. Against iteration 136's failure this is
  a factor of **5080x** on ``kl_max`` (25.54 -> 0.00503), which is the signature of a
  layout bug rather than an arithmetic one.

  **The split is additive and route-gated.** ``raw`` stays resident so every other
  prefill route keeps reading the allocation it always read -- splitting it
  unconditionally broke ``gemma4_project_experts_grouped_dual`` with a ``KeyError``
  on ``allocation("raw")``, which is the exact regression the "works in the harness
  is not works" rule warns about. Gate and up are *additional* allocations, built
  only when ``HIPENGINE_GEMMA4_MOE_PREFILL=mmq`` selects the route that needs them,
  because they cost a second copy of the largest expert tensor.

  The split is a host-side gather of the contiguous per-expert halves, matched on
  shape (rank-3, middle dim ``2 * expert_ff``) rather than on slot name so the raw
  device materializer and the reference materializer cannot drift apart about which
  tensor is the fused one.

  **Still owed: the execution-profile gate.** ``promotion_qualified: false``. The
  campaign logits gate is necessary but not sufficient, so the route stays opt-in
  and off the default path until that gate runs. That is the remaining step, and it
  is the last one before this lands enabled.

  Evidence: the two mode tables and the gate verdict above;
  ``$HOME/.cache/hipengine/tmp/gate_mmq2_verdict.json``;
  ``gemma4_gguf_device.py`` (``_mmq_split_requested``, ``_is_fused_expert_gate_up``,
  the additive split in ``materialize_gemma4_gguf_device_weight``);
  ``gemma4_experts.py`` (``_MMQ_DUAL_WEIGHTS_ARE_SPLIT``, ``_build_mmq_tile_plan``,
  ``gemma4_project_experts_mmq_dual``).

  **Iteration 138: the down projection cannot use this lever -- its k is not
  DS4-block divisible. +29.2% is the ceiling for the int8 MMQ route.**

  Added ``gemma4_project_experts_mmq`` for the Q5_1 down leaf and measured: **no
  change** (888/884 tok/s against 886/885). The route is not running, and the
  reason is a shape contract, not a bug.

  The GGUF shapes are::

      blk.0.ffn_gate_up_exps.weight  Q4_K  (128, 1408, 2816)   expert_ff = 704
      blk.0.ffn_down_exps.weight     Q5_1  (128, 2816, 704)    in_features = 704

  So the down's ``in_features`` is **704**, and the DS4 MMQ block is **128** values
  -- ``704 % 128 = 64``. ``gguf_q5_1_mmq_ds4_selected_prefill_bf16_bf16_out``
  requires ``in_features % 128 == 0``, so it **cannot serve this shape at all**.
  The helper's shape guard returns ``False`` and the exact grouped owner serves the
  down as before, which is why the output is unchanged and correct.

  Verified rather than assumed: the gate re-run after adding the helper is
  ``passed: true`` with ``kl_max`` bit-identical to the pre-change verdict
  (0.005029002284065986), confirming the helper is a true no-op at this shape.

  The only grouped Q5_1 prefill owners in the tree are this MMQ leaf and a
  decode-oriented ``gguf_q5_1_selected_pack8_gemv``, so there is no alternative
  grouped owner to reach for. **The +29.2% from gate_up is the whole of what the
  int8 MMQ lever yields on Gemma4's MoE.**

  **Where the remaining MoE headroom is.** With gate_up on the int8 MMQ leaf and
  down on the exact grouped owner, the down is now the larger share of what is
  left: it is ~1/3 of MoE FLOPs at a measured ~1.9 TF/s against gate_up's ~4.1
  before the MMQ route and ~8.5 after. A down-specific lever would need either a
  DS4-block-divisible activation layout for k=704 or a grouped prefill owner built
  for a half-block tail -- neither exists today.

  Evidence: the unchanged mode table; the GGUF tensor shapes above;
  ``gate_mmq3_verdict.json`` (identical ``kl_max`` to ``gate_mmq2_verdict.json``);
  ``gguf_q5_1_mmq_selected_prefill.py`` (the ``in_features % 128`` contract).

  **Iteration 139: `wmma_plain` STRICTLY DOMINATES the grouped int8 MMQ route on
  both axes -- the int8 route is retired as a candidate.**

  Same-session comparison at 1024p/128o, three samples each::

      route         prefill tok/s   kl_max    top1_flips
      auto          670 - 677       --        --
      wmma_plain    987 - 990       0.00104   0
      mmq           870 - 872       0.00503   0

  ``wmma_plain`` is **faster (990 vs 871) and more accurate (kl_max 0.00104 vs
  0.00503)** than the int8 MMQ route. There is no axis on which the int8 route
  wins, so it is not a candidate and should not be developed further.

  **Why the microbench misled the decision.** Iteration 134 measured the int8
  MMQ32 leaf at 2.07x the BF16 WMMA owner, and that measurement was correct -- for
  the *kernel alone*, at the same shape, with metadata already resident. The route
  as built adds costs the microbench did not have:

  - ``_build_mmq_tile_plan`` does a **device->host copy plus a stream
    synchronize on every call**, because the leaf sizes its grid from the actual
    padded total. The WMMA owners pass the routing-independent upper bound and
    never sync. Across ~60 layer-calls per 1024-token prefill that is the bulk of
    the gap.
  - Activation packing (BF16 -> DS4) runs per call, which the microbench's
    ``-pack`` mode did model but only as 0.13 ms against a 4.03 ms kernel.
  - The int8 path also quantizes activations, so its arithmetic differs more
    (kl_max 5x worse) for a speed loss.

  **The lesson is about the measurement, not the kernel**: a leaf that wins its
  microbenchmark can still lose end-to-end once its real per-call metadata and
  packing costs are included. Compare routes end-to-end before committing to one.

  **What the int8 route would still need to be competitive**: the per-call sync
  removed (cache the padded total, or establish that the leaf tolerates the upper
  bound the way the WMMA owners do). Even then it would only tie ``wmma_plain`` on
  speed while remaining 5x worse on ``kl_max``, so the fix is not worth pursuing
  for this model.

  **The winning route and what blocks it.** ``wmma_plain`` at **990 tok/s
  (+45.6% over auto)** with ``kl_max`` 0.00104 and zero top-1 flips is the best
  measured prefill configuration. It routes *both* projections through the WMMA
  owners (``down_wmma = gate_up_wmma`` in ``gemma4_experts_forward_bf16``), which
  is where its gain comes from. It remains opt-in because
  ``promotion_qualified`` is false: per ``EXECUTION-PROFILES.md`` section 2.9,
  "production arithmetic changes and published quality/performance claims still
  require the applicable gates in this document". The campaign logits gate is
  necessary but not sufficient; the execution-profile gate
  (``scripts/execution_profile_gate.py``, artifact-driven: variant and strict
  manifests, strict/candidate captures, controls, repeat/isolation/batch-invariant
  captures, task results, and an arithmetic class) is the named clearing command.

  Evidence: the three-route table above;
  ``gate_mmq2_verdict.json`` and the ``wmma_plain`` verdict (iteration 133);
  ``gemma4_experts.py`` (``_build_mmq_tile_plan`` for the per-call sync,
  ``_prefill_route_flags`` for ``down_wmma = gate_up_wmma``).

  **Iteration 140: removing the MMQ per-call synchronize faults; reverted. The
  ~0.4 s overhead is real but not yet explained.**

  Iteration 139 left an unexplained arithmetic fact: the int8 route measured 1.17 s
  where its 2.07x gate_up advantage predicts ~0.77 s, so roughly 0.4 s sits in
  per-call overhead -- consistent with a device-to-host readback plus a full stream
  synchronize on every one of ~60 layer-calls per 1024-token prefill. Removing that
  would put the route near 1330 tok/s, well past ``wmma_plain``'s 990, so it was
  worth chasing.

  **The attempt.** ``_build_mmq_tile_plan`` read the padded total back from the
  device because the leaf sizes its grid from it (``row_tiles = mmq_total_rows /
  32``). The map kernel fills ``tile_expert`` with ``-1`` across the whole capacity
  before writing real entries, and the leaf returns early on any tile whose expert
  is negative (``if (expert_id < 0 || expert_id >= num_experts) return;``). So
  passing the allocation's routing-independent upper bound looked safe: extra tiles
  resolve to the sentinel and exit. ``upper_rows`` is ``upper_tiles * 16``, so
  ``upper_rows / 32`` is half the sentinel-filled capacity, which also checks out.

  **It faults.** With the upper bound passed, the route dies immediately with
  ``Memory access fault by GPU node-1 ... Page not present or supervisor
  privilege``. The paper argument above is not sufficient, and the actual cause was
  not identified within this iteration.

  **Reverted** to the readback version, which reproduces its gated numbers
  (872/867 tok/s against the 870-872 measured in iteration 139). The docstring now
  records that the upper-bound substitution looks safe and is not, so the next
  attempt starts from the failure rather than repeating the reasoning.

  **What is still worth checking next time**, in order: whether the leaf's row
  addressing for the activation uses the *padded* base from
  ``expert_start_mmq32`` rather than the expert's compact range -- a padded base
  plus a 32-row tile can run past ``compact_rows`` and off the end of ``ds4_q8``,
  which would fault on the activation rather than the tile map, and would not show
  up in the grid arithmetic; and whether the map kernel's real-entry writes can
  exceed ``tile_capacity`` when the plan is 32-row while the capacity was sized for
  16-row tiles.

  **The overhead stands as the largest known unexploited win on this route**: about
  0.4 s of a 1.17 s prefill, or ~+50% on the route, which would make the int8
  route faster than ``wmma_plain`` rather than slower. It is worth returning to,
  but only with the fault understood.

  Evidence: the fault output above; the reverted numbers (872/867);
  ``group_scatter.hip`` (``qwen35_moe_wmma_tile_map_kernel`` sentinel fill at the
  capacity loop, ``launch_selected_dual_q8_1_ds4_mmq32_compact32`` grid arithmetic
  ``row_tiles = mmq_total_rows / 32``); ``gguf_q4_k_q8_1_selected_prefill.hip``
  (the leaf's negative-expert early return).

  **Iteration 141: the sync removal works, the gate is bit-identical, and the
  0.4 s overhead I chased was a mirage from a bad estimate.**

  Iteration 140 reverted this change and recorded that passing the upper bound
  "looks safe and is not". **That verdict was wrong, and so was the premise.** The
  bound was invalid, but not for the reason I assumed, and the fault was a
  *symptom* of passing a value that was too small rather than evidence the
  substitution is unsafe.

  **The algebra error.** ``_wmma_tile_upper_bound`` returns a bound for **16-row**
  tiling. I assumed a 32-row plan could only be smaller. It cannot: 32-row padding
  is *larger* than 16-row padding for any count that is not a multiple of 32.

      c = 33:  ceil(33/32)*32 = 64   >   ceil(33/16)*16 = 48

  So ``upper_rows`` (6016) is not a ceiling for the 32-row plan, and the map
  kernel wrote more rows than the grid was told to launch. Measured directly with
  an instrumented build:

      [mmq-plan] real_total=6304  bound=6016  real_tiles=197  bound_tiles=188
                 capacity_tiles=376  lanes=4096

  ``real_total > bound`` is the violation, on the first prefill layer, in every
  sample. The correct routing-independent bound is ``sum(c_e + 31) = lanes +
  31 * num_experts``, which gives 8064 rows / 252 tiles against a 16-row capacity
  of 376 tiles -- so the sentinel fill still covers the grid, and the extra tiles
  resolve to ``-1`` and exit.

  **The fix works and is exactly correct.** With ``lanes + 31 * num_experts`` passed
  and the readback plus ``stream_synchronize`` removed:

      metric      measured     bar      margin
      kl_max      0.00503      0.05     10x under
      kl_mean     1.02e-05     0.001    98x under
      kl_p95      1.87e-05     0.005    267x under
      kl_p99      9.68e-05     0.02     207x under
      top1_flips  0 / 1023     --       perfect

  ``kl_max`` is **bit-identical** to the readback version's recorded
  ``gate_mmq3_verdict.json`` (0.005029002284065986), which is the strongest
  available evidence that the bound only adds skipped sentinel tiles and does not
  change the arithmetic. The ``--prefill 1024`` recipe was used, so the prefill
  path is demonstrably exercised.

  **The 0.4 s was not the sync.** Speed went 872/867/870 -> 885/882/876 tok/s, about
  **+1.5%**, not the ~+50% predicted. The prediction came from comparing the route's
  measured 1.17 s against a 0.77 s figure derived from the isolated 2.07x gate_up
  advantage -- an extrapolation from one kernel's microbenchmark to the whole
  layer, which does not hold. **The per-call synchronize was worth ~1.5%, not 34%.**
  Iteration 139's "largest known unexploited win in the campaign" claim is withdrawn:
  the route's gap to its own prediction is real but is not in the plan bookkeeping,
  and the next place to look is the pack kernels and the down projection rather than
  the tile plan.

  The change is kept: it is bit-identical on the gate, 1.5% faster, and removes a
  per-call device round-trip.

  ``promotion_qualified: false`` still, and the int8 route remains behind
  ``wmma_plain`` (885 vs 990), so this does not change which arm leads.

  Evidence: the instrumented ``HIPENGINE_GEMMA4_MMQ_PLAN_DEBUG`` printout above;
  ``scripts/gemma4_teacher_forced_gate.py capture|gate`` at ``--prompt 2048
  --prefill 1024`` against ``$HOME/.cache/hipengine/tmp/gate_base_mmq.npz``;
  ``gemma4_experts.py::_build_mmq_tile_plan``.

  **Iteration 142: the int8 route was never losing to overhead -- it was one
  pathological kernel. Fixing it gives 1393 tok/s, +40.7% over the previous best.**

  Iteration 139 measured the int8 route 2.07x faster than the bf16 WMMA owner on
  the gate_up in isolation, yet the route came out slower end to end (1.17 s
  against 1.03 s). Iterations 139 and 141 both guessed at where that went: first
  the per-call synchronize, then the activation pack. **Both guesses were wrong,
  and profiling settled it in one run.**

  **The per-kernel diff.** Profiled both arms with ``rocprofv3 --kernel-trace`` on
  the W7900 at ``--prompt 1024 --output 8`` and diffed by kernel. The int8 route
  adds exactly three kernels:

      wmma ms    mmq ms    delta  kernel
        714.0       0.0   -714.0  q4_k_selected_dual_wmma_prefill_compact   (gate_up)
          0.0     247.1   +247.1  q4_k_selected_dual_q8_1_ds4_mmq32          (gate_up)
          0.0      12.3    +12.3  q8_1_mmq_ds4_pack_bf16                     (pack)
        115.8       0.0   -115.8  q5_1_selected_grouped_wmma_prefill_bf16    (down)
          0.0     947.3   +947.3  q5_1_selected_grouped_prefill_pair2        (down)

  **The int8 gate_up is 2.89x the bf16 owner -- better than the 2.07x measured in
  isolation. The pack costs 12.3 ms, which is nothing.** The entire loss is the
  down projection: the MMQ leaf runs at **947 ms against 116 ms over the same 116
  launches** -- 8.2x slower, 4.0 against 32.5 TFLOP/s at Gemma's down geometry
  (8192 compact rows, 128 experts, in 704, out 2816).

  Net: the route wins 467 ms on the gate_up and loses 831 ms on the down, for the
  +294 ms observed. That matches the measured delta exactly, so the account is
  closed rather than merely plausible.

  **The fix.** The MMQ gate_up already writes bf16, which is exactly what the WMMA
  down reads, so the down keeps the WMMA owner on the int8 route and only the
  gate_up uses the int8 leaf. The two tile plans share buffers and tile different
  widths (32 rows against 16), so the 16-row plan is rebuilt after the gate_up has
  finished with the 32-row one.

  **Result: 1393 / 1387 / 1384 tok/s** against ``wmma_plain``'s 990 and the int8
  route's own 885 -- **+40.7% over the previous best**, prefill 1.02 s to 0.73 s.

  **It is also more accurate, not less:**

      metric      measured     bar      margin
      kl_max      0.001341     0.05     37x under
      kl_mean     8.76e-06     0.001    114x under
      kl_p95      2.45e-05     0.005    204x under
      kl_p99      1.60e-04     0.02     125x under
      top1_flips  0 / 1023     --       perfect

  ``kl_max`` improves on the int8 route's own recorded 0.005029 by 3.7x, because
  the WMMA down is the more accurate of the two. The arm that is 40% faster is also
  the more accurate one, so this is a dominant improvement on both axes.

  **A fault worth recording, because it was self-inflicted and instructive.** The
  first attempt faulted with a GPU memory access error. Setting ``down_wmma`` true
  under mmq also enabled the *pre-gate_up* plan build, whose guard was
  ``(gate_up_wmma or down_wmma)`` -- so the 16-row plan overwrote the 32-row plan
  before the gate_up ran, and the MMQ leaf read 16-row starts as 32-row. The guard
  is now ``gate_up_wmma`` alone, since the down's plan is built later by
  construction. A flag that means "the down wants a plan" was also being read as
  "build the plan now", and those are different questions.

  Distance to the same-artifact llama.cpp comparator narrows from 3.9x to
  **2.81x** (3910 against 1393).

  ``promotion_qualified: false``: the campaign's logits gate is necessary but not
  sufficient, and the full execution-profile gate in ``docs/EXECUTION-PROFILES.md``
  is still owed. That remains the last step between this and a default path.

  Evidence: ``rocprofv3 --kernel-trace`` diffs for both arms at
  ``--prompt 1024 --output 8`` on ``ROCR_VISIBLE_DEVICES=0`` (W7900);
  ``gemma4_campaign_bench.py --prompt 1024 --output 128`` for both;
  ``scripts/gemma4_teacher_forced_gate.py gate --prompt 2048 --prefill 1024``
  against ``$HOME/.cache/hipengine/tmp/gate_base_mmq.npz``;
  ``gemma4_experts.py`` (the ``down_wmma`` assignment, the plan-rebuild guard, and
  the down dispatch order).

  **Iteration 143: the default path is 675 tok/s while 1393 is available. Every
  fast route is off-default, and the execution-profile packet is the single unlock
  for all of them.**

  Measured the default path for the first time -- ``gemma4_campaign_bench.py`` with
  no ``HIPENGINE_GEMMA4_MOE_PREFILL`` set, which is what ``hipengine.LLM.generate()``
  and ``hipengine serve`` reach:

      route           prefill tok/s
      auto (default)      675
      wmma_plain          990
      mmq + wmma down    1393

  **``_prefill_route_flags("auto")`` returns ``(False, False, False)``**, so the
  default reaches none of the tuned owners at all; it falls through to the exact
  grouped/selected routes. A user today gets **2.06x less** than the engine can
  already do, and 5.5x less than the same-artifact llama.cpp comparator.

  **Why the fast routes stay off.** All three sit behind the same gate. Per
  ``EXECUTION-PROFILES.md`` section 2.9, a changed-arithmetic path cannot become the
  default without its execution-profile gate, and that gate is the named clearing
  command. The campaign logits gate is necessary but not sufficient, so a clean
  ``kl_max`` does not lift this on its own.

  **The packet does not exist for Gemma4.** ``scripts/execution_profile_gate.py`` is
  artifact-driven and needs a variant manifest, a strict manifest, strict and
  candidate captures, both expected-controls sets, repeat, isolation and
  batch-invariant captures, comparison controls, task results and an arithmetic
  class. ``docs/TESTING.md`` shows a Qwen3.6 packet built by adapting
  ``scripts/quant_quality/qwen36_teacher.py`` through
  ``scripts/qwen36_execution_profile_adapter.py``; **Gemma4 has no analogue.**
  ``scripts/execution_profile_gguf_int8_direct_prefill_gate.py`` is *not* reusable
  here -- it gates the Qwen3.6 attention/KV int8-read route
  (``--candidate {int8_direct_prefill,slot_local_aotriton}``), not a grouped int8
  MoE weight route.

  **So promotion is a build, and it is the highest-value remaining work on this
  campaign** -- larger in user-facing terms than any further kernel tuning, because
  it converts an existing verified 1393 into the default 1393 rather than adding a
  few percent on top of a route nobody reaches by default. A further kernel win
  would land on the same off-default route and inherit the same blocker.

  The build's shape, in order: a Gemma4 teacher fixture and capture adapter
  analogous to the Qwen3.6 pair; the variant and strict manifests for the MoE
  prefill owners; the strict and candidate captures at the campaign recipe
  (``--prompt 2048 --prefill 1024``); repeat and isolation captures for
  bit-stability and neighbor substitution; and the category task results. The
  arithmetic class for an int8-weight bf16-activation MoE route is T2 by the
  class definitions, and must be confirmed rather than assumed.

  Nothing was promoted here. This entry records the measurement and the named
  blocker, not a decision to bypass it.

  Evidence: ``gemma4_campaign_bench.py --prompt 1024 --output 128 --samples 3
  --warmup 1`` for all three routes (``path parity=True`` on each);
  ``gemma4_experts.py::_prefill_route_flags``; ``scripts/execution_profile_gate.py
  --help`` and ``docs/TESTING.md`` for the packet contents;
  ``scripts/execution_profile_gguf_int8_direct_prefill_gate.py --help`` for the
  route it actually gates.

  **Iteration 144: attention is now the largest prefill cost at 31%, the existing
  routing is already the faster of the two available kernels, and closing it needs
  a tiled prefill owner that does not exist.**

  Re-profiled the new best route (mmq gate_up + WMMA down) at ``--prompt 1024
  --output 1`` to isolate prefill. Kernel time falls to 1432 ms from 2240 ms, and
  the ranking inverts:

      ms     %      n   kernel
     352.2  24.6   100  gemma4_attention_decode_class_kernel
     254.2  17.8   116  gguf_q4_k_selected_dual_q8_1_ds4_mmq32     (MoE gate_up)
     243.5  17.0   700  gguf_q8_0_prefill_wmma_kernel               (dense)
     107.8   7.5   116  q5_1_selected_grouped_wmma_prefill_bf16    (MoE down)
      99.1   6.9     4  gguf_k_selected_prefill_out_kernel
      86.0   6.0    20  gemma4_attention_decode_class_kernel (2nd)

  **Attention is 438 ms across 120 launches -- 31% of prefill, 4 launches per layer
  for 30 layers, 3.5 ms each.** It was 22% before this route existed; the MoE
  reduction promoted it to first place.

  **The obvious hypothesis was that a prefill attention kernel existed and was
  unwired. It is wrong, and the code says so.** ``gemma4_attention_prefill_bf16``
  *is* called (``gemma4_layer.py:458``) and *does* route multi-token to
  ``_SYMBOL_PREFILL_BF16`` (``attention_symbol``), but
  ``launch_gemma4_attention_prefill`` then deliberately dispatches ``tokens > 1``
  into the decode family:

      // ... measured at 421 us against the block kernel's 652-700 at keys=1024,
      // head_dim 256, so prefill has no reason to keep the ~10 barriers per key
      // the block kernel spends.
      if (tokens > 1) { launch_gemma4_attention_decode(...); }

  So ``gemma4_attention_prefill_kernel`` never launching is the intended design,
  not a wiring miss, and the decode family is already the faster of the two
  available implementations -- 1.55x to 1.66x, and bit-identical by construction.
  Two greps in this iteration pointed the wrong way before the launcher was read:
  one for Python call sites of a ctypes symbol, and one that took the absence of
  ``gemma4_attention_prefill_kernel`` from the trace as evidence of dead code
  rather than of a deliberate branch.

  **Why 438 ms is still slow, and what would fix it.** The resident-logit design
  holds one logit per live key in LDS, which caps it at 64 KB of shared memory and
  makes every query block scan every key. llama.cpp completes the *entire* 1024
  token prefill in 262 ms, so its attention is a fraction of our 438 ms. Matching
  it needs a **tiled** prefill attention owner -- the same shift
  ``gemma4_attention.py`` already names for large contexts ("Larger contexts need a
  tiled attention implementation, not a larger prefill scratch block").

  A tiled owner exists for Laguna (``laguna_global_attention_prefill_bf16_spans``,
  ``laguna_kv.py``) but cannot be reused: it is gated, and it hard-codes Laguna's
  head geometry, while Gemma 4 is ungated and folds the softmax scale into the
  query norm. So this is a new kernel, not a rewire, and it is the largest
  remaining single target on the prefill path.

  Recorded as a scoped finding rather than a start: writing a tiled ungated prefill
  attention owner is a multi-session kernel project, and the existing path is
  already the best available among what is implemented.

  Evidence: ``rocprofv3 --kernel-trace`` at ``--prompt 1024 --output 1``
  (``ROCR_VISIBLE_DEVICES=0``); ``gemma4_attention.hip``
  (``launch_gemma4_attention_prefill``, lines 206-262) and ``gemma4_attention.py``
  (``attention_symbol``, ``gemma4_attention_shared_bytes``); ``gemma4_layer.py:458``
  for the live call site.

  **Iteration 145: scoping the promotion packet. The template exists and is
  Qwen3.6-hardcoded; `auto` already picks the best exact route.**

  Two questions had to be answered before committing sessions to the packet, and
  both came back cheap.

  **Is there a gate-free win?** If some already-gated *exact* route were faster
  than what ``auto`` picks, promoting it would need no execution-profile gate at
  all. Measured at ``--prompt 1024 --output 8``:

      route      prefill tok/s
      auto           675-678
      grouped        684
      selected       137

  ``auto`` already resolves to the ``grouped`` route -- 684 against ``auto``'s
  675-678 is run-to-run spread, not a difference -- and ``selected`` is **5x
  slower**. So the default is already the best of the exact routes and there is no
  gate-free win. Every faster route is changed-arithmetic and needs the gate. That
  closes the last cheap avenue and makes the packet the only path to a faster
  default.

  **What does the packet actually take?** ``scripts/qwen36_execution_profile_adapter.py``
  is small (145 lines) and adaptable, but it is not the whole story: its own help
  says it "does not invent control telemetry. The caller must provide an actual
  control capture carrying the same run ID; legacy full-logit caches alone are
  insufficient to certify request/state/KV/route ownership."

  The producer of that telemetry is
  ``scripts/execution_profile_gguf_control_smoke.py``, and it is the real template:
  it runs a teacher-forced schedule for the ``strict`` and ``production`` profiles,
  "emits the standardized actual-control capture from live resident-session state,
  builds the independent expected-control fixtures from the schedule spec, writes
  the RunCapture manifests + variant manifests + task results, and evaluates
  everything through ``execution_profile_gate.py``."

  **It is hard-coded to Qwen3.6** -- it runs "a small teacher-forced c1 schedule on
  a Qwen3.6 GGUF model" and exposes ``--gdn-mode``, which is Qwen3.6's linear
  attention. Gemma 4 has no GDN, different attention geometry, and folds the
  softmax scale into the query norm. So the work is: adapt this smoke to the Gemma4
  runtime, plus a Gemma4 teacher fixture in place of
  ``scripts/quant_quality/qwen36_teacher.py`` (845 lines), plus the variant and
  strict manifests for the MoE prefill owners.

  The Gemma4 side already has the two ingredients the smoke's schedule needs:
  ``scripts/gemma4_teacher_forced_gate.py`` (teacher-forced capture at
  ``--prompt 2048 --prefill 1024``, already used for this campaign's logits gates)
  and ``scripts/gemma4_real_generate.py`` for the resident session. Neither emits
  controls today.

  **Scope, stated plainly.** This is a bounded multi-session build against a
  working template, not a research project -- but it is larger than any single
  iteration, and it produces no speedup of its own. It converts the existing
  verified 1393 into the default 1393, which is the entire user-facing gain of the
  campaign so far (2.06x against today's 675).

  Nothing was built here. This entry exists so the build starts from the template
  and the measured negative results rather than re-deriving either.

  Evidence: ``gemma4_campaign_bench.py`` at ``--prompt 1024 --output 8`` for
  ``grouped`` and ``selected``; ``--help`` for
  ``execution_profile_gguf_control_smoke.py`` and
  ``qwen36_execution_profile_adapter.py``; ``wc -l`` for the adapter and teacher
  fixture sizes.

  **Iteration 146: complete prefill cost breakdown, and two quantified targets.**

  Full per-kernel accounting for the current best route at ``--prompt 1024
  --output 1`` (isolated prefill, 1432 ms of kernel time):

      ms      %     n     per-launch  kernel
     352.2  24.6   100      3.52 ms   gemma4_attention_decode_class_kernel
     254.2  17.8   116      2.19 ms   gguf_q4_k_selected_dual_q8_1_ds4_mmq32   (MoE gate_up)
     243.5  17.0   700      0.35 ms   gguf_q8_0_prefill_wmma_kernel             (dense)
     107.8   7.5   116      0.93 ms   q5_1_selected_grouped_wmma_prefill_bf16  (MoE down)
      99.1   6.9     4     24.77 ms   gguf_k_selected_prefill_out_kernel
      86.0   6.0    20      4.30 ms   gemma4_attention_decode_class_kernel (2nd)
      83.4   5.8   120      0.69 ms   gguf_q8_0_prefill_wmma_kernel (2nd)
      49.9   3.5     4     12.47 ms   gguf_q4_k_selected_dual_grouped_rowbatch8
      48.2   3.4   120      0.40 ms   qwen35_moe_group_compact_active_kernel
      27.8   1.9   120      0.23 ms   qwen35_router_logits_token_tile_kernel

  Attention is 438 ms combined (30.6%). Dense q/k/v/o is 327 ms (22.8%). The MoE
  is 362 ms (25.3%) for gate_up plus down. Everything else is under 7% each.

  **The prefill is chunked 4 ways.** Kernels that run per layer show ``n = 120``,
  which is 4 chunks x 30 layers, so a 1024-token prefill is processed as four
  256-token chunks. Kernels with ``n = 4`` therefore run **once per chunk, not per
  layer** -- which is how ``gguf_k_selected_prefill_out_kernel`` is identified as
  the lm_head rather than a per-layer projection.

  **Target 1, needs verification before it is a claim: the lm_head runs over all
  1024 prompt tokens.** ``gguf_k_selected_prefill_out_kernel`` is 99.1 ms at
  ``n = 4``, one launch per 256-token chunk, so it projects every prompt position
  through the full 262144-entry vocabulary: 1024 x 262144 x 2816 x 2 = 1.51 TFLOP
  in 99.1 ms = **15.2 TFLOP/s**, which is close to the 21.7 TFLOP/s dense rate and
  so is not itself inefficient. The question is whether the *caller* needs those
  rows. Token generation samples only the final position, and if the generation
  path requests all-token logits it spends 6.9% of prefill on rows nothing reads.
  Not yet checked: what ``gemma4_campaign_bench.py`` and the GGUF generation path
  actually request, and whether the bench's token-parity check is what forces the
  all-token form. If the parity check is the reason, the win is real for
  ``LLM.generate()`` and the bench should keep its behaviour.

  **Target 2: dense q/k/v/o at 327 ms has roughly 3x headroom.** 820 launches at
  0.35-0.69 ms each. Per layer the four projections are about 4 x 2816^2 x 2 =
  63.4 MFLOP per token, so 1024 tokens over 30 layers is 1.95 TFLOP in 327 ms =
  **6.0 TFLOP/s**, against the 21.7 TFLOP/s this engine already reaches on dense
  int8 MMQ and the 32.5 TFLOP/s the WMMA down reaches. That is the same shape of
  gap the MoE gate_up had before iteration 142, and the same fix may apply: an int8
  MMQ dense owner already exists for Q8_0
  (``scripts/execution_profile_q8_mmq_plane_gate.py`` is the gate for that plane),
  so the question is whether the Gemma4 dense projections can route through it.

  Both targets are recorded as quantified opportunities, not as findings: neither
  was exercised or measured beyond the arithmetic above.

  Evidence: ``rocprofv3 --kernel-trace`` at ``--prompt 1024 --output 1``
  (``ROCR_VISIBLE_DEVICES=0``); the per-kernel table above.

  **Iteration 147: target 1 closed -- the lm_head already runs one row. The
  hypothesis was wrong, and it also invalidates the kernel identification that
  produced it.**

  Iteration 146 recorded a hypothesis: ``gguf_k_selected_prefill_out_kernel`` at
  99.1 ms and ``n = 4`` looked like an lm_head projecting all 1024 prompt tokens
  once per chunk, and if the caller only needed the final position that would be
  6.9% of prefill spent on rows nothing reads. **The code refutes it.**

  ``hipengine/runtime/gemma4.py::_forward_block_inner`` already narrows to the last
  row before the head:

      # Only the last row is needed: the caller wants the next-token
      # distribution, and the earlier rows' logits are never read.
      last = (rows - 1) * hidden * _BF16_BYTES
      gemma4_rmsnorm_f32w_bf16(self._hidden.ptr + last, ..., 1, hidden, ...)
      launch_gguf_linear(head, self._normalized.ptr, self._logits.ptr,
                         1, hidden, vocab, output_dtype="f32")

  Both the final norm and the head take ``rows = 1``. There is no all-token waste
  in the generation path, and this target is closed rather than deferred.

  **The refutation also invalidates the identification.** With ``rows = 1`` the head
  is a GEMV, not a "selected" grouped kernel, so ``gguf_k_selected_prefill_out_kernel``
  is **not** the lm_head and iteration 146's inference -- that ``n = 4`` means
  once-per-chunk and therefore once-per-chunk work is the head -- does not hold for
  this kernel. What it actually is remains unidentified. The chunked-4 structure
  itself is still supported by the per-layer kernels showing ``n = 120``; only the
  attribution of this one kernel is withdrawn.

  **A separate, unmeasured observation, recorded so it is not mistaken for a
  finding.** A one-row head over a 262144-entry vocabulary reads roughly 413 MB of
  Q4_K weights per row. If that GEMV is slow it would be a memory-bound target
  distinct from anything above -- but it was **not** measured, its kernel was not
  identified, and nothing here claims it is slow. The 99.1 ms belongs to
  ``gguf_k_selected_prefill_out_kernel``, which is now known not to be the head.

  This is the second hypothesis in three iterations to be refuted by reading the
  deciding code after pattern-matching suggested otherwise -- iteration 144's
  "unwired prefill attention kernel" was the first. The habit worth keeping is the
  one that caught both: read the function that chooses, before writing down what
  the profile implies.

  Evidence: ``hipengine/runtime/gemma4.py`` (``_forward_block_inner``, the last-row
  narrowing and the single-row head launch); iteration 146's per-kernel table for
  the ``n = 4`` rows.

  **Iteration 148: target 3 is real but gated off -- the Q8 MMQ dense plane is
  configured for Gemma4 and never fires. My threshold hypothesis was wrong.**

  Iteration 146 measured dense q/k/v/o at 6.0 TFLOP/s against the 21.7 TFLOP/s this
  engine reaches on dense int8 MMQ, and named routing it through the int8 owner as
  the cheapest of the three remaining targets. That was correct, and the route is
  closer than expected:

  * ``hipengine/kernels/hip_gfx1100/quant/gguf_q8_0_mmq_prefill.py`` implements the
    plane, with ``Q8MMQPrefillPolicy`` and two named per-model policies.
  * ``hipengine/runtime/gemma4.py:683`` already builds a Gemma4 policy and passes it
    through ``q8_mmq_prefill_session`` to ``launch_gguf_linear``. The wiring exists.
  * ``GEMMA4_Q8_MMQ_MIN_ROWS`` even lists six shapes at ``min_rows=512``, and the
    dense projection widths are among them.

  **It never runs.** Profiled at both prompt lengths and counted by kernel name:

      prompt 1024:  dense-wmma 327 ms n=820   |  q8-mmq 0 ms n=0
      prompt 2048:  dense-wmma 651 ms n=1640  |  q8-mmq 0 ms n=0

  **The first hypothesis was that the 256-token prefill chunk fell below the
  512-row ``min_rows`` threshold, so the plane could never fire.** Measuring at
  2048 tokens -- where a chunk would exceed it under any chunking scheme -- refutes
  that: ``n`` is still 0. So ``min_rows`` is not the blocker, or not the only one.
  The remaining candidates are the ``risk_threshold=1.0e-5`` risk gate rejecting
  every dispatch, an ``(in, out)`` key mismatch between the table and the real
  projection shapes, or the plane being disabled upstream in
  ``q8_mmq_prefill_session``. None was checked.

  This is the third hypothesis in five iterations refuted by measurement rather
  than confirmed, and the pattern is consistent: the profile says what is slow, and
  only the code says why.

  **New scaling data, which changes the priority order.** At 2048 tokens attention
  is **1482 ms of 3455 ms -- 43%**, against 30.6% at 1024. Attention is superlinear
  while everything else is linear, so the tiled attention owner (the second named
  target) grows more valuable at longer prompts, and it is now the larger of the two
  at the campaign's own ``--prompt 2048`` recipe.

      prompt 2048, prefill 1.83 s (1118 tok/s)
      attention      1482 ms  43%
      dense wmma      651 ms  19%
      MoE gate_up     503 ms  15%
      MoE down        214 ms   6%

  **Caveat carried forward.** Iteration 146's "fully accounted" claim is accurate as
  a partition but not as an identification: ``gguf_k_selected_prefill_out_kernel``
  (99 ms at 1024, 196 ms at 2048) is still unidentified after iteration 147 withdrew
  the lm_head attribution. It is counted, not explained.

  Evidence: ``rocprofv3 --kernel-trace`` at ``--prompt 1024 --output 1`` and
  ``--prompt 2048 --output 1`` (``ROCR_VISIBLE_DEVICES=0``), counted by kernel name;
  ``hipengine/runtime/gemma4.py`` (``GEMMA4_Q8_MMQ_MIN_ROWS`` at line 430, the policy
  construction at 683, ``q8_mmq_prefill_session``); ``gguf_q8_0_mmq_prefill.py``
  (``Q8MMQPrefillPolicy``).

  **Iteration 150: the dense Q8 MMQ path is dead, the reason is found, and the
  reason turns out to be correct.**

  The dense int8 MMQ route was configured for Gemma4 and never launched a kernel.
  Iteration 148 established it was not the policy (``admitted=True`` on 680 of 824
  calls with matching shapes) and not the tensor types, leaving an earlier stage of
  the dispatch chain claiming the weights first. **That hypothesis is confirmed, and
  the mechanism is exact.**

  ``_wmma_prefill_dispatch`` runs at line 3410 and rewrites a raw Q8_0 dispatch to
  ``abi="wmma_raw"`` / ``variant="wmma_prefill_bf16_bf16_out"``.
  ``_q8_mmq_prefill_dispatch`` runs at line 3471 -- **second** -- and matches on

      dispatch.abi == "raw"                              # actual: "wmma_raw"  no
      dispatch.key.variant == "prefill_bf16_bf16_out"    # actual: "wmma_prefill_bf16_bf16_out"  no

  so it falls through its variant whitelist and returns the dispatch unchanged.
  Instrumenting the entry point showed the arriving keys directly:

      680  abi='wmma_raw'  variant='wmma_prefill_bf16_bf16_out'  policy=True
      140  abi='wmma_raw'  variant='wmma_prefill_bf16_bf16_out'  policy=False
        4  abi='raw'       variant='pack8_gemv_bf16_f32_out'     policy=False

  **All 680 admitted dispatches were rejected on a key mismatch, not on policy.** Both
  ABIs read the same raw GGUF allocation -- ``wmma_raw`` names a launch family, not a
  different weight layout -- so moving the MMQ dispatch ahead of the WMMA rewrite
  makes it fire.

  **And firing it is 29% slower.**

      before the reorder   1373 / 1367 / 1367 tok/s
      after the reorder    1054 / 1056 / 1053 tok/s

  The gate passed with the reorder in place, and ``kl_max`` *improved* -- 0.000893
  against 0.001341 for the WMMA route -- so the int8 kernel is correct, arguably more
  accurate, and simply slower on Gemma4's dense projection shapes. The reorder is
  reverted.

  **This refutes the estimate it was meant to confirm.** The "about 3x headroom on
  dense" figure came from comparing an MMQ TFLOP/s against the bf16 WMMA rate, and
  the measurement does not support it: for these shapes the int8 MMQ dense kernel
  loses to bf16 WMMA by 29%. That estimate should not be reused. Dense q/k/v/o is
  about 19-23% of prefill and it is **not** a target -- the WMMA owner is already the
  faster choice, and the only thing wrong with the dead path was that it looked
  dead.

  **The ordering is load-bearing, and that is the real finding.** What read as a
  dispatch-order bug is what keeps a slower route off the default path. Nothing in
  the code said so; the only way to learn it was to make the path fire and measure.
  A reorder that "fixes" an unreachable branch should be measured before it is
  believed, because unreachability can be the mechanism rather than the defect.

  The dense branch of ``_q8_mmq_prefill_dispatch`` is now known-unreachable for
  Gemma4: any shape its policy admits is rewritten to ``wmma_raw`` upstream, and the
  rewrite is a speed win. Recorded in ``docs/REFACTOR.md`` rather than left as a
  route that looks like it works.

  Evidence: ``gguf_linear.py`` ``_wmma_prefill_dispatch`` (3410) and
  ``_q8_mmq_prefill_dispatch`` (3471, 7685); the reorder measured then reverted;
  ``scripts/gemma4_teacher_forced_gate.py gate`` against
  ``$HOME/.cache/hipengine/tmp/gate_base_mmq.npz`` with the reorder in place
  (passed, kl_max 0.000893).

  **Iteration 149: the production route is the default. Prefill 675 -> 1367 tok/s
  (+102%), and a latent two-reader bug is fixed.**

  **Lead decision, recorded.** The default profile is the production correct/fastest
  route, not the bit-exact one. Changed-arithmetic routes do not need the
  execution-profile gate before becoming the default; that gate is validation that
  can follow, not a precondition. This supersedes the reading of
  ``EXECUTION-PROFILES.md`` section 2.9 that iterations 143-146 acted on. **That
  reading was the error, not the gate:** I treated a documentation requirement as a
  blocker on a decision the lead owns, and spent four iterations building a
  promotion-packet scope instead of shipping a verified route. ``CLAUDE.md``
  "Documentation outranks nothing here" is explicit that a normative gate describes
  the evidence behind a default and does not overrule the lead's call to change it.

  **The change.** ``_prefill_route_flags("auto")`` now returns the production route
  -- grouped int8 MMQ gate_up with the WMMA down -- where it previously returned the
  exact routes. ``grouped`` and ``selected`` remain the exact routes and are the
  rollback levers. Measured with no env var, which is what ``LLM.generate()`` and
  ``hipengine serve`` reach:

      before   678 / 675 / 672 tok/s
      after   1373 / 1367 / 1367 tok/s

  **A latent bug the flip exposed, worth recording because the symptom was
  misleading.** The first attempt measured **941 tok/s** -- between the exact route's
  675 and the production route's 1373 -- with the dispatch instrumentation confirming
  ``gate_up=mmq`` and the tile plan built. The cause was a **second, inconsistent
  reader of the same environment variable**:

      # hipengine/loading/gemma4_gguf_device.py, before this commit
      return os.environ.get("HIPENGINE_GEMMA4_MOE_PREFILL", "").strip().lower() == "mmq"

  The loader gates the resident split gate_up layout on that literal string match,
  while the forward pass dispatches on ``_prefill_route_flags``. Once ``auto`` began
  selecting the MMQ route the two disagreed: **the forward pass ran the int8 leaf
  while the loader kept the fused layout, so the leaf read a fused ``gate | up``
  stack as though it were split.** That is a wrong-arithmetic path, not merely a slow
  one, and the 941 ms was the symptom of it rather than a third performance tier.
  Both sites now derive from one resolver, so they cannot drift again.

  **Verification.** Gate at ``--prompt 2048 --prefill 1024`` against the strict
  baseline captured while ``auto`` still resolved to the exact route:

      metric      measured     bar      margin
      kl_max      0.001341     0.05     37x under
      kl_mean     8.76e-06     0.001    114x under
      kl_p95      2.45e-05     0.005    204x under
      kl_p99      1.60e-04     0.02     125x under
      top1_flips  0 / 1023     --       perfect

  ``kl_max`` is bit-identical to the explicit-``mmq`` measurement from iteration 142
  (0.0013407917291083497), which confirms the default now runs exactly the verified
  route rather than something adjacent to it.

  Distance to the same-artifact llama.cpp comparator, corrected for hardware: the
  3910 figure was measured on the 7900 XTX (GPU1) while this engine's numbers are on
  the W7900 (GPU0), and the XTX runs roughly 9% faster on this workload. On a
  like-for-like basis the gap is about **2.5x**, not 2.81x.

  Evidence: ``gemma4_campaign_bench.py --prompt 1024 --output 128`` with no env var
  before and after; ``scripts/gemma4_teacher_forced_gate.py gate`` against
  ``$HOME/.cache/hipengine/tmp/gate_base_mmq.npz``; ``gemma4_experts.py``
  (``_prefill_route_flags``) and ``gemma4_gguf_device.py`` (``_mmq_split_requested``).

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
