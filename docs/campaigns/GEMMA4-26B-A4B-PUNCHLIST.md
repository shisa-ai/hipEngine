---
status: current
owns: Gemma 4 26B-A4B gfx1100 kernel-family scoreboard against llama.cpp and the prefill, decode and MTP optimization punchlist.
---
# Gemma 4 26B-A4B punchlist: family scoreboard and candidates

This page is the working list for making Gemma 4 26B-A4B faster on gfx1100. It
has three parts:

1. A **family scoreboard**: every kernel in a prefill and in a decode step,
   bucketed into functional families and set against llama.cpp on the same GPU,
   artifact and shapes. Regenerate it after every landed change and update the
   tables in place.
2. A **punchlist** of candidates, organized by phase (prefill, decode, MTP) and
   family, each with the evidence behind it and an estimated gain. Candidates are
   hypotheses: the job is to try each one, measure it, and record the result in
   its row, whether that is a win, a wash or a refutation.
3. A short list of **measured dead ends** so they are not re-tried without new
   evidence.

The long-form history of the campaign lives in
[GEMMA4-26B-A4B-OPTIMIZATION.md](GEMMA4-26B-A4B-OPTIMIZATION.md). This page does
not replace it. Normative rules stay in [OPTIMIZATION](../OPTIMIZATION.md) and
[EXECUTION-PROFILES](../EXECUTION-PROFILES.md). Every candidate that changes
arithmetic must pass the campaign's teacher-forced logits gate
(`scripts/gemma4_teacher_forced_gate.py gate --prompt 2048 --prefill 1024`,
`kl_max` below 0.05 and no top-1 flips), and every win must be confirmed on the
path `hipengine.LLM.generate()` reaches.

## Scoreboard

### Basis

- Host `epyc`. Primary lane: **RX 7900 XTX** (gfx1100, 24 GB). The W7900 is a
  secondary lane.
- Artifact `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf`: expert gate_up Q4_K (29 layers),
  expert down Q5_1 (29 layers), layer 29 experts Q5_K gate_up / Q8_0 down, every
  dense projection Q8_0, tied Q8_0 `token_embd` as the lm_head (262144 × 2816).
  30 layers: 25 sliding-window (head_dim 256, window 1024, 8 KV heads) and 5
  global (head_dim 512, 2 KV heads, K and V share one projection).
- hipEngine `gemma4` at `df867de1f`, BF16 KV, default production route.
- llama.cpp upstream `a97cce8`, HIP build with `-DAMDGPU_TARGETS=gfx1100
  -DGGML_NATIVE=OFF`, flags `-ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 4096 -ub 1024`.
- Measured 2026-09-28 with both GPUs otherwise idle.

### Headline (RX 7900 XTX, unprofiled)

| Shape | hipEngine | llama.cpp | hipEngine as % |
| --- | ---: | ---: | ---: |
| Prefill, 1024 tokens (tok/s) | 1977 | 4969 | 39.8% |
| Prefill, 4096 tokens (tok/s) | 1060 | 4505 | 23.5% |
| Decode at 1024 context (tok/s) | 43.15 | 84.45 | 51.1% |
| Decode at 4096 context (tok/s) | 39.79 | 81.67 | 48.7% |
| Decode with MTP assistant, draft 3 (tok/s) | not implemented | 164.6 | — |

With f16 KV, llama.cpp decodes at 88.85 / 85.82 tok/s (1024 / 4096 context).
Its prefill does not change. hipEngine's decode rows come from
`scripts/gemma4_campaign_bench.py --prompt P --output 128 --samples 3 --warmup 1`;
llama.cpp's come from `llama-bench -r 3`. The MTP row is described under
[MTP](#mtp).

On the W7900 on 2026-09-27, hipEngine measured 1759.5 prefill / 38.79 decode
against llama.cpp's 4377.0 / 70.1 at 1024 tokens. That comparison is recorded in
`worklog/entries/20260927T174143.751391Z-lhl-gemma4-gemma4-three-engine-comparison-632e87.md`.

### Prefill families (RX 7900 XTX, milliseconds per prefill)

These are device-busy sums from `rocprofv3 --kernel-trace`. hipEngine uses its
steady-state (second) prefill. llama.cpp uses the per-run average of two
`llama-bench` runs. "Gap" is hipEngine minus llama.cpp.

| Family | hipEngine 1024 | llama.cpp 1024 | Gap 1024 | hipEngine 4096 | llama.cpp 4096 | Gap 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attn.sliding (25 layers) | 29.9 | 12.3 | +17.6 | **1478.7** | 84.0 | **+1394.7** |
| attn.global (5 layers) | 39.2 | 6.7 | +32.4 | **610.4** | 85.7 | **+524.7** |
| dense.q8_0 (q/k/v/o, shared MLP) | 115.6 | 59.0 | +56.6 | 461.6 | 233.4 | +228.2 |
| moe.gate_up (Q4_K, 29 layers) | 112.6 | 49.4 | +63.2 | 453.9 | 194.2 | +259.7 |
| moe.down (Q5_1, 29 layers) | 46.8 | 30.2 | +16.6 | 191.4 | 120.4 | +71.0 |
| moe.l29_experts (Q5_K / Q8_0) | **66.2** | 2.6 | **+63.6** | 273.1 | 10.4 | +262.7 |
| moe.route_glue (schedule, gather, combine) | 34.5 | 10.5 | +24.0 | 137.5 | 41.2 | +96.3 |
| moe.router_gemm (router logits, prescale) | 13.6 | 4.0 | +9.6 | 53.3 | 15.5 | +37.8 |
| moe.act_quant (Q8_1 activation pack) | 5.7 | 1.5 | +4.1 | 22.8 | 6.0 | +16.9 |
| norm | 9.0 | 11.9 | −2.9 | 35.8 | 46.8 | −11.0 |
| elementwise (rope, GEGLU, adds) | 4.6 | 4.2 | +0.4 | 18.5 | 16.5 | +2.0 |
| attn.kv_write | in attention | 2.0 | −2.0 | in attention | 9.2 | −9.2 |
| lm_head | 1.9 | 1.1 | +0.9 | 7.8 | 1.1 | +6.7 |
| runtime copy / fill | 1.1 | 1.3 | −0.2 | 3.0 | 1.7 | +1.3 |
| **Device busy** | **480.6** | **196.7** | **+283.9** | **3747.9** | **866.2** | **+2881.7** |

llama.cpp runs a second HIP stream during prefill. Its shared-expert MLP branch
overlaps the MoE branch, so its busy total (≈187 ms in a warm run) fits into a
≈140 ms span. hipEngine's single-stream kernel-busy fraction is about 94%
(480.6 ms busy in a 511.3 ms span). This is a timeline fraction, not hardware
occupancy or a measurement of recoverable host overhead.

### Decode families (RX 7900 XTX, milliseconds per token)

| Family | hipEngine @1024 | llama.cpp @1024 | Gap @1024 | hipEngine @4096 | llama.cpp @4096 | Bytes / token | Floor at 816 GB/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| moe.gate_up | **5.42** | 0.71 | **+4.70** | 4.79 | 0.71 | 517 MB | 0.63 |
| attn.sliding | **4.28** | 0.35 | **+3.93** | 3.64 | 0.45 | 210 MB | 0.26 |
| dense.q8_0 | 3.02 | 3.48 | −0.46 | 2.96 | 3.49 | 1748 MB | 2.14 |
| moe.down | **2.83** | 1.28 | **+1.55** | 2.63 | 1.27 | 345 MB | 0.42 |
| norm | 1.51 | 1.05 | +0.46 | 1.50 | 1.05 | — | ~0.1 |
| moe.route_glue | 1.29 | 0.44 | +0.86 | 1.27 | 0.44 | — | — |
| lm_head | 1.15 | 1.05 | +0.10 | 0.97 | 1.06 | 784 MB | 0.96 |
| attn.global | 0.75 | 0.11 | +0.64 | **2.82** | 0.15 | 10 MB → 42 MB | 0.05 |
| moe.router_gemm | 0.33 | 0.11 | +0.22 | 0.32 | 0.11 | 43 MB | 0.05 |
| runtime copy / fill | 0.24 | 0.05 | +0.19 | 0.23 | 0.05 | — | — |
| moe.l29_experts | 0.24 | 0.03 | +0.21 | 0.24 | 0.03 | 39 MB | 0.05 |
| elementwise | 0.23 | 0.26 | −0.03 | 0.23 | 0.26 | — | — |
| attn.kv_write | in attention | 0.73 | — | in attention | see note | — | — |
| **Device busy** | **21.30** | **9.65** | **+11.65** | **21.59** | see note | ≈3.7 GB | **≈4.9** |
| **Unprofiled step** | **23.17** | **11.84** | | **25.13** | **12.24** | | |
| **Launches / token** | **1146** | ≈1330 | | 1146 | ≈1330 | | |

Reading the decode table:

- **Four families account for 11.3 ms of the 11.65 ms busy gap:**
  moe.gate_up, attn.sliding, moe.down and moe routing (glue + router). Closing
  those four to llama.cpp's level would put hipEngine at about llama.cpp's step
  time.
- **hipEngine's dense decode already beats llama.cpp's** (3.02 against 3.48 ms,
  since llama.cpp also pays a separate Q8_1 activation quantize). Its MoE
  GEMVs do not: gate_up moves 517 MB in 5.42 ms, about 95 GB/s, or 10% of the
  XTX's bandwidth.
- **The floor column** is the bytes each family must read, divided by 816 GB/s
  (85% of the XTX's 960 GB/s). Its total, ≈4.9 ms per token (~200 tok/s), is
  the memory-bound ceiling for this artifact. llama.cpp runs at 42% of it and
  hipEngine at 21%.
- **Note on llama.cpp at 4096 context with BF16 KV.** The trace charges
  7.79 ms per token to `convert_unary<bf16→f16>`. llama.cpp converts the BF16
  cache to f16 for its flash kernels on every step, on a side stream. Its
  unprofiled step is still 12.24 ms (81.67 tok/s), so the conversion overlaps
  other work and the profiled busy sum (16.86 ms) overstates it. Treat
  llama.cpp's 4096-context per-family decode numbers as indicative only. f16 KV
  removes the conversion (85.82 tok/s).

### How to regenerate

```bash
# From the gemma4 worktree. N = physical GPU (1 = XTX, 0 = W7900).
hipcc --version > /tmp/hipcc_version.txt
ROCR_VISIBLE_DEVICES=N .venv/bin/python scripts/gemma4_family_census.py drive --prompt 1024 --warm-only
env -u HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES=N HIPENGINE_REQUIRE_CACHED_BUILD=1 \
  HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipcc_version.txt PYTHONPATH=. \
  rocprofv3 --kernel-trace --output-format csv -d /tmp/g4_trace_1024 -o run -- \
  .venv/bin/python scripts/gemma4_family_census.py drive --prompt 1024 --decode 64
python3 scripts/gemma4_family_census.py rollup hipengine /tmp/g4_trace_1024/run_kernel_trace.csv \
  --decode-steps 64 --json /tmp/g4_family_1024.json

# llama.cpp comparator (same flags as the scoreboard):
LB="llama-bench -m <gguf> -ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 4096 -ub 1024 --no-warmup"
rocprofv3 --kernel-trace --output-format csv -d /tmp/lc_pp -o run -- $LB -p 1024 -n 0 -r 2
rocprofv3 --kernel-trace --output-format csv -d /tmp/lc_tg -o run -- $LB -p 0 -n 64 -d 1024 -r 1
python3 scripts/gemma4_family_census.py rollup llamacpp /tmp/lc_tg/run_kernel_trace.csv \
  --pp-trace /tmp/lc_pp/run_kernel_trace.csv --pp-runs 2 --decode-steps 64
```

Repeat with `--prompt 4096` / `-p 4096` / `-d 4096` for the long shape. Omitting
the compiler-version pair deadlocks the profiled process on `clang++ --version`.
The driver replays the campaign bench's natural corpus rather than a repeated
token, because a repeated token sends every row to the same eight experts and
misstates the MoE cost. Profiled decode runs about 25% slower than unprofiled
because of tracing overhead. Compare families only between traces taken the same
way, and take headline rates from the unprofiled benches.

### Measurement contract

For each scoreboard update, link compact artifacts under `benchmarks/results/`
for the headline runs and family rollups. Record the model artifact, engine and
comparator revisions, physical host/GPU, exact commands, workload/token corpus,
KV dtype, execution profile, selected variants, warmups, repetitions, and
correctness result. Keep raw traces outside Git.

- Compare baseline and candidate with the same protocol on the same physical
  GPU. Predeclare repetitions, report individual samples and their spread, and
  use paired, interleaved runs to limit thermal or clock drift. Keep XTX and
  W7900 results separate.
- Report summed kernel durations, elapsed GPU span, and host wall time as
  distinct measurements. Overlapping streams can make summed durations exceed
  elapsed time. A difference between an unprofiled wall time and a profiled
  kernel sum is not a measured host gap; neither is a timeline gap necessarily
  removable by graph capture. Attribute copies, synchronization, and sampling
  directly before estimating D8 or X5.
- For MTP, record proposed and accepted draft counts, accepted-prefix length
  histograms, emitted tokens per cycle, and draft/verify/acceptance wall times.
  Define the acceptance denominator and account for EOS and truncated cycles.
  Measure speedup against a true no-MTP autoregressive run using the same
  protocol and the category suite plus heldouts required by
  [OPTIMIZATION](../OPTIMIZATION.md). The five-prompt comparator below is not
  a substitute for that promotion suite.
- Candidate gains can overlap. Refresh the affected family measurements after
  landing a change rather than adding estimates for overlapping work, such as
  P1/P14 or D8/D10/X7. Confirm the selected route through `LLM.generate()`;
  also exercise `hipengine serve` for changes to server behavior.

### Not yet in this scoreboard

Known holes, so a later pass does not read the tables as complete:

- **8192 context.** The campaign's configured context is 8192 and only 1024 and
  4096 are traced. Attention is 56% of device-busy at 4096; at 8192 it is more,
  and the sliding fallback runs twice as many blocks.
- **Time to first token.** The campaign bench records `first_token_s`; nothing
  here reports it. It is the user-visible prefill number and the first place a
  prompt-processing regression shows up.
- **A prefill floor.** The decode table carries a bytes / 816 GB/s floor; the
  prefill table does not, so "2.3x behind llama.cpp" cannot be read as "2.3x off
  the roofline". Without it there is no way to tell a tuneable kernel from a
  structurally wrong one.
- **A prefill launch census.** Decode has 1146 launches per token; prefill's
  per-block launch count is not tabulated (P6 cites 412 dense launches).
- **Sampler and host-transfer cost.** The runner copies vocabulary logits to
  the host, applies softcap with NumPy, and selects tokens there. Attribute
  those costs directly; subtracting the profiled device-busy sum from an
  unprofiled step does not isolate them. X7 covers the device-resident route.
- **The serve path.** Every number here comes from `LLM.generate()` or the
  campaign bench. Add a real `hipengine serve` request and route confirmation
  for server-facing changes. `AGENTS.md` accepts either public surface for a
  kernel change; a server benchmark is a separate comparison.

## Verification prerequisites

These gate the candidates rather than compete with them.

| ID | Area | Check | Why | Status |
| --- | --- | --- | --- | --- |
| V1 | attention | **Extend the campaign gate past the window: `--prompt 5120 --prefill 4096 --context 8192`** (1023 scored rows, the campaign row-count standard). The gate requires `prefill < prompt - 1`; capture the baseline and candidate with matching prompt, prefill, context, and block geometry. `capture_chain()` discards the prefill return value and scores subsequent single-token forwards, so this checks the cache produced by long prefill, not its output logits directly. Add direct prefill-logit comparisons under V2/V5. Every recorded gate run uses `--prefill 1024` or `--prefill 0` (13 and 3 occurrences, `GEMMA4-26B-A4B-OPTIMIZATION.md`), and the frozen `teacher-forced-prefilled-*-001c4fec3.npz` baselines are `--prompt 2048 --prefill 1024` -- two 512-row blocks with keys 512 and 1024, **both at or under the window and both on `attn_fwd`**. So no gate has ever exercised a prefill block with `keys > 1024`, which is the path P1 replaces. | P1 changes a path with no numerical coverage today, and P1 is the largest estimated win on the board. | in progress |
| V2 | attention | **Build an independent oracle for the windowed path.** The gate compares two hipEngine arms against a captured baseline, so it cannot detect a wrong mask that both arms share. Options: score hipEngine's 4096-token prefill logits against llama.cpp's, or add a naive CPU reference for windowed attention. | The kernel-level parity test P1 proposes (keys 1536 / 2048 / 4096) is necessary but it is a kernel test, not a model-level one. | open |
| V3 | attention | **Answered: the fallback applies the window mask. The 49x is a kernel swap plus the fallback's own quadratic growth.** The exact key-class kernel reads `keep_mask` and pass 1 skips fully-masked key tiles (`active[t] = mask_row[...] != 0`), so the window is applied. This rules out the missing-mask explanation, not every possible numerical defect; V1/V2 still cover model-level correctness. Measured at the prefill block shape (`rows=512 keys=4096 heads=16 head_dim=256`, causal mask vs causal-and-windowed): **26.75 -> 17.29 ms, so the window buys +35.4%** on that block, and **+29.9%** (121.18 -> 84.98 ms) summed over all 8 blocks of a 4096 prefill. Cross-checked against the model: the 5 global layers measure 122.1 ms/layer, and dividing by the **measured head_dim ratio of 1.327x** (mean over all 8 block sizes, range 1.240-1.398; not 2.0 -- the kernel is dominated by the key walk, so doubling head_dim costs only 1.327x) predicts 92.0 ms/layer for an unwindowed sliding row against 59.1 measured, a **~36%** window saving, which agrees with the direct 30% within the probe's launch-overhead bias. Both artifacts: `benchmarks/results/2026-09-28-gemma4-attention-window-value-w7900.json` and `-head-dim-ratio-w7900.json`, reproducible with `scripts/gemma4_attention_window_value_probe.py`. The 49x decomposes as 15.7x (fallback quadratic: 7.74 ms/layer at 1024 to 121.18 at 4096, matching the global layers' 15.6x) x 3.15x (fallback where 1024 used `attn_fwd`). | P1's premise and its ~1300 ms estimate both stand. The window already recovers ~30-38%, so the remaining win is the kernel swap plus passes 2 and 3's full key walk. P14 attacks that walk directly and is now the better-evidenced half. | **answered** |
| V4 | decode | **Re-run the decode gate at 4096 context with BF16 KV.** The recorded decode gate results are at 1024-key chains. D12 (INT8 KV) and D2 / D4 (new attention kernels) all change arithmetic at long context, and llama.cpp's own 4096-context per-family numbers are already flagged indicative because of its BF16-to-F16 conversion. | The long-context decode path is where D2 and D4 spend their estimate and it is the least-covered numerically. | open |
| V5 | attention / chunking | **Test window and chunk boundaries.** Cover prompt lengths 511/512/513 and 1023/1024/1025, plus 1537 and the long-context V1 shape. Compare chunked prefill, tokenwise execution, and an independent oracle under the declared numerical contract. Capture prefill logits at matched positions directly, then check decode continuation. Record block sizes and selected attention routes. | V1 scores continuation after prefill; boundary transitions, chunk-dependent behavior, and direct prefill outputs need separate coverage. | open |

## Punchlist

**Status** is one of `open`, `in progress`, `won` (landed on the default path),
`wash`, `refuted`, or `blocked (cause)`. **Est. gain** comes from the trace
arithmetic above: it is a hypothesis, not a measurement. When a row closes,
record the measured result, the commit and the gate result in its row.

Estimates convert milliseconds to rates against the unprofiled baseline: a
1024-token prefill takes 518 ms, a 4096-token prefill 3.87 s, and a decode step
23.17 ms at 1024 context and 25.13 ms at 4096.

### Prefill

| ID | Family | Candidate | Evidence | Est. gain (1024 / 4096) | Status | Result |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | attn.sliding | **Windowed AOTriton for sliding chunks past the window.** The vendored AOTriton 0.11.2b already ships `CausalType::WindowedAttention` images (`FONLY__*bf16@16_256_*_3_0`) and takes integer `window_left` / `window_right`. `aotriton_wrap.cc` passes only the bottom-right-aligned sentinels, so `aotriton_prefill_admits` refuses every block once the 1024-token window binds, and those blocks fall back to the exact class kernel. Pass `window_left = sliding_window − 1`, `window_right = 0`, extend the admission predicate to "mask is exactly causal ∧ windowed", and extend the parity test to keys 1536 / 2048 / 4096. | 4096: sliding attention is 1478.7 ms (39.5% of the prefill). Only blocks 1–2 use `attn_fwd`; blocks 3–8 run `gemma4_attention_decode_class_kernel<…,1,2>` — **verified**: `DEFAULT_PREFILL_BLOCK` is 512, so 4096 is 8 blocks, and `launch_gemma4_attention_prefill` tries the decode family first whenever `tokens > 1`. The plumbing point is also verified: `aotriton_wrap.cc` already selects `CausalType::WindowedAttention` and passes the `BottomRightAligned` sentinel for both bounds, which is why the `_3_0` windowed image currently runs as plain causal. llama.cpp: 84 ms. Note the recorded alternative: `a4c7cdfc8`'s "Next" proposes a tiled kernel for this same gap rather than the window plumbing, so the two bets are on the record. | 0 / **~1300 ms** (4096: ~1060 → ~1600 tok/s) | open | |
| P2 | attn.global | **Tiled flash attention at head_dim 512** for the 5 global layers. Options: (a) port llama.cpp's `flash_attn_tile<512,512,4,8>`, which measured 6.7 ms at 1024; (b) an AOTriton head_dim-512 image -- **closed as vendored**: the shipped `aotriton.images/amd-gfx11xx/flash/attn_fwd` set is 12 files, all `*bf16@16_256_*`, and `_AOTRITON_PREFILL_HEAD_DIMS` is `(256,)`; (c) an in-tree WMMA kernel. K and V share the raw projection (no `attn_v`), but not the cached tensor: K uses weighted head normalization and RoPE; V uses weightless normalization without RoPE (`gemma4_layer.py`). Load the distinct K and V tiles. GQA is 8 query heads per KV head, so pack them into the WMMA M dimension. | 39.2 ms at 1024 (8.1%), 610.4 ms at 4096. `gemma4_attention_decode_class_kernel<…,2,2>` runs 10 / 40 launches. llama.cpp: 6.7 / 85.7 ms. | **~32 ms** (+6.6%) / **~525 ms** | open | |
| P3 | moe.l29_experts | **Give layer 29's experts a fast owner.** Its Q5_K gate_up runs `gguf_q4_k_selected_dual_grouped_rowbatch` (21.9 ms / 1024) and its Q8_0 down runs `gguf_k_selected_prefill_out_kernel<…,8>` (44.4 ms / 1024). Step 1: determine whether this is routing or a missing leaf. `gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out` guards only on `in_features % 32`, and 704 passes. Step 2: add a Q5_K tile loader to the int8 MMQ32 gate_up leaf. | One layer costs 13.8% of prefill. llama.cpp runs it through its ordinary MMQ in 2.6 ms. | **~60 ms** (+13%) / ~258 ms | open | |
| P4 | moe.gate_up | **Larger int8 MMQ tiles for the Q4_K gate_up.** llama.cpp runs `mul_mat_q<Q4_K, mmq_x=64>` (64 weight rows × 64 tokens per tile, 128 threads, occupancy 2) at 49.4 ms. hipEngine's `q4_k_selected_dual_q8_1_ds4_mmq32_prefill_compact32` runs 112.6 ms over two 512-token blocks. Candidates: J = 64 tokens per tile (a 1024-token prompt averages 64 rows per expert), I = 64 rows, and one launch across all experts. | 2.3× behind llama.cpp at both lengths. The 2026-09-27 comparison measured hipEngine's MoE experts at about 16 TFLOP/s against about 33 TFLOP/s for llama.cpp. | up to ~63 ms / ~260 ms | open | |
| P5 | dense.q8_0 | **Single-plane int8 MMQ for the dense Q8_0 projections**, llama.cpp-shaped: `mul_mat_q<Q8_0, 128>` with I = 64 rows and J = 128 tokens, a Q8_1 activation quantize fused into the preceding norm, and no guard planes. The refuted route (iterations 150 and 155) is the guarded three-plane `d4x3` chain, which reproduces exact results with three int8 planes per tile, so it is a structurally different kernel. | 115.6 ms against 59.0 ms (2.0×). Dense BF16 WMMA runs about 6 TFLOP/s. | up to ~57 ms / ~228 ms | open | |
| P6 | dense.q8_0 | **Concatenate q/k/v, and the shared-MLP gate/up, into one weight each at load.** This must be a replacement layout, not an added copy. Fewer launches (412 → ~180 per prefill) and fuller tiles on the 1024 / 2048 / 2112-row outputs. | 412 dense launches per 1024-token prefill. | 5–15 ms (unmeasured) | open | |
| P7 | moe.route_glue | **Replace the MoE scheduler.** `qwen35_moe_group_compact_active_kernel` costs 22.2 ms / 1024 and 88.2 ms / 4096, plus `gather_packed_hidden` at 6.4 / 25.4 ms. Replace them with the Qwen3.6 path's count / prefix / scatter scheduler, or a `mm_ids_helper`-style sort. | llama.cpp's whole routing family is 10.5 ms / 1024. | ~24 ms / ~96 ms | open | |
| P8 | moe.router_gemm | **Router logits through WMMA or hipBLASLt, with the prescale fused in.** `qwen35_router_logits_token_tile_kernel` (F32 weight, 2816 × 128) runs about 1.7 TFLOP/s: 12.8 ms / 1024. Fold `gemma4_router_prescale` (RMSNorm without weight × scale) into the A load. Fuse softmax + top-8 + renormalization + per-expert scale into one kernel, as llama.cpp's `topk_moe_cuda` does (0.22 ms / 1024). | 13.6 ms against 4.0 ms. | ~10 ms / ~38 ms | open | |
| P9 | moe.act_quant | **Fuse the Q8_1 activation pack into `pre_ffw_norm_2`.** The norm already reads the row. | `gguf_q8_1_mmq_ds4_pack_bf16` costs 5.7 / 22.8 ms. | ~4 ms / ~17 ms | open | |
| P10 | moe.down | **A llama.cpp-shaped Q5_1 MMQ for the down projection** (I = 64, J = 64, K = 704 as 22 blocks of 32). llama.cpp's `mul_mat_q<Q5_1, 64>` reaches about 31 TFLOP/s on this shape. hipEngine's grouped BF16 WMMA owner reaches about 20. The int8 leaf that iteration 142 refuted (4 TFLOP/s) is a different, 32-row design. | 46.8 against 30.2 ms. | ~17 ms / ~71 ms | open | |
| P11 | whole layer | **Run the shared-expert MLP branch and the MoE branch on two streams.** llama.cpp forks at the branch point (its `concurrent_events`), and its 187 ms of busy time fits in a 140 ms span. hipEngine is single-stream. This needs stream / event plumbing in the Gemma layer loop. | llama.cpp overlap ≈47 ms / 1024. | 20–40 ms (estimate) | open | |
| P12 | lm_head | **Run the lm_head only on the final prefill block.** It runs once per 512-token block today (2 launches at 1024, 8 at 4096, about 1 ms each), but only the final block's last row is used. | `gguf_k_pack8_prefill_out_kernel<…,float>` × blocks. | ~1 ms / ~6.7 ms | open | |
| P13 | whole prefill | **Re-test the 1024-token prefill block after P4 / P10.** Block 1024 was a wash with the current compute-bound MoE kernels (`0b2ce993a`). Larger MMQ tiles make rows-per-expert matter again. | llama.cpp uses a 1024-token ubatch. | unknown | blocked (P4) | |
| P14 | attn.sliding | **Exploit the mask's lower bound in pass 3 of the multi-row exact kernel.** Pass 3 already derives `last_active` from the mask row (a byte walk plus a warp max-reduce) and walks `[0, last_active + 1)`. The *first* kept key is never computed, so a sliding row still starts at key 0 and walks the whole causal triangle. Add the mirror reduction (`first_active`, one more comparison in a loop that already runs, min-reduced the same way) and start pass 3 there; bound pass 2's range likewise. **Bit-exact by the argument the existing tail trim already relies on**: a masked key contributes weight 0 to every dimension, so dropping it leaves the ascending-key accumulation unchanged -- the same argument `key_begin` makes for one-row blocks. No AOTriton dependency, and it applies to any mask with a lower bound, so an eviction policy gets it too. Composes with P1: it covers the blocks AOTriton still refuses. | Pass 3 is the dominant pass and is memory-latency bound -- the kernel's own comment records 34 GB/s unique against 493 GB/s with the loads removed. **Measured: the per-block cost of the causal mask tracks the pass-3 walk `sum(q+1)` to within about 5% over the first four blocks** (1.94 / 5.81 / 9.65 / 13.50 ms against ratios 1 / 3 / 5 / 7), which is direct evidence that pass 3 sets the cost. Today it walks `[0, q]` on sliding *and* global rows alike -- the window's lower bound reaches pass 1 alone, and that is worth +29.9% over the whole prefill. Cutting the sliding walk to its window drops the keys walked from 8.39M to 3.67M per layer at 4096, a **2.29x** reduction on that pass, on top of the window saving already banked. Global layers gain nothing (a causal row's first kept key is already 0). | hypothesis: up to ~2.3x on sliding pass 3, whose share of the 1478.7 ms is unmeasured | open | |

If P1–P3 and P7–P9 hit their estimates, a 1024-token prefill falls from 518 ms
to about 385 ms (~2650 tok/s), and a 4096-token prefill from 3.87 s to about
1.7 s (~2400 tok/s). P4, P5, P10 and P11 are what close the rest of the gap to
llama.cpp.

### Decode

| ID | Family | Candidate | Evidence | Est. gain (ms / token, @1024 / @4096) | Status | Result |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | moe.gate_up | **A real MoE GEMV for Q4_K.** Today decode runs `gguf_q4_k_selected_prefill_out_kernel`: 1408 × 8 workgroups of 128 threads, 16 VGPRs, 187 µs per launch, about 91 GB/s. Port llama.cpp's `mul_mat_vec_q<Q4_K>` with expert ids: quantize the activation to Q8_1 once per layer, sudot4 dot products, several rows per wave, and all 8 experts in one launch. **Check for a regression first.** Iteration 40 measured both MoE projections at 0.207 ms per layer (6.2 ms per token) through `selected_gemv_bf16_bf16_out`. They now cost 8.25 ms per token through a `*_prefill_out` owner. Confirm whether the resolver `13532ae2b` introduced sends one-row decode to the prefill owner. | 5.42 ms against llama.cpp's 0.71. Floor 0.63. | **~4.7 / ~4.1** (decode 43 → ~53 tok/s alone) | open | |
| D2 | attn.sliding | **Flash-decoding (split-K) for the sliding layers.** `gemma4_attention_decode_dimension_kernel` runs 25 launches per token of 128 wave32 workgroups (grid 4096 threads, about one wave per CU on 96 CUs) at 116 µs, plus the class kernel at 55 µs. Replace it with many (KV-split × KV-head) workgroups, 2 query heads packed per KV head, and a combine kernel; llama.cpp uses `flash_attn_tile<256,256,1,2>` + `flash_attn_combine_results`. The key-slice split of iterations 36–38 failed the corrected gate, so the new kernel must pass the teacher-forced gate. Reassociation itself is allowed on the production route. | 4.28 ms against 0.35 ms. Floor 0.26. | **~3.9 / ~3.2** | open | |
| D3 | moe.down | **A Q5_1 MoE GEMV that reaches bandwidth.** `q5_1_selected_gemv_bf16_bf16_kernel` runs 22528 workgroups × 256 threads at 97 µs, about 122 GB/s. Q5_1's 24-byte block defeats 16-byte loads. Options: an mmvq-style kernel on Q8_1 activations, or a load-time *replacement* repack into 16-byte-aligned planes (qs / qh / d·m). | 2.83 ms against 1.28 ms. Floor 0.42. | ~1.5–2.4 / ~1.4–2.2 | open | |
| D4 | attn.global | **Split-K flash-decoding at head_dim 512.** The same design as D2. Preserve distinct cached K and V loads: sharing the raw projection does not share their normalization or RoPE transforms (P2). | 0.75 → 2.82 ms from 1024 to 4096 context, against llama.cpp's 0.11 → 0.15. | ~0.6 / **~2.7** | open | |
| D5 | moe.route_glue + router | **One fused routing kernel for single-token decode:** RMSNorm without weight × scale → router logits (2816 × 128) → softmax → top-8 → renormalization and per-expert scale → expert ids. The one-token path needs none of `compact_active` / `prefix` / `count` / `lane_to_row` / `fill`. This matches llama.cpp's single `topk_moe_cuda` launch. | 240 glue launches per token (1.29 ms) plus router 0.33 ms, against 0.55 ms. | ~1.1 / ~1.1 | open | |
| D6 | norm | **Fuse the per-layer norms that share an input.** Three normalizations read `attn_out` (`ffn_norm`, `pre_ffw_norm_2`, and the weightless router norm), and the tail chain is `post_ffw_norm_1`, `post_ffw_norm_2`, add, `post_ffw_norm`, residual, then layer scale. Emit them from 2–3 kernels per layer. The fusion that iteration 45 refuted was Qwen's rmsnorm+rotate, which serializes a reduction before a transform. It is not this multi-output same-input shape. | 301 norm launches per token (10 per layer), 1.51 ms. | ~0.5–1.1 | open | |
| D7 | moe / elementwise | **Fold the MoE weighted accumulate, the GEGLU and the branch add into neighbouring epilogues.** Weighted accumulate goes into the down GEMV's epilogue, and the GEGLU into the gate_up GEMV. | `gemma4_moe_weighted_accumulate` 0.47 ms per token, GEGLU 0.12, branch add 0.05. | ~0.5 | open | |
| D8 | host / launch | **Cut launches, then graph the decode step.** A token takes 1146 launches. Investigate the 97 per-token rocclr copies / fills (0.24 ms busy); zeroed counters can move into the count kernel where ownership permits. After D5–D7, use X7 to remove host dependencies from the captured region and enable decode graph replay for Gemma. Measure host launch and synchronization gaps directly before assigning a gain. | Launch census above. The unprofiled step and profiled busy sum use different instrumentation; their difference does not measure recoverable overhead. | unknown until host/timeline attribution | open | |
| D9 | dense.q8_0 | **Fuse q/k/v, and the shared-MLP gate/up, into one GEMV each** (206 → ~120 launches), and tune the pack8 GEMV for the 2112 / 4096 / 8192-row shapes. It is already ahead of llama.cpp, but runs at 579 GB/s against a 2.14 ms floor. | 3.02 ms against a 2.14 ms floor. | ~0.5–0.9 | open | |
| D10 | lm_head | **Fuse greedy argmax (and the final softcap) into the lm_head epilogue**, so greedy decode never materializes 262144 logits. Tune toward the 0.96 ms floor. | 1.15 ms. Sampling and softcap time is outside this table. | ~0.2+ | open | |
| D11 | moe.l29_experts | **Decode owners for layer 29's Q5_K / Q8_0 experts** (together with P3). | 0.24 ms against 0.03 ms. | ~0.2 | open | |
| D12 | attention (long context) | **INT8 KV for Gemma decode at long context**, reusing the Qwen INT8 KV machinery. Start after D2 / D4. It halves the attention bytes that dominate from about 8k context. | Attention bytes scale with the global layers' context. | context-dependent | blocked (D2, D4) | |

The combined D1–D8 forecast needs recalculation after D8's host costs are
attributed. Do not count the difference between unprofiled wall time and
profiled device busy as a graph-capture gain. D1 and D2 alone have a combined
hypothetical saving of about 8.6 ms (to ~69 tok/s); confirm that estimate with
paired end-to-end runs rather than assuming independent gains add.

### MTP

Gemma 4 ships a trained MTP "assistant", `MTP/mtp-gemma-4-26B-A4B-it-Q8_0.gguf`
(440 MB, arch `gemma4-assistant`). hipEngine does not implement it:
`Gemma4GGUFGenerator.supports_speculative_mtp = False`.

**What the assistant is** (from the GGUF and llama.cpp's
`src/models/gemma4-assistant.cpp`):

- 4 layers (3 sliding, 1 global), hidden size 1024, dense GEGLU FFN of 8192.
- Q-only attention with no K or V projections. It reads the **target's** KV
  cache: the sliding layers read target layer 28's cache, and the global layer
  reads layer 29's. It keeps no KV cache of its own.
- Input is `concat(target_embed(token) · √2816, h)` → `nextn.pre_projection`
  (5632 → 1024). `h` is the target's post-final-norm hidden state, which is
  the lm_head input.
- Output is `output_norm` → tied `token_embd` (262144 × 1024 Q8_0, 285 MB) as
  the draft lm_head. `nextn.post_projection` (1024 → 2816) produces the next
  step's `h`, so draft steps chain recurrently.

**Comparator (measured).** llama.cpp `a97cce8` `llama-server --spec-type
draft-mtp -md <assistant> --spec-draft-n-max N`, greedy, 5 chat prompts × 256
tokens after one warm-up request, BF16 KV, `--no-cache-prompt`:

| Draft length | XTX tok/s | W7900 tok/s | Acceptance |
| ---: | ---: | ---: | ---: |
| none | 86.54 | 74.72 | — |
| 1 | 139.98 | 118.65 | 84.4% |
| 2 | 156.54 | 131.97 | 74.7% |
| **3** | **164.59** (1.90×) | **139.26** (1.86×) | 67.3% |
| 4 | 157.39 | 131.80 | 59.2% |
| 6 | 150.10 | 128.12 | 48.1% |

At draft length 3, a cycle yields about 3.0 tokens (1 + 3 × 0.673).

| ID | Area | Candidate | Evidence / notes | Est. gain | Status | Result |
| --- | --- | --- | --- | --- | --- | --- |
| M1 | loader | **Load `gemma4-assistant` GGUFs** and admit them on capability: the arch, plus `embedding_length_out` equal to the target's hidden size. Also admit on the shared-KV layer mapping (sliding → last sliding target layer, global → last global target layer). Auto-discover `MTP/*.gguf` next to the target. | Tensor census in the list above. | prerequisite | open | |
| M2 | target | **Export `h`**, the post-`output_norm` hidden state of every verified row, from `Gemma4Runner.forward`. The runner currently normalizes only the last row into a single-row buffer. Extend normalization and storage to all verifier rows, then expose the same states consumed by the multi-row lm_head without recomputing them. | llama.cpp exposes the same tensor as `t_h_nextn`. | prerequisite | open | |
| M3 | draft step | **A draft step on the existing decode kernels:** embedding row lookup, pre-projection GEMV, 4 layers (q GEMV, head norm, RoPE, decode attention over the *target's* layer 28 / 29 cache, o GEMV, norms, GEGLU FFN), output norm, lm_head, and post-projection. Per draft token that reads about 285 MB (head) + ~140 MB (layers), a floor of about 0.52 ms on the XTX. | Tensor sizes from the GGUF. | enables MTP | open | |
| M4 | draft lm_head | **Fused GEMV + argmax for the 262144 × 1024 draft head**, the largest draft cost. Separately, check whether the HF assistant checkpoint carries masked-embedding centroids. llama.cpp loads them optionally (`MASKED_EMBD_CENTROIDS` / `ORDERING`), but this GGUF has none. A clustered sparse head could cut draft-head bytes about 10×. | Draft head ≈ 0.35 ms per draft token at 816 GB/s. | ~0.3 ms per draft token | open | |
| M5 | verify | **Multi-row target verify** (1 + n rows, n ≤ 3) at no more than ~1.3× a single decode step. Today's decode kernels are one-row: MoE, attention and dense GEMV would each cost about n+1 times. Needed: an MoE GEMV over up to 32 expert rows, where D1's mmvq-with-ids shape handles several tokens natively; decode attention for 2–4 query rows (D2 / D4 with a q-row dimension); and dense GEMV at rows 2–4. This is the economic crux of MTP. | Every verify runs the full target model. | decides MTP's net | open | |
| M6 | policy | **Draft length.** Start at 3, which peaked on both GPUs for llama.cpp. Then make it acceptance-adaptive with `hipengine/speculative/adaptive_budget.py`. | Comparator table. | ±10% | open | |
| M7 | product | **`speculative_mtp: true` on `LLM.generate()` and `hipengine serve`.** An explicit request either runs or fails naming the missing capability. Greedy acceptance first, then sampled acceptance through `hipengine/generation/mtp_sampled_accept.py`. On rejection, commit only the accepted prefix through `KVLiveSpans`: drafts write no KV, but target verification may write candidate rows before acceptance is known. Make rejected rows invisible and select the matching assistant hidden state; M8 tests those transitions. | Product rules in `AGENTS.md`. | — | open | |
| M8 | state correctness | **Verify speculative commit and rollback.** Cover all accepted, rejection at every draft position, EOS inside a draft, output-limit truncation, context exhaustion, and consecutive requests. Assert committed KV positions, assistant hidden-state selection, and exclusion of rejected rows from later attention. Check continuation against the no-MTP reference under the declared acceptance/numerical contract, including sampled acceptance when implemented. | The assistant reads target KV; target verification can write rows that are later rejected. Test exact state ownership separately from arithmetic quality and speed. | correctness prerequisite | open | |

**MTP economics (estimate).** At today's 23.2 ms target step: a 1.3×
verify (30.2 ms) plus three drafts (about 3 ms) yields 3.0 tokens per 33.2 ms,
about 91 tok/s (2.1×). With a hypothetical target step of 9 ms, the same arithmetic gives about
3.0 tokens per 14.7 ms, roughly 200 tok/s. This is a sensitivity example, not
a forecast that the decode punchlist will reach 9 ms. MTP speedup depends on
measured acceptance and verifier cost after each optimization. D1, D2 and M5
share kernel work: a multi-row MoE GEMV and multi-row flash-decoding serve both.

### Cross-cutting

These do not belong to one phase or one family.

| ID | Area | Candidate | Evidence / notes | Est. gain | Status | Result |
| --- | --- | --- | --- | --- | --- | --- |
| X1 | KV dtype | **Evaluate f16 KV storage against the bf16 default.** llama.cpp's f16 comparison is recorded above, but its BF16 path pays a BF16-to-F16 conversion. Establish hipEngine's actual conversion and attention routes before predicting a benefit. Gemma's runner allocates BF16 caches in `hipengine/runtime/gemma4.py`. | No equivalent conversion penalty has been established for hipEngine. KV dtype changes arithmetic and needs V4 plus the applicable production-profile gate. | unknown; do not transfer llama.cpp's percentage | open | |
| X2 | capacity | **Track XTX headroom on every candidate that allocates.** Peak is 23.58 GB of 25.75 GB (91.6%) at 1024/128 today. P6 (concatenated replacement layouts), P11 (a second stream), D8 (graph capture) and D12 (INT8 KV) all move it, in both directions. | The XTX already failed to load once at 32.35 GB, and the fix was a layout change. A candidate that adds scratch needs its peak measured, not assumed. | constraint | open | |
| X3 | lm_head | **Cut the lm_head bytes.** It is the tied Q8_0 `token_embd`, 262144 x 2816: 784 MB per token, 0.96 ms of the 4.9 ms memory floor -- about 20% of the floor for one projection. A lower-precision or clustered head changes output arithmetic and needs the gate. | 1.15 ms against a 0.96 ms floor. D10 fuses the argmax; this reduces what has to be read at all. | ~0.2-0.9 | open | |
| X4 | attention | **Re-evaluate the vendored AOTriton release.** P1 and P2 are both written against 0.11.2b. A newer release may ship a head_dim-512 image (P2's option b) or better windowed coverage. | P2 records no head_dim-512 image in the vendored release. Check newer releases' image inventories and API compatibility; availability and performance on gfx1100 remain to be tested. | unknown | open | |
| X5 | prefill | **Prefill graph capture.** Measure host launch, transfer, and synchronization costs separately from kernel time, then test capture for the production chunk shapes. Coordinate stable device buffers and position/mask updates with X7. | Kernel-busy fractions do not measure hardware occupancy or prove a recoverable host gap. Validate capture/replay against uncaptured execution and report end-to-end time to first token. | unknown until host/timeline attribution | open | |
| X6 | scope | **Batching is untested and out of scope for this page.** Everything here is single-request. llama.cpp's server batches, hipEngine has `hipengine/dispatch/batch.py` and `batch_scheduler.py`, and the `KVLiveSpans` ABI is batch-shaped. A serving comparison is a different measurement with a different ranking. | Recorded so the single-request numbers are not read as serving numbers. | — | note | |
| X7 | runtime / sampling | **Keep the generation path device-resident where practical.** Keep logits, softcap, token selection, and reusable position/mask data on device; transfer only the results the caller needs. Preserve an explicit host-logits path for diagnostics and numerical gates. Integrate D10's greedy epilogue and prepare stable buffers for D8/X5 capture. | `Gemma4Runner._forward_block` copies vocabulary logits to the host and applies NumPy softcap; `next_token` selects on the host. Attribute those costs directly. Check softcap rounding/ties, supported sampling semantics, route selection, and request isolation. | unknown; overlaps D8/D10 | open | |
| X8 | KV capacity | **Bound sliding-layer KV storage by the live window.** Explore a replacement layout for expired sliding entries while retaining full live global context. Preserve absolute positions, chunked prefill, assistant reads, and speculative rollback; use `KVLiveSpans` and do not overwrite entries still needed by a verifier or reader. | `Gemma4Runner.__post_init__` allocates full-capacity K and V for every layer. Use the tier-1 allocation probe before full-prompt validation; measure capacity benefit and peak scratch before claiming speed. V5/M8 cover the relevant state transitions. | capacity benefit to measure; speed unknown | open | |

## Measured dead ends

These were measured on this model and lost. Re-open one only with new evidence
that changes the premise. The decode-attention rows below were measured against
the pre-repair split decode at 45.75 tok/s (1024/128), before the correctness
repair that cost 2.0 tok/s; the deltas are same-host paired, so the absolute
baseline moving does not change them.

| Idea | Result | Where |
| --- | --- | --- |
| Prefill block 1024 instead of 512 with the current MoE kernels | Wash: +1.5% at 2048 tokens, −2% decode, larger scratch | commit `0b2ce993a` |
| Dense Q8_0 through the guarded three-plane `d4x3` MMQ chain | 1.46–1.52× slower than the BF16 WMMA owner | iterations 150, 155 |
| Int8 MMQ leaf for the MoE down (32-row design) | 8.2× slower (4.0 TFLOP/s); the down stays on WMMA | iteration 142 |
| Registered Q5_1 decode variants `logical256_t128` / `wave64` | −46% decode | iteration 42 |
| MoE expert-grid retune, wider out-blocks, occupancy ladder | Neutral or refuted | iterations 101, 102, 112, 131 |
| Rows-per-expert reuse for the MoE | Refuted: the MoE is bandwidth-bound, and one routed row already costs ~55× the arithmetic and ~65× the byte time | iteration 126 |
| Key-slice split decode attention | Failed the corrected teacher-forced gate; replaced by dimension partitioning | iterations 36–38 |
| Dimension-partition decode attention, full grid | −4.4% decode (43.73 against 45.75), `kl_max` 0.0 | `2026-09-26-gemma4-dimension-full-grid-rejected.json` |
| Dimension-partition decode attention, prefetch 16 | −9.7% decode (41.31 against 45.75), `kl_max` 0.0 | iteration 55, `…-dimension-prefetch16-rejected.json` |
| Dimension-partition decode attention, four-key tiled value pass | −10.8% decode (40.92 against 45.86), `kl_max` 0.0 | iteration 57, `…-dimension-tiled-rejected.json` |
| FP64 accumulation in the decode slice kernel | **Numerical failure**: `kl_max` 0.166 against a 0.05 bar, despite 100% top-1 agreement | iteration 51, `…-fp64-slice-rejected.json` |
| Two-plane Q8 MMQ instead of the three-plane `d4x3` | Speed a wash; the real cost is a load-time repack and its memory | iteration 89 |
| Routing dense Q/K/V/O through the Q8 MMQ plane by reordering dispatch | 29% slower; the order is what keeps it off | iteration 150 |

Two of these bound live candidates, so read them before opening P4 or D2.
**D2 is the fourth entry in the dimension-partition design space that has lost**
(full grid, prefetch 16, tiled value pass, plus the key-slice split before them).
All three dimension-partition variants were numerically exact — `kl_max` 0.0 — and
lost on speed alone, at 4.4%, 9.7% and 10.8%. A flash-decoding replacement for
the sliding layers has to explain why it is not the same kernel again; the
one-row-at-a-time structure is what those four share, and D2's KV-split with a
combine kernel is a different decomposition, but the burden is on the candidate.
**P4 assumes rows-per-expert matters, and iteration 126 refuted that** for the
current owner: the MoE is bandwidth-bound and quadrupling weight reuse did not
help. P4's premise is tile shape and occupancy against llama.cpp's `mmq_x=64`,
which is a different claim — but state which one the change rests on.
