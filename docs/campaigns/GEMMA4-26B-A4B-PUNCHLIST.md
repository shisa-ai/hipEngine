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
- hipEngine `gemma4`, BF16 KV, default production route. The tables below were
  taken at `df867de1f` and reproduced at `6d886f473` -- 8 commits later, none of
  them touching `hipengine/` -- by
  `benchmarks/results/2026-09-28-gemma4-scoreboard-baseline.json`, which is the
  artifact every later optimization is compared against. See
  [Baseline at HEAD](#baseline-at-head) for what reproduced and the one row that
  did not.
- The prefill columns now carry
  `benchmarks/results/2026-09-28-gemma4-scoreboard-p14fix.json`, captured after
  P14 landed (the class kernel's pass 3 walks only `[first_active,
  last_active + 1)` instead of starting at key 0). Every non-sliding prefill
  family moves by at most 1.4% against the baseline artifact -- the one larger
  figure is 0.1 ms of copy/fill, 2.1% of 1.0 ms -- so those cells remain
  interchangeable with the values measured alongside llama.cpp. The intended
  change is `attn.sliding` at 4096, -18.1%. Decode columns are untouched, since
  P14 does not run during decode.
- llama.cpp upstream `a97cce8`, HIP build with `-DAMDGPU_TARGETS=gfx1100
  -DGGML_NATIVE=OFF`, flags `-ngl 99 -fa 1 -ctk bf16 -ctv bf16 -b 4096 -ub 1024`.
- Measured 2026-09-28 with both GPUs otherwise idle.

### Headline (RX 7900 XTX, unprofiled)

| Shape | hipEngine | llama.cpp | hipEngine as % |
| --- | ---: | ---: | ---: |
| Prefill, 1024 tokens (tok/s) | 1983 | 4969 | 39.9% |
| Prefill, 4096 tokens (tok/s) | 1146 | 4505 | 25.4% |
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

### Baseline at HEAD

`benchmarks/results/2026-09-28-gemma4-scoreboard-baseline.json`, captured on
the XTX at `6d886f473` by `scripts/gemma4_scoreboard_snapshot.py --gpu 1
--tag baseline`. It bundles both halves, four per-run artifacts
(`-topline-{1024,4096}-baseline.json`, `-family-{1024,4096}-baseline.json`),
the exact commands, the prompt-token hash and the toolchain (HIP 7.2.53211).
Two runs 112 s apart agreed to within 0.4% on every row, so this artifact is the
reference a landed change is diffed against.

| Measurement | Baseline (pre-P14) | Table above | Delta |
| --- | ---: | ---: | ---: |
| Prefill 1024 (tok/s) | 1979.7 | 1983 | +0.2% |
| Prefill 4096 (tok/s) | 1061.6 | 1146 | +7.9% |
| Decode at 1024 (tok/s) | 43.87 | 43.15 | +1.7% |
| Decode at 4096 (tok/s) | 40.19 | 39.79 | +1.0% |
| Prefill busy 1024 (ms) | 483.1 | 479.4 | −0.8% |
| Prefill busy 4096 (ms) | 3751.8 | 3474.4 | −7.4% |
| Decode busy at 1024 (ms) | **19.47** | 21.30 | **−8.6%** |
| Decode busy at 4096 (ms) | 21.56 | 21.59 | −0.1% |
| Launches per decode token | 1146 | 1146 | 0 |

At capture time every prefill family reproduced within 0.4% and decode at 4096
within 1%. P14 has since landed, so the prefill rows above now read as P14's
effect rather than as reproduction error -- `attn.sliding` at 4096 supplies
essentially all of the +7.9% headline delta, and the decode rows, which P14
does not touch, still carry the reproduction check. One row never reproduced:
decode at 1024 measures 19.47 ms of device busy against the table's 21.30. The
whole difference sits in `moe.gate_up` (4.79 against 5.42),
`attn.sliding` (3.65 against 4.28), `moe.down` (2.64 against 2.83) and
`lm_head` (0.97 against 1.15) -- and in this baseline every one of those is the
same at 1024 as at 4096, which is what their inputs predict, since none of them
reads the context. Only `attn.global` moves between the two lengths (0.75 ->
2.82), as it must. The published 1024 and 4096 columns for those families
differ where they should not, and this baseline's own 4096 column matches the
published 4096 column to 1%. Launch counts are identical (1146 per token), so
what moved is time per launch, not work. Treat it as clock or thermal state in
the older 1024 trace.

**Use this artifact, not the table's 1024 decode column, as the hipEngine decode
baseline.** The paired table keeps its cells: its llama.cpp column was measured
in the same session as its hipEngine column, and replacing half a pair from an
unpaired run would break the pairing the head-to-head rests on. Re-pairing that
row needs a fresh llama.cpp trace, which this baseline does not include.

The prefill columns are the exception, and now carry the `p14fix` artifact:
every non-sliding family moved at most 1.4% against this baseline, so the
substituted cells are interchangeable with their paired originals at that
tolerance, while the one row that does move is the change being published.
Decode does not get that latitude -- its unexplained spread is 8.6%, so its
pairing stays intact until a fresh llama.cpp trace exists to re-pair it with.

### Prefill families (RX 7900 XTX, milliseconds per prefill)

These are device-busy sums from `rocprofv3 --kernel-trace`. hipEngine uses its
steady-state (second) prefill. llama.cpp uses the per-run average of two
`llama-bench` runs. "Gap" is hipEngine minus llama.cpp. hipEngine's columns are
`benchmarks/results/2026-09-28-gemma4-scoreboard-p14fix.json`, taken after P14
landed; the pre-P14 reproduction of these same cells is recorded under
[Baseline at HEAD](#baseline-at-head).

| Family | hipEngine 1024 | llama.cpp 1024 | Gap 1024 | hipEngine 4096 | llama.cpp 4096 | Gap 4096 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| attn.sliding (25 layers) | 29.6 | 12.3 | +17.3 | **1211.3** | 84.0 | **+1127.3** |
| attn.global (5 layers) | 38.7 | 6.7 | +32.0 | **612.2** | 85.7 | **+526.5** |
| dense.q8_0 (q/k/v/o, shared MLP) | 115.1 | 59.0 | +56.1 | 458.7 | 233.4 | +225.3 |
| moe.gate_up (Q4_K, 29 layers) | 112.3 | 49.4 | +62.9 | 451.4 | 194.2 | +257.2 |
| moe.down (Q5_1, 29 layers) | 46.9 | 30.2 | +16.7 | 190.4 | 120.4 | +70.0 |
| moe.l29_experts (Q5_K / Q8_0) | **66.9** | 2.6 | **+64.3** | 272.7 | 10.4 | +262.3 |
| moe.route_glue (schedule, gather, combine) | 34.1 | 10.5 | +23.6 | 137.0 | 41.2 | +95.8 |
| moe.router_gemm (router logits, prescale) | 13.4 | 4.0 | +9.4 | 53.1 | 15.5 | +37.6 |
| moe.act_quant (Q8_1 activation pack) | 5.6 | 1.5 | +4.1 | 22.7 | 6.0 | +16.7 |
| norm | 9.0 | 11.9 | −2.9 | 35.5 | 46.8 | −11.3 |
| elementwise (rope, GEGLU, adds) | 4.6 | 4.2 | +0.4 | 18.4 | 16.5 | +1.9 |
| attn.kv_write | in attention | 2.0 | −2.0 | in attention | 9.2 | −9.2 |
| lm_head | 2.0 | 1.1 | +0.9 | 7.8 | 1.1 | +6.7 |
| runtime copy / fill | 1.1 | 1.3 | −0.2 | 3.0 | 1.7 | +1.3 |
| **Device busy** | **479.4** | **196.7** | **+282.7** | **3474.4** | **866.2** | **+2608.2** |

llama.cpp runs a second HIP stream during prefill. Its shared-expert MLP branch
overlaps the MoE branch, so its busy total (≈187 ms in a warm run) fits into a
≈140 ms span. hipEngine's single-stream kernel-busy fraction is about 94%
(479.4 ms busy in a 510.0 ms span). This is a timeline fraction, not hardware
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
| attn.global | 0.75 | 0.11 | +0.64 | **2.82** | 0.15 | 21 MB → 84 MB | 0.03 → 0.10 |
| moe.router_gemm | 0.33 | 0.11 | +0.22 | 0.32 | 0.11 | 43 MB | 0.05 |
| runtime copy / fill | 0.24 | 0.05 | +0.19 | 0.23 | 0.05 | — | — |
| moe.l29_experts | 0.24 | 0.03 | +0.21 | 0.24 | 0.03 | 39 MB | 0.05 |
| elementwise | 0.23 | 0.26 | −0.03 | 0.23 | 0.26 | — | — |
| attn.kv_write | in attention | 0.73 | — | in attention | see note | — | — |
| **Device busy** | **21.30** | **9.65** | **+11.65** | **21.59** | see note | ≈3.7 GB | **≈4.6** |
| **Unprofiled step** | **23.17** | **11.84** | | **25.13** | **12.24** | | |
| **Launches / token** | **1146** | ≈1330 | | 1146 | ≈1330 | | |

The hipEngine `@1024` column is the older, slower of the two hipEngine traces:
[Baseline at HEAD](#baseline-at-head) re-measures it at 19.47 ms busy against
the 21.30 below, with the difference sitting entirely in families whose cost
does not depend on context. Keep the cells paired as measured; diff changes
against the baseline artifact instead.

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
  (85% of the XTX's 960 GB/s). Its total, ≈4.6 ms per token (~215 tok/s) at
  1024 context, is the memory-bound ceiling for this artifact. llama.cpp runs
  at 39% of it and hipEngine at 20%. The attention rows count K and V
  separately: the global layers share one raw projection but cache distinct K
  and V tensors (`Gemma4Runner.__post_init__`).
- **Note on llama.cpp at 4096 context with BF16 KV.** The trace charges
  7.79 ms per token to `convert_unary<bf16→f16>`. llama.cpp converts the BF16
  cache to f16 for its flash kernels on every step, on a side stream. Its
  unprofiled step is still 12.24 ms (81.67 tok/s), so the conversion overlaps
  other work and the profiled busy sum (16.86 ms) overstates it. Treat
  llama.cpp's 4096-context per-family decode numbers as indicative only. f16 KV
  removes the conversion (85.82 tok/s).

### How to regenerate

hipEngine's two tables come from one command, so a landed optimization refreshes
both halves against a paired baseline:

```bash
# From the gemma4 worktree. 1 = XTX primary lane, 0 = W7900.
.venv/bin/python scripts/gemma4_scoreboard_snapshot.py --gpu 1 --tag p14
```

It warms the JIT cache, runs the campaign bench at each prompt length, traces the
same driver under `rocprofv3 --kernel-trace`, rolls the trace up by family, and
writes `<date>-gemma4-{topline,family,scoreboard}-<prompt>-<tag>.json` under
`benchmarks/results/`. Compare a candidate artifact against the `baseline`-tagged
one: same protocol, same GPU, same commit range. Raw traces land in `/tmp` and
are not committed.

The llama.cpp comparator is not in the script. The block below is the manual
recipe for both engines, still valid, and is what reproduces the llama.cpp
columns:

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
| V1 | attention | **Extend the campaign gate past the window: `--prompt 5120 --prefill 4096 --context 8192`** (1023 scored rows, the campaign row-count standard). The gate requires `prefill < prompt - 1`; capture the baseline and candidate with matching prompt, prefill, context, and block geometry. `capture_chain()` discards the prefill return value and scores subsequent single-token forwards, so this checks the cache produced by long prefill, not its output logits directly. Add direct prefill-logit comparisons under V2/V5. Every recorded gate run uses `--prefill 1024` or `--prefill 0` (13 and 3 occurrences, `GEMMA4-26B-A4B-OPTIMIZATION.md`), and the frozen `teacher-forced-prefilled-*-001c4fec3.npz` baselines are `--prompt 2048 --prefill 1024` -- two 512-row blocks with keys 512 and 1024, **both at or under the window and both on `attn_fwd`**. So no gate has ever exercised a prefill block with `keys > 1024`, which is the path P1 replaces. **Baseline frozen (W7900, `c3ea386af`):** `teacher-forced-prefilled-p4096-c3ea386af.npz`, 1023 rows, `max_block` 512; its self-gate passes with `kl_max` 0.0 and 0 top-1 flips (`benchmarks/results/2026-09-27-gemma4-prefill-4096-freeze-w7900.json`, `-selfgate-w7900.json`). Remaining: run P1 or P14 as the candidate against it on the same GPU. The XTX primary lane has no 4096 baseline yet. **Candidate run 2026-09-29 (W7900, HEAD `13564b97d`, identical geometry): FAILS `kl_max` only.** `kl_mean` 3.71e-4 (limit 1e-3), `kl_p95` 5.28e-6 (5e-3), `kl_p99` 1.12e-4 (2e-2), `top1_rate` 1.0 with 0 flips (0.99) all pass; `kl_max` 0.2177 against the 0.05 limit. **2 of 1023 rows clear the limit** -- row 894 (key 4991) at 0.2177 and row 327 (key 4424) at 0.0832; row 254 (key 4351) sits just under at 0.0413 and everything else is <= 0.0173 with p99 1.12e-4. Artifact `benchmarks/results/2026-09-29-gemma4-p4096-p14-gate-w7900.json`. **P14's own 4096 gate does not cover this:** `2026-09-28-gemma4-p14-gate-w7900.json` reports `kl_max` 0.0, but its candidate `c41aa2217` (01:55) predates P14's trim `7e9670c9a` (02:20), and `git diff c3ea386af c41aa2217` over both attention sources is empty -- that run compared identical attention code, so the trim itself has never been numerically gated. Attribution is open: 8 commits touch `hipengine/kernels/hip_gfx1100/gemma4/` since `c41aa2217` (P14 trim, P1's cell, MoE compaction x2, router tile, router selection, gather+Q8_1 fold), any of which can move decode logits. **Split path ruled out (2026-09-29):** re-running with `--slices 1` gives byte-identical statistics (`kl_max` 0.217717, rows 894/327/254, 0 flips) while `observed_routes` moves from `split_launches` 30690 / `selections [2]` to `split_launches` 0 / `selections [1]` / `split_key_range` null -- the override is confirmed applied (`candidate_forced_slices` 1). Every scored row sits at keys 4097-5119, so the default run exercised the split on all 1023 rows and `slices=1` bypassed it entirely, yet the output is unchanged. The deviation therefore exists before decode selects a path: it is in the KV cache prefill wrote or in layer kernels shared by both arms (MoE compaction, router tile/selection, gather+Q8_1 fold), not in the split. Artifact `benchmarks/results/2026-09-29-gemma4-p4096-p14-gate-slices1-w7900.json`. **Attributed by single-variable gating in a side worktree at `/mnt/nvme1` (added, run, removed; this tree never moved), in age order over `c41aa2217..HEAD`:** `7e9670c9a` (P14's pass-3 trim alone -- `git diff c3ea386af` against it touches only `gemma4_attention.hip`, 93+/14-) **passes with `kl_max` 0.0 exactly**, so P14's bit-exactness now has a measurement behind its argument; `cc44f33f2` (gather + Q8_1 fold) **passes with `kl_max` 0.0**; **`74fb74ffe` (select the router logits kernel by token count) fails at `kl_max` 0.224071** -- the deviation first appears here; `d0f5b1ff6` (router token tile at 128 threads) fails at 0.217717, as does HEAD. Rows 894 and 327 are the failing pair in every failing run, so the router change moves the same two positions throughout and the later tile change only modulates its magnitude. **Culprit: `74fb74ffe`, with `d0f5b1ff6` modifying it -- both router, neither attention.** **The router kernels are not defective** (`tests/test_gpu_gemma4_router_variant_selection.py`): at the production shape (512 x 2816 -> 128, top-8) the base and `token_tile_16` logits differ by 3.815e-06 -- exactly the 3.81e-06 the commit reports -- yet they select identical experts on **0 of 512 tokens disagreeing**, and both match a float64 reference on all 512. Selection is reproducible across launches. So the 4096 failure is not a routing flip or a bad kernel; it is a small arithmetic difference compounding through 8 prefill blocks x 30 layers, which makes this a **changed-arithmetic promotion question under `docs/EXECUTION-PROFILES.md`, not a bug**. Note `74fb74ffe` *was* gated when it landed, but against a temporary `gate_base_mmq.npz` at `--prefill 1024` where it passed at `kl_max` 1.30e-03 -- the exact configuration V1's row says had never been extended. Artifacts `benchmarks/results/2026-09-29-gemma4-p4096-attrib-{p14trim-pass,cc44f33f2-pass,74fb74ffe-fail,d0f5b1ff6-fail}-w7900.json`. **Routing-flip hypothesis now closed on the production chain (2026-09-29).** The 0-of-512 reading above was on random weights. `scripts/gemma4_router_variant_flip_probe.py` wraps the tile at the live call site and, during the gate's own prefill, also runs the untiled kernel into scratch from the same prescaled hidden, comparing both buffers elementwise and against float64. At the gate's exact geometry (`--prompt-tokens 5120 --prefill 4096 --context 8192`): **240 tile invocations (8 blocks x 30 layers), 122880 tokens compared, 0 selecting differently, max `|tiled - base|` 9.537e-07, and 0 rows where either kernel's top-8 differs from float64.** A free-running generation agrees (30 calls, 8310 tokens, 0 flips). So a flip is excluded on real activations as well, and the two prefill arms differ only in low-order logits bits that the diff above carries into the KV cache. **Promotion status: not admissible automatically, and the acceptance call belongs to the lead.** `docs/EXECUTION-PROFILES.md` classes this as T2 (reduction reorder), admitted to `production` only *after full gate*; §6.1's maximum-row-KL `5e-2` is a binding absolute ceiling rather than a tuning target, and 0.2177 sits 4.35x above it, with §6.1 additionally barring rows over `2e-2` from automatic admission even when the ceiling passes. The commit's own `--prefill 1024` gate (`kl_max` 1.30e-03) does not extend to 4096. **No bit-exact fix exists:** `scripts/gemma4_router_logits_variant_bench.py` records the *same* `2.861e-06` max float64 drift for both variants, so neither kernel is wrong -- the tiling reorders the reduction, and forcing identical order would undo the tiling. **Cost of reverting the prefill selection, measured in the commit:** the tile's entire benefit is prefill `+0.94%` (1024) / `+0.56%` (4096), decode unchanged, so restoring the untiled kernel on prefill buys a clean gate for that margin while the tile kernel, its bench, and `tests/test_gpu_gemma4_router_variant_selection.py` stay in the tree. Clearing command either way: `python3 scripts/gemma4_teacher_forced_gate.py gate --baseline /mnt/nvme1/gemma4-eval/teacher-forced-prefilled-p4096-c3ea386af.npz --prompt 5120 --prefill 4096 --context 8192`. | P1 changes a path with no numerical coverage today, and P1 is the largest estimated win on the board. **Now measured: the 4096 gate exists, HEAD does not pass it, and the cause is the router selection commit, not attention.** | candidate run 2026-09-29: **fails `kl_max` (0.2177 vs 0.05), 2 rows over, 0 flips -- attributed to `74fb74ffe` + `d0f5b1ff6`; P14 cleared at `kl_max` 0.0** |
| V2 | attention | **Build an independent oracle for the windowed path.** The gate compares two hipEngine arms against a captured baseline, so it cannot detect a wrong mask that both arms share. Options: score hipEngine's 4096-token prefill logits against llama.cpp's, or add a naive CPU reference for windowed attention. | The kernel-level parity test P1 proposes (keys 1536 / 2048 / 4096) is necessary but it is a kernel test, not a model-level one. | **answered (2026-09-29)** -- the naive-CPU-reference option is built and now runs at the gate's own geometry. `tests/test_unit_gemma4_gpu_kernels.py::test_windowed_attention_matches_the_reference_at_the_gate_shape` feeds production `_keep_mask` through the HIP kernel and compares to numpy at `start=3584 rows=512 keys=4096`, in **both** layer classes: sliding (head_dim 256, window 1024 -- the window binds, row 0 sees exactly 1024 of 4096 columns and every row is capped at `window`) and global (head_dim 512, `sliding_window=None` -- asserts pure causality with `max(axis=1) == keys`, so a window cannot leak where none is declared). Both agree at `atol=rtol=1e-5`. The routing seam was already closed: `test_unit_gemma4_attention_flash_admission.py` pins `keys=1024` admitting and `1025` refusing, so a binding window always reaches the exact kernel that reads `keep_mask` while the flash kernel reads no mask at all, and `13564b97d` binds the mask builder to the HuggingFace-gated reference. **What was missing was the pair at production scale** -- the existing composition test ran head_dim 16, six rows and a four-token window, which a bug spanning thousands of key tiles (the pass-1 skip over fully-masked tiles that V3 found, or the window bound past row 1024) would sail past. **Also fixed in the same pass:** three attention tests in that file constructed device allocations with no `@_needs_hip` guard, so a no-ROCm runner would fail them rather than skip -- the guard AGENTS.md requires. The llama.cpp-scoring option was not needed: the reference now covers the shared-mask risk that was the row's actual concern, which is the dependency V5 named when it wrote that both arms being hipEngine arms makes a shared wrong mask invisible. |
| V3 | attention | **Answered: the fallback applies the window mask. The 49x is a kernel swap plus the fallback's own quadratic growth.** The exact key-class kernel reads `keep_mask` and pass 1 skips fully-masked key tiles (`active[t] = mask_row[...] != 0`), so the window is applied. This rules out the missing-mask explanation, not every possible numerical defect; V1/V2 still cover model-level correctness. Measured at the prefill block shape (`rows=512 keys=4096 heads=16 head_dim=256`, causal mask vs causal-and-windowed): **26.75 -> 17.29 ms, so the window buys +35.4%** on that block, and **+29.9%** (121.18 -> 84.98 ms) summed over all 8 blocks of a 4096 prefill. Cross-checked against the model: the 5 global layers measure 122.1 ms/layer, and dividing by the **measured head_dim ratio of 1.327x** (mean over all 8 block sizes, range 1.240-1.398; not 2.0 -- the kernel is dominated by the key walk, so doubling head_dim costs only 1.327x) predicts 92.0 ms/layer for an unwindowed sliding row against 59.1 measured, a **~36%** window saving, which agrees with the direct 30% within the probe's launch-overhead bias. Both artifacts: `benchmarks/results/2026-09-28-gemma4-attention-window-value-w7900.json` and `-head-dim-ratio-w7900.json`, reproducible with `scripts/gemma4_attention_window_value_probe.py`. The 49x decomposes as 15.7x (fallback quadratic: 7.74 ms/layer at 1024 to 121.18 at 4096, matching the global layers' 15.6x) x 3.15x (fallback where 1024 used `attn_fwd`). | P1's premise and its ~1300 ms estimate both stand. The window already recovers ~30-38%, so the remaining win is the kernel swap plus passes 2 and 3's full key walk. P14 attacks that walk directly and is now the better-evidenced half. | **answered** |
| V4 | decode | **Re-run the decode gate at 4096 context with BF16 KV.** The recorded decode gate results are at 1024-key chains. D12 (INT8 KV) and D2 / D4 (new attention kernels) all change arithmetic at long context, and llama.cpp's own 4096-context per-family numbers are already flagged indicative because of its BF16-to-F16 conversion. | The long-context decode path is where D2 and D4 spend their estimate and it is the least-covered numerically. | **landed (baseline frozen and self-gated).** No new code and no kv-dtype flag: `HIPENGINE_KV_STORAGE` already defaults to `bf16` and D12/INT8 is `blocked`, so the existing evaluator covered it. Captured 2026-09-29 with `--prompt 5120 --prefill 4096 --context 8192` in 43.1 s: 1023 rows x 262144 vocab over `scored_key_range [4097, 5119]` and `split_key_range [1024, 5119]`, against the recorded baselines' `[1024, 2047]`. Self-gate on the same tree (the capture path's documented smoke) reports `kl_max/mean/p95/p99 = 0.0`, `top1_rate = 1.0`, `top1_flips = 0`, `passed`. npz 1072717924 B, sha256 `43538d0b1f201f42a1c50b73fd04a7553ed3f6f57a1fb10d4ffc3ab4f473ed56`, chain sha256 `3958e56f2e3752dd800875532e1d3997344df5bad28e4835a35352bedabe3630`; manifest records `performance_claim: false`. Held outside the repo at `~/.cache/hipengine/tmp/gate_v4_4096.npz`. **Awaiting a D2/D4 candidate to gate against it.** |
| V5 | attention / chunking | **Test window and chunk boundaries.** Cover prompt lengths 511/512/513 and 1023/1024/1025, plus 1537 and the long-context V1 shape. Compare chunked prefill, tokenwise execution, and an independent oracle under the declared numerical contract. Capture prefill logits at matched positions directly, then check decode continuation. Record block sizes and selected attention routes. | V1 scores continuation after prefill; boundary transitions, chunk-dependent behavior, and direct prefill outputs need separate coverage. | **measured -- the boundary premise is refuted; the oracle leg is still open.** New `scripts/gemma4_boundary_compare.py` (the campaign `gate` refuses arms with different `--prefill`, so a cross-geometry comparator was the real gap) aligns rows on absolute position and judges them with the gate's own `evaluate`. Determinism control passes: a repeated capture is **bit-identical**. **All seven named lengths measured.** `kl_max` and top-1 flip counts are **bit-identical across every prompt length for a given bulk width** (64 -> 1.7444816155494507 / 12 flips, 128 -> 3.5428931232712273 / 18, 256 -> 1.7446299083839871 / 5, 512 -> 0.1593 / 0, 1024 -> 0.00102 / 0), so **511/512/513 are indistinguishable from 1023/1024/1025 and 1537**. Neither the 512 sliding window nor the 1024 split threshold (which does engage from L=1024, `selections [1,2]`, `split_keys [1024, L-1]`) changes the pattern. The effect is a **pure function of bulk width**: a single incomplete block diverges from tokenwise (mean KL 8.9e-03..5.4e-02 against a 1e-03 limit, up to 18 flips), while every production-shaped width **passes with top-1 = 1.0** -- 576 (kl_mean 2.17e-05 / max 7.97e-03), 640 (8.42e-05 / 3.05e-02), 768 (6.78e-06 / 7.77e-04), 1024 (6.50e-06 / 1.02e-03). Exactly 512 misses on one row only (max 0.159 vs 0.05) with zero flips. **Not established: which arm is correct** -- both are hipEngine arms, so a shared wrong mask is invisible; that is V2's oracle, and it is the binding dependency. Row premise should be rewritten around bulk width, not prompt length. 33 arm captures under `~/.cache/hipengine/tmp/v5/`; `performance_claim: false`. |

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
| P1 | attn.sliding | **Windowed AOTriton for sliding chunks past the window.** The vendored AOTriton 0.11.2b already ships `CausalType::WindowedAttention` images (`FONLY__*bf16@16_256_*_3_0`) and takes integer `window_left` / `window_right`. `aotriton_wrap.cc` passes only the bottom-right-aligned sentinels, so `aotriton_prefill_admits` refuses every block once the 1024-token window binds, and those blocks fall back to the exact class kernel. Pass `window_left = sliding_window − 1`, `window_right = 0`, extend the admission predicate to "mask is exactly causal ∧ windowed", and extend the parity test to keys 1536 / 2048 / 4096. | 4096: sliding attention is 1478.7 ms (39.5% of the prefill). Only blocks 1–2 use `attn_fwd`; blocks 3–8 run `gemma4_attention_decode_class_kernel<…,1,2>` — **verified**: `DEFAULT_PREFILL_BLOCK` is 512, so 4096 is 8 blocks, and `launch_gemma4_attention_prefill` tries the decode family first whenever `tokens > 1`. The plumbing point is also verified: `aotriton_wrap.cc` already selects `CausalType::WindowedAttention` and passes the `BottomRightAligned` sentinel for both bounds, which is why the `_3_0` windowed image currently runs as plain causal. llama.cpp: 84 ms. Note the recorded alternative: `a4c7cdfc8`'s "Next" proposes a tiled kernel for this same gap rather than the window plumbing, so the two bets are on the record. **Attempted; blocked on the vendored binary rather than on the plumbing.** The plumbing point verified as written -- `aotriton_wrap.cc` already selects `CausalType::WindowedAttention` and the shim can carry real `window_left` / `window_right` -- but the measurement below shows the bounds are interpreted **top-left aligned** (row `i` of the *query block*), while a sliding layer's mask is **bottom-right aligned** over absolute positions, which is exactly what a suffix query block needs. Measured with the flash launch held at `window_left = window - 1, window_right = 0` and scored against exact-kernel outputs built from four candidate masks, at the discriminating shape `rows=512 keys=1536 window=128` (`keys > rows` so the two frames differ, and a window narrower than the block so it binds inside it): `top_left_window` matches at maxabs 0.0078 / meanabs 0.00009, while `bottom_right_window` misses at 2.7266 / 0.35227, `plain causal` at 2.6406 / 0.28943 and `block_local_causal` at 2.5859 / 0.13893 -- so it is genuinely a window and genuinely top-left, not causality under another name. A sweep over `window_left` in {sentinel, 0, W-2, W-1, W, W+1} x `window_right` in {sentinel, 0, -1} shows `window_left` has **no effect at all** while `window_right` is the sentinel: all five integer values give output identical to plain causal (meanabs 0.00008 against the causal reference). The two frames therefore cannot be mixed -- integer bounds are top-left, only the sentinel is bottom-right, and the sentinel carries no bound. Nor can top-left be bent into bottom-right by offsetting `window_left`, because the integer bounds place the *right* bound at block-local `i` as well, so a suffix block cuts at `i` instead of `start + i`, and `window_left = window - 1 - start` would have to go negative for every block past `start >= window`. Causal is unaffected: at every one of keys 1536 / 2048 / 3000 / 4096 the flash path matches the exact kernel to maxabs 0.0078, so the failure is the window alone. | ~1300 ms (4096: ~1060 -> ~1600 tok/s), **not achievable through AOTriton at any released version** | closed (measured: integer windows stay top-left under CompactVarlen) | Implemented end to end -- shim `window_left` / `window_right` parameters, a `sliding_window` argument on `gemma4_attention_prefill_aotriton`, the admission predicate relaxed, the layer passing `geometry.sliding_window`, and parity cases at keys 1536 / 2048 / 4096 -- then reverted uncommitted when the parity test failed all five windowed cases while all four causal cases passed. Nothing landed, so production behaviour is unchanged. **Named capability miss, now understood from the shipped header rather than from behaviour alone:** `flash.h` carries the comment *"TopLeftAligned and BottomRightAligned are supported in Triton kernel, but not compiled into the binary GPU kernels"* with `CausalType::TopLeftAligned = 1` and `BottomRightAligned = 2` **commented out** — which is exactly why both the vendored and the 0.14.2b image sets show `CAUSAL_TYPE ∈ {0,3}`. So this is a compile-time exclusion that upstream has now left in place for three feature releases, not a feature on a roadmap, and waiting for a release will not clear it. Second fact from the same header: `WindowValue::TopLeftAligned` / `BottomRightAligned` are documented as **"Special value for varlen"** — and `aotriton_wrap.cc` already exposes `hipengine_aotriton_attn_fwd_v3_compact_varlen` with `cu_seqlens_q` / `cu_seqlens_k`. That is consistent with the sweep's otherwise puzzling observation that "the sentinel carries no bound": in non-varlen mode the marker has no sequence length to derive one from. **Second revision, after checking what production already runs.** The `compact_varlen` shim is not an untried configuration — it *is* the current path: `gemma4_attention_prefill_aotriton` calls `aotriton_attn_fwd_v3_compact_varlen`, and `aotriton_wrap.cc:457-461` hardcodes `causal_type = WindowedAttention`, `varlen_type = CompactVarlen`, and **both** `window_left`/`window_right = BottomRightAligned`. The passing parity test already proves that exact configuration — varlen plus both sentinels — reproduces plain causal. So the varlen re-run above would have returned a known answer, and running it would have been a wasted GPU pass. **What is genuinely untested is an *integer* window bound under `CompactVarlen`**: P1's sweep measured integer bounds only in the non-varlen launch (where they came back top-left), and the sentinel-only case in varlen (which is causal). No integer bound is reachable at all today — `window_left` / `window_right` appear in **zero** `.py` files and the shim takes no such parameters. **Revised clearing command: parameterize `hipengine_aotriton_attn_fwd_v3_compact_varlen` with `window_left` / `window_right`, thread them through the Python argtypes, then run the four-mask comparison at `rows=512 keys=1536 window=128` with `window_left = W-1, window_right = 0` under varlen.** That is the one cell of the matrix neither the sweep nor the parity test covers. It is a C++ change plus rebuild, which is why it is a gate to run deliberately rather than an afternoon's probe — and the conclusion scope is unchanged: a bottom-right match earns restoring the plumbing after observing the flip, a top-left match closes P1 for good and leaves P2's option (a) as this gap's only route, and neither outcome is assumed here. Until then the exact class kernel is the only path for a binding window, which is what P14 just made 18% cheaper on `attn.sliding` -- so this row is a smaller loss than its estimate suggests, and `a4c7cdfc8`'s tiled-kernel alternative remains the live bet for the gap, and it resolves to a row already on this list: it is **P2's option (a)**, which a4c7cdfc8's entry also credits with removing this very fallback — **measured 1228.9 tok/s at 2048 tokens**, a number this list previously carried nowhere. One port is the exit for both rows. **Closing measurement (2026-09-28):** the last untested cell — integer `window_left = W-1` / `window_right = 0` **under `CompactVarlen`**, which nothing in Python could express until `3d084638d` — was run by `scripts/gemma4_p1_window_alignment_probe.py` at `rows=512 keys=1536 window=128`. The output matches the **top-left** candidate at maxabs 0.007812 / meanabs 0.000091 (the same BF16 noise floor as the causal baseline) and misses every other candidate: causal 2.640625, bottom-right 2.726662, block-local causal 2.585938, and a left-edge-only control at 2.640625 — that last one matters, because it isolates the **right** bound and shows it cutting a suffix block at block-local `i` instead of `start + i`. All four figures reproduce P1's original sweep to four decimals, so varlen changes nothing. **Varlen was the one configuration nobody had run, and it behaves identically: the alignment does not flip, the blocker is structural, and this row closes.** P2's option (a) is the sole route for both this fallback and the five global layers. |
| P2 | attn.global | **Tiled flash attention at head_dim 512** for the 5 global layers. **This kernel is also P1's alternative route**: a4c7cdfc8's entry credits it with removing the past-window sliding fallback as well as serving these five layers, so the port pays twice and P1's blocked plumbing route is not the only way that gap closes. Options: (a) port llama.cpp's `flash_attn_tile<512,512,4,8>`, which measured 6.7 ms at 1024; (b) an AOTriton head_dim-512 image -- **closed for AOTriton as a whole, not just vendored**: the shipped `aotriton.images/amd-gfx11xx/flash/attn_fwd` set is 12 files, all `*bf16@16_256_*`, `_AOTRITON_PREFILL_HEAD_DIMS` is `(256,)`, and X4's direct listing of the newest upstream `0.14.2b` gfx110x bundle found `BLOCK_DMODEL` topping out at 256 across all 396 images, so no release ships one; (c) an in-tree WMMA kernel. K and V share the raw projection (no `attn_v`), but not the cached tensor: K uses weighted head normalization and RoPE; V uses weightless normalization without RoPE (`gemma4_layer.py`). Load the distinct K and V tiles. GQA is 8 query heads per KV head, so pack them into the WMMA M dimension. | 39.2 ms at 1024 (8.1%), 610.4 ms at 4096. `gemma4_attention_decode_class_kernel<…,2,2>` runs 10 / 40 launches. llama.cpp: 6.7 / 85.7 ms. | **~32 ms** (+6.6%) / **~525 ms** | open | **Feasibility proven on gfx1100 (2026-09-29) -- route (a) is viable, not yet wired.** The route question is answered by building the port target rather than reading it. Extracting the device region of `~/amd-gpu-tuning/llama.cpp/ggml/src/ggml-cuda/fattn-tile.cuh` (config tables lines 7-347, device helpers + kernel lines 348-1115) with a small hand-written shim for its `common.cuh` helpers, `hipcc -std=c++17 --offload-arch=gfx1100` compiles it with **zero errors and instantiates both `flash_attn_tile<512,512,4,8,false>` and `<...true>`** -- the symbols are in the object, `__hip_fatbin_wrapper` and `hipLaunchKernel` resolve, 184,808 bytes of real device code. That proves three things the row previously only argued: (1) the RDNA config table carries `CASE(512, 512, 32, 256, 2, 128, 64)` -- ncols 32 is exactly `ncols1=4 x ncols2=8`, so the `<512,512,4,8>` shape the row quotes resolves to 256 threads / occupancy 2 on RDNA3; (2) the source is already ROCm-hardened (it carries its own comment and macro workaround for "the ROCm compiler cannot handle templating in `__launch_bounds__`"); (3) the whole dependency surface is four tiny helpers -- `ggml_cuda_get_max_cpy_bytes`, `ggml_cuda_unroll`, `ggml_cuda_mad`, `ggml_cuda_memcpy_1` -- plus `fastdiv`/`fastmodulo`, the two `warp_reduce_{sum,max}` families, `get_alibi_slope`, and macros `FATTN_KQ_MAX_OFFSET` / `NO_DEVICE_CODE` / `GGML_UNUSED_VARS`. **Two build traps are worth recording because both first appeared as misleading signals.** First, a plain extract reports zero errors while producing an 848-byte object with **no kernel in it**: `flash_attn_tile` is `static __global__` and is dead-stripped unless something takes its address, so "it compiles" proved only that it type-checks. Forcing instantiation by taking the kernel address is what surfaced the real failures. Second, the device-side `get_config` selects its table via `#ifdef GGML_USE_HIP` then `#ifdef RDNA` -- **not `RDNA3`** -- so without those two defines it falls through to `nvidia_fp16`, whose table has no ncols=32 entry and the config decodes to all zeros: that produced `__launch_bounds__(0,0)`, a failing `static_assert(ggml_cuda_fattn_tile_get_config(512,512,4*8) != 0)`, a non-constant `cpw`, and two zero-length arrays -- five distinct-looking errors from one wrong macro. Defining `GGML_USE_HIP` and `RDNA` cleared every one. A sixth, independent error was ROCm 7.2's rule that `__shfl_xor_sync`'s mask must be 64-bit; those call sites are all in the shim and none in the kernel body, fixed with `0xffffffffffffffffULL`. **Still owed:** the host wrapper mapping Gemma's `(tokens, num_heads, head_dim)` / `(keys, num_kv_heads, head_dim)` / `(tokens, keys)` uint8 keep-mask onto ggml's `ne*/nb*` descriptors, the `init_fastdiv_values` host call the `fastmodulo` sites need, registration on the `attn` axis with a strict unfused fallback to the exact block kernel, parity against `tests/test_gpu_gemma4_attention_geometry.py` (which already parametrises head_dim 512 over keys 3/8192/15616, so the oracle exists), and the 39.2 -> ~32 ms measurement. |
| P3 | moe.l29_experts | **Give layer 29's experts a fast owner.** Its Q5_K gate_up runs `gguf_q4_k_selected_dual_grouped_rowbatch` (21.9 ms / 1024) and its Q8_0 down runs `gguf_k_selected_prefill_out_kernel<…,8>` (44.4 ms / 1024). Step 1: determine whether this is routing or a missing leaf. `gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out` guards only on `in_features % 32`, and 704 passes. Step 2: add a Q5_K tile loader to the int8 MMQ32 gate_up leaf. | One layer costs 13.8% of prefill. llama.cpp runs it through its ordinary MMQ in 2.6 ms. | **~60 ms** (+13%) / ~258 ms | **Step 1 answered and the down half fixed (2026-09-29).** The row splits: the down was **routing**, the gate_up is a genuine **missing leaf**. **Down (Q8_0), routing -- fixed.** `gemma4_project_experts_wmma` resolves `KernelKey(backend, "moe_linear", quant_key, "selected_grouped_wmma_prefill_compact_bf16_bf16_out")`. Measured before the fix: `moe_linear` + `gguf_q5_1` -> `qwen4_exp_q5_1_selected_grouped_wmma_prefill_compact_bf16_bf16_out` (which is why every other layer's down is fast), while `moe_linear` + `gguf_q8_0` -> `MissingKernelError`. The Q8_0 owner `gguf_q8_0_selected_grouped_wmma_prefill_compact_bf16_bf16_out` already carries the expert ABI -- it takes `expert_start_compact_ptr` / `expert_start_wmma_ptr` / `tile_expert_ptr` and documents itself as the promoted Q5_1 grouped WMMA down contract applied to Q8_0 weights -- is pinned against a NumPy dequant reference through `qwen35_moe_wmma_tile_map` in `tests/test_gpu_qwen4exp_q8_0_grouped_wmma_down.py`, and was bound only on the dense axis (`linear`) under the dense name `selected_grouped_wmma_prefill_bf16_bf16_out`. So the down fell through WMMA -> MMQ -> grouped to the selected GEMV and ran 44.4 ms. **Fix:** one `register()` in `register_gguf_q8_0_prefill_kernels()` binding `(moe_linear, gguf_q8_0, selected_grouped_wmma_prefill_compact_bf16_bf16_out)` to that owner -- plain form only, because `auto` runs `compensated=False` and no compensated Q8_0 wrapper exists. Resolve now returns the owner; the kernel's own reference test, `test_unit_gemma4_gpu_kernels.py`, and the gemma4 unit bundle plus `test_unit_kernel_registry.py` (636 tests) all pass. **Gate_up (Q5_K), missing leaf -- Step 2 stands.** `moe_linear` + `gguf_q5_k` resolves to nothing for either WMMA variant, and `_MMQ_DUAL_QUANT_KEY` is pinned to `gguf_q4_k`, so the fused-gate_up int8 MMQ leaf structurally cannot serve Q5_K. That is the row's Step 2: a Q5_K tile loader in that leaf. **Re-measured 2026-09-29 (perf) and gated same day -- both were owed, both are below.** The perf evidence was owed as "the 21.9 / 44.4 ms re-measurement showing the fast leaf now runs at 1024"; the gate was owed as arithmetic change on the default path, with the attribution constraint that HEAD's gate is already red at `kl_max` 0.2177 from the router commit so the two arithmetic changes must not be stacked. **Perf re-measurement (HIP-event timing, `scripts/gemma4_p3_down_ab.py`, `--repeat 34` = 1021 prefill tokens in 2 blocks of 512/509, 30630 token-layers, four paired runs).** The control reproduces pre-fix dispatch by `unregister()`-ing the `(moe_linear, gguf_q8_0, selected_grouped_wmma_prefill_compact_bf16_bf16_out)` key and blocking lazy re-registration for that one key, so the real dispatcher's internal `resolve()` misses and returns `False` exactly as it did before, while every Q5_1 layer keeps its owner in both arms -- the delta is layer 29 alone. **Expert-FFN total: 455.163 / 456.382 / 457.575 ms (fixed) vs 491.029 / 491.365 / 494.112 ms (pre-fix), a delta of 35.866 / 34.983 / 36.537 ms -- mean 35.8 ms, spread 0.8 ms, i.e. -7.3% of routed-expert work at prompt 1024.** Layer 29's own down measures **1.43 ms** on the fixed side; adding the delta puts the pre-fix figure near 37 ms, consistent with the 44.4 ms this row recorded from a kernel-level breakdown (different method, same order, so it corroborates rather than equals). The population view is the method-independent part: layer 29's down now runs at **0.803-0.814x** the average of the already-fast Q5_1 layers, where before it was the lone outlier that this entire row exists to describe. Attempting the control by raising `MissingKernelError` from outside the dispatcher failed and is worth recording: that function wraps `resolve()` in `try/except MissingKernelError: return False`, so an external raise escapes to the engine and is fatal during the decode tick, whereas the real missing-registration path never propagates an exception at all. **Separately:** `test_unit_gemma4_generate.py` and `test_unit_gemma4_runner.py` hang past 600 s in this worktree -- reproduced against a pristine copy of the edited file, so pre-existing and unrelated to this change; it needs its own row. **Gate run 2026-09-29 (W7900, `7a91b5735`): the gate is structurally blind to this change, and the fix is confirmed live by a different measurement.** Running `gate --baseline teacher-forced-prefilled-p4096-c3ea386af.npz --prompt 5120 --prefill 4096 --context 8192` gives `kl_max` **0.2177167513641361** -- bit-identical to `2026-09-29-gemma4-p4096-attrib-d0f5b1ff6-fail-w7900.json`, with `kl_worst_rows` equal to the last digit (rows 894 / 327 / 254), `kl_mean` 3.707e-4, `kl_p95` 5.279e-6, `kl_p99` 1.123e-4, `top1_rate` 1.0, 0 flips, 1023 rows. Identical output is **not** evidence the fix does nothing -- it is that this gate cannot see it, for two independent reasons. (1) The WMMA plan gate is `lanes >= 16 * num_experts` = `512*8=4096 >= 2048`, so a **512-row prefill block takes WMMA** while a **scored decode row is `lanes = 1*8 = 8`** and falls back to exactly what HEAD runs; the gate's scored range is 4097-5119, all decode, and prefill is cache-only and never scored. (2) Layer 29 is terminal -- its down output feeds only the final norm and lm_head, so no amount of prefill-side change to it can reach the KV cache the scored rows read. **The fix is live, measured directly:** with a production-length prompt the probe records `block rows [1, 225, 512]`, `lanes [8, 1800, 4096]`, threshold 2048, and **`plan calls 30 | wmma calls 30`** -- one per MoE layer on the qualifying block, layer 29 included, against **0** before the registration existed. At a short prompt (153-row block, lanes 1224) it correctly declines, so the gate condition behaves on both sides. Correctness for the newly-exposed path rests on the kernel's own NumPy-dequant reference test (4 passed), and the gemma4 unit bundle plus registry is 636 green. **Consequence for the campaign:** the teacher-forced gate cannot qualify or reject MoE prefill-route changes that only engage at `lanes >= 2048`, because it scores decode only -- that is a gap in the gate itself and belongs in the row rather than being papered over with a pass. | |
| P4 | moe.gate_up | **Larger int8 MMQ tiles for the Q4_K gate_up.** **Refuted 2026-09-28 -- do not reopen without a mechanism, not a comparison.** The roofline half of the premise is *right*: 1.884 TFLOP in 114.1 ms is 16.51 TFLOP/s, 22% of the 75.3 TOPS measured INT8 peak, while weights stream at only 9% of the 864 GB/s memory peak and arithmetic intensity is 224.6 FLOP/byte against a 70.9 ridge point -- so gate_up is compute-bound, not bandwidth-bound. **But being far from peak is not the same as the gap being shape-shaped**, and the campaign measured that directly four times: wider out-block at two accumulator footprints (iteration 101, both regressions), a wider expert grid (102, neutral), the input re-read fix (104, ~4%, reverted), and rows-per-expert reuse (126, refuted). This row's own `I = 64` candidate **is** iteration 101's wider out-block. "Nothing that changes how the work is shaped moves these kernels, and nothing that changes how much traffic they move does either." What remains is the inner loop's instruction mix and latency structure, or a different compute path: the WMMA arm, measured at **~12%** (iteration 102) and held off the default path by the arithmetic gate. | 2.3x behind llama.cpp at both lengths; hipEngine ~16 TFLOP/s against llama.cpp ~33. | ~63 ms / ~260 ms (hypothesis, now refuted) | refuted | |
| P5 | dense.q8_0 | **Single-plane int8 MMQ for the dense Q8_0 projections.** **Refuted 2026-09-28 -- do not reopen on a TFLOP/s comparison.** The row's "structurally different from the three-plane `d4x3`" claim is true and irrelevant: iteration 150 did not refute the guard planes, it refuted *this* path by making it actually run. The int8 MMQ dispatch was configured but unreachable -- `_wmma_prefill_dispatch` rewrote the key upstream, so `_q8_mmq_prefill_dispatch` fell through its whitelist on `abi='wmma_raw'` rather than on policy. Moving it ahead made it fire, and firing it was **29% slower** (1373/1367/1367 -> 1054/1056/1053 tok/s) while the gate passed and `kl_max` *improved* (0.000893 against 0.001341). The int8 kernel is correct and slightly more accurate and simply loses to bf16 WMMA on these shapes. Iteration 155 independently puts the penalty at **+0.25 s per prefill**. Iteration 150's own verdict: "the only thing wrong with the dead path was that it looked dead", and the ordering that keeps it unreachable is load-bearing. | 115.6 ms against 59.0 ms (2.0x) -- **this is the refuted estimate class**: it compares an MMQ TFLOP/s against a bf16 WMMA rate, which is precisely the reasoning iteration 150 measured out of existence. | up to ~57 ms / ~228 ms (hypothesis, refuted) | refuted | |
| P6 | dense.q8_0 | **Concatenate q/k/v, and the shared-MLP gate/up, into one weight each at load.** This must be a replacement layout, not an added copy. **Launch recount 2026-09-28:** the census (`/tmp/gemma4-prefill-census-dual.json`) gives `dense:gguf_q8_0` = **410 calls over 60 layers = 6.83 per layer**, so the row's 412 is right. But the two fusions described here save only **3 launches per layer** (q/k/v 3→1, gate/up 2→1) = **180 saved, landing at ~230, not the ~180 this row claimed** -- reaching 180 needs a third fusion, and `ffn_down` (different input dim) and `attn_output` (reads attention output, not the input) both cannot join. The saving is launch overhead plus fuller tiles, not less work: each dense call still runs its full arithmetic. | 412 dense launches per 1024-token prefill (**recounted: 410**). | 5–15 ms (unmeasured; landing figure corrected 412 -> ~230) | open | |
| P7 | moe.route_glue | **Replace the MoE scheduler.** **Won 2026-09-28.** The count / prefix / scatter scheduler this row asked for already existed as `qwen35_moe_group_compact_active_parallel`, was exported, and had no production caller -- the single call site took the serial default. Enabling it moved the family 28.3 -> 7.3 ms / 1024 and 112.9 -> 29.1 / 4096; dropping the caller's now-redundant count + prefix launches, since the parallel launcher issues all three itself, finished at **6.8 / 27.0 ms** against this row's pre-fix 22.2 + 6.4 and 88.2 + 25.4. Outputs are byte-identical at every lane count tested, sentinel coverage is clean, and the gate verdict did not move. Prefill and decode both landed above baseline: 2106.0 / 1188.2 tok/s (+3.19% / +1.99%) and 44.26 / 40.60 (+1.05% / +1.07%). | llama.cpp's whole routing family is 10.5 ms / 1024. | **6.8 / 27.0 ms** against ~24 / ~96 est. | won | |
| P8 | moe.router_gemm | **Router logits through WMMA or hipBLASLt, with the prescale fused in.** **Variant selection shipped 2026-09-28 and is measured**: the route called the generic `qwen35_router_logits_bf16_f32w`, which defaults to `threads=512` with a four-token tile -- at hidden 2816 that leaves threads 352..511 with no K range at all, so 31% of every block idles behind a nine-round barrier tree for 64 FLOPs per useful thread. Pointing prefill at the already-built `token_tile_16` and setting the width to 128 (not the binding's 256) takes 0.2309 ms -> 0.0666 ms per launch (1.60 -> 5.43 TFLOP/s, 3.5x) for a drift under 4e-06 against a float64 reference, for **-6.2 ms / 1024 and -24.8 ms / 4096** of `router_gemm` at width 256 and more at 128; `moe.router_gemm` is now 7.2 ms / 1024 and 28.4 ms / 4096 at 256. Width is the larger lever and is non-monotone: 64 gives 0.0927 ms, **128 gives 0.0666**, 256 gives 0.0943, 512 gives 0.1546. An expert x token retile (one block owning four experts over sixteen tokens) was built and measured and **lost at every width** -- 0.1070 / 0.1333 / 0.2091 ms against the token tile's 0.0927 / 0.0680 / 0.0993 -- so the direction is closed and the code was not kept. Decode keeps the untiled path below 32 tokens, where it is faster -- the crossover sits between 16 and 20 tokens. **The prescale fold into the A load was built, measured, and rejected (2026-09-28).** Prescale is 0.0198 ms of the 0.0913 ms pair -- 22%, 4.6 ms per 4096-prefill, exactly the gap to llama.cpp's implied 16 ms -- so the ceiling looked real. It does not capture: the halves factor cleanly (`scale[k]` per-hidden-dim folds into the A load, `inv_rms * root_size` per-row multiplies the finished dot product, so one pass over K writes the same expression), and the fused kernel still **loses at every width** -- 0.1223 / 0.1483 / 0.2375 ms at 64 / 128 / 256 threads against the pair's 0.0913, best case 0.75x. Hoisting `scale` out of the inner loop, the obvious suspect, moved it only 0.1546 -> 0.1483. The row-sumsq accumulation and the second reduction array cost the GEMM more than the separate norm costs to run. Worth noting for whoever retries: the fused route is **~10,000x more accurate** against a float64 reference (4.0e-08 against 4.06e-04) because the intermediate bf16 rounding of the prescaled buffer disappears -- if that accuracy ever matters more than the 34%, the code is recoverable from the gate artifact's history. Still open, but **re-scoped by measurement**: the top-8 chain the punchlist proposed fusing (`qwen35_router_select` + `gemma4_expert_weight_scale`) is only **6.9% of `moe.route_glue`** across a full prefill+decode trace -- 5.766 and 0.994 ms over 450 launches each, 12.8 and 2.2 us apiece -- so even zeroing the whole chain moves about 2 ms of a 469 ms prefill. llama.cpp's 0.22 ms `topk_moe_cuda` was a comparison figure, not a profile of our chain, and building it would not have paid. The family's real target is **`qwen35_moe_group_compact_active_kernel`: 68.905 ms over 450 launches, 153.1 us each, 71% of route_glue**, with `gemma4_moe_weighted_accumulate_kernel` second at 14.152 ms / 31.4 us. | 7.2 ms against 4.0 ms. | ~3 ms / ~13 ms | open | Measurement `benchmarks/results/2026-09-28-gemma4-scoreboard-p8b.json`; gate `benchmarks/results/2026-09-28-gemma4-26b-a4b-router-token-tile-gate.json` (mean 7.06e-06, max 1.30e-03, top-1 1.0); sweep `scripts/gemma4_router_logits_variant_bench.py`. |
| P9 | moe.act_quant | **Fuse the Q8_1 activation pack into `pre_ffw_norm_2`.** The norm already reads the row. | `gguf_q8_1_mmq_ds4_pack_bf16` costs 5.7 / 22.8 ms. | ~4 ms / ~17 ms | **landed (re-scoped)** | The stated mechanism does not exist: a top-k gather sits between the norm and the pack. `rmsnorm` writes `normalized` at `tokens` rows, `qwen35_moe_gather_packed_hidden_lowp` turns that into `packed_hidden` at `lanes = tokens * top_k` rows, and the pack reads *that* -- norm rows and pack rows are not the same rows, so the pack cannot fold into the norm. The same saving was reached from the other end: **the pack now gathers and packs in one kernel**, and the BF16 `packed_hidden` staging buffer is skipped entirely on the MMQ route. That route is legal to skip because the gate_up projection is a single if/elif ladder, so on this artifact `packed_hidden`'s only reader was the pack (`_mmq_dual_route` extracts the projection's four pure guards so the caller can decide before the gather and cannot disagree with it). New `gguf_q8_1_mmq_gather_ds4_pack_bf16` quantizes straight out of the source rows through the same lane index; since the gather is a bit copy the output is **byte-identical**, pinned by `test_q8_1_mmq_gather_ds4_pack_is_byte_exact_to_gather_then_pack` (RED then GREEN, 3 shapes). **Measured on the XTX primary lane** (`--gpu 1`, tag `p9`, artifact `benchmarks/results/2026-09-28-gemma4-scoreboard-p9.json`), A/B against tag `p12`: at 4096 `moe.route_glue` 137.3 -> **112.6 ms (-24.7)** while `moe.act_quant` 22.8 -> 25.1 (+2.3, it now carries both), **net -22.4 ms** which is exactly the busy change 3466.8 -> **3444.3 ms (-0.65%)**; at 1024 route_glue -6.3 and act_quant +0.5, busy 481.1 -> 475.8 (-5.4 ms). Topline prefill 1149.9 -> **1156.7 tok/s (+0.6%)** at 4096 and 1989.3 -> **2020.1 (+1.6%)** at 1024; decode unchanged at 43.80 / 40.06, as it must be since a one-token decode block takes the same route. Correctness: teacher-forced gate `passed: true`, 1023 rows, top-1 1.0, 0 flips, **KL identical to the recorded run to every digit** (mean 5.519899e-06). The family attribution shift is itself the proof the fused route runs: the gather left `route_glue` and its work appeared under `act_quant`. |
| P10 | moe.down | **A llama.cpp-shaped Q5_1 MMQ for the down projection** (I = 64, J = 64, K = 704 as 22 blocks of 32). llama.cpp's `mul_mat_q<Q5_1, 64>` reaches about 31 TFLOP/s on this shape. hipEngine's grouped BF16 WMMA owner reaches about 20. The int8 leaf that iteration 142 refuted (4 TFLOP/s) is a different, 32-row design. | 46.8 against 30.2 ms. | ~17 ms / ~71 ms | open | |
| P11 | whole layer | **Run the shared-expert MLP branch and the MoE branch on two streams.** llama.cpp forks at the branch point (its `concurrent_events`), and its 187 ms of busy time fits in a 140 ms span. hipEngine is single-stream. This needs stream / event plumbing in the Gemma layer loop. | llama.cpp overlap ≈47 ms / 1024. | 20–40 ms (estimate) | open | |
| P12 | lm_head | **Run the lm_head only on the final prefill block.** It runs once per 512-token block today (2 launches at 1024, 8 at 4096, about 1 ms each), but only the final block's last row is used. | `gguf_k_pack8_prefill_out_kernel<…,float>` × blocks. | ~1 ms / ~6.7 ms | **landed** | **Landed and measured on the XTX primary lane** (`--gpu 1`, tag `p12`, artifact `benchmarks/results/2026-09-28-gemma4-scoreboard-p12.json`), A/B against tag `p14fix` on one protocol with no intervening change: `lm_head` **7.8 -> 1.0 ms at 4096 (-6.9 ms)** and **2.0 -> 1.0 ms at 1024 (-1.0 ms)**, which is the estimate above -- the census counts exactly the launches the loop drops, 7 of 8 blocks at 4096 and 1 of 2 at 1024. Every other family is flat within run-to-run noise (largest is `moe.gate_up` at +1.1 ms) and decode is unchanged at 43.88 / 40.11 tok/s, as expected because a decode block holds one token, so its head already ran only on that token. Total prefill device busy 3474.4 -> 3466.8 ms at 4096 (-7.6 ms); topline prefill 1145.8 -> 1149.9 tok/s (+0.4%) and 1982.7 -> 1989.3 at 1024, both inside topline noise, so **the family census carries this result rather than the topline**. Correctness: the teacher-forced gate `passed: true` over 1023 rows with KL identical to the recorded run to every digit (mean 5.519899e-06, p95 1.536121e-05, p99 1.435788e-04, max 7.734222e-04), top-1 1.0 and 0 flips -- the head now runs only on the block whose logits survive, and the returned distribution is computed on exactly the inputs it was before. Pinned RED-then-GREEN by `test_multi_block_forward_takes_the_head_only_on_the_final_block`, which counts the norm and projection calls across a two-block forward. |
| P13 | whole prefill | **Re-test the 1024-token prefill block after P4 / P10.** Block 1024 was a wash with the current compute-bound MoE kernels (`0b2ce993a`). Larger MMQ tiles make rows-per-expert matter again. | llama.cpp uses a 1024-token ubatch. | unknown | blocked (P4) | |
| P14 | attn.sliding | **Exploit the mask's lower bound in pass 3 of the multi-row exact kernel.** Pass 3 already derives `last_active` from the mask row (a byte walk plus a warp max-reduce) and walks `[0, last_active + 1)`. The *first* kept key is never computed, so a sliding row still starts at key 0 and walks the whole causal triangle. Add the mirror reduction (`first_active`, one more comparison in a loop that already runs, min-reduced the same way) and start pass 3 there; bound pass 2's range likewise. **Bit-exact by the argument the existing tail trim already relies on**: a masked key contributes weight 0 to every dimension, so dropping it leaves the ascending-key accumulation unchanged -- the same argument `key_begin` makes for one-row blocks. No AOTriton dependency, and it applies to any mask with a lower bound, so an eviction policy gets it too. **P1 and P14 compete for the same milliseconds.** Once P1 extends admission to causal-and-windowed masks, every multi-row sliding block at these shapes goes to AOTriton. P14's remaining scope is the class kernel's other callers: non-causal masks (eviction policies), head dims with no AOTriton image, and hosts where the AOTriton runtime is unavailable. Pick one to land first and re-measure before starting the other. | Pass 3 is the dominant pass and is memory-latency bound -- the kernel's own comment records 34 GB/s unique against 493 GB/s with the loads removed. **Measured: the per-block cost of the causal mask tracks the pass-3 walk `sum(q+1)` to within about 5% over the first four blocks** (1.94 / 5.81 / 9.65 / 13.50 ms against ratios 1 / 3 / 5 / 7), which is direct evidence that pass 3 sets the cost. Today it walks `[0, q]` on sliding *and* global rows alike -- the window's lower bound reaches pass 1 alone, and that is worth +29.9% over the whole prefill. Cutting the sliding walk to its window drops the keys walked from 8.39M to 3.67M per layer at 4096 over all rows, a **2.29x** reduction on that pass, on top of the window saving already banked. Over the rows the class kernel actually serves (blocks 3–8, rows 1024–4095, since blocks 1–2 run on `attn_fwd`), the reduction is 7.87M to 3.15M, **2.50x**. Global layers gain nothing (a causal row's first kept key is already 0). **Landed and measured on the XTX primary lane**, three scoreboard snapshots on one protocol (`--gpu 1`, tags `baseline` / `p14` / `p14fix`, artifacts `benchmarks/results/2026-09-28-gemma4-scoreboard-{baseline,p14,p14fix}.json`): `attn.sliding` 1479.5 -> 1211.3 ms (**-18.1%**), `attn.global` 611.3 -> 612.2 ms (+0.1%, i.e. restored to parity), total prefill device busy 3751.8 -> 3474.4 ms (**-7.4%**), topline prefill @4096 1061.6 -> 1145.8 tok/s (**+7.9%**), decode @4096 unchanged at 40.2 tok/s. Only two commits separate the baseline tag from the candidate tag (`8307c9068`, `c41aa2217`), and neither touches production attention code, so the comparison is a clean A/B. | **-18.1% on `attn.sliding`; +7.9% prefill @4096** | landed | The first attempt won on sliding and still lost overall: `attn.global` regressed **+41.1%** (611.3 -> 862.3 ms) while sliding improved -17.8%, reproduced twice on an idle GPU. The cause was ruled out by isolation rather than by reading the code. A variant carrying only a runtime `keys_begin` -- no `first_active` reduction, no second shuffle, no `partial` write, no extra barrier -- reproduced the full slowdown (causal hd256 sum 127.95 ms against baseline 98.45), which clears the bookkeeping and the barrier; a variant carrying only the extra `__syncthreads()` measured baseline (98.31), and a variant carrying only the `keys_begin` parameter with a literal `0` measured baseline (98.32), which clears the signature change too. The compiled kernel was identical in `.amdhsa_next_free_vgpr` (64 / 53), `.amdhsa_private_segment_fixed_size` (0, so no spills), `.amdhsa_group_segment_fixed_size`, `.amdhsa_kernarg_size` and launch grid, with **fewer** instructions (1876 vs 2008 at hd512), so register pressure and occupancy are excluded. What remains is the runtime lower bound itself: on the causal sums it costs **+49.9%** (126.37 -> 189.46 ms at hd512) with nothing else changed, which sits on pass 3's dependent chain -- the loop the kernel's own comment measures at 34 GB/s unique. **Fix: call pass 3 with a literal `0` on the path that uses no trim**, so the bound is a compile-time constant and force-inlining reproduces the original loop, and keep the runtime bound only for windowed rows, where the span they cut outweighs the penalty. `first_active` is block-wide, so the branch never diverges. Same-GPU isolated probe, sum over all 8 blocks at keys=4096: causal hd256 98.45 -> 98.15 ms, causal hd512 126.37 -> 125.96 ms (both back to baseline), windowed hd256 68.56 -> **60.33 ms (-12.0%)**. Pass 2's range was left unbounded on purpose: its own comment records that skipping it entirely saves no time. Correctness: 166 GPU attention tests, 0 failures, including the `window` mask mode added in `c41aa2217` which gives both branch paths a non-zero lower bound; teacher-forced gate `passed: true` with kl 0.0 across mean / p95 / p99 / max, top-1 1.0 and 0 flips over 1023 rows (`benchmarks/results/2026-09-29-gemma4-p14-lowerbound-gate-w7900.json`). |

If P1–P3 and P7–P9 hit their estimates, a 1024-token prefill falls from 518 ms
to about 385 ms (~2650 tok/s), and a 4096-token prefill from 3.87 s to about
1.7 s (~2400 tok/s). P4, P5, P10 and P11 are what close the rest of the gap to
llama.cpp.

### Decode

| ID | Family | Candidate | Evidence | Est. gain (ms / token, @1024 / @4096) | Status | Result |
| --- | --- | --- | --- | --- | --- | --- |
| D1 | moe.gate_up | **A real MoE GEMV for Q4_K.** Today decode runs `gguf_q4_k_selected_prefill_out_kernel`: 1408 × 8 workgroups of 128 threads, 16 VGPRs, 187 µs per launch, about 95 GB/s. Port llama.cpp's `mul_mat_vec_q<Q4_K>` with expert ids: quantize the activation to Q8_1 once per layer, sudot4 dot products, several rows per wave, and all 8 experts in one launch. **This is not a prefill owner on the decode path.** The registered decode GEMV's host entry, `hipengine_gguf_q4_k_selected_gemv_bf16_bf16_out` (`launch_gguf_q4_k_selected_gemv_out` in `gguf_q4_k_gemv.hip`), launches the device kernel named `gguf_q4_k_selected_prefill_out_kernel` on an `out_features × rows` grid, which is exactly the 1408 × 8 shape in the trace. The trace name is misleading, not the route. **The per-layer gap is real, though:** both MoE projections now cost 0.284 ms per layer (8.25 ms / 29, about 104 GB/s combined), against the 0.207 ms per layer (145 GB/s) that `GEMMA4-26B-A4B-OPTIMIZATION.md` cites from iteration 19's profile. 104 GB/s is exactly what iteration 31's microbench measured for the *selected* (3-launch) form, and iteration 19 predates it on the per-expert (16-launch) form. Hypothesis to test before the port: the per-expert structure beat the selected one on bandwidth. Re-measure both structures with the same trace method on the XTX; that determines whether some of D1's gain is available without a new kernel. | 5.42 ms against llama.cpp's 0.71. Floor 0.63. | **~4.7 / ~4.1** (decode 43 → ~53 tok/s alone) | open | |
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
| X3 | lm_head | **Cut the lm_head bytes.** It is the tied Q8_0 `token_embd`, 262144 x 2816: 784 MB per token, 0.96 ms of the ≈4.6 ms memory floor -- about 21% of the floor for one projection. A lower-precision or clustered head changes output arithmetic and needs the gate. | 1.15 ms against a 0.96 ms floor. D10 fuses the argmax; this reduces what has to be read at all. | ~0.2-0.9 | open | |
| X4 | attention | **Re-evaluate the vendored AOTriton release.** P1 and P2 are both written against 0.11.2b. A newer release may ship a head_dim-512 image (P2's option b) or better windowed coverage. **Both halves now run, sha256-verified, and both come back negative — a result worth having because it closes two routes that looked promising.** The 0.14.2b `gfx110x` bundle (306,999,903 bytes, digest `a54fecf9b4e1…` matching the published one) holds 396 `attn_fwd` images with `BLOCK_DMODEL` ∈ {16,32,48,64,80,96,128,160,192,224,256} — **max 256, no 512** — and `CAUSAL_TYPE` ∈ {0,3}. The vendored 0.11.2b ships 12 images spanning `@16_256` with the **same** `CAUSAL_TYPE` ∈ {0,3}. So: (1) **P2's option (b) is closed upstream-wide, not merely vendored** — no gfx110x image at head_dim 512 exists at any released version, which retires the "upgrade AOTriton" idea instead of leaving it as an untaken ticket; (2) **P1 gains nothing from upgrading** — the causal-type domain is identical across versions, and window alignment is host-side (`left_window = window - 1`), a property images cannot express, so this inventory could never have settled it. What 0.14 does add for gfx110x is fp16/fp32 dtypes and a per-head-dim split — neither touches either gate. **What remains for P1 is reading 0.14's host source or running P1's own four-mask comparison; what remains for head_dim 512 is P2's option (a) alone.** | P2 records no head_dim-512 image in the vendored release. Checked newer releases' image inventories directly: none ships one for gfx110x either, and the causal-type domain is unchanged, so neither P1 nor P2(b) moves. | negative result (closes both routes) | closed | |
| X5 | prefill | **Prefill graph capture.** Measure host launch, transfer, and synchronization costs separately from kernel time, then test capture for the production chunk shapes. Coordinate stable device buffers and position/mask updates with X7. | Kernel-busy fractions do not measure hardware occupancy or prove a recoverable host gap. Validate capture/replay against uncaptured execution and report end-to-end time to first token. | unknown until host/timeline attribution | open | |
| X6 | scope | **Batching is untested and out of scope for this page.** Everything here is single-request. llama.cpp's server batches, hipEngine has `hipengine/dispatch/batch.py` and `hipengine/generation/batch_scheduler.py`, and the `KVLiveSpans` ABI is batch-shaped. A serving comparison is a different measurement with a different ranking. | Recorded so the single-request numbers are not read as serving numbers. | — | note | |
| X7 | runtime / sampling | **Keep the generation path device-resident where practical.** Keep logits, softcap, token selection, and reusable position/mask data on device; transfer only the results the caller needs. Preserve an explicit host-logits path for diagnostics and numerical gates. Integrate D10's greedy epilogue and prepare stable buffers for D8/X5 capture. | `Gemma4Runner._forward_block` copies vocabulary logits to the host and applies NumPy softcap; `next_token` selects on the host. Attribute those costs directly. Check softcap rounding/ties, supported sampling semantics, route selection, and request isolation. **Measured 2026-09-29** (`scripts/gemma4_host_logits_cost.py`, 200 repeats, vocab 262144 f32 = 1 MiB, real device buffer): D2H logits copy **114.7 us** mean / 119.9 p95, `np.tanh(x/c)*c` softcap **528.2 us** mean / 657.8 p95, greedy `next_token` argmax **14.4 us**. Host total **655 us = 0.655 ms per decode step**. The surprise is that **the transfer is not the cost** -- it is 0.45% of a 24.88 ms decode step while NumPy softcap + argmax are 2.18%, because `np.tanh` over 262144 elements dominates at 81% of the host total. Moving softcap alone to the device is therefore worth roughly 8x what moving the copy is. The softcap also allocates a fresh array every step. `next_token`'s argmax is negligible at 14 us and is not worth moving on its own. | **0.655 ms/token recoverable (2.63% of decode); softcap alone 0.528 ms (2.12%)** | measured 2026-09-29; device-resident move **not started** -- overlaps D8/D10 | |
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
