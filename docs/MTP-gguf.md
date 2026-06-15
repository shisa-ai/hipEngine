# MTP-GGUF Plan

Last updated: 2026-06-15
Branch: `mtp-gguf`

This document is the working plan for making hipEngine's **GGUF** inference path
use the same model-side MTP/NextN approach that llama.cpp uses for
Qwen3.6-35B-A3B, then optimizing that path until its acceptance and speed are
close to llama.cpp's draft-MTP rows.

The short version:

- The current PARO+MTP-BF16 sidecar path is useful infrastructure, but it is not
  a clean 1:1 comparison with llama.cpp. It mixes a PARO-packed target, a copied
  BF16 MTP sidecar, and exact fallback flags.
- The MTP-bearing GGUF file already carries target and NextN tensors in one
  artifact. That is the right parity target for acceptance-quality debugging.
- hipEngine already detects and **ignores** trailing GGUF `nextn` blocks for AR.
  This branch turns that ignored block into a target-attached MTP draft context.
- First objective is acceptance parity, not kernel heroics: run the same GGUF,
  same prompts, same budgets, and same acceptance denominators as llama.cpp.

## Goal

Build a native hipEngine GGUF MTP path for:

```text
/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
source: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
```

and compare against llama.cpp HIP/Vulkan `--spec-type draft-mtp` using the shared
D32 prompt suite.

Primary success criteria:

1. hipEngine GGUF AR remains correct and non-regressed on the same GGUF.
2. hipEngine can load and execute the GGUF `nextn`/MTP block without a sidecar.
3. hipEngine MTP accepted/output at B1-B4 is close to llama.cpp on matched token
   prompts before performance tuning.
4. After acceptance parity, optimize cycle cost enough to make GGUF MTP beat
   hipEngine GGUF AR on gfx1151 and W7900.

Non-goals for this branch:

- Do not optimize the PARO sidecar path first. It can borrow infrastructure later,
  but it is not the parity lane.
- Do not add `import torch` to `LLM.generate()` or any GGUF/MTP hot path.
- Do not fork dispatcher/model logic with `if backend == ...` / `if quant == ...`;
  new kernels and layouts must register through the existing plugin/registry
  model.
- Do not edit local llama.cpp checkouts. They are read-only references and
  benchmark baselines.

## Starting Evidence

### Current gfx1151 Diagnostic Matrix

Artifact:
[`2026-06-15-gfx1151-mtp-diagnostics-20260615-081020-summary.json`](../benchmarks/results/2026-06-15-gfx1151-mtp-diagnostics-20260615-081020-summary.json)

Hardware/runtime:

- AMD Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`.
- TheRock HIP `7.13.60980-c76140fa27`.
- llama.cpp reference commit:
  `6e9007ae61f4e994c27484759caac6ef2aa32b30`.
- GGUF tensor check: `753` tensors with trailing `blk.40.nextn.*` / MTP tensors.

Measured D32 rows:

| Engine / mode | Exact | tok/s | Speedup | accept/draft | accepted/output |
| --- | ---: | ---: | ---: | ---: | ---: |
| hipEngine PARO+MTP B1 `decode_batched` | 9/9 | 59.72 | 0.916x vs AR | 0.544 | 0.344 |
| hipEngine PARO+MTP B3 `decode_batched` | 9/9 | 47.64 | 0.730x vs AR | 0.298 | 0.455 |
| hipEngine PARO+MTP B1 `c1_loop` | 9/9 | 57.59 | 0.883x vs AR | 0.544 | 0.344 |
| llama.cpp HIP B4 | n/a | 92.57 | 1.801x vs llama HIP base | 0.915 | 0.743 |
| llama.cpp Vulkan B4 | n/a | 108.45 | 1.726x vs llama Vulkan base | 0.923 | 0.747 |

Key blocker from the PARO sidecar lane:

- B2 exact probe failed with `exact_ar_mismatch` on `explain_concept`.
- Every B1 fallback ablation failed on `explain_concept`; the public
  PARO-packed+sidecar artifact needs GDN exact, linear-out exact, and full-attn
  exact suffix fallbacks for B1.

Interpretation:

- llama.cpp is not only faster; it is drafting from a better-aligned model path.
  Its B1 accepted/output already exceeds hipEngine B1, and B4 reaches about 2x
  hipEngine B1 density.
- hipEngine B3 raises density but loses more to cycle cost. This points to both
  model/acceptance mismatch and verifier/runtime economics.
- GGUF MTP parity is the clean way to separate model identity from runtime cost.

### Current hipEngine GGUF State

See [`GGUF.md`](GGUF.md) and [`GGUF_DECODE_REPACK.md`](GGUF_DECODE_REPACK.md).
Relevant current state:

- hipEngine can scan GGUF v2/v3, map qwen35/qwen35moe tensor names, run public
  GGUF generation smokes, tokenize from GGUF metadata, and benchmark resident
  GGUF prefill/decode.
- `hipengine/loading/qwen35_gguf.py` already handles MTP-bearing files by
  reducing the AR executable block count when trailing `blk.N.nextn.*` tensors
  are present. Those tensors are intentionally ignored for AR today.
- Qwen3.6 35B-A3B GGUF baseline rows exist for gfx1151, including the MTP-bearing
  `UD-Q4_K_M` file. GGUF decode is currently slower than PARO on hipEngine but is
  directly comparable to llama.cpp.
- GGUF decode-repack work established a memory rule that matters for MTP too:
  prefer replacement resident layouts over duplicate raw+packed sidecars. A
  duplicate expert sidecar for Qwen3.6-class models can exceed the 24 GiB-class
  deployment envelope.

### Prior MTP / DFlash / Megakernel Lessons

Current branch already contains the prior `gguf-bulk-prefill`, `gfx1151`, and
`mpt-dflash` branch content; they are ancestors of this branch. Reuse these
lessons:

From [`MTP.md`](MTP.md):

- Always compare accepted/output, not llama.cpp `draft_n_accepted / draft_n` vs
  hipEngine per-cycle acceptance.
- W7900 retained B1 works because it lowers cycle cost enough; higher density is
  not useful if the verify cycle becomes too expensive.
- `decode_batched`, draft vocab cap, proposer caches, small-B W4 paths, and
  verifier scratch reuse are worth retesting only after the GGUF model path is
  accepted/correct.
- Many plausible optimizations no-held: full-vocab draft LM-head, B5/global large
  budgets, current graph capture, LM-head thread retunes, and oversized fused
  kernels.

From [`DFLASH.md`](DFLASH.md):

- Reuse provider-neutral `DraftBatch`, target verify, accept, commit, and KV
  transaction infrastructure. MTP-GGUF should not create a separate verifier
  stack.
- Native accept/commit must summarize on device and avoid full-logit host copies
  in the fast path where possible.
- Bulk/tree verifier correctness is tractable, but row cost dominates. Do not
  count a new draft policy as a throughput win until same-session AR is beaten.

From [`MEGAKERNEL.md`](MEGAKERNEL.md):

- Fusing small kernels or collapsing launches can regress at small verifier row
  counts. The failed PARO FFN megakernel showed that single-launch fusion is not
  automatically better than wide, GPU-filling staged kernels.
- For rows B+1 around 2-5, profile first. Optimize the actual buckets rather than
  assuming launch count is the bottleneck.

From [`TUNING-gfx1151.md`](TUNING-gfx1151.md):

- gfx1151 is memory/cache/launch sensitive and not just a smaller W7900.
- The first gfx1151 win was row-shape/chunking, not attention work.
- llama.cpp Vulkan beating llama.cpp HIP on this APU is a driver/roofline clue,
  not a direct implementation target.

## llama.cpp MTP Contract To Match

Reference source basis from the gfx1151 audit: local read-only
`/home/lhl/llama.cpp/llama.cpp-hip` at
`6e9007ae61f4e994c27484759caac6ef2aa32b30`.

The behavior to match conceptually:

1. **Integrated NextN tensors.** Qwen35MoE loads explicit `nextn` tensors such as
   `eh_proj`, `enorm`, `hnorm`, optional `embed_tokens`, and optional shared
   head tensors. They are not copied from a separate sidecar.
2. **Separate MTP graph/context on the target model.** llama.cpp creates an
   `LLAMA_CONTEXT_TYPE_MTP` context against the target model. It reserves only
   context/compute memory, not another full model copy.
3. **Target hidden-row seed.** The target decode path can expose `h_nextn` rows;
   the MTP drafter consumes the target hidden row plus the next token embedding.
4. **Filtered MTP layer set.** For Qwen35/Qwen35MoE hybrid models, the MTP
   context is limited to the NextN layer and uses plain dense-attention KV.
5. **Backend draft sampling.** Draft sampling can happen through backend sampler
   chains instead of always moving full logits to the host.
6. **Central accept accounting.** Server metrics expose `draft_n` and
   `draft_n_accepted`; cross-engine comparison must derive accepted/output.

Useful source links are recorded in [`TUNING-gfx1151.md`](TUNING-gfx1151.md) so
this document can focus on hipEngine implementation.

## Acceptance Accounting

Use these metric names in every artifact:

| Metric | Definition | Why |
| --- | --- | --- |
| `accept_per_draft` | accepted draft tokens / generated draft tokens or active candidate budget | Native engine/accounting diagnostic only |
| `accepted_per_output` | accepted draft tokens / predicted output tokens | Cross-engine density comparison |
| `visible_tokens_per_cycle` | target token + accepted draft tokens per verify cycle | hipEngine economics |
| `cycle_cost_ar_tokens` | MTP cycle wall / AR token wall | Break-even cost |
| `speedup_total_time` | AR total decode time / MTP total decode time | Noise-resistant speedup cross-check |

Never compare llama.cpp `accept_rate` directly to hipEngine
`acceptance_rate_mean`; they use different denominators.

## Implementation Milestones

### M0 — Inventory and Oracles

Deliverables:

- Add/extend an inspection script that reports:
  - declared GGUF block count;
  - AR executable block count;
  - ignored MTP block ids;
  - all `blk.N.nextn.*` tensor names, shapes, quant types, and byte sizes;
  - presence/fallback status for NextN embed/head tensors.
- Create a compact fixture for the local MTP GGUF inventory.
- Capture llama.cpp prompt tokenization/rendered prompt hashes for the D32 suite.

Acceptance:

- `python3 scripts/inspect_gguf.py /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`
  or a new dedicated script reports the `753` tensor / `20` MTP-like tensor
  inventory.
- Existing GGUF AR correctness fixtures still pass.

### M1 — Expose NextN/MTP Metadata in hipEngine

Deliverables:

- Extend the qwen35moe GGUF mapper with a first-class MTP block descriptor rather
  than only `ignored_block_ids`.
- Keep AR layer validation strict for blocks `0..39`; validate MTP block `40`
  separately.
- Add shape checks for expected NextN tensors:
  - `nextn.eh_proj.weight`
  - `nextn.enorm.weight`
  - `nextn.hnorm.weight`
  - `nextn.embed_tokens.weight` if present
  - shared head norm/head tensors if present
  - the MTP block's attention/FFN tensors.

Acceptance:

- Unit tests prove AR tensor validation still ignores MTP-only tensors for AR.
- New tests prove MTP tensor validation fails on missing/mis-shaped `nextn`
  tensors.

### M2 — GGUF AR Baseline Lock

Deliverables:

- Reproduce hipEngine GGUF AR baseline on gfx1151 with the MTP-bearing file.
- Confirm no regression versus current README/rationalization rows.
- Add exact commands and a JSON artifact before enabling MTP.

Acceptance:

- Same prompt suite, same `max_tokens=32`, same tokenizer path.
- No MTP execution yet; this is the control row.

### M3 — Draft-Only NextN Execution

Deliverables:

- Implement a correctness-first MTP draft head over GGUF resident weights:
  - consumes target hidden seed and accepted token id;
  - runs the NextN block once;
  - emits draft logits/top-k for one depth;
  - records logits/top-k for parity debugging.
- Use dense-BF16 fallback materialization first if needed. Speed is not the gate
  for this milestone.

Acceptance:

- Fixed hidden/token fixture produces deterministic finite logits.
- Draft top-k agrees with a llama.cpp trace or a CPU reference within the defined
  KL/top-1 gate.
- No full target trunk re-execution inside the MTP draft-only path.

### M4 — Target-Attached MTP Context

Deliverables:

- Add a `Qwen35GGUFMTPContext` or equivalent target-attached object that:
  - owns MTP scratch/KV/state buffers;
  - references target resident weights without duplicating large tensors;
  - captures/updates pending hidden seeds;
  - can run B1-B4 draft proposals.
- Integrate with existing `DraftBatch`/verifier/accept/commit infrastructure.

Acceptance:

- B1 exact D32 prompt suite passes against same-session GGUF AR.
- Artifact records accepted/output, accept/draft, visible tokens/cycle, cycle
  cost, and total-time speedup.

### M5 — B1-B4 Parity Sweep Against llama.cpp

Deliverables:

- Add a hipEngine GGUF MTP prompt-suite runner or extend the existing MTP
  economics runner to support `model=.gguf` and `candidate_budgets=1,2,3,4`.
- Run matched prompt/token suite against:
  - hipEngine GGUF AR;
  - hipEngine GGUF MTP B1-B4;
  - llama.cpp HIP B1-B4;
  - llama.cpp Vulkan B1-B4.

Acceptance:

- hipEngine B1-B4 exactness and accepted/output are reported per prompt.
- If hipEngine accepted/output lags llama.cpp by more than ~10% relative on the
  same budget, stop performance tuning and debug draft logits/model identity.

### M6 — Runtime/Kernel Optimization

Only after M5 acceptance parity:

- Profile B1 and best-density B row into:
  - NextN draft block;
  - target verifier;
  - LM-head/logit/sampling/readback;
  - accept/commit/KV update;
  - host gaps.
- Reuse GGUF decode-repack/T16 layouts where they reduce measured buckets.
- Add backend-side top-k/sampling for draft logits to avoid full-vocab D2H.
- Retest chunk/row shapes on gfx1151; do not assume W7900 B=1 is optimal.

Acceptance:

- Each retained optimization has same-suite exactness, same-device baseline, and
  a compact artifact.
- A default path is promoted only when it is exact and non-regressive.

## Benchmark Protocol

### Required Local Setup

Follow [`THEROCK.md`](THEROCK.md) and [`TUNING-gfx1151.md`](TUNING-gfx1151.md)
for gfx1151 environment setup. For profiled runs, precompute compiler version and
require cached builds:

```bash
hipcc --version > /tmp/hipengine-gfx1151-hipcc-version.txt
```

### llama.cpp Comparator

Use the committed sweep helper so acceptance denominators are stable:

```bash
python3 scripts/llamacpp_vulkan_mtp_sweep.py \
  --llama-dir /home/lhl/llama.cpp/llama.cpp-hip \
  --server-bin /tmp/llamacpp-hip-server-gfx1151-6e9007ae6/bin/llama-server \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --gpu 0 \
  --max-tokens 32 \
  --draft-max-values 1,2,3,4 \
  --prompts-file benchmarks/fixtures/llamacpp_mtp_bench_prompts.json \
  --ctx-size 8192 \
  --cache-type-k f16 \
  --cache-type-v f16 \
  --out-dir /tmp/llamacpp-hip-mtp-gguf-d32
```

For Vulkan, point `--llama-dir` / `--server-bin` at the Vulkan build and keep
all model/prompt/max-token flags identical.

### hipEngine GGUF AR Control

Use `scripts/qwen35_gguf_bench.py` for repeated fixed-shape AR rows until a
GGUF-MTP prompt runner lands. Required fields in artifacts:

- model path and GGUF tensor inventory hash;
- prompt source/token IDs;
- prefill tok/s and decode tok/s;
- tracked allocator peak;
- backend/quant/layout flags;
- exact command.

### hipEngine GGUF MTP Rows

The future runner must write:

- per-prompt AR and MTP token streams;
- exact AR match boolean and first mismatch window;
- accepted lengths by cycle;
- active budgets by cycle;
- `accept_per_draft`, `accepted_per_output`, visible density, cycle cost;
- draft/logit movement mode (`full_vocab_d2h`, `topk_device`, etc.);
- kernel/profile summaries when performance is claimed.

## Artifact Policy

Every retained diagnostic or performance row must update:

- `WORKLOG.md` with exact command, hardware, model, flags, and result;
- `benchmarks/results/<date>-mtp-gguf-*.json` compact artifact;
- `benchmarks/README.md` and `benchmarks/CHANGELOG.md` only once a row is a
  retained benchmark, not for every exploratory failed smoke.

Performance claims require:

- same-session AR baseline;
- exactness/correctness gate;
- artifact with acceptance denominators;
- no hidden fallback that changes model identity;
- commit immediately after validation.

## Open Questions

1. Does hipEngine GGUF B1 accepted/output match llama.cpp B1 once the same NextN
   tensors and prompt tokens are used?
2. Does llama.cpp's B4 advantage come mostly from model/draft density, backend
   sampling/logit movement, or verifier row economics?
3. Can hipEngine reuse current GGUF T16 decode-repack layouts for the NextN block
   without duplicating raw GGUF residency?
4. Is gfx1151's best MTP budget B1, B3, or B4 after the model path is matched?
5. Which exact tensors are optional in Qwen3.6 MTP GGUF exports, and what are the
   correct fallbacks when they are absent?

## Initial Backlog

- [ ] Add a GGUF MTP inventory fixture for the Unsloth `UD-Q4_K_M` MTP file.
- [ ] Extend `Qwen35GGUFModelMap` with an MTP block descriptor.
- [ ] Add unit tests for AR block-count exclusion plus MTP block validation.
- [ ] Implement draft-only NextN forward with dense fallback.
- [ ] Capture llama.cpp draft logits/top-k trace for one short prompt if possible.
- [ ] Add hipEngine GGUF MTP B1 prompt-suite runner.
- [ ] Run B1 exactness and accepted/output parity against llama.cpp B1.
- [ ] Extend to B2-B4 after B1 is exact.
- [ ] Add backend-side top-k draft sampling to avoid full-vocab D2H.
- [ ] Profile best exact row with `rocprofv3 --kernel-trace` after cached build
      warmup.

## Decision Log

- 2026-06-15: Created `mtp-gguf` branch and this plan after gfx1151 diagnostics
  showed llama.cpp B4 around `0.743/0.747` accepted/output and `1.7-1.8x` speedup
  while hipEngine PARO+sidecar MTP stayed below AR. The branch goal is to match
  llama.cpp's integrated GGUF NextN model path before further PARO-sidecar tuning.
