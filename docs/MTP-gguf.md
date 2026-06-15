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

The two things this plan's premise rests on — and that earlier drafts
under-specified — are made explicit here: **(1)** the exact NextN seed/forward
contract M3/M4 must reproduce (post-norm fp32 hidden seed, full attn+MoE NextN
sublayer, KVLiveSpans attention), and **(2)** the cross-engine parity-oracle
machinery (a `cpu_reference` NextN forward, a captured llama.cpp draft trace,
token-id parity, sampling parity). Fix the oracle and pin the seed contract
before any M3 implementation.

## Goal

Build a native hipEngine GGUF MTP path for:

```text
/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
source: https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF
```

and compare against llama.cpp HIP/Vulkan `--spec-type draft-mtp` using the shared
D32 prompt suite.

Throughout this doc, "B-N" (B1-B4) is shorthand for the llama.cpp
`--spec-draft-n-max` **cap** (default 3), not a fixed draft length: the drafter
can undershoot the cap via `p_min`/`n_min` (see "llama.cpp MTP Contract").

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
  model. M3/M4 name the concrete `(backend, layer, quant, variant)` keys instead
  of any branch (CLAUDE.md:28).
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
- GGUF tensor check: `753` tensors with trailing `blk.40.nextn.*` / MTP tensors
  (the `753`/`20` pair was reported by the gfx1151 diagnostic harness, not by
  `scripts/inspect_gguf.py`; see M0).

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
  are present. Those tensors are intentionally ignored for AR today, and are
  dropped before the resident layer map is built (so they are never
  materialized).
- Qwen3.6 35B-A3B GGUF baseline rows exist for gfx1151, including the MTP-bearing
  `UD-Q4_K_M` file. GGUF decode is currently slower than PARO on hipEngine but is
  directly comparable to llama.cpp.
- GGUF decode-repack work established a memory rule that matters for MTP too:
  prefer replacement resident layouts over duplicate raw+packed sidecars. A
  duplicate expert sidecar for Qwen3.6-class models can exceed the 24 GiB-class
  deployment envelope.
- **Spec-decode infrastructure is PARO/safetensors-only today.** `DraftBatch`,
  `TargetVerifyBatch`, accept/commit, and the target-attached MTP loader
  (`hipengine/loading/mtp.py`, validating safetensors `mtp.*` via `WeightIndex`)
  are wired into the PARO runner (`qwen35_paro_runner.py`), not the GGUF runner.
  The GGUF runner has **no** spec-decode wiring. M4 therefore builds net-new
  GGUF-side proposal/verify/accept/commit plumbing; it does not just "attach" to
  an existing GGUF hook.

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
- The prior W7900 device-chain candidate-buffering experiment was exact but
  same-suite **negative** (`0.6876x -> 0.6795x`, MTP.md ~L749). Treat "move draft
  sampling onto the device" as something that must net-remove D2H/launch work,
  not relocate it.

From [`DFLASH.md`](DFLASH.md):

- Reuse provider-neutral `DraftBatch`, target verify, accept, commit, and KV
  transaction infrastructure. MTP-GGUF should not create a separate verifier
  stack — but note (above) that this infra is currently PARO-only, so M4 ports it
  to the GGUF path rather than reusing it as-is.
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
- **Do not inherit W7900 B=1 as the gfx1151 operating point.** Retest B1/B2/B3 on
  gfx1151; the APU may amortize scarce memory traffic differently
  (TUNING-gfx1151.md:81-83). This is the operating-point question OQ4 / M6 must
  answer per-device.

## llama.cpp MTP Contract To Match

Reference source basis from the gfx1151 audit: local read-only
`/home/lhl/llama.cpp/llama.cpp-hip` at
`6e9007ae61f4e994c27484759caac6ef2aa32b30`. All file:line citations below are
against that checkout.

The behavior to match:

1. **Integrated NextN tensors.** Qwen35MoE loads explicit `nextn` tensors such as
   `eh_proj`, `enorm`, `hnorm`, optional `embed_tokens`, and optional shared
   head/norm tensors. They are not copied from a separate sidecar.
2. **Separate MTP graph/context on the target model.** llama.cpp creates an
   `LLAMA_CONTEXT_TYPE_MTP` context against the target model
   (`llama-context.cpp:28` maps it to `LLM_GRAPH_TYPE_DECODER_MTP`). It reserves
   only context/compute memory, not another full model copy — but that is **not
   free**: it allocates its own single-NextN-layer dense KV cache, its own
   compute/sched buffers, and a reused `embd_nextn` host buffer. Budget these
   against the 24 GiB envelope (see M4 allocator-peak gate).
3. **Target hidden-row seed = POST output-norm hidden, at fp32.** The trunk's
   `h_nextn` seed is the hidden state captured **after** the trunk `output_norm`
   (`qwen35moe.cpp:230-234`: `build_norm(cur, model.output_norm, …)` then
   `res->t_h_nextn = cur`; the same post-norm tensor feeds both the LM head and
   the MTP seed). It is exposed to the host via
   `ggml_backend_tensor_get_async` into a single reused `embd_nextn` buffer
   (`llama-context.cpp:1516-1522,1969-1977`) at **fp32** (`inp->h` is
   `GGML_TYPE_F32`). The NextN block re-normalizes the seed with `nextn.hnorm`
   before `eh_proj` (`qwen35moe.cpp:604`). The drafter pairs this hidden row with
   the **next-token ID** (written into `batch.token`); the NextN block embeds
   that token internally — the host does not assemble a separate "next token
   embedding" vector (`speculative.cpp:1071-1074`). The MTP batch carries **both**
   `batch.token` and `batch.embd` (the usual mutually-exclusive assert is
   relaxed, `speculative.cpp:870-874`).
4. **Filtered MTP layer set, dense KV, KVLiveSpans on the hipEngine side.** For
   Qwen35/Qwen35MoE the MTP context is limited to the NextN layer
   (`llama-model.cpp:2050,2149`) and that layer runs a **full dense-attention
   sublayer** over its own `wq/wk/wv/wo` + q/k norms with IMRoPE
   (`qwen35moe.cpp:599-661`), **plus a full MoE FFN** (routed experts + gated
   shared expert). The MTP context maintains its **own** single-NextN-layer KV by
   a catch-up decode mirroring the accepted token stream; it does **not** read the
   target's 40-layer KV. On the hipEngine side this is an attention-decode +
   paged-KV-write path and therefore **MUST** use the `KVLiveSpans` ABI
   (CLAUDE.md:31; `hipengine/kvcache/spans.py`), dense-filled
   (`spans_mode='uniform'`, `base_offsets`/`live_counts` set,
   `token_positions=None`, `evict_mask=None`); B>1 rows set
   `span_role='verify_chain'` (or `'verify_tree'`). Do not shortcut to
   `(block_table, context_len)`.
5. **Per-cycle pass count: N draft tokens = N+1 NextN forward passes.** The
   drafter runs one seed decode of `(id_last, pending_h)` (`speculative.cpp:1077`)
   then one `ctx_dft` decode per drafted token (`speculative.cpp:1148`, in the
   draft while-loop). The loop stops early when the drafted token's top-1
   probability `< p_min` (default 0.0, `speculative.cpp:1113-1118`) and drafts
   shorter than `n_min` (default 0, `speculative.cpp:1163-1164`) are discarded.
   So effective draft length = `min(n_max, first-token-where-p<p_min)`, and
   "B-N" is the cap. M3's "runs the block once" is true only for B1.
6. **Backend draft sampling is the DEFAULT.** `backend_sampling=true`
   (`common.h:309`; CLI advertises "default: enabled", `arg.cpp:3616-3624`).
   A per-seq backend `llama_sampler_chain` with `top_k=10` is attached to
   `ctx_dft` (`speculative.cpp:887-899`); only on backend-offload failure does it
   fall back to a CPU `common_sampler` (also `top_k=10`). Draft selection is
   **greedy top-1 from the top-k set** (`speculative.cpp:1110`). M3 draft-logit
   parity must compare greedy-top-1-from-`top_k=10`, **not** full-vocab argmax.
7. **Seed lifecycle / state machine.** After the target verify decode, capture all
   needed `h`-rows from `ctx_tgt` into a private `verify_h` snapshot **in the same
   step** (the `embd_nextn` buffer is reused and overwritten on the next decode);
   carry the last row across cycles via `pending_h`; `accept(n_accepted)`
   re-seeds `pending_h` from `verify_h[min(n_accepted, n_rows-1)]` (the last
   accepted row). Positions are explicit: seed at `pos=n_past`, chained draft
   token `i` at `pos=n_past+i+1` (a single fixed position is Gemma4-shared-mem
   only). `embeddings_nextn` is enabled on both contexts (target `masked=false`,
   MTP `masked=true`).
8. **Central accept accounting.** The server exposes `draft_n` and
   `draft_n_accepted` (`server-task.h:280-281`, set from
   `n_draft_total`/`n_draft_accepted` at `server-context.cpp:435-436`;
   `draft_n_accepted += ids.size()-1` at `:3595`). `draft_n` is **generated**
   draft tokens (`n_draft_total += draft.size()`, `server-context.cpp:2656`),
   which can be `< B`. Derive accepted/output from these **server timings**
   fields, not the common-layer `n_gen_tokens`/`n_acc_tokens` stats
   (`speculative.cpp:2030-2031,2062-2064`), which count whole-draft events
   differently.

The CLI knob that drives B-N is `--spec-draft-n-max` (default 3, `common.h:303`,
`arg.cpp:3587-3593`); `--spec-draft-p-min` / `--spec-draft-n-min` control the
early-stop and floor. The legacy `--draft` / `--draft-n` / `--draft-max` flags
are **removed** in this checkout and now error (`arg.cpp:3798-3804`).

Useful source links are also recorded in [`TUNING-gfx1151.md`](TUNING-gfx1151.md).

## Acceptance Accounting

Use these metric names in every artifact:

| Metric | Definition | Why |
| --- | --- | --- |
| `accept_per_draft` | accepted draft tokens / **generated** draft tokens | Matches llama.cpp `draft_n` denominator (`draft.size()`, can be < B via p_min/n_min); native diagnostic |
| `accepted_per_output` | accepted draft tokens / predicted output tokens | Cross-engine density comparison |
| `visible_tokens_per_cycle` | target token + accepted draft tokens per verify cycle | hipEngine economics |
| `cycle_cost_ar_tokens` | MTP cycle wall / AR token wall | Break-even cost |
| `speedup_total_time` | AR total decode time / MTP total decode time | Noise-resistant speedup cross-check |

`accept_per_draft` uses **generated** draft tokens as the denominator to match
llama.cpp's `draft_n`. If a hipEngine-only "active candidate budget" denominator
is reported, label it explicitly as a hipEngine budget metric distinct from
`draft_n` — do not silently swap denominators.

Never compare llama.cpp `accept_rate` directly to hipEngine
`acceptance_rate_mean`; they use different denominators. Derive the cross-engine
numbers from llama.cpp's server timings `draft_n` / `draft_n_accepted`, not the
common-layer `gen/acc tokens` stats.

### Parity Preconditions

These gate **before** any cross-engine accepted/output comparison (referenced
from M5). A divergence in any of them gets misattributed to draft logits and
defeats the parity goal:

- **(a) Token-id parity.** Capture llama.cpp prompt token-id arrays for the D32
  suite and assert hipEngine produces **identical ids** on the matched prompt
  (exact equality, not just a prompt hash). hipEngine's GGUF tokenizer is an
  explicit byte-BPE approximation and the bench fixture stores raw text, so equal
  text + a hash does not guarantee equal ids. Pin BOS/chat-template/special
  tokens.
- **(b) Sampling parity.** Greedy/argmax draft+target on both engines. Launch
  llama.cpp with `--temp 0` (or `top-k 1`) and a fixed seed; document the argmax
  tie-break (lowest token id); assert `hipEngine-AR-greedy == llama.cpp-AR-greedy`
  on the same tokens. hipEngine's verifier already requires greedy-fast sampling
  and its spec oracle is same-session greedy AR equality.
- **(c) Numeric gate.** `KL <= 0.05` AND top-1 agreement `>= 90%` vs
  `kernels/cpu_reference/` on fixture inputs (the standard CLAUDE.md kernel gate).

Current status: `scripts/gguf_mtp_parity_precheck.py` wraps the token-id
inventory comparison and optional exact sampling-settings comparison into a
single fail-fast JSON gate. The llama.cpp HIP `/tokenize` D32 artifact is
captured at
`benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json`,
and the hipEngine fixture now matches it on all 9 D32 prompts after porting the
Qwen3.5 pre-tokenizer semantics. The B1 deterministic sampling/request artifact
is committed at `benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json`,
so Parity Preconditions (a) and (b) have fixture coverage; M5 still also requires
the numeric KL/top-1 gate and actual GGUF MTP execution.

## Implementation Milestones

### M0 — Inventory and Oracles

Deliverables:

- Add/extend an inspection script that reports:
  - declared GGUF block count;
  - AR executable block count;
  - ignored MTP block ids;
  - all `blk.N.nextn.*` tensor names, shapes, quant types, and byte sizes;
  - the full trailing MTP block tensors (attn/ffn/norm), not just `nextn.*`;
  - presence/fallback status for NextN embed/head tensors.
  - **Current status:** `scripts/inspect_gguf.py --json` now reports the
    metadata-only `qwen35_mtp_inventory` block, including per-tensor shape/qtype
    rows and explicit optional NextN present/fallback status.
- Create a compact fixture for the local MTP GGUF inventory. Current fixture:
  `benchmarks/fixtures/qwen36_35b_a3b_ud_q4_k_m_mtp_inventory.json`.
- **Capture a llama.cpp draft logits/top-k trace for at least one short D32
  prompt (required, promoted from backlog).** This is one of the two M3 parity
  oracles; M3 blocks on having it or the `cpu_reference` NextN forward.
- Capture llama.cpp prompt tokenization / token-id arrays + rendered prompt
  hashes for the D32 suite (feeds Parity Precondition (a)). Current fixtures:
  `benchmarks/fixtures/hipengine_gguf_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json`
  and
  `benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json`.
  The committed fixtures now match exactly on all 9 prompts and are covered by
  regression tests; re-run `scripts/gguf_mtp_parity_precheck.py` before any M5
  accepted/output comparison.

Acceptance:

- The new/extended script reports the full inventory. The MTP block is the full
  **20-tensor** trailing block (`4` `nextn.*` — `eh_proj`, `enorm`, `hnorm`,
  `shared_head_norm` — plus `16` attn/ffn/norm); only `4` are `nextn.*`. The
  `753`/`20` pair quoted in Starting Evidence came from the gfx1151 diagnostic
  harness, not `inspect_gguf.py`.
- Quant inventory observed in the local `UD-Q4_K_M` file (confirm via the M0
  script run, treat as expected not authoritative): `eh_proj`=Q8_0,
  `enorm`/`hnorm`/`shared_head_norm`=F32, block-40 attention=Q8_0, routed experts
  Q4_K (gate/up) + Q5_K (down), `ffn_gate_inp`=BF16. `embed_tokens` and
  `shared_head_head` are **absent** in this file.
- Existing GGUF AR correctness fixtures still pass.

### M1 — Expose NextN/MTP Metadata in hipEngine

Deliverables:

- Extend the qwen35moe GGUF mapper (`hipengine/loading/qwen35_gguf.py`,
  `Qwen35GGUFModelMap` — confirm the exact symbol name in-tree) with a
  first-class MTP block descriptor rather than only the ignored-block-count
  reduction.
- Keep AR layer validation strict for blocks `0..39`; validate MTP block `40`
  separately.
- Required vs optional NextN tensor table (answers former Open Question #5,
  verified in `qwen35moe.cpp`; tensor names use the `blk.<id>.` prefix and the
  hipEngine `attn_output`/`post_attention_norm` naming may diverge from
  llama.cpp's enum strings — map both):

  | Tensor | Status | Fallback when absent |
  | --- | --- | --- |
  | `nextn.eh_proj` `[2*n_embd, n_embd]` | required | — |
  | `nextn.enorm` `[n_embd]` | required | — |
  | `nextn.hnorm` `[n_embd]` | required | — |
  | block-40 attn (`attn_norm`, `attn_post_norm`, `wq/wk/wv/wo`, `attn_q_norm`, `attn_k_norm`) | required | — |
  | block-40 MoE (`ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_gate_inp_shexp`, `ffn_{gate,up,down}_shexp`) | required | — |
  | `nextn.embed_tokens` | optional (`TENSOR_NOT_REQUIRED`) | `model.tok_embd` (target) |
  | `nextn.shared_head_head` | optional | `model.output` (target) |
  | `nextn.shared_head_norm` | optional | `model.output_norm` (target) |

  In the local `UD-Q4_K_M` file `embed_tokens` and `shared_head_head` are absent
  (only `shared_head_norm` present), so the draft head/embedding reuse target
  weights and their shape checks **must be optional**.

Current status:

- `Qwen35GGUFModelMap.mtp_blocks`, `qwen35_gguf_mtp_block_inventories()`, and
  `validate_qwen35_gguf_mtp_blocks()` expose the AR-ignored trailing block as a
  separate metadata gate. The validation gate now fails missing required tensors,
  unexpected trailing-block tensors, and mis-shaped effective MTP slots while
  tolerating absent optional embed/head tensors through target-weight fallbacks.

Acceptance:

- Existing AR-block-exclusion coverage is **extended**, not added:
  `tests/test_qwen35_gguf_mtp_mapping.py` already proves AR tensor validation
  ignores MTP-only tensors. Scope new work to the missing/mis-shaped `nextn`
  validation path.
- New tests prove MTP tensor validation fails on missing/mis-shaped **required**
  `nextn` tensors and tolerates absent **optional** tensors via the fallback
  table above.

### M2 — GGUF AR Baseline Lock

Deliverables:

- Reproduce hipEngine GGUF AR baseline on gfx1151 with the MTP-bearing file.
- Confirm no regression versus current README/rationalization rows.
- Add exact commands and a JSON artifact before enabling MTP.

Acceptance:

- Same prompt suite, same `max_tokens=32`, same tokenizer path.
- No MTP execution yet; this is the control row.

### M2.5 — Expose Target Hidden Seed from GGUF Decode

This is the single load-bearing plumbing prerequisite between M2 and M3: nothing
in M0-M3 otherwise **produces** the seed M3 consumes.

Deliverables:

- Add a GGUF AR decode-path hidden-seed tap that exposes the per-accepted-token
  **POST-`output_norm`** hidden row (`h_nextn`) at **fp32** (llama.cpp `inp->h`
  is `GGML_TYPE_F32`), analogous to
  `qwen35_paro_runner.step_with_hidden_taps`. `run_prompt_hidden` today returns
  the post-norm hidden as **BF16** with no per-token tap — correct provenance,
  wrong dtype, no per-token hook.
- Numeric contract (referenced by M3): capture the seed and next-token embedding
  at fp32; apply `enorm`/`hnorm` RMSNorm in fp32; `eh_proj` input =
  `concat([enorm(tok_embd), hnorm(target_hidden)], dim=feature)` with the
  **embedding segment FIRST**; `eh_proj` is `2*n_embd -> n_embd`. Record the
  chosen seed dtype in the parity artifact and treat BF16-vs-fp32 seed as a
  parity variable to ablate if top-k disagrees.

Current status:

- `Qwen35GGUFResidentSession.step(..., capture_hidden_seed_fp32=True)` and
  `prefill(..., capture_hidden_seed_fp32=True)` populate a guarded fp32
  post-`output_norm` device seed row and expose it via `mtp_draft_seed()`.
  Default AR generation leaves the tap off.
- Fixture `benchmarks/fixtures/qwen35_gguf_hidden_seed_output_norm_fixture.json`
  pins a deterministic post-`output_norm` seed row. A HIP-guarded test proves
  the `gguf_rmsnorm_bf16_f32_weight_out_f32` tap is finite and matches the CPU
  RMSNorm oracle for that row.

Acceptance:

- A fixture asserts the captured seed is finite and matches the `cpu_reference`
  trunk output within tolerance. M3 depends on this milestone.

### M3 — Draft-Only NextN Execution

Deliverables:

- Implement a correctness-first MTP draft head over GGUF resident weights. The
  NextN block is **not** a self-contained projection+norm head — spell out the
  full forward:
  `enorm(embed(token))` and `hnorm(target_hidden)` RMS-normed separately ->
  `concat` (embedding segment first) -> `eh_proj` (`2*n_embd -> n_embd`) ->
  full dense self-attention over its own `wq/wk/wv/wo` (+ gated sigmoid output,
  IMRoPE) -> MoE FFN (routed experts + gated shared expert) -> `shared_head_norm`
  (-> `model.output_norm` if absent) -> `shared_head_head` LM-head
  (-> `model.output` if absent). Verified `qwen35moe.cpp:583,599-661,719-733`.
- Specific deliverables this implies:
  - consumes the M2.5 target hidden seed (post-norm, fp32) and accepted token id;
  - wire the draft LM-head + embedding to **target** weights when the optional
    `nextn.embed_tokens` / `shared_head_head` tensors are absent (they are, in
    the local file);
  - provide a **KVLiveSpans** attention path for the NextN block (dense-filled
    `spans_mode='uniform'`, `token_positions=None`, `evict_mask=None`); append
    K/V via a registered `paged_kv_write` span variant and decode via a
    registered `paged_attn_decode` span variant (CLAUDE.md:31);
  - materialize/route the NextN **MoE experts** (Q4_K gate/up, Q5_K down) in the
    dense-BF16 fallback — not just norms+projection; `eh_proj` is Q8_0, norms are
    F32;
  - emits draft logits/top-k for one depth (B1); depth>1 runs the block once per
    depth (N+1 passes — see contract item 5);
  - records logits/top-k for parity debugging.
- **Registry keys (no branches).** The NextN draft attention/FFN/sampler kernels
  register under `KernelKey(backend, layer, quant='w4_gguf', variant)`, resolved
  via `registry.resolve` / the fusion planner (`hipengine/kernels/registry.py`),
  never an `if backend==`/`if quant==` branch (CLAUDE.md:28). GGUF K-quant
  (Q4_K_M) dequant is the `w4_gguf` quant-axis plugin; the dense-BF16 fallback is
  reached through the registry's generic quant->fp16/bf16 fallback, not a
  hand-written branch.
- **RED-first:** commit a failing fixed `(token, hidden)` fixture + expected
  top-k before implementation (math change — guilty until proven correct,
  CLAUDE.md). Keep the numpy `cpu_reference` NextN forward in
  `kernels/cpu_reference/ops.py` registered through the four-axis registry,
  implementing the forward above; ship a fixture as the offline oracle.

#### CPU Call-Spec Bridge

Use the metadata-only call-spec dumper to hand future parity harnesses the
validated GGUF tensor names, qtypes, scalar kwargs, and runtime inputs for the
`cpu_reference` NextN oracle without materializing weights:

```bash
/home/lhl/miniforge3/envs/therock/bin/python scripts/gguf_mtp_call_spec.py \
  /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --require-mtp --layer 40 --indent 2
```

Expected local shape:

- one `mtp_draft_call_specs[]` entry for `layer_id: 40`;
- `cpu_reference_kernel == ["cpu_reference", "mtp_nextn_layer", "gguf_moe",
  "qwen35_dense_logits"]`;
- direct tensor args include `wq_weight -> blk.40.attn_q.weight` and
  `shared_head_weight -> output.weight`;
- qtype args resolve to `gate_qtype=Q4_K`, `up_qtype=Q4_K`,
  `down_qtype=Q5_K`, `shared_qtype=Q8_0`;
- dynamic inputs document the harness-provided values: fp32 `hidden_seed`,
  gathered `token_embedding` rows, optional dense CPU `key_cache`/`value_cache`,
  positions/context counts, and paired RoPE cos/sin tables.

The CLI reads GGUF metadata/tensor headers only. It does **not** load tensor
payloads, run kernels, allocate runtime KV, or bypass the future M4 KVLiveSpans
attention/KV-write requirement. Use `--layer` for exact block selection;
`--require-mtp` makes non-MTP files or over-filtered selections fail fast instead
of emitting an empty spec list.

Current status:

- The NumPy `cpu_reference` full NextN layer is registered as
  `KernelKey("cpu_reference", "mtp_nextn_layer", "gguf_moe", "qwen35_dense_logits")`.
  Fixture `benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json`
  pins a deterministic hidden/token row, finite logits, and top-k IDs for the
  full `eh_proj -> attention -> MoE/shared expert -> shared head` oracle.
  The attention sublayer now has both the older dense CPU-cache path and a
  NumPy-only KVLiveSpans-shaped paged-cache path using
  `(kv_base_offsets, kv_live_counts, kv_token_positions, kv_evict_mask)`, so the
  M4 ABI can be exercised before HIP kernels exist.
- llama.cpp verbose `draft-mtp` candidate fixture
  `benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json` captures a
  short `explain_concept` prompt trace: 2 draft calls, top-3 candidates per call,
  and 0/2 accepted drafts. It was captured with backend draft sampling disabled
  only to expose candidate probabilities in the log; it is an oracle/debug
  fixture, not a performance benchmark.

Acceptance:

- Fixed hidden/token fixture produces deterministic finite logits.
- Draft top-k agrees with the captured llama.cpp trace (M0) **or** the
  `cpu_reference` NextN forward within the gate: `KL <= 0.05` AND top-1 agreement
  `>= 90%` vs `kernels/cpu_reference/`. M3 blocks on at least one of these
  oracles actually existing.
- No full target trunk re-execution inside the MTP draft-only path.

### M4 — Target-Attached MTP Context

Current status:

- `hipengine.speculative.gguf_mtp.Qwen35GGUFMTPContext` is a target-attached
  scaffold for the GGUF path. It references the target resident session, records
  ready fp32 post-`output_norm` seed rows, applies the llama.cpp
  `verify_h[min(n_accepted, n_rows - 1)]` accept/reseed rule, and builds B1 draft
  rows carrying both token IDs and embedding-seed pointers. It does not allocate
  MTP KV buffers or run draft kernels yet.

Deliverables:

- Add a `Qwen35GGUFMTPContext` or equivalent target-attached object that:
  - owns MTP scratch/KV/state buffers — its **own** single-NextN-layer dense KV
    cache, populated by a catch-up decode mirroring the accepted token stream (it
    does **not** read the target's 40-layer KV);
  - references target resident weights without duplicating large tensors;
  - captures/updates pending hidden seeds via the contract item 7 state machine:
    snapshot `verify_h` in-step before the reused `embd_nextn` buffer is
    overwritten; carry `pending_h` across cycles; `accept(n_accepted)` re-seeds
    from `verify_h[min(n_accepted, n_rows-1)]`; set explicit positions
    (seed `pos=n_past`, chained draft token `i` at `pos=n_past+i+1`); the batch
    carries both token id and embd; enable `embeddings_nextn` on both contexts;
  - can run B1-B4 draft proposals.
- Build net-new GGUF-side `DraftBatch`/verify/accept/commit wiring. The existing
  `DraftBatch`/`TargetVerifyBatch`/accept/commit live in the PARO/safetensors
  runner stack (`qwen35_paro_runner.py`, `batch_scheduler.py`,
  `hipengine/speculative/`, `loading/mtp.py`); the GGUF runner has none of it.
  The `DraftBatch` ABI must permit a row carrying **both** a token id and an
  embedding seed.

Acceptance:

- B1 exact D32 prompt suite passes against same-session GGUF AR.
- Artifact records accepted/output, accept/draft, visible tokens/cycle, cycle
  cost, and total-time speedup.
- **Allocator-peak gate:** record the **measured** tracked allocator peak
  (`core/memory.py` `peak_allocated_bytes` + amdgpu VRAM peak) with the MTP
  context resident alongside the AR model; it must stay within the 24 GiB-class
  envelope. The runner must emit a measured peak, not a placeholder constant (the
  existing spec artifact hard-codes ~22 GB). Budget the MTP-context overhead: a
  separate single-NextN-layer dense KV cache (sized by `n_ctx_seq` and the draft
  `cache_type_k/v`), its own compute/sched buffers, and the `embd_nextn` host
  buffer — weights are shared with the target, but this is not free.

### M5 — B1-B4 Parity Sweep Against llama.cpp

Deliverables:

- Add a hipEngine GGUF MTP prompt-suite runner. Note scope: `candidate_budgets`
  is **already** supported by `scripts/mtp_prompt_suite_economics.py`
  (`--candidate-budgets`), but `model=.gguf` is a **large** extension gated on
  M3/M4 — the MTP execution lives in the PARO-only
  `mtp_verifier_economics.py -> mtp_chain_e2e_smoke.py` child stack with no GGUF
  path (`grep gguf` in all three returns nothing). Plan for a new GGUF MTP child
  runner, not a flag flip on the wrapper. Current preflight child:
  `scripts/gguf_mtp_b1_prompt_suite.py`, which validates MTP metadata plus
  token/sampling parity and emits a blocked artifact until native GGUF MTP draft
  execution is implemented.
- Run matched prompt/token suite against:
  - hipEngine GGUF AR;
  - hipEngine GGUF MTP B1-B4;
  - llama.cpp HIP B1-B4;
  - llama.cpp Vulkan B1-B4.

Acceptance:

- **Parity Preconditions (a)/(b)/(c) pass per-prompt before any accepted/output
  number is compared.**
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
- Reuse GGUF decode-repack/T16 layouts where they reduce measured buckets — note
  this is **conditional**: today `_spec_for_tensor` has no `nextn` slot-path cases
  and T16/pack8 selection is gated to `.ffn_*_exps` + `root.lm_head`, so NextN
  tensors need new slot-path predicates or default to RAW_GGUF/dense-BF16 (see
  Open Question 3).
- Add backend-side top-k/sampling for draft logits to avoid full-vocab D2H. This
  matches llama.cpp's existing greedy-top-1-from-`top_k=10` behavior (not a novel
  win). It **must** register as a variant (e.g. `topk_device`) and keep the
  numerically-equivalent `full_vocab_d2h` path registered as the unfused fallback
  and correctness oracle (CLAUDE.md:29). Caveat: the prior W7900 device-chain
  candidate-buffering attempt was exact but suite-negative
  (`0.6876x -> 0.6795x`, MTP.md ~L749); a device-side top-k path must
  net-remove D2H/launch work, not relocate it.
- Retest chunk/row shapes on gfx1151; do not assume W7900 B=1 is optimal
  (TUNING-gfx1151.md:81-83).

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

The live draft-length knob is `--spec-draft-n-max` (default 3); the legacy
`--draft` / `--draft-n` / `--draft-max` flags are removed and will error if you
invoke `llama-server` directly. `--spec-draft-p-min` / `--spec-draft-n-min`
control early-stop/floor — record them so B-N is like-for-like (B-N is the
`n_max` cap, not a fixed draft length).

Use the committed sweep helper so acceptance denominators are stable (it maps
`--draft-max-values` to `--spec-draft-n-max` internally,
`scripts/llamacpp_vulkan_mtp_sweep.py:146`):

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

Note: `qwen35_gguf_bench.py` now emits a GGUF tensor-inventory hash plus
exact command/argv capture for AR control artifacts. It still hardcodes
`backend='hip_gfx1100'` even for gfx1151 runs, so record the actual hardware in
WORKLOG/artifact context until backend detection is added.

### hipEngine GGUF MTP Rows

The future runner must write:

- per-prompt AR and MTP token streams;
- exact AR match boolean and first mismatch window;
- accepted lengths by cycle;
- active budgets by cycle;
- `accept_per_draft`, `accepted_per_output`, visible density, cycle cost;
- draft/logit movement mode (`full_vocab_d2h`, `topk_device`, etc.);
- the captured seed dtype (fp32 vs bf16) for the parity artifact;
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
   tensors and prompt tokens are used (and the fp32 post-norm seed contract is
   honored)?
2. Does llama.cpp's B4 advantage come mostly from model/draft density, backend
   sampling/logit movement, or verifier row economics?
3. Can hipEngine reuse current GGUF T16 decode-repack layouts for the NextN block
   without duplicating raw GGUF residency? (Conditional, not automatic: NextN
   tensors have no slot-path predicate today and T16/pack8 selection is gated to
   `.ffn_*_exps` + `root.lm_head`, so they need new predicates or default to
   RAW_GGUF/dense-BF16.)
4. Is gfx1151's best MTP budget B1, B3, or B4 after the model path is matched?
   Do not inherit W7900 B=1 (TUNING-gfx1151.md:81-83).

(Former Open Question 5 — which exact tensors are optional and their fallbacks —
is now answered by the M1 required/optional table.)

## Initial Backlog

- [x] Add a GGUF MTP inventory fixture for the Unsloth `UD-Q4_K_M` MTP file
      (full 20-tensor trailing block, 4 `nextn.*`):
      `benchmarks/fixtures/qwen36_35b_a3b_ud_q4_k_m_mtp_inventory.json`.
- [x] **Capture a llama.cpp draft logits/top-k trace for one short prompt
      (required M0 oracle, not "if possible").** Fixture:
      `benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json`.
- [x] Capture llama.cpp D32 prompt token-id arrays (Parity Precondition (a));
      hipEngine-side and llama.cpp-side D32 fixtures are committed at
      `benchmarks/fixtures/hipengine_gguf_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json`
      and
      `benchmarks/fixtures/llamacpp_hip_prompt_tokens_qwen36_35b_a3b_ud_q4_k_m_d32.json`.
- [x] Extend `Qwen35GGUFModelMap` with an MTP block descriptor + required/optional
      fallback table.
- [x] Extend `tests/test_qwen35_gguf_mtp_mapping.py` for MTP-block validation
      (missing/mis-shaped required, tolerated optional).
- [x] **M2.5:** expose the fp32 post-`output_norm` per-token hidden seed from the
      GGUF decode path (per-token tap; `run_prompt_hidden` returns BF16 today).
      Fixture: `benchmarks/fixtures/qwen35_gguf_hidden_seed_output_norm_fixture.json`.
- [x] **Add a `cpu_reference` NextN forward** in `kernels/cpu_reference/ops.py`
      registered under `(cpu_reference, mtp_nextn_layer, gguf_moe,
      qwen35_dense_logits)` as the offline oracle, with a deterministic fixture:
      `benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json`.
- [ ] Implement draft-only NextN forward (full attn+MoE) with a KVLiveSpans
      attention path and dense fallback; CPU-reference coverage now includes
      both dense and KVLiveSpans-shaped paged-cache attention, and
      `Qwen35GGUFMTPContext` now covers the B1 seed/batch state scaffold, but
      HIP/runtime registration under `KernelKey(backend, layer, quant='w4_gguf',
      variant)` remains open.
- [x] Add hipEngine GGUF MTP B1 prompt-suite runner (new GGUF child, not a wrapper
      flag): `scripts/gguf_mtp_b1_prompt_suite.py` currently implements
      preflight + blocked-artifact emission; B1 exactness/execution remains in
      the next backlog row.
- [x] Gate Parity Preconditions (token-id + sampling parity) before comparison;
      `scripts/gguf_mtp_parity_precheck.py` now provides the fail-fast gate,
      `benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json` pins the
      B1 deterministic sampling settings, and the committed hipEngine/llama.cpp
      D32 token fixtures match on all 9 prompts.
- [ ] Run B1 exactness and accepted/output parity against llama.cpp B1.
- [ ] Extend to B2-B4 after B1 is exact.
- [ ] Add backend-side top-k draft sampling as a `topk_device` variant, keeping
      `full_vocab_d2h` registered as the unfused fallback/oracle.
- [ ] Profile best exact row with `rocprofv3 --kernel-trace` after cached build
      warmup.

## Decision Log

- 2026-06-15: Created `mtp-gguf` branch and this plan after gfx1151 diagnostics
  showed llama.cpp B4 around `0.743/0.747` accepted/output and `1.7-1.8x` speedup
  while hipEngine PARO+sidecar MTP stayed below AR. The branch goal is to match
  llama.cpp's integrated GGUF NextN model path before further PARO-sidecar tuning.
- 2026-06-15: Sharpened the plan against the real llama.cpp source
  (`@6e9007ae6`), the hipEngine GGUF loader/registry, and CLAUDE.md invariants.
  Key corrections: seed is **post-`output_norm` at fp32** (not pre-norm);
  draft-length knob is `--spec-draft-n-max` (legacy `--draft*` removed); backend
  `top_k=10` sampling is the **default**; N draft tokens = N+1 NextN passes; the
  NextN block is a full attn+MoE sublayer needing the **KVLiveSpans** ABI;
  `accept_per_draft` denominator = generated draft tokens. Added M2.5 (fp32
  post-norm hidden seed tap), a `cpu_reference` NextN forward + captured
  llama.cpp trace as required M0/M3 oracles, a Parity Preconditions subsection
  (token-id + sampling parity), explicit four-axis registry keys, an unfused
  fallback requirement for device top-k, and an M4 measured allocator-peak gate.
  Clarified that DraftBatch/verifier infra is PARO/safetensors-only (M4 is
  net-new GGUF wiring) and that M5 `model=.gguf` is a large extension, not a flag
  flip.
