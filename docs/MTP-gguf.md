# MTP-GGUF Plan

Last updated: 2026-07-19
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

## 2026-07-12 gfx1100 Finding and Cross-Backend Native Cycle Launcher

The current W7900 result changes the optimization diagnosis. The hardware does
not lack enough compute to make MTP useful: llama.cpp demonstrates a large MTP
speedup on the same card and GGUF. hipEngine's remaining gap is primarily how
we submit and orchestrate a speculative cycle, not a missing gfx1151 kernel port
or insufficient W7900 arithmetic throughput.

Durable evidence:

- hipEngine graph-AR correction and current MTP economics:
  [`2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json`](../benchmarks/results/2026-07-12-w7900-gfx1100-gguf-graph-ar-refresh.json).
- Current rebuilt llama.cpp base-vs-MTP diagnostic:
  [`2026-07-12-w7900-llamacpp-mtp-natural25-diagnostic.json`](../benchmarks/results/2026-07-12-w7900-llamacpp-mtp-natural25-diagnostic.json).

Matched natural25 request / 24 timed-transition results on W7900:

| Route | AR tok/s | MTP tok/s | MTP / AR | accepted/output |
| --- | ---: | ---: | ---: | ---: |
| hipEngine `llama-compat` | **93.30** | 79.70 | **0.8542x** | **0.608** |
| llama.cpp HIP build 9648 | 78.25 | **116.88** | **1.4936x** | 0.584 |

The llama.cpp row is an external diagnostic (`performance_claim=false`): it
uses F16 KV and the preserved dirty instrumentation patchset, while hipEngine
uses BF16 KV. It is nevertheless decisive for architecture planning. llama.cpp
is **46.6% faster than hipEngine MTP despite slightly lower accepted/output**,
so acceptance and available GPU compute are not the primary blockers.

### Current Break-Even Accounting

For hipEngine `llama-compat` B2:

- `240 / 94 = 2.553` visible outputs/cycle;
- graph AR costs `10.718 ms/transition`, so the equivalent AR budget is
  `2.553 * 10.718 = 27.36 ms/cycle`;
- hipEngine MTP costs `12.578 ms/output`, or `32.12 ms/cycle`;
- target verification alone costs about `28.41 ms/cycle`, before proposal,
  acceptance, and commit overhead.

At current acceptance, the complete cycle therefore needs at least a **14.8%**
wall reduction merely to beat hipEngine AR. Matching llama.cpp's
`8.556 ms/transition` requires about a **32%** reduction in hipEngine MTP wall
per visible output. Perfect B2 acceptance (three visible outputs every cycle)
would approximately break even at current cycle cost, but llama.cpp wins with
lower acceptance, confirming that execution cost is the more important lever.

Exact/default B3 has a harder current regime: about two visible outputs per
cycle, `29.39 ms/cycle` MTP wall, and only `20.25 ms/cycle` of equivalent graph
AR work. It needs roughly a **31%** cycle reduction at unchanged density.

### gfx1151 Parity Status

The GGUF MTP implementation is source-level shared across both RDNA backends:
`hip_gfx1151` aliases the registered gfx1100 callables and recompiles the same
HIP bodies for gfx1151. The current exact and `llama-compat` routes, dp4a/X8/Q8
sidecars, draft optimizations, target verifier, and accept/commit machinery are
not missing a gfx1151-only kernel family on W7900.

What did not transfer was the performance environment. gfx1151's integrated
CPU/GPU architecture has much cheaper submission, so Python/ctypes launch
orchestration was less visible and the shared path could match or beat
llama.cpp there. On discrete gfx1100, graph AR removes the eager launch tax,
but MTP still executes a data-dependent proposal/verify/accept/commit cycle
through many host submissions. The AR graph admission fixed the denominator;
it did not solve MTP cycle submission.

This means the next implementation should not be a gfx1100-only workaround.
The same host-bound orchestration is avoidable on gfx1151 and in the PARO and
DFlash speculative stacks, even where it is not yet the largest measured
bucket. Build one provider-neutral native path, then validate and promote it
separately per backend/provider.

### Target Architecture: Native Speculative Cycle Launcher

Build a native C/C++ launcher (working name `NativeSpecCycleLauncher`) that
reduces the hot loop to one host-language boundary per speculative cycle, with
a later option to run multiple cycles per call. It is an orchestration layer,
not a monolithic math megakernel.

The launcher should:

1. Consume a versioned, provider-neutral device control block containing raw
   pointers and bounded shapes for active rows, candidate budgets, token IDs,
   positions, parent/depth metadata, `KVLiveSpans`, hidden seeds, acceptance
   summaries, commit rows, and output cursors.
2. Use kernel handles/launch descriptors resolved once through the four-axis
   registry before the hot loop. Do **not** add backend/quant/provider branches
   to engine or model code. GGUF MTP, PARO MTP, and DFlash attach adapters that
   populate the common control block and resolve their registered primitives.
3. Submit proposal/NextN, target verification, sampling/acceptance, recurrent
   and KV commit/scatter, reseed, and cursor update from native code on
   session-owned streams. Python must not dispatch each layer or kernel.
4. Keep candidate IDs, probabilities, accept lengths, commit rows, recurrent
   state choices, and KV transaction metadata device-resident. Return only the
   bounded visible-output/result payload needed by the scheduler/API; avoid
   per-stage scalar D2H synchronization.
5. Preserve `KVLiveSpans` as the attention ABI for every provider. The native
   launcher receives the spans/control pointers; it must not introduce a
   `(block_table, context_len)` shortcut.
6. Support stable shape buckets such as `(provider, B, verifier rows, context
   bucket, route)` and optionally use HIP graphs for proven stable subgraphs.
   Native C++ submission is the baseline because acceptance and commit are
   data-dependent; correctness must not depend on one giant fixed graph.
7. Keep the current Python/orchestrated chain as the exact fallback and oracle.
   Existing unfused primitive chains remain registered for every fused kernel,
   as required by the project invariants.

The C ABI must take raw device pointers and scalar metadata, never framework
tensors. Provider adapters belong at the plugin boundary. A conceptual registry
shape is `(backend, speculative_cycle, provider_quant, native_v1)`, with the
exact layer/quant names following the existing registry catalog rather than
hard-coded dispatch branches.

### Shared Scope

The launcher is intended for all current speculative providers:

- GGUF integrated NextN/MTP (`exact/default` and explicit `llama-compat`);
- PARO target plus MTP sidecar;
- dense 27B DFlash;
- future tree/coverage speculative policies that use the shared
  `DraftBatch`/`TargetVerifyBatch`/accept/commit contracts.

Provider-neutral does not mean one performance policy. Each provider keeps its
own proposal graph, verifier shape, state semantics, and registry keys. The
common layer owns cycle control, native launch sequencing, device-resident
accept/commit metadata, and scheduler-facing results.

### Delivery Order

1. **N0 — ABI and oracle (landed 2026-07-13):** the public
   `hipengine.speculative.native_cycle` contract and
   `hipengine/speculative/native_cycle_abi.h` define matching version-1 control
   and result layouts, lifecycle/error enums, borrowed-pointer ownership,
   bounded capacities, explicit dtypes, stage-dependent validation, and a
   CPU/fake launcher. The target-only adapter consumes the existing
   `TargetVerifyBuffers` plus `KVLiveSpans`; the Python chain remains unchanged.
2. **N1 — Native target block (landed diagnostic 2026-07-19):** the initial
   gfx1100 single-request B2 adapter proved byte-exact target/state/KV behavior
   and one-call graph submission, but rejected per-cycle position-bound capture.
3. **N1R — Reusable B1/B2 target graphs (retained 2026-07-19):** one
   fixed-address graph per two-/three-row bucket now consumes live device token,
   position, context, and cursor metadata. The real 35B oracle is byte-exact at
   two B2 positions plus B1. Two clean full-suite runs reach **123.33/122.67
   tok/s**, preserve all prior IDs/cycles and 80.45% acceptance, beat every true
   AR split/category, and clear llama.cpp's 115.44 tok/s floor. The measured
   profiler windows contain zero recaptures and only 5.00 ms host residual.
4. **N2 — Device accept/commit:** consume target top-1 on device and produce the
   accepted count, commit rows, reseed row, recurrent/KV transaction, and output
   summary without intermediate host reads.
5. **N3 — Complete GGUF cycle adapter (landed 2026-07-19):** one
   scheduler-facing GGUF call now owns strict device-chained proposal/NextN,
   the N2 target verify/accept/selected-state transaction, MTP-KV rollback and
   accepted-row repair, reseed, and target/MTP cursor accounting. The public
   single-request MTP path uses it when the registered B1/B2 target graph admits
   the shape, with the prior exact loop retained as the unsupported-shape/
   backend fallback. The clean committed W7900 gate matches N2 for all 240 IDs /
   96 cycles at **118.592 tok/s / 1.2858x true AR**, versus clean N2 **117.557
   tok/s** (+0.88%, aggregate-neutral); cycle wall is **8.497 ms/output** versus
   N2's 8.529 ms. It preserves separate proposal, target, MTP-KV, complete-call,
   and cycle-wall measurements. Proposal leaves still submit through the
   retained Python device-chain implementation; moving those submissions into
   the native launcher remains the next launch-collapse step rather than being
   implied by this ownership milestone.
6. **N3P — Reusable NextN proposal graph (retained 2026-07-19):** strict
   gfx1100 B1/B2 proposal now stages the changing hidden seed, root embedding,
   RoPE, position/context, and K/V row indices into fixed runner buffers and
   replays the existing device chain through one proposal-only NativeSpecCycle
   graph submission. Runner-owned 1,023-row FP32 draft K/V keeps capture
   addresses stable across requests, and independent B1/B2 graph buckets
   coexist. The N3 scheduler-facing adapter therefore owns two native graph
   submissions per cycle—one proposal and one target—rather than dispatching
   proposal leaves from Python. It is not yet one combined native submission.
   A same-source W7900 full-suite pair is exact for all 240 IDs / 96 cycles and
   aggregate-neutral: N3P **117.589 tok/s / 8.653 ms-output** versus N3
   **116.793 / 8.634**. Excluding the one B1+B2 capture (**14.477 ms total**),
   proposal wall improves **0.964 -> 0.953 ms/output**. Matched cached eight-cycle
   tracing replaces 542 `hipLaunchKernel` and 80 synchronous `hipMemcpy` host
   calls with eight proposal `hipGraphLaunch` calls while preserving all 22 IDs.
   The clean detached confirmation at `2395ad33` is **118.183 tok/s / 1.2820x
   true AR / 8.610 ms-output**, matches clean N3 across 97 non-timing fields for
   every cycle, and remains diagnostic because N1 is faster.
7. **N4 — Shared provider adapters (first gfx1100 target slice landed
   diagnostic 2026-07-19):** the registered `w4_paro/native_v1_target_graph`
   adapter lets both PARO MTP and DFlash submit their existing single-request
   B1/B2/B3/B4/B5/B8 chain target+accept graph through NativeSpecCycle ABI v1.
   The control binds real verify-chain `KVLiveSpans`, fixed INT32 metadata and
   accept buffers, and either resident FP16 verifier rows or BF16 sidecar hidden
   taps. It accurately declares only `VERIFY|ACCEPT`; provider linear/KV/hidden
   commit remains the unchanged exact path. Graph-off, tree, inactive, unsupported,
   pre-launch failure, and unregistered-backend cases retain direct fallback.
   Admission is explicit/default-off via
   `HIPENGINE_PARO_NATIVE_SPEC_TARGET_GRAPH=1`. A clean detached `7bf3439e`
   B3 pair matches all 265 compared non-timing/non-route leaves while recording
   four native `VERIFY|ACCEPT` replays, but both arms inherit the available
   packed artifact's token-2 true-AR mismatch and zero acceptance. The blocked
   packet is
   [`benchmarks/results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-blocked.json`](../benchmarks/results/2026-07-19-w7900-paro-mtp-native-target-graph-n4-blocked.json);
   complete PARO/DFlash cycle ownership and independent gfx1100/gfx1151
   promotion gates remain open.
8. **N5 — Multi-cycle option:** only after N3/N4 are exact, allow the native
   launcher to continue until EOS, cancellation/deadline, output-buffer limit,
   or an explicit scheduler yield point.

### Promotion Gates

- Full multi-prompt category suite plus heldouts; no single-prompt tuning.
- Exact/default token, hidden, all Conv/GDN state, and all live K/V equality.
  Accuracy-traded routes retain their explicit semantic labels and compare
  against their own current outputs.
- Standard new-kernel gate: KL `<= 0.05`, top-1 `>= 90%`, plus a profiler trace
  proving the expected registered kernels ran.
- gfx1100 `llama-compat` break-even is closed: the conservative clean run is
  **122.67 tok/s / 8.186 ms-output**, **1.2679x** its 96.75 tok/s graph AR and
  **6.26% above** the refreshed 115.44 tok/s llama.cpp floor. Exact/default
  still requires its own same-protocol break-even.
- Preserve the cross-engine closure on the complete category+heldout suite:
  both clean repetitions must remain at least **115.44 transition-normalized
  tok/s / at most 8.662 ms-output** without reducing acceptance or changing the
  explicit accuracy-traded route contract.
- gfx1151 must be non-regressive on its full suite even if host submission is a
  smaller fraction there. A gfx1100 win does not transfer automatically.
- Retain as default only when exact/non-regressive for the provider/backend;
  otherwise keep the native launcher as an explicit diagnostic and preserve the
  fallback.

This work supersedes a gfx1100-only launch-collapse plan. The implementation is
shared infrastructure; performance evidence and defaults remain backend- and
provider-specific.

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
is committed at `benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json`
and now encodes the llama.cpp draft sampler contract (`top_k=10`, greedy top-1
from the top-k set). Matching B2-B4 deterministic fixtures are also committed as
`benchmarks/fixtures/gguf_mtp_b{2,3,4}_sampling_greedy_seed12345.json`, so the
preflight child can pick a budget-matched fixture for `--draft-max {1,2,3,4}` by
default or emit a consolidated B1-B4 blocked matrix with `--all-budgets`. The
matrix includes a compact `readiness_by_budget` section for precheck booleans,
missing native keys, exactness status, KVLiveSpans paged-cache smoke status,
llama.cpp trace budget coverage, metrics-contract status, per-budget
`parity_precheck_by_budget` token/sampler parity evidence, per-budget
`draft_budget_precheck_by_budget` and
`draft_sampling_contract_precheck_by_budget` budget/top-k contract evidence,
per-budget `hidden_seed_contract_precheck_by_budget` seed-contract evidence,
per-budget
`execution_by_budget` exactness/next-action stubs, per-budget
`oracle_gate_by_budget` KL/top-1 plus KVLiveSpans oracle evidence, per-budget
`llamacpp_trace_oracle_by_budget` denominator/trace evidence, a matrix-level
`target_context_contract`, self-validating per-budget
`runtime_kernel_precheck_by_budget` payloads, per-budget
`hipengine_metrics_contract_by_budget`, per-budget
`hipengine_metrics_contract_validation_by_budget`, a compact
`hipengine_metrics_contract_validation_summary`, and blocker codes without
requiring reviewers to inspect each full child artifact. It also
includes matrix-level `kvlivespans_paged_cache_smoke_by_budget`,
`kvlivespans_paged_cache_max_abs_diff_by_budget`,
`all_kvlivespans_paged_cache_smokes_pass`,
`llamacpp_trace_budget_coverage_by_budget`,
`partial_llamacpp_trace_budget_budgets`, and
`all_llamacpp_trace_budgets_full` rollups, plus accepted/draft and
accepted/output denominator comparability rollups
(`accepted_per_draft_status_by_budget`,
`noncomparable_accepted_per_draft_budgets`,
`all_accepted_per_draft_metrics_comparable`,
`accepted_per_output_status_by_budget`,
`noncomparable_accepted_per_output_budgets`, and
`all_accepted_per_output_metrics_comparable`); add `--compact-matrix` to omit
full child artifacts for compact benchmark evidence. Use
`--fail-on-partial-trace-budget` to return exit code `3` when a single-budget or
matrix artifact does not exercise the requested draft budget, preventing partial
B2-B4 debug-trace provenance from being mistaken for full-budget parity evidence.
Each single-budget or matrix artifact embeds `cli_gate_exit_codes` with the
stable CLI gate names and exit codes plus `cli_gate_failures` with the currently
failing gate names and `cli_gate_failure_exit_codes` for the current failure-name
to exit-code subset; matrix artifacts also include `cli_gate_failures_by_budget`
and `cli_gate_failure_exit_codes_by_budget` for compact per-budget diagnostics.
Use `--fail-on-precheck-fail` to return
exit code `11` when token/sampling/budget/hidden-seed prechecks fail. Use
`--fail-on-exactness-fail` to return exit code `10` when the CPU-reference or
llama.cpp trace exactness gate fails.
Use `--fail-on-kvlivespans-smoke-fail` to return exit code `9` when the
CPU-reference dense-vs-paged KVLiveSpans smoke fails. Use
`--fail-on-noncomparable-accepted-output` to return exit code `4`
when the trace artifact lacks visible output-token counts and therefore cannot support
`accepted_per_output` comparisons. Use
`--fail-on-noncomparable-accepted-draft` to return exit code `6` when the trace
cannot support `accepted_per_draft` comparisons. Use
`--fail-on-native-runtime-missing` to return exit code `7` when the native NextN
or KVLiveSpans runtime component keys are absent. Use
`--fail-on-optimization-missing` to return exit code `8` when optimization keys
(such as device-side draft top-k) are absent. Matrix artifacts also carry an M6
performance-comparison readiness rollup (`performance_readiness_contract`,
`performance_readiness_by_budget`, `performance_comparison_ready_by_budget`,
`performance_comparison_blockers_by_budget`, `performance_unready_budgets`, and
`all_performance_comparisons_ready`). The blocker derivation is shared through
the torch-free, self-validating `Qwen35GGUFMTPPerformanceReadiness`
speculative contract whose required fields include its own validator metadata;
`--fail-on-performance-unready` returns exit code `5`
until parity, exactness, KVLiveSpans paged-cache smoke, full trace-budget
coverage, comparable accepted/draft and accepted/output denominators, native
runtime kernels,
optimization kernels, and hipEngine metrics are all present.
`--fail-on-metrics-contract-invalid` returns exit code `12` if the blocked or
native hipEngine metrics contract validation fails.
Parity
Preconditions (a) and (b) have fixture coverage; M5 still also requires the
numeric KL/top-1 gate and actual GGUF MTP execution.

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
- `cpu_reference_kernel == ["cpu_reference", "mtp_nextn_layer", "w4_gguf",
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
  `KernelKey("cpu_reference", "mtp_nextn_layer", "w4_gguf", "qwen35_dense_logits")`
  with a legacy `gguf_moe` alias for older fixtures/tests.
  Fixture `benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json`
  pins a deterministic hidden/token row, finite logits, and top-k IDs for the
  full `eh_proj -> attention -> MoE/shared expert -> shared head` oracle.
  The attention sublayer and full NextN logits path now have both the older
  dense CPU-cache path and a NumPy-only KVLiveSpans-shaped paged-cache path using
  `(kv_base_offsets, kv_live_counts, kv_token_positions, kv_evict_mask)`, so the
  M4 ABI can be exercised before HIP kernels exist. The metadata-only draft call
  spec emitted by `scripts/gguf_mtp_call_spec.py` now advertises those
  KVLiveSpans dynamic inputs plus `block_size` in addition to the dense-cache
  placeholders. The metadata-only draft tensor plan/call spec also advertises
  the draft-token selection contract used for llama.cpp parity: greedy top-1
  from `top_k=10` candidates via the registered
  `KernelKey("cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h")`
  fallback until a backend `topk_device` variant exists.
  `scripts/gguf_mtp_oracle_gate.py` turns that fixture into a
  reusable mechanical artifact with KL/top-1 metrics before any performance
  comparison is allowed, uses the registered
  `KernelKey("cpu_reference", "mtp_draft_topk", "w4_gguf", "full_vocab_d2h")`
  unfused top-k fallback/oracle for draft token selection, embeds the
  metadata-only `draft_execution_plan` contract that ties selected token rows to
  KVLiveSpans-shaped append/decode kwargs, and now runs a dense-vs-paged
  KVLiveSpans cache smoke (`kvlivespans_paged_cache_smoke`) against the same
  fixture.
- llama.cpp verbose `draft-mtp` candidate fixture
  `benchmarks/fixtures/llamacpp_mtp_explain_concept_draft_trace.json` captures a
  short `explain_concept` prompt trace: 2 draft calls, top-3 candidates per call,
  and 0/2 accepted drafts. It was captured with backend draft sampling disabled
  only to expose candidate probabilities in the log; it is an oracle/debug
  fixture, not a performance benchmark. The B1 preflight child validates and
  embeds a compact `llamacpp_trace_oracle` summary from this fixture, including
  draft-denominator checks for `draft_n` / `draft_n_accepted`, trace budget
  coverage (`full_requested_budget_exercised` vs
  `partial_trace_did_not_exercise_full_budget`), and an explicit
  `accepted_per_output` status of not-comparable until visible output tokens are
  available, so blocked artifacts carry both CPU-reference and captured llama.cpp
  oracle provenance.

Acceptance:

- Fixed hidden/token fixture produces deterministic finite logits.
- Draft top-k agrees with the captured llama.cpp trace (M0) **or** the
  `cpu_reference` NextN forward within the gate: `KL <= 0.05` AND top-1 agreement
  `>= 90%` vs `kernels/cpu_reference/`. M3 blocks on at least one of these
  oracles actually existing. Use `scripts/gguf_mtp_oracle_gate.py --fail-on-fail`
  for the committed CPU-reference fixture gate.
- No full target trunk re-execution inside the MTP draft-only path.

### M4 — Target-Attached MTP Context

Current status:

- `hipengine.speculative.gguf_mtp.Qwen35GGUFMTPContext` is a target-attached
  scaffold for the GGUF path (also exported through the public
  `hipengine.speculative` package boundary consumed by the preflight/oracle
  scripts). It references the target resident session, records
  ready fp32 post-`output_norm` seed rows, emits a self-validating context
  snapshot for pending/verify seed-state artifacts, applies the llama.cpp
  `verify_h[min(n_accepted, n_rows - 1)]` accept/reseed rule, exports these GGUF
  MTP contracts through `hipengine.speculative`, builds B1 draft rows carrying
  both token IDs and embedding-seed pointers, can build B2-B4 draft
  batches when supplied one explicit seed row per proposed token, and has a
  torch-free proposal bridge that resolves the registered draft top-k kernel and
  converts runtime logits into selected draft rows plus top-k evidence. It can
  also emit a metadata-only uniform KVLiveSpans plan for the future single-NextN
  append/decode cache from a draft batch. The KVLiveSpans payload now advertises
  and validates its ABI contract (`base_offsets`, append/decode live counts,
  token positions, optional evict mask, dtype/mode, and block shape) before it is
  bundled with the proposal into a single draft execution-plan contract carrying
  CPU-reference-shaped append/decode kwargs. The execution-plan payload now also
  validates the nested proposal/KVLiveSpans contracts, top-level proposed-token
  summary, proposal/KV token-position alignment, and CPU-reference kwargs before
  future native NextN attention/KV integration consumes it. The context can
  project GGUF-specific draft rows into the shared
  `DraftBatch`/`TargetVerifyBatch` verifier ABI while deriving root rows from
  depth-1 GGUF parent token/position metadata, keeping embedding-seed pointers
  on GGUF rows, producing a shared `TargetAcceptSummary` CPU oracle from target
  top-1 rows, deriving scheduler-facing `KVTransaction` metadata for target
  verification, validating shared `TargetCommitPlan` metadata from that summary
  or directly from target top-1 rows, applying the summary, validated commit
  plan, or serializable direct top-1 accept-step result back to GGUF hidden-seed
  reseed state, aggregating individual accept steps or per-cycle top-1 specs
  into the `accepted_per_draft` and `accepted_per_output` denominator contract,
  and scoring proposed draft tokens against target tokens while applying the
  llama.cpp verify-row reseed rule. It does not allocate MTP KV buffers or run
  NextN draft kernels yet.

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
  `Qwen35GGUFMTPDraftBatch.to_shared_draft_batch()` now bridges candidate token
  topology into that shared ABI, `to_target_verify_batch()` derives root rows
  from the depth-1 GGUF parent token/position metadata,
  `Qwen35GGUFMTPDraftExecutionPlan.target_accept_summary_from_top1()` builds the
  shared accept-summary oracle from target top-1 rows,
  `target_verify_transaction()` derives speculative `KVTransaction` metadata
  from the same GGUF-derived target verifier batch,
  `target_commit_plan_from_summary()` validates commit rows against the
  GGUF-derived `TargetVerifyBatch` before building the shared `TargetCommitPlan`,
  `target_commit_plan_from_top1()` ties the derived transaction, accept summary,
  and commit-plan validation together for CPU-oracle tests, and
  `Qwen35GGUFMTPContext.accept_target_summary()` /
  `accept_target_commit_plan()` / `accept_target_top1()` apply the accepted
  count back to the llama.cpp verify-row hidden-seed reseed rule;
  `accept_target_top1()` returns a serializable `Qwen35GGUFMTPAcceptStep` while
  still supporting tuple unpacking as `(commit_plan, reseed)`, and
  `Qwen35GGUFMTPTop1AcceptSpec` now exposes a self-validating per-cycle payload
  contract for nested execution plans, target top-1 rows, transaction IDs, and
  verify-seed rows;
  `Qwen35GGUFMTPContext.accept_target_top1_metrics()` feeds those per-cycle target
  top-1 rows and verify seeds into one metrics artifact; each serialized
  `Qwen35GGUFMTPAcceptStep` now validates its commit-plan fields, nested
  seed-row contract, and reseed row before aggregation; and
  `Qwen35GGUFMTPAcceptStepMetrics` aggregates those serializable steps with a
  self-identifying schema/kind/source payload, explicit validator hook, a
  centralized required-field list with internal-consistency payload validation,
  inferred `B{candidate_budget}`
  label, compact per-step transaction/candidate/accepted rows, and the same
  accepted/draft and accepted/output denominator labels used for llama.cpp
  parity; the GGUF draft-row objects still carry the extra embedding seed
  pointer until native MTP runtime buffers exist and now expose their own
  required-field/validator contract. Draft-batch payloads also advertise and
  validate that nested row contract, row count, and request-id summary. Draft
  proposal payloads now advertise/validate their nested draft-batch contract,
  top-k kernel key, selected-token rows, and proposed-token summary so serialized
  B1-B4 proposals can reject malformed embedding-seed/top-k metadata before
  native NextN execution is wired in.

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
  token/sampling parity, enforces that the requested `--draft-max` B1-B4 budget
  and the llama.cpp draft sampler contract (`top_k=10`, greedy top-1 from top-k)
  match both engines' sampling fixtures, embeds the MTP draft tensor/call specs
  (including KVLiveSpans dynamic inputs), records a `hidden_seed_contract_precheck`
  that pins the required fp32 post-`output_norm` seed dtype/provenance and call-spec
  shape, records an exact four-axis `runtime_kernel_precheck` backed by the shared
  `Qwen35GGUFMTPRuntimeKernelPlan` for required CPU-reference oracles and missing
  native runtime/optimization keys, runs the CPU-reference oracle exactness gate plus
  the captured llama.cpp draft-trace oracle summary, and emits a blocked artifact
  until native GGUF MTP draft execution is implemented.
- Run matched prompt/token suite against:
  - hipEngine GGUF AR;
  - hipEngine GGUF MTP B1-B4;
  - llama.cpp HIP B1-B4;
  - llama.cpp Vulkan B1-B4.

Acceptance:

- **Parity Preconditions (a)/(b)/(c) pass per-prompt before any accepted/output
  number is compared.**
- hipEngine B1-B4 exactness and accepted/output are reported per prompt.
  `scripts/llamacpp_mtp_bench.py` now emits llama.cpp `accepted_per_output`
  from `draft_n_accepted / predicted_n` alongside draft acceptance so the
  denominator is explicit in external comparison artifacts.
- If hipEngine accepted/output lags llama.cpp by more than ~10% relative on the
  same budget, stop performance tuning and debug draft logits/model identity.

### M6 — Runtime/Kernel Optimization

Only after M5 acceptance parity:

- Make the cross-backend `NativeSpecCycleLauncher` described in the 2026-07-12
  section the primary orchestration workstream. Start with a native target-block
  bucket, then move accept/commit metadata and the complete cycle device-side;
  reuse the same ABI for GGUF MTP, PARO MTP, and DFlash.
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
      registered under `(cpu_reference, mtp_nextn_layer, w4_gguf,
      qwen35_dense_logits)` (plus legacy `gguf_moe` alias) as the offline oracle,
      with a deterministic fixture:
      `benchmarks/fixtures/qwen35_gguf_mtp_nextn_cpu_reference_fixture.json`.
      The oracle gate now embeds the metadata-only draft execution-plan summary
      derived from those exact logits and a dense-vs-KVLiveSpans paged-cache
      smoke over the same fixture.
- [ ] Implement draft-only NextN forward (full attn+MoE) with a KVLiveSpans
      attention path and dense fallback; CPU-reference coverage and call specs
      now include dense and KVLiveSpans-shaped paged-cache paths through the full
      NextN logits oracle, and `Qwen35GGUFMTPContext` covers the B1-B4
      seed/batch/proposal/verification state scaffold plus metadata-only
      KVLiveSpans (including an explicit payload validator) and self-validating
      execution-plan contracts exported through `hipengine.speculative`;
      `Qwen35GGUFMTPRuntimeKernelPlan` now enumerates and validates
      the missing native composite NextN runtime key plus the KVLiveSpans-shaped
      `paged_kv_write/mixed_bf16_spans` append and
      `paged_attn_decode/bf16_context_spans` decode keys under
      `quant='w4_gguf'`; each nested registry-check row now carries and validates
      its own required-field/validator metadata, but HIP/runtime registration for
      those keys remains open.
- [x] Add hipEngine GGUF MTP B1 prompt-suite runner (new GGUF child, not a wrapper
      flag): `scripts/gguf_mtp_b1_prompt_suite.py` currently implements
      preflight + blocked-artifact emission with embedded MTP draft tensor/call
      specs, hidden-seed dtype/provenance precheck, exported target-context
      snapshot contract, exact runtime-kernel registry
      precheck backed by a self-validating `Qwen35GGUFMTPRuntimeKernelPlan`
      payload (including self-describing nested registry checks, the native NextN
      composite key, and the KVLiveSpans paged-KV append/decode component keys),
      CPU-reference oracle
      gate output, a hipEngine metrics contract with schema/kind/source labels,
      B-budget fields, compact per-step fields, full serialized `steps`, explicit
      accepted-per-draft/output denominator labels, and captured
      llama.cpp draft-trace oracle provenance/denominator checks; B1 runtime
      execution remains in the next backlog row.
- [x] Wire `scripts/gguf_mtp_bench.py` fixed-prompt native diagnostic for
      `--draft-n-max 1..5`. This is a local chained-draft acceptance smoke, not
      the prompt-suite persistent GGUF MTP context; artifacts label
      `mtp_context_mode` so B3-B5 diagnostic rows are not mistaken for retained
      performance parity. B5 prompt-suite preflight fixtures/oracles remain a
      separate follow-up if we decide to compare llama.cpp `--spec-draft-n-max 5`.
- [x] Gate Parity Preconditions (token-id + sampling parity) before comparison;
      `scripts/gguf_mtp_parity_precheck.py` now provides the fail-fast gate,
      `benchmarks/fixtures/gguf_mtp_b1_sampling_greedy_seed12345.json` pins the
      B1 deterministic sampling settings and llama.cpp draft sampler contract,
      and the committed hipEngine/llama.cpp D32 token fixtures match on all 9
      prompts.
- [ ] Run B1 exactness and accepted/output parity against llama.cpp B1. The
      target-attached context now has torch-free proposal verification and
      aggregate denominator contracts for accepted-count/reseed accounting, and
      the preflight child records `draft_budget_precheck`,
      `draft_sampling_contract_precheck`, and `hidden_seed_contract_precheck`
      sections so a requested B1-B4 artifact cannot silently reuse mismatched
      budget, stale draft `top_k=1` sampling, or a non-fp32/non-post-`output_norm`
      seed contract. It also exposes the self-validating `target_context_contract`
      that future pending/verify hidden-seed snapshots must satisfy. The
      llama.cpp comparison helper now reports
      `accepted_per_output` with explicit `draft_n_accepted / predicted_n`
      denominator for future parity rows, while the preflight artifact exposes
      the hipEngine-side `Qwen35GGUFMTPAcceptStepMetrics` schema/kind/source,
      `candidate_budget` / `budget_label`, compact per-step row fields, full
      serialized `steps`, validator hook, and denominator contract from the
      centralized `Qwen35GGUFMTPAcceptStepMetrics.blocked_contract()` /
      `validate_blocked_contract()` / `required_fields()` helpers; prompt-suite
      artifacts self-validate the blocked contract at construction time, and
      native runtime artifacts can be checked with
      `Qwen35GGUFMTPAcceptStepMetrics.validate_payload()`
      for labels, the advertised required-field list, validator hook, advertised
      nested accept-step contract, nested accept-step payload validation, compact
      step rows, full serialized step cross-checks, advertised nested seed-row
      contract, reseed-row metadata, aggregate counts, visible-output bounds, and
      ratio consistency before parity comparison. The draft-token verification
      aggregate (`Qwen35GGUFMTPVerificationMetrics`) now carries the same
      schema/kind/source, `candidate_budget` / `budget_label`, required-field,
      validator, nested verification-result contract, nested seed-row contract,
      token-prefix, mismatch-metadata, reseed-row metadata, and ratio-consistency
      contract so B1-B4 parity rows can reject malformed hipEngine payloads
      before llama.cpp comparison.
- [ ] Extend to B2-B4 after B1 is exact. The preflight child accepts
      `--draft-max {1,2,3,4}` for budget-aware blocked artifacts, selects the
      matching `gguf_mtp_bN_sampling_greedy_seed12345.json` fixture by default,
      and `--all-budgets` emits a single B1-B4 matrix artifact with compact
      `readiness_by_budget` parity/preflight/native-key status, per-budget
      `parity_precheck_by_budget` token/sampler parity evidence, per-budget
      `draft_budget_precheck_by_budget` and
      `draft_sampling_contract_precheck_by_budget` budget/top-k contract evidence,
      per-budget `hidden_seed_contract_precheck_by_budget` seed-contract evidence,
      per-budget `execution_by_budget` exactness/next-action stubs, per-budget
      `oracle_gate_by_budget` KL/top-1 plus KVLiveSpans oracle
      evidence, per-budget `llamacpp_trace_oracle_by_budget` denominator/trace
      evidence, a matrix-level `target_context_contract`, self-validating
      `runtime_kernel_precheck_by_budget` payloads, current CLI
      failure-to-exit-code maps, plus per-budget and matrix-level KVLiveSpans
      paged-cache smoke status, llama.cpp trace budget coverage, and
      accepted/draft plus accepted/output denominator comparability
      (`--compact-matrix` omits
      full child artifacts for compact evidence, and
      `--fail-on-partial-trace-budget` exits `3` for B2-B4 partial-coverage
      trace provenance, `--fail-on-precheck-fail` exits `11` when token,
      sampling, budget, or hidden-seed prechecks fail,
      `--fail-on-exactness-fail` exits `10` when the CPU-reference
      or llama.cpp trace exactness gate fails,
      `--fail-on-kvlivespans-smoke-fail` exits `9` when the CPU-reference
      dense-vs-paged KVLiveSpans smoke fails,
      `--fail-on-noncomparable-accepted-output` exits `4` when
      accepted/output denominators are not comparable,
      `--fail-on-noncomparable-accepted-draft` exits `6` when accepted/draft
      denominators are not comparable,
      `--fail-on-native-runtime-missing` exits `7` when native NextN/KVLiveSpans
      runtime keys are absent, `--fail-on-optimization-missing` exits `8` when
      device-side optimization keys are absent,
      `--fail-on-metrics-contract-invalid` exits `12` when hipEngine metrics
      contract validation fails, and `--fail-on-performance-unready`
      exits `5` until the combined, self-validating M6 readiness
      rollup from `Qwen35GGUFMTPPerformanceReadiness` is complete (including the
      exported `performance_readiness_contract`, per-budget
      `performance_readiness_by_budget` payloads, and their own
      required-field/validator metadata) and clean, including
      accepted/draft comparability, the KVLiveSpans smoke, and
      optimization-kernel readiness bits); actual B2-B4
      execution/parity still waits on native draft execution.
- [x] Add backend-side top-k draft sampling as a `topk_device` variant, keeping
      `full_vocab_d2h` registered as the unfused fallback/oracle. The CPU
      `full_vocab_d2h` fallback/oracle is registered and advertised in MTP
      draft tensor plans/call specs with the llama.cpp parity contract
      (`top_k=10`, greedy top-1). The `topk_device` optimization key now
      resolves through the four-axis registry on gfx1100/gfx1151 to the native
      bounded top-k sampler wrapper; runtime integration still waits on native
      NextN execution.
- [x] Define and validate the provider-neutral native speculative cycle
      control/result ABI, raw-pointer ownership, shape buckets, and exact
      Python fallback contract (N0). Landed 2026-07-13 with a **496-byte**
      control block, **64-byte** result, C-header/ctypes field-order guard,
      exact `TargetVerifyBuffers` + `KVLiveSpans` adapter, and CPU/fake-launcher
      oracle tests. This is contract infrastructure only: no native math is
      enabled and no performance result is claimed.
- [x] Implement the N1 native C++ target-block launcher for one GGUF B2 bucket.
      The one-shot recapture version remains the rejected ownership control.
- [x] Reuse fixed-address B1/B2 target graphs with live device metadata (N1R).
      The clean W7900 full-suite gate is **123.332/122.667 tok/s**, byte-exact
      against all prior llama-compat IDs/cycles, and the slower run is **6.26%**
      above llama.cpp. The real-model oracle covers repeated B2 plus B1 hidden,
      Conv/GDN, K/V, and cursor state; cached rocprof sees zero measured capture
      plus the dynamic metadata/cursor/widening leaves.
- [x] Implement N2 device-resident accept/commit summary for the admitted GGUF
      B1/B2 bucket. The clean W7900 full-suite packet preserves every 240-ID / 96-cycle
      record at **117.557 tok/s / 8.529 ms-output**.
- [x] Implement N3 complete single-request GGUF cycle ownership, including
      proposal invocation, N2 target transaction, MTP-KV rollback/repair,
      reseed, and cursor accounting. Clean W7900 N3 is exact at **118.592
      tok/s / 1.2858x true AR**; proposal leaves remained Python-submitted.
- [x] Collapse strict B1/B2 NextN proposal host submissions through reusable
      proposal-only NativeSpecCycle graphs (N3P). The full-suite semantic gate,
      proposal K/V byte oracle, aggregate-neutral same-source pair, and cached
      host-submission trace pass. Keep N1/N2/N3 and unsupported-shape fallbacks.
- [x] Add the first shared gfx1100 PARO MTP/DFlash `VERIFY|ACCEPT` target-graph
      adapter through the provider-neutral NativeSpecCycle ABI, retaining
      provider commit and exact unsupported-shape fallback (N4 target slice).
- [ ] Extend N4 through complete PARO MTP and DFlash proposal/commit ownership;
      validate gfx1100 and gfx1151 independently before any default promotion.
- [x] Profile N1 and reusable N1R after cached build warmup. The retained N1R
      windows average **18.671 ms host / 13.670 ms kernels / 5.001 ms residual**
      across six graph replays, with zero capture charged in every measured
      step. Each replay contains one dynamic-metadata unpack, three cursor
      advances, and two top-1 widening leaves, all zero-scratch. Repeat the
      complete-cycle attribution after N3.

## Decision Log

- 2026-07-19: Retained N3P reusable gfx1100 B1/B2 NextN proposal graphs as a
  submission-ownership milestone, not a new topline. A five-cycle mixed
  accept/reject/B1 oracle matches N3 candidate IDs and every committed/speculative
  FP32 K/V prefix hash. The full category+heldout gate preserves all **240 IDs /
  96 cycles**, **80.45% draft acceptance**, and **60.00% accepted-output**. A
  same-source pair measures N3P **117.589 tok/s / 8.653 ms-output / 1.2743x AR**
  versus N3 **116.793 / 8.634 / 1.2691x**; aggregate wall is neutral, while
  capture-excluded proposal wall improves **0.964 -> 0.953 ms/output**. For the
  same eight cycles and 22 IDs, cached HIP API tracing changes
  `hipLaunchKernel` **3273 -> 2731**, synchronous `hipMemcpy` **1204 -> 1124**,
  and `hipGraphLaunch` **8 -> 16**. A detached clean `2395ad33` publication is
  **118.183 tok/s / 1.2820x AR / 8.610 ms-output** and matches all 97 common
  non-timing fields across 96 cycles. N1 remains the **122.667 tok/s** canonical
  llama-compat row. N3P retains exact N3/unsupported-shape fallback and does not
  claim one combined proposal+target native submission or gfx1151 admission.
- 2026-07-19: Retained reusable gfx1100 B1/B2 native target graphs at clean
  `0d7b86e7`. Two full category+heldout processes measure **123.33/122.67
  tok/s** versus **96.91/96.75 true AR**, with **8.143/8.186 ms/output** and
  unchanged **80.45% draft acceptance / 60.00% accepted-output**. Both preserve
  every prior 240-ID/96-cycle trajectory; the real 35B repeated-B2+B1 target,
  hidden, 60 Conv/GDN-state, 20 K/V-buffer, and cursor oracle is byte-exact.
  The conservative run is **6.26% above** llama.cpp's 115.44 tok/s floor. A
  cached six-step trace records zero measured captures and cuts profiler host
  residual from **38.41 to 5.00 ms/step**. N2 device accept/commit is now the
  next ownership milestone; exact/default and gfx1151 remain unchanged.
  Artifact:
  `benchmarks/results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json`.
- 2026-07-19: Established the clean current W7900 `llama-compat` baseline at
  `637be21d`: graph AR **92.26 tok/s**, B2 **54.88 tok/s (0.5948x)**,
  80.45% draft acceptance, and 18.259 ms/output. Target verification is
  **41.319/45.649 ms per cycle**. A cached rocprof trace measures **52.42 ms
  host / 14.01 ms kernels / 38.41 ms residual** and **977 launches/step**;
  accept/commit is below 0.22 ms/output. A same-environment child comparison
  makes current source faster than the July source (43.22 vs 48.20 ms), so the
  old 79.70 tok/s difference is not attributed to a source regression. N2 is
  useful state ownership but cannot close parity alone; reusable N3 submission
  is required. Artifact:
  `benchmarks/results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json`.
- 2026-07-19: Refreshed the W7900 llama.cpp B2 natural25 external floor on all
  ten category prompts at clean hipEngine `8d67f072`: transition-normalized
  **78.05 AR -> 115.44 MTP tok/s (1.4791x)**, 81.56% draft acceptance, and
  58.40% accepted/output. The prior 116.88 tok/s row is within 1.23%.
  hipEngine `llama-compat` remains 79.70 tok/s and therefore needs **+44.85%**
  to meet the current floor. Artifact:
  `benchmarks/results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json`.
- 2026-07-19: Landed N1 as a correctness-accepted, performance-rejected gfx1100
  diagnostic under registry key
  `(hip_gfx1100, speculative_cycle, w4_gguf,
  native_v1_b2_target_graph)`. One host-only C++ call launches and synchronizes
  the captured three-row target graph; unsupported backends/configurations fall
  back before capture, while launch-time failures never silently re-execute a
  potentially mutating verifier. W7900 target/state/KV parity is byte-exact.
  Collapsed submission cuts target forward host wall **47.64%**, but the
  required position-bound recapture costs **32.755 ms/cycle** and regresses the
  same-tree three-cycle diagnostic **84.35 -> 52.48 tok/s (-37.78%)** with
  identical 6/6 acceptance. Keep `--native-spec-target-cycle` default-off;
  proceed through N2, then make position/cursor metadata dynamic and reuse the
  complete-cycle graph/launcher in N3. Artifact:
  `benchmarks/results/2026-07-19-gfx1100-native-spec-cycle-n1-b2.json`.
- 2026-07-13: Landed N0 as provider-neutral contract infrastructure in
  `hipengine.speculative.native_cycle`. Version 1 carries explicit live counts
  and capacities, chain/tree and stage masks, metadata/hidden/KV dtypes,
  `KVLiveSpans`, proposal/verifier/accept/commit/cursor pointers, cancellation
  and deadline fields, and terminal/yield status. All pointers are borrowed;
  launchers may mutate only output/state destinations and never retain/free
  caller storage. The C header and ctypes mirrors are **496/64 bytes** and are
  field-order tested. Selected N1 as the existing single-request B2
  (`rows=3`) GGUF target verifier, leaving proposal and commit on the exact
  Python fallback.
- 2026-07-12: W7900 graph AR corrected the production denominator to
  `93.30 tok/s`; hipEngine `llama-compat` MTP is `79.70 tok/s (0.8542x)`, while
  transition-normalized llama.cpp on the same W7900 reaches
  `78.25 -> 116.88 tok/s (1.4936x)` with slightly lower accepted/output. This
  disproves an insufficient-compute explanation and identifies native cycle
  launch/orchestration as the portability gap. Chose a provider-neutral C/C++
  speculative cycle launcher with device-resident accept/commit metadata for
  GGUF MTP, PARO MTP, DFlash, gfx1100, and gfx1151 rather than a gfx1100-only
  workaround.
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
