# Qwen3.8-27B `UD-Q4_K_M` gfx11 campaign

- **Status:** planned; implementation and binding baselines not started
- **Created:** 2026-08-28
- **Target:** exact Unsloth Qwen3.8-27B `UD-Q4_K_M` GGUF
- **Backends:** Radeon Pro W7900 / `hip_gfx1100` and Radeon 8060S / `hip_gfx1151`
- **Execution profile:** strict first; production only after an independent profile gate
- **Normative dependencies:** [`PLAN.md`](PLAN.md), [`KERNELS.md`](KERNELS.md),
  [`TESTING.md`](TESTING.md), [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md),
  and [`BENCHMARK.md`](BENCHMARK.md)
- **Opening evidence:**
  [`QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md`](QWEN38-STRIX-HALO-EXTERNAL-SURVEY.md)

## September 6 Verified Scope Update

Read [`UD-QUANTS.md`](UD-QUANTS.md) before execution. It supplies the corrected
AR/NextN audit, reproducible allocation accounting, current upstream kernel
review, and the additional K_S campaign. The artifact key, single-owner rules,
quality requirements and binding K_M performance charter in this document
remain in force; use `KM-U0..U8` for the tasks below and `UD-U0..U7` for the
cross-file dependency map.

The 18 refused tensors are only part of the work: all 117 IQ4_XS tensors still
expand to BF16 on the dense route, and raw-IQ's file-global repack veto expands
many Q5/Q6 roles too. Dense IQ4_XS consumption and qualification of existing
Q5/Q6 routes must precede full-model bring-up. Raw Q5/Q6 and generic selected-IQ
math already exist; new IQ repacking is not a prerequisite.

The corrected current AR plan contains 851 tensors across 64 AR layers,
with block64/15 tensors handled separately by NextN. Hypothetically expanding
all refused AR tensors gives 39.725985 GiB of weight allocations before runtime
memory, not just the 2.220 GiB incremental cost for the refused subset.
See the new document for sidecar/backend policy accounting and its limits.

## 1. Objective and definition of success

This campaign adds native support for the exact Qwen3.8-27B `UD-Q4_K_M`
artifact that currently fails during materialization because it contains dense
`Q3_K`, `IQ4_NL`, and `IQ3_S` tensors. The work is not complete when the file
merely loads. Completion requires all of the following on **each backend as an
independent hardware lane**:

1. The public torch-free `hipengine.LLM.generate()` path loads the exact file and
   completes strict greedy prefill and decode with native compressed weights.
2. Dense `Q3_K`, `IQ4_NL`, and `IQ3_S` have CPU-reference oracles, registered
   strict HIP fallbacks, native decode kernels, and native bulk-prefill kernels.
3. The production materialization plan keeps one physical device payload per
   logical weight. It does not expand the 18 tensors to resident BF16 and does
   not keep a raw-plus-repacked duplicate by default.
4. The exact artifact passes the required strict full-model, lifecycle, graph,
   c>N, and natural-prompt gates.
5. On the same physical host and under the same hipEngine protocol,
   `UD-Q4_K_M` is faster than the currently supported Qwen3.8-27B `Q4_K_M`
   artifact in both prefill and true autoregressive decode at `512/128`,
   `1024/128`, and `4096/128`.
6. On the exact `UD-Q4_K_M` file, hipEngine is faster than the faster valid
   same-host llama.cpp HIP or Vulkan result in both prefill and true
   autoregressive decode at those three shapes.
7. The retained route remains operation-complete for public generation,
   Generation-2 serving, and native MTP target verification. A fast c1-only
   kernel does not qualify the quant preset as supported.

A lane that becomes correct but misses a performance target remains a supported
or diagnostic lane according to its completed gates; it must not be described
as having completed this campaign. A performance loss is a valid measured
outcome. Do not change prompts, timing boundaries, cache policy, KV type, or
correctness criteria to manufacture a win.

### 1.1 Binding comparisons

Keep three comparisons separate:

| Comparison | Purpose | Binding rule |
| --- | --- | --- |
| hipEngine `UD-Q4_K_M` versus hipEngine `Q4_K_M` | Prove that the mixed lower-bit artifact produces a runtime benefit rather than only a smaller file. | Same host, commit, backend, execution profile, KV type, prompt IDs, shapes, graph policy, warmup, timing owner, and measurement count. Quant-quality equality is not implied; each artifact must pass its own declared quality gate. |
| hipEngine versus llama.cpp on `UD-Q4_K_M` | Establish the external speed target. | Exact same GGUF SHA-256, host, GPU, prompt IDs, context/decode shape, sampling, KV type, cache state, and timing boundary. Compare against both HIP and Vulkan; the faster correctness-valid row is binding. |
| hipEngine strict versus hipEngine production | Qualify any changed-arithmetic optimization. | Strict-teacher full logits at identical contexts plus the calibrated production envelope, exact control ownership, deterministic repeats, isolation, BF16-relative evidence where available, and task non-inferiority. |

If hipEngine and llama.cpp do not share a supported KV representation, their
native-default rows are diagnostic rather than binding. Prefer common BF16 K/V.
If llama.cpp cannot execute that contract, add a common F16 K/V hipEngine lane
or record the mismatch explicitly; never silently compare BF16 K/V with F16 K/V
as the campaign-closing row.

### 1.2 Statistical win rule

For each binding speed cell:

- run one discarded warmup and at least five paired, counterbalanced
  measurements per arm;
- use synchronized complete operation wall time, excluding model load, JIT
  compilation, graph capture, and artifact writing;
- require the candidate median to exceed the comparator median;
- require either a positive paired result in all five pairs or a 95% paired
  bootstrap confidence interval whose lower bound is above `1.00x`;
- report every sample, coefficient of variation, median ratio, and timing scope.

A sub-window or kernel-family win may be retained under the repository's normal
performance policy, but it does not satisfy the full prefill or decode target.

## 2. Frozen artifact and opening blocker

| Field | Value |
| --- | --- |
| Path | `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` |
| Bytes | `16,464,440,224` |
| Full SHA-256 | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` |
| GGUF tensors | `866` |
| File type | `MOSTLY_Q4_K_M` |
| Geometry | hidden `5120`, FFN `17408`, 64 autoregressive blocks plus trailing NextN block |
| Layer mix | 48 linear-attention/GDN blocks and 16 full-attention blocks |
| Context declaration | 262,144 tokens |

The GGUF payload inventory is:

| GGML type | Tensors | Payload | Current relevance |
| --- | ---: | ---: | --- |
| `F32` | 360 | 0.010 GiB | Supported metadata/norm tensors |
| `IQ3_S` | 4 | 0.143 GiB | Unsupported dense projections; no hipEngine CPU dequantizer |
| `IQ4_NL` | 7 | 0.308 GiB | CPU dequantizer exists; no dense projection route |
| `IQ4_XS` | 117 | 4.433 GiB | Selected-MoE math exists; dense route still expands to BF16 and needs compressed consumers |
| `Q3_K` | 7 | 0.250 GiB | CPU oracle and selected-expert kernels exist; dense rank-2 route is rejected |
| `Q4_K` | 104 | 3.915 GiB | Existing native family |
| `Q5_K` | 131 | 4.502 GiB | Existing native family |
| `Q6_K` | 30 | 1.688 GiB | Existing native family |
| `Q8_0` | 106 | 0.075 GiB | Existing native family |

The 18 unsupported tensors occupy about **0.700 GiB** in the GGUF. Expanding
only those tensors to BF16 would consume about **2.920 GiB**, an increase of
about **2.220 GiB**. Resident BF16 is therefore a bring-up diagnostic at most,
not an admissible production fallback.

### 2.1 Exact unsupported tensor map

| Type | Shape | Tensor |
| --- | --- | --- |
| `Q3_K` | `[17408,5120]` | `blk.0.ffn_up.weight` |
| `IQ4_NL` | `[5120,17408]` | `blk.1.ffn_down.weight` |
| `IQ4_NL` | `[5120,17408]` | `blk.2.ffn_down.weight` |
| `IQ4_NL` | `[5120,17408]` | `blk.3.ffn_down.weight` |
| `IQ3_S` | `[17408,5120]` | `blk.11.ffn_gate.weight` |
| `Q3_K` | `[5120,17408]` | `blk.13.ffn_down.weight` |
| `Q3_K` | `[17408,5120]` | `blk.13.ffn_gate.weight` |
| `IQ3_S` | `[5120,17408]` | `blk.14.ffn_down.weight` |
| `Q3_K` | `[17408,5120]` | `blk.14.ffn_gate.weight` |
| `Q3_K` | `[17408,5120]` | `blk.14.ffn_up.weight` |
| `IQ3_S` | `[5120,17408]` | `blk.15.ffn_down.weight` |
| `Q3_K` | `[17408,5120]` | `blk.15.ffn_gate.weight` |
| `Q3_K` | `[17408,5120]` | `blk.16.ffn_gate.weight` |
| `IQ3_S` | `[5120,17408]` | `blk.17.ffn_down.weight` |
| `IQ4_NL` | `[10240,5120]` | `blk.21.attn_qkv.weight` |
| `IQ4_NL` | `[17408,5120]` | `blk.27.ffn_gate.weight` |
| `IQ4_NL` | `[17408,5120]` | `blk.27.ffn_up.weight` |
| `IQ4_NL` | `[17408,5120]` | `blk.50.ffn_gate.weight` |

The opening failure is expected and correct:

```text
ValueError: unsupported Qwen3.5 GGUF tensor type 'Q3_K' outside rank-3 expert slots: blk.0.ffn_up.weight
```

Do not remove this rejection until the relevant registered strict route exists.

## 3. Existing assets and gaps

### 3.1 Reusable assets

- `hipengine/quant/gguf.py` already defines all three block layouts and provides
  validated NumPy dequantization for `Q3_K` and `IQ4_NL`.
- `hipengine/kernels/hip_gfx1100/quant/gguf_q3_k_gemv.{hip,py}` provides
  raw `Q3_K` selected-expert GEMV arithmetic and launch patterns that can seed
  a dense rank-2 implementation.
- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_gemv.{hip,py}` already carries
  the `IQ4_NL` codebook and source decoding used by `IQ4_XS` kernels.
- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_source_mmq_prefill.{hip,py}`
  and `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_prefill.{hip,py}`
  provide source-native IQ prefill mechanisms and codebook expansion patterns.
- Existing Q4/Q5/Q6/Q8 dense kernels, sole-resident T16 layouts, graph replay,
  c>N runners, MTP target rows, and public/server lifecycle tests remain the
  operation-completeness template.
- The current `Q4_K_M` Qwen3.8 campaigns provide same-model-family baselines and
  known-good harnesses. Their numerical values are not transferred to this
  artifact without fresh same-host reruns.

### 3.2 Missing work

- Dense IQ4_XS operation-complete consumers and role-aware per-tensor resident
  policy for existing Q4/Q5/Q6/Q8. Complete UD-U3 before KM-U3; fixing only
  the 18 refusal sites cannot satisfy compact exact-artifact support.
- `IQ3_S` NumPy/CPU-reference dequantization and real-row llama.cpp oracle.
- Dense rank-2 quant plugin contracts for `Q3_K`, `IQ4_NL`, and `IQ3_S`.
- Strict raw dense GEMV for BF16 input and BF16/F32 output.
- Pair and fused-SiLU routes where adjacent gate/up tensors share a codec.
- Multirow/c>N decode kernels for physical rows 2, 4, and 8.
- Bulk-prefill consumers for prompt rows, tails, and the `attn_qkv` output
  geometry.
- Materialization, session-plan, graph, profiler-family, and variant-manifest
  admission for the mixed preset.
- Exact-file public, Generation-2, MTP, lifecycle, memory, and teardown gates.
- Fresh same-host Q4_K_M and llama.cpp HIP/Vulkan controls on both hardware
  lanes.

## 4. Architecture and implementation rules

1. **Register a mixed-preset quant plugin.** Use an artifact-qualified preset
   key such as `gguf_ud_q4_k_m` for session/model admission. Tensor-level
   kernels retain their concrete quant keys (`gguf_q3_k`, `gguf_iq4_nl`, and
   `gguf_iq3_s`). Do not treat every `MOSTLY_Q4_K_M` file as this preset.
2. **Bind admission to immutable structure.** Match architecture geometry,
   GGUF file type, complete tensor-type/shape manifest, and required codec set.
   The path and `general.name` are telemetry, not policy keys. Record the full
   artifact SHA-256 in benchmark evidence.
3. **Keep dispatch plugin-based.** Register `(backend, layer, quant, variant)`
   keys and backend capabilities. Do not add `if backend == ...` or
   `if quant == ...` branches to engine/model dispatch.
4. **Preserve one resident weight payload.** Raw GGUF is the initial strict
   owner. A replacement layout may become the sole owner only after it covers
   every required operation and wins complete-model gates. Do not retain raw
   plus a full repack by default.
5. **Provide strict fallbacks.** Every pair, fused-SiLU, MMQ, or changed-
   arithmetic kernel has a registered unfused or exact raw fallback. A dense
   BF16 expansion is not the normal strict fallback.
6. **Keep raw-pointer kernel ABIs.** Device kernels consume typed/raw pointers,
   dimensions, and streams; wrappers own validation and ctypes conversion.
7. **Separate hardware policy.** gfx1100 and gfx1151 may share source bodies,
   but launch geometry, compiler resources, crossover thresholds, and default
   variants require independent gates.
8. **Classify arithmetic.** Exact GGML-order kernels begin as strict/T0. Integer
   MMQ, activation quantization, changed reduction association, sparse repair,
   or reassociation must declare T1/T2 and pass the production-profile gate.
9. **Do not weaken the artifact.** Re-quantizing unsupported tensors to `Q4_K`
   is a useful control and remains supported by
   `scripts/qwen38_mixed_quant_plan.py`; it does not satisfy exact-artifact
   support.

## 5. Milestone plan and punchlist

Each implementation milestone is one logical unit with a RED test, immutable
worklog entry, focused validation, and immediate atomic commit. Do not combine
all three codecs into one unreviewable change.

### U0 — Freeze identity, comparators, and evaluator

- [ ] Record `sha256sum`, file bytes, GGUF metadata, tensor names, shapes, types,
      payload bytes, tensor-manifest hash, tokenizer hash, and NextN inventory.
- [ ] Freeze the supported Qwen3.8 `Q4_K_M` comparator identity separately on
      each host; do not assume the copies share a SHA-256.
- [ ] Record physical host identity, PCI identity where available, GPU target,
      ROCm/runtime/compiler versions, kernel, clocks/power policy, and available
      memory.
- [ ] Build clean pinned llama.cpp HIP and Vulkan comparators on each host.
- [ ] Select one common KV dtype and cache policy for binding cross-engine rows.
- [ ] Capture llama.cpp full-logit/selected-token strict oracles from the exact
      `UD-Q4_K_M` file at small fixed contexts and natural prompts.
- [ ] Freeze the repeated-token shape suite and the complete ten-prompt
      category/heldout suite before kernel tuning.
- [ ] Add a compact `benchmarks/results/` opening artifact containing the failed
      hipEngine load and all valid external baselines.

**Exit gate:** identities and protocols are reproducible; every baseline row
names its timing scope and correctness status. No implementation work starts
from an unpinned llama.cpp tree.

### U1 — CPU-reference codec contracts

#### U1A — `Q3_K`

- [ ] Reuse and re-audit the existing block parser against current pinned
      llama.cpp `dequantize_row_q3_K`.
- [ ] Add real rows from each required shape/orientation, including down and
      gate/up tensors.
- [ ] Cover packed 6-bit scales, hmask boundaries, negative values, zero scale,
      and multiple 256-value blocks.

#### U1B — `IQ4_NL`

- [ ] Re-audit the existing 32-value codebook dequantizer against pinned
      llama.cpp `dequantize_row_iq4_nl`.
- [ ] Add real FFN and `attn_qkv` rows, both matrix orientations, codebook
      extrema, zero/subnormal scale, and multi-block rows.

#### U1C — `IQ3_S`

- [ ] Add the GGML block structure and a NumPy dequantizer translated from the
      pinned llama.cpp reference.
- [ ] Add a tiny hand-checkable fixture before the HIP implementation.
- [ ] Add real rows from gate and down tensors and compare complete row output
      with a compiled llama.cpp oracle.
- [ ] Cover grid/codebook indices, sign masks, packed scales, block boundaries,
      and malformed-size rejection.

- [ ] Register `GGUFIQ4NLQuant` and `GGUFIQ3SQuant` contracts and update exports.
- [ ] Update `docs/KERNELS.md` only when native kernel families land, not when
      metadata alone exists.

**Exit gate:** CPU outputs agree with the independent llama.cpp oracle under the
format's exact declared tolerance; `python3 scripts/check_fixtures.py` passes.

### U2 — Strict dense c1 decode kernels

Implement in this order so the least-new arithmetic proves the dense ABI first:

1. `Q3_K` dense raw GEMV;
2. `IQ4_NL` dense raw GEMV;
3. `IQ3_S` dense raw GEMV.

For each codec:

- [ ] Add BF16-input to BF16-output and BF16-input to F32-output variants.
- [ ] Preserve the CPU/llama reduction contract for the strict variant.
- [ ] Validate K=`5120` and K=`17408`, all required N dimensions, row
      alignment, tails, and wrong-shape rejection.
- [ ] Test synthetic random blocks and real artifact rows against CPU reference.
- [ ] Register exact `linear` variants for `hip_gfx1100`.
- [ ] Register or transfer independently gated variants for `hip_gfx1151`.
- [ ] Capture cached `rocprofv3 --kernel-trace` evidence with expected symbols,
      plausible durations, VGPR/SGPR/LDS/scratch resources, and actual artifact
      pointers.
- [ ] Keep the materializer rejection in place until all three families pass.

Then add only profile-justified composites:

- [ ] `Q3_K` gate+up pair/fused-SiLU for `blk.14`.
- [ ] `IQ4_NL` gate+up pair/fused-SiLU for `blk.27`.
- [ ] Pair only same-codec, same-input, compatible-shape tensors; retain two
      registered primitive calls as fallback.

**Exit gate:** all 18 projections execute c1 from raw compressed bytes and pass
strict primitive gates on both backends. No full-model performance claim yet.

### U3 — Materialization and strict full-model bring-up

- [ ] Add artifact-structure admission for `gguf_ud_q4_k_m`.
- [ ] Materialize all 18 tensors as raw GGUF records with concrete tensor quant
      keys; reject an inventory or shape mismatch.
- [ ] Extend the projection resolver without adding backend/quant branches to
      model code.
- [ ] Verify one and only one resident owner per logical weight and zero BF16
      expansion bytes for the 18 tensors.
- [ ] Run a one-layer probe for each affected layer role before public E2E.
- [ ] Add a deterministic public `LLM.generate()` fixture with llama.cpp prompt
      tokenization and selected-token/logit oracle.
- [ ] Prove `torch` is absent from the generate hot path.
- [ ] Capture eager and one-step graph replay; verify selected/fallback variant
      manifest, graph reuse, and teardown to zero tracked bytes.
- [ ] Run fixed-context strict-teacher logits at 128, 512, 1K, and 4K.
- [ ] Run all ten natural category/heldout prompts for at least 24 transitions.

**Exit gate:** public c1 strict generation passes on gfx1100 and gfx1151 with
native raw kernels and exact control ownership. Publish this as functional
support only if performance milestones remain open.

### U4 — Native c>N decode and MTP target rows

- [ ] Add physical rows 2, 4, and 8 for each codec, initially through exact
      row-batched raw kernels.
- [ ] Cover `[C,K] x [N,K]`, per-row outputs, ragged active masks, and stable
      row ownership.
- [ ] Extend pair/fused-SiLU variants only where they beat primitive rows under
      actual-weight cold-stream tests.
- [ ] Wire native target verification for MTP B1/B2/B3 without row-local serial
      fallback.
- [ ] Run c1/c2/c4/c8 eager and graph parity, row permutation, neighbor
      substitution, sparse retirement, cancellation/reclaim, width transitions,
      compaction, page/KV ownership, and repeated-server lifecycle gates.
- [ ] Record actual physical widths and fail if a claimed c>N row decomposes to
      c1 calls.

**Exit gate:** operation-complete strict c>N and MTP target execution on both
backends. MTP speed is not a campaign requirement until true AR is optimized,
but MTP correctness cannot regress.

### U5 — Bulk-prefill baseline and ranked profile

- [ ] Implement an exact row-grid prefill fallback for every codec before
      changed-arithmetic work.
- [ ] Add affected-role markers to the existing Qwen3.8 prefill profiler.
- [ ] Profile 512, 1K, and 4K on each backend with cached builds and one loaded
      resident session.
- [ ] Report complete wall, trace span, kernel-family sums under overlap, launch
      counts, resources, memory traffic where counters are trustworthy, and
      per-role tensor bytes.
- [ ] Reconcile at least 95% of the marked prefill wall before selecting an
      optimization target.
- [ ] Rank `Q3_K`, `IQ4_NL`, and `IQ3_S` separately; do not prioritize by tensor
      count alone.

**Exit gate:** one current Amdahl ledger per backend determines the prefill
candidate order.

### U6 — Source-native bulk-prefill kernels

Use measured order from U5. Candidate families are a ladder, not assumptions:

1. **Exact raw multirow reuse:** reuse one decoded weight block across prompt
   rows without changing each row's reduction order.
2. **Source-native MMQ:** consume GGUF blocks directly with Q8_1 or residual-D4
   activations; keep codebooks/scales in the representation that wins actual
   role measurements.
3. **Sparse exact repair:** only if changed arithmetic fails strict parity near
   BF16 boundaries and a bounded repair queue passes the complete production
   gate with accounted memory.
4. **Sole replacement layout:** only if raw-source MMQ cannot reach the target.
   The replacement must cover c1, c>N, prefill, graph, composites, and root or
   attention roles before raw bytes can be dropped.

For each retained candidate:

- [ ] Measure actual artifact weights with cycling pools larger than relevant
      caches; do not promote a warm microbenchmark result.
- [ ] Test short rows, 512/1K/4K rows, chunk tails, non-multiple row counts, and
      both matrix orientations.
- [ ] Record activation preprocessing and scratch bytes in the complete memory
      ledger.
- [ ] Require strict parity for T0 or the complete production-profile packet for
      T1/T2.
- [ ] Keep exact raw prefill and decode fallbacks registered.
- [ ] Re-profile the full model after every structural keep.

**Exit gate:** hipEngine `UD-Q4_K_M` beats same-host hipEngine `Q4_K_M` prefill
at all three shapes on each backend under the statistical win rule.

### U7 — Decode optimization to the bandwidth target

Start from a current graph trace after U3/U4, not from prefill results.

- [ ] Compute bytes/token for the complete resident model and for the 18 new
      tensors; compare implied bandwidth with measured large-stream bandwidth.
- [ ] Profile c1 and physical c2/c4/c8 separately. c1 is primarily a weight-read
      problem; c>N may justify output tiling or MMQ.
- [ ] Screen output-column tiling, coalesced metadata/codebook loads, wave reuse,
      non-temporal loads, pair fusion, and sole replacement layouts in that
      order only when the trace supports the premise.
- [ ] Keep codebooks in constant memory only if cache/counter evidence and full
      wall improve on both hot and cold controls.
- [ ] Do not use launch-count reduction as the sole premise when graph replay is
      already healthy.
- [ ] Re-run graph/eager, fixed-shape, natural AR, MTP target, memory, and
      lifecycle gates after every retained decode change.

**Exit gate:** hipEngine `UD-Q4_K_M` beats same-host hipEngine `Q4_K_M` true AR
decode at 512, 1K, and 4K on each backend.

### U8 — Same-artifact llama.cpp parity and campaign closure

- [ ] Rebuild/reconfirm pinned llama.cpp HIP and Vulkan after the final
      hipEngine implementation; do not reuse a baseline collected under a
      different system load or software stack.
- [ ] Run matched repeated-token `512/128`, `1024/128`, and `4096/128` rows for
      hipEngine, llama.cpp HIP, and llama.cpp Vulkan.
- [ ] Verify exact prompt-token arrays, selected output IDs or declared
      strict-teacher result, common KV type, cache policy, and timing ownership.
- [ ] Run the complete ten-prompt natural true-AR suite with at least five
      measurements per engine after warmup.
- [ ] Compare hipEngine with the faster correctness-valid llama.cpp backend in
      every prefill and AR cell.
- [ ] Record model-load time, steady resident memory, transient peak, process
      GTT/VRAM as applicable, and teardown separately from throughput.
- [ ] Run a cached profiler smoke proving each new codec family executes under
      its expected final symbol.
- [ ] Run the milestone full test suite according to `docs/TESTING.md`; apply
      the focused-repair rule to isolated failures.
- [ ] Update `docs/KERNELS.md`, `docs/PLAN.md` if architecture moved,
      `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and compact artifacts.
- [ ] Remove superseded temporary flags or add precise removal conditions to
      `docs/REFACTOR.md`.

**Campaign exit gate:** all seven success conditions in section 1 pass on both
independent hardware lanes. Report gfx1100 and gfx1151 separately; never combine
or average their rates.

## 6. Correctness and quality matrix

### 6.1 Primitive gates

Every codec/kernel family requires:

- independent NumPy/CPU-reference output;
- pinned llama.cpp real-row oracle;
- synthetic edge blocks and real artifact rows;
- BF16 and F32 output checks where exposed;
- exact shape, byte-stride, alignment, and bounds validation;
- strict fallback resolution and missing/duplicate registration tests;
- HIP availability guards for GPU tests;
- cached `rocprofv3` symbol/resource evidence.

The outer smoke floor remains KL `<=0.05` and top-1 agreement `>=90%`, but that
floor alone cannot qualify a strict kernel or production default.

### 6.2 Full-model strict gate

At minimum:

- repeated-token 128/512/1K/4K contexts and 128 transitions;
- complete ten-prompt `mtpbench-code-general-ja.jsonl` suite;
- strict-teacher aligned full logits at identical contexts;
- fixed-schedule repeat determinism;
- eager versus graph parity under the strict contract;
- exact request, slot, token, position, mask, KV, recurrent-state, and
  transaction ownership;
- c1/c2/c4/c8 row isolation and permutations;
- prompt/decode boundary, chunk-tail, graph-reuse, cancellation, retirement,
  teardown, and fresh-session controls;
- public blocking and server streaming/non-streaming paths.

### 6.3 Quant-quality gate

`UD-Q4_K_M` and `Q4_K_M` are different weight artifacts, so generated-token
identity between them is not a correctness requirement. Before claiming the
smaller artifact as a superior product route:

- compare both against the same pinned BF16/F16 Qwen3.8 teacher with aligned
  full logits;
- report mean/p95/p99/max KL, top-1, teacher-token negative log-likelihood,
  category and prompt minima, and every top-1 mismatch;
- run the repository's sealed task-quality suite or its approved Qwen3.8
  successor;
- require `UD-Q4_K_M` to meet the preset's declared quality envelope and avoid
  task regression beyond the predeclared threshold.

Speed cannot compensate for a failed binding quality gate.

## 7. Benchmark protocol skeleton

The final commands may need harness extensions for the new quant key. U0 must
freeze the exact accepted command lines before measurements. The intended
hipEngine shape is:

```bash
export MODEL_UD=/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf
export MODEL_Q4=/models/gguf/Qwen3.8-27B-Q4_K_M.gguf
export BACKEND=hip_gfx1100                 # independent rerun: hip_gfx1151
export HIPENGINE_BACKEND=$BACKEND
export HIPENGINE_HIP_ARCH=${BACKEND#hip_}
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-udq4-hipcc-version.txt
hipcc --version > "$HIPENGINE_COMPILER_VERSION_FILE"

HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
  uv run python scripts/qwen35_gguf_bench.py \
  --model "$MODEL_UD" --quant gguf_ud_q4_k_m \
  --prompt-length 512 --decode-tokens 128 \
  --warmup-decode-tokens 4 --warmup-runs 1 --measured-runs 5 \
  --persistent-session --graph-replay-decode \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --require-cached-build --json /tmp/udq4-512-128.json
```

Run the same command with `MODEL_Q4`, its exact quant key, and a distinct output
path in counterbalanced order. Repeat for 1K and 4K. The natural AR/MTP control
uses the full suite and a true no-MTP denominator:

```bash
HIPENGINE_REQUIRE_CACHED_BUILD=1 PYTHONPATH=. \
  uv run python scripts/qwen36_dense_gguf_suite.py \
  --model "$MODEL_UD" --quant gguf_ud_q4_k_m \
  --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --max-new-tokens 25 --candidate-budgets 3 \
  --target-verify-mode native --runs 5 --limit 10 --warmup \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --require-cached-build --output /tmp/udq4-natural25.json
```

llama.cpp commands must be captured from the pinned binaries on each host. Use
raw token-ID prompts or an exact tokenizer-hash/token-array bridge. Record HIP
and Vulkan separately, disable prompt caching, use the common KV type, and keep
`llama-bench` split timing separate from complete public/server wall timing.

## 8. Performance candidate order

Do not start with a broad kernel sweep. After the first current profiles, use
this decision order:

1. **Route audit:** prove existing registered kernels are actually selected.
2. **Resident ownership:** remove accidental BF16 or duplicate payloads.
3. **Strict c1 raw kernels:** establish a correct, graph-safe floor.
4. **Actual-role pair fusion:** only `blk.14` Q3 gate/up and `blk.27` IQ4_NL
   gate/up are obvious same-codec candidates from the frozen inventory.
5. **Multirow weight reuse:** optimize physical c2/c4/c8 and prompt rows without
   changing quant math.
6. **Source-native MMQ:** attack prefill after a ranked trace.
7. **Replacement layout:** consider only when it can be the sole operation-
   complete payload and measured raw paths cannot reach target.
8. **Production arithmetic:** use only after exact avenues are measured and the
   calibrated gate is ready.

Rejected or neutral candidates remain rejected unless a new profile changes the
premise. Record them in the immutable worklog and `docs/REFACTOR.md` where code
or selectors survive.

## 9. Deliverables

The campaign is expected to produce:

- quant metadata and CPU-reference support for `IQ3_S` and `IQ4_NL`;
- dense raw HIP kernel families and wrappers for all three codecs;
- registered strict fallbacks and backend-qualified fast variants;
- artifact-scoped `gguf_ud_q4_k_m` materialization and runtime admission;
- c1, c>N, bulk-prefill, graph, public API, server, and MTP target support;
- tiny committed fixtures and independent llama.cpp oracle metadata;
- profiler and benchmark harness support for per-codec attribution;
- one compact correctness/performance artifact per retained or rejected unit;
- final same-host Q4_K_M and llama.cpp HIP/Vulkan comparison artifacts for each
  backend;
- updated kernel catalog, benchmark rollup, changelog, and immutable worklog
  entries.

Raw profiler dumps, full logits, model weights, JIT objects, and terminal logs
remain uncommitted.

## 10. Closure checklist

### Exact artifact and architecture

- [ ] Exact SHA-256 and complete tensor manifest match.
- [ ] No path/name-conditioned admission or hot-path backend/quant branches.
- [ ] One resident payload per logical weight; no production BF16 expansion.
- [ ] Strict fallbacks resolve for every fused/fast variant.

### Correctness

- [ ] `Q3_K`, `IQ4_NL`, and `IQ3_S` CPU/llama oracles pass.
- [ ] Primitive HIP gates and profiler-symbol smokes pass on gfx1100.
- [ ] Primitive HIP gates and profiler-symbol smokes pass on gfx1151.
- [ ] Public c1, graph, c>N, MTP target, lifecycle, and server gates pass.
- [ ] Strict full-model and quant-quality/task gates pass.

### Performance

- [ ] UD beats Q4 prefill at 512/1K/4K on gfx1100.
- [ ] UD beats Q4 decode at 512/1K/4K on gfx1100.
- [ ] UD beats faster valid llama HIP/Vulkan prefill at all three shapes on gfx1100.
- [ ] UD beats faster valid llama HIP/Vulkan decode at all three shapes on gfx1100.
- [ ] UD beats Q4 prefill at 512/1K/4K on gfx1151.
- [ ] UD beats Q4 decode at 512/1K/4K on gfx1151.
- [ ] UD beats faster valid llama HIP/Vulkan prefill at all three shapes on gfx1151.
- [ ] UD beats faster valid llama HIP/Vulkan decode at all three shapes on gfx1151.

### Evidence and publication

- [ ] Every row names exact model, quant, host, hardware, command, shape, KV
      policy, execution profile, variant manifests, result, and correctness gate.
- [ ] gfx1100 and gfx1151 artifacts and claims remain separate.
- [ ] `benchmarks/README.md` and `benchmarks/CHANGELOG.md` are current.
- [ ] `docs/KERNELS.md`, `docs/PLAN.md`, and `docs/REFACTOR.md` reflect retained
      architecture and cleanup obligations.
- [ ] All logical units have validated atomic commits and immutable worklog
      entries.
