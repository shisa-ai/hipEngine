# Qwen3.8-Flash-Next gfx1151 Performance Campaign

Status: **active plan, 2026-08-30.** hipEngine has a working named `production`
path, but the same-host llama.cpp HIP baseline still leads by **3.22x** on p508
prefill and **1.10x** on tg32 decode. This document is the performance-specific
plan and punchlist. [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md) remains the
model/bring-up authority; this file owns only the gap-closure campaign.

## 1. Objective and boundaries

Close the same-host, same-model, same-quant performance gap to llama.cpp HIP
first. Vulkan remains the stretch comparator. A retained win must preserve the
published execution-profile contract and must come with a compact artifact, a
worklog entry, and a benchmark rollup.

### In scope

- Host: `zbook`, AMD Ryzen AI Max+ Pro 395, Radeon 8060S, `gfx1151`.
- Model: `Qwen/Qwen3.8-Flash-Next` through the pinned Unsloth
  `UD-Q4_K_XL` split GGUF.
- KV/cache policy: current BF16 baseline unless a row explicitly declares a
  different KV profile.
- Profiles: named `strict` and `production` manifests.
- Workloads: p508 prefill, p1012 prefill, tg32 steady decode, then the existing
  long-context and MTP suites.

### Out of scope

- Changing model representation, quant recipe, or prompts to improve a score.
- Treating external EngramHalo/Nathan numbers as hipEngine results.
- Vulkan-specific code in this campaign except as design evidence for HIP work.
- New feature expansion that does not close a measured gap.
- Inferring W7900/gfx1100 performance from this host.

## 2. Current verified gap

Evidence anchor:
[`2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json).
It pins hipEngine commit `15a436766`, llama.cpp remote HEAD
`57291f2644af8c9df0dd8d44395881c5bdcf0ecd`, HIP/Vulkan comparator binary
hashes, production manifest `9e27fec0...`, and strict manifest `42509601...`.

### End-to-end rows

| Workload | hipEngine production | llama.cpp HIP | llama.cpp Vulkan | HIP advantage | Vulkan advantage |
| --- | ---: | ---: | ---: | ---: | ---: |
| p508 prefill | **84.83 tok/s** | 272.83 | 331.03 | **3.22x** | **3.90x** |
| tg32 steady decode | **15.19 tok/s** | 16.64 | 24.22 | **1.10x** | **1.59x** |

### Device-kernel windows

| Window | hipEngine kernel sum | llama HIP kernel sum | HIP advantage | hipEngine rows | llama rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| p508 | **5.959 s** | 1.625 s | **3.67x** | 3,328 | 5,119 |
| tg decode per token | **48.63 ms** | 38.90 ms | **1.25x** | 1,860 | 4,203 |

Kernel rows are not host launches. llama's p508 run expands one selected
measured graph to 5,119 kernels; hipEngine submits 2,796 direct launches and
adds 532 copy/fill rows. During decode, hipEngine submits 1,195 direct launches
plus 48 per-layer MoE graph launches per token, while llama submits nearly the
whole transition through 31 large graphs for 32 outputs.

### Prefill module gaps

| Module | hipEngine | llama HIP | HIP advantage |
| --- | ---: | ---: | ---: |
| Selected Q4 gate/up | 1.297 s | 0.477 s | 2.72x |
| Selected Q5_1 down | 1.131 s | 0.345 s | 3.28x |
| Layer-2 Q5_K gate/up | 301.5 ms | 15.4 ms | 19.61x |
| GDN prefill | 634.9 ms | 92.3 ms | 6.88x |
| QSA prefill | 110.5 ms | 13.9 ms | 7.94x |
| Total | 5.959 s | 1.625 s | 3.67x |

MoE owns **3.161 s** of the hipEngine p508 kernel sum; layers 0-26 alone own
**2.526 s**. Layer 2 owns about **397.95 ms**, or roughly **6.6%** of the whole
p508 device window.

### Decode module gaps per token

| Module | hipEngine | llama HIP | HIP advantage |
| --- | ---: | ---: | ---: |
| Dense Q8 | 25.28 ms | 21.84 ms | 1.16x |
| Selected Q4 gate/up | 7.64 ms | 4.60 ms | 1.66x |
| Selected Q5_1 down | 6.26 ms | 2.84 ms | 2.21x |
| GDN recurrence | 2.66 ms | 0.46 ms | 5.72x |
| QSA attention | 0.11 ms | 0.08 ms | 1.31x |
| Total device | 48.63 ms | 38.90 ms | 1.25x |

Decode GR projection/read/elementwise roles own **7.775 ms/token** and expose
up to **387** removable direct launches per token if the operations become
operation-complete. Decode also has a profiled span-minus-kernel gap of
**37.1 ms/token**, so decode has both device-kernel and submission headroom.

### Invalid path removed

The previous "GDN colwarps decode all layers" row is invalid: its selector sat
below the `rows == 1` branch and compared the strict owner with itself. Wiring
the actual candidate costs **6.832 + 0.117 ms/token**, versus **2.454 ms/token**
for the retained serial-column owner, and lowers full decode. Commit
`15a436766` removed the dead route. The corrected evidence is
[`2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json).

## 3. Profiling pattern

The committed tools replace the earlier `/tmp` harnesses. Use the same sequence
for every campaign claim.

### 3.1 Freeze identity

1. Record repository, host, power, model, quant, profile manifests, and
   comparator revisions before measuring.
2. Confirm HIP and device visibility:

   ```bash
   python3 -c "import ctypes; ctypes.CDLL('libamdhip64.so'); print('hip OK')"
   rocminfo | grep -E 'Name:|gfx'
   ```

3. Prebuild hipEngine kernels outside the profiler and pass
   `--compiler-version-file` plus `--require-cached-build` to the profiled
   process. Do not let `rocprofv3` spawn `hipcc` or clang children.

### 3.2 Build the comparator once

The 2026-08-30 comparator used llama.cpp remote HEAD
`57291f2644af8c9df0dd8d44395881c5bdcf0ecd` with these effective Release build
settings: HIP on, `AMDGPU_TARGETS=gfx1151`, `GGML_HIP_GRAPHS=ON`,
`GGML_HIP_MMQ_MFMA=ON`, and Vulkan in a separate build tree. The artifact stores
both `llama-bench` SHA-256 values. Refresh the comparator only as a separate
baseline event; do not compare old and new absolute rows as an old→new
optimization.

### 3.3 Collect end-to-end wall rows

Prompt fixture:
[`benchmarks/prompts/qwen4exp-p508.txt`](../benchmarks/prompts/qwen4exp-p508.txt),
SHA-256 `9cf9d353b81b6ce1df61405b590f037b0502b52c7f6c0c19a543c33cbcb6dbb4`.

hipEngine p508:

```bash
MODEL_ROOT=/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL
HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-qwen4exp-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 \
uv run python scripts/qwen4exp_profile_gap.py \
  --model-root "$MODEL_ROOT" \
  --mode prefill \
  --prompt-file benchmarks/prompts/qwen4exp-p508.txt \
  --expected-prompt-tokens 508 \
  --repetitions 3 \
  --output /tmp/qwen4exp-p508-wall.json
```

hipEngine steady decode and the exact-ID sync A/B:

```bash
uv run python scripts/qwen4exp_decode_sync_ab.py \
  --model-root "$MODEL_ROOT" \
  --steps 32 --warmup-steps 8 --pair-repetitions 3 \
  --output /tmp/qwen4exp-decode-sync-ab.json
```

llama.cpp HIP/Vulkan rows use the first split GGUF path, `-p 508 -n 0` for
prefill and `-p 0 -n 32` for decode, three repetitions, full GPU offload, BF16
K/V, and flash attention `auto`. The artifact is the source for the exact
binaries and retained JSON rows.

### 3.4 Collect role-resolved device traces

hipEngine p508:

```bash
TRACE=/tmp/qwen4exp-role-p508
rm -rf "$TRACE" && mkdir -p "$TRACE"
HIPENGINE_HIP_ARCH=gfx1151 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-qwen4exp-hipcc-version.txt \
HIPENGINE_REQUIRE_CACHED_BUILD=1 \
rocprofv3 --kernel-trace --hip-trace --marker-trace --output-format csv \
  -d "$TRACE" -o role-p508 -- \
  uv run python scripts/qwen4exp_profile_gap.py \
    --model-root "$MODEL_ROOT" \
    --mode prefill \
    --prompt-file benchmarks/prompts/qwen4exp-p508.txt \
    --expected-prompt-tokens 508 \
    --profile --role-markers --repetitions 1 \
    --output "$TRACE/child.json"
```

hipEngine decode uses the same driver in decode mode, normally 8 warmup steps
and 16 measured steps with `max_sequence_length=128` and `prefill_chunk_size=256`.
Do not treat the profiled wall as a speed claim; use the unprofiled wall rows for
that.

llama.cpp HIP traces use `rocprofv3 --kernel-trace --hip-trace
--hip-graph-trace --memory-copy-trace --stats` around `llama-bench -r 1`, then
select the measured graph/window. Record raw CSV hashes in the artifact.

### 3.5 Analyze without conflating rows and launches

```bash
uv run python scripts/qwen4exp_trace_analyze.py \
  --trace-dir "$TRACE" --engine hipengine \
  --marker-prefix qwen4exp_prefill_p508_ \
  --output "$TRACE/summary.json"

uv run python scripts/qwen4exp_role_analyze.py \
  --trace-dir "$TRACE" \
  --measure-prefix qwen4exp_prefill_p508_ \
  --output "$TRACE/roles.json"
```

The analyzer reports the selected marker window, kernel sum/span, row counts,
family totals, HIP API launch correlations, unmatched graph/copy rows, and
memory-copy rows. The role analyzer correlates ROCTX ranges to HIP launch
correlation IDs and then to kernel rows. `scripts/qwen4exp_perf_gap_report.py`
renders the compact artifact as markdown tables:

```bash
uv run python scripts/qwen4exp_perf_gap_report.py \
  benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json
```

Interpretation rules:

- End-to-end unprofiled wall is the headline.
- Kernel sum ranks device dataflow.
- Kernel span minus sum exposes gaps, but profiler inflation is not Python
  overhead without a separate unprofiled event/control.
- Direct `hipLaunchKernel` correlations count host submissions; graph-expanded
  kernels and copy/fill kernels appear as trace rows without direct launches.
- A/B decisions use same-session counterbalanced orders and identical IDs or
  the applicable production-profile gate.

## 4. External evidence review

This section records useful external hypotheses and their evidentiary status.
None of these numbers are hipEngine results.

| Source | Mechanism or claim | Status for this campaign |
| --- | --- | --- |
| [Sleeping Robots, 2026-08-29](https://sleepingrobots.com/dreams/engramhalo-qwen38-flash-next-strix-halo/) | Independently tested EngramHalo on Strix Halo with a different quant. MTP reaches 28-38 tok/s at working depths; 26K MTP regresses to 15.0; kernel-only prefill improves up to about 35% at 26K. | Useful cross-check of the direction, not a same-quant baseline. Confidence: medium-high for the external fork, low for transfer magnitude. |
| [Aristo94/EngramHalo.cpp](https://github.com/Aristo94/EngramHalo.cpp), inspected at `1423f689986f670417128fd545a0aa1241166103` | Wide radix top-k (`33766da`), masked-slice FA skip (`bf8412d`), QSA top-k row gather (`2606d49`), MTP sidecar (`afb80ed` + `2ba3009`), PLE lazy row prefetch (`c911e6b`), load-page drop-behind (`5486559`). Chunked GDN prefill exists (`62160a7`) but was explicitly not active in the published numbers. | Code mechanisms verified by source inspection. hipEngine already has device QSA top-k and sparse selected-position attention; PLE prefetch and MTP economics remain open. |
| [Nathanw1014/strix-halo-llamacpp v0.7.2](https://github.com/Nathanw1014/strix-halo-llamacpp/releases/tag/v0.7.2), branch source inspected at `ad914eb6587d3da8b2bf50f0056cc20b3d3e91f5` | `TENSOR_READ_LAZY` + `MADV_RANDOM` alone is not enough; batched `WILLNEED` row prefetch is the paying half. Release reports pp512 99.01→352.38 on a 64 GB box, about 1.35-1.53x prefill on three 128 GB boxes, and decode around neutral. | Strong design evidence for a separate cold-cache/PLE lane. It does not explain the current warm p508 GPU-kernel gap. Test on this host before claiming any win. |
| Upstream llama.cpp [#27742](https://github.com/ggml-org/llama.cpp/pull/27742) | Qwen4Exp architecture support; merged at `6c84c7d5`. | Already represented in the fresh comparator. |
| Upstream [#27794](https://github.com/ggml-org/llama.cpp/pull/27794) | `TENSOR_READ_LAZY` plumbing; merged at `fac889fb`. Nathan's branch keeps the missing batched row-prefetch half. | Useful PLE hypothesis. |
| Upstream [#27836](https://github.com/ggml-org/llama.cpp/pull/27836) | Qwen4Exp NextN/MTP draft head; open. Its key note is that the hyper-connection combiner must run per stream; mean pooling first destroys acceptance. | Matches our retained lesson; use it to audit, not re-derive, the Qwen4Exp MTP combiner. |
| Upstream [#26592](https://github.com/ggml-org/llama.cpp/pull/26592) and [#26388](https://github.com/ggml-org/llama.cpp/pull/26388) | hipCUB/CUB paths for top-k/argsort on HIP. | Superseded for our purposes by the in-tree GPU QSA selector; keep as lineage context only. |
| Upstream [#27466](https://github.com/ggml-org/llama.cpp/pull/27466) | ROCm radix top-k for long rows; open. | Confirms the long-context failure mode; hipEngine already uses a device selector. |
| Upstream [#26001](https://github.com/ggml-org/llama.cpp/pull/26001) | Chunked GDN prefill using tensor-core fragments; CUDA/NVIDIA-focused and open. | Hypothesis only. Our nearest unit is the retained colwarps owner plus the rejected decode correction. |
| Nathan's Vulkan evidence pack and branches | q8 KV dequant-once, contiguous K/V, MoE row-list prepass, scale epilogue, SiLU/mul fusion, concat transpose. | Vulkan/RADV-first; useful patterns, not HIP evidence. Map only the algorithmic dataflow, not shader specifics. |
| Upstream [#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | Vulkan q8_0 KV dequant-once for prefill; merged at `dc72703f`. | Not directly applicable to HIP; reinforces "dequantize/reorganize once, then attend". |
| Upstream [#26419](https://github.com/ggml-org/llama.cpp/pull/26419) | MMA FlashAttention at head-dim 256 on RDNA; open. | Relevant to QSA prefill geometry, but measured on RDNA4, not gfx1151. |
| Upstream [#27880](https://github.com/ggml-org/llama.cpp/pull/27880) | qwen4exp graph-split reduction; merged at `6fe74980`. | Already in the remote-HEAD comparator. |
| Upstream [#27925](https://github.com/ggml-org/llama.cpp/pull/27925) and [#26686](https://github.com/ggml-org/llama.cpp/pull/26686) | Vulkan MoE padding/row-ID changes that improve the Vulkan comparator. | Already in the remote-HEAD Vulkan comparator; no HIP action implied. |

### 4.1 Mechanism transfer audit

The useful part of the external forks is the mechanism, not their headline rates.
This is how each mechanism maps to the current hipEngine implementation:

| Mechanism | Current hipEngine state | Campaign action |
| --- | --- | --- |
| Wide, graph-safe QSA radix top-k | Already implemented as the exact stable radix `qsa_topk_expand` owner in `qwen4_exp_qsa.hip`. | No port. Retain it as the reference when long-context QSA is re-profiled. |
| Gather only selected QSA K/V rows | The paged sparse QSA owner consumes explicit selected positions and counts rather than scanning a dense mask. | Audit selected-row count and page locality at 16K+; do not reimplement llama's graph-level gather. |
| Host-side PLE row gather | `Qwen4ExpPLEMMapTable` already gathers and dequantizes only requested rows on the host. | Keep this ownership; add measurement, advice, and batched prefetch in phase P6. |
| Reuse decode graphs | 48 stateless per-layer MoE graphs are reused; stateful GDN/QSA/dense transitions remain direct launches. | Phase P5 must first prove exact state ownership and rollback before enlarging graph scope. |
| Per-stream MTP combiner | The Qwen4Exp MTP sidecar runner normalizes embedding and the widened hidden row independently and executes the one-block hyper-connection path. | Reconfirm against PR #27836, then spend effort on verifier batching and full-suite economics rather than a mean-pool rewrite. |
| q8_0 KV, `-ub 2048`, and `ROCBLAS_USE_HIPBLASLT` | Different llama.cpp representation/config knobs; the current hipEngine baseline is BF16 KV at chunk 512. | Do not import as defaults. A separate declared KV-profile experiment could compare them after AR parity. |
| MoE row-list prepass | Selected grouped/tile-map and rowbatch owners already reuse routed rows and weights. | Fold any missing row-list behavior into phase P2's exact-association design; do not copy Vulkan shader topology. |
| Projection/activation epilog fusion | Existing exact operation-complete Qwen4Exp owners prove the pattern. | Extend it to GR down+inject and activation epilogs in phase P3. |
| Quantized-KV dequant-once and K/V contiguization | Current QSA uses BF16 paged K/V; there is no quantized-KV owner in this campaign. | Backend-disjoint Vulkan evidence only. Do not spend phase capacity on it before AR parity. |
| Tiled transposed concat / SiLU-mul fusion | Prior exact PLE/Conv and operation-complete wins addressed analogous dataflow. | Revisit only if a fresh role trace names the remaining elementwise owner after P1-P5. |
| Fully masked FA slice skip | Current key-parallel QSA flash may expose different slice geometry. | A local reduced-fixture A/B in phase P7; reject if it only helps masked dense rows hipEngine no longer scans. |

## 5. Plan

### Phase P0 — durable baseline and measurement hygiene

Goal: make every future unit start from the committed scripts and a named
comparator instead of `/tmp` state.

- [x] Promote the reusable wall/profile driver, trace analyzer, role
      attributor, sync A/B diagnostic, and report generator.
- [x] Commit the p508 prompt fixture.
- [ ] Add a cold-page-cache PLE mode to the profiling driver, separate from
      warm-cache GPU dataflow claims.
- [ ] Record `rocm-smi` power/clock state and free memory in the next retained
      artifact.
- [ ] Refresh llama.cpp HIP/Vulkan only as a deliberate comparator event.

### Phase P1 — layer-2 high-precision MoE prefill

Goal: remove the largest single-layer miss before touching broad early-layer
numerics.

- [ ] Write the RED fixture for the layer-2 Q5_K gate/up and Q8_0 down shapes.
- [ ] Route the existing selected Q5_K WMMA body for layer 2 and microbench it.
- [ ] Add or select the grouped Q8_0 down path for layer 2.
- [ ] Run the complete 450-row/three-repeat production packet, task gates,
      physical c2, lifecycle, and paired p508/p1012 A/B.
- [ ] Bind only after the full gate passes; retain strict fallbacks and update
      the profile manifest.

Expected evidence: layer-2 role drops from about 397.95 ms toward the llama
role range; p508 gain is bounded at roughly 6.6% before stacking other work.

### Phase P2 — early MoE layers 0-26

Goal: attack the **2.526 s** early-MoE p508 owner without widening a numerical
suffix that already failed.

- [ ] Build a per-layer quant/shape map for layers 0-26 from the registered
      owners and GGUF tensor map.
- [ ] Prototype exact-association multiwarp weight reuse first; then test
      split-precision WMMA where the production envelope permits it.
- [ ] Keep the strict owner registered for every candidate.
- [ ] Gate by category, shape, transition, repeat, task, c2, and manifest.
- [ ] Do not promote from a final-prompt screen alone.

### Phase P3 — decode GR operation-complete fusion

Goal: reduce both decode kernel time and the 1,195 direct launches/token.

- [ ] Fuse GR down+inject using the existing Q8 F32 unequal-pair owner.
- [ ] Add down+scaled-SiLU and up+sigmoid+gated-mean epilogues.
- [ ] Prove exact bits or declare the production T1/T2 class before timing.
- [ ] Measure each fusion stacked on the previous one; retain only measured
      non-regressions.
- [ ] Update the role trace and launch census after each retained fusion.

### Phase P4 — normalized/transposed GDN decode

Goal: replace the measured 2.659 ms/token recurrence with a decode-shaped
layout, without reusing the rejected prefill colwarps path.

- [ ] Normalize Q/K once instead of per column.
- [ ] Keep recurrent state transposed across steps, with explicit strict
      conversion/fallback boundaries.
- [ ] Port the relevant llama four-warp decode recurrence dataflow, not the
      prefill body.
- [ ] Add reduced fixtures, CPU-reference parity, trace evidence, and the full
      profile packet.
- [ ] Do not revive the invalidated decode-colwarps route.

### Phase P5 — state-safe larger decode graph

Goal: reduce the 48 tiny MoE graph launches plus 1,195 direct launches per
token only after state ownership is proven.

- [ ] Audit the historical third-replay state corruption.
- [ ] Capture one state-safe layer or transition first, not the whole model.
- [ ] Include GDN/QSA pointer, state, rollback, and teardown controls.
- [ ] Compare direct-launch API time, graph launches, kernel rows, and exact
      IDs before and after.

### Phase P6 — PLE cold-cache and host-mmap lane

Goal: test the Nathan/Engram PLE mechanism as a separate workload class.

- [ ] Instrument `Qwen4ExpPLEMMapTable.gather_rows()` with row counts, unique
      pages, gather wall, and page-cache state.
- [ ] Add optional random-access advice and batched row prefetch to the
      host-side owner, with an opt-out rollback.
- [ ] Measure cold-cache and warm-cache p508 separately; never mix those rows.
- [ ] Gate output IDs and tracked memory; document that external 3.5x rows are
      64 GB/cold-cache claims, not warm 128 GB expectations.
- [ ] Consider drop-behind load policy only if it improves a retained workload
      without hurting reload-heavy serving.

### Phase P7 — QSA/GDN prefill suffix and long-context follow-through

Goal: revisit lower-Amdahl prefill owners after the dominant MoE and decode
units land.

- [ ] Re-profile QSA after P1/P2 with the same marker roles.
- [ ] Evaluate suffix widening only with fresh all-category boundary packets.
- [ ] Audit whether the selected-position gather already covers the external
      dense-mask problem for our runtime; do not port llama graph code blindly.
- [ ] Evaluate the masked-slice FA skip and head-dim-256 MMA geometry against
      the current key-parallel QSA flash owner.
- [ ] Keep chunked GDN prefill behind an explicit experiment until a local
      packet proves it.

### Phase P8 — MTP economics after AR parity work

Goal: use external MTP evidence as a stretch path without contaminating the AR
parity baseline.

- [ ] Reconfirm the per-stream hyper-connection combiner against the sidecar
      fixture and PR #27836's warning.
- [ ] Build the rows≤8 batch-invariant verifier path before increasing draft
      budget beyond the current 1..4 envelope.
- [ ] Sweep budgets on the full mtp-bench category suite with a true no-MTP AR
      denominator from the same protocol.
- [ ] Record acceptance by category and context depth; do not promote from a
      synthetic repeated prompt.
- [ ] Treat external 39 tok/s MTP rows as a different quant/KV/provider
      configuration, not as the target for this UD-Q4_K_XL/BF16 campaign.

### Phase P9 — closure and rollup

Goal: make every retained result discoverable and reversible.

- [ ] Emit a compact JSON artifact for every accepted, rejected, or blocked
      unit.
- [ ] Update `benchmarks/README.md`, `benchmarks/CHANGELOG.md`, and the
      Qwen3.8-Flash-Next checkpoint row after each retained result.
- [ ] Update `docs/KERNELS.md` and lineage metadata whenever kernel ownership
      changes.
- [ ] Add `docs/REFACTOR.md` entries for temporary flags and rejected paths.
- [ ] Commit each validated logical unit immediately.

## 6. Acceptance criteria

The campaign closes the first milestone when all of the following are true on
the same host, model, quant, and declared profile:

1. p508 and p1012 production prefill meet or exceed the current same-host
   llama.cpp HIP comparator after a deliberate comparator refresh.
2. tg32 production decode meets or exceeds the same comparator without a
   correctness or lifecycle regression.
3. The role-resolved profile no longer shows layer-2 MoE, early MoE, GDN, or
   QSA as unexplained multi-x outliers.
4. The production manifest, strict fallbacks, complete numerical packet, task
   gates, deterministic repeats, physical c2, and teardown gates pass.
5. The final artifact includes exact commands, source/binary hashes, raw trace
   hashes, launch/API/copy census, and a generated report from
   `scripts/qwen4exp_perf_gap_report.py`.

The Vulkan stretch milestone uses the same gates against llama.cpp Vulkan. MTP
is a separate post-parity milestone because it changes provider economics and
requires its own full-suite gate.

## 7. References

- Campaign authority: [`QWEN3.8-FLASH-NEXT.md`](QWEN3.8-FLASH-NEXT.md)
- Benchmark policy: [`BENCHMARK.md`](BENCHMARK.md)
- Testing policy: [`TESTING.md`](TESTING.md)
- Profile contracts: [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md)
- Kernel catalog and port rules: [`KERNELS.md`](KERNELS.md)
- gfx1151 roofline: [`ROOFLINE-gfx1151.md`](ROOFLINE-gfx1151.md)
- Fresh profile artifact:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-fresh-full-profile.json)
- Corrected invalid decode route:
  [`benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json`](../benchmarks/results/2026-08-30-gfx1151-qwen38-flash-next-gdn-colwarps-decode-all.json)
- Durable profiling tools: `scripts/qwen4exp_profile_gap.py`,
  `scripts/qwen4exp_trace_analyze.py`, `scripts/qwen4exp_role_analyze.py`,
  `scripts/qwen4exp_decode_sync_ab.py`, and
  `scripts/qwen4exp_perf_gap_report.py`.
