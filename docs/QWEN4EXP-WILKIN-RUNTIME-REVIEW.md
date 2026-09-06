# Wilkin Strix Halo Runtime Review

Source review: September 6, 2026. No custom runtime installed, no external
benchmark reproduced, and no new hipEngine performance claim.

## Frozen Sources

| Source | Reviewed revision | Relevant paths |
| --- | --- | --- |
| [ROCm experiments](https://github.com/pwilkin/rocm-systems/tree/78d1160060bb6ada29b3b21e20c998a48161b257) | `78d1160060bb6ada29b3b21e20c998a48161b257` | `projects/clr/hipamd/src/hip_graph_internal.{cpp,hpp}`, `projects/clr/rocclr/device/rocm/rocvirtual.cpp`, `projects/rocr-runtime/runtime/hsa-runtime/core/runtime/hsa_ven_amd_graph.cpp` |
| [Site and installer](https://github.com/pwilkin/strix-halo/tree/4d0bf821cab29734dacce5321fcd73add72908c0) | `4d0bf821cab29734dacce5321fcd73add72908c0` | `README.md`, `install.sh`, `data/reproduction-results.json` |
| [Linked llama.cpp branch](https://github.com/pwilkin/llama.cpp/tree/d3b5cc43d1fcfce891f2de94d5274ee40eceb21c) | `d3b5cc43d1fcfce891f2de94d5274ee40eceb21c` | `ggml/src/ggml-cuda/top-k.cu`, `ggml/src/ggml-backend.cpp`, `docs/development/backend-scheduler.md`, `tests/test-backend-sched-ring.cpp` |

The requested [public site](https://pwilkin.github.io/strix-halo/) is mutable.
The installer pins exactly the ROCm and llama revisions above. The September 6
ROCm tip fixes selected-ROCr header precedence in the HIP build; preserve that
fix when constructing an isolated runtime. The preceding gfx1201 register fix
is not evidence of a new gfx1151 kernel optimization.

The ROCm qualification narrative is
[`retained-pm4-command-lists.rst`](https://github.com/pwilkin/rocm-systems/blob/78d1160060bb6ada29b3b21e20c998a48161b257/projects/rocr-runtime/runtime/docs/contribution/retained-pm4-command-lists.rst).
The scheduler's ownership contract is
[`backend-scheduler.md`](https://github.com/pwilkin/llama.cpp/blob/d3b5cc43d1fcfce891f2de94d5274ee40eceb21c/docs/development/backend-scheduler.md).

## What Transfers

This is worth revisiting **in light of our existing PM4 work**, not as an
unimplemented concept. [PM4.md](PM4.md) already records retained-IB submission,
state-register elision, local-cache dependencies, exact graph inspection,
lifecycle gates, and scoped gfx1100 production wins. The native implementation
explicitly rejects non-gfx1100 agents (`hipengine/core/pm4/native.cpp`), and its
separate ROCr queue requires a synchronous HIP-to-HSA boundary. None of its
absolute rates transfer to Framework or Flash-Next.

### 1. PM4 Through HIP's Existing Queue

ROCm commit `04210923fec12e8080fca16885f09de8bb364d0c` adds an opaque prepared
command-list API and HIP graph integration. `CreateGraphPm4Batch` validates
captured kernel packets and capability; `DispatchGraphPm4Batch` publishes the
vendor packet through HIP's existing AQL queue and queue-progress machinery.
This is the most consequential difference from our dedicated-queue transport.
It is a runtime experiment first, not a reason to duplicate our encoder.

The implementation owns each prepared list through a graph packet batch and
retains it during launch. Queue-specific executable bindings retain scratch
leases; dynamic stacks and unsupported ABI/packet shapes reject. The declared
qualification allowlist includes gfx1151. That is the author's qualification,
not ours: reproduce dependency visibility, update/destroy/trim, c2 isolation,
scratch and teardown gates before admission here. Review publication failure
handling explicitly; no fallback after a packet may have executed.

Both designs already enforce conservative dependent-dispatch ordering.
Wilkin documents compute-idle plus scalar/vector cache invalidation on GFX11;
do not present this as discovering a barrier optimization absent from our
stateful/local-cache PM4 path.

**Important measurement trap:** HIP activity tracing disables this PM4 route
in both graph replay and batch creation. A `rocprofv3 --kernel-trace` run can
therefore measure ordinary AQL even with `DEBUG_HIP_GRAPH_PM4=1`. Use a separate
unprofiled, engagement-logged PM4 timing lane and attributable AQL trace.
Record prepared dispatch counts, fallback reasons, loaded DSOs and cold
preparation cost; an environment flag alone is not engagement evidence.

### 2. Graph Update and Batch Construction

Commit `4f1d7ef58959a4026523627b495e2f727318953d` caches topology order,
batches AQL packet rebuilding, and permits kernarg-slot reuse only when no
in-flight reference prevents it. Commit
`2f725f1807a17efd772156649829052012a949f7` optionally merges eligible collapsed
single-stream batches (`DEBUG_HIP_GRAPH_MERGE_COLLAPSED`), preserving a final
join for independent work.

These are plausible cold-capture, rebuild, memory and submission improvements,
not faster Q4 arithmetic. Our `MoeGraphCache` already captures once for stable
keys and replays; count actual graph updates before targeting an update-only
fix. Compare parent HIP graph with custom ordinary-AQL graph before attributing
any gain to PM4. Treat collapsed-batch merging as a separate factor.

### 3. UMA Input Ownership

The llama scheduler change adds a capability-detected input ring, default depth
two, rather than identifying an APU by name. It rotates only while previous
compute can remain in flight, waits before reusing a slot, and invalidates
graph-cache identity when input addresses change. It also pins allocation roots
read asynchronously by another backend and provides a scheduler race checker.
It does not supply per-submission output snapshots.

Our PLE staging already uses pinned buffers, and
`stage_qwen4_exp_ple_rows` documents immediate synchronous H2D consumption.
There is no demonstrated race to fix from this review. Before making that
boundary asynchronous or reading host memory directly on GPU, add delayed-
consumer, slot-wrap, cancellation and c2 tests; track input version, completion
event, graph pointer generation and output consumption separately. Measure
copy/wait cost first and charge extra buffers against context admission.
Do not copy the entire ggml scheduler into our single-backend runtime.

### 4. TOP_K, With Different Semantics

The linked llama merge adds HIP small-k/n-ary selection, specialized k=1
reductions, and multi-CTA radix thresholding. Shape dispatch is version-gated;
preserve the exact code and compiler identity rather than assuming one policy
is optimal on every ROCm release.

Our `qsa_topk_expand_f32_i64_kernel` already uses four radix passes. Its defining
contract is lower block indices at threshold ties, increasing block order,
four-token expansion and tail publication. Wilkin's
`top_k_parallel_radix_gather` uses atomic output/equality counters, so copying
it would not preserve our deterministic tie subset or ordered attention input.
Its general signed-float key mapping also needs comparison with our explicitly
finite, non-negative QSA score domain.

The useful hypothesis is **multi-CTA threshold computation followed by our
stable compaction**, only if fresh long-context selection timing pays for its
extra launches/scratch. Test 513 through 65536 pooled blocks, varied row counts,
all-equal/threshold ties, repeated runs and exact selected positions/counts.
MoE's 512-expert/top10 route and vocabulary k=1 are separate owners with
different tie/output contracts; measure each complete owner before porting.
Include NaN/infinity/signed-zero policy where the target owner admits them;
do not widen the QSA finite-score domain just to reuse a generic kernel.
This is not evidence that the dominant FFN projection costs are solved.

## How To Revisit PM4

Keep prefill first, as requested. The first follow-up is a read-only
graph-eligibility and overhead census on current Framework production:

1. Count captured/replayed/updated nodes, kernel-only batches, host/copy
   boundaries and fallback reasons, split by prefill and decode. Name the
   current commit, UD-Q4_K_XL/BF16 KV, profile, capacity and fixture.
2. Establish the exposed submission cost and Amdahl ceiling. PM4 does not
   reduce the instruction/traffic cost inside our slow Q4/Q5/GR kernels.
   Do not rank it above those owners from an external headline alone.
3. If the census justifies it, build only the necessary HIP/ROCr libraries in
   a private prefix. Keep the current therock compiler and model fixed; do
   not run the site's model installer, replace `/opt/rocm`, alter global
   library resolution, or enable unqualified-ASIC overrides.
4. Measure separate arms: current-runtime HIP graphs; custom-runtime ordinary
   AQL graphs; custom-runtime retained PM4. Keep graphs enabled in all three.
   Add collapsed-batch merging as a separate on/off experiment if engaged.
   Log actual DSO paths/build IDs and unsupported-path fallback counts.
5. Run exact dependency and lifecycle tests, then full logits/state/KV and
   category/c2 gates. For production retention use our canonical 12-case
   same-host A/B and context-transition gates, keeping cold build/capture,
   persistent memory and decode tradeoffs visible. Do not use profiler timings
   as evidence that PM4 itself ran.
6. A native in-tree gfx1151 transport is a **separate** peer-encoder/registry
   qualification under PM4.md, not removing the gfx1100 guard. Prefer reusing
   our inspection/tests; do not reopen known reset-risk queue-recreation tests
   or claim ROCm #6529 resolved by this review.

The site launcher is **not** the three-arm harness above:
`ENABLE_RETAINED_PM4=0` also sets `GGML_CUDA_DISABLE_GRAPHS=1`. That compares
PM4 graphs with graph-disabled execution, not PM4 versus ordinary HIP graphs.
An independent harness must control graph enablement directly.

## External Evidence Limits

The site reports a two-pass, 31497-token prompt/256-output comparison on its
Radeon8060S: Qwen3.8-27B IQ4_XS plus DFlash2, not Flash-Next. Its same-model
Vulkan arm is useful within that experiment; the ROCmFP4 arm changes both
target and drafter. Output hashes repeat within each arm but differ between
the published HIP and Vulkan arms. Coherent text and equal acceptance are
not full-logit equivalence or our numerical/task gate.

The ROCm design note separately reports PM4 versus custom AQL decode gains of
17.10% on Ornith35B-A3B IQ4_XS and 1.72% on Nanbeige3B BF16, with bounded
full-logit/lifecycle evidence. The variation reinforces the need to measure our
exposed dispatch cost. Neither result establishes a Flash-Next prefill gain,
our native-context correctness, or parity with pinned halo-box Vulkan.

## Decision

Add the runtime/graph census and qualified three-arm PM4 revisit to the
campaign; retain UMA ownership and semantic-safe TOP_K as measured-cost-gated
follow-ups. Keep the existing halo-box Vulkan target and binding quant.
No dependency, runtime default, architecture admission or quality gate changes.
