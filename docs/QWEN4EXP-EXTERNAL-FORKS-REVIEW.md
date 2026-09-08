# Flash-Next External Fork Review

Reviewed September 8, 2026 using GitHub's API through `gh`. This is a
source/documentation review, not reproduced performance or hipEngine admission.
No external runtime, weights, container or kernel was installed or executed.
The [campaign](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md) retains
UD-Q4_K_XL/BF16 KV, its profile gates and prefill-first priority.

## Pinned Sources

| Repository | Reviewed tip | Publication boundary |
| --- | --- | --- |
| [Halogen](https://github.com/peonist-ai/halogen-flash-server/tree/0208581a5a0c54c53fc11f2e9e06622f168f5b6f) | `0208581a5a0c54c53fc11f2e9e06622f168f5b6f`, September 4 | Public deployment/docs/benchmark scripts; engine is container-only. Version 0.4.4 documentation preserves earlier benchmark version attribution. |
| [Myhacsint production branch](https://github.com/myhacsint/llama.cpp/tree/2dff8596dcb7bdf765d24d44e1155d51f04c82b7) | `2dff8596dcb7bdf765d24d44e1155d51f04c82b7`, September 1 | Requested `production/strix-halo-qwen4exp-b10685`, not its upstream-tracking master. Squashed source snapshots plus a shared-MTP fit fix. |
| [Halo-box master](https://github.com/halo-box/strix-llama.cpp/tree/7449a0fe9710ab584c5f9a6d25e7a31eea2708b8) | `7449a0fe9710ab584c5f9a6d25e7a31eea2708b8`, September 7 UTC / September 8 JST | 66 commits ahead of our `b212548e0` comparator at review time. Includes HIP PR18, UMA input-ring PR29 and ROCmFPx PR30. |
| [PR18 results](https://github.com/gaetan-puleo/strix-halo-pull-18-results/tree/c09bef366291556235b7bc473005a95483cb3eac) | `c09bef366291556235b7bc473005a95483cb3eac` | Author's separate benchmark and per-model correctness tables, not a run at the latest halo-box master. |

Representative discovery commands; use pinned refs for subsequent file reads:

```bash
gh api 'repos/peonist-ai/halogen-flash-server/commits?per_page=20'
gh api 'repos/myhacsint/llama.cpp/commits?sha=production%2Fstrix-halo-qwen4exp-b10685&per_page=25'
gh api 'repos/halo-box/strix-llama.cpp/compare/b212548e0...7449a0fe9710ab584c5f9a6d25e7a31eea2708b8'
gh api repos/halo-box/strix-llama.cpp/pulls/18
gh api repos/halo-box/strix-llama.cpp/pulls/29
gh api 'repos/halo-box/strix-llama.cpp/contents/ggml/src/ggml-cuda/mmq.cuh?ref=7449a0fe9710ab584c5f9a6d25e7a31eea2708b8' -H 'Accept: application/vnd.github.raw+json'
```

## Headline Audit

### Halogen: Different Weights and Speculative Decode

The pinned [README](https://github.com/peonist-ai/halogen-flash-server/blob/0208581a5a0c54c53fc11f2e9e06622f168f5b6f/README.md)
attributes its engine rates to earlier 0.2.0 measurements, carried into the
0.3.0 table: Ryzen AI Max+ 395 / Radeon 8060S, 128 GB, ROCm 7.14,
custom `.hgn` checkpoint plus quality overlay, tuned GEMM plan and native
262144 capacity. Author-reported cold real-text prefill is about
1175/1309 tok/s at 8192/32768 tokens. Short-context serial decode is
37.6 tok/s at context1500; 42.4 prose / 48.3 code uses MTP. Its served
32K/256-output ten-prompt speculative row is 41.7 tok/s.
These are not a 50-tok/s serial-AR result or an apples-to-apples comparator.

The [precision document](https://github.com/peonist-ai/halogen-flash-server/blob/0208581a5a0c54c53fc11f2e9e06622f168f5b6f/docs/QUANT.md)
describes a 115.55-GiB base file, groupwise 4-bit trunk/experts, FP8 PLE and
a 2.31-GiB replacement overlay. The overlay recalibrates non-expert tensors
and promotes twelve output projections to 8-bit; the full tensor quant map
is not published. This is not our GGUF payload. Its quality evidence includes
short teacher-forced comparisons, corpus PPL and retrieval; it does not
establish our strict-teacher/category/state/lifecycle contract. Its language
about original BF16 goldens must not replace this campaign's FP8 teacher
identity without independently verifying the actual reference export.

The public tree has no engine kernel source or release-gate implementation.
Commit history starts with a flattened release, then mostly deployment,
sampling, cache, memory and API fixes. The [license](https://github.com/peonist-ai/halogen-flash-server/blob/0208581a5a0c54c53fc11f2e9e06622f168f5b6f/LICENSE.md)
is a custom EULA, not the MIT source license of the llama forks. Treat it as
a black-box comparison and public design-hypothesis source; this review does
not authorize copying/disassembling/reimplementing proprietary kernel bodies.

Useful documented mechanisms in [FLAGS.md](https://github.com/peonist-ai/halogen-flash-server/blob/0208581a5a0c54c53fc11f2e9e06622f168f5b6f/docs/FLAGS.md):
fixed per-shape GEMM plans, large prefill chunks, resident weight pinning,
page-cache-aware KV budgets, in-place KV retention with small recurrent-state
snapshots, and switching from singleton speculation to batched AR under load.
These are hypotheses; their implementation and isolated causal gains are not
publicly auditable.

Benchmark traps from the public [size-sweep script](https://github.com/peonist-ai/halogen-flash-server/blob/0208581a5a0c54c53fc11f2e9e06622f168f5b6f/tools/halogen-bench.py):

- Its prefill input repeats one filler sentence; it is not the README's
  separate real-text engine-prefill benchmark. MoE routing and PLE locality
  can depend on content, so do not adopt the comment asserting content
  independence.
- It repeats the same requests without explicitly disabling/resetting server
  prompt caching. Default cache mode2 may reuse prefixes and permits
  cold/resumed output differences. A cold comparison must set cache0 and
  verify processed-token counts; exact resumed comparison needs cache1.
- Its TG samples divide completion tokens by entire HTTP request wall,
  including prefill, and aggregate arithmetic-mean rates. Preserve raw
  token/time totals and distinguish request throughput from decode-only TG.
- Serial and MTP must be explicitly selected and measured on the full
  category/heldout suite. Do not compare its code-heavy MTP maximum to our
  true AR or import its public cross-machine competitor ratios.

### Myhacsint: Reviewable Snapshot, Unverified Headline

[PRODUCTION-SNAPSHOT.md](https://github.com/myhacsint/llama.cpp/blob/2dff8596dcb7bdf765d24d44e1155d51f04c82b7/PRODUCTION-SNAPSHOT.md)
pins the b10685 source tree, binary SHA, Fedora/RADV environment and build
command. It documents a UD-IQ4_XS target with shared Q8_0 MTP and also a
separate uncensored derivative. The public deployment preset and benchmark
prompts are intentionally omitted. The reviewed snapshot/provenance/MTP docs
do not substantiate the supplied approximately 60 TG / 600 PP headline with
a complete reproducible protocol; retain it as user-reported, unverified.

[PATCHSET.md](https://github.com/myhacsint/llama.cpp/blob/2dff8596dcb7bdf765d24d44e1155d51f04c82b7/PATCHSET.md)
credits Nathan's Vulkan line, apepojken, Unsloth and upstream. This is not a
few independent performance commits to cherry-pick. Useful source deltas:
per-expert tile selection, GQA gathered attention, lazy PLE prefetch, alias
dependency handling, recurrent restore and shared target/draft ownership.

Source inspection of
[ggml-vulkan.cpp](https://github.com/myhacsint/llama.cpp/blob/2dff8596dcb7bdf765d24d44e1155d51f04c82b7/ggml/src/ggml-vulkan/ggml-vulkan.cpp):

- `ggml_vk_mul_mat_id` selects tiles using expected expert rows
  `ceil(tokens * active_experts / experts)`, not the full prompt width.
  It is an average, not actual skew-aware expert populations.
- `ggml_vk_flash_attn_gather_compact` accepts small batches below64 and
  handles separate GQA K/V, but the multiple-KV-head/separate-V case requires
  **F16** at this pin. It is not a demonstrated BF16-KV replacement.
- Its stale union-count estimate selects a performance path, while scratch
  remains worst-case sized. This does not justify stale ownership metadata,
  nondeterministic route selection or changing our ordered selected positions.

The public [rollback test](https://github.com/myhacsint/llama.cpp/blob/2dff8596dcb7bdf765d24d44e1155d51f04c82b7/tests/test-recurrent-state-rollback.cpp)
replays after full/partial checkpoint restore, including a destination already
holding another history. That dirty-destination case is worth checking against
our coverage. It uses one prompt, rollback3 and tolerance1e-5, and can skip;
it is not our full rejection-depth or exact-state qualification.
Its `fabs(a-b) > eps` comparisons do not explicitly reject nonfinite logits.

The [shared-MTP fit fix](https://github.com/myhacsint/llama.cpp/commit/10bb3cff8fe0d58cc8a21a6ad34c4956f876c87f)
uses target metadata during no-allocation draft sizing, resolving omitted
shared tensors without allocating/counting weights twice. Reuse the ownership
idea only where our planner lacks equivalent coverage, not as a new decode
kernel or evidence that adaptive MTP is profitable here.

### Halo-box: Real New Source, Different Benchmark Payload

The [PR18 wide table](https://github.com/gaetan-puleo/strix-halo-pull-18-results/blob/c09bef366291556235b7bc473005a95483cb3eac/results-all-wide.md)
contains the supplied approximately 800 PP / 30 TG class of result:
Flash-Next **UD-IQ4_XS**, depth0, HIP/ROCm7.14 on Ryzen AI Max+395/8060S,
ubatch2048: PP2048 **844.30 +/- 15.18**, PP4096 **804.18 +/- 11.09**,
TG128 **26.86 +/- 0.00 tok/s**. The published protocol is
`-b 2048 -ub 2048 -fa 1 -ngl 999 --load-mode mmap -r 4`, retaining
repetitions2-4. These are author rows, not our UD-Q4_K_XL/BF16-KV fixture
or proof of current Vulkan performance. Complete KV/cache/clock and token
fixture identity must be bound in our reproduction.

[PR18](https://github.com/halo-box/strix-llama.cpp/pull/18) reports broad
backend/model testing, but the [results README](https://github.com/gaetan-puleo/strix-halo-pull-18-results/blob/c09bef366291556235b7bc473005a95483cb3eac/README.md)
and [per-model table](https://github.com/gaetan-puleo/strix-halo-pull-18-results/blob/c09bef366291556235b7bc473005a95483cb3eac/model-correctness-vs-master.md)
disagree about all-model logit exactness. The Flash-Next table row says
logits non-exact, max absolute difference3.68859, NMSE0.0728605, while
tokenization and generated decode match. PR prose separately mentions a
controlled one-token exact check and different backend test counts.
Keep these evidence scopes separate; neither token equality nor backend NMSE
substitutes for our full-vocabulary production gate.

High-value inspected commit mechanisms:

| Commit / source | What is worth testing | Constraint from its own history |
| --- | --- | --- |
| [`90ad6cd267`](https://github.com/halo-box/strix-llama.cpp/commit/90ad6cd26753b1eab62ff3ee39e17bfdcca3b6b5), `mmq.cuh`, `mmq-load-tiles.cuh` | Prefetch next-K activations and selected Q8 weight tiles into registers while computing current LDS tiles; preserve arithmetic order. | Whitelisted by quant/tile/fallback because extra registers spill elsewhere. Their tests include packing/ID overhead. Our rejected scalar Q8-prefetch2 and Q4-block-pair paths are not automatically reopened. |
| [`d7550c14bf`](https://github.com/halo-box/strix-llama.cpp/commit/d7550c14bf36e2b6a48265c1a07d399ce07fce9e), final `mmq.cuh` | Compact expert grids and row-range-specific J16/J32/J48/J128 choices, including remainder batches. | Final source also covers IQ expert types and 512/top10. Our weights, skew distribution and already compact tile maps differ; copy neither thresholds nor IQ4 gains blindly. |
| [`ab55b8fdc3`](https://github.com/halo-box/strix-llama.cpp/commit/ab55b8fdc393d7afa83a47354bf2a24939286946), `mmvq.cu`; [`37f02eb14e`](https://github.com/halo-box/strix-llama.cpp/commit/37f02eb14ec5d58461596fb9f0ce6314cf6fb2d4) | Fuse activation quantization into single-column Q8_0/Q6_K matvec, sharing staged activations inside a CTA and preserving its parent dot/reduction sequence. | Their Q8_1 quantizer is not our three-plane/strict arithmetic. Charge duplicated packing across CTAs; preserve alignment, tail, zero and nonfinite contracts. |
| [`f8524a5033`](https://github.com/halo-box/strix-llama.cpp/commit/f8524a5033018e93a8aec856a93ba8c054b468b9), `mmvf.cu` | Issue four iterations of F32 weight/activation loads before consuming them in original order. | Relevant only if our surviving router/inject/alpha/beta leaf has the same load-latency deficit; no general router speedup assumed. |
| [`6130b7262a`](https://github.com/halo-box/strix-llama.cpp/commit/6130b7262ae97d353556903e0175b8993db77bef), final `hyperconn.cu` | Ordered stream mixing/combine with explicit separate mul/add rounding and overlap-aware scratch; consider producer/consumer fusion under R5. | [`930a8bdad3`](https://github.com/halo-box/strix-llama.cpp/commit/930a8bdad3d6a1ed7011df038962a188e7432606) removes decode HC-mix and mixed-F32 grouped experiments, and tightens external-consumer/alias guards. Do not resurrect removed routes as validated examples. |

Correctness additions are at least as transferable as kernels:

- [`551ce30fe5`](https://github.com/halo-box/strix-llama.cpp/commit/551ce30fe5c371258333d5681be76b8f3f440d65)
  restores the batch-one reduction topology after wider waves changed output.
  [`312ea53c6c`](https://github.com/halo-box/strix-llama.cpp/commit/312ea53c6c637cff78df25bbd5e6bbb26b3c76be)
  fixes zero-input fused quantization and reads of uninitialized prefetch tails.
  Include these RED cases before adapting the associated mechanisms.
- [`d22fa655a3`](https://github.com/halo-box/strix-llama.cpp/commit/d22fa655a32c73196cda7a7fa3dace2f17901609)
  retracts unmeasured matvec chunking for most quants: rereading dense weights
  harmed verification. Compare exact single-sweep controls before chunking.
- [`c69f0e8cc6`](https://github.com/halo-box/strix-llama.cpp/commit/c69f0e8cc603ac5c68b0aab1b84b26db9a1e9af2)
  restores MTP recurrent rollback slots after checkpoint fallback caused
  replay forwards. Audit target-forward counts per accepted cycle, not just
  accepted drafts; do not transfer its dense-27B gain to Flash-Next.
- [PR29](https://github.com/halo-box/strix-llama.cpp/pull/29) adds input-ring
  rotation on graph reuse and allocation-root lifetime tracking. Its merger
  explicitly did not compile/run CPU or GPU tests. The pinned
  `tests/test-backend-sched-ring.cpp` only checks a no-extra-sync path with a
  single CPU-like mock, not delayed asynchronous ring wrap. Our
  [Wilkin review](QWEN4EXP-WILKIN-RUNTIME-REVIEW.md) already owns this hypothesis.

Open [PR17](https://github.com/halo-box/strix-llama.cpp/pull/17), head
`792acdfd09bbe10f6d7f509e50d338f6f26ec890`, is a separate unmerged Vulkan
lane at review time. Its prefill claims cover dense Qwen3.8-27B, Qwen3.6,
Coder and truncated DSv4, not a Flash-Next qualification. Do not attribute its
dequant-once KV, tile, wave32 or concat changes to current master unless the
specific code is independently found there. ROCmFPx additions likewise
describe new weight formats, not native IU4 acceleration of our Q4_K payload.

## Campaign Experiments

Do not restart completed R4 work or interrupt current Q5_1 admission.
Attach these experiments to the ranked queue, then rerank from the next
complete-owner refresh. The R2 ledger predates subsequent candidate work;
external headline ratios do not supply current recoverable milliseconds.

| ID / rank attachment | First bounded experiment | Advance / stop rule |
| --- | --- | --- |
| E0 / comparator prerequisite | Pin new halo-box HIP and Vulkan binaries and the requested Myhacsint branch; reproduce our exact UD-Q4_K_XL/BF16-KV, p512/p1024/p4096-TG128 fixture with logger off. Separate serial AR/MTP, cache modes and launch-blocking policy. | Keep `b212548e0` as historical evidence; do not overwrite it or freeze a new target before same-host correctness and counterbalanced measurements. A separate IQ4_XS or Halogen product row cannot close this lane. |
| E1 / R4, R8, R7 prefill | Inspect current admitted WMMA/MMQ wait cycles and spills, then screen register-staged next-K loads and useful-expert-row tile scheduling against the current chain. | Require actual routing/activations, tail cases and operation-complete packing/map/repair cost. Preserve accumulation order and registered fallback. Stop if extra registers or padding dominate. Prefer this representation-preserving screen over expanding IU4 scope when its measured saving is larger. |
| E2 / R6 decode | Attribute remaining shared-input Q8/Q6 projections and quantize launches. Screen grouped projection or fused quantize+matvec only where not already fused; separately screen F32 ordered prefetch. | No single-plane substitution for multi-plane/exact parents. Include zero/tiny/nonfinite and tail REDs, exact control, c2/alias tests and complete model gate. Keep MoE's largest measured gap first unless new costs reorder it. |
| E3 / R5 GR | Compare surviving ordered hyper-connection mix/combine and normalization producer boundaries with `hyperconn.cu`. | Preserve every F32 nonlinear/BF16 publication boundary and external consumer. Include overlapping outputs and restored state. Reject any route relying on the removed decode-HC experiment or implicit FMA contraction. |
| E4 / R7 long-context prefill | Audit current chunk/shape dispatch and eligible library GEMM algorithms; test immutable per-shape choices, then larger chunks only with bounded scratch and state equality. | Halogen's fixed-plan/large-chunk story is documentation-only. Existing chunk1024 and exact linear rejections stay binding until a different mechanism justifies a screen. Include c2 peak/admission, first-token wall and fairness; no per-start timing-selected arithmetic or cache-hit PP inflation. |
| E5 / R3 followup, R6 | Extend the existing UMA audit with delayed host-input consumers, ring wrap, pointer-generation changes, cross-view lifetime, cancellation and ordered output consumption. | No generic scheduler port or asynchronous rewrite without an exposed-cost bucket. PR29 is not a PM4 implementation or proof that our drift is a race; do not remove synchronization on that inference. |
| E6 / P11 MTP | Count duplicate target replay forwards at each rejection depth; test full/partial restore into dirty destinations and shared target/draft allocation accounting. | Complete true-AR category/heldout economics and exact state/control remain binding. Capacity beyond1K and batch-invariant verification precede budget/confidence tuning. No headline-driven MTP default. |
| E7 / separate serving-memory followup | Measure PLE page-cache pressure, pinned resident bytes, startup fragmentation, and ownership-preserving in-place prefix snapshots. | First compare with existing PLE/prefix ownership. Charge all retained KV/state and additional buffers; preserve cache-hit/cold policy and isolation. No host sysctl/THP change, Halogen installation or new weight format is authorized by this review. |

Already covered mechanisms include radix QSA selection, gathered decode,
incremental pooled keys, host PLE gathering, grouped experts/weighted down,
device checkpoints and the GDN concat problem. See the
[earlier idea audit](QWEN3.8-FLASH-NEXT.md#03-next-units-priority-order),
the campaign's September 7 Strix followups and the Wilkin review before adding
duplicate dispatch routes. New evidence may justify better coverage or
scheduling, not recreating those features.

Every kernel experiment still requires in-tree work, source-file/commit
lineage review, a CPU-reference/parent oracle, named cache-only profiling,
applicable numerical/state/category/lifecycle gates and same-host complete
operation/model timing. Public "75/80/90% theory" labels have no defined
payload/byte/operation denominator here and do not enter our Amdahl ledger.
