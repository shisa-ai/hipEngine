# hipEngine Q3 optimization review

**Date:** 2026-07-19 review; implementation status updated 2026-07-20

**Scope:** retained qwen-kernel/hipEngine Q3 analysis plus measured implementation status

**Primary target:** Qwen3.6-35B-A3B `UD-Q3_K_M` on gfx1100

**Provenance:** the original review below was read-only; the status section records later in-tree work without rewriting the source-derived analysis

## Current implementation status (2026-07-20)

The original profile and recommendations remain useful provenance, but they no
longer describe the current default path. The campaign has completed the first
raw-IQ, decode, and scheduler tranches:

- Expert-major raw IQ3/IQ4 prefill is default-on with an exact direct fallback.
  In the retained native-attention profile it reduced selected-IQ time
  `994.668 -> 613.995 ms` (-38.27%) and total kernel sum
  `4,396.145 -> 4,078.667 ms` (-7.22%), although the full 512 wall result was
  flat within run spread because the old row scheduler dominated.
- The temporary native-attention quarantine is gone. The quant-axis
  `gguf_ud_q3_k_m` plugin now preserves decode-order GDN and full-attention
  arithmetic while running the whole prompt in bulk. Mixed-64 hidden rows and
  logits are bit-exact to token serial, as are the full 4K logits. The first
  retained 4K result was `211.936 tok/s`, versus `10.907 tok/s` for the same
  token-serial oracle.
- Dense raw-Q8 prefill has completed three exact profile-driven steps. The
  existing eight-output schedule first raised 512/mixed-4K to
  `364.414/342.902 tok/s`; bounded 8x2/8x4 row tiles then reused each encoded
  weight row across prompt rows and reached `573.288/523.321 tok/s`. Every dot
  keeps the old K traversal and reduction association, primitive outputs are
  BF16-bit equal, and the full-model exact gates remain green.
- The first post-Q8 selected-IQ step specializes grouped IQ4 down's production
  `K=512` shape from local128 to one wave: only lanes 0–15 ever owned work, so
  the old three extra waves reduced zeros. The exact local32 key raised
  512/mixed-4K again to `693.325/613.576 tok/s`, cut traced IQ4 down
  `1,666.039 -> 502.039 ms` (-69.87%), and retained the general local128 and
  direct selected fallbacks.
- Reprofiling returned dense Q8 to first. The third exact Q8 step widens output
  reuse from 8x4 to 16x4 only at measured row thresholds, reaching
  `707.420/626.077 tok/s` at 512/mixed-4K. The exact 8x8 alternative regressed
  every production shape and was removed. A subsequent exact 32x2 schedule was
  also removed after regressing all nine production shapes by 8.28–89.99%
  despite matching 16x4's VGPR/LDS/scratch resources. A wave-uniform Q8-scale
  broadcast then regressed the same shapes by 38.77–191.80% because its sixteen
  shuffles per K iteration outweighed uniform cached scale loads. Short-row and
  unaligned shapes retain the 8x4/8x2/pack8 fallbacks; local exact-Q8 scheduling
  is closed at 16x4 pending a new profile-backed premise.
- Exact GDN now uses a corrected-contract 32-value-column LDS schedule instead
  of the 4,096-block K2 reduction leaf. Each lane executes the same ordered
  128-term KV contraction, state FMA, and eight-term output groups while a
  conflict-free 16 KiB LDS tile keeps recurrent state across the token loop.
  Recurrent output and final state are bit-exact at 64/512/4,096 rows; the full
  hidden/logit gates remain exact. The new default reaches
  `763.221/670.417 tok/s` at 512/mixed-4K and cuts traced GDN
  `1,310.186 -> 882.716 ms` (-32.63%) with local32, VGPR248, and zero scratch.
- Grouped IQ3 now batches four independent compact rows through one pair of
  block barriers while preserving every row's lane dot, wave32 shuffle tree,
  and serial wave-0..7 accumulation. A measured auto crossover keeps RT1 below
  four rows/expert. Production outputs are BF16-bit exact, the final trace cuts
  IQ3 `1,093.856 -> 961.231 ms` (-12.12%), and mixed-pattern 4K moves
  `670.417 -> 684.499 tok/s` (+2.10%) with local256, VGPR80, and zero scratch.
  The 512 counterbalanced aggregate is only +0.25% and remains within spread;
  the traced exact sub-window, not a large 512 headline, is the promotion basis.
- Exact split attention now uses its existing grouped-GQA producer from two
  prompt rows while retaining per-Q-head warp split for a singleton below
  context 4,096. The prefill batch supplies enough independent blocks to reuse
  each K/V stream across eight Q heads without changing per-head arithmetic.
  Fifteen primitive shapes are BF16-bit exact; full 4K logits are bit-exact;
  traced attention falls `936.900 -> 464.773 ms` (-50.39%), and mixed-pattern
  4K reaches `741.180 tok/s` (+8.28%) with unchanged launches and memory.
- A final exact GDN data-boundary candidate removed prompt-sized normalized
  Q/K/V materialization and read raw rows directly from `conv_out`. It was
  bit-exact at 64/512/4,096 rows and cut prepare `16.150 -> 7.995 ms`, but the
  repeated normalization multiplies raised recurrence `866.824 -> 889.363 ms`.
  The complete GDN pair regressed 1.63%, total kernel sum 0.44%, and trace span
  0.24%; matched 4K wall was flat at +0.07%. The candidate and tests were
  removed, closing this direct-conv premise.
- Task #27 then changed the dense-Q8 algebra instead of retrying an exact local
  schedule. A llama.cpp-derived 128x128 K256 integer-MMQ body consumes raw Q8_0
  weights and three residual D4 activation planes. One-pass MMQ, tail-layer
  heuristics, unguarded D4x2/D4x3, and tighter `3e-6`/`6e-6` repair policies all
  failed model gates. The retained `1e-5` policy queues values near BF16 rounding
  boundaries and recomputes them with the exact raw-Q8 reduction; only
  `(K,N)=(2048,8192)` from 32 rows and `(2048,4096)` from 48 rows are admitted.
  All other shapes keep exact 16x4/8x4/8x2/pack8 fallbacks. The final 18-workload
  by 9-position text/decode suite is logit-bit-exact (`KL=0`, top-1 `1.0`).
  Post-hardening official medians reach `848.543/828.003 tok/s` at
  512/repeated-4K, while matched mixed-pattern 4K is
  `743.906 -> 831.393 tok/s`. D4x3 reuses bulk
  scratch; the bounded risk queue raises tracked peak by 16/128 MiB to
  `15.821/17.080 GiB` at 512/4K.
- Decode work retained the wave-uniform IQ3 address cleanup and aggregate
  MoE-tail/next-RMS fusion, rejected hierarchical top-k, IQ4 tile4, and routed
  stream overlap, and currently measures `101.216 tok/s` at 512 and
  `108.383 tok/s` at 4K on GPU1. The final-tree profile put dense Q8 first, so
  task #32 tested one source-derived block-serial raw-Q8 mapping. The fast
  association cut representative leaves 34–55% but failed exact full logits at
  512/1K/4K; preserving the current reduction association made those leaves
  21–80% slower. Both forms were removed. The source-derived 150–190 tok/s
  feasibility argument remains valid, but this bounded HIP campaign is closed
  without reaching that band.

The guarded-MMQ cache-only 4K trace is now the active prefill Amdahl ledger:

| Family | Time | Share of 4,815.413 ms kernel sum |
|---|---:|---:|
| Dense Q8 total | 1,569.232 ms | **32.59%** |
| ├ exact fallback | 753.045 ms | 15.64% |
| ├ guarded D4x3 MMQ | 602.924 ms | 12.52% |
| ├ sparse exact repair | 207.578 ms | 4.31% |
| └ residual quantize | 5.685 ms | 0.12% |
| Grouped IQ3 gate/up | 1,075.210 ms | **22.33%** |
| Exact GDN recurrent | 741.083 ms | **15.39%** |
| Grouped IQ4 down | 491.646 ms | **10.21%** |
| Full attention | 445.168 ms | **9.24%** |

The changed algebra cuts dense Q8 `2,052.066 -> 1,569.232 ms` (-23.53%), total
kernel sum `5,350.508 -> 4,815.413 ms` (-10.00%), and trace span
`5,534.073 -> 4,973.718 ms` (-10.13%). Dense Q8 remains first in aggregate and
grouped IQ3 is second; their already-rejected local schedules are not reopened.
Task #27 closes this bounded source-faithful MMQ tranche with an explicit
quality and memory contract. **The ~3,000 tok/s objective remains open:**
~825–850 tok/s is a retained architectural step, not a target claim, and the
empirical BF16-boundary guard must not be weakened below `1e-5`. Evidence is in
[`benchmarks/results/2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json`](../benchmarks/results/2026-07-20-gpu1-q3-guarded-d4x3-mmq-prefill.json),
with the preceding exact-attention and final exact-GDN rejection in
[`benchmarks/results/2026-07-20-gpu1-q3-exact-attn-gqa-batch-prefill.json`](../benchmarks/results/2026-07-20-gpu1-q3-exact-attn-gqa-batch-prefill.json)
and
[`benchmarks/results/2026-07-20-gpu1-q3-exact-gdn-direct-conv-rejected.json`](../benchmarks/results/2026-07-20-gpu1-q3-exact-gdn-direct-conv-rejected.json).

## Current post-prefill roadmap

This section supersedes the historical task numbers and dependency language in
the source-derived review below. The measured state is:

- **c=1 prefill, GPU1 RX 7900 XTX:** guarded D4x3 MMQ reaches
  `848.543 tok/s` at 512 repeated tokens, `828.003 tok/s` in the official
  repeated-token 4K run, and `831.393 tok/s` in the matched mixed-pattern 4K
  A/B. The `1e-5` sparse exact-repair policy passes the 18-workload continuation
  gate bit-for-bit and retains exact fallbacks for every non-admitted shape.
  Task #27's bounded changed-algebra tranche is complete; the ~3,000 tok/s
  objective remains open to a future, separately justified architecture.
- **c=1 decode, GPU1 RX 7900 XTX:** retained rows remain `101.216 tok/s`
  at 512 and `108.383 tok/s` at 4K. Task #32 is complete/no-hold: its final-tree
  D0 ranked dense Q8 first, but the only new source-backed block-serial premise
  either changed logits or regressed exact leaves. No candidate code remains.
  These current-code rows have not been rerun on GPU0/W7900.
- **c=N GGUF, GPU1 RX 7900 XTX:** task #29 is promoted. One resident weight set
  owns `[C,...]` token/hidden/logit scratch, per-slot linear state, paged KV
  ranges, and `KVLiveSpans`; indexed Conv/GDN, row-batched paged attention,
  selected-row MoE, row lm-head/argmax, and C/context HIP graphs advance C=2/4/8
  natively. Greedy prompt lists use stable scheduler ids, EOS/length reclaim,
  state/KV compaction, readmission into freed slots, full shape-key graph caches,
  per-request timestamps, and explicit no-serial-fallback provenance. C=8 is
  `207.780 tok/s` at 512/128 and `211.177 tok/s` at 4K/128, or `2.053x/1.948x`
  retained c=1 aggregate, with exact IDs/full-logit gates and native rocprof
  symbols. Evidence: [`benchmarks/results/2026-07-21-gpu1-q3-native-cn-retained.json`](../benchmarks/results/2026-07-21-gpu1-q3-native-cn-retained.json).
  This is synchronous in-call prompt-list scheduling; persistent cross-call HTTP
  admission, elastic KV/prefix sharing, cancellation, and non-greedy row
  sampling remain outside task #29.
- **GGUF MTP:** tasks #30 and #31 are complete under the locked shared ABI.
  The 40-layer AR map still excludes trailing blk.40; a separate strict tensor
  map/materializer owns its 20 draft tensors and target embedding/output
  fallbacks. Native raw Q3_K selected single/dual-SiLU kernels execute gate/up,
  the existing Q4_K/Q8_0/full-attention paths execute the rest of the block,
  and `Qwen35GGUFNextNDraftProvider` returns candidate-only `DraftBatch` rows.
  The target consumes root-prefixed `TargetVerifyBatch` rows with
  `KVLiveSpans(span_role="verify_chain")`, GPU accept/CPU-oracle parity,
  scheduler-owned transactions, per-row state snapshots, accepted-prefix KV
  publication, rollback, and full shape-key stable-buffer buckets. B=1/2/3
  logits, reject/partial/full commits, and real greedy output are exact. Matched
  GPU1 D16 ratios are only `0.544x/0.346x/0.271x` AR at `1.071` visible
  tokens/cycle, so public GGUF generation remains MTP-disabled.

The real local UD-Q3_K_M blk.40 has 20 tensors: Q8_0 attention, output,
shared-expert, and `nextn.eh_proj` weights; Q3_K expert gate/up; Q4_K expert
down; BF16 router/shared-gate weights; norms; and four `nextn.*` tensors. It has
no NextN-specific embedding or lm-head tensor, so the documented target
embedding/output fallbacks apply.

| Task | Work | Actual dependency |
|---:|---|---|
| #27 | Land and close the first changed-algebra prefill tranche | **Completed:** guarded residual-D4 MMQ retained; ~3,000 tok/s objective remains unmet |
| #28 | Build one row-shaped GGUF target executor for independent decode and verify rows | **Completed:** C=2 decode and V=2 serial-chain layer/full-logit parity are exact |
| #29 | Promote native UD-Q3_K_M c=2/4/8 decode and replace the blocked template | **Completed:** exact native C=2/4/8 plus reclaim/readmit scheduling retained |
| #30 | Materialize blk.40 and emit candidate-only `DraftBatch` rows | **Completed:** separate map/residency, raw Q3_K kernels, real one-step parity, and provider landed |
| #31 | Integrate and benchmark GGUF MTP end to end | **Completed/no-hold:** exact shared-ABI transaction path; B=1/2/3 economics regressive, default disabled |
| #32 | Reprofile residual c=1 Q3 decode | **Completed/no-hold:** final tree is 8.82493 ms/token and 671 launches/token; the source-shaped dense-Q8 block-serial candidate failed exact full logits, while its exact-association salvage regressed real leaves 21–80%; candidate removed |

## Executive recommendation

The review now has two distinct conclusions.

**Prefill:** the Q3 gap is not a general HIP, attention, GDN, or launch-overhead
problem. It is overwhelmingly the correctness-first selected-MoE
implementation:

- At 512 tokens, raw IQ selected gate/up and down consume **692.30 ms of
  787.14 ms** profiled kernel time (**87.95%**).
- At 4K, they consume **5,277.84 ms of 5,902.88 ms** (**89.41%**).
- The matched Q4 grouped-WMMA selected path consumes only **67.83 ms** and
  **440.43 ms** at the same two shapes.
- The non-profiled Q4/Q3 prefill ratios are **4.656x at 512** and **5.119x at
  4K**. The kernel-time ratios are nearly identical: **4.530x** and **5.258x**.

**Decode:** the grouped-prefill change does not run at `c=1` and therefore
cannot explain or close the decode gap. The retained same-W7900 results are
**92.285/97.373/98.111 tok/s** for hipEngine versus
**161.560/158.710/146.610 tok/s** for qwen-kernel at 512/1K/4K. qwen-kernel's
XTX results are **189.960/187.770/173.490 tok/s**. These are different engines
and numerical contracts, not a drop-in backend ratio, but they prove that the
same model and raw quantization admit the **150-190 tok/s regime on gfx1100**.
There is no retained HIP-versus-Vulkan evidence for a fundamental HIP kernel
ceiling: production-shaped serialized Q4, Q6, and dense-Q8 controls favor HIP
on gfx1100.

The original #11 → #20 → #15 → #21 → #16 handoff is complete and retained
only as provenance below. Grouped prefill, the exact-IQ decode audit, and the
next-RMS tail landed; hierarchical top-k, IQ4 tile4, and routed/shared overlap
were measured and rejected. The authoritative remaining order is the current
post-prefill roadmap above, not those historical task dependencies.

Do **not** reopen full-column register-resident GDN, broad Q4 launch/repack
sweeps, broad attention geometry, or generic launch reduction. The original
raw-Q3 IQ4 underfill premise is no longer open either: K512 grouped prefill now
uses the retained exact one-wave leaf, while the exact c=1 IQ4 tile4 attempt
regressed the real family and was removed. Reopening requires a different
algorithm/layout premise and a fresh production profile.

For future models and cards, adopt only **bounded, coverage-first auto-tuning**
as described in §14: inventory every static and runtime shape class, compare a
small source-justified set of exact variants on the production kernel/path, and
cache a fully identified card-specific winner. This is portability and audit
infrastructure, not permission to benchmark in the request path or to reopen
closed Q4/PARO sweeps without a changed premise.

---

## 1. Review and measurement provenance

### Source revisions

- qwen-kernel: `52e240f9c6d91750d0e5e692976cfb67fd9bc603`.
  - The checkout also contains the pre-existing local Q3 baseline work in
    `src/main.cpp`; the shader and `docs/amd-opt/` evidence cited below is
    tracked at the named revision.
- hipEngine Q3 worktree: clean
  `d47e63cd85fe2b06f242de23855bb932ca9f09ff` (`q3-k-m`).
- hipEngine main was inspected read-only as well. It had unrelated pre-existing
  worktree changes and was not used as the profile source.

### Matched profile environment

- GPU: **AMD Radeon RX 7900 XTX**, gfx1100 (GPU1).
- Runtime/compiler: hermetic TheRock **HIP 7.15**, AMD clang 23 snapshot.
- KV: BF16.
- Prompt: repeated token ID `9707`.
- Modes: persistent session, forced bulk prefill, WMMA requested, cached builds
  required, decode disabled for the prefill captures. The automatic 4K policy
  resolved MoE chunks to 1,024 tokens.
- Wall rate: one discarded warmup plus one non-profiled measured run.
- Attribution: a separate single `rocprofv3 --kernel-trace` run. Profiler wall
  rates are not used as throughput claims.
- hipEngine remained clean after all captures.

Models:

| Model | Size | Sampled fingerprint | Tensor inventory hash |
|---|---:|---|---|
| `Qwen3.6-35B-A3B-UD-Q3_K_M.gguf` | 17,104,402,720 B | `0e7a765b...c3a11cb` | `59b2a47b...ab49c4d` |
| `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | 22,663,387,424 B | `936659d6...64c89fb` | `bd93ba3d...912d64e` |

Raw evidence is under `/tmp/hipengine-q3-review-profile/`. The four profiler
summary hashes are:

- Q3 512: `1afa186850dd197b...`;
- Q3 4K: `b7028f70631a92cd...`;
- Q4 512: `e0354c2d9fd600dc...`;
- Q4 4K: `0a990365ea4802d7...`.

These are diagnostic source-attribution captures, not a replacement for the
repository's repeated multi-prompt promotion gates.

The decode addendum below uses **retained artifacts, source inspection, and
first-principles accounting only**. No new GPU command was run: GPU1 is reserved
for the active grouped-IQ prefill task (#11), while GPU0 is in use by MTP work.
The retained hipEngine Q3 decode result is the clean W7900 baseline at
`44a1f963`; the source inspection is pinned to clean `d47e63cd`. A matched
post-#11/#20 decode profile is deliberately the first future campaign step,
not evidence silently inferred here.

---

## 2. Matched Q3 versus Q4 result

### Wall and total kernel time

| Prompt | Q3 wall prefill | Q4 wall prefill | Q4/Q3 | Q3 kernel sum | Q4 kernel sum | Kernel ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 665.928 tok/s | 3,100.424 tok/s | **4.656x** | 787.137 ms | 173.745 ms | **4.530x** |
| 4,096 | 689.485 tok/s | 3,529.696 tok/s | **5.119x** | 5,902.879 ms | 1,122.663 ms | **5.258x** |

The close agreement between wall and kernel ratios rules out Python overhead,
model loading, or profiler classification as the main explanation.

Tracked peak is lower for Q3, as expected:

| Prompt | Q3 peak | Q4 peak |
|---:|---:|---:|
| 512 | 15.692 GiB | 21.228 GiB |
| 4,096 | 16.134 GiB | 21.670 GiB |

Q3 therefore has enough memory headroom for scheduler metadata and the existing
selected-row scratch. It does not need a large duplicate weight layout to solve
this problem.

### Selected-MoE attribution

The generic summary script classifies the new IQ kernels by old names:
`iq_selected_gate_up_silu_kernel` appears under `silu_mul`, while
`iq4_xs_selected_gemv_kernel` appears under `other`. The actual kernel names,
not those bucket labels, give the following result:

| Q3 selected family | 512 | Share | 4K | Share |
|---|---:|---:|---:|---:|
| IQ3_XXS gate/up + SiLU | 236.235 ms | 30.01% | 1,981.642 ms | 33.57% |
| IQ4_XS gate/up + SiLU | 5.981 ms | 0.76% | 52.177 ms | 0.88% |
| IQ4_XS down | 434.053 ms | 55.14% | 3,109.802 ms | 52.68% |
| Q6_K down | 16.029 ms | 2.04% | 134.218 ms | 2.27% |
| **All selected expert work** | **692.298 ms** | **87.95%** | **5,277.839 ms** | **89.41%** |

For comparison, Q4's compact grouped-WMMA selected families total:

| Prompt | Q4 selected time | Share of Q4 prefill |
|---:|---:|---:|
| 512 | 67.832 ms | 39.04% |
| 4,096 | 440.432 ms | 39.23% |

The non-MoE portions are already close. For example, dense Q8 is 50.34 versus
54.12 ms at 512 and 252.82 versus 268.91 ms at 4K; GDN recurrence is 16.99
versus 18.29 ms and 126.65 versus 135.35 ms. Optimizing those first cannot close
a 4.7-5.1x wall gap.

As an illustrative bound, replacing only Q3's selected-family time with Q4's
absolute selected-family time would reduce the profiled totals to roughly
162.67 ms at 512 and 1,065.47 ms at 4K: **4.84x** and **5.54x** kernel-time
speedups. This is a target decomposition, not a promise that IQ formats can
match Q4 WMMA exactly.

---

## 3. Why the current Q3 prefill path scales poorly

Current Q3 selected kernels are in:

- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_gemv.hip`;
- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_gemv.py`;
- dispatch call sites in
  `hipengine/runtime/qwen35_gguf_runner.py`.

They are good correctness-first decode kernels:

- raw GGUF IQ3_XXS/IQ4_XS blocks stay resident;
- gate and up are fused through SiLU;
- four adjacent output rows reuse each lane's eight activations;
- gate/up explicitly round through BF16 before SiLU, preserving the previous
  split-kernel boundary;
- launches are stream ordered and graph safe;
- no host read or resident repack is required.

The missing property is **expert grouping across prompt rows**. The launch grid
has one Y row per selected `(token, slot)`, so a prompt with `T` tokens and
`top_k=8` streams the selected expert rows `8T` times even when many rows select
the same expert.

For the model's 256 experts, hidden size 2,048, and expert FFN size 512:

| Weights per layer | All 256 experts once | Token-major bytes, 512 | Avg. reuse per live chunk | Token-major bytes, full 4K prompt | Avg. reuse in each 1,024-token MoE chunk |
|---|---:|---:|---:|---:|---:|
| IQ3 gate + up | 196 MiB | 3.0625 GiB | **16x** | 24.5 GiB | **32x** |
| IQ4 down | 136 MiB | 2.125 GiB | **16x** | 17.0 GiB | **32x** |

The unchunked full-prompt/all-expert ratio would be 128x at 4K, but that is not
the matched execution: its MoE chunk is 1,024, so a grouped kernel reloads an
active expert row in each of four chunks. The practical average opportunity is
therefore 16x at 512 and 32x per 4K chunk. If a chunk uses fewer than all
experts, its unique-weight set is smaller and potential reuse is larger.
Arithmetic and activation traffic remain, so these are not speedup predictions;
they explain why the measured gap grows from 512 to 4K.

qwen-kernel solves this class of problem with:

- `shaders/moe_group_pairs.comp` — counting-sort `(expert, token, slot)` pairs;
- `shaders/moe_gateup_iq3_grouped.comp` — one `(expert, output row)` workgroup,
  dequantize once, loop that expert's pairs;
- `shaders/moe_down_iq4_grouped.comp` and
  `shaders/moe_down_q6k_grouped.comp` — the analogous grouped down path;
- grouped layer-tail reduction in `src/main.cpp`.

On qwen-kernel's 35B path, grouped gate/up alone improved an XT smoke from
595.31 to 767.69 tok/s, and grouped gate/up plus down reached 858.33 tok/s
(**+44.2%**). Final grouped prefill improved the earlier short-prompt result by
43.25% on XT and 35.97% on XTX. Those percentages should not be transplanted to
HIP, but they validate the algorithm and exactness strategy.

---

## 4. Current feature map: what is new versus already present

| Technique | qwen-kernel | hipEngine Q3 | hipEngine Q4/PARO | Recommendation |
|---|---|---|---|---|
| Raw selected gate/up fusion | Yes | **Already present** | Present in format-specific paths | No new work |
| IQ3 decode output-row tiling | Four-row retained; eight rows rejected on VGPR pressure | **Four-row already present** | T16 kernels already compute multiple columns | Do not repeat IQ3 rowtile 4/8 |
| IQ4 selected-down work distribution | One output WG spans all top-8 slots; local128 retained | One `(slot, four outputs)` local256 WG; only 64/256 threads enter the dot loop | Different compact/T16 or AWQ contracts | Open a Q3-only exact local64 and all-slot design |
| Grouped expert prefill | Raw IQ grouped | Missing for IQ3/IQ4_XS | **Already compact grouped** with WMMA | Port scheduler use to Q3 only |
| Router + top-k completion fusion | Last router WG selects | **Already present** | **Already present** | Keep |
| Hierarchical top-k union | Retained | Missing | Missing | Adopt across formats |
| Split-K and adaptive GQA | Retained | Present through common runtime | Present | Closed/table stakes |
| Whole-step command replay | Static Vulkan CB | HIP graph replay | HIP graph replay | Already adopted; API effects do not transfer |
| Register-resident GDN state | Retained under ACO | Current c1 rereads state | Same common GDN | Do not port directly; HIP resource gate failed |
| Attention add + post-attention RMSNorm | Fused | Present | Present | Already adopted |
| MoE tail + **next-layer input** RMSNorm | `add_rms3.comp` | Missing | Missing | Adopt after stable decode ABI/top-k |
| Routed/shared branch overlap | Barrier-free independent dispatches | Serialized stream | Serialized stream | Bounded experiment only |
| Shared expert right-sizing | GU64/down32 | Existing HIP W8 paths are already right-sized and small | Existing optimized paths | No generic port |

Three distinctions matter:

1. hipEngine's existing `add_rmsnorm` fusion is after attention and produces
   the current layer's MLP input. qwen-kernel's `add_rms3.comp` is at the **MoE
   tail** and also computes the **next layer's** input normalization. They are
   different boundaries.
2. The Q3 HIP gate/up kernel already incorporates qwen-kernel's useful four-row
   IQ3 idea. Recommending “add IQ3 row tiling” again would duplicate work, and
   qwen-kernel already rejected eight rows on register pressure.
3. The Q3 HIP down kernel also computes four output rows, but that does **not**
   make its work distribution equivalent to qwen-kernel. It still launches one
   local256 workgroup per selected slot with only 64 dot-product threads for
   `K=512`; the open decode work is slot/wave utilization and contraction, not
   another generic output-row sweep.

---

## 5. P0 design: grouped raw-IQ prefill

### 5.1 Reuse hipEngine's scheduler, not qwen's Vulkan ABI

hipEngine already owns the required device metadata path in
`qwen35_gguf_runner.py` and the MoE scheduler kernels:

- `qwen35_moe_group_count`;
- `qwen35_moe_group_prefix`;
- `qwen35_moe_group_scatter_gather_lowp`;
- `expert_start_compact`, sorted lanes/experts/weights, and inverse mapping;
- existing compact selected-output and weighted-lane scratch.

The Q4/PARO route uses this metadata before its compact WMMA kernels. Q3 should
reuse the same metadata contract and add only raw-IQ expert-major compute.
There is no need for a host scalar read or a WMMA tile map: a static grid over
`(num_experts, out_features)` can let empty experts exit immediately.

Suggested future kernel family (name illustrative):

```text
gguf_iq_grouped_prefill.hip
  iq3_xxs_grouped_gate_up_silu_bf16_out
  iq4_xs_grouped_gate_up_silu_bf16_out
  iq4_xs_grouped_down_bf16_out
```

### 5.2 Gate/up schedule

For each `(expert, output row)` workgroup:

1. Read `[expert_start[e], expert_start[e+1])`; return before touching weights
   when empty.
2. Dequantize that expert's IQ3_XXS gate and up row once, distributed across
   lanes. Retain each lane's small dequantized segment in registers.
3. Loop over sorted selected rows for the expert.
4. For each row, load its activation segment, perform the existing term order
   and wave/block reduction, and produce gate/up totals.
5. Explicitly round both totals through BF16 **before** SiLU, exactly as
   `iq_selected_gate_up_silu_kernel` does now.
6. Store BF16 intermediate in compact sorted-lane order.

Start with one output row per workgroup. Combining qwen's grouped reuse with the
current four-row decode tile would require retaining four gate and four up
weight segments while also looping many pairs; that is a likely VGPR trap.
Only test row tiles after the one-row grouped kernel passes full-model gates.

### 5.3 Down schedule

Use one `(expert, output row)` workgroup:

1. Dequantize the IQ4_XS row once.
2. Loop the expert's compact selected rows.
3. Preserve the existing reduction and BF16 output boundary.
4. Keep outputs **unweighted** in the first implementation and reuse the
   existing inverse/weighted combine. Do not silently copy qwen-kernel's F32
   “weight before reduction” arithmetic into hipEngine.

An unchunked 4K selected output would be 128 MiB
(`4096 * 8 * 2048 * 2`), but the matched route's live 1,024-token MoE chunk is
about 32 MiB. A grouped implementation can reuse that chunk-local BF16 scratch
rather than add qwen-kernel's separate F32 contribution buffer.

### 5.4 Routing threshold

Keep the current token-major kernel for small selected-row counts. At minimum,
sweep selected assignments around:

```text
32, 64, 128, 256, 512, 1024, 4096, 8192, 32768
```

qwen-kernel falls back when assignments are fewer than experts. hipEngine should
measure its own crossover because HIP wave size, barriers, cache, and sorted
scatter cost differ. The threshold belongs in the package policy, not a global
format assumption.

### 5.5 Expected impact and stop conditions

The hard Amdahl caps are 87.95% and 89.41% of Q3 prefill kernel time. The first
scalar grouped version does not need to match WMMA to be worthwhile. It should,
however, show all of the following before promotion:

- selected gate/up and down both improve at 512 and improve more strongly at
  4K, demonstrating actual expert reuse rather than a launch-only effect;
- at least a material double-digit full-prefill gain at both primary shapes;
- no per-chunk host read or synchronization;
- no duplicate resident IQ weight allocation;
- no unexpected fallback kernel in rocprof;
- bounded scratch growth and no loss of Q3's current memory advantage.

If the grouped kernel is fast at 512 but fails to scale at 4K, profile barriers
and the sequential pair loop before considering an IQ-to-WMMA repack. Do not
jump directly to a second full-model tiled weight layout.

---

## 6. Q3 decode feasibility and campaign

### 6.1 Scope correction and empirical target

The matched profiles in Sections 2-3 disabled decode. They prove the prefill
bottleneck and nothing about the decode ceiling. P0 groups many prompt rows by
expert; a `c=1`, top-8 decode token has no cross-token expert reuse to group.
The decode campaign must therefore be measured and optimized separately.

The retained same-W7900 baselines give the relevant existence proof:

| Context | hipEngine Q3 | hipEngine latency | qwen-kernel Q3 | qwen latency | Latency gap |
|---:|---:|---:|---:|---:|---:|
| 512 | 92.285 tok/s | 10.836 ms | 161.560 tok/s | 6.190 ms | **4.646 ms** |
| 1,024 | 97.373 tok/s | 10.270 ms | 158.710 tok/s | 6.301 ms | **3.969 ms** |
| 4,096 | 98.111 tok/s | 10.193 ms | 146.610 tok/s | 6.821 ms | **3.372 ms** |

Both engines use the exact same `UD-Q3_K_M` file, but their runtimes,
activation/KV arithmetic, prompt handoff, and reduction contracts differ. The
ratios are therefore target evidence, not a promise that a copied shader will
produce the same wall time. They do establish that **145-160 tok/s is a
reasonable W7900 campaign band**. qwen-kernel's XTX results at the same three
shapes are 189.960/187.770/173.490 tok/s; hipEngine has no retained same-revision
Q3 XTX decode baseline yet, so **170-190 tok/s is an XTX escalation target only
after a matched baseline**. A 200 tok/s result means 5.000 ms/token and remains a
short-context stretch target, not an acceptance claim.

### 6.2 Why this is an implementation gap, not a HIP prohibition

The retained gfx1100 HIP/Vulkan timing-contract-v2 matrix is important context:

- Vulkan command-buffer replay wins tiny dispatch floors by 2.437-10.122x, but
  this is runtime evidence; hipEngine already uses one-step graph replay.
- Required-order geometry, reduction, memory/waitcnt, sampler, and two-stage
  controls favor HIP. Independent-throughput Vulkan rows do not model one
  request's dependent decode chain.
- Production-shaped serialized Q4 selected-dual, Q6 selected-down X8, and dense
  Q8_0 all favor HIP. The synthetic packed-dot Vulkan lead on gfx1100 is only
  1.052-1.133x and does not explain a 1.5-1.75x full-engine gap.
- The ACO register-resident GDN result is a real compiler-specific exception,
  but the corresponding HIP full-column design spilled and is already closed.
  That leaf does not establish a general HIP ceiling.

The right inference from qwen-kernel is therefore “port the useful dependency,
work-distribution, and fusion boundaries,” not “move hipEngine to Vulkan” or
“LLVM cannot reach this rate.” A matched raw-IQ cross-backend microbenchmark is
optional only if a production HIP body remains unexplained after source,
resource, and counter inspection.

### 6.3 The current raw-IQ decode geometry

The Q3 target contains:

- 39 IQ3_XXS gate tensors and 39 IQ3_XXS up tensors in layers 0-38;
- 37 IQ4_XS selected-down tensors;
- IQ4_XS gate/up only in layer 39;
- Q6_K selected down in layers 34, 38, and 39.

The correctness-first HIP gate/up body is already well shaped for its fixed
`K=2048` case. With `groups8 = 2048 / 8 = 256`, all 256 threads perform dot
work, and each workgroup computes four output rows. Its retained real-shape
sample is 22.000 us for IQ3, 64 VGPR, and zero scratch.

The ordinary IQ4 down shape is different. In
`gguf_iq_selected_gemv.hip`, the grid is:

```text
ceil(2048 outputs / 4-row tile) × 8 selected slots
= 512 × 8 = 4,096 workgroups
```

Each workgroup launches 256 threads, but `groups8 = 512 / 8 = 64`; only threads
0-63 enter the dot loop. Two waves perform useful dot work while six waves carry
zero accumulators through the reduction and barrier. Statically this is 32,768
scheduled workgroup-waves for only 8,192 dot-producing waves. The retained
real-shape unit sample is **82.921 us**, 40 VGPR, 512 B LDS, and zero scratch.
Multiplying an isolated unit duration by 37 layers would be about 3.07 ms, which
is a prioritization signal, **not** a full-step attribution claim: cache state,
power residency, surrounding kernels, and profiler perturbation require the D0
trace below.

qwen-kernel's `moe_down_iq4.comp` instead launches one workgroup per output.
For this exact shape it has `8 selected slots × 16 32-value subblocks = 128`
useful tasks, matching its retained local128 option. It multiplies routing
weights before a cross-slot subgroup reduction and writes an F32 selected sum.
That arithmetic is not hipEngine's contract and must not be copied literally.
hipEngine currently rounds every slot's down result to BF16, performs routing
FMAs in slot order, rounds the selected aggregate, then combines residual and
shared output. The opportunity is the qwen work distribution, adapted to those
boundaries.

This premise is new. The older Q4/Q5/Q6 T16 launch-width and repack sweeps did
not include the raw Q3 IQ4_XS kernel introduced at `8ece5a57`; documenting them
as closed does not close this shape.

### 6.4 D0 — post-stabilization decode attribution

No new profile is run in this documentation pass. After task #11 stabilizes and
task #20 has a rollback selector, task #15 should begin with a cached-build,
selected-region trace rather than a kernel edit.

Required captures:

1. On the first explicitly released gfx1100 device, capture fresh wall medians
   at 512/128 and 4K/128, with 1K as a cheap shape control; profile a short
   16-token steady window separately. Current ownership makes the XTX/GPU1 the
   likely first device after #11. Repeat the same protocol on W7900 after GPU0's
   MTP owner releases it and before a cross-card/global promotion.
2. Eager and one-step graph wall controls. Use the trace for GPU attribution,
   not profiler wall throughput.
3. Per-token time and launch count for raw IQ3 gate/up, raw IQ4 down, Q6 down,
   shared Q8, router/select, MoE combine, dense Q8, GDN/Conv, attention, RMSNorm,
   and lm-head.
4. Per-layer distributions for IQ4 down rather than only an aggregate; confirm
   all 37 expected launches and no generic fallback.
5. HSACO VGPR/SGPR/LDS/private bytes and achieved wave/occupancy counters for
   the exact hot symbols.
6. `#20 off` and `#20 on` controls on the same commit so hierarchical top-k is
   not incorrectly credited to the later IQ change.

Whichever card runs first must receive a fresh same-revision hipEngine baseline.
Do not compare the current W7900 hipEngine row directly to the 190 tok/s XTX
qwen row. A shared gfx1100 default needs both cards; a card-scoped policy still
needs the other card as a non-regression control before a broad claim.

### 6.5 D1A — exact local64 IQ4 down

The first implementation should change only the ordinary
`in_features=512/out_features=2048` raw IQ4 down specialization:

- retain one `(selected slot, four output rows)` workgroup;
- launch 64 threads instead of 256;
- retain the same two wave reductions, their order, and the BF16 store;
- size LDS for two waves and leave all Q6 and gate/up paths unchanged;
- keep the local256 symbol as an explicit rollback.

This reduces scheduled waves from 32,768 to 8,192 while leaving useful dot work
and the current rowtile arithmetic unchanged. It is not expected to be a 4x
kernel win because weight/dequant work is identical; it removes empty waves,
block overhead, six zero-wave partials, and their occupancy cost. qwen-kernel's
right-sized IQ4 and shared-down results show that this class can win, while
hipEngine's own small-K PARO/Vulkan review independently found 256-thread
expert-down shapes underfilled.

Require bitwise equality for every BF16 selected row, zero private bytes, and a
repeatable full-step gain. If the leaf improves but full wall is flat, retain it
only if broader D1B composition demonstrates the saved resource/time is real.

### 6.6 D1B — exact all-top-8, four-output-row down contraction

The higher-EV design should be statically guarded to
`top_k=8, in_features=512, out_features=2048` and fall back otherwise. It should
combine qwen-kernel's slot coverage with hipEngine's existing four-row reuse
without adopting qwen's F32 association:

1. Launch **512 local256 workgroups**, one per four adjacent output columns,
   instead of 4,096 workgroups split by selected slot.
2. Assign wave `s` to selected slot `s` for top-8. Each wave performs two
   32-term passes over that slot's 64 eight-value groups. Keep the first wave
   sum live, compute the second, then add them in the same order as the current
   two useful waves.
3. Retain four output accumulators per lane, preserving the current activation
   reuse and per-output term order.
4. BF16-round each slot's completed down value exactly as the current kernel
   does before exposing it to routing.
5. Store the eight slot values in LDS; one wave applies the eight routing FMAs
   in slot order, then BF16-rounds the selected aggregate exactly as
   `weighted_sum_shared_gate_combine_residual_out_kernel` does today.
6. Write one BF16 selected-sum vector. Reuse the existing
   `shared_gate_combine_residual_out_bf16` boundary initially; task #21 may
   later consume this stable ABI in the next-layer RMS fusion.

Scope the first route to the production BF16-residual contract. Keep the
optional F32 residual/diagnostic combine paths on the current per-slot kernel
until a sibling fused ABI passes their own exact boundary tests.

This schedule keeps all eight waves useful, reduces workgroups eightfold, and
eliminates the `8 × 2048` selected-down BF16 materialization plus its later
weighted read. It does **not** change resident weights, add an indirect host
read, use atomics, or reassociate across slots. If LLVM cannot keep the four
accumulators and two-pass state spill-free, test a two-output tile before any
new weight layout; reject private memory immediately.

Correctness must compare more than final tokens:

- each slot's pre-weight BF16 value against the current kernel;
- the slot-ordered F32 routing accumulator;
- the BF16 selected aggregate;
- shared-gated residual output and next-layer input;
- all 37 IQ4-down layers, adversarial selected IDs/weights, and graph replay.

### 6.7 D1C — targeted IQ3/codegen follow-up only after down

The IQ3 four-row gate/up design is already retained in both engines. qwen's
four-row change moved wall only about 0.5-1.5%, and its eight-row variant lost on
register pressure. Do not rerun rowtile 4/8 or a broad thread sweep.

If D0 still shows IQ3 gate/up material after D1A/B, inspect the exact HSACO for
64-bit address arithmetic, codebook loads, waits, and invariant expert/row
bases. A bounded source experiment may hoist wave-uniform bases or make
wave/block traversal explicit, following hipEngine's retained Q8T16
wave/block-index lesson. Keep it only with a production-shaped leaf and
full-model win; a compiling builtin or lower instruction count is not proof.

### 6.8 Campaign composition, targets, and stop rules

The active task graph already encodes the safe composition order:

| Task | Decode role | Expected scale before measurement | Dependency/read |
|---:|---|---:|---|
| #11 grouped compact IQ prefill | Stabilizes Q3 route/ABI; no direct `c=1` gain | 0% decode | Finish before profiling shared source |
| #20 hierarchical exact top-k | Exact barrier reduction | Usually sub-1% | Measure off/on before D1 |
| #15 row-stationary IQ/codegen | D1A local64, then D1B exact all-slot down | Potentially material; profile required | Primary Q3 decode campaign |
| #21 MoE tail + next RMS | Removes one boundary and hidden reread | ~0.3-1% | Build on stable D1 selected sum |
| #16 overlap/broader fusion | Hide independent shared/routed work | 0-3%, uncertain | Require real overlap and >=1% wall |

GGUF NextN remains outside the one-token AR optimization campaign. Task #30
now owns and executes blk.40 separately from AR, including raw Q3_K selected
gate/up kernels and a candidate-only provider under the locked speculative ABI.
Task #31 connects that proposer to the row-shaped target verifier and shared
transactional accept/commit path. The route is exact but remains an explicit
diagnostic after the matched B=1/2/3 economics gate rejected promotion.

The latency budget makes clear why #20/#21 alone cannot reach the target:

| Context | Current | 125 tok/s requires | 150 tok/s requires | Same-card qwen parity requires |
|---:|---:|---:|---:|---:|
| 512 | 10.836 ms | save 2.836 ms | save **4.169 ms** | save 4.646 ms |
| 1K | 10.270 ms | save 2.270 ms | save **3.603 ms** | save 3.969 ms |
| 4K | 10.193 ms | save 2.193 ms | save **3.526 ms** | save 3.372 ms |

Use 125 tok/s (8.000 ms) as an intermediate architecture milestone, not a final
success condition. The W7900 campaign target is the measured qwen band,
approximately 145-160 tok/s depending on context. Every exact repeatable win is
independently retainable; do not reject a sound leaf merely because one change
does not close four milliseconds.

Stop or pivot when:

- D0 shows the proposed family has insufficient Amdahl share and the leaf test
  cannot remove at least a measurable portion of full-step wall;
- local64 fails to reduce the named kernel or creates a neighboring regression;
- D1B changes any BF16/slot-order boundary, spills, or wins only by changing the
  numerical contract;
- a synthetic micro wins but the intended symbol or route is absent from the
  full trace;
- a candidate improves 512 but regresses 4K without a justified
  context-specific selector;
- the remaining gap moves to dense Q8, GDN, attention, or head—in that case
  start a new profile-backed common-kernel tranche rather than continuing IQ
  geometry by inertia.

### 6.9 gfx1151 scope and first-principles bound

The gfx1100 throughput target must not be copied to gfx1151. qwen-kernel's exact
active parameter payload is 2,457,104,528 bytes/token before mandatory state,
KV, scratch, and activation traffic. The cold-stream parameter model on Strix
Halo is:

```text
2.457 GB / 256 GB/s theoretical = 9.598 ms/token = 104.2 tok/s
2.457 GB / 221 GB/s practical read = 11.118 ms/token = 89.9 tok/s
```

Even the impossible-friendly case that subtracts the full 32 MiB MALL from
every token is about 9.467 ms / 105.6 tok/s at the theoretical bus. Reaching
150 tok/s would require 368.6 GB/s for active parameters alone. Dequantization,
recurrent state, KV, scratch, and activations lower the achievable rate further.
Thus 150-200 tok/s is not a credible gfx1151 single-stream AR campaign target
even though the 15.9 GiB model fits unified-memory capacity.
For gfx1151:

- first add native package registration/correctness and obtain its own profile;
- retest D1A/D1B with architecture-scoped policy, wave32, and native resource
  evidence rather than inheriting a W7900 launch decision;
- prefer fusions that remove memory traffic, but reject occupancy cliffs;
- do not prioritize task #16 overlap while production requires
  `GPU_MAX_HW_QUEUES=1` for stability; multiple logical streams are unlikely to
  provide a promotable concurrent-kernel result under that policy.

### 6.10 Decode no-go rules

- Do not credit P0 grouped prefill to decode.
- Do not copy qwen's pre-weighted cross-slot F32 reduction into the strict HIP
  path without an explicitly approved numerical-contract change.
- Do not infer HIP/XTX performance from the retained W7900 baseline or vice
  versa.
- Do not reopen IQ3 rowtile8, full-column register GDN, generic Q4 T16 launch
  sweeps, blanket Wave64, `-ffast-math`, undocumented scheduler flags, or a
  second resident IQ layout.
- Do not interpret qwen-kernel's Vulkan result as a mandate for a Vulkan backend.
  Use its source to form HIP hypotheses, then require HIP code-object and
  full-model evidence.

---

## 7. P1 design: hierarchical exact top-k

qwen-kernel's `shaders/moe_route_select_hier.comp` uses a simple exact result:
a candidate below local rank `k` in its wave already has `k` candidates ahead
of it and therefore cannot be in the global top-k. Each wave keeps its local
`k`; one wave selects the global `k` from that union.

hipEngine's current helper in
`hipengine/kernels/hip_gfx1100/moe/router.hip`,
`qwen35_router_select_from_logits`, instead performs one whole-block argmax for
each selected expert. For top-8 it executes two block barriers per pick after
the initial load. qwen-kernel removes 24 whole-workgroup barriers in its fused
selector accounting.

### 7.1 HIP implementation shape

- Keep the existing router-logit producer and last-block completion protocol.
- Add a sibling selection helper; do not change router accumulation.
- Each HIP wave selects its local top-k with shuffles only.
- Store at most `num_waves * top_k` value/ID pairs in LDS.
- One wave performs the final top-k.
- Specialize the production shapes (256 experts/top-8 and any admitted
  512-expert/top-10 shape) while keeping the existing helper as rollback.
- Write the code against `warpSize`; do not assume Vulkan subgroup width.

### 7.2 Exactness gates

Test IDs and routing weights bit-for-bit against the current selector for:

- random logits and realistic router outputs;
- all equal values and repeated ties;
- lowest/highest expert winning ties;
- NaN/Inf sanitation behavior;
- 256 and 512 experts;
- top-k 1, 8, 10, and the supported maximum 16;
- standalone batched select and last-router-workgroup decode select.

Preserve lower-ID ties and the current softmax accumulation order. A selector
that merely returns the same set in a different order is not exact enough,
because slot order feeds later reduction order.

### 7.3 Expected impact

Retained common hipEngine profiles put the router family near 4-5% of decode,
and selection is only part of that family. qwen-kernel's hierarchical selector
moved its combined router/shared stage from 606.5 to 581.3 us in an XTX sample;
its broader hierarchical/right-sized composition was about 0.8-1.2% wall.

Expect a sub-1% hipEngine full-step result, not the change that carries Q3 from
98 to 150 tok/s. Retain only with zero scratch/spills and a repeatable
full-model improvement outside noise. Keep the old helper selectable through
task #15's D0 capture so this gain is measured separately from IQ4 down. The
value is that one small exact change applies to Q3, Q4, and PARO.

---

## 8. P2 design: MoE tail plus next-layer RMSNorm

qwen-kernel's `shaders/add_rms3.comp` and the scheduling in `src/main.cpp` sum
residual, routed output, and shared output, store the next residual, and compute
the next layer's normalized input in one stage.

hipEngine currently does:

1. `weighted_sum_shared_gate_combine_residual_{out,batch_out}` in
   `hipengine/kernels/hip_gfx1100/fused/paro_combine.hip`;
2. write the MoE-combined residual;
3. at the next layer, call input RMSNorm (`gguf_rmsnorm_*` for GGUF or
   `paro_rmsnorm_*` for PARO).

That boundary is visible in:

- GGUF combine call sites near `_run_post_attention_moe_c1` and
  `_run_post_attention_moe_rows` in `qwen35_gguf_runner.py`;
- next-layer attention normalization in `_run_attention_norm_rows`;
- PARO combine call sites in `qwen35_paro.py`, followed by the next layer's
  `input_rmsnorm_{bf16,fp16}`.

### 8.1 Correct HIP arithmetic

Do not copy qwen-kernel's F32 shader literally. hipEngine's current contract
rounds the combined residual to BF16/FP16, then the next RMSNorm reads that
rounded value. A fused kernel must preserve this boundary:

1. form the weighted selected sum in existing slot order;
2. form `sigmoid(shared_gate) * shared`;
3. add residual in the current order;
4. round/store the residual to the model activation dtype;
5. use that rounded value for sum-of-squares;
6. apply the next layer's norm weight and write normalized activation.

For decode, use one workgroup per token looping over hidden dimensions. Emit
both raw residual and normalized output. The next layer skips its input RMSNorm
node and consumes the normalized buffer. Leave the final layer separate first;
then test binding `output_norm` only if final-hidden/debug contracts remain
intact.

### 8.2 Bound and gate

Common hipEngine decode profiles and the prior launch/dataflow cost model put
MoE combine near 1% and input normalization at roughly another 2-3% of a step.
Only a fraction is removable, so a realistic target is approximately
**0.3-1.0%** wall. D0 must replace this cross-format estimate with a Q3-local
number before performance attribution.

Require:

- exact residual and normalized buffers at every layer boundary;
- exact generated trajectories for Q3, Q4, and PARO;
- coverage across linear-attention/full-attention transitions and the last
  layer;
- zero scratch and acceptable VGPR/LDS;
- a repeatable decode gain at both 512 and 4K contexts;
- no prefill regression if the batch variant is enabled. Decode-only promotion
  is acceptable if batching reduces occupancy.

---

## 9. P3 experiment: routed/shared branch overlap

qwen-kernel deliberately leaves no barrier between independent work:

- router work can proceed while shared gate/up computes from the same normalized
  input;
- routed down and shared down write disjoint outputs and join only at the layer
  tail.

hipEngine's Q3/Q4/PARO layer helpers currently issue router, routed expert, and
shared expert chains serially on one stream.

A safe experiment is:

```text
post-norm
  ├─ stream A: router -> selected gate/up -> selected down
  └─ stream B: shared gate/up -> shared down
join -> combine/residual[/next RMS]
```

Requirements:

- disjoint scratch and outputs on both branches;
- explicit event fork/join;
- compare serial eager, two-stream eager, and graph DAG separately;
- rocprof timeline must show overlap, not merely separate stream labels;
- correctness must include graph replay and repeated state reset;
- require at least **1%** repeated full-step gain at both 512 and 4K before graph
  integration.

This is low priority. The selected kernels are large memory/compute consumers,
so shared work may contend rather than hide. ROCm issue #6409 also documents a
HIP graph/concurrency gap relevant to assuming that a mathematically parallel
DAG will execute concurrently. If the eager real-kernel microbenchmark does not
win, stop; do not build graph infrastructure around it. On gfx1151 this is
additionally blocked as a production priority by the retained
`GPU_MAX_HW_QUEUES=1` stability policy; do not disable that safety policy merely
to manufacture overlap.

---

## 10. Q4 and PARO transfer summary

### Worth adopting

1. **Hierarchical exact top-k** — same router helper and exactness contract;
   format independent.
2. **MoE tail + next-layer input RMSNorm** — same layer boundary; activation
   dtype needs BF16 and FP16 specializations.
3. **Branch overlap microbenchmark** — same dependency graph, but retain only
   with measured HIP overlap.

D1's local64/all-slot IQ4_XS down work is **not** a Q4/PARO transfer. Q4 uses
its mature T16 selected path and PARO uses AWQ/WMMA. The general lesson—fill
waves and remove a materialized selected branch while preserving contraction
order—may motivate a future profile-backed design, but the kernel, layout, and
numerical contract do not transfer.

### Useful Q4/PARO infrastructure for Q3, not new Q4/PARO work

- compact group count/prefix/scatter metadata;
- sorted-lane and inverse-lane buffers;
- device-only static upper-bound routes;
- compact scratch ownership/liveness;
- grouped weighted-lane reduction.

Q3 should plug into those interfaces rather than fork a second scheduler.

### Already present or measured closed

A distributed profile does not mean Q4/PARO cannot become faster. It means the
qwen-kernel review exposes no equivalent single missing 90%-share boundary;
major progress would need several profile-backed wins or a genuinely new
algorithm/layout. The following *specific existing designs and sweeps* are
closed unless their premise changes:

- Q4/PARO grouped compact prefill and WMMA;
- 16-column T16 selected decode kernels, which already express the output-row
  reuse that a generic “row tile” recommendation would seek;
- direct selected expert dispatch and fused gate/up paths;
- right-sized router and shared kernels;
- whole-step graph replay and GPU-resident sampling;
- split-K attention, adaptive/grouped GQA, and long-context parallel reduction;
- attention residual plus post-attention normalization;
- broad Q4/Q5/Q6 selected launch-width, launch-bound, hot-tile, side-metadata,
  and repack sweeps.

The Q4 down64 experiment is especially important not to repeat: it improved
local throughput but changed Q4_K_M generation; the Q4-only narrowed version
then regressed. The current T16 reduction topology is part of the numerical
contract. That rejection does not apply to D1A, whose raw IQ4_XS local64
specialization removes six zero waves while retaining the current two useful
wave reductions.

---

## 11. Why register-resident GDN is not on the queue

qwen-kernel's `shaders/dn_step_gate.comp` keeps a full 128-float state column in
registers and measured a 25.9-30.0% `dn.step` reduction under Vulkan/ACO. The
current hipEngine c1 kernel in
`hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` reads a state element during
`k·S` and reads it again during the update, so the source-level opportunity is
real.

However, hipEngine has already run the decisive HIP compiler gate. Its
strict-exact full-column register candidate compiled on gfx1100 with:

- **256 VGPR**;
- roughly **1,060 bytes of scratch per thread**;
- workgroup 32;
- zero LDS, but no actual register residency.

It was rejected before full-model timing. Ordered/grouped register schedules,
state-layout changes, tiled updates, scalar broadcast, and atomic completion
variants were also tested or closed. Historical production prefill used its
admitted peer-wave route; its strict exact rollback used the much faster
nonvolatile LDS32 path.

The 2026-07-20 Q3 campaign did **not** reopen that failed local-array design. It
re-derived the old spill-free LDS32 schedule under the corrected grouped-head,
rsqrt-epsilon, and decode-association contract: one lane owns one value column,
the source state is an explicit `[128][32]` LDS tile, and the full-model path is
bit-exact. LLVM reports VGPR248 but zero private scratch; the real 4K trace cuts
GDN 32.63%, so this bounded Q3 adaptation is retained without changing the
full-column-register closure below.

The final campaign experiment attacked a different boundary: compact Q/K scales
plus beta/decay were materialized once, while the exact LDS32 recurrence read
raw Q/K/V from `conv_out`. Primitive and full-model bits were exact and prepare
fell 50.50%, but repeated normalization inside each value tile made recurrence
2.60% slower. A matched production trace regressed the combined GDN pair 1.63%,
total kernel sum 0.44%, and span 0.24%, so the direct path was removed.

Therefore:

- do not port the GLSL `vec4 state[32]` body mechanically;
- do not treat ACO resource behavior as evidence for LLVM;
- do not reopen another reduction-width/storage micro-sweep;
- do not retry direct-conv LDS32 unless normalization can be shared without the
  already-rejected per-token barriers or repeated per-value-tile multiplies;
- revisit GDN only with a genuinely different chunked/prefix algebra and an
  explicit numerical contract.

This closure applies to Q3, Q4, and PARO because they share the GDN family.

---

## 12. HIP versus Vulkan constraints

The completed HIP/Vulkan review found that on gfx1100:

- serialized geometry, reductions, memory, VOPD, sampler, and two-stage
  production-shaped controls usually favor HIP;
- Vulkan's repeatable advantages are mainly tiny-dispatch command replay
  (**2.44-10.12x**) and a synthetic packed-integer edge (**1.052-1.133x**);
- production-shaped combined Q4/Q6/dense-Q8 kernels favor HIP;
- qwen-kernel's live `vkCmdDispatchIndirect` split count has no direct static
  HIP-graph equivalent;
- routed/shared overlap is algorithmically portable but runtime dependent;
- packed IQ row tiling and full-column state retention are compiler sensitive.

Adopt algorithms, not shader/backend ratios:

- use HIP templates, `__restrict__`, launch bounds, and wave-uniform indexing;
- inspect HSACO resources and ISA for every new IQ kernel;
- reject private/scratch spills even if a microbenchmark occasionally wins;
- require the real layer/full-model route, not only a synthetic dot product;
- separate serialized latency from independent throughput and graph wall from
  eager attribution;
- keep HIP rather than porting hipEngine to Vulkan.

The same-model 190 tok/s qwen result is therefore evidence that the gfx1100
hardware/model regime is reachable, not evidence that Vulkan intrinsically
executes the current HIP kernel body 1.7x faster.

---

## 13. Exact experiment gates

### 13.1 Repository and build gate

- Pin a clean hipEngine revision and record it in every artifact.
- Wait for task #11/#20 stabilization; do not benchmark or edit the worktree
  while another agent owns its kernels or GPU.
- Require cached builds for timed runs; compile outside the timing process.
- Record HIP/clang versions and model fingerprints.
- Verify `git status --short` before and after.
- Respect explicit device ownership: GPU1 only when its Q3 owner releases it,
  and GPU0 only when MTP work releases it.

### 13.2 Correctness gate

For grouped IQ kernels:

1. block dequant/dot tests against the existing GGUF CPU oracle;
2. current token-major versus grouped output equality for gate/up, post-SiLU,
   down, and weighted combine;
3. empty experts, one hot expert, uneven counts, all experts active, duplicate
   routes, and chunk boundaries;
4. exact BF16 boundary checks, especially pre-SiLU rounding;
5. full-layer comparisons before generated-token tests;
6. multi-prompt 512/4K generated IDs and logits, not repeated token 9707 alone;
7. recurrent and KV state equality where the test protocol requires it.

For D1 decode kernels:

1. compare local256 and local64 per-slot BF16 outputs bit-for-bit on canonical,
   random, real-weight, and all-37-layer cases;
2. expose D1B debug taps for every slot's pre-weight BF16 down value, the
   slot-ordered routing accumulator, and the BF16 selected aggregate;
3. compare the D1B selected aggregate to the current combine contraction before
   testing residual/shared fusion;
4. cover duplicate expert IDs, zero/negative/extreme routing weights, every
   expert boundary, layer-39 IQ4 gate/up, and the three Q6-down layers;
5. compare eager and graph layer outputs, generated IDs/logits, recurrent state,
   and KV state at 512 and 4K.

For top-k, add adversarial ties/NaNs and exact slot order. For cross-layer norm
fusion, compare both raw residual and normalized output at every boundary.

### 13.3 HIP codegen/resource gate

Capture the expected symbol with `rocprofv3` and inspect its code object:

- VGPR/SGPR count;
- private/scratch bytes — **must be zero** for these candidates;
- LDS and occupancy;
- expected workgroup/grid geometry: D1A must show local64 with 4,096 blocks and
  D1B local256 with 512 four-output blocks for the ordinary top-8 shape;
- useful/issued wave and occupancy counters where available, confirming that
  the source-level zero-wave reduction occurred;
- no unexpected generic/token-major fallback;
- no device-to-host scalar read in the timed route.

A fast isolated sample with spills is not promotable. Preserve the rollback
symbol in the same code object so A/Bs do not confound compiler/library builds.

### 13.4 Performance gate

- Separate non-profiled wall tests from profiler attribution.
- Use counterbalanced/ABBA order with at least five measured samples when
  variance warrants it.
- Primary Q3 prefill shapes: 512 and 4K; include resolved chunk boundaries.
- Primary per-card decode shapes: 512 and 4K, with 1K as the retained crossover
  control; add 32K before a broad context claim.
- Establish a fresh same-revision baseline on whichever gfx1100 card becomes
  available first. A shared default requires XTX and W7900 A/Bs; never pair
  W7900 hipEngine with XTX qwen as a speedup ratio.
- Report median, sample spread, milliseconds/token, and the leaf/family/full
  Amdahl reconciliation—not only tok/s or best run.
- Keep any exact repeatable non-regressive improvement, but label 125 tok/s an
  intermediate milestone, 145-160 W7900 and 170-190 XTX empirical parity
  bands, and 200 a stretch target.
- Preserve peak-memory accounting.
- Shared-code changes must run Q3, Q4, and PARO safety controls; package-policy
  changes need gfx1151 safety before becoming global.

### 13.5 No-go rules

Stop or revert when any of these occurs:

- generated trajectory or exact boundary changes without a pre-approved
  numerical-contract change;
- scratch/private spills;
- raw plus repacked IQ weights coexist without a demonstrated memory plan;
- a host read is added to the chunk/layer loop;
- P0 improves 512 but 4K loses or fails to show expert-reuse scaling;
- a decode candidate improves 512 but regresses 4K without a measured,
  architecture-scoped context selector;
- branch overlap lacks profiler-visible overlap;
- a fused down changes per-slot BF16 or routing accumulation merely to match
  qwen-kernel's arithmetic;
- full-model movement is below noise after a leaf-only win.

---

## 14. Bounded auto-tuning and shape-coverage policy

hipEngine and qwen-kernel are already hand-tuned more deeply than a generic
`(quant, N, K)` launcher. The useful role for auto-tuning is therefore not to
replace profiling or invent kernels. It is to make three things systematic:

1. prove that every supported model shape reaches an intentional kernel rather
   than an accidental generic fallback;
2. select among a **small, predeclared, exact** variant set when a new model or
   physical card changes the winning geometry; and
3. retain enough build, model, card, correctness, and measurement identity that
   a cached winner cannot silently survive after its premise changes.

The current `kernel-anvil` audit reinforces the workflow but is not a drop-in
implementation. Its inference inventory is rank-2 and bucketed only by quant,
`N`, and `K`; it therefore misses the rank-3 selected-expert tensors central to
Qwen3.6 MoE. Its newest llama.cpp ablation path also records that the earlier
Triton proxy explored parameters the production MMVQ runtime could not honor;
the honest selector measures the real model and real kernel. Borrow that
inventory/A-B/cache discipline. Do not borrow proxy timings, coarse bucket
collisions, Torch/Triton dependencies, or its correctness policy.

### 14.1 What “fully audited and tuned” means

“Fully tuned” does **not** mean exhaustively searching every integer context,
batch size, or tile product. It means every supported shape class has one
explicit status and no hot shape is silently unknown:

| Status | Meaning |
|---|---|
| `tuned` | Multiple exact candidates were measured on the named card and a winner passed all §13 gates. |
| `inherited_exact` | The shape provably uses the same symbol, loop/launch geometry, resource envelope, and arithmetic contract as an already tuned shape. |
| `intentional_fallback` | A generic or token-major path is deliberately retained because the shape is cold, irregular, unsupported by the fast contract, or lost its A/B. |
| `closed` | The candidate family was tested and rejected; a new model/card alone does not reopen it unless the premise actually changes. |
| `uncovered` | No reconciled dispatch/evidence exists. This is a promotion blocker for a supported hot path. |

Each model/card coverage manifest should join four sources of truth:

| Inventory class | Required fields |
|---|---|
| Static model | tensor role/name, rank, quant block type, resident layout, exact dimensions, layer/tensor count, tied/shared identity |
| Runtime shape | phase, operation/fusion role, input/output dtype, `M/N/K`, rows, expert count, `top_k`, selected/compact rows, context/KV policy, batch/speculative width |
| Actual dispatch | backend package, registry key/variant, resolved symbol, build hash, grid/workgroup, graph/eager owner, fallback reason |
| Evidence | coverage status, correctness artifact, VGPR/SGPR/LDS/scratch, timing samples, full-wall attribution, retained/closed decision |

The static scanner must include rank-2 dense weights **and rank-3 expert
weights**. A GGUF tensor inventory alone is insufficient because dynamic MoE
and serving shapes depend on routing and scheduler state. Reconcile it with
production traces that record:

- expected versus observed calls per layer and family;
- the distribution of active experts, assignments, compact rows, padding, and
  hot/cold experts for prefill;
- all selected slots and exceptional quant layers for decode;
- resolved chunk, attention split, and graph routes at each canonical context;
- generic/token-major fallbacks and their exact reason; and
- target versus draft/verifier ownership when MTP or DFlash is enabled.

Audit all **decision boundaries**, not just comfortable interior points. For a
threshold `T`, test `T-1`, `T`, and `T+1` when those shapes are legal, then the
canonical workload points. At minimum:

- one-token decode at 512, 1K, 4K, and 32K context, plus each supported
  long-context endpoint;
- prefill at the retained 512/4K points and around every GEMV/WMMA, grouped,
  AOTriton, or chunk crossover;
- empty, one-hot, skewed, balanced, and all-active expert distributions plus
  assignment counts around the grouped/token-major threshold;
- every supported c>N, concurrency, and speculative budget class on the paths
  that actually enable them; and
- real multi-prompt routing distributions, not only repeated token `9707`.

A coverage report should fail closed when a supported hot shape is
`uncovered`, an expected symbol is absent, or an unapproved fallback appears.
Report both absolute class counts and launch/time-weighted coverage: a high
weighted percentage can hide a cold correctness hole, while a static 100% can
hide one dominant dynamic fallback. Cold intentional fallbacks are acceptable;
silent fallbacks are not.

### 14.2 Shape identity and safe equivalence classes

A tuning key must describe the work the kernel actually performs, not merely the
weight matrix. The logical key is approximately:

```text
(model inventory, backend/card, phase, op/tensor role, quant/layout,
 input/output dtype, M, N, K, rank, expert_count, top_k,
 selected/compact-row class, context/KV policy,
 fusion + numerical contract, eager/graph/speculative mode)
```

Not every field needs a separate table axis when it cannot affect code or
launch behavior, but any omitted field needs an explicit equivalence argument.
Two shapes may share a result only when they resolve to the same kernel body,
launch geometry, loop trip counts, reduction association, resource class, and
memory layout. Record that inheritance in the manifest.

Do not use coarse `N/K` buckets merely to keep the table small. They can merge
shapes such as small expert-down, square dense projections, and tall lm-heads
that need different schedules. Bucket genuinely dynamic quantities only after
measuring their boundaries. For example, compact-row or context ranges may use
a piecewise selector when sampled boundary points establish a stable winner and
the fallback remains exact.

This policy controls combinatorial growth: tune one representative per proven
execution-equivalence class, not every tensor name, but never infer equivalence
from dimensions alone when tensor role changes fusion, output dtype, or
numerical boundaries.

### 14.3 Settings worth tuning

Candidate menus belong to each kernel family. The table below is a finite menu
of important axes, **not** a request for a global Cartesian sweep:

| Axis | Typical candidates or decision | Tune when | Mandatory gate |
|---|---|---|---|
| Workgroup/waves | 32/64/128/256 threads, bounded by useful K/tasks | Dead lanes, too little work per thread, or occupancy evidence | Preserve exact reduction order; expected useful/issued waves; zero scratch |
| Output/row tile | Family-supported 1/2/4/8/16 rows or columns | Activation reuse or launch count can improve without changing association | Per-row/slot boundary equality; VGPR and tail handling |
| GEMM/WMMA tile | `TM/TN/TK`, waves per tile, compact padding/tail schedule | `M>1` prefill or verifier rows and a matrix-core route exists | Exact/toleranced family oracle, no spill, real-shape full wall |
| Work mapping | token-major vs expert-major, slot-per-block vs slot-per-wave/all-slot, split count | Profile shows reuse, dead waves, or undersubscription | Same ABI and contraction order; distribution/tail coverage |
| Crossover | GEMV/WMMA rows, grouped assignments, context split cap, chunk sizes | Different exact kernels win in different regimes | `T-1/T/T+1`, guard band, no oscillating selector |
| Launch/codegen | `__launch_bounds__`, fixed production dimensions, 32-bit-safe offsets, explicit wave/block traversal, bounded unroll | HSACO/counters show a concrete occupancy or address/codegen premise | Same compiler/build controls, ISA/resource report, production slice |
| Build profile | decode/prefill, `-mcumode`, narrowly scoped compiler option | The same source has a measured phase/compiler mismatch | Never blanket-enable; cross-phase and cross-card controls |
| Fusion/replay | unfused/fused boundary, graph/eager route, replay grouping | Launch/materialization share is material in the real step | Intermediate equality, graph reset/state equality, full-step gain |

The highest-value shape settings for a **new model** are usually `K` versus
useful lanes, output width/tile divisibility, decode/prefill `M`, `top_k`, expert
and compact-row distributions, and context/chunk boundaries. For a **new card**,
repeat workgroup, launch-bound/resource, tile, and crossover selection even when
the ISA family is shared. W7900 and XTX are both gfx1100 but differ in memory,
clock/power, and capacity behavior; they must not share a locally tuned winner
unless a cross-card admission explicitly says they may.

The following are not ordinary tuner knobs:

- BF16/FP16 rounding points, reduction association, fast math, relaxed precision,
  quantization, and routing semantics—these require separately named numerical
  variants and their own policy approval;
- a second weight repack, side metadata, or duplicate resident layout without a
  prior memory/capacity admission;
- Wave64, blanket compiler flags, or arbitrary launch-bound menus absent a
  source/code-object premise; and
- any Q4/PARO design already marked closed in §10. Auto-tuning infrastructure
  is not a changed premise and must carry the closure ledger into candidate
  generation.

### 14.4 Ground-truth tuning workflow

Use a staged, lexicographic selector rather than “pick the fastest sample”:

1. **Inventory and reconcile.** Emit the static tensor inventory, dynamic shape
   classes, expected dispatches, and current coverage statuses before creating
   a candidate.
2. **Profile the retained route.** Establish family/leaf Amdahl share and actual
   limiter. Do not tune a cold shape because it merely exists in the file.
3. **Declare a bounded candidate manifest.** Every candidate names its source
   hypothesis, supported shape guard, numerical contract, rollback, and closed
   variants it intentionally excludes.
4. **Prebuild outside timing.** Use hipEngine's source/flags/compiler/arch JIT
   identity. Put rollback and candidates in one code object when practical so
   `dlopen` and unrelated code generation do not contaminate the A/B.
5. **Run correctness first.** Apply primitive, layer, state, graph, and
   multi-prompt oracles from §13. A candidate that changes an unapproved bit or
   route is removed, not assigned a score penalty.
6. **Apply resource gates.** Reject scratch/private spills, unexpected symbols,
   wrong grids, occupancy cliffs, hidden host reads, and memory-budget failures
   before throughput ranking.
7. **Use leaf timing only as a filter.** Time the actual compiled production
   kernel with real layouts/data. Synthetic or proxy kernels may eliminate an
   obviously bad candidate but may not select the retained winner.
8. **Select on the real path.** Run counterbalanced eager and graph A/Bs at the
   canonical and boundary shapes, with real prompt/routing distributions and
   peak-memory accounting. Rank by repeatable full-wall movement; use leaf,
   resources, and calibration cost only as tie-breakers.
9. **Run safety controls.** Shared code needs affected Q3/Q4/PARO and card
   controls. A model-local/card-local result stays scoped to that identity.
10. **Emit an immutable decision artifact.** Store all candidates, failures,
    samples, oracle/resource results, winner, fallback, and coverage delta—not
    only the winning JSON cell.

This order prevents the failure kernel-anvil itself identified: a convenient
proxy can optimize a parameter that the production runtime does not consume.
It also prevents an isolated micro win from replacing a graph route whose
neighboring kernels or numerical association erase the gain.

### 14.5 Offline, first-run, and dynamic behavior

The retained default remains a **static package policy** for known
model/card/software combinations. “Dynamic tuning” should mean dynamic
*selection from already verified variants*, not live search during generation.

Preferred modes:

| Situation | Behavior |
|---|---|
| Known model + admitted card | Load retained package selector; no calibration or compilation in the request path. |
| New model on known backend | Run an explicit offline/provisioning inventory and calibration; use safe generic/known-family fallbacks until admitted. |
| Known model on a new physical card in an admitted architecture | Start from the architecture default, optionally run card-local calibration, and keep the result local until cross-card promotion. |
| New architecture/package | Require native build and correctness bring-up first; do not auto-alias another gfx package and “tune through” missing support. |
| Unknown or stale shape/cache | Fall back, log the uncovered key, and queue it for a later calibration session. Never benchmark it mid-request. |

An optional first-run calibrator is acceptable only if it is explicit and
separable from serving:

- all candidate code objects are prebuilt before timing and before graph
  capture;
- it runs on dedicated scratch with no live KV/recurrent state and validates
  every output against the retained fallback;
- it checks GPU idleness, records power/clock/thermal and software provenance,
  uses warmups and counterbalanced repetitions, and refuses contaminated runs;
- it writes atomically only after the complete candidate set and all gates
  pass; an interrupted or partial file is ignored; and
- serving can skip or disable calibration and always has a deterministic
  fallback.

For variable shapes, runtime may perform an O(1) lookup in a verified piecewise
selector before capture/session construction. It must not sample clocks, switch
variants inside a captured graph, mutate a selector mid-generation, or time a
fraction of user requests. Passive dispatch telemetry may identify an uncovered
shape, but any new A/B happens between sessions under the full workflow above.

Card-local calibration is diagnostic/local policy, not a repository performance
claim. Promoting it to a default still requires clean retained artifacts and the
cross-card/non-regression gates in §13.

### 14.6 Tuning-cache identity and invalidation

The build cache and tuning cache solve different problems. A `.so` cache hit
proves only that source/flags/compiler identity matches; it does not prove that
the winner remains valid for the model, card, route, or correctness policy.

A tuning manifest must include at least:

- schema/generator version and hipEngine revision;
- model file/inventory hash, architecture, quant/layout, and relevant sidecar or
  repack identity;
- backend package, GPU architecture, PCI/device identity, CU count, memory size,
  and the physical card class;
- driver, HIP runtime, `hipcc`/clang version, build profile, wave mode, flags,
  and every candidate `BuildArtifact.cache_key`;
- exact shape/equivalence key, registry key, symbol, grid/workgroup, and
  eager/graph/speculative/KV policy;
- numerical/precision contract and correctness artifact hashes;
- resource reports, timing protocol/samples, peak memory, winner, fallback,
  confidence/noise decision, and timestamp; and
- measurement power/performance profile and contamination checks.

Any mismatch in a correctness-affecting field, model inventory, backend/arch,
kernel build key, compiler/runtime, or route contract invalidates the entry.
Do not “nearest-match” an unknown shape or reuse W7900 calibration on XTX merely
because both report gfx1100. Fall back and recalibrate. Writes must be atomic and
checksummed, and stale/corrupt manifests must degrade to the retained default,
not fail model startup.

### 14.7 Original adoption order (historical)

The `#11 -> #20 -> #15 -> #21 -> #16` campaign described here has completed.
The sequence below is retained as methodology provenance; new work follows the
current post-prefill roadmap and does not reopen its rejected candidates.

1. **Coverage auditor first (CPU/offline capable).** Extend model inventory and
   dispatch reporting to rank-3 experts and emit the status matrix above. Join a
   later real trace rather than pretending static inference proves dispatch.
2. **Variant/decision schema second.** Reuse the existing four-axis registry and
   deterministic HIP build cache; add a separate tuning-manifest schema and
   closure-aware finite candidate descriptors.
3. **First bounded consumer: Q3 D1.** After D0 and after D1A/D1B variants exist,
   compare only retained local256, exact local64, exact all-top-8, and any
   specifically justified fixed-shape/codegen sibling. The framework must not
   generate IQ3 rowtile8, Wave64, or repack retries.
4. **New-model/card qualification next.** Use the auditor and calibrator to map
   new tensor/dispatch shapes and to derive card-local crossover tables. Keep
   unknown shapes on safe fallbacks until evidence exists.
5. **Optional dynamic lookup last.** Add no serving-time calibration until
   offline manifests, invalidation, fallback, graph construction, and telemetry
   are proven robust.

For existing Q4/PARO models, the first output should be a **coverage audit**, not
a fresh sweep. Reopen a kernel family only if the audit finds a real hot fallback
or a new model/card changes a documented source/layout/resource premise. If the
tuner repeatedly chooses different winners, wins only on proxies, cannot clear
the full-wall noise floor, or requires prompt-conditioned keys, retain the
static default and mark the experiment closed.

---

## 15. Original prioritized implementation handoff (historical)

The table records the source review's implementation order. Its numbered tasks
are closed or superseded; see **Current post-prefill roadmap** for live task IDs.

| Order / owner | Work item | Formats / phase | Expected scale | Dependency | Promotion signal |
|---:|---|---|---|---|---|
| 1 / #11 | Expert-major raw IQ3 gate/up + IQ4 down | Q3 prefill | Large | Existing compact scheduler | Strong 512 gain, larger 4K gain, exact BF16, no duplicate weights |
| 2 / #20 | Hierarchical exact top-k | Q3/Q4/PARO decode | Sub-1% | Stable #11 route | Exact IDs/weights, zero spill, measured off/on wall |
| 3 / #15-D0 | Fresh selected-region decode profile | Q3 decode | Attribution only | #11 and #20 with rollback | Named family/leaf/launch/resource Amdahl at 512/1K/4K |
| 4 / #15-D1A | Exact local64 raw IQ4_XS selected down | Q3 decode | Unknown; bounded low-risk | D0 confirms family | Bitwise slot outputs, zero spill, intended symbol and full-wall gain |
| 5 / #15-D1B | Exact all-top-8 four-output IQ4 down + selected contraction | Q3 decode | Potentially material | D1A evidence/stable ABI | Exact slot BF16 + selected sum, 512-block route, leaf/family/full gain |
| 6 / #15-D1C | Targeted IQ3 address/codegen cleanup | Q3 decode | Small/conditional | D1B moves bottleneck | ISA/counter premise and production-shaped gain; no rowtile8 retry |
| 7 / #21 | MoE selected/shared/residual tail + next input RMS | Q3/Q4/PARO decode | ~0.3-1% | Stable D1 selected-sum ABI and #20 | Exact raw+norm outputs and graph gain |
| 8 / #16 | Routed/shared two-stream DAG or broader profile-backed fusion | Q3/Q4/PARO | 0-3%, uncertain | #15 and #21 | Real overlap and >=1% wall; gfx1151 queue policy unchanged |
| Defer | IQ WMMA/repack | Q3 | Potentially large, high cost | Only if raw grouped/down schedules stall | Must beat raw paths without memory duplication |
| Closed | Full-column register GDN | All | Compiler rejected | New algebra required | Do not retry current design |
| Closed | Generic Q4 selected retuning | Q4/PARO | Existing sweep exhausted | New source/layout evidence | Do not repeat historical sweeps by analogy |

This historical table is no longer the active task graph. Current tasks #27–#32
encode the measured post-prefill work and their actual dependencies; device
ownership must still be checked immediately before any benchmark.

---

## 16. Evidence map

### qwen-kernel

- `BASELINE-PERF.md` — same-model W7900/XTX 512/1K/4K baselines and protocol.
- `docs/amd-opt/REPORT.md` — active-byte/practical roofs, stage attribution,
  retained/failed decode experiments, grouped prefill, and row-tile results.
- `docs/amd-opt/NOTES.md` — chronological AB evidence and exactness gates.
- `shaders/moe_group_pairs.comp` — expert counting sort.
- `shaders/moe_gateup_iq3_grouped.comp` — expert-major IQ3 reuse.
- `shaders/moe_down_iq4_grouped.comp` — expert-major IQ4 prefill reuse.
- `shaders/moe_gateup_iq3_rowtile.comp` — retained four-row decode reuse and
  the source shape whose eight-row variant failed.
- `shaders/moe_down_iq4.comp` — all-selected-slot, one-output decode work
  distribution; its pre-weighted F32 association is reference, not HIP ABI.
- `shaders/moe_route_select_hier.comp` — exact hierarchical top-k.
- `shaders/add_rms3.comp` — MoE tail plus next norm.
- `shaders/dn_step_gate.comp` — ACO register-state reference, not a direct HIP
  port target.
- `src/main.cpp` — branch scheduling, barriers, grouped dispatch, and layer
  chaining.

### hipEngine

- `hipengine/kernels/hip_gfx1100/quant/gguf_iq_selected_gemv.hip` — current Q3
  raw selected kernels.
- `hipengine/runtime/qwen35_gguf_runner.py` — Q3/Q4 dispatch, compact scheduler,
  combine, and next input norm boundaries.
- `hipengine/kernels/hip_gfx1100/quant/gguf_q4_k_t16_selected_prefill.hip` —
  mature Q4 grouped WMMA reference.
- `hipengine/kernels/hip_gfx1100/quant/gguf_t16_selected_gemv.hip` — existing
  16-column Q4/Q5/Q6 decode tiling.
- `hipengine/kernels/hip_gfx1100/moe/router.hip` — current repeated-barrier
  selector.
- `hipengine/kernels/hip_gfx1100/fused/paro_combine.hip` — current MoE tail.
- `hipengine/kernels/hip_gfx1100/fused/gguf_ops.hip` — separate GGUF RMSNorm.
- `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` — shared GDN family.
- `hipengine/runtime/qwen35_paro.py` — PARO layer/combine/norm boundaries.
- `benchmarks/results/2026-07-19-gfx1100-qwen36-q3-selected-decode-kernels.json`
  — Q3 real-shape primitive geometry, resources, and correctness-first gate.
- `benchmarks/results/2026-07-19-w7900-qwen36-q3-k-m-benchmark.json` — retained
  hipEngine Q3 same-card wall baseline.
- `docs/HIP-vs-VULKAN.md` — timing-contract-v2 gfx1100/gfx1151 backend matrix
  and production-slice interpretation.
- `docs/LESSONS-LEARNED.md` — small-K expert-down underfill, graph/fusion, and
  HIP/Vulkan transfer lessons.
- `docs/ROOFLINE-gfx1151.md` — 256 GB/s theoretical and ~221 GB/s practical
  Strix Halo memory bounds.
- `benchmarks/results/2026-07-15-gfx1151-gguf-decode-closure-profile.json` —
  current gfx1151 Q4 decode family balance and exact launch-only closure.
- `benchmarks/results/2026-07-16-gfx1100-gguf-gdn-nonvolatile-exact-rollback.json`
  — admitted exact GDN rollback.
- `benchmarks/results/2026-05-22-hipengine-qwen36-35b-a3b-q4km-q4ks-selected-moe-down64-rejected.json`
  — Q4 reduction-width correctness/performance rejection.
- `benchmarks/results/2026-05-18-hipengine-qwen36-35b-a3b-q4km-p9_c3-selected-moe-profile.json`
  — Q4 selected WMMA profile/codegen evidence.

### Auto-tuning methodology reference

- `apollosenvy/kernel-anvil` at `aa100d85de4e1340853d4bdeb20fc4c000aaa80e`
  — AMD/llama.cpp shape-tuning reference, inspected read-only.
- `kernel_anvil/cell_ablation.py` — the useful ground-truth correction: select
  only production-reachable behavior by A/Bing the real model and real kernel.
- `kernel_anvil/gguf.py` and `kernel_anvil/codegen.py` — rank-2 inventory and
  coarse `(quant, N_bucket, K_bucket)` scope that is insufficient for hipEngine
  rank-3 MoE and dynamic shape coverage.
- `kernel_anvil/autoforge.py` and `kernel_anvil/hip_codegen.py` — exact-shape HIP
  generation/sweep ideas, but synthetic zero-data timing and no production
  correctness/integration gate; methodology input, not retained performance
  evidence.

---

## Bottom line

Q3 has two separate optimization stories.

For **prefill**, the source diagnosis was correct and the fully-bulk path now
uses guarded residual-D4 MMQ for two wide Q8 shapes plus exact fallbacks. It
reaches `848.543 tok/s` at 512 and `831.393 tok/s` on the matched mixed-pattern
4K workload on GPU1, with the final 18-workload continuation suite logit-bit
exact. The first changed-algebra tranche is therefore retained and closed, but
the gap to ~3,000 tok/s remains architectural; reopening it requires another
new algebra/layout premise with its own explicit memory and correctness contract.

For **decode**, current GPU1 graph rows are `101.216/108.383 tok/s` at 512/4K.
The same-model W7900/XTX qwen results still prove that approximately 145-190
tok/s is feasible on gfx1100, and the retained HIP/Vulkan matrix gives no reason
to infer a fundamental HIP ceiling. Task #32's final-tree D0 profile records
`8.82493 ms/token` and `671` launches/token. Dense Q8 remains first at
`2.83934 ms/token` (32.17%), ahead of full attention at `1.42134` (16.11%),
lm-head Q6 at `1.05068` (11.91%), weighted IQ4 down at `1.00066` (11.34%),
and IQ3 gate/up at `0.70532` (7.99%). Task #32 then tested exactly one new
source-backed premise from that ranking: qwen-kernel's block-serial raw-Q8 work
mapping. Its source association improved representative `8192x2048`,
`4096x2048`, and `2048x4096` leaves by 53.02%, 54.76%, and 34.26%, but changed
full logits at every required context. Emulating the existing reduction
association restored BF16 bit equality and zero scratch, yet regressed the same
leaves by 29.51%, 20.65%, and 80.26%. The candidate is removed and the retained
`101.216/108.383 tok/s` rows are unchanged. This bounded local campaign is
closed; reopening decode requires a genuinely different algorithm or resident
layout, not another hierarchical-top-k, IQ4-tile, rowtile/repack, stream, or
raw-Q8 reduction-geometry retry.

For **concurrent and speculative execution**, task #29 retains native exact
C=2/4/8 serving, task #30 retains a separately materialized/executed blk.40
NextN proposer, and task #31 completes exact end-to-end wiring through the
shared row-shaped verifier, transactional accept/commit, and graph buckets.
The first matched economics gate is materially negative, so public GGUF MTP is
disabled by evidence rather than by an approval or ABI blocker.

For future models and cards, bounded auto-tuning should make coverage and
portability systematic: reconcile every tensor and runtime shape with its
actual registry symbol, preserve intentional fallbacks and closure evidence,
measure only source-justified exact candidates on production paths, and cache
winners under complete model/card/software/build identity. Optional dynamic
behavior is a verified lookup chosen before session/graph construction—not
live compilation or request-path benchmarking.

For Q4 and PARO, a distributed profile still means no single qwen-derived
90%-share fix—not that the formats are incapable of becoming faster. The
cross-format transfer tranche is now measured: hierarchical exact top-k was
rejected, aggregate MoE-tail/next-RMS was retained where exact and profitable,
and eager routed/shared overlap produced zero concurrent kernel time and was
removed. Future work needs genuinely new profile-backed algorithms/layouts.
Treat Vulkan command replay and ACO register allocation as backend-specific
evidence, keep the documented HIP negative sweeps closed, and do not transfer
the gfx1100 150-190 tok/s target to bandwidth-limited gfx1151.
