# Flash-Next Strix Journey Campaign

> Current state: [`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md). This file is a dated record of the September 13-15 journey and is not updated to reflect new state.

Date: September 13, 2026 UTC (runs span September 13-14 JST).
Status: September 15 JST recovery review complete; guarded Q8/QSA restored.
Optimization execution resumed September 15 JST. The reconciled
[remaining inventory](QWEN4EXP-JOURNEY-REMAINING.md) tracks every next
candidate from this campaign and the runtime review.

### Active Execution

The user has authorized testing every remaining mechanism and adopting
qualified improvements. Each experiment gets an immutable worklog entry and
compact result. No item is closed by source review alone.

[Current measured checkpoint](../benchmarks/results/2026-09-14-journey-progress/README.md)
records the two adopted defaults, all completed numerical/owner/I/O probes,
failed candidates, and work not yet executed.

| Work | State | Next evidence |
| --- | --- | --- |
| J2 named production versus strict | Guarded Q8/QSA restoration validated | 594 short, 780 canonical-base and 516 overlapping sparse-decode rows match strict; short task/repeat/state gates pass; no complete-EOS certification |
| J2 prefill/decode owner census | Measured four-category4K | MoE~6.1-6.3s, linear~3.46s, GR~1.46s, QSA~1.33s, GDN~0.79-0.85s; decode48 graphs |
| J2 PLE/routing/repair census | Partial | PLE CPU/read evidence measured; real expert-population and repair telemetry remain |
| J3 PLE sorted/deduplicated mmap | Screen in progress | Initial sorted-unique wins canonical rows but loses all-unique controls; test copy elision separately |
| J3 bounded pread / persistent workers | Screened, unpromoted | Warm unique rows regress; workers improve cold unique gathers6.3-6.7x versus random mmap; no safe cache-state policy |
| J3 copy elision / staging | Screened / pending | Copy-elision full12 one-pair model screen exact but timing mixed; async staging still pending |
| J3 mapping-only random advice | Adopted | Exact72-sample cold/warm matrices, neutral five-pair warm replication; cold PP3.095/2.198/1.287x |
| J4 GDN DPP suffix | Kernel retained, restored composition not promoted | Corrected-Q8/QSA short/depth numerics pass; targeted EOS explanation fails predeclared task rule ([review](../benchmarks/results/2026-09-15-journey-gdn-restoration/README.md)) |
| J4 multi-column | Numerical pass; task criterion blocks promotion | Corrected-production short/depth gates pass; complete quantization answer has a native-support task finding. Historical owner speed is not current-stack timing ([review](../benchmarks/results/2026-09-15-journey-gdn-multi-restoration/task-review.json)) |
| J4 serial prefix | Pending | No blind all-layer widening; full numerical gates required |
| J5 chunk2048/4096 | Qualified explicit tradeoffs; automatic workspace selection open | 2048 matched4K PP/TG +2.32/+1.22%;4096 independently +3.94/+1.11%. All scoped quality gates pass, but short costs keep default1024. No direct2048/4096 speed claim ([evidence](../benchmarks/results/2026-09-15-chunk4096-accounting/README.md)) |
| J5 Q8 residual-weight down | Rejected/removed | Model top1 587/594 and prefill-last mean fail despite better weight MSE |
| J5 guarded Q8 block-scale down | Adopted | FP32 block scales plus empirical sparse repair; same-session combined Q8/QSA PP +4.34/+4.73/+38.48% |
| J5 other mixed-quant matrix / routing | Pending | Real expert populations and repair costs, complete-owner comparisons |
| J6 HC/BF16/conv/gather fusions | Pending | Last-reader census and operation-complete gates |
| J7 QSA indexer/top-k/packing/attention | Quad prefill and ordered/v2 decode restored; other work pending | 780-row base plus targeted 516-row gate; complete requests improve, individual decode categories mixed |
| J8 graph/PM4 transport | Pending | Exposed submission bound, exact runtime-pin review, isolated qualification if justified |
| J8 host backend refresh | Adopted | Exact 72-sample cache A/B: PP +3.29/+1.59/+1.88%; all complete requests improve, 4K TG -1.36%; no PM4 or arithmetic change |
| J9 batched MTP target verification | Pending | Rejection-depth/accepted-prefix state gates and full-suite true-AR economics |
| J1 external compatibility gaps / J10 closure | Pending | No unchanged replay of known faults; final counterbalanced comparison |

The historical no-override production baseline at `6ba40769f` fails its
numerical gate despite passing repeat determinism and teardown:
[baseline evidence](../benchmarks/results/2026-09-14-journey-production-baseline/README.md).
Overall mean KL0.000625/top1 590/594 passes, but max KL0.054642,
prefill-last mean/p95 and Japanese mean fail. Four deterministic free32
differences require task review rather than automatic semantic rejection.
This is an incumbent failure, not a new candidate regression. Arithmetic
promotion of that composition is blocked. Subsequent localization and recovery
are recorded in `benchmarks/results/2026-09-14-q8-prefill-numerics/README.md`.

The explicit Q8-down/GDN fallback also fails prefill-last scope despite
lower maxKL0.024408. A separate two-Q8 fallback passes automatic numerical
limits but fails the predeclared factual task check. Conservative recovery,
retaining exact optimized owners, matches strict on1374 short/depth rows
and was initially selected by the UD-Q4_K_XL production binder. Owner costs and graph
gaps now come from four fresh4K category captures. Decode has48 graphs and
~1.3ms intra-graph gaps, not a measured PM4 speedup. PLE/mapping and exact
DPP are measured improvements; DPP's tiled suffix is inactive during recovery.

The [September 15 JST restoration](../benchmarks/results/2026-09-15-q8-blockscale-restoration/README.md)
now selects corrected guarded Q8 and verified QSA paths. Dense MMQ fails
canonical depth; dense/GR restoration fails as a combination, not an
individual attribution. Tiled GDN and other disabled MoE paths are not
thereby proven defective. The earlier factual task failure stands; neither
short exact outputs nor depth parity establish complete-EOS reliability.
Final timing uses 108 unique samples/36 warmups with shared unchanged
decode graphs and no overlapping tests. The earlier cost timings are not
used as its denominator.

Independent September 14 JST audit:
[Prefill and Wilkin audit](PREFILL-WILKIN-INDEPENDENT-AUDIT.md).
The six dense-prefill review findings are repaired in `70dd4055a`.
The audit independently checks the journey mechanisms against active code,
distinguishes the numerical GDN suffix boundary from a missing optimization,
and leaves J2 profiling/quality and all new performance promotions open.

**Overlap restart:** the user reported a resolved GPU/CPU process overlap
when resuming the interrupted current-HEAD HIP AR/MTP comparison. Its old
attempt is excluded from timing evidence. Both AR and MTP restart with new
warmups and three repetitions under
`/tmp/hipengine-journey-restart-20260913`; preflight found no benchmark/compiler
process, 0% GPU utilization and approximately 111 GiB available. The shared
branch advanced to `00602e556` through another agent's commits; those are
preserved. The journey driver, Flash-Next runner and profile files are unchanged.
Earlier timing screens are provisional because the overlap interval
was not specified. Fresh HE and current Vulkan timing arms are collected
again in clean worktree `/tmp/hipengine-journey-clean-20260913` at `00602e556`;
new results below replace the earlier timing arms. Earlier API failures remain observations, not a
contention-free numerical diagnosis. The source-proven F16 graph restriction
is independent of that timing uncertainty.

The restarted current-HEAD HIP natural AR arm completes all 30 measured
cases. The first MTP warmup then faults in `k_get_rows_float<float, float>`
with `HSA_STATUS_ERROR_MEMORY_FAULT`; the kernel journal identifies the same
owned server PID 1785193 at September 13 23:35:43 JST. No MTP timing ratio is
valid for that attempt. The serial queue was paused; the process exited and
an independent 1024-element HIP add smoke passed with max absolute error 0.
No GPU reset, runtime restore or driver change was performed before continuing
with the independent Vulkan lane. Diagnose draft/target gather-buffer identity
and lifetime before repeating this HIP MTP configuration.

The fresh Vulkan B2/d16 pair completes 30 measured cases per arm, all
deterministic and exact to its own AR, with positive draft telemetry.
Complete-request rates are 16.031 AR and 16.894 MTP tok/s (1.0538x).
Train/heldout ratios are 1.0616x/1.0427x; code/English/Japanese/mixed are
1.0472x/1.0743x/0.9764x/1.1282x. Japanese regresses, so this is not a
category-nonregressive promotion precedent. Raw results and exact commands:
`/tmp/hipengine-journey-restart-20260913/halobox-head-vulkan-mtp.json`
and `controller.json`. AR and MTP servers ran sequentially, not as
counterbalanced pairs.

## Objective And Authority

Improve Qwen3.8-Flash-Next on the Framework Desktop by applying the useful
mechanisms and experimental method in pwilkin's optimization journey to our
current implementation. Keep the existing **UD-Q4_K_XL target, Q8_0 MTP
sidecar and BF16 KV**. No alternate target quantization is authorized by this
campaign. AR prefill, context-conditioned AR decode, and MTP economics are
separate outcomes.

This document owns the new journey-guided execution queue. The older
[performance campaign](QWEN3.8-FLASH-NEXT-PERFORMANCE-CAMPAIGN.md) is the
historical experiment/evidence ledger; its rejected candidates and correctness
gates are not erased. [EXECUTION-PROFILES.md](EXECUTION-PROFILES.md),
[BENCHMARK.md](BENCHMARK.md), [TESTING.md](TESTING.md) and the four-axis
registry remain normative. This planning task does not promote kernels,
change runtime defaults, recalibrate numerical limits, or certify performance
parity.

## Frozen Inputs

- Host: `gfx1151`, Framework Desktop, machine ID
  `55ea6c509d0b49eea8de7094a1023668`, Ryzen AI Max+ 395 / Radeon 8060S,
  40 CUs, 128 GB installed UMA. Never substitute the older `zbook` host.
- Starting runtime code: main `a0dd4c47485a03d067816f71cb95e03dffe22309`;
  benchmark-driver commit `7bd913ae4896f588adbef0c02fa5708e891903ab` changes
  no inference code.
- Target:
  `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`,
  four shards, model revision `8bdc666649440e9bdc97e16f3f75782c98478ff5`.
  Revalidate the four-shard fingerprint, not only its first filename.
- Draft:
  `/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`,
  expected SHA-256
  `9db03a687670608286e99b563fcc86d0ee76c8dd863f64b2afc0b54eb0eb975d`.
- Local ROCm 10.0 platform, HIP 7.15.26333, clang 23
  `8f497e0992fb7513f7f78a6f6b6f1056c375e961`, Linux
  `7.1.6-1-cachyos`. Initial governor: performance, GPU policy: high.
  Capture loaded libraries and policies per arm. Do not install the external
  ROCm fork over this environment.

## Source Review

Primary sources, pinned September 13:

| Source | Pin | Scope |
| --- | --- | --- |
| [Journey and installer](https://github.com/pwilkin/strix-halo/tree/98bc5b2a2a8009543c167a31f80b9dbb2aa90e33) | `98bc5b2a2a8009543c167a31f80b9dbb2aa90e33` | `journey.html`, `index.html`, `install.sh` |
| [Integration fork](https://github.com/pwilkin/llama.cpp/tree/f5daaa3cfa6358e5dd398911ec741813745a5440) | `f5daaa3cfa6358e5dd398911ec741813745a5440` | Primary same-artifact comparator |
| [Halo-box transfer](https://github.com/pwilkin/llama.cpp/tree/27fd0cf2cd068447f4e0b530f258be6349661420) | `27fd0cf2cd068447f4e0b530f258be6349661420` | Separate graph-agnostic patchset; not the integration binary |
| [Current halo-box HEAD](https://github.com/halo-box/strix-llama.cpp/tree/654803517b06da47f5210553a661bf6c80deb97f) | `654803517b06da47f5210553a661bf6c80deb97f` | User-requested additional HIP and Vulkan controls; HEAD resolved September 13, commit September 12 21:25:34 +01:00 |
| [ROCm fork](https://github.com/pwilkin/rocm-systems/tree/7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1) | `7dda3ac6cfe6bbe0b7f08c23a67cfa118d8641a1` | Retained command-list API/design; source-reviewed, not installed |

### What The Journey Establishes

The author's September 12 walk reports 191.25 -> 1160.02 tok/s pp16384,
with its largest step at tiled GDN (499.52 -> 1181.77). Final-stack
one-variable ablations identify additional costs hidden by the earlier GDN
bottleneck: BF16 GEMM, lazy PLE reads, gate/mix fusion and mask construction.
These are **author measurements**, not Framework hipEngine measurements:
IQ4_NL target, FP16 KV, pp16384/tg128, ubatch24576, depth0/40000,
three repetitions, external retained-PM4 runtime. The later installer uses
ubatch16384 and an IQ4_NL-PROJFIX target with a different shared MTP sidecar.
Do not turn the headline into an expected UD speedup.

The reproducibility command in `journey.html` omits the environment gates
set by `install.sh`. At the pinned source, `LLAMA_MMB`, `LLAMA_HC_GATEMIX`,
and `LLAMA_QSA_FA_V3` default off. A bare build plus that command is not a
fully enabled reproduction. Freeze both command and environment, and verify
selected kernels rather than inferring engagement from requested flags.

Important caveats from the source:

- `ggml/src/ggml-cuda/mmb.cu:ggml_cuda_mmb_supported_mmid` accepts
  **IQ4_NL only**. Dense support covers IQ4_NL, Q8_0, optional BF16/F32,
  and Q6_K only with an existing BF16 shadow. Our Q4_K/Q5_K/Q5_1 expert
  path does not inherit the headline routed-GEMM speedup.
- `qsa.cu:ggml_cuda_flash_attn_ext_qsa_supported` requires F16 K/V and
  F16 packed K/V, D256, 12 query heads per KV head and qualifying shapes.
  BF16 KV rejects that leaf. A graph that still prepares sparse metadata
  before a fallback can pay for both paths; profile the complete owner.
- The journey's whole-sparse-off ablation is approximately neutral even at
  depth140000, while sparse-graph-without-sparse-kernel is slower.
  Dense attention is not automatically the same model computation as
  indexed QSA. Do not remove learned selection under a "more exact" label:
  changing selected attention positions is a model-policy change.
- `f5daaa3c` is newer than the last measured table point `7abec5c24`.
  It fixes Q6_K selection without a shadow and frees shadow allocations.
  The earlier operator tests did not expose that shape.
- The halo-box transfer explicitly excludes QSA (incompatible graph slots),
  tiled GDN (its base already has a more advanced version), and HC fusions
  (different existing layout). Its reported operator pass uses BF16-only
  marking features **off**; recycled raw tensor addresses can otherwise
  match stale marks. Do not enable integration-only flags on this branch.
- Our pinned MTP file has 34 tensors, including root-level `output_hc_norm`,
  `output_hc_down` and `output_hc_up`. Integration `qwen4exp.cpp:load_tensors`
  instead requires layer-level `NEXTN_HC_HEAD_*` tensors. Record the actual
  load outcome; a renamed/reconverted draft is a different artifact unless
  byte-level semantic equivalence and metadata compatibility are established.
  The halo-box transfer still uses root HC tensors in its MTP graph.
- `src/llama-lazy-reader.h` sorts row/slot pairs, deduplicates within each
  worker range, retries EINTR/partial pread, dequantizes each read and scatters
  duplicates. It creates worker threads per gather, not a persistent pool.
- The ROCm contribution document assigns each command list to one graph
  packet batch, keeps in-flight references and queue-specific scratch leases,
  invalidates on graph updates, and falls back before publication. No
  fallback after partial execution is permitted. Its model evidence is not
  Flash-Next evidence. The site says these graphs do not engage in its
  prefill walk; this is a decode hypothesis, not the explanation for pp gains.

### What Current Halo-Box Already Has

The September 13 follow-up adds current halo-box HEAD, not the older
`b212548e0` comparator or pwilkin's earlier halo transfer base. Its source
already contains DPP/multi-column tiled GDN (`gated_delta_net.cu`), fused
HC/grouped matvec and GDN decode chains, device speculative checkpoints,
quant-aware compact MMQ and Vulkan batch-column splitting. It does **not**
contain `mmb.cu` at this pin. Its on-disk n-gram feature
(`--ngram-on-disk`, `--ngram-cache`, `--ngram-io-threads`) is an independent
candidate/control for PLE I/O, so direct reads are not uniquely pwilkin's.
`src/llama-ple-disk.cpp` provides persistent workers, sorted unique rows,
aligned direct I/O and a raw-row direct-mapped cache; defaults are 64 workers,
256 MiB cache, O_DIRECT enabled. Contrast with the integration reader's
per-gather thread creation and buffered pread. Keep row-cache hit rates and
warm/cold distinctions visible; a row cache is not a prompt/KV cache, but
repeated corpus results need a new-row workload before generalizing.

The transfer branch and current HEAD have byte-identical
`gated_delta_net.cu` (SHA-256 `a3164ebe189dc65c77e04f7958a3e21d41d5935f56e05359eb208b8874fcccfe`)
and `hyperconn.cu`
(`f81776dd4c82c9d91ed10d7287e8d62e6e705dde40f2e8ca36961c68d02819f4`).

Its README explicitly warns that asynchronous HIP batched inference can
produce wrong outputs on gfx1151 and its CI uses `HIP_LAUNCH_BLOCKING=1`.
The new HEAD HIP correctness-workaround baseline must record that setting;
asynchronous timing is a separate diagnostic, not automatically a valid fast
competitor. This warning is a hypothesis for the earlier integration failures,
not proof of their cause. Vulkan uses RADV, records selected queue/device and
allows the graphics queue; timing/performance loggers remain off.

Compare complete same-artifact engine rows first. Attribution of a unique
pwilkin gain requires a same-base patch/ablation pair; comparing `27fd0cf2`
with newer `65480351` mixes the patch with intervening halo-box changes.
Do not ascribe all timing differences to pwilkin's added kernels.

### Source-To-Code Map

Host paths below are relative to `hipengine/`; kernel-family shorthand is
relative to `hipengine/kernels/hip_gfx1100/`. These shared implementation files
are registered by the gfx1151 peer backend. New work remains in this tree.

| Journey mechanism / source commit | Current implementation | Experiment and falsifier |
| --- | --- | --- |
| Lazy PLE reader, `ddaf5214b`, `src/llama-lazy-reader.h` | `loading/qwen4_exp_materialize.py:Qwen4ExpPLEMMapTable.gather_rows` uses indexed mmap + NumPy dequant; staging ring, cache advice and telemetry already exist | Sorted/deduplicated bounded pread batches, then asynchronous staging. Reject if full read/dequant/scatter/H2D wall or cold/warm request wall loses. |
| Tiled GDN/DPP, `964c6f2f0`, `gated_delta_net.cu` | `linear_attn/qwen4_exp_gdn.hip` already has tile16 register-state recurrence; runner selects it in admitted scope. `warp_sum_f32` still expresses XOR shuffles | Inspect generated ISA first. Test DPP/permlanex16 and multi-column reuse only where it removes real instructions; reject unchanged ISA or complete-owner loss. Audit serial-prefix and suffix separately. |
| BF16 WMMA, `e55085251`, `mmb.cu` | `quant/gguf_q4_k_selected_prefill`, `gguf_q5_k_q8_1_selected_prefill`, `qwen4_exp_q5_1`, and `gguf_q8_0_prefill` implement grouped WMMA or three-plane iu8 risk+repair; mixed quant owners differ by layer | Quant-native on-the-fly dequant BF16/F16 WMMA with actual expert routing; cap dense shadows. Production numerical gate, not bit-identity, decides eligibility. Charge pack/route/repair/setup/memory. |
| HC combine/norm and gate/mix, `90aba037c`, `6ec4a5f0d` | `fused/qwen4_exp_gr.{hip,py}` and `runtime/qwen4_exp_runner.py:run_qwen4_exp_gr_read`; several fusions and iu8 GR projections already present | Census remaining gate/residual writes and consumers. Fuse only actual redundant materialization, retaining registered unfused primitives. |
| BF16 end-to-end streams, `816667e9d` | Authoritative BF16 residual branches coexist with F32 gate/PLE/recurrent computation | Prove last-reader/lifetime ownership before removing F32 mirrors. Generation-owned identity, never process-global raw-pointer marks. |
| Conv/get_rows and small fusions, `0540c6948`, `2e3ee2afa` | Bulk PLE/GDN convolution and selected weighted reductions already implemented | Inspect remaining gather/mean/conv/norm boundaries and load/store bytes. Do not repeat completed bulk-conv ports. |
| QSA + D256 FA, `df8ad5b10`, `03733c23f` | `attention/qwen4_exp_qsa{,_flash}.{hip,py}` includes ordered-v2 BF16 selected attention and strict fallback | Preserve ordered selected positions and KVLiveSpans; compare complete selection/packing/attention cost. Never port only the graph or silently use dense attention. |
| MMQ route compaction, `be39c4ff0` | Device expert maps and prior skew/padding/next-K experiments exist | Re-measure routing for larger chunks, then population-shaped tiles. Old padding bounds were at specific chunks; do not universalize them or rerun unchanged negatives. |
| MTP HC tensors/shared ownership, `dda64492f` | `loading/qwen4_exp_mtp_*`, `runtime/qwen4_exp_mtp.py`, `generation/qwen4_exp_mtp.py` | Verify current sidecar compatibility; shared target tensors are ownership, not a new draft. Profile draft, serial target verification, restore/replay and head separately. |
| Retained PM4, ROCm `91155796e` | Existing per-layer/MoE graphs, merged position uploads, in-tree transport and lifecycle work | Measure eligible graph coverage and exposed submission wall first; isolate external runtime in a separate prefix/process. Never install globally or attribute kernel gains to runtime transport. |

## Measurement Contract

1. **Canonical AR screen:** existing exact-token four-category fixture,
   p512/p1024/p4096, 128 post-first-output transitions (129 visible outputs),
   c1, production, chunk1024, BF16 KV, one warmup and three repetitions.
   External server gets identical token arrays, `cache_prompt=false`,
   `ignore_eos=true`, greedy, and verified processed-token counts. Record
   batch/ubatch separately; tune external batches as a declared additional arm.
2. **MTP baseline:** `scripts/qwen4exp_journey_mtp_bench.py`, all ten canonical
   prompts, fixed six-train/four-heldout split, B2, 16 output tokens, capacity
   1024, one warmup per case/arm and three repetitions. True AR is a separate
   no-MTP call/server. Report complete-request tok/s as **request throughput**,
   never decode-only TG. Each engine compares with its own AR IDs.
3. **Method boundaries:** direct synchronized HE and HTTP external timers
   are named, not disguised as identical public-service overhead. Initial
   external arms are sequential screens, not paired statistical wins.
   Warmed repeated prompts do not establish cold PLE performance. No
   throughput promotion follows from repeat IDs alone.
4. **Long shapes:** allocation-only probes before 16K/40K/64K/150K inference.
   Use `qwen4exp_chunk_memory_probe.py` for chunk scratch and the capacity
   protocol in HARNESSES; never bracket capacity with full prompts. Capture
   actual unique-token/category PLE locality, not just repeated benchmark
   material. Keep context depth distinct from appended prefill length.
5. **Final comparison:** five counterbalanced pairs on this physical host,
   matched cache/clock/thermal policies; full samples, median, CV and paired
   95% confidence intervals. No cross-host old->new rates. Keep BF16 KV
   binding; any FP16 KV arm is a separately labeled diagnostic configuration.

### Initial Observations

The fresh post-overlap production canonical screen completes 36 measured rows:

| Engine / configuration | p512 PP / TG | p1024 PP / TG | p4096 PP / TG |
| --- | ---: | ---: | ---: |
| hipEngine production, chunk1024, BF16 KV | 297.114 / 20.396 | 316.907 / 19.711 | 294.071 / 19.166 |
| halo-box HEAD HIP, launch blocking, ubatch1024, on-disk PLE | 427.063 / 15.441 | 571.258 / 15.347 | Unverified |
| halo-box HEAD Vulkan, ubatch4096, on-disk PLE | 423.376 / 27.477 | 475.121 / 27.104 | 506.261 / 25.721 |

Rates are token/time-weighted tok/s on the named Framework, one warmup and
three repetitions, exact fixture and 128 AR transitions. Every measured
case repeats its IDs; tracked teardown is zero. This is an incumbent
performance screen, not a fresh production numerical/task certificate.
The largest HE per-case PP CV is 2.30%, so it is not a <=2% parity freeze.
The HIP subset consumes unchanged canonical token arrays for all four categories
at 512/1024, one warmup and three repetitions (24 measured rows); all IDs repeat.
The prior full HIP attempts failed at 4K and remain pre-overlap observations.
Vulkan completes all 36 measured rows with repeatable IDs. These are different
engine configurations in sequential windows, not a paired attribution of
individual kernel gains. They establish a useful local prefill gap without
borrowing the author's IQ4_NL rates. The pwilkin transfer's earlier
261.735/508.153/781.389 PP screen remains provisional and is not ranked
against these fresh runs; its natural AR prerequisite also failed before
overlap clearance. Unique pwilkin performance is therefore still unproven.

The fresh Framework full-category MTP baseline at `00602e556`
is deterministic and own-AR exact in all 30 measured cases. Complete-request
throughput is **14.949 AR versus 11.197 MTP tok/s (0.7490x)** at B2/d16;
heldout is 0.7428x, and category ratios are code 0.7504, English 0.7754,
Japanese 0.7599 and mixed 0.7167. All 480 outputs per arm are counted and
tracked teardown is zero. These are not decode-only TG rates.

The recorded MTP census shows **450 serial target verification rows for
450 post-first-output transitions**. Across measured MTP requests, target
verification is 21.340 s, target prefill/hidden export 10.743 s, proposals
3.476 s, acceptance control 0.818 ms, and draft trim 8.348 ms, against
42.870 s complete request wall. These are instrumented host phase windows,
not a device-only Amdahl ledger; draft stage subwindows overlap proposal.
There are 348 accepted drafts out of 528 proposals, yet no target row is
saved. J9 must address target verification/body execution before acceptance
micro-tuning. Do not reuse the old zbook 0.7407x as this host's denominator.

Integration-fork compatibility attempts:

- Full installer kernel gates with our BF16 KV abort during graph reserve:
  `qwen4exp.cpp:1395` asserts F16 K/V in the direct-index graph. No measured
  samples. The controller's health wait was interrupted after the child
  had already exited with SIGABRT; this is not a GPU reset.
- `LLAMA_QSA_DIRECT_INDICES=0` preserves selected-mask attention and serves
  all four p512 warmups. The first p1024 request returns HTTP500:
  `The model produced output that does not match the expected Content-only
  format`, with visibly malformed output in the server log. No measured
  samples. Do not promote its warmup timing. Root cause is not established;
  BF16-only marking is the next controlled compatibility arm, not a proven
  diagnosis.
- With BF16-only marks disabled as well, all p512/p1024 warmups complete,
  but the first p4096 request returns the same HTTP500 with malformed output.
  The issue is not closed by removing the marks. Later discovery of halo-box's
  documented asynchronous HIP issue motivates a separately labeled blocking
  control; it does not retroactively validate these runs.

Exact commands, source/model/binary hashes, samples and outcomes are in the
[compact baseline artifact](../benchmarks/results/2026-09-13-framework-qwen4exp-strix-journey-baselines.json).
The post-overlap short HIP screen is recorded separately from the provisional
earlier three-shape comparisons.

## Plan And Punchlist

Run the next incomplete item in dependency order. Ranking of performance
items is provisional until J2; reorder by measured recoverable request
milliseconds after every accepted change.

- [x] **J0: Source review.** Pin journey, installer, integration, halo transfer
  and ROCm sources; map precision, graph and shape restrictions to our code.
- [ ] **J1: Starting baselines.** Complete local canonical AR and full-suite
  AR/MTP plus integration-fork same-target/same-sidecar attempts. Store failures
  as failures. Verify no hidden quant/KV/prompt-cache substitution. Pin the
  halo-transfer binary as a separate lane if integration compatibility blocks.
  Include September 13 halo-box HEAD `65480351` in both HIP and Vulkan, as
  requested; HIP uses its documented launch-blocking correctness workaround.
  Attempt inventory, fresh HE AR/MTP, current Vulkan AR/MTP and current HIP
  short AR are available. A complete comparison matrix is not closed:
  current HIP MTP faults, current HIP4K is not recertified, and the earlier
  pwilkin natural AR prerequisites failed. Re-run those provisional timing
  arms before deriving paired speed claims; do not repeat the fresh completed
  HE/Vulkan matrices merely because another comparator is blocked.
- [ ] **J2: Owner and correctness refresh.** Capture complete prefill and
  decode owner costs, launch/copy census, PLE fault/read time, repair rates,
  expert populations, runtime gaps and current profile manifest. Use cached-only
  profiling after warm compilation. Recheck incumbent production versus strict
  with 500-1000 shared-chain rows before arithmetic promotion. Historical
  incumbent-envelope discrepancies are not permission to widen limits.
- [ ] **J3: PLE I/O.** RED tests: duplicate/unsorted/empty rows, quantized row
  offsets, short read/EINTR/EOF, last row, cancellation, reuse and teardown.
  Separate warm-cache, file-scoped cold and realistic long-unique corpora.
  Compare sorted mmap, bounded pread and persistent workers independently,
  including current halo-box's cached direct-I/O reader as a source/control.
- [ ] **J4: GDN instruction/parallelism audit.** Check ISA and profile serial
  prefix versus already-tiled suffix. Screen DPP and multi-column workgroup
  layouts with all recurrent outputs, chunk tails and carried state. A prior
  tile16 admission does not close a new DPP mechanism, nor imply a 2.37x gain.
- [ ] **J5: Larger-chunk MoE/dense prefill.** Probe allocation first, then
  sweep 1024/2048/4096 and eligible larger chunks on diverse real routing.
  Test dequant-on-load matrix instructions for actual Q4_K/Q5_K/Q5_1/Q8_0
  roles, with bounded shadows where justified. Preserve all prior failed
  numerical/throughput evidence and declare exactly what differs.
- [ ] **J6: Remaining HC/conv/gather fusions.** Use tensor-consumer and
  traffic census; remove redundant writes, not required publication. Test
  fallback chains, recycled allocations and interleaved requests.
- [ ] **J7: Complete BF16 QSA owner.** Time indexer/top-k/packing/attention,
  preserve tie ordering, causal boundaries and live spans at 2051/2052,
  4096/4097 and chunk/page tails. No dense-for-sparse policy substitution.
- [ ] **J8: Decode graphs and transport.** Measure graph engagement, submission
  and synchronization costs with unprofiled controls. Qualify runtime fork
  separately only when graph coverage shows a useful upper bound; require
  update/destroy/in-flight lifetime, scratch, profiler fallback and cancellation
  gates. No unsafe global ROCm replacement or unqualified-ASIC override.
- [ ] **J9: MTP economics and capacity.** Use all categories and heldouts.
  Verify every rejection depth, accepted prefix, target/draft state and
  rollback. Optimize target body execution before budget tuning; compact-head
  or high acceptance alone cannot overcome serial verification. Extend >=4K
  capacity only with allocation and state gates, then measure at that depth.
- [ ] **J10: Closure.** Full applicable correctness/task/lifecycle packet,
  paired final comparator, shipped defaults for qualified wins, complete
  artifact/rollup/changelog, kernel lineage/catalog and refactor cleanup.
  Record remaining blockers precisely; do not label baseline collection
  campaign completion.

### Per-Experiment Record

Before coding, record source file+commit, hypothesis, measured baseline request
wall `W`, comparator wall `C`, exclusive owner cost `O`, achievable owner
speed factor `s`, and arithmetic class. Maximum saving is `O`; projected
saving `O*(1-1/s)` is a hypothesis. Include overlaps, memory/setup costs,
shape/quant coverage, oracle and registered fallback.

Execute RED, complete-owner screen, profile engagement, then same-session
whole-model A/B and the full binding quality gate. Keep exact non-regressive
microsecond/launch/copy wins even if headline variance hides the gain.
For arithmetic changes, require exact control/ownership plus mean KL <=1e-3,
p95<=5e-3, p99<=2e-2, max<=5e-2, overall top1>=99% and each scope>=97%,
determinism/isolation and task non-inferiority. A failed bit-identity check is
not a rejection without production review. Full-model FP8 is the potential
external teacher; do not call BF16 KV a BF16 model oracle.

Repeat the journey's two useful views: sequential one-change measurements,
then one-variable ablations on the finished package. Keep neutral/rejected
rows, re-profile after bottlenecks move, and never multiply historical
per-step ratios into a projected product headline.

## Completion Gates

- AR: qualified production matches/beats the best correct same-artifact
  comparator independently for each canonical PP and TG row; no averaging
  away a category or decode regression.
- MTP: exact to own declared target policy, deterministic, full/train/heldout
  and every category non-regressive versus same-protocol true AR. Keep
  automatic MTP off until this passes. Aim for >=1.5x true AR or the best
  correctness-valid same-config competitor, whichever is higher.
- Long context: only measured admitted lengths count; allocated capacity is
  not a successful 150K retrieval or 262K generation.
- Evidence: clean runtime source, reproducible build/runtime environment,
  all samples and quality results, current catalog/refactor/worklog/rollups,
  and atomic commits. No changes to the public README as an experiment diary.
