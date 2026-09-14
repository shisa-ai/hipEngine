# Independent Prefill And Wilkin Audit

Review date: September 14, 2026 JST (September 13 UTC).
hipEngine source: `70dd4055a`, whose parent runtime is unchanged by the
six-finding evidence repair. Status: source/evidence audit complete; new
performance candidates are not implemented or qualified by this document.

Execution update: actual testing and two default promotions are recorded in
[the execution checkpoint](../benchmarks/results/2026-09-14-journey-progress/README.md)
and the live journey table. Mapping-only random PLE advice and exact DPP
suffix reductions are adopted. Multi-column GDN is numerically blocked,
Q8 residual-weight down is rejected/removed, larger chunks exceed the current
scratch model, and the incumbent numerical envelope still fails. The
remaining inventory is not claimed complete.

## Separate The Two Workloads

The dense Qwen3.8-27B Q4_K_M comparison and the Flash-Next UD-Q4_K_XL journey
are separate experiments. The former's Q4/Q5 MMQ table ablation cannot
attribute the latter's mixed-expert prefill gap. Neither the author's
IQ4_NL/FP16-KV headline nor a different physical host supplies our denominator.

For Flash-Next, keep the four-shard UD-Q4_K_XL target, existing Q8_0 MTP
sidecar, BF16 KV and Framework identity from
[the journey campaign](QWEN4EXP-STRIX-JOURNEY-CAMPAIGN.md). Preserve its
uncontended completed HE/Vulkan baselines; external HIP4K/MTP failures do
not require repeating those completed runs.

## Six Findings: Disposition

| Finding | Repair | What remains unknown |
| --- | --- | --- |
| Incorrect occupancy | `scripts/gguf_prefill_kernel_resources.py` reports 1536 physical wave32 VGPRs, 24-register granule and a register-only ceiling; CU/WGP aggregate fields are null | Actual residency, stalls and memory traffic |
| Vacuous finite-logit check | New profile runs read nonempty logits after timing and check `isfinite`; historical derived flags are null | Historical logits were never fetched and cannot be certified retrospectively |
| Hardcoded Q6 gate success | Generator rejects empty/failed 18-prompt gates, nonexact screens and baseline-state disagreement before publishing | Broader qualification beyond the recorded scope |
| Incorrect universal col16 mechanism | Derived block counts distinguish row256 and row512 arms; source-count slowdown explanation withdrawn | Why each candidate loses on this machine |
| Logical bytes treated as physical traffic | Artifacts label logical volume and withdraw bandwidth exclusions | Cache transactions, repeated loads and bandwidth/stall limits |
| Mutable measurement provenance | Col16 source/model identity is frozen and raw inputs hashed; deterministic `--check` | Original raw runs do not contain a complete machine snapshot |

Validation: 15 repair tests plus 57 backend unit tests pass. Original timing
samples are unchanged, both deterministic checks pass, and no production
dispatch or published throughput changed.

The Q4 removal is byte-identical to its pre-candidate state for all four
reverted implementation/test paths. This justifies rejecting the measured
arms, not a ban on every 16-column implementation or a declaration that all
tile geometry is solved.

## Independently Inspected Sources

The local external repositories were read-only. These are fixed revisions,
not claims about today's upstream HEAD:

| Repository | Revision | Inspected paths |
| --- | --- | --- |
| pwilkin/strix-halo | `98bc5b2a2a8009543c167a31f80b9dbb2aa90e33` | `journey.html`, `install.sh` |
| pwilkin/llama.cpp integration | `f5daaa3cfa6358e5dd398911ec741813745a5440` | `mmb.cu`, `qsa.cu`, `gated_delta_net.cu`, `src/llama-lazy-reader.h`, commit changes for lightning indexer, MMQ and HC |
| pwilkin/llama.cpp transfer | `27fd0cf2cd068447f4e0b530f258be6349661420` | Revision verified; distinct from integration and current halo-box |
| halo-box/strix-llama.cpp | `654803517b06da47f5210553a661bf6c80deb97f` | `src/llama-ple-disk.cpp`, previously reviewed GDN/MMQ/attention sources |
| pwilkin/rocm-systems | `78d1160060bb6ada29b3b21e20c998a48161b257` | `projects/clr/hipamd/src/hip_graph_internal.cpp` retained PM4 engagement guard |

The journey plan names a newer ROCm pin `7dda3ac6...`. This independent audit
does not substitute the older inspected implementation for that newer runtime;
review the exact proposed runtime revision before any private-prefix build.
No external runtime was installed or changed.

## Mechanism Inventory

### PLE Reader And Host Staging

Integration `llama-lazy-reader.h::gather` sorts row/slot pairs, splits them
between workers, retries short reads/EINTR, and scatters duplicate rows.
Deduplication is within a worker range; duplicate rows at split boundaries
can still be read more than once. Threads are created per gather.

Current halo-box `llama-ple-disk.cpp` independently provides persistent
workers, unique rows, aligned direct I/O and a raw-row cache. It is not
evidence unique to pwilkin.

Our `hipengine/loading/qwen4_exp_materialize.py:Qwen4ExpPLEMMapTable.gather_rows`
indexes the mmap by requested indices then dequantizes every selected row.
It has semantic bounds, lifecycle checks and opt-in telemetry, but does not
deduplicate those dequantizations. This is a genuine source-level difference.

First experiment: sorted unique mmap rows with inverse scatter, retaining
original output order, dtype and independent result ownership. Screen
all-unique, highly duplicated, unsorted, empty and realistic n-gram accesses.
Only then consider bounded pread and persistent workers separately. Warm
page-cache, cold file-scoped cache and new-row workloads must be distinct.
Do not claim the author's reader ratio for our mmap/dequant/H2D owner.

Required gates: quant row offsets, first/last semantic row, duplicates,
empty inputs, invalid indices, short read/EINTR/EOF for a pread variant,
teardown, cancellation, reuse and staging-ring wrap. Complete read/dequant/
scatter/stage/H2D wall decides retention, not isolated syscall latency.

### Tiled GDN, DPP And Multi-Column Reuse

Integration `gated_delta_net.cu:launch_gated_delta_net` selects
`<128,8,8,16>` only for non-KDA H48/D128, one sequence, 16..32768 tokens.
This is eight columns per warp and a 16-token input tile, with explicit
DPP/permlanex16 reductions.

Our `linear_attn/qwen4_exp_gdn.hip` already stages 16 tokens, but each warp
owns one state column and `warp_sum_f32` uses XOR shuffles. It also computes
normalization/decay from different input boundaries; this is not a drop-in
copy of the fork's normalized-input recurrence.

The production profile's `PRODUCTION_GDN_COLWARPS_PREFILL_LAYERS` is
27..47, and the runner checks that layer admission before choosing tiled16.
Earlier layers have a different route. The historical performance campaign
records an all-layer widening failure at mean KL 0.0068: do not simply widen
the suffix flag or call the remaining prefix a missed optimization.

First experiments: inspect loaded-code ISA, separate prefix/suffix costs,
then DPP-only and multi-column variants independently. A textual shuffle
change is not a win if the compiler already emits equivalent instructions.
Test all output tokens, carried state, tails 15/16/17, chunk boundaries and
full production numerical/task gates. Non-bit-identical candidates receive
production review, not automatic rejection.

September14 reverse-isolation follow-up: current admitted GDN over strict
non-GDN arithmetic passes all numerical/scoped gates on594 rows
(mean KL0.0000643436,max0.003782627,top1 593/594), with deterministic
repeats and clean lifecycle. One free trajectory requires task review.
This does not repair or qualify the full production composition, whose large
tail also persists with GDN disabled. Do not attribute that failure solely to
GDN or promote multi-column from its50-row isolated screen.
See `benchmarks/results/2026-09-14-gdn-numerical-isolation/README.md`.

### MMB Matrix Kernels And Mixed-Quant Prefill

Integration `mmb.cu::ggml_cuda_mmb_supported_mmid` accepts IQ4_NL only.
Dense support includes IQ4_NL/Q8_0 and qualified BF16/F32 routes; Q6_K
requires an already prepared BF16 shadow. Our mixed Q4_K/Q5_K/Q5_1 expert
weights cannot use that routed kernel unchanged.

Useful mechanisms: dequant-on-load matrix instructions, activation reuse,
population-shaped expert tiles and fused gate/up publication. Our current
quant kernels already implement grouped WMMA and iu8/risk/repair routes.
The comparison must include route/pack/dequant/repair/reduction and setup,
and use actual per-expert populations rather than balanced synthetic routing.

First measure owner costs and repair rates by quant/role/layer. For larger
chunks, run allocation probes before full prompts, then screen admitted
1024/2048/4096 shapes. Dense shadows count against persistent memory and
context capacity. No wholesale BF16 expansion or target-quant substitution.

The existing dense Qwen3.8 Q6 retile is a separate validated dispatch change,
not evidence that Flash-Next's experts should inherit its geometry.

### HC Combine, Gate/Mix And BF16 Streams

Journey commits `90aba037c`, `6ec4a5f0d`, `816667e9d` cover distinct
mechanisms. Our `runtime/qwen4_exp_runner.py:run_qwen4_exp_gr_read` already
has grouped norm, iu8 projection choices and registered fused-up/gated-mean
paths. Some admitted branches still materialize intermediate gates.

First census consumers, writes and lifetimes for each branch. A fused
projection-plus-gate/mix is useful only if it removes a real materialization
or improves the complete owner. Keep precision boundaries explicit and count
fused-kernel resource changes. Our BF16 residual state does not imply all
F32 intermediates are redundant.

Do not adopt process-global tensor-address marks or shadow maps as ownership.
The integration source uses raw tensor/data pointer keys; recycled allocation
identity needs a generation-scoped owner and last-reader proof here.

### Conv, Gather, Norm And Weighted Reductions

The journey groups multiple small fusions in some ablations. Our bulk
PLE/GDN convolution and reduction implementations already exist. First list
the remaining concat/transpose/conv/get_rows/norm boundaries on the active
route. Porting a named fusion again without checking consumers is not work.

Retain only operations that remove actual traffic/launches and pass state,
layout, tail and unfused-fallback gates. A contiguous vector-copy leaf does
not justify changing strided gather semantics or authoritative state writes.

### QSA Selection, Indexer, Mask And Attention

The integration QSA leaf requires F16 K/V and packed buffers, D256 and
12 query heads per KV head. Our BF16 selected-position ABI rejects that leaf
as written. Adapting storage and arithmetic is a new kernel qualification.

Our runner uses a dense prefix where all positions fit the selection budget,
then computes scores/top-k and passes explicit ordered selected positions.
The score path includes `qsa_score_f32_kernel`; indexer matrix scoring is
therefore a distinct candidate from optimizing attention itself. Reassociation
can change selected blocks at threshold ties, so exact selection ownership
must be checked even when score KL looks small.

Inventory four separate costs: index projection/pooling; scoring/top-k;
selected-key/value packing; attention/output. Preserve causal boundaries,
stable tie subsets, increasing selected order and KVLiveSpans. Test 2051/2052,
4096/4097, page/chunk tails and carried state. Do not replace learned sparse
selection with dense attention or pay for selection plus an unintended dense
fallback.

Maskless graph construction is only an opportunity if our current route
actually allocates/writes the redundant dense mask. The fork's large mask
ablation is not proof that hipEngine has that cost.

Multi-CTA top-k thresholding is plausible at depth, but must feed stable
compaction. Atomic gather order is not our deterministic selection contract.
Large-context allocation and correct completion precede performance claims.

### Retained PM4 And Graph Submission

The inspected HIP graph implementation only attempts retained PM4 when
dispatch activity tracing is disabled. The installer also disables HIP graphs
when `ENABLE_RETAINED_PM4=0`: that switch is not a PM4-versus-AQL ablation.

Our existing native PM4 work is not a missing concept, and gfx1100 transport
qualification does not grant gfx1151 admission. Inspect actual graph coverage
and exposed submission gaps first. A prefill kernel bottleneck is not repaired
by graph transport.

Only if justified: isolated prefix/runtime, with current HIP graphs, custom
ordinary-AQL graphs and custom PM4 graphs as separate arms. No global runtime
replacement or unqualified-ASIC override. Require in-flight update/destroy,
scratch lease, fallback-before-publication, cancellation and cross-request
ownership tests, plus unprofiled engagement-logged timing.

### MTP Tensors, Verification And Economics

Draft tensor-name compatibility is a prerequisite, not a speed optimization.
Keep the existing Q8_0 sidecar; a renamed/reconverted draft needs a separate
semantic identity review.

Our `generation/qwen4_exp_mtp.py` loops through candidates and calls
`target.step(root_token, capture_hidden_seed=True)` once per verified row.
Accepted proposals do not save target body execution on that route. This
source-level fact agrees with the recorded local phase census; acceptance
bookkeeping is not the first target.

First design a genuinely batched target verifier with accepted-prefix state
ownership, rejection replay, cancellation and exact control. Measure hidden
export, proposal, verification, commit/replay and head cost without adding
overlapping subwindows. Full category/heldout true-AR economics decide
admission; no budget or acceptance-score tuning can replace saved body work.

## Execution Queue

The independent source audit establishes opportunities, not measured ranking.
Next work should use the campaign's J2 owner/numerical refresh:

1. Identify active loaded kernels, source/manifest, physical host and profile.
   Keep prefill, decode, PLE and MTP windows separate and use interval unions
   where events overlap. Correct logits validation must run outside timing.
2. Recheck incumbent numerical quality on shared-chain rows before changing
   arithmetic. An existing discrepancy is not permission to widen limits.
3. Rank by recoverable complete-request milliseconds, then run bounded
   independent experiments for PLE, GDN, matrix owners and actual fusions.
4. Keep every qualified non-regressive improvement and re-profile the new
   package. Record failed candidates without unsupported universal exclusions.
5. Qualify long-context QSA, transport and batched verification separately;
   do not rerun a known faulting external configuration without a new hypothesis.

No new speedup, runtime default, long-context capacity or MTP qualification is
claimed by this audit. Hardware experiments and full campaign closure remain
explicit work, not implied by completion of the source inventory.
