# HIP vs Vulkan Attribution Results And Plan

This document records the HIP vs Vulkan/RADV attribution suite and the retained
conclusions from it. It started as a plan for turning "Vulkan/RADV/ACO is better
than HIP/LLVM here" into actionable compiler or backend work; it is now also the
project ledger for what the microbenchmarks have actually proven. Retained
results must follow the evidence policy in `docs/BENCHMARK.md`.

The motivating observation from `docs/ROOFLINE.md` and
`docs/MTP-LLAMACPP-PARITY.md` is narrow:

- hipEngine HIP can beat llama.cpp HIP on exact AR decode.
- llama.cpp Vulkan is still the higher absolute backend ceiling on several
  GGUF decode/prefill rows.
- The suspected causes are a mix of compiler quality, workgroup/subgroup
  shape, launch/runtime behavior, fusion topology, and quant precision/layout.

The goal of this suite is to split those causes cleanly enough to decide
whether the next high-leverage path is an LLVM issue, a HIP kernel rewrite, a
Vulkan backend, or a tiny hand-ISA path.

## Current Conclusion

As of the retained gfx1151/STRIX_HALO runs on 2026-07-08, the retained
conclusion is split. The short answer is **no**: the evidence does not support
"HIP is slower simply because Mesa RADV/ACO is better optimized than
LLVM-AMDGPU" as a single explanation.

The practical conclusion is:

- Locally, there is nothing else broad to test on gfx1151 before changing
  engineering direction. The remaining useful tests are decision-gated, not
  open attribution debt: cross-GPU portability reruns, a narrow Q4_K HIP
  recovery experiment if we want to act on the measured Q4 delta, a true
  production-registry Q4 Vulkan probe if backend work becomes product-relevant,
  and new production slices only when profiling exposes them as shipped hot
  buckets.
- Stop broad gfx1151 attribution sweeps for now. Dispatch, f32 geometry,
  VOPD, memory/waitcnt, dot-path, HIP wave64, HIP fixed-shape, HIP q8_1
  real-slice layout, dense Q8_0 real-slice probes, RADV shaderstats, HIP
  fixed-wave64 geometry, and two selected Vulkan real slices are already
  retained. LDS/barrier/subgroup and 4/8/16 lane-local accumulator reduction
  controls are also retained for the f32 reduction question, and the positive
  Q4_K Vulkan slice now has a targeted HIP/RADV ISA comparison plus a bounded
  setup/amortization probe. The negative Q6_K X8 slice also now has a targeted
  HIP/RADV ISA comparison. The sampler/top-1 and deterministic top-k8 argmax
  bucket now has matched HIP/Vulkan diagnostic rows as well.
- The matched real-slice evidence is now split and shape-specific: Q6_K
  selected-down X8 is slower on Vulkan, Q4_K selected-dual gate/up is faster on
  Vulkan, and dense raw Q8_0 mostly favors HIP except for a few smaller
  `768x2048` rowtile cases. The targeted Q4_K HIP/RADV ISA comparison explains
  what that one win is not: not missing HIP dot4, not HIP spills, and not RADV
  VOPD pairing. It is useful slice-specific evidence, but still not a broad
  `compiler_aco` proof. The targeted Q6_K X8 comparison points the other
  direction: the memory-heavy production-shaped slice is faster on HIP even
  though both paths emit dot4 and neither spills, so the synthetic
  memory/waitcnt win does not transfer there.
- Do not start a production Vulkan backend from the current evidence. The
  dispatch win is real, but it is a runtime result; the retained
  production-shaped slices disagree by quant/layout shape, and the new dense
  Q8_0 row is mostly negative for Vulkan. The Q4_K
  setup/amortization probe shows the standalone steady win requires persistent
  device/pipeline/buffer residency: if all measured standalone backend setup is
  charged to each call, the `47.9 ms` setup cost needs roughly `908` retained
  Q4 quantize+dot calls to amortize. Production Vulkan still needs another
  retained hot-slice win or a true registry/end-to-end probe that moves wall
  time.
- Do not start a broad hand-ISA path from the current generic VOPD or
  packed-dot evidence. HIP already emits VOPD in the retained VOPD rows and
  dot4 in the retained q8/q4/q6 rows. The Q4_K selected-dual slice now has a
  targeted ISA comparison, and it still does not show the easy misses
  hand-ISA would normally fix first. Hand-ISA only becomes reasonable for a
  specific hot HIP slice after we isolate a concrete avoidable instruction,
  waitcnt, reduction, or addressing sequence.

Stop condition for local gfx1151 work:

- Do not add more generic VOPD, wave64, fixed-block, dispatch-only, dot-lowering,
  memory/waitcnt, LDS/subgroup, accumulator, or standalone Q4/Q6/Q8 real-slice
  sweeps. They will not change the current engineering decision.
- Do run cross-GPU retained-suite reruns when gfx1100/W7900 or 7900 XTX is
  available, because portability is the one broad question still open.
- Do run a narrow HIP Q4_K selected-dual recovery only if the patch attacks the
  retained Q4 instruction/waitcnt/reduction delta and is judged against the same
  real-slice oracle and timing.
- Do build a production-registry Vulkan Q4 probe only if we are prepared to test
  persistent residency and end-to-end wall time, because standalone replay has
  already answered what it can.

The detailed retained reads are:

- Vulkan/RADV command-buffer replay has a real runtime-dispatch advantage for
  launch-heavy tiny-kernel bursts. This is retained as `runtime_dispatch`, not
  `compiler_aco`.
- Vulkan remains much faster on the matched repeat-shifted f32 GEMV/reduction
  harness, and both backends prefer wg256 on all retained best-native rows.
  This rules out the simple "HIP picked the wrong workgroup size" explanation
  for that harness.
- The first retained ISA/stat extraction does **not** support a simple "ACO
  wins through VOPD pairing" story for this f32 geometry harness. HIP emits two
  static VOPD instructions while RADV emits none in the final shader disassembly
  for the tested wg64/wg256 shapes. HIP also reports no scratch or spills.
- The targeted VOPD scheduling sweep also does **not** support the idea that
  RADV/ACO's ceiling comes from better VOPD pairing on gfx1151. HIP emits VOPD
  in all retained VOPD rows; RADV emits zero VOPD in all final shader
  disassemblies. Vulkan is only modestly faster in the heavier independent and
  mixed/dequant rows, while HIP is faster in independent-2, independent-4, and
  dependent-4.
- The retained memory/waitcnt sweep is the first direct evidence that Vulkan has
  a real advantage on memory-side scheduling/access shapes. On matched
  device-memory load+accumulate rows, Vulkan is `1.02x-2.25x` faster for most
  coalesced, strided, and interleave variants; gather is essentially tied
  (`1.02x`). HIP and RADV both report no scratch and no spills in the refreshed
  shaderstats artifacts. Simple rows show slightly fewer RADV waitcnt-family
  instructions; fixed-shape controls do not close the gap. However, the first
  memory-heavy production-shaped transfer check is negative: Q6_K X8
  selected-down is faster on HIP, and the targeted ISA join shows RADV has more
  static instructions and waitcnt-family instructions in that dot shader. Do
  not file an LLVM-AMDGPU waitcnt/scheduling issue from the synthetic memory
  sweep alone.
- The retained packed dot-path sweep rules out a missing-HIP-dot-instruction
  story for the current q8/q4/q6 idiom. HIP and RADV both emit final dot4
  instructions for q8 signed, q4 unsigned-byte by signed-q8, and q6 zero-point
  correction rows; HIP reports no scratch/spills. Vulkan is still `3.28x-3.42x`
  faster, including the scalar-dequant row (`3.30x`). After the HIP wave64 and
  fixed-block controls, the remaining dot-path gap is more likely surrounding
  scheduling or layout/activation quantization economics than basic dot
  lowering, wave mode, or runtime block indexing.
- The first retained HIP wave64 controls do not close the gap. On packed-dot
  rows, forcing HIP wave64 makes HIP `1.007x-1.061x` slower than the retained
  wave32 HIP rows. On memory/waitcnt rows, HIP wave64 is mixed but still leaves
  Vulkan faster on most shapes; gather regresses `6.35x` versus HIP wave32.
- The retained HIP fixed-shape controls also do not close the gap. Dot-path
  fixed block indexing is flat versus same-commit runtime HIP
  (`0.993x-1.000x` fixed/runtime) and Vulkan remains `3.31x-3.43x` faster.
  Memory fixed block indexing is mixed (`0.906x-1.290x` fixed/runtime) and
  Vulkan remains faster on every retained row (`1.04x-2.36x`). Fixed-workgroup
  geometry improves some HIP wg256 rows by up to `6.3%`, but Vulkan still leads
  best-native geometry by `5.56x-14.03x`.
- The retained HIP fixed-workgroup plus wave64 geometry control removes the
  remaining f32 geometry wave-mode confound and still does not close the gap.
  HIP fixed-wave64 is `1.13x-1.23x` slower than HIP fixed-wave32 on
  best-native rows, and Vulkan remains `6.31x-16.18x` faster than HIP
  fixed-wave64. HIP fixed-wave64 reports wave64, `11` VGPR / `20` SGPR,
  no scratch/spills, and `0` VOPD for wg64/wg256.
- The retained LDS/barrier/subgroup reduction sweep does not identify reduction
  topology as the missing f32 geometry switch. On K=512/2048/8192 rows=1
  wg64/wg256, HIP extra barriers are only `1.002x-1.028x` slower than HIP LDS,
  HIP wave-shuffle reduction is essentially flat versus HIP LDS
  (`0.991x-1.005x`), Vulkan extra barriers are flat versus Vulkan LDS
  (`0.991x-1.005x`), and Vulkan subgroup reduction is mostly flat to modestly
  slower versus Vulkan LDS (`0.984x-1.132x`). Matched Vulkan LDS remains
  `8.19x-14.55x` faster than matched HIP LDS. This is a reduction/topology
  control, not a clean `compiler_aco` proof.
- The retained accumulator reduction sweep extends that negative result to
  4/8/16 lane-local accumulators and one-wave/no-shared-final controls. On
  K=512/2048/8192 rows=1 wg32/wg64/wg256, HIP multi-accumulator variants are
  mostly slower than HIP LDS (`1.02x-2.40x` slower), Vulkan
  multi-accumulator variants are mixed (`0.90x-1.77x` versus Vulkan LDS), and
  matched Vulkan accumulator rows remain `9.57x-15.81x` faster than matched
  HIP. This closes the proposed no-LDS-accumulator diagnostic as a negative
  HIP-recovery result. It still does **not** cover a true two-stage
  block-partial plus final-reduce launch sequence.
- The retained sampler argmax sweeps cover the previously deferred
  sampler/argmax bucket for deterministic top-1 and deterministic top-k8
  reductions. Rows=`1/4/8`, vocab=`32768`, and workgroups=`64/128/256` all
  pass CPU oracles on both backends. Vulkan is `12.75x-26.94x` faster than HIP
  across matched top-1 rows and `12.79x-25.93x` faster across matched top-k8
  rows; best-native wg256 top-k8 rows are HIP `132.701/135.923/162.030 us`
  versus Vulkan `5.7205/5.9910/12.6654 us` for rows `1/4/8`. HIP reports no
  scratch/spills and emits VOPD in both sampler variants; RADV reports no
  scratch/spills and `0` VOPD. This is a real exposed-bucket diagnostic lead,
  but it is not a stochastic sampler, probability-filtered sampler, or fused
  lm-head+sample result.
- The first retained HIP real-slice q8_1 layout controls are positive for HIP:
  Q4_K selected-dual gate/up q8_1 quantize+dp4a is `2.77x` faster than the raw
  selected-dual path, and Q6_K selected-down X8 q8_1 quantize+dp4a is `1.68x`
  faster than the production T16 float path. q8_1 materialization itself is
  small (`0.0025-0.0027 ms`) in these retained slices. This is **not** a Vulkan
  real-slice comparison; it only says q8_1 layout cost is not the blocker on
  the HIP side.
- The first matched Vulkan production-slice probe does **not** transfer the
  synthetic Vulkan dot-path win to the Q6_K selected-down X8 hot shape. On the
  retained rows=8, experts=256, in=512, out=2048 slice, the best Vulkan
  local-size row is local_size=64 at `0.03076 ms` prequantized dot and
  `0.03217 ms` quantize+dot. The retained HIP path is `0.01665 ms` dot and
  `0.01925 ms` quantize+dot, so Vulkan is `1.85x` slower on dot and `1.67x`
  slower on the combined path. This is retained as `real_slice_probe`, not
  `compiler_aco`.
- The targeted HIP/RADV ISA comparison for that Q6_K X8 negative slice shows
  both paths emit the intended `9` final dot4 instructions and neither reports
  scratch/spills. HIP emits wave32 with `30` SGPR / `51` VGPR, `599` static
  instructions, `39` waitcnt-family instructions, and `51` VOPD instructions.
  RADV emits wave64 with official `108` SGPR / `48` VGPR, `1117` static
  instructions, `89` waitcnt-family instructions, and `0` VOPD. This is a
  negative production-transfer result for the broad synthetic memory/waitcnt
  claim, not a missing RADV dot-path or Vulkan-spill problem.
- The second matched Vulkan production-slice probe **does** transfer a real
  Vulkan win to the Q4_K selected-dual gate/up hot shape. On rows=32,
  x_rows=4, experts=256, in=2048, out=512, the best Vulkan local_size=64 row is
  `0.29607 ms` prequantized dot and `0.29238 ms` quantize+dot. The retained HIP
  row is `0.34638 ms` dot and `0.34582 ms` quantize+dot, so Vulkan is `1.17x`
  faster on dot and `1.18x` faster combined. Correctness passes the full CPU
  q8_1+Q4_K selected-dual oracle with top-1 `1.0`, max abs `1`, and mean abs
  `0.03408`. RADV emits `3` final dot4 instructions, `0` VOPD, official
  `48` VGPR / `108` SGPR, and no scratch/spills in the dot shader. This is
  retained as `real_slice_probe`, not `compiler_aco`.
- The targeted HIP/RADV ISA comparison for that Q4_K selected-dual win shows
  both compilers emit the intended `3` final dot4 instructions for the dot
  shader and both report `0` scratch/spills. HIP emits wave32 with
  `31` SGPR / `22` VGPR, `564` static instructions, `35` waitcnt-family
  instructions, and `4` VOPD instructions. RADV emits wave64 with official
  `108` SGPR / `48` VGPR, `526` static instructions, `26` waitcnt-family
  instructions, and `0` VOPD. This preserves Q4_K as a real hot-slice lead, but
  it narrows the likely explanation to surrounding scheduling, reduction/source
  structure, or memory/address code rather than basic dot lowering, spills, or
  dual-issue pairing.
- The bounded Q4_K setup/amortization probe keeps the local_size=64 Vulkan
  steady-state result positive: `0.292745 ms` prequantized dot and
  `0.293117 ms` quantize+dot versus retained HIP `0.34638 ms` and
  `0.34582 ms`. It also measures `47.8645 ms` of standalone backend setup
  before steady replay, dominated by `25.3268 ms` host staging for the
  synthetic dual-weight fixture and `17.4106 ms` Vulkan instance/device setup.
  Pipeline creation is only `0.1736 ms`, device upload is `3.4389 ms`, and
  descriptor plus command recording are tiny. This says Q4 Vulkan is only
  interesting with persistent device/pipeline/buffer residency; it is not a
  one-shot-call backend win.
- The retained dense Q8_0 real-slice probe closes the prior dense q8_0 GEMV /
  attention-projection gap for the tested shapes. HIP and Vulkan both pass CPU
  q8_1/Q8_0 correctness. Across 54 matched rows, Vulkan/HIP speedup ranges from
  `0.279x` to `1.120x` for quantize+dot and `0.238x` to `1.169x` for
  prequantized dot. The only useful Vulkan wins are smaller `768x2048` cases:
  rows=1 row_tile=1 is `1.120x` combined but `0.978x` on dot, rows=1
  row_tile=4 is `1.112x` combined and `1.001x` on dot, and rows=4 row_tile=4
  is `1.115x` combined and `1.169x` on dot. The larger production-shaped rows
  favor HIP; for example `2048x2048` rows=4 row_tile=4 is Vulkan `0.863x`
  combined, and `2048x6144` rows=8 row_tile=4 is Vulkan `0.707x` combined.
  This is negative evidence for making dense Q8_0 a near-term Vulkan target.
- We still cannot claim `compiler_aco` for the f32 geometry gap. The refreshed
  RADV shaderstats extraction reports no Vulkan scratch/spills and official
  `12` VGPR / `108` SGPR allocation for the retained wg64/wg256 shaders, while
  HIP fixed-wave64 reports `11` VGPR / `20` SGPR and no scratch/spills. That
  removes both the missing-allocation-stat blocker and the broad wave-mode
  blocker, but the f32 geometry row is still `diagnostic_unclassified` because
  source/runtime structure and memory/address scheduling remain confounds.

What we have ruled out:

- Do not attribute the current Vulkan ceiling to RADV/ACO finding VOPD/dual
  issue opportunities that LLVM/HIP misses. The retained gfx1151 VOPD evidence
  points the other way: HIP emits VOPD, RADV does not.
- Do not attribute the f32 geometry gap to HIP spills or scratch use. HIP
  reports `0` scratch and `0` spills in the retained ISA/stat rows.
- Do not attribute the f32 geometry gap only to a bad HIP workgroup-size choice.
  HIP and Vulkan both prefer wg256 in the retained best-native rows.
- Do not attribute the f32 geometry gap to HIP wave32 versus RADV wave64 alone.
  HIP fixed-wave64 makes the best-native rows slower than HIP fixed-wave32 and
  leaves Vulkan much faster.
- Do not attribute the f32 geometry gap to HIP's LDS tree reduction or a missing
  subgroup/shuffle reduction alone. The isolated reduction sweep keeps the
  backend gap after HIP wave-shuffle and Vulkan subgroup controls.
- Do not attribute the f32 geometry gap to missing 4/8/16 lane-local
  accumulator variants. The retained accumulator sweep keeps the backend gap
  and usually slows HIP versus the LDS tree.
- Do not attribute current q8/q4/q6 dot-path gaps to HIP failing to emit dot4.
  HIP emits the expected dot4 instructions in the retained packed-dot rows.
- Do not treat HIP wave64 as the missing switch for the retained dot/memory
  gaps. It does not close dot, and it severely hurts the retained gather row.
- Do not treat HIP runtime `blockDim`/block-indexing overhead as the missing
  switch for retained dot, memory, or f32 geometry gaps. Fixed-shape HIP
  controls are flat or small relative to the remaining Vulkan lead.
- Do not reject q8_1/dp4a purely because of activation materialization cost in
  the tested HIP selected-MoE slices. The measured quantization cost is small
  and the full quantize+dot path is faster than the retained HIP float controls.
- Do not treat dense Q8_0 attention/shared projections as an obvious Vulkan
  backend target from current gfx1151 data. The retained dense Q8_0 probe is
  mostly HIP-faster on larger shapes, with only small-shape Vulkan exceptions.
- Do not pursue a Vulkan port for the Q6_K selected-down X8 q8_1+dp4a slice as
  currently implemented. The matched Vulkan real-slice probe is slower than HIP
  despite passing correctness and using the intended SPIR-V/RADV dot path.
  The targeted ISA comparison also shows the slow Vulkan row is not missing
  dot4 and is not spilling; RADV has more static instructions and waitcnt-family
  instructions in the production-shaped dot shader.
- Do not generalize the Q4_K selected-dual Vulkan win to all quantized GEMV
  paths. The Q4_K slice wins, the Q6_K slice loses, and neither row shows RADV
  VOPD pairing. Treat the Q4_K row as a specific hot-slice lead for backend or
  HIP-codegen follow-up, not as generic proof that ACO is better than LLVM.
- Do not attribute the Q4_K selected-dual Vulkan win to an easy HIP dot/VOPD
  codegen miss. The targeted ISA comparison shows HIP emits dot4 and VOPD with
  no spills; RADV wins this slice with fewer static instructions and fewer
  waitcnt-family instructions while using no VOPD.
- Do not attribute the sampler top-1/top-k8 argmax gap to a simple
  workgroup-size mistake, HIP spills, or missed RADV VOPD. Both backends prefer
  wg256 on best-native sampler rows, both report no scratch/spills, HIP emits
  VOPD, and RADV emits none.

What remains plausible:

- Vulkan has a proven dispatch/runtime advantage on gfx1151.
- Vulkan has a large matched-math advantage in one f32 diagnostic, but that row
  is still `diagnostic_unclassified`.
- Reduction topology and lane-local accumulator count are not the missing f32
  geometry fix. The remaining f32 geometry lead is more likely source/runtime
  structure, address/memory scheduling, or another compiler/runtime effect
  outside the isolated reduction variants.
- Memory/access scheduling remains a useful diagnostic lead, but it is not yet
  an LLVM-AMDGPU filing target. HIP wave64 and fixed block indexing are not the
  fix, refreshed RADV shaderstats remove the allocation-count visibility
  blocker, and the first memory-heavy production-shaped transfer check is
  negative for a generic waitcnt/scheduling claim.
- Dot-instruction availability is no longer untested for the packed q8/q4/q6
  idiom: HIP emits dot4. HIP wave64 also does not close the dot gap. The
  fixed-block control also does not close it. HIP real-slice q8_1 materialization
  is positive, and Vulkan real-slice probes are split: Q6_K X8 selected-down and
  dense Q8_0 are mostly negative while Q4_K selected-dual is positive. A narrow
  hand-ISA sequence is only worth considering for a specific hot slice after the
  remaining layout/stat confounds are gone.
- Real Vulkan inference slices still need caution: the retained production
  probes are split, with Q6_K selected-down negative, dense Q8_0 mostly negative,
  and Q4_K selected-dual positive. The Q4 setup/amortization probe says
  persistent residency is mandatory for that positive slice. Synthetic Vulkan
  dot-path wins should not be promoted without a matching real slice, and one
  real-slice win plus a standalone setup probe is still not enough to justify a
  second production backend.
- Sampler top-1/top-k8 argmax is now a concrete Vulkan diagnostic win in an
  exposed server bucket. The useful HIP follow-up is to profile whether
  sampler work is still exposed after launch fusion and lm-head work, then
  decide between a HIP reduction rewrite, a fused lm-head+sample path, or a
  narrow Vulkan probe. The retained rows do not cover stochastic sampling.
- The most actionable HIP-side work from this suite is not "switch to Vulkan";
  it is to preserve the q8_1/dp4a real-slice wins, keep reducing launches and
  fusion boundaries, isolate memory/waitcnt behavior inside shipped hot
  kernels before filing LLVM work, and decide whether the Q4_K selected-dual
  delta survives integration or can be recovered in HIP with a narrower source,
  reduction, waitcnt/addressing, or inline-asm experiment.

Operationally, keep the Vulkan work as an attribution/probe path until real
inference slices prove production value. The HIP roadmap should first try to
reproduce the useful pieces inside HIP: fixed-shape kernels, dot intrinsics or
small hand-ISA sequences where proven, better launch fusion, and memory/waitcnt
controls. A production Vulkan backend is not justified by the dispatch row,
generic geometry row, one positive Q4_K real-slice probe, or mostly negative
dense Q8_0 rows.

HIP also does not give us a PTX-equivalent escape hatch in the normal runtime
path. We can inspect LLVM IR, AMDGPU assembly, and code-object metadata, and we
can use AMDGCN builtins, inline AMDGCN assembly, or standalone HSACO/module
kernels for narrow cases. But normal HIP source is ultimately relying on
LLVM-AMDGPU codegen, so confirmed compiler misses become either LLVM roadmap
items or carefully scoped hand-ISA candidates.

## What To Test Next

The broad attribution phase is done for gfx1151. The answer to "is there
anything else we should test?" is: yes, but only tests that can change an
implementation decision. Do **not** spend more gfx1151 time on dispatch-only,
broad geometry-only, generic VOPD, generic memory, basic dot-lowering, HIP
wave64, HIP fixed-block, generic LDS/subgroup/accumulator reduction variants,
or more Q4/Q6 standalone real-slice sweeps. Those have already answered the
coarse questions. The remaining useful work is decision-grade only:

Immediate triage:

| Decision | Area | Reason |
| --- | --- | --- |
| Stop for now | Generic VOPD/dual-issue sweeps | Retained gfx1151 rows show HIP emits VOPD and RADV does not; this is not the current ACO win. |
| Stop for now | More HIP wave64 dot/memory rows | Wave64 did not close dot or memory gaps and badly regressed gather. |
| Stop for now | More HIP fixed-block dot/memory controls | Fixed block indexing was flat or mixed and did not close the Vulkan lead. |
| Stop for now | Dispatch-only Vulkan probes | The runtime-dispatch win is already classified and does not prove compiler quality. |
| Stop for now | Vulkan Q6_K selected-down X8 port | The matched real-slice probe is slower than retained HIP by `1.67x` on quantize+dot. |
| Done | Better RADV allocation/stat extraction | `RADV_DEBUG=shaders,shaderstats` now gives official RADV VGPR/SGPR/spill/scratch data for retained Vulkan rows. |
| Done | Vulkan Q4_K selected-dual gate/up slice | The matched real-slice probe is faster than retained HIP by `1.18x` on quantize+dot, but remains slice-specific `real_slice_probe` evidence. |
| Done | Targeted Q4_K selected-dual HIP/RADV ISA comparison | Both paths emit `3` dot4 instructions and no scratch/spills; HIP emits VOPD, RADV emits none, and RADV has fewer static instructions/waitcnts. This narrows but does not finish Q4 attribution. |
| Done | HIP wave64 plus fixed-workgroup geometry | HIP fixed-wave64 is slower than fixed-wave32 and leaves Vulkan `6.31x-16.18x` faster, removing the remaining f32 geometry wave-mode confound. |
| Done | LDS/barrier/subgroup reduction sweep | HIP wave-shuffle is flat versus HIP LDS, Vulkan subgroup is flat to modestly slower versus Vulkan LDS, and matched Vulkan LDS remains `8.19x-14.55x` faster than HIP LDS. |
| Done | Reduction accumulator controls | HIP 4/8/16 lane-local accumulator variants are mostly slower than HIP LDS and matched Vulkan accumulator rows remain `9.57x-15.81x` faster. True two-stage block-partial reduction remains unrun because no retained profile makes it decision-grade. |
| Done, deterministic top-1/top-k8 | Sampler/argmax diagnostic | Vulkan top-1 argmax is `12.75x-26.94x` faster and deterministic top-k8 is `12.79x-25.93x` faster with CPU correctness and no scratch/spills on either backend; stochastic sampling remains separate. |
| Done, bounded | Q4_K selected-dual Vulkan setup/amortization probe | Steady Q4 remains positive, but standalone backend setup is `47.9 ms`, so the win needs persistent device/pipeline/buffer residency and does not justify a production backend by itself. |
| Done, negative for broad LLVM claim | Q6_K X8 memory-heavy production-slice transfer | The matched Vulkan Q6_K X8 slice is `1.67x` slower than HIP combined, and the targeted HIP/RADV ISA comparison shows RADV has more static instructions and waitcnt-family instructions. Do not file a generic LLVM waitcnt/scheduling issue from the synthetic memory sweep alone. |
| Done, mostly negative for Vulkan | Dense Q8_0 production-shaped slice | HIP and Vulkan both pass correctness across retained `768x2048`, `2048x2048`, and `2048x6144` rows. Vulkan only wins a few smaller `768x2048` rowtile cases; larger rows favor HIP, so dense Q8_0 is not a near-term Vulkan target from current gfx1151 data. |
| Test only if it changes HIP implementation priority | Narrow Q4_K HIP source/inline-asm experiment | Useful only if it targets the measured Q4 instruction/waitcnt/reduction delta and validates a real-slice speedup; not justified as a broad hand-ISA path. |

Recommended next tests, in order:

1. Cross-GPU reruns on gfx1100/W7900 and 7900 XTX using the retained harness
   set. This checks whether the split real-slice result and negative VOPD
   conclusion are gfx1151-specific.
2. Narrow HIP Q4_K selected-dual source or inline-asm experiment only if we are
   willing to validate a concrete real-slice recovery. The first ISA comparison
   is already retained and rules out the easy dot4/spill/VOPD explanations; a
   follow-up should target the measured instruction/waitcnt/reduction shape and
   be retained only if it moves the same real slice.
3. True production-registry Vulkan Q4 probe only after we decide the bounded
   probe and cross-GPU portability justify backend work. The retained
   setup/amortization row does not cover hipEngine registry integration or
   full inference wall time.
4. A second memory-bound production slice only if new profiling points to one.
   The Q6_K X8 transfer check is already negative, so another memory issue
   should start from a shipped hot bucket rather than from the synthetic memory
   sweep alone.
5. Another dense Q8_0 production slice only if profiling identifies a different
   dense Q8_0 shape as exposed and backend-deciding. The retained local row is
   mostly negative for Vulkan on larger shapes.

Priority summary:

| Priority | Test | Decision It Enables |
| --- | --- | --- |
| Done | Dot-path q8/q4/q6 kernels with dot ISA counts | HIP emits dot4; remaining gap is not basic dot lowering |
| Done | HIP wave64 dot/memory controls | Wave64 does not close dot/memory gaps and regresses gather |
| Done | HIP fixed-shape memory/dot/geometry controls | Runtime `blockDim`/fixed-shape overhead does not close retained gaps |
| Done | HIP q8_1 real-slice layout controls | q8_1 materialization is small and positive on tested HIP selected-MoE slices |
| Done | Vulkan Q6_K selected-down X8 real-slice probe | Synthetic Vulkan dot-path win does not transfer to this shipped hot bucket |
| Done | Targeted Q6_K X8 HIP/RADV ISA comparison | Q6 negative slice is not missing RADV dot4 or caused by Vulkan spills; RADV has more static instructions/waitcnts, so the synthetic memory/waitcnt claim does not transfer here |
| Done | RADV shaderstats allocation extraction | Official RADV allocation counts show no Vulkan scratch/spills in retained rows |
| Done | Vulkan Q4_K selected-dual gate/up slice | One production-shaped Q4 hot bucket is faster on Vulkan, but not broad `compiler_aco` proof |
| Done | Targeted Q4_K selected-dual HIP/RADV ISA comparison | Q4 win is not missing HIP dot4, HIP spills, or RADV VOPD pairing; remaining lead is narrower scheduling/source/reduction work |
| Done | Dense Q8_0 production-shaped slice | Closes the dense Q8_0 local gap as mostly HIP-faster on larger rows, with small `768x2048` exceptions |
| Done | HIP wave64 plus fixed-workgroup geometry | Wave mode plus fixed workgroup does not close the f32 geometry gap |
| Done | LDS/barrier/subgroup reduction sweep | Reduction topology does not close the f32 geometry gap |
| Done, deterministic top-1/top-k8 | Sampler argmax sweep | Covers deterministic top-1 and top-k8 argmax; stochastic sampling remains separate |
| Done, bounded | Q4_K Vulkan setup/amortization probe | Steady Q4 win needs persistent residency; one-shot setup cost swamps the per-call win |
| P1 | gfx1100/W7900 and 7900 XTX reruns | Check portability of the classified gfx1151 conclusions |
| P3 | Narrow Q4_K HIP recovery experiment | Decide whether the one positive Vulkan slice can be recovered without a Vulkan backend |
| P3 | True production-registry Vulkan Q4 probe | Decide whether a persistent Vulkan path moves real hipEngine wall time |

## Coverage Audit

Current local hardware coverage is gfx1151/RADV STRIX_HALO only. The retained
local results cover the gfx1151 decision-grade attribution list currently
justified by the HIP-vs-Vulkan question; the older broader matrix row status is
audited below so deferred rows are explicit rather than silently treated as
done. Local probes on 2026-07-08 showed HIP `gfx1151` and Vulkan
`AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, Mesa `26.1.2-arch2.1`; no
gfx1100/W7900 or 7900 XTX device is exposed on this host.

| Area | Retained evidence | Coverage state | Current read |
| --- | --- | --- | --- |
| Dispatch/runtime | `dispatch-floor-comparison.json` | Done on gfx1151; cross-GPU external | Vulkan command-buffer replay is much cheaper for tiny launch bursts, classified `runtime_dispatch`, not compiler quality. |
| Workgroup shape | `geometry-sweep-comparison.json`, `geometry-sweep-fixed-workgroup-comparison.json` | Done on gfx1151; cross-GPU external | HIP and Vulkan both prefer wg256 in the f32 geometry harness; fixed HIP workgroups do not close the gap. |
| Wave/subgroup mode | `dot-path-wave64-comparison.json`, `memory-waitcnt-wave64-comparison.json`, `geometry-sweep-fixed-workgroup-wave64-comparison.json`, `reduction-sweep.json` | Done on gfx1151; cross-GPU external | HIP wave64 does not close dot/memory/geometry gaps, and subgroup/shuffle reduction topology does not explain the f32 geometry lead. |
| VOPD / dual issue | `vopd-sweep-comparison.json`, `geometry-isa-stats-comparison.json`, `q4-selected-dual-real-slice-isa-comparison.json`, `q6-x8-real-slice-isa-comparison.json` | Done on gfx1151; cross-GPU external | Current evidence is negative for "ACO wins through VOPD": HIP emits VOPD in retained rows and RADV emits none in final shaders. |
| Register allocation / spills | HIP code-object metadata plus RADV shaderstats across geometry, memory, dot, VOPD, Q4, and Q6 artifacts | Done on gfx1151; cross-GPU external | No retained row shows HIP scratch/spills as the explanation. RADV official allocation counts are recorded, but they do not prove a broad allocation win. |
| Waitcnt / memory scheduling | `memory-waitcnt-comparison.json`, wave64/fixed controls, `q6-x8-real-slice-isa-comparison.json` | Done on gfx1151 for synthetic rows plus first production transfer; cross-GPU external | Synthetic memory rows favor Vulkan, but the first memory-heavy production transfer is negative, so no generic LLVM waitcnt claim is justified. |
| Dot lowering / packed integer path | `dot-path-comparison.json`, Q4/Q6 real-slice ISA comparisons | Done on gfx1151; cross-GPU external | HIP and RADV both emit dot4 in q8/q4/q6 and Q4/Q6 real-slice rows; missing HIP dot4 is ruled out. |
| Layout / quantization economics | `hip-real-q4-selected-dual-q8_1-dp4a.json`, `hip-real-q6-selected-down-x8-q8_1-dp4a.json` | Done for tested HIP selected-MoE slices | q8_1 materialization is small and the HIP q8_1+dp4a path is faster than retained HIP float controls. |
| Production-shaped Vulkan slices | Q6 X8, Q4 selected-dual, and dense Q8_0 Vulkan real-slice artifacts plus ISA joins where retained | Done for the retained hot slices; more slices only if profiling picks one | Result is split: Q6 is slower on Vulkan, Q4 is faster on Vulkan, and dense Q8_0 is mostly HIP-faster on larger rows. Treat Q4 as slice-specific, not broad `compiler_aco`. |
| Sampler top-1/top-k8 argmax | `sampler-argmax-comparison.json`, `sampler-topk8-comparison.json` | Deterministic top-1/top-k8 done on gfx1151; stochastic sampling still separate | Vulkan is `12.75x-26.94x` faster on top-1 and `12.79x-25.93x` faster on top-k8 with no scratch/spills on either backend. This is an exposed-bucket diagnostic, not a full sampler backend. |
| Vulkan setup / amortization | `vulkan-real-q4-selected-dual-q8_1-dp4a-integration.json` | Bounded standalone probe done | Q4 steady replay remains positive, but one-shot setup costs require persistent pipeline/device/buffer residency. |
| Production Vulkan backend | No `vulkan_radv_gfx11` registry backend yet | Not locally testable without backend implementation | Current evidence does not justify starting a second production backend; a registry/end-to-end Q4 probe is future product work, not remaining gfx1151 attribution. |
| Hand-ISA path | Q4/Q6 ISA joins plus generic VOPD/dot rows | No broad path; Q4-only experiment is decision-gated | Only consider a narrow Q4 HIP source/inline-asm recovery if we decide to act on its measured instruction/waitcnt delta. |

### Original Matrix Row Status

This table audits the older proposed matrix rows below against retained
artifacts. `Covered` means the row has direct retained HIP/Vulkan evidence or a
strictly stronger retained replacement. `Covered, scoped` means the production
subcase that mattered was retained, while adjacent subcases are not open unless
fresh profiling exposes them. `Decision-gated` means the exact row was not run
and should not be run as broad attribution work; it needs a production profile
or backend decision that would make the answer actionable.

| Matrix row | Status | Evidence / reason |
| --- | --- | --- |
| no-op kernel, grid sweep | Covered | `dispatch-floor-comparison.json` includes tiny dispatch/grid rows and classifies launch/grid overhead as `runtime_dispatch`. |
| tiny ALU kernel, narrow/wide args | Covered | `dispatch-floor-comparison.json` covers tiny/wide HIP dispatch rows and Vulkan replay. |
| command burst | Covered | `dispatch-floor-comparison.json` covers N-kernel HIP launch/graph versus Vulkan command-buffer replay. |
| dependent chain for compiler body scheduling | Covered by replacement | `vopd-sweep-comparison.json` includes dependent f32 chains with ISA/stat extraction; a separate dispatch-floor dependent-chain row would not change the current conclusion. |
| f32 GEMV row geometry | Covered | `geometry-sweep-comparison.json`, fixed-workgroup controls, and fixed-wave64 controls cover K/rows/workgroup/wave shape. |
| reduction only | Covered | `reduction-sweep.json` covers LDS, extra-barrier LDS, HIP wave-shuffle, and Vulkan subgroup controls; `reduction-accum-sweep.json` adds wg32/wg64/wg256 accumulator controls. |
| selected-MoE index gather | Covered by replacement | `memory-waitcnt-comparison.json` covers gather-ID address behavior, and Q4/Q6 selected real slices cover production selected-row behavior. |
| rows>1 verifier GEMV | Decision-gated | Geometry rows=1/4/8 are covered; HIP rowtile verifier evidence exists in benchmark rollups, but no matched Vulkan rowtile verifier microbench is retained. Run only if verifier profiling makes it a Vulkan/LLVM decision. |
| coalesced, strided, gather, interleave memory/waitcnt | Covered | `memory-waitcnt-comparison.json` plus HIP wave64/fixed-block controls and Q6 X8 production transfer. |
| independent/dependent/dequant/mixed VOPD rows | Covered | `vopd-sweep-comparison.json`; current evidence is negative for RADV VOPD pairing. |
| q8_1 x q4 dot / scalar dequant | Covered by replacement | `dot-path-comparison.json` covers q4 unsigned x q8 and scalar q4 dequant; Q4 selected-dual real slice covers production q4 q8_1+dp4a. |
| q8_1 x q5/q6 selected-down | Covered, scoped | Q6 selected-down X8 is retained for HIP layout, Vulkan real slice, and HIP/RADV ISA. Q5 selected-down has no retained Vulkan row; run only if profiling identifies Q5 as a separate blocker. |
| q8_0 dense GEMV | Covered | `q8-0-dense-real-slice-comparison.json` covers raw Q8_0 dense q8_1+dp4a rows for shapes `768x2048`, `2048x2048`, and `2048x6144`, rows=`1/4/8`, row_tile=`1/4`, with Vulkan local_size=`64/128/256`. Result is mostly HIP-faster on larger shapes. |
| q8_1 activation quantize | Covered | HIP and Vulkan Q4/Q6 real-slice probes include q8_1 quantize timing and correctness. |
| two-stage reduction | Decision-gated | One-row reduction controls did not identify reduction topology as the missing switch. Run two-stage only if a large-K reduction bucket reappears in profiling. |
| no-LDS accumulator reduction | Covered negative | `reduction-accum-sweep.json` covers 4/8/16 lane-local accumulators plus one-wave/no-shared-final controls. Accumulators do not recover HIP and matched Vulkan accumulator rows remain `9.57x-15.81x` faster. |
| barrier stress | Covered | `reduction-sweep.json` includes extra-barrier HIP and Vulkan reduction variants. |
| small-K expert-down | Covered | Q6_K selected-down X8 real-slice probe plus ISA comparison covers the retained small-K selected-down production bucket. |
| selected gate+up dual | Covered | Q4_K selected-dual HIP layout, Vulkan real slice, HIP/RADV ISA comparison, and setup/amortization probe are retained. |
| dense q8_0 attention projection | Covered, scoped | The retained dense Q8_0 probe covers production-shaped attention/shared-projection sizes for the tested shapes and is mostly negative for Vulkan. Run another row only if profiling identifies a different dense Q8_0 shape as backend-deciding. |
| GDN/recurrent chain | Decision-gated | No matched Vulkan GDN/recurrent microbench is retained. Run only if verifier profiling isolates GDN/recurrent scheduling/register pressure as the exposed backend limiter. |
| q6 lm-head rowtile | Decision-gated | HIP rowtile evidence exists in broader benchmark rollups, but no matched Vulkan lm-head rowtile microbench is retained. Run only if lm-head rowtile becomes the backend decision target. |
| sampler/top-k/argmax | Covered, scoped | `sampler-argmax-comparison.json` covers deterministic top-1 argmax and `sampler-topk8-comparison.json` covers deterministic top-k8 with CPU correctness and ISA stats. Stochastic sampling and fused lm-head+sample integration are not covered; run only if sampler work remains exposed after launch fusion and real-slice work. |

Tooling that would improve attribution quality, but should not displace the
remaining portability and integration work:

- A small comparison utility that rolls HIP/Vulkan artifacts into a one-page
  retained-result diff with timing, correctness, wave mode, instruction counts,
  waits, dot/VOPD counts, and classification.

Cross-GPU reruns on gfx1100/W7900 and 7900 XTX are now the most useful
microbenchmark follow-up. They should confirm portability of the retained
gfx1151 conclusions, not restart broad attribution.

## Current Retained Evidence

### gfx1151 Dispatch/Grid Floor

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/dispatch-floor-comparison.json`.
The artifact records exact HIP/Vulkan commands and references the shared
environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: dispatch/grid floor with trivial shader bodies.
- Classification: `runtime_dispatch`.

Retained result:

| Shape | Vulkan command-buffer replay | HIP direct | HIP graph |
| --- | ---: | ---: | ---: |
| 941 dispatches, 1 block, tiny args | `0.043621 us/dispatch` | `2.0087 us/dispatch` | `1.8069 us/dispatch` |

Grid sweep at 941 dispatches:

| Blocks | Vulkan | HIP graph | Vulkan vs HIP graph |
| ---: | ---: | ---: | ---: |
| 1 | `0.042902 us/dispatch` | `1.85857 us/dispatch` | `43.3x` |
| 128 | `0.230992 us/dispatch` | `1.86143 us/dispatch` | `8.1x` |
| 1024 | `1.69237 us/dispatch` | `3.07608 us/dispatch` | `1.8x` |
| 8192 | `11.977879 us/dispatch` | `13.02259 us/dispatch` | `1.09x` |

Conclusion: Vulkan/RADV command-buffer replay is dramatically cheaper than HIP
direct launches and HIP graph replay for launch-heavy one-block bursts on this
gfx1151 system. The gap collapses as grid work grows, so this result supports
kernel fusion, fewer launches, and a narrow Vulkan probe for launch-heavy paths.
It does **not** prove that RADV/ACO emits better math code than LLVM-AMDGPU,
because the shader body is intentionally trivial and no ISA/stat evidence was
collected for compiler scheduling.

Immediate reads:

- HIP graph replay trims the direct-launch floor only modestly in the retained
  N=941 row (`2.0087` to `1.8069 us/dispatch` for tiny args).
- HIP argument count is not the dominant cost in this harness: wide args add
  about `0.05 us/dispatch` at N=941.
- The remaining compiler questions need matched math kernels with disassembly,
  VGPR/SGPR/scratch, waitcnt, VOPD, and dot-instruction evidence.

### gfx1151 F32 GEMV Geometry Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/geometry-sweep-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-geometry-sweep.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-geometry-sweep.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-geometry-sweep.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: repeat-shifted f32 GEMV/reduction, one workgroup per row,
  CPU oracle, K=`512/2048/8192`, rows=`1/4/8`,
  workgroup=`32/64/128/256`, body repeats=`128`.
- Classification: `diagnostic_unclassified`.

Retained best-native shape summary:

| K | Rows | HIP best | Vulkan best | Vulkan vs HIP |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 1 | wg256, `23.5944 us` | wg256, `3.0412 us` | `7.76x` |
| 512 | 4 | wg256, `23.6302 us` | wg256, `3.2942 us` | `7.17x` |
| 512 | 8 | wg256, `23.6439 us` | wg256, `4.0837 us` | `5.79x` |
| 2048 | 1 | wg256, `85.6870 us` | wg256, `7.3849 us` | `11.60x` |
| 2048 | 4 | wg256, `86.2845 us` | wg256, `7.7645 us` | `11.11x` |
| 2048 | 8 | wg256, `82.9754 us` | wg256, `8.7624 us` | `9.47x` |
| 8192 | 1 | wg256, `392.8771 us` | wg256, `28.0126 us` | `14.03x` |
| 8192 | 4 | wg256, `396.4413 us` | wg256, `31.5848 us` | `12.55x` |
| 8192 | 8 | wg256, `408.1349 us` | wg256, `33.9988 us` | `12.00x` |

Conclusion: in this matched f32 GEMV/reduction harness, HIP and Vulkan both
prefer the 256-thread workgroup. Moving HIP from 64 to 256 threads improves
substantially, but it does **not** close the Vulkan gap. Vulkan remains
`5.79x-14.03x` faster on best-native rows, and the largest identical-shape
speedups are `12.88x-15.34x` at smaller workgroups.

This rules out the simple "HIP used the wrong workgroup size" explanation for
this microbench. It still does **not** prove `compiler_aco`; the ISA/stat
extraction below rules out some simple compiler stories but does not yet
identify a primary cause.

Immediate reads:

- Workgroup shape matters for HIP, but all retained best rows are wg256 on both
  backends.
- The matched-shape Vulkan gap is large enough that wave/subgroup mode, fixed
  workgroup specialization, memory scheduling, or a hidden harness/runtime
  difference must be tested directly.
- This result makes VOPD-specific, memory/waitcnt, dot-path, and HIP
  fixed-workgroup-specialization controls higher priority than more geometry
  sweeps on gfx1151.

### gfx1151 F32 Geometry ISA/Stat Extraction

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/geometry-isa-stats-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-geometry-isa-stats.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-geometry-isa-stats.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-isa-stats.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: ISA/stat extraction for the retained f32 geometry kernel,
  K=`2048`, rows=`1`, workgroup=`64/256`, body repeats=`128`.
- HIP evidence: `hipcc --save-temps`, `llvm-readobj --notes`, and
  `llvm-objdump -d --no-show-raw-insn`.
- Vulkan evidence: `RADV_DEBUG=shaders,shaderstats` final disassembly, ACO
  after-RA presence, and RADV shaderstats allocation counts. Estimated physical
  register spans are retained as a cross-check, but the primary RADV VGPR/SGPR
  rows below are official shaderstats allocation counts.
- Classification: `diagnostic_unclassified`.

Retained ISA/stat summary:

| Workgroup | HIP ISA | Vulkan/RADV ISA |
| ---: | --- | --- |
| 64 | actual `18` SGPR, `11` VGPR, scratch `0`, spills `0`, wave32, `118` static instructions, `6` waitcnt-family instructions, `2` VOPD instructions / `4` VOPD ops | official `108` SGPR, `12` VGPR, scratch `0`, spills `0`, estimated span `16` SGPR / `9` VGPR, wave64, `100` static instructions, `9` waitcnt-family instructions, `0` VOPD |
| 256 | actual `18` SGPR, `11` VGPR, scratch `0`, spills `0`, wave32, `118` static instructions, `6` waitcnt-family instructions, `2` VOPD instructions / `4` VOPD ops | official `108` SGPR, `12` VGPR, scratch `0`, spills `0`, estimated span `16` SGPR / `9` VGPR, wave64, `142` static instructions, `20` waitcnt-family instructions including `9` depctr waits, `0` VOPD |

Matched timing context from the geometry sweep:

| Shape | HIP median | Vulkan median | Vulkan vs HIP |
| --- | ---: | ---: | ---: |
| K=2048 rows=1 wg64 | `292.1010 us` | `21.6447 us` | `13.50x` |
| K=2048 rows=1 wg256 | `85.6870 us` | `7.3849 us` | `11.60x` |

Conclusion: the retained f32 geometry gap is **not** explained by LLVM missing
VOPD pairing. In this kernel, HIP emits VOPD and RADV does not. It is also not
explained by HIP spills or scratch use; HIP reports no spills and no scratch.
The remaining plausible causes are narrower: Vulkan specialization constants
versus HIP runtime `blockDim` code, wave64/subgroup reduction behavior,
memory/address scheduling, pipeline/runtime effects not captured by static ISA,
or a source/harness detail still not controlled.

Immediate reads:

- Do not file an LLVM VOPD issue from this geometry kernel.
- Do not claim RADV has better register allocation from this artifact. Official
  RADV shaderstats now show no Vulkan scratch/spills, but they do not explain
  the geometry timing gap or prove better allocation than HIP.
- Add fixed-workgroup HIP variants to remove dynamic `blockDim` overhead before
  using this f32 geometry row as a compiler-codegen proxy.
- The next compiler-facing tests should be memory/waitcnt microbenches where
  load scheduling is isolated by design.

### gfx1151 VOPD/VALU Scheduling Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/vopd-sweep-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-vopd-sweep.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-vopd-sweep.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-vopd-sweep.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: pure VALU VOPD diagnostics with sampled CPU oracle,
  N=`65536`, body iters=`2048`, block size=`256`.
- Variants: independent f32 FMA with `2/4/8` accumulators, dependent f32 FMA
  with `4` chained operations, mixed int+float with `4` accumulators, and
  dequant-like shift/mask/cvt/FMA with `4` accumulators.
- Classification: `diagnostic_unclassified`, with a negative result for the
  "ACO wins through VOPD" hypothesis.

Retained timing and VOPD summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP VOPD | RADV VOPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| independent f32 FMA, 2 accum | `250.7432 us` | `285.6322 us` | `0.88x` | `4` | `0` |
| independent f32 FMA, 4 accum | `447.9288 us` | `462.4270 us` | `0.97x` | `9` | `0` |
| independent f32 FMA, 8 accum | `877.3634 us` | `833.0101 us` | `1.05x` | `12` | `0` |
| dependent f32 FMA, 4 ops | `426.6930 us` | `453.9037 us` | `0.94x` | `1` | `0` |
| mixed int+float, 4 accum | `1020.2676 us` | `946.9001 us` | `1.08x` | `6` | `0` |
| dequant-like, 4 accum | `1019.7762 us` | `978.8506 us` | `1.04x` | `5` | `0` |

Register/stat summary:

- HIP reports no scratch in every retained VOPD row. VGPRs rise with
  independent accumulators: `8/11/21` VGPR for `2/4/8` independent accumulators.
- RADV shaderstats reports official `12/12/24` VGPR for independent `2/4/8`,
  official `12/24/24` VGPR for dependent/mixed/dequant rows, `108` SGPR in all
  rows, and `0` scratch/spills in all rows.
- RADV emits wave64 final shaders in these rows; HIP emits wave32 code objects.
- Both backends pass the sampled CPU oracle with max abs `2.384185791e-07`.

Conclusion: do **not** attribute the current Vulkan ceiling to RADV/ACO finding
better VOPD pairing than LLVM/HIP. In this targeted family, LLVM/HIP is the
backend emitting VOPD. Vulkan's modest wins on independent-8, mixed int+float,
and dequant-like rows must come from something else: wave64 execution shape,
non-VOPD scheduling, instruction selection, runtime/pipeline effects, or
measurement noise. The retained Q6_K X8 production transfer check is already
negative for a broad memory/waitcnt claim, so the next relevant compiler work
is cross-GPU portability or Q4_K-specific recovery, not more generic VOPD
speculation.

### gfx1151 Memory / Waitcnt Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/memory-waitcnt-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-memory-waitcnt.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-memory-waitcnt.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-memory-waitcnt.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: device-memory load+accumulate diagnostics with sampled CPU
  oracle, N=`32768`, body iters=`128`, block size=`256`.
- Variants: coalesced vector-width loads `1/2/4/8`, strided loads
  `2/4/8/16`, gather-ID loads, and load/compute interleave
  `1/2/4/8/16`.
- Classification: `diagnostic_unclassified`, with strong memory/waitcnt
  evidence but not a pure `compiler_aco` proof because wave32 vs wave64 remains
  a confound and the first memory-heavy production transfer check is negative.

Retained timing and ISA summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP GB/s | Vulkan GB/s | HIP wait/load | RADV wait/load |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| coalesced width 1 | `26.6127 us` | `18.9185 us` | `1.41x` | `630.42` | `887.02` | `4/1` | `3/1` |
| coalesced width 2 | `44.6948 us` | `32.0182 us` | `1.40x` | `750.75` | `1048.01` | `5/2` | `4/2` |
| coalesced width 4 | `282.1829 us` | `125.2440 us` | `2.25x` | `237.82` | `535.82` | `7/4` | `6/4` |
| coalesced width 8 | `639.5150 us` | `303.7756 us` | `2.11x` | `209.87` | `441.94` | `11/8` | `10/8` |
| strided stride 2 | `44.5751 us` | `34.3159 us` | `1.30x` | `376.38` | `488.92` | `4/1` | `3/1` |
| strided stride 4 | `281.3058 us` | `146.2196 us` | `1.92x` | `59.64` | `114.75` | `4/1` | `3/1` |
| strided stride 8 | `557.0604 us` | `249.6609 us` | `2.23x` | `30.12` | `67.19` | `4/1` | `3/1` |
| strided stride 16 | `1144.5523 us` | `576.7278 us` | `1.98x` | `14.66` | `29.09` | `4/1` | `3/1` |
| gather IDs | `493.0500 us` | `484.7561 us` | `1.02x` | `68.05` | `69.22` | `5/2` | `4/2` |
| interleave unroll 1 | `25.8722 us` | `15.8026 us` | `1.64x` | `648.47` | `1061.67` | `4/1` | `3/1` |
| interleave unroll 2 | `49.8455 us` | `40.9710 us` | `1.22x` | `673.17` | `818.98` | `5/2` | `4/2` |
| interleave unroll 4 | `281.1671 us` | `135.2524 us` | `2.08x` | `238.68` | `496.19` | `6/4` | `6/4` |
| interleave unroll 8 | `580.6280 us` | `289.8444 us` | `2.00x` | `231.16` | `463.06` | `10/8` | `10/8` |
| interleave unroll 16 | `1611.1334 us` | `1452.4936 us` | `1.11x` | `166.61` | `184.80` | `13/16` | `18/16` |

Register/stat summary:

- All HIP and Vulkan rows pass the sampled CPU oracle with max abs `0.0`.
- HIP reports wave32, no scratch, and no spills in all retained rows. HIP VGPR
  rises with interleave width from `8` at unroll 1 to `36` at unroll 16.
- RADV final shaders are wave64. RADV shaderstats reports official VGPR
  allocation of `12` for simple coalesced/strided/gather rows, `12/24/24/48/48`
  for interleave `1/2/4/8/16`, `108` SGPR in all rows, and `0`
  scratch/spills in all rows. Estimated physical register spans remain in the
  artifacts as cross-checks.
- Simple coalesced, strided, gather, and low-unroll interleave rows show one
  fewer RADV waitcnt-family instruction than HIP at the same static load count.
  Wider interleave rows have equal or higher RADV waitcnt counts, so waitcnt
  count alone is not the whole story.

Conclusion: memory/access scheduling is a serious diagnostic lead, but not yet
a clean LLVM `compiler_aco` issue. Vulkan is consistently faster on coalesced,
strided, and most interleave rows, while gather is essentially tied. However,
the retained rows still compare HIP wave32 against RADV wave64, and the first
memory-heavy production-shaped transfer check, Q6_K X8 selected-down, goes the
other way: Vulkan is slower and RADV has more static instructions and
waitcnt-family instructions in the targeted ISA join. The fixed-block control
below does not close the synthetic memory gap; the dot-path result below
separately shows basic q8/q4/q6 dot lowering is present in HIP.

### gfx1151 Packed Dot Path

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/dot-path-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-dot-path.json` and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-dot-path.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-dot-path.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: packed int8 dot diagnostics with exact sampled CPU oracle,
  N=`32768`, body iters=`128`, groups=`16`, block size=`256`.
- Variants: q8 signed dot, q4 unsigned-byte by signed-q8 dot, q6 zero-point
  correction (`dot_u - 32 * q8_sum`), and scalar q4 dequant.
- Classification: `diagnostic_unclassified`; useful dot-lowering evidence, but
  not a clean `compiler_aco` proof because HIP wave32 vs RADV wave64 and
  specialization/lowering differences remain.

Retained timing and ISA summary:

| Variant | HIP median | Vulkan median | Vulkan vs HIP | HIP dot4 | RADV dot4 | SPIR-V dot op | HIP wait/load | RADV wait/load |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| q8 signed | `7114.77 us` | `2079.47 us` | `3.42x` | `16` | `16` | `OpSDot=1` | `19/32` | `18/32` |
| q4 unsigned x q8 | `7109.06 us` | `2088.64 us` | `3.40x` | `16` | `16` | `OpSUDot=1` | `19/32` | `18/32` |
| q6 zero-corrected | `6831.89 us` | `2082.13 us` | `3.28x` | `32` | `32` | `OpSUDot=2` | `20/32` | `18/32` |
| scalar q4 dequant | `7342.70 us` | `2223.05 us` | `3.30x` | `0` | `0` | none | `35/32` | `20/32` |

Register/stat summary:

- All HIP and Vulkan rows pass the exact sampled CPU oracle with max abs `0.0`.
- HIP reports wave32, no scratch, and no spills. HIP dot rows use
  `41-42` VGPR and `14` SGPR; scalar dequant uses `50` VGPR.
- RADV final shaders are wave64. RADV shaderstats reports official `36` VGPR
  for q8/q4, `48` VGPR for q6/scalar, `108` SGPR in all rows, and `0`
  scratch/spills in all rows. Estimated physical register spans remain in the
  artifact as a cross-check.
- HIP and RADV emit the same final dot4 counts in q8/q4/q6 rows. Vulkan SPIR-V
  also contains the expected `OpSDot`/`OpSUDot` operations before RADV lowering.
- The scalar row is also `3.30x` faster on Vulkan despite using no dot4, so the
  retained gap cannot be explained only by dot-instruction selection.

Conclusion: do **not** spend more gfx1151 time proving whether HIP can emit the
basic q8/q4/q6 dot instruction; it can. The retained dot-path gap is still
large, but the fixed-block control below shows runtime block indexing is not
the missing switch. HIP q8_1 real-slice layout controls are positive, and the
matched Vulkan real-slice controls are split: Q6_K X8 selected-down is
negative, while Q4_K selected-dual gate/up is positive. A hand-ISA path is not
justified by this artifact alone; it would need to beat the same HIP dot body
after wave/fixed-shape controls and then move a shipped selected-MoE or q6
lm-head slice.

### gfx1151 HIP Wave64 Controls

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/dot-path-wave64-comparison.json`
and
`benchmarks/micro/results/gfx1151/strix-halo/memory-waitcnt-wave64-comparison.json`.
Backend artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-dot-path-wave64.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-dot-path-wave64-control.json`,
`benchmarks/micro/results/gfx1151/strix-halo/hip-memory-waitcnt-wave64.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-memory-waitcnt-wave64-control.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-wave64-controls.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- HIP flag: `-mwavefrontsize64` via `--hip-wavefront-size 64`.
- Classification: `diagnostic_unclassified`; this is a wave-mode control, not a
  same-geometry compiler proof.

Dot-path wave64 result:

| Variant | HIP wave32 median | HIP wave64 median | Wave64 / wave32 | Same-commit Vulkan vs HIP wave64 |
| --- | ---: | ---: | ---: | ---: |
| q8 signed | `7114.77 us` | `7548.48 us` | `1.061x` slower | `3.13x` |
| q4 unsigned x q8 | `7109.06 us` | `7350.55 us` | `1.034x` slower | `3.55x` |
| q6 zero-corrected | `6831.89 us` | `7106.02 us` | `1.040x` slower | `2.63x` |
| scalar q4 dequant | `7342.70 us` | `7397.10 us` | `1.007x` slower | `3.16x` |

Memory/waitcnt wave64 result:

| Group | HIP wave64 vs wave32 | Same-commit Vulkan vs HIP wave64 |
| --- | --- | --- |
| coalesced | width 1/2/4 slower by `1.017x-1.098x`; width 8 faster by `0.921x` | Vulkan still `1.70x-2.08x` faster |
| strided | mixed `0.977x-1.022x` vs wave32 | Vulkan still `1.68x-2.29x` faster |
| gather | `6.349x` slower than wave32 | Vulkan `6.59x` faster than HIP wave64 |
| interleave | mixed; unroll 16 faster by `0.922x`, unroll 1 slower by `1.146x` | Vulkan faster except unroll 16, where HIP wave64 is `1.15x` faster |

Conclusion: HIP wave64 is not the missing switch. It does not close the packed
dot gap, and the memory result is mixed with a severe gather regression. The
later fixed-shape controls also fail to close these gaps; do not promote broad
HIP wave64 routing from this evidence.

### gfx1151 HIP Fixed-Shape Controls

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/dot-path-fixed-block-comparison.json`,
`benchmarks/micro/results/gfx1151/strix-halo/memory-waitcnt-fixed-block-comparison.json`,
`benchmarks/micro/results/gfx1151/strix-halo/geometry-sweep-fixed-workgroup-comparison.json`,
`benchmarks/micro/results/gfx1151/strix-halo/geometry-sweep-fixed-workgroup-wave64-comparison.json`,
`benchmarks/micro/results/gfx1151/strix-halo/geometry-sweep-fixed-wave64-delta.json`,
`benchmarks/micro/results/gfx1151/strix-halo/hip-geometry-isa-stats-fixed-wave64.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/geometry-isa-stats-fixed-wave64-comparison.json`.
Backend artifacts include same-commit runtime HIP controls, fixed HIP controls,
and same-commit Vulkan controls:
`hip-dot-path-runtime-control.json`, `hip-dot-path-fixed-block.json`,
`vulkan-dot-path-fixed-control.json`,
`hip-memory-waitcnt-runtime-control.json`,
`hip-memory-waitcnt-fixed-block.json`,
`vulkan-memory-waitcnt-fixed-control.json`,
`hip-geometry-sweep-runtime-control.json`,
`hip-geometry-sweep-fixed-workgroup.json`,
`hip-geometry-sweep-fixed-workgroup-wave64.json`, and
`vulkan-geometry-sweep-fixed-control.json`. The run uses
`environment-fixed-shape-controls.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- HIP dot/memory control: `--hip-fixed-block-index`, which compiles
  `__launch_bounds__(256)` and `kBlockSize` global indexing instead of
  `blockDim.x`.
- HIP geometry control: `--hip-workgroup-specialization fixed`, which compiles
  one HIP binary per requested workgroup size and replaces runtime
  `blockDim.x` in the reduction/indexing path.
- HIP combined geometry control: `--hip-workgroup-specialization fixed
  --hip-wavefront-size 64`, which compiles one fixed-workgroup wave64 code
  object per requested workgroup.
- Classification: `diagnostic_unclassified`; this is a runtime-shape control,
  not final compiler attribution.

Dot fixed-block result:

| Variant | Fixed / runtime HIP | Same-commit Vulkan vs fixed HIP |
| --- | ---: | ---: |
| q8 signed | `1.000x` | `3.43x` |
| q4 unsigned x q8 | `1.000x` | `3.42x` |
| q6 zero-corrected | `0.997x` | `3.31x` |
| scalar q4 dequant | `0.993x` | `3.31x` |

Memory fixed-block result:

| Group | Fixed / runtime HIP | Same-commit Vulkan vs fixed HIP |
| --- | --- | --- |
| coalesced | width 1 slower `1.063x`; widths 2/4 flat; width 8 faster `0.906x` | Vulkan `1.47x-2.28x` faster |
| strided | `1.004x-1.026x` slower | Vulkan `1.35x-2.36x` faster |
| gather | `1.290x` slower | Vulkan `1.36x` faster |
| interleave | mixed: unroll 1/16 faster `0.913x/0.930x`, unroll 2 slower `1.120x`, others flat | Vulkan `1.04x-2.29x` faster |

Geometry fixed-workgroup result:

| Shape group | Fixed / runtime HIP at best HIP wg256 | Same-commit Vulkan vs fixed HIP best-native |
| --- | ---: | ---: |
| K=512 rows=1/4/8 | `0.985x-0.987x` | `5.56x-7.60x` |
| K=2048 rows=1/4/8 | `0.937x-0.999x` | `9.79x-11.56x` |
| K=8192 rows=1/4/8 | `0.976x-0.991x` | `12.07x-14.03x` |

Geometry fixed-workgroup plus wave64 result:

| Shape group | HIP fixed wave64 / fixed wave32 at best HIP wg256 | Same-commit Vulkan vs HIP fixed wave64 best-native |
| --- | ---: | ---: |
| K=512 rows=1/4/8 | `1.131x-1.135x` slower | `6.31x-8.61x` |
| K=2048 rows=1/4/8 | `1.153x-1.234x` slower | `11.29x-13.88x` |
| K=8192 rows=1/4/8 | `1.148x-1.167x` slower | `14.09x-16.18x` |

Fixed-wave64 geometry ISA/stat result for K=2048 rows=1:

| Workgroup | HIP fixed-wave64 ISA | Vulkan/RADV ISA |
| ---: | --- | --- |
| 64 | actual `20` SGPR, `11` VGPR, scratch `0`, spills `0`, wave64, `0` VOPD, `20` waitcnt-family instructions | official `108` SGPR, `12` VGPR, scratch `0`, spills `0`, wave64, `0` VOPD, `9` waitcnt-family instructions |
| 256 | actual `20` SGPR, `11` VGPR, scratch `0`, spills `0`, wave64, `0` VOPD, `24` waitcnt-family instructions | official `108` SGPR, `12` VGPR, scratch `0`, spills `0`, wave64, `0` VOPD, `20` waitcnt-family instructions |

Conclusion: HIP runtime `blockDim`/shape specialization is not the missing
switch for the retained gfx1151 gaps. Fixed geometry gives a useful small HIP
improvement, especially K=2048 rows=1 wg256, but the Vulkan geometry lead
remains large. Dot fixed-block is flat, and memory fixed-block is mixed with a
gather regression. Combining fixed-workgroup geometry with wave64 makes HIP
slower than fixed wave32 and still leaves Vulkan much faster, so wave mode is
not the missing geometry switch either. Treat memory/access scheduling and
geometry as still unclassified until source/runtime structure and production
real-slice transfer are resolved.

### gfx1151 LDS / Barrier / Subgroup / Accumulator Reduction Sweep

Retained artifact:
`benchmarks/micro/results/gfx1151/strix-halo/reduction-sweep.json`.
Accumulator extension artifact:
`benchmarks/micro/results/gfx1151/strix-halo/reduction-accum-sweep.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-fixed-shape-controls.json`.
The accumulator extension uses
`benchmarks/micro/results/gfx1151/strix-halo/environment-reduction-accum.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: one-row repeat-shifted f32 reduction, K=`512/2048/8192`,
  rows=`1`, workgroup=`64/256` in the original reduction sweep and
  workgroup=`32/64/256` in the accumulator extension, body repeats=`128`, CPU
  oracle.
- HIP variants: LDS tree, LDS tree with an extra barrier per reduction stage,
  wave-shuffle reduction with one LDS value per wave for cross-wave merge, and
  4/8/16 lane-local accumulator variants before the workgroup reduction.
- Vulkan variants: LDS tree, LDS tree with an extra barrier per reduction
  stage, `GL_KHR_shader_subgroup_arithmetic` subgroup reduction, and 4/8/16
  lane-local accumulator variants before the workgroup reduction.
- Classification: `diagnostic_unclassified`; this is a reduction-topology
  control for the f32 geometry gap, not a final compiler attribution.

Matched LDS timing:

| K | Workgroup | HIP LDS | Vulkan LDS | Vulkan vs HIP |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 64 | `80.0352 us` | `6.8729 us` | `11.64x` |
| 512 | 256 | `23.5467 us` | `2.8734 us` | `8.19x` |
| 2048 | 64 | `291.7382 us` | `21.5059 us` | `13.57x` |
| 2048 | 256 | `86.0937 us` | `7.1910 us` | `11.97x` |
| 8192 | 64 | `1463.4992 us` | `100.5804 us` | `14.55x` |
| 8192 | 256 | `391.1092 us` | `27.6244 us` | `14.16x` |

Variant deltas:

| Control | Retained range |
| --- | ---: |
| HIP extra barrier / HIP LDS | `1.002x-1.028x` |
| HIP wave-shuffle / HIP LDS | `0.991x-1.005x` |
| Vulkan extra barrier / Vulkan LDS | `0.991x-1.005x` |
| Vulkan subgroup / Vulkan LDS | `0.984x-1.132x` |
| Vulkan subgroup / HIP wave-shuffle | `0.072x-0.121x` wall-time ratio, i.e. Vulkan is `8.24x-13.86x` faster |

Accumulator extension matched backend timing:

| Variant | Shapes | Vulkan vs HIP |
| --- | --- | ---: |
| LDS tree | K=`512/2048/8192`, wg=`32/64/256` | `9.14x-14.54x` faster |
| multi_accum4 | K=`512/2048/8192`, wg=`32/64/256` | `10.03x-15.74x` faster |
| multi_accum8 | K=`512/2048/8192`, wg=`32/64/256` | `9.58x-15.23x` faster |
| multi_accum16 | K=`512/2048/8192`, wg=`32/64/256` | `12.41x-15.81x` faster |

Accumulator extension variant deltas:

| Control | Retained range |
| --- | ---: |
| HIP multi_accum4 / HIP LDS | `1.037x-1.236x` |
| HIP multi_accum8 / HIP LDS | `1.023x-1.531x` |
| HIP multi_accum16 / HIP LDS | `1.023x-2.405x` |
| Vulkan multi_accum4 / Vulkan LDS | `0.927x-1.126x` |
| Vulkan multi_accum8 / Vulkan LDS | `0.939x-1.462x` |
| Vulkan multi_accum16 / Vulkan LDS | `0.904x-1.772x` |

Conclusion: reduction topology is not the missing f32 geometry switch. HIP
wave-shuffle reduction is essentially flat versus HIP's LDS tree, and Vulkan
subgroup reduction is mostly flat to modestly slower than Vulkan's LDS tree on
the retained shapes. Adding one extra barrier per reduction stage is also flat
to small relative to the backend gap. The accumulator extension closes the
proposed no-LDS-accumulator control as a negative HIP recovery result:
4/8/16 lane-local accumulators mostly slow HIP and do not reduce the matched
Vulkan lead. The one-wave HIP wave-shuffle and one-subgroup Vulkan subgroup
paths now bypass shared memory for the final write, but multi-wave workgroups
still need shared memory for the cross-wave final reduction. A true two-stage
block-partial plus final-reduce launch sequence remains unrun because no
retained profile currently makes it decision-grade. The f32 geometry row
remains `diagnostic_unclassified`; remaining explanations are more likely
source/runtime structure, address/memory scheduling, or another compiler/runtime
effect outside the isolated reduction topology.

### gfx1151 Sampler Top-1 And Top-K8 Argmax Sweep

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-sampler-argmax.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-sampler-argmax.json`,
`benchmarks/micro/results/gfx1151/strix-halo/hip-sampler-topk8.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-sampler-topk8.json`,
`benchmarks/micro/results/gfx1151/strix-halo/sampler-topk8-comparison.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/sampler-argmax-comparison.json`.
The runs use the shared environment artifacts
`benchmarks/micro/results/gfx1151/strix-halo/environment-sampler-argmax.json`
and
`benchmarks/micro/results/gfx1151/strix-halo/environment-sampler-topk.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Vulkan driver: RADV, Mesa `26.1.2-arch2.1`.
- Benchmark family: deterministic sampler top-1 and top-k8 argmax over
  synthetic logits, one workgroup per row, vocab=`32768`, rows=`1/4/8`,
  workgroup=`64/128/256`, CPU oracle with stable value/index ordering.
- Classification: `diagnostic_unclassified`; this covers deterministic top-1
  and top-k8 argmax, not stochastic sampling, probability filtering, or fused
  lm-head+sample.

Matched timing:

| Rows | Workgroup | HIP median | Vulkan median | Vulkan vs HIP |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | `60.4667 us` | `2.3422 us` | `25.82x` |
| 1 | 128 | `31.7681 us` | `1.3852 us` | `22.93x` |
| 1 | 256 | `17.3869 us` | `1.0406 us` | `16.71x` |
| 4 | 64 | `61.7742 us` | `2.3588 us` | `26.19x` |
| 4 | 128 | `34.6676 us` | `1.3932 us` | `24.88x` |
| 4 | 256 | `17.6572 us` | `1.0480 us` | `16.85x` |
| 8 | 64 | `64.1006 us` | `2.3793 us` | `26.94x` |
| 8 | 128 | `33.0409 us` | `1.5401 us` | `21.45x` |
| 8 | 256 | `17.9019 us` | `1.4042 us` | `12.75x` |

Best-native read:

| Rows | HIP best | Vulkan best | Vulkan vs HIP |
| ---: | ---: | ---: | ---: |
| 1 | wg256, `17.3869 us` | wg256, `1.0406 us` | `16.71x` |
| 4 | wg256, `17.6572 us` | wg256, `1.0480 us` | `16.85x` |
| 8 | wg256, `17.9019 us` | wg256, `1.4042 us` | `12.75x` |

Top-k8 matched timing:

| Rows | Workgroup | HIP median | Vulkan median | Vulkan vs HIP |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 64 | `477.1200 us` | `20.8353 us` | `22.90x` |
| 1 | 128 | `248.4100 us` | `10.6300 us` | `23.37x` |
| 1 | 256 | `132.7010 us` | `5.7205 us` | `23.20x` |
| 4 | 64 | `488.4448 us` | `20.8315 us` | `23.45x` |
| 4 | 128 | `254.1818 us` | `10.9534 us` | `23.21x` |
| 4 | 256 | `135.9226 us` | `5.9910 us` | `22.69x` |
| 8 | 64 | `593.5294 us` | `22.8864 us` | `25.93x` |
| 8 | 128 | `305.7790 us` | `12.7718 us` | `23.94x` |
| 8 | 256 | `162.0304 us` | `12.6654 us` | `12.79x` |

Top-k8 best-native read:

| Rows | HIP best | Vulkan best | Vulkan vs HIP |
| ---: | ---: | ---: | ---: |
| 1 | wg256, `132.7010 us` | wg256, `5.7205 us` | `23.20x` |
| 4 | wg256, `135.9226 us` | wg256, `5.9910 us` | `22.69x` |
| 8 | wg256, `162.0304 us` | wg256, `12.6654 us` | `12.79x` |

ISA/stat summary:

| Workgroup | HIP ISA | Vulkan/RADV ISA |
| ---: | --- | --- |
| 64 | wave32, `15` SGPR / `7` VGPR, scratch `0`, spills `0`, `185` static instructions, `23` waitcnt-family instructions, `3` VOPD | wave64, official `108` SGPR / `12` VGPR, scratch `0`, spills `0`, `141` static instructions, `14` waitcnt-family instructions, `0` VOPD |
| 128 | wave32, `15` SGPR / `7` VGPR, scratch `0`, spills `0`, `205` static instructions, `26` waitcnt-family instructions, `3` VOPD | wave64, official `108` SGPR / `12` VGPR, scratch `0`, spills `0`, `183` static instructions, `24` waitcnt-family instructions, `0` VOPD |
| 256 | wave32, `15` SGPR / `7` VGPR, scratch `0`, spills `0`, `230` static instructions, `29` waitcnt-family instructions, `3` VOPD | wave64, official `108` SGPR / `12` VGPR, scratch `0`, spills `0`, `206` static instructions, `27` waitcnt-family instructions, `0` VOPD |

Top-k8 ISA/stat summary:

| Workgroup | HIP ISA | Vulkan/RADV ISA |
| ---: | --- | --- |
| 64 | wave32, `32` SGPR / `18` VGPR, scratch `0`, spills `0`, `256` static instructions, `26` waitcnt-family instructions, `5` VOPD | wave64, official `108` SGPR / `12` VGPR, scratch `0`, spills `0`, `258` static instructions, `24` waitcnt-family instructions, `0` VOPD |
| 128 | wave32, `33` SGPR / `18` VGPR, scratch `0`, spills `0`, `277` static instructions, `29` waitcnt-family instructions, `6` VOPD | wave64, official `108` SGPR / `12` VGPR, scratch `0`, spills `0`, `294` static instructions, `36` waitcnt-family instructions, `0` VOPD |
| 256 | wave32, `34` SGPR / `20` VGPR, scratch `0`, spills `0`, `297` static instructions, `32` waitcnt-family instructions, `7` VOPD | wave64, official `108` SGPR / `12` VGPR, scratch `0`, spills `0`, `320` static instructions, `39` waitcnt-family instructions, `0` VOPD |

Conclusion: deterministic top-1 and top-k8 argmax are real Vulkan diagnostic
wins on this gfx1151 system. The win is not explained by missing HIP workgroup
tuning, missing HIP VOPD, or HIP spills: both backends prefer wg256 for
best-native rows, HIP emits VOPD while RADV emits none, and neither backend
reports scratch/spills. The row is useful because sampler work is an exposed
server bucket after lm-head in some profiles. It should drive a HIP-side
reduction rewrite or fused lm-head+sample experiment only if current profiling
still shows sampler work as exposed. It does not cover stochastic sampling,
probability filtering, or production registry integration.

### gfx1151 HIP q8_1 Real-Slice Layout Controls

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-real-q4-selected-dual-q8_1-dp4a.json`
and
`benchmarks/micro/results/gfx1151/strix-halo/hip-real-q6-selected-down-x8-q8_1-dp4a.json`.
The run uses the shared environment artifact
`benchmarks/micro/results/gfx1151/strix-halo/environment-real-slice-q8_1.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics`, `gfx1151`.
- Backend: HIP only. These controls do **not** include a Vulkan shader for the
  same production slice.
- Slice 1: Q4_K selected-dual gate/up, `x_rows=4`, selected rows=`32`,
  experts=`256`, in=`2048`, out=`512`, threads=`256`.
- Slice 2: Q6_K selected-down, selected rows=`8`, experts=`256`, in=`512`,
  out=`2048`, production T16 float versus X8 q8_1 dp4a.
- Classification: `layout_quant` for HIP-side economics; not a Vulkan compiler
  attribution.

Q4_K selected-dual gate/up result:

| Metric | Result |
| --- | ---: |
| Raw selected-dual | `0.9584 ms` |
| q8_1 quantize | `0.00247 ms` |
| q8_1 dp4a dot, prequantized | `0.3464 ms` |
| q8_1 quantize+dp4a dot | `0.3458 ms` |
| Raw / quantize+dot | `2.77x` |
| Correctness vs raw | KL mean `0.00311`, max abs `2.0`, top-1 `1.0` for both gate and up |

Q6_K selected-down result:

| Metric | Result |
| --- | ---: |
| Production T16 float | `0.03228 ms` |
| q8_1 quantize | `0.00275 ms` |
| X8 q8_1 dp4a dot, prequantized | `0.01665 ms` |
| X8 q8_1 quantize+dp4a dot | `0.01925 ms` |
| Production T16 / quantize+dot | `1.68x` |
| Raw float / quantize+dot | `2.38x` |
| Correctness vs production T16 | KL mean `0.00137`, max abs `0.5`, top-1 `1.0` |
| X8 vs raw dp4a | max abs `1.91e-6`, top-1 `1.0` |

Conclusion: q8_1 materialization cost is not the reason to avoid q8_1/dp4a on
these HIP selected-MoE slices. The quantization stage is only about
`0.0025-0.0027 ms` and the full quantize+dot path is faster than the tested HIP
float controls. The matched Vulkan Q6_K X8 selected-down probe below shows that
this positive HIP layout result does **not** automatically transfer to a Vulkan
backend.

### gfx1151 Vulkan Q6_K X8 Real-Slice Probe

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-ls128.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-ls256.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q6-selected-down-x8-q8_1-dp4a-isa-stats.json`,
`benchmarks/micro/results/gfx1151/strix-halo/q6-x8-real-slice-hip-vulkan-comparison.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/q6-x8-real-slice-isa-comparison.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Shape: Q6_K selected-down X8, rows=`8`, experts=`256`, in=`512`,
  out=`2048`, q8_1 activation format, bf16 output.
- Vulkan implementation: standalone microbench probe with a q8_1 quantize
  shader plus Q6_K X8 selected-down dp4a shader. Timings use pre-recorded
  command buffers and exclude shader compilation, pipeline creation, and
  host-device transfers.
- Correctness: full CPU reference for q8_1 quantize plus Q6_K X8 selected-down
  dp4a. All retained local-size rows pass with top-1 `1.0`; best local_size=64
  max abs is `0.25`, mean abs `0.00308`.
- Classification: `real_slice_probe`; this is production-shaped backend
  evidence, not a generic compiler proof.

Timing result:

| Backend / local size | q8_1 quantize | X8 dot, prequantized | q8_1 quantize+X8 dot | Correctness |
| --- | ---: | ---: | ---: | --- |
| HIP retained, threads=64 | `0.00275 ms` | `0.01665 ms` | `0.01925 ms` | top-1 `1.0` vs production T16 |
| Vulkan local_size=64 | `0.000357 ms` | `0.03076 ms` | `0.03217 ms` | top-1 `1.0` vs CPU |
| Vulkan local_size=128 | `0.000356 ms` | `0.03258 ms` | `0.03326 ms` | top-1 `1.0` vs CPU |
| Vulkan local_size=256 | `0.000356 ms` | `0.03500 ms` | `0.03707 ms` | top-1 `1.0` vs CPU |

Best retained Vulkan row is local_size=64. Relative to retained HIP, Vulkan is:

| Metric | Vulkan / HIP |
| --- | ---: |
| X8 dot, prequantized | `1.85x` slower |
| q8_1 quantize+X8 dot | `1.67x` slower |

ISA/stat extraction for local_size=64:

| Shader | SPIR-V dot ops | RADV final dot4 | RADV subgroup | RADV VOPD | RADV registers | Wait/load notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| q8_1 quantize | `0` | `0` | `64` | `0` | official `96` VGPR / `108` SGPR, no scratch/spills; span `96/16` | `6` waitcnt, `1` buffer load |
| Q6_K X8 dot | `9` `OpSUDot` | `9` | `64` | `0` | official `48` VGPR / `108` SGPR, no scratch/spills; span `48/24` | `89` waitcnt, `82` buffer loads |

Targeted HIP/RADV ISA comparison:

| Shader | HIP/LLVM ISA | Vulkan/RADV ISA | Read |
| --- | --- | --- | --- |
| Q6_K X8 dot | wave32, `30` SGPR / `51` VGPR, scratch `0`, spills `0`, `599` static instructions, `39` waitcnt-family instructions, `51` VOPD, `9` dot4, `34` global loads, `10` LDS loads, `4` LDS stores | wave64, official `108` SGPR / `48` VGPR, scratch `0`, spills `0`, `1117` static instructions, `89` waitcnt-family instructions, `0` VOPD, `9` dot4, `82` buffer loads, `4` LDS loads, `2` LDS stores | The negative Q6 result is not missing RADV dot4 or Vulkan spills. RADV has more static instructions and waitcnt-family instructions while running slower, so this production-shaped memory-heavy slice does not support a generic LLVM waitcnt/scheduling issue. |

Conclusion: the first matched Vulkan production slice is negative. The
synthetic packed dot-path sweep showed Vulkan can be much faster on a simplified
dot loop, but this Q6_K selected-down X8 production-shaped shader does not beat
HIP once the real layout, q8_1 materialization, selected rows, output shape, and
subgroup reduction are present. Do not promote a Vulkan backend or hand-ISA
path from this q6 selected-down evidence. Also do not promote the synthetic
memory/waitcnt sweep to an LLVM-AMDGPU filing from this row; the first
memory-heavy production transfer check goes the other way. The remaining
compiler-facing work is now Q4_K-specific follow-up because the next section
shows a different real-slice answer. The HIP fixed-wave64 geometry control
below removes the broad f32 geometry wave-mode confound and is negative: it
makes HIP slower, while Vulkan remains much faster.

### gfx1151 Vulkan Q4_K Selected-Dual Real-Slice Probe

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-ls128.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-ls256.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-isa-stats.json`,
`benchmarks/micro/results/gfx1151/strix-halo/q4-selected-dual-real-slice-hip-vulkan-comparison.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/q4-selected-dual-real-slice-isa-comparison.json`.
Bounded setup/amortization artifact:
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-real-q4-selected-dual-q8_1-dp4a-integration.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Shape: Q4_K selected-dual gate/up, x_rows=`4`, selected rows=`32`,
  experts=`256`, in=`2048`, out=`512`, q8_1 activation format, bf16 output.
- Vulkan implementation: standalone microbench probe with the same q8_1
  quantize shader family plus a Q4_K selected-dual dp4a shader. Timings use
  pre-recorded command buffers and exclude shader compilation, pipeline
  creation, and host-device transfers.
- Correctness: full CPU reference for q8_1 quantize plus Q4_K selected-dual
  gate/up dp4a. All retained local-size rows pass with top-1 `1.0`; best
  local_size=64 max abs is `1`, mean abs `0.03408`.
- Classification: `real_slice_probe`; this is production-shaped backend
  evidence, not a generic compiler proof.

Timing result:

| Backend / local size | q8_1 quantize | Q4_K dot, prequantized | q8_1 quantize+Q4_K dot | Correctness |
| --- | ---: | ---: | ---: | --- |
| HIP retained, threads=256 | `0.00247 ms` | `0.34638 ms` | `0.34582 ms` | top-1 `1.0` vs raw HIP |
| Vulkan local_size=64 | `0.000537 ms` | `0.29607 ms` | `0.29238 ms` | top-1 `1.0` vs CPU |
| Vulkan local_size=128 | `0.000562 ms` | `0.30254 ms` | `0.29648 ms` | top-1 `1.0` vs CPU |
| Vulkan local_size=256 | `0.000549 ms` | `0.31116 ms` | `0.31316 ms` | top-1 `1.0` vs CPU |

Best retained Vulkan row is local_size=64. Relative to retained HIP, Vulkan is:

| Metric | Vulkan speedup vs HIP |
| --- | ---: |
| Q4_K dot, prequantized | `1.17x` faster |
| q8_1 quantize+Q4_K dot | `1.18x` faster |

ISA/stat extraction for local_size=64:

| Shader | SPIR-V dot ops | RADV final dot4 | RADV subgroup | RADV VOPD | RADV registers | Wait/load notes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| q8_1 quantize | `0` | `0` | `64` API, wg32 shader | `0` | official `96` VGPR / `108` SGPR, no scratch/spills; span `96/16` | `6` waitcnt, `1` buffer load |
| Q4_K selected-dual dot | `3` `OpSUDot` | `3` | `64` | `0` | official `48` VGPR / `108` SGPR, no scratch/spills; span `48/31` | `26` waitcnt, `22` buffer loads, LDS `1024` bytes |

Targeted HIP/RADV ISA comparison:

| Shader | HIP/LLVM ISA | Vulkan/RADV ISA | Read |
| --- | --- | --- | --- |
| q8_1 quantize | wave32, `16` SGPR / `11` VGPR, scratch `0`, spills `0`, `157` static instructions, `12` waitcnt-family instructions, `5` VOPD, `0` dot4 | wave64 API, official `108` SGPR / `96` VGPR, scratch `0`, spills `0`, `111` static instructions, `6` waitcnt-family instructions, `0` VOPD, `0` dot4 | RADV has a smaller static quantize shader, but this stage is sub-microsecond in both retained real-slice timings. |
| Q4_K selected-dual dot | wave32, `31` SGPR / `22` VGPR, scratch `0`, spills `0`, `564` static instructions, `35` waitcnt-family instructions, `4` VOPD, `3` dot4, `18` global loads, `6` LDS loads, `2` LDS stores | wave64, official `108` SGPR / `48` VGPR, scratch `0`, spills `0`, `526` static instructions, `26` waitcnt-family instructions, `0` VOPD, `3` dot4, `22` buffer loads, `1` LDS load, `1` LDS store | The Q4 win is not basic dot lowering, spills, or VOPD. The remaining attribution is narrower: surrounding instruction count, wait placement, memory/address structure, or reduction/LDS shape. |

Conclusion: the second matched Vulkan production slice is positive. Unlike the
Q6_K selected-down X8 probe, this Q4_K selected-dual gate/up shape does transfer
a real Vulkan win. The win is not explained by RADV VOPD pairing: the final
RADV dot shader emits `0` VOPD and no scratch/spills, and the targeted HIP
comparison shows HIP also emits the intended dot4 instructions with no
scratch/spills. Treat this as a slice-specific lead. The next useful work is
either an integrated Q4 Vulkan probe to see whether the standalone win survives
backend costs, or a targeted HIP Q4 recovery experiment that attacks the
measured instruction/waitcnt/reduction delta for this exact slice.

Bounded setup/amortization probe:

| Metric | Retained value |
| --- | ---: |
| Vulkan steady Q4_K dot, prequantized | `0.292745 ms` |
| Vulkan steady q8_1 quantize+Q4_K dot | `0.293117 ms` |
| Retained HIP Q4_K dot, prequantized | `0.34638 ms` |
| Retained HIP q8_1 quantize+Q4_K dot | `0.34582 ms` |
| Standalone Vulkan backend setup before steady replay | `47.8645 ms` |
| Breakeven calls if all measured setup is charged to the Q4 quantize+dot win | `~908` |

Setup breakdown:

| Phase | Time |
| --- | ---: |
| Vulkan instance/device setup | `17.4106 ms` |
| Pipeline creation | `0.1736 ms` |
| Buffer allocation | `0.4923 ms` |
| Host staging fill for synthetic dual weights | `25.3268 ms` |
| Device upload | `3.4389 ms` |
| Descriptor setup | `0.0171 ms` |
| Correctness run and readback | `0.9352 ms` |
| Command recording | `0.0639 ms` |

Interpretation: the Q4 Vulkan steady-state win survives the instrumented rerun,
but the standalone first-use envelope only makes sense with persistent Vulkan
objects and resident weights. Pipeline creation is not the blocker in this
probe; the one-shot costs are instance/device setup and synthetic weight
staging/upload. This is still not a production `vulkan_radv_gfx11` backend
result because it does not exercise hipEngine registry integration, whole-layer
residency, or end-to-end decode wall time.

### gfx1151 Dense Q8_0 Real-Slice Probe

Retained artifacts:
`benchmarks/micro/results/gfx1151/strix-halo/hip-q8-0-dense-real-slice.json`,
`benchmarks/micro/results/gfx1151/strix-halo/vulkan-q8-0-dense-real-slice.json`,
`benchmarks/micro/results/gfx1151/strix-halo/q8-0-dense-real-slice-comparison.json`,
and
`benchmarks/micro/results/gfx1151/strix-halo/environment-q8-0-dense.json`.

Hardware/software context:

- GPU: `AMD Radeon 8060S Graphics (RADV STRIX_HALO)`, `gfx1151`.
- Shapes: raw GGUF Q8_0 dense q8_1+dp4a, in/out `768x2048`,
  `2048x2048`, and `2048x6144`; rows=`1/4/8`; row_tile=`1/4`.
- HIP implementation: single-row and rowtile4 probes. Timings use HIP events
  around repeated launches and exclude transfer/build time.
- Vulkan implementation: standalone dense Q8_0 shader, local_size=`64/128/256`,
  row_tile=`1/4`. Timings use pre-recorded command buffers and exclude shader
  compilation, pipeline creation, and transfers.
- Correctness: CPU q8_1/Q8_0 oracle. HIP passes all `18` rows; Vulkan passes
  all `54` rows; the comparison artifact matches all `54` rows.
- Classification: `real_slice_probe`; this closes the dense Q8_0 local matrix
  row for the tested shapes, but it is not ISA attribution because no paired
  HIP/RADV ISA extraction is retained for this Q8_0 shader.

Retained range:

| Metric | Vulkan / HIP range across matched rows |
| --- | ---: |
| q8_1 quantize + Q8_0 dot | `0.279x-1.120x` |
| Q8_0 dot, prequantized | `0.238x-1.169x` |

Best Vulkan row per representative configuration:

| Shape | Rows | row_tile | Best Vulkan local_size | HIP q8_1+dot | Vulkan q8_1+dot | Vulkan/HIP | HIP dot | Vulkan dot | Vulkan/HIP dot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `768x2048` | 1 | 1 | 64 | `6.0111 us` | `5.3684 us` | `1.120x` | `3.7412 us` | `3.8247 us` | `0.978x` |
| `768x2048` | 1 | 4 | 64 | `6.9148 us` | `6.2180 us` | `1.112x` | `4.6516 us` | `4.6449 us` | `1.001x` |
| `768x2048` | 4 | 4 | 64 | `10.3230 us` | `9.2544 us` | `1.115x` | `8.0294 us` | `6.8694 us` | `1.169x` |
| `2048x2048` | 4 | 4 | 64 | `18.8074 us` | `21.7877 us` | `0.863x` | `16.3644 us` | `18.4655 us` | `0.886x` |
| `2048x6144` | 8 | 4 | 64 | `76.0105 us` | `107.5856 us` | `0.707x` | `73.6030 us` | `104.8472 us` | `0.702x` |

Conclusion: dense Q8_0 is not the missing Vulkan win on gfx1151 for these
production-shaped rows. Vulkan has a few small-shape parity/win cases at
`768x2048`, but the larger `2048x2048` and `2048x6144` rows favor HIP,
especially at rows=`4/8`. This closes the prior q8_0 dense GEMV / dense
attention-projection matrix gap as mostly negative for Vulkan. Do not start a
Vulkan dense-Q8 backend path from this row unless future profiling identifies a
different dense Q8_0 shape as the exposed backend decision.

## Answered Questions And Remaining Gates

The original questions now have a useful gfx1151 checkpoint answer:

1. **Compiler scheduling:** Not proven as a broad RADV/ACO advantage. The
   retained Q4_K slice has fewer RADV static instructions and waitcnt-family
   instructions, but the retained Q6_K slice goes the other way and is faster
   on HIP. Treat scheduling as slice-specific until another production slice
   confirms the direction.
2. **Geometry:** Workgroup shape matters, but it is not the whole Vulkan
   ceiling. HIP and Vulkan both prefer wg256 in the retained f32 geometry rows,
   fixed HIP workgroups do not close the gap, and HIP fixed-wave64 makes the
   geometry rows slower.
3. **Wave mode:** HIP wave64 is not the missing switch. It does not close the
   dot/memory gaps, badly regresses gather, and does not close the f32 geometry
   gap when combined with fixed workgroups.
4. **Dispatch/runtime:** Vulkan command-buffer replay has a real launch/runtime
   advantage for tiny bursts. This supports launch fusion and bounded Vulkan
   probes, but it is classified `runtime_dispatch`, not `compiler_aco`.
5. **Memory scheduling:** Synthetic memory rows favor Vulkan, but the first
   memory-heavy production transfer, Q6_K selected-down X8, is faster on HIP
   and has fewer HIP static instructions/waitcnt-family instructions. A generic
   LLVM waitcnt issue is not justified from the current data.
6. **VOPD portability:** gfx1151 evidence is negative for "ACO finds VOPD that
   LLVM misses." HIP emits VOPD in retained diagnostic and real-slice rows;
   RADV emits none in the retained final shaders. Cross-GPU reruns are the
   remaining portability gate.
7. **dp4a/sudot4:** Basic dot lowering is not the blocker. HIP and RADV both
   emit dot4 in retained q8/q4/q6 diagnostics and in the Q4/Q6 real-slice dot
   shaders. Synthetic dot-path wins do not transfer automatically: Q6 loses on
   Vulkan, dense Q8_0 mostly loses on larger Vulkan rows, while Q4 wins.
8. **LLVM roadmap:** No broad LLVM issue should be filed yet. The only current
   LLVM/HIP improvement lead is narrow Q4_K selected-dual recovery, focused on
   the measured instruction count, waitcnt placement, memory/address structure,
   or reduction/LDS shape for that exact hot slice.

Remaining gates:

- **Portability:** rerun the retained harness set on gfx1100/W7900 and
  7900 XTX before treating gfx1151 conclusions as a family-wide RDNA3 result.
- **Q4 recovery:** try a narrow HIP source, builtin, or inline-asm experiment
  only if it targets the retained Q4_K ISA delta and improves the same real
  slice under the CPU oracle.
- **Vulkan product value:** build a production-registry Q4 probe only if we
  decide the bounded setup/amortization result justifies real backend work.
- **New slices:** add another HIP/Vulkan production slice only when profiling
  identifies a shipped hot bucket whose answer would change the backend plan.
  For dense Q8_0, this means a different exposed shape than the retained mostly
  negative local row.

## Ground Rules

- Run HIP and Vulkan on the same hardware, power/clock state, model-shape
  constants, and input data.
- Prebuild HIP code and Vulkan pipelines outside the timed region. Do not time
  shader compilation, `hipcc`, pipeline creation, or descriptor setup unless the
  row is explicitly a startup/build benchmark.
- Every numerical kernel must compare against a CPU oracle or bitwise
  cross-backend reference before timing rows are retained.
- Each benchmark must report both "same algorithm/shape" and "best native shape"
  when possible. The first isolates compiler quality; the second measures the
  backend ceiling users actually care about.
- Record ISA/stat evidence, not only wall time: VGPR, SGPR, scratch/private
  memory, LDS, wave size, workgroup size, occupancy estimate, instruction mix,
  `v_dot4_i32_iu8` presence, VOPD presence, and waitcnt density.
- Treat all Vulkan/RADV diagnostics as platform-specific unless rerun on both
  W7900/gfx1100 and gfx1151.

## Measurement Harness Shape

Build one harness that can generate paired HIP and Vulkan kernels from the same
benchmark descriptor:

```json
{
  "bench": "q4k_gemv_decode",
  "backend": "hip|vulkan",
  "hardware": "gfx1100|gfx1151",
  "compiler": "llvm-amdgpu|radv-aco",
  "algorithm": "dequant_f32|q8_1_sudot4|lds_reduce|subgroup_reduce",
  "workgroup_size": 64,
  "wave_or_subgroup": 32,
  "rows": 4,
  "k": 2048,
  "n": 8192,
  "quant": "q4_k",
  "iters": 1000,
  "warmup": 100,
  "correctness": {
    "max_abs": 0.0,
    "kl": 0.0,
    "top1": 1.0
  },
  "timing": {
    "median_ns": 0,
    "p05_ns": 0,
    "p95_ns": 0
  },
  "isa": {
    "vgpr": 0,
    "sgpr": 0,
    "scratch_bytes": 0,
    "lds_bytes": 0,
    "wave_size": 0,
    "vopd_count": 0,
    "dot4_count": 0,
    "waitcnt_count": 0
  }
}
```

The first implementation can use separate HIP and Vulkan source templates. A
later version can auto-generate both from a small DSL, but that is not required
for the first attribution pass.

## Microbenchmark Matrix

### 1. Dispatch And Grid Floor

Purpose: separate command/runtime overhead from shader body speed.

| Bench | Variants | Attribution |
| --- | --- | --- |
| no-op kernel | 1, 16, 64, 256, 1024, 8192, 65536 blocks | Per-dispatch and grid-size scheduling cost |
| tiny ALU kernel | 2 args vs 16 args, same grid | Arg marshaling vs GPU dispatch |
| command burst | N kernels in one host call / command buffer | HIP launch loop vs Vulkan command-buffer replay |
| dependent chain | one block, fixed cycles | Compiler body scheduling without grid effects |

Retain this first. If Vulkan only wins here, then a Vulkan backend may help
launch-heavy paths, but LLVM kernel codegen is not the target.

Status: retained on gfx1151/STRIX_HALO. Repeat on gfx1100/W7900 before treating
the magnitude as cross-GPU, but do not spend more iteration time here until the
matched math kernels exist.

### 2. Geometry Sweep

Purpose: quantify dead lanes and subgroup shape effects independently of quant.

| Bench | Shapes | Variants |
| --- | --- | --- |
| f32 GEMV row | K=512, 2048, 8192 | workgroup 32/64/128/256; subgroup/wave 32/64 |
| reduction only | K=512, 2048, 8192 | LDS tree, shuffle/subgroup, one-wave |
| selected-MoE index gather | 8 selected experts, K=512 and 2048 | compact IDs vs scattered IDs |
| rows>1 verifier GEMV | rows=1/2/4/8 | `grid.y=rows` vs rowtile vs subgroup batch |

Attribution rule: if matched 64-thread HIP closes the Vulkan gap, this is a
kernel geometry issue, not ACO superiority. If Vulkan still wins at identical
geometry, inspect compiler statistics.

### 3. Memory Coalescing And Waitcnt

Purpose: isolate ACO/LLVM memory scheduling and wait-state differences.

| Bench | Variants | What To Inspect |
| --- | --- | --- |
| coalesced load+accumulate | vector widths 1/2/4/8 | GB/s, waitcnt density, VGPR |
| strided load+accumulate | stride 2/4/8/16 | cache behavior, waitcnt placement |
| gather IDs | random selected expert rows | scalar/vector address math |
| load-compute interleave | unroll 1/2/4/8/16 | whether compiler overlaps memory with ALU |

Attribution rule: same memory traffic but lower waitcnt density and higher
effective GB/s on Vulkan is an ACO scheduling win. Same waitcnt but better
coalescing is a source/layout issue.

Status: retained on gfx1151. Vulkan is `1.02x-2.25x` faster on most coalesced,
strided, and interleave rows, while gather is essentially tied at `1.02x`.
HIP wave64 and fixed-block controls do not close the gap; both severely regress
gather in their retained runs. This is strong memory-side diagnostic evidence,
but the first memory-heavy production-shaped transfer check, Q6_K X8
selected-down, is negative: Vulkan is `1.67x` slower combined and the targeted
ISA join shows RADV has more static instructions and waitcnt-family
instructions. Do not file an LLVM-AMDGPU waitcnt/scheduling claim from this
synthetic sweep alone. RADV shaderstats now gives official allocation counts
and shows no Vulkan scratch/spills.

### 4. VOPD And VALU Scheduling

Purpose: determine whether RADV/ACO materially beats LLVM on RDNA3 dual-issue.

| Bench | Variants | Expected Read |
| --- | --- | --- |
| independent f32 FMA pairs | 2, 4, 8 accumulators | VOPD pairing opportunity |
| dependent f32 chain | no independent accumulators | VOPD should not help |
| dequant-like chain | shift/mask/sub/cvt/mul/fma | limited VOPD opportunity |
| mixed integer+float chain | address math plus fma | register-bank and scheduler stress |

Attribution rule: count VOPD instructions and compare elapsed cycles. ACO only
"wins on dual issue" if the Vulkan shader emits more useful VOPD or equivalent
dual-issue scheduling at matched occupancy and instruction count.

Status: retained on gfx1151. HIP emitted VOPD in all retained rows and RADV
emitted none, so the current retained evidence is a negative result for
"Vulkan wins because ACO finds VOPD that LLVM misses." Cross-GPU reruns can
check portability, but the next gfx1151 compiler tests should move to real-slice
confirmation, not more generic VOPD sweeps.

### 5. Dot4 / q8_1 / sudot4

Purpose: determine whether the dot path is compiler-bound or layout-bound.

| Bench | Shapes | Variants |
| --- | --- | --- |
| q8_1 x q4 dot | K=512/2048/8192 | scalar dequant vs `sudot4` |
| q8_1 x q5/q6 selected-down | K=512/2048 | raw sidecar vs T16 decode-repack |
| q8_0 dense GEMV | rows=1/4/8 | exact f32 path vs q8_1 dot diagnostic |
| quantize activation | rows=1/4/8, K=2048/8192 | cost of q8_1 materialization |

Attribution rule: if both compilers emit `v_dot4_i32_iu8` and timing is close,
the next work is layout/quant economics, not LLVM. If HIP fails to emit the
expected dot instruction or surrounds it with worse waitcnt/spills, that is an
LLVM codegen target or a hand-ISA candidate.

Status: retained on gfx1151 for packed q8 signed, q4 unsigned-byte by signed-q8,
q6 zero-point correction, and scalar q4 dequant rows. HIP and RADV both emit
final dot4 instructions in q8/q4/q6 rows, and HIP reports no scratch/spills.
Vulkan remains `3.28x-3.42x` faster, including the scalar row, and HIP wave64
and fixed-block controls do not close the gap, so basic dot-instruction
availability, wave mode, and runtime block indexing are no longer the main
questions. HIP real-slice q8_1 materialization/layout controls are retained and
positive; the Vulkan real-slice checks are retained and split, with Q6_K X8
negative, dense Q8_0 mostly negative on larger shapes, and Q4_K selected-dual
positive.

### 6. LDS, Barriers, And Subgroup Reductions

Purpose: test the old HIP LDS/barrier reduction shape against Vulkan subgroup
reduction and HIP shuffle variants.

| Bench | Variants | Retain If |
| --- | --- | --- |
| one-row reduction | LDS tree, wave shuffle, subgroup reduce | Shows crossover by K |
| two-stage reduction | block partials + final reduce | Useful for large K |
| no-LDS accumulators | 4/8/16 accumulators per lane | Beats LDS without spills |
| barrier stress | same math with inserted barriers | Quantifies barrier tax |

Attribution rule: subgroup reduction wins are not automatically compiler wins.
They are usually algorithm/geometry wins unless matched HIP shuffle code still
lags with similar ISA quality.

Status: retained on gfx1151 for one-row K=`512/2048/8192`, wg=`64/256` in the
original LDS/barrier/subgroup sweep and wg=`32/64/256` in the accumulator
extension. HIP wave-shuffle reduction is flat versus HIP LDS
(`0.991x-1.005x`), Vulkan subgroup reduction is mostly flat to modestly slower
versus Vulkan LDS (`0.984x-1.132x`), and matched Vulkan LDS remains
`8.19x-14.55x` faster than matched HIP LDS. HIP 4/8/16 lane-local accumulator
variants are mostly slower than HIP LDS (`1.02x-2.40x`), while matched Vulkan
accumulator rows remain `9.57x-15.81x` faster than HIP. Reduction topology and
accumulator count do not close the f32 geometry gap. True two-stage
block-partial reduction remains deferred until profiling shows a large-K
reduction bucket where it can change an implementation decision.

### 7. Representative Inference Slices

Purpose: keep microbenchmarks tied to real hipEngine work.

| Slice | Current Importance | What To Compare |
| --- | --- | --- |
| small-K expert-down | known dead-lane risk | 64-thread subgroup/Vulkan vs HIP 64/128/256 |
| selected gate+up dual | hot MoE bucket | T16 exact, raw q8_1 dot, subgroup variants |
| dense q8_0 attention proj | hot AR/verify bucket | row=1 and rows=4/8 |
| GDN/recurrent chain | verifier bucket | memory scheduling and register pressure |
| q6 lm-head rowtile | shipped rowtile win | matched Vulkan large GEMV bandwidth |
| sampler/top-k/argmax | exposed server bucket | subgroup reductions and command fusion |

Attribution rule: no real-kernel lesson is retained unless a microbench result
predicts the direction and the actual slice confirms it.

Status: retained on gfx1151 for HIP q8_1 layout economics on Q4_K selected-dual
gate/up and Q6_K selected-down X8. The retained HIP slices confirm q8_1
materialization cost is small and the full q8_1+dp4a path is faster than the
tested HIP float controls. The matched Vulkan Q6_K X8 selected-down production
slice is retained and negative, with a targeted HIP/RADV ISA comparison showing
the negative Q6 result is not missing RADV dot4 or caused by Vulkan spills; the
matched Vulkan Q4_K selected-dual gate/up slice is retained and positive. The
dense Q8_0 real-slice probe is retained and mostly negative for Vulkan on the
larger tested shapes, with small `768x2048` exceptions. A bounded
setup/amortization probe for that
Q4_K slice is also retained: steady replay remains positive, but one-shot
standalone setup requires roughly `908` calls to amortize if charged directly.
The sampler top-1/top-k8 argmax diagnostics are also retained and strongly
positive for Vulkan, but they are not stochastic sampler or lm-head+sample
integration rows.
This split result is why production backend work needs true registry/end-to-end
or cross-GPU confirmation instead of more generic microbenchmarks.

## Result Classification

Each retained result should classify the win into exactly one primary bucket:

| Bucket | Definition | Likely Next Action |
| --- | --- | --- |
| `compiler_aco` | Same algorithm/layout/geometry, Vulkan faster with better ISA stats | File LLVM issue or try hand-ISA on that kernel |
| `geometry` | Matched HIP geometry closes most of the gap | Rewrite HIP kernel shape |
| `wave_mode` | HIP wave64 or subgroup-size control changes result materially | Add guarded wave experiment, then full correctness gate |
| `runtime_dispatch` | No-op/grid/command rows explain the gap | Consider Vulkan backend or fewer/larger kernels |
| `layout_quant` | Dot/layout changes dominate compiler choice | Port layout, not backend |
| `fusion_topology` | Per-op kernels match, fused Vulkan command wins | Fuse kernels or add backend-level composite |
| `not_reproducible` | Difference disappears under controlled harness | Do not roadmap work from the old observation |
| `diagnostic_unclassified` | Gap remains but evidence is insufficient for one primary cause | Add the missing control or ISA/stat evidence |

## LLVM Improvement Map

If a row lands in `compiler_aco`, map it to an LLVM-facing request:

| Symptom | Evidence Required | LLVM/HIP Improvement Target |
| --- | --- | --- |
| Under-unrolled loops | Same source, Vulkan fewer loop overhead instructions | gfx11 loop unroll heuristic for small GEMV/reduction loops |
| Excess waitcnt | Similar memory traffic, HIP has more waitcnt/nops and lower GB/s | waitcnt placement and memory scheduling |
| Register spill | HIP `Scratch_Size > 0`, Vulkan no spill at same live values | VGPR allocator / scheduling pressure reduction |
| Missed VOPD | Independent VALU pairs present, ACO emits VOPD, LLVM does not | VOPD pairing and register-bank aware scheduling |
| Bad wave64 codegen | HIP wave64 compiles but regresses from spills/incorrect reductions | wave64 lowering and shuffle/subgroup correctness |
| Missed dot pattern | HIP source expresses dot, no `v_dot4_i32_iu8` appears | intrinsic use or pattern matching for q8_1/q4 block dots |
| Poor immediate/address code | Same loads, HIP has more scalar/vector address ops | address-combine and scalarization passes |

Rows without ISA/stat evidence should not be used as LLVM asks. They can still
justify hipEngine kernel rewrites or a Vulkan backend.

Current status: no retained row is clean enough to file as a broad
LLVM-AMDGPU-vs-RADV/ACO issue. The Q4_K selected-dual slice is the only current
LLVM/HIP recovery lead, and even that is a slice-specific follow-up: it must
target the measured instruction count, waitcnt placement, memory/address
structure, or reduction/LDS shape and prove a real-slice speedup under the same
oracle.

## Vulkan Backend Effort Scale

There are two distinct scopes. The retained evidence supports the first scope
as an attribution tool. It does **not** yet support the second scope as product
work.

### Narrow Vulkan Probe Backend

Purpose: enough Vulkan compute infrastructure to run paired microbenchmarks and
one or two hot inference slices.

Effort class: **bounded probe**. This is a contained runtime/tooling project,
not a production backend. The value is high because it can falsify or confirm
the Vulkan ceiling without touching `hipengine.LLM.generate()`.

Current status: **implemented enough for attribution and already useful**. The
retained dispatch, geometry, memory/waitcnt, VOPD, dot-path, shaderstats, Q6_K
X8 real-slice, and Q4_K selected-dual real-slice rows all came from this style
of standalone Vulkan probe.

Work:

- Create a tiny Vulkan compute runtime: instance/device/queue selection,
  command pool, command buffer, fence/timeline synchronization, buffer
  allocation, descriptor sets, pipeline cache.
- Add shader build flow for GLSL or SPIR-V templates with specialization
  constants for K, rows, quant, and workgroup size.
- Add CPU-oracle validation and JSON result output compatible with benchmark
  artifacts.
- Add shader dump/disassembly collection where RADV exposes it; record exact
  Mesa/RADV/ACO version and environment.
- Keep it outside `hipengine.LLM.generate()` until correctness and packaging are
  understood.

This scope is the right first step because it answers whether Vulkan is worth
productizing without disturbing the current HIP backend.

The retained gfx1151 dispatch result strengthens the case for this probe, but
only for launch-heavy or command-fusion-sensitive paths. It is not sufficient
evidence for a production backend by itself.

Decision gate for continuing the probe: add only tests that can change an
engineering decision. More generic dispatch/VOPD/dot/memory rows should stop.
The next Vulkan work should be cross-GPU validation or a true production-registry
Q4_K selected-dual probe only if the bounded setup/amortization result is enough
to justify backend work. The targeted Q4 ISA comparison and bounded setup probe
are already retained, so more standalone Q4 work is only useful if paired with
a HIP source/inline-asm recovery attempt or end-to-end integration.

### Production Vulkan Backend

Purpose: a real hipEngine backend registered as a peer of `hip_gfx1100`, likely
`vulkan_radv_gfx11`, with enough kernels to run GGUF/PARO decode.

Effort classes:

- **Large, narrow path** for a single-model/single-quant decode backend after
  the probe exists. This means real registry integration, real device memory
  ownership, and a small set of hot shader ports. It is still narrow enough to
  stay below "second full backend" scope.
- **Very large, systemic path** for a maintainable Vulkan backend with server
  integration, profiling, startup caching, multiple quant paths, and parity
  tests comparable to HIP. This should be treated as a second kernel stack
  unless the probe proves only a few shaders need Vulkan.

Major work items:

- Backend runtime abstraction: Vulkan device buffers, command submission,
  descriptor/pipeline caches, error reporting, and lifecycle management.
- Kernel registry integration using the existing `(backend, layer, quant,
  variant)` model. Do not add `if backend == "vulkan"` branches in engine code.
- Shader ports for the hot decode kernels: selected MoE gate/up/down, dense
  q8/q4 GEMVs, attention decode, GDN/recurrent pieces, lm-head/sample buckets,
  and KV writers.
- Persistent pipeline and specialization-cache policy keyed by GPU, Mesa/RADV,
  shader source hash, quant, and shape.
- Correctness gates against `kernels/cpu_reference/` or existing CPU oracles for
  each shader family.
- Benchmark integration for prefill/decode/server rows and RGP/RADV profiling
  capture.
- Packaging story for systems without RADV or without required subgroup/dot
  extensions.

Risks:

- Shader duplication can become a second kernel stack as large as HIP if not
  limited to proven hot kernels.
- Vulkan memory/descriptors are a different ABI from HIP pointers; zero-copy
  sharing is not the default design.
- Some wins may depend on RADV/ACO behavior and not general Vulkan.
- Debug/profiling loops are slower than HIP until the harness is mature.

Current decision: **do not start production Vulkan yet**. The retained
dispatch-row win is a runtime result, and the first two matched
production-shaped slices are split: Q6_K X8 is slower on Vulkan, Q4_K
selected-dual is faster. The bounded Q4 setup/amortization probe shows steady
replay remains positive but one-shot standalone setup swamps the per-call win
unless Vulkan objects and weights are persistent. Production Vulkan should wait
for another retained hot-slice win or a true production-registry/end-to-end Q4
probe showing that persistent residency and registry costs still move wall time,
and that HIP cannot recover the delta with a narrower geometry/codegen/hand-ISA
fix.

## Hand-ISA / Inline Assembly Candidates

Hand-ISA is narrower than a Vulkan backend. It is justified when a hot HIP
kernel is stable, isolated, and blocked by LLVM codegen rather than algorithm.

Current decision: **no broad hand-ISA path yet**. The retained gfx1151 evidence
does not show LLVM missing VOPD or dot4 in the generic diagnostics. HIP emits
VOPD in the retained VOPD rows, HIP emits dot4 in the retained q8/q4/q6
dot-path rows, the matched Vulkan Q6_K X8 real slice is slower than HIP, and
the matched Vulkan Q4_K selected-dual slice is faster without using VOPD. The
targeted Q4_K ISA comparison also shows HIP emits dot4 and no spills for the
positive Q4 slice. A hand-ISA candidate must therefore come from a specific hot
HIP slice with a measured avoidable instruction, waitcnt, reduction, or address
sequence, not from the generic Vulkan ceiling observation.

Good candidates:

- Inner q8_1/q4/q5/q6 dot loops only where the desired `v_dot4_i32_iu8`
  sequence is known, HIP's final ISA contains avoidable surrounding work, and a
  real slice proves that fixing it moves wall time.
- Q4_K selected-dual only if a HIP source rewrite or inline-asm experiment
  specifically reduces the retained `564` instruction / `35` waitcnt-family
  HIP dot shader toward the RADV `526` instruction / `26` waitcnt-family shape
  and improves the real-slice timing without correctness drift.
- Small-K selected-MoE kernels only where ACO proves better waitcnt/register
  scheduling at identical geometry and the same production slice is faster on
  Vulkan.
- F32/BF16 dequant chains only if a future microbench proves missed pairing or
  a specific instruction sequence matters after occupancy and memory traffic are
  controlled. Current gfx1151 VOPD rows do not provide that proof.
- Tiny sampler reductions only if stochastic sampling or fused lm-head+sample
  profiling keeps the sampler bucket exposed after the deterministic top-1/top-k8
  evidence.

Bad candidates:

- Launch overhead or command scheduling problems.
- Kernels whose performance changes mostly with workgroup geometry.
- Kernels that still have unresolved correctness drift or volatile ABIs.
- Whole-model assembly rewrites.
- Attention/GDN kernels where math/control complexity would make maintenance
  worse than the expected gain.

Implementation options, in increasing maintenance cost:

1. Use AMDGCN builtins such as `__builtin_amdgcn_sudot4` in normal HIP source.
2. Add small inline-asm blocks inside HIP kernels for a proven instruction
   sequence.
3. Build a standalone amdgcn assembly/HSACO kernel and load it through the HIP
   module path.

Promotion requirements:

- CPU-reference correctness gate.
- Disassembly proving the intended instruction sequence.
- `rocprofv3 --kernel-trace` proving the kernel ran.
- Same-suite non-regression in the real inference slice.
- A `docs/REFACTOR.md` entry for any diagnostic env flag.

Do not use hand-ISA to address launch overhead, backend command replay, or
generic geometry gaps. Those need launch fusion, backend scheduling, or HIP
kernel-shape work, not inline assembly.

## Initial Execution Order

1. Implement dispatch/grid floor rows for HIP and Vulkan. Status: retained on
   gfx1151; rerun on gfx1100/W7900 when that machine is available.
2. Implement geometry sweep for f32 GEMV/reduction at K=512/2048/8192. Status:
   retained on gfx1151; workgroup shape alone does not explain the gap in the
   repeat-shifted f32 GEMV/reduction harness.
3. Extract ISA/stat evidence for the retained f32 GEMV/reduction geometry
   kernel. Status: retained on gfx1151 for K=2048 rows=1 wg64/wg256; HIP emits
   VOPD, RADV does not, and the row remains `diagnostic_unclassified`.
4. Add dependent-chain and independent-accumulator VOPD microbenches with
   ISA/stat extraction. Status: retained on gfx1151; HIP emits VOPD and RADV
   does not in all retained rows, so VOPD pairing is not the current ACO
   explanation.
5. Add memory waitcnt microbenches: coalesced load+accumulate, strided
   load+accumulate, gather IDs, and load-compute interleave. Status: retained
   on gfx1151; Vulkan has a broad memory-side advantage except gather, but
   wave64 and fixed-block controls do not close it. RADV shaderstats now shows
   no Vulkan scratch/spills. The first memory-heavy production-slice transfer
   check, Q6_K X8 selected-down, is negative for a generic LLVM-AMDGPU
   waitcnt/scheduling claim, so the synthetic memory rows remain diagnostic.
6. Add q8_1/sudot4 and scalar-dequant GEMV pairs. Status: retained on gfx1151
   for packed dot-path diagnostics; HIP and RADV both emit dot4 in q8/q4/q6
   rows, but Vulkan remains `3.28x-3.42x` faster. HIP wave64 and fixed-block
   controls do not close the gap, and the row is `diagnostic_unclassified`.
7. Add HIP fixed-shape controls for dot, memory, and geometry. Status:
   retained on gfx1151; runtime block indexing/workgroup specialization is not
   the missing switch.
8. Add HIP fixed-wave64 geometry controls. Status: retained on gfx1151; HIP
   fixed-wave64 is slower than fixed wave32 and leaves Vulkan
   `6.31x-16.18x` faster, so wave mode is not the missing f32 geometry switch.
9. Port HIP real-slice q8_1 layout controls: selected-MoE small-K and q6
   selected-down. Status: retained on gfx1151; q8_1 materialization is small
   and production-layout HIP q8_1+dp4a is positive.
10. Port matched Vulkan real slices. Status: retained for Q6_K selected-down X8
   and Q4_K selected-dual gate/up on gfx1151. Q6_K is slower than HIP by
   `1.67x` on quantize+dot, while Q4_K is faster than HIP by `1.18x` on
   quantize+dot. This split result argues for targeted follow-up, not a broad
   backend jump.
11. Add targeted HIP/RADV ISA comparison for the negative Q6_K X8 slice.
   Status: retained on gfx1151; both paths emit `9` dot4 instructions and no
   scratch/spills, but RADV has more static instructions and waitcnt-family
   instructions while running slower. This is the production-shaped negative
   transfer check for the broad synthetic memory/waitcnt claim.
12. Add targeted HIP/RADV ISA comparison for the positive Q4_K selected-dual
   slice. Status: retained on gfx1151; both paths emit `3` dot4 instructions
   and no scratch/spills, HIP emits VOPD while RADV emits none, and RADV has
   fewer static instructions and waitcnt-family instructions. This narrows Q4
   follow-up but does not make it a broad `compiler_aco` row.
13. Add LDS/barrier/subgroup reduction controls. Status: retained on gfx1151;
   HIP wave-shuffle and Vulkan subgroup reduction do not close the f32 geometry
   gap, so reduction topology is not the missing switch.
14. Add 4/8/16 lane-local accumulator reduction controls. Status: retained on
   gfx1151; accumulators do not recover HIP and matched Vulkan accumulator rows
   remain `9.57x-15.81x` faster. True two-stage reduction is still unrun and
   not decision-grade without a new large-K reduction profile.
15. Add bounded Q4_K Vulkan setup/amortization probe. Status: retained on
   gfx1151; steady Q4 remains positive, but standalone setup requires persistent
   residency and does not justify production Vulkan by itself.
16. Classify each retained row using the result buckets above.
17. Only then decide between LLVM issue, HIP rewrite, hand-ISA, or production
   Vulkan backend.

The next useful tests are cross-GPU reruns of the retained harnesses, a narrow
HIP Q4_K selected-dual recovery experiment only if we intend to act on the
measured ISA delta, and a true production-registry Q4 Vulkan probe only if we
decide to invest in backend work. Run another memory-bound production-slice
test only if new profiling identifies a shipped hot bucket that should transfer;
the retained Q6_K X8 check already answers the first one negatively.
The gfx1151 geometry, VOPD, memory/waitcnt, and dot-path extractions already
found that the current gap is not a missed-HIP-VOPD, HIP-spill, missed-dot4,
broad HIP wave-mode, simple LDS/subgroup reduction-topology story, or missing
lane-local accumulator variant. The next stop is portability and integration
validation rather than rerunning broader geometry, generic VOPD, generic
memory, reduction-topology, accumulator, or basic dot-lowering sweeps.

The expected useful output is not a single "Vulkan is faster" number. It is a
ranked list of deltas like: "Vulkan wins small-K expert-down by X%; Y% is
geometry, Z% is ACO waitcnt/VGPR quality, remaining is dispatch." That is the
level of evidence needed to guide LLVM work or justify a backend investment.
