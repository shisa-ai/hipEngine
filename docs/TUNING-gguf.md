# GGUF Tuning Plan

Date: 2026-06-17
Branch/worktree: `main` / `/home/lhl/hipEngine`
Scope: Qwen3.6-35B-A3B GGUF on GPU1/gfx1100 (`AMD Radeon RX 7900 XTX`, 24 GiB-class) as the active eval/testbed. The canonical performance target is now `Q4_K_M` to match the local `llama.cpp` rows 1:1; `Q4_K_S` remains a secondary memory/consumer-card diagnostic unless explicitly requested. W7900 rows are comparison references and the 0.8B GGUF fixtures are fast correctness sentinels.

## Thesis

Our GGUF hot path is HIP/C++ and resident-weight based. Once a model is loaded,
there should be no structural reason for hipEngine GGUF to trail either local
`~/llama.cpp/` or the native PARO path on the same gfx1100 GPU. The gap should be
closed the same way the recent MTP/DFlash work moved from a slow but correct
path to retained wins: lock a same-suite baseline, profile the exact phase that
is slow, keep every exact non-regressive micro-win, and reject attractive
launch-reduction ideas when the profile says they move work to a slower bucket.

This file is the active GGUF-specific tuning playbook and punchlist. The running `pi-multiloop` lane is `gguf-tuning/run-20260615-103446`; keep iteration notes here and detailed evidence in `WORKLOG.md`.

Active gates:

- **Primary acceptance gates:** GPU1 `512/128` and `4K/128` on `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` / `gguf_q4_k_m`, measuring the full vector: prefill median, decode median, final generated IDs/logits, tracked peak, and sampled HIP-used peak when available. The scalar multiloop metric is advisory only; it is not an acceptance criterion by itself.
- **Promotion check:** run `128K/128` before claiming a default-path GGUF win; if GPU1 cannot fit, record the blocker and rerun on the W7900 only as an explicitly labeled fallback.
- **Correctness/memory:** stable generated IDs/logits on the gates, targeted GGUF tests green, and no raw+packed duplicate residency or unexplained peak-memory growth.
- **Tradeoff policy:** do not unilaterally retain a candidate that regresses any gate dimension (prefill, decode, correctness, tracked/sampled memory, or launch/kernel fallback mix), even if another dimension improves. Re-run to separate noise from signal; if a tradeoff remains, park it as no-hold and ask the user/human lead to decide whether the tradeoff is acceptable.

It complements:

- [`GGUF.md`](GGUF.md) — loader/runtime status and GGUF format notes.
- [`GGUF_DECODE_REPACK.md`](GGUF_DECODE_REPACK.md) — T16 decode-repack layout.
- [`KERNELS.md`](KERNELS.md) — kernel catalog and port/playbook requirements.
- [`ROOFLINE.md`](ROOFLINE.md) — RDNA3/W7900 roofline and do-not-chase rules.
- [`BENCHMARK.md`](BENCHMARK.md) — promotion/evidence contract.
- [`MTP.md`](MTP.md), [`DFLASH.md`](DFLASH.md), and [`MEGAKERNEL.md`](MEGAKERNEL.md) — the recent tuning pattern to copy.

## Compact goal

Close the GGUF gap without weakening the architecture:

1. **Beat local `~/llama.cpp/` matched GGUF rows** on the same model/quant,
   backend/device, prompt token IDs, KV dtype, context, and run environment.
   Re-measure current `~/llama.cpp/` first; the README rows are comparison
   anchors, not a substitute for a fresh matched run.
2. **Reach PARO-class decode for c=1 short/mid shapes** where quant/model
   differences do not make the comparison meaningless. Primary acceptance is
   GPU1 `512/128` and `4K/128`; later promotion must also survive `128K/128`.
   Use the parent PARO c=1 W7900 rows at `512/128`, `4K/128`, and `32K/128` as
   target anchors, but refresh same-host PARO rows whenever they fit the active
   device.
3. **Keep the GGUF value proposition:** no torch hot path, no llama.cpp FFI shim
   on the hot path, no backend/quant branches in model/dispatch code, and no
   raw+packed duplicate residency in promoted paths.
4. **Restore 24 GiB-class viability where possible.** The Q4_K_M primary gate
   currently fits `512/128` and `4K/128` at about `22.49 GiB` tracked peak
   (`~23.04 GiB` sampled HIP-used), leaving less than 1 GiB free on GPU1. Any
   promoted consumer-card row needs a specific memory plan and must not trade
   away prefill/decode/correctness unless the user explicitly accepts it.

## Current scorecard to explain, not yet the final baseline

The table below stitches together older documented rows and is no longer the
active 1:1 baseline because it mixes hipEngine `Q4_K_S` with llama.cpp `Q4_K_M`.
Treat it as historical context only. New active GGUF tuning must use Q4_K_M
across hipEngine and llama.cpp unless the user explicitly asks for a Q4_K_S
memory/consumer-card diagnostic.

| Workload | hipEngine GGUF Q4_K_S, W7900 TheRock 7.13 | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M | PARO parent c=1 target |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | `2262.097` prefill / `109.347` decode / `25.108 GiB` | `2436.049` / `85.487` / `21.125 GiB` | `1816.927` / `127.515` / `20.844 GiB` | `2696.4` / `116.05` / `18.80 GiB` |
| 4K/128 | `2544.475` / `99.873` / `25.108 GiB` | `2176.905` / `87.375` / `21.197 GiB` | `1705.093` / `120.163` / `20.969 GiB` | `2741.5` / `113.05` / `21.64 GiB` |
| 32K/128 | `1878.052` / `86.486` / `25.108 GiB` | `1496.409` / `76.994` / `21.738 GiB` | `1128.554` / `98.073` / `21.533 GiB` | `1880` / `98.8` / `21.37 GiB` |
| 128K/128 | `995.295` / `58.066` / `25.108 GiB` | `710.213` / `57.341` / `23.605 GiB` | `480.539` / `64.478` / `23.596 GiB` | `914` / `62.6` / `27.42 GiB` |

Sources: `benchmarks/README.md` 2026-06-14/15 rows and the source-lineage
PARO table. Caveats:

- The hipEngine column is Q4_K_S while the llama.cpp columns are Q4_K_M; do not
  use this table for 1:1 performance acceptance. The current active loop should
  instead use `/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` with
  `--quant gguf_q4_k_m`.
- The GGUF README rows are retained diagnostics and explicitly carry
  `performance_claim=false`; promotion requires the full `docs/BENCHMARK.md`
  evidence and correctness gate.
- The user's current observation that GGUF is slower than local `~/llama.cpp/`
  takes precedence over stale table interpretation. Refresh the local
  `~/llama.cpp/` rows before deciding whether the primary gap is HIP, Vulkan, or
  PARO.

Initial read: GGUF decode is already ahead of older llama.cpp HIP rows but behind
llama.cpp Vulkan and PARO on most c=1 decode shapes, and it uses more memory.
That points to kernel shape/layout, dispatch mix, and residency policy rather
than file parsing or Python host code.

## Active GPU1 gate baseline

Canonical GGUF tuning is now `Q4_K_M`, not `Q4_K_S`, so comparisons to
llama.cpp are 1:1. Current correctness-restored GPU1 Q4_K_M baseline is
`benchmarks/results/2026-06-17-gpu1-gguf-q4km-correctness-restored.json`, after
removing the e089 fused selected-expert/raw-Q8 correctness branch and restoring
the pre-e089 dense Q4_K materialization policy:

| Workload | Prefill tok/s median | Decode tok/s median | Tracked peak | Sampled HIP used peak | Correctness sanity |
| --- | ---: | ---: | ---: | ---: | --- |
| 512/128 | `2307.515` | `125.095` | `22.487 GiB` | `23.039 GiB` | stable final token `318` |
| 4K/128 | `2599.357` | `114.195` | `22.487 GiB` | `23.041 GiB` | stable final token `220` |

This restores the previous Q4_K_M correctness-good IDs from
`benchmarks/results/2026-06-16-gpu1-gguf-q4km-current-parity-diagnostic.json`
while improving both primary prefill medians and keeping tracked memory flat
(`-0.004 GiB` vs that artifact). The interim current-main diagnostic
`benchmarks/results/2026-06-17-gpu1-gguf-q4km-current-512-4k-diagnostic.json`
was invalid because it changed IDs (`318/220 -> 38118/1076`). The primary
multiloop scalar metric remains the minimum gate decode rate, now `114.195 tok/s`, but acceptance is the full gate vector; do not keep any
prefill/decode/memory tradeoff without user approval.

The GPU1 Q4_K_M 24 GiB memory baseline is now recorded in
`benchmarks/results/2026-06-18-gpu1-gguf-q4km-memory-baseline.json`: before the
mid-context policy, `51K/128` was the largest observed pass (`1597.225` prefill
/ `86.356` decode tok/s, `23.434 GiB` tracked peak) and `52K/128` failed during
prefill with zero free memory. The retained mid-context policy artifact
`benchmarks/results/2026-06-18-gpu1-gguf-q4km-memory-policy-midcontext.json`
lowers only the 24 GiB >=52K full-attention query chunk from 4096 to 1024 rows;
short gates stay below the threshold and keep stable IDs/peak, while the largest
observed pass moves to `103K/128` (`866.728` / `71.053 tok/s`, `23.484 GiB`) and
`104K/128` is the first observed OOM. The default `128K/128` final memory gate
remains blocked by BF16 full-attention KV/weight residency:
`benchmarks/results/2026-06-17-gpu1-gguf-q4km-128k-final-gate-blocked.json`.
A 2026-06-18 opt-in diagnostic offloaded the sole `raw_gguf` resident weight
(the Q8_0 token embedding, `0.503 GiB`) via
`HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING=1` and proved `128K/128` can allocate and
run on GPU1 (`761.864` prefill / `11.141` decode tok/s, `23.400 GiB` tracked /
`23.913 GiB` sampled, finite logits), but decode graph replay is disabled in
that mode because generated token IDs are device-resident inside the graph. This
artifact (`benchmarks/results/2026-06-18-gpu1-gguf-q4km-host-token-embedding-128k-diagnostic.json`)
is a capacity proof, **not** a promoted path. A follow-up explicit
`--kv-storage int8_per_token_head` diagnostic kept graph-class decode and fit
`128K/128` on GPU1 (`760.724` prefill / `64.923` decode tok/s, `22.911 GiB`
tracked / `23.472 GiB` sampled) via INT8 full-attention KV plus a temporary
BF16 prefill-oracle cache, but it is also **not** promoted: the initial short
primary gates drifted IDs (`512/128` `318 -> 220`, `4K/128` `220 -> 34105`) and
`512/128` decode regressed (`123.726 -> 112.313 tok/s`) versus the default BF16
rerun. Artifact:
`benchmarks/results/2026-06-18-gpu1-gguf-q4km-int8kv-128k-diagnostic.json`.
The 2026-06-21 short BF16 mirror fixes short-gate IDs/logits, but the unmirrored
INT8-only route still fails BF16-vs-INT8 logits (`4K` no-mirror W7900 gate:
`KL=0.275781`, top-1 `0.5`). The 2026-06-22 localization found the error source:
early full-attention layers amplify small INT8 value-quantization perturbations
(layer 3 alone gives `KL=0.618776`), while later layers are safe enough with FP32
scales. Long explicit GGUF INT8 now uses a correctness-admitted hybrid layout by
default: 3 BF16-primary full-attention layers followed by 7 INT8 layers with
effective FP32 scales. The W7900 forced-long `4K` BF16-vs-hybrid gate passes
(`KL mean=0.014025`, `KL max=0.028051`, top-1 `1.0`, no BF16 mirror); pure
INT8-only reproduction still requires `HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1`.
Do not promote or claim a throughput row for 24 GiB-class `128K/128` Q4_K_M until
a full benchmark completes; GPU1 allocation of the hybrid `131328`-position
session fits (`25,008,050,176` tracked bytes), but the attempted `128K/128`
throughput run timed out before artifact. Evidence:
`benchmarks/results/2026-06-22-gguf-q4km-int8kv-hybrid-correctness.json`.

The older Q4_K_S gate (`512/128` `1958.693 / 126.924`, `4K/128` `2293.994 /
114.991`, stable IDs `220/570`, `21.335 GiB`) is now secondary memory context,
not the active 1:1 llama.cpp comparison.

## Active GPU1 profile findings

G-M2 paired `rocprofv3` captures for the active Q4_K_M target were refreshed on
2026-06-17 with `decode_tokens=16`; prefill-only traces strip the prefill prefix
from decode traces. Raw CSV/logs live under
`/tmp/hipengine-gguf-q4km-profile-20260618-093656`; compact summaries are
`benchmarks/results/2026-06-17-gpu1-gguf-q4km-512-rocprof-summary.json` and
`benchmarks/results/2026-06-17-gpu1-gguf-q4km-4k-rocprof-summary.json`.

| Shape | Phase | Total kernel time | Top buckets |
| --- | --- | ---: | --- |
| 512/16 Q4_K_M | prefill | `218.615 ms` (`0.427 ms/token`) | dense Q8_0 WMMA `59.884 ms` (`27.39%`); GDN prefill recurrent `49.541 ms` (`22.66%`); selected dual Q4_K WMMA `42.095 ms` (`19.26%`); selected Q5_K WMMA `24.372 ms` (`11.15%`) |
| 512/16 Q4_K_M | decode | `122.273 ms` (`7.642 ms/token`) | dense Q8_0 T16 GEMV `46.532 ms` (`38.06%`); selected dual Q4_K T16 GEMV `16.375 ms` (`13.39%`); selected Q5_K T16 GEMV `12.988 ms` (`10.62%`); RMSNorm `8.786 ms` (`7.19%`); lm-head Q6 T16 `8.348 ms` (`6.83%`) |
| 4K/16 Q4_K_M | prefill | `1455.701 ms` (`0.355 ms/token`) | GDN prefill recurrent `374.015 ms` (`25.69%`); dense Q8_0 WMMA `327.706 ms` (`22.51%`); selected dual Q4_K WMMA `260.100 ms` (`17.87%`); selected Q5_K WMMA `161.865 ms` (`11.12%`); router `102.376 ms` (`7.03%`) |
| 4K/16 Q4_K_M | decode | `130.999 ms` (`8.187 ms/token`) | dense Q8_0 T16 GEMV `46.101 ms` (`35.19%`); full-attn decode `16.843 ms` (`12.86%`); selected dual Q4_K T16 GEMV `15.115 ms` (`11.54%`); selected Q5_K T16 GEMV `11.978 ms` (`9.14%`); lm-head Q6 T16 `8.434 ms` (`6.44%`) |

Historical Q4_K_S paired captures were taken initially on 2026-06-15, refreshed
for the retained-default 4K path on 2026-06-16, and refreshed again after the
G-P1 selected-WMMA half-sequential rewrite. They remain useful for lineage but
are no longer the active 1:1 llama.cpp comparison.

G-M3 resource census on the refreshed 4K/16 trace is recorded in
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-resource-census-diagnostic.json`;
the G-P1 half-seq artifact refreshes the selected-WMMA resource row after the
kernel rewrite. `qwen35_gguf_rocprof_summary.py` now emits
VGPR/SGPR/scratch/LDS/workgroup/grid summaries per bucket and per kernel.

| Phase | Bucket | Scratch | VGPR | LDS | Read |
| --- | --- | ---: | ---: | ---: | --- |
| prefill | selected dual Q4_K WMMA | `160 B` | `256` | `0` | half-seq rewrite cut spill footprint and moved the 4K/16 bucket `800.455 -> 454.370 ms`; still VGPR-capped, but no longer the dominant 4K prefill bucket. |
| prefill | dense Q8_0 WMMA | `0/8 B` | `96/128/192` | `0` | mostly scratch-free, with one tiny-spill shape; do tile-specific census before changing tile policy. |
| prefill | full-attn prefill | `3200 B` | `256` | `0` | AOTriton/full-attn spill exists but is only `~4%` of the 4K prefill trace. |
| decode | dense Q8_0 T16 GEMV | `0 B` | `64` | `512` | top decode bucket is scratch-free; blind launch-bound retuning has little evidence now. |
| decode | selected dual Q4_K T16 GEMV | `0 B` | `200` | `1024` | scratch-free but high VGPR; candidate only if ISA shows an easy pressure cut. |
| decode | Q6 lm-head T16 GEMV | `0 B` | `72` | `512` | scratch-free; prior Q6 d-load/lb variants stayed no-hold. |

G-P1 ISA audit for the selected dual Q4T16 WMMA prefill kernel is recorded in
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-selected-wmma-isa-audit.json`.
The pre-rewrite BF16 kernel metadata matched rocprof: `private_segment_fixed_size=676`,
`vgpr_count=256`, `vgpr_spill_count=320`, and `42` scratch loads / `80` scratch
stores in the disassembly. Building the same kernel with
`HIPENGINE_DISABLE_UNROLL600=1` produced identical metadata. The retained
half-seq rewrite computes/stores one 16-column half at a time while preserving
the original per-lane store bounds guard, reducing the metadata to
`private_segment_fixed_size=160`, `vgpr_spill_count=113`, and `57` scratch loads /
`33` scratch stores while preserving the same `32` WMMA ops.
Further selected-WMMA work should now target the remaining 256-VGPR cap or a
true 16-column code-object variant without giving back the prefill win.

G-D2 ISA/code-object audit for the active Q8_0 T16 GEMV decode family is
recorded in
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-decode-isa-audit.json`.
The active code object contains nine Q8T16 decode kernels and all are
scratch-free with no VGPR spills: `private_segment_fixed_size=0`,
`vgpr_count=56/57`, `sgpr_count=28/31/33/41`, and `group_segment_fixed_size=256 B`.
Disassembly found zero scratch ops. In the current half-seq 4K/16 decode profile,
the dense Q8_0 T16 bucket remains the largest decode bucket (`50.638 ms`,
`35.31%`, `2720` dispatches), but the pressure profile is already clean; do not
repeat blind Q8 launch-bound or scale-load tweaks without a new algorithmic or
dispatch-level reason.

G-M4 memory-census instrumentation is now emitted in GGUF benchmark memory
snapshots and summarized in
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-memory-census.json`. On the primary
GPU1 gate, owned resident memory remains `21.335 GiB`, split as `19.858 GiB`
weights/T16 tiles, `0.148 GiB` decode scratch, and `1.329 GiB` session buffers
(`1.295 GiB` bulk-prefill scratch plus `0.033 GiB` full-sequence prefill hidden
for the 4K capacity). A static projection matching the resident-session scratch
formulas put pre-policy `128K/128` at `25.108 GiB` (`19.858 GiB` weights,
`2.638 GiB` decode scratch including `2.505 GiB` full-attention KV, and
`2.611 GiB` session buffers including `1.608 GiB` bulk-prefill scratch and
`1.002 GiB` full-sequence prefill hidden). GPU1 reports `23.984 GiB` total, so
that blocker was a real `~1.12 GiB` capacity shortfall rather than hidden
raw+packed duplicate residency.

G-P4 128K memory policy/full gate is recorded in
`benchmarks/results/2026-06-16-gpu1-gguf-q4ks-128k-gate.json`. The 24GB
low-memory chunk policy now triggers at 128K-class contexts with 768-token
chunks for linear/MoE/full-attn query/post/RoPE (keeping AOTriton active); the
low-memory prefill path uses one full-sequence hidden buffer plus a chunk
staging buffer, and bulk-prefill dense block-table metadata is compacted to
scratch rows. GPU1 `128K/128` now runs successfully with `730.191` prefill
tok/s, `67.733` decode tok/s, stable final ID `[220]`, finite logits,
tracked/owned peak `23.310 GiB`, and sampled HIP-used `23.898 GiB`. This
replaces the pre-policy `25.108 GiB` OOM projection and improves the first
512-query low-memory gate by `+27.63%` prefill for `+0.082 GiB` tracked peak;
final long-context promotion is now gated by further speed, not construction
memory.

Retained notes:

- **Q8_0 T16 1024-row square-projection TN16 retained (2026-06-16).** The
  resident Q8_0 T16 WMMA prefill tile policy now uses `TM32/TN16` for the
  1024-row `in=2048,out=2048` medium-square projection shape only. GPU1 Q4_K_S
  primary gate moved versus the retained decode-policy-cache baseline to
  `512/128` `1965.587 / 127.250 tok/s` and `4K/128` `2293.367 / 115.168 tok/s`,
  stable IDs `220/570`, with tracked peak flat at `21.335 GiB`; the tiny
  `512/128` decode dip (`127.263 -> 127.250 tok/s`) is treated as run noise, not
  a decode regression. The required `128K/128` final check measured
  `741.581 / 67.636 tok/s`, stable ID `[220]`, and `23.310 GiB` tracked peak.
  Related 512/768-row square, large up-projection, and output-projection TN16
  probes are recorded as no-holds below. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-1024-square2048-tn16-gate.json`.

- **Q8_0 T16 small-shape prefill tiles retained (2026-06-16).** A follow-up
  resident Q8_0 T16 tile-policy pass uses `TN16` for the shared-expert
  `in=2048,out=512` and `in=512,out=2048` shapes only when `rows > 512`, while
  preserving the previous `TN32` policy for 512-row primary chunks. GPU1 Q4_K_S
  primary gate moved to `512/128` `1958.693 / 126.924 tok/s` and `4K/128`
  `2293.994 / 114.991 tok/s`, stable IDs `220/570`, with tracked peak unchanged
  at `21.335 GiB`. The required 128K check measured `747.033 / 67.592 tok/s`,
  stable ID `[220]`, and `23.310 GiB` tracked peak. A broader variant that also
  changed 512-row small-shape chunks regressed the `512/128` prefill median to
  `1909.506 tok/s`, so only the rows>512 policy is retained. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-smallshape-tiles-gate.json`.

- **Q8_0 T16 WMMA prefill tile policy retained (2026-06-16).** The resident
  Q8_0 T16 prefill wrapper now uses a T16-specific tile policy instead of
  inheriting the raw-Q8 heuristic: large `in<=2048,out>=4096` projections use
  `TM64` at 512-row chunks (and selected 768-row `out<8192` chunks) but `TM32`
  for 1024-row chunks, and `in>=4096,out>=2048` projections use `TM32` instead
  of `TM64`. GPU1 Q4_K_S primary gate moved to `512/128`
  `1951.832 / 127.138 tok/s` and `4K/128` `2278.421 / 114.736 tok/s`, stable
  IDs `220/570`, with tracked peak unchanged at `21.335 GiB`. The required
  128K check measured `747.085 / 67.728 tok/s`, stable ID `[220]`, and
  `23.310 GiB` tracked peak. This is retained for prefill; decode is not claimed
  as improved because the 4K decode median was lower than the immediately prior
  run but within observed same-session noise. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-prefill-tiles-gate.json`.

- **GDN single-segment `k2` threshold retained (2026-06-16).** The GGUF GDN
  prefill recurrent-segments default threshold is now `1025` instead of `256`,
  so primary 512/1024-row chunks and 128K low-memory 768-row chunks use the
  exact single-segment `k2` recurrent kernel by default. The primary gate moved
  to `512/128` `1837.509 / 126.699 tok/s` and `4K/128`
  `2212.932 / 115.335 tok/s`, with stable IDs `220/570` and unchanged
  `21.335 GiB` tracked peak. The required 128K final-promotion check measured
  `737.108 / 67.822 tok/s`, stable ID `[220]`, and `23.310 GiB` tracked peak.
  `segments_k2` remains available through
  `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` for larger or batched probes.
  Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-gdn-k2-th1025-gate.json`.

- **G-P1 selected-WMMA half-sequential rewrite retained (2026-06-16).** The
  selected dual Q4_K T16 WMMA prefill kernel now computes/stores each 16-column
  half sequentially with one accumulator instead of keeping two accumulators live
  across the K loop. Gate IDs stayed stable, tracked peak stayed `21.335 GiB`,
  focused CPU-reference tests passed (`9 passed`), and the targeted GGUF guard
  passed (`154 passed`). The retained safe gate moved `512/128` to
  `1816.758 / 126.473 tok/s` and `4K/128` to `2151.851 / 115.798 tok/s`; two
  intermediate same-command runs without the restored per-lane store guard
  measured higher prefill but were superseded. This is a prefill promotion, not a
  decode promotion: the 4K/16 rocprof selected-WMMA bucket fell
  `800.455 -> 454.370 ms` and total prefill kernel time fell
  `2094.198 -> 1761.552 ms`, while gate decode stayed within noise of the
  selected-down-lb2 row. The follow-up GPU1 `128K/128` single-run final check
  now fits and is deterministic; the latest all-768 low-memory gate measures
  `730.191 / 67.733 tok/s`, stable ID `[220]`, and `23.310 GiB` tracked peak.
  Long-context prefill remains slower than the stale W7900 README row, so future
  G-P4/P3 work should optimize the 24GB low-memory policy rather than memory
  construction.
- **G-P1 launch-bound default retained (2026-06-15).** Lowering
  `HIPENGINE_SELECTED_WMMA_LAUNCH_BOUNDS` from `2` to `1` in the selected Q4_K
  T16 WMMA prefill kernel kept gate IDs stable, kept tracked peak at
  `21.335 GiB`, and moved the GPU1 gate medians to `1658.695 / 126.334 tok/s`
  (`512/128`) and `1850.311 / 115.114 tok/s` (`4K/128`).
- **G-D3 selected dual+SiLU GEMV launch-bound retained (2026-06-15).** Relaxing
  `q4_k_t16_selected_dual_silu_direct_gemv_kernel` from
  `__launch_bounds__(256, 2)` to `__launch_bounds__(256, 1)` preserved gate IDs,
  kept tracked peak at `21.335 GiB`, and moved the GPU1 gate medians to
  `1644.664 / 126.993 tok/s` (`512/128`) and `1851.330 / 115.703 tok/s`
  (`4K/128`).
- **G-D3 selected down GEMV launch-bound retained (2026-06-15).** Relaxing
  `qk_t16_selected_direct_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 2)` preserved gate IDs, kept tracked peak at
  `21.335 GiB`, and moved the GPU1 gate medians to `1647.390 / 127.012 tok/s`
  (`512/128`) and `1855.806 / 115.805 tok/s` (`4K/128`). Final promotion still
  needs the `128K/128` memory/throughput check.
- **G-H1/G-D4 decode-policy cache retained (2026-06-16).** Resident GGUF
  `_FullStackScratch` now snapshots the full-attention split threshold plus
  grouped-GQA/warp-split decode policy at allocation, so decode full-attention
  layers avoid repeated env parsing while preserving the fake-scratch dynamic-env
  fallback used by dispatch tests. GPU1 primary gate with the retained
  small-shape tile baseline moved to `1958.536 / 127.263 tok/s` (`512/128`) and
  `2292.684 / 114.991 tok/s` (`4K/128`) with stable IDs `220/570` and flat
  `21.335 GiB`; the required `128K/128` promotion check measured
  `745.977 / 67.326 tok/s`, stable ID `[220]`, peak `23.310 GiB`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-decode-policy-cache-gate.json`.

Current focused lanes from evidence:

0. **Correctness branch fixed; resume from the restored Q4_K_M baseline.** The
   W7900 bisect showed `e089d1c2` was the first tested point with the large 4K
   prefill drop and the `220/570 -> 1813/151531` ID change on Q4_K_S; the Q4_K_M
   diagnostic similarly changed `318/220 -> 38118/1076`. The correctness fix
   restores the pre-e089 GGUF runner/materialization/selected-T16 code path
   (including Q8_0 T16 and dense Q4_K pack8 materialization), so Q4_K_M IDs are
   back to `318/220`. Future fused selected-expert, raw-Q8, or dense-Q4 raw
   memory work must be reintroduced as split candidates with a targeted oracle
   plus the full Q4_K_M gate; do not resurrect the bundled e089 path.
1. **llama.cpp prefill parity:** the 2026-06-16 local llama.cpp
   `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` bench on GPU1 reported HIP
   `pp512=2736.98 ± 53.51 tok/s`, `tg128=94.13 ± 1.29 tok/s`, and Vulkan
   `pp512=2389.53 ± 16.94 tok/s`, `tg128=79.90 ± 0.22 tok/s` with `-fa 1`.
   The correctness-good hipEngine Q4_K_M decode gate is still ahead of those
   llama.cpp tg128 rows, but llama.cpp HIP pp512 is far ahead of hipEngine
   prefill. Treat this as the main parity target after recovery, not as
   permission to trade away decode/memory.
2. **Decode work should move above blind Q8 launch-bound tweaks:** dense Q8_0
   T16 GEMV is still the largest decode bucket (`~35-38%`), but the active code
   object is scratch-free and low-VGPR and prior Q8 launch-bound/load variants
   no-held. Prefer dispatch/graph/full-attention or an algorithmic Q8 reduction
   over more occupancy pokes.
3. **G-P4/P3 long-context follow-up:** GPU1 `128K/128` now fits and completes,
   but low-memory throughput is prefill-limited (`730.191 tok/s` vs the stale
   W7900 diagnostic `995.295 tok/s`). Next targets are lower-overhead
   long-context prefill staging, full-attention prefill chunk/AOTriton tuning,
   or KV/cache residency reductions that permit larger chunks without regressing
   short gates.
4. **Secondary prefill checks:** after the half-seq rewrite, selected dual Q4_K
   WMMA is no longer the dominant 4K prefill bucket; GDN prefill recurrent and
   dense Q8_0 WMMA are now comparable follow-up targets, and any further G-P1
   live-state work must beat the remaining 256-VGPR cap without extra decode
   noise.

2026-06-21 return-plan update after the corrected W7900 final sweep:

- **Do not do more blind GGUF micro-tuning.** The apparent final-sweep prefill
  regression was a non-hermetic W7900 environment artifact, not a kernel change.
  Retained W7900 rows must use `scripts/run_w7900_readme_refresh.sh hipengine`
  or its exact `THEROCK_ENV`; if GGUF prefill falls while decode and IDs stay
  normal, rerun hermetically before blaming kernels.
- **GGUF INT8 KV correctness is now localized and guarded.** Short explicit
  `--kv-storage int8_per_token_head` sessions still use a BF16 mirror and match
  BF16 IDs/logits. Long sessions no longer use the rejected pure INT8-only route:
  by default they keep a 3-layer BF16 full-attention prefix, use INT8 for the
  remaining 7 full-attention layers, and promote scales to FP32. The forced-long
  W7900 `4K` BF16-vs-hybrid gate passes (`KL mean=0.014025`, top-1 `1.0`, no
  BF16 mirror). Continue with full `128K/128` capacity/throughput validation
  before promoting any 24 GiB benchmark row.
- **Refresh attribution before the next optimization pass.** Before touching more
  kernels, regenerate hermetic Q4_K_M `rocprofv3` bucket summaries for the
  current tree at `512`, `4K`, `32K`, and `128K`/largest-fitting context. Pick the
  next change from the largest current bucket, not from stale Q4_K_S traces or
  launch-count intuition.
- **Likely optimization lanes once INT8 correctness is understood:** prefill
  remains the more plausible c=1 target than decode (dense Q8_0 WMMA, GDN
  recurrent prefill, selected Q4/Q5 WMMA, and long-context full-attn/chunk
  policy); memory/residency work is still valuable but must be split into
  oracle-backed candidates because the bundled raw-Q8/fused-selected experiment
  caused ID drift; true c>N GGUF batching or MTP/spec decode may yield larger
  user-visible throughput than further c=1 decode micro-tuning.

llama.cpp HIP/Vulkan codepath detour (2026-06-16):

- Reviewed local llama.cpp source trees: HIP `~/llama.cpp/llama.cpp-hip`
  (`e37abd6b5fc91ba951d5b08ac7cdf2bc225512b6`) and Vulkan
  `~/llama.cpp/llama.cpp-vulkan`
  (`263cc04a5405fc55122bf59383dd8195519b30f4`). The user's `llama-bench`
  output reports build `263cc04a5`; verify the HIP binary/source SHA before
  copying exact code.
- HIP prefill reaches llama.cpp's quantized MMQ path for Q4_K on RDNA3:
  `ggml_cuda_mul_mat` selects `ggml_cuda_mul_mat_q` when `src1->ne[1]` is above
  the small-vector MMVQ range and `ggml_cuda_should_use_mmq()` returns true for
  RDNA3/WMMA Q4_K (`ggml/src/ggml-cuda/ggml-cuda.cu:2590-2668`,
  `ggml/src/ggml-cuda/mmq.cu:267-374`). That path quantizes the activation
  matrix to Q8_1 in the pool, tiles up to `mmq_x=128` and `mmq_y=128` on AMD
  WMMA, and maps Q4_K through the Q8_1 MMA tile shape
  (`mmq.cu:77-160`, `mmq.cuh:109-160`, `mmq.cuh:237-254`). The Q4_K loader
  repacks nibbles/scales into shared tiles, and Q4_K's MMQ trait uses
  `vec_dot_q8_1_q8_1_mma` for the matrix path while retaining a DP4A fallback
  (`mmq.cuh:2100-2237`, `mmq.cuh:3358-3363`). A focused GPU1 rocprof capture of
  llama.cpp HIP `Q4_K_M` pp512 confirms this is the hot path: the measured pass
  is `176.818 ms` of kernel time at `2746.089 tok/s`, with `mul_mat_q` Q4_K/Q5_K
  /Q8_0/Q6_K plus `quantize_mmq_q8_1` taking `72.95%` of kernel time, while
  flash attention is only `1.63%`
  (`benchmarks/results/2026-06-16-gpu1-llamacpp-hip-q4km-pp512-rocprof-diagnostic.json`).
  The first hipEngine-side baseline for that comparison is
  `scripts/gguf_q4_k_t16_selected_prefill_microbench.py`, which times the
  current selected-dual Q4_K T16 WMMA prefill kernel on a synthetic compact-MoE
  shape. Its qwen-like reduced-expert run (`hidden=2048`, gate/up `4096+4096`,
  `compact_rows=4096`) measured `11.585 ms/call` (`11.86` logical TFLOP/s) with
  finite output and `0.150 GiB` tracked scratch/fixture memory
  (`benchmarks/results/2026-06-16-gpu1-gguf-q4k-t16-selected-prefill-microbench.json`).
  A first `q8-1-dot` prototype now runs in that same harness using raw Q4_K
  weights and prequantized Q8_1 activations. It is intentionally scalar (not a
  tiled llama.cpp MMQ clone) and is slower on the same shape: `21.879 ms/call`
  (`6.28` logical TFLOP/s) vs a same-run selected-WMMA comparison of
  `11.933 ms/call` (`11.52` logical TFLOP/s), with finite output and lower
  tracked fixture memory (`0.143 GiB`). Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-selected-prefill-prototype.json`.
  Follow-up source audit of llama.cpp HIP commit
  `e37abd6b5fc91ba951d5b08ac7cdf2bc225512b6` shows why the scalar prototype
  missed the mark: Q4_K uses the DS4 Q8_1 activation layout (`block_q8_1_mmq` =
  128 int8 values plus four half2 scale/sum pairs) and the AMD WMMA/MMA path,
  not the DP4A Q4_K scalar loop. `load_tiles_q4_K` first stages each raw Q4_K
  weight tile into shared memory as low/high nibble int matrices plus half2
  scale/min products (`ggml/src/ggml-cuda/mmq.cuh:2093-2165`). The Q4_K trait
  then dispatches `vec_dot_q8_1_q8_1_mma` on AMD WMMA (`mmq.cuh:3358-3363`),
  where 16x8 int tiles are loaded with `load_ldmatrix`, accumulated as a 16x16
  int tile, and converted with both Q4_K `dm` and Q8_1 `ds` terms
  (`mmq.cuh:1330-1380`). The outer kernel stages `mmq_x` activation columns and
  `mmq_y` weight rows in shared memory, processes two 128-value Q8_1 activation
  blocks per 256-wide Q4_K block, and writes back a full tile
  (`mmq.cuh:3447-3518`). On RDNA3, `mmq_y=128`; `mmq_x` is chosen up to 128 by
  shared-memory fit and output column tiling (`mmq.cuh:3943-4138`). A follow-up
  `q8-1-ds4-dot` diagnostic now feeds the same scalar raw-Q4_K dot loop with the
  real DS4 `block_q8_1_mmq` activation layout. Layout alone did not help: on
  the same qwen-like reduced-expert shape, same-session runs measured
  `q8-1-ds4-dot` at `22.497 ms/call` (`6.11` logical TFLOP/s), `q8-1-dot` at
  `21.727 ms/call`, and selected-WMMA at `11.951 ms/call`; the DS4 scalar path
  is therefore ~`1.88x` slower than selected-WMMA and ~`3.5%` slower than the
  separate-array scalar prototype, with finite output and `0.142 GiB` tracked
  fixture memory. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-selected-prefill-prototype.json`.
  A host-side `GGUFQ4KMMQTile16Preview` scaffold now centralizes DS4 activation
  packing plus the 16-column Q4_K nibble/scale/min operands and has an exact
  CPU oracle test that reconstructs raw Q4_K values. A tiny RDNA3
  `wmma_i8_probe_16x16` kernel now verifies the key
  `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` fragment/store mapping against
  CPU int32 matmul and has a rocprof kernel-trace smoke. The first DS4
  integer-WMMA selected-prefill prototype (`q8-1-ds4-wmma`) uses that mapping
  over raw Q4_K nibbles and measures `8.233 ms/call` (`16.69` logical TFLOP/s)
  on the synthetic qwen-like shape, faster than same-script selected-WMMA
  `11.581 ms/call` and DS4 scalar `21.842 ms/call`. A two-wave/32-column block
  variant (`q8-1-ds4-wmma32`) nudges this to `8.191 ms/call` (`16.78` logical
  TFLOP/s), only `+0.7%` over the one-wave variant. A four-wave/64-column
  variant (`q8-1-ds4-wmma64`) measured `8.216 ms/call` in a same-session run vs
  WMMA32 `8.268 ms/call`, so independent-wave output widening has only flat /
  sub-1% returns after 32 columns. A naive expanded-Q4 LDS staging variant
  (`q8-1-ds4-wmma32-lds`) regressed to `18.257 ms/call`, `2.22x`
  slower than raw WMMA32 and slower than current selected-WMMA, so do **not**
  promote that staging shape. A packed-Q4 LDS staging variant
  (`q8-1-ds4-wmma32-ldspack`) recovers much of that loss at `11.438 ms/call`
  and is slightly faster than selected-WMMA on the same synthetic run, but it is
  still `1.38x` slower than raw WMMA32. A pre-unpacked host-preview variant
  (`q8-1-ds4-preview-wmma32`) feeds the two-wave WMMA kernel from
  `GGUFQ4KMMQTile16Preview` q4/scales/mins arrays and measured
  `12.020 ms/call` (`11.43` logical TFLOP/s), slower than raw WMMA32
  (`8.209 ms/call`) and slightly slower than selected-WMMA (`11.584 ms/call`)
  while raising synthetic fixture memory to `0.228 GiB`; Q4_K metadata decode is
  therefore not the standalone bottleneck. A GPU BF16→DS4 Q8_1 activation pack
  diagnostic now measures the missing quantization cost for a runtime version:
  `q8-1-ds4-wmma32-pack` (pack + raw WMMA32 in the timed loop) measured
  `8.391 ms/call` (`16.38` logical TFLOP/s) versus same-run raw WMMA32
  `8.261 ms/call`, WMMA64 `8.229 ms/call`, selected-WMMA `11.579 ms/call`, and
  DS4 scalar `21.826 ms/call`. rocprof saw pack launches at roughly
  `18-22 us` on the smaller rows-per-expert=64 trace. Activation packing is
  therefore not the immediate blocker. A resident-layout follow-up
  (`q8-1-ds4-t16-wmma32`) consumes the existing `gguf_q4_k_t16_v1` tiles instead
  of raw GGUF Q4_K weights and measured `7.764 ms/call` (`17.70` logical
  TFLOP/s) versus same-run raw WMMA32 `8.224 ms/call` and selected-WMMA
  `11.566 ms/call`; with GPU BF16→DS4 packing in the timed loop
  (`q8-1-ds4-t16-wmma32-pack`) it measured `7.810 ms/call` (`17.60` logical
  TFLOP/s). This is the first DS4/MMQ diagnostic that both beats raw WMMA32 and
  fits the no-raw-duplicate resident layout story. A guarded runtime follow-up
  now wires the same resident-T16 DS4 path behind
  `HIPENGINE_GGUF_T16_DS4_PREFILL=1`: on the full Q4_K_S GPU1 gate it improves
  same-iteration median prefill from `1833.185 -> 1989.578 tok/s` at `512/128`
  and `2159.561 -> 2372.228 tok/s` at `4K/128`, with decode flat/in-noise and
  deterministic opt-in IDs. It is **not promoted** because the opt-in final
  token IDs change versus default (`220/570 -> 3241/1510`) and the extra DS4
  activation scratch raises opt-in peak by `+0.070 GiB`. The flag allocates that
  scratch only when enabled, so default-path verify remains at `21.335 GiB` with
  default IDs `220/570`.
  Artifacts:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-wmma-selected-prefill-prototype.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-wmma32-selected-prefill-prototype.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-wmma64-selected-prefill-probe.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-wmma32-lds-selected-prefill-probe.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-wmma32-ldspack-selected-prefill-probe.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-preview-wmma32-selected-prefill-probe.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-ds4-pack-wmma32-selected-prefill-probe.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4k-q8-1-t16-ds4-wmma32-selected-prefill-probe.json`,
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-t16-ds4-runtime-probe.json`.
  **Next code test:** stop adding raw-Q4_K same-shape staging variants. The
  resident-T16 DS4 runtime route has speed headroom but fails exact-ID
  promotion, so either investigate an exact-enough Q8_1/DS4 calibration path
  before any default promotion, or switch focus to the now-comparable exact GDN
  prefill recurrent / dense Q8_0 WMMA buckets.
- HIP decode/MoE fusion is a lower-priority but useful reference: llama.cpp has
  explicit graph fusions for top-k MoE and for `MUL_MAT(_ID)+GLU`/bias patterns,
  then launches fused `mul_mat_vec_q` only for `ncols_dst=1`
  (`ggml-cuda.cu:3447-4162`, `ggml-cuda/mmvq.cu:475-672`). hipEngine decode is
  already faster on the current gate, so use this mainly to audit launch count
  and fusion coverage, not as the first prefill fix.
- HIP `-fa 1` routes through `ggml_cuda_flash_attn_ext`; on RDNA3 it chooses
  between vector/tile/WMMA/MMA kernels by head size, GQA applicability, and
  `Q->ne[1] * gqa_ratio_eff`, with an AMD WMMA branch for head sizes up to 128
  when the effective query batch is large enough (`ggml-cuda/fattn.cu:332-596`).
  Because llama.cpp's pp512 lead is visible at short context, first profile its
  prefill kernel-family breakdown before assuming attention is the limiter.
- Vulkan provides a second sanity oracle: before falling back to dequant+f16
  matmul, `ggml_vk_mul_mat_q_f16` tries to quantize F32 activations to Q8_1 and
  use `pipeline_dequant_mul_mat_mat_q8_1` when integer dot support is available
  (`ggml-vulkan.cpp:6848-6917`, `8279-8387`). Its Q4_K MMQ shader packs Q4_K
  quants/scales into a shared-cache form and uses `dotPacked4x8EXT`, with
  K-quant warptiles deliberately reducing `WMITER` to `1` to contain register
  pressure (`vulkan-shaders/mul_mmq_funcs.glsl:303-374`,
  `ggml-vulkan.cpp:3448-3496`, `3651-3753`, `4318-4332`). **Next test:** run a
  rocprofv3/RGP-equivalent llama.cpp HIP pp512 capture and a Vulkan pipeline log,
  then compare launch families against hipEngine's current `512/128` profile.
- Apples-to-apples before adopting: the user benchmark is `Q4_K_M`, so the
  active loop gates must also be `Q4_K_M`. The current correctness-restored
  Q4_K_M GPU1 baseline is
  `benchmarks/results/2026-06-17-gpu1-gguf-q4km-correctness-restored.json`:
  `512/128` `2307.515` prefill / `125.095` decode tok/s with stable ID `[318]`,
  and `4K/128` `2599.357` / `114.195` with stable ID `[220]`, tracked peak
  `22.487 GiB`, sampled HIP used `~23.04 GiB`. This supersedes the invalid
  current-main diagnostic that changed IDs to `38118/1076`. Against the user's
  same-model llama.cpp row, hipEngine Q4_K_M pp512 is still behind llama.cpp HIP
  by `18.6%` and Vulkan by `3.6%`, while decode remains faster (`+32.9%` vs
  llama.cpp HIP tg128, `+56.6%` vs Vulkan tg128). The immediate 1:1 parity gap
  is short prefill.

No-hold notes:

- **GPU0/W7900 prefill regression bisect (2026-06-17).** The first current
  W7900 spot-check without the isolated README script env exaggerated the drop;
  rerunning under the same TheRock env style as `scripts/run_w7900_readme_refresh.sh`
  and forcing `max_sequence_length=131202` shows two effects. The G-P4 chunk-outer
  memory fix (`4c0a2521`) saved about `1.04 GiB` and moved W7900 prefill from
  `2286.082 -> 2181.710 tok/s` (`512/128`, `-4.6%`) and `2532.888 ->
  2479.558 tok/s` (`4K/128`, `-2.1%`) while keeping IDs `220/570`. The later
  `e089d1c2` bundle (`GGUF MoE fused activate+down and drop Q8_0 T16 repack`)
  is the first tested commit where IDs changed to `1813/151531` and where 4K
  prefill fell sharply (`2479.133 -> 2250.151 tok/s`, `-9.2%`). A temporary
  e089 patch restoring Q8_0 T16 materialization recovered prefill to
  `2218.379/2485.416 tok/s` but did not restore IDs, so Q8_0 raw-layout dispatch
  is the main prefill culprit and another e089 runtime/kernel change is the
  correctness culprit. Treat current W7900 rows as diagnostic until e089 is split
  or fixed. Artifacts:
  `benchmarks/results/2026-06-17-gpu0-w7900-gguf-q4ks-prefill-regression-bisect.json`,
  `benchmarks/results/2026-06-17-gpu0-w7900-gguf-q4ks-current-512-4k-diagnostic.json`.

- **G-D5 lm-head argmax fusion rejected (2026-06-17).** Reapplied the
  quarantined Q6T16 lm-head GEMV+argmax fusion to clean `main`, fixed wrapper
  validation/allocation issues, and added a focused synthetic Q6T16 argmax-vs-
  GEMV oracle. Focused Q6/dispatch tests passed (`21` tests), but the full
  GPU1 Q4_K_S primary gate changed deterministic final IDs (`512/128` `220 ->
  1813`, `4K/128` `570 -> 151531`) and regressed the primary min decode metric
  to `114.591 tok/s` versus the current `114.670 tok/s`. Code/test changes were
  reverted; keep only the rejected artifact. The configured `154`-test guard
  also currently segfaults in `test_t16_weights_route_direct_selected_tiles_allocations`
  at the selected `silu_x` kernel; the same segfault reproduced on clean
  `e5a63784`, so it is pre-existing relative to G-D5 but still blocks retention.
  Artifact:
  `benchmarks/results/2026-06-17-gpu1-gguf-q4ks-q6t16-lmhead-argmax-rejected.json`.

- **GGUF chunk tune min=513 rejected (2026-06-16).** Lowering the auto
  chunk-tuning minimum from `1025` to `513` max-sequence tokens made
  `512/128`-class sessions resolve the retained `1024/4096` mid-context chunk
  policy instead of staying below-min/unchunked. The focused config smoke and
  `154`-test GGUF guard passed with stable IDs and flat memory, but the
  primary gate regressed versus the retained decode-policy-cache baseline:
  `512/128` prefill `1958.536 -> 1881.896 tok/s`, decode `127.263 ->
  127.104 tok/s`; `4K/128` prefill nudged up `2292.684 -> 2296.736 tok/s` but
  decode regressed `114.991 -> 114.736 tok/s`. Code/test changes were
  reverted; keep the chunk-tune minimum at `1025`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-prefill-chunk-min513-rejected.json`.

- **GGUF linear-attention chunk=512 rejected (2026-06-16).** Lowering only the
  auto-tuned mid-context linear-attention prefill chunk from `1024` to `512`
  rows passed the prefill-config smoke and `154`-test GGUF guard with stable IDs
  and flat memory, but it regressed prefill sharply on both retained gates
  (`512/128` `1958.536 -> 1844.457 tok/s`, `4K/128` `2292.684 -> 2131.252
  tok/s`) while decode was flat/small mixed. Code/test changes were reverted;
  keep the mid-context linear-attention chunk at `1024`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-linear-chunk512-rejected.json`.

- **GGUF MoE chunk=512 rejected (2026-06-16).** Lowering only the auto-tuned
  mid-context MoE prefill chunk from `1024` to `512` rows, while leaving linear
  attention at `1024` and full-attention query/post/RoPE at `4096/1024/1024`,
  passed the prefill-config smoke and `154`-test GGUF guard with stable IDs and
  flat memory. It regressed primary prefill sharply versus the retained
  decode-policy-cache baseline (`512/128` `1958.536 -> 1892.253 tok/s`,
  `4K/128` `2292.684 -> 2121.682 tok/s`), so code/test changes were reverted.
  Keep the mid-context MoE chunk at `1024`; both smaller and larger MoE-only
  probes are now no-hold. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-moe-chunk512-rejected.json`.

- **GGUF linear dispatch resolve cache rejected (2026-06-16).** Adding an
  `lru_cache`-backed base-dispatch resolver passed focused dispatch tests and
  the `154`-test GGUF guard with stable IDs and flat memory, but measured gate
  movement was mixed/noisy: `4K/128` decode improved (`114.991 -> 115.253
  tok/s`) while `512/128` regressed (`1958.536 / 127.263 -> 1953.288 / 126.952
  tok/s`). Code/test changes were reverted; do not add dispatch-object caching
  without a Python-profiled host bottleneck. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-dispatch-resolve-cache-rejected.json`.

- **Q8_0 T16 row512 out8192 TM32 rejected (2026-06-16).** Changing resident
  Q8_0 T16 WMMA prefill for 512-row `in<=2048,out>=8192` projections from
  `TM64/TN32` to `TM32/TN32` passed the focused tile-policy smoke and `154`-test
  GGUF guard with stable IDs and flat memory. It improved the `4K/128` gate
  (`2292.684 / 114.991 -> 2298.061 / 115.309 tok/s`) but regressed the short
  gate beyond noise (`512/128` `1958.536 / 127.263 -> 1937.572 / 126.933
  tok/s`), so code/test changes were reverted. Keep 512-row out8192-like Q8_0
  T16 projections at `TM64`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-out8192-row512-tm32-rejected.json`.

- **Q6_K T16 lm-head block256 rejected (2026-06-16).** Changing dense Q6_K
  T16 lm-head GEMV decode from `__launch_bounds__(128,4)` / `blockDim=128` to
  `__launch_bounds__(256,2)` / `blockDim=256` passed the Q6 correctness suite
  and `154`-test GGUF guard with stable IDs and flat memory, but it regressed
  decode sharply on both retained gates (`512/128` `127.263 -> 125.297 tok/s`,
  `4K/128` `114.991 -> 113.797 tok/s`) and also lowered `512/128` prefill. Code
  was reverted; keep Q6_K T16 decode at `128,4` / `blockDim=128`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q6t16-block256-rejected.json`.

- **Q8_0 T16 large-input TM16 rejected (2026-06-16).** Changing the resident
  Q8_0 T16 WMMA prefill tile policy for `in>=4096,out>=2048` projections from
  `TM32/TN32` to `TM16/TN32` passed the focused tile-policy smoke and `154`-test
  GGUF guard with stable IDs and flat memory, but it regressed both retained
  gates (`512/128` `1958.536 / 127.263 -> 1938.859 / 127.166 tok/s`, `4K/128`
  `2292.684 / 114.991 -> 2270.846 / 114.832 tok/s`). Code and test changes
  were reverted; keep the large-input Q8_0 T16 default at `TM32` rather than
  under-tiling it. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-largein-tm16-rejected.json`.

- **Selected Q4_K WMMA default tile 16x16 rejected (2026-06-16).** Changing the
  selected dual Q4_K WMMA compact prefill default from `32x16` to `16x16`
  passed the focused tile smoke and `154`-test GGUF guard with stable IDs and
  flat memory, but the full gate was mixed/regressive versus the decode-policy
  cache baseline (`512/128` `1958.536 / 127.263 -> 1965.114 / 126.910 tok/s`,
  `4K/128` `2292.684 / 114.991 -> 2288.047 / 114.999 tok/s`). Code and test
  changes were reverted; keep the selected dual Q4_K default at `32x16` unless a
  code-object/ISA variant reduces the remaining VGPR cap without the gate tradeoff.
  Artifact: `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q4selected-tile16-rejected.json`.

- **Q8_0 T16 768-row out8192 TM64 rejected (2026-06-16).** Isolating the
  resident Q8_0 T16 WMMA prefill policy so only 768-row `in<=2048,out>=8192`
  projections use `TM64/TN32` (leaving the 512-row and 1024-row out8192 rules
  unchanged) passed the focused tile-policy smoke, primary gate stability, the
  `154`-test GGUF guard, and a `128K/128` memory check. It was not retained
  because the targeted 128K low-memory prefill path regressed versus the latest
  retained references (`745.977/747.033 -> 739.646 tok/s`), and `512/128` decode
  was not clearly non-regressive versus the decode-policy-cache baseline
  (`127.263 -> 126.941 tok/s`). Code/test changes were reverted; keep 768-row
  out8192-like Q8_0 T16 projections at `TM32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-row768-out8192-tm64-rejected.json`.

- **Q8_0 T16 768/1024-row out8192 TM64 rejected (2026-06-16).** Extending the
  resident Q8_0 T16 WMMA prefill tile policy so 768/1024-row
  `in<=2048,out>=8192` projections use `TM64/TN32` instead of `TM32/TN32`
  passed the focused tile-policy smoke and `154`-test GGUF guard with stable IDs
  and flat memory, but it regressed both retained gates (`512/128`
  `1958.536 / 127.263 -> 1952.566 / 126.868 tok/s`, `4K/128` `2292.684 /
  114.991 -> 2283.735 / 114.867 tok/s`). Code and test changes were reverted;
  keep the out8192 large-projection rule at `TM32` for rows above 512. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-out8192-tm64-rejected.json`.

- **Q8_0 T16 1024-row out8192 TN16 rejected (2026-06-16).** Narrowing only the
  N tile for resident Q8_0 T16 WMMA prefill 1024-row `in=2048,out=8192`
  projections from `TM32/TN32` to `TM32/TN16` passed the focused tile-policy
  smoke and `154`-test GGUF guard with stable IDs and flat memory, but regressed
  both primary prefill medians versus the retained 1024-square TN16 baseline
  (`512/128` `1965.587 -> 1949.431 tok/s`, `4K/128` `2293.367 -> 2255.344
  tok/s`) and regressed `4K/128` decode (`115.168 -> 114.689 tok/s`). Code/test
  changes were reverted; keep 1024-row out8192 projections at `TM32/TN32`.
  Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-1024-out8192-tn16-rejected.json`.

- **Q8_0 T16 768-row square-projection TN16 rejected (2026-06-16).**
  Narrowing only the N tile for resident Q8_0 T16 WMMA prefill 768-row
  `in=2048,out=2048` projections from `TM32/TN32` to `TM32/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory. It regressed `512/128` versus the retained decode-policy-cache
  baseline (`1958.536 / 127.263 -> 1956.257 / 127.048 tok/s`) and slightly
  regressed `4K/128` decode (`114.991 -> 114.983 tok/s`); the small `4K/128`
  prefill nudge (`2292.684 -> 2293.643 tok/s`) was not enough to retain it.
  Code/test changes were reverted; keep this medium-square rule at
  `TM32/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-768-square2048-tn16-rejected.json`.

- **Q8_0 T16 512-row square-projection TN16 rejected (2026-06-16).**
  Narrowing only the N tile for resident Q8_0 T16 WMMA prefill 512-row
  `in=2048,out=2048` projections from `TM32/TN32` to `TM32/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory, but regressed both primary gates versus the retained
  decode-policy-cache baseline (`512/128` `1958.536 / 127.263 -> 1890.102 /
  126.833 tok/s`; `4K/128` `2292.684 / 114.991 -> 2290.779 / 114.878 tok/s`).
  Code/test changes were reverted; keep this medium-square rule at
  `TM32/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-512-square2048-tn16-rejected.json`.

- **Q8_0 T16 512-row output-projection TN16 rejected (2026-06-16).**
  Narrowing only the N tile for resident Q8_0 T16 WMMA prefill 512-row
  `in>=4096,out=2048` projections from `TM32/TN32` to `TM32/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory, but regressed `512/128` prefill sharply and `4K/128` prefill as well
  versus the retained decode-policy-cache baseline (`512/128` `1958.536 /
  127.263 -> 1895.145 / 127.042 tok/s`; `4K/128` `2292.684 / 114.991 ->
  2283.901 / 115.130 tok/s`). The `4K/128` decode nudge alone was not enough
  to retain it. Code/test changes were reverted; keep this output-projection
  rule at `TM32/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-512-outproj-tn16-rejected.json`.

- **Q8_0 T16 1024-row output-projection TN16 rejected (2026-06-16).**
  Narrowing only the N tile for resident Q8_0 T16 WMMA prefill 1024-row
  `in>=4096,out=2048` projections from `TM32/TN32` to `TM32/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory, but regressed both primary gates versus the retained
  decode-policy-cache baseline (`512/128` `1958.536 / 127.263 -> 1950.269 /
  126.998 tok/s`; `4K/128` `2292.684 / 114.991 -> 2285.292 / 114.948 tok/s`).
  Code/test changes were reverted; keep this output-projection rule at
  `TM32/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-1024-outproj-tn16-rejected.json`.

- **Q8_0 T16 768-row output-projection TN16 rejected (2026-06-16).** Narrowing
  only the N tile for resident Q8_0 T16 WMMA prefill 768-row
  `in>=4096,out=2048` projections from `TM32/TN32` to `TM32/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory. It regressed `512/128` versus the retained decode-policy-cache
  baseline (`1958.536 / 127.263 -> 1947.156 / 127.086 tok/s`) and regressed
  `4K/128` decode (`114.991 -> 114.972 tok/s`); the small `4K/128` prefill
  nudge (`2292.684 -> 2295.285 tok/s`) was not enough to retain it. Code/test
  changes were reverted; keep this output-projection rule at `TM32/TN32`.
  Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-768-outproj-tn16-rejected.json`.

- **Q8_0 T16 768-row up-projection TN16 rejected (2026-06-16).** Narrowing
  only the N tile for resident Q8_0 T16 WMMA prefill 768-row
  `in<=2048,out=4096` projections from `TM64/TN32` to `TM64/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory. It still regressed `512/128` versus the retained decode-policy-cache
  baseline (`1958.536 / 127.263 -> 1951.756 / 127.153 tok/s`); the small
  `4K/128` nudge (`2292.684 / 114.991 -> 2294.546 / 115.009 tok/s`) was not
  enough to retain it. Code/test changes were reverted; keep the 768-row
  up-projection rule at `TM64/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-768-out4096-tn16-rejected.json`.

- **Q8_0 T16 512-row up-projection TN16 rejected (2026-06-16).** Narrowing
  only the N tile for resident Q8_0 T16 WMMA prefill 512-row
  `in<=2048,out=4096` projections from `TM64/TN32` to `TM64/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory, but regressed both primary prefill medians versus the retained
  decode-policy-cache baseline (`512/128` `1958.536 -> 1931.444 tok/s`,
  `4K/128` `2292.684 -> 2288.636 tok/s`) and regressed `512/128` decode
  (`127.263 -> 127.115 tok/s`). Code/test changes were reverted; keep the
  512-row up-projection rule at `TM64/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-512-out4096-tn16-rejected.json`.

- **Q8_0 T16 1024-row up-projection TN16 rejected (2026-06-16).** Narrowing
  only the N tile for resident Q8_0 T16 WMMA prefill 1024-row
  `in<=2048,out=4096` projections from `TM32/TN32` to `TM32/TN16` passed the
  focused tile-policy smoke and `154`-test GGUF guard with stable IDs and flat
  memory, but regressed both primary prefill medians versus the retained
  decode-policy-cache baseline (`512/128` `1958.536 -> 1950.337 tok/s`,
  `4K/128` `2292.684 -> 2273.690 tok/s`) and regressed `512/128` decode
  (`127.263 -> 127.073 tok/s`). Code/test changes were reverted; keep the
  1024-row up-projection rule at `TM32/TN32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-1024-out4096-tn16-rejected.json`.

- **Q8_0 T16 1024-row up-projection TM16 rejected (2026-06-16).** Narrowing
  only resident Q8_0 T16 WMMA prefill 1024-row `in<=2048,out=4096`
  projections from `TM32/TN32` to `TM16/TN32` passed the focused tile-policy
  smoke and `154`-test GGUF guard with stable IDs and flat memory, but regressed
  both primary prefill medians versus the retained decode-policy-cache baseline
  (`512/128` `1958.536 -> 1933.731 tok/s`, `4K/128` `2292.684 -> 2272.399
  tok/s`). Code and test changes were reverted; keep the 1024-row up-projection
  rule at `TM32`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-1024-out4096-tm16-rejected.json`.

- **Q8_0 T16 1024-row up-projection TM64 rejected (2026-06-16).** Extending
  the resident Q8_0 T16 WMMA prefill tile policy so 1024-row
  `in<=2048,out=4096` projections use `TM64/TN32` instead of `TM32/TN32`
  passed the focused tile-policy smoke and `154`-test GGUF guard with stable IDs
  and flat memory, but it regressed prefill versus the decode-policy cache
  baseline (`512/128` `1958.536 / 127.263 -> 1921.592 / 127.022 tok/s`,
  `4K/128` `2292.684 / 114.991 -> 2278.406 / 115.087 tok/s`). Code and test
  changes were reverted; keep the 1024-row large-projection rule at `TM32`.
  Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-1024-up-tm64-rejected.json`.

- **GDN segment-threshold runner cache rejected (2026-06-16).** Caching
  `HIPENGINE_GGUF_GDN_PREFILL_SEGMENT_THRESHOLD` once on the full-stack runner
  instead of reparsing it in every GDN prefill chunk passed the focused routing
  smoke and `154`-test GGUF guard with stable IDs and flat memory. It was not
  retained because the primary gate regressed prefill versus the decode-policy
  cache baseline (`512/128` `1958.536 / 127.263 -> 1930.867 / 127.050 tok/s`,
  `4K/128` `2292.684 / 114.991 -> 2285.353 / 115.044 tok/s`). Code and test
  changes were reverted; keep the dynamic helper until a host profile shows this
  parse is material. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-gdn-threshold-cache-rejected.json`.

- **GDN segment threshold=769 rejected (2026-06-16).** Lowering the GGUF GDN
  prefill recurrent-segments threshold from `1025` to `769` so 1024-row chunks
  use `segments_k2` passed the focused routing smoke and `154`-test GGUF guard
  with stable IDs and flat memory, but it regressed prefill versus the retained
  decode-policy cache baseline (`512/128` `1958.536 / 127.263 -> 1919.168 /
  127.082 tok/s`, `4K/128` `2292.684 / 114.991 -> 2236.029 / 114.977 tok/s`).
  Code and test changes were reverted; keep the default threshold at `1025`.
  Artifact: `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-gdn-th769-rejected.json`.

- **README sweep graph steps/replay=2 rejected (2026-06-16).** Changing
  `scripts/qwen35_readme_sweep.py` default `--graph-steps-per-replay` from `1`
  to `2` passed the CLI divisibility smoke and `154`-test GGUF guard, but it
  changed the deterministic `512/128` final token (`220 -> 148536`) and regressed
  decode versus the retained decode-policy cache baseline (`512/128` `127.263 ->
  126.838 tok/s`, `4K/128` `114.991 -> 114.693 tok/s`). Keep the README sweep
  default at one step per replay until GGUF decode graph capture stops baking
  fixed position/context scalar arguments. Code was reverted. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-readme-graphsteps2-rejected.json`.

- **Full-attention post/RoPE chunk=512 rejected (2026-06-16).** Lowering only
  the mid-context full-attention post/RoPE prefill chunks from `1024` to `512`,
  while keeping query chunks at `4096` and linear/MoE chunks at `1024`, passed
  the prefill-config smoke and `154`-test GGUF guard with stable IDs and flat
  memory. It was not retained because the short gate regressed versus the
  retained decode-policy-cache baseline (`512/128` `1958.536 / 127.263 ->
  1917.323 / 127.128 tok/s`) while the `4K/128` prefill nudge was tiny/noisy.
  Code/test changes were reverted; keep mid-context post/RoPE chunks at `1024`.
  Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-fullattn-postrope512-rejected.json`.

- **Full-attention post/RoPE chunk=2048 rejected (2026-06-16).** Raising the
  mid-context full-attention post/RoPE prefill chunks from `1024` to `2048`
  while keeping query chunks at `4096` passed the config smoke and `154`-test
  GGUF guard with stable IDs and flat memory, but it regressed the retained
  decode-policy cache baseline (`512/128` `1958.536 / 127.263 -> 1884.846 /
  127.043 tok/s`, `4K/128` `2292.684 / 114.991 -> 2294.975 / 114.711 tok/s`).
  Code and test changes were reverted. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-fullattn-postrope2048-rejected.json`.

- **Full-attention KV pair cache rejected (2026-06-16).** Adding cached
  per-layer full-attention `(key_cache, value_cache)` tuples to
  `_FullStackScratch` passed the focused full-cache/routing smoke and `154`-test
  GGUF guard with stable IDs and flat memory, but it regressed the retained
  decode-policy cache baseline (`512/128` `1958.536 / 127.263 -> 1913.472 /
  127.079 tok/s`, `4K/128` `2292.684 / 114.991 -> 2288.777 / 114.916 tok/s`).
  Code and test changes were reverted. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-full-cache-pair-rejected.json`.

- **Full-attention scratch library handles rejected (2026-06-16).** Moving
  cached cast, paged-KV-write, and paged-attention library handles from the
  runner accessors onto `_FullStackScratch` passed the focused routing smoke and
  `154`-test GGUF guard with stable IDs and flat memory, but it regressed the
  retained decode-policy cache baseline (`512/128` `1958.536 / 127.263 ->
  1901.611 / 127.149 tok/s`, `4K/128` `2292.684 / 114.991 -> 2288.138 /
  114.695 tok/s`). Code and test changes were reverted. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-scratch-library-cache-rejected.json`.

- **Full-attention cached Qwen-GQA shape rejected (2026-06-16).** Caching the
  Qwen full-attention GQA-shape boolean on `_FullStackScratch` passed the
  focused routing smoke and the `154`-test GGUF guard, with stable IDs and flat
  memory, but it regressed the `512/128` prefill median versus the retained
  decode-policy cache baseline (`1958.536 -> 1854.538 tok/s`). The small 4K
  decode nudge (`114.991 -> 115.089 tok/s`) was not enough to retain it. Code
  and test changes were reverted. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-cached-gqa-shape-rejected.json`.

- **Q8_0 T16 decode block64 rejected (2026-06-16).** Reducing all resident
  Q8_0 T16 GEMV decode kernels from `blockDim/launch_bounds=128` to `64` passed
  unit tests but changed deterministic generated IDs on both primary gates
  (`512/128` `220 -> 97799`, `4K/128` `570 -> 28944`) and regressed decode
  (`126.924 -> 122.694 tok/s`, `114.991 -> 111.998 tok/s`). Code was reverted;
  do not repeat Q8 decode block-size pokes without a reduction-order fix and a
  correctness oracle. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-decode-block64-rejected.json`.

- **Full-attention query chunk 8192 rejected (2026-06-16).** Raising only the
  mid-context full-attention query prefill chunk from `4096` to `8192` passed
  the prefill-config smoke and `154`-test GGUF guard with stable IDs, but it was
  not neutral: tracked peak grew from `21.335` to `21.416 GiB` and `512/128`
  regressed versus the retained decode-policy-cache baseline (`1958.536 /
  127.263 -> 1952.309 / 127.068 tok/s`). The small `4K/128` nudge was not enough
  to retain it. Code/test changes were reverted; keep the mid-context query
  chunk at `4096`. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-fullattn-query8192-rejected.json`.

- **Full-attention query chunk 2048 rejected (2026-06-16).** Splitting the
  mid-context auto-tuned full-attention query chunk from `4096` to `2048`
  reduced tracked primary-gate peak (`21.335 -> 20.689 GiB`) and nudged decode
  up, but it changed the deterministic `4K/128` final token from `570` to `15`
  and regressed 4K prefill (`2293.994 -> 2233.790 tok/s`). Code was reverted;
  keep the retained `4096` full-attention query chunk unless the chunked full-
  attention equivalence bug is fixed and covered by a gate. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-fullattn-query2048-rejected.json`.

- **Q8_0 T16 4K shared-gate/up TM64 rejected (2026-06-16).** A 4K-oriented
  resident Q8_0 T16 tile override changed shared-expert gate/up shapes
  (`out<=512`, rows `>=2048`) from `TM32/TN16` to `TM64/TN16`, after synthetic
  microbenching showed `0.416 ms` vs `0.472 ms` for rows4096/in2048/out512.
  Full-model primary gate failed promotion: IDs/memory were stable and decode
  improved slightly, but prefill regressed versus the retained small-shape
  default (`512/128` `1958.693 -> 1884.288 tok/s`, `4K/128`
  `2293.994 -> 2287.071 tok/s`). Code was reverted. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-4k-gateup-tm64-rejected.json`.

- **Q8_0 T16 shared-down rows=512 TN16 rejected (2026-06-16).** Extending the
  resident Q8_0 T16 shared-expert down (`in=512,out=2048`) `TN16` tile from
  rows `>512` to rows `>=512` kept IDs and memory stable and passed the `154`
  test guard, but regressed the primary gate versus the retained small-shape
  policy: `512/128` `1958.693 / 126.924 -> 1956.322 / 126.753 tok/s`, `4K/128`
  `2293.994 / 114.991 -> 2284.550 / 114.931 tok/s`. Code was reverted to the
  rows `>512` policy. Artifact:
  `benchmarks/results/2026-06-16-gpu1-gguf-q4ks-q8t16-down512-tn16-rejected.json`.

- **G-D2 drop T16 decode-repack for Q8_0 now no-hold pending split (2026-06-17).** Removing `gguf_q8_0` from the T16 repack layout looked like a memory win in the bundled `e089d1c2` run, but the W7900 bisect showed that restoring Q8_0 T16 materialization recovers most of the prefill loss (`4K/128` about `2250 -> 2485 tok/s`) while staying in the same tracked peak in the max-sequence probe. Do not keep the raw-Q8_0 replacement as a default without a fresh exact gate and explicit human approval for any memory/speed tradeoff.
- **G-D3 fused activate+down for selected experts now correctness-blocked (2026-06-17).** The `e089d1c2` fused selected-expert bundle is the first tested point with deterministic ID drift (`220/570 -> 1813/151531`). Treat the fused `qk_t16_selected_silu_x_direct_gemv_kernel` / runtime changes as no-hold until split, covered by a targeted oracle, and proven on the full GGUF gate with no prefill/decode/memory regression.
- **G-P4 chunk-outer loop prefill memory fix (2026-06-17).** Changed `_run_bulk_prefill_and_sample` to use a chunk-outer layer-inner loop when `use_expert_sidecar=False`, allocating `_prefill_hidden_a/b` and `_GGUFFullAttentionPrefillScratch` using `chunk_size` instead of `max_positions`. This saved ~1.25 GiB of memory, allowing 128K/128 prefill on `Q4_K_S` to fit easily within 24 GiB using the standard 4096-query chunks (`22.746 GiB` peak), superseding the previous 768-token low-memory limit.
- **G-P4 low-memory non-attn chunks=512 not retained (2026-06-16).** Raising
  linear/MoE/post/RoPE chunks from 256 to 512 while keeping full-attn query at
  768 improved 128K prefill to `718.895 tok/s`, but all-768 was faster
  (`730.191 tok/s`) at the same tracked/sampled peak because the 768 full-attn
  query chunk already determines staging rows.
- **G-P4 low-memory full-attn query=1024 not retained (2026-06-16).** A
  1024-token query chunk still fit on GPU1 and improved the original 512-query
  low-memory row, but it was slower than the retained all-768 policy
  (`618.642` vs `730.191` prefill tok/s) and consumed more headroom
  (`23.393 GiB` tracked / `23.969 GiB` sampled HIP-used). Keep 768 as the
  24GB-class default unless a later multi-run sweep shows otherwise.
- **G-P1 selected-down T16 prefill launch-bound=1 rejected (2026-06-15).**
  Relaxing `gguf_k_t16_selected_prefill` (selected single-output/down WMMA
  prefill) from `__launch_bounds__(32, 2)` to `__launch_bounds__(32, 1)`
  preserved generated IDs and memory and improved the short gate
  (`512/128` prefill/decode `1647.390 / 127.012 -> 1654.917 / 127.066 tok/s`),
  but regressed the retained `4K/128` gate (`1855.806 / 115.805 ->
  1850.369 / 115.758 tok/s`). Keep selected-down T16 WMMA prefill at `32,2`.
- **G-P2 AOTriton prefill threshold=1024 rejected (2026-06-15).** Raising
  the default GGUF AOTriton prefill threshold from `512` to `1024` preserved
  memory and the `4K/128` generated ID, but changed the `512/128` final token
  (`220 -> 318`) and dropped short-prefill throughput (`1647.390 ->
  1516.829 tok/s`). Keep the GGUF AOTriton prefill threshold default at `512`.
- **G-P2 Q8 T16 prefill launch-bound=4 rejected (2026-06-15).** Relaxing
  `gguf_q8_0_t16_prefill_wmma_kernel` from `__launch_bounds__(32, 8)` to
  `__launch_bounds__(32, 4)` preserved generated IDs and memory and improved
  prefill (`512/128` `1647.390 -> 1655.202 tok/s`, `4K/128` `1855.806 ->
  1859.871 tok/s`), but regressed both retained decode medians (`127.012 ->
  126.995 tok/s`, `115.805 -> 115.610 tok/s`). Keep Q8_0 T16 prefill WMMA at
  `32,8` until a shape policy can protect decode.
- **G-P2 Q8 T16 prefill tile_n=16 rejected (2026-06-15).** Changing the default
  T16 WMMA prefill tile policy so rows `<2048` used `tile_n=16` preserved IDs
  and memory, but regressed both prefill medians (`1647.390 -> 1621.701 tok/s`,
  `1855.806 -> 1806.887 tok/s`) and both retained decode medians (`127.012 ->
  126.991 tok/s`, `115.805 -> 115.624 tok/s`). Keep the default `tile_n=32` for
  rows `>=32`.
- **G-P2 Q8 T16 dual gate+up WMMA prefill rejected (2026-06-16).** Adding a
  resident Q8_0 T16 dual gate+up WMMA prefill kernel/wrapper/dispatch path
  preserved generated IDs, memory, and bit-exact focused dual-vs-single tests,
  but regressed retained prefill (`512/128` `1647.390 -> 1621.518 tok/s`,
  `4K/128` `1855.806 -> 1850.689 tok/s`) and the `4K/128` decode gate
  (`115.805 -> 115.764 tok/s`) despite improving `512/128` decode
  (`127.012 -> 127.189 tok/s`). Keep Q8_0 T16 rows>1 gate/up prefill on the
  existing singleton WMMA path unless a tile/codegen profile proves a fused
  variant avoids the prefill loss.
- **G-P4 chunk-min=8192 rejected (2026-06-15).** Raising the
  auto-chunk minimum from `1025` to `8192` disabled chunking for the 4K gate and
  preserved generated IDs, but raised tracked peak (`21.335 -> 21.416 GiB`),
  dropped `512/128` prefill (`1647.390 -> 1592.474 tok/s`), and regressed both
  retained decode medians (`127.012 -> 126.952 tok/s`, `115.805 ->
  115.617 tok/s`) despite improving `4K/128` prefill (`1855.806 ->
  1864.260 tok/s`). Keep the auto-chunk minimum at `1025`.
- **G-P4 MoE-only chunk=2048 rejected (2026-06-15).** Raising only the default
  auto-tuned MoE prefill chunk from `1024` to `2048` preserved generated IDs and
  memory and improved `4K/128` prefill (`1855.806 -> 1857.311 tok/s`), but left
  `512/128` prefill flat/slightly lower and regressed both retained decode
  medians (`127.012 -> 126.959 tok/s`, `115.805 -> 115.717 tok/s`). Keep the
  mid-context MoE chunk at `1024`.
- **G-P4 linear/MoE chunk=2048 rejected (2026-06-15).** Raising the default
  auto-tuned linear/MoE prefill chunks from `1024` to `2048` preserved generated
  IDs and memory and improved prefill (`512/128` `1647.390 -> 1661.258 tok/s`,
  `4K/128` `1855.806 -> 1859.452 tok/s`), but regressed both retained decode
  medians (`127.012 -> 126.874 tok/s`, `115.805 -> 115.670 tok/s`). Keep the
  mid-context linear/MoE chunks at `1024` until a shape-specific policy can
  protect decode.
- **G-P3/G-P4 full-attn query chunk=2048 rejected (2026-06-15).** Lowering the
  auto-tuned full-attention query chunk from `4096` to `2048` reduced tracked
  peak memory (`21.335 -> 20.689 GiB`), but changed the `4K/128` final token
  (`570 -> 15`) and regressed throughput (`4K/128` prefill `1855.806 ->
  1814.443 tok/s`, decode `115.805 -> 115.546 tok/s`; `512/128` decode
  `127.012 -> 126.881 tok/s`). Keep the mid-context full-attention query chunk
  at `4096`; smaller query chunks need an exactness fix before memory-policy
  promotion.
- **G-D3 selected SiLU maxThreads=128 rejected (2026-06-15).** Matching
  `q4_k_t16_selected_dual_silu_direct_gemv_kernel` launch bounds to its actual
  128-thread launch (`256,1 -> 128,1`) preserved generated IDs and memory, but
  regressed the retained selected-down gate (`115.805 -> 115.669 tok/s`) and
  lowered `512/128` prefill to `1597.871 tok/s`. Keep the selected dual+SiLU T16
  GEMV launch bound at `256,1`.
- **G-D3 selected down launch-bound=1 rejected (2026-06-15).** Further
  relaxing `qk_t16_selected_direct_gemv_kernel` from `__launch_bounds__(128, 2)`
  to `__launch_bounds__(128, 1)` preserved generated IDs and memory, but
  regressed the retained selected-down gate (`115.805 -> 115.570 tok/s`) and
  lowered both prefill medians. Keep selected-down T16 GEMV at `128,2`.
- **G-D3 selected down launch-bound=3 rejected (2026-06-15).** Tightening
  `qk_t16_selected_direct_gemv_kernel` from `__launch_bounds__(128, 2)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and nudged
  `512/128` decode up (`127.012 -> 127.040 tok/s`), but regressed `512/128`
  prefill (`1647.390 -> 1618.596 tok/s`), `4K/128` prefill (`1855.806 ->
  1850.923 tok/s`), and the retained `4K/128` decode gate (`115.805 ->
  115.617 tok/s`). Keep selected-down T16 GEMV at `128,2`.
- **G-D2 Q8 launch-bound=3 rejected (2026-06-15).** Relaxing all four Q8_0 T16
  GEMV decode kernels from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and nudged
  `512/128` decode up (`127.012 -> 127.112 tok/s`), but regressed the retained
  `4K/128` gate (`115.805 -> 115.704 tok/s`) and lowered 4K prefill. Keep Q8_0
  T16 GEMV decode at `128,4`.
- **G-D2 Q8 dual-split launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_dual_split_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  `512/128` prefill (`1647.390 -> 1664.017 tok/s`), but regressed both retained
  decode gates (`127.012 -> 126.960 tok/s`, `115.805 -> 115.688 tok/s`) and
  `4K/128` prefill (`1855.806 -> 1853.900 tok/s`). Keep all Q8_0 T16 GEMV
  decode variants at `128,4` until variant-level codegen evidence says
  otherwise.
- **G-D2 Q8 single launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  `512/128` decode (`127.012 -> 127.190 tok/s`) and `4K/128` prefill
  (`1855.806 -> 1857.904 tok/s`), but regressed `512/128` prefill sharply
  (`1647.390 -> 1624.324 tok/s`) and the retained `4K/128` decode gate
  (`115.805 -> 115.740 tok/s`). Keep the single Q8_0 T16 GEMV variant at
  `128,4`.
- **G-D2 Q8 dual launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_dual_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  `512/128` decode (`127.012 -> 127.140 tok/s`) plus `4K/128` prefill
  (`1855.806 -> 1859.269 tok/s`), but regressed `512/128` prefill
  (`1647.390 -> 1645.535 tok/s`) and the retained `4K/128` decode gate
  (`115.805 -> 115.692 tok/s`). Keep the dual Q8_0 T16 GEMV variant at
  `128,4`.
- **G-D2 Q8 triple-split launch-bound=3 rejected (2026-06-15).** Relaxing only
  `q8_0_t16_triple_split_gemv_kernel` from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory and improved
  prefill (`512/128` `1647.390 -> 1665.444 tok/s`, `4K/128` `1855.806 ->
  1856.539 tok/s`), but regressed both retained decode gates (`127.012 ->
  126.937 tok/s`, `115.805 -> 115.633 tok/s`). Keep the triple-split Q8_0 T16
  GEMV variant at `128,4`.
- **G-D2 Q8 launch-bound=5 rejected (2026-06-15).** Tightening all four Q8_0
  T16 GEMV decode kernels from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 5)` preserved generated IDs and memory, but regressed
  both retained decode medians (`127.012 -> 126.870 tok/s`, `115.805 ->
  115.565 tok/s`) and did not improve prefill. Keep Q8_0 T16 GEMV decode at
  `128,4`.
- **G-D2 Q8 half-pointer d-load rejected (2026-06-15).** Loading all four Q8_0
  T16 GEMV decode variants' fp16 `d[16]` scales through
  `reinterpret_cast<const half_t*>(tile)[col]` instead of `fp16_bytes_to_float`
  preserved generated IDs and memory, but regressed both prefill medians
  (`1647.390 -> 1626.168 tok/s`, `1855.806 -> 1853.323 tok/s`) and both decode
  medians (`127.012 -> 126.941 tok/s`, `115.805 -> 115.560 tok/s`). Keep the
  byte/union `fp16_bytes_to_float` loads for Q8_0 T16 GEMV.
- **G-D2 scale broadcast rejected (2026-06-15).** Replacing per-lane Q8_0 T16
  scale loads with `__shfl` broadcast preserved correctness but regressed the
  gate metric from `114.602` to `106.905 tok/s` (`-6.7%`) and reduced both gate
  decode medians. The original per-lane scale loads are faster on this GPU;
  do not retry this exact change without new ISA/rocprof evidence.
- **G-D2 block64 rejected (2026-06-15).** Launching Q8_0 T16 GEMV with 64
  threads instead of 128 passed the synthetic Q8 fixture but changed full-model
  gate tokens (`220 -> 97799`, `570 -> 28944`) and still regressed decode vs the
  original baseline (`114.602 -> 111.797 tok/s` primary metric). Keep the 128-
  thread launch unless a new accumulation-order-safe variant is proven.
- **G-D2 Q8 launch-bound=2 rejected (2026-06-15).** Relaxing the Q8_0 T16 GEMV
  `__launch_bounds__(128, 4)` minimum-block count to `2` preserved generated IDs
  and memory, but slightly regressed the current lb1 gate metric
  (`115.114 -> 115.085 tok/s`) and hurt `512/128` prefill noise. Keep the Q8_0
  T16 GEMV launch-bound at `128,4` unless an occupancy/code-object census shows
  a clearer pressure problem.
- **G-D5 Q6 launch-bound=3 rejected (2026-06-15).** Relaxing the Q6_K T16
  lm-head GEMV from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 3)` preserved generated IDs and memory, but regressed
  both gate decode medians versus the retained selected-down lb2 row
  (`127.012 -> 126.932 tok/s`, `115.805 -> 115.631 tok/s`). Keep Q6_K T16
  GEMV at `128,4`.
- **G-D5 Q6 launch-bound=2 rejected (2026-06-15).** Relaxing the Q6_K T16
  lm-head GEMV `__launch_bounds__(128, 4)` minimum-block count to `2` preserved
  generated IDs and memory, but regressed the current lb1 gate metric
  (`115.114 -> 115.013 tok/s`) and lowered `512/128` prefill to `1635.484 tok/s`.
  Keep Q6_K T16 GEMV at `128,4`.
- **G-D5 Q6 launch-bound=5 rejected (2026-06-15).** Tightening the Q6_K T16
  lm-head GEMV from `__launch_bounds__(128, 4)` to
  `__launch_bounds__(128, 5)` preserved generated IDs and memory, but regressed
  both retained decode medians (`127.012 -> 126.784 tok/s`, `115.805 ->
  115.604 tok/s`) and lowered both prefill medians. Keep Q6_K T16 GEMV at
  `128,4`.
- **G-D5 Q6 per-tile d preload rejected (2026-06-15).** Preloading the Q6_K T16
  lm-head GEMV per-column `d[16]` and scales pointer outside the `k` loop
  preserved generated IDs and memory and improved `512/128` prefill
  (`1647.390 -> 1656.158 tok/s`), but regressed `4K/128` prefill
  (`1855.806 -> 1855.017 tok/s`) and the retained `4K/128` decode gate
  (`115.805 -> 115.579 tok/s`). Keep the original in-loop `d` loads unless an
  ISA/occupancy profile proves the preload no longer costs the 4K path.
- **G-D5 Q6 half-pointer d-load rejected (2026-06-15).** Loading the Q6_K T16
  lm-head GEMV per-column `d` scale through
  `reinterpret_cast<const half_t*>(tile + Q6_T16_D_OFFSET)[col]` preserved
  generated IDs and memory and slightly improved `512/128` decode
  (`127.012 -> 127.089 tok/s`), but regressed `512/128` prefill
  (`1647.390 -> 1593.584 tok/s`), `4K/128` prefill
  (`1855.806 -> 1852.882 tok/s`), and the retained `4K/128` decode gate
  (`115.805 -> 115.640 tok/s`). Keep the byte/union `fp16_bytes_to_float` Q6
  `d` load unless a disassembly-guided variant avoids the prefill/4K loss.
- **G-D4 split decode threshold=8192 rejected (2026-06-15).** Raising the
  full-attention split/gate fused decode threshold from `1024` to `8192` forced
  the `4K/128` gate onto the direct context + gate-mul path and looked faster
  (`115.805 -> 125.592 tok/s`), but changed the final token (`570 -> 263`).
  Keep the split threshold default at `1024`; direct 4K decode is
  correctness-invalid until fixed.
- **G-D4 split decode threshold=512 rejected (2026-06-15).** Lowering the
  full-attention split/gate fused decode threshold from `1024` to `512` routed
  the short gate through the warp-split path and preserved memory, but changed
  the `512/128` final token (`220 -> 17`) and regressed decode sharply
  (`127.012 -> 113.635 tok/s`); `4K/128` decode also slipped
  (`115.805 -> 115.742 tok/s`). Keep the split threshold default at `1024`.
- **G-D4 grouped-GQA min-context=512 rejected (2026-06-15).** Lowering the
  grouped GQA split decode default threshold from `4096` to `512` enabled the
  grouped path on the short gate and preserved generated IDs and memory, but
  regressed both retained decode medians (`127.012 -> 126.935 tok/s`,
  `115.805 -> 115.652 tok/s`) and sharply lowered `512/128` prefill. Keep the
  grouped GQA min-context default at `4096`.
- **G-D4 GQA warp-split default rejected (2026-06-15).** Disabling the grouped
  GQA split full-attention decode default preserved generated IDs and memory, but
  regressed `4K/128` decode sharply (`115.114 -> 112.597 tok/s`). Keep grouped
  GQA split decode enabled for mid-context GGUF unless a later shape-specific
  threshold proves otherwise.
- **G-H2 graph4 rejected (2026-06-15).** `--graph-steps-per-replay 4` reused the
  existing multi-step capture support and kept the 4K token stable, but changed
  the `512/128` final token (`220 -> 11`). Treat GGUF multi-step graph replay as
  correctness-blocked until a fixture proves the captured position/token state
  advances exactly across replay groups.

## What we copy from the MTP/DFlash/megakernel successes

The MTP sprint moved from `0.758x / 27.8 ms` to a retained `1.023x / 14.134 ms`
row by treating every cycle component as measurable. DFlash accepted a `1.231x`
27B dense row by adding an online confidence gate only after exactness and
same-session AR baselines were locked. The megakernel work prevented a bad
default by proving a seductive one-launch FFN kernel was slower on the GPU.

For GGUF, copy these rules:

1. **Same-suite or it does not count.** Every candidate compares against the
   same branch, model file, prompt tokens, context, warmups, GPU, ROCm stack, and
   correctness fixture.
2. **Profile before changing.** Start each lane with a `rocprofv3 --kernel-trace`
   plus bucket summary. If a candidate does not target a measured bucket, park it.
3. **Keep exact micro-wins.** Small retained wins from MTP were additive
   (`0.04-0.11 ms/cycle` was worth keeping). GGUF should keep small exact wins
   when the same benchmark suite is non-regressive.
4. **Re-profile when the operating point moves.** MTP's B=1 retune found wins
   that were no-holds at B=3. GGUF must re-check candidates separately for
   `512`, `4K`, `32K`, and c>N shapes.
5. **Do not worship launch count.** The FFN megakernel removed launches but lost
   grid parallelism and became `2.66x` slower than production. GGUF fusions must
   preserve occupancy/coalescing and remove real memory traffic, not just calls.
6. **Reject memory mirages.** Sidecars or raw+packed duplication that make a
   benchmark impossible on 24 GiB cards are blocked even if W7900 runs.
7. **Promote only with rollback semantics.** Exact, non-regressive performance
   paths become default; default-off paths need a concrete blocker in
   `docs/REFACTOR.md` or this file.

## Baseline refresh protocol

Run these before editing kernels. Use a clean shell and do not let profiled
processes spawn `hipcc`. The active tuning loop uses GPU1
(`HIP_VISIBLE_DEVICES=1`), which maps to the 24 GiB-class testbed on this host;
verify the sysfs card name before llama.cpp peak-memory runs.

```bash
# From /home/lhl/hipEngine
ROOT=/home/lhl/mambaforge/envs/therock
PY=$ROOT/bin/python3.12
$ROOT/bin/hipcc --version > /tmp/hipengine-hipcc-version-713.txt
```

### hipEngine GGUF sweep

```bash
GGUF_M=/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$GGUF_M" --quant gguf_q4_k_m \
  --workloads 512/128 4K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-gate-hipengine-q4km.json

# Promotion/final check, after a candidate survives the primary gates.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine gguf --model "$GGUF_M" --quant gguf_q4_k_m \
  --workloads 128K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 1 \
  --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-final-128k-hipengine-q4km.json
```

Run Q4_K_S only as an explicit secondary memory/consumer-card diagnostic; it is
not part of the active 1:1 llama.cpp parity gate.

### llama.cpp refresh

Use the checked-out local binaries so we compare against what the user is
actually seeing:

```bash
MODEL=/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
HIP_VISIBLE_DEVICES=1 python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-hip/build/bin/llama-bench \
  --model "$MODEL" --backend hip \
  --workloads 512/128 4K/128 \
  --repetitions 3 --ngl 99 --flash-attn 1 \
  --cache-type-k f16 --cache-type-v f16 --poll 10 --card-name card0 \
  --extra-args "-dev ROCm0" \
  --output benchmarks/results/<date>-gpu1-gguf-tuning-baseline-llamacpp-hip-q4km.json

python3 scripts/llamacpp_bench_with_peak.py \
  --llama-bench /home/lhl/llama.cpp/llama.cpp-vulkan/build/bin/llama-bench \
  --model "$MODEL" --backend vulkan \
  --workloads 512/128 4K/128 \
  --repetitions 3 --ngl 99 --flash-attn 1 \
  --cache-type-k f16 --cache-type-v f16 --poll 10 --card-name card0 \
  --extra-args "-dev Vulkan0" \
  --output benchmarks/results/<date>-gpu1-gguf-tuning-baseline-llamacpp-vulkan-q4km.json
```

### PARO c=1 reference on the same host

```bash
PARO=/home/lhl/.cache/huggingface/hub/models--shisa-ai--Qwen3.6-35B-A3B-PARO-packed/snapshots/437eba06df05aad71a4dacdcaf3fff70ae1ee8a1
HIP_VISIBLE_DEVICES=1 PYTHONPATH=. "$PY" scripts/qwen35_readme_sweep.py \
  --engine paro --model "$PARO" --backend hip_gfx1100 \
  --shared-expert-format packed_paro_w4 --token-id 9707 \
  --workloads 512/128 4K/128 \
  --warmup-runs 1 --measured-runs 3 --warmup-decode-tokens 4 \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --attn-aotriton-min-tokens 512 --graph-replay-decode \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-baseline-hipengine-paro.json
```

### Correctness gates for GGUF tuning

Use the narrowest gate first, then the full model gate before retaining a perf
row:

```bash
# Dense 0.8B and 35B smoke fixture path.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
  PYTHONPATH=. python3 scripts/qwen35_gguf_e2e_correctness.py --repeat 2

# qwen35moe safety gate when touching 35B GGUF kernels/materialization.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
  PYTHONPATH=. python3 scripts/qwen35_gguf_p9_e2e_correctness.py

# Targeted bundles for decode-repack / T16 work.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
  python3 -m pytest \
    tests/test_gguf_t16_repack.py \
    tests/test_gguf_q8_0_t16_gemv_decode.py \
    tests/test_gguf_t16_selected_gemv_decode.py \
    tests/test_gguf_q6_k_t16_gemv_decode.py \
    tests/test_gguf_gemv_decode_dispatch.py \
    tests/test_qwen35_gguf_compact_moe_gemv_routing.py -q
```

### Profile capture and summary

Warm the JIT outside the profiler, then trace only cached runs. Store raw CSVs
under `/tmp`; commit only compact summaries.

```bash
RUN=/tmp/hipengine-gguf-tuning/<date>-q4km-512
mkdir -p "$RUN"

# Warmup/build outside rocprofv3.
HIP_VISIBLE_DEVICES=1 \
HIPENGINE_GGUF_DECODE_REPACK=1 \
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
PYTHONPATH=. "$PY" scripts/qwen35_gguf_bench.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --quant gguf_q4_k_m --prompt-length 512 --decode-tokens 16 \
  --persistent-session --force-bulk-prefill --bulk-prefill-attention-mode bulk \
  --use-wmma-prefill --use-gemv-decode \
  --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
  --json "$RUN/warmup.json"

# Trace a short decode window for kernel mix.
rocprofv3 --kernel-trace -d "$RUN/rocprof" -o q4km512 -f csv -- \
  env HIP_VISIBLE_DEVICES=1 \
      HIPENGINE_GGUF_DECODE_REPACK=1 \
      HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version-713.txt \
      PYTHONPATH=. \
      "$PY" scripts/qwen35_gguf_bench.py \
        --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
        --quant gguf_q4_k_m --prompt-length 512 --decode-tokens 16 \
        --persistent-session --force-bulk-prefill --bulk-prefill-attention-mode bulk \
        --use-wmma-prefill --use-gemv-decode \
        --compiler-version-file /tmp/hipengine-hipcc-version-713.txt --require-cached-build \
        --json "$RUN/profiled.json"

# Summarize the CSV into a compact artifact.
python3 scripts/qwen35_gguf_rocprof_summary.py \
  --csv "$RUN/rocprof/q4km512_kernel_trace.csv" \
  --tokens-decode 16 \
  --json benchmarks/results/<date>-gpu1-gguf-tuning-rocprof-summary.json
```

## Tuning lanes

Use stable IDs in commits, artifacts, and `WORKLOG.md`.

### M lane — Measurement and attribution

| ID | Candidate | Why | Acceptance |
| --- | --- | --- | --- |
| G-M0 | Refresh hipEngine GGUF, llama.cpp HIP/Vulkan, and PARO rows from this worktree. | The current scorecard mixes quant variants and dates. | Compact JSON artifacts for each runner, exact commands, clean environment metadata. |
| G-M1 | Build a matched-token prompt fixture shared by hipEngine and llama.cpp. | Avoid token/template drift when comparing decode and MTP-bearing GGUFs. | Same token IDs and prompt length proven in artifacts. |
| G-M2 | Produce per-shape GGUF rocprof bucket summaries for `512`, `4K`, `32K`, `128K`. | P9.C showed selected-MoE and Q8 buckets moved with shape; do not optimize blind. | `qwen35_gguf_rocprof_summary.py` artifacts list top buckets, dispatches, VGPR/scratch, and legacy fallback presence. |
| G-M3 | Add/refresh code-object occupancy and scratch census for hot GGUF kernels. | Decode is memory-bound and occupancy-sensitive; any scratch on hot kernels is a bug. | Hot kernels report `Scratch_Size=0`, acceptable VGPR, and no unexpected LDS expansion. |
| G-M4 | Memory residency census by tensor family/layout. | GPU1 gates fit but leave only ~2 GiB free, and stale full-sweep rows exceeded 24 GiB; duplicate layouts hide in totals. | `qwen35_gguf_bench.py` snapshots now emit an owned-session breakdown; `2026-06-16-gpu1-gguf-q4ks-memory-census.json` attributes the 128K GPU1 blocker to projected KV/prefill residency, not duplicate layouts. |

### D lane — Decode throughput

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-D1 | Eliminate any legacy `*_prefill_out*` or raw GGUF fallback from decode traces. | `GGUF_DECODE_REPACK.md` says remaining legacy fallback is a regression, especially lm-head. | rocprof summary shows T16/GEMV decode buckets and no unplanned legacy decode buckets. |
| G-D2 | Tune T16 GEMV launch bounds / tile shape for Q8T16, Q4T16, Q5T16, Q6T16 at rows=1. | T16 is the right replacement layout but may not be at the best occupancy point. | Same-suite decode improvement, `Scratch_Size=0`, no correctness regression. |
| G-D3 | Router + selected-MoE decode fusion where it removes real memory traffic. | MTP router fusion was a large retained win; GGUF MoE routing still has small kernels per layer. | Generated IDs/logits stable; launch count and wall improve; no extra HBM staging. |
| G-D4 | Direct full-attention short-context producer/reduce path parity with PARO direct-gate work. | MTP one-split direct gate saved wall exactly; GGUF full-attn decode may carry similar split/reduce overhead. | Exact full-attention fixture; `512`/`4K` decode improved or no-held. |
| G-D5 | lm-head Q6T16/argmax fusion or top-1-only path. | Final logits are a large dense projection; avoid full materialization if only greedy token is needed. | Greedy token/logprob semantics preserved; sampling/logprob paths fall back explicitly. |
| G-D6 | c>N GGUF serial bridge replacement with true resident batch path. | PARO c=4/c=8 shows large aggregate gains; GGUF currently has more serial behavior. | Separate c>N correctness/provenance gate; not required for c=1 promotion. |

### P lane — Prefill throughput

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-P1 | Follow up the retained selected-MoE half-seq WMMA rewrite only if a 16-column/codegen variant can reduce the remaining 256-VGPR cap. | Half-seq moved the 4K/16 target bucket `800.455 -> 454.370 ms`; remaining upside is smaller and must not trade back decode stability. | Target bucket moves materially beyond half-seq without >24 GiB duplicate storage or decode noise. |
| G-P2 | Shape-specific Q8_0 shared/dense WMMA schedule. | Q8_0 bucket is still large; P9.C1 showed shape-specific tile rules mattered. | 512/0 and 4K/0 prefill both non-regressive; code path remains registered by quant/layout key. |
| G-P3 | Full-attention prefill glue parity with PARO/AOTriton path. | Long-context prefill is chunk/attention sensitive. | 32K/128 and 128K/128 prefill improve without decode/memory regression. |
| G-P4 | Chunk auto-tune and memory budget policy for Q4_K_M, with Q4_K_S as secondary memory context only. | Current Q4_K_M GPU1 short gates fit at `22.487 GiB`; baseline max-context was only `51K/128` before the 4096-row full-attn scratch cliff. | 24 GiB >=52K contexts now use 1024-row full-attn query chunks: max observed pass moved `51K/128 -> 103K/128` (`+102%` prompt tokens) with stable one-run IDs and `23.484 GiB` tracked peak; `104K/128` still OOM and `128K/128` still needs KV/weight residency reduction. Historical Q4_K_S `128K/128` fit with low-memory chunks at `730.191 / 67.733 tok/s`, stable ID `[220]`, `23.310 GiB` peak. |

### H lane — Host/runtime and graph replay

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-H1 | Cache resident tensor views, scratch handles, and per-layer dispatch objects in GGUF like MTP verifier caches. | MTP host caches removed real milliseconds without math changes. | Profile shows host window reduction, exact outputs unchanged. |
| G-H2 | Audit graph capture/replay setup for per-token validation or recapture overhead. | Graph replay helped PARO only a few percent; GGUF may still be doing avoidable host work. | Wall improves with same kernel time, or candidate is rejected. |
| G-H3 | Collapse ctypes wrapper overhead only if measured >3% of decode wall. | C++ host loop is not worth doing without proof. | Before/after host-marker or Python profile evidence. |

### L lane — Layout and memory

| ID | Candidate | Starting hypothesis | Gate |
| --- | --- | --- | --- |
| G-L1 | Make replacement-layout materialization truly replace raw tensors for all promoted T16 paths. | Decode-repack is only useful if it does not duplicate model weights. | Allocation census proves no raw+packed duplicate for covered tensors. |
| G-L2 | Q4_K_M/Q4_K_S model-specific residency plan. | Q4_K_S saves selected-down bytes but current run still peaks high. | Peak target under 24 GiB for 512/128 and 4K/128, or documented blocker. |
| G-L3 | Decide whether any sidecar is worth keeping. | P9.C5 side metadata and expert pack8 sidecar were slower or too large. | Only retained if replacement, not duplicate, and E2E faster. Otherwise add `docs/REFACTOR.md` removal note. |

## First sprint checklist

1. Create GPU1 gate artifacts for hipEngine GGUF Q4_K_M, local llama.cpp
   HIP/Vulkan Q4_K_M, and PARO where it fits on TheRock 7.13. Keep W7900
   comparison artifacts only when GPU1 cannot fit a required final/promotion
   shape. Q4_K_S artifacts are secondary memory/consumer-card diagnostics only
   when explicitly requested.
2. Generate `G-M2` rocprof bucket summaries for at least `512/128` and `4K/128`.
3. Answer these from data before editing kernels:
   - Is the gap primarily decode kernels, prefill kernels, host overhead, or memory/chunk policy?
   - Which kernel family is the top decode bucket after T16 repack: selected MoE, dense Q8/Q6, full attention, router, lm-head, or glue?
   - Are any legacy GGUF raw/prefill-out kernels still running in measured decode?
   - How much peak memory is raw tensors, T16 tensors, KV, scratch, and graph capture?
4. Pick **one** highest-share bucket and run a focused multiloop-style pass:
   hypothesis -> code -> correctness gate -> profile -> keep/revert/log. Primary
   pass/fail is `512/128` plus `4K/128`; `128K/128` is the final promotion gate.
5. If a change is exact and same-suite non-regressive, make it default and
   update artifacts/rollups. If it stays gated, record the blocker.

## Promotion policy

A GGUF tuning change is promoted only when all of the following hold:

- Relevant GGUF correctness fixtures pass, including qwen35moe when 35B paths are touched.
- `rocprofv3 --kernel-trace` confirms the intended kernel(s) ran and no unexpected fallback dominates.
- Same-suite Q4_K_M benchmark improves prefill and/or decode on the GPU1 `512/128` and `4K/128` gates, or removes memory/launch/KV overhead without throughput regression in any primary gate dimension. A candidate with a remaining prefill/decode/memory/correctness tradeoff is no-hold until the user/human lead explicitly accepts it.
- Benchmark artifact follows `docs/BENCHMARK.md` and rollup updates are made for accepted performance rows.
- No torch hot-path import, no llama.cpp hot-path FFI, no model/dispatch `if backend == ...` or `if quant == ...` branch.
- No unbounded duplicate residency; GPU1 `128K/128` is checked before promotion, and W7900-only rows are labeled as such.

## Do-not-chase list for this lane

- **Do not** start with another raw-GGUF `prefill_out` micro-tweak if the profile
  still shows replacement-layout coverage gaps.
- **Do not** re-enable unsafe qwen35moe fast paths unless the P9.E2-style gate
  passes with effective fast paths actually enabled.
- **Do not** copy llama.cpp kernels blindly. Use llama.cpp for block math,
  subgroup-shape ideas, and baselines; hipEngine kernels must fit the raw-pointer
  ABI, registry, and resident layout.
- **Do not** assume wave64/Vulkan subgroup behavior maps to HIP. Prove it with a
  tensor-level fixture and `rocprofv3` evidence.
- **Do not** treat a single synthetic 512-token prompt as a quality proof. It is
  a speed sentinel; promotion still needs deterministic generated IDs/logits and
  the documented correctness gate.

## Open questions

- Is the user's latest `~/llama.cpp/` faster than the stale README HIP rows, and
  if so, which commit/build flag changed the target?
- Q4_K_M is the canonical tuning target for hipEngine GGUF when comparing with
  local llama.cpp rows; Q4_K_S is secondary memory/consumer-card context only.
- Does the Q4_K_M path fit the 24 GiB-class envelope at the required contexts
  after replacement-layout and scratch cleanup, or does consumer-card support
  need an explicit Q4_K_S diagnostic/default decision?
- Which GGUF path should get native sampling/logprob parity first: c=1 greedy
  only, c=1 sampled, or c>N batch?
- Is any approximate/relaxed GGUF math acceptable, or do we keep strict GGML
  dequant parity for all promoted rows?

- **G-D2 Pack8 layout optimization tradeoff needs human review.** Disabling `LAYOUT_Q4_K_PACK8` layout materialization and reverting dense Q4_K tensors to `LAYOUT_RAW_GGUF` was recorded as a memory win, but it also recorded prefill/decode losses. Under the current policy this is not an agent-auto-retain decision: keep it only if a fresh full-vector gate proves no regression, or if the user/human lead explicitly accepts the speed-for-memory tradeoff after reviewing the artifact.
- **G-H1 C-dispatch for GGUF MoE.** Invalidated. Analysis showed that GGUF uses HIP graph capture for decode (if enabled, as it is by default), so Python dispatch overhead is already bypassed entirely. This also applied to other optimizations involving C-dispatch, confirming that the current logic is optimal without relying on `moe_c1_dispatch` which was designed for PARO where graph capture is not always available.
- **G-H2 Fuse RMSNorm+rotate.** Invalidated. GGUF doesn't use AWQ input-rotation; this optimization is specific to PARO's w4a16 layout. The current GGUF architecture uses rotary position embeddings (`gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight`) applied directly to the query/key outputs AFTER the linear layer, making this specific fusion inapplicable to GGUF.
