# hipEngine

hipEngine is a ROCm-native local LLM inference engine designed from the ground
up for AMD RDNA GPUs (starting with gfx1100, gfx1151). It pairs a small
purpose-built Python host with a complete suite of custom-tuned HIP kernels
developed through 100+ iterations of profiling and tuning.

hipEngine has lightweight dependencies with no PyTorch required for fully
supported GPUs and models.

## Core principles

- **HIP-first, not CUDA-ported.** Kernels directly target AMD hardware like 
  gfx1100/RDNA3 with wave32, vec8 FMA, and the actual cache hierarchy.
- **Torch-free runtime.** `import torch` is **not** on the hot path. The
  runtime owns a thin `hipengine.Tensor` over raw HIP/CUDA device pointers and
  drives `hipblasLt`, `hipGraph`, AOTriton, and JIT builds through `ctypes`.
  Torch appears only as an optional dlpack bridge behind the `hipengine[torch]`
  extra (~125 MiB install including the vendored AOTriton subset vs ~2 GiB with
  torch).
- **Multi-backend from day one.** Kernels live under `kernels/hip_gfx1100/`,
  `kernels/hip_gfx1151/`, `kernels/cuda_sm86/`, `kernels/cpu_reference/` as
  peer trees.
- **Four-axis plugin registry.** Kernels are keyed by
  `(backend, layer, quant, variant)`. Models, quant schemes, and layers are
  plugins. No `if backend == "..."` or `if quant == "..."` branches in
  dispatch / engine / model code.
- **Fused + unfused coexist.** Every fused composite
  (`rmsnorm+rotate`, `gate_combine_residual`, …) has a numerically-equivalent
  unfused chain registered under its primitives, used as both fallback and
  correctness baseline.
- **Evidence-backed performance.** Every performance claim ships with
  model + quant + workload shape + hardware + exact command + correctness gate
  (KL ≤ 0.05, top-1 ≥ 90% vs `kernels/cpu_reference/`). See
  [`docs/BENCHMARK.md`](docs/BENCHMARK.md) and
  [`benchmarks/README.md`](benchmarks/README.md).

## Status

**v0.3.0 alpha.** The runtime hot path is torch-free by construction, and the
first two 35B-class model-loading surfaces are available on gfx1100 and gfx1151:
[shisa-ai/Qwen3.6-35B-A3B-PARO-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format, plus Qwen3.6 GGUF
`Q4_K_M` / `Q4_K_S` files through the resident GGUF path, plus native
`UD-Q3_K_M` execution with retained gfx1100 evidence. Older benchmark artifacts
may still show the historical
`Qwen3.6-35B-A3B-PARO-full4096-e5-packed` name or local MTP-BF16 assembly path;
those rows use the same packed PARO architecture and remain the evidence for the
numbers below.

- Model-aware `backend="auto"` / `quant="auto"` defaults select the registered
  PARO or GGUF route without environment-variable setup. Direct generation now
  supports exact token-id prompts, detailed outputs, logprobs, structured finish
  details, and backend execution telemetry.
- PARO and GGUF support ordinary sampling controls including top-k/min-p,
  penalties, logit bias, suppression, deterministic seeds, EOS/min-token policy,
  token stops, and multi-token stops. Covered PARO shapes use a native GPU
  sampler; unsupported shapes use an explicit host fallback.
- The OpenAI-compatible server includes capability/readiness discovery, token
  and context diagnostics, exact usage accounting, request batching, deadlines,
  cancellation, opt-in Prometheus metrics, and detailed streaming metadata.
- Local-agent support includes OpenAI-style tools, Qwen thinking controls,
  structured-output result validation, deterministic continuation handles, and
  app-local transcript sessions with fork, rollback, snapshot, and overflow
  policies.
- Qwen3.6 GGUF models with NextN tensors expose detailed MTP generation and a
  guarded explicit non-streaming server route. Dense PARO DFlash and the shared
  speculative proposal/verify/commit infrastructure are available as retained
  runtime and benchmark paths.
- The same Laguna family is supported on W7900/gfx1100 with the
  `UD-Q2_K_XL` GGUF. Its exact matrix512/attention128 prefill default combines
  role-qualified Q5/Q6 output tiling, pair16 grouped IQ gate/up/down,
  adjacent-row qrow4 SWA after a measured C256 crossover, and transient exact-F32
  Q5 expansion plus production-ordered 8x4/12x4/8x10/16x5/8x12/12x8 reduction
  on all eight roles. No dequantized weight persists. H5E reached
  **184.997/172.104/131.496 tok/s** at 512/1K/4K; H5F retained the narrow exact
  N48 micro-win. H5G's constant-80/96 tiles now publish
  **188.393/175.042/132.743 tok/s**, another **+2.192%/+2.055%/+1.329%** over
  H5F. H5H closes larger exact tiles: constant-112 loses every role and
  constant-128 reaches VGPR256 with 28–52 B scratch. The retained H5G request
  reclassifies to **2,667.034 ms / 1,720 dispatches**: Q5 **920.633 ms**, IQ
  down **560.642 ms**, attention **468.533 ms**, and Q6 **177.047 ms**. With Q5
  geometry and prior attention lanes closed, WPF-H5I's exact-Q6 F32 expansion
  plus ordered consumer clears the all-role leaf and production gates. Four roles
  select exact `16x5`/`16x4`/`8x4`; both long-K roles and the wide-N F32 role
  retain raw coltile. Q5+Q6 reuse one **150,994,944-byte** plane with no new
  allocation. Integrated tracing records **143+143** candidate launches and
  three exact fallbacks, moving Q6 **177.047 -> 110.170 ms (-37.774%)** and
  request kernel sum **2,667.034 -> 2,600.260 ms (-2.504%)**. Clean
  selector-unset production is **191.713/178.080/134.411 tok/s** at 512/1K/4K.
  H5J then promotes exact resident-segment IQ3 plus a local32 launch of the
  retained physical IQ4 body for K1024/N3072 selected down. Complete logits,
  all 48 hidden boundaries, routing prefixes, active K/V, and every
  `KVLiveSpans` field are bit-exact at KL 0; repeat and teardown match. Cached
  integrated tracing observes all **45 IQ3 + 2 IQ4** production calls, moving
  selected down **556.749 -> 497.145 ms (-10.706%)** and complete request kernel
  sum **2,600.260 -> 2,532.020 ms (-2.624%)** at unchanged **1,862** dispatches.
  Clean selector-unset production reaches **196.103/181.859/137.169 tok/s** at
  512/1K/4K, **+2.290%/+2.122%/+2.052%** over H5I and a **3.540x** matched M512
  gap. Every sample is byte-exact, deterministic, and lifecycle-clean; no
  allocation, workspace, or sidecar is added. Map/shape/registration misses and
  gfx1151 retain their preceding exact routes. H5K closes larger resident IQ3
  row ownership: rowbatch12 loses all 45 actual layers by **+6.893%/+5.771%**
  event/wall, and rowbatch16 worsens to **+10.770%/+9.870%** despite exact bytes
  and zero scratch. All temporary surfaces are removed and H5J is unchanged.
  Post-H5K attribution assigns **919.697 ms** to exact Q5, including **904.399
  ms** of ordered consumers; two roles own **741.721 ms (82.0%)**. H5L promotes
  separately registered exact weight-tile-major traversal on six material
  roles while F32 N48/N72 retain H5G. Complete M512 state is KL0/byte-exact
  across all **48** boundaries, logits, K/V/live spans, repeat, workspace, and
  lifecycle. Cached tracing observes **235** producers, **188** candidates, and
  **47** fallbacks, cutting Q5 **919.697 -> 466.986 ms (-49.224%)** and request
  kernel sum **2,532.020 -> 2,074.261 ms (-18.079%)** at unchanged **1,862**
  dispatches. H5L package-default 512/1K/4K reaches
  **237.956/217.888/157.366 tok/s (+21.342%/+19.812%/+14.725% over H5J)**.
  Post-H5L tracing ranks matched residuals attention **437.720 ms**, Q5
  **408.035 ms**, and IQ down **338.619 ms**; exact SWA qrow4 owns **268.720 ms
  / 58.49%** of attention. H5M's separately registered source-qualified qrow4
  keeps every admitted two-pass operation while skipping unused current/cache
  K/V loads. Dense starts 0/128/256/384 plus 508..515 wrap/eviction/ragged cases
  are bit-exact, and production starts 256/384 improve event/wall sums
  **6.728/6.737 -> 6.437/6.443 ms (-4.324%/-4.354%)**. Complete M512 state is
  KL0/byte-exact across all 48 boundaries, logits, K/V/live spans, repeat, and
  teardown. Cached tracing observes exactly **48 global + 72 wave32 + 72 H5M**
  calls; qrow4 falls **268.720 -> 260.500 ms (-3.059%)**, attention **459.445 ->
  450.790 ms (-1.884%)**, and request kernel sum **2,074.261 -> 2,060.485 ms
  (-0.664%)** at unchanged **1,862** dispatches. Clean package-default
  512/1K/4K promotes **238.565/218.182/158.138 tok/s
  (+0.256%/+0.135%/+0.490% over H5L)**, narrowing the matched M512 gap
  **2.91728x -> 2.90983x** with no allocation or sidecar. The production-identical
  post-H5M trace reconciles **2,060.485 ms / 1,862 dispatches** and ranks matched
  gaps attention **429.065 ms**, Q5 **406.709**, and IQ down **336.162**. Exact
  source-qualified qrow4 still owns **260.500 ms / 57.79%** of attention at starts
  256/384. H5N therefore screens a separately registered dense-first-fill exact
  specialization that derives identity-ring visibility without token-position/
  eviction reads while retaining base-offset mapping and every H5M two-pass
  operation. This is target selection, not a performance claim. Both short rows
  exceed 150 tok/s and 4K remains positive; 16K+ stays closed below the 800/700
  stretch target
  ([current production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-sourcequal-exact-production.json) ·
  [post-H5M residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5m-residual.json) ·
  [H5M leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-qrow4-sourcequal-exact-candidate.json) ·
  [post-H5L residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5l-residual.json) ·
  [H5L production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-production.json) ·
  [H5L leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-weight-major-candidate.json) ·
  [H5J production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq-row-ownership-production.json) ·
  [post-H5K residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5k-residual.json) ·
  [H5K rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-iq3-larger-resident-rowbatch-rejected.json)).
  A separately registered WPF-H2 F16-WMMA FlashAttention leaf keeps BF16 K/V
  and complete `KVLiveSpans` while moving the standalone 12-global/36-SWA M512
  family **490.919 -> 21.719 ms (22.603x)**, nominally matching llama.cpp's
  **21.725-ms** trace. Runtime promotion is rejected: the complete
  18-prompt/576-step gate reaches max KL **1.804860 > 0.05** despite **564/576**
  top-1 and **1.027x** diagnostic prefill. The temporary runtime path is removed
  and production stays exact
  ([rejection](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-source-flash-attention-rejected.json) ·
  [leaf evidence](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-source-flash-attention-candidate.json)).
  The separately registered WPF-H3 IQ3/IQ4 source-MMQ leaf moves all 47 actual
  M512 selected-down layers **565.437 -> 115.951 ms (4.877x)**; IQ3 alone is
  **27.145% below** llama.cpp's matched family trace. Runtime promotion is
  rejected: complete quality reaches max KL **0.373028 > 0.05** despite
  **567/576** top-1 and **1.192x** diagnostic prefill, while an IQ3-only source
  followup still reaches **0.372917**. The temporary owner is removed and
  production stays exact
  ([rejection](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-rejected.json) ·
  [leaf evidence](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-iq3-iq4-source-mmq-candidate.json)).
  The separately registered WPF-H4 Q6 F16/rocBLAS leaf moves the six-shape,
  144-call M512 family **174.351 -> 14.349 ms (12.151x)**, **3.825% below**
  llama.cpp's matched **14.919865-ms** stack. Runtime promotion is rejected:
  complete changed-arithmetic quality reaches max KL **0.338657 > 0.05** despite
  **567/576** top-1 and **1.042x** diagnostic prefill. The temporary
  97,517,568-byte owner is removed and production stays exact
  ([rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-rejected.json) ·
  [leaf evidence](benchmarks/results/2026-07-29-gfx1100-laguna-q2-xl-q6-k-f16-rocblas-candidate.json)).
  A clean post-H4 apples-to-apples refresh measures exact hipEngine
  **169.516 tok/s** versus llama.cpp HIP **694.184 tok/s (4.095x)**. Its cached
  trace is **3,001.692/3,016.780 ms** kernel sum/span with only **0.500%**
  outside kernels; Q5 exact coltile alone owns **1,270.458 ms / 42.325%**.
  WPF-H5A's separately registered exact-value F32 Q5 producer/SGEMM leaf now
  moves the role-qualified 235-call family **1,256.936 -> 221.137 ms
  (5.684x)** by HIP events, corroborated by **5.273x** synchronized wall. The
  regressive N48 gate remains exact fallback; all candidate outputs are finite
  at max mean KL **1.59e-9** and top-1 **100%**. Its default-off owner passes
  natural M512 at KL **0.0003742**, but the binding 18-prompt/576-step lane
  rejects SGEMM reassociation at maximum KL **1.143627 > 0.05** despite
  **564/576 (97.917%)** top-1 and diagnostic prefill **152.359 -> 202.707 tok/s
  (1.330x)**. The owner/workspace/selector are removed and exact production is
  unchanged. H5B's existing packed F32 dense-initial hipBLASLt attention route
  clears its W7900 transfer screen: tuned selected-context leaf timing is
  **109.897 -> 62.655 ms (1.754x)**, natural M512 passes KL **0.000429** / top-1
  **100%**, and cached request tracing cuts attention **488.304 -> 60.669 ms
  (8.049x)** plus complete kernel sum **3,001.692 -> 2,603.520 ms (-13.265%)**.
  The binding extension preserves all 18 natural prompts as M512 suffixes and
  observes all **10,512** expected package-mapped candidate launches. It rejects
  QK/PV reassociation at maximum KL **0.444675 > 0.05** despite **564/576
  (97.917%)** top-1, deterministic repeats, lifecycle recovery, and diagnostic
  prefill **165.555 -> 190.103 tok/s (1.148x)** with every category positive.
  The gfx1100 capability/map/owner seam is removed; exact production remains.
  H5C/H5D then returns to exact Q5 arithmetic: a transient exact-value weight
  expansion feeds local128 ordered **8x4/4x8** consumers that preserve coltile
  K/FMA/wave/store order byte-for-byte. H5E extends that invariant to
  **4x16/8x8/16x4**, owns all eight roles, and removes universally regressive
  1x64/2x32. The final-source 235-call gate moves H5D weighted event/wall
  **1,085.630/1,040.166 -> 951.876/961.993 ms (-12.320%/-7.515%)** with the same
  bounded 150,994,944-byte plane and no persistent sidecar. H5F's 12x4 N48
  micro-policy saves another **4.224/1.989 us** per M512 request. H5G retains
  exact 8x10/16x5/8x12/12x8 on five roles; its strong changed-role gate cuts
  H5F **8.639%/7.479%** by event/wall and traces at VGPR168/200 with zero
  scratch. The H5G package-default route remains KL0/byte-exact across all 48
  boundaries, logits, K/V, repeats, and lifecycle; H5I reuses that plane for
  exact Q6 and publishes **191.713/178.080/134.411 tok/s** at 512/1K/4K. H5H
  removes all larger Q5 candidates after universal regressions and the
  constant-128 spill cliff
  ([current H5I production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-production.json) ·
  [H5G Q5 production](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-production.json) ·
  [H5H boundary rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-register-boundary-rejected.json) ·
  [post-H5G residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5g-residual.json) ·
  [post-H5I residual](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-post-h5i-residual.json) ·
  [H5I leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q6-k-f32-ordered-candidate.json) ·
  [H5C leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-ordered-candidate.json) ·
  [reprofile](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-exact-residual-reprofile.json) ·
  [H5A rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-q5-k-f32-sgemm-rejected.json) ·
  [H5B rejection](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-rejected.json) ·
  [H5B leaf](benchmarks/results/2026-07-30-gfx1100-laguna-q2-xl-f32-hipblaslt-attention-candidate.json)).
- The pinned Poolside Laguna S 2.1 Q4_K_M target is supported on gfx1151 for
  torch-free c=1 blocking/streaming generation, Poolside-v1 reasoning/tool
  parsing, and exact source-bound cached loading. Its quality-admitted
  selector-unset production prefill reaches **354.820 tok/s** at pp512
  (**353.421/355.584/354.820** across three clean repetitions), up
  **4.655x** from the preceding 76.226 tok/s default. The complete category
  lane passes at maximum KL **0.040725**, **317/320 (99.0625%)** top-1,
  neutral decode, deterministic repeats, and exact lifecycle recovery. The
  gfx1151 package combines D8/D4 resident-T16 integer-dot expert tiles,
  row-scaled hipBLASLt source-F16 projections, Q4/Q6 WMMA dense/shared
  projections, and online-softmax qrow2 global/sliding attention; exact routes
  remain rollback paths
  ([production evidence](benchmarks/results/2026-07-25-gfx1151-laguna-prefill-350-production.json)).
  Its matched BF16 DFlash B4
  drafter is supported only as an explicit library/OpenAI opt-in; true AR stays
  default because the canonical full-suite DFlash economics are `0.9469x` with
  heldout and non-code regressions.
- PARO BF16 KV has retained W7900 evidence through 128K, and **208 Ki is the
  recommended safe BF16 cap** on a physical 24 GB XTX. The all-layer 256K INT8
  layout fits its tracked-memory gate but fails Qwen3.6 fidelity. The milder
  six-BF16/four-INT8 native layout passes the tested GGUF accuracy gates, but
  fails PARO accuracy and quality-preserving 256 Ki request-scratch capacity.
  On gfx1151, explicit short-context uniform `int8_per_token_head` GGUF requests
  now support continuous c1/c2/c4/c8 ownership through rounded context 8192 by
  retaining bounded BF16 attention mirrors. That route is not default or
  memory-saving; tail4, direct/no-mirror INT8 attention, longer c>N INT8, and
  PARO INT8 remain unsupported for continuous serving. Current capacity,
  throughput, speculative-decode, and concurrency evidence is reported below
  with separate gfx1100/gfx1151 provenance and correctness gates.

This remains an alpha, single-GPU release. Production PARO native `c>1` decode
is retained only for the certified gfx1151 profile; gfx1100 remains direct-c2
only and broader PARO shapes are still gated. General app-local sessions do not
reuse resident KV, structured outputs are not grammar-constrained decoding, and
the server MTP route is explicit-only. See [the API limitations](docs/API.md#current-limitations)
and [concurrency status](docs/CONCURRENCY.md#current-answer) for the exact
boundaries.


## Hardware targets

| Backend | Hardware | Status |
| --- | --- | --- |
| `cpu_reference` | Any CPU, numpy | Correctness oracle; CI without GPU |
| `hip_gfx1100` | AMD Radeon Pro W7900 / RX 7900 XTX (RDNA3) | Active backend |
| `hip_gfx1151` | AMD Ryzen AI MAX+ 395 / Radeon 8060S (Strix Halo, RDNA3.5) | Active backend |
| `cuda_sm86` | NVIDIA Ampere consumer (3090-class) | Planned peer backend |

`backend="auto"` is the public API/server default. It maps exact `gfx1100` and
`gfx1151` detections to the matching HIP backend; unknown ROCm targets warn and
select `cpu_reference` where a CPU implementation exists. Users on nearby targets
such as `gfx1101`/`gfx1102` can force a backend with `backend="hip_gfx1100"`,
`--backend hip_gfx1100`, or `HIPENGINE_BACKEND=hip_gfx1100` after validating
correctness/performance.

On gfx1151, hipEngine sets `GPU_MAX_HW_QUEUES=1` before HIP loads because it
reduces a retained gfx11 low-power queue failure and is non-regressive at short
context. It is not a repeated-128K lifecycle guarantee: current production can
still hit the firmware/scheduler stall. A matched follow-up reproduces the stall
under both HIP 7.15 and HIP 7.13, so downgrading ROCm is not a safe workaround.
Explicit values are preserved; set `GPU_MAX_HW_QUEUES=4` before process start to
restore ROCm's documented default for diagnosis. gfx1100 is unchanged. See
[`docs/ENVS.md`](docs/ENVS.md) and the
[cross-stack lifecycle artifact](benchmarks/results/2026-07-15-gfx1151-128k-hip713-vs-715-lifecycle.json).

Wave32 is the default for `hip_gfx1100` device code; wave64 is treated as an
isolated experiment with its own gates (see
[`docs/PLAN.md`](docs/PLAN.md#rdna3-wavefront-and-scheduling-caveat)).

## Memory Usage

The clean 2026-07-13 profile-aware BF16 frontier (`5a49b16d`) directly tests
the current Qwen3.6 packed PARO model on a physical 24 GB gfx1100 card. The
automatic low-memory prefill profile makes **208 Ki the recommended safe BF16
cap** with 0.361 GiB observed headroom; 220 Ki completes but leaves only about
78 MiB and is edge-only. Separately, compact 256K all-layer INT8 (`d6504544`)
fits its tracked layout gate but fails fidelity. Native tail-four mixed KV saves
18.75% of K/V and passes the tested GGUF accuracy gates, but PARO accuracy and
the quality-preserving 256 Ki XTX request-scratch allocation reject. Accordingly,
256K remains diagnostic allocation capacity—not a supported route.

<!-- BEGIN TOPLINE:W7900_MEMORY_CAPACITY -->
| Route / profile | Hardware | Context/decode | Tracked peak | Observed device peak | Device/card margin | Capacity / quality status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| PARO BF16 KV reference | W7900, default chunks | 128K/128 | **22.124 GiB** | 21.107 GiB phase sample | n/a | Reference path |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | **208 Ki (212,992)/128** | **23.082 GiB** | **23.623 GiB** | **+0.361 GiB** | **Recommended practical safe cap** |
| PARO BF16 KV, automatic 24 GB low-memory profile | RX 7900 XTX 24 GB | 220 Ki (225,280)/128 | 23.369 GiB | **23.908 GiB** | **+0.076 GiB (~78 MiB)** | Physical pass, but **edge only—not safe cap** |
| PARO BF16 KV, default 48 GB-card profile | W7900 | 220 Ki (225,280)/128 | 24.090 GiB | at least 24.832 GiB | at most -0.848 GiB vs 24 GB card | Rejected for this larger-chunk profile |
| PARO INT8 per-token/head KV, FP16 scales | W7900 | 256K/128 | **22.971 GiB** | 21.041 GiB phase sample | +1.029 GiB tracked | **Rejected** by Qwen3.6 matched-context and task gates |
| PARO tail-four Hadamard-group32 mixed KV, BF16-oracle prefill | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.469 GiB before failed allocation** | 22.566 GiB after clean OOM | 1.418 GiB free before request scratch; insufficient | **Rejected:** `HIP error 2` OOM and PARO fidelity failure; no segfault |
| PARO tail-four Hadamard-group32 mixed KV, direct-streaming control | RX 7900 XTX 24 GB | 256 Ki (262,144)/128 request scratch | **23.290 GiB** | **23.590 GiB** live sample | **+0.394 GiB** live | Allocation passes, but direct packed prefill is **correctness-rejected** |

The native explicit `tail4_hadamard_group32` layout keeps K/V for
full-attention layers `3,7,11,15,19,23` in BF16 and stores only layers
`27,31,35,39` as Hadamard-group32 INT8 with FP16 scales. At 262,400 retained
rows it uses `4,366,336,000` K/V bytes—**18.75% below BF16**—with no persistent
BF16 shadow. PARO's quality-preserving prefill uses a temporary BF16 oracle;
GGUF's post-quality layout audit reports zero persistent oracle/mirror buffers.
Native PARO still fails 1/11 prompts at 512/8 and 2/11 at 4K/16 (58.82%
worst-prompt top-1), and its 256 Ki quality-preserving request scratch OOMs.

The clean `c971262f` therock-7.15 GGUF-only closure passes all 11 prompts at
512/8 (max KL `0.007455`, top-1 100%) and 4K/16 (mean/max KL
`0.0001369/0.009926`, aggregate/minimum-prompt top-1 `99.47%/94.12%`) plus
bounded `mixed_v1` at 128K/16 (max KL `5.19e-5`, top-1 100%). At 128K,
persistent K/V is `2,185,297,920` bytes versus BF16 `2,689,597,440` bytes and
live owned memory falls `24.168 -> 23.698 GiB`. It still rejects promotion:
production 4K prefill/decode regress `0.67%/0.75%`, one-shot 128K decode
regresses `3.82%`, and production prefill allocates then frees
`1,075,838,976` bytes—byte-exact to four BF16 layer caches—raising allocator
high water `24.168 -> 24.700 GiB`. The transient attribution is inferred from
the exact bytes; it is not a persistent shadow. The policy remains explicit
and non-default. Evidence:
`benchmarks/results/2026-07-15-gfx1100-gguf-tail4-hadamard-clean-gate.json` and
`benchmarks/results/2026-07-14-gfx1100-native-tail4-hadamard-kv-outcome.json`.
<!-- END TOPLINE:W7900_MEMORY_CAPACITY -->

The INT8 layout retains 2,686,976,000 payload bytes plus 20,992,000 FP16 scale
bytes across ten full-attention layers and no BF16 K/V shadow. Its final
BF16-reference-token matched 128K/16 gate rejects at mean/max KL
`0.85128/4.97382` and 41.18% top-1 agreement. Format and mixed-policy screens
did not find a candidate that transferred through 4K.

The former llama.cpp Q8_0 pass is now a repeated-token saturation control, not
representative quality evidence. On identical Q4_K_M weights at exact mixed
4K/16, native Q8_0 rejects at mean/max KL `0.075654/1.26009` despite 94.12%
top-1; F16/F16 is exactly zero. K-only and V-only Q8 reach `0.096682` and
`0.243219` mean KL, while full Q8 benefits from non-additive K/V cancellation.
The repeated full-Q8 control is only `0.00000619` KL, confirming prompt content
as the dominant difference.

hipEngine shows the same protocol effect. Host per-head/group32/Hadamard all
pass repeated 4K/16 near `0.000002` KL but reject mixed at
`0.12779/0.28106/0.25180`. Pure native per-head INT8 rejects mixed at
`0.19038/2.99555`, 88.24% top-1, with all ten layers INT8 and no BF16 mirror.
Direct arithmetic is therefore not a universal fidelity repair. The separate
same-weight hipEngine-GGUF-BF16 versus llama.cpp-F16 bridge preserves 100%
top-1; its `0.26606` all-position mean KL is prompt-final dominated, while 16
decode rows average `0.000510`.

The original five-category free-generation reference is unscorable. In the
replacement restricted-choice diagnostic, INT8 flips one of two
BF16-qualified 4K answers (multihop `D -> C`) but retains all three qualified
32K answers. This shows that large KL can change a bounded functional decision
without implying every answer changes; it remains partial evidence, not support
for 256K INT8. Memory was measured once; timing is diagnostic.

See the
[`capacity/fidelity outcome`](benchmarks/results/2026-07-13-w7900-paro-int8-kv-accuracy-outcome.json),
[llama.cpp repeated-token Q8_0 control](benchmarks/results/2026-07-13-w7900-llamacpp-q8-kv-matched-quality.json),
[repeated/mixed prompt and native arithmetic isolation](benchmarks/results/2026-07-13-w7900-gguf-q8-kv-protocol-arithmetic-isolation.json),
[same-weight GGUF bridge](benchmarks/results/2026-07-13-w7900-gguf-llamacpp-matched-parity.json),
[bounded functional check](benchmarks/results/2026-07-13-w7900-paro-int8-kv-functional-mc.json),
[format screen](benchmarks/results/2026-07-13-w7900-paro-kv-format-ablation.json),
and [policy screen](benchmarks/results/2026-07-13-w7900-paro-kv-policy-ablation.json).

### llama.cpp configuration note

The repository has no compact artifact or source revision for the former
llama.cpp Q8_0 memory tables, so those numbers are not toplines. The tested
configuration was:

```bash
--flash-attn on -ctk q8_0 -ctv q8_0 -c 262144 -b 128 -ub 128
```

A replacement capacity table must record the GGUF fingerprint, llama.cpp
commit/build, GPU, full command, and whole-card sampling artifact.

## Model Performance

### gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)

**Status: retained.** The GGUF column is the clean 2026-07-16 final
selector-unset BF16-KV sweep at `28b37356` on therock HIP 7.15: independent
right-sized sessions, one discarded warmup, three measured runs, and production
graph decode. All 18 GGUF final IDs are exact and maximum prefill/decode stdev
over median is `0.658%/0.223%`. PARO and llama.cpp retain their clean July 12
protocols. PARO is W4 PARO/BF16 KV; the other columns use Q4_K_M with BF16/F16
KV, so bold values are descriptive rather than same-quant wins. GGUF prefill now
beats llama.cpp HIP at every shape and Vulkan through 64K; GGUF decode beats HIP
everywhere and is closest to Vulkan at 4K (`-2.47%`).

<!-- BEGIN TOPLINE:W7900_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2917.732** | 2716.648 | 2412.320 | 2627.990 |
| 1K/128 | 2995.876 | **3052.541** | 2389.670 | 2631.750 |
| 4K/128 | 2943.038 | **2953.101** | 2255.080 | 2521.770 |
| 32K/128 | **2108.868** | 2078.038 | 1667.640 | 1943.920 |
| 64K/128 | **1584.131** | 1559.878 | 1291.820 | 1414.470 |
| 128K/128 | 1056.252 | 1037.378 | 891.949 | **1079.280** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.599** | 92.833 | 80.756 | 107.786 |
| 1K/128 | 103.238 | 98.148 | 80.805 | **107.555** |
| 4K/128 | **105.943** | 100.522 | 79.768 | 103.066 |
| 32K/128 | **92.438** | 88.240 | 74.304 | 91.835 |
| 64K/128 | 78.260 | 76.691 | 69.010 | **83.746** |
| 128K/128 | 60.663 | 62.669 | 60.933 | **70.833** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.144** | 21.228 | 21.606 | 21.260 |
| 1K/128 | **18.367** | 21.295 | 21.618 | 21.220 |
| 4K/128 | **19.161** | 21.670 | 21.674 | 21.278 |
| 32K/128 | **19.864** | 22.234 | 22.216 | 21.855 |
| 64K/128 | **20.403** | 22.879 | 22.895 | 22.512 |
| 128K/128 | **22.124** | 24.168 | 24.089 | 23.824 |
<!-- END TOPLINE:W7900_SWEEP -->

W7900 row sources: [final hipEngine GGUF sweep](benchmarks/results/2026-07-16-gfx1100-gguf-final-optimization-sweep.json),
[July 12 accepted summary](benchmarks/results/2026-07-12-w7900-v030-8116c453-summary.json),
[hipEngine PARO](benchmarks/results/2026-07-12-w7900-v030-8116c453-hipengine-paro-packed-5run.json),
[superseded July 12 hipEngine GGUF](benchmarks/results/2026-07-12-w7900-v030-8116c453-hipengine-gguf-q4km-5run.json),
[llama.cpp HIP](benchmarks/results/2026-07-12-w7900-v030-8116c453-llamacpp-hip-q4km-f16kv.json),
[llama.cpp Vulkan](benchmarks/results/2026-07-12-w7900-v030-8116c453-llamacpp-vulkan-q4km-f16kv.json),
and [W7900 correctness oracle](benchmarks/results/2026-07-12-w7900-v030-gguf-eager-p512-d4.json).

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

> Thanks to Framework for sending a dedicated Framework Desktop Strix Halo motherboard for this profiling and tuning work.

**Status: current IOMMU-off refresh retained through 64K; repeated GGUF 128K
blocked.** The clean 2026-07-17 table at `2edbb2ee` refreshes PARO, GGUF, and
both llama.cpp backends under `amd_iommu=off`. GGUF 512-64K passes clean
provenance, finite logits, exact final IDs, and the 5% variance gate; maximum
prefill/decode stdev over median is **0.122%/0.028%**, and all 15 IDs are
`9707`.

Relative to the previous published IOMMU-on rows, the arithmetic mean change
across 11 eligible hipEngine cells is **+4.60% prefill / +6.20% decode**; GGUF
alone averages **+8.84% / +5.84%**. This is directional, not causal, because
the hipEngine revision/routing also changed; a same-commit reboot A/B remains
necessary. The setting leaves zero IOMMU groups and disables the XDNA/NPU
driver. GGUF 128K still times out after a 584.059 tok/s warmup and 583.464 tok/s
measured pass, so no stale 128K number is carried forward. Bold values remain
descriptive because quant/KV types and memory scopes differ.

<!-- BEGIN TOPLINE:GFX1151_SWEEP -->
#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 1298.259 | **1395.379** | 1184.628 | 1161.498 |
| 1K/128 | 1332.199 | **1481.943** | 1192.768 | 1154.327 |
| 4K/128 | 977.252 | **1444.733** | 1148.155 | 1114.081 |
| 32K/128 | 827.350 | **1132.215** | 843.252 | 873.573 |
| 64K/128 | 690.642 | **892.663** | 632.774 | 702.742 |
| 128K/128 | 498.101 | — (blocked) | 432.033 | **499.728** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **70.750** | 52.761 | 53.222 | 63.795 |
| 1K/128 | **65.905** | 54.658 | 53.044 | 63.391 |
| 4K/128 | **66.728** | 55.297 | 52.338 | 61.863 |
| 32K/128 | **53.458** | 45.983 | 45.946 | 52.286 |
| 64K/128 | 44.793 | 39.388 | 40.353 | **45.160** |
| 128K/128 | 32.615 | — (blocked) | 32.728 | **35.569** |

#### Peak memory GiB

| Workload | hipEngine PARO | hipEngine GGUF | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **18.039** | 21.478 | 21.375 | 21.551 |
| 1K/128 | **18.051** | 21.710 | 21.387 | 21.501 |
| 4K/128 | **19.026** | 22.995 | 21.444 | 21.507 |
| 32K/128 | **19.716** | 23.559 | 21.987 | 22.191 |
| 64K/128 | **20.344** | 24.203 | 22.666 | 22.627 |
| 128K/128 | **21.881** | — (blocked) | 23.862 | 24.254 |
<!-- END TOPLINE:GFX1151_SWEEP -->

The memory columns have different scopes: hipEngine reports tracked allocator
high-water, while llama.cpp reports absolute whole-device amdgpu GTT used,
sampled every 10 ms. Use them for within-column context growth, not small
cross-column allocator comparisons. Row source: [`current IOMMU-off refresh and
128K blocker`](benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-topline-refresh.json).
Exact settings and gates are in the canonical
[`benchmarks/README.md`](benchmarks/README.md#gfx1151-model-throughput).

### Current gfx1151 GGUF decode baselines

These are separate exact repeated-token SOL-G4/G5 controls. The model sweep
above excludes graph capture from steady decode throughput; SOL-G5 charges one
capture/instantiate and destroy to each 128-token window.

<!-- BEGIN TOPLINE:GFX1151_GGUF_EAGER -->
| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| GGUF eager c1 | Radeon 8060S/gfx1151; Qwen3.6-35B-A3B UD-Q4_K_M; BF16 KV; `[9707] * 512`; TheRock HIP 7.15; TuneD accelerator-performance; clean scalar/candidate/scalar, 1 discarded + 4 measured runs per leg; 128 eager steps; graph off | **48.850 tok/s** (`20.471 ms/token`), **+0.309%** vs clean scalar control | Retained for this exact repeated-token protocol; control/candidate ranges do not overlap, every output ID is 9707, and the G1 hidden/state/KV oracle is linked |
| GGUF state-bound graph c1 | Radeon 8060S/gfx1151; same current model/KV/prompt/stack; 1 warmup + 4 measured rotating same-session runs; 128 steps; capture and destroy charged | **48.704 tok/s** (`20.532 ms/token`), **-0.293%** vs same-run eager; **+0.201%** vs scalar graph | Exact 128/128 state/KV/token replay, but current G5 rejects a graph-over-eager speed claim; graph default policy is tracked separately |
<!-- END TOPLINE:GFX1151_GGUF_EAGER -->

Artifacts: [`SOL-G4 eager audit`](benchmarks/results/2026-07-11-sol-g4-gfx1151-gguf-eager-decode-audit.json)
and [`SOL-G5 production graph audit`](benchmarks/results/2026-07-11-sol-g5-gfx1151-gguf-decode-graph-production-audit.json).

See [`benchmarks/README.md`](benchmarks/README.md) for the platform freshness
index, exact settings, run commands, and evidence status.

## Speculative decode (DFlash / MTP)

Every displayed route has its own same-protocol AR control. The exact/default
and `llama-compat` columns are separate because only `llama-compat` shares the
B2 natural24 structure used by the llama.cpp comparison.

<!-- BEGIN TOPLINE:SPECULATIVE -->
#### GGUF MTP comparison, Radeon Pro W7900/gfx1100

| Metric | hipEngine GGUF true AR | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP base AR | llama.cpp HIP bundled MTP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Route | State-bound graph, no MTP | B3, fixed 10 cycles | B2 natural24, reusable B1/B2 target graphs | Natural25 request / 24 timed transitions | B2, natural25 request / 24 timed transitions |
| Decode | **98.75 tok/s fixed / 96.75 tok/s natural24** | 68.50 tok/s | **122.67 tok/s** | 78.05 tok/s transition-normalized | 115.44 tok/s transition-normalized |
| Own true AR | same route | 98.75 tok/s | 96.75 tok/s | same route | 78.05 tok/s |
| MTP / own AR | 1.0000x | **0.6936x** | **1.2679x** | n/a | **1.4791x** |
| Draft acceptance | n/a | 73.53% | 80.45% | n/a | 81.56% |
| Accepted draft/output | n/a | 50.00% | 60.00% | n/a | 58.40% |
| Complete wall per output/transition | 10.336 ms natural24 | 14.696 ms | **8.186 ms** | 12.812 ms | 8.662 ms |
| State/commit contract | serial autoregressive | serial-prefix preserving | direct partial commit/dp4a; accuracy-traded | native llama.cpp autoregressive | native llama.cpp compatibility target |

The W7900 route now reuses one fixed-address target graph per B1/B2 shape
bucket. Live token, position, context, and cursor metadata are staged on device;
the five two-row output-cap tails use B1 and the four true one-row/no-draft
cycles stay on AR. Unsupported configurations fall back before launch, while a
post-launch failure never re-executes a possibly mutating verifier.

Two clean full-suite processes at `0d7b86e7` measure **123.33 and 122.67 tok/s**
(0.54% spread). The conservative run is **1.2679x** its true graph AR and
**6.26% faster** than llama.cpp's **115.44 tok/s / 8.662 ms-transition** floor,
while complete wall is **5.50% lower** at **8.186 ms/output**. Draft acceptance
and accepted/output remain exactly **80.45% / 60.00%**. All 240 output IDs and
all 96 cycle semantics in both runs match the prior eager-target
`llama-compat` baseline. hipEngine uses BF16 KV versus llama.cpp F16 KV, and the
external row remains `performance_claim=false` because its preserved
instrumentation checkout is dirty.

The target graph also passes the real 35B oracle at two B2 positions plus B1:
target top-1, 16,384 FP32 hidden values, each set of 60 captured and 60 resident
Conv/GDN buffers, all 20 K/V buffers, and cursors are byte-exact. A cached
six-step trace records zero measured recaptures, **18.67 ms host / 13.67 ms
kernels / 5.00 ms residual**, and the expected dynamic-metadata, cursor-advance,
and top-1 widening leaves. The prior eager profiler residual was 38.41 ms.

Exact/default remains the semantic control. `llama-compat` remains explicit-only
because direct partial commit is not serial-prefix-equivalent; this retained
speed result does not make it the automatic exact route. The fixed-cycle exact
and natural24 compatibility rows remain different protocols.

##### W7900 reusable-native `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | `llama-compat` tok/s | MTP / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 96.75 | **122.67** | **1.2679x** | 80.45% | 60.00% | 8.186 ms |
| Train | 6 | 96.13 | **124.70** | **1.2973x** | **87.25%** | 61.81% | 8.052 ms |
| Heldout | 4 | 97.68 | **119.73** | **1.2257x** | **71.43%** | 57.29% | 8.388 ms |
| `code` | 4 | 97.06 | **127.81** | **1.3168x** | 93.94% | 64.58% | 7.854 ms |
| `general_en` | 2 | 94.25 | **123.37** | **1.3091x** | 75.68% | 58.33% | 8.138 ms |
| `general_ja` | 2 | 98.04 | **118.42** | **1.2079x** | 69.23% | 56.25% | 8.480 ms |
| `mixed_ja_en` | 2 | 97.40 | **116.78** | **1.1990x** | 72.97% | 56.25% | 8.604 ms |

Every category and the heldout split beat their true same-protocol AR control;
even the slowest category remains above the aggregate external floor in the
conservative run. The corrected 54.88 tok/s eager-target row remains the
optimization baseline, not the current route. Artifacts:
[`retained reusable route`](results/2026-07-19-w7900-llama-compat-reusable-native-cycle.json),
[`N2 ownership diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n2.json),
[`N3 complete-cycle diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n3.json),
[`N3P proposal-submission diagnostic`](results/2026-07-19-w7900-llama-compat-native-cycle-n3p.json),
[`prior baseline`](results/2026-07-19-w7900-hipengine-llama-compat-current-baseline.json),
and [`llama.cpp floor`](results/2026-07-19-w7900-llamacpp-mtp-natural25-refresh.json).

#### GGUF MTP comparison, Radeon 8060S/gfx1151

| Metric | hipEngine GGUF exact/default | hipEngine GGUF `llama-compat` | llama.cpp HIP |
| --- | ---: | ---: | ---: |
| Route | B5, fixed 10 cycles | B2 natural24, NativeSpecCycle N3 | B2, natural25 request / 24 timed transitions |
| Canonical/native MTP decode | 56.39 tok/s (0.9895x own AR) | **80.10 tok/s (1.4282x own AR)** | 70.99 tok/s native (1.3530x own AR; not cross-engine comparable) |
| Cross-engine MTP decode-transition rate | n/a: fixed-cycle horizon | **80.10 tok/s** | 68.15 tok/s |
| Cross-engine own AR transition rate | n/a: fixed-cycle horizon | **56.09 tok/s** | 50.37 tok/s |
| Cross-engine MTP / own AR | n/a | 1.4282x | 1.3530x |
| Draft acceptance | 72.33% | 77.72% | 79.56% |
| Accepted draft/output | 53.49% | 59.58% | 57.60% |
| Full-cycle/predicted wall per counted output or timed transition | 17.808 ms/output | 12.551 ms/output | 14.673 ms/transition |
| State/commit contract | exact/default, serial-prefix preserving | N3 complete public cycle; accuracy-traded | native llama.cpp compatibility target |

The IOMMU-off exact/default B5 route remains the current semantic control at
**56.39 vs 56.98 true-AR tok/s (0.9895x)**. `llama-compat` is separate,
explicit-only, and not serial-prefix-equivalent. On current main, registering
the reusable gfx1151 target graph moves the clean direct-commit control
**70.020 -> 80.132 tok/s (+14.44%)**; N3 public complete-cycle ownership retains
**80.099 tok/s (+14.39%)** and cuts complete wall **14.314 -> 12.551 ms/output
(-12.32%)**. N3 is only **0.042%** below target-only N1.

All **240 output IDs / 97 cycle semantics** match across clean control, N1, and
N3, with unchanged **77.72% draft acceptance / 59.58% accepted-output**. The
prior clean `2edbb2ee` direct-commit row remains slightly higher at **81.90
tok/s** (-2.20% versus current N3), but it is a different revision/run and no
source regression is attributed. Against the preserved transition-normalized
llama.cpp context, current N3 is **80.10 vs 68.15 tok/s (+17.53%)**; BF16 versus
F16 KV and the dirty preserved llama.cpp source remain disclosed.

##### gfx1151 NativeSpecCycle N3 `llama-compat` full-suite gate

| Scope | Prompts | True AR tok/s | N3 tok/s | N3 / AR | Draft acceptance | Accepted/output | Cycle wall/output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full | 10 | 56.09 | **80.10** | **1.4282x** | 77.72% | 59.58% | 12.551 ms |
| Train | 6 | 55.97 | **80.91** | **1.4457x** | **82.08%** | 60.42% | 12.429 ms |
| Heldout | 4 | 56.26 | **78.91** | **1.4025x** | **71.79%** | 58.33% | 12.733 ms |
| `code` | 4 | 56.12 | **86.08** | **1.5338x** | 91.04% | 63.54% | 11.684 ms |
| `general_en` | 2 | 57.26 | **78.98** | **1.3795x** | 71.79% | 58.33% | 12.716 ms |
| `general_ja` | 2 | 55.61 | **75.12** | **1.3509x** | 69.23% | 56.25% | 13.388 ms |
| `mixed_ja_en` | 2 | 55.35 | **75.66** | **1.3669x** | 69.23% | 56.25% | 13.282 ms |

Every category and the heldout split beats its true same-protocol AR control and
improves versus the clean current-main direct-commit route by **9.91% to
19.45%**. The real 35B N1/N2 oracle passes target IDs, FP32 hidden rows, all 60
Conv/GDN and 20 full-KV buffers, selected commits, and cursors. The six-step
cached trace records zero recaptures, **24.891 ms host / 21.674 ms kernels /
3.218 ms residual**, 940 calls/step, and the expected zero-scratch metadata
leaf. N3P remains unregistered on gfx1151 because it is not needed for this win
and was not the gfx1100 topline. Artifact:
[`gfx1151 NativeSpecCycle transfer`](results/2026-07-19-gfx1151-llama-compat-native-cycle-transfer.json).

#### Dense PARO DFlash

| Path | Platform and protocol | Result | Evidence status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | W7900/gfx1100; Qwen3.6-27B PARO target plus Qwen3.6-27B DFlash drafter; 9 prompts; 64 decode tokens | 40.10 vs 32.57 AR tok/s, **1.231x** | Retained under the recorded DFlash gate; source tree was dirty and must be refreshed before changing the claim |
<!-- END TOPLINE:SPECULATIVE -->

Artifacts: [`W7900 GGUF MTP transfer`](benchmarks/results/2026-07-12-w7900-gfx1100-gguf-mtp-transfer.json),
[`DFlash`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json),
and [`current gfx1151 IOMMU-off MTP refresh`](benchmarks/results/2026-07-17-gfx1151-amd-iommu-off-mtp-refresh.json).
Historical gfx1151 controls remain linked from the canonical benchmark record.
Historical hipEngine OpenAI MTP server rows are excluded. The current raw-ID
route counts exact completion IDs across every choice and owns batch timing
once. The
corrected 2026-07-11 server matrix finds that compatibility MTP changes true-AR
IDs even at c1, so it must remain explicit-only despite diagnostic c1/c2 speed
gains; SOL-S1 routes automatic requests to exact/default AR while keeping the
compatibility hook explicit-only. See the
[`route-gate artifact`](benchmarks/results/2026-07-11-sol-s1-gfx1151-server-auto-route-gate.json)
and canonical [`benchmarks/README.md`](benchmarks/README.md#gfx1151-gguf-server-automatic-route-gate-2026-07-11).

The clean gfx1151 PARO DFlash S4 profile is exact but not competitive:
`9.68` versus `65.27 tok/s` AR (`0.148x`) at B4/32 tokens. Branch-copy is
faster but diverges at generated token 1, and fused target LM-head is 5.16%
slower than unfused. See the
[`compact profile`](benchmarks/results/2026-07-11-sol-s4-gfx1151-paro-dflash-profile.json)
and the canonical
[`benchmark analysis`](benchmarks/README.md#gfx1151-paro-dflash-s4-profile-2026-07-11).

## Concurrency

Current GGUF direct-model-step tables are retained separately for gfx1100 and
gfx1151. Both have exact native c2/c4/c8 graph routes, direct throughput, and
live OpenAI membership. gfx1151 additionally retains occupancy-adaptive GGUF
serving, explicit short mirrored-INT8 c1/c2/c4/c8, and production PARO
c2/c4/c8; gfx1100 PARO remains direct-c2 only. See
[`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) for the exact boundaries.

The linked records keep gfx1100 and gfx1151 separate because the model files,
ROCm stacks, and comparison backends differ. *Aggregate* is total tok/s across
the batch; *per-sequence* is tok/s seen by one request. See
[`docs/VLLM_RDNA3.md`](docs/VLLM_RDNA3.md) for vLLM RDNA3 setup notes.

### gfx1100 / W7900 direct and server GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct native-c4/c8 model-step throughput and retained real
OpenAI SSE arbitrary-C server scaling.** All rows use `UD-Q4_K_M`, BF16 KV,
greedy top-1, W7900/gfx1100, and TheRock HIP 7.15. Timing scopes stay separate:
direct rows time synchronized graph steps; server rows time complete concurrent
SSE cycles including admission, prompt work, decode, delivery, and completion.

<!-- BEGIN TOPLINE:W7900_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / serial-c4 | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 85.469 | 85.469 | 1.000x | 1.009x | 0.209 / 0.209 s | 11.693 / 11.955 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 127.427 | 63.714 | 1.491x | 1.504x | 0.951 / 0.954 s | 15.765 / 16.023 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 184.575 | 46.144 | 2.160x | 2.178x | 2.020 / 2.023 s | 21.715 / 22.021 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **246.872** | **30.859** | **2.888x** | **2.913x** | **3.475 / 3.479 s** | **32.414 / 32.749 ms** | **25.401 GiB** |
| chunked c8 control | 8 | 2x c4, serialized | 183.020 | 22.878 | 2.141x | 2.160x | 3.055 / 4.084 s | 43.767 / 44.281 ms | 26.069 GiB* |
| serial-c4 rate control | 4 | 4x c1, serialized | 84.738 | 21.185 | 0.991x | 1.000x | 0.548 / 0.877 s | 47.225 / 48.142 ms | 26.985 GiB* |

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| logical-c1 control | 1 | masked physical c8 | 25.583 | 25.583 | 1.000x | 0.807x | 5.003 s | 0.271 / 0.276 s | 36.499 / 39.809 ms | 29.312 GiB |
| physical c8 | 8 | 1x c8 | **136.122** | 17.015 | **5.321x** | 4.293x | 7.523 s | 1.751 / 2.088 s | 41.712 / 45.068 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + sparse c8 | 88.592 | 9.844 | 3.463x | 2.794x | 13.003 s | 1.709 / 2.235 s | 81.087 / 88.112 ms | 31.969 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **111.380** | **8.568** | **4.354x** | **3.513x** | **14.940 s** | **1.886 / 3.323 s** | **87.502 / 93.631 ms** | **32.869 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 31.708 | 2.439 | 1.239x | 1.000x | 52.479 s | 2.424 / 3.390 s | 382.821 / 396.004 ms | 32.869 GiB* |
<!-- END TOPLINE:W7900_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and median
of three; one physical c8 is **2.888x** c1 and **+34.89%** over c4+c4, with a
**748 packed-native / 0 row-local / 0 copy** trace. Server protocol uses 512
exact prompt IDs and 128 generated outputs/request, a 20 ms admission window,
one discarded plus three measured bursts, and scheduler latency. Logical c1 is
honestly a masked physical-c8 production control; C9/C13 are multiple declared
buckets, never wider native widths. All **189/189** server requests match
resident prompt IDs, direct-c1 outputs, usage, and finish metadata. Grouped C13
is **4.354x** logical-c1 and **3.513x** serial; one exact c8→c13 live trace emits
**1,664/1,664** IDs at **107.284 aggregate tok/s** and drains ownership to zero.
Starred server memory is cumulative in one prepared process.

Artifacts: [`C4`](benchmarks/results/2026-07-16-gfx1100-gguf-concurrency-c4-native-graph-scaling-closure.json),
[`E2 native c8`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e2-native-c8-scaling-closure.json),
[`E3 arbitrary C`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-e3-arbitrary-c-correctness.json), and
[`F1 real server`](benchmarks/results/2026-07-17-gfx1100-gguf-concurrency-f1-server-scaling-closure.json).
Historical mixed-quant/mixed-scope results remain in
[`benchmarks/HISTORY.md`](benchmarks/HISTORY.md).

### gfx1151 / Radeon 8060S PARO direct c2/c4/c8 and production shape catalog (Qwen3.6 35B-A3B, 512/128)

**Status: direct and resident OpenAI c2/c4/c8 are retained; gfx1151 selects
them by package default.** Equivalent clean tree `e175e28f` (pushed as
`8c8cc15e`) generalizes the exact c2 route into true physical c4/c8 without c2
stacking. G5 attaches those identity-matched widths to one shared stable-slot
owner for public `LLM`, blocking OpenAI, and concurrent SSE. Explicit legacy
`=0` flags remain rollback opt-outs.

Three p512/d128 processes per width pass all **5,754/5,754** recorded IDs plus
all-layer state/KV, sparse lifecycle, ten-prompt category/heldout, primitive,
and cached-profiler gates.

| Explicit direct route | Median aggregate decode | Per-request decode | Classification |
| --- | ---: | ---: | --- |
| c1 graph | **70.810 tok/s** | 70.810 tok/s | independent reference |
| serial c2 bridge | **65.574 tok/s** | 32.787 tok/s | exact fallback control |
| native selected-batch c2 | **79.237 tok/s** | 39.619 tok/s | **retained direct c2; 1.1190x c1 / 1.2084x serial** |
| true physical c4 | **100.209 tok/s** | 25.052 tok/s | **retained direct c4; 1.4152x c1** |
| true physical c8 | **99.943 tok/s** | 12.493 tok/s | **retained direct c8; 1.4114x c1** |

The production table below uses the clean blocking F1 wall: 512 raw prompt IDs,
128 generated IDs/request, one warmup plus three measured bursts, and a fresh
server per width. All **68/68** warmup/measured/live rows are exact; c1/c2/c4/c8
scale to **47.124/51.962/60.323/61.253 aggregate tok/s** with <=0.994% variance.
The complementary exact-roundtrip SSE packet keeps all **100/100** rows exact at
**36.327/38.666/42.471/41.487/35.633 tok/s** for c1/c2/c4/c8/serial-c8; native
c8 is **1.164x** serial and live c4->c8 admission is **38.191 tok/s**. A 1+7 c8
stress adds **72/72** exact rows, and a no-native-flag OpenAI c4 gate run from
`/tmp` loads the packaged profile, observes physical widths 2/4, and records no
fallback.

<!-- BEGIN TOPLINE:GFX1151_PARO_CURRENT -->
| Client c | Production backend groups | Exact classification | Retained OpenAI aggregate |
| ---: | --- | --- | ---: |
| 1 | `1` | c1 oracle / accepted | **47.124 tok/s** |
| 2 | native `2` | retained physical width | **51.962 tok/s** |
| 3 | `2+1` | exact partition; not native c3 | no separate claim |
| 4 | native `4` | retained physical width | **60.323 tok/s** |
| 5 | `4+1` | exact partition; not native c5 | no separate claim |
| 6 | `4+2` | exact partition; not native c6 | no separate claim |
| 7 | `4+2+1` | exact partition; not native c7 | no separate claim |
| 8 | native `8` | retained physical width | **61.253 tok/s** |
<!-- END TOPLINE:GFX1151_PARO_CURRENT -->

Protocol: W4 PARO/BF16 KV, 40 layers, 8 warmup decode steps, 128 measured
decode steps, greedy sampling, TheRock HIP 7.15, one hardware queue, TuneD
`accelerator-performance`, and `amd_iommu=off`. All direct-width process
variance is <=0.054%. c4/c8 are **+41.52%/+41.14% vs c1**; c8 aggregate is
**0.265% below c4**, while its median model-step time is **0.183% faster than
two c4 steps**. c3/c5/c6/c7 retain no native-width claim. See the
[retained direct-c2/c4/c8 artifact](benchmarks/results/2026-07-18-gfx1151-paro-g3-native-c248-direct-retained.json),
[G5 blocking F1](benchmarks/results/2026-07-18-gfx1151-paro-g5-f1-server-scaling.json),
[G5 SSE](benchmarks/results/2026-07-18-gfx1151-paro-g5-sse-server-scaling.json),
[c8 repeatability](benchmarks/results/2026-07-18-gfx1151-paro-g5-c8-sse-repeatability.json),
[package-default OpenAI c4](benchmarks/results/2026-07-18-gfx1151-paro-g5-default-openai-c4.json),
and the [canonical run record](benchmarks/README.md#paro-concurrency-and-production-routing).

### gfx1151 / Radeon 8060S direct and server GGUF concurrency (Qwen3.6 35B-A3B, 512/128)

**Status: retained direct native-c2/c4/c8 model steps and c1-preserving
occupancy-adaptive OpenAI SSE serving.** F2 maps only ephemeral execution rows
into c1/c2/c4/c8 while stable scheduler, state, and KV ownership stays fixed.
F3 adds exact singleton-indexed packed-AR GDN: direct c2/c4/c8 improve
**+8.71%/+5.25%/+4.04%**, while c1 is structurally unchanged. The prior F2
server packet remains retained but was not remeasured for this direct-only F3
refresh. All rows use `UD-Q4_K_M`, BF16 KV, greedy top-1, one HIP queue, and the
active `amd_iommu=off` boot. Direct/category and E3 gates retain **188,080** and
**134,160** exact hidden comparisons; the c8 trace is **748 packed-native / 0
row-local / 0 copies**, with diagnostic Conv/GDN time down **50.94%**.

<!-- BEGIN TOPLINE:GFX1151_CONCURRENCY -->
| Direct route | Logical C | Native groups | Aggregate decode tok/s | Per-request tok/s | Aggregate / c1 | Aggregate / retained serial-c4† | TTFT p50 / p95 | Model-step ITL p50 / p95 | Tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| direct c1 | 1 | 1x c1 | 50.335 | 50.335 | 1.000x | 1.002x | 0.370 / 0.372 s | 19.874 / 20.160 ms | 21.783 GiB |
| direct c2 | 2 | 1x c2 | 78.552 | 39.276 | 1.561x | 1.564x | 2.177 / 2.177 s | 25.459 / 25.820 ms | 22.394 GiB |
| direct c4 | 4 | 1x c4 | 108.050 | 27.013 | 2.147x | 2.151x | 3.394 / 3.403 s | 37.026 / 37.399 ms | 23.396 GiB |
| **direct c8** | **8** | **1x c8** | **133.251** | **16.656** | **2.647x** | **2.653x** | **6.841 / 6.841 s** | **60.004 / 60.641 ms** | **25.401 GiB** |
| chunked c8 control† | 8 | 2x c4, serialized | 102.724 | 12.841 | 2.043x | 2.045x | 5.089 / 6.787 s | 77.902 / 78.467 ms | 26.069 GiB* |
| serial-c4 rate control† | 4 | 4x c1, serialized | 50.235 | 12.559 | 0.999x | 1.000x | 0.927 / 1.485 s | 79.643 / 80.637 ms | 26.985 GiB* |

† Controls are retained from clean pre-F3 `ef46ee8c`. The serial-c4 c1 path is
structurally unchanged and remains the ratio reference; chunked c8 is historical
because the F3 candidate would also change each physical c4 group.

| Real OpenAI SSE route | Logical C | Physical execution | Aggregate generated tok/s | Per-request tok/s | Aggregate / logical-c1 | Aggregate / serial-c13 | Cycle wall p50 | Scheduler TTFT p50 / p95 | Scheduler ITL p50 / p95 | Cumulative tracked peak |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| occupancy-adaptive c1 | 1 | 1x native c1 | 43.033 | 43.033 | 1.000x | 0.999x | 2.974 s | 0.417 / 0.417 s | 19.759 / 20.057 ms | 29.046 GiB |
| physical c8 | 8 | 1x c8 | **86.942** | 10.868 | **2.020x** | 2.019x | 11.778 s | 2.614 / 3.347 s | 64.949 / 68.260 ms | 30.805 GiB* |
| grouped c9 | 9 | c8 + c1 | 77.302 | 8.589 | 1.796x | 1.795x | 14.903 s | 2.734 / 3.783 s | 85.747 / 91.143 ms | 31.035 GiB* |
| **grouped c13** | **13** | **c8 + sparse c8** | **73.235** | **5.633** | **1.702x** | **1.701x** | **22.721 s** | **3.624 / 5.526 s** | **132.547 / 141.787 ms** | **32.386 GiB*** |
| serial-c13 bridge | 13 | 13x c1 serial | 43.066 | 3.313 | 1.001x | 1.000x | 38.638 s | 3.386 / 5.390 s | 257.799 / 271.455 ms | 32.386 GiB* |
<!-- END TOPLINE:GFX1151_CONCURRENCY -->

Direct protocol uses 128 decode transitions, one discarded warmup, and the
median of three. F3 direct c1/c2/c4/c8 is
**50.335/78.552/108.050/133.251 aggregate tok/s** with maximum rate
stdev/median **0.096%**; one physical c8 is **2.647x** c1 and **+23.32%** over
the current direct-c4 rate. Server protocol remains the clean F2 packet: 512
exact prompt IDs and 128 generated outputs/request, a 20 ms admission window,
one discarded plus three measured bursts, and scheduler latency. C9/C13 are
multiple declared buckets, never wider native widths. All **189/189** requests
match resident prompt IDs, direct-c1 outputs, usage, and finish metadata.
Grouped C13 is **1.702x** logical-c1 and **1.701x** serial; one exact c8→c13
live trace emits **1,664/1,664** IDs at **71.891 aggregate tok/s** and drains
ownership. Clean C2→C8 and C4→C8 traces preserve **256/256** IDs each. Starred
server memory is cumulative.

Artifacts: [`F3 singleton-indexed GDN`](benchmarks/results/2026-07-19-gfx1151-gguf-f3-singleton-gdn-retained.json),
[`F2 occupancy-adaptive serving`](benchmarks/results/2026-07-19-gfx1151-gguf-f2-occupancy-adaptive-serving.json),
[`E1 direct correctness`](benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-direct-correctness.json),
[`E1 direct scaling`](benchmarks/results/2026-07-17-gfx1151-gguf-concurrency-e1-native-c8-scaling-closure.json),
[`E3 arbitrary C`](benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-e3-arbitrary-c-correctness.json), and
[`F1 real server`](benchmarks/results/2026-07-18-gfx1151-gguf-concurrency-f1-server-scaling-closure.json).

The explicit short mirrored-INT8 server packet separately records blocking
c1/c2/c4/c8 **40.467/57.211/72.037/72.514 tok/s** and exact SSE
**39.665/52.225/68.665/79.789 tok/s**. All **117** server rows and the full
11-prompt/99-position KL/top-1 gate pass; C8 drains ownership and packed
workspace. Bounded BF16 mirrors mean this is not a memory-saving default, and
strict high-C SLO plus tail4/direct/long INT8 remain open. Evidence:
[`mirrored INT8 continuous concurrency`](benchmarks/results/2026-07-19-gfx1151-gguf-mirrored-int8-continuous-concurrency.json).

### gfx1151 historical cross-engine concurrency (2026-06-15)

**Status: stale diagnostic.** hipEngine used PARO W4/BF16 KV; llama.cpp used
Vulkan Q4_K_S/f16 KV, and vLLM did not produce a healthy server. The summary
lacks the measured hipEngine commit and has incomplete device provenance. No
eligible historical row is inferred from the current direct GGUF table.

Source artifacts: [`gfx1151 summary`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[`llama.cpp Vulkan`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[`vLLM blocked`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

A 2026-06-13 RX 7900 XTX rerun reached c1/c2/c4 but c8 blocked with HIP OOM;
see [`XTX partial`](benchmarks/results/2026-06-13-hipengine-qwen35-concurrency-decode-latest-xtx-blocked-c8.json).
Replicate the W7900 hipEngine, llama.cpp Vulkan, and vLLM concurrency rows with:

```bash
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

The exact settings and gfx1151 runner gap are recorded in
[`benchmarks/README.md`](benchmarks/README.md#readme-sweep-test-procedure).

## GGUF Support

As of v0.2.0, hipEngine includes resident Qwen3.6 GGUF support for `Q4_K_M` and
`Q4_K_S` model files (with more formats planned). This is a major runtime path,
not just a loader shim: GGUF has its own quant readers, bulk-prefill path,
decode-repacked T16 layouts, and fast-path controls.

The 40-layer `UD-Q3_K_M` target keeps resident `IQ3_XXS`/`IQ4_XS`
selected-expert weights compressed and executes native gate/up and down kernels,
bulk prefill, and graph decode. The merged W7900 branch preserves its first
correctness-oriented baseline: **614.089/92.285**, **623.583/97.373**, and
**616.135/98.111** prefill/decode tok/s at 512/128, 1K/128, and 4K/128. Those
2026-07-19 measurements describe the historical branch implementation; the
newer optimized direct/native Q3 route and current records are documented in
[`benchmarks/README.md`](benchmarks/README.md#merged-ud-q3_k_m-gpu1-and-w7900-records).

Current caveats:

- PARO models take ~22s to load on the W7900 test host in the current refresh;
  GGUF Q4_K_M currently takes about 74s because decode-repack happens on load.
  On-disk caching could reduce startup time later, but would require additional
  storage for repacked layouts.
- GGUF has higher base weight residency than packed PARO before KV cache is the
  deciding factor. The full-attention KV slope is the same 10-layer Qwen3.6
  shape; the 24 GiB long-context gap is mostly the loaded-weight baseline.
  Packed PARO is ~19.07 GiB on disk, while the local GGUF tensor payloads are:

  | GGUF tensor family | Q4_K_M GiB | Q4_K_M mix | Q4_K_S GiB | Q4_K_S mix | Q4_K_S - Q4_K_M |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Q4_K | 11.531 | 54.7% | 16.875 | 84.8% | +5.344 |
  | Q5_K | 6.531 | 31.0% | 0.000 | 0.0% | -6.531 |
  | Q8_0 | 1.932 | 9.2% | 1.932 | 9.7% | +0.000 |
  | Q6_K | 1.004 | 4.8% | 1.004 | 5.0% | +0.000 |
  | F32/BF16 metadata | 0.098 | 0.5% | 0.098 | 0.5% | +0.000 |
  | **Total tensor payload** | **21.097** | **100.0%** | **19.909** | **100.0%** | **-1.188** |

  In other words, `Q4_K_S` saves ~1.19 GiB versus `Q4_K_M` by replacing the
  selected-MoE `Q5_K` expert-down payload with `Q4_K`; it still starts above
  packed PARO, and hipEngine's resident T16/pack8 decode layouts add their own
  allocator shape. On 24 GiB cards, current `Q4_K_M` BF16-KV support is a
  mid-context path unless a lower-memory KV/weight policy is explicitly enabled.
  The current clean gfx1151 p512/d128 census is **21.478 GiB** owned/tracked:
  **20.461 GiB** replacement weights, **0.503 GiB** required raw token embedding,
  **0.097 GiB** dense weights/metadata, and **0.417 GiB** scratch/session buffers.
  It confirms the default T16 path replaces source layouts rather than retaining
  raw+packed copies; this short-context margin is not a 128K capacity claim.
- GGUF is close enough to PARO to share some high-level scheduling ideas, but in
  practice it needs substantial GGUF-only kernels and dispatch. The goal for
  future releases is to keep closing the remaining PARO/GGUF speed gap.


## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  USER API                                                       │
│  hipengine.LLM.generate()           library API                 │
│  hipengine serve                    OpenAI-compatible server    │
├─────────────────────────────────────────────────────────────────┤
│  LOADING (torch-free)                                           │
│  safetensors mmap + hipMemcpyAsync / HF config / jinja2 chat    │
│  templates / HF tokenizers (Rust)                               │
├─────────────────────────────────────────────────────────────────┤
│  DISPATCH                                                       │
│  Scheduler / Block Manager (KVPolicy) / Prefix Cache            │
│  Fusion Planner (chain → kernel plan, fused preferred)          │
│  Model / Quant / Layer plugins / Engine loop (hipGraph replay)  │
├─────────────────────────────────────────────────────────────────┤
│  CORE (torch-free primitives)                                   │
│  hipengine.Tensor / device / memory / stream / graph / blas     │
│  build (hipcc subprocess + ctypes.CDLL + .so cache)             │
├─────────────────────────────────────────────────────────────────┤
│  KERNELS (backend-keyed custom HIP implementations)             │
│  kernels/hip_gfx1100/  attention / linear_attn / moe / quant    │
│                        wmma / norm / rotary / fused             │
│  kernels/hip_gfx1151/  native target-arch peer backend          │
│  kernels/cuda_sm86/    (future)                                 │
│  kernels/cpu_reference/ correctness oracle, no GPU required     │
└─────────────────────────────────────────────────────────────────┘
```

Full layer diagram, plugin axes, KV cache ABI, and roadmap are in
[`docs/PLAN.md`](docs/PLAN.md).

## Installation

```bash
# PyPI wheel: runtime, JIT kernel sources, vendored AOTriton, and server
pip install hipengine

# Source checkout: fetch Git LFS payloads before an editable install
git lfs install
git lfs pull
pip install -e .

# with the optional dlpack torch bridge for user-boundary interop
pip install "hipengine[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.10+. A working ROCm install with `libamdhip64.so` on the loader path
is required for any GPU run; CPU-reference correctness tests run without a GPU.

### ROCm / TheRock setup for retained benchmark rows

For retained gfx1100 benchmark rows, use the pinned AMD TheRock environment in
[`docs/THEROCK.md`](docs/THEROCK.md), not an ad-hoc mixed `/opt/rocm` runtime.
Current retained rows use TheRock ROCm `7.13.0a20260423` with:

```text
HIP version: 7.13.26162-1140233ffe
```

On this host (`Linux 7.0.10-1-cachyos`, W7900 VBIOS `113-D7070100-138`, RX 7900
XTX VBIOS `113-EXT89622-001`), ROCm 7.14 nightly diagnostics showed GGUF prefill
and MTP wall-time regressions, so 7.13 remains the canonical stack until a newer
ROCm release beats the same gates. See `docs/THEROCK.md` for the exact `pip
install`/repair commands, clean process wrapper, and the upstream TheRock
[`RELEASES.md`](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) reference.

The installed app exposes a small command group:

```bash
hipengine --help
hipengine serve --help
hipengine bench list
```

## Quickstart

Model loading does not start network downloads. Populate the Hugging Face cache
before using a repository ID:

```bash
hf download shisa-ai/Qwen3.6-35B-A3B-PARO-packed
```

Then construct `LLM` with the same repository ID:

```python
from hipengine import LLM, SamplingParams

llm = LLM("shisa-ai/Qwen3.6-35B-A3B-PARO-packed")
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
```

`LLM(model)` auto-detects `gfx1100` or `gfx1151` and selects the model plugin's
quantization. The Qwen3.6 GGUF path also selects T16 decode-repack plus the
retained WMMA-prefill/GEMV-decode session profile. Explicit `backend=` and
`quant=` arguments are overrides; supported PARO and GGUF models do not require
hipEngine environment variables. Unsupported registry combinations fail instead
of falling back to a torch path.

## OpenAI-compatible server

The OpenAI-compatible FastAPI layer is installed by default:

```bash
pip install hipengine
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --served-model-name qwen-paro
```

`--model` accepts either a local filesystem path or a Hugging Face model ID
already present in the local HF cache; hipEngine resolves IDs locally and does
not download weights during startup.

Core endpoints are `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions`, with token-level SSE streaming, logprobs,
OpenAI-style tools, structured-output validation, and Qwen thinking controls.
hipEngine extensions provide readiness/capability discovery, token and context
diagnostics, and app-local session management. Chat responses separate
`<think>` reasoning into `reasoning_content`. The server eagerly warms the model
on startup, caps omitted chat `max_tokens` with
`--chat-default-max-tokens` (default 4096), and has an explicit `--debug` mode
for full request/response payload logging. See the complete
[`docs/API.md` endpoint table](docs/API.md#endpoints) for bearer-token auth,
request examples, feature contracts, diagnostics, and current limitations.

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture, plugin axes, phase roadmap, LoC budgets |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark protocols, baselines, correctness gate, artifact format |
| [`docs/TESTING.md`](docs/TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog, source-lineage drift workflow, JIT cache gotchas, build profiles |
| [`docs/ENVS.md`](docs/ENVS.md) | Environment variables, TheRock setup, benchmark/profiling profiles |
| [`docs/ROOFLINE.md`](docs/ROOFLINE.md) | RDNA3 / W7900 performance model and decision tree |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | Implementation status and concrete milestones |
| [`docs/API.md`](docs/API.md) | OpenAI-compatible server usage and endpoint support |
| [`docs/LAGUNA.md`](docs/LAGUNA.md) | Laguna S 2.1 gfx1151 support contract, implementation record, DFlash boundary, and evidence index |
| [`docs/LAGUNA-prefill.md`](docs/LAGUNA-prefill.md) | Laguna prefill plans: W7900 exact WPF-1T coltile/grouped-IQ/qrow4 production through 4K, plus the gfx1151 campaign record |
| [`docs/PREFILL.md`](docs/PREFILL.md) | Native prefill implementation spec |
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Normal sampling parameter support plan |
| [`docs/MTP.md`](docs/MTP.md) | Multi-token prediction plan |
| [`docs/NATIVE_SPEC_CYCLE.md`](docs/NATIVE_SPEC_CYCLE.md) | Canonical N0-N5 milestone glossary, ownership distinctions, current speculative performance scorecard, and evidence index |
| [`docs/DFLASH.md`](docs/DFLASH.md) | DFlash draft-model speculative decode plan |
| [`docs/SOL-OPTIMIZATION.md`](docs/SOL-OPTIMIZATION.md) | gfx1151 PARO/GGUF optimization ledger and completion gates |
| [`docs/MTP-LLAMACPP-PARITY.md`](docs/MTP-LLAMACPP-PARITY.md) | Current GGUF MTP parity results and open reruns |
| [`docs/PARO-GGUF-MTP-TRANSFER.md`](docs/PARO-GGUF-MTP-TRANSFER.md) | PARO follow-up queue from GGUF/MTP server and verifier work |
| [`docs/HIP-vs-VULKAN.md`](docs/HIP-vs-VULKAN.md) | Current timing-contract v2 backend conclusions and portability gates |
| [`benchmarks/README.md`](benchmarks/README.md) | Canonical topline scoreboard, platform freshness, protocols, and refresh commands |
| [`AGENTS.md`](AGENTS.md) | Ground rules for every coding / review / benchmarking task |
| [`WORKLOG.md`](WORKLOG.md) | Append-only cross-session journal of decisions and measurements |

## Development

```bash
# narrowest test suite (CPU-only paths run without a GPU)
pytest -q

# kernel source-lineage drift check before any port
python3 scripts/check_lineage.py --kind kernel --diff stat
```

See [`AGENTS.md`](AGENTS.md) for the full workflow: when to run the
CPU-reference correctness gate, when to add a `rocprofv3 --kernel-trace` smoke,
and what a retained benchmark row requires.

## References & lineage

hipEngine is not a fork of any project; it is a brand new codebase with from-scratch
code and kernels. Of course it builds on the work of many others:

- [ROCm](https://github.com/ROCm/rocm) - of course this all sits on AMD's open-source
  compute stack, notably on [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - most of the original
  kernel tuning iteration loops used this as a host-layer. Some of the performance 
  limitations of the architecture motivated the hipEngine rewrite, but we remain
  grateful and deeply appreciative of nano-vllm as a great research platform.
- [ParoQuant](https://github.com/z-lab/paroquant) - after reviewing the current SOTA on model
  quantization, we chose ParoQuant as the first target due to both its excellent accuracy
  *and* its efficiency (QTIP/[YAQA](https://github.com/Cornell-RelaxML/yaqa-quantization) is 
  very cool but proved challenging to implement performant RDNA3 kernels)
- [FastDMS](https://github.com/shisa-ai/FastDMS) - our KVCache ABI is shaped by the lessons 
   learned from building our DMS reference implementation.

Greetz: [ROCmFPX](https://github.com/charlie12345/ROCmFPX), [hipfire](https://github.com/Kaden-Schutt/hipfire), [Lucebox](https://github.com/Luce-Org/lucebox-hub), [DS4](https://github.com/antirez/ds4), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and ofc the og [llama.cpp](https://github.com/ggml-org/llama.cpp)

See also: [Marlin](https://github.com/IST-DASLab/marlin), [kernel-anvil](https://github.com/apollosenvy/kernel-anvil), [wmma_ops](https://github.com/glovepost/wmma_ops), [tilelang](https://github.com/tile-ai/tilelang), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [ROCm examples](https://github.com/ROCm/rocm-examples)


## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. It is built and distributed
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
