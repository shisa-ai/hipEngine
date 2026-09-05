# Photon 2 / Kestrel review for gfx1100 and gfx1151

> **Reviewed:** 2026-08-04
>
> **Decision:** learn from the generated-decode runtime contract and benchmark
> framing; do **not** port Photon’s compiler or start a hipEngine whole-model
> persistent megakernel.

## Executive verdict

Photon 2 is useful evidence that a model-aware compiler can beat generic serving
stacks at low concurrency when it controls physical weight layout, persistent
state, KV access, shape buckets, and device scheduling together. It is not a
portable implementation for hipEngine:

- the compiler is proprietary;
- the shipped device programs are CUDA `sm90` binaries rather than source;
- the public Kestrel host depends on Torch and a separately packaged binary
  kernel wheel;
- Photon’s published launch results are H100-only and do not isolate the
  megakernel from the rest of its serving system; and
- hipEngine already has AMD measurements rejecting a wholesale persistent
  VRAM-streaming megakernel on gfx1100, while the current gfx1151 dispatch residue
  is below the project’s trigger for another execution mechanism.

The transferable part is the **contract around generated execution**, not the
CUDA code:

1. compiler-emitted physical-weight recipes;
2. explicit carried-state representation requirements;
3. capacity buckets with dynamic active-row extents;
4. direct binding to the serving runtime’s paged KV ABI;
5. exact artifact/model/backend matching plus transparent fallback; and
6. admission by complete serving measurements, including cold start.

hipEngine should retain its registered kernels and `hipGraph` paths. If a future
profile reopens this lane, the first experiment should be an in-tree,
AMD-native generated-decode **descriptor and bounded subgraph**, not a Photon
binary import and not a whole Qwen 35B persistent kernel.

## Sources and confidence

### Primary sources

- [Photon 2 launch post](https://moondream.ai/blog/photon-2-launch), including
  its throughput and cold-start SVGs.
- Public Kestrel repository at
  [`m87-labs/kestrel@1b23055d`](https://github.com/m87-labs/kestrel/tree/1b23055df49b13feff65f960baa02d46d44e9641)
  (`v0.5.0`, committed 2026-08-02).
- [`kestrel-kernels==0.4.9`](https://pypi.org/project/kestrel-kernels/0.4.9/)
  Linux CPython 3.12 wheel, SHA-256
  `d7f5a4cd7302b913f2369cdbb69c9e5613328d7728b1bdd113c2c4bc42ef4a27`.
  The review inspected its public Python runtime and bundle catalog only; it did
  not disassemble or reproduce the packaged device programs.

### Evidence labels

- **Published:** stated by Moondream; not independently reproduced here.
- **Source-observed:** present in the public Kestrel host or PyPI runtime.
- **Inferred:** engineering interpretation; not a Photon claim.

No Photon/Kestrel performance number is a hipEngine, ROCm, gfx1100, or gfx1151
measurement.

## What Photon 2 publishes

The launch post positions Photon for low-concurrency “Physical AI” rather than
maximum-throughput fleet serving. It says its compiler emits one megakernel for
the complete inference path, reducing CPU/GPU launch and synchronization cost
and enabling optimization across operator boundaries.

The published throughput chart reports request-throughput ratios on one NVIDIA
H100 80 GB HBM3 using ChartQA request streams at concurrency 1/2/4/8. Its footer
identifies Photon 2.0.0, vLLM 0.25.1 (2026-08-03), and SGLang 0.5.3
(2026-08-02):

| Model | Photon / vLLM C1/C2/C4/C8 | Photon / SGLang C1/C2/C4/C8 |
| --- | --- | --- |
| Moondream 3 | 2.33x / 1.68x / 1.43x / 1.34x | not tested |
| Qwen3.5 0.8B | 1.54x / 1.48x / 1.48x / 1.29x | 1.57x / 1.51x / 1.54x / 1.40x |
| Qwen3.5 2B | 1.29x / 1.28x / 1.30x / 1.20x | 1.44x / 1.44x / 1.51x / 1.42x |
| Qwen3.5 4B | 1.25x / 1.21x / 1.13x / 1.07x | 1.35x / 1.31x / 1.22x / 1.23x |
| Qwen3.5 9B | 1.10x / 1.10x / 1.10x / 1.04x | 1.14x / 1.15x / 1.14x / 1.14x |
| Gemma 4 E2B | 1.27x / 1.20x / 1.12x / 1.01x | 1.55x / 1.50x / 1.41x / 1.32x |
| Gemma 4 E4B | 1.17x / 1.14x / 1.06x / 1.02x | 1.40x / 1.35x / 1.31x / 1.23x |

The cold-start chart defines the metric as fresh process/server start through
first completed C1 inference, including model load, runtime initialization, and
client/data preparation:

| Model | Photon | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Moondream 3 | 36 s | 42 s | not tested |
| Qwen3.5 0.8B | 35 s | 78 s | 101 s |
| Qwen3.5 2B | 36 s | 79 s | 99 s |
| Qwen3.5 4B | 37 s | 79 s | 105 s |
| Qwen3.5 9B | 39 s | 79 s | 112 s |
| Gemma 4 E2B | 68 s | 73 s | 180 s |
| Gemma 4 E4B | 74 s | 74 s | 185 s |

These are useful system-level results, but the post does not publish the exact
commands, absolute request rates, checkpoint/precision matrix, request-length
distribution, per-component profile, or raw samples. It therefore supports
“the complete Photon system won this disclosed H100 test,” not “one persistent
kernel causes the full ratio” and not an AMD speedup estimate.

## What the public implementation shows

### Two related decode paths

Kestrel 0.5.0 distinguishes an older Moondream-specific `whole_model` runtime
from the generalized `generated_decode` runtime used by Qwen and Gemma.
The [changelog](https://github.com/m87-labs/kestrel/blob/1b23055df49b13feff65f960baa02d46d44e9641/CHANGELOG.md#L24-L32)
explicitly says both use bundled programs and retain native fallbacks.

The Moondream path is a persistent instruction-tape VM compiled for an exact SM
architecture, SM count, and batch bucket. Its public policy enables only
measured H100 buckets: Moondream 3 C1 and Moondream 2 C1/C2 on the 132-SM
variant. Measured or packaged wider buckets are not automatically admitted when
slower. That distinction—**available is not the same as selected**—is directly
useful.

The generalized path is described by
[`GeneratedDecodeSpec`](https://github.com/m87-labs/kestrel/blob/1b23055df49b13feff65f960baa02d46d44e9641/kestrel/runtime/generated_decode.py#L100-L110)
and bound by
[`GeneratedDecode`](https://github.com/m87-labs/kestrel/blob/1b23055df49b13feff65f960baa02d46d44e9641/kestrel/runtime/generated_decode.py#L195-L378):

- require CUDA and BF16 before resolving a program;
- structurally match compiler-emitted descriptors to model storage;
- materialize physical weight layouts once;
- bind per-capacity programs to fixed serving slots;
- prepare only missing device inputs;
- choose the smallest compiled capacity that covers the active batch; and
- run the generated program or fall back to the ordinary decode graph.

The current wheel catalog contains 24 usable H100 generated-decode programs:
C1/C2/C4/C8 for Qwen3.5 0.8B/2B/4B/9B and Gemma 4 E2B/E4B. This matches the
launch benchmark matrix. It does not provide an AMD program or source compiler.

### The useful contracts

1. **Physical weight recipes.** The runtime descriptor can request identity,
   dtype conversion, concatenation, conv squeezing, gate/up interleaving, or a
   view. View-compatible native weights are rebound to the generated physical
   slab; irreversible transforms retain both representations.
2. **Named physical ABI.** The bundle argument plan names tensors, layer address
   tables, raw pointers, dynamic scalars, counters, and stream ownership. The
   runtime rejects missing, extra, or mismatched arguments before launch.
3. **Carried-state forms.** Qwen’s Conv/GDN recurrent state declares its required
   representation per capacity. A coordinator converts only when switching
   between native and generated owners.
4. **Direct paged KV.** Generated Qwen/Gemma consumes the engine page table,
   compact active-row indices, positions, and per-layer K/V storage rather than
   maintaining a shadow cache.
5. **Capacity, not exact live count.** Programs are compiled for C1/C2/C4/C8,
   while `active_batch` remains a launch extent bounded by capacity.
6. **One model owner, measured fallback.** The generated path and native graph
   share resident model/session state. Unsupported storage, device, capacity,
   or artifact combinations fall back; genuine launch errors remain fatal.

The public Kestrel host is about 30,955 Python lines and imports Torch. Its
pinned kernel wheel exposes about 4,879 Python runtime lines plus a 58,909,254
byte packed CUDA collection. This is a source-free deployment runtime around a
proprietary compiler, not a small compiler implementation that hipEngine can
port.

## Mapping to hipEngine

| Photon/Kestrel idea | hipEngine status | Decision |
| --- | --- | --- |
| Whole-model persistent decode | gfx1100 persistent-barrier and fused-FFN campaigns already measured the relevant AMD economics; Moonshine’s gfx1151 graph-era ladder also rejected larger rungs before coding | **Do not port** |
| One host launch | hipEngine already uses fixed-address `hipGraph` replay where correctness and wall gates pass | **Keep current graph owner** |
| C1/C2/C4/C8 capacity buckets | Native GGUF and gfx1151 PARO already have physical c-aware buckets and lifecycle gates | **Already adopted** |
| Direct paged KV | hipEngine’s `KVLiveSpans` is a stronger policy-neutral ABI than Kestrel’s page-table/context-length binding | **Preserve `KVLiveSpans`; do not simplify it** |
| Physical weight recipes | Quant plugins already own repacks/materialization, but the contracts are distributed through model-specific code | **Potentially adopt as descriptor metadata when an AOT/generated path needs it** |
| Carried-state representation requirements | hipEngine already validates Conv/GDN state and graph ownership, but lacks one generic requirement object across generated/native owners | **Useful medium-term abstraction, not a current hot-path change** |
| Exact artifact matching | Four-axis registry and build cache already key backend/layer/quant/variant | **Extend only through registry/package capability metadata; never add engine branches** |
| Source-free AOT bundle | hipEngine ships AOTriton but JIT-builds many native HIP libraries | **Measure readiness-to-first-request first; package more AOT objects only if compile/load is material** |
| Low-concurrency and cold-start benchmark lens | hipEngine already records C1-C8 server rows and startup phases, but not every headline uses fresh-process-to-first-completed-request | **Adopt as a protocol row, not as a Photon comparison** |
| Proprietary generalized compiler | No source is available; a clean AMD implementation would be a new compiler project | **Defer** |

### gfx1100

The W7900 persistent-barrier experiment found approximately 1 us grid barriers,
but only 1.08-1.27x opportunity for AR-faithful 3-12 MiB fresh-weight stages;
128-256 MiB VRAM-bound stages were 0.93-0.96x. The selected PARO FFN megakernel
was 2.66x slower than the production staged path because it lost GPU-filling
parallelism. See [`docs/MEGAKERNEL.md` §10](MEGAKERNEL.md).

Photon does not overturn those local measurements. It does reinforce their
redirect: eliminate dispatch-bound glue, intermediate traffic, and redundant
state conversions while preserving wide VRAM-streaming kernels. A generated
scheduler could eventually automate those choices, but a whole-model kernel is
not admitted merely because H100 benefits.

### gfx1151

Current Q4_K_M graph replay improves matched wall by only +1.00%/+0.86%/+0.36%
at 512/4K/bounded-128K, below the `>~3%` trigger for another dispatch mechanism.
The unresolved long-prefill non-retiring AQL queue also argues against adding a
second persistent cooperative owner now. PARO c1 graph replay has an independent
correctness rejection on gfx1151, so a generated owner would first owe the same
state/KV/trajectory gate rather than bypass it.

Photon’s exact-SM-count artifacts also warn against treating `gfx1151` alone as
a sufficient persistent-program key. Any future cooperative program must bind
the actual CU/residency shape through backend capability metadata and fail
closed on unmatched devices.

## Prioritized recommendation

### P0 — do now (documentation/benchmark discipline only)

1. Keep registered eager/graph decode and every unfused fallback unchanged.
2. Add fresh-process-to-first-completed-request to future server comparison
   artifacts, alongside the existing startup phase breakdown.
3. Continue C1/C2/C4/C8 and category/heldout gates; do not use Photon’s H100
   ratios as an AMD baseline.

### P1 — only after a profile trigger

Open a bounded generated-decode experiment only when `rocprofv3` shows more than
approximately 3% of relevant decode wall remains attributable to dispatch,
state-layout conversion, or removable intermediate traffic after graph replay.
The first RED artifact should define a descriptor containing:

- `(backend, layer/subgraph, quant, variant)` registry identity;
- model/weight-layout fingerprint and physical recipes;
- active-C capacity and context/mode bucket;
- required carried-state representations;
- `KVLiveSpans` ABI version;
- fixed pointer/workspace ownership;
- kernel plan and synchronization proof; and
- fallback plus correctness/performance artifact IDs.

Prototype one bounded layer/subgraph already covered by a CPU oracle. Compare
registered eager kernels, the current `hipGraph`, and the generated candidate on
identical state and request streams. Do not begin with a 35B whole-model tape.

### P2 — only after a retained P1 result

Consider precompiled HIP code objects in wheels when startup decomposition shows
JIT/build time is material. Key artifacts by backend architecture, actual
CU/residency shape where relevant, model/quant/layout fingerprint, C/context
bucket, compiler version, and source hash. A missing or mismatched artifact must
resolve through the registry to the existing unfused/graph path.

A generalized AMD inference compiler is justified only after at least two model
families and both gfx11 backends demonstrate repeated retained subgraph wins.
Until then it would duplicate a proprietary project without evidence that the
compiler, rather than hipEngine’s existing tuned kernels, is the limiting work.

## Admission gates for any reopened lane

- Torch-free public runtime remains intact.
- No backend/quant branches enter engine, dispatch, or model code.
- Every fused/generated composite retains the registered unfused fallback.
- Attention reads the complete `KVLiveSpans` ABI.
- Full logits, token trajectory, Conv/GDN state, live K/V, spans, reset,
  cancellation, and lifecycle pass the applicable exact gate; any new kernel
  also passes KL <= 0.05 and top-1 >= 90% versus `cpu_reference`.
- The exact same multi-prompt/server protocol measures baseline and candidate.
- At least two independent samples clear the project’s dispatch-lever threshold
  and all correctness/guard checks pass.
- Fresh-process startup, memory, artifact size, and fallback cost are reported.
- No fixed-prompt or benchmark-conditioned specialization is permitted.

## Licensing boundary

The launch post says “The Photon inference engine is Apache 2.0” and separately
says the compiler is proprietary. As of the reviewed Kestrel commit and PyPI
0.5.0 metadata, however, the repository has no `LICENSE` file, `pyproject.toml`
has no license field, PyPI reports no license files/expression, and the README’s
“License” section describes free local use rather than Apache terms.

Treat this as unresolved packaging/license metadata, not permission to copy the
public Python runtime or packaged device artifacts. hipEngine may independently
implement the architectural ideas documented here. Any source-level reuse must
wait for an explicit upstream license file covering the specific repository and
package.
