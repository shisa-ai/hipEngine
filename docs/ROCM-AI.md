# ROCm.AI: relevance to hipEngine

_Research snapshot: 2026-08-05; adoption inventory rechecked 2026-08-06. This
is a source-backed decision document, not a benchmark report. Product status,
support matrices, repository branches, and cloud offers can change; recheck the
linked primary sources before acting on a recommendation._

## Executive conclusion

[ROCm.AI](https://www.amd.com/en/products/software/rocm.html) is a new umbrella
for three different things:

1. [AMD Skills](https://github.com/amd/skills), which give coding agents AMD
   workflow instructions;
2. [Hyperloom](https://github.com/AMD-AGI/Hyperloom), an agentic optimizer for
   supported LLM serving stacks; and
3. the [ROCm Core SDK](https://www.amd.com/en/products/software/rocm/sdk.html),
   which is the compiler/runtime/library/tool foundation hipEngine already
   uses directly.

The Core SDK is immediately and substantially relevant. AMD Skills and
Hyperloom are not substitutes for hipEngine's existing optimization workflow.
Some pieces of their ecosystem are useful as isolated tools, source-lineage
references, or process ideas.

| Offering | Decision for hipEngine | Why |
| --- | --- | --- |
| ROCm Core SDK 7.14 / TheRock | **Adopt as the pinned platform baseline** | It officially packages both `gfx1100` and `gfx1151`, and it contains the HIP runtime, compiler, BLAS libraries, graph APIs, profiling, debugging, and monitoring surfaces hipEngine needs. |
| `rocprofv3`, ROCprofiler-SDK, ROCTx | **Use now and deepen** | They fit the torch-free runtime, support Radeon/Ryzen, and can add HIP API, memory, dispatch, and counter evidence to the kernel traces hipEngine already retains. |
| hipBLASLt offline tuning | **High-priority pilot** | hipEngine currently selects a fixed heuristic position for some shapes. Offline tuning can find a shape-specific solution, but results must be keyed by ROCm/hipBLASLt version, GPU architecture, shape, dtype, transpose/layout, and workspace limit. |
| AMD SMI 26.5 | **Use now for evidence; controlled APU-memory pilot** | It can capture clocks, power, temperature, throttling, and memory state. ROCm 7.14 also adds Ryzen APU GTT/carve-out controls, directly relevant to gfx1151 residency, but changing them is privileged and reboot-scoped. |
| TraceLens | **Isolated offline pilot** | It is the one Hyperloom component explicitly described as hardware-agnostic and accepts `rocprofv3` JSON/PFTrace. It may add structured trace diffs and reports without entering the runtime. |
| Magpie | **Optional isolated pilot** | Its custom HIP-kernel analyze/compare mode could standardize a microbenchmark, but its framework benchmark path overlaps hipEngine's stricter harnesses. |
| AITER | **Reference and selective source pilot; no runtime dependency** | W7900 and gfx1151 are experimental, most optimized CK/ASM paths remain CDNA-only, and the normal install brings Python/Triton/framework dependencies. Its HIP tests, operator catalog, raw C++ integration examples, and Opus header are nevertheless valuable. |
| AITER Opus | **Build-time experiment** | Its minimal-header and device-only compilation guidance closely matches hipEngine's raw-pointer/ctypes design and may reduce cold JIT time. It does not itself provide optimized GEMM/attention kernels. |
| Composable Kernel / CK-Tile | **Reference and narrow prototype only** | It recognizes gfx11 targets, including gfx1151 in current configuration headers, but individual instances and performance are target-specific and template compilation is heavy. Do not replace proven small-batch HIP kernels wholesale. |
| rocWMMA | **Useful on gfx1100; treat gfx1151 as unvalidated** | The current official architecture list includes `gfx1100` but not `gfx1151`, despite build-history references to gfx1151. Use only behind target-specific correctness and performance gates. |
| Quark | **Offline quantization/import pilot** | It can export GGUF and JSON+safetensors and supports AWQ/GPTQ/rotation/KV quantization. It must remain outside `LLM.generate()` and its exact packing/scales need importer fixtures and full quality validation. |
| vLLM / SGLang | **External baselines and idea sources** | They are not dependencies or architecture templates for the torch-free runtime. ROCm's own release notes expose important Radeon caveats that must accompany any comparison. |
| RCCL | **Future multi-GPU plugin** | It is officially available for Instinct, Radeon, and Ryzen, but it has no benefit to the current single-GPU path. |
| Hyperloom end to end | **Do not adopt now** | Hyperloom 1.0.0a2 is validated on MI300X/MI325X/MI355X, ROCm 7.2.x, and SGLang/vLLM. It is not validated for gfx1100/gfx1151 or a custom torch-free engine, and setup deliberately mutates a dedicated workspace. |
| AMD Skills catalog | **Do not blanket-install** | The catalog is a Tech Preview. Most current skills target local Lemonade apps, EPYC, Instinct serving, or PyTorch traces; several most relevant skills are only planned. Inspect and pin a specific skill if a concrete need appears. |
| MIGraphX / ONNX Runtime ROCm EP | **Not a current target path** | ROCm 7.14's AI matrix validates them only on `gfx942`/`gfx950`, and a graph runtime would conflict with hipEngine's kernel-first architecture. |
| Enterprise AI stack / Infinity Hub | **Deployment context only** | The current validated path is Kubernetes and Instinct. It may matter when hipEngine targets fleet deployment, but it does not help current gfx11 kernel work. |

The best near-term work is therefore not “install ROCm.AI.” It is to use the
new Core SDK distribution and observability surfaces more systematically,
pilot hipBLASLt tuning and TraceLens on already-cached hipEngine workloads,
and selectively mine AITER/Opus/CK for ideas that survive hipEngine's own
correctness and benchmark gates.

## Current adoption inventory

This is the consolidated answer to “what are we already using, and what are we
not using yet?” It records hipEngine tree state, not merely upstream
availability. The 2026-08-06 check searched production code, kernels, scripts,
tests, and package metadata; a name appearing only in documentation or a model
artifact is not counted as an integration.

### Already used or partially used

| Surface | Current hipEngine state | Remaining gap |
| --- | --- | --- |
| HIP runtime, ROCr-facing APIs, native `hipcc`, and HIP graphs | **Actively used.** The torch-free runtime owns raw pointers, streams, events, graph capture/replay, native gfx1100/gfx1151 builds, and cached shared-object loading. | Continue version-pinned upgrade tests; do not replace native code objects with the currently problematic generic SPIR-V path. |
| TheRock / ROCm SDK wheel layout | **Partially used.** Scripts resolve TheRock roots and architecture-specific libraries, including the SDK ROCTx library. | There is no single normalized benchmark manifest capturing SDK version, targets, package lock, driver, compiler, profiler, SMI, and memory policy together. |
| rocBLAS and hipBLASLt | **Actively used.** Direct `ctypes` bindings and shape-qualified hipBLASLt heuristic schedules exist; some workloads exhaustively screen returned zero-workspace heuristics. | The official offline tuner and version/target-qualified tuning override files are not used. The generic convenience selector still defaults to a fixed preferred position where a more specific route does not override it. |
| `rocprofv3`, HIP traces, and ROCTx selected regions | **Actively used.** Cached final-child kernel/HIP traces, selected regions, resource rows, and compact summaries are established evidence paths. | Counter discovery with `rocprofv3-avail`, a standard architecture-qualified counter set, and repeatable counter-to-bottleneck reports are absent. |
| AOTriton | **Selectively used.** A versioned runtime is retained for separately gated prefill attention paths. | This does not provide AITER, CK, rocWMMA, `KVLiveSpans` compatibility, or general framework integration. Each target/shape remains separately gated. |
| AMD matrix instructions | **Actively used through handwritten HIP/compiler builtins.** hipEngine has in-tree gfx11 WMMA kernels. | This is not an integration of the rocWMMA header library; there are no rocWMMA includes or calls in the production tree. |
| Hardware inventory | **Manually used.** `rocminfo`, compiler versions, profiler versions, and machine-specific facts appear in benchmark protocols and artifacts. | AMD SMI APU/GTT/power/clock/throttle capture is not yet a uniform automated artifact field. |

### Not yet used

“Not yet used” does not automatically mean “should be installed.” The
disposition column distinguishes the next useful pilots from capabilities that
should remain deferred under the current architecture and hardware.

| Surface not currently integrated | Evidence of current state | Disposition / next gate |
| --- | --- | --- |
| Reproducible ROCm platform manifest | TheRock paths are resolved in several scripts, but no helper captures `rocm_sdk version`, `rocm_sdk targets`, package set, driver/compiler/profiler versions, AMD SMI state, and GTT policy as one normalized record. | **P0 adopt.** Add a read-only helper or common artifact section with negligible benchmark overhead. |
| hipBLASLt offline tuner | No `hipblaslt-bench`, `HIPBLASLT_TUNING_FILE`, or `HIPBLASLT_TUNING_OVERRIDE_FILE` use exists. Current screens enumerate the algorithms returned through the C API. | **P0 bounded pilot.** Tune only real production descriptors; key results by library build, gfx target, shape/layout/types, epilogue, and workspace ceiling. Compare against the strong existing manual screens rather than assuming a win. |
| `rocprofv3-avail` and standardized hardware counters | No scripted counter-availability query or retained common counter bundle exists. Existing profiling is predominantly kernel/HIP/marker tracing. | **P0 adopt/pilot.** Enumerate on each exact agent, collect the smallest question-specific set, and never use profiler-perturbed throughput as a performance claim. |
| TraceLens | No code, script, dependency, or report artifact uses it. | **P0 isolated offline pilot.** Run it in a disposable environment against a copy of an existing gfx11 ROCprof JSON/PFTrace; retain only if it preserves kernel identity and adds actionable attribution. |
| Automated AMD SMI provenance | No production or benchmark script invokes `amd-smi`; current hardware metadata is assembled through other/manual paths. | **P0 adopt for read-only metadata.** Capture clocks, power, temperature, throttling, memory, and `node --gtt` where available. Keep privileged mutations separate. |
| AITER Opus and device-only HSACO loading | The build path invokes `hipcc -shared -fPIC` and loads `.so` files with `ctypes`. There is no Opus include, `-D__HIPCC_RTC__`, `hipcc --genco`, or raw HIP-module loader path. | **P1 build-time pilot.** Compare one isolated kernel's cold compile wall, artifact size, cached load, graph compatibility, trace identity, and exact output on both native targets. |
| ROCm Systems Profiler / ROCPD | No `rocprofiler-systems`, `rocprof-sys`, or ROCPD workflow exists in code or scripts. | **P1 pilot.** Use only for a measured host-minus-device gap in an isolated final process. |
| AMD Quark tooling/importer | No Quark dependency, converter, or importer fixture exists. References to older “Quark” model artifacts are model names, not use of AMD Quark. | **P1 offline fixture.** Generate one tiny artifact, inspect packing/scales/metadata, and require a quant-registry key plus CPU/quality gates before any new runtime path. |
| gfx1151 GTT/carve-out tuning | GTT size has been observed, but hipEngine has not run a controlled alternate-policy matrix through AMD SMI. | **P2 privileged experiment.** Reboot-scoped, explicitly scheduled, fully reversible, and measured separately for capacity and speed. |
| TransferBench | No invocation or retained artifact exists. | **Deferred diagnostic.** Use for a concrete copy/topology/residency question; it is not an LLM streaming-bandwidth benchmark. |
| RCCL | No RCCL/NCCL binding, plugin, or collective path exists. | **P2 future plugin.** Start only with suitable multi-GPU hardware, topology, failure semantics, and a matched single-GPU baseline. |
| HIP Execution Context / CU partitioning | No execution-context API path exists. | **Deferred research.** Relevant to QoS or multi-tenancy, not a default single-request speed path. |
| Batch managed-memory discard/prefetch APIs | No `hipMemDiscardBatchAsync` or `hipMemPrefetchBatchAsync` path exists. | **Deferred until paging exists.** Do not make resident weights/workspaces pageable merely to use the API. |
| Direct AITER, Composable Kernel, or rocWMMA dependency | No production integration exists. AOTriton's internal `aiter` namespace is unrelated to installing AMD AITER; handwritten WMMA builtins are not rocWMMA. | **Reference/narrow target-qualified pilots only.** Preserve in-tree source ownership, raw-pointer ABI, exact fallback, and gfx-specific gates. |
| Hyperloom, Magpie, and AMD Skills automation | None is installed or invoked by hipEngine. | **Intentional non-adoption except isolated pilots.** TraceLens is the first useful component; Magpie is optional for one microbenchmark. Do not run Hyperloom end to end or blanket-install Skills on current gfx11/shared-worktree scope. |
| MIGraphX, ONNX Runtime ROCm EP, Enterprise AI stack, and Infinity Hub | No integration exists. | **Intentional defer/reject for current scope.** Their validated hardware/runtime/deployment layers do not match the current gfx11 kernel-first engine. |

The shortest useful implementation sequence is therefore: (1) normalized ROCm
and AMD SMI provenance, (2) standardized counter availability/capture, (3) an
offline hipBLASLt tuner comparison against existing manual heuristic screens,
then (4) one disposable TraceLens trial. The Opus/HSACO, Systems Profiler, and
Quark experiments follow only when their stated build-time, host-gap, or quant
questions are active. Multi-GPU, paging, QoS, and deployment-stack work remains
conditional rather than latent required work.

## Scope and method

This review assumes no prior ROCm knowledge and evaluates each offering against
the current architecture in [PLAN.md](PLAN.md), the evidence rules in
[BENCHMARK.md](BENCHMARK.md), the kernel workflow in [KERNELS.md](KERNELS.md),
and the W7900 performance model in [ROOFLINE.md](ROOFLINE.md).

Primary sources were preferred:

- AMD product pages and ROCm 7.14 documentation;
- AMD/ROCm GitHub repositories and their compatibility files;
- current public component support matrices and known-issue lists; and
- the live, read-only toolchain on the hipEngine workstation.

No third-party performance number is treated as a hipEngine result. No AMD
marketing comparison is imported into the benchmark scoreboard. An upstream
claim becomes relevant evidence only after an exact in-tree reproduction with
model, quantization, workload, hardware, command, result, and correctness gate.

The labels used below mean:

- **Adopt:** stable enough to become part of the standard platform or evidence
  workflow after normal validation.
- **Pilot:** run a bounded experiment with an explicit success/failure gate.
- **Reference:** inspect source, tests, configuration, or algorithms without
  creating a runtime dependency.
- **Watch:** promising, but current hardware or framework validation does not
  match hipEngine.
- **Reject now:** do not integrate under current architecture and targets; this
  is not a claim that the project is generally poor.

## ROCm and ROCm.AI from first principles

### What ROCm is

[ROCm](https://rocm.docs.amd.com/en/latest/about/what-is-rocm.html) is AMD's
mostly open-source GPU software stack. At the lowest useful level for
hipEngine it provides:

- the HIP kernel language and runtime API;
- AMD LLVM/Clang and `hipcc`;
- the ROCr/HSA runtime that submits work to the GPU;
- math, communication, media, and primitive libraries;
- profilers, tracers, debuggers, and hardware monitoring; and
- packages containing architecture-specific GPU code objects.

HIP is analogous to CUDA as a programming/runtime surface, but ROCm is the
larger distribution around it. PyTorch, vLLM, and SGLang are consumers of
ROCm, not ROCm itself. hipEngine is another consumer: it loads HIP and BLAS
libraries with `ctypes`, builds HIP kernels, owns raw device pointers, and
captures/replays HIP graphs without putting PyTorch on the hot path.

### What the ROCm.AI name adds

The [ROCm.AI landing page](https://www.amd.com/en/products/software/rocm.html)
presents a developer-product hierarchy rather than a new binary runtime:

| Layer | Intended role | hipEngine interpretation |
| --- | --- | --- |
| AMD Skills | Teach coding agents AMD-specific, opinionated workflows | Potential workflow input; never evidence by itself. |
| Hyperloom | Analyze and automatically optimize supported LLM workloads | A separate optimization orchestrator whose present validation does not match hipEngine. |
| ROCm Core SDK | Runtime, compiler, libraries, framework integrations, and tools | The real platform dependency and the immediately relevant layer. |

That distinction matters. Installing an agent skill does not improve a kernel,
and running Hyperloom does not change the HIP runtime. Conversely, hipEngine
can benefit from Core SDK 7.14 without adopting either agent layer.

## hipEngine compatibility lens

Any candidate must fit these existing constraints:

1. **Torch-free generation path.** A Python package that imports PyTorch is not
   admissible inside the modules reached by `hipengine.LLM.generate()`.
2. **Four-axis registry.** Backend, layer, quant, and variant behavior remains
   registered, not branched through engine/model code.
3. **Raw-pointer device ABI.** New kernel bodies and launch wrappers stay
   compatible with raw device pointers and in-tree JIT/AOT ownership.
4. **`KVLiveSpans` attention ABI.** An external paged-attention implementation
   is not drop-in if it assumes only a block table and scalar context length.
5. **In-tree kernel development.** External repositories are read-only lineage
   and idea sources. Retained implementations, tests, and measurement live in
   `kernels/<backend>/` and `hipengine/`.
6. **Exact target support.** “ROCm supported” is insufficient. W7900 is
   `gfx1100` RDNA3; Ryzen AI Max+ 395 is `gfx1151` RDNA3.5. CDNA-only results
   are not transferable.
7. **Correctness before speed.** New or ported math must satisfy the CPU
   reference gate (KL <= 0.05 and top-1 >= 90%, with stronger bit/exact gates
   where the path already requires them).
8. **Anti-gaming and full-suite evidence.** MTP and sampling changes need the
   full multi-category prompt protocol plus heldouts, not a fixed-prompt lift.

### Current overlap with the Core SDK

hipEngine already uses the right architectural level:

- [`hipengine/core/hip.py`](../hipengine/core/hip.py) is a direct HIP runtime
  surface, including graph capture/replay.
- [`hipengine/core/hipblaslt.py`](../hipengine/core/hipblaslt.py) and
  [`hipengine/core/rocblas.py`](../hipengine/core/rocblas.py) are torch-free
  `ctypes` bindings.
- [`hipengine/core/build.py`](../hipengine/core/build.py) owns cached `hipcc`
  builds.
- [`scripts/mtp_verifier_rocprof.py`](../scripts/mtp_verifier_rocprof.py) and
  other profiling harnesses already isolate final children and require cached
  JIT artifacts, avoiding nested profiler/JIT corruption.
- the tree contains a versioned AOTriton runtime for selected prefill paths;
  that is an existing, separately gated integration, not permission to add a
  general framework dependency.

One especially concrete opportunity is visible in `hipblaslt.py`: the wrapper
queries up to 16 heuristic algorithms and its convenience selector defaults to
`preferred_index=4`, described as the measured fast gfx1151 heuristic. That can
be correct for the measured shapes and library version while still being
fragile across shapes, architectures, or hipBLASLt releases. The official
offline tuner is designed for exactly this problem.

### Target hardware in ROCm 7.14

ROCm 7.14's [hardware list and GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
now give both active targets first-class identities:

| Target | Public specification relevant to hipEngine | Consequence |
| --- | --- | --- |
| Radeon Pro W7900, `gfx1100` | 48 GiB VRAM, 96 CUs, wave32 or wave64, 128 KiB LDS per WGP, 96 MiB Infinity Cache, 6 MiB L2 | Native package, compiler, runtime, profiler, hipBLASLt, and rocWMMA experiments can be target-specific rather than “Navi family” guesses. |
| Ryzen AI Max+ PRO 395 / Radeon 8060S, `gfx1151` | Dynamic/carve-out memory, 40 CUs, wave32 or wave64, 128 KiB LDS, 32 MiB Infinity Cache, 2 MiB L2 | Unified/GTT memory policy and tool support are first-order concerns; support must never be inferred from `gfx1100` or generic `gfx11`. |

### Local toolchain snapshot

Read-only checks on 2026-08-05 showed:

| Item | Local value |
| --- | --- |
| HIP runtime | `libamdhip64.so` loads successfully |
| GPU | Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`, 40 CUs, wave32 |
| HIP compiler | development HIP 7.15.0, AMD Clang 23 |
| ROCprofiler | `rocprofv3` 1.3.2, ROCm 7.15.0 |
| AMD SMI | 26.5.0, ROCm 7.15.0 |
| ROCm SDK wheel | `7.15.0a20260711` |
| GTT visible through AMD SMI | 120.00 GB |

These facts prove that the APIs/tools are locally present; they do **not** make
a performance claim. The public production release reviewed here is ROCm
7.14.0, while the local environment is a newer development build. Any retained
benchmark must record the exact local component manifest rather than labeling
it merely “ROCm 7.”

## AMD Skills

### What it is

The [official `amd/skills` repository](https://github.com/amd/skills) packages
instructions, scripts, references, and conventions in the Agent Skills format
for tools including Codex, Claude Code, Cursor, and Gemini CLI. AMD explicitly
labels the catalog a **Tech Preview** and says to expect frequent changes. It
can be browsed with `npx skills add amd/skills --list` and installed
selectively, but installation is not needed to read the source.

A skill is an opinionated automation playbook. It can choose commands and edit
a workspace; it is neither a ROCm library nor an independent correctness
oracle. The safest policy for hipEngine is source review first, immutable
revision pinning second, and installation only for a bounded task.

### Current catalog assessment

| Skill | Status in catalog | Relevance |
| --- | --- | --- |
| `local-ai-use` | Available | **Reject now.** Routes image/TTS/STT requests to a Lemonade local-AI server. It is unrelated to LLM kernel optimization and could alter local agent setup. |
| `local-ai-app-integration` | Available | **Reject now.** Helps applications consume a Lemonade server. It overlaps product serving at a different layer and does not improve hipEngine compute. |
| `serving-llms-on-instinct` | Available | **Reference only for a future Instinct baseline.** Its own metadata excludes consumer/Radeon, Ryzen AI, MI250X, and MI100. It cannot be used for W7900 or gfx1151 work. |
| `serving-llms-on-epyc` | Available | **Low-priority reference.** Could inform a CPU fallback or baseline later, but it uses vLLM + zentorch and is not GPU optimization. |
| `magpie-kernel-evaluator` | Available, federated from Magpie | **Pilot candidate.** It accepts custom kernel compilation/test cases and emits structured analyze/compare results. Use only on an isolated microbenchmark; keep hipEngine's oracle and benchmark as authority. |
| `tracelens-analysis-orchestrator` | Available, federated from TraceLens | **Do not run as-is.** The skill is centered on PyTorch traces and agent/subagent reporting. The underlying TraceLens ROCprof parser is more relevant than this orchestration wrapper. |
| `apu-memory-tuner` | Planned | **Watch.** Directly relevant to gfx1151, but there is no shipped skill. ROCm 7.14's AMD SMI now exposes the underlying GTT/carve-out controls, so hipEngine need not wait for the skill. |
| `rocm-doctor` | Planned | **Watch.** Could help environment triage, but there is no usable artifact yet and hipEngine already has explicit HIP/toolchain preflight checks. |
| `hyperloom-kernel-optimizer` | Planned | **Watch, not actionable.** The current catalog points to a future skill, not a supported hipEngine optimizer. |
| `vllm-semantic-router` | Planned | **Reject for current scope.** It routes requests among platforms; it does not optimize hipEngine. |
| `hrr-replay-analysis` | Planned | **Watch.** Cross-hardware HIP record/replay could become useful for reproducible issue archives, but no current skill can be evaluated. |

### Catalog and supply-chain considerations

- The repository is MIT licensed, but a skill can invoke installers, edit agent
  configuration, create files, or launch containers. License permissiveness is
  not runtime safety.
- Federated sources currently name mutable `main` branches for Magpie and
  TraceLens. The import workflow records a commit in the vendored marker, but
  a fresh upstream install still needs revision capture.
- A catalog update can change instructions without changing hipEngine. A
  benchmark artifact must record the exact skill/repository commit if a skill
  participated in generating a candidate.
- Do not blanket-install the catalog into the repository. That would add broad
  agent behavior and high-conflict files without a concrete optimization unit.

**Recommendation:** inspect a specific skill at a pinned revision only when a
matching task arrives. The first plausible candidate is the Magpie kernel
evaluator for one non-production microbenchmark; even there, do not delegate
the final correctness or retention decision.

## Hyperloom

### What it is

[Hyperloom](https://github.com/AMD-AGI/Hyperloom) describes itself as an
autonomous optimization system for end-to-end LLM inference. Its high-level
pipeline combines:

| Component | Role | Current hipEngine fit |
| --- | --- | --- |
| [TraceLens](https://github.com/AMD-AGI/TraceLens) | Parse traces, generate workload/performance views, roofline summaries, and comparisons | **Best fit.** Its ROCprof input path can run offline and is described as hardware-agnostic. |
| [Magpie](https://github.com/AMD-AGI/Magpie) | Collect/evaluate kernels and framework benchmarks locally, in containers, or through Ray | **Possible microbenchmark fit.** Framework modes do not match hipEngine's benchmark contracts. |
| [IntelliKit](https://github.com/AMDResearch/intellikit) | Lower-level profiling support | **Watch.** Validated Hyperloom combinations are Instinct-centric. |
| [GEAK](https://github.com/AMD-AGI/GEAK) | Multi-agent kernel and vLLM/SGLang optimization | **Reject now.** Explicitly targets Instinct/CDNA and can launch highly privileged autonomous agent workflows. |
| [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena) | Kernel-agent evaluation environment | **Watch.** Current support matrix lists only MI300X/MI325X/MI355X. |
| Arbor | Tree-based long-horizon optimization/search | **Process reference only.** It is not a hipEngine runtime component. |

The documented phase flow establishes a baseline, profiles/constructs a
roofline, explores configuration and source patches, invokes kernel agents,
sweeps workload frontiers, and produces session artifacts. A Critic,
PolicyGate, accuracy gate, and runnable gate govern candidate retention. Those
are sound process ideas, especially reproducible baseline commands, durable
session state, novelty/stall detection, and end-to-end revalidation after each
kept change.

### Compatibility mismatch

The current [Hyperloom compatibility matrix](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/compatibility.rst)
is decisive:

| Requirement | Validated Hyperloom 1.0.0a2 path | hipEngine path |
| --- | --- | --- |
| GPU | MI300X, MI325X, MI355X | W7900 `gfx1100`; Ryzen AI Max+ `gfx1151` |
| Architecture | CDNA3/CDNA4 | RDNA3/RDNA3.5 |
| ROCm | 7.2.x | production analysis at 7.14; local development 7.15 |
| Serving framework | SGLang >= 0.5.12 or vLLM >= 0.21 | custom torch-free engine |
| Kernel languages | HIP, Triton, FlyDSL | primarily in-tree HIP plus selected separately gated libraries |

TraceLens is explicitly exempted from the hardware/ROCm constraint. GEAK,
IntelliKit, AgentKernelArena, and Magpie are not.

### Operational mismatch

Hyperloom's own [quickstart](https://github.com/AMD-AGI/Hyperloom/blob/main/examples/README.md)
recommends a dedicated clean workspace and Docker. Its setup can install a
wheel into the current directory, create or update `.env`, install/check ROCm
PyTorch and SGLang/vLLM, clone dependencies, and use an Anthropic-compatible
provider. That is appropriate for a self-contained optimizer demo and
inappropriate inside the shared hipEngine worktree.

GEAK's documented workflows target Instinct (`gfx942`/`gfx950`) and include an
example that launches Claude with `--dangerously-skip-permissions`. This is a
hard operational rejection for an unattended run against a repository with
expensive GPU benchmarks, append-only evidence, and a shared worktree.

### What is still worth borrowing

1. **TraceLens offline ROCprof analysis.** It accepts `rocprofv3` JSON and
   PFTrace, not only PyTorch traces. A disposable-environment pilot can analyze
   an already-produced hipEngine trace without touching the runtime.
2. **Trace comparison artifacts.** Structured baseline/candidate trace diffs
   can complement, not replace, existing compact benchmark JSON.
3. **Durable campaign state.** Manifest, state, candidate ledger, rejection
   reason, and final breakdown mirror good practices already present in
   hipEngine's WORKLOG/results process.
4. **Novelty and stop gates.** Detecting identical repeated attempts is useful
   protection against blind kernel tweaking.
5. **Frontier sweeps.** A candidate should survive context/batch/category
   frontiers, but hipEngine's full MTP suite, category heldouts, and exact
   correctness contracts remain stricter than a generic accuracy floor.

### Decision

Do not install or run Hyperloom in the hipEngine repository. Reconsider only
when its public matrix includes `gfx1100`/`gfx1151` and a generic command-based
engine contract that does not require PyTorch, SGLang, or vLLM. A future trial
must use a disposable clone/container, pinned commits, no secrets in prompts,
one leased GPU, a fixed wall-time/attempt budget, and human review before any
patch is moved into the real tree.

TraceLens alone can be piloted sooner under the same pinned/disposable rule.

## ROCm Core SDK 7.14 and TheRock

### Why the distribution change matters

[ROCm Core SDK 7.14 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
date the production release to 2026-07-15 and make TheRock the production build
and release system. The new distribution separates architecture-neutral host
code from architecture-specific kernel packs. The
[TheRock release guide](https://github.com/ROCm/TheRock/blob/main/RELEASES.md)
explains that one multi-architecture package index can select a target with a
`device-*` extra, reducing unrelated downloads.

This aligns unusually well with hipEngine's peer backend tree. A reproducible
environment can request only the device code it actually needs:

```bash
python -m pip install \
  --index-url https://repo.amd.com/rocm/whl-multi-arch/ \
  "rocm[libraries,devel,profiler,device-gfx1100]==7.14.0"
```

or replace the device extra with `device-gfx1151`. The exact valid extras and
commands should always come from the current
[installation page](https://rocm.docs.amd.com/en/develop/install/rocm.html);
the page explicitly publishes both `gfx1100` and `gfx1151` device packs.

The practical recommendation is:

- maintain separate, version-pinned gfx1100 and gfx1151 environments rather
  than one mutable “ROCm latest” environment;
- record `python -m rocm_sdk version`, `python -m rocm_sdk targets`,
  `python -m rocm_sdk path --root`, `hipcc --version`, `rocprofv3 --version`,
  `amd-smi version`, driver/kernel information, and package lock data with
  benchmark provenance;
- do not compare a production 7.14 result with a local 7.15-alpha result as if
  only hipEngine changed; and
- retain native `--offload-arch=gfx1100` or `gfx1151` builds. ROCm 7.14 lists a
  first-launch segfault for SPIR-V-targeted HIP kernels while native targets
  are unaffected.

TheRock is also a useful issue/source tracker for packaging, driver, and
architecture enablement. It is not a place for hipEngine kernel development;
the repository's in-tree lineage rule still applies.

### ROCm 7.14 changes that affect hipEngine

| Change or issue | Relevance | Recommended response |
| --- | --- | --- |
| HIP Execution Context APIs partition device resources, primarily CUs, and create context-scoped streams/events | Potential future QoS or multi-tenant isolation on one GPU; partitioning can reduce single-request throughput | **Research later.** Add only as an optional runtime/serving policy with registry-safe ownership and exact throughput/latency evidence. It is not a default speed optimization. |
| Batch async discard/prefetch APIs (`hipMemDiscardBatchAsync`, `hipMemPrefetchBatchAsync`, combined variants) | Could help future expert/weight paging or managed-memory residency with lower host API overhead | **Watch/pilot only when paging exists.** Current resident weights and preallocated workspaces should not be made pageable merely because the API exists. |
| Faster graph replay for graphs containing asynchronous allocation nodes | hipEngine already depends heavily on graph capture/replay | **Low immediate upside.** hipEngine generally preallocates; keep allocation nodes out of steady-state graphs. Revalidate graph timing on upgrade but do not add allocations to exploit the fix. |
| Graph capture recovery, node-ID race, child-graph synchronization, allocation-lifetime, and memset validation fixes | Direct correctness/stability benefit for graph-heavy code | **Upgrade value.** Run existing graph capture/update/regrow tests when moving to 7.14+ and record the exact runtime version. |
| Improved illegal-memory-access errors include host/GPU/kernel, plus exported `__hipOnError` and ROCgdb `catch hiperr` | Makes opaque kernel faults easier to localize | **Use for debugging.** Add to the documented failure playbook when the next nontrivial HIP fault is investigated. |
| Native architecture builds unaffected by SPIR-V first-launch segfault | Confirms current explicit native architecture policy | **Keep native code objects.** Do not switch the JIT cache to generic SPIR-V yet. |
| `cooperative_groups::reduce()` may be wrong or fail when block `.y` or `.z` is not 1 | Possible latent correctness risk for multidimensional workgroups | **Audit on use.** Search new/ported kernels for this combination and add an oracle before retention. |
| hipBLASLt strided-batched GEMM supports a distinct bias per matrix | May help future batched projections if layout and epilogue exactly match | **Shape-specific pilot only.** It is irrelevant to most c=1 custom-quant GEMV and cannot bypass quant layouts. |
| Radeon/Ryzen LLM workloads on PyTorch <2.14 can underperform unless hipBLASLt is selected | Independent confirmation that Lt is important on RDNA, even though the workaround itself is PyTorch-specific | **Keep direct hipBLASLt evaluation.** Never set a PyTorch environment variable in hipEngine and call it a win; tune the direct C API path. |
| vLLM 0.21-0.25 can have very long Radeon warmup; fixed in >=0.26 | Can contaminate startup and external-baseline comparisons | **Record warmup separately.** Do not use a vLLM 0.23 cold-start row as a clean engine comparison. |
| SGLang on Radeon should disable AITER and fused MLA; some MoE/Qwen ASR models fail | Demonstrates that generic “ROCm production” does not mean RDNA feature parity | **Do not use SGLang as an oracle for unsupported models.** If benchmarked, publish exact disabled features and validation scope. |
| ROCm Compute Profiler `per_kernel` normalized averages can be inflated | Can produce impossible Avg > Min/Max analysis | **Use another normalization** until fixed, and retain raw counters/commands. |
| gfx1151 Compute Profiler GL0 counters and max-memory-clock metadata have known gaps | Directly affects current Strix Halo profiling | **Fail closed.** Supply max memory clock with `--specs-correction`; do not use the zero `TCP_REQ_sum`-derived GL0 metric as evidence. |
| ROCm Bandwidth Test reaches end of life | Existing guidance can become stale | **Use TransferBench and RVS** for new transfer/topology work. Keep old RBT rows only as historical evidence. |
| RCCL 64-512 MB multi-node operations may regress | Future distributed inference risk | **Record before scaling.** Recheck the known issue and fault-injection workaround before any multi-node claim. |
| SPM counter sampling is beta, Instinct-only in 7.14, and can destabilize the system | Not usable on current gfx11 targets | **Reject now.** Do not enable beta SPM on production or infer Radeon support. |

### Libraries and compute components

The Core SDK product page lists a broad catalog. The following assessment
reviews the entire set rather than assuming every GPU library helps LLM
inference.

| Component | What it provides | Target/support reality | hipEngine decision |
| --- | --- | --- | --- |
| HIP runtime, ROCr, AMD LLVM/`hipcc` | Allocation, streams, events, graph APIs, module loading, compilation, launch | Core support for Instinct/Radeon/Ryzen | **Adopt.** This is hipEngine's foundation. Pin it, test upgrades, and keep raw-pointer wrappers small. |
| hipBLASLt | Flexible GEMM, heuristics, epilogues, workspace, offline tuning | ROCm 7.14 lists Linux/Windows support across Instinct/Radeon/Ryzen | **Adopt and tune.** Highest-value external math library for dense prefill/batched shapes. |
| rocBLAS / hipBLAS | BLAS GEMM/GEMV and portability layer | Broad target support | **Keep fallback/reference paths.** Compare per exact shape; do not assume Lt always wins. |
| AITER | Attention, paged attention, MoE, GEMM, norms, quantization, sampling, communication | Fully supported on MI300/MI350; W7900 and gfx1151 are explicitly experimental, with most CK/ASM kernels CDNA-only | **Reference plus isolated source pilots.** No normal AITER Python install in the runtime. See the dedicated section below. |
| Composable Kernel / CK-Tile | C++ template and tile abstractions for ML kernels | ROCm 7.14 lists CK for Instinct/Radeon; current config recognizes gfx1151 as gfx11, but individual device instances vary | **Reference/narrow prototype.** Best suited to compute-heavy prefill or fused ops, not automatic replacement of decode kernels. |
| rocWMMA | Header-only matrix-fragment/matrix-core API | Official current list includes gfx1100/1101/1102 and gfx12, but not gfx1151 | **gfx1100 pilot; gfx1151 unverified.** Preserve hand-tuned fallback and prove trace/correctness for every target. |
| hipCUB, rocPRIM, rocThrust | Reductions, scans, sorting, selection, and parallel primitives | Broad ROCm component support | **Reference or build-time header use.** Candidate for top-k/scan primitives only if a raw-ABI in-tree wrapper beats the current path. Template/code-size cost matters. |
| hipRAND / rocRAND | GPU random-number generation | Broad target support | **Optional oracle/reference.** Sampling reproducibility and distribution correctness matter more than library adoption; keep the native sampler hot path unless a full sampling suite proves a win. |
| MIOpen | Convolution, normalization, activation, and deep-learning primitives | Broad math-component support, but typical consumption is framework-oriented | **Conditional for Moonshine/VLM convolution.** Benchmark a raw C path only where convolution dominates; not relevant to standard LLM decode. |
| hipTensor | Tensor contractions and permutations | Current docs include gfx1100 support | **Low relevance.** Transformer hot paths are not general tensor contractions. Consider only if a concrete contraction/permutation dominates. |
| MIGraphX | AMD graph inference compiler/runtime | ROCm 7.14 AI matrix validates only gfx942/gfx950 | **Reject now.** Hardware mismatch and wrong architectural layer. |
| ONNX Runtime ROCm execution path | ONNX graph inference | ROCm 7.14 validates only gfx942/gfx950 in the AI matrix | **Reject now.** Could be an offline functional reference on supported hardware, not a gfx11 backend. |
| hipSPARSE / rocSPARSE | General sparse matrix operations, including BSR SpMM/SpMV additions | Broad library distribution | **Low relevance.** Unstructured/general sparsity does not match current packed quant GEMV. Revisit for a measured block-sparse model only. |
| hipSPARSELt | Structured sparse matrix multiplication | ROCm 7.14 lists only Instinct gfx942/gfx950 | **Reject for gfx11.** |
| hipFFT / rocFFT | Fast Fourier transforms | Broad support | **No current LLM relevance.** Possible future audio/signal preprocessing only. |
| hipSOLVER / rocSOLVER | Dense/sparse factorizations and solvers | Broad support | **No generation-path relevance.** Offline quant calibration/research might use it outside the runtime. |

The core libraries are increasingly consolidated under
[`ROCm/rocm-libraries`](https://github.com/ROCm/rocm-libraries). Older
component repositories can be read-only mirrors. Record the actual source
location and commit whenever an idea is ported.

### AITER and Opus in more detail

The current [AITER repository](https://github.com/ROCm/aiter) is more relevant
than its Instinct history alone suggests. It now labels W7900 `gfx1100`, Ryzen
AI Max `gfx1151`, and Radeon AI Pro `gfx1201` as **experimental**. Its note is
important: Triton, most FlyDSL, and many HIP norm/RoPE/quant/activation kernels
can run on RDNA, while most CK and assembly kernels remain CDNA-only.

Useful read-only material includes:

- paged/MHA/MLA attention tests and layouts;
- fused MoE and grouped/top-k routing experiments;
- RMSNorm, RoPE+KV, quantization, sampling, skinny GEMM, and communication
  operator tests;
- the experimental JAX FFI bridge, which demonstrates operator exposure
  without a PyTorch dependency; and
- its lightweight [Opus header](https://github.com/ROCm/aiter/tree/main/csrc/include/opus).

Opus deliberately sits between handwritten HIP and CK. It offers lightweight
layout helpers, vectorized global loads/stores, type conversions, and matrix
instruction wrappers, but explicitly does **not** ship pre-optimized GEMM,
attention, or reduction pipelines. That makes it an implementation aid and
source reference, not an operator library.

Its compile-time guidance is unusually compatible with hipEngine:

- replace the huge HIP runtime header in device code with AMDGCN builtins or a
  minimal header;
- use `-D__HIPCC_RTC__` where compatible to suppress implicit headers;
- use `hipcc --genco --offload-arch=<native gfx>` for device-only code objects;
- load predictable `extern "C"` kernels with HIP module APIs; and
- avoid PyTorch/pybind binding compilation, which hipEngine already does via
  `ctypes`.

**Bounded experiment:** compile one representative, non-production kernel both
through the existing shared-object path and a device-only HSACO/module path.
Measure cold compiler wall, artifact size, cached-load wall, launch overhead,
graph-capture compatibility, exact output, and `rocprofv3` kernel identity.
Retain only if it improves build/load cost without broadening the runtime ABI or
slowing steady state. Port individual ideas with AITER commit/source provenance;
do not vendor the whole package.

### hipBLASLt offline tuning

The official [offline tuning guide](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to-use-hipblaslt-offline-tuning.html)
states that `hipblaslt-bench` can find the best solution for a GEMM problem.
Setting `HIPBLASLT_LOG_MASK=32` emits a replayable bench command. A tuning run
can write results through `HIPBLASLT_TUNING_FILE`; a later process can load them
through `HIPBLASLT_TUNING_OVERRIDE_FILE`.

The most important warning is also explicit: **solution indices cannot be
reused across library releases or different device architectures**. For
hipEngine, a tuning record therefore needs at least:

```text
ROCm build + hipBLASLt version + gfx target + device identity
M/N/K + batch count + transposes + leading dimensions/strides
input/output/compute/scale types + epilogue/bias + workspace ceiling
exact source/call-site identity + correctness checksum + benchmark command
```

A global override file should not silently control every benchmark. The safe
pilot is:

1. capture actual hipEngine GEMM descriptors for a bounded list of retained
   model shapes;
2. run the official tuner on gfx1100 and gfx1151 separately;
3. compare current heuristic selection, first heuristic, fixed position, and
   tuned solution with rotating buffers and sufficient warmup;
4. validate complete outputs through the applicable CPU/model gate; and
5. register a version/target/shape-qualified selection with an exact fallback,
   or reject the tuning result.

This is a stronger path than changing `preferred_index` globally based on one
shape or one library build.

### Communications, media, and data movement

| Component | What it provides | hipEngine decision |
| --- | --- | --- |
| RCCL | Multi-GPU/multi-node collectives; ROCm 7.14 lists Instinct/Radeon/Ryzen | **Future backend/plugin.** Add only with real multi-GPU hardware and topology-aware correctness/perf evidence. Single-GPU work gains nothing. |
| rocSHMEM | GPU/host one-sided communication and collectives | **Future research.** 7.14 lists gfx1100 Radeon but not gfx1151; not needed before distributed/expert scaling. |
| rocAL | Accelerated data loading/augmentation | **Boundary-only.** Could help future multimodal preprocessing, never transformer decode. |
| RPP | Image/audio preprocessing primitives | **Boundary-only.** Same rule as rocAL; compare to simple CPU/native paths before adding a dependency. |
| MIVisionX | Vision graph/primitives | **Not current.** Wrong layer for LLM compute; possible future VLM preprocessing reference. |
| rocDecode | Hardware video decode | **Future VLM/video ingestion.** Current docs cover Navi3x/gfx1100 and 7.14 lists gfx1151 media support. Keep outside model kernels and measure end-to-end ingestion. |
| rocJPEG | Hardware JPEG decode | **Future image ingestion.** Not a decode-token optimization. |
| hipFile | Direct storage-to-GPU I/O | **Reject for current targets.** ROCm 7.14 lists it for Instinct only; model load is not the steady-state decode bottleneck. |

ROCm Bandwidth Test should no longer be the default recommendation because
7.14 declares it end-of-life. [TransferBench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html)
can exercise simultaneous CPU/GPU transfers using CPU, GPU, or SDMA executors.
It is useful for:

- establishing H2D/D2H and concurrent-copy baselines;
- understanding gfx1151 shared-memory and transfer behavior;
- evaluating future asynchronous model/expert loading; and
- separating topology/transfer limits from kernel limits.

TransferBench does not measure the streaming weight bandwidth achieved by an
LLM kernel and must not replace operation-specific roofline/counter evidence.

### Profiling, debugging, and monitoring

| Tool | Best use for hipEngine | Limits/cautions | Decision |
| --- | --- | --- | --- |
| `rocprofv3` / ROCprofiler-SDK | Kernel dispatch/duration, HIP API, memory-copy/allocation/scratch, ROCTx, and hardware-counter traces | Profiling perturbs timing; counters can require replay/multiplexing; never compile JIT code inside the profiled child | **Adopt/deepen.** This is the primary low-level evidence path. |
| `rocprofv3-avail` | Enumerate agent properties and counters supported on the exact GPU | Availability is architecture/build-specific | **Adopt.** Store selected counter names and agent properties with the profile. |
| ROCm Compute Profiler (`rocprof-compute`) | Kernel-level counters, selected metric blocks, baseline comparisons, and roofline analysis | Primarily Instinct-oriented; 7.14 lists Ryzen support but not Radeon. gfx1151 has GL0/max-mclk issues and RDNA3.5 roofline is still described as upcoming | **Conditional.** Use on gfx1151 with corrections; do not assume W7900 support. |
| ROCm Systems Profiler (`rocprofiler-systems`) | CPU+GPU timeline, Python/ctypes scheduling, call stacks, allocations, streams, SDMA, page faults, and host gaps | Broader and heavier than a kernel trace; Python bindings/interpreter must match | **Pilot.** Useful when host-minus-device wall or scheduler stalls are the target. |
| ROCprof Compute Viewer | Visualize thread-trace/SQTT data | Early access; not for production workloads | **Watch.** Potential deep instruction/source analysis on a reproduced kernel. |
| ROCTx markers | Label phases/requests/layers in traces | Marker overhead and capture placement must be measured | **Use selectively.** Especially useful in final-child/system traces, not necessarily every production launch. |
| AMD SMI | Capture clocks, temperatures, power, activity, memory, throttling, and APU-specific metrics | Sampling interval and permissions matter; changing GTT/carve-out is reboot-scoped | **Adopt for metadata; controlled tuning only.** |
| `rocminfo` | Identify GPU/ISA, CUs, wave size, memory/cache/runtime properties | Static inventory, not a performance measurement | **Adopt.** Already part of preflight. |
| ROCgdb / ROCdbgapi | Source-level debugging and HIP error catchpoints | Debug builds and debugger attachment can alter behavior | **Use for faults, not perf.** `catch hiperr` and `__hipOnError` are useful new paths. |
| ROCr Debug Agent | Dump wavefront state after queue errors | Diagnostic only | **Use for hangs/faults.** |
| ROCm Validation Suite (RVS) | Platform and transfer validation | More valuable on clusters than one workstation | **Use for new machine/driver bring-up.** It replaces some retired bandwidth-test use cases. |
| HIPIFY | Translate CUDA source to HIP | Mechanical translation does not establish numerical/performance suitability | **Porting aid only.** Follow the in-tree lineage and RED-test workflow afterward. |
| RDC | Data-center telemetry/management | ROCm 7.14 lists Instinct only | **Reject for current workstations.** |

The [rocprofv3 usage guide](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html)
documents a useful broad trace:

```bash
rocprofv3 --runtime-trace --output-format json -- <cached-final-child>
```

Runtime trace covers HIP API calls, markers, kernels, memory copies,
allocations, and scratch behavior. For hipEngine this command is a template,
not permission to wrap a parent harness. Continue to prebuild shared objects,
precompute the compiler-version file, require cached JIT, and profile the final
child as required by [KERNELS.md](KERNELS.md).

[`rocprofv3-avail`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3-avail.html)
is immediately useful because the official documentation itself demonstrates
W7900/gfx1100 agent enumeration and derived counters such as
`ALUStalledByLDS`. A small, reviewed counter set can test a roofline diagnosis:
memory traffic/utilization, VALU/matrix utilization, occupancy/resource
pressure, LDS conflicts/stalls, cache behavior, and launch/copy wall. Counter
names are not portable enough to hardcode without an availability check.

The [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html)
is the better tool when the question is not “why is this kernel slow?” but
“where did the rest of the token wall go?” It supports CPU/GPU tracing, Python
scripts, call-stack sampling, allocations, unified memory, SDMA, and
system-level metrics. It succeeds the older Omnitrace name shown in some
diagrams.

### AMD SMI and gfx1151 memory policy

ROCm 7.14 / AMD SMI 26.5 adds APU metrics and consolidates APU GTT and VRAM
carve-out management that previously relied on `amd-ttm`. This is directly
relevant to the Strix Halo memory campaign:

- `amd-smi node --gtt` provides a read-only GTT size record;
- APU metrics include graphics/SoC temperatures, graphics activity, average
  socket/CPU/SoC/GPU power, clocks, voltages/current, and throttle state; and
- carve-out/GTT changes are privileged, rebuild boot configuration/initramfs,
  and take effect after reboot.

Therefore:

1. add the read-only GTT and APU metric snapshot to hardware/benchmark
   provenance where available;
2. never alter GTT or carve-out as an incidental benchmark setup step;
3. if memory policy is tested, create a dedicated experiment with before/after
   boot verification, the same full workload matrix, and restoration steps;
4. measure both capacity and performance. A larger GTT ceiling is not a speed
   win if it increases faults/migration or changes CPU memory pressure; and
5. keep environment policy outside backend dispatch/model code.

This makes the planned AMD `apu-memory-tuner` skill less urgent: the supported
primitive exists now, while hipEngine's own experiment protocol is stricter
than the future skill can be assumed to provide.

## ROCm AI ecosystem: frameworks, quantization, and baselines

### vLLM and SGLang

The [ROCm AI ecosystem portal](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/index.html)
covers PyTorch/JAX, vLLM, SGLang, ATOM, MIGraphX, ONNX Runtime, distributed
inference, training, quantization, and operator optimization. Most recipes are
framework-level and Instinct-first. For hipEngine, vLLM and SGLang are useful
as:

- external serving baselines on exactly supported model/hardware versions;
- source references for scheduling, paged attention, MoE, and AITER integration;
- compatibility probes for ROCm releases; and
- independent full-model output/quality references where supported.

They are not useful as:

- hot-path dependencies;
- proof that an AITER/CK kernel supports gfx11;
- apples-to-apples performance evidence without matching model, quant,
  prompt/output, concurrency, warmup, serving mode, and hardware; or
- substitutes for `kernels/cpu_reference/`.

ROCm 7.14 validates vLLM 0.23 on both gfx1100 and gfx1151, while SGLang 0.5.13
is listed for gfx1100-class Radeon but not gfx1151. The same release warns that
vLLM 0.21-0.25 can have long Radeon warmup and recommends >=0.26 for the fix.
Consequently, an external comparison should separate:

1. the AMD-validated 7.14 container/package baseline, with its known warmup
   issue recorded; and
2. a later upstream vLLM baseline, with its exact ROCm/framework commit and
   compatibility status recorded.

ROCm's PyTorch workaround `TORCH_BLAS_PREFER_HIPBLASLT=1` is not a hipEngine
setting. It does, however, justify checking whether an external baseline really
used hipBLASLt and reinforces the direct hipBLASLt tuning pilot.

### AOTriton

hipEngine already contains a separately versioned AOTriton runtime for
selected attention paths. Continue to track the official
[`ROCm/aotriton`](https://github.com/ROCm/aotriton) repository and its packaged
kernel images, but treat every version/target/shape as an explicit plugin
variant with an unfused/reference fallback. The presence of AOTriton in a
framework does not validate hipEngine's `KVLiveSpans` ABI, graph behavior, or
full model outputs.

### Quark

[AMD Quark](https://github.com/amd/Quark) is a cross-platform model optimizer
for PyTorch and ONNX inputs. Its current public feature table includes INT4,
INT8, FP8, MX formats, weight-only/dynamic/static quantization, per-group
quantization, SmoothQuant, AWQ, GPTQ, QuaRot, FP8 KV-cache quantization, and
exports including JSON+safetensors and GGUF Q4_1. The
[Quark documentation](https://quark.docs.amd.com/latest/) and
[open-source announcement](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-quark-model-optimization-library-now-available-as-open-sourc.html)
emphasize runtime-agnostic export.

This is relevant offline, not at generation time:

- use Quark as a checkpoint/scale/rotation generator or importer-fixture
  source;
- do not add it to `pyproject.toml` as a hard runtime dependency;
- compare exported tensor packing, zero points, group sizes, scale precision,
  rotation order, and metadata against hipEngine's quant registry;
- add a golden fixture before implementing a new importer/quant path; and
- run full perplexity/KL/top-1/task quality plus exact performance protocols.

Quark exporting “GGUF” does not mean its layout equals PARO, AWQ, GPTQ, or an
existing hipEngine packed kernel. FP8/MX availability also does not create
hardware FP8 matrix support on gfx1100; [PLAN.md](PLAN.md) correctly keeps that
target distinction.

### Other ecosystem entries

| Offering | Relevance |
| --- | --- |
| ATOM | AITER-native serving stack; useful as an Instinct/AITER reference, not a gfx11 dependency. |
| Infera / MoRI | Distributed inference and communication recipes; revisit with real multi-GPU/multi-node scope. |
| Primus | Large-scale training platform; outside inference-engine scope. |
| xDiT / ComfyUI | Diffusion serving/app ecosystem; irrelevant to current LLM kernels. |
| AI Developer Hub / Playbooks | Selective source and profiler tutorials can be useful, but recipes are often PyTorch/Instinct-specific. Reproduce locally rather than copying environment flags. |

## Developer programs, containers, and enterprise deployment

The surrounding resource links are useful mainly for future hardware access
and deployment context:

| Resource | What it offers | hipEngine relevance |
| --- | --- | --- |
| [AMD AI Developer Program](https://developer.amd.com/ai-developer-program/) | Free membership, currently advertised $100 AMD Developer Cloud credit, training, events, and community access | **Potentially useful** for an isolated Instinct portability/baseline smoke. Cloud credits and terms are time-sensitive; never treat availability as guaranteed. |
| [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html) | Containers and deployment guides for HPC/AI on Instinct | **External baseline/bring-up only.** Not a W7900/gfx1151 environment. |
| [ROCm AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/) | Tutorials and playbooks | **Reference selectively.** Prefer profiling/kernel tutorials that state target hardware and versions. |
| [Enterprise AI Reference Stack](https://enterprise-ai.docs.amd.com/en/latest/) | Technical-preview Kubernetes platform, AI workbench, resource manager, inference microservices, and solution blueprints | **Future serving/deployment context.** Quickstart validation is on MI300X/MI325X/MI350X/MI355X, not current gfx11 workstations. |

If hipEngine later targets Instinct or Kubernetes, these resources can help
answer deployment questions—GPU operator, telemetry, scheduling, OpenAI API
packaging, container pinning. They should not pull Kubernetes, AIM, PyTorch,
or vLLM concepts into the current kernel/runtime architecture prematurely.

## Recommended experiment plan

No item below is a retained win until it passes hipEngine's normal evidence
policy. Estimated priorities express likely value, not measured speedup.

### P0: reproducible ROCm manifests

**Hypothesis:** current benchmark artifacts under-specify the rapidly changing
TheRock/tool/library composition even when they name a top-level ROCm version.

Capture, through one reusable read-only helper or artifact section:

```bash
python -m rocm_sdk version
python -m rocm_sdk targets
python -m rocm_sdk path --root
hipcc --version
rocprofv3 --version
amd-smi version
amd-smi node --gtt
rocminfo
```

Also capture driver/kernel, device identity, relevant package lock, and AMD SMI
clock/power/throttle state. Keep volatile/full logs outside Git; retain a
compact normalized subset. Success is reproducible platform identity without
materially changing benchmark wall or creating privileged setup requirements.

### P0: hipBLASLt shape tuner

**Hypothesis:** at least one retained dense prefill/batched shape has a faster
supported algorithm than the current fixed heuristic position.

- Extract only actual production descriptors.
- Tune gfx1100 and gfx1151 separately with a bounded workspace equal to the
  runtime contract.
- Compare complete call wall and end-to-end relevant sub-window, not tuner
  kernel time alone.
- Gate exact outputs and model correctness.
- Reject any global override not qualified by library build and target.

Success is an exact, same-suite non-regressive improvement or a durable
negative result showing the current heuristic is sufficient.

### P0: standardized counter trace plus TraceLens pilot

**Hypothesis:** one existing target kernel/family has a profiler-supported
bottleneck classification that narrows the next optimization more than launch
duration alone.

1. Use the existing non-profiled cache warmup and final-child profiler rule.
2. Enumerate counters on the exact agent.
3. Collect the smallest counter/runtime trace that answers one question.
4. Retain raw command, selected dispatches, counter definitions, and profiler
   version; do not promote profiler-perturbed token throughput.
5. In a disposable Python environment, run TraceLens on a copy of the JSON or
   PFTrace and compare its report with hipEngine's existing attribution.

Keep TraceLens only if it parses gfx11 traces correctly and adds a repeatable
actionable view. Reject it if it requires PyTorch metadata, loses kernel
identity, or merely reformats existing summaries.

### P1: Opus/device-only compile pilot

**Hypothesis:** minimal device headers or HSACO-only compilation reduces cold
JIT wall and cache artifact size without changing launch/graph behavior.

Use one isolated kernel, two native targets, exact output, cached and cold
measurements, and kernel-trace confirmation. This experiment is about build and
load time, not token throughput unless steady-state launch mechanics also
change. Any ported helper must be small, attributed, target-audited, and
maintained in-tree.

### P1: system-level host-gap trace

**Hypothesis:** a measured host-minus-device token wall contains an attributable
Python/ctypes/synchronization/allocation gap.

Run ROCm Systems Profiler only on the already isolated final process. Use ROCPD
where Python child-thread capture requires it. Compare the system trace to the
same-scope `rocprofv3` kernel/HIP API trace and existing wall attribution. Do
not make an optimization until a stable gap has an owner.

### P1: Quark importer fixture

**Hypothesis:** Quark can reproducibly generate one quantized artifact whose
layout maps cleanly to an existing or justified new quant plugin.

Start with a tiny fixture and inspect metadata/packing before full weights.
Prefer a format already understood by a current importer. A new quant path
requires its own registry key, CPU oracle, quality gate, model-suite evidence,
and kernel work; a converter-only success is not a runtime performance win.

### P2: controlled gfx1151 GTT/carve-out matrix

**Hypothesis:** a different supported memory policy increases usable model/KV
capacity or reduces memory pressure without regressing generation.

This is a privileged, rebooted system experiment and must be scheduled
separately. Record the original state and recovery steps. Use the same four
context depths, same model/quant, fresh processes, whole-GTT/physical ownership,
fault/migration data, and exact outputs. Capacity and speed are separate claims.

### P2: multi-GPU RCCL plugin

Only start with appropriate hardware. Define the plugin/registry boundary,
topology, process model, graph behavior, collective sizes, failure semantics,
and exact single-GPU baseline first. Recheck the release's multi-node known
issues and never import MI350-specific tuning flags onto Radeon/Ryzen without
measurement.

### Watch list

- Hyperloom support for gfx1100/gfx1151 and generic command-driven engines;
- AITER promotion of RDNA from experimental, especially raw C++/FFI operator
  surfaces and tuned RDNA attention/MoE/GEMM kernels;
- rocWMMA's explicit gfx1151 support list and real compiler/runtime coverage;
- ROCm Compute Profiler's complete RDNA3/RDNA3.5 roofline and counter fixes;
- the planned `rocm-doctor`, `apu-memory-tuner`, and HIP record/replay skills;
- stable ROCprof Compute Viewer/thread-trace support for Navi;
- ROCm 7.15 production notes relative to the local alpha environment; and
- TheRock package/ABI changes affecting cached kernel code and BLAS solution
  selection.

## Integration and evidence guardrails

Any experiment derived from this review must preserve the project rules:

1. **No new torch hot-path dependency.** Quark, AITER Python, vLLM, SGLang,
   TraceLens, and Hyperloom remain offline/reference tools unless an explicit
   architectural decision changes that invariant.
2. **No backend/quant branches in engine code.** External kernels become
   registered variants with exact fallback chains.
3. **No direct external-tree development.** Port the smallest idea in-tree,
   cite file and commit, run `scripts/check_lineage.py`, and update
   [KERNELS.md](KERNELS.md) when paths/parents change.
4. **No framework headline reuse.** Upstream AITER/Hyperloom/vLLM speedups are
   not hipEngine evidence.
5. **No hidden global tuning state.** Record and qualify hipBLASLt override
   files, environment variables, GPU memory policy, clocks, and profiler
   configuration.
6. **No profiler throughput claim.** Profiler runs prove attribution and
   counters. Non-profiled exact benchmark runs prove performance.
7. **No benchmark gaming.** Use the complete prompt/category/heldout gates and
   true no-MTP baseline where required.
8. **Keep exact fallbacks.** An AITER/CK/rocWMMA/fused path needs an unfused or
   CPU-reference chain and must fail closed on unsupported targets/shapes.
9. **Respect expensive-validation policy.** Use the narrowest test first and
   ask before repeating an equivalent >5-minute run.
10. **Version everything.** ROCm umbrella version alone is insufficient for
    TheRock packages, hipBLASLt solution indices, profilers, or agent tools.

## Source and repository index

### Primary product and documentation sources

- [ROCm.AI landing page](https://www.amd.com/en/products/software/rocm.html)
- [ROCm Core SDK product page](https://www.amd.com/en/products/software/rocm/sdk.html)
- [ROCm Core SDK 7.14 release notes](https://rocm.docs.amd.com/en/docs-7.14.0/about/release-notes.html)
- [ROCm installation and TheRock wheel extras](https://rocm.docs.amd.com/en/develop/install/rocm.html)
- [TheRock transition guide](https://rocm.docs.amd.com/en/7.13.0-preview/about/transition-guide-TheRock.html)
- [ROCm component overview](https://rocm.docs.amd.com/en/latest/about/what-is-rocm.html)
- [ROCm GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
- [ROCm AI ecosystem](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/index.html)
- [vLLM on ROCm and known issues](https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/inference/vllm.html)
- [ROCm on Radeon and Ryzen](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)
- [RDNA3.5 system optimization](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
- [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [ROCm environment-variable index](https://rocm.docs.amd.com/en/latest/reference/environment-variables/index.html)

### Libraries and kernel sources

- [ROCm libraries monorepo](https://github.com/ROCm/rocm-libraries)
- [AITER](https://github.com/ROCm/aiter)
- [AITER Opus](https://github.com/ROCm/aiter/tree/main/csrc/include/opus)
- [Composable Kernel documentation](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/)
- [Composable Kernel repository/mirror](https://github.com/ROCm/composable_kernel)
- [CK-Tile concepts](https://rocm.docs.amd.com/projects/composable_kernel/en/develop/conceptual/ck_tile/index.html)
- [rocWMMA API and supported architectures](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/api-reference/api-reference-guide.html)
- [hipBLASLt documentation](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html)
- [hipBLASLt offline tuning](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/how-to-use-hipblaslt-offline-tuning.html)
- [AOTriton](https://github.com/ROCm/aotriton)
- [Quark](https://github.com/amd/Quark)
- [Quark documentation](https://quark.docs.amd.com/latest/)
- [rocDecode format/architecture support](https://rocm.docs.amd.com/projects/rocDecode/en/latest/reference/rocDecode-formats-and-architectures.html)

### Profiling, debugging, monitoring, and transfer sources

- [ROCprofiler-SDK / `rocprofv3`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3.html)
- [`rocprofv3-avail`](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-rocprofv3-avail.html)
- [ROCm Compute Profiler](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/what-is-rocprof-compute.html)
- [ROCm Systems Profiler](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/index.html)
- [ROCm profiling/debugging overview](https://rocm.docs.amd.com/en/latest/components/profilers-and-debuggers.html)
- [AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/index.html)
- [TransferBench](https://rocm.docs.amd.com/projects/TransferBench/en/latest/index.html)
- [ROCm Bandwidth Test](https://rocm.docs.amd.com/projects/rocm_bandwidth_test/en/latest/)
- [ROCm examples](https://github.com/ROCm/rocm-examples)

### Agentic and workflow sources

- [AMD Skills catalog](https://github.com/amd/skills)
- [Hyperloom](https://github.com/AMD-AGI/Hyperloom)
- [Hyperloom compatibility](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/compatibility.rst)
- [Hyperloom quickstart](https://github.com/AMD-AGI/Hyperloom/blob/main/examples/README.md)
- [Hyperloom optimization loop](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/conceptual/optimization-loop.md)
- [TraceLens](https://github.com/AMD-AGI/TraceLens)
- [TraceLens ROCprof report path](https://github.com/AMD-AGI/TraceLens/blob/main/docs/how-to/generate-perf-report-rocprof.md)
- [Magpie](https://github.com/AMD-AGI/Magpie)
- [GEAK](https://github.com/AMD-AGI/GEAK)
- [AgentKernelArena](https://github.com/AMD-AGI/AgentKernelArena)
- [IntelliKit](https://github.com/AMDResearch/intellikit)

### Deployment and developer resources

- [AMD AI Developer Program](https://developer.amd.com/ai-developer-program/)
- [AMD Infinity Hub](https://www.amd.com/en/developer/resources/infinity-hub.html)
- [ROCm AI Developer Hub](https://rocm.docs.amd.com/projects/ai-developer-hub/en/latest/)
- [AMD Enterprise AI Reference Stack](https://enterprise-ai.docs.amd.com/en/latest/)

### Moving-repository snapshot

For auditability, these were the remote default-branch `HEAD` values observed
on 2026-08-05. They are not ROCm release tags and do not replace a package
manifest; links intentionally point to the immutable snapshot.

| Repository | Observed commit |
| --- | --- |
| AMD Skills | [`9d1ba59ac468fa590c8ca63248698b7dfa7b8fbf`](https://github.com/amd/skills/tree/9d1ba59ac468fa590c8ca63248698b7dfa7b8fbf) |
| Hyperloom | [`a56bf465ab87a5bf923584ca11ddae9c3fe677df`](https://github.com/AMD-AGI/Hyperloom/tree/a56bf465ab87a5bf923584ca11ddae9c3fe677df) |
| TraceLens | [`5c081edadc829259563684276ac8ee7c0a2c2510`](https://github.com/AMD-AGI/TraceLens/tree/5c081edadc829259563684276ac8ee7c0a2c2510) |
| Magpie | [`ce74fb8be5ab8ec432fdd2973292dde66634d06c`](https://github.com/AMD-AGI/Magpie/tree/ce74fb8be5ab8ec432fdd2973292dde66634d06c) |
| GEAK | [`ee2b8ae98cbd3718a078cf08968be97b1db43449`](https://github.com/AMD-AGI/GEAK/tree/ee2b8ae98cbd3718a078cf08968be97b1db43449) |
| AgentKernelArena | [`2623c6fc8ce3a1fbb5cbe3b11f9efc74ae19d79f`](https://github.com/AMD-AGI/AgentKernelArena/tree/2623c6fc8ce3a1fbb5cbe3b11f9efc74ae19d79f) |
| IntelliKit | [`3a4fd131aaeab9b09d0b84b1c76bb33723762d35`](https://github.com/AMDResearch/intellikit/tree/3a4fd131aaeab9b09d0b84b1c76bb33723762d35) |
| AITER | [`de9f1f84a3b0fda34ec613842e8a012f5e6c4da6`](https://github.com/ROCm/aiter/tree/de9f1f84a3b0fda34ec613842e8a012f5e6c4da6) |
| TheRock | [`ac9ec78c7eb1924d586887e7631731ed35354eb3`](https://github.com/ROCm/TheRock/tree/ac9ec78c7eb1924d586887e7631731ed35354eb3) |
| Composable Kernel mirror | [`a39b67be27f8ed3d76e9b0ab161d2f2c5f439666`](https://github.com/ROCm/composable_kernel/tree/a39b67be27f8ed3d76e9b0ab161d2f2c5f439666) |
| Quark | [`1b229f781a1974cc742884e42d8eefc1eebb4f0a`](https://github.com/amd/Quark/tree/1b229f781a1974cc742884e42d8eefc1eebb4f0a) |

## Bottom line

ROCm.AI is relevant, but not because hipEngine needs another framework or an
autonomous optimizer. The important shift is that AMD now exposes a more
modular, explicitly gfx1100/gfx1151-capable Core SDK and a richer set of
observability, tuning, and APU management surfaces around the exact low-level
runtime hipEngine already chose.

The strongest opportunities are concrete and bounded: version-complete
TheRock manifests, direct hipBLASLt offline tuning, richer `rocprofv3` evidence,
AMD SMI APU metadata, one TraceLens offline trial, and one Opus/device-only JIT
experiment. AITER/CK/rocWMMA/Quark are useful when treated as target-qualified
sources or offline tools. Hyperloom and broad AMD Skills installation should
wait until their supported hardware and engine contracts actually match the
project.
