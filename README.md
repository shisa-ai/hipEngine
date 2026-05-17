# hipENGINE

hipENGINE is a ROCm-native local LLM inference engine designed from the ground
up for AMD RDNA GPUs (starting with gfx1100, gfx1151). It pairs a small 
purpose-built Python host with a complete suite of custom-tuned HIP kernels 
developed through 100+ iterations of profiling and tuning.

hipENGINE has lightweight dependencies with no PyTorch required for fully
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

**v0.1.x.** The runtime hot path is torch-free by construction; kernel
families and registry plumbing are landing under
[`hipengine/kernels/hip_gfx1100/`](hipengine/kernels/hip_gfx1100/). Current
single-model tuning targets
[shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format.

While we are far from [gfx1100 roofline](https://github.com/shisa-ai/hipENGINE/blob/main/docs/ROOFLINE.md), currently, our implementation does well compared to Q4_K_M quants of recent llama.cpp builds (`b9042`):

### Prefill tok/s

| Workload | hipENGINE shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | **2518.836** | 2436.049 | 1816.927 |
| 4K/128 | **2711.013** | 2176.905 | 1705.093 |
| 32K/128 | **2130.562** | 1496.409 | 1128.554 |
| 128K/128 | **1048.543** | 710.213 | 480.539 |

### Decode tok/s

| Workload | hipENGINE shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | 111.738 | 85.487 | **127.515** |
| 4K/128 | 113.231 | 87.375 | **120.163** |
| 32K/128 | 97.779 | 76.994 | **98.073** |
| 128K/128 | 62.014 | 57.341 | **64.478** |

### Peak GiB

| Workload | hipENGINE shisa Qwen3.6 packed PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | **18.123** | 21.125 | 20.844 |
| 4K/128 | **19.995** | 21.197 | 20.969 |
| 32K/128 | **20.267** | 21.738 | 21.533 |
| 128K/128 | **23.235** | 23.605 | 23.596 |

See [`benchmarks/README.md`](benchmarks/README.md) for full protocol details,
correctness status, source-lineage targets, and external comparison baselines.

## Hardware targets

| Backend | Hardware | Status |
| --- | --- | --- |
| `hip_gfx1100` | AMD Radeon Pro W7900 / RX 7900 XTX (RDNA3) | Primary, in active bring-up |
| `hip_gfx1151` | Strix Halo (RDNA3.5) | Planned peer backend |
| `cuda_sm86` | NVIDIA Ampere consumer (3090-class) | Planned peer backend |
| `cpu_reference` | Any CPU, numpy | Correctness oracle; CI without GPU |

Wave32 is the default for `hip_gfx1100` device code; wave64 is treated as an
isolated experiment with its own gates (see
[`docs/PLAN.md`](docs/PLAN.md#rdna3-wavefront-and-scheduling-caveat)).

## Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────┐
│  USER API                                                       │
│  hipengine.LLM.generate()           library API                 │
│  hipengine.server                   optional [server] extra     │
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
│  KERNELS (backend-keyed, 120 __global__ in the Qwen/PARO port)  │
│  kernels/hip_gfx1100/  attention / linear_attn / moe / quant    │
│                        wmma / norm / rotary / fused             │
│  kernels/hip_gfx1151/  (future)                                 │
│  kernels/cuda_sm86/    (future)                                 │
│  kernels/cpu_reference/ correctness oracle, no GPU required     │
└─────────────────────────────────────────────────────────────────┘
```

Full layer diagram, plugin axes, KV cache ABI, and roadmap are in
[`docs/PLAN.md`](docs/PLAN.md).

## Installation

```bash
# one-time: fetch Git LFS payloads, including the vendored AOTriton runtime/images
git lfs install
git lfs pull

# core runtime (torch-free)
pip install -e .

# with the OpenAI-compatible server
pip install -e ".[server]"

# with the optional dlpack torch bridge for user-boundary interop
pip install -e ".[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.11+. A working ROCm install with `libamdhip64.so` on the loader path
is required for any GPU run; CPU-reference correctness tests run without a GPU.

## Quickstart (Phase 0 — bring-up only)

The public API surface is stable:

```python
from hipengine import LLM, SamplingParams

llm = LLM("/path/to/model", backend="hip_gfx1100", quant="w4_paro")
outputs = llm.generate(
    ["Hello, hipENGINE."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
```

Today `LLM.generate()` only resolves to narrow Qwen3.5 / PARO bring-up paths
registered in `hipengine.generation`; unsupported `(model, backend, quant)`
combinations fail loudly rather than falling back to a generic torch path. See
[`docs/PLAN.md`](docs/PLAN.md) for the model / quant roadmap.

## OpenAI-compatible server

Install the optional server extra and run the FastAPI layer:

```bash
pip install -e ".[server]"
python -m hipengine.server \
  --model /path/to/model \
  --backend hip_gfx1100 \
  --quant w4_paro \
  --served-model-name qwen-paro
```

Supported v0.1 endpoints: `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions` (including one-chunk SSE for `stream=true`). See
[`docs/API.md`](docs/API.md) for request examples, bearer-token auth, and
current limitations.

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture, plugin axes, phase roadmap, LoC budgets |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark protocols, baselines, correctness gate, artifact format |
| [`docs/TESTING.md`](docs/TESTING.md) | RED/GREEN workflow, correctness oracles, fixture policy |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog, source-lineage drift workflow, JIT cache gotchas, build profiles |
| [`docs/ROOFLINE.md`](docs/ROOFLINE.md) | RDNA3 / W7900 performance model and decision tree |
| [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) | Implementation status and concrete milestones |
| [`docs/API.md`](docs/API.md) | OpenAI-compatible server usage and endpoint support |
| [`docs/PREFILL.md`](docs/PREFILL.md) | Native prefill implementation spec |
| [`docs/MTP.md`](docs/MTP.md) | Multi-token prediction plan |
| [`docs/DFLASH.md`](docs/DFLASH.md) | DFlash draft-model speculative decode plan |
| [`benchmarks/README.md`](benchmarks/README.md) | Current-fastest rollup and external comparison baselines |
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

hipENGINE is not a fork of any project; it is a brand new codebase with from-scratch
code and kernels. Of course it builds on the work of many others:

- [ROCm](https://github.com/ROCm/rocm) - of course this all sits on AMD's open-source
  compute stack, notably on [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - most of the original
  kernel tuning iteration loops used this as a host-layer. Some of the performance 
  limitations of the architecture motivated the hipENGINE rewrite, but we remain
  greatful and deeply appreciative of nano-vllm as a great research platform.
- [ParoQuant](https://github.com/z-lab/paroquant) - after reviewing the current SOTA on model
  quantization, we chose ParoQuant as the first target due to both its excellent accuracy
  *and* its efficiency (QTIP/[YAQA](https://github.com/Cornell-RelaxML/yaqa-quantization) is 
  very cool but proved challenging to implement performant RDNA3 kernels)
- [FastDMS](https://github.com/shisa-ai/FastDMS) - our KVCache ABI is shaped by the lessons 
   learned from building our DMS reference implementation.

Greetz: [hipfire](https://github.com/Kaden-Schutt/hipfire), [Lucebox](https://github.com/Luce-Org/lucebox-hub), [DS4](https://github.com/antirez/ds4), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and ofc the og [llama.cpp](https://github.com/ggml-org/llama.cpp)

See also: [Marlin](https://github.com/IST-DASLab/marlin), [kernel-anvil](https://github.com/apollosenvy/kernel-anvil), [wmma_ops](https://github.com/glovepost/wmma_ops), [tilelang](https://github.com/tile-ai/tilelang), [fsr4-rdna3-optimization](https://github.com/lhl/fsr4-rdna3-optimization), [ROCm examples](https://github.com/ROCm/rocm-examples)


## License

hipENGINE source code is licensed under **AGPL-3.0-or-later**. It is built and distributed 
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
