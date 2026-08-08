# hipEngine

hipEngine is a ROCm-native local inference engine for AMD RDNA GPUs. It pairs a
small Python host with custom HIP kernels for torch-free model loading,
generation, and OpenAI-compatible serving on supported hardware.

## Supported models

Support is model-, format-, backend-, and workload-specific. The table names the
checkpoints used by retained runtime paths; some are correctness- or
benchmark-qualified rather than full public-serving routes. It is not a claim
that arbitrary models in the same container format will run.

| Family | Tested checkpoints and formats | Current support boundary | Details |
| --- | --- | --- | --- |
| **Qwen3.5 dense** | [`ggml-org/Qwen3.5-0.8B-GGUF`](https://huggingface.co/ggml-org/Qwen3.5-0.8B-GGUF); native GGUF `Q4_K_M` plus correctness coverage for `Q8_0`, `Q4_1`, and `UD-Q4_K_XL` | Text-only public generation on gfx1100/gfx1151; the 0.8B path is correctness-qualified, not a promoted performance route | [`docs/GGUF.md`](docs/GGUF.md) |
| **Qwen3.5/3.6 MoE — PARO** | [`shisa-ai/Qwen3.6-35B-A3B-PARO-packed`](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed); W4 ParoQuant | Public generation and serving on gfx1100/gfx1151; native concurrency is qualified per backend | [`docs/PLAN.md`](docs/PLAN.md) · [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md) |
| **Qwen3.5/3.6 MoE — GGUF** | [`unsloth/Qwen3.5-35B-A3B-GGUF`](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) and [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF); `Q4_K_M`, `Q4_K_S`, and retained `UD-Q3_K_M` execution | Public generation and serving on gfx1100/gfx1151; bundled NextN/MTP is available only through its documented guarded routes | [`docs/GGUF.md`](docs/GGUF.md) · [`docs/MTP-gguf.md`](docs/MTP-gguf.md) |
| **Laguna S 2.1** | [`poolside/Laguna-S-2.1-GGUF`](https://huggingface.co/poolside/Laguna-S-2.1-GGUF) `Q4_K_M`; [`unsloth/Laguna-S-2.1-GGUF`](https://huggingface.co/unsloth/Laguna-S-2.1-GGUF) `UD-Q2_K_XL` | Public c1 generation/serving through 4K on gfx1151; measured resident W7900 path for `UD-Q2_K_XL`; matched DFlash is explicit-only | [`docs/LAGUNA.md`](docs/LAGUNA.md) · [`docs/LAGUNA-PARITY-STATUS.md`](docs/LAGUNA-PARITY-STATUS.md) |
| **Maple-Preview 20B-A1B** | [`deepgrove/maple-preview-2bit-mlx`](https://huggingface.co/deepgrove/maple-preview-2bit-mlx); native ternary 2-bit MLX checkpoint (5.31 GB) | Exact full-head public c1/c2/c4/c8 generation on gfx1151; correctness-qualified c1 `LLM.generate()` on `cuda_sm120a` with serial prompt admission (no CUDA throughput/batch claim); gfx11 public default context is 4K and the retained long-prompt exactness gate reaches 770 tokens | [`docs/MAPLE.md`](docs/MAPLE.md) · [`docs/MAPLE-PERF.md`](docs/MAPLE-PERF.md) |

Maple's deployment configuration declares 128,000 positions (the model is
marketed at 131,072), and Laguna's GGUF declares 256K. Those model limits are
not blanket hipEngine support claims: use the qualified boundaries above and in
the linked model documents.

## Performance highlights

Different rows use different models, quantization, and protocols. Compare rates
only within a row; these are concise retained highlights, not a cross-model
ranking.

<!-- BEGIN TOPLINE:README_HIGHLIGHTS -->
| Model / route | Hardware | Qualified workload | Prefill tok/s | Decode tok/s |
| --- | --- | --- | ---: | ---: |
| Qwen3.6-35B-A3B PARO W4 | Radeon Pro W7900 / gfx1100 | 512 prompt / 128 decode | **2917.732** | **115.599** |
| Qwen3.6-35B-A3B GGUF `Q4_K_M` | Radeon Pro W7900 / gfx1100 | 512 prompt / 128 decode | **2716.648** | **92.833** |
| Qwen3.6-35B-A3B GGUF `UD-Q4_K_M` | Radeon 8060S / gfx1151 | 512 prompt / 128 decode | **1369.489** | **54.330** |
| Laguna S 2.1 GGUF `Q4_K_M` | Radeon 8060S / gfx1151 | retained pp512 / p512+d128 c1 | **654.249** | **23.221** |
| Laguna S 2.1 GGUF `UD-Q2_K_XL` | Radeon Pro W7900 / gfx1100 | natural C4096 / direct M512 prefill | **440.893** | — |
| Maple-Preview ternary 2-bit | Radeon 8060S / gfx1151 | native p512 / natural+heldout exact c1 | **754.458** | **153.201** |

Where a row names separate prefill and decode workloads, its cells are the
latest retained results for those explicitly named protocols, not one combined
run.

Selected parallel and speculative results:

| Route | Hardware and scope | Aggregate decode | Relative result |
| --- | --- | ---: | ---: |
| Qwen3.6 GGUF physical c8 (PM4) | W7900 direct model step; c1 uses HIP graph, c2/c4/c8 use PM4 | **266.479 tok/s** | **2.712x** c1 |
| Qwen3.6 GGUF physical c8 | Radeon 8060S direct model step | **133.251 tok/s** | **2.647x** c1 |
| Maple public c8 | Radeon 8060S, 64 tokens/request including admission and reclaim | **214.788 tok/s** | **1.744x** public c1 |
| Qwen3.6 GGUF MTP `llama-compat` | W7900, explicit accuracy-traded route | **122.67 tok/s** | **1.2679x** own true AR |
| Qwen3.6 GGUF NativeSpecCycle N3 | Radeon 8060S, explicit accuracy-traded route | **80.10 tok/s** | **1.4282x** own true AR |

The current source-pinned W7900 direct c1/c2/c4/c8 packet reaches
**98.263/148.944/209.304/266.479 aggregate tok/s**. The matched real OpenAI SSE
c1/c8/c9/c13 packet reaches **72.169/158.542/137.001/129.507 aggregate tok/s**;
all retained trajectories/IDs are exact, with zero PM4 fallback and clean
ownership drain.
<!-- END TOPLINE:README_HIGHLIGHTS -->

Full model hashes, software stacks, commands, samples, correctness gates, memory
measurements, and historical results live in the canonical
[`benchmarks/README.md`](benchmarks/README.md) and compact artifacts under
[`benchmarks/results/`](benchmarks/results/).

## Status

**v0.3.0 alpha.** hipEngine is currently a single-GPU engine with active gfx1100
and gfx1151 backends.

- `backend="auto"` and `quant="auto"` select a registered path for recognized
  hardware and models; unsupported combinations fail rather than silently
  falling back to PyTorch.
- PARO, Qwen GGUF, Laguna, and Maple expose deterministic greedy generation and
  their qualified sampling, streaming, cancellation, and lifecycle behavior.
  Exact capabilities vary by model and are discoverable through the server.
- The OpenAI-compatible server supports completions, chat, token-level SSE,
  logprobs, tools, structured-output validation, Qwen thinking controls,
  readiness/capability discovery, and request diagnostics.
- Speculative routes remain explicit unless their own correctness and economics
  gates justify automatic use. In particular, `llama-compat` MTP is not
  serial-prefix-equivalent and Laguna DFlash remains slower than true AR on its
  canonical full suite.

See [`docs/API.md#current-limitations`](docs/API.md#current-limitations) and
[`docs/CONCURRENCY.md#current-truth`](docs/CONCURRENCY.md#current-truth) for
precise public API and serving boundaries.

## Core principles

- **HIP-first, not CUDA-ported.** Kernels directly target AMD hardware such as
  gfx1100/RDNA3 and gfx1151/RDNA3.5.
- **Torch-free runtime.** `import torch` is not on the generation hot path.
  Torch is an optional DLPack bridge behind the `hipengine[torch]` extra.
- **Four-axis plugin registry.** Kernels are keyed by
  `(backend, layer, quant, variant)`; models, quant schemes, and layers are
  plugins rather than dispatch special cases.
- **Fused and unfused paths coexist.** Every fused composite has a numerically
  equivalent primitive chain for fallback and correctness testing.
- **Evidence-backed performance.** Retained claims include the model, quant,
  workload, hardware, command, artifact, and correctness gate. See
  [`docs/BENCHMARK.md`](docs/BENCHMARK.md).

## Hardware targets

| Backend | Hardware | Status |
| --- | --- | --- |
| `cpu_reference` | Any CPU with NumPy | Correctness oracle and GPU-free CI |
| `hip_gfx1100` | Radeon Pro W7900 / RX 7900 XTX (RDNA3) | Active |
| `hip_gfx1151` | Ryzen AI MAX+ 395 / Radeon 8060S (RDNA3.5) | Active |
| `cuda_sm120a` | NVIDIA consumer Blackwell GPUs | Partial (some models) |

Nearby unqualified ROCm targets can force a HIP backend only after independent
correctness validation. gfx1151 defaults to one hardware queue to avoid a
retained low-power queue failure; operational overrides are documented in
[`docs/ENVS.md`](docs/ENVS.md).

## Architecture at a glance

```text
┌─────────────────────────────────────────────────────────────────┐
│  USER API                                                       │
│  hipengine.LLM.generate()           library API                 │
│  hipengine serve                    OpenAI-compatible server    │
├─────────────────────────────────────────────────────────────────┤
│  LOADING                                                        │
│  safetensors / GGUF / MLX metadata / tokenizer / chat template  │
├─────────────────────────────────────────────────────────────────┤
│  DISPATCH                                                       │
│  Scheduler / KV policy / prefix cache / fusion planner          │
│  Model / quant / layer plugins / engine loop / graph replay     │
├─────────────────────────────────────────────────────────────────┤
│  CORE                                                           │
│  Tensor / device / memory / stream / graph / BLAS / JIT cache   │
├─────────────────────────────────────────────────────────────────┤
│  KERNELS                                                        │
│  kernels/hip_gfx1100/      active RDNA3 implementation          │
│  kernels/hip_gfx1151/      active RDNA3.5 peer backend          │
│  kernels/cpu_reference/    correctness oracle                   │
│  kernels/cuda_sm120a/      partial NVIDIA Blackwell backend     │
└─────────────────────────────────────────────────────────────────┘
```

The full layer diagram, plugin mechanics, KV cache ABI, and roadmap are in
[`docs/PLAN.md`](docs/PLAN.md).

## Installation

```bash
# PyPI: runtime, JIT kernel sources, vendored AOTriton subset, and server
pip install hipengine

# Source checkout
git lfs install
git lfs pull
pip install -e .

# Optional DLPack bridge at the user boundary
pip install "hipengine[torch]"

# Development dependencies
pip install -e ".[dev]"
```

Python 3.10+ and a working ROCm installation with `libamdhip64.so` on the loader
path are required for GPU execution. CPU-reference tests run without a GPU. Use
the pinned stack instructions in [`docs/THEROCK.md`](docs/THEROCK.md) when
reproducing benchmark rows; each artifact records its exact software stack.

The installed command group includes:

```bash
hipengine --help
hipengine serve --help
hipengine bench list
```

## Quickstart

hipEngine does not download weights during model construction. Populate the
Hugging Face cache first:

```bash
hf download shisa-ai/Qwen3.6-35B-A3B-PARO-packed
```

Then use the same local repository ID:

```python
from hipengine import LLM, SamplingParams

llm = LLM("shisa-ai/Qwen3.6-35B-A3B-PARO-packed")
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
llm.close()
```

`LLM(model)` detects gfx1100 or gfx1151 and selects the registered model and
quantization route. Explicit `backend=` and `quant=` arguments are overrides.
Local filesystem paths are supported for GGUF and Maple checkpoints.

## OpenAI-compatible server

```bash
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --served-model-name qwen-paro
```

`--model` accepts a local path or a Hugging Face repository already present in
the local cache. Core endpoints are `GET /v1/models`, `POST /v1/completions`,
and `POST /v1/chat/completions`. See [`docs/API.md`](docs/API.md) for
authentication, request examples, streaming, capability discovery, diagnostics,
and model-specific limitations.

## Documentation

| File | Purpose |
| --- | --- |
| [`docs/PLAN.md`](docs/PLAN.md) | Architecture, plugin axes, and roadmap |
| [`docs/API.md`](docs/API.md) | Library and OpenAI-compatible server contracts |
| [`docs/GGUF.md`](docs/GGUF.md) | Qwen dense/MoE GGUF support and format boundaries |
| [`docs/LAGUNA.md`](docs/LAGUNA.md) | Laguna model contract, public boundary, and DFlash status |
| [`docs/MAPLE.md`](docs/MAPLE.md) | Maple model summary, runtime support, performance, and audit |
| [`docs/MOONSHINE.md`](docs/MOONSHINE.md) | Moonshine internal backend status and gfx1151 transfer campaign |
| [`docs/KERNELS.md`](docs/KERNELS.md) | Kernel catalog, lineage, JIT, and profiling workflow |
| [`docs/BENCHMARK.md`](docs/BENCHMARK.md) | Benchmark protocols and evidence requirements |
| [`docs/TESTING.md`](docs/TESTING.md) | Correctness oracles, fixtures, and validation tiers |
| [`docs/ENVS.md`](docs/ENVS.md) | Runtime and benchmark environment variables |
| [`benchmarks/README.md`](benchmarks/README.md) | Canonical benchmark scoreboard and full evidence index |
| [`WORKLOG.md`](WORKLOG.md) | Append-only engineering journal |

## Development

```bash
# Narrowest relevant tests first
pytest -q

# Before kernel ports
python3 scripts/check_lineage.py --kind kernel --diff stat

# Verify the compact root performance export
python3 scripts/sync_benchmark_readme.py --check
```

See [`AGENTS.md`](AGENTS.md) for the complete workflow, validation tiers, evidence
policy, and commit discipline.

## References and lineage

hipEngine is an independent codebase built on the work of many projects:

- [ROCm](https://github.com/ROCm/rocm) and
  [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip)
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)
- [ParoQuant](https://github.com/z-lab/paroquant)
- [FastDMS](https://github.com/shisa-ai/FastDMS)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Marlin](https://github.com/IST-DASLab/marlin),
  [kernel-anvil](https://github.com/apollosenvy/kernel-anvil),
  [wmma_ops](https://github.com/glovepost/wmma_ops), and
  [ROCm examples](https://github.com/ROCm/rocm-examples)

Thanks also to [ROCmFPX](https://github.com/charlie12345/ROCmFPX),
[hipfire](https://github.com/Kaden-Schutt/hipfire),
[Lucebox](https://github.com/Luce-Org/lucebox-hub),
[DS4](https://github.com/antirez/ds4), and
[ExLlamaV3](https://github.com/turboderp-org/exllamav3).

## License

hipEngine source code is licensed under **AGPL-3.0-or-later**. Model weights,
checkpoints, and external datasets remain under their own licenses.
