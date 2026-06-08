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

**v0.2.1 alpha.** The runtime hot path is torch-free by construction, and the
first two 35B-class model-loading surfaces are now available on gfx1100:
[shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format, plus Qwen3.6 GGUF
`Q4_K_M` / `Q4_K_S` files through the new resident GGUF path.

- INT8 KV cache support has been added for PARO. Qwen 3 MoE's full 256K context window can fit in <24GB tracked memory; see [Memory Usage](#memory-usage).
- Qwen 3.6 [Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf) and [Q4_K_S](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-IQ4_XS.gguf) GGUF support has landed (W7900 Q4_K_S sweep is in [Performance](#performance) alongside packed PARO and llama.cpp Q4_K_M HIP/Vulkan baselines). GGUF uses a substantial GGUF-specific runtime path with bulk prefill, graph decode, and on-load decode-repack into T16 tile layouts. Q4_K_S is recommended on 24 GiB cards because Q4_K_M is bigger; on the 48 GiB W7900 Q4_K_S fits all the way to 128K context, while on 24 GiB cards expect roughly 64K. GGUF also has a higher per-session load cost (~60 s vs ~24 s for PARO packed on the same hardware) for the same decode-repack reason.
- Current gfx1100 performance snapshots are summarized in [Performance](#performance) and compared against recent llama.cpp Q4_K_M baselines.


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

Wave32 is the default for `hip_gfx1100` device code; wave64 is treated as an
isolated experiment with its own gates (see
[`docs/PLAN.md`](docs/PLAN.md#rdna3-wavefront-and-scheduling-caveat)).

## Memory Usage

With BF16 KV cache, hipEngine running the packed Qwen 3.6 PARO model fits a
128K context window in a 24GB-class memory budget. The INT8 KV cache option
(with FP16 per-token/per-head scales) uses the
`--kv-storage int8_per_token_head` flag and lets the **full 256K context** fit
under 24 GiB tracked allocator peak.

The numbers below are for
`shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed` on W7900/gfx1100 with q3072
full-attention prefill chunks:

| Model                | Context | KV cache | Sampled peak | Allocator peak | Retained KV | Prefill      | Decode     |
| -------------------- | ------: | -------- | -----------: | -------------: | ----------: | -----------: | ---------: |
| Qwen3.6 35B-A3B PARO |    128K | BF16     |    21.04 GiB |      21.88 GiB |    2.69 GiB | 1091.9 tok/s | 62.2 tok/s |
| Qwen3.6 35B-A3B PARO |    128K | INT8     |    19.80 GiB |      20.89 GiB |    1.36 GiB | 1076.5 tok/s | 60.0 tok/s |
| Qwen3.6 35B-A3B PARO |    256K | INT8     |    21.96 GiB |      23.71 GiB |    2.71 GiB |  670.2 tok/s | 40.3 tok/s |

Regardless of the difference in PARO weight storage (legacy or packed),
loaded-weight memory is about the same — approximately 16.4 GiB in VRAM.

The INT8 KV correctness gate is currently the deterministic Qwen3.5 PARO
fixture `fixtures/qwen35_paro/parent_512_32_seed1234.json` (512-token prompt,
32 greedy decode tokens): `max_kl=0.015328`, `mean_kl=0.001639`, top-1 agreement
100%, and generated IDs match BF16 KV exactly. Layer attention probes at context
64 and 520 also had top-1 agreement 100% with max quantized-vs-BF16 KL
`2.34e-7`. This is a fixture/regression gate, not a long-rollout perplexity
study, so long context generations may have unmeasured compounding errors.

The same 128K/128 Qwen3.5 BF16-vs-INT8 run measured -0.99% prefill tok/s and
-3.20% decode tok/s for INT8 KV, so speed loss is also very small.

See
[`benchmarks/results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json`](benchmarks/results/2026-05-19-hipengine-qwen36-packed-int8-kv-readme-memory-diagnostic.json),
[`benchmarks/README.md`](benchmarks/README.md#blocked--diagnostic-benchmark-attempts),
and [`docs/KVCACHE.md`](docs/KVCACHE.md) for commands, artifacts, and the full
no-shadow memory audit.

### llama.cpp

When run with `q8_0` kvcache, llama.cpp can also fit in 24GB:

```bash
--flash-attn on -ctk q8_0 -ctv q8_0 -c 262144 -b 128 -ub 128
```

Results:

| Model | llama.cpp model buffer | KV cache | Compute buffer | rocm-smi VRAM used | Free VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K_M | 20583 MiB | 2720 MiB | 203 MiB | 24017 MiB / 23.45 GiB | ~543 MiB |
| Q4_K_S | 19399 MiB | 2720 MiB | 203 MiB | 22832 MiB / 22.30 GiB | ~1728 MiB |

With `-ub 512`:

| Model | Compute buffer | rocm-smi VRAM used | Free VRAM |
| --- | ---: | ---: | ---: |
| Q4_K_M | 812 MiB | 24540 MiB | ~20 MiB |
| Q4_K_S | 812 MiB | 23443 MiB | ~1117 MiB |

- Note Q4_K_M is incredibly tight with only 20 MiB of headroom and you may either need to resize down or set `-b 512 -ub 128`.
- Q4_K_S does not need small `-b`/`-ub`; `-ub 512` fits fine, and can even increase to `-b 2048` (but `-ub` is the more important VRAM knob that controls the physical microbatch / compute buffer size for llama.cpp).

## Performance

### gfx1100 (Radeon RX 7900 XTX / Radeon Pro W7900)

While we are far from [gfx1100 roofline](https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md), the current gfx1100 implementation does well compared to Q4_K_M quants of recent llama.cpp builds (`b9042`) on the same model family. The latest W7900 hipEngine rows use TheRock ROCm 7.13 and load each resident model once for 1 warmup + 5 measured in-session repetitions per shape. PARO uses the default prefill policy: 512-token prompts stay unchunked and prompts above 1K use `1024/1024/4096/1024/1024` chunks. The `hipEngine GGUF Q4_K_S` column uses the same chunked-prefill policy plus the WMMA prefill + GEMV decode fast paths and the persistent on-load decode-repack into T16 tile layouts.

### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_S | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **2718.497** | 2258.847 | 2436.049 | 1816.927 |
| 4K/128 | **2838.773** | 2576.673 | 2176.905 | 1705.093 |
| 32K/128 | **2074.699** | 1893.967 | 1496.409 | 1128.554 |
| 128K/128 | **1055.454** | 998.143 | 710.213 | 480.539 |

### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_S | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 103.460 | 109.152 | 85.487 | **127.515** |
| 4K/128 | 101.964 | 100.048 | 87.375 | **120.163** |
| 32K/128 | 90.438 | 86.774 | 76.994 | **98.073** |
| 128K/128 | 59.598 | 57.954 | 57.341 | **64.478** |

### Peak GiB

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_S | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 20.962 | 25.108 | 21.125 | **20.844** |
| 4K/128 | 21.906 | 25.108 | 21.197 | **20.969** |
| 32K/128 | 22.016 | 25.108 | 21.738 | **21.533** |
| 128K/128 | **22.122** | 25.108 | 23.605 | 23.596 |

hipEngine W7900 row source: [`benchmarks/results/2026-05-25-w7900-hipengine-readme-persistent-5run-diagnostic.json`](benchmarks/results/2026-05-25-w7900-hipengine-readme-persistent-5run-diagnostic.json). Both hipEngine columns are 5-run medians from one resident session allocated for the maximum requested context (`128K/128`), so the peak-memory column is a max-context persistent-session high-water mark rather than each shape's minimum allocation. Existing W7900 llama.cpp HIP/Vulkan Q4_K_M rows are reused unchanged. The hipEngine GGUF Q4_K_S column is compared against the existing llama.cpp Q4_K_M baselines because that is the lineage of measured baselines we have on this host; cross-quant comparisons should be read as approximate.

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

The gfx1151 backend is a native `--offload-arch=gfx1151` peer backend using the same registry-keyed kernel surface. The Strix Halo snapshot below uses 256-row prefill chunks, which removed the 4K prefill gap without hurting long-context decode.

### Prefill tok/s

| Workload | hipEngine PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | 983.206 | **1058.738** | 638.008 |
| 4K/128 | **1029.402** | 1004.220 | 595.400 |
| 32K/128 | **792.296** | 735.534 | 407.984 |
| 128K/128 | **413.489** | 376.070 | 181.453 |

### Decode tok/s

| Workload | hipEngine PARO | llama.cpp HIP | llama.cpp Vulkan |
| --- | ---: | ---: | ---: |
| 512/128 | **62.060** | 50.537 | 57.615 |
| 4K/128 | **63.605** | 49.379 | 55.027 |
| 32K/128 | **50.629** | 43.435 | 44.576 |
| 128K/128 | 30.245 | **31.286** | 26.935 |

On Strix Halo, `rocm-smi` / sysfs expose only a 512 MiB VRAM aperture, so cross-engine memory comparisons are omitted here. The hipEngine allocator high-water mark for the chunk256 sweep was 17.997 GiB (512/128), 18.097 GiB (4K/128), 18.909 GiB (32K/128), and 21.877 GiB (128K/128).

See [`benchmarks/README.md`](benchmarks/README.md) for full protocol details,
correctness status, source-lineage targets, and external comparison baselines.

## Concurrency (batched decode)

hipEngine has a native `c>1` decode path: a scheduler-owned compact prefill plus
a device-resident batched decode step (token feedback through `batch_lm_out_index`,
device batched LM-head argmax, on-stream position advance) that can be captured
and replayed as a single HIP graph. See [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md)
for the design and the C3.0a/b/c decode-throughput work.

The table below is an **initial** snapshot of PARO decode throughput as the
number of concurrent sequences `c` grows on a fixed 512-prompt / 128-decode
shape (gfx1100 / RX 7900 XTX, BF16 KV, median of 3 runs). *Aggregate* is total
tok/s across the batch; *per-sequence* is tok/s seen by one request. llama.cpp
and vLLM concurrency baselines are not yet measured on this host and will be
added later.

### Decode tok/s vs concurrency (Qwen3.6 PARO, 512/128, gfx1100)

| Concurrency `c` | Aggregate decode tok/s | Per-sequence decode tok/s | llama.cpp | vLLM |
| --- | ---: | ---: | ---: | ---: |
| 1 | 133.84 | 133.84 | _TBD_ | _TBD_ |
| 2 | 131.09 | 65.54 | _TBD_ | _TBD_ |
| 4 | 181.56 | 45.39 | _TBD_ | _TBD_ |
| 8 | **225.90** | 28.24 | _TBD_ | _TBD_ |

Aggregate throughput scales with concurrency (c1 → c8 is **1.69×**) while
per-sequence latency falls as expected. The `c=2` aggregate dipping slightly
below `c=1` is a known artifact of the small-context `c>1` regime being
dispatch-bound: at `c=2` the batched step does not yet amortize per-step
dispatch over enough rows. Correctness is gated by `native_batch_vs_independent_c1`
generated-token equality (0 mismatches at c2/c4/c8) plus the per-kernel
CPU-reference gates — these are throughput measurements, not retained
performance claims.

Source artifact:
[`benchmarks/results/2026-06-08-hipengine-qwen35-concurrency-decode/summary.json`](benchmarks/results/2026-06-08-hipengine-qwen35-concurrency-decode/summary.json).
Replicate with:

```bash
HIP_VISIBLE_DEVICES=1 python3 scripts/qwen35_concurrency_decode_sweep.py \
    --model /models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16 \
    --fixture <8x512-prompt-ids>.json \
    --compiler-version-file <hipcc-version>.txt \
    --prompt-length 512 --decode-tokens 128 --warmup-decode-tokens 8 \
    --concurrencies 1,2,4,8 --reps 3 \
    --json benchmarks/results/2026-06-08-hipengine-qwen35-concurrency-decode/summary.json
```

`c=1` is measured with `scripts/qwen35_paro_bench.py --graph-replay-decode`
(single-sequence generate path); `c>=2` with `scripts/qwen35_batch_retained_bench.py`
(native batched path). The sweep driver wires both up; see its module docstring
for the exact per-`c` sub-commands.

## GGUF Support

As of v0.2.0, hipEngine includes resident Qwen3.6 GGUF support for `Q4_K_M` and
`Q4_K_S` model files (with more formats planned). This is a major runtime path,
not just a loader shim: GGUF has its own quant readers, bulk-prefill path,
decode-repacked T16 layouts, and fast-path controls.

Current caveats:

- PARO models take ~24s to load on the W7900 test host; GGUF currently takes
  about 60s because decode-repack happens on load. On-disk caching could reduce
  startup time later, but would require additional storage for repacked layouts.
- GGUF has higher resident memory than packed PARO. In the current W7900 README
  sweep, the max-context Q4_K_S session peaks at ~25.1 GiB tracked, so 128K is
  W7900/48 GiB territory; on 24 GiB cards, expect roughly 64K context with
  Q4_K_S.
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
│  KERNELS (backend-keyed, 120 __global__ in the Qwen/PARO port)  │
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
# one-time: fetch Git LFS payloads, including the vendored AOTriton runtime/images
git lfs install
git lfs pull

# runtime + OpenAI-compatible server (torch-free hot path)
pip install -e .

# with the optional dlpack torch bridge for user-boundary interop
pip install -e ".[torch]"

# dev / test
pip install -e ".[dev]"
```

Python 3.11+. A working ROCm install with `libamdhip64.so` on the loader path
is required for any GPU run; CPU-reference correctness tests run without a GPU.

The installed app exposes a small command group:

```bash
hipengine --help
hipengine serve --help
hipengine bench list
```

## Quickstart (Phase 0 — bring-up only)

The public API surface is stable:

```python
from hipengine import LLM, SamplingParams

llm = LLM("/path/to/model", quant="w4_paro")  # backend="auto" by default
outputs = llm.generate(
    ["Hello, hipEngine."],
    SamplingParams(max_tokens=64, temperature=0.0),
)
print(outputs[0])
```

Today `LLM.generate()` only resolves to narrow Qwen3.5 / PARO bring-up paths
registered in `hipengine.generation`; unsupported `(model, backend, quant)`
combinations fail loudly rather than falling back to a generic torch path. See
[`docs/PLAN.md`](docs/PLAN.md) for the model / quant roadmap.

## OpenAI-compatible server

The OpenAI-compatible FastAPI layer is installed by default:

```bash
pip install hipengine
hipengine serve \
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-full4096-e5-packed \
  --quant w4_paro \
  --served-model-name qwen-paro
```

`--model` accepts either a local filesystem path or a Hugging Face model ID
already present in the local HF cache; hipEngine resolves IDs locally and does
not download weights during startup.

Supported endpoints: `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions` with token-level SSE streaming. Chat responses
separate `<think>` reasoning into `reasoning_content` (matching the OpenAI
reasoning-content convention). The server eagerly warms the model on startup
by default so the first request does not pay load/compile cost. See
[`docs/API.md`](docs/API.md) for request examples, bearer-token auth, and
current limitations.

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

hipEngine is not a fork of any project; it is a brand new codebase with from-scratch
code and kernels. Of course it builds on the work of many others:

- [ROCm](https://github.com/ROCm/rocm) - of course this all sits on AMD's open-source
  compute stack, notably on [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
- [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm) - most of the original
  kernel tuning iteration loops used this as a host-layer. Some of the performance 
  limitations of the architecture motivated the hipEngine rewrite, but we remain
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

hipEngine source code is licensed under **AGPL-3.0-or-later**. It is built and distributed
for anyone who has an AMD card that hasn't been living up to its compute potential.

Model weights, checkpoints, and external datasets remain under their own licenses.
