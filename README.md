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

**v0.2.2 alpha.** The runtime hot path is torch-free by construction, and the
first two 35B-class model-loading surfaces are now available on gfx1100:
[shisa-ai/Qwen3.6-35B-A3B-PARO-packed](https://huggingface.co/shisa-ai/Qwen3.6-35B-A3B-PARO-packed)
(19.07 GiB, 4.68 bpw) in packed
[ParoQuant](https://github.com/shisa-ai/paroquant) format, plus Qwen3.6 GGUF
`Q4_K_M` / `Q4_K_S` files through the resident GGUF path. Older benchmark
artifacts may still show the historical
`Qwen3.6-35B-A3B-PARO-full4096-e5-packed` name or local MTP-BF16 assembly path;
those rows use the same packed PARO architecture and remain the evidence for the
numbers below.

- INT8 KV cache support has been added for PARO. Qwen 3 MoE's full 256K context window can fit in <24GB tracked memory; see [Memory Usage](#memory-usage).
- The OpenAI-compatible server now has resident context/KV preallocation, startup warmup, max-prompt scratch probing, bounded chat-shaped startup smoke, `/ready` diagnostics, request context admission, and `max_tokens=auto` defaults for chat requests that omit an output cap.
- `LLM.stream()` and `stream=true` chat completions run token-level resident decode, with Qwen/DeepSeek-style `<think>...</think>` spans split into `reasoning_content` in both streaming and non-streaming responses.
- Qwen 3.6 [Q4_K_M](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-Q4_K_M.gguf) and [Q4_K_S](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF?show_file_info=Qwen3.6-35B-A3B-UD-Q4_K_S.gguf) GGUF support has landed (W7900 Q4_K_M sweep is in [Performance](#performance) alongside packed PARO and llama.cpp HIP/Vulkan Q4_K_M baselines). GGUF uses a substantial GGUF-specific runtime path with bulk prefill, graph decode, and on-load decode-repack into T16 tile layouts. Q4_K_S is the lower-memory secondary file; Q4_K_M is the active 1:1 llama.cpp comparison target and current 24 GiB BF16-KV support is mid-context unless a lower-memory KV/weight policy is enabled. GGUF also has a higher per-session load cost (~60 s vs ~38 s for PARO packed on the same W7900/TheRock stack) for the same decode-repack reason.
- Current gfx1100 and gfx1151 performance snapshots are summarized in [Performance](#performance) with hardware-separated tables and recent llama.cpp baselines.


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
`shisa-ai/Qwen3.6-35B-A3B-PARO-packed` on W7900/gfx1100 with q3072
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

While we are far from [gfx1100 roofline](https://github.com/shisa-ai/hipEngine/blob/main/docs/ROOFLINE.md), the current gfx1100 implementation is competitive with current local llama.cpp builds on the same W7900/GPU0 host. The W7900 table below combines the hipEngine PARO refresh from `2026-06-14`, the hipEngine GGUF Q4_K_M final current-tree rerun from `2026-06-21`, and the llama.cpp Q4_K_M HIP/Vulkan sweeps from `2026-06-16`. All hipEngine rows use the hermetic TheRock ROCm 7.13 wrapper from `scripts/run_w7900_readme_refresh.sh` (`HIP version: 7.13.26162-1140233ffe`) and one resident max-context session for 2 warmups + 5 measured in-session repetitions per shape. Direct non-hermetic W7900 shell runs can under-report GGUF prefill while leaving decode in-family, so retained rows must use the wrapper. llama.cpp rows use Q4_K_M GGUF split prefill/decode with f16 KV and one `llama-bench` repetition per phase while sampling W7900 whole-card VRAM. PARO uses the default prefill policy: 512-token prompts stay unchunked and prompts above 1K use `1024/1024/4096/1024/1024` chunks. The GGUF AR loader accepts current MTP-bearing GGUF files by ignoring trailing `blk.40.nextn.*` predictor tensors while keeping strict mapping for the 40 executable AR layers.

### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 2729.701 | 2109.6 | 2515.628 | **2822.688** |
| 1K/128 | **2906.950** | 2331.3 | 2430.947 | 2710.994 |
| 4K/128 | **2879.578** | 2332.8 | 2303.237 | 2581.819 |
| 32K/128 | **2079.424** | 1799.8 | 1685.089 | 1968.760 |
| 64K/128 | **1559.096** | 1398.1 | 1324.775 | 1411.790 |
| 128K/128 | 1053.919 | 971.1 | 918.128 | **1081.646** |

### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **115.227** | 106.5 | 79.568 | 106.235 |
| 1K/128 | 102.927 | 95.8 | 79.256 | **106.204** |
| 4K/128 | **105.253** | 97.1 | 78.688 | 102.556 |
| 32K/128 | **91.965** | 84.9 | 71.842 | 91.644 |
| 64K/128 | 77.666 | 72.4 | 66.453 | **83.326** |
| 128K/128 | 60.349 | 57.2 | 57.713 | **70.476** |

### Peak GiB (lower is better)

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **21.029** | 24.985 | 21.621 | 21.261 |
| 1K/128 | 21.241 | 24.985 | 21.633 | **21.220** |
| 4K/128 | 21.973 | 24.985 | 21.690 | **21.278** |
| 32K/128 | 22.082 | 24.985 | 22.232 | **21.855** |
| 64K/128 | **22.082** | 24.985 | 22.910 | 22.512 |
| 128K/128 | **22.124** | 24.985 | 24.103 | 23.824 |

W7900 row sources: PARO from the prior `2026-06-14` refresh, GGUF from the corrected hermetic `2026-06-21` final Q4_K_M current-tree rerun ([hipEngine GGUF](benchmarks/results/2026-06-21-w7900-gpu0-gguf-q4km-final-readme-sweep.json)), llama.cpp from the `2026-06-16` Q4_K_M sweep ([HIP](benchmarks/results/2026-06-16-w7900-gpu0-llamacpp-hip-q4km-f16kv-sweep.json), [Vulkan](benchmarks/results/2026-06-16-w7900-gpu0-llamacpp-vulkan-q4km-f16kv-sweep.json)). hipEngine columns are 5-run medians from one resident session allocated for the maximum requested context (`128K/128`). The GGUF peak memory includes a recent chunk-outer loop prefill rewrite that reduced scratch by ~1.3 GiB (from 26.3 to 25.0 GiB). The corrected final GGUF rerun kept stable IDs and stayed near the `2026-06-17` row; all GGUF columns use Q4_K_M, while PARO is a different quant/format (`w4_paro`).

### gfx1151 (AMD Ryzen AI MAX+ 395 / Radeon 8060S)

The gfx1151 backend is a native `--offload-arch=gfx1151` peer backend using the
same registry-keyed kernel surface. The Strix Halo rows below are local
single-repetition diagnostics (`performance_claim=false`) so they are separated
from the W7900 retained refresh above instead of interleaving hardware contexts.
They use `HIPENGINE_HIP_ARCH=gfx1151`, TheRock ROCm `7.13.60980-c76140fa27`, and
match the W7900 table's shape set where available. hipEngine GGUF uses
`UD-Q4_K_M` here because that is the gfx1151 file measured alongside the local
llama.cpp HIP/Vulkan `UD-Q4_K_M` baselines.

#### Prefill tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | 956.666 | 833.366 | 1016.696 | **1043.209** |
| 1K/128 | 1067.175 | 854.308 | **1069.681** | 1055.050 |
| 4K/128 | **1062.248** | 729.117 | 1021.186 | 1027.069 |
| 32K/128 | **822.255** | 619.570 | 742.869 | 809.619 |
| 64K/128 | 622.752 | 522.872 | 569.611 | **658.399** |
| 128K/128 | 425.727 | 384.011 | 384.959 | **473.651** |

#### Decode tok/s

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M | llama.cpp HIP Q4_K_M | llama.cpp Vulkan Q4_K_M |
| --- | ---: | ---: | ---: | ---: |
| 512/128 | **66.967** | 56.581 | 51.640 | 62.434 |
| 1K/128 | **61.768** | 52.832 | 51.446 | 61.572 |
| 4K/128 | **62.910** | 53.638 | 49.581 | 60.012 |
| 32K/128 | 50.368 | 44.383 | 43.628 | **50.911** |
| 64K/128 | 41.966 | 37.741 | 38.604 | **44.010** |
| 128K/128 | 30.286 | 28.043 | 31.598 | **34.714** |

#### hipEngine tracked allocator peak GiB

| Workload | hipEngine PARO | hipEngine GGUF Q4_K_M |
| --- | ---: | ---: |
| 512/128 | 20.924 | 26.264 |
| 1K/128 | 20.926 | 26.264 |
| 4K/128 | 20.937 | 26.264 |
| 32K/128 | 21.047 | 26.264 |
| 64K/128 | 21.047 | 26.264 |
| 128K/128 | 21.248 | 26.264 |

On Strix Halo, `rocm-smi` / sysfs expose only a 512 MiB VRAM aperture, so
cross-engine memory comparisons are omitted here. Row sources: [`gfx1151 summary`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-paro-packed-1run.json),
[`hipEngine GGUF`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-hipengine-gguf-ud-q4km-1run.json),
[`llama.cpp HIP`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-hip-ud-q4km-f16kv.json), and
[`llama.cpp Vulkan`](benchmarks/results/2026-06-15-gfx1151-readme-udq4km-20260615-040438-llamacpp-vulkan-ud-q4km-f16kv.json).

See [`benchmarks/README.md`](benchmarks/README.md) for full protocol details,
correctness status, source-lineage targets, and external comparison baselines.

## Speculative decode (DFlash / MTP)

Speculative decode is active but split by model class. Dense 27B DFlash has a
retained exact speedup; 35B-A3B MTP now has its first exact break-even row, with
more policy/kernel margin work still active because the MoE target AR path is
cheap.

### gfx1100 (W7900)

| Path | Model / workload | W7900 result | Status |
| --- | --- | ---: | --- |
| DFlash B=4 online-gated | Qwen3.6-27B-PARO dense target + z-lab Qwen3.6-27B-DFlash drafter, 9-prompt D64 | **1.231x AR** (`40.10` vs `32.57 tok/s`) | Exact `9/9`, deployable retained row; artifact: [`2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json`](benchmarks/results/2026-06-11-hipengine-dflash-27b-dense-hardening-rerun.json). |
| MTP B=3 persistent chain, locked sprint baseline | Qwen3.6-35B-A3B-PARO packed trunk + MTP-BF16 sidecar, graph-auto verifier, draft vocab cap 32768 | **0.758x AR** (`83.4` vs `~110 tok/s`), `27.8 ms/cycle` | Exact but below AR; retained as the sprint baseline. Artifacts: [`baseline`](benchmarks/results/2026-06-11-hipengine-mtp-b3-locked-baseline.json) / [`rocprof`](benchmarks/results/2026-06-11-hipengine-mtp-b3-locked-rocprof.json). |
| MTP B=1 persistent chain, current best | Qwen3.6-35B-A3B-PARO packed trunk + MTP-BF16 sidecar, `decode_batched`, graph off, draft vocab cap 65536 default | **1.023x prompt-mean / 1.014x total-time AR**, `14.134 ms/cycle` | Exact `9/9`, 3-run retained break-even row. B=3 remains higher-density but just short (`0.968x` same-session); full vocab was exact but no-held (`0.880x`). See [`docs/MTP.md`](docs/MTP.md) and [`B=1 artifact`](benchmarks/results/2026-06-13-hipengine-mtp-b1-current-default-3run-retained.json). |

### gfx1151 (Radeon 8060S diagnostic)

The gfx1151 MTP rows are diagnostics, not retained wins. The hipEngine row used
the public PARO packed trunk plus copied BF16 MTP sidecar with exact fallback
flags to pass the D32 prompt suite; it remained below AR speed. The llama.cpp
rows use the MTP-bearing `UD-Q4_K_M` GGUF and are cross-engine comparison
baselines.

| Path | Model / workload | gfx1151 result | Status |
| --- | --- | ---: | --- |
| hipEngine MTP B=1 | Qwen3.6-35B-A3B-PARO packed trunk + MTP-BF16 sidecar, D32, 9 prompts | `0.912x` prompt-mean / `0.904x` total-time AR (`59.56` vs `65.37 tok/s`) | Exact `9/9` with exact fallbacks; diagnostic, no retained performance claim. |
| llama.cpp HIP MTP B=4 | Qwen3.6-35B-A3B `UD-Q4_K_M` MTP GGUF, D32, 9 prompts | mean `91.11 tok/s`, wall `52.47 tok/s`, `1.79x` mean vs base | Cross-engine diagnostic; accept rate `91.5%`. |
| llama.cpp Vulkan MTP B=4 | Qwen3.6-35B-A3B `UD-Q4_K_M` MTP GGUF, D32, 9 prompts | mean `108.96 tok/s`, wall `57.69 tok/s`, `1.73x` mean vs base | Cross-engine diagnostic; accept rate `92.3%`. |

Source artifact: [`gfx1151 MTP comparison`](benchmarks/results/2026-06-15-gfx1151-mtp-compare-20260615-060801-summary.json).

## Concurrency (batched decode)

hipEngine has a native `c>1` decode path: a scheduler-owned compact prefill plus
a device-resident batched decode step (token feedback through `batch_lm_out_index`,
device batched LM-head argmax, on-stream position advance) that can be captured
and replayed as a single HIP graph. See [`docs/CONCURRENCY.md`](docs/CONCURRENCY.md)
for the design and the C3.0a/b/c decode-throughput work.

The snapshots below keep gfx1100 and gfx1151 separate because the model files,
ROCm stacks, and comparison backends differ. *Aggregate* is total tok/s across
the batch; *per-sequence* is tok/s seen by one request. See
[`docs/VLLM_RDNA3.md`](docs/VLLM_RDNA3.md) for vLLM RDNA3 setup notes.

### gfx1100 / W7900 decode tok/s vs concurrency (Qwen3.6 35B-A3B, 512/128)

This current-code diagnostic uses median-of-3 runs. hipEngine uses PARO W4A16
with BF16 KV. llama.cpp uses the available Qwen3.6 GGUF `UD-Q4_K_S` with Vulkan
RADV, f16 KV, exact token-id prompts, and `llama-server -np c -c 1024*c`, so it
is a useful server-side comparison but not same-quant. vLLM uses a local
`v0.22.1rc1.dev499+g470229c37.d20260613` source build with
`palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4`, no MTP, exact token-id prompts, and the
OpenAI `/v1/completions` API. vLLM values are wall-throughput because the OpenAI
response path does not expose llama.cpp-style pure decode timings.

| Concurrency `c` | hipEngine aggregate | hipEngine per-seq | llama.cpp Vulkan aggregate | llama.cpp per-seq | vLLM OpenAI aggregate | vLLM per-seq |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **115.36** | **115.36** | 105.76 | 105.76 | 20.04 | 20.04 |
| 2 | 115.35 | 57.67 | **157.38** | **78.69** | 38.42 | 19.21 |
| 4 | **160.74** | **40.19** | 75.29 | 18.82 | 73.28 | 18.32 |
| 8 | **190.15** | **23.77** | 25.15 | 3.14 | 116.56 | 14.57 |

hipEngine aggregate throughput scales from c1 to c8 by **1.65x**. The `c=2`
aggregate is effectively flat versus `c=1`; this is the known small-context
`c>1` dispatch-bound regime. llama.cpp wins this protocol at c=2, then falls off
at c4/c8 with this server/Vulkan setup. vLLM now runs via the local source build;
this no-MTP OpenAI-wall measurement is slower than hipEngine, but it reaches
116.56 aggregate tok/s at c8 and its Prometheus post-TTFT aggregate estimates are
20.62/40.15/77.22/126.95 tok/s for c1/c2/c4/c8.

Source artifacts:
[`hipEngine W7900`](benchmarks/results/2026-06-14-w7900-gpu0-readme-refresh-20260614-141414-hipengine-concurrency-w7900/summary.json),
[`llama.cpp Vulkan W7900`](benchmarks/results/2026-06-14-w7900-gpu0-readme-refresh-20260614-141414-llamacpp-vulkan-concurrency-w7900/summary.json),
[`vLLM local build W7900`](benchmarks/results/2026-06-14-w7900-gpu0-readme-refresh-20260614-141414-vllm-localbuild-gptq-int4-concurrency-c1-c8-w7900.json),
[`full W7900 refresh summary`](benchmarks/results/2026-06-14-w7900-gpu0-readme-refresh-20260614-141414-summary.json),
and [`vLLM RDNA3 notes`](docs/VLLM_RDNA3.md).

### gfx1151 / Radeon 8060S decode tok/s vs concurrency (Qwen3.6 35B-A3B, 512/128)

This Strix Halo diagnostic uses median-of-3 runs and is not a retained
performance claim (`performance_claim=false`, `retained_ready=false`). hipEngine
PARO completed c=1/2/4/8 with primitive c>N correctness passing for c=2/4/8 and
generated-token equality passing for every c>N repetition. llama.cpp Vulkan uses
`UD-Q4_K_S`, f16 KV, and a rebuilt `llama-server` from the standard
`/home/lhl/llama.cpp/llama.cpp-vulkan/build` tree. vLLM GPTQ Int4 did not
produce OpenAI rows on gfx1151: non-text-only startup hit a 256 GiB ViT SDPA OOM,
and text-only attempts loaded weights but never bound port 8008.

| Concurrency `c` | hipEngine aggregate | hipEngine per-seq | llama.cpp Vulkan aggregate | llama.cpp Vulkan per-seq | vLLM OpenAI |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 66.62 | 66.62 | 62.16 | 62.16 | blocked: server unhealthy |
| 2 | 69.54 | 34.77 | 94.12 | 47.06 | blocked |
| 4 | 88.39 | 22.10 | 119.51 | 29.88 | blocked |
| 8 | 100.68 | 12.59 | 119.94 | 14.99 | blocked |

Source artifacts: [`gfx1151 summary`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-summary.json),
[`hipEngine PARO`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-hipengine-paro/summary.json),
[`llama.cpp Vulkan`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-213804-llamacpp-vulkan/summary.json), and
[`vLLM blocked`](benchmarks/results/2026-06-15-gfx1151-readme-concurrency-20260615-122207-vllm-gptq-int4-blocked.json).

A current-code RX 7900 XTX rerun reached c1/c2/c4 but c8 now blocks with HIP OOM;
see [`XTX partial`](benchmarks/results/2026-06-13-hipengine-qwen35-concurrency-decode-latest-xtx-blocked-c8.json).
Replicate the W7900 hipEngine, llama.cpp Vulkan, and vLLM concurrency rows with:

```bash
scripts/run_w7900_readme_refresh.sh concurrency
scripts/run_w7900_readme_refresh.sh vllm
```

The exact expanded commands, device selectors, model paths, and vLLM server flags
are recorded in [`benchmarks/README.md`](benchmarks/README.md#readme-sweep-test-procedure). `c=1` is measured with `scripts/qwen35_paro_bench.py --graph-replay-decode` (single-sequence generate path); `c>=2` with `scripts/qwen35_batch_retained_bench.py` (native batched path). The sweep driver wires both up; see its module docstring for the exact per-`c` sub-commands.

## GGUF Support

As of v0.2.0, hipEngine includes resident Qwen3.6 GGUF support for `Q4_K_M` and
`Q4_K_S` model files (with more formats planned). This is a major runtime path,
not just a loader shim: GGUF has its own quant readers, bulk-prefill path,
decode-repacked T16 layouts, and fast-path controls.

Current caveats:

- PARO models take ~24s to load on the W7900 test host; GGUF currently takes
  about 60s because decode-repack happens on load. On-disk caching could reduce
  startup time later, but would require additional storage for repacked layouts.
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
  --model shisa-ai/Qwen3.6-35B-A3B-PARO-packed \
  --quant w4_paro \
  --served-model-name qwen-paro
```

`--model` accepts either a local filesystem path or a Hugging Face model ID
already present in the local HF cache; hipEngine resolves IDs locally and does
not download weights during startup.

Supported endpoints: `GET /v1/models`, `POST /v1/completions`, and
`POST /v1/chat/completions` with token-level SSE streaming, OpenAI-style tool
calling, and Qwen no-think controls. Chat responses separate `<think>` reasoning
into `reasoning_content` (matching the OpenAI reasoning-content convention). The
server eagerly warms the model on startup by default, logs startup load/warmup
timing, caps omitted chat `max_tokens` with `--chat-default-max-tokens` (default
4096), and has an explicit `--debug` mode for full request/response payload
logging. See [`docs/API.md`](docs/API.md) for request examples, bearer-token
auth, diagnostics, and current limitations.

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
| [`docs/SAMPLING.md`](docs/SAMPLING.md) | Normal sampling parameter support plan |
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
