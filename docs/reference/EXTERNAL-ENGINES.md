---
status: current
owns: The external inference-engine landscape, split into AMD-focused, NVIDIA-focused, Apple-silicon, and cross-platform engines — repositories, target hardware, distribution model, license, and the strengths that make each one worth knowing about.
---
# External inference engines

A reference list of other local inference engines: what each project is, what
hardware it targets, how you get it, and what it is genuinely good at. It is a
landscape page, not a scoreboard. Listing an engine here says nothing about its
relationship to hipEngine — a few are comparison targets, and most are simply
worth knowing about.

Entries are grouped by what they are built for. **AMD-focused** engines target
RDNA hardware — usually one specific APU or GPU — and tune for it.
**NVIDIA-focused** engines do the same for consumer Blackwell parts.
**Apple-silicon** engines target unified-memory Macs, where the device-specific
choices are a fixed memory budget and a Metal-only kernel path.
**Cross-platform** engines target many vendors and treat any single device as
one configuration among many. The first three groups are where device-specific
technique is being worked out; this group is what that technique has to beat on
the same device.

Grouping follows the hardware an engine is tuned for, not every vendor it can
build for. Several entries in the first two groups also compile for other
backends, and where that changes how a result should be read, the entry says
so.

Every throughput number a project publishes about itself is author-reported.
Treat those tables as leads, not evidence, until the run is reproduced on this
host under [CROSS-ENGINE-PROTOCOL.md](../CROSS-ENGINE-PROTOCOL.md). The
confound checklist that has caught every inflated external row reviewed so far
lives in
[QWEN4EXP-EXTERNAL-FORKS-REVIEW.md](../archive/QWEN4EXP-EXTERNAL-FORKS-REVIEW.md#reading-any-external-throughput-row-protocol-confounds).

## At a glance

| Engine | Repository | Hardware | Model format | How you get it | License | Best at |
| --- | --- | --- | --- | --- | --- | --- |
| strix-llama.cpp | [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) | Strix Halo only (`gfx1151`) | GGUF | Source build | MIT | Deepest public llama.cpp tuning for one APU, without giving up llama.cpp's surface. |
| strixite | [shawnshekari/strixite](https://github.com/shawnshekari/strixite) | Strix Halo (`gfx1151`), Linux, 128 GB | Own `.strixw` weights and n-gram table | Source build or container image | AGPL-3.0 | A single-model HIP engine with long-context serving and native multi-token prediction. |
| Gufo | [gufo-org/gufo](https://github.com/gufo-org/gufo) · [gufo.blog](https://gufo.blog/) | Strix Halo only (`gfx1151`) | GGUF | Podman image, Nix flake, or CMake source build | MIT | Per-model kernel specialization, with quality reports to match. |
| halogen | [peonist-ai/halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server) · [docs](https://peonist-ai.github.io/halogen-flash-server/) | Strix Halo only (`gfx1151`) | Own checkpoint, or a llama.cpp GGUF of the same model | Container image only | Proprietary EULA | One model family tuned to the last kernel, with agent-session serving built in. |
| hipfire | [warpfront/hipfire](https://github.com/warpfront/hipfire) · [hipfire.dev](https://hipfire.dev/) | AMD RDNA first (`gfx1100`/`gfx1151`/`gfx1201`), CDNA supported | Own MQ/MFP4 formats | Source build, container image, prebuilt kernels | Apache-2.0 | A torch-free engine with its own transport layer (Redline), shipped end to end. |
| radiance | [StillDeadcode/radiance](https://codeberg.org/StillDeadcode/radiance) | AMD; bundled kernels for RDNA4 (`gfx1200`/`gfx1201`) | `.rad` containers | Source build, Docker Compose | Apache-2.0 | Text, image, video, speech, and music behind one server, with a real plugin boundary. |
| Brilliance | [Dad_Vibes/Brilliance](https://codeberg.org/Dad_Vibes/Brilliance) | Two RDNA4 R9700 cards, Windows 11 | `.rad` containers | PowerShell setup and build scripts | Apache-2.0 | radiance on native Windows, without WSL or Docker. |
| vllm-radiance | [magiccodingman/vllm-radiance](https://github.com/magiccodingman/vllm-radiance) | RDNA4 R9700, qualified on two | Native FP8 or Quark MXFP4 checkpoints | Container image, Docker Compose | Not stated; vLLM and radiance are Apache-2.0 | vLLM's serving with RDNA4 kernels underneath. |
| paiton-vllm-plugin | [Eliovp-BV/paiton-vllm-plugin](https://github.com/Eliovp-BV/paiton-vllm-plugin) | RDNA4 R9700, one or two | The checkpoint named in each model guide | Docker launchers per model, pip plugin | Apache-2.0 | Per-model guides and containers that carry their own runtime. |
| DS41Engine | [vincalkr/DS41Engine](https://github.com/vincalkr/DS41Engine) | One or two RX 7900 XTX (`gfx1100`), Linux | DeepSeek-V4.1-Flash Q2 GGUF plus Engram tables | Source build, fetch script | MIT | Exact routing: every selected expert is computed, from VRAM, RAM, or NVMe. |
| WHIRL | [tsaipifong/whirl-llm](https://github.com/tsaipifong/whirl-llm) | Radeon 8060S (`gfx1151`) or R9700 (`gfx1201`), Windows 11 | Its own MXFP4 GGUFs | Zip download, run | Apache-2.0 | Zero-install Windows HIP, with three explicit numerics modes. |
| Kyojin | [Yamz-Labs/kyojin](https://github.com/Yamz-Labs/kyojin) | Strix Halo only (`gfx1151`), one 128 GB mini PC | Own EXL3 packs | Source build, published HF packs | MIT | 300B-class MoE models on one APU, with quality and long-context reports to match. |
| Lucebox | [Luce-Org/lucebox](https://github.com/Luce-Org/lucebox) · [lucebox.com](https://www.lucebox.com/) | AMD RDNA (`gfx1201`/`gfx1151`) and NVIDIA; separate CUDA and HIP builds | Its own checkpoints and quants | Source build | Apache-2.0 | Heterogeneous execution across a discrete card, an iGPU, and SSD. |
| NInfer | [Neroued/ninfer](https://github.com/Neroued/ninfer) | One NVIDIA RTX 5090 (`sm_120a`) | Own `.ninfer` artifacts | Source build only | Apache-2.0 | Single-GPU residency and long-context reuse behind a deliberately small product boundary. |
| Infernix | [Wallawalla47/Infernix](https://github.com/Wallawalla47/Infernix) | One NVIDIA RTX 50-series GPU (tuned on the 5090) | Own `.infernix` artifacts, plus NInfer `.ninfer` | Source build, release zip | Apache-2.0 | Flash-Next on one 32 GB card, with a prefix cache built for Qwen's linear-attention layers. |
| CapyFlash | [ValerioDolci/capy-flash](https://github.com/ValerioDolci/capy-flash) | NVIDIA RTX 50-series (`sm_120`), 16 GB cards, one or two | Own `.ninfer` artifacts, converted from NVFP4 or GSQ checkpoints | Installer script, release tarball, Docker | Apache-2.0 | A 125B MoE on consumer 16 GB cards, with agent-session serving. |
| Xolotl | [Indras-Mirror/Xolotl](https://github.com/Indras-Mirror/Xolotl) | NVIDIA, compute capability 7.5+, 24 GB measured | Own GSQ quants, plus GGUF | Source build | MIT | A 552B MoE on one 24 GB card, by tiering experts across VRAM, RAM, and NVMe. |
| Basalt | [jesdga95/basalt](https://github.com/jesdga95/basalt) | NVIDIA Blackwell (`sm_120`), one or two cards | One `.basalt` pack per model | Source build | MIT | One-file model packs, and a second GPU used as an expert tier rather than a parallel rank. |
| MegaCapybara | [perkel666/MegaCapybara](https://github.com/perkel666/MegaCapybara) | NVIDIA RTX 5090, Windows 11 and Linux | Own MXFP4/MXFP6 conversions | Prebuilt binaries, HF quants | MIT | Many coding agents on one 5090, with a launcher that prices each setting. |
| Decapybara | [giveen/Decapybara](https://github.com/giveen/Decapybara) | One RTX 5090 (`sm_120`), Linux x86_64 | MegaCapybara `.mcapy` containers | Source build | Not stated | A readable reconstruction of a closed engine's server and kernel structure. |
| SparkInfer | [gittensor-ai-lab/sparkinfer](https://github.com/gittensor-ai-lab/sparkinfer) | NVIDIA Blackwell (`sm_120`), 32 GB | In-house NVFP4 or FP8 checkpoints | Container image | MIT | A short model list optimized under an automated per-PR eval. |
| tqLLM | [turbo-tan/tqllm](https://github.com/turbo-tan/tqllm) | NVIDIA RTX 30 or newer with 24 GB, or a GB10; Linux | TurboQuant TQ3/TQ4 GGUF | Source build, Docker, release images | Apache-2.0 | A 3-bit 27B and Flash-Next on one 3090, with a built-in request monitor. |
| fngine | [satellitedown/fngine](https://github.com/satellitedown/fngine) | One RTX 5090 (`sm_120a`), ~60 GiB host RAM, AVX-512 CPU | Its own pack, built from upstream IQ GGUF shards | Source build, pinned installer | Apache-2.0 | Keeping native IQ expert formats while splitting experts across GPU and CPU. |
| FastLocalExperts | [sirfyyn/FastLocalExperts](https://github.com/sirfyyn/FastLocalExperts) | One 32 GB GPU plus 120 GB RAM | Its own FLE-IQ3_M GGUF | Source build, releases | MIT | A hot-expert GPU cache in a llama.cpp fork, documented with a measured report. |
| lean-qwenflash | [MaxDevv/lean-qwenflash](https://github.com/MaxDevv/lean-qwenflash) | One NVIDIA card with 16 GB VRAM, 8 GB host RAM, NVMe | EXL3 quant plus its own packed expert store | Source build | Not stated; ExLlamaV3 is MIT | A 125B MoE at full 262k context inside 8 GB of host RAM, by streaming experts from NVMe to the GPU. |
| Splash | [incoai/splash](https://github.com/incoai/splash) · [blog](https://inco.ai/blog/splash/) | Apple silicon, M3 or newer, 24–48 GB | Unsloth GGUF, MLX, NVFP4, Ternary Bonsai 2 | Homebrew | Apache-2.0 | Per-model Metal kernels and drafts behind one agent-ready server. |
| MTPLX | [youssofal/MTPLX](https://github.com/youssofal/MTPLX) · [mtplx.com](https://www.mtplx.com/) | Apple silicon, 16 GB and up | MLX quants | Mac app, PyPI | Apache-2.0 | Native MTP speculation that keeps the model's own sampling distribution. |
| llama.cpp | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Almost everything: CPU, NVIDIA, AMD, Apple, Intel, RISC-V, mobile | GGUF | Source, prebuilt binaries, Docker, curl installer | MIT | Breadth. The reference GGUF runtime and quant zoo. |
| Strata | [Niko1221/Strata](https://github.com/Niko1221/Strata) | NVIDIA RTX 20–50, AMD RX 6800+/R9700/RX 9000, Intel Arc, Apple Silicon, Strix Halo; Windows, Linux, macOS | GGUF, from a curated model set | Desktop app, one-line CLI installer, source build | MIT | One 125B MoE on almost any consumer PC, with agent setup included. |
| ds4 (DwarfStar) | [antirez/ds4](https://github.com/antirez/ds4) | Metal on Apple Silicon, CUDA including multi-GPU, ROCm on Strix Halo | Its own GGUF releases | Source build, one `make` target per platform | MIT | A few large models on consumer hardware, with SSD streaming and multi-machine tensor parallelism. |
| vLLM | [vllm-project/vllm](https://github.com/vllm-project/vllm) · [docs](https://docs.vllm.ai/) | NVIDIA, AMD, Intel GPUs and CPUs | HF checkpoints, many quant formats | pip/uv, container, source | Apache-2.0 | Serving throughput at scale: paged KV, continuous batching, every parallelism axis. |
| SGLang | [sgl-project/sglang](https://github.com/sgl-project/sglang) · [docs](https://docs.sglang.io/) | NVIDIA, AMD Instinct, TPU, Intel, Apple Silicon, Ascend | HF checkpoints | Docker, pip/uv, source | Apache-2.0 | Prefix-heavy agentic serving through a radix cache and a structured-generation frontend. |
| ExLlamaV3 | [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) · [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) | NVIDIA consumer GPUs (CUDA only) | EXL3 quants, HF checkpoints | Wheels, PyPI, source | MIT | Quality per bit (EXL3/QTIP), plus consumer tensor and expert parallelism. |
| TensorFold | [ashhart/TensorFold](https://github.com/ashhart/TensorFold) · [tensorfold.dev](https://tensorfold.dev/) | Apple Silicon (Metal) and NVIDIA (CUDA, incl. GB10) | MLX and HF checkpoints | Homebrew, release archives, desktop app | Apache-2.0 | One Zig binary across Metal and CUDA, with explicit capacity reporting. |
| kvmem | [kvmem/kvmem](https://github.com/kvmem/kvmem) | NVIDIA (Windows packages); llama.cpp on CUDA, ROCm beta | Whatever the backend loads | Windows packages, source build | Not stated | Virtualizing KV history across RAM and SSD with backend-exact restore. |

## AMD-focused engines

These target AMD GPUs specifically, and usually one AMD part: Strix Halo
(`gfx1151`) and RDNA4 (`gfx1200`/`gfx1201`) account for most entries, with
`gfx1100` and CDNA appearing where an engine spans more. Several are forks of
each other or of the same upstream, so each entry names its parent.

### strix-llama.cpp (halo-box)

[`halo-box/strix-llama.cpp`](https://github.com/halo-box/strix-llama.cpp) —
llama.cpp fork for AMD Strix Halo (Ryzen AI Max/Max+ 300, `gfx1151`, RDNA 3.5,
up to 128 GB unified LPDDR5X), MIT.

A community fork that keeps everything upstream and adds changes that only make
sense on this one APU, or that need more measurement on it before anyone
proposes them upstream. Vulkan/RADV is the default recommendation; HIP is
supported for `gfx1151` builds.

Strengths:

- **The deepest public `gfx1151` tuning of llama.cpp.** Additions include
  speculative prefill, adaptive speculative draft length, Vulkan coopmat and
  batched-matvec fixes for RDNA 3.5, ROCm/HIP quantized matmul tiles and
  register prefetching, compact `MUL_MAT_ID` for 256-expert MoE prefill, fused
  MoE decode, a fused Gated DeltaNet decode chain, grouped decode matvecs,
  WMMA flash attention for `D=256` prefill, and device-resident speculative
  checkpoints. Every HIP change is guarded on architecture, shape, and layout,
  so other devices see upstream behavior.
- **ROCmFPx quant types**, so ROCmFP4 GGUFs published for Strix Halo load and
  run.
- **You keep llama.cpp.** Same tools, same server, same GGUF surface, plus
  `--help` for the additions. Nothing is removed.
- **Fork policy that welcomes outside work**: AI coding agents may open pull
  requests here, and device-specific claims must carry numbers measured on the
  device against a baseline the contributor built.

Limits that matter for comparison:

- Fork drift. Its `gfx1151` deltas are ahead of what a pinned upstream
  comparator has, so an A/B needs both trees built, not a version bump.
- Non-Strix changes go to the companion
  [`halo-box/llama.cpp`](https://github.com/halo-box/llama.cpp) instead, so the
  Strix tree is not the route to upstream for general work.

hipEngine has reviewed its kernel deltas commit by commit in
[QWEN4EXP-EXTERNAL-FORKS-REVIEW.md](../archive/QWEN4EXP-EXTERNAL-FORKS-REVIEW.md).

### strixite

[`shawnshekari/strixite`](https://github.com/shawnshekari/strixite) — a
from-scratch HIP engine for Qwen3.8-Flash-Next on Linux and Strix Halo
(`gfx1151`), AGPL-3.0. The published setup requires 128 GB of system memory and
fast NVMe storage; model weights have a separate license.

Strengths:

- **Model-specific kernels and weights.** Hand-written HIP kernels run the
  model's mixture of experts, Gated DeltaNet, and sparse attention; its n-gram
  table streams from SSD. The project provides converted weights and a tool to
  make them from the original checkpoint.
- **Long-context serving.** The server supports 512k context, an 8-bit KV
  cache, and prompt caching in RAM and on disk.
- **Native multi-token prediction** using the model's own draft head, plus
  concurrent conversations whose decode steps share a forward pass.
- **OpenAI-compatible chat serving**, including streaming, tool calls,
  structured output, and image inputs from a separate weights file.

Limits that matter for comparison:

- One model and one tested hardware configuration; arbitrary GGUF models and
  other GPUs are not the target.
- Its own weight format and SSD-resident n-gram table differ from a standard
  GGUF setup, so comparisons need to account for model artifacts and storage.

### Gufo

[`gufo-org/gufo`](https://github.com/gufo-org/gufo) — C++20, MIT, ROCm 7.2.3
qualified, Strix Halo (`gfx1151`) only.

A deliberately vertical engine: one APU, and a short curated list of models per
size class rather than a general model loader. Text, speech recognition,
speech synthesis, embeddings, image generation, and video are all first-class
under one `gufo serve` with an OpenAI-compatible API.

Strengths:

- **Per-model kernels instead of one kernel set for everything.** Their stated
  policy is not to reuse kernels across models, to bound the blast radius of a
  change.
- **A quality report per model**, recording numerical checks, execution
  consistency, and unresolved gaps alongside the speed tables.
- **Concurrency, cancellation, and conversation caching are treated as primary
  workloads**, not additions bolted onto a single-user loop.
- **Reproducible builds.** Nix pins the whole toolchain and carries a ROCr
  queue-memory patch for Strix Halo; a plain CMake build uses the same source,
  flags, and install rules.
- **Multi-modality in one binary** — the audio, embedding, image, and video
  models are served by the same engine, not a stack of separate projects.

Limits that matter for comparison:

- Portability is not a goal. Smaller-memory Strix configurations should work,
  but no other GPU is a target.
- The model list is intentionally short; an arbitrary GGUF is not the use case.
- Two unmerged variants cover the edges: a
  [Windows port](https://github.com/pixmaate/gufo) and
  [RDMA for dual Strix Halo](https://github.com/neuhaus/gufo).

### halogen

[`peonist-ai/halogen-flash-server`](https://github.com/peonist-ai/halogen-flash-server)
— proprietary EULA, container image only. The engine binary, serving front end,
and benchmark tooling are closed source; documentation and benchmark scripts are
public.

A single-model, single-GPU engine: Qwen3.8-Flash-Next on Strix Halo, with every
kernel written for that one GPU and that one model family. There is no
portability layer and no fallback path, by design.

Strengths:

- **Depth per kernel.** With one model and one device to satisfy, it can take
  routes a general engine cannot afford to maintain.
- **Serving built around agent sessions**: prompt cache, on-disk cache that
  survives a restart, session keep/fork/join, per-request token budgets, and
  documented setups for Codex (Responses API) and Claude Code (Messages API).
- **An exactness guarantee on speculation.** At temperature 0 the output is
  byte-identical to serial greedy decode, verified per release, across both
  draft sources.
- **Reads a llama.cpp GGUF of the same model** directly since 0.7.0, so an
  existing file can run on these kernels with no conversion step.
- **Publishes its own precision accounting** by tensor family rather than
  quoting a format name.

Limits that matter for comparison:

- Closed source. Its mechanisms can be read about but not ported, and the EULA
  governs use of the binary.
- One model family. A Flint (40B MoE) preview image exists; that is the extent
  of the model surface.
- Its published benchmark script is the audited example of the protocol
  confounds listed in
  [QWEN4EXP-EXTERNAL-FORKS-REVIEW.md](../archive/QWEN4EXP-EXTERNAL-FORKS-REVIEW.md#reading-any-external-throughput-row-protocol-confounds).

### hipfire

[`warpfront/hipfire`](https://github.com/warpfront/hipfire) — Rust runtime plus
hand-written HIP kernels, Apache-2.0, dynamically loaded ROCm.

The closest architectural peer on this list. It ships its own runtime, model
implementations, quantization formats (MQ3–MQ6, MQ4R, MFP4), typed kernel
dispatch, and HIP kernels. There is no Python, PyTorch, or third-party inference
engine in the hot path — the same torch-free position hipEngine holds, reached
from a different language.

Strengths:

- **Own formats end to end.** Repacking, quantization, and kernels are designed
  together, which is why it can carry formats that GGUF does not express.
- **Ships the hard features, not just the kernels**: speculative decoding with
  DFlash draft models and native MTP heads, expert parallelism across 2–8 GPUs,
  and an OpenAI-compatible daemon, behind one CLI (`pull`, `run`, `chat`,
  `serve`, `bench`, `quantize`).
- **A curated registry** of models, quant tiers, and draft sidecars, with a
  per-GPU pick-a-model guide.
- **Redline**, a retained-PM4 replay transport that removes launch overhead
  after the ordinary graph is proven safe, with a published graft contract
  covering ROCr/HSA resource ownership, artifact and kernarg identity, and
  shadow validation. hipEngine's assessment of whether that idea transfers to
  `gfx1151` is in [ROOFLINE.md](../ROOFLINE.md) §9.9.
- **Fast-moving, well-instrumented releases.** Each changelog entry names the
  fixture, method, and kill switch for its figures.

Limits that matter for comparison:

- Model support is the registry. An arbitrary GGUF is not the entry point; you
  quantize into its formats first.
- Several headline routes are architecture-gated (`gfx1201`-only prefill paths,
  `gfx1151`-only sidecars and NPU routes), so a cross-GPU comparison has to
  check which route actually ran.
- Multi-GPU expert parallelism is opt-in and described as experimental.

### radiance

[`StillDeadcode/radiance`](https://codeberg.org/StillDeadcode/radiance) — C++
and HIP, Apache-2.0. AMD GPUs; the bundled kernels target RDNA4 (`gfx1200`,
`gfx1201`).

An inference server that covers more than text: language models plus image,
video, speech, and music generation, behind one HTTP API that speaks OpenAI's
chat, completions, Responses, images, audio, and video APIs, Anthropic's
Messages API, and llama-server's native one. Model architectures, kernel
libraries, and quantisers are `.so` plugins, and the bundled ones are built and
loaded the same way a third party's would be.

Strengths:

- **A plugin boundary that is actually exercised.** The in-tree kernels
  (`libr4d`) and model plugins go through the same loader as an external one, so
  a new architecture or a new AMD family is a plugin rather than a branch in
  dispatch code.
- **Multi-modal breadth on one server**: image generation and editing, video
  with stereo audio, speech synthesis, music, OCR, and embeddings beside chat.
- **An honest fallback.** The engine starts on any AMD family, and a machine
  with no GPU builds the AVX-512/AVX2 host backend that serves every model —
  slowly for the 27B, but it runs.
- **Quantisation recipes published per model** — FP8, MXFP4 through AMD Quark
  AWQ, GPTQ int4, w4a8, and ParoQuant learned rotations — each with a `.rad`
  container and a compose file.
- **Speculation as a menu, not a mode**: MTP, DFlash, DFlash2, and DSpark,
  chosen per model.

Limits that matter for comparison:

- The bundled device kernels are RDNA4 only. Another AMD family needs a kernel
  library that does not ship here.
- `.rad` containers are the entry point; it is not a GGUF runner.
- Distributed as Compose files rather than a package, with the documented serve
  flags sized for two 32 GB R9700s.

### Brilliance

[`Dad_Vibes/Brilliance`](https://codeberg.org/Dad_Vibes/Brilliance) — a Windows
fork of radiance, Apache-2.0, for RDNA4 R9700 cards.

The Windows port of radiance, built natively against AMD's ROCm 10.1 driver —
no WSL, no Linux, no Docker — and tuned there for two R9700s in tensor parallel.
It forked from radiance 1.10.0 and diverges deliberately: the author states
there is no intention of keeping compatibility with upstream.

Strengths:

- **Native Windows HIP with no Linux layer underneath.** A PowerShell setup
  installs the build tools and the matching ROCm driver, then builds and runs
  the test suite on the cards.
- **A documented port, not just a build.** `docs/WINDOWS-PORT.md` records the
  design decisions and the measurements behind them.
- **Results named by the exact `.rad` file served**, so each row traces to an
  artifact rather than a model nickname.
- **Radiance's model surface comes along**: the same `.rad` containers and the
  same API dialects.

Limits that matter for comparison:

- Pre-alpha, tested on exactly one machine (9950X3D, two R9700s, 128 GB).
- One driver version. Any other Adrenalin or PRO driver replaces the ROCm one
  and breaks the engine.
- Windows gives the two cards no direct peer path, so they exchange through
  system memory, which costs short-prompt prefill most.
- No image or audio input, and the Linux code still in the tree is neither built
  nor tested. It serves `.rad` files and some raw checkpoints, not GGUF.

### vllm-radiance

[`magiccodingman/vllm-radiance`](https://github.com/magiccodingman/vllm-radiance)
— a container image around vLLM 0.30.0 for the R9700, not an engine of its own.

It pairs a pinned vLLM ROCm stack with radiance's hand-written `libr4d` RDNA4
kernels — attention, gated-delta-net, vision, all-reduce, MXFP4, and DFlash —
while keeping vLLM's scheduler and serving surface.

Strengths:

- **vLLM's batching and serving with RDNA4 kernels underneath**, a combination
  neither project ships alone.
- **A qualified default, described as one.** The primary qualified environment
  is two R9700s with FP8 or Quark MXFP4 weights and FP8 KV; speculative modes
  stay opt-in because their cross-mode output-equivalence gate has not passed.
- **Negative results are published.** Its RX4 and RX5 experiments regressed
  inter-token latency and failed greedy and tool-call gates, and the reports say
  so and leave the switches off by default.
- **Capacity tables that separate an admission ceiling from a promise**, plus a
  documented KV-offload pinning policy that falls back coherently when pinned
  registration fails on one rank.

Limits that matter for comparison:

- R9700 only, and qualified on two of them. Other models, quantisation recipes,
  and GPU counts "may work" but are not qualified.
- No licence file in the mirror or its upstream, though both vLLM and radiance
  are Apache-2.0.
- A distribution to run rather than an engine to extend: its value is the pin,
  the kernel selection, and the qualification record.

### paiton-vllm-plugin

[`Eliovp-BV/paiton-vllm-plugin`](https://github.com/Eliovp-BV/paiton-vllm-plugin)
— Apache-2.0, a vLLM plugin plus per-model guides for AMD Radeon.

A vendor-shaped package: one plugin and a set of Docker launchers per model,
each pinning the runtime that model was qualified in, for R9700 cards. It also
ships image and video packages driven through ComfyUI or a CLI. It does not
install or replace vLLM, PyTorch, or ROCm.

Strengths:

- **Per-model guides with benchmark tables**, so choosing between an accuracy,
  a speed, and a long-conversation profile is documented rather than guessed.
- **Containers that carry their own runtime**, which takes the ROCm and PyTorch
  version matrix out of the user's problem.
- **A path for existing vLLM users**: the pip-installable plugin adds named
  presets to an environment you already have.
- **Coverage past chat**: FLUX and Qwen-Image for images, Wan and MiniMax H3 for
  video, plus a meeting-notes preview.

Limits that matter for comparison:

- R9700 only; other GPUs are explicitly unqualified.
- One model at a time on a single card.
- A distribution layer over vLLM, so engine-level behaviour is vLLM's.

### DS41Engine

[`vincalkr/DS41Engine`](https://github.com/vincalkr/DS41Engine) — C++/HIP, MIT.
DeepSeek-V4.1-Flash on one or two RX 7900 XTX (`gfx1100`).

An engine for a single large model, built around one promise: every expert the
model routes to is computed, wherever it lives — card 0's VRAM, a second card,
an 80 GiB pinned RAM arena, or the NVMe. No route bias, no skipped experts.

Strengths:

- **Exact routing as a design constraint**, which makes its output comparable to
  a full-precision reference rather than to another approximate engine.
- **A conversation pool in RAM and on disk**: conversations a request does not
  continue wait in pinned memory, survive a restart, and resume without
  re-reading the prompt.
- **Measurement discipline around a host-level problem.** Its power-tune script
  keeps the NVMe and CPU cores awake between expert reads and reports what that
  alone is worth; the expert rank is rebuilt from observed router counts and
  applied only if it measures faster.
- **Quality reported against a reference** — perplexity, top-1 agreement, KL —
  instead of speed alone.

Limits that matter for comparison:

- `gfx1100` only, on Linux with ROCm. A second card is an expert tier, not a
  parallel rank.
- About 570 GB of NVMe for the model and its Engram tables.
- One model, and decode depends on host RAM and NVMe behaviour, so results move
  with the box.

### WHIRL

[`tsaipifong/whirl-llm`](https://github.com/tsaipifong/whirl-llm) — Apache-2.0,
native Windows C++/HIP for the Radeon 8060S (`gfx1151`) and the R9700
(`gfx1201`).

A Windows engine that ships every kernel in the zip: no HIP SDK, no ROCm, no
Visual C++ redistributable, only the AMD driver. It runs two MXFP4 models
quantised for it, Swift-1.5 27B dense and Ornith-1.5-35B-A3B MoE.

Strengths:

- **The lowest-friction Windows install on this page.** Unzip, run
  `whirl.exe devices` to confirm the card, then start a server that speaks the
  OpenAI API.
- **Three explicit numerics modes** (`precise`, `balance`, `fast`), so
  bitwise-lossless output and maximum speed are choices, not hidden flags.
- **Kernels and quantisation designed together**: FP8 prefill on RDNA4 matrix
  units, Q8_0 SSM parameters so MTP verification stays bitwise identical to
  plain greedy decode, and a Q6_K output head chosen for the draft step.
- **A comparison against a pinned llama.cpp release** with the same GGUF files,
  prompts, and greedy decoding, and the caveats — single runs, a noise
  threshold, where speculation differs between engines — written beside the
  tables.
- **RAM and SSD KV tiers** for sessions that outgrow GPU memory.

Limits that matter for comparison:

- Two GPUs supported, one GPU per process; other AMD cards are untested and
  under 16 GB is unsupported.
- Windows 11 only.
- Its MXFP4 models are its own GGUFs, quantised for its kernels; other files are
  not the tested path.
- On the 8060S the user must set the UMA frame buffer first, because Windows'
  shared GPU memory is not used for the model.

### Kyojin

[`Yamz-Labs/kyojin`](https://github.com/Yamz-Labs/kyojin) — Python on ROCm,
MIT, one Strix Halo APU (`gfx1151`, Ryzen AI Max+ 395) with 128 GB of unified
memory.

An ExLlamaV3-based engine with its own EXL3 packs, aimed at 300B-class MoE
models — 100 GB and up — on a single mini PC: Qwen3.8-Flash-Next,
GLM-5.3-Flash, MiMo-V2.6-Flash, and an experimental 185 GB DeepSeek-V4.1-Flash
that streams most experts from NVMe.

Strengths:

- **Quantization is the product.** The published packs are sized to the memory
  the APU actually has, with an 8-bit KV cache used past 262K tokens and a
  500K-token context path (`-c 524288`) that is a documented mode rather than a
  stretch goal.
- **Its benchmarks name their conditions.** Per-model prefill and decode tables,
  a needle test at roughly 500K tokens, and a `doc/benchmarks.md` that records
  method and sources, including which cells are warm and which are cold.
- **A model-level refusal hook applied at load**, leaving the weights untouched,
  with ready-made packs where that is wanted.

Limits that matter for comparison:

- One APU. No discrete-card path and no multi-device configuration.
- ExLlamaV3 means PyTorch on the hot path, which is the architectural distance
  from a torch-free engine; what `gfx1151` does here is bounded by that layer.
- The model list is the set of packs it publishes; a new model needs an EXL3
  conversion and a working architecture path.
- A single lab's project rather than a community-maintained one.

### Lucebox

[`Luce-Org/lucebox`](https://github.com/Luce-Org/lucebox) ·
[lucebox.com](https://www.lucebox.com/) — C++17 with separate CUDA 12+ and HIP 7+
builds, Apache-2.0. Tuned on RDNA4 (`gfx1201`) and Strix Halo (`gfx1151`), with
RTX 3090 results published as well.

A speculative inference server built around per-model, per-device
specialization, including a heterogeneous mode that spans a discrete card and
an iGPU in one process.

Strengths:

- **Heterogeneous execution as a first-class mode.** Experts tier across a
  discrete GPU, an integrated GPU and SSD. That is a different split from
  tensor-parallel ranks, and it is the one that matters for APU-plus-card
  machines.
- **Prefill technique published with code and numbers.** PFlash and KVFlash, a
  KV pool sized independently of the context window, and a megakernel experiment
  with its own results page — each linked to a write-up rather than asserted.
- **One stack across model families and modalities**, including DFlash2 and
  DSpark speculation, paged attention with continuous batching, and an image
  generation path on the same server.
- Separate CUDA and HIP build targets, so the AMD path is not an emulation of
  the NVIDIA one.

Limits that matter for comparison:

- Specialization is per optimization: each published result is tied to a named
  model and device pair, and the general path is narrower than the tuned ones.
- A CUDA result does not transfer to HIP; the two builds are maintained
  separately.
- Its most interesting configuration needs both device classes present.
- A company-led project: the repository is open, and the site carries the
  benchmark write-ups and the community channels.

## NVIDIA-focused engines

Targets NVIDIA consumer and prosumer cards rather than datacenter parts. The
constraint that shapes these engines is VRAM per card, not bandwidth per
device.

### NInfer

[`Neroued/ninfer`](https://github.com/Neroued/ninfer) — C++/CUDA written from
scratch, Apache-2.0, one RTX 5090 (`sm_120a`).

A single-GPU engine with a deliberately small product boundary: one card, one
resident model, one to eight execution lanes fixed at startup. It serves a
short curated list — Qwen3.6-27B, Qwen3.8-27B, Qwen3.6-35B-A3B, and
Qwen3.8-Flash-Next — from its own `.ninfer` artifacts, which carry the model
configuration, encoded weights, logical bindings, and frontend resources
together. CapyFlash, below, is its two-GPU fork.

Strengths:

- **Residency and scheduling are the design, not an addition.** A startup-fixed
  typed KV pool is shared by active requests and retained prefixes; completed
  conversation endpoints and stable input boundaries are kept as reusable
  checkpoints; under pressure the scheduler reclaims retained resources before
  pausing a request, and a paused request resumes from a snapshot or by
  replaying already-committed tokens.
- **Speculation chosen per architecture**: MTP with one to five draft windows,
  DFlash for 35B-A3B, and DFlash2 with up to fifteen drafts on the 27B when the
  companion weights are installed — each named for the architecture that
  implements it, with the unsupported combinations stated.
- **A wide KV storage choice** (BF16, INT8, FP8, NVFP4, K8V4) instead of one
  fixed dtype, so context capacity and precision are a startup decision.
- **A precise capability statement.** The README says which model supports
  which speculative mode, what an architecture does not support, and exactly
  where the product boundary ends.
- **Convert your own weights.** Recipes are published, and an existing v2
  artifact can be upgraded locally without downloading the weights again.

Limits that matter for comparison:

- One RTX 5090. No multi-GPU, no distributed serving, and no other
  architecture — the build rejects anything but `sm_120a`.
- No install target and no packaged binary: it runs from its source build tree,
  so "how you get it" is a compiler toolchain rather than a download.
- Residency is fixed at startup, so `--spec` and `--vision` are decisions made
  before the server starts, not per request.
- The curated model list is the entry point; an arbitrary checkpoint needs
  conversion and an implemented architecture path.
- A single maintainer's personal project, with direction set by that
  maintainer.

Its fork family is the largest on this page, and a fork is usually what a
specific hardware report means by "NInfer": `wamansou/ninfer-tp2-1m` adds
two-GPU tensor parallelism and YaRN context scaling on the 27B,
`ivanov84/ninfer-windows-tp2` moves that to native Windows with a pinned-host
peer mailbox, `parallelno/ninfer-dflash2-tp2` adds DFlash2 on Windows, and
`t-infra/ninfer-tp2-5060ti` targets two 16 GB 5060 Ti cards. CapyFlash began as
this line (`ninfer-tp2`) and Infernix began as a fork as well. Compare a fork
against the upstream commit it names, not against the current NInfer release.

### Infernix

[`Wallawalla47/Infernix`](https://github.com/Wallawalla47/Infernix) — C++/CUDA,
Apache-2.0, one RTX 50-series GPU (tuned on the 5090). Grew from a NInfer fork
into its own engine; it reads NInfer's `.ninfer` artifacts as well as its own
`.infernix` files.

Serves Qwen3.8-Flash-Next — 24,576 routed experts, far more than a 32 GB card
holds — by keeping experts in pinned system RAM, caching the hot ones in VRAM,
computing part of each layer's cache misses on the CPU while the rest cross
PCIe, and reading the n-gram embedding table from NVMe. Qwen3.8-27B runs on the
same engine.

Strengths:

- **A prefix cache built for Qwen's linear-attention layers**: KV cached by
  content in 64-token blocks, snapshots of the recurrent state at the points
  that get reused, and three tiers — GPU, host RAM, and a file that survives a
  restart. In its own agentic replay this is what most of the time-to-first-token
  gap comes from, and it is the piece such comparisons usually turn on.
- **Four speculative routes in one engine**: MTP, DFlash, DFlash2, and n-gram
  copy drafting, all with exact sampling.
- **Seven KV formats**, including `vq2` and `k4v2`, which hold several times
  INT8's context.
- **Agent-facing correctness work, not just API shape**: recovery of broken XML
  tool calls and of calls stranded in unclosed thinking, JSON Schema / GBNF /
  regex constrained decoding, thinking budgets with a repetition guard, and
  `/v1/decide` reading answers off token probabilities.
- **Failure containment**: a failed expert read fails the requests running at
  that moment rather than the server, and the prefix cache persists across
  restarts.
- **Native Windows and Linux builds**, with a release measured on the same
  machine and link as the comparators it reports against.

Limits that matter for comparison:

- It targets one RTX 50-series GPU; the README describes no multi-GPU path.
- Its model set is Qwen3.8-Flash-Next and Qwen3.8-27B. Broader support is not
  claimed.
- **A stated provenance.** The project says everything added since the NInfer
  fork, including most of its README, was written with AI assistance and that
  this is hobby development. That belongs beside any claim taken from it.
- Its ratios against Strata and NInfer come from one machine over a PCIe Gen5
  x8 link, where expert-cache misses cross PCIe on every token. A wider link
  moves the offloaded rows.

### CapyFlash

[`ValerioDolci/capy-flash`](https://github.com/ValerioDolci/capy-flash) — C++,
Apache-2.0, built for RTX 50-series cards (`sm_120`) with 16 GB of VRAM, one or
two of them.

An inference server for two models on hardware that should not fit them.
Qwen3.8-Flash-Next is a 125B mixture-of-experts model: capy-flash keeps the
routed experts that do not fit in VRAM in host RAM and computes them on the
CPU, while everything else runs on the GPU, split across two when there are
two. Qwen3.8-27B is dense and runs entirely on the GPUs. It was `ninfer-tp2`,
the two-GPU tensor-parallel fork of
[`Neroued/ninfer`](https://github.com/Neroued/ninfer), until v0.5.0.

Strengths:

- **Consumer-hardware MoE serving.** A 125B model on one or two 16 GB cards plus
  64–96 GB of system RAM, with a startup check that refuses with the actual
  numbers when the machine cannot hold it.
- **Agent-first serving, not just an API.** Prefix cache across turns, idle
  conversations moved to RAM and optionally to disk instead of recomputed, and
  an expert cache that follows the conversation. It speaks Chat Completions,
  Responses, and Anthropic Messages, so Claude Code and Codex CLI point at it
  without a translation layer.
- **Tuned profiles instead of flag archaeology.** `--profile flash-next`,
  `27b`, and `27b-16gb` expand to the tested configuration for a model and card
  count, with one launcher per row of its hardware table.
- **An installer that diagnoses the host** — GPUs, driver, AVX2/AVX-512, RAM,
  disk — before it downloads a release and one model repository.
- **A stated measurement boundary.** Its README separates numbers measured on
  the author's machine from user-reported ones, says when a comparison engine
  took a route this one deliberately does not, and notes that output is not
  bit-reproducible run to run because expert placement depends on timing.

Limits that matter for comparison:

- NVIDIA only, and specifically `sm_120` with CUDA 13.1. There is no AMD path.
- Decode of the MoE model is bound by host-RAM bandwidth for the experts
  computed on the CPU, so a slower CPU or slower memory decodes slower. That is
  a different bottleneck from a fully-resident engine, and it does not
  transfer as a kernel comparison.
- Flash-Next is tested with one request at a time by default; higher
  concurrency is accepted, but the multi-request measurements are on the dense
  27B.
- It is a fork of a fork, with a lineage of prior TP2 work recorded in
  `NOTICE`. Upstream is `Neroued/ninfer`, not this tree.

### Xolotl

[`Indras-Mirror/Xolotl`](https://github.com/Indras-Mirror/Xolotl) — MIT, a
heavily extended fork of Strata whose engine, server, and kernels were merged at
0.1.44.1. NVIDIA only.

Where Strata runs one model family across every consumer vendor, Xolotl points
the same tiered-expert design at the largest models and one vendor:
DeepSeek V4.1-Flash (552B MoE), GLM-5.3-Flash (320B MoE),
Qwen3.8-Flash-Next, and a Qwen3.8-27B dense port, on one 24 GB card plus system
RAM and NVMe. It was Strata-DSV4.1 before the rename.

Strengths:

- **Three explicit memory tiers with prefetch**: hot experts in VRAM, a large
  pinned host arena the GPU reads by DMA, and the remainder on NVMe, with the
  next layer's experts fetched a layer ahead.
- **Its own quantization and format work.** GSQ mixed ternary/2-bit/3-bit
  experts, an SSD-resident Engram lookup table, speculative decoding with exact
  sampling, and a bit-exact 4-bit KV cache for 512K context.
- **Per-model engines rather than one generic path.** DeepSeek-V4.1, GLM-5.3,
  and the dense 27B each carry their own quantization and tiering work.
- **Agent-loop behavior that ships**: a conversation pool that resumes from the
  longest matching prefix (images included), batch slots, and exact resume
  inside the park window ring.
- **Horizontal options when one card is not enough**: `--layer-split` across
  cards with a PCIe hand-off, and `--remote-device` for a second card that
  holds nothing but extra experts.
- **A stated measurement basis.** Every number names the one machine, the task
  set, and the server path it came through, and numbers from other hardware are
  labelled as reported rather than reproduced.

Limits that matter for comparison:

- NVIDIA only, compute capability 7.5 and newer, with a per-architecture build
  and CUDA 13 for RTX 50.
- 24 GB of VRAM is what has been measured. 16 GB is expected to load but is
  untested, and below that it will not run.
- RAM size sets the expert cache size, and the expert cache size sets decode
  speed. The storage tier matters too: NVMe on ext4 or NTFS, not exFAT and not
  a hard drive.
- Research code from one developer, measured on one machine.
- Under a layer split the engine serves one request at a time, and batch slots
  and the conversation pool are single-card features.

### Basalt

[`jesdga95/basalt`](https://github.com/jesdga95/basalt) — a fork of Strata,
MIT, targeting NVIDIA Blackwell (`sm_120`) on Linux and Windows.

A single-model engine: Qwen3.8-Flash-Next, with the experts that do not fit in
VRAM living in system RAM and, optionally, on a second GPU used as a peer
expert tier. It packages a model as one `.basalt` file — weights, tokenizer,
MTP drafter, expert profile, and vision projector together — and ships packs
built from NVIDIA's NVFP4 experts, unsloth's UD GGUFs, and ISTA-DASLab's
GSQ-RCO quants.

Strengths:

- **A second GPU as an expert tier, not a tensor-parallel rank.** The peer-card
  layout adds capacity without requiring the model to fit across both cards.
- **One-file model packs with a documented layout**, so what runs is a single
  artifact rather than a directory of shards plus sidecars.
- **Quality measured per pack against Q8_0**, so the trade between an NVFP4, a
  `UD-Q4_K_XL`, and an `IQ3_S` pack is stated rather than implied.
- **`basalt-probe` measures the machine** — including its cold decode speed —
  and suggests configuration values, so the tiering knobs start from a
  measurement instead of a guess.
- **It carries forward Strata's kernel base**, with ggml/llama.cpp
  quantization formats and kernels extracted into its own tree.

Limits that matter for comparison:

- Qwen3.8-Flash-Next only, on compute capability 12.0 with CUDA 13. Other
  models and other cards are out of scope.
- A fork of Strata: it inherits that engine's structure, while its model file,
  packs, and probe are its own. A fix in one tree does not imply the other.
- Its own contribution page asks for spare compute to test on, which is a fair
  signal for how much of the hardware matrix has actually been exercised.

### MegaCapybara

[`perkel666/MegaCapybara`](https://github.com/perkel666/MegaCapybara) — MIT,
prebuilt Windows 11 and Linux binaries for one RTX 5090. The repository carries
documentation, the launcher's explanations and releases; the kernels ship as
binaries.

Serves Qwen3.8-27B and Qwen3.8-Flash-Next from its own MXFP4 and MXFP6
conversions, and treats a team of coding agents as the workload rather than a
single chat.

Strengths:

- **Weight formats chosen for the card.** MXFP4 and MXFP6 are computed natively
  by the 5090's tensor cores, so a step is close to one pass over the weights —
  the roofline argument that applies to any card, applied here through the
  weight format.
- **Multi-agent serving is the design.** One shared pool of 512-token pages
  instead of per-agent fixed slots; eviction that orders candidates by whether a
  waiting request needs them; preemption by swapping the youngest conversation
  to RAM with exact resume; chunked prefill so a 100K-token prompt does not
  stall agents that are decoding; and drafts from several conversations
  verified in one pass.
- **A launcher that prices settings before load.** Each model file carries
  measured top-1 agreement against the original weights, and the launcher totals
  accuracy, speed and memory for the combination chosen.
- Prefix reuse across VRAM, RAM and disk tiers, with the return cost stated per
  tier.

Limits that matter for comparison:

- One card. `sm_120a` and FP4/FP6 tensor cores are the target; nothing else is
  claimed.
- The model surface is its own Hugging Face conversions, and its accuracy
  figures are its own top-1 measurement rather than an independent eval.
- No source in the repository, so a kernel cannot be read, ported or checked; a
  third party has begun a C++/CUDA reconstruction
  ([Decapybara](https://github.com/giveen/Decapybara), below) for that reason.
- Windows and Linux only.

### Decapybara

[`giveen/Decapybara`](https://github.com/giveen/Decapybara) — C++20 and CUDA, no
license file, Linux x86_64, targeting the same model and card as MegaCapybara
(Qwen3.8-27B on an RTX 5090).

A third-party reconstruction of MegaCapybara's server and kernel suite, written
against that project's public documentation because its source is not published.
It is the only readable account of how that engine is put together.

What it covers, following the shape of the original's documentation: a
persistent single-launch decode megakernel, NVFP4/MXFP8 block GEMV,
FlashAttention prefill with KV dequantization, DeltaNet linear-attention state
updates, MTP vocabulary slicing with a DFlash probability-tree drafter, top-8
MoE routing, MXFP8 activation quantization, VRAM-to-pinned-RAM-to-disk KV
paging, and a launcher that estimates capacity and pairs a drafter. It reads
MegaCapybara's own `.mcapy` container format.

Limits that matter for comparison:

- **It is not MegaCapybara, and nothing published shows it reproduces it.** No
  tests and no parity results against the original's outputs, over roughly 73 KB
  of source in 35 files: a reconstruction of the architecture, not evidence that
  the architecture works.
- No license file, so the terms for reading or reusing the code are unstated.
- A single contributor's project, started in October 2026.
- Its value here is as documentation of a closed binary's structure. Treat any
  technique taken from it as an unverified lead.

### SparkInfer

[`gittensor-ai-lab/sparkinfer`](https://github.com/gittensor-ai-lab/sparkinfer)
— C++/CUDA, MIT, Blackwell (`sm_120`), a 2.5 MB binary distributed as a
container image.

Optimizes a deliberately short model list — Qwen3.8-27B is the headline
checkpoint — and is unusual in how it decides what counts as an improvement.

Strengths:

- **An automated eval scores every pull request** on a pinned RTX 5090 and
  regenerates the published table from that run, with a separate no-regression
  guard on a second checkpoint so an optimization cannot win by making the other
  one worse.
- **Speculation with an explicit degradation path.** Up to eight concurrent
  requests speculate together; at higher concurrency the drafter's memory steps
  aside because it cannot be used; requests with tools, `response_format`,
  vision, penalties, logit bias or logprobs take the ordinary path; and a
  request that outruns the drafter's 16,384-position window ends speculation for
  its group rather than silently changing what is served.
- **Attested artifacts.** `gh attestation verify` against the image digest, and
  an in-house ModelOpt quantization recipe published with the checkpoint.
- Four API surfaces (OpenAI, Anthropic Messages, Ollama, LM Studio) behind one
  endpoint.

Limits that matter for comparison:

- Blackwell only, container-first, with a fixed checkpoint layout.
- The optimized model list is short by design; breadth is not the goal.
- In-house quantization means the artifacts are its own conversions, so a
  same-weights comparison against a GGUF engine needs care.
- Development is competition-driven: the eval bot optimizes what it scores, and
  a property that is not scored will not be defended.

### tqLLM

[`turbo-tan/tqllm`](https://github.com/turbo-tan/tqllm) — C++/CUDA, Apache-2.0,
Linux with an NVIDIA GPU (RTX 30 or newer with 24 GB, or a GB10).

An engine with its own weight format (TurboQuant TQ3/TQ4) that targets the
small-VRAM end: a 27B in 3 bits, and Qwen3.8-Flash-Next on a single RTX 3090
with 64 GB of host RAM.

Strengths:

- **A quantization format with a documented contract.** TQ3, TQ4 and the
  compatible formats are specified in `docs/reference/formats.md`, with the
  loader's acceptance rules rather than a naming convention.
- **Observability in the same process.** A monitor on the API port shows prefill
  and output rates per request, with no extra service, second port or frontend
  build.
- **Comparisons against a named peer on the same machine, card, file and
  session**, with the method in `docs/BENCHMARKS.md` — the shape an external row
  needs before it can be reproduced here.
- Long-context cache (TqCache), DFlash2 and MTP drafters, and up to eight
  concurrent users with the same output as a single user.

Limits that matter for comparison:

- Linux and NVIDIA only. There is no ROCm path.
- It loads supported Qwen GGUFs and TurboQuant formats, not arbitrary GGUF.
- The 24 GB floor for the 27B and the host-RAM requirement for Flash-Next are
  hard preconditions, not recommendations.
- One maintainer, community-funded.

### fngine

[`satellitedown/fngine`](https://github.com/satellitedown/fngine) — C++20/CUDA,
Apache-2.0, Linux x86_64 on one RTX 5090 (`sm_120a`) with about 60 GiB of host
RAM and an AVX-512 CPU tier.

One model (Qwen3.8-Flash-Next, `qwen4_exp`), one machine, and a pack format
built around keeping the routed experts in the IQ formats they were published
in.

Strengths:

- **No requantization of the experts.** Routed experts keep their native GSQ-RCO
  IQ bytes, dense weights are FP8-row derived from BF16, the n-gram table is
  page-packed on NVMe, and the MTP experts stay resident.
- **Routing profiles as pack data.** Prefill and decode routing counts are
  collected once and shipped in the pack to rank expert placement, and a draft
  vocabulary trims the drafter's `lm_head` rows — with drafts verified exactly,
  so the vocabulary changes acceptance and never output.
- **It states its own gate.** The README records which HF-reference numerical
  thresholds are not met, what was verified instead, and the PCIe link width
  observed on the target box, rather than reporting a green run.
- An optional pack-level ablation manifest (FP32 refusal directions applied at
  load) shows the pack format carrying more than weights.

Limits that matter for comparison:

- One model on one card, with a required AVX-512 CPU and a specific host-RAM
  budget.
- Building a pack needs the upstream GGUF shards, the BF16 weights, and an
  expensive golden/reference pass.
- The original numerical thresholds are not all met; the project says so.
- The installer path pins a build to one published pack.

### FastLocalExperts

[`sirfyyn/FastLocalExperts`](https://github.com/sirfyyn/FastLocalExperts) — MIT,
a llama.cpp fork for one 32 GB GPU with 120 GB of RAM, built and measured for a
single model (Qwen3.8-Flash-Next as an 88.87 GB FLE-IQ3_M file).

Splits a large MoE along the expert boundary: attention, shared weights, KV
cache and a hot-expert cache on the GPU, the remaining routed experts computed
on the CPU, and the model's own MTP head as the drafter.

Strengths:

- **A measured technical report, not a README table.** Paired brackets with
  bootstrap intervals, a stated drift bound between reference runs, and a
  section on what did not work.
- **The split is documented as a decision**, including the expert-cache byte
  budget and why a llama.cpp fork was the cheaper route than a new engine.
- **Quality is checked against the format it replaces**: KL divergence against
  BF16 within the noise of a larger IQ4_XS file, plus benchmark scores reported
  beside the model card's own.
- It reports where it loses: prefill is slower than upstream llama.cpp.

Limits that matter for comparison:

- One model on one machine; the author says explicitly that it is not a general
  system.
- Inherits llama.cpp's surface, build and upstream drift.
- Prefill is the acknowledged weak point, which matters for agent workloads.
- The speed series used an internal retrained MTP head; the published head
  differs, and that is disclosed rather than smoothed over.

### lean-qwenflash

[`MaxDevv/lean-qwenflash`](https://github.com/MaxDevv/lean-qwenflash) — Python on
a vendored ExLlamaV3 fork, no license stated, Linux x86_64 on one NVIDIA card
with 16 GB of VRAM and a fast NVMe. The headline configuration is an RTX 4090
capped at 16 GB of VRAM and 8 GB of host RAM.

Qwen3.8-Flash-Next (125B MoE, 512 experts, 3.05 bpw EXL3) at its full 262k
context, with the host-RAM budget as the design constraint rather than VRAM.
Uncached experts are read from an NVMe expert store straight into a bounded GPU
cache with `O_DIRECT`, and the 51B-parameter n-gram embedding table (32 GB) is
streamed from disk the same way, so what stays resident is the attention mixers,
the DeltaNet state, the routers, the shared experts, the output head and the KV
cache.

Strengths:

- **Host RAM is the target, not the spillover.** The whole model runs at 8 GB of
  system RAM and 16 GB of VRAM, where several sibling Flash-Next engines ask for
  60 GB or more of host RAM.
- **All expert math stays on the GPU.** A cache miss streams the expert over PCIe
  into the compute stream instead of falling back to a CPU tier, which sidesteps
  the host-DRAM bandwidth ceiling the CPU-offload engines are built around.
- **Prefix caching for recurrent state.** DeltaNet state checkpoints are saved
  beside the KV cache, so a cached follow-up turn starts decoding instead of
  replaying the conversation.
- **A thin layer over a known base.** The server and the offload path sit on
  ExLlamaV3's loader and kernels and were adapted from `sybil-solutions`'
  GLM-5.3-Flash offloader, so the same approach already covers a second model
  family.

Limits that matter for comparison:

- One model on one card, from one named EXL3 checkpoint; the expert store and the
  n-gram table are built by its own packing scripts.
- No license file in the repository. The vendored ExLlamaV3 fork is MIT; the
  engine's own terms are unstated.
- Python and Torch at the server layer, over native offload extensions compiled
  from source, with a CUDA-only path. There is no ROCm build and no CPU
  fallback.
- A fast NVMe is a precondition, not a tuning option. The project's own to-do
  lists 12 GB and 8 GB cards, and the 24 GB and 32 GB cards, as untested.

## Apple-silicon engines

Unified memory removes the PCIe boundary that shapes the NVIDIA entries, and a
fixed memory budget shared with the OS replaces VRAM as the constraint. Metal is
the only kernel path.

### Splash

[`incoai/splash`](https://github.com/incoai/splash) ·
[inco.ai/blog/splash](https://inco.ai/blog/splash/) — Apache-2.0, Apple silicon
(M3 or newer, macOS 26.4+), installed with Homebrew.

Built around a short model list rather than a model zoo: per-model Metal
kernels, a matching DFlash2 draft, and an automatic memory plan for each
supported model.

Strengths:

- **Format tolerance unusual for a specialized engine.** Unsloth GGUF from 1 to
  8 bits including mixed-precision UD variants, MLX affine 2–8 bit with mxfp4
  and nvfp4, NVFP4 GGUFs, and Ternary Bonsai 2 in PQ2_0 — while refusing an
  all-BF16/F16/F32 file outright instead of half-loading it.
- **Agent-first serving.** OpenAI Chat Completions, Responses and Completions
  plus Anthropic Messages, prefix reuse, automatic batching, a built-in chat
  page, and `splash opencode|claude|codex|hermes|pi` to point an installed agent
  at the server.
- **Memory is planned, not guessed**: context and memory sized up to the model's
  native window, with `--max-memory`, `--max-context`, `--idle-release` and an
  INT8 KV cache by default.
- Its checkpoints are packaged so LM Studio's Bionic runtime can drive the same
  engine.

Limits that matter for comparison:

- Apple silicon only, so it is not a comparison target for a ROCm card; the
  ideas transfer, the kernels do not.
- A narrow model list, and its Hugging Face packages are built for Splash and
  load nowhere else.
- 36 GB of unified memory for the 4-bit examples (24 GB via smaller GGUFs), on
  macOS 26.4 or later.
- No multi-machine or multi-GPU path.

### MTPLX

[`youssofal/MTPLX`](https://github.com/youssofal/MTPLX) ·
[mtplx.com](https://www.mtplx.com/) — Python on MLX, Apache-2.0, Apple silicon,
shipped as a Mac app and as a PyPI package.

Uses the model's own multi-token-prediction heads as the only drafter, so no
second draft model competes for the same memory.

Strengths:

- **Sampling semantics are preserved.** Drafts are committed through rejection
  sampling with residual correction at the model's native temperature, rather
  than a greedy shortcut that changes what the model would have said.
- **Exactness is tested and reported**: a thousand four-token samples compared
  by token id between the fast and plain paths, with the result stated against
  the plain path's own noise.
- **Depth is measured per machine.** The app benchmarks the host and picks the
  decoding depth, and every published row carries a version, a machine and a
  date, with raw logs where they exist.
- Targets small memory: a 27B on a 16 GB Mac.

Limits that matter for comparison:

- Apple silicon only.
- Requires a model that ships MTP heads; the list is Qwen 3.5/3.6/3.8,
  Flash-Next, Ternary Bonsai 2 and Gemma 4.
- Image requests moved to the compiled verifier without the sampled comparison
  being repeated, which the project discloses.
- A Python/MLX runtime rather than a compiled engine.

## Cross-platform engines

Engines built for many vendors at once, where any single device is one
configuration among many. Before crediting a fast kernel to a new approach,
check what one of these already achieves on the same device.

### llama.cpp

[`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) — C/C++ over
[ggml](https://github.com/ggml-org/ggml), MIT.

The project every other engine here is either a fork of or a reaction to. It
carries 1.5- to 8-bit integer quantization, `llama-server` with an
OpenAI-compatible API and a built-in web UI, GBNF grammars, multimodal input,
and CPU+GPU hybrid inference that splits a model larger than VRAM.

Strengths:

- **Backend coverage nothing else matches**: BLAS, CANN, CUDA, HIP, Hexagon,
  IBM zDNN, MUSA, Metal, OpenCL, RPC, SYCL, Vulkan, WebGPU, ZenDNN, and tuned
  CPU paths for x86 (AVX/AVX-512/AMX), ARM (NEON), and RISC-V.
- **GGUF is defined here.** Model and quantization support lands upstream
  first, so an unusual architecture or a new quant type usually runs here
  before anywhere else.
- **Distribution.** Prebuilt binaries, Docker, a curl installer, and packages
  in every ecosystem that wraps an inference engine.

Limits that matter for comparison:

- Kernels are written for a large device population, so per-device tuning is
  generic by construction. Device-specific work shows up in forks like
  strix-llama.cpp long before — or instead of — landing upstream.
- The AMD HIP path has device-specific failure modes. halo-box's README
  documents an async-execution correctness bug on `gfx1151` where batched
  inference returns badly wrong output unless `HIP_LAUNCH_BLOCKING=1`
  serializes launches.

### Strata

[`Niko1221/Strata`](https://github.com/Niko1221/Strata) — C++, MIT. Runs
Qwen3.8-Flash-Next (125B MoE) on consumer PCs.

An engine plus a desktop app for one large MoE model on hardware that was never
meant to hold it: the hottest experts stay in VRAM, every expert lives in RAM,
and a large lookup table is read from the SSD. It ships as per-backend
installers for Windows, Linux, and macOS alongside a `strata` CLI. Its kernels
build on ggml/llama.cpp's block layouts, codebooks, and MMQ scheduler.
Xolotl, under NVIDIA-focused engines above, is its fork.

Strengths:

- **Hardware breadth that is unusual at this model size**: NVIDIA RTX 20–50,
  AMD RX 6800 and newer plus Radeon AI PRO R9700 and RX 9000, Intel Arc, Apple
  Silicon, and Strix Halo or other unified-memory systems, across Windows,
  Linux, and macOS. Two or three cards of the same make can share a model.
- **One install story per backend.** The app picks an engine for the detected
  card, so a reader needs no toolchain; `strata doctor`, `strata monitor`, and
  `strata update` cover the operational side.
- **Agent setup as a first-class feature.** `strata connect` prints or writes
  the exact settings for OpenCode, Codex CLI, Claude Code, and other harnesses,
  and an MCP server lets an agent install, start, and stop the engine.
- **Three client dialects on one API surface**: OpenAI Chat Completions,
  OpenAI Responses, and Anthropic Messages.
- **A shareable measurement harness.** `BENCH.bat` and `bench.sh` produce a
  report designed to be posted, which is how its community results table is
  built.
- MIT, no account, no telemetry, and nothing leaves the machine except the
  model download and an optional web search.

Limits that matter for comparison:

- Serves one request at a time by default. `--parallel N` exists, but on a small
  card each answer gets slower.
- The model surface is curated: Qwen3.8-Flash-Next in several sizes,
  Qwen3.6-35B-A3B, and DeepSeek V4.1-Flash from the terminal only.
- It is an application rather than a library; the documented integration
  surface is HTTP.
- Large downloads and a first start that pages tens of GB into RAM, which can
  make the machine unresponsive for minutes.

### ds4 (DwarfStar)

[`antirez/ds4`](https://github.com/antirez/ds4) — C, MIT. Metal, CUDA, and
ROCm.

A deliberately narrow engine: it runs a short list of large models — DeepSeek
V4 Flash, V4.1 Flash, and V4 PRO, GLM 5.2, 5.3, and 5.3 Flash, and Qwen3.8
Flash Next — and it loads only the GGUF files the project itself produces. It
does not link against ggml, but keeps the GGML authors' copyright notice
because it retains or adapts MIT-licensed quant layouts, tables, CPU quant and
dot logic, and some kernels.

Strengths:

- **Consumer hardware is the target, not a fallback.** A 64–128 GB Mac, a DGX
  Spark, a Strix Halo Framework Desktop, or a set of Ada Lovelace cards.
- **SSD streaming as a supported mode** for models larger than RAM, with Engram
  tables kept on disk in every mode rather than paged in.
- **Multi-machine and multi-GPU options most local engines do not offer**:
  tensor parallelism over RDMA between two 128 GB Macs, pipeline parallelism to
  sum RAM across systems, and multi-GPU CUDA including Ada and L40S, which
  newer-model implementations usually exclude.
- **A native coding agent beside the server.** `ds4-agent` runs inference
  in-process with live model state and the model's own tool format;
  `ds4-server` serves Pi, OpenCode, Codex CLI, and Claude Code, with session KV
  snapshots on disk that avoid rebuilding a prompt.
- **Per-model extras that actually ship**: DSpark and MTP speculation with an
  exact-sampling mode, direction steering, thinking levels, and a `--power`
  dial that trades throughput for sustained GPU load.
- **An embedded capability harness.** `ds4-eval` runs regression suites against
  a real GGUF, so a model and engine change is checked together rather than
  separately.
- **Two disclosures stated up front**: the code is developed with heavy AI-agent
  assistance under human direction, and the project is beta with fast-moving
  behavior.

Limits that matter for comparison:

- Not a general GGUF runner. A model outside its list, or a GGUF it did not
  produce, is not the use case.
- Model support is opportunistic by design; a model can be dropped when a
  better one arrives.
- Backend coverage is uneven across models. V4.1 Flash runs on Metal for text
  and vision, with text also on a DGX Spark, and Qwen3.8 Flash Next is Metal and
  CUDA only.
- The ROCm build is the Strix Halo path (`make strix-halo`). Discrete AMD cards
  are not listed as a target.
- Beta quality, changing quickly, with regressions possible between releases.

### vLLM

[`vllm-project/vllm`](https://github.com/vllm-project/vllm) ·
[docs.vllm.ai](https://docs.vllm.ai/) — Apache-2.0, Python with CUDA and HIP
kernels; NVIDIA, AMD, Intel GPUs and CPUs.

The serving baseline. Most of the vocabulary on this page — paged KV, continuous
batching, prefix caching — either comes from vLLM or was popularized by it.

Strengths:

- **Throughput at scale is the target.** PagedAttention, continuous batching,
  chunked prefill, prefix caching, and piecewise or full CUDA/HIP graphs.
- **Every parallelism axis, plus disaggregation**: tensor, pipeline, data,
  expert and context parallelism, with prefill, decode and encode as separate
  workers.
- **Kernel and quantization breadth**: FlashAttention, FlashInfer, TRTLLM-GEN,
  FlashMLA and Triton attention; CUTLASS, TRTLLM-GEN and CuTeDSL GEMM and MoE;
  FP8, MXFP8, MXFP4, NVFP4, INT8, INT4, GPTQ, AWQ, GGUF, compressed-tensors,
  ModelOpt and TorchAO; speculation through n-gram, suffix, EAGLE and DFlash.
- **A real plugin boundary** for hardware and quantization, which is why the
  RDNA4 forks above exist at all — the alternative is a private tree.
- 200+ architectures, multi-LoRA, structured outputs, and OpenAI, Anthropic and
  gRPC surfaces.

Limits that matter for comparison:

- PyTorch is on the hot path. That is what buys its model breadth, and it is the
  cost a torch-free engine trades away.
- Batch-1 decode is not where it wins; a hand-written single-model engine beats
  it there, which is the premise of the NVIDIA-focused group.
- Consumer RDNA support arrives through forks and plugins rather than upstream.
- Published throughput is configuration-dependent enough that a row is not
  comparable without the exact launch flags.

### SGLang

[`sgl-project/sglang`](https://github.com/sgl-project/sglang) ·
[docs.sglang.io](https://docs.sglang.io/) — Apache-2.0, Python; NVIDIA, AMD
Instinct, Google TPU, Intel, Apple Silicon, Ascend and Moore Threads.

A serving framework built around prefix reuse as the central abstraction, with a
frontend language for constrained generation.

Strengths:

- **The cache is the design.** RadixAttention keeps KV in a radix tree over token
  sequences, which is why shared system prompts and branching tool calls across
  agent sessions are served well rather than as a special case.
- **The program and the scheduler are one artifact.** Structured output,
  branching and control flow are expressed in its frontend instead of bolted on
  as a post-processor.
- **Large-scale and RL workloads**: tensor, expert and data parallelism across
  hosts, disaggregated serving, and rollouts treated as a first-class use.
- Hardware breadth beyond the two GPU vendors, including Apple Silicon through
  Metal/MLX, and image and video diffusion in the same package.

Limits that matter for comparison:

- PyTorch runtime, like vLLM.
- Optimized for many concurrent requests, not a single-user decode path.
- Upstream AMD coverage is Instinct-first; consumer RDNA is a fork story.
- The frontend is an adoption cost: you serve through its programming model
  rather than a plain OpenAI-compatible endpoint.

### ExLlamaV3

[`turboderp-org/exllamav3`](https://github.com/turboderp-org/exllamav3) — MIT,
Python with compiled CUDA extensions, NVIDIA consumer GPUs. The recommended
server is [`theroyallab/tabbyAPI`](https://github.com/theroyallab/tabbyAPI/).

The quantization-and-inference library the EXL3 format comes from, and the
quality-per-bit reference that the IQ/GSQ and NVFP4 families are argued against.

Strengths:

- **EXL3, based on QTIP, plus 2–8 bit KV cache quantization**, with the
  quality-versus-bits curve published rather than asserted.
- **Parallelism for consumer hardware**: flexible tensor and expert parallelism,
  and CPU offload with AVX2/AVX512 so a large MoE fits a small card — the same
  problem the Flash-Next engines above solve, approached from the quantization
  side.
- **Continuous dynamic batching, speculative decoding and multimodal support**
  in the library, with a Transformers plugin for embedding it in other code.
- Prebuilt wheels per CUDA and Torch version, which is unusual for a project
  this extension-heavy, plus a documented JIT path.

Limits that matter for comparison:

- CUDA only upstream. ROCm appears downstream (Kyojin, above) rather than as a
  supported target.
- Torch is a hard dependency with a version matrix to satisfy.
- Linux and Windows x86_64; no Apple or CPU-only path.
- The library is not a server, and TabbyAPI has its own release cadence and
  configuration surface.

### TensorFold

[`ashhart/TensorFold`](https://github.com/ashhart/TensorFold) ·
[tensorfold.dev](https://tensorfold.dev/) — Apache-2.0, a Zig binary with Metal
and CUDA backends; Apple silicon and NVIDIA, including a GB10 on aarch64.

Reads checkpoints, tokenizes requests and runs kernels directly, with no Python
or MLX at runtime.

Strengths:

- **One engine binary across two vendor backends**, which is a rare shape here:
  most cross-vendor projects on this page are Python frameworks, and most
  compiled engines pick one vendor.
- **Capacity is reported rather than assumed.** `--parallel auto` asks for up to
  eight streams and refuses an explicit count it cannot fit; startup prints the
  accepted number, and `/health` separates active from waiting requests.
- **Contributor process as part of the product**: an agent traffic-control
  protocol to prevent duplicate work, a contributor skill for coding agents, and
  per-platform runbooks including a native Windows build guide.
- A desktop workspace app on Apple silicon that drives the same binary.

Limits that matter for comparison:

- A small model surface relative to llama.cpp, served from pinned revisions.
- The CUDA path needs glibc 2.28+ and a driver-provided `libcuda.so.1`; Windows
  needs its own build guide rather than the release archive.
- "Living Weights" is experimental and should not be read as a stable feature.
- An early-stage tree that moves quickly.

### kvmem

[`kvmem/kvmem`](https://github.com/kvmem/kvmem) — no license file in the
repository. A KV-memory layer over llama.cpp and NInfer rather than an engine of
its own.

Keeps native KV history in host RAM with optional SSD tiering, and brings back
only the history a query selects, inside a bounded GPU working set.

Strengths:

- **Eviction is a policy over spans, not a truncation.** Selected history is
  restored at its original logical positions, which is the same requirement
  hipEngine's `KVLiveSpans` ABI encodes; kvmem is external evidence that the
  spans need not be dense.
- **Multi-turn and tool state are the contract.** Exact-prefix and checkpoint
  rules are honored per backend, so a tool continuation, edit or cancellation
  restores state the way that backend would have.
- **Backend-agnostic by construction**: one memory, retrieval and storage
  contract with separate llama.cpp and NInfer integrations, pinned in a backend
  manifest with patches and provenance.
- Packaged Windows binaries from `sm70` through `sm120a`, with an embedded Web
  UI.

Limits that matter for comparison:

- Not a serving engine; you still run llama.cpp or NInfer underneath it.
- Published binaries are Windows NInfer only. The Linux, ROCm and llama.cpp
  paths are source builds.
- Without a license file, the terms for reuse are unclear.
- Rebuilding from `master` changes the recall algorithm relative to the released
  binaries, so a source build is not the released behavior.

## Multi-GPU and mixed generations

Several engines spread one model across cards. The mechanism decides whether the
cards may differ: a **layer split** (pipeline) gives each card a contiguous block
of layers, so unequal cards work and each keeps its own expert cache; **tensor
parallel** splits every layer, so it needs symmetric shards and in practice
identical cards. *Mixed generation* here means cards of different compute
capabilities — `sm_86` Ampere beside `sm_120` Blackwell — in one run. Two cards
of the same architecture but different models (a 5090 and a 5060 Ti) are not
that case, and several tensor-parallel engines accept the pair.

The AMD entries have their own multi-card paths: Strata's HIP build shares a
model among AMD cards, and Lucebox's headline configuration is an R9700 plus a
Strix Halo iGPU.

| Engine | How it spreads a model | Mixed NVIDIA generations |
| --- | --- | --- |
| llama.cpp | Layer split by default (`--split-mode layer`), with `-ts` proportions per card; `tensor` is experimental | Yes, if the build carries kernels for both architectures. A native build compiles for the cards present at build time; otherwise set `CMAKE_CUDA_ARCHITECTURES`. |
| Strata | Layer split with Main and Helper roles; the faster card runs the last stage | Yes. Its installer recommends an RTX 5080 + RTX 3090 pair, and orders cards by measured speed rather than by model. |
| Xolotl | The same layer split, with a per-GPU expert cache, an optional head device and `--remote-device` | Yes, with the same recommendation: newest generation as Main, older card as Helper. |
| ds4 | Complete layers per device with a per-device VRAM budget (`--gpu-vram auto` or explicit GiB per device); Flash tensor parallelism pairs cards | Not stated. Per-device budgets and a native multi-architecture build would allow it; the tested configuration is eight L40S. |
| Lucebox | Layer split within a backend; a CUDA and a HIP process can split one model through host IPC | Not documented for two NVIDIA cards. Cross-vendor CUDA + HIP is the documented heterogeneous mode. |
| Basalt | One main card plus an optional second card as a peer expert tier (`"gpu": "0,1"`) | No. Both cards must be `sm_120`, and Windows runs one card only. |
| CapyFlash | Tensor parallel across two cards, experts split between them | Same architecture only (`sm_120`). |
| vLLM | Tensor, pipeline, expert, data and context parallelism | No documented support; the parallelism docs assume a node of identical GPUs. |
| SGLang | Tensor, expert and data parallelism, including across hosts | No. Mixed-architecture nodes are untested, and a reported L40S + 4090 attempt failed inside a Triton kernel on shared-memory limits. |
| ExLlamaV3 | Tensor and expert parallelism | Not documented. |

The remaining engines drive one device at a time: NInfer (its TP2 forks add a
two-card path), Infernix, MegaCapybara, SparkInfer, tqLLM, fngine,
FastLocalExperts, lean-qwenflash, Decapybara, Kyojin, and the two Apple-silicon
engines.

Two constraints apply across the table. The build must contain kernels for every
architecture in the machine — a fat binary covering both, not just the card that
was present at compile time. And a layer split distributes by free memory, so
the smaller card still has to hold the layers it is given; the engines that
recommend pairs enforce a floor, and Strata and Xolotl warn below 8 GB.

## Adding an engine

One section per engine, plus a row in the at-a-glance table. Put it under
**AMD-focused** or **NVIDIA-focused** if it targets a named part from one
vendor, under **Apple-silicon** if it targets Macs, and under **cross-platform**
if it treats hardware as one configuration among many. A useful entry records:

- repository URL and, if it has one, the project site;
- the hardware it targets, named precisely enough that a reader knows whether
  their machine is in scope;
- the model format it loads, and whether that format is shared or its own;
- how it is distributed (source, package, container, binary) and its license;
- two or three things it is genuinely better at than the alternatives;
- the limits that would distort a comparison — gated routes, a narrow model
  surface, a closed kernel tree, a fork that has drifted from its parent.

Do not put throughput numbers here. If a row is worth reporting, it belongs in
[benchmarks/README.md](../../benchmarks/README.md) with its provenance, not in a
landscape page where it will go stale and lose its method.
