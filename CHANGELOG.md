# Changelog

All notable user-facing changes for hipEngine releases are documented here.

This changelog is for package/API releases. Performance rollup history remains in
[`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md), with detailed benchmark
evidence under [`benchmarks/results/`](benchmarks/results/).

## Unreleased

### Added

- Added a top-level `hipengine` console command. `hipengine serve` launches the
  OpenAI-compatible server and `hipengine bench` lists/launches packaged
  benchmark helpers.

### Changed

- FastAPI/Uvicorn server dependencies now install by default because most users
  want the OpenAI-compatible API. The old `hipengine-server` console script has
  been replaced by `hipengine serve`.

### Fixed

- Missing Hugging Face repo IDs now report that the full model ID is absent from
  the local cache instead of falling through to a misleading partial-path
  `config.json` error.

## v0.2.2 - 2026-05-26

Patch release improving server startup context preallocation, KV memory
admission, and request defaults.

### Added

- Server-wide resident context/KV preallocation controls:
  `--max-context-tokens`, `--kv-storage`, `--kv-scale-dtype`, and
  `--kv-scale-granularity`. Eager startup prepares the resident PARO session for
  the configured context, and requests beyond that context or with a different
  KV policy are rejected instead of resizing/reloading the model.
- Automatic server context sizing when `--max-context-tokens` is omitted: after
  resident weights load, the runtime estimates the selected KV dtype plus
  persistent context metadata and preallocates
  `min(model_max_context_tokens, allocatable_context_tokens)`.
- Fast PARO retained-KV capacity estimate during resident session build. The
  runtime uses current `hipMemGetInfo` after model weights load to report the
  estimated max context for the selected KV dtype and for INT8 KV, warning when
  INT8 still falls below the model's advertised max context.

### Changed

- Chat requests that omit `max_tokens` now use `max_tokens=auto`, meaning the
  remaining admitted context (`max_context_tokens - prompt_tokens - 1`).

### Fixed

- Clean up partially-built PARO resident sessions if capacity preflight or
  allocation fails, avoiding leaked resident buffers on startup/admission OOM.

## v0.2.1 - 2026-05-25

Patch release improving server session management, streaming, and
OpenAI-compatible reasoning output.

### Added

- Eager model warmup on server startup: the configured model and a short
  warmup generation run before uvicorn reports ready, so the first real
  request does not pay load/compile cost. Controlled by `--eager-load` /
  `--no-eager-load` (default: on), `--eager-load-prompt`, and
  `--eager-load-max-tokens`, with `HIPENGINE_EAGER_LOAD`,
  `HIPENGINE_EAGER_LOAD_PROMPT`, and `HIPENGINE_EAGER_LOAD_MAX_TOKENS`
  environment variable equivalents.
- `LLM.stream()` method for single-prompt token-by-token generation when
  the underlying text generator supports it.
- Reasoning-content splitting for chat completions: `<think>…</think>`
  spans (Qwen/DeepSeek-style) are now separated into
  `message.reasoning_content` (non-streaming) or `delta.reasoning_content`
  chunks (streaming), matching the OpenAI reasoning-content convention.

### Changed

- PARO text generators and their resident sessions are now cached on the
  `LLM` instance and reused across requests. Session capacity is bucketed
  (floor 4 Ki tokens, configurable via `HIPENGINE_SESSION_MIN_TOKENS` and
  `HIPENGINE_SESSION_BUCKET_TOKENS`) so normal chat-history growth does not
  force reallocation every turn.
- Chat `stream=true` now yields token-level SSE chunks from the resident
  decode loop instead of buffering the full response and wrapping it in a
  single SSE frame.
- Chat completions default `max_tokens` raised from 16 to 8192 so clients
  that omit the field get usable reply lengths, including verbose
  chain-of-thought reasoning.

### Fixed

- Fixed `LLM.generate()` re-resolving the generation factory on every call,
  which discarded generator-local caches and caused the PARO resident
  session (layer weights, KV buffers) to be allocated and freed per request.

## v0.2.0 - 2026-05-25

Minor release for the GGUF runtime path and W7900 benchmark refresh. GGUF is a
meaningful new model-loading surface rather than a patch-level fix, so this
supersedes the previously planned v0.1.2 patch.

### Added

- Added Qwen3.6 35B MoE GGUF support for `Q4_K_M` and `Q4_K_S` model files,
  including resident GGUF loading, bulk prefill, graph-replay decode,
  decode-repacked T16 layouts, and WMMA/GEMV fast-path controls used by the
  W7900 benchmark profile.
- Added `docs/ENVS.md` as the canonical environment-variable reference, including
  TheRock ROCm process setup, cached-build profiling guidance, and safe GGUF
  benchmark profiles.
- Added a persistent README sweep harness that loads each hipEngine model once
  and runs repeated in-session workload measurements, matching llama-bench-style
  repetition without multiplying model load/decode-repack time by every shape.

### Changed

- Refreshed W7900 README performance tables with 5-run persistent-session medians
  for packed PARO and GGUF Q4_K_S while keeping the existing llama.cpp HIP/Vulkan
  comparison rows unchanged.
- Documented the current GGUF tradeoffs: higher one-time load cost and resident
  memory from decode-repack, Q4_K_S preferred for tighter VRAM budgets, and
  performance still behind PARO on some shapes while already competitive in the
  broader W7900 comparison.

### Fixed

- Fixed the PARO resident prefill workspace-overlap regression that shipped in
  v0.1.1: short and mid prompts now keep prefill workspaces resident through
  32K tokens, restoring 512/128-class prefill throughput while retaining the
  long-context memory-saving path for prompts above 32K when active chunking
  splits the prompt.
- Fixed GGUF non-split full-attention decode in max-context persistent sessions
  by launching the context kernel with the active decode context instead of the
  session's maximum allocation length.

### Known limitations

- GGUF support remains alpha: production correctness and performance coverage is
  strongest for the documented Qwen3.6 35B MoE Q4_K_M/Q4_K_S paths on gfx1100,
  and other GGUF quants/models require local validation.
- GGUF model load is slower than packed PARO on the same host because current
  decode-repack happens on load and is not yet cached on disk.

## v0.1.1 - 2026-05-19

Patch release focused on long-context memory documentation and the INT8 KV cache
bring-up that landed after v0.1.0.

### Added

- INT8 KV cache policy controls and dispatch coverage for Qwen/PARO resident
  inference paths, including CPU/layer/E2E correctness gates and memory audits.
- Documented Qwen3.6 packed PARO memory rows for 128K BF16 KV, 128K INT8 KV, and
  256K INT8 KV on W7900/gfx1100, with retained-KV and loaded-weight VRAM notes.

### Changed

- Reduced the 256K INT8 KV tracked allocator high-water mark below the 24 GiB
  class target by releasing/reusing prefill scratch and AOTriton query buffers.
- Clarified that packed vs unstripped PARO checkpoint size does not translate to
  meaningfully different resident model-weight VRAM for the current text runtime.

### Known limitations

- INT8 KV correctness is gated by deterministic fixtures and layer probes; it is
  not yet a long-rollout perplexity or compounding-error study.
- Qwen3.6 packed throughput rows remain diagnostic pending a promoted public
  `LLM.generate()` correctness/repetition gate.

## v0.1.0 - 2026-05-18

Initial public alpha release.

### Added

- Torch-free Python runtime hot path for local ROCm inference bring-up.
- Plugin registries keyed by model/backend/quant/layer variants.
- HIP backends for `gfx1100` and `gfx1151`, plus `backend="auto"` detection with
  `HIPENGINE_BACKEND` force override guidance for nearby targets.
- Qwen3.5/Qwen3.6 PARO W4 runtime path, JIT HIP build/cache plumbing, AOTriton
  prefill runtime packaging, and OpenAI-compatible server entry point.
- CPU reference kernels and focused correctness/performance documentation.

### Packaging

- PyPI project name: `hipengine`.
- Python import package: `hipengine`.
- Canonical repository/wordmark: `hipEngine`.
- Release wheels are Linux x86-64 `manylinux_2_39` platform wheels because the
  package bundles a ROCm/AOTriton shared-library runtime; ROCm runtime libraries
  remain external system dependencies.

### Known limitations

- Alpha-quality API and model coverage; expect sharp edges outside the documented
  Qwen/PARO paths.
- Default supported GPU targets are `gfx1100` and `gfx1151`; other AMD targets
  require explicit backend forcing and local validation.
- Model weights are not distributed with the package.
