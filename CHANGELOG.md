# Changelog

All notable user-facing changes for hipEngine releases are documented here.

This changelog is for package/API releases. Performance rollup history remains in
[`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md), with detailed benchmark
evidence under [`benchmarks/results/`](benchmarks/results/).

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
