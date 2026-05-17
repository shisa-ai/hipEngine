# Changelog

All notable user-facing changes for hipEngine releases are documented here.

This changelog is for package/API releases. Performance rollup history remains in
[`benchmarks/CHANGELOG.md`](benchmarks/CHANGELOG.md), with detailed benchmark
evidence under [`benchmarks/results/`](benchmarks/results/).

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
