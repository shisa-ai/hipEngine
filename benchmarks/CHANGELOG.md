# HIPENGINE Benchmark Changelog

Reverse-chronological human-readable history for benchmark rollup changes. Keep
entries short; detailed evidence belongs in `benchmarks/results/*.json` and
`WORKLOG.md`.

Entry format:

```text
- [scope] model / quant / workload: metric old -> new (+/-X%) due to reason/change; artifact/source.
```

Examples:

```text
- [HIPENGINE] Qwen3-0.6B / fp16 / 4K/4K: decode 80.0 -> 88.0 tok/s (+10.0%) due to paged-attn split-K; `benchmarks/results/2026-..json`.
- [lineage target] Qwen3.5-PARO / w4a16 / 512/128: prefill 1300 -> 2557 tok/s (+96.7%) due to compact WMMA; `~/amd-gpu-tuning/docs/OPTIMAL.md`.
```

## 2026-05-13

- [rollup] Added initial `benchmarks/README.md` scoreboard; no accepted HIPENGINE E2E inference rows yet.
- [lineage target] Qwen3.5-35B-A3B-PARO / w4a16 / 512-128K sweeps: recorded compact-WMMA + graph-replay target rows from `~/amd-gpu-tuning/docs/OPTIMAL.md`.
- [external baseline] Added llama.cpp ROCm and Qwen3-0.6B host-architecture comparison baselines from `docs/BENCHMARK.md` / `~/amd-gpu-tuning/WORKLOG.md`.
- [smoke] Recorded `smoke_add` as a build/runtime correctness smoke only, not a throughput row.
