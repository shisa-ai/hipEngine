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

## 2026-05-15

- [correctness] HIPENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / c=1 512/32 parent fixture: generated-token equality blocked -> accepted after parent-mixed MoE parity fixes; `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`.
- [correctness] HIPENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / c=2/4/8 generated equality: blocked -> accepted for scheduler-backed serial bridge with finite logits and graph/occupancy metadata; `benchmarks/results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`.
- [rollup] HIPENGINE Qwen3.5/PARO correctness rows: no retained throughput row -> accepted non-throughput c=1 and c=N correctness gates; `benchmarks/README.md` smoke/non-throughput table.

## 2026-05-13

- [lineage measured] Qwen3.5-35B-A3B-PARO / w4a16 / 512/128: prefill 2557 -> 2696.4 tok/s (+5.5%), decode 115.7 -> 116.05 tok/s (+0.3%) from local OPTIMAL parent rerun; `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`.
- [lineage measured] Qwen3.5-35B-A3B-PARO / w4a16 / 4K/128: prefill 2703 -> 2741.5 tok/s (+1.4%), decode 112.0 -> 113.05 tok/s (+0.9%) from local OPTIMAL parent rerun; `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`.
- [blocked] HIPENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / OPTIMAL parity: no accepted row -> blocked due to missing `LLM.generate`, `w4_paro` layout, model plugin, and dependent kernels; `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json`.
- [rollup] Added initial `benchmarks/README.md` scoreboard; no accepted HIPENGINE E2E inference rows yet.
- [lineage target] Qwen3.5-35B-A3B-PARO / w4a16 / 512-128K sweeps: recorded compact-WMMA + graph-replay target rows from `~/amd-gpu-tuning/docs/OPTIMAL.md`.
- [external baseline] Added llama.cpp ROCm and Qwen3-0.6B host-architecture comparison baselines from `docs/BENCHMARK.md` / `~/amd-gpu-tuning/WORKLOG.md`.
- [smoke] Recorded `smoke_add` as a build/runtime correctness smoke only, not a throughput row.
