# hipENGINE Benchmark Changelog

Reverse-chronological human-readable history for benchmark rollup changes. Keep
entries short; detailed evidence belongs in `benchmarks/results/*.json` and
`WORKLOG.md`.

Entry format:

```text
- [scope] model / quant / workload: metric old -> new (+/-X%) due to reason/change; artifact/source.
```

Examples:

```text
- [hipENGINE] Qwen3-0.6B / fp16 / 4K/4K: decode 80.0 -> 88.0 tok/s (+10.0%) due to paged-attn split-K; `benchmarks/results/2026-..json`.
- [lineage target] Qwen3.5-PARO / w4a16 / 512/128: prefill 1300 -> 2557 tok/s (+96.7%) due to compact WMMA; `~/amd-gpu-tuning/docs/OPTIMAL.md`.
```

## 2026-05-15

- [diagnostic retained] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / native c=1 512/128 + 4K/128: 512 prefill 516.236 -> 529.985 tok/s (+2.7%) and 4K prefill 316.586 -> 322.359 tok/s (+1.8%) due to 64-thread transposed prefill projection GEMV; not promoted to current-fastest because `LLM.generate()` row/full sweep remain open; `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-multiloop-512-4k-diagnostic.json`.
- [correctness] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / native compact c=2/4/8 prompt8: generated equality blocked -> accepted with native compact prefill and serial decode; no throughput row because c-aware decode graph replay remains missing; `benchmarks/results/2026-05-15-hipengine-qwen35-c{2,4,8}-native-compact-prefill-correctness-accepted.json`.
- [blocked diagnostic] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / compact c=8 prompt8: retained throughput no row -> no row (blocked); physical slot metadata and final-row commit helpers landed, packed layer orchestration/equality gates remain; `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json`.
- [correctness] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / varlen full-attn prefill: kernel gate blocked -> accepted with max abs 0 and mismatch 0; no throughput row because this is a kernel smoke; `benchmarks/results/2026-05-15-hipengine-qwen35-varlen-full-attn-prefill-accepted.json`.
- [correctness] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / segment linear-attn prefill: kernel gate blocked -> accepted with conv/GDN segment max abs <= 1.86e-09; no throughput row because this is a kernel smoke; `benchmarks/results/2026-05-15-hipengine-qwen35-linear-attn-segment-prefill-accepted.json`.
- [blocked diagnostic] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / compact c=8 prompt8: retained throughput no row -> no row (blocked) after adding `CompactPromptSlab` metadata and `bucketize_by_block_count`; native packed execution still needs segment-aware linear-attn and varlen full-attn kernels; `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json`.
- [correctness] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / native c=1 512/32: native single-request prefill gate blocked -> accepted with max KL 0.0168 and top-1 100%, but perf row remains no row because native prefill is 45.72 tok/s vs serial 117.24 tok/s; `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`.
- [blocked diagnostic] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / native c=1 512/32: retained throughput no row -> no row (blocked) because `single_request_native_full` is correctness-clean but slower than serial and parent baselines; `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`.
- [blocked diagnostic] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / c=1/2/4/8 prompt8/decode1: retained throughput no row -> no row (blocked) because `scheduler_serial_slot_bridge` is serial and not c=N 512/128 protocol; artifacts `benchmarks/results/2026-05-15-hipengine-qwen35-c{1,2,4,8}-scheduler-serial-bench-blocked.json`.
- [correctness] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / c=1 512/32 parent fixture: generated-token equality blocked -> accepted after parent-mixed MoE parity fixes; `benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`.
- [correctness] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / c=2/4/8 generated equality: blocked -> accepted for scheduler-backed serial bridge with finite logits and graph/occupancy metadata; `benchmarks/results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`.
- [rollup] hipENGINE Qwen3.5/PARO correctness rows: no retained throughput row -> accepted non-throughput c=1 and c=N correctness gates; `benchmarks/README.md` smoke/non-throughput table.

## 2026-05-13

- [lineage measured] Qwen3.5-35B-A3B-PARO / w4a16 / 512/128: prefill 2557 -> 2696.4 tok/s (+5.5%), decode 115.7 -> 116.05 tok/s (+0.3%) from local OPTIMAL parent rerun; `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`.
- [lineage measured] Qwen3.5-35B-A3B-PARO / w4a16 / 4K/128: prefill 2703 -> 2741.5 tok/s (+1.4%), decode 112.0 -> 113.05 tok/s (+0.9%) from local OPTIMAL parent rerun; `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`.
- [blocked] hipENGINE / Qwen3.5-35B-A3B-PARO / w4_paro / OPTIMAL parity: no accepted row -> blocked due to missing `LLM.generate`, `w4_paro` layout, model plugin, and dependent kernels; `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json`.
- [rollup] Added initial `benchmarks/README.md` scoreboard; no accepted hipENGINE E2E inference rows yet.
- [lineage target] Qwen3.5-35B-A3B-PARO / w4a16 / 512-128K sweeps: recorded compact-WMMA + graph-replay target rows from `~/amd-gpu-tuning/docs/OPTIMAL.md`.
- [external baseline] Added llama.cpp ROCm and Qwen3-0.6B host-architecture comparison baselines from `docs/BENCHMARK.md` / `~/amd-gpu-tuning/WORKLOG.md`.
- [smoke] Recorded `smoke_add` as a build/runtime correctness smoke only, not a throughput row.
