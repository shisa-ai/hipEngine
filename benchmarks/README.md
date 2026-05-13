# HIPENGINE Benchmark Rollup

Last updated: 2026-05-13

Human-readable scoreboard for HIPENGINE performance. Machine-readable benchmark
attempts live under [`benchmarks/results/`](results/); this file tracks the
current fastest accepted rows, external baselines, and source-lineage targets so
humans can see what we are testing against without opening every JSON artifact.

Historical rollup changes are tracked in [`benchmarks/CHANGELOG.md`](CHANGELOG.md).

## Maintenance contract

Update this file whenever a benchmark artifact is retained or a comparison
baseline changes:

1. Update the `Last updated` line at the top.
2. Add or replace the row for the relevant `(model, quant, backend, workload,
   policy)` tuple.
3. Link the compact JSON artifact in `benchmarks/results/` when the row is a
   HIPENGINE measurement.
4. Include correctness status, memory, command/source, and the date the row was
   last refreshed.
5. Add a short reverse-chronological one-liner to `benchmarks/CHANGELOG.md` in the form: model / quant / workload, metric `old -> new`, percent delta, reason/change, and artifact/source.
6. Keep blocked/rejected attempts in JSON artifacts and `WORKLOG.md`; do not put
   them in "current fastest" tables unless clearly marked as blocked/rejected.

A row is not retained unless it satisfies `docs/BENCHMARK.md`: exact command,
hardware/software context, workload shape, correctness gate, repeated-run stats
where applicable, and post-run quality gates.

## Current fastest HIPENGINE rows

No HIPENGINE end-to-end inference benchmark has been accepted yet. `smoke_add`
only proves build/runtime correctness and is not a throughput row.

| Model | Quant | Backend | Workload | Prefill tok/s | Decode tok/s | Peak GiB | Correctness | Artifact | Last updated | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | 2026-05-13 | Await first accepted `LLM.generate()` benchmark. |

## Source-lineage target: Qwen3.5-35B-A3B-PARO

These rows are **not HIPENGINE measurements**. They are the current parent
source-lineage target from `~/amd-gpu-tuning/docs/OPTIMAL.md` that HIPENGINE's
Qwen3.5/PARO port should reproduce or beat. Hardware: W7900/gfx1100. Engine:
`nano-vllm-amd` PARO native c=1. Path: compact-WMMA prefill plus one-step
graph-replay decode, all parent-listed quality gates passing.

| Model | Quant | Backend/source | Workload | Prefill tok/s | Decode tok/s | Peak GiB | Validation | Source | Last updated |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Qwen3.5-35B-A3B-PARO | w4a16 AWQ/PARO | `nano-vllm-amd` parent | 512/128 | 2696.4 | 116.05 | 18.80 | graph/step true | [`2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`](results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json) | 2026-05-13 |
| Qwen3.5-35B-A3B-PARO | w4a16 AWQ/PARO | `nano-vllm-amd` parent | 1K/128 | 2876 | 112.9 | 19.34 | graph/step true | `~/amd-gpu-tuning/docs/OPTIMAL.md` | 2026-05-13 |
| Qwen3.5-35B-A3B-PARO | w4a16 AWQ/PARO | `nano-vllm-amd` parent | 4K/128 | 2741.5 | 113.05 | 21.64 | graph/step true | [`2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`](results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json) | 2026-05-13 |
| Qwen3.5-35B-A3B-PARO | w4a16 AWQ/PARO | `nano-vllm-amd` parent | 32K/128 | 1880 | 98.8 | 21.37 | graph/step true | `~/amd-gpu-tuning/docs/OPTIMAL.md` | 2026-05-13 |
| Qwen3.5-35B-A3B-PARO | w4a16 AWQ/PARO | `nano-vllm-amd` parent | 128K/128 | 914 | 62.6 | 27.42 | graph/step true | `~/amd-gpu-tuning/docs/OPTIMAL.md` | 2026-05-13 |

## External comparison baselines

Rows below are comparison targets, not HIPENGINE results. They stay here so a
future HIPENGINE result can be interpreted quickly.

### llama.cpp ROCm / Qwen3.6-35B-A3B Q8_K_XL

Source: `~/amd-gpu-tuning/WORKLOG.md` 2026-04-28 entry and
`docs/BENCHMARK.md` baseline table.

| Model | Quant | Backend | Workload | Prefill tok/s | Decode tok/s | VRAM / memory | Source | Last updated | Notes |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- |
| Qwen3.6-35B-A3B | Q8_K_XL GGUF | llama.cpp ROCm | pp512/tg128 | 949.89 ± 9.59 | 74.32 ± 0.02 | — | `~/amd-gpu-tuning/WORKLOG.md` | 2026-04-28 | `llama-bench`, `-fa 1`. |
| Qwen3.6-35B-A3B | Q8_K_XL GGUF | llama.cpp ROCm server | 4K/4K | 1139.72 | 71.49 | 44.94 GiB used | `~/amd-gpu-tuning/WORKLOG.md` | 2026-04-28 | `/completion`, temp 0, `ignore_eos=true`. |

### Host-architecture comparator: Qwen3-0.6B FP16 c=1

Source: `~/amd-gpu-tuning/WORKLOG.md` 2026-04-28 shootout entry and
`docs/BENCHMARK.md` baseline table.

| Model | Quant | Engine/backend | Workload | Prefill tok/s | Decode tok/s | KV GiB | Source | Last updated | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Qwen3-0.6B | FP16 | nano-vllm / ROCm SDPA | 4K/4K | 30167.12 | 15.33 | 38.39 | `~/amd-gpu-tuning/WORKLOG.md` | 2026-04-28 | Reference for host overhead to beat. |
| Qwen3-0.6B | FP16 | mini-sglang / torch SDPA | 4K/4K | 20195.46 | 22.58 | 39.10 | `~/amd-gpu-tuning/WORKLOG.md` | 2026-04-28 | Reference for host overhead to beat. |

## Smoke / non-throughput rows

| Check | Backend | Command | Result | Artifact | Last updated | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| `smoke_add` HIP runtime/build | `hip_gfx1100` | `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` | `max_abs=0.0` | `~/.cache/hipengine/build/smoke-101db2a5ad5526c3/smoke_add.so` | 2026-05-13 | Correctness/build smoke only; no throughput claim. |

## Table conventions

- Workload format is `prompt_tokens/decode_tokens` unless otherwise stated.
- `Peak GiB` means peak allocated/reserved as emitted by the benchmark; note the
  exact field in the linked artifact when ambiguous.
- `Validation` summarizes correctness quality gates; detailed KL/top-1 or
  fixture results belong in the JSON artifact.
- For parent/source-lineage rows, use the parent doc path as `Source` and keep
  them clearly separated from HIPENGINE measurements.
