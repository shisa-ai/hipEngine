# hipENGINE Benchmark Rollup

Last updated: 2026-05-15

Human-readable scoreboard for hipENGINE performance. Machine-readable benchmark
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
   hipENGINE measurement.
4. Include correctness status, memory, command/source, and the date the row was
   last refreshed.
5. Add a short reverse-chronological one-liner to `benchmarks/CHANGELOG.md` in the form: model / quant / workload, metric `old -> new`, percent delta, reason/change, and artifact/source.
6. Keep blocked/rejected attempts in JSON artifacts and `WORKLOG.md`; do not put
   them in "current fastest" tables unless clearly marked as blocked/rejected.

A row is not retained unless it satisfies `docs/BENCHMARK.md`: exact command,
hardware/software context, workload shape, correctness gate, repeated-run stats
where applicable, and post-run quality gates.

## Current fastest hipENGINE rows

No hipENGINE end-to-end throughput benchmark has been accepted yet. Accepted
correctness-only gates are recorded under "Smoke / non-throughput rows"; their
timings are diagnostic context only and are not retained as performance rows.

| Model | Quant | Backend | Workload | Prefill tok/s | Decode tok/s | Peak GiB | Correctness | Artifact | Last updated | Notes |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| — | — | — | — | — | — | — | — | — | 2026-05-15 | Await first accepted `LLM.generate()` throughput benchmark. |

## Source-lineage target: Qwen3.5-35B-A3B-PARO

These rows are **not hipENGINE measurements**. They are the current parent
source-lineage target from `~/amd-gpu-tuning/docs/OPTIMAL.md` that hipENGINE's
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

Rows below are comparison targets, not hipENGINE results. They stay here so a
future hipENGINE result can be interpreted quickly.

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
| Qwen3.5/PARO native single-request prefill fixture gate | `hip_gfx1100` | `python3 scripts/qwen35_native_prefill_fixture_gate.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json` | `passed=true`, `max_kl=0.0168`, `top1=100%`, fixture IDs match | [`2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`](results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json) | 2026-05-15 | Correctness gate only; native prefill timing is diagnostic and not a promoted throughput row. |
| Qwen3.5/PARO segment-aware linear-attn prefill kernels | `hip_gfx1100` | `python3 scripts/smoke.py --mode qwen35-linear-attn-segments-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` | conv/state max abs `1.86e-09/0`, GDN/state max abs `1.86e-09/9.31e-10`; profiler confirms segment kernels launched | [`2026-05-15-hipengine-qwen35-linear-attn-segment-prefill-accepted.json`](results/2026-05-15-hipengine-qwen35-linear-attn-segment-prefill-accepted.json) | 2026-05-15 | Kernel correctness/profiler smoke only; no throughput claim. |
| Qwen3.5/PARO varlen full-attn prefill kernel | `hip_gfx1100` | `python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` | `varlen_prefill_gate_fp16_max_abs=0`, mismatch `0`; profiler confirms prompt KV writer + varlen prefill attention launched | [`2026-05-15-hipengine-qwen35-varlen-full-attn-prefill-accepted.json`](results/2026-05-15-hipengine-qwen35-varlen-full-attn-prefill-accepted.json) | 2026-05-15 | Kernel correctness/profiler smoke only; no throughput claim. |
| Qwen3.5/PARO c=1 parent fixture equality | `hip_gfx1100` | `python3 scripts/qwen35_e2e_correctness.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-new-tokens 32 --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json` | `passed=true`, `expected_match=true`, generated `[1739, 220, 16, 15, …]` | [`2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`](results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json) | 2026-05-15 | Correctness fixture only; timings are diagnostic, no throughput claim. |
| Qwen3.5/PARO c=N generated equality | `hip_gfx1100` | `python3 scripts/qwen35_batch_serial_correctness.py --scheduler ...` for c=2/4/8; exact commands in artifact | c=2/4/8 `finite_logits=true`, `generated_match=true`, `passed=true` | [`2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`](results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json) | 2026-05-15 | Correctness gate only; current c>N bridge is serial, not a native compact/c-aware throughput path. |
| `smoke_add` HIP runtime/build | `hip_gfx1100` | `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` | `max_abs=0.0` | `~/.cache/hipengine/build/smoke-101db2a5ad5526c3/smoke_add.so` | 2026-05-13 | Correctness/build smoke only; no throughput claim. |

## Blocked / diagnostic benchmark attempts

These rows are **not** current-fastest hipENGINE results. They are committed so
we do not lose exact commands, hardware/software context, correctness status,
and blocker evidence for attempted shapes. Their timing fields are diagnostic
only unless a future artifact has `status="accepted"` and `performance_claim=true`.

| Model | Quant | Workload | Path | Correctness / status | Diagnostic timing | Memory | Artifact | Last updated | Blocker / notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=8, prompt 8 first compact slab | `native_prefill_compact_cN_plan` | `status=blocked`, slab metadata valid, `performance_claim=false` | no packed prefill launched; metadata only | n/a | [`2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json`](results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json) | 2026-05-15 | `CompactPromptSlab`, `bucketize_by_block_count`, segment-aware linear-attn, and varlen full-attn kernels landed; blocked on final packed state commit/orchestration. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1, prompt 512 / decode 32, max_layers=40 | `single_request_native_full` | `status=accepted`, `max_kl=0.0168`, `top1=100%`, `performance_claim=false` | fixture native prefill 45.72 tok/s; repeated-token bench prefill 46.96 tok/s, decode 101.61 tok/s | owned device 1.51 GiB (diagnostic accounting) | [`2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`](results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json) | 2026-05-15 | Correctness accepted, but slower than serial c=1 fixture prefill 117.24 tok/s and parent 2682.66 tok/s; no throughput row promoted. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1, prompt 8 / decode 1, max_layers=40 | `scheduler_serial_slot_bridge` | `status=blocked`, `finite_logits=true`, `performance_claim=false` | prefill 90.464 tok/s; aggregate decode 106.765 tok/s; per-request decode 106.765 tok/s | peak allocator n/a; max batch 1, max sequence 10 | [`2026-05-15-hipengine-qwen35-c1-scheduler-serial-bench-blocked.json`](results/2026-05-15-hipengine-qwen35-c1-scheduler-serial-bench-blocked.json) | 2026-05-15 | Reduced diagnostic shape and serial row execution; not the c=N 512/128 retained protocol. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=2, prompt 8 / decode 1, max_layers=40 | `scheduler_serial_slot_bridge` | `status=blocked`, `finite_logits=true`, `performance_claim=false` | prefill 103.567 tok/s; aggregate decode 107.149 tok/s; per-request decode 53.575 tok/s | peak allocator n/a; max batch 2, max sequence 10 | [`2026-05-15-hipengine-qwen35-c2-scheduler-serial-bench-blocked.json`](results/2026-05-15-hipengine-qwen35-c2-scheduler-serial-bench-blocked.json) | 2026-05-15 | `batch_execution.throughput_claim_eligible=false`; native compact/c-aware c>N kernels remain Task #15. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=4, prompt 8 / decode 1, max_layers=40 | `scheduler_serial_slot_bridge` | `status=blocked`, `finite_logits=true`, `performance_claim=false` | prefill 111.226 tok/s; aggregate decode 108.434 tok/s; per-request decode 27.108 tok/s | peak allocator n/a; max batch 4, max sequence 10 | [`2026-05-15-hipengine-qwen35-c4-scheduler-serial-bench-blocked.json`](results/2026-05-15-hipengine-qwen35-c4-scheduler-serial-bench-blocked.json) | 2026-05-15 | Aggregate decode stays flat because rows execute serially; do not compare as throughput win. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=8, prompt 8 / decode 1, max_layers=40 | `scheduler_serial_slot_bridge` | `status=blocked`, `finite_logits=true`, `performance_claim=false` | prefill 115.080 tok/s; aggregate decode 108.904 tok/s; per-request decode 13.613 tok/s | peak allocator n/a; max batch 8, max sequence 10 | [`2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json`](results/2026-05-15-hipengine-qwen35-c8-scheduler-serial-bench-blocked.json) | 2026-05-15 | Confirms serial bridge blocker: per-request decode falls with c while aggregate stays ~109 tok/s. |

## Table conventions

- Workload format is `prompt_tokens/decode_tokens` unless otherwise stated.
- `Peak GiB` means peak allocated/reserved as emitted by the benchmark; note the
  exact field in the linked artifact when ambiguous.
- `Validation` summarizes correctness quality gates; detailed KL/top-1 or
  fixture results belong in the JSON artifact.
- For parent/source-lineage rows, use the parent doc path as `Source` and keep
  them clearly separated from hipENGINE measurements.
