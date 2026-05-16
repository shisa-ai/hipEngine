# hipENGINE Benchmark Rollup

Last updated: 2026-05-16

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
| Qwen3.5/PARO decode graph replay fixture gate | `hip_gfx1100` | `python3 scripts/qwen35_decode_graph_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512 --json /tmp/task20-decode-graph-fixture-gate.json` | `passed=true`, graph IDs match eager and fixture, final KL `0`, final top-1 match | [`2026-05-16-hipengine-qwen35-decode-graph-replay-diagnostic.json`](results/2026-05-16-hipengine-qwen35-decode-graph-replay-diagnostic.json) | 2026-05-16 | Correctness gate only; generated IDs are recorded on device inside replay. |
| Qwen3.5/PARO native single-request prefill fixture gate | `hip_gfx1100` | `python3 scripts/qwen35_native_prefill_fixture_gate.py --model /models/huggingface/hub/models--z-lab--Qwen3.5-35B-A3B-PARO/snapshots/dca2736e88e9f70855128fc81a8e918043a163cd --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json` | `passed=true`, `max_kl=0.0168`, `top1=100%`, fixture IDs match | [`2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json`](results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json) | 2026-05-15 | Correctness gate only; native prefill timing is diagnostic and not a promoted throughput row. |
| Qwen3.5/PARO segment-aware linear-attn prefill kernels | `hip_gfx1100` | `python3 scripts/smoke.py --mode qwen35-linear-attn-segments-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` | conv/state max abs `1.86e-09/0`, GDN/state max abs `1.86e-09/9.31e-10`; profiler confirms segment kernels launched | [`2026-05-15-hipengine-qwen35-linear-attn-segment-prefill-accepted.json`](results/2026-05-15-hipengine-qwen35-linear-attn-segment-prefill-accepted.json) | 2026-05-15 | Kernel correctness/profiler smoke only; no throughput claim. |
| Qwen3.5/PARO varlen full-attn prefill kernel | `hip_gfx1100` | `python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` | `varlen_prefill_gate_fp16_max_abs=0`, mismatch `0`; profiler confirms prompt KV writer + varlen prefill attention launched | [`2026-05-15-hipengine-qwen35-varlen-full-attn-prefill-accepted.json`](results/2026-05-15-hipengine-qwen35-varlen-full-attn-prefill-accepted.json) | 2026-05-15 | Kernel correctness/profiler smoke only; no throughput claim. |
| Qwen3.5/PARO c=1 parent fixture equality | `hip_gfx1100` | `python3 scripts/qwen35_e2e_correctness.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-new-tokens 32 --max-layers 40 --json benchmarks/results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json` | `passed=true`, `expected_match=true`, generated `[1739, 220, 16, 15, …]` | [`2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json`](results/2026-05-15-hipengine-qwen35-c1-parent-fixture-accepted.json) | 2026-05-15 | Correctness fixture only; timings are diagnostic, no throughput claim. |
| Qwen3.5/PARO native compact prefill c=N generated equality | `hip_gfx1100` | `python3 scripts/qwen35_batch_packed_prefill_correctness.py --prompt-length 8 --max-layers 40 --batch-size {2,4,8} --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached --json ...` | c=2/4/8 `finite_logits=true`, `generated_match=true`, `passed=true`; decode remains serial | [`c2`](results/2026-05-15-hipengine-qwen35-c2-native-compact-prefill-correctness-accepted.json), [`c4`](results/2026-05-15-hipengine-qwen35-c4-native-compact-prefill-correctness-accepted.json), [`c8`](results/2026-05-15-hipengine-qwen35-c8-native-compact-prefill-correctness-accepted.json) | 2026-05-15 | Correctness gate only; native compact prefill is wired, c-aware decode graph replay is still not a throughput path. |
| Qwen3.5/PARO c=N generated equality | `hip_gfx1100` | `python3 scripts/qwen35_batch_serial_correctness.py --scheduler ...` for c=2/4/8; exact commands in artifact | c=2/4/8 `finite_logits=true`, `generated_match=true`, `passed=true` | [`2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json`](results/2026-05-15-hipengine-qwen35-cn-generated-equality-accepted.json) | 2026-05-15 | Historical serial bridge correctness gate; superseded for prefill by native compact prefill equality above, decode remains serial. |
| `smoke_add` HIP runtime/build | `hip_gfx1100` | `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` | `max_abs=0.0` | `~/.cache/hipengine/build/smoke-101db2a5ad5526c3/smoke_add.so` | 2026-05-13 | Correctness/build smoke only; no throughput claim. |

## Blocked / diagnostic benchmark attempts

These rows are **not** current-fastest hipENGINE results. They are committed so
we do not lose exact commands, hardware/software context, correctness status,
and blocker evidence for attempted shapes. Their timing fields are diagnostic
only unless a future artifact has `status="accepted"` and `performance_claim=true`.

| Model | Quant | Workload | Path | Correctness / status | Diagnostic timing | Memory | Artifact | Last updated | Blocker / notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1 comparison checkpoint: 512/128, 4K/128, 32K/128, 128K/128, max_layers=40 | opt-in AOTriton threshold 512 + decode graph replay; chunked rows use linear/MoE/post/RoPE `1024`, full-attn query `4096` | `status=diagnostic_retained`, Task 26 fixture gates passed, `performance_claim=false`; resident-runner diagnostic, not public `LLM.generate()` | current/chunked: 512/128 `2216.487` prefill / `109.105` decode; 4K/128 `2504.959` / `110.117`; 32K/128 `1886.344` / `93.923`; 128K/128 `1002.409` / `61.051`; chunked vs unchunked: 4K `+5.7%`, 32K `+8.9%`, 128K `OOM -> 1002.409` | tracked peak 512 `18.581 GiB`, 4K `19.875 GiB`, 32K `20.688 GiB`, 128K `23.656 GiB`; vs nano parent docs: 512 prefill/decode `-13.3%/-5.7%`, 32K `+0.3%/-4.9%`, 128K `+9.7%/-2.5%` | [`chunking`](results/2026-05-16-hipengine-qwen35-prefill-chunking-diagnostic.json), [`comparison tables`](results/2026-05-16-hipengine-qwen35-comparison-tables-diagnostic.json) | 2026-05-16 | Run `python3 scripts/qwen35_compare_tables.py {nano-vllm-amd,llama.cpp-hip,llama.cpp-vulkan,all}` for separate prefill/decode/memory tables; chunking fixes 128K OOM, long-context prefill is at/above parent docs, decode remains slightly behind. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1 long checkpoint: 4K/4K, 32K/128, attempted 128K/128, max_layers=40 | opt-in AOTriton threshold 512 + decode graph replay | `status=diagnostic_retained_with_blocker`, inherited threshold-512 fixture gates, `performance_claim=false`; 128K blocked by OOM | 4K/4K prefill `2379.818 tok/s`, decode `108.930 tok/s`; 32K/128 prefill `1718.308 tok/s`, decode `93.933 tok/s`; 128K/128 OOM during prefill scratch reservation | tracked peak 4K/4K `20.53 GiB`, 32K/128 `35.10 GiB`; 128K blocked at `linear_attn.out_rot` allocation | [`2026-05-16-hipengine-qwen35-long-checkpoint-diagnostic.json`](results/2026-05-16-hipengine-qwen35-long-checkpoint-diagnostic.json) | 2026-05-16 | Superseded for 32K/128 and 128K/128 by the chunking checkpoint above; retained as the pre-chunk baseline and 4K/4K context. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1 prompt sweep 32/64/128/256/512/1024/4096 plus 512/128 + 4K/128 graph rows, max_layers=40 | AOTriton threshold sweep | `status=diagnostic_retained`, prefill fixture `max_kl=0.0396`, top-1 `100%`, graph fixture final KL `0`, `performance_claim=false` | Forced AOTriton vs native prefill: 32 `-16.7%`, 64 `-16.6%`, 128 `-10.9%`, 256 `-3.5%`, 512 `+6.4%`, 1024 `+37.6%`, 4096 `+255.7%`; graph rows: 512/128 prefill `2270.750 tok/s`, decode `109.123 tok/s`; 4K/128 prefill `2345.670 tok/s`, decode `110.091 tok/s` | graph rows tracked peak 512 `18.58 GiB`, 4K `20.42 GiB`; AOTriton adds `+0.039/+0.315 GiB` vs native at 512/4K | [`2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json`](results/2026-05-16-hipengine-qwen35-aotriton-threshold-sweep-diagnostic.json) | 2026-05-16 | Crossover is between 256 and 512 prompt tokens; recommended installed-AOTriton policy is `--attn-aotriton-min-tokens 512`. Code default remains `0` because AOTriton is optional and absent-runtime sessions must not fail. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1, prompt 512/4096 / decode 128, max_layers=40 | opt-in `aotriton_v3_prefill` + decode graph replay | `status=diagnostic_retained`, graph fixture gate `passed=true`, final KL `0`, generated IDs match eager/fixture, `performance_claim=false` | 512/128 prefill `2312.754 tok/s`, decode `109.340 tok/s`; 4K/128 prefill `2372.725 tok/s`, decode `110.303 tok/s` | tracked peak 512 `18.58 GiB`, 4K `20.42 GiB`; sampled HIP used peak 512 `18.60 GiB`, 4K `19.01 GiB` | [`2026-05-16-hipengine-qwen35-decode-graph-replay-diagnostic.json`](results/2026-05-16-hipengine-qwen35-decode-graph-replay-diagnostic.json) | 2026-05-16 | One-step HIP graph replay closes the opt-in AOTriton decode gap to parent from roughly -12%/-10% to -5.8%/-2.4%; threshold policy is now covered by the sweep artifact above. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1, prompt 512/4096 / decode 128, max_layers=40 | opt-in `aotriton_v3_prefill_gate_rotate` | `status=diagnostic_retained`, fixture `max_kl=0.0396`, top-1 `100%`, generated IDs match, `performance_claim=false` | 512/128 prefill `2312.857 tok/s`, decode `101.703 tok/s`; 4K/128 prefill `2371.534 tok/s`, decode `102.211 tok/s` | tracked peak 512 `18.58 GiB`, 4K `20.42 GiB`; sampled HIP used peak 512 `18.60 GiB`, 4K `19.01 GiB` | [`2026-05-16-hipengine-qwen35-aotriton-gate-rotate-diagnostic.json`](results/2026-05-16-hipengine-qwen35-aotriton-gate-rotate-diagnostic.json) | 2026-05-16 | AOTriton remains opt-in (`attn_aotriton_min_tokens > 0`); the threshold sweep above now recommends `512`, while full `LLM.generate()` protocol remains pending. Gate is fused into PARO rotate and BF16 attention output aliases the old gated scratch, saving memory while throughput is neutral/slightly negative. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=1, prompt 512/4096 / decode 128, max_layers=40 | `native_prefill_c1_multiloop` | `status=diagnostic_retained`, fixture `max_kl=0.0341`, top-1 `100%`, dual fused-W4 trace confirms launch, `performance_claim=false` | 512/128 prefill median `2077.262 tok/s`, decode `~101.2 tok/s`; 4K/128 prefill median `659.950 tok/s`, decode `102.146 tok/s` | fixture owned device `1.51 GiB` diagnostic accounting | [`2026-05-15-hipengine-qwen35-native-prefill-multiloop-512-4k-diagnostic.json`](results/2026-05-15-hipengine-qwen35-native-prefill-multiloop-512-4k-diagnostic.json) | 2026-05-16 | Active multiloop retained diagnostic after fusing paired transposed W4 prefill projections into one dual fused-W4 launch; not an accepted `LLM.generate()` throughput row and still below parent target. |
| Qwen3.5-35B-A3B-PARO | w4_paro | c=8, prompt 8 first compact slab | `native_prefill_compact_cN_plan` | `status=blocked`, slab metadata valid, `performance_claim=false` | no packed prefill launched; metadata only | n/a | [`2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json`](results/2026-05-15-hipengine-qwen35-native-prefill-compact-c8-blocked.json) | 2026-05-15 | Historical plan-only blocker, superseded by native compact prefill correctness artifacts above. |
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
