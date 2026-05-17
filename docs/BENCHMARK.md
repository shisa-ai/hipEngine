# hipEngine Benchmark Procedures

Protocols, baselines, and artifact formats for every perf claim hipEngine retains. This doc is the companion to the "Evidence Policy" rule in `AGENTS.md` and `docs/PLAN.md`: when the rule says "record the exact command", it means the commands here.

See `docs/ROOFLINE.md` for the RDNA3 / W7900 hardware model, per-bucket decode analysis, and the "what not to chase" catalog. This doc is the operational layer on top of it.

Human-readable rollup: `benchmarks/README.md` tracks current fastest accepted rows, comparison baselines, source-lineage targets, and last-updated dates. `benchmarks/CHANGELOG.md` keeps reverse-chronological one-line history so the README stays compact. Machine-readable artifacts live under `benchmarks/results/`.

## Evidence Policy (restated)

Every retained performance number must carry:

- **Model** (exact path / HF snapshot SHA)
- **Quant** (fp16, w8a16, w8a8-dyn, w4-paro, …)
- **Workload shape** (prompt length, generation length, concurrency, KV policy, warmup)
- **Hardware** (W7900, ROCm version, `hipcc --version`, driver from `rocminfo`)
- **Exact command** (full shell invocation, reproducible from a clean shell)
- **Result** (prefill tok/s, decode tok/s, VRAM used, peak reserved)
- **Correctness gate** (KL ≤ 0.05 AND top-1 ≥ 90% vs `kernels/cpu_reference/` on the fixture set)

Claims without a correctness gate are disallowed. A perf win that regresses correctness is reverted. Raw terminal output is not evidence — retain a compact JSON artifact per the schema at the bottom of this doc.

## Benchmark Output Contract

A benchmark artifact must answer five questions without rereading raw logs:

1. **What ran?** Exact command, model, quant, workload shape, hardware/software context, commit/dirty state.
2. **Did correctness pass?** Fixture set, oracle, KL/top-1 or layer tolerance metrics, exact correctness command(s), pass/fail status.
3. **How stable is the number?** Warmup count, measured repetitions, per-phase samples, median/p95/min/max/stdev where applicable.
4. **What did the GPU actually execute?** Profiler trace status, expected kernel names, time-share summary, and any profiler blocker. Raw traces stay outside git; compact summaries go in JSON.
5. **Should we keep this number?** Baseline reference, delta, acceptance decision, and rejection/blocker reason if not retained.

Allowed artifact statuses:

| Status | Meaning |
| --- | --- |
| `accepted` | Correctness passed, benchmark protocol followed, variance acceptable, and result may be compared later. |
| `rejected_correctness` | Performance may have been measured but correctness failed; number must not be used as a perf claim. |
| `rejected_variance` | Correctness passed but timing was too noisy / contaminated for comparison. |
| `blocked` | Benchmark could not complete (OOM, hang, profiler failure, missing dependency, GPU busy). Record symptom and command. |

A JSON artifact with `status != "accepted"` is still useful evidence, but it is not a retained performance number.

## Human-readable Rollup

`benchmarks/README.md` is the scoreboard. It exists because humans need a quick way to see current fastest rows across models/quants/backends without diffing JSON artifacts.

Maintain it with every retained benchmark:

1. Update the top `Last updated: YYYY-MM-DD` line.
2. Add or replace the row keyed by `(model, quant, backend, workload, policy)`.
3. Link the compact JSON artifact in `benchmarks/results/` for hipEngine measurements.
4. Include correctness/validation, peak memory, source command/artifact, and per-row last-updated date.
5. Add a dated one-line entry to `benchmarks/CHANGELOG.md` for the rollup change: model / quant / workload, metric `old -> new`, percent delta, reason/change, and artifact/source.
6. Keep source-lineage targets and external baselines in separate tables from hipEngine measurements.

Blocked/rejected benchmark attempts may be summarized there only if clearly marked as blocked/rejected; otherwise they live in JSON artifacts and `WORKLOG.md`. Git history for `benchmarks/README.md` and `benchmarks/CHANGELOG.md` is the human-readable performance history; JSON artifacts are the durable evidence.

## Hardware & Software Context (default)

Unless explicitly stated otherwise, hipEngine benchmarks run on:

- GPU: AMD Radeon Pro W7900 (gfx1100, RDNA3, Navi 31)
- Compute: 96 CUs / 192 SIMD32 / wave32 native
- Memory: 48 GiB GDDR6, 864 GB/s peak bandwidth, 96 MiB Infinity Cache
- Peak throughput (FP16 matrix) 123 TFLOP/s, (INT8 matrix) 123 TOP/s, (FP32 vector) 61.3 TFLOP/s
- Host: `therock` Python 3.12 env; PyTorch `2.11.0+rocm7.13.0` only when the `[torch]` dlpack extra is used
- ROCm: 7.13.x series; HIP runtime `7.13.26162` (verify with `python3 -c "import torch; print(torch.version.hip)"` when torch is installed, otherwise `/opt/rocm/bin/hipcc --version`)

Full spec and roofline derivation: `docs/ROOFLINE.md` §1 (hardware) and §2 (roofline fundamentals).

Capture at the top of every benchmark run:

```bash
rocminfo | grep -E 'Name:|gfx' | head -4
rocm-smi --showmeminfo vram --showuse --showtemp
hipcc --version
python3 -c "import torch; print(torch.__version__, torch.version.hip)" 2>/dev/null || echo "(no torch)"
```

## Baselines to Beat

These numbers are measured on the shared `/home/lhl/` workspace and recorded in `~/amd-gpu-tuning/WORKLOG.md`. They are the "must beat" bar for hipEngine on the same hardware. When hipEngine claims a win, the claim is per-column vs the row it beats.

### Qwen3.6-35B-A3B Q8_K_XL on llama.cpp ROCm (current W7900 target)

Source: `~/amd-gpu-tuning/WORKLOG.md` 2026-04-28 entry.

| Workload | Prefill tok/s | Decode tok/s | VRAM used | Notes |
| --- | --- | --- | --- | --- |
| `llama-bench` native (pp512 / tg128) | 949.89 ± 9.59 | 74.32 ± 0.02 | — | `llama-bench -m Qwen3.6-35B-A3B-UD-Q8_K_XL.gguf -fa 1` |
| Localhost server 4K/4K | 1139.72 | 71.49 | 44.94 GiB | `/completion`, 4096 prompt, `n_predict=4096`, `temperature=0`, `ignore_eos=true`, `cache_prompt=false`, `stream=false` |

Build: `llama.cpp 0f1bb602d (8946)` with ROCm backend, `-fa 1` flash attention.

Decoder roofline: 71.49 tok/s at 4K/4K is ~27.5% of the optimistic GGUF-ratio memory roof (~260 tok/s) for the 3.33 GB active-weight bytes/token estimate. Prefill is ~5.6% of the matrix-compute roof. See `docs/ROOFLINE.md` §5 for Amdahl per-bucket framing.

### Qwen3-0.6B FP16 c=1 shootout (nano-vllm vs mini-sglang, 4K/4K)

Source: `~/amd-gpu-tuning/WORKLOG.md` 2026-04-28 shootout entry. Reference for the *host architecture* cost we're beating, not the kernel layer.

| Engine | Prefill tok/s | Decode tok/s | KV shape | KV GiB | Notes |
| --- | --- | --- | --- | --- | --- |
| nano-vllm (enforce_eager, ROCm SDPA) | 30,167.12 | 15.33 | `[2,28,1404,256,8,128]` | 38.39 | 267 s wall on 4096 decode tokens |
| mini-sglang (overlap disabled, `torch_sdpa`) | 20,195.46 | 22.58 | `[2,28,1430,256,8,128]` | 39.10 | 183 s wall on 4096 decode tokens |

mini-sglang is 1.47× faster on decode; nano-vllm is 1.49× faster on prefill. Both sit far below the 35B llama.cpp decode baseline despite being 0.6B — the current torch-SDPA paged decode path is the bottleneck.

## Standard Workloads

Every new perf number should match one of these shapes unless there's a documented reason not to. Protocol-shape drift is how baselines become uncomparable.

### c=1 short (4K/4K)

Matches the llama.cpp localhost server baseline above.

- Prompt: exact 4096 input token IDs (use `/v1/tokenize` or a fixed token-ID file)
- Generation: `n_predict = 4096`, `ignore_eos = true`, `temperature = 0`
- Concurrency: 1 request, TP = 1
- Warmup: 1 prior request (same shape) discarded
- Report: prefill ms + tok/s, decode ms + tok/s, wall-clock s, VRAM used after run, peak reserved

### c=1 long (16K/256)

For KV-policy and long-context work.

- Prompt: exact 16,384 input token IDs
- Generation: 256 tokens, `temperature = 0`, `ignore_eos = true`
- Concurrency: 1, TP = 1
- Warmup: 1 prior request (same shape) discarded
- Additional report: KV cache shape + bytes, eviction events if KVPolicy ≠ dense

### c=N concurrent (Phase 1+)

Correctness comes before throughput. A c=N benchmark row is not eligible for `accepted` status until all of the following are true:

- `scripts/qwen35_batch_correctness.py --rows N` passes with `append_key_mismatch=0`, `append_value_mismatch=0`, and `attn_batch_vs_c1_max_abs <= 1e-6` for the kernel families used by the runner.
- The resident batch runner emits generated-token ids equal to N independent c=1 resident runs for the same fixed prompts (`temperature=0`, SpecDec disabled).
- The artifact records scheduler occupancy, active mask shape, graph bucket key, KV policy, and whether compaction occurred.
- For continuous batching, include admission/completion timestamps and per-request p50/p95 latency in addition to aggregate tok/s.

Initial protocol shapes:

| Shape | Purpose | Required correctness command |
| --- | --- | --- |
| `c=2`, prompt 512 / decode 128 | bring-up parity and debugging | `python3 scripts/qwen35_batch_correctness.py --rows 2` plus generated-token equality vs two c=1 sessions |
| `c=4`, prompt 512 / decode 128 | first scheduler/graph bucket row | `python3 scripts/qwen35_batch_correctness.py --rows 4` plus generated-token equality vs four c=1 sessions |
| `c=8`, prompt 512 / decode 128 | primary early concurrent target | `python3 scripts/qwen35_batch_correctness.py --rows 8` plus generated-token equality vs eight c=1 sessions |

Report both aggregate tok/s and per-request tok/s. Do not compare a c=N aggregate row against c=1 without explicitly showing `aggregate/c1` and `per_request/c1` ratios. SpecDec must be disabled for these rows; SpecDec has a separate acceptance protocol because generated-token equality depends on target verification and KV commit semantics.

### OPTIMAL MoE/PARO parity rows

For the Qwen3.5-35B-A3B-PARO exercise, first keep source-lineage parent rows and hipEngine attempts as separate artifacts:

- Parent/source-lineage rows use `~/amd-gpu-tuning/scripts/bench_paro_native_engine.py` with the 23 base flags from `~/amd-gpu-tuning/docs/OPTIMAL.md` and `--decode-use-step-graph-replay`.
- Initial parity shapes are `512/128` and `4K/128`; later add `1K/128`, `32K/128`, and `128K/128` after the port path is stable.
- Parent rows can be `accepted` source-lineage artifacts when finite logits and graph/eager validation pass. They are comparison targets, not hipEngine measurements.
- hipEngine rows stay `blocked` until `LLM.generate()`, `w4_paro` loading/layout, Qwen3.5 model plugin, required kernels, and graph replay exist.
- When hipEngine runs, compare against the matching parent artifact and require the same post-run quality gates plus hipEngine's KL/top-1 gate.

Current local parent artifacts:

- `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-512-128.json`
- `benchmarks/results/2026-05-13-source-lineage-qwen35-paro-optimal-4k-128.json`
- `benchmarks/results/2026-05-13-hipengine-qwen35-paro-optimal-blocked.json`

### Microbenchmark (single kernel)

For kernel-local claims (port parity, fusion wins):

- Warmup: 50 iterations
- Measure: 200 iterations.
- Report for each measured metric: samples count, median, p95, min, max, and stdev.
- Report profiler fields: kernel name, grid size, workgroup size, duration, `VGPR_Count`, `Scratch_Size`, and `LDS_Block_Size` from `rocprofv3 --kernel-trace`. If the CSV has `Start_Timestamp` / `End_Timestamp` instead of `DurationNs`, compute `DurationNs = End_Timestamp - Start_Timestamp` in the compact summary/artifact.

Kernel-local wins that do not translate to ≥ 1% E2E impact on the c=1 short workload are recorded but not defended — see `docs/ROOFLINE.md` §11 "What Not To Chase" (~100 iterations on a 19%-of-time kernel while 76.9% sat untouched is the canonical anti-pattern).

## Measurement Statistics

Every accepted benchmark artifact records timing as **samples**, not just one number.

Minimum for E2E workloads:

- `warmup_runs`: normally `1` for full workload shapes.
- `measured_runs`: normally `3` for expensive E2E benchmarks unless cost is prohibitive; if fewer, explain in `notes`.
- For each phase (`prefill`, `decode`, `wall`): sample list plus `median`, `p95`, `min`, `max`, `stdev`.
- For memory: pre-run idle VRAM, post-run VRAM, peak allocator reservation when available, KV cache bytes/shape.

Minimum for microbenchmarks:

- `warmup_iters`: normally `50`.
- `measured_iters`: normally `200`.
- Duration stats in nanoseconds and, when meaningful, derived throughput.

Variance guard: if stdev is >5% of median for E2E or >10% for a microbenchmark, mark the artifact `rejected_variance` unless the variance is understood and documented.

## Correctness Gate

Two gates at two granularities. Both are required for any new/ported kernel before a perf claim is accepted.

### Layer-level (`kernels/cpu_reference/` oracle)

```bash
uv run pytest tests/test_<family>_correctness.py -q
```

For each registered `(backend, layer, quant, variant)` tuple, run the same fixture input through the HIP kernel and the CPU-reference implementation. Assert:

- Mean KL divergence ≤ 0.05 over the fixture set
- Top-1 logit agreement ≥ 90%

### End-to-end (fixed-prompt smoke)

```bash
uv run python scripts/smoke.py --model Qwen3-0.6B --prompt fixtures/smoke_prompts.jsonl \
  --reference outputs/cpu_reference/Qwen3-0.6B.logits.npy
```

Runs the full `LLM.generate()` path on a fixed prompt set, saves logits, diffs against the archived CPU-reference logits. Same KL ≤ 0.05 / top-1 ≥ 90% gate.

Fixtures (prompts + reference logits) are tiny (< 10 MB) and *are* committed under `fixtures/`. They are not "benchmark outputs" and do not count against the never-commit rule.

## Post-run Quality Gates

After every E2E benchmark attempt, extract and record these fields before presenting throughput:

1. **Correctness / sanity**
   - `finite_prefill_logits` must be `true` when the benchmark emits it. `false` or `null` means the run is NaN-corrupted or incomplete; mark it `rejected_correctness` or `blocked`.
   - Graph replay validation, when active, must pass (`decode_step_graph_validation=true` or equivalent).
   - For same-prompt A/B comparisons, `generated_sample` token sequences must match unless the run is explicitly a stochastic-quality comparison with a documented seed/protocol.
2. **Performance**
   - Report `prefill_tok_s`, `decode_tok_s`, and total `wall_seconds` with units.
   - If a run is warm-started, say so; do not compare warm-start to cold-start without labeling it.
3. **Memory**
   - Report `allocated_after_load_gib` when available and peak allocated/reserved bytes as GiB.
   - Flag any run above the 24 GiB PARO usability gate separately from W7900-only diagnostic rows.
4. **Presentation**
   - For multiple configs, use a compact table containing correctness, prefill/decode, wall time, and memory in one view.
   - Include external baselines (llama.cpp HIP/Vulkan, parent `docs/OPTIMAL.md`, or previous hipEngine artifact) when the shape has a known comparable row.

Throughput without these fields is not a retained benchmark number.

## Microbenchmark & rocprofv3

For any port-parity or fusion-win claim, capture a kernel trace. Dumps go under `/tmp/hipengine-profile/` (gitignored). Keep only the compact JSON artifact (below) per run.

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- \
  uv run python scripts/smoke.py --model Qwen3-0.6B --workload c1-short
```

Post-process the CSV to rank kernels by total `DurationNs`. Audit-first discipline (time share → occupancy → iters-per-thread → VGPR) lives in `~/amd-gpu-tuning/AGENTS.md`.

## Artifact Format

Every benchmark attempt writes one JSON file under `benchmarks/results/<date>-<tag>.json`. The JSON is committed when it is small and useful. Raw `rocprofv3` CSVs, terminal logs, large logits, and model outputs are not committed.

Schema `2` is the benchmark-output contract:

```json
{
  "schema": 2,
  "status": "accepted",
  "timestamp": "2026-05-12T18:30:00+09:00",
  "run_tag": "qwen06-c1-short-baseline",
  "summary": "Qwen3-0.6B fp16 c1-short baseline",
  "hardware": {
    "gpu": "AMD Radeon Pro W7900",
    "arch": "gfx1100",
    "cus": 96,
    "vram_total_bytes": 48301604864,
    "pre_run_vram_used_bytes": 27930624,
    "post_run_vram_used_bytes": 43307237376
  },
  "software": {
    "rocm_hip": "7.13.26162",
    "hipcc_version": "<from hipcc --version>",
    "python": "3.12.x",
    "torch_rocm": "2.11.0+rocm7.13.0 or null",
    "hipengine_commit": "<sha>",
    "hipengine_dirty": false
  },
  "workload": {
    "shape": "c1-short",
    "model": "Qwen3-0.6B",
    "model_path": "/home/lhl/gpu-tuning/models/Qwen3-0.6B",
    "model_revision": "<hf snapshot or git/ref>",
    "quant": "fp16",
    "prompt_tokens": 4096,
    "gen_tokens": 4096,
    "concurrency": 1,
    "kv_policy": "dense_paged",
    "warmup_runs": 1,
    "measured_runs": 3
  },
  "commands": {
    "environment": ["rocminfo | grep -E 'Name:|gfx' | head -4", "hipcc --version"],
    "correctness": ["python3 scripts/check_fixtures.py"],
    "benchmark": "python3 scripts/bench.py --shape c1-short --model Qwen3-0.6B --quant fp16",
    "profiler": "rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- ..."
  },
  "correctness": {
    "passed": true,
    "oracle": "cpu_reference",
    "fixtures": "tests/fixtures/qwen3-0.6b-smoke/",
    "kl_mean": 0.018,
    "kl_max": 0.049,
    "top1_agreement": 0.942,
    "layer_fixture_max_abs": 0.0003,
    "command_exit_code": 0
  },
  "measurements": {
    "prefill_ms": {
      "samples": [135.4, 135.8, 136.1],
      "median": 135.8,
      "p95": 136.1,
      "min": 135.4,
      "max": 136.1,
      "stdev": 0.29
    },
    "decode_tok_s": {
      "samples": [15.2, 15.3, 15.4],
      "median": 15.3,
      "p95": 15.4,
      "min": 15.2,
      "max": 15.4,
      "stdev": 0.08
    }
  },
  "memory": {
    "kv_shape": [2, 28, 1404, 256, 8, 128],
    "kv_bytes": 41221619712,
    "allocator_reserved_peak_bytes": 42859495424
  },
  "profiler": {
    "status": "captured",
    "raw_trace_path": "/tmp/hipengine-profile/results.csv",
    "expected_kernels_present": true,
    "top_kernels": [
      {
        "name": "qwen35_paged_full_attn_decode_splitk",
        "total_duration_ns": 123456789,
        "time_share": 0.42,
        "grid_size": 4096,
        "workgroup_size": 256,
        "vgpr_count": 80,
        "scratch_size": 0,
        "lds_block_size": 0
      }
    ],
    "notes": "raw_trace_path is not committed"
  },
  "baseline": {
    "name": "llama.cpp Qwen3.6-35B-A3B UD-Q8_K_XL 4K/4K",
    "source": "~/amd-gpu-tuning/WORKLOG.md 2026-04-28",
    "decode_tok_s": 71.49,
    "prefill_tok_s": 1139.72
  },
  "comparison": {
    "decode_delta_pct": -78.6,
    "prefill_delta_pct": 2547.0
  },
  "decision": {
    "accepted": true,
    "reason": "correctness passed and variance below threshold"
  },
  "notes": "baseline; no kernels ported yet, engine runs on cpu_reference backend"
}
```

If a benchmark is blocked or rejected, keep the same schema but set `status` and `decision.accepted=false`, then fill `decision.reason`, the exact failing command, and any symptom fields (`oom_bytes`, `signal`, `exception`, `profiler_status`, etc.).

Fields marked with `<...>` are filled at runtime by `scripts/bench.py` (to be written during Phase 0 scaffold). The `hipengine_commit` + `hipengine_dirty` pair means a dirty-tree number can be recorded but is visibly flagged.

## Playbook: Running a Benchmark

Minimum sequence for a retained number:

1. **Environment snapshot.** Capture `rocminfo`, `rocm-smi`, `hipcc --version` output into the JSON artifact.
2. **Context clear.** `rocm-smi` shows VRAM near idle; no other jobs on the GPU.
3. **Warmup run.** One full workload-shape pass, discarded.
4. **Measurement.** Run the workload; `torch.cuda.synchronize()` around prefill and decode phases when torch is in play; `hipStreamSynchronize` on the default stream otherwise.
5. **Correctness.** Run the layer-level and smoke gates (above). A failing gate kills the number — do not publish.
6. **Artifact + rollup.** Emit the JSON under `benchmarks/results/`, update `benchmarks/README.md`, and add a short entry to `benchmarks/CHANGELOG.md`.
7. **Log.** Append an entry to `WORKLOG.md` summarizing the number, the delta vs prior baseline, and any anomalies (high VGPR, scratch, unexpected kernel in trace). Stage and commit the artifact/rollup/changelog/log with the code change if there is one, or as its own `perf:` commit otherwise.

If the number contradicts the roofline prediction by > 2×, stop and re-audit before publishing. Overperformance usually means a measurement bug; underperformance usually means a pathology worth naming.

## Failure as Evidence

A benchmark that failed for a specific reason (OOM at shape X, hang on ROCm version Y, crash on concurrency Z) is still evidence and should be recorded in `WORKLOG.md` with the same rigor: exact command, exact symptom, workload shape, hardware context. "We tried this path and it doesn't work yet" keeps us from wasting time on the same path later.
