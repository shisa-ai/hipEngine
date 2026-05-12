# HIPENGINE Benchmark Procedures

Protocols, baselines, and artifact formats for every perf claim HIPENGINE retains. This doc is the companion to the "Evidence Policy" rule in `AGENTS.md` and `docs/PLAN.md`: when the rule says "record the exact command", it means the commands here.

See `docs/ROOFLINE.md` for the RDNA3 / W7900 hardware model, per-bucket decode analysis, and the "what not to chase" catalog. This doc is the operational layer on top of it.

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

## Hardware & Software Context (default)

Unless explicitly stated otherwise, HIPENGINE benchmarks run on:

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

These numbers are measured on the shared `/home/lhl/` workspace and recorded in `~/amd-gpu-tuning/WORKLOG.md`. They are the "must beat" bar for HIPENGINE on the same hardware. When HIPENGINE claims a win, the claim is per-column vs the row it beats.

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

Protocol TBD once the scheduler is stable. Will mirror `mini-sglang`'s concurrent decode harness so numbers are directly comparable.

### Microbenchmark (single kernel)

For kernel-local claims (port parity, fusion wins):

- Warmup: 50 iterations
- Measure: 200 iterations, report median + p95
- Report: `DurationNs`, `Grid_Size`, `Workgroup_Size`, `VGPR_Count`, `Scratch_Size`, `LDS_Block_Size` from `rocprofv3 --kernel-trace`

Kernel-local wins that do not translate to ≥ 1% E2E impact on the c=1 short workload are recorded but not defended — see `docs/ROOFLINE.md` §11 "What Not To Chase" (~100 iterations on a 19%-of-time kernel while 76.9% sat untouched is the canonical anti-pattern).

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

## Microbenchmark & rocprofv3

For any port-parity or fusion-win claim, capture a kernel trace. Dumps go under `/tmp/hipengine-profile/` (gitignored). Keep only the compact JSON artifact (below) per run.

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-profile -- \
  uv run python scripts/smoke.py --model Qwen3-0.6B --workload c1-short
```

Post-process the CSV to rank kernels by total `DurationNs`. Audit-first discipline (time share → occupancy → iters-per-thread → VGPR) lives in `~/amd-gpu-tuning/AGENTS.md`.

## Artifact Format

Every retained benchmark number writes one JSON file under `benchmarks/results/<date>-<tag>.json`. The JSON is committed; the raw rocprofv3 CSV and terminal logs are not.

```json
{
  "schema": 1,
  "timestamp": "2026-05-12T18:30:00+09:00",
  "run_tag": "qwen06-c1-short-baseline",
  "hardware": {
    "gpu": "AMD Radeon Pro W7900",
    "arch": "gfx1100",
    "cus": 96,
    "vram_total_bytes": 48301604864
  },
  "software": {
    "rocm_hip": "7.13.26162",
    "hipcc_version": "<from hipcc --version>",
    "python": "3.12.x",
    "torch_rocm": "2.11.0+rocm7.13.0",
    "hipengine_commit": "<sha>",
    "hipengine_dirty": false
  },
  "workload": {
    "shape": "c1-short",
    "model": "Qwen3-0.6B",
    "model_path": "/home/lhl/gpu-tuning/models/Qwen3-0.6B",
    "quant": "fp16",
    "prompt_tokens": 4096,
    "gen_tokens": 4096,
    "concurrency": 1,
    "kv_policy": "dense_paged",
    "warmup_runs": 1
  },
  "command": "uv run python scripts/bench.py --shape c1-short --model Qwen3-0.6B --quant fp16",
  "result": {
    "prefill_ms": 135.78,
    "prefill_tok_s": 30167.12,
    "decode_ms": 267147.80,
    "decode_tok_s": 15.33,
    "wall_s": 267.30,
    "vram_used_bytes_post": 43307237376,
    "torch_reserved_peak_bytes": 42859495424
  },
  "correctness": {
    "kl_mean": 0.018,
    "kl_max": 0.049,
    "top1_agreement": 0.942,
    "oracle": "cpu_reference",
    "fixtures": "fixtures/qwen3-0.6b-smoke/"
  },
  "notes": "baseline; no kernels ported yet, engine runs on cpu_reference backend"
}
```

Fields marked with `<...>` are filled at runtime by `scripts/bench.py` (to be written during Phase 0 scaffold). The `hipengine_commit` + `hipengine_dirty` pair means a dirty-tree number can be recorded but is visibly flagged.

## Playbook: Running a Benchmark

Minimum sequence for a retained number:

1. **Environment snapshot.** Capture `rocminfo`, `rocm-smi`, `hipcc --version` output into the JSON artifact.
2. **Context clear.** `rocm-smi` shows VRAM near idle; no other jobs on the GPU.
3. **Warmup run.** One full workload-shape pass, discarded.
4. **Measurement.** Run the workload; `torch.cuda.synchronize()` around prefill and decode phases when torch is in play; `hipStreamSynchronize` on the default stream otherwise.
5. **Correctness.** Run the layer-level and smoke gates (above). A failing gate kills the number — do not publish.
6. **Artifact write.** Emit the JSON under `benchmarks/results/`. Stage and commit with the code change if there is one, or as its own `perf:` commit otherwise.
7. **Log.** Append an entry to `WORKLOG.md` summarizing the number, the delta vs prior baseline, and any anomalies (high VGPR, scratch, unexpected kernel in trace).

If the number contradicts the roofline prediction by > 2×, stop and re-audit before publishing. Overperformance usually means a measurement bug; underperformance usually means a pathology worth naming.

## Failure as Evidence

A benchmark that failed for a specific reason (OOM at shape X, hang on ROCm version Y, crash on concurrency Z) is still evidence and should be recorded in `WORKLOG.md` with the same rigor: exact command, exact symptom, workload shape, hardware context. "We tried this path and it doesn't work yet" keeps us from wasting time on the same path later.
