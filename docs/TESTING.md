# hipEngine Testing Discipline

hipEngine is math-heavy software. A change that compiles, launches, and gets faster can still be wrong. The default posture is therefore:

> **Math changes are guilty until proven correct.**

This doc is the test-authoring playbook. Keep `AGENTS.md` short; put detailed test methodology here.

## Borrowed lesson from shisad

The useful shisad lesson is the distinction between structural tests and actual contract tests.

For hipEngine:

- **Structural correctness** is necessary but not sufficient:
  - a registry key resolves;
  - a build artifact path is deterministic;
  - a kernel launches;
  - output shape/dtype is right;
  - `rocprofv3` sees a kernel name.
- **Numerical correctness** is the real product contract:
  - layer output matches the CPU-reference oracle within tolerance;
  - logits preserve KL ≤ 0.05 and top-1 agreement ≥ 90%;
  - edge cases (masking, partial rotary, empty/short spans, non-power-of-two lengths) match the oracle;
  - correctness still holds before performance numbers are retained.

Rule: any test that touches math must include at least one numerical assertion, not just a structural assertion.

## RED / GREEN workflow

Every non-trivial behavior change follows:

1. **Define the contract.** Identify the oracle and tolerance before editing implementation.
2. **RED.** Add or update a targeted test/fixture that fails against the current or intentionally-broken implementation. For a regression, the new test must reproduce the bad behavior where practical.
3. **GREEN.** Implement the minimal change that passes the targeted test.
4. **Guard.** Run the relevant gate matrix below.
5. **Log and commit.** Record exact commands/results in `WORKLOG.md` for non-trivial code, kernel, or correctness changes; commit only after validation passes.

If a failing test cannot be written first, record why in `WORKLOG.md` before implementing. Avoid silent "trust me" math changes.

## Oracles

Preferred oracle order:

1. **Analytic / high-precision NumPy** for small fixtures (`kernels/cpu_reference/`).
2. **Existing monolithic kernel** when porting a known-good HIP kernel split.
3. **Framework oracle** (torch/HF) only outside the hot path and only through explicit optional test tooling.
4. **Golden fixture** committed under `tests/fixtures/` when the expected tensor is small and stable.

Do not use a new HIP kernel as its own oracle. CPU-reference exists so correctness is independent of GPU implementation bugs.

## Required coverage by change type

| Change type | Minimum tests before commit |
| --- | --- |
| Registry / fusion / plugin selection | Exact resolution, fallback order, duplicate/missing errors, negative path, and no backend/quant branch in dispatch code. |
| CPU-reference primitive | A hand-checkable fixture under `tests/fixtures/cpu_reference/`; direct unit test for the formula; shape/error edge case when relevant. |
| HIP kernel port | CPU-reference fixture gate, port-parity vs monolithic source when applicable, launch smoke, and `rocprofv3 --kernel-trace` showing the expected kernel name. |
| Math optimization | A RED fixture that would catch the previous/wrong math; compare against CPU-reference over representative and edge shapes; perf gate only after correctness passes. |
| Quant plugin | Round-trip pack/dequant fixture, scale/zero-point edge cases, dtype/shape assertions, and target layer correctness. |
| KV policy / attention span logic | Deterministic span fixtures for dense and variable-live-span cases; mask/position edge cases; no shortcut around `KVLiveSpans`. |
| Runtime / memory / build | Import-time no-side-effect tests, fake-runtime tests, dry-run build planning tests, and real HIP smoke only after GPU clearance. |
| Public API / server behavior | Unit/integration tests for success and failure paths; include user-visible output assertions once `LLM.generate()` exists. |
| Perf claim | Exact benchmark command from `docs/BENCHMARK.md`, correctness gate, hardware/software context, and compact JSON artifact. |

## Numerical fixture policy

Small deterministic fixtures are committed. Large model outputs, profiler dumps, and raw logs are not.

Commit:

- tiny JSON fixtures under `tests/fixtures/`;
- hand-checkable arrays;
- fixture metadata documenting purpose and tolerance.

Do not commit:

- model weights/checkpoints;
- raw `rocprofv3` CSVs;
- large logits dumps;
- benchmark terminal logs.

Fixture expectations:

- Include dtype and exact input data.
- Include tolerance (`atol`, `rtol`) or gate thresholds.
- Cover at least one non-trivial edge for the primitive over time: masks, odd sizes, partial rotary dims, non-power-of-two lengths, zero-length/one-token cases where valid.
- Regenerate fixtures from the oracle, not from the candidate implementation.

Current fixture runner:

```bash
python3 scripts/check_fixtures.py
```

## Validation matrix

Run the narrowest tier that covers the change. Escalate at milestone boundaries.

### 1. Targeted RED/GREEN

Examples:

```bash
python3 -m pytest tests/test_cpu_reference.py -q
python3 -m pytest tests/test_kernel_registry.py tests/test_fusion_spike.py -q
python3 -m pytest tests/test_build.py tests/test_smoke_add_plan.py -q
```

### 2. CPU deterministic bundle

Use for ordinary non-GPU code changes before commit:

```bash
python3 -m compileall -q hipengine tests scripts
python3 -m pytest -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode registry
python3 scripts/smoke.py --mode cpu-fixtures
python3 scripts/smoke.py --mode smoke-add-plan
rg -n "import torch|torch\." hipengine tests scripts pyproject.toml docs/IMPLEMENTATION.md || true
```

The torch audit may show docstrings/comments, but executable hot-path imports/usages are blockers.

### 3. GPU smoke bundle

Run only when the GPU is explicitly clear:

```bash
python3 scripts/smoke.py --mode smoke-add-hip --n 1024
python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode qwen35-router-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode qwen35-rotary-hip
python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip
python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip
python3 scripts/smoke.py --mode qwen35-paged-kv-write-hip
python3 scripts/smoke.py --mode qwen35-paged-attn-decode-hip
python3 scripts/smoke.py --mode qwen35-paged-attn-split-k-hip
python3 scripts/smoke.py --mode qwen35-paged-attn-gate-hip
python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip
python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip
python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-state-hip
python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16
python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode w8a16-shared-expert-hip --rows 2 --hidden-size 16
python3 scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8
python3 scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8
```

For real kernel ports, also require a working profiler trace. When the workload JIT-builds a ctypes-loaded HIP `.so`, prebuild it first and feed the exact compiler version into the profiled process so `rocprofv3` does not recursively preload into `hipcc`/clang children:

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
artifact = build_smoke_add(load=False, compiler_version=version)
print(artifact.output_path)
PY
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-trace -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

For future smoke modes, use the same pattern or set `HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt` and require a cached build before launching the profiled workload.

### 4. Kernel correctness gate

For every new/ported kernel:

- CPU-reference fixture pass;
- KL ≤ 0.05 and top-1 agreement ≥ 90% for logit-producing paths;
- launch smoke;
- profiler trace with expected kernel name and plausible `DurationNs`.

A perf win with a failed correctness gate is a failed change.

**Speculative-verify path (T1, adopted 2026-06-09).** For MTP/DFlash verify
kernels (e.g. the GDN chain recurrence), the gate is the KL/top-1 bound above
vs `cpu_reference` — **not** bit-exact `exact_ar_match`. `exact_ar_match`
(spec tokens == same-run AR tokens) is a self-consistency check between two
*different* kernels (the chain/verify kernel vs the AR decode kernel); a
KL-correct verify kernel can still flip it by ~1 ULP at a near-tie boundary
(amplified by the degenerate 1-token smoke prompt). Such a flip is **not** a
correctness regression — gate on KL vs cpu_reference. See `docs/MEGAKERNEL.md`
§5/§8.1/§9.4 for the T0→T1 rationale. Owed when claiming MTP *economics*
(not kernel correctness): re-baseline AR tok/s and acceptance rate on real prompts.

### 5. K1 dense INT8 KV gate

Use this gate for `storage_dtype="int8_per_token_head"` changes and for any
long-context K1 benchmark update. It is a capacity/storage protocol first; do
not describe it as a speed win unless the same artifact also shows an accepted
throughput improvement.

Required correctness commands:

```bash
python3 -m pytest tests/test_qwen35_resident_batch_layout.py \
  tests/test_qwen35_kv_e2e_fixture_gate.py \
  tests/test_qwen35_bench_memory_audit.py -q
python3 scripts/check_fixtures.py
python3 scripts/smoke.py --mode qwen35-paged-kv-write-int8-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/smoke.py --mode qwen35-paged-attn-int8-gqa-hip \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build
python3 scripts/qwen35_kv_int8_accuracy.py --device hip --contexts 64,520 \
  --block-size 256 --num-q-heads 16 --num-kv-heads 2 --head-dim 256 \
  --scale-dtype fp16 --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --require-int8-hip --json /tmp/hipengine-int8-accuracy.json
python3 scripts/qwen35_kv_e2e_fixture_gate.py --max-layers 40 \
  --kv-storage int8_per_token_head \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/hipengine-int8-kv-e2e-fixture-gate.json
```

For GGUF Qwen3.6, add the resident BF16-vs-INT8 logit gate. Short contexts are
expected to pass via the BF16 mirror. Long contexts must pass with
`--require-no-bf16-mirror`; the default admitted long layout is a hybrid
3-layer BF16 full-attention prefix plus 7 INT8 layers with effective FP32 scales,
while pure INT8-only remains a diagnostic reproduction path behind
`HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1`:

```bash
python3 scripts/qwen35_gguf_int8_kv_correctness.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --quant gguf_q4_k_m \
  --prompt-lengths 4K --decode-steps 1 --max-sequence-length 131202 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --require-no-bf16-mirror \
  --json /tmp/hipengine-gguf-int8-kv-correctness.json
```

Required benchmark/profiler evidence for retained or blocked K1 rows:

- exact benchmark command with model, quant, backend, hardware, prompt/decode
  shape, chunk settings, `--kv-storage`, and output JSON path;
- correctness status: layer-level INT8 accuracy, E2E fixture KL/top-1, generated
  token match status, and no-shadow memory audit;
- timing: prefill tok/s, warmed decode tok/s, and whether graph replay was used;
- memory: tracked allocator peak, sampled HIP VRAM peak, retained KV payload
  bytes/elements/bytes-per-element, scale bytes/dtype/granularity, and any
  persistent BF16 shadow candidates;
- `rocprofv3 --kernel-trace` summary for INT8 writer and decode kernels with
  call count plus plausible duration (`DurationNs` or computed equivalent), not
  raw CSVs.

The 2026-05-18 K1 artifacts are the current reference rows:

- 128K/128 BF16-vs-INT8 diagnostic:
  `benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-128k-quality-perf-diagnostic.json`
  (`max_kl=0.015328`, top-1 `100%`, no BF16 shadow; INT8 retained KV
  `1.355 GB`; speed `-0.99%` prefill / `-3.20%` decode vs BF16).
- 128K/256K INT8 AOTriton query-reuse + q3072 diagnostic:
  `benchmarks/results/2026-05-18-hipengine-qwen35-int8-kv-aotriton-query-reuse-diagnostic.json`
  (correctness/no-shadow pass, retained KV `2.708 GB` at 256K, sampled
  `22.013 GiB` and tracked `23.766 GiB` pass the 24GiB-class target; the
  temporary BF16 oracle workspace itself remains a follow-up).

### 6. Milestone closure

At milestone boundaries:

```bash
python3 -m pytest -q
python3 scripts/check_fixtures.py
# plus the phase's named GPU/perf target once available
```

Record exact commands and outcomes in `WORKLOG.md` before claiming closure.

## Definition of done for math/kernel changes

A math or kernel change is not done until all applicable evidence exists:

- [ ] Oracle identified (CPU-reference, monolithic source, or explicit external oracle).
- [ ] RED test/fixture added or an explicit no-RED rationale recorded.
- [ ] Targeted tests pass.
- [ ] CPU deterministic bundle passes if code changed outside docs.
- [ ] GPU smoke passes if GPU code changed and GPU is available.
- [ ] `rocprofv3` trace captured for new/ported kernels, or blocker recorded.
- [ ] `WORKLOG.md` records exact commands/results for non-trivial math, kernel, perf, or blocker evidence.

## Claim integrity

Any claim of "works", "correct", "faster", "done", or "ported" must include:

- **Runtime wiring evidence:** where the live path calls the implementation.
- **Numerical evidence:** oracle, fixture/gate, and thresholds.
- **Command evidence:** exact validation command(s) and outcome(s).
- **Scope:** model, quant, shape, backend, hardware when applicable.

Prefer truth-scoped wording:

- Good: "`rmsnorm` CPU-reference fixture `rmsnorm_basic` passes at max_abs=0 under `python3 scripts/check_fixtures.py`."
- Good: "W7900/gfx1100 smoke_add n=1024 passed max_abs=0.0; rocprof trace is currently blocked by profiler hang."
- Bad: "kernel is correct" without oracle/shape/command.
