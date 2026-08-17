# hipEngine Testing Discipline

hipEngine is math-heavy software. A change that compiles, launches, and gets faster can still be wrong. The default posture is therefore:

> **Math changes are guilty until proven correct.**

This doc is the test-authoring playbook. Keep `AGENTS.md` short; put detailed test methodology here.

Before choosing tolerances, declare the execution profile from
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md). Exact request/control
ownership binds in every profile. Strict arithmetic parity, production
same-quant drift, repeat determinism, and cross-composition invariance are
separate contracts and must not be collapsed into one generated-token check.

## Borrowed lesson from shisad

The useful shisad lesson is the distinction between structural tests and actual contract tests.

For hipEngine:

- **Structural correctness** is necessary but not sufficient:
  - a registry key resolves;
  - a build artifact path is deterministic;
  - a kernel launches;
  - output shape/dtype is right;
  - `rocprofv3` sees a kernel name.
- **Profile-qualified numerical correctness** is the real product contract:
  - strict layer output satisfies its declared exact/parent-parity oracle;
  - production logits pass calibrated strict-teacher mean/tail/max KL, top-1,
    deterministic/isolation, BF16-relative, and task gates;
  - the broad KL ≤ 0.05 and top-1 ≥ 90% CPU-reference gate remains an outer
    smoke/safety floor, not sufficient production-default evidence;
  - edge cases (masking, partial rotary, empty/short spans, non-power-of-two lengths) match the declared profile oracle;
  - correctness still holds before performance numbers are retained.

Rule: any test that touches math must include at least one numerical assertion, not just a structural assertion.

## RED / GREEN workflow

Every non-trivial behavior change follows:

1. **Define the contract.** Identify the oracle and tolerance before editing implementation.
2. **RED.** Add or update a targeted test/fixture that fails against the current or intentionally-broken implementation. For a regression, the new test must reproduce the bad behavior where practical.
3. **GREEN.** Implement the minimal change that passes the targeted test.
4. **Guard.** Run the relevant gate matrix below.
5. **Log and commit.** Record exact commands/results in the unit's immutable worklog entry for non-trivial code, kernel, or correctness changes; commit only after validation passes.

If a failing test cannot be written first, record why in the unit's worklog entry before implementing. Avoid silent "trust me" math changes.

## Oracles

Preferred oracle order:

1. **Analytic / high-precision NumPy** for small fixtures (`kernels/cpu_reference/`).
2. **Existing monolithic kernel** when porting a known-good HIP kernel split.
3. **Framework oracle** (torch/HF) only outside the hot path and only through explicit optional test tooling.
4. **Golden fixture** committed under `tests/fixtures/` when the expected tensor is small and stable.

Do not use a new HIP kernel as its own oracle. CPU-reference exists so correctness is independent of GPU implementation bugs.

## Execution-profile gate selection

Declare one binding gate before writing the RED test:

| Profile | Binding arithmetic/result gate | Always exact |
| --- | --- | --- |
| `strict` | Exact/parent-parity boundary named by the kernel or model fixture; fixed-schedule repeat bytes/IDs where declared | Request/slot/token/position/mask/KV/state ownership, transactions, lifecycle, provenance |
| `production` | Same-artifact strict-teacher full-logit distribution, calibrated mean/p95/p99/max KL and top-1 by category/shape/transition, same-schedule determinism, task non-inferiority, BF16-relative delta where available | Same control/ownership surfaces; same-width neighbor isolation; selected/fallback variant manifest |
| `batch_invariant` | Same fixed-seed API result across supported slots, neighbors, widths, admission order, cancellation, and compaction | Same control/ownership surfaces and lifecycle |

For T1/T2 arithmetic changes, the RED fixture must fail on the changed numerical
boundary or profile verdict, not merely on free-running generated IDs. Use the
strict trajectory as teacher input so candidate and strict logits are compared
at identical contexts. Record free-running identity as a diagnostic for
production and as a binding result only when the declared strict or
batch-invariant contract requires it.

Stateful or c>N production changes add the applicable dynamic scenario matrix:
fixed c1/c2/c4/c8, ragged lengths, row permutations and neighbor substitution,
sparse retirement, delayed admission, cancellation/reclaim, c1<->cN width
transitions, compaction, page/ring/eviction/transaction boundaries, and
graph/eager repeats. Numeric state values may drift in production; state/KV
ownership, metadata, finiteness, isolation, and transaction destinations may
not.

### Reusable evaluator contract

`hipengine.benchmark.execution_profiles` is the mechanical profile evaluator.
Model adapters emit external `.npy` full-logit arrays plus small capture JSON;
large logits remain uncommitted. The schemas are:

- `benchmarks/schemas/execution-profile-manifest.schema.json` — exact selected
  variants and strict fallbacks over the existing registry variant axis;
- `benchmarks/schemas/execution-profile-capture.schema.json` — aligned
  strict-teacher rows, selected IDs, exact request/slot/position/mask and
  route/scatter ownership, diagnostic route-decision hashes, publication/update
  order, KV/state/RNG ownership, transaction accounting, graph, and lifecycle
  control records;
- `benchmarks/schemas/execution-profile-control-capture.schema.json` — actual
  same-run control telemetry with a run ID;
- `benchmarks/schemas/execution-profile-control-fixture.schema.json` — separate
  expected scenario controls; and
- `benchmarks/schemas/execution-profile-evaluation.schema.json` — retained
  verdict with manifest/capture hashes.

The evaluator computes canonical quant-quality KL/NLL/top-k row metrics, then
mean/p95/p99/max KL and top-1 by category, shape, and transition. It separately
checks exact controls, same-schedule repeat bytes/IDs, same-width neighbor
isolation, optional cross-width/composition invariance, BF16-relative deltas,
and supplied task verdicts. Repeat/isolation/composition captures require
independent run IDs, and isolation/composition also require distinct scenario
IDs plus separate expected-control fixtures, preventing self-comparison from
passing a gate. Strict and candidate primary fixtures are also separate because
their declared graph/fallback metadata can differ even when logical ownership
is identical. Production generated-ID equality to strict remains non-binding.
A row above the `0.02` review boundary yields `requires_review` even
when the calibrated hard envelope passes. Every top-1 mismatch is also emitted
with prompt/step, strict and candidate winners, KL, top-k overlap, strict
margin/rank, teacher NLL/delta-p, and maximum absolute logit delta even when its
KL is below the review boundary. The binding values and calibration evidence
are in [`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md#61-calibrated-production-envelope).

Run a completed packet with:

```bash
uv run python scripts/execution_profile_gate.py \
  --variant-manifest /tmp/production-variants.json \
  --strict-manifest /tmp/strict-variants.json \
  --strict-capture /tmp/strict-capture.json \
  --candidate-capture /tmp/production-capture.json \
  --expected-controls /tmp/production-scenario-controls.json \
  --strict-expected-controls /tmp/strict-scenario-controls.json \
  --repeat-capture /tmp/production-repeat.json \
  --isolation-capture /tmp/production-neighbor-substitution.json \
  --comparison-controls /tmp/neighbor-substitution-controls.json \
  --task-results /tmp/category-task-results.json \
  --arithmetic-class T2 \
  --output /tmp/execution-profile-evaluation.json
```

Add `--batch-invariant-capture` for cross-composition certification and
`--bf16-logits` when an aligned BF16 cache is available. The Qwen3.6 PARO lane
can adapt the existing `scripts/quant_quality/qwen36_teacher.py` fixture/cache
with `scripts/qwen36_execution_profile_adapter.py`, but only when an actual
control capture carrying the same run ID and the resolved variant manifest are
supplied. Expected controls remain a separate gate input; a legacy logits cache
alone, or using actual telemetry as its own expected fixture, cannot certify
control ownership.

## Required coverage by change type

| Change type | Minimum tests before commit |
| --- | --- |
| Registry / fusion / plugin selection | Exact resolution, profile-manifest selection, strict fallback order, duplicate/missing errors, negative path, and no backend/quant/profile branch in dispatch code. |
| CPU-reference primitive | A hand-checkable fixture under `tests/fixtures/cpu_reference/`; direct unit test for the formula; shape/error edge case when relevant. |
| HIP kernel port | Declared strict exact/parent-parity or production numerical fixture gate, CPU-reference outer gate, strict fallback, launch smoke, and `rocprofv3 --kernel-trace` showing the expected kernel name. |
| Math optimization | Declare T0/T1/T2/T3 and execution profile; add a RED fixture that catches wrong math/control ownership; run representative/edge and applicable strict-teacher/dynamic/task gates; perf gate only after profile correctness passes. |
| Quant plugin | Round-trip pack/dequant fixture, scale/zero-point edge cases, dtype/shape assertions, and target layer correctness. |
| KV policy / attention span logic | Deterministic span fixtures for dense and variable-live-span cases; mask/position edge cases; no shortcut around `KVLiveSpans`. |
| Runtime / memory / build | Import-time no-side-effect tests, fake-runtime tests, dry-run build planning tests, and real HIP smoke only after GPU clearance. |
| Public API / server behavior | Unit/integration tests for success and failure paths; include user-visible output assertions once `LLM.generate()` exists. |
| Benchmark matrix / report contract | Synthetic PARO/GGUF direct/server grid; profile/manifest mismatch, binding-vs-diagnostic generated-ID handling, forged denominator, duplicate timing owner, incomplete grid, attachment pointer, and schema checks. |
| Profiler window/report contract | Synthetic marker/kernel CSVs proving exact window containment, family bucketing, per-token accounting, Amdahl arithmetic, and exact-token failure behavior. |
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
python3 -m pytest tests/test_benchmark_matrix.py tests/test_exact_token_benchmark.py -q
```

For matrix changes, also build and validate the committed diagnostic manifest.
The direct/server rows use different wall scopes, so the expected report has a
null rate ratio with an explicit scope-mismatch reason:

```bash
uv run python scripts/benchmark_matrix.py build \
  --manifest benchmarks/manifests/sol-m1-paro-e5-diagnostic.json \
  --json /tmp/sol-m1-paro-e5-diagnostic.json
uv run python scripts/benchmark_matrix.py validate \
  --json /tmp/sol-m1-paro-e5-diagnostic.json
```

For `SOL-G1` and any eager GGUF state/lifecycle change, run the four-step
teacher-forced oracle. It first proves the exact repeated-token prompt and
greedy trajectory with llama.cpp, then compares every eager checkpoint against
a fresh token-serial prefix recomputation. Hidden rows, all Conv/GDN state, and
the live full-attention K/V prefixes must be byte exact; any failure reports the
first layer/component:

```bash
uv run python scripts/gguf_eager_teacher_forced_oracle.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip_gfx1151 --prompt-token-id 9707 --prompt-length 512 \
  --decode-steps 4 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/sol-g1-gfx1151-p512-d4.json
```

For `SOL-G2` and any GGUF GDN prefill math/routing change, compare explicit
`fused` and `chain` production bulk prefill on the 17-token greeting. The exact
split chain keeps raw Q/K and their normalization scales separate so the
recurrence can preserve the fused decode-order arithmetic. The all-layer
diagnostic lane identifies the first hidden-output and resident Conv/GDN
divergence. Candidates claiming the exact contract must pass byte-for-byte. A
reassociated but algebraically equivalent candidate may use `--allow-mismatch`
for diagnosis, but that comparator artifact alone is not an acceptance:

```bash
uv run python scripts/gguf_gdn_prefill_compare.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip_gfx1151 --prompt-kind greeting \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/sol-g2-gfx1151-greeting.json
```

Exact-contract acceptance also runs repeated token `9707` at 512, 1024, 1025,
4095, and 4096 rows. The 1024/1025 pair crosses the exact recurrent-segment
threshold; the 4095/4096 pair exercises the retained 1024-row layer-chunk
boundary. Greeting and 512 retain the all-layer bisect; longer cases may use
`--skip-layer-bisect`. Single-order wall fields from this driver are
correctness diagnostics only and cannot select the default.

**Peer-aligned reassociated GDN precedent (adopted 2026-07-15).** llama.cpp
HIP, llama.cpp Vulkan, and hipEngine PARO all evaluate the same F32 recurrence
with parallel/tree reductions that are not guaranteed bit-exact to a scalar
decode-order contraction. This established that teacher-forced model quality,
not state bytes or free-running greedy identity, is the right denominator for
T2 production arithmetic. It does not remain a standalone broad admission
policy: new and re-certified routes use the tighter profile-wide mean/tail/max
KL, top-1, isolation, determinism, BF16-relative, and task gates in
`EXECUTION-PROFILES.md`. The historical KL <= 0.05 / top-1 >= 90% semantic gate
is calibration evidence and the outer safety ceiling. It never retroactively
admits a candidate that failed primitive, determinism, category, task, decode,
or speed gates.

Run the semantic gate before the speed gate:

```bash
python3 scripts/gguf_gdn_semantic_gate.py \
  --model /models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --backend hip_gfx1100 --baseline-mode chain_lds32_direct \
  --candidate-mode chain_k2 --correctness-decode-steps 24 \
  --performance-decode-steps 128 --performance-repetitions 2 \
  --kl-threshold 0.05 --top1-threshold 0.90 \
  --bulk-attention-mode bulk --graph-replay-decode --decode-repack \
  --use-wmma-prefill --use-gemv-decode --attn-aotriton-min-tokens 512 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/gdn-semantic.json
```

Exact default selection continues to use `scripts/gguf_gdn_prefill_ab.py`, not
comparator wall fields. Its contract gate requires unique positive contexts, a
passing exact G2 artifact that covers each context, balanced even repetitions,
exact timed tokens, clean provenance, and a win at both 512 and 4096. For a
reassociated candidate, use the semantic artifact above plus the same balanced
two-shape speed protocol; do not mislabel the G2 byte comparator as failed
product correctness. `--baseline-mode` defaults to `fused`; use the current
promoted mode explicitly for incremental A/Bs.

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

Run only when the GPU is explicitly clear. The default-off prefill flight
recorder has a HIP-availability-guarded host-mapping test: it publishes a
same-stream completion marker into file-backed mapped host memory and verifies
that a separate decoder process sees the retired sequence. Its device kernel
also requires a cached `rocprofv3 --kernel-trace` smoke under the name
`prefill_flight_recorder_mark_i64_kernel`.

```bash
HIPENGINE_HIP_ARCH=gfx1151 python3 -m pytest \
  tests/test_prefill_flight_recorder.py tests/test_hip_runtime.py -q
```

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

- declare execution profile and T0/T1/T2/T3 arithmetic source;
- strict exact/parent-parity RED or production numerical RED, as applicable;
- CPU-reference fixture and the KL ≤ 0.05 / top-1 ≥ 90% outer floor for
  logit-producing paths;
- registered strict fallback for every production/fused route;
- applicable strict-teacher category/shape/transition, isolation, determinism,
  BF16-relative, and task gates before production promotion;
- launch smoke; and
- profiler trace with expected kernel name and plausible `DurationNs`.

A perf win with a failed declared-profile gate is a failed change.

**Speculative-verify historical precedent (old T1, adopted 2026-06-09).** For
MTP/DFlash verify kernels such as GDN chain recurrence, bit-exact
`exact_ar_match` between two different kernels is not a universal production
quality gate. The CPU-reference KL/top-1 check remains the leaf floor, while a
new production promotion additionally uses the complete execution-profile
strict-teacher/task/economics packet. `exact_ar_match` (spec tokens ==
same-run AR tokens) is a self-consistency check between two *different* kernels
(the chain/verify kernel vs the AR decode kernel); a numerically close verifier
can flip it by about one ULP at a near-tie boundary. Such a flip alone is not a
control-ownership bug, but it also does not waive the full production quality
or MTP economics gate. See `docs/MEGAKERNEL.md` §5/§8.1/§9.4 for the historical
rationale. Any MTP speed claim re-baselines true AR tok/s, task quality,
acceptance, and complete cycle economics on the full suite.

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

For Qwen3.6/PARO long-context admission, also run
`scripts/qwen35_paro_int8_kv_quality_sweep.py --comparison-mode both` on the
promotion shapes. The quality gate uses the candidate rollout forced with the
BF16 reference token inputs, so every KL/top-1 position has the same token
history. The independent greedy rollout is diagnostic: only its
`matched_history_logit_gate` is an intrinsic fidelity comparison; metrics after
`first_context_divergent_logit_position` include rollout cascade and must not be
reported as quantization-only error.

Use the bounded task smoke as a fast product-level companion, not as a full
benchmark replacement:

```bash
python3 scripts/qwen35_paro_kv_quality_smoke.py \
  --suite benchmarks/prompts/kv-int8-long-context-smoke.jsonl \
  --context-tokens 4096 --max-new-tokens 48 \
  --kv-storage int8_per_token_head \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt \
  --require-cached-build --json /tmp/hipengine-kv-quality-smoke.json
```

The suite has one retrieval, multihop, aggregation, long-document, and code
row. It gates candidate regressions only when the BF16 reference answers the
same row correctly; a BF16 failure is `reference_unscorable`, not an INT8 pass.
Exact candidate/reference token equality remains diagnostic because distinct
valid wording is allowed when both task scores pass.

For GGUF Qwen3.6-35B-A3B, add the resident BF16-vs-INT8 logit gate. Short
contexts are expected to pass via the BF16 mirror. Long contexts must pass with
`--require-no-bf16-mirror`; the 35B safety fallback keeps 8 of 10 full-attention
layers as BF16 primary storage and uses INT8 only for the final two full-attention
layers. Lower prefixes and pure INT8-only remain diagnostic-only behind
`HIPENGINE_GGUF_INT8_KV_ALLOW_UNVERIFIED_LONG=1`: after the 2026-06-24
layer-local BF16 prefill-oracle fix, prefix 8 passes full W7900 `128K/128`
(`KL mean=0.01448`, top-1 `0.96124`, no persistent BF16 mirror), prefix 7 fails
`128K/16` top-1, and pure INT8 fails `4K/1`. Dense Qwen3.6-27B does not inherit
that result. Its native 24-query/4-KV-head split-K consumer is now CPU-reference
gated and traced, but pure FP32-scale INT8 fails the complete 512/8 prompt suite
at `77.78%` minimum-prompt top-1. The measured 9-BF16/7-INT8 diagnostic map
passes complete 512/8 and 4K/16 suites and bounded mixed 8K/16K/32K rows, but it
is not a promotion candidate: its layer map was selected on the sole failing
train prompt, GGUF prefill-oracle peak rises, graph capture is unsafe, and no
256K capacity gate exists. Any successor must rerun 512/8 before 4K/16, include
all category/heldout prompts, require no BF16 mirror, audit the exact layer
partition, then pass graph/eager safety and a 24GB long-context capacity gate.
Qwen3.8-27B must also be gated independently from Qwen3.6 despite identical
24Q/4KV geometry, and each artifact/backend combination is independent. The
local gfx1100 file (size `17,106,773,984`, SHA-256 `7b2aec...`) passes pure
FP32-scale INT8 complete 512/8 and 4K/16 plus bounded `mixed_v1` 64K/16 and
129,024/16. At 129,024/16 mean/max KL is `0.0000104/0.0001354`, top-1 is 100%,
and the layout has no BF16 mirror. Its layer-outer/shared-oracle route lowers
the 33,024-position tracked peak `18.943 -> 17.330 GiB`, runs matched prefill
within `-0.050%` of BF16 graph with overlapping samples, improves AR decode
`+6.499%`, and closes the XTX c1 physical ceiling at 126K. Four natural 112K
requests with `0.662 GiB` headroom remain the service recommendation; the 126K
row has only `0.022 GiB` headroom.

A different gfx1151 file (size `17,106,775,008`, SHA-256 `7e78da...`) rejects
pure native INT8 at complete 1K/8 with `77.78%` minimum-prompt top-1. Neither
result transfers by filename or geometry. The active
[`IKV-C0`-`IKV-C7` campaign](QWEN38-INT8-KV-CONTINUOUS.md) has completed its
first gate: demand-driven full-file SHA-256 plus backend/target/quant/layout/scale
identity qualifies the exact gfx1100 file, preserves the gfx1151 rejection, and
falls unknown or mismatched contracts back to BF16 unless an explicit
non-promotable diagnostic override is set. The remaining campaign requires
no-mirror c2/c4/c8 primitive, model, lifecycle, memory, and kernel-ownership
evidence.
Short mirrored continuous rows are scheduler evidence only. Pure INT8 graph
capture still fails closed, exact natural B3 MTP is only `0.6423x` true AR, and
BF16 remains supported/default. Any successor that changes support/default
status must preserve complete and long quality rows, prove graph/eager safety
for its claimed transport, retain actual server headroom, and make every
enabled AR/MTP mode same-suite non-regressive. The `4K` forced-long gate below
is a quick 35B guard;
promotion of a 24GB `128K/128` row also requires the same gate at
`--prompt-lengths 128K`,
`--decode-steps 128`, and `--max-sequence-length 131202`.

```bash
HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
HIPENGINE_HIPCC_VERSION_FILE=/tmp/hipengine-hipcc-version.txt \
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

Record exact commands and outcomes in the unit's immutable worklog entry before claiming closure.

## Definition of done for math/kernel changes

A math or kernel change is not done until all applicable evidence exists:

- [ ] Execution profile and T0/T1/T2/T3 source declared.
- [ ] Oracle identified (strict, CPU-reference, monolithic source, or explicit external oracle).
- [ ] RED test/fixture added or an explicit no-RED rationale recorded.
- [ ] Targeted tests pass.
- [ ] CPU deterministic bundle passes if code changed outside docs.
- [ ] GPU smoke passes if GPU code changed and GPU is available.
- [ ] Exact control/ownership plus applicable strict-teacher, dynamic,
      determinism, BF16-relative, and task gates pass.
- [ ] Strict fallback and profile/variant-manifest provenance are recorded.
- [ ] `rocprofv3` trace captured for new/ported kernels, or blocker recorded.
- [ ] The unit's immutable worklog entry records exact commands/results for non-trivial math, kernel, perf, or blocker evidence.

## Claim integrity

Any claim of "works", "correct", "faster", "done", or "ported" must include:

- **Runtime wiring evidence:** where the live path calls the implementation.
- **Numerical evidence:** declared profile, oracle, fixture/gate, thresholds,
  and whether generated-ID equality is binding or diagnostic.
- **Variant evidence:** execution-profile schema plus selected/fallback manifest
  hash for profile-sensitive paths.
- **Command evidence:** exact validation command(s) and outcome(s).
- **Scope:** model, quant, shape, backend, hardware when applicable.

Prefer truth-scoped wording:

- Good: "`rmsnorm` CPU-reference fixture `rmsnorm_basic` passes at max_abs=0 under `python3 scripts/check_fixtures.py`."
- Good: "W7900/gfx1100 smoke_add n=1024 passed max_abs=0.0; rocprof trace is currently blocked by profiler hang."
- Bad: "kernel is correct" without oracle/shape/command.
