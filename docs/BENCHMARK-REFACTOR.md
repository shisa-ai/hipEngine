# Benchmark Refactor Plan

Status: **proposal** (design doc, not yet implemented)
Owner: platform / benchmark
Companion: [`benchmarks/README.md`](../benchmarks/README.md) "Benchmark harness catalog",
[`docs/BENCHMARK.md`](BENCHMARK.md), [`docs/EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md).

This document proposes how to simplify hipEngine's benchmarking so it is
*manageable* — one place to add a new measurement, one canonical artifact
shape, and a set of *documented shapes* that produce the tables we actually
publish, instead of a family of bespoke scripts that each re-implement timing,
memory, and provenance by hand.

It contains (a) a concrete inventory of what exists today and how each piece
maps into the new system, and (b) a full catalog/manual of how the new system
works end-to-end, so we can reason about coverage before writing code.

It is a plan, not a promise that it ships today. Every step below is scoped so
it can land as its own reviewable unit, and nothing here weakens the evidence
policy in `docs/BENCHMARK.md` (that stays normative).

---

## 1. Problem statement

The catalog in `benchmarks/README.md` lists ~12 harnesses. They are the
accumulated answer to "how do we measure X?" and they work, but they have real
costs:

1. **Duplicated measurement plumbing.** At least **5 scripts** define their own
   `_memory_snapshot` / `_memory_summary`
   (`qwen35_paro_bench.py`, `paro_live_server_bench.py`, `gguf_live_server_bench.py`,
   `gguf_packed_ar_bench.py`, `qwen35_gguf_bench.py`), each with its own
   tracked/HIP/GTT memory rollup. Timing loops for prefill/decode/MTP are copied
   per harness. A bug or a new memory scope is fixed five times or drifts in one.
2. **Axes are baked into the harness rather than orthogonal.** AR vs MTP,
   single-request vs concurrency, prefill vs decode, memory scope — these are
   *orthogonal axes*, but today they are interleaved. Adding MTP to an AR-only
   bench (or prefill to an MTP-only bench) means editing that harness's core
   loop, not composing a shape.
3. **Results are hard to compare.** Because timing scope, memory scope, and
   numerical contract differ per harness, the README must carry heavy
   "compare only like-for-like" caveats. That is a symptom of the design, not a
   reader problem.
4. **No single place to add a test.** To answer "does this model do MTP at
   c=8?" today you must know which harness already has each piece and wire them
   together by hand.

The goal: **adding a new benchmark = declaring a new shape, not writing a new
script.**

---

## 2. What already exists (build on this, don't reinvent it)

The foundation is already centralised in the `hipengine.benchmark` package and
the runtime session classes. The refactor should **extend** these, not replace
them.

### 2.1 The shared `hipengine.benchmark` package

| Module | What it provides today | Reuse in the new system |
| --- | --- | --- |
| `provenance.py` | `collect_artifact_provenance`, `validate_artifact_provenance`, `collect_repo_state`, `collect_model_identity` | Unchanged; every artifact carries this block. |
| `prompts.py` | `load_prompt_records`, `text_sha256`, `file_sha256`, `build_prompt_records`, `StablePromptSpec`, category/holdout contract | Unchanged; the `prompt_set` axis resolves through this. |
| `correctness.py` | `LogitCorrectness`, `evaluate_logits` | Unchanged; composed into artifacts as the correctness gate. |
| `speculative.py` | `aggregate_speculative_rows`, `build_speculative_artifact`, `normalize_speculative_row`, `acceptance_summary`, `SpeculativeGraphStatus`, `D2HCounts` | Unchanged; reused for MTP acceptance/speed rollups. |
| `exact_tokens.py` | `ExactTokenOracle`, `validate_exact_token_parity`, `load_exact_token_fixture` | Unchanged; the `correctness` shape gate. |
| `matrix.py` | `build_benchmark_matrix`, `validate_benchmark_matrix`, `MatrixError` | Unchanged; post-hoc joining of direct/server rows. |
| `execution_profiles.py` | `EvaluationThresholds`, `ControlRecord`, `RunCapture`, profile gates | Unchanged; the `contract` axis gate. |
| `agentic*.py` | agentic / quality live harnesses | Separate concern; stays as-is, not merged. |
| `control_capture.py` | capture/replay helpers | Used by `correctness`/profiler shapes. |

### 2.2 The runtime session surfaces the timing primitives will wrap

**GGUF resident** (`hipengine/runtime/qwen35_gguf_runner.py`,
`Qwen35GGUFResidentSession`):
`reset()`, `prefill()`, `prefill_slot()`, `step()` (AR decode),
`step_async_top1()`, `step_rows()` / `step_rows_native()` / `step_batch_native()`
(c>N batch decode), `verify_rows()` / `verify_target_block()` (MTP verify),
`mtp_draft_seed()` / `mtp_verify_seed()` (MTP), `close()`.

**GGUF MTP** (`hipengine/runtime/qwen35_gguf_mtp.py`,
`Qwen35GGUFMTPDecodeSession`): `generate(...)`, `close()` — the higher-level
draft+verify cycle driver.

**PARO resident** (`hipengine/runtime/qwen35_paro_runner.py`,
`Qwen35ParoResidentSession`): `reset()`, `prefill_native()`, `step()` (AR),
`step_batch_native()` / `step_batch_serial()` (c>N), `verify_speculative_batch()` /
`commit_verified_state()` (MTP), `capture_decode_graph()` / `replay()`, `close()`.

These are the three surfaces the new `sessions.py` + `core.py` timing primitives
adapter to. They are **already** engine/quant agnostic at the session level;
what is missing is a uniform benchmark adapter over them.

### 2.3 What is missing and should be added to the core

- A central **memory snapshot/summary** helper. Today the GGUF and PARO
  `_memory_snapshot` produce near-identical dicts
  (`tracked_peak_allocated_gib`, `hip_used_peak_sampled_gib`, before/after-close
  tracked current), duplicated across 5 scripts. This becomes
  `hipengine/benchmark/memory.py`.
- Central **timing primitives** for the three decode modes (`prefill`,
  `ar_decode`, `mtp_decode(budget)`), each returning a normalized
  timing/throughput record. This becomes `hipengine/benchmark/core.py`.
- A **shape plan + orchestrator** that composes axes into one run and emits one
  canonical artifact. This becomes `shapes.py` + `runner.py` + `artifact.py`.

---

## 3. Current inventory → target mapping

How each existing entrypoint and script maps into the refactored modules. The
rule: **the thin CLI keeps its name and flags; the body delegates to one
`run_shape(...)` call.**

| Current entrypoint | Engine / surface | Currently measures | New shape it maps to | Primary new module |
| --- | --- | --- | --- | --- |
| `scripts/qwen35_readme_sweep.py` | paro+gguf / direct | single-request prefill+decode+memory per shape | `single-request-pp` (+ `single-request-mtp`) | `runner.py`, `core.py` |
| `scripts/qwen35_gguf_bench.py` | gguf / direct c1 | AR prefill+decode, fresh session per run, graph decode | `single-request-pp` (gguf variant) | `runner.py` |
| `scripts/qwen35_paro_bench.py` | paro / direct | AR prefill+decode | `single-request-pp` (paro variant) | `runner.py` |
| `scripts/gguf_true_ar_category_bench.py` | gguf / direct c1 | true no-MTP AR baseline over category suite | `ar-vs-mtp-suite` (AR leg) | `core.py` |
| `scripts/gguf_mtp_category_bench.py` | gguf / direct c1 | MTP category matrix over budgets 1..8 + guarded objective | `ar-vs-mtp-suite` (MTP leg) | `core.py`, `speculative.py` |
| `scripts/gguf_ar_mtp_suite.py` | gguf / direct c1 | one-command AR-vs-MTP decode ratio | `ar-vs-mtp-suite` | `core.py`, `runner.py` |
| `scripts/qwen35_batch_retained_bench.py` | paro / direct c>N | compact c>N batch decode (AR + MTP draft depth), equality | `concurrency-ar` / `concurrency-mtp` | `runner.py`, `core.py` |
| `scripts/qwen35_batch_gguf_diagnostic.py` | gguf / direct c>N | c>N generated-token **correctness** equality | `correctness` (c>N) | `correctness.py`, `runner.py` |
| `scripts/gguf_concurrency_baseline.py` | gguf / direct | c1 + serial c2/c4 control timing | `concurrency-ar` (serial leg) | `runner.py` |
| `scripts/server_f1_concurrency_bench.py` | hipEngine+llamacpp / server | HTTP concurrency c=1..8 combined throughput + memory | `concurrency-ar` / `concurrency-mtp` (server) | `runner.py`, `sessions.py` |
| `scripts/mtp-bench.py` | server / llama.cpp-compatible | MTP prompt-suite server economics | `concurrency-mtp` (server) / `ar-vs-mtp-suite` | `runner.py`, `speculative.py` |
| `scripts/exact_token_generation.py` | paro+gguf / direct+server | exact-token identity gate | `correctness` | `exact_tokens.py` |
| `scripts/benchmark_matrix.py` | join / both | join exact-token direct/server rows | (post-hoc) | `matrix.py` |

**Shared script helpers that move into the core:**

| Script helper (current) | Duplicated in | Becomes |
| --- | --- | --- |
| `_memory_snapshot` / `_memory_summary` | 5 scripts (paro, paro-live, gguf-live, packed-ar, gguf-bench) | `hipengine/benchmark/memory.py` |
| `_stats` (median/min/max/mean/stdev) | gguf-bench, readme-sweep, paro-bench, … | `hipengine/benchmark/core.py` |
| `add_kv_policy_args` / `resolve_args_kv_policy` / `kv_policy_json` | `scripts/qwen35_kv_policy_args.py` (shared) | Keep; import from core `sessions.py` |
| `_read_compiler_version` | paro-bench, readme-sweep | `hipengine/benchmark/env.py` |
| therock `THEROCK_ENV` block | `run_w7900_readme_refresh.sh`, `run_gfx1151_readme_refresh.sh` | `scripts/run_bench.sh` (single wrapper) |

---

## 4. Design principles

1. **Orthogonal axes, composed.** A benchmark is a cross-product of independent
   axes (see §5). Each axis has a small, tested primitive. A "shape" selects a
   value per axis; the orchestrator composes them.
2. **One canonical artifact.** Every shape emits one JSON artifact that answers
   the six questions in `docs/BENCHMARK.md` (what ran / which contract / did
   correctness pass / how stable / what did the GPU run / should we keep it),
   with a single shared schema and provenance block. No more per-harness
   rollup formats.
3. **Correctness is orthogonal, never fused with timing.** Generated-token
   equality and KL/top-1 gates stay separate primitives composed into the same
   artifact, exactly as `docs/BENCHMARK.md` requires. A speed run never doubles
   as the correctness oracle.
4. **Hermetic by default.** All GPU runs go through the therock wrapper that
   pins `HIPENGINE_HIP_ARCH`, the ROCm SDK libs, and the cached compiler-version
   file. The core never spawns `hipcc` in a timed region.
5. **Backward-compatible entrypoints.** The existing `scripts/*.py` CLIs remain
   runnable as thin wrappers that delegate to the core, so historical commands
   and artifacts keep reproducing while the implementation collapses.
6. **Anti-gaming preserved.** The core must not introduce any
   input-conditioned shortcut; the multi-prompt suite and train/holdout rules in
   `docs/BENCHMARK.md` remain binding for any shape that reports acceptance or
   speculative speed.

---

## 5. The axis model

Decompose any benchmark into these axes. Each axis is a closed enum of options
implemented once.

| Axis | Options (current) | Owned by |
| --- | --- | --- |
| `engine` | `paro`, `gguf`, `llamacpp-hip`, `llamacpp-vulkan` | session adapter |
| `surface` | `direct` (resident), `server` (OpenAI SSE) | orchestrator |
| `mode` | `ar`, `mtp-b1..b8` | timing primitive |
| `concurrency` | `c1..cN` | orchestrator (batch or server fan-out) |
| `shape` | `prompt_len/decode` (512/128, 4K/128, …) | shape plan |
| `memory_scope` | `tracked`, `hip_sampled`, `device_gtt`, `device_vram` | memory helper |
| `timing_scope` | `prefill`, `decode`, `full_wall`, `component` | timing primitive |
| `contract` | `strict`, `production`, `batch_invariant` | correctness/execution-profile |
| `prompt_set` | single fixture, category suite, train/holdout | prompts + split |

A **shape** is a named, documented value per axis (full catalog in §7).

---

## 6. Proposed architecture

```
scripts/bench_*.py            # thin CLI wrappers (backward-compatible entrypoints)
scripts/run_bench.sh          # therock hermetic wrapper (env -i, HIP_ARCH, compiler cache)
        │  delegate
        ▼
hipengine/benchmark/
  core.py         # timing primitives: prefill(), ar_decode(), mtp_decode(budget)
                  #   → normalized {timings, throughput, generated_ids} records
  memory.py       # snapshot()/summary() across tracked / HIP-sampled / device GTT/VRAM
  sessions.py     # load-once/reset-between-runs lifecycle for GGUF & PARO sessions
  env.py          # compiler-version read, therock env helper, memory_stats reset
  shapes.py       # SHAPES registry: name → axis values (the documented shapes)
  runner.py       # orchestrate a shape plan → one canonical artifact (+ correctness)
  artifact.py     # single canonical schema + validation (six questions)
  _existing: provenance.py, prompts.py, correctness.py, speculative.py,
             exact_tokens.py, execution_profiles.py, matrix.py   # unchanged, reused
```

**`runner.py` contract.** `run_shape(shape_name, args) -> Artifact`:

1. Resolve the shape → axis values.
2. Build the session once (largest context) via `sessions.py`.
3. For each repetition: `reset()`, run `prefill`, run `mode` decode, snapshot
   memory at the declared boundaries, record per-mode timings.
4. For `concurrency > 1`: fan out through the batch/decode path or server
   client per `surface`.
5. Compose the orthogonal correctness gate (exact-token equality, or
   strict/production profile) as a separate primitive.
6. Emit the canonical artifact with `collect_artifact_provenance`.
7. Optionally render the human table (same code that feeds the README rows).

**Thin wrappers.** `scripts/qwen35_readme_sweep.py --shape single-request-pp ...`,
`scripts/gguf_ar_mtp_suite.py --shape ar-vs-mtp-suite ...`, etc. each become
`run_shape(shape_name, parsed_args)`. Historical flags are kept as accepted
overrides so old commands still work.

---

## 7. Full system catalog / manual

This is the manual for the new system: the complete shape catalog, how a run
is driven, the CLI surface, the artifact contract, and the "add a new
measurement" walkthrough. It is the single reference for what the system can
measure, so we can verify coverage before implementing.

### 7.1 Shape catalog (the documented shapes)

Each shape is a row in `SHAPES` and has a stable name, a set of axis values,
and an "answers" note. This is the catalog the README table derives from (R6).

| Shape name | engine | surface | mode | c | prompt_set | mem | timing | Answers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `single-request-pp` | paro/gguf | direct | `ar` | 1 | fixture | tracked+GTT | prefill+decode | per-shape prefill/decode/memory (README single-request table) |
| `single-request-mtp` | paro/gguf | direct | `mtp-bN` | 1 | fixture | tracked+GTT | decode | MTP decode + MTP/AR ratio at c1 |
| `ar-vs-mtp-suite` | paro/gguf | direct | `ar` **and** `mtp-b1..b8` | 1 | category suite | tracked | decode | full AR-vs-MTP ratio, per category + train/holdout |
| `concurrency-ar` | paro/gguf | direct **and** server | `ar` | 1..8 | fixture | tracked+GTT | decode | aggregate + per-request decode, memory (README concurrency table) |
| `concurrency-mtp` | paro/gguf | direct **and** server | `mtp-bN` | 1..8 | fixture | tracked+GTT | decode | MTP decode at c=1..8 + memory |
| `prefill-pp` | any | any | `ar` | any | fixture | any | prefill | prefill-only (e.g. Laguna 4K prompt-only rows) |
| `correctness` | any | direct/server | `ar`/`mtp` | any | fixture | — | none | generated-token equality / KL/top-1; no throughput |
| `context-soak` | gguf | direct | `ar` | 1 | fixture | tracked+GTT | prefill+decode | long-context lifecycle (32K/64K/128K) |
| `profiler-trace` | any | direct | any | any | fixture | — | component | rocprofv3 kernel-family summary (never a throughput claim) |

The existing README scoreboards map directly:
- **Single-request tables** → `single-request-pp` (+ `single-request-mtp`).
- **Concurrency tables** → `concurrency-ar` / `concurrency-mtp`.
- **Speculative-decode tables** → `ar-vs-mtp-suite` (AR leg = `gguf_true_ar_category_bench`,
  MTP leg = `gguf_mtp_category_bench`, ratio computed in-suite).

### 7.2 The run lifecycle (per shape)

```
run_shape(name, args)
  shape = SHAPES[name]
  env.apply_therock()                  # HIP_ARCH, compiler cache, GPU_MAX_HW_QUEUES
  session = sessions.load_once(shape)  # largest shape/context
  plan = shape.build_plan()            # repetitions, warmup/measured, memory boundaries
  for rep in plan.repetitions:
      session.reset()
      prefill_rec = core.prefill(session, prompt, memory_scope=shape.mem)
      decode_rec  = core.mode_decode(session, shape.mode, n_tokens, memory_scope=shape.mem)
      runs.append(compose(prefill_rec, decode_rec))
  correctness = gates.run(shape.contract)          # orthogonal
  artifact = artifact.build(shape, runs, correctness, provenance)
  if shape.render: tables.render(artifact)         # feeds README rows
  return artifact
```

### 7.3 CLI surface

- Universal flags: `--shape`, `--model`, `--backend`, `--engine`, `--quant`,
  `--workloads`, `--budgets`, `--concurrencies`, `--warmup-runs`, `--measured-runs`,
  `--memory-scope`, `--compiler-version-file`, `--require-cached-build`, `--json`.
- Shape-specific flags collapse to axis overrides (e.g. `--budgets 2` selects
  `mtp-b2`; `--concurrencies 1,2,4,8` selects the c sweep).
- Backward-compat: `scripts/qwen35_readme_sweep.py --engine gguf --model X
  --workloads 512/128 ...` still works (delegates to `single-request-pp`).

### 7.4 Adding a new measurement (the manual walkthrough)

1. **Pick the axis to extend.** If it is a new *value* of an existing axis
   (e.g. a new memory scope), add it to the enum in that axis's owner and its
   primitive; no new shape is required.
2. **If it is a new combination of existing axes**, add a row to `SHAPES`
   naming the combination. No new code if all primitives exist.
3. **If it is a genuinely new primitive** (e.g. a new decode mode), add one
   function in `core.py` + a session adapter if needed, then reference it from
   a shape. Update the README catalog + this manual in the same unit.
4. Run through `scripts/run_bench.sh --shape <name>`; the artifact + README
   row + catalog stay consistent automatically.

### 7.5 Artifact contract (six questions, uniform)

Same schema as `docs/BENCHMARK.md` requires, made uniform across shapes:

```jsonc
{
  "schema": 1,
  "shape": "concurrency-ar",
  "axes": { "engine": "gguf", "surface": "server", "mode": "ar", "concurrency": 8 },
  "provenance": { /* collect_artifact_provenance() */ },
  "runs": [ { "repetition": 1, "timings": {}, "throughput": {},
              "memory": {}, "correctness": {} } ],
  "summary": { "prefill_tok_s": {}, "decode_tok_s": {}, "mtp_ratio": {},
               "memory": {} },
  "gpu_trace": { /* optional compact profiler summary */ }
}
```

### 7.6 Coverage check (do we have everything?)

Compare the manual against the README scoreboards and the current harnesses:

| Want to measure | Shape(s) | Covers current |
| --- | --- | --- |
| single-request prefill/decode/memory | `single-request-pp` | readme_sweep, gguf_bench, paro_bench ✅ |
| MTP speedup vs true AR (category, heldout) | `ar-vs-mtp-suite` | ar_mtp_suite + true_ar + mtp_category ✅ |
| concurrency decode AR (engine + server) | `concurrency-ar` | batch_retained + server_f1 ✅ |
| concurrency MTP decode | `concurrency-mtp` | **(new)** — needs MTP added to server/readme path |
| correctness / identity | `correctness` | batch_gguf_diagnostic, exact_token ✅ |
| long-context / soak | `context-soak` | gguf_long_context_pressure_gate ✅ |
| kernel-family trace | `profiler-trace` | gguf_decode_rocprof, mtp_verifier_rocprof ✅ |
| agentic / quality live | `agentic*` | separate concern (unchanged) |

The one gap this exercise exposes: **`concurrency-mtp`** (MTP at c>1) has no
home today — `server_f1` is AR-only and the batch path is PARO-only. That is the
first concrete thing the refactor enables.

---

## 8. Migration plan (phased, each a reviewable unit)

| Phase | Work | Exit condition |
| --- | --- | --- |
| **R0** | Land `memory.py` central helper; migrate the 5 script copies to it | Same memory numbers before/after on one GGUF + one PARO run |
| **R1** | Land `env.py` + `core.py` timing primitives (`prefill`, `ar_decode`, `mtp_decode`) | A unit test drives each primitive against the resident GGUF session on a tiny shape |
| **R2** | Land `shapes.py` + `runner.py` + `artifact.py` for the **single-request** shape | `run_shape("single-request-pp", …)` reproduces `qwen35_readme_sweep.py` output field-for-field |
| **R3** | Port `gguf_ar_mtp_suite.py` onto `ar-vs-mtp-suite` shape (adds prefill while there) | Suite output parity + prefill now present in artifact |
| **R4** | Land `concurrency-mtp` (add MTP to the batch/server path) | c=1..8 MTP rows from one orchestrator |
| **R5** | Retire the internal duplicated bodies; keep thin wrappers | `grep` for `def _memory_snapshot` returns only the central helper |
| **R6** | Auto-render README rows + catalog from the shape registry | Catalog table and scoreboards derive from one source of truth |

Each phase lands with its own worklog entry and a narrow test; phases are
ordered so the core is proven on the simplest shape before it is used to
justify a retained number.

---

## 9. Non-goals

- This is **not** a license to relax the evidence policy. Retained numbers still
  require the full gate.
- It does **not** promise a single "run everything" button that sweeps all
  models/surfaces at once (that would still be a different benchmark per axis);
  it promises one *system* to compose them.
- It does **not** change what a benchmark means; it changes how a benchmark is
  built and recorded so the meaning is uniform.
- The `agentic*` live harnesses are a separate concern and are not merged into
  the shape system in this pass.
