# Cross-Engine Comparison Protocol

Status: 2026-10-12, active campaign. Normative for every cross-engine
comparison run on this host. Companion documents:
[`QWEN4EXP-STATUS.md`](QWEN4EXP-STATUS.md) §1.2 (prior same-fixture rows),
[`EXECUTION-PROFILES.md`](EXECUTION-PROFILES.md) (the hipEngine-internal gates
this document references but does not modify), and
[`BENCHMARK.md`](BENCHMARK.md) (evidence policy).

This protocol exists to answer three questions with measurements rather than
author-reported rows: how fast is each engine under one fixed workload, what
numerical quality accompanies that speed, and where does each engine spend its
device time. It does not certify external engines against hipEngine's
production contract; it characterizes them on shared inputs.

## 1. Engines and payloads

| Engine | Payload | Format | Notes |
| --- | --- | --- | --- |
| hipEngine | unsloth `Qwen3.8-Flash-Next` `UD-Q4_K_XL`, 4 shards, fingerprint `fb1f2fbf73d588c9…` | GGUF | Production profile, BF16 KV, in-tree |
| gufo | same four shards | GGUF | Binary pinned by commit + sha256 per run |
| halo-box `strix-llama.cpp` | same four shards | GGUF | HIP backend build for `gfx1151` |
| kyojin (`Yamz-Labs`) | `yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz` (~95 GB) | EXL3 | Different weights format: every kyojin row is an engine-plus-format row and is never mixed into a same-file ratio |

Every run records engine identity (source commit, binary sha256, reported
version string), the payload sha256, and the host fingerprint (machine id,
kernel, memory, governor/power profile). A row without identity is discarded.

Rules inherited from the 2026-09-15 and 2026-10-04 comparisons still apply:
never telescope cross-engine ratios into causal decompositions; a faster
family bucket is not a qualified family; KV dtype is recorded and not assumed
comparable; engines that cannot load the payload are named, not dropped.

## 2. Canonical throughput protocol

- **Fixture:** [`benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json`](../benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json),
  committed, 12 cases — `code`, `general_en`, `general_ja`, `mixed_ja_en` at
  512/1024/4096 exact token ids.
- **Input discipline:** llama.cpp-family servers receive the token ids
  directly (the [`scripts/llamacpp_raw_suite_bench.py`](../scripts/llamacpp_raw_suite_bench.py)
  pattern). Text-driven adapters (gufo, kyojin) receive the detokenized raw
  text with no chat template, and the harness verifies per case that
  re-tokenization reproduces the fixture ids; any mismatch disqualifies that
  case for that engine and is recorded.
- **Timing boundary:** prefill-only wall, `n_predict=1` on the exact-id path
  (the first sampled token belongs to prefill). For HTTP engines the prefill
  wall is the engine-reported prompt-evaluation time (llama.cpp-family
  `timings.prompt_ms`, kyojin `timings.prompt_ms`, gufo usage rates), with
  the client-side request wall recorded alongside; hipEngine's in-process
  harness measures the equivalent boundary. One unmeasured warmup per case,
  three measured repetitions, median per case, equal-weight mean across cases
  per length. Decode is measured separately as exactly 128 transitions after
  the measured prefill and never enters the prefill wall.
- **Sampling:** greedy (`temperature 0` or the engine's deterministic mode),
  thinking off, no chat template, fixed seed where the engine supports one.
- **Host discipline:** one engine resident at a time; idle-VRAM check before
  each engine block; governor/power profile verified at block start and end;
  no concurrent GPU work during any measured window. Thermal drift is a known
  hazard on this host (see `QWEN4EXP-STATUS.md` 2026-10-09 note); blocks are
  run in the settled state and interleaved engine order is recorded.
- **Aggregation:** per-length means are never averaged into one headline;
  per-case medians and their spread are kept in the raw artifact.

## 3. Quality-gate ladder

Two separate things are called "gates" and they must not be conflated.

**hipEngine-internal gates are unchanged.** The calibrated production
envelope (`EXECUTION-PROFILES.md` §6.1), the layer2 strict-teacher gate
([`scripts/qwen4exp_layer2_profile_gate.py`](../scripts/qwen4exp_layer2_profile_gate.py)),
and the task battery govern hipEngine kernel-plan candidates only. No external
engine is ever "passed" or "failed" against them; their arithmetic is theirs.

**Cross-engine gates characterize.** For external engines the ladder is:

| Gate | What it measures | Method | Applicability |
| --- | --- | --- | --- |
| G0 — protocol conformance | identity, payload, tokenizer round-trip, host state, idle device | recorded per run by the harness | required for any retained row |
| G1 — greedy output identity | exact greedy token ids vs the hipEngine strict-teacher run on the same inputs | canonical 12 cases + the 10-prompt `mtpbench` suite + heldouts; per-prompt first-divergence position reported | all engines |
| G2 — distribution fidelity | top-1 agreement and top-20 KL at fixed teacher-forced positions vs the hipEngine strict teacher | logprob capture where the engine exposes one; full-vocabulary KL only where full logits exist (hipEngine, and any patched engine, with the patch recorded) | llama.cpp-family: top-N; kyojin: logprobs; gufo: not applicable (no logits surface — G1 only) |
| G3 — determinism | identical outputs across repeats in-session and across launches; rate CoV | three repeats minimum | all engines |
| G4 — task battery | the same 18-task strict-exact battery used by the production envelope, scored by the same scorer for every engine | per-task output comparison | all engines; for hipEngine candidates it feeds the existing `requires_task_review` machinery, for external engines it is characterization |

Interpretation rules, fixed in advance:

- G1 agreement is not a distribution claim, and a divergence is not
  automatically a defect for an external engine; it localizes where arithmetic
  differs and feeds G2/G4 follow-up.
- G2 metrics compare against the same-file strict teacher. kyojin's EXL3 pack
  cannot use that reference; its G2 leg uses its own published method
  (top-1/KL vs the official FP8 release) as the format-crossing anchor and is
  labeled as such, or is run against the strict teacher with the format
  caveat named.
- A speed row is always reported next to its G1–G4 columns. Speed without the
  gate columns is not a retained row.

## 4. Microbenchmarking (per-operator attribution)

- **Capture:** `rocprofv3 --kernel-trace` plus `--launch-census` around one
  request-delimited `code-p4096` prefill per engine, server warm. Profile the
  serving child only, never a parent harness that spawns children
  (the MTP profiling rule in `AGENTS.md`). hipEngine captures use the
  precomputed compiler-version file and `--require-cached-build`.
- **Attribution:** kernel names are bucketed into operator families with
  [`scripts/qwen4exp_comparator_role_map.py`](../scripts/qwen4exp_comparator_role_map.py),
  extended per engine for its kernel-name space. Families: dense projections,
  MoE expert gate/up, MoE expert down, hyper-connection/GR, attention, GDN,
  router/index, norm and elementwise, quantization staging, unattributed.
- **Honesty rules:** the unattributed share is reported, never silently
  rebucketed; PyTorch-based engines (kyojin) get best-effort buckets with the
  caveat named; launch counts are reported alongside kernel time because the
  2026-10-04 record showed launch count does not explain this gap.
- **Comparability:** per-family tables compare kernels doing similar work
  across engines; they are not an equal-arithmetic comparison, and no
  cross-family ratio is a causal claim.

## 5. Engine pins (pass 1, 2026-10-12)

| Engine | Pin | Status |
| --- | --- | --- |
| hipEngine | this tree, branch `qwen3.8-flash-next` | current |
| halo-box `strix-llama.cpp` | master `6aa81b88e` (2026-10-09, PR #117) | fresh clone, HIP build in progress |
| gufo | tag `v0.12.0` (2026-10-11) | build in progress; fallback pin is prebuilt edge `98641a6` (binary `8ca63994…`, the 2026-10-04 comparison binary) |
| kyojin | main branch, `tools/qwen/serve.py` | blocked on disk: needs ~120 GiB (95 GiB EXL3 pack + ROCm-torch venv/SDK) on a 100%-full 473 GiB NVMe |

Pass-2 refreshes (newer engine releases) re-run this protocol unchanged and add
rows; they never overwrite pass-1 rows.

## 6. Artifacts and rollup

- Results live in `benchmarks/results/<date>-qwen4exp-cross-engine-*/`:
  `README.md` (tables + gate verdicts), `artifact.json` (compact), `raw/`
  (per-engine outputs, identity records, logs).
- Rollups: one dated line in `benchmarks/CHANGELOG.md`; a cross-reference row
  in `QWEN4EXP-STATUS.md` §1.2 and the optimization ledger when a comparison
  changes a decision. The root `README.md` never carries cross-engine tables.
- Anti-gaming: engines run unmodified; a patch that only exposes token ids,
  logprobs, or timings is recorded with its sha256 in the artifact; no engine
  is tuned per-prompt; no fixture prompt is special-cased anywhere.
