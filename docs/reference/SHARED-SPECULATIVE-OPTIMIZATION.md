---
status: closed
owns: Completion evidence and consumer boundaries for the October 24 shared speculative-decoding optimization pass.
---
# Shared speculative decoding optimization pass

## Scope and baseline

The October 24 lead request authorizes adopting and testing four PARO MTP
optimizations and refactoring their reusable machinery across compatible model
providers. This document records the completed implementations, measured
selections and unavailable validation lane. Existing model-specific kernels
remain behind their registered capabilities; no second speculative scheduler
or admission based on model names, paths, or measured artifacts was introduced.

Implementation starts from `1906b54d2`. The
[PARO paired baseline](../../benchmarks/results/2026-10-24-gfx1151-paro-mtp-retest/NOTES.md)
measured clean `9c24d765b`: natural-suite public decode 67.54 tok/s AR and
64.32 tok/s MTP on Framework Desktop / Radeon 8060S, packed Qwen3.6-35B-A3B
ParoQuant W4, BF16 KV, one request/candidate. At 4096 repeated input tokens,
first-token latency was 4.13 s AR and 22.31 s MTP. These are baseline observations,
not predicted gains. Refresh the immediate baseline when intervening changes
can affect a comparison.

## Completion checklist

Every row records implementation and verification evidence, an individually
measured rejection, or a concrete unavailable dependency and clearing action.
An unavailable lane is not a successful live test.

| Deliverable | Status | Completion evidence |
| --- | --- | --- |
| Map existing ownership and extension points | Complete | Source-backed consumer matrix below; existing staged-provider, native-cycle and prompt-sink contracts are the extension points |
| Shared prompt-priming orchestration | Complete for existing GGUF consumers | Common protocol, split planner and publication ledger; dense/MoE public IDs, chunks and acceptance match clean `1906b54d2`; repeated-call and teardown checks pass. PARO chunk adoption is recorded in its bulk-priming row |
| PARO KV-only prompt priming | Complete | Default historical rows stop after cache writes. Eleven-length device oracle and 294 category/heldout/control calls preserve KV/state/IDs and acceptance; [paired measurements](../../benchmarks/results/2026-10-24-gfx1151-paro-cache-only/NOTES.md) confirm lower prompt latency, flat decode |
| PARO batched prompt priming | Complete | Default 16-row chunk integration passes the 16-length/64-call live-state oracle and all 294 paired category/heldout/control calls. [Measurements](../../benchmarks/results/2026-10-24-gfx1151-paro-bulk-prompt/NOTES.md): 4K first token 6.021 → 5.094 s; natural whole-call +0.41%, confirmed +0.40%; decode effectively flat |
| Shared rollback/state-journal contract | Complete for compatible adapter-level transactions | PARO, Qwen4 and dense GGUF single-request MTP use the completed-checkpoint journal. Public failure/cancel/recovery checks pass ([Qwen4](../testing/2026-10-24-qwen4-mtp-journal/NOTES.md), [dense GGUF](../testing/2026-10-24-dense-c1-checkpoint-journal/NOTES.md)). Packed routes retain owner-local transactions rather than manufacturing a duplicate adapter undo |
| PARO bounded rollback | Complete | Mutable-state/cursor backup omits live prefix KV; public failure tests and all 294 paired calls preserve state/IDs/acceptance. [Measurements](../../benchmarks/results/2026-10-24-gfx1151-paro-rollback/NOTES.md): natural decode +0.20%, 4K +1.88%, confirmed at +0.31%/+1.90% |
| Shared reusable graph-stage ownership | Complete for GGUF NextN and PARO | Bounded cache, address/shape/configuration binding, retryable capture failures and retained cleanup ownership. [NextN public evidence](../testing/2026-10-24-nextn-shared-graph-cache/NOTES.md) and the PARO row below verify real replay, teardown and failure paths |
| PARO graph-stage adoption | Complete; graph default rejected on measured slowdown | Shared lifecycle and actual-route telemetry pass 11 device tests and 567 paired calls. [Full comparison and confirmation](../../benchmarks/results/2026-10-24-gfx1151-paro-graphs/NOTES.md) show natural graph decode 3.23%/3.06% slower than eager. Eager stays the planned default; explicit graph plans remain executable |
| Shared candidate-budget policy | Complete for staged providers | Existing `plan_speculative_requests` bounds depth by declared capability, output room, context room and resource claims before mutation; 76 policy/contract/cross-model tests pass. Timing-adaptive scheduling is a separate optional mechanism, not required for this row |
| PARO deeper candidates | Complete | Explicit depths 1/2/3, automatic preference 2. [Comparison and confirmation](../../benchmarks/results/2026-10-24-gfx1151-paro-depth/NOTES.md): pooled natural decode +0.62%/+0.32% versus depth one, but plain AR remains faster and the category split regresses. All 756 timing calls, 41 device checks, 54 final-default public requests and 75 cycle-local numerical cases pass |
| Cross-model integration | Complete on available resources; DFlash live lane unavailable | Dense/MoE GGUF, Qwen4 and PARO public evidence linked above/below; five Gemma public GPU tests and 143 Gemma/DFlash/shared CPU checks pass. DFlash resources and clearing command are listed below; no live DFlash pass is claimed |
| Regression and numerical gates | Complete within stated scope | Default unit suite and CPU deterministic bundle pass. Full suite: 20,021 passed, 6,569 skipped, two old GGUF graph fixtures repaired; affected files then pass 57 tests with two skips. Audit, docs and benchmark sync pass. Numerical evidence is cycle-local C1, not a new BF16/task-quality certificate |
| Documentation and benchmark rollup | Complete | Architecture boundaries and consumers below; per-unit commands, source hashes and measurements linked in the benchmark scoreboard, changelog and immutable worklogs. No new flag introduced by the depth extension |
| Atomic commits and final audit | Complete | Shared owners and per-candidate units committed separately; final depth/integration repair is `8e0ef3e92`. Every consumer and candidate is accounted for, including measured graph slowdown and the unavailable DFlash lane |

## Validation contract

Refactoring must preserve existing arithmetic and selected variants. Exact
control/ownership, cancellation, rollback and cleanup bind every change.
Same-arm output/state parity is the initial oracle for work elimination;
cross-route AR/MTP equality is not substituted for the production numerical
gate. Arithmetic-changing batches or kernels require the applicable
`docs/EXECUTION-PROFILES.md` evaluation, strict fallback and task checks.

Performance candidates are measured separately against their immediate parent;
do not combine four changes and attribute the aggregate to each. Capture
prompt priming, proposal, target verification, state backup/commit, host
synchronization, and public call timing. Synchronized attribution runs are
separate from timing runs. Prebuild before profiler capture.

The natural suite contains all ten `mtpbench-code-general-ja.jsonl` prompts
and all eight `gdn-prefill-category-heldouts.jsonl` prompts. Preserve the prior
512/1024/4096 controls and add lengths on both sides of any new chunk or graph
bucket boundary plus unrelated lengths. Keep output-cap tails and warm/cold
cache behavior explicit. Run a confirmation command and preserve unsuccessful
attempts. An exact performance loss is reported, not hidden by changing a gate.

For ordinary shared-code changes, run focused tests followed by the applicable
CPU deterministic bundle from `docs/TESTING.md`: compileall, default pytest,
fixture checker, registry/CPU-fixture/smoke-add-plan checks and torch-import
review. Run `python3 audit/audit.py check` for cleanup/ledger changes.
Milestone closure requires `uv run pytest --suite all -v`; unavailable hardware
and model resources must produce explicit guarded skips, not fabricated passes.
Docs use `python3 scripts/docs/check_all.py`. Benchmark exports additionally
use the README-sync test and check. Root README changes remain limited to
supported updates of the requested existing result cells.

## Source-backed consumer matrix

| Consumer | Prompt priming | Rollback | Graph stages | Candidate depth |
| --- | --- | --- | --- | --- |
| PARO | Shared bounded hidden chunks; historical rows batch KV-only priming, final row computes a full prediction | Completed-checkpoint journal owns provider and target restore; spare-slot copy contains mutable state and cursors, not live KV | Shared verifier graph ownership binds split counts and allocation identities; eager is preferred after measured graph slowdown, while graph plans execute | Public adapter supports depths 1–3 and prefers two; the shared planner bounds each cycle by output/context/resource room |
| GGUF dense | Streaming chunk sink, carried hidden row and prefix checkpoints | Shared journal owns single-request provider/target undo and retryable release; packed routes retain owner-local checkpoints | Shared proposal-graph lifecycle cache; native target stage wrappers | Shared planner consumes declared capability; standalone benchmark controllers use a typed session seam |
| GGUF MoE | Request-sized target-hidden and draft-KV buffers; shared prompt chunk ledger | Transactional target and provider checkpoints | Compatible native graph buckets; temporary boundary transitions resume MTP | Shared planner selects the deepest legal provider/bucket depth |
| Qwen4 | Bounded cache-only prompt workspace with final full predictor row | Reusable device transaction snapshot and KV/index cursors | Model-specific execution, not currently a native-cycle graph consumer | Provider supports chains up to four candidates |
| Gemma | Assistant starts from target hidden; no historical draft KV cache | Attention cursors rewind; no device-state copy to eliminate | One-token AR graph owner is not used by multi-row MTP verification | Non-staged provider bounds its configured depth to 1–8 and each cycle to remaining outputs minus one |
| Generic/DFlash | Position-addressed target-hidden appends use a one-row capture owner; chunk priming would require a multi-row owner | Accepted-prefix append and cursor ownership; failure closes both owners rather than restoring a journal | DFlash bucket metadata owns no replay handle to migrate | Laguna uses its fixed four-candidate block; do not substitute an MTP depth ladder |

The first extraction moved the existing `TargetHiddenChunkSink` contract out of
`hipengine/runtime/qwen35_gguf_runner.py` and centralized chunk validation,
splitting and publication for `_StreamingNextNPromptSink` in
`hipengine/runtime/qwen35_gguf_mtp.py` and `_MoeTargetHiddenSink` in
`hipengine/generation/qwen35_gguf_moe_mtp2.py`. PARO now consumes the same
chunk contract for bounded historical priming and final predictor publication.
Model-specific hidden alignment and all kernel launches stay in their providers.

Existing shared owners remain authoritative: `hipengine/speculative/provider.py`
for the staged provider contract, `hipengine/speculative/prompt_prime.py` for
prompt chunk validation/publication, `hipengine/speculative/transaction.py` for
cycle results and the state journal, `hipengine/speculative/graph_stage.py` for
bounded graph-cache ownership, `hipengine/speculative/native_cycle.py`
and `hipengine/speculative/native_cycle_graph.py` for device stages and graph
binding, and `hipengine/speculative/policy.py` for capability-driven candidate
selection. `hipengine/speculative/mtp_budget.py` holds the observation and
policy protocol for a dense-session diagnostic seam. Concrete adaptive and
sequence controllers live in `scripts/mtp_budget_diagnostics.py`; runtime does
not import them, and they do not select public serving defaults.
These owners provide the common machinery without parallel engines.
The state journal has real PARO, Qwen4 and dense GGUF consumers; it is not an
unused rollback record.

## Integration results and unavailable lane

- The Gemma/DFlash/shared CPU integration bundle passes 143 tests. All five
  Gemma public greedy, sampled, request-reuse and resident-session GPU checks
  pass with the target and assistant under
  `/models/gguf/gemma-4-26B-A4B-it-GGUF`:
  `HIPENGINE_TEST_GEMMA4_MODEL_DIR=/models/gguf/gemma-4-26B-A4B-it-GGUF python -m pytest tests/test_live_gemma4_generate_mtp_flag.py tests/test_live_gemma4_resident_session.py -v -o addopts=''`
  through the [ROCm environment wrapper](../../benchmarks/results/2026-10-24-gfx1151-paro-graphs/environment.sh),
  with `PYTHONPATH` set to the current repository.
- The Laguna/DFlash public lane lacks its target
  `~/models/gguf/laguna-s-2.1-Q4_K_M.gguf`, its source-bound
  `.hipengine-repacked-v1` sibling, and the drafter snapshot
  `~/.cache/huggingface/hub/models--poolside--Laguna-S-2.1-DFlash/snapshots/b0486d1586daa0d56435c508108171fc1c8daff9`.
  After supplying those resources, run
  `uv run python scripts/laguna_dflash_public_gate.py <target.gguf> <drafter-snapshot-dir> --backend hip_gfx1151`.
  The synthetic provider tests pass; no live DFlash result is claimed.

- The GGUF MoE context repair now passes 22 greedy and four sampled public calls,
  including short/split-attention boundaries and prompts beyond 1,024 tokens.
  Prompt and draft buffers grow from request capacity; graph-boundary steps
  resume speculation instead of permanently disabling it. A named explicit
  context refusal is contained and the next request succeeds. See the
  [capacity and recovery evidence](../testing/2026-10-24-moe-mtp-context-capacity/NOTES.md).
  This closes the observed 95-token admission/sink defect. The final
  cross-model regression result is recorded in the checklist.
- A repeated one-token PARO MTP request exposed stale full-attention scratch
  views: the one-row projection writes a compact Q/K layout, while a cached
  wider view reads K at its old offset. Exact-row scratch rebinding fixes the
  device trace's live KV, recurrent state and output mismatch. Commit
  `ac99757d0` includes the repair and CPU regression; the 11-length public
  oracle and refreshed default CPU suite pass. This is separate from cache-only
  work elimination and makes no performance claim.

## Implementation boundaries

The shared layer owns lifecycle, bounded scheduling, cache/graph ownership,
and policies over declared capabilities. Providers retain architecture-specific
hidden-state alignment, cache representation, numerical kernels and legal
rollback semantics. A method cannot inherit a technique whose semantics it does
not implement; it inherits common orchestration once those operations exist.
Prompt sinks, adapter journals and graph-stage caches now have compatible
consumers, while Gemma and DFlash retain their distinct semantics. This pass
changes no kernel arithmetic and adds no backend/quant identity branches to the
shared planner. GPU evidence for the PARO candidates covers gfx1151; gfx1100
was not re-measured in this pass.
