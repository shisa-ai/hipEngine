---
status: current
owns: Active completion matrix and validation plan for shared speculative-decoding optimizations and their PARO adoption.
---
# Shared speculative decoding optimization pass

## Scope and baseline

The October 24 lead request authorizes adopting and testing four PARO MTP
optimizations and refactoring their reusable machinery across compatible model
providers. This document tracks unfinished work; it is not a claim that the
techniques are implemented or faster. Existing model-specific kernels remain
behind their registered capabilities. Do not introduce a second speculative
scheduler or admission based on model names, paths, or measured artifacts.

Implementation starts from `1906b54d2`. The
[PARO paired baseline](../../benchmarks/results/2026-10-24-gfx1151-paro-mtp-retest/NOTES.md)
measured clean `9c24d765b`: natural-suite public decode 67.54 tok/s AR and
64.32 tok/s MTP on Framework Desktop / Radeon 8060S, packed Qwen3.6-35B-A3B
ParoQuant W4, BF16 KV, one request/candidate. At 4096 repeated input tokens,
first-token latency was 4.13 s AR and 22.31 s MTP. These are baseline observations,
not predicted gains. Refresh the immediate baseline when intervening changes
can affect a comparison.

## Completion checklist

A row closes only with implementation and verification evidence, an individually
measured rejection, or a concrete unavailable dependency and clearing action.
An untested idea is not a rejected candidate. Keep all rows active across units.

| Deliverable | Status | Completion evidence required |
| --- | --- | --- |
| Map existing ownership and extension points | Complete | Source-backed consumer matrix below; existing staged-provider, native-cycle and prompt-sink contracts are the extension points |
| Shared prompt-priming orchestration | Complete for existing GGUF consumers | Common protocol, split planner and publication ledger; dense/MoE public IDs, chunks and acceptance match clean `1906b54d2`; repeated-call and teardown checks pass. PARO chunk adoption remains in its bulk-priming row |
| PARO KV-only prompt priming | Complete | Default historical rows stop after cache writes. Eleven-length device oracle and 294 category/heldout/control calls preserve KV/state/IDs and acceptance; [paired measurements](../../benchmarks/results/2026-10-24-gfx1151-paro-cache-only/NOTES.md) confirm lower prompt latency, flat decode |
| PARO batched prompt priming | Complete | Default 16-row chunk integration passes the 16-length/64-call live-state oracle and all 294 paired category/heldout/control calls. [Measurements](../../benchmarks/results/2026-10-24-gfx1151-paro-bulk-prompt/NOTES.md): 4K first token 6.021 → 5.094 s; natural whole-call +0.41%, confirmed +0.40%; decode effectively flat |
| Shared rollback/state-journal contract | In progress | Completed-checkpoint capture/restore journal adopted by PARO and Qwen4 single-request MTP; public device failure/cancel/recovery checks pass ([Qwen4 evidence](../testing/2026-10-24-qwen4-mtp-journal/NOTES.md)). Other compatible consumer migrations remain |
| PARO bounded rollback | Complete | Mutable-state/cursor backup omits live prefix KV; public failure tests and all 294 paired calls preserve state/IDs/acceptance. [Measurements](../../benchmarks/results/2026-10-24-gfx1151-paro-rollback/NOTES.md): natural decode +0.20%, 4K +1.88%, confirmed at +0.31%/+1.90% |
| Shared reusable graph-stage ownership | Complete for GGUF NextN; PARO adoption pending | Bounded cache, address/shape/configuration binding, retryable capture failures and retained cleanup ownership. [Public evidence](../testing/2026-10-24-nextn-shared-graph-cache/NOTES.md): six candidate calls match the parent and actually replay proposal graphs; teardown and focused failure tests pass |
| PARO graph-stage adoption | Pending | Exercise actual public graph route; graph/eager state and output checks, context bucket transitions, profiler trace, paired timing |
| Shared candidate-budget policy | Pending | Capability-driven depths, output-cap tails, per-request resource bounds; no model/quant identity switches |
| PARO deeper candidates | Pending | Implement and compare one/two/three candidates where supported; full category/heldout acceptance and timing; partial reject, cancel and rollback correctness |
| Cross-model integration | Pending | PARO, GGUF dense/MoE, Qwen4, Gemma and generic/DFlash consumer accounting; public tests on available hardware; explicit unavailable lanes |
| Regression and numerical gates | Pending | Focused RED/GREEN tests, CPU deterministic bundle, audit gate, appropriate GPU/profile tests and final integration coverage |
| Documentation and benchmark rollup | Pending | Architecture boundary, technique/consumer matrix, measured results, exact commands, immutable worklogs, cleanup ledger for any temporary flags |
| Atomic commits and final audit | Pending | Each complete validated unit committed; every row reconciled with evidence and no hidden unfinished work |

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
| PARO | Shared bounded hidden chunks; historical rows batch KV-only priming, final row computes a full prediction | Completed-checkpoint journal owns provider and target restore; spare-slot copy contains mutable state and cursors, not live KV | Native provider-target ABI exists; public adapter forces eager verification | Public adapter exposes one candidate; underlying verifier supports larger row batches |
| GGUF dense | Streaming chunk sink, carried hidden row and prefix checkpoints | Recurrent/hidden journals and append-only KV cursors | Shared proposal-graph lifecycle cache; native target stage wrappers | Shared planner consumes declared capability; standalone adaptive policy also exists |
| GGUF MoE | Target-hidden chunk sink; NextN also has a restricted bulk route | Transactional target and provider checkpoints | Uses compatible GGUF execution pieces, with route-specific capabilities | Adapter-specific bounds need reconciliation with actual provider support |
| Qwen4 | Bounded cache-only prompt workspace with final full predictor row | Reusable device transaction snapshot and KV/index cursors | Model-specific execution, not currently a native-cycle graph consumer | Provider supports chains up to four candidates |
| Gemma | Assistant starts from target hidden; no historical prompt-cache priming to transfer | Attention cursors rewind; no Qwen-style recurrent-state journal | Model-specific decode graphs | Assistant generation supports up to eight candidates |
| Generic/DFlash | Provider-specific context construction, not a target-hidden MTP prompt stream | Existing append plans and provider transactions | Existing DFlash bucket metadata and native-stage infrastructure where integrated | Provider-declared depths; do not substitute an MTP depth ladder |

The first extraction moves the existing `TargetHiddenChunkSink` contract out of
`hipengine/runtime/qwen35_gguf_runner.py` and centralizes chunk validation,
splitting and publication for `_StreamingNextNPromptSink` in
`hipengine/runtime/qwen35_gguf_mtp.py` and `_MoeTargetHiddenSink` in
`hipengine/generation/qwen35_gguf_moe_mtp2.py`. PARO now consumes the same
chunk contract for bounded historical priming and final predictor publication.
Model-specific hidden alignment and all kernel launches stay in their providers.

Existing shared owners remain authoritative: `hipengine/speculative/provider.py`
for the staged provider contract, `hipengine/speculative/transaction.py` for
cycle results and atomic ownership, `hipengine/speculative/native_cycle.py`
and `hipengine/speculative/native_cycle_graph.py` for device stages and graph
binding, and `hipengine/speculative/mtp_budget.py` for candidate policy. The
later units should extend these owners rather than introduce parallel engines.
A generic rollback-domain record alone would not eliminate any copy; require a
real consumer migration and tests before calling that row complete.

## Findings requiring integration follow-up

- The clean baseline's GGUF MoE C1 route silently leaves explicit MTP requests
  on ordinary decoding once its existing draft-context bound is exceeded.
  A 94-token prompt reaches the prompt sink but performs zero MTP cycles;
  a 1,025-token prompt does not reach the 95-row sink buffer. A 37-token
  prompt performs MTP cycles. These are observed pre-existing route limits,
  not regressions from the shared ledger. The cross-model integration row
  stays open until compatible requests execute and genuinely unsupported
  requests fail with a named capability error rather than silently downgrading.
- A repeated one-token PARO MTP request exposed stale full-attention scratch
  views: the one-row projection writes a compact Q/K layout, while a cached
  wider view reads K at its old offset. Exact-row scratch rebinding fixes the
  device trace's live KV, recurrent state and output mismatch. Commit
  `ac99757d0` includes the repair and CPU regression; the 11-length public
  oracle and refreshed default CPU suite pass. This is separate from cache-only
  work elimination and makes no performance claim.

## Implementation boundaries

The shared layer should own lifecycle, bounded scheduling, cache/graph ownership,
and policies over declared capabilities. Providers retain architecture-specific
hidden-state alignment, cache representation, numerical kernels and legal
rollback semantics. A method cannot inherit a technique whose semantics it does
not implement; it should inherit the common orchestration once those operations
are supplied. The first shared extraction preserves numerical calls and migrates
the existing dense and MoE GGUF sinks before extending PARO's chunk interface.
