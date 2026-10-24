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
| Shared prompt-priming orchestration | Pending | Explicit causal input/hidden alignment, chunk lifetime and final-row semantics; at least two real consumers; cancellation and reuse tests |
| PARO KV-only prompt priming | Pending | Compare cached K/V and final draft state to current full priming; remove discarded attention/expert work; public short/long calls |
| PARO batched prompt priming | Pending | Bounded row workspace; compare serial and bulk cache/final outputs; numerical gate if arithmetic changes; paired prompt-latency measurements |
| Shared rollback/state-journal contract | Pending | Separate mutable state, append-only KV and cursors; atomic target/provider rollback, failure/cancel/reclaim tests; migrate compatible consumers |
| PARO bounded rollback | Pending | Eliminate live-prefix copying only where append-only semantics hold; preserve recurrent-state rollback and output IDs; measure cycle/copy work and decode |
| Shared reusable graph-stage ownership | Pending | Stable address/shape keys, retirement/invalidation, eager fallback and capture-failure cleanup; reuse existing native-cycle interfaces |
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
| PARO | Row callbacks run a full draft block; contiguous target hidden slabs already exist | Spare-slot copy includes the live KV prefix and mutable recurrent state | Native provider-target ABI exists; public adapter forces eager verification | Public adapter exposes one candidate; underlying verifier supports larger row batches |
| GGUF dense | Streaming chunk sink, carried hidden row and prefix checkpoints | Recurrent/hidden journals and append-only KV cursors | Native proposal and target stage wrappers | Shared planner consumes declared capability; standalone adaptive policy also exists |
| GGUF MoE | Target-hidden chunk sink; NextN also has a restricted bulk route | Transactional target and provider checkpoints | Uses compatible GGUF execution pieces, with route-specific capabilities | Adapter-specific bounds need reconciliation with actual provider support |
| Qwen4 | Bounded cache-only prompt workspace with final full predictor row | Reusable device transaction snapshot and KV/index cursors | Model-specific execution, not currently a native-cycle graph consumer | Provider supports chains up to four candidates |
| Gemma | Assistant starts from target hidden; no historical prompt-cache priming to transfer | Attention cursors rewind; no Qwen-style recurrent-state journal | Model-specific decode graphs | Assistant generation supports up to eight candidates |
| Generic/DFlash | Provider-specific context construction, not a target-hidden MTP prompt stream | Existing append plans and provider transactions | Existing DFlash bucket metadata and native-stage infrastructure where integrated | Provider-declared depths; do not substitute an MTP depth ladder |

The first extraction moves the existing `TargetHiddenChunkSink` contract out of
`hipengine/runtime/qwen35_gguf_runner.py` and centralizes chunk validation,
splitting and publication for `_StreamingNextNPromptSink` in
`hipengine/runtime/qwen35_gguf_mtp.py` and `_MoeTargetHiddenSink` in
`hipengine/generation/qwen35_gguf_moe_mtp2.py`. PARO can then consume the same
chunk contract rather than retaining a second row-only orchestration path.
Model-specific hidden alignment and all kernel launches stay in their providers.

Existing shared owners remain authoritative: `hipengine/speculative/provider.py`
for the staged provider contract, `hipengine/speculative/transaction.py` for
cycle results and atomic ownership, `hipengine/speculative/native_cycle.py`
and `hipengine/speculative/native_cycle_graph.py` for device stages and graph
binding, and `hipengine/speculative/mtp_budget.py` for candidate policy. The
later units should extend these owners rather than introduce parallel engines.
A generic rollback-domain record alone would not eliminate any copy; require a
real consumer migration and tests before calling that row complete.

## Implementation boundaries

The shared layer should own lifecycle, bounded scheduling, cache/graph ownership,
and policies over declared capabilities. Providers retain architecture-specific
hidden-state alignment, cache representation, numerical kernels and legal
rollback semantics. A method cannot inherit a technique whose semantics it does
not implement; it should inherit the common orchestration once those operations
are supplied. The first shared extraction preserves numerical calls and migrates
the existing dense and MoE GGUF sinks before extending PARO's chunk interface.
