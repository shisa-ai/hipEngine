---
status: current
owns: Public-route and lifecycle verification for the shared GGUF NextN proposal graph cache.
---
# Shared NextN graph cache verification

This change moves proposal-graph lookup, binding checks, eviction and retirement
into `SpecGraphStageCache`. NextN still owns kernel capture and HIP handles. No
kernel arithmetic changes and no performance claim are part of this unit.

## Checks

The public driver calls `LLM.generate_detailed()` with dense Qwen3.8-27B
Q4_K_M, BF16 KV, one request, greedy sampling, 32 output tokens and prompts of
37, 63 and 257 repeated tokens. Each length runs twice. Instrumentation records
actual proposal-graph returns and replay counts, not target-graph counters.
The parent artifact is from `b9fef4625` on the same gfx1151 host.

The first candidate passed all six calls with parent-identical output IDs,
proposal replay and zero tracked allocations after close. Review then expanded
the binding to intermediate scratch and weight allocations, and added retained
partial-capture cleanup. This exposed a real regression: a repacked weight has
named allocations rather than a `raw` allocation. The invalid binding raised
`KeyError: 'raw'` during prompt preparation and the public test caught the
resulting absence of proposal execution. The binding now records every named
allocation; a CPU regression test uses the real device-weight class with no raw
allocation. Final public revalidation passed all six calls with parent-identical
IDs, actual proposal replay, no binding errors, no pending graph or partial
releases and zero tracked allocations after close.

The default CPU suite completed with two failures in MoE boundary tests being
added concurrently; all other nodes passed. The repaired affected bundle passed
190 tests with one explicit GPU skip. The final graph/NextN focused bundle passed
61 tests, including RED/GREEN regressions for repacked weights, blocking further
capture allocations after cleanup failure, and propagating a graph-destruction
interrupt after cleaning up other resources. The interrupt repair followed the
final GPU run and changes only that failure path. Compileall, fixture validation,
registry smoke, CPU fixtures, build-plan smoke and the audit gate also pass.

CPU coverage checks address/configuration drift, bounded diagnostic history,
retry after capture failure, eviction, partial cleanup, and repeated teardown
without double-destroy. Destroyed handles are zeroed. Failed releases remain
owned and are retried; unresolved cleanup reaches the executor owner. New
captures stop while failed partial-capture releases remain outstanding.

## Evidence and reproduction

`protocol.json` identifies the parent, candidate source hashes and configuration.
`public_check.py` is the public driver. `parent.json` is the supported parent
run; `candidate-before-final-review.json` records the initial passing candidate.
`invalid-eos-request.json` records the diagnostic's unsupported `ignore_eos=True`
request, which failed before execution; later runs use the default EOS behavior.
`failed-raw-weight-binding.json` and `binding-diagnostic.json` preserve the failed
binding runs. `candidate.json` records the repaired six-call result. Use the
archived environment wrapper and command in `protocol.json` to reproduce it.
The protocol distinguishes GPU-tested source hashes from the later interrupt-only
cleanup repair, which has focused CPU coverage.

The test validates deterministic repeated output and resource ownership. It does
not assess task quality or claim a speedup for these repeated-token prompts.
