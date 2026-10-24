---
status: closed
owns: Public-route, boundary and recovery evidence for GGUF MoE MTP context-capacity repair.
---
# GGUF MoE MTP context capacity

The MoE adapter now sizes prompt-hidden and draft-KV buffers from request extent
and allocated target capacity instead of fixed 95-row and 1,024-row buffers.
The native verifier's declared capability bounds execution. An explicit request
with no candidate context room fails by name rather than silently disabling MTP.

## Results

On Framework Desktop / Radeon 8060S (`gfx1151`), Qwen3.6-35B-A3B Q4_K_M, BF16 KV,
one request and 16 output tokens:

- All 22 greedy calls pass at prompt lengths 37, 94, 1021, 1023, 1024, 1025,
  1100, 1278, 1279, 1280 and 1531. Each length runs twice with identical IDs.
  Every request executes MTP and records no recoverable cycle failure.
- Four sampled calls pass at lengths 94 and 1025, temperature 0.7, seed 123.
  Both repetitions have identical IDs and use the native sampled MTP route.
- An injected 128-position capability limit refuses a 127-token explicit request
  with `MoePromptContextUnsupportedError`. Failure is contained to that request;
  a subsequent 37-token request reproduces its pre-failure IDs.
- Each completed run closes with zero tracked allocations.

These are route, repeatability and ownership checks, not performance or task-quality
claims. Parent-compatible short requests preserve their output IDs. Newly enabled
long requests are not claimed identical to the parent's silent ordinary-decode path.

## Boundary repair

The initial candidate passed twice at 37, 63, 94, 95, 96 and 257 tokens, then the
1,023-token request exposed a second defect: a target-graph bucket miss disabled
MTP for the rest of the request. `failed-boundary.json` preserves that attempt.

The final adapter asks for native split attention above the short-context bucket
and advertises the deepest candidate count that fits a legal graph bucket. Where
even one candidate cannot fit, ordinary decoding advances the request and draft
state until a speculative cycle fits again. It does not permanently disable MTP.
The greedy boundary results verify resumed graph cycles without recovered-failure
records. The capacity ceiling itself is not reduced.

## Allocation and cleanup

Replacement slabs are charged before allocation. If any allocation fails, earlier
new buffers are freed and the existing slab and charge remain intact. Live owners
prevent freeing a slab they still use. Every draft write checks its actual allocated
row bound. The shared dense-KV allocator also frees its key buffer if value-buffer
allocation fails; that separate failure-path repair covers its other callers.

Focused tests inject allocation failures, resource-budget refusals, invalid writes,
live-owner conflicts and unsupported explicit requests. The final MoE/native-cycle
bundle passes 150 tests with one guarded GPU skip. The completed default CPU run
had only two concurrently added MoE fixtures lacking backend metadata; those were
repaired and the affected 190-test bundle passed with one GPU skip. Fixture,
registry, CPU-fixture, build-plan and audit checks pass.

## Reproduction

`protocol.json` records source hashes and the exact greedy command. Use
`public_check.py --lengths 94 1025 --temperature 0.7 --seed 123` with the same
model and environment wrapper for sampled coverage. `recovery_check.py` runs the
injected refusal and subsequent-request check. Both scripts use the public API.

`candidate-greedy.json`, `candidate-sampled.json` and `recovery.json` contain
results. `parent-short.json` preserves the short parent comparison. The protocol
distinguishes tested source from the later shared-allocator failure-path repair;
that repair has RED/GREEN CPU coverage and does not change successful allocation.
