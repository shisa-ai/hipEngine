---
status: current
owns: gfx1151 dense raw-IQ decode/verify WIP scope, evidence map, qualification gates, and next steps.
---
# gfx1151 dense IQ decode/verify WIP

State: **work in progress; not qualified for promotion**.

This branch extends existing IQ support with a gfx1151 dense raw-IQ decode and
rows-2–4 verify policy. It is not the initial IQ-support implementation. The
branch is based on pushed `main` commit `91f56a7c8` and keeps production `main`
unchanged.

## Existing support

hipEngine already recognizes raw IQ quant types and has IQ kernel families for
GEMV, selected experts, grouped paths, and dense prefill. The current gfx1151
main path declares dense-IQ prefill routing. Strict GEMV fallback remains the
route when no dense-IQ session is active, the shape is outside the policy, or a
strict slot pin applies.

The existing gfx1100 dense-IQ policy is the source of the restored gfx1151
policy shape. The two backends share the relevant kernel owners through the
backend registration structure; this branch does not claim that shared owners
are automatically qualified on gfx1151.

## Restored WIP unit

| Path | Purpose |
| --- | --- |
| `hipengine/kernels/hip_gfx1151/__init__.py` | Adds gfx1151 dense-IQ decode/verify policy tables and strict-slot pins matching the shared arithmetic family. |
| `hipengine/runtime/gguf_linear.py` | Applies backend-declared IQ decode/verify policy, session ownership, slot pins, shape checks, and dispatch-cache state. |
| `scripts/gguf_iq_local32_decode_gate.py` | Teacher-forced incumbent-versus-candidate correctness gate; supports selecting the gfx1151 backend. |
| `scripts/gguf_ud_combined_stack_gate.py` | Related UD artifact route gate. |
| `scripts/gguf_iq_dense_decode_ab.py` | Same-process alternating incumbent/candidate performance A/B; refuses to report when both arms resolve the same owner. |
| `tests/test_unit_gfx1151_iq_dense_policy_parity.py` | Policy parity, registration, strict-pin, and dispatch reachability checks. |
| `tests/test_unit_gguf_linear_dispatch_cache.py` | Dispatch-cache and IQ owner behavior coverage. |

## Evidence status

The restored policy is based on existing gfx1100/shared-owner evidence recorded
in the backend source and prior IQ gate history. The focused policy/dispatch
suite currently passes **22 tests** on this branch.

The following evidence is still missing for gfx1151 and is intentionally not
claimed:

- teacher-forced correctness gate on the real gfx1151 model/artifact;
- holdout/category coverage for all declared dense-IQ quants;
- decode and rows-2–4 verify qualification under the gfx1151 execution profile;
- per-slot strict-pin validation against every target artifact used in the gate;
- same-host performance A/B from `gguf_iq_dense_decode_ab.py`;
- public `LLM.generate()` route confirmation;
- any production-default promotion decision.

No IQ benchmark result is exported by this WIP branch. The A/B script must
record model, quant, backend, host, compiler/cache identity, exact command,
owner identities, correctness gate, and repeated timing before any performance
claim is made.

## Next steps

1. Run the teacher-forced gfx1151 gate on the declared real artifact and holdout
   prompts; preserve all failures.
2. Run the policy parity suite with the exact backend/package state used by the
   gate and verify strict pins against the loaded artifact.
3. Run the alternating incumbent/candidate A/B only after correctness passes;
   require distinct owner resolution and report coefficient of variation.
4. Exercise the route through the public generation surface, not only direct
   dispatch tests.
5. If the correctness, profile, and public-route gates pass, update the IQ
   campaign evidence and review promotion separately. Until then, keep this
   branch WIP and do not merge its policy into `main`.

This document records implementation state and missing evidence; it does not
create an IQ model allowlist or change product admission behavior.
