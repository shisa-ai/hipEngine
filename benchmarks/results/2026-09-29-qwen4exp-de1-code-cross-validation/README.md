# DE-1 code-prompt cross-validation: the candidate implementation is functionally non-inferior

The task-specific check `docs/EXECUTION-PROFILES.md` section 6.5 names - "code execution/tests ...
when the product path claims those capabilities" - applied to the one divergent prompt that produces
executable output. Correctness evidence; `performance_claim: false`.

- **Source capture:** `benchmarks/results/2026-09-29-qwen4exp-decode-dp4a-linear-tasks/artifact.json`
- **Harness:** `scripts/qwen4exp_de1_cross_validate_code.py`
- **Host:** gfx1151, machine `55ea6c509d0b49eea8de7094a1023668`
- **Candidate:** `decode_dp4a_linear` (`HIPENGINE_QWEN4_EXP_Q4_DP4A64=1`), `--max-tokens 4096`

## Method

The `long_code_booking_merge` prompt asks for "only executable Python code including focused
assert-based tests", so both arms produced a complete `reconcile` plus their own test suite. Comparing
each implementation only against the tests it wrote would be biased - an implementation with weaker
self-tests would look clean. So both are run against the **union of both arms' test cases**.

The two implementations are genuinely different: `reconcile` bodies are 2578 chars (sha `ec4e5700…`)
and 2572 chars (sha `30a8e0b1…`), not identical.

## Result

```
test                                         owner        strict  candidate
---------------------------------------------------------------------------
test_adjacency                               strict         True       True
test_cancellation_of_earlier_active          strict         True       True
test_cancelled_does_not_block                strict         True       True
test_empty_input                             strict         True       True
test_equal_revision_last_arrival             strict         True       True
test_multiple_bookings_same_room_overlap     strict         True       True
test_no_mutation                             strict         True       True
test_out_of_order_revisions                  strict         True       True
test_rejected_does_not_block                 strict         True       True
test_revision_tie_with_cancellation          strict         True       True
test_room_independence                       strict         True       True

failures:      0
disagreements: 0 []
verdict: pass
```

**11 test cases, both implementations pass all 11, zero failures, zero disagreements.** The candidate
satisfies every behavioural assertion the strict arm wrote, and strict satisfies every one the
candidate wrote.

## The harness is falsifiable

A green result is only evidence if the harness can fail. A deliberately broken implementation
(accepts everything, ignores cancellation) run against the same tests:

```
failures:      7
disagreements: 7 [test_cancellation_of_earlier_active, test_cancelled_does_not_block,
                  test_equal_revision_last_arrival, test_multiple_bookings_same_room_overlap,
                  test_out_of_order_revisions, test_rejected_does_not_block,
                  test_revision_tie_with_cancellation]
verdict: review
```

So the passing result is not vacuous.

## Why this settles the code prompt

The candidate's output is **shorter** (1938 vs 2424 tokens) and its first textual difference is at
character 423 - a comment synonym, `# Step 1: Choose winning event` vs `# Step 1: Select winning
event`. A shorter output with a comment-level difference could in principle have dropped required
behaviour; cross-validation on the union of both suites is the check that rules that out, and it
rules it out completely.

## Limits

- **One prompt, one generation per arm.** Only `long_code_booking_merge` produces executable code in
  the natural-long suite; the other three are prose.
- **The test cases are model-written, not an external oracle.** Cross-validation removes the
  self-test bias but cannot cover behaviour neither arm thought to test. The prompt's spec is fully
  covered by the union on inspection (revision precedence, tie-breaking by later arrival,
  cancellation, adjacency, room independence, rejected-not-blocking, empty input, no mutation), but
  this is not a formal proof.
- **The original source capture was invalidated** (`invalid_source_or_lifecycle`: the repo was
  committed to while the run was in flight, so `_git_metadata(ROOT) != source` at the end). It is
  **superseded by the citable re-run** at
  `benchmarks/results/2026-09-29-qwen4exp-decode-dp4a-linear-tasks-rerun/artifact.json`
  (`status: captured_requires_manual_review`, `head: 3d681e391`, `tracked_clean: true`, both arms
  returning to 0 current bytes / 0 active allocations). The re-run **reproduced the outputs
  bit-exactly** - 1938 / 431 / 633 / 240 tokens, and both generated implementations identical
  (strict sha `b8685e99…`, candidate sha `97a8ef2e…`) - and cross-validation on the re-run is
  again **11/11 with 0 disagreements**.
- **No speedup claimed.** This artifact concerns task quality only.
