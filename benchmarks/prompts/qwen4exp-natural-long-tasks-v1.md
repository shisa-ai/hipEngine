# Natural long tasks for candidate engagement

`qwen4exp-natural-long-tasks-v1.jsonl` adds four independently specified tasks
for testing candidates whose prefill paths require more than 256 rows.
Inputs contain task requirements and source facts, not repeated padding.
These tasks supplement, never replace, the canonical category suite and
heldouts. No model output was used to select or revise them before this version
was committed. Passing these tasks alone does not qualify a candidate.

## Input lengths

Measured with `Qwen4ExpGGUFTokenizer.from_gguf_info(GGUFReader(file).info)` and
`scripts.gguf_mtp_bench.build_chat_prompt`, using
`/home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf`:

| ID | Category | Chat-formatted tokens |
|---|---|---:|
| long_code_booking_merge | code | 438 |
| long_general_en_release_decision | general_en | 508 |
| long_general_ja_retention_review | general_ja | 424 |
| long_mixed_ja_en_acceptance | mixed_ja_en | 512 |

These lengths establish input size, not kernel engagement. Actual dispatch
rows, route/layer ownership, GPU execution and the first decode transition
must be checked separately. Inputs fit the collector's default 1024-token
budget without chunking into multiple prefill calls.

## Predeclared review criteria

Review both strict and candidate outputs. Require complete EOS outputs and
repeatability, all factual/functional constraints below, and no per-task
quality degradation relative to strict. Truncation is incomplete evidence,
not a pass. A strict failure does not excuse a candidate failure. Do not change
inputs or criteria to accommodate the observed output.

- **Booking merge:** choose highest revision, then last input on revision ties,
  before removing cancellations. Sort survivors by `(start, end, booking_id)`.
  Accept non-overlapping half-open intervals per room; rejected bookings never
  reserve time. Return ordered accepted/rejected IDs without mutating input.
  Require the requested tests, including A/C accepted and B rejected, empty
  input, revisions, cancellations, adjacency, room independence and deep-copy
  equality. Inspect code before executing it in a bounded isolated test process;
  never execute generated code with access to credentials or the working tree.
- **Release decision:** no expansion; new build still has 20% east exposure.
  Preserve cohort/time/payload/build evidence and investigate payload mix.
  Do not claim causation, completed rollback or recovery. Keep the controller
  pause distinct from removing traffic. Avoid an unbounded connection increase;
  name missing causal and recovery evidence and retain policy thresholds.
  Require the four requested sections and fewer than 250 words.
- **Japanese retention:** A and D are deletion targets, not confirmed deleted;
  B is held pending investigation-owner approval; C and E are retained. D's
  seven-day limit does not apply to C. A normal-region deletion does not imply
  backup deletion. Require Japanese and the three specified headings; distinguish
  proposal, approval, scheduled work and execution/audit evidence.
- **Mixed-language acceptance:** do not approve release. Only sequential retry
  and object-key reordering have evidence. Concurrent requests, conflicting
  body/409, account isolation and restart durability remain unverified; test
  array-order preservation too. `queued` plus HTTP 202 is not download-ready.
  Recover an original timed-out operation using the same key and body, not a
  fresh key. Require Japanese prose, specified headings, and preserved API and
  status names.

## Collector use

Add `--extra-prompts benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl`
to `scripts/qwen4exp_q8_repair_tasks.py`. Without `--prompt-id` this includes
canonical prompts and all supplements. Selecting just the four IDs above is
an engagement screen, not canonical qualification; the report must retain
`canonical_suite_complete=false`.

The collector does not automatically grade these criteria. Host-call counters
also do not certify graph-replayed GPU execution. Numerical, control/lifecycle,
BF16-relative and performance gates remain separate requirements.
