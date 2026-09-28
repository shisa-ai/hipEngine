# Dense Q8 natural-task screen

The candidate is active in host prefill calls for every long task and repeat,
but this screen does **not pass task qualification**. Both arms truncate the
code response and contradict the supplied retention rule for backup E.
No production default changes and no performance claim follow from this run.

## Setup

- Source: `576363cbd50e130fde2a942f73d0564718d36b37`, tracked tree clean.
- Host: `gfx1151`, machine `55ea6c509d0b49eea8de7094a1023668`, Radeon 8060S;
  TheRock10 / HIP7.15.26333. Full host and model identity are in [artifact.json](artifact.json).
- Model: unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`.
- Arms: strict and `production_dense_q8_restore`; four supplemental natural
  tasks, two repeats, 2048-token output cap, 1024-token input budget/chunk.
- Fixture and criteria: [natural long tasks](../../prompts/qwen4exp-natural-long-tasks-v1.md).
  Canonical suite completeness is false. No prompt was revised after outputs.

| Task | Input tokens | Strict output | Candidate output | Result |
|---|---:|---|---|---|
| Booking merge | 438 | 2048, length cap | 2048, length cap | Incomplete, identical prefixes; no code executed |
| Release decision | 508 | 448, EOS | 418, EOS | Core facts preserved; both omit 15-minute policy duration |
| Japanese retention | 424 | 633, EOS | 370, EOS | Both contradict backup E's retention status before later correcting it |
| Mixed-language acceptance | 512 | 321, EOS | 245, EOS | Both reject release and cover missing tests, UI state and retry key; candidate leaves array-order preservation implicit |

All eight arm/case pairs are identical across their two repeats. This is
repeatability, not correctness. Removing Markdown list/emphasis markers and
EOS before whitespace counting gives 248/212 words for the English outputs;
the 250-word cap is not the blocker. The Japanese conclusion calls E a deletion
target even though E is only 45 days into a 90-day retention period. Corrections
later in each response do not remove the contradictory conclusion. Strict's
failure does not excuse the candidate's failure or prove candidate-specific
numerical degradation.

## Engagement and lifecycle

Every candidate prefill records 398 host calls to
`iu8_wmma_prefill_f32_f32_out`, across nine `(rows,K,N)` geometries. The total is
3184 calls across eight prefills. Candidate-prefill-wrapper calls are zero in
the first and later decode phases; this is not evidence that decode did not
execute. These counts lack GPU and layer/role attribution.

The log warns that dense-wide cannot replace the selected raw IU8 family for
an eligible shape. This is not a dense-wide qualification run. Both arms close
with zero tracked live allocations. Peak tracked allocations are about 84 GB;
no capacity or throughput claim is made.

## Reproduction

Use the environment exports in the
[scratch-growth packet](../2026-09-28-qwen4exp-scratch-growth/README.md), including
cached-build enforcement and the compiler-version file. Then run:

```bash
"$ENV_PREFIX/bin/python" scripts/qwen4exp_q8_repair_tasks.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --output /tmp/qwen4exp-natural-long-dense-q8.json \
  --candidate production_dense_q8_restore --max-tokens 2048 \
  --extra-prompts benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl \
  --prompt-id long_code_booking_merge \
  --prompt-id long_general_en_release_decision \
  --prompt-id long_general_ja_retention_review \
  --prompt-id long_mixed_ja_en_acceptance
```

The compact artifact stores one full output per arm/case plus hashes, token
counts and finish reasons for both repeats. The original local capture hash,
suite hashes, candidate manifest and lifecycle counters are preserved.

## Follow-up

Increase the output cap only for the truncated code task; preserve the
completed task evidence rather than repeating the whole screen. Do not revise
fixtures or relax criteria to erase shared failures. GPU route attribution,
full numerical/profile gates and complete canonical coverage remain open.
X-11 is not closed by these host-call observations.
