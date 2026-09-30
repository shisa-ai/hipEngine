# X-7 long-form heldout: `dense_wide` at 16-47 fails at prefill→c1, on the same scope as WMMA

X-7's first missing certificate was **category-heldout numerical evidence** for the
promoted 16-47 scope. The existing heldout protocol could not produce one: the
18-prompt admission suite tops out at 71 tokens against `dense_wide`'s 257-row
floor, so every heldout run over it compared two runs of the same kernel.

This packet is the first heldout run that can reach that floor. **The evidence is
now complete and the blocker is dropped. The route does not pass the quality
gate**, and it fails on exactly the scope the WMMA route failed on.

## Artifact

`artifact-dense-wide-16-47-longform-heldout.json`

```
python3 scripts/execution_profile_q8_wmma_prefill_layers_gate.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --fixture benchmarks/fixtures/qwen4exp_longform_heldout_4prompt.json \
  --route dense_wide --layers 16-47 --decode-steps 24 --repeat-runs 3 \
  --prefill-chunk-size 1024 \
  --output <artifact>
```

Protocol `category_heldout_longform_d128_three_repeat_v1`: 4 natural long tasks,
one per category, 424–512 tokens, 24 decode steps, 3 candidate repeats, 100
teacher rows. Fixture `b2d37b8f5f8c…`, built from
`benchmarks/prompts/qwen4exp-natural-long-tasks-v1.jsonl`.

**Host: `gfx1151` (AMD Ryzen AI MAX+ 395 / Radeon 8060S), not the default W7900.**
The comparison below is against the WMMA heldout run on the *same* host, so it is
same-host and like for like.

## Engagement, checked before any KL

An identical strict/candidate hash pair is this packet family's recorded failure
mode, so it is checked first:

```
strict_logits_sha256    = 4877307044f53ff8a93e2a9abb0783924a7a0eeb9386c69d40a45a71241a34a5
candidate_logits_sha256 = 0553c7f988acf28817a4a2ecefe7dcb7a40a8943d4b9f01e6446badf03d8be29
```

They differ, and the dispatch counts confirm it independently:

| arm | `dense_wide` selector active | chain |
| --- | --- | --- |
| candidate | **1056** | `dense_wide256_f32_f32_out` |
| teacher | **0** | `exact coltile8_rowbatch4_f32_f32_out` |

The route ran 1056 times in one arm and zero times in the other. `rows_seen` is
`[1, 424, 438, 508, 512]` — the long prompts reached the guard as their real
token counts, which is what the 257-row floor is evaluated against.

All seven heldout coverage checks pass, `blocked_reason: null`, and
`longest_prompt_rows: 512` against `route_min_engaging_rows: 257`.

## The evidence is complete; the blocker is dropped

`category_heldout_evidence_complete: True` and **`category-heldout numerical
evidence` is absent from `qualification_blockers`** — the blocker is dropped by
producing the evidence, not by asserting it.

`category_heldout_qualification_claim: False`, because the claim requires the
evidence *and* a passing quality gate. Collected is not passed.

`measurement_valid: False` is expected here and is not a defect: it is bound to
the **canonical** protocol's `complete` flag, and this run used the heldout
fixture. The canonical protocol is a different fixture and a different record.

## The aggregate passes; two per-scope gates fail

| metric | value | limit | |
| --- | --- | --- | --- |
| mean KL | 4.33572e-4 | 1e-3 | pass |
| p95 | 1.68697e-3 | 5e-3 | pass |
| p99 | 5.19085e-3 | 2e-2 | pass |
| max | 6.40050e-3 | 5e-2 | pass |
| top-1 | **1.0** (100/100) | 0.99 | pass |

`hard_gates_passed: False` anyway:

| failing scope | rows | mean KL | top-1 | limit |
| --- | --- | --- | --- | --- |
| `shape=prefill_last` | 4 | **1.438974e-3** | 1.0 | 1e-3 |
| `transition=prefill_to_c1` | 4 | **1.438974e-3** | 1.0 | 1e-3 |

Both are the same four rows — one per prompt — so the drift is concentrated at
the **prefill→c1 transition**, at **1.44x the limit**. `shape=c1` (96 rows)
passes at 3.91681e-4, and every category passes:

| category | rows | mean KL | p99 | max | top-1 |
| --- | --- | --- | --- | --- | --- |
| code | 25 | 1.53613e-4 | 9.71938e-4 | 1.01853e-3 | 1.0 |
| general_en | 25 | 4.70437e-4 | 2.80933e-3 | 3.16915e-3 | 1.0 |
| general_ja | 25 | 2.33998e-4 | 1.72528e-3 | 2.01093e-3 | 1.0 |
| mixed_ja_en | 25 | 8.76241e-4 | 6.10725e-3 | 6.40050e-3 | 1.0 |

`rows_over_review_boundary: []`, `top1_mismatch_rows: []`,
`requires_outlier_review: False`, and `repeat_determinism.passed: True` across 3
runs. The failure is a threshold miss on one scope, not instability or an
outlier.

## The finding: the locus is the transition, not the route

The WMMA route at the same window, on the same host, over the short admission
suite, fails **the same two scopes**:

| | WMMA 16-47 (18 prompts, 39–71 tok) | `dense_wide` 16-47 (4 prompts, 424–512 tok) |
| --- | --- | --- |
| failing scopes | `prefill_last`, `prefill_to_c1` | **same two** |
| rows at that scope | 18 | 4 |
| mean KL at that scope | 2.06179e-3 | **1.438974e-3** |
| top-1 at that scope | 1.0 | 1.0 |
| aggregate mean KL | 2.73717e-4 | 4.33572e-4 |
| aggregate max KL | 2.70641e-2 | 6.40050e-3 |
| aggregate top-1 | 0.99556 | **1.0** |

Two different routes, at the same layer window, failing the same scope on KL
only, with **perfect top-1 at that scope in both cases**. That is evidence the
failure locus is the **16-47 window at the prefill→c1 transition**, not the
`dense_wide` rewrite specifically. `dense_wide` is the milder of the two there.

**The aggregate would have passed both times.** Every aggregate metric is inside
its limit in both runs, including a perfect top-1 here. The per-scope gate is the
only thing that localizes the defect — which is what it exists for, and it is why
a passing aggregate is not admissible as clearance.

**This does not say the routes are equivalent.** The WMMA run's aggregate max KL
is 4.2x worse (2.70641e-2 vs 6.40050e-3) and its top-1 is imperfect (0.99556 vs
1.0), so the routes differ in aggregate behaviour. They agree only on *where* the
per-scope failure is.

## What this does not establish

- **Not a `dense_wide` regression.** This is the first heldout gate this route has
  been through, so there is no prior certificate it fails to match. It is a
  newly-localized defect, not a change.
- **Not host-independent.** Measured on `gfx1151`. The arithmetic is the same
  family as `gfx1100`, but no W7900 run backs this packet.
- **Not a full profile admission.** Six further blockers stand, including
  canonical numerical coverage, strict-teacher/BF16-relative evidence, exact
  control/isolation evidence, and task quality.
- **`execution-affecting worktree provenance` is a blocker here for a reason
  outside this packet**: 28 untracked files, which are other workers' raw logs
  and profiler dumps in a shared worktree. `staged_dirty` and `unstaged_dirty`
  are both `false` and `dirty_paths` is empty, so no execution-affecting tracked
  file was modified. It is recorded as it stands rather than worked around.

## Reproduce

```bash
bash scripts/run_qwen4exp_x7_longform_heldout.sh
```

The script pins the ROCm SDK library path and puts `hipcc` on `PATH`; without
either, the run fails at `libamdhip64.so` or at the first JIT build. Note that
the gate exits non-zero when `measurement_valid` is false, so a heldout run
always exits 1 — **the exit code is not the verdict**; read the artifact.
