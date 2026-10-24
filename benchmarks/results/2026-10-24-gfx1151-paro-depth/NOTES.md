# PARO candidate depths on Radeon 8060S

## Result and selection

Two candidates have a small pooled natural-prompt decode advantage over one:
64.571 → 64.970 tok/s (+0.62%), confirmed at 64.674 → 64.879 (+0.32%).
Whole-call throughput improves +0.44% and +0.31%. The automatic MTP preference
is two candidates; explicit depths one and three remain executable.

This is not a general MTP speedup over plain autoregressive decoding (AR).
AR is faster on the pooled natural suite in both commands. Depth two also
regresses the category split while improving the heldouts. Depth three helps
repeated-token length controls but is not the natural-suite leader. No
prompt-conditioned selection or new flag was added.

## Workload and provenance

- Physical host `gfx1151`: Framework Desktop, AMD Radeon 8060S Graphics.
- Packed Qwen3.6-35B-A3B ParoQuant W4 with BF16 MTP head; `w4_paro`, BF16 KV,
  one active request, production kernel arithmetic and eager verification.
- Artifact fingerprint `362ac5e97b22ed60e9c02a9b3b2341ad8ebedbc79b037289404812097d65e9d7`;
  full model/toolchain/environment provenance is in each timing JSON.
- All 10 category and 8 heldout prompts, plus 9 repeated-token length controls;
  128 greedy output tokens, ignored EOS, prefix cache off, capacity 4,352.
- Four rotating arms: true plain AR and MTP depths 1/2/3. Each command discards
  one warmup per case/arm. The first command has three measured repetitions;
  confirmation has two. Decode counts 127 post-first-token transitions;
  whole-call counts all 128 tokens including prompt processing and completion.
- Baseline commit `ba6e01cd25e105e6d151339c76bf79cdc00a20fb` plus
  [candidate.patch.gz](candidate.patch.gz). The manifest's patch hash covers
  the decompressed bytes; gzip is lossless and avoids treating patch context
  whitespace as source whitespace. The [source manifest](source-manifest.json)
  records 591 runtime source hashes, verified unchanged before and after the run.
- The sweep holds three-candidate capacity constant and changes only the desired
  count through the shared planner. It therefore uses four resident slots even
  for depths one and two. The final automatic depth-two configuration uses three
  slots. The timing table describes the controlled sweep, not a separate timing
  measurement of that final allocation layout.
- [Protocol](protocol.json), [first run](first.json),
  [confirmation](confirmation.json), [independent validation](validation.json),
  [split/category summary](summary.json). The validator checked 756 calls,
  including warmups, with repeatable IDs and acceptance within each arm,
  depth-one equality to the preceding eager result, actual eager routes, no
  recovered-cycle fallback, and zero device allocations after close.

## Natural suite

Pooled throughput in tok/s; each arm has 54 measured natural calls initially
and 36 in confirmation.

| Arm | First decode | Confirm decode | First whole-call | Confirm whole-call | Accepted/proposed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plain AR | 67.485 | 67.668 | 63.130 | 63.289 | — |
| MTP depth 1 | 64.571 | 64.674 | 59.945 | 59.980 | 88.42% |
| MTP depth 2 | 64.970 | 64.879 | 60.210 | 60.169 | 78.91% |
| MTP depth 3 | 64.729 | 64.779 | 59.955 | 59.986 | 69.42% |

First-command pooled decode by split/category; the JSON summary includes the
same breakdown for confirmation. Categories pool their category and heldout
prompts. These overlapping groups are not additional independent samples.

| Scope | AR | Depth 1 | Depth 2 | Depth 3 |
| --- | ---: | ---: | ---: | ---: |
| Category prompts | 67.546 | 63.824 | 63.384 | 61.724 |
| Heldouts | 67.410 | 65.530 | 67.068 | 68.922 |
| Code | 67.469 | 65.320 | 67.767 | 66.854 |
| General English | 67.661 | 65.277 | 64.362 | 66.642 |
| General Japanese | 67.650 | 62.288 | 61.940 | 57.336 |
| Mixed Japanese/English | 67.172 | 65.133 | 64.742 | 68.319 |

## Length controls

First-command median decode tok/s. These repeated-token prompts test lengths,
not natural-language performance. Every control's first-token, whole-call and
confirmation measurements are preserved in the timing JSONs.

| Input tokens | AR | Depth 1 | Depth 2 | Depth 3 |
| ---: | ---: | ---: | ---: | ---: |
| 37 | 67.722 | 68.643 | 75.673 | 82.844 |
| 255 | 66.131 | 66.219 | 73.198 | 80.241 |
| 257 | 66.194 | 66.109 | 73.064 | 80.199 |
| 511 | 64.480 | 64.629 | 71.398 | 78.236 |
| 512 | 64.466 | 64.645 | 71.345 | 78.091 |
| 513 | 64.474 | 64.584 | 71.330 | 78.149 |
| 1024 | 64.159 | 61.246 | 67.162 | 72.937 |
| 1025 | 64.157 | 61.226 | 67.079 | 73.563 |
| 4096 | 64.859 | 47.042 | 50.467 | 53.366 |

## Correctness and limitations

The 40 initial device checks cover depth 1/2/3 transitions, exact repeated
state, output tails, provider/target rollback after injected failure or
cancellation, request reuse, public constructor configuration, depth-4 named
refusal, and allocation cleanup. The final unpinned public test also passes, bringing targeted device coverage
to 41 checks. In [shipping.json](shipping.json), all 54 final-default requests
(27 cases twice) match the measured depth-two IDs, acceptance and eager routes,
with zero allocations remaining. No planner or depth override is used.

The default unit suite passes. The full `--suite all` run recorded 20,021 passes,
6,569 guarded skips and two outdated GGUF graph fixtures. After migrating those
fixtures to the shared cache's retry contract, the affected provider/cache files
pass 57 tests with two guarded skips. No production repair was required; the
completed broad run plus focused repair is the integration evidence.

The [cycle-local numerical diagnostic](numerics.json) compares the existing
batched verifier to serial C1 logits at identical cycle inputs. All 18 natural
prompts and seven length controls run at all three depths, with 16 outputs and
three instrumented repeats plus an uninstrumented output reference. Repeats
match byte-for-byte. Distinct compared rows total 384/417/442 at depths 1/2/3;
top-1 agreement is 100% at each depth. Mean KL is 0.00003673/0.00006088/0.00005756;
maximum KL is 0.003597/0.008053/0.009952. Every category passes the diagnostic's
recorded envelope. Repeats are not pooled to inflate the row count.

This is a serial C1 diagnostic, not an independent BF16 model oracle, task-quality
score, or new arithmetic-profile certificate. No kernel arithmetic or registry
variant is changed by this unit. Over the longer timing outputs, depth two
differs from depth one on seven natural prompts; depth three matches depth one
on all prompts, and AR differs on nine. Cross-depth equality is diagnostic,
not substituted for the numerical check. No PARO/GGUF quality equivalence is
claimed. GPU results here cover gfx1151, not gfx1100.

## Reproduction

The archived measurement drivers preserve their measured bytes. To reproduce
the controlled sweep, use the base commit plus the decompressed
`candidate.patch.gz` in a separate
worktree, with these artifact files available there. The final source's default
is two rather than the sweep's three-candidate capacity; running the frozen
driver against final source without restoring that capacity is not the same
experiment and its depth-three assertions should fail rather than mislabel data.

```bash
R=benchmarks/results/2026-10-24-gfx1151-paro-depth
PY="$HOME/miniforge3/envs/therock/bin/python"
MODEL=/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/depth_numerics.py" \
  --model "$MODEL" --output "$R/numerics.json"
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_depth.py" \
  --model "$MODEL" --repetitions 3 \
  --parent benchmarks/results/2026-10-24-gfx1151-paro-graphs/first.json \
  --output "$R/first.json"
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_depth.py" \
  --model "$MODEL" --repetitions 2 \
  --parent benchmarks/results/2026-10-24-gfx1151-paro-graphs/first.json \
  --output "$R/confirmation.json"
python3 "$R/validate_depth.py" --first "$R/first.json" \
  --confirmation "$R/confirmation.json" --output "$R/validation.json"
```

On final source, `verify_shipping.py` exercises all 27 cases twice through
unmodified `LLM.generate_detailed()` without a depth argument, comparing IDs,
acceptance and routes to the measured depth-two arm. It makes no timing claim.

```bash
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/verify_shipping.py" \
  --model "$MODEL" --reference "$R/first.json" --output "$R/shipping.json"
```

## Unsuccessful attempts

Initial public tests stopped before kernels at three admission defects: resolver
keyword mismatch, unresolved runtime identity, and classifying ignored EOS as
logit processing. Each was repaired with focused regression coverage. An initial
recovery assertion used a post-drafting provider snapshot instead of the actual
pre-drafting rollback point; correcting the test preserved exact comparisons
and all eight failure/cancellation cases passed. A unit-test double lacked the
session-capacity field, and a default expectation still assumed three after
selection changed to two; both tests were repaired without weakening coverage.
