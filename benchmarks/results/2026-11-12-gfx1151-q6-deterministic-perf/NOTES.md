# Q6 deterministic-selection performance

The repeatability fix has a small warmed-prefill cost, while first-use requests
improve by avoiding per-owner algorithm timing. Keep the correctness fix; future
tuning should select a stable algorithm rather than restore runtime timing.

## Protocol

Host `gfx1151`, Radeon 8060S; `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf`,
production profile, BF16 KV, context capacity 2048, prefix cache off, MTP budget 3,
greedy 64-token output. The category/held-out prompt file has 18 prompts.
Each run uses fresh owners in parent/fixed/fixed/parent order with a hash-table
warmup. The initial run has two suite passes per owner; confirmation has four.
Concurrency is one. Prompt hashes, profile manifests, source revisions,
per-request telemetry, output IDs and allocation counters are in the JSON files.

The parent arm executes the exact `_launch_q6_fullgemm` function extracted from
`197bf61ae^`; everything else runs at `197bf61ae`. This isolates the selector
change without reverting other files. First pass means first use of suite shapes
after warmup, not cold model load. Constructor timing is not model-loading time:
loading is lazy and contributes to the warmup request.

Commands from the repository root, using the session's TheRock environment:

```bash
bash /tmp/hipengine-repair-run.sh benchmarks/results/2026-11-12-gfx1151-q6-deterministic-perf/public_ab.py --out benchmarks/results/2026-11-12-gfx1151-q6-deterministic-perf/public_ab.json
bash /tmp/hipengine-repair-run.sh benchmarks/results/2026-11-12-gfx1151-q6-deterministic-perf/public_ab.py --passes 4 --out benchmarks/results/2026-11-12-gfx1151-q6-deterministic-perf/confirm_ab.json
python3 benchmarks/results/2026-11-12-gfx1151-q6-deterministic-perf/summarize.py
```

The wrapper selects `~/miniforge3/envs/therock/bin/python`, discovers the SDK with
`python -m rocm_sdk path --root`, and sets the SDK library/compiler paths. The
artifacts record the effective environment. No kernel source changed in this
measurement; builds are reused from the prior validated run.

## Results

Means of per-owner suite totals; warmed confirmation averages passes after the
first. Positive deltas mean slower. Both runs are retained, with no outlier removal.

| Metric | Initial parent → fixed | Delta | Confirmation parent → fixed | Delta |
| --- | --- | --- | --- | --- |
| First-pass wall, seconds | 63.072 → 61.190 | -2.98% | 63.066 → 61.137 | -3.06% |
| First-pass prefill, seconds | 10.353 → 8.591 | -17.02% | 10.405 → 8.589 | -17.45% |
| Warm wall, seconds | 61.277 → 61.469 | +0.31% | 62.722 → 63.670 | +1.51% |
| Warm prefill, seconds | 8.346 → 8.459 | +1.35% | 8.335 → 8.454 | +1.43% |

The warmed-prefill regression repeats across runs. Overall wall timing in the
longer run is less isolated: target verification time rose in later passes for
both arms, including the final parent owner. No hardware telemetry was captured
to identify the cause, so do not attribute all of the +1.51% wall delta to Q6.
Per-category results are in `summary.json`.

## Correctness and scope

All generated IDs match the parent reference. Every request engages MTP with no
recoverable failure; tracked allocations return to their initial value after
closing each owner. The Japanese held-out speculative prompt has differing
acceptance counts between selectors, consistent with the earlier diagnosed
reduction-order change. This is an end-to-end performance comparison, not a
same-control kernel microbenchmark. The unit, synthetic GPU numerical/ownership
and public fresh-owner repeatability gates for the fix are recorded in
`../2026-11-12-gfx1151-mtp-repeatability/resolution.json`.

No production code or default changes in this measurement. These results do not
establish performance at long contexts, higher concurrency, other models or on
gfx1100 hardware. A stable, offline-selected faster hipBLASLt algorithm is a
possible follow-up, not a measured improvement.
