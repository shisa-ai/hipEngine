# PARO cache-only MTP prompt priming — October 24, 2026

Historical PARO prompt rows now compute only the draft key/value cache. The
final prompt row still computes the full predictor state and proposal. On
Framework Desktop / Radeon 8060S, this reduces first-token latency without
changing the measured decode rate or any output IDs. This is a comparison of
two MTP prompt implementations, **not an MTP-versus-AR speedup claim**.

| Workload | Full priming: first token | Cache-only: first token | Reduction |
| --- | ---: | ---: | ---: |
| 18 natural/category-heldout prompts | 0.215 s | 0.181 s | 16.1% |
| 512 repeated input tokens | 1.335 s | 0.766 s | 42.6% |
| 1024 repeated input tokens | 3.133 s | 1.571 s | 49.9% |
| 4096 repeated input tokens | 22.304 s | 6.021 s | 73.0% |

These are medians from the first sweep. Natural-suite pooled decode is
64.4736 → 64.4747 tok/s; whole-call throughput is 58.55 → 59.47 tok/s (+1.57%).
The confirmation measures 64.42 → 64.37 tok/s decode and 58.51 → 59.39 tok/s
whole-call throughput. The 4K confirmation first-token medians are
22.290 → 6.018 s. No faster-run selection or sample exclusion is applied.

## Configuration and baseline

- Physical host `gfx1151`, Framework Desktop, Radeon 8060S, HIP
  `7.15.26333-0000000`; exact compiler identity is in the completed artifacts.
- Qwen3.6-35B-A3B packed ParoQuant W4, BF16 MTP weights and BF16 KV;
  model fingerprint `362ac5e97b22ed60e9c02a9b3b2341ad8ebedbc79b037289404812097d65e9d7`.
- One request, one draft candidate, eager MTP verification, context capacity
  4352, prefix cache off, greedy sampling, EOS ignored, 128 output tokens.
- All ten `mtpbench-code-general-ja.jsonl` prompts and eight
  `gdn-prefill-category-heldouts.jsonl` prompts, plus the three length controls.
  Source and prompt-ID hashes are stored in each measurement.
- Base commit `071d5a87a8a71ea1edfab31516ae1a16da8c557b` plus
  `candidate.patch`; `candidate-source.json` identifies the measured source.
  Both arms include the previously committed attention scratch-reuse repair.
- The full arm replaces only `append_prompt_kv` with the former full historical
  step, with result readbacks disabled as before. The cache-only arm uses the
  shipping adapter and primitive without substitution. Kernel arithmetic and
  selected variants are unchanged; this is work elimination, not an arithmetic
  promotion or a new model-quality assessment.

Each case has one discarded warmup pair and three measured pairs, alternating
full/cache-only and cache-only/full. Confirmation has one warmup and two
measured pairs. Both arms call `LLM.stream_speculative_mtp_detailed()`.
First-token latency includes target prefill and draft priming. Decode divides
127 outputs by the observed first-to-last-token interval; whole-call time
includes iterator exhaustion. Model construction and snapshot inspection are
outside measured calls. No synchronization for attribution is inserted.

## Correctness and limitations

All 294 sweep/confirmation calls produce 128 valid IDs and repeat exactly.
Full/cache-only IDs and accepted-count sequences match for all 21 cases,
including across invocations. Natural-suite acceptance is 3207/3627 (88.42%)
per arm in the first sweep. Every call executes eager single-candidate MTP;
first token, speculative cycles, accepted candidates and ordinary tails
reconcile to the output count. No speculative recovery occurs, and the driver
checks that each completed request has been reclaimed.

The initial 40-minute process limit interrupted confirmation after 19 complete
cases, not during a recorded case. The 1K and 4K cases alone were run in a new
process. `confirmation-timeout-partial.json` retains `passed: false` and has no
final provenance or teardown record. The validator checks its completed cases
but does **not** certify that process's teardown. The full sweep and resumed
process both report zero tracked allocations and bytes after close. The
resumption driver differs only by explicit case selection; both exact versions
are archived. The process interruption is not hidden by synthesizing a fully
passed confirmation artifact.

A separate public GPU oracle passes lengths 1, 2, 31, 32, 33, 255, 256, 257,
513, 1025 and 4096. It checks live KV bytes, final predictor state, output IDs,
repeatability and cleanup. CPU ownership and validity tests cover missing
hidden/proposal state and restoring a cache-only snapshot. The refreshed CPU
suite passes 15,462 tests with 271 skips; exact commands and logs are recorded
in the unit's worklog.

Decoded scheduler-owned IDs were inspected for all natural prompts. The fixed
128-token/EOS-ignored workload includes unfinished reasoning and repeated text
after EOS on short answers; this is not a completed-task score. The recorded
stream text can omit portions of the scheduler-owned sequence. Output parity
uses the full ID sequence, not those text fragments. No AR/MTP or strict-teacher
quality-equivalence claim follows from this comparison.

## Reproduction

The artifacts preserve every field and sample with whitespace-only JSON
compaction; `archive-manifest.json` records original and archived hashes.
`cache-only-protocol.json` is the predeclared protocol. The wrapper records the
exact environment used; substitute local model/interpreter paths as needed.
Run on the recorded host with an idle GPU and the same source patch:

```bash
ART="$(realpath benchmarks/results/2026-10-24-gfx1151-paro-cache-only)"
CHECKOUT=/tmp/hipengine-paro-cache-only-reproduction
OUT=/tmp/hipengine-paro-cache-only-results
git worktree add --detach "$CHECKOUT" 071d5a87a8a71ea1edfab31516ae1a16da8c557b
cd "$CHECKOUT"
git apply --unidiff-zero "$ART/candidate.patch"
for spec in first:3 confirmation:2; do
  bash "$ART/environment-as-run.sh" env PYTHONPATH="$PWD" \
    "$HOME/miniforge3/envs/therock/bin/python" "$ART/measure_cache_only.py" \
    --model /models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16 \
    --output "$OUT/${spec%:*}.json" --warmups 1 --repetitions "${spec#*:}"
done
python3 "$ART/validate_cache_only.py" --first "$OUT/first.json" \
  --confirmation "$OUT/confirmation.json" --output "$OUT/summary.json"
```

Rebuild the archived split-confirmation summary without a GPU:

```bash
python3 "$ART/validate_cache_only.py" --first "$ART/first.json" \
  --confirmation-partial "$ART/confirmation-timeout-partial.json" \
  --confirmation-tail "$ART/confirmation-tail.json" --output /tmp/cache-only-summary.json
python3 -m unittest discover -s "$ART" -p test_validate_cache_only.py -v
```

The actual resumed command used `measure_cache_only_resume.py` with
`--case-ids p1024 p4096 --warmups 1 --repetitions 2`; the complete invocation
is in `confirmation-tail.json`. Allow sufficient process time for both full
invocations; their lifecycle is longer than the measured token windows.
