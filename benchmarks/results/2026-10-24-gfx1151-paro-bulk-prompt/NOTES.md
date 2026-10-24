# PARO bounded bulk prompt priming — Radeon 8060S

## Result and scope

Qwen3.6-35B-A3B packed ParoQuant W4, BF16 target KV, one active request and
one MTP candidate, Framework Desktop / AMD Radeon 8060S on physical host
`gfx1151`. The model manifest fingerprint is
`362ac5e97b22ed60e9c02a9b3b2341ad8ebedbc79b037289404812097d65e9d7`.
Both arms use shipping defaults, eager target verification, prefix cache off,
a 4352-token allocation, greedy sampling and 128 EOS-ignored outputs.
This compares two MTP prompt implementations, not MTP against autoregressive decoding.

| Workload | Serial first token (s) | Bulk first token (s) | Serial whole-call tok/s | Bulk whole-call tok/s |
| --- | ---: | ---: | ---: | ---: |
| 18 natural/category-heldout prompts | 0.1818 | 0.1715 | 59.575 | 59.818 |
| 512 input tokens | 0.7623 | 0.6474 | 46.989 | 49.016 |
| 1024 input tokens | 1.5688 | 1.3386 | 35.173 | 37.557 |
| 4096 input tokens | 6.0210 | 5.0944 | 14.676 | 16.424 |

Natural first-token latency is the median across samples; natural throughput
pools tokens and wall time. Control rows are per-case medians. Natural decode
is effectively flat: 64.607 → 64.555 tok/s (−0.08%); whole-call throughput
improves 0.41%. Confirmation gives 64.569 → 64.522 tok/s decode and
59.569 → 59.809 tok/s whole-call (+0.40%). At 4K, confirmation first-token
latency is 6.0259 → 5.1010 s. Raw samples and counters are in
[first.json](first.json), [confirmation.json](confirmation.json), and
[summary.json](summary.json).

## Implementation and comparison

The serial arm substitutes the parent `07fba0543` packed prompt callback:
normalize and prime each historical row separately. The candidate uses the
shared target-hidden chunk contract, split planner and publication ledger,
normalizes rows together, and primes historical KV in bounded 16-row chunks.
The final prompt row still computes the complete predictor. Decode, rollback,
arithmetic kernels and variant selection are unchanged. Bulk priming is the
public default, not a benchmark-only route or an environment override.

The source is `c2815cbbb` plus [candidate.patch](candidate.patch).
[source-and-validation.json](source-and-validation.json) records frozen file
hashes, checked unchanged after all measurement. The unrelated dirty audit
triage file was neither edited nor part of the candidate. Provenance includes
compiler, runtime, model, host, environment and exact executed commands.

## Correctness and accounting

- The public device oracle compares full predictor priming, serial KV-only,
  bulk KV-only and repeated bulk calls at lengths
  1/2/15/16/17/31/32/33/37/255/256/257/511/513/1025/4096. All 64 calls match
  live KV bytes, final hidden/prediction state, output IDs and acceptance.
- All 294 timing calls match IDs and acceptance across arms and repetitions.
  Route counters confirm `prompt_tokens - 1` bulk rows on the candidate and
  zero on the serial arm. No recovered-failure fallback ran.
- First sweep: one warmup and three alternating pairs per case. Confirmation:
  one warmup and two pairs. The cases are all ten `mtpbench-code-general-ja`
  prompts, all eight `gdn-prefill-category-heldouts`, plus 512/1024/4096 controls.
- The 35-minute combined job limit interrupted confirmation after 15 complete
  cases and four calls of the next case. The original
  [partial artifact](confirmation-timeout-partial.json) is preserved. The
  [resume driver](measure_bulk_resume.py) preserves complete records exactly
  and reruns the interrupted case and five remaining cases with fresh warmups.
  The four interrupted-case calls are not pooled into the 294-call result.
  Cleanup of the terminated process was not observed. First-sweep, resumed
  confirmation and attribution processes close with zero tracked allocations.
- [Output sanity](output-sanity.json) decodes authoritative completion token
  IDs, not the joined streaming text fragments. Natural prefixes are coherent;
  EOS-ignored short prompts may continue with repeated dialogue. This is not
  a completed-task quality evaluation or a cross-route numerical claim.
- Focused CPU tests, the default unit suite, compileall, fixture/tier checks,
  registry/CPU fixtures and smoke-add-plan passed. Validator negative controls
  reject incorrect output hashes, acceptance, counters, timing and cleanup.
  Resume checks authenticate the partial artifact and unchanged preserved cases.

## Separate attribution

[attribution.json](attribution.json) contains 12 additional calls: one warmup
and one measured pair at a natural prompt, 512 and 4096 tokens. Synchronization
surrounds the packed capture method, including target prefill and draft priming.
Measured serial → bulk window times are 0.12775 → 0.12088 s,
0.71961 → 0.60802 s and 5.96671 → 5.04936 s. IDs and acceptance match the timing
sweep. These synchronized diagnostics are excluded from throughput aggregation;
they are not a kernel profiler trace or isolated draft-kernel timings.

## Reproduction

Run from the repository root with the local model and ROCm Python environment.
The archived environment wrapper uses `~/miniforge3/envs/therock` and the same
clean environment as the recorded runs. Substitute its paths for another host;
that creates a separate hardware lane.

```bash
R=benchmarks/results/2026-10-24-gfx1151-paro-bulk-prompt
MODEL=/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16
PY="$HOME/miniforge3/envs/therock/bin/python"
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_bulk.py" \
  --model "$MODEL" --output /tmp/paro-bulk-first.json --cases all --warmups 1 --repetitions 3
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_bulk.py" \
  --model "$MODEL" --output /tmp/paro-bulk-confirmation.json --cases all --warmups 1 --repetitions 2
python3 "$R/validate_bulk.py" --first /tmp/paro-bulk-first.json \
  --confirmation /tmp/paro-bulk-confirmation.json --output /tmp/paro-bulk-summary.json
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_bulk.py" \
  --model "$MODEL" --output /tmp/paro-bulk-attribution.json --cases smoke --warmups 1 --repetitions 1 --attribution
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" -m pytest \
  tests/test_live_paro_mtp_prompt_priming.py -v -o addopts=''
```

To revalidate the archived interrupted confirmation, pass its `first.json` and
`confirmation.json` to the validator from the repository root. The latter
contains the original partial-artifact path and hash. Exact original resume
arguments are also recorded in its provenance.
