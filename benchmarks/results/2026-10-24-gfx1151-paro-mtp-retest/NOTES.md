# gfx1151 PARO MTP retest — October 24, 2026

On Framework Desktop / Radeon 8060S, Qwen3.6-35B-A3B packed ParoQuant W4
with BF16 KV decodes more slowly with the repaired single-candidate MTP path
than without it. These are public streaming measurements, not kernel-only
rates or a new quantization-quality evaluation.

| Natural-prompt metric | Plain decoding | MTP |
| --- | ---: | ---: |
| Decode tok/s | 67.54 | 64.32 |
| Whole-call tok/s, including prompt processing | 63.43 | 58.59 |
| Median time to first token | 0.137 s | 0.209 s |

MTP decode is 4.76% slower; whole-call throughput is 7.63% lower. The MTP
README cell uses 64.3 tok/s from the first sweep. Its neighboring plain-decode
cell remains the earlier resident-harness measurement, not this matched
streaming baseline. The existing GGUF MTP cell is a separate published run,
not a fresh paired comparison with this PARO retest.

## Protocol and scope

- Clean commit `9c24d765b3bd66586a367eef24297da3e1411ad2`, physical host
  `gfx1151`, Framework Desktop (AMD Ryzen AI Max 300 Series), Radeon 8060S,
  ROCm 10 environment. Exact HIP/compiler identities are in each artifact.
- Local model `Qwen3.6-35B-A3B-PARO-packed-MTP-BF16`, fingerprint
  `362ac5e97b22ed60e9c02a9b3b2341ad8ebedbc79b037289404812097d65e9d7`.
- One active request, maximum sequence length 4352, BF16 KV, prefix cache off,
  greedy sampling, EOS ignored, 128 output tokens. MTP uses one candidate.
- All ten `mtpbench-code-general-ja.jsonl` prompts and eight
  `gdn-prefill-category-heldouts.jsonl` prompts, plus repeated-token controls
  at 512, 1024 and 4096 inputs. Prompt-source hashes and input hashes are saved.
- Two discarded warmups and five measured calls per case and arm, alternating
  AR/MTP and MTP/AR order. A second complete invocation confirms the first;
  no faster-run selection or sample removal.
- AR uses `LLM.stream_detailed()`; MTP uses
  `LLM.stream_speculative_mtp_detailed()`. A reclaim observer records
  scheduler-owned output IDs without replacing generation or commitment.
- Decode divides 127 outputs by the observed first-to-last-token interval.
  Public iterator overhead is included. First-token latency includes prompt
  processing and MTP priming; whole-call time includes iterator exhaustion.
  Model construction is excluded from measured samples by warmup.

Natural-suite rates pool token counts and time across all 18 prompts, giving
90 measured calls per arm. Length controls use medians and are not pooled
into the natural-suite result.

| Prompt tokens, repeated-token control | Plain decode tok/s | MTP decode tok/s | Plain first token | MTP first token |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 64.50 | 64.31 | 0.501 s | 1.330 s |
| 1024 | 64.18 | 60.77 | 1.071 s | 3.123 s |
| 4096 | 64.93 | 46.12 | 4.133 s | 22.312 s |

The confirmation changes natural-suite AR/MTP rates by -0.08%/-0.17%; all
length-control medians are within 0.22%. The slower 4K MTP result repeats.

## Validation and limits

Both sweeps pass 76 aggregation checks. All 588 calls, including warmups,
produce 128 valid IDs and repeat within each arm. All 42 case/arm output
sequences match between sweeps. MTP requests execute eager speculative
cycles; AR requests execute none. No speculative recovery errors occur,
request owners are reclaimed, and tracked allocations return to zero at close.

Natural-suite candidate acceptance is 5345/6045, or 88.42%. Cycle counts alone
omit ordinary output-cap tail steps. `counts-and-prefix-audit.json` reconciles
every output as serial steps + native steps + speculative cycles + accepted
candidates. In the first sweep, 40 measured natural-suite tokens and 15 control
tokens use serial tails. The summary builder's cycle-only `accounting_delta`
is therefore not a token-loss finding; throughput uses observed output counts.

AR and MTP trajectories match on 12/21 cases, including all three controls;
nine natural trajectories differ at 128 outputs. Every one of the 19 cases
shared with the prior adapter packet preserves that packet's first 64 IDs in
both arms. The [prior same-state diagnostic](../2026-10-23-gfx1151-paro-mtp-adapter/NOTES.md)
characterized four divergences at 64 outputs; it does not certify the additional
128-token continuations or establish a complete strict-teacher quality gate.
No arithmetic, kernel, execution-profile, or default changes accompany this
measurement. It is not a token-parity or quality-equivalence claim.

## Files and reproduction

`first.json` and `confirmation.json` preserve every timing sample, token event,
route counter, output ID, and provenance field. `summary.json` contains pooled
rates and confirmation deltas. `archive-manifest.json` maps the original JSON
hashes to whitespace-compacted copies; no fields or samples were dropped.
`protocol.json` records the predeclared protocol and review thresholds.

The two `smoke-v*-failed.json` files preserve rejected harness attempts: the
first expected final IDs from ordinary streaming; the second used ordinary
streaming for the MTP arm. Their drivers are archived separately. `smoke.json`
is the corrected two-case check. `confirmation-timeout-partial.json` preserves
the interrupted confirmation from the initial 50-minute job limit. None of
these files contributes to the published metrics. The replacement confirmation
used the same command and settings with a two-hour process limit.

The archived environment wrapper is the exact one used; its `PYTHONPATH` is
overridden below. Keep the archived directory outside the clean measurement
checkout, with an idle GPU and the model at the recorded path. Substitute local
model and interpreter paths as needed. The wrapper creates the compiler-version file
under `/tmp/hipengine-paro-refresh-20261023`, which the driver reads.

```bash
ART="$(realpath benchmarks/results/2026-10-24-gfx1151-paro-mtp-retest)"
PRIOR="$(realpath benchmarks/results/2026-10-23-gfx1151-paro-mtp-adapter/validation.json)"
CHECKOUT=/tmp/hipengine-paro-mtp-reproduction-checkout
git worktree add --detach "$CHECKOUT" 9c24d765b3bd66586a367eef24297da3e1411ad2
cd "$CHECKOUT"
OUT=/tmp/hipengine-paro-mtp-reproduction
mkdir -p "$OUT"
for pass in first confirmation; do
  bash "$ART/environment-as-run.sh" env PYTHONPATH="$PWD" \
    "$HOME/miniforge3/envs/therock/bin/python" "$ART/measure.py" \
    --model /models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16 \
    --cases all --warmups 2 --repetitions 5 --output "$OUT/$pass.json"
done
python3 "$ART/build_summary.py" --first "$OUT/first.json" \
  --confirmation "$OUT/confirmation.json" --protocol "$ART/protocol.json" \
  --output "$OUT/summary.json"
python3 "$ART/audit_counts.py" --first "$OUT/first.json" \
  --confirmation "$OUT/confirmation.json" \
  --prior "$PRIOR" \
  --output "$OUT/counts-and-prefix-audit.json"
python3 -m unittest discover -s "$ART" -p test_build_summary.py -q
```

The summary builder validates the recorded measurement commit and driver/source
hashes. Rebuilding a historical summary also requires its recorded driver and
checkout paths to remain available; archived copies preserve their contents.
A rerun from another revision is a new experiment, not an identical reproduction.
The 48 synthetic aggregation tests require only Python's standard library.
