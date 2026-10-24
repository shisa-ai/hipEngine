# PARO MTP target backup without live-prefix KV copies

## Result

On Framework Desktop / AMD Radeon 8060S (`gfx1151`), Qwen3.6-35B-A3B
packed ParoQuant W4 with BF16 KV, one active request and one MTP candidate,
omitting immutable prefix KV from the target backup slightly improves public
decode throughput. All 21 case medians improve in both runs.

| Workload | Full-prefix backup | Mutable state + cursors | Change | Confirmation, full → cursor-only |
| --- | ---: | ---: | ---: | ---: |
| 18 natural/category-heldout prompts, pooled decode tok/s | 64.45 | 64.58 | +0.20% | 64.43 → 64.63 (+0.31%) |
| 512 input tokens, median decode tok/s | 64.38 | 64.72 | +0.52% | 64.34 → 64.72 (+0.59%) |
| 1024 input tokens, median decode tok/s | 60.81 | 61.26 | +0.74% | 60.78 → 61.23 (+0.75%) |
| 4096 input tokens, median decode tok/s | 46.16 | 47.03 | +1.88% | 46.17 → 47.04 (+1.90%) |

The natural-suite improvement is small, not a material change in the MTP/AR
comparison. This experiment compares two MTP backup implementations; it does
not measure an autoregressive baseline or claim MTP beats autoregressive decode.
Prompt latency is essentially unchanged. Natural whole-call throughput is
59.44 → 59.55 tok/s, confirmed at 59.44 → 59.60 tok/s.

## Change and comparison

`SpeculativeStateJournal` publishes a restore callback only after its capture
succeeds. Rollback attempts all completed owners in reverse order, retains
failed restores for retry, and does not reuse a stale target backup when failure
precedes target capture. PARO uses it for the provider checkpoint and target
mutable state/cursors. Target verification appends KV after the root cursor;
rollback hides the provisional suffix instead of restoring the immutable prefix.
The default target copy now uses `kv_rows=0`.

Both timing arms use the same new journal. The full-prefix arm substitutes the
parent's successful-cycle copy extent, `kv_rows=position`; the cursor-only arm
uses the default zero extent. Mutable recurrent/hidden copies, sampling,
prompt priming, verification and commitment are identical. No arithmetic,
selected kernel, accepted candidate or prompt suite changed. Thus the timing
isolates copy extent, not the journal's error-handling overhead.

The measured source is commit `4d8ecc3a7` plus [candidate.patch](candidate.patch).
[Source hashes](source-sha256.json) matched after the run. The patch also contains
an unrelated bulk-prompt primitive that was not called in either arm; it is not
part of this performance claim. The exact tree was intentionally uncommitted
during evaluation, as the provenance records. The score-restore fix invalidates
unsnapshotted device head buffers; it affects failure recovery, not these
successful-cycle timings.

## Protocol and correctness

- Public surface: `LLM.stream_speculative_mtp_detailed`, `backend=hip_gfx1151`,
  `quant=w4_paro`, maximum sequence length 4352, prefix cache off, temperature 0,
  `ignore_eos=True`, 128 generated tokens. Model load is outside timing.
- All ten `mtpbench-code-general-ja.jsonl` prompts, all eight
  `gdn-prefill-category-heldouts.jsonl` prompts, and repeated token 9707 controls
  at 512/1024/4096 input tokens. Prompt hashes are stored in each artifact.
- First run: one warmup and three measured samples per arm/case. Confirmation:
  one warmup and two samples. Pair order alternates. Decode time is observed
  first-token to last-token wall time over 127 newly generated tokens; terminal
  duplicate chunks are excluded. The natural rate pools tokens and wall times.
- [Independent summary](summary.json) validates all **294 calls**, exact
  cross-arm/repeat IDs and acceptance sequences, MTP/eager route, token
  accounting, one target backup per cycle and zero tracked allocations/bytes
  after close. Output IDs also match the immediate parent's cache-only artifact
  on all 21 cases. There are no interrupted or missing confirmation cases.
- Ten public failure-injection tests pass at lengths 1/37/511/1025/4096,
  following target verification and provider update. They compare mutable
  state, host/device cursors, provider hidden/KV and live target KV after
  rollback. After the stale-score repair, both length-37 failure cases also
  pass with device score access explicitly refused until a new scoring advance.
- Decoding the stored IDs yields relevant text on all 18 natural prompts.
  Some outputs remain in thinking text at the fixed 128-token cap; EOS-ignored
  short answers can continue into repeated turns. These outputs are unchanged
  from the parent, not evidence of task completion or a new quality evaluation.
  The stored stream-text fragments are not the complete decoded output; use
  the stored token IDs or [decoded-output-sanity.json](decoded-output-sanity.json).
- ROCm/HIP version, host, model fingerprint, command and clean environment
  wrapper are recorded in the raw artifacts. No profiler runs inside the timed
  sweep. `smoke.json` has no warmup and includes cold setup: it is a correctness
  smoke only, not a fair latency comparison.

## Separate copy attribution

The separate `--attribution` run synchronizes immediately before and after each
target backup. It has one warmup and one measured call per arm on the first
natural prompt and the 512/4096 controls. These windows include host enqueue
and synchronization costs; they are not GPU event durations and are excluded
from throughput measurements.

| Input | Mean synchronized backup, full-prefix | Cursor-only |
| --- | ---: | ---: |
| `code_merge_intervals` | 748 µs | 696 µs |
| 512 tokens | 842 µs | 694 µs |
| 4096 tokens | 1497 µs | 691 µs |

The cursor-only arm records zero prefix rows copied at every cycle. Mutable
state still has to be copied. This explains the larger benefit at long context
without attributing unrelated costs to the change. See [attribution.json](attribution.json).

## Reproduction

Run from the repository root on the recorded host with the model installed:

```bash
R=benchmarks/results/2026-10-24-gfx1151-paro-rollback
PY="$HOME/miniforge3/envs/therock/bin/python"
MODEL=/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16
bash "$R/run-env.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_rollback.py" \
  --model "$MODEL" --output "$R/first.json" --warmups 1 --repetitions 3
bash "$R/run-env.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_rollback.py" \
  --model "$MODEL" --output "$R/confirmation.json" --warmups 1 --repetitions 2
python3 "$R/validate_rollback.py" --first "$R/first.json" \
  --confirmation "$R/confirmation.json" --output "$R/summary.json"
bash "$R/run-env.sh" env PYTHONPATH="$PWD" "$PY" "$R/measure_rollback.py" \
  --model "$MODEL" --output "$R/attribution.json" \
  --cases smoke --attribution --warmups 1 --repetitions 1
(cd "$R" && python3 -m unittest test_validate_rollback -v)
```

The wrapper initializes its compiler-version file under `/tmp`; keep it with the
driver. [rollback-protocol.json](rollback-protocol.json) records the pre-run
workload, acceptance rule and attribution split. CPU coverage includes the new
journal/failure tests, two stale-score tests, the default suite, fixture checks
and registry/CPU/smoke-plan checks. No root README cell changes in this unit.
