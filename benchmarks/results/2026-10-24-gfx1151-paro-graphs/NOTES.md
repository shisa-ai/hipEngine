# PARO verifier graphs — Radeon 8060S

## Result and scope

Graph verification was slower than eager verification in this one-candidate
comparison. This is not a speedup claim.

Qwen3.6-35B-A3B packed ParoQuant W4, BF16 target KV, one active request and
one MTP candidate, Framework Desktop / AMD Radeon 8060S on host `gfx1151`.
The model manifest fingerprint is
`362ac5e97b22ed60e9c02a9b3b2341ad8ebedbc79b037289404812097d65e9d7`.
All arms use shipping arithmetic kernels, prefix cache off, a 4352-token
allocation, greedy sampling and 128 EOS-ignored outputs. The graph candidate
changes host graph ownership and replay, not arithmetic or variant selection.

| Run, 18 natural/category-heldout prompts | AR decode tok/s | Eager MTP decode tok/s | Graph MTP decode tok/s | Eager MTP whole-call tok/s | Graph MTP whole-call tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| First sweep | 67.519 | 64.506 | 62.421 | 59.832 | 57.954 |
| Confirmation | 67.230 | 64.470 | 62.497 | 59.860 | 58.030 |

Natural throughput pools output tokens and wall time. Graph decode is 3.23%
slower in the first sweep and 3.06% slower in confirmation. These are paired
arms in one process, not isolated pristine-parent processes: the existing
capture-safe rotate-barrier mode stays enabled after the first graph capture
for subsequent launches. Full per-case results, including all nine length
controls and any gains, are in [validation.json](validation.json).

## Implementation and comparison

The candidate is `e8ad5eaf2` plus [candidate.patch](candidate.patch).
The measured runtime and adapter hashes are in
[source-hashes.json](source-hashes.json); they were checked against the frozen
execution tree before and after both sweeps. Unrelated audit changes and the
concurrent GGUF journal work were not loaded by the frozen PARO process.
[protocol.json](protocol.json) fixes the prompt set, repetitions, output work,
validation rule and driver hashes before measurement.

The PARO runtime uses `SpecGraphStageCache` for bounded graph ownership,
retryable release, and bindings over row shape, attention split counts,
configuration and live device addresses. Capture-time scratch and commit
buffers remain owned until graph destruction completes. The adapter reports
whether a graph actually replayed rather than reporting its planned route.
The obsolete force-recapture environment workaround is removed.

The first smoke comparison was slower. A separate host profile identified
repeated pointer traversal and duplicate binding normalization. The measured
candidate avoids recursive pointer-leaf calls, caches only scratch field names
with an exact live member-name check, and normalizes binding addresses once.
Neither pointer values nor allocation identities are cached without checking
live values. The final sweep above includes that repair.

## Correctness and accounting

- All **567 calls** pass independent validation: 27 cases × 3 arms × 4 calls
  in the first sweep, and 27 × 3 × 3 in confirmation. Each case has one warmup
  per arm; warmups are excluded from timing aggregation.
- Cases include all ten `mtpbench-code-general-ja.jsonl` prompts, all eight
  `gdn-prefill-category-heldouts.jsonl` prompts, and lengths
  37/255/257/511/512/513/1024/1025/4096. Arm order alternates AR/eager/graph
  and graph/eager/AR between repetitions.
- Eager and graph MTP match output IDs and acceptance counts for every case
  and repetition. Every graph call actually replays a graph; no capture or
  recovered-cycle fallback appears. AR/MTP IDs match in 18 of 27 cases in
  each sweep. Cross-route equality is diagnostic, not the repeatability oracle.
- Eleven public device tests compare graph/eager/repeated live state at seven
  lengths and inject failure/cancellation after replay at 37 and 1025 tokens.
  Checkpoints restore exactly, cancelled output stops at the committed prefix,
  and subsequent requests match the clean baseline.
- Both sweeps and both smoke runs close with zero tracked allocations.
  [Output sanity](output-sanity.json) records inspection of authoritative
  completion IDs. Fixed EOS-ignored output work can continue with repeated
  dialogue markers; this is not a completed-task quality evaluation.
- The final focused PARO bundle passes 206 tests. The default unit suite,
  compileall, fixtures, registry/CPU-fixture/smoke-add-plan checks and audit
  gate pass. The measurement validator passes its valid fixture and rejects
  20 corrupted records covering outputs, acceptance, routes, timing and cleanup.

## Profiling and default selection

The prebuilt [profiler trace summary](profiler-summary.json) records one graph
capture and 14 HIP graph launches during a 32-output public request, followed
by graph destruction. It also records actual verifier metadata and accept
kernel dispatches. This trace predates the pointer-walk optimization; kernel
calls and graph execution are unchanged. The profiler switched to system-memory
intercept queues, so its durations are not throughput evidence.

[Separate synchronized attribution](attribution.json) contains 18 additional
calls at a natural prompt and lengths 512/4096. Eager → graph verification
windows total 1.50614 → 1.57061 s, 1.46592 → 1.52521 s, and
1.47795 → 1.54698 s respectively. Each arm executes the same number of cycles
(65/63/63), with identical outputs and acceptance. These windows include host
binding and submission overhead as well as device execution; they are not
isolated kernel timings and are excluded from throughput aggregation.

**Default selection:** the adapter honors the shared planner's execution route.
Ordinary one-candidate PARO requests continue to select eager verification;
a graph plan still runs capture/replay. The graph candidate was rejected as the
default because of the repeated measured slowdown, not because evidence was
missing. No MTP capability is disabled and no new environment flag is added.
The final device oracle checks the unmodified public eager default against
explicit graph plans, including replay failure/cancellation. The measured
`candidate.patch` preserves the earlier graph-auto candidate; the final policy
selection is recorded separately in `final-selection.patch`.

## Reproduction

Run from the repository root with the local model and ROCm Python environment.
The wrapper restores the measured graph-auto candidate only inside the
comparison process. Its eager baseline still explicitly selects graph-off.
It does not change the production default or the archived measurement driver.

```bash
R=benchmarks/results/2026-10-24-gfx1151-paro-graphs
MODEL=/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16
PY="$HOME/miniforge3/envs/therock/bin/python"
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/reproduce_graph_candidate.py" \
  --model "$MODEL" --output /tmp/paro-graph-first.json --repetitions 3
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/reproduce_graph_candidate.py" \
  --model "$MODEL" --output /tmp/paro-graph-confirmation.json --repetitions 2
python3 "$R/validate_graph.py" --first /tmp/paro-graph-first.json \
  --confirmation /tmp/paro-graph-confirmation.json --output /tmp/paro-graph-validation.json
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" "$R/reproduce_graph_candidate.py" \
  --model "$MODEL" --output /tmp/paro-graph-attribution.json --cases smoke --repetitions 1 --attribution
bash "$R/environment.sh" env PYTHONPATH="$PWD" "$PY" -m pytest \
  tests/test_live_paro_mtp_graph.py -v -o addopts=''
```
