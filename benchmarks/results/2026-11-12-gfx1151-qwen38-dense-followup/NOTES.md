# Qwen3.8 dense serving follow-up and remaining costs

## Scope and provenance

This records the clean-source server rerun after mixed-pair batching and ranks
follow-up work. It changes no runtime behavior and makes no new production
numerical-approval claim.

- Physical host `gfx1151`, AMD Ryzen AI MAX+ 395 / Radeon 8060S, Linux
  `7.3.1-1-cachyos`; not the separate Desktop or W7900 lane.
- `/models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf`, sampled SHA-256
  `f343914a95e38d8f9f56886d10de4248738ad94aabb0fa0af8fd1b00c4e82f97`,
  16,464,440,224 bytes. The fingerprint scheme and offsets are in `server.json`.
- HIP `7.15.26333`, AMD clang `23.0.0git`, LLVM revision
  `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.
- Baseline `ff1c5ed4102692c5d47a8fd661c1c7ed30b7bc09`; candidate and attribution
  source `2db009da3b1ad236fd713671b4d4a1ddebfeef45`. Both server runs report clean
  tracked source. Attribution ran with only this unit's untracked worklog entry.
- Default `production` profile, manifest
  `c4a4a342e2243c2dcc430174606dde682393a2bd2e30acc83129027fcf572acc`,
  BF16 KV, one request, MTP candidate budget 3.

## Clean-source server result

The canonical ten-prompt category suite uses 64 greedy outputs, capacity 2048,
explicit MTP, two warmup arms per width and one measured run per arm/cell.
Timing is blocking OpenAI request completion, including prompt processing.

| Same-host server suite | Before batching | After batching |
| --- | ---: | ---: |
| MTP complete-request wall | 35.477980 s | 35.155692 s |
| MTP whole-call throughput | 18.039359 tok/s | 18.204733 tok/s |
| True AR complete-request wall | 57.863396 s | 57.845118 s |

MTP wall is 0.91% lower; the AR control changes by -0.03%. This single-run
confirmation is consistent with the counterbalanced library comparison in
[the batching evidence](../2026-11-12-gfx1151-qwen38-mixed-pair-batch/NOTES.md),
not a replacement for its repeats and additional held-outs.

Every per-arm output ID and the complete acceptance record match the baseline.
All requests engage MTP and conform to budget. Both artifacts retain status
`failed`: `general_en_explain` still differs between AR and MTP, first at output
index 51. The optimization did not introduce that difference. This is not a
passing AR-exact gate, and output mismatch alone does not establish a production
quality failure. No new AR/MTP headline or cross-host result is claimed.

## Public-request attribution

`attribution.py` exercises `LLM.generate_detailed()` after one warmup, with
prefix caching off, capacity 4096, 32 greedy outputs and ignored EOS. The short
prompt is 15 tokens. The long case is a neutral repeated-text control truncated
to 2053 tokens, not a task-quality example. Each context runs in its own process.

Python wrappers count copies with call stacks and observe tracked device
allocations after `LLM.close()` and garbage collection. `rocprofv3` records the
warmed request with cached builds. Profiled and non-profiled runs match output
IDs and non-timing diagnostic controls at each context. Stack inspection and
profiling perturb timings: these traces identify work, not request throughput
or a controlled short-versus-long comparison. They are not a same-process
phase-tagged cross-context study.

### Dense projections dominate short target verification

The short target graph has 1107 kernel dispatches per replay and runs 12 times.
Its summed kernel duration is 1450.970 ms:

| Kernel family | Summed duration | Share of target-graph kernel time |
| --- | ---: | ---: |
| Q5 rowtile projections | 356.566 ms | 24.57% |
| IQ4 local32 projections | 330.724 ms | 22.79% |
| Mixed IQ4/Q4 gate/up plus SiLU | 283.246 ms | 19.52% |
| Other strict IQ projections | 90.812 ms | 6.26% |
| Q6 target head | 55.408 ms | 3.82% |
| GDN recurrence | 35.744 ms | 2.46% |

Q5 and IQ4 projections together account for 47.37% of this target-graph kernel
sum. They are the first kernel-tuning candidates, ahead of recurrence or target
head work. The profile reports no private scratch for the target graph; it does
not establish whether bandwidth, instruction issue or occupancy limits these
kernels. Collect counters before selecting a tuning technique. Mixed-pair
batching reduced launches but did not remove its projection arithmetic.

### Long-context proposal graphs stop; target graphs still run

The short request uses 11 NextN proposal-graph replays. The long request uses
no proposal graph: `qwen35_gguf_nextn.py` caps that graph at context 1023 and
returns `eager_long_context` when the proposal would cross its bound.

This is not an all-graphs-off result. The profiler records seven long target
graph replays, each containing 2920 kernel dispatches, including scalar
projection and snapshot-copy work. Python `HipRuntime.graph_launch` wrappers
cannot see the native target owner's launches; the database is authoritative.
A follow-up should cover long proposal capture and reduce long target work,
not merely remove a bound without constructing a matching attention schedule.

### Snapshot copies are a separate host-side opportunity

On the short request, `prepare -> capture_initial -> _capture_state_index`
submits 1152 asynchronous state copies totaling 1,906,311,168 bytes. On the long
request it submits 768 totaling 1,270,874,112 bytes. These are Python submission
counts, not repeated copy nodes inside captured graphs. The count scales with
speculative cycles and rollback snapshots, not just prompt length.

`hip_gfx1151/__init__.py` excludes `linear_state_pair_copy/f32/chunked_i32` and
producer-folded snapshot variants with an independent-evidence comment. The
runtime therefore uses individual state copies. That comment names no observed
capability failure; it should not be treated as a permanent execution gate.
Exercise the existing pointer-table copy and transaction path on this backend,
then compare public rollback/control behavior and launch cost. No benefit from
that change has been measured here, and correctness requires preserving the
initial state needed by every serial/native fallback.

## Observed lifecycle issue

The 15-token request frees all tracked device allocations on close. After the
2053-token request, **727,711,744 bytes (694 MiB) in 12 allocations** remain
through close and garbage collection. Allocation stacks identify the module-global
Q6 full-GEMM caches in
`hipengine/kernels/hip_gfx1100/quant/gguf_q6_k_f16_rocblas_prefill.py`:

- 410 MiB of temporary dequantized weight storage;
- 28 MiB of FP32 output storage;
- 256 MiB of hipBLASLt workspace.

The dictionaries are keyed by shape and have no teardown route. This is an
observed lifetime/ownership gap, not evidence that every repeated request leaks
another 694 MiB. A repair needs explicit runtime/device/stream ownership and
safe retirement while other model sessions remain alive; blindly clearing
shared caches from one session's close would be unsafe.

The server suite also leaves 2,620,293,120 tracked bytes after teardown in both
baseline and candidate. The allocation-stack diagnostic used a different
workload and attributes only its own 694 MiB; it does not explain the entire
server residual. Batching did not increase either server allocation count or
tracked bytes.

## Commands and artifacts

Run from the repository root. The existing `run.sh` selects the local TheRock
environment at `~/miniforge3/envs/therock`. Commands below preserve the executed
arguments, using archived copies of the temporary diagnostic drivers.

```bash
P=benchmarks/results/2026-11-12-gfx1151-qwen38-mixed-pair-batch
D=benchmarks/results/2026-11-12-gfx1151-qwen38-dense-followup
bash "$P/run.sh" scripts/gguf_mtp_c1c8_server_bench.py \
  --model /models/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf \
  --backend hip_gfx1151 --quant gguf_q4_k_m --kv-storage bf16 \
  --execution-profile default --prompts benchmarks/prompts/mtpbench-code-general-ja.jsonl \
  --widths 1 --expected-mtp-widths 1 --max-tokens 64 --candidate-budget 3 \
  --max-sequence-length 2048 --correctness-contract ar_exact \
  --output /tmp/qwen38-dense-candidate.json
# Baseline: identical command at ff1c5ed41, output /tmp/qwen38-dense-baseline.json.
bash "$P/run.sh" "$D/attribution.py" --out /tmp/qwen38-attribution-short-warm.json
bash "$P/run.sh" "$D/attribution.py" --prompt-tokens 2053 --out /tmp/qwen38-attribution-long-warm.json
bash "$P/run.sh" -c 'import subprocess; from pathlib import Path; Path("/tmp/qwen38-hipcc-version.txt").write_text(subprocess.check_output(["hipcc", "--version"], text=True))'
bash "$P/run.sh" "$P/profile.py" /tmp/qwen38-attribution-short-profile "$D/attribution.py" --profile --out /tmp/qwen38-attribution-short.json
bash "$P/run.sh" "$P/profile.py" /tmp/qwen38-attribution-long-profile "$D/attribution.py" --profile --prompt-tokens 2053 --out /tmp/qwen38-attribution-long.json
python3 "$D/summarize.py"
```

`server.json` preserves compact before/after rows, generated IDs, failure
status, acceptance and memory counters. `short.json` and `long.json` contain
kernel/grid tables, copy stacks, route diagnostics and residual allocation
stacks. Raw input hashes are recorded. Raw profiler databases and logs remain
outside Git. `summarize.py` reads those local raw inputs without rerunning GPU
work. The full existing numerical and regression gates for the shipped batching
change remain in the linked batching artifact.
