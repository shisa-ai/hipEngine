# Current Qwen3.8 Flash Next prefill owners

On 2026-10-05 UTC, clean source `eb02f3d87` ran Qwen3.8-Flash-Next
UD-Q4_K_XL on Radeon 8060S host `gfx1151`, machine-id
`55ea6c509d0b49eea8de7094a1023668`. The production manifest is
`850829d509ebf741403e71e30450b8b7b681994f16e0f02038ba672c1f493d5b`.
This is a diagnostic owner capture, not a model-speed comparison.

The exact-token canonical `code-p4096` case uses a 4096-token chunk and
max sequence length 4352. An unprofiled warmup prebuilds outside the
profiler; the profiled child requires cached builds. Two separately marked
passes each attribute all 2,761 launches. The final full-logit hash matches
between warm and profiled processes. Both runs show zero steady allocation
or byte growth and zero allocations/bytes after close.

| Kernel owner | Mean ms/pass | Launches/pass | Kernel-union share |
| --- | ---: | ---: | ---: |
| Dense F32 pipelined DPP coltile | 2240.924 | 109 | 17.81% |
| MoE Q4_K gate/up iu8 risk | 2218.764 | 47 | 17.64% |
| Q8_0 GR-up ordered DPP | 1534.263 | 96 | 12.20% |
| MoE Q5_1 down iu8 risk | 1206.801 | 43 | 9.59% |
| GDN recurrence | 783.099 | 36 | 6.23% |
| Router F32 tile4 logits | 617.264 | 48 | 4.91% |
| Q5_1 sparse exact down repair | 526.680 | 43 | 4.19% |
| Q8_0 dense-wide | 469.238 | 219 | 3.73% |

The kernel unions are 12,580.107 and 12,579.225 ms in measured windows of
12,884.389 and 13,029.441 ms. Kernel-plus-copy union leaves 303.602 and
449.232 ms outside device intervals. Each window has 27 device-copy rows
and no scratch allocation/free operations. This does not test scratch
trigger/victim throttling. HIP API intervals overlap device work and are
not additive host overhead. The unprofiled measured wall is 12.848 s, one
sample only; it is not a throughput certificate.

## Next experiment

The router source preserves 256 per-thread FMA accumulations, then reduces
with eight block-wide LDS/barrier levels. Keep strides 128/64/32 in LDS,
then complete strides 16/8/4/2/1 with ordered wave DPP instructions. This
preserves the reduction tree instead of repeating the previously failed
warp-per-token reassociation. Removing five block barriers and shared-memory
round trips is a mechanism hypothesis; no saving is measured yet.

Before selection as a default, require direct raw-bit parent parity on
cancellation/tail fixtures, exact real-activation router values and expert
selections, exact state/full-logit model trajectories, cached trace engagement
and counterbalanced complete-owner/model timing. If arithmetic differs,
classify it separately and apply the production envelope; do not claim parity.

## Reproduction

Set ROCm libraries and compiler-version file as recorded in the worklog.
Warm outside the profiler:

```bash
uv run --no-sync python scripts/qwen4exp_profile_gap.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --mode prefill --case-id code-p4096 --execution-profile production \
  --prefill-chunk-size 4096 --max-sequence-length 4352 --repetitions 1 \
  --compiler-version-file /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt \
  --output /tmp/qwen4exp-current-prefill-20261005/warm.json
```

Profile the same arguments with `--repetitions 2 --require-cached-build
--profile --role-markers`, using `rocprofv3 --kernel-trace --marker-trace
--hip-trace --memory-copy-trace --scratch-memory-trace --output-format csv`.
[owners.json](owners.json) records the exact command, hashes, resources,
per-pass role attribution and correctness scope. Raw CSVs remain under
`/tmp/qwen4exp-current-prefill-20261005/raw` and are not committed.
The quantized launch census was empty; no shape/FLOP join is claimed.
Historical source stacks are not a controlled before/after comparison.
