# The production capture kwargs are equivalent to the runner defaults, and the loop split is not an attribution

Two measurement-validity results for the canonical AR bench. **No kernel,
default, flag or dispatch changed and no retained metric row moved.**

## 1. `capture_logits=True` costs nothing, so the retained decode rows are representative

`qwen4exp_canonical_ar_bench.py` calls `runner.step(current)` with the runner's
defaults, which are `capture_logits=True`, `capture_target_hidden=True`,
`token_id_resident=False`. The production text generator
(`Qwen4ExpGGUFTextGenerator`, uncontrolled greedy request) instead passes
`_qwen4_exp_compact_step_kwargs()`: `capture_logits=False`,
`capture_target_hidden=False`, `token_id_resident=True`. That difference is not
cosmetic — `capture_logits=True` copies the whole vocabulary to the host and
takes a host `argmax`, and it forces a `device_synchronize()` inside every
decode step.

If that mattered, every retained decode row would be a pessimistic measurement
of a configuration no serving client runs. It does not matter. The new
`--capture-mode {full,compact}` flag measures both, one case per process,
production profile, chunk 1024, warm PLE, one warmup and one measured
repetition:

| mode | p512 decode tok/s | p4096 decode tok/s | p512 prefill tok/s | p4096 prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| `full` (runner defaults) | 20.738 | 17.220 | 264.717 | 268.668 |
| `compact` (production generator) | 20.755 | 17.173 | 264.797 | 271.105 |
| delta | +0.08% | -0.27% | +0.03% | +0.91% |

All four deltas are at or inside this bench's run-to-run spread. The per-step
`device_synchronize`, the 608 KB logits D2H and the host `argmax` are fully
overlapped with device work, so **the retained decode and prefill rows already
describe the production path** and do not need re-baselining. The artifact's
protocol block records `capture_mode`, `prefill_kwargs` and `step_kwargs` so a
later reader can tell which configuration a row was taken in.

## 2. The host/device loop split is a loop split, not an attribution

The bench now also splits the decode loop at its single trailing synchronize
into `decode_host_enqueue_ms` and `decode_device_drain_ms`. Read those as
window sizes, not as an attribution: `runner.step()` ends with a **blocking**
device-to-host copy of the argmax result (`_finish_lm_head` ->
`copy_device_to_host`), so the "host enqueue" window already contains the device
wait and the trailing drain is zero by construction.

`decode_device_drain_ms_per_step` is **0.000 in all four rows above**, in both
capture modes, and that does not mean the step is host-bound. It means the last
step's blocking readback had already drained the device before the loop's own
synchronize ran. The earlier reading of a zero drain as "the GPU completed all
128 queued steps before the host finished enqueuing them"
(`20261004T073816`) is **withdrawn**: the same zero appears in `compact` mode,
where no explicit per-step synchronize exists at all, because the blocking
readback supplies it.

Measuring the host/device split requires removing that readback, not splitting
the loop around it. See
`benchmarks/results/2026-10-04-qwen4exp-decode-readback-split/`.

## Commands

```
uv run python scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --case-id code-p512 --prefill-chunk-size 1024 --execution-profile production \
  --capture-mode <full|compact> --ple-cache-mode warm --ple-telemetry \
  --warmups 1 --repetitions 1 \
  --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build \
  --output <out>.json
```

One case per process, ~45 s cooldown between runs. Host: AMD Radeon 8060S /
`gfx1151` (Strix Halo Framework), ROCm 7.15.26333, BF16 KV, greedy AR.
