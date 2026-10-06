# Qwen3.8 Flash Next production prefill owners after F16 publication

Tracked-clean `3e166ef12` runs Qwen3.8-Flash-Next UD-Q4_K_XL, BF16 KV,
chunk4096 and canonical `code-p4096` on Radeon 8060S host `gfx1151`,
machine-id `55ea6c509d0b49eea8de7094a1023668`. These are current diagnostic
owner prices, not an old→new throughput comparison. Production manifest:
`be32e7193f30c4c4a65f5cbaf99b62033fd6458edd295fc2d7806e734001469b`.

| Kernel owner | Mean ms/pass | Launches/pass | Kernel-union share |
| --- | ---: | ---: | ---: |
| Dense F32 pipelined DPP coltile | 2241.114 | 109 | 18.01% |
| MoE Q4_K gate/up iu8 risk | 2218.615 | 47 | 17.83% |
| Q8_0 GR-up ordered DPP | 1571.855 | 96 | 12.63% |
| MoE Q5_1 down iu8 risk | 1206.535 | 43 | 9.70% |
| Checked-index GDN recurrence | 743.301 | 36 | 5.97% |
| Router F32 ordered DPP | 544.003 | 48 | 4.37% |
| Q5_1 sparse exact down repair | 535.045 | 43 | 4.30% |
| Q8_0 dense-wide full tiles | 464.017 | 219 | 3.73% |
| Full-attention context | 454.615 | 12 | 3.65% |
| Sparse attention rows | 448.076 | 12 | 3.60% |
| Padded Q8_0 dense-wide HC-down tails | 187.558 | 70 | 1.51% |
| Grouped-norm F32/F16 publisher | 119.042 | 70 | 0.96% |

Both windows attribute all 2,691 launches. Each has 70 native publishers and
zero HC-down staging casts. Kernel unions are 12,443.236/12,438.157 ms in
12,761.947/12,966.555 ms windows. Each has 27 device-copy rows, leaving
318.041/527.727 ms outside the kernel-plus-copy union. HIP API host intervals
overlap device work and are not additive overhead. The unprofiled 12.844 s
is one sample, not a throughput certificate. Differences from independent
historical captures are not controlled performance deltas.

Warm/profile full-logit SHA256 agrees:
`219578ac13b323d50ecb8cd720f2f818517874f766c934883303549c96804e25`.
Both named-profile processes have no overrides, zero steady allocation/byte
growth and zero allocations/bytes after close. No diagnostic kernel
substitution is installed. The quantized launch census is empty, so no
shape/FLOP join is inferred.

## Next bounded screen: GR-up K320/full-row specialization

GR-up still costs 1,571.855 ms across 96 calls. Its current cached native
DPP symbol has 57 static VGPR, 64 SGPR, 512 B static LDS, no spills and no
private segment. The same cached object's disassembly has 748 instructions,
including generic row-validity branches and variable K-loop control; the
source still guards each of four activation-row loads inside the K loop.
At the model's K320, each lane executes two or three ordered K iterations.

Screen a separately registered specialization for K320, four branches,
H2560 and full four-row batches. Compile-time K and full-row admission can
remove generic control without widening the row batch, introducing staging,
changing operand precision, removing gate stores or changing the FMA/reduction
association. Keep ascending per-lane K FMAs, ordered DPP/four-wave publication,
`expf`, branch mean and both materialized outputs. Retain the generic DPP
parent for every shape/tail miss. This is not a rerun of the rejected row-width,
GR-staged-F32, no-gate-store or operand-class mechanisms.

Require RED admission/fallback tests, raw gate/mixed parity, CPU-reference
floor, guards, input immutability, nondefault stream, repeats and neighbor
isolation. Inspect cached resources, then counterbalance the complete natural
GR-read operation at 512/1024/4096 rows across the four categories. Native
state/KV/logit and canonical engine qualification follow only if the operation
wins. Any arithmetic difference must be reviewed under production numerical
gates rather than rejected solely for nonidentity. No speedup is predicted,
and no kernel or default changes in this pricing packet.

The static counts above use the actual cached code object, not the older
assembler/standalone-object census. Those counts are not directly comparable.
Register counts do not establish runtime occupancy. The trace does not hash
loaded libraries; current cached-build consistency is not retroactive
loaded-binary identity. One 1,310,720 B queue scratch allocation precedes
measurement, and none occurs inside either window. GDN reports 28 B private
segments; no scratch-throttle causal claim follows.

## Reproduction

[`owners.json`](owners.json) records the exact launcher, compiler/host/source
identities, role aggregates, top-24 resource rows, current dense/GR cached
metadata and raw-evidence hashes. Raw CSVs, libraries and disassembly stay
outside Git under `/tmp/qwen4exp-post-dp5-prefill-20261006/`.

The launcher sets TheRock loader libraries and `HIPENGINE_HIP_ARCH=gfx1151`,
then runs `uv run --no-sync python scripts/qwen4exp_profile_gap.py` unprofiled
with production/code-p4096/chunk4096/max-sequence4352 and one repetition.
A second process runs two repetitions with `--require-cached-build --profile
--role-markers` under `rocprofv3 --kernel-trace --marker-trace --hip-trace
--memory-copy-trace --scratch-memory-trace`. Both use
`/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt`. Audit with:

```bash
uv run --no-sync python scripts/qwen4exp_trace_audit.py \
  --trace-dir /tmp/qwen4exp-post-dp5-prefill-20261006/raw \
  --measure-prefix qwen4exp_prefill_p4096_ \
  --output /tmp/qwen4exp-post-dp5-prefill-20261006/audit.json
```

Use this tree's `uv run --no-sync`, not an ambient editable install. One
category/shape is diagnostic; candidate qualification needs its own natural
input coverage.
