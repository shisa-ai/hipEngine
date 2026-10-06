# Qwen3.8 Flash Next production prefill owners after tail promotion

Clean source `ea818e2c6` runs Qwen3.8-Flash-Next UD-Q4_K_XL, BF16 KV,
chunk4096 and canonical `code-p4096` on Radeon 8060S host `gfx1151`,
machine-id `55ea6c509d0b49eea8de7094a1023668`. This is diagnostic owner
pricing, not a controlled speed comparison. Production manifest:
`a3baefbeb6237a4a24b4751c8f46f5c0cfa4c5ef166a289eb7a5fbe124dfdcb0`.

| Kernel owner | Mean ms/pass | Launches/pass | Kernel-union share |
| --- | ---: | ---: | ---: |
| Dense F32 pipelined DPP coltile | 2255.260 | 109 | 17.92% |
| MoE Q4_K gate/up iu8 risk | 2218.753 | 47 | 17.63% |
| Q8_0 GR-up ordered DPP | 1598.935 | 96 | 12.71% |
| MoE Q5_1 down iu8 risk | 1207.148 | 43 | 9.59% |
| Checked-index GDN recurrence | 781.876 | 36 | 6.21% |
| Router F32 ordered DPP | 546.797 | 48 | 4.35% |
| Q5_1 sparse exact down repair | 539.257 | 43 | 4.29% |
| Q8_0 dense-wide, full tiles | 464.726 | 219 | 3.69% |
| Padded Q8_0 dense-wide HC-down tails | 167.362 | 70 | 1.33% |
| F32→F16 staging | 162.187 | 289 | 1.29% |

Both windows attribute all 2,761 launches. Kernel unions are
12,585.508/12,582.179 ms within 12,873.049/13,050.192 ms windows.
Each window has 27 copy rows; the kernel-plus-copy residual is
286.351/466.879 ms. HIP API host intervals overlap device work and are not
additive overhead. The unprofiled 12.855 s is one sample, not a throughput
certificate. Independent historical captures are not an old→new comparison.

Warm/profile final full-logit SHA256 agrees:
`219578ac13b323d50ecb8cd720f2f818517874f766c934883303549c96804e25`.
Both processes have zero steady allocation/byte growth and zero allocations/
bytes after close. The named production profile is intact with no overrides.
Native engagement confirms the padded tail, checked-index GDN, router DPP
and GR-up DPP defaults; no diagnostic leaf substitution is installed.

## Current resources and next screen

Three selected current cached symbols were audited, not the whole tree:

| Symbol | Private bytes/lane | Static VGPR | VGPR spills | Static scratch instructions |
| --- | ---: | ---: | ---: | ---: |
| Q4_K dual iu8-risk, 3 planes / 16 rows | 0 | 211 | 0 | 0 |
| Checked-index D128 GDN wave norm | 28 | 256 | 6 | 6 |
| Padded F16-input 128×256 dense-wide tail | 0 | 253 | 0 | 0 |

Profiler scratch fields agree with these cached code objects. The profiler
rounds VGPR allocation to 216/256/256. The trace does not hash loaded
libraries; object hashes establish current-build consistency, not retroactive
loaded-binary identity. One 1,310,720-byte scratch allocation occurs before
measurement and none inside either window. No scratch-throttle causal claim
follows.

The next bounded experiment is **EG-5: checked Q4_K gate/up indexing**.
The 2,218.753 ms owner still uses int64 metadata, row/column counters and
activation/output/weight offsets. Its cached disassembly has 4,388 static
instructions, including 69 vector add-carry pairs, 33 vector 64-bit shifts
and 22 vector 64-bit multiply-adds. These counts motivate a screen, not a
speedup forecast. The kernel is already scratch-free.

Register a separate checked-int32 sibling for the existing three-plane,
16-row body. Keep the int64 raw host ABI and reject unsafe dimensions before
launch. The admitted domain must bound activation, output and weight byte
ranges plus row/map/risk indices; retain the wide parent outside it. Do not
change activation quantization, sum/Kahan association, risk criterion, risk
membership or the exact sparse repair. This differs from the failed LDS
staging, operand-class and repair-removal experiments.

Require RED bounds/overflow tests, raw BF16 repaired-output parity, risk
count and sorted membership parity (atomic queue order is not deterministic),
empty-expert/row-tail coverage, nonfinite/subnormal inputs, risk-capacity
behavior, guards/input immutability and CPU reference. Then inspect cached
resources and counterbalance the complete packing+iu8-risk+repair+SiLU
operation on natural routing. No default change follows this pricing packet;
full-model state/ownership qualification and canonical engine A/B follow only
if the operation screen survives. If arithmetic differs, apply the production
numerical gate rather than rejecting solely on bit identity.

## Reproduction

`owners.json` contains the exact launcher, source/compiler/host identities,
commands, all resource rows for the 24 largest owners, full role aggregates,
selected object/opcode metadata, lifecycle checks and raw-evidence hashes.
Raw CSVs, libraries and disassembly remain outside Git under
`/tmp/qwen4exp-post-tail-prefill-20261006/`.

The launcher sets TheRock libraries and `HIPENGINE_HIP_ARCH=gfx1151`, then
runs `uv run --no-sync python scripts/qwen4exp_profile_gap.py` unprofiled
with production/code-p4096/chunk4096/max-sequence4352 and one repetition.
The second process uses two repetitions, `--require-cached-build --profile
--role-markers`, under `rocprofv3 --kernel-trace --marker-trace --hip-trace
--memory-copy-trace --scratch-memory-trace`. Both use
`/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt`. Audit with:

```bash
uv run --no-sync python scripts/qwen4exp_trace_audit.py \
  --trace-dir /tmp/qwen4exp-post-tail-prefill-20261006/raw \
  --measure-prefix qwen4exp_prefill_p4096_ \
  --output /tmp/qwen4exp-post-tail-prefill-20261006/audit.json
```

Use this tree's `uv run --no-sync`, not an ambient editable install. The
quantized launch census is empty, so no tensor-shape/FLOP join is inferred.
One category/shape is diagnostic; candidate timing and qualification need
their own natural-input coverage.
