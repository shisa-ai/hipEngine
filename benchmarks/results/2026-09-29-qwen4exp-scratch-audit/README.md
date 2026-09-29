# X-5 scratch-throttle audit: no kernel in the tree has any scratch segment

X-5 records halo-box PR #87: after a kernel with a private segment ran
(`flash_attn_ext_f16<256>`, 252 B/lane), their command processor throttled **every later
scratch-enabled dispatch ~3.3x**; their Q5_1 expert-down GEMV went **220 -> 67 us** by
replacing `volatile` float rounding barriers with a register-only `asm` barrier,
bit-identical.

X-5's next step: *"List `private_segment_fixed_size` for every Qwen4Exp kernel in the
prefill and decode census. Any non-zero scratch on a hot kernel warrants disassembly and
ordered warm-process A/B."*

**Audited. The answer is zero everywhere, so no A/B is warranted.**

## Coverage

| | |
|---|---:|
| `.hip` sources in the tree | 101 |
| compiled | **101** |
| failed | **0** |
| **kernels audited** | **2,245** |
| **kernels with non-zero `private_segment_fixed_size`** | **0** |
| kernels with non-zero `group_segment_fixed_size` (static LDS) | 0 |

All 101 `.hip` files live under `hipengine/kernels/hip_gfx1100/`; `hip_gfx1151` compiles
those same shared gfx11 sources, so this is the **whole** kernel tree, not a sample. The
audit covers prefill and decode alike - it is per-kernel, not per-workload.

Method: `hipcc -c --save-temps -O3 -mllvm -amdgpu-unroll-threshold-local=600
--offload-arch=gfx1151 -mcumode`, then read `.private_segment_fixed_size` and
`.group_segment_fixed_size` from the assembler directives of every `@function` in every
emitted `.s`.

## What this means

**The halo-box mechanism has neither a trigger nor a victim here.** Their throttle needed
a kernel that allocates a private segment; no kernel in this tree allocates one, and none
therefore depends on one. There is nothing to disassemble, nothing to A/B, and no
register-only-barrier candidate to test - X-5's stated precondition is not met anywhere.

X-5's named trigger is also not ours: `flash_attn_ext_f16` is a **llama.cpp** kernel name
and **does not appear** in this tree. Our full-attention kernels are the
`qwen35_paged_full_attn_*` family.

## A secondary finding worth recording

`private_segment_fixed_size` is non-zero whenever a kernel uses a stack - including
**register spills**. Zero across all 2,245 kernels means **no kernel in this tree spills
registers**, and none uses stack scratch. That is a clean register-allocation result and
it independently explains why the register censuses in this campaign find kernels sitting
at their occupancy ceilings rather than paying spill traffic.

## Limits

- **Static property, not a measurement.** This reads the compiler's declared segment
  sizes. It does not run anything and does not measure dispatch time, so it cannot
  *observe* a throttle - it establishes only that the condition X-5 names is absent.
- A throttle mechanism triggered by something *other* than a private segment would not be
  detected here. X-5's evidence names the private segment, so that is what was tested.
- One arch (`gfx1151`), one compiler version, one set of flags. A different build profile
  could in principle spill where this one does not.
- Compile-only: no numerical or performance claim.

## Verdict

**X-5's audit step is complete and closes with nothing to pursue.** Reopen only if a
kernel acquires a private segment, or if a throttle is observed with a different trigger.
