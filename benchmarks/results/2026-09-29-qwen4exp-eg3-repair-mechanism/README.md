# EG-3: the repair's mechanism, measured

**EG-3 stays closed.** The row was parked until "a measured mechanism (registers,
instruction count, occupancy) reopens it". Measured: **all three are falsified.**

The decisive evidence is not a new experiment - it is the already-measured
row-bitmap variant, which turns out to be a complete natural experiment.

## The census

`scripts/qwen4exp_repair_register_census.py` builds the kernel family with
`--save-temps` and reads `.amdhsa_next_free_vgpr` / `next_free_sgpr` plus the
device-code disassembly. Occupancy uses the gfx11 register file: 65536 VGPR per
CU, granules of 8, 32 waves/CU ceiling.

| kernel | VGPR | SGPR | static ins | waves/CU | occupancy |
|---|---:|---:|---:|---:|---:|
| `grouped_pair2` (the body the repair must reproduce) | 87 | 94 | 459 | 23 | 71.9% |
| **`sparse_exact_repair` (current)** | 57 | 63 | 336 | 32 | **100.0%** |
| `row_bitmap_repair` (the rejected alternative) | 49 | 43 | **238** | 32 | **100.0%** |

## Why this falsifies all three named mechanisms

**Registers.** The repair uses 57 VGPRs. Both 57 and the bitmap's 49 land in the
same 64-VGPR allocation granule, giving **32 waves/CU = 100% occupancy** - the
architectural ceiling. `__launch_bounds__(128, 2)` asks for only 25%. The repair
already has *more* occupancy headroom than the kernel it duplicates (71.9%).
There is no register-driven gain available, and none of the register pressure is
buying anything: it is already maximal.

**Occupancy.** At the ceiling. It cannot be improved by any means.

**Instruction count.** This is where the natural experiment is decisive. The
bitmap variant has **29% fewer static instructions** (238 vs 336) at the **same
100% occupancy**, and moves **3.3x fewer bytes** - and it measured **22% slower**
(3.5054 vs 2.8685 ms). Fewer instructions at equal occupancy produced a *slower*
kernel. Instruction count is not the lever.

## The natural experiment also kills the obvious candidate

Reading the bitmap kernel's source shows it *already implements* the optimization
one would propose from the instruction profile. Its own comment:

```
// One expert lookup per row instead of one per element.
```

The current kernel does a per-slot 64-bit division and a per-slot binary search
over expert starts; the bitmap does **one expert lookup per row**. That hoist -
plus fewer bytes and fewer instructions, at equal occupancy - is 22% slower.

So the entire class of "reduce per-slot scalar work" candidates is closed by
measurement, not by argument.

## What the mechanism actually is

The bitmap's per-row decomposition stages a whole activation row through shared
memory with a barrier:

```cpp
for (int64_t i = threadIdx.x; i < in_features; i += blockDim.x) {
  x_s[i] = bf16_bits_to_float(x_row[i]);
}
__syncthreads();
```

That adds a serialization point per row. The repair is limited by the **latency of
its dependent memory chain**, so trading fewer bytes and fewer instructions for an
extra barrier and an LDS round-trip is a net loss. This is consistent with every
prior observation in the row: grid size has no effect (17920 -> 262144 blocks:
3.05 -> 3.18 ms), 3.3x fewer bytes is slower, and L2-residency of the re-read
activation rows gains only 13.8%.

## What was and was not done

- **Done:** the mechanism is now measured rather than hypothesized, and the
  falsification is evidence-backed in both directions (a variant with strictly
  less work at strictly equal occupancy is slower).
- **Not done: no candidate was implemented.** A synthetic probe
  (`prologue_probe.hip`, kept for reproducibility) shows the per-slot prologue
  costs 136 instructions and a bit-exact magic-reciprocal division plus a cached
  expert search costs 70 - a 48.5% prologue reduction, with `s_delay_alu` stalls
  falling 16 -> 4. That number is **not** a projected win: the bitmap result is
  direct evidence that fewer instructions does not make this kernel faster, and
  the probe measures the prologue in isolation, not in situ. It is recorded so the
  next person does not re-derive it, not as a queued candidate.

## Limits

- One profile (`prefill`, `-mllvm -amdgpu-unroll-threshold-local=600`), one arch
  (gfx1151), one compile.
- The census is **static**: occupancy is the register limit only, not a runtime
  measurement. It does not account for LDS or scratch.
- The 22% bitmap figure is from the existing `20260917` packet, not re-measured
  here; this packet adds the register/instruction context that explains it.
