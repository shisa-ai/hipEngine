# QA-2: the HEADS 2->4 evidence is confounded, and the confound is inherent

QA-2 rests on one measurement: "HEADS 2 -> 4 halves K/V traffic and buys only
1.08x", read as proof that the kernel is **issue-limited, not traffic-limited**.
The census shows that experiment changed **three things at once**, and that the
occupancy loss is not incidental - it is a register-file limit the compiler
reports it cannot meet.

This does not overturn QA-2's direction. It **corrects the diagnosis and
strengthens the case for tensor cores**, for a different reason than the row
states.

## The census

`scripts/qwen4exp_repair_register_census.py --source .../qwen4_exp_qsa.hip`, same
gfx11 model (65536 VGPR/CU, granule 8, 32 waves/CU ceiling). The kernel is
`template <bool PAGE256, int HEADS = 1>` with `__launch_bounds__(32)` - one wave32
per block, so waves/CU is also blocks/CU.

| instantiation | VGPR | static ins | waves/CU | occupancy |
|---|---:|---:|---:|---:|
| `HEADS=1` | 40 | 262 | 32 | **100.0%** |
| `HEADS=2` | 66 | 331 | 28 | **87.5%** |
| `HEADS=4` | 108 | 442 | 18 | **56.2%** |

## What the 1.08x actually measured

Going HEADS 2 -> 4 changed three variables simultaneously:

| | HEADS=2 | HEADS=4 | effect |
|---|---:|---:|---|
| K/V traffic | - | **halved** | helps |
| occupancy | 87.5% | **56.2%** | hurts (36% fewer resident waves) |
| static instructions | 331 | **442** (+33.5%) | hurts |

A net 1.08x from three opposing changes cannot isolate an issue-limit from an
occupancy loss. The traffic saving and the occupancy loss very nearly cancel,
which is a sufficient explanation of the 1.08x on its own.

## The occupancy loss is inherent, not a compiler choice

The obvious objection is that 108 VGPRs is just an allocation decision. Tested by
forcing the compiler with `__launch_bounds__(32, N)` on that kernel:

| demanded occupancy | HEADS=4 VGPR | spills | compiler |
|---|---:|---:|---|
| `(32, 16)` | 96 | 0 | `-Wpass-failed` |
| `(32, 32)` | 108 | 0 | `-Wpass-failed` |
| `(32, 40)` | 108 | 0 | `-Wpass-failed` |
| `(32, 64)` | 108 | 0 | `-Wpass-failed` |

The compiler states it directly:

```
warning: failed to meet occupancy target given by 'amdgpu-waves-per-eu' in
  '...qsa_sparse_attention_h256_wave_rows_f32_kernelILb1ELi4EEE...':
  desired occupancy was 64, final occupancy is 12   [-Wpass-failed]
```

At a 32-VGPR demand - a 3.4x reduction - it still emits 108 registers and
**spills nothing**. So the live range genuinely requires the registers: it cannot
lower occupancy further without spilling, and it declines to spill. The HEADS=4
formulation holds a four-head tile in the register file, and that is what costs
the occupancy.

## Why this supports tensor cores

The corrected mechanism is **register-file capacity**, not issue width:

- Widening the scalar tile buys traffic but pays occupancy, and the two nearly
  cancel. That is why the scalar route has no cheap win here.
- The register cost comes from holding the wide tile and its intermediate
  products in VGPRs. Tensor cores perform the tile arithmetic on the matrix units,
  so the same tile does not need those intermediates live in the register file.
- So tensor cores are the route that can take the traffic saving **without**
  paying the occupancy that defeated the scalar wide tile.

That is a stronger and more specific justification than the row's "issue-limited",
and it is measured.

## Limits

- **Static census only.** Occupancy is the register limit; it does not account for
  LDS or scratch, and no kernel was timed here.
- The 1.08x and the bandwidth fall (1238.8 -> 667.1 GB/s) are the existing
  `20260927T090500` measurements, **not re-measured**; this packet adds the
  register/occupancy context that explains them.
- One profile, one arch, one compile.
- **No candidate was implemented.** QA-2's tensor-core build remains the
  experiment; this packet corrects its stated mechanism and gives it a measured
  basis, it does not discharge it.
