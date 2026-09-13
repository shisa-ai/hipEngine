# Q4 dual WMMA prefill tile screen: reviewed evidence

Measurement date: September 13, 2026. Diagnostic only; no default changed.
The original timing files are unchanged. The later review withdraws the
512-VGPR occupancy model and source-instruction-count performance predictions.

## Measured Result

Real Qwen3.8-27B Q4_K_M `blk.0.ffn_gate/up` tiles, K5120/N17408, on the
Framework gfx1151 Radeon 8060S. HIP-event bursts of four launches, two warmups,
three measured repetitions. Times are milliseconds:

| Token rows | Parent tile256 | tile128 | tile64 | tile48 | tile32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256 | 3.270 | 3.895 | 7.121 | 8.280 | 8.277 |
| 512 | 6.629 | 7.687 | 14.003 | 15.178 | 16.358 |
| 1024 | 13.129 | 15.037 | 27.819 | 30.360 | 32.526 |
| 2048 | 26.023 | 29.868 | 55.562 | 59.229 | 64.884 |
| 4096 | 52.245 | 59.456 | 110.963 | 118.227 | 129.304 |

All 20 non-parent cells match the parent bits. These smaller token tiles lose
on the measured shapes. Duration divided by block count is an amortized
quantity, not a measured block latency.

The separate synthetic single-matrix screen uses three warmups and ten
iterations. At 4096 tokens, shared_b (48 columns) takes 30.819 ms,
shared_b2w2 (32 columns) 39.109 ms, shared_b2w4 (32 columns) 42.970 ms,
lowvgpr (16 columns) 56.910 ms and lowvgpr48 (16 columns) 52.145 ms.
Other geometry dimensions also change. This is not an isolated column-width
ablation of the fused dual kernel.

## Resource Correction

gfx1151 wave32 has 1536 physical VGPRs per SIMD with a 24-register allocation
granule. A reported count of 248 rounds to 264, permitting at most five waves
by registers alone. Actual residency also depends on LDS, CU/WGP mode,
workgroup placement and other constraints; no occupancy counter was measured.
The old two-wave ceiling and 171-register threshold were incorrect.

`resource_table.json` is the unmodified historical capture. Its derived
occupancy fields are obsolete. The corrected assembler consumes only raw
compiler counts and recomputes register-only ceilings. A single caller-supplied
workgroup size cannot describe all 49 kernels in that capture.

## Open Experiments

The later [col16 A/B](../2026-09-13-q4-dual-prefill-col16-arm-ab-rejected/README.md)
rejects two concrete dual-kernel variants. Neither that result nor this screen
settles all tile geometry. The proposed decode-time explanation is not proven.

- Decode lane/load restructuring: inspect ISA, stalls and complete-owner cost.
- Wider column tiles: redesign decode coverage and verify epilogue resources.
- More token rows: assess the larger LDS union or register-local epilogue.
- Residency retuning: use target- and mode-correct limits before screening.

Every candidate needs exact ownership, its declared production numerical/task
gate, a registered strict fallback and same-host complete-owner/full-model
validation before promotion. Do not impose bit identity as the only promotion
criterion.

## Reproduce The Analysis

```bash
python3 benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/assemble.py
python3 benchmarks/results/2026-09-13-q4-dual-prefill-tile-knob-screen/assemble.py --check
```

Inputs: `resource_table.json`, `rowtile_prefill.json`, `col_tile_owners.json`.
Their hashes are in the regenerated artifact. The original benchmark commands
and experimental reasoning remain available at commit `88e0e6b38`.
No GPU run is needed to regenerate the corrected analysis.
