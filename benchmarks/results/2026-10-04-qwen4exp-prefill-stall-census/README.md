# Qwen3.8 Flash Next prefill stall census — 2026-10-04 UTC

Prefill-only hardware-counter capture, taken to test whether the prefill's gap
against the comparator is uniformly distributed across kernel families (a
wall-time reading) or has family-level structure.

## Protocol

| | |
| --- | --- |
| Host | `gfx1151`, physical host `55ea6c509d0b49eea8de7094a1023668` |
| Model / quant | Qwen3.8 Flash Next, UD-Q4_K_XL, BF16 KV |
| Case | `code-p4096`, prefill only |
| Profile / chunk | production, prefill chunk 4096 |
| Harness | `scripts/qwen4exp_profile_gap.py --mode prefill --case-id code-p4096` |
| Profiler | `rocprofv3 --pmc ... -f csv`, compiler-version file captured outside the profiler, prewarmed |
| Captured | 5,840 kernel rows, 27.102 s device time |

## Result

The prefill is **not** uniformly stalled. The two largest kernels have opposite
stall profiles:

| kernel | share | LDS wait | VALU issue | waves/busy | inst/CU-cycle |
| --- | ---: | ---: | ---: | ---: | ---: |
| `q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4` | 18.26% | **30.82%** | 2.34% | 55.3 | 2.45 |
| `gguf_k_prefill_out_coltile_rowbatch_pipelined` | 16.57% | 0.03% | 7.76% | 36.2 | 3.76 |
| `gguf_q4_k_selected_dual_wmma_iu8_risk_prefill` | 16.38% | 0.15% | 9.49% | 23.9 | 2.67 |
| `q5_1_selected_wmma_iu8_risk_prefill_kernel<16>` | 8.92% | 0.10% | 7.33% | 31.4 | 2.75 |
| `qwen4_exp_gdn_prefill_f32_kernel<true,true>` | 5.93% | 0.01% | 6.99% | **9.6** | **0.92** |
| `qwen35_router_logits_f32_token_tile_dense_exact` | 4.55% | 0.35% | 1.30% | 60.1 | 1.92 |
| `q5_1_selected_sparse_exact_repair_row_publish` | 3.88% | 0.03% | 1.77% | 53.9 | 3.00 |
| `qsa_sparse_attention_h256_wave_rows_f32` | 3.31% | 0.21% | 8.26% | 40.6 | 4.03 |

HC-1 is the only kernel with a material LDS stall; everything else is at ≤0.35%.
`inst/CU-cycle` is `(SQ_WAVE_CYCLES/SQ_BUSY_CYCLES) x (VALU+SALU+LDS instructions)/SQ_WAVE_CYCLES`
against a ~4-per-cycle ceiling on a 4-SIMD CU — an estimate used to rank, not a
measured issue rate.

The GDN prefill kernel is **register-bound at the hardware maximum**: `VGPR_Count`
256, `LDS_Block_Size` 0, 128-thread workgroups, 6,144 grid blocks. The grid is not
the constraint, and the kernel uses no LDS at all.

## Limits

- `SQ_WAIT_INST_ANY` is accepted by `rocprofv3 --pmc` and **silently dropped** on
  this build (absent from the collected set, 0.00% for every kernel), so the
  non-LDS stall component cannot be named. There is no VMEM wait counter.
- One case, one model, one quant, one host (gfx1151, not the gfx1100 default
  hardware). Kernel durations under the profiler are inflated for launch-dense
  phases, so the **shares** are the result, not the absolute times.

## Files

- `census.json` — per-kernel shares, stall and issue rates, occupancy, and the
  kernel-trace resource record (VGPR, LDS, workgroup, grid) for the top kernels.
  The raw `rocprofv3` CSVs are 13 MB and are not committed; re-run the capture
  with the recipe in the `protocol` block of `census.json` to regenerate them.

Full write-up: `worklog/entries/20261004T150839.132976Z-lhl-qwen4exp-prefill-stall-census-41401e.md`.
