# Q8_0 wide-row dense prefill: the shipped-default path launches the same kernels

Verifies that a caller who names no execution profile gets the wide-row Q8_0
dense prefill route on the certified lane, rather than the exact coltile chain.
This is a routing check, not a performance measurement: no rate here is
comparable to any other artifact, and no scoreboard row changes.

- Host: Strix Halo, `hip_gfx1151`, 125 GB unified memory.
- Model: `Qwen3.8-Flash-Next-UD-Q4_K_XL` (111 GB, four shards), quant
  `gguf_ud_q4_k_xl`, model plugin `qwen4_exp_gguf`.
- Workload: `--mode prefill --case-id code-p512` (512 prompt tokens), three
  repetitions, launch census on, no `--override` flags.
- Command, default arm:
  `scripts/qwen4exp_profile_gap.py --model-root <pack> --mode prefill
  --case-id code-p512 --execution-profile default --launch-census census.json
  --output run.json` with `HIPENGINE_KERNEL_CENSUS=1` exported inside the
  hermetic `therock.sh` wrapper (it uses `env -i`, so an unexported variable
  reads as an uninstrumented run).
- Command, explicit arm: the same, without `--execution-profile` (which defaults
  to `production`), run at the promotion commit `15b056111`.

`--execution-profile default` is not a fourth profile: it reproduces
`hipengine.llm`'s shipped default, where no caller request is resolved by
`resolve_default_execution_profile`. That call returned `production` for this
combination, with `fell_back_to_strict: false`.

## Result

Both arms produce a **byte-identical** census file:
`sha256 2b5dbcaf32b2e72b00b22bdba81ac18baa5322463ad48b58c230e30a5c59fd3e`,
113377 bytes, so the shipped default is the same launch set and not merely the
same route counts.

| Launch | Roles |
| --- | ---: |
| `hipengine_gguf_q8_0_dense_wide256_f32_f32_out` | 264, layers 16-47 |
| `hipengine_gguf_q8_0_gemv_coltile8_rowbatch4_wave_scale_f32_f32_out` | 134, layers 0-15 |
| `hipengine_gguf_q8_0_pack8_gemv_f32_f32_out` | 3 |
| `hipengine_gguf_q8_0_wmma_prefill_*` | 0 |

1203 launches total over the three repetitions. No coltile role appears at layer
16 or above, and no layer below 16 runs the wide kernel, so the promoted window
is the only thing the default changed. Both arms also produce the same
`logits_sha256` `e15dce7949b5402d2ef6…` and token `248068`, matching the
pre-promotion default.

`census.json` is the shipped-default arm; `census-explicit-production.json` is
the explicit production arm. They are committed twice so the equality above is
checkable from the artifact alone.

The correctness envelope for this route at this scope is separate:
[`2026-09-17-q8-dense-wide-16-47-gate`](../2026-09-17-q8-dense-wide-16-47-gate/README.md).
