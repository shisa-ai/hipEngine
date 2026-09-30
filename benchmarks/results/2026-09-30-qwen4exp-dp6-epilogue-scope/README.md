# DP-6 epilogue-fusion scope: priced, and 3.72x smaller than the row was credited

Completes DP-6's stated gap - *"the SSM causal conv and the KV store are not split
out and remain unpriced, so the row is not yet rankable against `DP-5`'s 131.001
ms"* - from the committed X-3 capture. Read-only re-read of a committed artifact;
no GPU, no capture, no kernel. `performance_claim: false`.

- **Source:** `benchmarks/results/2026-09-28-qwen4exp-current-attribution/artifact.json`
  (`2789c19d6`), pointer `audit.prefill.windows[0].engagement_by_role_kernel`
  joined to `kernel_names` by `kernel_id`. Case `code-p4096`, p4096 window,
  `kernel_union_ms` = 15,741.963266.
- **Reproduce:** `scripts/` - the join is two fields; see `pricing.json`.

## The three components

| component | ms | % of union | kernels |
| --- | ---: | ---: | --- |
| **SSM causal conv** | **58.036** | 0.369 | `qwen4_exp_linear_attn_conv_prefill_exact_kernel` 57.584 (144 launches), `qwen35_linear_attn_conv_prefill_state_kernel` 0.452 (144) |
| **q/k-norm + RoPE** | **26.122** | 0.166 | `qsa_split_norm_rope_rows_f32_kernel<false>` 23.959 (48), `qsa_norm_rope_rows_f32_kernel<false>` 1.847 (48), `qsa_pool_norm_rope_f32_kernel` 0.316 (24) |
| **KV store** | **1.633** | 0.010 | `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel<float>` 1.633 (48) |
| **DP-6 scope** | **85.791** | **0.545** | |

`ple_dilated_depthwise_conv_f32_kernel` (4.431 ms) is excluded: it is the PLE
conv, a different owner, and appears under `ple:layers.*.ple_key`.

## What this decides

1. **DP-6 is rankable now, and it ranks BELOW DP-5.** DP-5 is 131.001 ms (0.832%);
   DP-6 is 85.791 ms (0.545%). The row's own caution was right.
2. **The 319.494 ms figure overstated the row by 3.72x.** That is the whole
   `elementwise_norm` bucket, which also holds `grouped_rmsnorm_bf16_f32_kernel`
   at **115.351 ms - 57.012 under `gr_read:layers.*.hc_attn_down` and 56.862
   under `gr_read:layers.*.hc_ffn_down`, i.e. hyper-connection reads, not norms
   DP-6 would fuse.** The row's warning *"do not build the `attn_qkv` epilogue on
   the strength of 319.494 ms"* is confirmed and quantified.
3. **The SSM causal conv is 67.6% of the row (58.036 of 85.791), so the action
   targets the right thing** - and the q/k-norm+RoPE half, which the row called
   "the smaller half", is smaller still at 26.122 ms.
4. **The KV-store half is worthless: 1.633 ms = 0.010%.** It should be dropped
   from the row's scope rather than carried as an unpriced term.

## Ceiling, not net

85.791 ms is the **owner cost**, i.e. a ceiling. DP-6 fuses the SSM causal conv
into the `ssm_in` GEMM's LD path; the conv still has to happen, so the recoverable
part is the eliminated pass over the wide activations, not the whole component.
This is the same ceiling-vs-net distinction recorded for DE-2 and MX-1.

## Limits

One model, one quant, one host (gfx1151, not the gfx1100 default), one case
(`code-p4096`), one window, one execution profile, prefill only. Attribution is a
per-role-per-kernel join inside a marked window and is diagnostic, not a wall
measurement. No speedup claimed and no kernel built.
