# DP-6 pass pricing: the SSM causal conv is 2.1x the attn_qkv side, so the prescribed order inverts

**What this is.** `DP-6` (projection epilogue fusion) was **partially priced**: the X-3
capture's 319.494 ms "norms" bucket upper-bounded the q/k-norm plus RoPE passes, but the
SSM causal conv and the KV store were not split out, so the row could not be ranked
against `DP-5`'s 131.001 ms and its own experiment column left a conditional unresolved
(*"build the `attn_qkv` epilogue first if it is the larger"*).

This attributes those passes from the **same capture's raw kernel trace**, which is still
on disk. **No new measurement was run and no kernel was built.** `performance_claim: false`.

## Source

| | |
| --- | --- |
| capture | `benchmarks/results/2026-09-28-qwen4exp-current-attribution/` |
| kernel trace | `/tmp/qwen4exp-ledger-20260928-current/role-trace/gfx1151/3033290_kernel_trace.csv` |
| trace integrity | sha256 recorded in that artifact's `raw_sha256` |
| case | `code-p4096`, 4096 prompt tokens, `prefill_chunk_size: 1024` |
| host | `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668` |
| kernel union | **15,741.963 ms** (10,708 launches, 100% attributed) |
| duration basis | `End_Timestamp - Start_Timestamp`, ns → ms, summed per kernel name |

## Result

| pass | kernel(s) | calls | ms | % of union |
| --- | --- | ---: | ---: | ---: |
| **SSM causal conv** | `qwen4_exp_linear_attn_conv_prefill_exact_kernel` | 288 | 115.213 | |
| | `qwen35_linear_attn_conv_prefill_state_kernel` | 288 | 0.908 | |
| | **subtotal** | | **116.121** | **0.738%** |
| q/k norm + RoPE | `qsa_split_norm_rope_rows_f32_kernel<false>` | 96 | 47.872 | |
| | `qsa_norm_rope_rows_f32_kernel<false>` | 96 | 3.693 | |
| | `qsa_pool_norm_rope_f32_kernel` | 48 | 0.634 | |
| | **subtotal** | | **52.199** | **0.332%** |
| KV store | `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel<float>` | 96 | 3.263 | **0.021%** |
| | **attn_qkv side (norm+RoPE+KV store)** | | **55.462** | **0.352%** |
| | **DP-6's whole scope** | | **171.583** | **1.090%** |

**The SSM causal conv is 2.1x the entire `attn_qkv` side** (116.121 vs 55.462 ms), so the
conv fusion is the one to build first. The row's conditional is resolved, and it resolves
against the order the column implied.

## Why the 319.494 ms bucket was not the number

Decomposing it from the same trace:

| component | ms | share of bucket | in DP-6's scope? |
| --- | ---: | ---: | :---: |
| `grouped_rmsnorm_bf16_f32_kernel` | 230.677 | 72.2% | no |
| `grouped_rmsnorm_f32_kernel` | 5.956 | 1.9% | no |
| `qsa_split_norm_rope_rows_f32_kernel<false>` | 47.872 | 15.0% | yes |
| `qsa_gate_context_f32_kernel` | 28.511 | 8.9% | no |
| `qsa_norm_rope_rows_f32_kernel<false>` | 3.693 | 1.2% | yes |
| `qsa_pool_norm_rope_f32_kernel` | 0.634 | 0.2% | yes |
| (remainder) | 2.151 | 0.7% | — |
| **bucket total** | **319.494** | 100% | |

**74.1% of the bucket is grouped RMSNorm, which this row does not fuse.** So the row's own
warning — *"Do not build the `attn_qkv` epilogue on the strength of 319.494 ms — that number
covers more passes than this row would fuse"* — was correct, and acting on it would have
overstated the `attn_qkv` opportunity by **5.8x** (319.494 / 55.462).

## Applicability at the measured operating point

gufo's `DenseF16SsmGemm` conv fusion requires **>=1024 rows**. The capture ran
`prefill_chunk_size: 1024` and the launch census shows **264 launches at `rows=1024`** on
the wide route (`hipengine_gguf_q8_0_dense_wide256_f16in_f32_f32_out`), so the row
condition and the wide-route condition that DP-6 also requires both hold here. A chunk
size below 1024 would break the first condition, so this pricing does not transfer to a
smaller-chunk configuration without re-checking it.

## Incidental finding, recorded because it touches the GDN rows

In this capture the SSM causal conv is **116.121 ms of the 844.295 ms GDN bucket
(13.8%)** — the GDN family is not the recurrence alone. This is **not** comparable to
`GD-3`'s "the GDN recurrence *is* the GDN family: 194.487 ms of 3,914.468 ms", which is a
different capture at a different shape (p512 against this p4096/chunk-1024 run), and
neither figure is restated as the other.

## Limits

- **One case, one shape, one host.** `code-p4096` at chunk 1024 only. The passes' shares
  move with prompt length and chunk size.
- **Diagnostic capture.** The profiled arm's own wall is not used; only per-kernel times,
  which this capture attributes 100%.
- **Pricing, not a promotion.** This says what the passes cost and which is larger. It does
  not say either fusion wins: both are LDS-tile epilogues whose benefit depends on whether
  the fused pass is bandwidth- or latency-bound, and neither has been built.
- `End_Timestamp - Start_Timestamp` is the same basis the capture's own buckets use, so
  these numbers are comparable to its 15,741.963 ms union and to `DP-5`'s 131.001 ms.
