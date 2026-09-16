# halo-box PR #63 versus `69946438a`: corrected prefill comparison

**Result: on the full twelve-case canonical prefill suite, the PR #63 head
prefills 39.9% faster than `69946438a` (654.4 → 915.4 tokens/s weighted), with
identical greedy output.** Measured 2026-09-16 on Framework `gfx1151` (AMD
Radeon 8060S), machine `55ea6c509d0b49eea8de7094a1023668`, both arms in one
session.

This replaces an earlier version of this comparison that reported +13.9%. That
number was wrong, and the section "Why the earlier number was wrong" says how.
It also retracts an attention-regression claim made in the earlier version.

## What is being compared

| | baseline | candidate |
| --- | --- | --- |
| source | `halo-box/strix-llama.cpp` @ `69946438aa2c432ba365e40bec35c38f6e1de5bb` | PR #63 head `c4aa302294fcd5121af2039fd4d3dee0d472ec03` (branch `codex/halobox-modules-20260914`) |
| worktree | `/home/lhl/comparators-20260915/halobox-strix` | `/home/lhl/comparators-20260915/halobox-pr63` |
| build | `/home/lhl/comparators-20260915/build-halobox` | `/home/lhl/comparators-20260915/build-halobox-pr63` |
| dirty | false | false |

This is an **old snapshot versus PR head**, not an isolated PR effect. The PR
base `cfe6bb14e` is itself newer than `69946438a`, so the measured difference
includes everything between those two points, not only the 23 commits on the PR
branch.

The PR is **open, not merged**. The comparison target is the commit
`c4aa30229` specifically.

Both builds use the same configuration (`Release`, `AMDGPU_TARGETS=gfx1151`,
`GGML_HIP=ON`, `GGML_HIP_GRAPHS=ON`, `GGML_CUDA_FA=ON`) and the same server
arguments:

```
-m <first shard of UD-Q4_K_XL> --host 127.0.0.1 --port N --parallel 1 --no-webui
-ngl 999 -fa on -ctk bf16 -ctv bf16 -c 4352 -b 8192 -ub 2048 -t 4 --no-warmup
```

Both arms were measured **without the profiler**. The profiler is what made the
earlier comparison unreliable, so its absence is a requirement here rather than a
convenience.

## Rate

All twelve canonical cases (`code`, `general_en`, `general_ja`, `mixed_ja_en` at
512, 1024 and 4096 tokens), one unmeasured warmup request and three measured
repetitions per case, no profiler. The rate is total prompt tokens over total
prompt time, the same estimator for both arms. No repetition was discarded; both
runs report zero stalls.

| | base `69946438a` | PR #63 `c4aa30229` | ratio |
| --- | ---: | ---: | ---: |
| weighted tok/s | 654.4 | **915.4** | **1.399** |
| median of per-case medians | 621.1 | 858.8 | 1.383 |
| mean prompt_ms per repetition | 8607.0 | 6152.5 | 0.715 |

**+39.9%** on the weighted rate.

### By category

| category | base tok/s | PR63 tok/s | ratio |
| --- | ---: | ---: | ---: |
| code | 634.2 | 909.6 | 1.434 |
| general_en | 692.6 | 919.5 | 1.327 |
| general_ja | 696.0 | 922.5 | 1.325 |
| mixed_ja_en | 604.0 | 910.2 | 1.507 |

### By case

The 1024 and 4096 cases are tight, at 1.35–1.41. The 512 cases are wider
(0.996–1.875) and are the ones to distrust: the base arm's own repetitions drift
within the case at that shape (for example `code-p512` reads 355, 304, 302
tokens/s across three repetitions), so a single 512-token case carries more
run-order noise than the ratio between engines.

| case | base | PR63 | ratio |
| --- | ---: | ---: | ---: |
| code-p512 | 303.7 | 480.5 | 1.582 |
| code-p1024 | 613.7 | 856.5 | 1.395 |
| code-p4096 | 742.5 | 1043.4 | 1.405 |
| general_en-p512 | 475.2 | 473.4 | 0.996 |
| general_en-p1024 | 628.4 | 869.8 | 1.384 |
| general_en-p4096 | 756.7 | 1060.2 | 1.401 |
| general_ja-p512 | 481.7 | 498.7 | 1.035 |
| general_ja-p1024 | 637.0 | 860.6 | 1.351 |
| general_ja-p4096 | 759.4 | 1067.7 | 1.406 |
| mixed_ja_en-p512 | 251.7 | 472.1 | 1.875 |
| mixed_ja_en-p1024 | 610.0 | 856.9 | 1.405 |
| mixed_ja_en-p4096 | 677.2 | 1054.4 | 1.557 |

## Why the earlier number was wrong

The earlier version reported +13.9% from a single case and a single measured
repetition per arm. Two separate problems made that number too small.

**The candidate sample was not representative.** The earlier PR63 measurement
read 4873.5 ms on `code-p4096`. The same case now reads 3926 ms (median of three
repetitions) with no profiler. The earlier sample sat about 24% high.

**The baseline row it was compared against was contaminated.** An earlier
full-suite baseline run put `mixed_ja_en-p1024` at 351.8 and `mixed_ja_en-p4096`
at 363.6 tokens/s, against 610.0 and 677.2 in the clean run — 73% and 86% low.
Those three cases are measured last, and they drag that run's weighted rate from
654.4 down to 511.5, a 21.8% depression. Whatever slowed that run is not a
property of the engine: it does not reproduce, and the same build is flat across
categories when it is not happening. The earlier comparison used that row.

Profiler overhead itself is small and is not the main cause. Across the ten cases
where the profiled and unprofiled runs agree, the difference is 0.2–0.5%. It is
still a reason not to read a rate off a profiled run, because the overhead is not
guaranteed equal between two builds, but it is not what produced the discrepancy.

## Correctness

Greedy generation (temperature 0, top_k 1) from the fixture's exact token ids, 48
tokens, compared between the two builds by
`scripts/llamacpp_pair_token_compare.py`. The checker now fails closed: it exits
non-zero on divergence, rejects an arm that returns no token array, and requires
each arm to emit exactly the requested number of tokens, so it can no longer pass
by comparing two empty sequences.

| arm | tokens emitted | identical positions |
| --- | ---: | ---: |
| `69946438a` | 48 | 48/48 |
| `c4aa30229` | 48 | 48/48 |

**Sequences are identical over 48 tokens.** This is one prompt and one
continuation. It is evidence that the arithmetic difference did not change the
output here; it is not a general equivalence claim. The PR does change numerics:
its own commits record WikiText2 perplexity moving 2.0285 → 2.0259 → 2.0254.

## Where the gain came from

Kernel time per family, both arms, **delimited to single prefills**. The harness
was asked for a 2000 ms idle gap between requests so each prefill becomes its own
profiler burst, and each burst was matched to its request by duration. The match
is the check: kernel sum over measured prompt time lands at 0.93–0.97 for all
four matched prefills, against roughly 1.7 when a window holds two prefills.

| family | base ms | PR63 ms | delta | delta % | base share | PR63 share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dense_projection | 1536.7 | 1374.3 | −162.4 | −10.6% | 28.4% | 36.5% |
| expert_down | 1149.8 | 412.8 | −737.0 | **−64.1%** | 21.2% | 11.0% |
| expert_gate_up | 819.4 | 761.8 | −57.6 | −7.0% | 15.1% | 20.3% |
| elementwise_norm | 531.4 | 331.7 | −199.7 | −37.6% | 9.8% | 8.8% |
| hyper_connection | 428.9 | 335.7 | −93.2 | −21.7% | 7.9% | 8.9% |
| gdn | 287.6 | 263.8 | −23.8 | −8.3% | 5.3% | 7.0% |
| quantize_pack | 284.8 | 0.0 | −284.8 | **−100.0%** | 5.3% | 0.0% |
| moe_reduce | 204.0 | 103.9 | −100.1 | −49.1% | 3.8% | 2.8% |
| qsa_attention | 156.6 | 155.7 | −0.9 | −0.6% | 2.9% | 4.1% |
| indexer | 5.6 | 15.0 | +9.4 | +167.9% | 0.1% | 0.4% |
| other | 6.2 | 5.7 | −0.5 | −8.1% | 0.1% | 0.2% |
| **total** | **5411.0** | **3760.4** | **−1650.6** | **−30.5%** | | |

Measured prompt time over the same pair falls 5569.2 → 3989.3 ms (−28.4%). The
kernel sum accounts for essentially all of it; the remainder is host-side launch
time.

Both arms are the same case, `code-p4096`, measured in the same session under the
same protocol. The two traces are
`attrib-base-69946438a-code4096.json` and
`attrib-pr63-c4aa30229-code4096.json`; the table is generated from them by
`scripts/qwen4exp_prefill_component_gap.py`, so it cannot drift from the traces.

The two largest reductions are exactly what the commits describe:

- **`expert_down` −64.1%** and **`quantize_pack` −100%** are one change.
  `08de004` ports the MMB kernels: weights are dequantized to BF16 in LDS and the
  product accumulates on the 16x16x16 BF16 WMMA path, so the Q5_1 MMQ kernel and
  its per-matmul `quantize_mmq_q8_1` activation quantization both disappear. The
  PR63 trace contains `mmb_dense_kernel`, `mmb_routed_kernel`,
  `mmb_routed_glu_kernel` and `mmb_f32split_kernel` verbatim.
- **`moe_reduce` −49.1%** and **`hyper_connection` −21.7%** are `60c26d0`, which
  carries the hyper-connection streams as BF16 and gives the MoE weighted
  reduction BF16 and float4 variants.
- **`elementwise_norm` −37.6%** is largely `0106857`, one wave per row for narrow
  RMS normalization.

## Retraction: there is no attention regression

The earlier version of this comparison reported that `qsa_attention` had doubled,
calling it "one real regression" from the same kernel at the same dispatch count.
**That was an artifact of the undelimited comparison and is withdrawn.**

Delimited to single prefills, the attention family is unchanged:

| | base | PR63 | ratio |
| --- | ---: | ---: | ---: |
| `qsa_attention` family | 156.6 ms | 155.7 ms | 0.994 |

The dominant kernel agrees on every property that could be checked:

| | base | PR63 |
| --- | ---: | ---: |
| kernel | `flash_attn_ext_f16<256, 256, 16, 4, false, false, false>` | same |
| duration over two prefills | 273.1 ms | 270.2 ms |
| dispatches | 48 | 48 |
| grid | `(24576, 8, 1)` | `(24576, 8, 1)` |
| workgroup | `(32, 8, 1)` | `(32, 8, 1)` |

The earlier figure came from a window that did not hold the same amount of work
in each arm. This is why the delimited measurement exists, and why a per-component
number from an undelimited trace should not be reported at all.

## The `indexer` change is a reclassification plus a small real cost

`indexer` reads +167%, but the work moved between families. `ec05a66` replaces a
hipCUB sort with named `top_k_radix_*` kernels, so the same selection now appears
under `indexer` instead of inside `elementwise_norm`:

| | base | PR63 |
| --- | ---: | ---: |
| rocprim trampoline in `elementwise_norm` | 17.48 ms | 0.51 ms |
| `top_k_radix_*` | 0.0 ms | 18.06 ms |
| `idx_relu_sum_f32` | 0.0 ms | 1.54 ms |
| **combined, two prefills** | **17.48 ms** | **19.61 ms** |

Per prefill that is 8.74 → 9.80 ms, **+12.1%** on the component and about
+1.1 ms on a 4000 ms prefill. The absolute cost is negligible, but the earlier
"+6% is inside the noise floor" phrasing was wrong twice over: the number was
not 6%, and a consistent 1.1 ms cost is a measurement rather than noise. The two
runs it was originally derived from are the contaminated ones described above.

## What would make this stronger

- The 512-token cases carry run-order drift in the base arm. Repeating the suite
  with the case order shuffled would separate that from any engine effect.
- One prompt and one 48-token continuation is a weak correctness check. The
  comparison would be stronger with several prompts per category.
- The component table is one case. The family deltas are consistent with the
  per-case ratios at 1024 and 4096, but they are measured at 4096 only.
