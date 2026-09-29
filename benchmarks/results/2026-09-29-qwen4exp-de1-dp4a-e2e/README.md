# DE-1 decode dp4a linear — promotion evidence

Promotes the dp4a Q8_1-activation linear for the routed-expert gate+up decode GEMV
to the production default for `gguf_ud_q4_k_xl` on gfx1151.

## Result

One `code-p4096` decode step at live count 4097, 3 snapshot-restored repetitions,
one protocol on one host. Kernel rows are sliced to the
`qwen4exp_decode_live4097_rep{0,1,2}` ROCTX windows and divided by 3.

| kernel (ms/step) | baseline | promoted | Δ |
| --- | ---: | ---: | ---: |
| `…q4_k_selected_dual_silu_logical128_t64` (exact, the owner) | 14.155 | 1.486 | −12.669 |
| `…q4_k_selected_dual_q8_1_dp4a_silu_logical128_t64` (new) | — | 5.422 | +5.422 |
| `gguf_k_pack8_prefill_out` | 23.976 | 23.985 | +0.009 |
| `q5_1_selected_weighted_sum` | 5.511 | 5.492 | −0.019 |
| `qwen4_exp_gdn_decode_f32` | 2.243 | 2.241 | −0.002 |
| `dense_gemv_out` | 2.133 | 2.127 | −0.006 |
| `qsa_sparse_attention` | 1.346 | 1.346 | 0.000 |
| **decode kernel sum** | **54.603** | **47.359** | **−7.244 (−13.27%)** |

Owner speedup **2.049x**, matching the shape screen's 2.0x. The non-owner sum is
40.448 vs 40.451 ms — a 0.007% difference — so the change is confined to the kernel
it targets.

## Engagement, proven from the trace

This family has previously reported a pass on a route that never ran, so engagement
is checked directly rather than inferred:

- raw kernel-name grep: **0** dp4a dispatches in the baseline kernel trace, **210** in
  the candidate;
- `profile.overrides` records `{'HIPENGINE_QWEN4_EXP_Q4_DP4A64': '1'}` and post-override
  `route_env` reads `1`;
- the warmup state hash differs (`3466313123…` vs `418c337476…`), so the arithmetic
  changed.

Note `profile.bound_route_env` reads `0` in **both** arms — it is the *pre*-override
snapshot, not evidence that the override failed.

## The end-to-end wall is deliberately not claimed

| arm | min | median | mean | max |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.14511 | 0.14553 | 0.14973 | 0.15853 |
| promoted | 0.14565 | 0.14971 | 0.14871 | 0.15078 |

The median moves **+2.9% in the wrong direction**, the mean −0.7%, and the within-arm
spread (0.0134 s) exceeds the between-arm difference (0.0042 s). An expected ~7 ms/step
change on a ~145 ms step that is mostly inter-kernel gap is inside that noise. The
kernel sum is the instrument that carries this result, consistent with the ledger's
finding that the kernel sum is stable to 1.0% while the step span varies 2.8x.

## Post-promotion verification

On a clean tree at `c66db0264`, the default path with **no override** binds
`Q4_DP4A64=1`, runs the dp4a owner at **5.404 ms/step** plus a **1.480 ms/step** exact
residual (43 of 48 layers are in `Q4_DP4A64_LAYERS`), reproduces the candidate arm's
warmup state hash exactly, and totals **47.390 ms/step**.

## Gates

| gate | verdict |
| --- | --- |
| numerical | mean KL 2.44e-4 vs 1e-3, p99 3.67e-3 vs 2e-2, max 6.14e-3 vs 5e-2, top-1 0.9911 |
| determinism | 3/3 exact across all 18 prompts |
| isolation | both arms return to 0 bytes |
| task quality | cross-validated 11/11 on `code`, 0 disagreements |

The only non-passing signal anywhere was strict generated-ID equality
(`task_passed: false`), which is not the promotion bar and is why the gate reports
`requires_task_review` rather than `failed`. The four prose divergences are
synonym-level or article-drops that leave the decisions verbatim identical.

## Mechanism

`Q4_DP4A64` moved into `PRODUCTION_Q8_QSA_RESTORED_FLAGS`. The base profile already set
it to `"1"`; the blanket arithmetic-recovery sweep was the only thing suppressing it.
The gate's conservative base legitimately keeps the flag at `0`, so the recovery and
conservative lists remain equal.

## Headroom closure

The bandwidth gate localized DE-1's headroom by charging this owner's 888.83 MB/step
at the in-block comparator's 184.66 GB/s. Re-charging the promoted owner:

| | baseline | promoted |
| --- | ---: | ---: |
| owner bandwidth | 62.79 GB/s | **164.48 GB/s** |
| as % of comparator | 34.0% | **89.1%** |
| gap to comparator | 2.94x | **1.12x** |
| headroom | 9.342 ms/step | **0.591 ms/step** |

**93.7% of the identified headroom is consumed**, leaving 0.591 ms/step = 1.25% of the
new 47.390 ms/step kernel sum. The remaining work the ledger row asks for — an
*exact-activation* warp256 variant — is now chasing at most ~0.6 ms/step on this owner
rather than the ~10 ms/step that motivated it. That does not close it (the row's ask is
an exact-activation structure, which this promotion is not), but it changes its ranking.

## Scope

The warp256 dp4a variant's historical failure (max KL 0.121–0.431) is a **different
arithmetic outcome** and is not generalised away: this logical128 route passes at max KL
6.14e-3. DE-1's other follow-up unit — building an exact-activation warp256 variant —
**remains open** and is not licensed by this result. One model, one quant, one live
count, one host.

## Reproduce

```bash
# per-kernel comparison (both arms)
rocprofv3 --kernel-trace --marker-trace --hip-runtime-trace --output-format csv rocpd \
  -d <dir> -o e2ebase -- python3 scripts/qwen4exp_context_decode_profile.py \
  --model-root <root> --case-id code-p4096 --live-count 4097 --repetitions 3 \
  --prefill-chunk-size 1024 --hip-arch gfx1151 --require-cached-build \
  --compiler-version-file <f> --profile --role-markers --output trace-baseline.json
# then re-run with --override HIPENGINE_QWEN4_EXP_Q4_DP4A64=1 -o e2edp4a
```

Raw `rocprofv3` dumps live in `rocprof/` beside this file and are gitignored.
