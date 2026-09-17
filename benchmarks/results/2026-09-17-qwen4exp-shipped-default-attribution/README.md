# Shipped default: where one 4096-token prefill goes

A role-marked `rocprofv3` capture of the shipped default (no profile named, no
environment overrides) on `code-p4096`, in the taxonomy the cross-engine
comparison uses. It replaces the pre-promotion attribution
([`2026-09-17-qwen4exp-per-role-cost`](../2026-09-17-qwen4exp-per-role-cost/README.md))
as the current ranking of what is left.

## Integrity

| Measure | Value |
| --- | ---: |
| Window | 17196.5 ms |
| Attributed | 16733.4 ms |
| Unattributed | 0.0 ms (0.00%) |
| Kernels | 9652 |
| Census launches | 1595, of which 1056 are the wide route |
| Unmapped over the 1 ms floor | none |

The capture predates no default-path change: it is the same commit as the
[canonical screen](../2026-09-17-qwen4exp-shipped-default-baseline/README.md)
(`42ce71156`), whose `code-p4096` prefill is 17041 ms against this window's
17196.5 ms, a 0.9% profiler overhead. The family mapper covers every dense Q8_0
prefill symbol (`scripts/qwen4exp_comparator_role_map.py`, pinned by
`tests/test_unit_qwen4exp_role_map_dense_routes.py`); before that fix the
promoted route's kernel symbol classified as `other`, which the comparison's
`--strict --unmapped-floor-ms 1` gate turns into a failed run.

## What the promotion did

| Family | Before promotion | Now | Change |
| --- | ---: | ---: | ---: |
| `dense_projection` | 10562.2 | **5234.8** | −5327.4 (−50.4%) |
| `expert_gate_up` | 3414.2 | 3402.4 | −11.8 |
| `hyper_connection` | 2523.8 | 2525.4 | +1.6 |
| `expert_down` | 2389.0 | 2382.3 | −6.7 |
| `qsa_attention` | 1402.7 | 1402.5 | −0.2 |
| `gdn` | 821.2 | 821.6 | +0.4 |
| `moe_reduce` | 762.5 | 758.4 | −4.1 |
| `elementwise_norm` | 180.2 | 178.5 | −1.7 |
| **total** | **22083.6** | **16733.4** | **−5350.2** |

Every family except `dense_projection` is within 0.5% of the pre-promotion
capture, so the promotion moved the dense route and nothing else.

## Where the time goes now

| Role | ms | Share |
| --- | ---: | ---: |
| `moe:layers.*.expert_gate` | 6596.2 | 39.4% |
| `gr_read:layers.*.hc_attn_down` | 1272.2 | 7.6% |
| `gr_read:layers.*.hc_ffn_down` | 1270.7 | 7.6% |
| `qsa_prefill:layers.*.attn_q` | 1422.3 | 8.5% |
| `linear:layers.*.attn_qkv` | 1076.6 | 6.4% |
| `linear:layers.*.ssm_out` | 882.8 | 5.3% |
| `gdn:layers.*.attn_qkv` | 821.6 | 4.9% |
| `linear:layers.*.attn_gate` | 674.2 | 4.0% |
| `linear:layers.*.attn_q` | 466.3 | 2.8% |
| `linear:layers.*.hc_attn_down` | 439.6 | 2.6% |

The dense roles are the `linear:` prefix. They carry launches at layers 0-15 on
the exact coltile chain as well as the wide route at 16-47, which is why
`linear:layers.*.attn_qkv` (1076.6 ms) is much larger than the wide route's own
`attn_qkv` share (240.0 ms).

## The dense family is now two different things

The dense roles (`linear:`) split by layer group:

| Group | Dense total | Per layer | Wide-eligible roles |
| --- | ---: | ---: | ---: |
| Layers 0-15, exact coltile chain | 3511.6 | 219.5 | 3262.5 |
| Layers 16-47, wide route | 1720.1 | 53.8 | 1404.2 |

Wide-eligible means the Q8_0 GEMM roles the promoted route already owns:
`attn_qkv`, `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_gate`, `ssm_out`,
`shared_gate`, `shared_up`, `shared_down`, `hc_attn_down`, `hc_ffn_down`. The
rest of the dense family is not that shape class: `index_q`/`index_k` (the QSA
indexer projections), `ssm_alpha`/`ssm_beta`, `ple_key`/`ple_value`,
`hc_*_inject` and `shared_expert_gate`, which together are 315.9 ms at layers
16-47 and 249.1 ms at layers 0-15.

Role totals in the wide-eligible subset, layers 0-15 against layers 16-47:

| Role | Layers 0-15 | Layers 16-47 |
| --- | ---: | ---: |
| `attn_qkv` | 836.6 | 240.0 |
| `ssm_out` | 653.6 | 229.2 |
| `attn_gate` | 525.9 | 148.3 |
| `attn_q` | 378.9 | 87.4 |
| `attn_output` | 218.0 | 79.0 |
| `hc_ffn_down` | 183.5 | 251.1 |
| `hc_attn_down` | 183.4 | 256.2 |
| `shared_down` | 99.4 | 19.0 |
| `shared_gate` | 90.3 | 44.1 |
| `shared_up` | 61.9 | 31.9 |
| `attn_k` | 18.3 | 10.2 |
| `attn_v` | 12.7 | 8.0 |
| **total** | **3262.5** | **1404.2** |

`attn_q`, `attn_k`, `attn_v` and `attn_output` exist only at the
every-fourth-layer full-attention positions, so layers 0-15 contribute four of
each and layers 16-47 contribute eight; the `hc_*` and `shared_*` roles exist at
every layer, so layers 0-15 contribute half of the 16-47 total. Per launch, the
same shape costs 4-5x more on the coltile chain (for `attn_qkv`, 13.1 ms against
2.5 ms; for `attn_q`, 23.7 ms against 2.7 ms).

Applying the promoted route's measured per-launch times
([`2026-09-17-q8-dense-f16-activation`](../2026-09-17-q8-dense-f16-activation/README.md),
whose per-shape times match this capture to 1.4%) to the layers 0-15 role set
gives about 798 ms instead of 3262.5 ms, a **2.5 s** opportunity on a 17.2 s
prefill. That is an estimate from measured per-launch times, not a measurement,
and it is larger than the 0.5 s the pre-promotion byte-share interpolation
predicted for the same extension (`docs/QWEN4EXP-STATUS.md` §4).

It is also not free upside. The `layers 0-47` arm failed the numerical gate at
mean KL 1.099e-3 against the 1e-3 limit, on 387 code rows, with the earlier WMMA
route. Extending the wide route below layer 16 needs its own gate, which has not
been run.

## Gap against the comparator

Same case, same host, different arithmetic: the comparator's dense path is BF16
against this engine's F16. Ratios are opportunity indicators, not achievable
speedups, and its PR #63 arm is a separate build.

| Family | This engine | PR #63 | Ratio | Gap | Gap share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_projection` | 5234.8 | 1374.3 | 3.81x | 3860.5 | 29.8% |
| `expert_gate_up` | 3402.4 | 761.8 | 4.47x | 2640.6 | 20.4% |
| `hyper_connection` | 2525.4 | 335.7 | 7.52x | 2189.7 | 16.9% |
| `expert_down` | 2382.3 | 412.8 | 5.77x | 1969.5 | 15.2% |
| `qsa_attention` | 1402.5 | 155.7 | 9.01x | 1246.8 | 9.6% |
| `moe_reduce` | 758.4 | 103.9 | 7.30x | 654.5 | 5.0% |
| `gdn` | 821.6 | 263.8 | 3.11x | 557.8 | 4.3% |
| `elementwise_norm` | 178.5 | 331.7 | 0.54x | −153.2 | −1.2% |
| `indexer`, `ple`, `other` | 27.7 | 20.7 | — | 7.0 | 0.1% |
| **total** | **16733.4** | **3760.4** | **4.45x** | **12973.0** | 100% |

The dense share of the gap fell from 50.1% to 29.8%. The MoE pair
(`expert_gate_up` + `expert_down`, 5.78 s of engine time and 4.61 s of gap) is
now the largest absolute owner, and `hyper_connection` has the worst ratio after
`qsa_attention`.

## Reproduction

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
PY=$ENV_PREFIX/bin/python
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_KERNEL_CENSUS=1   # without this the census records nothing

OUT=benchmarks/results/2026-09-17-qwen4exp-shipped-default-attribution
rocprofv3 --kernel-trace --marker-trace --hip-trace --output-format csv \
  -d "$OUT/role-trace" -- \
  $PY scripts/qwen4exp_profile_gap.py \
    --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --mode prefill --case-id code-p4096 --repetitions 1 \
    --profile --role-markers --require-cached-build \
    --launch-census "$OUT/census.json" \
    --compiler-version-file "$OUT/hipcc-version.txt" \
    --prefill-chunk-size 1024 --output "$OUT/child.json"

TRACE="$OUT/role-trace/gfx1151"
$PY scripts/qwen4exp_role_analyze.py --trace-dir "$TRACE" \
  --measure-prefix qwen4exp_prefill_p4096_ --output "$OUT/role-analysis.json"
$PY scripts/qwen4exp_role_gap_table.py --role-analysis "$OUT/role-analysis.json" \
  --census "$OUT/census.json" --output "$OUT/artifact.json"
$PY scripts/qwen4exp_operation_cost.py --trace-kernels "$OUT/role-analysis.json" \
  --census "$OUT/census.json" --model-shapes "$OUT/model-shapes.json" \
  --output "$OUT/operation-cost.json"
$PY scripts/qwen4exp_shared_family_comparison.py \
  --hipengine-role-analysis "$OUT/role-analysis.json" \
  --hipengine-label hipengine-42ce71156-wide \
  --comparator halobox-base=benchmarks/results/2026-09-16-flashnext-delimited-components/attrib-base-69946438a-code4096.json \
  --comparator halobox-pr63=benchmarks/results/2026-09-16-flashnext-delimited-components/attrib-pr63-c4aa30229-code4096.json \
  --case-id code-p4096 --strict \
  --output "$OUT/shared-family-comparison.json" \
  --markdown "$OUT/shared-family-comparison.md"
```

`--require-cached-build` fails closed on any kernel not built under the current
environment's cache key, so no `hipcc` runs inside the profiler. It is stricter
than the 2026-09-17 per-role-cost command, which omitted it; every kernel it
needs was already cached by the canonical screen above.

## What this does not establish

- **One case, one repetition.** `code-p4096` only. The per-layer ranking is a
  single capture, not a spread.
- **Kernel time, not wall.** The window is a profiled prefill; the 0.9% gap to
  the unprofiled 17041 ms is profiler overhead.
- **Not a target list.** The comparator's arithmetic differs (BF16 dense path),
  so a 3.81x ratio is not 3.81x of recoverable time, and the two engines'
  taxonomies assign conversion and fused work to different families.
- **Repair is folded in.** 1420.9 ms of `iu8` exact-repair passes are counted in
  the matmul family they correct, as the taxonomy requires.
