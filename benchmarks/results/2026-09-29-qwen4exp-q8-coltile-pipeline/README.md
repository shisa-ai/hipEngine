# Q8 coltile software pipeline promotion

Promotes `coltile8_rowbatch4_wave_scale_pipelined_f32_f32_out` to the exact Q8
prefill leaf for the `linear/prefill_exact_q8_f32_coltile` scope.

The pipelined kernel is a scheduling-only sibling of
`coltile8_rowbatch4_wave_scale_f32_f32_out`: the next K iteration's Q8_0
weight-block and activation loads are issued before the current iteration's
FMAs retire. Dequantization, the per-output FMA order, the wave shuffle tree and
the four-wave publication are unchanged, so the candidate is bit-identical to
its parent.

## Result

Same-host A/B on real Q8_0 `attn_qkv.weight` tensors (2560 -> 10240, 1024 rows,
five layers, identical synthetic operands, 20 repetitions, compilation outside
the timed region):

| layer | wave-scale parent | pipelined | delta |
| ---: | ---: | ---: | ---: |
| 0 | 17.3214 ms | 16.1587 ms | −6.71% |
| 9 | 17.3089 ms | 16.1447 ms | −6.73% |
| 22 | 17.3108 ms | 16.1359 ms | −6.79% |
| 34 | 17.3340 ms | 16.1411 ms | −6.88% |
| 46 | 17.3341 ms | 16.1043 ms | −7.09% |
| **mean** | **17.3218 ms** | **16.1369 ms** | **−6.84%** |

`max_abs_diff_vs_production` is `0.0` on all five layers, so the win costs
nothing in the exact sense. This is a per-kernel microbenchmark, not a
full-engine rate.

## Correctness

`tests/test_gpu_qwen4exp_q8_coltile_pipeline.py`, 13 tests: uint32-view bit
equality against the parent at eight shapes, including the production
`1023 x 2560 x 10240` case, a partial row tile, a single K iteration, a K that
is not a multiple of the 128-thread stride, and a single column tile. Each case
runs twice to confirm repeatability. The narrow shapes also carry the
CPU-reference gate (max KL 0.05, top-1 agreement 0.9) and land exact with top-1
agreement 1.0. The two sibling variants are checked the same way against their
own parents.

## Routing

`_raw_k_coltile_pipelined_dispatch` runs last in the launch chain, so every
route keeps the parent it already matches on and only the leaf kernel is
replaced. It is `is_registered`-guarded, so the wave-scale parent and the
unscaled `coltile8_rowbatch4_f32_f32_out` exact parent remain the rollback
chain. 61 focused unit tests pass across the profile, wave-route, dense-wide,
iu8-scope, role-map and journey-localize suites.

## End-to-end gate

The route's default-state gate passes on `code-p4096` with `--full-kv`: three
arms (`enabled=0`, `enabled=1`, `enabled=0`) produce identical
`state_sha256`, `layout_sha256`, `prefill_logits_sha256`,
`step_logits_sha256` and `full_kv_sha256`. The enabled arm invokes the promoted
kernel 1072 times, all in prefill and none in decode, which shows the promotion
reaches the engine rather than only the registry. No live allocations remain
after close. The gate confirms exactness and routing; it is not a rate
measurement.

## Host

`gfx1151` / AMD Radeon 8060S, machine-id `55ea6c509d0b49eea8de7094a1023668`,
ROCm 10.0.0, HIP 7.15.26333, `power_dpm_force_performance_level=high`. The
comparison is same-host.

## Files

- `artifact.json` — promotion record, gate summary, per-layer result.
- `packet.json` — raw A/B harness output (`raw_packet_sha256` in the artifact).
- `state-gate.json` — raw default-state gate output.
