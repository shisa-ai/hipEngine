# DP-2 bounded two-consumer F16 ownership

The complete native owner operation is qualified only at rows=4096. The route
remains opt-in and default-off pending a separate full-engine A/B with the
narrowed policy. Rows=1024 is rejected on native cost and remains on the parent.

## Complete native owner screen

A tracked-clean screen at commit `5ed9c3e37` uses the actual runner pair helper
on eight canonical cases: four categories at p1024/p4096, layers13/24/46 and
both ordered pairs. It records one warmup and 12 alternating measured pairs per
arm for 48 owner captures. The timing boundary includes scope admission,
normal dispatch, cast/GEMMs, scope exit, HIP events and final device
synchronization; diagnostic allocation, byte downloads and surrounding model
work are outside the boundary and validated separately.

| Rows | Pair | Parent ms | Candidate ms | Change |
| ---: | --- | ---: | ---: | ---: |
| 1024 | `attn_qkv→attn_gate` | 31.232 | 31.623 | +1.252% |
| 1024 | `shared_gate→shared_up` | 7.712 | 8.585 | +11.314% |
| 1024 | all measured pairs | 38.945 | 40.208 | +3.245% |
| 4096 | `attn_qkv→attn_gate` | 109.378 | 107.259 | −1.937% |
| 4096 | `shared_gate→shared_up` | 20.254 | 17.206 | −15.051% |
| 4096 | all measured pairs | 129.633 | 124.465 | −3.986% |

Every p1024 order stratum loses and every p4096 order stratum wins. All 48
captures preserve exact native outputs and F16 staging bytes, immutable inputs
and weights, parent two-cast/candidate one-cast engagement, finite timings and
zero teardown allocations. This is an owner-level measurement, not a
whole-model throughput claim.

## Existing engine A/B

The earlier tracked-clean one-process/one-residency A/B covers 12 canonical and
16 category-heldout cases, with 168 measured arm runs. Its policy included both
p1024 and p4096. Qualified prefill duration is 0.999575x parent (−0.043%),
decode is 1.000842x (+0.084%) and complete client wall is 1.000254x (+0.025%).
Request-wall changes are +0.067% for parent-first cases and −0.016% for
candidate-first cases; these are different case groups and do not establish a
causal regression. Those samples remain valid evidence for the former scope,
but they do not qualify the narrowed rows4096-only policy as a model-wall
change.

## Decision and reproduction

The policy admits only gfx1151 production Q8_0 dense-wide layers13–47, rows4096
and immediate `attn_qkv→attn_gate` or `shared_gate→shared_up` pairs. P512,
p1024, unmeasured rows, strict, Q4_K_M, other backends and other roles retain
the parent. `HIPENGINE_QWEN4_EXP_DENSE_F16_PAIR` remains default-off. A
separate full-engine A/B is required before production binding or a model-wall
claim.

The compact native-owner artifact is
[`native-owner.json`](native-owner.json). The full 363,656-byte screen remains
outside Git at `/tmp/qwen4exp-dp2-native-owner-eight-case.json`, SHA-256
`20afe4030576b5b9dc538838c0a90da9b2ffd3e096d09cb3150acaf490853a8e`.
The earlier native qualification and engine samples remain in
[`native-qualification.json`](native-qualification.json) and
[`engine-ab.json`](engine-ab.json).
