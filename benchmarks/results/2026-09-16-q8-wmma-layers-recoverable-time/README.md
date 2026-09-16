# Recoverable prefill time in the f16 WMMA dense Q8_0 layer scopes

**What this is.** The performance half of the decision for the route whose
numerical envelope is gated in
`../2026-09-16-q8-wmma-dense-prefill-layers-gate/`. That gate certified the
arithmetic at layers 32-47 and failed it at layers 0-47 without measuring what
either scope is worth, and the selector that drives it
(`HIPENGINE_QWEN4_EXP_Q8_WMMA_LAYERS`) is not one of the nine families in
`../2026-09-16-disabled-family-recoverable-time/`, so the route had no measured
speed at any scope.

**What this is not.** These are diagnostics, not rate rows. Each scope is
enabled by a post-binder override, so `named_profile_intact` is false for both
candidates, which is expected for a diagnostic override rather than a profile
state. No numerics are re-derived here.

## Protocol

| | |
| --- | --- |
| host | `gfx1151`, AMD RYZEN AI MAX+ 395 / Radeon 8060S, 120 GB |
| model | unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL` |
| case | `code-p4096` from the canonical fixture (4096 exact token ids) |
| harness | `scripts/qwen4exp_profile_gap.py --mode prefill --risk-diagnostics` |
| driver | `scripts/qwen4exp_disabled_family_sweep.py` |
| repetitions | 2 per arm, median reported |
| fallback | the current default with no overrides |
| toolchain | ROCm `HIP version: 7.15.26333-0000000` |

```bash
python3 scripts/qwen4exp_disabled_family_sweep.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --case-id code-p4096 --repetitions 2 \
  --only Q8_WMMA_LAYERS_32_47 --only Q8_WMMA_LAYERS_0_47 \
  --scratch <scratch> --output artifact.json
```

## Result

Fallback median prefill: **23.840 s** (the same lane's earlier fallback measured
23.648 s, a 0.8% difference).

| Scope | Median s | Saved s | Speedup | Numerical verdict |
| --- | ---: | ---: | ---: | --- |
| layers 0-47 | 16.799 | **+7.041** | **1.419x** | fails mean and p95 marginally |
| layers 32-47 | 21.405 | **+2.436** | 1.114x | **passes every calibrated gate** |

Two things follow.

**The route is the largest single recoverable-time family measured in this
model.** At its maximal scope, `+7.041 s / 1.419x` places it ahead of
`Q8_IU8_WMM` (`+6.614 s / 1.388x`), which held rank 1 in the nine-family
ranking. It was absent from that ranking only because the selector is a layer
list rather than one of the `PRODUCTION_ARITHMETIC_RECOVERY_FLAGS`.

**The admissible scope captures about a third of it.** Layers 32-47 deliver
2.436 s of the 7.041 s available, so the numerical boundary between layer 0 and
layer 32 is currently withholding **4.6 s** — the largest identified block of
unclaimed prefill time in the tree.

## Recoverable time tracks route bytes almost exactly

`../2026-09-16-q8-wmma-dense-prefill-layers-gate/route-coverage.json` measures
how much of the route's Q8_0 weight each scope owns. Comparing that with the
time each scope recovers:

| Scope | Route bytes owned | Share of route bytes | Saved s | Share of 0-47 saving |
| --- | ---: | ---: | ---: | ---: |
| 32-47 | 2.83 GiB | 36.8% | +2.436 | 34.6% |
| 0-47 | 7.68 GiB | 100% | +7.041 | 100% |

The two shares agree to within 2.2 points, so on this workload the recoverable
time is close to proportional to the Q8_0 bytes the scope owns. That makes the
untested scopes predictable rather than speculative:

| Scope | Route bytes | Predicted saving |
| --- | ---: | ---: |
| 28-47 | 51.4% | ~3.6 s |
| 24-47 | 55.2% | ~3.9 s |

Recovering scope down to layer 24 would therefore be worth roughly +1.5 s over
the currently admissible 32-47, if the numerics hold there. The prediction is an
interpolation from two points and is not a measurement; the layer bisect
measures it.

## Reading these numbers

The repair rate is 0.712% for both arms and 0.712% for the fallback, unchanged
within noise. As the nine-family ranking established, that rate is a property of
the always-on exact `Q4_IU8_EXACT` route and the input, not of the family under
test, so it says nothing about either scope here.

Neither arm reproduces the fallback's logits, which is expected: the route
changes prefill arithmetic. Both produce the same next token as the fallback on
this case (248068). That is a single-case observation and is not a correctness
result; the correctness evidence is the layer-scope gate.

## Artifacts

- `artifact.json` — the sweep output, both arms plus the fallback.
