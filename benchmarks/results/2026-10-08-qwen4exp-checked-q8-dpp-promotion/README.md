# Dense Q8 checked full-rows DPP promoted to production default

This packet records the promotion of the checked-int32/full-four-row dense Q8
DPP sibling
(`coltile8_rowbatch4_wave_scale_pipelined_dpp_checked_full_rows_f32_f32_out`)
to the production default leaf for its qualified shape window, on Radeon
8060S/gfx1151 host `55ea6c509d0b49eea8de7094a1023668`, Qwen3.8-Flash-Next
UD-Q4_K_XL, production profile, BF16 KV, prefill chunk 4096.

The sibling is a **T0 exact-class** variant: the same ordered FMA/DPP
arithmetic as the plain DPP leaf with complete-int32 offsets and no per-row
validity control, bit-identical outputs. The promotion changes scheduling
only. The plain DPP parent stays registered as the strict fallback and remains
the owner of decode rows, prefill tails and every unqualified Q8_0 dense
shape. Selection: `_raw_k_coltile_pipelined_dispatch` prefers the sibling when
`q8_0_coltile_dpp_checked_full_rows_qualified(rows, in, out)` holds (rows
512-4096 and the six declared geometries: attn_qkv, attn_gate, ssm_out,
shared-expert gate/up, shared-expert down, HC down). Opt-outs:
`HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS=0` (bisection) and
`HIPENGINE_QWEN4_EXP_Q8_DPP_REDUCTION=0` (whole DPP family).

## Cooled production-path measurement (idle-settled, chunk 4096)

Compact capture (production greedy), one warmup + three repetitions per case,
12 canonical cases, 128 decode transitions.

| Metric | Baseline (pre-promotion) | Head (promotion) | Delta |
| --- | ---: | ---: | ---: |
| Total client wall | 457.017 s | **447.633 s** | **−2.05%** |
| Prefill p512 | 294.68 tok/s | **308.43 tok/s** | +4.7% |
| Prefill p1024 | 320.15 tok/s | **335.63 tok/s** | +4.8% |
| Prefill p4096 | 327.42 tok/s | **340.24 tok/s** | +3.9% |
| Decode p512 | 20.40 tok/s | 20.37 tok/s | flat |
| Decode p1024 | 19.85 tok/s | 19.73 tok/s | flat |
| Decode p4096 | 16.75 tok/s | 16.94 tok/s | flat |

All 12 cases deterministic; outputs exact — a fresh off/on pair produced
identical output token-ID digests on 12/12 cases.

## Controlling same-residency evidence

The balanced one-residency A/B (72 arms) remains the controlling instrument:
prefill 322.442 -> 336.954 tok/s (**1.0450x**), decode 16.211 -> 16.409
tok/s (**1.0122x**), client wall 493.845 -> 481.402 s (**0.9748x**), all 12
trajectories exact, 3,564 candidate calls, zero teardown allocations, and a
cached trace with the expected checked/parent kernel signatures.
[Engine A/B packet](../2026-10-07-qwen4exp-dense-dpp-checked-full-rows-engine-ab/README.md).

## Measurement-protocol finding

Four consecutive full-load canonical runs degraded monotonically regardless
of arm: identical pre-promotion code measured 457.017 s and then 470.627 s on
back-to-back runs, and a promotion run chained immediately after another full
run measured 500.302 s. The host drifts ~1.5-3% per sustained-load run, worst
on bandwidth-bound p4096 decode; the GPU returned to 41 °C at full clocks
minutes after. **Cross-process single-arm comparisons cannot resolve sub-3%
effects on this host.** Use an idle-settled run (18 min here), a same-session
reference arm, or the same-residency balanced A/B harness. The cooled
re-measure landed on the A/B's prediction (457.017 × 0.9748 ≈ 445.5 vs
measured 447.6).

## Reproduction

```bash
export ROCMENV=/home/lhl/miniforge3/envs/therock10-staging-20260828
export ROCMLIB=$ROCMENV/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH="$ROCMENV/bin:$ROCMLIB/bin:$PATH"
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib"
export HIPENGINE_HIP_ARCH=gfx1151 PYTHONPATH=/home/lhl/hipEngine
uv run python -u scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --prefill-chunk-size 4096 --capture-mode compact \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --repetitions 3 --warmups 1 --output /tmp/metric.json
# bisection arm: prepend HIPENGINE_QWEN4_EXP_Q8_DPP_CHECKED_FULL_ROWS=0
```

Baseline: commit `e1d97a921`; head: this commit (promotion diff). Full
machine-readable record: [promotion.json](promotion.json).
