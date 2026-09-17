# Shipped default at HEAD: canonical PP/TG, four categories, three lengths

The Qwen3.8-Flash-Next campaign's topline screen had no rate for the path it
ships. The last full-matrix row for this model
([`2026-09-13-framework-qwen4exp-strix-journey-baselines.json`](../2026-09-13-framework-qwen4exp-strix-journey-baselines.json))
was measured at `00602e556` with fifteen arithmetic-recovery flags bound **on**
for this quant, a composition whose prefill
[measured max KL 0.0546](../2026-09-14-q8-prefill-numerics/README.md) — outside
the production envelope. `49ffa3cb5` zeroed those flags, `04f1dde42` and
`15b056111` then replaced the dense prefill route at layers 16-47. This record is
the first full-matrix measurement of the resulting default.

## Result

AMD Radeon 8060S (`gfx1151`), Framework Desktop, `gguf_ud_q4_k_xl`, 1024-token
prefill chunks, BF16 KV, warm PLE cache, 128 autoregressive decode transitions,
one warmup and three repetitions per case, 36 measured samples. Production
profile, `fell_back_to_strict: false`.

| Category | 512 PP / TG | 1024 PP / TG | 4096 PP / TG |
| --- | ---: | ---: | ---: |
| code | 238.7 / 17.70 | 252.3 / 17.19 | 240.4 / 16.22 |
| general_en | 239.5 / 17.68 | 254.1 / 17.18 | 242.8 / 16.27 |
| general_ja | 241.6 / 17.67 | 254.2 / 17.20 | 243.1 / 16.29 |
| mixed_ja_en | 238.2 / 17.70 | 252.6 / 17.19 | 241.3 / 16.24 |
| **equal-weight mean** | **239.5 / 17.70** | **253.4 / 17.18** | **241.9 / 16.24** |
| artifact token-weighted rollup | 239.1 / 17.43 | 253.4 / 17.19 | 241.4 / 15.42 |

Each cell is the median of that case's three samples. Prefill and decode are
tokens per second; decode counts the 128 measured transitions. The
equal-weight mean is the aggregation `docs/QWEN4EXP-STATUS.md` §1.2 uses; the
artifact's own rollup is token-weighted and its 4096 decode figure is lower
because the last five samples of the run degrade; see [Stability](#stability).

## Comparison basis

- **The route A/B's wide arm is the same path and it agrees.**
  [`2026-09-17-q8-dense-route-ab`](../2026-09-17-q8-dense-route-ab/README.md)
  measured the promoted route at 17.217 s on `code-p4096` (three interleaved
  repetitions). This run measures 17.041 s for the same case, 1.0% faster.
- **The 2026-09-13 journey row is not the denominator.** Its 294.1 PP / 19.17 TG
  at 4096 came from the fifteen-recovery-flag composition that failed the
  numerical envelope, and it is 22% faster in prefill than the path that ships.
  Recovering that speed with correct arithmetic is an open problem, not a
  regression to explain.

## Stability

Samples 0-30 hold prefill within 0.3% and decode within 0.5% of their case
medians. Samples 31-35 degrade monotonically:

| Sample | Case | Prefill PP | Decode TG |
| ---: | --- | ---: | ---: |
| 8 | code-p4096 | 240.4 | 16.215 |
| 30 | code-p4096 | 240.3 | 16.195 |
| 31 | general_en-p4096 | 241.3 | 15.261 |
| 32 | general_ja-p4096 | 240.7 | 13.151 |
| 33 | mixed_ja_en-p4096 | 239.1 | 12.123 |
| 34 | code-p512 | 236.2 | 15.826 |
| 35 | general_en-p512 | 237.1 | 16.572 |

The GPU reported 41 °C and the host load average was 1.2 immediately after the
run, and nothing else held `/dev/kfd`, so this is not sustained thermal or
contention throttling. The cause is unidentified. It costs the p512 and p1024
rows nothing (their samples are spread across the whole run) and it lowers the
4096 decode aggregate by about 5%.

Any *paired* comparison on this host must therefore interleave its arms within
one run, which the route A/B does. This screen is a single-arm rate and cannot
be used to attribute the degradation.

## Reproduction

```bash
ENV_PREFIX=/home/lhl/miniforge3/envs/therock10-staging-20260828
PY=$ENV_PREFIX/bin/python
SITE=$ENV_PREFIX/lib/python3.12/site-packages
export PATH="$ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SITE/_rocm_sdk_core/lib:$SITE/_rocm_sdk_devel/lib:$SITE/_rocm_sdk_libraries/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIPENGINE_HIP_ARCH=gfx1151

OUT=benchmarks/results/2026-09-17-qwen4exp-shipped-default-baseline
$PY -c "import subprocess;open('$OUT/hipcc-version.txt','w').write(subprocess.run(['hipcc','--version'],capture_output=True,text=True).stdout)"
$PY scripts/qwen4exp_canonical_ar_bench.py hipengine \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --prefill-chunk-size 1024 --ple-cache-mode warm --warmups 1 --repetitions 3 \
  --compiler-version-file "$OUT/hipcc-version.txt" \
  --output "$OUT/canonical-ar.json"
```

Source: `42ce711564dcfb2c619d0f8560855f3dfeac7ef3`, `tracked_clean: true`.
The run took 13 minutes and reads 80.9 GB from storage on its first pass, after
which the model is resident in 85 GB of GPU-visible system memory and 40 GB of
page cache; later samples are not I/O-bound.

## What this does not establish

- **No numerical certificate.** The profile manifest is recorded
  (`35e57360…`), but this run gates no KL, top-1, task or heldout evidence.
- **No decode attribution.** Decode is a separate follow-up
  (`docs/QWEN4EXP-STATUS.md` §5); the composition behind these rates is not
  attributed, and the prefill promotion does not imply a decode gain.
- **One host, one quant.** AMD Radeon 8060S / `gfx1151`, `gguf_ud_q4_k_xl`.
  Rates for other hosts, backends or quantizations are not implied.
