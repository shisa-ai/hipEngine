# Dense-wide row-size scaling screen

The current F16-input dense-wide path already exceeds the historical DP-7
30-TFLOP/s target on the large attention projections in this packet. Increasing
rows from1024 to4096 produces modest kernel-efficiency improvements, not the
historical jump from20.9 to30 TFLOP/s. These are real-weight/synthetic-activation
microbenchmarks, not full-engine throughput or chunk-size qualification.

## Results

| Role | F16-input TFLOP/s,1024 rows | 2048 rows | 4096 rows |
|---|---:|---:|---:|
| attn_qkv |40.54|41.32|41.66|
| attn_q |39.26|39.81|41.45|
| attn_gate |36.20|38.47|39.93|

Measured cast-plus-matmul milliseconds normalized to1024 rows:

| Role |1024 rows|2048 rows|4096 rows|
|---|---:|---:|---:|
| attn_qkv |1.4694|1.3973|1.3598|
| attn_q |1.7627|1.6871|1.6216|
| attn_gate |0.9566|0.9308|0.8783|

These normalized values are arithmetic summaries of measured per-launch
medians, not measurements of a4096-token request. Narrow geometries have
mixed scaling; all36 cases are in [artifact.json](artifact.json). The
harness's aggregate uses1024 rows only and must not be interpreted as a
cross-chunk engine comparison.

All36 cases have zero maximum absolute output difference between the F32-input
and F16-input paths. The harness calls this `bit_identical`, but the check is
numerical subtraction, not a byte comparison (signed zeros are not distinguished).
No CPU-teacher numerical envelope or full model quality gate was run.

## Protocol and provenance

Physical host `gfx1151`, machine `55ea6c509d0b49eea8de7094a1023668`, Radeon8060S,
TheRock10/HIP7.15.26333. Source `092f88d61`, tracked tree clean. Real Q8_0 weights
from unsloth Qwen3.8-Flash-Next `UD-Q4_K_XL`, layers16/19 according to the
harness's12 admitted geometries; seeded standard-normal F32 activations.
Twenty event-timed repetitions, eight rotating weight copies, fixed arm order.
A post-run host snapshot is preserved, not contemporaneous clock telemetry.
The harness saves medians only, not raw timing samples; no stability or
counterbalanced-comparison claim is made.

Use the TheRock environment exports from
[scratch-growth](../2026-09-28-qwen4exp-scratch-growth/README.md), including
`HIPENGINE_REQUIRE_CACHED_BUILD=1`, then:

```bash
flock -n /tmp/hipengine-gfx1151-benchmark.lock "$ENV_PREFIX/bin/python" \
  scripts/qwen4exp_dense_wide_f16_activation.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --rows 1024,2048,4096 --repetitions 20 --weight-buffers 8 \
  --output /tmp/qwen4exp-dp7-row-scaling.json
```

ROCm loading succeeded in this environment. No runtime/kernel code changed.
The raw packet's `performance_claim` describes its microbenchmark, not an
engine speedup or promotion. Larger runtime chunks still require RT-1's
complete-owner, population, numerical, task and allocation gates.
