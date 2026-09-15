# Generic Q8 IU8 Numerical Gate

Source `458d83f4e`, Framework machine `55ea6c509d0b49eea8de7094a1023668`,
Radeon 8060S/gfx1151, Flash-Next UD-Q4_K_XL revision
`8bdc666649440e9bdc97e16f3f75782c98478ff5`, BF16 KV, chunk1024.
Only the generic Q8 IU8 switch is enabled over corrected production;
dedicated GR-up/down and dense MMQ switches are off.

| Metric | Measured | Required |
| --- | ---: | ---: |
| Mean KL | 0.0013944530 | <=0.001 |
| p95 KL | 0.0064351029 | <=0.005 |
| p99 KL | 0.0173673607 | <=0.02 |
| Maximum KL | 0.0389201680 | <=0.05 |
| Top-1 | 767/780 (98.333%) | >=99% |

**Not promoted.** All12 canonical512/1K/4K cases,64 shared-teacher decode
transitions and three candidate repeats completed. Determinism, sampled
state layout/metadata/finiteness and teardown pass; mean/p95/top1 fail.
This is not an exact-ID rejection.

The registered candidate executed28,656 times across18 `(rows,K,N)` shapes.
Of these,6912 are the previously identified GR-down `(K,N)=(10240,320)`
route:1152 at512 rows and5760 at1024 rows. The other21,744 calls cover
eight other projection geometries. Full counts are in `artifact.json`.
The generic selector therefore has real non-GR coverage, but this result
does not independently attribute the failure to that coverage. A counted
role-exclusion arm is still needed before rejecting non-GR restoration.

No production default changed. No task or timing run follows a binding
numerical failure; no performance claim is made. Sampled state checks are
not a complete dynamic-serving isolation certificate.

The artifact contains the exact capture argv, host/model identity, manifests,
flags, numerical scope results and dispatch counts. Environment:

```bash
env LD_LIBRARY_PATH=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib:/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib64:/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel/lib/llvm/lib \
 PATH=/home/lhl/miniforge3/envs/therock/bin:/usr/bin:/bin \
 HIPENGINE_HIP_ARCH=gfx1151 HIPENGINE_REQUIRE_CACHED_BUILD=1 \
 HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-journey-hipcc-version-20260913.txt \
 GPU_MAX_HW_QUEUES=2 PYTHONPATH=. \
 .venv/bin/python scripts/qwen4exp_q8_repair_depth_gate.py \
 --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
 --compiler-version-file /tmp/hipengine-journey-hipcc-version-20260913.txt \
 --candidate production_dense_q8_restore \
 --output /tmp/hipengine-journey-execute-20260914/resume-dense-q8-depth.json

.venv/bin/python benchmarks/results/2026-09-15-journey-dense-q8/assemble.py \
 --capture /tmp/hipengine-journey-execute-20260914/resume-dense-q8-depth.json
```
