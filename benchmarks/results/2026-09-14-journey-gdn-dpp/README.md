# GDN DPP: Exact Suffix Prefill Improvement

Framework gfx1151 / Radeon8060S40CU, Qwen3.8-Flash-Next UD-Q4_K_XL,
BF16 KV, chunk1024, d128, four canonical categories at512/1024/4096.
The existing admitted tiled16 GDN suffix now uses the separately registered
DPP reduction variant. Early serial layers and strict fallback are unchanged.

## Results

One warmup per arm/case, three counterbalanced pairs, one residency:

| Shape | Parent PP | DPP PP | Change |
| --- | ---: | ---: | ---: |
| 512/128 | 295.765 | 296.136 | +0.125% |
| 1K/128 | 312.115 | 312.942 | +0.265% |
| 4K/128 | 286.962 | 287.713 | +0.262% |

Mean36 paired PP ratios1.002194, approximate95% interval1.000367..1.004021.
Some individual cases are flat within uncertainty; the complete GDN owner
screen wins at every tested row count (8-19%). No decode-speedup claim:
late decode drift in both arms remains in the recorded samples.

All72 measured samples preserve generated IDs, final logits and recurrent/KV
state fingerprints; teardown is zero. DPP engagement is15 calls at512/1024
and60 at4096, versus zero in the control. Kernel tests compare all outputs and
carried state at16/17/64/512/1024, plus a CPU-reference outer gate at16 rows.
Cached-only rocprofv3 smoke is recorded under the raw directory below.

This exact change does not resolve the independent incumbent production
numerical failure. No threshold is changed and no early-layer scope is widened.

## Reproduce

With the existing TheRock environment, queues2 and HIPENGINE_HIP_ARCH=gfx1151:

```bash
PYTHONPATH=. .venv/bin/python scripts/qwen4exp_ple_gather_ab.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipengine-journey-hipcc-version-20260913.txt \
  --method gdn_dpp --output /tmp/gdn-dpp-model.json
```

The harness explicitly bypasses the default registry selector for its parent
and candidate library arms. Source commit, model/host identity, library hash,
commands and paired samples are in `artifact.json`. Raw model, owner and trace
files are under `/tmp/hipengine-journey-execute-20260914`.

The ordinary `qwen4exp_gdn_tiled16_prefill` and
`qwen4exp_sigmoid_strict_prefill` keys remain registered.
