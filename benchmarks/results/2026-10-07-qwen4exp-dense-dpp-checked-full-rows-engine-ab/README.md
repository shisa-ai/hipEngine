# Dense Q8 checked/full-row complete-engine diagnostic

The separately registered checked-int32/full-four-row dense Q8 DPP sibling was
measured through the complete Qwen4Exp request runner on Radeon 8060S/gfx1151.
The candidate remains diagnostic-only: production dispatch, the profile manifest
and the default are unchanged.

## Measurement identity

- Host: Radeon 8060S/gfx1151, machine-id `55ea6c509d0b49eea8de7094a1023668`
- Model: Qwen3.8-Flash-Next UD-Q4_K_XL
- Tracked-clean source: `7c3319a02`
- Fixture SHA256: `42b562bd8e9644bea5b8891c61633dce7f6e75daca64cf79e9cb45c432099da1`
- Production manifest SHA256: `be32e7193f30c4c4a65f5cbaf99b62033fd6458edd295fc2d7806e734001469b`
- Strict manifest SHA256: `1fcbf992271251220cc4707eb8d24d28506cd3714550756cf2e32f387d879670`
- Artifact: [`engine-ab.json`](engine-ab.json)

## Complete-engine A/B

One model residency and one Python process covered all 12 canonical cases at
p512/p1024/p4096 across code, general English, general Japanese and mixed
Japanese/English. Each case used one warmup per arm and six measured arms in
balanced parent/candidate order, with a synchronized 128-transition decode
window. The candidate was admitted only for the six previously qualified dense
Q8 geometries and complete four-row shapes; all other calls remained on the
parent.

- 72 measured request samples.
- 3,564 candidate calls and 4,284 parent calls.
- All 12 parent/candidate trajectories were exact and deterministic.
- Zero active allocations and zero current bytes after teardown.

| Metric | Parent | Checked sibling | Ratio |
| --- | ---: | ---: | ---: |
| Prefill | 322.442 tok/s | 336.954 tok/s | **1.0450x** |
| Decode | 16.211 tok/s | 16.409 tok/s | **1.0122x** |
| Aggregate client wall | 493.845 s | 481.402 s | **0.9748x** |

The client-wall result is a diagnostic same-run A/B measurement, not a public
model-rate row. It does not authorize default promotion or imply GUFO parity.

## Cached kernel trace

A separate one-case cached `rocprofv3 --kernel-trace` run completed with an
explicit TheRock profiler path and precomputed compiler-version file. The trace
contains 396 positive-duration checked-sibling launches and 476 parent launches.
Both use workgroup 128, grid `5120x128`, and zero scratch. The CSV SHA256 is
`b75eb1eb0de2df9e9679532e076099e126165db8a108f808bc91068d6c91b6e1`.

The checked signature is
`gguf_k_prefill_out_coltile_rowbatch_pipelined_kernel<float,float,8,8,4,true,false,true,int,true>`;
the parent signature is the same template with `long,false`. The positive-duration
medians are 0.692 ms and 1.007 ms respectively. Trace timing is corroborating
kernel engagement evidence; the synchronized engine A/B owns the request-level
comparison.

## Reproduction

```bash
export ROCMENV=/home/lhl/miniforge3/envs/therock10-staging-20260828
export ROCMLIB=$ROCMENV/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH="$ROCMENV/bin:$ROCMLIB/bin:$PATH"
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 PYTHONPATH=/home/lhl/hipEngine
uv run python -u scripts/qwen4exp_dense_dpp_checked_full_rows_engine_ab.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --warmups-per-mode 1 --require-cached-build \
  --output /tmp/qwen4exp-dense-checked-engine-ab-canonical.json
```

For the trace, wrap the same one-case command with:

```bash
rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/qwen4exp-dense-checked-engine-trace -- \
  uv run python -u scripts/qwen4exp_dense_dpp_checked_full_rows_engine_ab.py \
  --model-root /home/lhl/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file /tmp/hipcc-version-qwen4exp.txt \
  --fixture benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json \
  --case-id code-p512 --warmups-per-mode 1 --require-cached-build \
  --output /tmp/qwen4exp-dense-checked-engine-trace/bench.json
```
