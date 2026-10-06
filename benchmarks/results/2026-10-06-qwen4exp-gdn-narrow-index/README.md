# Qwen4Exp checked-index GDN operation screen

On 2026-10-06, Radeon 8060S `gfx1151` host
`55ea6c509d0b49eea8de7094a1023668` measured a separately registered D128
GDN wave-norm sibling with checked int32 element indexing. It preserves the
FP32 recurrence, state layout and norm tree. The parent uses int64 indexing.
This screen uses Qwen3.8-Flash-Next UD-Q4_K_XL geometry, synthetic F32
inputs and nonzero FP32 recurrent state; no model weights are loaded.

## Complete-operation result

HIP events cover the complete recurrence, normalization and sigmoid output
gate in one launch. State reset, synchronization and result downloads are
outside the timer for both arms. Three warmup pairs precede 21 alternating
parent/candidate pairs. All measured output/state pairs are bit-identical.

| Tokens | Parent median ms | Candidate median ms | Speedup | Speedup by order |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 0.102472 | 0.100227 | 1.0224x | 1.0396x / 1.0206x |
| 64 | 0.333825 | 0.324287 | 1.0294x | 1.0347x / 1.0226x |
| 512 | 2.709651 | 2.594705 | 1.0443x | 1.0500x / 1.0321x |
| 1024 | 5.371735 | 5.155940 | 1.0419x | 1.0460x / 1.0361x |
| 4096 | 21.318647 | 20.451752 | 1.0424x | 1.0432x / 1.0408x |

The one-token sample is order-sensitive (1.1408x / 0.9793x) and supports
no speed claim. It is retained in the JSON. Production does not select this
prefill route for one row. No model throughput or decode claim follows from
this operation screen.

## Resources and correctness

The actual cached code object reports:

| Resource | Wide-index parent | Narrow-index sibling |
| --- | ---: | ---: |
| Static VGPR | 256 | 256 |
| SGPR | 51 | 44 |
| Private bytes/lane | 36 | 28 |
| VGPR spills | 8 | 6 |
| Static scratch instructions | 11 | 6 |
| Dynamic LDS bytes | 2560 | 2560 |

A cached-only `rocprofv3 --kernel-trace` child records 14 launches per arm at
512/4096 rows, with scratch sizes 36/28 bytes and plausible durations
(about 2.6/2.6 ms and 21.3/20.5 ms). Timing claims use the unprofiled
measurements, not the profiler samples.

The focused GPU fixtures cover tokens 1/17/64/512/1024/4096, exact repeated
output/state and split-at-7 replay, zero Q/K heads, nonzero initial state,
read-only input guards, output/state guards, cross-head state isolation and
raw-ABI overflow rejection. CPU-reference output/state tolerance and the
KL≤0.05/top-1≥90% outer floor pass on 1/17/64 rows. Python admission tests
cover invalid dimensions, huge products and the accepted int32 boundaries.
The element counts for QKV and recurrent state must each fit signed int32;
that also bounds output/gate/alpha/beta offsets. The raw-pointer ABI retains
int64 host dimensions and rejects unsafe shapes before launch.

The candidate is registered for diagnostics only. The named production
profile still selects the wide-index parent. Natural-input/model-state
qualification is the next gate; the operation result alone does not justify
a binder change. Strict/unfused primitives and the parent remain registered.

## Reproduction

Source base is `c4619579b`; the candidate was measured with tracked edits.
`screen.json` records source/build hashes, compiler version, exact commands,
all timing samples and compact trace/resource evidence. No new external
kernel was ported; this is an in-tree specialization of the current parent.

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
unset HIPENGINE_REQUIRE_CACHED_BUILD
uv run --no-sync python -c 'from hipengine.kernels.hip_gfx1100.linear_attn.qwen4_exp_gdn import build_qwen4_exp_gdn; print(build_qwen4_exp_gdn(load=False))'
export HIPENGINE_REQUIRE_CACHED_BUILD=1
uv run --no-sync pytest -q tests/test_gpu_qwen4exp_gdn_narrow.py \
  tests/test_unit_qwen4exp_gdn_narrow.py tests/test_gpu_qwen4exp_gdn_register.py \
  tests/test_gpu_qwen4exp_gdn_wave_norm.py
uv run --no-sync python scripts/qwen4exp_gdn_register_micro.py \
  --narrow-index --tokens 1 17 64 512 1024 4096 --repeats 21 \
  --output /tmp/qwen4exp-gdn-narrow-20261006/screen.json
rocprofv3 --kernel-trace --output-format csv --output-file narrow \
  --output-directory /tmp/qwen4exp-gdn-narrow-20261006/trace -- \
  uv run --no-sync python scripts/qwen4exp_gdn_register_micro.py \
    --narrow-index --tokens 512 4096 --repeats 3 \
    --output /tmp/qwen4exp-gdn-narrow-20261006/trace.json
```

The compiler-version file must be produced by the same installed `hipcc`
before cached-only profiling. Raw traces and code objects stay in `/tmp`;
[screen.json](screen.json) is the committed compact evidence.
