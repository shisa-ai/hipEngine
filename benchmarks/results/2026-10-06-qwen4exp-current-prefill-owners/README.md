# Current Qwen3.8 Flash Next prefill owners

Clean source `4779b7278` ran Qwen3.8-Flash-Next UD-Q4_K_XL on Radeon
8060S host `gfx1151`, machine-id `55ea6c509d0b49eea8de7094a1023668`, on
2026-10-06 UTC. The production manifest is
`52865d16d4a985549cd90fe8c69a5a46b6eb4543eae17a46df46c79a70be7b89`.
This is a diagnostic owner capture, not a model-speed comparison.

The canonical `code-p4096` case uses BF16 KV, chunk4096 and maximum sequence
length4352. An unprofiled process prebuilds kernels; the profiled child
requires cached builds. Both measured windows attribute all 2,761 launches.
The final full-logit hash matches across processes. Both show zero steady
allocation/byte growth and zero live allocations/bytes after close.

| Kernel owner | Mean ms/pass | Launches/pass | Kernel-union share |
| --- | ---: | ---: | ---: |
| Dense F32 pipelined DPP coltile | 2249.802 | 109 | 17.91% |
| MoE Q4_K gate/up iu8 risk | 2217.705 | 47 | 17.65% |
| Q8_0 GR-up ordered DPP | 1592.421 | 96 | 12.68% |
| MoE Q5_1 down iu8 risk | 1206.285 | 43 | 9.60% |
| GDN recurrence | 805.945 | 36 | 6.42% |
| Router F32 ordered DPP tile4 | 546.167 | 48 | 4.35% |
| Q5_1 sparse exact down repair | 513.079 | 43 | 4.08% |
| Q8_0 dense-wide, full tiles | 459.877 | 219 | 3.66% |
| Q8_0 dense-wide, predicated tails | 203.788 | 70 | 1.62% |

Kernel unions are 12,562.412/12,561.814 ms in 12,836.704/13,022.511 ms
windows. Kernel-plus-copy union leaves 273.139/459.588 ms outside device
intervals; each window has 27 copy rows. HIP API intervals overlap device
work and are not additive host overhead. Unprofiled wall13.012s is one
sample, not a throughput certificate. Separate historical captures are not
controlled before/after samples.

## Resource finding and next experiment

The two active scratch-bearing project symbols agree with current cached
code-object metadata and disassembly:

| Symbol | Private bytes/lane | Static VGPR | VGPR spills | Static scratch instructions |
| --- | ---: | ---: | ---: | ---: |
| `qwen4_exp_gdn_prefill_f32_kernel<true,true>` | 36 | 256 | 8 | 11 |
| `q8_0_dense_wide_kernel<_Float16,128,256,64,64,true,1>` | 92 | 256 | 58 | 38 |

The GDN body includes three 64-bit scratch store/load pairs and uses 64-bit
token/head/element indexing. A bounded D128 sibling can narrow those integer
intermediates without changing the recurrence or reduction tree. That is the
next screen: checked integer bounds, unchanged FP32 saved state, nonzero-state
and chunk-replay parity, input/output ownership, CPU reference, cached
resource trace and counterbalanced complete recurrence+norm+gate wall. Keep
the wide-index parent outside the admitted domain. If arithmetic differs,
review it under the production gate rather than treating strict parity as the
promotion bar. Savings are unknown; the entire806ms is not recoverable by
assumption. Dense and routed experts remain the largest owners.

Scratch grows from 1,638,400 to 3,932,160 bytes before measured windows; there
are no scratch allocation/free operations inside them. This does not prove
scratch-throttle presence or absence. Loaded libraries were not hashed by the
trace, so the metadata hashes establish current-build consistency, not
retroactive loaded-binary identity. The September29 claim of zero scratch
tree-wide is contradicted, as already shown by the September28 census.

## Reproduction

Use this tree's `uv run --no-sync`, not an ambient Python editable install.
Set the compiler-version file and ROCm libraries:

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
unset HIPENGINE_REQUIRE_CACHED_BUILD
uv run --no-sync python scripts/qwen4exp_profile_gap.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --mode prefill --case-id code-p4096 --execution-profile production \
  --prefill-chunk-size 4096 --max-sequence-length 4352 --repetitions 1 \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --output /tmp/qwen4exp-current-prefill-20261006/warm.json
export HIPENGINE_REQUIRE_CACHED_BUILD=1
rocprofv3 --kernel-trace --marker-trace --hip-trace --memory-copy-trace \
  --scratch-memory-trace --output-format csv --output-file current \
  --output-directory /tmp/qwen4exp-current-prefill-20261006/raw -- \
  uv run --no-sync python scripts/qwen4exp_profile_gap.py \
    --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
    --mode prefill --case-id code-p4096 --execution-profile production \
    --prefill-chunk-size 4096 --max-sequence-length 4352 --repetitions 2 \
    --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --require-cached-build \
    --profile --role-markers \
    --launch-census /tmp/qwen4exp-current-prefill-20261006/launch-census.json \
    --output /tmp/qwen4exp-current-prefill-20261006/profile.json
uv run --no-sync python scripts/qwen4exp_trace_audit.py \
  --trace-dir /tmp/qwen4exp-current-prefill-20261006/raw \
  --measure-prefix qwen4exp_prefill_p4096_ \
  --output /tmp/qwen4exp-current-prefill-20261006/audit.json
```

[owners.json](owners.json) records commands, hashes, per-pass attribution,
resources and correctness scope. Raw CSVs/code objects remain in `/tmp` and
are not committed. The quantized launch census is empty; no shape/FLOP join
is inferred. This unit changes no kernel, default or profile manifest.
