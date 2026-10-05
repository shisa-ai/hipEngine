# GR-up ordered DPP reduction: promoted production route

The complete GR-up operation is **1.54–1.57x faster** than its wave-scale
parent on 12 real-weight cases, with both F32 outputs bit-identical. The
tracked-clean canonical engine A/B qualifies the DPP route for the measured
UD-Q4_K_XL production profile: prefill is **6.57% / 7.15% / 6.98% faster** at
512 / 1024 / 4096 prompt tokens, respectively. All 12 cases improve, all 72
measured trajectories are exact across arms, and within-mode runs are
deterministic. Decode is effectively flat and is not claimed as a gain.

## Measurement basis

- Physical host: `gfx1151`, machine-id
  `55ea6c509d0b49eea8de7094a1023668`, AMD Radeon 8060S (Strix Halo).
- Model: `/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL`;
  model quant UD-Q4_K_XL, real Q8_0 GR-up weight banks at layers 0 and 22.
- Shape: rows 512/1024/4096, K=320, N=10240, four branches, hidden=2560.
  Activations and normalized residuals are seeded synthetic F32 inputs.
- Parent: `gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_f32`.
  Candidate: the same symbol with `wave_scale_dpp_f32` suffix.
- Timing: synchronized host wall around one prebound complete composite;
  20 pairs per case, alternating parent-first and candidate-first.
  Gate and mixed output bits are checked after every pair.
- Source: `8cd0e448a` plus the candidate implementation committed with these
  artifacts. Each run records `tracked_clean: false`; this diagnostic packet
  is not tracked-clean publication evidence.
  ROCm HIP 7.15.26333, clang 23.0.0git, compiler commit
  `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.

| Rows | Parent median range (ms) | DPP median range (ms) | Speedup range |
| ---: | ---: | ---: | ---: |
| 512 | 3.219–3.232 | 2.061–2.091 | 1.539–1.564x |
| 1024 | 6.346–6.392 | 4.056–4.086 | 1.553–1.570x |
| 4096 | 24.767–24.870 | 15.968–15.981 | 1.550–1.556x |

Every mean and both order strata improve. [Samples and weight hashes](screen.json).
This is a complete fused-operation measurement, not a projection-only shortcut.

## Correctness and engagement

The candidate preserves the FMA order, published lane-zero addition tree,
four-wave publication, sigmoid and branch mean. It substitutes
`permlanex16` plus `row_shl8/4/2/1` data-parallel primitive (DPP) operations
for the shuffle tree. Registered wave-scale and unscaled fused parents and
the unfused primitive chain remain available.

- RED before implementation: missing callable/key and route selection fail.
- Initial focused kernel/route bundle: 24 passed. Final GPU candidate/parent
  bundle after connecting the CPU-reference oracle: 17 passed.
- Full state gate: all 12 canonical cases at chunk 4096, parent/candidate/parent,
  four decode steps, full logits/recurrent/index/layout/KV hashes exact.
  Candidate calls are 0/96/0 for every case, all in prefill and zero in decode;
  active allocations and allocated bytes return to zero after close.
  [State gate](state-gate.json).
- Cached-only `rocprofv3 --kernel-trace`: correct candidate name
  `<true, true, 4, true>`, durations 3,979,214 and 4,008,992 ns, LDS 512 bytes,
  scratch 0. Allocated VGPR is 64 versus the parent's 72; static VGPR is 57
  versus 72. Static instructions fall 311 to 243. The trace proves execution
  and resources; its durations are not used as benchmark timing.
  [Trace](trace-summary.json), [static census](register-census.json).

The state gate uses the production execution profile with BF16 KV. Its profile
manifest hash is `a55324f919900c20b244b65605e37d8e12623fe20ff9b076bc3573f11edf3786`;
strict hash is `1335b8244237ffb0cac0ec9e3eb670d17ebd9d8aca607eebb463ca291b9cfbed`.
This is an exact substitution against the live parent, not admission of a new
arithmetic budget. The profile manifest names the DPP owner for the measured UD
production quant, and the exact fused parent remains its strict fallback. The
candidate override and measured call counts are retained in the engine artifact.

The complete reviewed CPU unit tier has five failures, all reproduced in a clean
baseline worktree at `5bb831283` without the candidate: stale pair32 assertion,
stale four-timestamp mock, stale dense-Q8 DPP expectation, missing published
model-identities artifact, and a mixed-pack quant provenance mismatch. Focused
candidate tests and fixture/registry/CPU smoke checks pass. No full-suite green
claim is made. GPU execution on gfx1100 is unverified.

## Reproduce

Run from this repository. Use its `uv` environment, not a conda editable install
that may resolve a different clone. The installed TheRock loader paths are
required in this shell; no environment restoration was performed.

```bash
ROCMLIB=/home/lhl/miniforge3/envs/therock/lib/python3.12/site-packages/_rocm_sdk_devel
export LD_LIBRARY_PATH="$ROCMLIB/lib:$ROCMLIB/lib64:$ROCMLIB/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/home/lhl/miniforge3/envs/therock/bin:$PATH"
export HIPENGINE_HIP_ARCH=gfx1151
mkdir -p /tmp/hipengine-gr-dpp-20261005
hipcc --version > /tmp/hipengine-gr-dpp-20261005/hipcc-version.txt
export HIPENGINE_COMPILER_VERSION_FILE=/tmp/hipengine-gr-dpp-20261005/hipcc-version.txt

# Build outside the profiler.
uv run --no-sync pytest -q tests/test_gpu_qwen4exp_gr_dpp_reduction.py \
  tests/test_gpu_qwen4exp_gr_wave_scale.py
export HIPENGINE_REQUIRE_CACHED_BUILD=1

uv run --no-sync python scripts/qwen4exp_q8_wave_scale_screen.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" --gr-composite \
  --gr-parent gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_f32 \
  --gr-candidate gguf_q8_0_gr_up_sigmoid_mean_coltile2_branch4_rowbatch4_wave_scale_dpp_f32 \
  --rows 512 1024 4096 --pairs 20 \
  --tensor blk.0.hc_attn_up.weight --tensor blk.0.hc_ffn_up.weight \
  --tensor blk.22.hc_attn_up.weight --tensor blk.22.hc_ffn_up.weight \
  --output /tmp/gr-dpp-screen.json

uv run --no-sync python scripts/qwen4exp_row4_state_gate.py \
  --model-root /models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL \
  --compiler-version-file "$HIPENGINE_COMPILER_VERSION_FILE" \
  --route-package gr-dpp-reduction --prefill-chunk-size 4096 \
  --all-cases --decode-steps 4 --full-kv --output /tmp/gr-dpp-state.json

rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/hipengine-gr-dpp-20261005/trace -- \
  uv run --no-sync pytest -q \
  'tests/test_gpu_qwen4exp_gr_dpp_reduction.py::test_dpp_gate_and_mixed_match_parent_bits[1024-320-2560]'

uv run --no-sync python scripts/qwen4exp_repair_register_census.py \
  --source hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip \
  --threads-per-block 128 --output /tmp/gr-dpp-census.json
```
