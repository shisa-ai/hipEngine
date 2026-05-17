# hipENGINE Kernel Catalog and Port Playbook

This doc is both the live kernel catalog and the mechanics for landing a kernel in hipENGINE — porting from `~/amd-gpu-tuning/nano-vllm-amd/`, the JIT build layer, gotchas specific to this repo, and the correctness gate a port must pass.

**Kernel R&D does not live here.** Micro-tuning iteration loops (`rocprofv3 --kernel-trace` ranking, VGPR/occupancy hunting, `__launch_bounds__` sweeps, fusion experiments, the device-code gotcha catalog) belong in `~/amd-gpu-tuning/`. hipENGINE receives *stable* kernels via the port pipeline below. If you find yourself opening a profiler inside the hipENGINE tree, stop and move the experiment to the parent workspace.

See also:
- `docs/PLAN.md` "Kernel Port Strategy" — authoritative source inventory, split plan, per-family targets.
- `docs/TESTING.md` — RED/GREEN, CPU-reference fixtures, and math-correctness gates.
- `~/amd-gpu-tuning/AGENTS.md` — audit-first-via-rocprofv3, time-share/occupancy/iters-per-thread/VGPR discipline.
- `~/amd-gpu-tuning/LESSONS-LEARNED.md` — device-code gotchas and kernel lineage results.
- `~/amd-gpu-tuning/docs/OPTIMAL.md` — current optimal Qwen3.5/PARO native engine route and flags.
- `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md` — Qwen3.5/PARO design history and evidence rows.

## Status legend

| Status | Meaning |
| --- | --- |
| **hipENGINE landed** | Source lives in this repo, is registered or runnable through hipENGINE, and has this repo's tests/smokes. |
| **CPU reference landed** | Torch-free NumPy oracle lives in `hipengine/kernels/cpu_reference/`; it is correctness infrastructure, not a HIP port. |
| **Lineage green** | Implemented/validated in `~/amd-gpu-tuning/nano-vllm-amd/`; source for hipENGINE's copy+partition+retype port, but not yet landed here. |
| **Lineage dirty / experimental** | Observed in the parent checkout's uncommitted worktree or R&D notes. Do not make a default hipENGINE path from it until it is promoted in `~/amd-gpu-tuning/`. |
| **Planned** | Architecture path is decided, but no hipENGINE implementation yet. |

## hipENGINE-landed kernels and oracles

This is the authoritative list of kernels/oracles that exist in this repo today. Empty backend family packages under `hipengine/kernels/hip_gfx1100/*/` are placeholders, not implemented kernels.

### CPU-reference primitive oracles (**CPU reference landed**)

Registered by `hipengine.kernels.cpu_reference.register_cpu_reference_kernels()` under `KernelKey("cpu_reference", <layer>, "fp16")`:

- `embed`
- `rmsnorm`
- `linear`
- `qkv_proj`
- `rotate`
- `attention_decode`
- `full_attn_prefill` and `full_attn_prefill_varlen` (append-then-attend causal GQA + sigmoid gate oracles)
- `o_proj`
- `lm_head`

Fixture coverage currently includes `rmsnorm`, `linear`, `rotate`, masked `attention_decode`, and causal-GQA `full_attn_prefill`; varlen full-attn is covered by direct NumPy unit tests. Run committed fixtures with `python3 scripts/check_fixtures.py`.

### gfx1100 HIP kernels (**hipENGINE landed**)

| Layer key | Quant key | Source | Public wrapper | Current gate |
| --- | --- | --- | --- | --- |
| `smoke_add` | `fp16` registry key, FP32 buffers | `hipengine/kernels/hip_gfx1100/smoke/smoke_add.hip` | `smoke_add_f32(...)` | `python3 scripts/smoke.py --mode smoke-add-hip --n 1024` → `max_abs=0.0` on W7900 |
| `rmsnorm` | `bf16` | `hipengine/kernels/hip_gfx1100/norm/rmsnorm.hip` | `qwen35_rmsnorm_bf16(...)` | `python3 scripts/smoke.py --mode qwen35-rmsnorm-hip --rows 2 --hidden-size 16` → `max_abs=0.0`, `bit_mismatch=0`; `rocprofv3` shows `qwen35_rmsnorm_kernel`, computed `DurationNs=6560`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0` on W7900 |
| `add_rmsnorm`, `add_rmsnorm_f32`, `head_rmsnorm` | `bf16` | same | `qwen35_add_rmsnorm_bf16(...)`, `qwen35_add_rmsnorm_f32_bf16(...)`, `qwen35_head_rmsnorm_f32_bf16(...)` | Build/registration tests landed; launch wrappers are source-family peers of `qwen35_rmsnorm_kernel` and share the same `.so` |
| `rmsnorm`, `add_rmsnorm` variants `paro_out`, `paro_out_fp16` | `bf16`, `w4_paro` | same | `paro_rmsnorm_out_bf16(...)`, `paro_add_rmsnorm_out_bf16(...)`, `paro_rmsnorm_out_fp16(...)`, `paro_add_rmsnorm_out_fp16(...)` | `python3 scripts/smoke.py --mode paro-rmsnorm-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 norm/add/residual (`fp16_*_mismatch=0`); `rocprofv3` shows BF16 `paro_rmsnorm_out_kernel<uint16_t>`/`paro_add_rmsnorm_out_kernel<uint16_t>` and FP16 `paro_rmsnorm_out_kernel<_Float16>` (`DurationNs=5800`, `VGPR_Count=24`, `Scratch_Size=0`, `LDS_Block_Size=1024`) / `paro_add_rmsnorm_out_kernel<_Float16>` (`DurationNs=5320`, `VGPR_Count=32`, `Scratch_Size=0`, `LDS_Block_Size=1024`) on W7900 |
| `router_logits`, `router_select`, `router_topk_shared` variants `out`, `out_fp16_hidden`, `coop_out`, `coop_out_fp16_hidden` | `bf16`, `fp16`, `fp32` select, `w4_paro` shared route | `hipengine/kernels/hip_gfx1100/moe/router.hip` | `qwen35_router_logits_bf16(...)`, `qwen35_router_logits_fp16(...)`, `qwen35_router_select(...)`, `qwen35_router_topk_shared_out_{bf16,fp16}(...)`, `qwen35_router_topk_shared_coop_out_{bf16,fp16}(...)` | `python3 scripts/smoke.py --mode qwen35-router-hip --rows 8 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → BF16 and FP16-hidden top-k/routing vs NumPy oracle (`selected_match=True`, `fp16_selected_match=True`, BF16 logits max abs `0.0`, FP16 logits max abs `2.38e-07`); `rows=1 --hidden-size 256` also validates the opt-in cooperative wrappers (`coop_selected_match=True`, `coop_fp16_selected_match=True`). `rocprofv3 --kernel-trace` all-layer 512 prefill shows FP16 hidden `qwen35_router_logits_token_tile_kernel<_Float16,4>` ran 40 times (`8.683 ms` total, avg `217.1 us`) plus block-parallel select on W7900; tokens `<4` stay on the original one-token logits kernel by default. D1.5's cooperative decode producer is gated by `HIPENGINE_PARO_ROUTER_TOPK_COOP=1`; it is correct but rejected as default after 512/128 and 4K/128 graph replay regressions. |
| `moe_group_count`, `moe_group_prefix`, `moe_group_scatter`, `moe_group_scatter_gather`, `moe_gather_packed_hidden`, `moe_wmma_tile_map` | `w4_paro` grouped/compact MoE prefill metadata and packed-hidden gather | `hipengine/kernels/hip_gfx1100/moe/group_scatter.hip` | `qwen35_moe_group_count(...)`, `qwen35_moe_group_prefix(...)`, `qwen35_moe_group_scatter(...)`, `qwen35_moe_group_scatter_gather_lowp(...)`, `qwen35_moe_gather_packed_hidden_lowp(...)`, `qwen35_moe_wmma_tile_map(...)` | `python3 scripts/smoke.py --mode qwen35-moe-group-scatter-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → grouped metadata/gather fixture passes (`prefix_match=True`, `lane_match=True`, `packed_match=True`, `tile_match=True`); `rocprofv3 --kernel-trace` shows `qwen35_moe_group_count_kernel` (`DurationNs=6640`), `qwen35_moe_group_prefix_kernel` (`11601`), `qwen35_moe_group_scatter_gather_kernel` (`11241`), `qwen35_moe_gather_packed_hidden_kernel` (`5360`), and `qwen35_moe_wmma_tile_map_kernel` (`2561`) on W7900. Expert compact WMMA is still missing, so no retained MoE throughput artifact yet. |
| `selected_dual_pack8_gemv`, `selected_pack8_gemv` variants `strided`, `transposed`, `*_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_selected_dual_pack8_*_bf16(...)`, `gemv_awq_selected_pack8_*_bf16(...)`, `gemv_awq_selected_dual_pack8_*_fp16(...)`, `gemv_awq_selected_pack8_*_fp16(...)` | `python3 scripts/smoke.py --mode paro-selected-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 dual/single, strided/transposed (`dual_mismatch=0/0`, `single_mismatch=0/0`, `fp16_dual_mismatch=0/0`, `fp16_single_mismatch=0/0`); selected dual kernels support `rows = x_rows * lanes_per_token` for batched c1-style prefill gate/up; `rocprofv3` shows FP16 selected GEMV kernels with `Scratch_Size=0`, `LDS_Block_Size=256`, `Workgroup_Size_X=64` on W7900 |
| `rotate+selected_dual_pack8_gemv` variants `strided`, `strided_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_selected_dual_pack8_strided_rotate_out_bf16(...)`, `gemv_awq_selected_dual_pack8_strided_rotate_out_fp16(...)` | `python3 scripts/smoke.py --mode paro-selected-gemv-rotate-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16/FP16 (`mismatch=0`, `fp16_mismatch=0`, `fp16_max_abs=0.0`); `rocprofv3` shows FP16 `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel<_Float16,false>` with `DurationNs=21523`, `Scratch_Size=0`, `LDS_Block_Size=320`, `Workgroup_Size_X=64` on W7900 |
| `rotate+dual_pack8_gemv` variants `transposed`, `transposed_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_dual_pack8_transposed_rotate_staged_bf16(...)`, `gemv_awq_dual_pack8_transposed_rotate_staged_fp16(...)` | `python3 scripts/smoke.py --mode paro-pack8-rotate-staged-hip --rows 1 --hidden-size 128 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16/FP16 staged rotations and outputs (`mismatch=0`, `rotated_mismatch=0`, `fp16_mismatch=0`, `fp16_rotated_mismatch=0`); D1.1 opt-in graph fixture also matched generated tokens/logits (`final_kl=0`) and `rocprofv3` showed `gemv_awq_dual_pack8_transposed_rotate_staged_kernel<_Float16,true>` with `Scratch_Size=0`, `LDS_Block_Size=512`, `VGPR_Count=104`; 512/128 decode regressed `115.450 -> 110.457 tok/s`, so the runtime default remains off. Artifact: `benchmarks/results/2026-05-17-hipengine-qwen35-d11-rotate-dual-pack8-fusion-rejected.json` |
| `pack8_gemv`, `dual_pack8_gemv` variants `strided`, `transposed`, `*_fp16`; `pack8_gemm` variants `fusedw4_prefill_fp16`, `fusedw4_prefill_strided_fp16` | `w4_paro` with BF16/FP16 activations/scales | `hipengine/kernels/hip_gfx1100/quant/paro_awq_gemv.hip` | `gemv_awq_pack8_*_bf16(...)`, `gemv_awq_dual_pack8_*_bf16(...)`, `gemv_awq_pack8_*_fp16(...)`, `gemv_awq_dual_pack8_*_fp16(...)`, `awq_fusedw4_prefill_fp16(...)`, `awq_fusedw4_prefill_strided_fp16(...)` | `python3 scripts/smoke.py --mode paro-pack8-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact generic BF16 and FP16 single/dual (`single_mismatch=0/0`, `dual_mismatch=0/0`, `fp16_single_mismatch=0/0`, `fp16_dual_mismatch=0/0`); fused-W4 route fixture gate passes (`max_kl=0.02233`, top-1 `1.0`); `rocprofv3 --kernel-trace` all-layer 512 prefill after the dual-launch wiring shows `awq_fusedw4_prefill_dual_fp16_kernel<32,32>` ran 40 times (`21.957 ms` total, avg `548.9 us`, `Scratch_Size=0`) for paired transposed Q/K and QKV/Z projections, while `awq_fusedw4_prefill_fp16_kernel<32,32,false>` ran 50 times (`14.795 ms` total) for strided V/O/linear-out projections on W7900 |
| `marlin_k_gemv` variant `fma_fp16` | `w4_paro` qweight-neutral Marlin-K FP16 rows==1 decode | `hipengine/kernels/hip_gfx1100/quant/paro_marlin_k.hip` | `gemv_paro_marlin_k_fma_fp16(...)` | `python3 scripts/smoke.py --mode paro-marlin-k-hip --rows 2 --hidden-size 128 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact vs pack8 CPU oracle (`mismatch=0`, `max_abs=0`); `rocprofv3 --kernel-trace` shows `gemv_paro_marlin_k_fma_kernel<_Float16>` (`DurationNs=6720`, `VGPR_Count=104`, `Scratch_Size=0`, `LDS_Block_Size=512`) on W7900. Model fixture gates pass with default qweight-neutral replacement (`max_kl=0.0395688706`, top-1 `1.0`; graph/eager generated IDs match, final KL `0`), and D2.1 3-run diagnostic improves 512/128 and 4K/128 decode by `+5.57%/+5.61%` while lowering tracked peak by `0.411 GiB` vs `HIPENGINE_PARO_MARLIN_K_REPLACE=0`. |
| `silu_mul_dual`, `silu_mul_separate`, `silu_mul_dual_rotate`, `silu_mul_pair_rotate` variants `out`, `out_fp16` | `bf16`, `fp16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/fused/paro_silu.hip` | `silu_mul_dual_out_bf16(...)`, `silu_mul_dual_out_fp16(...)`, `silu_mul_separate_out_bf16(...)`, `silu_mul_separate_out_fp16(...)`, `silu_mul_dual_rotate_out_bf16(...)`, `silu_mul_dual_rotate_out_fp16(...)`, `silu_mul_pair_rotate_out_bf16(...)`, `silu_mul_pair_rotate_out_fp16(...)` | `python3 scripts/smoke.py --mode paro-silu-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 dual SiLU and dual/pair rotate (`*_mismatch=0`, `*_fp16_mismatch=0`); `rocprofv3` shows FP16 `silu_mul_dual_out_kernel<_Float16>` (`DurationNs=1680`, `Scratch_Size=0`), `silu_mul_dual_rotate_out_kernel<_Float16>` (`DurationNs=11960`, `Scratch_Size=0`, `LDS_Block_Size=64`), and `silu_mul_pair_rotate_out_kernel<_Float16>` (`DurationNs=8480`, `Scratch_Size=0`) on W7900. `silu_mul_separate_out_{fp16,bf16}` is a hipENGINE-original variant that takes two separate `[rows, features]` buffers (gate, up) instead of a packed `[rows, 2*features]` input; used by the W4 PARO dense shared expert where gate/up have distinct rotations and can’t share a packed layout. Bit-exact vs `silu_mul_dual_out_fp16` on synthetic data (`max_abs_diff=0`) and launches successfully on W7900. |
| `weighted_sum`, `weighted_lanes_sum`, `weighted_sum+shared_gate+residual`, `shared_gate_combine`, `shared_gate_combine+residual` variants `out`, `out_fp16`, `batch_out`, `batch_out_fp16` | `bf16`, `fp16`, `w4_paro` with FP32 weights/gate logits | `hipengine/kernels/hip_gfx1100/fused/paro_combine.hip` | `weighted_sum_out_{bf16,fp16}_f32w(...)`, `weighted_lanes_sum_out_{bf16,fp16}_f32w(...)`, `weighted_sum_shared_gate_combine_residual*_{bf16,fp16}_f32w(...)`, `shared_gate_combine*_{bf16,fp16}(...)` | `python3 scripts/smoke.py --mode paro-combine-hip --rows 4 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 weighted/shared/residual combine including batched selected weighted + shared/residual and grouped sorted-lane accumulation (`*_mismatch=0`, `*_fp16_mismatch=0`); `rocprofv3` shows weighted-lane and batch combine kernels, including `weighted_lanes_sum_out_kernel<unsigned short>` (`DurationNs=2080`), `weighted_lanes_sum_out_kernel<_Float16>` (`2000`), and `shared_gate_combine_residual_batch_out_kernel<_Float16>` (`2120`) on W7900 |
| `awq_wmma` compact selected dual/single pack8 | `bf16`, `fp16`, `w4_paro` compact grouped MoE | `hipengine/kernels/hip_gfx1100/wmma/paro_awq_wmma.hip` | `gemm_awq_selected_dual_pack8_wmma_compact_{bf16,fp16}(...)`, `gemm_awq_selected_pack8_wmma_compact_{bf16,fp16}(...)` | `python3 scripts/smoke.py --mode paro-awq-wmma-compact-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → tiny compact AWQ WMMA fixture passes (`dual_mismatch=0`, `single_mismatch=0`, FP16 mismatches `0`); `rocprofv3` shows compact dual/single WMMA kernels for BF16/FP16, e.g. BF16 dual `DurationNs=10520`, BF16 single `6361`, FP16 dual `6760`, FP16 single `5161` on W7900 |
| `dense_gemv`/`dense_dual_gemv` variants `out`, `out_fp16` | `bf16`, `fp16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/linear/dense_gemv.hip` | `dense_gemv_out_bf16(...)`, `dense_dual_gemv_out_bf16(...)`, `dense_gemv_out_fp16(...)`, `dense_dual_gemv_out_fp16(...)` | `python3 scripts/smoke.py --mode dense-gemv-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 single and FP16 single/dual GEMV (`mismatch=0`, `fp16_mismatch=0`, `dual_fp16_mismatch=0`, max abs `0.0`); `rocprofv3` shows FP16 `dense_gemv_out_kernel<_Float16>` (`DurationNs=3440`, `Scratch_Size=0`, `LDS_Block_Size=1024`) and `dense_dual_gemv_out_kernel<_Float16>` (`DurationNs=4040`, `Scratch_Size=0`) on W7900 |
| `lm_head` variant `fp16_argmax_bf16` | `w4_paro` BF16 hidden + FP16 checkpoint head | `hipengine/kernels/hip_gfx1100/linear/lm_head.hip` | `lm_head_fp16_argmax_bf16(...)` | `python3 scripts/smoke.py --mode lm-head-hip --hidden-size 32 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact id/logit (`index_match=True`, `abs=0.0`); `rocprofv3 --kernel-trace` shows `lm_head_fp16_logits_kernel`, `argmax_stage1_kernel`, and `argmax_stage2_kernel` with `Scratch_Size=0` on W7900 |
| `w8a16_linear` variants `bf16_f32_out`, `bf16_lowp_out`, `fp16_lowp_out`, `shared_gate_up_silu_fp16`, `shared_gate_up_silu_fp16_token_tiled`, `shared_gate_sigmoid_fp32`, `shared_down_combine_residual_fp16`, `shared_down_combine_residual_fp16_token_tiled`, `f32_f32_out` | `w8a16`, `w4_paro` | `hipengine/kernels/hip_gfx1100/quant/w8a16_linear.hip` | `w8a16_linear_bf16_f32_out(...)`, `w8a16_linear_bf16_lowp_out(...)`, `w8a16_linear_fp16_lowp_out(...)`, `w8a16_shared_gate_up_silu_fp16(...)`, `w8a16_shared_gate_up_silu_fp16_token_tiled(...)`, `w8a16_shared_gate_sigmoid_fp32(...)`, `w8a16_shared_down_combine_residual_fp16(...)`, `w8a16_shared_down_combine_residual_fp16_token_tiled(...)`, `w8a16_linear_f32_f32_out(...)` | `python3 scripts/smoke.py --mode w8a16-linear-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `bf16_f32_max_abs=0.0`, `f32_f32_max_abs=4.77e-07`, `lowp_mismatch=0`, `fp16_lowp_mismatch=0`; fused shared route fixture gate passes (`max_kl=0.03406`, top-1 `1.0`); P1.2 token-tiled gate/up and P1.3 token-tiled down+combine microchecks match the original kernels (`tile2/tile4 max_abs=0`) and fixture gates pass (`max_kl=0.0396`, top-1 `1.0`); `rocprofv3 --kernel-trace` confirms `w8a16_shared_gate_up_silu_fp16_token_tiled_kernel<2>` and `w8a16_shared_down_combine_residual_fp16_token_tiled_kernel<2>` for legacy prompts; previous all-layer 512 prefill shows FP16 `w8a16_shared_down_combine_residual_fp16_kernel` ran 40 times (`16.047 ms` total, avg `401.166 us`, 8-row tile), `shared_gate_sigmoid_fp32_kernel` ran 40 times (`0.092 ms` total), and `w8a16_shared_gate_up_silu_fp16_kernel` ran 40 times (`15.562 ms` total) on W7900 |
| `paro_rotate1`, `paro_rotate2`, `paro_rotate3` variants `bf16`, `fp16`; `paro_rotate1` variant `bf16_gate_fp16` | `w4_paro` | `hipengine/kernels/hip_gfx1100/rotary/paro_rotate.hip` | `paro_rotate1_bf16(...)`, `paro_rotate2_bf16(...)`, `paro_rotate3_bf16(...)`, `paro_rotate1_fp16(...)`, `paro_rotate2_fp16(...)`, `paro_rotate3_fp16(...)`, `paro_rotate1_bf16_gate_fp16(...)` | `python3 scripts/smoke.py --mode paro-rotate-hip --rows 2 --hidden-size 16 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 rotate2/3 (`mismatches=[0, 0, 0, 0, 0]`) and FP16 rotate1/2/3 (`fp16_mismatches=[0, 0, 0, 0, 0, 0]`, `fp16_max_abs=0.0`); AOTriton gate-rotate fixture passes (`max_kl=0.0396`, top-1 100%); `rocprofv3` shows FP16 `paro_rotate1_kernel<_Float16>` (`DurationNs=11680`, `Scratch_Size=0`, `LDS_Block_Size=32`), `paro_rotate2_kernel<_Float16>` (`DurationNs=2680`), and `paro_rotate3_kernel<_Float16>` (`DurationNs=2560`) on W7900 |
| `partial_rotary`, `head_rmsnorm+partial_rotary`, `split_qgate` variants `qwen35_f32`, `qwen35_f32_bf16`, `qwen35_position_f32_bf16`, `qwen35_positions_f32_bf16`, `qwen35_positions_q_bf16_key_f32`, `bf16`, `fp16` | `w4_paro` full-attention prelude plus resident q/gate split | `hipengine/kernels/hip_gfx1100/rotary/qwen35_rotary.hip` | `qwen35_partial_rotary_f32(...)`, `qwen35_head_rmsnorm_partial_rotary*_f32_bf16(...)`, `qwen35_head_rmsnorm_partial_rotary_positions_f32_bf16(...)`, `qwen35_head_rmsnorm_partial_rotary_positions_q_bf16_key_f32(...)`, `qwen35_split_qgate_{bf16,fp16}(...)` | `python3 scripts/smoke.py --mode qwen35-rotary-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → `partial_max_abs=0`, `head_max_abs=2.38e-07`, `position_max_abs=2.38e-07`, `vector_position_max_abs=2.38e-07`, `split_fp16_query_max_abs=0`, `split_fp16_gate_mismatch=0`; AOTriton cast-glue fixture gate passes with `max_kl=0.0396`, top-1 `100%` after using the BF16-Q/FP32-K vector-position variant; prior `rocprofv3` shows scalar parent kernels with `Scratch_Size=0` plus FP16 `qwen35_split_qgate_fp16_kernel` (`DurationNs=3720`) on W7900 |
| `cast_f32_to_bf16`, `cast_bf16_to_f32`, `cast_f32_to_fp16`, `cast_fp16_to_f32`, `cast_fp16_to_bf16` | `bf16`, `fp16`, `fp32` runtime glue | `hipengine/kernels/hip_gfx1100/convert/cast.hip` | `f32_to_bf16(...)`, `bf16_to_f32(...)`, `f32_to_fp16(...)`, `fp16_to_f32(...)`, `fp16_to_bf16(...)` | `python3 -m pytest tests/test_cast_plan.py tests/test_qwen35_decode_state.py -q`; resident parent-mixed E2E fixture exercises `fp16_to_bf16(...)` before the BF16 lm-head path, while standalone bit-exact GPU smoke remains pending. |
| `token_embedding`, `decode_position`, `scalar_state` runtime helpers | `w4_paro` graph-friendly state | `hipengine/kernels/hip_gfx1100/runtime/state.hip` | `embedding_lookup_{bf16,fp16}_i64(...)`, `embedding_lookup_batch_{bf16,fp16}_i64(...)`, `embedding_lookup_batch_mapped_{bf16,fp16}_i64(...)`, `set_i64_scalar(...)`, `set_i64_vector(...)`, `set_decode_position_i64(...)`, `set_decode_positions_i64(...)`, `advance_decode_position_i64(...)`, `advance_decode_positions_i64(...)`, `record_i64_scalar_indexed(...)` | `python3 -m pytest tests/test_runtime_state_plan.py -q`; GPU smoke copies BF16 and FP16 embedding rows through an optional row map and advances masked decode position/context vectors (`runtime_state batch smoke OK`); `python3 scripts/qwen35_decode_graph_fixture_gate.py --fixture fixtures/qwen35_paro/parent_512_32_seed1234.json --max-layers 40 --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build --attn-aotriton-min-tokens 512` validates device-side generated-token recording inside replay (`passed=true`, final KL `0`). |
| `linear_attn_conv_decode` variants `f32`, `bf16`, `fp16`; `linear_attn_conv_prefill` variants `f32`, `f32_segments` | `w4_paro` linear-attention decode/prefill | `hipengine/kernels/hip_gfx1100/linear_attn/conv.hip` | `qwen35_linear_attn_conv_decode_f32(...)`, `qwen35_linear_attn_conv_decode_bf16(...)`, `qwen35_linear_attn_conv_decode_fp16(...)`, `qwen35_linear_attn_conv_prefill_f32(...)`, `qwen35_linear_attn_conv_prefill_segments_f32(...)` | Decode: `python3 scripts/smoke.py --mode qwen35-linear-attn-conv-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → `f32_out_max_abs=7.45e-09`, `bf16_out_max_abs=7.45e-09`, `fp16_out_max_abs=7.45e-09`, state max abs `0`; `rocprofv3` shows FP16 `qwen35_linear_attn_conv_decode_lowp_kernel<_Float16>` (`DurationNs=5680`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`) on W7900. Prefill: `qwen35-linear-attn-prefill-hip` → `conv_out_max_abs=1.49e-08`, `conv_state_max_abs=0`. Segment prefill: `qwen35-linear-attn-segments-hip` → `segment_conv_out_max_abs=1.86e-09`, `segment_conv_state_max_abs=0`; `rocprofv3` shows `qwen35_linear_attn_conv_prefill_segments_kernel` (`DurationNs=5800`) and segment state kernel (`2200`) on W7900. |
| `gdn_recurrent_rmsnorm_gate` variants `bf16_lowp`, `fp16_lowp`; `linear_attn_prefill_prepare` variants `f32_bf16`, `f32_fp16`; `gdn_prefill_recurrent` variants `f32`, `f32_k2`, `f32_k2_segments`; `gdn_prefill_rmsnorm_gate` variants `bf16`, `fp16`; `gdn_prefill_rmsnorm_gate_rotate` variant `fp16` | `w4_paro` linear-attention decode/prefill | `hipengine/kernels/hip_gfx1100/linear_attn/gdn.hip` | `qwen35_gdn_recurrent_rmsnorm_gate_lowp_bf16(...)`, `qwen35_gdn_recurrent_rmsnorm_gate_lowp_fp16(...)`, `qwen35_linear_attn_prefill_prepare_f32_bf16(...)`, `qwen35_linear_attn_prefill_prepare_f32_fp16(...)`, `qwen35_gdn_prefill_recurrent*_f32(...)`, `qwen35_gdn_prefill_recurrent_segments_k2_f32(...)`, `qwen35_gdn_prefill_rmsnorm_gate_{bf16,fp16}(...)`, `qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16(...)` | Decode: `python3 scripts/smoke.py --mode qwen35-linear-attn-gdn-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → BF16/FP16 `out_max_abs=2.98e-08`, `state_max_abs=1.49e-08`; `rocprofv3` shows FP16 `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel<_Float16>` (`DurationNs=9920`, `VGPR_Count=56`, `Scratch_Size=0`, `LDS_Block_Size=1616`). Prefill: `qwen35-linear-attn-prefill-hip` → BF16 `gated_mismatch=0`, FP16 `fp16_gated_mismatch=0`, fused FP16 gate+rotate `fused_rotate_mismatch=0`, `fp16_prepare_max_abs=5.96e-08`; `qwen35-linear-attn-segments-hip` → `segment_gdn_out_max_abs=1.86e-09`, `segment_gdn_state_max_abs=9.31e-10`; `rocprofv3` shows `qwen35_gdn_prefill_recurrent_k2_segments_kernel` (`DurationNs=5480`) on W7900; all-layer 512 prefill after the fixed two-wave k2 reduction specialization shows `qwen35_gdn_prefill_recurrent_k2_kernel` ran 30 times (`40.993 ms` total, avg `1366.4 us`) on W7900. |
| `paged_kv_write` variants `mixed_bf16_spans`, `mixed_fp16_spans`, `mixed_bf16_batch_spans`, `mixed_fp16_batch_spans`, `mixed_fp16_prompt_spans`, `f32_spans` | `w4_paro` full-attention KV append, BF16 cache | `hipengine/kernels/hip_gfx1100/attention/paged_kv_write.hip` | `qwen35_write_paged_kv_mixed_value_{bf16,fp16}_spans(...)`, `qwen35_write_paged_kv_mixed_value_{bf16,fp16}_batch_spans(...)`, `qwen35_write_paged_kv_mixed_value_fp16_prompt_spans(...)`, `qwen35_write_paged_kv_f32_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-kv-write-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact KV append (`mixed_mismatch=0/0`, `mixed_fp16_mismatch=0/0`, `f32_mismatch=0/0`, `untouched_nonzero=0`); public wrapper accepts `KVLiveSpans`, where fixed-page `base_offsets` carries the parent block table and `live_counts` carries the position tensor; batched smoke validates row-major c>1 append, while `qwen35-paged-attn-prefill-hip` now validates single-request prompt append plus causal prefill attention; `rocprofv3` shows prompt writer `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel<_Float16>` (`DurationNs=12078`) and prefill attention (`DurationNs=26036`) on W7900 |
| `full_attn_decode` variant `bf16_context`; `full_attn_gate_mul` variants `bf16`, `fp16` | `w4_paro` short-context full-attention decode, BF16 dense KV cache, lowp gated output | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_full_attn_decode_context_bf16(...)`, `qwen35_full_attn_gate_mul_bf16(...)`, `qwen35_full_attn_gate_mul_fp16(...)` | `python3 scripts/smoke.py --mode qwen35-full-attn-decode-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax oracle `max_abs=1.19e-07`, BF16/FP16 gate outputs bit-exact (`gated_bf16_mismatch=0`, `gated_fp16_mismatch=0`); resident Qwen3.5/PARO uses this dense parent kernel for max context <1024 before the paged path; `rocprofv3` shows FP16 `qwen35_full_attn_gate_mul_fp16_kernel` (`DurationNs=1360`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=0`) on W7900 |
| `paged_attn_decode` variants `bf16_context_spans`, `bf16_context_batch_spans` | `w4_paro` full-attention decode, BF16 KV cache | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_context_bf16_spans(...)`, `qwen35_paged_full_attn_decode_context_bf16_batch_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-decode-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax oracle `max_abs=2.98e-08`; public wrapper accepts `KVLiveSpans` (`base_offsets` page table, `live_counts` context tensor); c>1 row-major smoke `batched paged kv+attn smoke OK` validates uneven context lengths; `rocprofv3` shows scalar `qwen35_paged_full_attn_decode_context_tensor_kernel` with `DurationNs=7640`, `VGPR_Count=40`, `Scratch_Size=0`, `Workgroup_Size_X=256` on W7900 |
| `paged_attn_decode` variant `bf16_split_k_spans` | `w4_paro` long-context full-attention decode, BF16 KV cache | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_bf16_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-split-k-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax oracle `max_abs=5.96e-08`; public wrapper runs parent split-K context kernel then reduce using caller-provided workspaces; `rocprofv3` shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel` (`DurationNs=17320`, `VGPR_Count=32`, `Scratch_Size=0`) and `qwen35_paged_full_attn_decode_split_k_reduce_kernel` (`DurationNs=6320`, `VGPR_Count=16`, `Scratch_Size=0`) on W7900 |
| `paged_attn_decode` variant `bf16_split_k_gate_f32_spans` | `w4_paro` long-context full-attention decode + gate, BF16 KV cache, FP32 gate/out | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_gate_f32_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-gate-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → NumPy softmax+sigmoid oracle `gated_max_abs=4.47e-08`; `rocprofv3` shows `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel` (`DurationNs=16320`, `VGPR_Count=32`, `Scratch_Size=0`) and `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<float>` (`DurationNs=5000`, `VGPR_Count=16`, `Scratch_Size=0`) on W7900 |
| `paged_attn_decode` variants `bf16_split_k_gate_bf16_spans`, `bf16_split_k_gate_fp16_spans` | `w4_paro` long-context full-attention decode + gate, BF16 KV cache, BF16/FP16 gate/out | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_gate_bf16_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gate_fp16_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-gate-bf16-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → bit-exact BF16 and FP16 outputs (`bf16_mismatch=0`, `fp16_mismatch=0`, max abs `0`); wrappers instantiate the parent gated reduce with `hip_bfloat16` and `_Float16`, not integer casts; `rocprofv3` shows FP16 `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel<_Float16>` (`DurationNs=10040`, `VGPR_Count=16`, `Scratch_Size=0`, `LDS_Block_Size=24`) on W7900 |
| `paged_attn_decode` variants `bf16_split_k_warp_spans`, `bf16_split_k_gqa_spans`, `bf16_split_k_gqa_gate_bf16_spans`, `bf16_split_k_gqa_gate_fp16_spans` | `w4_paro` Qwen3.5 GQA-specialized long-context full-attention decode | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_decode_split_k_warp_bf16_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_bf16_spans(...)`, `qwen35_paged_full_attn_decode_split_k_gqa_gate_{bf16,fp16}_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` → Qwen3.5 shape `[16,256] / 2 KV`, `ctx=512`, NumPy oracle `warp_max_abs=4.1e-08`, `gqa_max_abs=4.1e-08`, BF16 gated output bit-exact (`gqa_gate_bf16_mismatch=0`); FP16 GQA gated wrapper shares the same `_Float16` reduce instantiated by `qwen35-paged-attn-gate-bf16-hip`. `python3 scripts/smoke.py --mode qwen35-paged-attn-gqa-state-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt --require-cached-build` drives KV append + GQA gated decode through `Qwen35ParoDecodeState` and is bit-exact (`appended_key_mismatch=0`, `appended_value_mismatch=0`, `gqa_gate_bf16_mismatch=0`); `rocprofv3` shows state-path KV append (`DurationNs=3080`, `Scratch_Size=0`), `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel<8,16,2>` (`DurationNs=67560`, `VGPR_Count=80`, `Scratch_Size=0`), and BF16 gated reduce (`DurationNs=2760`, `Scratch_Size=0`) on W7900 |
| `full_attn_prefill` variants `qwen35_causal_gqa_gate_fp16`, `qwen35_varlen_causal_gqa_gate_fp16` | `w4_paro` append-then-attend causal GQA prefill, BF16 KV cache, FP16 gate/output | `hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.hip` | `qwen35_paged_full_attn_prefill_gqa_gate_fp16_spans(...)`, `qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_spans(...)` | `python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → tiny paged causal-GQA fixture vs CPU `full_attn_prefill` oracle after prompt KV append, `prefill_gate_fp16_max_abs=0`, `prefill_gate_fp16_mismatch=0`; `python3 scripts/smoke.py --mode qwen35-paged-attn-prefill-varlen-hip --compiler-version-file /tmp/hipengine-hipcc-version.txt` → two packed request segments with row-shaped block tables, `varlen_prefill_gate_fp16_max_abs=0`, mismatch `0`; `rocprofv3` shows prompt KV writer (`DurationNs=6880`) and `qwen35_paged_full_attn_prefill_varlen_gqa_gate_fp16_kernel` (`21520`) on W7900; all-layer 512 prefill after the shared-query cache/vector key-dot update, fixed `block_size=256` address fast path, and split short-row template shows `qwen35_paged_full_attn_prefill_gqa_gate_fp16_kernel<true>` ran 10 times (`26.362 ms` total, avg `2636.2 us`) on W7900. Full single-request fixture gate accepted in `benchmarks/results/2026-05-15-hipengine-qwen35-native-prefill-full-single-request-accepted.json` (`max_kl=0.0168`, top-1 100%), and active multiloop fixture gate remains green (`max_kl=0.03406`, top-1 100%), but no throughput row promoted. |

`smoke_add` is a build/runtime smoke, not a model-layer primitive. It proves `hipengine.core.build`, lazy `libamdhip64.so`, device allocation/copy, launch, synchronize, and copyback without torch.

`qwen35_rmsnorm` is the first real model-layer HIP family port. It is BF16-bit (`uint16_t`) at the raw pointer ABI; Qwen weights store deltas and the kernel applies `1.0 + weight_delta`. PARO `paro_out` RMSNorm variants use direct norm weights and caller-owned output buffers, matching the parent native PARO serving path.

`paro_awq_gemv` ports the selected-expert and generic pack8 GEMV bodies used by the current OPTIMAL MoE c=1 route and non-MoE projections. The fused rotate→selected-dual GEMV path is landed for the parent strided layout; generic non-MoE and selected-MoE wrappers now cover strided/transposed qweight layouts for both BF16 and parent-parity FP16 activation/scale buffers. D1.1 added a generic transposed rotate-staged dual GEMV surface for decode diagnostics, but it remains opt-in/default-off because the rotate-once barrier/staging path regressed 512/128 decode. The FP16 `awq_fusedw4_prefill_fp16` WMMA prefill projection is ported from `nano-vllm-amd@55fede9` (`paroquant_fusedw4.py`) and is used for multi-token transposed pack8 prompt projections. hipENGINE also provides `awq_fusedw4_prefill_dual_fp16`, a same-math dual-output launch used for paired transposed Q/K and QKV/Z prefill projections, plus a strided-layout instantiation for V/O/linear-out prompt projections without adding transposed weight copies. The GEMV wrappers are retained as c=1/small-row fallbacks.

`paro_marlin_k` ports the parent retained Marlin-K v0 vec8 FP32-FMA rows==1 decode path documented in `docs/MARLIN.md` and `/home/lhl/amd-gpu-tuning/PLAN-PAROQUANT2.md` (`nano-vllm-amd@7718fff` vec8 FMA and `@1522293` qweight-neutral replacement; those short SHAs are documented but not present in the current parent checkout). hipENGINE materializes `qweight_mk [N/8,K/128,128]`, small `qzeros_mk/scales_mk` decode metadata, and a zero-copy `qweight_pack8_decode [N/8,K]` alias so prefill and fused pair projections keep using the existing pack8/fusedw4 paths without duplicate large W4 buffers. The env gate `HIPENGINE_PARO_MARLIN_K_REPLACE` defaults on; setting it to `0` restores the old pack8/raw-qweight materialization for diagnostics.

`paro_silu` ports the selected-expert activation and down-rotation stage, including the fused `silu_mul_dual_rotate_out_kernel` path used by the parent default and the unfused/separate-gate fallback kernels.

`paro_combine` ports the c=1 selected-weighted/shared-gate/residual combine kernels. The current hipENGINE wrappers cover the parent default FP32 router-weight/gate-logit path; scalar-weight variants can be added if a future route needs them.

`dense_gemv` ports the parent PARO BF16 dense GEMV used by auxiliary dense paths such as linear-attention AB projections when they remain dense rather than W4/W8 quantized. `lm_head` is the temporary GPU E2E bring-up head: FP16 checkpoint weights, BF16 hidden input, FP32 logits, and two-stage GPU argmax so the one-token harness no longer depends on CPU NumPy for final-token selection.

`paro_rotate` ports the parent PARO pairwise rotation helpers used by PARO projection paths (`paro_rotate1`, `paro_rotate2`, `paro_rotate3`); rotate1 is the single-output specialization needed by projection tails such as linear-attention `out_proj`. The wrappers cover both BF16 and parent-parity FP16 activation/scales, plus `paro_rotate1_bf16_gate_fp16(...)` for the AOTriton prefill tail: it rounds `BF16 attention * sigmoid(FP16 gate)` to FP16 before applying the same PARO rotate1 math, matching the old gate-kernel + rotate1 sequence while removing one launch. `qwen35_rotary` ports the parent full-attention prelude (`partial_rotary`, fused head RMSNorm + partial rotary, table-positioned scalar fused head RMSNorm + partial rotary, and hipENGINE's vector-position `(tokens, heads)` prefill variant) plus resident q/gate split helpers for BF16 and parent-mixed FP16 activation streams; the AOTriton prefill path also has a vector-position variant that writes BF16 Q directly while preserving FP32 K for the paged-KV append. `convert/cast` provides small runtime glue casts for paths where a parent kernel emits FP32 or FP16 but the next PARO/lm-head projection consumes a different lowp dtype.

`w8a16_linear` ports the parent W8A16 GEMV kernels used by the current shared-expert default (`hip_w8a16_linear_lowp_out`) and W8A16 lm-head/auxiliary dense route. Lowp output wrappers now cover both BF16 and parent-parity FP16 activation streams. The FP16 `w8a16_shared_gate_up_silu_fp16` prefill helper adapts parent `w8a16_shared_gate_up_bulk4_kernel` to the raw-pointer lowp-output path, computing four shared-expert intermediate columns per block and writing the existing `shared_intermediate` scratch. `w8a16_shared_gate_up_silu_fp16_token_tiled` is a hipENGINE prefill variant that preserves W8A16 storage while sharing gate/up weights across adjacent prompt tokens; runtime defaults use `token_tile=2` for legacy shared experts only when `tokens >= 1024`, with the original helper retained as fallback/opt-out. `w8a16_shared_gate_sigmoid_fp32` precomputes the shared-expert sigmoid once per token in the router shared-gate column after top-k/routing weights are materialized. The FP16 `w8a16_shared_down_combine_residual_fp16` helper consumes that precomputed gate while fusing grouped-prefill shared down projection with selected-output/shared-gate/residual combine; its default tile computes eight hidden rows per block, preserving the already-rounded `selected_out` ABI and exact per-row accumulation order. `w8a16_shared_down_combine_residual_fp16_token_tiled` shares the same fused tail while reusing down rows across adjacent prompt tokens; runtime defaults use `token_tile=2` for legacy prefill `tokens >= 2`, with the original helper retained as fallback/opt-out. c=1 and non-grouped paths keep the unfused gate/up/down/combine fallbacks. `scripts/smoke.py --mode w8a16-shared-expert-hip` chains W8A16 gate/up → `silu_mul_dual_out` → W8A16 down and is bit-exact against the staged BF16 NumPy oracle. `scripts/smoke.py --mode paro-moe-c1-hip --hidden-size 8` is the direct synthetic c=1 decode vertical smoke; `scripts/smoke.py --mode paro-moe-c1-state-hip --hidden-size 8` drives the same staged fixture through `Qwen35ParoDecodeState.run_moe_c1_bf16(...)` and validates the normalized prepared-weight/runtime-workspace path.

## Source-lineage drift check

Before porting a family, check whether the parent source moved since the last hipENGINE catalog/audit baseline:

```bash
python3 scripts/check_lineage.py --diff stat
```

Useful filters:

```bash
python3 scripts/check_lineage.py --kind kernel --diff stat
python3 scripts/check_lineage.py --file '*paroquant*' --diff patch
python3 scripts/check_lineage.py --fail-on-drift
```

The script is read-only. It uses `docs/source_lineage.json` to compare tracked files in `~/amd-gpu-tuning/nano-vllm-amd/` against the recorded baseline ref, then reports:

- current child-repo branch/HEAD,
- per-file dirty status,
- commits since baseline,
- diffstat or patch for the file,
- matching lines in `~/amd-gpu-tuning/WORKLOG.md` and relevant parent docs.

If a file reports **DRIFT**, inspect the listed commits/diff and read the evidence hits before copying code. Update `docs/source_lineage.json`'s baseline only after the catalog/port source is intentionally refreshed and logged in `WORKLOG.md`.

## Source-lineage kernel catalog to port

The stable source-lineage port set at the current hipENGINE catalog baseline is the committed `nano-vllm-amd` Qwen3.5/PARO kernel set: **95** kernels from `csrc/amd/qwen35_expert.hip` plus **25** PARO kernels from `nanovllm/native/qwen35/paroquant_kernels.py` = **120 Qwen/PARO kernels**, plus the separate `smoke_add` build smoke. hipENGINE ports these by family; bodies are preserved byte-for-byte except for includes and raw-pointer host-wrapper retyping.

### Atomic / primitive-oriented kernel families (**source-lineage status; hipENGINE-landed where noted**)

- `wmma/wmma_i8_gemm.hip` (4):
  - `qwen35_wmma_i8_tile_kernel`
  - `qwen35_wmma_i8_gemm_kernel`
  - `qwen35_wmma_i8_gemm_a_row_major_kernel`
  - `qwen35_wmma_i8_gemm_grouped_a_row_major_kernel`
- `quant/w8a8_activation.hip` (2):
  - `qwen35_quantize_activation_i8_per_row_kernel`
  - `qwen35_quantize_activation_f32_i8_per_row_kernel`
- `moe/w8a8_grouped.hip` (10):
  - `qwen35_dequantize_w8a8_projection_kernel`
  - `qwen35_dequantize_w8a8_grouped_projection_kernel`
  - `qwen35_dequantize_w8a8_grouped_accumulate_kernel`
  - `qwen35_dequantize_w8a8_grouped_accumulate_deterministic_kernel`
  - `qwen35_dequantize_w8a8_c1_grouped_accumulate_kernel`
  - `qwen35_moe_grouped_accumulate_kernel`
  - `qwen35_moe_grouped_gate_up_kernel`
  - `qwen35_moe_grouped_down_kernel`
  - `qwen35_moe_grouped_down_flat_kernel`
  - `qwen35_moe_grouped_down_flat_accumulate_kernel`
- `moe/swiglu.hip` (2):
  - `qwen35_swiglu_packed_gate_up_kernel`
  - `qwen35_dequantize_swiglu_quantize_grouped_kernel`
- `quant/w8a16_moe.hip` (17):
  - `w8a16_selected_experts_kernel`
  - `w8a16_gate_up_kernel`
  - `w8a16_down_kernel`
  - `w8a16_gate_up_shared_kernel`
  - `w8a16_gate_up_shared_t_kernel`
  - `w8a16_gate_up_shared_t_decode_v2_kernel`
  - `w8a16_down_shared_kernel`
  - `w8a16_down_shared_bulk_combine_kernel`
  - `w8a16_down_shared_t_kernel`
  - `w8a16_down_shared_t_decode_v2_kernel`
  - `w8a16_down_shared_bulk_combine_t_kernel`
  - `w8a16_single_gate_up_kernel`
  - `w8a16_single_down_combine_kernel`
  - `w8a16_shared_gate_up_bulk_kernel`
  - `w8a16_shared_gate_up_bulk4_kernel`
  - `w8a16_shared_down_bulk_combine_kernel`
  - `w8a16_shared_down_bulk_combine_w8a8_c1_selected_kernel`
- `moe/group_scatter.hip` (11):
  - `qwen35_moe_group_count_kernel`
  - `qwen35_moe_group_prefix_kernel`
  - `qwen35_moe_group_scatter_kernel`
  - `qwen35_moe_group_scatter_gather_kernel`
  - `qwen35_moe_c1_group_metadata_kernel`
  - `qwen35_moe_c1_group_metadata_gather_kernel`
  - `qwen35_moe_c1_group_metadata_quantize_kernel`
  - `qwen35_moe_gather_packed_hidden_kernel`
  - `qwen35_moe_gather_quantize_packed_hidden_kernel`
  - `qwen35_build_lane_to_sorted_kernel`
  - `qwen35_moe_combine_kernel`
- `moe/router.hip` top-k subset (2) — **hipENGINE landed for BF16 hidden/weight raw-pointer wrappers**:
  - `qwen35_router_logits_kernel`
  - `qwen35_router_select_kernel`
- `moe/router.hip` token-rank/top2 subset (4):
  - `qwen35_token_rank_count_partial_kernel`
  - `qwen35_token_rank_count_finalize_kernel`
  - `qwen35_token_top2_partial_kernel`
  - `qwen35_token_top2_finalize_kernel`
- `quant/w8a16_linear.hip` (5):
  - `w8a16_linear_kernel`
  - `w8a16_linear_lowp_out_kernel`
  - `w8a16_linear_f32_kernel`
  - `w8a16_linear_batched_kernel`
  - `w8a16_linear_batched_f32_kernel`
- `linear_attn/conv.hip` (6):
  - `qwen35_linear_attn_conv_decode_kernel`
  - `qwen35_linear_attn_conv_decode_lowp_kernel`
  - `qwen35_linear_attn_conv_prefill_kernel`
  - `qwen35_linear_attn_conv_prefill_state_kernel`
  - `qwen35_linear_attn_conv_prefill_segments_kernel`
  - `qwen35_linear_attn_conv_prefill_segments_state_kernel`
- `linear_attn/gdn.hip` (8):
  - `qwen35_gdn_recurrent_rmsnorm_gate_lowp_kernel`
  - `qwen35_gdn_prefill_recurrent_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_kernel`
  - `qwen35_gdn_prefill_recurrent_k2_segments_kernel`
  - `qwen35_linear_attn_prefill_prepare_kernel`
  - `qwen35_gdn_prefill_rmsnorm_gate_bf16_kernel`
  - `qwen35_gdn_prefill_rmsnorm_gate_fp16_kernel`
  - `qwen35_gdn_prefill_rmsnorm_gate_rotate_fp16_kernel`
- `norm/rmsnorm.hip` Qwen primitive subset (4) — **hipENGINE landed for BF16 raw-pointer wrappers**:
  - `qwen35_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_kernel`
  - `qwen35_add_rmsnorm_f32_kernel`
  - `qwen35_head_rmsnorm_kernel`
- `norm/rmsnorm.hip` PARO subset (2) — **hipENGINE landed for BF16 and FP16 raw-pointer wrappers**:
  - `paro_rmsnorm_out_kernel`
  - `paro_add_rmsnorm_out_kernel`
- `rotary/rotary.hip` Qwen primitive subset (1):
  - `qwen35_partial_rotary_kernel`
- `attention/full_attn_decode.hip` (2):
  - `qwen35_full_attn_decode_kernel`
  - `qwen35_full_attn_decode_context_tensor_kernel`
- `attention/paged_attn_decode.hip` (13):
  - `qwen35_paged_full_attn_decode_kernel`
  - `qwen35_paged_full_attn_decode_context_tensor_kernel`
  - `qwen35_paged_full_attn_decode_8k_context_tensor_kernel`
  - `qwen35_paged_full_attn_decode_4k_kernel`
  - `qwen35_paged_full_attn_decode_8k_dyn_kernel`
  - `qwen35_paged_full_attn_decode_split_k_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_warp_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_gqa_kernel`
  - `qwen35_paged_full_attn_decode_split_k_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_ctx_tensor_int8_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_kernel`
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`
- `attention/paged_kv_write.hip` (7):
  - `qwen35_write_paged_kv_kernel`
  - `qwen35_write_paged_kv_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_kernel`
  - `qwen35_write_paged_kv_mixed_value_position_tensor_kernel`
  - `qwen35_write_paged_kv_mixed_value_prompt_position_tensor_kernel`
  - `qwen35_write_paged_kv_int8_kernel`
  - `qwen35_write_paged_kv_position_tensor_int8_kernel`
- `quant/paro_awq_gemv.hip` stable PARO GEMV/projection subset (7; projection-pair fused variants are called out again below):
  - `gemv_awq_v8_kernel`
  - `gemv_awq_pack8_kernel`
  - `gemv_awq_dual_pack8_kernel`
  - `gemv_awq_selected_dual_pack8_strided_kernel`
  - `gemv_awq_selected_pack8_kernel`
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`
  - `dense_gemv_out_kernel`
- `quant/paro_awq_dequant.hip` (2):
  - `dequant_awq_pack8_kernel`
  - `dequant_awq_pack8_dual_kernel`
- `rotary/rotary.hip` PARO subset (2):
  - `paro_rotate2_kernel`
  - `paro_rotate3_kernel`

### Fused / composite kernel families (**lineage green, not yet hipENGINE-landed**)

Each fused kernel still requires an unfused fallback chain registered under its primitive components.

- Norm + rotary:
  - `qwen35_head_rmsnorm_partial_rotary_kernel`: `head_rmsnorm -> partial_rotary`.
  - `qwen35_head_rmsnorm_partial_rotary_position_kernel`: `head_rmsnorm -> position-indexed partial_rotary`.
- PARO selected-expert activation / rotation:
  - `silu_mul_dual_out_kernel`: `silu(gate) * up` for dual selected-expert outputs over packed `[rows, 2*features]` input.
  - `silu_mul_separate_out_kernel`: `silu(gate) * up` where gate and up live in separate `[rows, features]` buffers; used by the W4 PARO dense shared expert path.
  - `silu_mul_dual_rotate_out_kernel`: `silu(gate) * up -> PARO down-rotation`.
  - `silu_mul_pair_rotate_out_kernel`: paired `silu(gate) * up -> rotate` variant.
- Weighted routing reductions:
  - `weighted_index_add_out_kernel`: routed weighted add into output rows.
  - `weighted_index_add_atomic_float_out_kernel`: atomic-float routed weighted add variant.
  - `weighted_lanes_inverse_kernel`: lane/weight inverse helper.
  - `weighted_lanes_sum_out_kernel`: lane-group weighted sum.
  - `weighted_sum_out_kernel`: selected-expert weighted sum.
- Shared-expert + selected-expert combine:
  - `shared_gate_combine_out_kernel`: `selected_moe + sigmoid(shared_gate) * shared_expert`.
  - `shared_gate_combine_residual_out_kernel`: above plus residual add.
  - `weighted_sum_shared_gate_combine_residual_out_kernel`: selected weighted sum + shared gate combine + residual add in one c=1 decode kernel.
- Full-attention gate fusion:
  - `full_attn_gate_mul_out_kernel`: `sigmoid(attn_gate) * attention_out` plus output conversion.
  - `qwen35_paged_full_attn_decode_split_k_reduce_gate_kernel`: paged split-K reduce fused with PARO full-attention gate for device-context decode.
- Projection-pair fusion routes used by the PARO path:
  - `gemv_awq_dual_pack8_kernel`: dual W4 pack8 GEMV for two projections over the same input.
  - `gemv_awq_dual_pack8_transposed_rotate_staged_kernel`: opt-in/default-off decode diagnostic that stages two input rotations once, then runs dual transposed W4 pack8 GEMV after a device barrier.
  - `gemv_awq_selected_dual_pack8_strided_kernel`: selected-expert dual W4 pack8 GEMV over compact/repacked expert weights.
  - `gemv_awq_selected_dual_pack8_strided_rotate_out_kernel`: selected-expert dual W4 pack8 GEMV plus output rotation.

### Source catalog drift requiring refresh before PARO/WMMA ports

The last manual hipENGINE catalog audit (`docs/source_lineage.json` baseline `22405a9`) counted the committed PARO embedded-HIP set at 25 kernels and observed six additional parent-worktree kernels beyond that committed set:

- `gemv_awq_mbatch_dual_pack8_kernel`
- `gemv_awq_mbatch_pack8_kernel`
- `gemv_awq_expert_seq_dual_pack8_kernel`
- `gemv_awq_expert_seq_pack8_kernel`
- `gemm_awq_selected_dual_pack8_wmma_kernel`
- `gemm_awq_selected_pack8_wmma_kernel`

`~/amd-gpu-tuning/docs/OPTIMAL.md` now promotes a compact-WMMA route, and `scripts/check_lineage.py` reports drift in `qwen35_expert.hip`, `extension.cpp`, `paroquant_kernels.py`, `paroquant.py`, and `expert.py` after `22405a9`. Therefore, treat the 120-kernel catalog above as the **baseline catalog**, not the final PARO/WMMA port inventory.

Current OPTIMAL source refresh at `nano-vllm-amd@59195ed` adds **5 kernels** over the baseline catalog: `qwen35_moe_wmma_tile_map_kernel` in `qwen35_expert.hip`, plus `gemm_awq_selected_dual_pack8_wmma_kernel`, `gemm_awq_selected_pack8_wmma_kernel`, `gemm_awq_selected_dual_pack8_wmma_compact_kernel`, and `gemm_awq_selected_pack8_wmma_compact_kernel` in `paroquant_kernels.py`. That refresh's full Qwen/PARO HIP inventory is **96** monolithic kernels + **29** PARO/WMMA kernels = **125** kernels, excluding `smoke_add`. Additional parent drift observed at `nano-vllm-amd@b95eaa5` adds five tree/speculative linear-attention kernels (`qwen35_linear_attn_tree_conv_decode_lowp*`, `qwen35_gdn_tree_recurrent_rmsnorm_gate_lowp*`, and `qwen35_gdn_tree_rmsnorm_gate_finalize_kernel`), bringing the observed parent inventory to **101** monolithic kernels + **29** PARO/WMMA kernels = **130** kernels. The dense-projection fused W4 prefill source is tracked separately in `paroquant_fusedw4.py`; hipENGINE has ported its FP16 raw-pointer WMMA kernel for transposed pack8 prompt projections and added a strided-layout instantiation for existing non-transposed prompt weights. Those tree kernels are not the compact prompt-slab `cu_seqlens` ABI and were not ported for the segment-prefill task. Before porting PARO/WMMA or tree kernels, read the listed WORKLOG/OPTIMAL evidence and keep this checklist synchronized with the source commit used.

## Qwen3.5 MoE / PARO path map

This section maps the current source-lineage inference path that hipENGINE should preserve when porting `z-lab/Qwen3.5-35B-A3B-PARO` (`w4_paro`, W4A16) from `nano-vllm-amd`. It is **not** an hipENGINE performance claim yet; it is the target graph/kernel route to reproduce after the port.

### Current optimal route

Canonical source: `~/amd-gpu-tuning/docs/OPTIMAL.md` (2026-05-13 snapshot). Supporting design/history remains in `~/amd-gpu-tuning/PLAN-PAROQUANT.md` and `~/amd-gpu-tuning/docs/PARO.md`.

The current optimal parent route is compact-WMMA prefill plus one-step graph-replay decode, with all listed parent quality gates passing. Latest retained parent sweep:

| Shape | PARO prefill tok/s | PARO decode tok/s | Peak VRAM | Validation |
| --- | ---: | ---: | ---: | --- |
| 512/128 | 2557 | 115.7 | 18.86 GiB | graph/step true |
| 1K/128 | 2876 | 112.9 | 19.34 GiB | graph/step true |
| 4K/128 | 2703 | 112.0 | 21.64 GiB | graph/step true |
| 32K/128 | 1880 | 98.8 | 21.37 GiB | graph/step true |
| 128K/128 | 914 | 62.6 | 27.42 GiB | graph/step true |

Post-sweep parent spot checks retained these defaults:

- Native weighted-lane grouped-stacked accumulation: `2642.1` vs `2561.5` prefill tok/s at 512/128, graph validation true.
- Grouped SiLU + down-rotation fusion: `2632.2` vs `2631.4` prefill tok/s at 512/128, graph validation true.
- WMMA extension load is not the Vulkan decode-gap source: graph decode was `115.56` vs `115.04` tok/s with WMMA disabled.

Correctness hierarchy for these rows: HF PARO oracle for model correctness; scalar eager pure-native as the native debug reference; tensorized eager as serving/graph ABI reference; graph replay must match tensorized eager. Long scalar-vs-tensorized greedy equality is a diagnostic, not the only promotion gate; use KL/NLL/top-k/top-1 and repetition/coherence/long-context quality gates.

### Base flags to preserve

`OPTIMAL.md` lists 23 base environment flags. hipENGINE should preserve the same routing decisions as registry/plugin configuration rather than copying env-var checks into engine code:

- **MoE dispatch:** compact stacked layout, in-place selected-MoE repack replacement, GPU expert gather, grouped-stacked max tokens `4096`, native weighted lanes, grouped-stacked SiLU+rotate fusion, decode selected-MoE SiLU/down-rotate fusion, native router.
- **GEMV / WMMA:** PARO vec8 GEMV, pack8 qweight replacement, transposed pack8 disabled on W7900, WMMA GEMM enabled for prefill MoE, compact WMMA buffers, parent `WMMA_MIN_TOKENS=64` (crossover vs GEMV ~48 tokens); hipENGINE P1.4 retains compact WMMA for all multi-token single-request prefill (`HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS=2`) after GEMV fallback lost at 128/256/512/4096 prompts.
- **Attention:** full-attention gate fusion, full-attention Q/K pack8 fusion, grouped-GQA paged context attention, paged max splits `512`.
- **Linear/projections:** W8A16 `lm_head`, W8A16 shared expert dense branch, fused linear-attention A/B projection, pack8 fused linear-attention QKV+Z projection.
- **Routing threshold:** native router prefill path begins at `512` tokens.

### Current OPTIMAL MoE port checklist (`nano-vllm-amd@59195ed`)

The checklist below is the active port map for reproducing the parent compact-WMMA + graph-replay route. Status values are hipENGINE status, not parent status.

#### Source refresh deltas since baseline `22405a9`

| Source | Current status | Required action |
| --- | --- | --- |
| `csrc/amd/qwen35_expert.hip` | DRIFT; 96 kernels | Include new `qwen35_moe_wmma_tile_map_kernel` with grouped MoE / compact WMMA port. |
| `csrc/amd/extension.cpp` | DRIFT; + bindings for tile-map path | Retype affected launch wrapper(s), do not copy PyTorch/TORCH_LIBRARY plumbing. |
| `nanovllm/native/qwen35/paroquant_kernels.py` | DRIFT; 29 kernels, 35 `m.def` exports | Extract current V8 + WMMA embedded HIP, including four WMMA kernels and compact wrappers. |
| `nanovllm/native/qwen35/paroquant.py` | DRIFT; dispatch logic changed | Adapt routing decisions into model/quant/kernel-plan plugins, not env-var branches in engine code. |
| `nanovllm/native/qwen35/expert.py` | DRIFT; added `hip_qwen35_moe_wmma_tile_map` | Port tile-map raw-pointer wrapper with grouped MoE metadata family. |

#### MoE decode c=1 path

| Stage | Parent kernels / wrappers | hipENGINE status | Notes / gate |
| --- | --- | --- | --- |
| RMSNorm / residual | `paro_rmsnorm_out_kernel`, `paro_add_rmsnorm_out_kernel`; Qwen BF16 `qwen35_*rmsnorm*` family | **Landed for BF16 and FP16 PARO raw-pointer wrappers** | PARO out-kernels multiply direct norm weights and now cover the parent FP16 activation path; Qwen kernels use `1.0 + weight_delta`. |
| Router + shared gate | `qwen35_router_logits_kernel`, `qwen35_router_select_kernel`, `hip_qwen35_router_topk_shared_out` | **Landed for BF16 and FP16 hidden raw-pointer shared-out routes; cooperative decode fold is opt-in/rejected** | Current wrappers write logits/selected/routing buffers and shared-gate logits with BF16 router weights; FP16 hidden specialization covers parent-mixed activation materialization. `HIPENGINE_PARO_ROUTER_TOPK_COOP=1` runs a diagnostic atomic last-producer fold that preserves the one-block-per-row logits grid but is not default after D1.5 regressed graph replay. |
| Selected gate/up GEMV | `gemv_awq_selected_dual_pack8_strided_kernel`, `gemv_awq_selected_dual_pack8_kernel`, optional rotate-out variant | **Landed for BF16 and FP16 raw-pointer strided/transposed dual pack8 wrappers plus fused rotate-out** | Decode path uses stacked/repacked selected-expert W4 pack8 qweights. Preserve small-K safety fix from `59195ed`. |
| Activation + down rotation | `silu_mul_dual_rotate_out_kernel` (fallback `silu_mul_dual_out_kernel` + rotate) | **Landed for BF16 and FP16 raw-pointer fused and fallback wrappers** | Default `NANOVLLM_PARO_MOE_SILU_DOWN_ROTATE_FUSED=1`; fused dual rotate plus dual/pair fallback kernels are registered for parent-mixed activations. |
| Selected down GEMV | `gemv_awq_selected_pack8_kernel` / strided wrapper | **Landed for BF16 and FP16 raw-pointer strided/transposed pack8 wrappers** | Used for selected down projection; small-K specialization applies where safe. |
| Shared expert | W8A16 shared gate/up/down (`w8a16_*shared*`, `w8a16_single_*`, `w8a16_linear*`) | **Landed for current parent lowp-linear route, including FP16 lowp wrapper, multi-token FP16 shared gate/up+SiLU, and grouped-prefill shared down+combine helper** | `w8a16-shared-expert-hip` validates W8A16 gate/up → `silu_mul_dual_out` → W8A16 down (`gate_up_mismatch=0`, `intermediate_mismatch=0`, `out_mismatch=0`); fused FP16 shared gate/up+SiLU and shared down+combine are covered by all-layer fixture gate (`max_kl=0.03406`, top-1 `1.0`) and the c=1/non-grouped fallbacks remain registered. |
| Weighted combine + residual | `weighted_sum_shared_gate_combine_residual_out_kernel`; fallback `weighted_sum_out_kernel`, `shared_gate_combine*` | **Landed for BF16 and FP16 values with FP32 weights/gate logits** | c=1 decode promoted path fuses selected sum, shared sigmoid/gate combine, and residual add; scalar-weight fallback remains unported. |
| Synthetic c=1 vertical smoke | RMSNorm → router → selected W4 gate/up/down → W8A16 shared → weighted/shared/residual combine | **Landed** | `paro-moe-c1-hip --hidden-size 8`: direct wrapper chain bit-exact; `paro-moe-c1-state-hip --hidden-size 8`: decode-state path bit-exact (`final_mismatch=0`) and uses normalized prepared weights + `RuntimeWorkspace`; full model path still needs tokenizer/model loop/attention plumbing. |

#### MoE prefill compact-WMMA path

| Stage | Parent kernels / wrappers | hipENGINE status | Notes / gate |
| --- | --- | --- | --- |
| Lane grouping | `qwen35_moe_group_count_kernel`, `qwen35_moe_group_prefix_kernel`, `qwen35_moe_group_scatter[_gather]_kernel`, `qwen35_moe_gather_packed_hidden_kernel` | **Landed for metadata + lowp packed-hidden gather** | `qwen35-moe-group-scatter-hip` validates count/prefix/scatter_gather/gather; expert GEMM/WMMA still required before retained MoE prefill. |
| Compact WMMA tile map | `qwen35_moe_wmma_tile_map_kernel` | **Landed** | Maps compact expert starts to WMMA tiles without pad-multiple=16 overhead. |
| Gate/up compact WMMA | `gemm_awq_selected_dual_pack8_wmma_compact_kernel` | **Landed for BF16 and FP16** | Current grouped prefill route calls compact WMMA over packed/sorted lanes; noncompact WMMA and GEMV-only remain fallback/comparison paths. |
| Activation + down rotation | `silu_mul_dual_rotate_out_kernel` | **Landed / reused for grouped packed lanes** | `NANOVLLM_PARO_MOE_GROUPED_STACKED_SILU_ROTATE_FUSED=1` default; current grouped prefill calls the existing fused rotate over sorted lanes. |
| Down compact WMMA | `gemm_awq_selected_pack8_wmma_compact_kernel` | **Landed for BF16 and FP16** | Paired with compact tile map and compact buffers; `paro-awq-wmma-compact-hip` validates compact dual/single kernels on a tiny fixture. |
| Weighted lane accumulation | `weighted_lanes_sum_out_kernel` | **Landed for BF16 and FP16** | `paro-combine-hip` validates `weighted_lanes_sum_out_{bf16,fp16}` and batched shared-gate residual combine; `rocprofv3` shows the weighted-lane kernels and batch combine on W7900. |
| GEMV fallback/comparison | `gemv_awq_selected_dual_pack8*`, `gemv_awq_selected_pack8*` | **Available as fallback/comparison** | Single-request multi-token prefill layer orchestration defaults to grouped metadata + compact WMMA for `tokens >= 2` (`HIPENGINE_MOE_PREFILL_COMPACT_WMMA_MIN_TOKENS`, large value forces c1 GEMV diagnostics); P1.4 found no useful GEMV crossover at 128/256/512/4096 prompts. |

#### Full-inference dependencies outside MoE

| Area | Required for reproducing parent inference | hipENGINE status |
| --- | --- | --- |
| PARO quant plugin / weight layout | `w4_paro` plugin, pack8 replacement layout, compact stacked MoE weights, W8A16 shared/lm-head replacements | Missing; only `bf16` plugin landed. |
| Model plugin / scheduler | Qwen3.5 hybrid full-attn + linear-attn/GDN + MoE layer sequence, static decode buffers, one-step graph replay | Missing; `LLM.generate()` is still scaffolded. |
| Linear projections | `gemv_awq_pack8`, `gemv_awq_dual_pack8`, `awq_fusedw4_prefill_fp16`, `awq_fusedw4_prefill_strided_fp16`, `dense_gemv_out`, rotation helpers | **Partially landed**: c=1 GEMV wrappers plus FP16 fused-W4 WMMA prompt projection for transposed QKV/Z and strided linear out-proj; dense/W8 projection WMMA remains open. |
| Linear attention / GDN | `qwen35_linear_attn_conv_*`, `qwen35_gdn_*` incl. lowp recurrent RMSNorm gate | Landed for current Qwen3.5/PARO path: decode conv/GDN, single-request prefill conv/GDN, and segment-aware compact-slab conv/GDN state kernels are available; remaining c>N work is packed orchestration with varlen full-attn/final commit. |
| Full attention / KV | `awq_fusedw4_prefill_fp16`, `awq_fusedw4_prefill_strided_fp16`, `qwen35_head_rmsnorm_partial_rotary*`, `qwen35_write_paged_kv_mixed_value*`, paged/split-K full-attention decode family, `full_attn_gate_mul_out` | Partial: FP16 fused-W4 prompt projection for transposed Q/K and strided V/O, full-attention prelude, span-shaped paged KV append, span-shaped paged context-tensor decode, generic split-K reduce, FP32/BF16 gated split-K reduce, and Qwen3.5 GQA-specialized split-K context variants landed; remaining gaps are non-context/int8/8K legacy variants and engine allocation/plumbing. |
| Final head | W8A16 `lm_head` replacement path | Missing. |
| Eval harness | Parent baseline JSON capture + hipENGINE JSON schema-2 artifacts + KL/top-1/sample/graph validation gates | Not yet landed. |

#### Port order for the OPTIMAL exercise

1. **Measurement harness first:** run/record the parent `512/128` and `4K/128` OPTIMAL commands as source-lineage artifacts, then create a blocked hipENGINE artifact until `LLM.generate()` exists.
2. **MoE c=1 decode vertical slice:** PARO RMSNorm out-kernels → router/shared-gate → selected pack8 GEMV → fused activation/down-rotation → W8A16 shared expert → weighted shared-gate residual combine.
3. **MoE prefill compact-WMMA slice:** lane grouping/gather → compact tile map → compact dual/single WMMA → weighted-lane accumulation → GEMV fallback.
4. **Full-inference closure:** weight loader/model plugin, non-MoE projections, linear attention/GDN, full attention/KV, final head, graph replay, then end-to-end correctness/perf comparison.

### Prefill route

- Benchmark protocol: `OPTIMAL.md` uses the parent `scripts/run_moe2_baselines.py` sweep and graph-replay bench command; short/mid-context quick start targets `--prompt-len 4096 --decode-len 128 --decode-use-step-graph-replay`.
- Router/MoE:
  - Real router runs per MoE layer; no HF model execution in the pure-native path.
  - Compact WMMA prefill MoE is the current optimal grouped-stacked route for `>=64` tokens; GEMV-only only wins at ~32 tokens.
  - Grouped-stacked max tokens is now `4096`, not the older 1024 short-prefill cap.
  - Weighted-lane accumulation and grouped-stacked SiLU+down-rotation fusion are default-on parent optimizations.
- Long prefill:
  - For `>=32K`, add chunking overrides from `OPTIMAL.md`: linear chunk `1024`, MoE chunk `1024`, full-attention post/RoPE chunk `1024`, and full-attention query chunk `4096`.
  - Do **not** set long-prefill chunking overrides for `<=4K`; they change the MoE prefill path and reduce throughput.
- Projection/quant:
  - Non-expert W4 pack8 replacement uses `[out/8, in]` pack8 qweights and frees original eligible AWQ qweights.
  - `lm_head` uses the W8A16 replacement path in the optimal route.

### Decode route

The target c=1 decode path is static-buffer, graph-replay-friendly, and mostly device-resident:

1. **RMSNorm / residual:** PARO-native `rmsnorm` and `add+rmsnorm` kernels; avoid per-token framework glue.
2. **Router:** native combined router/shared-gate logits with hot BF16 cache and FP16/BF16 hidden input; reuse decode-only output buffers.
3. **Selected MoE:** compact stacked selected-expert layout plus repacked replacement qweights; selected gate/up via dual W4 pack8 GEMV; selected down uses small-K specialization where applicable.
4. **Selected activation/down rotation:** `silu(gate) * up` and PARO down-rotation fused on the stacked decode path.
5. **Shared expert:** dense shared expert c=1 branch uses W8A16 gate/up/down where enabled; grouped multi-token FP16 prefill fuses shared gate/up + SiLU into the four-column bulk helper (or token-tiled helper for legacy prompts `>=1024`) and fuses shared down with selected/shared-gate/residual combine (token-tiled for legacy prompts `>=2`).
6. **MoE combine:** selected-expert weighted sum, shared-expert sigmoid/gate combine, and residual add fuse into `weighted_sum_shared_gate_combine_residual_out_kernel` on c=1 decode.
7. **Linear attention:** native conv/GDN recurrence; lowp FP16/BF16 inputs feed kernels while recurrent state/math stay FP32. A/B projections are concatenated for c=1; QKV/Z and out-proj use fused W4→WMMA for multi-token FP16 prompt projection after rotation, with pack8 W4 GEMV retained for c=1/fallback.
8. **Full attention projections:** q/k use fused W4→WMMA for multi-token FP16 prompt projection after batched input rotation; v/o also use the strided fused-W4 prefill instantiation, and c=1 uses the dual/single pack8 W4 GEMV path.
9. **KV append:** BF16 full-attention KV cache with native mixed-input paged-KV writer; no tiny per-token framework appends.
10. **Full attention decode:** contiguous path for short contexts; paged/split-K path defaults at context `>= 1024`, with warp-cooperative context tensor QK, physical-offset address hoist, grouped-GQA reuse, split cap 512 for 128K-class rows, and gated split-K reduce where applicable.
11. **Final head:** W8A16 `lm_head` replacement path.
12. **Graph replay:** one reusable decode-step graph replay is the promoted graph shape; keep `--decode-step-graph-capture-steps=1`. Multi-step capture was tested and not promoted.

Parent decode profiling note from `OPTIMAL.md`: fused `lm_head + argmax` is not a current lever; the next decode target is the AWQ/GEMV decode family, about 40% of selected-region kernel time in the 512/128 graph profile.

### Alternative paths and caveats

- **W8A8 comparison path:** stays quality-safe and useful as a comparator; do not regress it while porting PARO.
- **40GB+ diagnostic PARO path:** stacked selected-expert diagnostics proved speed hypotheses but are not promotion candidates because 24GB W4 usability is a hard gate.
- **24GB non-stacked baseline:** green but slow; retained as a deployable-memory fallback, not the speed target.
- **Long-context decode:** contiguous full-attention decode cannot launch at 32K because dynamic LDS scales with context; long decode must use paged/split-K over the dense cache viewed as pages.
- **Tensorized paged-attention drift:** current parent docs localize long-tail scalar-vs-tensorized drift to paged context-tensor full attention. Graph replay matching tensorized eager is necessary; scalar-eager greedy equality alone is not sufficient promotion evidence.
- **Rejected standalone kernel ideas:** PARO v8 unroll-threshold 600, isolated wave32/no-LDS W4 GEMV, naive AWQ W4xQ8 dp4a, caller-owned paged workspace, and non-split-K 4K attention were tested but not promoted. Do not import them into hipENGINE defaults without a fresh audit and correctness/perf evidence.

## Port = copy + partition + retype

The initial port is mechanical, not creative. Kernel bodies are preserved byte-for-byte (modulo `#include` headers). The three things that change during port:

1. **File split by family.** The monolithic `nano-vllm-amd/csrc/amd/qwen35_expert.hip` (13,769 lines, 95 `__global__`s) and the 3,766-line embedded HIP string in `nano-vllm-amd/nanovllm/native/qwen35/paroquant_kernels.py` partition into `kernels/<backend>/<family>/*.hip` per the table in `docs/PLAN.md`. The near-duplicate `qwen35_expert_hip.hip` is dropped.
2. **Launch wrappers retyped.** Host-side wrappers go from `torch::Tensor` to raw pointer + shape/stride/dtype signatures. Scripted, ~1 day.
3. **Embedded HIP extracted.** `paroquant_kernels.py`'s `r'''...'''` block becomes real `.hip` files compiled through `hipengine.core.build` instead of `torch.utils.cpp_extension.load_inline`.

Preserve all `__launch_bounds__`, template specializations, and compiler flags (`-mllvm -amdgpu-unroll-threshold-local=600` for decode/prefill, plus `-mcumode` for decode). A port that rewrites kernel bodies is not a port.

## Port correctness gate (non-negotiable)

A kernel split / port may only land when all three of these pass on the stated fixture set:

1. **Registry resolution.** Every kernel name still resolves via the 4-axis registry (`resolve(KernelKey(...))` returns a callable for every key previously exported by the monolithic `.so`).
2. **Profiler parity.** `rocprofv3 --kernel-trace` on the target decode smoke (Qwen3.6-35B-A3B unless noted) reports the same kernel set with matching `DurationNs` distribution as the monolithic build. A new kernel name, a missing kernel name, or a >10% duration shift is a split bug.
3. **Numerical parity.** KL ≤ 0.05 AND top-1 agreement ≥ 90% vs the monolithic build on the correctness fixtures. (For a *net-new* kernel, the oracle is `kernels/cpu_reference/`, not the monolithic build.)

Never land a split that regresses any of these.

## Build layer (`hipengine.core.build`)

hipENGINE uses its own build layer, not `torch.utils.cpp_extension`. It calls `hipcc` (or `nvcc` for CUDA backends) via `subprocess.run`, links with `ctypes.CDLL`, and caches `.so` files by a hash of `(source, flags, hipcc version)` under `~/.cache/hipengine/build/`. Edit → bench loop stays at ~5–10 s per kernel change.

### Three build profiles (from `nano-vllm-amd/nanovllm/native/amd/extension.py`)

| Profile | Flags | Wavefront | Used for |
| --- | --- | --- | --- |
| `decode` | `-mllvm`, `-amdgpu-unroll-threshold-local=600`, `-mcumode` | 32 | Decode-phase kernels (paged attention, W8A8 grouped MoE decode, PARO GEMV). `-mcumode` is not `-mwavefrontsize64`. |
| `prefill` | `-mllvm`, `-amdgpu-unroll-threshold-local=600` (WGP mode) | 32 | Prefill-phase kernels (GEMM, W8A16 linear prefill) |
| `baseline` | (none) | 32 | Debug / fallback |

Write device code for wave32 by default on gfx1100. Use `warpSize` for probes and dispatch metadata, but do not assume a 64-thread block is one wave. For block-wide reductions over more than 32 lanes, reduce within 32-lane waves with shuffles and exchange partials through LDS/shared memory.

### Wave32 default; wave64 experiments only

For nano-vllm-amd lineage kernels on W7900/gfx1100:

- Default to **wave32**. Current HIP build flags do not include `-mwavefrontsize64`, and
  parent probes showed `-mcumode` does not change `warpSize` by itself.
- RDNA3 wave64 is architecturally supported, but the hardware still issues through
  32-lane halves. RDNA3 can co-issue eligible wave64 halves on the dual-issue VALU path,
  while wave32 exposes VOPD pairing directly to the compiler. These scheduling features
  are orthogonal to the wavefront-size flag.
- Prefer wave32 + ILP: multiple independent accumulators, unrolled loops, fewer long
  dependent VALU chains, and low enough VGPR/scratch/LDS pressure to preserve occupancy.
- Prefer wave32-compatible collectives: `__shfl_down` within 32 lanes, then LDS/shared
  memory exchange for cross-wave reductions. Do not remove barriers on the theory that a
  64-thread block is a single wave.
- Only pursue wave64 as an isolated experiment with explicit `-mwavefrontsize64` build
  flags, `warpSize`/shuffle probes, correctness fixtures, ISA checks, and E2E benchmarks.
  There is no retained wave64 default in hipENGINE.

### JIT cache gotcha

Symptom: kernel calls hang with GPU at 0% utilization and no error. This is almost always a stale cached `.so` that doesn't match the current source. Nuke the matching cache dir before re-importing:

```bash
rm -rf ~/.cache/hipengine/build/<family>-<hash>*
```

If the family is unknown, clearing the whole cache is cheap (~5 s per kernel to rebuild):

```bash
rm -rf ~/.cache/hipengine/build/
```

The hash incorporates the source file content, the flag set, and the `hipcc --version` string. If you change `hipcc` underneath an existing cache, the hash will change and old entries will be ignored — not overwritten. Prune manually when the cache grows unbounded.

## rocprofv3 smoke (port parity + new kernel check)

Minimum smoke a port or a new kernel must produce:

```bash
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  uv run python scripts/smoke.py <model> <workload>
```

If the target path uses `hipengine.core.build` JIT from Python, prebuild outside the profiler and make the profiled process cache-only. `rocprofv3` launch mode preloads into child processes; letting a profiled Python process spawn `hipcc`/clang can hang or abort in LLVM initialization.

```bash
hipcc --version > /tmp/hipengine-hipcc-version.txt
python3 - <<'PY'
from pathlib import Path
from hipengine.kernels.hip_gfx1100.smoke import build_smoke_add
version = Path('/tmp/hipengine-hipcc-version.txt').read_text()
print(build_smoke_add(load=False, compiler_version=version).output_path)
PY
rocprofv3 --kernel-trace --output-format csv -d /tmp/hipengine-smoke -- \
  python3 scripts/smoke.py --mode smoke-add-hip --n 1024 \
    --compiler-version-file /tmp/hipengine-hipcc-version.txt \
    --require-cached-build
```

Expected output: a CSV with a kernel-name column (`Kernel_Name` / `KernelName`), grid/workgroup columns, `VGPR_Count`, `Scratch_Size`, and `LDS_Block_Size`. Some ROCm 7.13 traces emit `Start_Timestamp` + `End_Timestamp` instead of `DurationNs`; compute `DurationNs = End_Timestamp - Start_Timestamp` for summaries. Check:

- The expected kernel name appears.
- Duration is plausible (same order of magnitude as the reference).
- `Scratch_Size > 0` on a hot-path kernel is a red flag — escalate to `~/amd-gpu-tuning/` for audit.
- `VGPR_Count ≥ 96` may be squeezing occupancy — same.

rocprofv3 dumps are **not committed**. Store under `/tmp/` or outside the repo.

## Registering a kernel

Kernels self-register on module import:

```python
# hipengine/kernels/hip_gfx1100/attention/paged_attn_decode.py
from hipengine.kernels.registry import KernelKey, register
from hipengine.core.build import build_hip

_so = build_hip(
    sources=["paged_attn_decode.hip"],
    profile="decode",
    family="attention",
)

def paged_attn_decode_fp16(...): ...

register(
    KernelKey(backend="hip_gfx1100", layer="paged_attn_decode",
              quant="fp16", variant="split_k_warp"),
    paged_attn_decode_fp16,
)
```

The resolver does narrowest-to-broadest match: `variant` → no-variant → `quant="fp16"` fallback → `backend="cpu_reference"`. A new backend implementation or a new fused composite is a `register(...)` call, never an `if backend == "..."` branch in dispatch code.

## Per-family port checklist

When bringing up a family (`attention/`, `moe/`, `quant/`, …), follow in order:

1. Copy the relevant kernels from the monolithic source into `kernels/hip_gfx1100/<family>/*.hip`. Preserve bodies byte-for-byte.
2. Retype the host-side launch wrappers.
3. Move the `PYBIND11_MODULE` / `TORCH_LIBRARY` entries for this family from `csrc/amd/extension.cpp` into `kernels/hip_gfx1100/common/extension.cpp` (the aggregator).
4. Write `register(KernelKey(...), ...)` calls in the Python wrapper module so the kernels resolve.
5. Add a CPU-reference implementation for every new `layer` key in `kernels/cpu_reference/`.
6. Run the port correctness gate (all three checks above).
7. Commit the family as one logical unit with `port:` prefix and `nano-vllm-amd@<sha>` in the body.

Do not interleave families in one commit. A commit that touches `attention/` and `moe/` together is harder to bisect and harder to review.
