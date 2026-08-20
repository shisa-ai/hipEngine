"""PN3-MOESELECT decisive dispatch-fast-path A/B (counter-rotated).

Fast-paths the three selected-expert launch wrappers for the exact default
c1 branches (q4_k_t16_v1 pair/pair_silu, single via registry), caching the
resolved kernel fn + tile pointers and skipping the per-call env-flag reads /
allocation lookups / branch chain. Reproduces the same kernel launch (same
fn + same args) so trajectories are byte-identical. If the wall does NOT drop,
the slice is GPU-bound and the host dispatch is hidden (PN4/PN6-t16 pattern).
"""
import os, sys, time, statistics, ctypes
sys.path.insert(0, '/home/lhl/hipEngine')
import hipengine.runtime.qwen35_gguf_runner as rm
ctypes.CDLL('libamdhip64.so')

_FAST = {}

def _fast_pair_silu(orig):
    def w(weight_a, weight_b, x_ptr, selected_ptr, out_ptr, *, x_rows, rows,
          num_experts, in_features, out_features, stream, runtime,
          allow_legacy=True):
        if not (weight_a.spec.quant_key == 'gguf_q4_k_t16_v1'
                and weight_b.spec.quant_key == 'gguf_q4_k_t16_v1'):
            return orig(weight_a, weight_b, x_ptr, selected_ptr, out_ptr,
                        x_rows=x_rows, rows=rows, num_experts=num_experts,
                        in_features=in_features, out_features=out_features,
                        stream=stream, runtime=runtime, allow_legacy=allow_legacy)
        key = (id(weight_a), id(weight_b), x_rows, rows, num_experts,
               in_features, out_features)
        hit = _FAST.get(('ps', key))
        if hit is None:
            fn = rm.gguf_q4_k_t16_selected_dual_silu_gemv_bf16_bf16_out
            pa = weight_a.allocation('tiles').tensor.ptr
            pb = weight_b.allocation('tiles').tensor.ptr
            hit = (fn, pa, pb)
            _FAST[('ps', key)] = hit
        fn, pa, pb = hit
        fn(x_ptr, selected_ptr, pa, pb, out_ptr, x_rows, rows, num_experts,
           in_features, out_features, stream=stream, runtime=runtime)
        return True
    return w

def _fast_pair(orig):
    def w(weight_a, weight_b, x_ptr, selected_ptr, out_a_ptr, out_b_ptr, *,
          x_rows, rows, num_experts, in_features, out_features,
          q8_1_workspace_ptr=None, x_f32_ptr=None, stream, runtime,
          stage_timings=None, sync_stage_timings=False, stage_prefix=None):
        if not (weight_a.spec.quant_key == 'gguf_q4_k_t16_v1'
                and weight_b.spec.quant_key == 'gguf_q4_k_t16_v1'
                and q8_1_workspace_ptr is None):
            return orig(weight_a, weight_b, x_ptr, selected_ptr, out_a_ptr,
                        out_b_ptr, x_rows=x_rows, rows=rows,
                        num_experts=num_experts, in_features=in_features,
                        out_features=out_features,
                        q8_1_workspace_ptr=q8_1_workspace_ptr,
                        x_f32_ptr=x_f32_ptr, stream=stream, runtime=runtime,
                        stage_timings=stage_timings,
                        sync_stage_timings=sync_stage_timings,
                        stage_prefix=stage_prefix)
        key = (id(weight_a), id(weight_b), x_rows, rows, num_experts,
               in_features, out_features)
        hit = _FAST.get(('p', key))
        if hit is None:
            fn = rm.gguf_q4_k_t16_selected_dual_gemv_bf16_bf16_out
            pa = weight_a.allocation('tiles').tensor.ptr
            pb = weight_b.allocation('tiles').tensor.ptr
            hit = (fn, pa, pb)
            _FAST[('p', key)] = hit
        fn, pa, pb = hit
        fn(x_ptr, selected_ptr, pa, pb, out_a_ptr, out_b_ptr, x_rows, rows,
           num_experts, in_features, out_features, stream=stream, runtime=runtime)
        return True
    return w

def _fast_linear(orig):
    def w(weight, x_ptr, selected_ptr, out_ptr, *, x_rows, rows, num_experts,
          in_features, out_features, q8_1_workspace_ptr=None, x_f32_ptr=None,
          prefer_f32_out=False, backend=None, stream, runtime,
          stage_timings=None, sync_stage_timings=False, stage_prefix=None,
          allow_legacy=True):
        if q8_1_workspace_ptr is not None or prefer_f32_out:
            return orig(weight, x_ptr, selected_ptr, out_ptr, x_rows=x_rows,
                        rows=rows, num_experts=num_experts,
                        in_features=in_features, out_features=out_features,
                        q8_1_workspace_ptr=q8_1_workspace_ptr,
                        x_f32_ptr=x_f32_ptr, prefer_f32_out=prefer_f32_out,
                        backend=backend, stream=stream, runtime=runtime,
                        stage_timings=stage_timings,
                        sync_stage_timings=sync_stage_timings,
                        stage_prefix=stage_prefix)
        key = (id(weight), x_rows, rows, num_experts, in_features, out_features)
        hit = _FAST.get(('l', key))
        if hit is None:
            quant_key = weight.spec.quant_key
            fn = rm._resolve_exact_selected_moe_kernel(
                quant_key, rm._SELECTED_MOE_SINGLE_VARIANT)
            if fn is None:
                return orig(weight, x_ptr, selected_ptr, out_ptr, x_rows=x_rows,
                            rows=rows, num_experts=num_experts,
                            in_features=in_features, out_features=out_features,
                            q8_1_workspace_ptr=q8_1_workspace_ptr,
                            x_f32_ptr=x_f32_ptr, prefer_f32_out=prefer_f32_out,
                            backend=backend, stream=stream, runtime=runtime,
                            stage_timings=stage_timings,
                            sync_stage_timings=sync_stage_timings,
                            stage_prefix=stage_prefix)
            pp = weight.allocation(rm._selected_gemv_allocation_name(weight)).tensor.ptr
            hit = (fn, pp)
            _FAST[('l', key)] = hit
        fn, pp = hit
        fn(x_ptr, selected_ptr, pp, out_ptr, x_rows=x_rows, rows=rows,
           num_experts=num_experts, in_features=in_features,
           out_features=out_features, stream=stream, runtime=runtime)
        return True
    return w

def run(s, n=30, warmup=6):
    tok = 9707
    for _ in range(warmup):
        tok = int(s.step(tok, return_logits=False).token_id)
    walls = []
    for _ in range(n):
        t0 = time.perf_counter()
        tok = int(s.step(tok, return_logits=False).token_id)
        s.runtime.device_synchronize()
        walls.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(walls)

SEL = ("_launch_selected_raw_gguf_moe_pair_silu",
       "_launch_selected_raw_gguf_moe_pair",
       "_launch_selected_raw_gguf_moe_linear")
orig = {n: getattr(rm, n) for n in SEL}
def install(mode):
    _FAST.clear()
    if mode == 'fast':
        setattr(rm, SEL[0], _fast_pair_silu(orig[SEL[0]]))
        setattr(rm, SEL[1], _fast_pair(orig[SEL[1]]))
        setattr(rm, SEL[2], _fast_linear(orig[SEL[2]]))
    else:
        for n in SEL:
            setattr(rm, n, orig[n])

with rm.Qwen35GGUFResidentSession('/models/gguf/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf',
        max_sequence_length=900, backend='hip_gfx1151') as s:
    s.prefill([9707]*512, use_bulk=True, return_logits=False)
    s.runtime.device_synchronize()
    install('old');   w_old1 = run(s)
    install('fast');  w_fast = run(s)
    install('old');   w_old2 = run(s)
    install('fast');  w_fast2 = run(s)
    print(f"old (dispatch) #1: {w_old1:.2f} ms/tok")
    print(f"fast (memoized)   : {w_fast:.2f} ms/tok")
    print(f"old (dispatch) #2: {w_old2:.2f} ms/tok")
    print(f"fast (memoized) #2: {w_fast2:.2f} ms/tok")
    mean_old = (w_old1 + w_old2) / 2
    mean_fast = (w_fast + w_fast2) / 2
    print(f"mean old={mean_old:.2f} mean fast={mean_fast:.2f} delta={(mean_old-mean_fast)*1000:+.0f} us/tok ({(mean_old/mean_fast-1)*100:+.1f}%)")
    for n in SEL:
        setattr(rm, n, orig[n])
