"""Microbench + numpy oracle for the GDN chain-recurrent decode kernel
(``qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop`` + finalize), the biggest
verify-cycle kernel family (14.1 ms/pass, ~85 us/call at the deployed dims).

Times the fp16 recurrence+finalize via HIP-graph replay (launch-overhead-free
GPU time, same methodology as the PARO FFN microbench) and validates ``out`` +
``leaf_recurrent_state`` against a numpy delta-rule reference. Deployed verify
shape: num_v_heads=32, num_k_heads=16, head_k_dim=head_v_dim=128, max_nodes=B+1.

``performance_claim=false`` -- diagnostic A/B to drive the GDN optimize loop.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _silu(x):
    return x * _sigmoid(x)


def gdn_chain_reference(conv_out, gate, a, b, dt_bias, a_log, norm_weight, base_state,
                        num_k_heads, num_v_heads, head_k_dim, head_v_dim, eps):
    """f32 numpy mirror of the chain recurrence + rmsnorm-gate finalize. fp16
    inputs (gate/a/b) are pre-rounded to fp16 by the caller to match the kernel."""
    T = conv_out.shape[0]
    Dk, Dv = head_k_dim, head_v_dim
    key_dim = num_k_heads * head_k_dim
    repeat = num_v_heads // num_k_heads
    state = base_state.astype(np.float32).copy()  # [Hv, Dk, Dv]
    out = np.zeros((T, num_v_heads, Dv), np.float32)
    leaf = np.zeros((T, num_v_heads, Dk, Dv), np.float32)
    inv_dk = 1.0 / np.sqrt(np.float32(Dk))
    for t in range(T):
        for vh in range(num_v_heads):
            kh = vh // repeat
            q = conv_out[t, kh * Dk: kh * Dk + Dk].astype(np.float32)
            k = conv_out[t, key_dim + kh * Dk: key_dim + kh * Dk + Dk].astype(np.float32)
            val = conv_out[t, 2 * key_dim + vh * Dv: 2 * key_dim + vh * Dv + Dv].astype(np.float32)
            q_scale = (1.0 / np.sqrt(np.sum(q * q) + 1.0e-6)) * inv_dk
            k_scale = 1.0 / np.sqrt(np.sum(k * k) + 1.0e-6)
            beta = _sigmoid(np.float32(b[t, vh]))
            decay = np.exp(-np.exp(np.float32(a_log[vh])) * _softplus(np.float32(a[t, vh]) + np.float32(dt_bias[vh])))
            qn = q * q_scale
            kn = k * k_scale
            st = state[vh]
            kv_mem = kn @ (st * decay)               # [Dv]
            delta = (val - kv_mem) * beta            # [Dv]
            new_state = st * decay + np.outer(kn, delta)  # [Dk, Dv]
            acc = qn @ new_state                     # [Dv]
            state[vh] = new_state
            leaf[t, vh] = new_state
            rms = 1.0 / np.sqrt(np.mean(acc * acc) + eps)
            g = _silu(gate[t, vh * Dv: vh * Dv + Dv].astype(np.float32))
            out[t, vh] = acc * rms * norm_weight * g
    return out.reshape(T, num_v_heads * Dv), leaf.reshape(T, -1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compiler-version-file", type=Path, default=None)
    ap.add_argument("--require-cached-build", action="store_true")
    ap.add_argument("--num-k-heads", type=int, default=16)
    ap.add_argument("--num-v-heads", type=int, default=32)
    ap.add_argument("--head-k-dim", type=int, default=128)
    ap.add_argument("--head-v-dim", type=int, default=128)
    ap.add_argument("--max-nodes", type=int, nargs="+", default=[4])
    ap.add_argument("--eps", type=float, default=1.0e-6)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    import os
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    from hipengine.core.hip import get_hip_runtime
    from hipengine.core.memory import copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
    from hipengine.kernels.hip_gfx1100.linear_attn.gdn import (
        build_qwen35_linear_attn_gdn,
        qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16,
    )

    library = build_qwen35_linear_attn_gdn(load=True, require_cached=args.require_cached_build)
    rt = get_hip_runtime()
    Hk, Hv, Dk, Dv = args.num_k_heads, args.num_v_heads, args.head_k_dim, args.head_v_dim
    key_dim = Hk * Dk
    value_dim = Hv * Dv
    conv_stride = 2 * key_dim + value_dim
    state_stride = Hv * Dk * Dv
    CUS = 48

    print(f"num_k_heads={Hk} num_v_heads={Hv} head_k_dim={Dk} head_v_dim={Dv}  CUs={CUS} (W7900)")
    print(f"grid = num_v_heads*(head_v_dim/VTILE) blocks (dv-tiled; <= {Hv*Dv}) x 32 threads")
    print(f"GPU time via HIP graph replay; out/leaf max_abs vs numpy f32 oracle")
    print(f"{'T':>3} {'blocks':>7} {'us/call':>9} {'out_max_abs':>12} {'leaf_max_abs':>13}")

    rng = np.random.default_rng(7)
    rows_out = []
    for T in args.max_nodes:
        store = []

        def dev(arr):
            a = np.ascontiguousarray(arr)
            b = malloc(a.nbytes)
            copy_host_to_device(b, host_array_ptr(a), a.nbytes)
            store.append((b, a))
            return b

        conv_out = (rng.standard_normal((T, conv_stride)) * 0.5).astype(np.float32)
        gate = (rng.standard_normal((T, value_dim)) * 0.5).astype(np.float16)
        a = (rng.standard_normal((T, Hv)) * 0.5).astype(np.float16)
        b = (rng.standard_normal((T, Hv)) * 0.5).astype(np.float16)
        dt_bias = (rng.standard_normal(Hv) * 0.1).astype(np.float32)
        a_log = (rng.standard_normal(Hv) * 0.1).astype(np.float32)
        norm_weight = (rng.standard_normal(Dv) * 0.1 + 1.0).astype(np.float32)
        base_state = (rng.standard_normal((Hv, Dk, Dv)) * 0.1).astype(np.float32)

        d_conv = dev(conv_out); d_gate = dev(gate); d_a = dev(a); d_b = dev(b)
        d_dt = dev(dt_bias); d_alog = dev(a_log); d_nw = dev(norm_weight); d_base = dev(base_state)
        d_leaf = malloc(T * state_stride * 4); store.append((d_leaf, None))
        d_acc = malloc(T * Hv * Dv * 4); store.append((d_acc, None))
        d_out = malloc(T * value_dim * 4); store.append((d_out, None))

        def run(stream=0):
            qwen35_gdn_chain_recurrent_rmsnorm_gate_lowp_tloop_fp16(
                d_conv.ptr, d_gate.ptr, d_a.ptr, d_b.ptr, d_dt.ptr, d_alog.ptr, d_nw.ptr,
                d_base.ptr, d_leaf.ptr, d_acc.ptr, d_out.ptr, args.eps,
                T, Hk, Hv, Dk, Dv, stream=stream, library=library)

        def graph_us(enqueue):
            cap = rt.stream_create()
            enqueue(0); rt.device_synchronize()
            rt.stream_begin_capture(cap)
            enqueue(cap)
            graph = rt.stream_end_capture(cap)
            gexec = rt.graph_instantiate(graph)
            try:
                for _ in range(args.warmup):
                    rt.graph_launch(gexec, cap)
                rt.stream_synchronize(cap)
                t0 = time.perf_counter()
                for _ in range(args.iters):
                    rt.graph_launch(gexec, cap)
                rt.stream_synchronize(cap)
                return (time.perf_counter() - t0) / args.iters * 1e6
            finally:
                rt.graph_exec_destroy(gexec); rt.graph_destroy(graph); rt.stream_destroy(cap)

        us = graph_us(run)
        run(); rt.device_synchronize()
        out_h = np.zeros((T, value_dim), np.float32)
        leaf_h = np.zeros((T, state_stride), np.float32)
        copy_device_to_host(host_array_ptr(out_h), d_out, out_h.nbytes)
        copy_device_to_host(host_array_ptr(leaf_h), d_leaf, leaf_h.nbytes)
        ref_out, ref_leaf = gdn_chain_reference(
            conv_out, gate, a, b, dt_bias, a_log, norm_weight, base_state, Hk, Hv, Dk, Dv, args.eps)
        out_max = float(np.max(np.abs(out_h - ref_out)))
        leaf_max = float(np.max(np.abs(leaf_h - ref_leaf)))
        print(f"{T:>3} {Hv*Dv:>7} {us:>9.2f} {out_max:>12.3e} {leaf_max:>13.3e}")
        rows_out.append({"max_nodes": T, "blocks": Hv * Dv, "us_per_call": us,
                         "out_max_abs": out_max, "leaf_max_abs": leaf_max})
        for bb, _ in store:
            free(bb)

    if args.json is not None:
        import json
        args.json.write_text(json.dumps(
            {"num_k_heads": Hk, "num_v_heads": Hv, "head_k_dim": Dk, "head_v_dim": Dv,
             "performance_claim": False, "sweep": rows_out}, indent=2) + "\n")


if __name__ == "__main__":
    main()
