"""Bisect GPU vs CPU Surya layer-3 (first full-attention) prefill."""

import math
import sys

import numpy as np

sys.path.insert(0, "scripts")
from surya_gpu_smoke import build_inputs  # noqa: E402

from hipengine.kernels.cpu_reference.evie import (  # noqa: E402
    apply_rope_half,
    rms_norm,
)
from hipengine.kernels.cpu_reference.surya import SuryaSpec  # noqa: E402
from hipengine.runtime.surya import SuryaGpuRunner  # noqa: E402
import dataclasses  # noqa: E402


def main():
    weights, spec, ids, pos, visual = build_inputs()
    globals()["w_"] = weights
    from hipengine.kernels.cpu_reference.evie import text_rope_tables

    cos, sin = text_rope_tables(spec, pos)
    T = ids.shape[1]
    s = spec
    nq, nk, hd = s.num_attention_heads, s.num_key_value_heads, s.head_dim

    # numpy mirror of layer 3 attention given the true layer-2 output.
    # Cheap trick: get the true input by running the CPU reference up to
    # layer 3's input via text_prefill on a 3-layer spec.
    spec3 = dataclasses.replace(spec, num_layers=3)
    from hipengine.kernels.cpu_reference.surya import text_prefill
    _, st3 = text_prefill(weights, spec3, ids, pos, visual_features=visual)
    # recompute layer-3 input hidden by re-running with the returned state:
    # simplest is to instrument: run text_prefill with a patched rms_norm?
    # Instead: recompute x after 3 layers via the same loop.
    from hipengine.kernels.cpu_reference.surya import _gdn_layer_prefill, _mlp
    emb = weights["model.language_model.embed_tokens.weight"]
    x = emb[ids].copy()
    for bi in range(x.shape[0]):
        mask = ids[bi] == spec.image_token_id
        x[bi, mask] = visual[bi]
    cpu_x_after = {}
    for layer in range(3):
        lp = f"model.language_model.layers.{layer}."
        h = rms_norm(x, weights[lp + "input_layernorm.weight"])
        attn_out, gstate = _gdn_layer_prefill(weights, spec, layer, h)
        x = x + attn_out
        x = x + _mlp(weights, lp, rms_norm(x, weights[lp + "post_attention_layernorm.weight"]))
        cpu_x_after[layer] = x.copy()
    h = rms_norm(x, weights["model.language_model.layers.3.input_layernorm.weight"])

    # CPU attention mirror (causal)
    p = "model.language_model.layers.3.self_attn."
    qp = h @ weights[p + "q_proj.weight"].astype(np.float32).T
    q, gate = np.split(qp.reshape(1, T, nq, hd * 2), 2, axis=-1)
    gate = gate.reshape(1, T, -1)
    k = (h @ weights[p + "k_proj.weight"].astype(np.float32).T).reshape(1, T, nk, hd)
    v = (h @ weights[p + "v_proj.weight"].astype(np.float32).T).reshape(1, T, nk, hd)
    qn_cpu = rms_norm(q, weights[p + "q_norm.weight"])
    q = qn_cpu
    k = rms_norm(k, weights[p + "k_norm.weight"])
    q = apply_rope_half(q, cos, sin)
    k = apply_rope_half(k, cos, sin)
    qh = q.transpose(0, 2, 1, 3)
    kh = k.transpose(0, 2, 1, 3)
    vh = v.transpose(0, 2, 1, 3)
    att = qh @ np.repeat(kh, nq // nk, axis=1).transpose(0, 1, 3, 2) * (hd ** -0.5)
    causal = np.triu(np.ones((T, T), dtype=bool), 1)
    att = np.where(causal[None, None], -np.inf, att)
    att = np.exp(att - att.max(axis=-1, keepdims=True))
    att = att / att.sum(axis=-1, keepdims=True)
    out = (att @ np.repeat(vh, nq // nk, axis=1)).transpose(0, 2, 1, 3).reshape(1, T, -1)
    out = out * (1.0 / (1.0 + np.exp(-gate.astype(np.float32))))
    ref_attn_out = out @ weights[p + "o_proj.weight"].astype(np.float32).T

    runner = SuryaGpuRunner(weights, spec)
    try:
        orig_rt = runner._rope_tables_device
        _cos_holder = {}

        def rt_dbg(positions):
            cb, sb = orig_rt(positions)
            _cos_holder["buf"] = cb
            print(f"[rt] post-upload cos row0={runner.debug_read(cb.ptr, 4)} "
                  f"row1={runner.debug_read(cb.ptr + 256, 4)}", flush=True)
            return cb, sb

        runner._rope_tables_device = rt_dbg

        def rd(tag):
            cb = _cos_holder.get("buf")
            if cb is not None:
                print(f"  [bis] {tag}: cos={runner.debug_read(cb.ptr, 2)}", flush=True)

        orig_k = runner._k

        def k_dbg(symbol, argtypes):
            fn = orig_k(symbol, argtypes)

            def call(*a):
                r = fn(*a)
                rd(symbol.replace("hipengine_", ""))
                return r
            return call

        runner._k = k_dbg
        import hipengine.runtime.surya as _smod
        for _fname in ("surya_gdn_l2norm_f32", "surya_split_qgate_f32", "surya_scatter_kv_f32"):
            _orig = getattr(_smod, _fname, None)
            if _orig is not None:
                def _mk(name, f):
                    def w(*a, **kw):
                        r = f(*a, **kw)
                        rd(name)
                        return r
                    return w
                setattr(_smod, _fname, _mk(_fname, _orig))

        # full-stack per-layer x comparison (after both residual adds)
        from hipengine.kernels.cpu_reference.surya import _full_attention_layer, text_rope_tables as _trt
        cpu_x_all = {}
        xcpu = emb[ids].copy()
        for bi in range(xcpu.shape[0]):
            mask = ids[bi] == spec.image_token_id
            xcpu[bi, mask] = visual[bi]
        cos_c, sin_c = _trt(spec, pos)
        _state_cache = {}
        for layer in range(spec.num_layers):
            lp = f"model.language_model.layers.{layer}."
            hh2 = rms_norm(xcpu, weights[lp + "input_layernorm.weight"])
            if spec.is_full_attention(layer):
                ao, _state_cache[layer] = _full_attention_layer(weights, spec, layer, hh2, cos_c, sin_c, _state_cache.get(layer))
            else:
                ao, _state_cache[layer] = _gdn_layer_prefill(weights, spec, layer, hh2, init=_state_cache.get(layer))
            xcpu = xcpu + ao
            cpu_x_all[("post_attn", layer)] = xcpu.copy()
            hh2 = rms_norm(xcpu, weights[lp + "post_attention_layernorm.weight"])
            xcpu = xcpu + _mlp(weights, lp, hh2)
            cpu_x_all[("post_mlp", layer)] = xcpu.copy()
        del _state_cache

        add_state = {"n": 0}
        orig_add = runner._add

        def add_dbg(x_ptr, y_ptr, out_ptr, n):
            orig_add(x_ptr, y_ptr, out_ptr, n)
            layer = add_state["n"] // 2
            which = "post_attn" if add_state["n"] % 2 == 0 else "post_mlp"
            add_state["n"] += 1
            if layer < spec.num_layers:
                gx = runner.debug_read(out_ptr, n)
                d = np.abs(gx - cpu_x_all[(which, layer)][0].ravel())
                print(f"x[{which},{layer}]: max|d|={d.max():.3e}", flush=True)

        runner._add = add_dbg

        orig_attn = runner._attn_layer_prefill

        def attn_dbg(layer, norm_ptr, out_ptr, tokens, cos_buf, sin_buf):
            if layer != 3:
                return orig_attn(layer, norm_ptr, out_ptr, tokens, cos_buf, sin_buf)
            nbuf = runner._buf("norm", 1)
            gn = runner.debug_read(nbuf.ptr, tokens * spec.hidden_size).reshape(tokens, spec.hidden_size)
            print(f"  norm(input to layer3): max|d|={np.abs(gn - h[0]).max():.3e}", flush=True)
            print(f"  [dbg] out_ptr={out_ptr:#x} tokens={tokens} bytes={tokens * spec.hidden_size * 4}", flush=True)
            orig_attn(layer, norm_ptr, out_ptr, tokens, cos_buf, sin_buf)
            print(f"  [dbg] orig_attn returned", flush=True)
            xbuf = runner._buf("x", 1)
            print(f"  [dbg] x ptr={xbuf.ptr:#x} nbytes={xbuf.nbytes}", flush=True)
            ab = runner._scratch.get("attn_out")
            print(f"  [dbg] scratch attn_out: {None if ab is None else hex(ab.ptr)} nbytes={None if ab is None else ab.nbytes}", flush=True)
            try:
                _ = runner.debug_read(xbuf.ptr, 16)
                print("  [dbg] read x ok", flush=True)
            except Exception as e:
                print(f"  [dbg] read x FAILED: {e}", flush=True)
            a = runner.debug_read(out_ptr, tokens * spec.hidden_size).reshape(tokens, spec.hidden_size)
            d = np.abs(a - ref_attn_out[0])
            print(f"attn layer3 out: max|d|={d.max():.3e} mean={d.mean():.3e}", flush=True)
            # stage dumps: re-run stages manually against the same input
            # (the GPU layer already completed; scratch holds its buffers)
            qpbuf = runner._buf("attn_qp", 1)
            qbuf3 = runner._buf("attn_q", 1)
            gqp = runner.debug_read(qpbuf.ptr, tokens * nq * hd * 2).reshape(tokens, nq * hd * 2)
            qpre = h[0] @ weights[p + "q_proj.weight"].astype(np.float32).T
            print(f"  qp(gemm): max|d|={np.abs(gqp - qpre).max():.3e}", flush=True)
            qbuf = runner._buf("attn_q", 1)
            gq = runner.debug_read(qbuf.ptr, tokens * nq * hd).reshape(tokens, nq, hd)
            dqr = np.abs(gq - q[0])
            print(f"  q(post-rope): nan={np.isnan(gq).any()} max|d|={dqr.max():.3e}", flush=True)
            per_tok = dqr.max(axis=(1, 2))
            print("  q per-token max[:8]:", [f"{v:.2e}" for v in per_tok[:8]], flush=True)
            bad = np.argwhere(dqr > 1.0)
            print(f"  q bad slots: n={len(bad)} first10={bad[:10].tolist()}", flush=True)
            print("  q per-dim max[:12]:", [f"{v:.2e}" for v in dqr.max(axis=(0,1))[:12]], flush=True)
            print("  gpu q[0,0,:8]:", gq[0, 0, :8], flush=True)
            nan_mask = np.isnan(gq)
            if nan_mask.any():
                tk = np.where(nan_mask.any(axis=(1, 2)))[0]
                print(f"  q nan tokens: n={len(tk)} first[:6]={tk[:6]} "
                      f"count_tok0={nan_mask[0].sum()}", flush=True)
                hd_nan = np.where(nan_mask[0].any(axis=0))[0]
                print(f"  q token0 nan dims: {hd_nan[:12]}", flush=True)
            # rmsnorm-stage check: rerun rmsnorm on qp into a fresh buffer and compare
            from hipengine.kernels.hip_gfx1100.evie.evie_ops import _evie_rmsnorm_f32 as rnf
            pp = p
            qn = runner._buf("attn_q", 1)
            rnf(runner.library, qpbuf.ptr, runner._w[pp + "q_norm.weight"].ptr,
                qn.ptr, tokens * nq, hd, 1e-6, stream=0, runtime=runner.runtime)
            gqn = runner.debug_read(qn.ptr, tokens * nq * hd).reshape(tokens, nq, hd)
            dq = np.abs(gqn - qn_cpu[0])
            print(f"  q rmsnorm rerun: max|d|={dq.max():.3e} nan={np.isnan(gqn).any()}", flush=True)
            dg = np.abs(gqn - gq)
            print(f"  rerun-vs-pipeline-q: max|d|={dg.max():.3e}", flush=True)
            dt = dq.max(axis=(1, 2))
            bt = np.where(dt > 1e-3)[0]
            print(f"  rerun bad tokens: n={len(bt)} first={bt[:6]}", flush=True)
            print("  gpu q[1,0,:8]:", gq[1, 0, :8], flush=True)
            # manual split of token 0 from qp
            src = gqp.reshape(tokens, nq, hd * 2)
            print("  split ref q[0,0,:8]:", src[0, 0, :8], flush=True)
            print("  mirror q(post-rope)[0,0,:8]:", q[0][0, 0, :8], flush=True)
            cbuf = runner._buf("rope_cos", 1)
            gcos = runner.debug_read(cbuf.ptr, 128).reshape(2, 64)
            from hipengine.kernels.cpu_reference.evie import text_rope_tables as trt
            hcos, hsin = trt(spec, pos)
            print("  dev cos row0[:4]:", gcos[0, :4], "host:", hcos[0, :4], flush=True)
            print("  dev cos row1[:4]:", gcos[1, :4], "host:", hcos[1, :4], flush=True)
            print("  dev cos row105[:4]:", runner.debug_read(cbuf.ptr + 105 * 256, 4), "host:", hcos[105, :4], flush=True)
            kbuf = runner._buf("attn_k", 1)
            gk = runner.debug_read(kbuf.ptr, tokens * nk * hd).reshape(tokens, nk, hd)
            dk = np.abs(gk - k[0])
            print(f"  k(post-rope): nan={np.isnan(gk).any()} max|d|={dk.max():.3e}", flush=True)
            print("  k per-dim max[:12]:", [f"{v:.2e}" for v in dk.max(axis=(0,1))[:12]], flush=True)
            kt = dk.max(axis=(1, 2))
            bad_k = np.where(kt > 1e-3)[0]
            print(f"  k bad tokens: n={len(bad_k)} first={bad_k[:8]} max={kt.max():.3e}", flush=True)
            qt = dqr.max(axis=(1, 2))
            bad_q = np.where(qt > 1e-3)[0]
            print(f"  q bad tokens: n={len(bad_q)} first={bad_q[:8]}", flush=True)
            kc, vc = runner._kv_cache[3]
            plane = runner.max_seq * hd
            gkc = np.stack([
                runner.debug_read(kc.ptr + h * plane * 4, tokens * hd).reshape(tokens, hd)
                for h in range(nk)], axis=0)  # (nk, tokens, hd)
            dkv = np.abs(gkc - k[0].transpose(1, 0, 2))
            print(f"  kcache: nan={np.isnan(gkc).any()} max|d|={dkv.max():.3e}", flush=True)
            kt = dkv.max(axis=(0, 2))
            bt = np.where(kt > 1e-3)[0]
            print(f"  kcache bad tokens: n={len(bt)} first={bt[:8]}", flush=True)
            hd_d = dkv.max(axis=(0, 1))
            print(f"  kcache per-dim max[:8]: {[f'{v:.2e}' for v in hd_d[:8]]}", flush=True)
            sbuf = runner._buf("scores", 1)
            gs = runner.debug_read(sbuf.ptr, nq * tokens * tokens).reshape(nq, tokens, tokens)
            print(f"  scores: nan={np.isnan(gs).any()}", flush=True)
            if np.isnan(gs).any():
                nan_rows = np.where(np.isnan(gs).any(axis=(1, 2)))[0]
                print(f"  nan score heads: {nan_rows}", flush=True)
            ob = runner._buf("attn_heads_out", 1)
            go = runner.debug_read(ob.ptr, tokens * nq * hd).reshape(tokens, nq, hd)
            print(f"  heads_out: nan={np.isnan(go).any()}", flush=True)
            mout = out[0].reshape(tokens, nq, hd)
            do = np.abs(go - mout)
            print(f"  heads_out vs mirror: max|d|={do.max():.3e} mean={do.mean():.3e}", flush=True)
            ot = do.max(axis=(1, 2))
            bt = np.where(ot > 1e-2)[0]
            print(f"  heads_out bad tokens: n={len(bt)} first={bt[:8]}", flush=True)
            sbuf2 = runner._buf("scores", 1)
            gs = runner.debug_read(sbuf2.ptr, nq * tokens * tokens).reshape(nq, tokens, tokens)
            mask_u = np.triu(np.ones((tokens, tokens), dtype=bool), 1)
            dpu = np.abs(gs - att[0])[:, ~mask_u]
            print(f"  probs(unmasked) vs mirror: max|d|={dpu.max():.3e} mean={dpu.mean():.3e}", flush=True)
            bad = np.argwhere((np.abs(gs - att[0]) > 0.1) & ~mask_u[None])
            print(f"  probs bad slots: n={len(bad)} first6={bad[:6].tolist()}", flush=True)
            if len(bad):
                h0, q0, k0 = bad[0]
                print(f"  gpu row (h={h0},q={q0})[:12]={gs[h0, q0, :12].round(3).tolist()}", flush=True)
                print(f"  mir row (h={h0},q={q0})[:12]={att[0][h0, q0, :12].round(3).tolist()}", flush=True)
            # isolated GEMM orientation test
            from hipengine.core.memory import malloc
            gemm_out = malloc(nq * tokens * tokens * 4 + 4096)
            plane = runner.max_seq * hd
            kc3, vc3 = runner._kv_cache[3]
            runner.rocblas.sgemm_batched(
                runner._dev_ptr_array([kc3.ptr + (h // (nq // nk)) * plane * 4 for h in range(nq)]).ptr,
                runner._dev_ptr_array([qbuf3.ptr + h * hd * 4 for h in range(nq)]).ptr,
                runner._dev_ptr_array([gemm_out.ptr + h * tokens * 4 for h in range(nq)]).ptr,
                batch=nq, m=tokens, n=tokens, k=hd,
                lda=hd, ldb=nq * hd, ldc=tokens,
                trans_a=True, trans_b=False)
            graw = runner.debug_read(gemm_out.ptr, nq * tokens * tokens).reshape(nq, tokens, tokens)
            from hipengine.kernels.cpu_reference.evie import apply_rope_half as _arh
            mir_pre = (qh @ np.repeat(kh, nq // nk, axis=1).transpose(0, 1, 3, 2) * (hd ** -0.5))[0]
            d1 = np.abs(graw[:, ~mask_u] - mir_pre[:, ~mask_u]).max()
            d2 = np.abs(graw.transpose(0, 2, 1)[:, ~mask_u] - mir_pre[:, ~mask_u]).max()
            print(f"  raw-gemm vs mirror: as-is={d1:.3e} transposed={d2:.3e}", flush=True)
            print(f"  graw[0,1,:4]={graw[0,1,:4].round(3).tolist()} mir_pre[0,1,:4]={mir_pre[0,1,:4].round(3).tolist()}", flush=True)
            print(f"  graw[0,0,:4]={graw[0,0,:4].round(3).tolist()} mir_pre[0,0,:4]={mir_pre[0,0,:4].round(3).tolist()}", flush=True)
            print(f"  graw[1,0,:4]={graw[1,0,:4].round(3).tolist()} mir_pre[0,:4,0]={mir_pre[0,:4,0].round(3).tolist()}", flush=True)
            print(f"  gpu scores[0,0,:4]={gs[0,0,:4]} mirror={att[0][0,0,:4]} "
                  f"gpu masked-sample={gs[0,0,5]} mirror={att[0][0,0,5]}", flush=True)

        runner._attn_layer_prefill = attn_dbg
        orig_mlp = runner._mlp
        cur_layer = {"l": 0}

        def mlp_dbg(prefix, norm_ptr, out_ptr, tokens):
            orig_mlp(prefix, norm_ptr, out_ptr, tokens)
            try:
                cb = runner._buf("rope_cos", 1)
                print(f"  cos@mlp{prefix.split('.')[3]}: {runner.debug_read(cb.ptr, 4)}", flush=True)
            except Exception:
                pass
            l = int(prefix.split(".")[3])
            xbuf = runner._buf("x", 1)
            gx = runner.debug_read(xbuf.ptr, tokens * spec.hidden_size).reshape(tokens, spec.hidden_size)
            d = np.abs(gx - cpu_x_after.get(l, cpu_x_after.get(min(l, max(cpu_x_after))))[0])
            print(f"x after layer {l}: max|d|={d.max():.3e}", flush=True)

        runner._mlp = mlp_dbg
        runner.prefill(ids[0], pos, visual_features=visual[0])
        from hipengine.kernels.cpu_reference.evie import rms_norm as _rn
        hid = _rn(xcpu, weights["model.language_model.norm.weight"])
        gl = runner.prefill(ids[0], pos, visual_features=visual[0])
        grow = runner.debug_read(runner._final_norm_row, spec.hidden_size)
        print(f"final-norm row vs cpu: max|d|={np.abs(grow - hid[0, -1]).max():.3e}", flush=True)
        cpu_logits_from_gpu_row = grow @ weights["model.language_model.embed_tokens.weight"].astype(np.float32).T
        print(f"cpu-lmhead(gpu row) vs gpu logits: max|d|={np.abs(cpu_logits_from_gpu_row - gl).max():.3e} "
              f"mean={np.abs(cpu_logits_from_gpu_row - gl).mean():.3e}", flush=True)
        cpu_logits = hid[0, -1] @ weights["model.language_model.embed_tokens.weight"].astype(np.float32).T
        print(f"cpu logits vs gpu logits: max|d|={np.abs(cpu_logits - gl).max():.3e}", flush=True)
    finally:
        runner.close()


if __name__ == "__main__":
    main()
