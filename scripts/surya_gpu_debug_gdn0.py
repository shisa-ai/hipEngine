"""Layer-0 stage-by-stage GPU vs CPU-reference comparison for the Surya decoder."""

import sys

import numpy as np

sys.path.insert(0, "scripts")
from surya_gpu_smoke import build_inputs  # noqa: E402

from hipengine.kernels.cpu_reference.evie import rms_norm  # noqa: E402
from hipengine.kernels.cpu_reference.surya import _gdn_layer_prefill, _mlp  # noqa: E402
from hipengine.runtime.surya import SuryaGpuRunner  # noqa: E402


def main() -> int:
    weights, spec, ids, pos, visual = build_inputs()
    T = ids.shape[1]
    h_dim = spec.hidden_size

    # --- CPU reference stages -------------------------------------------------
    emb = weights["model.language_model.embed_tokens.weight"]
    x0 = emb[ids].copy()
    for bi in range(x0.shape[0]):
        mask = ids[bi] == spec.image_token_id
        x0[bi, mask] = visual[bi]
    lp = "model.language_model.layers.0."
    h_in = rms_norm(x0, weights[lp + "input_layernorm.weight"])
    attn_out, gstate = _gdn_layer_prefill(weights, spec, 0, h_in)
    x1 = x0 + attn_out
    m = rms_norm(x1, weights[lp + "post_attention_layernorm.weight"])
    mlp_out = _mlp(weights, lp, m)
    x2 = x1 + mlp_out

    runner = SuryaGpuRunner(weights, spec)
    try:
        x0_buf = runner._embed(ids[0], visual[0])
        gx0 = runner.debug_read(x0_buf.ptr, T * h_dim).reshape(T, h_dim)
        print(f"embed:      max|d|={np.abs(gx0 - x0[0]).max():.3e}", flush=True)

        norm_buf = runner._buf("norm", T * h_dim * 4)
        runner._rmsnorm(x0_buf.ptr, runner._w[lp + "input_layernorm.weight"].ptr,
                        norm_buf.ptr, T, h_dim)
        gh = runner.debug_read(norm_buf.ptr, T * h_dim).reshape(T, h_dim)
        print(f"rmsnorm:    max|d|={np.abs(gh - h_in[0]).max():.3e}", flush=True)

        ao_buf = runner._buf("gdn_attn_out", T * h_dim * 4)
        runner._gdn_layer_prefill(0, norm_buf.ptr, ao_buf.ptr, T)
        gao = runner.debug_read(ao_buf.ptr, T * h_dim).reshape(T, h_dim)
        d = np.abs(gao - attn_out[0])
        print(f"gdn block:  max|d|={d.max():.3e} mean={d.mean():.3e} "
              f"scale={np.abs(attn_out[0]).max():.1f}", flush=True)
        dt = d.max(axis=1)
        print(f"gdn per-token max[:6]: {[f'{v:.2e}' for v in dt[:6]]}", flush=True)

        x1_buf = runner._buf("x", T * h_dim * 4)
        runner._add(x0_buf.ptr, ao_buf.ptr, x1_buf.ptr, T * h_dim)
        gx1 = runner.debug_read(x1_buf.ptr, T * h_dim).reshape(T, h_dim)
        print(f"residual:   max|d|={np.abs(gx1 - x1[0]).max():.3e}", flush=True)

        runner._rmsnorm(x1_buf.ptr, runner._w[lp + "post_attention_layernorm.weight"].ptr,
                        norm_buf.ptr, T, h_dim)
        gm = runner.debug_read(norm_buf.ptr, T * h_dim).reshape(T, h_dim)
        print(f"rmsnorm2:   max|d|={np.abs(gm - m[0]).max():.3e}", flush=True)

        mo_buf = runner._buf("mlp_out", T * h_dim * 4)
        runner._mlp(lp, norm_buf.ptr, mo_buf.ptr, T)
        gmo = runner.debug_read(mo_buf.ptr, T * h_dim).reshape(T, h_dim)
        d = np.abs(gmo - mlp_out[0])
        print(f"mlp:        max|d|={d.max():.3e} mean={d.mean():.3e} "
              f"scale={np.abs(mlp_out[0]).max():.1f}", flush=True)
    finally:
        runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
