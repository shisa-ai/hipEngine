#!/usr/bin/env python3
"""Bit-identity of Muse Glimmer verify projections vs per-row decode (diagnostic).

The exact DFlash verify contract needs every verify row's projection to equal
the rows=1 decode owner bit-for-bit. For each blk.0 projection role plus the
lm_head this runs ``launch_gguf_linear_exact_rows`` on N random BF16 rows and
``launch_gguf_linear`` rows=1 on each row, and reports mismatching elements.
Run it after changing a verify (rowtile/exact-rows) or decode GEMV owner; the
end-to-end gate is tests/test_live_muse_glimmer_dflash_exactness.py.

  .venv/bin/python scripts/muse_glimmer_verify_gemv_bitidentity.py --rows 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL = Path("/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--rows", type=int, default=5)
    args = parser.parse_args(argv)

    from hipengine.core.memory import copy_device_to_host, copy_host_array_to_device, free, malloc
    from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
    from hipengine.loading.gguf import scan_gguf
    from hipengine.loading.materialize import float_array_to_bf16_bits
    from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF
    from hipengine.runtime.gguf_linear import launch_gguf_linear, launch_gguf_linear_exact_rows

    gen = make_muse_glimmer_gguf_generator_gfx1151(
        model_path=args.model, weight_index=scan_gguf(args.model),
        model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=2048,
    )
    failures = 0
    try:
        s = gen._ensure_session()
        rt, cfg, n = s.runtime, s.config, args.rows
        rng = np.random.default_rng(0)
        roles = [
            ("blk.0.attn_q.weight", cfg.n_embd, cfg.qkv_dim),
            ("blk.0.attn_k.weight", cfg.n_embd, cfg.kv_dim),
            ("blk.0.attn_v.weight", cfg.n_embd, cfg.kv_dim),
            ("blk.0.attn_gate.weight", cfg.n_embd, cfg.qkv_dim),
            ("blk.0.attn_output.weight", cfg.qkv_dim, cfg.n_embd),
            ("blk.0.ffn_gate.weight", cfg.n_embd, cfg.n_ff),
            ("blk.0.ffn_up.weight", cfg.n_embd, cfg.n_ff),
            ("blk.0.ffn_down.weight", cfg.n_ff, cfg.n_embd),
            ("output.weight", cfg.n_embd, cfg.vocab_size),
        ]
        for name, inf, outf in roles:
            w = s.loaded.weight(name)
            x = float_array_to_bf16_bits((rng.standard_normal(n * inf) * 0.1).astype(np.float32))
            xbuf, ref_buf, out_buf = malloc(x.nbytes, runtime=rt), malloc(outf * 2, runtime=rt), malloc(n * outf * 2, runtime=rt)
            try:
                copy_host_array_to_device(xbuf, x, runtime=rt)
                ref = np.empty((n, outf), dtype=np.uint16)
                for r in range(n):
                    launch_gguf_linear(w, xbuf.ptr + r * inf * 2, ref_buf.ptr, 1, inf, outf,
                                       output_dtype="bf16", backend=s.backend, runtime=rt)
                    copy_device_to_host(ref[r].ctypes.data, ref_buf, runtime=rt)
                launch_gguf_linear_exact_rows(w, xbuf.ptr, out_buf.ptr, n, inf, outf,
                                              output_dtype="bf16", backend=s.backend, runtime=rt)
                got = np.empty((n, outf), dtype=np.uint16)
                copy_device_to_host(got.ctypes.data, out_buf, runtime=rt)
            finally:
                for b in (xbuf, ref_buf, out_buf):
                    free(b, runtime=rt)
            diff = int(np.count_nonzero(ref != got))
            failures += bool(diff)
            print(f"{name:28s} {w.spec.quant_key:10s} {inf:>6}->{outf:<7} rows={n} "
                  f"{'bit-exact' if not diff else f'MISMATCH {diff}/{ref.size}'}")
    finally:
        gen.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
