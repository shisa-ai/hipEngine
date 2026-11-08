#!/usr/bin/env python3
"""Resolve Muse Glimmer projection dispatch variants at rows=1..16 (read-only).

Prints the registry variant the default linear dispatcher picks per projection
role and row count, to confirm which owner a decode (rows=1), verify (rows 2-16)
or draft shape reaches. Note the exact verify path bypasses this dispatcher
(launch_gguf_linear_exact_rows); see muse_glimmer_verify_gemv_bitidentity.py.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from hipengine.runtime.gguf_linear import resolve_gguf_linear_dispatch
from hipengine.generation.muse_glimmer_gguf import make_muse_glimmer_gguf_generator_gfx1151
from hipengine.loading.gguf import scan_gguf
from hipengine.models.muse_glimmer import MUSE_GLIMMER_GGUF

TARGET = "/models/gguf/Muse-Glimmer-30B-GGUF/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf"
gen = make_muse_glimmer_gguf_generator_gfx1151(
    model_path=TARGET, weight_index=scan_gguf(TARGET),
    model_plugin=MUSE_GLIMMER_GGUF, max_sequence_length=2048,
)
session = gen._ensure_session()
loaded = session.loaded
cfg = session.config
print(f"hidden={cfg.n_embd} q_dim={cfg.qkv_dim} kv_dim={cfg.kv_dim} ffn={cfg.n_ff} vocab={cfg.vocab_size}")
shapes = [
    ("attn_q", "blk.0.attn_q.weight", cfg.n_embd, cfg.qkv_dim, "bf16"),
    ("attn_k", "blk.0.attn_k.weight", cfg.n_embd, cfg.kv_dim, "bf16"),
    ("attn_v", "blk.0.attn_v.weight", cfg.n_embd, cfg.kv_dim, "bf16"),
    ("attn_gate", "blk.0.attn_gate.weight", cfg.n_embd, cfg.qkv_dim, "bf16"),
    ("attn_output", "blk.0.attn_output.weight", cfg.qkv_dim, cfg.n_embd, "bf16"),
    ("ffn_gate", "blk.0.ffn_gate.weight", cfg.n_embd, cfg.n_ff, "bf16"),
    ("ffn_up", "blk.0.ffn_up.weight", cfg.n_embd, cfg.n_ff, "bf16"),
    ("ffn_down", "blk.0.ffn_down.weight", cfg.n_ff, cfg.n_embd, "bf16"),
    ("output", "output.weight", cfg.n_embd, cfg.vocab_size, "bf16"),
]
for label, name, inf, outf, outdtype in shapes:
    w = loaded.weight(name)
    print(f"\n{label} {name} quant={w.spec.quant_key} layout={w.spec.layout} ({inf}->{outf})")
    for rows in (1, 2, 4, 8, 15, 16):
        d = resolve_gguf_linear_dispatch(w, activation_dtype="bf16", output_dtype=outdtype, rows=rows)
        print(f"  rows={rows:2d}: abi={d.abi:10s} variant={d.key.variant}")
gen.close()