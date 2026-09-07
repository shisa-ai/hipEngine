"""Pinned real-header context for profile tests (never reads tensor payloads).

No synthetic manifest is declared numerically certified. Without local controls,
these positive integration cases skip; unknown-header negatives remain portable.
"""
from pathlib import Path
from hipengine.loading.gguf import scan_gguf
import pytest


def profile_context(model='qwen3_5_gguf', *, control=None):
    if control is None:
        control = ('Qwen3.6-35B-A3B-UD-Q4_K_M.gguf' if model == 'qwen3_5_moe_gguf'
                   else 'Qwen3.8-27B-Q4_K_M.gguf')
    path = Path('/models/gguf') / control
    if not path.is_file():
        pytest.skip(f'pinned profile header absent: {path}')
    return {'weight_index': scan_gguf(path), 'model_path': str(path)}
