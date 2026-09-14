"""Q8 parent registration must not depend on unrelated test imports."""

from pathlib import Path
import subprocess
import sys


def test_q8_grouped_parent_registers_in_a_fresh_process():
    result = subprocess.run([
        sys.executable, "-c",
        "from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels;"
        "from hipengine.kernels.registry import resolve;"
        "register_gfx1151_kernels();"
        "resolve(backend='hip_gfx1151',layer='linear',quant='gguf_q8_0',"
        "variant='selected_grouped_wmma_prefill_bf16_bf16_out')",
    ], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
