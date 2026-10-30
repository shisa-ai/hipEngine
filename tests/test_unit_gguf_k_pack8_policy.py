"""The row-tile launcher selects pack8 by shape, not process state."""
from pathlib import Path


def test_rowtile_pack8_has_no_environment_rollback():
    source = (Path(__file__).resolve().parents[1] /
              'hipengine/kernels/hip_gfx1100/quant/gguf_k_gemv.hip').read_text()
    assert 'HIPENGINE_GGUF_K_PACK8_ROWS' not in source
    launcher = source.split('int launch_gguf_k_gemv_rowtile_out(', 1)[1].split(
        '// Dispatch fixed-grid-Y', 1)[0]
    assert 'if ((out_features % 8) == 0)' in launcher
    assert 'gguf_k_prefill_out_rowtile_kernel<' in launcher
