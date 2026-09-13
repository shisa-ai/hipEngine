from __future__ import annotations

import numpy as np
import pytest

from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.quant.gguf import bf16_to_float32
from scripts.gguf_gdn_replay_compare import GDNShape, compare_gdn_replay, replay_gdn


def test_compare_gdn_replay_matches_synthetic_device_outputs() -> None:
    shape = GDNShape(num_k_heads=1, num_v_heads=1, head_k_dim=2, head_v_dim=2)
    conv_out = np.asarray(
        [
            [0.2, -0.3, 0.4, 0.5, 0.1, -0.2],
            [-0.1, 0.25, 0.3, -0.4, 0.5, 0.75],
        ],
        dtype=np.float32,
    )
    gate = np.asarray([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32)
    alpha = np.asarray([[0.05], [0.1]], dtype=np.float32)
    beta = np.asarray([[0.2], [-0.3]], dtype=np.float32)
    norm = np.asarray([1.0, -0.5], dtype=np.float32)
    dt_bias = np.asarray([0.01], dtype=np.float32)
    a_log = np.asarray([-0.25], dtype=np.float32)
    recurrent = replay_gdn(
        conv_out=conv_out,
        gate=gate,
        alpha=alpha,
        beta=beta,
        norm_weight=norm,
        dt_bias=dt_bias,
        a_log=a_log,
        shape=shape,
        eps=1.0e-6,
    )
    recurrent_bf16 = bf16_to_float32(float_array_to_bf16_bits(recurrent)).astype(np.float32)

    comparison = compare_gdn_replay(
        conv_out=conv_out,
        gate=gate,
        alpha=alpha,
        beta=beta,
        norm_weight=norm,
        dt_bias=dt_bias,
        a_log=a_log,
        device_recurrent_out=recurrent,
        device_recurrent_bf16=recurrent_bf16,
        shape=shape,
        eps=1.0e-6,
    )

    assert comparison["recurrent_out_vs_cpu"]["max_abs_diff"] == 0.0
    assert comparison["recurrent_bf16_vs_cpu_bf16"]["max_abs_diff"] == 0.0
    assert comparison["cpu_f32_vs_cpu_bf16"]["count"] == 4


def test_replay_gdn_rejects_shape_mismatches() -> None:
    shape = GDNShape(num_k_heads=1, num_v_heads=1, head_k_dim=2, head_v_dim=2)
    with pytest.raises(ValueError, match="conv_out shape"):
        replay_gdn(
            conv_out=np.ones((2, 5), dtype=np.float32),
            gate=np.ones((2, 2), dtype=np.float32),
            alpha=np.ones((2, 1), dtype=np.float32),
            beta=np.ones((2, 1), dtype=np.float32),
            norm_weight=np.ones((2,), dtype=np.float32),
            dt_bias=np.ones((1,), dtype=np.float32),
            a_log=np.ones((1,), dtype=np.float32),
            shape=shape,
            eps=1.0e-6,
        )
