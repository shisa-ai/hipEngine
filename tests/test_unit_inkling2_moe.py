import json
import struct

import numpy as np
import pytest

from hipengine.kernels.cpu_reference.inkling2_mini import routed_feed_forward
from hipengine.loading.inkling2_mini import Inkling2TextWeights
from hipengine.loading.safetensors import load_weight_index


def test_checkpoint_expert_slices_and_gate_up_rows(tmp_path):
    tensors = {
        "embed.weight": ("BF16", np.array([[0x3F80, 0xBF80], [0x4000, 0]], dtype="<u2")),
        "large.weight": ("BF16", np.full((257, 129), 0x3f80, dtype="<u2")),
        "layers.0.mlp.experts.w13_weight": ("U8", np.stack([
            np.full((2, 8), 0x11, dtype=np.uint8), np.full((2, 8), 0x22, dtype=np.uint8),
        ])),
        "layers.0.mlp.experts.w13_weight_scale": ("F8_E4M3", np.full((2, 2, 1), 0x38, dtype=np.uint8)),
        "layers.0.mlp.experts.w13_weight_amax": ("F32", np.array([1, 2], dtype="<f4")),
        "layers.0.mlp.experts.w13_weight_outer_scale_denominator": ("F32", np.array([2], dtype="<f4")),
    }
    header, parts, offset = {}, [], 0
    for name, (dtype, array) in tensors.items():
        header[f"model.llm.{name}"] = {
            "dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes],
        }
        parts.append(array.tobytes())
        offset += array.nbytes
    data = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(data)) + data + b"".join(parts))
    (tmp_path / "config.json").write_text("{}")
    weights = Inkling2TextWeights(load_weight_index(tmp_path))
    embedding = weights.dense("embed.weight")
    assert weights.dense("embed.weight") is embedding
    assert not embedding.flags.writeable
    selected = weights.dense("embed.weight", 1)
    assert np.shares_memory(embedding, selected)
    np.testing.assert_array_equal(selected, [2.0, 0.0])
    np.testing.assert_array_equal(weights.dense("large.weight", 17), np.ones(129, dtype=np.float32))
    assert weights.dense("large.weight") is not weights.dense("large.weight")
    assert not weights.raw("embed.weight").flags.writeable
    np.testing.assert_array_equal(weights.expert(0, "w13", 0), np.full((2, 16), 0.25))
    np.testing.assert_array_equal(weights.expert(0, "w13", 1), np.ones((2, 16)))
    with pytest.raises(IndexError, match="expert"):
        weights.expert(0, "w13", -1)


def test_block_router_groups_projection_and_matches_scalar_top_k_and_mix():
    class Weights:
        def dense(self, name):
            assert name.endswith("gate.bias")
            return np.array([0, 0, 0.25, -0.25], dtype=np.float32)

    hidden = np.arange(15, dtype=np.float32).reshape(5, 3)
    shapes, routed = [], []

    def project(name, x):
        assert name.endswith("gate.weight")
        shapes.append(x.shape)
        col = x[..., 0]
        return np.stack((col, col, 3 - col, col / 4), axis=-1)

    def feed(layer, ids, mix, x, gate, up):
        routed.append((ids.copy(), mix.copy()))
        return x

    options = dict(top_k=2, route_scale=8, gate_cap=4, up_cap=25, project_dense=project, feed_group=feed)
    for row in hidden:
        routed_feed_forward(Weights(), 0, row, **options)
    expected_ids, expected_mix = map(np.stack, zip(*routed, strict=True))
    shapes.clear()
    routed.clear()
    assert routed_feed_forward(Weights(), 0, hidden, **options) is hidden
    assert shapes == [hidden.shape]
    [(ids, mix)] = routed
    np.testing.assert_array_equal(ids, expected_ids)
    np.testing.assert_array_equal(mix.view(np.uint32), expected_mix.view(np.uint32))


def test_feed_forward_corrects_ranking_not_weights_and_preserves_row_order():
    class Weights:
        reads = []

        def dense(self, name):
            if name.endswith("gate.weight"):
                return np.array([[0.0], [0.0], [np.log(3.0)]], dtype=np.float32)
            return np.array([-1.0, 1.0, 0.0], dtype=np.float32)

        def expert(self, layer, projection, index):
            self.reads.append((layer, projection, index))
            if projection == "w13":
                return np.array([[1.0], [2.0], [-1.0], [3.0]], dtype=np.float32)
            return np.array([[2.0, float(index)]], dtype=np.float32)

    weights = Weights()
    actual = routed_feed_forward(weights, 5, np.ones(1, dtype=np.float32), top_k=2,
                                 route_scale=8, gate_cap=4, up_cap=25)
    gate = 4 * np.tanh(1 / 4)
    first = gate / (1 + np.exp(-1)) * 25 * np.tanh(2 / 25)
    second = -gate / (1 + np.exp(1)) * 25 * np.tanh(3 / 25)
    expected = 8 * (0.4 * (2 * first + second) + 0.6 * (2 * first + 2 * second))
    np.testing.assert_allclose(actual, [expected], rtol=1e-6)
    assert weights.reads == [(5, "w13", 1), (5, "w2", 1), (5, "w13", 2), (5, "w2", 2)]
