from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.generation import GenerationRequest
from hipengine.generation.inkling2_mini import Inkling2ReferenceGenerator
from hipengine.runtime.inkling2_mini_vision import interpolate_positions, prepare_vision_patches


def test_patches_are_merge_block_major_and_keep_edge_padding():
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    patches, positions, grid = prepare_vision_patches(rgb, patch=1, merge=2)
    assert grid == (1, 2)
    assert positions.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [0, 3], [1, 2], [1, 3]]
    pixels = np.pad(rgb.astype(np.float32), ((0, 0), (0, 1), (0, 0)), constant_values=-1)
    np.testing.assert_array_equal(patches, pixels[positions[:, 0], positions[:, 1]] * (2 / 255) - 1)


def test_position_interpolation_uses_half_pixel_centers_and_clamps_edges():
    table = np.array([[0], [2], [4], [6]], dtype=np.float32)
    np.testing.assert_allclose(interpolate_positions(table, np.array([[0, 0], [0, 1], [0, 2]]), (1, 3), 2),
                               [[2.0], [3.0], [4.0]])


def test_multimodal_expands_only_the_placeholder_and_replaces_its_embeddings(tmp_path):
    (tmp_path / "processor_config.json").write_text('{"image_token": "<image>"}')
    features = np.array([[1, 0], [3, 7]], dtype=np.float32)
    calls = []

    class Session:
        def step(self, token, *, logits=True, embedding=None):
            calls.append((token, None if embedding is None else tuple(embedding)))
            return np.array([0.0, 2.0], dtype=np.float32) if logits else None

    gen = Inkling2ReferenceGenerator.__new__(Inkling2ReferenceGenerator)
    gen.model_path = tmp_path
    gen.max_sequence_length = 5
    gen.eos_token_id = 1
    gen.model = SimpleNamespace(session=lambda size: Session())
    gen.vision = SimpleNamespace(encode=lambda image, check: features)
    gen.tokenizer = SimpleNamespace(token_to_id=lambda token: 4, decode=lambda ids, **_: "result")
    gen.tokenize = lambda prompt: (0, 4, 1) if prompt.count("<image>") == 1 else (4, 4)
    request = GenerationRequest(prompts=("<image>",), max_tokens=1, temperature=0, top_p=1, ignore_eos=False)
    output = gen.generate_multimodal_detailed("<image>", object(), request)
    assert calls == [(0, None), (4, (1, 0)), (4, (3, 7)), (1, None)]
    assert output.prompt_tokens == 4
    assert output.generated_token_ids == (1,)
    with pytest.raises(ValueError, match="exactly one"):
        gen.generate_multimodal_detailed("<image><image>", object(), request)
