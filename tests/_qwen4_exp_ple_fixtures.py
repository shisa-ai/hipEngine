"""CPU-only small PLE payloads shared across test tiers."""

from math import prod

import numpy as np

from hipengine.loading.gguf import GGUFTensorInfo
from hipengine.quant.gguf import GGMLQuantizationType


def _iq4_nl_rows(scales):
    raw = np.zeros((len(scales), 5 * 18), dtype=np.uint8)
    for row, scale in enumerate(scales):
        encoded = np.asarray([scale], dtype=np.float16).view(np.uint8)
        for block in range(5):
            raw[row, block * 18:block * 18 + 2] = encoded
    return raw


def _ple_tensor(rows):
    shape = (rows, 160)
    return GGUFTensorInfo(
        name="per_layer_token_embd.weight", shape=shape,
        ggml_shape=tuple(reversed(shape)),
        ggml_type=int(GGMLQuantizationType.IQ4_NL), ggml_type_name="IQ4_NL",
        n_elements=prod(shape), nbytes=rows * 90, offset=0, data_offset=0,
        byte_shape=(rows, 90),
    )
