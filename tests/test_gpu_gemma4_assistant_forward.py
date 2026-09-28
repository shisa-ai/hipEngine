"""The Gemma 4 assistant (MTP) head forward, against a real backbone.

The oracle is the head's own training objective. An MTP head is trained so that
``nextn_proj_post @ cur`` predicts the *target model's* hidden state one position
ahead, so a correct forward makes ``h_next`` track the backbone's own hidden state
at the next position. That is a much stronger check than "the logits look
plausible": it exercises the pre-projection, all four blocks, the shared-KV
attention, both post-norms, and the output projection, and a sign error or a
missing ``wo`` moves it.

The comparison is a cosine similarity rather than an exact match. The head is an
approximation of the target by construction -- four blocks against thirty -- so
the bar is "clearly tracking", not "equal". A wrong forward scores near zero or
negative.

The head's input hidden state has to be preserved across the extra backbone step
that produces the target, because ``Gemma4Runner`` reuses one hidden buffer.
It is staged into its own device buffer here; passing ``runner.hidden_state()``
after the second forward would silently hand the head the *target* as its input,
which would make the comparison meaningless and would pass.

Guarded on both artifacts and on HIP.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import (
    copy_device_to_host,
    copy_host_array_to_device,
    free,
    host_array_ptr,
    malloc,
)
from hipengine.loading.gguf import GGUFReader, scan_gguf
from hipengine.loading.gemma4_assistant_device import (
    load_gemma4_assistant_device_weights,
)
from hipengine.loading.gemma4_gguf import gemma4_gguf_config_from_metadata
from hipengine.runtime.gemma4 import Gemma4Runner, load_gemma4_device_weights
from hipengine.runtime.gemma4_assistant import Gemma4AssistantHead
from tests._rocm_guard import hip_runtime_available

BACKBONE = Path(
    "/models/gguf/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
)
HEAD = Path("/models/gguf/gemma-4-26B-A4B-it-GGUF/mtp-gemma-4-26B-A4B-it-Q8_0.gguf")

_needs = pytest.mark.skipif(
    not (hip_runtime_available() and BACKBONE.exists() and HEAD.exists()),
    reason="the Gemma 4 backbone, the MTP head, and HIP are all required",
)

# A short prompt with repeated structure. The head's prediction is a function of
# the backbone's hidden state, so a degenerate prompt would make the comparison
# vacuous.
PROMPT = [2, 818, 5279, 529, 7001, 108, 818, 5279, 529, 22172, 108, 107]
BACKBONE_WIDTH = 2816
VOCAB = 262144


def _to_float32(buffer) -> np.ndarray:
    """Copy a BF16 device buffer to host as float32.

    BF16 is the top 16 bits of a float32, not float16. ``view(np.float16)``
    reinterprets rather than converts, and produced a stable but meaningless
    cosine while this test was being written.
    """

    raw = np.empty(buffer.nbytes // 2, dtype=np.uint16)
    copy_device_to_host(host_array_ptr(raw), buffer, raw.nbytes)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@_needs
def test_the_head_predicts_the_backbones_next_hidden_state() -> None:
    """The forward's end-to-end correctness signal."""

    backbone_reader = GGUFReader(str(BACKBONE))
    backbone_weights = load_gemma4_device_weights(backbone_reader)
    runner = Gemma4Runner(weights=backbone_weights, capacity=64)
    head = None
    head_weights = None
    staged = None
    try:
        runner.forward(PROMPT, apply_softcap=False)
        position = runner.position

        # Stage the head's input hidden state into its own buffer, and pin the KV
        # read views at this live count. The views stay valid across the next
        # forward because the cache is append-only and the extra position lies
        # beyond `live`, so the mask still excludes it.
        source_hidden = _to_float32(runner.hidden_state())
        shared = {index: runner.shared_kv(index) for index in range(runner.layer_count)}
        staged = malloc(BACKBONE_WIDTH * 2)
        as_int = np.asarray(source_hidden, dtype=np.float32).view(np.uint32)
        bf16 = (
            (as_int + np.uint32(0x7FFF) + ((as_int >> 16) & np.uint32(1))) >> 16
        ).astype(np.uint16)
        copy_host_array_to_device(staged, np.ascontiguousarray(bf16).view(np.uint8))

        # Advance the backbone by one token to obtain what the head is trained to
        # predict. This overwrites the runner's hidden buffer, which is why the
        # input was staged above.
        runner.forward([PROMPT[-1]], apply_softcap=False)
        target_hidden = _to_float32(runner.hidden_state())

        assert not np.allclose(source_hidden, target_hidden), (
            "the two hidden states are identical, so this test could not detect a "
            "forward that returns its input"
        )

        backbone_config = gemma4_gguf_config_from_metadata(scan_gguf(BACKBONE))
        head_weights = load_gemma4_assistant_device_weights(str(HEAD))
        head = Gemma4AssistantHead(
            weights=head_weights,
            backbone=backbone_config,
            backbone_embedding=backbone_weights.embed_tokens,
            capacity=64,
            eps=1e-6,
        )

        logits, h_next = head.forward(
            PROMPT[-1],
            position=position - 1,
            backbone_hidden=staged,
            shared_kv=shared,
        )

        assert logits.shape == (VOCAB,)
        assert np.isfinite(logits).all()
        assert np.isfinite(h_next).all()
        assert h_next.shape == (BACKBONE_WIDTH,)
        assert np.abs(h_next).max() > 0, "the head produced an all-zero hidden state"

        cosine = _cosine(np.asarray(h_next, dtype=np.float32), target_hidden)
        assert cosine > 0.5, (
            f"the head's predicted hidden state does not track the backbone's: "
            f"cosine {cosine:.4f}. A correct MTP forward should be well above 0.5; "
            f"a missing wo, a wrong norm order, or reading the head's own "
            f"token_embd for the input scores near zero or negative."
        )
    finally:
        if head is not None:
            head.close()
        if head_weights is not None:
            head_weights.free()
        if staged is not None:
            free(staged)
        runner.close()
        backbone_weights.free()
