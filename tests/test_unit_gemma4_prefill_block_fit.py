"""The prefill block must shrink to fit the card it is running on.

`3871b31c7` raised ``DEFAULT_PREFILL_BLOCK`` from 512 to 1024 after six
paired measurements on the Radeon Pro W7900, whose 48 GB holds the resulting
25.43 GB resident set. The Radeon RX 7900 XTX has 24 GB, so the same
configuration ran the card to 99.76% and ``hipengine.LLM.generate()`` -- the
path a user actually reaches -- died with ``HipError: HIP error 2: out of
memory`` partway through decode, because the scratch that scales with the
block is taken lazily on the first forward rather than at load.

These tests pin the selection itself: the larger block where it fits, a
step down where it does not, and no change of behaviour when the runtime
cannot answer the memory query.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime.gemma4 import (
    _FIT_RESERVE_BYTES,
    _fit_prefill_block,
    _resident_bytes,
)

HIDDEN = 2816
CAPACITY = 8192


def _config(layers: int = 4) -> SimpleNamespace:
    """A small but structurally real config: real widths, real layer shapes."""

    attention = [
        SimpleNamespace(
            num_heads=8,
            num_kv_heads=2,
            head_dim=256,
            scale=0.05,
            k_eq_v=False,
            sliding_window=1024 if index % 2 == 0 else None,
        )
        for index in range(layers)
    ]
    return SimpleNamespace(
        hidden_size=HIDDEN,
        vocab_size=268_000,
        num_experts=8,
        top_k_experts=8,
        moe_intermediate_size=1024,
        attention=attention,
    )


def _dense_widths(config: SimpleNamespace) -> list[int]:
    return [14336] * len(config.attention)


def _fit(block: int, free_bytes: int, *, config: SimpleNamespace) -> int:
    """Run the selector against a stub runtime reporting ``free_bytes``."""

    import hipengine.core.hip as hip

    stub = SimpleNamespace(mem_get_info=lambda: (free_bytes, free_bytes))
    original = hip.get_hip_runtime
    hip.get_hip_runtime = lambda: stub  # type: ignore[assignment]
    try:
        return _fit_prefill_block(
            block,
            capacity=CAPACITY,
            hidden=HIDDEN,
            config=config,
            dense_widths=_dense_widths(config),
        )
    finally:
        hip.get_hip_runtime = original  # type: ignore[assignment]


def test_the_resident_set_grows_with_the_block() -> None:
    """The premise of the whole check: scratch scales with the block."""

    config = _config()
    kwargs = dict(
        capacity=CAPACITY,
        hidden=HIDDEN,
        config=config,
        dense_widths=_dense_widths(config),
    )
    at_512 = _resident_bytes(block=512, **kwargs)
    at_1024 = _resident_bytes(block=1024, **kwargs)
    assert at_1024 > at_512
    # The KV cache does not scale with the block, so the growth is the scratch
    # plus the two runner buffers that do.
    assert (at_1024 - at_512) > 0


def test_the_larger_block_is_kept_when_the_card_can_hold_it() -> None:
    config = _config()
    needed = _resident_bytes(
        block=1024,
        capacity=CAPACITY,
        hidden=HIDDEN,
        config=config,
        dense_widths=_dense_widths(config),
    )
    assert _fit(1024, needed + _FIT_RESERVE_BYTES + 1, config=config) == 1024


def test_the_block_steps_down_when_the_card_cannot_hold_it() -> None:
    """The XTX case: 1024 overflows, 512 fits, so 512 is what ships."""

    config = _config()
    kwargs = dict(
        capacity=CAPACITY,
        hidden=HIDDEN,
        config=config,
        dense_widths=_dense_widths(config),
    )
    needed_512 = _resident_bytes(block=512, **kwargs)
    needed_1024 = _resident_bytes(block=1024, **kwargs)
    # A budget that admits 512 (plus reserve) but not 1024.
    free = needed_512 + _FIT_RESERVE_BYTES + 1
    assert free < needed_1024 + _FIT_RESERVE_BYTES
    assert _fit(1024, free, config=config) == 512


def test_a_runtime_that_cannot_answer_leaves_the_block_alone() -> None:
    """A missing measurement is not a reason to refuse to run."""

    import hipengine.core.hip as hip

    class NoQuery:
        pass

    class Raises:
        @staticmethod
        def mem_get_info():
            raise RuntimeError("no device")

    config = _config()
    original = hip.get_hip_runtime
    try:
        hip.get_hip_runtime = lambda: NoQuery()  # type: ignore[assignment]
        assert (
            _fit_prefill_block(
                1024,
                capacity=CAPACITY,
                hidden=HIDDEN,
                config=config,
                dense_widths=_dense_widths(config),
            )
            == 1024
        )
        hip.get_hip_runtime = lambda: Raises()  # type: ignore[assignment]
        assert (
            _fit_prefill_block(
                1024,
                capacity=CAPACITY,
                hidden=HIDDEN,
                config=config,
                dense_widths=_dense_widths(config),
            )
            == 1024
        )
    finally:
        hip.get_hip_runtime = original  # type: ignore[assignment]


def test_the_selector_only_ever_shrinks_the_block() -> None:
    """The fit check may lower a block and never raise one.

    Also the structural reason an explicit ``max_block`` is safe: ``__post_init__``
    records ``auto_block`` before the fit call and only invokes it when it set the
    block itself, so a block handed in by a caller reaches the allocator untouched.
    """

    config = _config()
    kwargs = dict(
        capacity=CAPACITY,
        hidden=HIDDEN,
        config=config,
        dense_widths=_dense_widths(config),
    )
    needed_512 = _resident_bytes(block=512, **kwargs)
    for free in (0, needed_512 // 2, needed_512 + _FIT_RESERVE_BYTES + 1):
        chosen = _fit(1024, free, config=config)
        assert 1 <= chosen <= 1024
        # Never larger than requested, and always a power-of-two step down.
        assert 1024 % chosen == 0