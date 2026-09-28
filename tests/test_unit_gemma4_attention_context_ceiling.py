"""The context ceiling the attention kernel's LDS budget puts on the runner.

``gemma4_attention_shared_bytes`` stores one logit per live key in LDS, so the key
count it accepts is bounded: ``(head_dim + keys + threads) * 4`` must fit 65,536
bytes. ``Gemma4Runner.__post_init__`` validates that against ``capacity`` -- the
number of positions the cache holds, which is the runner's context length -- so the
budget becomes a hard ceiling on how long a context the model can be configured for.

The ceiling is a capability refusal, not a slow path: construction raises. That is the
correct failure mode, and this file pins it so the number cannot move silently and so
the limit is visible to anyone changing the attention kernel.

CPU-only: ``__post_init__`` reads only ``weights.config``, so a ``SimpleNamespace``
stands in for the weights and no device is touched. The malloc/free pair is stubbed
for the same reason. The constructor is expected to raise before allocating anything,
which is itself worth asserting.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.kernels.cpu_reference.gemma4 import (
    Gemma4AttentionGeometry,
    Gemma4RopeConfig,
    Gemma4TextConfig,
)
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_attention import (
    gemma4_attention_shared_bytes,
)
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import Gemma4Runner


def _config(*, head_dim: int, window: int | None) -> Gemma4TextConfig:
    geometry = Gemma4AttentionGeometry(
        layer_type="sliding_attention" if window else "full_attention",
        num_heads=16,
        num_kv_heads=2,
        head_dim=head_dim,
        rope=Gemma4RopeConfig(
            rope_theta=10_000.0, head_dim=head_dim, rope_angles=head_dim // 2
        ),
        sliding_window=window,
        k_eq_v=False,
    )
    return Gemma4TextConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        top_k_experts=2,
        rms_norm_eps=1e-6,
        attention=(geometry,),
        vocab_size=128,
    )


def _construct(config: Gemma4TextConfig, capacity: int, monkeypatch) -> None:
    allocated: list[int] = []

    def _malloc(nbytes: int):
        allocated.append(int(nbytes))
        raise AssertionError("the capacity check must run before any allocation")

    monkeypatch.setattr(gemma4_module, "malloc", _malloc)
    monkeypatch.setattr(gemma4_module, "free", lambda buffer: None)
    Gemma4Runner(
        weights=SimpleNamespace(config=config), capacity=capacity, max_block=8
    )


def _largest_accepted_capacity(head_dim: int) -> int:
    """Bisect the real function rather than re-deriving its formula."""

    low, high = 1, 1 << 20
    while low < high:
        mid = (low + high + 1) // 2
        try:
            gemma4_attention_shared_bytes(head_dim=head_dim, keys=mid)
        except NotImplementedError:
            high = mid - 1
        else:
            low = mid
    return low


@pytest.mark.parametrize("head_dim", [256, 512])
def test_the_ceiling_is_where_the_lds_budget_runs_out(head_dim: int) -> None:
    """The boundary is the function's, and the runner uses the same function.

    Asserting the value here and not only in the attention module means a change to
    the LDS layout that moves the ceiling has to move it in one place, and the test
    fails where the number is written down.
    """

    ceiling = _largest_accepted_capacity(head_dim)
    threads = min(256, 1 << (head_dim - 1).bit_length())
    assert (head_dim + ceiling + threads) * 4 <= 64 * 1024
    assert (head_dim + ceiling + 1 + threads) * 4 > 64 * 1024


@pytest.mark.parametrize(
    "head_dim,capacity",
    [(256, 15_872), (512, 15_616)],
)
def test_a_capacity_at_the_ceiling_constructs(
    head_dim: int, capacity: int, monkeypatch
) -> None:
    """The ceiling itself is accepted. Off-by-one in the wrong direction would
    refuse a context the kernel can actually serve."""

    # Construction proceeds past the check and reaches allocation, which the stub
    # turns into a distinguishable failure. Reaching it is the pass condition.
    with pytest.raises(AssertionError, match="before any allocation"):
        _construct(_config(head_dim=head_dim, window=None), capacity, monkeypatch)


@pytest.mark.parametrize(
    "head_dim,capacity",
    [(256, 15_873), (512, 15_617)],
)
def test_a_capacity_above_the_ceiling_is_refused_loudly(
    head_dim: int, capacity: int, monkeypatch
) -> None:
    """One key past the ceiling refuses, by name, before allocating.

    This is the reproduction of the limit: a runner configured with a context the
    attention kernel cannot serve raises instead of truncating the context or
    silently serving a shorter one.
    """

    with pytest.raises(NotImplementedError) as caught:
        _construct(_config(head_dim=head_dim, window=None), capacity, monkeypatch)
    message = str(caught.value)
    assert str(head_dim) in message, "the refusal must name the geometry"
    assert str(capacity) in message, "the refusal must name the key count"


def test_a_sliding_window_does_not_lift_the_ceiling(monkeypatch) -> None:
    """The check is on capacity, not on the live key count, so a window does not help.

    A sliding layer bounds how many keys any one query attends to, but the runner
    still validates the full capacity against the LDS budget. That is the reading the
    construction path gives; this asserts it so the two cannot drift apart.
    """

    with pytest.raises(NotImplementedError):
        _construct(
            _config(head_dim=512, window=1024), 15_617, monkeypatch
        )


def test_the_generator_refuses_before_loading_any_weights(monkeypatch) -> None:
    """The refusal must not cost a multi-gigabyte load to reach.

    ``_ensure_runner`` loads the artifact and then constructs the runner, and the
    runner is where the ceiling was originally enforced. The geometry the ceiling
    depends on comes from metadata alone, so the generator now checks it first. This
    asserts that ordering: the load is stubbed to fail the test if it is reached, so
    a regression that moves the check back after the load is caught rather than
    quietly costing the user a load before the refusal.
    """

    import hipengine.generation.gemma4_gguf as module
    from hipengine.generation.gemma4_gguf import Gemma4GGUFGenerator

    def _load(*args, **kwargs):
        raise AssertionError("weights must not be loaded before the ceiling check")

    monkeypatch.setattr(module, "load_gemma4_device_weights", _load)
    monkeypatch.setattr(
        module,
        "gemma4_text_config_from_reader",
        lambda *a, **k: _config(head_dim=512, window=None),
    )

    generator = Gemma4GGUFGenerator(
        "/unused.gguf", object(), object(), context_length=32_768
    )
    generator._reader = object()

    with pytest.raises(NotImplementedError):
        generator._ensure_runner()
