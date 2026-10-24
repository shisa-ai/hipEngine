"""Size-only boundary tests for the GGUF full-stack scratch planner.

``_full_stack_scratch_plan(..., materialize=False)`` is the form the resident
capacity estimator uses to price dozens of candidate contexts. Its contract has
two halves:

* **Byte accounting** — every numeric field, ``field_sizes``, and the ordered
  ``owner_sizes`` sequence must be identical to the materialized form, so the
  estimator prices exactly what the allocator will ask for.
* **Size-only** — it must build no host arrays and must not run the rope table
  construction, because a 262144-token candidate would otherwise allocate a
  rope table per candidate.

These tests use small synthetic geometry (no HIP, no model load) so the CPU cost
stays bounded, and they resolve the planner's *defining* module at call time, so
the guard targets keep working if the planner moves into its own module.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from hipengine.loading.qwen35_gguf import FULL_ATTENTION, LINEAR_ATTENTION
from hipengine.runtime import qwen35_gguf_runner as gguf_runner

_PLAN_FN = gguf_runner._full_stack_scratch_plan

#: Host arrays the plan owns. ``materialize=False`` must leave every one ``None``.
_HOST_ARRAY_FIELDS = (
    "conv_zero",
    "recurrent_zero",
    "block_table_arr",
    "position_host",
    "context_host",
    "cos_arr",
    "sin_arr",
)

#: ``numpy`` constructors that only ever run for host materialization.
_ARRAY_CONSTRUCTORS = frozenset({"zeros", "ones", "empty", "full", "arange"})

_INT8 = "int8_per_token_head"

#: Small fake geometry: byte accounting is exact arithmetic, so the absolute
#: widths do not matter and staying small keeps the materialized control cheap.
_GEOMETRY = {
    "hidden_size": 64,
    "ffn_size": 128,
    "q_width": 96,
    "kv_width": 32,
    "linear_qkv_width": 48,
}

_DENSE_CONTEXT_LENGTH = 32_768
_MOE_CONTEXT_LENGTH = 32_768


# ---------------------------------------------------------------------------
# Fixtures: fake configs / runners
# ---------------------------------------------------------------------------


def _dense_config(context_length: int = _DENSE_CONTEXT_LENGTH):
    from types import SimpleNamespace

    return SimpleNamespace(
        context_length=context_length,
        layer_types=tuple([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 4,
        expert_used_count=0,
        is_moe=False,
        expert_count=0,
        expert_shared_feed_forward_length=0,
        ssm_inner_size=64,
        ssm_conv_kernel=4,
        ssm_time_step_rank=8,
        ssm_state_size=4,
        ssm_group_count=2,
        head_count_kv=2,
        key_length=32,
        rope_dimension_count=8,
        rope_freq_base=10_000.0,
        head_count=4,
    )


def _moe_config(context_length: int = _MOE_CONTEXT_LENGTH):
    from types import SimpleNamespace

    return SimpleNamespace(
        context_length=context_length,
        layer_types=tuple([LINEAR_ATTENTION] * 3 + [FULL_ATTENTION]) * 4,
        expert_used_count=4,
        is_moe=True,
        expert_count=16,
        expert_shared_feed_forward_length=32,
        ssm_inner_size=64,
        ssm_conv_kernel=4,
        ssm_time_step_rank=8,
        ssm_state_size=4,
        ssm_group_count=2,
        head_count_kv=2,
        key_length=32,
        rope_dimension_count=8,
        rope_freq_base=10_000.0,
        head_count=4,
    )


def _runner(config, *, fp16_recurrent_state: bool = False):
    from types import SimpleNamespace

    return SimpleNamespace(
        weights=SimpleNamespace(config=config),
        fp16_recurrent_state=bool(fp16_recurrent_state),
        **_GEOMETRY,
    )


def _dense_runner():
    return _runner(_dense_config())


def _dense_runner_fp16_recurrent():
    return _runner(_dense_config(), fp16_recurrent_state=True)


def _moe_runner():
    return _runner(_moe_config())


def _plan(runner, *, materialize: bool, **kwargs):
    """Call the planner through the runner's public entry point."""

    return _PLAN_FN(
        runner.weights.config,
        hidden_size=runner.hidden_size,
        ffn_size=runner.ffn_size,
        q_width=runner.q_width,
        kv_width=runner.kv_width,
        linear_qkv_width=runner.linear_qkv_width,
        fp16_recurrent_state=runner.fp16_recurrent_state,
        materialize=materialize,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Guard plumbing
# ---------------------------------------------------------------------------


def _planning_module():
    """Module that actually holds the planner body.

    Resolved from the function object through ``sys.modules`` rather than a
    hard-coded name, so the guard targets follow the planner when it is extracted
    into its own module (``hipengine.runtime.qwen35_gguf_planning``) while the
    runner keeps an alias.
    """

    return sys.modules[_PLAN_FN.__module__]


class _NumpyConstructionSpy:
    """Stand-in for the planning module's ``numpy`` reference.

    Records (and optionally rejects) host-array construction while delegating
    every other attribute, so dtype parsing and ``np.prod`` keep working.
    """

    def __init__(self, real_module, *, block: bool) -> None:
        self._real = real_module
        self._block = bool(block)
        self.calls: list[str] = []

    def __getattr__(self, name: str):
        if name not in _ARRAY_CONSTRUCTORS:
            return getattr(self._real, name)
        real_constructor = getattr(self._real, name)

        def _construct(*args, **kwargs):
            self.calls.append(name)
            if self._block:
                raise AssertionError(
                    f"size-only planning constructed a host array via numpy.{name}(); "
                    "materialize=False must return byte sizes only"
                )
            return real_constructor(*args, **kwargs)

        return _construct


def _install_numpy_spy(monkeypatch, *, block: bool) -> _NumpyConstructionSpy:
    module = _planning_module()
    real_numpy = getattr(module, "np", np)
    spy = _NumpyConstructionSpy(real_numpy, block=block)
    monkeypatch.setattr(module, "np", spy, raising=False)
    return spy


def _install_rope_spy(monkeypatch, *, block: bool) -> list:
    module = _planning_module()
    real = getattr(module, "_rope_tables", None)
    if real is None:
        pytest.fail(
            f"{module.__name__} no longer exposes _rope_tables; retarget the rope guard "
            "to wherever the planner now resolves its rope table builder"
        )
    calls: list = []

    def _spy(*args, **kwargs):
        calls.append((args, kwargs))
        if block:
            raise AssertionError(
                "size-only planning built rope tables via _rope_tables(); "
                "materialize=False must return byte sizes only"
            )
        return real(*args, **kwargs)

    monkeypatch.setattr(module, "_rope_tables", _spy)
    return calls


def _mirror_threshold_positions() -> int:
    return int(getattr(gguf_runner, "_GGUF_INT8_SHORT_BF16_MIRROR_MAX_POSITIONS", 8_192))


# ---------------------------------------------------------------------------
# Case matrix
# ---------------------------------------------------------------------------
#
# Context lengths bracket the INT8 short-context BF16 mirror cutoff on both
# sides plus values unrelated to it, and vary slot count, block size, KV storage
# policy, scale precision, and recurrent-state precision.

_PLAN_CASES = (
    ("bf16_block_aligned_256", _dense_runner, {"max_sequence_length": 256}),
    ("bf16_unrelated_640", _dense_runner, {"max_sequence_length": 640}),
    ("bf16_unrelated_4096", _dense_runner, {"max_sequence_length": 4096}),
    ("bf16_max_context_32768", _dense_runner, {"max_sequence_length": 32_768}),
    (
        "int8_fp32_below_mirror_7936",
        _dense_runner,
        {"max_sequence_length": 7936, "kv_storage_dtype": _INT8, "kv_scale_dtype": "fp32"},
    ),
    (
        "int8_fp32_at_mirror_8192",
        _dense_runner,
        {"max_sequence_length": 8192, "kv_storage_dtype": _INT8, "kv_scale_dtype": "fp32"},
    ),
    (
        "int8_fp32_past_mirror_8193",
        _dense_runner,
        {"max_sequence_length": 8193, "kv_storage_dtype": _INT8, "kv_scale_dtype": "fp32"},
    ),
    (
        "int8_fp16_past_mirror_8448",
        _dense_runner,
        {"max_sequence_length": 8448, "kv_storage_dtype": _INT8, "kv_scale_dtype": "fp16"},
    ),
    (
        "int8_fp32_block128_at_mirror_8192",
        _dense_runner,
        {
            "max_sequence_length": 8192,
            "block_size": 128,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
        },
    ),
    (
        "int8_fp32_block128_past_mirror_8320",
        _dense_runner,
        {
            "max_sequence_length": 8320,
            "block_size": 128,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
        },
    ),
    (
        "int8_fp32_bf16_prefix_layers",
        _dense_runner,
        {
            "max_sequence_length": 16_384,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
            "int8_bf16_prefix_full_attention_layers": 2,
        },
    ),
    (
        "int8_fp32_key_only_bf16_values",
        _dense_runner,
        {
            "max_sequence_length": 4096,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
            "int8_kv_value_bf16": True,
        },
    ),
    (
        "int8_fp32_four_slots_at_mirror",
        _dense_runner,
        {
            "max_sequence_length": 8192,
            "max_batch_size": 4,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
        },
    ),
    (
        "int8_fp32_three_slots_past_mirror",
        _dense_runner,
        {
            "max_sequence_length": 8193,
            "max_batch_size": 3,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
        },
    ),
    (
        "int8_fp32_four_slots_no_kv_cache",
        _dense_runner,
        {
            "max_sequence_length": 32_768,
            "max_batch_size": 4,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
            "allocate_kv_cache": False,
        },
    ),
    (
        "int8_fp32_recurrent_fp16_16384",
        _dense_runner_fp16_recurrent,
        {"max_sequence_length": 16_384, "kv_storage_dtype": _INT8, "kv_scale_dtype": "fp32"},
    ),
    ("moe_bf16_16384", _moe_runner, {"max_sequence_length": 16_384}),
    (
        "moe_int8_fp32_two_slots_prefix_layers",
        _moe_runner,
        {
            "max_sequence_length": 16_384,
            "max_batch_size": 2,
            "kv_storage_dtype": _INT8,
            "kv_scale_dtype": "fp32",
            "int8_bf16_prefix_full_attention_layers": 2,
        },
    ),
)

_CASE_IDS = tuple(case[0] for case in _PLAN_CASES)
_CASE_PARAMS = pytest.mark.parametrize(
    ("runner_factory", "plan_kwargs"),
    tuple((case[1], case[2]) for case in _PLAN_CASES),
    ids=_CASE_IDS,
)


# ---------------------------------------------------------------------------
# Size-only contract: no rope table, no host arrays
# ---------------------------------------------------------------------------


@_CASE_PARAMS
def test_size_only_plan_does_not_build_rope_tables(
    monkeypatch, runner_factory, plan_kwargs
) -> None:
    calls = _install_rope_spy(monkeypatch, block=True)

    _plan(runner_factory(), materialize=False, **plan_kwargs)

    assert calls == []


@_CASE_PARAMS
def test_size_only_plan_does_not_construct_host_arrays(
    monkeypatch, runner_factory, plan_kwargs
) -> None:
    spy = _install_numpy_spy(monkeypatch, block=True)

    _plan(runner_factory(), materialize=False, **plan_kwargs)

    assert spy.calls == []


@_CASE_PARAMS
def test_size_only_plan_returns_no_host_arrays(runner_factory, plan_kwargs) -> None:
    plan = _plan(runner_factory(), materialize=False, **plan_kwargs)

    for field_name in _HOST_ARRAY_FIELDS:
        assert getattr(plan, field_name) is None, field_name

    # The byte sizes the estimator consumes must still be populated.
    assert plan.block_table_nbytes > 0
    assert plan.position_nbytes > 0
    assert plan.context_nbytes > 0
    assert plan.cos_nbytes > 0
    assert plan.sin_nbytes == plan.cos_nbytes
    assert plan.conv_zero_nbytes > 0
    assert plan.recurrent_zero_nbytes > 0
    assert plan.dedicated_bytes == sum(plan.owner_sizes)
    assert plan.arena_bytes() > 0


# ---------------------------------------------------------------------------
# Size-only contract: byte accounting identical to the materialized plan
# ---------------------------------------------------------------------------


@_CASE_PARAMS
def test_size_only_plan_matches_materialized_byte_accounting(runner_factory, plan_kwargs) -> None:
    runner = runner_factory()
    sized_only = _plan(runner, materialize=False, **plan_kwargs)
    materialized = _plan(runner, materialize=True, **plan_kwargs)

    assert sized_only.to_json_dict() == materialized.to_json_dict()
    assert sized_only.field_sizes == materialized.field_sizes
    assert list(sized_only.field_sizes) == list(materialized.field_sizes)
    assert sized_only.owner_sizes == materialized.owner_sizes
    assert sized_only.dedicated_bytes == materialized.dedicated_bytes
    assert sized_only.fixed_bytes == materialized.fixed_bytes
    assert sized_only.context_scaled_bytes == materialized.context_scaled_bytes
    assert sized_only.arena_bytes() == materialized.arena_bytes()


@_CASE_PARAMS
def test_materialized_plan_is_the_positive_control(monkeypatch, runner_factory, plan_kwargs) -> None:
    """The materialized form must exercise the guards the size-only form skips."""

    rope_calls = _install_rope_spy(monkeypatch, block=False)
    numpy_spy = _install_numpy_spy(monkeypatch, block=False)

    runner = runner_factory()
    plan = _plan(runner, materialize=True, **plan_kwargs)

    assert len(rope_calls) >= 1, "rope guard never fired on the materialized path"
    rope_kwargs = rope_calls[0][1]
    assert rope_kwargs["max_positions"] == plan.max_positions
    assert rope_kwargs["rotary_dim"] == runner.weights.config.rope_dimension_count
    assert "arange" in numpy_spy.calls
    assert "zeros" in numpy_spy.calls

    expected = {
        "conv_zero": plan.conv_zero_nbytes,
        "recurrent_zero": plan.recurrent_zero_nbytes,
        "block_table_arr": plan.block_table_nbytes,
        "position_host": plan.position_nbytes,
        "context_host": plan.context_nbytes,
        "cos_arr": plan.cos_nbytes,
        "sin_arr": plan.sin_nbytes,
    }
    for field_name, nbytes in expected.items():
        array = getattr(plan, field_name)
        assert array is not None, field_name
        assert int(array.nbytes) == int(nbytes), field_name

    assert plan.block_table_arr.dtype == np.int32
    assert plan.position_host.dtype == np.int64
    assert plan.context_host.dtype == np.int64
    assert plan.cos_arr.dtype == np.float32
    assert plan.sin_arr.dtype == np.float32
    cfg = runner.weights.config
    assert plan.cos_arr.shape == (plan.max_positions, cfg.rope_dimension_count)
    assert plan.conv_zero.dtype == np.float32
    assert plan.conv_zero.shape == (
        plan.slot_count,
        _GEOMETRY["linear_qkv_width"],
        cfg.ssm_conv_kernel,
    )
    expected_recurrent_dtype = np.float16 if runner.fp16_recurrent_state else np.float32
    assert plan.recurrent_zero.dtype == expected_recurrent_dtype
    if plan.slot_count == 1:
        assert plan.block_table_arr.shape == (plan.block_count,)
    else:
        assert plan.block_table_arr.shape == (plan.slot_count, plan.block_count)


def test_materialized_plan_builds_host_arrays_once(monkeypatch) -> None:
    """The materialized form must build its host arrays once, not twice.

    Repeating block-table/position/rope construction after INT8 scale sizing
    would discard the first arrays. Pin the call count so that duplicate work
    cannot hide behind correct returned values.
    """

    rope_calls = _install_rope_spy(monkeypatch, block=False)

    _plan(_dense_runner(), materialize=True, max_sequence_length=8192)

    assert len(rope_calls) == 1


def test_array_and_rope_guards_can_fire() -> None:
    """Both guards must be live, so a size-only pass is not vacuous.

    With ``block=False`` the spies record instead of rejecting, which is what the
    materialized positive control above relies on. Here we confirm the same
    guards reject when armed, using the materialized path as the trigger.
    """

    patch = pytest.MonkeyPatch()
    try:
        numpy_spy = _install_numpy_spy(patch, block=True)
        _install_rope_spy(patch, block=True)
        with pytest.raises(AssertionError, match="host array"):
            _plan(_dense_runner(), materialize=True, max_sequence_length=512)
        assert numpy_spy.calls, "numpy guard recorded no constructor calls"
    finally:
        patch.undo()


# ---------------------------------------------------------------------------
# Owner ordering and metadata/workspace decomposition
# ---------------------------------------------------------------------------


@_CASE_PARAMS
def test_owner_sizes_end_with_metadata_then_workspace_fields(runner_factory, plan_kwargs) -> None:
    plan = _plan(runner_factory(), materialize=False, **plan_kwargs)

    metadata = (
        plan.block_table_nbytes,
        plan.position_nbytes,
        plan.context_nbytes,
        plan.cos_nbytes,
        plan.sin_nbytes,
    )
    tail = plan.owner_sizes[-len(metadata) - len(plan.field_sizes) :]
    assert tail[: len(metadata)] == metadata
    assert tail[len(metadata) :] == tuple(plan.field_sizes.values())
    assert plan.metadata_bytes == sum(metadata)
    assert plan.workspace_bytes == sum(plan.field_sizes.values())


def test_recurrent_state_precision_changes_only_its_own_bytes() -> None:
    """fp16 recurrent state must halve exactly the recurrent owner bytes."""

    kwargs = {"max_sequence_length": 16_384, "kv_storage_dtype": _INT8, "kv_scale_dtype": "fp32"}
    fp32_runner = _dense_runner()
    fp16_plan = _plan(_dense_runner_fp16_recurrent(), materialize=False, **kwargs)
    fp32_plan = _plan(fp32_runner, materialize=False, **kwargs)

    assert fp16_plan.recurrent_zero_nbytes * 2 == fp32_plan.recurrent_zero_nbytes
    assert fp16_plan.conv_zero_nbytes == fp32_plan.conv_zero_nbytes
    # Each linear-attention layer owns a recurrent-zero buffer, and the workspace
    # keeps one more as ``linear_recurrent_state_tmp``.
    linear_layers = sum(1 for kind in fp32_runner.weights.config.layer_types if kind == LINEAR_ATTENTION)
    saved = fp32_plan.recurrent_zero_nbytes - fp16_plan.recurrent_zero_nbytes
    assert fp32_plan.dedicated_bytes - fp16_plan.dedicated_bytes == saved * (linear_layers + 1)
    assert fp32_plan.field_sizes["linear_recurrent_state_tmp"] == fp32_plan.recurrent_zero_nbytes
    assert fp16_plan.field_sizes["linear_recurrent_state_tmp"] == fp16_plan.recurrent_zero_nbytes


def test_size_only_plan_preserves_the_mirror_threshold_decision() -> None:
    """The size-only form must still price the BF16 mirror step at the cutoff."""

    threshold = _mirror_threshold_positions()
    common = {"kv_storage_dtype": _INT8, "kv_scale_dtype": "fp32"}
    runner = _dense_runner()

    at_threshold = _plan(runner, materialize=False, max_sequence_length=threshold, **common)
    below = _plan(runner, materialize=False, max_sequence_length=threshold - 256, **common)
    past = _plan(runner, materialize=False, max_sequence_length=threshold + 1, **common)

    assert at_threshold.max_positions == threshold
    assert at_threshold.short_int8_bf16_mirror is True
    assert at_threshold.kv_mirror_bytes > 0
    assert below.max_positions < threshold
    assert below.short_int8_bf16_mirror is True
    assert below.kv_mirror_bytes > 0
    assert past.max_positions > threshold
    assert past.short_int8_bf16_mirror is False
    assert past.kv_mirror_bytes == 0


# ---------------------------------------------------------------------------
# Repeatability
# ---------------------------------------------------------------------------


def test_size_only_plan_is_repeatable_and_not_a_cached_materialization() -> None:
    runner = _dense_runner()
    kwargs = {
        "max_sequence_length": 8192,
        "max_batch_size": 2,
        "kv_storage_dtype": _INT8,
        "kv_scale_dtype": "fp32",
    }

    first = _plan(runner, materialize=False, **kwargs)
    materialized = _plan(runner, materialize=True, **kwargs)
    second = _plan(runner, materialize=False, **kwargs)

    assert first.to_json_dict() == second.to_json_dict() == materialized.to_json_dict()
    assert first.owner_sizes == second.owner_sizes
    for field_name in _HOST_ARRAY_FIELDS:
        assert getattr(first, field_name) is None, field_name
        assert getattr(second, field_name) is None, field_name


# ---------------------------------------------------------------------------
# Validation still runs in the size-only form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("plan_kwargs", "message"),
    (
        ({"kv_storage_dtype": "fp32"}, "must be bf16 or int8_per_token_head"),
        ({"kv_storage_dtype": "fp8_e4m3"}, "unknown dtype"),
        ({"kv_storage_dtype": _INT8, "kv_scale_dtype": "int8_per_token_head"}, "fp16 or fp32"),
        ({"kv_storage_layout": "tail8"}, "unsupported GGUF resident KV storage layout"),
        ({"kv_scale_granularity": "per_block"}, "scale granularity"),
        ({"max_sequence_length": _DENSE_CONTEXT_LENGTH + 1}, "exceeds GGUF context length"),
        ({"max_batch_size": 0}, "max_batch_size must be positive"),
        (
            {"kv_storage_dtype": _INT8, "int8_bf16_prefix_full_attention_layers": 99},
            "full-attention layer indices",
        ),
    ),
    ids=(
        "bad_kv_storage",
        "unknown_kv_storage_dtype",
        "bad_scale_dtype",
        "bad_layout",
        "bad_granularity",
        "context_overflow",
        "zero_slots",
        "bf16_index_out_of_range",
    ),
)
def test_size_only_plan_still_validates_the_kv_policy(plan_kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        _plan(_dense_runner(), materialize=False, **plan_kwargs)
