"""CPU tests for the producer-derived layer-0 capture descriptors.

The point of these tests is to pin the two failure modes that invalidated the
first layer-0 comparison:

1. a descriptor whose layout is not derived from its producer (wrong dtype or
   width), and
2. a capture that reads a scratch field after an aliasing writer reused its
   arena bytes.

The lifetime table used here is the real
``_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES`` the allocator consumes, so the
alias/lifetime assertions cannot drift away from production.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_gguf_runner import (
    _GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES,
)
from scripts import tp2_layer0_capture as capture

HIDDEN = 5120
QKV_WIDTH = 10240
SSM_INNER = 6144
SSM_GROUPS = 16
SSM_STATE = 128
SSM_HEADS = 48


class _FakeBuffer:
    def __init__(self, ptr: int, nbytes: int) -> None:
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)


def _exact_scratch(rows: int = 64) -> object:
    widths = capture.linear_layer0_widths(
        hidden_size=HIDDEN,
        linear_qkv_width=QKV_WIDTH,
        ssm_inner_size=SSM_INNER,
        ssm_group_count=SSM_GROUPS,
        ssm_state_size=SSM_STATE,
        ssm_time_step_rank=SSM_HEADS,
    )
    fields = {}
    ptr = 0x1000
    for name, (dtype, width) in widths.items():
        nbytes = rows * width * capture.ITEMSIZE[dtype]
        fields[name] = _FakeBuffer(ptr, nbytes)
        ptr += 0x10000
    return SimpleNamespace(**fields)


def _descriptors(rows: int = 64, scratch=None):
    return capture.describe_layer0_linear(
        scratch=scratch or _exact_scratch(rows),
        rows=rows,
        hidden_size=HIDDEN,
        linear_qkv_width=QKV_WIDTH,
        ssm_inner_size=SSM_INNER,
        ssm_group_count=SSM_GROUPS,
        ssm_state_size=SSM_STATE,
        ssm_time_step_rank=SSM_HEADS,
        lifetimes=_GGUF_PREFILL_SCRATCH_DENSE_LIFETIMES,
    )


def _by_name(descriptors):
    return {item.name: item for item in descriptors}


# --------------------------------------------------------------------------
# descriptor validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [True, False, 1.0, 1.5, "1", 0, -3, None])
def test_require_positive_int_rejects_non_positive_integers(value):
    with pytest.raises(capture.CaptureError):
        capture.require_positive_int(value, what="rows")


def test_require_positive_int_accepts_a_real_positive_integer():
    assert capture.require_positive_int(7, what="rows") == 7


def test_descriptor_rejects_a_boolean_row_count():
    with pytest.raises(capture.CaptureError, match="rows"):
        capture.descriptor(
            name="norm",
            producer="attn_norm",
            ptr=0x10,
            rows=True,
            width=HIDDEN,
            dtype=capture.BF16,
            allocated_nbytes=1 << 20,
            lifetime=(("linear", 0, 1),),
        )


def test_descriptor_rejects_an_unknown_dtype():
    with pytest.raises(capture.CaptureError, match="dtype"):
        capture.descriptor(
            name="norm",
            producer="attn_norm",
            ptr=0x10,
            rows=4,
            width=HIDDEN,
            dtype="f16",
            allocated_nbytes=1 << 20,
            lifetime=(("linear", 0, 1),),
        )


def test_descriptor_rejects_a_null_pointer():
    with pytest.raises(capture.CaptureError, match="ptr"):
        capture.descriptor(
            name="norm",
            producer="attn_norm",
            ptr=0,
            rows=4,
            width=HIDDEN,
            dtype=capture.BF16,
            allocated_nbytes=1 << 20,
            lifetime=(("linear", 0, 1),),
        )


def test_descriptor_rejects_a_read_past_the_allocation():
    with pytest.raises(capture.CaptureError, match="allocation"):
        capture.descriptor(
            name="conv_out",
            producer="conv_prefill",
            ptr=0x10,
            rows=64,
            width=QKV_WIDTH,
            dtype=capture.F32,
            allocated_nbytes=64 * QKV_WIDTH * 4 - 1,
            lifetime=(("linear", 2, 5),),
        )


def test_descriptor_rejects_an_empty_or_degenerate_lifetime():
    with pytest.raises(capture.CaptureError, match="lifetime"):
        capture.descriptor(
            name="norm",
            producer="attn_norm",
            ptr=0x10,
            rows=4,
            width=HIDDEN,
            dtype=capture.BF16,
            allocated_nbytes=1 << 20,
            lifetime=(),
        )
    with pytest.raises(capture.CaptureError, match="start < end"):
        capture.descriptor(
            name="norm",
            producer="attn_norm",
            ptr=0x10,
            rows=4,
            width=HIDDEN,
            dtype=capture.BF16,
            allocated_nbytes=1 << 20,
            lifetime=(("linear", 3, 3),),
        )


def test_descriptor_reports_the_producer_derived_layout():
    item = capture.descriptor(
        name="conv_out",
        producer="conv_prefill",
        ptr=0x2000,
        rows=64,
        width=QKV_WIDTH,
        dtype=capture.F32,
        allocated_nbytes=64 * QKV_WIDTH * 4,
        lifetime=(("linear", 2, 5),),
    )
    assert item.shape == (64, QKV_WIDTH)
    assert item.itemsize == 4
    assert item.nbytes == 64 * QKV_WIDTH * 4
    assert item.to_json()["dtype"] == "f32"


# --------------------------------------------------------------------------
# widths / describe
# --------------------------------------------------------------------------


def test_widths_come_from_the_producer_call_sites():
    widths = capture.linear_layer0_widths(
        hidden_size=HIDDEN,
        linear_qkv_width=QKV_WIDTH,
        ssm_inner_size=SSM_INNER,
        ssm_group_count=SSM_GROUPS,
        ssm_state_size=SSM_STATE,
        ssm_time_step_rank=SSM_HEADS,
    )
    assert widths["norm"] == (capture.BF16, HIDDEN)
    assert widths["linear_qkv"] == (capture.BF16, QKV_WIDTH)
    assert widths["linear_qkv_f32"] == (capture.F32, QKV_WIDTH)
    assert widths["conv_out"] == (capture.F32, QKV_WIDTH)
    assert widths["prefill_query"] == (capture.F32, SSM_GROUPS * SSM_STATE)
    assert widths["prefill_value"] == (capture.F32, SSM_INNER)
    assert widths["recurrent_out"] == (capture.F32, SSM_INNER)
    assert widths["recurrent_bf16"] == (capture.BF16, SSM_INNER)
    assert widths["attn_out"] == (capture.BF16, HIDDEN)


def test_describe_layer0_linear_uses_the_producer_row_count_not_capacity():
    scratch = _exact_scratch(rows=200)
    items = _descriptors(rows=64, scratch=scratch)
    by_name = _by_name(items)
    assert by_name["norm"].rows == 64
    assert by_name["norm"].shape == (64, HIDDEN)
    # the allocation is capacity-sized, the capture is not
    assert by_name["norm"].allocated_nbytes == 200 * HIDDEN * 2


def test_describe_layer0_linear_fails_closed_on_an_over_read():
    scratch = _exact_scratch(rows=64)
    scratch.conv_out = _FakeBuffer(0x9000, 64 * QKV_WIDTH * 4 - 8)
    with pytest.raises(capture.CaptureError, match="conv_out"):
        _descriptors(scratch=scratch)


def test_describe_layer0_linear_requires_every_producer_field():
    scratch = _exact_scratch(rows=64)
    del scratch.recurrent_bf16
    with pytest.raises(capture.CaptureError, match="recurrent_bf16"):
        _descriptors(scratch=scratch)
    assert not hasattr(scratch, "recurrent_bf16")


# --------------------------------------------------------------------------
# alias / lifetime ordering
# --------------------------------------------------------------------------


def test_lifetimes_overlap_is_half_open():
    assert capture.lifetimes_overlap((("linear", 2, 5),), (("linear", 4, 5),))
    assert not capture.lifetimes_overlap((("linear", 0, 2),), (("linear", 2, 5),))
    assert not capture.lifetimes_overlap((("linear", 2, 5),), (("full", 2, 5),))
    # the endpoints are closed on the inside: (5, 6) and (5, 7) are both live at 5
    assert capture.lifetimes_overlap((("linear", 5, 6),), (("linear", 5, 7),))


def test_aliasing_pairs_match_the_arena_coloring_rules():
    pairs = set(capture.aliasing_pairs(_descriptors(), route="linear"))

    # stage-disjoint on the linear route -> the arena may reuse these bytes
    assert ("linear_qkv", "conv_out") in pairs
    assert ("linear_qkv", "recurrent_bf16") in pairs
    assert ("norm", "attn_out") in pairs
    assert ("recurrent_out", "attn_out") in pairs

    # overlapping lifetimes -> separate arena bytes
    assert ("conv_out", "recurrent_out") not in pairs
    assert ("recurrent_bf16", "attn_out") not in pairs
    assert ("linear_qkv", "linear_z") not in pairs
    assert ("norm", "linear_qkv") not in pairs


def test_capture_plan_follows_the_helper_producer_order():
    steps = capture.build_capture_plan(
        _descriptors(), capture.linear_layer0_producers(), route="linear"
    )
    assert [step.producer for step in steps] == [
        "attn_norm",
        "qkv_gate",
        "alpha_beta",
        "qkv_bf16_to_f32",
        "conv_prefill",
        "gdn_prepare",
        "gdn_recurrent",
        "gdn_rmsnorm_gate",
        "ssm_out",
    ]
    assert steps[0].buffers == ("norm",)
    assert steps[1].buffers == ("linear_qkv", "linear_z")
    assert steps[5].buffers == (
        "prefill_query",
        "prefill_key",
        "prefill_value",
        "prefill_beta",
        "prefill_decay",
    )
    assert steps[-1].buffers == ("attn_out",)


def test_capture_plan_rejects_an_unknown_producer():
    producers = capture.linear_layer0_producers()[:-1]
    with pytest.raises(capture.CapturePlanError, match="unknown producer"):
        capture.build_capture_plan(_descriptors(), producers, route="linear")


def test_capture_plan_rejects_a_duplicate_producer():
    producers = capture.linear_layer0_producers() + (
        capture.Producer("attn_norm", "again", ("norm",)),
    )
    with pytest.raises(capture.CapturePlanError, match="duplicate producer"):
        capture.build_capture_plan(_descriptors(), producers, route="linear")


def test_sequential_capture_plan_is_valid():
    items = _descriptors()
    producers = capture.linear_layer0_producers()
    steps = capture.build_capture_plan(items, producers, route="linear")
    write_index = {
        buffer: index
        for index, producer in enumerate(producers)
        for buffer in producer.buffers
    }
    capture_index = dict(write_index)
    # sanity: the plan validator accepts the per-producer capture
    capture.validate_capture_order(
        items, capture_index=capture_index, write_index=write_index, route="linear"
    )
    assert [step.producer for step in steps][0] == "attn_norm"


def test_end_of_layer_bulk_read_is_rejected():
    """The historical capture read every field once after the helper returned."""

    items = _descriptors()
    producers = capture.linear_layer0_producers()
    write_index = {
        buffer: index
        for index, producer in enumerate(producers)
        for buffer in producer.buffers
    }
    with pytest.raises(capture.CapturePlanError) as excinfo:
        capture.validate_capture_order(
            items,
            capture_index=capture.bulk_end_of_layer_capture_index(items),
            write_index=write_index,
            route="linear",
        )
    message = str(excinfo.value)
    assert "share arena bytes" in message
    # the first stage-disjoint pair in descriptor order: norm (0,1) is dead
    # before the QKV->f32 cast writes linear_qkv_f32 (1,3)
    assert "norm" in message
    assert "linear_qkv_f32" in message


def test_end_of_layer_bulk_read_is_rejected_for_the_real_lifetime_table():
    """Every stage-disjoint linear pair must be reported, not just one."""

    items = _descriptors()
    producers = capture.linear_layer0_producers()
    write_index = {
        buffer: index
        for index, producer in enumerate(producers)
        for buffer in producer.buffers
    }
    rejected = 0
    for lhs, rhs in capture.aliasing_pairs(items, route="linear"):
        first, second = sorted((lhs, rhs), key=lambda name: write_index[name])
        assert capture.bulk_end_of_layer_capture_index(items)[first] >= write_index[second]
        rejected += 1
    assert rejected >= 10


# --------------------------------------------------------------------------
# hook installation (no GPU: only the wrapper surface is exercised)
# --------------------------------------------------------------------------

_HOOKED_MODULE_FUNCTIONS = (
    "launch_gguf_linear",
    "launch_gguf_linear_pair",
    "bf16_to_f32",
    "f32_to_bf16",
    "_try_launch_dense_q8_pair_dp4a",
    "_try_launch_dense_q8_single_dp4a",
)


def test_install_hooks_wraps_the_real_producer_surface():
    """A typo in a wrapped attribute must fail on CPU, not after two model loads."""

    import hipengine.runtime.qwen35_gguf_runner as qr
    from scripts import tp2_bulk_vs_resident_layer0 as diag

    originals = {
        name: getattr(qr, name) for name in _HOOKED_MODULE_FUNCTIONS
    }
    runner_cls = qr.Qwen35GGUFFullStackRunner
    method_names = (
        "_run_attention_norm_rows",
        "_run_linear_attention_alpha_beta_rows",
        "_linear_attn_conv_prefill_kernel",
        "_run_gdn_prefill",
        "_run_linear_attention_prefill_attn_rows",
    )
    originals_methods = {name: getattr(runner_cls, name) for name in method_names}
    try:
        diag._install_hooks(qr, {}, {}, set())
        for name in _HOOKED_MODULE_FUNCTIONS:
            assert getattr(qr, name) is not originals[name], name
        for name in method_names:
            assert getattr(runner_cls, name) is not originals_methods[name], name
        # the layer-0 producer order the hook installs must be the one the
        # capture plan validates against
        assert [p.name for p in capture.linear_layer0_producers()] == [
            "attn_norm",
            "qkv_gate",
            "alpha_beta",
            "qkv_bf16_to_f32",
            "conv_prefill",
            "gdn_prepare",
            "gdn_recurrent",
            "gdn_rmsnorm_gate",
            "ssm_out",
        ]
    finally:
        for name, original in originals.items():
            setattr(qr, name, original)
        for name, original in originals_methods.items():
            setattr(runner_cls, name, original)
