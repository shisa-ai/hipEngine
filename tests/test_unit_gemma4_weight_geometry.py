"""CPU regressions for Gemma 4 resident weight geometry.

Two defects are pinned without any HIP execution:

* the resident planner names the untied output projection slot ``lm_head``
  while the runtime loader looked it up as ``output``, so an untied artifact
  failed to load; and
* ``gemma4_text_config_from_gguf`` collapses the per-layer
  ``feed_forward_length`` array to its maximum, which the runner then handed to
  every layer as its dense-MLP width. On an artifact whose layers have
  different widths that dispatches projections at the wrong size and reads out
  of bounds.

Both are checked by driving the real loader over a tiny in-memory GGUF fixture
with the device allocator faked out. No device, no kernel, no ROCm import.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer import (
    gemma4_layer_forward_bf16,
)
from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType
from hipengine.runtime import gemma4 as gemma4_module
from hipengine.runtime.gemma4 import Gemma4Runner
from tests._gemma4_gguf_fixture import (
    FIXTURE_HIDDEN,
    FIXTURE_VOCAB,
    default_fixture_tensors,
    fixture_metadata,
    write_fixture_gguf,
)


# ---------------------------------------------------------------------------
# Fake device layer
# ---------------------------------------------------------------------------


class _FakeResidentWeight:
    """Stand-in for a materialized resident weight, keyed by its real spec."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.allocations: dict[str, object] = {}
        self.free_calls = 0

    def free(self, **kwargs) -> None:
        self.free_calls += 1


class _FakeScalarAllocation:
    def __init__(self, ptr: int) -> None:
        self.buffer = SimpleNamespace(ptr=ptr)
        self.free_calls = 0

    def free(self, **kwargs) -> None:
        self.free_calls += 1


def _install_fake_device(monkeypatch: pytest.MonkeyPatch) -> None:
    next_ptr = 0x10000

    def fake_materialize(reader, spec, **kwargs):
        return _FakeResidentWeight(spec)

    def fake_scalar(name, values, dtype, **kwargs):
        nonlocal next_ptr
        next_ptr += 0x100
        return _FakeScalarAllocation(next_ptr)

    monkeypatch.setattr(
        gemma4_module, "materialize_gemma4_gguf_device_weight", fake_materialize
    )
    monkeypatch.setattr(
        gemma4_module, "load_host_array_to_device_as_dtype", fake_scalar
    )


# ---------------------------------------------------------------------------
# Fixture artifacts
# ---------------------------------------------------------------------------


def _with_dense_widths(
    tensors: list[tuple[str, tuple[int, ...], GGMLQuantizationType]],
    widths: tuple[int, ...],
) -> list[tuple[str, tuple[int, ...], GGMLQuantizationType]]:
    """Reshape each layer's dense MLP tensors to that layer's width."""

    result = []
    for name, shape, qtype in tensors:
        if name.endswith(".ffn_gate.weight") or name.endswith(".ffn_up.weight"):
            layer = int(name.split(".")[1])
            result.append((name, (widths[layer], shape[1]), qtype))
        elif name.endswith(".ffn_down.weight"):
            layer = int(name.split(".")[1])
            result.append((name, (shape[0], widths[layer]), qtype))
        else:
            result.append((name, shape, qtype))
    return result


def _write_artifact(
    tmp_path,
    *,
    widths: tuple[int, ...] | None = None,
    untied: bool = False,
) -> GGUFReader:
    tensors = default_fixture_tensors()
    metadata = fixture_metadata()
    if widths is not None:
        tensors = _with_dense_widths(tensors, widths)
        metadata = [
            (key, 9, (4, list(widths)))
            if key == "gemma4.feed_forward_length"
            else (key, value_type, value)
            for key, value_type, value in metadata
        ]
    if untied:
        tensors.append(
            ("output.weight", (FIXTURE_VOCAB, FIXTURE_HIDDEN), GGMLQuantizationType.Q8_0)
        )
    path = write_fixture_gguf(tmp_path / "geometry.gguf", tensors, metadata)
    return GGUFReader(path)


# ---------------------------------------------------------------------------
# Untied head
# ---------------------------------------------------------------------------


def test_untied_artifact_loads_the_planner_lm_head_slot(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The planner emits ``lm_head``; the loader must ask for that slot."""

    reader = _write_artifact(tmp_path, untied=True)
    _install_fake_device(monkeypatch)

    weights = gemma4_module.load_gemma4_device_weights(reader)
    try:
        assert weights.config.tie_word_embeddings is False
        assert weights.lm_head is not None
        assert weights.lm_head.spec.slot_path == "lm_head"
        assert weights.lm_head.spec.source.name == "output.weight"
    finally:
        weights.free()


def test_tied_artifact_still_has_no_lm_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The untied fix must not invent a head on a tied artifact."""

    reader = _write_artifact(tmp_path)
    _install_fake_device(monkeypatch)

    weights = gemma4_module.load_gemma4_device_weights(reader)
    try:
        assert weights.config.tie_word_embeddings is True
        assert weights.lm_head is None
    finally:
        weights.free()


# ---------------------------------------------------------------------------
# Unequal-width layers
# ---------------------------------------------------------------------------


def test_loader_preserves_each_layers_dense_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The loader records per-layer widths, not one maximum."""

    widths = (192, 320)
    reader = _write_artifact(tmp_path, widths=widths)
    _install_fake_device(monkeypatch)

    weights = gemma4_module.load_gemma4_device_weights(reader)
    try:
        assert weights.dense_intermediate == widths
        # The collapsed config value is still the maximum; it is no longer what
        # the runner sizes layers from.
        assert weights.config.intermediate_size == max(widths)
    finally:
        weights.free()


def _install_fake_runner_allocator(monkeypatch: pytest.MonkeyPatch) -> list[DeviceBuffer]:
    allocated: list[DeviceBuffer] = []
    next_ptr = 0x20000

    def fake_malloc(nbytes: int) -> DeviceBuffer:
        nonlocal next_ptr
        buffer = DeviceBuffer(next_ptr, nbytes)
        next_ptr += nbytes + 0x1000
        allocated.append(buffer)
        return buffer

    monkeypatch.setattr(gemma4_module, "malloc", fake_malloc)
    monkeypatch.setattr(gemma4_module, "free", lambda buffer: None)
    return allocated


def test_runner_sizes_each_scratch_from_its_own_layer_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A narrow layer must not be sized from the widest layer."""

    widths = (192, 320)
    reader = _write_artifact(tmp_path, widths=widths)
    _install_fake_device(monkeypatch)
    _install_fake_runner_allocator(monkeypatch)

    weights = gemma4_module.load_gemma4_device_weights(reader)
    runner = Gemma4Runner(weights=weights, capacity=32, max_block=4)
    try:
        got = tuple(scratch.dense_intermediate for scratch in runner._scratches)
        assert got == widths
    finally:
        runner.close()


def _install_layer_dispatch_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, int, int]]:
    """Replace every layer kernel with a no-op and record dense projections."""

    import hipengine.kernels.hip_gfx1100.gemma4.gemma4_layer as layer_module

    calls: list[tuple[object, int, int]] = []

    def fake_project(x_ptr, weight, out_ptr, rows, in_features, out_features, *, stream=0):
        calls.append((weight, in_features, out_features))

    monkeypatch.setattr(layer_module, "gemma4_project", fake_project)
    for name in (
        "gemma4_rmsnorm_f32w_bf16",
        "gemma4_rmsnorm_weightless_bf16",
        "gemma4_head_rmsnorm_f32w_bf16",
        "gemma4_partial_rotary_bf16",
        "gemma4_attention_prefill_bf16",
        "gemma4_add_rmsnorm_scale_bf16",
        "gemma4_gelu_tanh_mul_bf16",
        "gemma4_router_topk_bf16",
        "gemma4_experts_forward_bf16",
        # D6 tail fold: the three-kernel chain (branch_add +
        # add_rmsnorm_scale + post_ffw_norm_1) collapsed into one launch, so
        # the recorder follows the new call set -- a stale name here raises
        # AttributeError, and a missing one faults on the fake buffers.
        "gemma4_dense_combine_rmsnorm_scale_bf16",
        # Added when the fused q/k/v projection landed: this recorder fakes
        # malloc to hand out pointers in the 0x30000 range, so any kernel that
        # actually launches writes to unmapped GPU memory. Every other real
        # kernel in the forward is already no-op'd here; the split joined the
        # forward later and was missed, which surfaced as a teardown-time
        # "Memory access fault ... on address 0x31000" -- the second fake
        # buffer -- rather than as a failing assertion.
        "gemma4_qkv_split_bf16",
    ):
        monkeypatch.setattr(layer_module, name, lambda *args, **kwargs: None)

    next_ptr = 0x30000

    def fake_malloc(nbytes: int) -> DeviceBuffer:
        nonlocal next_ptr
        buffer = DeviceBuffer(next_ptr, nbytes)
        next_ptr += nbytes + 0x1000
        return buffer

    monkeypatch.setattr(layer_module, "malloc", fake_malloc)
    monkeypatch.setattr(layer_module, "hip_free", lambda *args, **kwargs: None)
    return calls


def test_layer_forward_dispatches_the_layers_own_dense_width(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Run the real layer forward on the runner's scratch and read the dims.

    ``gemma4_layer_forward_bf16`` takes the dense-MLP width from the scratch it
    is handed, so this exercises the exact ``out_features``/``in_features`` the
    kernels would receive for each layer.
    """

    widths = (192, 320)
    reader = _write_artifact(tmp_path, widths=widths)
    _install_fake_device(monkeypatch)
    _install_fake_runner_allocator(monkeypatch)
    calls = _install_layer_dispatch_recorder(monkeypatch)

    weights = gemma4_module.load_gemma4_device_weights(reader)
    runner = Gemma4Runner(weights=weights, capacity=32, max_block=4)
    try:
        for index, layer in enumerate(runner.weights.layers):
            gemma4_layer_forward_bf16(
                0x1,
                0x2,
                0x3,
                0x4,
                layer,
                scratch=runner._scratches[index],
                rows=1,
            )
    finally:
        runner.close()

    for index, layer in enumerate(weights.layers):
        gate_up = [call for call in calls if call[0] is layer.mlp_gate_up_proj]
        down = [call for call in calls if call[0] is layer.mlp_down_proj]
        assert gate_up, f"layer {index} never dispatched its fused gate/up projection"
        assert down, f"layer {index} never dispatched its down projection"
        # One launch carries both halves, so its out_features spans twice the
        # layer's dense width while the down projection still reads that width.
        assert gate_up[0][2] == 2 * widths[index]
        assert down[0][1] == widths[index]
