"""CPU tests for when the Inkling 2 GGUF device-side route selection is active (no GPU)."""

import threading

import pytest

from hipengine.loading.inkling2_gguf_device import Inkling2GGUFDeviceProjections


def _projection(*, requested=True, fast=True):
    projection = Inkling2GGUFDeviceProjections.__new__(Inkling2GGUFDeviceProjections)
    projection._device_routes = None
    projection._device_routes_requested = requested
    projection._mix = lambda *args, **kwargs: None
    # The split activation exists only with the fast expert chain; the selected kernels do not.
    projection._activation = (lambda *args, **kwargs: None) if fast else None
    projection._selected_launches = {"q8_0": object()}
    projection._shared_q8_dot = (lambda *args, **kwargs: None) if fast else None
    return projection


def test_device_routes_are_inert_without_the_fast_expert_chain():
    # The strict profile has no shared Q8 dot and no split activation. Selecting routes on the
    # device there reached a kernel that is None and failed mid-generation with a TypeError.
    assert _projection(fast=False).device_routes_enabled() is False
    assert _projection(fast=False).device_routes_supported(_visit()) is False


def test_device_routes_follow_the_request_with_the_fast_expert_chain():
    assert _projection().device_routes_enabled() is True
    assert _projection(requested=False).device_routes_enabled() is False


def _visit():
    from types import SimpleNamespace

    return SimpleNamespace(weight_layer=0)


def test_device_routes_report_a_named_capability_miss_for_unsupported_banks():
    from types import SimpleNamespace

    from hipengine.quant.gguf import GGMLQuantizationType as Q

    projection = _projection()
    layer = SimpleNamespace(
        gate_up=SimpleNamespace(weight=SimpleNamespace(ggml_type=Q.Q4_K, shape=(128, 4608, 2304))),
        down=SimpleNamespace(weight=SimpleNamespace(ggml_type=Q.Q4_K, shape=(128, 2304, 1152))),
    )
    projection._weights = SimpleNamespace(mapping=SimpleNamespace(layers=[layer]))
    assert projection.device_routes_supported(_visit()) is False


def _routed_projection():
    """A projection whose kernels, scratch and copies only record what the route chain asked of them."""
    from types import SimpleNamespace

    from hipengine.quant.gguf import GGMLQuantizationType as Q

    calls, copies = [], []
    projection = Inkling2GGUFDeviceProjections.__new__(Inkling2GGUFDeviceProjections)
    projection._lock = threading.Lock()
    projection._closed = False
    projection._device_routes = None
    projection._device_routes_requested = True
    gate = SimpleNamespace(weight=SimpleNamespace(ggml_type=Q.NVFP4, shape=(128, 2304, 2304)))
    down = SimpleNamespace(weight=SimpleNamespace(ggml_type=Q.NVFP4, shape=(128, 2304, 1152)))
    layer = SimpleNamespace(gate_up=gate, down=down)
    projection._weights = SimpleNamespace(mapping=SimpleNamespace(layers=[layer], gate_cap=8.0, up_cap=8.0))
    banks = {id(gate): (SimpleNamespace(ptr=0x1000), None), id(down): (SimpleNamespace(ptr=0x2000), None)}
    projection._bank = lambda bank: banks[id(bank)]
    projection._selected_launches = {Q.NVFP4: lambda *args: calls.append(("selected", *args))}
    projection._shared_selected_launches = {
        Q.NVFP4: lambda *args, **kwargs: calls.append(("shared", *args, kwargs["routes_per_input"]))}
    projection._replicate = lambda *args: calls.append(("replicate", *args))
    projection._activation = lambda *args: calls.append(("activation", *args))
    projection._mix = lambda *args, **kwargs: calls.append(("mix", *args, kwargs))
    pointers = {"_device_route_out": 0x9000, "_device_route_interim": 0xA000, "_device_route_hidden_rep": 0xB000}
    projection._scratch = lambda name, nbytes: SimpleNamespace(ptr=pointers[name])
    projection.copy_device = lambda dst, src, nbytes: copies.append((dst, src, nbytes))
    # The Q8-dot chain belongs to the eager visit; it is not what the device route runs.
    projection._q8_dot = projection._shared_q8_dot = projection._split_q8_dot = None
    return projection, calls, copies


def test_device_route_experts_use_the_selected_f32_kernels_with_the_route_scales():
    """The Q8-dot chain put the opt-in path outside the section 2.11 envelope (2026-11-25)."""
    projection, calls, copies = _routed_projection()
    route = {"ids": 0x3000, "mix": 0x4000, "gate_scales": 0x5000, "down_scales": 0x6000,
             "top_k": 8, "experts": 128, "routes": 8}
    out = projection.project_experts_device_routes(_visit(), hidden_ptr=0x7000, route=route)
    assert out == 0x9000
    active = 0xA000 + 8 * 2304 * 4
    # Every route reads the one resident hidden row through the shared-input projection: no copy.
    assert copies == []
    assert calls == [
        ("shared", 0x7000, 0x1000, 0xA000, 0x3000, 0x5000, 8, 128, 2304, 2304, 8),
        ("activation", 0xA000, active, 8, 1152, 8.0, 8.0),
        ("selected", active, 0x2000, 0xA000, 0x3000, 0x6000, 8, 128, 1152, 2304),
        ("mix", 0xA000, 0x4000, 0x9000, 8, 2304, {"tokens": 1}),
    ]
    assert projection.device_routes_supported(_visit()) is True


def test_device_route_experts_replicate_the_row_in_one_launch_without_a_shared_projection():
    projection, calls, copies = _routed_projection()
    projection._shared_selected_launches = {}
    route = {"ids": 0x3000, "mix": 0x4000, "gate_scales": 0, "down_scales": 0,
             "top_k": 8, "experts": 128, "routes": 8}
    projection.project_experts_device_routes(_visit(), hidden_ptr=0x7000, route=route)
    assert copies == []
    assert calls[0] == ("replicate", 0x7000, 0xB000, 2304, 8)
    assert calls[1] == ("selected", 0xB000, 0x1000, 0xA000, 0x3000, 0, 8, 128, 2304, 2304)


def test_device_route_experts_copy_per_route_when_nothing_else_is_registered():
    projection, calls, copies = _routed_projection()
    projection._shared_selected_launches = {}
    projection._replicate = None
    row = 2304 * 4
    projection.project_experts_device_routes(
        _visit(), hidden_ptr=0x7000,
        route={"ids": 0x3000, "mix": 0x4000, "gate_scales": 0, "down_scales": 0,
               "top_k": 8, "experts": 128, "routes": 8})
    assert copies == [(0xB000 + index * row, 0x7000, row) for index in range(8)]
    assert calls[0][0] == "selected" and calls[0][1] == 0xB000


def test_device_route_experts_need_the_selected_kernels_for_every_bank():
    projection, calls, _copies = _routed_projection()
    projection._selected_launches = {}
    with pytest.raises(ValueError, match="does not support"):
        projection.project_experts_device_routes(
            _visit(), hidden_ptr=0x7000,
            route={"ids": 0x3000, "mix": 0x4000, "gate_scales": 0, "down_scales": 0,
                   "top_k": 8, "experts": 128, "routes": 8})
    assert calls == [] and projection.device_routes_supported(_visit()) is False
