"""CPU tests for when the Inkling 2 GGUF device-side route selection is active (no GPU)."""

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
