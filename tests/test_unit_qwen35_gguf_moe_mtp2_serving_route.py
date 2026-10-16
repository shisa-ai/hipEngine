"""The Qwen3.6 MoE sampled-MTP serving route.

The MoE plugin's implementation declaration admits the sampled mode, so a
temperature/top-p/min-p request keeps typed MTP intent instead of being sent to
K0 (autoregressive) before provider mutation. This module pins the route
boundaries the serving checklist names: a normal request and a streaming
request reach MTP, a sampled request reaches MTP, and ``speculative_mtp: false``
reaches the autoregressive default. It reads the shipped plugin declaration, so
a metadata-only change that no runtime can execute would not pass.
"""

from __future__ import annotations

from types import SimpleNamespace

from hipengine.models.qwen35 import Qwen35MoeGGUFModel
from hipengine.server.api import (
    _SPECULATIVE_MTP_BATCH_ROUTE,
    _SPECULATIVE_MTP_DEFAULT_ROUTE,
    _speculative_mtp_route_for_request,
)

from tests.test_unit_speculative_sampling_modes import (
    _params,
    _serving_engine,
    _server_config,
)


def _moe_engine():
    """An engine advertising exactly the modes the MoE plugin declares."""

    modes = tuple(
        mode
        for declaration in Qwen35MoeGGUFModel().speculative_mtp_serving_implementations
        for mode in declaration.sampling_modes
    )
    return _serving_engine(*modes)


def _request(*, explicit=True, stream=False):
    return SimpleNamespace(speculative_mtp=explicit, stream=stream)


def _route(*, explicit=True, stream=False, sampling=None, mode="auto"):
    return _speculative_mtp_route_for_request(
        _server_config(speculative_mtp_serving=mode),
        _request(explicit=explicit, stream=stream),
        engine=_moe_engine(),
        sampling=_params() if sampling is None else sampling,
    )


def test_moe_plugin_declares_the_sampled_mode() -> None:
    declaration = Qwen35MoeGGUFModel().speculative_mtp_serving_implementations[0]
    assert "sampled" in declaration.sampling_modes
    assert "greedy_fast" in declaration.sampling_modes


def test_normal_request_reaches_mtp() -> None:
    assert _route(explicit=True) == _SPECULATIVE_MTP_BATCH_ROUTE


def test_streaming_request_reaches_the_same_mtp_route() -> None:
    """Streaming is not a separate route; it rides the resolved MTP route."""

    assert _route(explicit=True, stream=True) == _SPECULATIVE_MTP_BATCH_ROUTE
    assert _route(explicit=True, stream=True, sampling=_params(temperature=0.7)) == (
        _SPECULATIVE_MTP_BATCH_ROUTE
    )


def test_sampled_request_reaches_mtp() -> None:
    assert _route(sampling=_params(temperature=0.7)) == _SPECULATIVE_MTP_BATCH_ROUTE
    assert (
        _route(sampling=_params(temperature=0.7, top_p=0.9, min_p=0.05))
        == _SPECULATIVE_MTP_BATCH_ROUTE
    )


def test_explicit_false_reaches_the_autoregressive_default() -> None:
    assert _route(explicit=False) == _SPECULATIVE_MTP_DEFAULT_ROUTE
    assert (
        _route(explicit=False, sampling=_params(temperature=0.7))
        == _SPECULATIVE_MTP_DEFAULT_ROUTE
    )
    assert (
        _route(explicit=False, stream=True)
        == _SPECULATIVE_MTP_DEFAULT_ROUTE
    )
