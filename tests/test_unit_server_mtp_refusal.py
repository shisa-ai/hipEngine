"""An explicit MTP request refuses by name; an automatic one falls back to AR.

The readiness checklist requires that an explicit request either runs or raises
a named capability error, and that an automatic one may take a supported
fallback -- with no silent downgrade of the explicit case.

``_speculative_mtp_route_for_request`` (``hipengine/server/api.py:14290``) is
where that split is decided. Its route-selection branches were covered by
``test_unit_speculative_sampling_modes.py``; its two **refusal** branches were
not:

- a server with ``speculative_mtp_serving="off"`` raises
  ``OpenAIHTTPError(400, code="unsupported_parameter")`` for an explicit
  request;
- an engine that cannot run MTP raises
  ``OpenAIHTTPError(501, error_type="unsupported_feature",
  code="unsupported_feature")`` for an explicit request.

In both conditions an automatic request returns the default (autoregressive)
route instead, and a client that asked for ``speculative_mtp: false`` gets AR
regardless of what the server and engine could do.

The function is pure over ``(config, request, engine, sampling)``, so the whole
split is CPU-testable.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hipengine.server.api import (
    OpenAIHTTPError,
    _SPECULATIVE_MTP_DEFAULT_ROUTE,
    _speculative_mtp_route_for_request,
)

from tests.test_unit_speculative_sampling_modes import (
    _params,
    _serving_engine,
    _serving_request,
    _server_config,
)


def _route(*, mode: str, explicit, engine, sampling=None) -> str:
    return _speculative_mtp_route_for_request(
        _server_config(speculative_mtp_serving=mode),
        _serving_request(explicit),
        engine=engine,
        sampling=_params() if sampling is None else sampling,
    )


def _incapable_engine():
    """An engine with no MTP entry point at all."""

    return SimpleNamespace(
        supports_speculative_mtp=False,
        speculative_mtp_sampling_modes=(),
    )


def test_explicit_request_is_refused_by_name_when_serving_is_off() -> None:
    """A disabled server names the parameter instead of quietly serving AR."""

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(mode="off", explicit=True, engine=_serving_engine("greedy_fast"))

    error = excinfo.value
    assert error.status_code == 400
    assert error.code == "unsupported_parameter"
    assert error.param == "speculative_mtp"
    assert "disabled" in error.message


def test_explicit_request_is_refused_by_name_when_the_engine_cannot_run_mtp() -> None:
    """An incapable engine names the capability instead of quietly serving AR."""

    with pytest.raises(OpenAIHTTPError) as excinfo:
        _route(mode="auto", explicit=True, engine=_incapable_engine())

    error = excinfo.value
    assert error.status_code == 501
    assert error.code == "unsupported_feature"
    assert error.error_type == "unsupported_feature"
    assert error.param == "speculative_mtp"
    assert "not supported" in error.message


@pytest.mark.parametrize(
    "mode,engine_factory",
    [
        pytest.param("off", lambda: _serving_engine("greedy_fast"), id="serving-off"),
        pytest.param("auto", _incapable_engine, id="engine-incapable"),
        pytest.param("enabled", _incapable_engine, id="enabled-but-incapable"),
    ],
)
def test_automatic_request_falls_back_to_ar_without_raising(
    mode: str, engine_factory
) -> None:
    """The same conditions that refuse an explicit request serve an automatic one.

    This is the permitted fallback arm: the server keeps a completion it can
    produce rather than failing a request that never asked for MTP.
    """

    assert (
        _route(mode=mode, explicit=None, engine=engine_factory())
        == _SPECULATIVE_MTP_DEFAULT_ROUTE
    )


@pytest.mark.parametrize(
    "mode,engine_factory",
    [
        pytest.param("off", lambda: _serving_engine("greedy_fast"), id="serving-off"),
        pytest.param("auto", _incapable_engine, id="engine-incapable"),
    ],
)
def test_client_opt_out_beats_a_refusal_condition(mode: str, engine_factory) -> None:
    """``speculative_mtp: false`` is honoured, not turned into an error.

    A client that declined MTP asked for nothing the server must refuse, so
    the same configuration that raises for an explicit ``true`` returns AR.
    """

    assert (
        _route(mode=mode, explicit=False, engine=engine_factory())
        == _SPECULATIVE_MTP_DEFAULT_ROUTE
    )


def test_explicit_request_never_silently_degrades_to_ar() -> None:
    """The refusal conditions raise rather than returning the AR route.

    Stated as its own case because it is the property the checklist names: a
    client that explicitly asked for MTP must not receive an autoregressive
    completion with no error. If a future change made these branches return the
    default route, the two refusal cases above would still pass on their own
    message assertions only if they kept raising, so this case pins the shape
    of the failure directly.
    """

    for mode, engine in (
        ("off", _serving_engine("greedy_fast")),
        ("auto", _incapable_engine()),
    ):
        with pytest.raises(OpenAIHTTPError):
            _route(mode=mode, explicit=True, engine=engine)
