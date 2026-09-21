"""Error status and retryability must match the taxonomy the API advertises.

The readiness checklist requires that a busy/rejected/unavailable error keeps a
correct status and retryability, and that an unsupported explicit request is a
501 that is not retryable.

Three of the four acceptance conditions are decided by pure functions:

- engine closed -> ``engine_unavailable`` -> 503, retryable;
- over capacity -> ``engine_busy`` -> 429, retryable, with ``Retry-After``;
- unsupported explicit request -> ``unsupported_feature`` -> 501, not
  retryable.

``_ERROR_TAXONOMY`` (``hipengine/server/api.py:664``) is the single table those
decisions read, and ``_error_extension`` is what a client actually sees. Neither
had unit coverage; the only existing references were in
``test_integration_server_api.py``, which needs a live server.

The fourth condition -- a concurrent ``/health`` or ``/ready`` returning within
a bounded time while MTP holds the GPU -- is a live-server property and is not
covered here.
"""

from __future__ import annotations

import pytest

from hipengine.generation.engine_loop import GenerationAdmissionRejected
from hipengine.server.api import (
    OpenAIHTTPError,
    _canonical_error_code,
    _ERROR_CODE_ALIASES,
    _ERROR_TAXONOMY,
    _error_extension,
    _error_taxonomy_manifest,
    _generation_admission_http_error,
)


def _rejection(**overrides) -> GenerationAdmissionRejected:
    values = {
        "message": "kv pool capacity exhausted for 4 rows",
        "resource": "kv_pool_pages",
        "request_id": 7,
        "requested_units": 64,
        "current_units": 4096,
        "capacity_units": 4096,
    }
    values.update(overrides)
    return GenerationAdmissionRejected(**values)


# --- The three acceptance conditions -------------------------------------


def test_engine_closed_is_a_typed_retryable_503() -> None:
    """Condition: engine-closed returns the typed 503."""

    metadata = _ERROR_TAXONOMY["engine_unavailable"]
    assert metadata["status_code"] == 503
    assert metadata["retryable"] is True

    extension = _error_extension(503, "engine_unavailable")
    assert extension == {
        "code": "engine_unavailable",
        "status_code": 503,
        "retryable": True,
    }


def test_over_capacity_is_a_retryable_429_carrying_retry_after() -> None:
    """Condition: an over-capacity request returns a correct status and Retry-After."""

    metadata = _ERROR_TAXONOMY["engine_busy"]
    assert metadata["status_code"] == 429
    assert metadata["retryable"] is True

    error = _generation_admission_http_error(_rejection(), retry_after_seconds=3)

    assert error.status_code == 429
    assert error.code == "engine_busy"
    assert error.error_type == "rate_limit_error"
    assert error.headers["Retry-After"] == "3"
    assert error.extra["hipengine"]["routing"]["reason"] == "engine_busy"
    assert error.extra["hipengine"]["routing"]["overload_source"] == "kv_pool_capacity"
    assert error.extra["hipengine"]["routing"]["admission"] == {
        "resource": "kv_pool_pages",
        "requested_units": 64,
        "current_units": 4096,
        "capacity_units": 4096,
    }
    assert _error_extension(429, "engine_busy")["retryable"] is True


def test_unsupported_explicit_request_is_a_non_retryable_501() -> None:
    """Condition: an unsupported explicit request is 501 and not retryable."""

    metadata = _ERROR_TAXONOMY["unsupported_feature"]
    assert metadata["status_code"] == 501
    assert metadata["retryable"] is False

    extension = _error_extension(501, "unsupported_feature")
    assert extension == {
        "code": "unsupported_feature",
        "status_code": 501,
        "retryable": False,
    }


# --- Retry-After semantics ------------------------------------------------


@pytest.mark.parametrize(
    "requested,expected",
    [
        pytest.param(0, "1", id="zero-clamped-to-one"),
        pytest.param(-5, "1", id="negative-clamped-to-one"),
        pytest.param(1, "1", id="one"),
        pytest.param(30, "30", id="thirty"),
    ],
)
def test_retry_after_is_always_a_positive_second_count(
    requested: int, expected: str
) -> None:
    """A Retry-After of zero or a negative value would tell a client to retry now.

    The header is a delay, so the floor is one second; a client that retried
    immediately would be rejected again and could hot-loop.
    """

    error = _generation_admission_http_error(
        _rejection(), retry_after_seconds=requested
    )
    assert error.headers["Retry-After"] == expected
    assert int(error.headers["Retry-After"]) >= 1


def test_admission_error_preserves_caller_routing_metadata() -> None:
    """Existing routing evidence survives, and the busy fields are added to it."""

    error = _generation_admission_http_error(
        _rejection(),
        retry_after_seconds=2,
        error_extra={
            "hipengine": {
                "routing": {"requested_model": "qwen3.5-27b", "reason": "old"},
                "request_id": "abc",
            }
        },
    )

    routing = error.extra["hipengine"]["routing"]
    assert routing["requested_model"] == "qwen3.5-27b"
    assert routing["reason"] == "engine_busy"
    assert error.extra["hipengine"]["request_id"] == "abc"


def test_admission_error_builds_routing_metadata_when_absent() -> None:
    """A rejection with no caller extras still reports a complete busy payload."""

    error = _generation_admission_http_error(_rejection(), retry_after_seconds=1)

    assert set(error.extra["hipengine"]) == {"routing"}
    assert error.extra["hipengine"]["routing"]["overload_source"] == "kv_pool_capacity"


# --- Taxonomy integrity ---------------------------------------------------


def test_error_extension_resolves_aliases_and_keeps_the_legacy_code() -> None:
    """A client that sent a legacy code still learns the canonical one."""

    extension = _error_extension(400, "context_length_exceeded")

    assert extension["code"] == "context_overflow"
    assert extension["legacy_code"] == "context_length_exceeded"
    assert extension["status_code"] == 400
    assert extension["retryable"] is False


def test_error_extension_omits_the_legacy_code_when_already_canonical() -> None:
    extension = _error_extension(429, "engine_busy")

    assert extension["code"] == "engine_busy"
    assert "legacy_code" not in extension


@pytest.mark.parametrize("code", ["not_a_real_code", "engine_busy_v2", ""])
def test_error_extension_omits_retryability_for_an_unknown_code(code: str) -> None:
    """An unrecognised code is reported, but never given an invented retry verdict.

    The payload still names the code so a client can log it, and it carries no
    ``retryable`` field. That absence is meaningful: it distinguishes "known to
    be non-retryable" (``retryable: false``) from "this server does not know
    this code", which a client must not read as permission to retry.
    """

    extension = _error_extension(500, code)

    assert extension == {"code": code, "status_code": 500}
    assert "retryable" not in extension


def test_error_extension_is_none_only_when_no_code_is_given() -> None:
    """A missing code yields no extension at all, which is a different case."""

    assert _error_extension(500, None) is None


def test_every_alias_targets_a_code_in_the_taxonomy() -> None:
    """An alias pointing at a missing code would erase a client's retry decision."""

    missing = sorted(
        legacy
        for legacy, canonical in _ERROR_CODE_ALIASES.items()
        if canonical not in _ERROR_TAXONOMY
    )
    assert missing == []
    for legacy, canonical in _ERROR_CODE_ALIASES.items():
        assert _canonical_error_code(legacy) == canonical


def test_taxonomy_entries_are_complete_and_well_typed() -> None:
    """Every entry carries the fields the response builder reads."""

    for code, metadata in _ERROR_TAXONOMY.items():
        assert set(metadata) == {"status_code", "retryable", "emitted", "description"}, code
        assert 400 <= int(metadata["status_code"]) <= 599, code
        assert isinstance(metadata["retryable"], bool), code
        assert isinstance(metadata["emitted"], bool), code
        assert str(metadata["description"]).strip(), code


def test_taxonomy_manifest_round_trips_the_table_and_aliases() -> None:
    """The advertised manifest must describe the table the server actually uses."""

    manifest = _error_taxonomy_manifest()

    assert manifest["schema"] == "hipengine.error_taxonomy.v1"
    assert {entry["code"] for entry in manifest["codes"]} == set(_ERROR_TAXONOMY)
    for entry in manifest["codes"]:
        assert entry["status_code"] == _ERROR_TAXONOMY[entry["code"]]["status_code"]
        assert entry["retryable"] == _ERROR_TAXONOMY[entry["code"]]["retryable"]
    assert {
        (entry["legacy_code"], entry["code"]) for entry in manifest["aliases"]
    } == set(_ERROR_CODE_ALIASES.items())


def test_every_emitted_code_is_known_to_the_taxonomy() -> None:
    """Every code the server emits must resolve to a retry verdict.

    A code marked ``emitted`` but absent from the table would reach a client
    with no ``retryable`` field, which is the same shape as an unknown code.
    """

    for code, metadata in _ERROR_TAXONOMY.items():
        if metadata["emitted"]:
            extension = _error_extension(int(metadata["status_code"]), code)
            assert "retryable" in extension, code
            assert extension["retryable"] is metadata["retryable"], code


def test_only_declared_retryable_codes_are_retryable() -> None:
    """Pin the retryability split the acceptance depends on.

    A code that tells a client to retry when the condition is permanent (or the
    reverse) is a contract bug, so the split is asserted explicitly rather than
    left to a reader of the table.
    """

    retryable = {
        code for code, meta in _ERROR_TAXONOMY.items() if meta["retryable"]
    }
    not_retryable = set(_ERROR_TAXONOMY) - retryable

    # Transient: the same request can succeed later.
    assert {
        "engine_busy",
        "engine_unavailable",
        "deadline_exceeded",
        "cancelled",
        "execution_failed",
        "routing_failed",
    } <= retryable
    # Permanent for this request: retrying reproduces the same error.
    assert {
        "unsupported_parameter",
        "unsupported_feature",
        "schema_violation",
        "invalid_tool_call",
        "invalid_continuation",
        "continuation_expired",
        "context_overflow",
        "model_unavailable",
        "internal_error",
    } <= not_retryable


def test_unsupported_feature_is_not_retryable_while_engine_busy_is() -> None:
    """The acceptance's contrast, stated directly.

    Both are refusals, and a client must be able to tell them apart: an
    over-capacity request is worth retrying, an unsupported explicit request
    never becomes supported.
    """

    assert _ERROR_TAXONOMY["engine_busy"]["retryable"] is True
    assert _ERROR_TAXONOMY["unsupported_feature"]["retryable"] is False
    assert _ERROR_TAXONOMY["engine_busy"]["status_code"] == 429
    assert _ERROR_TAXONOMY["unsupported_feature"]["status_code"] == 501


def test_engine_unavailable_message_is_typed_and_actionable() -> None:
    """The 503's description must tell an operator what to do about it."""

    description = _ERROR_TAXONOMY["engine_unavailable"]["description"]
    assert "Restart the server" in description
