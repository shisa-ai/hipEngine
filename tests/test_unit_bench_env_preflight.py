"""Unit tests for the environment preflight and the server's resolved-env report.

The regression these cover: a launcher that builds a clean environment
(``exec env -i ...``) silently strips a tuning variable, the server keeps its
default, and a benchmark reports a number for the wrong configuration without
anything failing.  Both halves are checked here - the server reporting what it
resolved, and the preflight refusing to pass when it cannot confirm a request.
"""

from __future__ import annotations

import pytest

from scripts.bench_env_preflight import (
    EnvPreflightError,
    assert_env_effective,
    env_preflight_failures,
    find_env_resolution_block,
    main,
    parse_assignments,
)

WINDOW_ENV = "HIPENGINE_MTP2_MAX_CONTEXT_TOKENS"


def _window_block(
    *,
    exported: str | None,
    resolved: object,
    error: str | None = None,
) -> dict[str, object]:
    return {
        "env": WINDOW_ENV,
        "exported": exported,
        "resolved": resolved,
        "qualified_default": 1023,
        "error": error,
    }


def _capabilities(
    effective_env: dict[str, str] | None,
    *,
    window: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"object": "hipengine.capabilities"}
    if effective_env is not None:
        payload["effective_env"] = effective_env
    if window is not None:
        payload["features"] = {"sampling": {"speculative_mtp": {"context_window": window}}}
    return payload


class TestServerResolvedEnvReport:
    """The server must report the values it resolved, credentials excepted."""

    def test_reports_only_hipengine_variables_sorted(self, monkeypatch) -> None:
        from hipengine.server.api import _effective_hipengine_env

        monkeypatch.setenv("HIPENGINE_ZZZ", "1")
        monkeypatch.setenv("HIPENGINE_AAA", "2")
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("OTHER_TOOL_FLAG", "3")

        reported = _effective_hipengine_env()

        assert list(reported) == sorted(reported)
        assert reported["HIPENGINE_AAA"] == "2"
        assert reported["HIPENGINE_ZZZ"] == "1"
        assert "PATH" not in reported
        assert "OTHER_TOOL_FLAG" not in reported

    def test_redacts_credentials_by_exact_name(self, monkeypatch) -> None:
        from hipengine.server.api import _effective_hipengine_env

        monkeypatch.setenv("HIPENGINE_API_KEY", "sk-live-secret")
        monkeypatch.setenv("HIPENGINE_KEY", "another-secret")

        reported = _effective_hipengine_env()

        assert reported["HIPENGINE_API_KEY"] == "<redacted>"
        assert reported["HIPENGINE_KEY"] == "<redacted>"
        assert "sk-live-secret" not in repr(reported)

    def test_keeps_tuning_knobs_whose_names_contain_key_or_token(
        self, monkeypatch
    ) -> None:
        """A substring redaction rule would hide the knobs this report exists for."""

        from hipengine.server.api import _effective_hipengine_env

        monkeypatch.setenv(WINDOW_ENV, "8192")
        monkeypatch.setenv("HIPENGINE_MAX_CONTEXT_TOKENS", "32768")
        monkeypatch.setenv("HIPENGINE_GGUF_INT8_KV_KEY_ONLY", "1")
        monkeypatch.setenv("HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED", "0")

        reported = _effective_hipengine_env()

        assert reported[WINDOW_ENV] == "8192"
        assert reported["HIPENGINE_MAX_CONTEXT_TOKENS"] == "32768"
        assert reported["HIPENGINE_GGUF_INT8_KV_KEY_ONLY"] == "1"
        assert reported["HIPENGINE_FULL_QKV_SPLIT_KEY_FUSED"] == "0"

    def test_window_policy_reports_target_capacity_with_no_fixed_cap(self) -> None:
        """The 1,023 window is gone, and the payload says so.

        Prompt context is not an admission axis (docs/EXECUTION-PROFILES.md
        section 2.9), so the adapter is bounded by the target's allocated
        capacity rather than by an evidence window. A harness asserting which
        policy is in effect must read this block rather than infer it from
        rates, and a lingering ``fixed_cap`` would mean a guard survived the
        removal.
        """

        from hipengine.server.api import _mtp2_context_window_resolution

        resolution = _mtp2_context_window_resolution()

        assert resolution["policy"] == "target_capacity"
        assert resolution["fixed_cap"] is None
        assert resolution["env"] is None

    def test_window_policy_ignores_a_retired_export(self, monkeypatch) -> None:
        """The removed variable must not be able to re-impose a cap."""

        from hipengine.server.api import _mtp2_context_window_resolution

        monkeypatch.setenv(WINDOW_ENV, "8192")

        resolution = _mtp2_context_window_resolution()

        assert resolution["policy"] == "target_capacity"
        assert resolution["fixed_cap"] is None
        assert resolution["resolved"] is None

    def test_log_fields_render_the_policy(self, monkeypatch) -> None:
        """The startup line names the policy instead of an env/resolved pair."""

        from hipengine.server.api import _mtp2_context_window_log_fields

        monkeypatch.setenv(WINDOW_ENV, "8192")

        assert _mtp2_context_window_log_fields() == ("none", "target_capacity")


class TestPreflightDetectsTheStrippedExport:
    """The preflight must fail when the requested value is not in effect."""

    def test_confirms_a_requested_knob_that_is_in_effect(self) -> None:
        capabilities = _capabilities(
            {WINDOW_ENV: "8192"},
            window=_window_block(exported="8192", resolved=8192),
        )

        assert env_preflight_failures(capabilities, {WINDOW_ENV: "8192"}) == []
        assert assert_env_effective(capabilities, {WINDOW_ENV: "8192"}) == {
            WINDOW_ENV: "8192"
        }

    def test_fails_when_a_clean_environment_stripped_the_export(self) -> None:
        """The recorded footgun: wrapper forwarded two variables, not this one."""

        capabilities = _capabilities(
            {"HIPENGINE_HIP_ARCH": "gfx1100"},
            window=_window_block(exported=None, resolved=1023),
        )

        failures = env_preflight_failures(capabilities, {WINDOW_ENV: "8192"})

        assert failures
        assert any("not present in the server environment" in item for item in failures)
        assert any("resolved to 1023, not 8192" in item for item in failures)

    def test_fails_when_the_server_reports_a_different_value(self) -> None:
        capabilities = _capabilities({WINDOW_ENV: "1023"})

        failures = env_preflight_failures(capabilities, {WINDOW_ENV: "8192"})

        assert len(failures) == 1
        assert "requested '8192'" in failures[0]
        assert "resolved '1023'" in failures[0]

    def test_fails_when_a_name_is_mistyped(self) -> None:
        """A typo is the same silent no-op as a stripped variable."""

        capabilities = _capabilities({WINDOW_ENV: "8192"})

        failures = env_preflight_failures(
            capabilities, {"HIPENGINE_MTP2_MAX_CONTEXT_TOKEN": "8192"}
        )

        assert len(failures) == 1
        assert "not present in the server environment" in failures[0]

    def test_fails_when_the_report_is_missing_instead_of_passing(self) -> None:
        """A check that cannot see its evidence must not report success."""

        failures = env_preflight_failures({"object": "x"}, {WINDOW_ENV: "8192"})

        assert len(failures) == 1
        assert "no 'effective_env' block" in failures[0]

    def test_fails_when_the_reported_value_is_redacted(self) -> None:
        capabilities = _capabilities({"HIPENGINE_API_KEY": "<redacted>"})

        failures = env_preflight_failures(
            capabilities, {"HIPENGINE_API_KEY": "sk-live-secret"}
        )

        assert len(failures) == 1
        assert "cannot be verified" in failures[0]

    def test_fails_when_a_knob_expected_to_be_unset_is_present(self) -> None:
        capabilities = _capabilities({"HIPENGINE_STALE_FLAG": "1"})

        failures = env_preflight_failures(
            capabilities, {}, unset=["HIPENGINE_STALE_FLAG"]
        )

        assert len(failures) == 1
        assert "expected to be unset" in failures[0]

    def test_confirms_a_knob_that_is_absent_when_absence_was_requested(self) -> None:
        capabilities = _capabilities({"HIPENGINE_HIP_ARCH": "gfx1100"})

        assert env_preflight_failures(
            capabilities, {}, unset=["HIPENGINE_STALE_FLAG"]
        ) == []

    def test_catches_an_export_that_did_not_change_the_resolved_value(self) -> None:
        """The export can be present while the resolved value stays the default."""

        capabilities = _capabilities(
            {WINDOW_ENV: "8192"},
            window=_window_block(exported="8192", resolved=1023),
        )

        failures = env_preflight_failures(capabilities, {WINDOW_ENV: "8192"})

        assert len(failures) == 1
        assert "resolved to 1023, not 8192" in failures[0]

    def test_error_message_names_the_source_server(self) -> None:
        capabilities = _capabilities({WINDOW_ENV: "1023"})

        with pytest.raises(EnvPreflightError) as excinfo:
            assert_env_effective(
                capabilities, {WINDOW_ENV: "8192"}, source="http://127.0.0.1:8097"
            )

        assert "http://127.0.0.1:8097" in str(excinfo.value)


class TestResolutionBlockDiscoveryIsPathAgnostic:
    """Hard-coded nesting would fail open the day the payload is reorganized."""

    def test_finds_the_block_at_a_different_nesting(self) -> None:
        payload = {
            "some": {
                "deeper": [
                    {"context_window": _window_block(exported=None, resolved=1023)}
                ]
            }
        }

        found = find_env_resolution_block(payload, WINDOW_ENV)

        assert found is not None
        assert found["resolved"] == 1023

    def test_returns_none_when_no_block_is_present(self) -> None:
        assert find_env_resolution_block({"effective_env": {}}, WINDOW_ENV) is None

    def test_does_not_mistake_another_env_block_for_the_one_named(self) -> None:
        payload = {"block": {"env": "HIPENGINE_SOMETHING_ELSE", "resolved": 7}}

        assert find_env_resolution_block(payload, WINDOW_ENV) is None

    def test_the_named_block_is_found(self) -> None:
        payload = {"block": {"env": WINDOW_ENV, "resolved": 7}}

        assert find_env_resolution_block(payload, WINDOW_ENV) == payload["block"]


class TestAssignmentParsing:
    def test_parses_name_value_pairs(self) -> None:
        assert parse_assignments(["A=1", "B=two"]) == {"A": "1", "B": "two"}

    def test_accepts_an_empty_value(self) -> None:
        assert parse_assignments(["A="]) == {"A": ""}

    def test_rejects_a_bare_name_and_points_at_unset(self) -> None:
        with pytest.raises(EnvPreflightError) as excinfo:
            parse_assignments(["A"])

        assert "--unset" in str(excinfo.value)

    def test_rejects_a_duplicate_name(self) -> None:
        with pytest.raises(EnvPreflightError):
            parse_assignments(["A=1", "A=2"])


class TestCliContract:
    """Exit status is the interface a launcher script depends on."""

    def test_exits_zero_and_prints_the_confirmation(self, monkeypatch, capsys) -> None:
        capabilities = _capabilities(
            {WINDOW_ENV: "8192"},
            window=_window_block(exported="8192", resolved=8192),
        )
        monkeypatch.setattr(
            "scripts.bench_env_preflight.fetch_capabilities",
            lambda *args, **kwargs: capabilities,
        )

        status = main(["--url", "http://127.0.0.1:8097", f"{WINDOW_ENV}=8192"])

        assert status == 0
        output = capsys.readouterr().out
        assert f"confirmed {WINDOW_ENV}=8192" in output
        assert f"{WINDOW_ENV}: exported='8192' resolved=8192" in output

    def test_exits_nonzero_and_explains_a_stripped_export(
        self, monkeypatch, capsys
    ) -> None:
        capabilities = _capabilities(
            {"HIPENGINE_HIP_ARCH": "gfx1100"},
            window=_window_block(exported=None, resolved=1023),
        )
        monkeypatch.setattr(
            "scripts.bench_env_preflight.fetch_capabilities",
            lambda *args, **kwargs: capabilities,
        )

        status = main(["--url", "http://127.0.0.1:8097", f"{WINDOW_ENV}=8192"])

        assert status == 1
        error = capsys.readouterr().err
        assert "ENV PREFLIGHT FAILED" in error
        assert "resolved to 1023, not 8192" in error

    def test_exits_nonzero_on_a_malformed_assignment(self, capsys) -> None:
        status = main(["--url", "http://127.0.0.1:8097", "NOPE"])

        assert status == 1
        assert "ENV PREFLIGHT FAILED" in capsys.readouterr().err

    def test_exits_nonzero_when_the_server_is_unreachable(self, capsys) -> None:
        status = main(
            ["--url", "http://127.0.0.1:1", f"{WINDOW_ENV}=8192", "--timeout", "0.5"]
        )

        assert status == 1
        assert "ENV PREFLIGHT FAILED" in capsys.readouterr().err
