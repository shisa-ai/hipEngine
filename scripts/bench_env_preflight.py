#!/usr/bin/env python3
"""Verify that a running hipEngine server actually resolved the requested knobs.

Why this exists
---------------
Several benchmark launchers build a clean environment for the server, for
example with ``exec env -i ...``.  That is fine until someone exports a new
tuning variable: the launcher strips it, the server silently keeps its
default, and the benchmark measures the default while reporting the raised
value.  Nothing fails, and the numbers look plausible.

A real instance: a sweep exported ``HIPENGINE_MTP2_MAX_CONTEXT_TOKENS=8192``
through a wrapper that forwarded only two hard-coded variables.  The server kept
its default, and the "forced window" arm reproduced it to the digit (11.40 tok/s
both arms).  The near-miss was recording "raising the window changes nothing" as
a finding about the model.  That particular knob has since been removed as an
unjustified guard, but the hazard is general: any variable a launcher forwards
by hand can be lost the same way.

The server reports what it resolved in two places:

* the startup log line ``EFFECTIVE_MTP: ... mtp2_context_window=env:... resolved:...``
* ``/v1/hipengine/capabilities`` -> ``effective_env`` (every ``HIPENGINE_*``
  variable) and the MTP block's ``context_window`` (exported vs resolved)

This script turns those reports into an assertion.  Run it after the server is
ready and before the measurement, and a stripped or mistyped knob fails loudly
instead of quietly producing a number for the wrong configuration.

Usage
-----
    python3 scripts/bench_env_preflight.py --url http://127.0.0.1:8097 \
        HIPENGINE_GGUF_STAGED_LINEAR_ROWS_LONG=1

    # assert a knob is NOT set, catching a stale export
    python3 scripts/bench_env_preflight.py --url ... --unset HIPENGINE_FOO

Exit status is 0 only when every requested value is confirmed in effect.  A
missing or unreadable ``effective_env`` block is a failure, never a pass: a
check that cannot see its evidence must not report success.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

__all__ = [
    "EnvPreflightError",
    "assert_env_effective",
    "env_preflight_failures",
    "find_env_resolution_block",
    "parse_assignments",
]

CAPABILITIES_PATH = "/v1/hipengine/capabilities"
REDACTED_VALUE = "<redacted>"


class EnvPreflightError(RuntimeError):
    """A requested environment knob is not the value the server resolved."""


def parse_assignments(assignments: Iterable[str]) -> dict[str, str]:
    """Parse ``NAME=VALUE`` pairs, rejecting malformed or duplicate names."""

    parsed: dict[str, str] = {}
    for item in assignments:
        name, separator, value = str(item).partition("=")
        name = name.strip()
        if not separator or not name:
            raise EnvPreflightError(
                f"expected NAME=VALUE, got {item!r} "
                "(use --unset NAME to assert a variable is not set)"
            )
        if name in parsed:
            raise EnvPreflightError(f"{name} was requested more than once")
        parsed[name] = value
    return parsed


def find_env_resolution_block(payload: Any, env_name: str) -> Mapping[str, Any] | None:
    """Find a payload block that reports how ``env_name`` resolved.

    Some knobs resolve into a value the server reports separately from the raw
    export, so the export alone does not prove the setting took effect. This
    looks for such a block by the variable it names.

    The search is path-agnostic on purpose.  Hard-coding the nesting would make
    this script fail open the day the payload is reorganized, which is the
    same silent-success failure it exists to prevent.
    """

    if isinstance(payload, Mapping):
        if payload.get("env") == env_name and "resolved" in payload:
            return payload
        for value in payload.values():
            found = find_env_resolution_block(value, env_name)
            if found is not None:
                return found
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes)):
        for value in payload:
            found = find_env_resolution_block(value, env_name)
            if found is not None:
                return found
    return None


def env_preflight_failures(
    capabilities: Any,
    requested: Mapping[str, str],
    *,
    unset: Iterable[str] = (),
) -> list[str]:
    """Return one message per requested knob the server did not confirm."""

    failures: list[str] = []

    if not isinstance(capabilities, Mapping):
        return ["capabilities payload is not a JSON object"]

    reported = capabilities.get("effective_env")
    if not isinstance(reported, Mapping):
        return [
            "capabilities payload has no 'effective_env' block, so no requested "
            "knob can be confirmed (the server may predate the report)"
        ]

    for name, expected in sorted(requested.items()):
        if name not in reported:
            failures.append(
                f"{name}: not present in the server environment "
                f"(requested {expected!r}, server resolved its default or nothing)"
            )
            continue
        actual = reported[name]
        if actual == REDACTED_VALUE:
            failures.append(
                f"{name}: reported as {REDACTED_VALUE}, so the value in effect "
                "cannot be verified from the capabilities payload"
            )
            continue
        if str(actual) != str(expected):
            failures.append(
                f"{name}: requested {expected!r} but the server resolved {actual!r}"
            )

    for name in sorted(set(unset)):
        if name in reported:
            failures.append(
                f"{name}: expected to be unset but the server resolved "
                f"{reported[name]!r}"
            )

    for name, expected in sorted(requested.items()):
        block = find_env_resolution_block(capabilities, name)
        if block is None:
            continue
        resolved = block.get("resolved")
        if resolved is None:
            continue
        try:
            expected_int: int | None = int(str(expected).strip())
        except ValueError:
            expected_int = None
        if expected_int is not None and int(resolved) != expected_int:
            failures.append(
                f"{name}: the export is in the server environment but resolved "
                f"to {resolved!r}, not {expected_int} (exported "
                f"{block.get('exported')!r}, error {block.get('error')!r})"
            )

    return failures


def assert_env_effective(
    capabilities: Any,
    requested: Mapping[str, str],
    *,
    unset: Iterable[str] = (),
    source: str | None = None,
) -> dict[str, str]:
    """Raise :class:`EnvPreflightError` unless every requested knob is in effect."""

    failures = env_preflight_failures(capabilities, requested, unset=unset)
    if failures:
        where = "" if source is None else f" on {source}"
        raise EnvPreflightError(
            "environment preflight failed"
            + where
            + ":\n  - "
            + "\n  - ".join(failures)
        )
    return {name: str(capabilities["effective_env"][name]) for name in requested}


def fetch_capabilities(
    url: str,
    *,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> Any:
    """Fetch the capabilities payload from a running server."""

    request = urllib.request.Request(
        url.rstrip("/") + CAPABILITIES_PATH,
        headers={"Accept": "application/json"},
    )
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise EnvPreflightError(
            f"{url}{CAPABILITIES_PATH} returned HTTP {exc.code}: {exc.reason}"
        ) from None
    except urllib.error.URLError as exc:
        raise EnvPreflightError(f"{url}{CAPABILITIES_PATH} is unreachable: {exc.reason}") from None
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise EnvPreflightError(
            f"{url}{CAPABILITIES_PATH} did not return JSON: {exc}"
        ) from None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assert that a running hipEngine server resolved the requested "
            "HIPENGINE_* environment knobs, so a launcher that strips them "
            "fails loudly instead of measuring the default."
        )
    )
    parser.add_argument("--url", required=True, help="Server base URL, e.g. http://127.0.0.1:8097")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("HIPENGINE_API_KEY"),
        help="Bearer token (defaults to $HIPENGINE_API_KEY)",
    )
    parser.add_argument(
        "--unset",
        action="append",
        default=[],
        metavar="NAME",
        help="Assert NAME is not set in the server environment (repeatable)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30)",
    )
    parser.add_argument(
        "assignments",
        nargs="*",
        metavar="NAME=VALUE",
        help="Knobs that must be in effect on the server",
    )
    args = parser.parse_args(argv)

    try:
        requested = parse_assignments(args.assignments)
        capabilities = fetch_capabilities(
            args.url, api_key=args.api_key, timeout=args.timeout
        )
        confirmed = assert_env_effective(
            capabilities, requested, unset=args.unset, source=args.url
        )
    except EnvPreflightError as exc:
        print(f"ENV PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1

    for name, value in sorted(confirmed.items()):
        print(f"confirmed {name}={value}")
    for name in sorted(set(args.unset)):
        print(f"confirmed {name} is unset")
    for name in sorted(requested):
        block = find_env_resolution_block(capabilities, name)
        if block is not None and block.get("resolved") is not None:
            print(
                f"{name}: exported={block.get('exported')!r} "
                f"resolved={block.get('resolved')!r}"
            )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
