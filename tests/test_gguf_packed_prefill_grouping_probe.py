from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from scripts import gguf_packed_prefill_grouping_probe as probe


def test_probe_is_runner_level_and_disallows_serving_claims() -> None:
    assert probe._PROBE_SCOPE == {
        "level": "runner",
        "serving_path_claim_eligible": False,
        "serving_attribution_source": (
            "scripts/gguf_mtp_c1c8_server_bench.py --capture-prefill-attribution"
        ),
    }


def test_peer_session_kwargs_preserve_owner_runtime_and_runner_identity() -> None:
    runtime = object()
    runner = object()

    kwargs = probe._shared_peer_session_kwargs(
        SimpleNamespace(runtime=runtime, runner=runner)
    )

    assert kwargs["runtime"] is runtime
    assert kwargs["shared_runner"] is runner


@pytest.mark.parametrize(
    "owner",
    [SimpleNamespace(runtime=None, runner=object()), SimpleNamespace(runtime=object(), runner=None)],
)
def test_peer_session_kwargs_reject_incomplete_owner(owner: object) -> None:
    with pytest.raises(RuntimeError, match="runtime and shared runner"):
        probe._shared_peer_session_kwargs(owner)


def test_explicit_argv_provenance_includes_interpreter_and_script() -> None:
    command = probe._invocation(("--model", "/model.gguf", "--output", "/tmp/out.json"))

    assert command[0] == sys.executable
    assert command[1].endswith("/scripts/gguf_packed_prefill_grouping_probe.py")
    assert command[2:] == ["--model", "/model.gguf", "--output", "/tmp/out.json"]


def _route(wall_ms: float, token_ids: list[int]) -> dict[str, object]:
    return {
        "wall_ms_median": wall_ms,
        "token_ids_identical_across_reps": True,
        "token_ids": token_ids,
    }


def _order(*, equal_packed_tokens: list[int] | None = None) -> dict[str, dict[str, object]]:
    return {
        "mixed_serial": _route(20.0, [1, 2]),
        "mixed_packed": _route(10.0, [1, 2]),
        "equal_serial": _route(18.0, [3, 4]),
        "equal_packed": _route(12.0, equal_packed_tokens or [3, 4]),
    }


def test_verdict_compares_only_identical_prompt_pairs_in_both_orders() -> None:
    verdict = probe._build_verdict({"forward": _order(), "reverse": _order()})

    assert verdict == {
        "mixed_packed_vs_serial_forward_order": 2.0,
        "mixed_packed_vs_serial_reversed_order": 2.0,
        "equal_packed_vs_serial_forward_order": 1.5,
        "equal_packed_vs_serial_reversed_order": 1.5,
        "mixed_prompt_pair_exact": True,
        "equal_prompt_pair_exact": True,
        "passed": True,
    }


def test_verdict_fails_an_equal_prompt_pair_mismatch() -> None:
    verdict = probe._build_verdict(
        {"forward": _order(), "reverse": _order(equal_packed_tokens=[3, 9])}
    )

    assert verdict["mixed_prompt_pair_exact"] is True
    assert verdict["equal_prompt_pair_exact"] is False
    assert verdict["passed"] is False
