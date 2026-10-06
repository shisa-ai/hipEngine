"""The finish-rule gate must be able to fail, not just to pass.

The gate's whole job is to catch a commit that publishes past a stop, so each
check is driven against a fake server that violates exactly one thing: a stop
that overshoots its chain, arms that disagree about the terminal prefix, a
finish reason that is not the stop, an EOS floor that suppresses the token
instead of the rule, and a concurrent neighbour whose cursor moves.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The fake trajectory: consecutive ids whose text is ``" t<id>"``, so the gate's
# detokenize-then-stop round trip maps a stop string back to one token index.
BASE_TOKEN = 100
DEFAULT_LENGTH = 20


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "mtp_finish_rule_gate", ROOT / "scripts" / "mtp_finish_rule_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = _load_module()


def token_text(token: int) -> str:
    return f" t{token}"


class _Response:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code != 200:
            raise RuntimeError(f"status {self.status_code}")

    def json(self) -> dict:
        return self._payload


class _FakeServer:
    """A server that runs the finish rule, with one violation injected at a time."""

    def __init__(
        self,
        *,
        storage: str = "int8_per_token_head",
        trajectory_length: int = DEFAULT_LENGTH,
        used: bool = True,
        stop_overshoot: int = 0,
        stop_publishes_whole_chain: bool = False,
        finish_reason_override: str | None = None,
        eos_floor_stops: bool = False,
        eos_rule_reports_stop: bool = False,
        eos_floor_emits_token: bool = False,
        route_parity: bool = True,
        publish_unknown_token: bool = False,
        neighbour_shift: bool = False,
        diagnostic_override: bool = False,
    ) -> None:
        self.storage = storage
        self.trajectory = list(range(BASE_TOKEN, BASE_TOKEN + trajectory_length))
        self.used = used
        self.stop_overshoot = stop_overshoot
        self.stop_publishes_whole_chain = stop_publishes_whole_chain
        self.finish_reason_override = finish_reason_override
        self.eos_floor_stops = eos_floor_stops
        self.eos_rule_reports_stop = eos_rule_reports_stop
        self.eos_floor_emits_token = eos_floor_emits_token
        self.route_parity = route_parity
        self.publish_unknown_token = publish_unknown_token
        self.neighbour_shift = neighbour_shift
        self.diagnostic_override = diagnostic_override
        self.requests: list[dict] = []
        self.seen_nonstop_seeds: dict[int, int] = {}

    def ready(self) -> dict:
        return {
            "status": "ready",
            "model": {
                "id": "fake-model",
                "kv_capability": {
                    "effective_kv_storage": self.storage,
                    "capability_id": "fake",
                    "status": "advertised",
                    "runtime_action": "run",
                    "diagnostic_override": self.diagnostic_override,
                    "persistent_bf16_mirror": False,
                },
            },
        }

    def detokenize(self, token_ids: list[int]) -> dict:
        return {"text": "".join(token_text(int(token)) for token in token_ids)}

    def _stop_index(self, stop_text: str) -> int | None:
        """Map a stop string back to the trajectory index that produced it."""

        for index, token in enumerate(self.trajectory):
            if token_text(token) == stop_text:
                return index
        for index in range(len(self.trajectory) - 1):
            pair = token_text(self.trajectory[index]) + token_text(self.trajectory[index + 1])
            if pair == stop_text:
                return index
        return None

    @staticmethod
    def detail_reason(finish: str, stop_text: str | None) -> str:
        """The rule that fired, which is what separates a stop from an EOS."""

        if finish == "stop" and stop_text is None:
            return "eos"
        return finish

    def completion(self, payload: dict) -> dict:
        self.requests.append(payload)
        speculative = bool(payload.get("speculative_mtp"))
        seed = int(payload.get("seed", 0))
        stop = payload.get("stop")
        if isinstance(stop, list):
            stop_text = stop[0] if stop else None
        else:
            stop_text = stop
        eos_token_id = payload.get("eos_token_id")
        min_tokens = int(payload.get("min_tokens", 0) or 0)

        ids = list(self.trajectory)
        finish = "length"
        if stop_text is not None:
            index = self._stop_index(str(stop_text))
            assert index is not None, f"fake server cannot place stop {stop_text!r}"
            if self.stop_publishes_whole_chain:
                published = len(self.trajectory)
            else:
                published = index + 1 + self.stop_overshoot
            ids = list(self.trajectory[:published])
            finish = "stop"
        elif eos_token_id is not None:
            index = self.trajectory.index(int(eos_token_id))
            if min_tokens <= index + 1 or self.eos_floor_stops:
                # At or above the floor the EOS rule fires; ``eos_floor_stops``
                # is the violation that fires it below the floor anyway.
                ids = list(self.trajectory[: index + 1])
                finish = "eos"
            elif self.eos_floor_emits_token:
                # The withheld token is emitted anyway, below the floor.
                ids = list(self.trajectory[: index + 1 + 3])
                finish = "length"
            else:
                # Below the floor the route withholds the EOS token and emits a
                # different tokenization of the same text, so the trajectory
                # diverges and generation continues past the EOS position.
                ids = [
                    *self.trajectory[:index],
                    700,
                    701,
                    *self.trajectory[index + 1 : index + 4],
                ]
                finish = "length"
        else:
            self.seen_nonstop_seeds[seed] = self.seen_nonstop_seeds.get(seed, 0) + 1
            if self.neighbour_shift and self.seen_nonstop_seeds[seed] > 1:
                ids = [*ids[:-1], ids[-1] + 1000]

        if self.finish_reason_override is not None:
            finish = self.finish_reason_override
        if self.publish_unknown_token and finish == "stop":
            # Both arms stay in parity, but the last token is not on the trajectory.
            ids = [*ids, 999999]
        elif not speculative and not self.route_parity and finish == "stop":
            # A shorter prefix: still a prefix, so only the parity check can see it.
            ids = ids[:-1]

        # The engine excludes the terminal token from the visible text, so the
        # fake does too: ``retokenized_visible_tokens`` trails the published ids
        # by one on a stop run.
        text_tokens = ids[:-1] if finish in {"stop", "eos"} else ids
        cycles = 5 if speculative and self.used else 0
        # A route that did not speculate published nothing from a cycle, so its
        # whole output is autoregressive tail -- which is what the gate's
        # per-row speculation exemption is keyed on.
        from_cycles = len(ids) if speculative and self.used else 0
        return {
            "choices": [
                {
                    "finish_reason": finish,
                    "text": "".join(token_text(token) for token in text_tokens),
                    "hipengine": {
                        "generated_token_ids": ids,
                        "finish_details": {
                            "reason": (
                                "stop"
                                if self.eos_rule_reports_stop and stop_text is None
                                else self.detail_reason(finish, stop_text)
                            ),
                            "cache_action": "append_none",
                        },
                        "decode_state": {"step_index": len(ids), "generated_tokens": len(ids)},
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": len(ids),
                "total_tokens": 5 + len(ids),
            },
            "hipengine": {
                "speculative_mtp": {
                    "used": self.used if speculative else False,
                    "effective_route": "speculative_mtp" if speculative else "default",
                    "draft_cycles": cycles,
                    "draft_tokens": 4 * cycles,
                    "accepted_draft_tokens": 3 * cycles,
                    "mtp_output_tokens": from_cycles,
                    "ar_output_tokens": len(ids) - from_cycles,
                }
            },
        }


class _FakeClient:
    def __init__(self, server: _FakeServer) -> None:
        self._server = server

    def __enter__(self) -> "_FakeClient":
        return self

    def __exit__(self, *exc) -> None:
        return None

    def get(self, path: str) -> _Response:
        assert path == "/ready"
        return _Response(self._server.ready())

    def post(self, path: str, json: dict) -> _Response:
        if path == "/v1/hipengine/detokenize":
            return _Response(self._server.detokenize(json["token_ids"]))
        assert path == "/v1/completions"
        return _Response(self._server.completion(json))


def _run_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    server: _FakeServer,
    *extra: str,
) -> tuple[int, dict]:
    monkeypatch.setattr(gate.httpx, "Client", lambda **kwargs: _FakeClient(server))
    output = tmp_path / "finish.json"
    status = gate.main([
        "--model", "fake-model", "--json", str(output),
        "--sweep-limit", "4", "--allow-kv-diagnostic-override", *extra,
    ])
    return status, json.loads(output.read_text())


def test_the_gate_passes_a_server_that_applies_the_finish_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer())

    assert status == 0
    assert report["passed"] is True
    assert report["summary"]["storage"] == "int8_per_token_head"
    assert report["summary"]["sweep_positions"] == 4
    assert report["summary"]["stop_sequence_rows"] >= 1
    assert report["summary"]["eos_floor_rows"] == 2
    assert report["summary"]["neighbour_isolated"] is True


def test_the_gate_refuses_a_cell_it_was_not_pointed_at(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(storage="bf16"), "--expect-storage", "int8_per_token_head"
    )

    assert status == 2
    assert "int8_per_token_head" in report["error"]
    assert "bf16" in report["error"]


def test_the_gate_refuses_the_diagnostic_override_by_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    server = _FakeServer(diagnostic_override=True)
    monkeypatch.setattr(gate.httpx, "Client", lambda **kwargs: _FakeClient(server))
    output = tmp_path / "finish.json"

    status = gate.main(["--model", "fake-model", "--json", str(output)])

    assert status == 2
    assert "diagnostic override" in json.loads(output.read_text())["error"]


def test_the_gate_records_the_override_when_it_is_allowed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(diagnostic_override=True)
    )

    assert status == 0
    assert report["server"]["diagnostic_override"] is True


def test_the_gate_fails_when_a_stop_publishes_past_the_terminal_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(stop_overshoot=1)
    )

    assert status == 1
    assert "token_published_after_the_stop" in report["error"]


def test_the_gate_fails_when_a_stop_publishes_the_whole_chain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(stop_publishes_whole_chain=True)
    )

    assert status == 1
    assert "token_published_after_the_stop" in report["error"]


def test_the_gate_fails_when_the_arms_disagree_about_the_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer(route_parity=False))

    assert status == 1
    assert "route_parity_mismatch" in report["error"]


def test_the_gate_fails_when_a_published_token_is_not_on_the_trajectory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(publish_unknown_token=True)
    )

    assert status == 1
    assert "published_token_outside_the_trajectory" in report["error"]


def test_the_gate_fails_when_the_finish_reason_is_not_the_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(finish_reason_override="length")
    )

    assert status == 1
    assert "finish_reason_not_stop" in report["error"]


def test_the_gate_fails_when_the_speculative_arm_did_not_speculate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer(used=False))

    assert status == 1
    assert "speculative_arm_did_not_speculate" in report["error"]


def test_a_row_too_short_to_cycle_is_exempt_and_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer())

    assert status == 0
    assert report["summary"]["sweep_rows_that_speculated"] == len(report["stop_sweep"])
    assert all(row["speculation_exempt"] is None for row in report["stop_sweep"])


def test_the_gate_fails_when_the_eos_floor_stops_below_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(eos_floor_stops=True)
    )

    assert status == 1
    assert "eos_floor_did_not_suppress_the_rule" in report["error"]


def test_the_gate_fails_when_the_eos_token_is_emitted_below_the_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(eos_floor_emits_token=True)
    )

    assert status == 1
    assert "eos_token_emitted_below_the_floor" in report["error"]


def test_the_gate_fails_when_eos_fires_above_the_floor_for_another_rule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(eos_rule_reports_stop=True)
    )

    assert status == 1
    assert "eos_did_not_fire_the_eos_rule_above_the_floor" in report["error"]


def test_the_gate_fails_when_a_neighbour_cursor_moves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(neighbour_shift=True)
    )

    assert status == 1
    assert "neighbour_cursor_moved_by_a_stopping_request" in report["error"]


def test_the_gate_fails_on_a_trajectory_too_short_to_sweep(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(
        monkeypatch, tmp_path, _FakeServer(trajectory_length=5)
    )

    assert status == 1
    assert "free_trajectory_too_short" in report["error"]


def test_the_gate_records_the_terminal_token_and_the_stop_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    status, report = _run_gate(monkeypatch, tmp_path, _FakeServer())

    assert status == 0
    for row in report["stop_sweep"]:
        assert row["stop_matched_at_build_position"] is True
        assert row["stop_text_in_text"] is False
        assert row["cache_action"] == "append_none"


def test_distinct_stop_positions_drops_a_stop_already_taken() -> None:
    """Repetitive text repeats tokens, so the later position must not be re-run."""

    # The token 101 recurs at positions 1 and 2; only the first is worth a request.
    trajectory = [100, 101, 101, 102, 103]

    def detokenize_one(ids: list[int]) -> str:
        return "".join(token_text(token) for token in ids)

    positions = gate.distinct_stop_positions(trajectory, detokenize_one, 1, 4)

    # sweep_positions stops before the final token, so position 4 is not offered.
    assert [position for position, _ in positions] == [1, 3]
    assert [text for _, text in positions] == [" t101", " t102"]


def test_sweep_positions_never_starts_at_the_first_token() -> None:
    positions = gate.sweep_positions(list(range(100, 130)), stride=1, limit=50)

    assert positions[0] == 1
    assert positions[-1] <= 28
    assert gate.sweep_positions(list(range(100, 130)), stride=3, limit=50) == list(range(1, 29, 3))
