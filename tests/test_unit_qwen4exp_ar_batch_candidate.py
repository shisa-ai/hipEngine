"""CPU tests for the Qwen4Exp AR batch numerical candidate adapter."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen4exp_ar_batch_candidate as candidate
from scripts.execution_profile_gdn_calibration import (
    PromptCalibrationCapture,
    build_candidate_quality,
)


# -- fixtures ---------------------------------------------------------------


class FakeSourceRunner:
    """c1 request runner: one request at a time, prefill then reset."""

    max_sequence_length = 64

    def __init__(self) -> None:
        self.position = 0

    def reset(self) -> None:
        self.position = 0

    def prefill(self, ids):
        self.position = len(ids)
        return SimpleNamespace(
            token_id=(int(ids[-1]) + 1) % 97,
            logits=np.full(8, 0.5, dtype=np.float32),
        )


class FakeBatchRunner:
    def __init__(self) -> None:
        self.positions: dict[int, int] = {}
        self.runtime = None
        self.attention_states: tuple = ()
        self.index_states: tuple = ()
        self.gdn_states: tuple = ()
        self.ple_state = None
        self.ple_hash_states: dict = {}

    def position(self, slot: int) -> int:
        return self.positions[int(slot)]


class FakeOwner:
    """Minimal AR batch owner: distinct slots/owners, per-row results."""

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.batch_runner = FakeBatchRunner()
        self.slots: dict[int, int] = {}
        self.owners: dict[int, int] = {}
        self.free = list(range(self.width))
        self.next_owner = 100
        self.released: list[int] = []
        self.fail = False
        self.import_offset = 0

    def ensure(self) -> None:
        return None

    def close(self) -> None:
        return None

    def reserve(self, request_id: int, *, positions: int) -> int:
        slot = self.free.pop(0)
        self.slots[int(request_id)] = slot
        self.batch_runner.positions[slot] = 0
        self.owners[int(request_id)] = self.next_owner
        self.next_owner += 1
        return slot

    def import_prefilled(self, request_id: int, *, source) -> int:
        position = int(source.position) + self.import_offset
        self.batch_runner.positions[self.slots[int(request_id)]] = position
        return position

    def step(self, rows, *, capture_logits: bool = True):
        if self.fail:
            raise RuntimeError("poisoned batch step")
        results = {}
        for request_id, token in rows:
            slot = self.slots[int(request_id)]
            self.batch_runner.positions[slot] += 1
            results[int(request_id)] = SimpleNamespace(
                token_id=(int(token) + 1) % 97,
                logits=np.full(8, float(token), dtype=np.float32),
            )
        return results

    def lease(self, request_id: int):
        request_id = int(request_id)
        if request_id not in self.slots:
            return None
        return (self.slots[request_id], self.owners[request_id])

    def release(self, request_id: int) -> bool:
        request_id = int(request_id)
        if request_id not in self.slots:
            return False
        slot = self.slots.pop(request_id)
        self.owners.pop(request_id, None)
        self.free.append(slot)
        self.released.append(request_id)
        return True

    @property
    def runner(self) -> FakeBatchRunner:
        return self.batch_runner


class FakeTokenizer:
    def decode(self, ids, *, skip_special: bool = False) -> str:
        return "|".join(str(int(token)) for token in ids)


class FakeGenerator:
    def __init__(self) -> None:
        self.runner = FakeSourceRunner()
        self.tokenizer = FakeTokenizer()


def _request(prompt_id: str, request_id: int, prompt, forced) -> candidate.BatchCandidateRequest:
    return candidate.BatchCandidateRequest(
        prompt_id=prompt_id,
        request_id=request_id,
        prompt_ids=tuple(prompt),
        forced_ids=tuple(forced),
    )


def _fake_slot_state(runner, slot, *, prompt_id):
    position = int(runner.position(slot))
    return {
        "prompt_id": prompt_id,
        "state_sha256": f"{prompt_id}:{slot}:{position}",
        "layout_sha256": "layout",
        "buffer_bytes": {"kv": 1},
        "finite": True,
        "position": position,
        "attention_positions": [[position, position] for _ in range(2)],
        "index_counts": [[position, 0] for _ in range(2)],
        "ple": None,
    }


@pytest.fixture
def fake_adapter(monkeypatch):
    monkeypatch.setattr(candidate, "batch_slot_state", _fake_slot_state)
    owner = FakeOwner(2)

    def build(width=2, owner_factory=None):
        generator = FakeGenerator()
        adapter = candidate.Qwen4ExpARBatchCandidate(
            generator,
            width=width,
            owner_factory=owner_factory or (lambda _width: owner),
        )
        return adapter, owner, generator

    return build


# -- composition ------------------------------------------------------------


def test_plan_batch_composition_keeps_natural_final_width():
    requests = [_request(f"p{index}", index + 1, [1], [2]) for index in range(5)]
    groups = candidate.plan_batch_composition(requests, 4)
    assert [group.width for group in groups] == [4, 1]
    assert [group.index for group in groups] == [0, 1]
    assert groups[0].prompt_ids == ("p0", "p1", "p2", "p3")
    assert groups[1].prompt_ids == ("p4",)


def test_plan_batch_composition_rejects_unsupported_width():
    with pytest.raises(ValueError, match="width"):
        candidate.plan_batch_composition([_request("p", 1, [1], [2])], 3)


def test_plan_batch_composition_rejects_duplicate_ids():
    requests = [_request("p", 1, [1], [2]), _request("p", 2, [3], [4])]
    with pytest.raises(ValueError, match="duplicate prompt"):
        candidate.plan_batch_composition(requests, 2)


def test_plan_batch_composition_rejects_unequal_forced_lengths():
    requests = [_request("p0", 1, [1], [2]), _request("p1", 2, [3], [4, 5])]
    with pytest.raises(ValueError, match="teacher-forced length"):
        candidate.plan_batch_composition(requests, 2)


def test_plan_batch_composition_rejects_empty():
    with pytest.raises(ValueError, match="at least one"):
        candidate.plan_batch_composition([], 2)


# -- adapter ----------------------------------------------------------------


def test_run_group_prefills_on_c1_then_steps_width_rows(fake_adapter):
    adapter, owner, generator = fake_adapter()
    requests = [
        _request("p0", 1, [1, 2, 3], [10, 11]),
        _request("p1", 2, [4, 5], [20, 21]),
    ]
    group = candidate.plan_batch_composition(requests, 2)[0]

    result = adapter.run_group(group)

    assert [step["token_id"] for step in result.trajectories["p0"]] == [4, 11, 12]
    assert [step["token_id"] for step in result.trajectories["p1"]] == [6, 21, 22]
    assert [step["logits"].shape for step in result.trajectories["p0"]] == [(8,)] * 3
    assert result.states["p0"]["position"] == 5
    assert result.states["p1"]["position"] == 4
    evidence = result.evidence
    assert evidence["width"] == 2
    assert evidence["decode_steps"] == 2
    assert evidence["state_distinct"] is True
    assert evidence["distinct_state_sha256"] == 2
    assert {row["slot"] for row in evidence["rows"]} == {0, 1}
    assert {row["owner"] for row in evidence["rows"]} == {100, 101}
    assert adapter.step_widths == {2: 2}
    assert adapter.evidence["imports"] == 2
    assert sorted(owner.released) == [1, 2]
    assert sorted(owner.free) == [0, 1]


def test_run_group_rejects_import_position_mismatch(fake_adapter):
    adapter, owner, _ = fake_adapter()
    owner.import_offset = 1
    group = candidate.plan_batch_composition(
        [_request("p0", 1, [1, 2], [10]), _request("p1", 2, [3], [11])], 2
    )[0]
    with pytest.raises(RuntimeError, match="imported at position"):
        adapter.run_group(group)
    assert sorted(owner.free) == [0, 1]


def test_run_free_group_feeds_each_row_its_own_token(fake_adapter):
    adapter, owner, _ = fake_adapter()
    group = candidate.plan_batch_composition(
        [_request("p0", 1, [1, 2], [10]), _request("p1", 2, [3], [11])], 2
    )[0]

    free = adapter.run_free_group(group, steps=3)

    assert free["p0"]["ids"] == [3, 4, 5]
    assert free["p1"]["ids"] == [4, 5, 6]
    assert free["p0"]["text"] == "3|4|5"
    assert free["p0"]["state"]["position"] == 4
    assert adapter.step_widths == {2: 2}
    assert sorted(owner.free) == [0, 1]


def test_failed_step_poisons_adapter_and_releases_slots(fake_adapter):
    adapter, owner, _ = fake_adapter()
    owner.fail = True
    group = candidate.plan_batch_composition(
        [_request("p0", 1, [1, 2], [10]), _request("p1", 2, [3], [11])], 2
    )[0]

    with pytest.raises(RuntimeError, match="poisoned"):
        adapter.run_group(group)

    assert "poisoned" in adapter.evidence["failure"]
    assert sorted(owner.free) == [0, 1]
    with pytest.raises(RuntimeError, match="disabled"):
        adapter.run_group(group)


def test_adapter_refuses_unsupported_width():
    with pytest.raises(ValueError, match="width"):
        candidate.Qwen4ExpARBatchCandidate(FakeGenerator(), width=3)


def test_build_variant_manifest_uses_live_runner_and_dispatch_records(
    fake_adapter, monkeypatch
):
    adapter, owner, generator = fake_adapter()
    generator.execution_profile_manifest = {"kind": "profile"}
    adapter.open()
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return {"manifest_sha256": "a" * 64, "entries": []}

    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_batch_manifest."
        "build_qwen4_exp_batch_variant_manifest",
        fake_build,
    )

    manifest = adapter.build_variant_manifest(source_revision="b" * 40)

    assert manifest["manifest_sha256"] == "a" * 64
    assert calls[0]["source_revision"] == "b" * 40
    assert calls[0]["runner"] is owner.runner
    assert calls[0]["profile_manifest"] == generator.execution_profile_manifest
    assert calls[0]["dispatch_records"] == []
    assert calls[0]["step_widths"] == {}
    assert adapter.evidence["variant_manifest"]["manifest_sha256"] == "a" * 64


def test_build_variant_manifest_requires_profile_manifest(fake_adapter):
    adapter, _owner, _generator = fake_adapter()
    adapter.open()
    with pytest.raises(RuntimeError, match="execution profile manifest"):
        adapter.build_variant_manifest(source_revision="b" * 40)


# -- state gate -------------------------------------------------------------


def _strict_state(prompt_id: str, position: int, *, finite: bool = True):
    return {
        "prompt_id": prompt_id,
        "position": position,
        "finite": finite,
        "attention_positions": [[position, position] for _ in range(2)],
        "index_counts": [[position, 0] for _ in range(2)],
    }


def _candidate_state(prompt_id: str, position: int, *, salt: str = ""):
    return {
        "prompt_id": prompt_id,
        "state_sha256": f"{prompt_id}:{position}:{salt}",
        "layout_sha256": "layout",
        "buffer_bytes": {"kv": 1},
        "finite": True,
        "position": position,
        "attention_positions": [[position, position] for _ in range(2)],
        "index_counts": [[position, 0] for _ in range(2)],
        "ple": None,
    }


def _distinct_groups() -> list[dict]:
    return [{"width": 2, "state_distinct": True}]


def test_batch_state_gate_passes_deterministic_distinct_state():
    strict = [_strict_state("p0", 5), _strict_state("p1", 4)]
    runs = [
        [_candidate_state("p0", 5) for _ in range(3)],
        [_candidate_state("p1", 4) for _ in range(3)],
    ]
    gate = candidate.batch_state_gate(strict, runs, groups=_distinct_groups())
    assert gate["passed"] is True
    assert gate["layout_equivalence"].startswith("not_applicable")
    assert all(prompt["strict_candidate_state_exact"] is False for prompt in gate["prompts"])


def test_batch_state_gate_detects_repeat_mismatch():
    strict = [_strict_state("p0", 5)]
    runs = [[_candidate_state("p0", 5) for _ in range(3)]]
    runs[0][1] = _candidate_state("p0", 5, salt="drift")
    gate = candidate.batch_state_gate(strict, runs, groups=_distinct_groups())
    assert gate["passed"] is False
    assert gate["mismatches"][0]["prompt_id"] == "p0"


def test_batch_state_gate_detects_position_mismatch():
    strict = [_strict_state("p0", 5)]
    runs = [[_candidate_state("p0", 6) for _ in range(3)]]
    gate = candidate.batch_state_gate(strict, runs, groups=_distinct_groups())
    assert gate["passed"] is False
    assert gate["mismatches"][0]["position_exact"] is False


def test_batch_state_gate_detects_non_finite_state():
    strict = [_strict_state("p0", 5, finite=False)]
    runs = [[_candidate_state("p0", 5) for _ in range(3)]]
    gate = candidate.batch_state_gate(strict, runs, groups=_distinct_groups())
    assert gate["passed"] is False


def test_batch_state_gate_detects_shared_group_state():
    strict = [_strict_state("p0", 5)]
    runs = [[_candidate_state("p0", 5) for _ in range(3)]]
    groups = [{"width": 2, "state_distinct": False}]
    gate = candidate.batch_state_gate(strict, runs, groups=groups)
    assert gate["passed"] is False
    assert gate["group_mismatches"] == groups


def test_batch_state_gate_requires_three_repeats():
    strict = [_strict_state("p0", 5)]
    runs = [[_candidate_state("p0", 5) for _ in range(2)]]
    with pytest.raises(ValueError, match="three"):
        candidate.batch_state_gate(strict, runs, groups=_distinct_groups())


# -- row scopes -------------------------------------------------------------


def test_build_candidate_quality_accepts_batch_row_scopes():
    logits = np.arange(4, dtype=np.float32)
    trajectory = tuple(
        {"token_id": index, "logits": logits.copy()} for index in range(2)
    )
    captures = (
        PromptCalibrationCapture(
            prompt_id="p0",
            category="code",
            strict=trajectory,
            candidate_runs={"qwen4_exp_ar_batch_w2": (trajectory, trajectory, trajectory)},
        ),
        PromptCalibrationCapture(
            prompt_id="p1",
            category="general_en",
            strict=trajectory,
            candidate_runs={"qwen4_exp_ar_batch_w2": (trajectory, trajectory, trajectory)},
        ),
    )
    quality = build_candidate_quality(
        captures,
        candidate_mode="qwen4_exp_ar_batch_w2",
        scenario_id="qwen4exp-ar-batch",
        shape_for_step=lambda step, prompt_id: (
            "prefill_last"
            if step == 0
            else "batch_w2"
            if prompt_id == "p0"
            else "batch_w1"
        ),
        transition_for_step=lambda step, prompt_id: (
            "prefill_to_c1" if step == 0 else "prefill_to_batch"
        ),
    )
    by_scope = quality["quality"]["by_scope"]
    assert set(by_scope["shape"]) == {"prefill_last", "batch_w2", "batch_w1"}
    assert "prefill_to_batch" in by_scope["transition"]
    assert quality["repeat_determinism"]["passed"] is True


def test_build_candidate_quality_default_labels_unchanged():
    logits = np.arange(4, dtype=np.float32)
    trajectory = tuple(
        {"token_id": index, "logits": logits.copy()} for index in range(2)
    )
    capture = PromptCalibrationCapture(
        prompt_id="p0",
        category="code",
        strict=trajectory,
        candidate_runs={"c1": (trajectory, trajectory, trajectory)},
    )
    quality = build_candidate_quality(
        (capture,), candidate_mode="c1", scenario_id="scenario"
    )
    by_scope = quality["quality"]["by_scope"]
    assert set(by_scope["shape"]) == {"prefill_last", "c1"}
    assert "prefill_to_c1" in by_scope["transition"]
