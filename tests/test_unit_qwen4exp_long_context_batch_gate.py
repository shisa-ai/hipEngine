"""CPU contracts for the long-context gate's AR batch execution arm.

The long-context gate originally used the c1 candidate trajectory for every
entry in ``CANDIDATES``.  ``ar_batch`` was listed there before this adapter
existed, so the gate could silently measure c1 and report it as the batch
candidate.  These tests make that fallback a hard failure and pin the batch
arm's composition, teacher-row mapping, perturbation basis, and provenance.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import qwen4exp_long_context_gate as gate


def _trajectory(rows: int) -> tuple[dict, ...]:
    return tuple(
        {
            "token_id": index,
            "logits": np.zeros(8, dtype=np.float32),
        }
        for index in range(rows)
    )


def _state(prompt_id: str, position: int) -> dict:
    return {
        "prompt_id": prompt_id,
        "state_sha256": f"{prompt_id}:{position}",
        "layout_sha256": "layout",
        "buffer_bytes": {"kv": 1},
        "finite": True,
        "position": position,
        "attention_positions": [[position, position] for _ in range(2)],
        "index_counts": [[position, 0] for _ in range(2)],
        "ple": None,
    }


def _fake_rows() -> tuple[list[dict], list[dict]]:
    numeric = [
        {
            "id": "numeric_a",
            "category": "code",
            "messages": [{"role": "user", "content": "numeric a"}],
        },
        {
            "id": "numeric_b",
            "category": "general_en",
            "messages": [{"role": "user", "content": "numeric b"}],
        },
    ]
    retrieval = [
        {
            "id": "retrieval_a",
            "category": "general_en",
            "messages": [{"role": "user", "content": "retrieval a"}],
            "answers": [11, 22],
        },
        {
            "id": "retrieval_b",
            "category": "general_ja",
            "messages": [{"role": "user", "content": "retrieval b"}],
            "answers": [33, 44],
        },
    ]
    return numeric, retrieval


class _FakeTokenizer:
    eos_token_id = 7

    def encode(self, text):
        return [ord(char) % 5 + 1 for char in str(text)]

    def decode(self, ids, *, skip_special=False):
        del skip_special
        return ":".join(str(int(token)) for token in ids)


class _FakeRunner:
    max_sequence_length = 128

    def __init__(self):
        self.position = 0

    def reset(self):
        self.position = 0

    def prefill(self, ids, **kwargs):
        self.position = len(ids)
        return SimpleNamespace(
            token_id=len(ids),
            logits=np.zeros(8, dtype=np.float32),
        )

    def step(self, token, **kwargs):
        self.position += 1
        return SimpleNamespace(
            token_id=int(token),
            logits=np.zeros(8, dtype=np.float32),
        )


class _FakeGenerator:
    def __init__(self):
        self.runner = _FakeRunner()
        self.tokenizer = _FakeTokenizer()
        self.execution_profile_manifest = {"selections": []}

    def close(self):
        return None


class _FakeProfile:
    manifest_sha256 = "p" * 64
    manifest = {
        "backend": "hip_gfx1151",
        "model": "qwen4exp",
        "quant": "gguf_ud_q4_k_xl",
    }


def _patch_gate_for_dispatch(
    monkeypatch, *, c1_calls, batch_calls, forced_calls
):
    numeric, retrieval = _fake_rows()

    monkeypatch.setattr(
        gate, "_git_metadata",
        lambda path: {"tracked_clean": True, "head": "a" * 40},
    )
    monkeypatch.setattr(
        gate, "_load",
        lambda path: retrieval if "retrieval" in str(path) else numeric,
    )
    monkeypatch.setattr(gate, "sha256_path", lambda path: "b" * 64)
    monkeypatch.setattr(gate, "_host_metadata", lambda: {"hostname": "unit"})
    monkeypatch.setattr(
        gate, "_make_generator",
        lambda args, profile, quant=None: (_FakeGenerator(), _FakeProfile(), None),
    )
    monkeypatch.setattr(
        gate, "_strict_trajectory",
        lambda runner, ids, steps: _trajectory(int(steps) + 1),
    )

    def fake_forced(runner, ids, forced, overrides=None):
        forced_calls.append({
            "ids": tuple(int(token) for token in ids),
            "overrides": overrides,
        })
        return _trajectory(len(forced) + 1)

    monkeypatch.setattr(gate, "_forced_trajectory", fake_forced)
    monkeypatch.setattr(
        gate, "_state_summary",
        lambda runner: _state("strict", 3),
        raising=False,
    )
    monkeypatch.setattr(
        gate, "roundoff_override",
        lambda runner, ids, seed: ({0: np.zeros(8, dtype=np.float32)},
                                   {"position": 0, "element": 0, "ulp_step": 1}),
    )
    monkeypatch.setattr(
        gate, "score_retrieval",
        lambda trajectory, gold, spans: {
            "correct": [True] * len(spans),
            "correct_count": len(spans),
            "gold_nll_mean": 0.0,
            "answer_nll_mean": 0.0,
        },
    )
    monkeypatch.setattr(
        gate, "_summary",
        lambda strict, arm, categories, name: {
            "kl_mean": 0.001,
            "kl_p95": 0.002,
            "kl_max": 0.003,
            "top1_matches": 1,
        },
    )
    monkeypatch.setattr(
        gate, "decide",
        lambda *args, **kwargs: {
            "floor": {},
            "checks": {},
            "measurement_valid": True,
            "status": "passed",
        },
    )
    monkeypatch.setattr(
        gate, "_batch_manifest_binding",
        lambda manifest, **kwargs: {"valid": True, "bound": True},
        raising=False,
    )
    monkeypatch.setattr(
        gate, "_candidate_trajectory",
        lambda *args, **kwargs: c1_calls.append((args, kwargs)) or _trajectory(3),
    )

    def fake_batch(generator, *, groups, width, repeats, candidate_mode,
                   source_revision, owner_factory=None):
        batch_calls.append(
            {
                "generator": generator,
                "groups": groups,
                "width": width,
                "repeats": repeats,
                "candidate_mode": candidate_mode,
                "source_revision": source_revision,
                "owner_factory": owner_factory,
            }
        )
        prompt_ids = [
            prompt_id for group in groups for prompt_id in group.prompt_ids
        ]
        runs = {
            prompt_id: [_trajectory(3) for _ in range(int(repeats))]
            for prompt_id in prompt_ids
        }
        states = {
            prompt_id: [_state(prompt_id, 3) for _ in range(int(repeats))]
            for prompt_id in prompt_ids
        }
        return {
            "runs": runs,
            "states": states,
            "groups_by_repeat": [[{"repeat": 0}]],
            "evidence": {
                "groups": [],
                "step_widths": {str(width): 1},
                "variant_manifest": {"manifest_sha256": "m" * 64},
            },
            "manifest": {"manifest_sha256": "m" * 64},
        }

    monkeypatch.setattr(gate, "_run_batch_candidate", fake_batch, raising=False)

    import hipengine.generation.qwen4_exp_profiles as profiles
    import hipengine.kernels.hip_gfx1100.quant.gguf_q4_k_selected_prefill as prefill
    import hipengine.kernels.hip_gfx1151 as gfx1151
    import scripts.qwen4exp_ar_batch_candidate as candidate

    monkeypatch.setattr(
        profiles, "register_qwen4_exp_gfx1151_profiles", lambda **kwargs: None
    )
    monkeypatch.setattr(
        prefill, "register_gguf_q4_k_selected_prefill_kernels",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        gfx1151, "register_gfx1151_kernels", lambda **kwargs: None
    )
    def fake_state_gate(strict_states, candidate_states, *, groups):
        # ``_state_summary`` has no prompt id; the gate must inject it before
        # calling the adapter's gate, whose result keys off that field.
        assert all("prompt_id" in state for state in strict_states)
        assert len(strict_states) == len(candidate_states)
        return {
            "passed": True,
            "mismatches": [],
            "group_mismatches": [],
            "prompts": [],
        }

    monkeypatch.setattr(candidate, "batch_state_gate", fake_state_gate)


def _run_args(tmp_path: Path, *, width: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        model_root=tmp_path,
        candidate="ar_batch",
        batch_width=width,
        numeric_prompts=tmp_path / "numeric.jsonl",
        retrieval_prompts=tmp_path / "retrieval.jsonl",
        controls=2,
        control_seed=1,
        decode_steps=2,
        candidate_repeats=3,
        max_sequence_length=128,
        prefill_chunk_size=8,
        output=tmp_path / "gate.json",
    )


@pytest.mark.parametrize("width", [1, 2])
def test_ar_batch_long_gate_uses_batch_arm_and_never_calls_c1(
    monkeypatch, tmp_path, width
):
    """The pre-adapter bug evaluated c1 for every candidate; fail closed now."""

    c1_calls: list = []
    batch_calls: list = []
    forced_calls: list = []
    _patch_gate_for_dispatch(
        monkeypatch, c1_calls=c1_calls, batch_calls=batch_calls,
        forced_calls=forced_calls,
    )

    result = gate.run(_run_args(tmp_path, width=width), command=["test"])

    assert c1_calls == []
    # Strict retrieval is the only unperturbed forced call; a c1 candidate
    # repeat would add numeric/retrieval forced calls with overrides=None.
    unperturbed = [call for call in forced_calls if call["overrides"] is None]
    assert len(unperturbed) == 2  # strict retrieval only; no c1 candidate
    assert len(batch_calls) == 1
    call = batch_calls[0]
    assert call["width"] == width
    assert call["repeats"] == 3
    assert {
        prompt_id
        for group in call["groups"]
        for prompt_id in group.prompt_ids
    } == {"numeric_a", "numeric_b", "retrieval_a", "retrieval_b"}
    assert result["candidate"]["arm_kind"] == "batch"
    assert result["candidate"]["batch_width"] == width


def test_ar_batch_rejects_missing_width_before_candidate_setup(monkeypatch, tmp_path):
    args = _run_args(tmp_path)
    args.batch_width = None
    monkeypatch.setattr(
        gate, "_git_metadata",
        lambda path: {"tracked_clean": True, "head": "a" * 40},
    )
    with pytest.raises(gate.GateError, match="requires --batch-width"):
        gate.run(args, command=["test"])


def test_ar_batch_rejects_width4_instead_of_lowering_context(monkeypatch, tmp_path):
    args = _run_args(tmp_path)
    args.batch_width = 4
    monkeypatch.setattr(
        gate, "_git_metadata",
        lambda path: {"tracked_clean": True, "head": "a" * 40},
    )
    with pytest.raises(gate.GateError, match="2048"):
        gate.run(args, command=["test"])


class _BatchRunner:
    def __init__(self):
        self.positions: dict[int, int] = {}

    def position(self, slot: int) -> int:
        return int(self.positions[int(slot)])


class _BatchOwner:
    """Records the actual row count of every owner.step call."""

    def __init__(self, width: int):
        self.width = int(width)
        self.runner = _BatchRunner()
        self.slots: dict[int, int] = {}
        self.owners: dict[int, int] = {}
        self.free = list(range(self.width))
        self.next_owner = 100
        self.step_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.released: list[int] = []

    def ensure(self):
        return None

    def close(self):
        return None

    def reserve(self, request_id: int, *, positions: int) -> int:
        slot = self.free.pop(0)
        self.slots[int(request_id)] = slot
        self.owners[int(request_id)] = self.next_owner
        self.next_owner += 1
        self.runner.positions[slot] = 0
        return slot

    def import_prefilled(self, request_id: int, *, source) -> int:
        slot = self.slots[int(request_id)]
        position = int(source.position)
        self.runner.positions[slot] = position
        return position

    def step(self, rows, *, capture_logits: bool = True):
        request_ids = tuple(int(request_id) for request_id, _token in rows)
        tokens = tuple(int(token) for _request_id, token in rows)
        self.step_calls.append((request_ids, tokens))
        results = {}
        for request_id, token in zip(request_ids, tokens):
            slot = self.slots[request_id]
            self.runner.positions[slot] += 1
            results[request_id] = SimpleNamespace(
                token_id=token + 100 * request_id,
                logits=np.full(8, float(request_id), dtype=np.float32),
            )
        return results

    def lease(self, request_id: int):
        request_id = int(request_id)
        if request_id not in self.slots:
            return None
        return self.slots[request_id], self.owners[request_id]

    def release(self, request_id: int) -> bool:
        request_id = int(request_id)
        if request_id not in self.slots:
            return False
        slot = self.slots.pop(request_id)
        self.owners.pop(request_id, None)
        self.free.append(slot)
        self.released.append(request_id)
        return True


def _slot_state(runner, slot, *, prompt_id):
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


def test_long_batch_groups_map_numeric_and_retrieval_rows_to_n_row_calls(
    monkeypatch,
):
    from scripts.qwen4exp_ar_batch_candidate import BatchCandidateRequest
    import scripts.qwen4exp_ar_batch_candidate as candidate

    monkeypatch.setattr(candidate, "batch_slot_state", _slot_state)
    manifest_calls = []

    def fake_manifest(**kwargs):
        manifest_calls.append(kwargs)
        return {
            "manifest_sha256": "m" * 64,
            "source_revision": kwargs["source_revision"],
            "entries": [],
        }

    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_batch_manifest."
        "build_qwen4_exp_batch_variant_manifest",
        fake_manifest,
    )
    generator = _FakeGenerator()
    generator.execution_profile_manifest = {"selections": []}

    numeric = [
        BatchCandidateRequest("numeric_a", 1, (11, 12, 13), (21, 22)),
        BatchCandidateRequest("numeric_b", 2, (14, 15, 16), (23, 24)),
    ]
    retrieval = [
        BatchCandidateRequest("retrieval_a", 3, (31, 32), (41, 42, 43)),
        BatchCandidateRequest("retrieval_b", 4, (33, 34), (44, 45, 46)),
        BatchCandidateRequest("retrieval_c", 5, (35, 36), (47,)),
    ]
    groups, schedule = gate.plan_long_batch_groups(numeric, retrieval, 2)
    assert [group.width for group in groups] == [2, 2, 1]
    assert [group.prompt_ids for group in groups] == [
        ("numeric_a", "numeric_b"),
        ("retrieval_a", "retrieval_b"),
        ("retrieval_c",),
    ]
    assert schedule["numeric_a"] == {"schedule": "numeric", "forced_length": 2}
    assert schedule["retrieval_a"] == {"schedule": "retrieval", "forced_length": 3}
    assert schedule["retrieval_c"] == {"schedule": "retrieval", "forced_length": 1}

    width1_groups, _ = gate.plan_long_batch_groups(numeric, retrieval, 1)
    assert [group.width for group in width1_groups] == [1] * 5

    owner = _BatchOwner(2)
    result = gate._run_batch_candidate(
        generator,
        groups=groups,
        width=2,
        repeats=3,
        candidate_mode="qwen4_exp_ar_batch_w2",
        source_revision="s" * 40,
        owner_factory=lambda width: owner,
    )

    assert result["manifest"]["source_revision"] == "s" * 40
    assert manifest_calls[0]["source_revision"] == "s" * 40
    assert manifest_calls[0]["profile_manifest"] == generator.execution_profile_manifest
    assert len(result["groups_by_repeat"]) == 3
    assert len(result["evidence"]["groups"]) == 9
    assert result["evidence"]["step_widths"] == {"1": 3, "2": 15}
    assert result["evidence"]["imports"] == 15
    first_group = result["evidence"]["groups"][0]
    assert first_group["width"] == 2
    assert first_group["prompt_ids"] == ["numeric_a", "numeric_b"]
    assert len({row["slot"] for row in first_group["rows"]}) == 2
    assert len({row["owner"] for row in first_group["rows"]}) == 2
    assert first_group["ownership_sha256"]
    assert len(owner.step_calls) == 18
    assert [len(request_ids) for request_ids, _ in owner.step_calls] == [
        2, 2, 2, 2, 2, 1
    ] * 3

    # The forced token and result stay with their own request; neighbours cannot
    # be swapped into a row's trajectory.
    assert [row["token_id"] for row in result["runs"]["numeric_a"][0]] == [
        3, 121, 122
    ]
    assert [row["token_id"] for row in result["runs"]["numeric_b"][0]] == [
        3, 223, 224
    ]
    assert [row["token_id"] for row in result["runs"]["retrieval_a"][0]] == [
        2, 341, 342, 343
    ]
    assert len(result["states"]["numeric_a"]) == 3
    assert result["states"]["numeric_a"][0]["position"] == 5
    assert result["states"]["retrieval_c"][0]["position"] == 3


def test_summary_rejects_trajectory_length_mismatch():
    with pytest.raises(gate.GateError, match="length mismatch"):
        gate._summary(
            {"prompt": _trajectory(3)},
            {"prompt": _trajectory(2)},
            {"prompt": "code"},
            "candidate",
        )


def test_batch_manifest_binding_pins_source_and_profile_identity(monkeypatch):
    calls = []

    def fake_validate(manifest, **kwargs):
        calls.append((manifest, kwargs))
        return {"valid": True, "bound": True}

    monkeypatch.setattr(
        "hipengine.runtime.qwen4_exp_batch_manifest."
        "validate_batch_variant_manifest",
        fake_validate,
    )
    profile = SimpleNamespace(
        manifest_sha256="p" * 64,
        manifest={
            "backend": "hip_gfx1151",
            "model": "qwen4exp",
            "quant": "gguf_ud_q4_k_xl",
        },
    )
    result = gate._batch_manifest_binding(
        {"manifest_sha256": "m" * 64},
        candidate_profile=profile,
        source_revision="s" * 40,
    )
    assert result == {"valid": True, "bound": True}
    _manifest, kwargs = calls[0]
    assert kwargs == {
        "expected_profile_manifest_sha256": "p" * 64,
        "expected_source_revision": "s" * 40,
        "expected_backend": "hip_gfx1151",
        "expected_model": "qwen4exp",
        "expected_quant": "gguf_ud_q4_k_xl",
    }


def test_decide_makes_batch_state_and_manifest_binding_binding():
    candidate = {"kl_mean": 1e-3, "kl_p95": 2e-3, "kl_max": 3e-3,
                 "top1_matches": 10}
    controls = [
        {"kl_mean": 2e-3, "kl_p95": 4e-3, "kl_max": 5e-3, "top1_matches": 9},
        {"kl_mean": 3e-3, "kl_p95": 5e-3, "kl_max": 6e-3, "top1_matches": 10},
    ]
    passed = gate.decide(
        candidate, controls, 2, 2, deterministic=True,
        batch_state_passed=True, batch_manifest_bound=True,
    )
    assert passed["status"] == "passed"
    state_failed = gate.decide(
        candidate, controls, 2, 2, deterministic=True,
        batch_state_passed=False, batch_manifest_bound=True,
    )
    assert state_failed["status"] == "failed"
    assert state_failed["checks"]["batch_state_gate"] is False
    manifest_invalid = gate.decide(
        candidate, controls, 2, 2, deterministic=True,
        batch_state_passed=True, batch_manifest_bound=False,
    )
    assert manifest_invalid["status"] == "invalid"
    assert manifest_invalid["measurement_valid"] is False


def test_control_is_one_embedding_bf16_ulp_and_uses_embedding_overrides(monkeypatch):
    original = np.zeros(8, dtype=np.uint16)
    original[3] = 0x3F80  # BF16 1.0
    monkeypatch.setattr(
        gate, "_embedding_bits",
        lambda runner, token_id: original.copy(),
    )
    overrides, meta = gate.roundoff_override(object(), [10, 11, 12], seed=7)
    assert meta["position"] in (0, 1)
    element = int(meta["element"])
    assert 0 <= element < original.size
    assert set(overrides) == {meta["position"]}
    values = overrides[meta["position"]]
    assert values.dtype == np.float32
    original_values = (original.astype(np.uint32) << 16).view(np.float32)
    changed = np.flatnonzero(values.view(np.uint32) != original_values.view(np.uint32))
    assert changed.tolist() == [element]
    assert int(values[element].view(np.uint32)) == (
        int(original[element]) + int(meta["ulp_step"])
    ) << 16

    class _RecordingRunner:
        def __init__(self):
            self.prefill_kwargs = None
            self.position = 0

        def reset(self):
            self.position = 0

        def prefill(self, ids, **kwargs):
            self.prefill_kwargs = kwargs
            self.position = len(ids)
            return SimpleNamespace(token_id=1, logits=np.zeros(8, dtype=np.float32))

        def step(self, token):
            self.position += 1
            return SimpleNamespace(token_id=int(token), logits=np.zeros(8, dtype=np.float32))

    runner = _RecordingRunner()
    gate._forced_trajectory(runner, [10, 11, 12], [13], overrides)
    assert runner.prefill_kwargs == {"embedding_overrides": overrides}
    assert gate.CONTROL_BASIS == "prompt_embedding_row_one_bf16_ulp"
    assert gate.CONTROL_APPLICATION == "runner.prefill(embedding_overrides=...)"


def test_batch_width_is_rejected_for_a_non_batch_candidate(monkeypatch, tmp_path):
    args = _run_args(tmp_path, width=1)
    args.candidate = "production_baseline"
    monkeypatch.setattr(
        gate, "_git_metadata",
        lambda path: {"tracked_clean": True, "head": "a" * 40},
    )
    with pytest.raises(gate.GateError, match="only valid"):
        gate.run(args, command=["test"])
