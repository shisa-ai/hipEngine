"""The Q5_1T16 down-prefill comparison arm after the env lever's removal (R0049).

Production runs the tile leaf (``_gemma4_project_experts_down_wmma_t16``); the
raw-block grouped WMMA owner stays as its fallback for weights that carry no
tile allocation and as the compensated contract's owner. The lever's only
remaining job was the in-process A/B its ledger entry describes, so it moved
into ``scripts/gemma4_p3_down_ab.py`` as ``--down-t16``. These tests pin that
arm, its restore on a raising run, the leaf's own selection and fall-through,
and the absence of an environment read on both sides of it. The harness's own
accounting is pinned too: a decline adds no time to the owner that fell
through, the event pair is destroyed on an exception, a stale arm exits
nonzero, and the session is closed on every exit.
"""

from __future__ import annotations

import inspect
import sys
from types import SimpleNamespace

import pytest

import hipengine.kernels.hip_gfx1100.gemma4.gemma4_experts as ge
import hipengine.kernels.hip_gfx1100.quant.qwen4_exp_q5_1 as q5_1
import scripts.gemma4_p3_down_ab as harness

TILE_LEAF = ge._gemma4_project_experts_down_wmma_t16
TILE_GUARD = ge._gemma4_down_t16_tiles_ready


class _FakeWeight:
    """Only the attributes the tile leaf reads, and nothing else."""

    def __init__(self, quant_key: str = "gguf_q5_1", tiles: bool = True) -> None:
        self.spec = SimpleNamespace(quant_key=quant_key)
        self.allocations = (
            {"t16_down": SimpleNamespace(buffer=SimpleNamespace(ptr=0x1000))}
            if tiles
            else {}
        )

    def allocation(self, name: str):
        return self.allocations[name]


@pytest.fixture(autouse=True)
def _restore_leaf():
    yield
    ge._gemma4_project_experts_down_wmma_t16 = TILE_LEAF
    ge._gemma4_down_t16_tiles_ready = TILE_GUARD


def _route_count(name: str) -> int:
    return ge.gemma4_moe_expert_route_counts().get(name, 0)


def _patch_kernel(monkeypatch, launches: list) -> None:
    monkeypatch.setattr(q5_1, "build_qwen4_exp_q5_1", lambda load=True: "library")
    monkeypatch.setattr(
        q5_1,
        "qwen4_exp_q5_1_t16_selected_grouped_prefill_bf16_bf16_out",
        lambda *a, **k: launches.append((a, k)),
    )


def _call_leaf(weight) -> bool:
    return ge._gemma4_project_experts_down_wmma_t16(
        weight, 1, 2, 3, 4, 5, 64, 8, 2816, 704, 128, stream=0,
    )


def test_the_leaf_reads_no_environment():
    # The retired lever's name is deliberately not spelled here:
    # scripts/check_envs_docs.py treats every `HIPENGINE_*` literal under tests/
    # as an environment read, so naming it would demand documenting it again.
    source = inspect.getsource(TILE_LEAF)
    assert "os.environ" not in source
    assert "getenv" not in source
    assert "import os" not in source


def test_the_arm_guard_is_the_guard_the_leaf_reads():
    """The arm moves the leaf's own guard, not a second selector."""

    assert "if not _gemma4_down_t16_tiles_ready(weight):" in inspect.getsource(TILE_LEAF)
    weight = _FakeWeight()
    assert ge._gemma4_down_t16_tiles_ready(weight) is True

    patches = harness._Patches()
    try:
        harness._pin_raw_block_owner(patches)
        assert ge._gemma4_down_t16_tiles_ready(weight) is False
    finally:
        patches.restore()
    assert ge._gemma4_down_t16_tiles_ready is TILE_GUARD


def test_a_raising_run_restores_the_guard():
    patches = harness._Patches()
    with pytest.raises(RuntimeError):
        try:
            harness._pin_raw_block_owner(patches)
            assert ge._gemma4_down_t16_tiles_ready is not TILE_GUARD
            raise RuntimeError("capture failed")
        finally:
            patches.restore()
    assert ge._gemma4_down_t16_tiles_ready is TILE_GUARD


def test_restore_is_idempotent_and_puts_the_owner_back():
    def sentinel(*a, **k):
        return True

    patches = harness._Patches()
    patches.set(ge, "_gemma4_project_experts_down_wmma_t16", sentinel)
    assert ge._gemma4_project_experts_down_wmma_t16 is sentinel
    patches.restore()
    patches.restore()  # main's finally may run after any exit
    assert ge._gemma4_project_experts_down_wmma_t16 is TILE_LEAF


def test_the_leaf_launches_when_the_tiles_are_present(monkeypatch):
    launches: list = []
    _patch_kernel(monkeypatch, launches)
    before = _route_count("down_t16")

    assert _call_leaf(_FakeWeight()) is True

    assert len(launches) == 1
    assert launches[0][1]["library"] == "library"
    assert _route_count("down_t16") == before + 1


def test_the_leaf_declines_without_the_tile_allocation(monkeypatch):
    launches: list = []
    _patch_kernel(monkeypatch, launches)
    before = _route_count("down_t16")

    assert _call_leaf(_FakeWeight(tiles=False)) is False

    assert launches == []
    assert _route_count("down_t16") == before


def test_the_leaf_declines_for_another_quant(monkeypatch):
    launches: list = []
    _patch_kernel(monkeypatch, launches)

    assert _call_leaf(_FakeWeight(quant_key="gguf_q8_0")) is False
    assert launches == []


def test_the_leaf_declines_outside_its_geometry(monkeypatch):
    launches: list = []
    _patch_kernel(monkeypatch, launches)

    assert ge._gemma4_project_experts_down_wmma_t16(
        _FakeWeight(), 1, 2, 3, 4, 5, 64, 8, 2817, 704, 128,
    ) is False
    assert launches == []


def test_the_script_parses_the_arm(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gemma4_p3_down_ab.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        harness.main()
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--down-t16" in out
    assert "{on,off}" in out


def test_the_script_rejects_an_unknown_arm(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["gemma4_p3_down_ab.py", "--model", "m", "--down-t16", "maybe"]
    )
    with pytest.raises(SystemExit) as exit_info:
        harness.main()
    assert exit_info.value.code == 2


def test_the_arm_check_requires_the_owner_to_have_launched():
    """A run whose owner never launched is a mismatch, not a zero-cost result."""

    assert harness._arm_check("on", 43, 5, {"down_t16": 43}) is True
    # on: the tile leaf launched nothing -> the arm did not take effect
    assert harness._arm_check("on", 0, 48, {}) is False
    # on: the engine's counter disagrees with the timed owner
    assert harness._arm_check("on", 43, 5, {"down_t16": 42}) is False
    # off: any tile launch means the pin did not take effect
    assert harness._arm_check("off", 0, 48, {}) is True
    assert harness._arm_check("off", 1, 48, {"down_t16": 1}) is False
    assert harness._arm_check("off", 0, 0, {}) is False


class _FakeRuntime:
    """The HIP event calls the harness makes, counted."""

    def __init__(self, elapsed_ms: float = 2.0) -> None:
        self.elapsed_ms = elapsed_ms
        self.created = 0
        self.destroyed = 0

    def event_create(self):
        self.created += 1
        return object()

    def event_record(self, event, stream):
        pass

    def event_synchronize(self, event):
        pass

    def event_elapsed_time_ms(self, start, end):
        return self.elapsed_ms

    def event_destroy(self, event):
        self.destroyed += 1


def _sink() -> dict:
    return {"n": 0, "ms": 0.0, "declined": 0}


def test_a_declined_call_adds_no_time_to_the_owner():
    """A fall-through is not this owner's duration, whatever the events read."""

    runtime = _FakeRuntime(elapsed_ms=2.5)
    sink = _sink()
    assert harness._timed(runtime, lambda *a, **k: False, sink)() is False
    assert sink == {"n": 1, "ms": 0.0, "declined": 1}
    assert runtime.created == runtime.destroyed == 2


def test_a_launched_call_adds_its_time_and_reports_it():
    runtime = _FakeRuntime(elapsed_ms=2.5)
    sink = _sink()
    seen: list = []
    wrapped = harness._timed(runtime, lambda *a, **k: True, sink, on_launch=seen.append)

    assert wrapped() is True

    assert sink == {"n": 1, "ms": 2.5, "declined": 0}
    assert seen == [2.5]
    assert runtime.created == runtime.destroyed == 2


def test_the_event_pair_is_destroyed_when_the_owner_raises():
    runtime = _FakeRuntime()
    sink = _sink()

    def boom(*a, **k):
        raise RuntimeError("kernel launch failed")

    with pytest.raises(RuntimeError):
        harness._timed(runtime, boom, sink)()
    assert runtime.created == runtime.destroyed == 2
    assert sink == {"n": 0, "ms": 0.0, "declined": 0}


class _FakeLLM:
    """A session that runs one forward through the real harness wrappers."""

    def __init__(self, model) -> None:
        self.model = model
        self.closed = False
        self.generated = 0

    def generate(self, prompt, params):
        self.generated += 1
        ge.gemma4_experts_forward_bf16(scratch=SimpleNamespace(tokens=512), rows=None)

    def close(self) -> None:
        self.closed = True


def _fake_run(monkeypatch, *, weight, arm: str = "on"):
    """Drive ``main`` end to end with a fake runtime, session, and owners."""

    runtime = _FakeRuntime()
    sessions: list = []
    monkeypatch.setattr(harness, "get_hip_runtime", lambda: runtime)
    monkeypatch.setattr(
        harness.hipengine, "LLM", lambda model: sessions.append(_FakeLLM(model)) or sessions[-1]
    )

    def raw_owner(*a, **k):
        return True

    monkeypatch.setattr(ge, "gemma4_project_experts_wmma", raw_owner)

    def forward(*a, **k):
        # The production down chain: the tile leaf first, the raw-block owner
        # on a decline.
        if not ge._gemma4_project_experts_down_wmma_t16(
            weight, 1, 2, 3, 4, 5, 64, 8, 2816, 704, 128
        ):
            ge.gemma4_project_experts_wmma(
                weight, 1, 2, 3, 4, 5, 64, 8, 2816, 704, 128
            )

    _patch_kernel(monkeypatch, [])
    monkeypatch.setattr(ge, "gemma4_experts_forward_bf16", forward)
    monkeypatch.setattr(
        sys, "argv", ["gemma4_p3_down_ab.py", "--model", "fake", "--down-t16", arm]
    )
    return runtime, sessions, raw_owner


def test_a_launched_arm_exits_zero_and_closes_the_session(monkeypatch, capsys):
    runtime, sessions, raw_owner = _fake_run(monkeypatch, weight=_FakeWeight())

    assert harness.main() == 0

    out = capsys.readouterr().out
    assert "arm check          : ok" in out
    assert "tile down          : 1 calls, 1 launched" in out
    assert sessions[-1].closed is True
    assert runtime.created == runtime.destroyed
    assert ge._gemma4_project_experts_down_wmma_t16 is TILE_LEAF
    assert ge.gemma4_project_experts_wmma is raw_owner


def test_a_stale_arm_exits_nonzero(monkeypatch, capsys):
    """`on` selected, but no weight carries tiles, so the arm ran nothing."""

    runtime, sessions, _ = _fake_run(monkeypatch, weight=_FakeWeight(tiles=False))

    assert harness.main() == 1

    out = capsys.readouterr().out
    assert "arm check          : MISMATCH" in out
    assert sessions[-1].closed is True
    assert runtime.created == runtime.destroyed


def test_the_off_arm_exits_zero_and_restores_the_guard(monkeypatch, capsys):
    runtime, _, _ = _fake_run(monkeypatch, weight=_FakeWeight(), arm="off")

    assert harness.main() == 0

    out = capsys.readouterr().out
    assert "tile down          : 1 calls, 0 launched" in out
    assert "raw-block down     : 1 calls, 1 launched" in out
    assert ge._gemma4_down_t16_tiles_ready is TILE_GUARD
    assert runtime.created == runtime.destroyed


def test_a_failing_generation_still_closes_the_session_and_restores(monkeypatch):
    runtime, sessions, raw_owner = _fake_run(monkeypatch, weight=_FakeWeight())

    def boom(self, prompt, params):
        raise RuntimeError("generation failed")

    monkeypatch.setattr(_FakeLLM, "generate", boom)
    with pytest.raises(RuntimeError):
        harness.main()

    assert sessions[-1].closed is True
    assert ge._gemma4_project_experts_down_wmma_t16 is TILE_LEAF
    assert ge.gemma4_project_experts_wmma is raw_owner
    assert runtime.created == runtime.destroyed


def test_a_failing_forward_destroys_its_events_and_closes_the_session(monkeypatch):
    runtime, sessions, _ = _fake_run(monkeypatch, weight=_FakeWeight())

    def boom(*a, **k):
        raise RuntimeError("forward failed")

    # _fake_run installed the working forward; main wraps whatever attribute it
    # finds, so replacing it here is what main's own wrapper observes.
    monkeypatch.setattr(ge, "gemma4_experts_forward_bf16", boom)
    with pytest.raises(RuntimeError):
        harness.main()

    assert sessions[-1].closed is True
    assert runtime.created == runtime.destroyed == 2
    assert ge.gemma4_experts_forward_bf16 is boom  # main's wrapper is gone
