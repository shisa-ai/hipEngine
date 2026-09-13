"""Unit tests for the poison-probe helper itself.

``tests/_poison_probe.py`` is the instrument every runtime hygiene test trusts,
so its own behaviour needs pinning: what it walks, what it refuses to walk, how
it buckets buffers for localization, and what it counts as a difference.  All of
it runs without a GPU -- ``DeviceBuffer`` is a ``(ptr, nbytes)`` dataclass and
the only runtime capability the helper uses is ``memset`` plus
``device_synchronize``, which a fake records.

The point of these tests is that a probe which silently collects nothing, or
silently buckets everything into one group, reports "clean" for the same reason
a broken smoke detector reports "no fire".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer

from _poison_probe import (
    POISON_BYTE,
    assert_poison_invariant,
    collect_device_buffers,
    group_by_prefix,
    iter_buffers,
    poison,
)


class FakeRuntime:
    """Records the memset/synchronize calls the helper makes."""

    def __init__(self) -> None:
        self.memsets: list[tuple[int, int, int]] = []
        self.syncs = 0

    def memset(self, ptr: int, byte: int, nbytes: int) -> None:
        self.memsets.append((ptr, byte, nbytes))

    def device_synchronize(self) -> None:
        self.syncs += 1


@dataclass
class Scratch:
    hidden: DeviceBuffer
    partials: list[DeviceBuffer] = field(default_factory=list)
    table: dict = field(default_factory=dict)


def test_collect_walks_dataclass_fields_sequences_and_mappings() -> None:
    root = Scratch(
        hidden=DeviceBuffer(0x1000, 64),
        partials=[DeviceBuffer(0x2000, 32), DeviceBuffer(0x3000, 32)],
        table={"a": DeviceBuffer(0x4000, 16)},
    )
    found = dict(collect_device_buffers(root))
    assert set(found) == {
        "Scratch.hidden",
        "Scratch.partials[0]",
        "Scratch.partials[1]",
        "Scratch.table['a']",
    }
    assert found["Scratch.partials[1]"] == DeviceBuffer(0x3000, 32)


def test_collect_skips_weights_runtime_and_other_non_state() -> None:
    """Weights and once-uploaded metadata must never be poisoned."""

    @dataclass
    class Holder:
        runtime: object
        weights: dict
        tokenizer: object
        scratch: DeviceBuffer

    holder = Holder(
        runtime=FakeRuntime(),
        weights={"w": DeviceBuffer(0x9000, 128)},
        tokenizer=DeviceBuffer(0x9100, 8),
        scratch=DeviceBuffer(0x9200, 16),
    )
    found = collect_device_buffers(holder)
    assert [path for path, _ in found] == ["Holder.scratch"], found


def test_collect_reports_an_arena_through_its_owner() -> None:
    """A poison write has to cover the owning allocation, not one view of it."""

    class FakeArena:
        def __init__(self, owner: DeviceBuffer) -> None:
            self.owner = owner

    @dataclass
    class Holder:
        arena: object

    owner = DeviceBuffer(0x5000, 4096)
    found = dict(collect_device_buffers(Holder(arena=FakeArena(owner))))
    assert found == {"Holder.arena.owner": owner}, "the owning allocation, not the view"


def test_collect_respects_the_depth_cap() -> None:
    @dataclass
    class Node:
        child: object

    leaf = DeviceBuffer(0x6000, 8)
    deep = leaf
    for _ in range(8):
        deep = Node(child=deep)
    assert collect_device_buffers(deep, max_depth=2) == []
    deep_paths = dict(collect_device_buffers(deep, max_depth=10))
    assert list(deep_paths) == ["Node." + "child." * 7 + "child"], deep_paths
    assert list(deep_paths.values()) == [leaf]


def test_collect_terminates_on_a_cycle() -> None:
    @dataclass
    class Node:
        child: object = None

    first, second = Node(), Node()
    first.child, second.child = second, first
    assert collect_device_buffers(first) == []


def test_iter_buffers_matches_collect() -> None:
    root = Scratch(hidden=DeviceBuffer(0x1000, 64))
    assert list(iter_buffers(root)) == collect_device_buffers(root)


def test_poison_writes_every_non_empty_buffer_and_synchronizes() -> None:
    runtime = FakeRuntime()
    buffers = [DeviceBuffer(0x1000, 64), DeviceBuffer(0x2000, 0), DeviceBuffer(0x3000, 16)]
    written = poison(runtime, buffers)
    assert written == 80, "an empty buffer contributes no bytes"
    assert runtime.memsets == [(0x1000, POISON_BYTE, 64), (0x3000, POISON_BYTE, 16)]
    assert runtime.syncs == 1
    assert poison(runtime, buffers, synchronize=False) == 80
    assert runtime.syncs == 1


def test_group_by_prefix_strips_indices_and_buckets_by_trailing_components() -> None:
    buffers = [
        ("_buffers[(2, 16, 20)].caches_v[7]", DeviceBuffer(0x1000, 4)),
        ("_buffers[(2, 16, 20)].caches_v[19]", DeviceBuffer(0x2000, 4)),
        ("_buffers[(2, 16, 20)].caches_k[7]", DeviceBuffer(0x3000, 4)),
    ]
    groups = group_by_prefix(buffers)
    assert set(groups) == {"_buffers.caches_v", "_buffers.caches_k"}
    assert len(groups["_buffers.caches_v"]) == 2, "same family, both indices"
    assert len(groups["_buffers.caches_k"]) == 1


def test_assert_poison_invariant_reports_a_clean_run() -> None:
    runtime = FakeRuntime()
    buffers = [DeviceBuffer(0x1000, 64)]

    def run() -> np.ndarray:
        return np.zeros(4, dtype=np.float32)

    report = assert_poison_invariant(
        runtime, lambda: {"scratch": buffers}, run, label="clean"
    )
    assert report["verdict"] == "ok"
    assert report["buffers"] == 1
    assert report["bytes"] == 64
    assert runtime.syncs == 1, "one poison pass"


def test_assert_poison_invariant_names_the_offender_group() -> None:
    """Localization is the diagnostic that matters; a whole-run failure is not."""

    runtime = FakeRuntime()
    state = {"poisoned": False}

    class _Group:
        def __init__(self, name: str, buffer: DeviceBuffer, guilty: bool) -> None:
            self.name, self.buffer, self.guilty = name, buffer, guilty

    innocent = DeviceBuffer(0x1000, 64)
    guilty = DeviceBuffer(0x2000, 64)

    def collect() -> dict:
        return {"innocent": [innocent], "guilty": [guilty]}

    def run() -> np.ndarray:
        # Depends only on the guilty group's poisoned-ness, which the fake
        # runtime records; a real runner depends on the bytes.
        return np.full(2, 1.0 if state["poisoned"] else 0.0, dtype=np.float32)

    def reset() -> None:
        state["poisoned"] = False

    def poison_tracking(rt: FakeRuntime, buffers, **kwargs) -> int:
        if guilty in buffers:
            state["poisoned"] = True
        return poison(rt, buffers, **kwargs)

    import _poison_probe as helper

    original = helper.poison
    helper.poison = poison_tracking
    try:
        with pytest.raises(AssertionError) as excinfo:
            assert_poison_invariant(runtime, collect, run, label="dirty", reset=reset)
    finally:
        helper.poison = original
    message = str(excinfo.value)
    assert "guilty" in message, message
    assert "offender groups" in message, message


def test_assert_poison_invariant_flags_non_finite_output() -> None:
    runtime = FakeRuntime()
    calls = {"n": 0}

    def run() -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 1:
            return np.zeros(4, dtype=np.float32)
        return np.full(4, np.nan, dtype=np.float32)

    with pytest.raises(AssertionError) as excinfo:
        assert_poison_invariant(runtime, lambda: {"s": [DeviceBuffer(0x1000, 4)]}, run, label="nan")
    assert "not finite" in str(excinfo.value)
