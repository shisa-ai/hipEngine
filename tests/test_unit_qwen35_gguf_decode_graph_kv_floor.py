"""KV-policy decode-graph capture floors (R0088 flag retirement).

``HIPENGINE_GGUF_INT8_KV_DECODE_GRAPH`` used to be the only way to admit the
IKV-C2 INT8 production layout (``int8_per_token_head`` storage, ``uniform``
layout, ``per_token_head`` scales) to C1 decode-graph capture. The layout is a
capture-path capability, not an arithmetic hazard, so admission is now
unconditional and the *horizon* is the gate: a backend declares a per-KV-policy
replay floor next to ``GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS``.

gfx1100 measured INT8 ``uniform``/``per_token_head`` capture as a repeatable
regression below the pm4 transport threshold (+1.36%/+0.81% at 32/64 steps) and
a win at and above it (-0.73% mean at 128), so it declares 128 — the threshold
for the dense 5120 geometry at one physical row. gfx1151's backend floor is
already 128, the horizon its INT8 capture measured neutral at, so it declares
nothing. BF16 and ``tail4_hadamard_group32`` keep the backend floor.

These tests pin the policy table's validity, both backends' resolved floors,
the unsupported layouts that stay eager, and — at the two real generation
sites — that nothing captures below the resolved floor.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import hipengine.runtime.qwen35_gguf_runner as gguf_runner
from hipengine.core.dtype import DType
from hipengine.generation import qwen35_gguf as gguf_generation
from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.policy import QWEN35_DENSE_H5120_GEOMETRY
from hipengine.kvcache import KV_STORAGE_TAIL4_HADAMARD_GROUP32
from hipengine.runtime.gguf_decode_graph import _decode_graph_kv_layout_admitted
from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFResidentSession

_FLOOR_POLICY_NAME = "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS_BY_KV_STORAGE"
_INT8_POLICY_KEY = ("int8_per_token_head", "uniform", "per_token_head")
_TAIL4_POLICY_KEY = ("int8_per_token_head", KV_STORAGE_TAIL4_HADAMARD_GROUP32, "hadamard_group32")


def _floor_session(
    *,
    backend: str,
    storage: DType = DType.BF16,
    layout: str = "uniform",
    granularity: str = "per_token_head",
) -> Qwen35GGUFResidentSession:
    """Minimum session shape ``_resolve_decode_graph_min_replay_steps`` reads."""

    session = object.__new__(Qwen35GGUFResidentSession)
    session.backend = "auto"
    session.runner = SimpleNamespace(
        backend=backend,
        weights=SimpleNamespace(
            weights=[SimpleNamespace(spec=SimpleNamespace(quant_key="gguf_q4_k_m_t16_v1"))]
        ),
    )
    session.scratch = SimpleNamespace()
    session.host_token_embedding_enabled = False
    session.use_gemv_decode = True
    session.kv_storage_dtype = storage
    session.kv_storage_layout = layout
    session.kv_scale_granularity = granularity
    return session


def _resolve(session: Qwen35GGUFResidentSession) -> int | None:
    return session._resolve_decode_graph_min_replay_steps()


def test_decode_graph_kv_floor_policy_is_a_validated_table() -> None:
    """A malformed declaration must raise instead of silently un-gating."""

    for backend in ("hip_gfx1100", "hip_gfx1151"):
        table = backend_package_capability(backend, _FLOOR_POLICY_NAME, {})
        assert isinstance(table, dict), backend
        for key, floor in table.items():
            assert isinstance(key, tuple) and len(key) == 3, (backend, key)
            assert all(isinstance(part, str) for part in key), (backend, key)
            assert isinstance(floor, int) and not isinstance(floor, bool), (backend, key)
            assert floor > 0, (backend, key)


@pytest.mark.parametrize(
    "bad_table",
    [
        pytest.param([("int8_per_token_head", "uniform", "per_token_head")], id="sequence"),
        pytest.param({_INT8_POLICY_KEY: 0}, id="zero"),
        pytest.param({_INT8_POLICY_KEY: -8}, id="negative"),
        pytest.param({_INT8_POLICY_KEY: True}, id="bool"),
        pytest.param({_INT8_POLICY_KEY: "128"}, id="string"),
    ],
)
def test_decode_graph_kv_floor_policy_rejects_malformed_declarations(
    monkeypatch: pytest.MonkeyPatch,
    bad_table: object,
) -> None:
    def capability(backend: str, name: str, default=None):
        if name == _FLOOR_POLICY_NAME:
            return bad_table
        return 24

    monkeypatch.setattr(gguf_runner, "backend_package_capability", capability)
    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend="hip_gfx1100", storage=DType.INT8_PER_TOKEN_HEAD
    )

    with pytest.raises(RuntimeError):
        _resolve(session)


def test_decode_graph_kv_floor_declarations_name_admitted_layouts() -> None:
    """A floor for a layout the capture path refuses would be dead config."""

    declared = 0
    for backend in ("hip_gfx1100", "hip_gfx1151"):
        table = backend_package_capability(backend, _FLOOR_POLICY_NAME, {})
        for storage, layout, granularity in table:
            session = _floor_session(
                backend=backend,
                storage=DType.parse(storage),
                layout=layout,
                granularity=granularity,
            )
            assert _decode_graph_kv_layout_admitted(session), (backend, storage, layout)
            declared += 1
    assert declared >= 1


def test_decode_graph_int8_floor_matches_the_measured_pm4_threshold() -> None:
    """The gfx1100 INT8 floor is the transport horizon the win was measured at."""

    policies = backend_package_capability(
        "hip_gfx1100", "GGUF_DECODE_GRAPH_SUBMISSION_POLICIES", {}
    )
    thresholds = policies[(QWEN35_DENSE_H5120_GEOMETRY, "MOSTLY_Q4_K_M")][
        "min_replay_steps_by_physical_rows"
    ]
    table = backend_package_capability("hip_gfx1100", _FLOOR_POLICY_NAME, {})

    assert thresholds[1] == 128
    assert table[_INT8_POLICY_KEY] == thresholds[1]


def test_decode_graph_kv_floor_raises_gfx1100_int8_above_the_backend_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend="hip_gfx1100", storage=DType.INT8_PER_TOKEN_HEAD
    )

    assert backend_package_capability("hip_gfx1100", "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS") == 24
    assert session._decode_graph_storage_admitted() is True
    assert _resolve(session) == 128


def test_decode_graph_kv_floor_keeps_the_gfx1151_backend_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """gfx1151 measured INT8 capture neutral at its own 128 floor."""

    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend="hip_gfx1151", storage=DType.INT8_PER_TOKEN_HEAD
    )

    assert backend_package_capability("hip_gfx1151", _FLOOR_POLICY_NAME, {}) == {}
    assert backend_package_capability("hip_gfx1151", "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS") == 128
    assert _resolve(session) == 128


@pytest.mark.parametrize(
    "backend,storage,layout,granularity,expected",
    [
        pytest.param(
            "hip_gfx1100", DType.BF16, "uniform", "per_token_head", 24, id="gfx1100-bf16"
        ),
        pytest.param(
            "hip_gfx1100",
            DType.INT8_PER_TOKEN_HEAD,
            KV_STORAGE_TAIL4_HADAMARD_GROUP32,
            "hadamard_group32",
            24,
            id="gfx1100-tail4",
        ),
        pytest.param(
            "hip_gfx1100",
            DType.INT8_PER_TOKEN_HEAD,
            "uniform",
            "per_token_head",
            128,
            id="gfx1100-uniform-per-token-head",
        ),
        pytest.param(
            "hip_gfx1151", DType.BF16, "uniform", "per_token_head", 128, id="gfx1151-bf16"
        ),
        pytest.param(
            "hip_gfx1151",
            DType.INT8_PER_TOKEN_HEAD,
            KV_STORAGE_TAIL4_HADAMARD_GROUP32,
            "hadamard_group32",
            128,
            id="gfx1151-tail4",
        ),
        pytest.param(
            "hip_gfx1151",
            DType.INT8_PER_TOKEN_HEAD,
            "uniform",
            "per_token_head",
            128,
            id="gfx1151-uniform-per-token-head",
        ),
    ],
)
def test_decode_graph_kv_floor_resolves_per_backend_and_storage_policy(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    storage: DType,
    layout: str,
    granularity: str,
    expected: int,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend=backend, storage=storage, layout=layout, granularity=granularity
    )
    backend_floor = backend_package_capability(backend, "GGUF_DECODE_GRAPH_MIN_REPLAY_STEPS")
    table = backend_package_capability(backend, _FLOOR_POLICY_NAME, {})
    key = (str(getattr(storage, "value", storage)), layout, granularity)

    assert _resolve(session) == expected
    # BF16 and tail4 keep the backend floor, and no backend declares them.
    if key != _INT8_POLICY_KEY:
        assert expected == backend_floor
        assert key not in table
    elif backend == "hip_gfx1100":
        assert expected > backend_floor
        assert table[key] == expected
    else:
        assert key not in table
        assert expected == backend_floor


@pytest.mark.parametrize("backend", ["hip_gfx1100", "hip_gfx1151"])
@pytest.mark.parametrize(
    "storage,layout,granularity",
    [
        pytest.param(DType.INT8_PER_TOKEN_HEAD, "uniform", "block16", id="uniform-block16"),
        pytest.param(
            DType.INT8_PER_TOKEN_HEAD, "uniform", "hadamard_group32", id="uniform-hadamard"
        ),
        pytest.param(
            DType.INT8_PER_TOKEN_HEAD,
            KV_STORAGE_TAIL4_HADAMARD_GROUP32,
            "block16",
            id="tail4-block16",
        ),
        pytest.param(
            DType.INT8_PER_TOKEN_HEAD,
            KV_STORAGE_TAIL4_HADAMARD_GROUP32,
            "per_token_head",
            id="tail4-per-token-head",
        ),
    ],
)
def test_decode_graph_kv_floor_never_admits_unsupported_layouts(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    storage: DType,
    layout: str,
    granularity: str,
) -> None:
    """A floor cannot admit a layout the capture path rejects."""

    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend=backend, storage=storage, layout=layout, granularity=granularity
    )

    assert session._decode_graph_storage_admitted() is False
    assert _resolve(session) is None


def test_decode_graph_int8_env_flag_no_longer_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The retired flag must not reappear as a second admission route."""

    import hipengine.runtime.gguf_decode_graph as graph_module

    assert not hasattr(graph_module, "_gguf_int8_kv_decode_graph_enabled")
    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend="hip_gfx1100", storage=DType.INT8_PER_TOKEN_HEAD
    )
    for value in ("0", "1"):
        monkeypatch.setenv("HIPENGINE_GGUF_INT8_KV_DECODE_GRAPH", value)
        assert session._decode_graph_storage_admitted() is True
        assert _resolve(session) == 128


def test_packed_decode_graph_stays_bf16_only_for_the_admitted_int8_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The packed floor inherits the session floor, but capture still refuses INT8."""

    from hipengine.runtime.gguf_packed_decode_graph import (
        capture_qwen35_gguf_packed_decode_graph,
    )

    monkeypatch.delenv("HIPENGINE_GGUF_MOE_GRAPH", raising=False)
    session = _floor_session(
        backend="hip_gfx1100", storage=DType.INT8_PER_TOKEN_HEAD
    )
    session._decode_graph_min_replay_steps_cache = _resolve(session)
    owner = SimpleNamespace(
        runner=SimpleNamespace(
            weights=SimpleNamespace(
                geometry=QWEN35_DENSE_H5120_GEOMETRY, file_type_name="MOSTLY_Q4_K_M"
            ),
            vocab_size=64,
        ),
        scratch=object(),
        _prefill_token_buf=object(),
        _bulk_prefill_scratch=object(),
        host_token_embedding_enabled=False,
        kv_storage_dtype=DType.INT8_PER_TOKEN_HEAD,
        use_expert_sidecar=False,
    )

    assert session._decode_graph_min_replay_steps_cache == 128
    # The packed floor is None for INT8, so no generation round ever attempts
    # a packed capture that the path cannot bind.
    assert session.packed_decode_graph_min_replay_steps(2) is None
    with pytest.raises(NotImplementedError, match="BF16 KV only"):
        capture_qwen35_gguf_packed_decode_graph(
            owner,
            token_ids=[5],
            sessions=[SimpleNamespace(position=8, runner=owner.runner, scratch=object())],
        )


class _FakeGraph:
    def __init__(self) -> None:
        self.replay_calls = 0
        self.closed = False

    def replay(self, steps: int) -> None:
        self.replay_calls += int(steps)

    def read_sample(self, *, return_logits: bool = False):
        return SimpleNamespace(token_id=9)

    def close(self) -> None:
        self.closed = True


class _FakeDecodeSession:
    def __init__(self, floor: int | None) -> None:
        self.position = 100
        self._floor = floor
        self.captures: list[dict[str, object]] = []
        self.eager_steps = 0
        self.graphs: list[_FakeGraph] = []

    def decode_graph_min_replay_steps(self) -> int | None:
        return self._floor

    def prefill(self, prompt_ids, *, return_logits: bool = False):
        return SimpleNamespace(token_id=7)

    def step(self, token, *, return_logits: bool = False):
        self.eager_steps += 1
        return SimpleNamespace(token_id=8)

    def capture_decode_graph(self, **kwargs):
        self.captures.append(dict(kwargs))
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph


# floor, remaining decode transitions, captured
_FLOOR_CASES = [
    pytest.param(128, 32, False, id="floor128-32"),
    pytest.param(128, 64, False, id="floor128-64"),
    pytest.param(128, 127, False, id="floor128-below-127"),
    pytest.param(128, 128, True, id="floor128-boundary-128"),
    pytest.param(128, 256, True, id="floor128-256"),
    pytest.param(200, 199, False, id="floor200-199"),
    pytest.param(200, 200, True, id="floor200-200"),
]


def _greedy_generator() -> gguf_generation.Qwen35GGUFBringupGenerator:
    generator = object.__new__(gguf_generation.Qwen35GGUFBringupGenerator)
    generator.tokenizer = SimpleNamespace(eos_token_id=-1)
    return generator


@pytest.mark.parametrize("floor,remaining,captured", _FLOOR_CASES)
def test_generate_greedy_never_captures_below_the_resolved_floor(
    monkeypatch: pytest.MonkeyPatch,
    floor: int,
    remaining: int,
    captured: bool,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_GRAPH", raising=False)
    session = _FakeDecodeSession(floor)
    request = SimpleNamespace(
        max_tokens=remaining + 1,
        ignore_eos=True,
        cancellation_token=None,
        deadline_at=None,
    )

    generated = _greedy_generator()._generate_greedy(session, [1, 2, 3], request)

    assert len(generated) == remaining + 1
    if captured:
        assert len(session.captures) == 1
        assert session.captures[0]["max_replay_steps"] == remaining
        assert session.captures[0]["steps_per_replay"] == 1
        assert session.graphs[0].replay_calls == remaining
        assert session.graphs[0].closed is True
        assert session.eager_steps == 0
    else:
        assert session.captures == []
        assert session.eager_steps == remaining


@pytest.mark.parametrize("floor,remaining,captured", _FLOOR_CASES)
def test_resident_c1_never_captures_below_the_resolved_floor(
    monkeypatch: pytest.MonkeyPatch,
    floor: int,
    remaining: int,
    captured: bool,
) -> None:
    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_GRAPH", raising=False)
    session = _FakeDecodeSession(floor)
    generated = [11] * 5
    slot = SimpleNamespace(
        session=session,
        prev_token=13,
        seq_position=100,
        generated_ids=generated,
        c1_decode_graph=None,
    )
    row = SimpleNamespace(slot=slot, request=SimpleNamespace(max_tokens=5 + remaining))
    runner = object.__new__(gguf_generation.Qwen35GGUFResidentModelRunner)
    runner._rows = {1: row}

    runner._step_native_c1_graph(row)

    if captured:
        assert len(session.captures) == 1
        assert session.captures[0]["max_replay_steps"] == remaining
        assert session.captures[0]["position"] == 100
        assert session.captures[0]["input_token_id"] == 13
        assert session.graphs[0].replay_calls == 1
        assert session.eager_steps == 0
    else:
        assert session.captures == []
        assert session.eager_steps == 1
