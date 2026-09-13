from __future__ import annotations

import shlex
import sys
from pathlib import Path

from hipengine.loading.gguf import GGUFModelInfo, GGUFTensorInfo
from scripts import qwen35_gguf_bench as bench


def _tensor(
    name: str,
    *,
    shape: tuple[int, ...] = (2, 4),
    ggml_shape: tuple[int, ...] = (4, 2),
    ggml_type: int = 2,
    ggml_type_name: str = "Q4_K",
    n_elements: int = 8,
    nbytes: int = 144,
    offset: int = 0,
    data_offset: int = 4096,
    byte_shape: tuple[int, ...] = (2, 72),
) -> GGUFTensorInfo:
    return GGUFTensorInfo(
        name=name,
        shape=shape,
        ggml_shape=ggml_shape,
        ggml_type=ggml_type,
        ggml_type_name=ggml_type_name,
        n_elements=n_elements,
        nbytes=nbytes,
        offset=offset,
        data_offset=data_offset,
        byte_shape=byte_shape,
    )


def _model_info(*tensors: GGUFTensorInfo) -> GGUFModelInfo:
    return GGUFModelInfo(
        path=Path("/models/gguf/fake.gguf"),
        version=3,
        alignment=32,
        metadata={"general.architecture": "qwen3moe", "general.file_type": 15},
        tensors=tuple(tensors),
        tensor_data_offset=4096,
    )


def test_main_rejects_undersized_explicit_sequence_capacity(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qwen35_gguf_bench.py",
            "--prompt-length",
            "512",
            "--decode-tokens",
            "8",
            "--warmup-decode-tokens",
            "1",
            "--max-sequence-length",
            "520",
        ],
    )

    try:
        bench.main()
    except ValueError as exc:
        assert "below required 522" in str(exc)
    else:
        raise AssertionError("expected undersized explicit sequence capacity to fail")


def test_exact_command_payload_preserves_argv_and_shell_command() -> None:
    argv = ["python3", "scripts/qwen35_gguf_bench.py", "--model", "path with spaces.gguf"]

    payload = bench._exact_command_payload(argv)

    assert payload["argv"] == argv
    assert shlex.split(payload["command"]) == argv


def test_precomputed_compiler_version_bypasses_provenance_probe(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def collect(**kwargs):
        calls.append(kwargs)
        return {"kind": "test-provenance"}

    monkeypatch.setattr(bench, "collect_artifact_provenance", collect)
    result = bench._collect_benchmark_provenance(
        compiler_version="HIP version: cached",
        repo_root=Path("/repo"),
    )

    assert result == {"kind": "test-provenance"}
    assert calls == [
        {
            "repo_root": Path("/repo"),
            "hipcc_version": "HIP version: cached",
        }
    ]


def test_decode_graph_disabled_reason_tracks_production_graph_capability() -> None:
    class NoGraphSession:
        pass

    class GraphSession:
        def capture_decode_graph(self) -> None:  # pragma: no cover - only capability probe
            raise AssertionError("helper should not call capture_decode_graph")

    class HostEmbeddingGraphSession(GraphSession):
        host_token_embedding_enabled = True

    assert bench._decode_graph_disabled_reason(NoGraphSession(), requested=False) is None
    assert bench._decode_graph_disabled_reason(NoGraphSession(), requested=True) == "capture_decode_graph_unavailable"
    assert bench._decode_graph_disabled_reason(GraphSession(), requested=True) is None
    assert bench._decode_graph_disabled_reason(HostEmbeddingGraphSession(), requested=True) == "host_token_embedding"


def test_reset_existing_session_drains_prior_work_before_zeroing_state() -> None:
    calls: list[object] = []

    class Runtime:
        def stream_synchronize(self, stream: int) -> None:
            calls.append(("sync", stream))

    class Session:
        def reset(self) -> None:
            calls.append("reset")

    bench._reset_existing_session(Session(), Runtime())

    assert calls == [("sync", 0), "reset"]


def test_rearm_reused_decode_graph_resets_window_and_device_position() -> None:
    calls: list[object] = []

    class Graph:
        def rearm_replay_window(self) -> None:
            calls.append("rearm")

    class Runtime:
        def stream_create(self) -> int:
            calls.append("create")
            return 17

        def stream_synchronize(self, stream: int) -> None:
            calls.append(("sync", stream))

        def stream_destroy(self, stream: int) -> None:
            calls.append(("destroy", stream))

    class Session:
        position = 641

        def _set_full_attention_position_device(self, position: int, *, stream: int) -> None:
            calls.append(("position", position, stream))

    bench._rearm_reused_decode_graph(Session(), Graph(), Runtime())

    assert calls == [
        "rearm",
        "create",
        ("position", 641, 17),
        ("sync", 17),
        ("destroy", 17),
    ]


def test_gguf_tensor_inventory_hash_is_stable_and_metadata_sensitive() -> None:
    first = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor("blk.0.attn_q.weight", offset=144, data_offset=4240),
    )
    same = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor("blk.0.attn_q.weight", offset=144, data_offset=4240),
    )
    changed_shape = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor(
            "blk.0.attn_q.weight",
            shape=(2, 5),
            ggml_shape=(5, 2),
            n_elements=10,
            nbytes=160,
            offset=144,
            data_offset=4240,
            byte_shape=(2, 80),
        ),
    )
    changed_qtype = _model_info(
        _tensor("token_embd.weight", offset=0, data_offset=4096),
        _tensor(
            "blk.0.attn_q.weight",
            ggml_type=8,
            ggml_type_name="Q8_0",
            offset=144,
            data_offset=4240,
        ),
    )

    digest = bench._gguf_tensor_inventory_hash(first)

    assert len(digest) == 64
    assert digest == bench._gguf_tensor_inventory_hash(same)
    assert digest != bench._gguf_tensor_inventory_hash(changed_shape)
    assert digest != bench._gguf_tensor_inventory_hash(changed_qtype)


def test_gguf_tensor_inventory_summary_exposes_baseline_artifact_fields() -> None:
    info = _model_info(
        _tensor("token_embd.weight", nbytes=144, offset=0, data_offset=4096),
        _tensor("output.weight", nbytes=288, offset=144, data_offset=4240),
    )

    summary = bench._gguf_tensor_inventory_summary(info)

    assert summary["path"] == "/models/gguf/fake.gguf"
    assert summary["architecture"] == "qwen3moe"
    assert summary["tensor_count"] == 2
    assert summary["total_tensor_nbytes"] == 432
    assert summary["tensor_data_offset"] == 4096
    assert summary["tensor_inventory_hash_algorithm"] == "sha256"
    assert summary["tensor_inventory_hash"] == bench._gguf_tensor_inventory_hash(info)


def test_default_decode_graph_request_mirrors_engine_admission(monkeypatch) -> None:
    """An unset sweep flag must reproduce the engine's own graph decision."""

    class Session:
        def __init__(self, minimum: int | None) -> None:
            self._minimum = minimum

        def decode_graph_min_replay_steps(self) -> int | None:
            return self._minimum

    monkeypatch.delenv("HIPENGINE_GGUF_DECODE_GRAPH", raising=False)
    # Backend publishes no break-even: the engine never captures a graph.
    assert bench._default_decode_graph_request(Session(None), 4096) is False
    # Below the published floor the engine stays eager, at or above it captures.
    assert bench._default_decode_graph_request(Session(128), 127) is False
    assert bench._default_decode_graph_request(Session(128), 128) is True
    assert bench._default_decode_graph_request(Session(128), 4096) is True
    # The explicit opt-out disables the graph regardless of the floor.
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_GRAPH", "0")
    assert bench._default_decode_graph_request(Session(128), 4096) is False
    monkeypatch.setenv("HIPENGINE_GGUF_DECODE_GRAPH", "1")
    assert bench._default_decode_graph_request(Session(128), 4096) is True


def test_default_decode_graph_request_tolerates_absent_capability() -> None:
    class Bare:
        pass

    assert bench._default_decode_graph_request(Bare(), 512) is False


def test_decode_graph_disabled_reason_reports_backend_floor() -> None:
    class GraphSession:
        def __init__(self, minimum: int | None) -> None:
            self._minimum = minimum

        def capture_decode_graph(self) -> None:  # pragma: no cover - capability probe only
            raise AssertionError("helper should not call capture_decode_graph")

        def decode_graph_min_replay_steps(self) -> int | None:
            return self._minimum

    session = GraphSession(128)
    # Without a decode length the floor is not evaluated (existing callers).
    assert bench._decode_graph_disabled_reason(session, requested=True) is None
    assert (
        bench._decode_graph_disabled_reason(session, requested=True, decode_tokens=64)
        == "below_backend_min_replay_steps"
    )
    assert bench._decode_graph_disabled_reason(session, requested=True, decode_tokens=128) is None
    # A backend without a floor keeps the previous behaviour.
    assert bench._decode_graph_disabled_reason(GraphSession(None), requested=True, decode_tokens=8) is None
