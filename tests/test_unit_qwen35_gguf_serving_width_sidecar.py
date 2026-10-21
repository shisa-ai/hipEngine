"""Serving-width-scoped weight layouts for the GGUF resident generator.

The raw recurrent-output Q5 sidecar duplicates every ``ssm_out`` tensor
(about 1 GiB on the 27B) for the eight-request MTP verify owner. A server whose
request width cannot reach eight never reads it, so it must not be loaded.
"""

from types import SimpleNamespace

import pytest

from hipengine import llm as llm_module
from hipengine.generation import qwen35_gguf


@pytest.mark.parametrize(
    "bound, expected",
    [
        (None, {}),
        (1, {"dense_q5_raw_mmq_ssm_out": False, "dense_q5_qmicro_planar_ssm_out": False}),
        (4, {"dense_q5_raw_mmq_ssm_out": False, "dense_q5_qmicro_planar_ssm_out": False}),
        (7, {"dense_q5_raw_mmq_ssm_out": False, "dense_q5_qmicro_planar_ssm_out": False}),
        (8, {}),
        (32, {}),
    ],
)
def test_sidecar_overrides_follow_the_serving_width(bound, expected) -> None:
    assert qwen35_gguf._gguf_serving_width_dense_flag_overrides(bound) == expected


def _generator():
    return qwen35_gguf.Qwen35GGUFBringupGenerator.__new__(
        qwen35_gguf.Qwen35GGUFBringupGenerator
    )


def test_unpinned_capacity_bounds_width_by_the_automatic_default() -> None:
    generator = _generator()
    generator.declare_serving_capacity(None)
    assert generator._serving_capacity_bound == qwen35_gguf._GGUF_RESIDENT_MODEL_LOOP_DEFAULT_CAPACITY
    generator.declare_serving_capacity(16)
    assert generator._serving_capacity_bound == 16


def test_declaration_after_the_weights_loaded_changes_nothing() -> None:
    generator = _generator()
    generator._shared_runner = object()
    generator.declare_serving_capacity(1)
    assert getattr(generator, "_serving_capacity_bound", None) is None


@pytest.mark.parametrize(
    "declared, sidecar",
    [(None, False), (4, False), (8, True), ("undeclared", True)],
)
def test_shared_runner_receives_the_width_overrides(monkeypatch, declared, sidecar) -> None:
    captured = {}

    def fake_runner(model_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(qwen35_gguf, "Qwen35GGUFFullStackRunner", fake_runner)
    generator = _generator()
    generator.model_path = "/tmp/fake.gguf"
    generator.backend = "hip_gfx1100"
    generator.native_batch_decode = False
    if declared != "undeclared":
        generator.declare_serving_capacity(declared)
    generator._get_shared_runner()
    overrides = captured["dense_flag_overrides"]
    assert ("dense_q5_raw_mmq_ssm_out" not in overrides) is sidecar


@pytest.mark.parametrize("pinned", [None, 2, 8])
def test_llm_declares_its_pinned_capacity(pinned) -> None:
    seen = []
    generator = SimpleNamespace(declare_serving_capacity=seen.append)
    llm_module._declare_serving_capacity(generator, pinned)
    assert seen == [pinned]
    # Generators without the hook are unaffected.
    llm_module._declare_serving_capacity(SimpleNamespace(), pinned)


# ---------------------------------------------------------------------------
# Shared-runner token table placement
# ---------------------------------------------------------------------------


def _fake_reader(type_name):
    from hipengine.runtime import qwen35_gguf_runner as gguf_runner

    class Reader:
        def __init__(self, path):
            self.info = SimpleNamespace(tensors=(
                SimpleNamespace(name="output.weight", ggml_type_name="Q6_K"),
                SimpleNamespace(name="token_embd.weight", ggml_type_name=type_name),
            ))

    return gguf_runner, Reader


@pytest.mark.parametrize(
    "type_name, env, owned, expected",
    [
        ("Q4_K", None, False, "host"),
        ("Q4_K", "0", False, "device"),
        ("Q4_K", "device", False, "device"),
        ("Q4_K", None, True, "device"),
        ("Q6_K", None, False, "device"),
    ],
)
def test_shared_token_table_maps_from_host_when_the_backend_declares_it(
    monkeypatch, type_name, env, owned, expected
) -> None:
    gguf_runner, reader = _fake_reader(type_name)
    monkeypatch.setattr(gguf_runner, "GGUFReader", reader)
    if env is None:
        monkeypatch.delenv("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING", raising=False)
    else:
        monkeypatch.setenv("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING", env)
    placement = gguf_runner._gguf_shared_token_embedding_placement(
        backend="hip_gfx1100", model_path="/tmp/fake.gguf", externally_owned=owned,
    )
    assert placement == expected


def test_backend_without_the_mapped_capability_keeps_the_table_on_device(monkeypatch) -> None:
    gguf_runner, reader = _fake_reader("Q4_K")
    monkeypatch.setattr(gguf_runner, "GGUFReader", reader)
    monkeypatch.delenv("HIPENGINE_GGUF_HOST_TOKEN_EMBEDDING", raising=False)
    monkeypatch.setattr(
        gguf_runner, "_gguf_mapped_host_token_embedding_enabled", lambda **kwargs: False
    )
    assert gguf_runner._gguf_shared_token_embedding_placement(
        backend="hip_gfx1100", model_path="/tmp/fake.gguf", externally_owned=False,
    ) == "device"


def test_shared_runner_requests_automatic_token_table_placement(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        qwen35_gguf, "Qwen35GGUFFullStackRunner",
        lambda model_path, **kwargs: captured.update(kwargs) or SimpleNamespace(),
    )
    generator = _generator()
    generator.model_path = "/tmp/fake.gguf"
    generator.backend = "hip_gfx1100"
    generator.native_batch_decode = False
    generator._get_shared_runner()
    assert captured["token_embedding_placement"] == "auto"
