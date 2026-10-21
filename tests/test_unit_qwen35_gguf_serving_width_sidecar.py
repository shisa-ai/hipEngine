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

