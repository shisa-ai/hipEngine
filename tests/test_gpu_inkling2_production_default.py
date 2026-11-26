"""An omitted Inkling GGUF profile runs the production plan where that plan is registered.

The generator reads its route flags when it is built, so each case opens the model again. These
tests need ROCm and the GGUF of `HIPENGINE_INKLING2_MINI_GGUF`.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from hipengine import LLM, SamplingParams
from hipengine.generation.inkling2_gguf import PRODUCTION_BACKENDS, ROUTE_DEVICE, ROUTE_DECODE_GRAPH, ROUTE_RESIDENT_DECODE

ROUTE_FLAGS = (ROUTE_DEVICE, ROUTE_RESIDENT_DECODE, ROUTE_DECODE_GRAPH)
PROMPT = "<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>" \
         "Name three primary colors.<|end_message|><|message_model|><|content_text|>"


def _skip_unless_runnable() -> str:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("ROCm runtime is not available")
    model = os.environ.get("HIPENGINE_INKLING2_MINI_GGUF", "")
    if not model or not Path(model).is_file():
        pytest.skip("HIPENGINE_INKLING2_MINI_GGUF must point at the Inkling 2 GGUF model")
    return model


@pytest.fixture
def route_environment(monkeypatch):
    for name in ROUTE_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HIPENGINE_EXECUTION_PROFILE", raising=False)


def _routes(llm) -> dict[str, bool]:
    generator = llm._get_text_generator()
    model = generator.model
    projection = model.projection
    return {
        "device_routes": bool(projection.device_routes_enabled()),
        "resident_decode": bool(model.resident_decode),
        "decode_graph": bool(model.decode_graph),
        "q8_dot_experts": projection._q8_dot is not None,
        "wide_dense": projection.wide_dense,
    }


def _open(model, **kwargs):
    kwargs = {"model": model, "prefix_cache": "off", "speculative_mtp_serving": "off", "max_sequence_length": 512, **kwargs}
    llm = LLM(**kwargs)
    llm._get_text_generator()
    return llm


def _generate(llm, tokens=16) -> tuple[int, ...]:
    ids = llm.tokenize(PROMPT)
    output, = llm.generate_detailed(ids, SamplingParams(temperature=0, max_tokens=tokens))
    return tuple(output.generated_token_ids)


def _shipped(llm) -> bool:
    return llm.resolved_backend in PRODUCTION_BACKENDS


def test_omitted_profile_runs_the_production_plan(route_environment):
    model = _skip_unless_runnable()
    llm = _open(model)
    try:
        shipped = _shipped(llm)
        assert llm.resolved_execution_profile == ("production" if shipped else None)
        assert not llm.execution_profile_fell_back_to_strict
        routes = _routes(llm)
        assert routes == {
            "device_routes": shipped, "resident_decode": shipped, "decode_graph": shipped, "q8_dot_experts": False,
            "wide_dense": "dense_wide64x128_residual_f32_f32_out" if shipped else None,
        }
        if shipped:
            # The plan names the resident single-token graph and keeps F32 experts on every path.
            manifest = llm.execution_profile_manifest
            assert manifest["graph_policy"] == "resident_single_token_whole_token_graph"
            selected = {(item["layer"], item["scope"]): item["selected_variant"] for item in manifest["selections"]}
            assert not any("q8_dot" in variant for (layer, _), variant in selected.items() if layer == "moe_linear")
        # Long enough for the first eager step, the capture and replays.
        assert len(_generate(llm)) == 16
    finally:
        llm.close()


@pytest.mark.parametrize("flag", [None, ROUTE_RESIDENT_DECODE, ROUTE_DECODE_GRAPH, ROUTE_DEVICE])
def test_explicit_production_and_its_levers(route_environment, flag):
    model = _skip_unless_runnable()
    reference = None
    if flag is None:
        first = _open(model)
        try:
            reference = _generate(first)
        finally:
            first.close()
    if flag is not None:
        os.environ[flag] = "0"
    try:
        llm = _open(model, execution_profile="production")
        try:
            assert llm.resolved_execution_profile == "production" and not llm.execution_profile_fell_back_to_strict
            assert _routes(llm)["q8_dot_experts"] is False
            routes = _routes(llm)
            assert routes["wide_dense"] == "dense_wide64x128_residual_f32_f32_out"
            ids = _generate(llm)
            assert len(ids) == 16
            if flag is None:
                # Naming the plan and omitting the profile are the same route.
                assert ids == reference
            else:
                assert routes[{ROUTE_RESIDENT_DECODE: "resident_decode", ROUTE_DECODE_GRAPH: "decode_graph",
                               ROUTE_DEVICE: "device_routes"}[flag]] is False
                if flag == ROUTE_RESIDENT_DECODE:
                    # Nothing records a graph for a step that does not run resident.
                    assert routes["decode_graph"] is True
        finally:
            llm.close()
    finally:
        if flag is not None:
            os.environ.pop(flag, None)


def test_strict_stays_the_reference(route_environment):
    model = _skip_unless_runnable()
    llm = _open(model, execution_profile="strict")
    try:
        assert llm.resolved_execution_profile == "strict" and not llm.execution_profile_fell_back_to_strict
        assert _routes(llm) == {
            "device_routes": False, "resident_decode": False, "decode_graph": False, "q8_dot_experts": False,
            "wide_dense": None,
        }
        assert llm.execution_profile_manifest["graph_policy"] == "eager"
        assert len(_generate(llm)) == 16
    finally:
        llm.close()
