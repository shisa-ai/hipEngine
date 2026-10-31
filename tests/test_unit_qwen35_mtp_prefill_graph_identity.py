"""Prefill capture and later MTP admission must use the same profile identity."""

import os
from types import SimpleNamespace

import pytest

from hipengine.generation.qwen35_gguf import Qwen35GGUFResidentModelRunner
from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
from hipengine.runtime import gguf_native_spec_cycle as native


@pytest.mark.parametrize("capacity", [1, 4])
@pytest.mark.parametrize("manifest", ["strict-manifest", "production-manifest"])
def test_sampled_prefill_graph_survives_later_provider_profile_binding(
    capacity, manifest,
):
    target = SimpleNamespace(position=23, runner=SimpleNamespace(fp16_recurrent_state=False))
    captured = []

    def warm(tokens, *, request_id, sampled_accept, capture_prefill_gdn):
        assert tokens == (9, 0, 0)
        assert request_id == 7 and sampled_accept
        assert capture_prefill_gdn is True
        graph = object.__new__(native.Qwen35GGUFNativeB2TargetGraph)
        graph.session = target
        graph.closed = False
        graph.rows = len(tokens)
        graph.context_limit = 1024
        graph.binding_signature = native._native_target_binding_signature(target)
        profile, dtype = native._native_target_execution_identity(target)
        graph.configuration_key = native._native_target_configuration_key(
            **options, execution_profile_manifest_sha256=profile,
            recurrent_state_dtype=dtype,
        )
        target._native_spec_b2_target_graph_n2_sampled = graph
        captured.append(graph)

    target.prepare_native_spec_target_graph = warm
    row = SimpleNamespace(lease=SimpleNamespace(session=target))
    adapter = object.__new__(Qwen35GGUFMTP2Adapter)
    adapter.generator = SimpleNamespace(execution_profile_manifest_sha256=manifest)
    adapter.owner = SimpleNamespace(capacity=capacity, _row=lambda rid: row)
    adapter._intents = {7: 2}
    # Provider creation is deferred until the first cycle, after graph capture.
    adapter._states = {}
    adapter._prompt_hidden_rows = {}
    adapter._physical_c1_request = lambda rid: False
    adapter._sampled_accept_candidate = lambda rid: True
    options = dict(
        bulk_attention_mode="native", use_wmma_prefill=False,
        capture_linear_state_rows=True, capture_prefill_gdn=True,
        capture_pre_output_norm_hidden=True,
        defer_linear_state_commit=True, device_accept_commit=True, sampled_accept=True,
    )

    adapter.observe_prefill_result(7, (1, 2, 3), SimpleNamespace(token_id=9))
    assert len(captured) == 1
    # The actual provider-open path binds metadata here too. It must not change
    # the identity of a graph captured before the provider existed.
    adapter._bind_target_profile_metadata(target)
    graph = captured[0]
    assert graph.launch_ineligibility_reason(
        target, position=23, rows=3, remaining_decode=12, **options,
    ) is None
    assert graph.configuration_key[-3:-1] == (manifest, "fp32")
    # A second observation uses the existing capture rather than recapturing.
    adapter.observe_prefill_result(7, (1, 2, 3), SimpleNamespace(token_id=9))
    assert captured == [graph]


@pytest.mark.parametrize("initial", [None, "0", "1"])
@pytest.mark.parametrize("capture_fails", [False, True])
def test_prefill_graph_capture_uses_the_staged_cycle_environment(
    monkeypatch, initial, capture_fails,
):
    name = "HIPENGINE_GGUF_VERIFY_CAPTURE_PREFILL_GDN"
    if initial is None:
        monkeypatch.delenv(name, raising=False)
    else:
        monkeypatch.setenv(name, initial)
    monkeypatch.setenv("HIPENGINE_HIP_ARCH", "gfx1100")
    observed = []

    def capture(*args):
        observed.append((os.environ.get(name), os.environ.get("HIPENGINE_HIP_ARCH")))
        if capture_fails:
            raise RuntimeError("capture failed")

    runner = object.__new__(Qwen35GGUFResidentModelRunner)
    runner.generator = SimpleNamespace(target_arch="gfx1151")
    runner._mtp2_adapter_resolved = True
    runner._mtp2_adapter = SimpleNamespace(observe_prefill_result=capture)
    row = SimpleNamespace(mtp2_candidate_budget=2, request_id=7, prompt_ids=(1, 2, 3))
    if capture_fails:
        with pytest.raises(RuntimeError, match="capture failed"):
            runner._observe_mtp2_prefill(row, object())
    else:
        runner._observe_mtp2_prefill(row, object())
    with runner._staged_adapter_arch_environment():
        cycle_environment = (os.environ.get(name), os.environ.get("HIPENGINE_HIP_ARCH"))
    # The capture mode is an explicit operation input; the staged environment
    # carries only the target architecture.
    assert observed == [cycle_environment] == [(initial, "gfx1151")]
    assert os.environ.get(name) == initial
    assert os.environ["HIPENGINE_HIP_ARCH"] == "gfx1100"
