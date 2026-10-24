"""Public PARO oracle for cache-only prompt KV and final predictor state."""
from __future__ import annotations

from contextlib import closing
import ctypes
import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine import LLM, SamplingParams
from hipengine.core.memory import copy_device_to_host, host_array_ptr, malloc, memory_stats
from hipengine.generation.qwen35_paro import Qwen35ParoResidentModelRunner
from hipengine.generation.qwen35_paro_mtp2 import Qwen35ParoMTP2Adapter
from hipengine.speculative.mtp_native import NativeMtpChainProposer


def _hip_available():
    try:
        runtime = ctypes.CDLL("libamdhip64.so")
        count = ctypes.c_int()
        return runtime.hipGetDeviceCount(ctypes.byref(count)) == 0 and count.value > 0
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is unavailable")


def _digest(proposer, buffer, nbytes):
    array = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(array), buffer, nbytes, runtime=proposer.runtime)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _prompt_state(proposer):
    proposer.runtime.device_synchronize()
    return {
        "cache_len": proposer.cache_len,
        "position": proposer.position,
        "keys": _digest(proposer, proposer.key_cache_buf, proposer.cache_len * proposer.kv_features * 4),
        "values": _digest(proposer, proposer.value_cache_buf, proposer.cache_len * proposer.kv_features * 2),
        "hidden": _digest(proposer, proposer.final_hidden_buf, proposer.hidden * 2),
        "prediction": _digest(proposer, proposer.w8_out_index_buf, proposer.w8_out_index_buf.nbytes),
    }


def _routes(value):
    if isinstance(value, dict):
        if "specdec2_mtp2" in value and "routes" in value:
            return value["routes"]
        for child in value.values():
            found = _routes(child)
            if found is not None:
                return found
    return None


@pytest.fixture(scope="module")
def llm():
    model = Path(os.environ.get("HIPENGINE_TEST_PARO_MTP_MODEL", "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"))
    if not model.is_dir():
        pytest.skip("local packed PARO model with BF16 MTP weights is unavailable")
    initial = memory_stats()
    with closing(LLM(str(model), backend="auto", quant="w4_paro", max_active_requests=1,
                     max_sequence_length=4352, prefix_cache="off")) as owner:
        yield owner
    final = memory_stats()
    assert final["active_allocations"] == initial["active_allocations"]
    assert final["current_allocated_bytes"] == initial["current_allocated_bytes"]


def _legacy_packed_prompt_capture(owner, row, slab, *, sample_final):
    """Parent row-at-a-time normalization and provider path, kept as an oracle."""
    session = owner._session
    adapter = owner._resolved_mtp2_adapter()
    adapter.begin_prompt(row.request_id)
    if row.mtp2_prompt_hidden_buffer is None:
        row.mtp2_prompt_hidden_buffer = malloc(session.hidden_nbytes, runtime=session.runtime)
    pointer = int(row.mtp2_prompt_hidden_buffer.ptr)

    def consume(request_id, prompt_index, hidden, seed_token):
        assert request_id == row.request_id
        session._write_final_normalized_hidden_bf16(hidden, destination_ptr=pointer, stream=0)
        adapter.consume_prompt_row(request_id, prompt_index=prompt_index,
                                   target_hidden_ptr=pointer, seed_token=seed_token)

    return tuple(session.prefill_native_packed(slab, sample=sample_final, final_hidden_row_sink=consume))


@pytest.mark.parametrize("length", [1, 2, 15, 16, 17, 31, 32, 33, 37, 255, 256, 257, 511, 513, 1025, 4096])
def test_public_cache_only_prompt_matches_full_kv_final_state_and_outputs(llm, monkeypatch, length):
    original_append = NativeMtpChainProposer.append_prompt_kv
    bulk_capture = Qwen35ParoResidentModelRunner._prefill_row_packed_with_mtp2_capture
    original_observe = Qwen35ParoMTP2Adapter.observe_prefill_result
    captured = []
    appended = []

    def observe(adapter, request_id):
        original_observe(adapter, request_id)
        captured.append(_prompt_state(adapter._states[request_id].proposer))

    def full_priming(proposer, *, input_token, target_hidden_ptr, position, stream=0):
        # Independent pre-optimization oracle: full attention and expert work.
        proposer.advance_with_target_hidden(
            input_token=input_token, target_hidden_ptr=target_hidden_ptr, position=position,
            need_result=False, read_token_id=False, read_expert_topk=False,
            read_lm_head_value=False, stream=stream,
        )

    def cache_only(proposer, **kwargs):
        original_append(proposer, **kwargs)
        appended.append(kwargs["position"])

    monkeypatch.setattr(Qwen35ParoMTP2Adapter, "observe_prefill_result", observe)
    params = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True,
                            speculative_mtp=True, kv_storage="bf16")
    outputs = []
    acceptances = []
    for arm in ("full", "serial", "bulk", "bulk_repeat"):
        bulk = arm.startswith("bulk")
        monkeypatch.setattr(Qwen35ParoResidentModelRunner, "_prefill_row_packed_with_mtp2_capture",
                            bulk_capture if bulk else _legacy_packed_prompt_capture)
        monkeypatch.setattr(NativeMtpChainProposer, "append_prompt_kv",
                            full_priming if arm == "full" else cache_only)
        output = llm.generate_detailed([[9707] * length], params)[0]
        outputs.append(tuple(output.generated_token_ids))
        routes = _routes(llm.live_loop_snapshot())
        assert routes is not None
        assert not any(name.startswith("specdec2_mtp2_recovered") for name in routes["fallback_reasons"])
        route = routes["recent_completed"][-1]
        assert route["specdec2_mtp2_used"] and route["specdec2_mtp2_cycles"] > 0
        assert route["specdec2_mtp2_bulk_prompt_rows"] == (length - 1 if bulk else 0)
        assert bool(route["specdec2_mtp2_bulk_prompt_chunks"]) == (bulk and length > 1)
        acceptances.append(route["specdec2_mtp2_accepted_counts"])
    assert len(captured) == 4
    assert captured[0] == captured[1] == captured[2] == captured[3]
    assert captured[0]["cache_len"] == length
    assert outputs[0] == outputs[1] == outputs[2] == outputs[3]
    assert acceptances[0] == acceptances[1] == acceptances[2] == acceptances[3]
    assert len(outputs[0]) == 32
    assert appended == list(range(1, length))
