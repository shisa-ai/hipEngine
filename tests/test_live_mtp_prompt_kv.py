"""Real PARO prompt KV-only parity against the full-step priming parent."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.dtype import DType
from hipengine.core.memory import DeviceBuffer, copy_device_to_host, copy_host_to_device, free, host_array_ptr, malloc
from hipengine.speculative import mtp_native
from hipengine.speculative.mtp_native import NativeMtpChainProposer, _f32_to_bf16_bits


@pytest.fixture(scope="module", params=["private_fp16", "borrowed_w8"])
def proposers(request):
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        pytest.skip("HIP runtime is unavailable")
    model = Path(os.environ.get("HIPENGINE_TEST_PARO_MODEL", "/models/hipengine/Qwen3.6-35B-A3B-PARO-packed-MTP-BF16"))
    if not (model / "config.json").is_file():
        pytest.skip("PARO model fixture is unavailable")
    owners = []
    head_buffers = []
    try:
        head = None
        if request.param == "borrowed_w8":
            from hipengine.core.hip import get_hip_runtime
            from hipengine.loading import load_weight_index, qwen35_paro_config_from_hf
            from hipengine.speculative.mtp_native import NativeMtpW8A16Head
            config = qwen35_paro_config_from_hf(load_weight_index(model).config)
            runtime = get_hip_runtime()
            hidden, vocab = int(config.hidden_size), int(config.vocab_size)
            weight = malloc(vocab * hidden, runtime=runtime)
            scale = malloc(vocab * np.dtype(np.float32).itemsize, runtime=runtime)
            head_buffers.extend((weight, scale))
            runtime.memset(weight.ptr, 1, weight.nbytes)
            scales = np.full((vocab,), 0.0003, dtype=np.float32)
            copy_host_to_device(scale, host_array_ptr(scales), scales.nbytes, runtime=runtime)
            head = NativeMtpW8A16Head(weight.ptr, scale.ptr, vocab)
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(mtp_native, "_draft_vocab_from_env", lambda _vocab: 64)
            for _ in range(2):
                owners.append(NativeMtpChainProposer(model, max_positions=1024, max_mtp_tokens=512, scoring_head=head))
        yield tuple(owners)
    finally:
        for owner in reversed(owners):
            owner.close()
        for buffer in reversed(head_buffers):
            free(buffer, runtime=runtime)


def _read(owner, ptr, shape, dtype):
    host = np.empty(shape, dtype=dtype)
    copy_device_to_host(host_array_ptr(host), DeviceBuffer(ptr, host.nbytes), host.nbytes, runtime=owner.runtime)
    return host


def _assert_live_equal(parent, candidate):
    parent.runtime.device_synchronize()
    assert parent.cache_len == candidate.cache_len
    assert parent.position == candidate.position
    for name, dtype in (("key_cache_buf", np.float32), ("value_cache_buf", np.uint16)):
        shape = (parent.cache_len, parent.kv_features)
        a = _read(parent, getattr(parent, name).ptr, shape, dtype)
        b = _read(candidate, getattr(candidate, name).ptr, shape, dtype)
        assert a.tobytes() == b.tobytes(), name
        if dtype == np.float32:
            assert np.isfinite(a).all(), name
    a = parent.read_final_hidden_bf16()
    b = candidate.read_final_hidden_bf16()
    assert a.tobytes() == b.tobytes(), "final hidden"
    assert np.isfinite(mtp_native._bf16_bits_to_f32(a)).all()
    if parent.scoring_head is None:
        a = _read(parent, parent.out_index_buf.ptr, (1,), np.int64)
        b = _read(candidate, candidate.out_index_buf.ptr, (1,), np.int64)
    else:
        a = _read(parent, parent.device_candidate_token_ids().ptr, (1,), np.int32)
        b = _read(candidate, candidate.device_candidate_token_ids().ptr, (1,), np.int32)
    assert a.tobytes() == b.tobytes(), "candidate token"


@pytest.mark.parametrize("length", [1, 3, 17, 63, 129, 257])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_prompt_kv_only_preserves_full_prefix_final_hidden_and_continuation(proposers, length, asynchronous):
    parent, candidate = proposers
    for owner in proposers:
        owner.reset()
    rng = np.random.default_rng(913 + length)
    rows = _f32_to_bf16_bits(rng.normal(0, 0.4, (length + 4, parent.hidden)).astype(np.float32))
    buffer = malloc(rows.nbytes, runtime=parent.runtime)
    copy_host_to_device(buffer, host_array_ptr(rows), rows.nbytes, runtime=parent.runtime)
    try:
        for index in range(length):
            ptr = buffer.ptr + index * parent.hidden * DType.BF16.itemsize
            kwargs = dict(input_token=(index * 73 + 17) % parent.vocab, target_hidden_ptr=ptr, position=index + 1)
            parent.advance_with_target_hidden(**kwargs, need_result=index == length - 1, read_token_id=False, read_expert_topk=False, read_lm_head_value=False)
            if index == length - 1:
                candidate.advance_with_target_hidden(**kwargs, read_token_id=False, read_expert_topk=False, read_lm_head_value=False)
            else:
                if not asynchronous:
                    before = candidate.read_final_hidden_bf16().copy()
                candidate.append_prompt_kv_with_target_hidden(**kwargs)
                if not asynchronous:
                    after = candidate.read_final_hidden_bf16()
                    assert before.tobytes() == after.tobytes(), "KV-only append touched hidden"
        _assert_live_equal(parent, candidate)
        snapshots = [owner.save_state(0) for owner in proposers]
        # Both no-accept reseeding and accepted-token hidden-producing repair
        # must consume equivalent state after a prompt with omitted suffixes.
        for accepted in (False, True):
            for owner, snapshot in zip(proposers, snapshots, strict=True):
                owner.restore_state(snapshot)
                if accepted:
                    owner.advance_with_previous_hidden(input_token=31, position=owner.position + 1, need_result=False, read_expert_topk=False, read_lm_head_value=False)
                owner.advance_with_target_hidden(input_token=43, target_hidden_ptr=buffer.ptr + (length + int(accepted)) * parent.hidden * DType.BF16.itemsize, position=owner.position + 1, read_token_id=False, read_expert_topk=False, read_lm_head_value=False)
            _assert_live_equal(parent, candidate)
            for owner in proposers:
                owner.advance_with_previous_hidden(input_token=23, position=owner.position + 1, read_token_id=False, read_expert_topk=False, read_lm_head_value=False)
            _assert_live_equal(parent, candidate)
    finally:
        free(buffer, runtime=parent.runtime)
