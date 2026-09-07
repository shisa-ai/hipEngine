"""Synthetic native acceptance ABI coverage, not model/state qualification."""
import ctypes
from types import SimpleNamespace as NS
import numpy as np
import pytest


@pytest.fixture(scope='module')
def runtime():
    try:
        ctypes.CDLL('libamdhip64.so')
        from hipengine.core.hip import get_hip_runtime
        rt = get_hip_runtime()
        rt.current_device()
    except (OSError, RuntimeError):
        pytest.skip('HIP unavailable')
    return rt


@pytest.mark.parametrize('padding', [0, 3])
@pytest.mark.parametrize('depth', range(1, 8))
def test_native_c1_all_rejections_horizons_and_eos(runtime, depth, padding):
    from hipengine.core.memory import malloc, free, copy_host_to_device, copy_device_to_host
    from hipengine.kernels.hip_gfx1100.speculative import dflash_accept_chain_i32_packed
    from hipengine.generation.qwen35_gguf_mtp2 import Qwen35GGUFMTP2Adapter
    from hipengine.generation.engine_service import EngineService
    from hipengine.generation.registry import GenerationOutput, FinishDetails
    rows = depth + 1 + padding
    # Distinct synthetic IDs are fixtures, never benchmark candidate reranks.
    tokens = np.arange(100, 100 + rows, dtype=np.int32)
    tokens[depth + 1:] = 900  # Would falsely extend full acceptance if padding were active.
    buffers = []
    def upload(values, dtype=np.int32):
        host = np.asarray(values, dtype=dtype)
        buf = malloc(host.nbytes, runtime=runtime)
        buffers.append(buf)
        copy_host_to_device(buf, host.ctypes.data, runtime=runtime)
        return buf
    try:
        token = upload(tokens)
        position = upload(np.arange(73, 73 + rows))
        parent = upload(np.arange(-1, rows - 1))
        depths = upload(np.arange(rows))
        mask = upload([1] * (depth + 1) + [0] * padding, np.uint8)
        top = upload(np.zeros(rows))
        remaining = upload([0])
        outputs = [upload([0]) for _ in range(5)]
        full = upload([0], np.uint8)
        committed = upload(np.full(rows, -1))
        length = upload([0])
        payload = upload(np.zeros(7))
        for reject in range(depth + 1):  # depth means all accepted
            expected_top = np.append(tokens[1:], 900).astype(np.int32)
            expected_top[reject] = 900
            copy_host_to_device(top, expected_top.ctypes.data, runtime=runtime)
            for horizon in range(1, depth + 3):
                h = np.asarray([horizon], dtype=np.int32)
                copy_host_to_device(remaining, h.ctypes.data, runtime=runtime)
                dflash_accept_chain_i32_packed(
                    token.ptr, position.ptr, parent.ptr, depths.ptr, mask.ptr, top.ptr,
                    remaining.ptr, *(b.ptr for b in outputs), full.ptr,
                    committed.ptr, length.ptr, payload.ptr, rows, 1, rows, runtime=runtime)
                accepted = min(reject, horizon - 1)
                status = np.empty(7, dtype=np.int32)
                copy_device_to_host(status.ctypes.data, payload, runtime=runtime)
                assert status.tolist() == [accepted, accepted, int(tokens[accepted]),
                    73 + accepted, int(expected_top[accepted]), int(accepted == depth), accepted + 1]
                raw_ids = np.empty(rows, dtype=np.int32)
                copy_device_to_host(raw_ids.ctypes.data, committed, runtime=runtime)
                assert raw_ids.tolist() == tokens[:accepted + 1].tolist() + [-1] * (rows - accepted - 1)
                pending = NS(request_count=1, output_stride=rows, payload=payload,
                    buffers=NS(committed_output_ids=committed, transaction_id=19),
                    batch=NS(candidate_counts=(depth,), request_ids=(41,), draft_depth=depth,
                             tree_shape=tuple(range(depth)), mode='verify_chain'))
                summary = Qwen35GGUFMTP2Adapter._read_target_batch_accept(pending, runtime=runtime)
                assert summary.accepted_tokens == (tuple(map(int, tokens[1:accepted + 1])),)
                assert summary.next_tokens == (int(expected_top[accepted]),)
                visible = summary.accepted_tokens[0] + summary.next_tokens
                for eos_index, eos in enumerate(visible):
                    state = NS(request=NS(max_tokens=horizon, min_tokens=0,
                        eos_token_id=eos, stop_token_ids=(), stop_token_sequences=(), ignore_eos=False))
                    service = NS(_driver=NS(detokenize=lambda ids: str(tuple(ids))))
                    output = GenerationOutput(text=str(visible), generated_token_ids=visible,
                        token_logprobs=tuple(-float(i + 1) for i in range(len(visible))),
                        finish_details=FinishDetails(reason='length'))
                    tail = EngineService._normalize_speculative_output(service, state, output)
                    assert tail.generated_token_ids == visible[:eos_index + 1]
                    assert tail.finish_details.reason == 'eos'
                    assert tail.token_logprobs == output.token_logprobs[:eos_index + 1]
                    assert tail.text == str(visible[:eos_index + 1])
    finally:
        for buf in reversed(buffers):
            free(buf, runtime=runtime)
