"""Compare tiled predictor cache construction with the full scalar predictor."""
import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer, malloc, free, copy_host_to_device, host_array_ptr
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner
from hipengine.runtime.qwen4_exp_mtp_prefill import Qwen4ExpMTPPrefillWorkspace
from tests.test_live_qwen4exp_mtp_cache_only import sidecar, _state, _bf16_values, _hip_available

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.mark.parametrize("mode", ["workspace", "device", "device_unfused", "device_wmma"])
@pytest.mark.parametrize("rows", [17, 65, 257, 2052])
def test_bulk_cache_and_final_carry_match_full_scalar(sidecar, rows, mode, monkeypatch):
    if mode == "device_unfused":
        from hipengine.runtime import qwen4_exp_runner
        monkeypatch.setattr(qwen4_exp_runner, "_qwen4_exp_gr_decode_fused_enabled",
                            lambda *args: False)
    if mode == "device_wmma":
        from hipengine.runtime import gguf_linear
        monkeypatch.setattr(gguf_linear, "_wmma_prefill_session_enabled", True)
    resident, config, runtime = sidecar
    rng = np.random.default_rng(rows + 923)
    tokens = rng.integers(1, config.vocab_size - 1, size=rows)
    hidden = _bf16_values(rng.normal(0, .25, (rows, config.residual_width)).astype(np.float32))
    shifted = np.concatenate((np.zeros((1, config.residual_width), dtype=np.float32), hidden[:-1]))
    runner = Qwen4ExpGGUFMTPDraftRunner(resident, target_config=config,
        max_sequence_length=rows + 4, backend="hip_gfx1151", runtime=runtime)
    workspace = device_hidden = None
    try:
        runner.reset()
        for token, seed in zip(tokens, shifted, strict=True):
            runner.forward(int(token), seed, run_head=False)
        expected = _state(runner)
        oracle = runner.propose_chain(start_token=42, target_hidden_seed=hidden[-1], draft_n_max=3)
        runner.reset()
        runtime.memset(runner.index.pooled_keys.ptr, 0xFF, runner.index.pooled_keys.nbytes)
        if mode == "workspace":
            workspace = Qwen4ExpMTPPrefillWorkspace(runner)
            for offset in range(0, rows - 1, workspace.rows):
                end = min(rows - 1, offset + workspace.rows)
                workspace.append(tokens[offset:end], shifted[offset:end])
                runner.position = runner.index.count
            runner.forward(int(tokens[-1]), shifted[-1], run_head=False)
        else:
            bits = np.ascontiguousarray(float_array_to_bf16_bits(hidden), dtype=np.uint16)
            device_hidden = malloc(bits.nbytes, runtime=runtime)
            copy_host_to_device(device_hidden, host_array_ptr(bits), bits.nbytes, runtime=runtime)
            previous = None
            stride = config.residual_width * 2
            for offset in range(0, rows, 103):
                end = min(rows, offset + 103)
                previous = runner.prime_prompt_device_chunk(tokens[offset:end],
                    DeviceBuffer(device_hidden.ptr + offset * stride, (end - offset) * stride),
                    previous_hidden=previous)
                np.testing.assert_array_equal(previous, hidden[end - 1])
        actual = _state(runner)
        for key, value in expected.items():
            if actual[key] != value:
                pytest.fail(f"bulk/scalar state mismatch: {key}")
        proposals = runner.propose_chain(start_token=42, target_hidden_seed=hidden[-1], draft_n_max=3)
        for left, right in zip(oracle, proposals, strict=True):
            assert left.token_id == right.token_id
            np.testing.assert_array_equal(left.logits, right.logits)
            np.testing.assert_array_equal(left.hidden_seed, right.hidden_seed)
    finally:
        if device_hidden is not None:
            free(device_hidden, runtime=runtime)
        if workspace is not None:
            workspace.close()
        runner.close()
