from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, host_array_ptr
from hipengine.kernels.cpu_reference.qwen4_exp import qwen4_exp_mtp_fuse_inputs
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
from hipengine.loading.qwen4_exp_mtp_gguf import build_qwen4_exp_mtp_gguf_map
from hipengine.loading.materialize import float_array_to_bf16_bits
from hipengine.loading.qwen4_exp_mtp_materialize import (
    materialize_qwen4_exp_mtp_weights,
    plan_qwen4_exp_mtp_residency,
)
from hipengine.quant.gguf import GGMLQuantizationType, dequantize_gguf_data
from hipengine.runtime.qwen4_exp_mtp import (
    Qwen4ExpGGUFMTPDraftRunner,
    Qwen4ExpMTPDraftResult,
)

_TARGET = Path(
    "/models/gguf/unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
)
_SIDECAR = Path(
    "/models/gguf/Qwen3.8-Flash-Next-MTP-Q8_0/"
    "mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


def _bf16_values(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(float_array_to_bf16_bits(values), dtype=np.uint16)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def test_qwen4_exp_mtp_input_reference_normalizes_globally_and_pairs_branches() -> None:
    embedding = np.asarray([3.0, 4.0], dtype=np.float32)
    hidden = np.asarray([1.0, 2.0, 5.0, 6.0], dtype=np.float32)
    fused = qwen4_exp_mtp_fuse_inputs(
        embedding,
        hidden,
        np.asarray([2.0, 3.0], dtype=np.float32),
        np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        branches=2,
        eps=0.0,
    )

    assert fused.shape == (2, 4)
    np.testing.assert_array_equal(fused[0, :2], fused[1, :2])
    hidden_scale = np.sqrt(np.mean(hidden * hidden, dtype=np.float32))
    np.testing.assert_allclose(
        fused[:, 2:],
        (hidden / hidden_scale * np.asarray([1.0, 2.0, 3.0, 4.0])).reshape(2, 2),
        rtol=1e-6,
        atol=1e-6,
    )
    with pytest.raises(ValueError, match="one embedding-width row per branch"):
        qwen4_exp_mtp_fuse_inputs(embedding, hidden[:3], [1.0, 1.0], [1.0] * 3, branches=2)


def test_qwen4_exp_mtp_draft_result_contract() -> None:
    logits = np.asarray([0.0, 2.0, 1.0], dtype=np.float32)
    hidden = np.arange(16, dtype=np.float32)
    result = Qwen4ExpMTPDraftResult(1, logits, hidden)

    assert result.token_id == 1
    np.testing.assert_array_equal(result.logits, logits)
    np.testing.assert_array_equal(result.hidden_seed, hidden)
    compact = Qwen4ExpMTPDraftResult(1, None, None)
    assert compact.logits is None
    assert compact.hidden_seed is None


@pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")
@pytest.mark.skipif(
    not (_TARGET.exists() and _SIDECAR.exists()),
    reason="real Qwen4Exp target and MTP sidecar are local-only",
)
def test_real_qwen4_exp_mtp_draft_is_deterministic_and_transactional() -> None:
    from hipengine.core.hip import get_hip_runtime

    runtime = get_hip_runtime()
    target_info = GGUFReader(discover_gguf_files(_TARGET)[0]).info
    target_config = qwen4_exp_gguf_config_from_metadata(target_info)
    reader = GGUFReader(_SIDECAR)
    model_map = build_qwen4_exp_mtp_gguf_map((reader.info,))
    plan = plan_qwen4_exp_mtp_residency(model_map)
    resident = materialize_qwen4_exp_mtp_weights(
        (reader,), plan=plan, backend="hip_gfx1151", runtime=runtime
    )
    runner = None
    try:
        runner = Qwen4ExpGGUFMTPDraftRunner(
            resident,
            target_config=target_config,
            max_sequence_length=16,
            backend="hip_gfx1151",
            runtime=runtime,
        )
        token_id = 248_068
        rng = np.random.default_rng(380)
        hidden = _bf16_values(
            rng.normal(0.0, 0.25, size=target_config.residual_width).astype(np.float32)
        )
        runner._fuse_inputs(token_id, hidden)
        fused_bits = np.empty(
            (target_config.residual_branch_count, 2 * target_config.hidden_size),
            dtype=np.uint16,
        )
        copy_device_to_host(
            host_array_ptr(fused_bits),
            runner.fused_input,
            fused_bits.nbytes,
            runtime=runtime,
        )
        fused_gpu = (fused_bits.astype(np.uint32) << 16).view(np.float32)
        embedding = _bf16_values(
            dequantize_gguf_data(
                np.asarray(reader.tensor_data("token_embd.weight")[token_id]),
                GGMLQuantizationType.Q8_0,
            ).astype(np.float32)
        )
        fused_ref = qwen4_exp_mtp_fuse_inputs(
            embedding,
            hidden,
            np.asarray(reader.tensor_data("blk.48.nextn.enorm.weight"), dtype=np.float32),
            np.asarray(reader.tensor_data("blk.48.nextn.hnorm.weight"), dtype=np.float32),
            branches=target_config.residual_branch_count,
            eps=target_config.attention_rms_epsilon,
        )
        np.testing.assert_allclose(
            fused_gpu,
            _bf16_values(fused_ref),
            rtol=1e-2,
            atol=2e-3,
        )

        runner.reset()
        first = runner.forward(token_id, hidden)
        runner.reset()
        repeat = runner.forward(token_id, hidden)

        assert first.token_id == repeat.token_id
        assert np.isfinite(first.logits).all()
        assert np.isfinite(first.hidden_seed).all()
        np.testing.assert_array_equal(first.logits, repeat.logits)
        np.testing.assert_array_equal(first.hidden_seed, repeat.hidden_seed)

        prompt_hidden = np.stack((hidden, first.hidden_seed), axis=0)
        runner.prime_prompt((248_068, first.token_id), prompt_hidden)
        checkpoint = runner.snapshot()
        chain = runner.propose_chain(
            start_token=repeat.token_id,
            target_hidden_seed=repeat.hidden_seed,
            draft_n_max=2,
            compact_output=True,
        )
        assert runner.position == checkpoint.position + 2
        runner.restore(checkpoint)
        replay = runner.propose_chain(
            start_token=repeat.token_id,
            target_hidden_seed=repeat.hidden_seed,
            draft_n_max=2,
            compact_output=True,
        )
        assert [row.token_id for row in replay] == [row.token_id for row in chain]
        assert all(row.logits is None for row in chain)
        assert all(row.hidden_seed is None for row in chain)
        assert all(row.logits is None for row in replay)
        assert all(row.hidden_seed is None for row in replay)
        assert "draft_logits_d2h" not in runner.last_proposal_stage_timings_ms
        assert "draft_hidden_d2h" not in runner.last_proposal_stage_timings_ms
        assert "draft_device_argmax_and_token_d2h" in runner.last_proposal_stage_timings_ms
        runner.restore(checkpoint)
        debug = runner.propose_chain(
            start_token=repeat.token_id,
            target_hidden_seed=repeat.hidden_seed,
            draft_n_max=2,
            compact_output=False,
        )
        assert [row.token_id for row in debug] == [row.token_id for row in chain]
        assert all(row.logits is not None for row in debug)
        assert all(row.hidden_seed is not None for row in debug)
    finally:
        if runner is not None:
            runner.close()
        resident.close()
