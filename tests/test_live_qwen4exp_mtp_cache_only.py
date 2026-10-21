"""Real sidecar oracle for exact cache-only priming, carry and proposal replay."""
from pathlib import Path

import numpy as np
import pytest

from hipengine.core.memory import copy_device_to_host, host_array_ptr
from hipengine.loading.gguf import GGUFReader, discover_gguf_files
from hipengine.loading.qwen4_exp_gguf import qwen4_exp_gguf_config_from_metadata
from hipengine.loading.qwen4_exp_mtp_gguf import build_qwen4_exp_mtp_gguf_map
from hipengine.loading.qwen4_exp_mtp_materialize import (
    materialize_qwen4_exp_mtp_weights, plan_qwen4_exp_mtp_residency,
)
from hipengine.runtime.qwen4_exp_mtp import Qwen4ExpGGUFMTPDraftRunner
from tests.test_live_qwen4_exp_mtp_runner import _hip_available, _bf16_values

pytestmark = pytest.mark.skipif(not _hip_available(), reason="HIP runtime is not available")


@pytest.fixture(scope="module")
def sidecar():
    root = Path.home() / "models/gguf"
    target = root / "unsloth-Qwen3.8-Flash-Next-UD-Q4_K_XL/UD-Q4_K_XL"
    path = root / "Qwen3.8-Flash-Next-MTP-Q8_0/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
    if not target.is_dir() or not path.is_file():
        pytest.skip("real Flash-Next target metadata and MTP sidecar are local-only")
    from hipengine.core.hip import get_hip_runtime
    runtime = get_hip_runtime()
    config = qwen4_exp_gguf_config_from_metadata(GGUFReader(discover_gguf_files(target)[0]).info)
    reader = GGUFReader(path)
    plan = plan_qwen4_exp_mtp_residency(build_qwen4_exp_mtp_gguf_map((reader.info,)))
    resident = materialize_qwen4_exp_mtp_weights((reader,), plan=plan, backend="hip_gfx1151", runtime=runtime)
    try:
        yield resident, config, runtime
    finally:
        resident.close()


def _bytes(buffer, nbytes, runtime):
    if nbytes == 0:
        return b""
    array = np.empty(nbytes, dtype=np.uint8)
    copy_device_to_host(host_array_ptr(array), buffer, nbytes, runtime=runtime)
    return array.tobytes()


def _state(runner):
    n, config, runtime = runner.position, runner.config, runner.runtime
    kv_bytes = n * config.attention_kv_head_count * config.attention_key_length * 2
    return {
        "position": n, "index_count": runner.index.count, "pooled_count": runner.index.pooled_count,
        "keys": _bytes(runner.attention.key_cache, kv_bytes, runtime),
        "values": _bytes(runner.attention.value_cache, kv_bytes, runtime),
        "index_keys": _bytes(runner.index.raw_keys, n * config.indexer_key_length * 4, runtime),
        "pooled_keys": _bytes(runner.index.pooled_keys,
                              runner.index.pooled_count * config.indexer_key_length * 4, runtime),
        "carry": _bytes(runner.last_hidden, runner.last_hidden.nbytes, runtime),
    }


@pytest.mark.parametrize("chunks", [(1,), (17,), (257, 1), (2051,), (997, 1023, 31, 3)])
def test_cache_only_prime_matches_full_scalar_state_and_next_proposals(sidecar, chunks):
    resident, config, runtime = sidecar
    total = sum(chunks)
    rng = np.random.default_rng(4100 + total)
    tokens = rng.integers(1, config.vocab_size - 1, size=total).tolist()
    hidden = _bf16_values(rng.normal(0, 0.25, (total, config.residual_width)).astype(np.float32))
    runner = Qwen4ExpGGUFMTPDraftRunner(resident, target_config=config,
        max_sequence_length=max(16, total + 4), backend="hip_gfx1151", runtime=runtime)
    try:
        expected = []
        offset = 0
        runner.reset()
        for size in chunks:
            for index in range(offset, offset + size):
                previous = np.zeros(config.residual_width, dtype=np.float32) if index == 0 else hidden[index - 1]
                runner.forward(tokens[index], previous, run_head=False)
            offset += size
            expected.append(_state(runner))
        oracle = runner.propose_chain(start_token=42, target_hidden_seed=hidden[-1], draft_n_max=3)
        assert all(np.isfinite(row.logits).all() and np.isfinite(row.hidden_seed).all() for row in oracle)
        runner.reset()
        # Poison lazy pooled storage so a bogus restored cursor cannot pass by
        # reusing bytes left by the oracle's earlier sparse selection.
        runtime.memset(runner.index.pooled_keys.ptr, 0xFF, runner.index.pooled_keys.nbytes)
        offset = 0
        for size, state in zip(chunks, expected, strict=True):
            runner.prime_prompt_chunk(tokens[offset:offset + size], hidden[offset:offset + size],
                                      previous_hidden=None if offset == 0 else hidden[offset - 1])
            assert _state(runner) == state
            offset += size
        # Reuse a committed prefix after trimming; a one-token suffix still
        # executes the complete final row and refreshes hidden carry.
        if total > 1:
            runner.trim(total - 1)
            runner.prime_prompt_chunk(tokens[-1:], hidden[-1:], previous_hidden=hidden[-2])
            assert _state(runner) == expected[-1]
        actual = runner.propose_chain(start_token=42, target_hidden_seed=hidden[-1], draft_n_max=3)
        for left, right in zip(oracle, actual, strict=True):
            assert left.token_id == right.token_id
            np.testing.assert_array_equal(left.logits, right.logits)
            np.testing.assert_array_equal(left.hidden_seed, right.hidden_seed)
    finally:
        runner.close()
