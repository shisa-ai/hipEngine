"""Dense Llama decode: KV-cache contents, per-step logits, and the public path.

Three levels are asserted, because they fail differently:

* the prompt pass must leave the reference's roped K and projected V in the cache,
  **bit-identical**, which pins the cache layout a decode step indexes;
* each decode step's logits must track the BF16-rounded KV-cache reference, which
  pins the append position, the rotary table, and the GQA mapping;
* ``LLM.generate`` must produce the reference's greedy tokens, which is the only
  check that the product surface reaches this path at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._llama_tiny import (
    TINY_LLAMA_CONFIG,
    TINY_LLAMA_EOS,
    bf16_bits,
    tiny_llama_reference,
    tiny_llama_reference_decode,
    write_tiny_llama_checkpoint,
)
from tests._rocm_guard import hip_runtime_available

pytestmark = pytest.mark.skipif(not hip_runtime_available(), reason="HIP runtime unavailable")

_VOCAB = int(TINY_LLAMA_CONFIG["vocab_size"])
_LAYERS = int(TINY_LLAMA_CONFIG["num_hidden_layers"])
_KV_WIDTH = int(TINY_LLAMA_CONFIG["num_key_value_heads"]) * int(TINY_LLAMA_CONFIG["head_dim"])


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    directory = tmp_path_factory.mktemp("tiny-llama-decode")
    return write_tiny_llama_checkpoint(directory)


def _runner(checkpoint_path, max_context=64):
    from hipengine.loading.llama_safetensors import load_llama_spec, materialize_llama_weights
    from hipengine.loading.safetensors import load_weight_index
    from hipengine.runtime.llama_dense import LlamaDenseRunner

    index = load_weight_index(checkpoint_path)
    loaded = materialize_llama_weights(index, load_llama_spec(index))
    return LlamaDenseRunner(loaded, max_context=max_context), loaded


def _read_cache(cache, layer: int, rows: int, *, values: bool = False) -> np.ndarray:
    from hipengine.core.memory import copy_device_to_host, host_array_ptr

    buffer = cache.value(layer) if values else cache.key(layer)
    host = np.empty((rows, _KV_WIDTH), dtype=np.uint16)
    copy_device_to_host(
        host_array_ptr(host), buffer, rows * _KV_WIDTH * 2, runtime=cache.runtime
    )
    return host


def test_prefill_leaves_the_reference_kv_in_the_cache(tiny_checkpoint):
    path, _f32, bf16 = tiny_checkpoint
    runner, loaded = _runner(path)
    tokens = np.asarray((5, 11, 17, 23), dtype=np.int64)
    try:
        runner.prefill(tokens)
        cached_key = _read_cache(runner.cache, 0, tokens.size)
        cached_value = _read_cache(runner.cache, 0, tokens.size, values=True)
        context_len = runner.cache.context_len
    finally:
        runner.close()
        loaded.free()

    reference = tiny_llama_reference(bf16, TINY_LLAMA_CONFIG, tokens)
    assert context_len == tokens.size
    assert np.array_equal(
        cached_key, bf16_bits(reference.keys[0].reshape(tokens.size, _KV_WIDTH))
    )
    assert np.array_equal(
        cached_value, bf16_bits(reference.values[0].reshape(tokens.size, _KV_WIDTH))
    )


def test_decode_step_logits_track_the_kv_cache_reference(tiny_checkpoint):
    path, _f32, bf16 = tiny_checkpoint
    prompt = np.asarray((3, 9, 14, 27), dtype=np.int64)
    steps = 5
    reference_tokens, reference_logits = tiny_llama_reference_decode(
        bf16, TINY_LLAMA_CONFIG, prompt, max_new_tokens=steps
    )

    runner, loaded = _runner(path)
    try:
        runner.reset()
        first = runner.prefill(prompt)
        produced = [np.asarray(first[0], dtype=np.float32)]
        for step in range(steps - 1):
            produced.append(runner.decode_step(int(reference_tokens[step])))
        context_len = runner.cache.context_len
    finally:
        runner.close()
        loaded.free()

    produced = np.stack(produced)
    assert context_len == prompt.size + steps - 1
    assert produced.shape == reference_logits.shape == (steps, _VOCAB)
    peak = float(np.max(np.abs(reference_logits)))
    assert peak > 0.0
    for step in range(steps):
        assert float(np.max(np.abs(produced[step] - reference_logits[step]))) <= 1.0e-2 * peak
        assert int(np.argmax(produced[step])) == int(np.argmax(reference_logits[step]))


def test_greedy_tokens_match_the_reference(tiny_checkpoint):
    path, _f32, bf16 = tiny_checkpoint
    prompt = np.asarray((2, 4, 6, 8, 10, 12), dtype=np.int64)
    steps = 6
    reference_tokens, _logits = tiny_llama_reference_decode(
        bf16, TINY_LLAMA_CONFIG, prompt, max_new_tokens=steps, eos_token_id=TINY_LLAMA_EOS
    )

    runner, loaded = _runner(path)
    generated: list[int] = []
    try:
        runner.reset()
        # ``prefill`` returns the requested rows; ``decode_step`` returns one row.
        logits = np.reshape(runner.prefill(prompt), -1)
        for step in range(steps):
            token = int(np.argmax(logits))
            generated.append(token)
            if token == TINY_LLAMA_EOS or step + 1 >= steps:
                break
            logits = runner.decode_step(token)
    finally:
        runner.close()
        loaded.free()

    assert tuple(generated) == reference_tokens


def test_llm_generate_runs_the_dense_llama_path(tiny_checkpoint):
    from hipengine.llm import LLM, SamplingParams

    path, _f32, bf16 = tiny_checkpoint
    llm = LLM(str(path), max_sequence_length=64)
    try:
        prompt_ids = llm.tokenize("hello world")
        outputs = llm.generate_detailed("hello world", SamplingParams(max_tokens=4))
        # Every registered generator is driven through the submit/poll bridge, so
        # the route check looks at the inner generator the bridge holds.
        driver = llm._get_text_generator()
        generator_type = type(getattr(driver, "_inner", driver)).__name__
        resolved_quant = llm.resolved_quant
        resolved_backend = llm.resolved_backend
    finally:
        close = getattr(llm, "close", None)
        if callable(close):
            close()

    reference_tokens, _logits = tiny_llama_reference_decode(
        bf16,
        TINY_LLAMA_CONFIG,
        np.asarray(prompt_ids, dtype=np.int64),
        max_new_tokens=4,
        eos_token_id=TINY_LLAMA_EOS,
    )

    assert generator_type == "LlamaDenseGenerator"
    assert resolved_quant == "bf16"
    assert resolved_backend in {"hip_gfx1100", "hip_gfx1151"}
    assert prompt_ids == (1, 2)
    assert len(outputs) == 1
    output = outputs[0]
    assert output.generated_token_ids == reference_tokens
    assert output.prompt_tokens == len(prompt_ids)
    assert output.finish_details is not None
    assert isinstance(output.text, str)
