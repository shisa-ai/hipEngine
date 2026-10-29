"""Tiny dense Llama checkpoint fixture and BF16-rounded NumPy reference.

The reference mirrors the execution profile the hipEngine runner uses, so a
comparison isolates kernel defects from representation differences: activations
are rounded to BF16 at every op boundary, projections accumulate in FP32, and
attention normalizes in FP64 over the BF16 Q/K the kernel consumes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

TINY_LLAMA_CONFIG: dict[str, object] = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "intermediate_size": 512,
    "vocab_size": 512,
    "rms_norm_eps": 1.0e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "mlp_bias": False,
    "hidden_act": "silu",
    "eos_token_id": 7,
    "bos_token_id": 6,
}

#: Word-level fixture vocabulary. The chat markers are ordinary entries, so the
#: adapter's ``token_to_id`` check exercises the same lookup a real checkpoint's
#: ``tokenizer.json`` answers.
TINY_LLAMA_VOCAB: dict[str, int] = {
    "[UNK]": 0,
    "hello": 1,
    "world": 2,
    "user": 3,
    "assistant": 4,
    "system": 5,
    "<|im_start|>": 6,
    "<|im_end|>": 7,
}
TINY_LLAMA_BOS = TINY_LLAMA_VOCAB["<|im_start|>"]
TINY_LLAMA_EOS = TINY_LLAMA_VOCAB["<|im_end|>"]


def bf16_round(values: np.ndarray) -> np.ndarray:
    """Round FP32 values to BF16 precision, returned as FP32."""

    bits = np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> np.uint32(16) << np.uint32(16)).astype(np.uint32).view(np.float32)


def bf16_bits(values: np.ndarray) -> np.ndarray:
    bits = np.ascontiguousarray(np.asarray(values, dtype=np.float32)).view(np.uint32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    return np.ascontiguousarray(((bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)).astype(np.uint16))


def fp8_e4m3fn_decode(codes: np.ndarray) -> np.ndarray:
    """Decode E4M3FN codes to FP32 using the reference table."""

    table = np.arange(256, dtype=np.uint16)
    sign = np.where((table >> 7) & 1, -1.0, 1.0).astype(np.float32)
    exponent = ((table >> 3) & 0xF).astype(np.int32)
    mantissa = (table & 0x7).astype(np.float32)
    magnitude = np.where(
        exponent == 0,
        mantissa / 8.0 * np.float32(2.0**-6),
        (1.0 + mantissa / 8.0) * np.power(2.0, (exponent - 7).astype(np.float32)),
    )
    values = sign * magnitude.astype(np.float32)
    values[(exponent == 0xF) & (mantissa == 7)] = np.nan
    return values[np.asarray(codes, dtype=np.uint8)]


def tiny_llama_weights(config: dict[str, object], *, seed: int = 7) -> dict[str, np.ndarray]:
    """Return FP32 weights for every tensor the dense Llama decoder requires."""

    rng = np.random.default_rng(seed)
    hidden = int(config["hidden_size"])
    layers = int(config["num_hidden_layers"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    inter = int(config["intermediate_size"])
    vocab = int(config["vocab_size"])
    q_width = heads * head_dim
    kv_width = kv_heads * head_dim

    def matrix(rows: int, cols: int, scale: float = 0.08) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * scale).astype(np.float32)

    weights: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": matrix(vocab, hidden, 0.5),
        "model.norm.weight": np.ones(hidden, dtype=np.float32),
        "lm_head.weight": matrix(vocab, hidden, 0.08),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        weights[f"{prefix}.input_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.post_attention_layernorm.weight"] = np.ones(hidden, dtype=np.float32)
        weights[f"{prefix}.self_attn.q_proj.weight"] = matrix(q_width, hidden)
        weights[f"{prefix}.self_attn.k_proj.weight"] = matrix(kv_width, hidden)
        weights[f"{prefix}.self_attn.v_proj.weight"] = matrix(kv_width, hidden)
        weights[f"{prefix}.self_attn.o_proj.weight"] = matrix(hidden, q_width)
        weights[f"{prefix}.mlp.gate_proj.weight"] = matrix(inter, hidden)
        weights[f"{prefix}.mlp.up_proj.weight"] = matrix(inter, hidden)
        weights[f"{prefix}.mlp.down_proj.weight"] = matrix(hidden, inter)
    return weights


def write_tiny_llama_tokenizer(directory: Path) -> Path:
    """Write a word-level ``tokenizer.json`` plus a ChatML ``tokenizer_config.json``.

    Splitting the chat markers out of a word is what keeps a rendered ChatML
    prompt tokenizing into the markers the adapter validated rather than into
    ``[UNK]``.
    """

    import json

    from tokenizers import Regex, Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel

    directory = Path(directory)
    # Every decoder row gets a tokenizer entry, so decoding any token the model can
    # emit is well defined rather than an out-of-vocabulary error.
    vocab = dict(TINY_LLAMA_VOCAB)
    for token_id in range(len(vocab), int(TINY_LLAMA_CONFIG["vocab_size"])):
        vocab[f"<t{token_id}>"] = token_id
    encoder = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    encoder.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(r"<\|im_(?:start|end)\|>"), behavior="isolated"),
            pre_tokenizers.Split(Regex(r"\s+"), behavior="removed"),
        ]
    )
    path = directory / "tokenizer.json"
    encoder.save(str(path))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "chat_template": (
                    "{% for message in messages %}"
                    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}<|im_start|>assistant\n"
                    "{% if enable_thinking %}<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}"
                    "{% endif %}"
                ),
                "eos_token": "<|im_end|>",
                "bos_token": "<|im_start|>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def write_tiny_llama_checkpoint(
    directory: Path,
    *,
    config: dict[str, object] | None = None,
    seed: int = 7,
    with_tokenizer: bool = True,
) -> tuple[Path, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Write a BF16 tiny Llama checkpoint and return ``(path, f32, bf16)`` weights."""

    import json

    resolved = dict(TINY_LLAMA_CONFIG if config is None else config)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(resolved, indent=2), encoding="utf-8")
    f32 = tiny_llama_weights(resolved, seed=seed)
    rounded = {name: bf16_round(values) for name, values in f32.items()}
    save_file(rounded, str(directory / "model.safetensors"))
    if with_tokenizer:
        write_tiny_llama_tokenizer(directory)
    return directory, f32, rounded


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    variance = np.mean(np.asarray(x, dtype=np.float32) ** 2, axis=-1, keepdims=True)
    normed = np.asarray(x, dtype=np.float32) / np.sqrt(variance + np.float32(eps))
    return bf16_round(normed * np.asarray(weight, dtype=np.float32))


def _linear(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return bf16_round(np.asarray(x, dtype=np.float32) @ np.asarray(weight, dtype=np.float32).T)


def _rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """Rotate-half RoPE on ``(tokens, heads, head_dim)`` BF16-precision values."""

    values = np.asarray(x, dtype=np.float32)
    head_dim = values.shape[-1]
    half = head_dim // 2
    inv_freq = 1.0 / np.power(theta, np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
    angles = positions.astype(np.float64)[:, None] * inv_freq[None, :]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    first = values[..., :half]
    second = values[..., half:]
    out = np.empty_like(values)
    out[..., :half] = first * cos[:, None, :] - second * sin[:, None, :]
    out[..., half:] = second * cos[:, None, :] + first * sin[:, None, :]
    return bf16_round(out)


def tiny_llama_reference_logits(
    weights: dict[str, np.ndarray],
    config: dict[str, object],
    token_ids: np.ndarray,
    *,
    row: int = -1,
) -> np.ndarray:
    """Return FP32 logits for one prompt row, mirroring the runner's numerics."""

    return tiny_llama_reference(weights, config, token_ids).logits(row)


class LlamaReferenceTrace:
    """Layer taps, cached K/V, and logits from the BF16-rounded reference forward."""

    def __init__(self, taps, hidden_layers, final_norm, lm_head, keys=None, values=None):
        self.taps = taps
        self.hidden_layers = hidden_layers
        self._final_norm = final_norm
        self._lm_head = lm_head
        self.keys = list(keys or ())
        self.values = list(values or ())

    def logits(self, row: int = -1) -> np.ndarray:
        return np.asarray(self._final_norm[row], dtype=np.float32) @ np.asarray(
            self._lm_head, dtype=np.float32
        ).T


def tiny_llama_reference(
    weights: dict[str, np.ndarray],
    config: dict[str, object],
    token_ids: np.ndarray,
) -> LlamaReferenceTrace:
    """Run the BF16-rounded reference and retain every DMS tap."""

    hidden_size = int(config["hidden_size"])
    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    inter = int(config["intermediate_size"])
    eps = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    scale = float(head_dim) ** -0.5
    tokens = np.asarray(token_ids, dtype=np.int64)
    rows = int(tokens.size)
    positions = np.arange(rows, dtype=np.int64)

    hidden = bf16_round(np.asarray(weights["model.embed_tokens.weight"])[tokens])
    group = heads // kv_heads
    causal = np.tril(np.ones((rows, rows), dtype=bool))
    taps: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cached_keys: list[np.ndarray] = []
    cached_values: list[np.ndarray] = []

    for layer in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        normed = _rmsnorm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)
        query = _linear(normed, weights[f"{prefix}.self_attn.q_proj.weight"]).reshape(
            rows, heads, head_dim
        )
        key = _linear(normed, weights[f"{prefix}.self_attn.k_proj.weight"]).reshape(
            rows, kv_heads, head_dim
        )
        value = _linear(normed, weights[f"{prefix}.self_attn.v_proj.weight"]).reshape(
            rows, kv_heads, head_dim
        )
        query = _rope(query, positions, theta)
        key = _rope(key, positions, theta)
        taps[layer] = (normed, query, key)
        cached_keys.append(key)
        cached_values.append(value)

        attended = np.zeros((rows, heads, head_dim), dtype=np.float32)
        for head in range(heads):
            kv_head = head // group
            logits = (
                query[:, head, :].astype(np.float64)
                @ key[:, kv_head, :].astype(np.float64).T
                * scale
            )
            logits = np.where(causal, logits, -np.inf)
            logits -= logits.max(axis=1, keepdims=True)
            probabilities = np.exp(logits)
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            attended[:, head, :] = (probabilities @ value[:, kv_head, :].astype(np.float64)).astype(
                np.float32
            )
        attended = bf16_round(attended.reshape(rows, heads * head_dim))
        hidden = bf16_round(hidden + _linear(attended, weights[f"{prefix}.self_attn.o_proj.weight"]))

        normed = _rmsnorm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
        gate = _linear(normed, weights[f"{prefix}.mlp.gate_proj.weight"])
        up = _linear(normed, weights[f"{prefix}.mlp.up_proj.weight"])
        activation = bf16_round((gate / (1.0 + np.exp(-gate))) * up)
        hidden = bf16_round(hidden + _linear(activation, weights[f"{prefix}.mlp.down_proj.weight"]))

    final = _rmsnorm(hidden, weights["model.norm.weight"], eps)
    return LlamaReferenceTrace(
        taps,
        hidden,
        final,
        weights["lm_head.weight"],
        cached_keys,
        cached_values,
    )


def _reference_decode_row(
    weights: dict[str, np.ndarray],
    config: dict[str, object],
    token: int,
    position: int,
    keys: list[np.ndarray],
    values: list[np.ndarray],
) -> np.ndarray:
    """One BF16-rounded decode row; appends this position's K/V and returns hidden."""

    heads = int(config["num_attention_heads"])
    kv_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    eps = float(config["rms_norm_eps"])
    theta = float(config["rope_theta"])
    scale = float(head_dim) ** -0.5
    group = heads // kv_heads
    position_array = np.array([int(position)], dtype=np.int64)

    hidden = bf16_round(np.asarray(weights["model.embed_tokens.weight"])[int(token)])[None, :]

    for layer in range(int(config["num_hidden_layers"])):
        prefix = f"model.layers.{layer}"
        normed = _rmsnorm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)
        query = _linear(normed, weights[f"{prefix}.self_attn.q_proj.weight"]).reshape(
            1, heads, head_dim
        )
        key = _linear(normed, weights[f"{prefix}.self_attn.k_proj.weight"]).reshape(
            1, kv_heads, head_dim
        )
        value = _linear(normed, weights[f"{prefix}.self_attn.v_proj.weight"]).reshape(
            1, kv_heads, head_dim
        )
        query = _rope(query, position_array, theta)
        key = _rope(key, position_array, theta)
        keys[layer] = np.concatenate([keys[layer], key], axis=0)
        values[layer] = np.concatenate([values[layer], value], axis=0)

        attended = np.zeros((heads, head_dim), dtype=np.float32)
        for head in range(heads):
            kv_head = head // group
            scores = (
                query[0, head, :].astype(np.float64)
                @ keys[layer][:, kv_head, :].astype(np.float64).T
                * scale
            )
            scores -= scores.max()
            probabilities = np.exp(scores)
            probabilities /= probabilities.sum()
            attended[head] = (
                probabilities @ values[layer][:, kv_head, :].astype(np.float64)
            ).astype(np.float32)
        attended = bf16_round(attended.reshape(1, heads * head_dim))
        hidden = bf16_round(
            hidden + _linear(attended, weights[f"{prefix}.self_attn.o_proj.weight"])
        )

        normed = _rmsnorm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
        gate = _linear(normed, weights[f"{prefix}.mlp.gate_proj.weight"])
        up = _linear(normed, weights[f"{prefix}.mlp.up_proj.weight"])
        activation = bf16_round((gate / (1.0 + np.exp(-gate))) * up)
        hidden = bf16_round(hidden + _linear(activation, weights[f"{prefix}.mlp.down_proj.weight"]))

    return hidden[0]


def tiny_llama_reference_decode(
    weights: dict[str, np.ndarray],
    config: dict[str, object],
    token_ids: np.ndarray,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[tuple[int, ...], np.ndarray]:
    """Greedy decode against a KV cache, mirroring the runner's rounding.

    Returns the generated ids and the FP32 logits each id was chosen from, so a
    caller can compare distributions and not only the argmax.
    """

    trace = tiny_llama_reference(weights, config, token_ids)
    keys = trace.keys
    values = trace.values
    hidden = trace.hidden_layers
    eps = float(config["rms_norm_eps"])
    generated: list[int] = []
    logits_rows: list[np.ndarray] = []
    prompt_rows = int(np.asarray(token_ids).size)

    for step in range(int(max_new_tokens)):
        final = _rmsnorm(hidden, weights["model.norm.weight"], eps)
        logits = np.asarray(final[-1], dtype=np.float32) @ np.asarray(
            weights["lm_head.weight"], dtype=np.float32
        ).T
        token = int(np.argmax(logits))
        generated.append(token)
        logits_rows.append(logits)
        if eos_token_id is not None and token == int(eos_token_id):
            break
        if step + 1 >= int(max_new_tokens):
            break
        hidden = _reference_decode_row(
            weights, config, token, prompt_rows + step, keys, values
        )[None, :]

    return tuple(generated), np.stack(logits_rows)


__all__ = [
    "TINY_LLAMA_BOS",
    "TINY_LLAMA_CONFIG",
    "TINY_LLAMA_EOS",
    "TINY_LLAMA_VOCAB",
    "bf16_bits",
    "bf16_round",
    "fp8_e4m3fn_decode",
    "LlamaReferenceTrace",
    "tiny_llama_reference",
    "tiny_llama_reference_decode",
    "tiny_llama_reference_logits",
    "tiny_llama_weights",
    "write_tiny_llama_checkpoint",
    "write_tiny_llama_tokenizer",
]
