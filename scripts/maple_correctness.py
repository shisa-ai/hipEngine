#!/usr/bin/env python3
"""Teacher-forced Maple correctness gates for the packed hipEngine runtime.

Two independent torch workers are available: ``packed`` executes the published
Maple formulas over the official 5.31 GB MLX checkpoint, while ``hf`` loads the
40.4 GB dense source checkpoint with Transformers ``trust_remote_code``. The
remote model hard-imports FlashAttention, so the HF worker supplies only that API
through a deterministic pure-torch shim; its code remains unmodified. For a
same-weight implementation gate, ``--hf-match-packed-affine4`` replaces only the
dense embedding/head parameters with official packed-checkpoint dequantization;
without it, the run is a raw quantization-quality diagnostic. Torch is never
imported by hipEngine's runtime worker.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
import types
from contextlib import ExitStack, contextmanager, nullcontext
from importlib.machinery import ModuleSpec
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_HF_MODEL = "deepgrove/maple-preview"
_HF_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
_HF_TRANSFORMERS_VERSION = "4.57.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="deepgrove/maple-preview-2bit-mlx")
    parser.add_argument("--backend", default="hip_gfx1151")
    parser.add_argument("--token-ids", default="9707")
    parser.add_argument("--oracle", choices=("packed", "hf"), default="packed")
    parser.add_argument("--hf-model", default=_HF_MODEL)
    parser.add_argument("--hf-revision", default=_HF_REVISION)
    parser.add_argument("--hf-device", default="cuda:0")
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--hf-local-files-only", action="store_true")
    parser.add_argument("--hf-offload-experts", action="store_true")
    parser.add_argument("--hf-offload-dir", type=Path)
    parser.add_argument("--hf-match-packed-affine4", action="store_true")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--keep-logits", type=Path)
    parser.add_argument("--worker", choices=("hipengine", "torch", "hf"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def token_ids(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not result or any(token < 0 for token in result):
        raise ValueError("--token-ids must contain non-negative comma-separated IDs")
    return result


def hipengine_worker(model: str, backend: str, tokens: tuple[int, ...], output: Path) -> None:
    from hipengine.kernels.backends import backend_package_capability
    from hipengine.loading.maple import load_maple_checkpoint
    from hipengine.runtime.maple import MapleRunner

    runner_type_factory = backend_package_capability(
        backend,
        "maple_runner_type",
        None,
    )
    runner_type = (
        MapleRunner
        if runner_type_factory is None
        else runner_type_factory()
    )
    checkpoint = load_maple_checkpoint(model)
    runner = runner_type.load(
        checkpoint,
        backend=backend,
        max_context=max(64, len(tokens) + 1),
    )
    rows: list[np.ndarray] = []
    sampled_ids: list[int] = []
    hidden_rows: list[np.ndarray] = []
    elapsed: list[float] = []
    try:
        for token in tokens:
            step = runner.step(token, capture_hidden=True)
            rows.append(runner.copy_logits())
            sampled_ids.append(step.token_id)
            hidden_rows.append(np.stack(runner.last_hidden_states))
            elapsed.append(step.elapsed_ms)
    finally:
        runner.close()
    logits = np.stack(rows)
    top_ids = np.argmax(logits, axis=-1).astype(np.int64)
    sampled = np.asarray(sampled_ids, dtype=np.int64)
    if not np.array_equal(sampled, top_ids):
        raise RuntimeError(
            "Maple device greedy sampler disagrees with exact FP32 logit argmax"
        )
    np.savez(
        output,
        logits=logits,
        top_ids=sampled,
        hidden=np.stack(hidden_rows),
        elapsed_ms=np.asarray(elapsed, dtype=np.float64),
        greedy_argmax_exact=np.asarray(True),
    )


def torch_flash_attention(
    query,
    key,
    value,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    deterministic: bool | None = None,
    return_attn_probs: bool = False,
    **_kwargs,
):
    """Pure-torch FlashAttention-compatible GQA used only by the HF oracle."""

    del deterministic
    import torch

    if float(dropout_p) != 0.0:
        raise ValueError("Maple correctness oracle does not support attention dropout")
    if return_attn_probs:
        raise ValueError("Maple correctness oracle does not return attention probabilities")
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FlashAttention shim expects [batch, sequence, heads, head_dim]")
    if key.shape != value.shape:
        raise ValueError("FlashAttention shim requires matching key/value shapes")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("FlashAttention shim query/key batch and head_dim must match")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("FlashAttention shim requires integral grouped-query heads")
    if key.shape[1] < query.shape[1]:
        raise ValueError("FlashAttention shim requires key sequence >= query sequence")

    groups = query.shape[2] // key.shape[2]
    expanded_key = key.repeat_interleave(groups, dim=2)
    expanded_value = value.repeat_interleave(groups, dim=2)
    scale = float(softmax_scale) if softmax_scale is not None else query.shape[-1] ** -0.5
    scores = torch.einsum(
        "bqhd,bkhd->bhqk",
        query.float(),
        expanded_key.float(),
    ) * scale
    if softcap is not None:
        cap = float(softcap)
        if cap <= 0.0:
            raise ValueError("FlashAttention softcap must be positive")
        scores = torch.tanh(scores / cap) * cap

    query_length = query.shape[1]
    key_length = key.shape[1]
    query_positions = torch.arange(query_length, device=query.device) + (
        key_length - query_length
    )
    key_positions = torch.arange(key_length, device=query.device)
    live = torch.ones(
        (query_length, key_length),
        dtype=torch.bool,
        device=query.device,
    )
    if causal:
        live &= key_positions[None, :] <= query_positions[:, None]
    left_window, right_window = (int(window_size[0]), int(window_size[1]))
    if left_window >= 0:
        live &= key_positions[None, :] >= query_positions[:, None] - left_window
    if right_window >= 0:
        live &= key_positions[None, :] <= query_positions[:, None] + right_window
    if not bool(torch.all(live.any(dim=-1))):
        raise ValueError("FlashAttention shim produced a fully masked query row")

    scores = scores.masked_fill(~live[None, None, :, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    output = torch.einsum(
        "bhqk,bkhd->bqhd",
        probabilities,
        expanded_value.float(),
    )
    return output.to(query.dtype)


def torch_flash_attention_varlen(
    query,
    key,
    value,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float | None = None,
    deterministic: bool | None = None,
    **kwargs,
):
    """Packed-sequence companion for the remote model's prefill helper."""

    del max_seqlen_q, max_seqlen_k
    import torch

    if cu_seqlens_q.numel() != cu_seqlens_k.numel():
        raise ValueError("FlashAttention shim requires matching sequence batches")
    outputs = []
    for batch in range(int(cu_seqlens_q.numel()) - 1):
        q_start = int(cu_seqlens_q[batch])
        q_stop = int(cu_seqlens_q[batch + 1])
        k_start = int(cu_seqlens_k[batch])
        k_stop = int(cu_seqlens_k[batch + 1])
        output = torch_flash_attention(
            query[q_start:q_stop].unsqueeze(0),
            key[k_start:k_stop].unsqueeze(0),
            value[k_start:k_stop].unsqueeze(0),
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            deterministic=deterministic,
            **kwargs,
        )
        outputs.append(output.squeeze(0))
    return query.new_empty((0, *query.shape[1:])) if not outputs else torch.cat(outputs)


def install_hf_flash_attention_shim() -> None:
    """Publish the minimal FlashAttention API imported by pinned Maple code."""

    import torch

    flash_attn = types.ModuleType("flash_attn")
    flash_attn.__path__ = []
    flash_attn.__spec__ = ModuleSpec("flash_attn", loader=None, is_package=True)
    flash_attn.flash_attn_func = torch_flash_attention
    flash_attn.flash_attn_varlen_func = torch_flash_attention_varlen

    def index_first_axis(values, indices):
        return values[indices]

    def pad_input(values, indices, batch_size, sequence_length):
        output = values.new_zeros((int(batch_size) * int(sequence_length), *values.shape[1:]))
        output[indices] = values
        return output.reshape(int(batch_size), int(sequence_length), *values.shape[1:])

    def unpad_input(values, attention_mask):
        flat_mask = attention_mask.reshape(-1).to(torch.bool)
        indices = torch.nonzero(flat_mask, as_tuple=False).flatten()
        unpadded = values.reshape(-1, *values.shape[2:])[indices]
        lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        cumulative = torch.nn.functional.pad(torch.cumsum(lengths, dim=0), (1, 0))
        return unpadded, indices, cumulative, int(lengths.max())

    bert_padding = types.ModuleType("flash_attn.bert_padding")
    bert_padding.__spec__ = ModuleSpec("flash_attn.bert_padding", loader=None)
    bert_padding.index_first_axis = index_first_axis
    bert_padding.pad_input = pad_input
    bert_padding.unpad_input = unpad_input
    flash_attn.bert_padding = bert_padding

    interface = types.ModuleType("flash_attn_interface")
    interface.__spec__ = ModuleSpec("flash_attn_interface", loader=None)
    interface.flash_attn_func = torch_flash_attention
    interface.flash_attn_varlen_func = torch_flash_attention_varlen

    sys.modules["flash_attn"] = flash_attn
    sys.modules["flash_attn.bert_padding"] = bert_padding
    sys.modules["flash_attn_interface"] = interface


def maple_hf_expert_device_map(
    *, layers: int, experts: int, device: str
) -> dict[str, str]:
    """Keep dense control/attention on device and lazily mmap routed experts."""

    device_map = {
        "model.word_embeddings": device,
        "model.norm": device,
        "model.rotary_emb": device,
        "lm_head": device,
    }
    for layer in range(int(layers)):
        prefix = f"model.layers.{layer}"
        device_map[f"{prefix}.input_layernorm"] = device
        device_map[f"{prefix}.post_attention_layernorm"] = device
        device_map[f"{prefix}.self_attn"] = device
        device_map[f"{prefix}.mlp.gate"] = device
        for expert in range(int(experts)):
            device_map[f"{prefix}.mlp.experts.{expert}"] = "disk"
    return device_map


def match_hf_affine4_from_packed(
    oracle,
    packed_model: str,
    *,
    device: str,
    chunk_rows: int = 512,
) -> None:
    """Replace dense HF embeddings/head with exact official affine4 dequant."""

    import torch
    from safetensors import safe_open

    from hipengine.loading.maple import load_maple_checkpoint

    checkpoint = load_maple_checkpoint(packed_model)
    spec = checkpoint.spec
    if (oracle.config.vocab_size, oracle.config.hidden_size) != (
        spec.vocab_size,
        spec.hidden_size,
    ):
        raise ValueError("dense HF and packed Maple dimensions do not match")

    with ExitStack() as stack:
        handles = {
            shard: stack.enter_context(
                safe_open(str(shard), framework="pt", device="cpu")
            )
            for shard in checkpoint.index.shards
        }

        def tensor_slice(name: str, start: int, stop: int):
            info = checkpoint.index.tensors[name]
            return handles[info.shard_path].get_slice(name)[start:stop]

        def dequantize(prefix: str, group_size: int):
            rows, features = spec.vocab_size, spec.hidden_size
            output = torch.empty(
                (rows, features), dtype=torch.bfloat16, device=device
            )
            shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=device)
            for start in range(0, rows, int(chunk_rows)):
                stop = min(rows, start + int(chunk_rows))
                packed = tensor_slice(f"{prefix}.weight", start, stop).to(
                    device=device, dtype=torch.int64
                )
                scales = tensor_slice(f"{prefix}.scales", start, stop).to(device)
                biases = tensor_slice(f"{prefix}.biases", start, stop).to(device)
                codes = ((packed.unsqueeze(-1) >> shifts) & 0xF).reshape(
                    stop - start, features
                )
                output[start:stop].copy_(
                    codes.to(torch.bfloat16)
                    * scales.repeat_interleave(group_size, dim=-1)
                    + biases.repeat_interleave(group_size, dim=-1)
                )
            return output

        oracle.model.word_embeddings.weight = torch.nn.Parameter(
            dequantize("model.word_embeddings", spec.embedding_group_size),
            requires_grad=False,
        )
        oracle.lm_head.weight = torch.nn.Parameter(
            dequantize("lm_head", spec.lm_head_group_size),
            requires_grad=False,
        )
        torch.cuda.synchronize()


@contextmanager
def hf_maple_expert_preload_dispatch():
    """Teach Accelerate that MapleMLP reads child weights through F.linear."""

    from transformers import modeling_utils

    original = modeling_utils.dispatch_model

    def dispatch_with_maple_preload(*args, **kwargs):
        preload = list(kwargs.pop("preload_module_classes", ()) or ())
        if "MapleMLP" not in preload:
            preload.append("MapleMLP")
        return original(*args, preload_module_classes=preload, **kwargs)

    modeling_utils.dispatch_model = dispatch_with_maple_preload
    try:
        yield
    finally:
        modeling_utils.dispatch_model = original


class TorchMapleOracle:
    """Independent packed-weight torch implementation of official HF Maple math."""

    def __init__(self, model: str, *, device: str = "cuda") -> None:
        import torch
        from safetensors import safe_open

        from hipengine.loading.maple import load_maple_checkpoint

        self.torch = torch
        self.checkpoint = load_maple_checkpoint(model)
        self.spec = self.checkpoint.spec
        self.device = torch.device(device)
        self.stack = ExitStack()
        self.handles = {
            shard: self.stack.enter_context(
                safe_open(str(shard), framework="pt", device="cpu")
            )
            for shard in self.checkpoint.index.shards
        }
        self.k_cache: list[list[object]] = [
            [] for _ in range(self.spec.num_hidden_layers)
        ]
        self.v_cache: list[list[object]] = [
            [] for _ in range(self.spec.num_hidden_layers)
        ]
        self.position = 0
        self.last_hidden_states: tuple[np.ndarray, ...] = ()

    def close(self) -> None:
        self.stack.close()
        self.torch.cuda.empty_cache()

    def tensor(self, name: str):
        info = self.checkpoint.index.tensors[name]
        return self.handles[info.shard_path].get_tensor(name).to(self.device)

    def slice(self, name: str, start: int, stop: int):
        info = self.checkpoint.index.tensors[name]
        value = self.handles[info.shard_path].get_slice(name)[start:stop]
        return value.to(self.device)

    def selected(self, name: str, experts: list[int]):
        return self.torch.cat(
            [self.slice(name, expert, expert + 1) for expert in experts], dim=0
        )

    def dequant_ternary(self, packed, alpha):
        torch = self.torch
        shifts = torch.arange(0, 32, 2, dtype=torch.int64, device=self.device)
        codes = (
            (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0x3
        ).reshape(*packed.shape[:-1], packed.shape[-1] * 16)
        return (codes.to(torch.bfloat16) - 1) * alpha.unsqueeze(-1)

    def dequant_affine4(self, packed, scales, biases):
        torch = self.torch
        shifts = torch.arange(0, 32, 4, dtype=torch.int64, device=self.device)
        codes = (
            (packed.to(torch.int64).unsqueeze(-1) >> shifts) & 0xF
        ).reshape(*packed.shape[:-1], packed.shape[-1] * 8)
        expanded_scales = scales.repeat_interleave(64, dim=-1)
        expanded_biases = biases.repeat_interleave(64, dim=-1)
        return codes.to(torch.bfloat16) * expanded_scales + expanded_biases

    def ternary_linear(self, x, prefix: str):
        torch = self.torch
        weight = self.dequant_ternary(
            self.tensor(f"{prefix}.weight"),
            self.tensor(f"{prefix}.row_alpha"),
        )
        return torch.nn.functional.linear(x, weight)

    def selected_ternary_linear(self, x, prefix: str, experts: list[int]):
        torch = self.torch
        weight = self.dequant_ternary(
            self.selected(f"{prefix}.weight", experts),
            self.selected(f"{prefix}.row_alpha", experts),
        )
        return torch.einsum("rnk,rk->rn", weight, x)

    def rmsnorm(self, x, weight):
        torch = self.torch
        x32 = x.float()
        inv = torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + self.spec.rms_norm_eps)
        return (x32 * inv * weight.float()).to(torch.bfloat16)

    def rope(self, x, position: int):
        torch = self.torch
        rotary_dim = self.spec.rotary_dim
        half = rotary_dim // 2
        frequency = torch.arange(half, dtype=torch.float32, device=self.device)
        theta = float(position) * self.spec.rope_theta ** (-frequency / half)
        cosine, sine = theta.cos(), theta.sin()
        first = x[..., :half].float()
        second = x[..., half:rotary_dim].float()
        rotated = torch.cat(
            (first * cosine - second * sine, second * cosine + first * sine), dim=-1
        )
        return torch.cat((rotated, x[..., rotary_dim:].float()), dim=-1).to(
            torch.bfloat16
        )

    def attention(self, q, keys, values):
        torch = self.torch
        group = self.spec.num_attention_heads // self.spec.num_key_value_heads
        k = torch.stack(keys).repeat_interleave(group, dim=1)
        v = torch.stack(values).repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", q.float(), k.float()) * (
            self.spec.head_dim**-0.5
        )
        probs = torch.softmax(scores, dim=-1)
        return torch.einsum("ht,thd->hd", probs, v.float()).to(torch.bfloat16)

    def embed(self, token_id: int):
        weight = self.slice("model.word_embeddings.weight", token_id, token_id + 1)
        scales = self.slice("model.word_embeddings.scales", token_id, token_id + 1)
        biases = self.slice("model.word_embeddings.biases", token_id, token_id + 1)
        return self.dequant_affine4(weight, scales, biases)[0].to(
            self.torch.bfloat16
        )

    def step(self, token_id: int) -> np.ndarray:
        torch = self.torch
        spec = self.spec
        position = self.position
        h = self.embed(token_id)
        captured = [h.float().cpu().numpy()]
        with torch.inference_mode():
            for layer in range(spec.num_hidden_layers):
                prefix = f"model.layers.{layer}"
                residual = h
                hn = self.rmsnorm(
                    h, self.tensor(f"{prefix}.input_layernorm.weight")
                )
                q = self.ternary_linear(hn, f"{prefix}.self_attn.q_proj").reshape(
                    spec.num_attention_heads, spec.head_dim
                )
                k = self.ternary_linear(hn, f"{prefix}.self_attn.k_proj").reshape(
                    spec.num_key_value_heads, spec.head_dim
                )
                v = self.ternary_linear(hn, f"{prefix}.self_attn.v_proj").reshape(
                    spec.num_key_value_heads, spec.head_dim
                )
                q = self.rmsnorm(q, self.tensor(f"{prefix}.self_attn.q_norm.weight"))
                k = self.rmsnorm(k, self.tensor(f"{prefix}.self_attn.k_norm.weight"))
                if spec.uses_rope(layer):
                    q = self.rope(q, position)
                    k = self.rope(k, position)
                self.k_cache[layer].append(k)
                self.v_cache[layer].append(v)
                keys = self.k_cache[layer]
                values = self.v_cache[layer]
                if spec.attention_kind(layer) == "sliding_attention":
                    keys = keys[-spec.sliding_window :]
                    values = values[-spec.sliding_window :]
                attention = self.attention(q, keys, values).reshape(-1)
                o = self.ternary_linear(
                    attention, f"{prefix}.self_attn.o_proj"
                )
                h = (residual + o).to(torch.bfloat16)

                residual = h
                hn = self.rmsnorm(
                    h,
                    self.tensor(f"{prefix}.post_attention_layernorm.weight"),
                )
                router = self.tensor(f"{prefix}.mlp.gate.weight")
                logits = torch.nn.functional.linear(hn.float(), router.float())
                probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32)
                route_weights, route_ids = torch.topk(
                    probabilities, spec.num_experts_per_tok
                )
                route_weights /= route_weights.sum() + 1.0e-20
                experts = [int(expert) for expert in route_ids.cpu().tolist()]
                switch = f"{prefix}.mlp.switch_mlp"
                repeated = hn.unsqueeze(0).expand(spec.num_experts_per_tok, -1)
                gate = self.selected_ternary_linear(
                    repeated, f"{switch}.gate_proj", experts
                )
                up = self.selected_ternary_linear(
                    repeated, f"{switch}.up_proj", experts
                )
                gate = torch.minimum(gate, torch.tensor(7.0, device=self.device))
                up = torch.clamp(up, -7.0, 7.0)
                intermediate = torch.nn.functional.silu(gate) * up
                down = self.selected_ternary_linear(
                    intermediate, f"{switch}.down_proj", experts
                )
                combined = (
                    down.float() * route_weights[:, None]
                ).sum(dim=0).to(torch.bfloat16)
                h = (residual + combined).to(torch.bfloat16)
                captured.append(h.float().cpu().numpy())

            h = self.rmsnorm(h, self.tensor("model.norm.weight"))
            captured.append(h.float().cpu().numpy())
            head = self.dequant_affine4(
                self.tensor("lm_head.weight"),
                self.tensor("lm_head.scales"),
                self.tensor("lm_head.biases"),
            )
            logits = torch.nn.functional.linear(h, head).float()
        self.last_hidden_states = tuple(captured)
        self.position += 1
        result = logits.cpu().numpy()
        del head, logits, h
        torch.cuda.empty_cache()
        return result


def torch_worker(model: str, tokens: tuple[int, ...], output: Path) -> None:
    oracle = TorchMapleOracle(model)
    rows: list[np.ndarray] = []
    hidden_rows: list[np.ndarray] = []
    elapsed: list[float] = []
    try:
        for token in tokens:
            started = time.perf_counter()
            rows.append(oracle.step(token))
            hidden_rows.append(np.stack(oracle.last_hidden_states))
            oracle.torch.cuda.synchronize()
            elapsed.append((time.perf_counter() - started) * 1_000.0)
    finally:
        oracle.close()
    logits = np.stack(rows)
    np.savez(
        output,
        logits=logits,
        top_ids=np.argmax(logits, axis=-1).astype(np.int64),
        hidden=np.stack(hidden_rows),
        elapsed_ms=np.asarray(elapsed, dtype=np.float64),
    )


def hf_worker(
    model: str,
    revision: str,
    device: str,
    tokens: tuple[int, ...],
    output: Path,
    *,
    packed_model: str,
    cache_dir: Path | None,
    local_files_only: bool,
    offload_experts: bool,
    offload_dir: Path | None,
    match_packed_affine4: bool,
) -> None:
    """Run pinned Transformers remote code one teacher-forced token at a time."""

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.activations import ACT2FN as _act2fn
    from transformers.modeling_utils import PreTrainedModel as _pretrained_model

    if transformers.__version__ != _HF_TRANSFORMERS_VERSION:
        raise RuntimeError(
            "Dense Maple oracle requires the checkpoint-pinned "
            f"transformers=={_HF_TRANSFORMERS_VERSION}, found {transformers.__version__}"
        )
    # Let Transformers cache the real package-availability state before publishing
    # the API-only shim needed by Maple's pinned remote ``fa3.py`` import.
    del _act2fn, _pretrained_model
    install_hf_flash_attention_shim()
    load_kwargs = {
        "revision": revision,
        "code_revision": revision,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
    }
    load_context = nullcontext()
    if offload_experts:
        config = AutoConfig.from_pretrained(
            model,
            revision=revision,
            code_revision=revision,
            trust_remote_code=True,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        device_map = maple_hf_expert_device_map(
            layers=config.num_hidden_layers,
            experts=config.num_experts,
            device=device,
        )
        target_offload_dir = offload_dir or output.parent / "hf-expert-offload"
        target_offload_dir.mkdir(parents=True, exist_ok=True)
        load_kwargs.update(
            config=config,
            device_map=device_map,
            offload_folder=target_offload_dir,
        )
        load_context = hf_maple_expert_preload_dispatch()
    else:
        load_kwargs["device_map"] = {"": device}
    with load_context:
        oracle = AutoModelForCausalLM.from_pretrained(model, **load_kwargs)
    oracle.eval()
    if match_packed_affine4:
        match_hf_affine4_from_packed(
            oracle,
            packed_model,
            device=device,
        )
    disk_offloaded_expert_modules = (
        sum(value == "disk" for value in oracle.hf_device_map.values())
        if offload_experts
        else 0
    )
    rows: list[np.ndarray] = []
    elapsed: list[float] = []
    past_key_values = None
    try:
        with torch.inference_mode():
            for token in tokens:
                started = time.perf_counter()
                result = oracle(
                    input_ids=torch.tensor([[token]], dtype=torch.long, device=device),
                    past_key_values=past_key_values,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                past_key_values = result.past_key_values
                torch.cuda.synchronize()
                rows.append(result.logits[0, -1].float().cpu().numpy())
                elapsed.append((time.perf_counter() - started) * 1_000.0)
    finally:
        del past_key_values, oracle
        torch.cuda.empty_cache()
    logits = np.stack(rows)
    np.savez(
        output,
        logits=logits,
        top_ids=np.argmax(logits, axis=-1).astype(np.int64),
        elapsed_ms=np.asarray(elapsed, dtype=np.float64),
        oracle_model=np.asarray(model),
        oracle_revision=np.asarray(revision),
        transformers_version=np.asarray(transformers.__version__),
        torch_version=np.asarray(torch.__version__),
        expert_disk_offload=np.asarray(offload_experts),
        matched_packed_affine4=np.asarray(match_packed_affine4),
        packed_affine4_model=np.asarray(packed_model),
        disk_offloaded_expert_modules=np.asarray(
            disk_offloaded_expert_modules, dtype=np.int64
        ),
    )


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def compare(
    hip_path: Path,
    oracle_path: Path,
    tokens: tuple[int, ...],
    *,
    oracle_description: str,
) -> dict:
    with np.load(hip_path) as hip, np.load(oracle_path) as oracle:
        hip_logits = hip["logits"]
        oracle_logits = oracle["logits"]
        p = softmax(oracle_logits)
        q = softmax(hip_logits)
        kl = np.sum(
            p * np.log(np.maximum(p, 1e-300) / np.maximum(q, 1e-300)),
            axis=-1,
        )
        hip_top = hip["top_ids"].astype(np.int64)
        oracle_top = oracle["top_ids"].astype(np.int64)
        agreement = float(np.mean(hip_top == oracle_top))
        max_abs = np.max(np.abs(hip_logits - oracle_logits), axis=-1)
        maximum_kl = float(np.max(kl))
        result = {
            "schema_version": 1,
            "model": "deepgrove/maple-preview-2bit-mlx",
            "token_ids": list(tokens),
            "positions": len(tokens),
            "evaluation_mode": "teacher_forced_token_serial_decode",
            "max_kl": maximum_kl,
            "mean_kl": float(np.mean(kl)),
            "top1_agreement": agreement,
            "hip_top_ids": hip_top.tolist(),
            "oracle_top_ids": oracle_top.tolist(),
            "per_position_kl": kl.tolist(),
            "per_position_max_abs": max_abs.astype(float).tolist(),
            "hip_elapsed_ms": hip["elapsed_ms"].astype(float).tolist(),
            "oracle_elapsed_ms": oracle["elapsed_ms"].astype(float).tolist(),
            "gate": {
                "kl_le_0_05": bool(maximum_kl <= 0.05),
                "top1_ge_0_90": bool(agreement >= 0.90),
            },
            "passed": bool(maximum_kl <= 0.05 and agreement >= 0.90),
            "oracle": oracle_description,
        }
        if "greedy_argmax_exact" in hip.files:
            result["hip_greedy_argmax_exact"] = bool(
                hip["greedy_argmax_exact"].item()
            )
        if "hidden" in hip.files and "hidden" in oracle.files:
            hip_hidden = hip["hidden"].astype(np.float64)
            oracle_hidden = oracle["hidden"].astype(np.float64)
            hidden_delta = hip_hidden - oracle_hidden
            hidden_dot = np.sum(hip_hidden * oracle_hidden, axis=-1)
            hidden_norm = np.linalg.norm(hip_hidden, axis=-1) * np.linalg.norm(
                oracle_hidden, axis=-1
            )
            layer_count = hip_hidden.shape[1] - 2
            result.update(
                hidden_labels=[
                    "embedding",
                    *[f"layer_{layer}" for layer in range(layer_count)],
                    "final_norm",
                ],
                hidden_max_abs=np.max(np.abs(hidden_delta), axis=-1)
                .astype(float)
                .tolist(),
                hidden_cosine=(
                    hidden_dot / np.maximum(hidden_norm, 1.0e-30)
                ).astype(float).tolist(),
            )
        for key in (
            "oracle_model",
            "oracle_revision",
            "transformers_version",
            "torch_version",
        ):
            if key in oracle.files:
                result[key] = str(oracle[key].item())
        if "expert_disk_offload" in oracle.files:
            result["expert_disk_offload"] = bool(
                oracle["expert_disk_offload"].item()
            )
        if "disk_offloaded_expert_modules" in oracle.files:
            result["disk_offloaded_expert_modules"] = int(
                oracle["disk_offloaded_expert_modules"].item()
            )
        if "matched_packed_affine4" in oracle.files:
            result["matched_packed_affine4"] = bool(
                oracle["matched_packed_affine4"].item()
            )
        if "packed_affine4_model" in oracle.files:
            result["packed_affine4_model"] = str(
                oracle["packed_affine4_model"].item()
            )
        return result


def main() -> int:
    args = parse_args()
    tokens = token_ids(args.token_ids)
    if args.worker:
        if args.output is None:
            raise ValueError("--output is required for worker mode")
        if args.worker == "hipengine":
            hipengine_worker(args.model, args.backend, tokens, args.output)
        elif args.worker == "torch":
            torch_worker(args.model, tokens, args.output)
        else:
            hf_worker(
                args.hf_model,
                args.hf_revision,
                args.hf_device,
                tokens,
                args.output,
                packed_model=args.model,
                cache_dir=args.hf_cache_dir,
                local_files_only=args.hf_local_files_only,
                offload_experts=args.hf_offload_experts,
                offload_dir=args.hf_offload_dir,
                match_packed_affine4=args.hf_match_packed_affine4,
            )
        return 0

    with tempfile.TemporaryDirectory(prefix="maple-correctness-") as temp:
        root = Path(temp)
        hip_path = root / "hipengine.npz"
        oracle_path = root / "oracle.npz"
        common = [
            "--model",
            args.model,
            "--backend",
            args.backend,
            "--token-ids",
            args.token_ids,
        ]
        subprocess.run(
            [
                sys.executable,
                __file__,
                *common,
                "--worker",
                "hipengine",
                "--output",
                str(hip_path),
            ],
            check=True,
        )
        oracle_worker = "torch" if args.oracle == "packed" else "hf"
        oracle_command = [
            sys.executable,
            __file__,
            *common,
            "--worker",
            oracle_worker,
            "--hf-model",
            args.hf_model,
            "--hf-revision",
            args.hf_revision,
            "--hf-device",
            args.hf_device,
            "--output",
            str(oracle_path),
        ]
        if args.hf_cache_dir is not None:
            oracle_command.extend(("--hf-cache-dir", str(args.hf_cache_dir)))
        if args.hf_local_files_only:
            oracle_command.append("--hf-local-files-only")
        if args.hf_offload_experts:
            oracle_command.append("--hf-offload-experts")
        if args.hf_offload_dir is not None:
            oracle_command.extend(("--hf-offload-dir", str(args.hf_offload_dir)))
        if args.hf_match_packed_affine4:
            oracle_command.append("--hf-match-packed-affine4")
        subprocess.run(oracle_command, check=True)
        oracle_description = (
            "independent torch execution of official HF Maple formulas over packed weights"
            if args.oracle == "packed"
            else "Transformers trust_remote_code dense Maple checkpoint with pure-torch attention API shim"
        )
        if args.oracle == "hf" and args.hf_offload_experts:
            oracle_description += " and safetensors-backed routed-expert residency"
        if args.oracle == "hf" and args.hf_match_packed_affine4:
            oracle_description += " with checkpoint-matched affine4 embeddings/head"
        result = compare(
            hip_path,
            oracle_path,
            tokens,
            oracle_description=oracle_description,
        )
        result["model"] = args.model
        result["backend"] = args.backend
        result["oracle_kind"] = args.oracle
        if args.keep_logits:
            args.keep_logits.parent.mkdir(parents=True, exist_ok=True)
            with np.load(hip_path) as hip, np.load(oracle_path) as oracle:
                np.savez(
                    args.keep_logits,
                    hip_logits=hip["logits"],
                    oracle_logits=oracle["logits"],
                )
        rendered = json.dumps(result, indent=2, sort_keys=True)
        print(rendered)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(rendered + "\n", encoding="utf-8")
        return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
