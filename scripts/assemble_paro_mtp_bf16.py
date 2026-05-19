#!/usr/bin/env python3
"""Assemble a local PARO+MTP-BF16 artifact for Qwen3.6-35B-A3B.

The shipped PARO target contains the quantized trunk but no target-attached
``mtp.*`` tensors.  Qwen's FP8 artifact publishes ``mtp.safetensors`` with
block-FP8 MTP projections/experts plus BF16 norms/router/fc.  This script
creates a local, non-committed model directory that reuses the existing PARO
trunk (symlinks by default) and writes a BF16 MTP sidecar shard in the compact
runtime layout expected by ``hipengine.loading.mtp``.

Model weights are deliberately written outside the git tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from hipengine.loading import DFLASH_PACKED_TARGET_MODEL, validate_qwen35_mtp_model
from hipengine.loading.qwen35_paro import qwen35_paro_config_from_hf

DEFAULT_FP8_MTP_REPO = "Qwen/Qwen3.6-35B-A3B-FP8"
DEFAULT_OUTPUT = Path("/models/hipengine/Qwen3.6-35B-A3B-PARO-full4096-e5-packed-MTP-BF16")
BLOCK_SIZE = 128

_METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md",
}


def _resolve_mtp_source(value: str | Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(str(value), filename="mtp.safetensors"))


def _link_or_copy(src: Path, dst: Path, *, symlink: bool, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve())
    else:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def _prepare_output_dir(target: Path, output: Path, *, symlink_trunk: bool, overwrite: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for src in sorted(target.iterdir()):
        if src.name == "model.safetensors.index.json":
            # Avoid an index that excludes the new MTP sidecar; hipEngine can
            # discover all shards directly when no index is present.
            continue
        if src.suffix == ".safetensors" or src.name in _METADATA_FILES:
            _link_or_copy(src, output / src.name, symlink=symlink_trunk, overwrite=overwrite)


def _expand_scale(scale_inv: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    rows, cols = int(shape[0]), int(shape[1])
    expanded = scale_inv.to(torch.float32).repeat_interleave(BLOCK_SIZE, dim=0).repeat_interleave(BLOCK_SIZE, dim=1)
    return expanded[:rows, :cols]


def _dequant_fp8_weight(fh: safe_open, name: str) -> torch.Tensor:
    weight = fh.get_tensor(name)
    if weight.dtype == torch.bfloat16:
        return weight.contiguous()
    if weight.dtype not in {torch.float8_e4m3fn, torch.float8_e4m3fnuz}:
        raise TypeError(f"unsupported MTP source dtype for {name}: {weight.dtype}")
    scale_name = f"{name}_scale_inv"
    scale_inv = fh.get_tensor(scale_name)
    deq = weight.to(torch.float32) * _expand_scale(scale_inv, weight.shape)
    return deq.to(torch.bfloat16).contiguous()


def _copy_bf16(fh: safe_open, name: str) -> torch.Tensor:
    tensor = fh.get_tensor(name)
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"expected BF16 tensor {name}, got {tensor.dtype}")
    return tensor.contiguous()


def _assemble_mtp_tensors(source: Path, *, config: dict[str, Any]) -> dict[str, torch.Tensor]:
    qwen = qwen35_paro_config_from_hf(config)
    if qwen.hidden_size != 2048 or qwen.num_experts != 256 or qwen.moe_intermediate_size != 512:
        raise ValueError(
            "assembler currently targets Qwen3.6-35B-A3B MTP dimensions "
            f"(got hidden={qwen.hidden_size}, experts={qwen.num_experts}, moe={qwen.moe_intermediate_size})"
        )
    out: dict[str, torch.Tensor] = {}
    with safe_open(source, framework="pt", device="cpu") as fh:
        for name in [
            "mtp.fc.weight",
            "mtp.pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight",
            "mtp.layers.0.input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight",
            "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.shared_expert_gate.weight",
            "mtp.norm.weight",
        ]:
            out[name] = _copy_bf16(fh, name)

        for name in [
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight",
            "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
            "mtp.layers.0.mlp.shared_expert.up_proj.weight",
            "mtp.layers.0.mlp.shared_expert.down_proj.weight",
        ]:
            out[name] = _dequant_fp8_weight(fh, name)

        gate_up = torch.empty(
            (qwen.num_experts, 2 * qwen.moe_intermediate_size, qwen.hidden_size),
            dtype=torch.bfloat16,
        )
        down = torch.empty(
            (qwen.num_experts, qwen.hidden_size, qwen.moe_intermediate_size),
            dtype=torch.bfloat16,
        )
        for expert in range(qwen.num_experts):
            gate = _dequant_fp8_weight(fh, f"mtp.layers.0.mlp.experts.{expert}.gate_proj.weight")
            up = _dequant_fp8_weight(fh, f"mtp.layers.0.mlp.experts.{expert}.up_proj.weight")
            gate_up[expert] = torch.cat((gate, up), dim=0)
            down[expert] = _dequant_fp8_weight(fh, f"mtp.layers.0.mlp.experts.{expert}.down_proj.weight")
        out["mtp.layers.0.mlp.experts.gate_up_proj"] = gate_up.contiguous()
        out["mtp.layers.0.mlp.experts.down_proj"] = down.contiguous()
    return out


def _tensor_summary(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    return {name: {"dtype": str(tensor.dtype).replace("torch.", ""), "shape": list(tensor.shape)} for name, tensor in sorted(tensors.items())}


def assemble_paro_mtp_bf16(
    *,
    target_model: Path,
    mtp_source: Path,
    output: Path,
    symlink_trunk: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    target_model = target_model.resolve()
    mtp_source = mtp_source.resolve()
    output = output.resolve()
    if not target_model.exists():
        raise FileNotFoundError(f"target model not found: {target_model}")
    if not mtp_source.exists():
        raise FileNotFoundError(f"MTP source safetensors not found: {mtp_source}")
    config = json.loads((target_model / "config.json").read_text(encoding="utf-8"))
    _prepare_output_dir(target_model, output, symlink_trunk=symlink_trunk, overwrite=overwrite)
    tensors = _assemble_mtp_tensors(mtp_source, config=config)
    sidecar = output / "mtp-bf16.safetensors"
    if sidecar.exists() and not overwrite:
        raise FileExistsError(f"MTP sidecar already exists: {sidecar} (use --overwrite)")
    save_file(tensors, str(sidecar), metadata={"format": "pt", "source": str(mtp_source), "conversion": "fp8_block128_to_bf16"})
    validation = validate_qwen35_mtp_model(output)
    summary = {
        "artifact_kind": "qwen36_paro_mtp_bf16_local_assembly",
        "target_model": str(target_model),
        "mtp_source": str(mtp_source),
        "output_model": str(output),
        "sidecar": str(sidecar),
        "symlink_trunk": bool(symlink_trunk),
        "sidecar_bytes": sidecar.stat().st_size,
        "tensor_count": len(tensors),
        "tensors": _tensor_summary(tensors),
        "validation": validation.to_json_dict(),
    }
    (output / "hipengine_mtp_assembly.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not validation.passed:
        validation.raise_for_errors()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-model", type=Path, default=Path(DFLASH_PACKED_TARGET_MODEL), help="Existing PARO packed target directory")
    parser.add_argument("--mtp-source", default=DEFAULT_FP8_MTP_REPO, help="Path to mtp.safetensors or HF repo id containing it")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output local PARO+MTP artifact directory")
    parser.add_argument("--copy-trunk", action="store_true", help="Copy trunk files instead of symlinking them")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files/sidecar")
    args = parser.parse_args()

    source = _resolve_mtp_source(args.mtp_source)
    summary = assemble_paro_mtp_bf16(
        target_model=args.target_model,
        mtp_source=source,
        output=args.output,
        symlink_trunk=not args.copy_trunk,
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({
        "output_model": summary["output_model"],
        "sidecar_bytes": summary["sidecar_bytes"],
        "validation_passed": summary["validation"]["passed"],
        "present_count": summary["validation"]["present_count"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    raise SystemExit(main())
