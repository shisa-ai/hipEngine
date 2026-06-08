"""Shared Qwen3.5/PARO KV policy CLI helpers."""

from __future__ import annotations

import argparse
from typing import Any

from hipengine.kvcache import (
    KV_SCALE_DTYPE_CHOICES,
    KV_SCALE_GRANULARITY_CHOICES,
    KV_STORAGE_AUTO,
    KV_STORAGE_CHOICES,
    ResolvedKVPolicy,
    resolve_kv_policy,
)


def add_kv_policy_args(
    parser: argparse.ArgumentParser,
    *,
    default_storage: str = KV_STORAGE_AUTO,
    legacy_storage_flags: tuple[str, ...] = (),
    help_prefix: str = "Resident full-attention KV",
) -> None:
    """Add common KV policy controls to a script parser."""

    parser.add_argument(
        "--kv-storage",
        *legacy_storage_flags,
        dest="kv_storage",
        choices=KV_STORAGE_CHOICES,
        default=default_storage,
        help=(
            f"{help_prefix} storage policy. 'auto' resolves conservatively to BF16 "
            "unless the runtime marks INT8 as admission-gated."
        ),
    )
    parser.add_argument(
        "--kv-scale-dtype",
        choices=KV_SCALE_DTYPE_CHOICES,
        default="fp16",
        help="Scale tensor dtype for int8_per_token_head KV storage.",
    )
    parser.add_argument(
        "--kv-scale-granularity",
        choices=KV_SCALE_GRANULARITY_CHOICES,
        default="per_token_head",
        help="Scale granularity for int8_per_token_head KV storage.",
    )


def resolve_args_kv_policy(
    args: argparse.Namespace,
    *,
    block_size: int = 256,
    admission_gated_int8: bool = False,
) -> ResolvedKVPolicy:
    return resolve_kv_policy(
        getattr(args, "kv_storage", KV_STORAGE_AUTO),
        block_size=block_size,
        scale_dtype=getattr(args, "kv_scale_dtype", "fp16"),
        scale_granularity=getattr(args, "kv_scale_granularity", "per_token_head"),
        admission_gated_int8=admission_gated_int8,
    )


def kv_policy_json(policy: ResolvedKVPolicy) -> dict[str, Any]:
    return policy.to_json_dict()


def append_kv_policy_flags(command: str, args: argparse.Namespace) -> str:
    """Append non-default KV policy flags to a reproduced command string."""

    if getattr(args, "kv_storage", KV_STORAGE_AUTO) != KV_STORAGE_AUTO:
        command += f" --kv-storage {args.kv_storage}"
    if getattr(args, "kv_scale_dtype", "fp16") != "fp16":
        command += f" --kv-scale-dtype {args.kv_scale_dtype}"
    if getattr(args, "kv_scale_granularity", "per_token_head") != "per_token_head":
        command += f" --kv-scale-granularity {args.kv_scale_granularity}"
    return command
