#!/usr/bin/env python3
"""Capture one identical-operand packet from a real production prefill.

A cross-engine comparison is only meaningful if both engines compute the same
operation on the same bytes. This script records exactly that for one dense
projection: the resident weight bytes with a hash identity, the activation
matrix the runner actually fed to the kernel, the output the selected hipEngine
kernel produced, and the dispatch key it was produced by.

It does not change any model output. The hook wraps the selected kernel, calls
it unchanged, and copies buffers aside afterwards.

The packet is written as a pair:

* ``<stem>.npz``      the raw bytes: ``w_raw`` (uint8), ``x`` (float32 or
                      uint16 bf16, per the recorded activation dtype), and
                      ``out`` (float32 or the recorded output dtype).
* ``<stem>.json``     the manifest: model and host identity, prompt and category,
                      chunk geometry, layer and slot, the resolved hipEngine
                      variant, and the shape/dtype/stride of every array.

Activation and output arrays are stored in the dtype the kernel was actually
handed, not converted, so the replay adapters receive byte-identical operands.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The runner names each dense projection by its own slot path, which is
# ``layers.<i>.<slot>`` rather than the GGUF tensor name. The layer index and
# the slot are read back from the weight rather than assumed.
SLOT_TEMPLATE = "layers.{layer}.attn_qkv"

_DTYPE_BYTES = {"f32": 4, "bf16": 2, "fp16": 2}

# Activation dtype per launch ABI, for the ABIs whose variant name does not spell
# it out. ``raw_mmq_d4x3`` reads its activation as bf16 (it quantizes bf16 to
# Q8_1 into the session workspace), which the ``..._f32_f32_out`` variant name
# does not reveal. ABIs absent here fall back to the variant-name parse.
ABI_ACTIVATION_DTYPE = {
    "raw_mmq_d4x3": "bf16",
    "raw_mmq_d4x3_f32": "f32",
    "raw_mmq_d4x2_f32": "f32",
}


def _parse_key(key: object) -> dict[str, str]:
    return {
        "backend": str(getattr(key, "backend", "")),
        "layer": str(getattr(key, "layer", "")),
        "quant": str(getattr(key, "quant", "")),
        "variant": str(getattr(key, "variant", "")),
    }


def _dtypes_from_variant(variant: str, abi: str | None = None) -> tuple[str, str]:
    """Recover (activation, output) dtype names for a launched kernel.

    Most Q8_0 prefill variants end in ``<activation>_<output>_out``, so the dtype
    can be read back from the selected key instead of guessed. A few ABIs take an
    activation whose dtype the name does not describe, so those are declared
    explicitly in :data:`ABI_ACTIVATION_DTYPE`.
    """
    parts = variant.split("_")
    if len(parts) >= 3 and parts[-1] == "out":
        act, out = parts[-3], parts[-2]
        if act in _DTYPE_BYTES and out in _DTYPE_BYTES:
            return ABI_ACTIVATION_DTYPE.get(abi or "", act), out
    raise ValueError(f"cannot read dtypes from variant {variant!r}")


def _download(runtime, ptr: int, shape: tuple[int, ...], dtype: str) -> np.ndarray:
    from hipengine.core.hip import MemcpyKind

    if dtype == "f32":
        array = np.empty(shape, dtype=np.float32)
    elif dtype in ("bf16", "fp16"):
        array = np.empty(shape, dtype=np.uint16)
    else:
        raise ValueError(f"unsupported capture dtype {dtype!r}")
    runtime.memcpy(array.ctypes.data, ptr, array.nbytes, MemcpyKind.DEVICE_TO_HOST)
    return array


def _to_float(array: np.ndarray, dtype: str) -> np.ndarray:
    if dtype == "f32":
        return array.astype(np.float32, copy=False)
    if dtype == "bf16":
        return (array.astype(np.uint32) << 16).view(np.float32)
    return array.view(np.float16).astype(np.float32)


def capture(args: argparse.Namespace) -> int:
    from hipengine.core.memory import memory_stats, reset_memory_stats
    from hipengine.generation.qwen4_exp_profiles import register_qwen4_exp_gfx1151_profiles
    from hipengine.kernels.hip_gfx1151 import register_gfx1151_kernels
    from hipengine.kernels.registry import KernelKey
    from hipengine.runtime import gguf_linear as linear
    from hipengine.runtime import qwen4_exp_runner as runner
    from scripts.qwen4exp_canonical_ar_bench import load_fixture
    from scripts.qwen4exp_framework_family_refresh import check_host, model_identity
    from scripts.qwen4exp_layer2_profile_gate import _make_generator

    check_host()
    os.environ.setdefault("HIPENGINE_HIP_ARCH", "gfx1151")
    if args.compiler_version_file is not None:
        os.environ["HIPENGINE_COMPILER_VERSION_FILE"] = str(args.compiler_version_file)

    fixture, fixture_sha256 = load_fixture(args.fixture)
    case = None
    for row in fixture["cases"]:
        if str(row["id"]) == args.case_id:
            case = row
            break
    if case is None:
        raise SystemExit(
            f"case {args.case_id!r} not in {args.fixture}; available: "
            + ", ".join(sorted(str(r["id"]) for r in fixture["cases"]))
        )
    tokens = [int(token) for token in case["prompt_token_ids"]]

    args.max_sequence_length = max(int(args.max_sequence_length), len(tokens) + 64)
    args.prefill_chunk_size = int(args.chunk_size)
    args.decode_steps = 0

    register_gfx1151_kernels(replace=True)
    register_qwen4_exp_gfx1151_profiles()

    reset_memory_stats()
    identity = model_identity(args.model_root)
    generator, profile, _ = _make_generator(args, args.profile)

    target_layer = args.layer
    seen: dict[tuple[int, str], int] = {}
    launch_log: dict[str, int] = {}
    packets: list[dict] = []
    active: dict[str, object] = {}
    original_launch = runner.launch_gguf_linear
    original_resolve = linear.resolve
    original_abi = dict(linear._LAUNCH_ABI)

    def _instrumented_abi(abi_name, abi_fn):
        def wrapper(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs):
            active["abi"] = abi_name
            return abi_fn(fn, weight, x_ptr, out_ptr, rows, in_features, out_features, kwargs)

        return wrapper

    def _is_target(slot: object, out_features: int) -> bool:
        return (
            isinstance(slot, str)
            and slot == SLOT_TEMPLATE.format(layer=target_layer)
            and out_features == args.out_features
        )

    def launch(weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs):
        previous = dict(active)
        active.clear()
        slot = getattr(weight.spec, "slot_path", None)
        layer = -1
        if isinstance(slot, str):
            parts = slot.split(".")
            if len(parts) >= 2 and parts[0] == "layers":
                with contextlib.suppress(ValueError):
                    layer = int(parts[1])
        active.update(
            weight=weight, slot=slot, layer=layer, rows=int(rows),
            in_features=int(in_features), out_features=int(out_features),
        )
        label = f"{slot}|rows={int(rows)}|K={int(in_features)}|M={int(out_features)}"
        launch_log[label] = launch_log.get(label, 0) + 1
        try:
            return original_launch(
                weight, x_ptr, out_ptr, rows, in_features, out_features, **kwargs
            )
        finally:
            active.clear()
            active.update(previous)

    def resolve(*call_args, **call_kwargs):
        function = original_resolve(*call_args, **call_kwargs)
        context = dict(active)
        if not context:
            return function
        key_args = dict(call_kwargs)
        if call_args:
            key_args.setdefault("backend", call_args[0])
        key = KernelKey(
            str(key_args.get("backend", "")),
            str(key_args.get("layer", "")),
            str(key_args.get("quant", "")),
            str(key_args.get("variant", "")),
        )
        if not _is_target(context.get("slot"), context.get("out_features")):
            return function

        def recorded(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs):
            call_kwargs_inner = dict(kwargs)
            runtime = call_kwargs_inner.get("runtime")
            if runtime is None:
                return function(
                    x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs
                )
            ordinal_key = (int(context["layer"]), str(context["slot"]))
            ordinal = seen.get(ordinal_key, 0)
            seen[ordinal_key] = ordinal + 1
            should_capture = ordinal == args.chunk and rows == args.rows
            if not should_capture:
                return function(
                    x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs
                )

            activation_dtype, output_dtype = _dtypes_from_variant(
                key.variant, active.get("abi")
            )
            function(x_ptr, w_ptr, out_ptr, rows, in_features, out_features, **kwargs)
            runtime.device_synchronize()

            weight_nbytes = int(out_features) * (int(in_features) // 32 * 34)
            w_raw = np.empty(weight_nbytes, dtype=np.uint8)
            from hipengine.core.hip import MemcpyKind

            runtime.memcpy(
                w_raw.ctypes.data, w_ptr, weight_nbytes, MemcpyKind.DEVICE_TO_HOST
            )
            x = _download(runtime, x_ptr, (int(rows), int(in_features)), activation_dtype)
            out = _download(runtime, out_ptr, (int(rows), int(out_features)), output_dtype)
            packets.append(
                {
                    "key": _parse_key(key),
                    "abi": active.get("abi"),
                    "weight": context["weight"],
                    "ordinal": ordinal,
                    "rows": int(rows),
                    "in_features": int(in_features),
                    "out_features": int(out_features),
                    "activation_dtype": activation_dtype,
                    "output_dtype": output_dtype,
                    "w_raw": w_raw,
                    "x": x,
                    "out": out,
                    "weight_nbytes": weight_nbytes,
                }
            )
            return None

        return recorded

    runner.launch_gguf_linear = launch
    linear.resolve = resolve
    for abi_name, abi_fn in original_abi.items():
        linear._LAUNCH_ABI[abi_name] = _instrumented_abi(abi_name, abi_fn)
    try:
        import fcntl

        with open("/tmp/hipengine-gfx1151-benchmark.lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            generator.runner.reset()
            generator.runner.prefill([int(token) for token in tokens])
            generator.runner.runtime.device_synchronize()
    finally:
        runner.launch_gguf_linear = original_launch
        linear.resolve = original_resolve
        linear._LAUNCH_ABI.clear()
        linear._LAUNCH_ABI.update(original_abi)

    if not packets:
        distinct = ", ".join(f"{k} x{v}" for k, v in sorted(launch_log.items()))
        raise SystemExit(
            f"no packet captured for layer {target_layer} chunk {args.chunk} "
            f"rows {args.rows} out_features {args.out_features}; "
            f"{len(launch_log)} distinct launches observed: {distinct}"
        )
    if len(packets) > 1:
        raise SystemExit(f"capture is not unique: {len(packets)} packets matched")

    packet = packets[0]
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stem.with_suffix(".npz"),
        w_raw=packet["w_raw"],
        x=packet["x"],
        out=packet["out"],
    )

    memory = memory_stats()
    manifest = {
        "schema": 1,
        "kind": "identical-operand-packet",
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "gpu": os.environ.get("HIPENGINE_HIP_ARCH", ""),
        },
        "model": identity,
        "model_root": str(args.model_root),
        "prompt": {
            "fixture": str(args.fixture),
            "fixture_sha256": fixture_sha256,
            "case_id": str(case["id"]),
            "category": str(case["category"]),
            "tokens": len(tokens),
            "prompt_token_ids_sha256": str(case["prompt_token_ids_sha256"]),
        },
        "profile": str(profile.profile.value) if hasattr(profile, "profile") else str(profile),
        "chunk": {
            "size": int(args.chunk_size),
            "index": int(args.chunk),
            "row_start": int(args.chunk) * int(args.chunk_size),
        },
        "layer": int(target_layer),
        "slot": str(packet["weight"].spec.slot_path),
        "quant": str(packet["weight"].spec.quant_key),
        "hipengine_variant": packet["key"],
        "hipengine_abi": packet["abi"],
        "geometry": {
            "rows": packet["rows"],
            "in_features": packet["in_features"],
            "out_features": packet["out_features"],
        },
        "arrays": {
            "w_raw": {
                "dtype": "uint8",
                "shape": list(packet["w_raw"].shape),
                "sha256": hashlib.sha256(packet["w_raw"].tobytes()).hexdigest(),
                "nbytes": int(packet["w_raw"].nbytes),
            },
            "x": {
                "dtype": packet["activation_dtype"],
                "shape": list(packet["x"].shape),
                "sha256": hashlib.sha256(packet["x"].tobytes()).hexdigest(),
                "nbytes": int(packet["x"].nbytes),
            },
            "out": {
                "dtype": packet["output_dtype"],
                "shape": list(packet["out"].shape),
                "sha256": hashlib.sha256(packet["out"].tobytes()).hexdigest(),
                "nbytes": int(packet["out"].nbytes),
            },
        },
        "x_float_summary": {
            "max_abs": float(np.max(np.abs(_to_float(packet["x"], packet["activation_dtype"])))),
            "rms": float(np.sqrt(np.mean(_to_float(packet["x"], packet["activation_dtype"]) ** 2))),
            "nonfinite": int(np.count_nonzero(~np.isfinite(_to_float(packet["x"], packet["activation_dtype"])))),
        },
        "launches_observed": {
            f"layer{layer}:{slot}": count for (layer, slot), count in sorted(seen.items())
        },
        "launch_log": dict(sorted(launch_log.items())),
        "memory": memory,
    }
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    print(f"packet      : {stem.with_suffix('.npz')}")
    print(f"manifest    : {stem.with_suffix('.json')}")
    print(f"variant     : {packet['key']['backend']}/{packet['key']['layer']}/"
          f"{packet['key']['quant']}/{packet['key']['variant']} (abi={packet['abi']})")
    print(f"geometry    : rows={packet['rows']} K={packet['in_features']} "
          f"M={packet['out_features']}")
    print(f"dtypes      : activation={packet['activation_dtype']} "
          f"output={packet['output_dtype']}")
    print(f"weight      : {packet['w_raw'].nbytes} bytes "
          f"sha256={manifest['arrays']['w_raw']['sha256'][:16]}")
    print(f"activations : max_abs={manifest['x_float_summary']['max_abs']:.4f} "
          f"rms={manifest['x_float_summary']['rms']:.4f} "
          f"nonfinite={manifest['x_float_summary']['nonfinite']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--compiler-version-file", type=Path)
    parser.add_argument("--output", type=Path, required=True,
                        help="packet stem; .npz and .json are written beside it")
    parser.add_argument(
        "--fixture",
        type=Path,
        default=ROOT / "benchmarks/fixtures/qwen4exp_canonical_ar_p512_p1024_p4096.json",
        help="canonical exact-token fixture, so the packet is reproducible",
    )
    parser.add_argument("--case-id", default="code-p4096")
    parser.add_argument("--layer", type=int, default=8,
                        help="layer index of the target dense projection")
    parser.add_argument("--chunk", type=int, default=0,
                        help="zero-based prefill chunk index to capture")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=1024,
                        help="rows the target launch must carry")
    parser.add_argument("--out-features", type=int, default=10240)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--profile", default="strict")
    return capture(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
