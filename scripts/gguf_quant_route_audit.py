#!/usr/bin/env python3
"""Report which GGUF quant types a Qwen dense GGUF file needs and how they load.

Answers one question per file: after hipEngine validates the tensor map and runs
each tensor through the production weight planner, which tensors get a real
quantized kernel, which get expanded to dense BF16 at load, and which make the
loader refuse the file. Reads metadata and the tensor-info table only; no weights
are read, so a partially downloaded file can still be inspected. Nothing here runs
a model, allocates device memory, or touches a GPU.

Why not hipengine.loading.gguf.scan_gguf: it validates every tensor byte range
against the on-disk file size, which aborts on an incomplete download. This script
reuses the same decoders and skips only that final range check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from math import prod
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hipengine.loading.gguf import (  # noqa: E402
    GGUF_DEFAULT_ALIGNMENT,
    GGUF_MAGIC,
    GGUFModelInfo,
    GGUFTensorInfo,
    _align_up,
    _read_exact,
    _read_scalar,
    _read_string,
    _read_value,
)
from hipengine.loading.qwen35_gguf import (  # noqa: E402
    build_qwen35_gguf_tensor_map,
)
from hipengine.loading.qwen35_gguf_materialize import (  # noqa: E402
    gguf_decode_repack_enabled,
    plan_qwen35_gguf_weight_spec,
)
from hipengine.quant.gguf import (  # noqa: E402
    GGUFValueType,
    ggml_type,
    ggml_type_name,
    llama_file_type_name,
    nbytes_for_shape,
    quant_shape_to_byte_shape,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# hipengine.kernels.<backend> holds these as plain module constants. Importing the
# package runs kernel registration, which needs the HIP runtime, so the values are
# read from source instead. A renamed constant therefore shows up as False here,
# not as a silent pass: --check-constants fails if a name disappears.
CAPABILITY_NAMES = (
    "GGUF_DENSE_Q4_T16",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP",
    "GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES",
    "GGUF_DENSE_Q4_T16_ATTN_Q_08B",
    "GGUF_DENSE_Q5_T16_SSM_OUT",
    "GGUF_DENSE_Q5_T16_SSM_OUT_08B",
    "GGUF_DENSE_Q5_T16_QKV",
    "GGUF_DENSE_Q5_T16_H5120",
    "GGUF_DENSE_Q6_T16_QMICRO_PLANAR",
    "GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES",
)
LAYOUT_MEANING = {
    "dense_bf16": "bf16-expand",
    "dense_f32": "f32-resident",
    "raw_gguf": "raw-gguf-kernel",
}
GIB = 2**30


def quoted_members(source_value: str) -> set[str]:
    """Members of a container constant written as a source expression.

    ``frozenset({"a", "b"})``, ``("a",)``, and ``set()`` all reduce to the quoted
    strings inside them. Used to read backend capability constants without
    importing the kernel package.
    """

    return {value.strip().lower() for value in re.findall(r"['\"]([^'\"]+)['\"]", source_value)}


def fp16_recurrent_state_default(backend: str, file_type_name: str) -> bool:
    """Mirror the runner's check: compare normalized file-type names.

    Absent capability means the backend has no such default, which is False.
    """

    raw = backend_capabilities(backend)["GGUF_FP16_RECURRENT_STATE_DEFAULT_FILE_TYPES"]
    if raw == "<missing>":
        return False
    return str(file_type_name).strip().lower() in quoted_members(raw)


def backend_capabilities(backend: str) -> dict[str, str]:
    source = (REPO_ROOT / "hipengine" / "kernels" / backend / "__init__.py").read_text()
    caps: dict[str, str] = {}
    for name in CAPABILITY_NAMES:
        match = re.search(rf"^{name} = (.+)$", source, re.M)
        caps[name] = match.group(1).strip() if match else "<missing>"
    return caps


def read_header(path: Path) -> tuple[dict, list[GGUFTensorInfo], int, int]:
    metadata: dict = {}
    raw: list[tuple[str, tuple[int, ...], int, int]] = []
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file: {path}")
        _read_scalar(handle, GGUFValueType.UINT32)
        tensor_count = int(_read_scalar(handle, GGUFValueType.UINT64))
        for _ in range(int(_read_scalar(handle, GGUFValueType.UINT64))):
            key = _read_string(handle)
            metadata[key] = _read_value(handle, GGUFValueType(_read_scalar(handle, GGUFValueType.UINT32)))
        for _ in range(tensor_count):
            try:
                name = _read_string(handle)
            except Exception:
                break
            n_dims = int(_read_scalar(handle, GGUFValueType.UINT32))
            ggml_shape = tuple(
                int(_read_scalar(handle, GGUFValueType.UINT64)) for _ in range(n_dims)
            )
            qtype = int(_read_scalar(handle, GGUFValueType.UINT32))
            offset = int(_read_scalar(handle, GGUFValueType.UINT64))
            raw.append((name, tuple(reversed(ggml_shape)), qtype, offset))
        alignment = int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT))
        data_start = _align_up(handle.tell(), alignment)

    tensors = []
    for name, shape, qtype_id, offset in raw:
        qtype = ggml_type(qtype_id)
        tensors.append(
            GGUFTensorInfo(
                name=name,
                shape=shape,
                ggml_shape=tuple(reversed(shape)),
                ggml_type=int(qtype),
                ggml_type_name=ggml_type_name(qtype),
                n_elements=int(prod(shape)),
                nbytes=nbytes_for_shape(shape, qtype),
                offset=offset,
                data_offset=data_start + offset,
                byte_shape=quant_shape_to_byte_shape(shape, qtype),
            )
        )
    return metadata, tensors, data_start, int(tensor_count)


def slot_path(name: str) -> str:
    if name == "token_embd.weight":
        return "root.token_embd"
    if name == "output.weight":
        return "root.lm_head"
    if name == "output_norm.weight":
        return "root.output_norm"
    match = re.match(r"^blk\.(\d+)\.(.+?)\.weight$", name) or re.match(r"^blk\.(\d+)\.(.+)$", name)
    return f"layers.{match.group(1)}.{match.group(2)}" if match else name


def plan(backend: str, metadata: dict, tensors: list[GGUFTensorInfo]) -> dict:
    caps = backend_capabilities(backend)
    file_type = llama_file_type_name(metadata.get("general.file_type"))
    # plan_qwen35_gguf_materialization disables decode repack when the file carries
    # raw-IQ weights, because those residents are consumed as compressed rank-3
    # blocks. Copy that rule; otherwise this table would claim repack for a file
    # that never gets it.
    raw_iq = any(t.ggml_type_name in ("IQ2_XS", "IQ3_XXS", "IQ4_XS") for t in tensors)
    repack = gguf_decode_repack_enabled(None) and not raw_iq
    qmicro_types = quoted_members(caps["GGUF_DENSE_Q4_QMICRO_T16_GATE_UP_FILE_TYPES"])
    flags = dict(
        dense_q4_t16=caps["GGUF_DENSE_Q4_T16"] == "True",
        dense_q4_qmicro_t16_gate_up=(
            caps["GGUF_DENSE_Q4_QMICRO_T16_GATE_UP"] == "True"
            and str(file_type).lower() in qmicro_types
        ),
        dense_q4_t16_attn_q_08b=caps["GGUF_DENSE_Q4_T16_ATTN_Q_08B"] == "True",
        dense_q5_t16_ssm_out=caps["GGUF_DENSE_Q5_T16_SSM_OUT"] == "True",
        dense_q5_t16_ssm_out_08b=caps["GGUF_DENSE_Q5_T16_SSM_OUT_08B"] == "True",
        dense_q5_t16_qkv=caps["GGUF_DENSE_Q5_T16_QKV"] == "True",
        dense_q5_t16_h5120=caps["GGUF_DENSE_Q5_T16_H5120"] == "True",
        dense_q6_qmicro_planar=caps["GGUF_DENSE_Q6_T16_QMICRO_PLANAR"] == "True",
    )
    routes: dict[str, Counter] = defaultdict(Counter)
    rejections: dict[str, list[str]] = defaultdict(list)
    expand_stored = expand_resident = 0.0
    for tensor in tensors:
        try:
            spec = plan_qwen35_gguf_weight_spec(slot_path(tensor.name), tensor, decode_repack=repack, **flags)
        except ValueError as error:
            routes[tensor.ggml_type_name]["rejected"] += 1
            rejections[tensor.ggml_type_name].append(f"{tensor.name} ({'x'.join(map(str, tensor.shape))})")
            continue
        kind = LAYOUT_MEANING.get(spec.layout, f"kernel:{spec.quant_key}")
        routes[tensor.ggml_type_name][kind] += 1
        if spec.layout == "dense_bf16":
            expand_stored += tensor.nbytes / GIB
            expand_resident += tensor.n_elements * 2 / GIB
    return {
        "backend": backend,
        "file_type_name": file_type,
        "decode_repack_enabled": repack,
        "package_flags": flags,
        "fp16_recurrent_state_default_on": fp16_recurrent_state_default(backend, file_type),
        "routes": {k: dict(v) for k, v in routes.items()},
        "rejections": {k: v for k, v in rejections.items()},
        "rejected_tensors": sum(c.get("rejected", 0) for c in routes.values()),
        "bf16_expand_stored_gib": round(expand_stored, 3),
        "bf16_expand_resident_gib": round(expand_resident, 3),
    }


def mapping_result(path: Path, metadata: dict, tensors: list[GGUFTensorInfo]) -> dict:
    """Validate the file against the dense-Qwen plugin's expected tensor map."""

    info = GGUFModelInfo(
        path=path.resolve(),
        version=3,
        alignment=int(metadata.get("general.alignment", GGUF_DEFAULT_ALIGNMENT)),
        metadata=metadata,
        tensors=tuple(tensors),
        tensor_data_offset=0,
    )
    validation = build_qwen35_gguf_tensor_map(info, strict=False).validation
    return {
        "present": len(validation.present),
        "missing": list(validation.missing),
        "unexpected": list(validation.unexpected),
        "shape_errors": list(validation.shape_errors),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF files to inspect")
    parser.add_argument("--json", type=Path, help="write the full result here")
    parser.add_argument("--check-constants", action="store_true", help="fail if a package capability constant vanished")
    args = parser.parse_args(argv)

    # A capability that only one backend qualifies is legitimately absent on the
    # other: backend_package_capability() returns the caller's default. A rename
    # removes the name from every backend, so that is what this gate catches.
    if args.check_constants:
        caps = {b: backend_capabilities(b) for b in ("hip_gfx1100", "hip_gfx1151")}
        gone = [n for n in CAPABILITY_NAMES if all(caps[b][n] == "<missing>" for b in caps)]
        if gone:
            print(f"capability constants not defined by any backend: {gone}", file=sys.stderr)
            return 2

    report: dict = {"files": []}
    for path in args.paths:
        metadata, tensors, _, declared = read_header(path)
        types = Counter(t.ggml_type_name for t in tensors)
        entry = {
            "file": str(path),
            "file_size_on_disk_bytes": path.stat().st_size,
            "tensors_parsed": len(tensors),
            "tensors_declared_in_header": declared,
            "tensor_table_complete": len(tensors) == declared,
            "file_type": metadata.get("general.file_type"),
            "file_type_name": llama_file_type_name(metadata.get("general.file_type")),
            "dtype_histogram": dict(types.most_common()),
            "stored_weight_gib": round(sum(t.nbytes for t in tensors) / GIB, 3),
            "backends": [plan(b, metadata, tensors) for b in ("hip_gfx1100", "hip_gfx1151")],
            "plugin_tensor_map": mapping_result(path, metadata, tensors),
        }
        report["files"].append(entry)
        print(f"\n=== {path.name}: file_type {entry['file_type']} = {entry['file_type_name']}")
        print(f"    dtype histogram: {entry['dtype_histogram']}")
        for backend in entry["backends"]:
            print(
                f"    {backend['backend']}: repack={'on' if backend['decode_repack_enabled'] else 'OFF'}"
                f" fp16_recurrent_state={'on' if backend['fp16_recurrent_state_default_on'] else 'off'}"
                f" rejected={backend['rejected_tensors']}"
                f" bf16_expand={backend['bf16_expand_stored_gib']} -> {backend['bf16_expand_resident_gib']} GiB"
            )
            for qtype, kinds in sorted(backend["routes"].items(), key=lambda kv: -sum(kv[1].values())):
                print(f"      {qtype:<8} x{sum(kinds.values()):<4} {kinds}")
                for example in backend["rejections"].get(qtype, [])[:2]:
                    print(f"               rejected: {example}")
        map_result = entry["plugin_tensor_map"]
        print(
            f"    plugin tensor map: present={map_result['present']} missing={len(map_result['missing'])}"
            f" unexpected={len(map_result['unexpected'])} shape_errors={len(map_result['shape_errors'])}"
        )

    if args.json:
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
