"""Generate the committed IQ3_XXS/IQ4_XS/Q3_K dequant oracle fixture.

Builds a tiny C shim around llama.cpp ``ggml/src/ggml-quants.c`` (the exact
``dequantize_row_iq3_xxs`` / ``dequantize_row_iq4_xs`` / ``dequantize_row_q3_K``
reference implementations), dequantizes a few real tensor rows from the local
UD-Q3_K_M GGUF, and writes ``tests/fixtures/gguf/q3km_iq_dequant_oracle.json``.

The committed fixture lets ``tests/test_gguf_iq_dequant_oracle.py`` validate
hipEngine's NumPy dequantizers against llama.cpp bit-exactly without needing a
C toolchain or the llama.cpp checkout at test time. Re-run this script when
the reference model or ggml layout understanding changes:

    python3 scripts/gguf_q3km_dequant_oracle_fixture.py

Requires: a C compiler and the clean llama.cpp checkout at ``--llamacpp``
(default ``/home/lhl/llama.cpp/llama.cpp-hip``), pinned to the revision recorded
by ``--expected-llamacpp-commit``.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from hipengine.loading.gguf import GGUFReader
from hipengine.quant.gguf import GGMLQuantizationType, quant_layout

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "/models/gguf/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "gguf" / "q3km_iq_dequant_oracle.json"
LLAMACPP_ORACLE_COMMIT = "1ebf790cda38d827559548f67b0469189690cc8c"
LLAMACPP_ORACLE_SOURCES = (
    "ggml/src/ggml-quants.c",
    "ggml/src/ggml-common.h",
)

_SHIM_C = r"""
#include <stdint.h>
#include "ggml-quants.h"

// Minimal stubs for ggml.c symbols referenced by ggml-quants.c quantize paths
// (never reached by the dequantize_row_* entry points below).
const char *ggml_type_name(enum ggml_type type) { (void)type; return "stub"; }
void ggml_abort(const char *file, int line, const char *fmt, ...) {
    (void)file; (void)line; (void)fmt; __builtin_trap();
}
size_t ggml_type_size(enum ggml_type type) { (void)type; return 0; }
size_t ggml_row_size(enum ggml_type type, int64_t ne) { (void)type; (void)ne; return 0; }

void oracle_dequant_iq3_xxs(const void *blocks, float *out, int64_t k) {
    dequantize_row_iq3_xxs((const block_iq3_xxs *)blocks, out, k);
}
void oracle_dequant_iq4_xs(const void *blocks, float *out, int64_t k) {
    dequantize_row_iq4_xs((const block_iq4_xs *)blocks, out, k);
}
void oracle_dequant_q3_k(const void *blocks, float *out, int64_t k) {
    dequantize_row_q3_K((const block_q3_K *)blocks, out, k);
}
"""

_ROWS_PER_TENSOR = 3


def _reference_provenance(llamacpp: Path, *, expected_commit: str) -> dict[str, object]:
    """Require and describe the exact clean llama.cpp oracle checkout."""

    try:
        commit = subprocess.run(
            ["git", "-C", str(llamacpp), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(llamacpp), "status", "--porcelain", "--", "ggml"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot inspect llama.cpp checkout at {llamacpp}: {exc}") from exc
    if commit != expected_commit:
        raise SystemExit(
            f"llama.cpp checkout revision mismatch: expected {expected_commit}, got {commit}"
        )
    if dirty:
        raise SystemExit(f"llama.cpp ggml sources are dirty at {llamacpp}:\n{dirty}")
    return {
        "checkout": str(llamacpp.resolve()),
        "commit": commit,
        "sources": list(LLAMACPP_ORACLE_SOURCES),
    }


def _build_oracle(llamacpp: Path, build_dir: Path) -> ctypes.CDLL:
    shim = build_dir / "oracle_shim.c"
    shim.write_text(_SHIM_C)
    so_path = build_dir / "libq3km_oracle.so"
    cmd = [
        os.environ.get("CC", "cc"),
        "-O2",
        "-shared",
        "-fPIC",
        f"-I{llamacpp / 'ggml' / 'include'}",
        f"-I{llamacpp / 'ggml' / 'src'}",
        str(shim),
        str(llamacpp / "ggml" / "src" / "ggml-quants.c"),
        "-lm",
        "-o",
        str(so_path),
    ]
    subprocess.run(cmd, check=True)
    return ctypes.CDLL(str(so_path))


def _oracle_rows(
    lib: ctypes.CDLL,
    symbol: str,
    row_bytes: np.ndarray,
    rows: int,
) -> np.ndarray:
    """Dequantize ``rows`` consecutive blocks with the llama.cpp reference."""

    fn = getattr(lib, symbol)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    fn.restype = None
    in_arr = np.ascontiguousarray(row_bytes, dtype=np.uint8)
    out = np.empty((rows, 256), dtype=np.float32)
    fn(in_arr.ctypes.data, out.ctypes.data, ctypes.c_int64(rows * 256))
    return out


def _tensor_fixture(
    reader: GGUFReader,
    lib: ctypes.CDLL,
    name: str,
    symbol: str,
    qtype: GGMLQuantizationType,
) -> dict:
    raw = np.ascontiguousarray(reader.tensor_data(name))
    layout = quant_layout(qtype)
    assert raw.ndim == 3, f"{name}: expected rank-3 expert tensor, got {raw.shape}"
    num_experts, out_features, row_nbytes = raw.shape
    assert row_nbytes % layout.type_size == 0
    blocks_per_row = row_nbytes // layout.type_size
    assert blocks_per_row * layout.type_size == row_nbytes
    # First expert, rows 0.._ROWS_PER_TENSOR-1 (row = 256-value blocks).
    expert0 = raw[0]
    row_blocks = expert0.reshape(out_features, blocks_per_row * layout.type_size)
    picked = row_blocks[: _ROWS_PER_TENSOR]
    rows_total = picked.shape[0] * blocks_per_row
    all_blocks = picked.reshape(rows_total, layout.type_size)
    expected = _oracle_rows(lib, symbol, all_blocks, rows_total)
    expected = expected.reshape(_ROWS_PER_TENSOR, blocks_per_row * 256)
    return {
        "name": name,
        "ggml_type": qtype.name,
        "symbol": symbol,
        "shape": [int(dim) for dim in raw.shape],
        "expert": 0,
        "rows": [int(r) for r in range(_ROWS_PER_TENSOR)],
        "row_bytes_b64": base64.b64encode(picked.tobytes()).decode("ascii"),
        "expected_shape": [int(dim) for dim in expected.shape],
        "expected_f32_b64": base64.b64encode(
            np.ascontiguousarray(expected, dtype=np.float32).tobytes()
        ).decode("ascii"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--llamacpp", default="/home/lhl/llama.cpp/llama.cpp-hip")
    parser.add_argument("--expected-llamacpp-commit", default=LLAMACPP_ORACLE_COMMIT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    llamacpp = Path(args.llamacpp)
    if not (llamacpp / "ggml" / "src" / "ggml-quants.c").exists():
        raise SystemExit(f"llama.cpp checkout not found at {llamacpp}")
    provenance = _reference_provenance(
        llamacpp,
        expected_commit=args.expected_llamacpp_commit,
    )

    with tempfile.TemporaryDirectory(prefix="q3km-oracle-") as tmp:
        lib = _build_oracle(llamacpp, Path(tmp))
        reader = GGUFReader(args.model)
        tensors = [
            ("blk.0.ffn_gate_exps.weight", "oracle_dequant_iq3_xxs", GGMLQuantizationType.IQ3_XXS),
            ("blk.0.ffn_down_exps.weight", "oracle_dequant_iq4_xs", GGMLQuantizationType.IQ4_XS),
            ("blk.40.ffn_gate_exps.weight", "oracle_dequant_q3_k", GGMLQuantizationType.Q3_K),
        ]
        entries = [_tensor_fixture(reader, lib, name, symbol, qtype) for name, symbol, qtype in tensors]

    fixture = {
        "schema_version": 1,
        "description": (
            "llama.cpp ggml-quants.c dequantize_row_* oracle output for real "
            "rows of Qwen3.6-35B-A3B-UD-Q3_K_M expert tensors. Generated by "
            "scripts/gguf_q3km_dequant_oracle_fixture.py."
        ),
        "model": args.model,
        "oracle": {
            "name": "llama.cpp GGML row dequantization",
            **provenance,
        },
        "comparison": {"kind": "bit_exact", "atol": 0.0, "rtol": 0.0},
        "rows_per_tensor": _ROWS_PER_TENSOR,
        "tensors": entries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
