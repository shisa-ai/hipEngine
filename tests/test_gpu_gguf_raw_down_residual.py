"""E6c-2 raw-IQ FFN-down residual composites: exactness at the prod shape.

Every composite here must replace [parent rows-1 owner, gguf_bf16_add]
bit-for-bit at (17408 -> 5120) rows=1. The reference chain runs the same
parent kernel into a buffer, then the production add kernel, all on device;
the candidate runs the residual sibling. Both parents matter: local32
(iq4_xs / iq4_nl down slots while the dense-IQ session is bound and the
slot is unpinned) and strict (pinned slots, policy-less q3_k, and every
session-unbound launch). The final test drives the integrated
`launch_gguf_linear_residual` registry path the runner calls.
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hipengine.kernels.hip_gfx1100.quant import gguf_iq_dense

FIXTURE = Path(__file__).parent / "fixtures/gguf_ud"

PROD_K = 17_408
PROD_N = 5_120
# Per-quant raw block geometry (mirrors gguf_iq_dense.hip B and S).
_BLOCK = {
    "gguf_iq4_xs": (256, 136),
    "gguf_iq4_nl": (32, 18),
    "gguf_iq3_s": (256, 110),
    "gguf_q3_k": (256, 110),
}
_FIXTURE_TYPE = {
    "gguf_iq4_xs": "IQ4_XS",
    "gguf_iq4_nl": "IQ4_NL",
    "gguf_iq3_s": "IQ3_S",
    "gguf_q3_k": "Q3_K",
}
LOCAL32_QUANTS = ("gguf_iq4_xs", "gguf_iq4_nl")
RAW_DOWN_QUANTS = tuple(_BLOCK)


def _hip_available() -> bool:
    try:
        ctypes.CDLL("libamdhip64.so")
    except OSError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _hip_available(), reason="HIP runtime is not available"
)


def bf16(x):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((b + 0x7fff + ((b >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_f32(u):
    return (np.asarray(u, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _prod_weight(quant: str) -> np.ndarray:
    """Real fixture columns cycled to the production (17408 -> 5120) shape.

    Output columns are independent and quant blocks are independent across
    K, so cycling real blocks keeps every byte structurally real while
    reaching the down shape the fold admits.
    """

    ftype = _FIXTURE_TYPE[quant]
    block, block_bytes = _BLOCK[quant]
    target_blocks = PROD_K // block
    entries = json.loads((FIXTURE / "real_rows.json").read_text())["entries"]
    wanted = [e for e in entries if e["type"] == ftype]
    assert wanted, f"no {ftype} fixture row"
    cols = []
    with np.load(FIXTURE / "real_rows.npz") as data:
        for entry in wanted:
            raw = data[entry["key"] + "_raw"]
            blocks_in_fixture = raw.shape[1] // block_bytes
            for row in raw:
                blocks = row.reshape(blocks_in_fixture, block_bytes)
                if blocks_in_fixture < target_blocks:
                    blocks = np.resize(blocks, (target_blocks, block_bytes))
                elif blocks_in_fixture > target_blocks:
                    blocks = blocks[:target_blocks]
                cols.append(blocks.reshape(-1))
    cols = np.stack(cols)
    reps = -(-PROD_N // len(cols))
    return np.ascontiguousarray(np.tile(cols, (reps, 1))[:PROD_N])


def _inputs(quant: str):
    w = _prod_weight(quant)
    assert w.shape == (PROD_N, PROD_K // _BLOCK[quant][0] * _BLOCK[quant][1])
    rng = np.random.default_rng(0xE6C2)
    x = bf16(rng.normal(0.0, 0.1, size=(1, PROD_K)))
    residual = bf16(rng.normal(0.0, 0.1, size=(1, PROD_N)))
    return w, x, residual


class _DeviceColumnWeight:
    """Minimal raw-weight stand-in for launch_gguf_linear_residual."""

    def __init__(self, quant: str, backend: str, raw_ptr: int,
                 slot_path: str | None = None):
        self.spec = SimpleNamespace(
            quant_key=quant,
            layout="raw_gguf",
            slot_path=slot_path,
        )
        self.backend = backend
        self._raw = SimpleNamespace(tensor=SimpleNamespace(ptr=raw_ptr))

    def allocation(self, name: str):
        assert name == "raw"
        return self._raw


def _paired_run(quant: str, mode: str, *, launch_composite):
    """[parent, gguf_bf16_add] vs the candidate composite; exact compare."""

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused import gguf_bf16_add

    w, x, residual = _inputs(quant)
    control = np.zeros((1, PROD_N), dtype=np.uint16)
    candidate = np.zeros((1, PROD_N), dtype=np.uint16)
    bufs = []

    def dev(a):
        b = malloc(a.nbytes)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes)
        return b

    try:
        x_b = dev(x)
        w_b = dev(w)
        res_b = dev(residual)
        parent_b = malloc(control.nbytes)
        bufs.append(parent_b)
        control_b = malloc(control.nbytes)
        bufs.append(control_b)
        cand_b = malloc(candidate.nbytes)
        bufs.append(cand_b)
        lib = gguf_iq_dense.build_gguf_iq_dense()
        if mode == "strict":
            gguf_iq_dense.launch(
                x_b.ptr, w_b.ptr, parent_b.ptr, 1, PROD_K, PROD_N,
                quant=quant, output="bf16", library=lib,
            )
        else:
            gguf_iq_dense.launch_local32(
                x_b.ptr, w_b.ptr, parent_b.ptr, 1, PROD_K, PROD_N,
                quant=quant, library=lib,
            )
        gguf_bf16_add(res_b.ptr, parent_b.ptr, control_b.ptr, PROD_N)
        launch_composite(
            x_b.ptr, w_b.ptr, res_b.ptr, cand_b.ptr, lib=lib, weight=None
        )
        copy_device_to_host(host_array_ptr(control), control_b, control.nbytes)
        copy_device_to_host(host_array_ptr(candidate), cand_b,
                            candidate.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    assert np.isfinite(bf16_f32(candidate)).all()
    np.testing.assert_array_equal(
        candidate, control,
        err_msg=(
            f"{quant} {mode} residual composite diverged from "
            f"[parent, gguf_bf16_add]: "
            f"{int((candidate != control).sum())}/{control.size} differ"
        ),
    )


@pytest.mark.parametrize("quant", RAW_DOWN_QUANTS)
def test_strict_composite_is_bit_exact_at_prod_shape(quant):
    """Strict parent + add == gguf_iq_dense_residual, element for element."""

    def composite(x_ptr, w_ptr, res_ptr, out_ptr, *, lib, weight):
        gguf_iq_dense.launch_dense_residual(
            x_ptr, w_ptr, res_ptr, out_ptr, 1, PROD_K, PROD_N,
            quant=quant, library=lib,
        )

    _paired_run(quant, "strict", launch_composite=composite)


@pytest.mark.parametrize("quant", LOCAL32_QUANTS)
def test_local32_composite_is_bit_exact_at_prod_shape(quant):
    """Local32 parent + add == local32 residual sibling, bit for bit."""

    def composite(x_ptr, w_ptr, res_ptr, out_ptr, *, lib, weight):
        gguf_iq_dense.launch_local32_residual(
            x_ptr, w_ptr, res_ptr, out_ptr, 1, PROD_K, PROD_N,
            quant=quant, library=lib,
        )

    _paired_run(quant, "local32", launch_composite=composite)


@pytest.mark.parametrize(
    "quant,session,pinned,expect_mode",
    [
        # Session-bound + unpinned IQ4_XS down: production runs local32,
        # so the registry path must composite under the local32 parent.
        ("gguf_iq4_xs", True, False, "local32"),
        # Session unbound: production runs strict for every quant.
        ("gguf_iq4_xs", False, False, "strict"),
        ("gguf_q3_k", True, False, "strict"),  # no decode-policy entry
        ("gguf_iq3_s", True, True, "strict"),  # artifact pin keeps strict
    ],
)
def test_registry_path_folds_the_production_parent(
    quant, session, pinned, expect_mode
):
    """launch_gguf_linear_residual (the runner's call) picks the same parent
    production runs and replaces [parent, gguf_bf16_add] bit-for-bit."""

    from hipengine.core.memory import (
        copy_device_to_host,
        copy_host_to_device,
        free,
        host_array_ptr,
        malloc,
    )
    from hipengine.kernels.hip_gfx1100.fused import gguf_bf16_add
    from hipengine.kernels.hip_gfx1100.quant.gguf_iq_source_mmq_prefill import (
        iq_dense_mmq_session,
    )
    from hipengine.runtime.gguf_linear import launch_gguf_linear_residual

    w, x, residual = _inputs(quant)
    control = np.zeros((1, PROD_N), dtype=np.uint16)
    candidate = np.zeros((1, PROD_N), dtype=np.uint16)
    bufs = []

    def dev(a):
        b = malloc(a.nbytes)
        bufs.append(b)
        copy_host_to_device(b, host_array_ptr(a), a.nbytes)
        return b

    slot = "layers.14.ffn_down" if pinned else "layers.0.ffn_down"
    try:
        x_b = dev(x)
        w_b = dev(w)
        res_b = dev(residual)
        parent_b = malloc(control.nbytes)
        bufs.append(parent_b)
        control_b = malloc(control.nbytes)
        bufs.append(control_b)
        cand_b = malloc(candidate.nbytes)
        bufs.append(cand_b)
        lib = gguf_iq_dense.build_gguf_iq_dense()
        if expect_mode == "strict":
            gguf_iq_dense.launch(
                x_b.ptr, w_b.ptr, parent_b.ptr, 1, PROD_K, PROD_N,
                quant=quant, output="bf16", library=lib,
            )
        else:
            gguf_iq_dense.launch_local32(
                x_b.ptr, w_b.ptr, parent_b.ptr, 1, PROD_K, PROD_N,
                quant=quant, library=lib,
            )
        gguf_bf16_add(res_b.ptr, parent_b.ptr, control_b.ptr, PROD_N)
        weight = _DeviceColumnWeight(quant, "hip_gfx1151", w_b.ptr,
                                     slot_path=slot)
        pins = ("layers.14.ffn_down", "layers.15.ffn_down",
                "layers.17.ffn_down") if pinned else ()
        with iq_dense_mmq_session(session, decode_strict_slots=pins):
            folded = launch_gguf_linear_residual(
                weight,
                x_b.ptr,
                res_b.ptr,
                cand_b.ptr,
                1,
                PROD_K,
                PROD_N,
                registered_decode=True,
            )
        assert folded, (
            f"registry path declined the fold for {quant} "
            f"(session={session}, pinned={pinned})"
        )
        copy_device_to_host(host_array_ptr(control), control_b, control.nbytes)
        copy_device_to_host(host_array_ptr(candidate), cand_b,
                            candidate.nbytes)
    finally:
        for b in reversed(bufs):
            free(b)

    np.testing.assert_array_equal(
        candidate, control,
        err_msg=(
            f"registry fold for {quant} (session={session}, pinned={pinned}) "
            f"diverged from the {expect_mode} parent + gguf_bf16_add: "
            f"{int((candidate != control).sum())}/{control.size} differ"
        ),
    )