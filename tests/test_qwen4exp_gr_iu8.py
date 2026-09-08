"""Focused tests for the R5 GR up iu8-WMMA route (T1 candidate).

Covers the wrapper geometry validation, the default-off routing (the fused
exact parent stays the production path without the override), route
engagement under the override, and the screened three-plane drift bound
against a dequantized host reference on small random Q8_0 weights.
"""

from __future__ import annotations

import ctypes
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from tests.test_qwen4_exp_pf3_moe_schedules import (
    _alloc, _download, _upload,
)

HIP_AVAILABLE = True
try:
    ctypes.CDLL("libamdhip64.so")
except OSError:
    HIP_AVAILABLE = False


def _q8_0_pack(weights: np.ndarray) -> np.ndarray:
    """Pack a float32 (out, k) matrix into Q8_0 blocks (k % 32 == 0)."""

    out_features, in_features = weights.shape
    assert in_features % 32 == 0
    blocks = in_features // 32
    raw = np.zeros((out_features, blocks, 34), dtype=np.uint8)
    q = np.zeros((out_features, in_features), dtype=np.int8)
    d = np.zeros((out_features, blocks), dtype=np.float32)
    for b in range(blocks):
        tile = weights[:, b * 32:(b + 1) * 32]
        amax = np.maximum(np.abs(tile).max(axis=1, keepdims=True), 1e-12)
        d[:, b:b + 1] = (amax / 127.0).astype(np.float32)
        qi = np.clip(np.rint(tile / (amax / 127.0)), -127, 127)
        q[:, b * 32:(b + 1) * 32] = qi.astype(np.int8)
    fp16 = d.astype(np.float16).view(np.uint16).reshape(out_features, blocks)
    raw[:, :, 0] = (fp16 & 0xFF).astype(np.uint8)
    raw[:, :, 1] = (fp16 >> 8).astype(np.uint8)
    raw[:, :, 2:] = q.reshape(out_features, in_features).view(np.uint8).reshape(
        out_features, blocks, 32)
    return raw.reshape(out_features, blocks * 34)


@unittest.skipUnless(HIP_AVAILABLE, "HIP runtime unavailable")
class Qwen4ExpGRIu8KernelTests(unittest.TestCase):
    def _environment(self):
        from hipengine.core.hip import get_hip_runtime
        from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
            build_gguf_k_gemv,
            gguf_q8_0_iu8_wmma_prefill_f32_f32,
        )
        return (get_hip_runtime(), build_gguf_k_gemv(load=True),
                gguf_q8_0_iu8_wmma_prefill_f32_f32)

    def test_wrapper_rejects_bad_geometry(self):
        from hipengine.kernels.hip_gfx1100.quant.gguf_k_gemv import (
            gguf_q8_0_iu8_wmma_prefill_f32_f32,
        )
        with self.assertRaises(ValueError):
            gguf_q8_0_iu8_wmma_prefill_f32_f32(0, 0, 0, 0, 32, 64)
        with self.assertRaises(ValueError):
            gguf_q8_0_iu8_wmma_prefill_f32_f32(0, 0, 0, 8, 33, 64)
        with self.assertRaises(ValueError):
            gguf_q8_0_iu8_wmma_prefill_f32_f32(0, 0, 0, 8, 32, 0)

    def test_projection_matches_host_reference_within_three_plane_bound(self):
        runtime, library, launch = self._environment()
        rng = np.random.default_rng(20260908)
        rows, in_features, out_features = 48, 320, 256
        weights = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
        x = rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        raw = _q8_0_pack(weights)
        dw = _upload(raw, runtime, [])
        dx = _upload(x, runtime, [])
        dout = _alloc(rows * out_features, np.float32, runtime, [])
        launch(dx.ptr, dw.ptr, dout.ptr, rows, in_features, out_features,
               library=library, runtime=runtime)
        runtime.device_synchronize()
        got = _download(dout, (rows, out_features), np.float32, runtime)

        # Host reference: exact dequantized weights times F32 activations.
        dequant = np.zeros_like(weights)
        for b in range(in_features // 32):
            dcol = np.frombuffer(
                raw[:, b * 34:b * 34 + 2].tobytes(), dtype=np.float16
            ).astype(np.float32).reshape(out_features, 1)
            qi = raw[:, b * 34 + 2:(b + 1) * 34].view(np.int8).astype(np.float32)
            dequant[:, b * 32:(b + 1) * 32] = dcol * qi
        reference = x.astype(np.float64) @ dequant.astype(np.float64).T

        diff = np.abs(got.astype(np.float64) - reference)
        scale = np.maximum(np.abs(reference), 1e-3)
        rel = diff / scale
        # Three-plane staging carries activation quantization residual down
        # to roughly the fp32 ulp scale of the accumulator.
        self.assertLess(float(np.median(rel)), 1e-6)
        self.assertLess(float(np.percentile(rel, 99)), 1e-5)
        self.assertTrue(np.isfinite(got).all())

    def test_projection_is_deterministic_across_runs(self):
        runtime, library, launch = self._environment()
        rng = np.random.default_rng(7)
        rows, in_features, out_features = 33, 320, 512
        weights = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
        x = rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        raw = _q8_0_pack(weights)
        dw = _upload(raw, runtime, [])
        dx = _upload(x, runtime, [])
        outs = []
        for _ in range(2):
            dout = _alloc(rows * out_features, np.float32, runtime, [])
            launch(dx.ptr, dw.ptr, dout.ptr, rows, in_features, out_features,
                   library=library, runtime=runtime)
            runtime.device_synchronize()
            outs.append(_download(dout, (rows, out_features), np.float32, runtime))
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_partial_row_tiles_and_unaligned_out_features(self):
        runtime, library, launch = self._environment()
        rng = np.random.default_rng(11)
        rows, in_features, out_features = 17, 320, 130  # < 128*2, odd tiles
        weights = rng.normal(0.0, 0.05, size=(out_features, in_features)).astype(np.float32)
        x = rng.normal(0.0, 0.1, size=(rows, in_features)).astype(np.float32)
        raw = _q8_0_pack(weights)
        dw = _upload(raw, runtime, [])
        dx = _upload(x, runtime, [])
        dout = _alloc(rows * out_features, np.float32, runtime, [])
        launch(dx.ptr, dw.ptr, dout.ptr, rows, in_features, out_features,
               library=library, runtime=runtime)
        runtime.device_synchronize()
        got = _download(dout, (rows, out_features), np.float32, runtime)
        dequant = np.zeros_like(weights)
        for b in range(in_features // 32):
            dcol = np.frombuffer(
                raw[:, b * 34:b * 34 + 2].tobytes(), dtype=np.float16
            ).astype(np.float32).reshape(out_features, 1)
            qi = raw[:, b * 34 + 2:(b + 1) * 34].view(np.int8).astype(np.float32)
            dequant[:, b * 32:(b + 1) * 32] = dcol * qi
        reference = x.astype(np.float64) @ dequant.astype(np.float64).T
        rel = np.abs(got.astype(np.float64) - reference) / np.maximum(
            np.abs(reference), 1e-3)
        self.assertLess(float(np.percentile(rel, 99)), 1e-5)
        self.assertTrue(np.isfinite(got).all())

    def test_route_is_default_off(self):
        from hipengine.runtime.qwen4_exp_runner import _qwen4_exp_gr_iu8_enabled
        saved = os.environ.pop("HIPENGINE_QWEN4_EXP_GR_IU8", None)
        try:
            self.assertFalse(_qwen4_exp_gr_iu8_enabled(1024))
            os.environ["HIPENGINE_QWEN4_EXP_GR_IU8"] = "1"
            self.assertTrue(_qwen4_exp_gr_iu8_enabled(1024))
            self.assertFalse(_qwen4_exp_gr_iu8_enabled(256))
            self.assertFalse(_qwen4_exp_gr_iu8_enabled(16))
        finally:
            if saved is None:
                os.environ.pop("HIPENGINE_QWEN4_EXP_GR_IU8", None)
            else:
                os.environ["HIPENGINE_QWEN4_EXP_GR_IU8"] = saved

    def test_down_route_is_default_off(self):
        from hipengine.runtime.qwen4_exp_runner import (
            _qwen4_exp_gr_iu8_down_enabled,
        )
        saved = os.environ.pop("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN", None)
        try:
            self.assertFalse(_qwen4_exp_gr_iu8_down_enabled(1024))
            os.environ["HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"] = "1"
            self.assertTrue(_qwen4_exp_gr_iu8_down_enabled(1024))
            self.assertFalse(_qwen4_exp_gr_iu8_down_enabled(256))
        finally:
            if saved is None:
                os.environ.pop("HIPENGINE_QWEN4_EXP_GR_IU8_DOWN", None)
            else:
                os.environ["HIPENGINE_QWEN4_EXP_GR_IU8_DOWN"] = saved


@unittest.skipUnless(HIP_AVAILABLE, "HIP runtime unavailable")
class Qwen4ExpQ8Iu8DenseDispatchTests(unittest.TestCase):
    FLAG = "HIPENGINE_QWEN4_EXP_Q8_IU8_WMM"

    def test_dense_dispatch_default_off_and_gated(self):
        import hipengine.runtime.gguf_linear as gl
        saved = os.environ.pop(self.FLAG, None)
        try:
            registered = gl.is_registered(
                gl.KernelKey(
                    "hip_gfx1100", "linear", "gguf_q8_0",
                    "iu8_wmma_prefill_f32_f32_out"))
            if not registered:
                self.skipTest("iu8 dense variant not registered")

            class Spec:
                layout = gl.LAYOUT_RAW_GGUF
                quant_key = "gguf_q8_0"

            class Weight:
                spec = Spec()

            for flag in ("0", "1"):
                os.environ[self.FLAG] = flag
                d = gl._q8_iu8_wmma_dispatch(
                    gl.GGUFLinearDispatch(
                        gl.KernelKey(
                            "hip_gfx1100", "linear", "gguf_q8_0",
                            "coltile8_rowbatch4_wave_scale_f32_f32_out"),
                        "raw"),
                    rows=512, in_features=2560, out_features=6144)
                expected = ("iu8_wmma_prefill_f32_f32_out"
                            if flag == "1"
                            else "coltile8_rowbatch4_wave_scale_f32_f32_out")
                self.assertEqual(d.key.variant, expected)
            os.environ[self.FLAG] = "1"
            # sub-256 rows and odd geometry keep the parent
            for rows, in_f in ((256, 2560), (16, 2560), (512, 33)):
                d = gl._q8_iu8_wmma_dispatch(
                    gl.GGUFLinearDispatch(
                        gl.KernelKey(
                            "hip_gfx1100", "linear", "gguf_q8_0",
                            "coltile8_rowbatch4_wave_scale_f32_f32_out"),
                        "raw"),
                    rows=rows, in_features=in_f, out_features=6144)
                self.assertEqual(
                    d.key.variant,
                    "coltile8_rowbatch4_wave_scale_f32_f32_out")
        finally:
            if saved is None:
                os.environ.pop(self.FLAG, None)
            else:
                os.environ[self.FLAG] = saved


if __name__ == "__main__":
    unittest.main()
