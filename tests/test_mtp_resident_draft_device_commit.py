from __future__ import annotations

import ctypes
from types import SimpleNamespace

import numpy as np

from hipengine.core.hip import HipMemcpyKind
from hipengine.core.memory import DeviceBuffer
from hipengine.speculative import mtp_resident_draft as resident_draft_mod
from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner


def test_write_kv_rows_from_device_seed_base_uses_d2d_hidden_rows(monkeypatch) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.memcpy_calls = []

        def memcpy(self, dst, src, nbytes, kind) -> None:
            self.memcpy_calls.append((int(dst), int(src), int(nbytes), int(kind)))

    runtime = Runtime()
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = runtime
    runner.hidden_size = 4
    runner.token_embd_f32 = np.arange(40, dtype=np.float32).reshape(10, 4)
    runner.seed_a = DeviceBuffer(0x1000, 16)
    runner.token_embed = DeviceBuffer(0x2000, 16)

    writes = []

    def write_one_kv(**kwargs) -> None:
        writes.append(
            (
                int(kwargs["dense_cache_len"]),
                np.asarray(kwargs["cos"]).copy(),
                np.asarray(kwargs["sin"]).copy(),
            )
        )

    runner._write_one_kv = write_one_kv

    result_len = runner.write_kv_rows_from_device_seed_base(
        0x5000,
        np.asarray([2, 3], dtype=np.int64),
        positions=np.asarray([1, 2], dtype=np.int64),
        rope_cos=np.arange(12, dtype=np.float32).reshape(3, 4),
        rope_sin=np.arange(12, 24, dtype=np.float32).reshape(3, 4),
        dense_key_cache=DeviceBuffer(0x6000, 128),
        dense_value_cache=DeviceBuffer(0x7000, 128),
        dense_cache_len=7,
    )

    assert result_len == 9
    d2d_calls = [
        call for call in runtime.memcpy_calls
        if call[3] == int(HipMemcpyKind.DEVICE_TO_DEVICE)
    ]
    assert d2d_calls == [
        (0x1000, 0x5000, 16, int(HipMemcpyKind.DEVICE_TO_DEVICE)),
        (0x1000, 0x5010, 16, int(HipMemcpyKind.DEVICE_TO_DEVICE)),
    ]
    assert [item[0] for item in writes] == [7, 8]
    np.testing.assert_array_equal(writes[0][1], np.asarray([[4, 5, 6, 7]], dtype=np.float32))
    np.testing.assert_array_equal(writes[1][1], np.asarray([[8, 9, 10, 11]], dtype=np.float32))
    np.testing.assert_array_equal(writes[0][2], np.asarray([[16, 17, 18, 19]], dtype=np.float32))
    np.testing.assert_array_equal(writes[1][2], np.asarray([[20, 21, 22, 23]], dtype=np.float32))


def test_record_top1_probs_resets_and_records_resident_draft_confidence(monkeypatch) -> None:
    monkeypatch.setattr(resident_draft_mod, "copy_host_to_device", lambda *args, **kwargs: None)

    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = None
    runner.hidden_size = 4
    runner.experts_used = 8
    runner._device_chain_enabled = False
    runner._draft_chain_cap = 16
    runner.token_embd_f32 = np.arange(40, dtype=np.float32).reshape(10, 4)
    runner.seed_a = DeviceBuffer(0x1000, 16)
    runner.seed_b = DeviceBuffer(0x1100, 16)
    runner.token_embed = DeviceBuffer(0x2000, 16)
    runner.cos = DeviceBuffer(0x3000, 16)
    runner.sin = DeviceBuffer(0x4000, 16)
    runner.position_i64 = DeviceBuffer(0x5000, 8)
    runner.context_i64 = DeviceBuffer(0x6000, 8)
    runner.last_top1_probs = [9.9]

    run_calls = []
    prob_rows = iter([([2, 1], 0.75), ([3, 2], 0.25)])

    def run_one(*args, **kwargs) -> None:
        run_calls.append((args, kwargs))

    runner._run_one = run_one
    runner._read_topk_with_prob = lambda top_k: next(prob_rows)

    tokens, topk_rows, cache_len = runner._propose_chain_from_seed_buffer(
        start_token=1,
        start_position=2,
        draft_n_max=2,
        top_k=2,
        rope_cos=np.ones((8, 4), dtype=np.float32),
        rope_sin=np.zeros((8, 4), dtype=np.float32),
        dense_key_cache=None,
        dense_value_cache=None,
        dense_cache_len=7,
        draft_p_min=0.0,
        record_top1_probs=True,
    )

    assert tokens == [2, 3]
    assert topk_rows == [[2, 1], [3, 2]]
    assert cache_len == 7
    assert len(run_calls) == 2
    assert runner.last_top1_probs == [0.75, 0.25]


def test_ensure_device_chain_ready_preloads_embed_table() -> None:
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    calls = []
    runner._ensure_embed_table = lambda: calls.append("ensure")

    runner.ensure_device_chain_ready()

    assert calls == ["ensure"]


def test_draft_dense_q8_dp4a_helper_is_default_off(monkeypatch) -> None:
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner._draft_dense_q8_dp4a_enabled = False
    runner.dense_q8_1 = DeviceBuffer(0x4000, 36 * 64)

    monkeypatch.setattr(
        resident_draft_mod,
        "gguf_q4_k_quantize_f32_q8_1",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("quantize should not run")),
    )
    monkeypatch.setattr(
        resident_draft_mod,
        "gguf_q8_0_dp4a_gemv_f32_f32_out",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("dp4a should not run")),
    )

    assert not runner._try_dense_q8_dp4a_f32(
        0x1000,
        0x2000,
        0x3000,
        rows=1,
        in_features=2048,
        out_features=2048,
    )


def test_draft_dense_q8_dp4a_single_helper_quantizes_once(monkeypatch) -> None:
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner._draft_dense_q8_dp4a_enabled = True
    runner.dense_q8_1 = DeviceBuffer(0x4000, 36 * 64)
    runner.runtime = SimpleNamespace()
    runner._q4_lib = object()
    runner._q8_dp4a_lib = object()
    calls: list[tuple[str, tuple, dict]] = []

    def quantize(*args, **kwargs) -> None:
        calls.append(("quantize", args, kwargs))

    def dp4a(*args, **kwargs) -> None:
        calls.append(("dp4a", args, kwargs))

    monkeypatch.setattr(resident_draft_mod, "gguf_q4_k_quantize_f32_q8_1", quantize)
    monkeypatch.setattr(resident_draft_mod, "gguf_q8_0_dp4a_gemv_f32_f32_out", dp4a)

    assert runner._try_dense_q8_dp4a_f32(
        0x1000,
        0x2000,
        0x3000,
        rows=1,
        in_features=2048,
        out_features=4096,
    )

    assert [name for name, _args, _kwargs in calls] == ["quantize", "dp4a"]
    assert calls[0][1][:4] == (0x1000, 0x4000, 1, 2048)
    assert calls[1][1][:6] == (0x4000, 0x2000, 0x3000, 1, 2048, 4096)


def test_draft_dense_q8_dp4a_split_helpers_quantize_once(monkeypatch) -> None:
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner._draft_dense_q8_dp4a_enabled = True
    runner.dense_q8_1 = DeviceBuffer(0x4000, 36 * 64)
    runner.runtime = SimpleNamespace()
    runner._q4_lib = object()
    runner._q8_dp4a_lib = object()
    calls: list[tuple[str, tuple, dict]] = []

    def quantize(*args, **kwargs) -> None:
        calls.append(("quantize", args, kwargs))

    def dual(*args, **kwargs) -> None:
        calls.append(("dual", args, kwargs))

    def triple(*args, **kwargs) -> None:
        calls.append(("triple", args, kwargs))

    monkeypatch.setattr(resident_draft_mod, "gguf_q4_k_quantize_f32_q8_1", quantize)
    monkeypatch.setattr(resident_draft_mod, "gguf_q8_0_dp4a_dual_split_rowtile4_gemv_f32_f32_out", dual)
    monkeypatch.setattr(resident_draft_mod, "gguf_q8_0_dp4a_triple_split_rowtile4_gemv_f32_f32_out", triple)

    assert runner._try_dense_q8_dp4a_dual_f32(
        0x1000,
        0x2100,
        0x2200,
        0x3100,
        0x3200,
        rows=1,
        in_features=2048,
        out_features_a=768,
        out_features_b=768,
    )
    assert runner._try_dense_q8_dp4a_triple_f32(
        0x1000,
        0x2100,
        0x2200,
        0x2300,
        0x3100,
        0x3200,
        0x3300,
        rows=1,
        in_features=2048,
        out_features_a=4096,
        out_features_b=512,
        out_features_c=512,
    )

    assert [name for name, _args, _kwargs in calls] == ["quantize", "dual", "quantize", "triple"]
    assert calls[0][1][:4] == (0x1000, 0x4000, 1, 2048)
    assert calls[1][1][:9] == (0x4000, 0x2100, 0x2200, 0x3100, 0x3200, 1, 2048, 768, 768)
    assert calls[2][1][:4] == (0x1000, 0x4000, 1, 2048)
    assert calls[3][1][:11] == (
        0x4000,
        0x2100,
        0x2200,
        0x2300,
        0x3100,
        0x3200,
        0x3300,
        1,
        2048,
        4096,
        512,
    )


def test_device_chain_stage_timings_split_drain_and_d2h(monkeypatch) -> None:
    class Runtime:
        def __init__(self) -> None:
            self.syncs = 0

        def device_synchronize(self) -> None:
            self.syncs += 1

    def fake_copy_device_to_host(dst, src, nbytes, *, runtime=None) -> None:
        out = (ctypes.c_int32 * (int(nbytes) // 4)).from_address(int(dst))
        for index in range(len(out)):
            out[index] = 3 + index

    monkeypatch.setattr(resident_draft_mod, "copy_host_to_device", lambda *args, **kwargs: None)
    monkeypatch.setattr(resident_draft_mod, "copy_device_to_host", fake_copy_device_to_host)

    runtime = Runtime()
    runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
    runner.runtime = runtime
    runner.hidden_size = 4
    runner.qk_head_dim = 2
    runner.vocab = 8
    runner.token_embd_f32 = np.arange(32, dtype=np.float32).reshape(8, 4)
    runner.token_embed = DeviceBuffer(0x1000, 16)
    runner.cos_all = DeviceBuffer(0x2000, 32)
    runner.sin_all = DeviceBuffer(0x3000, 32)
    runner.pos_all = DeviceBuffer(0x4000, 16)
    runner.ctx_all = DeviceBuffer(0x5000, 16)
    runner.topk_all = DeviceBuffer(0x6000, 16)
    runner._embed_table_f32 = DeviceBuffer(0x7000, 128)
    runner._ensure_embed_table = lambda: None
    runner._run_one = lambda *args, **kwargs: None
    runner._topk_indices_into = lambda *args, **kwargs: None

    stage_timings: dict[str, float] = {}
    tokens, topk_rows, cache_len = runner._propose_chain_device(
        current_seed=DeviceBuffer(0x8000, 16),
        next_seed=DeviceBuffer(0x9000, 16),
        start_token=1,
        start_position=0,
        draft_n_max=1,
        top_k=1,
        rope_cos=np.ones((4, 2), dtype=np.float32),
        rope_sin=np.zeros((4, 2), dtype=np.float32),
        dense_key_cache=None,
        dense_value_cache=None,
        dense_cache_len=5,
        stage_timings=stage_timings,
    )

    assert tokens == [3]
    assert topk_rows == [[3]]
    assert cache_len == 5
    assert runtime.syncs == 1
    assert "draft_device_chain_drain" in stage_timings
    assert "draft_topk_d2h" in stage_timings
    assert "draft_topk_readback" in stage_timings
