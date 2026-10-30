from __future__ import annotations

import ctypes
import hashlib

import numpy as np
import pytest

from hipengine.core.memory import DeviceBuffer
from hipengine.speculative import mtp_resident_draft as resident_draft_mod
from hipengine.speculative.mtp_resident_draft import Qwen35GGUFResidentMTPDraftRunner
from scripts.gguf_mtp_bench import (
    hidden_state_summary,
    llama_cpp_acceptance_from_target_samples,
    llama_cpp_mtp_catchup_rows,
)

_STAGE_FIELDS = (
    ("token_embed", "draft_stage_token_embed"),
    ("e_norm", "draft_stage_e_norm"),
    ("h_norm", "draft_stage_h_norm"),
    ("projected", "draft_stage_projected"),
    ("attn_normed", "draft_stage_attn_normed"),
    ("query", "draft_stage_query_rope"),
    ("key_cur", "draft_stage_key_cur_rope"),
    ("value_cur", "draft_stage_value_cur"),
    ("attn", "draft_stage_attn_pregate"),
    ("gated", "draft_stage_attn_gated"),
    ("wo_out", "draft_stage_attn_out"),
    ("attended", "draft_stage_attn_residual"),
    ("post_norm", "draft_stage_attn_post_norm"),
    ("ffn_out", "draft_stage_ffn_out"),
)


def _expected_summary(
    values,
    *,
    label: str,
    depth: int | None = None,
    token_id: int | None = None,
    position: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = np.ascontiguousarray(np.asarray(values, dtype=np.float32)).reshape(-1)
    values64 = row.astype(np.float64)
    result: dict[str, object] = {
        "label": str(label),
        "size": int(row.size),
        "sha256_16": hashlib.sha256(row.view(np.uint8)).hexdigest()[:16],
        "mean": round(float(np.mean(values64)), 8),
        "rms": round(float(np.sqrt(np.mean(values64 * values64))), 8),
        "min": round(float(np.min(values64)), 8),
        "max": round(float(np.max(values64)), 8),
        "first8": [round(float(value), 8) for value in row[:8]],
        "last8": [round(float(value), 8) for value in row[-8:]],
    }
    if depth is not None:
        result["depth"] = int(depth)
    if token_id is not None:
        result["token_id"] = int(token_id)
    if position is not None:
        result["position"] = int(position)
    if extra:
        result.update(extra)
    return result


def _cpu_attention_output(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    *,
    heads: int,
    kv_heads: int,
    head_dim: int,
) -> np.ndarray:
    kv_group = heads // kv_heads
    scale = np.float32(head_dim**-0.5)
    out = np.zeros((heads, head_dim), dtype=np.float32)
    for qh in range(heads):
        kh = qh // kv_group
        scores = np.empty(key.shape[0], dtype=np.float32)
        for row in range(key.shape[0]):
            scores[row] = np.float32(np.dot(key[row, kh], query[qh]) * scale)
        max_score = np.float32(np.max(scores))
        exp_scores = np.exp((scores - max_score).astype(np.float32)).astype(np.float32)
        exp_sum = np.float32(np.sum(exp_scores, dtype=np.float32))
        weights = exp_scores / exp_sum
        for row in range(key.shape[0]):
            out[qh] += np.float32(weights[row]) * value[row, kh]
    return out


def _expected_attention_debug(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    device_attn: np.ndarray,
    *,
    depth: int,
    token_id: int,
    position: int,
) -> dict[str, object]:
    heads, head_dim = query.shape
    cache_tokens, kv_heads, _ = key.shape
    kv_group = heads // kv_heads
    scale = np.float32(head_dim**-0.5)
    per_head: list[dict[str, object]] = []
    mean_abs_diffs: list[float] = []
    max_abs_diffs: list[float] = []
    for qh in range(heads):
        kh = qh // kv_group
        scores = np.empty(cache_tokens, dtype=np.float32)
        for row in range(cache_tokens):
            scores[row] = np.float32(np.dot(key[row, kh], query[qh]) * scale)
        max_score = np.float32(np.max(scores))
        exp_scores = np.exp((scores - max_score).astype(np.float32)).astype(np.float32)
        exp_sum = np.float32(np.sum(exp_scores, dtype=np.float32))
        weights = exp_scores / exp_sum if float(exp_sum) > 0.0 else np.zeros_like(exp_scores)
        cpu_out = np.zeros(head_dim, dtype=np.float32)
        for row in range(cache_tokens):
            cpu_out += np.float32(weights[row]) * value[row, kh]
        diff = np.abs(cpu_out - device_attn[qh])
        mean_abs = float(np.mean(diff.astype(np.float64)))
        max_abs = float(np.max(diff.astype(np.float64)))
        mean_abs_diffs.append(mean_abs)
        max_abs_diffs.append(max_abs)
        top_count = min(5, cache_tokens)
        top_rows = np.argsort(weights)[-top_count:][::-1]
        probe_rows: list[int] = []
        for row in (0, 1, 2, cache_tokens - 2, cache_tokens - 1):
            if 0 <= row < cache_tokens and row not in probe_rows:
                probe_rows.append(int(row))
        for row in top_rows:
            row_int = int(row)
            if row_int not in probe_rows and len(probe_rows) < 10:
                probe_rows.append(row_int)
        per_head.append(
            {
                "query_head": int(qh),
                "kv_head": int(kh),
                "visible_count": int(cache_tokens),
                "max_score": round(float(max_score), 8),
                "exp_sum": round(float(exp_sum), 8),
                "query_first4": [round(float(x), 8) for x in query[qh, :4]],
                "top_rows": [int(row) for row in top_rows],
                "top_scores": [round(float(scores[row]), 8) for row in top_rows],
                "top_weights": [round(float(weights[row]), 8) for row in top_rows],
                "row_probes": [
                    {
                        "row": int(row),
                        "score": round(float(scores[row]), 8),
                        "weight": round(float(weights[row]), 8),
                        "key_first4": [round(float(x), 8) for x in key[row, kh, :4]],
                        "value_first4": [round(float(x), 8) for x in value[row, kh, :4]],
                    }
                    for row in probe_rows
                ],
                "cpu_out_first4": [round(float(x), 8) for x in cpu_out[:4]],
                "device_out_first4": [round(float(x), 8) for x in device_attn[qh, :4]],
                "cpu_device_mae": round(mean_abs, 8),
                "cpu_device_max_abs": round(max_abs, 8),
            }
        )
    return {
        "label": "draft_stage_attention_debug",
        "depth": int(depth),
        "token_id": int(token_id),
        "position": int(position),
        "cache_tokens": int(cache_tokens),
        "heads": int(heads),
        "kv_heads": int(kv_heads),
        "head_dim": int(head_dim),
        "scale": round(float(scale), 8),
        "cpu_device_mae_mean": round(float(np.mean(mean_abs_diffs)), 8),
        "cpu_device_max_abs": round(float(np.max(max_abs_diffs)), 8),
        "per_head": per_head,
    }


class _HostTraceHarness:
    """Drive the host proposal loop with real host-backed device buffers."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.hidden_size = 4
        self.heads = 2
        self.kv_heads = 1
        self.head_dim = 2
        self.hd = self.heads * self.head_dim
        self.kvd = self.kv_heads * self.head_dim
        self.cache_len = 3
        self.cache_tokens = 4
        self.initial_seed = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        self._arrays: dict[str, np.ndarray] = {}
        self._buffers: dict[str, DeviceBuffer] = {}
        self._by_ptr: dict[int, np.ndarray] = {}
        self.cache_key = np.asarray(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
                [1.0, 2.0],
            ],
            dtype=np.float32,
        ).reshape(self.cache_tokens, self.kv_heads, self.head_dim)
        self.cache_value = np.asarray(
            [
                [1.0, 3.0],
                [5.0, 7.0],
                [9.0, 11.0],
                [13.0, 15.0],
            ],
            dtype=np.float32,
        ).reshape(self.cache_tokens, self.kv_heads, self.head_dim)

        self._alloc(
            "token_embd_f32",
            (
                np.arange(128 * self.hidden_size, dtype=np.float32)
                .reshape(128, self.hidden_size)
                .astype(np.float32)
                / np.float32(10.0)
            ),
        )
        self._alloc("seed_a", self.initial_seed.copy())
        self._alloc("seed_b", np.zeros(self.hidden_size, dtype=np.float32))
        self._alloc("token_embed", np.zeros(self.hidden_size, dtype=np.float32))
        self._alloc("cos", np.zeros(self.head_dim, dtype=np.float32))
        self._alloc("sin", np.zeros(self.head_dim, dtype=np.float32))
        self._alloc("position_i64", np.zeros(1, dtype=np.int64))
        self._alloc("context_i64", np.zeros(1, dtype=np.int64))
        for name, _ in _STAGE_FIELDS:
            elements = {
                "query": self.hd,
                "key_cur": self.kvd,
                "value_cur": self.kvd,
                "attn": self.hd,
                "gated": self.hd,
            }.get(name, self.hidden_size)
            self._alloc(name, np.zeros(elements, dtype=np.float32))
        self._alloc(
            "dense_key_cache",
            np.ascontiguousarray(self.cache_key).reshape(-1),
        )
        self._alloc(
            "dense_value_cache",
            np.ascontiguousarray(self.cache_value).reshape(-1),
        )

        runner = object.__new__(Qwen35GGUFResidentMTPDraftRunner)
        runner.runtime = object()
        runner.hidden_size = self.hidden_size
        runner.num_heads = self.heads
        runner.num_kv_heads = self.kv_heads
        runner.qk_head_dim = self.head_dim
        runner.experts_used = 8
        runner._device_chain_enabled = False
        runner._draft_chain_cap = 16
        runner.token_embd_f32 = self._arrays["token_embd_f32"]
        for name, buffer in self._buffers.items():
            if name != "token_embd_f32":
                setattr(runner, name, buffer)
        runner._run_one = self._run_one
        runner._read_topk = self._read_topk
        self.runner = runner
        self.stage_rows: list[dict[str, np.ndarray]] = []
        self.topk_calls = 0

        monkeypatch.setattr(
            resident_draft_mod,
            "copy_host_array_to_device",
            self._copy_host_array_to_device,
        )
        monkeypatch.setattr(
            resident_draft_mod,
            "copy_device_to_host",
            self._copy_device_to_host,
        )

    def _alloc(self, name: str, values) -> None:
        array = np.ascontiguousarray(np.asarray(values))
        self._arrays[name] = array
        buffer = DeviceBuffer(int(array.ctypes.data), int(array.nbytes))
        self._buffers[name] = buffer
        self._by_ptr[buffer.ptr] = array

    def _array_for(self, buffer: DeviceBuffer) -> np.ndarray:
        return self._by_ptr[int(buffer.ptr)]

    def _copy_host_array_to_device(
        self,
        buffer: DeviceBuffer,
        array,
        nbytes: int | None = None,
        *,
        runtime=None,
    ) -> None:
        count = int(getattr(array, "nbytes") if nbytes is None else nbytes)
        ctypes.memmove(
            int(buffer.ptr),
            int(getattr(array, "ctypes").data),
            count,
        )

    def _copy_device_to_host(
        self,
        host_ptr: int,
        buffer: DeviceBuffer,
        nbytes: int | None = None,
        *,
        runtime=None,
    ) -> None:
        count = int(buffer.nbytes if nbytes is None else nbytes)
        ctypes.memmove(int(host_ptr), int(buffer.ptr), count)

    def _read_topk(self, top_k: int) -> list[int]:
        assert int(top_k) == 1
        token = 20 + self.topk_calls
        self.topk_calls += 1
        return [token]

    def _run_one(self, current_seed, next_seed, **kwargs) -> None:
        depth = len(self.stage_rows)
        current = self._array_for(current_seed).copy()
        query = np.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=np.float32)
        attention = _cpu_attention_output(
            query,
            self.cache_key,
            self.cache_value,
            heads=self.heads,
            kv_heads=self.kv_heads,
            head_dim=self.head_dim,
        )
        # Keep one head exact and perturb the other so the trace's independent
        # CPU/device MAE and max-abs fields exercise a nonzero path.
        attention[1] += np.float32(0.25)
        values = {
            "token_embed": self._array_for(self.runner.token_embed).copy(),
            "e_norm": current + np.float32(10.0),
            "h_norm": current + np.float32(20.0),
            "projected": current + np.float32(30.0),
            "attn_normed": current + np.float32(40.0),
            "query": query.reshape(-1),
            "key_cur": np.asarray([1.0, 0.0], dtype=np.float32),
            "value_cur": np.asarray([1.0, 3.0], dtype=np.float32),
            "attn": attention.reshape(-1),
            "gated": query.reshape(-1) + np.float32(1.0),
            "wo_out": current + np.float32(50.0),
            "attended": current + np.float32(60.0),
            "post_norm": current + np.float32(70.0),
            "ffn_out": current + np.float32(80.0),
        }
        for name, value in values.items():
            destination = self._array_for(getattr(self.runner, name))
            destination[...] = np.asarray(value, dtype=np.float32).reshape(
                destination.shape
            )
        next_values = self._array_for(next_seed)
        next_values[...] = current + np.float32(100.0)
        self.stage_rows.append(
            {
                **{
                    name: self._array_for(getattr(self.runner, name)).copy()
                    for name, _ in _STAGE_FIELDS
                },
                "next_seed": next_values.copy(),
                "query_2d": query.copy(),
                "attention_2d": attention.copy(),
            }
        )

    def run(
        self,
        *,
        draft_n_max: int,
        start_token: int,
        start_position: int,
        dense_cache_len: int,
        record_stage_stats: bool,
        record_hidden_stats: bool,
        record_cache_rows: tuple[int, ...] | None = None,
        record_attention_debug: bool = False,
        dense_cache: bool = True,
        initial_seed=None,
    ):
        if initial_seed is not None:
            self.initial_seed = np.ascontiguousarray(
                np.asarray(initial_seed, dtype=np.float32)
            )
        self._array_for(self.runner.seed_a)[...] = self.initial_seed
        self._array_for(self.runner.seed_b)[...] = np.float32(0.0)
        self.stage_rows = []
        self.topk_calls = 0
        tokens, topk_rows, cache_len = self.runner._propose_chain_from_seed_buffer(
            start_token=int(start_token),
            start_position=int(start_position),
            draft_n_max=int(draft_n_max),
            top_k=1,
            rope_cos=np.ones((128, self.head_dim), dtype=np.float32),
            rope_sin=np.zeros((128, self.head_dim), dtype=np.float32),
            dense_key_cache=(
                self.runner.dense_key_cache if dense_cache else None
            ),
            dense_value_cache=(
                self.runner.dense_value_cache if dense_cache else None
            ),
            dense_cache_len=int(dense_cache_len),
            draft_p_min=0.0,
            record_hidden_stats=bool(record_hidden_stats),
            record_stage_stats=bool(record_stage_stats),
            record_cache_rows=record_cache_rows,
            record_attention_debug=bool(record_attention_debug),
        )
        return (
            tokens,
            topk_rows,
            cache_len,
            self.runner.last_hidden_state_summaries,
            list(self.stage_rows),
        )


def test_host_stage_cache_attention_trace_matches_independent_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _HostTraceHarness(monkeypatch)

    tokens, topk_rows, cache_len, summaries, stage_rows = harness.run(
        draft_n_max=1,
        start_token=7,
        start_position=11,
        dense_cache_len=harness.cache_len,
        record_stage_stats=True,
        record_hidden_stats=True,
        record_cache_rows=(0, 2, 3, 4, -1),
        record_attention_debug=True,
    )

    assert tokens == [20]
    assert topk_rows == [[20]]
    assert cache_len == harness.cache_len + 1
    assert len(summaries) == 1 + len(_STAGE_FIELDS) + 6 + 1 + 1

    expected: list[dict[str, object]] = [
        _expected_summary(
            harness.initial_seed,
            label="draft_seed_input",
            depth=-1,
            token_id=7,
            position=11,
        )
    ]
    expected.extend(
        _expected_summary(
            stage_rows[0][name],
            label=label,
            depth=0,
            token_id=7,
            position=11,
        )
        for name, label in _STAGE_FIELDS
    )
    cache_common = {"depth": 0, "token_id": 7, "position": 11, "cache_tokens": 4}
    for label, values, row in (
        ("draft_stage_dense_key_current", harness.cache_key[3], 3),
        ("draft_stage_dense_value_current", harness.cache_value[3], 3),
        ("draft_stage_dense_key_prev", harness.cache_key[2], 2),
        ("draft_stage_dense_value_prev", harness.cache_value[2], 2),
        ("draft_stage_dense_key_history", harness.cache_key[0], 0),
        ("draft_stage_dense_value_history", harness.cache_value[0], 0),
    ):
        expected.append(
            _expected_summary(
                values,
                label=label,
                extra={**cache_common, "cache_row": row},
            )
        )
    expected.append(
        _expected_attention_debug(
            stage_rows[0]["query_2d"],
            harness.cache_key,
            harness.cache_value,
            stage_rows[0]["attention_2d"],
            depth=0,
            token_id=7,
            position=11,
        )
    )
    expected.append(
        _expected_summary(
            stage_rows[0]["next_seed"],
            label="draft_next_seed",
            depth=0,
            token_id=7,
            position=11,
        )
    )

    assert summaries == expected
    attention_summary = summaries[-2]
    assert attention_summary["per_head"][0]["cpu_device_mae"] == 0.0
    assert attention_summary["per_head"][0]["cpu_device_max_abs"] == 0.0
    assert attention_summary["per_head"][1]["cpu_device_mae"] == 0.25
    assert attention_summary["per_head"][1]["cpu_device_max_abs"] == 0.25
    assert attention_summary["cpu_device_mae_mean"] == 0.125
    assert attention_summary["cpu_device_max_abs"] == 0.25


def test_host_trace_multiple_depths_and_reset_repeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _HostTraceHarness(monkeypatch)

    first = harness.run(
        draft_n_max=3,
        start_token=7,
        start_position=11,
        dense_cache_len=harness.cache_len,
        record_stage_stats=False,
        record_hidden_stats=True,
    )

    assert first[0] == [20, 21, 22]
    assert first[1] == [[20], [21], [22]]
    assert first[2] == harness.cache_len + 3
    first_expected = [
        _expected_summary(
            harness.initial_seed,
            label="draft_seed_input",
            depth=-1,
            token_id=7,
            position=11,
        ),
        _expected_summary(
            first[4][0]["next_seed"],
            label="draft_next_seed",
            depth=0,
            token_id=7,
            position=11,
        ),
        _expected_summary(
            first[4][1]["next_seed"],
            label="draft_next_seed",
            depth=1,
            token_id=20,
            position=12,
        ),
        _expected_summary(
            first[4][2]["next_seed"],
            label="draft_next_seed",
            depth=2,
            token_id=21,
            position=13,
        ),
    ]
    assert first[3] == first_expected
    first_summaries = first[3]

    second = harness.run(
        draft_n_max=2,
        start_token=8,
        start_position=21,
        dense_cache_len=10,
        record_stage_stats=False,
        record_hidden_stats=True,
    )

    assert second[0] == [20, 21]
    assert second[1] == [[20], [21]]
    assert second[2] == 12
    assert second[3] == [
        _expected_summary(
            harness.initial_seed,
            label="draft_seed_input",
            depth=-1,
            token_id=8,
            position=21,
        ),
        _expected_summary(
            second[4][0]["next_seed"],
            label="draft_next_seed",
            depth=0,
            token_id=8,
            position=21,
        ),
        _expected_summary(
            second[4][1]["next_seed"],
            label="draft_next_seed",
            depth=1,
            token_id=20,
            position=22,
        ),
    ]
    assert second[3] is not first_summaries
    assert first[3] == first_expected


def test_initial_catchup_and_rejected_tail_pending_hidden_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt_tokens = [101, 102, 103]
    prompt_hidden = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ],
        dtype=np.float32,
    )
    catchup_tokens, catchup_hidden = llama_cpp_mtp_catchup_rows(
        prompt_tokens,
        prompt_hidden,
    )

    assert catchup_tokens == prompt_tokens
    np.testing.assert_array_equal(
        catchup_hidden,
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            dtype=np.float32,
        ),
    )

    harness = _HostTraceHarness(monkeypatch)
    initial = harness.run(
        draft_n_max=1,
        start_token=101,
        start_position=0,
        dense_cache_len=0,
        record_stage_stats=False,
        record_hidden_stats=True,
        dense_cache=False,
        initial_seed=catchup_hidden[0],
    )

    assert initial[0] == [20]
    assert initial[3][0] == _expected_summary(
        catchup_hidden[0],
        label="draft_seed_input",
        depth=-1,
        token_id=101,
        position=0,
    )
    initial_summaries = initial[3]

    second = harness.run(
        draft_n_max=1,
        start_token=102,
        start_position=1,
        dense_cache_len=1,
        record_stage_stats=False,
        record_hidden_stats=True,
        initial_seed=catchup_hidden[1],
    )

    assert second[3][0] == _expected_summary(
        catchup_hidden[1],
        label="draft_seed_input",
        depth=-1,
        token_id=102,
        position=1,
    )
    assert second[3] is not initial_summaries
    assert initial[3] == [
        _expected_summary(
            catchup_hidden[0],
            label="draft_seed_input",
            depth=-1,
            token_id=101,
            position=0,
        ),
        _expected_summary(
            initial[4][0]["next_seed"],
            label="draft_next_seed",
            depth=0,
            token_id=101,
            position=0,
        ),
    ]

    repeated = harness.run(
        draft_n_max=1,
        start_token=101,
        start_position=0,
        dense_cache_len=0,
        record_stage_stats=False,
        record_hidden_stats=True,
        dense_cache=False,
        initial_seed=catchup_hidden[0],
    )

    assert repeated[3] == initial[3]
    assert repeated[3] is not initial[3]

    acceptance = llama_cpp_acceptance_from_target_samples(
        draft_tokens=[10, 11, 12],
        target_samples=[10, 99, 100],
    )
    assert acceptance["accepted_draft_tokens"] == 1
    assert acceptance["pending_hidden_row_index"] == 1
    assert acceptance["output_tokens"] == [10, 99]

    pending_row = prompt_hidden[1]
    pending_summary = hidden_state_summary(
        pending_row,
        label="target_pending_hidden_seed",
        depth=1,
        token_id=99,
        position=10,
    )
    assert pending_summary == _expected_summary(
        pending_row,
        label="target_pending_hidden_seed",
        depth=1,
        token_id=99,
        position=10,
    )
    assert pending_summary["sha256_16"] != _expected_summary(
        prompt_hidden[2],
        label="target_pending_hidden_seed",
    )["sha256_16"]

    committed = harness.run(
        draft_n_max=1,
        start_token=99,
        start_position=10,
        dense_cache_len=2,
        record_stage_stats=False,
        record_hidden_stats=True,
        initial_seed=pending_row,
    )

    assert committed[3][0] == _expected_summary(
        pending_row,
        label="draft_seed_input",
        depth=-1,
        token_id=99,
        position=10,
    )
    assert committed[3][0]["sha256_16"] == pending_summary["sha256_16"]
    committed_summaries = committed[3]

    replayed = harness.run(
        draft_n_max=1,
        start_token=10,
        start_position=9,
        dense_cache_len=0,
        record_stage_stats=False,
        record_hidden_stats=True,
        initial_seed=prompt_hidden[0],
    )

    assert replayed[3][0] == _expected_summary(
        prompt_hidden[0],
        label="draft_seed_input",
        depth=-1,
        token_id=10,
        position=9,
    )
    assert replayed[3][0]["sha256_16"] != committed[3][0]["sha256_16"]
    assert committed[3] == committed_summaries

    repeated_commit = harness.run(
        draft_n_max=1,
        start_token=99,
        start_position=10,
        dense_cache_len=2,
        record_stage_stats=False,
        record_hidden_stats=True,
        initial_seed=pending_row,
    )

    assert repeated_commit[3] == committed[3]
    assert repeated_commit[3] is not committed[3]
