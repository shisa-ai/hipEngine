"""Unit contracts for the extracted GGUF decode-batch attention owner."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hipengine.runtime import qwen35_gguf_full_attention_decode_batch as decode_batch
from hipengine.runtime import qwen35_gguf_runner as runner_mod
from hipengine.runtime.qwen35_gguf_layer_attn import AttentionGeometry


def _ptr(name: str) -> int:
    return 1000 + sum(ord(ch) for ch in name)


def _buf(name: str) -> NS:
    return NS(ptr=_ptr(name))


class _Layer:
    def weight(self, name: str) -> NS:
        return NS(allocation=lambda: NS(tensor=_buf(f"w_{name}")))


class _Runtime:
    def __init__(self, calls: list[tuple]) -> None:
        self.calls = calls

    def device_synchronize(self) -> None:
        self.calls.append(("sync",))


class _Harness:
    """Direct owner harness with fake kernels and explicit selected operations."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        rows: int = 3,
        max_context_len: int = 1023,
        retained: bool = False,
        direct: bool = False,
        int8_kv_value_bf16: bool = True,
        workspace: bool = True,
        context_available: bool = True,
        split_available: bool = True,
        f32_ok: bool = True,
        bf16_ok: bool = False,
        triple_ok: bool = True,
        pair_ok: bool = True,
        metadata: bool = True,
        decode_metadata: bool | None = None,
    ) -> None:
        self.calls: list[tuple] = []
        self.rows = rows
        self.runtime = _Runtime(self.calls)
        self.layer = _Layer()
        self.cfg = NS(
            rms_norm_eps=1e-6,
            head_count=4,
            head_count_kv=2,
            key_length=8,
            rope_dimension_count=8,
        )
        self.geometry = AttentionGeometry(
            hidden_size=16,
            q_width=32,
            kv_width=16,
        )
        self.f32_ok = f32_ok
        self.bf16_ok = bf16_ok
        self.triple_ok = triple_ok
        self.pair_ok = pair_ok
        self.attn_norm_f32_ptr = _buf("norm_f32").ptr
        self.workspace = (
            NS(
                rows=rows,
                num_splits=max(1, (max_context_len + 3) // 4),
                partial_out=_buf("partial_out"),
                partial_m=_buf("partial_m"),
                partial_l=_buf("partial_l"),
            )
            if workspace
            else None
        )
        scale_metadata = (
            NS(k_scale=_buf("k_scale"), v_scale=_buf("v_scale"))
            if metadata
            else None
        )
        decode_scale_metadata = (
            scale_metadata
            if decode_metadata is None
            else (
                NS(k_scale=_buf("decode_k_scale"), v_scale=_buf("decode_v_scale"))
                if decode_metadata
                else None
            )
        )
        self.retained_metadata = scale_metadata
        self.views = decode_batch.DecodeBatchAttentionViews(
            norm=_buf("norm"),
            post_norm_f32=_buf("post_norm_f32"),
            full_q=_buf("full_q"),
            full_k=_buf("full_k"),
            full_v=_buf("full_v"),
            full_query_raw=_buf("full_query_raw"),
            full_key_raw=_buf("full_key_raw"),
            full_query=_buf("full_query"),
            full_key=_buf("full_key"),
            full_gate=_buf("full_gate"),
            full_gated=_buf("full_gated"),
            positions_tensor=_buf("positions"),
            cos_table=_buf("cos"),
            sin_table=_buf("sin"),
            key_cache=_buf("key_cache"),
            value_cache=_buf("value_cache"),
            append_spans=_buf("append_spans"),
            prefill_spans=NS(max_live_count=max_context_len),
            retained_key_cache=_buf("retained_key") if retained else None,
            retained_value_cache=_buf("retained_value") if retained else None,
            retained_append_spans=(
                NS(scale_metadata=scale_metadata) if retained else None
            ),
            retained_decode_spans=(
                NS(scale_metadata=decode_scale_metadata) if direct else None
            ),
            retained_decode_kernel=self._direct_kernel if direct else None,
            int8_kv_value_bf16=int8_kv_value_bf16,
            block_size=4,
            max_positions=64,
        )
        self.context_batch = self._context_batch if context_available else None
        self.split_batch = self._split_batch if split_available else None
        self.selectors = decode_batch.DecodeBatchAttentionSelectors(
            cast_library=lambda: "cast",
            paged_kv_write_library=lambda: "kv",
            paged_attn_decode_library=lambda: "attn",
            run_attention_norm_rows=self._attention_norm,
            try_launch_dense_q8_triple_dp4a_f32=self._q8_f32,
            try_launch_dense_q8_triple_dp4a=self._q8_bf16,
            int8_kv_prompt_write_fn=self._int8_writer,
            paged_attn_context_batch=lambda: self.context_batch,
            packed_ar_attention_batch_kernel=lambda: self.split_batch,
            split_workspace=self._split_workspace_view,
        )
        monkeypatch.setattr(decode_batch, "qwen35_split_qgate_bf16", self._split_qgate)
        monkeypatch.setattr(decode_batch, "bf16_to_f32", self._bf16_to_f32)
        monkeypatch.setattr(
            decode_batch,
            "gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight",
            self._head_norm,
        )
        monkeypatch.setattr(
            decode_batch,
            "qwen35_write_paged_kv_mixed_value_bf16_prompt_spans",
            self._mixed_write,
        )
        monkeypatch.setattr(
            decode_batch,
            "qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans",
            self._int8_bf16_write,
        )
        monkeypatch.setattr(
            decode_batch, "qwen35_full_attn_gate_mul_bf16", self._gate_mul
        )
        monkeypatch.setattr(decode_batch, "launch_gguf_linear_triple", self._triple)
        monkeypatch.setattr(decode_batch, "launch_gguf_linear_pair", self._pair)
        monkeypatch.setattr(decode_batch, "launch_gguf_linear", self._single)

    def _attention_norm(self, **kwargs):
        self.calls.append(("attention_norm", kwargs))
        return self.attn_norm_f32_ptr

    def _q8_f32(self, *args, **kwargs):
        self.calls.append(("q8_f32", args, kwargs))
        return self.f32_ok

    def _q8_bf16(self, *args, **kwargs):
        self.calls.append(("q8_bf16", args, kwargs))
        return self.bf16_ok

    def _split_workspace_view(self, required_rows, required_splits):
        self.calls.append(("split_workspace", required_rows, required_splits))
        workspace = self.workspace
        if workspace is None:
            return None
        return decode_batch.DecodeBatchSplitWorkspace(
            rows=int(workspace.rows),
            num_splits=int(workspace.num_splits),
            partial_out=workspace.partial_out,
            partial_m=workspace.partial_m,
            partial_l=workspace.partial_l,
        )

    def _context_batch(self, *args, **kwargs):
        self.calls.append(("context_batch", args, kwargs))

    def _split_batch(self, *args, **kwargs):
        self.calls.append(("split_batch", args, kwargs))

    def _direct_kernel(self, *args, **kwargs):
        self.calls.append(("direct_kernel", args, kwargs))

    def _int8_writer(self, metadata):
        self.calls.append(("int8_writer", metadata))
        return self._int8_write

    def _int8_write(self, *args, **kwargs):
        self.calls.append(("int8_write", args, kwargs))

    def _int8_bf16_write(self, *args, **kwargs):
        self.calls.append(("int8_bf16_write", args, kwargs))

    def _mixed_write(self, *args, **kwargs):
        self.calls.append(("mixed_write", args, kwargs))

    def _bf16_to_f32(self, *args, **kwargs):
        self.calls.append(("bf16_to_f32", args, kwargs))

    def _split_qgate(self, *args, **kwargs):
        self.calls.append(("split_qgate", args, kwargs))

    def _head_norm(self, *args, **kwargs):
        self.calls.append(("head_norm", args, kwargs))

    def _gate_mul(self, *args, **kwargs):
        self.calls.append(("gate_mul", args, kwargs))

    def _triple(self, *args, **kwargs):
        self.calls.append(("triple", args, kwargs))
        return self.triple_ok

    def _pair(self, *args, **kwargs):
        self.calls.append(("pair", args, kwargs))
        return self.pair_ok

    def _single(self, *args, **kwargs):
        self.calls.append(("single", args, kwargs))

    def call(self, *, kv_write_only: bool = False, sync: bool = False, hidden_f32_ptr=None):
        self.stage_timings = {} if sync else None
        return decode_batch.run_full_attention_decode_batch_attn_rows(
            self.layer,
            _ptr("hidden"),
            self.views,
            cfg=self.cfg,
            geometry=self.geometry,
            runtime=self.runtime,
            rows=self.rows,
            start=0,
            selectors=self.selectors,
            stream=7,
            hidden_f32_ptr=hidden_f32_ptr,
            stage_timings=self.stage_timings,
            sync_stage_timings=sync,
            stage_prefix="p",
            kv_write_only=kv_write_only,
        )

    def names(self) -> list[str]:
        return [call[0] for call in self.calls]


@pytest.mark.parametrize("max_context_len", [1023, 777])
def test_short_context_neighbors_and_unrelated_lengths(monkeypatch, max_context_len):
    harness = _Harness(monkeypatch, max_context_len=max_context_len)

    result = harness.call()

    assert result.attention_route == "kv_live_spans_batch"
    assert result.kv_write_only is False
    assert "context_batch" in harness.names()
    assert "gate_mul" in harness.names()
    assert "split_batch" not in harness.names()
    assert "split_workspace" not in harness.names()


@pytest.mark.parametrize("max_context_len", [1024, 1025, 1500])
def test_long_context_neighbors_and_unrelated_lengths(monkeypatch, max_context_len):
    harness = _Harness(monkeypatch, max_context_len=max_context_len)

    result = harness.call()

    assert result.attention_route == "kv_live_spans_batch"
    assert "split_batch" in harness.names()
    assert "context_batch" not in harness.names()
    assert "gate_mul" not in harness.names()
    required = [
        call for call in harness.calls if call[0] == "split_workspace"
    ]
    assert required == [("split_workspace", 3, (max_context_len + 3) // 4)]


def test_kv_write_only_stops_before_attention(monkeypatch):
    harness = _Harness(monkeypatch)

    result = harness.call(kv_write_only=True)

    assert result.kv_write_only is True
    assert result.attention_route == "kv_write_only"
    assert "mixed_write" in harness.names()
    assert "context_batch" not in harness.names()
    assert "split_batch" not in harness.names()
    assert "gate_mul" not in harness.names()


def test_direct_retained_batch_route(monkeypatch):
    harness = _Harness(
        monkeypatch,
        max_context_len=1023,
        retained=True,
        direct=True,
    )

    result = harness.call()

    assert result.attention_route == "kv_live_spans_int8_batch"
    assert "direct_kernel" in harness.names()
    assert "context_batch" not in harness.names()
    assert "gate_mul" not in harness.names()
    assert "mixed_write" not in harness.names()


@pytest.mark.parametrize(
    "f32_ok,bf16_ok,triple_ok,pair_ok,expected",
    [
        (True, False, True, True, ["q8_f32"]),
        (False, True, True, True, ["q8_f32", "q8_bf16"]),
        (False, False, True, True, ["q8_f32", "q8_bf16", "triple"]),
        (False, False, False, True, ["q8_f32", "q8_bf16", "triple", "single", "pair"]),
        (
            False,
            False,
            False,
            False,
            ["q8_f32", "q8_bf16", "triple", "single", "pair", "single", "single"],
        ),
    ],
)
def test_qkv_route_priority(
    monkeypatch, f32_ok, bf16_ok, triple_ok, pair_ok, expected
):
    harness = _Harness(
        monkeypatch,
        f32_ok=f32_ok,
        bf16_ok=bf16_ok,
        triple_ok=triple_ok,
        pair_ok=pair_ok,
    )

    harness.call()

    actual = [
        name
        for name in harness.names()
        if name in {"q8_f32", "q8_bf16", "triple", "pair", "single"}
    ]
    assert actual == expected


@pytest.mark.parametrize("int8_kv_value_bf16", [True, False])
def test_retained_kv_write_order(monkeypatch, int8_kv_value_bf16):
    harness = _Harness(
        monkeypatch,
        retained=True,
        int8_kv_value_bf16=int8_kv_value_bf16,
    )

    harness.call()

    names = harness.names()
    assert names.index("bf16_to_f32") < names.index(
        "int8_bf16_write" if int8_kv_value_bf16 else "int8_write"
    )
    assert names.index(
        "int8_bf16_write" if int8_kv_value_bf16 else "int8_write"
    ) < names.index("mixed_write")


def test_stage_timing_order_and_sync(monkeypatch):
    harness = _Harness(monkeypatch, max_context_len=1023)

    result = harness.call(sync=True)

    assert result.t_stage > 0
    assert [name for name in harness.names() if name == "sync"] == ["sync"] * 4
    assert list(harness.stage_timings) == [
        "p_norm_qkv_split",
        "p_head_norm_rope",
        "p_kv_write",
        "p_paged_attn",
    ]


def test_split_workspace_callback_rereads_replacement(monkeypatch):
    harness = _Harness(monkeypatch, max_context_len=1024)
    assert harness.workspace is not None
    replacement_out = _buf("replacement_out")
    replacement_m = _buf("replacement_m")
    replacement_l = _buf("replacement_l")
    harness.workspace.rows = 5
    harness.workspace.num_splits = 300
    harness.workspace.partial_out = replacement_out
    harness.workspace.partial_m = replacement_m
    harness.workspace.partial_l = replacement_l

    harness.call()

    split = [call for call in harness.calls if call[0] == "split_batch"][0]
    args = split[1]
    assert args[5] == replacement_out.ptr
    assert args[6] == replacement_m.ptr
    assert args[7] == replacement_l.ptr


@pytest.mark.parametrize(
    "kwargs,message,error",
    [
        (
            {"retained": True, "metadata": False},
            "packed INT8 KV write requires retained payload and scale metadata",
            RuntimeError,
        ),
        (
            {
                "retained": True,
                "direct": True,
                "decode_metadata": False,
            },
            "direct INT8 batch attention requires payload and scale planes",
            RuntimeError,
        ),
        (
            {"retained": True, "direct": True, "workspace": False},
            "direct INT8 batch attention requires a row-sized split-K workspace",
            NotImplementedError,
        ),
        (
            {"max_context_len": 1024, "workspace": False},
            "long-context packed AR decode requires a row-sized split-K workspace",
            NotImplementedError,
        ),
        (
            {"context_available": False},
            "paged context-batch attention kernel is unavailable",
            RuntimeError,
        ),
        (
            {"max_context_len": 1024, "split_available": False},
            "long-context packed AR BF16 batch attention kernel is unavailable",
            NotImplementedError,
        ),
    ],
)
def test_error_messages(monkeypatch, kwargs, message, error):
    harness = _Harness(monkeypatch, **kwargs)

    with pytest.raises(error, match=message):
        harness.call()


def test_owner_views_have_no_scratch_or_workspace_proxy():
    fields = decode_batch.DecodeBatchAttentionViews.__dataclass_fields__
    assert "scratch" not in fields
    assert "q8_1_workspace_owner" not in fields


def test_owner_signature_and_module_take_no_runner_session_or_scratch():
    parameters = inspect.signature(
        decode_batch.run_full_attention_decode_batch_attn_rows
    ).parameters
    assert not {"runner", "session", "scratch"} & set(parameters)
    source = Path(decode_batch.__file__).read_text()
    assert "qwen35_gguf_runner" not in source
    assert "Qwen35GGUFResidentSession" not in source


def _adapter_scratch(*, rows: int = 3, key_cache: bool = True, value_cache: bool = True):
    return NS(
        rows=rows,
        start=0,
        norm=_buf("norm"),
        post_norm_f32=_buf("post_norm_f32"),
        full_q=_buf("full_q"),
        full_k=_buf("full_k"),
        full_v=_buf("full_v"),
        full_query_raw=_buf("full_query_raw"),
        full_key_raw=_buf("full_key_raw"),
        full_query=_buf("full_query"),
        full_key=_buf("full_key"),
        full_gate=_buf("full_gate"),
        full_gated=_buf("full_gated"),
        attn_out=_buf("attn_out"),
        positions_tensor=_buf("positions"),
        cos_table=_buf("cos"),
        sin_table=_buf("sin"),
        key_cache=_buf("key_cache") if key_cache else None,
        value_cache=_buf("value_cache") if value_cache else None,
        append_spans=_buf("append_spans"),
        prefill_spans=NS(max_live_count=1023),
        block_size=4,
        max_positions=64,
    )


class _AdapterRunner:
    def __init__(self, calls: list[tuple], *, output: int = 777) -> None:
        self.calls = calls
        self.output = output
        self.hidden_size = 16
        self.q_width = 32
        self.kv_width = 16
        self.runtime = _Runtime(calls)
        self._layers: dict[int, NS] = {}
        self.weights = NS(
            config=NS(
                rms_norm_eps=1e-6,
                head_count=4,
                head_count_kv=2,
                key_length=8,
                rope_dimension_count=8,
            ),
            layer=self._layer,
        )

    def _layer(self, layer_id: int) -> NS:
        if layer_id not in self._layers:
            self._layers[layer_id] = NS(layer_id=layer_id)
        return self._layers[layer_id]

    def _run_full_attention_output_rows(self, *args, **kwargs):
        self.calls.append(("output", args, kwargs))
        return self.output

    def _run_post_attention_ffn_rows(self, *args, **kwargs):
        self.calls.append(("ffn", args, kwargs))


def test_adapter_q8_selectors_bind_scratch(monkeypatch):
    calls: list[tuple] = []

    def f32(*args, **kwargs):
        calls.append(("f32", args, kwargs))
        return True

    def bf16(*args, **kwargs):
        calls.append(("bf16", args, kwargs))
        return True

    monkeypatch.setattr(runner_mod, "_try_launch_dense_q8_triple_dp4a_f32", f32)
    monkeypatch.setattr(runner_mod, "_try_launch_dense_q8_triple_dp4a", bf16)
    scratch = NS()
    selectors = runner_mod._decode_batch_attention_selectors(NS(), scratch, None)

    assert calls == []
    assert selectors.try_launch_dense_q8_triple_dp4a_f32(
        "qa", "qb", "qc", 1, 2, 3, 4, rows=5
    )
    assert selectors.try_launch_dense_q8_triple_dp4a(
        "ba", "bb", "bc", 6, 7, 8, 9, rows=6
    )

    assert calls[0][1] == ("qa", "qb", "qc", 1, 2, 3, 4, scratch)
    assert calls[0][2] == {"rows": 5}
    assert calls[1][1] == ("ba", "bb", "bc", 6, 7, 8, 9, scratch)
    assert calls[1][2] == {"rows": 6}


def test_adapter_split_workspace_callback_rereads_replacement():
    workspace = NS(
        rows=3,
        num_splits=256,
        partial_out=_buf("old_out"),
        partial_m=_buf("old_m"),
        partial_l=_buf("old_l"),
    )
    selectors = runner_mod._decode_batch_attention_selectors(NS(), NS(), workspace)

    first = selectors.split_workspace(3, 256)
    assert first is not None
    assert first.partial_out is workspace.partial_out

    replacement_out = _buf("new_out")
    replacement_m = _buf("new_m")
    replacement_l = _buf("new_l")
    workspace.rows = 5
    workspace.num_splits = 300
    workspace.partial_out = replacement_out
    workspace.partial_m = replacement_m
    workspace.partial_l = replacement_l

    second = selectors.split_workspace(3, 256)
    assert second is not None
    assert second.rows == 5
    assert second.num_splits == 300
    assert second.partial_out is replacement_out
    assert second.partial_m is replacement_m
    assert second.partial_l is replacement_l


def test_adapter_kv_write_only_skips_output_and_ffn(monkeypatch):
    calls: list[tuple] = []
    runner = _AdapterRunner(calls)
    scratch = _adapter_scratch()

    def fake_owner(layer, hidden_ptr, views, **kwargs):
        calls.append(("owner", layer, hidden_ptr, views, kwargs))
        return decode_batch.DecodeBatchAttentionResult("kv_write_only", True, 5.0)

    monkeypatch.setattr(
        runner_mod._decode_batch_attn,
        "run_full_attention_decode_batch_attn_rows",
        fake_owner,
    )

    result = runner_mod.Qwen35GGUFFullStackRunner._run_full_attention_decode_batch_layer_rows(
        runner, 2, 111, 222, scratch, stream=7, kv_write_only=True
    )

    assert result == "kv_write_only"
    assert [call[0] for call in calls] == ["owner"]
    assert calls[0][1] is runner.weights.layer(2)
    assert calls[0][3].norm is scratch.norm
    assert calls[0][4]["rows"] == 3
    assert calls[0][4]["kv_write_only"] is True
    assert calls[0][4]["start"] == 0
    assert calls[0][4]["stage_prefix"] == "target_block_full_attn"
    assert not [call for call in calls if call[0] in {"output", "ffn", "sync"}]


def test_adapter_output_and_ffn_composition(monkeypatch):
    calls: list[tuple] = []
    runner = _AdapterRunner(calls, output=777)
    scratch = _adapter_scratch()
    stage_timings: dict[str, float] = {}
    expert_sidecar = NS(name="sidecar")
    split_workspace = NS(rows=3, num_splits=256)

    def fake_owner(layer, hidden_ptr, views, **kwargs):
        calls.append(("owner", layer, hidden_ptr, views, kwargs))
        return decode_batch.DecodeBatchAttentionResult(
            "kv_live_spans_batch", False, 42.0
        )

    monkeypatch.setattr(
        runner_mod._decode_batch_attn,
        "run_full_attention_decode_batch_attn_rows",
        fake_owner,
    )

    result = runner_mod.Qwen35GGUFFullStackRunner._run_full_attention_decode_batch_layer_rows(
        runner,
        2,
        111,
        222,
        scratch,
        stream=7,
        expert_sidecar=expert_sidecar,
        hidden_f32_ptr=999,
        out_f32_ptr=888,
        stage_timings=stage_timings,
        sync_stage_timings=True,
        stage_prefix="p",
        split_workspace=split_workspace,
    )

    assert result == "kv_live_spans_batch"
    owner = [call for call in calls if call[0] == "owner"][0]
    assert owner[4]["rows"] == 3
    assert owner[4]["start"] == 0
    assert owner[4]["hidden_f32_ptr"] == 999
    assert owner[4]["stream"] == 7
    assert owner[4]["sync_stage_timings"] is True
    assert owner[4]["kv_write_only"] is False
    assert owner[4]["stage_prefix"] == "p"
    output = [call for call in calls if call[0] == "output"][0]
    assert output[1] == (
        runner.weights.layer(2),
        scratch.full_gated.ptr,
        scratch.attn_out.ptr,
        scratch,
    )
    assert output[2]["rows"] == 3
    assert output[2]["hidden_f32_ptr"] == 999
    assert output[2]["out_f32_ptr"] == 888
    assert output[2]["stream"] == 7
    assert output[2]["runtime"] is runner.runtime
    assert list(stage_timings) == ["p_gate_output"]
    assert [call for call in calls if call[0] == "sync"]
    ffn = [call for call in calls if call[0] == "ffn"][0]
    assert ffn[1][0] == 2
    assert ffn[1][1] == 111
    assert ffn[1][3] == 222
    assert ffn[2]["attn_out_f32_ptr"] == 777
    assert ffn[2]["expert_sidecar"] is expert_sidecar
    assert ffn[2]["stage_prefix"] == "p_ffn"
    assert ffn[2]["sync_stage_timings"] is True


def test_adapter_preconditions(monkeypatch):
    calls: list[tuple] = []
    runner = _AdapterRunner(calls)

    def fail_owner(*args, **kwargs):
        raise AssertionError("owner must not be reached")

    monkeypatch.setattr(
        runner_mod._decode_batch_attn,
        "run_full_attention_decode_batch_attn_rows",
        fail_owner,
    )

    with pytest.raises(ValueError, match="rows must be positive"):
        runner_mod.Qwen35GGUFFullStackRunner._run_full_attention_decode_batch_layer_rows(
            runner, 2, 111, 222, _adapter_scratch(rows=0)
        )
    with pytest.raises(
        RuntimeError, match="cache-backed key/value buffers"
    ):
        runner_mod.Qwen35GGUFFullStackRunner._run_full_attention_decode_batch_layer_rows(
            runner, 2, 111, 222, _adapter_scratch(key_cache=False)
        )
    with pytest.raises(
        RuntimeError, match="cache-backed key/value buffers"
    ):
        runner_mod.Qwen35GGUFFullStackRunner._run_full_attention_decode_batch_layer_rows(
            runner, 2, 111, 222, _adapter_scratch(value_cache=False)
        )


def test_owner_q8_selectors_receive_no_scratch(monkeypatch):
    harness = _Harness(monkeypatch, f32_ok=False, bf16_ok=True)

    harness.call()

    q8_f32 = [call for call in harness.calls if call[0] == "q8_f32"][0]
    q8_bf16 = [call for call in harness.calls if call[0] == "q8_bf16"][0]
    assert len(q8_f32[1]) == 7
    assert len(q8_bf16[1]) == 7


def test_single_row_retained_batch_keeps_bf16_route(monkeypatch):
    harness = _Harness(
        monkeypatch,
        rows=1,
        retained=True,
        direct=True,
        max_context_len=1023,
    )

    result = harness.call()

    assert result.attention_route == "kv_live_spans_batch"
    assert "direct_kernel" not in harness.names()
    assert "mixed_write" in harness.names()
    assert "context_batch" in harness.names()
