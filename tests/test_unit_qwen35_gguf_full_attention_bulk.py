"""CPU regressions for the bulk prefill attention owner.

The owner is the extracted attention half of
``Qwen35GGUFFullStackRunner._run_full_attention_prefill_layer_aotriton``. These
tests pin its explicit-record boundary, branch selection, launch order and
error text without a GPU or a runner instance.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from hipengine.core.dtype import DType
from hipengine.runtime import qwen35_gguf_full_attention_bulk as bulk
from hipengine.runtime import qwen35_gguf_runner as runner_mod
from hipengine.runtime.qwen35_gguf_layer_attn import AttentionGeometry


class _Layer:
    def __init__(self) -> None:
        self._ptrs = {
            "attn_norm": 101,
            "attn_q": 102,
            "attn_k": 103,
            "attn_v": 104,
            "attn_q_norm": 105,
            "attn_k_norm": 106,
            "attn_output": 107,
        }

    def weight(self, slot: str):
        if slot not in self._ptrs:
            raise AssertionError(f"unexpected weight slot {slot!r}")
        ptr = self._ptrs[slot]
        return NS(slot=slot, allocation=lambda: NS(tensor=NS(ptr=ptr)))


class _Bridge:
    stream = 99

    def __init__(self, log: list) -> None:
        self.log = log

    def wait_for_inputs(self, runtime, stream):
        self.log.append(("bridge_wait", (), {"runtime": runtime, "stream": stream}))

    def release_output(self, runtime, stream):
        self.log.append(("bridge_release", (), {"runtime": runtime, "stream": stream}))


class _Harness:
    def __init__(
        self,
        monkeypatch,
        *,
        rows: int,
        start: int = 0,
        threshold: int | None = 4,
        allow_aotriton: bool = True,
        mode: str = "v2",
        direct: bool = False,
        granularity: str = "per_token_head",
        prefill_granularity: str | None = None,
        kernel: str = "sequential",
        retained: bool = False,
        int8_value_bf16: bool = False,
        head_major: bool = False,
        copy_kernel: bool = False,
        bridge: bool = False,
        dms: bool = False,
        missing_caches: bool = False,
        max_positions: int | None = None,
        replace_split: bool = False,
        replacement_split_count: int = 7,
        replacement_batch_rows: int | None = None,
        q8_output: bool = False,
    ) -> None:
        self.rows = rows
        self.start = start
        self.threshold = threshold
        self.allow_aotriton = allow_aotriton
        self.mode = mode
        self.kernel = kernel
        self.direct = direct
        self.retained = retained
        self.int8_value_bf16 = int8_value_bf16
        self.head_major = head_major
        self.copy_kernel = copy_kernel
        self.missing_caches = missing_caches
        self.replace_split = replace_split
        self.replacement_split_count = replacement_split_count
        self.replacement_batch_rows = replacement_batch_rows
        self.q8_output = q8_output
        self.log: list[tuple[str, tuple, dict]] = []
        self.stage_marks: list[str] = []
        self.runtime = object()
        self.layer = _Layer()
        self.max_positions = (
            max_positions if max_positions is not None else max(64, rows + 16)
        )
        self._next_ptr = 1000

        def buf(name: str) -> NS:
            self._next_ptr += 1
            return NS(ptr=self._next_ptr, name=name)

        metadata = NS(
            granularity=granularity,
            scale_dtype=DType.FP32,
            k_scale=buf("k_scale"),
            v_scale=buf("v_scale"),
        )
        prefill_metadata = metadata
        if prefill_granularity is not None:
            prefill_metadata = NS(
                granularity=prefill_granularity,
                scale_dtype=DType.FP32,
                k_scale=buf("prefill_k_scale"),
                v_scale=buf("prefill_v_scale"),
            )
        append_spans = NS(
            storage_dtype=DType.INT8_PER_TOKEN_HEAD if direct else DType.BF16,
            scale_metadata=metadata if direct else None,
        )
        prefill_spans = NS(
            storage_dtype=DType.INT8_PER_TOKEN_HEAD if direct else DType.BF16,
            scale_metadata=prefill_metadata if direct else None,
        )
        retained_spans = None
        if retained:
            retained_spans = NS(
                storage_dtype=DType.INT8_PER_TOKEN_HEAD,
                scale_metadata=NS(
                    granularity="per_token_head",
                    scale_dtype=DType.FP32,
                    k_scale=buf("retained_k_scale"),
                    v_scale=buf("retained_v_scale"),
                ),
            )
        self.metadata = metadata
        self.prefill_metadata = prefill_metadata
        self.split_partial = buf("split_partial")
        self.split_m = buf("split_m")
        self.split_l = buf("split_l")
        self.split_batch_rows = rows
        self.split_count = 4
        self.views = bulk.BulkFullAttentionViews(
            norm=buf("norm"),
            full_q=buf("full_q"),
            full_k=buf("full_k"),
            full_v=buf("full_v"),
            full_query_raw=buf("full_query_raw"),
            full_key_raw=buf("full_key_raw"),
            full_query=buf("full_query"),
            full_key=buf("full_key"),
            full_query_bf16=buf("full_query_bf16"),
            full_gate=buf("full_gate"),
            full_gated=buf("full_gated"),
            full_attn_bf16=buf("full_attn_bf16"),
            attn_out=buf("attn_out"),
            positions_tensor=buf("positions"),
            key_cache=None if missing_caches else buf("key_cache"),
            value_cache=None if missing_caches else buf("value_cache"),
            retained_key_cache=buf("retained_key_cache") if retained else None,
            retained_value_cache=buf("retained_value_cache") if retained else None,
            append_spans=append_spans,
            prefill_spans=prefill_spans,
            retained_append_spans=retained_spans,
            block_size=256,
            max_positions=self.max_positions,
            int8_kv_value_bf16=int8_value_bf16,
            cu_q=buf("cu_q"),
            cu_k=buf("cu_k"),
            softmax_lse=buf("softmax_lse"),
            atomic=buf("atomic"),
            head_major_key_cache=buf("head_major_key") if head_major else None,
            head_major_value_cache=buf("head_major_value") if head_major else None,
            head_major_kv_capacity=self.max_positions + 16 if head_major else 0,
            head_major_kv_dense_prefix=False,
            full_attn_split_batch_rows=rows,
        )
        self.bridge = _Bridge(self.log) if bridge else None
        self.dms = NS(name="dms_sink") if dms else None
        self.selectors = bulk.BulkFullAttentionSelectors(
            cast_library=lambda: "cast_library",
            paged_kv_write_library=lambda: "kv_write_library",
            paged_attn_decode_library=lambda: "attn_library",
            aotriton_prefill_library=lambda: "aotriton_library",
            int8_kv_prompt_write_fn=self._int8_writer,
            try_launch_dense_q8_single_dp4a=self._try_q8,
            native_prefill_fn=lambda: self._native,
            head_major_buffers=self._head_major_buffers,
            paged_kv_copy_resolve=self._resolve_copy,
            ensure_split_rows=self._ensure_split,
        )
        self.policy = bulk.BulkFullAttentionPolicy(
            aotriton_prefill_allowed=lambda backend: True,
            aotriton_prefill_mode=lambda start, rows, end: self.mode,
            int8_prefill_kernel=lambda **kwargs: self.kernel,
        )
        self._install(monkeypatch)

    def _record(self, label: str, returns=None):
        def call(*args, **kwargs):
            self.log.append((label, args, kwargs))
            return returns

        return call

    def _install(self, monkeypatch) -> None:
        names = (
            "gguf_rmsnorm_bf16_f32_weight",
            "qwen35_split_qgate_bf16",
            "bf16_to_f32",
            "bf16_to_fp16",
            "fp16_to_bf16",
            "f32_to_bf16",
            "gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight",
            "qwen35_write_paged_kv_mixed_value_bf16_prompt_spans",
            "qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans",
            "qwen35_full_attn_gate_mul_bf16_to_bf16",
            "qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans",
            "qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_spans",
            "qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_flash_spans",
            "qwen35_paged_attn_prefill_int8_gqa_gate_bf16_out_wmma_spans",
            "aotriton_attn_fwd_compact_varlen",
            "aotriton_attn_fwd_v3_compact_varlen",
        )
        for name in names:
            monkeypatch.setattr(bulk, name, self._record(name))
        monkeypatch.setattr(
            bulk,
            "launch_gguf_linear_triple",
            lambda *a, **k: (self.log.append(("linear_triple", a, k)), True)[1],
        )
        monkeypatch.setattr(
            bulk,
            "launch_gguf_linear_pair",
            lambda *a, **k: (self.log.append(("linear_pair", a, k)), False)[1],
        )
        monkeypatch.setattr(bulk, "launch_gguf_linear", self._record("linear"))
        for name in ("aotriton_tensor1", "aotriton_tensor2", "aotriton_tensor4"):
            monkeypatch.setattr(
                bulk,
                name,
                lambda *a, _name=name, **k: (
                    self.log.append((_name, a, k)),
                    NS(ptr=a[0], shape=a[1], strides=a[2], dtype=a[3]),
                )[1],
            )

    def _int8_writer(self, metadata):
        def write(*args, **kwargs):
            self.log.append(
                ("int8_prompt_write", (metadata.granularity,), kwargs)
            )

        return write

    def _native(self, *args, **kwargs):
        self.log.append(("native", args, kwargs))

    def _try_q8(self, weight, x_ptr, out_ptr, **kwargs):
        self.log.append(("q8_output", (weight, x_ptr, out_ptr), kwargs))
        return self.q8_output

    def _ensure_split(self, rows):
        self.log.append(("ensure_split", (rows,), {}))
        if self.replace_split:
            self.split_partial = NS(ptr=self._next_ptr + 1, name="split_partial_grown")
            self.split_m = NS(ptr=self._next_ptr + 2, name="split_m_grown")
            self.split_l = NS(ptr=self._next_ptr + 3, name="split_l_grown")
            self.split_count = self.replacement_split_count
            self.split_batch_rows = (
                rows if self.replacement_batch_rows is None else self.replacement_batch_rows
            )
        return bulk.BulkFullAttentionSplitView(
            partial=self.split_partial,
            m=self.split_m,
            l=self.split_l,
            batch_rows=self.split_batch_rows,
            count=self.split_count,
        )

    def _head_major_buffers(self, views, *, context_len):
        self.log.append(("head_major", (), {"context_len": context_len}))
        if not self.head_major:
            return None
        return (
            views.head_major_key_cache,
            views.head_major_value_cache,
            views.head_major_kv_capacity,
        )

    def _resolve_copy(self, **kwargs):
        self.log.append(("resolve_copy", (), kwargs))
        if not self.copy_kernel:
            return None

        def copy(*args, **kwargs):
            self.log.append(("copy", args, kwargs))

        return copy

    def _capture(self):
        self.log.append(("capture", (), {}))

    def call(self):
        stage = NS(mark=self.stage_marks.append)
        return bulk.run_full_attention_prefill_attn_rows(
            0,
            self.layer,
            1,
            self.views,
            cfg=NS(
                head_count=4,
                head_count_kv=2,
                key_length=16,
                rope_dimension_count=8,
                rms_norm_eps=1e-6,
            ),
            geometry=AttentionGeometry(hidden_size=64, q_width=64, kv_width=32),
            backend="hip_gfx1100",
            runtime=self.runtime,
            rows=self.rows,
            start=self.start,
            selectors=self.selectors,
            policy=self.policy,
            cos_table_ptr=2,
            sin_table_ptr=3,
            max_positions=self.max_positions,
            stream=7,
            allow_aotriton=self.allow_aotriton,
            aotriton_min_tokens=self.threshold,
            aotriton_bridge=self.bridge,
            capture_qk=self._capture if self.dms is not None else None,
            stage_prefix="p",
            gpu_stage_recorder=stage,
        )

    def labels(self) -> list[str]:
        return [label for label, _, _ in self.log]

    def one(self, label: str) -> tuple[tuple, dict]:
        matches = [(args, kwargs) for name, args, kwargs in self.log if name == label]
        assert len(matches) == 1, (label, self.labels())
        return matches[0]

    def index(self, label: str) -> int:
        return self.labels().index(label)


def test_bulk_module_does_not_import_the_runner_or_a_session() -> None:
    tree = ast.parse(Path(bulk.__file__).read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {
        "hipengine.runtime.qwen35_gguf_runner",
        "hipengine.runtime.qwen35_paro",
        "hipengine.runtime.qwen35_paro_runner",
        "hipengine.runtime.qwen35_gguf_prefix_state",
    }
    assert forbidden.isdisjoint(imported), sorted(forbidden.intersection(imported))


@pytest.mark.parametrize(
    ("rows", "threshold", "expected_aotriton"),
    [
        (3, 4, False),
        (4, 4, True),
        (7, 3, True),
        (511, None, False),
        (512, None, True),
        (777, None, True),
    ],
)
def test_aotriton_threshold_neighbors_and_unrelated_rows(
    monkeypatch, rows, threshold, expected_aotriton
) -> None:
    harness = _Harness(
        monkeypatch,
        rows=rows,
        threshold=threshold,
        allow_aotriton=True,
        mode="v2",
    )

    assert harness.call() is expected_aotriton
    if expected_aotriton:
        assert "aotriton_attn_fwd_compact_varlen" in harness.labels()
        assert "native" not in harness.labels()
    else:
        assert "aotriton_attn_fwd_compact_varlen" not in harness.labels()
        assert "native" in harness.labels()


def test_native_path_grows_split_rows_before_launch(monkeypatch) -> None:
    harness = _Harness(monkeypatch, rows=3, threshold=4, allow_aotriton=False)

    assert harness.call() is False
    assert harness.index("ensure_split") < harness.index("native")
    assert harness.index("native") < harness.index("linear")
    args, kwargs = harness.one("native")
    assert args[5] is harness.views.prefill_spans
    assert args[6] == 3
    assert args[7] == 3
    assert args[8] == 256
    assert kwargs["split_batch_rows"] == 3
    assert kwargs["split_count"] == 4


def test_native_path_rereads_split_buffers_after_growth(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        rows=3,
        threshold=4,
        allow_aotriton=False,
        replace_split=True,
        replacement_split_count=7,
        replacement_batch_rows=5,
    )

    assert harness.call() is False
    _, kwargs = harness.one("native")
    assert kwargs["split_partial_out_ptr"] == harness.split_partial.ptr
    assert kwargs["split_partial_m_ptr"] == harness.split_m.ptr
    assert kwargs["split_partial_l_ptr"] == harness.split_l.ptr
    assert kwargs["split_batch_rows"] == 5
    assert kwargs["split_count"] == 7


def test_q8_selected_operation_receives_no_scratch_object(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        rows=3,
        threshold=4,
        allow_aotriton=False,
        q8_output=True,
    )

    assert harness.call() is False
    args, kwargs = harness.one("q8_output")
    assert len(args) == 3
    assert kwargs["rows"] == 3
    assert kwargs["in_features"] == 64
    assert kwargs["out_features"] == 64
    assert kwargs["stream"] == 7
    assert "linear" not in harness.labels()


def test_adapter_publishes_current_split_view_and_scratch_owns_growth(monkeypatch) -> None:
    class Scratch:
        def __init__(self):
            self.full_attn_split_partial = NS(ptr=1, owner="initial")
            self.full_attn_split_m = NS(ptr=2, owner="initial")
            self.full_attn_split_l = NS(ptr=3, owner="initial")
            self.full_attn_split_batch_rows = 3
            self.full_attn_split_count = 4
            self.freed = []

        def replace(self):
            self.full_attn_split_partial = NS(ptr=11, owner="grown")
            self.full_attn_split_m = NS(ptr=12, owner="grown")
            self.full_attn_split_l = NS(ptr=13, owner="grown")
            self.full_attn_split_batch_rows = 5
            self.full_attn_split_count = 7

        def free(self):
            self.freed = [
                self.full_attn_split_partial,
                self.full_attn_split_m,
                self.full_attn_split_l,
            ]
            self.full_attn_split_partial = None
            self.full_attn_split_m = None
            self.full_attn_split_l = None

    scratch = Scratch()
    runtime = object()
    calls = []

    def grow(owner, rows, *, runtime):
        assert owner is scratch
        calls.append((rows, runtime))
        scratch.replace()

    monkeypatch.setattr(runner_mod, "_ensure_full_attn_split_rows", grow)
    selectors = runner_mod._bulk_full_attention_selectors(NS(), scratch, runtime, rows=3)
    assert calls == []

    split = selectors.ensure_split_rows(3)
    assert isinstance(split, bulk.BulkFullAttentionSplitView)
    assert split.partial is scratch.full_attn_split_partial
    assert split.m is scratch.full_attn_split_m
    assert split.l is scratch.full_attn_split_l
    assert split.batch_rows == 5
    assert split.count == 7
    assert calls == [(3, runtime)]

    scratch.free()
    assert scratch.freed == [split.partial, split.m, split.l]
    assert scratch.full_attn_split_partial is None


def test_adapter_binds_scratch_for_q8_selected_operation(monkeypatch) -> None:
    scratch = NS(name="caller_scratch")
    seen = []

    def fake(weight, x_ptr, out_ptr, owner, **kwargs):
        seen.append((weight, x_ptr, out_ptr, owner, kwargs))
        return True

    monkeypatch.setattr(runner_mod, "_try_launch_dense_q8_single_dp4a", fake)
    selectors = runner_mod._bulk_full_attention_selectors(NS(), scratch, object(), rows=3)
    assert seen == []

    assert selectors.try_launch_dense_q8_single_dp4a(
        "weight",
        1,
        2,
        rows=3,
        in_features=4,
        out_features=5,
        stream=7,
        runtime="runtime",
    ) is True
    assert seen == [
        (
            "weight",
            1,
            2,
            scratch,
            {
                "rows": 3,
                "in_features": 4,
                "out_features": 5,
                "stream": 7,
                "runtime": "runtime",
            },
        )
    ]


def test_aotriton_bridge_head_major_order_and_streams(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        rows=5,
        start=3,
        threshold=1,
        mode="v3",
        head_major=True,
        copy_kernel=True,
        bridge=True,
    )

    assert harness.call() is True
    labels = harness.labels()
    assert labels.index("f32_to_bf16") < labels.index("head_major")
    assert labels.index("head_major") < labels.index("resolve_copy")
    assert labels.index("resolve_copy") < labels.index("copy")
    assert labels.index("copy") < labels.index("bridge_wait")
    assert labels.index("bridge_wait") < labels.index("aotriton_attn_fwd_v3_compact_varlen")
    assert labels.index("aotriton_attn_fwd_v3_compact_varlen") < labels.index("bridge_release")
    assert labels.index("bridge_release") < labels.index("qwen35_full_attn_gate_mul_bf16_to_bf16")
    _, resolve_kwargs = harness.one("resolve_copy")
    assert resolve_kwargs["variant"] == "head_major_spans"
    _, copy_kwargs = harness.one("copy")
    assert copy_kwargs["stream"] == 7
    _, aotriton_kwargs = harness.one("aotriton_attn_fwd_v3_compact_varlen")
    assert aotriton_kwargs["stream"] == 99
    assert aotriton_kwargs["max_seqlen_k"] == 8
    _, gate_kwargs = harness.one("qwen35_full_attn_gate_mul_bf16_to_bf16")
    assert gate_kwargs["stream"] == 7


def test_direct_hadamard_converts_gate_and_output_at_the_boundary(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        rows=1,
        threshold=4,
        direct=True,
        granularity="hadamard_group32",
        kernel="hadamard",
    )

    assert harness.call() is False
    labels = harness.labels()
    assert labels.index("bf16_to_fp16") < labels.index(
        "qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans"
    )
    assert labels.index(
        "qwen35_paged_attn_prefill_int8_hadamard_group32_gqa_gate_fp16_spans"
    ) < labels.index("fp16_to_bf16")
    assert labels.index("fp16_to_bf16") < labels.index("linear")
    _, gate_cast = harness.one("bf16_to_fp16")
    assert gate_cast["stream"] == 7


def test_direct_int8_metadata_mismatch_is_named(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        rows=3,
        threshold=4,
        direct=True,
        granularity="per_token_head",
        prefill_granularity="other",
    )

    with pytest.raises(
        RuntimeError,
        match="GGUF direct INT8 prefill requires matching append/attention metadata",
    ):
        harness.call()
    assert "int8_prompt_write" not in harness.labels()


def test_retained_write_order_and_value_bf16_choice(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        rows=3,
        threshold=4,
        allow_aotriton=False,
        retained=True,
        int8_value_bf16=True,
    )

    harness.call()
    labels = harness.labels()
    assert labels.index("qwen35_write_paged_kv_mixed_value_bf16_prompt_spans") < labels.index(
        "qwen35_write_paged_kv_int8_key_bf16_value_prompt_spans"
    )
    assert "int8_prompt_write" not in labels

    other = _Harness(
        monkeypatch,
        rows=3,
        threshold=4,
        allow_aotriton=False,
        retained=True,
        int8_value_bf16=False,
    )
    other.call()
    labels = other.labels()
    assert labels.index("qwen35_write_paged_kv_mixed_value_bf16_prompt_spans") < labels.index(
        "int8_prompt_write"
    )
    assert other.one("int8_prompt_write")[0] == ("per_token_head",)


def test_dms_capture_runs_after_head_norm_before_kv_write(monkeypatch) -> None:
    harness = _Harness(monkeypatch, rows=3, threshold=4, allow_aotriton=False, dms=True)

    harness.call()
    labels = harness.labels()
    assert labels.index(
        "gguf_qwen35_head_rmsnorm_partial_rotary_positions_f32_weight"
    ) < labels.index("capture")
    assert labels.index("capture") < labels.index(
        "qwen35_write_paged_kv_mixed_value_bf16_prompt_spans"
    )
    assert harness.stage_marks == [
        "p_attn_norm",
        "p_qkv_head_norm_rope",
        "p_kv_write",
        "p_core_gate",
        "p_output",
    ]


def test_missing_cache_and_paged_context_errors_are_unchanged(monkeypatch) -> None:
    harness = _Harness(monkeypatch, rows=3, threshold=4, missing_caches=True)
    with pytest.raises(
        RuntimeError,
        match="GGUF full-attention prefill requires cache-backed key/value buffers",
    ):
        harness.call()

    harness = _Harness(monkeypatch, rows=3, threshold=-1)
    with pytest.raises(ValueError, match="aotriton_min_tokens must be non-negative"):
        harness.call()
    assert "qwen35_write_paged_kv_mixed_value_bf16_prompt_spans" in harness.labels()

    harness = _Harness(monkeypatch, rows=3, threshold=4, max_positions=0)
    with pytest.raises(
        ValueError,
        match="paged_max_context_len must be positive and within the prefill scratch capacity",
    ):
        harness.call()
