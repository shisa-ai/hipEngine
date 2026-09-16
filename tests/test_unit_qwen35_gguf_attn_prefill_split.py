"""Contract tests for the attention/FFN split in ``qwen35_gguf_runner``.

The bulk TP2 prefill path needs to run a layer's attention/GDN subgraph on its
own and then a *sharded* MLP, without re-implementing the attention ops.  To make
that safe the two monolithic bulk-prefill layer methods were split into:

* ``_run_linear_attention_prefill_attn_rows``  -- GDN/attention subgraph only,
  returns the optional f32 attention output pointer.
* ``_run_full_attention_prefill_attn_rows``    -- full-attention subgraph only,
  returns ``used_aotriton``.
* the original public methods (``_run_linear_attention_prefill_layer_rows`` and
  ``_run_full_attention_prefill_layer_aotriton``) which now call the matching
  helper and then the *unchanged* ``_run_post_attention_ffn_rows``.

These tests pin the split contract:

* the helpers never launch the FFN/down/residual subgraph themselves;
* the original wrappers keep their signatures and launch the attention helper
  exactly once followed by the FFN exactly once (no double MLP);
* every pointer / row count / state argument / stream is forwarded unchanged.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.runtime.qwen35_gguf_runner import Qwen35GGUFFullStackRunner

Runner = Qwen35GGUFFullStackRunner

WRAPPER_LINEAR = "_run_linear_attention_prefill_layer_rows"
HELPER_LINEAR = "_run_linear_attention_prefill_attn_rows"
WRAPPER_FULL = "_run_full_attention_prefill_layer_aotriton"
HELPER_FULL = "_run_full_attention_prefill_attn_rows"

FFN = "_run_post_attention_ffn_rows"


def _src(name: str) -> str:
    return inspect.getsource(getattr(Runner, name))


def _new_runner():
    # Bypass ``__init__``: the split methods only touch ``self`` through the
    # attributes the test patches and the forwarded scratch objects.
    return object.__new__(Runner)


class _Recorder:
    def __init__(self, events, tag, ret=None):
        self.events = events
        self.tag = tag
        self.ret = ret
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.events.append(self.tag)
        self.calls.append((args, kwargs))
        return self.ret


# ---------------------------------------------------------------------------
# Structural guards: single definition, helpers do not own the FFN.
# ---------------------------------------------------------------------------


def test_split_methods_are_defined_exactly_once():
    source = inspect.getsource(inspect.getmodule(Runner))
    for name in (WRAPPER_LINEAR, HELPER_LINEAR, WRAPPER_FULL, HELPER_FULL):
        assert source.count(f"def {name}(") == 1, name


@pytest.mark.parametrize("helper", [HELPER_LINEAR, HELPER_FULL])
def test_attention_helper_never_runs_ffn(helper):
    src = _src(helper)
    assert f"self.{FFN}(" not in src
    # The FFN owns gate/up/down/residual; the attention helper must not reach
    # into that subgraph through the shared post-attention entrypoint either.
    assert "_run_post_attention_norm_residual" not in src


@pytest.mark.parametrize(
    ("wrapper", "helper"),
    [(WRAPPER_LINEAR, HELPER_LINEAR), (WRAPPER_FULL, HELPER_FULL)],
)
def test_wrapper_runs_helper_then_ffn_once(wrapper, helper):
    src = _src(wrapper)
    assert src.count(f"self.{helper}(") == 1
    assert src.count(f"self.{FFN}(") == 1
    # The helper call must appear before the FFN call so the launch order is the
    # original attention -> FFN order.
    assert src.index(f"self.{helper}(") < src.index(f"self.{FFN}(")


def test_linear_helper_returns_capture_attn_out_or_segment_none():
    src = _src(HELPER_LINEAR)
    assert "return attn_out_f32_ptr" in src
    assert "return None" in src


# ---------------------------------------------------------------------------
# Mocked launch-sequence contract.
# ---------------------------------------------------------------------------


def test_linear_wrapper_forwards_everything_and_launches_once():
    runner = _new_runner()
    events: list[str] = []
    attn = _Recorder(events, "attn", ret=0xF32A77)
    ffn = _Recorder(events, "ffn")
    runner._run_linear_attention_prefill_attn_rows = attn
    runner._run_post_attention_ffn_rows = ffn

    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=0x0C0277))
    decode_scratch = SimpleNamespace(name="decode")
    kv_live = object()
    stage_timings = {"a": 1.0}

    Runner._run_linear_attention_prefill_layer_rows(
        runner,
        7,
        0x1111,
        0x2222,
        scratch,
        rows=5,
        decode_scratch=decode_scratch,
        stream=99,
        expert_sidecar="sidecar",
        linear_state_rows=(kv_live, "conv"),
        commit_final_linear_state=False,
        hidden_f32_ptr=0xF32,
        out_f32_ptr=0xF0,
        stage_timings=stage_timings,
        sync_stage_timings=True,
        stage_prefix="pref",
        gpu_stage_recorder=None,
    )

    assert events == ["attn", "ffn"]
    assert len(attn.calls) == 1 and len(ffn.calls) == 1

    (a_args, a_kw) = attn.calls[0]
    assert a_args == (7, 0x1111, scratch)
    assert a_kw["rows"] == 5
    assert a_kw["decode_scratch"] is decode_scratch
    assert a_kw["stream"] == 99
    assert a_kw["linear_state_rows"] == (kv_live, "conv")
    assert a_kw["commit_final_linear_state"] is False
    assert a_kw["hidden_f32_ptr"] == 0xF32
    assert a_kw["out_f32_ptr"] == 0xF0
    assert a_kw["stage_timings"] is stage_timings
    assert a_kw["sync_stage_timings"] is True
    assert a_kw["stage_prefix"] == "pref"

    (f_args, f_kw) = ffn.calls[0]
    assert f_args == (7, 0x1111, scratch.attn_out.ptr, 0x2222, scratch)
    assert f_kw["rows"] == 5
    assert f_kw["stream"] == 99
    assert f_kw["expert_sidecar"] == "sidecar"
    assert f_kw["attn_out_f32_ptr"] == 0xF32A77
    assert f_kw["stage_timings"] is stage_timings
    assert f_kw["sync_stage_timings"] is True
    assert f_kw["stage_prefix"] == "pref_ffn"


def test_linear_wrapper_returns_none():
    runner = _new_runner()
    runner._run_linear_attention_prefill_attn_rows = _Recorder([], "attn", ret=None)
    runner._run_post_attention_ffn_rows = _Recorder([], "ffn")
    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=1))
    assert (
        Runner._run_linear_attention_prefill_layer_rows(
            runner, 0, 1, 2, scratch, rows=1, decode_scratch=None
        )
        is None
    )


def test_full_wrapper_forwards_everything_and_returns_used_aotriton():
    runner = _new_runner()
    events: list[str] = []
    attn = _Recorder(events, "attn", ret=True)
    ffn = _Recorder(events, "ffn")
    runner._run_full_attention_prefill_attn_rows = attn
    runner._run_post_attention_ffn_rows = ffn

    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=0x0C0277), rows=6)

    used = Runner._run_full_attention_prefill_layer_aotriton(
        runner,
        3,
        0x3333,
        0x4444,
        scratch,
        cos_table_ptr=0xC0,
        sin_table_ptr=0xC1,
        max_positions=4096,
        attn_aotriton_min_tokens=128,
        stream=42,
        aotriton_bridge="bridge",
        expert_sidecar="sidecar",
        dms_capture="dms",
        allow_aotriton=False,
        aotriton_min_tokens=64,
        paged_max_context_len=2048,
        stage_prefix="full",
        gpu_stage_recorder=None,
    )

    assert used is True
    assert events == ["attn", "ffn"]
    (a_args, a_kw) = attn.calls[0]
    assert a_args == (3, 0x3333, scratch)
    assert a_kw["cos_table_ptr"] == 0xC0
    assert a_kw["sin_table_ptr"] == 0xC1
    assert a_kw["max_positions"] == 4096
    assert a_kw["stream"] == 42
    assert a_kw["aotriton_bridge"] == "bridge"
    assert a_kw["dms_capture"] == "dms"
    assert a_kw["allow_aotriton"] is False
    assert a_kw["aotriton_min_tokens"] == 64
    assert a_kw["paged_max_context_len"] == 2048
    assert a_kw["stage_prefix"] == "full"

    (f_args, f_kw) = ffn.calls[0]
    assert f_args == (3, 0x3333, scratch.attn_out.ptr, 0x4444, scratch)
    assert f_kw["rows"] == 6
    assert f_kw["stream"] == 42
    assert f_kw["expert_sidecar"] == "sidecar"
    assert f_kw["stage_prefix"] == "full_ffn"
    # Full attention never produced an f32 attention output.
    assert f_kw.get("attn_out_f32_ptr") is None


def test_full_wrapper_propagates_false_from_helper():
    runner = _new_runner()
    runner._run_full_attention_prefill_attn_rows = _Recorder([], "attn", ret=False)
    runner._run_post_attention_ffn_rows = _Recorder([], "ffn")
    scratch = SimpleNamespace(attn_out=SimpleNamespace(ptr=1), rows=1)
    assert (
        Runner._run_full_attention_prefill_layer_aotriton(
            runner,
            0,
            1,
            2,
            scratch,
            cos_table_ptr=0,
            sin_table_ptr=0,
            max_positions=1,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Public signature preservation (callers and monkeypatchers depend on it).
# ---------------------------------------------------------------------------


def test_wrapper_signatures_preserved():
    lin = inspect.signature(Runner._run_linear_attention_prefill_layer_rows)
    assert list(lin.parameters) == [
        "self",
        "layer_id",
        "hidden_ptr",
        "out_ptr",
        "scratch",
        "rows",
        "decode_scratch",
        "stream",
        "expert_sidecar",
        "linear_state_rows",
        "commit_final_linear_state",
        "hidden_f32_ptr",
        "out_f32_ptr",
        "stage_timings",
        "sync_stage_timings",
        "stage_prefix",
        "gpu_stage_recorder",
    ]
    full = inspect.signature(Runner._run_full_attention_prefill_layer_aotriton)
    assert list(full.parameters) == [
        "self",
        "layer_id",
        "hidden_ptr",
        "out_ptr",
        "scratch",
        "cos_table_ptr",
        "sin_table_ptr",
        "max_positions",
        "attn_aotriton_min_tokens",
        "stream",
        "aotriton_bridge",
        "expert_sidecar",
        "dms_capture",
        "allow_aotriton",
        "aotriton_min_tokens",
        "paged_max_context_len",
        "stage_prefix",
        "gpu_stage_recorder",
    ]


# ---------------------------------------------------------------------------
# Post-attention norm+residual split (sharded-TP2 residual contract).
# ---------------------------------------------------------------------------

NORM_HELPER = "_run_post_attention_norm_residual_rows"


def test_norm_residual_helper_defined_once_and_owned_by_ffn():
    source = inspect.getsource(inspect.getmodule(Runner))
    assert source.count(f"def {NORM_HELPER}(") == 1
    ffn_src = _src(FFN)
    assert ffn_src.count(f"self.{NORM_HELPER}(") == 1
    # The stage mark stays in the FFN wrapper so the diagnostic t_stage
    # baseline flows into the next stage exactly as it did before the split.
    helper_src = _src(NORM_HELPER)
    assert "_mark_sync_stage" not in helper_src
    assert ".mark(" not in helper_src
    assert "_post_norm_residual" in ffn_src


def test_norm_residual_helper_launches_single_add_rmsnorm(monkeypatch):
    import hipengine.runtime.qwen35_gguf_runner as runner_mod

    runner = _new_runner()
    runner.runtime = object()  # truthy; the fake kernel ignores it
    calls = []

    def fake_kernel(owner, *, layer, rows, hidden_size):
        def launch(
            hidden_ptr,
            attn_out_ptr,
            weight_ptr,
            post_norm_ptr,
            residual_ptr,
            *,
            rows,
            hidden_size,
            eps,
            stream,
            runtime,
        ):
            calls.append(
                dict(
                    hidden_ptr=hidden_ptr,
                    attn_out_ptr=attn_out_ptr,
                    weight_ptr=weight_ptr,
                    post_norm_ptr=post_norm_ptr,
                    residual_ptr=residual_ptr,
                    rows=rows,
                    hidden_size=hidden_size,
                    eps=eps,
                    stream=stream,
                )
            )

        return launch

    monkeypatch.setattr(runner_mod, "_gguf_norm_residual_decode_kernel", fake_kernel)
    layer = SimpleNamespace(
        weight=lambda name: SimpleNamespace(
            allocation=lambda: SimpleNamespace(tensor=SimpleNamespace(ptr=0x0A11))
        )
    )
    runner.weights = SimpleNamespace(
        layer=lambda lid: layer,
        config=SimpleNamespace(rms_norm_eps=1e-6, hidden_size=16),
    )
    scratch = SimpleNamespace(
        post_norm=SimpleNamespace(ptr=0x0B01), residual=SimpleNamespace(ptr=0x0B02)
    )

    out = Runner._run_post_attention_norm_residual_rows(
        runner, 2, 0x0C01, 0x0C02, scratch, rows=3, stream=7
    )

    assert out is None
    assert len(calls) == 1
    call = calls[0]
    assert call["hidden_ptr"] == 0x0C01
    assert call["attn_out_ptr"] == 0x0C02
    assert call["weight_ptr"] == 0x0A11
    assert call["post_norm_ptr"] == 0x0B01
    assert call["residual_ptr"] == 0x0B02
    assert call["rows"] == 3 and call["hidden_size"] == 16 and call["stream"] == 7


def test_ffn_wrapper_uses_norm_helper_then_moe_once():
    runner = _new_runner()
    runner.runtime = object()
    events = []

    def norm(*args, **kwargs):
        events.append(("norm", args, kwargs))
        return 0x0B03

    def moe(*args, **kwargs):
        events.append(("moe", args, kwargs))

    runner._run_post_attention_norm_residual_rows = norm
    runner._run_post_attention_moe_rows = moe
    runner.weights = SimpleNamespace(
        layer=lambda lid: None,
        config=SimpleNamespace(is_moe=True, hidden_size=16),
    )
    scratch = SimpleNamespace(
        post_norm=SimpleNamespace(ptr=0x0B01), residual=SimpleNamespace(ptr=0x0B02)
    )

    Runner._run_post_attention_ffn_rows(
        runner, 3, 0x0C01, 0x0C02, 0x0C03, scratch, rows=4, stream=5, stage_prefix="p"
    )

    assert [event[0] for event in events] == ["norm", "moe"]
    (_, n_args, n_kw) = events[0]
    assert n_args == (3, 0x0C01, 0x0C02, scratch)
    assert n_kw["rows"] == 4 and n_kw["stream"] == 5
    # The FFN branch consumes the norm helper's F32 diagnostic pointer.
    (_, _, m_kw) = events[1]
    assert m_kw["post_norm_f32_ptr"] == 0x0B03
    assert m_kw["rows"] == 4
