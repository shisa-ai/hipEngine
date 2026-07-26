"""Default-off runtime contract for gated single-page global attention."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.registry import KernelKey
from hipengine.loading.laguna_gguf import FULL_ATTENTION, SLIDING_ATTENTION
from hipengine.runtime import laguna_kv
from hipengine.runtime.laguna_gguf_runner import LagunaGGUFResidentSession

_CANDIDATE_LAYER = "laguna_attention_decode+attention_gate"
_CANDIDATE_VARIANT = "global_single_page_softplus_bf16_spans"
_CANDIDATE_KEY = KernelKey(
    "hip_gfx1100",
    _CANDIDATE_LAYER,
    "bf16",
    _CANDIDATE_VARIANT,
)


def test_gated_single_page_capability_is_gfx1100_default_off() -> None:
    import hipengine.kernels.hip_gfx1100 as gfx1100
    import hipengine.kernels.hip_gfx1151 as gfx1151

    assert gfx1100.LAGUNA_GLOBAL_SINGLE_PAGE_GATED_ATTENTION is False
    assert not hasattr(gfx1151, "LAGUNA_GLOBAL_SINGLE_PAGE_GATED_ATTENTION")


def test_gated_single_page_resolver_is_explicit_and_exact_key_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolve_candidate = laguna_kv.resolve_laguna_global_single_page_gated_attention
    assert not resolve_candidate("hip_gfx1100")
    assert resolve_candidate("hip_gfx1100", True)
    assert not resolve_candidate("hip_gfx1100", False)
    assert not resolve_candidate("hip_gfx1151", True)

    monkeypatch.setattr(
        laguna_kv,
        "is_registered",
        lambda key: key != _CANDIDATE_KEY,
    )
    assert not resolve_candidate("hip_gfx1100", True)


def _state(layer_id: int, attention_type: str, q_heads: int, capacity: int):
    return SimpleNamespace(
        layer_id=layer_id,
        attention_type=attention_type,
        q_heads=q_heads,
        capacity=capacity,
        key_cache=SimpleNamespace(ptr=0x1000 + layer_id * 0x100),
        value_cache=SimpleNamespace(ptr=0x2000 + layer_id * 0x100),
        spans=object(),
        attention_variant=(
            "global_context_spans"
            if attention_type == FULL_ATTENTION
            else "swa_context_token4_exact_spans"
        ),
        attention_prefill_variant=(
            "global_context_rows_spans"
            if attention_type == FULL_ATTENTION
            else "swa_context_rows_spans"
        ),
    )


def _cache(*, enabled: bool):
    cache = laguna_kv.LagunaKVCache(
        layers=(
            _state(0, FULL_ATTENTION, 48, 4096),
            _state(1, SLIDING_ATTENTION, 72, 512),
        ),
        buffers=(),
        context_length=4096,
        sliding_window=512,
        backend="hip_gfx1100",
        row_position=SimpleNamespace(ptr=0),
        split_score_scratch=SimpleNamespace(ptr=0x3000),
        split_physical_scratch=SimpleNamespace(ptr=0x4000),
        global_split_min_live=127,
        swa_split_min_live=None,
        swa_split_tile16_min_live=None,
        split_gate_fusion=True,
        swa_split_wave_local=True,
        use_global_single_page_gated_attention=enabled,
        runtime=SimpleNamespace(),
    )
    calls: list[dict[str, object]] = []

    def resolve(layer: str, variant: str):
        row: dict[str, object] = {"layer": layer, "variant": variant}
        calls.append(row)

        def launch(*args, **kwargs):
            row["args"] = args
            row["kwargs"] = kwargs

        return launch

    cache._resolve = resolve
    return cache, calls


def test_gated_single_page_runtime_is_live126_full_only_with_split_precedence() -> None:
    candidate, calls = _cache(enabled=True)
    candidate.position = 125
    assert candidate.attend(
        0,
        0x5000,
        0x6000,
        gate_ptr=0x7000,
        gated_out_ptr=0x8000,
    )
    assert calls[-1]["layer"] == _CANDIDATE_LAYER
    assert calls[-1]["variant"] == _CANDIDATE_VARIANT
    assert calls[-1]["args"][:6] == (
        0x5000,
        0x1000,
        0x2000,
        0x6000,
        0x7000,
        0x8000,
    )

    candidate.position = 126
    assert candidate.attend(
        0,
        0x5000,
        0x6000,
        gate_ptr=0x7000,
        gated_out_ptr=0x8000,
    )
    assert calls[-1]["layer"] == "laguna_attention_decode"
    assert calls[-1]["variant"] == "global_context_split_exact_gated_spans"

    candidate.position = 125
    assert not candidate.attend(0, 0x5000, 0x6000)
    assert calls[-1]["variant"] == "global_context_spans"

    assert not candidate.attend(
        1,
        0x5000,
        0x6000,
        gate_ptr=0x7000,
        gated_out_ptr=0x8000,
    )
    assert calls[-1]["variant"] == "swa_context_token4_exact_spans"

    candidate._pending_positions = (126, 127)
    candidate.attend_prefill(0, 0x5000, 0x5100, 0x5200, 0x6000, 2)
    assert calls[-1]["variant"] == "global_context_rows_spans"

    control, control_calls = _cache(enabled=False)
    control.position = 125
    assert not control.attend(
        0,
        0x5000,
        0x6000,
        gate_ptr=0x7000,
        gated_out_ptr=0x8000,
    )
    assert control_calls[-1]["variant"] == "global_context_spans"


def test_gated_single_page_session_allocator_and_benchmark_opt_in_are_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import laguna_target_ar_bench as benchmark

    option = "use_global_single_page_gated_attention"
    assert option in inspect.signature(LagunaGGUFResidentSession).parameters
    assert option in inspect.signature(laguna_kv.allocate_laguna_kv_cache).parameters
    assert option in inspect.signature(laguna_kv.LagunaKVCache).parameters

    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert benchmark._parse_args().enable_global_single_page_gated_attention is False
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        [
            "laguna_target_ar_bench.py",
            "--enable-global-single-page-gated-attention",
        ],
    )
    assert benchmark._parse_args().enable_global_single_page_gated_attention is True
