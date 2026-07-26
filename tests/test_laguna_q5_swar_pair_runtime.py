from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from hipengine.kernels.backends import backend_package_capability
from hipengine.kernels.registry import KernelKey, is_registered
from hipengine.loading.laguna_gguf import laguna_gguf_config_from_metadata
from hipengine.runtime import laguna_gguf_runner as runner_module
from scripts import laguna_target_ar_bench as benchmark
from tests._laguna_synthetic import make_laguna_info

_OUTPUT_CONTROL = "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out"
_OUTPUT_CANDIDATE = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_QUERY_GATE_CONTROL = "wave32x2_fixed_meta_gemv_decode_bf16_f32_out"
_SHARED_CONTROL = "wave32x2_fixed_meta_gemv_decode_bf16_bf16_out"
_SHARED_CANDIDATE = "wave32x2_swar_pair_fixed_meta_gemv_decode_bf16_bf16_out"
_MIXED_CONTROL = "mixed_local32_fixed_meta_pack8_gemv_decode_bf16_f32_out"
_MIXED_CANDIDATE = (
    "mixed_local32_q5_swar_pair_fixed_meta_gemv_decode_bf16_f32_out"
)
_MIXED_QUANT = "gguf_q5_k+gguf_q6_k+gguf_q6_k+gguf_q5_k"
_OUTPUT_KEY = KernelKey("hip_gfx1100", "linear", "gguf_q5_k", _OUTPUT_CANDIDATE)
_SHARED_KEY = KernelKey(
    "hip_gfx1100",
    "linear_pair",
    "gguf_q5_k",
    _SHARED_CANDIDATE,
)
_MIXED_KEY = KernelKey(
    "hip_gfx1100",
    "attention_projection_quad",
    _MIXED_QUANT,
    _MIXED_CANDIDATE,
)


def _mock_session_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    class Resource:
        split_gate_fusion = False
        swa_split_wave_local = False

        def free(self, **kwargs) -> None:
            del kwargs

    monkeypatch.setattr(
        runner_module.LagunaGGUFResidentSession,
        "_validate_resident_weights",
        lambda self: None,
    )
    monkeypatch.setattr(
        runner_module,
        "load_laguna_eager_libraries",
        lambda **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "materialize_laguna_rope_tables",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_kv_cache",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module.LagunaEagerScratch,
        "allocate",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module.LagunaRowsScratch,
        "allocate",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "resolve_laguna_moe_plan",
        lambda *args, **kwargs: Resource(),
    )
    monkeypatch.setattr(
        runner_module,
        "allocate_laguna_moe_scratch",
        lambda *args, **kwargs: Resource(),
    )


def _session(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_q5_swar_output: bool | None,
    **kwargs,
):
    _mock_session_dependencies(monkeypatch)
    config = laguna_gguf_config_from_metadata(make_laguna_info())
    weights = SimpleNamespace(config=config, backend="hip_gfx1100")
    return runner_module.LagunaGGUFResidentSession(
        resident_weights=weights,
        backend="hip_gfx1100",
        runtime=SimpleNamespace(),
        use_q5_swar_output=use_q5_swar_output,
        **kwargs,
    )


def test_q5_swar_output_capability_is_default_off_and_gfx1100_only() -> None:
    assert backend_package_capability(
        "hip_gfx1100",
        "LAGUNA_Q5_SWAR_OUTPUT",
        None,
    ) is False
    assert backend_package_capability(
        "hip_gfx1151",
        "LAGUNA_Q5_SWAR_OUTPUT",
        None,
    ) is None

    resolver = getattr(runner_module, "resolve_laguna_q5_swar_output", None)
    assert callable(resolver)
    assert not resolver("hip_gfx1100")
    assert resolver("hip_gfx1100", True)
    assert not resolver("hip_gfx1100", False)
    assert not resolver("hip_gfx1151", True)
    assert "use_q5_swar_output" in inspect.signature(
        runner_module.LagunaGGUFResidentSession
    ).parameters

    assert is_registered(_OUTPUT_KEY)
    assert not is_registered(
        KernelKey("hip_gfx1151", "linear", "gguf_q5_k", _OUTPUT_CANDIDATE)
    )


def test_q5_swar_output_requires_only_the_exact_direct_bf16_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolver = getattr(runner_module, "resolve_laguna_q5_swar_output", None)
    assert callable(resolver)
    assert is_registered(_SHARED_KEY)
    assert is_registered(_MIXED_KEY)

    checked: list[KernelKey] = []
    original_is_registered = runner_module.is_registered

    def record(key: KernelKey) -> bool:
        checked.append(key)
        return original_is_registered(key)

    monkeypatch.setattr(runner_module, "is_registered", record)
    assert resolver("hip_gfx1100", True)
    assert checked == [_OUTPUT_KEY]

    monkeypatch.setattr(
        runner_module,
        "is_registered",
        lambda key: False if key == _OUTPUT_KEY else original_is_registered(key),
    )
    assert not resolver("hip_gfx1100", True)


def test_q5_swar_output_session_changes_only_attention_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control = _session(monkeypatch, use_q5_swar_output=None)
    try:
        assert not control.use_q5_swar_output
        assert control._q5_output_variant == _OUTPUT_CONTROL
        assert control._q5_query_gate_variant == _QUERY_GATE_CONTROL
        assert control._q5_shared_pair_variant == _SHARED_CONTROL
        assert control.use_mixed_local32_fixed_meta_attention
    finally:
        control.close()

    candidate = _session(monkeypatch, use_q5_swar_output=True)
    try:
        assert candidate.use_q5_swar_output
        assert candidate._q5_output_variant == _OUTPUT_CANDIDATE
        assert candidate._q5_query_gate_variant == _QUERY_GATE_CONTROL
        assert candidate._q5_shared_pair_variant == _SHARED_CONTROL
        assert candidate.use_mixed_local32_fixed_meta_attention
    finally:
        candidate.close()

    rollback = _session(
        monkeypatch,
        use_q5_swar_output=True,
        use_q5_fixed_meta_output=False,
    )
    try:
        assert not rollback.use_q5_swar_output
        assert rollback._q5_output_variant != _OUTPUT_CANDIDATE
        assert rollback._q5_query_gate_variant == _QUERY_GATE_CONTROL
        assert rollback._q5_shared_pair_variant == _SHARED_CONTROL
    finally:
        rollback.close()

    original_is_registered = runner_module.is_registered
    monkeypatch.setattr(
        runner_module,
        "is_registered",
        lambda key: False if key == _OUTPUT_KEY else original_is_registered(key),
    )
    missing = _session(monkeypatch, use_q5_swar_output=True)
    try:
        assert not missing.use_q5_swar_output
        assert missing._q5_output_variant == _OUTPUT_CONTROL
        assert missing._q5_query_gate_variant == _QUERY_GATE_CONTROL
        assert missing._q5_shared_pair_variant == _SHARED_CONTROL
    finally:
        missing.close()

    session_source = inspect.getsource(runner_module.LagunaGGUFResidentSession.__init__)
    assert "_Q5_SWAR_PAIR_OUTPUT_VARIANT" in session_source
    assert "_Q5_SWAR_PAIR_SHARED_VARIANT" not in session_source
    assert "_MIXED_ATTENTION_Q5_SWAR_PAIR_VARIANT" not in session_source
    assert "q5_swar_pair_variant" not in inspect.signature(
        runner_module.launch_laguna_attention_projections
    ).parameters


def test_q5_swar_output_cli_opt_in_is_explicit_and_output_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.sys, "argv", ["laguna_target_ar_bench.py"])
    assert not benchmark._parse_args().enable_q5_swar_output
    monkeypatch.setattr(
        benchmark.sys,
        "argv",
        ["laguna_target_ar_bench.py", "--enable-q5-swar-output"],
    )
    args = benchmark._parse_args()
    assert args.enable_q5_swar_output

    captured: dict[str, object] = {}

    def session_factory(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(benchmark, "LagunaGGUFResidentSession", session_factory)
    owner = SimpleNamespace(weights=object(), runtime=object())
    benchmark._session(owner, args)
    assert captured["use_q5_swar_output"] is True
    assert "use_q5_swar_pair" not in captured
    assert captured["use_q5_shared_fixed_meta"] is None
    assert captured["use_mixed_local32_fixed_meta_attention"] is None
